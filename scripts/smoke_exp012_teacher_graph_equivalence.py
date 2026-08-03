#!/usr/bin/env python
"""DEC-032: does complete-residue expansion change the ligand latent at all?

EXP-010's complete-residue expansion existed for a fragment-energy
decomposition that required whole residues to subtract a valid fragment
energy. EXP-012's ligand latent never computes a fragment energy, so a
residue only partially inside the two-hop closure may not need to be
expanded to completeness. This script does not assume an answer either
way -- it runs the *same* frame through both graphs and reports the
difference, as a comparison, not a gate.

Graph A: ``local_residual.teacher_graph.build_teacher_graph_for_frame`` --
the exact two-hop closure only, no fixed manifest, no residue completion.

Graph B: ``local_residual.mace_graph.build_mace_graph`` against an existing
sealed environment manifest/atom mapping (the frame0 ``derived_5a`` C1
artifacts) -- the complete-residue-expanded graph already used for DEC-028/029.

Both graphs order nodes by ascending topology index (``manifest_canonical``
for graph B; the closure itself, sorted, for graph A), and both share the
exact same frame-invariant ligand identity -- so ``ligand_latent[i]`` in A and
B refer to the same physical atom for every ``i``, letting a direct,
per-element comparison stand in for "does dropping residue completion change
the cached teacher's output."
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

from local_residual.atom_mapping import load_atom_mapping  # noqa: E402
from local_residual.environment import canonical_json_bytes, load_environment_manifest  # noqa: E402
from local_residual.mace_graph import MaceGraphConfig, build_mace_graph  # noqa: E402
from local_residual.mace_latent import MaceLatentBasisAdapter, load_c0_report  # noqa: E402
from local_residual.teacher_graph import build_teacher_graph_for_frame  # noqa: E402


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


def _tensor_stats(tensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "finite": bool(tensor.isfinite().all().item()),
        "norm": float(tensor.norm().item()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--c0-report", required=True)
    parser.add_argument(
        "--environment-manifest", required=True,
        help="sealed frame0 derived_5a manifest -- source of graph B and of the shared ligand identity",
    )
    parser.add_argument("--atom-mapping", required=True, help="sealed frame0 derived_5a atom mapping")
    parser.add_argument("--topology", required=True, help="mdtraj-compatible topology")
    parser.add_argument("--trajectory", required=True, help="mdtraj-compatible frame source")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--edge-cutoff-angstrom", type=float, required=True)
    parser.add_argument("--interaction-layers", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--product-layer-index", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite a frozen report: {args.output}")

    import mdtraj
    import torch

    started = time.perf_counter()
    manifest = load_environment_manifest(args.environment_manifest)
    mapping = load_atom_mapping(args.atom_mapping, environment_manifest=manifest)
    c0 = load_c0_report(args.c0_report)
    ligand_indices = manifest["payload"]["ligand_indices"]

    torch_dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    trajectory = mdtraj.load_frame(args.trajectory, index=args.frame_index, top=args.topology)
    if trajectory.n_frames != 1 or trajectory.unitcell_vectors is None:
        raise RuntimeError("frame source must provide exactly one frame with triclinic cell vectors")
    if trajectory.n_atoms != manifest["payload"]["atom_count"]:
        raise RuntimeError("frame atom count differs from the environment manifest")
    atomic_numbers_by_topology_index = [
        int(atom.element.atomic_number) for atom in trajectory.topology.atoms
    ]

    graph_config = MaceGraphConfig(
        edge_cutoff_angstrom=args.edge_cutoff_angstrom,
        interaction_layers=args.interaction_layers,
        geometric_upper_bound_angstrom=args.edge_cutoff_angstrom * args.interaction_layers,
    )
    adapter = MaceLatentBasisAdapter(
        c0_report=c0, model_path=args.model, device=args.device, dtype=args.dtype,
        product_layer_index=args.product_layer_index,
    )

    def _positions_and_cell():
        positions = torch.tensor(
            trajectory.xyz[0] * 10.0, dtype=torch_dtype, device=device, requires_grad=True
        )
        cell = torch.tensor(trajectory.unitcell_vectors[0] * 10.0, dtype=torch_dtype, device=device)
        return positions, cell

    positions_a, cell_a = _positions_and_cell()
    graph_a = build_teacher_graph_for_frame(
        positions_a, cell_a,
        ligand_indices=ligand_indices,
        atomic_numbers_by_topology_index=atomic_numbers_by_topology_index,
        model_atomic_numbers=c0["expected"]["atomic_numbers"],
        config=graph_config,
    )
    result_a = adapter(graph_a)
    latent_a = result_a["ligand_latent"]
    weights = torch.linspace(1.0, 2.0, latent_a.shape[1], dtype=latent_a.dtype, device=latent_a.device)
    scalar_probe_a = (latent_a * weights).sum()
    grad_a = torch.autograd.grad(scalar_probe_a, positions_a, retain_graph=False)[0]
    ligand_grad_a = grad_a[torch.tensor(ligand_indices, dtype=torch.long, device=device)]

    positions_b, cell_b = _positions_and_cell()
    graph_b = build_mace_graph(
        positions_b, cell_b,
        environment_manifest=manifest, atom_mapping=mapping,
        model_atomic_numbers=c0["expected"]["atomic_numbers"], config=graph_config,
    )
    result_b = adapter(graph_b)
    latent_b = result_b["ligand_latent"]
    scalar_probe_b = (latent_b * weights).sum()
    grad_b = torch.autograd.grad(scalar_probe_b, positions_b, retain_graph=False)[0]
    ligand_grad_b = grad_b[torch.tensor(ligand_indices, dtype=torch.long, device=device)]

    if latent_a.shape != latent_b.shape:
        raise RuntimeError(
            f"ligand latent shape differs between graphs: A={list(latent_a.shape)} "
            f"B={list(latent_b.shape)} -- ligand identity/order assumption is violated"
        )
    latent_diff = (latent_a.detach() - latent_b.detach())
    grad_diff = (ligand_grad_a.detach() - ligand_grad_b.detach())

    inputs = {
        name: {"path": str(Path(path).resolve()), "sha256": _sha(path)}
        for name, path in (
            ("model", args.model), ("c0_report", args.c0_report),
            ("environment_manifest", args.environment_manifest), ("atom_mapping", args.atom_mapping),
            ("topology", args.topology), ("trajectory", args.trajectory),
        )
    }
    body = {
        "schema_version": "exp012-teacher-graph-equivalence-v1",
        "status": "COMPARISON_ONLY_NOT_A_GATE",
        "frame_index": args.frame_index,
        "graph_a": {
            "policy": "per_frame_exact_two_hop_closure_no_residue_expansion",
            "node_count": graph_a["diagnostics"]["node_count"],
            "edge_count": graph_a["diagnostics"]["edge_count"],
            "hop_counts_by_layer": graph_a["diagnostics"]["hop_counts_by_layer"],
            "ligand_latent": _tensor_stats(latent_a),
            "scalar_probe": float(scalar_probe_a.item()),
            "ligand_coordinate_gradient": _tensor_stats(ligand_grad_a),
        },
        "graph_b": {
            "policy": "sealed_manifest_complete_residue_expansion",
            "node_count": graph_b["diagnostics"]["node_count"],
            "edge_count": graph_b["diagnostics"]["edge_count"],
            "ligand_latent": _tensor_stats(latent_b),
            "scalar_probe": float(scalar_probe_b.item()),
            "ligand_coordinate_gradient": _tensor_stats(ligand_grad_b),
        },
        "comparison": {
            "ligand_latent_max_abs_diff": float(latent_diff.abs().max().item()),
            "ligand_latent_mean_abs_diff": float(latent_diff.abs().mean().item()),
            "scalar_probe_abs_diff": float(abs(scalar_probe_a.item() - scalar_probe_b.item())),
            "ligand_gradient_max_abs_diff": float(grad_diff.abs().max().item()),
            "ligand_gradient_mean_abs_diff": float(grad_diff.abs().mean().item()),
        },
        "inputs": inputs,
        "elapsed_seconds": time.perf_counter() - started,
        "policy": {
            "provisional_not_sealed": True,
            "scientific_qualification": False,
            "hard_gate": False,
            "decision_reference": "DEC-032",
            "training_executed": False,
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
