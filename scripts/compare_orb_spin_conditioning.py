#!/usr/bin/env python
"""Compare frozen parent-system spin conditioning on the existing nine audits.

The primary arm is always ``Q=0, M=1`` under the frozen parent-system
conditioning contract.  ``Q=0, M=3`` is sensitivity-only and cannot change
the primary choice.  Both arms use the same full-parent-derived L2 closure
membership recorded by the existing graph-audit JSON reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.orb_latent import (  # noqa: E402
    OrbLatentAdapter,
    OrbModelSpec,
    OrbParentConditioningContract,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cosine_rows(left, right):
    import numpy as np

    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = left_norm * right_norm
    if np.any(denominator <= 0.0):
        raise RuntimeError("spin sensitivity encountered a zero-norm ligand latent row")
    return np.sum(left * right, axis=1) / denominator


def _summary_vector(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-reports", nargs="+", required=True)
    parser.add_argument("--model-name", default="orb-v3-conservative-omol")
    parser.add_argument("--model-path", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--latent-output")
    args = parser.parse_args(argv)

    import mdtraj as md
    import numpy as np

    output = Path(args.output)
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")
    latent_output = Path(args.latent_output) if args.latent_output else None
    if latent_output is not None and latent_output.exists():
        parser.error(f"refusing to overwrite existing latent NPZ: {latent_output}")
    if len(args.graph_reports) != 9:
        parser.error("the pre-registered sensitivity set contains exactly 9 graph-audit reports")

    reports = []
    for report_arg in args.graph_reports:
        path = Path(report_arg).resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        layers = report["graph"]["layers"]
        layer2 = next(layer for layer in layers if int(layer["layer"]) == 2)
        reports.append((path, report, layer2))
    reports.sort(key=lambda item: (
        item[1]["inputs"]["trajectory"]["path"],
        int(item[1]["inputs"]["frame_index"]),
    ))

    adapter = OrbLatentAdapter(
        OrbModelSpec(model_name=args.model_name, model_path=args.model_path, primary_layer=2)
    )
    primary_contract = OrbParentConditioningContract(
        total_charge=0.0, spin_multiplicity=1.0, role="primary"
    )
    sensitivity_contract = OrbParentConditioningContract(
        total_charge=0.0, spin_multiplicity=3.0, role="sensitivity"
    )

    primary_latents = []
    sensitivity_latents = []
    per_frame = []
    for report_path, report, layer2 in reports:
        topology_path = Path(report["inputs"]["topology"]["path"]).resolve()
        trajectory_path = Path(report["inputs"]["trajectory"]["path"]).resolve()
        frame_index = int(report["inputs"]["frame_index"])
        ligand_indices = [int(value) for value in report["inputs"]["ligand_indices"]]
        trajectory = md.load_frame(str(trajectory_path), frame_index, top=str(topology_path))
        if trajectory.unitcell_vectors is None:
            raise SystemExit(f"{report_path}: trajectory has no periodic cell")
        positions = np.asarray(trajectory.xyz[0], dtype=np.float64) * 10.0
        cell = np.asarray(trajectory.unitcell_vectors[0], dtype=np.float64) * 10.0
        topology_indices = np.asarray(layer2["topology_indices"], dtype=np.int64)
        local_index_by_topology = {int(value): index for index, value in enumerate(topology_indices)}
        local_ligand_indices = [local_index_by_topology[index] for index in ligand_indices]
        topology = trajectory.topology
        atomic_numbers = [
            int(topology.atom(int(index)).element.atomic_number) for index in topology_indices
        ]
        local_positions = positions[topology_indices]

        primary_result = adapter.extract_frame(
            local_positions,
            cell,
            atomic_numbers=atomic_numbers,
            ligand_indices=local_ligand_indices,
            topology_indices=topology_indices,
            conditioning_contract=primary_contract,
            layer=2,
        )
        sensitivity_result = adapter.extract_frame(
            local_positions,
            cell,
            atomic_numbers=atomic_numbers,
            ligand_indices=local_ligand_indices,
            topology_indices=topology_indices,
            conditioning_contract=sensitivity_contract,
            layer=2,
        )
        primary = primary_result.ligand_latent.detach().cpu().numpy().astype(np.float32)
        sensitivity = sensitivity_result.ligand_latent.detach().cpu().numpy().astype(np.float32)
        if primary.shape != sensitivity.shape or primary.shape != (len(ligand_indices), 256):
            raise SystemExit(f"{report_path}: unexpected latent shape {primary.shape}/{sensitivity.shape}")
        primary_latents.append(primary)
        sensitivity_latents.append(sensitivity)
        pooled_primary = primary.mean(axis=0)
        pooled_sensitivity = sensitivity.mean(axis=0)
        pooled_delta = pooled_sensitivity - pooled_primary
        per_frame.append(
            {
                "graph_report": str(report_path),
                "trajectory": str(trajectory_path),
                "frame_index": frame_index,
                "node_count": int(layer2["node_count"]),
                "edge_count": int(layer2["edge_count"]),
                "pooled_latent_relative_l2_difference": float(
                    np.linalg.norm(pooled_delta) / max(np.linalg.norm(pooled_primary), 1e-12)
                ),
                "ligand_node_cosine_similarity": _summary_vector(_cosine_rows(primary, sensitivity)),
                "primary_latent_sha256_raw_float32": hashlib.sha256(
                    np.ascontiguousarray(primary).tobytes()
                ).hexdigest(),
                "sensitivity_latent_sha256_raw_float32": hashlib.sha256(
                    np.ascontiguousarray(sensitivity).tobytes()
                ).hexdigest(),
            }
        )

    primary_array = np.stack(primary_latents, axis=0)
    sensitivity_array = np.stack(sensitivity_latents, axis=0)
    primary_flat = primary_array.reshape(-1, 256)
    sensitivity_flat = sensitivity_array.reshape(-1, 256)
    primary_std = primary_flat.std(axis=0)
    sensitivity_std = sensitivity_flat.std(axis=0)
    std_delta = sensitivity_std - primary_std
    pooled_primary_all = primary_array.mean(axis=1)
    pooled_sensitivity_all = sensitivity_array.mean(axis=1)
    pooled_relative_l2 = np.linalg.norm(
        pooled_sensitivity_all - pooled_primary_all, axis=1
    ) / np.maximum(np.linalg.norm(pooled_primary_all, axis=1), 1e-12)
    all_cosines = _cosine_rows(primary_flat, sensitivity_flat)
    body = {
        "schema_version": "orb-spin-conditioning-sensitivity-v1",
        "status": "COMPLETED_SENSITIVITY_ONLY",
        "command": " ".join(sys.argv),
        "model_name": args.model_name,
        "layer": 2,
        "primary_contract": primary_contract.to_dict(),
        "sensitivity_contract": sensitivity_contract.to_dict(),
        "sample_count": int(primary_array.shape[0]),
        "latent_shape": [int(value) for value in primary_array.shape],
        "primary_qualification": "primary is fixed to Q=0,M=1; sensitivity cannot select multiplicity",
        "aggregate": {
            "pooled_latent_relative_l2_difference": _summary_vector(pooled_relative_l2),
            "ligand_node_cosine_similarity": _summary_vector(all_cosines),
            "primary_latent_std_by_dimension": primary_std.tolist(),
            "sensitivity_latent_std_by_dimension": sensitivity_std.tolist(),
            "std_change_by_dimension": std_delta.tolist(),
            "std_change_l2": float(np.linalg.norm(std_delta)),
            "std_change_max_abs": float(np.max(np.abs(std_delta))),
            "std_change_relative_l2": float(
                np.linalg.norm(std_delta) / max(np.linalg.norm(primary_std), 1e-12)
            ),
            "primary_latent_sha256_raw_float32": hashlib.sha256(
                np.ascontiguousarray(primary_array).tobytes()
            ).hexdigest(),
            "sensitivity_latent_sha256_raw_float32": hashlib.sha256(
                np.ascontiguousarray(sensitivity_array).tobytes()
            ).hexdigest(),
        },
        "per_frame": per_frame,
        "policy": {
            "primary_spin_is_preregistered": True,
            "sensitivity_spin_used_for_model_selection": False,
            "spin_zero_null_sentinel_tested": False,
            "orb_total_energy_used_as_target": False,
            "same_parent_derived_l2_membership_for_both_arms": True,
        },
    }
    if latent_output is not None:
        latent_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            latent_output,
            primary_ligand_latent=primary_array,
            sensitivity_ligand_latent=sensitivity_array,
            frame_index=np.asarray([int(item[1]["inputs"]["frame_index"]) for item in reports], dtype=np.int64),
        )
        body["latent_output"] = {
            "path": str(latent_output.resolve()),
            "sha256": _sha256_file(latent_output),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sample_count": body["sample_count"], "layer": 2}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
