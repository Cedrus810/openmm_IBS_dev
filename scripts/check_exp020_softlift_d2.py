#!/usr/bin/env python
"""Run the offline R1 D2 geometry, force, and locality qualification.

This check is deliberately reference-only.  It does not build a native
OpenMM force, start online production, or modify any existing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.geometry import ligand_environment_cross_edges  # noqa: E402
from local_residual.softlift import (  # noqa: E402
    PackedSoftLiftBatch,
    SoftLiftConfig,
    _quintic_c2,
    build_softlift_model,
)


EXPERIMENT_ID = "EXP-020"
FINITE_DIFFERENCE_ABS_TOLERANCE = 1.0e-5
FINITE_DIFFERENCE_RELATIVE_TOLERANCE = 1.0e-3
INVARIANCE_TOLERANCE = 1.0e-10
NONPARTICIPANT_FORCE_TOLERANCE = 1.0e-10


class D2Error(RuntimeError):
    """A D2 input or fail-closed contract is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(checkpoint: Path):
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise D2Error("checkpoint is not an EXP-020 artifact")
    raw = dict(payload["config"])
    raw["type_vocabulary"] = tuple(int(value) for value in raw["type_vocabulary"])
    config = SoftLiftConfig(**raw)
    if config.rung != "R1":
        raise D2Error("this checker only qualifies R1")
    model = build_softlift_model(config).double()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, payload


def _live_batch(model_config: SoftLiftConfig, positions, box, atomic_numbers, ligand_indices):
    import torch

    ligand_indices = [int(value) for value in ligand_indices]
    ligand = torch.tensor(ligand_indices, dtype=torch.int64, device=positions.device)
    ligand_set = set(ligand_indices)
    environment_indices = [index for index in range(len(atomic_numbers)) if index not in ligand_set]
    environment = torch.tensor(environment_indices, dtype=torch.int64, device=positions.device)
    edges = ligand_environment_cross_edges(
        positions,
        box,
        ligand,
        environment,
        outer_cutoff=model_config.outer_cutoff_angstrom,
    )
    type_map = {int(value): index for index, value in enumerate(model_config.type_vocabulary)}
    try:
        ligand_types = torch.tensor(
            [type_map[int(atomic_numbers[index])] for index in ligand_indices],
            dtype=torch.int64,
            device=positions.device,
        )
        environment_types = torch.tensor(
            [type_map[int(atomic_numbers[index])] for index in environment_indices],
            dtype=torch.int64,
            device=positions.device,
        )
    except KeyError as exc:
        raise D2Error(f"atomic number {exc.args[0]} is outside the frozen R1 vocabulary") from exc
    edge_ligand_global = edges["edge_index"][0]
    edge_environment = edges["edge_index"][1]
    local_lookup = {value: index for index, value in enumerate(ligand_indices)}
    edge_local = torch.tensor(
        [local_lookup[int(value)] for value in edge_ligand_global.tolist()],
        dtype=torch.int64,
        device=positions.device,
    )
    environment_lookup = {value: index for index, value in enumerate(environment_indices)}
    edge_environment_local = torch.tensor(
        [environment_lookup[int(value)] for value in edge_environment.tolist()],
        dtype=torch.int64,
        device=positions.device,
    )
    return PackedSoftLiftBatch(
        ligand_type_index=ligand_types[None, :],
        edge_frame=torch.zeros(edge_local.shape[0], dtype=torch.int64, device=positions.device),
        edge_ligand_local=edge_local,
        edge_ligand_type=ligand_types[edge_local],
        edge_environment_type=environment_types[edge_environment_local],
        edge_distance_angstrom=edges["distance"],
        edge_displacement_angstrom=edges["displacement"],
    )


def _evaluate(model, config, positions, box, atomic_numbers, ligand_indices):
    import torch

    coordinates = positions.detach().clone().requires_grad_(True)
    batch = _live_batch(config, coordinates, box, atomic_numbers, ligand_indices)
    basis = model(batch)[0]
    gradient = torch.autograd.grad(basis, coordinates, allow_unused=True)[0]
    if gradient is None:
        gradient = torch.zeros_like(coordinates)
    return float(basis.detach().item()), gradient.detach(), batch


def _energy(model, config, positions, box, atomic_numbers, ligand_indices) -> float:
    import torch

    coordinates = positions.detach().clone()
    batch = _live_batch(config, coordinates, box, atomic_numbers, ligand_indices)
    with torch.no_grad():
        return float(model(batch)[0].item())


def _synthetic_system():
    import torch

    box = torch.tensor(
        [[20.0, 0.0, 0.0], [3.0, 19.0, 0.0], [1.0, 2.0, 21.0]], dtype=torch.float64
    )
    ligand_fraction = torch.tensor([0.10, 0.10, 0.10], dtype=torch.float64).repeat(41, 1)
    ligand_fraction[:, 1] += torch.linspace(-0.01, 0.01, 41, dtype=torch.float64)
    participant_fraction = torch.tensor([[0.95, 0.10, 0.10]], dtype=torch.float64)
    nonparticipant_fraction = torch.tensor([[0.50, 0.50, 0.50]], dtype=torch.float64)
    fractions = torch.cat((ligand_fraction, participant_fraction, nonparticipant_fraction), dim=0)
    positions = fractions @ box
    atomic_numbers = [6] * 41 + [8, 17]
    return positions, box, atomic_numbers, list(range(41))


def _check_c2_cutoff() -> dict[str, Any]:
    import torch

    inner = 4.0
    outer = 5.0
    errors = {}
    for name, location, expected in (("inner", inner, 1.0), ("outer", outer, 0.0)):
        distance = torch.tensor([location], dtype=torch.float64, requires_grad=True)
        value = _quintic_c2(distance, inner, outer)
        first = torch.autograd.grad(value.sum(), distance, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), distance)[0]
        errors[name] = {
            "value_error": abs(float(value.item()) - expected),
            "first_derivative_abs": abs(float(first.item())),
            "second_derivative_abs": abs(float(second.item())),
        }
    outside = _quintic_c2(torch.tensor([outer + 1.0], dtype=torch.float64), inner, outer)
    inside = _quintic_c2(torch.tensor([inner - 1.0], dtype=torch.float64), inner, outer)
    errors["exact_piecewise_values"] = {
        "inside_is_one": float(inside.item()) == 1.0,
        "outside_is_zero": float(outside.item()) == 0.0,
    }
    maximum = max(
        value
        for boundary in (errors["inner"], errors["outer"])
        for value in boundary.values()
    )
    passed = bool(
        maximum <= INVARIANCE_TOLERANCE
        and errors["exact_piecewise_values"]["inside_is_one"]
        and errors["exact_piecewise_values"]["outside_is_zero"]
    )
    return {"passed": passed, "max_boundary_error": maximum, "details": errors}


def _check_triclinic_pbc() -> dict[str, Any]:
    import torch

    box = torch.tensor(
        [[20.0, 0.0, 0.0], [3.0, 19.0, 0.0], [1.0, 2.0, 21.0]], dtype=torch.float64
    )
    fractional = torch.tensor([[0.10, 0.10, 0.10], [0.95, 0.10, 0.10]], dtype=torch.float64)
    positions = fractional @ box
    result = ligand_environment_cross_edges(
        positions,
        box,
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
        outer_cutoff=5.0,
    )
    expected = torch.tensor([-0.15, 0.0, 0.0], dtype=torch.float64) @ box
    displacement_error = float(torch.max(torch.abs(result["displacement"][0] - expected)).item()) if result["distance"].numel() else float("inf")
    return {
        "passed": bool(result["distance"].numel() == 1 and displacement_error <= INVARIANCE_TOLERANCE),
        "edge_count": int(result["distance"].numel()),
        "expected_displacement_angstrom": expected.tolist(),
        "actual_displacement_angstrom": result["displacement"][0].tolist() if result["distance"].numel() else None,
        "displacement_max_abs_error": displacement_error,
    }


def _check_synthetic_invariance_and_locality(model, config) -> dict[str, Any]:
    import torch

    positions, box, atomic_numbers, ligand_indices = _synthetic_system()
    base_energy, base_gradient, base_batch = _evaluate(model, config, positions, box, atomic_numbers, ligand_indices)
    translation = torch.tensor([1.234, -2.345, 0.456], dtype=torch.float64)
    translated_energy, translated_gradient, translated_batch = _evaluate(
        model, config, positions + translation, box, atomic_numbers, ligand_indices
    )
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    rotated_positions = positions @ rotation.T
    rotated_box = box @ rotation.T
    rotated_energy, rotated_gradient, rotated_batch = _evaluate(
        model, config, rotated_positions, rotated_box, atomic_numbers, ligand_indices
    )
    rotation_gradient_expected = base_gradient @ rotation.T
    invariance_error = max(
        abs(base_energy - translated_energy),
        abs(base_energy - rotated_energy),
        float(torch.max(torch.abs(base_gradient - translated_gradient)).item()),
        float(torch.max(torch.abs(rotation_gradient_expected - rotated_gradient)).item()),
    )

    no_contact_positions = positions.clone()
    no_contact_positions[41] = torch.tensor([0.50, 0.50, 0.50], dtype=torch.float64) @ box
    no_contact_energy, no_contact_gradient, no_contact_batch = _evaluate(
        model, config, no_contact_positions, box, atomic_numbers, ligand_indices
    )
    nonparticipant_force = float(torch.max(torch.abs(base_gradient[42])).item())
    no_contact_force = float(torch.max(torch.abs(no_contact_gradient)).item())
    return {
        "passed": bool(
            invariance_error <= INVARIANCE_TOLERANCE
            and no_contact_energy == 0.0
            and no_contact_force <= NONPARTICIPANT_FORCE_TOLERANCE
            and nonparticipant_force <= NONPARTICIPANT_FORCE_TOLERANCE
        ),
        "invariance_max_abs_reduced": invariance_error,
        "translation_energy_abs_reduced": abs(base_energy - translated_energy),
        "rotation_energy_abs_reduced": abs(base_energy - rotated_energy),
        "base_edge_count": base_batch.n_edges,
        "translated_edge_count": translated_batch.n_edges,
        "rotated_edge_count": rotated_batch.n_edges,
        "no_contact_energy_reduced": no_contact_energy,
        "no_contact_edge_count": no_contact_batch.n_edges,
        "no_contact_force_max_abs_reduced_per_angstrom": no_contact_force,
        "nonparticipant_atom_index": 42,
        "nonparticipant_force_max_abs_reduced_per_angstrom": nonparticipant_force,
    }


def _check_finite_difference(model, config, positions, box, atomic_numbers, ligand_indices, step, atom):
    base_energy, base_gradient, base_batch = _evaluate(model, config, positions, box, atomic_numbers, ligand_indices)
    coordinates = []
    for coordinate in range(3):
        plus = positions.clone()
        minus = positions.clone()
        plus[atom, coordinate] += step
        minus[atom, coordinate] -= step
        plus_energy = _energy(model, config, plus, box, atomic_numbers, ligand_indices)
        minus_energy = _energy(model, config, minus, box, atomic_numbers, ligand_indices)
        central = (plus_energy - minus_energy) / (2.0 * step)
        autograd = float(base_gradient[atom, coordinate].item())
        absolute = abs(central - autograd)
        relative = absolute / max(abs(central), abs(autograd), 1.0e-12)
        coordinates.append({
            "coordinate": coordinate,
            "central_difference": central,
            "autograd": autograd,
            "absolute_error": absolute,
            "relative_error": relative,
        })
    max_absolute = max(item["absolute_error"] for item in coordinates)
    max_relative = max(item["relative_error"] for item in coordinates)
    return {
        "passed": bool(
            max_absolute <= FINITE_DIFFERENCE_ABS_TOLERANCE
            and max_relative <= FINITE_DIFFERENCE_RELATIVE_TOLERANCE
        ),
        "atom": int(atom),
        "step_angstrom": float(step),
        "base_energy_reduced": base_energy,
        "edge_count": base_batch.n_edges,
        "max_absolute_error": max_absolute,
        "max_relative_error": max_relative,
        "coordinates": coordinates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--ligand-indices", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--atom", type=int, default=None)
    parser.add_argument("--step-angstrom", type=float, default=1.0e-5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    import mdtraj
    import torch

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise D2Error(f"missing checkpoint: {checkpoint}")
    model, config, payload = _load_model(checkpoint)
    ligand_indices = [int(value) for value in json.loads(Path(args.ligand_indices).read_text())["ligand_indices"]]
    if args.frame < 0:
        raise D2Error("--frame must be non-negative")
    trajectory = mdtraj.load_frame(args.trajectory, index=args.frame, top=args.topology)
    if trajectory.unitcell_vectors is None:
        raise D2Error("selected trajectory frame has no periodic box")
    positions = torch.tensor(trajectory.xyz[0] * 10.0, dtype=torch.float64)
    box = torch.tensor(trajectory.unitcell_vectors[0] * 10.0, dtype=torch.float64)
    atomic_numbers = [int(atom.element.atomic_number) for atom in trajectory.topology.atoms]
    atom = ligand_indices[0] if args.atom is None else int(args.atom)
    if atom not in ligand_indices:
        raise D2Error("--atom must be one of the ligand indices")
    if args.step_angstrom <= 0.0:
        raise D2Error("--step-angstrom must be positive")

    checks = {
        "finite_difference": _check_finite_difference(
            model, config, positions, box, atomic_numbers, ligand_indices, args.step_angstrom, atom
        ),
        "c2_cutoff": _check_c2_cutoff(),
        "triclinic_pbc": _check_triclinic_pbc(),
        "invariance_and_locality": _check_synthetic_invariance_and_locality(model, config),
    }
    full_d2 = bool(all(check["passed"] for check in checks.values()))
    body: dict[str, Any] = {
        "schema_version": "exp020-softlift-d2-v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETED_R1_D2_OFFLINE_REFERENCE_CHECK",
        "full_d2_qualification": full_d2,
        "rung": "R1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_seed": payload.get("seed"),
        "topology": str(Path(args.topology).resolve()),
        "trajectory": str(Path(args.trajectory).resolve()),
        "frame": int(args.frame),
        "checks": checks,
        "gates": {
            "finite_difference_energy_abs_reduced_max": FINITE_DIFFERENCE_ABS_TOLERANCE,
            "finite_difference_force_relative_max": FINITE_DIFFERENCE_RELATIVE_TOLERANCE,
            "nonparticipant_force_abs_reduced_per_angstrom_max": NONPARTICIPANT_FORCE_TOLERANCE,
            "invariance_abs_reduced_max": INVARIANCE_TOLERANCE,
        },
        "policy": {"native_started": False, "online_started": False, "production_promotion": False},
    }
    body["report_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    output = Path(args.output)
    if output.exists():
        raise D2Error(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": body["status"], "full_d2_qualification": full_d2}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
