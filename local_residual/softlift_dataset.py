"""EXP-019 dataset-v1 construction and fail-closed validation.

This module is intentionally separate from the EXP-012 dataset builder.  The
old dataset is immutable and distance-only; EXP-019 adds the per-frame box,
MIC displacement and image shift required for D0/D2/R3.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence


class SoftLiftDatasetError(ValueError):
    """A dataset identity, geometry, support, or canonical-order check failed."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_npz(path: Path, arrays: dict[str, Any]) -> None:
    import numpy as np

    if path.exists():
        raise SoftLiftDatasetError(f"refusing to overwrite existing dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise SoftLiftDatasetError(f"refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _box_metrics(box: Any, outer_cutoff_angstrom: float) -> dict[str, float | bool | list[float]]:
    import numpy as np

    cell = np.asarray(box, dtype=np.float64)
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise SoftLiftDatasetError("box must be a finite [3,3] row-vector matrix")
    determinant = float(np.linalg.det(cell))
    if determinant == 0.0 or not math.isfinite(determinant):
        raise SoftLiftDatasetError("box determinant is zero or non-finite")
    lengths = np.linalg.norm(cell, axis=1)
    face_heights = [
        abs(determinant) / float(np.linalg.norm(np.cross(cell[1], cell[2]))),
        abs(determinant) / float(np.linalg.norm(np.cross(cell[0], cell[2]))),
        abs(determinant) / float(np.linalg.norm(np.cross(cell[0], cell[1]))),
    ]
    condition = float(np.linalg.cond(cell))
    safe = bool(2.0 * float(outer_cutoff_angstrom) < min(face_heights))
    return {
        "determinant_angstrom3": determinant,
        "condition_number": condition,
        "edge_lengths_angstrom": lengths.tolist(),
        "face_heights_angstrom": face_heights,
        "safe_2cutoff_lt_min_face_height": safe,
    }


def _unit_shift_and_displacement(
    ligand_position: Any,
    environment_position: Any,
    box: Any,
    *,
    half_box_tolerance: float = 1e-12,
):
    """Return ``env-ligand`` MIC displacement and its integer image shift.

    The shift is the image applied to the environment atom relative to the
    unshifted ligand: ``d = (x_env-x_lig + shift @ box)``.  A centered-cell
    half-box tie is ambiguous and therefore fails closed.
    """

    import numpy as np

    raw = np.asarray(environment_position, dtype=np.float64) - np.asarray(ligand_position, dtype=np.float64)
    cell = np.asarray(box, dtype=np.float64)
    fractional = np.linalg.solve(cell.T, raw)
    if np.any(np.isclose(np.abs(fractional), 0.5, atol=half_box_tolerance, rtol=0.0)):
        raise SoftLiftDatasetError("half-box MIC tie detected")
    nearest = np.rint(fractional).astype(np.int64)
    shift = -nearest
    displacement = (fractional + shift) @ cell
    return displacement, shift


def _numpy_cross_edges(
    positions_angstrom: Any,
    box_angstrom: Any,
    ligand_indices: Any,
    environment_indices: Any,
    outer_cutoff_angstrom: float,
) -> tuple[Any, Any, Any, Any]:
    """Build exact strict-cutoff edges with a diagonal-box cell list.

    EXP-019's audited data boxes are diagonal and satisfy the safe-domain
    condition.  This path avoids the ``41 * 73k`` all-pairs allocation in the
    D0 builder.  Non-diagonal boxes use a vectorized brute-force fallback;
    online Torch deployment remains separately marked as brute-force for
    triclinic boxes.
    """

    import numpy as np

    positions = np.asarray(positions_angstrom, dtype=np.float64)
    box = np.asarray(box_angstrom, dtype=np.float64)
    ligand = np.asarray(ligand_indices, dtype=np.int64)
    environment = np.asarray(environment_indices, dtype=np.int64)
    cutoff = float(outer_cutoff_angstrom)
    diagonal = np.allclose(box, np.diag(np.diag(box)), rtol=0.0, atol=1.0e-12)
    if diagonal:
        lengths = np.diag(box)
        if np.any(lengths <= 0.0):
            raise SoftLiftDatasetError("diagonal box lengths must be positive")
        n_bins = np.maximum(np.floor(lengths / cutoff).astype(np.int64), 1)
        fractional = np.mod(positions / lengths[None, :], 1.0)
        environment_bins: dict[tuple[int, int, int], list[int]] = {}
        for topology_index in environment.tolist():
            key = tuple(np.floor(fractional[topology_index] * n_bins).astype(np.int64).tolist())
            environment_bins.setdefault(key, []).append(int(topology_index))
        edge_ligand: list[int] = []
        edge_environment: list[int] = []
        edge_displacement: list[Any] = []
        edge_distance: list[float] = []
        for ligand_index in ligand.tolist():
            ligand_bin = np.floor(fractional[ligand_index] * n_bins).astype(np.int64)
            candidates: set[int] = set()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor = tuple(((ligand_bin + np.asarray((dx, dy, dz))) % n_bins).tolist())
                        candidates.update(environment_bins.get(neighbor, ()))
            for environment_index in sorted(candidates):
                raw = positions[environment_index] - positions[ligand_index]
                shift = -np.rint(raw / lengths).astype(np.int64)
                displacement = raw + shift * lengths
                distance = float(np.linalg.norm(displacement))
                if distance < cutoff:
                    edge_ligand.append(int(ligand_index))
                    edge_environment.append(environment_index)
                    edge_displacement.append(displacement)
                    edge_distance.append(distance)
        return (
            np.asarray(edge_ligand, dtype=np.int64),
            np.asarray(edge_environment, dtype=np.int64),
            np.asarray(edge_displacement, dtype=np.float64).reshape((-1, 3)),
            np.asarray(edge_distance, dtype=np.float64),
        )

    raw = positions[environment][None, :, :] - positions[ligand][:, None, :]
    fractional = np.linalg.solve(box.T, raw.reshape((-1, 3)).T).T.reshape(raw.shape)
    if np.any(np.isclose(np.abs(fractional), 0.5, atol=1e-12, rtol=0.0)):
        raise SoftLiftDatasetError("half-box MIC tie detected")
    shift = -np.rint(fractional).astype(np.int64)
    displacement = np.matmul((fractional + shift).reshape((-1, 3)), box).reshape(raw.shape)
    distance = np.linalg.norm(displacement, axis=-1)
    retained = distance < cutoff
    ligand_grid = np.broadcast_to(ligand[:, None], distance.shape)[retained]
    environment_grid = np.broadcast_to(environment[None, :], distance.shape)[retained]
    return ligand_grid.astype(np.int64), environment_grid.astype(np.int64), displacement[retained], distance[retained]


def canonicalize_frame_edges(
    *,
    ligand_topology: Any,
    environment_topology: Any,
    displacement_angstrom: Any,
    distance_angstrom: Any,
    pair_type: Any,
    unit_shift: Any,
    ligand_topology_indices: Sequence[int],
    outer_cutoff_angstrom: float,
    min_distance_support_angstrom: float,
    max_environment_atoms: int,
    max_edges: int,
    max_neighbors_per_ligand: int,
) -> dict[str, Any]:
    """Validate and canonicalize one frame's strict-cutoff edge arrays."""

    import numpy as np

    lig = np.asarray(ligand_topology, dtype=np.int64)
    env = np.asarray(environment_topology, dtype=np.int64)
    disp = np.asarray(displacement_angstrom, dtype=np.float64)
    dist = np.asarray(distance_angstrom, dtype=np.float64)
    pair = np.asarray(pair_type, dtype=np.int64)
    shift = np.asarray(unit_shift, dtype=np.int64)
    count = int(dist.shape[0])
    if lig.shape != (count,) or env.shape != (count,) or disp.shape != (count, 3) or pair.shape != (count, 2) or shift.shape != (count, 3):
        raise SoftLiftDatasetError("frame edge arrays have inconsistent shapes")
    if not np.isfinite(dist).all() or not np.isfinite(disp).all() or (dist < 0.0).any():
        raise SoftLiftDatasetError("frame edge distances/displacements are non-finite or negative")
    if not np.allclose(np.linalg.norm(disp, axis=1), dist, rtol=0.0, atol=1e-8):
        raise SoftLiftDatasetError("edge distance does not match displacement norm")
    if (dist >= float(outer_cutoff_angstrom)).any():
        raise SoftLiftDatasetError("dataset contains an edge at or beyond strict outer cutoff")
    if (dist < float(min_distance_support_angstrom)).any():
        raise SoftLiftDatasetError("edge enters the forbidden minimum-distance support domain")
    ligand_order = np.asarray(ligand_topology_indices, dtype=np.int64)
    if len(np.unique(ligand_order)) != len(ligand_order) or not np.array_equal(ligand_order, np.sort(ligand_order)):
        raise SoftLiftDatasetError("ligand topology ordering must be sorted and unique")
    local = np.searchsorted(ligand_order, lig)
    if count and np.any(local >= len(ligand_order)):
        raise SoftLiftDatasetError("edge ligand topology is outside ligand ordering")
    if count and not np.array_equal(ligand_order[local], lig):
        raise SoftLiftDatasetError("edge ligand topology is not in ligand ordering")
    keys = [
        (int(local[index]), int(env[index]), int(shift[index, 0]), int(shift[index, 1]), int(shift[index, 2]))
        for index in range(count)
    ]
    if len(set(keys)) != len(keys):
        raise SoftLiftDatasetError("duplicate canonical edge key")
    order = np.asarray(sorted(range(count), key=keys.__getitem__), dtype=np.int64)
    local = local[order].astype(np.int64, copy=False)
    env = env[order].astype(np.int64, copy=False)
    disp = disp[order]
    dist = dist[order]
    pair = pair[order].astype(np.int64, copy=False)
    shift = shift[order].astype(np.int64, copy=False)
    unique_environment = np.unique(env)
    neighbor_counts = np.bincount(local, minlength=len(ligand_order))
    if unique_environment.size > max_environment_atoms:
        raise SoftLiftDatasetError("unique environment atom count exceeds hard ceiling")
    if count > max_edges:
        raise SoftLiftDatasetError("directed edge count exceeds hard ceiling")
    if neighbor_counts.size and int(neighbor_counts.max()) > max_neighbors_per_ligand:
        raise SoftLiftDatasetError("per-ligand neighbor count exceeds hard ceiling")
    return {
        "edge_ligand_local": local,
        "edge_environment_topology": env,
        "edge_displacement_angstrom": disp,
        "edge_distance_angstrom": dist,
        "edge_pair_type": pair,
        "edge_unit_shift": shift,
        "unique_environment_count": int(unique_environment.size),
        "edge_count": count,
        "max_neighbors_per_ligand": int(neighbor_counts.max()) if neighbor_counts.size else 0,
        "target_environment_overflow": int(unique_environment.size) > 256,
        "target_edge_overflow": count > 1536,
        "target_neighbor_overflow": bool(neighbor_counts.size and neighbor_counts.max() > 64),
    }


def canonical_edge_hash(arrays: dict[str, Any]) -> str:
    """Hash canonical edge identity and geometry with explicit dtypes."""

    import numpy as np

    digest = hashlib.sha256()
    for key in (
        "edge_offsets", "edge_ligand_topology", "edge_environment_topology", "edge_ligand_local",
        "edge_distance_angstrom", "edge_displacement_angstrom", "edge_unit_shift", "edge_pair_type",
    ):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def validate_softlift_dataset_arrays(
    arrays: dict[str, Any],
    *,
    outer_cutoff_angstrom: float = 5.0,
    min_distance_support_angstrom: float = 0.1,
    max_environment_atoms: int = 320,
    max_edges: int = 2048,
    max_neighbors_per_ligand: int = 80,
) -> dict[str, Any]:
    """Validate all dataset-v1 fields and return audit statistics."""

    import numpy as np

    required = {
        "partition_index", "frame_index_within_run", "box_vectors_angstrom", "edge_offsets",
        "edge_ligand_topology", "edge_environment_topology", "edge_ligand_local",
        "edge_distance_angstrom", "edge_displacement_angstrom", "edge_unit_shift", "edge_pair_type",
        "adjacent_gap_reduced", "log_importance_unnormalized", "ligand_topology_indices",
        "ligand_atomic_numbers", "all_topology_atomic_numbers", "type_vocabulary", "delta_A", "A_k_window",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise SoftLiftDatasetError(f"dataset is missing required fields: {missing}")
    partition = np.asarray(arrays["partition_index"])
    frame_index = np.asarray(arrays["frame_index_within_run"])
    boxes = np.asarray(arrays["box_vectors_angstrom"])
    offsets = np.asarray(arrays["edge_offsets"])
    if partition.ndim != 1 or frame_index.shape != partition.shape or boxes.shape != (partition.size, 3, 3):
        raise SoftLiftDatasetError("frame-level fields have inconsistent shapes")
    if offsets.shape != (partition.size + 1,) or offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise SoftLiftDatasetError("edge_offsets is not a valid CSR pointer array")
    edge_count = int(offsets[-1])
    one_d = ("edge_ligand_topology", "edge_environment_topology", "edge_ligand_local", "edge_distance_angstrom")
    if any(np.asarray(arrays[key]).shape != (edge_count,) for key in one_d):
        raise SoftLiftDatasetError("one-dimensional edge fields disagree with edge_offsets")
    if np.asarray(arrays["edge_displacement_angstrom"]).shape != (edge_count, 3) or np.asarray(arrays["edge_unit_shift"]).shape != (edge_count, 3) or np.asarray(arrays["edge_pair_type"]).shape != (edge_count, 2):
        raise SoftLiftDatasetError("vector/pair edge fields disagree with edge_offsets")
    ligand_order = np.asarray(arrays["ligand_topology_indices"], dtype=np.int64)
    if ligand_order.ndim != 1 or not ligand_order.size or len(np.unique(ligand_order)) != ligand_order.size:
        raise SoftLiftDatasetError("ligand_topology_indices must be a unique non-empty array")
    edge_local = np.asarray(arrays["edge_ligand_local"], dtype=np.int64)
    if edge_count and (edge_local.min() < 0 or edge_local.max() >= ligand_order.size):
        raise SoftLiftDatasetError("edge_ligand_local exceeds ligand-local range")
    all_atomic_numbers = np.asarray(arrays["all_topology_atomic_numbers"], dtype=np.int64)
    if all_atomic_numbers.ndim != 1 or ligand_order.max(initial=-1) >= all_atomic_numbers.size:
        raise SoftLiftDatasetError("topology atomic-number identity is invalid")
    if not np.array_equal(np.asarray(arrays["ligand_atomic_numbers"], dtype=np.int64), all_atomic_numbers[ligand_order]):
        raise SoftLiftDatasetError("ligand_atomic_numbers disagrees with topology identity")
    vocabulary = np.asarray(arrays["type_vocabulary"], dtype=np.int64)
    if vocabulary.ndim != 1 or len(np.unique(vocabulary)) != vocabulary.size or not set(np.unique(all_atomic_numbers)).issubset(set(vocabulary.tolist())):
        raise SoftLiftDatasetError("type_vocabulary does not cover topology atomic numbers")
    edge_ligand_topology = np.asarray(arrays["edge_ligand_topology"], dtype=np.int64)
    edge_environment_topology = np.asarray(arrays["edge_environment_topology"], dtype=np.int64)
    if edge_count and (
        edge_ligand_topology.min() < 0 or edge_environment_topology.min() < 0
        or edge_ligand_topology.max() >= all_atomic_numbers.size
        or edge_environment_topology.max() >= all_atomic_numbers.size
    ):
        raise SoftLiftDatasetError("edge topology index is outside all_topology_atomic_numbers")
    if edge_count:
        if not np.array_equal(edge_ligand_topology, ligand_order[edge_local]):
            raise SoftLiftDatasetError("edge_ligand_topology disagrees with local reindex")
        expected_pair = np.stack((all_atomic_numbers[edge_ligand_topology], all_atomic_numbers[edge_environment_topology]), axis=1)
        if not np.array_equal(np.asarray(arrays["edge_pair_type"], dtype=np.int64), expected_pair):
            raise SoftLiftDatasetError("edge_pair_type disagrees with topology atomic numbers")
    if not np.isfinite(np.asarray(arrays["adjacent_gap_reduced"], dtype=np.float64)).all() or not np.isfinite(np.asarray(arrays["log_importance_unnormalized"], dtype=np.float64)).all():
        raise SoftLiftDatasetError("ledger arrays contain non-finite values")
    if np.asarray(arrays["log_importance_unnormalized"]).shape[0] != partition.size or np.asarray(arrays["adjacent_gap_reduced"]).shape[0] != partition.size:
        raise SoftLiftDatasetError("ledger arrays are not frame-aligned")
    if np.asarray(arrays["delta_A"]).shape != (np.asarray(arrays["adjacent_gap_reduced"]).shape[1],) or np.asarray(arrays["A_k_window"]).shape != (np.asarray(arrays["adjacent_gap_reduced"]).shape[1] + 1,):
        raise SoftLiftDatasetError("delta_A/A_k_window shapes do not match ledger state count")
    if partition.size:
        for label in np.unique(partition):
            indices = np.flatnonzero(partition == label)
            if not np.array_equal(frame_index[indices], np.arange(indices.size, dtype=frame_index.dtype)):
                raise SoftLiftDatasetError("frame_index_within_run is not contiguous from zero")
    per_frame_stats = []
    for frame in range(partition.size):
        start, end = int(offsets[frame]), int(offsets[frame + 1])
        chunk = {
            "edge_ligand_local": edge_local[start:end],
            "edge_environment_topology": np.asarray(arrays["edge_environment_topology"])[start:end],
            "edge_displacement_angstrom": np.asarray(arrays["edge_displacement_angstrom"])[start:end],
            "edge_distance_angstrom": np.asarray(arrays["edge_distance_angstrom"])[start:end],
            "edge_pair_type": np.asarray(arrays["edge_pair_type"])[start:end],
            "edge_unit_shift": np.asarray(arrays["edge_unit_shift"])[start:end],
        }
        # Re-check canonical ordering at the dataset boundary, including the
        # topology-index key rather than trusting the builder report.
        keys = [
            (int(chunk["edge_ligand_local"][i]), int(chunk["edge_environment_topology"][i]), *map(int, chunk["edge_unit_shift"][i]))
            for i in range(end - start)
        ]
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise SoftLiftDatasetError(f"frame {frame} edge ordering/key uniqueness failed")
        checked = canonicalize_frame_edges(
            ligand_topology=ligand_order[chunk["edge_ligand_local"]],
            environment_topology=chunk["edge_environment_topology"],
            displacement_angstrom=chunk["edge_displacement_angstrom"],
            distance_angstrom=chunk["edge_distance_angstrom"],
            pair_type=chunk["edge_pair_type"],
            unit_shift=chunk["edge_unit_shift"],
            ligand_topology_indices=ligand_order,
            outer_cutoff_angstrom=outer_cutoff_angstrom,
            min_distance_support_angstrom=min_distance_support_angstrom,
            max_environment_atoms=max_environment_atoms,
            max_edges=max_edges,
            max_neighbors_per_ligand=max_neighbors_per_ligand,
        )
        if not np.array_equal(checked["edge_ligand_local"], chunk["edge_ligand_local"]):
            raise SoftLiftDatasetError(f"frame {frame} is not in canonical local edge order")
        per_frame_stats.append({key: checked[key] for key in ("unique_environment_count", "edge_count", "max_neighbors_per_ligand")})
        if not _box_metrics(boxes[frame], outer_cutoff_angstrom)["safe_2cutoff_lt_min_face_height"]:
            raise SoftLiftDatasetError(f"frame {frame} box is outside the safe minimum-image domain")
    return {
        "frame_count": int(partition.size),
        "edge_count": edge_count,
        "n_ligand_atoms": int(ligand_order.size),
        "max_unique_environment_atoms": max((item["unique_environment_count"] for item in per_frame_stats), default=0),
        "max_edges_per_frame": max((item["edge_count"] for item in per_frame_stats), default=0),
        "max_neighbors_per_ligand": max((item["max_neighbors_per_ligand"] for item in per_frame_stats), default=0),
        "canonical_edge_hash": canonical_edge_hash(arrays),
    }


def build_dataset_v1(
    *,
    runs: Sequence[dict[str, Any]],
    ligand_topology_indices: Sequence[int],
    topology_path: str | Path,
    output_path: str | Path,
    outer_cutoff_angstrom: float = 5.0,
    min_distance_support_angstrom: float = 0.1,
    max_environment_atoms: int = 320,
    max_edges: int = 2048,
    max_neighbors_per_ligand: int = 80,
    device: str = "cpu",
    delta_A: Sequence[float] | None = None,
    A_k_window: Sequence[float] | None = None,
    protocol_sha256: str | None = None,
    atomic_numbers_override: Sequence[int] | None = None,
    skip_unsupported_frames: bool = False,
) -> dict[str, Any]:
    """Build dataset-v1 from the three fixed EXP-012 trajectories and ledgers.

    Each run dictionary contains ``run_id``, ``trajectory_path``,
    ``ledger_path`` and ``ledger_report_path``.  The function refuses to write
    over any existing dataset/report.
    """

    import mdtraj
    import numpy as np
    import torch

    if not runs or len(runs) < 3:
        raise SoftLiftDatasetError("at least three whole-run partitions are required")
    ligand_order = np.asarray(sorted(int(value) for value in ligand_topology_indices), dtype=np.int64)
    if ligand_order.size == 0 or len(np.unique(ligand_order)) != ligand_order.size:
        raise SoftLiftDatasetError("ligand_topology_indices must be non-empty and unique")
    topology_path = Path(topology_path)
    output_path = Path(output_path)
    if output_path.exists() or output_path.with_name(output_path.stem + "_report.json").exists():
        raise SoftLiftDatasetError("dataset-v1 output or report already exists")
    all_frames: list[dict[str, Any]] = []
    run_reports: dict[str, Any] = {}
    skipped_unsupported: dict[str, int] = {}
    reference_atomic_numbers = None
    for partition_index, run in enumerate(runs):
        run_id = str(run["run_id"])
        trajectory_path = Path(run["trajectory_path"])
        ledger_path = Path(run["ledger_path"])
        ledger_report_path = Path(run["ledger_report_path"])
        trajectory = mdtraj.load(str(trajectory_path), top=str(topology_path))
        if trajectory.unitcell_vectors is None:
            raise SoftLiftDatasetError(f"{run_id}: trajectory has no per-frame unitcell_vectors")
        with np.load(ledger_path) as ledger:
            frame_indices = np.asarray(ledger["frame_index"])
            gaps = np.asarray(ledger["adjacent_gap_reduced"], dtype=np.float64)
            log_weights = np.asarray(ledger["log_importance_unnormalized"], dtype=np.float64)
        ledger_report = json.loads(ledger_report_path.read_text(encoding="utf-8"))
        frame_count = int(ledger_report["frame_count"])
        if trajectory.n_frames != frame_count or not np.array_equal(frame_indices, np.arange(frame_count, dtype=frame_indices.dtype)):
            raise SoftLiftDatasetError(f"{run_id}: trajectory/ledger frame alignment failed")
        if gaps.shape[0] != frame_count or log_weights.shape[0] != frame_count or log_weights.shape[1] != gaps.shape[1] + 1:
            raise SoftLiftDatasetError(f"{run_id}: ledger shapes are inconsistent")
        topology = trajectory.topology
        if atomic_numbers_override is not None:
            atomic_numbers = np.asarray(atomic_numbers_override, dtype=np.int64)
            if atomic_numbers.size != topology.n_atoms:
                raise SoftLiftDatasetError(
                    f"{run_id}: atomic_numbers 长度({atomic_numbers.size})与拓扑原子数"
                    f"({topology.n_atoms})不一致"
                )
        else:
            # ⚠️ mdtraj 对读不出元素的原子给 `element.virtual`（atomic_number = 0），
            # 不抛错。调用方能拿到权威元素时应显式传进来（见 local_residual.autofit）。
            atomic_numbers = np.asarray([int(atom.element.atomic_number) for atom in topology.atoms], dtype=np.int64)
        if reference_atomic_numbers is None:
            reference_atomic_numbers = atomic_numbers
        elif not np.array_equal(reference_atomic_numbers, atomic_numbers):
            raise SoftLiftDatasetError("runs do not share identical topology atomic numbers")
        environment_order = np.asarray(sorted(set(range(topology.n_atoms)) - set(ligand_order.tolist())), dtype=np.int64)
        edge_total = 0
        for frame_index in range(frame_count):
            positions_np = np.asarray(trajectory.xyz[frame_index], dtype=np.float64) * 10.0
            box_np = np.asarray(trajectory.unitcell_vectors[frame_index], dtype=np.float64) * 10.0
            metrics = _box_metrics(box_np, outer_cutoff_angstrom)
            if not metrics["safe_2cutoff_lt_min_face_height"]:
                raise SoftLiftDatasetError(f"{run_id}/frame{frame_index}: box safety domain failed")
            diagonal_box = np.allclose(box_np, np.diag(np.diag(box_np)), rtol=0.0, atol=1.0e-12)
            if device == "cpu" and diagonal_box:
                edge_ligand, edge_environment, displacement, distance = _numpy_cross_edges(
                    positions_np, box_np, ligand_order, environment_order, outer_cutoff_angstrom,
                )
            else:
                ligand_tensor = torch.tensor(ligand_order, dtype=torch.int64, device=device)
                environment_tensor = torch.tensor(environment_order, dtype=torch.int64, device=device)
                positions = torch.tensor(positions_np, dtype=torch.float64, device=device)
                box = torch.tensor(box_np, dtype=torch.float64, device=device)
                from .geometry import ligand_environment_cross_edges

                edges = ligand_environment_cross_edges(
                    positions, box, ligand_tensor, environment_tensor, outer_cutoff=outer_cutoff_angstrom,
                )
                edge_ligand = edges["edge_index"][0].detach().cpu().numpy().astype(np.int64)
                edge_environment = edges["edge_index"][1].detach().cpu().numpy().astype(np.int64)
                displacement = edges["displacement"].detach().cpu().numpy().astype(np.float64)
                distance = edges["distance"].detach().cpu().numpy().astype(np.float64)
            shifts = []
            for lig_index, env_index in zip(edge_ligand.tolist(), edge_environment.tolist()):
                _disp, shift = _unit_shift_and_displacement(positions_np[lig_index], positions_np[env_index], box_np)
                shifts.append(shift)
            shifts_np = np.asarray(shifts, dtype=np.int64).reshape((-1, 3))
            if edge_ligand.size:
                exact_displacements = np.stack([
                    _unit_shift_and_displacement(positions_np[lig_index], positions_np[env_index], box_np)[0]
                    for lig_index, env_index in zip(edge_ligand.tolist(), edge_environment.tolist())
                ])
                if not np.allclose(exact_displacements, displacement, rtol=0.0, atol=1e-8):
                    raise SoftLiftDatasetError(f"{run_id}/frame{frame_index}: MIC displacement mismatch")
            # 配体在深度解耦端是鬼影，环境原子可以压到它身上——那是**真实构型**，
            # 但落在模型支撑域 [min_distance, outer_cutoff) 之外（径向基在 r→0 发散）。
            # 冻结路径对这种帧一律整条数据集作废（默认行为不变）；在线重训时必须能
            # 跳过它们，否则一帧就能让整个 run 起不来。跳了多少帧会记进 report，
            # 由调用方设上限——大面积跳说明采样区间选错了，不是个别构型。
            if (
                skip_unsupported_frames
                and distance.size
                and float(np.min(distance)) < float(min_distance_support_angstrom)
            ):
                skipped_unsupported[run_id] = skipped_unsupported.get(run_id, 0) + 1
                continue
            pair_type = np.stack((atomic_numbers[edge_ligand], atomic_numbers[edge_environment]), axis=1).astype(np.int64) if edge_ligand.size else np.empty((0, 2), dtype=np.int64)
            checked = canonicalize_frame_edges(
                ligand_topology=edge_ligand,
                environment_topology=edge_environment,
                displacement_angstrom=displacement,
                distance_angstrom=distance,
                pair_type=pair_type,
                unit_shift=shifts_np,
                ligand_topology_indices=ligand_order,
                outer_cutoff_angstrom=outer_cutoff_angstrom,
                min_distance_support_angstrom=min_distance_support_angstrom,
                max_environment_atoms=max_environment_atoms,
                max_edges=max_edges,
                max_neighbors_per_ligand=max_neighbors_per_ligand,
            )
            local = checked["edge_ligand_local"]
            all_frames.append({
                "partition_index": partition_index,
                # 跳过支撑域外的帧之后帧号必须重新连续（下游
                # `validate_softlift_dataset_arrays` 要求每条 run 从 0 连续）。
                # 没有跳过任何帧时这个计数器与 frame_index 逐值相同。
                "frame_index_within_run": frame_index - skipped_unsupported.get(run_id, 0),
                "box_vectors_angstrom": box_np,
                "edge_ligand_topology": ligand_order[local],
                **{key: checked[key] for key in ("edge_environment_topology", "edge_ligand_local", "edge_distance_angstrom", "edge_displacement_angstrom", "edge_unit_shift", "edge_pair_type")},
                "adjacent_gap_reduced": gaps[frame_index],
                "log_importance_unnormalized": log_weights[frame_index],
                "edge_count": int(checked["edge_count"]),
            })
            edge_total += int(checked["edge_count"])
        run_reports[run_id] = {
            "frame_count": frame_count,
            "kept_frame_count": frame_count - skipped_unsupported.get(run_id, 0),
            "skipped_unsupported_frames": skipped_unsupported.get(run_id, 0),
            "trajectory_sha256": sha256_file(trajectory_path),
            "ledger_sha256": sha256_file(ledger_path),
            "ledger_report_sha256": sha256_file(ledger_report_path),
            "max_edges": max(int(frame["edge_count"]) for frame in all_frames if frame["partition_index"] == partition_index),
        }

    assert reference_atomic_numbers is not None
    type_vocabulary = np.unique(np.concatenate((reference_atomic_numbers, reference_atomic_numbers[ligand_order]))).astype(np.int64)
    offsets = [0]
    frame_fields: dict[str, list[Any]] = {key: [] for key in (
        "partition_index", "frame_index_within_run", "box_vectors_angstrom", "edge_ligand_topology", "edge_environment_topology",
        "edge_ligand_local", "edge_distance_angstrom", "edge_displacement_angstrom", "edge_unit_shift", "edge_pair_type",
        "adjacent_gap_reduced", "log_importance_unnormalized",
    )}
    for frame in all_frames:
        for key in frame_fields:
            frame_fields[key].append(frame[key])
        offsets.append(offsets[-1] + int(frame["edge_count"]))
    arrays = {
        "partition_index": np.asarray(frame_fields["partition_index"], dtype=np.int64),
        "frame_index_within_run": np.asarray(frame_fields["frame_index_within_run"], dtype=np.int64),
        "box_vectors_angstrom": np.stack(frame_fields["box_vectors_angstrom"]).astype(np.float64),
        "edge_offsets": np.asarray(offsets, dtype=np.int64),
        "edge_ligand_topology": np.concatenate(frame_fields["edge_ligand_topology"]).astype(np.int64),
        "edge_environment_topology": np.concatenate(frame_fields["edge_environment_topology"]).astype(np.int64),
        "edge_ligand_local": np.concatenate(frame_fields["edge_ligand_local"]).astype(np.int64),
        "edge_distance_angstrom": np.concatenate(frame_fields["edge_distance_angstrom"]).astype(np.float64),
        "edge_displacement_angstrom": np.concatenate(frame_fields["edge_displacement_angstrom"]).astype(np.float64).reshape((-1, 3)),
        "edge_unit_shift": np.concatenate(frame_fields["edge_unit_shift"]).astype(np.int64).reshape((-1, 3)),
        "edge_pair_type": np.concatenate(frame_fields["edge_pair_type"]).astype(np.int64).reshape((-1, 2)),
        "adjacent_gap_reduced": np.stack(frame_fields["adjacent_gap_reduced"]).astype(np.float64),
        "log_importance_unnormalized": np.stack(frame_fields["log_importance_unnormalized"]).astype(np.float64),
        "ligand_topology_indices": ligand_order,
        "ligand_atomic_numbers": reference_atomic_numbers[ligand_order],
        "all_topology_atomic_numbers": reference_atomic_numbers,
        "type_vocabulary": type_vocabulary,
        "delta_A": np.asarray([] if delta_A is None else delta_A, dtype=np.float64),
        "A_k_window": np.asarray([] if A_k_window is None else A_k_window, dtype=np.float64),
    }
    stats = validate_softlift_dataset_arrays(
        arrays,
        outer_cutoff_angstrom=outer_cutoff_angstrom,
        min_distance_support_angstrom=min_distance_support_angstrom,
        max_environment_atoms=max_environment_atoms,
        max_edges=max_edges,
        max_neighbors_per_ligand=max_neighbors_per_ligand,
    )
    arrays["edge_offsets"] = arrays["edge_offsets"].astype(np.int64)
    _atomic_write_npz(output_path, arrays)
    report_body = {
        "schema_version": "softlift_dataset_v1",
        "status": "D0_DATASET_COMPLETED_NOT_TRAINED",
        "dataset_path": str(output_path.resolve()),
        "dataset_sha256": sha256_file(output_path),
        "protocol_sha256": protocol_sha256,
        "topology_path": str(topology_path),
        "topology_sha256": sha256_file(topology_path),
        "run_id_by_partition_index": [str(run["run_id"]) for run in runs],
        "inputs": run_reports,
        "n_ligand_atoms": int(ligand_order.size),
        "type_vocabulary": type_vocabulary.tolist(),
        "outer_cutoff_angstrom": outer_cutoff_angstrom,
        "inner_cutoff_angstrom": 4.0,
        "min_distance_support_angstrom": min_distance_support_angstrom,
        "budgets": {
            "unique_environment_atoms_hard": max_environment_atoms,
            "directed_edges_hard": max_edges,
            "neighbors_per_ligand_hard": max_neighbors_per_ligand,
        },
        "statistics": stats,
        "canonical_edge_hash": stats["canonical_edge_hash"],
        "policy": {
            "candidate_semantics": "strict_environment_to_ligand_bipartite",
            "edge_membership_is_discrete": True,
            "c2_envelope_is_not_edge_list_differentiability": True,
            "partial_charge_features": False,
            "no_silent_truncation": True,
            "no_old_output_overwrite": True,
        },
    }
    report = {**report_body, "report_sha256": hashlib.sha256(json.dumps(report_body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()}
    _atomic_write_json(output_path.with_name(output_path.stem + "_report.json"), report)
    return report


__all__ = [
    "SoftLiftDatasetError",
    "build_dataset_v1",
    "canonical_edge_hash",
    "canonicalize_frame_edges",
    "sha256_file",
    "validate_softlift_dataset_arrays",
]
