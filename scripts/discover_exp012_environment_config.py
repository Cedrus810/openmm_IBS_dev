#!/usr/bin/env python
"""Discover an exact two-hop EXP-012 environment and emit an explicit config.

This is the missing step between "we have a topology and a ligand" and
"we have an exp012-environment-config-v1 file we can hand to
scripts/build_exp012_environment_manifest.py".  It exists specifically to
avoid repeating the EXP-010 failure mode: EXP-010's 216-atom protein
environment came from a single-frame per-atom radius cut, which sliced through
26 residues without including a single complete one.  Selection here is the
exact N-hop closure on the MACE cutoff graph, not a ligand-centred
``cutoff * layers`` sphere.  Every residue touched by that closure is returned
whole, including water.

This module only discovers *candidates*.  It does not decide the final
cutoff, does not choose the reference frames, and does not seal anything --
those remain explicit choices made (and recorded) by the caller.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.environment import CONFIG_SCHEMA  # noqa: E402
from local_residual.mace_graph import topology_n_hop_closure  # noqa: E402


class Exp012EnvironmentDiscoveryError(ValueError):
    """A discovery input is missing, out of range, or internally inconsistent."""


_WATER_RESIDUE_NAMES = {"HOH", "WAT", "SOL", "TP3", "TIP3"}


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


_MINIMUM_FRAMES_FOR_DEFAULT_PARALLELISM = 8


def _resolve_num_workers(num_workers: int | None, frame_count: int) -> int:
    if num_workers is not None:
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 1:
            raise Exp012EnvironmentDiscoveryError("num_workers must be a positive integer")
        return num_workers
    # A handful of reference frames (the common case in tests and single- or
    # few-frame discovery) isn't worth process-pool startup cost; a real
    # multi-hundred frame union is the case this exists for.
    if frame_count < _MINIMUM_FRAMES_FOR_DEFAULT_PARALLELISM:
        return 1
    return min(_default_num_workers(), frame_count)


def _init_hop_worker() -> None:
    """Pin each worker to one Torch thread to avoid core oversubscription."""
    import torch

    torch.set_num_threads(1)


def _frame_hop_worker(task: tuple) -> Any:
    """Picklable top-level entry point for ProcessPoolExecutor.

    Reproduces exactly the per-frame closure computation the serial path used
    to do inline, so parallel and serial reference-frame reduction agree.
    """
    frame_array, box_array, edge_cutoff_angstrom, ligand_all, interaction_layers = task
    import numpy as np
    import torch

    positions_angstrom = torch.tensor(np.asarray(frame_array) * 10.0, dtype=torch.float64)
    if box_array is None:
        span_angstrom = np.ptp(np.asarray(frame_array) * 10.0, axis=0)
        side = float(np.max(span_angstrom) + 4.0 * float(edge_cutoff_angstrom))
        cell_angstrom = torch.eye(3, dtype=torch.float64) * side
    else:
        cell_angstrom = torch.tensor(np.asarray(box_array) * 10.0, dtype=torch.float64)
    _, frame_hop = topology_n_hop_closure(
        positions_angstrom,
        cell_angstrom,
        ligand_all,
        edge_cutoff_angstrom=float(edge_cutoff_angstrom),
        interaction_layers=interaction_layers,
    )
    return frame_hop.cpu().numpy()


def _sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise Exp012EnvironmentDiscoveryError(f"source file does not exist: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atom_identity(atom: Any) -> dict[str, Any]:
    residue = atom.residue
    chain = residue.chain
    element = getattr(atom, "element", None)
    residue_atoms = list(residue.atoms)
    return {
        "name": str(atom.name),
        "residue_atom_ordinal": residue_atoms.index(atom),
        "element": str(getattr(element, "symbol", "")) if element is not None else None,
        "atomic_number": (
            int(getattr(element, "atomic_number")) if element is not None else None
        ),
        "residue_name": str(residue.name),
        "residue_id": int(getattr(residue, "resSeq", residue.index)),
        "residue_index": int(residue.index),
        "chain_index": int(chain.index),
    }


def discover_complete_residue_environment(
    topology: Any,
    reference_frames_nm: Sequence[Sequence[Sequence[float]]],
    ligand_indices: Sequence[int],
    *,
    box_vectors_nm: Sequence[Sequence[Sequence[float]]] | None = None,
    edge_cutoff_angstrom: float,
    interaction_layers: int,
    num_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Return whole residues touched by the exact cutoff-graph N-hop closure.

    ``reference_frames_nm`` is one or more frames (shape ``[F, N, 3]``); a
    For each frame, ``S0`` is the ligand, ``S1`` is the set of atoms connected
    to ``S0`` by a strict-cutoff edge, and each later layer expands from the
    previous newly reached frontier.  Frame closures are unioned.  A residue
    qualifies if any non-ligand atom in it belongs to that union, after which
    the complete residue is emitted.  The geometric ``cutoff * layers`` value
    is only an upper bound and is never used as a radial selection criterion.
    """

    import numpy as np

    if (
        not isinstance(edge_cutoff_angstrom, (int, float))
        or isinstance(edge_cutoff_angstrom, bool)
        or not math.isfinite(float(edge_cutoff_angstrom))
        or float(edge_cutoff_angstrom) <= 0.0
    ):
        raise Exp012EnvironmentDiscoveryError(
            "edge_cutoff_angstrom must be finite and positive"
        )
    if (
        isinstance(interaction_layers, bool)
        or not isinstance(interaction_layers, int)
        or interaction_layers < 1
    ):
        raise Exp012EnvironmentDiscoveryError("interaction_layers must be a positive integer")

    frames = list(reference_frames_nm)
    if not frames:
        raise Exp012EnvironmentDiscoveryError("reference_frames_nm must contain at least one frame")

    atoms = list(topology.atoms)
    atom_count = len(atoms)
    frame_arrays = [np.asarray(frame, dtype=np.float64) for frame in frames]
    for frame_array in frame_arrays:
        if frame_array.shape != (atom_count, 3):
            raise Exp012EnvironmentDiscoveryError(
                "each reference frame must have shape (n_topology_atoms, 3)"
            )

    if box_vectors_nm is None:
        box_arrays: list[Any] = [None] * len(frame_arrays)
    else:
        box_list = list(box_vectors_nm)
        if len(box_list) != len(frame_arrays):
            raise Exp012EnvironmentDiscoveryError(
                "box_vectors_nm must supply exactly one box per reference frame"
            )
        box_arrays = [np.asarray(box, dtype=np.float64) for box in box_list]

    ligand = {int(index) for index in ligand_indices}
    if not ligand:
        raise Exp012EnvironmentDiscoveryError("ligand_indices must not be empty")
    if min(ligand) < 0 or max(ligand) >= atom_count:
        raise Exp012EnvironmentDiscoveryError("ligand_indices is out of range for this topology")

    ligand_all = sorted(ligand)

    def min_distance_to_ligand(index: int) -> float:
        best = math.inf
        for frame_array, box_array in zip(frame_arrays, box_arrays, strict=True):
            delta = frame_array[ligand_all] - frame_array[index]
            if box_array is not None:
                inverse_box = np.linalg.inv(box_array)
                fractional = delta @ inverse_box
                delta = delta - np.floor(fractional + 0.5) @ box_array
            best = min(best, float(np.min(np.linalg.norm(delta, axis=1))))
        return best

    # Validate box shapes up front so a malformed box fails loudly in the
    # caller's process rather than inside a worker.
    for box_array in box_arrays:
        if box_array is not None and (
            box_array.shape != (3, 3) or not np.isfinite(box_array).all()
        ):
            raise Exp012EnvironmentDiscoveryError(
                "each periodic box must have shape (3, 3) and finite values"
            )

    # Reference frames are independent: each contributes its own per-atom
    # minimum hop, reduced by an elementwise minimum (order-independent, so
    # parallel and serial reduction agree exactly). This is the same
    # Torch/PBC closure implementation the runtime smoke validator uses, to
    # prevent generator/validator semantic drift.
    minimum_hop = np.full(atom_count, interaction_layers + 1, dtype=np.int64)
    tasks = [
        (frame_array, box_array, edge_cutoff_angstrom, ligand_all, interaction_layers)
        for frame_array, box_array in zip(frame_arrays, box_arrays, strict=True)
    ]
    effective_workers = _resolve_num_workers(num_workers, len(tasks))
    if effective_workers <= 1:
        frame_hop_arrays = (_frame_hop_worker(task) for task in tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=effective_workers, initializer=_init_hop_worker)
        frame_hop_arrays = pool.map(_frame_hop_worker, tasks)
    try:
        for frame_hop_array in frame_hop_arrays:
            reached = frame_hop_array >= 0
            minimum_hop[reached] = np.minimum(minimum_hop[reached], frame_hop_array[reached])
    finally:
        if effective_workers > 1:
            pool.shutdown()

    # Chain-terminal position is computed over the polymer only. Water is
    # still eligible as a complete-residue environment candidate below, but
    # it is never classified as a polymer terminus.
    chain_residue_order: dict[int, list[int]] = {}
    for residue in topology.residues:
        if residue.name.upper() in _WATER_RESIDUE_NAMES:
            continue
        chain_residue_order.setdefault(residue.chain.index, []).append(residue.index)
    for chain_index in chain_residue_order:
        chain_residue_order[chain_index].sort()

    candidates: list[dict[str, Any]] = []
    for residue in topology.residues:
        residue_atoms = list(residue.atoms)
        if not residue_atoms:
            continue
        residue_indices = [int(atom.index) for atom in residue_atoms]
        if ligand & set(residue_indices):
            continue
        reached_hops = [int(minimum_hop[index]) for index in residue_indices if minimum_hop[index] <= interaction_layers]
        if not reached_hops:
            continue
        minimum_distance = min(min_distance_to_ligand(index) for index in residue_indices)
        is_water = residue.name.upper() in _WATER_RESIDUE_NAMES
        if is_water:
            is_chain_terminal = False
        else:
            order = chain_residue_order[residue.chain.index]
            position_in_chain = order.index(residue.index)
            is_chain_terminal = position_in_chain in (0, len(order) - 1)
        candidates.append(
            {
                "candidate_type": "complete_residue_environment",
                "atom_indices": sorted(residue_indices),
                "atoms": [_atom_identity(atom) for atom in residue_atoms],
                "reference_min_ligand_distance_nm": minimum_distance,
                "minimum_graph_hop": min(reached_hops),
                "residue_atom_count": len(residue_indices),
                "is_chain_terminal": is_chain_terminal,
                "stable_id": (
                    f"complete_residue:chain{residue.chain.index}:"
                    f"{residue.name}:{getattr(residue, 'resSeq', residue.index)}"
                ),
            }
        )
    candidates.sort(
        key=lambda item: (item["minimum_graph_hop"], item["reference_min_ligand_distance_nm"])
    )
    return candidates


def assemble_environment_config(
    topology: Any,
    ligand_indices: Sequence[int],
    environment_candidates: Sequence[Mapping[str, Any]],
    *,
    sources: Mapping[str, tuple[str, str]],
    metadata_fields: Sequence[str] = ("element", "atomic_number"),
) -> dict[str, Any]:
    """Assemble discovered candidates into an ``exp012-environment-config-v1`` document."""

    atoms = list(topology.atoms)
    atom_count = len(atoms)
    ligand = sorted({int(index) for index in ligand_indices})
    if not ligand:
        raise Exp012EnvironmentDiscoveryError("ligand_indices must not be empty")
    environment = sorted(
        {int(index) for candidate in environment_candidates for index in candidate["atom_indices"]}
    )
    overlap = set(ligand) & set(environment)
    if overlap:
        raise Exp012EnvironmentDiscoveryError(
            f"ligand and environment candidates overlap: {sorted(overlap)}"
        )

    def atom_entry(index: int) -> dict[str, Any]:
        atom = atoms[index]
        identity = _atom_identity(atom)
        entry: dict[str, Any] = {
            "index": index,
            "stable_id": (
                f"chain{identity['chain_index']}:{identity['residue_name']}:"
                f"{identity['residue_id']}:{identity['name']}@"
                f"{identity['residue_atom_ordinal']}"
            ),
        }
        if "atomic_number" in metadata_fields:
            if identity["atomic_number"] is None:
                raise Exp012EnvironmentDiscoveryError(f"atom {index} has no element/atomic_number")
            entry["atomic_number"] = identity["atomic_number"]
        if "element" in metadata_fields:
            if not identity["element"]:
                raise Exp012EnvironmentDiscoveryError(f"atom {index} has no element symbol")
            entry["element"] = identity["element"]
        return entry

    required_sources = {"topology", "base_system", "box"}
    if set(sources) != required_sources:
        raise Exp012EnvironmentDiscoveryError(f"sources must contain exactly {sorted(required_sources)}")

    payload = {
        "sources": {
            name: {"path": path, "sha256": digest} for name, (path, digest) in sources.items()
        },
        "atom_count": atom_count,
        "ligand_indices": ligand,
        "environment_candidate_indices": environment,
        "metadata_fields": list(metadata_fields),
        "atoms": [atom_entry(index) for index in sorted(set(ligand) | set(environment))],
    }
    return {"schema_version": CONFIG_SCHEMA, "payload": payload}


def _relative_to_root(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise Exp012EnvironmentDiscoveryError(
            f"source path must stay inside the workspace: {resolved}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True, help="topology.cif")
    parser.add_argument(
        "--trajectory",
        required=True,
        action="append",
        help="one or more coordinate sources providing reference frames (repeatable)",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        action="append",
        help="frame index within each --trajectory to use as a reference (repeatable; default: last frame of each trajectory)",
    )
    parser.add_argument(
        "--frame-stride-all",
        type=int,
        help=(
            "use every Nth frame (0, N, 2N, ...) of every --trajectory as a reference "
            "frame, instead of one index per --trajectory; mutually exclusive with "
            "--frame-index. This is the exact-coverage mode: when every frame that will "
            "ever be looked up is already known (an offline cache over completed runs, "
            "not live MD), pass --frame-stride-all 1 to union the closure over literally "
            "every recorded frame instead of guessing which single frame generalizes."
        ),
    )
    parser.add_argument("--ligand-indices", required=True, help="JSON file with a ligand_indices array")
    parser.add_argument("--edge-cutoff-angstrom", type=float, required=True)
    parser.add_argument("--interaction-layers", type=int, required=True)
    parser.add_argument("--base-system-xml", required=True, help="System XML used as the sources.base_system hash target")
    parser.add_argument("--box-vectors", required=True, help="box vectors artifact used as the sources.box hash target")
    parser.add_argument(
        "--num-workers", type=int, default=_default_num_workers(),
        help=(
            "reference frames are independent, so the per-frame closure fans out across "
            "processes; default is this process's CPU affinity mask. No GPU is used."
        ),
    )
    parser.add_argument("--report-output", required=True, help="diagnostic discovery report path")
    parser.add_argument("--config-output", required=True, help="exp012-environment-config-v1 output path")
    args = parser.parse_args(argv)

    if args.frame_stride_all is not None and args.frame_index:
        parser.error("--frame-stride-all cannot be combined with --frame-index")
    if args.frame_stride_all is not None and args.frame_stride_all < 1:
        parser.error("--frame-stride-all must be a positive integer")
    if args.num_workers < 1:
        parser.error("--num-workers must be a positive integer")

    try:
        import mdtraj as md
    except ImportError as exc:
        raise Exp012EnvironmentDiscoveryError("this script requires mdtraj (openmm_dev env)") from exc

    topology_path = Path(args.topology).expanduser()
    ligand_payload = json.loads(Path(args.ligand_indices).expanduser().read_text(encoding="utf-8"))
    ligand_indices = ligand_payload.get("ligand_indices")
    if not isinstance(ligand_indices, list) or not ligand_indices:
        raise Exp012EnvironmentDiscoveryError("--ligand-indices JSON must contain a non-empty ligand_indices array")

    reference_frames_nm: list[Any] = []
    box_vectors_nm: list[Any] = []
    trajectory_sha256 = {}
    topology_object = None
    reference_frame_provenance: list[dict[str, Any]] = []

    if args.frame_stride_all is not None:
        for trajectory_arg in args.trajectory:
            trajectory_path = Path(trajectory_arg).expanduser()
            trajectory_sha256[str(trajectory_path)] = _sha256_file(trajectory_path)
            loaded = md.load(str(trajectory_path), top=str(topology_path))
            if topology_object is None:
                topology_object = loaded.topology
            selected_indices = list(range(0, loaded.n_frames, args.frame_stride_all))
            for local_index in selected_indices:
                reference_frames_nm.append(loaded.xyz[local_index])
                box_vectors_nm.append(
                    loaded.unitcell_vectors[local_index] if loaded.unitcell_vectors is not None else None
                )
            reference_frame_provenance.append(
                {
                    "path": str(trajectory_path),
                    "frame_count": loaded.n_frames,
                    "frame_stride": args.frame_stride_all,
                    "reference_frame_count": len(selected_indices),
                }
            )
    else:
        frame_indices = args.frame_index or [None] * len(args.trajectory)
        if len(frame_indices) != len(args.trajectory):
            raise Exp012EnvironmentDiscoveryError(
                "--frame-index must be omitted or repeated exactly once per --trajectory"
            )
        for trajectory_arg, frame_index in zip(args.trajectory, frame_indices, strict=True):
            trajectory_path = Path(trajectory_arg).expanduser()
            trajectory_sha256[str(trajectory_path)] = _sha256_file(trajectory_path)
            with md.open(str(trajectory_path)) as handle:
                frame_count = len(handle)
            resolved_index = frame_count - 1 if frame_index is None else frame_index
            if resolved_index < 0 or resolved_index >= frame_count:
                raise Exp012EnvironmentDiscoveryError(
                    f"frame index {resolved_index} out of range for {trajectory_path} ({frame_count} frames)"
                )
            loaded = md.load_frame(str(trajectory_path), resolved_index, top=str(topology_path))
            if topology_object is None:
                topology_object = loaded.topology
            reference_frames_nm.append(loaded.xyz[0].tolist())
            box_vectors_nm.append(
                loaded.unitcell_vectors[0].tolist() if loaded.unitcell_vectors is not None else None
            )
            reference_frame_provenance.append(
                {"path": str(trajectory_path), "frame_index": resolved_index, "frame_count": frame_count}
            )

    if any(box is None for box in box_vectors_nm) and any(box is not None for box in box_vectors_nm):
        raise Exp012EnvironmentDiscoveryError("all reference frames must consistently declare (or omit) a periodic box")
    if all(box is None for box in box_vectors_nm):
        box_vectors_nm = None

    candidates = discover_complete_residue_environment(
        topology_object,
        reference_frames_nm,
        ligand_indices,
        box_vectors_nm=box_vectors_nm,
        edge_cutoff_angstrom=args.edge_cutoff_angstrom,
        interaction_layers=args.interaction_layers,
        num_workers=args.num_workers,
    )

    report = {
        "ok": True,
        "command": "discover-exp012-environment-config",
        "topology": str(topology_path.resolve()),
        "topology_sha256": _sha256_file(topology_path),
        "trajectories": [
            {"path": path, "sha256": digest} for path, digest in trajectory_sha256.items()
        ],
        "reference_frame_mode": "frame_stride_all" if args.frame_stride_all is not None else "explicit_frame_index",
        "reference_frame_provenance": reference_frame_provenance,
        "reference_frame_count": len(reference_frames_nm),
        "num_workers": args.num_workers,
        "ligand_indices": sorted(int(index) for index in ligand_indices),
        "edge_cutoff_angstrom": float(args.edge_cutoff_angstrom),
        "interaction_layers": int(args.interaction_layers),
        "geometric_upper_bound_angstrom": (
            float(args.edge_cutoff_angstrom) * int(args.interaction_layers)
        ),
        "support_definition": "exact_cutoff_graph_n_hop_closure",
        "candidate_residue_count": len(candidates),
        "candidate_atom_count": sum(candidate["residue_atom_count"] for candidate in candidates),
        "chain_terminal_candidate_count": sum(
            1 for candidate in candidates if candidate["is_chain_terminal"]
        ),
        "candidates": candidates,
    }
    report_output_path = Path(args.report_output).expanduser()
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    config = assemble_environment_config(
        topology_object,
        ligand_indices,
        candidates,
        sources={
            "topology": (_relative_to_root(topology_path), _sha256_file(topology_path)),
            "base_system": (
                _relative_to_root(Path(args.base_system_xml)),
                _sha256_file(args.base_system_xml),
            ),
            "box": (
                _relative_to_root(Path(args.box_vectors)),
                _sha256_file(args.box_vectors),
            ),
        },
    )
    config_output_path = Path(args.config_output).expanduser()
    config_output_path.parent.mkdir(parents=True, exist_ok=True)
    config_output_path.write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"report": args.report_output, "config": args.config_output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
