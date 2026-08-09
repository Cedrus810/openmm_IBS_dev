#!/usr/bin/env python
"""ORB-003: matched-path cost probe for the frozen ORB-v3 layer-2 prefix.

This is a measurement-only experiment.  It does not train a scalar readout,
change the frozen ORB model, alter OpenMM production files, or enable MTS.

The four reported components are deliberately kept separate:

* exact full-parent L2 closure and official ORB graph construction;
* the first two ORB GNS blocks (256-dimensional node latent);
* a temporary scalar ``mean(ligand layer-2 latent)`` coordinate backward;
* a real OpenMM TorchForce evaluation on the hash-verified production system
  and checkpoint.

The temporary scalar is only a probe to exercise the same coordinate-gradient
path.  It is not an ORB-004 basis and must not be interpreted scientifically.
The OpenMM result is eligible for the online budget only when measured on the
same CUDA path as the frozen production baseline.  CPU/Reference runs are
diagnostics and cannot be promoted to an online conclusion.
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_json_write(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n"
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


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timed_calls(function, *, warmups: int, repeats: int, synchronize=None) -> dict[str, Any]:
    for _ in range(warmups):
        function()
        if synchronize is not None:
            synchronize()
    samples = []
    for _ in range(repeats):
        if synchronize is not None:
            synchronize()
        started = time.perf_counter()
        function()
        if synchronize is not None:
            synchronize()
        samples.append(time.perf_counter() - started)
    return {
        "samples_seconds": samples,
        "median_ms": None if not samples else 1000.0 * _percentile(samples, 0.5),
        "p95_ms": None if not samples else 1000.0 * _percentile(samples, 0.95),
        "warmups": warmups,
        "repeats": repeats,
    }


def _time_simulation_steps(simulation, *, warmups: int, repeats: int, steps: int) -> dict[str, Any]:
    for _ in range(warmups):
        simulation.step(steps)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        simulation.step(steps)
        samples.append(1000.0 * (time.perf_counter() - started) / steps)
    return {
        "samples_ms_per_step": samples,
        "median_ms_per_step": _percentile(samples, 0.5),
        "p95_ms_per_step": _percentile(samples, 0.95),
        "warmups": warmups,
        "repeats": repeats,
        "steps_per_repeat": steps,
    }


def _read_model_path(default_report: Path) -> str:
    if not default_report.is_file():
        return "auto"
    try:
        payload = json.loads(default_report.read_text(encoding="utf-8"))
        return str(payload["model_provenance"]["checkpoint_path"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return "auto"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="output_lrc_fix")
    parser.add_argument("--stage-type", default="vdw", choices=["vdw"])
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--platform", default="CPU", help="CUDA is required for online-budget eligibility")
    parser.add_argument(
        "--orb-device", default="auto",
        help="ORB compute device: auto, cpu, cuda, or cuda:N; graph membership remains CPU float64",
    )
    parser.add_argument(
        "--fallback-trajectory",
        default="output/outer_lambda_slow_variable_screen/hard_window0_run1/scratch_sample/hard_window_screening.dcd",
        help="real EXP-012 frame used only when the requested OpenMM platform cannot load the CUDA checkpoint",
    )
    parser.add_argument("--fallback-frame-index", type=int, default=0)
    parser.add_argument("--model-name", default="orb-v3-conservative-omol")
    parser.add_argument(
        "--model-path",
        default=None,
        help="frozen ORB checkpoint; defaults to the ORB-001b recorded path when available",
    )
    parser.add_argument("--graph-warmups", type=int, default=1)
    parser.add_argument("--graph-repeats", type=int, default=3)
    parser.add_argument("--compute-warmups", type=int, default=1)
    parser.add_argument("--compute-repeats", type=int, default=3)
    parser.add_argument("--bridge-warmups", type=int, default=1)
    parser.add_argument("--bridge-repeats", type=int, default=3)
    parser.add_argument("--step-warmups", type=int, default=1)
    parser.add_argument("--step-repeats", type=int, default=3)
    parser.add_argument("--steps-per-repeat", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    output_path = Path(args.output).resolve()
    if output_path.exists():
        parser.error(f"refusing to overwrite existing ORB-003 report: {output_path}")
    if min(
        args.graph_warmups, args.graph_repeats, args.compute_warmups,
        args.compute_repeats, args.bridge_warmups, args.bridge_repeats,
        args.step_warmups, args.step_repeats, args.steps_per_repeat,
    ) < 1:
        parser.error("all warmup/repeat/step counts must be positive")

    import numpy as np
    import openmm
    import torch
    from ase import Atoms
    from openmm import XmlSerializer, app, unit

    from ibs_engine import (
        ACESoftcorePotential,
        _build_platform_properties,
        _gpu_memory_mib,
        _system_has_global_parameter,
        build_ibs_dual_system,
    )
    from local_residual.orb_graph import audit_lhop_graphs
    from local_residual.orb_latent import (
        OrbLatentAdapter,
        OrbModelSpec,
        OrbParentConditioningContract,
        _prefix_node_features,
    )
    from outer_lambda_neural_basis import NeuralBasisModelSpec, build_torchforce_from_spec

    output_root = Path(args.output_root).resolve()
    checkpoints = output_root / "checkpoints"
    window_dir = checkpoints / "production_window" / args.stage_type / f"window_{args.window_index}"
    manifest_path = window_dir / "manifest.json"
    checkpoint_path = window_dir / "openmm.chk"
    stage_protocol_path = checkpoints / "stage2_vanishing.json"
    ibs_state_path = checkpoints / f"ibs_state_{args.stage_type}_window_{args.window_index}.json"
    system_xml_path = output_root / "system_native.xml"
    topology_cif_path = output_root / "topology.cif"
    box_vectors_path = output_root / "box_vectors.npy"

    required = (
        manifest_path, checkpoint_path, stage_protocol_path, ibs_state_path,
        system_xml_path, topology_cif_path, box_vectors_path,
    )
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"required production artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_payload = json.loads(stage_protocol_path.read_text(encoding="utf-8"))["protocol_key"]["payload"]
    ibs_state = json.loads(ibs_state_path.read_text(encoding="utf-8"))
    system_xml_text = system_xml_path.read_text(encoding="utf-8")
    system_xml_sha256 = _sha256_text(system_xml_text)
    if system_xml_sha256 != stage_payload["system_xml_sha256"]:
        raise RuntimeError("system_native.xml SHA-256 does not match stage2 protocol")
    base_system = XmlSerializer.deserialize(system_xml_text)
    topology = app.PDBxFile(str(topology_cif_path)).topology
    stale_box_vectors = unit.Quantity(np.load(box_vectors_path), unit.nanometer)
    alchemical_params = ACESoftcorePotential.from_dict(stage_payload["aces_softcore_params"])

    resolved_platform_name, platform_properties = _build_platform_properties(args.platform)
    platform = openmm.Platform.getPlatformByName(resolved_platform_name)
    if args.orb_device == "auto":
        orb_device_name = "cuda" if resolved_platform_name.upper() == "CUDA" else "cpu"
    else:
        orb_device_name = args.orb_device
    if orb_device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"--orb-device={orb_device_name} requested but torch.cuda.is_available() is false"
        )
    orb_device = torch.device(orb_device_name)
    cuda_synchronize = torch.cuda.synchronize if orb_device.type == "cuda" else None
    print(
        f"ORB-003 platform={resolved_platform_name} orb_device={orb_device} "
        f"torch_cuda={torch.cuda.is_available()}", flush=True
    )

    def _build_system(box_vectors):
        return build_ibs_dual_system(
            base_system, topology, stage_payload["ligand_indices"],
            manifest["lambdas_coul"], manifest["lambdas_vdw"], alchemical_params,
            potential_type=stage_payload["potential_type"],
            restraint_params=stage_payload["boresch_params"],
            temperature=float(manifest["temperature_K"]) * unit.kelvin,
            prefix=ibs_state["prefix"], box_vectors=box_vectors,
            reference_positions=None,
            dispersion_protocol="legacy_uniform_density_lrc",
            environment_type="soluble",
        )

    # Match the production benchmark's checkpoint-derived box procedure.
    probe_system, probe_ibs = _build_system(stale_box_vectors)
    probe_integrator = openmm.LangevinMiddleIntegrator(
        float(manifest["temperature_K"]) * unit.kelvin,
        float(manifest["friction_per_ps"]) / unit.picosecond,
        float(manifest["step_size_ps"]) * unit.picosecond,
    )
    probe_simulation = app.Simulation(topology, probe_system, probe_integrator, platform, platform_properties)
    checkpoint_restore = {"status": "NOT_ATTEMPTED", "platform": resolved_platform_name}
    checkpoint_box_vectors = None
    try:
        probe_simulation.loadCheckpoint(str(checkpoint_path))
        state = probe_simulation.context.getState(getPositions=True)
        checkpoint_box_vectors = state.getPeriodicBoxVectors()
        positions_nm = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=np.float64)
        cell_nm = np.asarray(checkpoint_box_vectors.value_in_unit(unit.nanometer), dtype=np.float64)
        checkpoint_restore.update({"status": "COMPLETED"})
    except Exception as exc:
        # OpenMM checkpoints are platform-bound.  Keep the ORB-only part
        # useful on a CPU-only node by falling back to a real registered frame,
        # but never label the OpenMM bridge as measured in this mode.
        checkpoint_restore.update({
            "status": "FAILED_PLATFORM_OR_CHECKPOINT_RESTORE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "checkpoint_path": str(checkpoint_path),
        })
        import mdtraj as md

        fallback_path = Path(args.fallback_trajectory).resolve()
        if not fallback_path.is_file():
            raise RuntimeError(
                f"checkpoint restore failed and fallback trajectory is missing: {fallback_path}"
            ) from exc
        fallback = md.load_frame(
            str(fallback_path), args.fallback_frame_index, top=str(topology_cif_path)
        )
        if fallback.unitcell_vectors is None:
            raise RuntimeError("fallback trajectory has no periodic cell") from exc
        positions_nm = np.asarray(fallback.xyz[0], dtype=np.float64)
        cell_nm = np.asarray(fallback.unitcell_vectors[0], dtype=np.float64)
        checkpoint_restore.update({
            "fallback_trajectory": str(fallback_path),
            "fallback_trajectory_sha256": _sha256_file(fallback_path),
            "fallback_frame_index": int(args.fallback_frame_index),
        })
    del probe_simulation, probe_integrator, probe_system, probe_ibs

    ligand_indices = sorted(int(value) for value in json.loads((output_root / "ligand_indices.json").read_text())["ligand_indices"])
    atomic_numbers = [int(atom.element.atomic_number) for atom in topology.atoms()]
    positions_angstrom = positions_nm * 10.0
    cell_angstrom = cell_nm * 10.0

    print("auditing exact full-parent L2 closure", flush=True)
    closure_audit = audit_lhop_graphs(
        positions_angstrom, cell_angstrom, ligand_indices=ligand_indices,
        cutoff_angstrom=6.0, max_num_neighbors=120, max_layer=2,
    )
    layer2 = closure_audit["layers"][1]
    if layer2["cap_hit"]:
        raise RuntimeError("ORB-003 fail-closed: layer-2 closure hits the 120-neighbor cap")
    topology_indices = np.asarray(layer2["topology_indices"], dtype=np.int64)
    local_index_by_topology = {int(value): index for index, value in enumerate(topology_indices)}
    local_ligand_indices = [local_index_by_topology[index] for index in ligand_indices]
    local_positions_angstrom = positions_angstrom[topology_indices]
    bridge_box_vectors = (
        checkpoint_box_vectors
        if checkpoint_box_vectors is not None
        else unit.Quantity(cell_nm, unit.nanometer)
    )

    # The ORB-001 report contains a machine-specific absolute cache path.
    # Never inherit that path on another GPU node; ``auto`` resolves the
    # current node's cached_path metadata and fails closed if the frozen
    # checkpoint is not present there.
    model_path = args.model_path or "auto"
    adapter = OrbLatentAdapter(OrbModelSpec(
        model_name=args.model_name, model_path=model_path, device=str(orb_device),
        primary_layer=2, edge_method="knn_alchemi", graph_construction_dtype="float64",
        output_dtype="float32", half_supercell=True, wrap=True, compile=False,
    ))
    contract = OrbParentConditioningContract(role="primary")
    contract.validate()

    atoms = Atoms(
        numbers=[atomic_numbers[int(index)] for index in topology_indices],
        positions=local_positions_angstrom,
        cell=cell_angstrom, pbc=True,
    )
    atoms.info["charge"] = 0.0
    atoms.info["spin"] = 1.0

    def _build_official_graph():
        return adapter.atoms_adapter.from_ase_atoms(
            atoms, device="cpu", max_num_neighbors=120, edge_method="knn_alchemi",
            output_dtype=torch.float32, graph_construction_dtype=torch.float64,
            half_supercell=True, wrap=True,
        )

    print("timing official ORB graph construction", flush=True)
    official_graph_timing = _timed_calls(
        _build_official_graph, warmups=args.graph_warmups, repeats=args.graph_repeats,
    )
    batch_cpu = _build_official_graph()
    if int(batch_cpu.n_node.sum()) != int(topology_indices.size):
        raise RuntimeError("official ORB graph node count changed during probe")
    # Freeze graph membership on CPU float64, then move the already-built
    # AtomGraphs payload to the ORB compute device.  This is the ORB-001
    # provenance contract and avoids letting a CUDA neighbor backend redefine
    # the discrete edge set.
    batch = batch_cpu.to(device=orb_device, dtype=torch.float32)

    class _OrbLayer2Scalar(torch.nn.Module):
        """Measurement-only differentiable ORB prefix with a temporary scalar readout."""

        def __init__(
            self, orb_regressor, template_batch, topology_indices_value,
            local_ligand_indices_value,
        ):
            super().__init__()
            # _prefix_node_features intentionally accepts the frozen ORB
            # regressor and then resolves its internal ``.model`` GNS.
            self.orb_regressor = orb_regressor
            self.batch = template_batch
            self.register_buffer(
                "topology_indices", torch.as_tensor(topology_indices_value, dtype=torch.long)
            )
            self.register_buffer(
                "local_ligand_indices",
                torch.as_tensor(local_ligand_indices_value, dtype=torch.long),
            )

        def forward(self, positions_nm, box_vectors_nm):
            # The official adapter receives NumPy float64 coordinates, does
            # PBC wrapping/neighbor construction in float64, and only then
            # casts the AtomGraphs payload to float32.  Preserve that exact
            # precision lineage in this wrapper.
            local_positions = positions_nm.index_select(0, self.topology_indices).to(torch.float64) * 10.0
            cell_angstrom = box_vectors_nm.to(torch.float64) * 10.0
            # Match ForcefieldAtomsAdapter's frozen wrap=True contract.  The
            # operation remains differentiable with respect to coordinates,
            # while avoiding a silent discrepancy between the offline adapter
            # and the prospective OpenMM path for unwrapped PBC coordinates.
            fractional = torch.linalg.solve(cell_angstrom.t(), local_positions.t()).t()
            fractional = fractional.remainder(1.0)
            local_positions = fractional @ cell_angstrom
            self.batch.node_features["positions"] = local_positions.to(torch.float32)
            self.batch.system_features["cell"] = cell_angstrom.unsqueeze(0).to(torch.float32)
            nodes = _prefix_node_features(self.orb_regressor, self.batch, 2)
            return nodes.index_select(0, self.local_ligand_indices).mean()

    module = _OrbLayer2Scalar(
        adapter.model, batch, topology_indices, local_ligand_indices
    ).eval()
    module = module.to(device=orb_device)
    full_positions_tensor = torch.as_tensor(positions_nm, dtype=torch.float32, device=orb_device)
    cell_tensor = torch.as_tensor(cell_nm, dtype=torch.float32, device=orb_device)

    # Check the temporary wrapper against the same official CPU-built graph
    # after it is moved to the compute device.  This is the exact layer-2
    # prefix, not a trained scalar readout.
    with torch.no_grad():
        reference_nodes = _prefix_node_features(adapter.model, batch, 2)
        reference_ligand = reference_nodes.index_select(
            0, torch.as_tensor(local_ligand_indices, dtype=torch.long, device=orb_device)
        )
        expected_scalar = float(reference_ligand.mean().detach().cpu().item())
    with torch.no_grad():
        wrapped_scalar = float(module(full_positions_tensor, cell_tensor).detach().cpu().item())
    scalar_abs_diff = abs(wrapped_scalar - expected_scalar)
    if scalar_abs_diff > 1e-5:
        raise RuntimeError(f"temporary ORB wrapper disagrees with adapter: abs diff={scalar_abs_diff}")

    def _forward_only():
        with torch.no_grad():
            return module(full_positions_tensor, cell_tensor)

    def _scalar_backward():
        coordinates = full_positions_tensor.detach().clone().requires_grad_(True)
        scalar = module(coordinates, cell_tensor)
        gradient = torch.autograd.grad(scalar, coordinates, retain_graph=False, create_graph=False)[0]
        if not bool(torch.isfinite(gradient).all().item()):
            raise RuntimeError("ORB scalar backward returned non-finite coordinate gradients")
        return gradient

    print("timing layer-2 forward and scalar coordinate backward", flush=True)
    forward_timing = _timed_calls(
        _forward_only, warmups=args.compute_warmups, repeats=args.compute_repeats,
        synchronize=cuda_synchronize,
    )
    backward_timing = _timed_calls(
        _scalar_backward, warmups=args.compute_warmups, repeats=args.compute_repeats,
        synchronize=cuda_synchronize,
    )
    gradient = _scalar_backward()
    if cuda_synchronize is not None:
        cuda_synchronize()
    gradient_norm = float(torch.linalg.vector_norm(gradient).detach().cpu().item())

    bridge_report: dict[str, Any] = {
        "status": (
            "NOT_ATTEMPTED_CHECKPOINT_RESTORE_FAILED"
            if checkpoint_restore["status"] != "COMPLETED"
            else "NOT_ATTEMPTED"
        ),
        "platform_requested": args.platform,
        "platform_resolved": resolved_platform_name,
        "checkpoint_restore_status": checkpoint_restore["status"],
    }
    traced_path = output_path.with_suffix(".orb_layer2_scalar.ts")
    if traced_path.exists():
        raise RuntimeError(f"refusing to overwrite TorchScript artifact: {traced_path}")
    try:
        example_positions = full_positions_tensor.detach().clone().requires_grad_(True)
        traced = torch.jit.trace(
            module, (example_positions, cell_tensor), strict=False, check_trace=False,
        )
        traced.save(str(traced_path))
        traced_sha256 = _sha256_file(traced_path)
        bridge_report.update({
            "status": "TRACED",
            "torchscript_path": str(traced_path),
            "torchscript_sha256": traced_sha256,
        })
    except Exception as exc:
        bridge_report.update({
            "status": "TRACING_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })

    if bridge_report["status"] == "TRACED":
        try:
            win_system, ibs_wrap = _build_system(bridge_box_vectors)
            existing_groups = {int(force.getForceGroup()) for force in win_system.getForces()}
            orb_group = max(existing_groups) + 1 if existing_groups else 0
            if orb_group > 31:
                raise RuntimeError("no free OpenMM force group for ORB TorchForce")
            spec = NeuralBasisModelSpec(
                name="orb003_layer2_temporary_scalar_probe",
                backend="torchforce", model_path=str(traced_path), sha256=traced_sha256,
                energy_offset_kj_mol=0.0, atom_selection="orb003_full_parent_bridge_probe",
                atom_indices_path=str(output_root / "ligand_indices.json"),
                atom_indices_sha256=_sha256_file(output_root / "ligand_indices.json"),
                output_unit="kJ_per_mol", precision="single", periodic=True,
                model_name="orb-v3-conservative-omol-layer2-temporary-scalar",
            )
            orb_force = build_torchforce_from_spec(spec)
            orb_force.setForceGroup(orb_group)
            win_system.addForce(orb_force)
            integrator = openmm.LangevinMiddleIntegrator(
                float(manifest["temperature_K"]) * unit.kelvin,
                float(manifest["friction_per_ps"]) / unit.picosecond,
                float(manifest["step_size_ps"]) * unit.picosecond,
            )
            integrator.setConstraintTolerance(1e-3)
            if hasattr(integrator, "setRemoveCMMotion"):
                integrator.setRemoveCMMotion(True)
            simulation = app.Simulation(topology, win_system, integrator, platform, platform_properties)
            if checkpoint_restore["status"] == "COMPLETED":
                simulation.loadCheckpoint(str(checkpoint_path))
                bridge_state_source = "production_cuda_checkpoint"
            else:
                simulation.context.setPositions(unit.Quantity(positions_nm, unit.nanometer))
                simulation.context.setPeriodicBoxVectors(*bridge_box_vectors)
                bridge_state_source = "registered_trajectory_frame_not_production_checkpoint"
            if _system_has_global_parameter(win_system, "lambda_boresch_scale"):
                simulation.context.setParameter("lambda_boresch_scale", float(manifest["lambda_boresch_scale"]))
            if _system_has_global_parameter(win_system, "lambda_shield"):
                simulation.context.setParameter("lambda_shield", float(manifest["lambda_shield"]))
            if hasattr(ibs_wrap, "update_parameters"):
                ibs_wrap.update_parameters(simulation.context, np.asarray(ibs_state["f_k"], dtype=float))
            group_mask = {orb_group}
            print("timing real OpenMM TorchForce group evaluation", flush=True)
            for _ in range(args.bridge_warmups):
                simulation.context.getState(getEnergy=True, getForces=True, groups=group_mask)
            bridge_samples = []
            for _ in range(args.bridge_repeats):
                started = time.perf_counter()
                simulation.context.getState(getEnergy=True, getForces=True, groups=group_mask)
                bridge_samples.append(1000.0 * (time.perf_counter() - started))
            bridge_report.update({
                "status": (
                    "COMPLETED_CHECKPOINT_MATCHED"
                    if checkpoint_restore["status"] == "COMPLETED"
                    else "COMPLETED_DIAGNOSTIC_NOT_CHECKPOINT_MATCHED"
                ),
                "state_source": bridge_state_source,
                "force_group": orb_group,
                "group_eval_samples_ms": bridge_samples,
                "group_eval_median_ms": _percentile(bridge_samples, 0.5),
                "group_eval_p95_ms": _percentile(bridge_samples, 0.95),
                "gpu_memory_mib": _gpu_memory_mib(),
            })
            del simulation, integrator, win_system, ibs_wrap
        except Exception as exc:
            bridge_report.update({
                "status": "BRIDGE_FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    matched_path_report: dict[str, Any] = {"status": "NOT_ATTEMPTED"}
    if (
        bridge_report["status"] == "COMPLETED_CHECKPOINT_MATCHED"
        and checkpoint_restore["status"] == "COMPLETED"
    ):
        try:
            baseline_system, baseline_ibs = _build_system(checkpoint_box_vectors)
            baseline_integrator = openmm.LangevinMiddleIntegrator(
                float(manifest["temperature_K"]) * unit.kelvin,
                float(manifest["friction_per_ps"]) / unit.picosecond,
                float(manifest["step_size_ps"]) * unit.picosecond,
            )
            baseline_integrator.setConstraintTolerance(1e-3)
            baseline_sim = app.Simulation(topology, baseline_system, baseline_integrator, platform, platform_properties)
            baseline_sim.loadCheckpoint(str(checkpoint_path))
            if _system_has_global_parameter(baseline_system, "lambda_boresch_scale"):
                baseline_sim.context.setParameter("lambda_boresch_scale", float(manifest["lambda_boresch_scale"]))
            if _system_has_global_parameter(baseline_system, "lambda_shield"):
                baseline_sim.context.setParameter("lambda_shield", float(manifest["lambda_shield"]))
            if hasattr(baseline_ibs, "update_parameters"):
                baseline_ibs.update_parameters(baseline_sim.context, np.asarray(ibs_state["f_k"], dtype=float))
            baseline_timing = _time_simulation_steps(
                baseline_sim, warmups=args.step_warmups, repeats=args.step_repeats,
                steps=args.steps_per_repeat,
            )
            del baseline_sim, baseline_integrator, baseline_system, baseline_ibs

            with_orb_system, with_orb_ibs = _build_system(checkpoint_box_vectors)
            orb_force = build_torchforce_from_spec(spec)
            orb_force.setForceGroup(orb_group)
            with_orb_system.addForce(orb_force)
            with_orb_integrator = openmm.LangevinMiddleIntegrator(
                float(manifest["temperature_K"]) * unit.kelvin,
                float(manifest["friction_per_ps"]) / unit.picosecond,
                float(manifest["step_size_ps"]) * unit.picosecond,
            )
            with_orb_integrator.setConstraintTolerance(1e-3)
            with_orb_sim = app.Simulation(topology, with_orb_system, with_orb_integrator, platform, platform_properties)
            with_orb_sim.loadCheckpoint(str(checkpoint_path))
            if _system_has_global_parameter(with_orb_system, "lambda_boresch_scale"):
                with_orb_sim.context.setParameter("lambda_boresch_scale", float(manifest["lambda_boresch_scale"]))
            if _system_has_global_parameter(with_orb_system, "lambda_shield"):
                with_orb_sim.context.setParameter("lambda_shield", float(manifest["lambda_shield"]))
            if hasattr(with_orb_ibs, "update_parameters"):
                with_orb_ibs.update_parameters(with_orb_sim.context, np.asarray(ibs_state["f_k"], dtype=float))
            with_orb_timing = _time_simulation_steps(
                with_orb_sim, warmups=args.step_warmups, repeats=args.step_repeats,
                steps=args.steps_per_repeat,
            )
            matched_path_report = {
                "status": "COMPLETED",
                "baseline": baseline_timing,
                "with_temporary_orb_scalar": with_orb_timing,
                "incremental_delta_median_ms_per_step": (
                    with_orb_timing["median_ms_per_step"] - baseline_timing["median_ms_per_step"]
                ),
            }
            del with_orb_sim, with_orb_integrator, with_orb_system, with_orb_ibs
        except Exception as exc:
            matched_path_report = {
                "status": "MATCHED_STEP_FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    online_budget_eligible = bool(
        resolved_platform_name.upper() == "CUDA"
        and orb_device.type == "cuda"
        and torch.cuda.is_available()
        and checkpoint_restore["status"] == "COMPLETED"
    )
    incremental_delta = matched_path_report.get("incremental_delta_median_ms_per_step")
    if not online_budget_eligible:
        verdict = "DEVICE_MISMATCH_NOT_ELIGIBLE"
    elif incremental_delta is None:
        verdict = "ORB003_INCOMPLETE_BRIDGE"
    elif float(incremental_delta) > 0.2:
        verdict = "OFFLINE_TEACHER_ONLY"
    elif float(incremental_delta) > 0.1:
        verdict = "CONDITIONAL_REVIEW_REQUIRED"
    else:
        verdict = "ORB004_ALLOWED_BY_COST_ONLY"

    body: dict[str, Any] = {
        "schema_version": "orb003-cost-probe-v1",
        "status": "COMPLETED_COST_PROBE",
        "verdict": verdict,
        "online_budget_eligible": online_budget_eligible,
        "online_budget_ms_per_step": {"lower": 0.1, "upper": 0.2},
        "frozen_contract": {
            "model_name": args.model_name,
            "layer": 2,
            "latent_dimension": 256,
            "compile": False,
            "total_charge": 0.0,
            "spin_multiplicity": 1.0,
            "conditioning_scope": "parent_full_system",
        },
        "production_provenance": {
            "output_root": str(output_root),
            "system_native_xml": str(system_xml_path),
            "system_native_xml_sha256": system_xml_sha256,
            "topology_cif": str(topology_cif_path),
            "topology_cif_sha256": _sha256_file(topology_cif_path),
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "checkpoint_derived_box_vectors_nm": cell_nm.tolist(),
            "model_path": str(adapter.model_path),
            "model_sha256": adapter.model_sha256,
            "orb_models_version": adapter.provenance["orb_models_version"],
            "orb_models_source_commit": adapter.provenance["orb_models_source_commit"],
            "torch_version": torch.__version__,
            "openmm_version": openmm.__version__,
            "openmm_platform_requested": args.platform,
            "openmm_platform_resolved": resolved_platform_name,
            "openmm_platform_properties": platform_properties,
            "orb_compute_device": str(orb_device),
            "orb_graph_membership_device": "cpu",
            "orb_graph_membership_dtype": "float64",
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_device_count": int(torch.cuda.device_count()),
            "checkpoint_restore": checkpoint_restore,
        },
        "graph": {
            "closure_full_parent_l2": layer2,
            "closure_node_count": int(topology_indices.size),
            "closure_topology_indices_sha256": hashlib.sha256(np.ascontiguousarray(topology_indices).tobytes()).hexdigest(),
            "official_adapter": official_graph_timing,
            "edge_method": "knn_alchemi",
            "graph_construction_dtype": "float64",
            "output_dtype": "float32",
            "half_supercell": True,
            "wrap": True,
            "online_path_note": "official adapter is timed on the frozen frame's exact L2 local closure; dynamic full-parent closure is reported separately and is not hidden in the prefix timing",
        },
        "layer2_forward": forward_timing,
        "scalar_backward": {
            **backward_timing,
            "temporary_readout": "mean(ligand layer-2 node latent), measurement-only",
            "coordinate_gradient_shape": list(gradient.shape),
            "coordinate_gradient_l2": gradient_norm,
        },
        "bridge": bridge_report,
        "matched_production_step": matched_path_report,
        "wrapper_validation": {
            "reference_scalar_source": "same official CPU-float64-built AtomGraphs moved to ORB compute device; direct frozen layer-2 prefix",
            "offline_adapter_scalar": expected_scalar,
            "wrapper_scalar": wrapped_scalar,
            "absolute_difference": scalar_abs_diff,
            "edge_cap_hit": bool(layer2["cap_hit"]),
        },
        "policy": {
            "training_executed": False,
            "checkpoint_changed": False,
            "mts_executed": False,
            "openmm_production_system_modified": False,
            "temporary_torchforce_only": True,
            "orb004_orb005_started": False,
            "cpu_or_reference_not_online_eligible": True,
            "if_cuda_increment_exceeds_0_2_ms": "register OFFLINE_TEACHER_ONLY and stop ORB-004/005",
        },
    }
    body["report_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    _atomic_json_write(output_path, body)
    print(json.dumps({"output": str(output_path), "verdict": verdict, "report_sha256": body["report_sha256"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
