#!/usr/bin/env python
"""DEC-037 D3, sub-item 4: real production Context student vs no-student timing/memory.

Extends the DEC-039/DEC-041-closed no-student baseline methodology
(`scripts/benchmark_exp012_no_student_window0_baseline.py`, whose two-pass
checkpoint-derived-box construction and real-artifact loading is reused
directly here, not re-derived) by adding the student TorchForce to a
**separate force group** on a copy of the same real, hash-verified
`win_sys`, then timing both configurations back-to-back on the exact same
restored production Context state (same checkpoint, same warmup discipline).

This is NOT the full production integration (no CustomCVForce multi-state
`A_k` composition across a window, no wiring into `IBSWindowManagerDualLambda`)
-- it measures the concrete question D3 sub-item 4 asks: what does adding
this one TorchForce cost, in ms/step and GPU memory, on the real System this
project actually runs, relative to the already-established no-student floor.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.student import build_local_residual_student  # noqa: E402
from local_residual.student_deploy import build_deployable_student_module, export_torchscript  # noqa: E402

_STEPS_PER_UPDATE = 500


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
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


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _time_repeats(simulation, *, warmup_chunks: int, repeats: int, chunks_per_repeat: int):
    for _ in range(warmup_chunks):
        simulation.step(_STEPS_PER_UPDATE)
    results = []
    for repeat_index in range(repeats):
        started = time.perf_counter()
        for _ in range(chunks_per_repeat):
            simulation.step(_STEPS_PER_UPDATE)
        elapsed = time.perf_counter() - started
        total_steps = _STEPS_PER_UPDATE * chunks_per_repeat
        ms_per_step = 1000.0 * elapsed / total_steps
        results.append({"repeat_index": repeat_index, "elapsed_seconds": elapsed, "ms_per_step": ms_per_step})
        print(f"  repeat {repeat_index}: {ms_per_step:.4f} ms/step", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="output_lrc_fix")
    parser.add_argument("--stage-type", default="vdw", choices=["vdw"])
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--checkpoint", required=True, help="a direct_gap .pt checkpoint from student_checkpoints/")
    parser.add_argument("--a-k", type=float, default=0.5, help="frozen constant for this timing smoke; not real per-window wiring")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-chunks", type=int, default=2)
    parser.add_argument("--chunks-per-repeat", type=int, default=4)
    parser.add_argument("--torchscript-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite a frozen report: {args.output}")
    if Path(args.torchscript_output).exists():
        parser.error(f"--torchscript-output already exists, refusing to overwrite: {args.torchscript_output}")

    import numpy as np
    import openmm
    import torch
    from openmm import XmlSerializer, app, unit

    from ibs_engine import (  # noqa: E402
        ACESoftcorePotential,
        _build_platform_properties,
        _gpu_memory_mib,
        _system_has_global_parameter,
        build_ibs_dual_system,
    )
    from outer_lambda_neural_basis import NeuralBasisModelSpec, build_torchforce_from_spec  # noqa: E402

    output_root = Path(args.output_root)
    checkpoints = output_root / "checkpoints"
    window_dir = checkpoints / "production_window" / args.stage_type / f"window_{args.window_index}"
    manifest_path = window_dir / "manifest.json"
    checkpoint_path = window_dir / "openmm.chk"
    stage_protocol_path = checkpoints / "stage2_vanishing.json"
    ibs_state_path = checkpoints / f"ibs_state_{args.stage_type}_window_{args.window_index}.json"
    system_xml_path = output_root / "system_native.xml"
    topology_cif_path = output_root / "topology.cif"
    box_vectors_path = output_root / "box_vectors.npy"
    ligand_indices_path = output_root / "ligand_indices.json"

    for path in (manifest_path, checkpoint_path, stage_protocol_path, ibs_state_path,
                 system_xml_path, topology_cif_path, box_vectors_path, ligand_indices_path):
        if not path.is_file():
            raise RuntimeError(f"required real production artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_payload = json.loads(stage_protocol_path.read_text(encoding="utf-8"))["protocol_key"]["payload"]
    ibs_state = json.loads(ibs_state_path.read_text(encoding="utf-8"))

    system_xml_text = system_xml_path.read_text(encoding="utf-8")
    if _sha256_text(system_xml_text) != stage_payload["system_xml_sha256"]:
        raise RuntimeError("system_native.xml SHA-256 does not match stage2 protocol record")
    base_system = XmlSerializer.deserialize(system_xml_text)
    topology = app.PDBxFile(str(topology_cif_path)).topology
    stale_box_vectors = unit.Quantity(np.load(box_vectors_path), unit.nanometer)
    alchemical_params = ACESoftcorePotential.from_dict(stage_payload["aces_softcore_params"])

    resolved_platform_name, platform_properties = _build_platform_properties(args.platform)
    platform = openmm.Platform.getPlatformByName(resolved_platform_name)

    # DEC-039/DEC-041 two-pass box derivation: reuse, not re-derive.
    probe_win_sys, _probe_ibs = build_ibs_dual_system(
        base_system, topology, stage_payload["ligand_indices"],
        manifest["lambdas_coul"], manifest["lambdas_vdw"], alchemical_params,
        potential_type=stage_payload["potential_type"],
        restraint_params=stage_payload["boresch_params"],
        temperature=manifest["temperature_K"] * unit.kelvin,
        prefix=ibs_state["prefix"],
        box_vectors=stale_box_vectors, reference_positions=None,
        dispersion_protocol="legacy_uniform_density_lrc", environment_type="soluble",
    )
    probe_integrator = openmm.LangevinMiddleIntegrator(
        manifest["temperature_K"] * unit.kelvin,
        manifest["friction_per_ps"] / unit.picosecond,
        manifest["step_size_ps"] * unit.picosecond,
    )
    probe_simulation = app.Simulation(topology, probe_win_sys, probe_integrator, platform, platform_properties)
    probe_simulation.loadCheckpoint(str(checkpoint_path))
    box_vectors = probe_simulation.context.getState(getPositions=True).getPeriodicBoxVectors()
    del probe_simulation, probe_integrator, probe_win_sys, _probe_ibs

    def _build_win_sys():
        return build_ibs_dual_system(
            base_system, topology, stage_payload["ligand_indices"],
            manifest["lambdas_coul"], manifest["lambdas_vdw"], alchemical_params,
            potential_type=stage_payload["potential_type"],
            restraint_params=stage_payload["boresch_params"],
            temperature=manifest["temperature_K"] * unit.kelvin,
            prefix=ibs_state["prefix"],
            box_vectors=box_vectors, reference_positions=None,
            dispersion_protocol="legacy_uniform_density_lrc", environment_type="soluble",
        )

    def _run_timing(win_sys, ibs_wrap, label: str, gpu_memory_key: str):
        integrator = openmm.LangevinMiddleIntegrator(
            manifest["temperature_K"] * unit.kelvin,
            manifest["friction_per_ps"] / unit.picosecond,
            manifest["step_size_ps"] * unit.picosecond,
        )
        integrator.setConstraintTolerance(1e-3)
        if hasattr(integrator, "setRemoveCMMotion"):
            integrator.setRemoveCMMotion(True)
        simulation = app.Simulation(topology, win_sys, integrator, platform, platform_properties)
        simulation.loadCheckpoint(str(checkpoint_path))
        if _system_has_global_parameter(win_sys, "lambda_boresch_scale"):
            simulation.context.setParameter("lambda_boresch_scale", float(manifest["lambda_boresch_scale"]))
        if _system_has_global_parameter(win_sys, "lambda_shield"):
            simulation.context.setParameter("lambda_shield", float(manifest["lambda_shield"]))
        ibs_wrap.update_parameters(simulation.context, np.asarray(ibs_state["f_k"], dtype=float))
        gpu_after_context = _gpu_memory_mib()
        print(f"[{label}] warming up...", flush=True)
        repeats = _time_repeats(
            simulation, warmup_chunks=args.warmup_chunks, repeats=args.repeats, chunks_per_repeat=args.chunks_per_repeat,
        )
        gpu_after_repeats = _gpu_memory_mib()
        ms_values = [entry["ms_per_step"] for entry in repeats]
        del simulation, integrator
        return {
            "repeats": repeats,
            "ms_per_step_summary": {
                "median": _percentile(ms_values, 0.5), "p95": _percentile(ms_values, 0.95),
                "min": min(ms_values), "max": max(ms_values), "mean": sum(ms_values) / len(ms_values),
            },
            "gpu_memory_mib": {
                "after_context_and_checkpoint_load": gpu_after_context, "after_timed_repeats": gpu_after_repeats,
            },
        }

    # --- No-student baseline (same win_sys construction, no added Force) ---
    win_sys_baseline, ibs_wrap_baseline = _build_win_sys()
    baseline_result = _run_timing(win_sys_baseline, ibs_wrap_baseline, "no_student", "no_student")

    # --- Student TorchForce added on a new force group ---
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("variant") != "direct_gap":
        raise RuntimeError(f"--checkpoint variant={payload.get('variant')!r}, only direct_gap is a D3 candidate")
    model = build_local_residual_student(payload["type_vocabulary"], **payload["model_kwargs"]).to(torch.float64)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    ligand_payload = json.loads(ligand_indices_path.read_text(encoding="utf-8"))
    ligand_topology_indices = sorted(int(index) for index in ligand_payload["ligand_indices"])
    all_topology_atomic_numbers = [int(atom.element.atomic_number) for atom in topology.atoms()]

    deployable = build_deployable_student_module(
        model, ligand_topology_indices=ligand_topology_indices,
        all_topology_atomic_numbers=all_topology_atomic_numbers,
        temperature_kelvin=manifest["temperature_K"], a_k=args.a_k,
    ).to(torch.float64)
    deployable.eval()
    torchscript_sha256 = export_torchscript(deployable, args.torchscript_output)

    ligand_indices_sha256 = _sha256_file(ligand_indices_path)
    spec = NeuralBasisModelSpec(
        name="local_residual_student_hard_window0",
        backend="torchforce",
        model_path=str(Path(args.torchscript_output).resolve()),
        sha256=torchscript_sha256,
        energy_offset_kj_mol=0.0,
        atom_selection="dynamic_funnel_environment",
        atom_indices_path=str(ligand_indices_path.resolve()),
        atom_indices_sha256=ligand_indices_sha256,
        output_unit="kJ_per_mol",
        precision="double",
        periodic=True,
    )
    student_force = build_torchforce_from_spec(spec)

    win_sys_student, ibs_wrap_student = _build_win_sys()
    existing_groups = {int(force.getForceGroup()) for force in win_sys_student.getForces()}
    new_group = max(existing_groups) + 1 if existing_groups else 0
    if new_group > 31:
        raise RuntimeError("no free OpenMM force group (0-31) left for the student TorchForce")
    student_force.setForceGroup(new_group)
    win_sys_student.addForce(student_force)
    student_result = _run_timing(win_sys_student, ibs_wrap_student, "with_student", "with_student")

    baseline_median = baseline_result["ms_per_step_summary"]["median"]
    student_median = student_result["ms_per_step_summary"]["median"]
    overhead_fraction = (student_median - baseline_median) / baseline_median if baseline_median > 0 else None

    body = {
        "schema_version": "exp012-student-vs-no-student-window0-d3-v1",
        "status": "COMPLETED_D3_4_TIMING",
        "platform": {"requested": args.platform, "resolved_name": resolved_platform_name},
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_held_out_run_id": payload.get("held_out_run_id"),
        "a_k_used": args.a_k,
        "a_k_note": "frozen constant for this D3 timing smoke; not real per-window/per-state A_k wiring "
                    "into the production multi-state IBS Hamiltonian",
        "torchscript_sha256": torchscript_sha256,
        "student_force_group": new_group,
        "no_student": baseline_result,
        "with_student": student_result,
        "student_overhead": {
            "median_ms_per_step_delta": student_median - baseline_median,
            "median_relative_overhead_fraction": overhead_fraction,
            "gpu_memory_delta_after_repeats_mib": (
                (student_result["gpu_memory_mib"]["after_timed_repeats"][0]
                 - baseline_result["gpu_memory_mib"]["after_timed_repeats"][0])
                if student_result["gpu_memory_mib"]["after_timed_repeats"] is not None
                and baseline_result["gpu_memory_mib"]["after_timed_repeats"] is not None
                else None
            ),
        },
        "timing_methodology": {
            "steps_per_chunk": _STEPS_PER_UPDATE,
            "warmup_chunks_discarded": args.warmup_chunks,
            "chunks_per_repeat": args.chunks_per_repeat,
            "repeats": args.repeats,
            "note": "both configurations restored from the same real production checkpoint, same box-vector "
                    "fix (DEC-039/DEC-041), same warmup discipline as the closed no-student baseline",
        },
        "policy": {
            "decision_reference": "DEC-037 D3, sub-item 4",
            "injected_into_real_production_win_sys": True,
            "custom_cv_force_multi_state_wiring": False,
            "training_executed": False,
            "nvt_executed": False,
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    print(
        f"no_student_median={baseline_median:.4f} with_student_median={student_median:.4f} "
        f"overhead_fraction={overhead_fraction}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
