#!/usr/bin/env python
"""DEC-037 D3, sub-item 4 profiling: decompose the student TorchForce overhead.

`benchmark_exp012_student_vs_no_student_window0_d3.py` measured a single
aggregate number (258% ms/step overhead) against the (d0-5) frozen ceiling
(<=50% hard, <=15% target). Per explicit instruction, that result is
registered as a failure of the CURRENT brute-force deployment implementation
(`local_residual/student_deploy.py`'s `_DeployableStudent.forward()`, which
does an O(n_ligand * n_system) all-pairs distance computation every call with
no real neighbor list), not a failure of the frozen student model -- but that
attribution must be measured, not assumed. This script profiles four
DISTINCT stages separately, on the same real production window-0 System,
checkpoint, and trained weights, before any neighbor-discovery code changes:

  (a) graph construction        -- the all-pairs distance + cutoff-mask +
                                    reindex block (`student_deploy.py` lines
                                    mirrored below), pure Python/Torch, no
                                    OpenMM involved, under `torch.no_grad()`.
  (b) model forward/backward    -- the embedding + interaction-block +
                                    readout math, plus the full autograd
                                    backward pass, pure Python/Torch, no
                                    OpenMM involved. Reported both as a
                                    stand-alone full forward+backward call
                                    (includes its own internal graph
                                    construction, since that is inlined in
                                    `_DeployableStudent.forward()` and not
                                    separable at the module boundary) and,
                                    by subtraction against (a), an
                                    approximation of the network-math-only
                                    marginal cost.
  (c) TorchForce synchronization -- calling `simulation.context.getState(
                                    getEnergy=True, getForces=True,
                                    groups={student_force_group})` repeatedly
                                    on the REAL production Context (no
                                    integrator stepping), isolating "OpenMM
                                    asks TorchForce to evaluate this one
                                    force group" from everything else. The
                                    delta against (b)'s pure-Python
                                    forward+backward cost on the same
                                    positions approximates the marshalling/
                                    CUDA-synchronization overhead of the
                                    OpenMM<->LibTorch bridge itself.
  (d) OpenMM step cost           -- the existing no-student baseline
                                    methodology (DEC-039/DEC-041 two-pass
                                    box-vector fix, reused not re-derived):
                                    plain `simulation.step()` cost with the
                                    student Force absent entirely.

A fifth number, the full "with-student" `simulation.step()` cost, is also
remeasured here (same process, same GPU state) purely as a self-consistency
cross-check against the frozen 258% report -- not a replacement for it.

No code in `local_residual/student_deploy.py` is changed by this script.
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


def _summarize_ms(samples_seconds: list[float]) -> dict:
    ms_values = [1000.0 * value for value in samples_seconds]
    return {
        "median": _percentile(ms_values, 0.5), "p95": _percentile(ms_values, 0.95),
        "min": min(ms_values), "max": max(ms_values), "mean": sum(ms_values) / len(ms_values),
        "n_samples": len(ms_values),
    }


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
        results.append(elapsed / total_steps)
        print(f"  repeat {repeat_index}: {1000.0 * elapsed / total_steps:.4f} ms/step", flush=True)
    return results


def _time_calls(callable_fn, *, warmup_calls: int, timed_calls: int, cuda_sync: bool):
    import torch

    for _ in range(warmup_calls):
        callable_fn()
    if cuda_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    samples = []
    for _ in range(timed_calls):
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        callable_fn()
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="output_lrc_fix")
    parser.add_argument("--stage-type", default="vdw", choices=["vdw"])
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--checkpoint", required=True, help="a direct_gap .pt checkpoint from student_checkpoints/")
    parser.add_argument("--a-k", type=float, default=0.5, help="frozen constant for this profiling smoke; not real per-window wiring")
    parser.add_argument("--openmm-repeats", type=int, default=3)
    parser.add_argument("--openmm-warmup-chunks", type=int, default=2)
    parser.add_argument("--openmm-chunks-per-repeat", type=int, default=4)
    parser.add_argument("--microbench-warmup-calls", type=int, default=20)
    parser.add_argument("--microbench-timed-calls", type=int, default=200)
    parser.add_argument("--torchforce-eval-warmup-calls", type=int, default=20)
    parser.add_argument("--torchforce-eval-timed-calls", type=int, default=200)
    parser.add_argument("--torchscript-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.openmm_repeats < 3:
        parser.error("--openmm-repeats must be at least 3")
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
    probe_state = probe_simulation.context.getState(getPositions=True)
    box_vectors = probe_state.getPeriodicBoxVectors()
    real_positions_nm_quantity = probe_state.getPositions(asNumpy=True)
    del probe_simulation, probe_integrator, probe_win_sys, _probe_ibs, probe_state

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

    def _prepare_simulation(win_sys, ibs_wrap):
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
        return simulation, integrator

    # --- (d) OpenMM step cost, no student (fresh remeasurement, same process) ---
    win_sys_baseline, ibs_wrap_baseline = _build_win_sys()
    sim_baseline, integrator_baseline = _prepare_simulation(win_sys_baseline, ibs_wrap_baseline)
    print("[stage d] plain OpenMM step cost, no student force...", flush=True)
    no_student_step_seconds = _time_repeats(
        sim_baseline, warmup_chunks=args.openmm_warmup_chunks,
        repeats=args.openmm_repeats, chunks_per_repeat=args.openmm_chunks_per_repeat,
    )
    del sim_baseline, integrator_baseline

    # --- Build the trained student module (shared by all remaining stages) ---
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

    device = torch.device("cuda" if (args.platform.upper() == "CUDA" and torch.cuda.is_available()) else "cpu")
    deployable_device = deployable.to(device)
    positions_nm_real = torch.tensor(
        np.asarray(real_positions_nm_quantity.value_in_unit(unit.nanometer)), dtype=torch.float64, device=device,
    )
    box_nm_real = torch.tensor(
        np.asarray(box_vectors.value_in_unit(unit.nanometer)), dtype=torch.float64, device=device,
    )

    # --- (a) graph construction only, pure Torch, no OpenMM, no_grad ---
    # Calls `deployable_device`'s ACTUAL candidate-generation methods
    # directly (`_cell_list_candidates` / `_brute_force_candidates`), not a
    # separately hand-copied mirror of them. An earlier version of this
    # script duplicated the (then brute-force-only) forward() logic here
    # instead of calling the real object; after the neighbor-list rewrite
    # that duplicate silently went stale (it kept measuring the old,
    # already-replaced all-pairs code), producing a nonsensical negative
    # "network math only" figure. Reusing the real bound methods on the
    # real `deployable_device` instance is the fix, and also removes a
    # second independent-reimplementation risk of exactly the kind that
    # caused the earlier CPU64-vs-CPU64 `.pow(2)` vs `.square()` bug.
    nm_to_angstrom = deployable_device.nm_to_angstrom
    outer_cutoff_angstrom = deployable_device.outer_cutoff_angstrom

    def _candidate_edges(positions_nm, box_nm):
        positions = positions_nm * nm_to_angstrom
        box = box_nm * nm_to_angstrom
        box_diagonal = torch.stack((box[0, 0], box[1, 1], box[2, 2]))
        off_diagonal = box - torch.diag(box_diagonal)
        is_diagonal_box = bool((off_diagonal.abs().max() < 1e-6).item())
        n_bins = torch.floor(box_diagonal / outer_cutoff_angstrom).to(torch.int64)
        min_bins = int(n_bins.min().item()) if is_diagonal_box else 0
        if is_diagonal_box and min_bins >= 3:
            edge_ligand_topology, edge_environment_topology, edge_distance = (
                deployable_device._cell_list_candidates(positions, box, n_bins)
            )
        else:
            edge_ligand_topology, edge_environment_topology, edge_distance = (
                deployable_device._brute_force_candidates(positions, box)
            )
        return edge_ligand_topology, edge_environment_topology, edge_distance

    print("[stage a] graph construction only (no_grad, no OpenMM)...", flush=True)
    with torch.no_grad():
        n_edges_probe = _candidate_edges(positions_nm_real, box_nm_real)[-1].numel()
        graph_construction_seconds = _time_calls(
            lambda: _candidate_edges(positions_nm_real, box_nm_real),
            warmup_calls=args.microbench_warmup_calls, timed_calls=args.microbench_timed_calls,
            cuda_sync=True,
        )

    # --- (b) model forward/backward, pure Torch, no OpenMM ---
    def _full_forward_and_backward():
        positions = positions_nm_real.clone().detach().requires_grad_(True)
        energy = deployable_device(positions, box_nm_real)
        energy.backward()
        return energy

    print("[stage b] full forward+backward, pure Torch, no OpenMM...", flush=True)
    full_forward_backward_seconds = _time_calls(
        _full_forward_and_backward,
        warmup_calls=args.microbench_warmup_calls, timed_calls=args.microbench_timed_calls,
        cuda_sync=True,
    )

    def _full_forward_only_no_grad():
        with torch.no_grad():
            return deployable_device(positions_nm_real, box_nm_real)

    print("[stage b'] full forward only, no_grad, pure Torch, no OpenMM...", flush=True)
    full_forward_only_seconds = _time_calls(
        _full_forward_only_no_grad,
        warmup_calls=args.microbench_warmup_calls, timed_calls=args.microbench_timed_calls,
        cuda_sync=True,
    )

    # --- (c) TorchForce synchronization, real Context, no integrator stepping ---
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

    sim_student, integrator_student = _prepare_simulation(win_sys_student, ibs_wrap_student)
    student_group_mask = {new_group}

    print("[stage c] TorchForce group evaluation via real Context.getState, no stepping...", flush=True)
    for _ in range(args.torchforce_eval_warmup_calls):
        sim_student.context.getState(getEnergy=True, getForces=True, groups=student_group_mask)
    torchforce_eval_seconds = []
    for _ in range(args.torchforce_eval_timed_calls):
        started = time.perf_counter()
        sim_student.context.getState(getEnergy=True, getForces=True, groups=student_group_mask)
        torchforce_eval_seconds.append(time.perf_counter() - started)

    # --- Cross-check: full "with student" simulation.step() cost, same process ---
    print("[stage d'] plain OpenMM step cost, with student force (cross-check)...", flush=True)
    with_student_step_seconds = _time_repeats(
        sim_student, warmup_chunks=args.openmm_warmup_chunks,
        repeats=args.openmm_repeats, chunks_per_repeat=args.openmm_chunks_per_repeat,
    )
    del sim_student, integrator_student

    graph_construction_ms = _summarize_ms(graph_construction_seconds)
    full_forward_backward_ms = _summarize_ms(full_forward_backward_seconds)
    full_forward_only_ms = _summarize_ms(full_forward_only_seconds)
    torchforce_eval_ms = _summarize_ms(torchforce_eval_seconds)
    no_student_step_ms = _summarize_ms(no_student_step_seconds)
    with_student_step_ms = _summarize_ms(with_student_step_seconds)

    network_math_only_ms_median_approx = full_forward_only_ms["median"] - graph_construction_ms["median"]
    torchforce_sync_overhead_ms_median_approx = torchforce_eval_ms["median"] - full_forward_backward_ms["median"]
    student_step_delta_ms_median = with_student_step_ms["median"] - no_student_step_ms["median"]
    reconciliation_ratio = (
        torchforce_eval_ms["median"] / student_step_delta_ms_median
        if student_step_delta_ms_median > 0 else None
    )

    body = {
        "schema_version": "exp012-student-d3-profiling-v1",
        "status": "COMPLETED_D3_4_PROFILING",
        "device": str(device),
        "n_edges_at_real_frame": int(n_edges_probe),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_held_out_run_id": payload.get("held_out_run_id"),
        "a_k_used": args.a_k,
        "student_force_group": new_group,
        "stage_a_graph_construction_ms_per_call": graph_construction_ms,
        "stage_b_full_forward_and_backward_ms_per_call": full_forward_backward_ms,
        "stage_b_full_forward_only_no_grad_ms_per_call": full_forward_only_ms,
        "stage_b_network_math_only_ms_median_approx": network_math_only_ms_median_approx,
        "stage_c_torchforce_group_eval_ms_per_call": torchforce_eval_ms,
        "stage_c_sync_overhead_ms_median_approx": torchforce_sync_overhead_ms_median_approx,
        "stage_d_no_student_openmm_step_ms_per_step": no_student_step_ms,
        "cross_check_with_student_openmm_step_ms_per_step": with_student_step_ms,
        "cross_check_student_step_delta_ms_median": student_step_delta_ms_median,
        "cross_check_torchforce_eval_vs_step_delta_ratio": reconciliation_ratio,
        "interpretation": {
            "note_a_vs_b": "stage_a is the all-pairs distance + cutoff-mask + reindex block alone "
                           "(no_grad); stage_b_full_forward_only re-includes that same block plus the "
                           "embedding/interaction-block/readout math (still no_grad); the difference "
                           "approximates the network-math-only marginal cost. If stage_a dominates "
                           "stage_b_full_forward_only, all-pairs neighbor discovery is confirmed the "
                           "bottleneck, not the network itself.",
            "note_c": "stage_c calls Context.getState(groups={student_force_group}) with no integrator "
                      "stepping, isolating what OpenMM pays to evaluate just this one TorchForce (data "
                      "marshalling into/out of LibTorch, CUDA synchronization) from everything else in "
                      "the real production System. Comparing it to stage_b_full_forward_and_backward "
                      "(same computation done in plain Python/Torch, no OpenMM) approximates the "
                      "OpenMM<->TorchForce bridge's own synchronization overhead.",
            "note_d": "stage_d is the existing no-student baseline methodology (DEC-039/DEC-041 "
                      "two-pass box-vector fix), remeasured fresh in this same process/GPU state for "
                      "self-consistency; the cross-check with-student step cost is not a replacement "
                      "for the frozen 258%-overhead report, only a same-run sanity check against it.",
        },
        "policy": {
            "decision_reference": "DEC-037 D3, sub-item 4 profiling (pre-fix, per explicit instruction: "
                                   "profile before changing code)",
            "student_deploy_py_modified": False,
            "profiling_reimplements_forward_math_for_measurement_only": True,
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
    print(f"stage_a_graph_construction_median_ms={graph_construction_ms['median']:.5f}")
    print(f"stage_b_full_forward_only_median_ms={full_forward_only_ms['median']:.5f}")
    print(f"stage_b_network_math_only_median_ms_approx={network_math_only_ms_median_approx:.5f}")
    print(f"stage_b_full_forward_and_backward_median_ms={full_forward_backward_ms['median']:.5f}")
    print(f"stage_c_torchforce_group_eval_median_ms={torchforce_eval_ms['median']:.5f}")
    print(f"stage_c_sync_overhead_median_ms_approx={torchforce_sync_overhead_ms_median_approx:.5f}")
    print(f"stage_d_no_student_step_median_ms={no_student_step_ms['median']:.5f}")
    print(f"cross_check_with_student_step_median_ms={with_student_step_ms['median']:.5f}")
    print(f"cross_check_student_step_delta_median_ms={student_step_delta_ms_median:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
