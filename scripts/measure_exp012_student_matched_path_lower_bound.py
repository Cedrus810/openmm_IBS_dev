#!/usr/bin/env python
"""DEC-049 terminal experiment: matched-path lower-bound cost measurement.

The first profiling attempt (`scripts/profile_exp012_student_torchforce_
overhead_d3.py`) compared a standalone Python-side forward+backward
microbenchmark against a real in-Context TorchForce group evaluation and got
a NEGATIVE "sync overhead" (`stage_c_torchforce_group_eval_median_ms
(1.722) - stage_b_full_forward_and_backward_median_ms (2.882) < 0`) --
physically impossible, because standalone Python dispatch overhead does not
exist in the real C++-invoked TorchForce path, so subtracting numbers from
two different call harnesses is not meaningful. That "sync_overhead" concept
is discarded entirely here, not reused or referenced.

This script instead times FOUR variants end-to-end, each injected as a real
TorchForce into its own force group on a COPY of the real, hash-verified
`hard_window0` win_sys, using the exact same `Simulation.step()` timing
methodology already established by
`scripts/benchmark_exp012_student_vs_no_student_window0_d3.py` (same
platform, same checkpoint, same warmup discipline) -- so every number below
comes from the identical call path, never from subtracting across harnesses:

1. `baseline`      -- no student Force at all.
2. `zero_output`    -- student Force present, but its module returns a
                       differentiable identical zero without touching the
                       trained network at all. Isolates the pure TorchForce/
                       OpenMM bridging cost (Context sync, kernel launch)
                       with zero model computation.
3. `network_only`   -- student Force present, using a FIXED edge/topology
                       set (captured once from the real checkpoint frame via
                       the production dispatch logic) instead of rebuilding
                       the dynamic cell-list/brute-force neighbor graph every
                       call, but still recomputing real per-step distances
                       and running the REAL trained network forward+backward
                       on them. Isolates the network's own unavoidable cost
                       (the part DEC-049 explicitly may NOT touch -- no model
                       changes) from the dynamic graph-construction cost (the
                       part deployment-only optimization CAN still touch).
4. `full`           -- the real, current, unmodified deployment: dynamic
                       cell-list dispatch + real network. This is what D3/D4/
                       the wiring smoke/the WP-5A pilot already measured.

Pre-registered, frozen decision rule (DEC-049, stated BEFORE this script was
run, not fit to its output):

    budget_ms_per_step = (dec049_target_ratio - 1.0) * baseline_median_ms_per_step
    network_only_delta_ms_per_step = network_only_median_ms_per_step - baseline_median_ms_per_step

    if network_only_delta_ms_per_step > budget_ms_per_step:
        DEC-049's 1.10x target is UNREACHABLE for the current model under a
        deployment-implementation-only rescue (the network's own forward+
        backward cost alone already exceeds the whole allowed budget) ->
        `dec049_verdict = "TARGET_UNREACHABLE_CLOSE_ONLINE_PATH"`.
    else:
        Remaining deployment-only headroom exists (graph-construction cost
        can still theoretically be optimized down within budget) ->
        `dec049_verdict = "HEADROOM_REMAINS_CONTINUE_OPTIMIZING"`, with
        `remaining_headroom_ms_per_step = budget - network_only_delta`
        reported as the (very tight) target for any further cell-list/
        bridging optimization work.

This is explicitly the FINAL profiling step per DEC-049 -- the result is
binary and this script is not meant to be extended or re-run with different
methodology afterward; only the two possible verdicts above determine what
(if anything) happens next.
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

_STEPS_PER_UPDATE = 500  # matches production's own steps_per_update, same convention as D3/D4


class LowerBoundMeasurementError(RuntimeError):
    """A checkpoint/variant/timing step failed a fail-closed contract check."""


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="output_lrc_fix")
    parser.add_argument("--stage-type", default="vdw", choices=["vdw"])
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--checkpoint", required=True, help="a direct_gap .pt checkpoint from student_checkpoints/")
    parser.add_argument("--dec049-target-ratio", type=float, default=1.10,
                         help="DEC-049 frozen target: total step cost <= this * baseline")
    parser.add_argument("--repeats", type=int, default=5, help="matches D3/D4 minimum-3 convention, higher for a terminal measurement")
    parser.add_argument("--warmup-chunks", type=int, default=3)
    parser.add_argument("--chunks-per-repeat", type=int, default=4)
    parser.add_argument("--torchscript-output-dir", required=True, help="directory for the 4 exported .pt modules")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite a frozen report: {args.output}")
    torchscript_output_dir = Path(args.torchscript_output_dir)
    if torchscript_output_dir.exists() and any(torchscript_output_dir.iterdir()):
        parser.error(f"--torchscript-output-dir {torchscript_output_dir} already exists and is non-empty")

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
            raise LowerBoundMeasurementError(f"required real production artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_payload = json.loads(stage_protocol_path.read_text(encoding="utf-8"))["protocol_key"]["payload"]
    ibs_state = json.loads(ibs_state_path.read_text(encoding="utf-8"))
    target_temperature_k = float(manifest["temperature_K"])

    system_xml_text = system_xml_path.read_text(encoding="utf-8")
    if _sha256_text(system_xml_text) != stage_payload["system_xml_sha256"]:
        raise LowerBoundMeasurementError("system_native.xml SHA-256 does not match stage2 protocol record")
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
        potential_type=stage_payload["potential_type"], restraint_params=stage_payload["boresch_params"],
        temperature=target_temperature_k * unit.kelvin, prefix=ibs_state["prefix"],
        box_vectors=stale_box_vectors, reference_positions=None,
        dispersion_protocol="legacy_uniform_density_lrc", environment_type="soluble",
    )
    probe_integrator = openmm.LangevinMiddleIntegrator(
        target_temperature_k * unit.kelvin, manifest["friction_per_ps"] / unit.picosecond,
        manifest["step_size_ps"] * unit.picosecond,
    )
    probe_simulation = app.Simulation(topology, probe_win_sys, probe_integrator, platform, platform_properties)
    probe_simulation.loadCheckpoint(str(checkpoint_path))
    box_vectors = probe_simulation.context.getState(getPositions=True).getPeriodicBoxVectors()
    positions_nm_np = np.asarray(
        probe_simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=np.float64,
    )
    del probe_simulation, probe_integrator, probe_win_sys, _probe_ibs

    def _build_win_sys():
        return build_ibs_dual_system(
            base_system, topology, stage_payload["ligand_indices"],
            manifest["lambdas_coul"], manifest["lambdas_vdw"], alchemical_params,
            potential_type=stage_payload["potential_type"], restraint_params=stage_payload["boresch_params"],
            temperature=target_temperature_k * unit.kelvin, prefix=ibs_state["prefix"],
            box_vectors=box_vectors, reference_positions=None,
            dispersion_protocol="legacy_uniform_density_lrc", environment_type="soluble",
        )

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("variant") != "direct_gap":
        raise LowerBoundMeasurementError(f"--checkpoint variant={payload.get('variant')!r}, only direct_gap is qualified")
    model = build_local_residual_student(payload["type_vocabulary"], **payload["model_kwargs"]).to(torch.float64)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    ligand_payload = json.loads(ligand_indices_path.read_text(encoding="utf-8"))
    ligand_topology_indices = sorted(int(index) for index in ligand_payload["ligand_indices"])
    all_topology_atomic_numbers = [int(atom.element.atomic_number) for atom in topology.atoms()]

    # ---- capture a FIXED, legal edge set once, from the real checkpoint frame, via the
    # exact production dispatch logic (not re-derived by hand) ----
    reference_module = build_deployable_student_module(
        model, ligand_topology_indices=ligand_topology_indices,
        all_topology_atomic_numbers=all_topology_atomic_numbers,
        temperature_kelvin=target_temperature_k, a_k=1.0,
    ).to(torch.float64)
    reference_module.eval()
    positions_angstrom = torch.tensor(positions_nm_np * reference_module.nm_to_angstrom, dtype=torch.float64)
    box_angstrom = torch.tensor(
        np.asarray(box_vectors.value_in_unit(unit.nanometer), dtype=np.float64) * reference_module.nm_to_angstrom,
        dtype=torch.float64,
    )
    box_diagonal = torch.stack((box_angstrom[0, 0], box_angstrom[1, 1], box_angstrom[2, 2]))
    n_bins = torch.floor(box_diagonal / reference_module.outer_cutoff_angstrom).to(torch.int64)
    if int(n_bins.min().item()) < 3:
        raise LowerBoundMeasurementError(
            "real production box does not satisfy the cell-list's own >=3-bins-per-axis "
            "correctness condition -- cannot capture a fixed edge set via the cell-list path"
        )
    fixed_edge_ligand, fixed_edge_environment, _fixed_edge_distance = reference_module._cell_list_candidates(
        positions_angstrom, box_angstrom, n_bins
    )
    if fixed_edge_ligand.numel() == 0:
        raise LowerBoundMeasurementError("captured fixed edge set is empty -- cannot time a degenerate network_only variant")
    fixed_edges = (fixed_edge_ligand.tolist(), fixed_edge_environment.tolist())
    print(f"captured fixed edge set: {len(fixed_edges[0])} edges", flush=True)

    ligand_indices_sha256 = _sha256_file(ligand_indices_path)
    torchscript_output_dir.mkdir(parents=True, exist_ok=True)

    def _build_deployable(variant: str):
        if variant == "baseline":
            return None
        if variant == "zero_output":
            return build_deployable_student_module(
                model, ligand_topology_indices=ligand_topology_indices,
                all_topology_atomic_numbers=all_topology_atomic_numbers,
                temperature_kelvin=target_temperature_k, a_k=1.0, zero_output=True,
            ).to(torch.float64)
        if variant == "network_only":
            return build_deployable_student_module(
                model, ligand_topology_indices=ligand_topology_indices,
                all_topology_atomic_numbers=all_topology_atomic_numbers,
                temperature_kelvin=target_temperature_k, a_k=1.0, fixed_edges=fixed_edges,
            ).to(torch.float64)
        if variant == "full":
            return build_deployable_student_module(
                model, ligand_topology_indices=ligand_topology_indices,
                all_topology_atomic_numbers=all_topology_atomic_numbers,
                temperature_kelvin=target_temperature_k, a_k=1.0,
            ).to(torch.float64)
        raise LowerBoundMeasurementError(f"unknown variant {variant!r}")

    def _time_variant(variant: str) -> dict:
        win_sys, _ibs_wrap = _build_win_sys()
        deployable = _build_deployable(variant)
        student_group = None
        if deployable is not None:
            deployable.eval()
            torchscript_path = torchscript_output_dir / f"student_torchscript_{variant}.pt"
            torchscript_sha256 = export_torchscript(deployable, torchscript_path)
            spec = NeuralBasisModelSpec(
                name=f"local_residual_student_hard_window0_lower_bound_{variant}", backend="torchforce",
                model_path=str(torchscript_path.resolve()), sha256=torchscript_sha256,
                energy_offset_kj_mol=0.0, atom_selection="dynamic_funnel_environment",
                atom_indices_path=str(ligand_indices_path.resolve()), atom_indices_sha256=ligand_indices_sha256,
                output_unit="kJ_per_mol", precision="double", periodic=True,
            )
            student_force = build_torchforce_from_spec(spec)
            existing_groups = {int(force.getForceGroup()) for force in win_sys.getForces()}
            student_group = max(existing_groups) + 1 if existing_groups else 0
            if student_group > 31:
                raise LowerBoundMeasurementError("no free OpenMM force group (0-31) left for the student TorchForce")
            student_force.setForceGroup(student_group)
            win_sys.addForce(student_force)
        else:
            torchscript_sha256 = None

        integrator = openmm.LangevinMiddleIntegrator(
            target_temperature_k * unit.kelvin, manifest["friction_per_ps"] / unit.picosecond,
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

        gpu_before = _gpu_memory_mib()
        # OpenMM's Simulation.step()/getState() are synchronous from Python's perspective on
        # every platform including CUDA -- they do not return control until the requested work
        # has completed on-device, so wrapping them in time.perf_counter() already provides a
        # consistent, correct CUDA synchronization boundary; no extra torch.cuda.synchronize()
        # call is needed or would change anything measured here.
        for _ in range(args.warmup_chunks):
            simulation.step(_STEPS_PER_UPDATE)
        ms_per_step_values = []
        for repeat_index in range(args.repeats):
            started = time.perf_counter()
            for _ in range(args.chunks_per_repeat):
                simulation.step(_STEPS_PER_UPDATE)
            elapsed = time.perf_counter() - started
            total_steps = _STEPS_PER_UPDATE * args.chunks_per_repeat
            ms_per_step = 1000.0 * elapsed / total_steps
            ms_per_step_values.append(ms_per_step)
            print(f"  [{variant}] repeat {repeat_index}: {ms_per_step:.4f} ms/step", flush=True)
        gpu_after = _gpu_memory_mib()
        del simulation, integrator

        return {
            "variant": variant,
            "torchscript_sha256": torchscript_sha256,
            "student_force_group": student_group,
            "repeats_ms_per_step": ms_per_step_values,
            "ms_per_step_median": _percentile(ms_per_step_values, 0.5),
            "ms_per_step_p95": _percentile(ms_per_step_values, 0.95),
            "gpu_memory_mib": {"before": gpu_before, "after": gpu_after},
        }

    variants = ["baseline", "zero_output", "network_only", "full"]
    results = {variant: _time_variant(variant) for variant in variants}

    baseline_median = results["baseline"]["ms_per_step_median"]
    budget_ms_per_step = (args.dec049_target_ratio - 1.0) * baseline_median
    network_only_delta = results["network_only"]["ms_per_step_median"] - baseline_median
    zero_output_delta = results["zero_output"]["ms_per_step_median"] - baseline_median
    full_delta = results["full"]["ms_per_step_median"] - baseline_median

    target_unreachable = network_only_delta > budget_ms_per_step
    dec049_verdict = "TARGET_UNREACHABLE_CLOSE_ONLINE_PATH" if target_unreachable else "HEADROOM_REMAINS_CONTINUE_OPTIMIZING"
    remaining_headroom_ms_per_step = None if target_unreachable else (budget_ms_per_step - network_only_delta)

    body = {
        "schema_version": "exp012-student-matched-path-lower-bound-v1",
        "status": "COMPLETED_DEC049_TERMINAL_MEASUREMENT",
        "platform": {"requested": args.platform, "resolved_name": resolved_platform_name,
                     "precision": platform_properties.get("Precision"), "properties": platform_properties},
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_held_out_run_id": payload.get("held_out_run_id"), "checkpoint_seed": payload.get("seed"),
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "fixed_edge_set_size": len(fixed_edges[0]),
        "timing_methodology": {
            "steps_per_chunk": _STEPS_PER_UPDATE, "warmup_chunks_discarded": args.warmup_chunks,
            "chunks_per_repeat": args.chunks_per_repeat, "repeats": args.repeats,
            "note": "each variant is timed end-to-end via Simulation.step() on its own COPY of "
                    "the real win_sys with its own force group -- same call path for all 4 "
                    "variants, no cross-harness subtraction (that is exactly what produced the "
                    "physically-impossible negative 'sync_overhead' in the earlier profiling "
                    "attempt, which this script discards and does not reference)",
        },
        "variants": results,
        "dec049_decision": {
            "target_ratio": args.dec049_target_ratio,
            "baseline_ms_per_step_median": baseline_median,
            "budget_ms_per_step": budget_ms_per_step,
            "network_only_delta_ms_per_step": network_only_delta,
            "zero_output_delta_ms_per_step_context_only": zero_output_delta,
            "full_delta_ms_per_step_context_only": full_delta,
            "verdict": dec049_verdict,
            "remaining_headroom_ms_per_step": remaining_headroom_ms_per_step,
            "note": "the decision is based SOLELY on network_only_delta vs budget, per the "
                    "pre-registered DEC-049 rule; zero_output_delta and full_delta are recorded "
                    "for context/diagnostics only and do not enter the verdict",
        },
        "policy": {
            "decision_reference": "DEC-049 terminal matched-path lower-bound experiment",
            "terminal_experiment": True,
            "note": "this is the final profiling step per DEC-049; the two possible verdicts "
                    "above are the only outcomes this script is designed to produce, and no "
                    "further profiling-methodology iteration is planned after this report",
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    print(f"dec049_verdict={dec049_verdict} baseline_median={baseline_median:.4f} "
          f"budget={budget_ms_per_step:.4f} network_only_delta={network_only_delta:.4f} "
          f"zero_output_delta={zero_output_delta:.4f} full_delta={full_delta:.4f} "
          f"remaining_headroom={remaining_headroom_ms_per_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
