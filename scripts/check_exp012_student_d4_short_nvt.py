#!/usr/bin/env python
"""EXP-012 D4: short NVT dynamical stability qualification for the deployed student TorchForce.

D0-D3 only ever evaluated the student on single, static frames (D1/D2 offline
coordinates, D3 single-point TorchForce/OpenMM consistency and timing). D4 is
the first check that actually *integrates* real dynamics with the student
Force live in the System, and asks a different question than D3: not "is one
evaluation numerically consistent/fast enough" but "does adding this Force to
a real, running Langevin trajectory blow anything up".

Design (frozen here, in the absence of a sealed D4 preregistration -- D1-D3's
sub-items were pre-specified in `protocols/EXP-012_preregistration.json`; D4
never was, per PLAN doc §6 item 6 it was only ever described as "短 NVT、
稳定性，再做独立重复"):

- Reuses the DEC-039/DEC-041 checkpoint-derived-box construction and the
  DEC-037 D3 sub-item 4 pattern of injecting the student TorchForce into a
  COPY of the real, hash-verified `hard_window0` win_sys on its own force
  group (`scripts/benchmark_exp012_student_vs_no_student_window0_d3.py`) --
  not a second, parallel System-construction path.
- "独立重复" = same real production checkpoint (only one exists on disk for
  this window), N different `LangevinMiddleIntegrator` random seeds, so each
  repeat is a genuinely independent stochastic trajectory branching from the
  same starting state -- the same sense of "independent" D1/D2 used for their
  3-seeds-per-fold checks.
- Each repeat is run TWICE with the same integrator seed: once on the
  no-student win_sys, once on the with-student win_sys (student on its own
  force group) -- a paired comparison, so any instability difference is
  attributable to the added Force, not to a different noise realization.
- "短" NVT: `--warmup-chunks` (discarded) + `--chunks-per-repeat` (monitored)
  chunks of `_STEPS_PER_CHUNK` steps each; defaults give 500 warmup + 2000
  monitored steps per repeat per configuration, matching the same order of
  magnitude as the D3 sub-item 4 timing smoke (~7000 steps total), not a
  production-scale run (that is WP-5A's job).
- Diagnostics recorded every chunk (finer than production's own 500-step
  `steps_per_update` reporting granularity): total potential energy, the
  student-only energy contribution (via `getState(groups={...})`, isolating
  just the student's force group), kinetic energy, instantaneous temperature,
  whole-system max per-particle force norm, and the student-only max force
  norm (same group isolation).
- Pass criteria (engineering sanity, NOT a sealed numeric gate -- there is no
  precedent to calibrate one against, so these are honestly labeled
  provisional): every recorded quantity finite in both configurations and all
  repeats; the run completes without an OpenMM exception; the student-only
  max force norm never exceeds `--max-safe-force-norm-kj-mol-nm` (default
  500.0, taken from the PLAN doc §6 example config's
  `safety.max_force_norm_kj_mol_nm` -- an illustrative default in that doc,
  not a calibrated production threshold); instantaneous temperature in both
  configurations stays within `--temperature-sanity-factor` (default 2x) of
  the target temperature (a loose "did not literally explode" bound, not an
  equilibration or thermostat-quality check -- that is WP-5A's job).

Explicitly NOT this script's job: real per-window/per-state A_k wiring into
the production multi-state IBS Hamiltonian (CustomCVForce composition across
a window), ESS/sampling-quality assessment, or independent-repeat production
free-energy comparison -- those are WP-5A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.student import build_local_residual_student  # noqa: E402
from local_residual.student_deploy import build_deployable_student_module, export_torchscript  # noqa: E402

_STEPS_PER_CHUNK = 100  # diagnostic sampling granularity; production's own steps_per_update is 500


class D4NvtStabilityError(RuntimeError):
    """A checkpoint/frame/comparison failed a fail-closed contract check."""


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


def _degrees_of_freedom(system) -> int:
    import openmm
    from openmm import unit

    dof = 0
    for index in range(system.getNumParticles()):
        if system.getParticleMass(index).value_in_unit(unit.dalton) > 0.0:
            dof += 3
    dof -= system.getNumConstraints()
    if any(isinstance(force, openmm.CMMotionRemover) for force in system.getForces()):
        dof -= 3
    return dof


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="output_lrc_fix")
    parser.add_argument("--stage-type", default="vdw", choices=["vdw"])
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--checkpoint", required=True, help="a direct_gap .pt checkpoint from student_checkpoints/")
    parser.add_argument("--a-k", type=float, default=0.5, help="frozen constant for this D4 smoke; not real per-window wiring")
    parser.add_argument("--repeats", type=int, default=3, help="independent integrator-seed repeats, paired no-student/with-student")
    parser.add_argument("--base-integrator-seed", type=int, default=13500)
    parser.add_argument("--warmup-chunks", type=int, default=5, help="chunks of --steps-per-chunk discarded before monitoring")
    parser.add_argument("--chunks-per-repeat", type=int, default=20, help="monitored chunks of --steps-per-chunk per repeat")
    parser.add_argument("--steps-per-chunk", type=int, default=_STEPS_PER_CHUNK)
    parser.add_argument("--max-safe-force-norm-kj-mol-nm", type=float, default=500.0,
                         help="provisional engineering sanity bound on the student-only max per-particle force "
                              "norm (PLAN doc §6 example config default; not a calibrated/sealed threshold)")
    parser.add_argument("--temperature-sanity-factor", type=float, default=2.0,
                         help="instantaneous temperature must stay within this multiple of the target "
                              "temperature in both configurations (loose 'did not explode' bound only)")
    parser.add_argument("--torchscript-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.repeats < 3:
        parser.error("--repeats must be at least 3 (a single trajectory cannot qualify as 'independent repeats')")
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
            raise D4NvtStabilityError(f"required real production artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_payload = json.loads(stage_protocol_path.read_text(encoding="utf-8"))["protocol_key"]["payload"]
    ibs_state = json.loads(ibs_state_path.read_text(encoding="utf-8"))
    target_temperature_k = float(manifest["temperature_K"])

    system_xml_text = system_xml_path.read_text(encoding="utf-8")
    if _sha256_text(system_xml_text) != stage_payload["system_xml_sha256"]:
        raise D4NvtStabilityError("system_native.xml SHA-256 does not match stage2 protocol record")
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
        temperature=target_temperature_k * unit.kelvin,
        prefix=ibs_state["prefix"],
        box_vectors=stale_box_vectors, reference_positions=None,
        dispersion_protocol="legacy_uniform_density_lrc", environment_type="soluble",
    )
    probe_integrator = openmm.LangevinMiddleIntegrator(
        target_temperature_k * unit.kelvin,
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
            temperature=target_temperature_k * unit.kelvin,
            prefix=ibs_state["prefix"],
            box_vectors=box_vectors, reference_positions=None,
            dispersion_protocol="legacy_uniform_density_lrc", environment_type="soluble",
        )

    # --- student TorchForce, exported once and reused across all repeats ---
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("variant") != "direct_gap":
        raise D4NvtStabilityError(f"--checkpoint variant={payload.get('variant')!r}, only direct_gap is a D3/D4 candidate")
    model = build_local_residual_student(payload["type_vocabulary"], **payload["model_kwargs"]).to(torch.float64)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    ligand_payload = json.loads(ligand_indices_path.read_text(encoding="utf-8"))
    ligand_topology_indices = sorted(int(index) for index in ligand_payload["ligand_indices"])
    all_topology_atomic_numbers = [int(atom.element.atomic_number) for atom in topology.atoms()]

    deployable = build_deployable_student_module(
        model, ligand_topology_indices=ligand_topology_indices,
        all_topology_atomic_numbers=all_topology_atomic_numbers,
        temperature_kelvin=target_temperature_k, a_k=args.a_k,
    ).to(torch.float64)
    deployable.eval()
    torchscript_sha256 = export_torchscript(deployable, args.torchscript_output)
    ligand_indices_sha256 = _sha256_file(ligand_indices_path)
    spec = NeuralBasisModelSpec(
        name="local_residual_student_hard_window0_d4",
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

    def _run_nvt(win_sys, ibs_wrap, *, integrator_seed: int, student_group: int | None, label: str):
        integrator = openmm.LangevinMiddleIntegrator(
            target_temperature_k * unit.kelvin,
            manifest["friction_per_ps"] / unit.picosecond,
            manifest["step_size_ps"] * unit.picosecond,
        )
        integrator.setConstraintTolerance(1e-3)
        if hasattr(integrator, "setRemoveCMMotion"):
            integrator.setRemoveCMMotion(True)
        simulation = app.Simulation(topology, win_sys, integrator, platform, platform_properties)
        simulation.loadCheckpoint(str(checkpoint_path))
        # setRandomNumberSeed() AFTER loadCheckpoint(), not before: OpenMM
        # checkpoints are meant to support exact-continuation restarts, so a
        # checkpoint may carry its own saved stochastic-integrator RNG state
        # that loadCheckpoint() would restore -- if that happened, calling
        # setRandomNumberSeed() beforehand would be silently overwritten and
        # every "repeat" would replay the identical bit-for-bit trajectory
        # instead of an independent stochastic branch. Reseeding after load
        # guarantees our per-repeat seed always wins regardless of whether
        # the checkpoint format does or does not embed RNG state.
        integrator.setRandomNumberSeed(int(integrator_seed))
        if _system_has_global_parameter(win_sys, "lambda_boresch_scale"):
            simulation.context.setParameter("lambda_boresch_scale", float(manifest["lambda_boresch_scale"]))
        if _system_has_global_parameter(win_sys, "lambda_shield"):
            simulation.context.setParameter("lambda_shield", float(manifest["lambda_shield"]))
        ibs_wrap.update_parameters(simulation.context, np.asarray(ibs_state["f_k"], dtype=float))

        dof = _degrees_of_freedom(win_sys)
        gas_constant_kj_per_mol_k = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)

        def _snapshot(step_count: int) -> dict:
            state_all = simulation.context.getState(getEnergy=True, getForces=True)
            potential_kj_mol = state_all.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            kinetic_kj_mol = state_all.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            temperature_k = 2.0 * kinetic_kj_mol / (dof * gas_constant_kj_per_mol_k) if dof > 0 else float("nan")
            forces = state_all.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
            force_norms = np.linalg.norm(forces, axis=1)
            entry = {
                "step_count": step_count,
                "potential_energy_kj_mol": float(potential_kj_mol),
                "kinetic_energy_kj_mol": float(kinetic_kj_mol),
                "temperature_k": float(temperature_k),
                "max_force_norm_kj_mol_nm": float(np.max(force_norms)),
                "all_finite": bool(np.isfinite(potential_kj_mol) and np.isfinite(kinetic_kj_mol)
                                   and np.isfinite(temperature_k) and np.all(np.isfinite(forces))),
            }
            if student_group is not None:
                state_student = simulation.context.getState(getEnergy=True, getForces=True, groups={student_group})
                student_energy_kj_mol = state_student.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                student_forces = state_student.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
                student_force_norms = np.linalg.norm(student_forces, axis=1)
                entry["student_energy_kj_mol"] = float(student_energy_kj_mol)
                entry["student_max_force_norm_kj_mol_nm"] = float(np.max(student_force_norms))
                entry["all_finite"] = bool(
                    entry["all_finite"] and np.isfinite(student_energy_kj_mol) and np.all(np.isfinite(student_forces))
                )
            return entry

        gpu_after_context = _gpu_memory_mib()
        snapshots = []
        for _ in range(args.warmup_chunks):
            simulation.step(args.steps_per_chunk)
        snapshots.append(_snapshot(args.warmup_chunks * args.steps_per_chunk))
        for chunk_index in range(args.chunks_per_repeat):
            simulation.step(args.steps_per_chunk)
            total_steps = (args.warmup_chunks + chunk_index + 1) * args.steps_per_chunk
            snapshots.append(_snapshot(total_steps))
            print(f"  [{label} seed={integrator_seed}] step {total_steps}: "
                  f"T={snapshots[-1]['temperature_k']:.1f}K "
                  f"max_force={snapshots[-1]['max_force_norm_kj_mol_nm']:.3e}"
                  + (f" student_max_force={snapshots[-1]['student_max_force_norm_kj_mol_nm']:.3e}"
                     if student_group is not None else ""), flush=True)
        gpu_after_repeat = _gpu_memory_mib()
        del simulation, integrator
        return {
            "snapshots": snapshots,
            "gpu_memory_mib": {"after_context_and_checkpoint_load": gpu_after_context, "after_repeat": gpu_after_repeat},
        }

    repeats_results = []
    for repeat_index in range(args.repeats):
        seed = args.base_integrator_seed + repeat_index

        win_sys_no_student, ibs_wrap_no_student = _build_win_sys()
        print(f"repeat {repeat_index} (seed={seed}): no_student", flush=True)
        no_student_result = _run_nvt(
            win_sys_no_student, ibs_wrap_no_student, integrator_seed=seed, student_group=None, label="no_student",
        )

        win_sys_student, ibs_wrap_student = _build_win_sys()
        existing_groups = {int(force.getForceGroup()) for force in win_sys_student.getForces()}
        student_group = max(existing_groups) + 1 if existing_groups else 0
        if student_group > 31:
            raise D4NvtStabilityError("no free OpenMM force group (0-31) left for the student TorchForce")
        student_force = build_torchforce_from_spec(spec)
        student_force.setForceGroup(student_group)
        win_sys_student.addForce(student_force)
        print(f"repeat {repeat_index} (seed={seed}): with_student (force_group={student_group})", flush=True)
        with_student_result = _run_nvt(
            win_sys_student, ibs_wrap_student, integrator_seed=seed, student_group=student_group, label="with_student",
        )

        repeats_results.append({
            "repeat_index": repeat_index,
            "integrator_seed": seed,
            "no_student": no_student_result,
            "with_student": with_student_result,
            "with_student_force_group": student_group,
        })

    def _all_snapshots(key: str):
        for repeat in repeats_results:
            for snapshot in repeat[key]["snapshots"]:
                yield snapshot

    no_student_snapshots = list(_all_snapshots("no_student"))
    with_student_snapshots = list(_all_snapshots("with_student"))

    all_finite = all(snapshot["all_finite"] for snapshot in no_student_snapshots + with_student_snapshots)
    student_max_force_observed = max(
        (snapshot["student_max_force_norm_kj_mol_nm"] for snapshot in with_student_snapshots), default=0.0,
    )
    student_force_within_safety_bound = student_max_force_observed <= args.max_safe_force_norm_kj_mol_nm

    def _temperature_within_sanity(snapshots) -> bool:
        low = target_temperature_k / args.temperature_sanity_factor
        high = target_temperature_k * args.temperature_sanity_factor
        return all(low <= snapshot["temperature_k"] <= high for snapshot in snapshots)

    temperature_sane = (
        _temperature_within_sanity(no_student_snapshots) and _temperature_within_sanity(with_student_snapshots)
    )

    all_passed = bool(all_finite and student_force_within_safety_bound and temperature_sane)

    body = {
        "schema_version": "exp012-student-d4-short-nvt-stability-v1",
        "status": "COMPLETED_D4_SHORT_NVT_STABILITY",
        "window": {
            "stage_type": args.stage_type, "window_index": args.window_index,
            "K": manifest["K"], "lambdas_vdw": manifest["lambdas_vdw"], "lambdas_coul": manifest["lambdas_coul"],
        },
        "platform": {"requested": args.platform, "resolved_name": resolved_platform_name},
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_held_out_run_id": payload.get("held_out_run_id"),
        "checkpoint_seed": payload.get("seed"),
        "a_k_used": args.a_k,
        "a_k_note": "frozen constant for this D4 smoke; not real per-window/per-state A_k wiring into the "
                    "production multi-state IBS Hamiltonian",
        "torchscript_sha256": torchscript_sha256,
        "target_temperature_k": target_temperature_k,
        "nvt_methodology": {
            "steps_per_chunk": args.steps_per_chunk,
            "warmup_chunks_discarded": args.warmup_chunks,
            "chunks_per_repeat_monitored": args.chunks_per_repeat,
            "total_monitored_steps_per_repeat_per_configuration": args.chunks_per_repeat * args.steps_per_chunk,
            "repeats": args.repeats,
            "base_integrator_seed": args.base_integrator_seed,
            "note": "each repeat pairs a no-student and with-student run from the same real production "
                    "checkpoint using the SAME LangevinMiddleIntegrator random seed, isolating the effect of "
                    "adding the student Force from run-to-run noise-realization differences",
        },
        "safety_thresholds": {
            "max_safe_force_norm_kj_mol_nm": args.max_safe_force_norm_kj_mol_nm,
            "max_safe_force_norm_note": "provisional engineering sanity bound (PLAN doc §6 example config "
                                         "default), not a calibrated or sealed threshold",
            "temperature_sanity_factor": args.temperature_sanity_factor,
        },
        "results": {
            "all_finite": all_finite,
            "student_max_force_norm_observed_kj_mol_nm": student_max_force_observed,
            "student_force_within_safety_bound": student_force_within_safety_bound,
            "temperature_sane_in_both_configurations": temperature_sane,
        },
        "repeats": repeats_results,
        "all_passed": all_passed,
        "policy": {
            "decision_reference": "PLAN doc §6 item 6 (D4), post-DEC-043 D3 closure",
            "injected_into_real_production_win_sys": True,
            "custom_cv_force_multi_state_wiring": False,
            "nvt_executed": True,
            "note": "first check that actually integrates real dynamics with the student Force live in the "
                    "System; production per-window/per-state A_k CustomCVForce wiring and ESS/sampling-quality "
                    "assessment are WP-5A, not this script",
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    print(f"all_passed={all_passed} all_finite={all_finite} "
          f"student_max_force_observed={student_max_force_observed:.3e} "
          f"temperature_sane={temperature_sane}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
