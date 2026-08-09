#!/usr/bin/env python
"""EXP-013 方案②：independent additive student 的 N=1 ESS 信号检查。

方案① Qualification gate 未通过后，按 DEC-056 才允许进入这里。方案②不是把
student 塞回 IBS 的 fused log-sum-exp，而是构造一个独立的、线性相加的
``CustomCVForce``，放在单独的 OpenMM force group 中：

    V_sample = V_base + V_IBS(classical) + V_WCA + c1 * B_student

这里 ``c1 * B_student`` 被登记为额外的 sampling bias。它不进入物理目标态
``u_kn``；它进入 sampled-row 的 ``bias_history``，因此 TMBAR/重加权账本仍然
闭合。这是一个新的 sampling Hamiltonian，不能当作 DEC-048 的 fused 设计的
等价改写。

本入口只做一对 N=1 baseline / additive-student 短程对照：

* 同一个 CUDA production checkpoint-derived State；
* 同一组位置、速度、盒子、全局参数和 Langevin seed；
* classical IBS Group 1 保持原样，student 是额外 group；
* 只用 ``mixture_ess_proxy`` 判断是否存在正向 ESS 信号；
* 不运行 MTS/N>1，不做三重复 promotion，也不写 production checkpoint。

``mixture_ess_proxy`` 的定义沿用 ibs_engine 的既有 ESS_GATE_PROTOCOL_VERSION=3
实现；报告中明确标记它不是字面 pymbar overlap，也不是最终 promotion gate。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Design2N1ESSCheckError(RuntimeError):
    """A required input or fail-closed signal-check contract was violated."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
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


def _finite_float(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise Design2N1ESSCheckError(f"{name} is not finite: {value!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="output_lrc_fix")
    parser.add_argument("--stage-type", default="vdw", choices=["vdw"])
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--checkpoint", required=True,
                        help="frozen direct_gap checkpoint, used for provenance")
    parser.add_argument("--torchscript", required=True,
                        help="qualified a_k=1.0 TorchScript from the design-1 run")
    parser.add_argument("--coefficient", type=float, default=0.5,
                        help="frozen c1; do not retune after seeing this result")
    parser.add_argument("--max-abs-coefficient", type=float, default=1.0)
    parser.add_argument("--max-abs-additive-energy-kj-mol", type=float, default=25.0)
    parser.add_argument("--max-additive-force-norm-kj-mol-nm", type=float, default=500.0)
    parser.add_argument("--burn-in-steps", type=int, default=10_000,
                        help="discarded from ESS statistics but included in wall time")
    parser.add_argument("--production-steps", type=int, default=50_000,
                        help="monitored N=1 steps")
    parser.add_argument("--steps-per-chunk", type=int, default=500,
                        help="sample interval; matches production steps_per_update")
    parser.add_argument("--langevin-seed", type=int, default=56002)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.platform != "CUDA":
        parser.error("方案② N=1 ESS check 只允许 CUDA；不要用 CPU 替代 CUDA checkpoint")
    if args.burn_in_steps < 0 or args.production_steps < 1 or args.steps_per_chunk < 1:
        parser.error("burn-in/production/steps-per-chunk 必须为合法正整数")
    if args.production_steps % args.steps_per_chunk:
        parser.error("production_steps 必须能被 steps_per_chunk 整除")
    if abs(float(args.coefficient)) > float(args.max_abs_coefficient):
        parser.error("coefficient exceeds max-abs-coefficient")
    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite: {args.output}")
    for path_arg in (args.checkpoint, args.torchscript):
        if not Path(path_arg).is_file():
            raise Design2N1ESSCheckError(f"required frozen artifact is missing: {path_arg}")

    import numpy as np
    import openmm
    from openmm import XmlSerializer, app, unit

    from ibs_engine import (  # noqa: E402
        ACESoftcorePotential,
        IBSSampler,
        _build_platform_properties,
        _ibs_reweighting_quality_diagnostics,
        _system_has_global_parameter,
        build_ibs_dual_system,
    )
    from outer_lambda_neural_basis import (  # noqa: E402
        NeuralBasisModelSpec,
        build_torchforce_from_spec,
    )

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
    required_paths = (
        manifest_path, checkpoint_path, stage_protocol_path, ibs_state_path,
        system_xml_path, topology_cif_path, box_vectors_path, ligand_indices_path,
    )
    for path in required_paths:
        if not path.is_file():
            raise Design2N1ESSCheckError(f"required real production artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_protocol = json.loads(stage_protocol_path.read_text(encoding="utf-8"))
    stage_payload = stage_protocol["protocol_key"]["payload"]
    ibs_state = json.loads(ibs_state_path.read_text(encoding="utf-8"))
    target_temperature_k = float(manifest["temperature_K"])
    inner_dt_ps = float(manifest["step_size_ps"])
    lambdas_vdw = [float(value) for value in manifest["lambdas_vdw"]]
    lambdas_coul = [float(value) for value in manifest["lambdas_coul"]]
    n_states = len(lambdas_vdw)
    prefix = ibs_state["prefix"]
    f_k = np.asarray(ibs_state["f_k"], dtype=np.float64)
    if f_k.shape != (n_states,) or not np.all(np.isfinite(f_k)):
        raise Design2N1ESSCheckError("ibs_state f_k shape/finiteness does not match manifest K")
    kt_kj_per_mol = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
        unit.kilojoule_per_mole / unit.kelvin
    ) * target_temperature_k

    system_xml_text = system_xml_path.read_text(encoding="utf-8")
    if _sha256_text(system_xml_text) != stage_payload["system_xml_sha256"]:
        raise Design2N1ESSCheckError("system_native.xml SHA-256 does not match stage2 protocol")
    base_system = XmlSerializer.deserialize(system_xml_text)
    topology = app.PDBxFile(str(topology_cif_path)).topology
    stale_box_vectors = unit.Quantity(np.load(box_vectors_path), unit.nanometer)
    alchemical_params = ACESoftcorePotential.from_dict(stage_payload["aces_softcore_params"])
    resolved_platform_name, platform_properties = _build_platform_properties(args.platform)
    platform = openmm.Platform.getPlatformByName(resolved_platform_name)

    # The only loadCheckpoint() in this script: recover a public State using the
    # same ordinary Langevin integrator family as the production checkpoint.
    probe_system, probe_ibs = build_ibs_dual_system(
        base_system, topology, stage_payload["ligand_indices"],
        lambdas_coul, lambdas_vdw, alchemical_params,
        potential_type=stage_payload["potential_type"],
        restraint_params=stage_payload["boresch_params"],
        temperature=target_temperature_k * unit.kelvin,
        prefix=prefix, box_vectors=stale_box_vectors, reference_positions=None,
        dispersion_protocol="legacy_uniform_density_lrc", environment_type="soluble",
    )
    probe_integrator = openmm.LangevinMiddleIntegrator(
        target_temperature_k * unit.kelvin,
        float(manifest["friction_per_ps"]) / unit.picosecond,
        inner_dt_ps * unit.picosecond,
    )
    probe_simulation = app.Simulation(
        topology, probe_system, probe_integrator, platform, platform_properties,
    )
    probe_simulation.loadCheckpoint(str(checkpoint_path))
    probe_state = probe_simulation.context.getState(
        getPositions=True, getVelocities=True, getParameters=True,
    )
    source_positions = probe_state.getPositions()
    source_velocities = probe_state.getVelocities()
    box_vectors = probe_state.getPeriodicBoxVectors()
    source_parameters = dict(probe_state.getParameters())
    del probe_simulation, probe_integrator, probe_system, probe_ibs, probe_state

    ligand_indices_sha256 = _sha256_file(ligand_indices_path)
    basis_spec = NeuralBasisModelSpec(
        name="local_residual_student_hard_window0_exp013_design2_n1",
        backend="torchforce",
        model_path=str(Path(args.torchscript).resolve()),
        sha256=_sha256_file(args.torchscript),
        energy_offset_kj_mol=0.0,
        atom_selection="dynamic_funnel_environment",
        atom_indices_path=str(ligand_indices_path.resolve()),
        atom_indices_sha256=ligand_indices_sha256,
        output_unit="kJ_per_mol",
        precision="double",
        periodic=True,
    )
    additive_parameter = "exp013_design2_additive_scale"

    def _build_classical_system():
        return build_ibs_dual_system(
            base_system, topology, stage_payload["ligand_indices"],
            lambdas_coul, lambdas_vdw, alchemical_params,
            potential_type=stage_payload["potential_type"],
            restraint_params=stage_payload["boresch_params"],
            temperature=target_temperature_k * unit.kelvin,
            prefix=prefix, box_vectors=box_vectors, reference_positions=None,
            dispersion_protocol="legacy_uniform_density_lrc", environment_type="soluble",
        )

    def _build_additive_system():
        win_system, ibs_wrapper = _build_classical_system()
        existing_groups = {int(force.getForceGroup()) for force in win_system.getForces()}
        student_group = max(existing_groups) + 1 if existing_groups else 0
        if student_group > 31:
            raise Design2N1ESSCheckError("no free OpenMM force group (0-31) for additive student")
        student_force = build_torchforce_from_spec(basis_spec)
        additive = openmm.CustomCVForce(
            f"{additive_parameter} * student_basis"
        )
        additive.addGlobalParameter(additive_parameter, float(args.coefficient))
        additive.addCollectiveVariable("student_basis", student_force)
        additive.setForceGroup(student_group)
        win_system.addForce(additive)
        return win_system, ibs_wrapper, student_group

    def _make_simulation(win_system):
        integrator = openmm.LangevinMiddleIntegrator(
            target_temperature_k * unit.kelvin,
            float(manifest["friction_per_ps"]) / unit.picosecond,
            inner_dt_ps * unit.picosecond,
        )
        integrator.setConstraintTolerance(1e-3)
        if hasattr(integrator, "setRemoveCMMotion"):
            integrator.setRemoveCMMotion(True)
        simulation = app.Simulation(
            topology, win_system, integrator, platform, platform_properties,
        )
        # Both arms receive exactly the same public State. No second checkpoint
        # load is allowed; this keeps the comparison paired and fail-closed.
        simulation.context.setPositions(source_positions)
        simulation.context.setVelocities(source_velocities)
        simulation.context.setPeriodicBoxVectors(*box_vectors)
        for parameter_name, parameter_value in source_parameters.items():
            if _system_has_global_parameter(win_system, parameter_name):
                simulation.context.setParameter(parameter_name, parameter_value)
        for parameter_name in ("lambda_boresch_scale", "lambda_shield"):
            if _system_has_global_parameter(win_system, parameter_name) and parameter_name in manifest:
                simulation.context.setParameter(parameter_name, float(manifest[parameter_name]))
        integrator.setRandomNumberSeed(int(args.langevin_seed))
        return simulation, integrator

    def _degrees_of_freedom(system):
        dof = 0
        for particle_index in range(system.getNumParticles()):
            if system.getParticleMass(particle_index).value_in_unit(unit.dalton) > 0.0:
                dof += 3
        dof -= system.getNumConstraints()
        if any(isinstance(force, openmm.CMMotionRemover) for force in system.getForces()):
            dof -= 3
        return dof

    def _run_arm(*, student: bool) -> dict:
        if student:
            win_system, ibs_wrapper, student_group = _build_additive_system()
        else:
            win_system, ibs_wrapper = _build_classical_system()
            student_group = None
        simulation, integrator = _make_simulation(win_system)
        ibs_wrapper.update_parameters(simulation.context, f_k)
        if student:
            simulation.context.setParameter(additive_parameter, float(args.coefficient))
        sampler = IBSSampler(
            simulation.context, n_states, target_temperature_k * unit.kelvin,
            prefix=prefix, ibs_wrapper=ibs_wrapper,
        )
        dof = _degrees_of_freedom(win_system)
        gas_constant = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
            unit.kilojoule_per_mole / unit.kelvin
        )
        started = time.perf_counter()
        simulation.step(args.burn_in_steps)
        n_chunks = args.production_steps // args.steps_per_chunk
        frame_records = []
        ledger_closed = True
        all_finite = True
        max_abs_additive_energy = 0.0
        max_additive_force_norm = 0.0
        temperature_values = []
        for _ in range(n_chunks):
            simulation.step(args.steps_per_chunk)
            before_len = len(sampler.energy_history)
            sampler.collect_energies()
            after_len = len(sampler.energy_history)
            frame_closed = (
                after_len == before_len + 1
                and len(sampler.energy_buffer) == after_len
                and len(sampler.bias_history) == after_len
                and len(sampler.base_energy_history) == after_len
            )
            if not frame_closed:
                ledger_closed = False
                raise Design2N1ESSCheckError(
                    f"{'student' if student else 'baseline'} ledger did not append exactly one frame"
                )
            additive_energy = 0.0
            force_norm = 0.0
            if student:
                state_student = simulation.context.getState(
                    getEnergy=True, getForces=True, groups={student_group},
                )
                additive_energy = state_student.getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole
                )
                forces = np.asarray(
                    state_student.getForces(asNumpy=True).value_in_unit(
                        unit.kilojoule_per_mole / unit.nanometer
                    ),
                    dtype=np.float64,
                )
                force_norm = float(np.max(np.linalg.norm(forces, axis=1)))
                # The additive term belongs in the sampled distribution row only.
                sampler.bias_history[-1] += float(additive_energy)
            max_abs_additive_energy = max(max_abs_additive_energy, abs(float(additive_energy)))
            max_additive_force_norm = max(max_additive_force_norm, float(force_norm))
            kinetic_state = simulation.context.getState(getEnergy=True)
            kinetic_energy = kinetic_state.getKineticEnergy().value_in_unit(
                unit.kilojoule_per_mole
            )
            temperature_k = (
                2.0 * float(kinetic_energy) / (dof * gas_constant)
                if dof > 0 else float("nan")
            )
            temperature_values.append(float(temperature_k))
            latest_target = np.asarray(sampler.energy_history[-1], dtype=np.float64)
            latest_cv = np.asarray(sampler.energy_buffer[-1], dtype=np.float64)
            values_finite = bool(
                np.all(np.isfinite(latest_target))
                and np.all(np.isfinite(latest_cv))
                and math.isfinite(float(sampler.bias_history[-1]))
                and math.isfinite(float(sampler.base_energy_history[-1]))
                and math.isfinite(float(additive_energy))
                and math.isfinite(float(force_norm))
                and math.isfinite(float(temperature_k))
            )
            all_finite = all_finite and values_finite
            frame_records.append({
                "additive_energy_kj_mol": float(additive_energy),
                "additive_force_norm_kj_mol_nm": float(force_norm),
                "temperature_k": float(temperature_k),
                "all_finite": values_finite,
            })
        elapsed_seconds = time.perf_counter() - started
        u_kn = np.asarray(sampler.energy_history, dtype=np.float64).T
        bias_kj = np.asarray(sampler.bias_history, dtype=np.float64)
        base_kj = np.asarray(sampler.base_energy_history, dtype=np.float64)
        quality = _ibs_reweighting_quality_diagnostics(u_kn, bias_kj, f_k, kt_kj_per_mol)
        if quality.get("error") is not None:
            raise Design2N1ESSCheckError(
                f"mixture ESS diagnostic failed for {'student' if student else 'baseline'}: {quality['error']}"
            )
        mixture_ess = [float(value) for value in quality["mixture_ess"]]
        mixture_ess_proxy = float(min(mixture_ess))
        safety_passed = bool(
            max_abs_additive_energy <= float(args.max_abs_additive_energy_kj_mol)
            and max_additive_force_norm <= float(args.max_additive_force_norm_kj_mol_nm)
        )
        temperature_health_passed = bool(
            temperature_values
            and all(150.0 <= value <= 600.0 for value in temperature_values)
        )
        del simulation, integrator, win_system
        return {
            "arm": "student_additive" if student else "baseline_classical_ibs",
            "n_value": 1,
            "burn_in_steps": args.burn_in_steps,
            "production_steps": args.production_steps,
            "steps_per_chunk": args.steps_per_chunk,
            "n_frames": int(len(frame_records)),
            "elapsed_seconds": float(elapsed_seconds),
            "gpu_hours": float(elapsed_seconds / 3600.0),
            "ledger_closed": bool(ledger_closed),
            "all_finite": bool(all_finite),
            "safety_passed": safety_passed,
            "temperature_health_passed": temperature_health_passed,
            "mean_temperature_k": float(np.mean(temperature_values)),
            "min_temperature_k": float(np.min(temperature_values)),
            "max_temperature_k": float(np.max(temperature_values)),
            "temperature_sanity_range_k": [150.0, 600.0],
            "student_force_group": student_group,
            "max_abs_additive_energy_kj_mol": float(max_abs_additive_energy),
            "max_additive_force_norm_kj_mol_nm": float(max_additive_force_norm),
            "mixture_ess_per_state": mixture_ess,
            "mixture_ess_proxy": mixture_ess_proxy,
            "mixture_ess_proxy_per_gpu_hour": float(
                mixture_ess_proxy / (elapsed_seconds / 3600.0)
            ),
            "raw_ess_per_state": quality["raw_ess"],
            "mixture_occupancy_normalized": quality.get("mixture_occupancy_normalized"),
            "is_literal_pymbar_overlap": False,
            "is_physical_replica_round_trip": False,
            "frame_records": frame_records,
        }

    print("baseline: N=1", flush=True)
    baseline = _run_arm(student=False)
    print("student_additive: N=1", flush=True)
    student = _run_arm(student=True)

    improvement = float(student["mixture_ess_proxy"] - baseline["mixture_ess_proxy"])
    relative_improvement = (
        improvement / baseline["mixture_ess_proxy"]
        if baseline["mixture_ess_proxy"] > 0.0 else None
    )
    n1_signal_passed = bool(
        baseline["ledger_closed"] and student["ledger_closed"]
        and baseline["all_finite"] and student["all_finite"]
        and baseline["safety_passed"] and student["safety_passed"]
        and baseline["temperature_health_passed"] and student["temperature_health_passed"]
        and improvement > 0.0
    )

    body = {
        "schema_version": "exp013-design2-n1-ess-v1",
        "status": "COMPLETED_DESIGN2_N1_ESS",
        "platform": {
            "requested": args.platform,
            "resolved_name": resolved_platform_name,
            "precision": platform_properties.get("Precision"),
            "properties": platform_properties,
        },
        "window": {
            "stage_type": args.stage_type,
            "window_index": args.window_index,
            "K": n_states,
            "target_temperature_k": target_temperature_k,
            "inner_dt_ps": inner_dt_ps,
            "lambdas_coul": lambdas_coul,
            "lambdas_vdw": lambdas_vdw,
        },
        "frozen_candidate": {
            "checkpoint_path": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": _sha256_file(args.checkpoint),
            "production_checkpoint_path": str(checkpoint_path.resolve()),
            "production_checkpoint_sha256": _sha256_file(checkpoint_path),
            "torchscript_path": str(Path(args.torchscript).resolve()),
            "torchscript_sha256": _sha256_file(args.torchscript),
            "torchscript_a_k_baked_in": 1.0,
            "coefficient_c1": float(args.coefficient),
            "f_k_reused_unmodified": f_k.tolist(),
        },
        "design2_contract": {
            "classical_ibs_group1_unchanged": True,
            "student_is_independent_linear_additive_force": True,
            "student_force_group": "recorded_in_system_per_run",
            "student_is_sampling_bias_not_target_state_energy": True,
            "student_in_sampled_bias_history": True,
            "student_excluded_from_target_u_kn": True,
            "same_public_initial_state_for_both_arms": True,
            "same_langevin_seed_for_both_arms": int(args.langevin_seed),
            "same_integrator_probe_load_checkpoint_calls": 1,
            "n_values_run": [1],
            "n_gt1_mts_run": False,
            "production_checkpoints_written": False,
        },
        "protocol": {
            "burn_in_steps_uncounted_in_ess": args.burn_in_steps,
            "production_steps_monitored": args.production_steps,
            "steps_per_chunk": args.steps_per_chunk,
            "n_frames_per_arm": args.production_steps // args.steps_per_chunk,
            "same_initial_positions_velocities_box_parameters": True,
            "signal_gate": "student mixture_ess_proxy > baseline mixture_ess_proxy, with finite/ledger/safety gates",
            "not_a_three_repeat_promotion_pilot": True,
        },
        "metric_definitions": {
            "mixture_ess_proxy": "min-over-states IBS mixture-coverage ESS from _ibs_reweighting_quality_diagnostics, ESS_GATE_PROTOCOL_VERSION=3; exploratory proxy, not literal pymbar.compute_overlap",
            "student_additive_bias": "c1 * B_student is included in sampled-row bias_history and excluded from target u_kn",
            "physical_replica_round_trip": "not applicable; this is one IBS reference trajectory, not REMD",
        },
        "baseline": baseline,
        "student_additive": student,
        "comparison": {
            "mixture_ess_proxy_improvement": improvement,
            "mixture_ess_proxy_relative_improvement": _raw_or_none(relative_improvement),
            "mixture_ess_proxy_per_gpu_hour_improvement": float(
                student["mixture_ess_proxy_per_gpu_hour"]
                - baseline["mixture_ess_proxy_per_gpu_hour"]
            ),
            "n1_signal_passed": n1_signal_passed,
        },
        "policy": {
            "decision_reference": "DEC-056 / IMPLEMENTATION_PLAN WP-4D scheme (2)",
            "positive_signal_next_action": "only_if_n1_signal_passed_then_design2_MTS_qualification",
            "negative_signal_next_action": "STOP_design2_then_EXP014",
            "do_not_infer": [
                "N=1 signal is not MTS stability qualification",
                "mixture_ess_proxy is not literal BAR/MBAR mutual overlap",
                "one paired N=1 comparison is not an independent-repeat promotion gate",
            ],
        },
    }
    report = {
        **body,
        "report_sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    print(
        f"n1_signal_passed={n1_signal_passed} "
        f"baseline_mixture_ess_proxy={baseline['mixture_ess_proxy']:.6f} "
        f"student_mixture_ess_proxy={student['mixture_ess_proxy']:.6f} "
        f"improvement={improvement:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
