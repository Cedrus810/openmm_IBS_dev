#!/usr/bin/env python
"""EXP-013 方案①的低成本 MTS 物理预检。

方案①把已经通过 wiring smoke 的 fused ``OuterLambdaIBSBiasForce`` 整体保留在
Group 1，并把整个 Group 1 作为慢组；Group 0/2/3/4/5 是快组。这个脚本只跑
``N=1/2/4/8``，用于决定是否有资格进一步触碰 N=16。它不是 013-B 的三重复，
也不会运行 N=16/32 或修改任何 production artifact。

关键初始化契约（DEC-054/056）：生产 checkpoint 只在同类
``LangevinMiddleIntegrator`` probe Context 中恢复一次；所有 MTS Context 只通过
``setPositions/setVelocities/setPeriodicBoxVectors/setParameter`` 接收公开 State，
绝不跨 Integrator 调用 ``loadCheckpoint()``。
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_N_VALUES = (1, 2, 4, 8)
_MACRO_TICK_INNER_STEPS = 8  # LCM-exact for N in {1, 2, 4, 8}
_EXPECTED_INNER_DT_PS = 0.002  # 2 fs; DEC-056 precheck contract
_SMOKE_WARMUP_MACRO_TICKS = 16
_SMOKE_MONITORED_MACRO_TICKS = 32
_QUALIFICATION_WARMUP_MACRO_TICKS = 400
_QUALIFICATION_MONITORED_MACRO_TICKS = 2000
_QUALIFICATION_BLOCK_TICKS = 50


class Design1PrecheckError(RuntimeError):
    """A required input or fail-closed precheck contract was violated."""


def resolve_phase_lengths(
    phase: str,
    warmup_macro_ticks: int | None = None,
    monitored_macro_ticks: int | None = None,
) -> tuple[int, int]:
    """Return frozen run lengths; qualification lengths cannot be overridden."""

    if phase not in {"smoke", "qualification"}:
        raise ValueError(f"unknown phase: {phase!r}")
    if phase == "qualification":
        if (
            warmup_macro_ticks is not None
            and warmup_macro_ticks != _QUALIFICATION_WARMUP_MACRO_TICKS
        ) or (
            monitored_macro_ticks is not None
            and monitored_macro_ticks != _QUALIFICATION_MONITORED_MACRO_TICKS
        ):
            raise ValueError(
                "qualification 长度已预注册为 warmup=400、monitored=2000 ticks，不能覆盖"
            )
        return _QUALIFICATION_WARMUP_MACRO_TICKS, _QUALIFICATION_MONITORED_MACRO_TICKS
    return (
        _SMOKE_WARMUP_MACRO_TICKS
        if warmup_macro_ticks is None
        else warmup_macro_ticks,
        _SMOKE_MONITORED_MACRO_TICKS
        if monitored_macro_ticks is None
        else monitored_macro_ticks,
    )


def block_mean_and_sem(values: list[float], block_ticks: int) -> tuple[float, float, int]:
    """Compute SEM from contiguous block means, not correlated raw snapshots."""

    if block_ticks < 1:
        raise ValueError("block_ticks must be positive")
    if not values or len(values) % block_ticks:
        raise ValueError("values length must be a positive multiple of block_ticks")
    block_means = [
        sum(values[index:index + block_ticks]) / block_ticks
        for index in range(0, len(values), block_ticks)
    ]
    mean = sum(block_means) / len(block_means)
    if len(block_means) < 2:
        return mean, float("nan"), len(block_means)
    variance = sum((value - mean) ** 2 for value in block_means) / (len(block_means) - 1)
    return mean, math.sqrt(variance / len(block_means)), len(block_means)


def evaluate_precheck_gate(
    phase: str,
    all_finite_by_n: dict[int, bool],
    absolute_health_passed_by_n: dict[int, bool],
    systematic_shift_detected_by_n: dict[int, bool],
) -> dict[str, object]:
    """Apply the phase-specific gate; only qualification can authorize N=16."""

    expected = set(_N_VALUES)
    complete = (
        set(all_finite_by_n) == expected
        and set(absolute_health_passed_by_n) == expected
        and set(systematic_shift_detected_by_n) == set(_N_VALUES[1:])
    )
    all_finite = complete and all(all_finite_by_n.values())
    all_absolute_health_passed = complete and all(absolute_health_passed_by_n.values())
    no_systematic_shift = complete and not any(systematic_shift_detected_by_n.values())
    smoke_passed = bool(all_finite and all_absolute_health_passed)
    qualification_passed = bool(smoke_passed and no_systematic_shift)
    return {
        "phase": phase,
        "all_finite": all_finite,
        "all_absolute_health_passed": all_absolute_health_passed,
        "no_systematic_shift": no_systematic_shift,
        "smoke_passed": smoke_passed,
        "qualification_passed": qualification_passed if phase == "qualification" else False,
        "eligible_for_n16_followup": (
            qualification_passed if phase == "qualification" else False
        ),
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _degrees_of_freedom(system, unit) -> int:
    import openmm

    dof = 0
    for index in range(system.getNumParticles()):
        if system.getParticleMass(index).value_in_unit(unit.dalton) > 0.0:
            dof += 3
    dof -= system.getNumConstraints()
    if any(isinstance(force, openmm.CMMotionRemover) for force in system.getForces()):
        dof -= 3
    return dof


def _mean_and_sem(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, float("nan")
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def _z_score(
    reference: list[float],
    candidate: list[float],
    *,
    block_ticks: int | None = None,
    z_threshold: float = 3.0,
) -> dict:
    if block_ticks is None:
        ref_mean, ref_sem = _mean_and_sem(reference)
        candidate_mean, candidate_sem = _mean_and_sem(candidate)
        n_reference_blocks = n_candidate_blocks = None
        sem_method = "ordinary_snapshot_sem"
    else:
        ref_mean, ref_sem, n_reference_blocks = block_mean_and_sem(reference, block_ticks)
        candidate_mean, candidate_sem, n_candidate_blocks = block_mean_and_sem(
            candidate, block_ticks
        )
        sem_method = "contiguous_block_mean_sem"
    combined_sem = (
        math.sqrt(ref_sem ** 2 + candidate_sem ** 2)
        if math.isfinite(ref_sem) and math.isfinite(candidate_sem)
        else float("nan")
    )
    z_score = (
        abs(candidate_mean - ref_mean) / combined_sem
        if math.isfinite(combined_sem) and combined_sem > 0.0
        else None
    )
    return {
        "reference_mean": ref_mean,
        "reference_sem": ref_sem,
        "candidate_mean": candidate_mean,
        "candidate_sem": candidate_sem,
        "z_score": z_score,
        "sem_method": sem_method,
        "block_ticks": block_ticks,
        "n_reference_blocks": n_reference_blocks,
        "n_candidate_blocks": n_candidate_blocks,
        "systematic_shift_detected": bool(
            z_score is not None and z_score > z_threshold
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="output_lrc_fix")
    parser.add_argument("--stage-type", default="vdw", choices=["vdw"])
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--phase", choices=["smoke", "qualification"], default="smoke")
    parser.add_argument("--checkpoint", required=True, help="frozen direct_gap student checkpoint")
    parser.add_argument("--coefficient", type=float, default=0.5)
    parser.add_argument("--max-abs-coefficient", type=float, default=1.0)
    parser.add_argument("--max-abs-basis-energy-kj-mol", type=float, default=50.0)
    parser.add_argument("--max-abs-path-energy-kj-mol", type=float, default=25.0)
    parser.add_argument("--max-force-norm-kj-mol-nm", type=float, default=500.0)
    parser.add_argument("--warmup-macro-ticks", type=int, default=None)
    parser.add_argument("--monitored-macro-ticks", type=int, default=None)
    parser.add_argument("--systematic-shift-z-threshold", type=float, default=3.0)
    parser.add_argument("--min-mean-temperature-k", type=float, default=270.0)
    parser.add_argument("--max-mean-temperature-k", type=float, default=330.0)
    parser.add_argument("--max-relative-energy-drift", type=float, default=0.10)
    parser.add_argument("--random-seed", type=int, default=56001)
    parser.add_argument("--torchscript-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.platform != "CUDA":
        parser.error(
            "方案① smoke/qualification 只允许目标 CUDA backend；不要用 CPU/Reference "
            "替代 CUDA checkpoint 资格"
        )
    try:
        args.warmup_macro_ticks, args.monitored_macro_ticks = resolve_phase_lengths(
            args.phase, args.warmup_macro_ticks, args.monitored_macro_ticks
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.warmup_macro_ticks < 1 or args.monitored_macro_ticks < 2:
        parser.error("warmup must be >=1 and monitored ticks must be >=2")
    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite: {args.output}")
    if Path(args.torchscript_output).exists():
        parser.error(
            f"--torchscript-output already exists, refusing to overwrite: {args.torchscript_output}"
        )

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
    from outer_lambda_neural_basis import (  # noqa: E402
        NeuralBasisModelSpec,
        NeuralPathSafety,
        OuterLambdaController,
        OuterLambdaIBSBiasForce,
        build_torchforce_from_spec,
    )
    from local_residual.student import build_local_residual_student  # noqa: E402
    from local_residual.student_deploy import (  # noqa: E402
        build_deployable_student_module,
        export_torchscript,
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
            raise Design1PrecheckError(f"required production artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_payload = json.loads(stage_protocol_path.read_text(encoding="utf-8"))["protocol_key"]["payload"]
    ibs_state = json.loads(ibs_state_path.read_text(encoding="utf-8"))
    target_temperature_k = float(manifest["temperature_K"])
    inner_dt_ps = float(manifest["step_size_ps"])
    if not math.isclose(inner_dt_ps, _EXPECTED_INNER_DT_PS, rel_tol=0.0, abs_tol=1e-12):
        raise Design1PrecheckError(
            f"方案①预检只接受当前 inner_dt=2 fs；manifest 是 {inner_dt_ps:g} ps"
        )
    lambdas_vdw = [float(value) for value in manifest["lambdas_vdw"]]
    n_states = len(lambdas_vdw)
    prefix = ibs_state["prefix"]
    f_k = np.asarray(ibs_state["f_k"], dtype=float)
    if f_k.shape != (n_states,):
        raise Design1PrecheckError("ibs_state f_k length does not match manifest K")

    system_xml_text = system_xml_path.read_text(encoding="utf-8")
    if _sha256_text(system_xml_text) != stage_payload["system_xml_sha256"]:
        raise Design1PrecheckError("system_native.xml SHA-256 does not match stage2 protocol record")
    base_system = XmlSerializer.deserialize(system_xml_text)
    topology = app.PDBxFile(str(topology_cif_path)).topology
    stale_box_vectors = unit.Quantity(np.load(box_vectors_path), unit.nanometer)
    alchemical_params = ACESoftcorePotential.from_dict(stage_payload["aces_softcore_params"])
    resolved_platform_name, platform_properties = _build_platform_properties(args.platform)
    platform = openmm.Platform.getPlatformByName(resolved_platform_name)

    # The only loadCheckpoint() in this script: same-integrator source Context.
    probe_win_sys, _probe_ibs = build_ibs_dual_system(
        base_system, topology, stage_payload["ligand_indices"],
        manifest["lambdas_coul"], lambdas_vdw, alchemical_params,
        potential_type=stage_payload["potential_type"],
        restraint_params=stage_payload["boresch_params"],
        temperature=target_temperature_k * unit.kelvin,
        prefix=prefix, box_vectors=stale_box_vectors, reference_positions=None,
        dispersion_protocol="legacy_uniform_density_lrc", environment_type="soluble",
    )
    probe_integrator = openmm.LangevinMiddleIntegrator(
        target_temperature_k * unit.kelvin,
        manifest["friction_per_ps"] / unit.picosecond,
        inner_dt_ps * unit.picosecond,
    )
    probe_simulation = app.Simulation(
        topology, probe_win_sys, probe_integrator, platform, platform_properties
    )
    probe_simulation.loadCheckpoint(str(checkpoint_path))
    probe_state = probe_simulation.context.getState(
        getPositions=True, getVelocities=True, getParameters=True,
    )
    box_vectors = probe_state.getPeriodicBoxVectors()
    source_positions = probe_state.getPositions()
    source_velocities = probe_state.getVelocities()
    source_parameters = dict(probe_state.getParameters())
    del probe_simulation, probe_integrator, probe_win_sys, _probe_ibs, probe_state

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("variant") != "direct_gap":
        raise Design1PrecheckError(
            f"--checkpoint variant={payload.get('variant')!r}; only direct_gap is qualified"
        )
    model = build_local_residual_student(
        payload["type_vocabulary"], **payload["model_kwargs"]
    ).to(torch.float64)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    ligand_payload = json.loads(ligand_indices_path.read_text(encoding="utf-8"))
    ligand_topology_indices = sorted(int(index) for index in ligand_payload["ligand_indices"])
    all_atomic_numbers = [int(atom.element.atomic_number) for atom in topology.atoms()]
    deployable = build_deployable_student_module(
        model,
        ligand_topology_indices=ligand_topology_indices,
        all_topology_atomic_numbers=all_atomic_numbers,
        temperature_kelvin=target_temperature_k,
        a_k=1.0,
        energy_offset_reduced=0.0,
    ).to(torch.float64)
    deployable.eval()
    torchscript_sha256 = export_torchscript(deployable, args.torchscript_output)
    ligand_indices_sha256 = _sha256_file(ligand_indices_path)
    basis_spec = NeuralBasisModelSpec(
        name="local_residual_student_hard_window0_exp013_design1_precheck",
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
    controller = OuterLambdaController(
        enabled=True,
        stage="vanishing",
        baseline_potential=stage_payload["potential_type"],
        endpoint_tolerance=1e-12,
        coefficients=(float(args.coefficient),),
        max_abs_coefficient=float(args.max_abs_coefficient),
        bases=(basis_spec,),
        safety=NeuralPathSafety(
            max_abs_basis_energy_kj_mol=float(args.max_abs_basis_energy_kj_mol),
            max_abs_path_energy_kj_mol=float(args.max_abs_path_energy_kj_mol),
            max_force_norm_kj_mol_nm=float(args.max_force_norm_kj_mol_nm),
            fail_on_support_domain_violation=False,
        ),
    )

    def _build_fused_win_sys():
        win_sys, original_ibs = build_ibs_dual_system(
            base_system, topology, stage_payload["ligand_indices"],
            manifest["lambdas_coul"], lambdas_vdw, alchemical_params,
            potential_type=stage_payload["potential_type"],
            restraint_params=stage_payload["boresch_params"],
            temperature=target_temperature_k * unit.kelvin,
            prefix=prefix, box_vectors=box_vectors, reference_positions=None,
            dispersion_protocol="legacy_uniform_density_lrc", environment_type="soluble",
        )
        group1_indices = [
            index for index in range(win_sys.getNumForces())
            if win_sys.getForce(index).getForceGroup() == 1
        ]
        if len(group1_indices) != 1:
            raise Design1PrecheckError(
                f"方案①要求原始 Group 1 恰有一个 IBS force，实际 {len(group1_indices)} 个"
            )
        if len(original_ibs._int_cv_force_xmls) != n_states:
            raise Design1PrecheckError("original IBSBiasForce did not capture all state CV XMLs")
        win_sys.removeForce(group1_indices[0])

        fused = OuterLambdaIBSBiasForce(
            controller, lambdas_vdw, target_temperature_k,
            [build_torchforce_from_spec(basis_spec)], prefix=prefix,
        )
        for state_index, force_xml in enumerate(original_ibs._int_cv_force_xmls):
            fused.addCollectiveVariable(
                f"cv_{state_index}_int", XmlSerializer.deserialize(force_xml)
            )
            fused.addCollectiveVariable(
                f"cv_{state_index}_rest", openmm.CustomExternalForce("0")
            )
        fused.setForceGroup(1)
        win_sys.addForce(fused.get_force())
        return win_sys, fused

    def _make_mts_simulation(win_sys, n_value: int):
        force_groups = {int(force.getForceGroup()) for force in win_sys.getForces()}
        if 1 not in force_groups:
            raise Design1PrecheckError("fused Group 1 is missing from scheme ① system")
        fast_groups = sorted(force_groups - {1})
        mts_groups = [(group, n_value) for group in fast_groups] + [(1, 1)]
        outer_dt_ps = n_value * inner_dt_ps
        integrator = openmm.MTSLangevinIntegrator(
            target_temperature_k * unit.kelvin,
            manifest["friction_per_ps"] / unit.picosecond,
            outer_dt_ps * unit.picosecond,
            mts_groups,
        )
        integrator.setConstraintTolerance(1e-3)
        integrator.setRandomNumberSeed(args.random_seed)
        simulation = app.Simulation(
            topology, win_sys, integrator, platform, platform_properties
        )
        # DEC-054/056: explicit public State transfer; NEVER loadCheckpoint() here.
        simulation.context.setPositions(source_positions)
        simulation.context.setVelocities(source_velocities)
        simulation.context.setPeriodicBoxVectors(*box_vectors)
        for parameter_name, parameter_value in source_parameters.items():
            if _system_has_global_parameter(win_sys, parameter_name):
                simulation.context.setParameter(parameter_name, parameter_value)
        for parameter_name in ("lambda_boresch_scale", "lambda_shield"):
            if _system_has_global_parameter(win_sys, parameter_name) and parameter_name in manifest:
                simulation.context.setParameter(parameter_name, float(manifest[parameter_name]))
        return simulation, integrator, mts_groups, outer_dt_ps

    dof = _degrees_of_freedom(base_system, unit)
    gas_constant = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
        unit.kilojoule_per_mole / unit.kelvin
    )

    def _run_n_value(n_value: int) -> dict:
        win_sys, fused = _build_fused_win_sys()
        simulation, integrator, mts_groups, outer_dt_ps = _make_mts_simulation(win_sys, n_value)
        fused.update_parameters(simulation.context, f_k)

        def _snapshot() -> dict:
            all_state = simulation.context.getState(getEnergy=True, getForces=True)
            group_state = simulation.context.getState(getEnergy=True, groups={1})
            potential = all_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            kinetic = all_state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            fused_energy = group_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            temperature = 2.0 * kinetic / (dof * gas_constant) if dof > 0 else float("nan")
            forces = all_state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            max_force = float(np.max(np.linalg.norm(forces, axis=1)))
            return {
                "fused_group1_energy_kj_mol": fused_energy,
                "potential_total_kj_mol": potential,
                "kinetic_total_kj_mol": kinetic,
                "total_energy_kj_mol": potential + kinetic,
                "temperature_k": temperature,
                "max_force_norm_kj_mol_nm": max_force,
                "all_finite": bool(
                    math.isfinite(fused_energy)
                    and math.isfinite(potential)
                    and math.isfinite(kinetic)
                    and math.isfinite(temperature)
                    and np.all(np.isfinite(forces))
                ),
            }

        outer_steps_per_tick = _MACRO_TICK_INNER_STEPS // n_value
        for _ in range(args.warmup_macro_ticks):
            simulation.step(outer_steps_per_tick)
        warmup_end = _snapshot()
        snapshots = []
        for _ in range(args.monitored_macro_ticks):
            simulation.step(outer_steps_per_tick)
            snapshots.append(_snapshot())

        total_energies = [item["total_energy_kj_mol"] for item in snapshots]
        half = len(total_energies) // 2
        first_mean = sum(total_energies[:half]) / half
        second_mean = sum(total_energies[half:]) / (len(total_energies) - half)
        energy_drift = second_mean - first_mean
        mean_kinetic = sum(item["kinetic_total_kj_mol"] for item in snapshots) / len(snapshots)
        mean_temperature = sum(item["temperature_k"] for item in snapshots) / len(snapshots)
        relative_energy_drift = abs(energy_drift) / mean_kinetic if mean_kinetic > 0 else float("inf")
        absolute_health_passed = bool(
            all(item["all_finite"] for item in snapshots)
            and args.min_mean_temperature_k <= mean_temperature <= args.max_mean_temperature_k
            and args.min_mean_temperature_k <= warmup_end["temperature_k"] <= args.max_mean_temperature_k
            and relative_energy_drift <= args.max_relative_energy_drift
        )
        del simulation, integrator
        return {
            "n_value": n_value,
            "inner_dt_ps": inner_dt_ps,
            "outer_dt_ps": outer_dt_ps,
            "outer_steps_per_macro_tick": outer_steps_per_tick,
            "mts_groups": mts_groups,
            "physical_time_ps": {
                "warmup": args.warmup_macro_ticks * _MACRO_TICK_INNER_STEPS * inner_dt_ps,
                "monitored": args.monitored_macro_ticks * _MACRO_TICK_INNER_STEPS * inner_dt_ps,
            },
            "n_snapshots": len(snapshots),
            "all_finite": all(item["all_finite"] for item in snapshots),
            "warmup_end_temperature_k": warmup_end["temperature_k"],
            "mean_temperature_k": mean_temperature,
            "mean_kinetic_kj_mol": mean_kinetic,
            "total_energy_drift_first_vs_second_half_kj_mol": energy_drift,
            "relative_energy_drift": relative_energy_drift,
            "max_force_norm_kj_mol_nm_overall": max(
                item["max_force_norm_kj_mol_nm"] for item in snapshots
            ),
            "absolute_health_passed": absolute_health_passed,
            "snapshots": snapshots,
        }

    results_by_n = {n_value: _run_n_value(n_value) for n_value in _N_VALUES}

    def _series(n_value: int, key: str) -> list[float]:
        return [item[key] for item in results_by_n[n_value]["snapshots"]]

    comparisons = {}
    comparison_block_ticks = (
        _QUALIFICATION_BLOCK_TICKS if args.phase == "qualification" else None
    )
    for n_value in _N_VALUES[1:]:
        comparisons[n_value] = {
            "temperature_k": _z_score(
                _series(1, "temperature_k"),
                _series(n_value, "temperature_k"),
                block_ticks=comparison_block_ticks,
                z_threshold=args.systematic_shift_z_threshold,
            ),
            "fused_group1_energy_kj_mol": _z_score(
                _series(1, "fused_group1_energy_kj_mol"),
                _series(n_value, "fused_group1_energy_kj_mol"),
                block_ticks=comparison_block_ticks,
                z_threshold=args.systematic_shift_z_threshold,
            ),
        }

    absolute_health_by_n = {
        n_value: results_by_n[n_value]["absolute_health_passed"] for n_value in _N_VALUES
    }
    shifts_by_n = {
        n_value: any(metric["systematic_shift_detected"] for metric in comparisons[n_value].values())
        for n_value in _N_VALUES[1:]
    }
    gate = evaluate_precheck_gate(
        args.phase,
        {n_value: results_by_n[n_value]["all_finite"] for n_value in _N_VALUES},
        absolute_health_by_n,
        shifts_by_n,
    )
    all_absolute_health_passed = bool(gate["all_absolute_health_passed"])
    no_systematic_shift = bool(gate["no_systematic_shift"])
    phase_passed = bool(
        gate["smoke_passed"] if args.phase == "smoke" else gate["qualification_passed"]
    )
    temperature_means = [results_by_n[n]["mean_temperature_k"] for n in _N_VALUES]
    monotonic_temperature_dose_response = all(
        right > left for left, right in zip(temperature_means, temperature_means[1:])
    )

    body = {
        "schema_version": "exp013-design1-mts-precheck-v2",
        "status": f"COMPLETED_DESIGN1_{args.phase.upper()}",
        "platform": {
            "requested": args.platform,
            "resolved_name": resolved_platform_name,
            "properties": platform_properties,
        },
        "window": {
            "stage_type": args.stage_type,
            "window_index": args.window_index,
            "target_temperature_k": target_temperature_k,
            "inner_dt_ps": inner_dt_ps,
            "inner_dt_fs": inner_dt_ps * 1000.0,
        },
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "production_checkpoint_path": str(checkpoint_path.resolve()),
        "production_checkpoint_sha256": _sha256_file(checkpoint_path),
        "torchscript_path": str(Path(args.torchscript_output).resolve()),
        "torchscript_sha256": torchscript_sha256,
        "controller": {
            "coefficient_c1": float(args.coefficient),
            "protocol_sha256": controller.protocol_sha256(lambdas=lambdas_vdw),
        },
        "scheme1_contract": {
            "fused_group1_is_slow": True,
            "slow_force_groups": [1],
            "fast_force_groups": sorted(
                {group for group in results_by_n[1]["mts_groups"] if group[0] != 1}
            ),
            "state_api_initialization": True,
            "mts_context_load_checkpoint_calls": 0,
            "same_integrator_probe_load_checkpoint_calls": 1,
            "cross_integrator_load_checkpoint_forbidden": True,
            "n_values_run": list(_N_VALUES),
            "n16_n32_run": False,
        },
        "phase": {
            "name": args.phase,
            "macro_tick_inner_steps": _MACRO_TICK_INNER_STEPS,
            "sample_interval_ps": _MACRO_TICK_INNER_STEPS * inner_dt_ps,
            "warmup_macro_ticks": args.warmup_macro_ticks,
            "monitored_macro_ticks": args.monitored_macro_ticks,
            "same_physical_time_across_n": True,
            "sem_method": (
                "contiguous_block_mean_sem"
                if comparison_block_ticks is not None
                else "ordinary_snapshot_sem_diagnostic_only"
            ),
            "block_ticks": comparison_block_ticks,
            "block_physical_time_ps": (
                comparison_block_ticks * _MACRO_TICK_INNER_STEPS * inner_dt_ps
                if comparison_block_ticks is not None
                else None
            ),
        },
        "results_by_n": results_by_n,
        "comparisons_vs_n1": comparisons,
        "precheck": {
            "z_threshold": args.systematic_shift_z_threshold,
            "absolute_health_passed_by_n": absolute_health_by_n,
            "all_absolute_health_passed": all_absolute_health_passed,
            "systematic_shift_detected_by_n": shifts_by_n,
            "no_systematic_shift": no_systematic_shift,
            "gate": gate,
            "temperature_mean_by_n": dict(zip(_N_VALUES, temperature_means)),
            "monotonic_temperature_dose_response": monotonic_temperature_dose_response,
            "precheck_passed": phase_passed,
            "eligible_for_n16_followup": bool(gate["eligible_for_n16_followup"]),
            "next_action": (
                "run_qualification_before_N16"
                if args.phase == "smoke" and gate["smoke_passed"]
                else "eligible_to_design1_N16_followup"
                if gate["eligible_for_n16_followup"]
                else "STOP_design1_before_N16; investigate_precheck_failure"
            ),
        },
        "policy": {
            "decision_reference": "DEC-056",
            "not_013b_three_repeat": True,
            "preregistered_013b_threshold_not_relaxed": True,
            "production_checkpoints_written": False,
            "note": "Smoke never authorizes N=16. Only qualification with block-aware SEM can set eligible_for_n16_followup=true; neither phase reopens 013-C.",
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
        f"phase_passed={phase_passed} eligible_for_n16_followup={gate['eligible_for_n16_followup']} "
        f"all_absolute_health_passed={all_absolute_health_passed} "
        f"no_systematic_shift={no_systematic_shift} shifts_by_n={shifts_by_n}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
