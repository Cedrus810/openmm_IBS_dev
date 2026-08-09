#!/usr/bin/env python
"""EXP-013 013-B thermal-collapse root-cause diagnostic (NOT a qualification gate).

013-B (DEC-052/053 script, report `exp013_013b_mts_dynamics_qualification_report.json`)
found `systematic_shift_detected_by_n=False` for N in {8,16,32} vs N=1, but the N=1
reference itself is not a healthy 300 K trajectory: Phase 1 (right after loadCheckpoint,
one N=16 outer step) reads kinetic=524.78 kJ/mol (~298 K implied), but every Phase 2
monitored snapshot at every N reads temperature ~1e-3 K, and total energy
(potential+kinetic) drifts by ~-14,250 kJ/mol across the 32 ps monitored window,
uniformly across N=1/8/16/32. This is a shared, N-independent pathology the relative
N-vs-N=1 gate cannot see. 013-B must be re-run and re-judged only after this is
root-caused and fixed -- treat 013-B's `all_passed=True` as INVALID until then.

Working hypothesis (physically self-consistent with the observed numbers): the
Langevin "O" step's noise term is missing or using an effective kT near zero, so
friction damps velocities with nothing replacing the dissipated energy -- Langevin
dynamics degenerates into damped/steepest-descent relaxation, which drains BOTH
kinetic and potential energy as the system slides toward a nearby local minimum
(matches the large negative total-energy drift, not just a kinetic-only artifact).

This script does NOT step MTSLangevinIntegrator repeatedly. It runs four cheap,
independent checks, in the priority order the investigation agreed on:

  A. Construct MTSLangevinIntegrator for N in {1,8,16,32} (no stepping) and dump every
     CustomIntegrator global variable name/value. If OpenMM's own kT-like global is
     ~0 or NaN across all N, the bug is inside the integrator's own construction/
     initialization for this argument combination -- not in this repo's split/TorchForce
     code at all.
  B. Fine-grained temperature/KE time series for N=1 on the FULL 013-B split system
     (V_0 fast + dV slow), sampled at t = 0, 0.064, 0.128, 0.32, 0.64, 1.28, 3.2, 6.4 ps.
     An exponential-looking decay with time constant near 1/friction is the noise-loss
     signature the hypothesis predicts; a sudden one-step collapse or NaN points somewhere
     else (e.g. constraint failure).
  C. Minimal control 1: the ORIGINAL production System (no dV split, no CustomCVForce,
     no TorchForce -- exactly `_build_baseline_win_sys()`, the V_0 Hamiltonian this
     checkpoint was actually produced under) driven by a fresh MTSLangevinIntegrator at
     N=1 (degenerate single-group case). If this ALSO cools, the bug is generic to using
     MTSLangevinIntegrator + loadCheckpoint in this codebase, unrelated to EXP-013's
     residual split.
  D. Minimal control 2: same original production System, but with the plain
     LangevinMiddleIntegrator (the integrator this checkpoint's own production run
     actually used) instead of MTS. This isolates "is loadCheckpoint-into-a-rebuilt-
     Context itself unsafe" from "is MTSLangevinIntegrator itself the problem" -- if
     control 2 stays at ~300 K and control 1 does not, the fault is specifically in
     MTSLangevinIntegrator usage, not in the checkpoint-reload pattern.

Each check is independent and cheap (<=6.4 ps of dynamics, most of them 0 ps). This is
a diagnostic, not a qualification artifact: it does not write a schema_version'd,
sha256'd report and is not meant to be cited as a DEC input by itself. Its stdout is
the deliverable; if a follow-up qualification report is warranted once the root cause
is fixed, that is a separate, new 013-B (or 013-B-v2) run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MACRO_TICK_INNER_STEPS = 32
_SAMPLE_TIMES_PS = (0.0, 0.064, 0.128, 0.32, 0.64, 1.28, 3.2, 6.4)


def _degrees_of_freedom(system) -> int:
    dof = 0
    for index in range(system.getNumParticles()):
        if system.getParticleMass(index).value_in_unit(__import__("openmm").unit.dalton) > 0.0:
            dof += 3
    dof -= system.getNumConstraints()
    import openmm
    if any(isinstance(force, openmm.CMMotionRemover) for force in system.getForces()):
        dof -= 3
    return dof


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="output_lrc_fix")
    parser.add_argument("--stage-type", default="vdw", choices=["vdw"])
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--checkpoint", required=True, help="frozen direct_gap .pt candidate (DEC-045)")
    parser.add_argument("--coefficient", type=float, default=0.5)
    parser.add_argument("--constraint-tolerance", type=float, default=1e-3,
                         help="passed to setConstraintTolerance() for every integrator built here "
                              "(MTS in checks B/C, plain LangevinMiddleIntegrator in check D) -- "
                              "override to test whether constraint-solver quality explains the collapse")
    args = parser.parse_args(argv)

    import numpy as np
    import openmm
    import torch
    from openmm import XmlSerializer, app, unit

    from ibs_engine import (
        ACESoftcorePotential,
        _build_platform_properties,
        _system_has_global_parameter,
        build_ibs_dual_system,
    )
    from outer_lambda_neural_basis import (
        NeuralBasisModelSpec,
        NeuralPathSafety,
        OuterLambdaController,
        OuterLambdaResidualBiasForce,
        build_torchforce_from_spec,
    )
    from local_residual.student import build_local_residual_student
    from local_residual.student_deploy import build_deployable_student_module, export_torchscript

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

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_payload = json.loads(stage_protocol_path.read_text(encoding="utf-8"))["protocol_key"]["payload"]
    ibs_state = json.loads(ibs_state_path.read_text(encoding="utf-8"))
    target_temperature_k = float(manifest["temperature_K"])
    friction_per_ps = float(manifest["friction_per_ps"])
    inner_dt_ps = float(manifest["step_size_ps"])
    lambdas_vdw = [float(value) for value in manifest["lambdas_vdw"]]
    n_states = len(lambdas_vdw)
    prefix = ibs_state["prefix"]
    f_k = np.asarray(ibs_state["f_k"], dtype=float)

    system_xml_text = system_xml_path.read_text(encoding="utf-8")
    base_system = XmlSerializer.deserialize(system_xml_text)
    topology = app.PDBxFile(str(topology_cif_path)).topology
    stale_box_vectors = unit.Quantity(np.load(box_vectors_path), unit.nanometer)
    alchemical_params = ACESoftcorePotential.from_dict(stage_payload["aces_softcore_params"])

    resolved_platform_name, platform_properties = _build_platform_properties(args.platform)
    platform = openmm.Platform.getPlatformByName(resolved_platform_name)

    probe_win_sys, _probe_ibs = build_ibs_dual_system(
        base_system, topology, stage_payload["ligand_indices"],
        manifest["lambdas_coul"], lambdas_vdw, alchemical_params,
        potential_type=stage_payload["potential_type"], restraint_params=stage_payload["boresch_params"],
        temperature=target_temperature_k * unit.kelvin, prefix=prefix,
        box_vectors=stale_box_vectors, reference_positions=None,
        dispersion_protocol="legacy_uniform_density_lrc", environment_type="soluble",
    )
    probe_integrator = openmm.LangevinMiddleIntegrator(
        target_temperature_k * unit.kelvin, friction_per_ps / unit.picosecond,
        inner_dt_ps * unit.picosecond,
    )
    probe_simulation = app.Simulation(topology, probe_win_sys, probe_integrator, platform, platform_properties)
    probe_simulation.loadCheckpoint(str(checkpoint_path))
    box_vectors = probe_simulation.context.getState(getPositions=True).getPeriodicBoxVectors()
    del probe_simulation, probe_integrator, probe_win_sys, _probe_ibs

    def _build_baseline_win_sys():
        return build_ibs_dual_system(
            base_system, topology, stage_payload["ligand_indices"],
            manifest["lambdas_coul"], lambdas_vdw, alchemical_params,
            potential_type=stage_payload["potential_type"], restraint_params=stage_payload["boresch_params"],
            temperature=target_temperature_k * unit.kelvin, prefix=prefix,
            box_vectors=box_vectors, reference_positions=None,
            dispersion_protocol="legacy_uniform_density_lrc", environment_type="soluble",
        )

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_local_residual_student(payload["type_vocabulary"], **payload["model_kwargs"]).to(torch.float64)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    ligand_payload = json.loads(ligand_indices_path.read_text(encoding="utf-8"))
    ligand_topology_indices = sorted(int(index) for index in ligand_payload["ligand_indices"])
    all_topology_atomic_numbers = [int(atom.element.atomic_number) for atom in topology.atoms()]
    deployable = build_deployable_student_module(
        model, ligand_topology_indices=ligand_topology_indices,
        all_topology_atomic_numbers=all_topology_atomic_numbers,
        temperature_kelvin=target_temperature_k, a_k=1.0, energy_offset_reduced=0.0,
    ).to(torch.float64)
    deployable.eval()
    torchscript_path = "/tmp/exp013_mts_diag_student.pt"
    torchscript_sha256 = export_torchscript(deployable, torchscript_path)
    basis_spec = NeuralBasisModelSpec(
        name="local_residual_student_hard_window0_diag", backend="torchforce",
        model_path=str(Path(torchscript_path).resolve()), sha256=torchscript_sha256,
        energy_offset_kj_mol=0.0, atom_selection="dynamic_funnel_environment",
        atom_indices_path=str(ligand_indices_path.resolve()),
        atom_indices_sha256="unchecked-diagnostic-only",
        output_unit="kJ_per_mol", precision="double", periodic=True,
    )
    controller = OuterLambdaController(
        enabled=True, stage="vanishing", baseline_potential=stage_payload["potential_type"],
        endpoint_tolerance=1e-12, coefficients=(float(args.coefficient),),
        max_abs_coefficient=1.0, bases=(basis_spec,),
        safety=NeuralPathSafety(
            max_abs_basis_energy_kj_mol=50.0, max_abs_path_energy_kj_mol=25.0,
            max_force_norm_kj_mol_nm=500.0, fail_on_support_domain_violation=False,
        ),
    )

    def _build_split_win_sys():
        win_sys, original_ibs_wrap = _build_baseline_win_sys()
        student_force = build_torchforce_from_spec(basis_spec)
        delta_wrapper = OuterLambdaResidualBiasForce(
            controller, lambdas_vdw, target_temperature_k, [student_force], prefix=prefix,
        )
        for k in range(n_states):
            int_cv = XmlSerializer.deserialize(original_ibs_wrap._int_cv_force_xmls[k])
            delta_wrapper.addCollectiveVariable(f"cv_{k}_int", int_cv)
            delta_wrapper.addCollectiveVariable(f"cv_{k}_rest", openmm.CustomExternalForce("0"))
        existing_groups = {int(force.getForceGroup()) for force in win_sys.getForces()}
        delta_group = max(existing_groups) + 1 if existing_groups else 0
        delta_wrapper.setForceGroup(delta_group)
        win_sys.addForce(delta_wrapper.get_force())
        return win_sys, original_ibs_wrap, delta_wrapper, delta_group

    def _temperature_k(context, dof: int) -> tuple[float, float]:
        state = context.getState(getEnergy=True)
        ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        r = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
        return ke, (2.0 * ke / (dof * r) if dof > 0 else float("nan"))

    def _dump_integrator_globals(label: str, integrator) -> None:
        print(f"--- {label}: CustomIntegrator global variables ---")
        n = integrator.getNumGlobalVariables()
        for i in range(n):
            name = integrator.getGlobalVariableName(i)
            value = integrator.getGlobalVariable(i)
            print(f"    [{i}] {name} = {value}")
        print(f"    getStepSize() = {integrator.getStepSize()}")
        print(f"    --- per-step computation program ({integrator.getNumComputations()} steps) ---")
        for i in range(integrator.getNumComputations()):
            step = integrator.getComputationStep(i)
            print(f"    [{i}] {step}")

    print("=" * 80)
    print("CHECK A: MTSLangevinIntegrator global variables at construction, N in {1,8,16,32}, no stepping")
    print("=" * 80)
    for n_value in (1, 8, 16, 32):
        win_sys, _v0, _dv, delta_group = _build_split_win_sys()
        fast_groups = sorted({int(f.getForceGroup()) for f in win_sys.getForces()} - {delta_group})
        mts_groups = [(g, n_value) for g in fast_groups] + [(delta_group, 1)]
        outer_dt_ps = n_value * inner_dt_ps
        integrator = openmm.MTSLangevinIntegrator(
            target_temperature_k * unit.kelvin, friction_per_ps / unit.picosecond,
            outer_dt_ps * unit.picosecond, mts_groups,
        )
        _dump_integrator_globals(f"N={n_value} (outer_dt={outer_dt_ps} ps, groups={mts_groups})", integrator)
        del integrator, win_sys

    print()
    print("=" * 80)
    print("CHECK B: N=1 full split system, fine-grained temperature/KE time series over warmup")
    print("=" * 80)
    win_sys, v0_wrap, dv_wrap, delta_group = _build_split_win_sys()
    all_groups_present_b = sorted({int(f.getForceGroup()) for f in win_sys.getForces()})
    fast_groups = sorted({int(f.getForceGroup()) for f in win_sys.getForces()} - {delta_group})
    mts_groups = [(g, 1) for g in fast_groups] + [(delta_group, 1)]
    integrator = openmm.MTSLangevinIntegrator(
        target_temperature_k * unit.kelvin, friction_per_ps / unit.picosecond,
        inner_dt_ps * unit.picosecond, mts_groups,
    )
    print(f"    C1 coverage check: win_sys force groups present = {all_groups_present_b}, "
          f"mts_groups constructor arg = {mts_groups}, "
          f"getIntegrationForceGroups() = {integrator.getIntegrationForceGroups()}")
    if set(all_groups_present_b) != {g for g, _n in mts_groups}:
        print(f"    !! C1 MISMATCH: {set(all_groups_present_b) - {g for g, _n in mts_groups}} present in "
              f"win_sys but missing from mts_groups (or vice versa)")
    _dump_integrator_globals("CHECK B integrator (N=1, full split system)", integrator)
    integrator.setConstraintTolerance(args.constraint_tolerance)
    simulation = app.Simulation(topology, win_sys, integrator, platform, platform_properties)
    simulation.loadCheckpoint(str(checkpoint_path))
    if _system_has_global_parameter(win_sys, "lambda_boresch_scale"):
        simulation.context.setParameter("lambda_boresch_scale", float(manifest["lambda_boresch_scale"]))
    if _system_has_global_parameter(win_sys, "lambda_shield"):
        simulation.context.setParameter("lambda_shield", float(manifest["lambda_shield"]))
    v0_wrap.update_parameters(simulation.context, f_k)
    dv_wrap.update_parameters(simulation.context, f_k)
    dof = _degrees_of_freedom(win_sys)
    print(f"    dof = {dof}, target_temperature_k = {target_temperature_k}, friction_per_ps = {friction_per_ps}")
    time_elapsed_ps = 0.0
    for target_time_ps in _SAMPLE_TIMES_PS:
        steps_needed = round((target_time_ps - time_elapsed_ps) / inner_dt_ps)
        if steps_needed > 0:
            simulation.step(steps_needed)
            time_elapsed_ps += steps_needed * inner_dt_ps
        ke, temp = _temperature_k(simulation.context, dof)
        predicted_ratio = math.exp(-2.0 * friction_per_ps * target_time_ps)
        print(f"    t={time_elapsed_ps:.3f} ps  KE={ke:12.4f} kJ/mol  T={temp:10.6f} K  "
              f"(noise-free-decay KE-ratio prediction at this t: {predicted_ratio:.3e})")
    del simulation, integrator

    print()
    print("=" * 80)
    print("CHECK C: control 1 -- ORIGINAL production System (no dV split), MTSLangevinIntegrator N=1")
    print("=" * 80)
    win_sys_c, _wrap_c = _build_baseline_win_sys()
    fast_groups_c = sorted({int(f.getForceGroup()) for f in win_sys_c.getForces()})
    mts_groups_c = [(g, 1) for g in fast_groups_c]
    integrator_c = openmm.MTSLangevinIntegrator(
        target_temperature_k * unit.kelvin, friction_per_ps / unit.picosecond,
        inner_dt_ps * unit.picosecond, mts_groups_c,
    )
    print(f"    C1 coverage check: win_sys_c force groups present = {fast_groups_c}, "
          f"mts_groups_c constructor arg = {mts_groups_c}, "
          f"getIntegrationForceGroups() = {integrator_c.getIntegrationForceGroups()}")
    if set(fast_groups_c) != {g for g, _n in mts_groups_c}:
        print(f"    !! C1 MISMATCH: {set(fast_groups_c) - {g for g, _n in mts_groups_c}} present in "
              f"win_sys_c but missing from mts_groups_c (or vice versa)")
    _dump_integrator_globals("CHECK C integrator (N=1, original baseline system)", integrator_c)
    integrator_c.setConstraintTolerance(args.constraint_tolerance)
    simulation_c = app.Simulation(topology, win_sys_c, integrator_c, platform, platform_properties)
    simulation_c.loadCheckpoint(str(checkpoint_path))
    if _system_has_global_parameter(win_sys_c, "lambda_boresch_scale"):
        simulation_c.context.setParameter("lambda_boresch_scale", float(manifest["lambda_boresch_scale"]))
    if _system_has_global_parameter(win_sys_c, "lambda_shield"):
        simulation_c.context.setParameter("lambda_shield", float(manifest["lambda_shield"]))
    _wrap_c.update_parameters(simulation_c.context, f_k)
    dof_c = _degrees_of_freedom(win_sys_c)
    time_elapsed_ps = 0.0
    for target_time_ps in _SAMPLE_TIMES_PS:
        steps_needed = round((target_time_ps - time_elapsed_ps) / inner_dt_ps)
        if steps_needed > 0:
            simulation_c.step(steps_needed)
            time_elapsed_ps += steps_needed * inner_dt_ps
        ke, temp = _temperature_k(simulation_c.context, dof_c)
        print(f"    t={time_elapsed_ps:.3f} ps  KE={ke:12.4f} kJ/mol  T={temp:10.6f} K")
    del simulation_c, integrator_c

    print()
    print("=" * 80)
    print("CHECK D: control 2 -- ORIGINAL production System, plain LangevinMiddleIntegrator N/A (no MTS)")
    print("=" * 80)
    win_sys_d, _wrap_d = _build_baseline_win_sys()
    integrator_d = openmm.LangevinMiddleIntegrator(
        target_temperature_k * unit.kelvin, friction_per_ps / unit.picosecond,
        inner_dt_ps * unit.picosecond,
    )
    integrator_d.setConstraintTolerance(args.constraint_tolerance)
    simulation_d = app.Simulation(topology, win_sys_d, integrator_d, platform, platform_properties)
    simulation_d.loadCheckpoint(str(checkpoint_path))
    if _system_has_global_parameter(win_sys_d, "lambda_boresch_scale"):
        simulation_d.context.setParameter("lambda_boresch_scale", float(manifest["lambda_boresch_scale"]))
    if _system_has_global_parameter(win_sys_d, "lambda_shield"):
        simulation_d.context.setParameter("lambda_shield", float(manifest["lambda_shield"]))
    _wrap_d.update_parameters(simulation_d.context, f_k)
    dof_d = _degrees_of_freedom(win_sys_d)
    time_elapsed_ps = 0.0
    for target_time_ps in _SAMPLE_TIMES_PS:
        steps_needed = round((target_time_ps - time_elapsed_ps) / inner_dt_ps)
        if steps_needed > 0:
            simulation_d.step(steps_needed)
            time_elapsed_ps += steps_needed * inner_dt_ps
        ke, temp = _temperature_k(simulation_d.context, dof_d)
        print(f"    t={time_elapsed_ps:.3f} ps  KE={ke:12.4f} kJ/mol  T={temp:10.6f} K")
    del simulation_d, integrator_d

    print()
    print("=" * 80)
    print("CHECK E: state-transfer initialization -- NO loadCheckpoint() into the MTS Context")
    print("=" * 80)
    print("    rationale: Context.loadCheckpoint() restores integrator/platform-internal binary")
    print("    state (per OpenMM docs, tied to the exact Context/Integrator that wrote it); the")
    print("    checkpoint here was written by this window's own production run (a plain")
    print("    LangevinMiddleIntegrator), and every prior check loaded it directly into a")
    print("    MTSLangevinIntegrator (CustomIntegrator) Context instead -- a cross-integrator")
    print("    binary checkpoint load, which is not the documented use case. This check instead")
    print("    round-trips through the PUBLIC State API only: load the checkpoint into a source")
    print("    Context matching the integrator that actually wrote it, extract positions/")
    print("    velocities/box/parameters, and explicitly set them on a fresh MTS Context.")
    win_sys_src, _wrap_src = _build_baseline_win_sys()
    integrator_src = openmm.LangevinMiddleIntegrator(
        target_temperature_k * unit.kelvin, friction_per_ps / unit.picosecond,
        inner_dt_ps * unit.picosecond,
    )
    integrator_src.setConstraintTolerance(args.constraint_tolerance)
    simulation_src = app.Simulation(topology, win_sys_src, integrator_src, platform, platform_properties)
    simulation_src.loadCheckpoint(str(checkpoint_path))
    if _system_has_global_parameter(win_sys_src, "lambda_boresch_scale"):
        simulation_src.context.setParameter("lambda_boresch_scale", float(manifest["lambda_boresch_scale"]))
    if _system_has_global_parameter(win_sys_src, "lambda_shield"):
        simulation_src.context.setParameter("lambda_shield", float(manifest["lambda_shield"]))
    _wrap_src.update_parameters(simulation_src.context, f_k)
    source_state = simulation_src.context.getState(
        getPositions=True, getVelocities=True, getParameters=True, getEnergy=True,
    )
    source_ke = source_state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    dof_src = _degrees_of_freedom(win_sys_src)
    gas_constant = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
    source_temp = 2.0 * source_ke / (dof_src * gas_constant) if dof_src > 0 else float("nan")
    print(f"    source Context (LangevinMiddleIntegrator, loadCheckpoint): "
          f"KE={source_ke:.4f} kJ/mol, T={source_temp:.6f} K")
    source_positions = source_state.getPositions()
    source_velocities = source_state.getVelocities()
    source_box = source_state.getPeriodicBoxVectors()
    source_parameters = dict(source_state.getParameters())
    del simulation_src, integrator_src, win_sys_src

    win_sys_e, _wrap_e = _build_baseline_win_sys()
    integrator_e = openmm.MTSLangevinIntegrator(
        target_temperature_k * unit.kelvin, friction_per_ps / unit.picosecond,
        inner_dt_ps * unit.picosecond,
        [(g, 1) for g in sorted({int(f.getForceGroup()) for f in win_sys_e.getForces()})],
    )
    integrator_e.setConstraintTolerance(args.constraint_tolerance)
    simulation_e = app.Simulation(topology, win_sys_e, integrator_e, platform, platform_properties)
    simulation_e.context.setPositions(source_positions)
    simulation_e.context.setVelocities(source_velocities)
    simulation_e.context.setPeriodicBoxVectors(*source_box)
    for name, value in source_parameters.items():
        try:
            simulation_e.context.setParameter(name, value)
        except Exception as exc:  # noqa: BLE001 -- diagnostic script, report and continue
            print(f"    (note: could not setParameter({name!r}, {value!r}) on target Context: {exc})")
    _wrap_e.update_parameters(simulation_e.context, f_k)
    dof_e = _degrees_of_freedom(win_sys_e)
    state0_e = simulation_e.context.getState(getEnergy=True)
    ke0_e = state0_e.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    temp0_e = 2.0 * ke0_e / (dof_e * gas_constant) if dof_e > 0 else float("nan")
    print(f"    target Context (MTSLangevinIntegrator N=1, state-transfer, NO loadCheckpoint): "
          f"KE={ke0_e:.4f} kJ/mol, T={temp0_e:.6f} K  (should match source before any stepping)")
    time_elapsed_ps = 0.0
    for target_time_ps in _SAMPLE_TIMES_PS:
        steps_needed = round((target_time_ps - time_elapsed_ps) / inner_dt_ps)
        if steps_needed > 0:
            simulation_e.step(steps_needed)
            time_elapsed_ps += steps_needed * inner_dt_ps
        ke, temp = _temperature_k(simulation_e.context, dof_e)
        print(f"    t={time_elapsed_ps:.3f} ps  KE={ke:12.4f} kJ/mol  T={temp:10.6f} K")
    del simulation_e, integrator_e

    print()
    print("Diagnostic complete. This is NOT a qualification report -- no JSON/sha256 written.")
    print("Interpretation guide:")
    print("  A: if some global (kT-like) is ~0/NaN for all N -> bug is inside MTSLangevinIntegrator")
    print("     construction itself for these args, independent of this repo's code.")
    print("  B: compare observed KE decay against the printed noise-free-decay prediction column;")
    print("     a close match implicates missing/zeroed thermostat noise specifically.")
    print("  C vs D: if C (MTS, N=1, original system) cools but D (plain Langevin, same system)")
    print("     does not, the fault is specific to using MTSLangevinIntegrator in this codebase,")
    print("     not to loadCheckpoint-into-a-rebuilt-Context in general, and not to the dV split.")
    print("  If C stays at ~300 K too, re-check CHECK B's full-split-system result against C:")
    print("     the fault would then be located specifically in the dV/CustomCVForce/TorchForce")
    print("     additions, not in MTSLangevinIntegrator or checkpoint reload at all.")
    print("  E: if E stays at ~300 K where B/C/A all collapsed -> cross-integrator binary")
    print("     loadCheckpoint() into a CustomIntegrator Context is the root cause; 013-B's")
    print("     report is INVALIDATED_BY_INITIALIZATION_BUG, not a real MTS/physics finding.")
    print("     Fix: always state-transfer (positions/velocities/box/parameters) through the")
    print("     public State API into a fresh MTS Context, never loadCheckpoint() across a")
    print("     different Integrator class, then re-run 013-B from scratch.")
    print("  If E ALSO collapses -> checkpoint migration is excluded too; only then does")
    print("     suspecting MTSLangevinIntegrator + this System/platform/OpenMM build become")
    print("     warranted, and the next step is a minimal fresh-velocity (no checkpoint at all)")
    print("     CUDA-vs-CPU/Reference platform comparison, not further bisection here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
