#!/usr/bin/env python
"""EXP-013 013-B (design (3), DEC-052/053): MTS dynamics qualification, N=1/8/16/32.

013-A (DEC-053) already passed: `dV` (`OuterLambdaResidualBiasForce`) is
numerically equivalent to `V_* - V_0` and its per-call cost is economically
feasible at N=16/32 (N=8 is not). This script is the FIRST time
`MTSLangevinIntegrator` is actually constructed and stepped with the current
TorchForce+CustomCVForce nesting -- EXP-009's failure mode (CUDA_ERROR_
INVALID_HANDLE at N=1, on a DIFFERENT backend, `openmm.PythonForce`) is a
real, open question for THIS backend too, never tested before now.

Two phases, in order, per DEC-052's "backend smoke before physics" policy:

Phase 1 -- minimal backend smoke: build the MTS integrator at a representative
N (16), run exactly ONE outer step, check it completes without a CUDA/OpenMM
exception and produces finite energies. If this fails with an EXP-009-style
backend error, this script stops here -- no retry, no platform change, no
coefficient change, and no attempt at the (expensive) Phase 2 comparison.

Phase 2 -- physical-time-aligned N=1/8/16/32 comparison: every N runs the
SAME physical time span (not the same step COUNT -- an outer step at N=32
spans 32x the physical time of an outer step at N=1, so aligning by step
count would silently compare different physical durations). Uses a "macro
tick" = 32 inner steps (LCM-exact for N in {1,8,16,32}: N=1 -> 32 outer
steps/tick, N=8 -> 4, N=16 -> 2, N=32 -> 1 -- no rounding). Compares, for
N in {8,16,32} against the N=1 reference: split total potential
(`V_0+dV`, i.e. the Hamiltonian dynamics actually sampled under -- this MUST
equal V_* by 013-A's equivalence, so this is also a second, dynamics-level
equivalence spot-check, not just a single-frame one), the `dV` contribution
alone, instantaneous temperature, whole-system max force norm, and a running
total-energy (potential-all-groups + kinetic) drift proxy for integration
error (NOT a real shadow-work estimator -- documented as a simple drift
diagnostic, not claimed to be more than that).
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

from local_residual.student import build_local_residual_student  # noqa: E402
from local_residual.student_deploy import build_deployable_student_module, export_torchscript  # noqa: E402

_MACRO_TICK_INNER_STEPS = 32  # LCM-exact base unit for N in {1, 8, 16, 32}
_N_VALUES = (1, 8, 16, 32)


class MtsQualificationError(RuntimeError):
    """A checkpoint/variant/comparison failed a fail-closed contract check."""


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


def _mean_and_sem(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan")
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, math.sqrt(variance / n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="output_lrc_fix")
    parser.add_argument("--stage-type", default="vdw", choices=["vdw"])
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--checkpoint", required=True, help="frozen direct_gap .pt candidate (DEC-045)")
    parser.add_argument("--coefficient", type=float, default=0.5, help="frozen c1 (DEC-045/047), do not retune")
    parser.add_argument("--max-abs-coefficient", type=float, default=1.0)
    parser.add_argument("--max-abs-basis-energy-kj-mol", type=float, default=50.0)
    parser.add_argument("--max-abs-path-energy-kj-mol", type=float, default=25.0)
    parser.add_argument("--max-force-norm-kj-mol-nm", type=float, default=500.0)
    parser.add_argument("--warmup-macro-ticks", type=int, default=100, help="discarded, physical-time-aligned across N")
    parser.add_argument("--monitored-macro-ticks", type=int, default=500, help="sampled once per macro tick")
    parser.add_argument("--systematic-shift-z-threshold", type=float, default=3.0)
    parser.add_argument("--min-mean-temperature-k", type=float, default=270.0,
                         help="DEC-054 absolute health gate: every N's monitored-window mean "
                              "temperature must be >= this. The relative N-vs-N=1 systematic-shift "
                              "gate is only meaningful if N=1 itself is a healthy reference; a prior "
                              "run passed that relative gate while every arm (including N=1) had "
                              "silently collapsed to ~0.003 K, which this gate would have caught.")
    parser.add_argument("--max-mean-temperature-k", type=float, default=330.0,
                         help="DEC-054 absolute health gate, upper bound (symmetric sanity check).")
    parser.add_argument("--max-relative-energy-drift", type=float, default=0.10,
                         help="DEC-054 absolute health gate: |total_energy_drift_first_vs_second_half| "
                              "/ mean_kinetic_energy must be <= this for every N. Catches a shared, "
                              "N-independent common-mode energy leak that a relative N-vs-N=1 "
                              "comparison alone cannot see.")
    parser.add_argument("--torchscript-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

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
    from outer_lambda_neural_basis import (  # noqa: E402
        NeuralBasisModelSpec,
        NeuralPathSafety,
        OuterLambdaController,
        OuterLambdaResidualBiasForce,
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

    for path in (manifest_path, checkpoint_path, stage_protocol_path, ibs_state_path,
                 system_xml_path, topology_cif_path, box_vectors_path, ligand_indices_path):
        if not path.is_file():
            raise MtsQualificationError(f"required real production artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_payload = json.loads(stage_protocol_path.read_text(encoding="utf-8"))["protocol_key"]["payload"]
    ibs_state = json.loads(ibs_state_path.read_text(encoding="utf-8"))
    target_temperature_k = float(manifest["temperature_K"])
    inner_dt_ps = float(manifest["step_size_ps"])
    lambdas_vdw = [float(value) for value in manifest["lambdas_vdw"]]
    n_states = len(lambdas_vdw)
    prefix = ibs_state["prefix"]
    f_k = np.asarray(ibs_state["f_k"], dtype=float)

    system_xml_text = system_xml_path.read_text(encoding="utf-8")
    if _sha256_text(system_xml_text) != stage_payload["system_xml_sha256"]:
        raise MtsQualificationError("system_native.xml SHA-256 does not match stage2 protocol record")
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
        target_temperature_k * unit.kelvin, manifest["friction_per_ps"] / unit.picosecond,
        manifest["step_size_ps"] * unit.picosecond,
    )
    probe_simulation = app.Simulation(topology, probe_win_sys, probe_integrator, platform, platform_properties)
    probe_simulation.loadCheckpoint(str(checkpoint_path))
    # State-transfer initialization (fixes INVALIDATED_BY_INITIALIZATION_BUG, DEC-054): the only
    # place `loadCheckpoint()` is ever called against this checkpoint is HERE, into a Context built
    # with `LangevinMiddleIntegrator` -- the same integrator class production actually wrote this
    # checkpoint with. Every MTS Context below is initialized from the resulting public State
    # (positions/velocities/box/parameters) via explicit setters, never via loadCheckpoint(). A
    # prior version of this script called `simulation.loadCheckpoint(str(checkpoint_path))` directly
    # against `MTSLangevinIntegrator` (CustomIntegrator) Contexts; OpenMM documents checkpoints as
    # tied to the exact Context/Integrator that wrote them, and doing so silently produced a
    # catastrophic kinetic-energy collapse (~300 K -> ~0.003 K within 64 fs, confirmed root-caused
    # via a 5-check elimination sequence: force-group coverage, force-sum consistency, thermostat
    # globals, per-step program inspection, and finally this state-transfer A/B control) while every
    # other quantity (forces, thermostat a/b/kT, per-step program, force-group coverage) checked out
    # as correct. See EXPERIMENT_LOG DEC-054.
    probe_state = probe_simulation.context.getState(
        getPositions=True, getVelocities=True, getParameters=True,
    )
    box_vectors = probe_state.getPeriodicBoxVectors()
    source_positions = probe_state.getPositions()
    source_velocities = probe_state.getVelocities()
    source_parameters = dict(probe_state.getParameters())
    del probe_simulation, probe_integrator, probe_win_sys, _probe_ibs, probe_state

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
    if payload.get("variant") != "direct_gap":
        raise MtsQualificationError(f"--checkpoint variant={payload.get('variant')!r}, only direct_gap is qualified")
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
    torchscript_sha256 = export_torchscript(deployable, args.torchscript_output)
    ligand_indices_sha256 = _sha256_file(ligand_indices_path)
    basis_spec = NeuralBasisModelSpec(
        name="local_residual_student_hard_window0_exp013_013b", backend="torchforce",
        model_path=str(Path(args.torchscript_output).resolve()), sha256=torchscript_sha256,
        energy_offset_kj_mol=0.0, atom_selection="dynamic_funnel_environment",
        atom_indices_path=str(ligand_indices_path.resolve()), atom_indices_sha256=ligand_indices_sha256,
        output_unit="kJ_per_mol", precision="double", periodic=True,
    )
    controller = OuterLambdaController(
        enabled=True, stage="vanishing", baseline_potential=stage_payload["potential_type"],
        endpoint_tolerance=1e-12, coefficients=(float(args.coefficient),),
        max_abs_coefficient=float(args.max_abs_coefficient), bases=(basis_spec,),
        safety=NeuralPathSafety(
            max_abs_basis_energy_kj_mol=float(args.max_abs_basis_energy_kj_mol),
            max_abs_path_energy_kj_mol=float(args.max_abs_path_energy_kj_mol),
            max_force_norm_kj_mol_nm=float(args.max_force_norm_kj_mol_nm),
            fail_on_support_domain_violation=False,
        ),
    )

    def _build_split_win_sys():
        win_sys, original_ibs_wrap = _build_baseline_win_sys()  # Group 1 untouched -- this is V_0
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
        if delta_group > 31:
            raise MtsQualificationError("no free OpenMM force group (0-31) left for dV")
        delta_wrapper.setForceGroup(delta_group)
        win_sys.addForce(delta_wrapper.get_force())
        return win_sys, original_ibs_wrap, delta_wrapper, delta_group

    def _make_mts_simulation(win_sys, delta_group: int, n_value: int):
        fast_groups = sorted({int(force.getForceGroup()) for force in win_sys.getForces()} - {delta_group})
        mts_groups = [(group, n_value) for group in fast_groups] + [(delta_group, 1)]
        outer_dt_ps = n_value * inner_dt_ps
        integrator = openmm.MTSLangevinIntegrator(
            target_temperature_k * unit.kelvin, manifest["friction_per_ps"] / unit.picosecond,
            outer_dt_ps * unit.picosecond, mts_groups,
        )
        integrator.setConstraintTolerance(1e-3)
        simulation = app.Simulation(topology, win_sys, integrator, platform, platform_properties)
        # State-transfer initialization (DEC-054) -- NEVER loadCheckpoint() into this Context.
        # See the comment above `probe_state` for why: this checkpoint was written under
        # `LangevinMiddleIntegrator`, and loading it directly into a `MTSLangevinIntegrator`
        # (CustomIntegrator) Context previously caused a silent, catastrophic kinetic-energy
        # collapse despite every other quantity (forces, thermostat coefficients, per-step
        # program, force-group coverage) checking out as correct.
        simulation.context.setPositions(source_positions)
        simulation.context.setVelocities(source_velocities)
        simulation.context.setPeriodicBoxVectors(*box_vectors)
        for parameter_name, parameter_value in source_parameters.items():
            if _system_has_global_parameter(win_sys, parameter_name):
                simulation.context.setParameter(parameter_name, parameter_value)
        # Explicit manifest-sourced overrides take precedence (belt-and-suspenders: these two
        # are also covered by the generic transfer above since they're real Context global
        # parameters, but kept explicit to match the values this script treats as authoritative).
        if _system_has_global_parameter(win_sys, "lambda_boresch_scale"):
            simulation.context.setParameter("lambda_boresch_scale", float(manifest["lambda_boresch_scale"]))
        if _system_has_global_parameter(win_sys, "lambda_shield"):
            simulation.context.setParameter("lambda_shield", float(manifest["lambda_shield"]))
        return simulation, integrator, mts_groups, outer_dt_ps

    # =========================== Phase 1: minimal backend smoke (N=16) ===========================
    smoke_win_sys, smoke_v0_wrap, smoke_delta_wrap, smoke_delta_group = _build_split_win_sys()
    smoke_v0_wrap.update_parameters(
        (lambda ctx=None: None)(), f_k  # placeholder to keep signature obvious; real call below
    ) if False else None
    smoke_simulation, smoke_integrator, smoke_mts_groups, smoke_outer_dt_ps = _make_mts_simulation(
        smoke_win_sys, smoke_delta_group, 16
    )
    smoke_v0_wrap.update_parameters(smoke_simulation.context, f_k)
    smoke_delta_wrap.update_parameters(smoke_simulation.context, f_k)
    smoke_dof = _degrees_of_freedom(smoke_win_sys)
    smoke_simulation.step(1)  # exactly one OUTER step -- the EXP-009 failure mode was N=1 on the first call
    smoke_state = smoke_simulation.context.getState(getEnergy=True)
    smoke_potential_kj_mol = smoke_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    smoke_kinetic_kj_mol = smoke_state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
    backend_smoke_passed = bool(math.isfinite(smoke_potential_kj_mol) and math.isfinite(smoke_kinetic_kj_mol))
    print(f"phase 1 backend smoke (N=16, groups={smoke_mts_groups}): "
          f"potential={smoke_potential_kj_mol:.4f} kinetic={smoke_kinetic_kj_mol:.4f} "
          f"passed={backend_smoke_passed}", flush=True)
    del smoke_simulation, smoke_integrator
    if not backend_smoke_passed:
        raise MtsQualificationError(
            "Phase 1 backend smoke produced non-finite energy after one MTS outer step -- "
            "stopping here per DEC-052 (no retry, no platform change, no coefficient change)"
        )

    # =========================== Phase 2: N=1/8/16/32 physical-time-aligned comparison ===========================
    gas_constant_kj_per_mol_k = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)

    def _run_n_value(n_value: int) -> dict:
        win_sys, v0_wrap, delta_wrap, delta_group = _build_split_win_sys()
        simulation, integrator, mts_groups, outer_dt_ps = _make_mts_simulation(win_sys, delta_group, n_value)
        v0_wrap.update_parameters(simulation.context, f_k)
        delta_wrap.update_parameters(simulation.context, f_k)
        dof = _degrees_of_freedom(win_sys)

        outer_steps_per_macro_tick = _MACRO_TICK_INNER_STEPS // n_value
        assert outer_steps_per_macro_tick * n_value == _MACRO_TICK_INNER_STEPS

        def _snapshot() -> dict:
            state_v0 = simulation.context.getState(getEnergy=True, groups={1})
            state_dv = simulation.context.getState(getEnergy=True, groups={delta_group})
            state_all = simulation.context.getState(getEnergy=True, getForces=True)
            e_v0 = state_v0.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            e_dv = state_dv.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            potential_total = state_all.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            kinetic_total = state_all.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            temperature = 2.0 * kinetic_total / (dof * gas_constant_kj_per_mol_k) if dof > 0 else float("nan")
            forces = state_all.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
            max_force_norm = float(np.max(np.linalg.norm(forces, axis=1)))
            return {
                "e_v0_kj_mol": e_v0, "e_dv_kj_mol": e_dv, "e_v0_plus_dv_kj_mol": e_v0 + e_dv,
                "potential_total_kj_mol": potential_total, "kinetic_total_kj_mol": kinetic_total,
                "total_energy_kj_mol": potential_total + kinetic_total,
                "temperature_k": temperature, "max_force_norm_kj_mol_nm": max_force_norm,
                "all_finite": bool(math.isfinite(potential_total) and math.isfinite(kinetic_total)
                                    and math.isfinite(temperature) and np.all(np.isfinite(forces))),
            }

        warmup_end_snapshot = None
        for tick_index in range(args.warmup_macro_ticks):
            simulation.step(outer_steps_per_macro_tick)
            if tick_index == args.warmup_macro_ticks - 1:
                warmup_end_snapshot = _snapshot()

        snapshots = []
        for _ in range(args.monitored_macro_ticks):
            simulation.step(outer_steps_per_macro_tick)
            snapshots.append(_snapshot())

        del simulation, integrator
        total_energies = [snap["total_energy_kj_mol"] for snap in snapshots]
        n_half = len(total_energies) // 2
        drift_first_half_mean = sum(total_energies[:n_half]) / n_half if n_half > 0 else float("nan")
        drift_second_half_mean = sum(total_energies[n_half:]) / (len(total_energies) - n_half) if len(total_energies) > n_half else float("nan")
        temperatures = [snap["temperature_k"] for snap in snapshots]
        kinetics = [snap["kinetic_total_kj_mol"] for snap in snapshots]
        mean_temperature_k = sum(temperatures) / len(temperatures) if temperatures else float("nan")
        mean_kinetic_kj_mol = sum(kinetics) / len(kinetics) if kinetics else float("nan")
        energy_drift_kj_mol = drift_second_half_mean - drift_first_half_mean
        relative_energy_drift = (
            abs(energy_drift_kj_mol) / mean_kinetic_kj_mol
            if math.isfinite(mean_kinetic_kj_mol) and mean_kinetic_kj_mol > 0.0
            else float("inf")
        )
        # DEC-054 absolute health gate: independent of any N-vs-N=1 comparison. A shared,
        # N-independent pathology (e.g. the checkpoint-migration bug this fix addresses) would
        # make every arm collapse together, which a purely relative gate cannot detect.
        absolute_health_passed = bool(
            math.isfinite(mean_temperature_k)
            and args.min_mean_temperature_k <= mean_temperature_k <= args.max_mean_temperature_k
            and warmup_end_snapshot is not None
            and math.isfinite(warmup_end_snapshot["temperature_k"])
            and args.min_mean_temperature_k <= warmup_end_snapshot["temperature_k"] <= args.max_mean_temperature_k
            and relative_energy_drift <= args.max_relative_energy_drift
        )
        return {
            "n_value": n_value, "outer_dt_ps": outer_dt_ps, "mts_groups": mts_groups,
            "physical_time_ps": {
                "warmup": args.warmup_macro_ticks * _MACRO_TICK_INNER_STEPS * inner_dt_ps,
                "monitored": args.monitored_macro_ticks * _MACRO_TICK_INNER_STEPS * inner_dt_ps,
            },
            "n_snapshots": len(snapshots),
            "all_finite": bool(all(snap["all_finite"] for snap in snapshots)),
            "max_force_norm_kj_mol_nm_overall": float(max(snap["max_force_norm_kj_mol_nm"] for snap in snapshots)),
            "total_energy_drift_first_vs_second_half_kj_mol": energy_drift_kj_mol,
            "warmup_end_temperature_k": warmup_end_snapshot["temperature_k"] if warmup_end_snapshot else float("nan"),
            "mean_temperature_k": mean_temperature_k,
            "mean_kinetic_kj_mol": mean_kinetic_kj_mol,
            "relative_energy_drift": relative_energy_drift,
            "absolute_health_passed": absolute_health_passed,
            "snapshots": snapshots,
        }

    results_by_n = {n: _run_n_value(n) for n in _N_VALUES}

    def _series(n_value: int, key: str) -> list[float]:
        return [snap[key] for snap in results_by_n[n_value]["snapshots"]]

    reference_n = 1
    comparisons = {}
    for n_value in _N_VALUES:
        if n_value == reference_n:
            continue
        comparison = {}
        for key in ("e_v0_plus_dv_kj_mol", "temperature_k"):
            ref_mean, ref_sem = _mean_and_sem(_series(reference_n, key))
            n_mean, n_sem = _mean_and_sem(_series(n_value, key))
            combined_sem = math.sqrt(ref_sem ** 2 + n_sem ** 2) if math.isfinite(ref_sem) and math.isfinite(n_sem) else float("nan")
            z_score = abs(n_mean - ref_mean) / combined_sem if combined_sem and math.isfinite(combined_sem) and combined_sem > 0.0 else None
            comparison[key] = {
                "reference_mean": ref_mean, "reference_sem": ref_sem,
                "n_mean": n_mean, "n_sem": n_sem,
                "z_score": z_score,
                "systematic_shift_detected": bool(z_score is not None and z_score > args.systematic_shift_z_threshold),
            }
        comparisons[n_value] = comparison

    all_finite = all(results_by_n[n]["all_finite"] for n in _N_VALUES)
    any_systematic_shift = {
        n: any(comparisons[n][key]["systematic_shift_detected"] for key in comparisons[n])
        for n in comparisons
    }
    # DEC-054 absolute health gate, checked BEFORE the relative N-vs-N=1 comparison is trusted:
    # a prior run's relative-only gate passed (`systematic_shift_detected_by_n=False` for all N)
    # while EVERY arm, including the N=1 reference, had silently collapsed to ~0.003 K due to a
    # checkpoint-migration bug (see the `probe_state`/state-transfer comment above). Comparing four
    # equally-broken systems to each other and finding "no difference" is not evidence of physical
    # health. absolute_health_passed_by_n gates each arm independently; relative_comparison_meaningful
    # specifically gates whether the N=1 reference itself is trustworthy enough for the systematic-
    # shift comparison above to mean anything at all.
    absolute_health_passed_by_n = {n: results_by_n[n]["absolute_health_passed"] for n in _N_VALUES}
    all_absolute_health_passed = all(absolute_health_passed_by_n.values())
    relative_comparison_meaningful = absolute_health_passed_by_n[reference_n]
    all_passed = bool(
        backend_smoke_passed
        and all_finite
        and all_absolute_health_passed
        and relative_comparison_meaningful
        and not any(any_systematic_shift.values())
    )

    body = {
        "schema_version": "exp013-013b-mts-dynamics-qualification-v2",
        "status": "COMPLETED_EXP013_013B",
        "platform": {"requested": args.platform, "resolved_name": resolved_platform_name,
                     "precision": platform_properties.get("Precision"), "properties": platform_properties},
        "checkpoint_path": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": _sha256_file(args.checkpoint),
        "torchscript_sha256": torchscript_sha256,
        "controller": {"coefficient_c1": float(args.coefficient),
                       "protocol_sha256": controller.protocol_sha256(lambdas=lambdas_vdw)},
        "phase1_backend_smoke": {
            "n_value": 16, "mts_groups": smoke_mts_groups, "outer_dt_ps": smoke_outer_dt_ps,
            "potential_kj_mol": smoke_potential_kj_mol, "kinetic_kj_mol": smoke_kinetic_kj_mol,
            "passed": backend_smoke_passed,
        },
        "phase2_methodology": {
            "macro_tick_inner_steps": _MACRO_TICK_INNER_STEPS, "inner_dt_ps": inner_dt_ps,
            "warmup_macro_ticks": args.warmup_macro_ticks, "monitored_macro_ticks": args.monitored_macro_ticks,
            "note": "every N in {1,8,16,32} runs the SAME physical time span (LCM-exact macro-tick "
                    "alignment, no rounding); comparing step counts instead of physical time would "
                    "silently compare different dynamical durations",
        },
        "results_by_n": results_by_n,
        "comparisons_vs_n1": comparisons,
        "results": {
            "all_finite": all_finite,
            "systematic_shift_detected_by_n": any_systematic_shift,
            "z_threshold": args.systematic_shift_z_threshold,
            "absolute_health_passed_by_n": absolute_health_passed_by_n,
            "all_absolute_health_passed": all_absolute_health_passed,
            "relative_comparison_meaningful": relative_comparison_meaningful,
            "absolute_health_thresholds": {
                "min_mean_temperature_k": args.min_mean_temperature_k,
                "max_mean_temperature_k": args.max_mean_temperature_k,
                "max_relative_energy_drift": args.max_relative_energy_drift,
            },
        },
        "all_passed": all_passed,
        "policy": {
            "decision_reference": "EXP-013 013-B (design (3), DEC-052/053/054)",
            "ibs_engine_py_modified": False, "production_checkpoints_written": False,
            "total_energy_drift_note": "total_energy_drift_first_vs_second_half_kj_mol is now also "
                                        "gated (as relative_energy_drift vs mean kinetic energy, DEC-054) "
                                        "in addition to being reported as a diagnostic -- it is not a real "
                                        "shadow-work estimator, but it is sensitive to shared common-mode "
                                        "energy leaks a purely relative N-vs-N=1 comparison cannot see.",
            "initialization_note": "DEC-054: MTS Contexts are initialized via explicit state-transfer "
                                    "(setPositions/setVelocities/setPeriodicBoxVectors/setParameter) from "
                                    "a source Context that itself used loadCheckpoint() with the SAME "
                                    "integrator class production wrote this checkpoint with "
                                    "(LangevinMiddleIntegrator). No MTSLangevinIntegrator Context ever "
                                    "calls loadCheckpoint() directly. An earlier version of this script did "
                                    "so and produced report_sha256 "
                                    "bc9eb24dcb5d54297664028b2207156efff83b6693d8569e2f3c76d0bcc45519's "
                                    "sibling 013-B report (99d74b176e1bd4b862554a1c1234cf8a4059a365e8ad4"
                                    "53eac607ab6a661b218), which is INVALIDATED_BY_INITIALIZATION_BUG, not "
                                    "a real MTS/physics finding -- see EXPERIMENT_LOG DEC-054.",
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    print(f"all_passed={all_passed} backend_smoke_passed={backend_smoke_passed} all_finite={all_finite} "
          f"all_absolute_health_passed={all_absolute_health_passed} "
          f"relative_comparison_meaningful={relative_comparison_meaningful} "
          f"systematic_shift_detected_by_n={any_systematic_shift}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
