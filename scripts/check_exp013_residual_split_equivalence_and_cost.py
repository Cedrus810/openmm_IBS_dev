#!/usr/bin/env python
"""EXP-013 013-A (design (3), DEC-052): exact-residual-split equivalence + cost.

Two independent checks, in the order DEC-052 fixed (equivalence first, cost
second -- NOT ESS, and NOT "does MTS run" until both of these pass):

1. Numerical equivalence: `V_0 + dV === V_*` is a construction identity
   (`dV := V_* - V_0`), so this checks "did I implement `dV` as a
   `CustomCVForce` correctly", not new physics. Builds TWO Systems from the
   real, hash-verified `hard_window0` win_sys on the SAME checkpoint frame:
   (a) `fused`: Group 1 replaced by `OuterLambdaIBSBiasForce` (the
   wiring-smoke-validated `V_*`, DEC-047/048); (b) `split`: Group 1 left as
   the UNMODIFIED classical `IBSBiasForce` (`V_0`) plus a new
   `OuterLambdaResidualBiasForce` (`dV`) on its own group. Compares
   `E_fused` vs `E_v0 + E_dv` and the corresponding per-particle forces.
2. Matched-path cost of `dV`: extends DEC-050's methodology (same win_sys,
   same `Simulation.step()` call path, no cross-harness subtraction) with a
   `delta_v_residual` variant (classical `V_0` unmodified + `dV` on its own
   group) timed against the SAME `baseline` (no student anything) already
   established. `dV`'s own per-call cost must be measured directly, not
   assumed equal to the student's TorchForce cost alone -- `dV` also has to
   independently recompute the classical `cv_k_int`/`cv_k_rest` CVs inside
   its own `CustomCVForce` inner Context (cannot reuse what the fast `V_0`
   group already computed), and that redundant cost is real.

If a CUDA_ERROR-class exception is raised anywhere in this script (the
EXP-009 failure mode), this reports it verbatim and stops -- no retry, no
platform change, no coefficient change (DEC-052 policy).
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

_STEPS_PER_UPDATE = 500  # matches production's own steps_per_update, same convention as D3/D4/DEC-050


class ResidualSplitCheckError(RuntimeError):
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
    parser.add_argument("--checkpoint", required=True, help="frozen direct_gap .pt candidate (DEC-045)")
    parser.add_argument("--coefficient", type=float, default=0.5, help="frozen c1 (DEC-045/047), do not retune")
    parser.add_argument("--max-abs-coefficient", type=float, default=1.0)
    parser.add_argument("--max-abs-basis-energy-kj-mol", type=float, default=50.0)
    parser.add_argument("--max-abs-path-energy-kj-mol", type=float, default=25.0)
    parser.add_argument("--max-force-norm-kj-mol-nm", type=float, default=500.0)
    parser.add_argument("--equivalence-relative-tolerance", type=float, default=1e-3,
                         help="fused vs split are two different CustomCVForce expression graphs computing the "
                              "same algebraic identity under CUDA mixed precision -- relative tolerance, not "
                              "machine-precision (that would spuriously fail on float32-level GPU arithmetic "
                              "noise, the same mistake DEC-046 fixed for the wiring smoke)")
    parser.add_argument("--dec049-target-ratio", type=float, default=1.10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-chunks", type=int, default=3)
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
    from outer_lambda_neural_basis import (  # noqa: E402
        NeuralBasisModelSpec,
        NeuralPathSafety,
        OuterLambdaController,
        OuterLambdaIBSBiasForce,
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
            raise ResidualSplitCheckError(f"required real production artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_payload = json.loads(stage_protocol_path.read_text(encoding="utf-8"))["protocol_key"]["payload"]
    ibs_state = json.loads(ibs_state_path.read_text(encoding="utf-8"))
    target_temperature_k = float(manifest["temperature_K"])
    lambdas_vdw = [float(value) for value in manifest["lambdas_vdw"]]
    n_states = len(lambdas_vdw)
    prefix = ibs_state["prefix"]
    f_k = np.asarray(ibs_state["f_k"], dtype=float)

    system_xml_text = system_xml_path.read_text(encoding="utf-8")
    if _sha256_text(system_xml_text) != stage_payload["system_xml_sha256"]:
        raise ResidualSplitCheckError("system_native.xml SHA-256 does not match stage2 protocol record")
    base_system = XmlSerializer.deserialize(system_xml_text)
    topology = app.PDBxFile(str(topology_cif_path)).topology
    stale_box_vectors = unit.Quantity(np.load(box_vectors_path), unit.nanometer)
    alchemical_params = ACESoftcorePotential.from_dict(stage_payload["aces_softcore_params"])

    resolved_platform_name, platform_properties = _build_platform_properties(args.platform)
    platform = openmm.Platform.getPlatformByName(resolved_platform_name)

    # DEC-039/DEC-041 two-pass box derivation: reuse, not re-derive.
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

    # ---- student TorchForce, a_k=1.0/offset=0.0 (same convention as wiring smoke/pilot) ----
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("variant") != "direct_gap":
        raise ResidualSplitCheckError(f"--checkpoint variant={payload.get('variant')!r}, only direct_gap is qualified")
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
        name="local_residual_student_hard_window0_exp013_013a", backend="torchforce",
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

    def _build_fused_win_sys():
        """V_* : Group 1 replaced by OuterLambdaIBSBiasForce (wiring-smoke design, DEC-047/048)."""
        win_sys, original_ibs_wrap = _build_baseline_win_sys()
        group1_indices = [i for i in range(win_sys.getNumForces()) if win_sys.getForce(i).getForceGroup() == 1]
        if len(group1_indices) != 1:
            raise ResidualSplitCheckError(f"expected exactly one force group 1, found {len(group1_indices)}")
        win_sys.removeForce(group1_indices[0])
        student_force = build_torchforce_from_spec(basis_spec)
        new_wrapper = OuterLambdaIBSBiasForce(controller, lambdas_vdw, target_temperature_k, [student_force], prefix=prefix)
        new_wrapper.lj_tail_lrc_coeff_kj_mol = original_ibs_wrap.lj_tail_lrc_coeff_kj_mol
        for k in range(n_states):
            int_cv = XmlSerializer.deserialize(original_ibs_wrap._int_cv_force_xmls[k])
            new_wrapper.addCollectiveVariable(f"cv_{k}_int", int_cv)
            new_wrapper.addCollectiveVariable(f"cv_{k}_rest", openmm.CustomExternalForce("0"))
        win_sys.addForce(new_wrapper.get_force())
        return win_sys, new_wrapper

    def _build_split_win_sys():
        """V_0 (unmodified, Group 1) + dV (OuterLambdaResidualBiasForce, own group)."""
        win_sys, original_ibs_wrap = _build_baseline_win_sys()  # Group 1 left untouched -- this IS V_0
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
            raise ResidualSplitCheckError("no free OpenMM force group (0-31) left for dV")
        delta_wrapper.setForceGroup(delta_group)
        win_sys.addForce(delta_wrapper.get_force())
        return win_sys, original_ibs_wrap, delta_wrapper, delta_group

    def _make_simulation(win_sys):
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
        return simulation, integrator

    # =========================== check 1: numerical equivalence ===========================
    fused_win_sys, fused_wrapper = _build_fused_win_sys()
    fused_simulation, fused_integrator = _make_simulation(fused_win_sys)
    fused_wrapper.update_parameters(fused_simulation.context, f_k)
    fused_state = fused_simulation.context.getState(getEnergy=True, getForces=True, groups={1})
    e_fused = fused_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f_fused = np.asarray(fused_state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    del fused_simulation, fused_integrator

    split_win_sys, split_v0_wrapper, split_delta_wrapper, delta_group = _build_split_win_sys()
    split_simulation, split_integrator = _make_simulation(split_win_sys)
    split_v0_wrapper.update_parameters(split_simulation.context, f_k)
    split_delta_wrapper.update_parameters(split_simulation.context, f_k)
    state_v0 = split_simulation.context.getState(getEnergy=True, getForces=True, groups={1})
    state_dv = split_simulation.context.getState(getEnergy=True, getForces=True, groups={delta_group})
    e_v0 = state_v0.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    e_dv = state_dv.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f_v0 = np.asarray(state_v0.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    f_dv = np.asarray(state_dv.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    e_split_total = e_v0 + e_dv
    f_split_total = f_v0 + f_dv
    del split_simulation, split_integrator

    energy_absolute_error = abs(e_split_total - e_fused)
    energy_relative_error = energy_absolute_error / max(abs(e_fused), 1e-12)
    force_absolute_error = float(np.max(np.abs(f_split_total - f_fused)))
    force_scale = float(np.max(np.abs(f_fused))) if np.max(np.abs(f_fused)) > 0.0 else 1e-12
    force_relative_error = force_absolute_error / force_scale
    equivalence_passed = bool(
        energy_relative_error <= args.equivalence_relative_tolerance
        and force_relative_error <= args.equivalence_relative_tolerance
    )
    print(f"equivalence: E_fused={e_fused:.6f} E_v0+E_dv={e_split_total:.6f} "
          f"(E_v0={e_v0:.6f}, E_dv={e_dv:.6f}) "
          f"energy_rel_err={energy_relative_error:.3e} force_rel_err={force_relative_error:.3e} "
          f"passed={equivalence_passed}", flush=True)

    # =========================== check 2: matched-path cost of dV ===========================
    def _time_variant(label: str, win_sys) -> dict:
        simulation, integrator = _make_simulation(win_sys)
        gpu_before = _gpu_memory_mib()
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
            print(f"  [{label}] repeat {repeat_index}: {ms_per_step:.4f} ms/step", flush=True)
        gpu_after = _gpu_memory_mib()
        del simulation, integrator
        return {
            "repeats_ms_per_step": ms_per_step_values,
            "ms_per_step_median": _percentile(ms_per_step_values, 0.5),
            "ms_per_step_p95": _percentile(ms_per_step_values, 0.95),
            "gpu_memory_mib": {"before": gpu_before, "after": gpu_after},
        }

    baseline_win_sys, _baseline_wrap = _build_baseline_win_sys()
    baseline_timing = _time_variant("baseline", baseline_win_sys)
    split_win_sys_for_timing, _v0_wrap2, _dv_wrap2, _dv_group2 = _build_split_win_sys()
    delta_v_timing = _time_variant("delta_v_residual", split_win_sys_for_timing)

    baseline_median = baseline_timing["ms_per_step_median"]
    delta_v_delta = delta_v_timing["ms_per_step_median"] - baseline_median
    budget_ms_per_step_n = {n: (args.dec049_target_ratio - 1.0) * baseline_median * n for n in (8, 16, 32)}
    n_feasibility = {n: bool(delta_v_delta <= budget) for n, budget in budget_ms_per_step_n.items()}

    all_passed = bool(equivalence_passed and any(n_feasibility.values()))

    body = {
        "schema_version": "exp013-013a-residual-split-equivalence-and-cost-v1",
        "status": "COMPLETED_EXP013_013A",
        "platform": {"requested": args.platform, "resolved_name": resolved_platform_name,
                     "precision": platform_properties.get("Precision"), "properties": platform_properties},
        "checkpoint_path": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": _sha256_file(args.checkpoint),
        "checkpoint_held_out_run_id": payload.get("held_out_run_id"), "checkpoint_seed": payload.get("seed"),
        "torchscript_sha256": torchscript_sha256,
        "controller": {"coefficient_c1": float(args.coefficient),
                       "protocol_sha256": controller.protocol_sha256(lambdas=lambdas_vdw)},
        "equivalence": {
            "e_fused_kj_mol": e_fused, "e_v0_kj_mol": e_v0, "e_dv_kj_mol": e_dv,
            "e_split_total_kj_mol": e_split_total,
            "energy_absolute_error_kj_mol": energy_absolute_error, "energy_relative_error": energy_relative_error,
            "force_absolute_error_kj_mol_nm": force_absolute_error, "force_relative_error": force_relative_error,
            "tolerance_relative": args.equivalence_relative_tolerance,
            "passed": equivalence_passed,
            "note": "V_0+dV===V_* is a construction identity (dV:=V_*-V_0); this checks whether the "
                    "OuterLambdaResidualBiasForce CustomCVForce implementation is correct, not new physics. "
                    "Relative tolerance because fused/split are two different expression graphs under CUDA "
                    "mixed precision, not because the identity itself is approximate.",
        },
        "cost": {
            "baseline": baseline_timing, "delta_v_residual": delta_v_timing,
            "baseline_ms_per_step_median": baseline_median,
            "delta_v_delta_ms_per_step": delta_v_delta,
            "dec049_target_ratio": args.dec049_target_ratio,
            "budget_ms_per_step_by_N": budget_ms_per_step_n,
            "feasible_by_N": n_feasibility,
            "note": "delta_v_delta_ms_per_step is dV's real per-call cost (redundant classical-CV "
                    "recomputation + student TorchForce call + double log-sum-exp math), measured via the "
                    "same matched-path Simulation.step() methodology DEC-050 used -- not assumed equal to "
                    "the student's TorchForce cost alone.",
        },
        "all_passed": all_passed,
        "policy": {
            "decision_reference": "EXP-013 013-A (design (3), DEC-052)",
            "ibs_engine_py_modified": False, "production_checkpoints_written": False,
            "note": "if equivalence fails, the OuterLambdaResidualBiasForce implementation has a bug -- fix "
                    "before touching cost. if equivalence passes but no N in {8,16,32} is cost-feasible, "
                    "design (3) fails on engineering economics and DEC-052's order moves to design (1) "
                    "(whole fused Group-1 slow), not a retry of (3).",
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    print(f"all_passed={all_passed} equivalence_passed={equivalence_passed} "
          f"baseline_median={baseline_median:.4f} delta_v_delta={delta_v_delta:.4f} "
          f"feasible_by_N={n_feasibility}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
