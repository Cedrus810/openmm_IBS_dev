#!/usr/bin/env python
"""EXP-012 WP-5A step 3: baseline-MM vs MM+student paired-reseed pilot.

Design (frozen per user directive, 2026-08-07, post WP-5A step 1/2 closure
DEC-047):

- "配对重抽" (paired reseed), NOT three independently equilibrated
  configurations: there is only one real production checkpoint on disk for
  `hard_window0`. Each of 3 repeats redraws velocities from that SAME
  checkpoint's positions/box via a from-scratch, numpy-seeded Maxwell-Boltzmann
  draw (`_draw_maxwell_boltzmann_velocities`, injected via
  `Context.setVelocities()`) and uses a fresh `LangevinMiddleIntegrator` seeded
  with `langevin_seed`. Velocities are drawn ONCE per repeat in Python and the
  SAME array is injected into both arms -- NOT via OpenMM's own
  `Context.setVelocitiesToTemperature(seed)`, which the first real run of this
  script proved is NOT bit-identical across two differently-composed Systems
  given the same seed (CUDA's internal atom reordering for memory locality can
  depend on the Force graph, so baseline's `IBSBiasForce` vs student's
  `OuterLambdaIBSBiasForce` end up drawing different external-index velocities
  even from an identical internal RNG stream). Within one repeat, baseline and
  student use the IDENTICAL injected velocity array and langevin_seed
  (verified via an exact round-trip check, not just assumed) so any difference
  between the two arms is attributable to the added Force, not to a different
  noise realization. Across the 3 repeats, both the velocity draw and the
  Langevin seed differ. This is explicitly "same starting position,
  random-dynamics repeats", not "three independent equilibrium samples" -- do
  not read more independence into it than that.
- Fixed burn-in: 10,000 steps run and NOT counted into any statistic
  (energy/bias/base history, ESS, autocorrelation), only into `gpu_hours`
  (a real production window would also pay this cost). Then 50,000 monitored
  production steps, sampled every 500 steps (matching production's own
  `steps_per_update` granularity) -> 100 frames/repeat/arm.
- Frozen production candidate (DEC-045/047, do not touch here): checkpoint
  `hard_window0_run1__direct_gap__seed0.pt`, `c1=0.5`, sin2 envelope, frozen
  already-converged production `f_k` reused unmodified for BOTH arms (no
  relearning -- keeps this a single-variable comparison).

Metric naming (frozen per user directive -- these are NOT the literal
textbook quantities and must never be reported as such):

- `mixture_ess_proxy` / `mixture_ess_proxy_per_gpu_hour`: IBS's own
  established substitute for "mutual overlap" when only one reference
  mixture distribution is actually sampled per window (see
  `ibs_engine.py:13305-13319` and `_ibs_reweighting_quality_diagnostics`,
  `ESS_GATE_PROTOCOL_VERSION=3`) -- the minimum-over-states
  `mixture_ess` (compute_effective_sample_number-style reweighting quality),
  NOT a literal `pymbar.compute_overlap()` matrix (which degenerates for
  IBS's single-sampled-row augmented matrix). `is_literal_pymbar_overlap`
  is recorded as `False` in every report to make this impossible to miss.
- `dominant_component_switch_count` / `dominant_component_autocorrelation` /
  `endpoint_proxy_traversals`: derived from a self-constructed per-frame
  "which state currently has the largest IBS mixture weight" argmin-energy
  label series (`argmin_k(energy_buffer[k] - f_k[k])`, gauge-invariant to
  any frame-constant shift). This is NOT a physical replica trajectory (IBS
  runs one simulation per window, not REMD) -- `is_physical_replica_round_trip`
  is recorded as `False` in every report. `endpoint_proxy_traversals` counts
  arrivals at this window's own genuine global endpoint component (k=0,
  lambda_vdw=1.0), not a full round trip to a second endpoint (window 0 does
  not contain the vdw stage's other endpoint, lambda_vdw=0.0 -- see the D3/D4
  wiring-smoke endpoint-scope note, same caveat applies here).
- Delta-G per arm/repeat comes from `ibs_engine.solve_stage_integrated` fed a
  single-window `window_outputs` list built in-memory from this repeat's own
  collected `u_kn`/`bias_energies`/`base_energies`/`f_k` (same schema
  `IBSWindowManagerDualLambda.get_stage_data_for_analysis` produces from disk,
  reused here without touching disk) -- this is a real, already-audited
  TMBAR entry point, not a new estimator.

Pilot promotion rule (provisional, NOT a sealed WP-5A gate -- WP-5A's real
gate still needs literal BAR/MBAR overlap from a full reduced-potential
ledger if IBS's sampling scheme is ever changed to support it, and a formal
`not_applicable` verdict for round-trip if it stays single-reference; this
script does not attempt either of those, per user directive):

1. >=2/3 paired repeats show `mixture_ess_proxy_per_gpu_hour[student] >
   mixture_ess_proxy_per_gpu_hour[baseline]`.
2. Median of the 3 per-repeat improvements is positive.
3. Proxy switching/autocorrelation not CONSISTENTLY worse for student (not
   unanimously worse on both `dominant_component_switch_count` (lower) AND
   `dominant_component_autocorrelation` (higher) across all 3 repeats).
4. Ledger closure, all-finite, and Delta-G consistency (|ΔG_student -
   ΔG_baseline| / combined_sigma <= --delta-g-consistency-z-threshold) hold
   for every repeat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.student import build_local_residual_student  # noqa: E402
from local_residual.student_deploy import build_deployable_student_module, export_torchscript  # noqa: E402


class PilotError(RuntimeError):
    """A checkpoint/frame/comparison failed a fail-closed contract check."""


def _raw_or_none(value) -> float | None:
    """JSON-safe passthrough: None if `value` is missing or not a finite float."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="output_lrc_fix")
    parser.add_argument("--stage-type", default="vdw", choices=["vdw"])
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--checkpoint", required=True, help="frozen direct_gap .pt candidate (DEC-045)")
    parser.add_argument("--coefficient", type=float, default=0.5, help="frozen c1 (DEC-045/047)")
    parser.add_argument("--max-abs-coefficient", type=float, default=1.0)
    parser.add_argument("--max-abs-basis-energy-kj-mol", type=float, default=50.0)
    parser.add_argument("--max-abs-path-energy-kj-mol", type=float, default=25.0)
    parser.add_argument("--max-force-norm-kj-mol-nm", type=float, default=500.0)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--burn-in-steps", type=int, default=10_000)
    parser.add_argument("--production-steps", type=int, default=50_000)
    parser.add_argument("--steps-per-chunk", type=int, default=500, help="matches production's own steps_per_update")
    parser.add_argument("--base-velocity-seed", type=int, default=40_000)
    parser.add_argument("--base-langevin-seed", type=int, default=50_000)
    parser.add_argument("--delta-g-consistency-z-threshold", type=float, default=2.0)
    parser.add_argument("--torchscript-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.n_repeats < 3:
        parser.error("--n-repeats must be at least 3 (pilot promotion rule needs a 2/3 majority)")
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
        IBSSampler,
        _build_platform_properties,
        _gpu_memory_mib,
        _ibs_reweighting_quality_diagnostics,
        _system_has_global_parameter,
        build_ibs_dual_system,
        solve_stage_integrated,
    )
    from outer_lambda_neural_basis import (  # noqa: E402
        NeuralBasisModelSpec,
        NeuralPathSafety,
        OuterLambdaController,
        OuterLambdaIBSBiasForce,
        IBSSamplerNeuralPathAdapter,
        build_torchforce_from_spec,
        count_discrete_transitions,
        integrated_autocorrelation_time,
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
            raise PilotError(f"required real production artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_payload = json.loads(stage_protocol_path.read_text(encoding="utf-8"))["protocol_key"]["payload"]
    ibs_state = json.loads(ibs_state_path.read_text(encoding="utf-8"))
    target_temperature_k = float(manifest["temperature_K"])
    lambdas_vdw = [float(value) for value in manifest["lambdas_vdw"]]
    lambdas_coul = [float(value) for value in manifest["lambdas_coul"]]
    n_states = len(lambdas_vdw)
    prefix = ibs_state["prefix"]
    f_k = np.asarray(ibs_state["f_k"], dtype=float)
    if f_k.shape[0] != n_states:
        raise PilotError("ibs_state f_k length does not match manifest K")
    kt_kj_per_mol = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin) * target_temperature_k

    system_xml_text = system_xml_path.read_text(encoding="utf-8")
    if _sha256_text(system_xml_text) != stage_payload["system_xml_sha256"]:
        raise PilotError("system_native.xml SHA-256 does not match stage2 protocol record")
    base_system = XmlSerializer.deserialize(system_xml_text)
    topology = app.PDBxFile(str(topology_cif_path)).topology
    stale_box_vectors = unit.Quantity(np.load(box_vectors_path), unit.nanometer)
    alchemical_params = ACESoftcorePotential.from_dict(stage_payload["aces_softcore_params"])

    resolved_platform_name, platform_properties = _build_platform_properties(args.platform)
    platform = openmm.Platform.getPlatformByName(resolved_platform_name)

    # DEC-039/DEC-041 two-pass box derivation: reuse, not re-derive.
    probe_win_sys, _probe_ibs = build_ibs_dual_system(
        base_system, topology, stage_payload["ligand_indices"],
        lambdas_coul, lambdas_vdw, alchemical_params,
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
            lambdas_coul, lambdas_vdw, alchemical_params,
            potential_type=stage_payload["potential_type"], restraint_params=stage_payload["boresch_params"],
            temperature=target_temperature_k * unit.kelvin, prefix=prefix,
            box_vectors=box_vectors, reference_positions=None,
            dispersion_protocol="legacy_uniform_density_lrc", environment_type="soluble",
        )

    # ---- student TorchForce, a_k=1.0/offset=0.0 (same convention as the wiring smoke, DEC-047) ----
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("variant") != "direct_gap":
        raise PilotError(f"--checkpoint variant={payload.get('variant')!r}, only direct_gap is qualified")
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
        name="local_residual_student_hard_window0_wp5a_pilot", backend="torchforce",
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
    a_k_per_state = [row[0] for row in controller.coefficient_matrix(lambdas_vdw)]

    def _build_student_win_sys():
        win_sys, original_ibs_wrap = _build_baseline_win_sys()
        group1_indices = [i for i in range(win_sys.getNumForces()) if win_sys.getForce(i).getForceGroup() == 1]
        if len(group1_indices) != 1:
            raise PilotError(f"expected exactly one force group 1 (IBSBiasForce), found {len(group1_indices)}")
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

    def _draw_maxwell_boltzmann_velocities(masses_dalton: "np.ndarray", velocity_seed: int) -> "np.ndarray":
        """Draw Maxwell-Boltzmann velocities ourselves (numpy, seeded), rather than trusting
        OpenMM's own Context.setVelocitiesToTemperature(seed) to be bit-identical across two
        DIFFERENTLY-COMPOSED Systems given the same seed. It is not: the CUDA platform
        internally reorders atoms for memory locality, and that reordering can depend on the
        Force graph (baseline's Group-1 IBSBiasForce vs student's OuterLambdaIBSBiasForce have
        different internal CV Force XML) -- confirmed empirically (first real run of this
        script hit exactly this: identical velocity_seed, non-identical drawn velocities).
        Computing the array once in Python and injecting it via Context.setVelocities() on
        BOTH arms sidesteps the assumption entirely instead of trying to out-guess CUDA's
        internal reordering. Not KE-rescaled to the exact target temperature -- the 10,000-step
        Langevin burn-in (this script's own design, not skipped) fully re-thermalizes from any
        such initial-condition detail, so exact KE-matching would add complexity without adding
        anything the burn-in doesn't already guarantee.
        """
        rng = np.random.default_rng(int(velocity_seed))
        kb_kj_per_mol_k = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
        # kJ/mol == amu*(nm/ps)^2 in OpenMM's internal unit system, so this sigma is directly in nm/ps.
        sigma_nm_ps = np.sqrt(kb_kj_per_mol_k * target_temperature_k / masses_dalton)
        velocities = rng.standard_normal(size=(masses_dalton.shape[0], 3)) * sigma_nm_ps[:, None]
        # Zero net linear momentum (standard MD initialization practice; the System's own
        # CMMotionRemover -- present in Group 0 -- would remove any residual drift during
        # subsequent dynamics regardless, this just avoids injecting an avoidable artifact).
        total_mass = float(np.sum(masses_dalton))
        com_velocity = np.sum(velocities * masses_dalton[:, None], axis=0) / total_mass
        velocities -= com_velocity[None, :]
        return velocities

    def _run_arm(win_sys, ibs_wrapper, *, is_student: bool, velocities_nm_ps: "np.ndarray", langevin_seed: int):
        integrator = openmm.LangevinMiddleIntegrator(
            target_temperature_k * unit.kelvin, manifest["friction_per_ps"] / unit.picosecond,
            manifest["step_size_ps"] * unit.picosecond,
        )
        integrator.setConstraintTolerance(1e-3)
        if hasattr(integrator, "setRemoveCMMotion"):
            integrator.setRemoveCMMotion(True)
        simulation = app.Simulation(topology, win_sys, integrator, platform, platform_properties)
        simulation.loadCheckpoint(str(checkpoint_path))  # positions + box only; velocities overwritten next
        # setRandomNumberSeed() AFTER loadCheckpoint(), same reasoning as the D4/wiring-smoke
        # scripts: a checkpoint may embed its own saved stochastic state, which loadCheckpoint()
        # could otherwise silently restore over our chosen seed.
        integrator.setRandomNumberSeed(int(langevin_seed))
        simulation.context.setVelocities(unit.Quantity(velocities_nm_ps, unit.nanometer / unit.picosecond))
        readback_velocities_nm_ps = simulation.context.getState(getVelocities=True).getVelocities(
            asNumpy=True
        ).value_in_unit(unit.nanometer / unit.picosecond)
        if not np.array_equal(readback_velocities_nm_ps, velocities_nm_ps):
            raise PilotError(
                "Context.setVelocities()/getState(getVelocities=True) round-trip is not exact -- "
                "cannot guarantee the intended paired-reseed velocities were actually applied"
            )
        if _system_has_global_parameter(win_sys, "lambda_boresch_scale"):
            simulation.context.setParameter("lambda_boresch_scale", float(manifest["lambda_boresch_scale"]))
        if _system_has_global_parameter(win_sys, "lambda_shield"):
            simulation.context.setParameter("lambda_shield", float(manifest["lambda_shield"]))
        ibs_wrapper.update_parameters(simulation.context, f_k)

        sampler = IBSSampler(
            simulation.context, n_states, target_temperature_k * unit.kelvin, prefix=prefix, ibs_wrapper=ibs_wrapper,
        )
        adapter = IBSSamplerNeuralPathAdapter(sampler, controller, lambdas_vdw, ibs_wrapper) if is_student else None
        collect = adapter.collect_energies if adapter is not None else sampler.collect_energies

        gpu_before = _gpu_memory_mib()
        started = time.perf_counter()
        simulation.step(args.burn_in_steps)  # uncounted stats, counted wall-clock

        dominant_k_labels = []
        n_chunks = args.production_steps // args.steps_per_chunk
        ledger_closed_all = True
        all_finite_all = True
        for _ in range(n_chunks):
            simulation.step(args.steps_per_chunk)
            before_len = len(sampler.energy_buffer)
            collect()
            after_len = len(sampler.energy_buffer)
            frame_ledger_closed = (
                after_len == before_len + 1
                and len(sampler.energy_history) == after_len
                and len(sampler.bias_history) == after_len
                and len(sampler.base_energy_history) == after_len
            )
            if is_student:
                frame_ledger_closed = (
                    frame_ledger_closed
                    and len(adapter.neural_path_energy_history) == after_len
                    and len(adapter.basis_energy_history) == after_len
                )
            ledger_closed_all = ledger_closed_all and frame_ledger_closed
            latest_target = np.asarray(sampler.energy_history[-1], dtype=float)
            latest_bias_cv = np.asarray(sampler.energy_buffer[-1], dtype=float)
            frame_finite = bool(np.all(np.isfinite(latest_target)) and np.all(np.isfinite(latest_bias_cv))
                                 and math.isfinite(sampler.bias_history[-1]) and math.isfinite(sampler.base_energy_history[-1]))
            all_finite_all = all_finite_all and frame_finite
            # Gauge-invariant to any frame-constant shift (e.g. sampler.e_offset), so this
            # is valid for BOTH arms with the SAME formula: student's energy_buffer already
            # includes the neural term (compose_ibs_energy_frame folds it into bias_cv_
            # state_energies_kj_mol, exactly what drives the real Group-1 discriminant).
            dominant_k_labels.append(int(np.argmin(latest_bias_cv - f_k)))
        elapsed_seconds = time.perf_counter() - started
        gpu_after = _gpu_memory_mib()

        u_kn = np.asarray(sampler.energy_history, dtype=np.float64).T  # (K, N)
        bias_kj = np.asarray(sampler.bias_history, dtype=np.float64)
        base_kj = np.asarray(sampler.base_energy_history, dtype=np.float64)

        quality = _ibs_reweighting_quality_diagnostics(u_kn, bias_kj, f_k, kt_kj_per_mol)
        if quality.get("error") is not None:
            raise PilotError(f"mixture ESS diagnostic failed: {quality['error']}")
        mixture_ess_per_state = quality["mixture_ess"]
        mixture_ess_proxy = float(min(mixture_ess_per_state))

        window_output = {
            "window_index": args.window_index, "window_label": f"window_{args.window_index}",
            "window_range": [0, n_states], "u_kn": u_kn, "bias_energies": bias_kj, "base_energies": base_kj,
            "lambda_indices": list(range(n_states)), "lambdas_coul": lambdas_coul, "lambdas_vdw": lambdas_vdw,
            "f_k": f_k, "sampled_distribution_row": 0,
        }
        delta_g_result = solve_stage_integrated([window_output], kt_kj_per_mol, stage_name=f"{args.stage_type}_pilot")

        switch_count = count_discrete_transitions(dominant_k_labels)
        autocorrelation = integrated_autocorrelation_time([float(x) for x in dominant_k_labels])
        endpoint_traversals = sum(
            1 for i in range(1, len(dominant_k_labels))
            if dominant_k_labels[i] == 0 and dominant_k_labels[i - 1] != 0
        )

        del simulation, integrator
        return {
            "elapsed_seconds": elapsed_seconds,
            "gpu_hours": elapsed_seconds / 3600.0,
            "gpu_memory_mib": {"before": gpu_before, "after": gpu_after},
            "velocities_nm_ps_first_atom": velocities_nm_ps[0].tolist(),
            "ledger_closed": bool(ledger_closed_all),
            "all_finite": bool(all_finite_all),
            "n_frames": len(dominant_k_labels),
            "mixture_ess_per_state": mixture_ess_per_state,
            "mixture_ess_proxy": mixture_ess_proxy,
            "mixture_ess_proxy_per_gpu_hour": mixture_ess_proxy / (elapsed_seconds / 3600.0),
            "is_literal_pymbar_overlap": False,
            "raw_ess_per_state": quality["raw_ess"],
            # None (not float("nan")/float("inf")) whenever the value is missing or itself
            # non-finite -- json.dumps(allow_nan=False) at the very end of this whole 3-repeat
            # run rejects Infinity/NaN outright, and finding that out only after ~15 minutes of
            # GPU time (twice, now) is exactly the failure mode _raw_or_none guards against.
            "delta_g_kj_mol": _raw_or_none(delta_g_result.get("total_delta_G")),
            "delta_g_uncertainty_kj_mol": _raw_or_none(delta_g_result.get("total_error")),
            "delta_g_converged": bool(delta_g_result.get("converged", False)),
            "dominant_component_labels": dominant_k_labels,
            "dominant_component_switch_count": switch_count,
            "dominant_component_autocorrelation": autocorrelation,
            "is_physical_replica_round_trip": False,
            "endpoint_proxy_traversals": endpoint_traversals,
        }

    # Particle count/masses are identical for baseline and student (both built from the same
    # base_system particles; the student swap only touches Forces, never adds/removes particles).
    particle_masses_dalton = np.asarray(
        [base_system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(base_system.getNumParticles())],
        dtype=np.float64,
    )

    repeats = []
    for repeat_index in range(args.n_repeats):
        velocity_seed = args.base_velocity_seed + repeat_index
        langevin_seed = args.base_langevin_seed + repeat_index
        print(f"repeat {repeat_index}: velocity_seed={velocity_seed} langevin_seed={langevin_seed}", flush=True)
        # Drawn ONCE per repeat and reused verbatim (same Python array) for both arms --
        # see _draw_maxwell_boltzmann_velocities' docstring for why we don't rely on OpenMM's
        # own Context.setVelocitiesToTemperature(seed) to reproduce identically across two
        # differently-composed Systems.
        velocities_nm_ps = _draw_maxwell_boltzmann_velocities(particle_masses_dalton, velocity_seed)

        baseline_win_sys, baseline_wrap = _build_baseline_win_sys()
        print(f"  baseline: burn-in {args.burn_in_steps} + production {args.production_steps} steps", flush=True)
        baseline_result = _run_arm(baseline_win_sys, baseline_wrap, is_student=False,
                                    velocities_nm_ps=velocities_nm_ps, langevin_seed=langevin_seed)

        student_win_sys, student_wrap = _build_student_win_sys()
        print(f"  student:  burn-in {args.burn_in_steps} + production {args.production_steps} steps", flush=True)
        student_result = _run_arm(student_win_sys, student_wrap, is_student=True,
                                   velocities_nm_ps=velocities_nm_ps, langevin_seed=langevin_seed)

        velocity_draw_matches = bool(np.array_equal(
            baseline_result["velocities_nm_ps_first_atom"], student_result["velocities_nm_ps_first_atom"],
        ))
        if not velocity_draw_matches:
            raise PilotError(
                f"repeat {repeat_index}: baseline/student velocity draws differ despite an identical "
                f"injected array -- this should be structurally impossible now (same Python array set "
                f"via Context.setVelocities() on both), investigate before proceeding"
            )

        improvement = baseline_result["mixture_ess_proxy_per_gpu_hour"] > 0.0 and (
            student_result["mixture_ess_proxy_per_gpu_hour"] > baseline_result["mixture_ess_proxy_per_gpu_hour"]
        )
        both_converged = (
            baseline_result["delta_g_converged"] and student_result["delta_g_converged"]
            and baseline_result["delta_g_kj_mol"] is not None and student_result["delta_g_kj_mol"] is not None
            and baseline_result["delta_g_uncertainty_kj_mol"] is not None
            and student_result["delta_g_uncertainty_kj_mol"] is not None
        )
        # combined_sigma/delta_g_z are only ever computed from values already None-guarded above,
        # so no arithmetic here can hit a None operand or produce a non-finite JSON value.
        combined_sigma = (
            math.sqrt(baseline_result["delta_g_uncertainty_kj_mol"] ** 2 + student_result["delta_g_uncertainty_kj_mol"] ** 2)
            if both_converged else None
        )
        # Fail closed, not fail-quiet: solve_stage_integrated's own non-convergence sentinel
        # (total_error=999.9) would otherwise make combined_sigma huge and delta_g_z tiny,
        # silently reporting "consistent" for a ΔG comparison that never actually converged.
        # None (not float("inf")) when undefined -- json.dumps(allow_nan=False) below rejects
        # Infinity/NaN outright (correctly; that guard exists precisely so this report can never
        # silently carry a non-finite value), so the sentinel itself has to be JSON-representable.
        delta_g_z = (
            abs(student_result["delta_g_kj_mol"] - baseline_result["delta_g_kj_mol"]) / combined_sigma
            if both_converged and combined_sigma is not None and combined_sigma > 0.0 else None
        )

        repeats.append({
            "repeat_index": repeat_index, "velocity_seed": velocity_seed, "langevin_seed": langevin_seed,
            "velocity_draw_matches": velocity_draw_matches,
            "baseline": baseline_result, "student": student_result,
            "mixture_ess_proxy_per_gpu_hour_improvement": bool(improvement),
            "delta_g_both_arms_converged": both_converged,
            "delta_g_combined_sigma_kj_mol": combined_sigma,
            "delta_g_z_score": delta_g_z,
            "delta_g_consistent": bool(both_converged and delta_g_z is not None
                                        and delta_g_z <= args.delta_g_consistency_z_threshold),
        })
        delta_g_z_display = f"{delta_g_z:.3f}" if delta_g_z is not None else "N/A(not_converged)"
        print(f"  repeat {repeat_index}: improvement={improvement} delta_g_z={delta_g_z_display} "
              f"baseline_ess/gpu-hr={baseline_result['mixture_ess_proxy_per_gpu_hour']:.3f} "
              f"student_ess/gpu-hr={student_result['mixture_ess_proxy_per_gpu_hour']:.3f}", flush=True)

    n_improved = sum(1 for r in repeats if r["mixture_ess_proxy_per_gpu_hour_improvement"])
    improvements = [
        r["student"]["mixture_ess_proxy_per_gpu_hour"] - r["baseline"]["mixture_ess_proxy_per_gpu_hour"]
        for r in repeats
    ]
    median_improvement = statistics.median(improvements)
    majority_improved = n_improved >= math.ceil(2 * args.n_repeats / 3)

    switch_counts_worse = [
        r["student"]["dominant_component_switch_count"] < r["baseline"]["dominant_component_switch_count"]
        for r in repeats
    ]
    autocorrelation_worse = [
        r["student"]["dominant_component_autocorrelation"]["integrated_autocorrelation_time_frames"]
        > r["baseline"]["dominant_component_autocorrelation"]["integrated_autocorrelation_time_frames"]
        for r in repeats
    ]
    consistently_worse_mixing = all(switch_counts_worse) and all(autocorrelation_worse)

    all_ledger_closed = all(r["baseline"]["ledger_closed"] and r["student"]["ledger_closed"] for r in repeats)
    all_finite = all(r["baseline"]["all_finite"] and r["student"]["all_finite"] for r in repeats)
    all_delta_g_consistent = all(r["delta_g_consistent"] for r in repeats)

    pilot_promotion_verdict = bool(
        majority_improved and median_improvement > 0.0 and not consistently_worse_mixing
        and all_ledger_closed and all_finite and all_delta_g_consistent
    )

    body = {
        "schema_version": "exp012-wp5a-pilot-baseline-vs-student-v1",
        "status": "COMPLETED_WP5A_STEP3_PILOT",
        "window": {"stage_type": args.stage_type, "window_index": args.window_index, "K": n_states,
                   "lambdas_vdw": lambdas_vdw, "lambdas_coul": lambdas_coul},
        "platform": {"requested": args.platform, "resolved_name": resolved_platform_name,
                     "precision": platform_properties.get("Precision"), "properties": platform_properties},
        "frozen_candidate": {
            "checkpoint_path": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": _sha256_file(args.checkpoint),
            "checkpoint_held_out_run_id": payload.get("held_out_run_id"), "checkpoint_seed": payload.get("seed"),
            "torchscript_sha256": torchscript_sha256, "torchscript_a_k_baked_in": 1.0,
            "coefficient_c1": float(args.coefficient), "a_k_per_state": a_k_per_state,
            "protocol_sha256": controller.protocol_sha256(lambdas=lambdas_vdw),
            "f_k_reused_unmodified_for_both_arms": f_k.tolist(),
            "note": "frozen per DEC-045/047; not adjusted based on this pilot's own results",
        },
        "design": {
            "n_repeats": args.n_repeats, "burn_in_steps_uncounted_in_statistics": args.burn_in_steps,
            "production_steps_monitored": args.production_steps, "steps_per_chunk": args.steps_per_chunk,
            "n_frames_per_repeat_per_arm": args.production_steps // args.steps_per_chunk,
            "gpu_hours_includes_burn_in": True,
            "note": "'paired reseed' pilot, NOT three independently equilibrated configurations -- only one "
                    "real production checkpoint exists on disk; each repeat redraws velocities from that SAME "
                    "checkpoint's positions/box via a fresh Boltzmann draw + fresh Langevin seed, both shared "
                    "identically between the baseline/student arms of that repeat and different across repeats",
        },
        "metric_definitions": {
            "mixture_ess_proxy": "min-over-states IBS mixture-coverage ESS (_ibs_reweighting_quality_diagnostics, "
                                  "ESS_GATE_PROTOCOL_VERSION=3) -- NOT a literal pymbar.compute_overlap() matrix, "
                                  "which degenerates for IBS's single-sampled-row augmented matrix "
                                  "(ibs_engine.py:13305-13319). is_literal_pymbar_overlap=false always.",
            "dominant_component_switch_count / dominant_component_autocorrelation / endpoint_proxy_traversals":
                "derived from a self-constructed per-frame argmin_k(energy_buffer[k]-f_k[k]) label series -- "
                "NOT a physical replica trajectory (IBS runs one simulation per window, not REMD). "
                "is_physical_replica_round_trip=false always. endpoint_proxy_traversals counts arrivals at "
                "this window's own k=0 (lambda_vdw=1.0) genuine global endpoint component only; window 0 does "
                "not contain the vdw stage's other endpoint (lambda_vdw=0.0).",
            "delta_g_kj_mol": "ibs_engine.solve_stage_integrated on a single-window in-memory window_outputs "
                              "entry (same schema IBSWindowManagerDualLambda.get_stage_data_for_analysis "
                              "produces from disk) -- this window's own local TMBAR contribution, not a "
                              "full-stage bridged ΔG.",
        },
        "repeats": repeats,
        "results": {
            "n_repeats_improved": n_improved, "majority_improved": majority_improved,
            "median_mixture_ess_proxy_per_gpu_hour_improvement": median_improvement,
            "consistently_worse_mixing": consistently_worse_mixing,
            "all_ledger_closed": all_ledger_closed, "all_finite": all_finite,
            "all_delta_g_consistent": all_delta_g_consistent,
        },
        "pilot_promotion_verdict": pilot_promotion_verdict,
        "policy": {
            "decision_reference": "WP-5A step 3 pilot (post-DEC-047 step 1/2 closure)",
            "not_a_sealed_wp5a_gate": True,
            "literal_bar_mbar_overlap_status": "not_attempted -- would require a full reduced-potential ledger "
                                                "this pilot does not collect; not to be inferred from "
                                                "mixture_ess_proxy",
            "physical_replica_round_trip_status": "not_applicable -- IBS's single-reference sampling scheme has "
                                                    "no discrete replica/state trajectory to count round trips "
                                                    "on; endpoint_proxy_traversals is a named proxy, not a "
                                                    "substitute claim of applicability",
            "ibs_engine_py_modified": False, "production_checkpoints_written": False,
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    print(f"pilot_promotion_verdict={pilot_promotion_verdict} n_repeats_improved={n_improved}/{args.n_repeats} "
          f"median_improvement={median_improvement:.4f} consistently_worse_mixing={consistently_worse_mixing} "
          f"all_ledger_closed={all_ledger_closed} all_finite={all_finite} "
          f"all_delta_g_consistent={all_delta_g_consistent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
