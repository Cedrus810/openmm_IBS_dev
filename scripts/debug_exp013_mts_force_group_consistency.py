#!/usr/bin/env python
"""EXP-013 013-B thermal-collapse diagnostic, CHECK E: per-group force-sum consistency.

Prior diagnostics (see scripts/debug_exp013_mts_thermostat_diagnostic.py, CHECK A/B/C/D)
ruled out: (a) the ΔV_θ/CustomCVForce/TorchForce residual split (CHECK C reproduces the
same collapse WITHOUT it), (b) a missing/zeroed thermostat noise term (a/b/kT globals
match theory exactly for both N=1 and N=32), (c) a multi-group RESPA nesting/duplication
bug in the per-step program text (N=1's 21-step program and N=32's per-substep block are
both textbook-correct, O-step appears exactly once per true inner_dt interval).

Remaining leading hypothesis: MTSLangevinIntegrator drives dynamics using per-FORCE-GROUP
force queries (its per-step program literally references f0, f1, ..., f5 as separate
terms, summed via sequential half-kicks). If summing those per-group-restricted force
queries does not reconstruct the same total force a normal, group-unrestricted query
gives (e.g. because some Force object in this System has cross-group dependencies, or
a platform-level caching/synchronization quirk affects repeated group-restricted state
queries within one Context), MTS dynamics would evolve under a silently WRONG effective
force field even though every individual energy/force number "looks" finite and
reasonable in isolation -- exactly the kind of bug a single-frame 013-A equivalence
check (which uses group-unrestricted queries) cannot see.

This check needs zero integration steps: load the checkpoint, and at that exact
frame compare (sum of per-group getState(groups={g}) forces) against
(getState() with no group restriction). Cheap, static, and decisive -- if they already
disagree at t=0, that alone is likely sufficient to explain a catastrophic multi-step
collapse once integration starts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="output_lrc_fix")
    parser.add_argument("--stage-type", default="vdw", choices=["vdw"])
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--platform", default="CUDA")
    args = parser.parse_args(argv)

    import numpy as np
    import openmm
    from openmm import XmlSerializer, app, unit

    from ibs_engine import (
        ACESoftcorePotential,
        _build_platform_properties,
        _system_has_global_parameter,
        build_ibs_dual_system,
    )

    output_root = Path(args.output_root)
    checkpoints = output_root / "checkpoints"
    window_dir = checkpoints / "production_window" / args.stage_type / f"window_{args.window_index}"
    manifest_path = window_dir / "manifest.json"
    checkpoint_path = window_dir / "openmm.chk"
    stage_protocol_path = checkpoints / "stage2_vanishing.json"
    system_xml_path = output_root / "system_native.xml"
    topology_cif_path = output_root / "topology.cif"
    box_vectors_path = output_root / "box_vectors.npy"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_payload = json.loads(stage_protocol_path.read_text(encoding="utf-8"))["protocol_key"]["payload"]
    target_temperature_k = float(manifest["temperature_K"])
    lambdas_vdw = [float(value) for value in manifest["lambdas_vdw"]]

    system_xml_text = system_xml_path.read_text(encoding="utf-8")
    base_system = XmlSerializer.deserialize(system_xml_text)
    topology = app.PDBxFile(str(topology_cif_path)).topology
    box_vectors = unit.Quantity(np.load(box_vectors_path), unit.nanometer)
    alchemical_params = ACESoftcorePotential.from_dict(stage_payload["aces_softcore_params"])

    resolved_platform_name, platform_properties = _build_platform_properties(args.platform)
    platform = openmm.Platform.getPlatformByName(resolved_platform_name)

    win_sys, ibs_wrap = build_ibs_dual_system(
        base_system, topology, stage_payload["ligand_indices"],
        manifest["lambdas_coul"], lambdas_vdw, alchemical_params,
        potential_type=stage_payload["potential_type"], restraint_params=stage_payload["boresch_params"],
        temperature=target_temperature_k * unit.kelvin, prefix=stage_payload.get("prefix", "win0"),
        box_vectors=box_vectors, reference_positions=None,
        dispersion_protocol="legacy_uniform_density_lrc", environment_type="soluble",
    )

    forces = list(win_sys.getForces())
    print(f"total force objects: {len(forces)}")
    groups = sorted({int(f.getForceGroup()) for f in forces})
    print(f"force groups present: {groups}")
    for f in forces:
        print(f"  group {f.getForceGroup():2d}  {type(f).__name__}  name={f.getName()!r}")

    # A dummy integrator, never stepped -- only used to create a Context to query state.
    integrator = openmm.VerletIntegrator(manifest["step_size_ps"] * unit.picosecond)
    simulation = app.Simulation(topology, win_sys, integrator, platform, platform_properties)
    simulation.loadCheckpoint(str(checkpoint_path))
    if _system_has_global_parameter(win_sys, "lambda_boresch_scale"):
        simulation.context.setParameter("lambda_boresch_scale", float(manifest["lambda_boresch_scale"]))
    if _system_has_global_parameter(win_sys, "lambda_shield"):
        simulation.context.setParameter("lambda_shield", float(manifest["lambda_shield"]))

    context = simulation.context

    total_state = context.getState(getForces=True, getEnergy=True)
    total_force = total_state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    total_energy = total_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    print(f"\nno-group-restriction total potential energy: {total_energy:.6f} kJ/mol")
    print(f"no-group-restriction total force: max |F| = {float(np.max(np.linalg.norm(total_force, axis=1))):.6f} kJ/mol/nm")

    per_group_forces = {}
    per_group_energies = {}
    summed_force = np.zeros_like(total_force)
    summed_energy = 0.0
    for g in groups:
        state_g = context.getState(getForces=True, getEnergy=True, groups={g})
        force_g = state_g.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        energy_g = state_g.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        per_group_forces[g] = force_g
        per_group_energies[g] = energy_g
        summed_force += force_g
        summed_energy += energy_g
        print(f"  group {g}: energy={energy_g:14.6f} kJ/mol  max|F|={float(np.max(np.linalg.norm(force_g, axis=1))):12.6f} kJ/mol/nm")

    print(f"\nsum of per-group energies: {summed_energy:.6f} kJ/mol  "
          f"(vs total {total_energy:.6f}, diff={summed_energy - total_energy:.6e})")

    force_diff = summed_force - total_force
    abs_diff_norms = np.linalg.norm(force_diff, axis=1)
    max_abs_diff = float(np.max(abs_diff_norms))
    worst_atom = int(np.argmax(abs_diff_norms))
    total_norms = np.linalg.norm(total_force, axis=1)
    rel_diff = max_abs_diff / float(np.max(total_norms)) if float(np.max(total_norms)) > 0 else float("nan")

    print(f"\nFORCE-SUM CONSISTENCY:")
    print(f"  max |sum_of_group_forces - total_force| over atoms = {max_abs_diff:.6e} kJ/mol/nm")
    print(f"  (relative to max|total_force| = {float(np.max(total_norms)):.6e}: {rel_diff:.6e})")
    print(f"  worst atom index = {worst_atom}, "
          f"sum_force={summed_force[worst_atom]}, total_force={total_force[worst_atom]}")
    n_atoms_over_1e_minus3 = int(np.sum(abs_diff_norms > 1e-3))
    n_atoms_over_1 = int(np.sum(abs_diff_norms > 1.0))
    print(f"  atoms with |diff| > 1e-3 kJ/mol/nm: {n_atoms_over_1e_minus3} / {len(abs_diff_norms)}")
    print(f"  atoms with |diff| > 1.0   kJ/mol/nm: {n_atoms_over_1} / {len(abs_diff_norms)}")

    if max_abs_diff > 1.0:
        print("\n  ==> INCONSISTENT: per-group force sum does NOT reconstruct the total force.")
        print("      This alone is sufficient to explain wrong MTS dynamics: the integrator's")
        print("      own per-step program literally sums f0..fN, so if that sum is wrong here,")
        print("      every MTS step applies a wrong net force regardless of N, group order, or")
        print("      the presence/absence of the delta/student force.")
    else:
        print("\n  ==> CONSISTENT at this frame: per-group sum matches total force closely.")
        print("      This specific hypothesis is then NOT the explanation -- root cause is")
        print("      still open and likely needs an OpenMM-version/platform-specific angle")
        print("      (e.g. file a question upstream, or bisect by removing one group at a time")
        print("      from the MTS groups list and re-running CHECK B-style short dynamics).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
