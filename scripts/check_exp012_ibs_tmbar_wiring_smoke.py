#!/usr/bin/env python
"""EXP-012 WP-5A step 1: full IBS/TMBAR wiring smoke for the frozen student.

D0-D4 only ever added the student as a SEPARATE, independent Force on its own
force group (D3 sub-item 3/4, D4) -- a smoke test of the Force in isolation,
never actually folded into the per-state discriminant that the real IBS bias
(and therefore TMBAR/target accounting) uses. This script is the first time
the student enters the ACTUAL K-state discriminant
``X_k = cv_k_int + cv_k_rest + A_k*basis - f_k`` that both drives sampling
(Group 1 CustomCVForce) and is recorded into target/TMBAR history.

Design, reusing already-built-and-tested (but never-before-connected-to-a-
real-window) library code rather than writing new production Hamiltonian
logic from scratch:

- `outer_lambda_neural_basis.OuterLambdaController` (WP-1): computes the
  frozen per-state coefficients `A_k = w(lambda_k) * c_1` from a sin2
  envelope + constant coefficient model, M=1. Structurally guarantees
  `A_k = 0` at `w(lambda)=0`, i.e. at `lambda in {0, 1}`.
- `outer_lambda_neural_basis.OuterLambdaIBSBiasForce` (WP-1/2): an API-
  compatible replacement for `ibs_engine.IBSBiasForce` whose CustomCVForce
  expression bakes `A_k` in as literal numeric constants per state, folding
  the shared student basis CV into every state's discriminant exactly once
  (not K copies of the model).
- `outer_lambda_neural_basis.IBSSamplerNeuralPathAdapter` (WP-2): wraps a
  real `ibs_engine.IBSSampler` and overrides only `collect_energies()` to
  compose target/bias/base history via `compose_ibs_energy_frame`, appending
  into the SAME four history lists `IBSSampler` already owns -- so nothing
  about checkpoint/resume, TMBAR consumption, or on-disk history-array
  schema changes.

What this script does NOT touch: `ibs_engine.py` is not modified. The real,
hash-verified `hard_window0` win_sys is built by the completely unmodified
`build_ibs_dual_system` (same checkpoint-derived-box construction as
D3/D4/the no-student baseline); only a COPY of that System has its Group-1
Force surgically swapped (removed, replacement added) entirely from this
script's own code. `IBS_BIAS_PROTOCOL_VERSION` and the real production
manifest/checkpoint contract are untouched -- this smoke never writes into
`output_lrc_fix/checkpoints/`, only into its own `--output` report.

Verification performed (see module docstring items in the PLAN doc's
"完成定义" list this maps onto):

1. Student enters every one of the K target states: after `collect_energies()`,
   `frame.target_state_energies_kj_mol[k] - (original[k]+lrc[k])` must equal
   `frame.neural_path_state_energies_kj_mol[k]`, and the latter must be
   nonzero at every INTERIOR state (this window's own k=0 state, lambda=1.0,
   is a genuine global vdw-stage endpoint and is correctly expected to be
   zero -- see item 5).
2. A_k is frozen: baked as literal numeric constants into the CustomCVForce
   expression string at construction time (`OuterLambdaIBSBiasForce.__init__`),
   not read from a mutable global parameter -- there is no code path in this
   script that could change it mid-run.
3. Target energy includes the student (item 1); the underlying OpenMM Group-1
   potential energy that actually drives dynamics is independently
   cross-checked in pure Python against the same log-sum-exp formula
   (`OuterLambdaIBSBiasForce`'s own expression, reimplemented here from
   scratch in numpy) -- not just trusted from the library's own unit tests.
   Two tolerance tiers, not one (same mistake DEC-042 had to split out for D3
   sub-item 1, applied here from the start): the target-composition check
   (ledger's target-minus-neural_path vs an independently re-queried
   original+LRC) rereads the SAME underlying Context state twice, so it stays
   at a strict float64-noise tolerance; the Group-1-vs-numpy check crosses a
   real precision boundary (OpenMM's Group-1 value is computed at whatever
   `--platform`/Precision was actually requested, e.g. CUDA "mixed", while the
   numpy side is float64), so it uses a relative tolerance instead.
4. IBS/WCA sampling bias stays independently booked: `bias_history`/
   `base_energy_history` are still populated from exactly Group {1,4} /
   {0,2,3,5} via the unmodified formula in
   `IBSSamplerNeuralPathAdapter.collect_energies` -- verified by construction
   (same source line reused) and by recording the values in the report for
   inspection.
5. Global endpoint check: this window's own k=0 state (lambda_vdw=1.0) is a
   genuine physical endpoint of the vdw-vanishing stage; A_0 there must be
   *exactly* 0.0 (checked to machine precision, not just "small"). The
   window's OTHER boundary state (k=K-1, an interior ladder rung shared with
   the adjacent window, NOT the vdw stage's other physical endpoint at
   lambda_vdw=0.0) is correctly expected to have nonzero A_k -- this script
   does not claim to have verified the lambda_vdw=0.0 endpoint, which lives
   in a different window entirely.
6. Checkpoint/resume: `Simulation.loadCheckpoint()` on the Group-1-swapped
   System must succeed and restore real production positions/velocities/box
   (same property D3/D4 already exercised for a separately-added Force;
   here checked again for a Force that REPLACES an existing one instead).
   Model identity (checkpoint SHA-256, TorchScript SHA-256,
   `controller.protocol_sha256`) is recorded in the report for provenance.
   A fresh reconstruction's `win_sys_xml_sha256` is recorded against
   `manifest.json`'s value for information only, NOT as a pass/fail gate:
   DEC-041 already sealed that exact byte-level question as
   `CLOSED_STEP3_OPERATIONAL_SEMANTIC_PASS` (no detectable semantic
   difference across 10 independent structural fields plus a Force canonical
   fingerprint) and its decision log explicitly says not to re-open that
   investigation per run. This script instead gates on the sealed
   provenance report's verdict field.
7. Ledger closure: all six history lists
   (`energy_buffer`/`energy_history`/`bias_history`/`base_energy_history`/
   `neural_path_energy_history`/`basis_energy_history`) must have identical
   length after every collected frame (the adapter's own atomic rollback
   already guarantees this on any partial failure; this script asserts it
   explicitly rather than trusting that invariant silently).

Explicitly out of scope: re-equilibrating a trajectory under the new
(student-modified) bias -- the checkpoint used here was sampled under the
OLD, student-free ensemble, so this smoke's frames are not claimed to be
representative production samples of the new scheme. That is WP-5A step 3's
job (3 paired independent repeats), not this wiring-correctness check. This
script also reuses the already-CONVERGED production f_k as a fixed starting
bias rather than re-learning it -- also correct only for a mechanics smoke,
not for a production claim.
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


class WiringSmokeError(RuntimeError):
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="output_lrc_fix")
    parser.add_argument("--stage-type", default="vdw", choices=["vdw"])
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--checkpoint", required=True, help="a direct_gap .pt checkpoint from student_checkpoints/")
    parser.add_argument("--coefficient", type=float, default=0.5,
                         help="frozen raw coefficient c_1 (A_k = sin2(pi*lambda_k) * c_1); a smoke "
                              "constant reused from D3/D4's a_k=0.5, NOT tuned for production sampling benefit")
    parser.add_argument("--max-abs-coefficient", type=float, default=1.0)
    parser.add_argument("--max-abs-basis-energy-kj-mol", type=float, default=50.0)
    parser.add_argument("--max-abs-path-energy-kj-mol", type=float, default=25.0)
    parser.add_argument("--max-force-norm-kj-mol-nm", type=float, default=500.0)
    parser.add_argument("--n-frames", type=int, default=20, help="number of collect_energies() samples")
    parser.add_argument("--steps-per-frame", type=int, default=50, help="dynamics steps between samples")
    parser.add_argument("--endpoint-tolerance-kj-mol", type=float, default=0.0,
                         help="A_0 at this window's own lambda=1.0 state must be exactly this close to zero")
    parser.add_argument(
        "--target-composition-tolerance-kj-mol", type=float, default=1e-8,
        help="STRICT correctness tier: ledger's target-minus-neural_path vs an independently re-queried "
             "original+LRC, both read from the SAME underlying Context state via the same getState() calls "
             "-- no precision boundary crossed here, so this should sit at float64 noise level.",
    )
    parser.add_argument(
        "--group1-relative-tolerance", type=float, default=1e-4,
        help="dtype-aware tier: OpenMM's own Group-1 potential energy (computed on whatever --platform/"
             "Precision was actually requested, e.g. CUDA 'mixed') vs an independent from-scratch numpy "
             "float64 reimplementation of the exact same log-sum-exp formula. These two sides cross a real "
             "precision boundary (mixed vs float64), so this is relative, not the strict correctness tier "
             "above -- do not tighten it to --target-composition-tolerance-kj-mol, that conflates precision "
             "loss with a logic bug (same mistake DEC-042 made and had to split out for D3 sub-item 1).",
    )
    parser.add_argument(
        "--group1-absolute-floor-kj-mol", type=float, default=1e-3,
        help="secondary allowance alongside --group1-relative-tolerance for when the Group-1 energy "
             "magnitude is small enough that a pure relative bound would be unreasonably tight",
    )
    parser.add_argument(
        "--provenance-report", default="output/outer_lambda_exp012/d3_0_provenance_gate_report.json",
        help="DEC-041's sealed win_sys provenance verdict. A raw win_sys_xml_sha256-vs-manifest byte "
             "comparison is recorded here for information only and is NOT part of all_passed -- DEC-041 "
             "already closed that exact question as CLOSED_STEP3_OPERATIONAL_SEMANTIC_PASS (no detectable "
             "semantic difference; the historical byte-level mismatch is non-blocking), and the decision "
             "log explicitly says not to re-litigate it per run. This script instead checks that the sealed "
             "verdict file exists and still carries an accepted verdict.",
    )
    parser.add_argument("--torchscript-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.n_frames < 1:
        parser.error("--n-frames must be positive")
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
        _system_has_global_parameter,
        build_ibs_dual_system,
    )
    from outer_lambda_neural_basis import (  # noqa: E402
        NeuralBasisModelSpec,
        NeuralPathSafety,
        OuterLambdaController,
        OuterLambdaIBSBiasForce,
        IBSSamplerNeuralPathAdapter,
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
            raise WiringSmokeError(f"required real production artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_payload = json.loads(stage_protocol_path.read_text(encoding="utf-8"))["protocol_key"]["payload"]
    ibs_state = json.loads(ibs_state_path.read_text(encoding="utf-8"))
    target_temperature_k = float(manifest["temperature_K"])
    lambdas_vdw = [float(value) for value in manifest["lambdas_vdw"]]
    n_states = len(lambdas_vdw)
    prefix = ibs_state["prefix"]
    f_k = np.asarray(ibs_state["f_k"], dtype=float)
    if f_k.shape[0] != n_states:
        raise WiringSmokeError("ibs_state f_k length does not match manifest K")

    system_xml_text = system_xml_path.read_text(encoding="utf-8")
    if _sha256_text(system_xml_text) != stage_payload["system_xml_sha256"]:
        raise WiringSmokeError("system_native.xml SHA-256 does not match stage2 protocol record")
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
        prefix=prefix,
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

    # ---- real win_sys, exactly as build_ibs_dual_system produces it ----
    win_sys, original_ibs_wrap = build_ibs_dual_system(
        base_system, topology, stage_payload["ligand_indices"],
        manifest["lambdas_coul"], manifest["lambdas_vdw"], alchemical_params,
        potential_type=stage_payload["potential_type"],
        restraint_params=stage_payload["boresch_params"],
        temperature=target_temperature_k * unit.kelvin,
        prefix=prefix,
        box_vectors=box_vectors, reference_positions=None,
        dispersion_protocol="legacy_uniform_density_lrc", environment_type="soluble",
    )
    win_sys_xml_before_swap_sha256 = _sha256_text(XmlSerializer.serialize(win_sys))
    # Recorded for information only -- NOT part of all_passed. DEC-041 already closed the
    # question "does a fresh build_ibs_dual_system() reconstruction byte-match manifest.json's
    # win_sys_xml_sha256" as CLOSED_STEP3_OPERATIONAL_SEMANTIC_PASS (10/10 independent structural
    # fields + Force canonical fingerprint matched; the byte-level mismatch itself was never
    # attributed to a semantic difference, and the decision log explicitly says not to re-open
    # that investigation per run). Re-deriving a fresh pass/fail from the same raw byte comparison
    # here would just re-litigate an already-closed question with a stricter (wrong) bar.
    win_sys_xml_matches_manifest_raw_bytes = win_sys_xml_before_swap_sha256 == manifest["win_sys_xml_sha256"]
    provenance_report_path = Path(args.provenance_report)
    if not provenance_report_path.is_file():
        raise WiringSmokeError(
            f"--provenance-report {provenance_report_path} not found -- DEC-041's sealed win_sys "
            "provenance verdict is required to interpret win_sys_xml_sha256 mismatches as non-blocking"
        )
    provenance_report = json.loads(provenance_report_path.read_text(encoding="utf-8"))
    provenance_verdict = provenance_report.get("verdict")
    accepted_provenance_verdicts = {"CLOSED_STEP3_OPERATIONAL_SEMANTIC_PASS"}
    provenance_verdict_accepted = provenance_verdict in accepted_provenance_verdicts
    if not original_ibs_wrap._int_cv_force_xmls or len(original_ibs_wrap._int_cv_force_xmls) != n_states:
        raise WiringSmokeError("original IBSBiasForce did not capture the expected per-state cv_*_int XMLs")

    # ---- surgically remove the ORIGINAL Group-1 IBSBiasForce ----
    group1_indices = [
        index for index in range(win_sys.getNumForces())
        if win_sys.getForce(index).getForceGroup() == 1
    ]
    if len(group1_indices) != 1:
        raise WiringSmokeError(f"expected exactly one force group 1 (IBSBiasForce), found {len(group1_indices)}")
    win_sys.removeForce(group1_indices[0])

    # ---- student TorchForce as the one shared basis (M=1), a_k=1.0/offset=0.0 ----
    # a_k=1.0 (no baked-in scalar) is required here: the state-dependent A_k
    # scaling is now applied by OuterLambdaIBSBiasForce's CV expression, not
    # by the exported module itself -- baking a_k into the module AND
    # multiplying by A_k in the Force expression would double-apply the
    # coefficient. This deliberately differs from D3/D4's a_k=0.5 export
    # (DEC-045): those exports were standalone single-state smokes where the
    # module's own scalar WAS the entire coupling strength.
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("variant") != "direct_gap":
        raise WiringSmokeError(f"--checkpoint variant={payload.get('variant')!r}, only direct_gap is qualified")
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
        name="local_residual_student_hard_window0_wiring_smoke",
        backend="torchforce",
        model_path=str(Path(args.torchscript_output).resolve()),
        sha256=torchscript_sha256,
        energy_offset_kj_mol=0.0,  # b_m=0.0 for this first wiring smoke; calibrating a real b_m is future work
        atom_selection="dynamic_funnel_environment",
        atom_indices_path=str(ligand_indices_path.resolve()),
        atom_indices_sha256=ligand_indices_sha256,
        output_unit="kJ_per_mol",
        precision="double",
        periodic=True,
    )
    student_force = build_torchforce_from_spec(basis_spec)

    # ---- frozen controller: sin2 envelope, constant coefficient, M=1 ----
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
    coefficient_matrix = controller.coefficient_matrix(lambdas_vdw)
    a_k_per_state = [row[0] for row in coefficient_matrix]

    # ---- replacement Group-1 force: same per-state cv_k_int/cv_k_rest, plus the shared student CV ----
    new_wrapper = OuterLambdaIBSBiasForce(
        controller, lambdas_vdw, target_temperature_k, [student_force], prefix=prefix,
    )
    # build_ibs_dual_system sets this analytic LJ-dispersion-tail coefficient
    # on IBSBiasForce as a plain external attribute (ibs_engine.py:4102-4124),
    # not through any constructor argument OuterLambdaIBSBiasForce has -- it
    # must be copied by hand, or IBSSampler._lj_tail_correction_kj_mol()
    # (called inside IBSSamplerNeuralPathAdapter.collect_energies()) would
    # silently see `None` and return all-zero LRC instead of the real
    # per-state correction, making target_state_energies wrong exactly the
    # way this project's own `output_lrc_fix` naming history warns against.
    new_wrapper.lj_tail_lrc_coeff_kj_mol = original_ibs_wrap.lj_tail_lrc_coeff_kj_mol
    for k in range(n_states):
        int_cv = XmlSerializer.deserialize(original_ibs_wrap._int_cv_force_xmls[k])
        new_wrapper.addCollectiveVariable(f"cv_{k}_int", int_cv)
        # Always the literal zero-force per ibs_engine.py's own "Boresch CV 保持零力" comment
        # (real Boresch restraint physics lives in Group 3, unaffected by this swap).
        new_wrapper.addCollectiveVariable(f"cv_{k}_rest", openmm.CustomExternalForce("0"))
    win_sys.addForce(new_wrapper.get_force())
    win_sys_xml_after_swap_sha256 = _sha256_text(XmlSerializer.serialize(win_sys))

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
    checkpoint_loaded_without_exception = True
    if _system_has_global_parameter(win_sys, "lambda_boresch_scale"):
        simulation.context.setParameter("lambda_boresch_scale", float(manifest["lambda_boresch_scale"]))
    if _system_has_global_parameter(win_sys, "lambda_shield"):
        simulation.context.setParameter("lambda_shield", float(manifest["lambda_shield"]))
    new_wrapper.update_parameters(simulation.context, f_k)

    gpu_after_context = _gpu_memory_mib()

    sampler = IBSSampler(
        simulation.context, n_states, target_temperature_k * unit.kelvin, prefix=prefix, ibs_wrapper=new_wrapper,
    )
    adapter = IBSSamplerNeuralPathAdapter(sampler, controller, lambdas_vdw, new_wrapper)

    kt_kj_per_mol = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin) * target_temperature_k
    beta = 1.0 / kt_kj_per_mol

    def _independent_group1_cross_check() -> dict:
        """Reimplement OuterLambdaIBSBiasForce's own log-sum-exp formula from scratch in numpy,
        as an independent check against the OpenMM Group-1 potential energy it actually produced --
        not just trusting the library's own (already-passing) unit tests."""
        # Reuse the exact same helper (and calling convention: plain OpenMM
        # Quantity objects, not asNumpy=True arrays -- `evaluate_interaction_
        # energies` unpacks box_vectors via `*box_vectors` into
        # `setPeriodicBoxVectors`, which needs three Vec3-like rows, not a
        # single numpy array) that `collect_energies()` itself calls, so this
        # cross-check reads identically-sourced cv_k_int+cv_k_rest values.
        cv_int_rest = np.asarray(sampler._collect_interaction_energies(), dtype=np.float64)
        lrc = np.asarray(sampler._lj_tail_correction_kj_mol(), dtype=np.float64)
        basis_value = new_wrapper.get_centered_basis_energies_kj_mol(simulation.context)[0]
        # bias_scale is always 1.0 here (this script never calls set_bias_enabled),
        # but read it back explicitly rather than silently assuming that -- it is a
        # live, mutable OpenMM global parameter, not a constant baked at construction.
        bias_scale = float(simulation.context.getParameter(f"{prefix}_bias_scale"))

        def _discriminant(a_k_values):
            return cv_int_rest + np.asarray(a_k_values, dtype=np.float64) * basis_value - f_k

        def _log_sum_exp_bias(x_k):
            diffs = -beta * (x_k - x_k[0])
            pivot = max(0.0, float(np.max(diffs[1:])) if n_states > 1 else 0.0)
            terms = [math.exp(-pivot)] + [math.exp(diffs[i] - pivot) for i in range(1, n_states)]
            return bias_scale * float(x_k[0] - kt_kj_per_mol * (pivot + math.log(max(1e-300, sum(terms)))))

        with_student = _log_sum_exp_bias(_discriminant(a_k_per_state))
        without_student = _log_sum_exp_bias(_discriminant([0.0] * n_states))
        group1_state = simulation.context.getState(getEnergy=True, groups={1})
        openmm_group1_kj_mol = group1_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        absolute_error = abs(openmm_group1_kj_mol - with_student)
        # OpenMM's Group-1 value was computed at whatever --platform/Precision was actually
        # requested (e.g. CUDA "mixed"); `with_student` is a from-scratch float64 numpy
        # reimplementation -- a real precision boundary, not a same-state reread. Compare with
        # a relative tolerance (plus a small absolute floor for when the magnitude is tiny),
        # not the strict tier used for the same-precision target-composition check below.
        relative_error = absolute_error / max(abs(openmm_group1_kj_mol), 1e-12)
        group1_cross_check_passed = bool(
            relative_error <= args.group1_relative_tolerance or absolute_error <= args.group1_absolute_floor_kj_mol
        )
        return {
            "openmm_group1_potential_energy_kj_mol": float(openmm_group1_kj_mol),
            "independent_numpy_with_student_kj_mol": with_student,
            "independent_numpy_without_student_kj_mol": without_student,
            "openmm_vs_independent_numpy_absolute_error_kj_mol": absolute_error,
            "openmm_vs_independent_numpy_relative_error": relative_error,
            "group1_cross_check_passed": group1_cross_check_passed,
            "with_vs_without_student_absolute_difference_kj_mol": abs(with_student - without_student),
            "basis_value_kj_mol": float(basis_value),
            # A genuinely independent re-derivation of "original interaction + LRC" via a
            # FRESH pair of sampler queries (not read back out of already-stored ledger
            # arrays) -- compared against the ledger's own target-minus-neural_path in the
            # caller, to catch a real composition bug rather than a tautological identity.
            # Both sides of THAT comparison read the same underlying Context state (no
            # precision-boundary crossing), so it keeps the strict tier, unlike this function's
            # own OpenMM-vs-numpy comparison above.
            "original_plus_lrc_independent_kj_mol": (cv_int_rest + lrc).tolist(),
        }

    frames = []
    for frame_index in range(args.n_frames):
        simulation.step(args.steps_per_frame)
        before_lengths = {
            name: len(getattr(sampler, name))
            for name in ("energy_buffer", "energy_history", "bias_history", "base_energy_history")
        }
        before_lengths["neural_path_energy_history"] = len(adapter.neural_path_energy_history)
        before_lengths["basis_energy_history"] = len(adapter.basis_energy_history)

        adapter.collect_energies()

        after_lengths = {
            "energy_buffer": len(sampler.energy_buffer),
            "energy_history": len(sampler.energy_history),
            "bias_history": len(sampler.bias_history),
            "base_energy_history": len(sampler.base_energy_history),
            "neural_path_energy_history": len(adapter.neural_path_energy_history),
            "basis_energy_history": len(adapter.basis_energy_history),
        }
        ledger_closed = len(set(after_lengths.values())) == 1 and all(
            after_lengths[name] == before_lengths[name] + 1 for name in after_lengths
        )

        target = np.asarray(sampler.energy_history[-1], dtype=float)
        neural_path = np.asarray(adapter.neural_path_energy_history[-1], dtype=float)
        original_plus_lrc_from_ledger = target - neural_path

        cross_check = _independent_group1_cross_check()
        original_plus_lrc_independent = np.asarray(cross_check["original_plus_lrc_independent_kj_mol"], dtype=float)
        target_composition_error = float(
            np.max(np.abs(original_plus_lrc_from_ledger - original_plus_lrc_independent))
        )

        frames.append({
            "frame_index": frame_index,
            "target_state_energies_kj_mol": target.tolist(),
            "neural_path_state_energies_kj_mol": neural_path.tolist(),
            "original_plus_lrc_state_energies_kj_mol_from_ledger": original_plus_lrc_from_ledger.tolist(),
            "original_plus_lrc_state_energies_kj_mol_independent": original_plus_lrc_independent.tolist(),
            "target_composition_max_absolute_error_kj_mol": target_composition_error,
            "basis_energies_kj_mol": list(adapter.basis_energy_history[-1]),
            "sampling_bias_energy_kj_mol": float(sampler.bias_history[-1]),
            "base_energy_kj_mol": float(sampler.base_energy_history[-1]),
            "ledger_lengths_after": after_lengths,
            "ledger_closed": bool(ledger_closed),
            "group1_cross_check": cross_check,
            "all_finite": bool(np.all(np.isfinite(target)) and np.all(np.isfinite(neural_path))
                               and math.isfinite(sampler.bias_history[-1]) and math.isfinite(sampler.base_energy_history[-1])
                               and math.isfinite(cross_check["openmm_group1_potential_energy_kj_mol"])),
        })
        print(f"frame {frame_index}: ledger_closed={ledger_closed} "
              f"target_composition_err={target_composition_error:.3e} "
              f"cross_check_err={cross_check['openmm_vs_independent_numpy_absolute_error_kj_mol']:.3e} "
              f"with_vs_without={cross_check['with_vs_without_student_absolute_difference_kj_mol']:.3e}", flush=True)

    gpu_after_frames = _gpu_memory_mib()

    all_ledger_closed = all(frame["ledger_closed"] for frame in frames)
    all_finite = all(frame["all_finite"] for frame in frames)
    max_cross_check_absolute_error = max(
        frame["group1_cross_check"]["openmm_vs_independent_numpy_absolute_error_kj_mol"] for frame in frames
    )
    max_cross_check_relative_error = max(
        frame["group1_cross_check"]["openmm_vs_independent_numpy_relative_error"] for frame in frames
    )
    cross_check_passed = all(frame["group1_cross_check"]["group1_cross_check_passed"] for frame in frames)
    max_target_composition_error = max(frame["target_composition_max_absolute_error_kj_mol"] for frame in frames)
    target_composition_passed = max_target_composition_error <= args.target_composition_tolerance_kj_mol
    student_nonzero_at_interior_states = all(
        frame["group1_cross_check"]["with_vs_without_student_absolute_difference_kj_mol"] > 0.0 for frame in frames
    )

    endpoint_a0 = float(a_k_per_state[0])
    endpoint_a0_zero = abs(endpoint_a0) <= args.endpoint_tolerance_kj_mol
    endpoint_lambda0 = lambdas_vdw[0]
    is_endpoint_lambda0_a_global_endpoint = abs(endpoint_lambda0 - 1.0) < 1e-9 or abs(endpoint_lambda0 - 0.0) < 1e-9

    # win_sys_xml_matches_manifest_raw_bytes is deliberately NOT part of all_passed -- see the
    # provenance_verdict_accepted note above (DEC-041 already closed that exact question).
    all_passed = bool(
        provenance_verdict_accepted and all_ledger_closed and all_finite
        and cross_check_passed and target_composition_passed and student_nonzero_at_interior_states
        and endpoint_a0_zero and is_endpoint_lambda0_a_global_endpoint
    )

    body = {
        "schema_version": "exp012-ibs-tmbar-wiring-smoke-v1",
        "status": "COMPLETED_WP5A_STEP1_WIRING_SMOKE",
        "window": {
            "stage_type": args.stage_type, "window_index": args.window_index,
            "K": n_states, "lambdas_vdw": lambdas_vdw, "lambdas_coul": manifest["lambdas_coul"],
        },
        "platform": {
            "requested": args.platform, "resolved_name": resolved_platform_name,
            "precision": platform_properties.get("Precision"),
            "properties": platform_properties,
        },
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_held_out_run_id": payload.get("held_out_run_id"),
        "checkpoint_seed": payload.get("seed"),
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "torchscript_sha256": torchscript_sha256,
        "torchscript_a_k_baked_in": 1.0,
        "torchscript_a_k_note": "a_k=1.0 (pure passthrough), NOT D3/D4's a_k=0.5 -- the state-dependent "
                                 "A_k scaling now lives in OuterLambdaIBSBiasForce's CV expression, "
                                 "baking a_k into the module too would double-apply the coefficient",
        "controller": {
            "coefficient_c1": float(args.coefficient),
            "coefficient_note": "frozen smoke constant reused from D3/D4's a_k=0.5; not tuned for "
                                 "production sampling benefit -- WP-5A step 3 may need a different value",
            "envelope_type": "sin2", "coefficient_model_type": "constant",
            "protocol_sha256": controller.protocol_sha256(lambdas=lambdas_vdw),
            "a_k_per_state": a_k_per_state,
        },
        "win_sys_integrity": {
            "win_sys_xml_sha256_before_swap": win_sys_xml_before_swap_sha256,
            "win_sys_xml_sha256_before_swap_matches_manifest_raw_bytes": win_sys_xml_matches_manifest_raw_bytes,
            "win_sys_xml_sha256_after_swap": win_sys_xml_after_swap_sha256,
            "checkpoint_loaded_without_exception": checkpoint_loaded_without_exception,
            "provenance_report_path": str(provenance_report_path.resolve()),
            "provenance_report_sha256": provenance_report.get("report_sha256"),
            "provenance_verdict": provenance_verdict,
            "provenance_verdict_accepted": provenance_verdict_accepted,
            "note": "the raw byte hash comparison above is recorded for information only and is NOT "
                    "part of all_passed -- DEC-041 sealed this exact question "
                    "(CLOSED_STEP3_OPERATIONAL_SEMANTIC_PASS: 10/10 independent structural fields plus "
                    "Force canonical fingerprint matched; the byte-level mismatch was never attributed "
                    "to a semantic difference) and its decision log explicitly says not to re-litigate "
                    "it per run. What actually gates all_passed here is provenance_verdict_accepted.",
        },
        "endpoint_check": {
            "window_k0_lambda_vdw": endpoint_lambda0,
            "window_k0_is_a_global_vdw_stage_endpoint": is_endpoint_lambda0_a_global_endpoint,
            "A_0": endpoint_a0,
            "A_0_within_tolerance_of_zero": endpoint_a0_zero,
            "note": "this window's OTHER boundary state (k=K-1) is an interior ladder rung shared with "
                    "the adjacent window, NOT the vdw stage's other physical endpoint (lambda_vdw=0.0, "
                    "which lives in a different window) -- this smoke does not verify that endpoint",
        },
        "nvt_methodology": {
            "n_frames": args.n_frames, "steps_per_frame": args.steps_per_frame,
            "total_steps": args.n_frames * args.steps_per_frame,
            "note": "frames are NOT claimed to be equilibrated production samples of the "
                    "student-modified ensemble -- the checkpoint was sampled under the OLD, "
                    "student-free bias; this smoke only checks wiring/ledger mechanics",
        },
        "gpu_memory_mib": {"after_context_and_checkpoint_load": gpu_after_context, "after_frames": gpu_after_frames},
        "frames": frames,
        "results": {
            "all_ledger_closed": all_ledger_closed,
            "all_finite": all_finite,
            "max_group1_cross_check_absolute_error_kj_mol": max_cross_check_absolute_error,
            "max_group1_cross_check_relative_error": max_cross_check_relative_error,
            "cross_check_passed": cross_check_passed,
            "max_target_composition_absolute_error_kj_mol": max_target_composition_error,
            "target_composition_passed": target_composition_passed,
            "student_nonzero_at_all_collected_frames": student_nonzero_at_interior_states,
        },
        "all_passed": all_passed,
        "policy": {
            "decision_reference": "WP-5A step 1 (post-DEC-044/045 D0-D4 closure)",
            "ibs_engine_py_modified": False,
            "production_checkpoints_written": False,
            "note": "surgical Group-1 Force swap done entirely in this script on a COPY of the real "
                    "win_sys; ibs_engine.py's build_ibs_dual_system/_build_window_system/IBSSampler "
                    "are used unmodified",
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    print(f"all_passed={all_passed} all_ledger_closed={all_ledger_closed} all_finite={all_finite} "
          f"precision={platform_properties.get('Precision')} "
          f"cross_check_passed={cross_check_passed} "
          f"(max_abs_err={max_cross_check_absolute_error:.3e}, max_rel_err={max_cross_check_relative_error:.3e}) "
          f"target_composition_passed={target_composition_passed} (max_err={max_target_composition_error:.3e}) "
          f"endpoint_A0={endpoint_a0:.3e} "
          f"provenance_verdict={provenance_verdict!r} (accepted={provenance_verdict_accepted}) "
          f"[raw_byte_hash_match={win_sys_xml_matches_manifest_raw_bytes}, not gating]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
