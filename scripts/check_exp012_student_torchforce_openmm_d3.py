#!/usr/bin/env python
"""DEC-037 D3, sub-item 3: TorchForce/OpenMM Reference injection + endpoint zeroing.

Reuses the existing WP-1/2/3 deployment harness in `outer_lambda_neural_basis.py`
(`build_torchforce_from_spec`, `evaluate_openmm_outer_lambda_force`) rather than
building a second, parallel TorchForce/OpenMM integration path. That harness's
`NeuralBasisModelSpec.atom_selection` is only validated as `"fixed_indices"` by
`NeuralBasisModelSpec.from_mapping`'s config loader -- the dataclass itself has
no `__post_init__` and `build_torchforce_from_spec`/`evaluate_openmm_outer_lambda_force`
never look at `atom_selection` at all -- so the spec is constructed directly
here (bypassing `from_mapping`) with `atom_selection="dynamic_funnel_environment"`,
an honest label for `LocalResidualStudent`'s actual design (DEC-038), not the
static "fixed_indices" this field's stricter config-loading path enforces.

Two checks:

1. TorchForce/OpenMM Reference consistency: `evaluate_openmm_outer_lambda_force`
   builds a throwaway, minimal OpenMM System containing ONLY this one Force
   (mass=1 per particle, no real MM forces -- it is not testing the full
   production Hamiltonian, only whether OpenMM's Reference platform correctly
   executes and differentiates this exact scripted module), evaluated on a
   real frame's real positions/box. Its energy/forces are compared against
   direct Torch evaluation of the same scripted module on the same inputs.
2. Endpoint zeroing (PLAN doc §3, `A_0 = A_K = 0`): a second export with
   `a_k=0.0` must produce (near-)exactly zero energy and force through the
   same TorchForce/OpenMM path, confirming the physical endpoints are
   untouched regardless of anything OpenMM-side.

This does not inject into the real production win_sys (that is a separate,
bigger production-integration step, not part of D3's deployment-readiness
validation) and does not wire per-window/per-state A_k into a CustomCVForce
composition across a whole IBS ensemble.
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


class D3TorchForceCheckError(RuntimeError):
    """A checkpoint/frame/TorchForce evaluation failed a fail-closed contract check."""


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


def _direct_torch_energy_and_force(module_path: str, positions_nm, box_nm):
    import torch

    module = torch.jit.load(module_path).to(torch.float64)
    module.eval()
    positions_nm = positions_nm.clone().detach().requires_grad_(True)
    energy = module(positions_nm, box_nm)
    energy.backward()
    return float(energy.item()), positions_nm.grad.clone()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--ligand-indices", required=True)
    parser.add_argument("--temperature-kelvin", type=float, default=300.0)
    parser.add_argument("--a-k", type=float, default=0.5)
    parser.add_argument("--energy-tolerance-kj-mol", type=float, default=1e-6)
    parser.add_argument("--force-tolerance-kj-mol-nm", type=float, default=1e-4)
    parser.add_argument("--endpoint-energy-tolerance-kj-mol", type=float, default=1e-9)
    parser.add_argument("--work-dir", required=True, help="directory for the two exported .pt modules")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite a frozen report: {args.output}")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    nonzero_path = work_dir / "student_torchforce_a_k_nonzero.pt"
    zero_path = work_dir / "student_torchforce_a_k_zero.pt"
    for path in (nonzero_path, zero_path):
        if path.exists():
            parser.error(f"{path} already exists, refusing to overwrite")

    import mdtraj
    import torch

    from outer_lambda_neural_basis import (  # noqa: E402
        NeuralBasisModelSpec,
        build_torchforce_from_spec,
        evaluate_openmm_outer_lambda_force,
    )

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("variant") != "direct_gap":
        raise D3TorchForceCheckError(f"--checkpoint variant={payload.get('variant')!r}, only direct_gap is a D3 candidate")
    model = build_local_residual_student(payload["type_vocabulary"], **payload["model_kwargs"]).to(torch.float64)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    ligand_payload = json.loads(Path(args.ligand_indices).expanduser().read_text(encoding="utf-8"))
    ligand_topology_indices = sorted(int(index) for index in ligand_payload["ligand_indices"])

    trajectory = mdtraj.load_frame(args.trajectory, index=args.frame_index, top=args.topology)
    if trajectory.unitcell_vectors is None:
        raise D3TorchForceCheckError("frame has no periodic box vectors")
    positions_nm_np = trajectory.xyz[0].astype("float64")
    box_nm_np = trajectory.unitcell_vectors[0].astype("float64")
    positions_nm = torch.tensor(positions_nm_np, dtype=torch.float64)
    box_nm = torch.tensor(box_nm_np, dtype=torch.float64)
    n_atoms = trajectory.topology.n_atoms
    all_topology_atomic_numbers = [int(atom.element.atomic_number) for atom in trajectory.topology.atoms]

    ligand_indices_sha256 = _sha256_file(Path(args.ligand_indices))

    def _export_and_wrap(a_k: float, out_path: Path):
        deployable = build_deployable_student_module(
            model, ligand_topology_indices=ligand_topology_indices,
            all_topology_atomic_numbers=all_topology_atomic_numbers,
            temperature_kelvin=args.temperature_kelvin, a_k=a_k,
        ).to(torch.float64)
        deployable.eval()
        sha = export_torchscript(deployable, out_path)
        spec = NeuralBasisModelSpec(
            name=f"local_residual_student_a_k_{a_k}",
            backend="torchforce",
            model_path=str(out_path.resolve()),
            sha256=sha,
            energy_offset_kj_mol=0.0,
            atom_selection="dynamic_funnel_environment",
            atom_indices_path=str(Path(args.ligand_indices).resolve()),
            atom_indices_sha256=ligand_indices_sha256,
            output_unit="kJ_per_mol",
            precision="double",
            periodic=True,
        )
        return spec, sha

    # --- 1. TorchForce/OpenMM Reference consistency, real a_k ---
    spec, torchscript_sha256 = _export_and_wrap(args.a_k, nonzero_path)
    force = build_torchforce_from_spec(spec)
    evaluation = evaluate_openmm_outer_lambda_force(
        force, lambda_value=0.5,
        positions_nm=positions_nm_np.tolist(),
        particle_masses_dalton=[1.0] * n_atoms,
        box_vectors_nm=box_nm_np.tolist(),
        platform_name="Reference",
    )
    direct_energy, direct_force_nm = _direct_torch_energy_and_force(str(nonzero_path), positions_nm, box_nm)

    openmm_forces = torch.tensor(evaluation.forces_kj_mol_nm, dtype=torch.float64)
    # evaluate_openmm_outer_lambda_force reports OpenMM's FORCE convention
    # (F = -dE/dx); direct_force_nm above is the raw autograd GRADIENT
    # (dE/dx), so negate before comparing.
    force_diff = (openmm_forces - (-direct_force_nm)).abs().max().item()
    energy_diff = abs(evaluation.energy_kj_mol - direct_energy)

    torchforce_consistency = {
        "openmm_energy_kj_mol": evaluation.energy_kj_mol,
        "direct_torch_energy_kj_mol": direct_energy,
        "energy_absolute_error_kj_mol": energy_diff,
        "openmm_max_force_norm_kj_mol_nm": evaluation.max_force_norm_kj_mol_nm,
        "force_max_absolute_error_kj_mol_nm": force_diff,
        "passed": bool(
            energy_diff <= args.energy_tolerance_kj_mol and force_diff <= args.force_tolerance_kj_mol_nm
        ),
    }

    # --- 2. Endpoint zeroing, a_k=0.0 ---
    zero_spec, zero_torchscript_sha256 = _export_and_wrap(0.0, zero_path)
    zero_force = build_torchforce_from_spec(zero_spec)
    zero_evaluation = evaluate_openmm_outer_lambda_force(
        zero_force, lambda_value=0.5,
        positions_nm=positions_nm_np.tolist(),
        particle_masses_dalton=[1.0] * n_atoms,
        box_vectors_nm=box_nm_np.tolist(),
        platform_name="Reference",
    )
    endpoint_zeroing = {
        "a_k": 0.0,
        "openmm_energy_kj_mol": zero_evaluation.energy_kj_mol,
        "openmm_max_force_norm_kj_mol_nm": zero_evaluation.max_force_norm_kj_mol_nm,
        "passed": bool(
            abs(zero_evaluation.energy_kj_mol) <= args.endpoint_energy_tolerance_kj_mol
            and zero_evaluation.max_force_norm_kj_mol_nm <= args.endpoint_energy_tolerance_kj_mol
        ),
    }

    all_passed = bool(torchforce_consistency["passed"] and endpoint_zeroing["passed"])

    body = {
        "schema_version": "exp012-student-d3-torchforce-openmm-v1",
        "status": "COMPLETED_D3_3_CHECKS",
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_held_out_run_id": payload.get("held_out_run_id"),
        "checkpoint_seed": payload.get("seed"),
        "a_k_used": args.a_k,
        "a_k_note": "frozen constant for this D3 smoke only; real per-window/per-state A_k wiring into the "
                    "production multi-state IBS Hamiltonian (CustomCVForce composition across a window) is "
                    "separate, later production-integration work",
        "temperature_kelvin": args.temperature_kelvin,
        "torchscript_nonzero_a_k_path": str(nonzero_path.resolve()),
        "torchscript_nonzero_a_k_sha256": torchscript_sha256,
        "torchscript_zero_a_k_path": str(zero_path.resolve()),
        "torchscript_zero_a_k_sha256": zero_torchscript_sha256,
        "torchforce_openmm_reference_consistency": torchforce_consistency,
        "endpoint_zeroing": endpoint_zeroing,
        "all_passed": all_passed,
        "policy": {
            "decision_reference": "DEC-037 D3, sub-item 3",
            "torchforce_used": True,
            "openmm_platform": "Reference",
            "injected_into_real_production_win_sys": False,
            "custom_cv_force_multi_state_wiring": False,
            "nvt_executed": False,
            "note": "Force evaluated alone in a throwaway minimal OpenMM System via the existing "
                    "outer_lambda_neural_basis.evaluate_openmm_outer_lambda_force harness, not injected "
                    "into the real production win_sys; full production Hamiltonian integration is separate, "
                    "later work",
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    print(f"all_passed={all_passed}")
    print(f"  torchforce_consistency: passed={torchforce_consistency['passed']} "
          f"energy_err={energy_diff:.3e} force_err={force_diff:.3e}")
    print(f"  endpoint_zeroing: passed={endpoint_zeroing['passed']} "
          f"energy={zero_evaluation.energy_kj_mol:.3e} max_force={zero_evaluation.max_force_norm_kj_mol_nm:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
