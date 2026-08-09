#!/usr/bin/env python
"""DEC-037 D3, sub-items 1+2: eager vs TorchScript vs CPU/CUDA consistency.

Three-way chain on real frames, to isolate which layer any discrepancy comes
from (not just "deployment differs from training", but specifically where):

(a) reference eager: the same `LocalResidualStudent.forward()` call D1/D2
    already used (cached distances via `ligand_environment_cross_edges` +
    `reindex_ligand_environment_edges`), output in reduced units.
(b) deployable wrapper, eager: `local_residual.student_deploy`'s
    `_DeployableStudent.forward(positions_nm, box_nm)` -- a from-scratch
    reimplementation of the same funnel+reindex+network logic made
    TorchScript-compatible -- called WITHOUT scripting first, output
    converted from kJ/mol back to reduced units via the same a_k/kT used to
    build it. Comparing (a) vs (b) isolates "did the deployable
    reimplementation introduce a bug", independent of scripting.
(c) deployable wrapper, scripted: `torch.jit.script(...)` of the same
    module, same inputs. Comparing (b) vs (c) isolates "did scripting change
    behavior", independent of the reimplementation itself.

Then (c) is run on CPU float32, CPU float64, and CUDA float32 (if available)
and compared, isolating precision/device effects from everything above.

Energy is checked directly; force is checked via autograd on positions (both
(a) and (b)/(c) support gradient w.r.t. their respective position inputs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.geometry import ligand_environment_cross_edges  # noqa: E402
from local_residual.student import (  # noqa: E402
    build_local_residual_student,
    reindex_ligand_environment_edges,
)
from local_residual.student_deploy import build_deployable_student_module, export_torchscript  # noqa: E402


class D3CheckError(RuntimeError):
    """A checkpoint/frame/comparison failed a fail-closed contract check."""


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


def _reference_eager_energy_and_force(model, positions, box, ligand_tensor, environment_tensor, atomic_number_by_topology_index):
    import torch

    positions = positions.clone().detach().requires_grad_(True)
    edges = ligand_environment_cross_edges(
        positions, box, ligand_tensor, environment_tensor, outer_cutoff=model.outer_cutoff_angstrom,
    )
    ligand_topology_indices = ligand_tensor.tolist()
    reindexed = reindex_ligand_environment_edges(
        ligand_topology_indices, edges["edge_index"][0], edges["edge_index"][1]
    )
    ligand_type_index = model.atomic_numbers_to_type_index(
        [atomic_number_by_topology_index[index] for index in ligand_topology_indices]
    )
    if reindexed["environment_topology_indices"].numel() == 0:
        environment_type_index = torch.empty((0,), dtype=torch.int64)
    else:
        environment_type_index = model.atomic_numbers_to_type_index(
            [atomic_number_by_topology_index[int(index)] for index in reindexed["environment_topology_indices"].tolist()]
        )
    basis_reduced = model(
        ligand_type_index, environment_type_index,
        reindexed["edge_ligand_local"], reindexed["edge_environment_local"], edges["distance"],
    )
    basis_reduced.backward()
    return float(basis_reduced.item()), positions.grad.clone()


def _module_energy_and_force(module, positions_nm, box_nm):
    positions_nm = positions_nm.clone().detach().requires_grad_(True)
    energy = module(positions_nm, box_nm)
    energy.backward()
    return float(energy.item()), positions_nm.grad.clone()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="a direct_gap .pt checkpoint from student_checkpoints/")
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", required=True, help="a single real trajectory to probe a frame from")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--ligand-indices", required=True)
    parser.add_argument("--temperature-kelvin", type=float, default=300.0)
    parser.add_argument("--a-k", type=float, default=1.0, help="frozen envelope coefficient for this D3 smoke; not a real production A_k wiring")
    parser.add_argument(
        "--correctness-tolerance-reduced", type=float, default=1e-8,
        help="same-precision (float64 vs float64) comparisons: reference_eager vs deployable_eager "
             "(does the reimplementation match), and deployable_eager vs deployable_scripted (does "
             "scripting change behavior). A real logic bug, not precision loss, would show up here.",
    )
    parser.add_argument(
        "--precision-envelope-tolerance-reduced", type=float, default=5e-4,
        help="float64 vs float32 of the SAME scripted module: some difference is EXPECTED (float32 "
             "has ~7 decimal digits), this is not a correctness check. Do not tighten this to the "
             "correctness tolerance -- that conflates precision loss with a logic bug.",
    )
    parser.add_argument(
        "--device-consistency-tolerance-reduced", type=float, default=1e-4,
        help="CPU float32 vs CUDA float32: same precision, different device/kernel implementation. "
             "Looser than pure-correctness (GPU reduction order differs from CPU) but much tighter "
             "than the precision envelope, since both sides already lost the same order of precision.",
    )
    parser.add_argument("--torchscript-output", required=True, help="where to save the scripted .pt module")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite a frozen report: {args.output}")
    if Path(args.torchscript_output).exists():
        parser.error(f"--torchscript-output already exists, refusing to overwrite: {args.torchscript_output}")

    import mdtraj
    import torch

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("variant") != "direct_gap":
        raise D3CheckError(f"--checkpoint variant={payload.get('variant')!r}, only direct_gap is a D3 candidate")
    model = build_local_residual_student(payload["type_vocabulary"], **payload["model_kwargs"])
    model.load_state_dict(payload["state_dict"])
    model = model.to(torch.float64)
    model.eval()

    ligand_payload = json.loads(Path(args.ligand_indices).expanduser().read_text(encoding="utf-8"))
    ligand_topology_indices = sorted(int(index) for index in ligand_payload["ligand_indices"])

    trajectory = mdtraj.load_frame(args.trajectory, index=args.frame_index, top=args.topology)
    if trajectory.unitcell_vectors is None:
        raise D3CheckError("frame has no periodic box vectors")
    # mdtraj stores xyz/unitcell_vectors as float32. Upcast to float64 FIRST,
    # then scale nm->Angstrom in torch float64 -- not the other way around
    # (`trajectory.xyz[0] * 10.0` in numpy would multiply while still
    # float32, baking in ~1e-7 relative rounding error before this script's
    # own float64 comparison ever starts). This must match, bit-for-bit,
    # however `student_deploy.py`'s `_DeployableStudent.forward()` derives
    # its own Angstrom positions (upcast then scale) -- these two positions
    # tensors are the "same frame" only if they are constructed identically.
    positions_nm = torch.tensor(trajectory.xyz[0], dtype=torch.float64)
    box_nm = torch.tensor(trajectory.unitcell_vectors[0], dtype=torch.float64)
    positions_angstrom = positions_nm * 10.0
    box_angstrom = box_nm * 10.0
    n_atoms = trajectory.topology.n_atoms
    atomic_number_by_topology_index = {
        index: int(atom.element.atomic_number) for index, atom in enumerate(trajectory.topology.atoms)
    }
    all_topology_atomic_numbers = [atomic_number_by_topology_index[i] for i in range(n_atoms)]
    ligand_tensor = torch.tensor(ligand_topology_indices, dtype=torch.int64)
    environment_tensor = torch.tensor(sorted(set(range(n_atoms)) - set(ligand_topology_indices)), dtype=torch.int64)

    # (a) reference eager
    ref_basis_reduced, ref_force = _reference_eager_energy_and_force(
        model, positions_angstrom, box_angstrom, ligand_tensor, environment_tensor, atomic_number_by_topology_index,
    )

    # (b) deployable wrapper, eager
    deployable = build_deployable_student_module(
        model, ligand_topology_indices=ligand_topology_indices,
        all_topology_atomic_numbers=all_topology_atomic_numbers,
        temperature_kelvin=args.temperature_kelvin, a_k=args.a_k,
    ).to(torch.float64)
    deployable.eval()
    gas_constant_kj_per_mol_k = 0.0083144621
    kt_kj_per_mol = gas_constant_kj_per_mol_k * args.temperature_kelvin
    conversion = args.a_k * kt_kj_per_mol

    wrapper_energy_kj, wrapper_force_nm = _module_energy_and_force(deployable, positions_nm, box_nm)
    wrapper_basis_reduced = wrapper_energy_kj / conversion
    # x_angstrom = 10 * x_nm, so d(energy)/d(x_nm) = 10 * d(energy)/d(x_angstrom)
    # (chain rule) -- the autograd result above is d(energy)/d(x_nm), so
    # d(basis_reduced)/d(x_angstrom) = [d(energy)/d(x_nm) / 10] / conversion,
    # i.e. divide by 10 (not multiply) to undo the nm->Angstrom derivative scaling.
    wrapper_force_reduced_per_angstrom = (wrapper_force_nm / 10.0) / conversion

    # (c) deployable wrapper, scripted
    scripted_sha256 = export_torchscript(deployable, args.torchscript_output)
    scripted = torch.jit.load(args.torchscript_output)
    scripted = scripted.to(torch.float64)
    scripted.eval()
    scripted_energy_kj, scripted_force_nm = _module_energy_and_force(scripted, positions_nm, box_nm)
    scripted_basis_reduced = scripted_energy_kj / conversion
    scripted_force_reduced_per_angstrom = (scripted_force_nm / 10.0) / conversion

    def _compare(name_a, energy_a, force_a, name_b, energy_b, force_b, *, category, tolerance):
        energy_abs_err = abs(energy_a - energy_b)
        force_abs_err = float((force_a - force_b).abs().max().item())
        return {
            "comparison": f"{name_a}_vs_{name_b}",
            "tolerance_category": category,
            "tolerance_reduced": tolerance,
            f"{name_a}_basis_reduced": energy_a,
            f"{name_b}_basis_reduced": energy_b,
            "energy_absolute_error": energy_abs_err,
            "force_max_absolute_error": force_abs_err,
            "passed": bool(energy_abs_err <= tolerance and force_abs_err <= tolerance),
        }

    # Three-way comparison matrix (per explicit instruction: never compare
    # CPU64 directly against CUDA32 with one uniform tolerance -- each pairing
    # below tests a DIFFERENT thing and gets the tolerance appropriate to it).
    comparisons = [
        # CPU64 reference (independent reimplementation of the model math)
        # vs CPU64 deployable wrapper -- same precision on both sides, so any
        # real logic bug shows up here undiluted by float32 rounding.
        _compare(
            "reference_eager", ref_basis_reduced, ref_force,
            "deployable_eager", wrapper_basis_reduced, wrapper_force_reduced_per_angstrom,
            category="correctness_cpu64_vs_cpu64", tolerance=args.correctness_tolerance_reduced,
        ),
        # CPU64 eager vs CPU64 scripted -- does torch.jit.script itself change
        # behavior; also same precision on both sides.
        _compare(
            "deployable_eager", wrapper_basis_reduced, wrapper_force_reduced_per_angstrom,
            "deployable_scripted_cpu_float64", scripted_basis_reduced, scripted_force_reduced_per_angstrom,
            category="correctness_cpu64_vs_cpu64", tolerance=args.correctness_tolerance_reduced,
        ),
    ]

    # CPU float32
    scripted_f32 = torch.jit.load(args.torchscript_output).to(torch.float32)
    scripted_f32.eval()
    e32, f32_force_nm = _module_energy_and_force(scripted_f32, positions_nm.to(torch.float32), box_nm.to(torch.float32))
    basis32 = e32 / conversion
    force32_reduced = (f32_force_nm.to(torch.float64) / 10.0) / conversion
    comparisons.append(
        # CPU64 vs CPU32 of the SAME scripted module -- precision envelope,
        # not a correctness check: float32 losing ~1e-4..1e-5 here is expected.
        _compare(
            "deployable_scripted_cpu_float64", scripted_basis_reduced, scripted_force_reduced_per_angstrom,
            "deployable_scripted_cpu_float32", basis32, force32_reduced,
            category="precision_envelope_cpu64_vs_cpu32", tolerance=args.precision_envelope_tolerance_reduced,
        )
    )

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        scripted_cuda = torch.jit.load(args.torchscript_output).to(device="cuda", dtype=torch.float32)
        scripted_cuda.eval()
        e_cuda, f_cuda_nm = _module_energy_and_force(
            scripted_cuda, positions_nm.to(device="cuda", dtype=torch.float32), box_nm.to(device="cuda", dtype=torch.float32)
        )
        basis_cuda = e_cuda / conversion
        force_cuda_reduced = (f_cuda_nm.to("cpu", torch.float64) / 10.0) / conversion
        comparisons.append(
            # CPU32 vs CUDA32 -- same precision, different device/kernel/
            # reduction-order implementation. Deliberately NOT compared
            # against CPU64 directly (that would conflate device differences
            # with the much larger float32 precision envelope).
            _compare(
                "deployable_scripted_cpu_float32", basis32, force32_reduced,
                "deployable_scripted_cuda_float32", basis_cuda, force_cuda_reduced,
                category="device_consistency_cpu32_vs_cuda32", tolerance=args.device_consistency_tolerance_reduced,
            )
        )

    all_passed = all(comparison["passed"] for comparison in comparisons)

    body = {
        "schema_version": "exp012-student-d3-deployment-consistency-v1",
        "status": "COMPLETED_D3_1_2_CHECKS",
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_held_out_run_id": payload.get("held_out_run_id"),
        "checkpoint_seed": payload.get("seed"),
        "torchscript_output_path": str(Path(args.torchscript_output).resolve()),
        "torchscript_sha256": scripted_sha256,
        "a_k_used_for_this_smoke": args.a_k,
        "a_k_note": "frozen constant for this D3 smoke only; real per-window/per-state A_k wiring into the "
                    "production multi-state IBS Hamiltonian is separate, later production-integration work",
        "temperature_kelvin": args.temperature_kelvin,
        "cuda_available": cuda_available,
        "comparisons": comparisons,
        "all_passed": all_passed,
        "policy": {
            "decision_reference": "DEC-037 D3, sub-items 1+2",
            "torchforce_used": False,
            "openmm_used": False,
            "nvt_executed": False,
            "note": "pure-Torch consistency only; D3 sub-items 3 (TorchForce/OpenMM Reference injection) "
                    "and 4 (real production timing/memory) are separate scripts",
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    print(f"all_passed={all_passed}")
    for comparison in comparisons:
        print(f"  {comparison['comparison']}: passed={comparison['passed']} "
              f"energy_err={comparison['energy_absolute_error']:.3e} "
              f"force_err={comparison['force_max_absolute_error']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
