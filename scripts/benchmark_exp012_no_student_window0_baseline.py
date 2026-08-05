#!/usr/bin/env python
"""DEC-039 d0-5 gap 3: real no-student ms/step baseline for hard_window0.

v2 fix (DEC-039): `output_lrc_fix/box_vectors.npy` is written exactly once, at
initial system-cache creation (``runabfe.py``'s ``save_native_system``), and is
NEVER updated afterwards. The box vectors actually baked into window 0's real
System (and hashed into ``manifest["win_sys_xml_sha256"]`` by real production)
are whatever ``pipeline.box_vectors`` held in memory at the moment
``run_all_windows`` built this window -- which reflects at least one later
in-place reassignment (post-pre-equilibration NPT relaxation via
``abfe_pipeline.py``'s ``pre_equilibrate()``, and/or post-Boresch-rebalance via
``_rebalance_with_boresch``). ``box_vectors.npy`` on disk reflects neither and
is stale, so v1 of this script's own ``win_sys_xml_sha256_matches_manifest``
self-check spuriously read ``false`` even though the actual Context state
(positions/velocities/box) restored via ``loadCheckpoint`` -- and therefore the
timing measurement itself -- was already correct.

v2 fixes the self-check (not the timing, which was already valid) with a
two-pass construction: first build a throwaway *probe* System using the stale
``box_vectors.npy`` value, whose only purpose is to give ``loadCheckpoint`` a
valid Context to restore into; read the checkpoint's real box vectors back off
that Context; then rebuild the *real* System used for everything else
(hashing, timing) with those checkpoint-derived box vectors. This is valid
specifically because ``hard_window0``'s production System carries no
``MonteCarloBarostat`` (verified at runtime below, fail-closed if that
assumption is ever wrong) -- once a barostat-free window is built its box never
changes again for the rest of that window's production, so the box vectors
recorded in ANY later checkpoint of that window (including the one loaded
here) equal the box vectors used when the window was first built and hashed.
No production code, checkpoint, or protocol/cache fingerprint is touched by
this fix -- only this standalone diagnostic script.

The neural-path timing budget (target/ceiling added wall-clock per MD step)
must be defined relative to a real measurement of the CURRENT production
setup, not an absolute guess. This script reconstructs the exact real
production window-0 System/Integrator/Context for the ``output_lrc_fix`` run
(``runabfe.py --mode ibs --potential softcore --platform CUDA --preset
production``, decoupling=dual_lambda) -- with NO neural/student Force added
anywhere -- restores it from the real, already-on-disk production checkpoint
(not a fresh minimize/warmup ramp), and times ``Simulation.step()`` in the
same 500-step chunks production itself uses (``steps_per_update``).

Everything this script loads is a real production artifact, not a
reconstruction from guessed parameters:

- ``output_lrc_fix/system_native.xml`` -- the native (pre-window) System,
  integrity-checked against ``stage2_vanishing.json``'s recorded SHA-256.
- ``output_lrc_fix/checkpoints/stage2_vanishing.json`` -- ligand_indices,
  softcore alpha parameters, Boresch restraint parameters, potential_type.
- ``output_lrc_fix/checkpoints/production_window/<stage>/window_<n>/manifest.json``
  and its sibling ``openmm.chk`` -- the exact per-window lambda schedule and a
  real mid-production checkpoint (positions/velocities/box/integrator RNG
  state), used via ``Simulation.loadCheckpoint`` instead of re-minimizing or
  re-running the warmup ramp.
- ``output_lrc_fix/checkpoints/ibs_state_<stage>_window_<n>.json`` -- the
  frozen IBS bias weights (``f_k``), re-applied via
  ``IBSBiasForce.update_parameters`` exactly as production's own resume path
  does after loading a checkpoint.

``environment_type="soluble"`` and ``dispersion_protocol="legacy_uniform_density_lrc"``
are read directly from ``abfe_config.json``'s own documented default resolution
(``system_type``/``membrane`` are both ``null`` in that file, which its own
inline comment states resolves to soluble + legacy LRC) -- not guessed.

No student model, no TorchForce, no training, and no NVT run anywhere in this
script -- this measures the CURRENT production cost floor a neural path's
budget must be defined against.
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
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.environment import canonical_json_bytes  # noqa: E402

_STEPS_PER_UPDATE = 500  # matches production's steps_per_update (stage2_vanishing.json.run_config)
_ENVIRONMENT_TYPE = "soluble"  # abfe_config.json: system_type=null, membrane=null -> soluble
_DISPERSION_PROTOCOL = "legacy_uniform_density_lrc"  # abfe_config.json: dispersion_protocol=null, non-membrane -> this


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    parser.add_argument(
        "--stage-type", default="vdw", choices=["vdw"],
        help="only vdw (hard_window0's stage) is wired up; coul's stage-protocol filename is unverified",
    )
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--platform", default="CUDA", help="matches manifest.platform_name unless overridden")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-chunks", type=int, default=2, help="500-step chunks run and discarded first")
    parser.add_argument("--chunks-per-repeat", type=int, default=4, help="500-step chunks per timed repeat")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.repeats < 3:
        parser.error("--repeats must be at least 3 (median/P95 over fewer repeats is not meaningful)")
    if args.warmup_chunks < 1:
        parser.error("--warmup-chunks must be at least 1 (CUDA kernel JIT warmup)")
    if args.chunks_per_repeat < 1:
        parser.error("--chunks-per-repeat must be positive")
    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite a frozen report: {args.output}")

    import numpy as np
    import openmm
    from openmm import XmlSerializer, app, unit

    from ibs_engine import (  # noqa: E402  (heavy import; real production module)
        ACESoftcorePotential,
        _build_platform_properties,
        _gpu_memory_mib,
        _system_has_global_parameter,
        build_ibs_dual_system,
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

    for path in (manifest_path, checkpoint_path, stage_protocol_path, ibs_state_path,
                 system_xml_path, topology_cif_path, box_vectors_path):
        if not path.is_file():
            raise RuntimeError(f"required real production artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_payload = json.loads(stage_protocol_path.read_text(encoding="utf-8"))["protocol_key"]["payload"]
    ibs_state = json.loads(ibs_state_path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    system_xml_text = system_xml_path.read_text(encoding="utf-8")
    observed_system_xml_sha256 = _sha256_text(system_xml_text)
    if observed_system_xml_sha256 != stage_payload["system_xml_sha256"]:
        raise RuntimeError(
            "system_native.xml SHA-256 does not match stage2 protocol record; "
            "refusing to benchmark an unverified System"
        )
    base_system = XmlSerializer.deserialize(system_xml_text)
    topology = app.PDBxFile(str(topology_cif_path)).topology
    stale_box_vectors = unit.Quantity(np.load(box_vectors_path), unit.nanometer)
    stale_box_vectors_sha256 = _sha256_file(box_vectors_path)

    alchemical_params = ACESoftcorePotential.from_dict(stage_payload["aces_softcore_params"])

    resolved_platform_name, platform_properties = _build_platform_properties(args.platform)
    platform = openmm.Platform.getPlatformByName(resolved_platform_name)

    # ---- pass 1: throwaway probe System/Context, stale box, just to load the
    #      checkpoint and read back the box vectors production actually used ----
    probe_win_sys, _probe_ibs_wrap = build_ibs_dual_system(
        base_system, topology, stage_payload["ligand_indices"],
        manifest["lambdas_coul"], manifest["lambdas_vdw"], alchemical_params,
        potential_type=stage_payload["potential_type"],
        restraint_params=stage_payload["boresch_params"],
        temperature=manifest["temperature_K"] * unit.kelvin,
        prefix=ibs_state["prefix"],
        box_vectors=stale_box_vectors, reference_positions=None,
        dispersion_protocol=_DISPERSION_PROTOCOL, environment_type=_ENVIRONMENT_TYPE,
    )
    probe_has_barostat = any(
        isinstance(force, openmm.MonteCarloBarostat) for force in probe_win_sys.getForces()
    )
    if probe_has_barostat:
        raise RuntimeError(
            "hard_window0's System unexpectedly carries a MonteCarloBarostat. The "
            "checkpoint-derived-box shortcut (DEC-039 gap-3 fix) assumes a fixed "
            "box for this window's entire production (true only in the absence of "
            "a barostat) -- refusing to silently trust a stale/derived box when "
            "that assumption does not hold."
        )
    probe_integrator = openmm.LangevinMiddleIntegrator(
        manifest["temperature_K"] * unit.kelvin,
        manifest["friction_per_ps"] / unit.picosecond,
        manifest["step_size_ps"] * unit.picosecond,
    )
    probe_simulation = app.Simulation(topology, probe_win_sys, probe_integrator, platform, platform_properties)
    probe_simulation.loadCheckpoint(str(checkpoint_path))
    checkpoint_box_vectors = probe_simulation.context.getState(getPositions=True).getPeriodicBoxVectors()
    _stale_nm = np.array(stale_box_vectors.value_in_unit(unit.nanometer))
    _checkpoint_nm = np.array(checkpoint_box_vectors.value_in_unit(unit.nanometer))
    box_vectors_identical_to_stale = bool(np.array_equal(_stale_nm, _checkpoint_nm))
    box_vectors_max_abs_diff_nm = float(np.max(np.abs(_stale_nm - _checkpoint_nm)))
    del probe_simulation, probe_integrator, probe_win_sys, _probe_ibs_wrap

    # ---- pass 2: real System, built with the checkpoint-derived (correct) box ----
    box_vectors = checkpoint_box_vectors
    win_sys, ibs_wrap = build_ibs_dual_system(
        base_system, topology, stage_payload["ligand_indices"],
        manifest["lambdas_coul"], manifest["lambdas_vdw"], alchemical_params,
        potential_type=stage_payload["potential_type"],
        restraint_params=stage_payload["boresch_params"],
        temperature=manifest["temperature_K"] * unit.kelvin,
        prefix=ibs_state["prefix"],
        box_vectors=box_vectors, reference_positions=None,
        dispersion_protocol=_DISPERSION_PROTOCOL, environment_type=_ENVIRONMENT_TYPE,
    )
    observed_win_sys_xml_sha256 = _sha256_text(XmlSerializer.serialize(win_sys))
    win_sys_xml_matches_manifest = observed_win_sys_xml_sha256 == manifest["win_sys_xml_sha256"]

    force_groups = sorted(
        (
            {"force_class": force.__class__.__name__, "force_group": int(force.getForceGroup())}
            for force in win_sys.getForces()
        ),
        key=lambda item: (item["force_group"], item["force_class"]),
    )

    integrator = openmm.LangevinMiddleIntegrator(
        manifest["temperature_K"] * unit.kelvin,
        manifest["friction_per_ps"] / unit.picosecond,
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
    ibs_wrap.update_parameters(simulation.context, np.asarray(ibs_state["f_k"], dtype=float))

    gpu_memory_after_context_mib = _gpu_memory_mib()

    for _ in range(args.warmup_chunks):
        simulation.step(_STEPS_PER_UPDATE)

    gpu_memory_after_warmup_mib = _gpu_memory_mib()

    repeats = []
    for repeat_index in range(args.repeats):
        repeat_started = time.perf_counter()
        for _ in range(args.chunks_per_repeat):
            simulation.step(_STEPS_PER_UPDATE)
        elapsed_seconds = time.perf_counter() - repeat_started
        total_steps = _STEPS_PER_UPDATE * args.chunks_per_repeat
        ms_per_step = 1000.0 * elapsed_seconds / total_steps
        repeats.append(
            {
                "repeat_index": repeat_index,
                "total_steps": total_steps,
                "elapsed_seconds": elapsed_seconds,
                "ms_per_step": ms_per_step,
            }
        )
        print(f"repeat {repeat_index}: {total_steps} steps in {elapsed_seconds:.3f}s -> {ms_per_step:.4f} ms/step", flush=True)

    gpu_memory_after_repeats_mib = _gpu_memory_mib()
    ms_per_step_values = [repeat["ms_per_step"] for repeat in repeats]

    def _mib_report(value):
        if value is None:
            return None
        used, free, total = value
        return {"used_mib": used, "free_mib": free, "total_mib": total}

    body = {
        "schema_version": "exp012-no-student-window0-baseline-v2",
        "window": {
            "stage_type": args.stage_type,
            "window_index": args.window_index,
            "K": manifest["K"],
            "lambdas_vdw": manifest["lambdas_vdw"],
            "lambdas_coul": manifest["lambdas_coul"],
            "lambda_shield": manifest["lambda_shield"],
            "lambda_boresch_scale": manifest["lambda_boresch_scale"],
        },
        "integrator": {
            "class": "LangevinMiddleIntegrator",
            "temperature_K": manifest["temperature_K"],
            "friction_per_ps": manifest["friction_per_ps"],
            "step_size_ps": manifest["step_size_ps"],
            "constraint_tolerance": 1e-3,
        },
        "platform": {
            "requested": args.platform,
            "resolved_name": resolved_platform_name,
            "properties": platform_properties,
        },
        "force_groups": force_groups,
        "integrity": {
            "system_native_xml_sha256_matches_stage2_protocol": True,
            "win_sys_xml_sha256_matches_manifest": win_sys_xml_matches_manifest,
            "manifest_win_sys_xml_sha256": manifest["win_sys_xml_sha256"],
            "observed_win_sys_xml_sha256": observed_win_sys_xml_sha256,
            "restored_from_real_production_checkpoint": True,
            "checkpoint_path": str(checkpoint_path),
            "box_vectors_dec039_gap3_fix": {
                "method": "two_pass_checkpoint_derived_box (v2)",
                "note": "box_vectors.npy is written once at initial system-cache creation "
                        "and never updated; the real box used to build+hash window 0 is "
                        "whatever pipeline.box_vectors held at run_all_windows time, which "
                        "reflects later in-memory reassignments (post-pre-equilibration "
                        "NPT relax and/or post-Boresch-rebalance) never written back to "
                        "box_vectors.npy. A throwaway probe System/Context (built with the "
                        "stale box) loads the real checkpoint and reads back its box "
                        "vectors; the real System is then rebuilt with that box. Valid "
                        "because this window's System carries no MonteCarloBarostat "
                        "(checked below, fail-closed) -- box is frozen for the whole "
                        "window once built, so any later checkpoint's box equals the "
                        "build-time box.",
                "probe_system_has_barostat": probe_has_barostat,
                "stale_box_vectors_npy_sha256": stale_box_vectors_sha256,
                "stale_box_vectors_nm": _stale_nm.tolist(),
                "checkpoint_derived_box_vectors_nm": _checkpoint_nm.tolist(),
                "stale_identical_to_checkpoint_derived": box_vectors_identical_to_stale,
                "stale_vs_checkpoint_derived_max_abs_diff_nm": box_vectors_max_abs_diff_nm,
            },
        },
        "timing_methodology": {
            "steps_per_chunk": _STEPS_PER_UPDATE,
            "warmup_chunks_discarded": args.warmup_chunks,
            "chunks_per_repeat": args.chunks_per_repeat,
            "repeats": args.repeats,
            "note": "each repeat times Simulation.step(500) calls in a loop, matching production's own "
                    "steps_per_update call granularity; warmup chunks absorb CUDA kernel JIT cost and are excluded",
        },
        "gpu_memory": {
            "after_context_and_checkpoint_load": _mib_report(gpu_memory_after_context_mib),
            "after_warmup": _mib_report(gpu_memory_after_warmup_mib),
            "after_timed_repeats": _mib_report(gpu_memory_after_repeats_mib),
        },
        "repeats": repeats,
        "ms_per_step_summary": {
            "median": _percentile(ms_per_step_values, 0.5),
            "p95": _percentile(ms_per_step_values, 0.95),
            "min": min(ms_per_step_values),
            "max": max(ms_per_step_values),
            "mean": sum(ms_per_step_values) / len(ms_per_step_values),
        },
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)},
            "stage_protocol": {"path": str(stage_protocol_path), "sha256": _sha256_file(stage_protocol_path)},
            "ibs_state": {"path": str(ibs_state_path), "sha256": _sha256_file(ibs_state_path)},
            "system_native_xml": {"path": str(system_xml_path), "sha256": observed_system_xml_sha256},
            "topology_cif": {"path": str(topology_cif_path), "sha256": _sha256_file(topology_cif_path)},
            "box_vectors_npy_stale_reference_only": {
                "path": str(box_vectors_path),
                "sha256": stale_box_vectors_sha256,
                "note": "NOT what was used to build win_sys -- see "
                        "integrity.box_vectors_dec039_gap3_fix for the checkpoint-derived "
                        "box vectors actually used",
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
        "policy": {
            "student_model_executed": False,
            "torchforce_used": False,
            "training_executed": False,
            "nvt_executed": False,
            "note": "this is the current no-student production cost floor; the neural-path timing "
                    "budget in DEC-039 must be defined as an overhead relative to this measurement",
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    print(
        f"median_ms_per_step={report['ms_per_step_summary']['median']:.4f} "
        f"p95_ms_per_step={report['ms_per_step_summary']['p95']:.4f} "
        f"win_sys_xml_sha256_matches_manifest={win_sys_xml_matches_manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
