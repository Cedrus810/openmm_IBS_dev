"""Offline complete-MM ledger relabeling for EXP-012 scratch trajectories.

This module reconstructs the exact scratch IBS window, but never advances MD
and never mutates production files.  OpenMM and MDTraj are imported lazily so
the pure numerical contract remains testable on a login node.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .schema import (
    Exp012IntegrityError,
    Exp012ProtocolError,
    Exp012Preregistration,
    load_preregistration,
)


GAS_CONSTANT_KJ_MOL_K = 0.00831446261815324
LEDGER_SCHEMA_VERSION = "exp012-mm-ledger-v1"


def _as_finite_matrix(values: Any, field: str, *, columns: int | None = None):
    import numpy as np

    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0:
        raise Exp012ProtocolError(f"{field} must be a non-empty rank-2 array")
    if columns is not None and result.shape[1] != columns:
        raise Exp012ProtocolError(
            f"{field} must have {columns} columns, got {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise Exp012ProtocolError(f"{field} contains NaN or infinity")
    return result


def _as_finite_vector(values: Any, field: str, *, length: int | None = None):
    import numpy as np

    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise Exp012ProtocolError(f"{field} must be a non-empty vector")
    if length is not None and result.shape != (length,):
        raise Exp012ProtocolError(f"{field} must have length {length}")
    if not np.all(np.isfinite(result)):
        raise Exp012ProtocolError(f"{field} contains NaN or infinity")
    return result


def stable_logsumexp(values: Sequence[float]) -> float:
    normalized = [float(value) for value in values]
    if not normalized or not all(math.isfinite(value) for value in normalized):
        raise Exp012ProtocolError("logsumexp input must be finite and non-empty")
    pivot = max(normalized)
    return pivot + math.log(sum(math.exp(value - pivot) for value in normalized))


def analytic_ibs_bias_kj_mol(
    softcore_cv_kj_mol: Sequence[float],
    f_k_kj_mol: Sequence[float],
    temperature_K: float,
) -> float:
    softcore = [float(value) for value in softcore_cv_kj_mol]
    offsets = [float(value) for value in f_k_kj_mol]
    if len(softcore) != len(offsets) or not softcore:
        raise Exp012ProtocolError("softcore CV and f_k arrays must have equal non-zero length")
    temperature = float(temperature_K)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise Exp012ProtocolError("temperature_K must be finite and positive")
    kT = GAS_CONSTANT_KJ_MOL_K * temperature
    return -kT * stable_logsumexp(
        [-(energy - offset) / kT for energy, offset in zip(softcore, offsets)]
    )


def compose_mm_ledger_arrays(
    *,
    base_energy_kj_mol: Any,
    softcore_cv_kj_mol: Any,
    lrc_kj_mol: Any,
    ibs_bias_kj_mol: Any,
    wca_bias_kj_mol: Any,
    temperature_K: float,
) -> dict[str, Any]:
    """Compose target/sample reduced potentials without hiding raw components."""

    import numpy as np

    softcore = _as_finite_matrix(softcore_cv_kj_mol, "softcore_cv_kj_mol")
    frames, states = softcore.shape
    lrc = _as_finite_matrix(lrc_kj_mol, "lrc_kj_mol", columns=states)
    if lrc.shape != softcore.shape:
        raise Exp012ProtocolError("softcore_cv_kj_mol and lrc_kj_mol shapes differ")
    base = _as_finite_vector(base_energy_kj_mol, "base_energy_kj_mol", length=frames)
    ibs_bias = _as_finite_vector(ibs_bias_kj_mol, "ibs_bias_kj_mol", length=frames)
    wca_bias = _as_finite_vector(wca_bias_kj_mol, "wca_bias_kj_mol", length=frames)
    temperature = float(temperature_K)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise Exp012ProtocolError("temperature_K must be finite and positive")
    beta = 1.0 / (GAS_CONSTANT_KJ_MOL_K * temperature)

    target_interaction = softcore + lrc
    target_total = base[:, None] + target_interaction
    sampling_bias = ibs_bias + wca_bias
    sampling_total = base + sampling_bias
    target_u = beta * target_total
    sampling_u = beta * sampling_total
    # Exact actual-sampling -> target importance numerator.  Base cancels, but
    # is retained above for TMBAR/accounting audits.
    log_importance = sampling_u[:, None] - target_u
    adjacent_gap = np.diff(target_u, axis=1)
    return {
        "target_interaction_kj_mol": target_interaction,
        "target_total_kj_mol": target_total,
        "sampling_bias_kj_mol": sampling_bias,
        "sampling_total_kj_mol": sampling_total,
        "target_reduced_potential": target_u,
        "sampling_reduced_potential": sampling_u,
        "log_importance_unnormalized": log_importance,
        "adjacent_gap_reduced": adjacent_gap,
        "beta_mol_per_kj": float(beta),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Exp012ProtocolError(f"cannot read {field}: {path}") from error
    if not isinstance(value, Mapping):
        raise Exp012ProtocolError(f"{field} must contain an object")
    return value


def _select_run(registration: Exp012Preregistration, run_id: str) -> Mapping[str, Any]:
    runs = registration.payload["inputs"]["runs"]
    selected = [run for run in runs if run["run_id"] == run_id]
    if len(selected) != 1:
        raise Exp012ProtocolError(f"unknown or duplicate run_id: {run_id}")
    return selected[0]


def _resolve_ledger_slice_target(target: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the local ledger states by their IDs in the global schedule.

    The slice boundaries describe stored data, not physical path endpoints.  In
    particular, callers must not infer endpoint behavior from the first/last
    entry returned here.
    """

    try:
        schedule = target["global_schedule"]
        ledger_slice = target["ledger_slice"]
        schedule_ids = list(schedule["global_state_ids"])
        ledger_ids = list(ledger_slice["global_state_ids"])
        ledger_range = list(ledger_slice["schedule_index_range_half_open"])
        schedule_lambdas_coul = list(schedule["lambda_coul"])
        schedule_lambdas_vdw = list(schedule["lambda_vdw"])
        f_k = list(ledger_slice["f_k_kj_mol"])
    except (KeyError, TypeError) as error:
        raise Exp012ProtocolError(
            "target must define a global_schedule and an explicit ledger_slice"
        ) from error

    if (
        not schedule_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in schedule_ids)
        or len(set(schedule_ids)) != len(schedule_ids)
    ):
        raise Exp012ProtocolError("global schedule state IDs must be unique integers")
    if (
        not ledger_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in ledger_ids)
        or len(set(ledger_ids)) != len(ledger_ids)
    ):
        raise Exp012ProtocolError("ledger slice state IDs must be unique integers")
    if len(schedule_lambdas_coul) != len(schedule_ids) or len(schedule_lambdas_vdw) != len(
        schedule_ids
    ):
        raise Exp012ProtocolError("global schedule lambda arrays must match its state IDs")
    if len(ledger_range) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) for value in ledger_range
    ):
        raise Exp012ProtocolError("ledger slice schedule range must contain two integers")
    start, stop = ledger_range
    if start < 0 or stop <= start or stop > len(schedule_ids):
        raise Exp012ProtocolError("ledger slice schedule range is out of bounds")

    schedule_index = {state_id: index for index, state_id in enumerate(schedule_ids)}
    missing = [state_id for state_id in ledger_ids if state_id not in schedule_index]
    if missing:
        raise Exp012ProtocolError(
            f"ledger slice references global state IDs absent from the schedule: {missing}"
        )
    expected_ids = schedule_ids[start:stop]
    if ledger_ids != expected_ids:
        raise Exp012ProtocolError(
            "ledger slice state IDs must preserve the declared global schedule order"
        )
    if len(f_k) != len(ledger_ids):
        raise Exp012ProtocolError("ledger slice f_k must match its state count")
    if ledger_slice.get("boundaries_are_physical_endpoints") is not False:
        raise Exp012ProtocolError("ledger slice boundaries are not physical endpoints")

    def finite_values(values: Sequence[Any], field: str) -> list[float]:
        converted = [float(value) for value in values]
        if not all(math.isfinite(value) for value in converted):
            raise Exp012ProtocolError(f"{field} must contain only finite values")
        return converted

    indices = [schedule_index[state_id] for state_id in ledger_ids]
    lambdas_coul = finite_values(
        [schedule_lambdas_coul[index] for index in indices], "mapped lambda_coul"
    )
    lambdas_vdw = finite_values(
        [schedule_lambdas_vdw[index] for index in indices], "mapped lambda_vdw"
    )
    return {
        "global_state_ids": ledger_ids,
        "schedule_index_range_half_open": [start, stop],
        "lambda_coul": lambdas_coul,
        "lambda_vdw": lambdas_vdw,
        "f_k_kj_mol": finite_values(f_k, "ledger slice f_k_kj_mol"),
    }


def _validate_source_lambdas(
    mapped_values: Sequence[float],
    observed_sources: Sequence[tuple[str, Sequence[Any]]],
    *,
    absolute_tolerance: float = 1.0e-8,
) -> list[float]:
    """Validate source lambdas against mapped global states, retaining precision.

    The preregistered path is rounded to eight decimal places, while the
    historical OpenMM inputs retain additional digits.  Returning the first
    mutually identical source preserves reconstruction of the already-ledgered
    System without weakening the global-state identity check.
    """

    mapped = [float(value) for value in mapped_values]
    if not observed_sources:
        raise Exp012ProtocolError("at least one lambda source is required")
    reference: list[float] | None = None
    for label, observed in observed_sources:
        try:
            values = [float(value) for value in observed]
        except (TypeError, ValueError) as error:
            raise Exp012IntegrityError(f"{label} is not a numeric lambda array") from error
        if len(values) != len(mapped) or not all(math.isfinite(value) for value in values):
            raise Exp012IntegrityError(f"{label} has an invalid lambda array")
        if any(abs(actual - expected) > absolute_tolerance for actual, expected in zip(values, mapped)):
            raise Exp012IntegrityError(
                f"{label} does not map to the preregistered global schedule states"
            )
        if reference is None:
            reference = values
        elif values != reference:
            raise Exp012IntegrityError(f"{label} differs from the other frozen lambda sources")
    assert reference is not None
    return reference


def _resolve_platform(openmm, platform_name: str, device_index: str | None):
    spec = str(platform_name)
    if spec not in {"Reference", "CPU", "CUDA"}:
        raise Exp012ProtocolError("platform must be Reference, CPU, or CUDA")
    platform = openmm.Platform.getPlatformByName(spec)
    properties: dict[str, str] = {}
    if spec == "CUDA":
        properties["Precision"] = "mixed"
        if device_index is not None:
            properties["DeviceIndex"] = str(device_index)
    return platform, properties


def relabel_mm_ledger(
    preregistration_path: str | Path,
    run_id: str,
    output_dir: str | Path,
    *,
    workspace_root: str | Path | None = None,
    platform_name: str = "CPU",
    device_index: str | None = None,
    frame_start: int = 0,
    frame_stop: int | None = None,
    frame_stride: int = 1,
    verify_all_input_hashes: bool = True,
) -> Mapping[str, Any]:
    """Rebuild the scratch window and label selected DCD frames.

    The output directory must not exist or must be empty.  Any identity,
    closure, shape, or finite-value failure aborts without publishing a ledger.
    """

    import numpy as np
    import mdtraj as md
    import openmm
    from openmm import app, unit

    from ibs_engine import WCA_ACCOUNTING_VERSION, build_ibs_dual_system

    root = Path(workspace_root or ".").resolve()
    prereg_path = Path(preregistration_path)
    if not prereg_path.is_absolute():
        prereg_path = root / prereg_path
    registration = load_preregistration(
        prereg_path,
        workspace_root=root,
        verify_files=verify_all_input_hashes,
    )
    run = _select_run(registration, run_id)
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise Exp012IntegrityError(f"refusing to overwrite non-empty output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if isinstance(frame_start, bool) or not isinstance(frame_start, int) or frame_start < 0:
        raise Exp012ProtocolError("frame_start must be a non-negative integer")
    if isinstance(frame_stride, bool) or not isinstance(frame_stride, int) or frame_stride <= 0:
        raise Exp012ProtocolError("frame_stride must be a positive integer")
    expected_frames = int(run["frame_count"])
    stop = expected_frames if frame_stop is None else int(frame_stop)
    if stop <= frame_start or stop > expected_frames:
        raise Exp012ProtocolError("frame_stop must be within the registered trajectory")
    selected_indices = list(range(frame_start, stop, frame_stride))
    if not selected_indices:
        raise Exp012ProtocolError("frame selection is empty")

    artifacts = registration.payload["inputs"]["artifacts"]
    artifact_paths = {
        name: root / record["path"] for name, record in artifacts.items()
    }
    stage = _load_json(artifact_paths["stage_protocol"], "stage protocol")
    manifest = _load_json(artifact_paths["window_manifest"], "window manifest")
    bias_state = _load_json(artifact_paths["frozen_bias_state"], "bias state")
    sample_report_path = root / run["sample_report"]["path"]
    sample_report = _load_json(sample_report_path, "scratch sample report")
    target = registration.payload["target"]
    ledger_target = _resolve_ledger_slice_target(target)
    if list(sample_report.get("window_range", [])) != ledger_target[
        "schedule_index_range_half_open"
    ]:
        raise Exp012IntegrityError("scratch report window range differs from ledger_slice")
    lambdas_coul = _validate_source_lambdas(
        ledger_target["lambda_coul"],
        (
            ("manifest lambda_coul", manifest["lambdas_coul"]),
            ("bias lambda_coul", bias_state["lambdas_coul"]),
            ("report lambda_coul", sample_report["lambdas_coul"]),
        ),
    )
    lambdas_vdw = _validate_source_lambdas(
        ledger_target["lambda_vdw"],
        (
            ("manifest lambda_vdw", manifest["lambdas_vdw"]),
            ("bias lambda_vdw", bias_state["lambdas_vdw"]),
            ("report lambda_vdw", sample_report["lambdas_vdw"]),
        ),
    )
    f_k = ledger_target["f_k_kj_mol"]
    for label, observed in (
        ("bias f_k", bias_state["f_k"]),
        ("report f_k", sample_report["f_k_kj_mol"]),
    ):
        if list(map(float, observed)) != f_k:
            raise Exp012IntegrityError(f"{label} differs from preregistration")
    temperature = float(target["temperature_K"])
    if float(manifest["temperature_K"]) != temperature:
        raise Exp012IntegrityError("temperature differs between preregistration and manifest")
    if int(sample_report["random_seed"]) != int(run["random_seed"]):
        raise Exp012IntegrityError("scratch run seed differs from preregistration")
    expected_scratch_sha = str(sample_report["reconstructed_system_sha256"])
    if sample_report.get("historical_checkpoint_exactly_reconstructable") is not False:
        raise Exp012IntegrityError("scratch report no longer records the expected fresh rebuild")

    protocol = stage["protocol_key"]["payload"]
    topology_file = app.PDBxFile(str(artifact_paths["topology"]))
    base_system = openmm.XmlSerializer.deserialize(
        artifact_paths["base_system"].read_text(encoding="utf-8")
    )
    box_nm = np.asarray(np.load(artifact_paths["box_vectors"]), dtype=np.float64)
    if box_nm.shape != (3, 3) or not np.all(np.isfinite(box_nm)):
        raise Exp012IntegrityError("registered box_vectors.npy is invalid")
    box_vectors = tuple(openmm.Vec3(*row) * unit.nanometer for row in box_nm)
    window_system, ibs_wrapper = build_ibs_dual_system(
        base_system,
        topology_file.topology,
        protocol["ligand_indices"],
        lambdas_coul,
        lambdas_vdw,
        protocol["aces_softcore_params"],
        protocol["potential_type"],
        protocol["boresch_params"],
        temperature * unit.kelvin,
        "abfe_dual",
        box_vectors=box_vectors,
        reference_positions=None,
        dispersion_protocol=None,
    )
    rebuilt_xml = openmm.XmlSerializer.serialize(window_system)
    rebuilt_sha = hashlib.sha256(rebuilt_xml.encode("utf-8")).hexdigest()
    if rebuilt_sha != expected_scratch_sha:
        raise Exp012IntegrityError(
            "current builder does not reproduce the scratch trajectory System: "
            f"expected {expected_scratch_sha}, got {rebuilt_sha}"
        )

    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    platform, properties = _resolve_platform(openmm, platform_name, device_index)
    context = openmm.Context(window_system, integrator, platform, properties)
    context.setPeriodicBoxVectors(*box_vectors)
    parameters = set(context.getParameters())
    parameter_values = {
        "abfe_dual_bias_scale": 1.0,
        "lambda_shield": float(manifest["lambda_shield"]),
        "lambda_boresch_scale": float(manifest["lambda_boresch_scale"]),
    }
    parameter_values.update({f"abfe_dual_f_{index}": value for index, value in enumerate(f_k)})
    for name, value in parameter_values.items():
        if name in parameters:
            context.setParameter(name, float(value))
        elif name != "lambda_boresch_scale":
            raise Exp012IntegrityError(f"rebuilt Context is missing parameter {name}")

    trajectory_path = root / run["trajectory"]["path"]
    trajectory = md.load(str(trajectory_path), top=str(artifact_paths["topology"]))
    if trajectory.n_frames != expected_frames:
        raise Exp012IntegrityError("trajectory frame count differs from preregistration")
    selected = trajectory[selected_indices]
    if selected.unitcell_vectors is None:
        raise Exp012IntegrityError("scratch DCD lacks periodic box vectors")

    arrays: dict[str, list[Any]] = {
        "base_energy_kj_mol": [],
        "softcore_cv_kj_mol": [],
        "lrc_kj_mol": [],
        "ibs_bias_kj_mol": [],
        "wca_bias_kj_mol": [],
        "total_context_kj_mol": [],
        "potential_closure_error_kj_mol": [],
        "ibs_bias_closure_error_kj_mol": [],
        "reported_potential_delta_kj_mol": [],
    }
    kT = GAS_CONSTANT_KJ_MOL_K * temperature
    lrc_coeff = getattr(ibs_wrapper, "lj_tail_lrc_coeff_kj_mol", None)
    if lrc_coeff is None:
        lrc_coeff_array = np.zeros(len(lambdas_vdw), dtype=np.float64)
    else:
        lrc_coeff_array = np.asarray(lrc_coeff, dtype=np.float64)
    if lrc_coeff_array.shape != (len(lambdas_vdw),) or not np.all(np.isfinite(lrc_coeff_array)):
        raise Exp012IntegrityError("rebuilt System produced invalid LRC coefficients")

    started = time.perf_counter()
    diagnostics = sample_report.get("diagnostics", [])
    for local_index, frame_index in enumerate(selected_indices):
        positions = selected.xyz[local_index] * unit.nanometer
        frame_box_nm = np.asarray(selected.unitcell_vectors[local_index], dtype=np.float64)
        frame_box = tuple(openmm.Vec3(*row) * unit.nanometer for row in frame_box_nm)
        context.setPeriodicBoxVectors(*frame_box)
        context.setPositions(positions)
        base = context.getState(getEnergy=True, groups={0, 2, 3, 5}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        ibs_bias = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        wca_bias = context.getState(getEnergy=True, groups={4}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        total = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        cv_values = list(ibs_wrapper.get_force().getCollectiveVariableValues(context))
        if len(cv_values) != 2 * len(lambdas_vdw):
            raise Exp012IntegrityError("IBS CustomCVForce returned an unexpected CV count")
        softcore = np.asarray(cv_values[0::2], dtype=np.float64)
        restraint_cv = np.asarray(cv_values[1::2], dtype=np.float64)
        if not np.all(np.abs(restraint_cv) <= 1.0e-12):
            raise Exp012IntegrityError("state restraint CVs are no longer zero")
        volume = abs(float(np.linalg.det(frame_box_nm)))
        if not math.isfinite(volume) or volume <= 0.0:
            raise Exp012IntegrityError(f"frame {frame_index} has an invalid periodic volume")
        lrc = lrc_coeff_array / volume
        analytic_bias = analytic_ibs_bias_kj_mol(softcore, f_k, temperature)
        potential_closure = total - (base + ibs_bias + wca_bias)
        bias_closure = ibs_bias - analytic_bias
        closure_tolerance = 1.0e-3 + 1.0e-7 * max(1.0, abs(total))
        if not all(
            math.isfinite(value)
            for value in (base, ibs_bias, wca_bias, total, potential_closure, bias_closure)
        ) or not np.all(np.isfinite(softcore)) or not np.all(np.isfinite(lrc)):
            raise Exp012IntegrityError(f"frame {frame_index} contains non-finite energy components")
        if abs(potential_closure) > closure_tolerance:
            raise Exp012IntegrityError(
                f"frame {frame_index} force-group ledger does not close: {potential_closure} kJ/mol"
            )
        if abs(bias_closure) > closure_tolerance:
            raise Exp012IntegrityError(
                f"frame {frame_index} IBS bias does not match its CV/f_k definition: {bias_closure} kJ/mol"
            )
        reported_delta = float("nan")
        if frame_index < len(diagnostics):
            reported = float(diagnostics[frame_index]["potential_energy_kj_mol"])
            reported_delta = total - reported
        arrays["base_energy_kj_mol"].append(base)
        arrays["softcore_cv_kj_mol"].append(softcore)
        arrays["lrc_kj_mol"].append(lrc)
        arrays["ibs_bias_kj_mol"].append(ibs_bias)
        arrays["wca_bias_kj_mol"].append(wca_bias)
        arrays["total_context_kj_mol"].append(total)
        arrays["potential_closure_error_kj_mol"].append(potential_closure)
        arrays["ibs_bias_closure_error_kj_mol"].append(bias_closure)
        arrays["reported_potential_delta_kj_mol"].append(reported_delta)
    elapsed = time.perf_counter() - started

    composed = compose_mm_ledger_arrays(
        base_energy_kj_mol=arrays["base_energy_kj_mol"],
        softcore_cv_kj_mol=arrays["softcore_cv_kj_mol"],
        lrc_kj_mol=arrays["lrc_kj_mol"],
        ibs_bias_kj_mol=arrays["ibs_bias_kj_mol"],
        wca_bias_kj_mol=arrays["wca_bias_kj_mol"],
        temperature_K=temperature,
    )
    saved_arrays = {
        "frame_index": np.asarray(selected_indices, dtype=np.int64),
        **{name: np.asarray(values) for name, values in arrays.items()},
        **{name: value for name, value in composed.items() if name != "beta_mol_per_kj"},
    }
    array_path = destination / "ledger_arrays.npz"
    with array_path.open("wb") as handle:
        np.savez_compressed(handle, **saved_arrays)
    report = {
        "report_type": "exp012_complete_mm_target_ledger",
        "report_version": 1,
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "status": "COMPLETED",
        "run_id": run_id,
        "frame_indices": selected_indices,
        "frame_count": len(selected_indices),
        "state_count": len(lambdas_vdw),
        "global_state_ids": ledger_target["global_state_ids"],
        "schedule_index_range_half_open": ledger_target[
            "schedule_index_range_half_open"
        ],
        "slice_boundaries_are_physical_endpoints": False,
        "temperature_K": temperature,
        "kT_kj_mol": kT,
        "beta_mol_per_kj": composed["beta_mol_per_kj"],
        "energy_accounting": {
            "base_force_groups": [0, 2, 3, 5],
            "ibs_bias_force_group": 1,
            "wca_bias_force_group": 4,
            "target_interaction": "softcore_cv_plus_analytic_lrc",
            "target_total": "base_plus_target_interaction",
            "sampling_total": "base_plus_ibs_bias_plus_wca_bias",
            "log_importance_unnormalized": "sampling_reduced_minus_target_reduced",
            "weights_normalized": False,
            "reduced_potential_beta_applied_exactly_once": True,
            "wca_accounting_version": int(WCA_ACCOUNTING_VERSION),
        },
        "lambdas_coul": lambdas_coul,
        "lambdas_vdw": lambdas_vdw,
        "f_k_kj_mol": f_k,
        "platform": platform_name,
        "platform_properties": properties,
        "scratch_system_sha256_expected": expected_scratch_sha,
        "scratch_system_sha256_rebuilt": rebuilt_sha,
        "historical_production_system_sha256": manifest["win_sys_xml_sha256"],
        "historical_checkpoint_exactly_reconstructable": False,
        "preregistration_path": str(prereg_path.resolve()),
        "preregistration_payload_sha256": registration.payload_sha256,
        "trajectory_path": str(trajectory_path.resolve()),
        "trajectory_sha256": run["trajectory"]["sha256"],
        "arrays_path": str(array_path.resolve()),
        "arrays_sha256": _sha256_file(array_path),
        "elapsed_seconds": elapsed,
        "seconds_per_frame": elapsed / len(selected_indices),
        "max_abs_potential_closure_error_kj_mol": float(
            np.max(np.abs(saved_arrays["potential_closure_error_kj_mol"]))
        ),
        "max_abs_ibs_bias_closure_error_kj_mol": float(
            np.max(np.abs(saved_arrays["ibs_bias_closure_error_kj_mol"]))
        ),
        "reported_potential_delta_kj_mol": {
            "note": "DCD coordinate quantization diagnostic only; not a target-energy gate",
            "max_abs": float(np.nanmax(np.abs(saved_arrays["reported_potential_delta_kj_mol"]))),
        },
        "production_data_mutated": False,
        "scientific_qualification": False,
    }
    report_path = destination / "ledger_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del context, integrator
    return report


__all__ = [
    "GAS_CONSTANT_KJ_MOL_K",
    "LEDGER_SCHEMA_VERSION",
    "analytic_ibs_bias_kj_mol",
    "compose_mm_ledger_arrays",
    "relabel_mm_ledger",
    "stable_logsumexp",
]
