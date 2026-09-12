"""Fail-closed preregistration contract for the EXP-012 offline diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


_EXP012_STAGE2_LAMBDA_PATH_SHA256 = (
    "32571526d5074db285c2b15b4debf60d0c03811b4139c0e2b139f2264c4561cf"
)


class Exp012ProtocolError(ValueError):
    """The preregistration is incomplete or internally inconsistent."""


class Exp012IntegrityError(Exp012ProtocolError):
    """A frozen input or the sealed payload does not match its digest."""


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _payload_for_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(_canonical_bytes(payload))
    normalized.get("freeze", {}).pop("payload_sha256", None)
    return normalized


def preregistration_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(_payload_for_digest(payload))).hexdigest()


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Exp012ProtocolError(f"{field} must be an object")
    return value


def _require_sequence(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise Exp012ProtocolError(f"{field} must be an array")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Exp012ProtocolError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise Exp012ProtocolError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise Exp012ProtocolError(f"{field} must stay inside the workspace")
    return value


def _finite_sequence(value: Any, field: str) -> list[float]:
    values = _require_sequence(value, field)
    result: list[float] = []
    for index, item in enumerate(values):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise Exp012ProtocolError(f"{field}[{index}] must be numeric")
        converted = float(item)
        if not math.isfinite(converted):
            raise Exp012ProtocolError(f"{field}[{index}] must be finite")
        result.append(converted)
    return result


@dataclass(frozen=True)
class Exp012Preregistration:
    payload: Mapping[str, Any]
    payload_sha256: str
    status: str
    unresolved: tuple[str, ...]

    @property
    def executable(self) -> bool:
        return self.status == "sealed" and not self.unresolved

    def require_executable(self) -> None:
        if not self.executable:
            raise Exp012ProtocolError(
                "EXP-012 diagnostic is blocked until the preregistration is sealed "
                "and all unresolved fields are eliminated"
            )


def validate_preregistration(
    payload: Mapping[str, Any],
    *,
    workspace_root: str | Path | None = None,
    verify_files: bool = False,
    require_sealed: bool = False,
) -> Exp012Preregistration:
    root = Path(workspace_root or ".").resolve()
    schema_version = payload.get("schema_version")
    if schema_version not in {"exp012-prereg-v1", "exp012-local-residual-prereg-v2"}:
        raise Exp012ProtocolError("unsupported EXP-012 preregistration schema")
    if payload.get("experiment_id") != "EXP-012":
        raise Exp012ProtocolError("experiment_id must be EXP-012")
    expected_stage = (
        "feature_only_abc"
        if schema_version == "exp012-prereg-v1"
        else "local_residual_abcd"
    )
    if payload.get("stage") != expected_stage:
        raise Exp012ProtocolError(f"stage must be {expected_stage}")

    freeze = _require_mapping(payload.get("freeze"), "freeze")
    status = freeze.get("status")
    if status not in {"draft", "sealed"}:
        raise Exp012ProtocolError("freeze.status must be draft or sealed")
    if freeze.get("allow_postseal_override") is not False:
        raise Exp012ProtocolError("post-seal overrides must be disabled")
    unresolved_values = _require_sequence(payload.get("unresolved"), "unresolved")
    if any(not isinstance(item, str) or not item for item in unresolved_values):
        raise Exp012ProtocolError("unresolved entries must be non-empty strings")
    if len(set(unresolved_values)) != len(unresolved_values):
        raise Exp012ProtocolError("unresolved entries must be unique")
    if status == "draft" and not unresolved_values:
        raise Exp012ProtocolError("a draft must state at least one unresolved field")
    digest = preregistration_sha256(payload)
    if status == "sealed":
        if unresolved_values:
            raise Exp012ProtocolError("a sealed preregistration cannot be unresolved")
        recorded = _require_sha256(freeze.get("payload_sha256"), "freeze.payload_sha256")
        if recorded != digest:
            raise Exp012IntegrityError("sealed preregistration payload digest mismatch")

    inputs = _require_mapping(payload.get("inputs"), "inputs")
    artifacts = _require_mapping(inputs.get("artifacts"), "inputs.artifacts")
    if not artifacts:
        raise Exp012ProtocolError("at least one frozen artifact is required")
    file_records: list[tuple[str, str, str]] = []
    for name, record_value in artifacts.items():
        record = _require_mapping(record_value, f"inputs.artifacts.{name}")
        path = _require_relative_path(record.get("path"), f"inputs.artifacts.{name}.path")
        sha = _require_sha256(record.get("sha256"), f"inputs.artifacts.{name}.sha256")
        file_records.append((f"artifact {name}", path, sha))

    runs = _require_sequence(inputs.get("runs"), "inputs.runs")
    if len(runs) < 3:
        raise Exp012ProtocolError("at least three independent coordinate runs are required")
    run_ids: list[str] = []
    for index, run_value in enumerate(runs):
        run = _require_mapping(run_value, f"inputs.runs[{index}]")
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise Exp012ProtocolError(f"inputs.runs[{index}].run_id is invalid")
        run_ids.append(run_id)
        if isinstance(run.get("random_seed"), bool) or not isinstance(run.get("random_seed"), int):
            raise Exp012ProtocolError(f"inputs.runs[{index}].random_seed must be an integer")
        if isinstance(run.get("frame_count"), bool) or not isinstance(run.get("frame_count"), int) or run["frame_count"] <= 0:
            raise Exp012ProtocolError(f"inputs.runs[{index}].frame_count must be positive")
        for label in ("trajectory", "sample_report"):
            record = _require_mapping(run.get(label), f"inputs.runs[{index}].{label}")
            path = _require_relative_path(record.get("path"), f"inputs.runs[{index}].{label}.path")
            sha = _require_sha256(record.get("sha256"), f"inputs.runs[{index}].{label}.sha256")
            file_records.append((f"run {run_id} {label}", path, sha))
        if "ledger" in run:
            ledger = _require_mapping(run.get("ledger"), f"inputs.runs[{index}].ledger")
            path = _require_relative_path(ledger.get("path"), f"inputs.runs[{index}].ledger.path")
            sha = _require_sha256(ledger.get("sha256"), f"inputs.runs[{index}].ledger.sha256")
            file_records.append((f"run {run_id} ledger", path, sha))
            ledger_report = _require_mapping(
                run.get("ledger_report"), f"inputs.runs[{index}].ledger_report"
            )
            report_path = _require_relative_path(
                ledger_report.get("path"), f"inputs.runs[{index}].ledger_report.path"
            )
            report_sha = _require_sha256(
                ledger_report.get("sha256"), f"inputs.runs[{index}].ledger_report.sha256"
            )
            file_records.append((f"run {run_id} ledger report", report_path, report_sha))
        elif status == "sealed":
            raise Exp012ProtocolError(f"inputs.runs[{index}].ledger is required when sealed")
    if len(set(run_ids)) != len(run_ids):
        raise Exp012ProtocolError("run IDs must be unique")

    split = _require_mapping(payload.get("split"), "split")
    if split.get("unit") != "whole_run" or split.get("random_frame_split") is not False:
        raise Exp012ProtocolError("only whole-run, non-random frame splitting is allowed")
    folds = _require_sequence(split.get("folds"), "split.folds")
    if not folds:
        raise Exp012ProtocolError("at least one explicit fold is required")
    tested: list[str] = []
    known_runs = set(run_ids)
    for index, fold_value in enumerate(folds):
        fold = _require_mapping(fold_value, f"split.folds[{index}]")
        role_sets = []
        for role in ("train_run_ids", "validation_run_ids", "test_run_ids"):
            ids = _require_sequence(fold.get(role), f"split.folds[{index}].{role}")
            if not ids or any(item not in known_runs for item in ids):
                raise Exp012ProtocolError(f"split.folds[{index}].{role} references invalid runs")
            role_sets.append(set(ids))
        if any(role_sets[left] & role_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
            raise Exp012ProtocolError(f"split.folds[{index}] leaks runs across roles")
        tested.extend(fold["test_run_ids"])
    if set(tested) != known_runs:
        raise Exp012ProtocolError("every run must appear in a held-out test role")

    target = _require_mapping(payload.get("target"), "target")
    if schema_version == "exp012-prereg-v1":
        state_ids = _require_sequence(target.get("global_state_ids"), "target.global_state_ids")
        lambdas_coul = _finite_sequence(target.get("lambda_coul"), "target.lambda_coul")
        lambdas_vdw = _finite_sequence(target.get("lambda_vdw"), "target.lambda_vdw")
        f_k = _finite_sequence(target.get("f_k_kj_mol"), "target.f_k_kj_mol")
        if not state_ids or len({*state_ids}) != len(state_ids):
            raise Exp012ProtocolError("global state IDs must be non-empty and unique")
        if (
            len(state_ids) != len(lambdas_coul)
            or len(state_ids) != len(lambdas_vdw)
            or len(state_ids) != len(f_k)
        ):
            raise Exp012ProtocolError("state, lambda, and f_k arrays must have equal length")
        if target.get("window_boundaries_are_physical_endpoints") is not False:
            raise Exp012ProtocolError(
                "local window boundaries must not be treated as physical endpoints"
            )
        if status == "sealed":
            coefficients = _finite_sequence(target.get("A_k"), "target.A_k")
            if len(coefficients) != len(state_ids):
                raise Exp012ProtocolError("target.A_k must match the global state count")
    else:
        schedule = _require_mapping(target.get("global_schedule"), "target.global_schedule")
        if schedule.get("stage") != "vanishing":
            raise Exp012ProtocolError("target.global_schedule.stage must be vanishing")
        source_artifact_key = schedule.get("source_artifact_key")
        if source_artifact_key != "stage_protocol" or source_artifact_key not in artifacts:
            raise Exp012ProtocolError(
                "global Stage-2 schedule must be bound to inputs.artifacts.stage_protocol"
            )

        fingerprint = _require_mapping(
            schedule.get("lambda_path_fingerprint"),
            "target.global_schedule.lambda_path_fingerprint",
        )
        if fingerprint.get("schema_version") != 1:
            raise Exp012ProtocolError("lambda path fingerprint schema_version must be 1")
        fingerprint_sha = _require_sha256(
            fingerprint.get("sha256"),
            "target.global_schedule.lambda_path_fingerprint.sha256",
        )
        fingerprint_payload = _require_mapping(
            fingerprint.get("payload"),
            "target.global_schedule.lambda_path_fingerprint.payload",
        )
        computed_fingerprint_sha = hashlib.sha256(
            _canonical_bytes(fingerprint_payload)
        ).hexdigest()
        if fingerprint_sha != computed_fingerprint_sha:
            raise Exp012IntegrityError("lambda path fingerprint digest mismatch")
        if fingerprint_sha != _EXP012_STAGE2_LAMBDA_PATH_SHA256:
            raise Exp012IntegrityError("lambda path fingerprint is not the frozen Stage-2 path")
        fingerprint_lambdas = _finite_sequence(
            fingerprint_payload.get("lambdas_var"),
            "target.global_schedule.lambda_path_fingerprint.payload.lambdas_var",
        )
        fingerprint_windows = _require_sequence(
            fingerprint_payload.get("window_ranges"),
            "target.global_schedule.lambda_path_fingerprint.payload.window_ranges",
        )

        state_ids = _require_sequence(
            schedule.get("global_state_ids"), "target.global_schedule.global_state_ids"
        )
        lambdas_coul = _finite_sequence(
            schedule.get("lambda_coul"), "target.global_schedule.lambda_coul"
        )
        lambdas_vdw = _finite_sequence(
            schedule.get("lambda_vdw"), "target.global_schedule.lambda_vdw"
        )
        coefficients = _finite_sequence(
            schedule.get("A_k"), "target.global_schedule.A_k"
        )
        expected_state_ids = list(range(23))
        if state_ids != expected_state_ids:
            raise Exp012ProtocolError(
                "global Stage-2 schedule must map exactly to state IDs 0 through 22"
            )
        if not (
            len(lambdas_coul)
            == len(lambdas_vdw)
            == len(coefficients)
            == len(state_ids)
        ):
            raise Exp012ProtocolError(
                "global Stage-2 state, lambda, and A_k arrays must all have length 23"
            )
        if any(value != 0.0 for value in lambdas_coul):
            raise Exp012ProtocolError("global Stage-2 lambda_coul must be zero at all 23 states")
        if lambdas_vdw != fingerprint_lambdas:
            raise Exp012IntegrityError(
                "global lambda_vdw schedule does not match its hashed path fingerprint"
            )
        if schedule.get("physical_endpoint_global_state_ids") != [0, 22]:
            raise Exp012ProtocolError("physical endpoints must be global state IDs 0 and 22")
        if schedule.get("A_definition") != "sin_squared_pi_lambda_vdw":
            raise Exp012ProtocolError("target.global_schedule.A_definition changed")
        if schedule.get("A_formula") != "sin^2(pi * lambda_vdw)":
            raise Exp012ProtocolError("target.global_schedule.A_formula changed")
        expected_coefficients = [math.sin(math.pi * value) ** 2 for value in lambdas_vdw]
        expected_coefficients[0] = 0.0
        expected_coefficients[-1] = 0.0
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)
            for actual, expected in zip(coefficients, expected_coefficients)
        ):
            raise Exp012ProtocolError("target.global_schedule.A_k does not match A_definition")
        if coefficients[0] != 0.0 or coefficients[-1] != 0.0:
            raise Exp012ProtocolError("physical global endpoint A_k values must be exactly zero")

        ledger_slice = _require_mapping(target.get("ledger_slice"), "target.ledger_slice")
        ledger_state_ids = _require_sequence(
            ledger_slice.get("global_state_ids"), "target.ledger_slice.global_state_ids"
        )
        ledger_range = _require_sequence(
            ledger_slice.get("schedule_index_range_half_open"),
            "target.ledger_slice.schedule_index_range_half_open",
        )
        ledger_f_k = _finite_sequence(
            ledger_slice.get("f_k_kj_mol"), "target.ledger_slice.f_k_kj_mol"
        )
        if ledger_slice.get("source_window_index") != 0:
            raise Exp012ProtocolError("the existing ledger must remain source window 0")
        if not fingerprint_windows or fingerprint_windows[0] != [0, 5]:
            raise Exp012ProtocolError("hashed Stage-2 fingerprint must define window 0 as [0, 5)")
        if ledger_range != [0, 5] or ledger_state_ids != state_ids[0:5]:
            raise Exp012ProtocolError(
                "ledger slice must map exactly to global Stage-2 state IDs [0, 1, 2, 3, 4]"
            )
        if len(ledger_f_k) != len(ledger_state_ids):
            raise Exp012ProtocolError("ledger slice f_k length must match its global state mapping")
        if ledger_slice.get("boundaries_are_physical_endpoints") is not False:
            raise Exp012ProtocolError(
                "ledger slice boundaries must not be treated as physical endpoints"
            )
        if coefficients[ledger_state_ids[-1]] == 0.0:
            raise Exp012ProtocolError(
                "ledger slice state 4 is not the physical endpoint and its A_k must be nonzero"
            )
    if target.get("gap_direction") != "u_kplus1_minus_u_k":
        raise Exp012ProtocolError("the adjacent-gap direction must be explicit")
    if target.get("reduced_potential_units") != "dimensionless":
        raise Exp012ProtocolError("target reduced potentials must be dimensionless")
    if target.get("beta_application") != "ledger_already_reduced":
        raise Exp012ProtocolError("the ledger must own the kJ/mol to reduced conversion")
    if "weighting" in target:
        weighting = _require_mapping(target.get("weighting"), "target.weighting")
        required_weighting = {
            "method": "actual_sampling_to_target_importance_v1",
            "log_weight_unnormalized": "sampling_reduced_minus_target_reduced",
            "equivalent_kj_mol_formula": "beta_times_sampling_bias_minus_target_interaction",
            "sampling_bias_force_groups": [1, 4],
            "normalization": "per_target_state_per_split_partition",
            "clipping": "forbidden",
            "base_energy_treatment": "retained_in_ledger_cancels_in_log_weight",
        }
        for field, expected in required_weighting.items():
            if weighting.get(field) != expected:
                raise Exp012ProtocolError(f"target.weighting.{field} changed")
    features = _require_mapping(payload.get("features"), "features")
    if features.get("cutoff_family") != "quintic_c2":
        raise Exp012ProtocolError("EXP-012 requires the pre-registered C2 quintic cutoff")
    arms = _require_mapping(features.get("arms"), "features.arms")
    if schema_version == "exp012-prereg-v1":
        if arms.get("A") != "typed_atom_centered_radial_contact":
            raise Exp012ProtocolError("Arm A definition changed")
        if arms.get("B") != "A_plus_fixed_xed_offcenter_angular":
            raise Exp012ProtocolError("Arm B must inherit A and add fixed XED angular fields")
        if arms.get("C") != "B_plus_smooth_gaussian_overlap":
            raise Exp012ProtocolError("Arm C must inherit B and add smooth overlap fields")
    else:
        expected_arms = {
            "A": "typed_atom_centered_rbf_contact",
            "B": "lightweight_equivariant_ligand_environment_cross_encoder",
            "C": "frozen_mace_node_latent_plus_invariant_mlp",
            "D": "optional_xed_inspired_field",
        }
        if dict(arms) != expected_arms:
            raise Exp012ProtocolError("local-residual A/B/C/D representation definitions changed")
        if features.get("method_identity") != "cv_free_local_residual_path_potential":
            raise Exp012ProtocolError("EXP-012 v2 method identity must be CV-free local residual")
        if features.get("predefined_slow_cv") is not False:
            raise Exp012ProtocolError("EXP-012 v2 must not predefine a slow CV")
        if features.get("mace_final_interaction_energy_allowed") is not False:
            raise Exp012ProtocolError("Arm C must use frozen node latents, not final MACE energy")
        if features.get("fragment_energy_subtraction_allowed") is not False:
            raise Exp012ProtocolError("fragment energy subtraction remains forbidden")

    if verify_files:
        for label, relative, expected in file_records:
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise Exp012IntegrityError(f"{label} escaped the workspace") from error
            if not path.is_file():
                raise Exp012IntegrityError(f"missing {label}: {path}")
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != expected:
                raise Exp012IntegrityError(f"SHA-256 mismatch for {label}: {path}")

    result = Exp012Preregistration(payload, digest, str(status), tuple(unresolved_values))
    if require_sealed:
        result.require_executable()
    return result


def load_preregistration(
    path: str | Path,
    *,
    workspace_root: str | Path | None = None,
    verify_files: bool = False,
    require_sealed: bool = False,
) -> Exp012Preregistration:
    prereg_path = Path(path)
    try:
        payload = json.loads(prereg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Exp012ProtocolError(f"cannot read EXP-012 preregistration: {prereg_path}") from error
    return validate_preregistration(
        _require_mapping(payload, "root"),
        workspace_root=workspace_root,
        verify_files=verify_files,
        require_sealed=require_sealed,
    )


__all__ = [
    "Exp012IntegrityError",
    "Exp012ProtocolError",
    "Exp012Preregistration",
    "load_preregistration",
    "preregistration_sha256",
    "validate_preregistration",
]
