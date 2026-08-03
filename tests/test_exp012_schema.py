import copy
import json
from pathlib import Path

import pytest

from exp012_xed.schema import (
    Exp012IntegrityError,
    Exp012ProtocolError,
    load_preregistration,
    preregistration_sha256,
    validate_preregistration,
)


pytestmark = pytest.mark.cpu_only


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "protocols" / "EXP-012_preregistration.json"


def _draft():
    return json.loads(PREREG.read_text(encoding="utf-8"))


def test_repository_draft_is_valid_but_not_executable():
    registration = load_preregistration(PREREG, workspace_root=ROOT)
    assert registration.status == "draft"
    assert not registration.executable
    assert "inputs.runs[*].ledger" not in registration.unresolved
    assert "target.A_k" not in registration.unresolved
    with pytest.raises(Exp012ProtocolError, match="blocked"):
        registration.require_executable()


def test_repository_draft_freezes_global_stage2_schedule_and_local_ledger_slice():
    payload = _draft()
    schedule = payload["target"]["global_schedule"]
    ledger_slice = payload["target"]["ledger_slice"]

    assert schedule["global_state_ids"] == list(range(23))
    assert len(schedule["lambda_coul"]) == 23
    assert set(schedule["lambda_coul"]) == {0.0}
    assert len(schedule["lambda_vdw"]) == 23
    assert schedule["physical_endpoint_global_state_ids"] == [0, 22]
    assert schedule["A_definition"] == "sin_squared_pi_lambda_vdw"
    assert len(schedule["A_k"]) == 23
    assert schedule["A_k"][0] == 0.0
    assert schedule["A_k"][22] == 0.0
    assert schedule["A_k"][4] != 0.0
    assert ledger_slice["global_state_ids"] == [0, 1, 2, 3, 4]
    assert ledger_slice["schedule_index_range_half_open"] == [0, 5]
    assert ledger_slice["boundaries_are_physical_endpoints"] is False


def test_repository_draft_verifies_frozen_small_artifacts(tmp_path):
    payload = _draft()
    # Full verification would intentionally hash the three 441 MB trajectories.
    # Replace file records with tiny fixtures to exercise the same integrity path.
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"exp012")
    digest = __import__("hashlib").sha256(b"exp012").hexdigest()
    relative = artifact.relative_to(tmp_path).as_posix()
    payload["inputs"]["artifacts"] = {
        "stage_protocol": {"path": relative, "sha256": digest}
    }
    for run in payload["inputs"]["runs"]:
        run["trajectory"] = {"path": relative, "sha256": digest}
        run["sample_report"] = {"path": relative, "sha256": digest}
        run["ledger"] = {"path": relative, "sha256": digest}
        run["ledger_report"] = {"path": relative, "sha256": digest}
    registration = validate_preregistration(payload, workspace_root=tmp_path, verify_files=True)
    assert registration.payload_sha256 == preregistration_sha256(payload)


def test_whole_run_role_leakage_is_rejected():
    payload = _draft()
    payload["split"]["folds"][0]["validation_run_ids"] = ["hard_window0_run2"]
    with pytest.raises(Exp012ProtocolError, match="leaks"):
        validate_preregistration(payload)


def test_local_window_boundaries_cannot_be_declared_physical_endpoints():
    payload = _draft()
    payload["target"]["ledger_slice"]["boundaries_are_physical_endpoints"] = True
    with pytest.raises(Exp012ProtocolError, match="physical endpoints"):
        validate_preregistration(payload)


def test_ledger_slice_end_cannot_be_substituted_for_global_physical_endpoint():
    payload = _draft()
    payload["target"]["global_schedule"]["physical_endpoint_global_state_ids"] = [0, 4]
    with pytest.raises(Exp012ProtocolError, match="global state IDs 0 and 22"):
        validate_preregistration(payload)


def test_global_schedule_length_mismatch_is_rejected():
    payload = _draft()
    payload["target"]["global_schedule"]["lambda_coul"].pop()
    with pytest.raises(Exp012ProtocolError, match="length 23"):
        validate_preregistration(payload)


def test_ledger_to_global_state_mapping_mismatch_is_rejected():
    payload = _draft()
    payload["target"]["ledger_slice"]["global_state_ids"][-1] = 5
    with pytest.raises(Exp012ProtocolError, match="map exactly"):
        validate_preregistration(payload)


def test_lambda_path_fingerprint_digest_mismatch_is_rejected():
    payload = _draft()
    payload["target"]["global_schedule"]["lambda_path_fingerprint"]["sha256"] = "0" * 64
    with pytest.raises(Exp012IntegrityError, match="fingerprint digest mismatch"):
        validate_preregistration(payload)


def test_lambda_schedule_must_match_hashed_fingerprint_payload():
    payload = _draft()
    payload["target"]["global_schedule"]["lambda_vdw"][4] += 1e-6
    with pytest.raises(Exp012IntegrityError, match="does not match"):
        validate_preregistration(payload)


def test_Ak_must_follow_frozen_envelope_and_keep_local_state4_nonzero():
    payload = _draft()
    payload["target"]["global_schedule"]["A_k"][4] = 0.0
    with pytest.raises(Exp012ProtocolError, match="does not match A_definition"):
        validate_preregistration(payload)


def test_sealed_payload_requires_ledgers_Ak_and_matching_digest():
    payload = _draft()
    payload["freeze"]["status"] = "sealed"
    payload["unresolved"] = []
    for run in payload["inputs"]["runs"]:
        run["ledger"] = copy.deepcopy(run["sample_report"])
        run["ledger_report"] = copy.deepcopy(run["sample_report"])
    payload["freeze"]["payload_sha256"] = preregistration_sha256(payload)
    registration = validate_preregistration(payload, require_sealed=True)
    assert registration.executable

    tampered = copy.deepcopy(payload)
    tampered["target"]["global_schedule"]["A_k"][2] += 1e-3
    with pytest.raises(Exp012IntegrityError, match="digest mismatch"):
        validate_preregistration(tampered)


def test_draft_cannot_be_required_as_sealed():
    with pytest.raises(Exp012ProtocolError, match="blocked"):
        load_preregistration(PREREG, workspace_root=ROOT, require_sealed=True)
