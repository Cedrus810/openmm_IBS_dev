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
    """A self-contained draft-status fixture for exercising the validator's
    logic in isolation.

    This intentionally does NOT read the live `protocols/EXP-012_preregistration.json`
    (unlike an earlier version of this helper): that file is the real,
    mutable project artifact, and its `freeze.status` legitimately changed
    from "draft" to "sealed" on 2026-08-05 (DEC-039/§11A.12 arm retirement +
    reseal). Tests of the schema *validator's* behavior (does it reject role
    leakage, a corrupted fingerprint, a length mismatch, ...) must not depend
    on whichever state the real preregistration happens to be in at the time
    -- that coupling is exactly what broke every test below the moment the
    real file was sealed for real, permanent, correct reasons. The scientific
    content here (lambda schedule, A_k envelope, ledger slice, run records)
    is snapshotted from the real file since it's real, frozen, cross-checked
    data; only the freeze bookkeeping is forced back to an ordinary draft
    shape so mutate-one-field-and-expect-a-specific-error tests don't trip
    the (correct, fail-closed) "sealed payload digest must match" check
    before they ever reach the specific behavior they're testing.
    """

    payload = json.loads(PREREG.read_text(encoding="utf-8"))
    payload["freeze"] = {
        "status": "draft",
        "allow_postseal_override": False,
        "source_identity": payload["freeze"]["source_identity"],
    }
    payload["unresolved"] = ["schema_test_fixture_placeholder"]
    return payload


def test_repository_is_currently_sealed_and_executable():
    """Regression check on the real, live artifact (not the `_draft()` fixture):
    the real preregistration was resealed on 2026-08-05 (arm retirement +
    (d0-5) freeze, DEC-039) and is expected to stay sealed and executable
    from here on. If this ever fails, either the real file regressed to
    draft, or its content changed without the digest being recomputed --
    both are real problems, not something to silence by reverting this test.
    """
    registration = load_preregistration(PREREG, workspace_root=ROOT)
    assert registration.status == "sealed"
    assert registration.executable
    assert registration.unresolved == ()


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
    payload = _draft()
    with pytest.raises(Exp012ProtocolError, match="blocked"):
        validate_preregistration(payload, require_sealed=True)
