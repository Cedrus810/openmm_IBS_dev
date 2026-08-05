#!/usr/bin/env python
"""DEC-039: compute and stamp the real payload_sha256 for
``protocols/EXP-012_preregistration.json``, then verify the resealed document
validates end-to-end (including that every referenced artifact/run file still
exists on disk with a matching SHA-256).

This is a small, standalone, in-place patch script -- it does not touch any
production module, checkpoint, or protocol version. It assumes the content
edits (arm A/B/D retirement, environment_selection, readout,
training_budget_and_seeds, evaluation.numeric_gates, decision deviation note,
unresolved=[]) have already been made and ``freeze.status`` is already
``"sealed"`` with a deliberately-invalid placeholder ``payload_sha256`` --
it only computes and stamps the real digest, using the project's own
canonicalization (``exp012_xed.schema.preregistration_sha256``), not a
second parallel implementation.

Usage (run from the repo root, in the ``openmm_dev`` conda env -- no GPU
needed, this only reads/hashes JSON and small on-disk artifacts):

    python scripts/reseal_exp012_preregistration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp012_xed.schema import (  # noqa: E402
    Exp012IntegrityError,
    Exp012ProtocolError,
    preregistration_sha256,
    validate_preregistration,
)

PREREG_PATH = ROOT / "protocols" / "EXP-012_preregistration.json"


def main() -> int:
    payload = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    freeze = payload.get("freeze", {})
    if freeze.get("status") != "sealed":
        print(
            "freeze.status is not 'sealed' -- expected the content edits to have "
            "already set it to 'sealed' with a placeholder payload_sha256; "
            "refusing to reseal a document still in 'draft'.",
            file=sys.stderr,
        )
        return 1
    # Drop the human-readable placeholder note left by the content-edit step;
    # it is not part of the real, permanent sealed protocol document.
    freeze.pop("payload_sha256_note", None)
    freeze.pop("payload_sha256", None)
    digest = preregistration_sha256(payload)
    freeze["payload_sha256"] = digest
    payload["freeze"] = freeze

    PREREG_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    try:
        result = validate_preregistration(
            payload, workspace_root=str(ROOT), verify_files=True, require_sealed=True,
        )
    except (Exp012ProtocolError, Exp012IntegrityError) as exc:
        print(f"FAILED post-reseal validation: {exc}", file=sys.stderr)
        return 1

    print(f"OK: sealed, payload_sha256={result.payload_sha256}, executable={result.executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
