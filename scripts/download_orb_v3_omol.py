#!/usr/bin/env python
"""Download and verify the frozen ORB-v3 Conservative OMol checkpoint.

This script only uses ``cached_path`` and verifies the file identity.  It does
not instantiate the model, construct an OpenMM System, or modify production
artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


MODEL_NAME = "orb-v3-conservative-omol"
EXPECTED_URL = (
    "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/"
    "forcefields/orb-v3-conservative-omol-20250820.ckpt"
)
EXPECTED_SIZE_BYTES = 103_417_970
EXPECTED_SHA256 = "c284e99c45df928ae28443fb27223188cc2c33cced593488a4d28595e75cb6e8"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="output/outer_lambda_orb/orb-v3-conservative-omol-20250820.ckpt",
        help="local checkpoint path",
    )
    args = parser.parse_args(argv)

    print(f"model={MODEL_NAME}", flush=True)
    print(f"url={EXPECTED_URL}", flush=True)
    path = Path(args.output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        print("existing_file_found=true", flush=True)
    else:
        partial = path.with_name(path.name + ".part")
        print(f"downloading_to={path}", flush=True)
        request = urllib.request.Request(
            EXPECTED_URL,
            headers={"User-Agent": "ORB-001-checkpoint-downloader/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        partial.replace(path)
    observed_size = path.stat().st_size
    observed_sha256 = _sha256_file(path)
    print(json.dumps({
        "path": str(path),
        "size_bytes": observed_size,
        "sha256": observed_sha256,
        "expected_size_bytes": EXPECTED_SIZE_BYTES,
        "expected_sha256": EXPECTED_SHA256,
    }, sort_keys=True, indent=2), flush=True)

    if observed_size != EXPECTED_SIZE_BYTES or observed_sha256 != EXPECTED_SHA256:
        raise RuntimeError(
            "downloaded ORB checkpoint identity mismatch; refusing to use it"
        )
    print("ORB_CHECKPOINT_VERIFIED=PASS", flush=True)
    print(f"ORB_MODEL_PATH={path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
