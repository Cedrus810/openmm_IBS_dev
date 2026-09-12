#!/usr/bin/env python
"""Build a canonical EXP-012 environment manifest from an explicit JSON config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.environment import (  # noqa: E402
    EnvironmentManifestError,
    build_environment_manifest,
    write_environment_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate explicit atom identities and bind them to hashed sources."
    )
    parser.add_argument("--config", required=True, help="Explicit JSON config path")
    parser.add_argument("--output", required=True, help="Manifest JSON output path")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    output_path = Path(args.output)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        document = build_environment_manifest(config, workspace_root=ROOT)
        write_environment_manifest(output_path, document)
    except (OSError, json.JSONDecodeError, EnvironmentManifestError) as error:
        parser.error(str(error))
    print(document["canonical_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
