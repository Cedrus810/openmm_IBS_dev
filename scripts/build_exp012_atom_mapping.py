#!/usr/bin/env python
"""Build a canonical EXP-012 atom mapping from a sealed environment manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.atom_mapping import (  # noqa: E402
    AtomMappingError,
    AtomMappingIntegrityError,
    build_atom_mapping,
)
from local_residual.environment import (  # noqa: E402
    EnvironmentManifestError,
    load_environment_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a topology/local-graph/MACE-node ordering config against an "
            "already-sealed environment manifest and emit the canonical atom mapping."
        )
    )
    parser.add_argument("--environment-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--rebind-source-environment-manifest-sha",
        action="store_true",
        help=(
            "replace the config template's source manifest SHA with the loaded manifest's "
            "canonical SHA before validation"
        ),
    )
    parser.add_argument(
        "--resolved-config-output",
        help="optional path for the SHA-bound config actually used",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    try:
        environment = load_environment_manifest(
            args.environment_manifest, workspace_root=ROOT, verify_sources=True
        )
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if args.rebind_source_environment_manifest_sha:
            config["source_environment_manifest_sha256"] = environment["canonical_sha256"]
        document = build_atom_mapping(environment, config)
    except (
        OSError,
        json.JSONDecodeError,
        EnvironmentManifestError,
        AtomMappingError,
        AtomMappingIntegrityError,
    ) as error:
        parser.error(str(error))
        return 2  # pragma: no cover - parser.error already raises SystemExit

    if args.resolved_config_output:
        resolved_config_path = Path(args.resolved_config_output)
        resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_config_path.write_text(
            json.dumps(config, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(document["canonical_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
