#!/usr/bin/env python
"""Generate an isolated EXP-012 complete-MM target ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.mm_ledger import relabel_mm_ledger


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", default="protocols/EXP-012_preregistration.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--platform", choices=("Reference", "CPU", "CUDA"), default="CPU")
    parser.add_argument("--device-index")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--skip-input-hash-verification", action="store_true")
    args = parser.parse_args(argv)
    report = relabel_mm_ledger(
        args.preregistration,
        args.run_id,
        args.output_dir,
        workspace_root=ROOT,
        platform_name=args.platform,
        device_index=args.device_index,
        frame_start=args.frame_start,
        frame_stop=args.frame_stop,
        frame_stride=args.frame_stride,
        verify_all_input_hashes=not args.skip_input_hash_verification,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
