#!/usr/bin/env python
"""Audit EXP-012 CUDA ledgers and emit structured JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.ledger_audit import audit_exp012_ledgers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda-root", default="output/outer_lambda_exp012/mm_ledger_cuda"
    )
    parser.add_argument("--cpu-reference-root")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = audit_exp012_ledgers(
        ROOT / args.cuda_root if not Path(args.cuda_root).is_absolute() else args.cuda_root,
        cpu_reference_root=(
            ROOT / args.cpu_reference_root
            if args.cpu_reference_root and not Path(args.cpu_reference_root).is_absolute()
            else args.cpu_reference_root
        ),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
