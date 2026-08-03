#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.metrics import assess_whole_run_data_support


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger-root",
        default="output/outer_lambda_exp012/mm_ledger_cuda",
    )
    parser.add_argument(
        "--minimum-ess",
        type=float,
        required=True,
        help="Draft gate supplied explicitly; freeze it in the v2 protocol before A/B/C/D results.",
    )
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = assess_whole_run_data_support(
        ROOT / args.ledger_root,
        minimum_raw_importance_ess_per_target=args.minimum_ess,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
