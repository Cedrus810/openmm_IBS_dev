#!/usr/bin/env python3
"""Inspect an explicitly configured EXP-012 MACE C0 contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_residual.mace_contract import (  # noqa: E402
    MaceContractError,
    MaceModelContract,
    inspect_mace_model_contract,
)


def _atomic_numbers(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated integer atomic numbers") from exc


def _irreps_list(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(","))
    if not result or any(not item for item in result):
        raise argparse.ArgumentTypeError("use comma-separated non-empty product irreps")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-class", required=True)
    parser.add_argument("--expected-torch-version", required=True)
    parser.add_argument("--expected-mace-version", required=True)
    parser.add_argument("--expected-e3nn-version", required=True)
    parser.add_argument("--expected-interaction-layers", required=True, type=int)
    parser.add_argument("--expected-product-layers", required=True, type=int)
    parser.add_argument("--expected-r-max-angstrom", required=True, type=float)
    parser.add_argument("--expected-atomic-numbers", required=True, type=_atomic_numbers)
    parser.add_argument("--expected-product-layer-index", required=True, type=int)
    parser.add_argument("--expected-product-layer-irreps", required=True, type=_irreps_list)
    parser.add_argument("--expected-node-feats-dimension", required=True, type=int)
    parser.add_argument("--expected-invariant-slice-start", required=True, type=int)
    parser.add_argument("--expected-invariant-slice-stop", required=True, type=int)
    parser.add_argument("--expected-invariant-slice-irreps", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        contract = MaceModelContract(
            model_path=args.model_path,
            expected_sha256=args.expected_sha256,
            expected_class=args.expected_class,
            expected_torch_version=args.expected_torch_version,
            expected_mace_version=args.expected_mace_version,
            expected_e3nn_version=args.expected_e3nn_version,
            expected_interaction_layer_count=args.expected_interaction_layers,
            expected_product_layer_count=args.expected_product_layers,
            expected_r_max_angstrom=args.expected_r_max_angstrom,
            expected_atomic_numbers=args.expected_atomic_numbers,
            expected_product_layer_index=args.expected_product_layer_index,
            expected_product_layer_irreps=args.expected_product_layer_irreps,
            expected_node_feats_dimension=args.expected_node_feats_dimension,
            expected_invariant_slice_start=args.expected_invariant_slice_start,
            expected_invariant_slice_stop=args.expected_invariant_slice_stop,
            expected_invariant_slice_irreps=args.expected_invariant_slice_irreps,
        )
        report = inspect_mace_model_contract(contract)
    except MaceContractError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
