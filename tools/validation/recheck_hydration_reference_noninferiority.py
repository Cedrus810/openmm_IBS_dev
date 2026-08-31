#!/usr/bin/env python3
"""Re-evaluate saved C2 hydration gates with the v3 C1 non-inferiority rule.

This is an evaluator-only post-processing step.  It reads the bootstrap CI
already stored in each slab_quality_gate.json; it does not read trajectories,
change a Hamiltonian, or rerun GPU/MD.  Original equality-gate evidence is
left untouched and all v3 outputs use separate filenames.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_CASES = (
    "Na_thin_pos0",
    "Na_thin_pos1",
    "Na_thick_pos0",
    "Na_thick_pos1",
)
DEFAULT_MARGIN_WATER = 0.5
DEFAULT_GATE_NAME = "slab_quality_gate_hydration_v3.json"
DEFAULT_REPORT_NAME = "report_hydration_v3.json"
DEFAULT_SUMMARY_NAME = "summary_hydration_v3.json"
DEFAULT_AUDIT_NAME = "c2_lipid_slab_v11_hydration_noninferiority_v3_summary.json"


def _read(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _write(path: Path, value: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, ensure_ascii=False)


def _default_directories(repo_root: Path) -> List[Path]:
    directories: List[Path] = []
    for case in DEFAULT_CASES:
        directories.append(repo_root / "validation/c2_lipid_slab_v11_full11" / case)
        directories.append(repo_root / "validation/c2_lipid_slab_v11_seeds" / f"{case}_seed2027")
        directories.append(repo_root / "validation/c2_lipid_slab_v11_seeds" / f"{case}_seed2028")
    return directories


def _state_passed(state: Dict[str, Any], margin_water: float) -> tuple[bool, bool]:
    """Return (new_state_passed, new_reference_comparison_passed)."""
    ci = state.get("sample_minus_reference_bootstrap_ci95", ())
    lower = float(ci[0]) if len(ci) >= 1 else float("nan")
    reference_ok = math.isfinite(lower) and lower >= -margin_water
    eligible = bool(state.get("hard_gate_eligible", False))
    if not eligible:
        return True, reference_ok

    lambda0_ok = True
    lam = float(state.get("lambda_coul", float("nan")))
    min_frames = state.get("lambda0_min_frames")
    if abs(lam) <= 1.0e-12 and min_frames is not None:
        lambda0_ok = int(state.get("n_frames", 0)) >= int(min_frames)

    passed = all(
        (
            lambda0_ok,
            bool(state.get("mean_gate_passed", False)),
            bool(state.get("bootstrap_lower_bound_ge_5_passed", False)),
            reference_ok,
            bool(state.get("severe_dehydration_gate_passed", False)),
            bool(state.get("supplement_stability_gate_passed", True)),
        )
    )
    return passed, reference_ok


def recheck_gate(path: Path, margin_water: float, gate_name: str) -> Dict[str, Any]:
    original = _read(path / "slab_quality_gate.json")
    gate = copy.deepcopy(original)
    old_reasons = list(original.get("failure_reasons", []))
    states = gate.get("coion_coordination_by_lambda", {})
    hydration_ok = True

    for state in states.values():
        passed, reference_ok = _state_passed(state, margin_water)
        state["reference_comparison_passed"] = reference_ok
        state["reference_comparison_rule"] = "non_inferiority_lower_ci_ge_minus_margin"
        state["reference_noninferiority_margin_water"] = margin_water
        state["passed"] = passed
        if state.get("hard_gate_eligible", False) and not passed:
            hydration_ok = False

    checks = dict(gate.get("checks", {}))
    checks["coion_water_coordination_sufficient"] = hydration_ok
    checks["coion_hydration_gate_at_charge_fraction_ge_0p9"] = hydration_ok
    passed = all(bool(value) for value in checks.values())

    non_hydration_reasons = [
        reason for reason in old_reasons if "hydration gate" not in str(reason)
    ]
    reasons = non_hydration_reasons
    if not passed and not reasons:
        reasons = [f"quality gate check failed: {key}" for key, value in checks.items() if not value]

    gate["checks"] = checks
    gate["failure_reasons"] = reasons
    gate["passed"] = passed
    gate["hydration_gate_statistical_version"] = 3
    gate["hydration_reference_comparison_rule"] = "non_inferiority_lower_ci_ge_minus_margin"
    gate["hydration_reference_noninferiority_margin_water"] = margin_water
    gate["original_equality_gate_file"] = "slab_quality_gate.json"
    gate["original_equality_gate_passed"] = bool(original.get("passed", False))
    gate["original_equality_gate_failure_reasons"] = old_reasons
    _write(path / gate_name, gate)
    return gate


def recheck_report(
    path: Path,
    gate: Dict[str, Any],
    margin_water: float,
    report_name: str,
    summary_name: str,
) -> Dict[str, Any]:
    original_report = _read(path / "report.json")
    report = copy.deepcopy(original_report)
    checks = dict(report.get("checks", {}))
    checks["slab_quality_gate_passed"] = bool(gate.get("passed", False))
    missing = list(report.get("missing_artifacts", []))
    passed = not missing and all(bool(value) for value in checks.values())
    report["slab_quality_gate"] = gate
    report["checks"] = checks
    report["passed"] = passed
    report["failure_reasons"] = missing + [key for key, value in checks.items() if not value]
    report["re_evaluation"] = {
        "rule": "non_inferiority_lower_ci_ge_minus_margin",
        "margin_water": margin_water,
        "source_gate": "slab_quality_gate.json",
        "source_report": "report.json",
    }
    _write(path / report_name, report)

    summary = _read(path / "summary.json")
    summary["slab_quality_gate_passed"] = bool(gate.get("passed", False))
    summary["passed"] = passed
    summary["status"] = "complete" if not missing else "incomplete"
    summary["failure_reasons"] = report["failure_reasons"]
    summary["re_evaluation"] = report["re_evaluation"]
    _write(path / summary_name, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--directory", type=Path, action="append", help="可重复；默认重评 C2 全部 12 个目录")
    parser.add_argument("--margin-water", type=float, default=DEFAULT_MARGIN_WATER)
    parser.add_argument("--gate-name", default=DEFAULT_GATE_NAME)
    parser.add_argument("--report-name", default=DEFAULT_REPORT_NAME)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--audit-name", default=DEFAULT_AUDIT_NAME)
    args = parser.parse_args()
    if args.margin_water < 0.0:
        parser.error("--margin-water 必须非负")

    directories = args.directory or _default_directories(args.repo_root)
    audit: List[Dict[str, Any]] = []
    for raw_path in directories:
        path = raw_path if raw_path.is_absolute() else args.repo_root / raw_path
        gate = recheck_gate(path, args.margin_water, args.gate_name)
        summary = recheck_report(path, gate, args.margin_water, args.report_name, args.summary_name)
        audit.append(
            {
                "directory": str(path.relative_to(args.repo_root)),
                "passed": bool(summary["passed"]),
                "original_equality_gate_passed": bool(gate["original_equality_gate_passed"]),
                "failure_reasons": summary["failure_reasons"],
            }
        )
        print(f"{'PASS' if summary['passed'] else 'FAIL'} {path}")

    audit_path = args.repo_root / "validation" / args.audit_name
    _write(
        audit_path,
        {
            "rule": "non_inferiority_lower_ci_ge_minus_margin",
            "margin_water": args.margin_water,
            "n_results": len(audit),
            "n_passed": sum(item["passed"] for item in audit),
            "results": audit,
            "original_equality_gate_evidence_retained": True,
            "output_gate_name": args.gate_name,
            "output_report_name": args.report_name,
            "output_summary_name": args.summary_name,
        },
    )
    print(f"AUDIT {audit_path}")
    return 0 if all(item["passed"] for item in audit) else 1


if __name__ == "__main__":
    raise SystemExit(main())
