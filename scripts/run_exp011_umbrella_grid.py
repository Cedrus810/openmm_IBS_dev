#!/usr/bin/env python3
"""Fail-closed, resumable launcher for one EXP-011 formal umbrella replicate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from outer_lambda_neural_basis import sha256_file, stable_payload_sha256  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def _center_slug(center: float) -> str:
    sign = "p" if center >= 0 else "m"
    return f"{sign}{abs(center):.1f}".replace(".", "p")


def _validate_plan(plan: dict[str, Any], protocol: dict[str, Any]) -> None:
    stored = plan.get("sampling_plan_sha256")
    core = dict(plan)
    core.pop("sampling_plan_sha256", None)
    actual = stable_payload_sha256(core)
    if stored != actual:
        raise RuntimeError(f"sampling plan hash mismatch: stored={stored}, actual={actual}")
    if plan.get("status") != "FROZEN_AFTER_PILOT_BEFORE_FORMAL_RESULTS":
        raise RuntimeError("sampling plan is not frozen for formal execution")
    if plan.get("core_protocol_sha256") != protocol.get("protocol_sha256"):
        raise RuntimeError("sampling plan/core protocol hash mismatch")
    centers = plan.get("centers_degrees")
    expected = [-172.5 + 15.0 * index for index in range(24)]
    if not isinstance(centers, list) or [float(value) for value in centers] != expected:
        raise RuntimeError("centers are not the 24 frozen 15-degree bin midpoints")


def _validate_existing_report(
    report_path: Path,
    *,
    protocol_sha256: str,
    run_id: str,
    center: float,
    seed: int,
    sampling: dict[str, Any],
    initial_trajectory: Path,
) -> None:
    report = _read_json(report_path)
    umbrella = report.get("umbrella")
    if not isinstance(umbrella, dict):
        raise RuntimeError(f"invalid umbrella report: {report_path}")
    checks = {
        "ok": report.get("ok") is True,
        "protocol": report.get("protocol_sha256") == protocol_sha256,
        "run_id": umbrella.get("run_id") == run_id,
        "center": abs(float(umbrella.get("center_degrees")) - center) < 1.0e-9,
        "force_constant": abs(
            float(umbrella.get("force_constant_kj_mol_radian2"))
            - float(sampling["force_constant_kj_mol_radian2"])
        )
        < 1.0e-9,
        "seed": int(report.get("random_seed")) == seed,
        "burnin": int(report.get("burnin_steps")) == int(sampling["burnin_steps"]),
        "sampling": int(report.get("sampling_steps")) == int(sampling["sampling_steps"]),
        "interval": int(report.get("report_interval_steps"))
        == int(sampling["report_interval_steps"]),
        "sample_count": int(umbrella.get("sample_count"))
        == int(sampling["sampling_steps"]) // int(sampling["report_interval_steps"]),
        "initial_trajectory": Path(str(report.get("initial_trajectory", ""))).resolve()
        == initial_trajectory,
        "initial_hash": report.get("initial_trajectory_sha256")
        == sha256_file(initial_trajectory),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"existing report fails resume validation {failed}: {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="protocols/EXP-011_umbrella_sampling_plan.json")
    parser.add_argument("--protocol", default="protocols/EXP-011_preregistration.json")
    parser.add_argument("--manifest", default="output/outer_lambda_exp011/slow_variable_manifest.json")
    parser.add_argument("--baseline-root", default="output_lrc_fix")
    parser.add_argument("--output-root", default="output/outer_lambda_exp011/formal_umbrella_v1")
    parser.add_argument("--replicate", required=True)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=1,
        help="Run at most this many pending windows; use 24 for one full replicate.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan_path = _resolve(args.plan)
    protocol_path = _resolve(args.protocol)
    manifest_path = _resolve(args.manifest)
    baseline_root = _resolve(args.baseline_root)
    output_root = _resolve(args.output_root)
    plan = _read_json(plan_path)
    protocol = _read_json(protocol_path)
    _validate_plan(plan, protocol)
    if not manifest_path.is_file() or not baseline_root.is_dir():
        raise RuntimeError("manifest or baseline root is missing")
    matches = [item for item in plan["replicates"] if item.get("run_id") == args.replicate]
    if len(matches) != 1:
        raise RuntimeError(f"unknown or duplicate replicate: {args.replicate}")
    replicate = matches[0]
    initial_trajectory = _resolve(str(replicate["initial_trajectory"]))
    if not initial_trajectory.is_file():
        raise RuntimeError(f"initial trajectory is missing: {initial_trajectory}")
    if args.max_windows < 1 or args.max_windows > 24:
        raise RuntimeError("--max-windows must be between 1 and 24")

    sampling = plan["sampling"]
    protocol_sha = str(protocol["protocol_sha256"])
    pending: list[tuple[float, int, Path, Path]] = []
    completed = 0
    for index, raw_center in enumerate(plan["centers_degrees"]):
        center = float(raw_center)
        seed = int(replicate["seed_base"]) + index
        window_dir = output_root / args.replicate / f"center_{_center_slug(center)}"
        report_path = window_dir / "report.json"
        if report_path.is_file():
            _validate_existing_report(
                report_path,
                protocol_sha256=protocol_sha,
                run_id=args.replicate,
                center=center,
                seed=seed,
                sampling=sampling,
                initial_trajectory=initial_trajectory,
            )
            completed += 1
        elif window_dir.exists() and any(window_dir.iterdir()):
            raise RuntimeError(f"nonempty incomplete window; inspect manually: {window_dir}")
        else:
            pending.append((center, seed, window_dir, report_path))

    summary = {
        "replicate": args.replicate,
        "completed": completed,
        "pending": len(pending),
        "will_run": min(args.max_windows, len(pending)),
        "initial_trajectory": str(initial_trajectory),
        "initial_trajectory_sha256": sha256_file(initial_trajectory),
        "sampling_plan_sha256": plan["sampling_plan_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    for center, seed, window_dir, report_path in pending[: args.max_windows]:
        command = [
            sys.executable,
            str(REPO_ROOT / "outer_lambda_neural_basis.py"),
            "exp011-umbrella-sample",
            "--baseline-root",
            str(baseline_root),
            "--manifest",
            str(manifest_path),
            "--protocol",
            str(protocol_path),
            "--output-dir",
            str(window_dir),
            "--run-id",
            args.replicate,
            "--center-degrees",
            str(center),
            "--force-constant-kj-mol-radian2",
            str(sampling["force_constant_kj_mol_radian2"]),
            "--initial-trajectory",
            str(initial_trajectory),
            "--burnin-steps",
            str(sampling["burnin_steps"]),
            "--minimize-max-iterations",
            str(sampling["minimize_max_iterations"]),
            "--sampling-steps",
            str(sampling["sampling_steps"]),
            "--report-interval-steps",
            str(sampling["report_interval_steps"]),
            "--platform",
            str(sampling["platform"]),
            "--seed",
            str(seed),
            "--output",
            str(report_path),
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        _validate_existing_report(
            report_path,
            protocol_sha256=protocol_sha,
            run_id=args.replicate,
            center=center,
            seed=seed,
            sampling=sampling,
            initial_trajectory=initial_trajectory,
        )
        print(f"completed {args.replicate} center={center:+.1f} report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
