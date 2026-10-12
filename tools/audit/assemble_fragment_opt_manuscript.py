#!/usr/bin/env python3
"""Freeze an existing fragment-opt CSV snapshot and audit its manuscript rows.

Reads existing result/provenance JSON only; never runs sampling or changes labels.
The output directory must be new so an earlier manuscript snapshot is preserved.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assemble(root, snapshot, out):
    sources = {
        name: root / "analysis/results/history" / f"abfe_{name}_{snapshot}.csv"
        for name in ("reps", "summary", "stats")
    }
    saved = {name: path.read_bytes() for name, path in sources.items()}
    reps = list(csv.DictReader(saved["reps"].decode().splitlines()))
    summary = list(csv.DictReader(saved["summary"].decode().splitlines()))
    ibs = [r for r in reps if r["protocol"] == "runs_lcr_s1long15"]
    if not ibs:
        raise ValueError("No s1long15 results in the selected snapshot")
    audit, source_manifest = [], []
    for row in ibs:
        result_path = root / row["source"]
        result = json.loads(result_path.read_text())
        if abs(result["delta_G_bind_kcal_mol"] - float(row["dG_kcal"])) > 0.000051:
            raise ValueError(f"Raw result differs from frozen CSV: {result_path}")
        record = dict(row)
        for key in ("results_untrusted", "publishable_as_accepted_result", "precision_status"):
            record[key] = result.get(key)
        record["untrusted_legs"] = json.dumps(
            result.get("result_status", {}).get("results_untrusted_legs", [])
        )
        for leg in ("complex", "solvent"):
            base = result_path.parent if leg == "complex" else result_path.parent / "solvent_leg"
            checkpoint = base / "checkpoints/stage2_vanishing.json"
            data = json.loads(checkpoint.read_text())
            rounds = data.get("stage2_rounds", {})
            record[f"{leg}_rounds"] = rounds.get("n_rounds")
            record[f"{leg}_analysis_status"] = data.get("analysis_status")
            record[f"{leg}_window_labels"] = json.dumps(rounds.get("labels", {}), sort_keys=True)
        audit.append(record)
        files = [result_path, result_path.parent / "run_provenance.json",
                 result_path.parent / "launch_provenance.txt"]
        files += [result_path.parent / prefix / "checkpoints" / name
                  for prefix in (".", "solvent_leg")
                  for name in ("stage2_vanishing.json", "stage2_rounds_ledger.json")]
        for path in files:
            source_manifest.append({"path": str(path), "sha256": sha256(path)})
    for row in summary:
        if row["protocol"] != "runs_lcr_s1long15":
            continue
        values = [float(r["dG_kcal"]) for r in ibs if r["system"] == row["system"]]
        if len(values) != int(row["n"]) or abs(statistics.mean(values) - float(row["mean_kcal"])) > 0.0001:
            raise ValueError(f"Inconsistent saved summary: {row['system']}")
        if len(values) > 1 and abs(statistics.stdev(values) - float(row["sd_kcal"])) > 0.0001:
            raise ValueError(f"Inconsistent saved SD: {row['system']}")
    eligible = {(r["method"], r["system"]): r for r in summary if r["benchmark_eligible"] == "1"}
    common = sorted({s for m, s in eligible if m == "ibs" and int(eligible[m, s]["n"]) >= 3}
                    & {s for m, s in eligible if m == "openfe" and int(eligible[m, s]["n"]) >= 3})
    if not common:
        raise ValueError("No common systems with at least three repeats per method")
    metrics = []
    for method in ("ibs", "openfe", "paper_gmx"):
        selected = [r for r in summary if r["method"] == method and r["system"] in common
                    and (method == "paper_gmx" or r["benchmark_eligible"] == "1")]
        for ref in ("paper_mean_kcal", "exp_kcal"):
            if method == "paper_gmx" and ref == "paper_mean_kcal":
                continue
            differences = [float(r["mean_kcal"]) - float(r[ref]) for r in selected]
            metrics.append({"method": method, "reference": ref, "n_systems": len(common),
                            "min_reps": 3, "mean_signed_error_kcal": statistics.mean(differences),
                            "mue_kcal": statistics.mean(map(abs, differences)),
                            "rmse_kcal": math.sqrt(statistics.mean(v * v for v in differences)),
                            "systems": ";".join(common)})
    out.mkdir(parents=True, exist_ok=False)
    for name, content in saved.items():
        (out / f"abfe_{name}.csv").write_bytes(content)
    write_csv(out / "aibs_run_status.csv", audit)
    write_csv(out / "matched_system_metrics.csv", metrics)
    manifest = {"assembled_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_root": str(root), "source_snapshot": snapshot,
                "aibs_protocol": "runs_lcr_s1long15", "aibs_runs": len(audit),
                "common_systems_min_three_repeats": common,
                "metric_inputs": "frozen CSV system means rounded to four decimals",
                "status_policy": "descriptive export; original labels are never revised",
                "source_csvs": [{"path": str(p), "sha256": hashlib.sha256(saved[n]).hexdigest()}
                                for n, p in sources.items()],
                "raw_aibs_sources": source_manifest,
                "generated_files": [{"path": p.name, "sha256": sha256(p)} for p in sorted(out.iterdir())],
                "assembly_script_sha256": sha256(Path(__file__)),
                "limits": ["CSV inclusion is not scientific acceptance",
                           "Reference target equivalence is not established",
                           "Raw source hashes locate evidence; full raw artifacts are not archived here",
                           "No simulation, trajectory reanalysis, or uncertainty calibration performed"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(audit)} AIBS statuses and {len(metrics)} matched-set metrics to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.source_root.resolve(), args.snapshot, args.out.resolve())
