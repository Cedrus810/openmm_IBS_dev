#!/usr/bin/env python3
"""Measure DCD subset/full reader peak RSS in a fresh process per frame count.

Example:
  python tools/diagnostics/measure_mdtraj_dcd_subset_memory.py --dcd run.dcd \
      --atom-indices 0 1 2 3 --frame-counts 25 50 100 200 400

The full mode can use substantial memory. Peak RSS includes Python, MDTraj,
and native reader allocations. Baseline-subtracted peaks are estimates, not
allocator-level measurements. DCD reading needs no topology.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def peak_rss_bytes():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def child_measure(spec):
    import mdtraj as md

    started = time.perf_counter()
    result = {
        "mode": spec["mode"], "requested_frames": spec["frames"],
        "mdtraj_version": md.__version__, "pid": os.getpid(),
    }
    if spec["mode"] != "baseline":
        indices = spec["indices"] if spec["mode"] == "subset" else None
        with md.open(spec["dcd"], mode="r") as reader:
            xyz, cell_lengths, cell_angles = reader.read(
                n_frames=spec["frames"], atom_indices=indices
            )
        result.update(
            actual_frames=int(xyz.shape[0]),
            xyz_shape=list(xyz.shape),
            xyz_dtype=str(xyz.dtype),
            coordinate_bytes=int(xyz.nbytes),
            cell_bytes=sum(int(x.nbytes) for x in (cell_lengths, cell_angles)
                           if x is not None),
        )
    result["elapsed_seconds"] = time.perf_counter() - started
    result["peak_rss_bytes"] = peak_rss_bytes()
    return result


def run_child(spec):
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(spec)],
        capture_output=True, text=True,
    )
    if proc.returncode:
        raise RuntimeError(
            f"{spec['mode']} / {spec['frames']} frames failed "
            f"(exit {proc.returncode}):\n{proc.stderr.strip()}"
        )
    # Imports/native libraries may log to stdout before the structured result.
    for line in reversed(proc.stdout.splitlines()):
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and "peak_rss_bytes" in result:
            return result
    raise RuntimeError(f"Child returned no measurement: {proc.stdout}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dcd", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--atom-indices", nargs="+", type=int)
    group.add_argument("--atom-indices-file", type=Path,
                       help="JSON array of zero-based atom indices")
    parser.add_argument("--frame-counts", nargs="+", type=int,
                        default=[25, 50, 100, 200, 400])
    parser.add_argument("--modes", nargs="+", choices=["subset", "full"],
                        default=["subset", "full"])
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(child_measure(json.loads(args.child))))
        return
    if args.dcd is None or not args.dcd.is_file():
        parser.error("--dcd must name an existing DCD file")
    indices = args.atom_indices
    if args.atom_indices_file:
        indices = json.loads(args.atom_indices_file.read_text(encoding="utf-8"))
    if "subset" in args.modes and (
        not isinstance(indices, list) or not indices
        or any(type(i) is not int or i < 0 for i in indices)
        or len(set(indices)) != len(indices)
    ):
        parser.error("subset mode requires unique, nonnegative atom indices")
    if any(n <= 0 for n in args.frame_counts):
        parser.error("--frame-counts must be positive")
    base_spec = dict(dcd=str(args.dcd.resolve()), indices=indices)
    baseline = run_child(dict(base_spec, mode="baseline", frames=0))
    rows = []
    print("mode    requested actual    xyz MiB   peak MiB  excess MiB  seconds",
          flush=True)
    for n in sorted(set(args.frame_counts)):
        for mode in dict.fromkeys(args.modes):
            row = run_child(dict(base_spec, mode=mode, frames=n))
            row["peak_above_baseline_bytes"] = max(
                0, row["peak_rss_bytes"] - baseline["peak_rss_bytes"]
            )
            rows.append(row)
            print(f"{mode:7} {n:9d} {row['actual_frames']:6d} "
                  f"{row['coordinate_bytes'] / 2**20:10.2f} "
                  f"{row['peak_rss_bytes'] / 2**20:10.2f} "
                  f"{row['peak_above_baseline_bytes'] / 2**20:11.2f} "
                  f"{row['elapsed_seconds']:8.2f}", flush=True)
            if row["actual_frames"] < n:
                print("  DCD ended early; use actual frame count for scaling.",
                      flush=True)
    report = dict(dcd=base_spec["dcd"], baseline=baseline, measurements=rows)
    if args.json_output:
        args.json_output.write_text(json.dumps(report, indent=2) + "\n",
                                    encoding="utf-8")


if __name__ == "__main__":
    main()
