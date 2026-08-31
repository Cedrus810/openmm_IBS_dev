#!/usr/bin/env python
"""对 vanishing stage 做 split-half 收敛判据（纯离线，不跑 MD）。

**为什么需要它**：溶剂盒扫描（P0-11）里 6.057 nm 那轮的 vanishing 比
3.000 / 4.257 nm 低 7.33 kJ/mol，而它报出的 σ 只有 1.43。所有生产门都过了
（最紧的 `max_endpoint_uncertainty` 0.9843 / 阈值 1.0），所以门本身证明不了
这 7.33 是物理还是欠采样。

**判据**：把每个窗口的帧按时间前一半 / 后一半切开，各自用**同一个生产求解器**
（`ibs_engine.solve_stage_integrated`）解一遍。

- 两半之差 ≲ 报出的 σ  → 采样够，7.33 需要物理解释
- 两半之差 ≫ 报出的 σ  → 欠采样，σ 被低估，7.33 不可采信

这不是新估计器：加载走生产 loader `ABFEPipeline._load_ibs_window_outputs_from_dir`
（三文件 manifest / f_k / expected-vs-loaded 门全部照旧），求解走
`solve_stage_integrated` + `diagnose_endpoint_sigma.FINAL_GATES`，
唯一的改动是喂进去的帧被切了一半。脚本会先用全量数据复算一遍，
和 `checkpoints/stage2_vanishing.json` 的 `total_delta_G` 对不上就直接报错退出——
对不上说明加载口径就错了，后面的结论一概不可信。

用法::

    python tools/diagnostics/diagnose_split_half_convergence.py \\
        --run-dir output/solvent_leg \\
        --run-dir solvent_box_scan/pad_1.5000/solvent_leg \\
        --run-dir solvent_box_scan/pad_2.4000/solvent_leg
"""

from __future__ import annotations

# Allow direct execution from tools/* while keeping live modules at repo root.
import sys as _abfe_sys
from pathlib import Path as _AbfePath

_ABFE_REPO_ROOT = _AbfePath(__file__).resolve().parents[2]
if str(_ABFE_REPO_ROOT) not in _abfe_sys.path:
    _abfe_sys.path.insert(0, str(_ABFE_REPO_ROOT))


import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

from diagnose_endpoint_sigma import FINAL_GATES, load_lambda_path, load_outputs

_R_KJ_PER_MOL_K = 0.008314462618


def _solve(windows: List[Dict], kt: float) -> Dict:
    from ibs_engine import solve_stage_integrated

    return solve_stage_integrated(
        window_outputs=windows, kt=kt, stage_name="vanishing", **FINAL_GATES
    )


def _slice_decharging(u_kn: np.ndarray, n_k: np.ndarray, lo_frac: float, hi_frac: float):
    """按时间顺序取每个态样本块的 [lo_frac, hi_frac) 段。

    stage1 是一整块全局 REMD MBAR（`decharging_pme_u_kn.npy`，形状 (K, ΣN_k)），
    不是 stage2 那种按窗口分块的结构，所以要按 n_k 的偏移逐态切。
    """
    offsets = np.concatenate(([0], np.cumsum(n_k))).astype(int)
    cols, new_n_k = [], []
    for k in range(len(n_k)):
        start, end = offsets[k], offsets[k + 1]
        n = end - start
        a = start + int(lo_frac * n)
        b = start + int(hi_frac * n)
        if b - a < 2:
            return None, None
        cols.append(np.arange(a, b))
        new_n_k.append(b - a)
    idx = np.concatenate(cols)
    return u_kn[:, idx], np.asarray(new_n_k, dtype=int)


def analyse_attachment(run_dir: str, temperature_k: float) -> Dict:
    """Boresch attachment 腿 (stage0) 的 split-half。

    结构与 stage1 同类（一整块全局 MBAR，按 n_k 分块），只是 λ 是限制强度。
    重点看**小 λ 那几档**：那里配体要探索大得多的构型空间，最容易没采够。
    """
    from ibs_engine import TraditionalMBARAnalyzer

    att_dir = os.path.join(run_dir, "attachment")
    u_kn = np.load(os.path.join(att_dir, "attachment_u_kn.npy"), allow_pickle=False)
    n_k = np.load(
        os.path.join(att_dir, "attachment_u_kn.npy.n_k.npy"), allow_pickle=False
    ).astype(int)
    with open(os.path.join(att_dir, "attachment_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    expected_dg = float(meta["attachment_delta_G_kJ_mol"])
    expected_err = float(meta["attachment_error_kJ_mol"])

    def solve(u, nk, decorrelate):
        analyzer = TraditionalMBARAnalyzer(temperature=temperature_k)
        analyzer._last_n_k = np.asarray(nk, dtype=int)
        return analyzer.solve(np.asarray(u, dtype=np.float64), decorrelate=decorrelate)

    mode, full = None, None
    for decorrelate in (True, False):
        cand = solve(u_kn, n_k, decorrelate)
        if abs(float(cand["delta_G"]) - expected_dg) < 1.0e-6:
            mode, full = decorrelate, cand
            break
    if full is None:
        tried = {d: float(solve(u_kn, n_k, d)["delta_G"]) for d in (True, False)}
        raise RuntimeError(
            f"{run_dir}: attachment 全量复现对不上落盘值 {expected_dg:.8f}"
            f"（decorrelate=True → {tried[True]:.8f}, False → {tried[False]:.8f}）"
        )

    halves = {}
    for label, frac in (("first", (0.0, 0.5)), ("second", (0.5, 1.0))):
        u_half, nk_half = _slice_decharging(u_kn, n_k, *frac)
        halves[label] = float(solve(u_half, nk_half, mode)["delta_G"]) if u_half is not None else None

    return {
        "run_dir": run_dir,
        "stage": "attachment",
        "decorrelate": bool(mode),
        "lambdas": meta.get("lambdas"),
        "mean_u_boresch_per_state_kJ_mol": meta.get("mean_u_boresch_per_state_kJ_mol"),
        "n_k": [int(x) for x in n_k],
        "total": {
            "full": float(full["delta_G"]),
            "full_error": float(full["error"]),
            "checkpoint": expected_dg,
            "checkpoint_error": expected_err,
            "first_half": halves["first"],
            "second_half": halves["second"],
        },
        "per_window": {},
        "n_frames_per_window": {},
    }


def analyse_decharging(run_dir: str, temperature_k: float) -> Dict:
    """stage1 (PME decharging) 的 split-half。"""
    from ibs_engine import TraditionalMBARAnalyzer

    dech_dir = os.path.join(run_dir, "decharging")
    u_kn_path = os.path.join(dech_dir, "decharging_pme_u_kn.npy")
    n_k_path = u_kn_path + ".n_k.npy"
    u_kn = np.load(u_kn_path, allow_pickle=False)
    n_k = np.load(n_k_path, allow_pickle=False).astype(int)

    with open(os.path.join(run_dir, "checkpoints", "stage1_decharging.json"), encoding="utf-8") as fh:
        ckpt = json.load(fh)
    expected_dg = float(ckpt["total_delta_G"])
    expected_err = float(ckpt["total_error"])

    def solve(u, nk, decorrelate):
        analyzer = TraditionalMBARAnalyzer(temperature=temperature_k)
        analyzer._last_n_k = np.asarray(nk, dtype=int)
        return analyzer.solve(np.asarray(u, dtype=np.float64), decorrelate=decorrelate)

    # 控制实验：不知道生产用的是去相关还是全帧，两个都试，取能复现落盘值的那个。
    # 两个都对不上就停——口径没对上，后面的结论一概不可信。
    mode = None
    full = None
    for decorrelate in (True, False):
        cand = solve(u_kn, n_k, decorrelate)
        if abs(float(cand["delta_G"]) - expected_dg) < 1.0e-6:
            mode, full = decorrelate, cand
            break
    if full is None:
        tried = {d: float(solve(u_kn, n_k, d)["delta_G"]) for d in (True, False)}
        raise RuntimeError(
            f"{run_dir}: stage1 全量复现对不上落盘值 {expected_dg:.8f}"
            f"（decorrelate=True → {tried[True]:.8f}, False → {tried[False]:.8f}），拒绝继续"
        )

    halves = {}
    for label, frac in (("first", (0.0, 0.5)), ("second", (0.5, 1.0))):
        u_half, nk_half = _slice_decharging(u_kn, n_k, *frac)
        halves[label] = float(solve(u_half, nk_half, mode)["delta_G"]) if u_half is not None else None

    return {
        "run_dir": run_dir,
        "stage": "decharging",
        "decorrelate": bool(mode),
        "n_k": [int(x) for x in n_k],
        "total": {
            "full": float(full["delta_G"]),
            "full_error": float(full["error"]),
            "checkpoint": expected_dg,
            "checkpoint_error": expected_err,
            "first_half": halves["first"],
            "second_half": halves["second"],
        },
        "per_window": {},
        "n_frames_per_window": {},
    }


def analyse(run_dir: str, temperature_k: float) -> Dict:
    kt = _R_KJ_PER_MOL_K * temperature_k
    lambdas_vdw, ranges = load_lambda_path(run_dir)
    windows = load_outputs(
        os.path.join(run_dir, "vanishing"),
        ranges,
        lambdas_vdw,
        os.path.join(run_dir, "checkpoints"),
    )

    # 控制实验：全量必须复现落盘值，否则加载口径就错了。
    ckpt_path = os.path.join(run_dir, "checkpoints", "stage2_vanishing.json")
    with open(ckpt_path, encoding="utf-8") as handle:
        ckpt = json.load(handle)
    expected_dg = float(ckpt["total_delta_G"])
    expected_err = float(ckpt["total_error"])

    full = _solve(windows, kt)
    dg_full = float(full["total_delta_G"])
    err_full = float(full["total_error"])
    if abs(dg_full - expected_dg) > 1.0e-6:
        raise RuntimeError(
            f"{run_dir}: 全量复现对不上落盘值 "
            f"({dg_full:.8f} vs {expected_dg:.8f})，加载口径有问题，拒绝继续"
        )

    n_frames = {int(w["window_index"]): int(np.asarray(w["u_kn"]).shape[1]) for w in windows}

    # 切帧与半程求解现在由 `ibs_engine.split_half_drift_diagnostics` 负责，
    # 由 `solve_stage_integrated` 自动挂进返回值——这里不再各留一份实现。
    drift = full.get("split_half_diagnostics") or {}
    if not drift.get("available"):
        raise RuntimeError(
            f"{run_dir}: split-half 诊断不可用（{drift.get('reason', '未知')}）"
        )
    per_window = {
        str(w["window_index"]): {
            "full": w["delta_G_full_kJ_mol"],
            "first_half": w["delta_G_first_half_kJ_mol"],
            "second_half": w["delta_G_second_half_kJ_mol"],
            "sigma": w["uncertainty_kJ_mol"],
            "drift_over_2sigma": w["drift_over_2sigma"],
        }
        for w in drift["per_window"]
    }

    return {
        "run_dir": run_dir,
        "n_frames_per_window": n_frames,
        "total": {
            "full": dg_full,
            "full_error": err_full,
            "checkpoint": expected_dg,
            "checkpoint_error": expected_err,
            "first_half": drift["total_delta_G_first_half_kJ_mol"],
            "second_half": drift["total_delta_G_second_half_kJ_mol"],
        },
        "max_window_drift_over_2sigma": drift["max_window_drift_over_2sigma"],
        "per_window": per_window,
    }


def _print_report(rep: Dict) -> None:
    t = rep["total"]
    drift = (
        t["second_half"] - t["first_half"]
        if t["first_half"] is not None and t["second_half"] is not None
        else float("nan")
    )
    print(f"  全量  ΔG = {t['full']:.4f}  (落盘 {t['checkpoint']:.4f}) ± {t['full_error']:.4f}")
    print(f"  前半段 ΔG = {t['first_half']:.4f}" if t["first_half"] is not None else "  前半段：帧数不足")
    print(f"  后半段 ΔG = {t['second_half']:.4f}" if t["second_half"] is not None else "  后半段：帧数不足")
    # 两个半程各自 SE ≈ √2·σ，它们之差的 SE ≈ 2σ —— 所以判据是 |漂移|/(2σ)，不是 |漂移|/σ。
    z = abs(drift) / (2.0 * t["full_error"]) if t["full_error"] else float("nan")
    print(f"  漂移 后−前 = {drift:+.4f} kJ/mol   （σ = {t['full_error']:.4f}，判据 |漂移|/2σ）")
    print(f"  |漂移| / 2σ = {z:.2f}  → {'⚠️ 超 2σ，σ 低估' if z > 2.0 else '与报出的 σ 自洽'}")

    if not rep.get("per_window"):
        return
    print(f"\n  逐窗最大 |漂移|/2σ = {rep.get('max_window_drift_over_2sigma', float('nan')):.2f}")
    print(f"  {'win':>4s} {'帧':>6s} {'全量':>9s} {'前半':>9s} {'后半':>9s} {'后−前':>8s} {'σ_win':>7s} {'/2σ':>6s}")
    for w, v in rep["per_window"].items():
        n = rep["n_frames_per_window"].get(str(w), rep["n_frames_per_window"].get(int(w), 0))
        d = v["second_half"] - v["first_half"]
        z = v.get("drift_over_2sigma")
        flag = " ⚠️" if z is not None and z > 2.0 else ""
        print(
            f"  {w:>4s} {n:>6d} {v['full']:>9.3f} {v['first_half']:>9.3f} "
            f"{v['second_half']:>9.3f} {d:>+8.3f} {v.get('sigma', float('nan')):>7.3f} "
            f"{(z if z is not None else float('nan')):>6.2f}{flag}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="溶剂腿运行目录（含 vanishing/ 与 checkpoints/），可重复给多个",
    )
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument(
        "--stage",
        choices=("vanishing", "decharging", "attachment", "both"),
        default="vanishing",
        help="查哪个 stage。decharging=stage1 全局 REMD MBAR，"
        "attachment=stage0 Boresch A′→A 腿（--run-dir 指向 attachment_rerun/<戳>），"
        "两者逐窗一栏都为空；both=vanishing+decharging",
    )
    parser.add_argument("--out", default="split_half_convergence.json")
    args = parser.parse_args(argv)

    stages = ("vanishing", "decharging") if args.stage == "both" else (args.stage,)
    reports = []
    for run_dir in args.run_dir:
        for stage in stages:
            print(f"\n{'=' * 70}\n{run_dir}  [{stage}]\n{'=' * 70}")
            if stage == "vanishing":
                rep = analyse(run_dir, args.temperature)
            elif stage == "attachment":
                rep = analyse_attachment(run_dir, args.temperature)
            else:
                rep = analyse_decharging(run_dir, args.temperature)
            rep.setdefault("stage", stage)
            reports.append(rep)
            _print_report(rep)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2, ensure_ascii=False)
    print(f"\n结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
