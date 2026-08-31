#!/usr/bin/env python
"""校验 Boresch 平衡值与 LambdaDependentBoreschForce 的几何约定是否一致。

背景（2026-07-29）：`output_lrc_fix/boresch_simple.json` 里同一个 dict 的
`equilibrium_values` 与 `diagnostics.fluctuation_distribution` 互相矛盾——
按 abfe_core.py:4635-4675，两者都是同一列数组的 `np.mean`，本不可能不同：

    equilibrium_values          fluctuation_distribution.mean
    r0      =  0.4320           r      =  0.4341
    thetaA0 =  1.2302           thetaA =  1.2098
    thetaB0 =  1.5193           thetaB =  1.5105
    phiA0   = -1.9541           phiA   = +1.9242   ← 反号
    phiB0   = +1.8516           phiB   = -1.7932   ← 反号
    phiC0   = +1.4224           phiC   = -1.5884   ← 反号

三个二面角整体反号。若 `equilibrium_values` 是错的，那么带 Boresch 的
rebalance 一开始就坐在 ~484 kJ/mol (194 kT) 的二面角势壁上，会把配体姿态
扭坏——这正是随后 `已用最后一帧安全更新 Boresch 平衡值` 把 thetaA0 提交到
偏离无约束系综 mode 5.29σ (30.1°) 的原因。

本脚本直接从无约束预平衡轨迹重算六个坐标，用的索引顺序与
`LambdaDependentBoreschForce` 的能量表达式（abfe_core.py:1260-1268）
逐项对齐，从而判定哪一组数是对的。只读，不改任何文件。

用法（openmm_dev 环境）：
    python tools/diagnostics/verify_boresch_anchor_convention.py [--output ./output]
"""
from __future__ import annotations

# 默认运行目录：统一由 tools/_run_dir.py 解析（ABFE_OUTPUT_DIR -> abfe_config.json
# 的 "output" -> ./output）。2026-08-31 前这里硬编码 output_lrc_fix，那是
# Atenolol-rank11 的验收基线目录，不在本工程区分支里。显式传参永远优先。
import sys as _abfe_rd_sys
from pathlib import Path as _AbfeRdPath

_ABFE_TOOLS_ROOT = _AbfeRdPath(__file__).resolve().parents[1]
if str(_ABFE_TOOLS_ROOT) not in _abfe_rd_sys.path:
    _abfe_rd_sys.path.insert(0, str(_ABFE_TOOLS_ROOT))
from _run_dir import DEFAULT_RUN_DIR  # noqa: E402


# Allow direct execution from tools/* while keeping live modules at repo root.
import sys as _abfe_sys
from pathlib import Path as _AbfePath

_ABFE_REPO_ROOT = _AbfePath(__file__).resolve().parents[2]
if str(_ABFE_REPO_ROOT) not in _abfe_sys.path:
    _abfe_sys.path.insert(0, str(_ABFE_REPO_ROOT))


import argparse
import json
import math
import os

import mdtraj
import numpy as np

KB_KJ = 0.0083144626  # kJ/mol/K


def wrap(d: np.ndarray) -> np.ndarray:
    """把角度差折回 [-pi, pi)。"""
    return (d + np.pi) % (2 * np.pi) - np.pi


def boresch_energy(coords: dict, eq: dict, fc: dict) -> np.ndarray:
    """复刻 abfe_core.py:1260-1268 的能量表达式（角度项也是 k*(1-cos)）。"""
    return (
        0.5 * fc["kr"] * (coords["r"] - eq["r0"]) ** 2
        + fc["kthetaA"] * (1 - np.cos(coords["thetaA"] - eq["thetaA0"]))
        + fc["kthetaB"] * (1 - np.cos(coords["thetaB"] - eq["thetaB0"]))
        + fc["kphiA"] * (1 - np.cos(coords["phiA"] - eq["phiA0"]))
        + fc["kphiB"] * (1 - np.cos(coords["phiB"] - eq["phiB0"]))
        + fc["kphiC"] * (1 - np.cos(coords["phiC"] - eq["phiC0"]))
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=DEFAULT_RUN_DIR)
    ap.add_argument("--traj", default=None, help="默认 <output>/pre_equilibration.dcd")
    ap.add_argument("--top", default=None, help="默认 <output>/topology.cif")
    ap.add_argument("--temperature", type=float, default=300.0)
    args = ap.parse_args()

    out = os.path.abspath(args.output)
    traj_path = args.traj or os.path.join(out, "pre_equilibration.dcd")
    top_path = args.top or os.path.join(out, "topology.cif")
    committed_path = os.path.join(out, "checkpoints", "boresch_equilibrium_committed.json")
    simple_path = os.path.join(out, "boresch_simple.json")

    for p in (traj_path, top_path, committed_path, simple_path):
        if not os.path.exists(p):
            raise SystemExit(f"缺文件: {p}")

    committed = json.load(open(committed_path))
    simple = json.load(open(simple_path))

    # 锚点：committed 的顺序约定是 receptor_indices[0] = 离配体最近 (R0)
    R0, R1, R2 = (int(i) for i in committed["receptor_indices"])
    L0, L1, L2 = (int(i) for i in committed["ligand_indices"])
    print(f"锚点  受体 R0={R0} R1={R1} R2={R2} | 配体 L0={L0} L1={L1} L2={L2}")
    if [int(i) for i in simple["receptor_indices"]] != [R0, R1, R2]:
        print(f"⚠️ boresch_simple.json 的受体锚点 {simple['receptor_indices']} 与 committed 不同")

    traj = mdtraj.load(traj_path, top=top_path)
    print(f"轨迹  {traj_path}  帧数={len(traj)}\n")

    # 索引顺序严格对齐 LambdaDependentBoreschForce：
    #   p1,p2,p3 = R0,R1,R2 ; p4,p5,p6 = L0,L1,L2
    #   distance(p1,p4) angle(p2,p1,p4) angle(p1,p4,p5)
    #   dihedral(p3,p2,p1,p4) dihedral(p2,p1,p4,p5) dihedral(p1,p4,p5,p6)
    coords = {
        "r": mdtraj.compute_distances(traj, [[R0, L0]])[:, 0],
        "thetaA": mdtraj.compute_angles(traj, [[R1, R0, L0]])[:, 0],
        "thetaB": mdtraj.compute_angles(traj, [[R0, L0, L1]])[:, 0],
        "phiA": mdtraj.compute_dihedrals(traj, [[R2, R1, R0, L0]])[:, 0],
        "phiB": mdtraj.compute_dihedrals(traj, [[R1, R0, L0, L1]])[:, 0],
        "phiC": mdtraj.compute_dihedrals(traj, [[R0, L0, L1, L2]])[:, 0],
    }

    print("=== 1) 无约束预平衡系综的真实几何（force 约定） ===")
    print(f"{'坐标':>8s}{'mean':>10s}{'std':>9s}{'median':>10s}"
          f"{'simple.eq':>11s}{'Δ/σ':>8s}{'simple.diag':>12s}{'Δ/σ':>8s}")
    eq_key = {"r": "r0", "thetaA": "thetaA0", "thetaB": "thetaB0",
              "phiA": "phiA0", "phiB": "phiB0", "phiC": "phiC0"}
    diag = {e["name"]: e for e in simple["diagnostics"]["fluctuation_distribution"]}
    score_eq = score_diag = 0.0
    for name, v in coords.items():
        m, s = float(np.mean(v)), float(np.std(v))
        e_eq = float(simple["equilibrium_values"][eq_key[name]])
        e_dg = float(diag[name]["mean"])
        d_eq = wrap(np.array([e_eq - m]))[0] if name != "r" else e_eq - m
        d_dg = wrap(np.array([e_dg - m]))[0] if name != "r" else e_dg - m
        score_eq += (d_eq / s) ** 2
        score_diag += (d_dg / s) ** 2
        print(f"{name:>8s}{m:10.4f}{s:9.4f}{float(np.median(v)):10.4f}"
              f"{e_eq:11.4f}{d_eq/s:7.2f}σ{e_dg:12.4f}{d_dg/s:7.2f}σ")
    print(f"\n  RMS 偏离(σ):  equilibrium_values = {math.sqrt(score_eq/6):.2f}"
          f"   |   diagnostics.mean = {math.sqrt(score_diag/6):.2f}")
    print("  → 偏离小的那一组才是与 force 约定一致的；另一组是错的。\n")

    # === 2) 三组平衡值各自的限制能 ===
    fc_committed = {k: float(v) for k, v in committed["force_constants"].items()}
    fc_simple = {k: float(v) for k, v in simple["force_constants"].items()}
    kT = KB_KJ * args.temperature
    print("=== 2) 在这条无约束轨迹上的 U_Boresch（k*(1-cos) 形式） ===")
    print(f"{'平衡值来源':<44s}{'⟨U_B⟩':>10s}{'std':>9s}{'max':>10s}{'kT倍':>8s}")
    cases = [
        ("boresch_simple.json equilibrium_values", simple["equilibrium_values"], fc_simple),
        ("boresch_simple.json diagnostics.mean",
         {eq_key[n]: float(diag[n]["mean"]) for n in coords}, fc_simple),
        ("checkpoints/boresch_equilibrium_committed", committed["equilibrium_values"], fc_committed),
        ("本轨迹自身的 mean（理想锚定）",
         {eq_key[n]: float(np.mean(v)) for n, v in coords.items()}, fc_committed),
    ]
    for label, eq, fc in cases:
        u = boresch_energy(coords, {k: float(v) for k, v in eq.items()}, fc)
        print(f"{label:<44s}{float(np.mean(u)):10.2f}{float(np.std(u)):9.2f}"
              f"{float(np.max(u)):10.2f}{float(np.mean(u))/kT:8.1f}")
    print(f"\n  参考：6 个自由度的等分定理 ⟨U_B⟩ = 3kT = {3*kT:.2f} kJ/mol")
    print("  一个锚定良好的 Boresch 限制，⟨U_B⟩ 应该在 3kT 附近；")
    print("  远大于此说明平衡值不在无约束系综的 mode 上，ΔG_attach 会被推高。")

    # === 3) 二面角是否双峰（真反转） ===
    print("\n=== 3) 二面角是否真的存在反转/双峰 ===")
    kphi = min(fc_committed[k] for k in ("kphiA", "kphiB", "kphiC"))
    print(f"  整周反转代价 = 2*min(kphi) = {2*kphi:.1f} kJ/mol")
    for name in ("phiA", "phiB", "phiC"):
        v = coords[name]
        hist, edges = np.histogram(v, bins=36, range=(-np.pi, np.pi))
        occupied = int(np.sum(hist > 0.02 * hist.max()))
        print(f"  {name}: range=[{v.min():+.3f},{v.max():+.3f}] "
              f"占据bin={occupied}/36  peak-to-peak={v.ptp():.3f} rad "
              f"({'疑似双峰/反转' if v.ptp() > 2.0 else '单峰'})")


if __name__ == "__main__":
    main()
