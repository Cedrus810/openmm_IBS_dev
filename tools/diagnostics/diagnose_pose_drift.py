#!/usr/bin/env python
"""量一下配体在无约束预平衡里跑了多远——用来证伪/坐实"pose 漂移"这个诊断。

背景
----
本流程的顺序（`runabfe.resolve_boresch_restraint` 的注释里写明）是：

    基线预平衡（**无约束** 5 ns）→ 用其结果拟合 Boresch 锚点 → 带 Boresch rebalance

也就是说 Boresch 限制力锚定的是**预平衡跑完之后**的配体位置，不是输入的对接
pose。如果配体在那 5 ns 里漂了，restraint 会把漂后的状态冻住，后面三条腿全在
测那个构象。

已知症状与之一致：
  * ⟨U_elec⟩ 口袋 −144 vs 水 −181 kJ/mol（配体没做出该有的极性接触）
  * vdW 与参考差 0.42 kcal，charging 差 4.84 kcal —— 埋着但取向错的典型特征
    （接触面积不变 → vdW 不敏感；方向性氢键丢失 → 静电极度敏感）

**证伪条件：若蛋白对齐后配体 RMSD 始终 < ~1.5 Å，则 pose 被保住了，
上述诊断不成立，需要换方向查。**

只读，不建 Context，不碰 GPU。
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
import os
import sys

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--ref-gro", default="solv_ions.gro",
                    help="输入的对接结构（参考态）")
    ap.add_argument("--traj", default=None,
                    help="默认 <run-dir>/pre_equilibration.dcd")
    ap.add_argument("--also-traj", default=None,
                    help="可选：再比一条，比如 <run-dir>/rebalance_traj.dcd")
    ap.add_argument("--ligand-resname", default="MOL")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    try:
        import mdtraj as md
    except ImportError:
        print("需要 mdtraj（openmm_dev 里有）", file=sys.stderr)
        return 2

    run_dir = os.path.abspath(a.run_dir)
    traj_path = a.traj or os.path.join(run_dir, "pre_equilibration.dcd")

    print(f"参考结构: {a.ref_gro}")
    ref = md.load(a.ref_gro)
    top = ref.topology

    lig = top.select(f"resname {a.ligand_resname} and not element H")
    if lig.size == 0:
        print(f"❌ 选不到配体 resname={a.ligand_resname}", file=sys.stderr)
        return 2
    # 对齐用蛋白骨架；只取配体附近的残基更灵敏，但先用全骨架给个稳的基准
    bb = top.select("protein and backbone")
    if bb.size == 0:
        print("❌ 选不到 protein backbone", file=sys.stderr)
        return 2
    print(f"  配体重原子 {lig.size} 个；蛋白骨架 {bb.size} 个原子")

    # 口袋残基：参考态下距配体 0.5 nm 以内的蛋白残基骨架，用来测局部漂移
    pairs = md.compute_neighbors(ref, 0.5, lig, haystack_indices=top.select("protein"))[0]
    pocket_res = sorted({top.atom(int(i)).residue.index for i in pairs})
    pocket_bb = top.select(
        "backbone and (" + " or ".join(f"resid {r}" for r in pocket_res) + ")"
    ) if pocket_res else bb
    print(f"  口袋残基 {len(pocket_res)} 个（参考态 0.5 nm 以内），骨架 {pocket_bb.size} 原子")

    results = {}
    for label, path in [("pre_equilibration", traj_path), ("extra", a.also_traj)]:
        if not path:
            continue
        if not os.path.exists(path):
            print(f"\n⚠️ 跳过 {label}: 找不到 {path}")
            continue
        print(f"\n=== {label}: {os.path.basename(path)} ===")
        tr = md.load(path, top=top, stride=a.stride)
        print(f"  载入 {tr.n_frames} 帧（stride={a.stride}）")

        for aligned_on, sel in (("全蛋白骨架", bb), ("口袋骨架", pocket_bb)):
            tr_a = tr.superpose(ref, atom_indices=sel)
            d = tr_a.xyz[:, lig, :] - ref.xyz[0, lig, :]
            rmsd_nm = np.sqrt((d ** 2).sum(axis=2).mean(axis=1))
            rmsd_A = rmsd_nm * 10.0
            print(f"  [对齐: {aligned_on}] 配体重原子 RMSD (Å):"
                  f" 首 {rmsd_A[0]:.2f} | 中位 {np.median(rmsd_A):.2f}"
                  f" | 末 {rmsd_A[-1]:.2f} | 最大 {rmsd_A.max():.2f}")
            results[f"{label}__{aligned_on}"] = {
                "first_A": float(rmsd_A[0]), "median_A": float(np.median(rmsd_A)),
                "last_A": float(rmsd_A[-1]), "max_A": float(rmsd_A.max()),
                "n_frames": int(tr.n_frames),
            }

        # 质心位移：区分"整体挪走"和"原地翻转"
        tr_a = tr.superpose(ref, atom_indices=pocket_bb)
        com = tr_a.xyz[:, lig, :].mean(axis=1)
        com_ref = ref.xyz[0, lig, :].mean(axis=0)
        com_shift_A = np.linalg.norm(com - com_ref, axis=1) * 10.0
        print(f"  [对齐: 口袋骨架] 配体质心位移 (Å):"
              f" 中位 {np.median(com_shift_A):.2f} | 末 {com_shift_A[-1]:.2f}"
              f" | 最大 {com_shift_A.max():.2f}")
        results[f"{label}__com_shift"] = {
            "median_A": float(np.median(com_shift_A)),
            "last_A": float(com_shift_A[-1]), "max_A": float(com_shift_A.max()),
        }

    print()
    print("=" * 72)
    key = "pre_equilibration__口袋骨架"
    if key in results:
        last = results[key]["last_A"]
        com = results.get("pre_equilibration__com_shift", {}).get("last_A", float("nan"))
        print(f"判定（预平衡末帧，口袋骨架对齐）：配体 RMSD = {last:.2f} Å，质心位移 = {com:.2f} Å")
        print()
        if last < 1.5:
            print("  → pose 被保住了。**'配体漂走'这个诊断被证伪**，charging 的偏差")
            print("     另有原因，需要换方向查（蛋白侧构象、质子化状态、参考值本身的口径）。")
        elif com < 1.5 <= last:
            print("  → 质心没动但 RMSD 大：配体**原地翻转/重排了取向**。")
            print("     这正好解释 vdW 对得上（埋着、接触面积不变）而 charging 对不上")
            print("     （方向性氢键丢失）。Boresch 锚点是拟合在这个翻转后的取向上的。")
        else:
            print("  → 配体整体挪位了。Boresch 锚点锁的是挪位后的构象，三条腿都在测它。")
        print()
        print("  若确认漂移：修法不是重启同一条链（会复现同样的漂移），而是让预平衡")
        print("  阶段就带上位置/Boresch 约束，或把 Boresch 锚点拟合到输入 pose 而不是")
        print("  预平衡末态——见 runabfe.resolve_boresch_restraint 的调用顺序。")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as h:
            json.dump(results, h, indent=2, ensure_ascii=False)
        print(f"\n📄 已写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
