#!/usr/bin/env python
"""把 decharging 的 ΔG 与纯系综平均 ⟨ΔU_elec⟩ 对照，判断偏差在系综还是在估计器。

为什么这个判据能一刀切开
------------------------
去电荷是**线性**电荷缩放（`NonbondedForce.addParticleParameterOffset`，
`q_i(λ) = λ·q_i⁰`），所以 U(λ) 对 λ 是严格二次的：

    U(λ) = U₀ + λ·A + λ²·B

线性响应（Zwanzig 二阶截断 / Marcus）给出

    ΔG(1→0) ≈ ½( ⟨ΔU⟩_{λ=1} + ⟨ΔU⟩_{λ=0} )，   ΔU ≡ U(λ=0) − U(λ=1)

其中 `⟨ΔU⟩_{λ=1}` 就是**配体与环境的全部静电相互作用能**（分子内已被
`_prepare_pme_coulomb_leg_system` 冻结，不随 λ 变，自动抵消）。

关键在于：**⟨ΔU⟩ 是纯系综平均，不经过 MBAR、不经过 BAR、不经过去相关**。
所以

  * 复合物腿的 ⟨ΔU⟩ 本身就比溶剂腿小 → 配体在口袋里没做出该有的极性接触，
    问题在 pose / 采样，与 charging 的实现无关；
  * ⟨ΔU⟩ 量级正常、但 MBAR 的 ΔG 明显偏小 → 问题在估计器或系统构造，
    要回去查 u_kn 的构造、λ 阶梯、exception 冻结那几段。

同时报线性响应估计与 MBAR 结果的差：对线性电荷缩放二者应当很接近
（几 kJ/mol 以内），差得多说明高阶项/极化响应显著，也是一条信息。

只读，不重采样，不碰 GPU。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

R_KJ = 8.31446261815324e-3


def load_leg(meta_path: str):
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    base = meta_path[: -len(".meta.json")] if meta_path.endswith(".meta.json") else meta_path
    u_kn = np.load(base + ".npy", allow_pickle=False)
    n_k_path = base + ".npy.n_k.npy"
    n_k = np.load(n_k_path, allow_pickle=False) if os.path.exists(n_k_path) else None
    return meta, u_kn, n_k


def analyse(label: str, meta_path: str) -> dict:
    meta, u_kn, n_k = load_leg(meta_path)
    kt = R_KJ * float(meta["temperature_k"])
    lam = np.asarray(meta["lambdas_coul"], dtype=float)
    K = u_kn.shape[0]
    if n_k is None:
        raise RuntimeError(f"{label}: 缺 n_k，无法定位每个态的样本区间")
    n_k = np.asarray(n_k, dtype=int)

    print(f"\n=== {label} ===")
    print(f"  n_states={K}  n_particles={meta.get('n_particles')}  "
          f"boresch={'有' if meta.get('boresch') else '无'}")
    print(f"  λ_coul: {np.round(lam, 4).tolist()}")
    print(f"  n_k: {n_k.tolist()}  总帧数={int(n_k.sum())}  u_kn.shape={u_kn.shape}")

    bounds = np.concatenate(([0], np.cumsum(n_k)))
    first, last = 0, K - 1  # λ=1 与 λ=0

    # ΔU = U(λ=0) − U(λ=1)，单位从 reduced 还原为 kJ/mol
    dU_all = (u_kn[last] - u_kn[first]) * kt

    def seg(k):
        return dU_all[bounds[k]:bounds[k + 1]]

    dU_at_1 = seg(first)   # 在 λ=1 系综上求平均 -> 配体-环境静电相互作用能
    dU_at_0 = seg(last)

    mean_1, mean_0 = float(np.mean(dU_at_1)), float(np.mean(dU_at_0))
    lr = 0.5 * (mean_1 + mean_0)

    print()
    print(f"  ⟨ΔU⟩ 在 λ=1 系综（= 配体-环境静电相互作用能，取负号即结合侧稳定化）"
          f" = {mean_1:10.2f} kJ/mol   (σ={float(np.std(dU_at_1)):.2f}, n={dU_at_1.size})")
    print(f"  ⟨ΔU⟩ 在 λ=0 系综                                        "
          f" = {mean_0:10.2f} kJ/mol   (σ={float(np.std(dU_at_0)):.2f}, n={dU_at_0.size})")
    print(f"  线性响应估计 ΔG ≈ ½(两者之和)                            = {lr:10.2f} kJ/mol")

    # 二次性自检：线性电荷缩放下 U(λ) 应严格二次，用中间态验证
    mids = []
    for k in range(K):
        u_lam = u_kn[:, bounds[k]:bounds[k + 1]]
        if u_lam.shape[1] == 0:
            continue
        mids.append((float(lam[k]), float(np.mean((u_lam[last] - u_lam[first]) * kt))))
    print()
    print("  逐态 ⟨ΔU⟩（λ_coul → kJ/mol），线性缩放下应随 λ 单调平滑：")
    for l, v in mids:
        print(f"    λ={l:6.4f}  ⟨ΔU⟩={v:10.2f}")

    return {
        "label": label,
        "mean_dU_at_lambda1_kJ_mol": mean_1,
        "mean_dU_at_lambda0_kJ_mol": mean_0,
        "linear_response_dG_kJ_mol": lr,
        "per_state_mean_dU": mids,
        "n_frames_total": int(n_k.sum()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--complex-meta", required=True,
                    help="复合物腿 decharging_pme_u_kn.meta.json")
    ap.add_argument("--solvent-meta", required=True,
                    help="溶剂腿 decharging_pme_u_kn.meta.json")
    ap.add_argument("--complex-mbar-dg", type=float, default=None,
                    help="复合物腿 MBAR 报出的 ΔG (kJ/mol)，用于对照")
    ap.add_argument("--solvent-mbar-dg", type=float, default=None,
                    help="溶剂腿 MBAR 报出的 ΔG (kJ/mol)，用于对照")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    res_c = analyse("复合物腿 decharging", a.complex_meta)
    res_s = analyse("溶剂腿 decharging", a.solvent_meta)

    print()
    print("=" * 74)
    print("判定")
    print("=" * 74)
    lr_c = res_c["linear_response_dG_kJ_mol"]
    lr_s = res_s["linear_response_dG_kJ_mol"]
    e_c = res_c["mean_dU_at_lambda1_kJ_mol"]
    e_s = res_s["mean_dU_at_lambda1_kJ_mol"]

    print(f"  配体-环境静电相互作用能 ⟨ΔU⟩_λ=1:  复合物 {e_c:.2f}   溶剂 {e_s:.2f}   "
          f"差 {e_c - e_s:+.2f} kJ/mol")
    print(f"  线性响应 ΔG:                       复合物 {lr_c:.2f}   溶剂 {lr_s:.2f}   "
          f"贡献 {lr_s - lr_c:+.2f} kJ/mol")
    print()
    print("  参考 result.txt 要求：复合物去电荷 ≈ 75.1、溶剂 ≈ 68.1，charging 对 ΔG_bind")
    print("  的贡献 ≈ −7.0 kJ/mol（即口袋里去电荷比水里更贵）。")
    print()

    if a.complex_mbar_dg is not None:
        d = lr_c - a.complex_mbar_dg
        print(f"  复合物腿：线性响应 {lr_c:.2f} vs MBAR {a.complex_mbar_dg:.2f}  →  差 {d:+.2f} kJ/mol")
    if a.solvent_mbar_dg is not None:
        d = lr_s - a.solvent_mbar_dg
        print(f"  溶剂腿  ：线性响应 {lr_s:.2f} vs MBAR {a.solvent_mbar_dg:.2f}  →  差 {d:+.2f} kJ/mol")

    print()
    print("  读法：")
    print("   · 若两腿的『线性响应 vs MBAR』都吻合（几 kJ 以内），说明估计器没问题，")
    print("     偏差在 ⟨ΔU⟩ 本身 —— 即复合物腿的系综里配体没做出该有的极性接触，")
    print("     问题是 pose / 采样，不是 charging 的实现。")
    print("   · 若复合物腿的两者显著不吻合而溶剂腿吻合，那才是构造/估计器的问题，")
    print("     下一步去查 u_kn 构造、λ 阶梯、L-L exception 冻结那几段。")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as handle:
            json.dump({"complex": res_c, "solvent": res_s}, handle, indent=2, ensure_ascii=False)
        print(f"\n📄 已写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
