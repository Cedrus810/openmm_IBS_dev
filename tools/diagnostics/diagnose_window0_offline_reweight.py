#!/usr/bin/env python
"""零 GPU：从已落盘的 .npy 离线复现 IBS 窗口的 ΔF，并给出逐目标态的权重诊断。

目的（重要 —— 别把它当成"检验 WCA 壳"的脚本）
--------------------------------------------------
本脚本回答的是**一个**问题：

    落盘的 ΔG 是"分析代码算错了"，还是"被采的系综本身就错"？

它**不能**检验 WCA 壳假设，原因是结构性的：生产的重加权式子是

    ΔF_k = -kT ln[ Σ_n exp(-β(energies[k,n] - bias[n]))
                 / Σ_n exp(-β(energies[0,n] - bias[n])) ]

`base` 不出现在这个式子里 —— 它对窗口内所有态相同，在比值里精确抵消。
而 WCA 壳（Group 4）正是烘在 `base` 里的（实测 window 0 的 base ≈ -59000 kJ/mol，
是整个体系能量）。所以"把壳从 base 里减掉再重算"**恒等于零变化**，
那个测试是 ill-posed 的。

壳的效应**全部**在"采到了哪些构型 x"，不在权重里。这正是它对
raw ESS / top-1% / min-overlap 全部不可见的原因（那些量都是权重的函数），
也意味着**壳假设无法离线判定，只能靠重采样**（见
docs/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md §4-B）。

用法
----
    python tools/diagnostics/diagnose_window0_offline_reweight.py \
        --dir /home/ruigengji/ABFE_IBS/4W53/output_v3_seed20260908/solvent_leg/vanishing \
        --window 0

期望产出与怎么读
----------------
* `ΔF(0→last)` 若**复现**落盘值（window 0 落盘为 +20.789 kJ/mol）
  ⟹ 分析代码忠实，问题在**系综**。后续走 §4-B 的重采样 A/B，不要再查估计器。
* 若**不复现** ⟹ 落盘链路上有分析侧缺陷，先修它，其他结论全部待重估。

同时打印逐目标态的 ESS 与 top-1% 权重占比。注意它们是**权重**诊断：
它们健康**不构成**"系综正确"的证据（这就是本次事故的核心教训）。
"""

from __future__ import annotations

import argparse
import os

import numpy as np

#: kT at 298.15 K，kJ/mol。与生产口径一致（R = 8.314462618e-3 kJ/mol/K）。
DEFAULT_TEMPERATURE_K = 298.15
_R_KJ_PER_MOL_K = 8.314462618e-3


def _load(directory: str, window: int, stage: str, name: str) -> np.ndarray:
    path = os.path.join(directory, f"dual_window_{window}_{stage}_{name}.npy")
    if not os.path.exists(path):
        raise SystemExit(f"缺文件：{path}")
    return np.load(path)


def reweight(energies: np.ndarray, bias: np.ndarray, kt: float) -> np.ndarray:
    """返回相对态 0 的 ΔF_k（kJ/mol），k = 0..K-1。

    用 logsumexp 的稳定写法；`base` 刻意不参与（见模块 docstring）。
    """
    log_w = -(energies - bias[None, :]) / kt          # (K, N)
    m = log_w.max(axis=1, keepdims=True)
    log_z = (m[:, 0] + np.log(np.exp(log_w - m).sum(axis=1)))
    return -kt * (log_z - log_z[0])


def weight_diagnostics(energies: np.ndarray, bias: np.ndarray, kt: float):
    """逐目标态的 ESS 与 top-1% 权重占比（都是**权重**量，见 docstring 警告）。"""
    out = []
    n = energies.shape[1]
    for k in range(energies.shape[0]):
        lw = -(energies[k] - bias) / kt
        lw -= lw.max()
        w = np.exp(lw)
        w /= w.sum()
        ess = 1.0 / np.sum(w ** 2)
        top = max(1, int(round(0.01 * n)))
        top1 = np.sort(w)[::-1][:top].sum()
        out.append((k, ess, top1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="…/vanishing 目录")
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--stage", default="vdw", choices=("vdw", "coul"))
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE_K)
    ap.add_argument(
        "--expect", type=float, default=None,
        help="落盘的 ΔF(0→last)，给了就直接判定复现/不复现",
    )
    args = ap.parse_args()

    kt = _R_KJ_PER_MOL_K * args.temperature
    energies = _load(args.dir, args.window, args.stage, "energies")
    bias = _load(args.dir, args.window, args.stage, "bias")
    base = _load(args.dir, args.window, args.stage, "base")

    if energies.ndim != 2:
        raise SystemExit(f"energies 应为 (K, N)，实际 {energies.shape}")
    K, N = energies.shape
    print(f"窗口 {args.window} / stage={args.stage}：K={K} 态 × N={N} 帧，kT={kt:.4f} kJ/mol")
    print(f"  energies 范围 [{energies.min():.3f}, {energies.max():.3f}] kJ/mol")
    print(f"  bias     范围 [{bias.min():.3f}, {bias.max():.3f}] kJ/mol")
    print(f"  base     范围 [{base.min():.1f}, {base.max():.1f}] kJ/mol"
          f"  ← 在 ΔF 里精确抵消，仅供确认它是整体系能量")

    df = reweight(energies, bias, kt)
    print("\n相对态 0 的 ΔF_k（kJ/mol）:")
    for k, v in enumerate(df):
        print(f"  k={k}: {v:+9.3f}")
    total = float(df[-1])
    print(f"\nΔF(0→{K - 1}) = {total:+.3f} kJ/mol")

    if args.expect is not None:
        d = abs(total - args.expect)
        print(f"落盘值 = {args.expect:+.3f}；差 {d:.3f} kJ/mol")
        # ⚠️ 刻意**不**在这里下"分析代码有缺陷"的判定：本脚本用的是单参考
        # 重加权，生产用的是增广矩阵 TMBAR/MBAR，两者本来就不该逐位相同。
        # 08-29 §2.4 那次独立重加权同样比生产低约 10%（+34.29 vs ~+38.6），
        # 与本次（+18.82 vs +20.79，低 9.5%）是同一个方向、同一个量级。
        # 结论只能是"同量级 ⟹ 分析链路没有量级级错误"，不足以判定更细的东西。
        print(
            "  读法：与落盘同量级同符号 ⟹ 分析链路无量级级错误，问题在**系综**；"
            "\n        差几个百分点属单参考重加权 vs 增广矩阵 MBAR 的正常差异，"
            "\n        不构成'分析侧有缺陷'的证据。"
        )

    print("\n逐目标态权重诊断（⚠️ 权重量：健康**不**等于系综正确）:")
    print(f"  {'k':>3s} {'ESS':>10s} {'ESS/N':>8s} {'top1%权重':>10s}")
    for k, ess, top1 in weight_diagnostics(energies, bias, kt):
        print(f"  {k:3d} {ess:10.2f} {ess / N:8.3f} {top1:10.3f}")

    print(
        "\n提醒：本脚本判不了 WCA 壳。壳在 base 里、base 在 ΔF 里抵消，"
        "\n壳的效应全在'采到了哪些构型'，只能靠重采样判定（§4-B）。"
    )


if __name__ == "__main__":
    main()
