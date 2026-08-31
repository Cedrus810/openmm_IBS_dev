"""
DEXP_KERNEL_PHYSICS_ISSUES.md §10.3/§10.4 核心结论图：K0(LJ) / K1(DEXP12,6) / K2(DEXP14,5)
对 MACE odd/even 局部扰动残差的投影能力对比，六联图，全部数据来自已生成的
`output/dexp_experiment/kernel_projection_benchmark_summary.json`（零新增计算，纯读取+画图）。

Panel A/B: 按 (pert_type, magnitude) 7 档分开的 odd/even RMSE —— 证明 DEXP 全面优于 LJ
           不依赖某个特定扰动幅度（§10.4 第一条复核）。
Panel C:   overall 的 weighted/median/trimmed10% 三种 robust 统计量 —— 证明结论不是被
           少数极端帧主导（§10.4 第四条复核）。
Panel D:   逐 anchor 的 overall RMSE（LJ vs DEXP12,6 vs DEXP14,5，log 轴，按 K2 排序）——
           可视化 §10.4"按 anchor 的整体胜负 LJ 0 / DEXP12,6 3 / DEXP14,5 17"。
Panel E/F: 逐 anchor 的 odd/even RMSE，只比较 K1 vs K2（哑铃图，按差值排序，颜色=谁赢）——
           可视化 §10.4 最关键的一条复核："odd 上 10:10 打平，even 上 19:1"。

本图是"未来多体系 benchmark 统一协议"的标准报告模板（DEXP_KERNEL_PHYSICS_ISSUES.md 相应
章节新增段落里会引用这个脚本/图作为协议的一部分）——新体系跑完 --kernel-projection-benchmark
后，只需把 BASE 指向新体系的 output 目录重新跑一遍本脚本，六个 panel 的定义不变。
"""

# Allow direct execution from tools/* while keeping live modules at repo root.
import sys as _abfe_sys
from pathlib import Path as _AbfePath

_ABFE_REPO_ROOT = _AbfePath(__file__).resolve().parents[2]
if str(_ABFE_REPO_ROOT) not in _abfe_sys.path:
    _abfe_sys.path.insert(0, str(_ABFE_REPO_ROOT))

import json
import os

import matplotlib
import matplotlib.font_manager as fm
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_cjk_font = "Source Han Sans CN"
try:
    _resolved = fm.findfont(_cjk_font, fallback_to_default=False)
    plt.rcParams["font.sans-serif"] = [_cjk_font] + plt.rcParams["font.sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"
    print(f"CJK 字体: {_cjk_font} -> {_resolved}")
except Exception as exc:
    print(f"⚠️ 未能解析 CJK 字体 {_cjk_font}，中文标签可能仍然缺字形: {exc}")
plt.rcParams["axes.unicode_minus"] = False

BASE = os.environ.get("DEXP_BENCHMARK_BASE", "output/dexp_experiment")
SYSTEM_LABEL = os.environ.get("DEXP_BENCHMARK_LABEL", "Atenolol pilot")
KERNEL_ORDER = ["K0_LJ", "K1_DEXP_12_6", "K2_DEXP_14_5"]
KERNEL_LABEL = {"K0_LJ": "LJ", "K1_DEXP_12_6": "DEXP(12,6)", "K2_DEXP_14_5": "DEXP(14,5)"}
KERNEL_COLOR = {"K0_LJ": "#4C6EF5", "K1_DEXP_12_6": "#F76707", "K2_DEXP_14_5": "#2F9E44"}
BIN_ORDER = [
    "rotation:0.5", "rotation:1.5", "rotation:3.0",
    "translation:0.005", "translation:0.01", "translation:0.02", "translation:0.04",
]
BIN_LABEL = {
    "rotation:0.5": "rot\n0.5°", "rotation:1.5": "rot\n1.5°", "rotation:3.0": "rot\n3.0°",
    "translation:0.005": "trans\n0.005nm", "translation:0.01": "trans\n0.01nm",
    "translation:0.02": "trans\n0.02nm", "translation:0.04": "trans\n0.04nm",
}

with open(f"{BASE}/kernel_projection_benchmark_summary.json", encoding="utf-8") as f:
    d = json.load(f)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    f"Kernel projection benchmark ({SYSTEM_LABEL}) — DEXP_KERNEL_PHYSICS_ISSUES.md §10.3/§10.4",
    fontsize=13, fontweight="bold",
)

# --- Panel A/B: 按幅度分档的 odd/even RMSE ---
for ax, comp, title in [(axes[0, 0], "odd", "A. odd RMSE by magnitude"),
                         (axes[0, 1], "even", "B. even RMSE by magnitude")]:
    x = np.arange(len(BIN_ORDER))
    width = 0.25
    for i, k in enumerate(KERNEL_ORDER):
        y = [d["odd_even_by_bin"][k][b][f"{comp}_residual_rmse_kjmol"] for b in BIN_ORDER]
        ax.bar(x + (i - 1) * width, y, width, label=KERNEL_LABEL[k], color=KERNEL_COLOR[k])
    ax.set_xticks(x)
    ax.set_xticklabels([BIN_LABEL[b] for b in BIN_ORDER], fontsize=8)
    ax.set_ylabel("RMSE (kJ/mol)")
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=8)

# --- Panel C: overall robust 统计量 ---
ax = axes[0, 2]
metrics = [
    ("weighted_rmse_kjmol", "weighted RMSE"),
    ("trimmed10pct_rmse_kjmol", "trimmed10% RMSE"),
    ("median_abs_residual_kjmol", "median |R|"),
]
x = np.arange(len(metrics))
width = 0.25
for i, k in enumerate(KERNEL_ORDER):
    y = [d["robust_stats_weighted_median_trimmed"][k]["overall"][m] for m, _ in metrics]
    ax.bar(x + (i - 1) * width, y, width, label=KERNEL_LABEL[k], color=KERNEL_COLOR[k])
ax.set_xticks(x)
ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=8)
ax.set_ylabel("kJ/mol")
ax.set_title("C. overall robust statistics (all 3 kernels)", fontsize=10, fontweight="bold", loc="left")
ax.legend(fontsize=8)

# --- Panel D: 逐 anchor overall RMSE，log 轴，按 K2 排序 ---
ax = axes[1, 0]
n_anchor = len(d["per_anchor_overall_rmse_mae"]["K2_DEXP_14_5"])
by_anchor = {k: {row["anchor"]: row["rmse_kjmol"] for row in d["per_anchor_overall_rmse_mae"][k]} for k in KERNEL_ORDER}
order = sorted(range(n_anchor), key=lambda a: by_anchor["K2_DEXP_14_5"][a])
x = np.arange(n_anchor)
width = 0.27
for i, k in enumerate(KERNEL_ORDER):
    y = [by_anchor[k][a] for a in order]
    ax.bar(x + (i - 1) * width, y, width, label=KERNEL_LABEL[k], color=KERNEL_COLOR[k])
ax.set_yscale("log")
ax.set_xticks([])
ax.set_xlabel(f"{n_anchor} anchors, sorted by DEXP(14,5) RMSE  —  win tally "
              f"LJ {d['win_counts_overall_rmse_by_anchor']['K0_LJ']} : "
              f"DEXP(12,6) {d['win_counts_overall_rmse_by_anchor']['K1_DEXP_12_6']} : "
              f"DEXP(14,5) {d['win_counts_overall_rmse_by_anchor']['K2_DEXP_14_5']}", fontsize=8)
ax.set_ylabel("overall RMSE (kJ/mol, log)")
ax.set_title("D. per-anchor overall RMSE, all 3 kernels", fontsize=10, fontweight="bold", loc="left")
ax.legend(fontsize=8)

# --- Panel E/F: 逐 anchor odd/even RMSE，K1 vs K2 哑铃图，按差值排序，颜色=谁赢 ---
for ax, comp, title, tally_key in [
    (axes[1, 1], "odd_rmse", "E. per-anchor odd RMSE: K1 vs K2", "odd_win_tally_k1_vs_k2"),
    (axes[1, 2], "even_rmse", "F. per-anchor even RMSE: K1 vs K2", "even_win_tally_k1_vs_k2"),
]:
    key1 = f"k1_vs_k2_per_anchor_{'odd' if comp.startswith('odd') else 'even'}_rmse"
    key2 = f"k2_per_anchor_{'odd' if comp.startswith('odd') else 'even'}_rmse"
    diff = d[key1]  # K1 - K2 (positive => K2 better, i.e. K1 rmse higher)
    k2v = d[key2]
    anchors = sorted(diff.keys(), key=lambda a: diff[a], reverse=True)
    y = np.arange(len(anchors))
    k1_vals = [k2v[a] + diff[a] for a in anchors]
    k2_vals = [k2v[a] for a in anchors]
    for yi, k1val, k2val in zip(y, k1_vals, k2_vals):
        ax.plot([k1val, k2val], [yi, yi], color="0.75", linewidth=1.5, zorder=1)
    ax.scatter(k1_vals, y, color=KERNEL_COLOR["K1_DEXP_12_6"], s=28, label="DEXP(12,6)", zorder=2)
    ax.scatter(k2_vals, y, color=KERNEL_COLOR["K2_DEXP_14_5"], s=28, label="DEXP(14,5)", zorder=2)
    ax.set_yticks([])
    ax.set_ylabel(f"{n_anchor} anchors, sorted by (K1−K2)")
    tally = d[tally_key]
    ax.set_xlabel(f"RMSE (kJ/mol) — win tally DEXP(12,6) {tally['K1_DEXP_12_6']} : "
                  f"DEXP(14,5) {tally['K2_DEXP_14_5']} : tie {tally['tie']}", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=8, loc="lower right")

fig.tight_layout(rect=[0, 0, 1, 0.96])
out_path = f"{BASE}/kernel_projection_benchmark_core_figure.png"
fig.savefig(out_path, dpi=160)
plt.close(fig)
OUTPUT_PATH = out_path
print(f"saved {out_path}")
