"""
DEXP_KERNEL_PHYSICS_ISSUES.md §9.1 追加图：沿 q=alpha+beta 固定的两条脊线(默认 18 和 19，
分别过 (12,6) 和 (14,5))做细扫，直接可视化"这两条脊线是不是同一条宽平山谷"这个问题。

数据来自 `dexp_experiment.py --alpha-beta-ridge-scan` 输出的
`alpha_beta_ridge_scan_by_point.csv`（每个 q 一条曲线，逐 beta 的池化 odd/even RMSE），
零新增计算，纯读取+画图。

每个 q 值一个面板：x 轴为 beta（alpha=q-beta），左 y 轴画 even RMSE，右 y 轴画 odd RMSE，
命名核（该 q 上如果落着 (12,6) 或 (14,5)）用竖直虚线+散点标出。两个面板共享同一组
y 轴范围，方便直接目视比较两条脊线的绝对高度差——如果 q=18 面板的整条曲线都明显高于
q=19 面板，说明两条脊线不等价，(12,6) 不在 MACE 认可的那条山谷上。
"""

# Allow direct execution from tools/* while keeping live modules at repo root.
import sys as _abfe_sys
from pathlib import Path as _AbfePath

_ABFE_REPO_ROOT = _AbfePath(__file__).resolve().parents[2]
if str(_ABFE_REPO_ROOT) not in _abfe_sys.path:
    _abfe_sys.path.insert(0, str(_ABFE_REPO_ROOT))

import csv
import json

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
    print(f"未能解析 CJK 字体 {_cjk_font}，中文标签可能仍然缺字形: {exc}")
plt.rcParams["axes.unicode_minus"] = False

BASE = "output/dexp_experiment"
EVEN_COLOR = "#2F9E44"
ODD_COLOR = "#4C6EF5"
NAMED_COLOR = "#E03131"

with open(f"{BASE}/alpha_beta_ridge_scan_summary.json") as f:
    summary = json.load(f)

rows_by_q = {}
with open(f"{BASE}/alpha_beta_ridge_scan_by_point.csv") as f:
    for row in csv.DictReader(f):
        rows_by_q.setdefault(float(row["q"]), []).append(
            {
                "beta": float(row["beta"]),
                "alpha": float(row["alpha"]),
                "odd": float(row["odd_rmse_kjmol"]),
                "even": float(row["even_rmse_kjmol"]),
            }
        )
for q in rows_by_q:
    rows_by_q[q].sort(key=lambda r: r["beta"])

q_values = sorted(rows_by_q.keys())
n_anchors = summary.get("n_anchors")

even_all = [r["even"] for q in q_values for r in rows_by_q[q]]
odd_all = [r["odd"] for q in q_values for r in rows_by_q[q]]
even_ylim = (0.0, max(even_all) * 1.08)
odd_ylim = (0.0, max(odd_all) * 1.08)

fig, axes = plt.subplots(1, len(q_values), figsize=(6.5 * len(q_values), 5.2), squeeze=False)
axes = axes[0]
fig.suptitle(
    f"Alpha/beta ridge scan along fixed q=alpha+beta (Atenolol pilot, n_anchors={n_anchors}) "
    "— DEXP_KERNEL_PHYSICS_ISSUES.md §9.1",
    fontsize=13, fontweight="bold",
)

named_by_q = summary.get("named_point_by_q", {})

for ax_even, q in zip(axes, q_values):
    curve = rows_by_q[q]
    beta = np.asarray([r["beta"] for r in curve])
    even = np.asarray([r["even"] for r in curve])
    odd = np.asarray([r["odd"] for r in curve])

    ax_even.plot(beta, even, color=EVEN_COLOR, linewidth=2, label="even RMSE (left axis)")
    ax_even.set_ylim(*even_ylim)
    ax_even.set_xlabel(f"beta  (alpha = {q:g} - beta)")
    ax_even.set_ylabel("even RMSE (kJ/mol)", color=EVEN_COLOR)
    ax_even.tick_params(axis="y", labelcolor=EVEN_COLOR)

    ax_odd = ax_even.twinx()
    ax_odd.plot(beta, odd, color=ODD_COLOR, linewidth=2, linestyle="--", label="odd RMSE (right axis)")
    ax_odd.set_ylim(*odd_ylim)
    ax_odd.set_ylabel("odd RMSE (kJ/mol)", color=ODD_COLOR)
    ax_odd.tick_params(axis="y", labelcolor=ODD_COLOR)

    named = named_by_q.get(str(q))
    if named is not None:
        ax_even.axvline(named[1], color=NAMED_COLOR, linewidth=1.2, linestyle=":", zorder=1)
        named_even = summary["ridge_by_q"][str(q)]["named_point_metrics"]["even_rmse_kjmol"]
        named_odd = summary["ridge_by_q"][str(q)]["named_point_metrics"]["odd_rmse_kjmol"]
        ax_even.scatter([named[1]], [named_even], color=NAMED_COLOR, s=60, zorder=3,
                        label=f"named (α={named[0]:g},β={named[1]:g})")
        ax_odd.scatter([named[1]], [named_odd], color=NAMED_COLOR, s=60, marker="D", zorder=3)

    best = summary["ridge_by_q"][str(q)]["best_on_ridge"]
    ax_even.axvline(best["beta"], color="0.6", linewidth=1.0, linestyle="-.", zorder=0)

    lines_even, labels_even = ax_even.get_legend_handles_labels()
    lines_odd, labels_odd = ax_odd.get_legend_handles_labels()
    ax_even.legend(lines_even + lines_odd, labels_even + labels_odd, fontsize=8, loc="upper center")
    ax_even.set_title(
        f"q = alpha+beta = {q:g}   (ridge best: α={best['alpha']:.2f}, β={best['beta']:.2f}, "
        f"even={best['even_rmse_kjmol']:.3f})",
        fontsize=10, fontweight="bold", loc="left",
    )
    ax_even.grid(alpha=0.25)

fig.tight_layout(rect=[0, 0, 1, 0.94])
out_path = f"{BASE}/alpha_beta_ridge_scan_core_figure.png"
fig.savefig(out_path, dpi=160)
print(f"saved {out_path}")
