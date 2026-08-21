"""
DEXP_KERNEL_PHYSICS_ISSUES.md §9.1 追加图：alpha/beta 的完整 2D 热图（真正的二维网格，不是
沿单条脊线的 1D 剖面）。数据来自 `dexp_experiment.py --alpha-beta-scale-diagnostic` 落盘的
`alpha_beta_scale_diagnostic_landscape.csv`（每个扫过的 (alpha,beta) 组合一行：odd/even
池化 RMSE），零新增计算，纯读取+画图。

想要更宽范围、更密采样的热图（"算力够"），先用更宽的 CLI 参数重新跑一遍网格再画图，例如：
    python dexp_experiment.py --alpha-beta-scale-diagnostic \
        --ab-alpha-min 9 --ab-alpha-max 19 --ab-alpha-step 0.1 \
        --ab-beta-min 2 --ab-beta-max 10 --ab-beta-step 0.1
这个范围能同时看清 q=18(过(12,6))这条线在不受原始12-16窗口边界截断时的真实最优值，
以及 q=19(过(14,5))附近谷底的宽度，而不只是两条独立脊线的 1D 切面。

两个面板：even RMSE（log色标，因为从谷底~2.7到边缘~30+kJ/mol跨一个数量级以上）、
odd RMSE（线性色标）。(14,5)/(12,6) 用白色菱形/圆点标出；q=alpha+beta=18/19 两条
对角参考线叠加，方便直接确认山谷是否真的贴着 q=19 这条线、以及 q=18 这条线在
热图上是否明显偏离谷底颜色。
"""

# Allow direct execution from tools/* while keeping live modules at repo root.
import sys as _abfe_sys
from pathlib import Path as _AbfePath

_ABFE_REPO_ROOT = _AbfePath(__file__).resolve().parents[2]
if str(_ABFE_REPO_ROOT) not in _abfe_sys.path:
    _abfe_sys.path.insert(0, str(_ABFE_REPO_ROOT))

import csv

import matplotlib
import matplotlib.font_manager as fm
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
NAMED_KERNELS = {"DEXP(14,5)": (14.0, 5.0), "DEXP(12,6)": (12.0, 6.0)}
Q_REFERENCE_LINES = [18.0, 19.0]

rows = []
with open(f"{BASE}/alpha_beta_scale_diagnostic_landscape.csv") as f:
    for row in csv.DictReader(f):
        rows.append(
            {
                "alpha": float(row["alpha"]),
                "beta": float(row["beta"]),
                "odd": float(row["odd_rmse_kjmol"]),
                "even": float(row["even_rmse_kjmol"]),
            }
        )

alphas = sorted({r["alpha"] for r in rows})
betas = sorted({r["beta"] for r in rows})
alpha_idx = {a: i for i, a in enumerate(alphas)}
beta_idx = {b: i for i, b in enumerate(betas)}

even_grid = np.full((len(betas), len(alphas)), np.nan)
odd_grid = np.full((len(betas), len(alphas)), np.nan)
for r in rows:
    ai, bi = alpha_idx[r["alpha"]], beta_idx[r["beta"]]
    even_grid[bi, ai] = r["even"]
    odd_grid[bi, ai] = r["odd"]

n_points = len(rows)
n_missing_even = int(np.isnan(even_grid).sum())
print(f"网格: {len(alphas)} alpha值 x {len(betas)} beta值，实际有效点 n={n_points}"
      f"（alpha<=beta的格子留空，图上显示为白色，共{n_missing_even}格）")

fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
fig.suptitle(
    "Alpha/beta full 2D landscape (Atenolol pilot) — DEXP_KERNEL_PHYSICS_ISSUES.md §9.1",
    fontsize=13, fontweight="bold",
)

alpha_edges = np.asarray(alphas + [alphas[-1] + (alphas[-1] - alphas[-2] if len(alphas) > 1 else 0.25)])
beta_edges = np.asarray(betas + [betas[-1] + (betas[-1] - betas[-2] if len(betas) > 1 else 0.25)])
alpha_edges -= (alpha_edges[1] - alpha_edges[0]) / 2.0 if len(alphas) > 1 else 0.125
beta_edges -= (beta_edges[1] - beta_edges[0]) / 2.0 if len(betas) > 1 else 0.125

panels = [
    (axes[0], even_grid, "A. even RMSE (log scale)", "viridis_r", LogNorm(vmin=np.nanmin(even_grid), vmax=np.nanmax(even_grid))),
    (axes[1], odd_grid, "B. odd RMSE (linear scale)", "viridis_r", None),
]

for ax, grid, title, cmap, norm in panels:
    mesh = ax.pcolormesh(alpha_edges, beta_edges, grid, cmap=cmap, norm=norm, shading="flat")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("RMSE (kJ/mol)")
    for q in Q_REFERENCE_LINES:
        a_line = np.linspace(max(alphas[0], q - betas[-1]), min(alphas[-1], q - betas[0]), 50)
        b_line = q - a_line
        mask = (b_line >= betas[0]) & (b_line <= betas[-1])
        ax.plot(a_line[mask], b_line[mask], color="white", linewidth=1.2, linestyle="--", alpha=0.85)
        if mask.any():
            ax.text(a_line[mask][-1], b_line[mask][-1], f" q={q:g}", color="white", fontsize=8, va="center")
    for label, (a0, b0) in NAMED_KERNELS.items():
        if alphas[0] <= a0 <= alphas[-1] and betas[0] <= b0 <= betas[-1]:
            ax.scatter([a0], [b0], color="white", edgecolor="black", s=90, marker="D", zorder=5)
            ax.annotate(label, (a0, b0), textcoords="offset points", xytext=(6, 6),
                        color="white", fontsize=9, fontweight="bold")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")

fig.tight_layout(rect=[0, 0, 1, 0.94])
out_path = f"{BASE}/alpha_beta_heatmap_core_figure.png"
fig.savefig(out_path, dpi=170)
print(f"saved {out_path}")
