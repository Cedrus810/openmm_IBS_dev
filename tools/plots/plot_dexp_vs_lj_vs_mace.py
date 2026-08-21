"""
LJ / DEXP(12,6) / DEXP(14,5) 势能函数形状对比，叠加真实 MACE 局部扰动数据。

Panel A: 绝对 U(r) 曲线——用实际系统里某个代表性 ligand-environment 原子对的
         真实 sigma/epsilon（来自 perturb_scan_geometry.npz，选 anchor 帧里挨得
         最近的那一对，最能代表短程排斥墙的真实尺度），三条曲线用同一对 sigma/eps。

Panel B: anchor-relative ΔU(Δr_closest) 分析曲线（同一代表性 pair，在该 anchor
         实际的最近距离 r0_anchor 附近展开），叠加 --perturb-scan 里全部 1480 条
         扰动记录的散点 (Δr_closest, ΔE_target)，并叠加分箱 mean±SEM 趋势线提高
         判读精细度。这不是逐 pair 精确对比（MACE 是多体的，没有单一 pair 曲线），
         而是"最近接触坐标"上的诊断性叠加，用来看三条解析曲线的形状是否跟真实
         局部势能面的整体走向一致。

Panel C/D（局部精细图，DEXP_KERNEL_PHYSICS_ISSUES.md §3.2-3.4 思路的延伸）：
         不再用单一代表性 pair 的解析曲线做近似，而是对每条扰动记录用
         --perturb-fit 同款的向量化公式重算*完整* ligand-environment pairwise
         DEXP 基线 ΔU_DEXP_full(alpha,beta)（对所有 pair 求和，含真实 cutoff），
         按 (pert_type, magnitude, sign) 这个数据本身的离散扰动档分箱，画出
         mean±SEM/std 的精细误差棒图：
           - Panel C：仅 translation，x=Δr_closest（该扰动类型的自然坐标），
             限制在 magnitude<=0.02nm 的"局部"档（0.04nm 那档在文档里被认定为
             最偏离"局部"定义、专门降权，这里也不画，避免拉伸坐标轴掩盖近场细节）。
           - Panel D：仅 rotation，x=signed 转角(度)（该扰动类型真正被控制的
             变量，不是派生的 Δr_closest——旋转并不主要通过改变最近距离起作用）。
         两个 kernel (12,6)/(14,5) 都画，可以直接看哪个核在哪个扰动类型/幅度上
         更贴近真实 ΔE_target，对应文档 §3.4 的 odd/even 分解结论。
"""

# Allow direct execution from tools/* while keeping live modules at repo root.
import sys as _abfe_sys
from pathlib import Path as _AbfePath

_ABFE_REPO_ROOT = _AbfePath(__file__).resolve().parents[2]
if str(_ABFE_REPO_ROOT) not in _abfe_sys.path:
    _abfe_sys.path.insert(0, str(_ABFE_REPO_ROOT))

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# DejaVu Sans(matplotlib 默认字体)没有 CJK 字形，中文标签会整段丢字。第一版想用
# "Noto Sans CJK SC"，但那是个 .ttc 多字体合集文件——matplotlib 扫描 .ttc 时实测
# 只登记出了 "Noto Sans CJK JP" 这个名字（尽管 fc-list 能看到 SC 变体在磁盘上），
# 精确名字匹配找不到、静默不生效。改用 adobe-source-han-sans 包装的
# "Source Han Sans CN"——是普通 .otf 单文件，用 fm.findfont 实测能正确解析路径。
_cjk_font = "Source Han Sans CN"
try:
    _resolved = fm.findfont(_cjk_font, fallback_to_default=False)
    plt.rcParams["font.sans-serif"] = [_cjk_font] + plt.rcParams["font.sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"
    print(f"CJK 字体: {_cjk_font} -> {_resolved}")
except Exception as exc:
    print(f"⚠️ 未能解析 CJK 字体 {_cjk_font}，中文标签可能仍然缺字形: {exc}")
plt.rcParams["axes.unicode_minus"] = False  # CJK 字体里负号"−"经常缺字形，退回 ASCII '-'

BASE = "output/dexp_experiment"
BASELINE_CUTOFF_NM = 0.70  # 与 --perturb-baseline-cutoff-nm / DEXP_VDW_CUTOFF_NM 一致
KERNELS = [(12.0, 6.0, "DEXP(12,6)", "#F76707"), (14.0, 5.0, "DEXP(14,5)", "#2F9E44")]

geo = np.load(f"{BASE}/perturb_scan_geometry.npz")

sigma_lig, eps_lig = geo["sigma_lig"], geo["eps_lig"]
sigma_env, eps_env = geo["sigma_env"], geo["eps_env"]
anchor_lig = geo["anchor_lig_positions"]
env_pos = geo["env_positions"]
perturbed_lig = geo["perturbed_lig_positions"]
box_vectors = geo["box_vectors"]
has_periodic = geo["has_periodic"].astype(bool)
anchor_local_idx = geo["perturbation_anchor_index"].astype(int)

# 找全部 anchor 里最近的那一对 lig-env 原子，作为代表性 pair——但必须限定
# eps_lig[i]>0 且 eps_env[j]>0，否则最近的那一对经常是氢原子（很多力场把 H 的
# LJ epsilon 设成 0，短程排斥完全靠重原子的 LJ 半径覆盖），选到 eps_ij=0 的 pair
# 会让三条曲线全部退化成一条 U(r)=0 的直线（第一版画图就踩了这个坑）。
best = None
for a in range(anchor_lig.shape[0]):
    d = np.linalg.norm(anchor_lig[a][:, None, :] - env_pos[a][None, :, :], axis=-1)
    valid_mask = (eps_lig[:, None] > 1.0e-6) & (eps_env[None, :] > 1.0e-6)
    d_masked = np.where(valid_mask, d, np.inf)
    i, j = np.unravel_index(np.argmin(d_masked), d_masked.shape)
    if not np.isfinite(d_masked[i, j]):
        continue
    if best is None or d_masked[i, j] < best[0]:
        best = (d_masked[i, j], a, i, j)
r0_anchor, a_star, i_star, j_star = best
sigma_ij = 0.5 * (sigma_lig[i_star] + sigma_env[j_star])
eps_ij = float(np.sqrt(max(eps_lig[i_star] * eps_env[j_star], 0.0)))
print(f"代表性 pair: anchor={a_star} sigma_ij={sigma_ij:.4f}nm eps_ij={eps_ij:.4f}kJ/mol r0(anchor实测最近距离)={r0_anchor:.4f}nm")

r0_lj = (2.0 ** (1.0 / 6.0)) * sigma_ij


def u_lj(r):
    return 4.0 * eps_ij * ((sigma_ij / r) ** 12 - (sigma_ij / r) ** 6)


def u_dexp(r, alpha, beta):
    x = r / r0_lj - 1.0
    c_a = beta / (alpha - beta)
    c_b = alpha / (alpha - beta)
    return eps_ij * (c_a * np.exp(-alpha * x) - c_b * np.exp(-beta * x))


kernels = [
    ("LJ 12-6", lambda r: u_lj(r), "#4C6EF5"),
    ("DEXP(12,6)", lambda r: u_dexp(r, 12.0, 6.0), "#F76707"),
    ("DEXP(14,5)", lambda r: u_dexp(r, 14.0, 5.0), "#2F9E44"),
]

rows = list(csv.DictReader(open(f"{BASE}/perturb_scan_diagnostics.csv")))
n_rows = len(rows)
dr_closest = np.array([float(r_["min_dist_perturbed_nm"]) - float(r_["min_dist_anchor_nm"]) for r_ in rows])
delta_e_target = np.array([float(r_["delta_e_target_kjmol"]) for r_ in rows])
pert_type = np.array([r_["pert_type"] for r_ in rows], dtype=object)
magnitude = np.array([float(r_["magnitude"]) for r_ in rows], dtype=float)
sign = np.array([float(r_["sign"]) for r_ in rows], dtype=float)

# --- 完整 ligand-environment pairwise DEXP 基线（与 --perturb-fit::_predict_delta_u 同一套公式），
# --- 用于 Panel C/D：比 Panel A/B 的单一代表性 pair 近似更精细、更忠实。
n_anchors = anchor_lig.shape[0]
n_lig, n_env = sigma_lig.shape[0], sigma_env.shape[0]
dists_anchor_full = np.empty((n_rows, n_lig, n_env), dtype=np.float64)
dists_pert_full = np.empty((n_rows, n_lig, n_env), dtype=np.float64)
for a in range(n_anchors):
    bv = box_vectors[a] if has_periodic[a] else None

    def _dists(a_pos, b_pos):
        delta = a_pos[:, None, :] - b_pos[None, :, :]
        if bv is not None:
            box_lens = np.linalg.norm(bv, axis=1)
            delta = delta - box_lens * np.round(delta / box_lens)
        return np.linalg.norm(delta, axis=-1)

    rows_of_anchor = np.where(anchor_local_idx == a)[0]
    if rows_of_anchor.size == 0:
        continue
    dists_anchor_full[rows_of_anchor] = _dists(anchor_lig[a], env_pos[a])[None, :, :]
    lig_block = perturbed_lig[rows_of_anchor]
    delta = lig_block[:, :, None, :] - env_pos[a][None, None, :, :]
    if bv is not None:
        box_lens = np.linalg.norm(bv, axis=1)
        delta = delta - box_lens * np.round(delta / box_lens)
    dists_pert_full[rows_of_anchor] = np.linalg.norm(delta, axis=-1)

sigma_ij_full = 0.5 * (sigma_lig[:, None] + sigma_env[None, :])
eps_ij_full = np.sqrt(np.clip(eps_lig[:, None] * eps_env[None, :], 0.0, None))
r0_ij_full = (2.0 ** (1.0 / 6.0)) * np.maximum(sigma_ij_full, 1.0e-6)
x_anchor_full = np.maximum(dists_anchor_full, 1.0e-6) / r0_ij_full[None, :, :] - 1.0
x_pert_full = np.maximum(dists_pert_full, 1.0e-6) / r0_ij_full[None, :, :] - 1.0
mask_anchor_full = dists_anchor_full <= BASELINE_CUTOFF_NM
mask_pert_full = dists_pert_full <= BASELINE_CUTOFF_NM


def predict_delta_u_full(alpha, beta):
    c_a = beta / (alpha - beta)
    c_b = alpha / (alpha - beta)
    u_anchor = np.sum(
        np.where(mask_anchor_full, eps_ij_full[None] * (c_a * np.exp(-alpha * x_anchor_full) - c_b * np.exp(-beta * x_anchor_full)), 0.0),
        axis=(1, 2),
    )
    u_pert = np.sum(
        np.where(mask_pert_full, eps_ij_full[None] * (c_a * np.exp(-alpha * x_pert_full) - c_b * np.exp(-beta * x_pert_full)), 0.0),
        axis=(1, 2),
    )
    return u_pert - u_anchor


delta_u_full = {(a, b): predict_delta_u_full(a, b) for a, b, _, _ in KERNELS}

fig, axes = plt.subplots(2, 2, figsize=(12, 9.4))
plt.rcParams.update({"font.size": 10})

# --- Panel A: 绝对 U(r) ---
ax = axes[0, 0]
r = np.linspace(0.5 * r0_lj, 1.8 * r0_lj, 400)
for label, fn, color in kernels:
    ax.plot(r, fn(r), label=label, color=color, linewidth=2)
ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--")
ax.axvline(r0_lj, color="0.6", linewidth=0.8, linestyle=":")
ax.set_ylim(-2.0 * eps_ij, 6.0 * eps_ij)
ax.set_xlabel("r (nm)")
ax.set_ylabel("U(r) (kJ/mol)")
ax.set_title(f"A. 绝对势能曲线（代表性 pair: σ={sigma_ij:.3f}nm, ε={eps_ij:.3f}kJ/mol）")
ax.legend(frameon=False)

# --- Panel B: anchor-relative ΔU(Δr) + 真实 MACE 扰动散点 + 分箱趋势线 ---
ax = axes[0, 1]
delta_r = np.linspace(-0.06, 0.06, 300)
for label, fn, color in kernels:
    if label == "LJ 12-6":
        du = u_lj(r0_lj + delta_r) - u_lj(r0_lj)
    else:
        alpha, beta = (12.0, 6.0) if "12,6" in label else (14.0, 5.0)
        du = u_dexp(r0_lj + delta_r, alpha, beta) - u_dexp(r0_lj, alpha, beta)
    ax.plot(delta_r, du, color=color, linewidth=2, label=label + " (解析)")

ax.scatter(dr_closest, delta_e_target, s=6, alpha=0.2, color="0.25", label="真实 MACE ΔE_target (全部记录)", zorder=0)

# 分箱 mean±SEM 趋势线：提高判读精细度，不受单点噪声影响
n_bins = 24
bin_edges = np.linspace(-0.06, 0.06, n_bins + 1)
bin_centers, bin_mean, bin_sem = [], [], []
for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    m = (dr_closest >= lo) & (dr_closest < hi)
    if np.sum(m) < 3:
        continue
    bin_centers.append(0.5 * (lo + hi))
    bin_mean.append(np.mean(delta_e_target[m]))
    bin_sem.append(np.std(delta_e_target[m], ddof=1) / np.sqrt(np.sum(m)))
ax.errorbar(bin_centers, bin_mean, yerr=bin_sem, fmt="o-", color="black", markersize=3.5,
            linewidth=1.2, capsize=2, label="真实 MACE 分箱 mean±SEM", zorder=1)

ax.set_xlim(-0.06, 0.06)
ax.set_ylim(-30, 60)
ax.set_xlabel("Δ(最近 lig-env 接触距离) (nm)，相对各自 anchor")
ax.set_ylabel("ΔE / ΔU (kJ/mol)，相对各自 anchor")
ax.set_title("B. anchor-relative：解析曲线(代表性pair) vs 真实局部扰动数据")
ax.legend(frameon=False, fontsize=7.5, loc="upper right")

# --- Panel C: 局部精细图 1 —— translation-only，完整 pairwise 基线 ---
ax = axes[1, 0]
trans_mask_local = (pert_type == "translation") & (magnitude <= 0.02 + 1.0e-9)
trans_mags = sorted(set(magnitude[pert_type == "translation"]))
trans_mags_local = [m for m in trans_mags if m <= 0.02 + 1.0e-9]
for m in trans_mags_local:
    for s in (1.0, -1.0):
        sel = trans_mask_local & np.isclose(magnitude, m) & (sign == s)
        if not np.any(sel):
            continue
        x_mean, x_std = np.mean(dr_closest[sel]), np.std(dr_closest[sel], ddof=1) if np.sum(sel) > 1 else 0.0
        y_mean = np.mean(delta_e_target[sel])
        y_sem = np.std(delta_e_target[sel], ddof=1) / np.sqrt(np.sum(sel)) if np.sum(sel) > 1 else 0.0
        ax.errorbar([x_mean], [y_mean], xerr=[x_std], yerr=[y_sem], fmt="o", color="black",
                    markersize=6, capsize=3, zorder=3,
                    label="真实 MACE ΔE_target (分箱 mean±SEM/std)" if (m == trans_mags_local[0] and s == 1.0) else None)
        for alpha, beta, klabel, kcolor in KERNELS:
            du = delta_u_full[(alpha, beta)][sel]
            ax.errorbar([x_mean], [np.mean(du)], xerr=[x_std],
                        yerr=[np.std(du, ddof=1) / np.sqrt(np.sum(sel)) if np.sum(sel) > 1 else 0.0],
                        fmt="^", color=kcolor, markersize=6, capsize=3, alpha=0.85, zorder=2,
                        label=f"{klabel} (完整 pairwise 基线)" if (m == trans_mags_local[0] and s == 1.0) else None)
ax.axhline(0.0, color="0.75", linewidth=0.8, linestyle="--")
ax.axvline(0.0, color="0.75", linewidth=0.8, linestyle="--")
ax.set_xlabel("Δ(最近 lig-env 接触距离) (nm)，相对各自 anchor")
ax.set_ylabel("ΔE / ΔU (kJ/mol)")
ax.set_title("C. 局部精细图：translation-only（<=0.02nm），完整 pairwise 基线 vs 真实 MACE")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")

# --- Panel D: 局部精细图 2 —— rotation-only，完整 pairwise 基线，x=signed 转角(度) ---
ax = axes[1, 1]
rot_mags = sorted(set(magnitude[pert_type == "rotation"]))
for m in rot_mags:
    for s in (1.0, -1.0):
        sel = (pert_type == "rotation") & np.isclose(magnitude, m) & (sign == s)
        if not np.any(sel):
            continue
        x_val = s * m
        y_mean = np.mean(delta_e_target[sel])
        y_sem = np.std(delta_e_target[sel], ddof=1) / np.sqrt(np.sum(sel)) if np.sum(sel) > 1 else 0.0
        ax.errorbar([x_val], [y_mean], yerr=[y_sem], fmt="o", color="black", markersize=6, capsize=3, zorder=3,
                    label="真实 MACE ΔE_target (分箱 mean±SEM)" if (m == rot_mags[0] and s == 1.0) else None)
        for alpha, beta, klabel, kcolor in KERNELS:
            du = delta_u_full[(alpha, beta)][sel]
            y_du_sem = np.std(du, ddof=1) / np.sqrt(np.sum(sel)) if np.sum(sel) > 1 else 0.0
            ax.errorbar([x_val], [np.mean(du)], yerr=[y_du_sem], fmt="^", color=kcolor, markersize=6,
                        capsize=3, alpha=0.85, zorder=2,
                        label=f"{klabel} (完整 pairwise 基线)" if (m == rot_mags[0] and s == 1.0) else None)
ax.axhline(0.0, color="0.75", linewidth=0.8, linestyle="--")
ax.axvline(0.0, color="0.75", linewidth=0.8, linestyle="--")
ax.set_xlabel("signed 旋转角 (度)，相对各自 anchor")
ax.set_ylabel("ΔE / ΔU (kJ/mol)")
ax.set_title("D. 局部精细图：rotation-only，完整 pairwise 基线 vs 真实 MACE（x=真实控制变量）")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")

fig.tight_layout()
out_path = "dexp_vs_lj_vs_mace.png"
fig.savefig(out_path, dpi=160)
print(f"saved: {out_path}")
