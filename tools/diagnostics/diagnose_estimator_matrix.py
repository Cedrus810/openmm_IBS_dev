#!/usr/bin/env python
"""四条腿 × 四个估计量的规范表 + 控制实验 + 诚实 σ（纯离线，不跑 MD）。

**为什么需要它**

本轮的数散落在 `attachment_split_half.json` / `split_half_both.json` /
`complex_split_half.json` / `docs/TODO.md` 的几张表里，已经因此产生过多次
「这个数是哪来的」的混乱。这个脚本把四条腿在四种口径下的值一次算全，
落到**唯一**的 `diagnostics/estimator_matrix.json`，之后文档只引用它。

**四个估计量**

* ``decorrelated_mbar`` —— 历史主值。先按最差目标态的 Δu 估 g 再子采样。
* ``full_frame_mbar``   —— 不做子采样。去相关**选帧本身**会移动点估计，
  且移动量依赖样本长度（实测 complex stage2：146.8241 → 143.1162）。
  正确命名是**「自相关子采样导致有限样本点估计不稳定」**——出问题的是丢帧后
  选中的有限子集，**不是「MBAR 本身有偏」**，MBAR 没有被否定。
* ``adjacent_bar``      —— 逐边双向 BAR。**只对 stage1/attachment 成立**：
  BAR 要求两个端点系综各自有样本。stage2 走 IBS，每窗只有 row 0（偏置采样
  分布）有样本、物理 λ 行 ``n_k=0``（见 ``ibs_engine.py`` 里
  ``n_k_local[sampled_row] = n_frames``），把偏置混合帧当端点帧会违反 BAR 前提，
  所以 stage2 这一格显式写 N/A 并附原因，**不留空、也不硬算**。
* ``reweighted_fd_ti``  —— ⟨∂U/∂λ⟩ 的重加权有限差分再对 λ 梯形积分。
  ∂U/∂λ 用**已有的 λ 网格**做中心差分（零新能量评估），权重取 MBAR 的 ``W_nk``。
  **stage2 同样不适用**：vdW softcore 对 λ 非线性，线性势 TI 的假设不成立，
  而这条腿从未落盘 ∂U/∂λ。所以 stage2 的 BAR 与 TI **两格都是 N/A**。

⛔ **stage2 = vdW 只能用 TMBAR。** 这张表里 stage2 只有一格有数（去相关 TMBAR，
即生产口径），BAR/TI 写 N/A + 原因。**不要为它补算任何其它口径**——2026-07-28
曾把全帧模式加进 ``GlobalMBARAnalyzer`` 并被整批撤回，原因见
``ibs_engine.ESTIMATOR_ANALYSIS_PROTOCOL_VERSION`` 的注释。

**本轮只改 charging（stage1）。** σ 口径不动：各腿只报渐近 σ，并注明它已知低估
（见 ``docs/TODO.md`` 的 P1-19）。σ 口径改造属于 vdW 侧，单独立项。

**单位陷阱**（两条腿的 u_kn 单位不同，弄反会得到差 kT 倍的结果）

* stage1 ``decharging_pme_u_kn.npy`` 是**约化势**（无量纲）→ 转 kJ/mol 要乘 kT。
* stage2 窗口三文件里的 ``energies`` 是 **kJ/mol** → 禁止再乘 beta 或 kT。

**控制实验**：先用历史口径重算 ΔG_bind，必须复现落盘的 −3.4464 kcal/mol。
对不上就说明加载口径已经不对，后面所有结论一概不可信，直接抛错退出。

**产物**：``diagnostics/estimator_matrix.json``（规范表）与
``diagnostics/candidate_new_estimator.json``（旧采样 + 新估计量的 ΔG_bind
**候选**冻结档）。**都不覆盖 final_results.json**。

用法::

    python tools/diagnostics/diagnose_estimator_matrix.py --run-dir output \\
        --attachment-dir output/attachment_rerun/20260728_154400
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
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

R_KJ = 0.008314462618
KCAL = 4.184
# 落盘的历史结果，控制实验拿它对齐。
EXPECTED_BIND_KCAL = -3.4464
BIND_TOLERANCE_KCAL = 5.0e-4

# stage1 BAR vs 重加权 FD-TI 的一致性门容差。实测 |BAR−TI| = 0.050（complex）/
# 0.226（solvent）kJ/mol，所以 0.5 是"能抓到真分歧、又不会一上线就误报"的量级。
# ⚠️ 刻意**不**沿用 attachment 的 ATTACHMENT_BAR_TI_ABS_TOL_KJ = 1.0：两条腿的
# 量级和 TI 变体都不同。
STAGE1_TI_GATE_TOL_KJ = 0.5

# attachment 腿的 σ 下界：两轮独立测量（12 态 5.3784 / 4 态 5.8238）的半程差。
# 单轮渐近 σ 0.0959 已被实测证伪 4.4 倍，**不得单独引用**。
ATTACHMENT_TWO_ROUND_HALF_RANGE_KJ = 0.2227

# 固定项（本轮冻结值，kJ/mol）。锚点/力常数一律不动，见 todo0728「已定论」。
BORESCH_ANALYTICAL_RELEASE_KJ = -37.649251830708394
CONSTRAINT_CORRECTION_COMPLEX_KJ = -1.9282057586809527
CONSTRAINT_CORRECTION_SOLVENT_KJ = -1.9280479236037105

# `result.txt`（GROMACS 参考）的逐项值，kcal/mol。只用于报告差异，不参与计算。
REFERENCE_KCAL = {
    "total": -6.279,
    "charging": -1.680,
    "vdw_annihilation_plus_lrc": -11.016 + -0.191,
    "restraint_attachment": -0.442,
    "restraint_analytical": 7.050,
}


# ---------------------------------------------------------------------------
# stage1（REMD，每态有样本）
# ---------------------------------------------------------------------------


def analyse_stage1(leg_dir: str, label: str, log=print) -> Dict:
    """四口径 + TI 门。估计量实现全部复用 `ibs_engine`，与生产口径同一份代码。"""
    from ibs_engine import stage1_estimator_crosschecks, stage1_ti_consistency_gate

    d = os.path.join(leg_dir, "decharging")
    u_path = os.path.join(d, "decharging_pme_u_kn.npy")
    u_kn = np.load(u_path, allow_pickle=False)          # 约化势
    n_k = np.load(u_path + ".n_k.npy", allow_pickle=False).astype(int)
    with open(os.path.join(d, "decharging_pme_u_kn.meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    T = float(meta["temperature_k"])
    kt = R_KJ * T
    lam = [float(x) for x in meta["lambdas_coul"]]

    cc = stage1_estimator_crosschecks(u_kn, n_k, kt, lambdas=lam)
    bar, ti = cc["adjacent_bar"], cc["reweighted_fd_ti"]
    gate = stage1_ti_consistency_gate(
        float(bar["delta_G_kJ_mol"]), float(bar["error_kJ_mol"]),
        ti.get("delta_G_kJ_mol"), STAGE1_TI_GATE_TOL_KJ,
    )

    log(f"  [{label} stage1] 去相关 {cc['decorrelated_mbar']['delta_G_kJ_mol']:.4f} | "
        f"全帧 {cc['full_frame_mbar']['delta_G_kJ_mol']:.4f} | "
        f"BAR {bar['delta_G_kJ_mol']:.4f}±{bar['error_kJ_mol']:.4f} | "
        f"FD-TI {ti['delta_G_kJ_mol']:.4f} | TI 门 {gate['passed']}")
    return {
        "leg": f"{label}_stage1",
        "sampler": "REMD (per-state samples)",
        "u_kn_units": "reduced (dimensionless) — 转 kJ/mol 需乘 kT",
        "temperature_K": T,
        "n_states": int(len(n_k)),
        "n_frames_total": int(np.sum(n_k)),
        "lambdas": lam,
        "primary_estimator": "adjacent_bar",
        "primary_delta_G_kJ_mol": float(bar["delta_G_kJ_mol"]),
        "decorrelated_mbar": cc["decorrelated_mbar"],
        "full_frame_mbar": cc["full_frame_mbar"],
        "adjacent_bar": bar,
        "reweighted_fd_ti": ti,
        "ti_consistency_gate": gate,
        # 主值的 σ = BAR 逐边方差相加（渐近）。**已知低估**（见 docs/TODO.md P1-19：
        # 渐近协方差假定样本独立同分布），但 σ 口径改造属于 vdW 侧，本轮不动。
        "sigma_kJ_mol": float(bar["error_kJ_mol"]),
        "sigma_method": "adjacent_bar_asymptotic_edge_variance_sum",
        "sigma_known_underestimated": True,
    }


# ---------------------------------------------------------------------------
# stage2（IBS，只有偏置采样分布有样本）
# ---------------------------------------------------------------------------

_STAGE2_BAR_NA = (
    "N/A —— BAR 前提不成立。IBS 每个窗口只有 row 0（偏置采样分布）有样本，"
    "物理 λ 行 n_k=0；BAR 要求两个端点系综各自有样本，把偏置混合帧当端点帧"
    "会违反其前提，得到的数看似正常实则无效。"
)

_STAGE2_TI_NA = (
    "N/A —— TI 前提不成立。vdW softcore 对 λ 非线性，`_attachment_ti()` 那种"
    "「势对 λ 线性 ⟹ ∂U/∂λ = U」的假设在这条腿上不成立；本腿也从未落盘 ∂U/∂λ，"
    "所以做不了真正的 TI。**不要**用相邻 λ 的能量差硬凑一个 FD-TI 当门——"
    "那个量（曾算出 144.85）不是这条路径的 ∂U/∂λ 积分，没有判据意义。"
)

_STAGE2_FULL_NA = (
    "N/A —— **vdW 只能用 TMBAR，本轮不动它的帧选择口径**。2026-07-28 曾给 "
    "GlobalMBARAnalyzer 加过 frame_selection='full'（complex 量到 143.1162、"
    "solvent 101.6877，相对去相关分别移动 −3.708 / −0.136 kJ/mol），随后整批撤回："
    "那是在不该动的 vdW 核心上做扩展。那两个数只作为历史观测留在 docs 里，"
    "不再是可复现的一格。若要重新评估 vdW 的帧选择，单独立项设计。"
)


def analyse_stage2(leg_dir: str, label: str, log=print) -> Dict:
    """⛔ **只跑生产口径的 TMBAR，一格数**。BAR/TI 写 N/A，不补算任何其它口径。"""
    from diagnose_endpoint_sigma import FINAL_GATES, load_lambda_path, load_outputs
    from ibs_engine import solve_stage_integrated

    lambdas_vdw, ranges = load_lambda_path(leg_dir)
    windows = load_outputs(
        os.path.join(leg_dir, "vanishing"), ranges, lambdas_vdw,
        os.path.join(leg_dir, "checkpoints"),
    )
    kt = R_KJ * 300.0

    r = solve_stage_integrated(
        window_outputs=windows, kt=kt, stage_name="vanishing", **FINAL_GATES,
    )
    per_window = [
        {"window_index": int(s["window_index"]),
         "delta_G_kJ_mol": float(s["delta_G_kJ_mol"]),
         "error_kJ_mol": float(s["uncertainty_kJ_mol"])}
        for s in (r.get("covariance_chain_segments") or [])
    ]

    log(f"  [{label} stage2] TMBAR {float(r['total_delta_G']):.4f} ± "
        f"{float(r['total_error']):.4f} | BAR N/A | TI N/A（本轮不动 vdW 口径）")

    return {
        "leg": f"{label}_stage2",
        "sampler": "IBS (single biased mixture per window; physical lambda rows n_k=0)",
        "u_kn_units": "kJ/mol — 禁止再乘 beta 或 kT",
        "n_windows": len(windows),
        "n_frames_total": int(sum(np.asarray(w["u_kn"]).shape[1] for w in windows)),
        "lambdas": [float(x) for x in lambdas_vdw],
        "primary_estimator": "local_tmbar_covariance_chain",
        "primary_delta_G_kJ_mol": float(r["total_delta_G"]),
        # 生产口径就是这一格；键名保留 decorrelated_mbar 以兼容既有读取方。
        "decorrelated_mbar": {
            "delta_G_kJ_mol": float(r["total_delta_G"]),
            "error_kJ_mol": float(r["total_error"]),
            "per_window": per_window,
        },
        "full_frame_mbar": {"delta_G_kJ_mol": None, "not_applicable_reason": _STAGE2_FULL_NA},
        "adjacent_bar": {"delta_G_kJ_mol": None, "not_applicable_reason": _STAGE2_BAR_NA},
        "reweighted_fd_ti": {"delta_G_kJ_mol": None, "not_applicable_reason": _STAGE2_TI_NA},
        "sigma_kJ_mol": float(r["total_error"]),
        "sigma_method": "local_tmbar_chain_asymptotic",
        "sigma_known_underestimated": True,
        "split_half_diagnostics": r.get("split_half_diagnostics"),
        "sigma_inflation_from_split_half": r.get("sigma_inflation_from_split_half"),
        "converged": r.get("converged"),
    }


# ---------------------------------------------------------------------------
# attachment（顺序独立窗口，每 λ 有样本）
# ---------------------------------------------------------------------------


def analyse_attachment(attachment_dir: str, log=print) -> Dict:
    """读落盘的 attachment 结果，并用原始 u_kn 复算 BAR 作一致性核对。"""
    from ibs_engine import adjacent_bar_chain

    a_dir = os.path.join(attachment_dir, "attachment")
    with open(os.path.join(a_dir, "attachment_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    cc = meta.get("crosschecks", {})
    T = float(meta.get("temperature_K", 300.0))
    kt = R_KJ * T

    bar_recomputed = None
    u_path = os.path.join(a_dir, "attachment_u_kn.npy")
    if os.path.exists(u_path):
        u_kn = np.load(u_path, allow_pickle=False)
        n_k = np.load(u_path + ".n_k.npy", allow_pickle=False).astype(int)
        # 复算一遍 BAR：证明 `_attachment_bar_chain` 改成 delegate 到
        # `adjacent_bar_chain` 之后数值逐位不变。
        dg, err, _ = adjacent_bar_chain(u_kn, n_k, kt)
        bar_recomputed = {"delta_G_kJ_mol": dg, "error_kJ_mol": err}

    primary = float(meta["attachment_delta_G_kJ_mol"])
    # ⚠️ 单轮渐近 σ (0.0959) 已被两轮独立测量证伪 4.4 倍，**不得单独引用**；
    # 采用两轮半程差。
    sigma = max(
        float(cc.get("adjacent_bar_error_kJ_mol") or 0.0),
        ATTACHMENT_TWO_ROUND_HALF_RANGE_KJ,
    )

    log(f"  [attachment] 主值(BAR) {primary:.4f} | TI "
        f"{cc.get('thermodynamic_integration_kJ_mol'):.4f} | 去相关 MBAR "
        f"{cc.get('decorrelated_mbar_kJ_mol'):.4f} | σ {sigma:.4f}（两轮半程差）")
    return {
        "leg": "attachment",
        "sampler": "sequential independent windows (per-lambda samples)",
        "primary_estimator": "adjacent_bar",
        "primary_delta_G_kJ_mol": primary,
        "adjacent_bar": {
            "delta_G_kJ_mol": cc.get("adjacent_bar_kJ_mol"),
            "error_kJ_mol": cc.get("adjacent_bar_error_kJ_mol"),
            "recomputed_from_arrays": bar_recomputed,
        },
        "reweighted_fd_ti": {
            "delta_G_kJ_mol": cc.get("thermodynamic_integration_kJ_mol"),
            "note": "attachment 的 TI 是解析线性势 TI（∂U/∂λ = U_Boresch），不是 FD-TI",
        },
        "decorrelated_mbar": {"delta_G_kJ_mol": cc.get("decorrelated_mbar_kJ_mol")},
        "full_frame_mbar": {
            "delta_G_kJ_mol": None,
            "not_applicable_reason": "attachment 腿主值走 BAR，未单独算全帧 MBAR",
        },
        "sigma_kJ_mol": sigma,
        "sigma_method": "two_round_half_range",
        "sigma_note": (
            "单轮渐近 σ 0.0959 已被两轮独立测量（12 态 5.3784 / 4 态 5.8238）"
            "证伪 4.4 倍，不得单独引用。"
        ),
    }


# ---------------------------------------------------------------------------


def _bind_kcal(complex_terms: Sequence[float], solvent_terms: Sequence[float]) -> float:
    return (float(sum(solvent_terms)) - float(sum(complex_terms))) / KCAL


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--attachment-dir", default=None,
                    help="attachment_rerun/<戳> 目录；不给就跳过 attachment 一行")
    ap.add_argument("--out", default="diagnostics/estimator_matrix.json")
    ap.add_argument("--candidate-out", default="diagnostics/candidate_new_estimator.json")
    args = ap.parse_args(argv)

    from ibs_engine import (
        ESTIMATOR_ANALYSIS_PROTOCOL_VERSION,
        estimator_policy_fingerprint,
    )

    root = os.path.abspath(args.run_dir)
    solvent = os.path.join(root, "solvent_leg")
    print(f"估计量规范表（分析协议 v{ESTIMATOR_ANALYSIS_PROTOCOL_VERSION}）"
          "：本轮只改 charging/stage1；vdW/stage2 只有 TMBAR 一格")

    legs = [
        analyse_stage1(root, "complex"),
        analyse_stage1(solvent, "solvent"),
        analyse_stage2(root, "complex"),
        analyse_stage2(solvent, "solvent"),
    ]
    by = {leg["leg"]: leg for leg in legs}

    attachment = None
    if args.attachment_dir:
        attachment = analyse_attachment(args.attachment_dir)

    # --- 控制实验 A：用历史口径复现落盘的 ΔG_bind ---
    boresch = BORESCH_ANALYTICAL_RELEASE_KJ
    cons_c, cons_s = CONSTRAINT_CORRECTION_COMPLEX_KJ, CONSTRAINT_CORRECTION_SOLVENT_KJ
    att = float(attachment["primary_delta_G_kJ_mol"]) if attachment else 0.0

    def bind(sel: str) -> float:
        return _bind_kcal(
            [by["complex_stage1"][sel]["delta_G_kJ_mol"],
             by["complex_stage2"][sel]["delta_G_kJ_mol"], att, cons_c, boresch],
            [by["solvent_stage1"][sel]["delta_G_kJ_mol"],
             by["solvent_stage2"][sel]["delta_G_kJ_mol"], cons_s],
        )

    control = bind("decorrelated_mbar")
    print(f"\n控制实验：历史口径 ΔG_bind = {control:.4f} kcal/mol"
          f"（期望 {EXPECTED_BIND_KCAL}）")
    if abs(control - EXPECTED_BIND_KCAL) > BIND_TOLERANCE_KCAL:
        raise RuntimeError(
            f"控制实验失败：复现 {control:.6f} vs 落盘 {EXPECTED_BIND_KCAL}，"
            f"差 {control - EXPECTED_BIND_KCAL:+.6f} kcal/mol。"
            "加载口径已经不对，拒绝输出估计量矩阵。"
        )
    print("  ✓ 逐位复现，加载口径正确")

    # --- 新口径：**只有 stage1 换成相邻 BAR**；stage2 仍是生产的 TMBAR，一字未动 ---
    c_terms = [by["complex_stage1"]["primary_delta_G_kJ_mol"],
               by["complex_stage2"]["primary_delta_G_kJ_mol"], att, cons_c, boresch]
    s_terms = [by["solvent_stage1"]["primary_delta_G_kJ_mol"],
               by["solvent_stage2"]["primary_delta_G_kJ_mol"], cons_s]
    bind_new = _bind_kcal(c_terms, s_terms)

    # --- σ_bind：各腿渐近 σ 平方相加。**已知低估**（见 docs/TODO.md P1-19），
    #     σ 口径改造属于 vdW 侧、单独立项，本轮不动。 ---
    sigma_terms = {leg["leg"]: float(leg["sigma_kJ_mol"]) for leg in legs}
    if attachment:
        sigma_terms["attachment"] = float(attachment["sigma_kJ_mol"])
    sigma_bind_kj = float(np.sqrt(sum(v ** 2 for v in sigma_terms.values())))
    sigma_bind_kcal = sigma_bind_kj / KCAL

    # 逐项 vs result.txt（kcal）。charging/vdW 都取"溶剂腿 − 复合物腿"的符号约定。
    charging_kcal = (by["solvent_stage1"]["primary_delta_G_kJ_mol"]
                     - by["complex_stage1"]["primary_delta_G_kJ_mol"]) / KCAL
    vdw_kcal = (by["solvent_stage2"]["primary_delta_G_kJ_mol"]
                - by["complex_stage2"]["primary_delta_G_kJ_mol"]) / KCAL
    attachment_kcal = -att / KCAL
    release_kcal = -boresch / KCAL
    breakdown = {
        "charging": {"ours": charging_kcal, "reference": REFERENCE_KCAL["charging"],
                     "ours_minus_ref": charging_kcal - REFERENCE_KCAL["charging"]},
        "vdw": {"ours": vdw_kcal,
                "reference": REFERENCE_KCAL["vdw_annihilation_plus_lrc"],
                "ours_minus_ref": vdw_kcal - REFERENCE_KCAL["vdw_annihilation_plus_lrc"]},
        "attachment": {"ours": attachment_kcal,
                       "reference": REFERENCE_KCAL["restraint_attachment"],
                       "ours_minus_ref": attachment_kcal - REFERENCE_KCAL["restraint_attachment"]},
        "analytical_release": {"ours": release_kcal,
                               "reference": REFERENCE_KCAL["restraint_analytical"],
                               "ours_minus_ref": release_kcal - REFERENCE_KCAL["restraint_analytical"]},
    }
    gap_kcal = bind_new - REFERENCE_KCAL["total"]

    print(f"\n新口径（**只换 stage1 = charging 的主值为相邻 BAR**；stage2 = vdW 未动）")
    print(f"  ΔG_bind = {bind_new:.4f} ± {sigma_bind_kcal:.4f} kcal/mol"
          f"（相对历史口径移动 {bind_new - control:+.4f}）")
    print(f"  σ 分解（kJ/mol，均为渐近值、已知低估）：" + "  ".join(
        f"{k}={v:.3f}" for k, v in sorted(sigma_terms.items())))
    print(f"  与 result.txt（{REFERENCE_KCAL['total']}）的差 = {gap_kcal:+.4f} kcal")
    for name, row in breakdown.items():
        print(f"    {name:<20} 我方 {row['ours']:+8.3f}  参考 {row['reference']:+8.3f}"
              f"  差 {row['ours_minus_ref']:+8.3f}")

    policy = {
        "estimator_analysis_protocol_version": int(ESTIMATOR_ANALYSIS_PROTOCOL_VERSION),
        "primary_estimator": "stage1=adjacent_bar;stage2=local_tmbar_unchanged",
        "charging_frame_selection": "all_frames_for_bar",
        "sigma_policy": "asymptotic_per_leg_known_underestimated",
        "sigma_inflation_applied": False,
        "ti_gate_tolerance_kJ_mol": STAGE1_TI_GATE_TOL_KJ,
    }
    fingerprint = estimator_policy_fingerprint(policy)

    payload = {
        "estimator_analysis_protocol_version": int(ESTIMATOR_ANALYSIS_PROTOCOL_VERSION),
        "estimator_policy": policy,
        "estimator_policy_fingerprint": fingerprint,
        "scope": "charging_only — vdW/stage2 未做任何改动",
        "run_dir": root,
        "attachment_dir": os.path.abspath(args.attachment_dir) if args.attachment_dir else None,
        "fixed_terms_kJ_mol": {
            "boresch_analytical_release": boresch,
            "constraint_correction_complex": cons_c,
            "constraint_correction_solvent": cons_s,
            "attachment_primary": att,
        },
        "legs": legs + ([attachment] if attachment else []),
        "binding_free_energy_kcal_mol": {
            "historical_decorrelated_mbar": control,
            "new_stage1_bar_only": bind_new,
            "shift": bind_new - control,
            "sigma_kcal_mol": sigma_bind_kcal,
            "control_expected": EXPECTED_BIND_KCAL,
            "control_passed": True,
        },
    }
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=_json_default)
    print(f"\n已写入 {out_path}")

    # --- 候选冻结档：旧采样 + 新估计量。**不覆盖 final_results.json** ---
    candidate = {
        "is_candidate": True,
        "final_results_overwritten": False,
        "note": (
            "旧采样 + 新 charging 估计量的 ΔG_bind 候选。放行顺序第 2 步要求的"
            "『一次只动一个变量』冻结档：重跑之后靠它区分「估计量修好移动了多少」与"
            "「新采样移动了多少」。**本轮唯一的变量是 charging 主值改成相邻 BAR**；"
            "vdW/stage2 与固定项一字未动。"
        ),
        "estimator_analysis_protocol_version": int(ESTIMATOR_ANALYSIS_PROTOCOL_VERSION),
        "estimator_policy": policy,
        "estimator_policy_fingerprint": fingerprint,
        "run_dir": root,
        "sampling_unchanged": True,
        "delta_G_bind_kcal_mol": bind_new,
        "sigma_kcal_mol": sigma_bind_kcal,
        "sigma_kJ_mol": sigma_bind_kj,
        "sigma_per_leg_kJ_mol": sigma_terms,
        "sigma_rule": policy["sigma_policy"],
        "historical_delta_G_bind_kcal_mol": control,
        "shift_from_historical_kcal_mol": bind_new - control,
        "reference_result_txt_kcal_mol": REFERENCE_KCAL["total"],
        "gap_vs_reference_kcal_mol": gap_kcal,
        "gap_in_sigma_ours_only_UNRELIABLE": abs(gap_kcal) / sigma_bind_kcal,
        "gap_in_sigma_note": (
            "σ 已知低估，这个倍数只是记录，**不得用来判定显著性**。"
        ),
        "per_leg_primary_kJ_mol": {
            leg["leg"]: leg["primary_delta_G_kJ_mol"] for leg in legs
        },
        "fixed_terms_kJ_mol": payload["fixed_terms_kJ_mol"],
        "term_breakdown_kcal_mol": breakdown,
        "stage2_estimators_not_applicable": {
            "adjacent_bar": _STAGE2_BAR_NA,
            "thermodynamic_integration": _STAGE2_TI_NA,
        },
        "caveats": [
            "charging 换 BAR 对 ΔG_bind 的净移动只有约 −0.140 kJ/mol = −0.033 kcal："
            "两条腿的偏移基本抵消。它是真问题，但**不是** charging 那 1.3 kcal 缺口的解释。",
            "vdW/stage2 本轮**未做任何改动**：它只能用 TMBAR。2026-07-28 曾给它加过"
            "全帧模式（complex 143.1162）并整批撤回；那个 −3.708 kJ/mol 的观测保留在"
            "docs 里，但不是本候选的一部分。",
            "σ 全部是渐近值、**已知低估**（见 docs/TODO.md P1-19）。σ 口径改造属于 "
            "vdW 侧，单独立项，本轮不动，所以这里的 σ 不要当作诚实误差棒引用。",
        ],
    }
    cand_path = os.path.abspath(args.candidate_out)
    os.makedirs(os.path.dirname(cand_path), exist_ok=True)
    with open(cand_path, "w", encoding="utf-8") as fh:
        json.dump(candidate, fh, indent=2, ensure_ascii=False, default=_json_default)
    print(f"已写入 {cand_path}（候选，未覆盖 final_results.json）")
    return 0


def _json_default(obj):
    """numpy 标量/数组 → 原生类型。诊断 dict 里混着 np.float64 与数组。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"无法序列化 {type(obj)!r}")


if __name__ == "__main__":
    sys.exit(main())
