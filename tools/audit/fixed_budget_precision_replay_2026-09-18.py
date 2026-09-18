#!/usr/bin/env python
"""fixed_budget_precision_replay —— 完全离线的固定预算方差分配反事实。

回答一个问题：**同样的总 GPU steps，如果把 block 按边际方差下降分配，
最终 σ_ΔG 能降多少？** 10%、30%、还是没东西。

这个脚本：
  · 只读 run 产物（`vanishing*/dual_window_*_vdw_*.npy` + `convergence.json`
    + `checkpoints/ibs_state_vdw_window_*.json` + `stage2_vanishing*.json`）；
  · 调用**生产同一个估计器** `ibs_engine.solve_stage_integrated`，不自己重造
    MBAR、不自己猜规范；
  · 不写任何东西进 run 目录，不碰 production，不起 GPU。

准入自检（`--self-check`，默认开）：用某个窗口的**全部**帧重解一次，必须复现
产物 `covariance_chain_segments` 里该窗口的 ΔG。复现不了的窗口标
`unverified` 并**排除在分配研究之外** —— 宁可少算，不拿构造错的输入出数。

## a_i 的来源

对独立窗口段 `Var(ΔG) = Σ_i Var_i`，取 `Var_i ≈ a_i / n_i`（n_i = block 数）。

  · n_blocks ≥ 2：`a_i = Var(块间 ΔG)`（ddof=1）。这是直接测量，不依赖
    MBAR 渐近 σ —— 本仓库已实测渐近 σ 低估真实散布约一个数量级。
  · n_blocks == 1：把那一块对半切，两个半程各解一次，
    `a_i = Var(两个半程) / 2`（一块 = 两个半程）。**这是退化估计**，
    在报告里单独标注。

## 成本模型

`c_i` = 该窗口每块的 steps（= n_frames × steps_per_update）。同一条腿同一个
体系每块都是同样步数，所以 c_i 实际是常数，`n_i ∝ sqrt(a_i)`。
launch.log 里没有逐块墙上时间，而它本身跨 run 追加、节点间还有时钟偏移
（见 docs/archive/STAGE2_OFFLINE_FORENSICS_2026-09-18.md §11），所以不从那里挖。
真有逐块 runtime 来源时，替换 `_block_cost_steps` 即可。

## 用法

    python tools/audit/fixed_budget_precision_replay_2026-09-18.py \
        --runs-root /home/ruigengji/abfe-benchmark/openmm_IBS/runs \
        --out /tmp/replay.json --limit 4        # 先 smoke

    ... --out /tmp/replay.json                  # 全量
"""

from __future__ import annotations

# PyMBAR 的 JAX 后端会预分配整卡显存并大量吃宿主内存；这里是纯 CPU 离线活。
import os
os.environ.setdefault("PYMBAR_DISABLE_JAX", "1")

import argparse
import glob
import json
import math
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import ibs_engine  # noqa: E402

# kB in kJ/mol/K. 用错会让所有 ΔG 整体缩放，准入自检一眼抓住。
KB_KJ_PER_MOL_K = 0.008314462618
LAMBDA_TOL = 1e-9
SELF_CHECK_TOL_KJ = 0.05


# --------------------------------------------------------------------------
# 产物读取
# --------------------------------------------------------------------------

def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _segment_dirs(leg_dir: str) -> List[str]:
    """段目录有三种命名：`vanishing`（首段）、`vanishing_2`/`_3`（补块），
    以及 `vanishing_rewindow_<hash>`（布局变更后重采）。只认数字后缀会漏掉
    第三种 —— 漏掉它会让多段窗口被误判成单段，逐段去相关口径随之出错。
    """
    found = []
    for path in glob.glob(os.path.join(leg_dir, "vanishing*")):
        if not os.path.isdir(path):
            continue
        base = os.path.basename(path)
        if base == "vanishing":
            found.append((1, base, path))
            continue
        m = re.fullmatch(r"vanishing_(\d+)", base)
        if m:
            found.append((int(m.group(1)), base, path))
        else:
            # rewindow 等非数字后缀：排在数字段之后，按名字稳定排序
            found.append((10 ** 6, base, path))
    return [p for _, _, p in sorted(found)]


def _window_arrays(seg_dir: str, w: int) -> Optional[Dict[str, np.ndarray]]:
    def p(name: str) -> str:
        return os.path.join(seg_dir, f"dual_window_{w}_vdw_{name}.npy")

    try:
        u_kn = np.load(p("energies"), allow_pickle=False)
        bias = np.load(p("bias"), allow_pickle=False)
        base = np.load(p("base"), allow_pickle=False)
    except Exception:
        return None
    if u_kn.ndim != 2 or bias.ndim != 1 or base.ndim != 1:
        return None
    if u_kn.shape[1] != bias.size or bias.size != base.size:
        return None
    return {"u_kn": u_kn, "bias": bias, "base": base}


def _blocks_of(seg_dir: str, w: int, n_frames: int) -> List[Tuple[int, int]]:
    """该目录下这个窗口的 block 帧区间。缺 production_segments ⟹ 整段一块。"""
    conv = _load_json(
        os.path.join(seg_dir, f"dual_window_{w}_vdw_convergence.json")
    )
    segs = (conv or {}).get("production_segments") or []
    out: List[Tuple[int, int]] = []
    for s in segs:
        lo, hi = s.get("start_frame"), s.get("end_frame")
        if not isinstance(lo, int) or not isinstance(hi, int):
            continue
        lo, hi = max(0, lo), min(n_frames, hi)
        if hi - lo >= 2:
            out.append((lo, hi))
    return out or ([(0, n_frames)] if n_frames >= 2 else [])


# --------------------------------------------------------------------------
# 用生产估计器解单个窗口的一段帧
# --------------------------------------------------------------------------

def _make_window_payload(
    arrays: Dict[str, np.ndarray],
    lo: int,
    hi: int,
    f_k: Sequence[float],
    lambda_indices: Sequence[int],
    lambdas_vdw: Sequence[float],
    window_index: int,
    segments: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """`solve_stage_integrated` 吃的窗口字典。键集来自 ibs_engine 里对
    window_outputs 的全部 `.get(...)`；residual 臂专有键一律不给
    （本批 run 的 residual_basis 全 0，不是 residual 臂）。

    ⚠️ `production_segments` **必须是真实块边界**（相对本切片重基到 0）。
    非多段路径下求解器直接拿它当逐段去相关的段边界
    （`ibs_engine` `_decorr_segments = w.get("production_segments")`）；
    声明成假的"单段"会让多块窗口按一条连续序列估 g，实测使全量重解与
    产物段 ΔG 差 1.4–2.9 kJ/mol。
    """
    n = hi - lo
    if segments:
        prod_segments = [
            {"start_frame": int(a - lo), "end_frame": int(b - lo),
             "n_frames": int(b - a), "reason": "replay_block",
             "session_id": f"replay_block_{i}"}
            for i, (a, b) in enumerate(segments)
        ]
    else:
        prod_segments = [
            {"start_frame": 0, "end_frame": int(n), "n_frames": int(n),
             "reason": "replay_slice", "session_id": "replay"}
        ]
    return {
        "window_index": int(window_index),
        "window_label": f"replay_{window_index}",
        "window_range": [int(lambda_indices[0]), int(lambda_indices[-1]) + 1],
        "u_kn": np.asarray(arrays["u_kn"][:, lo:hi], dtype=np.float64),
        "bias_energies": np.asarray(arrays["bias"][lo:hi], dtype=np.float64),
        "base_energies": np.asarray(arrays["base"][lo:hi], dtype=np.float64),
        "lambda_indices": [int(x) for x in lambda_indices],
        "lambdas_vdw": [float(x) for x in lambdas_vdw],
        "lambdas_coul": [0.0] * len(lambdas_vdw),
        "f_k": np.asarray(f_k, dtype=np.float64),
        "sampled_distribution_row": 0,
        "production_segments": prod_segments,
    }


def _solve_delta_g(payload: Dict[str, Any], kt: float) -> Optional[float]:
    """单窗口 ΔG（join→end）。解不出来返回 None —— 那本身就是信息。"""
    try:
        res = ibs_engine.solve_stage_integrated(
            [payload], kt,
            skip_split_half_diagnostics=True,
            min_frames_per_window=10,
        )
    except Exception:
        return None
    if not isinstance(res, dict) or "error" in res:
        return None
    val = res.get("total_delta_G")
    if val is None or not np.isfinite(float(val)):
        return None
    return float(val)


# --------------------------------------------------------------------------
# 诊断量（离线复算，不读那些已知噪声大的单次门读数）
# --------------------------------------------------------------------------

def _bias_to_signal_ratio(arrays: Dict[str, np.ndarray]) -> Optional[float]:
    """ibs_engine.py:20549 的定义：sd(bias_kj) / min_k sd(u_kj_raw[k])。"""
    sd_bias = float(np.std(arrays["bias"]))
    sd_u = np.std(arrays["u_kn"], axis=1)
    pos = sd_u[sd_u > 0.0]
    return float(sd_bias / float(np.min(pos))) if pos.size else None


def _occupancy_imbalance(
    arrays: Dict[str, np.ndarray], f_k: np.ndarray, kt: float
) -> Optional[float]:
    """max/min 归一化占据。与 `_softmax_occupancy_per_state` 同式。"""
    u = arrays["u_kn"]
    if u.shape[0] != f_k.size:
        return None
    logits = (f_k[:, None] - u) / kt
    logits -= logits.max(axis=0, keepdims=True)
    wts = np.exp(logits)
    p = wts / wts.sum(axis=0, keepdims=True)
    occ = p.mean(axis=1)
    total = occ.sum()
    if not np.isfinite(total) or total <= 0:
        return None
    occ = occ / total * occ.size
    return float(occ.max() / max(occ.min(), 1e-300))


# --------------------------------------------------------------------------
# 单条腿
# --------------------------------------------------------------------------

def _stage_result(leg_dir: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    ck = os.path.join(leg_dir, "checkpoints")
    return (
        _load_json(os.path.join(ck, "stage2_vanishing_autonomous_inprogress.json")),
        _load_json(os.path.join(ck, "stage2_vanishing.json")),
    )


def analyse_leg(leg_dir: str, kt: float, self_check: bool) -> Optional[Dict[str, Any]]:
    inprog, final = _stage_result(leg_dir)
    if not final:
        return None
    payload = ((final.get("lambda_path_fingerprint") or {}).get("payload") or {})
    lambdas_var = payload.get("lambdas_var")
    if not lambdas_var:
        return None
    lambdas_var = [float(x) for x in lambdas_var]

    stored_seg = {
        int(s["window_index"]): float(s["delta_G_kJ_mol"])
        for s in ((inprog or {}).get("covariance_chain_segments") or [])
    }
    g_per_window = (inprog or {}).get("statistical_inefficiency_per_window") or []

    seg_dirs = _segment_dirs(leg_dir)
    if not seg_dirs:
        return None
    ck = os.path.join(leg_dir, "checkpoints")

    windows: List[Dict[str, Any]] = []
    for w in range(32):
        state = _load_json(os.path.join(ck, f"ibs_state_vdw_window_{w}.json"))
        if state is None:
            break
        f_k = np.asarray(state.get("f_k") or [], dtype=np.float64)
        lam_w = [float(x) for x in (state.get("lambdas_vdw") or [])]
        if f_k.size < 2 or len(lam_w) != f_k.size:
            continue

        # 全局 λ 索引：把窗口自己的 λ 值在全局表里定位。对不上 ⟹ 路径变过，
        # 这个窗口的 npy 与最终路径不是一套，跳过（不猜）。
        idx: List[int] = []
        for value in lam_w:
            hit = [i for i, g in enumerate(lambdas_var) if abs(g - value) <= LAMBDA_TOL]
            if len(hit) != 1:
                idx = []
                break
            idx.append(hit[0])
        if len(idx) != len(lam_w) or idx != sorted(idx):
            continue

        # 所有段目录里这个窗口的 block
        blocks: List[Tuple[str, int, int, Dict[str, np.ndarray]]] = []
        for sd in seg_dirs:
            arrays = _window_arrays(sd, w)
            if arrays is None or arrays["u_kn"].shape[0] != f_k.size:
                continue
            for lo, hi in _blocks_of(sd, w, arrays["u_kn"].shape[1]):
                blocks.append((sd, lo, hi, arrays))
        if not blocks:
            continue

        primary = blocks[0][3]
        rec: Dict[str, Any] = {
            "window": w,
            "n_states": int(f_k.size),
            "lambda_span": float(max(lam_w) - min(lam_w)),
            "n_blocks": len(blocks),
            "n_frames_total": int(sum(hi - lo for _, lo, hi, _ in blocks)),
            "bias_to_signal_ratio": _bias_to_signal_ratio(primary),
            "occupancy_imbalance": _occupancy_imbalance(primary, f_k, kt),
            "g": (float(g_per_window[w]) if w < len(g_per_window) else None),
            "stored_delta_G": stored_seg.get(w),
            "learning_updates": state.get("learning_updates"),
            "a_source": None,
            "a_i": None,
            "self_check": "skipped",
        }

        # ---- 准入自检：全量重解 vs 产物存的段 ΔG ----
        # 粒度是**窗口**不是腿：同一条腿里可以有的窗口跨目录、有的不跨。
        # 按腿判会把本可验证的窗口一起放过。
        own_dirs = {sd for sd, _, _, _ in blocks}
        if self_check and rec["stored_delta_G"] is not None and len(own_dirs) == 1:
            arrays = blocks[0][3]
            full = _solve_delta_g(
                _make_window_payload(
                    arrays, 0, arrays["u_kn"].shape[1], f_k, idx, lam_w, 0,
                    segments=[(lo, hi) for _, lo, hi, _ in blocks],
                ),
                kt,
            )
            if full is None:
                rec["self_check"] = "unverified_solve_failed"
            else:
                rec["replay_full_delta_G"] = full
                dev = abs(full - float(rec["stored_delta_G"]))
                rec["self_check_dev_kJ"] = dev
                rec["self_check"] = (
                    "ok" if dev <= SELF_CHECK_TOL_KJ else "unverified_mismatch"
                )
        elif self_check and len(own_dirs) > 1:
            # 跨目录窗口的产物 ΔG 来自多段合并，跟"单段全量重解"本来就不是
            # 同一份输入，不拿它当自检靶子（如实标注，不伪造通过）。
            rec["self_check"] = "skipped_multi_segment_dirs"

        # ---- a_i ----
        block_dg: List[float] = []
        for sd, lo, hi, arrays in blocks:
            val = _solve_delta_g(
                _make_window_payload(arrays, lo, hi, f_k, idx, lam_w, 0), kt
            )
            block_dg.append(val if val is not None else float("nan"))
        rec["block_delta_G"] = block_dg
        good = [x for x in block_dg if np.isfinite(x)]
        rec["n_blocks_solved"] = len(good)

        if len(good) >= 2:
            rec["a_i"] = float(np.var(good, ddof=1))
            rec["a_source"] = "block_scatter"
        else:
            # 退化：单块对半
            sd, lo, hi, arrays = blocks[0]
            mid = lo + (hi - lo) // 2
            halves = []
            for a, b in ((lo, mid), (mid, hi)):
                if b - a >= 10:
                    halves.append(
                        _solve_delta_g(
                            _make_window_payload(arrays, a, b, f_k, idx, lam_w, 0), kt
                        )
                    )
            halves = [x for x in halves if x is not None and np.isfinite(x)]
            if len(halves) == 2:
                rec["a_i"] = float(np.var(halves, ddof=1) / 2.0)
                rec["a_source"] = "split_half_degraded"
                rec["half_delta_G"] = halves
        windows.append(rec)

    if not windows:
        return None
    return {
        "leg_dir": leg_dir,
        "n_windows": len(windows),
        "windows": windows,
        "reported_total_delta_G": final.get("total_delta_G"),
        "reported_total_error": final.get("total_error"),
    }


# --------------------------------------------------------------------------
# 分配模拟
# --------------------------------------------------------------------------

def _block_cost_steps(rec: Dict[str, Any], steps_per_update: int) -> float:
    """每块成本。同腿同体系为常数；留成函数便于接入真实 runtime。"""
    n = rec.get("n_frames_total") or 0
    nb = max(1, int(rec.get("n_blocks") or 1))
    return max(1.0, float(n) / nb * float(steps_per_update))


def simulate(
    windows: List[Dict[str, Any]], steps_per_update: int
) -> Optional[Dict[str, Any]]:
    """固定总 block 数，按边际方差下降/成本贪心分配。"""
    usable = [
        w for w in windows
        if w.get("a_i") is not None and w["a_i"] > 0.0
        and not str(w.get("self_check", "")).startswith("unverified")
    ]
    if len(usable) < 2:
        return None

    a = np.array([float(w["a_i"]) for w in usable])
    n_actual = np.array([max(1, int(w["n_blocks"])) for w in usable], dtype=int)
    cost = np.array([_block_cost_steps(w, steps_per_update) for w in usable])
    budget = int(n_actual.sum())

    n_opt = np.ones_like(n_actual)
    for _ in range(budget - len(usable)):
        gain = (a / n_opt - a / (n_opt + 1)) / cost
        n_opt[int(np.argmax(gain))] += 1

    var_actual = float(np.sum(a / n_actual))
    var_opt = float(np.sum(a / n_opt))
    var_equal = float(np.sum(a / np.full_like(n_actual, max(1, budget // len(usable)))))

    return {
        "n_windows_usable": len(usable),
        "n_windows_total": len(windows),
        "total_blocks": budget,
        # 🔑 固定总预算下可搬动的块数。每个窗口至少要 1 块，所以自由度 =
        # 超出最小值的那些块。为 0 时「降幅 0%」是**自由度为零**的恒等结果，
        # 不是「测出来没收益」—— 报告和汇总必须把这两类分开。
        "reallocatable_blocks": int(budget - len(usable)),
        "has_freedom": bool(budget > len(usable)),
        "sigma_actual_kJ": math.sqrt(var_actual),
        "sigma_optimal_kJ": math.sqrt(var_opt),
        "sigma_equal_kJ": math.sqrt(var_equal),
        "sigma_reduction_pct": (
            100.0 * (1.0 - math.sqrt(var_opt) / math.sqrt(var_actual))
            if var_actual > 0 else None
        ),
        "allocation_actual": {int(w["window"]): int(k)
                              for w, k in zip(usable, n_actual)},
        "allocation_optimal": {int(w["window"]): int(k)
                               for w, k in zip(usable, n_opt)},
        "a_i": {int(w["window"]): float(w["a_i"]) for w in usable},
        "a_source": {int(w["window"]): w["a_source"] for w in usable},
    }


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    pairs = [(a, b) for a, b in zip(x, y)
             if a is not None and b is not None
             and np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 5:
        return None
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank([p[0] for p in pairs]), rank([p[1] for p in pairs])
    ma, mb = float(np.mean(ra)), float(np.mean(rb))
    num = float(np.sum((np.array(ra) - ma) * (np.array(rb) - mb)))
    den = math.sqrt(
        float(np.sum((np.array(ra) - ma) ** 2))
        * float(np.sum((np.array(rb) - mb) ** 2))
    )
    return num / den if den > 0 else None


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", required=True, help="报告 JSON 落盘路径（不写进 run 目录）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个 run（smoke）")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--steps-per-update", type=int, default=500)
    ap.add_argument("--no-self-check", action="store_true")
    args = ap.parse_args()

    kt = KB_KJ_PER_MOL_K * args.temperature
    run_dirs = sorted(
        d for d in glob.glob(os.path.join(args.runs_root, "*", "rep*"))
        if os.path.isdir(d)
    )
    if args.limit:
        run_dirs = run_dirs[: args.limit]

    report: Dict[str, Any] = {
        "kt_kJ_per_mol": kt,
        "steps_per_update": args.steps_per_update,
        "cost_model": "steps per block (constant within a leg); no per-block runtime on disk",
        "legs": [],
    }
    all_windows: List[Dict[str, Any]] = []

    for run in run_dirs:
        for leg_name, leg_dir in (("cx", run), ("sv", os.path.join(run, "solvent_leg"))):
            if not os.path.isdir(leg_dir):
                continue
            res = analyse_leg(leg_dir, kt, not args.no_self_check)
            if res is None:
                continue
            sim = simulate(res["windows"], args.steps_per_update)
            entry = {
                "run": os.path.relpath(run, args.runs_root),
                "leg": leg_name,
                "reported_total_delta_G": res["reported_total_delta_G"],
                "reported_total_error": res["reported_total_error"],
                "self_check": {
                    k: sum(1 for w in res["windows"] if w["self_check"] == k)
                    for k in sorted({w["self_check"] for w in res["windows"]})
                },
                "simulation": sim,
                "windows": res["windows"],
            }
            report["legs"].append(entry)
            for w in res["windows"]:
                all_windows.append({**w, "run": entry["run"], "leg": leg_name})
            tag = "-" if sim is None else f"{sim['sigma_reduction_pct']:.1f}%"
            print(f"  {entry['run']:<28}{leg_name}  窗口={res['n_windows']:<3}"
                  f"σ_actual={'' if sim is None else format(sim['sigma_actual_kJ'],'.2f'):>6}"
                  f"  σ_opt={'' if sim is None else format(sim['sigma_optimal_kJ'],'.2f'):>6}"
                  f"  降幅={tag:>7}  自检={entry['self_check']}")

    # ---- 先验标定：哪个诊断量预测得了 a_i ----
    sd = [math.sqrt(w["a_i"]) if w.get("a_i") else None for w in all_windows]
    report["prior_calibration"] = {
        "n_windows": len(all_windows),
        "note": (
            "目标量是 sqrt(a_i)=块级 σ，直接测量、不用 MBAR 渐近 σ。"
            "bias_to_signal_ratio 原注释的 Spearman=+0.886 出处是 6 窗口/单体系，"
            "这里是跨体系重新标定。"
        ),
        "spearman_vs_sqrt_a": {
            "bias_to_signal_ratio": spearman(
                [w.get("bias_to_signal_ratio") for w in all_windows], sd),
            "occupancy_imbalance": spearman(
                [w.get("occupancy_imbalance") for w in all_windows], sd),
            "g": spearman([w.get("g") for w in all_windows], sd),
            "n_states": spearman([w.get("n_states") for w in all_windows], sd),
            "lambda_span": spearman([w.get("lambda_span") for w in all_windows], sd),
        },
    }

    sims = [e["simulation"] for e in report["legs"] if e["simulation"]]
    free = [s for s in sims if s.get("has_freedom")]
    if sims:
        red = sorted(s["sigma_reduction_pct"] for s in free
                     if s["sigma_reduction_pct"] is not None)
        report["summary"] = {
            "n_legs_simulated": len(sims),
            "n_legs_with_freedom": len(free),
            "n_legs_zero_freedom": len(sims) - len(free),
            "zero_freedom_note": (
                "自由度为零的腿（每个窗口都只拿了 1 块）降幅恒为 0%，是恒等式"
                "不是测量结果，已从下面的统计里排除。"
            ),
            "sigma_reduction_pct_median": red[len(red) // 2] if red else None,
            "sigma_reduction_pct_min": red[0] if red else None,
            "sigma_reduction_pct_max": red[-1] if red else None,
            "reallocatable_blocks_median": (
                sorted(s["reallocatable_blocks"] for s in free)[len(free) // 2]
                if free else None
            ),
        }

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(f"\n报告 -> {args.out}")
    if "summary" in report:
        s = report["summary"]
        if s["sigma_reduction_pct_median"] is None:
            print(f"没有一条腿有重分配自由度（{s['n_legs_zero_freedom']} 条腿"
                  f"每窗口都只拿 1 块）⟹ 固定预算下 P0 无可搬动的块。")
        else:
            print(f"σ 降幅（固定总预算、最优 vs 实际）：中位 "
                  f"{s['sigma_reduction_pct_median']:.1f}%  "
                  f"范围 [{s['sigma_reduction_pct_min']:.1f}%, "
                  f"{s['sigma_reduction_pct_max']:.1f}%]  "
                  f"n={s['n_legs_with_freedom']} 条有自由度的腿"
                  f"（另有 {s['n_legs_zero_freedom']} 条自由度为零，已排除）")
    print("先验标定 Spearman(诊断量, 块级σ)：")
    for k, v in report["prior_calibration"]["spearman_vs_sqrt_a"].items():
        print(f"  {k:<24}{'n/a' if v is None else format(v, '+.3f')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
