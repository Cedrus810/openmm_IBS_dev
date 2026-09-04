#!/usr/bin/env python
"""λ 调度事后审计：这次运行的态数够不够？

用途
----
跑完一个体系之后，回答两个问题，**零 GPU、只读已落盘的诊断**：

1. **vdW 腿的 λ 态数够不够？** 判据是每边热力学长度
   ``δ = ∫√g dλ``（g 是实测 Fisher 度规 β²Var[dU/dλ]）。pilot 已经把逐边
   δ 落进 ``checkpoints/preopt_dual_vanishing.json`` 的
   ``path_diagnostics.optimized_edge_thermodynamic_lengths``——本工具只是把它
   读出来、跟目标 δ 比、并给出"该用多少态"。
2. **去电荷腿的 λ 态数够不够？** 那条腿走 REMD，δ 可以从交换接受率**免费**
   反解，不需要任何额外采样：高斯近似下 ⟨P_acc⟩ = erfc(δ/2)，故
   δ = 2·erfc⁻¹(P_acc)。

为什么需要这个工具
------------------
这两个量一直在落盘，但**没有任何东西拿它们去定态数**：``stage2_final_n_states``
和 ``stage1_n_states`` 都是配置里手填的常量。所以"这次跑的密度对不对"只能事后
人工核对，而且很容易忘。对多体系 benchmark 来说这是唯一能证明"各体系跑在同一
采样密度上"的证据——不同 δ 工作点的结果不可互相校准。

⚠️ 两个必须知道的口径问题
--------------------------
* **δ 的约束是最差那条边，不是平均。** 去电荷腿的
  ``mean_acceptance`` 是逐**轮**平均（一轮里所有相邻对一起平均），
  ``min_acceptance`` 是最差那一**轮**而不是最差那条**边**。一条稀边会被平均
  稀释掉——实测过 P=[0.95,0.95,0.30,0.95] 的情形：最差边 δ=1.43，而按逐轮
  平均反解只得 δ=0.37，在决定态数的量上错 4 倍。逐边字段
  （``edge_acceptance``，2026-09-03 起落盘）才是对的口径；旧运行没有这个字段，
  本工具会明确降级并说明结论只到"整体是否超采样"。
* **δ = 2·erfc⁻¹(P_acc) 是高斯近似下的模型，不是测量。** 它假设相邻态能量差
  近似高斯。接受率贴近 1 时这个反解是**病态**的（P=0.967→δ=0.058，P 掉到
  0.95→δ=0.089，差 50%），所以高接受率下只能下"明显超采样"这种定性结论，
  不能拿反解出的 δ 去精算态数。低接受率区间（0.2–0.7）才是良态的。

判据来源
--------
δ_target = 1.0 即相邻态 σ(ΔU) ≈ 1 kT。这个值不是从文献拍的，是拿本仓库两个
体系的已验证生产路径反标定的：Atenolol 23 态实测 δ_max = 1.009、4W53 23 态
δ_max = 0.641。κ ≈ 1.2 是实测的 δ_max·(N−1)/L（Atenolol 1.15、4W53 1.17），
它 > 1 是因为布点带 β=0.3 的几何下限项、不是严格等长。

用法
----
    python tools/diagnostics/audit_lambda_schedule.py <run-dir> [<run-dir> ...]
    python tools/diagnostics/audit_lambda_schedule.py <run-dir> --delta-target 1.0
    python tools/diagnostics/audit_lambda_schedule.py <run-dir> --json

``<run-dir>`` 是 ``--output`` 指向的目录。溶剂腿在 ``<run-dir>/solvent_leg/``，
本工具自动一并审计。退出码：0 = 全部体系全部腿通过；1 = 有腿不达标（可直接用在
CI/launcher 里）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# --- 判据常量（见模块文档串「判据来源」）------------------------------------
DELTA_TARGET_DEFAULT = 1.0
DELTA_OK_UPPER = 1.1     # δ_max ≲ 这个值：态数合适
DELTA_TOO_SPARSE = 1.3   # δ_max ≳ 这个值：态数偏少，工作点与别的体系不同
DELTA_OVERSAMPLED = 0.7  # δ_max ≲ 这个值：明显超采样，可以减态数
KAPPA = 1.2              # 实测 δ_max·(N−1)/L

# 接受率反解的良态区间（见模块文档串的第二条口径警告）
ACC_ILL_CONDITIONED_ABOVE = 0.85
# 参考：δ_target=1.0 对应 erfc(0.5)≈0.48 的接受率；δ∈[0.8,1.2] 对应 [0.39, 0.57]。
# 判定本身统一走 verdict_from_delta_max()（δ 口径），不再用接受率阈值，见该函数说明。


def _erfc_inv(y: float) -> float:
    """erfc⁻¹，不依赖 scipy（这个工具要能在最小环境里跑）。"""
    if not (0.0 < y < 2.0):
        return float("nan")
    # erfc⁻¹(y) = erfinv(1-y)；用 Newton 迭代解 erf(x) = 1-y
    target = 1.0 - y
    x = 0.0
    for _ in range(60):
        fx = math.erf(x) - target
        dfx = 2.0 / math.sqrt(math.pi) * math.exp(-x * x)
        if dfx == 0.0:
            break
        step = fx / dfx
        x -= step
        if abs(step) < 1e-14:
            break
    return x


def delta_from_acceptance(p_acc: float) -> float:
    """每边热力学长度 δ，由交换接受率反解。⟨P_acc⟩ = erfc(δ/2)。"""
    if p_acc is None or not (0.0 < p_acc < 1.0):
        return float("nan")
    return 2.0 * _erfc_inv(p_acc)


def recommend_n(total_length: float, delta_target: float) -> int:
    """N = ceil(κ·L/δ_target) + 1。"""
    if not (total_length and total_length > 0):
        return 0
    return int(math.ceil(KAPPA * abs(total_length) / delta_target)) + 1


def verdict_from_delta_max(d_max: float, n_now: Optional[int], n_rec: Optional[int]) -> Tuple[str, str]:
    """由 δ_max 给判定。**所有腿、所有路径共用这一个判据**。

    早先版本 vdW 侧按 δ 阈值判、去电荷侧按接受率阈值判，结果同一个物理情形被判成
    两种结论（实测：溶剂腿去电荷 δ_mean=0.27 明显超采样，却因为 mean_acc=0.85 没到
    0.90 的接受率线而被判成 MARGINAL）。判据只能有一个。
    """
    if d_max is None or not math.isfinite(d_max):
        return "UNKNOWN", "没有可用的逐边热力学长度"
    if d_max > DELTA_TOO_SPARSE:
        return "TOO_SPARSE", (
            f"δ_max={d_max:.3f} > {DELTA_TOO_SPARSE}：态数偏少，这个体系的采样密度"
            "与 δ≈1 的体系不同，跨体系对比必须标注"
        )
    if d_max > DELTA_OK_UPPER:
        return "MARGINAL", f"δ_max={d_max:.3f} 略高于 {DELTA_OK_UPPER}，建议下次加态数"
    if d_max < DELTA_OVERSAMPLED:
        extra = ""
        if n_now and n_rec:
            extra = f"，按判据只需 {n_rec} 态（实跑 {n_now}）"
        return "OVERSAMPLED", f"δ_max={d_max:.3f} < {DELTA_OVERSAMPLED}：明显超采样{extra}"
    return "OK", f"δ_max={d_max:.3f} 落在目标附近"


def _load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # 坏 JSON 不该让整个审计崩掉
        print(f"    ⚠️ 读取 {path} 失败: {exc}")
        return None


# ---------------------------------------------------------------------------
# vdW 腿
# ---------------------------------------------------------------------------
def audit_vdw(leg_dir: str, delta_target: float) -> Optional[Dict[str, Any]]:
    payload = _load(os.path.join(leg_dir, "checkpoints", "preopt_dual_vanishing.json"))
    if payload is None:
        return None
    diag = payload.get("path_diagnostics") or {}
    alloc = diag.get("subdomain_allocation") or {}
    edges = diag.get("optimized_edge_thermodynamic_lengths") or []
    total_l = diag.get("total_thermodynamic_length")
    d_max = diag.get("realized_max_edge_thermodynamic_length")
    d_min = diag.get("realized_min_edge_thermodynamic_length")
    if d_max is None and edges:
        d_max, d_min = max(edges), min(edges)

    out: Dict[str, Any] = {
        "n_states": diag.get("actual_state_count") or payload.get("n_states"),
        "placement_method": diag.get("lambda_placement_method"),
        "path_protocol_version": diag.get("path_protocol_version"),
        "total_thermodynamic_length": total_l,
        "delta_max": d_max,
        "delta_min": d_min,
        "densify_points": alloc.get("free_energy_densify_points"),
        "base_state_count": alloc.get("base_state_count_before_densify"),
        "window_state_counts": alloc.get("subdomain_state_counts"),
        "total_window_state_slots": alloc.get("total_window_state_slots"),
        # ΔF 诊断只在协议 v22 起、且 pilot 采到了 <dU/dλ> 时才有
        "max_edge_free_energy_kJ_mol": alloc.get("max_edge_free_energy_kJ_mol"),
        "subdomain_free_energy_kJ_mol": alloc.get("subdomain_free_energy_kJ_mol"),
        "recommended_n_states": recommend_n(total_l, delta_target) if total_l else None,
    }

    if d_max is None:
        out["verdict"] = "UNKNOWN"
        out["note"] = "落盘里没有逐边热力学长度，无法判定（可能是无 pilot 的几何兜底路径）"
        return out

    out["verdict"], out["note"] = verdict_from_delta_max(
        d_max, out.get("n_states"), out.get("recommended_n_states")
    )
    return out


# ---------------------------------------------------------------------------
# 去电荷腿
# ---------------------------------------------------------------------------
def audit_decharging(leg_dir: str, delta_target: float) -> Optional[Dict[str, Any]]:
    payload = _load(
        os.path.join(leg_dir, "decharging", "decharging_exchange_diagnostics.json")
    )
    if payload is None:
        return None

    n_rep = payload.get("n_replicas")
    edge_acc: List[Optional[float]] = payload.get("edge_acceptance") or []
    per_edge_available = bool(edge_acc)

    out: Dict[str, Any] = {
        "n_replicas": n_rep,
        "mean_acceptance": payload.get("mean_acceptance"),
        "per_edge_available": per_edge_available,
        "edges_without_attempts": payload.get("edges_without_attempts") or [],
    }

    if per_edge_available:
        usable = [(i, p) for i, p in enumerate(edge_acc) if p is not None]
        if not usable:
            out["verdict"] = "UNKNOWN"
            out["note"] = "所有 λ 边都没有成功执行过交换判定（探针能量非有限）"
            return out
        deltas = [(i, delta_from_acceptance(p)) for i, p in usable]
        d_max_i, d_max = max(deltas, key=lambda kv: kv[1])
        total_l = sum(d for _, d in deltas)
        worst_p = min(p for _, p in usable)
        out.update(
            {
                "edge_acceptance": [None if p is None else round(p, 4) for p in edge_acc],
                "edge_delta": [
                    None if p is None else round(delta_from_acceptance(p), 4)
                    for p in edge_acc
                ],
                "delta_max": d_max,
                "delta_max_edge": d_max_i,
                "worst_edge_acceptance": worst_p,
                "total_thermodynamic_length": total_l,
                "recommended_n_states": recommend_n(total_l, delta_target),
            }
        )
        out["verdict"], note = verdict_from_delta_max(
            d_max, n_rep, out.get("recommended_n_states")
        )
        note += f"（最差边 {d_max_i}↔{d_max_i + 1}，接受率 {worst_p:.3f}）"
        if worst_p > ACC_ILL_CONDITIONED_ABOVE:
            note += (
                f"  ⚠️ 最差边接受率 > {ACC_ILL_CONDITIONED_ABOVE}，"
                "该区间 δ 反解病态，recommended_n_states 只作量级参考"
            )
        if out["verdict"] == "TOO_SPARSE":
            note += "  → 应加中间态，而不是加采样时间"
        # 🔑 有边完全没被测量过时，判定必须降级为 UNKNOWN。审计工具最不能做的事
        # 就是在存在未测量的边时报「达标」——那条边可能正是最差的一条。
        if out["edges_without_attempts"]:
            out["verdict"] = "UNKNOWN"
            note = (
                f"边 {out['edges_without_attempts']} 一次交换判定都没成功执行过"
                "（探针能量非有限），这些边的 δ 未知，无法判定整条腿。"
                f"已测量的边里：{note}"
            )
        out["note"] = note
        return out

    # --- 降级路径：旧运行只有逐轮口径 ---
    mean_acc = payload.get("mean_acceptance")
    out["degraded_reason"] = (
        "这份诊断没有 edge_acceptance 字段（2026-09-03 之前的运行）。"
        "只有逐轮平均可用，一条稀边会被平均稀释掉，所以下面只能判定"
        "「整体是否超采样」，判不出「哪条边不够」。"
    )
    if mean_acc is None:
        out["verdict"] = "UNKNOWN"
        return out
    d_mean = delta_from_acceptance(mean_acc)
    out["delta_mean"] = d_mean
    if n_rep and n_rep > 1:
        out["total_thermodynamic_length"] = d_mean * (int(n_rep) - 1)
        out["recommended_n_states"] = recommend_n(
            out["total_thermodynamic_length"], delta_target
        )
    # ⚠️ 降级路径只有 δ_mean，没有 δ_max。这里刻意把 δ_mean 喂给同一个判据，
    # 但结论一律附上"这是平均值、稀边可能被掩盖"的限定——δ_mean 只会**低估**
    # δ_max，所以 OVERSAMPLED 结论是稳的，而 OK/MARGINAL 结论并不可靠。
    out["verdict"], note = verdict_from_delta_max(
        d_mean, n_rep, out.get("recommended_n_states")
    )
    note = note.replace("δ_max=", "δ_mean=")
    note += f"（由逐轮平均接受率 {mean_acc:.3f} 反解）"
    if out["verdict"] in ("OK", "MARGINAL"):
        note += "  ⚠️ 这是平均值，可能有被掩盖的稀边；此结论不可靠"
    if mean_acc > ACC_ILL_CONDITIONED_ABOVE:
        note += f"  ⚠️ 接受率 > {ACC_ILL_CONDITIONED_ABOVE}，δ 反解病态，只作定性结论"
    out["note"] = note
    return out


# ---------------------------------------------------------------------------
# 预平衡（顺带：这才是实际平衡长度）
# ---------------------------------------------------------------------------
def audit_equilibration(leg_dir: str) -> Optional[Dict[str, Any]]:
    payload = _load(os.path.join(leg_dir, "checkpoints", "pipeline_state.json"))
    if payload is None:
        return None
    equil = (payload.get("stages") or {}).get("equilibration") or {}
    if not equil:
        return None
    return {
        "status": equil.get("status"),
        "total_steps": equil.get("total_steps"),
        "requested_steps": equil.get("requested_steps"),
        "stopped_early": equil.get("convergence_stopped_early"),
        "stopped_at_step": equil.get("convergence_stopped_at_step"),
    }


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
_VERDICT_MARK = {
    "OK": "✅",
    "MARGINAL": "⚠️ ",
    "TOO_SPARSE": "❌",
    "OVERSAMPLED": "🔵",
    "UNKNOWN": "❔",
}


def _fmt(value: Any, spec: str = ".3f") -> str:
    if value is None:
        return "—"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def audit_leg(leg_dir: str, label: str, delta_target: float) -> Dict[str, Any]:
    print(f"\n  ── {label} ──  {leg_dir}")
    result: Dict[str, Any] = {"leg": label, "dir": leg_dir}

    equil = audit_equilibration(leg_dir)
    result["equilibration"] = equil
    if equil:
        early = equil.get("stopped_early")
        tail = ""
        if early:
            tail = f"（收敛早停于 {equil.get('stopped_at_step')}）"
        print(
            f"     预平衡: status={equil.get('status')} "
            f"实际 {equil.get('total_steps')} / 目标 {equil.get('requested_steps')} 步{tail}"
        )
        print("             ↑ 这才是实际平衡长度；别用配置里的 n_equil_steps")

    vdw = audit_vdw(leg_dir, delta_target)
    result["vdw"] = vdw
    if vdw is None:
        print("     vdW 腿: 没有 preopt_dual_vanishing.json，跳过")
    else:
        mark = _VERDICT_MARK.get(vdw["verdict"], "?")
        print(
            f"     vdW 腿: {mark} {vdw['verdict']}  "
            f"N={vdw['n_states']}  L={_fmt(vdw['total_thermodynamic_length'])}  "
            f"δ_max={_fmt(vdw['delta_max'])}  δ_min={_fmt(vdw['delta_min'])}"
        )
        if vdw.get("densify_points"):
            print(
                f"             布点: {vdw['base_state_count']} 度规 + "
                f"{vdw['densify_points']} 自由能加密 = {vdw['n_states']} 态"
            )
        if vdw.get("window_state_counts"):
            print(
                f"             分窗: {vdw['window_state_counts']}  "
                f"槽位 {vdw.get('total_window_state_slots')}"
            )
        if vdw.get("max_edge_free_energy_kJ_mol") is not None:
            print(
                f"             最大边 ΔF = "
                f"{_fmt(vdw['max_edge_free_energy_kJ_mol'], '.1f')} kJ/mol"
                + (
                    f"  逐窗 ΔF = "
                    f"{[round(x, 1) for x in vdw['subdomain_free_energy_kJ_mol']]}"
                    if vdw.get("subdomain_free_energy_kJ_mol")
                    else ""
                )
            )
        print(
            f"             建议 N（δ_target={delta_target}）= "
            f"{vdw.get('recommended_n_states')}   |  {vdw['note']}"
        )

    coul = audit_decharging(leg_dir, delta_target)
    result["decharging"] = coul
    if coul is None:
        print("     去电荷腿: 没有 decharging_exchange_diagnostics.json，跳过")
    else:
        mark = _VERDICT_MARK.get(coul["verdict"], "?")
        print(
            f"     去电荷腿: {mark} {coul['verdict']}  N={coul['n_replicas']}  "
            f"L_coul={_fmt(coul.get('total_thermodynamic_length'))}  "
            f"建议 N={coul.get('recommended_n_states')}"
        )
        if coul["per_edge_available"]:
            print(
                "             逐边接受率: "
                + " ".join(
                    "--" if p is None else f"{p:.2f}"
                    for p in (coul.get("edge_acceptance") or [])
                )
            )
            print(
                "             逐边 δ:      "
                + " ".join(
                    "--" if d is None else f"{d:.2f}"
                    for d in (coul.get("edge_delta") or [])
                )
            )
        else:
            print(f"             ⚠️ 降级: {coul['degraded_reason']}")
        if coul.get("edges_without_attempts"):
            print(
                "             ❗ 这些边一次判定都没执行过: "
                f"{coul['edges_without_attempts']}"
            )
        print(f"             {coul['note']}")

    return result


def audit_run(run_dir: str, delta_target: float) -> Dict[str, Any]:
    print(f"\n{'=' * 78}\n运行目录: {run_dir}\n{'=' * 78}")
    legs = [(run_dir, "复合物腿")]
    solvent = os.path.join(run_dir, "solvent_leg")
    if os.path.isdir(solvent):
        legs.append((solvent, "溶剂腿"))
    return {
        "run_dir": run_dir,
        "delta_target": delta_target,
        "legs": [audit_leg(d, label, delta_target) for d, label in legs],
    }


def _collect_verdicts(report: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    rows = []
    for leg in report["legs"]:
        for stage in ("vdw", "decharging"):
            info = leg.get(stage)
            if info and info.get("verdict"):
                rows.append((leg["leg"], stage, info["verdict"]))
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="λ 调度事后审计（只读已落盘诊断，零 GPU）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_dirs", nargs="+", help="--output 指向的运行目录")
    parser.add_argument(
        "--delta-target",
        type=float,
        default=DELTA_TARGET_DEFAULT,
        help=f"目标每边热力学长度（默认 {DELTA_TARGET_DEFAULT}，即相邻态 σ(ΔU)≈1 kT）",
    )
    parser.add_argument("--json", action="store_true", help="额外输出机器可读 JSON")
    args = parser.parse_args(argv)

    reports = []
    for run_dir in args.run_dirs:
        if not os.path.isdir(run_dir):
            print(f"❌ 不是目录: {run_dir}")
            return 2
        reports.append(audit_run(run_dir, args.delta_target))

    print(f"\n{'=' * 78}\n汇总\n{'=' * 78}")
    failed = 0
    for report in reports:
        for leg, stage, verdict in _collect_verdicts(report):
            mark = _VERDICT_MARK.get(verdict, "?")
            name = "vdW" if stage == "vdw" else "去电荷"
            print(f"  {mark} {os.path.basename(report['run_dir']):<34} {leg:<8} {name:<6} {verdict}")
            # UNKNOWN 也算未达标：判不出来不等于通过
            if verdict in ("TOO_SPARSE", "MARGINAL", "UNKNOWN"):
                failed += 1

    if args.json:
        print("\n--- JSON ---")
        print(json.dumps(reports, ensure_ascii=False, indent=2, default=str))

    if failed:
        print(
            f"\n❌ {failed} 个腿/阶段未达标（TOO_SPARSE / MARGINAL / UNKNOWN）。"
            "跨体系对比必须标注这些体系跑在不同的采样密度上；UNKNOWN 表示"
            "落盘诊断不足以判定，不能当成通过。"
        )
        return 1
    print("\n✅ 全部达标（OK 或 OVERSAMPLED；OVERSAMPLED 不影响正确性，只是可以更省）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
