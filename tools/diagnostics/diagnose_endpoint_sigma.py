#!/usr/bin/env python
"""endpoint_σ 可信度诊断：复现最终解 + 对同一 λ 区间做重复测量对照。

背景
----
2026-07-27 的 vdW 腿最终报 ``ΔG_vdw = 145.90847 ± 1.38444 kJ/mol``，但：

1. **这个数没有任何审计痕迹。** rescue 分支在 ``abfe_pipeline.py`` 里直接调
   ``solve_stage_integrated``，绕过 ``_run_ibs_stage``，从不填
   ``stage2["diagnostics"]``；``_build_stage_cache_payload`` 存的是
   ``result.get("diagnostics", {})`` → 落盘一个空字典。逐段 ΔG、逐窗 σ、
   ``converged``、ESS、乃至"发生过 rescue"这件事全部没有记录，
   ``pipeline.log`` 在 11:48–12:12 之间也是空的。

2. **误差棒和实测漂移差一个数量级。** 窗口 3 走了拆窗 rescue：
   ``[11,16)`` → ``(11,14)`` + ``(13,16)``，原 w3 已投入的 1M 步被
   ``excluded_local_windows`` 整个排除。总量从 141.65 变成 145.91（+4.26），
   而扣掉未变的 w0/w1/w4/w5 后 w2+w3 的新误差棒只有约 0.46。

关键在于：两个 rescue ensemble 合起来覆盖 λ 索引 11–15，与原 w3 是**完全相同的
物理区间**。所以磁盘上已经躺着一次免费的独立重复测量——这是唯一能直接证伪
"误差棒可信"的证据，不需要任何新采样。

本脚本做两件事
--------------
A1  照抄生产调用复现最终解。必须对上 145.90847 / 1.38444，否则说明对生产路径的
    理解有误，A2 的结论一概不能信。复现成功即免费拿回那份从未落盘的逐段诊断。

A2  对 λ 区间 11→15 的两个独立估计（老 w3 vs 两个 rescue ensemble）算 z 值。

附带：扫 base/bias 时间序列的跳变，判断 P1-13（灾难回退不截断三份 history，
被丢弃分支与重启分支被当成一条连续轨迹估自相关）是不是真的发生过。

安全约束
--------
只读。溶剂腿可能正在跑，**绝不写入 ``--run-dir``**；所有输出到 ``--out-dir``。
不重采样、不建 Context、不碰 GPU。

用法
----
    source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh
    mamba activate openmm_dev
    python tools/diagnostics/diagnose_endpoint_sigma.py --run-dir output --out-dir /tmp/sigma_diag
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 生产口径的常量：kt = R*T。与 abfe_pipeline 里
# (unit.MOLAR_GAS_CONSTANT_R * self.temperature).value_in_unit(kilojoule_per_mole) 一致。
_R_KJ_PER_MOL_K = 8.31446261815324e-3

# 生产默认门槛，必须与 abfe_pipeline.py 里 kwargs.get(...) 的默认值逐一相同，
# 否则复现出来的 converged/各门实际值都不是那次运行的口径。
FINAL_GATES = dict(
    final_min_ess_ratio=0.05,
    final_min_absolute_ess=50.0,
    final_min_decorrelated_samples=20,
    final_max_uncertainty_kJ_mol=1.0,
)

# 生产报出的值，A1 用它做控制实验。
EXPECTED_TOTAL_DG = 145.90847168207642
EXPECTED_TOTAL_ERR = 1.3844433223361403


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def _to_native(obj):
    """递归转成 JSON 可序列化的原生类型。

    诊断 dict 里混着 numpy 标量与数组（f_k、window_overlap_diagnostics 等），
    直接 json.dump 会 TypeError。
    """
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _fmt(value, spec=".4f", none="—"):
    if value is None:
        return none
    try:
        if not math.isfinite(float(value)):
            return none
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _banner(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 载入
# ---------------------------------------------------------------------------


def load_lambda_path(run_dir: str) -> Tuple[List[float], List[Tuple[int, int]]]:
    """从预优化缓存读 λ 表与窗口划分（v21，23 个唯一 λ）。"""
    path = os.path.join(run_dir, "checkpoints", "preopt_dual_vanishing.json")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    lambdas_vdw = [float(x) for x in payload["lambdas_var"]]
    ranges = [(int(a), int(b)) for a, b in payload["window_ranges"]]
    return lambdas_vdw, ranges


def discover_rescue_plan(
    run_dir: str, lambdas_vdw: Sequence[float]
) -> Optional[Tuple[str, str, List[Tuple[int, int]]]]:
    """找 vanishing_rescue 的 plan 目录，并还原每个 rescue ensemble 的全局 λ 区间。

    ⚠️ rescue 的 ``*_convergence.json`` 里**没有** ``window_range`` /
    ``global_lambda_indices``（实测只有 ``lambdas_coul`` / ``lambdas_vdw``），
    所以只能拿 λ 值回查全局 23 点表。这是精确的：两边的浮点数来自同一个数组，
    逐位相同。仍然按精确匹配优先、极小容差兜底来做，匹配不上就 fail closed，
    绝不猜。
    """
    base = os.path.join(run_dir, "vanishing_rescue")
    if not os.path.isdir(base):
        return None
    plans = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if not plans:
        return None
    if len(plans) > 1:
        plans.sort(key=lambda d: os.path.getmtime(os.path.join(base, d)))
        print(f"  ⚠️ 发现多个 rescue plan {plans}，取 mtime 最新的 {plans[-1]}")
    plan_id = plans[-1]
    output_dir = os.path.join(base, plan_id)
    checkpoint_dir = os.path.join(run_dir, "checkpoints", "vanishing_rescue", plan_id)

    grid = [float(x) for x in lambdas_vdw]

    def _global_index(value: float) -> int:
        for i, g in enumerate(grid):
            if g == value:
                return i
        # λ 网格是不可变的；rescue 只允许复用现有节点。容差只为吸收 JSON
        # round-trip，不是为了容忍"新节点"。
        diffs = [abs(g - value) for g in grid]
        best = int(np.argmin(diffs))
        if diffs[best] > 1e-12:
            raise RuntimeError(
                f"rescue 用到的 λ={value!r} 不在全局 23 点表里（最近的是 "
                f"{grid[best]!r}，差 {diffs[best]:.3e}）。rescue 只允许复用现有 λ 节点，"
                "这说明落盘数据与预优化缓存对不上，拒绝继续。"
            )
        return best

    ranges: List[Tuple[int, int]] = []
    idx = 0
    while True:
        conv = os.path.join(output_dir, f"dual_window_{idx}_vdw_convergence.json")
        if not os.path.isfile(conv):
            break
        with open(conv, encoding="utf-8") as handle:
            payload = json.load(handle)
        lam_values = payload.get("lambdas_vdw")
        if not lam_values:
            raise RuntimeError(
                f"rescue 窗口 {idx} 的 convergence.json 里没有 lambdas_vdw，无法还原 λ 区间"
            )
        indices = [_global_index(float(v)) for v in lam_values]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise RuntimeError(
                f"rescue 窗口 {idx} 的 λ 索引 {indices} 不连续，与 (start, end) 半开区间"
                "契约不符，拒绝继续"
            )
        ranges.append((indices[0], indices[-1] + 1))
        idx += 1
    if not ranges:
        return None
    return output_dir, checkpoint_dir, ranges


def load_outputs(
    output_dir: str,
    ranges: Sequence[Tuple[int, int]],
    lambdas_vdw: Sequence[float],
    checkpoint_dir: str,
    *,
    excluded: Optional[set] = None,
    offset: int = 0,
    prefix: str = "window",
) -> List[Dict]:
    """复用生产 loader —— 三文件 manifest、f_k、expected-vs-loaded 门全部照旧。"""
    from abfe_pipeline import ABFEPipeline

    return ABFEPipeline._load_ibs_window_outputs_from_dir(
        output_dir,
        list(ranges),
        [0.0] * len(lambdas_vdw),
        list(lambdas_vdw),
        checkpoint_dir=checkpoint_dir,
        excluded_local_windows=excluded,
        window_index_offset=offset,
        window_label_prefix=prefix,
    )


# ---------------------------------------------------------------------------
# A1 复现
# ---------------------------------------------------------------------------


def reproduce_final_solve(run_dir: str, temperature_k: float) -> Dict:
    from ibs_engine import solve_stage_integrated

    lambdas_vdw, base_ranges = load_lambda_path(run_dir)
    kt = _R_KJ_PER_MOL_K * temperature_k
    rescue = discover_rescue_plan(run_dir, lambdas_vdw)

    _banner("A1 — 复现最终解（控制实验）")
    print(f"  λ 路径：{len(lambdas_vdw)} 个唯一节点，窗口 {base_ranges}")
    print(f"  kt = {kt:.6f} kJ/mol  (T = {temperature_k} K)")

    if rescue is None:
        print("  未发现 vanishing_rescue —— 按无 rescue 的普通路径复现")
        outputs = load_outputs(
            os.path.join(run_dir, "vanishing"), base_ranges, lambdas_vdw,
            os.path.join(run_dir, "checkpoints"),
        )
        replaced: List[int] = []
    else:
        rescue_dir, rescue_ckpt, rescue_ranges = rescue
        # 被 rescue 取代的原始窗口：λ 区间被 rescue_ranges 完全覆盖的那些。
        covered = {i for a, b in rescue_ranges for i in range(a, b)}
        replaced = [
            w for w, (a, b) in enumerate(base_ranges)
            if set(range(a, b)) <= covered
        ]
        print(f"  rescue plan：{os.path.basename(rescue_dir)}  ranges={rescue_ranges}")
        print(f"  被取代的原始窗口：{replaced}")
        original = load_outputs(
            os.path.join(run_dir, "vanishing"), base_ranges, lambdas_vdw,
            os.path.join(run_dir, "checkpoints"),
            excluded=set(replaced), prefix="original_window",
        )
        rescue_outputs = load_outputs(
            rescue_dir, rescue_ranges, lambdas_vdw, rescue_ckpt,
            offset=10_000, prefix="rescue_window",
        )
        outputs = original + rescue_outputs

    result = solve_stage_integrated(
        window_outputs=outputs, kt=kt, stage_name="vanishing", **FINAL_GATES
    )

    dg = float(result.get("total_delta_G", float("nan")))
    err = float(result.get("total_error", float("nan")))
    d_dg = abs(dg - EXPECTED_TOTAL_DG)
    d_err = abs(err - EXPECTED_TOTAL_ERR)
    ok = d_dg < 1e-6 and d_err < 1e-6

    print()
    print(f"  复现 total_delta_G = {dg:.8f}   (生产 {EXPECTED_TOTAL_DG:.8f}, Δ={d_dg:.2e})")
    print(f"  复现 total_error   = {err:.8f}   (生产 {EXPECTED_TOTAL_ERR:.8f}, Δ={d_err:.2e})")
    print(f"  → {'✅ 复现成功' if ok else '❌ 未复现——A2 的结论不可信，先查清楚再往下走'}")

    _print_solve_report(result)
    result["_reproduced"] = bool(ok)
    result["_replaced_windows"] = replaced
    return result


def _print_solve_report(result: Dict) -> None:
    print()
    print("  ---- 门 ----")
    for key, thr_key in [
        ("min_overlap", "min_overlap_threshold"),
        ("min_occupancy_normalized", "min_occupancy_normalized_threshold"),
        ("min_decorrelated_samples", "min_decorrelated_samples_threshold"),
        ("max_endpoint_uncertainty_kJ_mol", "max_endpoint_uncertainty_kJ_mol_threshold"),
        ("min_absolute_ess", "min_absolute_ess_threshold"),
    ]:
        print(f"    {key:42s} = {_fmt(result.get(key)):>10s}   阈值 {_fmt(result.get(thr_key)):>8s}")
    print(f"    {'converged':42s} = {result.get('converged')}")

    cov = result.get("coverage_diagnostics") or {}
    if cov:
        print()
        print("  ---- 覆盖 ----")
        print(f"    输入窗口 {cov.get('input_window_indices')}")
        print(f"    求解窗口 {cov.get('solved_window_indices')}   丢弃 {cov.get('dropped_window_indices')}")
        print(f"    覆盖 λ 索引 {cov.get('covered_lambda_index_first')}..{cov.get('covered_lambda_index_last')}"
              f"（共 {cov.get('n_covered_lambda_indices')} 个）")

    segments = result.get("covariance_chain_segments") or []
    if segments:
        print()
        print("  ---- 逐段（这份此前从未落盘）----")
        print(f"    {'src':>6} {'join':>5} {'end':>4} {'ΔG (kJ/mol)':>13} {'σ (kJ/mol)':>12}")
        for seg in segments:
            print(f"    {seg.get('source_window_index', seg.get('window_index')):>6} "
                  f"{seg.get('join_lambda_index'):>5} {seg.get('end_lambda_index'):>4} "
                  f"{_fmt(seg.get('delta_G_kJ_mol'), '.4f'):>13} "
                  f"{_fmt(seg.get('uncertainty_kJ_mol'), '.4f'):>12}")

    diags = result.get("window_overlap_diagnostics") or []
    if diags:
        print()
        print("  ---- 逐窗 ----")
        print(f"    {'label':>18} {'ESSr':>8} {'absESS':>9} {'Ndec':>6} {'occ':>8} {'endσ':>8}")
        for rec in diags:
            print(f"    {str(rec.get('window_label')):>18} "
                  f"{_fmt(rec.get('min_ess_ratio')):>8} "
                  f"{_fmt(rec.get('absolute_ess'), '.2f'):>9} "
                  f"{str(rec.get('n_frames_decorrelated')):>6} "
                  f"{_fmt(rec.get('min_occupancy_normalized')):>8} "
                  f"{_fmt(rec.get('endpoint_diff_uncertainty_kJ_mol')):>8}")
        _print_asymptotic_validity(diags)


def _print_asymptotic_validity(diags: Sequence[Dict]) -> None:
    """报出去的 σ 背后到底有几个独立样本。

    整条路径只有 pymbar 的**渐近** svd-ew 协方差（`uncertainty_method` 从不传，
    `mbar.py` 里 None → "svd-ew"），而 pymbar 自己的 docstring 就写着：
    "This will break down in cases where the number of samples is not large
    enough to reach the asymptotic normal limit."

    更要命的是 `n_k = [N, 0, ..., 0]`——K 个目标态的整个协方差矩阵都由同一批
    N 个样本估出来，而某个态真正有效的样本数是 `ESS_ratio × N`（即 absolute_ess）。
    absolute_ess ≈ 1 却报出 0.06 kJ/mol 的 σ，这个 σ 没有任何统计意义。
    """
    print()
    print("  ---- 渐近协方差还成不成立（absolute_ess 就是该窗真正的独立样本数）----")
    suspicious = []
    for rec in diags:
        abs_ess = rec.get("absolute_ess")
        sigma = rec.get("endpoint_diff_uncertainty_kJ_mol")
        if abs_ess is None:
            continue
        flag = ""
        try:
            if float(abs_ess) < 10.0:
                flag = "  ⚠️ 渐近极限不成立"
                suspicious.append((rec.get("window_label"), float(abs_ess), sigma))
        except (TypeError, ValueError):
            pass
        print(f"    {str(rec.get('window_label')):>18}  absolute_ess = {_fmt(abs_ess, '.2f'):>8}"
              f"   报出的 σ = {_fmt(sigma):>8} kJ/mol{flag}")
    if suspicious:
        print()
        print(f"    → {len(suspicious)} 个窗口的有效样本数 < 10。这些窗口的 σ 是在渐近极限"
              "之外外推出来的，")
        print("      数值可以任意小而不代表精度。这是误差棒被低估的**机制性**理由，")
        print("      与下面 A2 的实测证据相互独立。")


# ---------------------------------------------------------------------------
# A2 重复测量对照
# ---------------------------------------------------------------------------


def _segment_from_solve(result: Dict, lam_lo: int, lam_hi: int) -> Tuple[Optional[float], Optional[float]]:
    """从一次完整求解里取 λ_lo→λ_hi 的 ΔG 与 σ（按覆盖该区间的链段求和）。"""
    segments = result.get("covariance_chain_segments") or []
    total = 0.0
    var = 0.0
    hit = False
    for seg in segments:
        join = int(seg["join_lambda_index"])
        end = int(seg["end_lambda_index"])
        if join >= lam_lo and end <= lam_hi:
            total += float(seg["delta_G_kJ_mol"])
            var += float(seg["uncertainty_kJ_mol"]) ** 2
            hit = True
    if not hit:
        return None, None
    return total, math.sqrt(var)


def repeat_measurement_check(run_dir: str, temperature_k: float) -> Optional[Dict]:
    from ibs_engine import solve_stage_integrated

    lambdas_vdw, base_ranges = load_lambda_path(run_dir)
    rescue = discover_rescue_plan(run_dir, lambdas_vdw)
    if rescue is None:
        print("\n  未发现 vanishing_rescue —— 没有重复测量可对照，跳过 A2")
        return None

    rescue_dir, rescue_ckpt, rescue_ranges = rescue
    kt = _R_KJ_PER_MOL_K * temperature_k

    covered = {i for a, b in rescue_ranges for i in range(a, b)}
    replaced = [
        w for w, (a, b) in enumerate(base_ranges) if set(range(a, b)) <= covered
    ]
    if len(replaced) != 1:
        print(f"\n  ⚠️ 被取代的原始窗口不是恰好一个（{replaced}），A2 只处理单窗情形")
        return None
    w_old = replaced[0]
    lam_lo, lam_hi = base_ranges[w_old]
    lam_hi -= 1  # 半开 → 闭区间端点

    _banner(f"A2 — 同一 λ 区间 {lam_lo}→{lam_hi} 的重复测量对照")
    print(f"  老 window {w_old}：range={base_ranges[w_old]}（1 个 ensemble，生产中已被排除）")
    print(f"  新 rescue     ：ranges={rescue_ranges}（{len(rescue_ranges)} 个 ensemble）")

    # ---- 口径 1/2：各自单独求解 ----
    old_out = load_outputs(
        os.path.join(run_dir, "vanishing"),
        base_ranges, lambdas_vdw, os.path.join(run_dir, "checkpoints"),
        excluded={i for i in range(len(base_ranges)) if i != w_old},
        prefix="old_window",
    )
    old_solo = solve_stage_integrated(
        window_outputs=old_out, kt=kt, stage_name="old_w3_only", **FINAL_GATES
    )
    new_out = load_outputs(
        rescue_dir, rescue_ranges, lambdas_vdw, rescue_ckpt,
        offset=10_000, prefix="rescue_window",
    )
    new_solo = solve_stage_integrated(
        window_outputs=new_out, kt=kt, stage_name="rescue_only", **FINAL_GATES
    )

    dg_old_solo = float(old_solo.get("total_delta_G", float("nan")))
    sd_old_solo = float(old_solo.get("total_error", float("nan")))
    dg_new_solo = float(new_solo.get("total_delta_G", float("nan")))
    sd_new_solo = float(new_solo.get("total_error", float("nan")))

    # ---- 口径 3：走全链，两套输入各跑一次（与生产口径完全一致）----
    chain_old = load_outputs(
        os.path.join(run_dir, "vanishing"), base_ranges, lambdas_vdw,
        os.path.join(run_dir, "checkpoints"), prefix="original_window",
    )
    full_old = solve_stage_integrated(
        window_outputs=chain_old, kt=kt, stage_name="full_with_old", **FINAL_GATES
    )
    chain_new = load_outputs(
        os.path.join(run_dir, "vanishing"), base_ranges, lambdas_vdw,
        os.path.join(run_dir, "checkpoints"),
        excluded=set(replaced), prefix="original_window",
    ) + new_out
    full_new = solve_stage_integrated(
        window_outputs=chain_new, kt=kt, stage_name="full_with_rescue", **FINAL_GATES
    )

    seg_old, seg_old_sd = _segment_from_solve(full_old, lam_lo, lam_hi)
    seg_new, seg_new_sd = _segment_from_solve(full_new, lam_lo, lam_hi)

    rows = [
        ("① 各自单独求解", dg_old_solo, sd_old_solo, dg_new_solo, sd_new_solo),
        ("② 全链内该区间段", seg_old, seg_old_sd, seg_new, seg_new_sd),
        ("③ 全链总 ΔG",
         float(full_old.get("total_delta_G", float("nan"))),
         float(full_old.get("total_error", float("nan"))),
         float(full_new.get("total_delta_G", float("nan"))),
         float(full_new.get("total_error", float("nan")))),
    ]

    print()
    print(f"  {'口径':>18} {'老 ΔG':>11} {'老 σ':>8} {'新 ΔG':>11} {'新 σ':>8} {'差值':>9} {'z':>7}")
    out_rows = []
    for label, a, sa, b, sb in rows:
        if a is None or b is None:
            print(f"  {label:>18}   （该区间未被链段完整覆盖，跳过）")
            continue
        diff = b - a
        denom = math.sqrt((sa or 0.0) ** 2 + (sb or 0.0) ** 2)
        z = abs(diff) / denom if denom > 0 else float("inf")
        print(f"  {label:>18} {a:>11.4f} {_fmt(sa, '.4f'):>8} {b:>11.4f} "
              f"{_fmt(sb, '.4f'):>8} {diff:>9.4f} {z:>7.2f}")
        out_rows.append({
            "method": label, "old_delta_G": a, "old_sigma": sa,
            "new_delta_G": b, "new_sigma": sb, "difference": diff, "z": z,
        })

    z_primary = next((r["z"] for r in out_rows if r["method"].startswith("③")), None)
    print()
    if z_primary is None:
        print("  ⚠️ 拿不到主口径 z 值")
    elif z_primary < 2.0:
        print(f"  → z = {z_primary:.2f} < 2：两个估计相符，误差棒暂时可信。")
        print("     那么 141.65→145.91 的漂移主要来自 w2 而非 w3，按正常加采样推进即可。")
    else:
        print(f"  → z = {z_primary:.2f} ≫ 2：**误差棒与实测漂移不自洽。**")
        print("     报出去的 σ 不能直接用。结合上面逐窗 occupancy/ESS 判断是哪一边有偏；")
        print("     现有证据倾向老窗口有偏（它的最低占据态低于 floor）。")

    return {
        "lambda_range": [lam_lo, lam_hi],
        "replaced_window": w_old,
        "rescue_ranges": [list(r) for r in rescue_ranges],
        "comparisons": out_rows,
        "primary_z": z_primary,
        "old_solo": _to_native(old_solo),
        "new_solo": _to_native(new_solo),
    }


# ---------------------------------------------------------------------------
# 附带：P1-13 history 跳变扫描
# ---------------------------------------------------------------------------


def scan_history_discontinuities(run_dir: str, mad_factor: float = 12.0) -> Dict:
    """扫时间序列的跳变，判断 P1-13 是否真的发生过、以及有没有污染 g。

    生产灾难回退（ibs_engine.py 的两处 `sim.context.setPositions(pos_backup)`）
    只回退坐标，不截断 energy/bias/base 三份 history；被丢弃分支与重启分支因此
    共享祖先却仍被当作一条连续时间序列做自相关子采样。若真发生过，序列里应能
    看到远超正常热涨落的相邻帧跳变。

    **扫三种序列，缺一不可：**

    - ``bias``、``base``：直接看回退/续跑留下的痕迹。``base`` 是总势能量级，
      被溶剂涨落主导（MAD ~700 kJ/mol），所以它跳不跳只说明"发生过什么"。
    - ``decorr_k``：`(u_kn[k] - bias)/kT`，也就是
      `_decorrelate_by_worst_target_state` **真正拿去估 g 的那条序列**
      （`ibs_engine.py` 里 `series = (u_kj_raw[k] - bias_kj) / float(kt)`）。
      只有这条跳了，`g` 才真的被污染、`N_decorr` 才不可信。base 跳而 decorr
      不跳，说明 U_k 与 base 同步跳、在 `u_kn = U_k - base` 里抵消掉了——
      那么 P1-13 确实发生过，但没有伤到误差棒。这个区分是本函数的重点。
    """
    _banner("附带 — P1-13：回退/续跑的跳变有没有污染 g")
    from ibs_engine import _load_validated_window_data_triplet

    findings = {}
    scopes = [("vanishing", os.path.join(run_dir, "vanishing"))]
    lambdas_vdw, _ = load_lambda_path(run_dir)
    rescue = discover_rescue_plan(run_dir, lambdas_vdw)
    if rescue is not None:
        scopes.append(("rescue", rescue[0]))
    for scope, out_dir in scopes:
        idx = 0
        while True:
            stem = os.path.join(out_dir, f"dual_window_{idx}_vdw")
            conv_path = f"{stem}_convergence.json"
            if not os.path.isfile(conv_path):
                break
            with open(conv_path, encoding="utf-8") as handle:
                convergence = json.load(handle)
            u_kn, bias, base = _load_validated_window_data_triplet(
                f"{stem}_energies.npy", f"{stem}_bias.npy", f"{stem}_base.npy", convergence
            )

            def _count(series) -> Dict:
                d = np.diff(np.asarray(series, dtype=float))
                if d.size == 0:
                    return {"jumps": 0, "mad": None, "max_abs_diff": None}
                mad = float(np.median(np.abs(d - np.median(d))))
                if mad <= 0.0:
                    return {"jumps": 0, "mad": 0.0, "max_abs_diff": float(np.max(np.abs(d)))}
                hits = np.flatnonzero(np.abs(d - np.median(d)) > mad_factor * mad)
                return {
                    "jumps": int(hits.size),
                    "mad": mad,
                    "max_abs_diff": float(np.max(np.abs(d))),
                    # 跳变**位置**：用来区分成因。落在 500/1000 帧整数倍附近 =
                    # 跨进程续跑/窗口重建边界；散落在中间 = 别的东西。
                    "jump_frame_indices": [int(i) + 1 for i in hits[:50]],
                }

            entry = {"n_frames": int(np.asarray(base).size)}
            for name, series in (("bias", bias), ("base", base)):
                stats = _count(series)
                entry[f"{name}_jumps"] = stats["jumps"]
                entry[f"{name}_mad"] = stats["mad"]
                entry[f"{name}_max_abs_diff"] = stats["max_abs_diff"]
                entry[f"{name}_jump_frame_indices"] = stats.get("jump_frame_indices", [])

            # 真正决定 g 的那条序列，逐个目标态扫，取最坏的一个。
            # 与 ibs_engine._decorrelate_by_worst_target_state 里的
            #     series = (u_kj_raw[k] - bias_kj) / float(kt)
            # 逐字一致。
            kt = _R_KJ_PER_MOL_K * 300.0
            u_arr = np.asarray(u_kn, dtype=float)
            bias_arr = np.asarray(bias, dtype=float).ravel()
            worst = {"jumps": 0, "mad": None, "max_abs_diff": None, "state": None}
            for k in range(u_arr.shape[0]):
                stats = _count((u_arr[k] - bias_arr) / kt)
                if stats["jumps"] > worst["jumps"] or worst["state"] is None:
                    worst = dict(stats, state=k)
            entry["decorr_worst_state"] = worst["state"]
            entry["decorr_jumps"] = worst["jumps"]
            entry["decorr_mad_kT"] = worst["mad"]
            entry["decorr_max_abs_diff_kT"] = worst["max_abs_diff"]

            findings[f"{scope}_w{idx}"] = entry
            idx += 1

    base_bias_total = sum(
        v.get("bias_jumps", 0) + v.get("base_jumps", 0) for v in findings.values()
    )
    decorr_total = sum(v.get("decorr_jumps", 0) for v in findings.values())

    print(f"  阈值：相邻帧差偏离中位数 > {mad_factor}×MAD")
    print(f"  {'window':>14} {'bias':>6} {'base':>6} {'decorr':>7} "
          f"{'base MAD':>10} {'base maxΔ':>11} {'decorr MAD(kT)':>15} {'decorr maxΔ(kT)':>16}")
    for key, val in findings.items():
        print(f"  {key:>14} {val.get('bias_jumps', 0):>6} {val.get('base_jumps', 0):>6} "
              f"{val.get('decorr_jumps', 0):>7} "
              f"{_fmt(val.get('base_mad'), '.1f'):>10} {_fmt(val.get('base_max_abs_diff'), '.1f'):>11} "
              f"{_fmt(val.get('decorr_mad_kT'), '.4f'):>15} "
              f"{_fmt(val.get('decorr_max_abs_diff_kT'), '.4f'):>16}")

    print()
    if base_bias_total == 0:
        print("  → base/bias 一处跳变都没有：这批数据没触发过回退或跨进程续跑，")
        print("     P1-13 未在此发生，不是当前 σ 问题的成因。")
    elif decorr_total == 0:
        print(f"  → base 序列有 {base_bias_total} 处不连续，但真正拿去估 g 的序列"
              " (u_kn[k]-bias)/kT **一处都没跳**。")
        print("     说明 U_k 与 base 是同步跳的，在 u_kn = U_k − base 里抵消掉了：")
        print("     **g / N_decorr / 误差棒没有被污染**，不能拿这些跳变解释 σ 的行为。")
        print()
        print("     ⚠️ 另外注意：这些跳变**未必是 P1-13**。P1-13 说的是生产灾难回退")
        print("     （`sim.context.setPositions(pos_backup)` 那条路径）不截断 history；")
        print("     那条路径触发时会打印“灾难检测触发”。请对照日志：")
        print("       grep -c '触发回退\\|灾难检测触发' <run-dir>/pipeline.log")
        print("     若为 0，则这些不连续来自跨进程续跑/窗口重建边界，是另一回事——")
        print("     修 P1-13 并不会消除它们。上面 json 里的 base_jump_frame_indices")
        print("     可以定位：落在帧数整数倍附近即为会话边界。")
    else:
        print(f"  → base/bias {base_bias_total} 处、**decorr 序列 {decorr_total} 处**跳变：")
        print("     P1-13 不仅发生了，而且污染了自相关估计——g 与 N_decorr 不可信，")
        print("     依赖它们的 min_decorrelated_samples 门和误差棒都要重新审视。")

    return {
        "mad_factor": mad_factor,
        "per_window": findings,
        "base_bias_jumps": base_bias_total,
        "decorrelation_series_jumps": decorr_total,
        # 兼容旧字段名
        "total_jumps": base_bias_total,
    }


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, help="生产输出目录（只读）")
    parser.add_argument("--out-dir", default=None, help="诊断结果写到哪里（默认 <run-dir>_sigma_diag，绝不写 run-dir）")
    parser.add_argument("--temperature", type=float, default=300.0, help="温度 K（默认 300）")
    parser.add_argument("--mad-factor", type=float, default=12.0, help="history 跳变判据的 MAD 倍数")
    parser.add_argument("--skip-history-scan", action="store_true")
    args = parser.parse_args(argv)

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"❌ 找不到 run-dir: {run_dir}", file=sys.stderr)
        return 2
    out_dir = os.path.abspath(args.out_dir or (run_dir.rstrip("/") + "_sigma_diag"))
    if os.path.commonpath([out_dir, run_dir]) == run_dir:
        print(f"❌ out-dir 不能落在 run-dir 里面（溶剂腿可能正在跑）: {out_dir}", file=sys.stderr)
        return 2
    os.makedirs(out_dir, exist_ok=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    report = {"run_dir": run_dir, "temperature_K": args.temperature}
    final = reproduce_final_solve(run_dir, args.temperature)
    report["A1_reproduce"] = _to_native(final)

    if not final.get("_reproduced"):
        print()
        print("❌ A1 未复现生产结果——按计划在此停下。")
        print("   在没复现的基础上做 A2 对照没有意义：对不上说明输入组合或门槛与生产不一致，")
        print("   先查清楚是哪一处，再重跑本脚本。")
    else:
        report["A2_repeat_measurement"] = repeat_measurement_check(run_dir, args.temperature)

    if not args.skip_history_scan:
        report["history_scan"] = scan_history_discontinuities(run_dir, args.mad_factor)

    out_path = os.path.join(out_dir, "endpoint_sigma_diagnosis.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(_to_native(report), handle, indent=2, ensure_ascii=False)
    print()
    print(f"📄 完整结果已写入 {out_path}")
    return 0 if final.get("_reproduced") else 1


if __name__ == "__main__":
    raise SystemExit(main())
