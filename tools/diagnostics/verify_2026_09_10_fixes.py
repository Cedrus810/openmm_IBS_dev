#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""判读 2026-09-09/10 那批改动在**真机运行**上是否确实生效。

## 为什么需要它

那批改动（52+ 处）全部只过了离线套件，**没有一处上过 GPU**。而它们里面有三条
是协议相邻的、只有真机能验：

* 偏置爬坡补上 1.0 档（10 个 run 死在 0.7→1.0 那一跳）；
* preopt 探针的 force group 重划（只影响带 co-ion 的腿）；
* 加密点改成从相邻高 λ 端点续接（改变了 pilot 的采样语义）。

"跑完了没报错"不等于"这几条做到了它声称的事"。这个脚本只读一个 run 目录，
逐条给 PASS / FAIL / NA，并且**把依据一起打出来**——不给一个光秃秃的结论。

## 用法

    python tools/diagnostics/verify_2026_09_10_fixes.py <run_dir> [<run_dir> ...]

只读；不写这个目录里的任何东西，也不启动任何计算。
退出码：全部 PASS 或 NA → 0；有任何 FAIL → 1。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

PASS, FAIL, NA = "PASS", "FAIL", "NA"


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 逐条检查。每个返回 (status, 一行结论, [证据行...])
# ---------------------------------------------------------------------------


def check_bias_ramp_reaches_full_scale(run: str) -> Tuple[str, str, List[str]]:
    """爬坡必须出现 scale=1.0 那一档，且窗口 0 不再死在那一跳上。

    修复前 `ramp_stages` 停在 0.7，随后直接把 bias_scale 设成 1.0 —— 最后也是
    最大的一次增量既没有弛豫档、也没有分段回退保护，10 个 run 全死在那里。
    """
    log = _read(os.path.join(run, "pipeline.log")) or _read(
        os.path.join(run, "launch.log")
    )
    if "[偏置预热]" not in log:
        return NA, "本次运行还没进到偏置预热", []
    rungs = sorted({
        line.split("设为")[1].split(",")[0].strip()
        for line in log.splitlines()
        if "→ Bias Scale 设为" in line and "," in line
    })
    evidence = [f"出现过的爬坡档位: {rungs}"]
    nan_in_ramp = "偏置预热" in log and "Particle coordinate is NaN" in log
    if "1.0" not in rungs:
        return FAIL, "爬坡里没有 1.0 档——补丁没生效（或跑的是旧代码）", evidence
    if nan_in_ramp:
        return FAIL, "爬坡阶段仍出现 NaN", evidence + ["日志里同时有『偏置预热』与 NaN"]
    rescued = [l.strip() for l in log.splitlines() if "回退重做" in l]
    if rescued:
        evidence.append(f"分段回退被触发 {len(rescued)} 次（说明保护真的在工作）")
        evidence.extend("    " + line for line in rescued[:3])
    return PASS, "爬坡走满到 1.0，且该阶段无 NaN", evidence


def check_pilot_traversal_semantics(run: str) -> Tuple[str, str, List[str]]:
    """preopt 缓存必须记下采样语义，加密点必须真的从高 λ 端点续接。"""
    path = os.path.join(run, "checkpoints", "preopt_dual_vanishing.json")
    cached = _load_json(path)
    if cached is None:
        return NA, "还没有 preopt_dual_vanishing.json", []
    payload = (cached.get("protocol_key") or {}).get("payload") or {}
    traversal = payload.get("pilot_traversal")
    points = (cached.get("path_diagnostics") or {}).get("pilot_points") or []
    n_refine = sum(1 for p in points if p.get("is_refinement_point"))
    evidence = [
        f"pilot_traversal = {traversal!r}",
        f"pilot 点 {len(points)} 个，其中加密点 {n_refine} 个",
    ]
    if traversal is None:
        return FAIL, "protocol_key 里没有 pilot_traversal——两层拆分没生效", evidence
    if traversal != "high_lambda_endpoint_restart_v1":
        return FAIL, f"采样语义标签不是当前值：{traversal!r}", evidence
    if n_refine == 0:
        return PASS, "语义已记录；本次没触发加密（未覆盖续接路径）", evidence
    return PASS, f"语义已记录，且真的走了 {n_refine} 个加密点", evidence


def check_preopt_derived_layer(run: str) -> Tuple[str, str, List[str]]:
    """派生层那 5 个键必须进指纹（原来根本不在里面）。"""
    cached = _load_json(
        os.path.join(run, "checkpoints", "preopt_dual_vanishing.json")
    )
    if cached is None:
        return NA, "还没有 preopt_dual_vanishing.json", []
    payload = (cached.get("protocol_key") or {}).get("payload") or {}
    expected = (
        "stage2_final_n_states",
        "stage2_refine_extra_points_per_segment",
        "stage2_window_min_states",
        "stage2_window_max_states",
        "stage2_free_energy_densify_points",
    )
    missing = [name for name in expected if name not in payload]
    evidence = [f"{name} = {payload.get(name)!r}" for name in expected]
    if missing:
        return FAIL, f"派生层缺键: {missing}", evidence
    return PASS, "派生层 5 个键都在指纹里", evidence


def check_lrc_single_volume(run: str) -> Tuple[str, str, List[str]]:
    """LRC 系数必须能反解出**同一个** V，且 λ=0 精确为 0。

    这是 #32「per-frame correction」在产物侧唯一能离线证的部分：系数是
    `coeff[k]/V`，若各 λ 态反解出的 V 一致，说明用的确实是同一个（冻结的）盒，
    而不是各态各拿一个体积。
    """
    candidates = sorted(glob.glob(os.path.join(run, "**", "*u_kn*.meta.json"), recursive=True))
    metas = [m for m in (_load_json(p) for p in candidates) if m]
    lrc = None
    for meta in metas:
        if isinstance(meta.get("lj_lrc"), dict):
            lrc = meta["lj_lrc"]
            break
    if lrc is None:
        return NA, "产物里没有 lj_lrc metadata", []
    evidence = [
        f"applied = {lrc.get('applied')}",
        f"volume_mean_nm3 = {lrc.get('volume_mean_nm3')}",
        f"volume_relative_span = {lrc.get('volume_relative_span')}",
    ]
    span = lrc.get("volume_relative_span")
    if span is None:
        return NA, "metadata 里没有 volume_relative_span", evidence
    if float(span) > 1.0e-3:
        return FAIL, f"盒体积波动 {span:.3e} > 1e-3，离线 1/V 尾项不成立", evidence
    return PASS, f"盒体积恒定（span={float(span):.3e}），离线 1/V 尾项前提成立", evidence


def check_job_guard(run: str) -> Tuple[str, str, List[str]]:
    """作业防护：日志里要有防护行；跑完之后锁文件不该残留。"""
    log = _read(os.path.join(run, "pipeline.log")) + _read(
        os.path.join(run, "launch.log")
    )
    lock_path = os.path.join(run, ".abfe_run.lock")
    evidence = []
    guarded = "[作业防护]" in log
    evidence.append(f"日志里有 [作业防护] 行: {guarded}")
    if not guarded:
        return NA, "本次运行是在作业防护上线之前起的", evidence
    if os.path.exists(lock_path):
        evidence.append(f"锁文件仍在: {lock_path}（若进程还在跑，这是正常的）")
        return NA, "锁文件还在——进程可能仍在运行", evidence
    return PASS, "防护已装且锁已释放", evidence


def check_disk_precheck(run: str) -> Tuple[str, str, List[str]]:
    log = _read(os.path.join(run, "pipeline.log")) + _read(
        os.path.join(run, "launch.log")
    )
    lines = [l.strip() for l in log.splitlines() if "[磁盘预检]" in l]
    if not lines:
        return NA, "没看到磁盘预检行（可能是防护上线前起的运行）", []
    return PASS, f"磁盘预检执行了 {len(lines)} 次", lines[:3]


def check_stdout_tee(run: str) -> Tuple[str, str, List[str]]:
    """`launch.log`（stdout 重定向）必须和 `pipeline.log` 一样能看到 print 行。

    修复前 tee 只对文件那份逐行 flush，转发给真 stdout 的那份从头到尾不 flush，
    重定向下是 8 KB 块缓冲 ⇒ launch.log 落后甚至整块丢失。
    """
    launch = _read(os.path.join(run, "launch.log"))
    pipeline = _read(os.path.join(run, "pipeline.log"))
    if not launch or not pipeline:
        return NA, "缺 launch.log 或 pipeline.log", []
    # 挑一条只会经 print 产生（不带 logging 级别字段）的标志性输出。
    marker = "[preopt λ="
    in_launch = launch.count(marker)
    in_pipeline = pipeline.count(marker)
    evidence = [
        f"'{marker}' 在 launch.log 出现 {in_launch} 次、pipeline.log {in_pipeline} 次"
    ]
    if in_pipeline == 0:
        return NA, "本次运行还没产生 pilot 输出", evidence
    if in_launch == 0:
        return FAIL, "print 行只到了 pipeline.log，stdout 那份丢了", evidence
    if in_launch < in_pipeline:
        return FAIL, "launch.log 明显落后于 pipeline.log（仍在块缓冲）", evidence
    return PASS, "两个文件都拿到了 print 行", evidence


CHECKS = [
    ("偏置爬坡走满 1.0 档", check_bias_ramp_reaches_full_scale),
    ("pilot 采样语义已记录", check_pilot_traversal_semantics),
    ("preopt 派生层进指纹", check_preopt_derived_layer),
    ("LRC 盒体积恒定", check_lrc_single_volume),
    ("作业防护 / 输出目录锁", check_job_guard),
    ("磁盘预检", check_disk_precheck),
    ("stdout tee 不再丢行", check_stdout_tee),
]


def verify(run: str) -> int:
    print(f"\n{'=' * 78}\n运行目录: {run}\n{'=' * 78}")
    worst = 0
    for title, fn in CHECKS:
        try:
            status, summary, evidence = fn(run)
        except Exception as exc:  # noqa: BLE001 —— 判读脚本自己不该拖垮判读
            status, summary, evidence = FAIL, f"检查本身抛异常: {exc!r}", []
        mark = {PASS: "  [OK]", FAIL: "  [!!]", NA: "  [--]"}[status]
        print(f"{mark} {title}: {summary}")
        for line in evidence:
            print(f"        {line}")
        if status == FAIL:
            worst = 1
    return worst


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="判读 2026-09-09/10 那批改动在真机运行上是否生效（只读）"
    )
    parser.add_argument("run_dirs", nargs="+", help="一个或多个 run 输出目录")
    args = parser.parse_args(argv)

    exit_code = 0
    for run in args.run_dirs:
        if not os.path.isdir(run):
            print(f"[!!] 不是目录: {run}")
            exit_code = 1
            continue
        exit_code |= verify(run)
    print(
        "\n判读完成。[--] 表示这次运行还没走到那一步或产物缺失，不算失败；"
        "[!!] 才是真的没做到。"
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
