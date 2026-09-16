#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跨 run 的"采样预算"审计：每个 rep 花了多少帧、花在哪个窗口、控制器怎么退出的。

动机（2026-09-15）：同一体系不同 rep 的 ΔG 差 2-3 kcal/mol，而报出的 σ 只有
0.6。要判断这是"采样不够"还是"协议不一样"，需要把下面三件事摆在一张表上：

1. **配置**——两个 rep 的 λ 态数 / 窗口上下限 / 步数是不是真的一样；
2. **实际采样**——每个窗口最终拿到多少帧（自治控制器会补帧，各 rep 不同）；
3. **控制器退出方式**——`CONVERGED` 还是 `NO_FEASIBLE_ACTION`（补帧到顶仍不达标）。

只读，不写 run 目录。输出两张 TSV：
- `runs.tsv`   每个 rep 一行：配置指纹 + 各 stage 的 ΔG/σ + 控制器结局；
- `windows.tsv` 每个窗口一行：λ 跨度、帧数、去相关样本数、判据。

用法::

    python tools/diagnostics/audit_sampling_budget.py RUNS_ROOT [-o OUTDIR]

`RUNS_ROOT` 下的结构假定为 `<system>/<rep>/`，`_trash*` 前缀的目录跳过。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

KJ_PER_KCAL = 4.184

#: 进 runs.tsv 的配置键。挑的是"会改变采样量或 λ 布点"的那些——
#: 两个 rep 只要这里有一个不同，它们就不是同一协议的重复，sd 没有意义。
CONFIG_KEYS = (
    "n_steps_per_window",
    "steps_per_update",
    "stage1_n_states",
    "stage2_n_states",
    "stage2_final_n_states",
    "stage2_free_energy_densify_points",
    "stage2_window_min_states",
    "stage2_window_max_states",
    "n_equil_steps",
    "warmup_steps",
    "max_bias_warmup_steps",
    "rebalance_steps",
    "pilot_n_steps_per_state",
    "min_bias_updates",
    "max_bias_updates",
    "enable_lambda_refine",
    "enable_early_stop",
    "boresch_select",
    "decoupling",
    "potential",
    "stage2_max_production_blocks_per_window",
    "stage2_production_budget_steps",
)


def _effective_config(run_dir: Path) -> dict:
    """本次 run 真正生效的配置。

    `run_provenance.json` 的 `config` 是**求解后**的快照（preset 展开、CLI 覆盖都已
    落进去），`launch_config.json` 只是启动时那份文件、而且有的 run 根本没有它
    （配置放在 `configs/<system>.json`，run 目录里不留副本）。所以先认前者。
    """
    prov = _load_json(run_dir / "run_provenance.json")
    if isinstance(prov, dict) and isinstance(prov.get("config"), dict):
        return prov["config"]
    return _load_json(run_dir / "launch_config.json") or {}


def _provenance(run_dir: Path) -> dict:
    """`launch_provenance.txt` 是**追加**写的：一个 rep 被重跑/续跑过几次，
    这里就有几段。段数 > 1 意味着这个 rep 不是一次跑完的。
    """
    text = ""
    try:
        text = (run_dir / "launch_provenance.txt").read_text(encoding="utf-8")
    except OSError:
        return {}
    blocks = [b for b in text.split("system            ") if b.strip()]
    fields = {}
    for key in ("seed", "abfe_ibs_commit", "launched_at"):
        found = [v.strip() for v in re.findall(rf"^{key}\s+(.*)$", text, flags=re.MULTILINE)]
        uniq = sorted(set(found))
        fields[key] = found[-1] if found else None
        if key == "abfe_ibs_commit":
            # 纯信息。**不要拿 commit 跨度当"这个 rep 不可用"的判据**：协议变更后
            # 重跑 decharging+vanishing 是正常操作，续跑横跨 commit 恰恰说明它被重算过。
            # 该看的是各 stage 自己的 protocol_key / 落盘时间，见 _stage()。
            fields["n_commits"] = len(uniq)
            fields["commits"] = ",".join(c[:7] for c in uniq)
        elif len(uniq) > 1:
            fields[key] = "MIXED:" + "|".join(uniq)
    fields["n_launches"] = len(blocks)
    return fields


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _frames_per_window(leg_dir: Path, tag: str) -> dict[int, int]:
    """从 `dual_window_<i>_<tag>_energies.npy` 的第二维读实际帧数。

    用 mmap 只读 header，不把几百 MB 的能量矩阵拉进内存。
    """
    out: dict[int, int] = {}
    for f in leg_dir.glob(f"dual_window_*_{tag}_energies.npy"):
        try:
            idx = int(f.name.split("window_")[1].split("_")[0])
            out[idx] = int(np.load(f, mmap_mode="r").shape[1])
        except (ValueError, IndexError, OSError):
            continue
    return out


def _n_k(path: Path):
    """`*_u_kn.npy.n_k.npy` 里每个 λ 态的样本数。"""
    if not path.exists():
        return None
    try:
        return np.asarray(np.load(path)).astype(int).tolist()
    except OSError:
        return None


def _stage(ckpt_dir: Path, name: str) -> dict:
    """一个 stage 的结果 + **它是在哪套协议下、什么时候算出来的**。

    `mtime` 是判断"协议变更后这个 stage 有没有被重算"的唯一可靠依据——比数
    commit 靠谱：重跑过的 stage 时间戳会跟着动，没重跑的（比如只留了 stage0/1
    的 evidence 目录）一眼就看得出来落在变更之前。
    """
    path = ckpt_dir / f"{name}.json"
    j = _load_json(path) or {}
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        mtime = None
    return {
        "dG_kJ": j.get("total_delta_G"),
        "err_kJ": j.get("total_error"),
        "n_states": j.get("n_states"),
        "precision": j.get("precision_status"),
        "untrusted": j.get("results_untrusted"),
        "path_fp": (j.get("lambda_path_fingerprint") or {}).get("sha256", "")[:12] or None,
        "proto_ver": (j.get("estimator_analysis_protocol_version")
                      or j.get("vdw_nonbonded_protocol_version")),
        "proto_key": (j.get("protocol_key") or {}).get("sha256", "")[:12] or None,
        "mtime": mtime,
        "present": path.exists(),
    }


def _controller(ckpt_dir: Path) -> dict:
    """stage2 自治控制器的结局与补帧轨迹。

    `NO_FEASIBLE_ACTION` = 控制器想补帧但已经到顶了，这一 stage 是**带着未达标的
    窗口收的工**——ΔG 照样会出数，但它不是收敛结果。
    """
    h = _load_json(ckpt_dir / "stage2_autonomous_history.json")
    if not h:
        return {}
    outcome = h.get("outcome") or {}
    iters = h.get("iterations") or []
    return {
        "status": outcome.get("status"),
        "exit": outcome.get("exit"),
        "reason": (outcome.get("reason") or "").replace("\t", " ").replace("\n", " ")[:200],
        "n_iterations": len(iters),
        "final_ranges": h.get("final_ranges"),
        # 最后一次迭代的快照 = 每个窗口收工时的真实状态
        "last_snapshot": (iters[-1].get("snapshot") if iters else None) or [],
    }


def audit_rep(system: str, rep: str, run_dir: Path):
    ckpt = run_dir / "checkpoints"
    cfg = _effective_config(run_dir)
    prov = _provenance(run_dir)
    binding = _load_json(run_dir / "final_binding_results.json") or {}

    complex_kJ = binding.get("complex_delta_G_kJ_mol")
    solvent_kJ = binding.get("solvent_delta_G_kJ_mol")
    dG_bind = None
    if complex_kJ is not None and solvent_kJ is not None:
        dG_bind = -(complex_kJ - solvent_kJ) / KJ_PER_KCAL

    ctl = _controller(ckpt)
    row = {
        "system": system,
        "rep": rep,
        "dG_bind_kcal": dG_bind,
        "complex_kJ": complex_kJ,
        "solvent_kJ": solvent_kJ,
        "ctl_status": ctl.get("status"),
        "ctl_exit": ctl.get("exit"),
        "ctl_iterations": ctl.get("n_iterations"),
        "ctl_n_windows": len(ctl.get("final_ranges") or []),
        "seed": prov.get("seed"),
        "n_commits": prov.get("n_commits"),
        "commits": prov.get("commits"),
        "n_launches": prov.get("n_launches"),
    }
    for key in CONFIG_KEYS:
        row[f"cfg.{key}"] = cfg.get(key)

    for leg, root in (("complex", run_dir), ("solvent", run_dir / "solvent_leg")):
        st1 = _stage(root / "checkpoints", "stage1_decharging")
        st2 = _stage(root / "checkpoints", "stage2_vanishing")
        for name, st in (("s1", st1), ("s2", st2)):
            for k, v in st.items():
                row[f"{leg}.{name}.{k}"] = v
        frames = _frames_per_window(root / "vanishing", "vdw")
        row[f"{leg}.s2.total_frames"] = sum(frames.values()) if frames else None
        row[f"{leg}.s2.n_windows"] = len(frames) or None
        nk = _n_k(root / "decharging" / "decharging_pme_u_kn.npy.n_k.npy")
        row[f"{leg}.s1.n_k"] = ",".join(map(str, nk)) if nk else None

    row["ctl_reason"] = ctl.get("reason")

    # 每窗口一行：控制器最后一次快照 + 落盘帧数
    frames = _frames_per_window(run_dir / "vanishing", "vdw")
    win_rows = []
    for snap in ctl.get("last_snapshot") or []:
        idx = snap.get("window_idx")
        win_rows.append({
            "system": system,
            "rep": rep,
            "leg": "complex",
            "window": idx,
            "n_frames": frames.get(idx),
            "lambda_span": snap.get("lambda_span"),
            "n_decorrelated": snap.get("n_decorrelated"),
            "min_n_eff_over_g": snap.get("min_n_eff_over_g"),
            "verdict": snap.get("verdict"),
            "production_steps": snap.get("production_steps"),
            "derailment": snap.get("derailment_status"),
        })
    if not win_rows:  # 没有控制器历史（老 run / 非自治）时至少给帧数
        win_rows = [
            {"system": system, "rep": rep, "leg": "complex", "window": i, "n_frames": n}
            for i, n in sorted(frames.items())
        ]
    return row, win_rows


def _write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs_root", help="包含 <system>/<rep>/ 的根目录")
    ap.add_argument("-o", "--outdir", default=".", help="TSV 输出目录（默认当前目录）")
    ap.add_argument("--system", action="append", help="只看这些体系（可重复）")
    args = ap.parse_args(argv)

    root = Path(args.runs_root)
    if not root.is_dir():
        print(f"[错误] 不是目录: {root}", file=sys.stderr)
        return 2

    run_rows, win_rows = [], []
    for system in sorted(os.listdir(root)):
        if args.system and system not in args.system:
            continue
        sys_dir = root / system
        if not sys_dir.is_dir():
            continue
        for rep in sorted(os.listdir(sys_dir)):
            if rep.startswith("_trash") or not (sys_dir / rep).is_dir():
                continue
            r, w = audit_rep(system, rep, sys_dir / rep)
            run_rows.append(r)
            win_rows.extend(w)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_tsv(outdir / "runs.tsv", run_rows)
    _write_tsv(outdir / "windows.tsv", win_rows)

    # 终端上直接给最要紧的三列，剩下的去看 TSV
    print(f"{'system':16} {'rep':28} {'dG_kcal':>8} {'frames':>7} {'ctl_exit':>22}")
    for r in run_rows:
        dg = r["dG_bind_kcal"]
        print(f"{r['system']:16} {r['rep']:28} "
              f"{'' if dg is None else f'{dg:8.2f}'} "
              f"{r.get('complex.s2.total_frames') or '':>7} "
              f"{str(r.get('ctl_exit') or ''):>22}")
    print(f"\n-> {outdir/'runs.tsv'}  |  {outdir/'windows.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
