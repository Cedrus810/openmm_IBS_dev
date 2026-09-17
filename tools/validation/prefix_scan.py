"""平衡前缀丢弃扫描：离线重算一条腿的 stage2 (vanishing) ΔG vs 丢弃比例。

零 GPU，只读已落盘的 `dual_window_*_vdw_{energies,bias,base}.npy`。
唯一变量 = 每个窗口丢掉最前面 f 比例的帧。`f_k` 是窗口级冻结量、不随帧数变
⟹ 这是一个干净的单变量实验。

用法
----
    python tools/validation/prefix_scan.py <run 相对路径> [...]

    # 例（路径相对 abfe-benchmark/openmm_IBS/runs）
    python tools/validation/prefix_scan.py brd4_ligand2/rep1

自验
----
`f=0.5` 的重算应逐位复现该 run 自己记录的
`stage_diagnostics.stage2.split_half_diagnostics.total_delta_G_second_half_kJ_mol`。
实测 brd4_ligand2 rep1：两边都是 `103.587`。对不上就别信这个脚本的其它输出。

⚠️ 两条必读的解读警告
---------------------
1. **`f >= 0.7` 的点不可用。** 去相关子采样后有效帧 < 10 ⟹ `ANALYSIS_INCOMPLETE`；
   brd4_ligand1 rep1 在 `f=0.8` 返回哨兵 `ΔG=0.000, σ=999.9`。别把这些点画进曲线。

2. **零结果不能证伪初值偏差，只能证实。** 判据是不对称的：
   若空腔弛豫是 ns 尺度而单窗只有 500 ps，则整段都落在弛豫之前 ⟹ 留下的与丢掉的
   **同等偏** ⟹ ΔG 纹丝不动、split-half 两半同等偏 ⟹ 差值符号随机。
   所以"曲线平坦"同时符合「没有初值偏差」和「整段都在弛豫之前」。
   只有**单调变化后趋稳**是强结论。

   实测（2026-09-17，五个 brd4 run）：方向 +5.7 / ≈0 / ≈0 / −4 / −9 kJ/mol，
   符号不一致 ⟹ 零结果。**单 run 的正结果同样不可信** —— brd4_ligand2 rep1 单看
   就是教科书式的单调上升后平台，多跑四个才看出符号不一致。

   完整结论见 `docs/design/PROPOSAL_openfe_method_absorption.md` §2.2。
"""
import os, sys, json, copy, glob
os.environ.setdefault("PYMBAR_DISABLE_JAX", "1")
sys.path.insert(0, "/home/ruigengji/ABFE_IBS/ABFE_IBS")
import numpy as np
from ibs_engine import load_ibs_window_outputs_from_dir, solve_stage_integrated

KT = 8.31446261815324e-3 * 300.0  # kJ/mol @300K

FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def load_leg(leg_root):
    """leg_root 里有 vanishing/ 与 checkpoints/"""
    van = os.path.join(leg_root, "vanishing")
    ckpt = os.path.join(leg_root, "checkpoints")
    pre = json.load(open(os.path.join(ckpt, "preopt_dual_vanishing.json")))
    ranges = [tuple(int(x) for x in r) for r in pre["window_ranges"]]
    lv = [float(x) for x in pre["lambdas_var"]]
    lc = [0.0] * len(lv)
    outs = load_ibs_window_outputs_from_dir(
        van, ranges, lc, lv, checkpoint_dir=ckpt, stage_type="vdw",
    )
    return outs


def _shift_segments(segments, n0, n_new):
    """丢掉最前面 n0 帧之后重映射 production_segments。

    段是连续覆盖 [0, n_frames) 的；整段落在被丢区间内的删掉，
    跨界那段裁掉左半，其余整体左移 n0。校验器要求 start 从 0 连续。
    """
    out = []
    for seg in segments or []:
        s0, e0 = int(seg["start_frame"]), int(seg["end_frame"])
        if e0 <= n0:
            continue
        s1, e1 = max(s0, n0) - n0, e0 - n0
        seg2 = dict(seg)
        seg2["start_frame"], seg2["end_frame"], seg2["n_frames"] = s1, e1, e1 - s1
        out.append(seg2)
    if not out or out[0]["start_frame"] != 0 or out[-1]["end_frame"] != n_new:
        raise AssertionError(f"段重映射不自洽: {[(s['start_frame'],s['end_frame']) for s in out]} vs {n_new}")
    return out


def truncate(outs, frac):
    new = []
    for w in outs:
        w2 = dict(w)
        n = w["u_kn"].shape[1]
        n0 = int(np.floor(frac * n))
        if n - n0 < 20:          # 守住求解器的最小帧数，别把窗口掏空
            return None
        w2["u_kn"] = np.ascontiguousarray(w["u_kn"][:, n0:])
        w2["bias_energies"] = np.ascontiguousarray(w["bias_energies"][n0:])
        w2["base_energies"] = np.ascontiguousarray(w["base_energies"][n0:])
        w2["production_segments"] = _shift_segments(
            w.get("production_segments"), n0, n - n0)
        new.append(w2)
    return new


def scan(label, leg_root):
    try:
        outs = load_leg(leg_root)
    except Exception as exc:
        print(f"[SKIP] {label}: {type(exc).__name__}: {exc}")
        return
    n_frames = [w["u_kn"].shape[1] for w in outs]
    print(f"\n=== {label} ===")
    print(f"  窗口数={len(outs)}  每窗帧数={n_frames}")
    rows = []
    for f in FRACTIONS:
        wo = truncate(outs, f)
        if wo is None:
            print(f"  f={f:.1f}  帧数不足，停")
            break
        try:
            res = solve_stage_integrated(
                copy.deepcopy(wo), KT, stage_name="vanishing",
                skip_split_half_diagnostics=True,
            )
        except Exception as exc:
            print(f"  f={f:.1f}  求解失败 {type(exc).__name__}: {exc}")
            continue
        dg = res.get("total_delta_G")
        err = res.get("total_error")
        st = res.get("analysis_status")
        per = [round(float(s.get("delta_G_kJ_mol", float('nan'))), 2)
               for s in (res.get("local_results") or [])]
        rows.append((f, dg, err, st, per))
        print(f"  f={f:.1f}  n_kept={[w['u_kn'].shape[1] for w in wo]}  "
              f"ΔG={dg:9.3f}  σ={err:6.3f}  {st}  逐窗={per}")
    if len(rows) >= 2:
        d = rows[-1][1] - rows[0][1]
        print(f"  ⟹ 端到端变化 f=0 → f={rows[-1][0]:.1f}: {d:+.3f} kJ/mol")
    return rows


if __name__ == "__main__":
    B = "/home/ruigengji/abfe-benchmark/openmm_IBS/runs"
    targets = []
    for pat in sys.argv[1:] or ["brd4_ligand2/_backup_rep1_vdw_20260915-2015"]:
        root = os.path.join(B, pat)
        targets.append((pat + " [complex]", root))
        sv = os.path.join(root, "solvent_leg")
        if os.path.isdir(os.path.join(sv, "checkpoints")):
            targets.append((pat + " [solvent]", sv))
    for label, root in targets:
        scan(label, root)
