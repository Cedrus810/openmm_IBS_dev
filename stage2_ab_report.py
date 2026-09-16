#!/usr/bin/env python3
"""Stage-2 **后分析**：单跑演化报告 + 两跑 A/B 对比。

    python stage2_ab_report.py <run_dir> [<run_dir2> ...]     # 逐跑报告
    python stage2_ab_report.py --ab <baseline_dir> <cand_dir> # A/B 对比
    python stage2_ab_report.py <run_dir> --json               # 机读

只读 + 纯聚合：**不重算任何自由能、不改任何 run**。数据全部来自已落盘的产物
（λ 版本链 / 逐窗 self_support / convergence / 自治历史），本脚本只负责把它们
摆到一起。

为什么单独成脚本：这是**后分析**，不是控制器逻辑。控制器那边只管生成
`comparison_manifest`（`abfe_preoptimizer.Stage2RepairController`），
读它、排版它、对比它是这里的事。

A/B 的三块必看（缺一块结论就是空的）：
  · **身份** —— 两条臂的 λ 表 / 窗口划分 / 协议版本对不对得上。对不上就别比。
  · **验收量** —— 逐窗 min N_eff/g（主验收量，门 10）与路径 ΔG ± σ。
  · **代价** —— 总生产步数 / 开过几个采样段 / 几轮自治。
    outer 臂每步更贵，**只比 ΔG 不比代价等于没比**。
"""
from __future__ import annotations

import glob
import json
import os
import sys


def _j(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _segments(run_dir, stage="vanishing"):
    """基准段 + 所有 `<stage>_N`，按段号升序。"""
    base = os.path.join(run_dir, stage)
    out = [base] if os.path.isdir(base) else []
    numbered = []
    for d in glob.glob(base + "_*"):
        # 口径改动 2026-09-14：判 `<stage>_` 之后的**整段**后缀是不是数字，
        # 而不是 rsplit 的最后一节 —— rewindow 目录名是
        # `vanishing_rewindow_<sha256[:12]>`，末节纯数字的概率 ≈0.34%，
        # 命中就会被当成采样段（审计 #2 同形状）。
        suf = os.path.basename(d)[len(os.path.basename(base)) + 1:]
        if os.path.isdir(d) and suf.isdigit():
            numbered.append((int(suf), d))
    return out + [d for _n, d in sorted(numbered)]


# 口径改动 2026-09-14（审计 #63）：写侧文件名改成 `controller_comparison_manifest.json`
# —— 因为 `_read_stage_result()` 的兜底 glob 吃的是 `stage2_*.json`。旧名留作
# 回退，读旧 run 才不会空手而归。**顺序即权威：新名在前。**
_MANIFEST_NAMES = ("controller_comparison_manifest.json",
                   "stage2_comparison_manifest.json")


def manifest_for(run_dir, stage="vanishing", stage_type="vdw", refresh=True):
    """拿这个 run 的对比清单。`refresh=True` 时重新生成一份并落盘。"""
    ckpt = os.path.join(run_dir, "checkpoints")
    written = None
    if refresh:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from abfe_preoptimizer import Stage2RepairController

            # 口径改动：lo/hi 不再由本脚本从 run_provenance 抄一遍 —— 控制器
            # 构造函数自己就读那份文件（`config_source`），抄一遍只会抄错。
            ctl = Stage2RepairController.for_physical_stage(
                run_dir, stage, stage_type)
            written = ctl.write_comparison_manifest()
        except Exception as err:  # noqa: BLE001 —— 生成不了就读旧的，如实说
            print(f"  [WARN] 重新生成 manifest 失败（{err!r}），读盘上已有的那份。")
    for path in ([written] if written else []) + [
            os.path.join(ckpt, n) for n in _MANIFEST_NAMES]:
        m = _j(path)
        if m is not None:
            return m
    return None


def render_path_evolution(run_dir):
    """λ 路径演化：每个版本的态数 / 逐窗 K / 失败窗跨度 / 插了哪个 λ。"""
    vs = sorted(
        glob.glob(os.path.join(run_dir, "checkpoints", "path_versions", "v*.json")),
        key=lambda x: int(os.path.basename(x)[1:-5]),
    )
    if not vs:
        return "  （没有 λ 版本链）"
    lines = []
    for f in vs:
        d = _j(f) or {}
        v = int(os.path.basename(f)[1:-5])
        rngs = d.get("window_ranges") or []
        lam = [s.get("lambda_vdw") for s in (d.get("states") or [])]
        ev = d.get("event") or {}
        ks = [b - a for a, b in rngs]
        lines.append(f"  v{v}  {len(lam)}态  K={ks}  事件={ev.get('kind')}")
        det = ev.get("detail") or {}
        rng = det.get("failed_global_state_range")
        if rng and lam:
            a, b = int(rng[0]), int(rng[1])
            if b - 1 < len(lam):
                lines.append(
                    f"        目标窗 [{a},{b}) 跨度={lam[a] - lam[b - 1]:.4f}"
                    f"  插入 λ={[round(float(x), 6) for x in (det.get('inserted_lambda_vdw') or [])]}"
                )
    return "\n".join(lines)


def render_segment_matrix(run_dir, stage="vanishing"):
    """逐窗 × 逐段的主验收量。**段号不是新旧的代理**，空行就是空段。"""
    segs = _segments(run_dir, stage)
    n_win = 0
    for d in segs:
        n_win = max(n_win, len(glob.glob(os.path.join(
            d, "dual_window_*_vdw_self_support.json"))))
    n_win = max(n_win, 1)
    head = "  {:<15}".format("段") + " ".join(f"w{i:<6}" for i in range(n_win))
    rows = [head]
    for d in segs:
        cells = []
        for i in range(n_win):
            j = _j(os.path.join(d, f"dual_window_{i}_vdw_self_support.json"))
            cells.append(f"{j.get('min_n_eff_over_g', -1):>7.2f}" if j else "      ·")
        rows.append("  {:<15}".format(os.path.basename(d)) + " ".join(cells))
    return "\n".join(rows)


def render_actions(run_dir):
    h = _j(os.path.join(run_dir, "checkpoints",
                        "stage2_autonomous_history.json")) or {}
    its = h.get("iterations") or []
    if not its:
        return "  （没有自治历史）"
    out = []
    for it in its:
        out.append(
            f"  {it.get('iteration'):>3}  {str(it.get('action')):<22}"
            f" win={str(it.get('windows')):<7}"
            f" exit={it.get('exit')} route={it.get('routing_signal') or '-'}"
        )
        # 逐轮快照（2026-09-12 之后的跑才有）
        for w in (it.get("snapshot") or []):
            v = w.get("min_n_eff_over_g")
            out.append(
                f"        w{w.get('window_idx')} {str(w.get('segment')):<13}"
                f" minN/g={'   ·  ' if v is None else format(v, '6.2f')}"
                f" steps={w.get('production_steps')} {w.get('verdict')}"
            )
    return "\n".join(out)


def render_run(run_dir, refresh=True):
    m = manifest_for(run_dir, refresh=refresh) or {}
    idn, res, cost = (m.get("identity") or {}, m.get("path_result") or {},
                      m.get("cost") or {})
    L = [f"╔═ {run_dir}", "╟─ 身份"]
    L.append(f"   path v{idn.get('path_version')}  {idn.get('n_states')} 态  "
             f"{len(idn.get('window_ranges') or [])} 窗  "
             f"K={[b - a for a, b in (idn.get('window_ranges') or [])]}")
    L.append(f"   采样段={idn.get('aggregated_segments')}")
    L.append("╟─ 路径结果")
    dg, err = res.get("total_delta_G_kJ_mol"), res.get("total_error_kJ_mol")
    # 口径改动 2026-09-14（审计 #58）：`skipped_windows` 只剩**物理窗口**下标，
    # rewindow 子窗（solver 索引 ≥10000）搬到 `skipped_sampling_units`。
    # 键**缺失**（旧 manifest）≠ 一个都没跳，所以分开显示，不用 `or []` 抹平。
    _su = ("未知(旧manifest)" if "skipped_sampling_units" not in res
           else res.get("skipped_sampling_units"))
    L.append(f"   ΔG={dg if dg is None else round(dg, 4)} ± "
             f"{err if err is None else round(err, 4)} kJ/mol  "
             # [2026-09-15] `converged` 已删除：它把「算出了数」误报成「精度已验收」。
             # 换成两个正交状态；`precision_status` 在单次 run 里结构上恒为
             # UNMEASURED（一个 run 只有一个重复），**那既不是达标也不是不达标**。
             f"分析={res.get('analysis_status')}  "
             f"精度={res.get('precision_status')}  "
             f"跳过窗口={res.get('skipped_windows')}"
             f"  跳过子窗={_su}")
    L.append("╟─ 代价")
    L.append(f"   生产 {cost.get('total_production_steps')} 步 | "
             f"{cost.get('n_sampling_segments')} 个采样段 | "
             f"{cost.get('n_autonomous_iterations')} 轮自治")
    L.append("╟─ 逐窗验收量（主验收量 min N_eff/g，门 10）")
    L.append(f"   {'win':>4} {'来源段':<14}{'minN/g':>8} {'n_decorr':>9} "
             f"{'生产步':>9}  verdict")
    for w in (m.get("windows") or []):
        v = w.get("min_n_eff_over_g")
        L.append(f"   {w.get('window_idx'):>4} {str(w.get('segment')):<14}"
                 f"{'     ·  ' if v is None else format(v, '8.2f')}"
                 f" {str(w.get('n_decorrelated')):>9} "
                 f"{str(w.get('production_steps')):>9}  {w.get('verdict')}")
    L.append("╟─ λ 路径演化")
    L.append(render_path_evolution(run_dir))
    L.append("╟─ 逐窗 × 逐段验收量（⚠️ 段号**不是**新旧的代理；跨版本不可直接比）")
    L.append(render_segment_matrix(run_dir))
    L.append("╟─ 自治动作序列")
    L.append(render_actions(run_dir))
    L.append("╚═")
    return "\n".join(L)


def window_signature(w):
    """窗口的**内容身份** = 它装的那串 λ（量化到 1e-9）。

    🔑🔑 [2026-09-16] **A/B 的逐窗对齐不能用 `window_idx`。**
    布局一变（插 λ / 拆窗 / 换 min_states），同一个下标在两条臂上装的是
    **不同的 λ**。真机 cmet_ligand2：A 臂 6 窗、B 臂 7 窗，A-w4 对应 B-w5、
    A-w5 对应 B-w6 —— 按下标对齐会把 A 的失败窗口和 B 的健康窗口摆成一行，
    并算出一个毫无意义的 "B/A"。这次是人工发现的，报告本身没报警。

    返回 `None` 表示这个窗口没有 λ 记录（旧 manifest）⟹ **不猜，不对齐**。
    """
    lam = w.get("lambdas_vdw")
    if not lam:
        return None
    try:
        return tuple(round(float(x), 9) for x in lam)
    except (TypeError, ValueError):
        return None


def render_ab_windows(a, b):
    """逐窗主验收量，按 **λ 签名**对齐，并显示 `verdict_source` 与 `n_decorr`。

    `verdict_source` 必须显示：`HARD_INSUFFICIENT` 有**两个**来源，补救方向不同 ——
      · `min_n_eff_over_g`  比值 <1，支撑真的崩了；
      · `solver_eligibility` 去相关帧数 < 下限，**比值没参与判定**
        （`ibs_engine.window_self_support_check`：不够资格 ⟹ 强制 HARD，压过分档）。
    真机 cmet_ligand2 两条臂里**所有** HARD 都是后者（n_decorr=7/8 < 10），
    而比值 2.37/3.32/4.09 全落在 `INSUFFICIENT_DATA` 档。只显示 verdict
    会把「帧数还不够」读成「权重塌缩」。
    """
    wa = list(a.get("windows") or [])
    wb = list(b.get("windows") or [])
    sa = {window_signature(w): w for w in wa if window_signature(w)}
    sb = {window_signature(w): w for w in wb if window_signature(w)}
    unaligned = [w for w in wa + wb if window_signature(w) is None]

    L = ["╟─ 逐窗主验收量 min N_eff/g（按 **λ 签名**对齐，不是按 window_idx）"]
    if unaligned:
        L.append(f"   ⚠️ 有 {len(unaligned)} 个窗口没有 `lambdas_vdw`（旧 manifest）"
                 "⟹ **不对齐、不显示**，重新生成 manifest 再比。")
    L.append(f"   {'A win':>6} {'B win':>6} {'A':>8} {'B':>8} {'B/A':>7} "
             f"{'A n_dec':>8} {'B n_dec':>8}  verdict(source) A → B")

    def _cell(w, key, fmt="8.2f"):
        v = (w or {}).get(key)
        return "      · " if v is None else format(v, fmt)

    def _vs(w):
        if not w:
            return "·"
        if (w or {}).get("indeterminate"):
            return f"**INDETERMINATE**({w['indeterminate'].get('reason')})"
        return f"{w.get('verdict')}({w.get('verdict_source')})"

    def _ratio(wa_, wb_):
        """配对比值。**任一侧无结论就拒算**，并说明为什么空着。

        无结论 ≠ 失败 ≠ 0。硬算 B/A 等于把一个没测出来的量当成测出来了。
        """
        if (wa_ or {}).get("indeterminate") or (wb_ or {}).get("indeterminate"):
            return "  n.d. "
        va_, vb_ = (wa_ or {}).get("min_n_eff_over_g"), (wb_ or {}).get("min_n_eff_over_g")
        return f"{vb_ / va_:7.2f}" if (va_ and vb_ and va_ > 0) else "      ·"

    seen = set()
    for w in wa:
        sig = window_signature(w)
        if not sig:
            continue
        seen.add(sig)
        o = sb.get(sig)
        ratio = _ratio(w, o)
        L.append(f"   {w.get('window_idx'):>6} "
                 f"{'     ·' if o is None else format(o.get('window_idx'), '6d')} "
                 f"{_cell(w, 'min_n_eff_over_g')} {_cell(o, 'min_n_eff_over_g')} {ratio} "
                 f"{str((w or {}).get('n_decorrelated')):>8} "
                 f"{str((o or {}).get('n_decorrelated')):>8}  {_vs(w)} → {_vs(o)}")
    for w in wb:                      # B 独有的窗口（A 里没有这串 λ）
        sig = window_signature(w)
        if not sig or sig in seen:
            continue
        L.append(f"   {'     ·':>6} {w.get('window_idx'):>6} {'      · ':>8} "
                 f"{_cell(w, 'min_n_eff_over_g')} {'      ·':>7} {'       ·':>8} "
                 f"{str(w.get('n_decorrelated')):>8}  · → {_vs(w)}")
    matched = len(set(sa) & set(sb))
    L.append(f"   —— 对齐上 {matched} 个窗口；A 独有 {len(sa) - matched}、"
             f"B 独有 {len(sb) - matched}。**只有对齐上的行可以比 B/A**。")
    nd = [(w.get("window_idx"), arm)
          for arm, ws in (("A", wa), ("B", wb)) for w in ws
          if (w or {}).get("indeterminate")]
    if nd:
        L.append("   ⚠️ **不可判定**（`n.d.`）："
                 + "、".join(f"{arm}-w{i}" for i, arm in nd)
                 + " 的冻结验证在预算内始终没求出 Δf−ΔF ⟹ 对这份 f_k **无结论**。"
                 "既不是失败也不是 0，**不得**为它硬算 B/A，也**不得**把它从配对里删掉。"
                 "含无结论窗口的 run，其路径级结果不是一个完整观测。")
    return "\n".join(L)


def render_ab(base_dir, cand_dir):
    """A/B：先判身份可比性，再比验收量与代价。"""
    # 测试注入点：传进来的对象自带 manifest 时直接用它，不读盘。
    a = getattr(base_dir, "manifest", None) or manifest_for(base_dir) or {}
    b = getattr(cand_dir, "manifest", None) or manifest_for(cand_dir) or {}
    ia, ib = a.get("identity") or {}, b.get("identity") or {}
    L = ["╔═ A/B 对比", f"║  A(baseline) = {base_dir}", f"║  B(candidate) = {cand_dir}",
         "╟─ 身份可比性"]

    def _cmp(label, x, y):
        ok = "同" if x == y else "**不同**"
        L.append(f"   {label:<22} {ok:<8} A={x}  B={y}")

    _cmp("path_version", ia.get("path_version"), ib.get("path_version"))
    _cmp("态数", ia.get("n_states"), ib.get("n_states"))
    _cmp("window_ranges", ia.get("window_ranges"), ib.get("window_ranges"))
    if ia.get("lambdas_vdw") != ib.get("lambdas_vdw"):
        L.append("   ⚠️ **λ 表不同** —— 两条臂走的不是同一条路径，"
                 "逐窗数字不可直接相减；只有路径级 ΔG 可比，且要连代价一起看。")

    ra, rb = a.get("path_result") or {}, b.get("path_result") or {}
    da, db = ra.get("total_delta_G_kJ_mol"), rb.get("total_delta_G_kJ_mol")
    # 口径改动：误差**未知就是未知**，不 `or 0.0` —— 压成 0 会让下面的合并 σ
    # 只剩单边，报出一个偏大的「Nσ」。
    ea, eb = ra.get("total_error_kJ_mol"), rb.get("total_error_kJ_mol")
    L.append("╟─ 路径级结果")
    # [2026-09-15] 同上：分析完整性与统计精度是两件事，分开报，别再用一个布尔冒充。
    L.append(f"   A  ΔG={da} ± {ea}   分析={ra.get('analysis_status')}"
             f"  精度={ra.get('precision_status')}")
    L.append(f"   B  ΔG={db} ± {eb}   分析={rb.get('analysis_status')}"
             f"  精度={rb.get('precision_status')}")
    L.append("   ⚠️ 精度=UNMEASURED 表示跨重复离散度**没有测**（需要 ≥3 个同协议独立"
             "重复），既不是达标也不是不达标；上面的 ± 是单次 MBAR 的渐近 σ，"
             "它对「该采的构型一次都没采到」失明，不能当精度结论。")
    # 🔑🔑 [2026-09-16] **含"无结论"窗口的 run 不是一个完整观测。**
    # 那个窗口的 Δf−ΔF 在预算内从未求出 ⟹ 路径上有一个洞。任何"总和"都是把洞
    # 当成 0，任何 Δ(B−A) 都是在比两个口径不同的东西。所以这里**拒算**，
    # 并把是哪几个窗口、为什么说清楚 —— 不是静默留空。
    _nd_arm = {nm: [w.get("window_idx") for w in (m.get("windows") or [])
                    if (w or {}).get("indeterminate")]
               for nm, m in (("A", a), ("B", b))}
    if any(_nd_arm.values()):
        L.append("   ⛔ **不可判定，拒绝计算 Δ(B−A)**："
                 + "；".join(f"{nm} 臂窗口 {v}" for nm, v in _nd_arm.items() if v)
                 + " 的冻结验证无结论 ⟹ 该 run 的路径级结果**不完整**。"
                 "补齐的办法只有在**同一协议下**重跑（不是 resume —— resume 会开新的"
                 "冻结周期并进下一阶预算，那是自适应加预算，会毁掉固定预算设计）。")
    elif da is not None and db is not None:
        d = db - da
        sig = (None if (ea is None or eb is None)
               else (float(ea) ** 2 + float(eb) ** 2) ** 0.5)
        L.append(f"   Δ(B−A) = {d:+.4f} kJ/mol"
                 + (f"  = {abs(d) / sig:.2f}σ（合并 σ={sig:.4f}）"
                    if sig else "  （σ 未知，两臂至少一边没有 total_error）"))

    L.append(render_ab_windows(a, b))

    ca, cb = a.get("cost") or {}, b.get("cost") or {}
    L.append("╟─ 代价（⚠️ 只比 ΔG 不比代价等于没比）")
    for k, lab in (("total_production_steps", "生产步数"),
                   ("n_sampling_segments", "采样段数"),
                   ("n_autonomous_iterations", "自治轮数")):
        x, y = ca.get(k), cb.get(k)
        r = f"  ×{y / x:.2f}" if (x and y) else ""
        L.append(f"   {lab:<10} A={x}  B={y}{r}")
    L.append("╚═")
    return "\n".join(L)


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a for a in argv[1:] if a.startswith("--")}
    if not args:
        print(__doc__)
        return 2
    if "--ab" in flags:
        if len(args) != 2:
            print("  --ab 需要正好两个 run_dir（baseline 在前）")
            return 2
        print(render_ab(args[0], args[1]))
        return 0
    for d in args:
        if "--json" in flags:
            print(json.dumps(manifest_for(d), indent=2, ensure_ascii=False,
                             default=str))
        else:
            print(render_run(d))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
