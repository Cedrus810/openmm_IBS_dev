"""Stage-2 控制器的实际逻辑。对照 `_decide_once` 的 1832 行。

复用仓库已有的 `marginal_gain_stalled`；不新造判据。
"""
from abfe_preoptimizer import marginal_gain_stalled

# ---- 诊断：按顺序问 4 个问题，第一个 yes 就是病因。没有第 5 条。 ----
def diagnose(w, history):
    if w.get("stale_layout_evidence_only"):
        return "LAYOUT"                      # 证据描述的是另一套几何
    if w.get("solver_skip") or w.get("self_verdict") == "HARD_INSUFFICIENT":
        return "FRAMES"                      # 帧不够，还没测出来
    if marginal_gain_stalled([v for _, v in history])[0]:
        return "SPAN"                        # 加帧已被这个窗口自己的数据证伪
    if w.get("cum_fk_verdict") == "FAIL_CUMULATIVE_FK":
        return "FK"                          # f_k 偏
    if w.get("self_verdict") == "ANALYSIS_ELIGIBLE":
        return "OK"
    return "FRAMES"                          # 默认：还没测出来 ≠ 失败

# ---- 处方：一个病因一个动作。SPAN 用 rewindow —— 唯一对任意窗口都可行的缩跨度。 ----
REMEDY = {
    "LAYOUT": "RUN_PRODUCTION",
    "FRAMES": "RUN_PRODUCTION",
    "FK":     "RECALIBRATE_FK",
    "SPAN":   "IMMUTABLE_REWINDOW",
}

def decide(view, budget):
    for w in sorted(view["windows"], key=lambda x: x["window_idx"]):
        i = int(w["window_idx"])
        cause = diagnose(w, view["min_n_eff_over_g_history"].get(i, []))
        if cause == "OK":
            continue
        action = REMEDY[cause]
        why = budget.unaffordable(action, i)
        if why:
            # ⚠️ 付不起对症动作时**不降级**成便宜的动作。
            # 现行 plan() 的 O1 在这里改发 RUN_PRODUCTION —— 那就是「把错误的采样
            # 无限增大」的出处：f_k 偏的窗口被塞进更多同偏斜分布的帧。
            return {"action": "NO_ACTION", "window": i,
                    "exit": "HALT_BUDGET", "cause": cause, "reason": why}
        return {"action": action, "window": i, "cause": cause}
    return {"action": "DONE" if view["stage_analysis_status"] == "ANALYSIS_COMPLETE"
            else "ANALYZE"}
