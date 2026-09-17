"""S2-H + S2-F：会**开新采样段**的动作必须有次数预算 + 事后有效性判定。

`RECALIBRATE_FK` `probe_only=False`、每次开一个新采样段 ⟹
  · 停滞保护判「同一动作 + **盘面未变**」⟹ 盘面每次都变 ⟹ 计数恒 1，永远到不了 3；
  · `action_noop_fingerprint()` 含段维度 ⟹ 新段 = 新指纹 ⟹ 记录永不匹配。
⟹ **对唯一的通用刹车结构性免疫**。而插 λ 有 `max_path_insertions`、补帧有块数硬
上限 —— 只有这一族什么预算都没有。

真机 `cmet_ligand1/rep1`（31/40 轮里 17 轮花在窗口 4）：连发 4 次，每次丢掉上一段的
累计步数，验收量 9.645 → 4.182 → 0.837，盘上留下 7 个采样段，全程无人察觉。
它在第 20 轮时 ratio=9.645、离门 10 只差 0.355，**再补一块几乎必过**。
"""
import pytest

pytestmark = pytest.mark.cpu_only

from abfe_preoptimizer import (
    STRUCTURAL_ACTION_MAX_PER_WINDOW,
    structural_action_history,
    structural_action_refused,
)

# 真机 cmet_ligand1/rep1 窗口 4 的实际验收量序列
REAL = (8.968410609589649, 9.645148149977746, 4.181886348618463, 0.837359127077206)


def _view(vals, *, pv=2, win=4, action="RECALIBRATE_FK"):
    return {"path_version": pv, "autonomous_history": {"iterations": [
        {"path_version": pv, "action": action, "windows": [win],
         "snapshot": [{"window_idx": win, "min_n_eff_over_g": v}]}
        for v in vals]}}


def test_history_counts_only_this_action_this_window_this_layout():
    n, ser = structural_action_history(
        _view(REAL)["autonomous_history"]["iterations"],
        "RECALIBRATE_FK", 4, 2)
    assert n == len(REAL) and [round(x, 3) for x in ser] == [8.968, 9.645, 4.182, 0.837]
    # 别的窗口 / 别的动作 / 别的布局都不算
    its = _view(REAL)["autonomous_history"]["iterations"]
    assert structural_action_history(its, "RECALIBRATE_FK", 5, 2)[0] == 0
    assert structural_action_history(its, "RELEARN_FK_EPOCH", 4, 2)[0] == 0
    assert structural_action_history(its, "RECALIBRATE_FK", 4, 99)[0] == 0


def test_the_real_runaway_is_stopped_before_the_fourth():
    """真机那条链必须在第 3 次就被拦，而不是跑到 4 次、丢掉 750k 步三回。"""
    assert structural_action_refused(_view(REAL[:1]), "RECALIBRATE_FK", 4) is None
    assert structural_action_refused(_view(REAL[:2]), "RECALIBRATE_FK", 4) is None
    why = structural_action_refused(_view(REAL[:3]), "RECALIBRATE_FK", 4)
    assert why is not None, "连发三次仍放行 —— 次数预算没生效"


def test_the_effectiveness_gate_fires_on_its_own():
    """次数没到上限，但验收量在往下走 ⟹ 也要拦（这一道治「三次全变差却无人察觉」）。"""
    # 三点、末点明显低于前面各点的中位数
    why = structural_action_refused(_view((9.0, 9.6, 4.2)), "RECALIBRATE_FK", 4)
    assert why is not None and "没有把验收量推上去" in why


def test_a_still_improving_series_is_not_refused_by_the_effectiveness_gate():
    """反面：真的在变好，**效果闸**不得拦（否则变成"一律不重标定"）。

    ⚠️ 三点时**次数闸**会拦（上限 3），所以这里用两点验效果闸单独的行为：
    序列上升 ⟹ 不该给出"没有把验收量推上去"。
    """
    why = structural_action_refused(_view((2.0, 9.0)), "RECALIBRATE_FK", 4)
    assert why is None, why


def test_the_two_gates_do_not_shadow_each_other():
    """默认配置下 `MAX_PER_WINDOW == MARGINAL_GAIN_MIN_POINTS == 3`，两道同点触发。

    效果闸必须排在**前面**，否则它永远不可达（死代码）；而且它的理由更有信息量。
    """
    from abfe_preoptimizer import MARGINAL_GAIN_MIN_POINTS
    assert STRUCTURAL_ACTION_MAX_PER_WINDOW == MARGINAL_GAIN_MIN_POINTS
    bad = structural_action_refused(_view((9.0, 9.6, 4.2)), "RECALIBRATE_FK", 4)
    good = structural_action_refused(_view((2.0, 5.0, 9.0)), "RECALIBRATE_FK", 4)
    assert "没有把验收量推上去" in bad, "效果闸被次数闸遮住了"
    assert "已经对窗口" in good, "序列在变好时应由次数闸给出理由"


def test_non_segment_opening_actions_are_untouched():
    """这道闸只管**开新段**那一族 —— 补帧/布局动作各有自己的预算，不得被它波及。"""
    for act in ("RUN_PRODUCTION", "INSERT_LAMBDA", "SPLIT_TAIL_WINDOW",
                "CONTINUE_WARMUP", "ANALYZE", "IMMUTABLE_REWINDOW"):
        assert structural_action_refused(_view(REAL, action=act), act, 4) is None, act


def test_budget_is_per_layout_not_per_run():
    """布局一变就是另一套几何，重来一次是对的 —— 预算按 `path_version` 隔离。"""
    v = _view(REAL, pv=2)
    v["path_version"] = 3                      # 布局已演化
    assert structural_action_refused(v, "RECALIBRATE_FK", 4) is None


def test_the_cap_is_the_same_spirit_as_the_insertion_budget():
    assert STRUCTURAL_ACTION_MAX_PER_WINDOW >= 1


# ─────────────────────────────────────────────────────────────────────────────
# S2-L：补帧的**块数硬上限**是资源账，不该因为「换布局」就退钱
# ─────────────────────────────────────────────────────────────────────────────

def _ctl_with_history(tmp_path, iterations):
    import json
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tests"))
    from test_stage2_repair_controller import _mkrun, R4
    from abfe_preoptimizer import Stage2RepairController as C
    run = _mkrun(tmp_path, windows={i: {"K": 4} for i in range(4)},
                 ranges=R4, n_states=13)
    with open(os.path.join(run, "checkpoints",
                           "stage2_autonomous_history.json"), "w") as fh:
        json.dump({"iterations": iterations}, fh)
    return C.for_physical_stage(run, "vanishing", "vdw")


def test_the_hard_cap_counts_blocks_across_layout_changes(tmp_path):
    """插 λ 推进 `path_version` **不得**把补帧配额清零。

    BUD-03 修掉了「换**段**清零」，但两本账都仍按 `path_version` 过滤 ⟹
    「换**布局**清零」是同一个 bug 的另一个维度。而硬上限的定义是**资源账** ——
    烧掉的 GPU 不会因为布局变了就回来。

    真机（`abfe-benchmark-cb` 快照）：`brd4_ligand1/rep2` w3 累计 1,250,000 步
    = **5 块**，而上限是 4；`cmet_ligand1/rep1` w4 实测跨布局 **9 块**、
    而旧口径只看到 2 块。叠加 `S2-A`（边际增益刹车全程没有输入），
    补帧在布局反复演化的 run 上**实际没有有效上限**。
    """
    # ⚠️ 块账按**生产步数**去重（同一步数只算一块）。真机上步数是单调累加的
    #    （250k → 500k → 750k …，见"从生产 checkpoint 续算"那条日志），
    #    所以 fixture 也必须单调，否则会被去重折叠、测不到本条要测的东西。
    its = []
    for pv in (1, 2, 3):                       # 三个布局，每个布局两块
        for _ in range(2):
            its.append({
                "iteration": len(its) + 1, "action": "RUN_PRODUCTION",
                "windows": [1], "path_version": pv,
                "snapshot": [{"window_idx": 1, "segment": "vanishing",
                              "production_steps": 250000 * (len(its) + 1),
                              "solver_n_decorrelated": 30 + len(its),
                              "min_n_eff_over_g": 5.0}]})
    v = _ctl_with_history(tmp_path, its).read()
    _tot = len((v.get("production_blocks_total_by_window") or {}).get(1) or [])
    _same = len((v.get("production_blocks_by_window") or {}).get(1) or [])
    assert _tot == 6, (
        f"硬上限那本账只数到 {_tot} 块 —— 换布局把配额退回去了。"
        "它是**资源账**，跨布局累计。")
    # ⚠️ 边际增益那本**仍然**要按布局过滤：跨布局比 min N_eff/g 没有意义
    #    （窗口几何都变了）。两本账的过滤维度本来就不同。
    assert _same < _tot, (
        "边际增益那本也跨布局累计了 —— 它必须只留当前布局的点，"
        "否则是在一条不可比的曲线上判趋势")
