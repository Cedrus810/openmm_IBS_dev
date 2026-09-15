# -*- coding: utf-8 -*-
"""三条「读一个没人写的键」（2026-09-14 真机 cyclod_ligand1/rep2）。

同一个形状栽了三次，后果是自治循环的两个布局动作一个失效、一个失控：

  CTL-11 `tail_repartition_anchor` 读 `w["lambda_vdw_hi"]` —— 全仓**零处写**。
         anchor 恒为 None ⟹ 执行器每次都走「取不到 tail anchor，跳过」⟹
         `SPLIT_TAIL_WINDOW` **自诞生起一次都没真正执行过**。真机发了 4 次，
         版本链上一条 `tail_repartition` 都没有。
  CTL-12 决策侧只问「结构上拆得开吗」（K ∈ [2lo−1, 2hi−1]），不问执行器需要的
         anchor 拿不拿得到 ⟹ 发一个注定落空的动作（失败形状 ①）。
  CTL-13 `_read_path` 从 `record["kind"]` 数事件，而 kind 在 `record["event"]["kind"]`
         ⟹ `events` 恒为 {} ⟹ `path_insertions_left` 恒等于满额 ⟹ 插 λ 的跨
         resume 上限从没生效。真机 max_path_insertions=3 却插了 6 次，末窗
         K=6→12（失败形状 ⑥：没有停止条件）。
"""
import json

import pytest

from abfe_preoptimizer import Stage2RepairController

from test_stage2_repair_controller import _mkrun


def _with_lambdas(run, n_states, ranges, *, events=()):
    """把版本链改写成**真实形状**：states 带 lambda_vdw、kind 在 event 里。

    `_mkrun` 的 v1 是个极简壳（states 只有 id、kind 在顶层），本文件测的正是
    "读错键" 这一类，所以必须按真产物的形状造。
    """
    import os
    pv = os.path.join(run, "checkpoints", "path_versions")
    lam = [1.0 - i / (n_states - 1.0) for i in range(n_states)]
    rec = {
        "version": 1,
        "event": {"kind": "init", "reason": "initial_path"},
        "states": [{"id": f"s{i}", "lambda_vdw": lam[i]} for i in range(n_states)],
        "window_ranges": [list(r) for r in ranges],
    }
    with open(os.path.join(pv, "v1.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    for n, kind in enumerate(events, start=2):
        rec2 = dict(rec, version=n, event={"kind": kind, "reason": "t"})
        with open(os.path.join(pv, f"v{n}.json"), "w", encoding="utf-8") as fh:
            json.dump(rec2, fh)
        with open(os.path.join(run, "checkpoints", "path_current.json"),
                  "w", encoding="utf-8") as fh:
            json.dump({"version": n}, fh)
    return lam


# ranges 与真机 cyclod_ligand1/rep2 同形：末窗 K=12，lo/hi=4/8 ⟹ 可拆区间 [7,15]。
RANGES = [(0, 8), (7, 13), (12, 16), (15, 27)]
WINDOWS = {
    0: {"K": 8}, 1: {"K": 6}, 2: {"K": 4},
    3: {"K": 12, "self_verdict": "HARD_INSUFFICIENT"},
}


def _ctl(tmp_path, *, events=()):
    run = _mkrun(tmp_path, windows=WINDOWS, ranges=RANGES, n_states=27)
    lam = _with_lambdas(run, 27, RANGES, events=events)
    return Stage2RepairController(run), lam


# --------------------------------------------------------------- CTL-11
def test_tail_anchor_is_the_first_state_of_the_first_untrusted_window(tmp_path):
    ctl, lam = _ctl(tmp_path)
    view = ctl.read()
    assert ctl.first_untrusted_window(view) == 3
    # anchor = 那个窗口自己的首态（= 与前一窗共享的节点），不是任何别的字段。
    assert ctl.tail_repartition_anchor(view) == pytest.approx(lam[RANGES[3][0]])


def test_an_obtainable_anchor_actually_splits_the_tail(tmp_path):
    """光有 anchor 不算数 —— 它得真能喂给执行器用的那个重分函数。"""
    import abfe_preoptimizer as pre
    ctl, _ = _ctl(tmp_path)
    view = ctl.read()
    new_ranges, diag = pre.repartition_tail_from_anchor(
        view["path"]["lambdas_vdw"], view["path"]["window_ranges"],
        ctl.tail_repartition_anchor(view),
        min_states_per_window=4, max_states_per_window=8,
    )
    assert all(4 <= b - a <= 8 for a, b in new_ranges), new_ranges
    assert len(new_ranges) == len(RANGES) + 1
    assert [tuple(r) for r in diag["frozen_prefix_windows"]] == RANGES[:3]


# --------------------------------------------------------------- CTL-12
def test_split_is_infeasible_when_the_executor_could_not_get_an_anchor(tmp_path):
    """所有窗口都可信 ⟹ 没有 anchor ⟹ 拆窗**不可行**，哪怕 K 落在可拆区间。

    这是失败形状 ①：动作对当前状态在结构上不可能，却被判成"可以试"。
    """
    run = _mkrun(tmp_path, windows={0: {"K": 8}, 1: {"K": 6}, 2: {"K": 4}, 3: {"K": 12}},
                 ranges=RANGES, n_states=27)
    _with_lambdas(run, 27, RANGES)
    ctl = Stage2RepairController(run)
    view = ctl.read()
    assert ctl.first_untrusted_window(view) is None
    assert ctl.tail_repartition_anchor(view) is None
    # K=12 在 [7,15] 内 ⟹ 纯布局判据说"拆得开"；但执行器下不了刀。
    assert ctl.feasible(view)["split_tail_window"] is not None


# --------------------------------------------------------------- CTL-13
def test_insertion_events_are_counted_from_the_event_kind(tmp_path):
    ctl, _ = _ctl(tmp_path, events=("insert_lambda",) * 6)
    view = ctl.read()
    assert view["path"]["events"].get("insert_lambda") == 6
    assert view["path_insertions_budget"] == 3
    assert view["path_insertions_left"] == 0


def test_insertion_budget_is_actually_consulted(tmp_path):
    """预算算出来还得有人看 —— 原来它只出现在一行诊断打印里。"""
    ctl, _ = _ctl(tmp_path, events=("insert_lambda",) * 6)
    view = ctl.read()
    why = ctl.feasible(view)["insert_lambda"]
    assert why is not None and "预算" in why

    fresh, _ = _ctl(tmp_path / "fresh", events=("insert_lambda",))
    assert fresh.feasible()["insert_lambda"] is None


def test_no_decide_branch_can_emit_an_infeasible_layout_action(tmp_path):
    """收口：预算耗尽 + 无 anchor 时，decide() 绝不发这两个动作。"""
    run = _mkrun(tmp_path, windows={0: {"K": 8}, 1: {"K": 6}, 2: {"K": 4}, 3: {"K": 12}},
                 ranges=RANGES, n_states=27)
    _with_lambdas(run, 27, RANGES, events=("insert_lambda",) * 6)
    ctl = Stage2RepairController(run)
    feas = ctl.feasible()
    assert feas["insert_lambda"] is not None and feas["split_tail_window"] is not None
    assert ctl.decide()["action"] not in ("INSERT_LAMBDA", "SPLIT_TAIL_WINDOW")


# --------------------------------------------------------------- CTL-14
def test_noop_records_expire_when_the_layout_changes(tmp_path):
    """no-op 账本记的是「在这个盘面上没用」，不是「这个动作永远没用」。

    CTL-14（同形状第 4 次）：`action_noop_fingerprint()` 的两个调用方都写
    `view.get("path_version")`，而路径版本在 `view["path"]["version"]` ——
    两侧**一致地**拿到 None ⟹ 版本位恒为 0 ⟹ 布局变了记录也不失效。
    而插 λ / 拆末窗恰恰就是「让先前没用的动作重新变得有用」的操作。
    """
    from abfe_preoptimizer import action_noop_fingerprint

    ctl, _ = _ctl(tmp_path)
    v1 = ctl.read()
    assert v1["path_version"] == 1, "view 必须自己暴露路径版本"
    w = {x["window_idx"]: x for x in v1["windows"]}[3]
    fp1 = action_noop_fingerprint(w, v1["path_version"])

    # 布局演化一次（插 λ）：同一个窗口、同样的帧数，指纹必须变。
    ctl2, _ = _ctl(tmp_path / "evolved", events=("insert_lambda",))
    v2 = ctl2.read()
    assert v2["path_version"] == 2
    fp2 = action_noop_fingerprint(
        {x["window_idx"]: x for x in v2["windows"]}[3], v2["path_version"])
    assert fp1 != fp2, f"布局从 v1 演化到 v2，no-op 指纹却没变：{fp1}"


def test_noop_fingerprint_does_not_silently_drop_the_segment_axis(tmp_path):
    """窗口记录里没有 `segment` 键 —— 直接读它等于把「换段」这一维关掉。"""
    from abfe_preoptimizer import action_noop_fingerprint

    ctl, _ = _ctl(tmp_path)
    w = dict({x["window_idx"]: x for x in ctl.read()["windows"]}[3])
    assert "segment" not in w or w.get("segment") is None
    a = action_noop_fingerprint(dict(w, n_production_segments=1), 1)
    b = action_noop_fingerprint(dict(w, n_production_segments=2), 1)
    assert a != b, "开了新段，no-op 指纹必须变"
