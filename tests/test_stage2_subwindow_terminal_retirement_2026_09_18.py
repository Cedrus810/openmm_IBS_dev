"""2026-09-18 静态审查修掉的三条控制器缺陷的回归钉。

三条都不是「阈值调错」，而是**语义在下一层丢掉**：

① `plan()` 内部把 `action` 改写成 `NO_ACTION` 时配了**非终态**出口
   （`HALT_BUDGET`）⟹ `plan["terminal"]` 为 False ⟹ 主循环不 break ⟹
   分发段没有 `NO_ACTION` 的执行器 ⟹ 整跑以
   `status=HALTED_NO_EXECUTOR` / `exit=None` 收场。下游每一道门都按 `exit`
   分类，拿不到出口码就分不清这一跑是怎么停的。
   `test_stage2_controller_vocabulary.py` 那条 AST 扫描**看不见**这种内部改写
   （它自己的 docstring 写明了这个盲区），所以这里从行为侧钉。

② 两条**子窗**终态（`D3_REWINDOW_DEPTH_EXHAUSTED` / `SUPPORT_ATTRIBUTION_UNKNOWN`）
   不传 `blocking_window` ⟹ `plan()` 拿 `earliest` 兜底。而卡死的是子窗的
   **父窗**，`_spw < earliest` 时退役会退掉一个无辜窗口；`earliest is None`
   时 `blocking_window` 干脆是 `None`，`_retirable_window()` 第一行就返回，
   于是 S2-G 那条「布局里有、产物里一个都没有的窗口 = 还有活干」的兜底
   **永不执行** —— 整跑判终态，而那个窗口一步都没跑过。

③ 退役之后两条分支**不看 `retired`** ⟹ `_decide_once` 重跑时原样再触发、
   再交出同一个终态。退役只在 `retired_windows` 里留了条记录，下游该跑的
   窗口仍然一步没跑。
"""
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import abfe_preoptimizer as pre  # noqa: E402
from abfe_preoptimizer import (  # noqa: E402
    SUPPORT_FAILURE_STRUCTURAL,
    Stage2RepairController,
)


def _ctl():
    """不碰盘的控制器：本文件的用例全部直接喂 `view`，`read()` 不会被调用。"""
    c = object.__new__(Stage2RepairController)
    c.lo, c.hi, c.max_path_insertions = 4, 8, 3
    c.allow_untrusted = False
    c.stage_dir = c.path_checkpoint_dir = c.checkpoint_dir = "/nonexistent"
    c.stage_name = c._stage_base = "vanishing"
    c.partition_criterion = "state_count"
    return c


def _w(idx, **kw):
    w = {"window_idx": idx, "phase": "PRODUCTION", "verdict": "VALID_PASS",
         "self_verdict": "ANALYSIS_ELIGIBLE", "warmup_steps_left": 100_000,
         "production_steps": 250_000, "production_steps_target": 250_000,
         "f_k_evidence_status": "verified", "has_convergence": True}
    w.update(kw)
    return w


def _view(windows, *, units=(), ranges=((0, 4), (3, 7)), n_states=7,
          missing=(), parents_done=(), **kw):
    v = {
        "windows": list(windows),
        "path": {"window_ranges": [tuple(r) for r in ranges],
                 "n_states": n_states,
                 "lambdas_vdw": [1.0 - 0.1 * i for i in range(n_states)]},
        "path_version": 1,
        "skipped_windows": [], "skipped_sampling_units": [],
        "missing_windows": list(missing),
        "n_windows_expected": len(ranges) + len(missing),
        "n_windows_found": len(windows),
        "out_of_range_windows": [],
        "stage_analysis_status": "ANALYSIS_INCOMPLETE",
        "has_stage_result": False,
        "immutable_rewindow": {"parents_done": list(parents_done)},
        "sampling_units": list(units),
        "production_blocks_by_window": {}, "production_blocks_total_by_window": {},
        "production_blocks_by_unit": {}, "production_blocks_total_by_unit": {},
        "max_production_blocks_per_window": 4,
        "production_budget": {},
        "noop_actions": {}, "min_n_eff_over_g_history": {}, "fk_probe": {},
        "stale_layout_evidence": {}, "production_rescue_targets": {},
    }
    v.update(kw)
    return v


def _unit(**kw):
    u = {"unit_id": "u0a", "parent_window": 0, "local_index": 0,
         "needs_frames": False, "schedulable": True,
         "self_verdict": "HARD_INSUFFICIENT", "range": (0, 2), "identity": "abc"}
    u.update(kw)
    return u


# --------------------------------------------------------------------- ①


def test_plan_never_pairs_no_action_with_a_non_terminal_exit():
    """换 Epoch 付不起 + 归因不是样本量类 + 有界重窗不可行 ⟹ 必须是**终态**。

    这条路径先前配的是 `HALT_BUDGET`（`EXIT_SPECS` 里 `terminal=False`）。
    盘面构造：窗口 0 的 f_k 被统计驳回 ⟹ 分支 2 发 `RECALIBRATE_FK`；
    它的 warmup 余额（1000 步）低于冻结验证阶梯首档 ⟹ `plan()` 的换 Epoch
    预检判付不起；自检来源缺失 ⟹ 归因 `UNKNOWN`（不授权降级补帧）；
    父窗已经建过子系综 ⟹ 有界重窗也不可行。
    """
    # ⚠️ 让有界重窗不可行要用 `parents_abandoned` + 尝试次数封顶，**不能**用
    # `parents_done` —— 后者同时把这个窗口移出 `earliest` 排序（CTL-03：被子系综
    # 取代的父窗不参与路由），于是分支 2 根本不触发，盘面就不是要测的那个了。
    v = _view(
        [_w(0, verdict="STATISTICALLY_REJECTED", self_verdict="HARD_INSUFFICIENT",
            warmup_steps_left=1000, f_k_evidence_status="refuted"),
         _w(1)],
        immutable_rewindow={"parents_done": [], "parents_abandoned": [0],
                            "attempts_by_parent": {"0": 2}},
    )
    p = _ctl()._decide_once(v)
    # 先钉「确实走到了那道改写闸」，否则这个用例可能从别的分支空过。
    assert "付不起新 Epoch 的最低验证额度" in p["reason"], (
        f"没走到换 Epoch 预检那道改写闸，本用例失去意义：{p['reason'][:200]}"
    )
    assert p["action"] == "NO_ACTION"
    assert p["exit"] in Stage2RepairController.TERMINAL_EXITS, (
        f"`NO_ACTION` 配了非终态出口 {p['exit']!r} ⟹ 主循环不会 break，"
        "执行器会收到一个它不认识的动作，整跑以 exit=None 收场"
    )
    assert p["terminal"] is True
    assert p["halt_scope"] == "TARGET_LOCAL", "卡死的是这个窗口，别处该照常可修"


def test_every_terminal_exit_declares_itself_terminal_in_one_place():
    """出口的 terminal 语义只许在 `EXIT_SPECS` 声明一次（两张表都从它派生）。"""
    assert set(Stage2RepairController.TERMINAL_EXITS) == {
        k for k, spec in pre.EXIT_SPECS.items() if spec.terminal
    }
    assert "HALT_BUDGET" not in Stage2RepairController.TERMINAL_EXITS, (
        "`HALT_BUDGET` 是路由信号（动作仍要执行），不是终态"
    )


# ------------------------------------------------------------------- ②③


def test_d3_blames_the_parent_window_not_whatever_is_earliest():
    """`D3_REWINDOW_DEPTH_EXHAUSTED` 的 `blocking_window` 必须是**它自己的父窗**。

    分支的触发条件只要求 `父窗 <= earliest`，所以兜底成 `earliest` 时会指向一个
    与这条终态无关的窗口 —— 退役据此退掉无辜窗口，而且 `_skew_units` 先前不随
    `retired` 变化，下一轮再退一个，直到全窗退役。

    （`SUPPORT_ATTRIBUTION_UNKNOWN` 那条按设计只在 `earliest is None` 时触发，
    构造不出「父窗 < earliest」，它的回归由下面那个用例覆盖。）
    """
    v = _view([_w(0), _w(1, self_verdict="INSUFFICIENT_DATA",
                         phase="WARMUP_VALIDATE")],
              units=[_unit(support_attribution=SUPPORT_FAILURE_STRUCTURAL)],
              ranges=((0, 4), (3, 7), (6, 10)), n_states=10,
              parents_done=[0])
    p = _ctl()._decide_once(v)
    assert p["exit"] == "D3_REWINDOW_DEPTH_EXHAUSTED"
    assert p["earliest_unresolved_window"] == 1, "盘面前提：earliest 是别的窗口"
    assert p["blocking_window"] == 0, (
        "子窗终态点名的必须是它的父窗；兜底成 earliest 会退役无辜窗口"
    )


@pytest.mark.parametrize("attr", [
    {"attribution_unknown": True},
    {"support_attribution": SUPPORT_FAILURE_STRUCTURAL},
])
def test_a_dead_subwindow_does_not_strand_a_never_sampled_window(attr):
    """子窗修不动 ⟹ 退役父窗、去跑那个**一步都没跑过**的窗口，而不是终止整跑。

    这是 S2-G 那条兜底的语义（用户规格：「若还有其他窗口可做，则绕过该单元继续
    调度，不能让它停掉整跑」）。先前两处一起失效：`blocking_window` 是 `None`
    ⟹ 退役判定第一行就返回；就算退了，分支不看 `retired` ⟹ 原样再触发。

    后果不是「少跑一个窗口」：缺窗口的和**不是 ΔG**，而且退出前那次收尾 ANALYZE
    也会因为 `missing_windows` 非空被跳过 ⟹ 一份 stage 结果都拿不到。
    """
    v = _view([_w(0), _w(1)], units=[_unit(**attr)],
              missing=[2], parents_done=[0])
    p = _ctl().decide(v)
    assert p["action"] == "RUN_PRODUCTION"
    assert p["windows"] == [2], "该去跑布局里那个还没有任何产物的窗口"
    assert p["retired_windows"] == [0], "被退役的是子窗的父窗"
    assert not p["terminal"]


def test_a_subwindow_that_hits_its_block_cap_also_retires_the_parent():
    """1c 的子窗补帧被准入闸改写成终态时，同样要点名父窗。

    动作发出去的时候是 `RUN_PRODUCTION`，看着不像终态 —— 但 `plan()` 的补帧准入
    闸 / no-op 台账会把它就地改写成 `NO_ACTION` + 终态。到那一步 `blocking_window`
    才被用上，而它兜底成 `earliest`。与 1c-2 / 1c-3 同一形状的第三处。
    """
    v = _view([_w(0), _w(1)],
              units=[_unit(needs_frames=True, has_convergence=True)],
              missing=[2], parents_done=[0])
    v["production_blocks_total_by_unit"] = {"u0a": [{}, {}, {}, {}]}   # 配额打满
    c = _ctl()
    once = c._decide_once(v)
    assert once["action"] == "NO_ACTION" and once["terminal"]
    assert once["blocking_window"] == 0, "点名的必须是子窗的父窗"
    p = c.decide(v)
    assert p["action"] == "RUN_PRODUCTION" and p["windows"] == [2]
    assert p["retired_windows"] == [0]
