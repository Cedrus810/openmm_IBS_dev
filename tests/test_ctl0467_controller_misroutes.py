"""CTL-07 / CTL-06 / CTL-04①：控制器的三处误判。

* **CTL-07** held-out `REJECT` 时只问"末窗可不可拆"、不问"失败的是不是末窗"
  ⟹ 中间窗的 f_k 被驳回，却换来一次**与它无关**的尾段重分。
* **CTL-06** S2-C 取"点数最多的段"判加帧效果 ⟹ 旧段点数多时，拿**旧 f_k**
  的趋势去裁决新段。
* **CTL-04①** 预算闸把**非变异**动作（只重解已有帧、不产新帧）也按生产块收费
  ⟹ 预算紧时一个零成本的诊断动作也被拦掉。
"""
import json
import os
import pathlib
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abfe_preoptimizer import Stage2RepairController  # noqa: E402

from test_stage2_repair_controller import _mkrun, R4, FULL  # noqa: E402


# ---------------------------------------------------------------- CTL-07
def test_a_middle_window_rejected_by_heldout_is_not_answered_with_a_tail_split(
        tmp_path):
    """末窗**结构上可拆**，但失败的是中间窗 ⟹ 拆它一点用没有。"""
    w = {0: {"K": 4}, 1: {"K": 5}, 2: {"K": 9}}
    w[1] = {"K": 5, "self_verdict": "INSUFFICIENT_DATA", "min_n_eff_over_g": 3.1}
    run = _mkrun(
        tmp_path, windows=w, ranges=[(0, 4), (3, 8), (7, 16)], n_states=16,
        config={"stage2_window_min_states": 4, "stage2_window_max_states": 5,
                "max_path_insertions": 3},
        stage_result={"converged": False,
                      "cumulative_fk_residual_production": [
                          {"window_index": 1, "verdict": "FAIL_CUMULATIVE_FK",
                           "cumulative_residual_span_kJ_mol": 14.0}]},
    )
    (pathlib.Path(run) / "vanishing" / "dual_window_1_vdw_heldout.json").write_text(
        json.dumps({"verdict": "REJECT", "worst_before": 3.1, "worst_after": 2.4}))

    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] != "SPLIT_TAIL_WINDOW", plan["reason"]
    assert plan["windows"] == [1], plan["reason"]
    assert "不是末窗" in plan["reason"]


# ---------------------------------------------------------------- CTL-06
def test_marginal_gain_only_looks_at_the_current_segment(tmp_path):
    """旧段点数再多，也不许拿它的趋势裁决当前段（换段 = 换 f_k）。"""
    w = dict(FULL)
    w[0] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA", "prod": 500000,
            "min_n_eff_over_g": 3.8}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)
    # 逐轮快照：**旧段** `vanishing_9` 有 3 个点（下降趋势），
    # **当前段** `vanishing` 只有 1 个 ⟹ 判据必须判"不足两点，不判"。
    (pathlib.Path(run) / "checkpoints" / "stage2_autonomous_history.json").write_text(
        json.dumps({"iterations": [{
            "iteration": 1, "path_version": 1,
            "snapshot": [
                {"window_idx": 0, "segment": "vanishing_9",
                 "production_steps": s, "min_n_eff_over_g": v}
                for s, v in ((250000, 9.0), (500000, 6.0), (750000, 4.0))
            ],
        }]})
    )
    c = Stage2RepairController.for_physical_stage(run, "vanishing", "vdw")
    hist = c.read()["min_n_eff_over_g_history"] or {}
    pts = hist.get(0) or []
    assert len(pts) < 2, f"拿了旧段的点来判当前段：{pts}"

    # 因此不会走"加帧已被证伪 ⟹ 插 λ"那条
    assert c.decide()["action"] != "INSERT_LAMBDA"


# ---------------------------------------------------------------- CTL-04①
def test_a_non_mutating_probe_is_not_charged_a_production_block(tmp_path):
    """非变异动作只重解已有帧、不产新帧 ⟹ 预算闸不该按生产块收费。

    ⚠️ [审计 #32，2026-09-14 改写] **原来的动作对选错了。**
    旧断言要求 `_NON_SAMPLING` 里有 `"RECALIBRATE_FK"` —— 而实际派发时
    `RECALIBRATE_FK` 的 `probe_only=False`，会**开新段跑满 `n_steps_per_window`**，
    是最贵的那一个，被判零成本正好反了。真正 `probe_only=True`、零采样的是
    `PROBE_CANDIDATE_FK`；`PROBE_REANCHOR_EPOCH` 明确只花一块。
    现在预算闸改成白名单 `_PRODUCTION_CHARGED`：只有确实产生生产帧的动作进闸，
    非采样动作连闸都不进（等价于零成本，而且不用再维护一张"免单表"）。
    """
    import inspect

    import abfe_preoptimizer

    src = inspect.getsource(abfe_preoptimizer.Stage2RepairController.decide)
    assert "_PRODUCTION_CHARGED" in src
    charged = src.split("_PRODUCTION_CHARGED")[1][:300]
    # 真正零采样的探针**不在**收费名单里
    assert '"PROBE_CANDIDATE_FK"' not in charged
    # 会开新段跑满的那个**在**收费名单里
    assert '"RECALIBRATE_FK"' in charged
    # 只花一块的按一块收，不按 `_blk * n_windows`
    assert 'if action == "PROBE_REANCHOR_EPOCH":\n                    _cost = _blk' in src


def test_a_sampling_action_is_still_charged(tmp_path):
    """别把闸修成谁都不收费 —— 真采样的动作照收。"""
    cfg = {"stage2_window_min_states": 4, "stage2_window_max_states": 8,
           "max_path_insertions": 3, "n_steps_per_window": 250000,
           "stage2_production_budget_steps": 1_100_000}
    w = {i: {"K": 4, "prod": 250000} for i in range(4)}
    w[0] = {"K": 4, "prod": 250000, "self_verdict": "INSUFFICIENT_DATA",
            "min_n_eff_over_g": 3.1}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13, config=cfg)
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] == "NO_ACTION"
    assert plan["exit"] == "GLOBAL_BUDGET_EXHAUSTED"


# ---------------------------------------------------------------- CTL-04②
def test_unknown_usage_is_not_counted_as_zero(tmp_path):
    """**消耗未知 ≠ 消耗为零。** 与"上限未知 ≠ 零"是同一个错、方向相反：
    一个虚报余量、一个虚报耗尽。账不完整时余量必须是 unknown。"""
    cfg = {"stage2_window_min_states": 4, "stage2_window_max_states": 8,
           "max_path_insertions": 3, "stage2_production_budget_steps": 1_000_000}
    run = _mkrun(tmp_path, windows={i: {"K": 4, "prod": 250000} for i in range(4)},
                 ranges=R4, n_states=13, config=cfg)
    # 把 win2 的步数抹掉：证据还在（convergence 在），但读不到它烧了多少
    f = pathlib.Path(run) / "vanishing" / "dual_window_2_vdw_convergence.json"
    d = json.loads(f.read_text())
    d.pop("cumulative_production_steps", None)
    d.pop("actual_production_steps", None)
    f.write_text(json.dumps(d))

    pb = Stage2RepairController.for_physical_stage(
        run, "vanishing", "vdw").read()["production_budget"]
    assert pb["usage_complete"] is False
    assert any(x.get("window_idx") == 2 for x in pb["unknown_usage"])
    assert pb["stage_remaining_steps"] is None, "少算过的账被当成了可用余量"
    assert pb["exhausted"] is False, "账不完整时不得宣称耗尽"


def test_an_unknown_remaining_budget_does_not_block_every_action(tmp_path):
    """余量未知 ⟹ 拦不拦都没有依据，**不许当成 0 一律拦死**。"""
    cfg = {"stage2_window_min_states": 4, "stage2_window_max_states": 8,
           "max_path_insertions": 3, "stage2_production_budget_steps": 1_000_000}
    w = {i: {"K": 4, "prod": 250000} for i in range(4)}
    w[0] = {"K": 4, "prod": 250000, "self_verdict": "INSUFFICIENT_DATA",
            "min_n_eff_over_g": 3.1}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13, config=cfg)
    f = pathlib.Path(run) / "vanishing" / "dual_window_2_vdw_convergence.json"
    d = json.loads(f.read_text())
    d.pop("cumulative_production_steps", None)
    f.write_text(json.dumps(d))

    plan = Stage2RepairController.for_physical_stage(
        run, "vanishing", "vdw").decide()
    assert plan["exit"] != "GLOBAL_BUDGET_EXHAUSTED", plan["reason"]
