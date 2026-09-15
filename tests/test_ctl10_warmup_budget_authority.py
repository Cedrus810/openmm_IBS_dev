"""CTL-10：「还剩多少预热预算」有两份记录，控制器必须读**权威**那份。

`convergence.json` 的 ledger 是**生产跑完那一刻**写的；之后窗口还会继续消耗
（续跑再进一次预热、跨段继承），最新值在 `*_warmup_failure.json`，引擎自己还
算好了 `warmup_budget_remaining_steps`。

先前 `warm = conv.bias_warmup or fail...` **优先取 convergence** ⟹ 读到旧账。
实测 `cyclod_ligand2/rep2` win4：convergence 说 `left=140000`，引擎说 `remaining=0`。
于是控制器以为"预算有余"一直发动作，引擎每次以「本次可用 0 步」弹回 ——
**连发 40 轮、盘面一字节没变**，分支 1e 的 `left <= 0` 永远不触发。
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


def _run_with_stale_ledger(tmp_path, *, engine_remaining=0):
    """convergence 的账说还剩 140000，引擎说剩 `engine_remaining`。"""
    w = dict(FULL)
    w[0] = {"K": 4, "bias_status": "frozen_validation_indeterminate",
            "evidence": "indeterminate",
            # convergence ledger：learning=815000 / cap=955000 ⟹ 看着还剩 140000
            "warmup": 815000, "cap": 955000}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)
    (pathlib.Path(run) / "vanishing"
     / "dual_window_0_vdw_warmup_failure.json").write_text(json.dumps({
         "window_index": 0,
         "warmup_budget_remaining_steps": engine_remaining,
         "bias_warmup": {
             "warmup_budget_remaining_steps": engine_remaining,
             "warmup_budget_ledger": {
                 "learning_steps": 140000, "freeze_burn_in_steps": 40000,
                 "frozen_validation_steps": 775000,
                 "cumulative_cap_steps": 955000, "complete": True},
         },
     }))
    return run


def test_the_engine_reading_wins_over_the_stale_convergence_ledger(tmp_path):
    run = _run_with_stale_ledger(tmp_path, engine_remaining=0)
    w0 = next(x for x in Stage2RepairController(run, "vanishing").read()["windows"]
              if x["window_idx"] == 0)
    assert w0["warmup_steps_left"] == 0, "又读了 convergence 里那份过期账"
    assert w0["warmup_steps_left_source"] == "engine:warmup_budget_remaining_steps"


def test_an_exhausted_budget_now_actually_routes_away_from_topping_up(tmp_path):
    """这才是那 40 轮空转的出口：预算真耗尽 ⟹ 不再发补帧/续预热。"""
    run = _run_with_stale_ledger(tmp_path, engine_remaining=0)
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] not in ("RUN_PRODUCTION", "CONTINUE_WARMUP"), plan["reason"]
    assert "预热门上被" in plan["reason"] or "弹回" in plan["reason"]


def test_the_more_consumed_ledger_wins_when_the_engine_number_is_absent(tmp_path):
    """引擎读数拿不到 ⟹ 两份 ledger 取**消耗更多**的那份（保守，不虚报余量）。"""
    w = dict(FULL)
    w[0] = {"K": 4, "bias_status": "frozen_validation_indeterminate",
            "evidence": "indeterminate", "warmup": 815000, "cap": 955000}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)
    (pathlib.Path(run) / "vanishing"
     / "dual_window_0_vdw_warmup_failure.json").write_text(json.dumps({
         "window_index": 0,
         "bias_warmup": {"warmup_budget_ledger": {
             "learning_steps": 140000, "freeze_burn_in_steps": 40000,
             "frozen_validation_steps": 775000,
             "cumulative_cap_steps": 955000}},
     }))
    w0 = next(x for x in Stage2RepairController(run, "vanishing").read()["windows"]
              if x["window_idx"] == 0)
    assert w0["warmup_steps_left"] == 0          # 955000 - 955000，不是 140000
    assert w0["warmup_steps_left_source"] == "ledger:cap-spent"


def test_two_readings_take_the_more_conservative_one(tmp_path):
    """两边都有读数 ⟹ 取**更小**的那个，绝不虚报余量。

    fixture 里引擎说还剩 300000，而 warmup_failure 的 ledger 明写已耗满
    （955000/955000）⟹ 结论必须是 0。
    """
    run = _run_with_stale_ledger(tmp_path, engine_remaining=300000)
    w0 = next(x for x in Stage2RepairController(run, "vanishing").read()["windows"]
              if x["window_idx"] == 0)
    assert w0["warmup_steps_left"] == 0


def test_a_window_with_real_budget_left_is_not_blocked(tmp_path):
    """真的还有预算的窗口不该被这条改动误伤。"""
    import json as _json

    w = dict(FULL)
    w[0] = {"K": 4, "bias_status": "frozen_validation_indeterminate",
            "evidence": "indeterminate", "warmup": 500000, "cap": 955000}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)
    (pathlib.Path(run) / "vanishing"
     / "dual_window_0_vdw_warmup_failure.json").write_text(_json.dumps({
         "window_index": 0,
         "bias_warmup": {
             "warmup_budget_remaining_steps": 455000,
             "warmup_budget_ledger": {
                 "learning_steps": 500000, "cumulative_cap_steps": 955000}},
     }))
    w0 = next(x for x in Stage2RepairController(run, "vanishing").read()["windows"]
              if x["window_idx"] == 0)
    assert w0["warmup_steps_left"] == 455000
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] in ("CONTINUE_WARMUP", "RUN_PRODUCTION"), plan["reason"]
