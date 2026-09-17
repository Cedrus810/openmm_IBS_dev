"""不变量：**没有任何窗口信号会被静默丢掉**（2026-09-17）。

`_pick()` 只服务 `earliest`，目标窗口不在候选里就返空。这对「挑哪个窗口干活」
那类分支是对的（上游解决后该窗口自己会成为 `earliest`），但对**路由态 `ELIGIBLE`
的窗口**，押后 = **永远轮不到**：`earliest` 取第一个 `PROBLEM`，`ELIGIBLE` 永不当选；
而全窗合格时 `earliest is None` ⟹ `_pick()` 恒返空 ⟹ 每一条分支都不触发。

真机触发这条的是用户 2026-09-17 那一跑：窗口 4 烧完 17.5 万步验证预算、
打出 `LOCAL_VALIDATION_CAP`（引擎日志写着「交上层决定」），而控制器回了一句
`blocked_by_upstream` —— **提问连被看见都没有**。

本文件钉的是**类级不变量**，不是那几个具体分支：
    对任何一个窗口，只要它带着一个「需要控制器决定」的信号，
    那么要么它的路由态是 PROBLEM（⟹ 迟早成为 earliest、会被处理），
    要么它出现在 plan 的某个常设字段里（⟹ 至少看得见）。
两者都不成立 = 信号被静默丢掉 = 本条红。
"""
import json
import pathlib

import pytest

pytestmark = pytest.mark.cpu_only

import ibs_engine as ie
from abfe_preoptimizer import Stage2RepairController as C
from test_stage2_repair_controller import _mkrun, R4

CAP = int(ie.IBS_LOCAL_MBAR_GATE_MAX_BATCHES)

# 每个用例：窗口 3 带一个信号，窗口 0 另有问题（于是 0 才是 earliest）。
# `state_patch` / `conv_patch` 用于那些 `_mkrun` 造不出的真实键。
CASES = [
    ("身份不一致",        {"K": 4}, {"stage_protocol_key": "AAA"}, {"stage_protocol_key": "BBB"}),
    ("f_k 被统计驳回",    {"K": 4, "evidence": "refuted"}, None, None),
    ("验证算术不可达",    {"K": 4, "bias_status": "calibrated_pending_validation",
                          "evidence": "indeterminate", "warmup": 130000, "cap": 140000}, None, None),
    ("验证批次打满",      {"K": 4, "bias_status": "frozen_validation_indeterminate",
                          "evidence": "indeterminate", "warmup": 130000, "cap": 500000},
                         {"frozen_validation_batches_done": CAP}, None),
    ("预热卡住",          {"K": 4, "bias_status": "unconverged", "evidence": "none",
                          "warmup": 0, "cap": 100000}, None, None),
    ("生产步数未达标",    {"K": 4, "prod": 100000, "prod_target": 250000}, None, None),
    ("自检偏斜类",        {"K": 4, "self_verdict": "HARD_INSUFFICIENT",
                          "self_verdict_source": "top1pct_veto"}, None, None),
    ("自检样本量类",      {"K": 4, "self_verdict": "INSUFFICIENT_DATA",
                          "self_verdict_source": "solver_eligibility", "n_decorr": 5}, None, None),
]

# plan 里所有"常设可见性字段"。新增一个就加进来 —— 它们是这条不变量的另一半。
_VISIBILITY_FIELDS = (
    "unrouted_window_signals",
    "local_validation_cap_pending",
    "terminal_window_failures",
)


def _board(tmp_path, w3, state_patch, conv_patch):
    w = {i: {"K": 4} for i in range(4)}
    w[0] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA",
            "self_verdict_source": "solver_eligibility", "n_decorr": 5}
    w[3] = dict(w3)
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)
    for rel, patch in (("checkpoints/ibs_state_vdw_window_3.json", state_patch),
                       ("vanishing/dual_window_3_vdw_convergence.json", conv_patch)):
        if patch:
            f = pathlib.Path(run) / rel
            d = json.loads(f.read_text()); d.update(patch); f.write_text(json.dumps(d))
    return run


@pytest.mark.parametrize("name,w3,sp,cp", CASES, ids=[c[0] for c in CASES])
def test_a_non_earliest_signal_is_either_routable_or_visible(tmp_path, name, w3, sp, cp):
    c = C(_board(tmp_path, w3, sp, cp), "vanishing")
    view = c.read()
    rec = next(x for x in view["windows"] if int(x["window_idx"]) == 3)
    plan = c.decide()

    # A. 路由态是 PROBLEM ⟹ 它迟早会成为 earliest，押后不丢
    routable = C._window_routing_state(rec) == "PROBLEM"
    # B. 否则它必须出现在某个常设可见性字段里
    visible = any(
        any(int(e.get("window_idx", -1)) == 3 for e in (plan.get(f) or []))
        for f in _VISIBILITY_FIELDS
    )
    assert routable or visible, (
        f"窗口 3 带着「{name}」信号，却既不会成为 earliest"
        f"（路由态={C._window_routing_state(rec)}）、也不出现在任何常设字段里 "
        f"⟹ **被静默丢掉**。本轮动作={plan['action']}{plan['windows']}"
    )


def test_the_visibility_fields_exist_on_every_plan(tmp_path):
    """常设字段必须**恒常存在**（哪怕是空列表）——否则消费侧只能靠 `.get()` 猜。"""
    run = _mkrun(tmp_path, windows={i: {"K": 4} for i in range(4)},
                 ranges=R4, n_states=13)
    plan = C(run, "vanishing").decide()
    for f in _VISIBILITY_FIELDS:
        assert f in plan, f
        assert isinstance(plan[f], list), f


def test_a_below_target_window_that_can_never_be_routed_is_reported(tmp_path):
    """具体那一条：自检 ELIGIBLE + 生产步数未达标 ⟹ 永远轮不到，必须报出来。

    全窗合格时 `earliest is None` ⟹ `_pick()` 恒返空 ⟹ 补足生产那条分支一次都不
    触发 ⟹ run 会带着一个**没跑到目标步数**的窗口直接判完成。
    """
    run = _board(tmp_path, {"K": 4, "prod": 100000, "prod_target": 250000}, None, None)
    plan = C(run, "vanishing").decide()
    sig = [e for e in plan["unrouted_window_signals"]
           if e["signal"] == "PRODUCTION_BELOW_TARGET"]
    assert [e["window_idx"] for e in sig] == [3], plan["unrouted_window_signals"]
    assert sig[0]["production_steps"] == 100000 and sig[0]["target"] == 250000
    assert "永远不会成为" in sig[0]["why_never_routed"]
