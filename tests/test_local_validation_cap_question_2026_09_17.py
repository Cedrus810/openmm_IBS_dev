"""`LOCAL_VALIDATION_CAP` 是一个**提问**，不许被静默丢掉（2026-09-17 真机）。

真机日志（p38 那一跑）：

    窗口 4 冻结验证 15/15 批用尽（600 frames、175000 步）⟹ insufficient_frames
    …交上层决定是否延长预算…交顶层自治控制器重判
    [自治 1/40] 动作=RUN_PRODUCTION 窗口=[0] 阻塞下游=[1, 2, 3, 4]

引擎**明确把一个决定上交**给控制器；控制器**有**答案（分支 3a：
`PROVISIONAL_PRODUCTION`）；但那条分支走 `_pick()` ⟹ 只对 `earliest` 生效 ⟹
`earliest=0`、cap-hit=[4] ⟹ `0 not in [4]` ⟹ **整条分支不触发**。
于是 17.5 万步烧掉、窗口 4 问"现在怎么办"、回答是 `blocked_by_upstream`。
下一轮同样。**不是拒绝，是连问题都没被看见。**

⚠️ 本文件**不要求改路由**：押后的理由是真的（上游重锚会让下游取证落在即将
作废的布局上，分支 3a 自己的注释就警告过）。要求的只有一条 ——
**提问必须有回应**，押后也要说出来、说清为什么。
"""
import json
import pathlib

import pytest

pytestmark = pytest.mark.cpu_only

import ibs_engine as ie
from abfe_preoptimizer import Stage2RepairController
from test_stage2_repair_controller import _mkrun, R4

CAP = int(ie.IBS_LOCAL_MBAR_GATE_MAX_BATCHES)


def _board(tmp_path, *, w0_problem=True):
    """窗口 4 打满验证批次；窗口 0 另有问题（于是它才是 earliest）。"""
    w = {i: {"K": 4} for i in range(5)}
    if w0_problem:
        w[0] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA",
                "self_verdict_source": "solver_eligibility", "n_decorr": 5}
    # 窗口 4：还在冻结验证、批次打满、但预热账面还有钱
    w[4] = {"K": 4, "bias_status": "frozen_validation_indeterminate",
            "evidence": "indeterminate",
            "warmup": 130000, "cap": 500000}
    run = _mkrun(tmp_path, windows=w, ranges=[(0, 4), (3, 7), (6, 10),
                                              (9, 13), (12, 16)], n_states=16)
    st = pathlib.Path(run) / "checkpoints" / "ibs_state_vdw_window_4.json"
    d = json.loads(st.read_text())
    # ⚠️ 盘上的真名是 `frozen_validation_batches_done`；视图键才叫
    # `frozen_validation_batches`。写错这一个字母整条判据静默失效
    # （"读一个没人写的键"是本仓反复栽的形状）。
    d["frozen_validation_batches_done"] = CAP
    st.write_text(json.dumps(d))
    return run


def test_the_cap_question_is_visible_even_when_another_window_is_earliest(tmp_path):
    """要害：窗口 4 提了问，本轮路由在窗口 0 —— **提问必须仍然出现在 plan 里**。"""
    c = Stage2RepairController(_board(tmp_path), "vanishing")
    view = c.read()
    hits = c.local_validation_cap_hits(view)
    assert [h["window_idx"] for h in hits] == [4], (
        f"共享谓词没认出提问窗口：{hits}")

    plan = c.decide()
    pend = plan["local_validation_cap_pending"]
    assert [p["window_idx"] for p in pend] == [4], (
        f"窗口 4 的 LOCAL_VALIDATION_CAP 提问被丢掉了：{plan['reason']}")
    q = pend[0]
    assert q["frozen_validation_batches"] == CAP and q["batch_cap"] == CAP
    if plan["action"] == "PROVISIONAL_PRODUCTION" and plan["windows"] == [4]:
        assert q["answered_this_round"] is True
        assert q["deferred_because"] is None
    else:
        # 押后是允许的，但**必须说出理由**，而且理由要指名是谁挡着
        assert q["answered_this_round"] is False
        assert q["deferred_because"], "押后却没给理由 —— 那跟丢掉没区别"
        assert "押后，不是拒绝" in q["deferred_because"], q["deferred_because"]


def test_when_the_asking_window_is_earliest_it_actually_gets_answered(tmp_path):
    """反面：提问窗口自己就是 earliest 时，必须**真的**发 `PROVISIONAL_PRODUCTION`。

    没有这条，上面那条可以靠「永远押后」作弊通过。
    这也说明押后不会变成永远：`earliest` 无路可走会被退役，届时提问窗口
    自己成为 earliest，分支 3a 就触发。
    """
    plan = Stage2RepairController(
        _board(tmp_path, w0_problem=False), "vanishing").decide()
    assert plan["action"] == "PROVISIONAL_PRODUCTION", plan["reason"]
    assert plan["windows"] == [4], plan["reason"]
    assert plan["exit"] == "HALT_LOCAL_VALIDATION_CAP"
    q = plan["local_validation_cap_pending"][0]
    assert q["answered_this_round"] is True and q["deferred_because"] is None


def test_a_window_already_on_provisional_production_is_not_asking_again(tmp_path):
    """已经拿到答案（在跑临时生产）的窗口不再算"提问" —— 否则每轮都报一次假待办。"""
    run = _board(tmp_path)
    st = pathlib.Path(run) / "checkpoints" / "ibs_state_vdw_window_4.json"
    d = json.loads(st.read_text())
    d["bias_status"] = "provisional_production"
    st.write_text(json.dumps(d))
    c = Stage2RepairController(run, "vanishing")
    assert c.local_validation_cap_hits(c.read()) == []
