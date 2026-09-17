"""换 Epoch 付不起时，**偏斜类**窗口不得被降级去补帧。

`docs/archive/采样问题_2026-09-15.md` 上半部分那条链的落点：

    warmup 账干 → 换 Epoch 付不起 → 降级发 RUN_PRODUCTION（复用冻结 f_k）
      → 生产预算判不了它不可行 → 连补至 max_production_blocks=4
        → NO_FEASIBLE_ACTION，窗口仍未达标

实测：三个 rep **一次都没重标定过**，各自把 1.25M 步砸在一份冻结 f_k 上。

降级的原意是「换 Epoch 付不起 ⟹ 退而做一件便宜且有用的事」。但对偏斜类窗口，
补帧是**便宜且无用**：它拿的正是那份需要被换掉的冻结 f_k，采出来的帧权重剖面
一模一样（§5.1 实测同分布 250k→1M 让 top1% 0.545→0.762、ESS 比值反而更差）。

⚠️ **样本量类仍然降级**——那些窗口是真的帧不够，补帧对症、也确实便宜。
"""
import pytest

# 🔑 [2026-09-16] 本文件原来**没有任何标记** ⟹ 日常的 `pytest -m cpu_only` 整份
# 跳过。里面是纯 CPU 的源码契约探针，正好是最容易静默烂掉的那类（断言"某段
# 代码存在"，一旦指错函数就只是找不到、不报错）。实测就烂过：`decide()` 被
# 拆成外壳之后这里全挂，而没人看得见。
pytestmark = pytest.mark.cpu_only

from abfe_preoptimizer import Stage2RepairController

from test_stage2_repair_controller import R4, _mkrun


def _board(tmp_path, source):
    """win0 预热账全干（付不起换 Epoch），自检不合格，归因由 `source` 决定。"""
    w = {i: {"K": 4} for i in range(4)}
    w[0] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA", "min_n_eff_over_g": 3.1,
            "self_verdict_source": source,
            "n_decorr": 888 if source == "min_n_eff_over_g" else 3,
            "evidence": "indeterminate",
            "bias_status": "frozen_validation_indeterminate",
            "warmup": 555000, "cap": 555000}
    return _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)


def test_a_skew_window_is_not_handed_frames_as_a_consolation_prize(tmp_path):
    plan = Stage2RepairController(
        _board(tmp_path, "min_n_eff_over_g"), "vanishing").decide()
    assert plan["action"] != "RUN_PRODUCTION", (
        f"偏斜窗口又被降级去补帧了：{plan['reason'][:260]}")


def test_a_frame_starved_window_still_gets_the_cheap_fallback(tmp_path):
    """样本量类**保持**降级 —— 这条改的是适用范围，不是把降级删掉。"""
    plan = Stage2RepairController(
        _board(tmp_path, "solver_eligibility"), "vanishing").decide()
    assert plan["action"] in ("RUN_PRODUCTION", "INSERT_LAMBDA", "NO_ACTION"), plan["action"]
    # 只要它没被上面那条偏斜分支拦掉即可（具体动作由更前面的分支决定）
    assert "支撑/偏斜类" not in plan["reason"] or plan["action"] != "NO_ACTION"


def test_the_attribution_reuses_the_shared_implementation():
    """不许在这处另写一套归因 —— 那是本仓最贵的那类 bug。"""
    import inspect
# 🔑 [2026-09] `decide()` 现在只是 23 行的外壳（"退役一个窗口再判一次"），判断体是 `_decide_once`（1831 行）。
# 源码探针指着 `decide` 会一无所获 —— 断言"存在"的当场红，断言"不存在"的**静默变成假绿**。
    src = (inspect.getsource(Stage2RepairController.decide)
           + inspect.getsource(Stage2RepairController._decide_once))
    blk = src.split("_EPOCH_ACTIONS = (")[1].split("if action == \"RUN_PRODUCTION\":")[0]
    # 🔑 [2026-09-17] 字面量从 `support_failure_is_skew(` 改成
    # `support_failure_attribution(`：O1 改成了**三态**归因（用户拍板 A）——
    # `is_skew` 只区分两态，于是 `UNKNOWN`（射程只说明「还没被证伪」）被算进了
    # "样本量类"，一个乐观上界就足以把「f_k 不对，换 Epoch」静默改写成「补帧」。
    # `support_failure_is_skew()` 现在就是 `attribution(...) == STRUCTURAL` 的薄包装，
    # 所以**本条的意图一字未变**：O1 用的仍是那份共享实现，没有在这里另写一套。
    assert "support_failure_attribution(" in blk
    assert "n_decorrelated=" in blk
    # 反过来钉住：别在这块里出现自造的归因判据（那才是本条要防的）
    for forbidden in ("top1pct", "HARD_INSUFFICIENT", "_SAMPLE_SIZE_VERDICT_SOURCES"):
        assert forbidden not in blk, (
            f"O1 里出现了自造的归因判据 `{forbidden}` —— 归因只许走共享实现")
