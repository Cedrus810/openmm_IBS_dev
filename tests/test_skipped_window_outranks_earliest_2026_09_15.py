"""被求解器跳掉的窗口优先于"自检说支撑不足"的窗口。

两者不是同一强度的信号：

  · `skipped_windows` = **求解器的操作权威**。那个窗口真的没进 MBAR ⟹ 整条路径
    缺窗 ⟹ `analysis_status = ANALYSIS_INCOMPLETE` ⟹ 交出去的和**不是 ΔG**
    （硬不变量，本仓不许放宽）。
  · `self_verdict` = 逐窗自检的诊断，说"这个窗还没测够"。

真机 brd4_ligand1/rep2：
    win1  INSUFFICIENT_DATA，n_decorr=888（够得离谱）、min N_eff/g=8.68 ⟹ **偏斜**
    win4  HARD_INSUFFICIENT，n_decorr=7  ⟹ 被求解器跳掉，skipped_windows=[4]
6 轮全部路由到 win1（4 块帧烧光配额），win4 **一次都没被看过**；整条路径照常采完，
最后死在缺窗那道身份门上。而 win4 有现成的对症动作，只是轮不到。
"""
import json
import pytest

# 🔑 [2026-09-16] 本文件原来**没有任何标记** ⟹ 日常的 `pytest -m cpu_only` 整份
# 跳过。里面全是纯 CPU 的源码契约探针，正好是最容易静默烂掉的那类（它们
# 断言"某段代码存在"，一旦指错函数就只是找不到、不报错）。实测就烂过：
# `decide()` 被拆成外壳之后这里 6 条全挂，而没人看得见。
pytestmark = pytest.mark.cpu_only

import os

from abfe_preoptimizer import Stage2RepairController

from test_stage2_repair_controller import R4, _mkrun


def _board(tmp_path):
    """win1 自检不合格（在前），win3 被求解器跳掉（在后）。"""
    w = {i: {"K": 4} for i in range(4)}
    w[1] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA", "min_n_eff_over_g": 8.7,
            "n_decorr": 888}
    w[3] = {"K": 4, "self_verdict": "HARD_INSUFFICIENT", "min_n_eff_over_g": 0.21,
            "n_decorr": 7}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13,
                 stage_result={
                     "analysis_status": "ANALYSIS_INCOMPLETE",
                     "analysis_incomplete_reasons": ["存在被跳过的窗口 [3]"],
                     "precision_status": "UNMEASURED",
                     "total_delta_G": -20.0, "total_error": 0.5,
                     "skipped_windows": [{"window_index": 3,
                                          "n_frames_after_decorrelation": 7}],
                 })
    return run


def test_a_solver_skipped_window_is_routed_before_an_earlier_unhappy_one(tmp_path):
    c = Stage2RepairController.for_physical_stage(_board(tmp_path), "vanishing", "vdw")
    view = c.read()
    assert 3 in [int(x) if not isinstance(x, dict) else int(x["window_index"])
                 for x in (view.get("skipped_windows") or [])], view.get("skipped_windows")
    plan = c.decide()
    assert plan["windows"] == [3], (
        f"路由到了 {plan['windows']}（应当是被跳掉的 win3，不是下标更小的 win1）："
        f"{plan['reason'][:200]}")


def test_without_a_skip_the_old_earliest_order_is_unchanged(tmp_path):
    """没有跳窗时 `earliest` 仍按下标排 —— 这条改的只是优先级，不是判据。

    ⚠️ [2026-09-17 P0] 断言从「动作落在 w1」改成「`earliest` 仍是 w1」。
    三态归因之后 w1（ratio=8.7、门 10、headroom=5 ⟹ 8.7×5=43.5 ≥ 10 ⟹
    `reachable is True`）是 **UNKNOWN** —— 乐观上界尚未被证伪，**既不是**"帧不够"
    **也不是**结构性失败 ⟹ 两边都不授权动作。按用户规格「若还有其他窗口可做，
    则绕过该单元继续调度，不能让它停掉整跑」，它被退役，动作落到 w3
    （ratio=0.21 ⟹ 0.21×5=1.05 < 10 ⟹ `reachable is False` ⟹ STRUCTURAL）。

    **优先级本身没变**（`earliest` 照样是 w1），变的是"w1 现在没有被授权的动作"。
    本条守的是前者，所以断言改成直接看 `earliest_unresolved_window`。
    """
    w = {i: {"K": 4} for i in range(4)}
    w[1] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA", "min_n_eff_over_g": 8.7}
    w[3] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA", "min_n_eff_over_g": 0.21}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)
    plan = Stage2RepairController.for_physical_stage(run, "vanishing", "vdw").decide()
    # ⚠️ `decide()` 会退役后**重判**，所以返回值里的 `earliest` 已经是 3。
    # "优先级没变"体现在：w1 是**先**被考虑的那个（因此进了 retired），
    # 而不是被跳过去没看。
    assert plan.get("retired_windows") == [1], plan["reason"][:200]
    assert plan["windows"] == [3], plan["reason"][:200]


def test_a_replaced_parent_is_still_excluded(tmp_path):
    """`_replaced_parents` 的排除不得被这条优先级顺手绕过。"""
    import inspect
# 🔑 [2026-09] `decide()` 现在只是 23 行的外壳（"退役一个窗口再判一次"），判断体是 `_decide_once`（1831 行）。
# 源码探针指着 `decide` 会一无所获 —— 断言"存在"的当场红，断言"不存在"的**静默变成假绿**。
    src = (inspect.getsource(Stage2RepairController.decide)
           + inspect.getsource(Stage2RepairController._decide_once))
    blk = src.split("_skipped_now = {")[1].split("if earliest is None:")[0]
    assert "_replaced_parents" in blk, "跳窗优先分支漏了 _replaced_parents 排除"
