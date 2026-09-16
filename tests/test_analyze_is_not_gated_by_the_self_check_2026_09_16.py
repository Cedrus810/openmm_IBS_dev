"""5b 的偏斜死胡同不得在**证据从未做出来**时就判 `NO_FEASIBLE_ACTION`。

真机 cmet_ligand2/rep1 w4：`skipped_windows=[]`、去相关 **69** 帧（min_frames=10）
⟹ 求解器**确实**把这个窗口算进了 MBAR，累计 f_k 残差证据完全算得出来。但：

  · 5a-1（ANALYZE）的 guard 里有 `self_sufficient is not False` —— 那是**自检侧**
    的量，CTL-11 已裁定逐窗自检「只看单段帧、对多段窗口系统性偏悲观，**它不是
    权威**」。偏斜类窗口必然 `sufficient=False` ⟹ 拿不到 ANALYZE；
  · 于是 `cumulative_fk_residual_production` 永不出现 ⟹ 5a-2 的 guard
    （`cum_fk_verdict ∈ {FAIL_CUMULATIVE_FK, UNMEASURED}`）也永不匹配 ⟹ 整条
    「判累计偏差 → 生成候选 → held-out 验收 → 换 Epoch / 缩跨度」的链对它
    **结构上不可达**；
  · 只能掉进 5b 偏斜分支，布局动作一不可行就 `NO_FEASIBLE_ACTION` 收摊。

把「没查」说成「无路可走」。修法补在**死胡同本身**（而不是放宽 5a-1 的 guard ——
5a-1 排在边际增长判据和所有布局动作之前，放宽它会把「先 ANALYZE」插到全仓每一条
路由前面，实测打断 `加帧被证伪 ⟹ 插 λ` 等既有优先级）。
"""
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abfe_preoptimizer import Stage2RepairController  # noqa: E402

from test_stage2_repair_controller import _mkrun, R4, FULL  # noqa: E402


def _dead_end_board(tmp_path, *, n_decorr=69):
    """偏斜类失败 + 两个缩跨度动作都不可行 + 累计残差证据从未产出。

    末窗 K=4 ⟹ 落在可拆区间 [2·min−1, 2·max−1]=[7,15] 之外 ⟹ 拆窗不可行；
    `max_path_insertions=0` ⟹ 插 λ 不可行。这正是 rep1 w4 的处境。
    """
    w = dict(FULL)
    w[3] = {"K": 4, "self_verdict": "HARD_INSUFFICIENT",
            "min_n_eff_over_g": 0.96, "n_decorr": n_decorr}
    return _mkrun(
        tmp_path, windows=w, ranges=R4, n_states=13,
        config={"stage2_window_min_states": 4, "stage2_window_max_states": 8,
                "max_path_insertions": 0},
    )


def test_the_dead_end_runs_analyze_before_declaring_no_feasible_action(tmp_path):
    """证据算得出来却从没算过 ⟹ 先零采样地算出来，不许直接判无路可走。"""
    ctl = Stage2RepairController(_dead_end_board(tmp_path), "vanishing")
    view = ctl.read()
    # 前提钉死，否则测的就不是这件事：求解器**没有**跳过它，且证据确实缺失。
    assert not (view.get("skipped_windows") or []), view.get("skipped_windows")
    w3 = next(w for w in view["windows"] if w["window_idx"] == 3)
    assert w3["self_sufficient"] is False and w3.get("cum_fk_verdict") is None

    plan = ctl.decide(view)
    assert plan["action"] == "ANALYZE", plan["reason"]
    assert plan.get("exit") != "NO_FEASIBLE_ACTION", plan["reason"]


def test_a_window_the_solver_actually_dropped_still_stops(tmp_path):
    """反向守卫：求解器真够不到的窗口**仍然**如实停下。

    `skipped_windows` 是求解器的操作权威。那种窗口跑 ANALYZE 只会空转 —— 正是
    5a-1 的长注释拿真机 win0（9 帧 < 10、连发 4 次 ANALYZE）举证禁止的事。
    """
    ctl = Stage2RepairController(_dead_end_board(tmp_path, n_decorr=7), "vanishing")
    view = ctl.read()
    view["skipped_windows"] = [3]
    plan = ctl.decide(view)
    assert plan["action"] != "ANALYZE", plan["reason"]
