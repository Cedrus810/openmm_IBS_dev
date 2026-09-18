"""S2-N + S2-O：块账的**写侧与读侧必须是同一张表**；停下时必须留诊断。

S2-N —— `_BLOCK_CHARGING_ACTIONS` 声明 4 个动作消耗一个生产块，而 `plan()` 里那道
块数硬上限闸先前只挂在 `RUN_PRODUCTION` 上 ⟹ 重标定 / 探针 / 临时生产**扣配额却
从不过闸**。真机两例（2026-09-18 benchmark 批次）：

    cmet_ligand2/rep3  w4：2 块 RUN_PRODUCTION + 3 块 RECALIBRATE_FK = 5 块（上限 4）
    cyclod_ligand3/rep3 w1：同形，轮 15 又把累计 500k 作废回 250k

它们吃掉的正是那个窗口稍后**重采**要用的配额 —— 而重采是插 λ 作废下游之后唯一的
出路，于是 `NO_FEASIBLE_ACTION` → 收尾 ANALYZE 被跳过 → 整跑零产出。

⚠️ 但**三道闸不能整套套给它们**，只有第一道（块数硬上限）适用：第二道「上一块补帧
没带来增益」与第三道「剩余配额够不着门」问的都是**同分布加帧还有没有用**，而
`RECALIBRATE_FK` 恰恰是「加帧没用」时的**对症动作**。拿"加帧没用"去否决它，方向完全
反了 —— 那会把唯一对症的那条路掐掉。本文件第三条测试钉的就是这个方向。

全文：docs/archive/STAGE2_BENCHMARK_CRASH_TRIAGE_2026-09-18.md
"""
import inspect
import json
import os
import pathlib
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abfe_preoptimizer import Stage2RepairController  # noqa: E402

from test_stage2_repair_controller import _mkrun, R4, FULL  # noqa: E402


def _run(tmp_path, *, blocks, cap=4):
    """造一个「窗口 0 被 f_k 探针点名重标定」的盘面。

    `blocks` = 逐块的 `(production_steps, solver_n_decorrelated)`，记在窗口 0 名下。
    动作一律记成 `RUN_PRODUCTION` —— 这样 `structural_action_refused`（S2-H 的
    次数/效果闸，只数 `_SEGMENT_OPENING_ACTIONS`）数到 0 次，**不会**抢在块闸前面
    拦下来，本文件测的才确实是块闸。
    """
    w = dict(FULL)
    # 窗口 0：已进生产（phase=PRODUCTION，避开预热那几条分支）、生产步数已达目标
    # （避开分支 5「生产帧没攒够」）、自检判不足 ⟹ 路由态 PROBLEM ⟹ 它是 earliest。
    w[0] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA",
            "self_verdict_source": "min_n_eff_over_g",
            "min_n_eff_over_g": 4.0, "n_eff_target": 10.0,
            "prod": 250000, "prod_target": 250000}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13,
                 config={"stage2_window_min_states": 4,
                         "stage2_window_max_states": 8,
                         "max_path_insertions": 3,
                         "stage2_max_production_blocks_per_window": cap})
    ck = pathlib.Path(run) / "checkpoints"
    # f_k 探针点名窗口 0 ⟹ `_decide_once` 的 5a 分支发 `RECALIBRATE_FK`。
    (ck / "stage2_fk_recalibration_probe.json").write_text(json.dumps({
        "verdict": "REANCHOR_DUE",
        "recalibration_recommended_windows": [0],
        "min_adjacent_shift_kJ_mol": 0.5,
    }))
    (ck / "stage2_autonomous_history.json").write_text(json.dumps({"iterations": [
        {"iteration": k + 1, "action": "RUN_PRODUCTION", "path_version": 1,
         "windows": [0],
         "snapshot": [{"window_idx": 0, "segment": "vanishing",
                       "production_steps": st, "solver_n_decorrelated": nd}]}
        for k, (st, nd) in enumerate(blocks)]}))
    return run


def _decide(run):
    return Stage2RepairController(run, "vanishing").decide()


def test_the_board_really_reaches_the_recalibration_branch(tmp_path):
    """基线：零块时这个盘面必须给出 `RECALIBRATE_FK`。

    没有这条，下面两条都可能是**假绿**（盘面根本没走到 5a，测了个寂寞）。
    """
    plan = _decide(_run(tmp_path, blocks=[]))
    assert plan["action"] == "RECALIBRATE_FK", plan["reason"]


def test_recalibration_is_refused_once_the_block_cap_is_full(tmp_path):
    """**S2-N 的回归钉子**：重标定也要过块数硬上限。

    还原变异：把 `plan()` 的条件改回 `if action == "RUN_PRODUCTION":` ⟹ 本条变红。
    """
    plan = _decide(_run(tmp_path, blocks=[(250000, 9), (500000, 11),
                                          (750000, 12), (1000000, 13)], cap=4))
    assert plan["action"] == "NO_ACTION", (
        "四块配额已满，`RECALIBRATE_FK` 仍被放行 —— 写侧记账、读侧不拦，"
        f"真机就是这样吃到第 5 块的。plan={plan['action']}｜{plan['reason'][:200]}")
    assert "块" in plan["reason"] and "上限" in plan["reason"], plan["reason"]


def test_a_stalled_frame_series_does_not_veto_recalibration(tmp_path):
    """**方向钉子**：「加帧没带来增益」不得否决重标定 —— 那正是它的适应症。

    序列 13 → 12 → 4（末点远低于前面各点的中位数）会触发补帧的**边际增益刹车**；
    对 `RUN_PRODUCTION` 那是对的，对 `RECALIBRATE_FK` 则完全反了。
    还原变异：把 `_frames_admission(..., hard_cap_only=...)` 的早退删掉 ⟹ 本条变红。
    """
    plan = _decide(_run(tmp_path, blocks=[(250000, 13), (500000, 12), (750000, 4)],
                        cap=4))
    assert plan["action"] == "RECALIBRATE_FK", (
        "补帧的边际增益刹车被套到了重标定头上 —— 「同分布加帧没用」恰恰是该换 f_k "
        f"的理由，不是不换的理由。plan={plan['action']}｜{plan['reason'][:200]}")


def test_the_gate_reads_the_same_table_the_ledger_writes(tmp_path):
    """两侧同一张表：闸的条件必须引用 `_BLOCK_CHARGING_ACTIONS`，不是某个字面量。

    这条防的是「下次给 `_BLOCK_CHARGING_ACTIONS` 加一个动作，闸忘了跟」——
    与 S2-L / BUD-03 是同一族（块账语义两处实现）。
    """
    src = inspect.getsource(Stage2RepairController._decide_once)
    assert "if action in self._BLOCK_CHARGING_ACTIONS:" in src, (
        "块数准入闸没有按 `_BLOCK_CHARGING_ACTIONS` 挂 —— 写侧记 4 个、读侧拦 1 个"
        "正是 S2-N 那个缺陷")
    assert 'if action == "RUN_PRODUCTION":' not in src.split(
        "if action in self._BLOCK_CHARGING_ACTIONS:")[0], (
        "闸前面还留着按单个动作名分岔的旧条件")


def test_the_skipped_final_analyze_still_leaves_a_diagnostic():
    """**S2-O 的回归钉子**：跳过收尾 ANALYZE 时必须落一份零采样的诊断清单。

    真机五个 run 走到那一条，结果是**整跑零产出**（`cyclod_ligand3/rep3` 连一次
    `ANALYZE` 都没发过，中间结果也没有）。跳过的决定本身是对的（跑 `run_once`
    就是重新采样，绕过控制器刚做的停止决定），缺的只是"跳过之后留下点什么"。

    ⚠️ 这里只能做源码探针：真跑一遍要整条流水线的夹具。所以额外钉住
    `write_comparison_manifest` 这个名字 —— 它是那份**纯聚合、不跑求解**的产物
    （见它自己的 docstring），换成别的写法必须同步改本条。
    """
    from abfe_pipeline import ABFEPipeline

    src = inspect.getsource(ABFEPipeline._run_stage2_autonomous)
    blk = src.split('outcome["final_analyze_skipped"] = _blockers')[1]
    # 只看到下一次写盘为止 —— 再往后是另一条分支
    blk = blk.split("_write_history()")[0]
    assert "write_comparison_manifest" in blk, (
        "收尾 ANALYZE 被跳过时没有落任何诊断 ⟹ 整跑零产出，"
        "人工接手连「哪个窗口坏在哪」都要重新翻散文件")
    assert "final_diagnostic_manifest" in blk, (
        "诊断清单的落点没有记进 `outcome` ⟹ 事后不知道有没有这份东西")
