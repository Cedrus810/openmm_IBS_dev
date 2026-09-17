# -*- coding: utf-8 -*-
"""`docs/CONTROLLER_BUDGET_AUDIT_2026-09-14.md` 的 63 条缺陷的**回归钉子**。

本文件只钉**行为不变量**（"给这样一份盘面，`decide()` 必须 / 不得给出某个动作"），
不钉内部实现：审计里的修复正在并行落地，函数签名随时会变，按签名写的测试会在
别人合并的那一刻变成噪声。

约定：
  · fixture 一律复用 `test_stage2_repair_controller._mkrun` / `R4` / `FULL` 与
    `test_rewindow_sampling_units._with_rewindow`，**不另造一套**；
  · 还没落地的修复用 `@pytest.mark.xfail(strict=False)` 占位，注释写明它在等哪条 ——
    `strict=False` 保证修复落地之后 xpass 不会把整套测试变红，不需要再回来改标记；
  · 需要新签名/新键才能测的，用 `hasattr` / 键存在性守卫先跳过。
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
from test_rewindow_sampling_units import _with_rewindow, BASE  # noqa: E402


TERMINAL = Stage2RepairController.TERMINAL_EXITS
# 预算类出口。**它们出现在一个动作上，就等于宣称"这个动作付不起"** —— 所以哪本账
# 说的话算数，是可以逐条检查的。
_BUDGET_EXITS = ("GLOBAL_BUDGET_EXHAUSTED", "HALT_BUDGET",
                 "HALT_VALIDATION_BUDGET_UNREACHABLE", "HALT_LOCAL_VALIDATION_CAP")


def _ck(run):
    return pathlib.Path(run) / "checkpoints"


def _patch_convergence(run, idx, mutate, stage_name="vanishing", stage_type="vdw"):
    """就地改写某个窗口的 convergence 产物（`_mkrun` 造不出的形状用这个补）。"""
    f = (pathlib.Path(run) / stage_name
         / f"dual_window_{idx}_{stage_type}_convergence.json")
    d = json.loads(f.read_text())
    mutate(d)
    f.write_text(json.dumps(d))


# =============================================================== 终止性
# 审计根因 #2：「未知」在四处有四套语义；#34/#15 是同一条规矩的两个方向。

def _warmup_exhausted_board(tmp_path):
    """win0 只缺生产帧（自检不够），而**预热账全干**。"""
    w = dict(FULL)
    w[0] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA", "min_n_eff_over_g": 3.1,
            "self_verdict_source": "solver_eligibility",  # [2026-09-15] 本用例测的是预算/计费，低比值只是拿到 RUN_PRODUCTION 的载体；`min_n_eff_over_g` 现在归**偏斜**（加帧治不了）⟹ 按 docstring 的本意显式声明成缺帧
            "evidence": "verified", "warmup": 555000, "cap": 555000}
    return _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)


def test_warmup_budget_never_terminates_a_production_only_action(tmp_path):
    """#34 反方向：**A 账本的余额不得终止只花 B 账本的动作。**

    `RUN_PRODUCTION` 用的是**已冻结**的 f_k，一步验证预算都不花。预热账干了就给
    它挂一个终态出口（`GLOBAL_BUDGET_EXHAUSTED` 在 `TERMINAL_EXITS` 里）⟹ 主循环
    在分发**之前**就 break，补帧根本不执行。
    """
    plan = Stage2RepairController(_warmup_exhausted_board(tmp_path), "vanishing").decide()
    assert plan["action"] == "RUN_PRODUCTION", plan["reason"]
    assert plan["exit"] not in TERMINAL, (
        f"只花生产帧的动作被**预热**账的余额终止了：exit={plan['exit']}｜{plan['reason']}"
    )
    assert not plan["terminal"] and plan["execution_status"] == "IN_PROGRESS"


def test_a_production_only_action_is_not_attributed_to_the_warmup_ledger(tmp_path):
    """归因也得来自**它自己那本账**（#34 的弱形式，只差没终止）。

    动作没被终止不代表记账是对的：`RUN_PRODUCTION` 挂着 `HALT_BUDGET`「付不起新
    Epoch 的最低验证额度」——而它根本不开新 Epoch。下一个读日志的人会据此去调
    验证预算，而真正的缺口在别处。
    """
    plan = Stage2RepairController(_warmup_exhausted_board(tmp_path), "vanishing").decide()
    assert plan["action"] == "RUN_PRODUCTION"
    assert plan["exit"] not in _BUDGET_EXITS, (
        f"只花生产帧的动作被归因到**预热**账：exit={plan['exit']}｜{plan['reason']}")


def _warmup_action_board(tmp_path, *, prod_cap):
    """win1 需要 `CONTINUE_WARMUP`（只花预热帧），而**生产账**几乎见底。"""
    cfg = {"stage2_window_min_states": 4, "stage2_window_max_states": 8,
           "max_path_insertions": 3, "stage2_production_budget_steps": prod_cap}
    w = {i: {"K": 4, "prod": 250000} for i in range(4)}
    w[1] = {"K": 4, "prod": 250000, "bias_status": "frozen_validation_indeterminate",
            "evidence": "indeterminate", "warmup": 100000, "cap": 555000}
    return _mkrun(tmp_path, windows=w, ranges=R4, n_states=13, config=cfg)


def test_production_budget_never_terminates_a_warmup_only_action(tmp_path):
    """#34 正方向：`CONTINUE_WARMUP` 花的是**预热/验证**预算，不是生产块。

    现状是 `plan()` 里除了两个「非变异」动作之外一律按一整块**生产**预算收费
    ⟹ 生产账见底时，一个一帧生产都不产的续预热动作被判 `GLOBAL_BUDGET_EXHAUSTED`
    （终态）。这正是这套双账要防的事，方向刚好反了。
    """
    # 4 窗 × 250k = 1,000,000 已用，上限 1,000,001 ⟹ 生产账只剩 1 步
    run = _warmup_action_board(tmp_path, prod_cap=1_000_001)
    c = Stage2RepairController(run, "vanishing")
    pb = c.read()["production_budget"]
    assert pb["cap_known"] and pb["stage_remaining_steps"] == 1, pb   # fixture 自检

    plan = c.decide()
    assert plan["action"] == "CONTINUE_WARMUP", (
        f"生产账见底把一个**只花预热帧**的动作终止了：{plan['action']}／"
        f"{plan['exit']}｜{plan['reason']}"
    )
    assert plan["exit"] != "GLOBAL_BUDGET_EXHAUSTED"


def _unknown_warmup_ledger_board(tmp_path):
    w = dict(FULL)
    w[0] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA", "min_n_eff_over_g": 3.1,
            "evidence": "indeterminate", "bias_status": "frozen_validation_indeterminate"}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)
    # 账本整段读不到 —— 真机 cyclod_ligand1/rep3 win4 就是这个形状
    _patch_convergence(run, 0, lambda d: d.get("bias_warmup", {}).pop(
        "warmup_budget_ledger", None))

    return run


def test_an_unknown_warmup_budget_never_produces_a_terminal_budget_exit(tmp_path):
    """#15/#36：`warmup_steps_left is None` 是**未知**，不是零。

    未知压成 0 会据此 break 整个自治循环，日志还打印一个编造出来的「剩余 0 步」。
    """
    c = Stage2RepairController(_unknown_warmup_ledger_board(tmp_path), "vanishing")
    w0 = next(x for x in c.read()["windows"] if x["window_idx"] == 0)
    assert w0["warmup_steps_left"] is None, "fixture 没造出「账本读不到」"

    plan = c.decide()
    assert not (plan["terminal"] and plan["exit"] in _BUDGET_EXITS), (
        f"账本**读不到**被当成「预算耗尽」的终态：exit={plan['exit']}｜{plan['reason']}")


def test_an_unknown_warmup_budget_is_not_attributed_as_unaffordable(tmp_path):
    """「未知」两个方向都不是零 —— 也不能是「付不起」。

    读不到账本时，「这个窗口付不起验证额度」这个结论根本不成立；据此把窗口钉死，
    下游分支 1c/1d/3/3a/4/5/5a/5b/6 全部不可达（#20）。
    """
    plan = Stage2RepairController(
        _unknown_warmup_ledger_board(tmp_path), "vanishing").decide()
    assert plan["exit"] not in _BUDGET_EXITS, (
        f"账本**读不到**被归因成「预算耗尽」：exit={plan['exit']}｜{plan['reason']}")


def _fixed_lambda_middle_failure_board(tmp_path):
    """λ 表顶到溢出槽上限 + 失败的是中间窗 ⟹ 唯一对症的是有界 rewindow。

    ⚠️ [REWIND-01，2026-09-17] 窗口 1 原来**只在 stage 级的 `target_support_gate`
    里失败**，逐窗自检是正常的。而 2026-09-15 定案「未经标定的阈值不驱动动作」之后，
    逐段质量门只作报告（每条带 `drives_action: False`）⟹ 这块盘面**一个 PROBLEM
    窗口都没有**，`earliest is None`，控制器如实落到兜底分支。那不是 rewindow 被
    预算挡掉，是这块盘面已经不含它自己名字里那个「中间窗失败」。
    改成与 `test_stage2_repair_controller` 同一份 `SUPPORT_FAIL_WINDOW`
    （逐窗自检 HARD_INSUFFICIENT + 归因 top1pct ⟹ 偏斜类），让盘面名副其实。
    """
    from test_stage2_repair_controller import (
        SUPPORT_FAIL_WINDOW, _stage_support_failure)
    return _mkrun(
        tmp_path,
        windows={0: {"K": 4}, 1: dict(SUPPORT_FAIL_WINDOW), 2: {"K": 9}},
        ranges=[(0, 4), (3, 8), (7, 16)], n_states=16,
        config={"stage2_window_min_states": 4, "stage2_window_max_states": 5,
                "max_path_insertions": 3},
        stage_result=_stage_support_failure(worst_window=1),
    )


def test_a_stage_gate_only_failure_drives_no_action(tmp_path):
    """2026-09-15 定案的守卫：**未经标定的阈值不驱动动作。**

    盘面：逐窗自检**全部正常**，只有 stage 级 `target_support_gate` 失败。
    那些阈值（`raw_min_absolute_ess=20`、`max_top1pct_raw_weight=0.35` …）
    **一个都没标定过**（`docs/archive/AUDIT_GATES_AND_CRITERIA_2026-09-17.md` §3.3：
    其中一条的报错文本自承无依据；九道门与实验偏差之间测不到关系，且只有约
    2 个独立方向）⟹ 它们只作报告（每条带 `drives_action: False`），
    **不得**产生 PROBLEM 窗口、不得驱动补帧/缩跨度。

    ⟹ `earliest` 必须是 None，控制器如实落到兜底：`NO_ACTION`/`NO_FEASIBLE_ACTION`。

    ⚠️ 这块盘面此前**只被 `_fixed_lambda_middle_failure_board` 意外覆盖着**；
    2026-09-17 那个 fixture 改成「窗口 1 真的自检失败」之后覆盖就没了
    （abfe-ibs-08 的 AST 扫描：全仓 0 条测试同时碰 `drives_action` 和 `decide()`）。
    单独立一条，免得哪天有人把质量门接回路由而全套测试不红。
    """
    from test_stage2_repair_controller import _stage_support_failure
    run = _mkrun(
        tmp_path,
        windows={0: {"K": 4}, 1: {"K": 5}, 2: {"K": 9}},   # 三个窗自检都正常
        ranges=[(0, 4), (3, 8), (7, 16)], n_states=16,
        config={"stage2_window_min_states": 4, "stage2_window_max_states": 5,
                "max_path_insertions": 3},
        stage_result=_stage_support_failure(worst_window=1),
    )
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["earliest_unresolved_window"] is None, (
        f"stage 级质量门造出了一个 PROBLEM 窗口：{plan['reason']}")
    assert plan["action"] == "NO_ACTION", plan["reason"]
    assert plan["exit"] == "NO_FEASIBLE_ACTION", plan["reason"]
    # 读数照报，只是不驱动动作
    assert any(g.get("drives_action") is False
               for g in (plan.get("stage_quality_gates") or [])), plan


def test_an_unknown_production_remainder_does_not_reject_a_bounded_rewindow(tmp_path):
    """#37：「余量未知」不得当成「预留不出来」。

    `IMMUTABLE_REWINDOW` 是固定 λ 表下唯一对症的动作；把 `None` `or 0` 成零余量
    就把它误判成付不起，于是那条盘面上一个可行动作都不剩。
    """
    run = _fixed_lambda_middle_failure_board(tmp_path)
    c = Stage2RepairController(run, "vanishing")
    pb = c.read()["production_budget"]
    assert pb["stage_remaining_steps"] is None, pb        # 未知
    plan = c.decide()
    assert plan["action"] == "IMMUTABLE_REWINDOW", (
        f"余量未知把有界重窗挡掉了：{plan['action']}／{plan['exit']}｜{plan['reason']}")
    assert plan["exit"] not in _BUDGET_EXITS


def test_the_terminal_flag_always_agrees_with_the_terminal_exit_table(all_plans):
    """任何带 `exit` 的 plan，`terminal` 必须等于 `exit in TERMINAL_EXITS`。

    「看起来像终止、实际不终止」（#11）与它的反面在真机上各栽过一次：主循环只看
    `terminal` 就 break，两者一旦脱节，要么该停的不停、要么不该停的停。
    """
    for label, plan in all_plans:
        assert plan["terminal"] == (plan["exit"] in TERMINAL), (
            f"[{label}] exit={plan['exit']} terminal={plan['terminal']}")
        assert plan["routing"] == (
            plan["exit"] is not None and plan["exit"] not in TERMINAL), label


def test_done_and_complete_are_each_other_s_only_cause(all_plans):
    """`action == "DONE"` ⟺ `execution_status == "COMPLETE"`；`NO_ACTION` 绝不是 COMPLETE。"""
    for label, plan in all_plans:
        assert (plan["action"] == "DONE") == (plan["execution_status"] == "COMPLETE"), (
            f"[{label}] action={plan['action']} status={plan['execution_status']}")
        if plan["action"] == "NO_ACTION":
            assert plan["execution_status"] != "COMPLETE", label


# =============================================================== 路由可达性

def test_all_unknown_windows_with_short_production_still_ask_for_frames(tmp_path):
    """#22：全窗 UNKNOWN + 生产步数没到目标 ⟹ 必须发 `RUN_PRODUCTION`，不是空转 `ANALYZE`。

    `earliest is None` 时 `_pick()` 返 `[]`，分支 2/3/3a/4/5/5b/6 整批被打掉，
    于是连「生产帧没攒够」都发不出来，只能靠停滞保护退出 —— 而 `ANALYZE` 在一个
    帧都没攒够的盘面上产不出新证据，那正是真机上连发四次、盘面一字节没变的形状。
    """
    # `self_verdict=None` ⟹ UNKNOWN；`bias_status=converged` ⟹ 不在预热里
    w = {i: {"K": 4, "self_verdict": None, "prod": 100000, "prod_target": 250000}
         for i in range(4)}
    plan = Stage2RepairController(
        _mkrun(tmp_path, windows=w, ranges=R4, n_states=13), "vanishing").decide()
    assert plan["action"] == "RUN_PRODUCTION", (
        f"全窗 UNKNOWN 把补帧分支整批遮蔽了：{plan['action']}｜{plan['reason']}")


def test_a_terminal_bias_status_window_is_routed_by_its_own_cause(tmp_path):
    """#23：`phase == "TERMINAL"`（`bias_status=failed`）全仓无分支处理。

    它会当上 `earliest` 并挡住所有下游窗口，最后以一个与根因无关的理由退出。
    钉的是**归因**：动作或终态必须提到这个失败本身，不许静默落到别的分支。
    """
    w = dict(FULL)
    w[0] = {"K": 4, "bias_status": "failed", "evidence": "indeterminate"}
    plan = Stage2RepairController(
        _mkrun(tmp_path, windows=w, ranges=R4, n_states=13), "vanishing").decide()

    assert plan["windows"] == [0], (
        f"earliest 应该是那个 TERMINAL 窗口：{plan['windows']}｜{plan['reason']}")
    assert any(k in plan["reason"] for k in ("failed", "TERMINAL", "标定失败", "终态")), (
        f"归因里没提这个窗口为什么进了终态：{plan['reason']}")


def test_an_unreadable_warmup_ledger_does_not_pin_a_skipped_window_to_production(tmp_path):
    """#20：换 Epoch 预算预检**无条件**对 earliest 跑，且对未知 fail-closed
    ⟹ 账本读不到的窗口被永久钉在 `RUN_PRODUCTION`，下游分支全不可达。

    具体断言：同一个窗口**同时**被求解器跳窗时，动作必须是针对跳窗的那条路由
    （而不是一条与跳窗无关、只因为账本读不到才落下来的补帧）。
    """
    w = dict(FULL)
    w[0] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA", "min_n_eff_over_g": 3.1,
            "n_decorr": 9, "evidence": "indeterminate",
            "bias_status": "frozen_validation_indeterminate"}
    run = _mkrun(
        tmp_path, windows=w, ranges=R4, n_states=13,
        # 🔑 [2026-09-15] `converged` 已删键；写老键的 stage_result **根本不会被
        # `_read_stage_result()` 认成一份 stage 结果**（嗅探键是 `analysis_status` /
        # `total_delta_G`）⟹ 求解器证据整份读不到，归因判反。
        stage_result={"analysis_status": "ANALYSIS_INCOMPLETE",
                      "analysis_incomplete_reasons": ["存在被跳过的窗口 [0]。"],
                      "total_delta_G": -12.3, "total_error": 0.9,
                      "skipped_windows": [{
                          "window_index": 0, "n_frames_after_decorrelation": 9,
                          "min_frames_per_window": 10,
                          "reason": "insufficient_frames_after_decorrelation"}],
                      "cumulative_fk_residual_production": [
                          {"window_index": 0, "verdict": "PASS",
                           "cumulative_residual_span_kJ_mol": 1.2}]})
    _patch_convergence(run, 0, lambda d: d.get("bias_warmup", {}).pop(
        "warmup_budget_ledger", None))

    c = Stage2RepairController(run, "vanishing")
    assert next(x for x in c.read()["windows"]
                if x["window_idx"] == 0)["warmup_steps_left"] is None
    plan = c.decide()
    assert plan["windows"] == [0], plan["reason"]
    assert any(k in plan["reason"] for k in ("跳出", "踢出", "协方差链", "帧数不足",
                                             "去相关")), (
        "动作没按跳窗归因 —— 预算预检把根因盖掉了：" + plan["reason"])


def test_a_layout_action_is_only_ever_issued_when_it_is_feasible(all_plans):
    """#13/#14：布局动作必须**当场可行**。

    5a-2 的 REJECT 分支曾完全不查可行性（发一个执行器会 `RuntimeError` 的动作，
    或绕过 `max_path_insertions` 无限插点）；#14 的三元式又与它上面的注释正好相反。
    这条把所有能产出布局动作的盘面一起兜住：**发出来就必须在 feasible 表里。**
    """
    # ⚠️ 两套词汇表大小写不同：`plan["action"]` 是 `INSERT_LAMBDA`，
    # `feasible_structural_actions` 的键是 `insert_lambda`。直接 `in` 恒为假 ——
    # 一条按直觉写的守卫会静默永不触发，所以这里显式归一化。
    # 🔑 [2026-09-17] `IMMUTABLE_REWINDOW` 也是布局动作，也必须当场可行。
    # REWIND-01 接通之后它在多数盘面上比插 λ/拆末窗先命中（它不动 λ 表、
    # 不需要 tail anchor、不吃插点预算），所以只数那两个会让本条**测不到东西**。
    # ⚠️ 它的可行性不在 `feasible_structural_actions` 里（那张表只描述 λ 表上的
    # 两个动作），用控制器自己的谓词 `rewindow_feasible()` 查。
    seen = set()
    for label, plan in all_plans:
        if plan["action"] in ("INSERT_LAMBDA", "SPLIT_TAIL_WINDOW"):
            seen.add(plan["action"])
            feas = {str(x).lower() for x in plan["feasible_structural_actions"]}
            assert plan["action"].lower() in feas, (
                f"[{label}] 发了一个不可行的布局动作 {plan['action']}："
                f"{plan['infeasible_structural_actions']}｜{plan['reason']}")
        elif plan["action"] == "IMMUTABLE_REWINDOW":
            seen.add(plan["action"])
            assert plan["windows"], f"[{label}] 有界重窗没点名窗口"
    assert seen, "没有任何盘面产出布局动作 —— 这条断言没测到东西，补盘面"


# =============================================================== 子窗不是二等公民

def _rewindow_board(tmp_path, **kw):
    kw.setdefault("child_states",
                  [("ANALYSIS_ELIGIBLE", 250000), ("INSUFFICIENT_DATA", 250000)])
    return _with_rewindow(tmp_path, **kw)


def test_a_sub_window_topup_is_bounded_by_the_hard_block_cap(tmp_path):
    """#8：**子窗的两道刹车同时失效。**

    父窗的步数在被取代那一刻就冻结了 ⟹ 拿父窗的块账去问准入，去重后恒 1 行、
    永远批得过。BUD-03 修过的「单窗烧 150 万步」在新位置原样复发。
    钉法是行为级的：连续给同一个 `unit_id` 批帧，到硬上限必须停 ——
    台账按 `unit_id` 记，别的形状随修复者定。
    """
    run = _rewindow_board(tmp_path)
    c0 = Stage2RepairController(run, "vanishing")
    uid = c0.decide()["unit_id"]
    assert uid, "fixture 没造出「有待补子窗」的盘面"

    cap = int(c0.read().get("max_production_blocks_per_window") or 4)
    (_ck(run) / "stage2_autonomous_history.json").write_text(json.dumps({
        "iterations": [
            {"iteration": k + 1, "action": "RUN_PRODUCTION", "path_version": 1,
             "snapshot": [{"unit_id": uid, "window_idx": 1,
                           "segment": "vanishing_rewindow_abc123",
                           "production_steps": 250000 * (k + 1),
                           "solver_n_decorrelated": 9 + k}]}
            for k in range(cap)]}))

    plan = Stage2RepairController(run, "vanishing").decide()
    assert not (plan["action"] == "RUN_PRODUCTION" and plan.get("unit_id") == uid), (
        f"子窗 {uid} 已经批过 {cap} 块（硬上限 {cap}）还在继续批：{plan['reason']}")


def test_the_unit_noop_key_written_is_the_key_that_gets_read(tmp_path):
    """#9：子窗 no-op 写的 key 全仓**无人读**。

    这里**用真写侧写、用真读侧读** —— 两边的 key 格式一旦漂移，这条就红。
    契约：unit 用 `f"{action}:unit:{unit_id}"`，物理窗口用 `f"{action}:{window_idx}"`。
    """
    pytest.importorskip("openmm", reason="`_record_noop_action` 在 abfe_pipeline 里")
    from abfe_pipeline import ABFEPipeline

    run = _rewindow_board(tmp_path)
    c0 = Stage2RepairController(run, "vanishing")
    before = c0.decide()
    uid = before["unit_id"]
    assert before["action"] == "RUN_PRODUCTION" and uid, before["reason"]

    pipe = object.__new__(ABFEPipeline)
    pipe._log = lambda *_a, **_k: None
    pipe._record_noop_action(str(_ck(run)), before["action"], before["windows"],
                             c0.read(), reason="action_changed_nothing_on_disk",
                             unit_id=uid)

    led = json.loads((_ck(run) / "stage2_noop_actions.json").read_text())
    assert f"{before['action']}:unit:{uid}" in led, (
        f"写侧的 key 格式变了：{list(led)}")

    after = Stage2RepairController(run, "vanishing").decide()
    assert not (after["action"] == before["action"]
                and after.get("unit_id") == uid), (
        f"执行器记过「这个子窗上这个动作是 no-op」，控制器还是又发了一次："
        f"{after['action']}[{after.get('unit_id')}]｜{after['reason']}")


def test_an_abandoned_rewindow_entry_yields_no_schedulable_unit(tmp_path):
    """#7：`ABANDONED_NO_PRODUCT` 的条目每轮烧一块帧，而**一帧都进不了 ΔG**。

    合并求解只采信 `SAMPLED`，`_topup_rewindow_child` 又从不把 status 推回
    `SAMPLED` ⟹ 死循环直到 40 轮上限，而停滞保护看不见它（签名里
    `production_steps` 每轮都在变）。
    """
    run = _rewindow_board(tmp_path)
    f = _ck(run) / "stage2_rewindow_ledger.json"
    d = json.loads(f.read_text())
    d["abc123"]["status"] = "ABANDONED_NO_PRODUCT"
    f.write_text(json.dumps(d))

    plan = Stage2RepairController(run, "vanishing").decide()
    assert not str(plan.get("unit_id") or "").startswith("rw:abc123:"), (
        f"核销为「没有产物」的条目仍然被当成可调度子窗：{plan['unit_id']}｜{plan['reason']}")


# =============================================================== 账本口径

def test_stale_layout_evidence_is_cleared_once_the_window_is_resampled(tmp_path):
    """#18：窗口在新段按新布局重采成功之后，必须从 `stale_layout_evidence` 里消失。

    否则插过一次 λ 之后 `view["stale_layout_evidence"]` 恒非空 ⟹ 分支 0a 的 `DONE`
    与 `_evidence_status` 的 `CONVERGED` **永远不可达**，自治循环没有终点。
    """
    import shutil

    # 基准段是旧布局（3 态），新段 vanishing_2 是当前布局（4 态）
    run = _mkrun(tmp_path, windows={i: {"K": 4} for i in range(4)},
                 ranges=R4, n_states=13,
                 stage_result={"analysis_status": "ANALYSIS_COMPLETE",
                               "total_delta_G": -12.3, "total_error": 0.9,
                               "path_is_complete": True})
    _patch_convergence(run, 0, lambda d: d.update(
        {"lambdas_vdw": d["lambdas_vdw"][:3]}))      # 旧布局的残留证据

    seg = pathlib.Path(run) / "vanishing_2"
    seg.mkdir()
    (_ck(run) / "segment_2").mkdir()
    for nm in ("dual_window_0_vdw_convergence.json",
               "dual_window_0_vdw_self_support.json"):
        shutil.copy(pathlib.Path(run) / "vanishing" / nm, seg / nm)
    # 新段里按**当前**布局重采成功（4 态，与 ibs_state 一致）
    d = json.loads((seg / "dual_window_0_vdw_convergence.json").read_text())
    d["lambdas_vdw"] = [1.0 - 0.1 * i for i in range(4)]
    (seg / "dual_window_0_vdw_convergence.json").write_text(json.dumps(d))
    shutil.copy(_ck(run) / "ibs_state_vdw_window_0.json",
                _ck(run) / "segment_2" / "ibs_state_vdw_window_0.json")

    c = Stage2RepairController.for_physical_stage(run, "vanishing", "vdw")
    view = c.read()
    assert 0 not in (view.get("stale_layout_evidence") or {}), (
        "窗口 0 已在新段按当前布局重采成功，却仍留在 stale 表里 ⟹ DONE 永久不可达："
        f"{view['stale_layout_evidence']}")


def test_the_hard_block_cap_is_not_reset_by_a_segment_change(tmp_path):
    """BUD-03 防回归：**块账要分两个量**，换段不得把硬上限配额退回去。

      · 硬上限 = 资源账，**跨段累计**（换 Epoch 就换段，那是循环自己的动作）；
      · 边际增益 = 同一条曲线上的趋势，**只在同段**内比（跨 f_k epoch 比数字没意义）。

    真机 cyclod_ligand2/rep1 win5：history 记的是 `vanishing`、当前视图是
    `vanishing_7` ⟹ 全被过滤掉、块账空 `{}`，两道刹车一次都没触发过，单窗烧到 150 万步。
    """
    cfg = {"stage2_window_min_states": 4, "stage2_window_max_states": 8,
           "max_path_insertions": 3, "stage2_max_production_blocks_per_window": 4}
    w = dict(FULL)
    w[0] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA",
            "self_verdict_source": "solver_eligibility", "prod": 250000}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13, config=cfg,
                 stage_result={"analysis_status": "ANALYSIS_INCOMPLETE",
                               "analysis_incomplete_reasons": ["路径缺窗。"],
                               "total_delta_G": -12.3, "total_error": 0.9,
                               "min_decorrelated_samples_threshold": 20,
                               "window_overlap_diagnostics": [
                                   {"window_index": 0, "n_frames_decorrelated": 9}]})
    # 4 块帧分散在**四个不同的段**里，且判据量一路在涨（边际增益拦不住它）
    (_ck(run) / "stage2_autonomous_history.json").write_text(json.dumps({
        "iterations": [
            {"iteration": k + 1, "action": "RUN_PRODUCTION", "path_version": 1,
             "snapshot": [{"window_idx": 0, "segment": f"vanishing_{k + 2}",
                           "production_steps": 250000 * (k + 1),
                           "solver_n_decorrelated": 5 + 3 * k}]}
            for k in range(4)]}))

    # ⚠️ 只断言**行为**：两本账的内部形状（`by_window`/`by_unit` 等）正在被 #8
    # 的修复改，按形状断言会在别人合并的那一刻变成噪声。
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] != "RUN_PRODUCTION", (
        f"跨段已经批过 4 块（上限 4）还在继续批 —— 换段把硬上限配额退回去了："
        f"{plan['reason']}")


def test_skipped_windows_never_carry_solver_namespace_indices(tmp_path):
    """#58：`skipped_windows` 混了 solver 命名空间（子窗 `window_index ≥ 10000`）
    与物理窗口下标。

    一个被跳的子窗会以父窗之外的身份**永久封死 DONE**（分支 0a 要求
    `skipped_windows` 为空），`render` 还把 10000 当窗口号打印给人看。
    子窗的跳窗证据该挂在 `sampling_units[*]["solver_skip"]` 上 —— 那张表本来就有。
    """
    run = _rewindow_board(tmp_path, skipped=(0, 1))
    view = Stage2RepairController(run, "vanishing").read()

    assert [u["solver_skip"] for u in view["sampling_units"]] != [None, None], (
        "fixture 没把跳窗记录挂到子窗上")
    bad = [i for i in (view.get("skipped_windows") or []) if int(i) >= BASE]
    assert not bad, (
        f"solver 命名空间的索引 {bad} 漏进了物理窗口的 `skipped_windows`："
        f"{view['skipped_windows']}")


# =============================================================== 盘面集合（sweep 用）

@pytest.fixture(scope="module")
def all_plans(tmp_path_factory):
    """一组覆盖不同分支的真实盘面 ⟹ `(label, plan)`。

    sweep 类断言（terminal 语义、DONE⟺COMPLETE、布局动作可行性）靠它一次兜住
    多条分支 —— 逐分支挂断言必然漏，`decide()` 有十几个出口。
    """
    from test_stage2_repair_controller import _stage_support_failure, _STAGE_BASE

    def td(name):
        return tmp_path_factory.mktemp(name)

    boards = {}

    boards["missing_window"] = _mkrun(td("a"), windows={0: {}, 1: {}, 2: {}},
                                      ranges=R4, n_states=13)
    boards["all_eligible_no_stage"] = _mkrun(td("b"), windows=FULL, ranges=R4,
                                             n_states=13)
    boards["converged"] = _mkrun(td("c"), windows=FULL, ranges=R4, n_states=13,
                                 stage_result={"analysis_status": "ANALYSIS_COMPLETE",
                                               "total_error": 0.9,
                                               "total_delta_G": -12.3,
                                               "path_is_complete": True})
    _w = dict(FULL)
    _w[2] = {"K": 4, "bias_status": "calibrated_validation_failed",
             "evidence": "refuted"}
    boards["fk_refuted"] = _mkrun(td("d"), windows=_w, ranges=R4, n_states=13)

    _w = dict(FULL)
    _w[1] = {"K": 4, "bias_status": "frozen_validation_indeterminate",
             "evidence": "indeterminate", "warmup": 100000, "cap": 555000}
    boards["continue_warmup"] = _mkrun(td("e"), windows=_w, ranges=R4, n_states=13)

    _w = dict(FULL)
    _w[1] = {"K": 4, "bias_status": "frozen_validation_indeterminate",
             "evidence": "indeterminate", "warmup": 520000, "cap": 555000}
    boards["warmup_budget_tight"] = _mkrun(td("f"), windows=_w, ranges=R4, n_states=13)

    _w = dict(FULL)
    _w[3] = {"K": 4, "prod": 100000, "prod_target": 250000,
             "self_verdict": "INSUFFICIENT_DATA", "min_n_eff_over_g": 4.2}
    boards["short_production"] = _mkrun(td("g"), windows=_w, ranges=R4, n_states=13)

    # 支撑/偏斜类 stage 门失败 ⟹ 布局动作（末窗可拆、末窗最差）
    boards["tail_support_failure"] = _mkrun(
        td("h"), windows={0: {"K": 4}, 1: {"K": 5}, 2: {"K": 9}},
        ranges=[(0, 4), (3, 8), (7, 16)], n_states=16,
        config={"stage2_window_min_states": 4, "stage2_window_max_states": 5,
                "max_path_insertions": 3},
        stage_result=_stage_support_failure(worst_window=2))

    boards["overlap_gate_failure"] = _mkrun(
        td("i"), windows=FULL, ranges=R4, n_states=13,
        stage_result={**_STAGE_BASE, "min_overlap": 0.02,
                      "min_overlap_threshold": 0.10})

    boards["fixed_lambda_rewindow"] = _fixed_lambda_middle_failure_board(td("j"))
    boards["production_cap_exhausted"] = _mkrun(
        td("k"), windows={**FULL, 0: {"K": 4, "self_verdict": "INSUFFICIENT_DATA",
                                      "min_n_eff_over_g": 3.1, "prod": 250000}},
        ranges=R4, n_states=13,
        config={"stage2_window_min_states": 4, "stage2_window_max_states": 8,
                "max_path_insertions": 3, "stage2_production_budget_steps": 100000})
    boards["rewindow_children"] = _rewindow_board(td("l"))
    boards["empty"] = _mkrun(td("m"), windows={}, ranges=R4, n_states=13)

    return [(k, Stage2RepairController(v, "vanishing").decide())
            for k, v in boards.items()]


# =============================================================== 段号识别（#2/#3）

def test_a_rewindow_directory_is_never_mistaken_for_a_sampling_segment():
    """#2/#3：「什么算一个采样段」只能有一份实现，判据是**完整后缀纯数字**。

    旧写法 `rsplit("_", 1)[-1].isdigit()` 对 `vanishing_rewindow_<sha256[:12]>`
    有 ≈(10/16)¹² ≈ 0.34% 的概率命中 ⟹ 子系综目录被当成采样段合并，**子窗局部
    下标 0/1 被当成物理窗口 0/1**，覆盖度检查照样通过 ⟹ 静默错 ΔG。
    而段的合并顺序必须按**数字**排（#3：`vanishing_10` 的字符串序排在
    `vanishing_2` 之前，真机见过 39 个段目录）。
    """
    import abfe_preoptimizer as pre

    fn = getattr(pre, "segment_index_of_dir", None)
    if fn is None:
        pytest.xfail("等 #2/#3：模块级 `segment_index_of_dir(path, stage_dir)` 还没落盘")

    base = "/run/vanishing"
    assert fn(base, base) == 1, "基准段就是第 1 段"
    assert fn("/run/vanishing_2", base) == 2
    assert fn("/run/vanishing_10", base) == 10
    # 12 位全数字的 identity —— 旧判据在这里命中，新判据必须看**完整后缀**
    assert fn("/run/vanishing_rewindow_123456789012", base) is None
    assert fn("/run/vanishing_rewindow_0123456789ab", base) is None
    assert fn("/run/vanishing_autonomous_inprogress", base) is None

    # #3：按返回的数字排序，`vanishing_10` 必须排在 `vanishing_2` **之后**
    dirs = ["/run/vanishing_10", "/run/vanishing_2", base]
    assert sorted(dirs, key=lambda d: fn(d, base)) == [
        base, "/run/vanishing_2", "/run/vanishing_10"], "段序又按字符串排了"


# =============================================================== g ↔ τ_int（#54）

def test_the_join_report_never_hands_out_g_under_a_tau_shaped_name():
    """#54：`subsample_series_by_autocorrelation` 返回的是**统计低效率 g**，
    `g = 1 + 2τ_int` —— 名字写成 `tau_int` 读数就差约两倍。

    ⚠️ **这条不能写成「`tau_int == (g−1)/2`」**：`ibs_engine` 落地时明确**不再写
    `tau_int`**（那段注释的论证是「名字不变、值改了」才是最难发现的一类改动 ——
    grep 找得到键，漏改的读点静默差一倍且不抛异常；将来真要 τ 本身得用**新名字**
    `tau_int_from_g`）。所以这里钉的是**那个决定**本身：

      · 权威键是 `statistical_inefficiency_g`，且它装的是 g（g ≥ 1）；
      · `tau_int` 这个名字**一旦回来**，装的必须是真 τ=(g−1)/2，不许再装 g。

    两条合起来就是「键名不变、值变了」的锚：任一侧被改回去都会在这里红。
    """
    import tempfile

    import ibs_engine as ie
    from test_join_lambda_two_sided_support import _two_windows, KT

    with tempfile.TemporaryDirectory() as td:
        r = ie.join_lambda_two_sided_support(
            _two_windows(pathlib.Path(td)), "vdw", 0, 1, KT)
    assert r is not None

    for side in ("upstream", "downstream"):
        d = r[side]
        assert "statistical_inefficiency_g" in d, (
            f"权威键没了 —— 读点的回退写法 "
            f"`d.get('statistical_inefficiency_g', d.get('tau_int'))` 会静默取到旧键：{list(d)}")
        g = d["statistical_inefficiency_g"]
        assert g is not None and float(g) >= 1.0, (
            f"g < 1 在数学上不可能 —— 这个键装的多半已经是 τ 了：g={g}")
        if "tau_int" in d:
            assert d["tau_int"] == pytest.approx((float(g) - 1.0) / 2.0, rel=1e-12), (
                "`tau_int` 这个名字回来了，但装的还是 g（差约 2 倍）—— "
                "要么改名 `tau_int_from_g`，要么装真 τ")


# =============================================================== 两个命名空间（#58 正向）

def test_a_skipped_child_ensemble_is_reported_as_a_sampling_unit(tmp_path):
    """#58 正向：**拆键不得退化成丢信息。**

    只钉反向（`skipped_windows` 里没有 ≥10000）的话，「把子窗直接扔掉」也能让
    反向断言全绿 —— 而那等于所有下游对子窗的跳窗结构性失明。被跳的子窗必须
    出现在 `skipped_sampling_units` 里，**并且带 `unit_id`**（不是 solver 索引）。
    """
    view = Stage2RepairController(
        _rewindow_board(tmp_path, skipped=(0, 1)), "vanishing").read()

    units = view.get("skipped_sampling_units")
    assert units, f"被跳的子窗整个丢了：{view.get('skipped_windows')}／{units}"
    assert sorted(units) == ["rw:abc123:0", "rw:abc123:1"], (
        f"没翻成调度身份 `unit_id`（翻不出来才允许留 solver 索引）：{units}")
    # 反向仍然成立：物理窗口那本干净
    assert not [i for i in (view.get("skipped_windows") or []) if int(i) >= BASE]


@pytest.mark.parametrize("key", ["skipped_windows", "skipped_sampling_units"])
def test_the_single_segment_view_and_the_merged_view_speak_the_same_dialect(
        tmp_path, key):
    """**同一个量只能有一份口径。** 这个仓库最贵的 bug 形状就是「两份实现」。

    已知漏洞：单段 `read()` 只传 `skipped_windows`，合并视图传两个 ⟹ 非聚合路径
    上的消费者（影子对账、`render`）对子窗结构性失明，而聚合路径上一切正常 ——
    两条路同判才是对账的全部意义（#26 同形）。
    """
    run = _rewindow_board(tmp_path, skipped=(0, 1))
    single = Stage2RepairController(run, "vanishing").read()
    merged = Stage2RepairController.for_physical_stage(run, "vanishing", "vdw").read()

    assert key in single, f"单段视图缺键 `{key}`"
    assert key in merged, f"合并视图缺键 `{key}`"
    assert sorted(single.get(key) or []) == sorted(merged.get(key) or []), (
        f"同一份盘面上两个视图对 `{key}` 给了两个答案："
        f"{single.get(key)} vs {merged.get(key)}")


# =============================================================== comparison manifest

def test_the_manifest_carries_both_namespaces(tmp_path):
    """#58 导出层：`path_result` 里两个命名空间都得有。

    写侧有、导出层漏，是「读一个没人写的键」的镜像 —— 后分析层（`stage2_ab_report`
    等）会对子窗结构性失明，而它们没有别的来源。
    """
    c = Stage2RepairController(_rewindow_board(tmp_path, skipped=(0, 1)), "vanishing")
    res = c.comparison_manifest()["path_result"]
    assert "skipped_windows" in res and "skipped_sampling_units" in res, list(res)
    assert sorted(res["skipped_sampling_units"] or []) == [
        "rw:abc123:0", "rw:abc123:1"], res["skipped_sampling_units"]


def test_an_unknown_production_step_count_is_never_summed_as_zero(tmp_path):
    """代价栏的「未知不是零」。

    `total_production_steps` 是 manifest docstring 里点名的三大必看之一
    （「只比 ΔG 不比代价等于没比」）。把读不到的窗口按 0 步累加 ⟹ A/B 的代价栏
    系统性偏低，而且偏低多少完全看不出来。
    """
    w = {i: {"K": 4, "prod": 250000} for i in range(4)}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)
    _patch_convergence(run, 0, lambda d: d.pop("cumulative_production_steps", None))

    c = Stage2RepairController(run, "vanishing")
    assert next(x for x in c.read()["windows"]
                if x["window_idx"] == 0)["production_steps"] is None, "fixture 没造出未知"

    cost = c.comparison_manifest()["cost"]
    assert cost["total_production_steps"] == 3 * 250000, (
        f"未知窗口被按 0 步计进了总数：{cost}")
    assert cost.get("n_windows_with_unknown_production_steps") == 1, cost
    assert cost.get("total_production_steps_is_lower_bound") is True, (
        "有未知却没标出这个总数是下界 —— 读的人无从知道它偏低了多少：" + str(cost))


def test_the_manifest_filename_is_whatever_the_writer_says_it_is(tmp_path):
    """#65：写侧改名之后，唯一消费者必须仍然读得到。

    **用写侧的返回值，不在测试里硬编码文件名** —— 硬编码正是这条 bug 的成因
    （写侧改名 `stage2_comparison_manifest.json` → `controller_comparison_manifest.json`，
    理由是旧名会被 `_read_stage_result()` 的兜底 glob `stage2_*.json` 吃掉）。
    """
    import stage2_ab_report

    run = _mkrun(tmp_path, windows=FULL, ranges=R4, n_states=13)
    c = Stage2RepairController.for_physical_stage(run, "vanishing", "vdw")
    written = c.write_comparison_manifest()

    assert os.path.isfile(written), f"写侧返回的路径不存在：{written}"
    assert not os.path.basename(written).startswith("stage2_"), (
        f"文件名落回 `stage2_*` 命名空间了 —— `_read_stage_result()` 的兜底 glob "
        f"吃的正是它：{written}")

    got = stage2_ab_report.manifest_for(run, "vanishing", "vdw", refresh=False)
    assert got is not None, (
        f"写侧落在 {os.path.basename(written)}，消费者一份都没读到 —— "
        f"它认的名字是 {stage2_ab_report._MANIFEST_NAMES}")
    with open(written, encoding="utf-8") as fh:
        assert got == json.load(fh), "消费者读到的是**另一份**（旧名残留？）"


# =============================================================== 段枚举：生产路径（#2/#3）

def _segment_board(tmp_path):
    """一个同时含基准段、`_2`、`_10` 和一个 rewindow 子系综目录的盘面。

    rewindow 目录的 identity 故意取 **12 位全数字** —— 旧判据
    `rsplit("_", 1)[-1].isdigit()` 在这里命中（真实 sha256[:12] 有 ≈0.34% 概率
    长这样），于是子系综目录被当采样段合并，**子窗的局部下标 0/1 被当成物理
    窗口 0/1**，覆盖度检查照样通过 ⟹ 静默产出错误 ΔG。
    """
    root = pathlib.Path(tmp_path) / "run"
    for name in ("vanishing", "vanishing_2", "vanishing_10",
                 "vanishing_rewindow_123456789012"):
        d = root / name
        d.mkdir(parents=True)
        # 段必须"有产物"才算段（空壳目录空转循环每轮建一个，真机留下过 39 个）
        (d / "dual_window_0_vdw_convergence.json").write_text(
            json.dumps({"window_idx": 0, "lambdas_vdw": [1.0, 0.9, 0.8, 0.7]}))
    (root / "checkpoints").mkdir()
    return root


def test_the_real_merge_entry_point_ignores_a_rewindow_directory(tmp_path):
    """#2 **在生产路径上**：`_solve_merged_segments_if_any` 不得把子系综当采样段。

    行为判据用的是这个函数自己的早退契约 —— 「只有基准段则返回 None」。所以
    盘面里只放**基准段 + 一个 rewindow 目录**：
      · 旧判据数出 2 个段 ⟹ 往下走进真正的合并求解；
      · 正确判据数出 1 个 ⟹ 早退 None。
    两者的差别是可观测的，不需要跑 MBAR。
    """
    pytest.importorskip("openmm", reason="`_solve_merged_segments_if_any` 在 abfe_pipeline 里")
    from abfe_pipeline import ABFEPipeline

    root = _segment_board(tmp_path)
    import shutil
    shutil.rmtree(root / "vanishing_2")
    shutil.rmtree(root / "vanishing_10")

    pipe = object.__new__(ABFEPipeline)
    pipe._log = lambda *_a, **_k: None
    got = pipe._solve_merged_segments_if_any(
        str(root / "vanishing"), str(root / "checkpoints"),
        [(0, 4)], [1.0, 0.9, 0.8, 0.7], 2.494)
    assert got is None, (
        "`vanishing_rewindow_123456789012` 被当成了第二个采样段 —— "
        "子窗的局部下标会被当成物理窗口下标合并进 ΔG")


def test_the_segment_enumerators_all_route_through_the_one_helper():
    """#2/#3 的**其余枚举点**：helper 存在 ≠ 生产路径用了它。

    ⚠️ **这是源码级断言，平时是坏味道。** 用在这里的理由很具体：这个失效模式
    就是「正确的函数存在，但某个入口没走它」—— 行为断言要一个个把入口喂到能
    观测差异的状态（其中几个要跑完整 MBAR），成本远高于收益，而源码断言对这
    个模式恰好是充分的。
    **什么时候换成行为断言**：等某个入口能像 `_solve_merged_segments_if_any`
    那样被便宜地驱动（有早退契约 / 可注入的枚举结果），就把它从这里挪出去、
    按行为钉。

    用 `ast.unparse` 而不是裸字符串搜索：注释里到处引用着这个旧写法作为"别再
    这么写"的说明，裸搜会全部误报。
    """
    import ast

    for mod in ("abfe_pipeline", "abfe_preoptimizer"):
        src = pathlib.Path(__file__).resolve().parents[1] / f"{mod}.py"
        code = ast.unparse(ast.parse(src.read_text("utf-8")))    # 注释被丢掉
        bad = [ln for ln in code.splitlines()
               if "rsplit" in ln and "isdigit" in ln]
        assert not bad, (
            f"{mod}.py 里还有按 `rsplit('_',1)[-1].isdigit()` 判段号的活代码："
            f"{bad}")
        assert "segment_index_of_dir" in code, (
            f"{mod}.py 一次都没调用 `segment_index_of_dir` —— "
            "helper 存在但没人用，正是这条测试要抓的形状")


def test_segments_are_merged_in_numeric_order_not_string_order(tmp_path):
    """#3：`sorted(glob(...))` 是**字符串序** ⟹ `vanishing_10` 排在 `vanishing_2` 前。

    ΔG 本身不会因此拼错（`solve_stage_integrated` 内部按 lambda_indices 重排），
    但元数据捐赠段（`base = dict(parts[0])`）会选错，且枚举下标写进
    `sampling_source_id` ⟹ 归因表整张对不上号。真机见过 39 个段目录。
    """
    import abfe_preoptimizer as pre

    root = _segment_board(tmp_path)
    base = str(root / "vanishing")
    found = sorted(
        ((pre.segment_index_of_dir(str(d), base), d.name) for d in root.iterdir()
         if d.is_dir() and pre.segment_index_of_dir(str(d), base) is not None),
    )
    assert [n for _i, n in found] == ["vanishing", "vanishing_2", "vanishing_10"], (
        f"段序按字符串排了，或 rewindow 目录混了进来：{found}")


# =============================================================== #11 裁决：改注释不改表

def test_halt_fk_refuted_stays_a_routing_signal_and_says_so(tmp_path):
    """#11 裁决：`HALT_FK_REFUTED` **是路由，不是终态** —— 改注释，别动表。

    理由是它配的动作必须跑：把它加进 `TERMINAL_EXITS` 会让主循环在**分发之前**
    就 break，那次 `RECALIBRATE_FK` 根本不执行 —— 「有统计功效的否决 ⟹ 立刻换
    Epoch」这条规矩就成了空话。所以表是对的，自称终态的注释是错的。
    """
    # 🔑 [2026-09-17] **探针跟着表走。** 原来它用正则在**类源码**里找
    # `"HALT_FK_REFUTED",  # 注释` 这个字面形状；两张手写表已按用户拍板合并成模块级
    # 唯一注册表 `EXIT_SPECS`，理由存进 `why` 字段。**不变量一条没变**
    # （不得是终态、理由必须写明、且不得自称终态），变的只是它存在哪。
    from abfe_preoptimizer import EXIT_SPECS

    assert "HALT_FK_REFUTED" not in TERMINAL, (
        "把它加进 TERMINAL_EXITS 了 —— 主循环会在分发前 break，"
        "它配的 `RECALIBRATE_FK` 永远不跑")

    _spec = EXIT_SPECS["HALT_FK_REFUTED"]
    assert _spec.terminal is False, "注册表里把它声明成终态了"
    assert _spec.default_scope is None, (
        "路由信号不该有 halt_scope —— 那个字段回答的是「**停**的是谁」，"
        "而它根本不停")
    assert _spec.why.strip(), "注册表里这一条的说明没了"
    assert "终态" not in _spec.why or "**是路由**" in _spec.why, (
        f"说明仍把它讲成终态，而它不在 TERMINAL_EXITS 里：{_spec.why}")

    w = dict(FULL)
    w[2] = {"K": 4, "bias_status": "calibrated_validation_failed", "evidence": "refuted"}
    plan = Stage2RepairController(
        _mkrun(tmp_path, windows=w, ranges=R4, n_states=13), "vanishing").decide()
    assert plan["exit"] == "HALT_FK_REFUTED"
    assert plan["terminal"] is False and plan["routing"] is True
    assert plan["action"] == "RECALIBRATE_FK", "路由信号必须带着要执行的那个动作"


# =============================================================== #26 影子对账

def test_the_shadow_reconciliation_path_is_built_the_aggregating_way(tmp_path):
    """#26：**对账的全部意义就是两边同判** —— 而两边判据不同是被**构造方式**决定的。

    `min_n_eff_over_g_history`（边际增益判据的唯一输入）与 `stale_layout_evidence`
    （过期证据保护）只在**聚合视图**（`for_physical_stage` → `read_aggregated`）里
    产出。影子对账先前用普通构造函数 ⟹ 两个键恒 None ⟹ 这两道闸在对账路径上
    **结构性关闭**，而两边结论不同时没人分得清是判据差异还是真差异。

    ⚠️ 上一版这条测试钉错了对象：它断言**普通构造函数自己**必须产出这两个键。
    那是错的 —— 单段视图拿不到跨段历史是**设计**，不是 bug。真正的不变量是
    「对账路径必须用会产出它们的那个构造」，所以钉两件事：
      ① `abfe_pipeline` 里不得再出现普通构造（结构性，AST 判，不受注释干扰）；
      ② 聚合构造在有历史的盘面上**确实**产出这两个键 —— 否则 ① 只是拜物。
    """
    import ast

    src = (pathlib.Path(__file__).resolve().parents[1] / "abfe_pipeline.py"
           ).read_text("utf-8")
    plain = [n.lineno for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "Stage2RepairController"]
    assert not plain, (
        f"abfe_pipeline.py:{plain} 又用普通构造函数造控制器 —— 那份视图里 "
        "`min_n_eff_over_g_history` / `stale_layout_evidence` 恒 None，"
        "边际增益判据与过期证据保护在这条路径上结构性关闭")

    run = _mkrun(
        tmp_path,
        windows={0: {"K": 4, "self_verdict": "INSUFFICIENT_DATA",
                     "prod": 500000, "min_n_eff_over_g": 3.8},
                 1: {"K": 4}, 2: {"K": 4}, 3: {"K": 4}},
        ranges=R4, n_states=13)
    (_ck(run) / "stage2_autonomous_history.json").write_text(json.dumps({
        "iterations": [{
            "iteration": 1, "action": "RUN_PRODUCTION", "path_version": 1,
            "snapshot": [{"window_idx": 0, "segment": "vanishing",
                          "production_steps": 250000, "min_n_eff_over_g": 4.5}]}]}))

    view = Stage2RepairController.for_physical_stage(run, "vanishing", "vdw").read()
    assert view.get("min_n_eff_over_g_history"), (
        "聚合构造也拿不到逐块支撑历史 ⟹ 边际增益判据没有输入，"
        "上面那条「必须用聚合构造」的断言就只是拜物")
    assert "stale_layout_evidence" in view, list(view)


# =============================================================== 末窗豁免 / 拆窗语义

def test_both_kinds_of_tail_split_retire_the_overflow_exemption():
    """#29：末窗是 model B 的**溢出槽**，被拆过之后豁免作废。

    `feasible_repair_actions(tail_exempt_from_max=…)` 先前是个**死参数**（全仓
    没有任何调用方传过）⟹ 恒 `True` ⟹「拆过末窗 ⟹ 豁免作废 ⟹
    `HALT_LAMBDA_BUDGET_INSUFFICIENT`」那条分支结构上不可达，插 λ 在溢出槽用尽
    之后仍被判可行。

    ⚠️ **两种拆窗事件都算数**：路径演化分支写 `split_tail_window`，自治分支写
    `tail_repartition`（`record_tail_repartition_version`）。只数一种 = 漏掉自治
    循环拆的每一次 —— 而自治循环才是拆得最勤的那条路。
    """
    import abfe_preoptimizer as pre

    fn = getattr(pre, "tail_overflow_exemption_active", None)
    if fn is None:
        pytest.xfail("等 #29：模块级 `tail_overflow_exemption_active()` 还没落盘")

    assert fn({}) is True and fn(None) is True, "没拆过 ⟹ 末窗仍是溢出槽"
    assert fn({"insert_lambda": 7}) is True, "插 λ 不消耗溢出槽的豁免身份"
    assert fn({"split_tail_window": 1}) is False, "路径演化分支拆过了"
    assert fn({"tail_repartition": 1}) is False, (
        "自治分支写的是 `tail_repartition` —— 只数 `split_tail_window` "
        "等于漏掉自治循环拆的每一次")
    assert fn({"insert_lambda": 3, "tail_repartition": 1}) is False


def test_split_tail_window_is_judged_by_what_the_action_actually_does():
    """#24：判据问的问题必须和动作做的事是同一件事。

    控制器的 `SPLIT_TAIL_WINDOW` 落到执行器是 `repartition_tail_from_anchor`
    （从 anchor 首态起**重分整个尾段**），不是「把末窗一分为二」。拿后者冒充的
    后果是双向的：标准 23 态布局末窗 K=4 < 7 ⟹ 动作**永久不可行**，主力修复
    路径一次都没被选中过；反过来被放行时，改动范围远大于「拆末窗」，而可行性
    检查对作废的下游证据一无所知。
    """
    from abfe_preoptimizer import feasible_repair_actions as feas

    kw = dict(min_states_per_window=2, max_states_per_window=4)

    # ① 给不出重分起点 ⟹ **如实说判不了**，不拿末窗那条冒充
    d = feas(R4, 13, **kw)
    assert isinstance(d["split_tail_window"], str), (
        f"没给 `tail_repartition_start_state` 却给了裁决：{d['split_tail_window']}")
    assert "split_last_window_in_two" in d["split_tail_window"], (
        "「判不了」的理由里必须指出**另一个问题**的答案在哪个键，"
        "否则读的人只会以为动作不可行：" + d["split_tail_window"])

    # ② 两个键是**两个独立的问题**，同一盘面上可以给不同答案
    assert d["split_last_window_in_two"] is None, (
        "fixture 没造出「末窗可一分为二」—— 这条测不到独立性")
    assert d["split_tail_window"] != d["split_last_window_in_two"]

    # ③ 给了合法起点 ⟹ 按「尾段能否切成**更细**的合法分窗」判
    #    从态 0 起：13 个态、现 4 个窗口；m=5 时 5·1+1=6 ≤ 13 ≤ 5·3+1=16 ⟹ 可行
    assert feas(R4, 13, tail_repartition_start_state=0, **kw)["split_tail_window"] is None

    # ④ 起点不是任何窗口的首态 ⟹ 不是共享边界态，执行器会 fail-closed 抛错
    bad = feas(R4, 13, tail_repartition_start_state=5, **kw)["split_tail_window"]
    assert isinstance(bad, str) and "首态" in bad, bad


# =============================================================== 两张表的字段边界

def test_the_window_table_and_the_unit_table_do_not_share_each_other_s_keys(tmp_path):
    """**窗口记录与子窗记录是两张表**，把一张的键读到另一张上只会静默拿到 None。

    一天之内栽了三次：`segment` 在子窗上不存在；`solver_n_decorrelated` 其实叫
    `solver_n_frames_decorrelated`（**两张表上都没有前者**）；`identity` 只在子窗上、
    却被 `action_noop_fingerprint(window_record, …)` 的形参名当成窗口字段。
    三次都不抛异常 —— `dict.get` 返 None，判据静默失效。

    ⚠️ 键名全部**从真实 view 读出来的**，不是照意图写的。要改这里的名单，
    先跑一次 `read()` 把差集打出来，别凭印象加。
    """
    run = _rewindow_board(tmp_path)
    for label, ctl in (
            ("single", Stage2RepairController(run, "vanishing")),
            ("merged", Stage2RepairController.for_physical_stage(
                run, "vanishing", "vdw"))):
        view = ctl.read()
        win = view["windows"][0]
        unit = view["sampling_units"][1]

        # —— 子窗专属：调度身份与它自己的目录。窗口记录上没有 ——
        for k in ("unit_id", "identity", "kind", "local_index", "parent_window",
                  "parent_range", "range", "all_child_ranges", "output_dir",
                  "checkpoint_dir", "solver_index", "status", "schedulable",
                  "complete", "needs_frames", "support_failed",
                  "meets_final_solver_gate"):
            assert k in unit, f"[{label}] 子窗表少了 `{k}`"
            assert k not in win, (
                f"[{label}] 子窗专属键 `{k}` 出现在窗口记录上 —— "
                "两张表的边界破了，读点会在另一张上静默拿到 None")

        # —— 窗口专属：λ 布局、预热账、逐段证据。子窗记录上没有 ——
        for k in ("window_idx", "lambdas_vdw", "lambda_span", "n_states",
                  "bias_status", "f_k_evidence_status", "n_frames",
                  "n_production_segments", "warmup_steps_cap",
                  "warmup_steps_spent", "cum_fk_verdict", "verdict"):
            assert k in win, f"[{label}] 窗口表少了 `{k}`"
            assert k not in unit, (
                f"[{label}] 窗口专属键 `{k}` 出现在子窗记录上："
                "子窗有自己的 `range` / `production_steps`，别拿窗口那套去读它")

        # —— 两张表都有：调度判据必须能对两种单元同口径地问 ——
        for k in ("production_steps", "min_n_eff_over_g", "self_verdict",
                  "self_verdict_source", "self_sufficient", "phase",
                  "warmup_steps_left", "has_convergence", "solver_skip",
                  "solver_n_frames_decorrelated",
                  "solver_min_decorrelated_samples_threshold"):
            assert k in win and k in unit, (
                f"[{label}] `{k}` 只在一张表上 ⟹ 调度判据对另一种采样单元失明")

        # `segment` 只有**合并视图**的窗口记录才有；子窗**从来**没有。
        # 拿它去 `seg_of.get(unit)` 会恒取 None ⟹ 同段过滤对子窗恒不成立。
        assert "segment" not in unit, "子窗上出现了 `segment` —— 它住在自己的目录里"
        assert ("segment" in win) is (label == "merged"), (
            f"[{label}] `segment` 的产出条件变了：单段视图不该有、合并视图必须有")

        # 反例：这个名字**两张表上都不存在**，写错了只会静默拿 None
        assert "solver_n_decorrelated" not in set(win) | set(unit), (
            "`solver_n_decorrelated` 复活了 —— 真名是 `solver_n_frames_decorrelated`")


# ---------------------------------------------------------------------------
# [2026-09-17] 补帧准入的第三道：剩余配额的乐观上界够不着门 ⟹ 停
# ---------------------------------------------------------------------------

def _rising_but_short_board(tmp_path, blocks):
    """真机 cyclod_ligand1_outer/rep1 win0 的形状：逐块**在涨**，但涨不到门。

    `min N_eff/g` 逐块 4.459 / 4.553 / 5.287，门 10，补帧块上限 4。
    `marginal_gain_stalled` 的判据是「末点 < 前面各点中位数 × 0.9」⟹ 单调上升
    恒判「还在涨」、刹车永不响；按 +0.41/块要 11 块才够，而配额只有 4 块。
    """
    series = [4.459, 4.553, 5.287][:blocks]
    run = _mkrun(
        tmp_path,
        windows={0: {"K": 4, "self_verdict": "INSUFFICIENT_DATA",
                     "self_verdict_source": "min_n_eff_over_g",
                     "min_n_eff_over_g": series[-1], "n_decorr": 400,
                     "prod": 250000 * blocks},
                 1: {"K": 4}, 2: {"K": 4}, 3: {"K": 4}},
        # 门 = 10。写侧把它落在 `n_eff_over_g_eligible_threshold`，读侧是
        # `self_n_eff_over_g_eligible` —— 缺了它射程判不了（返回 None），
        # `support_failure_is_skew` 会保守判偏斜，整条准入路径根本走不到。
        self_extra={0: {"n_eff_over_g_eligible_threshold": 10.0}},
        ranges=R4, n_states=13,
        config={"stage2_window_min_states": 4, "stage2_window_max_states": 8,
                "max_path_insertions": 3,
                "stage2_max_production_blocks_per_window": 4},
    )
    (_ck(run) / "stage2_autonomous_history.json").write_text(json.dumps({
        "iterations": [
            {"iteration": k + 1, "action": "RUN_PRODUCTION", "path_version": 1,
             "snapshot": [{"window_idx": 0, "segment": "vanishing",
                           "production_steps": 250000 * (k + 1),
                           # 去相关帧数也单调上升 ⟹ 老刹车同样不响
                           "solver_n_decorrelated": 400 + 40 * k,
                           "min_n_eff_over_g": series[k]}]}
            for k in range(blocks)]}))
    return run


def test_frames_stop_when_the_whole_remaining_quota_cannot_reach_the_gate(tmp_path):
    """DEAD_LINES §3.1：前向判据乐观、事后刹车失灵 ⟹ 这一对少了一半。

    `n_eff_over_g_reachable_by_frames` 的 docstring 自己写着它是**上界**、
    「`True` 只表示值得一试」，并且「真正的刹车是事后的 `marginal_gain_stalled()`
    —— 两者一前一后，**缺一不可**」。而那个刹车对**单调上升但渐近在门以下**的
    序列永不触发 ⟹ 进的时候乐观、出的时候没人管，白烧 GPU。

    补的这一道**不引入新阈值、也不拟合斜率**（那个量单次噪声 34×，3 个点拟斜率
    同样在量噪声）：就是同一个前向判据，拿**现在的读数**和**剩下的配额**再问一次。

    真机数代进去：已批 2 块 ⟹ 剩余帧数倍率 (1+4)/(1+2)=1.67，
    4.553×1.67=7.6 < 门 10 ⟹ 连最乐观的情形都够不着 ⟹ 必须停。
    """
    run = _rising_but_short_board(tmp_path, blocks=2)
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] != "RUN_PRODUCTION", (
        f"剩余配额全花掉也够不着门，还在补帧：{plan['reason']}")
    # 而且**不是**停下来，是转去对症动作：射程判据与 `support_failure_is_skew`
    # 共用同一个函数 ⟹ 射程一旦够不着，归因立刻从「样本量类」翻成「偏斜类」，
    # 分支侧就先发缩跨度了（准入层是给**其余十几个**不查归因的补帧入口兜底的，
    # 见下一条）。
    assert plan["action"] in ("INSERT_LAMBDA", "SPLIT_TAIL_WINDOW",
                             "IMMUTABLE_REWINDOW"), plan["reason"]


def test_the_admission_choke_point_also_stops_entries_that_skip_attribution(tmp_path):
    """准入层是**咽喉**：`decide()` 有十几个发 `RUN_PRODUCTION` 的出口，
    其中只有「自检归因」那两条会问射程，其余（缺窗/跳窗/旧布局/步数未达标…）
    一概不问 —— 那正是 FLOW 的 D1「补帧 11 条入口、缩跨度 5 条且全部带闸」。

    所以射程判据必须挂在 `_frames_admission`（全部入口的唯一咽喉），
    不能只挂在归因上。这条走「求解器跳窗 ⟹ 补帧」那条入口，它不查归因。
    """
    run = _rising_but_short_board(tmp_path, blocks=2)
    (_ck(run) / "stage2_vanishing.json").write_text(json.dumps({
        "analysis_status": "ANALYSIS_INCOMPLETE",
        "analysis_incomplete_reasons": ["路径缺窗：求解器少解出一个窗口。"],
        "skipped_windows": [{"window_index": 0,
                             "reason": "n_decorrelated 太少"}],
    }))
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] != "RUN_PRODUCTION", (
        f"跳窗入口绕过了射程判据：{plan['action']}｜{plan['reason']}")
    assert "够不着门" in plan["reason"], plan["reason"]


def test_a_reachable_gate_is_unknown_and_authorizes_nothing(tmp_path):
    """[2026-09-17 P0，取代旧语义] 射程**够得着**只说明「尚未被证伪」⟹ `UNKNOWN`。

    本条原名 `test_frames_still_admitted_while_the_gate_is_optimistically_in_range`，
    断言的是「射程够得着 ⟹ 必须放行补帧」。那把一个**乐观上界**当成了"这是样本量
    问题"的结论 —— 而该上界假定 η 与 g 恒定，两个假设实测都朝不利方向走。

    新规则：只有**明确的帧数硬证据**（`solver_eligibility` 或 `n_decorrelated <
    min_frames`）才授权补帧；`reachable is True` 与 `reachable is None` 都是
    `UNKNOWN`，**两边都不授权**（不补帧、也不改布局）。

    ⚠️ 防作弊的那一半改由下一条守着：帧数硬证据的盘面**仍然**必须补帧。
    """
    run = _rising_but_short_board(tmp_path, blocks=1)
    plan = Stage2RepairController(run, "vanishing").decide()
    assert not (plan["action"] == "RUN_PRODUCTION" and plan["windows"] == [0]), (
        f"射程只是「尚未被证伪」，却被当成样本量证据放行了补帧：{plan['reason']}")


def test_explicit_frame_count_evidence_still_buys_frames(tmp_path):
    """反面（防「一律不补帧」）：**明确**帧数不足的窗口仍然必须拿到帧。

    `solver_eligibility` 是硬证据 —— 连喂进 MBAR 的去相关样本都不够，
    补帧就是对症动作。三态收紧不得波及它。
    """
    w = {i: {"K": 4} for i in range(4)}
    w[0] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA",
            "self_verdict_source": "solver_eligibility", "n_decorr": 5}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] == "RUN_PRODUCTION" and plan["windows"] == [0], (
        f"帧数硬证据也不补帧了 —— 三态收紧波及了它：{plan['reason']}")
