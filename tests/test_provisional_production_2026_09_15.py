"""临时生产（`PROVISIONAL_PRODUCTION`）：撞 15 批验证上限的窗口必须还有出路。

真机 cyclod_ligand2/rep2 win4（2026-09-15）死锁的三环，每一环一条钉子：

1. `windows_by_segment` 遇到 `production_steps is None` 直接 `continue` ⟹
   执行器循环空转、`run_once` 一次都没调 ⟹ 「执行完但盘面逐项未变」。
   （钉在 `tests/test_stage2_autonomous_segment_routing.py`）
2. no-op 指纹把未知步数写成 `?` ⟹ **从未生产过**的窗口指纹恒定 ⟹
   挂在它上面的 no-op 记录在结构上永不失效。
3. `PROVISIONAL_PRODUCTION` 只存在于注释里 —— 没有动作、没有状态、没有执行器，
   于是即便前两环修好，引擎走到 15/15 仍然抛路由信号，窗口永远进不了生产。
"""
import inspect
import json
import os

import abfe_pipeline
import abfe_preoptimizer as pre
import ibs_engine as ie
from abfe_preoptimizer import Stage2RepairController

from test_stage2_repair_controller import R4, _mkrun


# ── 2) no-op 指纹：未知步数 = 没有可比身份 ────────────────────────────────
def test_unknown_production_steps_has_no_comparable_identity():
    """`?` == `?` ⟹ 旧写法给「从未生产过」的窗口判了无期。"""
    assert pre.action_noop_fingerprint(
        {"production_steps": None, "segment": "vanishing"}, 1) is None
    # 真的跑了 0 步是**已知**，必须仍有身份，且与未知区分得开
    assert pre.action_noop_fingerprint(
        {"production_steps": 0, "segment": "vanishing"}, 1) == "1|0|vanishing"


def test_both_sides_refuse_to_use_an_identityless_fingerprint():
    """写侧不落账、读侧不认作匹配 —— 两边必须同时守，漏一边就还是锁死。"""
    reader = inspect.getsource(Stage2RepairController.decide)
    assert "fp is not None and rec.get(\"fingerprint\") == fp" in reader
    writer = inspect.getsource(abfe_pipeline.ABFEPipeline._record_noop_action)
    assert "if _fp is None:" in writer and "continue" in writer


# ── 3) 动作 / 状态 / 执行器三者都要真的存在 ───────────────────────────────
def test_the_action_exists_and_is_charged_like_a_production_block():
    assert "PROVISIONAL_PRODUCTION" in Stage2RepairController.ACTIONS
    # 它采的是**生产**帧 ⟹ 必须进块账，否则就是绕过块数硬上限的旁路
    assert "PROVISIONAL_PRODUCTION" in Stage2RepairController._BLOCK_CHARGING_ACTIONS
    # 也要挂 no-op 刹车，跟其它会烧 GPU 的动作一致
    src = inspect.getsource(Stage2RepairController.decide)
    assert '"PROVISIONAL_PRODUCTION",\n' in src


def test_the_engine_accepts_an_explicit_per_window_authorisation():
    """引擎默认仍然 fail-closed；授权必须是调用方**显式逐窗**给的。"""
    sig = inspect.signature(ie.IBSWindowManagerDualLambda.run_all_windows)
    assert "provisional_production_windows" in sig.parameters
    assert sig.parameters["provisional_production_windows"].default is None
    # 全链路透传：pipeline 的 stage 入口也要有
    assert "provisional_production_windows" in inspect.signature(
        abfe_pipeline.ABFEPipeline._run_dual_lambda_stage).parameters


def test_the_executor_has_a_branch_that_passes_the_authorisation():
    src = inspect.getsource(abfe_pipeline.ABFEPipeline._run_stage2_autonomous)
    assert 'elif act == "PROVISIONAL_PRODUCTION":' in src
    assert "_provisional_production_windows=sorted(overrides)" in src


def test_provisional_is_never_reported_as_a_verified_pass():
    """**最要紧的一条**：临时生产绝不能被写成 converged / verified。"""
    src = inspect.getsource(ie.IBSWindowManagerDualLambda.run_all_windows)
    assert '"provisional_production" if _prov else "converged"' in src
    assert "F_K_EVIDENCE_INDETERMINATE if _prov else F_K_EVIDENCE_VERIFIED" in src
    # 证据文件不得被当成"陈旧失败记录"删掉
    assert "elif os.path.exists(stale_failure_path):" in src
    # 控制器词汇表：indeterminate 永远不是 VALID_PASS
    assert Stage2RepairController._VERDICT["indeterminate"] == "INSUFFICIENT_DATA"
    assert Stage2RepairController._PHASE["provisional_production"] == "PRODUCTION"


# ── 真实盘面：win4 撞上限、预算有余、从未生产 ─────────────────────────────
def _win4_board(tmp_path, *, bias_status="frozen_validation_indeterminate"):
    """复刻 cyclod_ligand2/rep2 的盘面：win0-3 合格，win4 撞 15/15 且从未生产。"""
    windows = {i: {"K": 4} for i in range(4)}
    windows[4] = {"K": 4, "prod": None, "bias_status": bias_status,
                  "evidence": "indeterminate", "warmup": 305_000, "cap": 955_000}
    run = _mkrun(tmp_path, windows=windows, ranges=R4 + [(12, 16)], n_states=16)
    sd = os.path.join(run, "vanishing")
    ck = os.path.join(run, "checkpoints")
    # 从未生产过的窗口盘上**没有** convergence.json，只有 warmup_failure.json
    os.remove(os.path.join(sd, "dual_window_4_vdw_convergence.json"))
    with open(os.path.join(sd, "dual_window_4_vdw_warmup_failure.json"), "w") as fh:
        json.dump({
            "status": "validation_budget_exhausted_indeterminate",
            "warmup_budget_ledger": {"learning_steps": 80_000,
                                     "freeze_burn_in_steps": 15_000,
                                     "frozen_validation_steps": 210_000,
                                     "cumulative_cap_steps": 955_000},
            "warmup_budget_remaining_steps": 650_000,
            "validation_indeterminate": {
                "reason": "validation_budget_exhausted_indeterminate",
                "n_batches": 15, "max_batches": 15,
                "validation_sample_count": 600,
                "validation_g_history": [40.15, 77.25, 167.29],
                "decorrelated_frames_required": 10,
            },
        }, fh)
    p = os.path.join(ck, "ibs_state_vdw_window_4.json")
    with open(p) as fh:
        st = json.load(fh)
    st["frozen_validation_batches_done"] = 15
    st["frozen_validation_cumulative_steps"] = 305_000
    with open(p, "w") as fh:
        json.dump(st, fh)
    return run


def test_a_capped_never_produced_window_gets_a_provisional_block(tmp_path):
    """这是整条死锁链的出口：撞上限 + 预算有余 ⟹ 发一块临时生产，不是停机。"""
    c = Stage2RepairController(_win4_board(tmp_path), "vanishing")
    view = c.read()
    w4 = next(w for w in view["windows"] if w["window_idx"] == 4)
    assert w4["production_steps"] is None, "fixture 没造对：win4 应当从未生产过"
    assert w4["frozen_validation_batches"] == 15
    assert (w4["warmup_steps_left"] or 0) > 0, "全局预算必须还有钱，否则不是这条分支"

    plan = c.decide()
    assert plan["action"] == "PROVISIONAL_PRODUCTION", (
        f"{plan['action']}／{plan['exit']}｜{plan['reason'][:300]}")
    assert plan["windows"] == [4]
    # 「没测出来」永远不是「不合格」
    assert plan["evidence_status"] == "INSUFFICIENT_DATA"
    assert plan["exit"] == "HALT_LOCAL_VALIDATION_CAP"
    assert not plan.get("terminal"), "路由信号不是终态"


def test_the_one_block_is_not_handed_out_twice(tmp_path):
    """已经用掉那一块的窗口不再发这个动作 —— 否则就是"反复试到偶然通过"。"""
    c = Stage2RepairController(
        _win4_board(tmp_path, bias_status="provisional_production"), "vanishing")
    plan = c.decide()
    assert plan["action"] != "PROVISIONAL_PRODUCTION", plan["reason"][:300]


def test_a_stale_noop_record_cannot_pin_a_never_produced_window(tmp_path):
    """第 2 环的端到端形态：台账里有 win4 的旧 no-op 记录也不得把它钉死。"""
    run = _win4_board(tmp_path)
    with open(os.path.join(run, "checkpoints", "stage2_noop_actions.json"), "w") as fh:
        json.dump({"PROVISIONAL_PRODUCTION:4": {
            "action": "PROVISIONAL_PRODUCTION", "window_idx": 4,
            "reason": "action_changed_nothing_on_disk",
            "fingerprint": "1|?|vanishing"}}, fh)
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] == "PROVISIONAL_PRODUCTION", plan["reason"][:300]


# ── 标记必须跟着数据走到最终结果，不能只活在日志里 ────────────────────────
def test_the_flag_travels_from_convergence_json_to_the_stage_result():
    loader = inspect.getsource(ie.load_ibs_window_outputs_from_dir)
    assert 'convergence.get("provisional_production", False)' in loader, (
        "窗口加载器没把 convergence.json 的标记带出来")
    solver = inspect.getsource(ie.GlobalMBARAnalyzer.solve_stage_integrated)
    assert '"provisional_production_windows"' in solver, "stage result 没汇总"
    gate = inspect.getsource(abfe_pipeline.ABFEPipeline._assert_stage_result_sane)
    assert 'result.get("provisional_production_windows")' in gate
    assert "不是可信 PASS" in gate


def test_the_engine_records_the_flag_on_the_window_it_produced():
    src = inspect.getsource(ie.IBSWindowManagerDualLambda.run_all_windows)
    assert '"provisional_production": bool(provisional_production),' in src, (
        "convergence.json 没落这个标记 ⟹ 标记到不了求解器")


def test_the_hard_invariant_is_not_weakened_by_the_flag():
    """路径完整性与 f_k 证据强度是两维 —— 标记**不得**进 analysis_status 的合取。"""
    solver = inspect.getsource(ie.GlobalMBARAnalyzer.solve_stage_integrated)
    body = solver.split('"analysis_status": analysis_status')[0]
    assert "provisional" not in body.split("analysis_status =")[-1], (
        "provisional 混进了 analysis_status 的计算")
