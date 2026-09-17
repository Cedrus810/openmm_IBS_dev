"""裁决 1：immutable rewindow 的子窗是**控制器可调度的采样单元**。

`_immutable_rewindow_step()` 原本是一次性动作：建完子系综就没有下一步 ——
控制器看不见子窗（`_segment_stage_names()` 只认 `vanishing_<数字>`，而子系综在
`vanishing_rewindow_<id>`），`decide()` 发不出针对子窗的动作，子窗只有 11/20
也补不上帧，只能靠停滞保护退出。

现在：`view["windows"]` 仍是**物理窗口**，子窗另立 `view["sampling_units"]`，
每个带稳定 `unit_id`、父窗、局部下标、区间、目录、solver 索引。
⚠️ **不能靠放宽段发现来做** —— 子窗局部下标 0/1 与父窗索引**同名不同义**，
`10000` 偏移只是求解器命名空间、不是调度身份。
"""
import json
import os
import pathlib
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abfe_preoptimizer import Stage2RepairController  # noqa: E402

from test_stage2_repair_controller import _mkrun, R4  # noqa: E402


BASE = 10_000          # 台账里钉死的 solver_index_base


def _with_rewindow(tmp_path, *, child_states, skipped=(), solver_decorr=None):
    """父窗 1 被子系综取代。

    `child_states` 逐子窗给 `(self_verdict, prod_steps)`；
    `solver_decorr` 逐子窗给**求解器最终门**看到的去相关帧数（门槛 20）。
    """
    solver_decorr = solver_decorr or [200] * len(child_states)
    run = _mkrun(
        tmp_path,
        windows={i: {"K": 4} for i in range(4)},
        ranges=R4, n_states=13,
        # 🔑 [2026-09-15] `converged` 已删键。写老键的 stage_result **根本不会被
        # `_read_stage_result()` 认成一份 stage 结果**（嗅探键是 `analysis_status` /
        # `total_delta_G`）⟹ 整个 view 里没有求解器证据 ⟹ 下面那些
        # `solver_*` 字段全是 None，子窗完成度只好退回读自检产物的帧数，
        # 于是"哪个子窗没完成"整个判反。
        stage_result={"analysis_status": "ANALYSIS_INCOMPLETE",
                      "analysis_incomplete_reasons": [
                          "存在被跳过的窗口；缺窗口的总和是部分和，不是完整 ΔG。"],
                      "total_delta_G": -12.3, "total_error": 0.9,
                      # 最终门：`min_decorrelated_samples`（20），**不是**入场下限 10
                      "min_decorrelated_samples_threshold": 20,
                      "window_overlap_diagnostics": [
                          {"window_index": BASE + i,
                           "n_frames_decorrelated": int(n)}
                          for i, n in enumerate(solver_decorr)],
                      "skipped_windows": [
                          {"window_index": BASE + i,
                           "n_frames_after_decorrelation": 7,
                           "min_frames_per_window": 10,
                           "reason": "insufficient_frames_after_decorrelation"}
                          for i in skipped]},
    )
    rw = pathlib.Path(run) / "vanishing_rewindow_abc123"
    rwck = pathlib.Path(run) / "checkpoints" / "rewindow_abc123"
    rw.mkdir(parents=True)
    rwck.mkdir(parents=True)
    child_ranges = [[3, 6], [5, 8]]
    for li, (verdict, prod) in enumerate(child_states):
        lam = [1.0 - 0.1 * k for k in range(child_ranges[li][1] - child_ranges[li][0])]
        (rw / f"dual_window_{li}_vdw_convergence.json").write_text(json.dumps({
            "window_idx": li, "lambdas_vdw": lam,
            "cumulative_production_steps": prod,
            "n_steps_per_window_effective": prod,
            "production_segments": [{}],
            "window_data": {"n_frames": 500},
            "bias_warmup": {"warmup_budget_ledger": {
                "learning_steps": 50000, "cumulative_cap_steps": 555000},
                "bias_update_count": 12},
        }))
        (rwck / f"ibs_state_vdw_window_{li}.json").write_text(json.dumps({
            "bias_status": "converged", "f_k_evidence_status": "verified",
            "lambdas_vdw": lam,
        }))
        # ⚠️ [2026-09-17] `verdict_source` 与 `n_eff_over_g_eligible_threshold`
        # **真实写侧一定会写**（`ibs_engine.window_self_support_check` 里
        # verdict_source 是 top1pct_veto / min_n_eff_over_g / solver_eligibility
        # 三选一）。fixture 先前不写它们 ⟹ 归因恒判不出来 ⟹ 三态里只出现
        # `UNKNOWN`，测的是一个**真机不存在**的盘面。
        # 这里按真写侧补齐：11 帧 < 下限 20 ⟹ 真实写侧会判 `solver_eligibility`。
        (rw / f"dual_window_{li}_vdw_self_support.json").write_text(json.dumps({
            "window_idx": li, "verdict": verdict,
            "verdict_source": ("min_n_eff_over_g" if verdict == "ANALYSIS_ELIGIBLE"
                               else "solver_eligibility"),
            "sufficient": verdict == "ANALYSIS_ELIGIBLE",
            "n_frames_decorrelated": 11, "min_frames_per_window": 20,
            "min_n_eff_over_g": 4.0,
            "n_eff_over_g_eligible_threshold": 10.0,
        }))
    (pathlib.Path(run) / "checkpoints" / "stage2_rewindow_ledger.json").write_text(
        json.dumps({"abc123": {
            "identity": "abc123", "parent_window": 1,
            "solver_index_base": BASE,
            "parent_range": [3, 8], "child_ranges": child_ranges,
            "output_dir": str(rw), "checkpoint_dir": str(rwck),
            "f_k_scope": "own_frozen_f_k_per_child_ensemble",
            "blocks": [{"steps_per_child": 250000}],
        }})
    )
    return run


def test_children_become_first_class_sampling_units(tmp_path):
    run = _with_rewindow(tmp_path, child_states=[
        ("ANALYSIS_ELIGIBLE", 250000), ("INSUFFICIENT_DATA", 250000)])
    view = Stage2RepairController(run, "vanishing").read()

    units = view["sampling_units"]
    assert [u["unit_id"] for u in units] == ["rw:abc123:0", "rw:abc123:1"]
    u1 = units[1]
    assert u1["parent_window"] == 1 and u1["local_index"] == 1
    assert u1["range"] == [5, 8]
    assert u1["solver_index"] == BASE + 1          # 只是求解器命名空间
    assert u1["production_steps"] == 250000
    assert u1["complete"] is False               # 自检判不够
    assert units[0]["complete"] is True

    # 物理窗口表不受影响：仍然是 4 个，局部下标没污染它
    assert [w["window_idx"] for w in view["windows"]] == [0, 1, 2, 3]


def test_an_incomplete_child_gets_a_run_production_carrying_its_unit_id(tmp_path):
    run = _with_rewindow(tmp_path, child_states=[
        ("ANALYSIS_ELIGIBLE", 250000), ("INSUFFICIENT_DATA", 250000)])
    plan = Stage2RepairController(run, "vanishing").decide()

    assert plan["action"] == "RUN_PRODUCTION", plan["reason"]
    assert plan["unit_id"] == "rw:abc123:1", plan["reason"]
    assert plan["windows"] == [1]               # 父窗号只做展示/因果排序
    assert not plan["terminal"]


def test_a_solver_skipped_child_is_also_incomplete(tmp_path):
    """求解器把子窗踢出协方差链 ⟹ 同样是未完成，按 `solver_index` 映回来。"""
    run = _with_rewindow(
        tmp_path,
        child_states=[("ANALYSIS_ELIGIBLE", 250000), ("ANALYSIS_ELIGIBLE", 250000)],
        skipped=(1,),
    )
    view = Stage2RepairController(run, "vanishing").read()
    u1 = view["sampling_units"][1]
    assert u1["solver_skip"]["n_frames_after_decorrelation"] == 7
    assert u1["complete"] is False
    assert Stage2RepairController(run, "vanishing").decide()["unit_id"] == "rw:abc123:1"


def test_all_children_done_stops_routing_to_units(tmp_path):
    run = _with_rewindow(tmp_path, child_states=[
        ("ANALYSIS_ELIGIBLE", 250000), ("ANALYSIS_ELIGIBLE", 250000)])
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan.get("unit_id") is None, plan["reason"]


def test_a_child_that_only_clears_the_entry_floor_is_not_complete(tmp_path):
    """**11/20 的子窗必须仍然待补。**

    跳窗门是**入场**下限（10 帧）；求解器一旦放它进协方差链，先前的完成条件
    （自检通过 + 没被跳窗）就把它判成"已完成"、从待补列表消失。可最终门要的是
    `min_decorrelated_samples`（20）—— 于是最终门报
    `worst_window = solver 索引`，9c 发出**不带 `unit_id`** 的普通补帧，
    执行器把一个越界的窗口号交给原窗口路径。
    """
    run = _with_rewindow(
        tmp_path,
        child_states=[("ANALYSIS_ELIGIBLE", 250000), ("ANALYSIS_ELIGIBLE", 250000)],
        solver_decorr=[200, 11],          # 子窗 1 过了入场 10、没过最终 20
    )
    view = Stage2RepairController(run, "vanishing").read()
    u0, u1 = view["sampling_units"]

    assert u0["complete"] is True
    assert u1["solver_n_frames_decorrelated"] == 11
    assert u1["solver_min_decorrelated_samples_threshold"] == 20
    assert u1["meets_final_solver_gate"] is False
    assert u1["complete"] is False, "只过入场下限就被判完成 —— 正是审查点名的漏洞"

    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] == "RUN_PRODUCTION"
    assert plan["unit_id"] == "rw:abc123:1", plan["reason"]


def test_missing_final_gate_reading_is_not_a_pass(tmp_path):
    """读不到最终门的读数 ⟹ **不算完成**（缺证据 ≠ 通过）。"""
    run = _with_rewindow(
        tmp_path,
        child_states=[("ANALYSIS_ELIGIBLE", 250000), ("ANALYSIS_ELIGIBLE", 250000)],
        solver_decorr=[200, 200],
    )
    # 把 overlap 诊断整段拿掉 —— 最终门无从判起
    p = pathlib.Path(run) / "checkpoints" / "stage2_vanishing.json"
    d = json.loads(p.read_text())
    d.pop("window_overlap_diagnostics")
    p.write_text(json.dumps(d))

    units = Stage2RepairController(run, "vanishing").read()["sampling_units"]
    assert all(u["meets_final_solver_gate"] is None for u in units)
    assert all(u["complete"] is False for u in units)


def test_solver_index_is_unique_per_identity(tmp_path):
    """两组子窗不许串号 —— 基数从台账读（建窗时钉死），不是所有 identity 共用 10000。"""
    run = _with_rewindow(tmp_path, child_states=[("ANALYSIS_ELIGIBLE", 250000)])
    led = pathlib.Path(run) / "checkpoints" / "stage2_rewindow_ledger.json"
    d = json.loads(led.read_text())
    d["def456"] = dict(d["abc123"], identity="def456", parent_window=2,
                       solver_index_base=10_100, parent_range=[7, 11],
                       child_ranges=[[7, 10], [9, 11]])
    led.write_text(json.dumps(d))

    view = Stage2RepairController(run, "vanishing").read()
    idx = {u["unit_id"]: u["solver_index"] for u in view["sampling_units"]}
    assert len(set(idx.values())) == len(idx), f"solver 索引串号：{idx}"
    assert idx["rw:def456:0"] == 10_100
    # 反查表：最终门报的是 solver 索引，调度身份是 unit_id
    assert view["solver_index_to_unit"][10_100] == "rw:def456:0"
