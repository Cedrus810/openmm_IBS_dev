"""`abfe_preoptimizer.Stage2RepairController` 的决策优先级契约。

这个类是"统一控制器"的第一版：**只读盘 + 纯判断，不执行、不落盘。**
本文件钉的是**优先级**和**三条不许违反的语义**，不是渲染格式：

  1. 缺窗口优先于一切 —— 缺窗口的总和不是完整 ΔG，比任何单窗质量都严重。
  2. `UNMEASURED`（没测出来，INSUFFICIENT_DATA）永远不当 `FAIL`
     （STATISTICALLY_REJECTED）用：前者加预算，后者换 Epoch。混起来就退化成
     "再测一次直到碰巧通过"。
  3. `allow_untrusted_stage_results` **只能改 trust_level**，不得把
     evidence_status 改写成 CONVERGED —— 那是调用方的发布策略，不是科学证据。
"""

import json
import os

import pytest

pytestmark = pytest.mark.cpu_only

from abfe_preoptimizer import Stage2RepairController


def _mkrun(tmp_path, *, windows, ranges, n_states, config=None, stage_result=None,
           stage_name="vanishing", stage_type="vdw"):
    """造一个最小的 run 目录：路径版本链 + 每窗口的 convergence/ibs_state。"""
    run = tmp_path / "run"
    ck = run / "checkpoints"
    sd = run / stage_name
    (ck / "path_versions").mkdir(parents=True)
    sd.mkdir(parents=True)
    (run / "run_provenance.json").write_text(json.dumps({
        "config": config or {"stage2_window_min_states": 4,
                             "stage2_window_max_states": 8,
                             "max_path_insertions": 3}
    }))
    (ck / "path_versions" / "v1.json").write_text(json.dumps({
        "version": 1, "kind": "initial",
        "states": [{"id": f"s{i}"} for i in range(n_states)],
        "window_ranges": [list(r) for r in ranges],
    }))
    (ck / "path_current.json").write_text(json.dumps({"version": 1}))
    for idx, w in windows.items():
        lam = w.get("lambdas_vdw", [1.0 - 0.1 * i for i in range(w.get("K", 4))])
        (sd / f"dual_window_{idx}_{stage_type}_convergence.json").write_text(json.dumps({
            "window_idx": idx, "lambdas_vdw": lam,
            "cumulative_production_steps": w.get("prod", 250000),
            "n_steps_per_window_effective": w.get("prod_target", 250000),
            "production_segments": [{}] * w.get("segments", 1),
            "window_data": {"n_frames": w.get("frames", 500)},
            "bias_warmup": {"warmup_budget_ledger": {
                "learning_steps": w.get("warmup", 50000),
                "cumulative_cap_steps": w.get("cap", 555000),
            }, "bias_update_count": 12},
        }))
        (ck / f"ibs_state_{stage_type}_window_{idx}.json").write_text(json.dumps({
            "bias_status": w.get("bias_status", "converged"),
            "f_k_evidence_status": w.get("evidence", "verified"),
            "frozen_validation_cumulative_steps": w.get("valid_steps", 0),
            "lambdas_vdw": lam,
        }))
    if stage_result is not None:
        (ck / "stage2_vanishing.json").write_text(json.dumps(stage_result))
    return str(run)


def _plan(tmp_path, **kw):
    run = _mkrun(tmp_path, **kw)
    c = Stage2RepairController(run, kw.get("stage_name", "vanishing"))
    return c, c.decide()


FULL = {i: {"K": 4} for i in range(4)}
R4 = [(0, 4), (3, 7), (6, 10), (9, 13)]


def test_reads_lo_hi_from_the_run_itself():
    """lo/hi 手传错的后果很实在（可拆区间 4/5 是 7..9、4/8 是 7..15）——
    run 自己记了跑的是什么，默认就用那个。"""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        run = _mkrun(pathlib.Path(td), windows=FULL, ranges=R4, n_states=13)
        c = Stage2RepairController(run)
        assert (c.lo, c.hi, c.max_path_insertions) == (4, 8, 3)
        assert c.config_source == "run_provenance.json"
        # 显式传参优先
        c2 = Stage2RepairController(run, min_states_per_window=4, max_states_per_window=5)
        assert (c2.lo, c2.hi) == (4, 5)


def test_missing_window_beats_everything(tmp_path):
    """布局 4 窗、产物只有 3 个 ⟹ 先补那个窗口，别先谈质量。"""
    c, plan = _plan(tmp_path, windows={0: {}, 1: {}, 2: {}}, ranges=R4, n_states=13)
    assert plan["action"] == "RUN_PRODUCTION"
    assert plan["windows"] == [3]
    assert "不是 ΔG" in plan["reason"] and "另一个量" in plan["reason"]
    # 缺窗现在是 INSUFFICIENT_DATA（老板定案：不再用含糊的 INCONCLUSIVE）
    assert plan["evidence_status"] == "INSUFFICIENT_DATA"


def test_refuted_fk_is_terminal_not_another_validation_round(tmp_path):
    """有统计功效的否决 ⟹ 换 Epoch 重标定，**不许**再加验证预算。"""
    w = dict(FULL)
    w[2] = {"K": 4, "bias_status": "calibrated_validation_failed", "evidence": "refuted"}
    c, plan = _plan(tmp_path, windows=w, ranges=R4, n_states=13)
    assert plan["action"] == "RECALIBRATE_FK"
    assert plan["exit"] == "HALT_FK_REFUTED"
    assert plan["windows"] == [2]
    assert plan["evidence_status"] == "REJECTED"
    # 现状必须被说出来：这条路径在代码里没人 catch
    assert "没有任何 except 捕获" in plan["reason"]


def test_insufficient_data_with_budget_continues_warmup(tmp_path):
    """没测出来 + 预算有余 ⟹ 加同类预算，不换轴。"""
    w = dict(FULL)
    w[1] = {"K": 4, "bias_status": "frozen_validation_indeterminate",
            "evidence": "indeterminate", "warmup": 100000, "cap": 555000}
    c, plan = _plan(tmp_path, windows=w, ranges=R4, n_states=13)
    assert plan["action"] == "CONTINUE_WARMUP"
    assert plan["windows"] == [1]
    assert "不换轴" in plan["reason"]


def test_insufficient_data_without_budget_halts_on_attribution(tmp_path):
    """预算耗尽且 stage 证据读不到 ⟹ 归因不出来是**合法结局**，不许编一个动作。"""
    w = dict(FULL)
    w[1] = {"K": 4, "bias_status": "frozen_validation_indeterminate",
            "evidence": "indeterminate", "warmup": 555000, "cap": 555000}
    c, plan = _plan(tmp_path, windows=w, ranges=R4, n_states=13)
    assert plan["exit"] == "HALT_NO_ATTRIBUTION"
    assert plan["missing_evidence"], "必须写明缺哪个证据"
    # 2026-09-11：出口按"终止 vs 路由"重分类。`HALT_NO_ATTRIBUTION` 在**还有预算
    # 或还有可行动作**时是**路由信号**（交外层控制器换动作继续），流水线没停 ⟹
    # execution_status 仍是 IN_PROGRESS。只有 GLOBAL_BUDGET_EXHAUSTED /
    # NO_FEASIBLE_ACTION / 输入无效这三种才是真终止。
    assert plan["execution_status"] == "IN_PROGRESS"
    assert plan.get("terminal") is False and plan.get("routing") is True


def test_short_production_runs_production(tmp_path):
    w = dict(FULL)
    w[3] = {"K": 4, "prod": 100000, "prod_target": 250000}
    c, plan = _plan(tmp_path, windows=w, ranges=R4, n_states=13)
    assert plan["action"] == "RUN_PRODUCTION"
    assert plan["windows"] == [3]


def test_all_done_but_no_stage_result_asks_for_analysis(tmp_path):
    """窗口都跑完、stage 分析没跑 ⟹ ANALYZE，不是 RUN_PRODUCTION。"""
    c, plan = _plan(tmp_path, windows=FULL, ranges=R4, n_states=13)
    assert plan["action"] == "ANALYZE"
    assert plan["missing_evidence"] == ["stage2_*.json"]


def test_converged_stage_is_done_but_not_claimed_correct(tmp_path):
    c, plan = _plan(tmp_path, windows=FULL, ranges=R4, n_states=13,
                    stage_result={"converged": True, "total_delta_G": -12.3,
                                  "path_is_complete": True})
    assert plan["action"] == "DONE"
    assert plan["evidence_status"] == "CONVERGED"
    # 措辞不得暗示正确性
    assert "不等于答案正确" in plan["reason"]


def test_untrusted_switch_only_moves_trust_level(tmp_path):
    """`allow_untrusted` 是发布策略，不得把科学证据改写成 CONVERGED。"""
    run = _mkrun(tmp_path, windows={0: {}, 1: {}, 2: {}}, ranges=R4, n_states=13)
    strict = Stage2RepairController(run).decide()
    loose = Stage2RepairController(run, allow_untrusted_stage_results=True).decide()
    assert strict["trust_level"] == "STATISTICAL_ONLY"
    assert loose["trust_level"] == "OVERRIDDEN_UNTRUSTED"
    # 老板定案：缺窗/救援耗尽/支撑不足一律是 INSUFFICIENT_DATA，不再是含糊的 INCONCLUSIVE
    assert strict["evidence_status"] == loose["evidence_status"] == "INSUFFICIENT_DATA"


def test_skipped_windows_are_surfaced_as_rescue_targets(tmp_path):
    """被踢出协方差链的窗口必须能被控制器看见 —— 这条路径以前不可达（P0-2b）。"""
    c, plan = _plan(tmp_path, windows=FULL, ranges=R4, n_states=13,
                    stage_result={"converged": False, "path_is_complete": True,
                                  "skipped_windows": [{"window_index": 0,
                                                       "reason": "insufficient_frames"}]})
    assert plan["action"] == "RUN_PRODUCTION"
    assert plan["windows"] == [0]
    assert "去相关后有效帧数不足" in plan["reason"]


def test_controller_never_writes_anything(tmp_path):
    """只读。控制器一旦落盘，"拿历史 run 离线重放决策"就不成立了。"""
    run = _mkrun(tmp_path, windows=FULL, ranges=R4, n_states=13)
    before = {p: os.stat(os.path.join(dp, p)).st_mtime_ns
              for dp, _, fs in os.walk(run) for p in fs}
    c = Stage2RepairController(run)
    c.render(); c.decide(); c.feasible()
    after = {p: os.stat(os.path.join(dp, p)).st_mtime_ns
             for dp, _, fs in os.walk(run) for p in fs}
    assert before == after
