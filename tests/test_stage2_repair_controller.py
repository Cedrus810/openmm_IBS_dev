"""`abfe_preoptimizer.Stage2RepairController` 的决策优先级契约。

这个类是"统一控制器"的第一版：**只读盘 + 纯判断，不执行、不落盘。**
本文件钉的是**优先级**和**三条不许违反的语义**，不是渲染格式：

  1. **按因果顺序路由「最早的未解决窗口」**，不是按错误类型设全局优先级。
     缺窗口**仍然**是硬约束（缺窗口的和不是 ΔG，是另一个量），但它的正确位置在
     因果顺序**之后** —— 只有完整前缀全部合格，才轮到去跑缺的那个。
     [2026-09-13 改写] 旧版这条写的是"缺窗口优先于一切"，那正是被实测推翻的：
     缺窗分支排全局最高时，rep1 里 win3 支撑不足、win4 卡在 local cap，目标却被
     判成 **win5**，把上游两个未解决窗口整个盖住。
     ⚠️ 三态不是两态：`ELIGIBLE` / `PROBLEM` / `UNKNOWN`。**缺证据 ≠ 有问题** ——
     对"不知道"的正确动作是产出证据（ANALYZE），不是重标定。
  2. `UNMEASURED`（没测出来，INSUFFICIENT_DATA）永远不当 `FAIL`
     （STATISTICALLY_REJECTED）用：前者加预算，后者换 Epoch。混起来就退化成
     "再测一次直到碰巧通过"。
  3. `allow_untrusted_stage_results` **只能改 trust_level**，不得把
     evidence_status 改写成 CONVERGED —— 那是调用方的发布策略，不是科学证据。
"""

import json
import os
import pathlib

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
        # 🔑 [2026-09-13] **逐窗自检产物（P2-9c）必须造**，否则整个 fixture 测不到东西。
        # `decide()` 是三态的：没有这份产物 ⟹ `self_verdict is None` ⟹ 窗口判
        # `UNKNOWN` ⟹ 「产出证据」那条分支（ANALYZE）会在下面每一个分支**之前**
        # 把请求吞掉，于是缺窗/短生产/DONE/skipped 四条分支一条都走不到，
        # 六个测试会退化成同一个"没有自检产物就 ANALYZE"。
        # 默认给 ANALYSIS_ELIGIBLE（= 这个窗口自己没问题），要测别的态就传
        # `self_verdict=...`，语义见 docs/STAGE2_CONTROLLER_DESIGN_2026-09-12.md §3/§4。
        # ⚠️ **还在预热/验证的窗口不落这份产物** —— 它是生产跑完那一刻才写的
        # （`ibs_engine.window_self_support_check`）。给预热中的窗口伪造一份
        # `ANALYSIS_ELIGIBLE` 会让 `_window_state` 的 `self_verdict` 判据盖过
        # `phase in (WARMUP_LEARN, WARMUP_VALIDATE)`，于是预热类测试全部失效。
        _sv = w.get(
            "self_verdict",
            "ANALYSIS_ELIGIBLE" if w.get("bias_status", "converged") == "converged" else None,
        )
        if _sv is not None:
            (sd / f"dual_window_{idx}_{stage_type}_self_support.json").write_text(
                json.dumps({
                    "window_idx": idx,
                    "verdict": _sv,
                    "verdict_source": w.get("self_verdict_source", "min_n_eff_over_g"),
                    "sufficient": _sv == "ANALYSIS_ELIGIBLE",
                    "n_frames_decorrelated": w.get("n_decorr", 120),
                    "min_frames_per_window": 10,
                    "min_n_eff_over_g": w.get("min_n_eff_over_g", 25.0),
                })
            )
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
    # ⚠️ [审计 #11，2026-09-14 改写] 旧断言钉的是理由文本里那句「这条路径在代码里
    # 没人 catch、会直接炸穿整个 run」——**那句现在是假的**：
    # `abfe_pipeline._run_stage2_autonomous` 已有
    # `except _ie_exc.IBSFrozenCalibrationValidationError` 分支（封存候选 →
    # 落 `stage2_fk_refuted.json` → `continue` 回顶层重判）。
    # 钉一句已经过时的现状描述，会让下一个人以为这条路是断的。
    # 改钉真正的不变量：`HALT_FK_REFUTED` 是**路由**不是终态 —— 加进
    # `TERMINAL_EXITS` 会让主循环在分发前 break，那次重标定根本不执行。
    assert plan["terminal"] is False and plan["routing"] is True
    assert "HALT_FK_REFUTED" not in Stage2RepairController.TERMINAL_EXITS


def test_insufficient_data_with_budget_continues_warmup(tmp_path):
    """没测出来 + 预算有余 ⟹ 加同类预算，不换轴。"""
    w = dict(FULL)
    w[1] = {"K": 4, "bias_status": "frozen_validation_indeterminate",
            "evidence": "indeterminate", "warmup": 100000, "cap": 555000}
    c, plan = _plan(tmp_path, windows=w, ranges=R4, n_states=13)
    assert plan["action"] == "CONTINUE_WARMUP"
    assert plan["windows"] == [1]
    assert "不换轴" in plan["reason"]


def test_no_budget_is_consumed_before_action_selection(tmp_path):
    """预算耗尽的窗口 ⟹ **先判预算、再选动作**，且这是路由不是终止。

    [2026-09-13 改写] 旧断言要的是 `HALT_NO_ATTRIBUTION`（"归因不出来"）。
    现在归因是**成功的** —— 控制器明确说得出为什么不动它：付不起新 Epoch 的最低
    验证额度。decide() 把「预算可行性」提到了选动作**之前**（design §决策顺序），
    理由是实测：`run2/vanishing_2` 在 555k/555k 零余量时仍被判 RECALIBRATE_FK，
    开出来的新 f_k **永远验不了**（win2 连死三次就是这个形状）。
    所以出口是 `HALT_BUDGET`，不是归因失败。

    不变的语义（这才是本测试真正钉的）：**局部没预算 ≠ 终止**。只有
    GLOBAL_BUDGET_EXHAUSTED / NO_FEASIBLE_ACTION / 输入无效三种才是真终态。

    ⚠️ [2026-09-14] fixture 从「零余量」改成「**有余量但不够开新 Epoch**」
    （剩 35000 < 首档验证 50000）。零余量是**另一种**情形：那时窗口连重新进入
    采样都会被预热门弹回（真机 win0 连发 40 轮），补生产帧这条退路本身不成立，
    由 1e 分支单独处理。本分支要钉的是「付不起**新 Epoch**、但窗口还进得去」。

    ⚠️⚠️ [审计 #20，2026-09-14 改写] **断言从「发 HALT_BUDGET」改成「不终止、
    也不在未验证的 f_k 上开生产」。**

    旧断言钉的是那道**无条件**预算预检：`decide()` 一算出 `earliest` 就不分动作
    类型地判「这个窗口付不付得起换 Epoch」，付不起就 return `RUN_PRODUCTION` +
    `HALT_BUDGET`。审计 #20 判定这个行为本身是错的，两条理由：
      · 它把后面所有分支（1c/1d/3/3a/4/5/5a/5b/6）对该窗口静默删掉；
      · `left` 介于 0 和首档之间时，它会在一份**尚未冻结/验证**的 f_k 上直接开
        生产 —— 本 fixture 正是这个盘面（35000 < 50000，窗口仍在
        `frozen_validation_indeterminate`）。
    现在这道闸挪进 `plan()`、只对换 Epoch 类动作跑，于是这个窗口走分支 3
    （仍在预热、预热预算有余 35000 > 0）⟹ `CONTINUE_WARMUP`，那才是对症动作。

    本测试真正钉的语义**一个字没变**：局部没预算 ≠ 终止。
    """
    w = dict(FULL)
    w[1] = {"K": 4, "bias_status": "frozen_validation_indeterminate",
            "evidence": "indeterminate", "warmup": 520000, "cap": 555000}
    c, plan = _plan(tmp_path, windows=w, ranges=R4, n_states=13)
    assert plan["windows"] == [1]
    # 付不起**新 Epoch** ⟹ 绝不发换 Epoch 类动作（那会得到一份永远验不了的 f_k）
    assert plan["action"] not in (
        "RECALIBRATE_FK", "RELEARN_FK_EPOCH", "PROBE_REANCHOR_EPOCH")
    # 也不得在一份尚未冻结/验证的 f_k 上开生产 —— 该窗口还有预热预算，先把预热跑完
    assert plan["action"] == "CONTINUE_WARMUP", plan["reason"]
    # 低支撑/没预算永远是「尚不可测」，不是 FAIL
    assert plan["evidence_status"] == "INSUFFICIENT_DATA"
    # 路由信号：流水线没停，外层换个动作继续
    assert plan["execution_status"] == "IN_PROGRESS"
    assert plan.get("terminal") is False
    # 单窗耗尽不等于全局耗尽 —— 全局耗尽才是真终态
    assert plan["exit"] not in ("GLOBAL_BUDGET_EXHAUSTED",)


def test_short_production_runs_production(tmp_path):
    """生产步数没到目标 ⟹ 接着原段跑，不改结构。

    ⚠️ [2026-09-13] **光是步数没到目标不足以让控制器动它。** 决策按「最早的未解决
    窗口」路由（design §4），而「未解决」的判据是**主验收量** `min N_eff/g`
    （design §3），不是步数计数器。一个只跑了 100k 步但支撑已经够的窗口是
    `ANALYSIS_ELIGIBLE`，控制器**正确地**不去碰它。所以这里要把 win3 的自检
    verdict 一起设成不合格 —— 那才是「生产短了 ⟹ 补采」真实发生的形状。
    """
    w = dict(FULL)
    w[3] = {"K": 4, "prod": 100000, "prod_target": 250000,
            "self_verdict": "INSUFFICIENT_DATA", "min_n_eff_over_g": 4.2}
    c, plan = _plan(tmp_path, windows=w, ranges=R4, n_states=13)
    assert plan["action"] == "RUN_PRODUCTION"
    assert plan["windows"] == [3]
    # 归因必须落在「步数没到目标」这条，不是笼统的补采
    assert "生产步数未达目标" in plan["reason"]
    # win3 是末窗 ⟹ 没有下游可挡，`blocked_by` 只在真有被挡的窗口时才填
    assert plan["blocked_by"] is None and plan["blocked_by_upstream"] == []


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
    w = dict(FULL)
    # 被踢出协方差链的窗口，它自己的生产后自检**必然**也判不合格 —— 同一个量、
    # 同一个门槛，只是自检提前到窗口刚跑完那一刻（design §3 / decide 分支 5b）。
    w[0] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA", "min_n_eff_over_g": 3.1,
            "n_decorr": 7}
    c, plan = _plan(tmp_path, windows=w, ranges=R4, n_states=13,
                    stage_result={"converged": False, "path_is_complete": True,
                                  "skipped_windows": [{"window_index": 0,
                                                       "reason": "insufficient_frames"}],
                                  # stage 分析既然跑过（`skipped_windows` 就是它产出的），
                                  # 生产侧累计 f_k 残差也必然在。给 PASS = **f_k 没问题、
                                  # 缺的就是帧** —— 否则控制器会（正确地）先要这份证据，
                                  # 根本轮不到补采（decide 分支 5a）。
                                  "cumulative_fk_residual_production": [
                                      {"window_index": 0, "verdict": "PASS",
                                       "cumulative_residual_span_kJ_mol": 1.2}]})
    assert plan["action"] == "RUN_PRODUCTION"
    assert plan["windows"] == [0]
    # 语义：INSUFFICIENT_DATA ≠ FAIL，动作是补采、不作废已有帧
    assert "去相关帧数不足" in plan["reason"]
    assert "INSUFFICIENT_DATA ≠ FAIL" in plan["reason"]
    # 未解决的是 win0 ⟹ 下游 1/2/3 全部被挡住，不许"跳过前面去跑后面"
    assert plan["blocked_by"] == 0 and plan["blocked_by_upstream"] == [1, 2, 3]


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


# ---------------------------------------------------------------------------
# stage 判据没过时**绝不能**判 DONE（2026-09-13 回归）
# ---------------------------------------------------------------------------
# 曾经的分支 8 是 `"DONE" if has_stage_result else "ANALYZE"` —— 只要结果文件
# 存在就 DONE，哪怕 stage 明写 `converged=False`。这与设计 §2 直接冲突：
# `DONE` 的定义是「**全路径证据齐备且达标**」。
# 后果不是措辞别扭：主循环只看 `plan["terminal"]` 就 break
# （`abfe_pipeline._run_stage2_autonomous`），**不看** `evidence_status`，
# 于是判据没过时自治诊断会提前结束。
# （最终结果没被错误发布 —— 下游另有默认拒绝 converged=False 的门 —— 但循环白停了。）

_GATE_RECS = [{"window_index": i, "min_ess_ratio": 0.30 - 0.05 * i,
               "absolute_ess": 40.0 - 5 * i,
               "n_frames_decorrelated": 200 - 30 * i} for i in range(4)]
_STAGE_BASE = {"converged": False, "path_is_complete": True,
               "window_overlap_diagnostics": _GATE_RECS,
               "input_window_indices": [0, 1, 2, 3],
               "solved_window_indices": [0, 1, 2, 3]}


@pytest.mark.parametrize("label,extra", [
    # 归因不到任何一道门（最朴素的 converged=False）
    ("bare", {}),
    # 求解器把窗口丢了 —— 上游有专门分支，走到这里说明没抓住 ⟹ 不许瞎补采
    ("windows_dropped", {"solved_window_indices": [0, 1, 2]}),
    # 支撑/偏斜类：加帧治不了，对症是缩跨度
    ("overlap", {"min_overlap": 0.02, "min_overlap_threshold": 0.10}),
    ("target_support", {"target_support_gate": {
        "passed": False, "failed_checks": ["raw_min_absolute_ess"],
        "raw_min_absolute_ess_threshold": 20.0,
        "max_top1pct_raw_weight_threshold": 0.3}}),
    # 样本量/精度类：「尚不可测」，对症是加采样
    ("decorrelated", {"min_decorrelated_samples": 8,
                      "min_decorrelated_samples_threshold": 20}),
    ("endpoint_sigma", {"max_endpoint_uncertainty_kJ_mol": 4.1,
                        "max_endpoint_uncertainty_kJ_mol_threshold": 2.0}),
])
def test_stage_not_converged_is_never_done(tmp_path, label, extra):
    """全窗合格 + stage 结果存在 + `converged=False` ⟹ **不得返回 DONE**。"""
    c, plan = _plan(tmp_path, windows=FULL, ranges=R4, n_states=13,
                    stage_result={**_STAGE_BASE, **extra})
    assert plan["action"] != "DONE", "stage 判据没过就不能记为 DONE（设计 §2）"
    assert plan["exit"] != "DONE"
    # 三维状态：窗口采样确实执行完了，但那不等于「达标」
    assert plan["evidence_status"] != "CONVERGED"
    # 终止的话只许是 NO_FEASIBLE_ACTION，且必须写清它不是 DONE
    if plan["terminal"]:
        assert plan["exit"] == "NO_FEASIBLE_ACTION"
        assert plan["action"] == "NO_ACTION"
        assert plan["execution_status"] == "HALTED"
        assert "不是 DONE" in plan["reason"]


def test_stage_gate_failure_routes_by_the_gate_that_actually_failed(tmp_path):
    """归因**按真实失败的那道门**分岔：支撑/偏斜 → 缩跨度；样本量/精度 → 加采样。

    分类不是这里定的，出处是设计文档 §3（低支撑是「尚不可测」）与 §5.1
    （加帧治不了偏斜：实测 250k→1M 让 top1% 从 0.545 涨到 0.762）。
    """
    def act(extra):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as td:
            c, p = _plan(pathlib.Path(td), windows=FULL, ranges=R4, n_states=13,
                         stage_result={**_STAGE_BASE, **extra})
            return p

    # 支撑/偏斜类 ⟹ 布局动作，且落在最差的那个窗口上（win3 的 ess 最低）
    p = act({"min_overlap": 0.02, "min_overlap_threshold": 0.10})
    assert p["action"] in ("SPLIT_TAIL_WINDOW", "INSERT_LAMBDA")
    assert "加帧治不了" in p["reason"]
    if p["action"] == "INSERT_LAMBDA":
        assert p["windows"] == [3]

    # 样本量类 ⟹ 补采，落在去相关帧数最少的窗口
    p = act({"min_decorrelated_samples": 8, "min_decorrelated_samples_threshold": 20})
    assert p["action"] == "RUN_PRODUCTION" and p["windows"] == [3]
    assert "尚不可测" in p["reason"]

    # 端点 σ 是路径级量 ⟹ 没有逐窗归因，但**必须把窗口列全**：
    # 空列表会让执行器按原目标重跑（overrides 为空）⟹ resume 下空转。
    p = act({"max_endpoint_uncertainty_kJ_mol": 4.1,
             "max_endpoint_uncertainty_kJ_mol_threshold": 2.0})
    assert p["action"] == "RUN_PRODUCTION" and p["windows"] == [0, 1, 2, 3]


def test_untrusted_override_releases_but_never_claims_converged(tmp_path):
    """`allow_untrusted` 是**发布策略**：可以放行，但不得把证据改写成 CONVERGED。"""
    sr = {**_STAGE_BASE, "min_overlap": 0.02, "min_overlap_threshold": 0.10}
    run = _mkrun(tmp_path, windows=FULL, ranges=R4, n_states=13, stage_result=sr)
    loose = Stage2RepairController(
        run, "vanishing", allow_untrusted_stage_results=True).decide()
    assert loose["exit"] == "DONE_UNTRUSTED" and loose["terminal"] is True
    assert loose["trust_level"] == "OVERRIDDEN_UNTRUSTED"
    # 关键：证据维度没被放行改写
    assert loose["evidence_status"] != "CONVERGED"


# ---------------------------------------------------------------------------
# stage 级质量门归因（`stage_quality_gate_failures`，STAGE2_CONTROLLER_PROTOCOL_VERSION=2）
# ---------------------------------------------------------------------------
# 这个纯函数是分支 9 的全部依据：它读不出门 ⟹ 自治循环直接 NO_FEASIBLE_ACTION。
# 落地时它一条测试都没有，而键名全是从 `ibs_engine` 的落盘 payload 里抄的。

def _stage_with_support_failure(check="raw_absolute_ess_below_threshold"):
    """target_support 门失败；**mixture 排序与 raw 排序故意相反**。"""
    return {
        "converged": False,
        "target_support_gate": {
            "passed": False,
            "failed_checks": ["ibs_segment_target_support"],
            "raw_min_absolute_ess_threshold": 20.0,
            "ibs_segment_gate": {
                "failed_checks": [check],
                "max_top1pct_raw_weight_threshold": 0.10,
            },
        },
        "window_overlap_diagnostics": [
            # win0：mixture 看着最差，raw 却最好 —— 挑错尺子就会选中它。
            {"window_index": 0, "absolute_ess": 3.0,
             "raw_min_absolute_ess": 90.0, "top1pct_raw_weight": 0.02},
            # win1：raw 支撑最低 —— raw ESS 那一支判否时的元凶。
            {"window_index": 1, "absolute_ess": 300.0,
             "raw_min_absolute_ess": 1.1, "top1pct_raw_weight": 0.05},
            # win2：权重最集中 —— top1% 那一支判否时的元凶。
            {"window_index": 2, "absolute_ess": 500.0,
             "raw_min_absolute_ess": 40.0, "top1pct_raw_weight": 0.83},
        ],
    }


def test_support_gate_attribution_uses_the_gauge_the_gate_actually_judges():
    from abfe_preoptimizer import stage_quality_gate_failures, STAGE_GATE_NARROW_SPAN

    (fail,) = [f for f in stage_quality_gate_failures(_stage_with_support_failure())
               if f["gate"] == "target_support_gate"]
    assert fail["report_category"] == STAGE_GATE_NARROW_SPAN   # 仅报告的分类
    assert fail["worst_window"] == 1, (
        "最差窗口必须按门自己判的 raw 量挑；按 `absolute_ess`（去相关后的 mixture "
        "覆盖度、纯诊断）挑会选中 win0 —— 那是另一把尺子"
    )
    # 合并后的 stage 级门不带 top1% 阈值，得从嵌套的 IBS 段门里取回来。
    assert fail["threshold"]["max_top1pct_raw_weight"] == 0.10


def test_support_gate_attribution_switches_gauge_when_top1pct_is_the_failing_check():
    """两个 raw 量是正交的两份证据：判否的是哪一个，最差窗口就按哪一个挑。"""
    from abfe_preoptimizer import stage_quality_gate_failures

    (fail,) = [f for f in stage_quality_gate_failures(
        _stage_with_support_failure("top1pct_raw_weight_above_threshold"))
        if f["gate"] == "target_support_gate"]
    assert fail["worst_window"] == 2


def test_each_gate_maps_to_its_own_report_category():
    from abfe_preoptimizer import (
        stage_quality_gate_failures,
        STAGE_GATE_MORE_SAMPLING, STAGE_GATE_NARROW_SPAN,
    )

    got = {f["gate"]: f["report_category"] for f in stage_quality_gate_failures({
        "converged": False,
        "min_overlap": 0.01, "min_overlap_threshold": 0.05,
        "min_decorrelated_samples": 4, "min_decorrelated_samples_threshold": 20,
        "max_endpoint_uncertainty_kJ_mol": 9.9,
        "max_endpoint_uncertainty_kJ_mol_threshold": 2.0,
        "window_overlap_diagnostics": [
            {"window_index": 5, "min_ess_ratio": 0.01, "n_frames_decorrelated": 4},
        ],
    })}
    assert got == {
        "min_overlap": STAGE_GATE_NARROW_SPAN,
        "min_decorrelated_samples": STAGE_GATE_MORE_SAMPLING,
        "max_endpoint_uncertainty_kJ_mol": STAGE_GATE_MORE_SAMPLING,
    }


def test_missing_readings_are_never_reported_as_passing():
    """「缺证据 ≠ 通过」：阈值在、实测值缺 ⟹ 记一条 unattributed 失败。"""
    from abfe_preoptimizer import stage_quality_gate_failures, STAGE_GATE_UNATTRIBUTED

    (fail,) = stage_quality_gate_failures(
        {"converged": False, "min_overlap": None, "min_overlap_threshold": 0.05})
    assert fail["gate"] == "min_overlap" and fail["report_category"] == STAGE_GATE_UNATTRIBUTED


def test_a_window_the_solver_cannot_even_use_gets_frames_not_another_analyze(tmp_path):
    """去相关帧数低于求解器下限的窗口 ⟹ **补采**，不是再跑一次 ANALYZE。

    真机 2026-09-14：win0 去相关后 9 帧（下限 10）被求解器跳过 ⟹ 分析根本走不到
    这个窗口 ⟹ `cumulative_fk_residual_production` 永远不会出现 ⟹ 5a-1 那条
    「先把 f_k 偏差证据算出来，零额外采样」无限返回 ANALYZE：连发 4 次、盘上一个
    字节没变，靠停滞保护降级到探针（判「f_k 不是瓶颈」什么也没做），
    最后 NO_FEASIBLE_ACTION 退出 —— 而它真正需要的补采分支就在下面几行。

    「零额外采样就能算」有前提：这批帧求解器得能用。
    """
    _, plan = _plan(
        tmp_path,
        windows={
            0: {"K": 4, "self_verdict": "INSUFFICIENT_DATA", "n_decorr": 9},
            1: {"K": 4}, 2: {"K": 4}, 3: {"K": 4},
        },
        ranges=R4, n_states=13,
    )
    assert plan["action"] == "RUN_PRODUCTION", plan["reason"]
    assert plan["windows"] == [0]
    # 语义仍然是「尚不可测」，不是 FAIL
    assert plan["evidence_status"] == "INSUFFICIENT_DATA"
    assert not plan["terminal"]


def test_mixed_gate_failures_do_not_drop_the_sampling_action():
    """支撑类 + 样本量类**同时**失败、且布局动作不可行 ⟹ 必须去补帧，不是终止。

    真机 2026-09-14：`rescue_window_1` 支撑/偏斜（NARROW_SPAN）、
    `rescue_window_2` 11/20 帧（MORE_SAMPLING）。`_remedies` 是集合，9b 用 `in`
    判、9c 原来用 `==` 判 ⟹ 只要有任何一条支撑类失败，9c 永远不可达，
    11/20 那个窗口的帧**永远补不上**。
    """
    from abfe_preoptimizer import (
        stage_quality_gate_failures,
        STAGE_GATE_MORE_SAMPLING, STAGE_GATE_NARROW_SPAN,
    )

    fails = stage_quality_gate_failures({
        "converged": False,
        # 支撑类：raw ESS 不够
        "target_support_gate": {
            "passed": False, "failed_checks": ["ibs_segment_target_support"],
            "raw_min_absolute_ess_threshold": 20.0,
            "ibs_segment_gate": {"failed_checks": ["raw_absolute_ess_below_threshold"]},
        },
        # 样本量类：去相关帧数不够
        "min_decorrelated_samples": 11, "min_decorrelated_samples_threshold": 20,
        "window_overlap_diagnostics": [
            {"window_index": 1, "raw_min_absolute_ess": 1.1, "n_frames_decorrelated": 60},
            {"window_index": 2, "raw_min_absolute_ess": 90.0, "n_frames_decorrelated": 11},
        ],
    })
    categories = {f["report_category"] for f in fails}
    assert categories == {STAGE_GATE_NARROW_SPAN, STAGE_GATE_MORE_SAMPLING}

    # 两类各自归因到**不同**的窗口，别混着挑
    span = next(f for f in fails if f["report_category"] == STAGE_GATE_NARROW_SPAN)
    samp = next(f for f in fails if f["report_category"] == STAGE_GATE_MORE_SAMPLING)
    assert span["worst_window"] == 1
    assert samp["worst_window"] == 2


def test_epoch_budget_never_terminates_a_production_topup(tmp_path):
    """预热账本不许终止一个只花生产帧的动作。

    真机形状：`_epoch_validation_unaffordable` 判「付不起新 Epoch 的验证额度」
    ⟹ 退而求其次发 `RUN_PRODUCTION`（用**已冻结**的 f_k 补帧，一步验证预算都不花），
    但出口挂的是 `GLOBAL_BUDGET_EXHAUSTED` —— 它在 `TERMINAL_EXITS` 里，主循环在
    分发**之前**就 break，补帧根本不执行。拿 A 账本的余额终止只花 B 账本的动作。
    """
    from abfe_preoptimizer import Stage2RepairController as C

    # 每个窗口预热预算都是 0 ⟹ all_windows_budget_exhausted 必然为真
    _, plan = _plan(
        tmp_path,
        windows={
            0: {"K": 4, "self_verdict": "INSUFFICIENT_DATA",
                "warmup": 555000, "cap": 555000},
            1: {"K": 4}, 2: {"K": 4}, 3: {"K": 4},
        },
        ranges=R4, n_states=13,
    )
    if plan["action"] == "RUN_PRODUCTION":
        assert plan["exit"] not in C.TERMINAL_EXITS, (
            f"补生产帧的动作挂了终态出口 {plan['exit']} ⟹ 主循环会先 break，"
            "这个动作永远不执行"
        )
        assert not plan["terminal"]


def test_no_decide_branch_pairs_a_production_topup_with_a_terminal_budget_exit():
    """源码级：`decide()` 里不许再出现「RUN_PRODUCTION + GLOBAL_BUDGET_EXHAUSTED」。"""
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "abfe_preoptimizer.py").read_text("utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "decide")
    bad = []
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "plan"):
            continue
        action = (node.args[0].value
                  if node.args and isinstance(node.args[0], ast.Constant) else None)
        exits = {n.value for n in ast.walk(node)
                 if isinstance(n, ast.Constant) and n.value == "GLOBAL_BUDGET_EXHAUSTED"}
        if action == "RUN_PRODUCTION" and exits:
            bad.append(node.lineno)
    assert not bad, (
        f"这些行把补生产帧的动作配了全局预算终态：{bad}。"
        "预热预算耗尽不证明补帧不可行 —— 那个判断只有主循环的停滞保护做得出来。"
    )


def test_single_segment_window_still_gets_two_comparable_support_points(tmp_path):
    """S2-C：边际增长判据要两个点，而按**段**建史给不出。

    原来一个采样段只贡献一个点 ⟹ 单段窗口永远只有 1 个点 ⟹
    「同分布加帧已被证伪就别再加」这道唯一的刹车对单段窗口**从不触发**。
    补法是读主循环每轮已经在落的逐块快照，按 (path_version, segment) 保可比性。
    """
    import json as _json

    run = _mkrun(
        tmp_path,
        windows={
            0: {"K": 4, "self_verdict": "INSUFFICIENT_DATA",
                "prod": 500000, "min_n_eff_over_g": 3.8},
            1: {"K": 4}, 2: {"K": 4}, 3: {"K": 4},
        },
        ranges=R4, n_states=13,
    )
    # 主循环逐轮快照：同一布局、同一段、采样量更小的那两块支撑**更高**
    # ⟹ 加帧没带来增长。
    # ⚠️ [审计 #47，2026-09-14] 从**两块**加到**三块**：边际增益刹车现在要求
    # 至少 3 个点才判（两点分不出趋势与噪声，而 `min N_eff/g` 的早期点系统性偏高
    # —— 分子 Kish ESS 小 N 乐观、分母 ĝ 在 N≫τ 前还在涨，两个偏差同向）。
    # 本测试钉的东西没变：**单段窗口也要拿得到多个可比点**，刹车对它不能是死的。
    (pathlib.Path(run) / "checkpoints" / "stage2_autonomous_history.json").write_text(
        _json.dumps({"iterations": [
            {"iteration": 1, "action": "RUN_PRODUCTION", "path_version": 1,
             "windows": [0],
             "snapshot": [{"window_idx": 0, "segment": "vanishing",
                           "production_steps": 250000, "min_n_eff_over_g": 4.5}]},
            {"iteration": 2, "action": "RUN_PRODUCTION", "path_version": 1,
             "windows": [0],
             "snapshot": [{"window_idx": 0, "segment": "vanishing",
                           "production_steps": 375000, "min_n_eff_over_g": 4.4}]},
        ]})
    )

    # 逐块历史只在**物理 stage 的聚合视图**里建（`read_aggregated`），
    # 主循环用的也正是它。
    c = Stage2RepairController.for_physical_stage(run, "vanishing", "vdw")
    hist = c.read().get("min_n_eff_over_g_history") or {}
    pts = [v for _lbl, v in (hist.get(0) or [])]
    assert len(pts) >= 2, f"单段窗口仍然只有 {len(pts)} 个点，边际增长判据用不上"
    assert pts[0] == 4.5 and pts[-1] == 3.8, f"点没有按采样量排序：{hist.get(0)}"

    assert c.decide()["action"] == "INSERT_LAMBDA", (
        "加帧已被该窗口自己的数据证伪 ⟹ 对症动作是缩跨度"
    )


def test_support_points_from_another_layout_are_not_comparable(tmp_path):
    """布局变了跨度就变了，前后不是同一个量 —— 别拿来判"加帧有没有用"。"""
    import json as _json

    run = _mkrun(
        tmp_path,
        windows={0: {"K": 4, "self_verdict": "INSUFFICIENT_DATA",
                     "prod": 500000, "min_n_eff_over_g": 3.8},
                 1: {"K": 4}, 2: {"K": 4}, 3: {"K": 4}},
        ranges=R4, n_states=13,
    )
    (pathlib.Path(run) / "checkpoints" / "stage2_autonomous_history.json").write_text(
        _json.dumps({"iterations": [{
            "iteration": 1, "action": "RUN_PRODUCTION", "path_version": 7,  # 另一条布局
            "snapshot": [{"window_idx": 0, "segment": "vanishing",
                          "production_steps": 250000, "min_n_eff_over_g": 4.5}],
        }]})
    )
    hist = Stage2RepairController.for_physical_stage(
        run, "vanishing", "vdw").read().get("min_n_eff_over_g_history") or {}
    assert len(hist.get(0) or []) == 1, "跨布局的点不该进边际增长判据"


# ---------------------------------------------------------------------------
# DECORR-01：求解器对「能否进入求解」有操作权威
# ---------------------------------------------------------------------------

def _run_with_solver_skip(tmp_path, *, self_verdict, skip_frames):
    """win1 自检 `ANALYSIS_ELIGIBLE`，但求解器把它踢出了协方差链。"""
    return _mkrun(
        tmp_path,
        windows={0: {"K": 4}, 1: {"K": 4, "self_verdict": self_verdict},
                 2: {"K": 4}, 3: {"K": 4}},
        ranges=R4, n_states=13,
        stage_result={
            "converged": False,
            "skipped_windows": [{
                "window_index": 1,
                "n_frames_after_decorrelation": skip_frames,
                "min_frames_per_window": 10,
                "statistical_inefficiency": 55.0,
                "reason": "insufficient_frames_after_decorrelation",
            }],
        },
    )


def test_solver_skip_beats_a_passing_self_check(tmp_path):
    """win3 实测形状：自检 56 帧 `sufficient=True`，求解器报 9/7 并跳窗。

    先前 `_window_state` 只看自检 ⟹ 判 ELIGIBLE ⟹ earliest 越过它去处理下游，
    而它的帧一次都没补上。求解器的裁决必须优先。
    """
    run = _run_with_solver_skip(tmp_path, self_verdict="ANALYSIS_ELIGIBLE",
                                skip_frames=7)
    c = Stage2RepairController.for_physical_stage(run, "vanishing", "vdw")
    view = c.read()
    w1 = next(w for w in view["windows"] if w["window_idx"] == 1)

    assert w1["self_verdict"] == "ANALYSIS_ELIGIBLE"      # 自检说没问题
    assert w1["solver_skip"], "求解器的跳窗记录没挂到窗口上"
    plan = c.decide()
    assert plan["windows"] == [1], (
        f"被求解器跳掉的窗口必须当成未解决（实得 {plan['windows']}）：{plan['reason']}"
    )
    assert plan["action"] != "DONE"


def test_the_two_decorrelation_numbers_are_kept_as_separate_evidence(tmp_path):
    """两侧分别命名、分别展示 —— **不让数字强行看齐**。

    去相关抽样依赖各自的输入帧集与所选 g；拿一侧的帧数反推另一侧是错的。
    `self_sufficient` 里还混着 N_eff/g 与 top1%，不能读成"求解器帧数够"。
    """
    run = _run_with_solver_skip(tmp_path, self_verdict="ANALYSIS_ELIGIBLE",
                                skip_frames=7)
    view = Stage2RepairController.for_physical_stage(run, "vanishing", "vdw").read()
    ev = next(w for w in view["windows"] if w["window_idx"] == 1)["evidence_decorrelation"]

    assert ev["solver"]["n_frames_after_decorrelation"] == 7
    assert ev["self_check"]["n_frames_decorrelated"] == 120     # fixture 默认
    assert ev["authority"].startswith("solver_decides")
    # 自检那一项必须显式说明它**不只是**帧数
    assert "sufficient_INCLUDES_N_eff_AND_top1pct" in ev["self_check"]
    assert "不得用一侧的帧数反推另一侧" in ev["do_not_reconcile"]


# ---------------------------------------------------------------------------
# 有界 IMMUTABLE_REWINDOW + 9b 的中间窗归因 + S2-A 的 UNMEASURED
# ---------------------------------------------------------------------------

def _stage_support_failure(worst_window):
    """target_support 门失败，最差窗口按 raw ESS 落在 `worst_window` 上。"""
    return {
        "converged": False,
        "target_support_gate": {
            "passed": False, "failed_checks": ["ibs_segment_target_support"],
            "raw_min_absolute_ess_threshold": 20.0,
            "ibs_segment_gate": {
                "failed_checks": ["raw_absolute_ess_below_threshold"],
                "max_top1pct_raw_weight_threshold": 0.35,
            },
        },
        "window_overlap_diagnostics": [
            {"window_index": i, "raw_min_absolute_ess": (1.1 if i == worst_window
                                                         else 90.0),
             "top1pct_raw_weight": 0.02, "n_frames_decorrelated": 60}
            for i in range(4)
        ],
    }


def test_a_middle_window_failure_never_buys_an_unrelated_tail_split(tmp_path):
    """中间窗的支撑失败不许换来一次与它无关的尾段重分。

    先前 `_can_split` 只问"拆末窗结构上可不可行"，不问"失败的是不是末窗"
    ⟹ 烧 GPU、改布局、作废下游证据，而元凶窗口一个字节没动。
    """
    # 末窗 K=9 落在可拆区间 [2lo−1, 2hi−1]=[7,9] ⟹ 拆末窗**结构上可行**；
    # 但失败的是中间窗 1，拆它一点用没有。
    _, plan = _plan(
        tmp_path,
        windows={0: {"K": 4}, 1: {"K": 5}, 2: {"K": 9}},
        ranges=[(0, 4), (3, 8), (7, 16)], n_states=16,
        config={"stage2_window_min_states": 4, "stage2_window_max_states": 5,
                "max_path_insertions": 3},
        stage_result=_stage_support_failure(worst_window=1),   # 中间窗
    )
    assert plan["action"] != "SPLIT_TAIL_WINDOW", plan["reason"]
    assert plan["windows"] == [1], plan["reason"]


def test_fixed_lambda_table_middle_window_falls_back_to_bounded_rewindow(tmp_path):
    """λ 表上的两个动作都不可行 ⟹ 固定 λ 表上的**有界**重窗，而不是终止。"""
    # 末窗已经顶到**溢出槽上限** 2*hi−1=9 ⟹ 插 λ 不可行（λ 表不能再动）；
    # 失败的是中间窗 ⟹ 拆末窗不对症（它修的不是这个窗口）。
    _, plan = _plan(
        tmp_path,
        windows={0: {"K": 4}, 1: {"K": 5}, 2: {"K": 9}},
        ranges=[(0, 4), (3, 8), (7, 16)], n_states=16,
        config={"stage2_window_min_states": 4, "stage2_window_max_states": 5,
                "max_path_insertions": 3},
        stage_result=_stage_support_failure(worst_window=1),
    )
    assert plan["action"] == "IMMUTABLE_REWINDOW", plan["reason"]
    assert plan["windows"] == [1]
    assert not plan["terminal"], "有界动作不是终态"


def test_rewindow_is_built_once_per_parent_window(tmp_path):
    """一个父窗口只建一次子系综；要加帧走 RUN_PRODUCTION、落在同一个子系综里。"""
    import json as _json

    run = _mkrun(
        tmp_path,
        windows={0: {"K": 4}, 1: {"K": 5}, 2: {"K": 9}},
        ranges=[(0, 4), (3, 8), (7, 16)], n_states=16,
        config={"stage2_window_min_states": 4, "stage2_window_max_states": 5,
                "max_path_insertions": 3},
        stage_result=_stage_support_failure(worst_window=1),
    )
    (pathlib.Path(run) / "checkpoints" / "stage2_rewindow_ledger.json").write_text(
        _json.dumps({"abc123": {"identity": "abc123", "parent_window": 1,
                                "child_ranges": [[3, 6], [5, 7]],
                                "f_k_scope": "own_frozen_f_k_per_child_ensemble",
                                "blocks": [{"steps_per_child": 250000}]}})
    )
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] != "IMMUTABLE_REWINDOW", (
        f"窗口 1 已经建过子系综，不该再建一个：{plan['reason']}"
    )


def test_structurally_unmeasurable_residual_is_UNMEASURED_not_missing(tmp_path):
    """S2-A：多段窗口的残差**结构上**算不出来 ⟹ `UNMEASURED`，不是"还没算"。

    落成 `None` 会被 5a-1 读成"去跑 ANALYZE 把它算出来" —— 而它永远算不出来，
    于是又是一个无限 ANALYZE。有界 rewindow 的子系综各自锁自己的 f_k，
    **不能**替父窗口把这道门宣称通过。
    """
    run = _mkrun(
        tmp_path,
        windows={i: {"K": 4} for i in range(4)},
        ranges=R4, n_states=13,
        stage_result={
            "converged": False,
            "cumulative_fk_residual_production": [{
                "window_index": 1,
                "error": "no_effective_f_k_for_these_frames",
            }],
        },
    )
    view = Stage2RepairController(run, "vanishing").read()
    w1 = next(w for w in view["windows"] if w["window_idx"] == 1)
    assert w1["cum_fk_verdict"] == "UNMEASURED"
    assert w1["cum_fk_unmeasured_reason"] == "no_effective_f_k_for_these_frames"


def test_rewindow_cannot_claim_the_residual_gate_for_its_parent(tmp_path):
    """父窗口被子系综取代后连一条残差记录都没有 ⟹ 仍是 `UNMEASURED`。

    子系综各自锁自己的 f_k，**不能**替父窗口把累计残差门宣称通过；
    也不该因为"没有记录"就把控制器送去空跑 ANALYZE。
    """
    import json as _json

    run = _mkrun(
        tmp_path,
        windows={i: {"K": 4} for i in range(4)},
        ranges=R4, n_states=13,
        stage_result={"converged": False,
                      "cumulative_fk_residual_production": []},   # 父窗口没记录
    )
    (pathlib.Path(run) / "checkpoints" / "stage2_rewindow_ledger.json").write_text(
        _json.dumps({"abc123": {"identity": "abc123", "parent_window": 2,
                                "child_ranges": [[6, 9], [8, 10]],
                                "f_k_scope": "own_frozen_f_k_per_child_ensemble",
                                "blocks": [{"steps_per_child": 250000}]}})
    )
    view = Stage2RepairController(run, "vanishing").read()
    w2 = next(w for w in view["windows"] if w["window_idx"] == 2)
    assert w2["cum_fk_verdict"] == "UNMEASURED"
    assert w2["cum_fk_unmeasured_reason"] == (
        "replaced_by_immutable_rewindow_children_own_f_k")
    # 没被 rewindow 的窗口不受影响：证据确实还没产出 ⟹ None
    w0 = next(w for w in view["windows"] if w["window_idx"] == 0)
    assert w0["cum_fk_verdict"] is None


def test_a_solver_skipped_window_never_gets_a_recalibration_noop(tmp_path):
    """真机 cyclod：win0 被求解器跳掉（9 帧 < 10），控制器却连发 4 次
    `RECALIBRATE_FK[0]`，每次执行器都报「没有窗口超过 0.5 kJ/mol 位移阈值 ⟹
    不新开采样段」，盘面一个字节没变，最后 NO_FEASIBLE_ACTION 退出。

    根因：重标定/换 Epoch 都要**拿这个窗口已有的帧重解**，而"被跳掉"的意思正是
    "这些帧不够解" ⟹ 整族动作在构造上是 no-op。而且探针按**节奏**推荐、执行器按
    **位移**放行 —— 同一个决定两套判据，且被跳的窗口根本没有位移读数。
    """
    w = dict(FULL)
    w[0] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA", "min_n_eff_over_g": 3.1,
            "n_decorr": 9}
    _, plan = _plan(
        tmp_path, windows=w, ranges=R4, n_states=13,
        stage_result={
            "converged": False,
            "skipped_windows": [{"window_index": 0,
                                 "n_frames_after_decorrelation": 9,
                                 "min_frames_per_window": 10,
                                 "reason": "insufficient_frames_after_decorrelation"}],
            # 探针把它列进"建议重标定"（按节奏，不是按位移）
            "cumulative_fk_residual_production": [
                {"window_index": 0, "verdict": "PASS",
                 "cumulative_residual_span_kJ_mol": 1.2}],
        },
    )
    assert plan["action"] == "RUN_PRODUCTION", plan["reason"]
    assert plan["action"] not in ("RECALIBRATE_FK", "PROBE_REANCHOR_EPOCH",
                                 "RELEARN_FK_EPOCH")
    assert plan["windows"] == [0]
    assert "no-op" in plan["reason"]


def test_an_action_already_proven_noop_on_this_disk_state_is_not_reissued(tmp_path):
    """执行器报「重标定未产生新段」之后，控制器不许再发同一个动作。

    真机两次都是这个形状：`RECALIBRATE_FK[0]` 连发 4 次、盘面一字节没变，
    靠停滞保护退出 —— 而退出前**没有**去试那个真正对症的动作。
    执行器**知道**自己没做事，控制器却无从得知；这条记账把信息交回去。

    指纹只取「加帧/换段就会变」的量 ⟹ 补过帧之后记录自动失效、动作重新可选。
    """
    import json as _json

    from abfe_preoptimizer import action_noop_fingerprint

    w = dict(FULL)
    w[0] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA", "min_n_eff_over_g": 3.1,
            "prod": 250000}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13,
                 stage_result={"converged": False,
                               "cumulative_fk_residual_production": [
                                   {"window_index": 0, "verdict": "PASS",
                                    "cumulative_residual_span_kJ_mol": 1.2}]})
    # f_k 探针按**节奏**把 win0 列进建议重标定名单（真机就是这么来的）
    (pathlib.Path(run) / "checkpoints" / "stage2_fk_recalibration_probe.json").write_text(
        _json.dumps({"verdict": "RECALIBRATE",
                     "recalibration_recommended_windows": [0],
                     "min_adjacent_shift_kJ_mol": 0.5})
    )
    c = Stage2RepairController(run, "vanishing")
    before = c.decide()
    assert before["action"] == "RECALIBRATE_FK", "fixture 没造出真机那个形状"

    # 把"这个盘面上重标定/探针都是 no-op"记上
    w0 = next(x for x in c.read()["windows"] if x["window_idx"] == 0)
    fp = action_noop_fingerprint(w0, c.read().get("path_version"))
    (pathlib.Path(run) / "checkpoints" / "stage2_noop_actions.json").write_text(
        _json.dumps({
            f"{a}:0": {"action": a, "window_idx": 0, "fingerprint": fp,
                       "reason": "recalibration_produced_no_new_segment"}
            for a in ("RECALIBRATE_FK", "PROBE_REANCHOR_EPOCH")})
    )
    after = Stage2RepairController(run, "vanishing").decide()

    assert after["action"] not in ("RECALIBRATE_FK", "PROBE_REANCHOR_EPOCH"), (
        f"同一个盘面上已证明是 no-op 的动作又被发了一次：{after['reason']}"
    )
    assert after["action"] == "RUN_PRODUCTION", after["reason"]


def test_the_noop_record_expires_once_more_frames_land(tmp_path):
    """记的是「在这个盘面上没用」，不是「这个动作永远没用」。"""
    from abfe_preoptimizer import action_noop_fingerprint

    a = action_noop_fingerprint({"production_steps": 250000, "segment": "vanishing"}, 1)
    b = action_noop_fingerprint({"production_steps": 500000, "segment": "vanishing"}, 1)
    c = action_noop_fingerprint({"production_steps": 250000, "segment": "vanishing_2"}, 1)
    d = action_noop_fingerprint({"production_steps": 250000, "segment": "vanishing"}, 2)
    assert a != b and a != c and a != d


# ---------------------------------------------------------------------------
# 裁决 2：生产预算是**独立的一本账**
# ---------------------------------------------------------------------------

def _budget_run(tmp_path, cap=None, prod=250000):
    cfg = {"stage2_window_min_states": 4, "stage2_window_max_states": 8,
           "max_path_insertions": 3}
    if cap is not None:
        cfg["stage2_production_budget_steps"] = cap
    w = {i: {"K": 4, "prod": prod} for i in range(4)}
    w[0] = {"K": 4, "prod": prod, "self_verdict": "INSUFFICIENT_DATA",
            "min_n_eff_over_g": 3.1}
    return _mkrun(tmp_path, windows=w, ranges=R4, n_states=13, config=cfg)


def test_an_unknown_production_cap_is_unknown_not_zero(tmp_path):
    """上限读不到 ⟹ `unknown`。**不能当零，也不能由预热余额代替** ——
    「预算耗尽」这个结论在上限未知时根本不成立。"""
    pb = Stage2RepairController(_budget_run(tmp_path), "vanishing").read()[
        "production_budget"]
    assert pb["cap_known"] is False
    assert pb["stage_cap_steps"] is None
    assert pb["stage_remaining_steps"] is None
    assert pb["exhausted"] is False        # 未知 ≠ 耗尽
    assert pb["stage_used_steps"] == 4 * 250000


def test_production_and_warmup_are_two_separate_books(tmp_path):
    """预热余额为 0 **不**代表生产预算耗尽 —— 那正是 09-14 修掉的那个缺陷。"""
    cfg = {"stage2_window_min_states": 4, "stage2_window_max_states": 8,
           "max_path_insertions": 3}
    w = {i: {"K": 4, "warmup": 555000, "cap": 555000} for i in range(4)}
    w[0] = {"K": 4, "warmup": 555000, "cap": 555000,
            "self_verdict": "INSUFFICIENT_DATA", "min_n_eff_over_g": 3.1}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13, config=cfg)
    view = Stage2RepairController(run, "vanishing").read()
    assert view["all_windows_budget_exhausted"] is True       # 预热账干了
    assert view["production_budget"]["exhausted"] is False    # 生产账另算


def test_an_exhausted_known_production_budget_halts_every_gpu_action(tmp_path):
    """只有**已知且确实耗尽**才发 `GLOBAL_BUDGET_EXHAUSTED`，而且是真终态。"""
    from abfe_preoptimizer import Stage2RepairController as C

    plan = Stage2RepairController(
        _budget_run(tmp_path, cap=100000), "vanishing").decide()
    assert plan["action"] == "NO_ACTION", plan["reason"]
    assert plan["exit"] == "GLOBAL_BUDGET_EXHAUSTED"
    assert plan["exit"] in C.TERMINAL_EXITS and plan["terminal"]
    # 原动作与理由必须留在记录里，不能被这道闸抹掉
    assert "原动作与理由" in plan["reason"]


def test_a_plentiful_known_budget_does_not_halt_anything(tmp_path):
    plan = Stage2RepairController(
        _budget_run(tmp_path, cap=100_000_000), "vanishing").decide()
    assert plan["action"] != "NO_ACTION"
    assert plan["exit"] != "GLOBAL_BUDGET_EXHAUSTED"


def test_analysis_is_still_allowed_when_the_budget_is_gone(tmp_path):
    """ANALYZE / DONE / NO_ACTION 不花 GPU ⟹ 不该被这道闸拦。"""
    run = _mkrun(tmp_path, windows={i: {"K": 4} for i in range(4)},
                 ranges=R4, n_states=13,
                 config={"stage2_window_min_states": 4,
                         "stage2_window_max_states": 8,
                         "max_path_insertions": 3,
                         "stage2_production_budget_steps": 1},
                 stage_result={"converged": True, "stage": "vanishing"})
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] == "DONE", plan["reason"]


def test_continue_warmup_proven_noop_falls_through_to_production(tmp_path):
    """真机 rep2：`CONTINUE_WARMUP[4]` 连发 **40 次**，`production_steps` 与
    `warmup_steps_left` 一动不动，`repeat_count` 在 1→2→3 之间循环 ——
    停滞保护每次降级到探针、探针又改了一点盘面把计数清零，永远走不到退出。

    根因：窗口级 resume 缓存门在**走到预热之前**就跳过整个窗口
    （"已有有效缓存能量 … resume 模式下跳过重新采样"）⟹ 续预热一步也跑不了。
    它真正缺的是生产帧（端点 σ），那才是对症动作。
    """
    import json as _json

    from abfe_preoptimizer import action_noop_fingerprint

    w = dict(FULL)
    # 仍在预热验证、预算有余 ⟹ 正常会选 CONTINUE_WARMUP
    w[0] = {"K": 4, "bias_status": "frozen_validation_indeterminate",
            "prod": 1750000, "warmup": 130000, "cap": 955000}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)
    c = Stage2RepairController(run, "vanishing")
    assert c.decide()["action"] == "CONTINUE_WARMUP", "fixture 没造出真机那个形状"

    w0 = next(x for x in c.read()["windows"] if x["window_idx"] == 0)
    (pathlib.Path(run) / "checkpoints" / "stage2_noop_actions.json").write_text(
        _json.dumps({"CONTINUE_WARMUP:0": {
            "action": "CONTINUE_WARMUP", "window_idx": 0,
            "fingerprint": action_noop_fingerprint(w0, c.read().get("path_version")),
            "reason": "action_changed_nothing_on_disk"}})
    )
    after = Stage2RepairController(run, "vanishing").decide()
    assert after["action"] == "RUN_PRODUCTION", after["reason"]
    assert after["windows"] == [0]
    assert not after["terminal"]


def test_the_loop_records_a_noop_for_any_action_not_just_the_fk_family():
    """通用判据：执行前后比一次盘面指纹，一样就记账。

    逐动作打补丁必然漏下一个 —— 今天三起事故分别是重标定、探针、续预热。
    """
    import ast
    import inspect

    import abfe_pipeline

    src = inspect.getsource(abfe_pipeline.ABFEPipeline._run_stage2_autonomous)
    assert "_disk_signature(ctl.read()) == sig" in src
    assert "action_changed_nothing_on_disk" in src
    # 必须挂在**正常执行路径**末尾，不是某个 except 里
    fn = next(n for n in ast.walk(ast.parse(src.lstrip()))
              if isinstance(n, ast.FunctionDef))
    found = False
    for node in ast.walk(fn):
        if isinstance(node, ast.Try) and any(
                h.type is not None
                and "IBSValidationBudgetIndeterminateError" in ast.unparse(h.type)
                for h in node.handlers):
            found = "_disk_signature(ctl.read()) == sig" in ast.unparse(node.body[-1])
    assert found, "no-op 检测不在动作执行的正常路径末尾"


def test_a_window_that_cannot_be_re_entered_is_not_given_more_frames(tmp_path):
    """预热累计预算耗尽 + f_k 未判定 ⟹ **重新进入该窗口会在预热门上被弹回**。

    真机（cyclod_ligand1/rep3 win0）：`RUN_PRODUCTION[0]` 让缓存正确失效、
    重新进入窗口，入口处 warmup 门看到「上限 555000、已耗 555000，本次可用 0 步」
    直接抛 `LOCAL_VALIDATION_CAP`，路由回来又发同一个动作 —— 40 轮。
    补帧类动作在**构造上**推不动它；不需要重进预热的对症动作是缩跨度。
    """
    w = dict(FULL)
    w[0] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA", "min_n_eff_over_g": 3.1,
            "warmup": 555000, "cap": 555000, "evidence": "indeterminate",
            "bias_status": "frozen_validation_indeterminate"}
    _, plan = _plan(tmp_path, windows=w, ranges=R4, n_states=13)

    assert plan["action"] != "RUN_PRODUCTION", plan["reason"]
    assert plan["action"] != "CONTINUE_WARMUP", plan["reason"]
    assert "预热门上被" in plan["reason"] or "弹回" in plan["reason"]


def test_a_verified_window_with_no_warmup_budget_is_still_toppable(tmp_path):
    """f_k 已 `verified` 的窗口不需要再过验证门 ⟹ 预热余额为 0 不挡补帧。"""
    w = dict(FULL)
    w[0] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA", "min_n_eff_over_g": 3.1,
            "warmup": 555000, "cap": 555000, "evidence": "verified"}
    _, plan = _plan(tmp_path, windows=w, ranges=R4, n_states=13)
    assert plan["action"] == "RUN_PRODUCTION", plan["reason"]


def test_routing_signals_also_record_a_noop():
    """路由那条 `continue` 绕过了正常路径末尾的检测 —— 必须自己记一次。"""
    import inspect

    import abfe_pipeline

    src = inspect.getsource(abfe_pipeline.ABFEPipeline._run_stage2_autonomous)
    assert "routed_back_without_progress" in src
    assert src.count("_disk_signature(ctl.read()) == sig") == 2


def test_budget_admission_uses_the_full_cost_of_the_next_action(tmp_path):
    """剩 100k 不许再发 250k 块 —— 准入按**下一动作的完整成本**，不是"已经耗尽"。"""
    cfg = {"stage2_window_min_states": 4, "stage2_window_max_states": 8,
           "max_path_insertions": 3, "n_steps_per_window": 250000,
           # 4 窗 × 250k = 1,000,000 已用；上限给 1,100,000 ⟹ 只剩 100k
           "stage2_production_budget_steps": 1_100_000}
    w = {i: {"K": 4, "prod": 250000} for i in range(4)}
    w[0] = {"K": 4, "prod": 250000, "self_verdict": "INSUFFICIENT_DATA",
            "min_n_eff_over_g": 3.1}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13, config=cfg)
    c = Stage2RepairController(run, "vanishing")
    pb = c.read()["production_budget"]
    assert pb["exhausted"] is False and pb["stage_remaining_steps"] == 100_000

    plan = c.decide()
    assert plan["action"] == "NO_ACTION", plan["reason"]
    assert plan["exit"] == "GLOBAL_BUDGET_EXHAUSTED"
    assert "付不起" in plan["reason"]


def test_cap_scope_is_production_only_on_both_sides(tmp_path):
    """建窗准入与完成后的记账必须用**同一个**计量范围。"""
    pb = Stage2RepairController(
        _budget_run(tmp_path, cap=10_000_000), "vanishing").read()["production_budget"]
    assert pb["cap_scope"] == "production_steps_only"
    # 预留 = 每个子窗一个**生产**首块，不含预热
    assert pb["new_ensemble_reserve_steps"] == pb["production_block_steps"]


def test_all_segments_count_toward_the_budget_not_just_the_winning_one(tmp_path):
    """旧段的帧是真烧过的 GPU —— 只取胜出段，账永远比实际宽。"""
    import json as _json
    import shutil

    cfg = {"stage2_window_min_states": 4, "stage2_window_max_states": 8,
           "max_path_insertions": 3}
    run = _mkrun(tmp_path, windows={i: {"K": 4, "prod": 250000} for i in range(4)},
                 ranges=R4, n_states=13, config=cfg)
    # 造一个 vanishing_2 段：win0 在那里又烧了 500k
    seg = pathlib.Path(run) / "vanishing_2"
    seg.mkdir()
    (pathlib.Path(run) / "checkpoints" / "segment_2").mkdir()
    src = pathlib.Path(run) / "vanishing" / "dual_window_0_vdw_convergence.json"
    d = _json.loads(src.read_text())
    d["cumulative_production_steps"] = 500000
    (seg / "dual_window_0_vdw_convergence.json").write_text(_json.dumps(d))
    for nm in ("dual_window_0_vdw_self_support.json",):
        shutil.copy(pathlib.Path(run) / "vanishing" / nm, seg / nm)
    shutil.copy(pathlib.Path(run) / "checkpoints" / "ibs_state_vdw_window_0.json",
                pathlib.Path(run) / "checkpoints" / "segment_2"
                / "ibs_state_vdw_window_0.json")

    agg = Stage2RepairController.for_physical_stage(run, "vanishing", "vdw").read()
    # 4×250k（基准段）+ 500k（段 2 的 win0）
    assert agg["production_budget"]["stage_used_steps"] == 1_500_000
    assert agg["production_budget"]["per_window_used_steps"][0] == 750_000
