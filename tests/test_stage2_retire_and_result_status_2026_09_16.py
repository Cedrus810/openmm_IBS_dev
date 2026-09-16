"""2026-09-16 大修：三条经真机产物核实过的缺陷。

证据来自 `abfe-benchmark/openmm_IBS/runs` 的三个 run（brd4_ligand2 rep1/rep3、
cyclod_ligand2 rep2，全部 `results_untrusted=true`）：

① **一个窗口修不动 ≠ 整条 stage 没动作可做。**
   rep1 连发 4 块给 win2 撞上补帧配额 ⟹ `NO_FEASIBLE_ACTION` ⟹ 主循环 break，
   而末窗 win4 **一块都没批过**、预热预算还剩 37 万步，全程躺在
   `blocked_by_upstream` 里。整腿半程漂移 +8.01 kJ/mol 里 win4 独占 +8.005。
   （cyclod rep2 同形：4 块给 win3，win4 的 `N_eff/g=0.186`、剩 87 万步、零块。）

② **`min_n_eff_over_g` 偏低不蕴含「加帧治不了」。**
   `N_eff/g = n_decorr × (N_eff/N)`，右边第二项是强度量 ⟹ 同分布加帧让比值线性
   涨，5 能长到 10。先前无条件判成偏斜、送去改布局。

③ **不可信状态在汇总处丢失。**
   `abfe_pipeline` 标了 `results_untrusted=True` 且刻意不 raise，而
   `final_binding_results.json` 四个状态键一个都没带、收尾还无条件打印「计算完成」。
"""
import json
import os
import pathlib
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abfe_preoptimizer import (  # noqa: E402
    Stage2RepairController,
    frames_growth_headroom,
    n_eff_over_g_reachable_by_frames,
    support_failure_is_skew,
)
from ibs_engine import (  # noqa: E402
    WINDOW_SELF_SUPPORT_N_EFF_OVER_G_ELIGIBLE,
    window_self_support_path,
)

from test_stage2_repair_controller import _mkrun, R4  # noqa: E402


# --------------------------------------------------------------------- ②
#   加帧的射程：`ratio × headroom` 够不够得着门


def test_headroom_is_the_unspent_share_of_the_block_quota():
    v = {"max_production_blocks_per_window": 4,
         "production_blocks_total_by_window": {1: [{}, {}, {}, {}], 3: []}}
    assert frames_growth_headroom(v, 3) == 5.0      # 一块没花 ⟹ 还能到 5 倍
    assert frames_growth_headroom(v, 1) == 1.0      # 配额见底 ⟹ 一帧都加不了
    assert frames_growth_headroom(v, 2) == 5.0      # 账里没有 = 没花过
    # 上限未知 ⟹ 射程算不出来，**不猜**（返回 None ⟹ 调用方退回既有判法）
    assert frames_growth_headroom({"production_blocks_total_by_window": {}}, 0) is None


def test_reachability_is_an_identity_not_a_new_threshold():
    # 5 × 5 倍 = 25 >= 10 ⟹ 够得着
    assert n_eff_over_g_reachable_by_frames(5.0, 10.0, 5.0) is True
    # 0.186 × 5 = 0.93 < 10 ⟹ 够不着（cyclod rep2 win4 的真实读数）
    assert n_eff_over_g_reachable_by_frames(0.186, 10.0, 5.0) is False
    # 8.794 但配额见底 ⟹ 够不着（brd4 rep1 win2 的真实读数）
    assert n_eff_over_g_reachable_by_frames(8.794, 10.0, 1.0) is False
    # 缺任何一个数都判不了，**不许当成 False 用**
    for args in ((None, 10.0, 5.0), (5.0, None, 5.0), (5.0, 10.0, None)):
        assert n_eff_over_g_reachable_by_frames(*args) is None


def test_low_ratio_with_budget_left_is_a_sample_size_problem_not_skew():
    """要害：比值低 + 配额还在 + 射程够得着 ⟹ 补帧，别送去改布局。"""
    assert support_failure_is_skew(
        "INSUFFICIENT_DATA", "min_n_eff_over_g",
        n_decorrelated=40, min_frames=10,
        min_n_eff_over_g=5.0, n_eff_over_g_target=10.0, frames_headroom=5.0,
    ) is False


def test_low_ratio_out_of_reach_is_still_skew():
    """射程够不着 ⟹ 仍然是偏斜，对症动作是缩跨度（口径没放宽）。"""
    assert support_failure_is_skew(
        "HARD_INSUFFICIENT", "min_n_eff_over_g",
        n_decorrelated=20, min_frames=10,
        min_n_eff_over_g=0.186, n_eff_over_g_target=10.0, frames_headroom=5.0,
    ) is True


def test_missing_reachability_evidence_keeps_the_old_verdict():
    """老产物没有门槛键 ⟹ 判不了射程 ⟹ 保持既有（保守当偏斜）行为。"""
    assert support_failure_is_skew(
        "INSUFFICIENT_DATA", "min_n_eff_over_g",
        n_decorrelated=888, min_frames=10,
    ) is True


def test_sample_size_sources_are_untouched():
    """`solver_eligibility` 仍然是样本量类；通过的窗口仍然不算失败。"""
    assert support_failure_is_skew(
        "HARD_INSUFFICIENT", "solver_eligibility",
        min_n_eff_over_g=0.1, n_eff_over_g_target=10.0, frames_headroom=1.0,
    ) is False
    assert support_failure_is_skew("ANALYSIS_ELIGIBLE", "min_n_eff_over_g") is False


def test_the_threshold_travels_with_the_artifact():
    """门槛由写侧落盘、读侧从产物里读，不许两边各抄一份常量。

    （本仓 `docs/STAGE2_CONTROLLER_DESIGN_2026-09-12.md` 把「同一不变量两份实现」
    记成最贵的 bug —— 控制器要算"加帧的射程"就必须知道门在哪。）
    """
    root = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    engine = (root / "ibs_engine.py").read_text(encoding="utf-8")
    # 写侧：档位是常量，且随产物一起交出去
    assert "elif _ratio < 10.0:" not in engine       # 判读处不再有字面量
    assert "WINDOW_SELF_SUPPORT_N_EFF_OVER_G_ELIGIBLE" in engine
    assert '"n_eff_over_g_eligible_threshold"' in engine
    # 读侧：从窗口记录里取，不硬编码
    pre = (root / "abfe_preoptimizer.py").read_text(encoding="utf-8")
    assert 'selfchk.get(\n                "n_eff_over_g_eligible_threshold")' in pre
    assert "n_eff_over_g_target=w.get(\"self_n_eff_over_g_eligible\")" in pre
    assert WINDOW_SELF_SUPPORT_N_EFF_OVER_G_ELIGIBLE == 10.0
    assert callable(window_self_support_path)


# --------------------------------------------------------------------- ①
#   退役：对这个窗口没有动作 ≠ 对这条 stage 没有动作


def _blocks_history(run, *, window, n_blocks, path_version=1):
    """给某个窗口伪造 n 块已批的补帧记录（`_production_blocks_scan` 的输入格式）。"""
    p = pathlib.Path(run) / "checkpoints" / "stage2_autonomous_history.json"
    hist = json.loads(p.read_text()) if p.exists() else {"iterations": []}
    for k in range(n_blocks):
        hist["iterations"].append({
            "iteration": len(hist["iterations"]) + 1,
            "action": "RUN_PRODUCTION",
            "path_version": path_version,
            "windows": [window],
            "snapshot": [{"window_idx": window, "segment": "vanishing",
                          "production_steps": 250000 * (k + 2),
                          "solver_n_decorrelated": 30 + 10 * k}],
        })
    p.write_text(json.dumps(hist))


def _two_broken_windows(tmp_path, *, capped=1, starved=3):
    """win{capped} 配额烧光、win{starved} 一块没批 —— 真机 rep1/rep2 的形状。"""
    w = {i: {"K": 4} for i in range(4)}
    for idx in (capped, starved):
        w[idx] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA",
                  # 样本量类：这样两个窗口的对症动作都是补帧，检验的才是"路由到谁"
                  "self_verdict_source": "solver_eligibility", "n_decorr": 5}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)
    _blocks_history(run, window=capped, n_blocks=4)
    return run


def test_a_capped_window_no_longer_kills_the_whole_stage(tmp_path):
    """**要害**：上游窗口配额烧光 ⟹ 退役它，去修还有预算的下游窗口。

    先前这里是 `NO_ACTION` + `NO_FEASIBLE_ACTION`（终态）⟹ 主循环 break ⟹
    末窗一块都拿不到，正是 brd4_ligand2/rep1 与 cyclod_ligand2/rep2 的死法。
    """
    run = _two_broken_windows(tmp_path, capped=1, starved=3)
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] == "RUN_PRODUCTION", plan["reason"]
    assert plan["windows"] == [3], plan["reason"]
    assert plan["retired_windows"] == [1], plan.get("retired_windows")
    assert plan["terminal"] is False
    assert "已退役出路由顺序" in plan["reason"]


def test_retirement_is_not_a_pass(tmp_path):
    """退役只改路由顺序，**不放宽任何判据**：窗口 verdict 原样保留。"""
    run = _two_broken_windows(tmp_path, capped=1, starved=3)
    c = Stage2RepairController(run, "vanishing")
    view = c.read()
    rec = {int(w["window_idx"]): w for w in view["windows"]}
    assert rec[1]["self_verdict"] == "INSUFFICIENT_DATA"
    assert rec[1]["self_sufficient"] is False
    # 退役后 win1 的状态一个字段都没被改写
    plan = c.decide(view)
    assert plan["retired_windows"] == [1]
    assert rec[1]["self_verdict"] == "INSUFFICIENT_DATA"


def test_nothing_left_anywhere_still_terminates(tmp_path):
    """别处也没预算 ⟹ 原样交出终态，不退役、不空转。"""
    run = _two_broken_windows(tmp_path, capped=1, starved=3)
    _blocks_history(run, window=3, n_blocks=4)      # 下游也烧光
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] == "NO_ACTION", plan["reason"]
    assert plan["exit"] in Stage2RepairController.TERMINAL_EXITS
    assert plan["terminal"] is True
    assert "retired_windows" not in plan   # 没退役，也就没有"换个窗口"这回事


def test_a_single_broken_window_is_unchanged(tmp_path):
    """只有一个坏窗口时行为逐字不变（退役需要"别处还有活可干"）。"""
    w = {i: {"K": 4} for i in range(4)}
    w[1] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA",
            "self_verdict_source": "solver_eligibility", "n_decorr": 5}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)
    _blocks_history(run, window=1, n_blocks=4)
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["terminal"] is True, plan["reason"]
    assert "retired_windows" not in plan


# --------------------------------------------------------------------- ③
#   汇总文件必须带着状态


def _runabfe_src() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pathlib.Path(root, "runabfe.py").read_text(encoding="utf-8")


def _status(**kw):
    """只取两个纯函数跑，不 import runabfe（它拖着 openmm 一整条重依赖）。"""
    src = _runabfe_src()
    ns = {"Dict": dict, "Any": object}
    exec(compile(src[src.index("_LEG_STATUS_KEYS = ("):src.index("def main():")],
                 "runabfe_status", "exec"), ns)
    return ns["_binding_result_status"](kw.get("complex", {}), kw.get("solvent", {}))


_TRUSTED_LEG = {
    "results_untrusted": False,
    "precision_status": "MEETS_CROSS_REPEAT_TARGET",
    "publishable_as_accepted_result": True,
}


def test_an_untrusted_leg_makes_the_binding_result_untrusted():
    st = _status(
        complex={"results_untrusted": True, "precision_status": "UNMEASURED",
                 "publishable_as_accepted_result": False,
                 "stage_quality_failures": [{"gate": "target_support_gate"}]},
        solvent=dict(_TRUSTED_LEG),
    )
    assert st["results_untrusted"] is True
    assert st["results_untrusted_legs"] == ["complex"]
    assert st["publishable_as_accepted_result"] is False
    assert st["precision_status"] == "UNMEASURED"
    # 失败证据带着腿名一起交上去，别只剩一个布尔
    assert st["stage_quality_failures"] == [
        {"gate": "target_support_gate", "leg": "complex"}]


def test_a_silent_leg_is_not_a_passing_leg():
    """缺键 = 没表态。**不当成通过** —— 否则一条没写状态的腿会把 publishable 洗白。"""
    st = _status(complex=dict(_TRUSTED_LEG), solvent={})
    assert st["publishable_as_accepted_result"] is False
    assert st["results_untrusted"] is False      # 没表态也不伪造成"不可信"
    assert st["precision_status"] == "UNMEASURED"


def test_both_legs_clean_stays_publishable():
    st = _status(complex=dict(_TRUSTED_LEG), solvent=dict(_TRUSTED_LEG))
    assert st["publishable_as_accepted_result"] is True
    assert st["results_untrusted"] is False
    assert st["precision_status"] == "MEETS_CROSS_REPEAT_TARGET"


def test_the_drift_summary_carries_the_sigma_it_did_not_apply():
    """半程漂移与**未采用**的 σ 下界一起搬进汇总；`applied` 原样带出，不改判。"""
    leg = {
        "results_untrusted": True,
        "stage_diagnostics": {"stage2": {
            "split_half_diagnostics": {
                "available": True, "total_drift_kJ_mol": 8.012,
                "total_drift_over_2sigma": 2.295,
                "max_window_drift_over_2sigma": 3.205},
            "sigma_inflation_from_split_half": {
                "available": True, "total_error_mbar_kJ_mol": 1.745,
                "total_error_inflated_kJ_mol": 4.249},
            "sigma_inflation_applied": False,
        }},
    }
    d = _status(complex=leg, solvent=dict(_TRUSTED_LEG))["per_leg"]["complex"][
        "sampling_drift"]
    assert d["available"] is True
    assert d["total_drift_kJ_mol"] == pytest.approx(8.012)
    assert d["total_error_mbar_kJ_mol"] == pytest.approx(1.745)
    assert d["total_error_inflated_kJ_mol"] == pytest.approx(4.249)
    assert d["sigma_inflation_applied"] is False
    # 没有 stage-2 诊断的腿如实报 not available，不伪造 0
    assert _status(complex={}, solvent={})["per_leg"]["solvent"][
        "sampling_drift"] == {"available": False}


def test_the_final_summary_actually_writes_those_keys():
    """钉死落盘契约：四个键必须出现在 `final_bind_result` 里。"""
    src = _runabfe_src()
    _start = src.index("final_bind_result = {")
    body = src[_start:src.index('"thermodynamic_cycle_terms"', _start)]
    for key in ("result_status", "results_untrusted", "precision_status",
                "publishable_as_accepted_result", "stage_quality_failures"):
        assert f'"{key}"' in body, key
    # 收尾不再无条件报 OK
    tail = src[src.index("# ----- 7. 输出最终结果 -----"):]
    assert 'log.info("[OK] ABFE 计算完成")' in tail
    assert tail.index("results_untrusted") < tail.index('"[OK] ABFE 计算完成"')
