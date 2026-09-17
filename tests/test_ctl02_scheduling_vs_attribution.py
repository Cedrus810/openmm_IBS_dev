"""CTL-02：**调度状态不能代替失败归因。**

先前采样单元只有一个 `complete` 布尔，而它要求 `self_sufficient is True` ——
那个量**混合**了帧数、`N_eff/g`、top1% 三项。分支 1c 对所有 `complete=False`
一律发 `RUN_PRODUCTION` ⟹ **偏斜类**失败（top1% / raw ESS）的子窗被**反复加帧**，
而加帧治不了偏斜（§5.1 实测 250k→1M 让 top1% 从 0.545 涨到 0.762、ESS 比值更差）。

现在拆成两个正交的量：
  · `needs_frames`   —— 样本量不够 ⟹ 加帧
  · `support_failed` —— 支撑/偏斜不合格 ⟹ 缩跨度；没有有界动作就如实停下
"""
import json
import os
import pathlib
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abfe_preoptimizer import Stage2RepairController  # noqa: E402

from test_rewindow_sampling_units import _with_rewindow, BASE  # noqa: E402


def _units(run):
    return Stage2RepairController(run, "vanishing").read()["sampling_units"]


def _mark(run, local_idx, *, verdict, source, sufficient):
    """改写某个子窗的自检产物。"""
    led = json.loads((pathlib.Path(run) / "checkpoints"
                      / "stage2_rewindow_ledger.json").read_text())
    od = pathlib.Path(led["abc123"]["output_dir"])
    f = od / f"dual_window_{local_idx}_vdw_self_support.json"
    d = json.loads(f.read_text())
    d.update({"verdict": verdict, "verdict_source": source,
              "sufficient": sufficient})
    f.write_text(json.dumps(d))


def test_a_skew_failure_is_not_reported_as_needing_frames(tmp_path):
    """top1% 否决的子窗：帧数够，但支撑不合格 —— 两件事必须分得开。"""
    run = _with_rewindow(
        tmp_path,
        child_states=[("ANALYSIS_ELIGIBLE", 250000), ("ANALYSIS_ELIGIBLE", 250000)],
        solver_decorr=[200, 200],           # 最终门过了 ⟹ 不缺帧
    )
    _mark(run, 1, verdict="HARD_INSUFFICIENT", source="top1pct_veto",
          sufficient=False)
    u1 = _units(run)[1]

    assert u1["needs_frames"] is False, "帧数够却被判成缺帧 ⟹ 又会去反复加帧"
    assert u1["support_failed"] is True
    assert u1["support_failure_source"] == "top1pct_veto"
    assert u1["complete"] is False


def test_a_sample_size_failure_still_asks_for_frames(tmp_path):
    """样本量不足仍然走加帧 —— 这条不能被上一条误伤。"""
    run = _with_rewindow(
        tmp_path,
        child_states=[("ANALYSIS_ELIGIBLE", 250000), ("ANALYSIS_ELIGIBLE", 250000)],
        solver_decorr=[200, 11],            # 子窗 1 只有 11/20
    )
    u1 = _units(run)[1]
    assert u1["needs_frames"] is True
    assert u1["support_failed"] is False

    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] == "RUN_PRODUCTION"
    assert plan["unit_id"] == "rw:abc123:1"


def test_the_controller_stops_instead_of_topping_up_a_skewed_child(tmp_path):
    """**这才是 CTL-02 的要害**：加帧治不了偏斜，就如实停在 NO_FEASIBLE_ACTION。"""
    run = _with_rewindow(
        tmp_path,
        child_states=[("ANALYSIS_ELIGIBLE", 250000), ("ANALYSIS_ELIGIBLE", 250000)],
        solver_decorr=[200, 200],
    )
    _mark(run, 1, verdict="HARD_INSUFFICIENT", source="top1pct_veto",
          sufficient=False)
    plan = Stage2RepairController(run, "vanishing").decide()

    assert plan["action"] == "NO_ACTION", plan["reason"]
    # 🔑 [2026-09-17，用户拍板 D3] 出口从笼统的 `NO_FEASIBLE_ACTION` 换成专名。
    # 这条死线在 REWIND-01 接通之前**结构上不可达**（子窗只由 `IMMUTABLE_REWINDOW`
    # 产生，而它发不出来）；现在它第一次会被真的走到，所以值一个说得清的名字：
    # 不是"想不出动作"，是「**有界重窗这条路的数据模型到顶了**」——
    # 当前只能表达"一次物理父窗替代"（执行器/合并器没有 `parent_unit_id`、
    # 没有 ancestry、没有递归预算语义）。本条的**意图一字未变**：停下、不加帧。
    assert plan["exit"] == "D3_REWINDOW_DEPTH_EXHAUSTED", plan["reason"]
    assert plan["terminal"] is True, "有界重窗用尽是真终态，不是路由信号"
    assert plan["unit_id"] == "rw:abc123:1"
    assert "加帧治不了偏斜" in plan["reason"]
    assert "不递归、不补帧、不再建第二个" in plan["reason"]
    # 结构化诊断：走到这条终态时，人不该还要回去翻盘面
    d = plan["rewindow_depth_exhausted"]
    assert d["unit_id"] == "rw:abc123:1" and d["parent_window"] == 1
    assert d["support_failure_source"] == "top1pct_veto"
    assert d["self_verdict"] == "HARD_INSUFFICIENT"
    for k in ("range", "identity", "blocks_used",
              "bottleneck_state", "bottleneck_g", "bottleneck_eta",
              "bottleneck_ratio"):
        assert k in d, k


def test_a_healthy_child_is_complete(tmp_path):
    run = _with_rewindow(
        tmp_path,
        child_states=[("ANALYSIS_ELIGIBLE", 250000), ("ANALYSIS_ELIGIBLE", 250000)],
        solver_decorr=[200, 200],
    )
    assert all(u["complete"] for u in _units(run))


def test_branch_9b_does_not_unconditionally_top_up_a_rewindow_child():
    """9b（NARROW_SPAN）对子窗的补帧必须**先问它是不是真缺帧**。

    这是纵深防御：1c-2 只在该子窗的父窗是 `earliest` 时才拦得住；最终门把某个
    子窗点成 worst_window 而 1c-2 没触发时，9b 是最后一道。
    源码级判据 —— 这条分支很难在 fixture 里单独构造到。
    """
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "abfe_preoptimizer.py").read_text("utf-8")
# 🔑 [2026-09] `decide()` 现在只是 23 行的外壳（"退役一个窗口再判一次"），判断体是 `_decide_once`（1831 行）。
# 源码探针指着 `decide` 会一无所获 —— 断言"存在"的当场红，断言"不存在"的**静默变成假绿**。
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_decide_once")
    body = ast.unparse(fn)
    # 子窗分支里必须出现 needs_frames 的判断，且两支都存在
    # [2026-09] 循环变量从 `_u` 改名成 `u`；钉变量名是脆的，这里只钉**读的是哪个字段**。
    assert "get('needs_frames')" in body or 'get("needs_frames")' in body, \
        "子窗分支不再读 needs_frames ⟹ 样本量与支撑失败的分流没了"
    assert "加帧治不了" in body and "不拿加帧顶替" in body
