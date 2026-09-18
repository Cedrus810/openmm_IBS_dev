"""两条：① 现有 run 的采样身份**可证明地**恢复；② 物理窗口也要分"缺帧/偏斜"。

**①** 写侧此前从来没把 `stage_protocol_key` 盖到 sampler 上 ⟹ 所有已存在的
`ibs_state_*.json` 里都是 None ⟹ 身份闸恒判"缺失"⟹ 每次 resume 重走完整预热、
白烧预热预算。只修写侧的话，现有 run **还要再烧一轮**。
恢复**不是"缺了就当通过"**，而是从一份独立的、同一次生产写下的记录
（同窗口 `convergence.json`）里取回来，且 λ 必须逐位相等。

**②** `sufficient=False` 混了三项（帧数 / `N_eff/g` / top1%）。子窗那侧已按
CTL-02 拆开，物理窗口这一侧先前仍然一律补帧 —— 同一个毛病漏了一处。
"""
import json
import os
import pathlib
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ibs_engine as ie  # noqa: E402
from abfe_preoptimizer import Stage2RepairController  # noqa: E402

from test_stage2_repair_controller import _mkrun, R4, FULL  # noqa: E402

LAM = [1.0, 0.9, 0.8, 0.7]
KEY = {"schema_version": 3, "stage": "vanishing", "box": "abc"}


def _make(tmp_path, *, conv_lam=LAM, conv_key=KEY):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "vanishing").mkdir()
    if conv_key is not None:
        (run / "vanishing" / "dual_window_2_vdw_convergence.json").write_text(
            json.dumps({"stage_protocol_key": conv_key, "lambdas_vdw": conv_lam}))
    sp = run / "checkpoints" / "ibs_state_vdw_window_2.json"
    sp.write_text(json.dumps({"lambdas_vdw": LAM}))
    return str(sp)


# ---------------------------------------------------------------- ①
def test_identity_is_recovered_from_the_same_windows_convergence(tmp_path):
    sp = _make(tmp_path)
    key, src = ie._recover_stage_identity_from_convergence(
        sp, {}, None, LAM, "vdw")
    assert key == KEY
    assert src.endswith("dual_window_2_vdw_convergence.json")


def test_a_convergence_from_another_layout_is_never_borrowed(tmp_path):
    """λ 对不上 ⟹ 那份记录属于另一套布局，借过来等于用另一个系综的身份背书。"""
    sp = _make(tmp_path, conv_lam=[1.0, 0.9, 0.8])      # 少一个态
    key, src = ie._recover_stage_identity_from_convergence(
        sp, {}, None, LAM, "vdw")
    assert key is None and src is None


def test_recovery_stays_fail_closed_when_there_is_nothing_to_recover(tmp_path):
    sp = _make(tmp_path, conv_key=None)
    assert ie._recover_stage_identity_from_convergence(
        sp, {}, None, LAM, "vdw") == (None, None)


def test_the_loader_records_where_the_identity_came_from():
    """恢复来的身份必须可审计 —— 不能跟原生写入的混为一谈。"""
    import inspect
    src = inspect.getsource(ie.IBSSampler.load_ibs_state)
    assert "loaded_stage_protocol_key_source" in src
    assert "_recover_stage_identity_from_convergence(" in src


# ---------------------------------------------------------------- ②
def _run_with_self_failure(tmp_path, *, source):
    w = dict(FULL)
    w[0] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA",
            "self_verdict_source": source, "min_n_eff_over_g": 3.1}
    return _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)


def test_a_skewed_physical_window_is_not_answered_with_more_frames(tmp_path):
    run = _run_with_self_failure(tmp_path, source="top1pct_veto")
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] != "RUN_PRODUCTION", plan["reason"]
    # 🔑 [2026-09-18] `RELEARN_FK_EPOCH` 进这张表：本用例钉的是「**不拿加帧顶替
    # 偏斜**」，而换 f_k 同样不是加帧。`top1pct_veto` 现在先走它 —— 它和缩跨度
    # 同属 η 杠杆，但不像插 λ 那样必然把溢出推给末窗（model B 后置断言）。
    assert plan["action"] in (
        "RELEARN_FK_EPOCH", "INSERT_LAMBDA", "SPLIT_TAIL_WINDOW", "NO_ACTION")
    assert ("加帧治不了偏斜" in plan["reason"]
            or "不拿加帧顶替" in plan["reason"]
            or "top1pct_veto" in plan["reason"])


def test_a_sample_size_failure_still_gets_frames(tmp_path):
    run = _run_with_self_failure(tmp_path, source="solver_eligibility")
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] == "RUN_PRODUCTION", plan["reason"]
    assert plan["windows"] == [0]
