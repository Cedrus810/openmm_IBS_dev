"""布局变更之后，控制器不得发出「在构造上不可能成功」的动作。

2026-09-16 真机 28 个 run 的错误里，有 9 个是同一条链的不同出口：

    [自治] 插 λ：窗口 (17, 23) 跨度 0.2810 → 0.2810；末窗吸收溢出。
    ...
    [自治 5/40] 动作=PROBE_CANDIDATE_FK 窗口=[3]
    [自治] 执行 PROBE_CANDIDATE_FK 失败：ValueError('窗口 4 状态数与 window_ranges 不符')
    [自治] 主循环异常：ValueError('窗口 4 状态数与 window_ranges 不符')

`INSERT_LAMBDA` 明写「本动作不采样，受影响窗口的重采由下一轮逐块发
`RUN_PRODUCTION`」，所以从插 λ 到下游重采完成之间，下游产物描述的是**上一套布局**
—— 这是设计内的合法中间态。三个洞把它变成了崩溃：

  1. 局部动作（`only_windows=[3]`）的执行器去**载全路径**，于是崩在窗口 4 上；
  2. 停滞保护的降级直接改写 `act`，**绕过 `decide()` 的全部可行性守卫**；
  3. 插 λ 让末窗吸收溢出，可以把末窗顶到 `hi` 以上；而合法化只能靠拆末窗、
     拆末窗要 tail anchor，`tail_repartition_anchor` 在 `idx <= 0` 时恒为 None
     ⟹ **卡住的窗口是 window 0 时，插完就是一个拆不开的非法布局**，
     而那个非法布局在 fail-closed 之前**已经落盘**（cyclod_ligand3/rep1 的 v4，
     末窗 K=9 > hi=8）。
"""
import json
import os
import pathlib
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import abfe_pipeline  # noqa: E402
from abfe_preoptimizer import Stage2RepairController  # noqa: E402

from test_stage2_repair_controller import _mkrun, R4  # noqa: E402


# ---------------------------------------------------------------------------
# 洞 1：局部动作不得载全路径
# ---------------------------------------------------------------------------
def _stage_dir_with_all_windows(tmp_path, *, n_windows):
    """造一个「每个窗口的产物都在」的 stage 目录。

    这样 `_src_missing`（按文件是否存在算）为空，排除集合里出现的任何窗口
    都只能来自 `only_windows` 那一条 —— 否则测的就是别的东西。
    """
    stage_dir = tmp_path / "vanishing"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_windows):
        (stage_dir / f"dual_window_{i}_vdw_convergence.json").write_text(
            json.dumps({"window_idx": i}))
    return str(stage_dir)



def test_recalibrate_only_loads_the_windows_it_acts_on(tmp_path, monkeypatch):
    """`only_windows=[1]` ⟹ loader 必须把 0/2/3 一起排除掉。

    真机是这样炸的：窗口 4 的产物是插 λ **之前**的布局（态数对不上当前
    `window_ranges`），而 `PROBE_CANDIDATE_FK 窗口=[3]` 的执行器仍然载全路径，
    loader 对窗口 4 fail-closed 抛 ValueError ⟹ 炸穿主循环。
    下面那个循环本来就把 `only_windows` 之外的窗口 skip 掉，问题只在于它是
    **载完之后**才 skip 的。
    """
    seen = {}

    def _fake_loader(self, src_dir, ranges, lc, lv, **kw):
        seen["excluded"] = kw.get("excluded_local_windows")
        return []          # 没有窗口可重解 ⟹ 函数返回 (None, diagnostics)

    monkeypatch.setattr(
        abfe_pipeline.ABFEPipeline, "_load_ibs_window_outputs_from_dir",
        _fake_loader, raising=True,
    )
    pipe = abfe_pipeline.ABFEPipeline.__new__(abfe_pipeline.ABFEPipeline)
    pipe._log = lambda *a, **k: None
    pipe.sampling_score_sha256 = None

    # 四个窗口的产物都在盘上 ⟹ `_src_missing` 为空，排除集合只能来自 only_windows。
    stage_dir = _stage_dir_with_all_windows(tmp_path, n_windows=4)
    result, diag = pipe._recalibrate_f_k_and_resample_segment(
        lambda *a, **k: None,
        stage_dir=stage_dir,
        checkpoint_dir=str(tmp_path / "ck"),
        window_ranges=[(0, 4), (3, 7), (6, 10), (9, 13)],
        lambdas_var=[1.0 - 0.08 * i for i in range(13)],
        kt=2.5,
        only_windows=[1],
        probe_only=True,
    )

    assert seen["excluded"] is not None, "只修一个窗口却载了全路径"
    assert set(seen["excluded"]) == {0, 2, 3}, (
        f"应当排除 only_windows 之外的全部窗口，实际排除 {seen['excluded']}"
    )
    # `records` 的语义不变：被跳过的窗口仍然逐条出现在诊断里（重锚节奏逻辑读它）。
    skipped = {int(r["window"]): r.get("skipped") for r in diag["windows"]}
    assert skipped == {0: "not_in_only_windows", 2: "not_in_only_windows",
                       3: "not_in_only_windows"}


def test_no_only_windows_still_loads_the_whole_path(tmp_path, monkeypatch):
    """不限定窗口时行为逐字不变 —— 只排除「文件缺失」的那些。"""
    seen = {}

    def _fake_loader(self, src_dir, ranges, lc, lv, **kw):
        seen["excluded"] = kw.get("excluded_local_windows")
        return []

    monkeypatch.setattr(
        abfe_pipeline.ABFEPipeline, "_load_ibs_window_outputs_from_dir",
        _fake_loader, raising=True,
    )
    pipe = abfe_pipeline.ABFEPipeline.__new__(abfe_pipeline.ABFEPipeline)
    pipe._log = lambda *a, **k: None
    pipe.sampling_score_sha256 = None

    stage_dir = _stage_dir_with_all_windows(tmp_path, n_windows=4)
    pipe._recalibrate_f_k_and_resample_segment(
        lambda *a, **k: None,
        stage_dir=stage_dir,
        checkpoint_dir=str(tmp_path / "ck"),
        window_ranges=[(0, 4), (3, 7), (6, 10), (9, 13)],
        lambdas_var=[1.0 - 0.08 * i for i in range(13)],
        kt=2.5,
        only_windows=None,
        probe_only=True,
    )
    assert not seen["excluded"], (
        f"only_windows=None 且产物齐全时不该排除任何窗口，实际 {seen['excluded']}"
    )


# ---------------------------------------------------------------------------
# 洞 2/3 共用：过期布局的判据只有一份
# ---------------------------------------------------------------------------
def _run_with_stale_tail(tmp_path):
    """win3（末窗）的产物少一个态 —— 它描述的是插 λ 之前那套布局。"""
    lam = [round(1.0 - 0.07 * i, 8) for i in range(13)]
    w = {i: {"K": R4[i][1] - R4[i][0], "lambdas_vdw": lam[R4[i][0]:R4[i][1]]}
         for i in range(4)}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)
    pv = pathlib.Path(run) / "checkpoints" / "path_versions" / "v1.json"
    d = json.loads(pv.read_text())
    d["states"] = [{"id": f"s{i}", "lambda_vdw": lam[i]} for i in range(13)]
    d["lambdas_vdw"] = lam
    pv.write_text(json.dumps(d))
    f = (pathlib.Path(run) / "vanishing" / "dual_window_3_vdw_convergence.json")
    c = json.loads(f.read_text())
    c["lambdas_vdw"] = c["lambdas_vdw"][:-1]
    f.write_text(json.dumps(c))
    return run


def test_stale_layout_windows_is_one_implementation(tmp_path):
    """判据有两个消费者（`decide()` 的 1d-0、主循环的降级闸），只能有一份实现。"""
    run = _run_with_stale_tail(tmp_path)
    ctl = Stage2RepairController.for_physical_stage(run, "vanishing", "vdw")
    view = ctl.read()
    assert Stage2RepairController.stale_layout_windows(view) == {3}
    # 与 `decide()` 的路由一致：过期窗口只能重采。
    assert 3 in (view.get("stale_layout_evidence") or {})


def test_escalation_is_blocked_on_stale_layout_windows():
    """停滞保护的降级不得把重解类动作发给一个布局过期的窗口。

    真机 jnk1_ligand2/rep1：`RUN_PRODUCTION[3]` 被 `LOCAL_VALIDATION_CAP` 连弹
    3 次 ⟹ 降级成 `PROBE_REANCHOR_EPOCH[3]` ⟹ win3 的帧正是插 λ 之后的过期布局
    ⟹ `ValueError('窗口 3 lambda 内容与当前路径不匹配')` 炸穿主循环。
    降级绕过了 `decide()`，所以 1d-0 那道守卫一个字都没走到。
    """
    view = {
        "windows": [
            {"window_idx": 0}, {"window_idx": 1}, {"window_idx": 2},
            {"window_idx": 3, "stale_layout_evidence_only": True},
        ],
        "stale_layout_evidence": {3: ["vanishing"]},
    }
    stale = Stage2RepairController.stale_layout_windows(view)
    # 降级闸的判据：目标窗口与过期集合有交集 ⟹ 不许降级。
    assert sorted(stale & {3}) == [3]
    assert sorted(stale & {0, 1}) == [], "没过期的窗口不该被这道闸挡住"


def test_the_escalation_gate_actually_sits_before_the_downgrade():
    """闸必须在 `act = "PROBE_REANCHOR_EPOCH"` **之前**，否则等于没装。

    `_run_stage2_autonomous` 是个带真实执行副作用的大循环，仓库既有的办法是对它
    做源码/AST 检查（见 test_executor_never_declines_silently.py 等）—— 这里沿用。
    只钉两件事：闸调用存在，且位置在降级赋值之前。
    """
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(
        inspect.getsource(abfe_pipeline.ABFEPipeline._run_stage2_autonomous))
    tree = ast.parse(src)
    gate_lines = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "stale_layout_windows"
    ]
    downgrade_lines = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "act" for t in n.targets)
        and isinstance(n.value, ast.Constant)
        and n.value.value == "PROBE_REANCHOR_EPOCH"
    ]
    assert downgrade_lines, "找不到降级赋值 —— 测试假设过期了，先核对主循环"
    assert gate_lines, (
        "停滞保护的降级没有过 `stale_layout_windows` 闸：它绕过 `decide()`，"
        "会把重解类动作发给布局过期的窗口，真机直接 ValueError 炸穿主循环"
    )
    assert min(gate_lines) < min(downgrade_lines), (
        f"闸在降级之后（闸 @{min(gate_lines)}，降级 @{min(downgrade_lines)}）⟹ 等于没装"
    )



# ---------------------------------------------------------------------------
# 洞 3：插 λ 不得造出拆不开的末窗
# ---------------------------------------------------------------------------
def _run_where_window0_is_the_stuck_one(tmp_path, tail_k):
    """window 0 不可信 ⟹ `tail_repartition_anchor` 恒为 None（idx <= 0）。"""
    ranges = [(0, 8), (7, 13), (12, 16), (16, 16 + tail_k)]
    n_states = 16 + tail_k
    lam = [round(1.0 - 0.05 * i, 8) for i in range(n_states)]
    w = {}
    for i, (a, b) in enumerate(ranges):
        w[i] = {"K": b - a, "lambdas_vdw": lam[a:b]}
        if i == 0:
            # window 0 是那个卡住的窗口（支撑不足），其余合格。
            w[i]["self_verdict"] = "INSUFFICIENT_DATA"
    run = _mkrun(tmp_path, windows=w, ranges=ranges, n_states=n_states,
                 config={"stage2_window_min_states": 4,
                         "stage2_window_max_states": 8,
                         "max_path_insertions": 3})
    pv = pathlib.Path(run) / "checkpoints" / "path_versions" / "v1.json"
    d = json.loads(pv.read_text())
    d["states"] = [{"id": f"s{i}", "lambda_vdw": lam[i]} for i in range(n_states)]
    d["lambdas_vdw"] = lam
    pv.write_text(json.dumps(d))
    return run


def test_insert_is_infeasible_when_it_would_strand_the_tail(tmp_path):
    """末窗已经 K=8（= hi），再插一个就是 9 ⟹ 越过 hi 且拆不开 ⟹ 不可行。"""
    run = _run_where_window0_is_the_stuck_one(tmp_path, tail_k=8)
    ctl = Stage2RepairController.for_physical_stage(run, "vanishing", "vdw")
    view = ctl.read()
    assert ctl.tail_repartition_anchor(view) is None, (
        "前提：window 0 卡住时取不到 tail anchor"
    )
    feas = ctl.feasible(view=view, n_insert=1)
    assert feas.get("insert_lambda") is not None, (
        "插完末窗 K=9 > hi=8 且拆不开，这个动作必须判不可行"
    )
    assert "拆不开的非法布局" in feas["insert_lambda"]
    assert feas.get("split_tail_window") is not None, "拆末窗同样不可行"


def test_insert_stays_feasible_when_the_tail_has_room(tmp_path):
    """末窗 K=5，插一个到 6 ≤ hi=8 ⟹ 这道新闸不许误伤。"""
    run = _run_where_window0_is_the_stuck_one(tmp_path, tail_k=5)
    ctl = Stage2RepairController.for_physical_stage(run, "vanishing", "vdw")
    view = ctl.read()
    feas = ctl.feasible(view=view, n_insert=1)
    assert "拆不开的非法布局" not in (feas.get("insert_lambda") or ""), (
        f"末窗有余量却被判不可行：{feas.get('insert_lambda')}"
    )


# ---------------------------------------------------------------------------
# 配置里的 None 不是「给过」
# ---------------------------------------------------------------------------
def test_null_in_run_provenance_is_treated_as_absent(tmp_path):
    """`run_provenance.json` 的 config 是 argparse 命名空间落盘的，未给的键是 null。

    只滤调用方那半边不够：真机 13 个 run 的 provenance 每份都带 17~18 个 None 键。
    `int(None)` 在 09-15 已经炸过一次（cyclod_ligand1/rep1）。
    """
    run = _mkrun(
        tmp_path,
        windows={i: {"K": R4[i][1] - R4[i][0]} for i in range(4)},
        ranges=R4, n_states=13,
        config={"stage2_window_min_states": 4,
                "stage2_window_max_states": 8,
                "max_path_insertions": 3,
                # 这两个键存在但为 null ⟹ 必须退回读侧默认值，不许变成 int(None)
                "stage2_max_production_blocks_per_window": None,
                "n_steps_per_window": None},
    )
    ctl = Stage2RepairController.for_physical_stage(run, "vanishing", "vdw")
    assert ctl.max_blocks_per_window == 4
    assert ctl.production_block_steps == 250_000


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
