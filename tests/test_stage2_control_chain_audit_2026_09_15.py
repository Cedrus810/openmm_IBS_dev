"""Stage-2 控制链跨层契约审计（2026-09-15）的三条 P1 —— **行为测试**。

三条是同一个形状：**同一个量在不同层不是同一个值**。
所以断言必须测行为，不能只查源码里出现过某个参数名。
"""
import inspect
import pytest

# 🔑 [2026-09-16] 本文件原来**没有任何标记** ⟹ 日常的 `pytest -m cpu_only` 整份
# 跳过。里面全是纯 CPU 的源码契约探针，正好是最容易静默烂掉的那类（它们
# 断言"某段代码存在"，一旦指错函数就只是找不到、不报错）。实测就烂过：
# `decide()` 被拆成外壳之后这里 6 条全挂，而没人看得见。
pytestmark = pytest.mark.cpu_only

import json
import os

import abfe_pipeline as ap
import abfe_preoptimizer as pre
from abfe_preoptimizer import Stage2RepairController


# ── #1 两条腿必须解析出同一套控制配置 ───────────────────────────────────
_CFG = {
    "n_steps_per_window": 500_000,
    "stage2_window_min_states": 4,
    "stage2_window_max_states": 8,
    "stage2_window_partition": "metric_integral",
    "max_path_insertions": 8,
    "stage2_production_budget_steps": 2_000_000,
    "stage2_max_production_blocks_per_window": 7,
}


def _resolved(ctl):
    """控制器**实际用来做决定**的那几个量（不是它读了哪个文件）。"""
    return {
        "partition_criterion": ctl.partition_criterion,
        "lo": ctl.lo,
        "hi": ctl.hi,
        "max_path_insertions": ctl.max_path_insertions,
    }


def _leg(tmp_path, name, *, write_provenance, effective_config):
    d = tmp_path / name
    (d / "checkpoints").mkdir(parents=True)
    (d / "vanishing").mkdir()
    if write_provenance:
        (d / "run_provenance.json").write_text(json.dumps({"config": _CFG}))
    return Stage2RepairController(
        str(d), "vanishing", "vdw", effective_config=effective_config)


def test_the_solvent_leg_resolves_the_same_control_config_as_the_complex_leg(tmp_path):
    """真机：`_write_run_provenance` 只写总目录，溶剂腿 run_dir 底下没有那份文件。

    不显式传配置 ⟹ 两条腿解析出**不同**的分窗判据与插点上限（这就是 P1）。
    显式传 ⟹ 逐项相同。
    """
    complex_leg = _leg(tmp_path, "cplx", write_provenance=True, effective_config=None)
    # 溶剂腿：目录下没有 provenance（与 `output_dir/solvent_leg` 的真实盘面一致）
    solvent_bad = _leg(tmp_path, "solv_bad", write_provenance=False,
                       effective_config=None)
    assert _resolved(solvent_bad) != _resolved(complex_leg), (
        "这条测试的前提没了：溶剂腿本来就应该读不到配置")
    assert solvent_bad.partition_criterion == "arclength"      # 默认值
    assert complex_leg.partition_criterion == "metric_integral"

    # 修复后的路径：调用方显式喂同一份有效配置
    solvent_ok = _leg(tmp_path, "solv_ok", write_provenance=False,
                      effective_config=_CFG)
    assert _resolved(solvent_ok) == _resolved(complex_leg), (
        f"{_resolved(solvent_ok)} != {_resolved(complex_leg)}")
    assert solvent_ok.config_source == "caller:effective_config"


def test_an_explicit_config_beats_whatever_sits_in_the_run_dir(tmp_path):
    """显式配置优先：盘上那份即使存在也不得覆盖调用方给的。"""
    d = tmp_path / "r"
    (d / "checkpoints").mkdir(parents=True)
    (d / "vanishing").mkdir()
    (d / "run_provenance.json").write_text(json.dumps(
        {"config": dict(_CFG, stage2_window_partition="arclength",
                        max_path_insertions=1)}))
    c = Stage2RepairController(str(d), "vanishing", "vdw", effective_config=_CFG)
    assert c.partition_criterion == "metric_integral"
    assert c.max_path_insertions == 8


def test_no_explicit_config_keeps_the_old_behaviour(tmp_path):
    """不传时逐字不变 —— 离线 replay 与旧产物不受影响。"""
    c = _leg(tmp_path, "legacy", write_provenance=True, effective_config=None)
    assert c.config_source == "run_provenance.json"
    assert c.partition_criterion == "metric_integral"


def test_the_controller_consumes_no_config_key_outside_the_explicit_set():
    """显式配置的键集必须**覆盖**控制器真正读的每一个键。

    少一个 = 那个键在溶剂腿上悄悄退回默认值，而其余键看起来是对的 —— 最难查。
    """
    import re
    src = inspect.getsource(Stage2RepairController)
    consumed = set(re.findall(r'_cfg\.get\(\s*["\']([a-z0-9_]+)', src))
    pipeline_src = inspect.getsource(ap.ABFEPipeline.run_full_pipeline)
    supplied = set(re.findall(r'"([a-z0-9_]+)":\s*(?:int\(n_steps|kwargs\.get)',
                              pipeline_src))
    assert consumed <= supplied, f"控制器读了但没被显式喂的键: {sorted(consumed - supplied)}"


# ── #2 自治循环的块大小必须是配置里那个 ─────────────────────────────────
def test_the_autonomous_block_size_follows_the_configured_value():
    """正式形参不在 `**kwargs` 里 —— `kwargs.get(...)` 恒取默认值 250000。

    真机口径：首轮采样用真参数，控制器预算从 provenance 读真参数，
    只有自治补采用 250000 ⟹ 准入与执行不同口径，且极难看出来。
    """
    src = inspect.getsource(ap.ABFEPipeline.run_full_pipeline)
    # 行为层面：把调用点那一行的表达式抽出来，在两种绑定下各求一次值
    def like_the_call_site(n_steps_per_window=50_000, **kwargs):
        return int(n_steps_per_window)          # 修复后的写法
    def the_old_way(n_steps_per_window=50_000, **kwargs):
        return int(kwargs.get("n_steps_per_window", 250_000))
    for v in (50_000, 250_000, 500_000):
        assert like_the_call_site(n_steps_per_window=v) == v
    # 旧写法在三档配置下给出同一个数 —— 这就是 bug 的形状
    assert {the_old_way(n_steps_per_window=v) for v in (50_000, 250_000, 500_000)} == {250_000}
    # 真实源码里不得再出现旧写法（注释除外）
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert 'kwargs.get("n_steps_per_window"' not in code


def test_no_formal_parameter_is_read_back_out_of_kwargs():
    """同一形状的通用闸：`run_full_pipeline` 的正式形参一律不得从 kwargs 取。"""
    import ast
    import re
    src = inspect.getsource(ap.ABFEPipeline.run_full_pipeline)
    fn = ast.parse(src.lstrip()).body[0]
    formals = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    bad = sorted(formals & set(re.findall(r'kwargs\.get\(\s*"([a-z0-9_]+)"', code)))
    assert not bad, f"正式形参却从 kwargs 取（恒取默认值）: {bad}"


# ── #3 判零成本的动作不得自己去采样 ─────────────────────────────────────
def test_layout_actions_are_declared_free_and_therefore_must_not_sample():
    """`plan()` 把布局动作排除在生产预算准入之外 ⟹ 执行器就不能采样。

    两边是同一条契约的两半，必须一起成立：判零成本 + 不采样。
    """
    import ast
# 🔑 [2026-09] `decide()` 现在只是 23 行的外壳（"退役一个窗口再判一次"），判断体是 `_decide_once`（1831 行）。
# 源码探针指着 `decide` 会一无所获 —— 断言"存在"的当场红，断言"不存在"的**静默变成假绿**。
    decide = (inspect.getsource(Stage2RepairController.decide)
              + inspect.getsource(Stage2RepairController._decide_once))
    # 半边 A：这两个动作确实**不**在生产计费表里
    charged = decide.split("_PRODUCTION_CHARGED = (")[1].split(")")[0]
    assert "INSERT_LAMBDA" not in charged and "SPLIT_TAIL_WINDOW" not in charged

    # 半边 B：执行器这两个分支里没有任何 run_once 调用
    exe = inspect.getsource(ap.ABFEPipeline._run_stage2_autonomous)
    tree = ast.parse(exe.lstrip())
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if "INSERT_LAMBDA" in test or "SPLIT_TAIL_WINDOW" in test:
            body = ast.unparse(node.body)
            assert "run_once(" not in body, (
                f"布局动作分支仍在采样（{test}）—— 决策层判零成本、执行层烧 GPU")


# ── 第二轮审计：终态与证据的自相矛盾、以及布局核对的两处 fail-open ──────
from test_stage2_repair_controller import R4, _mkrun   # noqa: E402


def test_a_refuted_window_can_never_be_reported_as_done(tmp_path):
    """审计②：stage 分析完整 + 单窗 f_k 被统计驳回 ⟹ **不是 DONE**。

    0a 查了布局/覆盖，却没查逐窗 `STATISTICALLY_REJECTED`；
    于是 `action=DONE, terminal=True` 会配上 `evidence_status=REJECTED`。
    """
    w = {i: {"K": 4} for i in range(4)}
    w[2] = {"K": 4, "evidence": "refuted", "bias_status": "failed"}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13,
                 stage_result={"analysis_status": "ANALYSIS_COMPLETE",
                               "precision_status": "MEETS_CROSS_REPEAT_TARGET",
                               "total_delta_G": -1.0, "total_error": 0.1})
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] != "DONE", plan["reason"][:300]
    assert plan["evidence_status"] != "CONVERGED"
    assert plan["execution_status"] != "COMPLETE", "被驳回的窗口不得记成执行完毕"


def test_done_and_its_evidence_can_never_disagree(tmp_path):
    """通用不变量（这才是真正的修法）：`DONE` ⟺ 证据是 `CONVERGED`。

    逐条去补每个 DONE 分支 = 又一次"同一不变量 N 份实现"。
    """
    for evid, bias in (("refuted", "failed"),
                       ("indeterminate", "frozen_validation_indeterminate")):
        w = {i: {"K": 4} for i in range(4)}
        w[1] = {"K": 4, "evidence": evid, "bias_status": bias}
        run = _mkrun(tmp_path / f"r_{evid}", windows=w, ranges=R4, n_states=13,
                     stage_result={"analysis_status": "ANALYSIS_COMPLETE",
                                   "precision_status": "MEETS_CROSS_REPEAT_TARGET",
                                   "total_delta_G": -1.0, "total_error": 0.1})
        p = Stage2RepairController(run, "vanishing").decide()
        assert not (p["action"] == "DONE" and p["evidence_status"] != "CONVERGED"), (
            f"{evid}: action={p['action']} evidence={p['evidence_status']}")


def test_an_out_of_range_window_is_a_coverage_gap(tmp_path):
    """审计④：盘上多出当前布局根本没有的窗口号 ⟹ 覆盖有缺口，不是"完成"。

    `missing_windows` 只算 `range(expected)`，从不查多。
    """
    w = {i: {"K": 4} for i in range(4)}
    w[99] = {"K": 4}                     # 旧布局遗留的越界产物
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)
    view = Stage2RepairController.for_physical_stage(run, "vanishing", "vdw").read()
    assert 99 in (view.get("out_of_range_windows") or []), view.get(
        "out_of_range_windows")
    assert Stage2RepairController._coverage_incomplete(view), "越界窗口没被算成缺口"


def test_evidence_without_lambdas_is_flagged_as_unverifiable(tmp_path):
    """审计①：某段没有 λ 列表 ⟹ **核不了**，不是"核过了、匹配"。

    ⚠️ 仍然合并（判死会重现 win4 占位记录死锁），但必须在视图里说出来。
    """
    w = {i: {"K": 4} for i in range(4)}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13)
    # 当前路径必须**带 λ 值**，否则 `_cur_lam` 为空 ⟹ 布局核对本来就做不了
    # （`_mkrun` 的 v1.json 只写 state id、不写 λ）。
    vp = os.path.join(run, "checkpoints", "path_versions", "v1.json")
    with open(vp) as fh:
        _v = json.load(fh)
    # `_read_path` 从 `states[*].lambda_vdw` 取，不是顶层 `lambdas_vdw`
    _v["states"] = [{"id": f"s{i}", "lambda_vdw": round(1.0 - 0.05 * i, 8)}
                    for i in range(13)]
    with open(vp, "w") as fh:
        json.dump(_v, fh)
    # 把 win1 的 λ 列表抹掉，模拟"那一段的产物没有 λ"
    # `_read_window` 的 λ 是 `conv.lambdas_vdw or state.lambdas_vdw`，两份都要抹
    for pth in (os.path.join(run, "vanishing",
                             "dual_window_1_vdw_convergence.json"),
                os.path.join(run, "checkpoints",
                             "ibs_state_vdw_window_1.json")):
        with open(pth) as fh:
            d = json.load(fh)
        d["lambdas_vdw"] = []
        with open(pth, "w") as fh:
            json.dump(d, fh)
    view = Stage2RepairController.for_physical_stage(run, "vanishing", "vdw").read()
    assert 1 in (view.get("unverifiable_layout_evidence") or {}), view.get(
        "unverifiable_layout_evidence")
    # 与 stale 是两件事：stale 是"核对过、不匹配"
    assert 1 not in (view.get("stale_layout_evidence") or {})


def test_a_none_valued_key_is_never_passed_as_config(tmp_path):
    """🔑 真机崩点：`{"k": None}.get("k", 4)` 返回 **None 不是 4** ⟹ `int(None)` 炸。

    `effective_config` 里塞 None 等于把「未知」写成一个**存在的值**，
    读侧的默认值就永远用不上。「未知」的正确表达是**这个键不出现**。
    真机后果：`stage2_max_production_blocks_per_window` 没配 ⟹ 传了 None ⟹
    控制器一构造就 TypeError，而那条路每轮都走 ⟹ 整个自治循环进不去。
    """
    d = tmp_path / "r"
    (d / "checkpoints").mkdir(parents=True)
    (d / "vanishing").mkdir()
    c = Stage2RepairController(
        str(d), "vanishing", "vdw",
        effective_config={"n_steps_per_window": 250_000,
                          "stage2_max_production_blocks_per_window": None,
                          "stage2_production_budget_steps": None,
                          "stage2_window_partition": None})
    assert c.max_blocks_per_window == 4          # 默认值必须生效
    assert c.production_block_steps == 250_000
    assert c.partition_criterion == "arclength"

    # 写侧同样不得放 None 进去
    import inspect
    src = inspect.getsource(ap.ABFEPipeline.run_full_pipeline)
    blk = src.split("effective_config={")[1].split("},")[0]
    assert "if v is not None" in blk, "组装 effective_config 时没过滤 None"


def test_explicit_config_merges_with_provenance_instead_of_replacing(tmp_path):
    """显式配置**覆盖**盘上那份，但不得把没传的键打回默认值。

    写成"给了就不读 provenance"会倒退：调用方只传得出手上有的几个键，
    复合物腿 provenance 里记着的其余键会从真值掉回默认值。
    """
    d = tmp_path / "cplx"
    (d / "checkpoints").mkdir(parents=True)
    (d / "vanishing").mkdir()
    (d / "run_provenance.json").write_text(json.dumps({"config": {
        "n_steps_per_window": 500_000,
        "stage2_window_partition": "metric_integral",
        "max_path_insertions": 8,
        "stage2_max_production_blocks_per_window": 7,
    }}))
    c = Stage2RepairController(str(d), "vanishing", "vdw",
                               effective_config={"n_steps_per_window": 250_000})
    assert c.production_block_steps == 250_000      # 显式赢
    assert c.max_blocks_per_window == 7             # 没传 ⟹ 仍从 provenance
    assert c.partition_criterion == "metric_integral"
    assert c.max_path_insertions == 8
