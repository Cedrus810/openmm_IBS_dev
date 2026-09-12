"""vanishing 路径演化闭环的离线契约测试（无 GPU、不建 OpenMM Context）。

被测：`ABFEPipeline._run_stage2_with_path_evolution`。
重点：默认策略下行为逐字不变；失败窗口**只插 λ 不拆窗**；每个窗口始终 >= 4 态。
"""

import json

import pytest

pytestmark = pytest.mark.cpu_only
pytest.importorskip("openmm")

import ibs_engine as ie
import lambda_path_versions as lpv
from abfe_pipeline import ABFEPipeline


LAMBDAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
# 真实 vanishing 窗口相邻**共享一个状态**（边界节点在两个 ensemble 里各出现一次
# 以对齐自由能参考），不是互不相交的半开区间。
RANGES = [(0, 4), (3, 7), (6, 11)]


def _q(values):
    return [round(float(v), 8) + 0.0 for v in values]


def _stub():
    obj = object.__new__(ABFEPipeline)
    obj._log = lambda *a, **k: None
    return obj


def _preopt(tmp_path):
    path = tmp_path / "preopt_dual_vanishing.json"
    n = 21
    path.write_text(json.dumps({"path_diagnostics": {
        "pilot_lambdas": [1.0 - i / (n - 1) for i in range(n)],
        "pilot_cumulative_thermodynamic_length": [float(i) for i in range(n)],
    }}), encoding="utf-8")
    return str(path)


def _run(tmp_path, run_once, policy="path_evolution_v1", max_insertions=3,
         lambdas=None, ranges=None):
    return _stub()._run_stage2_with_path_evolution(
        run_once,
        list(LAMBDAS if lambdas is None else lambdas),
        list(RANGES if ranges is None else ranges),
        checkpoint_dir=str(tmp_path), preopt_file=_preopt(tmp_path),
        repair_policy=policy, max_insertions=max_insertions,
    )


def _fail(global_state_range):
    return ie.IBSWarmupConvergenceError(
        "f_k 未收敛", {"global_state_range": list(global_state_range)}
    )


def test_default_policy_is_byte_for_byte_the_old_behaviour(tmp_path):
    """不开路径演化时：跑一次、原样返回、**一个版本文件都不写**。"""
    calls = []
    def run_once(n, lam, rng, **kw):
        calls.append((n, list(lam), [tuple(r) for r in rng]))
        return {"ok": True}

    result, lam, rng = _run(tmp_path, run_once, policy="non_mutating_v1")
    assert result == {"ok": True}
    assert len(calls) == 1 and calls[0][1] == LAMBDAS and calls[0][2] == RANGES
    assert lam == LAMBDAS and [tuple(r) for r in rng] == RANGES
    assert lpv.load_current(str(tmp_path)) is None, "默认策略不该开始记账"


def test_success_records_v1_and_runs_once(tmp_path):
    calls = []
    result, lam, _ = _run(tmp_path, lambda n, l, r, **kw: calls.append(1) or {"ok": True})
    assert result == {"ok": True} and len(calls) == 1
    assert lpv.load_current(str(tmp_path))["version"] == 1 and lam == LAMBDAS


def test_failed_window_gets_one_lambda_inserted_not_split(tmp_path):
    """min=4/max=5 下拆窗不可能（两个 >=4 的子窗需母窗 >=7）——只插 λ。"""
    seen = []
    def run_once(n, lam, rng, **kw):
        seen.append((list(lam), [tuple(r) for r in rng]))
        if len(seen) == 1:
            raise _fail((3, 7))
        return {"ok": True}

    result, lam, rng = _run(tmp_path, run_once)
    assert result == {"ok": True} and len(seen) == 2
    # 插点数按"尾段能否多划出一个窗口"定：目的是把压不平的窗口变**小**，
    # 只加一个态而不多划窗口只会让它更大。所以可能插 1 个也可能插 2 个。
    assert len(LAMBDAS) < len(lam) <= len(LAMBDAS) + 3
    assert set(_q(LAMBDAS)) < set(_q(lam)), "已有 λ 一个都不能动"

    inserted = (set(_q(lam)) - set(_q(LAMBDAS))).pop()
    assert LAMBDAS[6] < inserted < LAMBDAS[3], "插点必须落在失败窗口内部"

    record = lpv.load_current(str(tmp_path))
    assert record["version"] == 2 and record["event"]["kind"] == "insert_lambda"
    assert record["event"]["detail"]["failed_global_state_range"] == [3, 7]


def test_every_window_always_keeps_at_least_four_states(tmp_path):
    """窗口两端各有一个共享边界态，内部只剩 K-2 个自由态，K<4 压不平。"""
    def run_once(n, lam, rng, **kw):
        for start, end in rng:
            assert end - start >= 4, f"窗口 {(start, end)} 少于 4 态，f_k 压不齐"
        raise _fail(tuple(rng[1]))

    with pytest.raises(ie.IBSWarmupConvergenceError):
        _run(tmp_path, run_once, max_insertions=3)


def test_prefix_windows_are_untouched_by_a_late_insertion(tmp_path):
    """插点之前的窗口 λ 内容必须逐个不变——它们的采样才能继续复用。"""
    seen = []
    def run_once(n, lam, rng, **kw):
        seen.append((list(lam), [tuple(r) for r in rng]))
        if len(seen) == 1:
            raise _fail((6, 11))
        return {"ok": True}

    _run(tmp_path, run_once)
    before_l, before_r = seen[0]
    after_l, after_r = seen[1]
    first_before = before_r[0]
    first_after = after_r[0]
    assert (_q(before_l[first_before[0]:first_before[1]])
            == _q(after_l[first_after[0]:first_after[1]])), "首窗 λ 内容不该变"


def test_exhausting_the_budget_reraises_and_keeps_progress(tmp_path):
    def run_once(n, lam, rng, **kw):
        raise _fail(tuple(rng[1]))

    with pytest.raises(ie.IBSWarmupConvergenceError):
        _run(tmp_path, run_once, max_insertions=2)
    # 撞上限时路径与进度必须保留，供加预算后从这里继续。
    assert lpv.load_current(str(tmp_path))["version"] == 3


def test_undiagnosable_failure_is_reraised_without_touching_the_path(tmp_path):
    """定位不到失败窗口就不猜插哪里 —— fail-closed。"""
    def run_once(n, lam, rng, **kw):
        raise ie.IBSWarmupConvergenceError("没有诊断", {})

    with pytest.raises(ie.IBSWarmupConvergenceError):
        _run(tmp_path, run_once)
    assert lpv.load_current(str(tmp_path))["version"] == 1, "不得产生新版本"


def test_an_already_evolved_path_is_adopted_not_rewound(tmp_path):
    """resume：已经演化到 v2 时，run_once 必须拿到 v2 的路径，而不是传入的 v1。"""
    evolved = sorted(LAMBDAS + [0.65], reverse=True)
    evolved_ranges = [(0, 4), (3, 8), (7, 12)]
    lpv.init_version(str(tmp_path), [0.0] * len(LAMBDAS), LAMBDAS, RANGES)
    lpv.append_version(
        str(tmp_path), [0.0] * len(evolved), evolved, evolved_ranges,
        kind="insert_lambda", reason="test",
        detail={"failed_global_state_range": [3, 7]},
    )
    seen = []
    _run(tmp_path, lambda n, l, r, **kw: seen.append((list(l), [tuple(x) for x in r]))
         or {"ok": True})
    assert seen[0][0] == _q(evolved), "必须用 v2 的 λ 表，不能被预优化打回 v1"
    assert seen[0][1] == evolved_ranges
    assert lpv.load_current(str(tmp_path))["version"] == 2


def test_unknown_policy_fails_closed():
    for bad in ("typo_v9", ""):
        with pytest.raises(ValueError, match="unknown sampling repair_policy"):
            ie.should_run_path_evolution(bad)
        with pytest.raises(ValueError, match="unknown sampling repair_policy"):
            ie.repair_policy_cache_class(bad)


def test_path_evolution_and_non_mutating_share_a_window_cache_class():
    """插点时不该因为策略名变了就把前面所有已完成窗口判废。"""
    assert (ie.repair_policy_cache_class("path_evolution_v1")
            == ie.repair_policy_cache_class("non_mutating_v1"))
    assert (ie.repair_policy_cache_class("legacy_mutating")
            != ie.repair_policy_cache_class("non_mutating_v1"))
    assert ie.should_run_legacy_repair("path_evolution_v1") is False


# ---------------------------------------------------------------------------
# CLI 透传：只在显式设置时才进 kwargs（否则会平白改动所有现有指纹）
# ---------------------------------------------------------------------------

def test_path_evolution_switches_are_absent_unless_configured():
    from runabfe import _path_evolution_kwargs

    class _Cfg(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    assert _path_evolution_kwargs(_Cfg()) == {}, (
        "没配置时一个键都不能加——run_config 是逐字段进协议指纹的"
    )
    assert _path_evolution_kwargs(_Cfg(sampling_repair_policy="path_evolution_v1")) == {
        "sampling_repair_policy": "path_evolution_v1"
    }
    assert _path_evolution_kwargs(
        _Cfg(sampling_repair_policy="path_evolution_v1", max_path_insertions=5)
    ) == {"sampling_repair_policy": "path_evolution_v1", "max_path_insertions": 5}


# ---------------------------------------------------------------------------
# CLI 透传：只在显式设置时才进 kwargs（否则会平白改动所有现有指纹）
# ---------------------------------------------------------------------------

class _Cfg(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


def test_switches_are_absent_unless_configured():
    """没配置时一个键都不能加 —— run_config 是逐字段进 stage 协议指纹的，
    多一个键（哪怕值是 None）就会让所有现有缓存失配、白重跑 GPU。"""
    from runabfe import _path_evolution_kwargs

    assert _path_evolution_kwargs(_Cfg()) == {}
    assert _path_evolution_kwargs(_Cfg(sampling_repair_policy=None,
                                       max_path_insertions=None)) == {}


def test_switches_are_passed_through_when_configured():
    from runabfe import _path_evolution_kwargs

    assert _path_evolution_kwargs(
        _Cfg(sampling_repair_policy="path_evolution_v1")
    ) == {"sampling_repair_policy": "path_evolution_v1"}
    assert _path_evolution_kwargs(
        _Cfg(sampling_repair_policy="path_evolution_v1", max_path_insertions=5)
    ) == {"sampling_repair_policy": "path_evolution_v1", "max_path_insertions": 5}


def test_both_legs_pass_the_switches_through():
    """两条腿（complex / solvent）都得透传，否则只有一条腿开了演化。"""
    import inspect
    import runabfe

    src = inspect.getsource(runabfe)
    assert src.count("**_path_evolution_kwargs(config),") == 2, (
        "run_full_pipeline 的两个调用点都必须透传路径演化开关"
    )
