"""按 ∫g dλ 均衡分窗的契约测试（独立函数，尚未接入生产）。

背景：等**弧长**布点均衡的是相邻态的重叠，而 IBS 是一条轨迹重加权到窗口内全部
K 个态，难度更接近窗口内的总方差 ∫g dλ。实测 4W53 cyclod 21 态五窗弧长大致相等
（2.28~3.29），∫g 却是 19/37/67/97/54 差 5 倍，失败的正是 ∫g 最大那个窗。
"""

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

from abfe_preoptimizer import (
    metric_integral_cumulative,
    partition_windows_by_metric_integral as partition,
)


# 尖峰度规：模仿真实 vanishing 路径（λ≈0.3 处 g 暴涨、λ→0 塌陷）
PILOT = list(np.linspace(1.0, 0.0, 41))
_pl = np.asarray(PILOT)
METRIC = list(40.0 + 1400.0 * np.exp(-((_pl - 0.30) / 0.07) ** 2) * (_pl > 0.12))
LAM = list(np.linspace(1.0, 0.0, 21))


def _sizes(ranges):
    return [b - a for a, b in ranges]


def test_relaxing_max_reduces_both_peak_and_imbalance():
    """卡住均衡的是 max 不是 min：它逼着便宜的地方也只能用小窗。"""
    _, tight = partition(LAM, PILOT, METRIC, min_states_per_window=4,
                         max_states_per_window=5, n_windows=5)
    _, loose = partition(LAM, PILOT, METRIC, min_states_per_window=4,
                         max_states_per_window=8, n_windows=5)
    assert loose["peak_metric_integral"] < tight["peak_metric_integral"]
    assert loose["imbalance_max_over_min"] < tight["imbalance_max_over_min"]
    # 便宜的头部应该被并进一个大窗
    assert max(loose["sizes"]) > max(tight["sizes"])


def test_layout_is_structurally_valid():
    for mx in (5, 8, 12):
        ranges, diag = partition(LAM, PILOT, METRIC,
                                 min_states_per_window=4, max_states_per_window=mx)
        assert all(4 <= b - a <= mx for a, b in ranges)
        # 相邻窗恰好共享一个边界态，且完整覆盖
        for (a1, b1), (a2, b2) in zip(ranges, ranges[1:]):
            assert a2 == b1 - 1, f"相邻窗必须恰好共享一个态：{ranges}"
        assert ranges[0][0] == 0 and ranges[-1][1] == len(LAM)
        assert diag["sizes"] == _sizes(ranges)


def test_result_is_deterministic():
    a = partition(LAM, PILOT, METRIC, min_states_per_window=4, max_states_per_window=8)
    b = partition(LAM, PILOT, METRIC, min_states_per_window=4, max_states_per_window=8)
    assert a[0] == b[0] and a[1]["sizes"] == b[1]["sizes"]


def test_peak_has_a_floor_set_by_the_smallest_window_at_the_spike():
    """峰值降不到一个**最小窗**在尖峰处的 ∫g 以下 —— 再低只能加 λ 态。"""
    gcum = metric_integral_cumulative(LAM, PILOT, METRIC)
    floor = min(
        abs(gcum[i + 3] - gcum[i]) for i in range(len(LAM) - 3)
    )
    for mx in (8, 12, 16):
        _, diag = partition(LAM, PILOT, METRIC,
                            min_states_per_window=4, max_states_per_window=mx)
        assert diag["peak_metric_integral"] >= floor - 1e-9


def test_windows_cover_the_whole_metric_integral():
    ranges, diag = partition(LAM, PILOT, METRIC,
                             min_states_per_window=4, max_states_per_window=8)
    gcum = metric_integral_cumulative(LAM, PILOT, METRIC)
    assert abs(sum(diag["metric_integral_per_window"])
               - abs(gcum[-1] - gcum[0])) < 1e-9


def test_metric_integral_is_not_the_same_as_arc_length():
    """同样弧长的窗口，落在尖峰上的那个 ∫g 大得多 —— 这正是判据要换的理由。"""
    gcum = metric_integral_cumulative(LAM, PILOT, METRIC)
    arc = np.concatenate([[0.0], np.cumsum(
        np.sqrt(np.interp(np.asarray(LAM), _pl[::-1], np.asarray(METRIC)[::-1]))[:-1]
        * np.abs(np.diff(np.asarray(LAM))))])
    head = (abs(gcum[4] - gcum[0]), abs(arc[4] - arc[0]))
    peak_i = int(np.argmin(np.abs(np.asarray(LAM) - 0.30)))
    peak_i = min(peak_i, len(LAM) - 5)
    spike = (abs(gcum[peak_i + 4] - gcum[peak_i]), abs(arc[peak_i + 4] - arc[peak_i]))
    assert spike[0] / head[0] > spike[1] / head[1] * 2, (
        f"∫g 的对比度应当远大于弧长：∫g {head[0]:.1f}->{spike[0]:.1f}，"
        f"弧长 {head[1]:.2f}->{spike[1]:.2f}"
    )


def test_infeasible_and_invalid_inputs_fail_closed():
    with pytest.raises(ValueError):
        partition(LAM, PILOT, METRIC, min_states_per_window=1, max_states_per_window=5)
    with pytest.raises(ValueError):
        partition(LAM, PILOT, METRIC, min_states_per_window=6, max_states_per_window=5)
    with pytest.raises(RuntimeError):
        partition(LAM, PILOT, METRIC, min_states_per_window=4,
                  max_states_per_window=5, n_windows=99)
    with pytest.raises(ValueError):
        metric_integral_cumulative(LAM, PILOT, [float("nan")] * len(PILOT))
    with pytest.raises(ValueError):
        metric_integral_cumulative(LAM, PILOT, [-1.0] * len(PILOT))


def test_it_is_wired_behind_a_switch_that_defaults_to_arclength():
    """接进生产，但默认仍是等弧长；只有显式设 stage2_window_partition 才切换。"""
    import ast
    import inspect

    import abfe_pipeline

    src = inspect.getsource(abfe_pipeline)
    tree = ast.parse(src)
    called = {
        ast.unparse(n.func).split(".")[-1]
        for n in ast.walk(tree) if isinstance(n, ast.Call)
    }
    assert "partition_windows_by_metric_integral" in called, "应当已接入"
    assert 'kwargs.get("stage2_window_partition", "arclength")' in src, (
        "默认必须是 arclength —— 不设开关时行为逐字不变"
    )
    # 缺 pilot 数据时必须 fail-closed，不能静默退回等弧长（那会悄悄改变布局）。
    assert "拒绝退回等弧长分窗" in src


def test_non_default_criterion_is_declared_authoritative():
    """非默认判据产出的布局与贪心等弧长不同，不声明权威就会被那道门拒掉。"""
    import inspect

    import abfe_pipeline

    src = inspect.getsource(abfe_pipeline.ABFEPipeline.run_full_pipeline)
    assert '!= "arclength"' in src
    # `_partition_criterion` 就是 stage2_window_partition 归一化后的那个值，
    # 解析已上移到 Stage 2 结果缓存检查之前（见那里的注释）。
    assert '_partition_criterion = str(' in src
    assert 'kwargs.get("stage2_window_partition", "arclength")' in src
    idx = src.index("_stage2_path_evolved = bool(")
    window = src[idx:idx + 400]
    assert "stage2_window_ranges" in window
    assert '_partition_criterion != "arclength"' in window
    # 路径演化过（前缀被冻结）同样与全局分窗器不同，也必须声明权威。
    assert '_path_record["version"]' in window


def test_partition_switches_pass_through_only_when_set():
    from runabfe import _path_evolution_kwargs

    class _Cfg(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    assert _path_evolution_kwargs(_Cfg()) == {}
    assert _path_evolution_kwargs(
        _Cfg(stage2_window_partition="metric_integral", stage2_n_windows=5)
    ) == {"stage2_window_partition": "metric_integral", "stage2_n_windows": 5}



# ---------------------------------------------------------------------------
# config 显式分窗的口子
# ---------------------------------------------------------------------------

def test_explicit_window_ranges_pass_through_only_when_set():
    from runabfe import _path_evolution_kwargs

    class _Cfg(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    assert _path_evolution_kwargs(_Cfg()) == {}
    assert _path_evolution_kwargs(_Cfg(stage2_window_ranges=[])) == {}, (
        "空列表当没设，避免把 run_config 平白改脏"
    )
    assert _path_evolution_kwargs(
        _Cfg(stage2_window_ranges=[[0, 8], [7, 12]])
    ) == {"stage2_window_ranges": [[0, 8], [7, 12]]}


def test_explicit_ranges_are_validated_not_trusted():
    """显式不等于免检：结构、覆盖、[min,max] 一条都不放过。"""
    import inspect

    import abfe_pipeline

    src = inspect.getsource(abfe_pipeline.ABFEPipeline.run_full_pipeline)
    block = src[src.index('_explicit_ranges = kwargs.get("stage2_window_ranges")'):]
    block = block[:block.index("else:")]
    assert "validate_single_shared_boundary_ranges(" in block
    assert "含越界窗口" in block
    # 用了显式分窗就必须声明权威，否则 _run_dual_lambda_stage 那道门会拒。
    idx = src.index("_stage2_path_evolved = bool(")
    assert "stage2_window_ranges" in src[idx:idx + 400]


def test_the_metric_partitioner_can_produce_the_value_to_put_in_config():
    """离线算出布局 → 填进 config 的闭环：产出必须是可直接写进 JSON 的形状。"""
    ranges, diag = partition(LAM, PILOT, METRIC,
                             min_states_per_window=4, max_states_per_window=8)
    as_config = [[int(a), int(b)] for a, b in ranges]
    assert all(isinstance(x, int) for pair in as_config for x in pair)
    assert as_config[0][0] == 0 and as_config[-1][1] == len(LAM)
    assert diag["sizes"] == [b - a for a, b in as_config]


def test_explicit_ranges_are_authoritative_from_the_very_first_run():
    """首跑时路径记录还是 v1，`version > 1` 是 False —— 不能用它去覆盖
    "config 显式分窗即权威"的兜底，否则 stage2_window_ranges 第一次跑就必被拒。

    真机踩过：planner 接受了 [8,5,4,4,4]，`_run_dual_lambda_stage` 却报
    `expected=[(0,7),(6,14),(13,21)], got=[(0,8),(7,12),...]`。
    """
    import inspect

    import abfe_pipeline

    src = inspect.getsource(
        abfe_pipeline.ABFEPipeline._run_stage2_with_path_evolution
    )
    assert "_authoritative_window_ranges=(" in src
    assert 'True if int(record["version"]) > 1 else None' in src, (
        "未演化时必须传 None 让闭包兜底生效，不能传 False 盖掉它"
    )


def test_explicit_ranges_pin_the_layout_and_disable_insertion():
    """显式布局的语义是"就要测这个"；插点会改变态数、让绝对下标失效。

    两者同时配置时钉住布局、关掉演化，并且明说 —— 不静默改掉使用者指定的实验。
    """
    import inspect

    import abfe_pipeline

    sig = inspect.signature(
        abfe_pipeline.ABFEPipeline._run_stage2_with_path_evolution
    )
    assert sig.parameters["explicit_window_ranges_pinned"].default is False
    src = inspect.getsource(
        abfe_pipeline.ABFEPipeline._run_stage2_with_path_evolution
    )
    body = src[src.index("if explicit_window_ranges_pinned:"):]
    # [2026-09-14] 判据从"字面出现 `return run_once(`"改成语义判据：
    # 这条分支必须**直接返回一次普通运行**、不进插点循环。实际调用现在包了一层
    # `_guarded_once()`（接住 LOCAL_VALIDATION_CAP 这个路由信号），原来的字面
    # 匹配会把这层包装误判成"改坏了"。
    _head = body[:1200]
    assert "return _guarded_once(), current_l, current_r" in _head or (
        "return run_once(" in _head
    ), "钉住时必须直接跑一次并返回"
    assert "insert_lambda_in_failed_ibs_window" not in _head, (
        "钉住时不许进插点循环"
    )
    assert "不做" in body[:1200] and "stage2_window_ranges" in body[:1200], (
        "必须打日志说明演化被关掉了"
    )
    caller = inspect.getsource(abfe_pipeline.ABFEPipeline.run_full_pipeline)
    assert 'explicit_window_ranges_pinned=bool(' in caller


def test_path_evolution_never_clears_the_config_based_authority():
    """路径演化只是**又一个**权威来源，不能覆盖基于配置的判断。

    真机踩过两次同形状：用 `=` 覆盖之后，路径没演化（v1）时标志被打回 False，
    于是显式分窗在 stage2 主调用通过、到生产补采时被那道门拒掉：
    `expected=[(0,7),(6,14),(13,21)], got=[(0,8),(7,12),(11,15),(14,18),(17,21)]`。
    """
    import ast
    import inspect
    import textwrap

    import abfe_pipeline

    src = inspect.getsource(abfe_pipeline.ABFEPipeline.run_full_pipeline)
    tree = ast.parse(textwrap.dedent(src))
    # 收集所有对 _stage2_path_evolved 的赋值；除第一处初始化外，
    # 其余都必须是「或上去」而不是覆盖。
    assigns = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "_stage2_path_evolved" for t in n.targets)
    ]
    assert len(assigns) >= 2
    later = sorted(assigns, key=lambda n: n.lineno)[1:]
    for node in later:
        rendered = ast.unparse(node.value)
        assert "_stage2_path_evolved" in rendered, (
            f"后续赋值必须保留原值（或上去），不能覆盖：{rendered}"
        )


def test_max_window_states_is_the_third_dp_objective():
    """🔑 [2026-09-15] 峰值 ∫g 判据在构造上**看不见 K**。

    真机 cyclod_ligand1（21 态、真实 pilot）：`[8,6,4,6]` 与 `[7,7,4,6]`
    峰值 ∫g **都是 85.5**、窗口数**都是 4**，先前 DP 只按平方和打平局
    （20252.7 < 20770.1）⟹ 交付 8 态首窗，预测 ΔF 跨度 74 kJ/mol，
    而 win0 事后没有任何布局修复路径。

    现在排序是 峰值 → 窗数 → maxK → 平方和：同样的峰值与窗数下取更小的 maxK。
    """
    import json
    import numpy as np

    from abfe_preoptimizer import (
        metric_integral_cumulative,
        partition_windows_by_metric_integral as part,
    )

    # 合成一条"耦合端便宜、中段陡"的度规：不依赖任何 run 目录。
    pilot = list(np.linspace(0.0, 1.0, 41))
    g = [1.0 + 40.0 * np.exp(-((x - 0.72) ** 2) / (2 * 0.05 ** 2)) for x in pilot]
    lam = list(np.linspace(1.0, 0.0, 21))

    ranges, diag = part(lam, pilot_lambdas=pilot, metric_g=g,
                        min_states_per_window=4, max_states_per_window=8)
    sizes = diag["sizes"]
    assert diag["max_window_states"] == max(sizes), diag
    peak = diag["peak_metric_integral"]

    # 关键性质：在**同样的窗口数**下，不存在峰值同样是 peak 而 maxK 更小的布局。
    gc = metric_integral_cumulative(np.asarray(lam, float), pilot, g)
    cost = lambda a, b: abs(float(gc[b - 1] - gc[a]))
    import itertools
    n, w = len(lam), len(ranges)

    def layouts(start, k):
        if k == 1:
            if n - start >= 4 and n - start <= 8:
                yield [(start, n)]
            return
        for size in range(4, 9):
            j = start + size - 1
            if j >= n:
                break
            for rest in layouts(j, k - 1):
                yield [(start, j + 1)] + rest

    better = [L for L in layouts(0, w)
              if max(cost(a, b) for a, b in L) <= peak * (1 + 1e-9)
              and max(b - a for a, b in L) < diag["max_window_states"]]
    assert not better, f"存在同峰值同窗数但 maxK 更小的布局，DP 没取到：{better[:2]}"


def test_window_count_still_outranks_max_window_states():
    """窗数排在 maxK **之前** —— 否则分窗器会用 GPU 去买 K。"""
    import inspect

    from abfe_preoptimizer import partition_windows_by_metric_integral as part

    src = inspect.getsource(part)
    assert "candidates.append((got[0], w, got[1], got[2], got[3]))" in src
    # 排序键顺序：peak(got[0]) → w → maxK(got[1]) → ssq(got[2])
    assert "peak, w_used, max_k, ssq, ranges = min(candidates)" in src
