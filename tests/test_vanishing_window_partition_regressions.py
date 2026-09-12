from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

import pytest



pytestmark = pytest.mark.cpu_only

ROOT = Path(__file__).resolve().parents[1]
PREOPT_PATH = ROOT / "abfe_preoptimizer.py"


def _load_partition_function():
    tree = ast.parse(PREOPT_PATH.read_text(encoding="utf-8"), filename=str(PREOPT_PATH))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_greedy_vanishing_window_ranges"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"List": List, "Tuple": Tuple}
    exec(compile(module, str(PREOPT_PATH), "exec"), namespace)
    return namespace["_greedy_vanishing_window_ranges"]


def _assert_partition_invariants(ranges, n_states, min_states, max_states):
    assert ranges[0][0] == 0
    assert ranges[-1][1] == n_states
    assert all(min_states <= end - start <= max_states for start, end in ranges)
    assert all(left[1] - 1 == right[0] for left, right in zip(ranges, ranges[1:]))
    assert all(
        right[0] >= left[1]
        for i, left in enumerate(ranges)
        for right in ranges[i + 2 :]
    )
    assert {state for start, end in ranges for state in range(start, end)} == set(
        range(n_states)
    )


def test_infeasible_exact_six_state_windows_are_rejected():
    partition = _load_partition_function()
    with pytest.raises(ValueError, match="不存在满足"):
        partition(7, 6, 6)


@pytest.mark.parametrize(
    "n_states,min_states,max_states",
    [(2, 2, 2), (12, 4, 6), (23, 4, 6), (31, 3, 7)],
)
def test_feasible_partitions_satisfy_every_window_invariant(
    n_states, min_states, max_states
):
    partition = _load_partition_function()
    ranges = partition(n_states, min_states, max_states)
    _assert_partition_invariants(ranges, n_states, min_states, max_states)
    sizes = [end - start for start, end in ranges]
    assert sizes == sorted(sizes)



def _load_literal_comparison_guard():
    """取出 run_full_pipeline 里那道"跳过逐字相等比对"的 if 条件。

    条件本身是纯逻辑（只依赖 _explicit_ranges / _partition_criterion 两个名字），
    但它长在一个五千行的方法里没法直接调用，只能按本文件既有做法从源码里挖。
    """
    pipeline_path = ROOT / "abfe_pipeline.py"
    tree = ast.parse(pipeline_path.read_text(encoding="utf-8"), filename=str(pipeline_path))
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and len(node.body) == 1
        and any(
            isinstance(stmt, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "normalized_vanishing_ranges"
                for t in stmt.targets
            )
            for stmt in node.body
        )
        and not node.orelse
    ]
    assert len(guards) == 1, f"期望正好一处跳过比对的 if，实际 {len(guards)} 处"
    expr = ast.Expression(body=guards[0].test)
    ast.fix_missing_locations(expr)
    code = compile(expr, str(pipeline_path), "eval")

    def guard(explicit_ranges, criterion):
        return bool(eval(code, {}, {
            "_explicit_ranges": explicit_ranges,
            "_partition_criterion": criterion,
        }))

    return guard


def test_literal_range_comparison_only_applies_to_the_arclength_partitioner():
    # window_ranges_2 永远是 preopt 的等弧长分窗；只有当前分窗权威也是等弧长时，
    # "expected 必须逐字等于 got"才成立。metric_integral 上 8/8 作业死在这里。
    guard = _load_literal_comparison_guard()
    assert guard(None, "metric_integral") is True
    assert guard([(0, 8), (7, 13)], "arclength") is True
    # 默认的等弧长路径必须仍然被逐字比对（它才是这道门原本要守的东西）。
    assert guard(None, "arclength") is False


def _insert():
    from abfe_preoptimizer import insert_lambda_in_failed_ibs_window
    return insert_lambda_in_failed_ibs_window


def _linear_pilot():
    import numpy as np
    return list(np.linspace(1, 0, 101)), list(np.linspace(0, 10, 101))


def test_insertion_must_actually_shrink_the_failed_window():
    """插点之后失败窗口必须真的变小 —— 变小指的是 **λ 跨度**。

    两次踩过的坑：
      · 旧旧版按"理论最少窗口数"推插点数，于是"2→3"其实是"3→3"，重划后失败
        窗口从 5 态变成 **8** 态、λ 跨度一点没缩。
      · 旧版（model A）把窗口区间跟着插点 +1：态数 +1、**跨度不变**。而
        ``ΔF(λ_a→λ_b)`` 是物理量，与中间放几个点无关 ⟹ bias 要压平的总落差
        一分没少，对"窗口太宽"是无效动作。

    model B：区间不动、末窗吸收溢出 ⟹ 失败窗口在底下左移、跨度严格变小。
    """
    lambdas = [1 - i / 20 for i in range(21)]
    ranges = [(0, 5), (4, 9), (8, 13), (12, 17), (16, 21)]
    failed = (8, 13)
    pilot_lam, pilot_s = _linear_pilot()
    span_before = abs(lambdas[failed[1] - 1] - lambdas[failed[0]])

    new_l, new_r, diag = _insert()(
        lambdas, ranges, failed, pilot_lam, pilot_s,
        min_states_per_window=4, max_states_per_window=8, n_insert=2,
    )

    # 不拆窗：窗口数不变
    assert diag["n_windows_after"] == diag["n_windows_before"] == len(ranges)
    # 失败窗口的跨度严格变小
    span_after = abs(new_l[failed[1] - 1] - new_l[failed[0]])
    assert span_after < span_before - 1e-12, (span_before, span_after)
    assert diag["failed_window_span_after"] == pytest.approx(span_after)
    # 前缀（已采完的窗口）逐字冻结
    assert [tuple(r) for r in diag["frozen_prefix_windows"]] == [(0, 5), (4, 9)]
    assert new_l[:failed[0] + 1] == lambdas[:failed[0] + 1]
    assert len(diag["inserted_global_edges"]) == diag["n_inserted"]
    # 溢出只落末窗
    assert [tuple(r) for r in new_r[:-1]] == ranges[:-1]
    assert tuple(new_r[-1]) == (16, 23)


def test_rescue_keeps_the_configured_partition_criterion():
    """初始按 ∫g 分窗，补救不能换回等弧长。"""
    import numpy as np
    pilot_lam = np.linspace(1, 0, 201)
    metric_g = 1.0 + 60 * np.exp(-((pilot_lam - 0.62) / 0.05) ** 2)
    pilot_s = np.concatenate(
        [[0.0], np.cumsum(np.abs(np.diff(pilot_lam)) * np.sqrt(metric_g[:-1]))]
    )
    lambdas = list(np.interp(np.linspace(0, pilot_s[-1], 21), pilot_s, pilot_lam))
    ranges = [(0, 8), (7, 13), (12, 16), (15, 21)]
    # 判据决定的是**切点**，所以窗口必须大到有多个合法切点才看得出差别：
    # 8 态窗口插 1 个 → 9 态，切成 (4,6)/(5,5)/(6,4) 三种都合法。
    # 5 态窗口插 2 个 → 7 态只有 4+4 一种切法，两个判据必然同解，那不算"没接上"。
    # [model B] 判据不再影响窗口区间（区间一律不动），它影响的是**选哪条边、
    # 插在哪里**：度规尖峰上的边在 ∫g 下最长、在弧长下不是。
    common = dict(min_states_per_window=4, max_states_per_window=8, n_insert=1)
    _, arc_r, arc_diag = _insert()(
        lambdas, ranges, (0, 8), list(pilot_lam), list(pilot_s),
        partition_criterion="arclength", **common
    )
    _, mi_r, mi_diag = _insert()(
        lambdas, ranges, (0, 8), list(pilot_lam), list(pilot_s),
        partition_criterion="metric_integral", pilot_metric_g=list(metric_g), **common
    )
    assert mi_diag["partition_criterion"] == "metric_integral"
    assert arc_diag["partition_criterion"] == "arclength"
    # 区间相同（model B 的不变量），但插入的位置必须不同 —— 否则判据又成了空参数
    assert [tuple(r) for r in arc_r] == [tuple(r) for r in mi_r]
    assert arc_diag["inserted_global_edges"] != mi_diag["inserted_global_edges"] or (
        arc_diag["inserted_lambdas"] != mi_diag["inserted_lambdas"]
    ), "partition_criterion 必须真的改变选边/定位，不能是无效参数"

    # 缺 metric_g 时 fail-closed，不静默退回等弧长
    with pytest.raises(ValueError, match="metric_integral"):
        _insert()(
            lambdas, ranges, (0, 8), list(pilot_lam), list(pilot_s),
            partition_criterion="metric_integral", **common
        )


def test_insertion_freezes_the_prefix_and_only_the_tail_grows():
    """**前缀逐位冻结、溢出只落末窗。**

    曾经这里是"前缀冻结 + 尾段整体重划"：失败的是靠前的窗口时前缀为空，"尾段"
    就是整条路径，一次补救把后面每个窗口的边界全改了、产物全部失配。
    model B 下窗口区间一律不动，只有末窗上界 += n。

    ⚠️ 下游窗口装的 λ **会**左移一格 —— 那是 model B 的代价，不是 bug：窗口按
    0,1,2… 顺序跑，win_i 失败时 i+1..N 还没采，所以顺序跑时这是零成本；而
    resume 安全性由 lambda_path_versions（λ 表 + window_ranges 都落盘）保证。
    """
    from abfe_preoptimizer import insert_lambda_in_failed_ibs_window

    lambdas = [1 - i / 20 for i in range(21)]
    ranges = [(0, 7), (6, 10), (9, 13), (12, 16), (15, 21)]
    pilot_lam, pilot_s = _linear_pilot()

    for failed in [(0, 7), (6, 10), (15, 21)]:
        new_l, new_r, diag = insert_lambda_in_failed_ibs_window(
            lambdas, ranges, failed, pilot_lam, pilot_s,
            min_states_per_window=4, max_states_per_window=8, n_insert=1,
        )
        # 窗口数不变、区间除末窗外逐字不变
        assert len(new_r) == len(ranges), (failed, new_r)
        assert [tuple(r) for r in new_r[:-1]] == ranges[:-1], (failed, new_r)
        assert tuple(new_r[-1]) == (15, 22), (failed, new_r)
        # 前缀（插入点之前）的 λ 逐位不变
        for a0, b0 in ranges:
            if b0 <= failed[0] + 1:
                assert [round(x, 12) for x in lambdas[a0:b0]] == [
                    round(x, 12) for x in new_l[a0:b0]
                ], (failed, (a0, b0))
        # 非末窗态数一个都没变
        assert [b - a for a, b in new_r[:-1]] == [b - a for a, b in ranges[:-1]]
