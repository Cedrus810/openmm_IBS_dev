"""失败窗口插 λ 的契约（**model B**，2026-09-11 重写）。

设计依据：`docs/PLAN_PATH_REPAIR_2026-09-11.md` §2 更正与 §3ter。

两次真机事故塑造了这个契约，都要记住：

  1. **别把整条新路径丢给全局分窗器。** 21 态 [5,5,5,5,5] 插一个变
     22 态 [4,4,4,5,5,5] —— 窗口 0 从 5 态变 4 态，插在末尾却把最前面早就跑完的
     窗口改掉、产物全部失配白白重采。
  2. **别用"区间跟着插点 +1"的记账（model A）。** 那样失败窗口的 λ **跨度**不变、
     只是多一个内部点；而 `ΔF(λ_a→λ_b)` 是物理量，与中间放几个点无关 ⟹ bias
     要压平的总落差一分没少，对"窗口太宽"是**无效动作**。

model B 的记账：**窗口区间一律不动，只有末窗上界 += n。**失败窗口因此在底下
左移、λ 跨度真的缩小；多出来的态由末窗（溢出槽，豁免 max_states）吸收。
"""

import inspect

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

from abfe_preoptimizer import insert_lambda_in_failed_ibs_window as insert_lambda
from abfe_preoptimizer import feasible_repair_actions


PILOT = list(np.linspace(1.0, 0.0, 41))
# 单调递增的累计热力学长度；具体形状不重要，只要严格递增。
CUM = list(np.cumsum(np.linspace(0.4, 1.6, 41)))

RANGES_21 = [(0, 5), (4, 9), (8, 13), (12, 17), (16, 21)]


def _path(n_states):
    return list(np.linspace(1.0, 0.0, n_states))


def _prefix_unchanged(old_lam, old_ranges, new_lam, failed_start):
    """插入点**之前**的窗口装的必须还是逐位相同的 λ —— 那些可能已经采完了。

    只覆盖前缀。下游窗口的 λ 内容在 model B 下**本来就会左移一格**，那是这套
    记账的代价（顺序跑时下游还没采，零成本），不是 bug。
    """
    for a, b in old_ranges:
        if b <= failed_start + 1:
            if not np.allclose(
                np.asarray(old_lam)[a:b], np.asarray(new_lam)[a:b], atol=1e-12
            ):
                return False
    return True


def test_ranges_are_fixed_and_only_the_tail_absorbs_overflow():
    """model B 的核心不变量：区间表除末窗上界外逐字不变。"""
    old_lam = _path(21)
    new_lam, new_ranges, diag = insert_lambda(
        old_lam, RANGES_21, (0, 5), PILOT, CUM,
        min_states_per_window=4, max_states_per_window=6, n_insert=2,
    )
    assert len(new_lam) == 23
    assert [tuple(r) for r in new_ranges[:-1]] == RANGES_21[:-1]
    assert tuple(new_ranges[-1]) == (16, 23), "末窗吸收 2 个溢出态"
    assert diag["accounting"] == "model_B_fixed_ranges_tail_absorbs_overflow"
    # 已有的 λ 一个都不许动，只是多了新的
    assert set(np.round(old_lam, 12)) < set(np.round(new_lam, 12))


def test_prefix_windows_keep_their_lambdas_bit_for_bit():
    """已经采完的窗口必须逐位不变，否则产物全部失配白白重采。"""
    old_lam = _path(21)
    new_lam, _new_ranges, diag = insert_lambda(
        old_lam, RANGES_21, (12, 17), PILOT, CUM,
        min_states_per_window=4, max_states_per_window=5, n_insert=1,
    )
    assert _prefix_unchanged(old_lam, RANGES_21, new_lam, 12)
    assert diag["frozen_prefix_windows"] == [[0, 5], [4, 9], [8, 13]]


def test_the_failing_window_span_must_shrink_not_its_state_count():
    """介入的目的是把那个压不平的窗口变小 —— 变小指的是 **λ 跨度**。

    态数不变、跨度缩小 ⟹ 同一个 bias 要压平的总落差 `ΔF` 变小，这才治得了
    "窗口太宽"。model A 的"态数 +1、跨度不变"在这一项上恒等，治不了。
    """
    old_lam = _path(21)
    for failed in ((0, 5), (4, 9), (12, 17)):
        new_lam, new_ranges, diag = insert_lambda(
            old_lam, RANGES_21, failed, PILOT, CUM,
            min_states_per_window=4, max_states_per_window=5, n_insert=1,
        )
        a, b = failed
        # 态数不变
        assert (b - a) == (new_ranges[RANGES_21.index(failed)][1]
                           - new_ranges[RANGES_21.index(failed)][0])
        # 跨度严格变小
        span_before = abs(old_lam[b - 1] - old_lam[a])
        span_after = abs(new_lam[b - 1] - new_lam[a])
        assert span_after < span_before - 1e-12, (
            f"失败窗口 {failed} 的 λ 跨度没缩小：{span_before} → {span_after}"
        )
        assert diag["failed_window_span_after"] == pytest.approx(span_after)


def test_tail_window_is_the_overflow_slot_and_grows_instead():
    """失败窗口本身就是末窗时，它**不**缩跨度 —— 末窗是溢出槽，正常行为是变大。

    那种情况下该做的是拆末窗（另一条路径），不是继续指望插点缩它。
    """
    old_lam = _path(21)
    new_lam, new_ranges, diag = insert_lambda(
        old_lam, RANGES_21, RANGES_21[-1], PILOT, CUM,
        min_states_per_window=4, max_states_per_window=5, n_insert=2,
    )
    assert diag["failed_window_is_last"] is True
    assert tuple(new_ranges[-1]) == (16, 23)
    assert (new_ranges[-1][1] - new_ranges[-1][0]) == 7, "5 → 7 态"
    # 跨度不变（末窗两端 λ 没动）
    assert abs(new_lam[22] - new_lam[16]) == pytest.approx(
        abs(old_lam[20] - old_lam[16])
    )
    # 长到 7 态之后才可拆
    assert feasible_repair_actions(
        new_ranges, len(new_lam), min_states_per_window=4, max_states_per_window=5
    )["split_tail_window"] is None


def test_it_never_splits_a_window():
    """拆窗不再是本函数的职责：窗口总数永远不变。

    旧实现无条件就地拆窗，而且为了凑拆窗门槛 `n = max(1,(2*lo−1)−K)` 强行插点 ——
    那些 λ 没有任何边级证据支持，纯粹是成本。插点是**边级**工具，不该为窗口级
    簿记门槛服务。
    """
    for failed in ((0, 5), (8, 13), (16, 21)):
        _, new_ranges, diag = insert_lambda(
            _path(21), RANGES_21, failed, PILOT, CUM,
            min_states_per_window=4, max_states_per_window=5, n_insert=1,
        )
        assert len(new_ranges) == len(RANGES_21)
        assert diag["n_windows_before"] == diag["n_windows_after"]
        assert "child_windows" not in diag


def test_non_tail_windows_stay_within_min_max_and_tail_is_exempt():
    """非末窗必须留在 [min,max]；末窗作为溢出槽豁免上限。"""
    from abfe_preoptimizer import vanishing_subdomain_ranges_from_lambdas as partition

    for n, mn, mx in ((18, 4, 6), (21, 4, 5), (21, 4, 6), (17, 4, 5)):
        lam = _path(n)
        ranges = [tuple(r) for r in partition(
            np.asarray(lam), min_states_per_window=mn, max_states_per_window=mx
        )]
        new_lam, new_ranges, _ = insert_lambda(
            lam, ranges, ranges[0], PILOT, CUM,
            min_states_per_window=mn, max_states_per_window=mx, n_insert=2,
        )
        sizes = [b - a for a, b in new_ranges]
        assert all(mn <= s <= mx for s in sizes[:-1]), (
            f"n={n} min={mn} max={mx} 非末窗越界: {sizes}"
        )
        assert sizes[-1] == (ranges[-1][1] - ranges[-1][0]) + 2, (
            f"n={n} 末窗没吸收溢出: {sizes}"
        )
        assert _prefix_unchanged(lam, ranges, new_lam, ranges[0][0])


def test_refuses_to_insert_without_edge_or_caller_evidence():
    """不给 n_insert、边级判据又算出 0 时必须明着报错，不许静默插 1 个。"""
    # 构造：λ 等距 + pilot 累计长度线性 ⟹ 各边等长 ⟹ 没有"太长"的边
    lam = list(np.linspace(1.0, 0.0, 16))
    flat_cum = list(np.linspace(0.0, 1.0, 41))
    with pytest.raises(RuntimeError, match="边级证据不支持插点"):
        insert_lambda(
            lam, [(0, 4), (3, 8), (7, 12), (11, 16)], (0, 4), PILOT, flat_cum,
            min_states_per_window=4, max_states_per_window=5,
        )


def test_it_must_not_call_the_global_partitioner():
    """全局分窗器不知道前面的窗口已经采完；这个函数一旦调它就会重铺整条路径。"""
    src = inspect.getsource(insert_lambda)
    assert "vanishing_subdomain_ranges_from_lambdas" not in src
    assert "partition_windows_by_thermodynamic_length" not in src


def test_diagnostic_does_not_claim_the_edge_was_proven_faulty():
    """插点是**按 pilot 插值选的候选加密位置**，不是本次 VALIDATE 证实的故障边。

    日志把后者说成前者会误导：让人以为程序一边诊断右端、一边修左端。
    """
    _, _, diag = insert_lambda(
        _path(21), RANGES_21, RANGES_21[-1], PILOT, CUM,
        min_states_per_window=4, max_states_per_window=6, n_insert=1,
    )
    assert "pilot" in diag["source"]
    assert "非本次 VALIDATE 证实" in diag["note"]
    assert diag["n_insert_source"] == "caller"


def test_evolved_layout_is_accepted_on_structural_grounds():
    """保前缀的布局与全局分窗器重算的**必然不同**，那道门不能再要求逐字相等。

    真机踩过：`vanishing v12 只接受热力学坐标上的 few-state IBS 子区间：
    expected=[(0,5),(4,8),...], got=[(0,5),(4,9),...]` —— 结构完全合法
    （[5,5,5,4,4,5]，相邻只共享一个节点），只是跟默认分窗不一样就被拒。
    """
    import abfe_pipeline

    sig = inspect.signature(abfe_pipeline.ABFEPipeline._run_dual_lambda_stage)
    assert "authoritative_window_ranges" in sig.parameters
    assert sig.parameters["authoritative_window_ranges"].default is False, (
        "默认必须为 False —— 不声明权威时逐字保持原有严格判据"
    )
    src = inspect.getsource(abfe_pipeline.ABFEPipeline._run_dual_lambda_stage)
    branch = src[src.index("if authoritative_window_ranges and normalized_ranges:"):]
    branch = branch[:branch.index("else:")]
    assert "validate_single_shared_boundary_ranges(" in branch
    assert "含越界窗口" in branch

    evo = inspect.getsource(abfe_pipeline.ABFEPipeline._run_stage2_with_path_evolution)
    assert 'True if int(record["version"]) > 1 else None' in evo, (
        "演化过才声明权威；**未演化时传 None** 而不是 False —— False 会盖掉"
        "闭包里 'config 显式分窗即权威' 的兜底，让 stage2_window_ranges 首跑必被拒"
    )
