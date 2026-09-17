# -*- coding: utf-8 -*-
"""末窗溢出槽的两道门（2026-09-14 真机）。

末窗是插 λ 的**溢出槽**，它身上有两个不同的上限，被混用过一次：

  · **可拆上限 ``2*hi−1``** —— path-version 层允许它涨到这里。再多一个就永远
    切不出两个落在 [lo,hi] 的子窗（两子窗共享边界态 ⟹ p+q−1=K）。
  · **执行层上限 ``hi``** —— stage 的权威 ``window_ranges`` 校验对**所有**窗口
    一律要求 [lo,hi]，跑之前必须合法化。

A. 插点侧（``insert_lambda_in_failed_ibs_window``）以可拆上限 fail-closed。
B. 合法化侧（``ABFEPipeline._legalize_tail_window``）以执行层上限判早退。

B 曾被写成 ``if tail_k <= 2*hi-1: return`` —— 与它自己的 docstring 矛盾，于是
K ∈ (hi, 2*hi−1] 的末窗被原样放行，带着一个超上限的窗口去跑，死在下游权威校验。
真机五个 run 里四个末窗超 hi（K=7/10/11/12），**每一个都还在自己配置的可拆
区间内** —— A 从头守住了，放跑非法布局的只有 B。
"""
import os

import pytest

import abfe_preoptimizer as pre
from abfe_pipeline import ABFEPipeline

pytestmark = pytest.mark.cpu_only


def _pilot(lambdas):
    """一条度量均匀的 pilot：弧长 = 序号，够 `_pilot_arclength_of` 插中点用。"""
    lam = [float(x) for x in lambdas]
    return lam, [float(i) for i in range(len(lam))]


# --------------------------------------------------------------------------
# A. 插点侧：末窗不许被顶出可拆区间
# --------------------------------------------------------------------------
def test_insertion_refuses_to_push_tail_past_split_ceiling():
    lo, hi = 4, 5
    # 末窗已经在可拆上限 2*hi−1 = 9 上；再插一个就永远拆不开。
    lam = [1.0 - i / 15.0 for i in range(16)]
    ranges = [(0, 4), (3, 8), (7, 16)]
    assert ranges[-1][1] - ranges[-1][0] == 2 * hi - 1

    with pytest.raises(RuntimeError, match="可拆上限"):
        pre.insert_lambda_in_failed_ibs_window(
            list(lam), list(ranges), (0, 4), *_pilot(lam),
            min_states_per_window=lo, max_states_per_window=hi, n_insert=1,
        )


def test_insertion_allowed_while_tail_stays_inside_split_range():
    """守卫不是"末窗一超 hi 就拒" —— 溢出槽的存在意义就是能超到 2*hi−1。"""
    lo, hi = 4, 5
    lam = [1.0 - i / 14.0 for i in range(15)]
    ranges = [(0, 4), (3, 8), (7, 15)]     # 末窗 K=8 < 9
    new_lam, new_ranges, _d = pre.insert_lambda_in_failed_ibs_window(
        list(lam), list(ranges), (0, 4), *_pilot(lam),
        min_states_per_window=lo, max_states_per_window=hi, n_insert=1,
    )
    assert new_ranges[-1][1] - new_ranges[-1][0] == 2 * hi - 1
    assert len(new_lam) == len(lam) + 1


# --------------------------------------------------------------------------
# B. 合法化侧：K ∈ (hi, 2*hi−1] 必须被拆，不许早退
# --------------------------------------------------------------------------
@pytest.mark.parametrize("lo,hi,tail_k", [(4, 5, 7), (4, 8, 10), (4, 8, 12)])
def test_legalize_splits_tail_between_hi_and_split_ceiling(tmp_path, lo, hi, tail_k):
    """真机四个超标末窗的形状，逐个过一遍。"""
    head = [(0, lo)]
    n = lo + tail_k - 1
    ranges = [(0, lo), (lo - 1, n)]
    assert ranges[-1][1] - ranges[-1][0] == tail_k
    assert hi < tail_k <= 2 * hi - 1, "本用例就是要落在这段被放行过的区间里"
    lam = [1.0 - i / (n - 1.0) for i in range(n)]

    import lambda_path_versions as lpv
    lpv.init_version(str(tmp_path), [0.0] * n, lam, [list(r) for r in ranges])

    pipe = object.__new__(ABFEPipeline)
    pipe._log = lambda *a, **k: None
    out_lam, out_ranges = pipe._legalize_tail_window(
        lam, ranges,
        failed_range=None, pilot=None, checkpoint_dir=str(tmp_path),
        min_states_per_window=lo, max_states_per_window=hi,
        # [审计 #29 / #12，2026-09-14] `anchor_getter` 已废弃并从签名摘除：
        # anchor 现在由 `_legalize_tail_window` 用**插完之后**的 lam/ranges 按
        # `first_untrusted` 的下标现解（先前从闭包里拿的是**插之前**那份 view，
        # model B 下 λ 数值移位 ⟹ anchor 值落到非窗口起点 ⟹ ValueError 炸穿）。
        first_untrusted=1,
    )

    # 每个窗口都回到执行层合法区间 —— 这才是本函数的契约。
    assert all(lo <= b - a <= hi for a, b in out_ranges), out_ranges
    # 尾段重分不动 λ 表。
    assert out_lam == [float(x) for x in lam]
    # 前缀逐字冻结。
    assert out_ranges[0] == head[0]
    # 落盘登记过，崩溃恢复才分得清"已经重分过没有"。
    cur = lpv.load_current(str(tmp_path))
    assert cur["event"]["kind"] == "tail_repartition"
    assert [tuple(r) for r in cur["window_ranges"]] == out_ranges


@pytest.mark.parametrize("first_untrusted", [0, None, 1])
def test_legalize_does_not_need_to_know_which_window_is_untrusted(
        tmp_path, first_untrusted):
    """真机 cyclod_ligand3/rep1+rep3：K=9 / lo=4 / hi=8，`first_untrusted=0` ⟹ 起不来。

    **这不是死区**：源码定义的死区是 `hi < K < 2*lo−1`，lo=4/hi=8 时是 `8 < K < 7`
    = 空集；K=9 落在可拆区间 `[2lo−1, 2hi−1] = [7, 15]` 里，`9 → 5+5` 两个都在
    `[4, 8]`，**数学上完全拆得开**。

    卡住它的是概念串线：「**合法化**」（末窗超 hi 就必须做，与谁不可信无关）
    借用了「**修复**」的定位谓词 `first_untrusted_window`，而后者在 `idx <= 0`
    时恒为 None ⟹ 抛 RuntimeError，且那个非法布局**已经落盘**。

    ⟹ 合法化必须对 `first_untrusted ∈ {0, None, 任意}` 一律成立。
    """
    lo, hi = 4, 8
    ranges = [(0, 4), (3, 7), (6, 15)]        # 末窗 K=9
    n = 15
    assert ranges[-1][1] - ranges[-1][0] == 9
    assert not (hi < 9 < 2 * lo - 1), "K=9 不在死区（死区是空集）"
    assert 2 * lo - 1 <= 9 <= 2 * hi - 1, "K=9 在可拆区间里"
    lam = [1.0 - i / (n - 1.0) for i in range(n)]

    import lambda_path_versions as lpv
    lpv.init_version(str(tmp_path), [0.0] * n, lam, [list(r) for r in ranges])

    pipe = object.__new__(ABFEPipeline)
    pipe._log = lambda *a, **k: None
    out_lam, out_ranges = pipe._legalize_tail_window(
        lam, ranges,
        failed_range=None, pilot=None, checkpoint_dir=str(tmp_path),
        min_states_per_window=lo, max_states_per_window=hi,
        first_untrusted=first_untrusted,
    )

    assert all(lo <= b - a <= hi for a, b in out_ranges), out_ranges
    assert out_lam == [float(x) for x in lam], "尾段重分不动 λ 表"
    # 前缀逐字冻结 —— anchor 是末窗首态 ⟹ 重分范围恰好是末窗那一段。
    assert out_ranges[:2] == [(0, 4), (3, 7)], out_ranges
    # 相邻窗口共享 λ 节点，自由能链不断。
    for (a1, b1), (a2, _b2) in zip(out_ranges, out_ranges[1:]):
        assert a2 == b1 - 1, (out_ranges, "接缝上没有共享节点")
    # 覆盖完整。
    assert out_ranges[0][0] == 0 and out_ranges[-1][1] == n
    cur = lpv.load_current(str(tmp_path))
    assert cur["event"]["kind"] == "tail_repartition"
    assert [tuple(r) for r in cur["window_ranges"]] == out_ranges


def test_legalize_refuses_a_single_window_layout(tmp_path):
    """只有一个窗口时没有「末窗首态」这个共享节点 ⟹ fail-closed，不许静默重分全路径。

    重分整条路径是**布点**的事，不是合法化的事。
    """
    lo, hi = 4, 8
    lam = [1.0 - i / 8.0 for i in range(9)]
    pipe = object.__new__(ABFEPipeline)
    pipe._log = lambda *a, **k: None
    with pytest.raises(RuntimeError, match="只有 1 个窗口"):
        pipe._legalize_tail_window(
            lam, [(0, 9)],
            failed_range=None, pilot=None, checkpoint_dir=str(tmp_path),
            min_states_per_window=lo, max_states_per_window=hi,
            first_untrusted=None,
        )


def test_legalize_is_a_noop_when_tail_already_legal(tmp_path):
    lo, hi = 4, 5
    lam = [1.0 - i / 7.0 for i in range(8)]
    ranges = [(0, 4), (3, 8)]
    pipe = object.__new__(ABFEPipeline)
    pipe._log = lambda *a, **k: None

    out_lam, out_ranges = pipe._legalize_tail_window(
        lam, ranges,
        failed_range=None, pilot=None, checkpoint_dir=str(tmp_path),
        min_states_per_window=lo, max_states_per_window=hi,
        # `first_untrusted=None` ⟹ 取不到 anchor。早退发生在解 anchor **之前**，
        # 所以契约不变：合法布局不该走到重分那一步（由下面「不写版本链」钉住）。
        first_untrusted=None,
    )
    assert out_ranges == ranges
    assert out_lam == [float(x) for x in lam]
    # 早退不该往版本链写任何东西。
    assert not os.path.exists(os.path.join(str(tmp_path), "path_current.json"))
