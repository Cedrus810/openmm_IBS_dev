"""非末窗与末窗是**两个上界**，而不是一个。

背景：`stage2_window_max_states` 先前同时表达两件事 —— 非末窗的可跑上限，和
末窗的溢出容量（model B 插 λ 时非末窗 ranges 逐字冻结、多出来的态一律落到末窗）。
共用一个数 ⟹ 要给末窗留空间只能把它抬高 ⟹ **每个非末窗跟着一起被抬上去**。
真机 45 个 run 的布局里 `[4,8,8,4]` / `[4,7,7,5,4]` / `[4,7,7,7,4]` 就是这么来的。

`stage2_last_window_max_states` 的上界是**推出来的，不是拍的**：末窗拆成两个共享
一个边界态的子窗时 `p+q−1=K`，两个都要 ≤ hi ⟹ 可拆的最大 K 是 `2*hi−1`。
「长到 last_hi、超了就拆」于是要求 `last_hi + 1 ≤ 2*hi − 1`，即 `last_hi ≤ 2*hi−2`。
lo=4 / hi=5 ⟹ last_hi ≤ 8，而 9 正好拆成 5+5。
"""
import json
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import abfe_pipeline as pipe  # noqa: E402
from abfe_preoptimizer import (  # noqa: E402
    partition_windows_by_metric_integral as partition,
)

_PIL = [1.0 - i / 60 for i in range(61)]
# g 故意很不均匀：`metric_integral` 会为了均衡 ∫g 去造大窗，这正是问题盘面
_G_SKEWED = [1.0 + (30.0 if i > 45 else 0.2) for i in range(61)]


def _layout(n, hi, last_hi, objective="metric_integral", g=None):
    lam = [1.0 - i / (n - 1) for i in range(n)]
    ranges, _ = partition(
        lam, _PIL, g or _G_SKEWED,
        min_states_per_window=4, max_states_per_window=hi,
        objective=objective, first_window_max_states=4,
        last_window_max_states=last_hi)
    return [b - a for a, b in ranges]


@pytest.mark.parametrize("n", [21, 23, 25])
def test_one_shared_cap_inflates_every_non_tail_window(n):
    """基线：共用一个 hi=8 时，**非末窗**确实被抬到 6/7/8（真机形状）。"""
    K = _layout(n, hi=8, last_hi=None)
    assert any(k > 5 for k in K[:-1]), (
        f"n={n} 的基线没复现出大非末窗（{K}），这条对照失去意义"
    )


@pytest.mark.parametrize("n", [21, 22, 23, 24, 25, 26, 27, 30])
def test_non_tail_windows_obey_the_non_tail_cap(n):
    """拆成两个上界之后：**非末窗**一律落在 [4, 5]。"""
    K = _layout(n, hi=5, last_hi=8)
    assert all(4 <= k <= 5 for k in K[:-1]), f"n={n} 非末窗越界：{K}"


@pytest.mark.parametrize("n", [21, 23, 25])
def test_the_tail_may_exceed_the_non_tail_cap(n):
    """末窗允许到 last_hi —— 否则它当不成溢出槽。"""
    K = _layout(n, hi=5, last_hi=8)
    assert 4 <= K[-1] <= 8, K


def test_last_cap_defaults_to_the_non_tail_cap():
    """不给这个键 ⟹ 逐位保持旧行为（末窗同样吃 hi）。"""
    assert _layout(23, hi=5, last_hi=None) == _layout(23, hi=5, last_hi=5)


def test_a_tail_cap_that_cannot_be_split_is_refused():
    """`last_hi > 2*hi−2` ⟹ 拆窗时必有子窗 > hi ⟹ 这种布局合法化不了，必须拒。

    lo=4 / hi=5 ⟹ 上限 8。给 9 的话，末窗长到 10 才拆，而 10 = p+q−1 只能拆成
    5+6 / 6+5，两个都不合法。
    """
    with pytest.raises(ValueError, match="2\\*max_states_per_window"):
        _layout(23, hi=5, last_hi=9)


def test_the_derived_bound_is_exactly_two_hi_minus_two():
    """边界值本身必须**被接受**（8 合法、9 不合法），否则 8 这个数就是拍的。"""
    _layout(23, hi=5, last_hi=8)                 # 不抛即通过
    with pytest.raises(ValueError):
        _layout(23, hi=5, last_hi=9)


def test_the_tail_cap_is_part_of_the_layout_identity():
    """它决定布局 ⟹ 必须进派生路径身份，否则改了不作废旧窗口缓存。"""
    assert "stage2_last_window_max_states" in pipe._PREOPT_DERIVED_PATH_KEYS


def test_the_shipped_config_asks_for_four_to_five_non_tail():
    """仓库配置就是用户要的那组：非末窗 4–5、末窗到 8。"""
    cfg = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "abfe_config.json"), encoding="utf-8"))
    assert cfg["stage2_window_min_states"] == 4
    assert cfg["stage2_window_max_states"] == 5
    assert cfg["stage2_last_window_max_states"] == 8
    assert (cfg["stage2_last_window_max_states"]
            <= 2 * cfg["stage2_window_max_states"] - 2), "配置自己就拆不开"
