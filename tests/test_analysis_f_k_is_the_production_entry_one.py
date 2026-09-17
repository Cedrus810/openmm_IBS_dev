"""分析要用的 f_k 是**生成这批生产帧的那一份**，不是状态文件里活的那份。

真机 `cyclod_ligand1/rep3` win0：warmup 预算在上一跑就耗尽
（`no_budget_remaining_on_resume`），本次续跑没验成、判 `indeterminate`，
于是学习器活的 `f_k` 与 `production_entry_f_k` 差 0.03~0.28 kJ/mol ——
而生产帧明明是入口那一份驱动出来的。先前的断言比的是
`ibs_state["f_k"] == production_entry_f_k`，**用错了字段**，整条分析被挡死：

    ValueError: 窗口 0 f_k 与冻结生产入口不一致

两个字段语义不同：`production_entry_f_k` 生产开始时锁一次、生产期间从不再赋值；
`f_k` 是学习器活的状态，续跑再进一次 warmup 就会动。"""
import numpy as np
import pytest

from ibs_engine import _resolve_analysis_f_k

pytestmark = pytest.mark.cpu_only

ENTRY = [-39.3731, -22.3923, -8.9490, 1.8983, 10.0506, 16.1082, 20.5266, 22.1308]
LIVE = [-39.4601, -22.5711, -9.1299, 1.7660, 10.0152, 16.2088, 20.7580, 22.4130]


def _conv(frozen):
    return {"bias_warmup": {"frozen_f_k_at_last_freeze": frozen}}


def test_live_f_k_may_drift_and_analysis_uses_the_production_entry_one():
    got = _resolve_analysis_f_k(
        {"production_entry_f_k": ENTRY}, _conv(ENTRY),
        np.asarray(LIVE, dtype=np.float64), 0,
    )
    assert np.array_equal(got, np.asarray(ENTRY, dtype=np.float64))


def test_identical_values_are_returned_unchanged():
    live = np.asarray(ENTRY, dtype=np.float64)
    got = _resolve_analysis_f_k(
        {"production_entry_f_k": ENTRY}, _conv(ENTRY), live, 0)
    assert got is live


def test_two_independent_records_disagreeing_is_still_a_hard_refusal():
    """入口标记 vs convergence 里的冻结快照 —— 这两份对不上才是真矛盾。

    这道横向核对是**新加的**：原来只把一个字段跟同一个文件里的另一个字段比，
    等于没有第二个信源。
    """
    bad = list(ENTRY)
    bad[3] += 0.5
    with pytest.raises(ValueError, match="两份本该相同的记录矛盾"):
        _resolve_analysis_f_k(
            {"production_entry_f_k": ENTRY}, _conv(bad),
            np.asarray(LIVE, dtype=np.float64), 0)


@pytest.mark.parametrize("state", [
    {},                                   # 完全没有标记（旧产物 / degraded 路径）
    {"production_entry_f_k": ENTRY[:4]},  # 长度不符
    {"production_entry_f_k": [float("nan")] * 8},
])
def test_a_missing_or_unusable_marker_is_still_fail_closed(state):
    with pytest.raises(ValueError, match="生产入口标记"):
        _resolve_analysis_f_k(
            state, _conv(ENTRY), np.asarray(LIVE, dtype=np.float64), 0)
