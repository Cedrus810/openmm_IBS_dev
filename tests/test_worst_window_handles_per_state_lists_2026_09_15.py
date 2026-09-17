"""逐窗记录里一半的指标是**逐 λ 态的 list**，`_worst_window_by` 不能假定标量。

真机：11 个 run 里 **5 个** 的 `decide()` 直接 `TypeError` 炸穿 ——

    File "abfe_preoptimizer.py", line 2196, in _worst_window_by
        (float(r[key]), int(r["window_index"]))
    TypeError: float() argument must be a string or a real number, not 'list'

`top1pct_raw_weight` 在 `window_overlap_diagnostics` 里是 `list[K]`（逐目标态），
该记录里**根本没有**标量版本（受门的 `max_top1pct_raw_weight` 在 stage 顶层）。

⚠️ **这是预先就存在的 bug（HEAD 上同样崩）**，先前没暴露只是因为 stage 结果
从不落盘（`ANALYZE` 轮不到 ⟹ `window_overlap_diagnostics` 读不到）⟹ 这段走不到。
「退出前必跑一次全路径 ANALYZE」把它变成**每轮必经**。
"""
import pytest

from abfe_preoptimizer import _worst_window_by

pytestmark = pytest.mark.cpu_only


def _rec(i, v, key="top1pct_raw_weight"):
    return {"window_index": i, key: v}


def test_a_per_state_list_does_not_raise():
    recs = [_rec(0, [0.01, 0.02, 0.19]), _rec(1, [0.03, 0.04])]
    assert _worst_window_by(recs, "top1pct_raw_weight", largest=True) == 0


def test_the_reduction_direction_follows_the_selection():
    """`largest=True` 越大越差 ⟹ 窗口代表值取 `max`（与写侧
    `max_top1pct_raw_weight = max(逐态)` 同口径）；反向取 `min`。"""
    recs = [_rec(0, [0.90, 0.01]), _rec(1, [0.50, 0.50])]
    # 越大越差：win0 的最差态 0.90 > win1 的 0.50
    assert _worst_window_by(recs, "top1pct_raw_weight", largest=True) == 0
    # 越小越差：win0 的最差态 0.01 < win1 的 0.50
    assert _worst_window_by(recs, "top1pct_raw_weight", largest=False) == 0


def test_scalars_still_work_unchanged():
    recs = [{"window_index": 0, "raw_min_absolute_ess": 2.8},
            {"window_index": 1, "raw_min_absolute_ess": 40.0}]
    assert _worst_window_by(recs, "raw_min_absolute_ess") == 0
    assert _worst_window_by(recs, "raw_min_absolute_ess", largest=True) == 1


def test_unreadable_values_are_skipped_not_raised():
    """docstring 的约定：读不出来就**不猜** —— 那个窗口不进候选，绝不抛。"""
    recs = [_rec(0, {"a": 1}), _rec(1, "nope"), _rec(2, []),
            _rec(3, [float("nan"), float("inf")]), _rec(4, [0.7])]
    assert _worst_window_by(recs, "top1pct_raw_weight", largest=True) == 4


def test_no_candidate_at_all_returns_none():
    assert _worst_window_by([], "top1pct_raw_weight") is None
    assert _worst_window_by([_rec(0, None)], "top1pct_raw_weight") is None


def test_every_real_board_can_decide():
    """真机 11 个 run 的盘面全部能走完 `decide()` —— 改前 5 个崩。"""
    import glob
    import os

    from abfe_preoptimizer import Stage2RepairController as C
    runs = "/home/ruigengji/abfe-benchmark/openmm_IBS/runs"
    boards = [d for d in glob.glob(f"{runs}/*/*/checkpoints/stage2_autonomous_history.json")
              if "_trash" not in d]
    if not boards:
        pytest.skip("本机没有这批真机 run（只在有盘面时跑）")
    for d in boards:
        run = os.path.join(runs, "/".join(d.split("/")[-4:-2]))
        C.for_physical_stage(run, "vanishing", "vdw").decide()
