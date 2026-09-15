"""判据与执行器背离的四条回归钉子（2026-09-15）。

四条 bug 是**同一个形状**：同一个不变量两份实现。逐条钉住修好之后的行为：

  A. 自治循环内的 f_k 探针必须**拿得到重锚节奏**（不传 = 把固定节奏重锚整个关掉）
  B. "探过一次"必须是「在**这个盘面**上探过」——否则补了帧也永不重探
  C. `SPLIT_TAIL_WINDOW` 的方向：判据要更细，执行器不许交付更宽
  D. 尾段重分必须用与生产**同一个**分窗判据
  E. 耦合端（win0）态数上限：∫g 均衡看不见 K
"""

import inspect

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

import abfe_preoptimizer as pre
from abfe_preoptimizer import (
    partition_windows_by_metric_integral as partition,
    redistribute_lambda_by_thermodynamic_length as redistribute,
    repartition_tail_from_anchor,
)

# 真实 repeat01 pilot（与 tests/test_core_physics_numerics.py 同一份落盘数值）
PILOT = [1.0, 0.9375, 0.875, 0.8125, 0.75, 0.6875, 0.625, 0.5625, 0.5,
         0.4375, 0.375, 0.3125, 0.25, 0.1875, 0.125, 0.0625, 0.0]
METRIC = [52.58438779527305, 70.23988604189759, 122.49496978382494,
          175.6242419205146, 211.15683247168505, 299.0521852276533,
          536.8973853468257, 645.6925015190681, 904.8956435886821,
          1032.7735031949487, 1484.5461340323218, 2315.094092957457,
          1429.9658848218626, 722.2923268411564, 145.30966292343663,
          8.921124106945888, 0.004647161528018394]
LAM22 = [float(x) for x in redistribute(
    np.asarray(PILOT), np.asarray(METRIC), 22)[0]]


# --------------------------------------------------------------- C：方向
def test_tail_repartition_never_delivers_fewer_windows_than_it_promised():
    """用户给的纸面复现：判据说"可以拆细"，执行器先前交付"合并了"。

    [(0,4),(3,8),(7,12),(11,16)]、anchor=lam[3]、lo/hi=4/8 ⟹ 尾段 3 个窗、13 个态。
    旧执行器调 `vanishing_subdomain_ranges_from_lambdas` 且不传窗口数，那个分窗器
    内部 `n_windows = min_windows = ceil(12/7) = 2` ⟹ [(3,10),(9,16)]，
    三个 5 态窗变成两个 7 态窗 —— 结构上完全合法，方向完全相反。
    """
    lam = list(np.linspace(1.0, 0.0, 16))
    ranges = [(0, 4), (3, 8), (7, 12), (11, 16)]
    # 不给目标窗口数 = 旧行为：分窗器取最少窗口数，尾段被**合并**。
    old, _ = repartition_tail_from_anchor(
        lam, ranges, lam[3], min_states_per_window=4, max_states_per_window=8,
    )
    tail_old = [r for r in old if r[0] >= 3]
    assert len(tail_old) == 2 and [b - a for a, b in tail_old] == [7, 7], (
        f"旧路径的行为变了，这条钉子的前提没了：{old}"
    )
    # 显式要求拆细 ⟹ 必须真的更细，否则 fail-closed。
    new, diag = repartition_tail_from_anchor(
        lam, ranges, lam[3], min_states_per_window=4, max_states_per_window=8,
        n_windows=4,
    )
    tail_new = [r for r in new if r[0] >= 3]
    assert len(tail_new) == 4 > len(tail_old)
    assert diag["tail_windows_after"] > diag["tail_windows_before"]
    assert diag["tail_max_window_span_after"] < diag["tail_max_window_span_before"]


def test_a_repartition_that_would_widen_is_refused_not_recorded_as_success():
    """窗口数够不着"更细"时必须抛，不许把一次扩大跨度记成成功的拆窗。"""
    lam = list(np.linspace(1.0, 0.0, 16))
    ranges = [(0, 4), (3, 8), (7, 12), (11, 16)]
    with pytest.raises(RuntimeError, match="切不出|没有把窗口切细"):
        # 尾段 13 态最多切 (13-1)/(4-1) = 4 个窗；要 5 个切不出来。
        repartition_tail_from_anchor(
            lam, ranges, lam[3],
            min_states_per_window=4, max_states_per_window=8, n_windows=5,
        )


def test_legalization_path_may_keep_the_window_count():
    """另一个调用点（末窗 K > hi 的合法化）的目标是**变合法**、不是变细。

    lo/hi=4/8、尾段 [9,4] ⟹ [7,6] 是一次完全正确的合法化，窗口数没变。
    "必须更细"那两条断言不许套在它头上。
    """
    lam = list(np.linspace(1.0, 0.0, 16))
    ranges = [(0, 4), (3, 12), (11, 16)]   # 尾段两个窗，第一个 K=9 > hi
    new, diag = repartition_tail_from_anchor(
        lam, ranges, lam[3], min_states_per_window=4, max_states_per_window=8,
    )
    tail = [r for r in new if r[0] >= 3]
    assert all(4 <= b - a <= 8 for a, b in tail), tail
    assert diag["tail_windows_after"] == diag["tail_windows_before"] == 2


# --------------------------------------------------------------- D：判据
def test_tail_repartition_uses_the_production_partition_criterion():
    """生产按 ∫g 分窗时，尾段不许改用等状态数贪心。

    ⚠️ 这里不传 `n_windows`（不要求更细）——本例只钉"判据"这一维；
    方向那一维由上面 C 的两条钉。
    """
    ranges = [(0, 8), (7, 12), (11, 15), (14, 18), (17, 22)]
    by_metric, diag = repartition_tail_from_anchor(
        LAM22, ranges, LAM22[7],
        min_states_per_window=4, max_states_per_window=8,
        pilot_lambdas=PILOT, metric_g=METRIC,
        partition_criterion="metric_integral",
    )
    assert diag["partition_criterion"] == "metric_integral"
    assert diag["tail_span_unit"] == "∫g dλ"
    by_count, _ = repartition_tail_from_anchor(
        LAM22, ranges, LAM22[7],
        min_states_per_window=4, max_states_per_window=8,
    )
    # 两套判据在真实 pilot 上给出不同布局——这正是"换判据"会静默发生的事。
    assert by_metric != by_count, (by_metric, by_count)


def test_split_tail_is_structurally_infeasible_on_this_path_and_says_so():
    """真机 22 态布局：尾段根本容不下多一个窗口（lo=4 ⟹ 需要 16 个态、只有 15）。

    先前这条路会**静默交付一个更宽的布局**；现在如实抛，让可行性判据看得见。
    """
    ranges = [(0, 8), (7, 12), (11, 15), (14, 18), (17, 22)]
    with pytest.raises(RuntimeError, match="切不出"):
        repartition_tail_from_anchor(
            LAM22, ranges, LAM22[7],
            min_states_per_window=4, max_states_per_window=8,
            n_windows=5, pilot_lambdas=PILOT, metric_g=METRIC,
            partition_criterion="metric_integral",
        )


def test_metric_integral_without_pilot_data_fails_closed():
    ranges = [(0, 8), (7, 12), (11, 15), (14, 18), (17, 22)]
    with pytest.raises(RuntimeError, match="metric_integral 需要"):
        repartition_tail_from_anchor(
            LAM22, ranges, LAM22[7],
            min_states_per_window=4, max_states_per_window=8,
            n_windows=5, partition_criterion="metric_integral",
        )


# --------------------------------------------------------------- E：win0 上限
def test_first_window_cap_is_applied_inside_the_dp_not_after():
    """真机 4W53 vanishing 的布局：不传新参数逐字不变，传了才收窄 win0。"""
    base, _ = partition(LAM22, PILOT, METRIC,
                        min_states_per_window=4, max_states_per_window=8)
    assert [tuple(int(i) for i in r) for r in base] == [
        (0, 8), (7, 12), (11, 15), (14, 18), (17, 22)
    ], "现状布局变了——这条钉子比对的是真机日志里的那一份"

    capped, diag = partition(LAM22, PILOT, METRIC,
                             min_states_per_window=4, max_states_per_window=8,
                             first_window_max_states=4)
    assert diag["sizes"][0] == 4
    assert diag["first_window_max_states"] == 4
    # 约束**进 DP**：先解再过滤拿不到"该约束下的最优"。这里 maxK 从 8 降到 6，
    # 且代价只有 +1 个窗口。
    assert len(diag["sizes"]) == len(base) + 1
    assert max(diag["sizes"]) == 6


def test_capping_win0_buys_nothing_on_the_metric_ledger():
    """别拿峰值 ∫g 去证明这个开关有效——它买的是 K，不是 ∫g。"""
    _, a = partition(LAM22, PILOT, METRIC,
                     min_states_per_window=4, max_states_per_window=8)
    _, b = partition(LAM22, PILOT, METRIC,
                     min_states_per_window=4, max_states_per_window=8,
                     first_window_max_states=4)
    assert abs(a["peak_metric_integral"] - b["peak_metric_integral"]) < 1.0


def test_predicted_delta_f_span_is_report_only():
    grad = [float(x) for x in np.linspace(-60.0, 60.0, len(PILOT))]
    _, with_g = partition(LAM22, PILOT, METRIC,
                          min_states_per_window=4, max_states_per_window=8,
                          pilot_mean_dU_dlambda=grad)
    _, without = partition(LAM22, PILOT, METRIC,
                           min_states_per_window=4, max_states_per_window=8)
    span = with_g["predicted_delta_f_span_kJ_mol_REPORT_ONLY"]
    assert span is not None and len(span) == len(with_g["sizes"])
    assert without["predicted_delta_f_span_kJ_mol_REPORT_ONLY"] is None
    # **仅报告**：它一个字都不许改布局。
    assert with_g["sizes"] == without["sizes"]


# --------------------------------------------------------------- A / B
def test_the_autonomous_loop_can_receive_a_reanchor_cadence():
    """不传 `reanchor_cadence_steps` ⟹ `_cadence = 0` ⟹ 固定节奏重锚整个失效。

    真机特征是日志里那句"没有窗口到重锚节奏（**0 步**）"。循环外的 rescue 路径
    一直传着 500k，而自治控制器一启用就把 rescue 关掉 ⟹ 这条设计从未在生产生效。
    """
    from abfe_pipeline import ABFEPipeline
    sig = inspect.signature(ABFEPipeline._run_stage2_autonomous)
    p = sig.parameters["f_k_reanchor_cadence_steps"]
    assert p.default == 500_000, "缺省必须非零，否则等于关掉固定节奏重锚"


def test_probe_suppression_is_keyed_to_the_board_not_to_ever():
    """探针的结论是**步数的函数**；只算一次等于把判据钉死在初始步数上。

    钉住的是"不再用 `_probed`（探针产物的 windows 列表）永久压制"——那份列表
    一次就含全部窗口，于是第一次探针之后全仓永不重探。
    """
    src = inspect.getsource(pre.Stage2RepairController.decide)
    assert "_is_noop(\"PROBE_CANDIDATE_FK\"" in src
    # 只看**代码**，注释里提到 `_probed` 是在解释为什么删掉它。
    code = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "_probed" not in code, "永久压制又回来了"


# ------------------------------------- 离线重算 / 分窗单一入口（2026-09-15 下半场）
def _path_diagnostics(n_pilot=None):
    """一份最小但真实的 path_diagnostics（pilot 数值取自 repeat01 落盘值）。"""
    grad = [float(x) for x in np.linspace(-80.0, 80.0, len(PILOT))]
    return {
        "pilot_lambdas": list(PILOT),
        "metric_g": list(METRIC),
        "pilot_points": [
            {"mean_dU_dlambda_kJ_mol": g, "n_derivative_samples": 100}
            for g in grad
        ],
    }


def test_offline_recompute_uses_the_production_partition_criterion():
    """派生层离线重算必须给出**与生产同一判据**的 ranges。

    先前 `recompute_vanishing_path_from_cached_pilot` 内部走
    `redistribute_vanishing_lambda_subdomains`，而后者把「λ 布点」与「分窗」耦在
    一个返回值里、分窗恒为等边数贪心 ⟹ 离线重算顺手带回旧判据的布局。
    """
    diag = _path_diagnostics()
    got = pre.recompute_vanishing_path_from_cached_pilot(
        diag, n_states=len(PILOT), final_state_count=22,
        min_states_per_window=4, max_states_per_window=8,
        free_energy_densify_points=0,
        partition_criterion="metric_integral",
        first_window_max_states=4,
    )
    assert got["partition_criterion"] == "metric_integral"
    assert got["partition_diagnostics"] is not None
    # 首窗上限真的生效了（等边数贪心给不出这种形状）
    ranges = [tuple(int(i) for i in r) for r in got["window_ranges"]]
    assert ranges[0][1] - ranges[0][0] == 4, ranges

    # 同一条 λ 表、同一份 pilot ⟹ 与直接调分窗器逐字相同
    direct, _ = partition(
        [float(x) for x in got["lambdas_vdw"]], PILOT, METRIC,
        min_states_per_window=4, max_states_per_window=8,
        first_window_max_states=4,
    )
    assert ranges == [tuple(int(i) for i in r) for r in direct]


def test_offline_recompute_rebuilds_every_per_window_field():
    """只换 ranges 不重建 `subdomain_*` = 缓存里 ranges 与描述对不上。"""
    diag = _path_diagnostics()
    got = pre.recompute_vanishing_path_from_cached_pilot(
        diag, n_states=len(PILOT), final_state_count=22,
        min_states_per_window=4, max_states_per_window=8,
        free_energy_densify_points=0,
        partition_criterion="metric_integral",
        first_window_max_states=4,
    )
    ranges = [tuple(int(i) for i in r) for r in got["window_ranges"]]
    alloc = got["subdomain_allocation"]
    lam = [float(x) for x in got["lambdas_vdw"]]
    assert alloc["subdomain_state_counts"] == [b - a for a, b in ranges]
    assert alloc["subdomain_interval_counts"] == [b - a - 1 for a, b in ranges]
    assert alloc["actual_state_index_sets"] == [
        list(range(a, b)) for a, b in ranges
    ]
    assert alloc["actual_shared_state_indices"] == [
        ranges[i][0] for i in range(1, len(ranges))
    ]
    assert alloc["total_window_state_slots"] == sum(b - a for a, b in ranges)
    assert alloc["subdomain_lambda_bounds"] == [
        [lam[a], lam[b - 1]] for a, b in ranges
    ]
    if "edge_free_energy_kJ_mol" in alloc:
        assert len(alloc["subdomain_free_energy_kJ_mol"]) == len(ranges)
    # λ 维的字段**不该**被分窗改动
    assert alloc["actual_state_count"] == len(lam)


def test_the_resume_check_and_the_recompute_agree_so_the_cache_is_not_rejected():
    """metric 模式 resume 不再判"缓存不是热力学 few-state 布局"。

    那条分支不是"打个 WARN"——它把 `optimized_lambdas_2` 置 None、**整份缓存拒掉、
    重跑一整轮 pilot**。先前它写死等边数贪心，metric_integral 下必然失配。
    """
    from abfe_pipeline import ABFEPipeline

    diag = _path_diagnostics()
    kw = {"stage2_window_partition": "metric_integral",
          "stage2_first_window_max_states": 4}
    vrk = {"min_states_per_window": 4, "max_states_per_window": 8}

    recomputed = pre.recompute_vanishing_path_from_cached_pilot(
        diag, n_states=len(PILOT), final_state_count=22,
        min_states_per_window=4, max_states_per_window=8,
        free_energy_densify_points=0,
        partition_criterion="metric_integral", first_window_max_states=4,
    )
    cached_lambdas = [float(x) for x in recomputed["lambdas_vdw"]]
    expected = ABFEPipeline._window_ranges_for_lambdas(
        cached_lambdas, kw, diag, vrk
    )
    assert [tuple(int(i) for i in r) for r in expected] == [
        tuple(int(i) for i in r) for r in recomputed["window_ranges"]
    ], "resume 校验与离线重算判据不一致 ⟹ 每次 resume 重跑 pilot"


def test_the_single_entry_actually_switches_on_the_criterion():
    from abfe_pipeline import ABFEPipeline

    diag = _path_diagnostics()
    vrk = {"min_states_per_window": 4, "max_states_per_window": 8}
    metric = ABFEPipeline._window_ranges_for_lambdas(
        LAM22, {"stage2_window_partition": "metric_integral"}, diag, vrk)
    arclen = ABFEPipeline._window_ranges_for_lambdas(
        LAM22, {"stage2_window_partition": "arclength"}, diag, vrk)
    assert [tuple(map(int, r)) for r in metric] != [tuple(map(int, r)) for r in arclen]
    # 显式分窗优先于一切
    explicit = ABFEPipeline._window_ranges_for_lambdas(
        LAM22, {"stage2_window_partition": "metric_integral",
                "stage2_window_ranges": [[0, 4], [3, 22]]}, diag, vrk)
    assert explicit == [(0, 4), (3, 22)]
    # metric 但缺 pilot 数据 ⟹ fail-closed，不静默退回等弧长
    with pytest.raises(RuntimeError, match="metric_integral 需要"):
        ABFEPipeline._window_ranges_for_lambdas(
            LAM22, {"stage2_window_partition": "metric_integral"}, {}, vrk)


def test_window_profile_is_report_only_and_carries_the_agreed_fields():
    grad = [float(x) for x in np.linspace(-80.0, 80.0, len(PILOT))]
    _, d = partition(LAM22, PILOT, METRIC, min_states_per_window=4,
                     max_states_per_window=8, pilot_mean_dU_dlambda=grad)
    rows = d["window_profile_REPORT_ONLY"]
    assert len(rows) == len(d["sizes"])
    for r in rows:
        assert set(r) == {
            "window", "K", "metric_integral", "thermodynamic_length",
            "predicted_f_span_kJ_mol", "observed_f_k_span_kJ_mol",
            "observed_min_N_eff_over_g",
        }
        # observed_* 要等采样跑完才存在，分窗这一刻必须是 None（不许伪造）
        assert r["observed_f_k_span_kJ_mol"] is None
        assert r["observed_min_N_eff_over_g"] is None
    assert [r["K"] for r in rows] == d["sizes"]
    assert [r["metric_integral"] for r in rows] == pytest.approx(
        d["metric_integral_per_window"]
    )
