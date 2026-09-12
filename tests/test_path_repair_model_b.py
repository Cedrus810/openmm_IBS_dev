"""PLAN_PATH_REPAIR 2026-09-11 的 P0/P1 修补回归。

覆盖四件事，每件都是一个**曾经真的错了**的行为：

1. ``frozen_candidate_fingerprint`` 的 gauge 规范化 —— f_k 整体加常数是同一个
   采样分布，逐位哈希会算成两个候选、白白重建一次验证批次记录。
2. ``insert_lambda_in_failed_ibs_window`` 的 model B 记账 —— 窗口区间不动、
   末窗吸收溢出、失败窗口的 λ **跨度**真的缩小。旧实现（model A）把区间 +1、
   跨度不变，于是 bias 要压平的总落差一分没少，对"窗口太宽"是无效动作。
3. 边级证据不足时**明着报错**，不静默插 1 个 —— 旧公式 ``max(1, (2*lo−1) − K)``
   为了凑拆窗门槛强行插点，那些 λ 没有任何边级证据支持。
4. 被跳过的窗口必须进失败清单 —— 否则"补采这个窗口"对最需要它的窗口永远不可达。
"""

import json
import os

import numpy as np
import pytest

import abfe_preoptimizer as pre
import ibs_engine as ie
from abfe_pipeline import ABFEPipeline


# ---------------------------------------------------------------- pilot 夹具
def _pilot(n=201):
    """λ 从 1 线性降到 0；累计弧长严格递增。"""
    lam = np.linspace(1.0, 0.0, n)
    s = np.linspace(0.0, 1.0, n)
    return list(lam), list(s)


RANGES = [(0, 4), (3, 8), (7, 12), (11, 16)]


def _uniform_lambdas():
    return [float(x) for x in np.linspace(1.0, 0.0, 16)]


# ------------------------------------------------------------------- 1. gauge
def test_fingerprint_is_gauge_invariant():
    f_k = [-45.33, -23.93, -6.74, 6.45, 16.57, 24.00, 28.98]
    base = ie.frozen_candidate_fingerprint(f_k)
    for c in (1.0, -7.5, 1234.0):
        shifted = [x + c for x in f_k]
        assert ie.frozen_candidate_fingerprint(shifted) == base, (
            f"f_k + {c} 是同一个采样分布，指纹必须相同"
        )
    # 真正换了形状仍然必须换指纹（相邻差变了）
    other = list(f_k)
    other[3] += 2.0
    assert ie.frozen_candidate_fingerprint(other) != base
    assert ie.frozen_candidate_fingerprint(None) is None


# -------------------------------------------------------------- 2. model B 记账
def test_model_b_ranges_fixed_tail_absorbs_and_span_shrinks():
    lam0 = _uniform_lambdas()
    pl, ps = _pilot()
    new_l, new_r, diag = pre.insert_lambda_in_failed_ibs_window(
        list(lam0), list(RANGES), (0, 4), pl, ps,
        min_states_per_window=4, max_states_per_window=5, n_insert=1,
    )
    assert len(new_l) == 17
    # 区间：除末窗外逐字不变；末窗上界 +1
    assert new_r[:-1] == RANGES[:-1]
    assert tuple(new_r[-1]) == (11, 17)
    # 非末窗态数一个都没变
    for (a0, b0), (a1, b1) in zip(RANGES[:-1], new_r[:-1]):
        assert (b1 - a1) == (b0 - a0)
    # 末窗吸收了那一个态
    assert (new_r[-1][1] - new_r[-1][0]) == (RANGES[-1][1] - RANGES[-1][0]) + 1
    # 失败窗口的 λ 跨度真的缩小了（model A 这一项会相等）
    span_before = abs(lam0[3] - lam0[0])
    span_after = abs(new_l[3] - new_l[0])
    assert span_after < span_before - 1e-12
    assert diag["failed_window_span_after"] < diag["failed_window_span_before"]
    assert diag["accounting"] == "model_B_fixed_ranges_tail_absorbs_overflow"
    # 不拆窗
    assert diag["n_windows_before"] == diag["n_windows_after"] == 4


def test_model_b_prefix_lambdas_are_bit_identical():
    """前缀（已采完的窗口）λ 必须逐位不变，否则已有产物全部失配。"""
    lam0 = _uniform_lambdas()
    pl, ps = _pilot()
    # 失败窗口取中间那个，前面有一个真正的前缀窗口
    new_l, new_r, _ = pre.insert_lambda_in_failed_ibs_window(
        list(lam0), list(RANGES), (7, 12), pl, ps,
        min_states_per_window=4, max_states_per_window=5, n_insert=2,
    )
    assert len(new_l) == 18
    for a, b in RANGES:
        if b <= 7 + 1:  # 插入点之前的窗口
            assert [round(x, 12) for x in lam0[a:b]] == [
                round(x, 12) for x in new_l[a:b]
            ], f"前缀窗口 {(a, b)} 的 λ 被改动了"
    assert tuple(new_r[-1]) == (11, 18)


def test_model_b_keeps_coverage_and_single_shared_boundary():
    lam0 = _uniform_lambdas()
    pl, ps = _pilot()
    new_l, new_r, _ = pre.insert_lambda_in_failed_ibs_window(
        list(lam0), list(RANGES), (0, 4), pl, ps,
        min_states_per_window=4, max_states_per_window=5, n_insert=3,
    )
    # 这一步本身就会在函数内部校验；这里再独立断言一次
    pre.validate_single_shared_boundary_ranges(new_r, len(new_l))
    assert sorted(set().union(*[set(range(a, b)) for a, b in new_r])) == list(
        range(len(new_l))
    )


# --------------------------------------------------- 3. 边级证据不足 → 明着报错
def test_uniform_path_refuses_to_insert_without_edge_evidence():
    """等热力学长度布点下，没有任何一条边"太长" ⟹ 不许静默插 1 个。"""
    lam0 = _uniform_lambdas()
    pl, ps = _pilot()
    with pytest.raises(RuntimeError, match="边级证据不支持插点"):
        pre.insert_lambda_in_failed_ibs_window(
            list(lam0), list(RANGES), (0, 4), pl, ps,
            min_states_per_window=4, max_states_per_window=5,
        )


def test_edge_driven_n_scales_with_edge_length():
    """窗内有一条明显超长的边时，n 由 ⌈L_edge/L_target⌉−1 决定。"""
    # win0 的第一条边拉长到约 4 倍平均，其余保持紧密
    lam = [1.0, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25,
           0.22, 0.19, 0.16, 0.13, 0.10, 0.07, 0.04, 0.0]
    pl, ps = _pilot()
    # 🔑 用 max=8（= 生产 config 的实际取值）而不是 5：这条超长边算出 n=6，
    # 而 max=5 时末窗可拆上限是 2·5−1=9、末窗只有 5 态 ⟹ 最多只能吸收 4 个，
    # 会先撞上"溢出槽死胡同"守卫（那条守卫有自己的两个测试）。这里要钉的是
    # **n 随边长增长**这一条性质，不该跟溢出槽预算耦合在一起。
    # max=8 ⟹ 可拆上限 15，5+6=11 ≤ 15，守卫不参与。
    _, _, diag = pre.insert_lambda_in_failed_ibs_window(
        list(lam), list(RANGES), (0, 4), pl, ps,
        min_states_per_window=4, max_states_per_window=8,
    )
    assert diag["n_insert_source"] == "edge_length"
    assert diag["n_inserted"] >= 1
    # 插点必须落在那条超长边里
    assert diag["inserted_global_edges"][0] == [0, 1]


# ------------------------------------------------------------- 4. 可行性规则
def test_feasible_split_needs_two_times_lo_minus_one():
    # 4/5 配置：末窗 5 态，拆不出两个 >=4 的子窗
    d = pre.feasible_repair_actions(
        RANGES, 16, min_states_per_window=4, max_states_per_window=5
    )
    assert d["insert_lambda"] is None, "末窗豁免上限 ⟹ 插点总是可行"
    assert d["split_tail_window"] is not None
    assert "最小可拆 K = 7" in d["split_tail_window"]
    # 末窗长到 7 态之后可拆（4+4）
    grown = RANGES[:-1] + [(11, 18)]
    d2 = pre.feasible_repair_actions(
        grown, 18, min_states_per_window=4, max_states_per_window=5
    )
    assert d2["split_tail_window"] is None


def test_feasible_insert_blocked_once_overflow_slot_is_gone():
    """末窗被拆过之后不再豁免上限 ⟹ 溢出槽耗尽 ⟹ 插点不可行。"""
    ranges = RANGES[:-1] + [(11, 16)]
    d = pre.feasible_repair_actions(
        ranges, 16, min_states_per_window=4, max_states_per_window=5,
        tail_exempt_from_max=False, n_insert=1,
    )
    assert d["insert_lambda"] is not None
    assert "HALT_LAMBDA_BUDGET_INSUFFICIENT" in d["insert_lambda"]


def test_feasible_rejects_unsorted_or_uncovered_ranges():
    with pytest.raises(ValueError):
        pre.feasible_repair_actions(
            [(3, 8), (0, 4)], 8, min_states_per_window=4, max_states_per_window=5
        )
    with pytest.raises(ValueError):
        pre.feasible_repair_actions(
            RANGES, 17, min_states_per_window=4, max_states_per_window=5
        )


# ------------------------------------------- 5. 被跳过的窗口必须进 rescue 候选
def test_skipped_windows_reach_the_failure_list():
    result = {
        "window_overlap_diagnostics": [],
        "skipped_windows": [
            {
                "window_index": 0,
                "lambda_indices": [0, 1, 2, 3, 4, 5, 6],
                "n_frames_after_decorrelation": 6,
                "min_frames_per_window": 10,
                "statistical_inefficiency": 85.6,
                "statistical_inefficiency_per_lambda": [1.1, 1.0, 23.1, 77.5, 90.3, 88.0, 85.6],
                "reason": "insufficient_frames_after_decorrelation",
            }
        ],
    }
    details = ABFEPipeline._stage_quality_failure_details(result)
    assert [d["window_index"] for d in details] == [0]
    assert details[0]["failed_gates"] == [
        "skipped_insufficient_frames_after_decorrelation"
    ]
    assert details[0]["n_frames_decorrelated"] == 6
    # 逐态剖面必须带过来：远端单调衰减 = 跨度太大，整体偏低 = 采样不够
    assert details[0]["statistical_inefficiency_per_lambda"][-1] == 85.6
    # 格式化不能因为缺字段而炸
    assert "window 0" in ABFEPipeline._format_stage_quality_failure_details(details)


def test_skipped_windows_do_not_pollute_overlap_diagnostics():
    """失败清单是独立产物；window_overlap_diagnostics 不许被掺入伪记录。"""
    result = {"window_overlap_diagnostics": [], "skipped_windows": [
        {"window_index": 2, "lambda_indices": [7, 8], "reason": "x"}
    ]}
    ABFEPipeline._stage_quality_failure_details(result)
    assert result["window_overlap_diagnostics"] == []


# ---------------------------------------------- 6. 逐态 g 剖面不再被丢掉
def test_per_state_g_profile_is_reported():
    rng = np.random.default_rng(0)
    n = 400
    # 三个目标态：第 0 个近乎白噪（g≈1），第 2 个强相关（g 大）
    fast = rng.normal(size=n)
    slow = np.convolve(rng.normal(size=n + 60), np.ones(60) / 60.0, mode="valid")[:n]
    u = np.vstack([fast, 0.5 * fast + 0.5 * slow, slow]) * 2.5
    prof: list = []
    _idx, g, _worst = ie._decorrelate_by_worst_target_state(
        u, np.zeros(n), 2.5, per_state_g_out=prof
    )
    assert len(prof) == 3, "逐态 g 必须每个目标态一条"
    assert prof[2] > prof[0], "强相关的那个态 g 必须更大"
    assert g == pytest.approx(max(prof)) or g == 1.0


# ------------------------- 7. 调用方必须给窗口级的 n（否则生产路径会直接中止）
def test_path_evolution_passes_a_window_level_n_insert():
    """触发路径演化的是**窗口级**证据，而布局按热力学长度等分 ⟹ 边级默认判据
    会算出 n=0 并明着报错 ⟹ 生产路径会中止在那里。调用方必须传 n_insert。

    见 PLAN_PATH_REPAIR §P3-9c：当前是每轮 1 个（每轮缩掉一条边），由外层
    max_insertions 限轮数；这是不引入未验证阈值的最小选择，不是最终判据。
    """
    import inspect
    import abfe_pipeline

    src = inspect.getsource(
        abfe_pipeline.ABFEPipeline._run_stage2_with_path_evolution
    )
    call = src[src.index("insert_lambda_in_failed_ibs_window("):]
    call = call[:call.index("\n                    )")]
    assert "n_insert=" in call, (
        "不传 n_insert 会让生产路径落到边级判据、算出 n=0 直接中止"
    )


# --------------------- 8. 溢出槽不是无底洞：超过 2·hi−1 就永远拆不开
def test_tail_cannot_grow_past_the_splittable_ceiling():
    """末窗豁免 max_states，但可拆区间只有 [2·lo−1, 2·hi−1]。

    4/5 配置 ⟹ 可拆 K ∈ {7,8,9}。插到 K=10 就永远切不出两个落在 [4,5] 的子窗，
    而拆末窗恰恰是唯一的收尾动作 ⟹ 溢出槽变成死胡同。必须 fail-closed。
    （当前 max_insertions=3 把 K 压在 5+3=8 以内只是巧合，不是守卫。）
    """
    pl, ps = _pilot()
    # 末窗已经 9 态（可拆上限），再插 1 个就越界
    lam = [float(x) for x in np.linspace(1.0, 0.0, 20)]
    ranges = [(0, 4), (3, 8), (7, 11), (10, 20)]  # 末窗 10 态 > 9，本身已越界
    with pytest.raises(RuntimeError, match="可拆上限"):
        pre.insert_lambda_in_failed_ibs_window(
            lam, ranges, (0, 4), pl, ps,
            min_states_per_window=4, max_states_per_window=5, n_insert=1,
        )
    d = pre.feasible_repair_actions(
        ranges, 20, min_states_per_window=4, max_states_per_window=5, n_insert=1
    )
    assert d["insert_lambda"] is not None
    assert "死胡同" in d["insert_lambda"]


def test_tail_at_the_ceiling_can_still_be_split():
    """K=9（上限）时插点不可行，但**拆**仍然可行 —— 出口存在，不是死局。"""
    ranges = [(0, 4), (3, 8), (7, 11), (10, 19)]  # 末窗 9 态
    d = pre.feasible_repair_actions(
        ranges, 19, min_states_per_window=4, max_states_per_window=5, n_insert=1
    )
    assert d["insert_lambda"] is not None, "再插会越界"
    assert d["split_tail_window"] is None, "但拆得开 —— 这才是该走的动作"


# ============================================================================
# 9. warmup 预算账本跨 checkpoint 子命名空间继承
#    （同一批 2026-09-11 修复；不属于 model B，但是同一次实测里发现的）
# ============================================================================
def _write_state(d, stage_type, idx, ledger, **extra):
    os.makedirs(d, exist_ok=True)
    payload = {"bias_status": "converged", "lambdas_vdw": [1.0, 0.9, 0.8, 0.7]}
    if ledger is not None:
        payload["warmup_budget_ledger"] = ledger
    payload.update(extra)
    with open(os.path.join(d, f"ibs_state_{stage_type}_window_{idx}.json"), "w") as fh:
        json.dump(payload, fh)


def test_new_sampling_segment_inherits_the_warmup_spend(tmp_path):
    """**每开一个采样段不许白送一整份 warmup 额度。**

    实测（cyclod_ligand2/rep1_evidence，window 4，cap=555000）：
        vanishing    110k+15k+350k = 475k 已耗，剩  80k
        vanishing_2   10k+ 5k+ 50k =  65k 已耗，剩 490k   ← 实际跨段烧了 540k

    段有自己的 checkpoint 命名空间（`checkpoints/segment_2/`），段内 ibs_state
    从零开始，于是账本也是空的。这与两处既有约定直接矛盾：
    `migrate_warmup_budget_ledger` 的"绝不默认『已耗 0 步』再按默认上限赠送一整份
    额度"，以及 `frozen_candidate_fingerprint` 的"账本里的总消耗永远不清零"。
    """
    base = tmp_path / "checkpoints"
    base.mkdir()
    (base / "path_current.json").write_text(json.dumps({"version": 1}))
    _write_state(str(base), "vdw", 4, {
        "learning_steps": 110000, "freeze_burn_in_steps": 15000,
        "frozen_validation_steps": 350000, "cumulative_cap_steps": 555000,
        "complete": True,
    }, frozen_validation_cumulative_steps=205000)

    seg = base / "segment_2"
    led = ie.inherit_warmup_ledger_across_segments(str(seg), "vdw", 4, 555000)
    assert led is not None
    assert ie.warmup_ledger_total_steps(led) == 475000, "消耗必须继承"
    assert ie.warmup_ledger_remaining_steps(led) == 80000, "不是 490000"
    # 只继承消耗桶，**不**继承那份候选的验证进度（新段是新的 f_k Epoch）
    assert "frozen_validation_cumulative_steps" not in led
    assert led["inherited_from"] == "checkpoints"


def test_rescue_namespace_is_two_levels_deep_and_still_inherits(tmp_path):
    """rescue 的命名空间是 `checkpoints/vanishing_rescue/{plan_id}/`（两层）。
    只认 `segment_N` 一种形状会漏掉它 —— 锚点用 `path_current.json`（λ 路径是全局的，
    只有基准目录才有它）。"""
    base = tmp_path / "checkpoints"
    base.mkdir()
    (base / "path_current.json").write_text(json.dumps({"version": 1}))
    _write_state(str(base), "vdw", 0, {
        "learning_steps": 30000, "freeze_burn_in_steps": 10000,
        "frozen_validation_steps": 150000, "cumulative_cap_steps": 555000,
        "complete": True,
    })
    deep = base / "vanishing_rescue" / "abc123def456"
    led = ie.inherit_warmup_ledger_across_segments(str(deep), "vdw", 0, 555000)
    assert ie.warmup_ledger_total_steps(led) == 190000


def test_base_checkpoint_dir_has_nothing_to_inherit(tmp_path):
    """基准目录自己不是子命名空间 ⟹ 返回 None，走原来的新建路径，行为不变。"""
    base = tmp_path / "checkpoints"
    base.mkdir()
    (base / "path_current.json").write_text(json.dumps({"version": 1}))
    _write_state(str(base), "vdw", 0, {"learning_steps": 1, "cumulative_cap_steps": 5})
    assert ie.base_checkpoint_dir_for(str(base)) is None
    assert ie.inherit_warmup_ledger_across_segments(str(base), "vdw", 0, 555000) is None


def test_missing_prior_ledger_is_not_treated_as_zero_spend(tmp_path):
    """上一段连账本都没有 ⟹ 消耗**不可知**，不许当成 0。走迁移路径并标 complete=False
    （那种账本不得用来论证升档）。"""
    base = tmp_path / "checkpoints"
    base.mkdir()
    (base / "path_current.json").write_text(json.dumps({"version": 1}))
    _write_state(str(base), "vdw", 1, None, frozen_validation_cumulative_steps=250000)
    led = ie.inherit_warmup_ledger_across_segments(
        str(base / "segment_3"), "vdw", 1, 555000
    )
    assert led is not None
    assert led.get("complete") is False, "消耗不可知的账本必须标不完整"
    assert ie.warmup_ledger_total_steps(led) > 0, "已知的那部分消耗要恢复出来"


def test_no_prior_state_at_all_returns_none(tmp_path):
    """基准目录里根本没有这个窗口的 ibs_state ⟹ None（真的从零开始，合法）。"""
    base = tmp_path / "checkpoints"
    base.mkdir()
    (base / "path_current.json").write_text(json.dumps({"version": 1}))
    assert ie.inherit_warmup_ledger_across_segments(
        str(base / "segment_2"), "vdw", 7, 555000
    ) is None


# ============================================================================
# 10. 四处接线（2026-09-11 晚）：拆末窗 / 可行性 / 控制器影子 / f_k 驳回 catcher
#     都是"写了但没人调"的缺口，这里用静态契约钉住入口存在。
# ============================================================================
def _evolution_src():
    """方法源码。**必须 dedent**：`inspect.getsource` 取出来的方法带类体那层缩进，
    直接喂 `ast.parse` 会 IndentationError（只有做 ast 解析的那两条会炸，纯字符串
    断言不会，所以在单测里表现成"时好时坏"）。"""
    import inspect
    import textwrap
    import abfe_pipeline
    return textwrap.dedent(
        inspect.getsource(
            abfe_pipeline.ABFEPipeline._run_stage2_with_path_evolution
        )
    )


def test_tail_split_executor_is_actually_called():
    """`split_window_from_ibs_lse_failure` 此前只被 import、从不被调用 ——
    等于"拆末窗"这个动作在生产里**没有入口**，而它是溢出槽唯一的收尾动作。"""
    src = _evolution_src()
    assert "split_window_from_ibs_lse_failure(" in src
    assert "_is_tail" in src, "只允许末窗拆"
    assert 'kind="split_tail_window"' in src, "拆窗事件要有自己的 kind"


def test_feasibility_is_consulted_not_reimplemented():
    """可拆区间 [2lo−1, 2hi−1] 只能有**一份**规则。"""
    src = _evolution_src()
    assert "feasible_repair_actions(" in src
    # **判断**必须来自 feasible_repair_actions，不许在调用方重算可拆性。
    # （诊断里回显一下区间边界是记录、不是决策，所以只禁那条枚举判据本身。）
    assert "p + q - 1 ==" not in src, "可拆性判据只能有一份，在 feasible_repair_actions 里"


def test_controller_runs_in_shadow_and_cannot_break_the_run():
    """控制器已接进生产，但**只记录、不据此做事**；而且影子失败不能弄坏 run。"""
    import inspect
    import abfe_pipeline

    src = _evolution_src()
    assert "_log_controller_shadow(" in src, "控制器要在生产里跑起来才有对账数据"
    shadow = inspect.getsource(abfe_pipeline.ABFEPipeline._log_controller_shadow)
    assert "except Exception" in shadow, "影子绝不允许阻断生产"
    # 影子只读：不得出现任何落盘/改状态的调用
    for forbidden in ("_atomic_write_json", "open(", "makedirs", "raise"):
        assert forbidden not in shadow, f"影子里不该出现 {forbidden}"


def test_fk_refuted_is_caught_diagnosed_and_reraised():
    """`IBSFrozenCalibrationValidationError` 此前**全仓库零 catcher**，直接炸穿
    整个 run、什么诊断都不留（三轴文档 C3）。

    现在必须：落盘诊断 → 说清结论 → **原样上抛**。
    刻意**不**决定策略（终态交人工 vs 回 LEARN 重来 = PLAN P3 待定 A）；
    更不许把它接到插 λ / 拆窗上 —— 那是有统计功效的 f_k 否决，不是 λ 太稀。
    """
    import ast, inspect
    import abfe_pipeline

    fn = ast.parse(_evolution_src()).body[0]
    caught = {
        ast.unparse(h.type)
        for n in ast.walk(fn) if isinstance(n, ast.Try)
        for h in n.handlers if h.type is not None
    }
    assert any("IBSFrozenCalibrationValidation" in c for c in caught), caught
    src = _evolution_src()
    assert "stage2_fk_refuted.json" in src, "驳回必须留下可审计的产物"
    assert "HALT_FK_REFUTED" in src
    assert "STATISTICALLY_REJECTED" in src
    assert "pending_decision" in src, "未定的策略要明写成待定，不许代码自己发明"
    # 驳回这条**不得**触发布局动作
    handler = next(
        h for n in ast.walk(fn) if isinstance(n, ast.Try)
        for h in n.handlers
        if h.type is not None and "IBSFrozenCalibrationValidation" in ast.unparse(h.type)
    )
    hsrc = ast.unparse(handler)
    for forbidden in ("insert_lambda_in_failed_ibs_window",
                      "split_window_from_ibs_lse_failure", "append_version"):
        assert forbidden not in hsrc, f"f_k 被驳回不该触发 {forbidden}"
    assert hsrc.rstrip().endswith("raise"), "落盘诊断之后必须原样上抛"
