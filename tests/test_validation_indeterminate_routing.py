"""数据不足 ≠ f_k 错了：local-MBAR 验证门的四路分诊与"无法判定"结局。

背景：以前门只有两路（解出没过 / "暂不可解"），后者把"结构性输入错误"和"根本
没测出来"一起塞进 learning，清空批次重新攒同样大小的数据集，最终以
f_not_converged 收场——还可能触发插 λ，把误判推迟 10 万步。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = Path(__file__).resolve().parents[1]


def test_gate_error_is_classified_into_four_distinct_outcomes():
    from ibs_engine import classify_local_mbar_gate_error as clf

    assert clf(None) == "solved"
    # 求解器真正会返回的两种"没测出来"
    assert clf("insufficient_frames") == "insufficient_data"
    assert clf("insufficient_frames_after_decorrelation") == "insufficient_data"
    # 解了但 f 是 NaN/形状不对：是 f_k/重叠的负面信号，不是数据不足
    assert clf("nan_or_shape_mismatch_in_local_mbar_f") == "unsolvable"
    # 结构性错误：继续采样不会让它消失，必须明着报错
    assert clf("窗口 3 energies/bias/base 含 NaN/Inf") == "input_identity"
    assert clf("窗口 3 数据维度不匹配: u_kn=(4, 10), lambdas=5") == "input_identity"


def test_tiered_budget_constants_are_consistent():
    import ibs_engine as ie

    assert ie.IBS_LOCAL_MBAR_GATE_MAX_BATCHES > ie.IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES
    # 分级求解按 SLIDING 的整数倍推进（5 → 10 → 15），上限必须正好落在一档上，
    # 否则最后一档会是个半截的档位。
    assert (
        ie.IBS_LOCAL_MBAR_GATE_MAX_BATCHES
        % ie.IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES
    ) == 0


def test_indeterminate_is_a_distinct_type_from_a_convergence_failure():
    """两种诊断必须是两个类型：一个是"测出来了 f_k 不对"，一个是"没测出来"。"""
    import ibs_engine as ie

    assert not issubclass(
        ie.IBSValidationBudgetIndeterminateError, ie.IBSWarmupConvergenceError
    )
    assert not issubclass(
        ie.IBSValidationBudgetIndeterminateError, ie.IBSFrozenCalibrationValidationError
    )


def test_splitting_is_driven_by_fk_only_never_by_insufficient_ess():
    """**拆窗的依据只有"f_k 压不平"，而且只对末窗。** ESS/帧数不够治不了。

    历史：曾经把 ``IBSValidationBudgetIndeterminateError``（"没测出来"）也接到
    **拆窗**上 —— 拆完照样测不出来，只会每轮拆一次直到预算耗尽，还白白改掉实验
    布局。老板当场否了（原话"不是 ess 不够拆"），已回退。

    ⚠️ 2026-09-11 晚：路径演化**现在会拆窗**了（之前这条测试断言它不拆）。
    但触发条件收得很紧，三条同时成立才拆：
      1. 捕获的是 ``IBSWarmupConvergenceError``（f_k 压不平），**不是** indeterminate；
      2. 失败窗口**就是末窗** —— 末窗是溢出槽，插点不改变它的 λ 跨度，所以插点
         对它是无效动作，拆窗才对症（PLAN §3ter.2）；
      3. ``feasible_repair_actions`` 判它可拆（K_tail ∈ [2·lo−1, 2·hi−1]）。
    非末窗一律不拆（插点能缩它的跨度）。
    """
    src = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_stage2_with_path_evolution"
    )
    caught = {
        ast.unparse(h.type)
        for n in ast.walk(fn) if isinstance(n, ast.Try)
        for h in n.handlers if h.type is not None
    }
    assert "_ie.IBSWarmupConvergenceError" in str(caught), caught
    # "没测出来"永远不许驱动布局动作
    assert not any("Indeterminate" in c for c in caught), caught
    # 两种 reason 在版本链里必须分得清
    assert "ibs_warmup_f_k_not_converged" in src
    body = ast.unparse(fn)
    # 拆窗必须同时受"是末窗"和"可行性"两道闸
    assert "_is_tail" in body, "拆窗必须以『失败窗口是末窗』为前提"
    assert "feasible_repair_actions" in body, (
        "可行性必须问 feasible_repair_actions，不许在这里另写一遍可拆区间规则"
    )
    assert "split_window_from_ibs_lse_failure" in body, (
        "拆窗要真的调执行器 —— 它此前只被 import、从不被调用（动作没有入口）"
    )
    assert "split_tail_window" in body, "拆窗事件要以自己的 kind 进版本链"
    # 插点函数自己不拆窗，调用方不该再读拆窗产物
    assert "child_windows" not in src
    # 插不动/拆不动时上抛**原**异常，不改写成"补救失败"
    assert "raise err" in src


def test_insertion_does_not_split_and_only_the_tail_grows():
    """实测形状：window 4 [15:21] 六态测不出来 → 插 1 个 λ。

    ⚠️ 旧断言是"→ 4+4 两个四态窗口"。model B 下**不拆窗**：末窗是溢出槽，
    它从 6 态长到 7 态；长到 7 态之后才**可以**拆（这是另一条动作、另一个判据）。
    """
    import numpy as np
    from abfe_preoptimizer import (
        insert_lambda_in_failed_ibs_window,
        feasible_repair_actions,
    )

    lambdas = [1 - i / 20 for i in range(21)]
    ranges = [(0, 8), (7, 13), (12, 16), (15, 21)]
    pilot_lam, pilot_s = list(np.linspace(1, 0, 101)), list(np.linspace(0, 10, 101))

    new_l, new_r, diag = insert_lambda_in_failed_ibs_window(
        lambdas, ranges, (15, 21), pilot_lam, pilot_s,
        min_states_per_window=4, max_states_per_window=8, n_insert=1,
    )
    assert diag["n_inserted"] == 1
    assert diag["failed_window_is_last"] is True
    assert "child_windows" not in diag
    # 不拆窗：窗口数不变，只有末窗长大
    assert [b - a for a, b in new_r] == [8, 6, 4, 7]
    assert [tuple(r) for r in new_r[:-1]] == ranges[:-1]
    # 现在（且只有现在）才可拆
    assert feasible_repair_actions(
        new_r, len(new_l), min_states_per_window=4, max_states_per_window=8
    )["split_tail_window"] is None


def test_indeterminate_state_is_persisted_and_restored():
    """冻结 f_k / 已耗批数 / 已耗步数必须跨 resume 恢复，不能重新领额度。

    整份 load_ibs_state 需要真实 OpenMM Context，所以这里静态校验三件事：
    落盘写了批数、load 有对应分支、resume 分支会把批次接回来。
    """
    src = (ROOT / "ibs_engine.py").read_text(encoding="utf-8")
    assert '"frozen_validation_batches_done": int(' in src, "save_ibs_state 没落盘批数"

    tree = ast.parse(src)
    load_fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "load_ibs_state"
    )
    branch = next(
        (n for n in ast.walk(load_fn)
         if isinstance(n, ast.If)
         and "frozen_validation_indeterminate" in ast.unparse(n.test)),
        None,
    )
    assert branch is not None, "load_ibs_state 没有 frozen_validation_indeterminate 分支"
    assigned = {
        t.attr
        for stmt in branch.body if isinstance(stmt, ast.Assign)
        for t in stmt.targets if isinstance(t, ast.Attribute)
    }
    assert {
        "bias_status", "frozen_f_k_pending",
        "frozen_validation_cumulative_steps", "frozen_validation_batches_done",
    } <= assigned, assigned

    # resume 时必须把已累计的批次接回 frozen_mbar_batches，否则等于重新领额度
    assert "frozen_mbar_batches = list(sampler.tmbar_history[-_n_done:])" in src
    # 无法判定态也要恢复主窗口 checkpoint，否则 freeze_burn_in 会把批次冲掉
    assert src.count('"frozen_validation_indeterminate",') >= 1


def test_insertion_budget_is_counted_across_resume():
    """max_path_insertions 必须是跨 resume 的累计轮次，不是本次调用的循环次数。"""
    import tempfile
    import lambda_path_versions as lpv

    with tempfile.TemporaryDirectory() as d:
        lam = [1.0, 0.75, 0.5, 0.25, 0.0]
        rng = [[0, 3], [2, 5]]
        lpv.init_version(d, [0.0] * 5, lam, rng)
        assert lpv.count_events(d, "insert_lambda") == 0

        lam2 = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        rng2 = [[0, 3], [2, 6]]
        lpv.append_version(
            d, [0.0] * 6, lam2, rng2,
            kind="insert_lambda", reason="ibs_warmup_f_k_not_converged",
            detail={"n_inserted": 1},
        )
        assert lpv.count_events(d, "insert_lambda") == 1
        # 新进程（新的 count_events 调用）读到的仍然是 1，不归零
        assert lpv.count_events(d, "insert_lambda") == 1
        assert lpv.count_events(d, "init") == 1


def test_evolution_loop_budget_reads_the_version_chain():
    src = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    assert 'rounds_done = _lpv.count_events(checkpoint_dir, "insert_lambda")' in src
    assert "if rounds_done >= int(max_insertions):" in src
    # 窗口复用只看本次调用内部的重试次数，跟跨 resume 的预算是两回事
    assert "_resume_override=True if local_attempts > 0 else None," in src


# ---------------------------------------------------------------------------
# warmup 预算账本：累计上限语义
# ---------------------------------------------------------------------------

def test_plain_resume_never_increases_remaining_budget():
    """验收：同一无法判定 checkpoint 连续 resume 三次，剩余预算不增加。"""
    import ibs_engine as ie

    ledger = ie.new_warmup_budget_ledger(300_000)
    ledger["learning_steps"] = 120_000
    ledger["freeze_burn_in_steps"] = 15_000
    ledger["frozen_validation_steps"] = 165_000
    assert ie.warmup_ledger_total_steps(ledger) == 300_000
    assert ie.warmup_ledger_remaining_steps(ledger) == 0

    # 三次普通 resume：上限不动、消耗不动 ⟹ 可用一直是 0
    for _ in range(3):
        assert ie.warmup_ledger_remaining_steps(ledger) == 0
        assert ledger["cumulative_cap_steps"] == 300_000

    # 只有显式升档才有新增可用 = max(0, B2 - S)
    ledger["cumulative_cap_steps"] = 500_000
    assert ie.warmup_ledger_remaining_steps(ledger) == 200_000


def test_remaining_is_cap_minus_consumed_not_consumed_plus_default():
    import ibs_engine as ie

    ledger = ie.new_warmup_budget_ledger(300_000)
    ledger["learning_steps"] = 250_000
    # 正确：B - S = 50_000。错误写法 S + 默认 warmup_steps 会给出 250_000+ 。
    assert ie.warmup_ledger_remaining_steps(ledger) == 50_000


def test_three_buckets_share_one_cap():
    import ibs_engine as ie

    ledger = ie.new_warmup_budget_ledger(100_000)
    for bucket in ie.WARMUP_LEDGER_BUCKETS:
        ledger[bucket] = 40_000
    # 三项各 40k，总 120k 已经超过上限 100k ⟹ 可用 0，不是"每项各有 100k"
    assert ie.warmup_ledger_total_steps(ledger) == 120_000
    assert ie.warmup_ledger_remaining_steps(ledger) == 0
    assert ie.warmup_ledger_bucket_for_mode("learning") == "learning_steps"
    assert ie.warmup_ledger_bucket_for_mode("freeze_burn_in") == "freeze_burn_in_steps"
    assert ie.warmup_ledger_bucket_for_mode("validating") == "frozen_validation_steps"


def test_old_checkpoint_is_migrated_and_flagged_incomplete():
    """旧 checkpoint 缺账本：先从运行记录恢复，恢复不了也要标记不完整。"""
    import ibs_engine as ie

    recovered = ie.migrate_warmup_budget_ledger(
        {"frozen_validation_cumulative_steps": 250_000}, 300_000
    )
    assert recovered["frozen_validation_steps"] == 250_000
    assert recovered["complete"] is False
    assert ie.warmup_ledger_remaining_steps(recovered) == 50_000

    nothing = ie.migrate_warmup_budget_ledger({}, 300_000)
    # 绝不能默认"已耗 0 步"就当账本可信
    assert nothing["complete"] is False
    assert nothing["migrated_from"] == "nothing_recoverable"


def test_batch_allowance_is_bound_to_the_frozen_candidate():
    import ibs_engine as ie

    a = ie.frozen_candidate_fingerprint([0.0, 1.0, 2.0])
    assert a == ie.frozen_candidate_fingerprint([0.0, 1.0, 2.0])
    assert a != ie.frozen_candidate_fingerprint([0.0, 1.0, 2.5])
    assert ie.frozen_candidate_fingerprint(None) is None


def test_resume_action_and_fk_evidence_are_separate_concepts():
    src = (ROOT / "ibs_engine.py").read_text(encoding="utf-8")
    # 动作：恢复冻结验证（两种状态共用）
    assert "resume_frozen_validation = bool(" in src
    # 语义：证据状态（无法判定绝不等于已校准）
    assert "F_K_EVIDENCE_INDETERMINATE" in src
    assert "F_K_EVIDENCE_CALIBRATED" in src
    # 预算：账本
    assert "_ledger[\"cumulative_cap_steps\"] = int(_stored_cap)" in src
    # 只有显式 override 才算升档，resolver 的自动往上找一档兜底不再用于上限
    assert "_explicit_frozen_validation_rung = (" in src


def test_no_early_give_up_extrapolation_from_an_unsaturated_g():
    """不得按小样本的 n_eff 线性外推提前判死。

    ESS 跟采样量走：n_eff = N/g，而 g 只有 N ≫ τ 之后才稳定。实测 window 4 的 g
    还在随 N 涨（200 帧 38.5 → 400 帧 66.9），估计器远没饱和，这个区间的外推无效。
    该做的是按预算老实攒到上限。
    """
    import ibs_engine as ie

    assert not hasattr(ie, "local_mbar_gate_batches_needed")
    src = (ROOT / "ibs_engine.py").read_text(encoding="utf-8")
    assert "_hopeless" not in src
    # 只有攒到上限才判无法判定
    assert "if _n_now >= IBS_LOCAL_MBAR_GATE_MAX_BATCHES:" in src


def test_already_converged_window_is_not_demoted_by_an_unmeasurable_recheck():
    """resume 复验测不出来时，绝不能把已通过验证的窗口降级或中止整个 run。"""
    src = (ROOT / "ibs_engine.py").read_text(encoding="utf-8")
    idx = src.index("if _n_now >= IBS_LOCAL_MBAR_GATE_MAX_BATCHES:")
    tail = src[idx:idx + 2000]
    # 已收敛窗口的分支必须**先于** validation_indeterminate_diag 出现
    assert "if skip_warmup_entirely:" in tail
    assert tail.index("if skip_warmup_entirely:") < tail.index(
        "validation_indeterminate_diag = {"
    )
    assert "bias_converged = True" in tail


def test_config_raise_of_the_cap_takes_effect_but_never_shrinks():
    """改 config 抬高预算上限必须生效（显式动作）；调小不追溯缩水。"""
    import ibs_engine as ie

    src = (ROOT / "ibs_engine.py").read_text(encoding="utf-8")
    assert "elif _default_cap_steps > _stored_cap:" in src, "config 升档必须生效"
    assert "[预算升档]" in src, "升档要留可审计的日志"
    # 语义自检：只升不降
    led = ie.new_warmup_budget_ledger(555_000)
    led["learning_steps"] = 305_000
    assert ie.warmup_ledger_remaining_steps(led) == 250_000
    led["cumulative_cap_steps"] = 955_000          # config 升档后
    assert ie.warmup_ledger_remaining_steps(led) == 650_000


def test_zero_budget_is_halt_budget_not_fk_not_converged():
    """可用预算为 0、循环一次没进 ⟹ HALT_BUDGET，不是"f_k 不收敛"。

    实测 win2 连死三次，最后一次 `steps_at_full_bias=0`、gate history 空，
    却被标成 f_not_converged —— 那是"测出来了 f_k 不对"的语义。
    这既不是不收敛，也不是"测了没测出来"，是**根本没开始**。
    """
    src = (ROOT / "ibs_engine.py").read_text(encoding="utf-8")
    assert "_no_budget_at_all = bool(" in src
    # 三个条件缺一不可
    i = src.index("_no_budget_at_all = bool(")
    blk = src[i:i + 300]
    assert "steps_at_full_bias == 0" in blk
    assert "full_bias_step_budget <= 0" in blk
    assert "not ever_completed_a_validate_attempt" in blk
    # 这一支可能没有冻结候选 ⟹ 不许伪造 f_k_pending
    assert "没有候选就是没有" in src
    assert '"halt_budget_no_steps_available"' in src


def test_old_pass_does_not_stick_across_a_different_sampling_identity():
    """旧 PASS 只在**同一个 ensemble fingerprint** 下有效。

    实测过的洞：能量缓存因 `stage_protocol_key` 不符被拒、整窗重采，而 IBS 状态里的
    `bias_converged=True` 照样粘过去 ⟹ 窗口跳过 learning 只做"只读复验"，
    复验的却是**另一个系综**的旧结论。
    """
    src = (ROOT / "ibs_engine.py").read_text(encoding="utf-8")
    # 身份要落盘
    assert '"stage_protocol_key": getattr(self, "stage_protocol_key", None),' in src
    # skip_warmup_entirely 必须三条同时成立
    i = src.index("skip_warmup_entirely = bool(")
    blk = src[i:i + 200]
    assert "is_resumed_ibs" in blk and "sampler.bias_converged" in blk
    assert "_identity_matches" in blk, "旧 PASS 必须受采样身份限制"
    # 身份缺失一律不给（fail-closed），不是"缺了就当匹配"
    j = src.index("_identity_matches = (")
    cond = src[j:j + 400]
    assert "is not None" in cond and "_stage_window_sampling_identity" in cond
    # 不是整份丢掉：仍走热启动那条路
    assert "f_k 仍作热启动" in src
