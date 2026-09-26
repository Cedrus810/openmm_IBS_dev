"""刀：_summarize_warmup（summarize_warmup）。引擎 spec，取代同名特化执行器。

⚠️ **本 spec 的形参表比历史产物多一个 `legacy_repair`（39 → 40）。**
历史那刀漏传了它，而方法体内 L4580 `if legacy_repair` / L4604 有两处引用
⟹ 走到那条路径就 NameError。`verify_extract` 抓不到（它只验搬走的块，
形参表是签名侧的新代码——与刀2 分派条件、Cut A return 解包同一个缺口）。
引擎的参数表对账抓到了。本 spec 产出**正确**代码，因此与历史产物
**不逐字节相同**，验收只取 AST 等价 + 该单一差异。

锚点为结构谓词；`slice_expect` 只作漂移报告。
"""
import ast  # noqa: F401  （spec 里的 stmt_type 谓词要用）

from cutter import assign_name, call_to, if_test, stmt_type  # noqa: F401

SPEC = dict(
    mode="extract", name="summarize_warmup",
    file="windows_sampling/window_manager_dual_lambda.py",
    klass="IBSWindowManagerDualLambda", fn="run_all_windows",
    new="_summarize_warmup",
    container="loop",
    start_after=call_to("self._run_warmup_loop"),
    end=call_to("self._commit_frozen_f_k"),
    slice_expect=(100, 120), stmts=20, span=(675, 760),   # 09-20 快照 686；09-23 主线 712
   
    params=['K', '_attempt_ledger', '_ledger', '_nan_diagnostics', 'accept_recalibrated_f_k_without_gate', 'allow_best_effort_warmup', 'check_chunk', 'coverage_ess_threshold', 'effective_frozen_validation_budget', 'effective_mbar_calibration_reserved_steps', 'end', 'expected_main_manifest', 'frozen_validation_reserved_steps', 'full_bias_step_budget', 'ibs_state_file', 'ibs_wrap', 'lc_win', 'legacy_repair', 'lse_log_residual_tolerance', 'lv_win', 'main_ckpt_path', 'main_manifest_path', 'max_bias_updates', 'min_bias_updates', 'min_probability_threshold', 'minimum_complete_validation_frames', 'required_consecutive_bias_updates', 'resume_frozen_validation', 'resumed_calibration_pending', 'sampler', 'sgd_step_budget', 'sim', 'stage_type', 'start', 'target_p', 'validation_attempt_budget_steps', 'warm', 'warmup_only', 'window_idx'],
    # k：只在推导式/for 局部里读，调用方的 k 是循环局部、未必已绑定（09-23 verify_bound）
    exempt={"e", "f", "k"},
    # 主线漂移（人工核过）：09-24「分档累计途中耗尽也走临时生产」新增
    # `if provisional_authorized and debug_provisional_on_warmup_failure and …`，
    # 搬移区内两名只读。前者全方法唯一写 = 循环体顶层无条件直赋（搬移区之前），
    # 后者是 run_all_windows 形参 ⟹ 调用点必然已绑定。同两名 09-23 已在 commit_frozen_f_k 放行。
    drift_ok={"debug_provisional_on_warmup_failure", "provisional_authorized"},
    indent=8,
    doc_text="""相位 4：预热收口/诊断—— 刀1 第二刀·刀c 自 run_all_windows 原样搬出。

纯搬运（verify_extract --block for --slice 逐节点等价）；跨界变量按
原名作 keyword-only 形参（CUTTING_PLAN.md §2.2.1）。""",
)
