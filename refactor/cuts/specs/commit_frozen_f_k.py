"""刀：_commit_frozen_f_k（commit_frozen_f_k）。引擎 spec，取代同名特化执行器。

⚠️ 形参表比历史多一个 `legacy_repair`（23 → 24）。历史那刀漏传，而方法体
L3406 有 `if not legacy_repair:` ⟹ 走到即 NameError。同型问题也出现在
_summarize_warmup。引擎的参数表对账抓到的；`verify_extract` 抓不到
（它只验搬走的块，形参表是签名侧的新代码）。本 spec 产出**正确**代码，
故与历史产物不逐字节相同。

锚点为结构谓词；`slice_expect` 只作漂移报告。
"""
import ast  # noqa: F401  （spec 里的 stmt_type 谓词要用）

from cutter import (assign_name, call_to, if_test, if_test_exact,  # noqa: F401
                    stmt_type)

SPEC = dict(
    mode="extract", name="commit_frozen_f_k",
    file="windows_sampling/window_manager_dual_lambda.py",
    klass="IBSWindowManagerDualLambda", fn="run_all_windows",
    new="_commit_frozen_f_k",
    container="loop",
    start=if_test_exact("warm.bias_converged"),
    single=True,
    slice_expect=(120, 121), stmts=1, span=(395, 480),   # 09-20 快照 405；09-23 主线 447
   
    params=['effective_frozen_validation_budget', 'ema_mean_p_values', 'expected_main_manifest', 'frozen_validation_is_final_rung', 'frozen_validation_reserved_steps', 'full_bias_step_budget', 'ibs_state_file', 'lc_win', 'legacy_repair', 'lv_win', 'main_ckpt_path', 'main_manifest_path', 'max_bias_updates', 'mode', 'prior_cumulative_steps', 'resumed_calibration_pending', 'sampler', 'sgd_step_budget', 'sim', 'skip_frozen_revalidation', 'stage_type', 'warm', 'warmup_only', 'window_idx'],
    exempt={"e", "f"},
    # 主线漂移（人工核过）：搬移区内只读，L18701 `if provisional_authorized and
    # debug_provisional_on_warmup_failure:`（前者循环体前段赋值，后者形参）
    drift_ok={"debug_provisional_on_warmup_failure", "provisional_authorized"},
    indent=8,
    doc_text="""相位 5：f_k 冻结/提交—— 刀1 第二刀·刀a 自 run_all_windows 原样搬出。

纯搬运（verify_extract --block for --slice 120:121 逐节点等价）；
跨界变量按原名作 keyword-only 形参（CUTTING_PLAN.md §2.2.1）。""",
)
