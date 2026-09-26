"""刀：_run_warmup_loop（run_warmup_loop）。引擎 spec，取代同名特化执行器。

锚点为结构谓词；`slice_expect` 只作漂移报告。
"""
import ast  # noqa: F401  （spec 里的 stmt_type 谓词要用）

from cutter import assign_name, call_to, if_test, stmt_type  # noqa: F401

SPEC = dict(
    mode="extract", name="run_warmup_loop",
    file="windows_sampling/window_manager_dual_lambda.py",
    klass="IBSWindowManagerDualLambda", fn="run_all_windows",
    new="_run_warmup_loop",
    container="loop",
    start=stmt_type(ast.While),
    single=True,
    slice_expect=(99, 100), stmts=1, span=(765, 785),
    params=['K', '_attempt_ledger', '_nan_diagnostics', 'check_chunk', 'expected_main_manifest', 'full_bias_step_budget', 'ibs_state_file', 'lc_win', 'lv_win', 'main_ckpt_path', 'main_manifest_path', 'max_bias_updates', 'prior_cumulative_steps', 'provisional_authorized', 'resumed_calibration_pending', 'sampler', 'sim', 'skip_frozen_revalidation', 'skip_warmup_entirely', 'stage_type', 'start', 'warm', 'window_idx'],
    # k：只在推导式/for 局部里读，调用方的 k 是循环局部、未必已绑定（09-23 verify_bound）
    exempt={"e", "f", "k"},
    indent=8,
    doc_text="""相位 3：预热循环—— 刀1 第二刀·刀b 自 run_all_windows 原样搬出。

纯搬运（verify_extract --block for --slice 逐节点等价）；跨界变量按
原名作 keyword-only 形参（CUTTING_PLAN.md §2.2.1）。""",
)
