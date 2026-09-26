"""刀：_produce_window（produce_window）。引擎 spec，取代同名特化执行器。

锚点为结构谓词；`slice_expect` 只作漂移报告。
"""
import ast  # noqa: F401  （spec 里的 stmt_type 谓词要用）

from cutter import assign_name, call_to, if_test, stmt_type  # noqa: F401

SPEC = dict(
    mode="extract", name="produce_window",
    file="windows_sampling/window_manager_dual_lambda.py",
    klass="IBSWindowManagerDualLambda", fn="run_all_windows",
    new="_produce_window",
    container="loop",
    take_leading_comments=True,   # 生产段头注释随代码搬（历史口径）
    start=assign_name("effective_n_steps_per_window"),
    end=if_test("_probe_context"),
    slice_expect=(129, 211),
    # 82 = 历史快照；88 = 09-24 主线 stage2 坐标旁车（人工核过）：循环体顶层新增
    # 6 条，全在搬移区内——生产准备段 5 条（production_coords_path 赋值、
    # sampler.coord_atom_indices/coord_sidecar 初始化、if STAGE2_COORDS_EVERY、
    # if not resumed_production_checkpoint … os.remove）+ 收口段 1 条
    # `_save_coords_sidecar(production_coords_path, sampler)`。
    stmts={82, 88}, span=(1200, 1290),   # 历史 1228；09-24 主线 1258
    params=['K', 'debug_mode', 'early_stop_check_interval_steps', 'early_stop_max_delta_g_drift_kJ_mol', 'early_stop_max_uncertainty_kJ_mol', 'early_stop_min_absolute_ess', 'early_stop_min_decorrelated_samples', 'early_stop_min_ess_ratio', 'early_stop_min_steps', 'early_stop_required_consecutive_passes', 'enable_early_stop', 'end', 'ibs_state_file', 'ibs_wrap', 'lc_win', 'lse_log_residual_tolerance', 'lv_win', 'n_steps_per_window', 'production_step_overrides', 'repair_policy', 'sampler', 'sim', 'skip_frozen_revalidation', 'stage_type', 'start', 'steps_per_update', 'warm', 'win_sys', 'win_sys_xml', 'window_idx'],
    exempt={"e", "f"},
    indent=8,
    doc_text="""相位 6-8（生产准备/生产循环/收口）—— 刀1 A 段自 run_all_windows 原样搬出。

纯搬运（verify_extract --block for --tail 82 逐节点等价）；跨界变量按原名
作 keyword-only 形参。窗口级控制流与引用释放协议（if warmup_only 块、
del sampler / sim = None / …）留在 run_all_windows 调用点
（CUTTING_PLAN.md §2.2/§2.3，实测修正见本文件头注释）。""",
)
