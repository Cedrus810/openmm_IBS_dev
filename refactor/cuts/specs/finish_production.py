"""刀d：[相位7+8] 联合段 → _finish_production。引擎 spec，取代同名特化执行器。

搬移区 = `_produce_window` 体内 `for up in ...:` 起直到方法体尾（无终点锚）。
**不**单独切 7/8：边界上 9 个相位 7 内嵌首写名（fmax/state_n 等）吸收后会把
n_updates==0 路径的 UnboundLocal 改成 None 流入相位 8（用户已拍板不批准）。
豁免：e（handler 局部）、gid/e_g（debug 循环局部）、i（for 局部）——流幻影。
"""
from cutter import for_target

SPEC = dict(
    mode="extract", name="finish_production",
    file="windows_sampling/window_manager_dual_lambda.py",
    klass="IBSWindowManagerDualLambda", fn="_produce_window",
    new="_finish_production",
    start=for_target("up"),
    end=None,
    slice_expect=(58, 83),
    # 25 = 历史快照；26 = 09-24 主线坐标旁车收口 `_save_coords_sidecar(production_coords_path, sampler)`（人工核过）
    stmts={25, 26}, span=(770, 840),
    params=['K', 'debug_mode', 'early_stop_check_interval_steps', 'early_stop_max_delta_g_drift_kJ_mol', 'early_stop_max_uncertainty_kJ_mol', 'early_stop_min_absolute_ess', 'early_stop_min_decorrelated_samples', 'early_stop_min_ess_ratio', 'early_stop_min_steps', 'early_stop_required_consecutive_passes', 'enable_early_stop', 'end', 'ibs_state_file', 'ibs_wrap', 'lc_win', 'lse_log_residual_tolerance', 'lv_win', 'n_steps_per_window', 'prod', 'repair_policy', 'sampler', 'sim', 'stage_type', 'start', 'steps_per_update', 'warm', 'win_sys', 'window_idx'],
    exempt={"e", "gid", "e_g", "i"},
    # 主线漂移（人工核过）。放行判据：①搬移区内只读 ②调用点**必然已绑定**
    # （首写是调用方体顶层无条件直赋——条件绑定的名字当形参传，会把原来只在
    # 某条路径上才读的 UnboundLocal 提前到调用点，改语义）。
    drift_ok={
        # 09-20 主线续采修复：_produce_window 体顶层无条件 `= max(...)`；本段两处只读
        "remaining_production_steps",
        # 09-24 主线 stage2 坐标旁车：_produce_window 体顶层无条件
        # `= _coords_sidecar_path(...)`（for up 之前）；本段仅收口 _save_coords_sidecar 一处只读
        "production_coords_path",
    },
    indent=8,
    doc_text="""相位 7+8：生产采样循环 + 收口/落盘—— 刀1 B段第3步自
_produce_window 原样搬出（联合段；7/8 细分等 n_updates==0 路径
语义拍板，见执行器头注释）。纯搬运（verify_extract --slice 逐节点
等价）；跨界变量按原名作 keyword-only 形参。""",
)
