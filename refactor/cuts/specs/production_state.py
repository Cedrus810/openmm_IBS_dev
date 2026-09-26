"""刀1 B 段第 2 步：ProductionState 收编 26 个生产段跨界名字（原地口径）。

原特化执行器 `cuts/_production_state_absorb.py`（207 行）的声明式等价物。

**原地口径的前提**（引擎会逐名核验，不符即 fail-closed）：26/26 的首写都是
`_produce_window` 体顶层无条件直赋 ⟹ 字段默认 None **不可观测**。
四个流幻影临时名已排除在名单外：`e`（except handler 局部）、`gid`/`e_g`
（debug 循环局部）、`i`（for 循环自覆盖）——它们的"跨段逃逸"是名字级分析的
假象，不是真的跨段。
"""
SPEC = dict(
    mode="absorb",
    name="production_state",
    in_place=True,

    file="windows_sampling/window_manager_dual_lambda.py",
    klass="IBSWindowManagerDualLambda",
    fn="_produce_window",

    bind="prod",
    dataclass="ProductionState",
    dataclass_after="WarmupState",     # 结构落点，不写行号
    decorator="@dataclass",
    dataclass_doc="""单窗生产段状态（刀1 B段第2步收编，2026-09-20）。

26 个跨相位 6/7/8 边界的名字（路径/预算/early-stop 累加器），原地收编：
初始化语句保留在 _prepare_production 段内（prod.<field> = ...），每窗经
`prod = ProductionState()` 重建。字段默认 None 不可观测（26/26 首写均为
体顶层无条件赋值，先写后读；执行器头注释有实测口径）。
""",

    names=[
        "actual_production_updates", "early_stop_check_history",
        "early_stop_check_interval_updates", "early_stop_kt",
        "early_stop_min_updates", "early_stop_pass_count",
        "early_stop_previous_local_dg", "early_stop_stop_reason",
        "early_stop_triggered", "effective_n_steps_per_window",
        "expected_production_manifest", "force_check_interval", "loop_timers",
        "n_updates", "prior_cumulative_production_steps", "production_base_path",
        "production_bias_path", "production_ckpt_path", "production_conv_path",
        "production_energies_path", "production_f_k_lock",
        "production_history_backup_len", "production_manifest_path",
        "production_pos_backup", "production_residual_basis_path",
        "production_sampling_states_path",
    ],
)
