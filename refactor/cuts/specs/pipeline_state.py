"""刀2 B 段：PipelineState 收编 _run_dual_lambda_scheme 的 26 个跨相位名字。

原特化执行器 `cuts/_pipeline_state_absorb.py`（203 行）的声明式等价物。
原地口径：初始化语句留在原处变成 `dual.<field> = ...`，字段默认 None
不可观测（26/26 首写为体顶层无条件直赋，引擎逐名核验）。
"""
SPEC = dict(
    mode="absorb", name="pipeline_state", in_place=True,
    file="pipeline_orchestration/abfe_pipeline_class.py",
    klass="ABFEPipeline", fn="_run_dual_lambda_scheme",
    bind="dual", dataclass="PipelineState", decorator="@dataclass",
    dataclass_doc="""dual_lambda 方案跨相位状态（刀2 B 段收编，2026-09-20）。

26 个跨 P-A~P-E 五相位的名字（键/文件/预算/stage 结果/pilot 统计），
原地收编：初始化语句保留原位（dual.<field> = ...），每调用经
`dual = PipelineState()` 重建。字段默认 None 不可观测：21 个首写为
体顶层无条件直赋；expected_vanishing_ranges 条件首写后由 L13930 顶层
无条件重写覆盖；stage1/stage2 条件首写由 should_run_stage1/2 ≡ True
（无条件）保证 P-D 必然绑定——26/26 先写后读，全路径证明见头注释。
""",

    # 首写在嵌套块里、但已证明默认值仍不可观测的三个。引擎强制要求写理由。
    proven_conditional={
        "expected_vanishing_ranges":
            "条件首写后，由 P-C 段顶层无条件重写覆盖（同一方法内后续必达）",
        "stage1":
            "条件首写受 should_run_stage1 守卫，而该量在 P-A 无条件赋值；"
            "P-D 的读发生在同一守卫之下，故读时必已绑定",
        "stage2":
            "同 stage1，守卫量为 should_run_stage2",
    },
    names=[
        "_partition_criterion", "_path_record", "_stage1_preopt_key",
        "_stage2_preopt_key", "expected_vanishing_ranges", "optimized_lambdas_1",
        "optimized_lambdas_2", "preopt1_file", "preopt2_file",
        "should_run_stage1", "should_run_stage2", "stage1", "stage1_file",
        "stage1_key", "stage1_status", "stage2", "stage2_file", "stage2_key",
        "stage2_pilot_lambdas", "stage2_pilot_mean_dU_dlambda",
        "stage2_pilot_n_dU_dlambda_samples", "stage2_pilot_std_dU_dlambda",
        "stage2_states", "stage2_status", "window_ranges_1", "window_ranges_2",
    ],
)
