"""刀2 B 段第二波：往已有的 PipelineState 追加 2 个字段。

`stage1_states` 是**形参**（P-A 读参数值、P-B 重绑、P-D 读重绑值），收编后
构造点传初值；`opt_res` 原地收编。extend=True ⟹ 不新建类、不插构造语句。
"""
SPEC = dict(
    mode="absorb", name="pipeline_state2", in_place=True, extend=True,
    file="pipeline_orchestration/abfe_pipeline_class.py",
    klass="ABFEPipeline", fn="_run_dual_lambda_scheme",
    bind="dual", dataclass="PipelineState",
    names=["stage1_states", "opt_res"],
    seed_from_param=["stage1_states"],
    field_order=["stage1_states", "opt_res"],   # 追加顺序＝历史产物顺序

    # 首写在嵌套块里但默认值不可观测——机械判据管不到，理由必须写下来。
    proven_conditional={
        "opt_res":
            "首写在 P-B 的 resume 分支内；非 resume 路径由 P-D 的预优化调用"
            "无条件重绑，两条路径都先写后读",
        "stage1_states":
            "本刀前它是方法形参（必然已绑定），收编为构造点传初值 "
            "PipelineState(stage1_states=stage1_states)，语义等价",
    },
)
