"""刀 P-C：窗口划分段 → `_partition_windows`。

原特化执行器 `cuts/_partition_windows_cut.py`（226 行）的声明式等价物。
锚点为结构谓词，不含任何行号/下标——主线小修只会让 `slice_expect` 报漂移，
不会让定位失败；真改了结构则 fail-closed。
"""
import ast

from cutter import assign_attr, if_test

SPEC = dict(
    mode="extract",
    name="partition_windows",

    file="pipeline_orchestration/abfe_pipeline_class.py",
    klass="ABFEPipeline",
    fn="_run_dual_lambda_scheme",
    new="_partition_windows",

    # 结构锚：P-C 段第一条 / P-C 之后第一条（不搬）
    start=assign_attr("dual", "_partition_criterion"),
    end=if_test("dual.should_run_stage1"),

    # 2026-09-20 快照实测值，只作漂移报告，不作定位
    slice_expect=(26, 41),
    stmts=15,
    span=(185, 240),   # 09-20 快照 193；09-23 主线 212（段内小修）

    params=["_stage2_protocol_key", "_vanishing_range_kwargs", "dual",
            "kwargs", "resume"],
    # e：无 Store，except handler 局部全覆盖；f：with 局部，Store/Load 同块内
    exempt={"e", "f"},
    extra_visible=("dual",),

    indent=8,
    doc=('        """P-C 窗口划分段—— 刀2 内部细分自 _run_dual_lambda_scheme 原样搬出。\n'
         '\n'
         '        纯搬运（verify_extract --block fn --slice 26:41 逐节点等价）；跨界\n'
         '        变量按原名作 keyword-only 形参（e/f 为段内自覆盖临时名，豁免）。\n'
         '        """\n'),
)
