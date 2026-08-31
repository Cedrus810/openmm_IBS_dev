"""回归：REMD 不得再默认预防性回退 CPU（2026-07-27 事故）。

现场
----
完整 dual_lambda 链路的 decharging 阶段"卡住"23 分钟，12 个 `decharging_rep*.dcd`
全是 0 字节，`pipeline.log` 最后一行是阶段表头、**没有任何 traceback**。终端里
（不在日志里）有一行：

    ⚠️ REMD replica 数超过 GPU 常驻 Context 上限 (12>1)；
       为避免单 GPU OOM，在创建任何 GPU Context 前回退 CPU。

两个独立缺陷叠在一起：

1. **默认值让 GPU 路径不可达。** `max_resident_contexts` 对 CUDA/OPENCL 默认 **1**，
   而当前交换实现天生要求每个 replica 同时持有独立 Context
   （`context_residency_mode = "all_resident"`）——于是 `n_replicas > 1` 恒成立，
   **任何** REMD 都会静默回退 CPU。
2. **参数只在一条路径上接通。** `--charging-max-resident-contexts` 只被
   `--only-complex-charging` 消费；完整 dual_lambda 链路没有透传，所以永远拿默认值。
   （这也解释了为什么 `--only-complex-charging` 那轮能留在 CUDA：manifest 记录
   `platform_name: CUDA`、`platform_fallback_reason: None`。）

代价极不对称：CPU 回退慢约两个数量级（GPU ~24 分钟跑完 500 轮交换；CPU 上 23 分钟
连第一个 DCD 帧都没写出来），而且**决定只 print 到终端**，归档日志里查不到——
表现得完全像卡死。而真正的 GPU OOM 本来就有优雅处理：`_build_replicas` 的 `except`
分支会 `_clear_replica_contexts()` 后回退 CPU 重建。OOM 是响亮且立即的，慢 100 倍是静默的。
"""

import ast
import inspect
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[1]


def _init_source():
    from ibs_engine import REMDManager
    return inspect.getsource(REMDManager.__init__)


def test_default_no_longer_hardcodes_one_resident_context():
    """默认分支不得再对 CUDA/OPENCL 取 1——那让 GPU 路径永远不可达。"""
    src = _init_source()
    tree = ast.parse(inspect.cleandoc(src))
    # 找 `if max_resident_contexts is None:` 分支里的赋值
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                and test.left.id == "max_resident_contexts"):
            continue
        assigns = [n for n in ast.walk(node) if isinstance(n, ast.Assign)]
        assert assigns, "默认分支里没有赋值？"
        for a in assigns:
            # 不得是 `1` 或含 `1` 的三元表达式
            consts = [
                n.value for n in ast.walk(a.value)
                if isinstance(n, ast.Constant) and isinstance(n.value, int)
            ]
            assert 1 not in consts, (
                "默认 max_resident_contexts 又出现常量 1。交换实现要求所有 replica "
                "同时常驻，取 1 等于让每一次 REMD 都静默回退 CPU（慢 ~100×）。"
                "真实 OOM 由 _build_replicas 的构建期回退兜底。"
            )
        return
    pytest.fail("找不到 `max_resident_contexts is None` 的默认分支")


def test_build_time_oom_fallback_still_exists():
    """放宽预防性上限的前提是构建期 OOM 回退仍在——否则真 OOM 就没人兜了。"""
    from ibs_engine import REMDManager

    src = inspect.getsource(REMDManager._build_replicas)
    assert "_clear_replica_contexts" in src, "构建失败后必须释放已建 Context"
    assert "_is_gpu_context_failure" in src, "必须判定是否 GPU 失败再决定回退"
    assert 'self.platform_name = "CPU"' in src, "构建期回退路径不见了"


def test_explicit_low_limit_still_falls_back():
    """显式传小值（小显存机器）必须仍然回退——这条通路不能被顺手删掉。"""
    src = _init_source()
    assert "cpu_fallback_bounded_gpu_contexts" in src, "显式回退分支不见了"
    assert 'self.platform_name = "CPU"' in src


def test_fallback_is_logged_not_just_printed():
    """回退决定必须进 logger，不能只 print 到终端。

    这是本次调试痛苦的直接来源：pipeline.log 里完全看不到这个决定，
    只能看到"阶段开始"然后 23 分钟空白。
    """
    src = _init_source()
    assert "logger.warning" in src, (
        "回退只 print 到 stdout 的话，归档 pipeline.log 里查不到——"
        "一个让整阶段慢 100 倍的决定必须留在日志里"
    )


def test_fallback_records_a_machine_readable_reason():
    """`platform_fallback_reason` 必须被设上，否则 exchange_diagnostics 会谎报 None。"""
    src = _init_source()
    assert "platform_fallback_reason" in src

    # 且 __init__ 末尾不得无条件把它重置成 None
    assert "if getattr(self, \"platform_fallback_reason\", None) is None:" in src, (
        "__init__ 里如果无条件 `platform_fallback_reason = None`，"
        "会把显式回退分支刚写进去的原因抹掉，落盘诊断就成了 CUDA+None 的假象"
    )


# ---------------------------------------------------------------------------
# 参数透传：完整 dual_lambda 链路必须能控制这个预算
# ---------------------------------------------------------------------------


def test_all_dual_lambda_stage_calls_pass_the_context_budget():
    """三个 `_run_dual_lambda_stage` 调用点都必须透传上限。

    漏掉任一个，那条 stage 就会拿默认值——修复前正是因为完整链路一个都没传。
    """
    src = (REPO_ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(src, filename="abfe_pipeline.py")
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_run_dual_lambda_stage"
    ]
    assert len(calls) >= 3, f"只找到 {len(calls)} 个调用点"
    missing = [
        c.lineno for c in calls
        if "remd_max_resident_contexts" not in {kw.arg for kw in c.keywords if kw.arg}
    ]
    assert not missing, (
        f"abfe_pipeline.py 第 {missing} 行的 _run_dual_lambda_stage 没透传 "
        "remd_max_resident_contexts —— 那条 stage 会拿默认值"
    )


def test_runabfe_wires_the_config_into_both_legs():
    """复合物腿与溶剂腿的 run_full_pipeline 都必须把配置值传下去。"""
    src = (REPO_ROOT / "runabfe.py").read_text(encoding="utf-8")
    tree = ast.parse(src, filename="runabfe.py")
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "run_full_pipeline"
    ]
    assert len(calls) >= 2, f"只找到 {len(calls)} 个 run_full_pipeline 调用点"
    missing = [
        c.lineno for c in calls
        if "charging_max_resident_contexts" not in {kw.arg for kw in c.keywords if kw.arg}
    ]
    assert not missing, (
        f"runabfe.py 第 {missing} 行的 run_full_pipeline 没传 "
        "charging_max_resident_contexts —— 修复前完整链路两条腿都没传"
    )
