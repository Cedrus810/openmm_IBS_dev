"""`stage2_autonomous_controller` 必须真的从 config 传到 `run_full_pipeline`。

`abfe_pipeline` 四处读 `kwargs.get("stage2_autonomous_controller", True)`，但
`runabfe.py` 的两个 `run_full_pipeline(...)` 调用点是**逐键显式传参**的，这个键
从来不在里面 ⟹ 写进 config 也不生效，自治循环无法关闭。

做「固定预算、不按中间结果提前停」的 A/B 对照时，自适应补帧本身就是要消除的
选择偏差（低读数的窗口才会被选中补帧 ⟹ 回归均值 + 自适应停止偏差），所以它必须能关。

这里按源码/AST 钉住"传了"，与仓库既有的
`test_executor_never_declines_silently.py` 等同一手法 —— 真跑一次 pipeline 太贵。
"""
import ast
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

KEY = "stage2_autonomous_controller"


def _run_full_pipeline_calls():
    tree = ast.parse(open(os.path.join(_ROOT, "runabfe.py"), encoding="utf-8").read())
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run_full_pipeline"):
            out.append(node)
    return out


def test_both_call_sites_pass_the_toggle():
    calls = _run_full_pipeline_calls()
    assert len(calls) == 2, f"预期复合物腿 + 溶剂腿两个调用点，实得 {len(calls)}"
    for c in calls:
        names = {kw.arg for kw in c.keywords if kw.arg}
        assert KEY in names, (
            f"run_full_pipeline (line {c.lineno}) 没传 {KEY} —— "
            "config 里写了也不会生效")


def test_the_default_keeps_the_controller_on():
    """不写这个键时行为必须逐字不变（默认 True）。"""
    calls = _run_full_pipeline_calls()
    for c in calls:
        kw = next(k for k in c.keywords if k.arg == KEY)
        # 形如 config.get("stage2_autonomous_controller", True)
        assert isinstance(kw.value, ast.Call), ast.dump(kw.value)
        assert kw.value.func.attr == "get", ast.dump(kw.value)
        default = kw.value.args[1]
        assert isinstance(default, ast.Constant) and default.value is True, (
            f"line {c.lineno}: 默认值必须是 True，实得 {ast.dump(default)}")


def test_the_pipeline_side_still_reads_the_same_key():
    """两侧键名必须对得上，否则这条线接了等于没接。"""
    src = open(os.path.join(_ROOT, "abfe_pipeline.py"), encoding="utf-8").read()
    assert f'kwargs.get("{KEY}"' in src, "abfe_pipeline 不再读这个键了，两侧对不上"
