"""[0831issue] REMDManager 的 DCD append 模式契约。

`REMDManager.run()` 用 `_steps_completed > 0` 决定 DCDReporter 的 append 模式。这个
计数器只在 `__init__` 归零、只在 `run()` 内累加，**从不从盘上恢复**，所以它表达的是
"本实例本次生命周期跑过多少步"，不是"这个 output_dir 里已有多少步轨迹"。两条推论：

* 新进程 + 盘上已有同名 DCD → append=False → 截断重写。这是**正确**的（调用方只有在
  判定缓存不可用时才走到这里，随后从第 0 步跑满 n_steps），原来的问题只是它完全静默。
* 同一实例换 `stage_name` 二次 `run()` → append=True 但新 DCD 不存在 → DCDReporter 以
  `'DCDReporter' object has no attribute '_out'` 崩掉，真实异常还被 `__del__` 的二次
  异常盖住。子类 `BoreschAttachmentREMDManager` 靠 `begin_production_stage()` 规避；
  该方法现已上提到基类，且 run() 对这种状态不一致直接 fail closed 并给出可读报错。

这里不建真实 REMD（要 GPU/体系），只在最小 stub 上驱动 `run()` 开头那段文件契约。
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

pytest.importorskip("openmm")

import ibs_engine  # noqa: E402


class _StubREMD(ibs_engine.REMDManager):
    """绕过 __init__（要真实 System/Context），只装 run() 那段需要的字段。"""

    def __init__(self, output_dir, n_replicas=2, steps_completed=0):
        self.output_dir = str(output_dir)
        self.n_replicas = int(n_replicas)
        self._steps_completed = int(steps_completed)
        self._is_warmed_up = True
        self.contexts = []
        self.integrators = []
        self.topology = None
        self.platform_name = "CPU"
        self.platform_fallback_reason = None
        self._state_to_context = list(range(n_replicas))
        self._context_to_state = list(range(n_replicas))


def _traj_paths(tmp_path, stage_name, n_replicas=2):
    return [
        os.path.join(str(tmp_path), f"{stage_name}_rep{i}.dcd")
        for i in range(n_replicas)
    ]


def test_append_requested_but_file_missing_fails_closed(tmp_path):
    """平衡段跑完（_steps_completed>0）后换 stage 直接 run() → 可读报错，不是 _out。"""
    mgr = _StubREMD(tmp_path, steps_completed=5000)
    with pytest.raises(RuntimeError) as exc:
        mgr.run(n_steps=0, exchange_interval=1000, stage_name="complex")
    msg = str(exc.value)
    assert "begin_production_stage" in msg
    assert "append" in msg
    assert "_out" not in msg


def test_begin_production_stage_on_base_class_makes_the_switch_work(tmp_path):
    """基类现在也有 begin_production_stage()；归零后新 stage 以写模式建立。"""
    mgr = _StubREMD(tmp_path, steps_completed=5000)
    assert hasattr(ibs_engine.REMDManager, "begin_production_stage")
    mgr.begin_production_stage()
    assert mgr._steps_completed == 0
    # n_steps=0 → 没有交换轮、没有余数步，run() 只建 reporter 再写诊断 JSON。
    traj_files = mgr.run(n_steps=0, exchange_interval=1000, stage_name="complex")
    assert traj_files == _traj_paths(tmp_path, "complex")
    for path in traj_files:
        assert os.path.exists(path)


def test_subclass_begin_production_stage_still_clears_state_history():
    """子类覆盖必须 super() 掉基类的归零，同时保留自己清 state_history 的行为。"""
    sub = ibs_engine.BoreschAttachmentREMDManager
    assert "begin_production_stage" in vars(sub)
    obj = object.__new__(sub)
    obj._steps_completed = 1234
    obj.state_history = [[0, 1], [1, 0]]
    obj.begin_production_stage()
    assert obj._steps_completed == 0
    assert obj.state_history == []


def test_fresh_process_overwrite_is_announced_not_silent(tmp_path, capsys):
    """_steps_completed==0 且盘上已有 DCD → 仍然截断重写（正确），但必须有日志。"""
    stage = "complex"
    for path in _traj_paths(tmp_path, stage):
        with open(path, "wb") as handle:
            handle.write(b"stale-frames")

    mgr = _StubREMD(tmp_path, steps_completed=0)
    mgr.run(n_steps=0, exchange_interval=1000, stage_name=stage)

    out = capsys.readouterr().out
    for path in _traj_paths(tmp_path, stage):
        assert os.path.basename(path) in out
    assert "写模式重建" in out
    # 行为不变：旧内容确实被覆盖（DCD 头部不会是原来那串字节）。
    with open(_traj_paths(tmp_path, stage)[0], "rb") as handle:
        assert handle.read(12) != b"stale-frames"


def test_no_cross_process_resume_path_exists_for_steps_completed():
    """钉住"截断是正确的"这个前提：_steps_completed 没有任何从盘恢复的入口。

    一旦有人加了跨进程续算（从 checkpoint 恢复 _steps_completed），上面
    `test_fresh_process_overwrite_is_announced_not_silent` 断言的"覆盖是对的"就不再
    成立，必须回来重新设计 append 判据。这条测试就是那个提醒。
    """
    import re

    with open(ibs_engine.__file__, encoding="utf-8") as handle:
        source = handle.read()
    assignments = re.findall(r"self\._steps_completed\s*=\s*([^\n]+)", source)
    # 只允许"归零"这一种赋值；`+=` 的累加不在此列（正则要求 `=` 前无 `+`）。
    assert assignments, "找不到 _steps_completed 的赋值点，测试本身失效了"
    for rhs in assignments:
        assert rhs.strip() in ("0", "int(steps_completed)"), (
            f"_steps_completed 出现了非归零赋值 {rhs!r}——若这是跨进程续算，"
            "run() 里的 append 判据与'截断旧 DCD 是正确的'这个前提都要重新审"
        )
