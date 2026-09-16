"""预热路径上三段积分的守卫**不对称** —— 无需 GPU。

## 这条测试钉的是什么

`run_all_windows` 的窗口预热依次跑三段动力学，它们用的守卫**不是同一个**：

| 段 | 调用 | 炸了会怎样 |
|---|---|---|
| 热化（10000 步） | `step_with_chunk_rollback` | **回滚这一小段 → 步长减半 → 重做**，减到 0.01 fs 仍炸才判失败 |
| 偏置爬坡 0.2→1.0（各 2000 步） | `step_with_chunk_rollback` | 同上 |
| **偏置学习循环**（`check_chunk` 步/轮） | **`guarded_step`** | **只把异常换个说法就往上抛** —— 整个窗口直接失败 |

`guarded_step` 的 docstring 写得很直白：「只改错误信息、**不改动力学**：该失败还是失败」。

## 为什么值得单独钉一条

2026-09-16 查 benchmark 的 11 次 `Particle coordinate is NaN` 时定位到：

- **10 次**是 2026-09-09 那一批（`brd4`/`cmet`/`cyclod` 共 10 个 run），全部死在偏置爬坡
  第一个非零档 `scale=0.2`，当时那里还是裸 `sim.step()`。守卫于 2026-09-12 15:50
  （commit `1769e71`）落地 ⟹ **那 10 次已修**，旧 traceback 只是留在追加式 `launch.log` 里。
- **剩下 1 次**是 `cyclod_ligand2/rep3`（2026-09-14 14:56，**守卫之后**），报的是
  `窗口 0 偏置学习：积分 250 步过程中出现非有限坐标/能量` —— 正是本文件钉的这段
  **没有回滚救援**的循环。诊断表里坐标已经到 3.7e9。

⚠️ **鬼影期（`bias_scale=0`）不是这一段的成因** —— 学习循环跑在 `bias_scale=1.0`
（`ibs_engine.py` 该循环之前有 `setParameter(..., 1.0)`）。`LR-06` 那条鬼影链路解释的是
爬坡**之前**的阶段，别把两者混成一件事。

## 这不是"缺了个守卫，补上就行"

学习循环每轮都在往 `sampler.energy_buffer` / `energy_history` /
`sampling_state_energy_history` 里追加，而回滚只还原 Context 的坐标/速度。
**回滚坐标却不回滚这些历史 = 偏置学习读到对不上的样本**，正是本仓反复栽的
「同一份状态两个副本不同步」。所以本测试**只钉现状、不主张改法**：
要改就得连采样器历史一起回滚，那是另一件事。

`ibs_engine.py` 自己的注释也写着：「warmup/learning 控制面没有 guard、也没有周期性
checkpoint」—— 是已知的，不是遗漏。
"""
import ast
import re
from pathlib import Path

import openmm
import pytest

from step_guard import guarded_step, step_with_chunk_rollback

_SRC = Path(__file__).resolve().parents[1] / "ibs_engine.py"


def _run_all_windows_calls():
    """`run_all_windows` 函数体内的 (守卫名, label 字面量) 列表。"""
    tree = ast.parse(_SRC.read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_all_windows"
    )
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in ("guarded_step", "step_with_chunk_rollback"):
            continue
        label = ""
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                label = arg.value
            elif isinstance(arg, ast.JoinedStr):
                label = "".join(
                    v.value for v in arg.values if isinstance(v, ast.Constant)
                )
        out.append((name, label, node.lineno))
    return out


def test_thermalization_and_bias_ramp_are_rollback_rescued():
    """热化与偏置爬坡必须是**可救**的那一档 —— 这是 09-09 那 10 个 run 的修复。"""
    calls = _run_all_windows_calls()
    labels = {lbl: nm for nm, lbl, _ in calls}

    thermalize = [nm for nm, lbl, _ in calls if "热化" in lbl]
    ramp = [nm for nm, lbl, _ in calls if "偏置预热" in lbl]

    assert thermalize, f"没找到热化那段积分；现有 label={list(labels)}"
    assert ramp, f"没找到偏置爬坡那段积分；现有 label={list(labels)}"
    assert set(thermalize) == {"step_with_chunk_rollback"}, (
        "热化退回成了不可救的守卫 —— 2026-09-09 那 10 个 run 就是这么炸穿的"
    )
    assert set(ramp) == {"step_with_chunk_rollback"}, (
        "偏置爬坡退回成了不可救的守卫 —— commit 1769e71 修的正是这里"
    )


def test_the_bias_learning_loop_is_deliberately_not_rescued():
    """偏置学习循环**没有**回滚救援。

    钉现状，不是主张它对：见本文件开头「这不是缺了个守卫」。
    哪天它真的接上回滚，这条会红 —— 那时请连 sampler 历史一起回滚再改断言。
    """
    calls = _run_all_windows_calls()
    learning = [(nm, ln) for nm, lbl, ln in calls if "偏置学习" in lbl]

    assert learning, "没找到偏置学习那段积分（label 改了？）"
    for name, lineno in learning:
        assert name == "guarded_step", (
            f"ibs_engine.py:{lineno} 的偏置学习改用了 {name}。"
            "若确实接上了回滚，必须同时回滚 sampler.energy_buffer / energy_history / "
            "sampling_state_energy_history，否则偏置学习会读到与构型对不上的样本。"
        )


def test_bias_scale_is_one_during_learning_so_this_is_not_the_ghost_period():
    """学习循环跑在满偏置下 ⟹ 它的 NaN **不是** `LR-06` 那条鬼影链路。

    `bias_scale` 乘在 Group-1 整个表达式外面（含配体↔环境软核），所以 `=0` 等于
    把配体关成鬼影。但那发生在爬坡**之前**；学习循环之前有一次显式 `=1.0`。
    """
    src = _SRC.read_text()
    lines = src.split("\n")
    learn_ln = next(
        ln for nm, lbl, ln in _run_all_windows_calls() if "偏置学习" in lbl
    )
    before = "\n".join(lines[max(0, learn_ln - 500):learn_ln])
    sets = re.findall(r'setParameter\(f?"\{self\.prefix\}_bias_scale",\s*([0-9.]+)\)', before)
    assert sets, "学习循环之前找不到任何 bias_scale 赋值"
    assert float(sets[-1]) == 1.0, (
        f"学习循环之前最后一次 bias_scale = {sets[-1]}，不是 1.0 —— "
        "若它变成 0，这段就落进 LR-06 的鬼影期，结论要重写"
    )


class _OneBadStep:
    """第一次 step() 抛 NaN，之后正常 —— 一次瞬态。"""

    def __init__(self):
        self.calls = 0
        self.finite = True
        sim = self

        class _I:
            dt = 0.002 * openmm.unit.picoseconds

            def getStepSize(self):
                return self.dt

            def setStepSize(self, dt):
                self.dt = dt

        class _S:
            def getPotentialEnergy(self):
                return (1.0 if sim.finite else float("nan")) * openmm.unit.kilojoule_per_mole

            def getForces(self, asNumpy=False):
                import numpy as np
                v = np.zeros((2, 3)) if sim.finite else np.full((2, 3), np.inf)
                return v * openmm.unit.kilojoule_per_mole / openmm.unit.nanometer

            def getPositions(self, asNumpy=False):
                import numpy as np
                return np.zeros((2, 3)) * openmm.unit.nanometer

        class _C:
            def getState(self, **kw):
                return _S()

            def setState(self, state):
                pass

        self.integrator = _I()
        self.context = _C()

    def step(self, n):
        self.calls += 1
        bad = self.calls == 1
        self.finite = not bad
        if bad:
            raise openmm.OpenMMException("Particle coordinate is NaN")


def test_the_same_transient_nan_is_survived_by_the_ramp_and_fatal_in_learning():
    """**这就是复现**：同一次瞬态 NaN，爬坡活下来，学习循环死。"""
    # 爬坡那一档：回滚 + 减半 + 重做 ⟹ 活
    ramp = _OneBadStep()
    step_with_chunk_rollback(ramp, 400, "窗口 0 偏置预热 scale=0.2", chunk=200)
    assert ramp.calls > 1, "回滚守卫应当重做过那一段"

    # 学习循环那一档：同样的瞬态 ⟹ 直接抛，窗口作废
    learn = _OneBadStep()
    with pytest.raises(RuntimeError) as exc:
        guarded_step(learn, 250, "窗口 0 偏置学习")
    assert "偏置学习" in str(exc.value)
    assert "非有限坐标/能量" in str(exc.value)
    assert learn.calls == 1, "guarded_step 不该重试 —— 它按设计只换错误信息"
