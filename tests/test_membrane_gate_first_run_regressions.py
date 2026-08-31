"""P1-10：膜体系首跑必须能先预平衡、再判 §9 质量门。

## 缺陷是什么（已修，本文件现在是契约测试）

`run_full_pipeline()` 在函数开头**无条件**调用
`ensure_membrane_quality_gate_passed()`，而后者在没有 `pre_equilibration.dcd`
时直接 raise（门不能因轨迹缺失跳过 —— 这个语义本身是对的）。于是
fresh 膜输出目录、`run_equilibration=True`、还没有任何轨迹的首次运行：
质量门先消费一个尚未生成的轨迹 → RuntimeError → 膜体系永远进不了第一次
预平衡（`--no-boresch` 或直接调用 `run_full_pipeline` 时最容易撞上）。

## 修法（2026-08-30）

- 预平衡块**之前**只在预平衡轨迹已存在（复用场景）时判门 —— 消费既有
  轨迹前仍然 fail closed；
- 预平衡块**之后**无条件再判一次（`ensure_...` 幂等）：fresh 首跑刚产出
  轨迹（`pre_equilibrate()` 末尾已有 fail-fast 判门，这里幂等复用）、断点
  续传、或 `run_equilibration=False` 直接带坐标进来，都逃不过这个最终硬门，
  门失败（enforce 模式）不会进入任何 Stage 0/λ 窗口。

## 不要这样让本文件变绿

把 `ensure_membrane_quality_gate_passed` 的"轨迹缺失 ⟹ raise"改成"缺失 ⟹
跳过"——那正是 MEM-14 修掉的绕过路径；或者把最终硬门挪到任何 λ 窗口工作
之后。
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "abfe_pipeline.py"


def _compile_run_full_pipeline():
    tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"), filename=str(PIPELINE_PATH))
    pipeline_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ABFEPipeline"
    )
    method = next(
        node for node in pipeline_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_full_pipeline"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    from typing import Dict as _Dict, List as _List, Optional as _Optional, Tuple as _Tuple
    # 🔑 [2026-08-31] 命名空间先用**真实模块的全局表**打底，再叠上下面这几个
    # 刻意替换掉的名字。
    #
    # 原来这里只给一个手工列举的最小 namespace。那样每当 `run_full_pipeline()`
    # 开头新引用一个模块级名字（今天这轮就新增了 `_compute_ligand_net_charge`、
    # `LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E`、
    # `CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER` 等），本文件就会在
    # **到达它真正要守的膜质量门之前** NameError —— 表现成"P1-10 回归失败"，
    # 实际是测试脚手架缺名字，等于这道守卫被静默停用。
    # 用真实全局表打底之后，被 exec 的方法看到的名字与生产完全一致；`self` 仍然
    # 是本文件的 stub，所以控制流照旧由这里决定。
    namespace = {}
    try:  # pragma: no cover - 环境缺 openmm 时退回纯 stub 命名空间
        import abfe_pipeline as _real_pipeline_module

        namespace.update(vars(_real_pipeline_module))
    except Exception:
        pass
    namespace.update({
        "Dict": _Dict,
        "Optional": _Optional,
        "Any": object,
        "List": _List,
        "Tuple": _Tuple,
        "os": os,
        "sys": SimpleNamespace(argv=[]),
        "unit": SimpleNamespace(kelvin=object()),
        "json": __import__("json"),
    })
    exec(compile(module, str(PIPELINE_PATH), "exec"), namespace)
    return namespace["run_full_pipeline"]


class _StopAfterPreEquilibration(Exception):
    pass


def _make_pipeline(tmp_path, *, membrane, trajectory_exists):
    class _Pipeline:
        pass

    pipeline = _Pipeline()
    pipeline.environment_type = "membrane" if membrane else "soluble"
    pipeline.temperature = SimpleNamespace(value_in_unit=lambda _u: 300.0)
    pipeline.pressure = SimpleNamespace(value_in_unit=lambda _u: 1.0)
    pipeline.platform_name = "CPU"
    pipeline.output_dir = str(tmp_path / "output")
    pipeline.checkpoint_dir = str(tmp_path / "checkpoints")
    pipeline.ligand_indices = []
    # 🔑 [2026-08-31] 真实 CLI 在**构造之前**就把净电荷路线解析好并传进
    # `ABFEPipeline(charge_treatment=...)`，所以生产实例这个字段永远是已解析值。
    # `run_full_pipeline()` 开头那道 fail-closed 门（"直接调 API 的人也必须传
    # 已解析的 charge_treatment，否则带电配体会静默退回旧 co-annihilation 默认
    # 路径"）就是据此判定的。本 stub 不声明它的话，门会认为"未解析"并进入
    # `_compute_ligand_net_charge(self.system, ...)` 分支——而本文件是用 AST 抽出
    # 方法、在**手工命名空间**里 exec 的，模块级 helper 和 self.system 都不存在，
    # 于是在到达它要守的膜质量门之前就先崩了（P1-10 的守卫因此形同失效）。
    # 这里补的是"生产实例本来就有的状态"，不是绕过那道门。
    pipeline.charge_treatment = "neutral"
    pipeline._boresch_rebalance_done_this_process = False
    pipeline._pre_equilibration_done_this_process = False
    pipeline.enable_equilibration_convergence_stop = False
    pipeline.logs = []
    pipeline._log = pipeline.logs.append
    pipeline.get_device_strategy = lambda **kwargs: {"strategy": "cpu", "devices": [], "n_gpus": 0}
    pipeline._load_pipeline_state = lambda: {}
    pipeline.calls = []

    def _gate():
        """复刻真实语义：没有轨迹就 fail closed；其余按测试需要。"""
        raise RuntimeError("门不能因为轨迹缺失就跳过")

    pipeline.ensure_membrane_quality_gate_passed = _gate

    def _pre_equilibrate(**kwargs):
        pipeline.calls.append("pre_equilibrate")
        raise _StopAfterPreEquilibration

    pipeline.pre_equilibrate = _pre_equilibrate

    os.makedirs(pipeline.output_dir, exist_ok=True)
    if trajectory_exists:
        with open(os.path.join(pipeline.output_dir, "pre_equilibration.dcd"), "wb") as fh:
            fh.write(b"\x00" * 4096)
    return pipeline


def test_fresh_membrane_run_reaches_pre_equilibration_without_gate_crash(tmp_path):
    """fresh 膜目录（无 DCD）：不得在预平衡之前消费不存在的轨迹而崩掉。"""
    method = _compile_run_full_pipeline()
    pipeline = _make_pipeline(tmp_path, membrane=True, trajectory_exists=False)

    with pytest.raises(_StopAfterPreEquilibration):
        method(pipeline, run_equilibration=True, n_equil_steps=100, decoupling_scheme="dual_lambda")
    assert pipeline.calls == ["pre_equilibrate"]


def test_existing_trajectory_is_still_gated_before_consumption(tmp_path):
    """复用既有预平衡轨迹：消费之前门必须先判（门失败 ⟹ 不得进入预平衡消费）。"""
    method = _compile_run_full_pipeline()
    pipeline = _make_pipeline(tmp_path, membrane=True, trajectory_exists=True)

    with pytest.raises(RuntimeError, match="门"):
        method(pipeline, run_equilibration=True, n_equil_steps=100, decoupling_scheme="dual_lambda")
    assert pipeline.calls == []


def test_final_hard_gate_runs_after_the_pre_equilibration_block():
    """预平衡块之后必须有一个无条件的最终硬门调用。

    结构断言：run_full_pipeline 里 `ensure_membrane_quality_gate_passed` 的
    无条件调用必须出现在 `self.pre_equilibrate(` 之后 —— 否则 fresh 首跑
    产出的轨迹可能永远不被判门。
    """
    tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"), filename=str(PIPELINE_PATH))
    pipeline_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ABFEPipeline"
    )
    method = next(
        node for node in pipeline_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_full_pipeline"
    )

    # 父链表：ast.walk 不带父指针，手工建一份，用于判断调用是否被 If 守卫。
    parents = {}
    for parent in ast.walk(method):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    def _is_guarded_by_trajectory_existence(node):
        ancestor = parents.get(node)
        while ancestor is not None:
            if isinstance(ancestor, ast.If):
                test_src = ast.unparse(ancestor.test)
                if "pre_equilibration.dcd" in test_src:
                    return True
            ancestor = parents.get(ancestor)
        return False

    unconditional_gate_lines = []
    pre_equil_lines = []
    for node in ast.walk(method):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "ensure_membrane_quality_gate_passed":
                if not _is_guarded_by_trajectory_existence(node):
                    unconditional_gate_lines.append(node.lineno)
            elif node.func.attr == "pre_equilibrate":
                pre_equil_lines.append(node.lineno)

    assert pre_equil_lines, "run_full_pipeline 必须调用 pre_equilibrate"
    assert unconditional_gate_lines, "预平衡块之后必须存在无条件的最终硬门"
    assert min(unconditional_gate_lines) > max(pre_equil_lines), (
        "最终硬门必须位于预平衡块之后（P1-10）：fresh 首跑的轨迹产出必须先于判门，"
        "而判门必须先于任何 λ 窗口工作"
    )
