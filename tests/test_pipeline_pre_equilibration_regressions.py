"""Executable regressions for the pre-equilibration control plane.

The production modules import OpenMM at module import time, while the default
test image intentionally does not require OpenMM.  These tests therefore
compile the two methods from the checked-in source and execute them with small
fakes.  This exercises the relevant branches and call arguments rather than
only checking source strings.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from datetime import datetime

import pytest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "abfe_pipeline.py"


def _pipeline_module():
    """真实的 abfe_pipeline 模块（本文件其余部分刻意只做源码级编译）。"""
    import abfe_pipeline

    return abfe_pipeline


def _compile_pipeline_method(name: str):
    tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"), filename=str(PIPELINE_PATH))
    pipeline_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ABFEPipeline"
    )
    method = next(
        node
        for node in pipeline_class.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Dict": Dict,
        "Optional": Optional,
        "Any": object,
        "List": list,
        "Tuple": tuple,
    }
    exec(compile(module, str(PIPELINE_PATH), "exec"), namespace)
    return namespace[name]


def _compile_pipeline_top_level(name: str):
    tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"), filename=str(PIPELINE_PATH))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Any": Any, "Dict": Dict, "Optional": Optional, "List": list, "Tuple": tuple}
    exec(compile(module, str(PIPELINE_PATH), "exec"), namespace)
    return namespace[name]


class _FakeQuantity:
    def __init__(self, value):
        self.value = value

    def value_in_unit(self, _unit):
        return self.value


class _FakeState:
    def getPositions(self):
        return "minimized_positions"

    def getPeriodicBoxVectors(self):
        return "minimized_box"

    def getPotentialEnergy(self):
        return _FakeQuantity(-12.5)


class _FakeContext:
    def __init__(self):
        self.calls = []

    def setPositions(self, positions):
        self.calls.append(("setPositions", positions))

    def setPeriodicBoxVectors(self, *vectors):
        self.calls.append(("setPeriodicBoxVectors", vectors))

    def setVelocitiesToTemperature(self, temperature, seed):
        self.calls.append(("setVelocitiesToTemperature", temperature, seed))

    def getState(self, **kwargs):
        self.calls.append(("getState", kwargs))
        return _FakeState()


class _FakeSimulation:
    instances = []
    load_succeeds = False

    def __init__(self, *_args):
        self.context = _FakeContext()
        self.reporters = []
        self.currentStep = 0
        self.load_calls = []
        self.step_calls = []
        self.__class__.instances.append(self)

    def loadCheckpoint(self, path):
        self.load_calls.append(path)
        if self.__class__.load_succeeds:
            self.currentStep = 2
            return
        raise RuntimeError("corrupt checkpoint")

    def step(self, n_steps):
        self.step_calls.append(n_steps)
        self.currentStep += int(n_steps)


class _FakeIntegrator:
    def __init__(self, *_args):
        self.seed = None

    def setRandomNumberSeed(self, seed):
        self.seed = seed


class _FakeReporter:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.__class__.instances.append(self)


def _pre_equilibration_namespace():
    unit = SimpleNamespace(
        picosecond=1.0,
        kelvin=object(),
        kilojoule_per_mole=object(),
    )

    class _Serializer:
        @staticmethod
        def serialize(_system):
            return "system-xml"

        @staticmethod
        def deserialize(_xml):
            return SimpleNamespace(thisown=0, getNumParticles=lambda: 2)

    class _Platform:
        @staticmethod
        def getPlatformByName(name):
            return name

    openmm = SimpleNamespace(
        LangevinMiddleIntegrator=_FakeIntegrator,
        Platform=_Platform,
        LocalEnergyMinimizer=SimpleNamespace(
            minimize=lambda context, maxIterations: context.calls.append(
                ("minimize", maxIterations)
            )
        ),
    )
    app = SimpleNamespace(
        Simulation=_FakeSimulation,
        DCDReporter=_FakeReporter,
        CheckpointReporter=_FakeReporter,
        StateDataReporter=_FakeReporter,
    )
    return {
        "unit": unit,
        "XmlSerializer": _Serializer,
        "openmm": openmm,
        "app": app,
        "PRE_EQUILIBRATION_TIMESTEP_PS": 0.002,
        "PRE_EQUILIBRATION_TRAJ_INTERVAL_STEPS": 10_000,
        "MEMBRANE_EQUILIBRATION_VELOCITY_SEED": 123,
        "ENVIRONMENT_TYPE_MEMBRANE": "membrane",
        "MEMBRANE_POST_MINIMIZATION_MAX_FORCE_KJ_PER_MOL_NM": 1.0,
        "_build_platform_props": lambda _name: ("CPU", {}),
        # [0831issue P2] `pre_equilibrate` 现在用 base 平台名判断要不要显式释放
        # CUDA Context（裸 `== "CUDA"` 匹配不上 "CUDA:1"）。这里注入**真实**的
        # 解析器而不是 stub：它是纯字符串函数、不碰 OpenMM，编进来能顺带保证
        # 生产代码与本测试用的是同一份 spec 语法。
        "_split_platform_spec": _compile_pipeline_top_level("_split_platform_spec"),
        # [2026-09-01] `pre_equilibrate` 的 resume 分支改走跨平台迁移入口。
        # 这里注入**真实**实现而不是 stub：本组用例要验的正是"loadCheckpoint 被
        # 真的调用了、失败后走完整 fresh 初始化"，而真实入口的第一件事就是调
        # `simulation.loadCheckpoint(path)`；只有当报错是"platform 不匹配"时才
        # 转迁移，其它异常原样抛给外层。注入 stub 反而会把要验的行为遮掉。
        "load_checkpoint_with_platform_migration": _pipeline_module().load_checkpoint_with_platform_migration,
        "ensure_barostat_for_protocol": lambda *args, **kwargs: {
            "action": "reused",
            "barostat_class": "MonteCarloBarostat",
        },
        "assert_starting_state_is_sane": lambda *args, **kwargs: None,
        "_pre_equilibration_fingerprint": lambda *args, **kwargs: "fp",
        "json": json,
        "os": __import__("os"),
        "gc": __import__("gc"),
    }


def test_checkpoint_load_failure_runs_the_complete_fresh_initialization(tmp_path):
    """A failed load must initialize the Context before any step is taken."""

    _FakeSimulation.instances = []
    method = _compile_pipeline_method("pre_equilibrate")
    namespace = _pre_equilibration_namespace()
    namespace.update({"Dict": Dict, "Optional": Optional})

    # The fingerprint matches, so this specifically exercises loadCheckpoint's
    # failure branch rather than the earlier identity rejection branch.
    checkpoint = tmp_path / "pre_equil.chk"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path.parent / "unused").mkdir(exist_ok=True)

    class _Pipeline:
        # 2026-09-01：`pre_equilibrate` 改走 `pre_equilibration_identity_fingerprint()`
        # 统一入口（见 test_platform_routing_regressions 里那组用例）。本组用例测的是
        # checkpoint/DCD 行为、不是指纹内容，所以沿用与 `_pre_equilibration_fingerprint`
        # 同一个桩值 "fp"，与下面写进磁盘的 fingerprint.json 对齐。
        def pre_equilibration_identity_fingerprint(self, requested_steps):
            return "fp"

    pipeline = _Pipeline()
    pipeline.output_dir = str(tmp_path)
    pipeline.checkpoint_dir = str(tmp_path)
    pipeline.system = object()
    pipeline.topology = object()
    pipeline.positions = "input_positions"
    pipeline.box_vectors = ("a", "b", "c")
    pipeline.ligand_indices = []
    pipeline.temperature = _FakeQuantity(300.0)
    pipeline.pressure = _FakeQuantity(1.0)
    pipeline.barostat_protocol = {}
    pipeline.environment_type = "soluble"
    pipeline.platform_name = "CPU"
    pipeline.charge_treatment = "neutral"
    pipeline.seed_ledger = object()  # exercise the velocity initialization path
    pipeline._pbc_integrity_repaired = True
    pipeline._pre_equilibration_done_this_process = False
    logs = []
    pipeline._log = logs.append
    pipeline._log_vram = lambda *args, **kwargs: None
    pipeline.repair_pbc_molecule_integrity = lambda **kwargs: None
    pipeline._seed_for = lambda *args: 7
    pipeline._update_stage_status = lambda *args, **kwargs: None

    # Compile the method with production names plus the test fakes.
    method.__globals__.update(namespace)
    fp_path = tmp_path / "pre_equilibration_fingerprint.json"
    fp_path.write_text(json.dumps({"fingerprint": "fp"}), encoding="utf-8")
    result = method(
        pipeline,
        n_steps=3,
        save_traj=False,
        resume=True,
        enable_convergence_stop=False,
    )

    simulation = _FakeSimulation.instances[-1]
    assert simulation.load_calls == [str(checkpoint)]
    assert ("setPositions", "input_positions") in simulation.context.calls
    assert ("setPeriodicBoxVectors", ("a", "b", "c")) in simulation.context.calls
    assert ("minimize", 1000) in simulation.context.calls
    assert ("setVelocitiesToTemperature", pipeline.temperature, 7) in simulation.context.calls
    assert simulation.step_calls == [3]
    assert result["resumed"] is False
    assert result["total_steps"] == 3
    assert any("Checkpoint 加载失败" in message for message in logs)


def test_successful_checkpoint_resume_starts_a_new_segment_without_appending_old_dcd(tmp_path):
    """A possibly-leading old DCD is archived; the new reporter never appends it."""

    _FakeSimulation.instances = []
    _FakeSimulation.load_succeeds = True
    _FakeReporter.instances = []
    method = _compile_pipeline_method("pre_equilibrate")
    namespace = _pre_equilibration_namespace()
    helper = _compile_pipeline_top_level("_begin_pre_equilibration_resume_segment")
    namespace.update(
        {
            "Any": Any,
            "Dict": Dict,
            "Optional": Optional,
            "datetime": datetime,
            "_atomic_write_json": lambda path, payload: Path(path).write_text(
                json.dumps(payload), encoding="utf-8"
            ),
            "PRE_EQUILIBRATION_SEGMENT_PROTOCOL_VERSION": 1,
            "PRE_EQUILIBRATION_SEGMENT_MANIFEST": "pre_equilibration_segments.json",
        }
    )
    helper.__globals__.update(namespace)
    helper.__globals__["_pre_equilibration_segment_manifest_path"] = (
        lambda output_dir: str(Path(output_dir) / "pre_equilibration_segments.json")
    )
    namespace["_begin_pre_equilibration_resume_segment"] = helper
    method.__globals__.update(namespace)

    checkpoint = tmp_path / "pre_equil.chk"
    checkpoint.write_bytes(b"opaque checkpoint")
    old_dcd = tmp_path / "pre_equilibration.dcd"
    old_dcd.write_bytes(b"old dcd bytes that must remain unchanged")
    old_bytes = old_dcd.read_bytes()
    (tmp_path / "pre_equilibration_fingerprint.json").write_text(
        json.dumps({"fingerprint": "fp"}), encoding="utf-8"
    )

    class _Pipeline:
        # 2026-09-01：`pre_equilibrate` 改走 `pre_equilibration_identity_fingerprint()`
        # 统一入口（见 test_platform_routing_regressions 里那组用例）。本组用例测的是
        # checkpoint/DCD 行为、不是指纹内容，所以沿用与 `_pre_equilibration_fingerprint`
        # 同一个桩值 "fp"，与下面写进磁盘的 fingerprint.json 对齐。
        def pre_equilibration_identity_fingerprint(self, requested_steps):
            return "fp"

    pipeline = _Pipeline()
    pipeline.output_dir = str(tmp_path)
    pipeline.checkpoint_dir = str(tmp_path)
    pipeline.system = object()
    pipeline.topology = object()
    pipeline.positions = "input_positions"
    pipeline.box_vectors = ("a", "b", "c")
    pipeline.ligand_indices = []
    pipeline.temperature = _FakeQuantity(300.0)
    pipeline.pressure = _FakeQuantity(1.0)
    pipeline.barostat_protocol = {}
    pipeline.environment_type = "soluble"
    pipeline.platform_name = "CPU"
    pipeline.charge_treatment = "neutral"
    pipeline.seed_ledger = None
    pipeline._pbc_integrity_repaired = True
    pipeline._pre_equilibration_done_this_process = False
    pipeline._log = lambda message: None
    pipeline._log_vram = lambda *args, **kwargs: None
    pipeline.repair_pbc_molecule_integrity = lambda **kwargs: None
    pipeline._seed_for = lambda *args: None
    pipeline._update_stage_status = lambda *args, **kwargs: None

    result = method(
        pipeline,
        n_steps=3,
        save_traj=True,
        resume=True,
        enable_convergence_stop=False,
    )

    simulation = _FakeSimulation.instances[-1]
    assert simulation.load_calls == [str(checkpoint)]
    assert simulation.step_calls == [1]
    dcd_reporter = next(
        reporter
        for reporter in _FakeReporter.instances
        if reporter.args and str(reporter.args[0]).endswith("pre_equilibration.dcd")
    )
    assert dcd_reporter.kwargs["append"] is False
    archived = tmp_path / "pre_equilibration.segment-0001.dcd"
    assert archived.read_bytes() == old_bytes
    assert not old_dcd.exists() or result["trajectory_file"] == str(old_dcd)
    manifest = json.loads((tmp_path / "pre_equilibration_segments.json").read_text())
    active = manifest["active_segment"]
    assert active["start_checkpoint_step"] == 2
    assert active["expected_end_step"] == 3
    assert active["trajectory"] == "pre_equilibration.dcd"
    assert result["trajectory_file"] == str(old_dcd)
    _FakeSimulation.load_succeeds = False


class _StopAfterPreEquilibration(Exception):
    pass


def test_run_full_pipeline_forwards_explicit_equilibration_budget_and_records_identity(tmp_path):
    """The public budget reaches pre_equilibrate and the run identity."""

    method = _compile_pipeline_method("run_full_pipeline")
    method.__globals__.update(
        {
            "CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER": "co_alchemical_charge_transfer",
            "unit": SimpleNamespace(kelvin=object()),
            "sys": SimpleNamespace(argv=[]),
            "os": __import__("os"),
        }
    )

    class _Pipeline:
        # 2026-09-01：`pre_equilibrate` 改走 `pre_equilibration_identity_fingerprint()`
        # 统一入口（见 test_platform_routing_regressions 里那组用例）。本组用例测的是
        # checkpoint/DCD 行为、不是指纹内容，所以沿用与 `_pre_equilibration_fingerprint`
        # 同一个桩值 "fp"，与下面写进磁盘的 fingerprint.json 对齐。
        def pre_equilibration_identity_fingerprint(self, requested_steps):
            return "fp"

    pipeline = _Pipeline()
    pipeline.temperature = _FakeQuantity(300.0)
    pipeline.pressure = _FakeQuantity(1.0)
    pipeline.platform_name = "CPU"
    pipeline.charge_treatment = "neutral"
    pipeline.output_dir = str(tmp_path / "output")
    pipeline.checkpoint_dir = str(tmp_path / "checkpoints")
    pipeline.ligand_indices = []
    pipeline._boresch_rebalance_done_this_process = False
    pipeline._pre_equilibration_done_this_process = False
    pipeline.enable_equilibration_convergence_stop = False
    pipeline.logs = []
    pipeline._log = pipeline.logs.append
    pipeline.ensure_membrane_quality_gate_passed = lambda: None
    pipeline.get_device_strategy = lambda **kwargs: {
        "strategy": "cpu",
        "devices": [],
        "n_gpus": 0,
    }
    pipeline._load_pipeline_state = lambda: {}
    captured = {}

    def _pre_equilibrate(**kwargs):
        captured.update(kwargs)
        raise _StopAfterPreEquilibration

    pipeline.pre_equilibrate = _pre_equilibrate

    with pytest.raises(_StopAfterPreEquilibration):
        method(
            pipeline,
            run_equilibration=True,
            n_equil_steps=1234,
            decoupling_scheme="dual_lambda",
        )

    assert captured["n_steps"] == 1234
    assert pipeline._last_run_config["n_equil_steps"] == 1234
