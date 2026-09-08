from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "abfe_pipeline.py"
PREOPT = ROOT / "abfe_preoptimizer.py"


def _functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in selected} == set(names)
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def test_platform_property_routing_covers_cpu_cuda_and_explicit_devices():
    ns = _functions(
        PIPELINE,
        {"_split_platform_spec", "_build_platform_props"},
        {"Tuple": Tuple, "Optional": Optional, "Dict": Dict, "shutil": SimpleNamespace(which=lambda _: None)},
    )
    build = ns["_build_platform_props"]
    assert build("CPU") == ("CPU", {})
    assert build("CUDA") == ("CUDA", {"Precision": "mixed"})
    assert build("CUDA:1") == (
        "CUDA",
        {"Precision": "mixed", "DeviceIndex": "1"},
    )
    assert build("OpenCL:2") == (
        "OpenCL",
        {"Precision": "mixed", "DeviceIndex": "2"},
    )


def test_context_creation_retries_cpu_only_after_real_requested_context_failure():
    attempts = []

    class _Platform:
        @staticmethod
        def getPlatformByName(name):
            return name

    def _context(_system, integrator, platform, props):
        attempts.append((integrator, platform, dict(props)))
        if platform == "CUDA":
            raise RuntimeError("device unavailable")
        return "cpu-context"

    ns = _functions(
        PIPELINE,
        {"_create_context_with_local_cpu_fallback"},
        {
            "openmm": SimpleNamespace(Platform=_Platform, Context=_context),
            "_build_platform_props": lambda spec: (
                "CUDA",
                {"Precision": "mixed", "DeviceIndex": spec.split(":")[1]},
            ),
        },
    )
    created = []
    result = ns["_create_context_with_local_cpu_fallback"](
        object(), lambda: created.append(object()) or created[-1], "CUDA:1", lambda _: None
    )
    assert result[0] == "cpu-context"
    assert result[2:] == ("CPU", {})
    assert attempts[0][1:] == ("CUDA", {"Precision": "mixed", "DeviceIndex": "1"})
    assert attempts[1][1:] == ("CPU", {})
    assert attempts[0][0] is not attempts[1][0]


def test_geodesic_optimizer_routes_cuda_device_through_shared_helper():
    tree = ast.parse(PREOPT.read_text(encoding="utf-8"), filename=str(PREOPT))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "optimize_2d_geodesic_path"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    captured = {}

    class _Softcore:
        @staticmethod
        def optimize_alpha(_n):
            return {}

        @staticmethod
        def from_dict(_params):
            return object()

    class _Context:
        def __init__(self, _system, _integrator, platform, props):
            captured["context"] = (platform, dict(props))

        def setPositions(self, _positions):
            pass

        def setPeriodicBoxVectors(self, *_box):
            pass

    ns = {
        "Any": object,
        "Dict": Dict,
        "Optional": Optional,
        "List": list,
        "Tuple": Tuple,
        "np": np,
        "print": lambda *args, **kwargs: None,
        "ACESoftcorePotential": _Softcore,
        "build_aces_probe_system_dual_lambda": lambda *args, **kwargs: object(),
        "_build_platform_properties": lambda spec: captured.setdefault("spec", spec) and (
            "CUDA",
            {"Precision": "mixed", "DeviceIndex": "1"},
        ),
        "openmm": SimpleNamespace(
            Platform=SimpleNamespace(getPlatformByName=lambda name: captured.setdefault("platform", name) or name),
            LangevinMiddleIntegrator=lambda *args: object(),
            Context=_Context,
        ),
        "unit": SimpleNamespace(picosecond=1.0),
        "compute_2d_metric_grid": lambda *args, **kwargs: (
            np.zeros((2, 2, 2, 2)),
            {
                "valid_points": 4,
                "total_points": 4,
                "valid_fraction": 1.0,
                "median_samples_per_valid_point": 10.0,
                "failed_points": 0,
                "unsafe_points": 0,
            },
        ),
        "dijkstra_monotonic_geodesic": lambda *_args: [(1.0, 1.0), (0.0, 0.0)],
    }
    exec(compile(module, str(PREOPT), "exec"), ns)
    path = ns["optimize_2d_geodesic_path"](
        object(), object(), [(0, 0, 0)], None, [0], n_grid=2, platform_name="CUDA:1"
    )
    assert path == [(1.0, 1.0), (0.0, 0.0)]
    assert captured["spec"] == "CUDA:1"
    assert captured["platform"] == "CUDA"
    assert captured["context"] == (
        "CUDA",
        {"Precision": "mixed", "DeviceIndex": "1"},
    )


# ---------------------------------------------------------------------------
# [2026-09-01] CUDA 可用性判据必须来自 OpenMM，不能来自 torch
#
# 事故：计算节点上 OpenMM 的 CUDA 平台完全可用，而 torch（自带 CUDA 12.9 运行时）
# 因驱动/可见性差异 `torch.cuda.is_available()` 返回 False，于是 get_device_strategy
# 报「未检测到可用 CUDA 设备，已强制降级至 CPU」。本管线的动力学全部跑在 OpenMM 上，
# torch 只是 MLP 侧的可选依赖 —— 拿它当判据是把两个无关的运行时绑在了一起；
# 纯 OpenMM 环境（根本没装 torch）更是必然误判。
# ---------------------------------------------------------------------------

def _device_strategy_source() -> str:
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"), filename=str(PIPELINE))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_device_strategy":
            return ast.get_source_segment(PIPELINE.read_text(encoding="utf-8"), node)
    raise AssertionError("没找到 get_device_strategy")


def test_cuda_availability_is_decided_by_openmm_not_torch():
    src = _device_strategy_source()
    # 必须真的问 OpenMM 要 CUDA 平台
    assert 'Platform.getPlatformByName("CUDA")' in src, (
        "CUDA 可用性必须由 OpenMM 平台判定"
    )
    # 「平台注册」不等于「驱动能用」——必须真建一个最小 Context 探一下
    assert "openmm.Context(" in src, "必须建最小 Context 验证驱动真的能用"
    # torch 可以出现（用来数卡），但**不得**作为可用性判据
    if "torch" in src:
        assert "仅用于数卡" in src or "不**用它判断可用性" in src, (
            "torch 只能用于数卡，不得当可用性判据"
        )
    # 旧的错误判据不得复活
    assert "Torch CUDA unavailable" not in src


def test_downgrade_message_carries_the_underlying_error():
    """降级时必须报出底层原因，否则下一个人只能靠猜。"""
    src = _device_strategy_source()
    assert "type(exc).__name__" in src and "{exc}" in src


def test_device_count_prefers_the_scheduler_visible_devices():
    """队列给的 CUDA_VISIBLE_DEVICES 是权威来源，优先于 torch 的计数。"""
    src = _device_strategy_source()
    i_env = src.index("CUDA_VISIBLE_DEVICES")
    i_torch = src.index("import torch")
    assert i_env < i_torch, "必须先读 CUDA_VISIBLE_DEVICES，再退回 torch 计数"


def test_equilibrium_rejection_does_not_claim_a_full_rerun():
    """`equilibrium_is_done` 返回 False ≠ 从零重跑。

    实测（4W53 resume_v3.log）：连报两次"将重新执行预平衡"，紧接着却是
    `♻️ 从 Checkpoint 恢复 | 已完成: 5000000 | 剩余: 0`，一步没跑。
    旧文案让人以为烧掉 500 万步 GPU——不报错、不影响数值，只误导读日志的人。
    """
    runabfe = (ROOT / "runabfe.py").read_text(encoding="utf-8")
    tree = ast.parse(runabfe, filename="runabfe.py")
    body = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "equilibrium_is_done":
            body = ast.get_source_segment(runabfe, node)
    assert body is not None
    assert "将重新执行预平衡" not in body, "文案不得声称从零重跑"
    assert "_REENTER_NOTE" in body, "拒绝复用时必须说明下游仍可能从 checkpoint 续跑"


# ---------------------------------------------------------------------------
# [2026-09-01] 预平衡指纹必须对"坐标变换"和"往返浮点噪声"免疫
#
# 事故：每次 --resume 都报「已有预平衡轨迹的指纹与当前 system/config 不匹配」。
# 两个独立的必然失败原因：
#   1. `pre_equilibrate()` 在算指纹**之前**先跑 repair_pbc_molecule_integrity()，
#      而调用方 equilibrium_is_done() 是在那**之前**算的期望值 ——
#      实测 4W53：mmCIF 坐标与预平衡 DCD 第 0 帧全部 48962 个原子都不同、最大差 13 nm。
#   2. `_positions_hash` 哈希原始 float64 字节，同一套坐标经 GRO 与经 mmCIF 读回
#      相差 ~1.8e-15，也足以换哈希。
# 修法：坐标在构造期冻结（不受后续变换影响）+ 量化后再哈希，两侧走同一个入口。
# ---------------------------------------------------------------------------

def _identity_stub(positions, box, steps=5_000_000):
    import abfe_pipeline as _P
    from openmm import unit as _u

    class _Stub:
        pressure = 1.0 * _u.bar
        temperature = 300.0 * _u.kelvin
        barostat_protocol = None
        system = None
        ligand_indices = [1, 2, 3]
        pre_equilibration_identity_fingerprint = (
            _P.ABFEPipeline.pre_equilibration_identity_fingerprint
        )

    s = _Stub()
    s._identity_positions = positions * _u.nanometer
    s._identity_box_vectors = box * _u.nanometer
    s.positions = s._identity_positions
    s.box_vectors = s._identity_box_vectors
    return s.pre_equilibration_identity_fingerprint(steps)


def test_pre_equilibration_identity_survives_coordinate_transforms():
    from openmm import unit as _u
    import abfe_pipeline as _P

    rng = np.random.default_rng(0)
    pos = rng.normal(size=(2000, 3)) * 3.0
    box = np.eye(3) * 7.893
    base = _identity_stub(pos, box)

    # self.positions 被 PBC 修复/居中/再平衡改写后，身份必须不变
    class _S:
        pressure = 1.0 * _u.bar
        temperature = 300.0 * _u.kelvin
        barostat_protocol = None
        system = None
        ligand_indices = [1, 2, 3]
        pre_equilibration_identity_fingerprint = (
            _P.ABFEPipeline.pre_equilibration_identity_fingerprint
        )

    s = _S()
    s._identity_positions = pos * _u.nanometer
    s._identity_box_vectors = box * _u.nanometer
    s.positions = (pos + 13.0) * _u.nanometer      # 模拟 13 nm 重排
    s.box_vectors = s._identity_box_vectors
    assert s.pre_equilibration_identity_fingerprint(5_000_000) == base

    # GRO↔mmCIF 往返的浮点噪声也必须被吸收
    assert _identity_stub(pos + 1.8e-15, box) == base


def test_pre_equilibration_identity_still_catches_real_changes():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(2000, 3)) * 3.0
    box = np.eye(3) * 7.893
    base = _identity_stub(pos, box)
    assert _identity_stub(pos + 2.0e-4, box) != base, "0.002 Å 的真实位移必须抓到"
    assert _identity_stub(pos + 0.05, box) != base, "换 pose 必须抓到"
    assert _identity_stub(pos, box * 1.01) != base, "换盒子必须抓到"
    assert _identity_stub(pos, box, steps=1_000_000) != base, "换步数必须抓到"


def test_both_sides_go_through_the_same_entry_point():
    """runabfe 与 pre_equilibrate 必须调同一个入口，否则时序不同 ⇒ 永远不匹配。"""
    runabfe = (ROOT / "runabfe.py").read_text(encoding="utf-8")
    live = [
        l for l in runabfe.split("\n")
        if "_pre_equilibration_fingerprint(" in l
        and not l.strip().startswith("#")
    ]
    assert live == [], f"runabfe 仍在自己算预平衡指纹：{live}"
    assert runabfe.count("pre_equilibration_identity_fingerprint(") >= 5

    pipeline = PIPELINE.read_text(encoding="utf-8")
    i = pipeline.index("def pre_equilibrate(")
    j = pipeline.index("def ", i + 10)
    body = pipeline[i:j]
    assert "self.pre_equilibration_identity_fingerprint(" in body
