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
