from __future__ import annotations

import ast
import math
import ntpath
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "abfe_core.py"
RUNABFE = ROOT / "runabfe.py"


class _Q:
    def __init__(self, value): self.value = value
    def value_in_unit(self, _unit): return self.value


class MonteCarloBarostat:
    def __init__(self, p=1.0, t=300.0, f=25): self.p, self.t, self.f = p, t, f
    def getDefaultPressure(self): return _Q(self.p)
    def getDefaultTemperature(self): return _Q(self.t)
    def getFrequency(self): return self.f


class MonteCarloMembraneBarostat(MonteCarloBarostat):
    def __init__(self, p=1.0, t=310.0, f=50, tension=12.5, xy=1, z=2):
        super().__init__(p, t, f); self.tension, self.xy, self.z = tension, xy, z
    def getDefaultSurfaceTension(self): return _Q(self.tension)
    def getXYMode(self): return self.xy
    def getZMode(self): return self.z


class _System:
    def __init__(self, force): self.forces = [force]
    def getForces(self): return self.forces
    def getForce(self, index): return self.forces[index]
    def addForce(self, force): self.forces.append(force); return len(self.forces) - 1


def _load_barostat_ensure():
    names = {"detect_barostats", "_barostat_actual_parameters", "_barostat_parameters_match", "ensure_barostat_for_protocol"}
    tree = ast.parse(CORE.read_text(encoding="utf-8"), filename=str(CORE))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    ns = {
        "Any": Any, "Dict": Dict, "List": List, "Tuple": Tuple, "math": math,
        "unit": SimpleNamespace(bar=1.0, kelvin=1.0, nanometer=1.0),
        "BAROSTAT_FORCE_CLASS_NAMES": {"MonteCarloBarostat", "MonteCarloMembraneBarostat"},
        "_openmm_membrane_xy_mode": lambda _name: 1, "_openmm_membrane_z_mode": lambda _name: 2,
        "openmm": SimpleNamespace(MonteCarloBarostat=MonteCarloBarostat, MonteCarloMembraneBarostat=MonteCarloMembraneBarostat),
    }
    exec(compile(module, str(CORE), "exec"), ns)
    return ns["ensure_barostat_for_protocol"]


def test_existing_soluble_barostat_compares_real_parameters():
    ensure = _load_barostat_ensure()
    protocol = {"barostat_class": "MonteCarloBarostat", "barostat_frequency": 25, "system_type": "soluble"}
    result = ensure(_System(MonteCarloBarostat()), protocol, 300.0, 1.0)
    assert result["action"] == "reused_existing"
    assert result["actual_parameters"]["frequency"] == 25
    with pytest.raises(RuntimeError, match="实际参数"):
        ensure(_System(MonteCarloBarostat(p=2.0)), protocol, 300.0, 1.0)


def test_existing_membrane_barostat_compares_tension_and_frequency():
    ensure = _load_barostat_ensure()
    protocol = {"barostat_class": "MonteCarloMembraneBarostat", "barostat_frequency": 50, "system_type": "membrane", "membrane": {"surface_tension_bar_nm": 12.5, "xy_mode": "isotropic", "z_mode": "free"}}
    assert ensure(_System(MonteCarloMembraneBarostat()), protocol, 310.0, 1.0)["action"] == "reused_existing"
    with pytest.raises(RuntimeError, match="surface_tension"):
        ensure(_System(MonteCarloMembraneBarostat(tension=0.0)), protocol, 310.0, 1.0)
    with pytest.raises(RuntimeError, match="frequency"):
        ensure(_System(MonteCarloMembraneBarostat(f=25)), protocol, 310.0, 1.0)


def test_forcefield_resolver_forwards_and_records_include_dir(tmp_path):
    tree = ast.parse(CORE.read_text(encoding="utf-8"), filename=str(CORE))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resolve_forcefield_family")
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    captured = {}
    ns = {"Dict": Dict, "Optional": Optional, "os": os, "FORCEFIELD_FAMILIES": ("amber", "charmm"), "FORCEFIELD_FAMILIES_UNSUPPORTED": (), "logger": SimpleNamespace(warning=lambda *a, **k: None), "detect_forcefield_family_from_top": lambda top, include: captured.update(top=top, include=include) or {"family": "amber", "reason": "test"}}
    exec(compile(module, str(CORE), "exec"), ns)
    include = tmp_path / "share" / "gromacs" / "top"
    result = ns["resolve_forcefield_family"]("input.top", gmx_include_dir=str(include))
    assert captured["include"] == str(include)
    assert result["detection"]["gmx_include_dir"] == str(include.resolve())


def test_windows_gmx_executable_uses_shutil_which():
    tree = ast.parse(RUNABFE.read_text(encoding="utf-8"), filename=str(RUNABFE))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "find_gmx_include_dir")
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    expected = ntpath.normpath(r"C:\Gromacs\share\gromacs\top")
    fake_path = SimpleNamespace(exists=lambda value: ntpath.normpath(value) == expected, join=ntpath.join, dirname=ntpath.dirname, abspath=ntpath.abspath)
    ns = {"Optional": Optional, "os": SimpleNamespace(path=fake_path, environ={}), "shutil": SimpleNamespace(which=lambda name: r"C:\Gromacs\bin\gmx.exe" if name == "gmx" else None), "log": SimpleNamespace(warning=lambda *a, **k: None)}
    exec(compile(module, str(RUNABFE), "exec"), ns)
    assert ntpath.normpath(ns["find_gmx_include_dir"]()) == expected


def test_both_production_legs_forward_n_equil_steps():
    tree = ast.parse(RUNABFE.read_text(encoding="utf-8"), filename=str(RUNABFE))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "run_full_pipeline"]
    assert sum(any(k.arg == "n_equil_steps" for k in call.keywords) for call in calls) >= 2
