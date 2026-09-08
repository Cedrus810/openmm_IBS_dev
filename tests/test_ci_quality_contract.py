"""Repository quality contracts introduced for GitHub issue #62."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MODULES = (
    ROOT / "abfe_core.py",
    ROOT / "abfe_pipeline.py",
    ROOT / "abfe_preoptimizer.py",
    # 2026-08-31 新增的生产模块，纳入同一套质量契约。
    ROOT / "free_energy_engine.py",
    ROOT / "rbfe_core.py",
)


def test_runtime_modules_do_not_use_bare_except():
    """Shutdown exceptions must never be swallowed by production helpers."""
    offenders: list[str] = []
    for path in RUNTIME_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                offenders.append(f"{path.name}:{node.lineno}")

    assert not offenders, f"bare except handlers remain: {offenders}"


def test_ci_workflow_excludes_gpu_tests_from_ordinary_runner():
    workflow = ROOT / ".github" / "workflows" / "cpu-ci.yml"
    source = workflow.read_text(encoding="utf-8")
    assert '-m "not needs_gpu"' in source
    assert "environment-ci.yml" in source
    assert "cuda" not in source.lower()


def test_cpu_ci_environment_contains_required_test_runtime():
    source = (ROOT / "environment-ci.yml").read_text(encoding="utf-8")
    for requirement in ("python=3.12", "openmm", "pymbar", "pytest", "mdtraj"):
        assert requirement in source
