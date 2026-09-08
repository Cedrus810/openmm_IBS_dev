"""Executable contracts for still-open GitHub bug/robustness issues.

These tests deliberately separate two states:

* ordinary tests describe behavior that is already implemented and must not
  regress;
* ``xfail(strict=True)`` tests describe an accepted issue whose fix is not in
  the tree yet.  When a fix lands, XPASS is an error so the contributor must
  remove the marker, attach evidence to the issue, and update the TODO.

The source-extraction helpers keep the corrupt-file tests independent of an
OpenMM installation: the current validators use only ``os`` and builtins.
"""

from __future__ import annotations

import ast
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "abfe_pipeline.py"
CORE_PATH = ROOT / "abfe_core.py"


def _compile_top_level_function(path: Path, function_name: str, namespace: dict):
    """Compile one top-level function without importing the production module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    ]
    assert len(matches) == 1, f"expected one top-level {function_name}, got {len(matches)}"
    module = ast.Module(body=[matches[0]], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function_name]


def test_checkpoint_validator_rejects_large_seekable_garbage(tmp_path):
    validate = _compile_top_level_function(PIPELINE_PATH, "_is_checkpoint_valid", {"os": os})
    fake_checkpoint = tmp_path / "garbage.chk"
    fake_checkpoint.write_bytes(b"not-an-openmm-checkpoint".ljust(1024, b"x"))

    assert validate(str(fake_checkpoint)) is False


def test_trajectory_validator_rejects_header_shaped_truncated_garbage(tmp_path):
    validate = _compile_top_level_function(PIPELINE_PATH, "_is_traj_valid", {"os": os})
    # Satisfies every shallow check in the current implementation but is not a
    # parseable DCD: there is no complete coordinate frame or closing record.
    header = bytearray(212)
    header[4:8] = b"CORD"
    fake_dcd = tmp_path / "truncated.dcd"
    fake_dcd.write_bytes(bytes(header) + (64).to_bytes(4, "little") + b"x" * 128)

    assert validate(str(fake_dcd), min_frames=1) is False


def _install_fake_mdtraj_dcd_reader(monkeypatch, *, frame_count=3, raises=False):
    class _Coordinates:
        shape = (frame_count, 4, 3)

    class _Reader:
        n_atoms = 4

        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            if raises:
                raise OSError("truncated DCD record")
            return (_Coordinates(), None, None)

    fake_mdtraj = types.SimpleNamespace(formats=types.SimpleNamespace(DCDTrajectoryFile=_Reader))
    monkeypatch.setitem(sys.modules, "mdtraj", fake_mdtraj)


def test_trajectory_validator_accepts_complete_low_level_reader_without_topology(
    tmp_path, monkeypatch
):
    validate = _compile_top_level_function(PIPELINE_PATH, "_is_traj_valid", {"os": os})
    dcd = tmp_path / "complete.dcd"
    dcd.write_bytes(b"opaque-dcd")
    _install_fake_mdtraj_dcd_reader(monkeypatch, frame_count=3)

    assert validate(str(dcd), min_frames=3) is True
    assert validate(str(dcd), min_frames=4) is False


def test_trajectory_validator_rejects_reader_reported_truncation(tmp_path, monkeypatch):
    validate = _compile_top_level_function(PIPELINE_PATH, "_is_traj_valid", {"os": os})
    dcd = tmp_path / "truncated-reader.dcd"
    dcd.write_bytes(b"opaque-dcd")
    _install_fake_mdtraj_dcd_reader(monkeypatch, raises=True)

    assert validate(str(dcd), min_frames=1) is False


def test_trajectory_validator_uses_low_level_reader_without_topology(tmp_path, monkeypatch):
    """A topology-free DCD is accepted only after the real reader reads all frames."""

    validate = _compile_top_level_function(PIPELINE_PATH, "_is_traj_valid", {"os": os})
    dcd_path = tmp_path / "valid.dcd"
    dcd_path.write_bytes(b"nonempty; the fake reader below performs validation")

    class _Reader:
        reads = 0

        def __init__(self, path):
            assert path == str(dcd_path)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            type(self).reads += 1
            coordinates = [
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[0.1, 0.0, 0.0], [1.1, 1.0, 1.0]],
                [[0.2, 0.0, 0.0], [1.2, 1.0, 1.0]],
            ]
            return coordinates, None, None, None

    fake_mdtraj = SimpleNamespace(
        formats=SimpleNamespace(DCDTrajectoryFile=_Reader),
        load_dcd=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("topology-dependent load_dcd must not be used")
        ),
    )
    monkeypatch.setitem(sys.modules, "mdtraj", fake_mdtraj)

    assert validate(str(dcd_path), min_frames=3) is True
    assert validate(str(dcd_path), min_frames=4) is False
    assert _Reader.reads == 2


def test_trajectory_validator_rejects_low_level_reader_truncation(tmp_path, monkeypatch):
    """A reader exception, such as a truncated DCD record, fails closed."""

    validate = _compile_top_level_function(PIPELINE_PATH, "_is_traj_valid", {"os": os})
    dcd_path = tmp_path / "truncated.dcd"
    dcd_path.write_bytes(b"nonempty")

    class _TruncatingReader:
        def __init__(self, _path):
            pass

        def read(self):
            raise ValueError("truncated DCD record")

        def close(self):
            pass

    fake_mdtraj = SimpleNamespace(formats=SimpleNamespace(DCDTrajectoryFile=_TruncatingReader))
    monkeypatch.setitem(sys.modules, "mdtraj", fake_mdtraj)

    assert validate(str(dcd_path), min_frames=1) is False


def _class_method_node(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
    assert len(classes) == 1
    methods = [
        n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == method_name
    ]
    assert len(methods) == 1
    return methods[0]


def test_optimize_alpha_does_not_use_assert_for_runtime_input_validation():
    """#64 fixed 2026-08-24: asserts replaced with ``raise ValueError`` (abfe_core.py)."""
    method = _class_method_node(CORE_PATH, "ACESoftcorePotential", "optimize_alpha")
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(method))


def test_boresch_last_frame_does_not_guess_transpose_for_three_atoms():
    """#64 fixed 2026-08-24: the (3, 3)-shape transpose guess was removed (abfe_core.py);
    ambiguous shapes now raise ValueError like every other bad-shape case."""
    tree = ast.parse(CORE_PATH.read_text(encoding="utf-8"), filename=str(CORE_PATH))
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "calc_boresch_from_last_frame"
    )
    source = ast.get_source_segment(CORE_PATH.read_text(encoding="utf-8"), function)
    assert source is not None
    normalized = "".join(source.split())
    assert "ifpos.shape==(3,3):pos=pos.T" not in normalized
