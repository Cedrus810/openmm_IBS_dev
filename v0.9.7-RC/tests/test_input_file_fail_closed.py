"""显式 config/torsion 输入不得静默降级为默认值。"""

import json
import types

import pytest

pytestmark = pytest.mark.cpu_only

from runabfe import RunConfig, _load_json_object_file


def _minimal_args(config=None):
    return types.SimpleNamespace(preset="production", config=config)


def test_explicit_missing_config_fails_closed(tmp_path, monkeypatch):
    missing = tmp_path / "missing.json"
    monkeypatch.setattr("sys.argv", ["runabfe.py", "--config", str(missing)])
    with pytest.raises(FileNotFoundError, match="显式配置文件不存在"):
        RunConfig(_minimal_args(str(missing)))


@pytest.mark.parametrize("content", ["{broken", "[]", "null"])
def test_explicit_invalid_config_fails_closed(tmp_path, monkeypatch, content):
    path = tmp_path / "config.json"
    path.write_text(content, encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["runabfe.py", "--config", str(path)])
    with pytest.raises((ValueError, json.JSONDecodeError)):
        RunConfig(_minimal_args(str(path)))


def test_explicit_valid_config_is_merged(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text('{"temperature": 315.0, "potential": "softcore"}', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["runabfe.py", "--config", str(path)])
    config = RunConfig(_minimal_args(str(path)))
    assert config.temperature == 315.0
    assert config.potential == "softcore"


def test_unspecified_config_keeps_preset(monkeypatch):
    monkeypatch.setattr("sys.argv", ["runabfe.py"])
    config = RunConfig(_minimal_args())
    assert config.potential == "softcore"


def test_explicit_missing_torsion_file_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="torsion 参数"):
        _load_json_object_file(str(tmp_path / "missing.json"), "torsion 参数")


@pytest.mark.parametrize("content", ["{broken", "[]", "null"])
def test_explicit_invalid_torsion_file_fails_closed(tmp_path, content):
    path = tmp_path / "torsion.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="torsion 参数"):
        _load_json_object_file(str(path), "torsion 参数")


def test_explicit_valid_torsion_file_loads_object(tmp_path):
    path = tmp_path / "torsion.json"
    path.write_text('{"periodicity": 3}', encoding="utf-8")
    assert _load_json_object_file(str(path), "torsion 参数") == {"periodicity": 3}


def test_main_uses_fail_closed_torsion_loader():
    source = (__import__("pathlib").Path(__file__).resolve().parents[1] / "runabfe.py").read_text(
        encoding="utf-8"
    )
    assert '_load_json_object_file(config.torsion_params, "torsion 参数")' in source
    assert "config.torsion_params and os.path.exists(config.torsion_params)" not in source
