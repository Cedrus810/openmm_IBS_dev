"""P0 的 CLI／配置接线（`docs/design/PLAN_openmm_8_6_remd_backend.md` §3、P0 交付物）。

选择器本身的分支逻辑在 `test_free_energy_engine_backend_selection.py` 测；本文件只测
**接线**：参数存在、默认值正确、配置文件能覆盖、解析结果能落进 provenance，
以及最关键的一条——**默认运行行为不变**。
"""

from __future__ import annotations

import json

import pytest

import free_energy_engine as fee
import runabfe


def _config(monkeypatch, argv_extra=(), config_payload=None, tmp_path=None):
    argv = ["runabfe.py", *argv_extra]
    if config_payload is not None:
        path = tmp_path / "conf.json"
        path.write_text(json.dumps(config_payload), encoding="utf-8")
        argv += ["--config", str(path)]
    monkeypatch.setattr("sys.argv", argv)
    return runabfe.RunConfig(runabfe.parse_arguments())


# ---------------------------------------------------------------------------
# 参数与默认值
# ---------------------------------------------------------------------------


def test_flag_exists_with_exactly_the_documented_choices(monkeypatch):
    monkeypatch.setattr("sys.argv", ["runabfe.py"])
    args = runabfe.parse_arguments()
    assert hasattr(args, "remd_backend")

    # 三个取值都能解析
    for choice in fee.REMD_BACKEND_CHOICES:
        monkeypatch.setattr("sys.argv", ["runabfe.py", "--remd-backend", choice])
        assert runabfe.parse_arguments().remd_backend == choice


def test_invalid_choice_is_rejected_by_argparse(monkeypatch):
    monkeypatch.setattr("sys.argv", ["runabfe.py", "--remd-backend", "openmmtools"])
    with pytest.raises(SystemExit):
        runabfe.parse_arguments()


def test_default_is_auto(monkeypatch):
    """计划 §3：目标默认值是 auto。"""
    assert _config(monkeypatch).get("remd_backend") == "auto"
    assert fee.REMD_BACKEND_DEFAULT == "auto"


def test_cli_flag_overrides_default(monkeypatch):
    assert _config(monkeypatch, ["--remd-backend", "legacy"]).get("remd_backend") == "legacy"


def test_config_file_can_set_it(monkeypatch, tmp_path):
    cfg = _config(monkeypatch, config_payload={"remd_backend": "legacy"}, tmp_path=tmp_path)
    assert cfg.get("remd_backend") == "legacy"


def test_cli_beats_config_file(monkeypatch, tmp_path):
    """RunConfig 的既有优先级：命令行 > 配置文件 > 预设。"""
    cfg = _config(
        monkeypatch,
        ["--remd-backend", "auto"],
        config_payload={"remd_backend": "legacy"},
        tmp_path=tmp_path,
    )
    assert cfg.get("remd_backend") == "auto"


# ---------------------------------------------------------------------------
# 当前阶段的实际行为：默认路径一字不变
# ---------------------------------------------------------------------------


def test_default_config_resolves_to_legacy_in_this_environment():
    """🔑 P0 的放行条件：默认仍保留旧运行行为。

    适配器未实现，所以无论环境如何，auto 都必须解析成 legacy——现有运行不受影响。
    """
    resolution = fee.resolve_remd_backend("auto")
    assert resolution.resolved == "legacy"
    assert resolution.exchange_scheme == fee.EXCHANGE_SCHEME_LEGACY


def test_explicit_official_fails_closed_rather_than_downgrading():
    with pytest.raises(fee.UnsupportedRemdBackendError):
        fee.resolve_remd_backend("openmm")


# ---------------------------------------------------------------------------
# provenance（计划 §4.3）
# ---------------------------------------------------------------------------


def test_resolution_provenance_is_json_serialisable():
    """要落进 run_provenance.json，就必须能被 json.dumps 直接吃掉。"""
    payload = fee.resolve_remd_backend("auto").to_provenance()
    json.dumps(payload)  # 不抛异常即通过


def test_provenance_records_both_requested_and_resolved():
    """只记 resolved 会丢掉「用户当初要的是什么」，事后无法区分
    「用户就要 legacy」和「用户要 auto 但环境不合格」。"""
    payload = fee.resolve_remd_backend("auto").to_provenance()
    assert payload["remd_backend_requested"] == "auto"
    assert payload["remd_backend_resolved"] == "legacy"
    assert payload["remd_backend_reason"]


def test_provenance_key_is_read_defensively_in_runabfe():
    """🔑 新增 provenance 字段必须用 config.get 兜底。

    本仓库有过先例：往协议键里加字段而不做缺失兜底，会炸掉用 object.__new__ 造的
    测试 stub 和旧的 resume 路径。这里静态检查那处写法。
    """
    from pathlib import Path

    source = Path(runabfe.__file__).read_text(encoding="utf-8")
    assert 'config.get("_remd_backend_resolution")' in source, (
        "provenance 必须用 config.get 读解析结果，不能用属性访问"
    )


def test_backend_is_resolved_before_any_context_creation():
    """§3 实施要求 6：后端只在阶段开始前解析一次。

    静态检查解析点的位置：必须在 main() 里、且排在 ligand 校验之前——那时还没有
    建过任何 Context、没写过任何轨迹。显式 openmm 不合格时才能在烧 GPU 前退出。
    """
    from pathlib import Path

    source = Path(runabfe.__file__).read_text(encoding="utf-8")
    resolve_at = source.index("free_energy_engine.resolve_remd_backend(")
    ligand_check_at = source.index('log.error("未提供配体残基名称')
    assert resolve_at < ligand_check_at


def test_only_one_resolution_call_site():
    """解析必须只有一处。多处解析等于允许"跑到一半换引擎"，§3 实施要求 6 明令禁止。"""
    from pathlib import Path

    source = Path(runabfe.__file__).read_text(encoding="utf-8")
    assert source.count("free_energy_engine.resolve_remd_backend(") == 1
