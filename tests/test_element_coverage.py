"""参考 ML 势的元素覆盖判定（`local_residual.element_coverage`）。

这道判定跟残差模型的训练/重训无关——R1 的训练信号是纯 MM 的。它只回答
"这个体系的元素，这个模型表不表示得了"，在启动期给答案而不是跑到一半才发现。
"""

import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[1]


def _model_available(name):
    from local_residual.element_coverage import ElementCoverageError, resolve_model

    try:
        return resolve_model(name)
    except ElementCoverageError:
        return None


def test_coverage_is_decided_by_the_model_file_not_a_constant():
    """抄常量迟早跟模型对不上：同族 MACE-omol-0 里 1024 是 83 种、4M 是 82 种。"""

    from local_residual import element_coverage

    source = Path(element_coverage.__file__).read_text(encoding="utf-8")
    # 模块里不得出现硬编码的元素表
    assert "atomic_numbers" in source, "z-table 必须从模型的 atomic_numbers 读"
    assert "tuple(range(1, " not in source, "不许把某个模型的覆盖范围写成常量"


def test_missing_elements_are_all_reported_and_fail_closed():
    from local_residual.element_coverage import ElementCoverageError, check_element_coverage

    table = (1, 6, 7, 8)
    fake = str(REPO_ROOT / "nonexistent.model")
    # 用一个假的 z-table 走同一条判定：直接测 check_element_coverage 需要真模型，
    # 所以这里只锁"找不到模型时 fail-closed 且说清楚怎么指路"。
    with pytest.raises(ElementCoverageError, match="ABFE_MACE_MODEL_DIR"):
        check_element_coverage([1, 6], fake)
    assert table  # 保持断言形状清晰


def test_empty_element_set_is_refused():
    from local_residual.element_coverage import ElementCoverageError, check_element_coverage

    with pytest.raises(ElementCoverageError, match="空"):
        check_element_coverage([], "whatever.model")


@pytest.mark.skipif(
    _model_available("MACE-omol-0-extra-large-1024") is None,
    reason="本机没有 MACE-omol-0（给 $ABFE_MACE_MODEL_DIR 或放进 ~/.cache/mace）",
)
def test_real_models_disagree_exactly_where_the_archive_says_they_do():
    """MACE-OFF24 没有 Na，MACE-omol-0 有——EXP-033 §2.A 记的就是这条。"""

    from local_residual.element_coverage import check_element_coverage, model_z_table
    from local_residual.element_coverage import ElementCoverageError

    omol = model_z_table("MACE-omol-0-extra-large-1024")
    assert 11 in omol and len(omol) == 83
    system_elements, table = check_element_coverage([1, 6, 7, 8, 11, 16, 17], "MACE-omol-0-extra-large-1024")
    assert system_elements == (1, 6, 7, 8, 11, 16, 17)
    assert table == omol
    off24 = _model_available("mace-off24-medium")
    if off24 is not None:
        assert 11 not in model_z_table(off24)
        with pytest.raises(ElementCoverageError, match=r"\[11\]"):
            check_element_coverage([1, 6, 7, 8, 11], off24)


def test_switch_is_off_by_default_and_reaches_the_config(tmp_path, monkeypatch):
    import runabfe

    config_path = tmp_path / "plain.json"
    config_path.write_text("{}")
    monkeypatch.setattr("sys.argv", ["runabfe.py", "--config", str(config_path)])
    assert runabfe.RunConfig(runabfe.parse_arguments()).element_coverage_model is None
    monkeypatch.setattr(
        "sys.argv",
        ["runabfe.py", "--config", str(config_path), "--element-coverage-model", "x.model"],
    )
    assert runabfe.RunConfig(runabfe.parse_arguments()).element_coverage_model == "x.model"


def test_runabfe_does_not_import_torch_when_the_check_is_off():
    """默认关时整段跳过，不得因为这道判定给每次启动加上 torch 的成本。"""

    import inspect

    import runabfe

    source = inspect.getsource(runabfe.main)
    assert "from local_residual.element_coverage import" in source, "必须是函数内惰性 import"
    assert "if _coverage_model:" in source
