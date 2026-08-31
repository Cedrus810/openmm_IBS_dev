"""P1-19 回归：`format_results_human` 不能把 solvent 腿打印成"复合物总自由能"。

`abfe_pipeline.ABFEPipeline.compute_final_results` 对 complex 腿和 solvent 腿
都往 `final["total_delta_G_complex_kJ_mol"]` 写值（字段名是历史遗留，两条腿
共用同一个键存"这条腿自己的总 ΔG"）。`UnitFormatter.format_results_human`
原来只按键名是否存在选标题，于是 solvent 腿的 pipeline.log 里也印出
"✅ 复合物总自由能 ΔG_complex"，容易让人把两条腿的数字看混。

修法：`compute_final_results` 现在把 `system_type`（"complex"/"solvent"）
一并写进 `final`；`format_results_human` 据此选标题，缺该字段的旧产物按
"complex" 处理，不改变既有行为。
"""

import pytest

pytestmark = pytest.mark.cpu_only


def test_complex_leg_keeps_complex_title():
    from abfe_core import UnitFormatter

    text = UnitFormatter.format_results_human({
        "total_delta_G_complex_kJ_mol": 87.3,
        "total_error_kJ_mol": 1.4,
        "system_type": "complex",
    })
    assert "复合物总自由能" in text
    assert "ΔG_complex" in text


def test_solvent_leg_does_not_claim_to_be_complex():
    from abfe_core import UnitFormatter

    text = UnitFormatter.format_results_human({
        "total_delta_G_complex_kJ_mol": 42.0,
        "total_error_kJ_mol": 0.9,
        "system_type": "solvent",
    })
    assert "复合物总自由能" not in text
    assert "solvent" in text


def test_missing_system_type_defaults_to_legacy_complex_behavior():
    """旧的 final_results.json 没有 system_type 字段，标题不能变。"""
    from abfe_core import UnitFormatter

    text = UnitFormatter.format_results_human({
        "total_delta_G_complex_kJ_mol": 10.0,
        "total_error_kJ_mol": 0.1,
    })
    assert "复合物总自由能" in text


def test_delta_g_bind_title_is_unaffected_by_system_type():
    from abfe_core import UnitFormatter

    text = UnitFormatter.format_results_human({
        "delta_G_bind_kJ_mol": -25.0,
        "total_error_kJ_mol": 0.5,
        "system_type": "solvent",
    })
    assert "结合自由能" in text


def _final_results_for_leg(tmp_path, monkeypatch, system_type):
    """跑一次真正的 compute_final_results，取 leg 身份如何落进 `final`。

    构造方式抄 test_lrc_reporting_honesty.py::_lrc_block：system=None 让约束
    Jacobian 那段短路，single_lambda 走 compute_final_results 的 else 分支，
    不需要伪造 stage0/stage1/stage2。
    """
    from openmm import unit

    import abfe_pipeline as ap

    pipeline = ap.ABFEPipeline.__new__(ap.ABFEPipeline)
    pipeline._last_run_config = {"potential_type": "softcore", "system_type": system_type}
    pipeline.output_dir = str(tmp_path)
    pipeline.system = None
    pipeline.topology = None
    pipeline.positions = None
    pipeline.ligand_indices = []
    pipeline.temperature = 300.0 * unit.kelvin
    pipeline._log = lambda *a, **k: None
    monkeypatch.setattr(ap, "_collect_pipeline_provenance", lambda **kw: {})

    return pipeline.compute_final_results(
        sampling_results={"total_delta_G": 1.0, "total_error": 0.1},
        correction_results={"delta_g_rest": 0.0, "error": 0.0},
        decoupling_scheme="single_lambda",
    )


def test_compute_final_results_writes_system_type_for_solvent_leg(tmp_path, monkeypatch):
    final = _final_results_for_leg(tmp_path, monkeypatch, "solvent")
    assert final["system_type"] == "solvent"


def test_compute_final_results_writes_system_type_for_complex_leg(tmp_path, monkeypatch):
    final = _final_results_for_leg(tmp_path, monkeypatch, "complex")
    assert final["system_type"] == "complex"
