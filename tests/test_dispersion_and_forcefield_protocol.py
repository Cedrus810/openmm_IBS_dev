"""B6 + §1.1 + §13：色散路线、力场族识别、验收阈值常量。

对应 memtodolist.md：
  §1.1  力场族按输入自动识别（amber 首选，charmm 因 OpenMM 无 force-switch 默认卡住），
        识别不出 fail closed，允许显式覆盖但必须留记录。
  §1.3  `dispersion_protocol` 显式配置；membrane 未选已验证路线 → fail closed；
        membrane + legacy 均匀密度 LRC → fail closed（§6.4 同条）。
  §13   全部"预设阈值"落成命名常量并进 provenance，不许运行时凭感觉判。
  §7.1  「membrane + 未指定 dispersion protocol：失败」这一条。
  §7.7  soluble 不声明时行为不变。

全部 CPU 可跑：纯校验层 + 文本解析，不建 System/Context。
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

pytest.importorskip("openmm")

import abfe_core as core

ROOT = Path(__file__).absolute().parents[1]


# ---------------------------------------------------------------------------
# §1.1 力场族识别
# ---------------------------------------------------------------------------


def test_production_topology_is_detected_as_amber():
    """实测：本仓库 `topol.top` 走 `amber14sb_OL15_fs1.ff` → amber。

    注意它**没有 `[ defaults ]` 段**（那在 forcefield.itp 里），所以主判据只能是
    include 路径。这条测试同时钉住"本地 include 不参与判定"。
    """
    result = core.detect_forcefield_family_from_top(str(ROOT / "topol.top"))
    assert result["family"] == "amber"
    assert result["reason"] == "single_family_include"
    assert result["defaults_row"] is None
    # 本地 include 出现在 includes 里但不构成族证据。
    assert "./Atenolol-rank1.itp" in result["includes"]
    assert set(result["family_evidence"]) == {"amber"}


def _write_top(tmp_path, includes, defaults=None):
    lines = ["; test topology", ""]
    if defaults is not None:
        lines += ["[ defaults ]", "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ", defaults]
    lines += [f'#include "{inc}"' for inc in includes]
    path = tmp_path / "topol.top"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_charmm_topology_is_detected(tmp_path):
    top = _write_top(
        tmp_path,
        ["charmm36.ff/forcefield.itp", "./ligand.itp", "charmm36.ff/tip3p.itp"],
        defaults="1               2               yes             1.0     1.0",
    )
    result = core.detect_forcefield_family_from_top(top)
    assert result["family"] == "charmm"
    # [ defaults ] 若存在则记录下来作交叉检查，但不是唯一判据。
    assert result["defaults_row"].startswith("1")


def test_mixed_family_includes_fail_closed(tmp_path):
    top = _write_top(tmp_path, ["amber14sb.ff/forcefield.itp", "charmm36.ff/tip3p.itp"])
    with pytest.raises(ValueError, match="无法从"):
        core.resolve_forcefield_family(top_path=top)


def test_unrecognized_forcefield_dir_fails_closed_and_does_not_default_to_amber(tmp_path):
    """§1.1：识别不出时 **fail closed，不许猜、不许默认回落到 amber**。"""
    top = _write_top(tmp_path, ["my_custom_ff/forcefield.itp", "./ligand.itp"])
    with pytest.raises(ValueError, match="不许猜、不许默认回落到 amber"):
        core.resolve_forcefield_family(top_path=top)


def test_explicit_override_is_recorded_not_silent(tmp_path):
    """§1.1：允许显式覆盖，但覆盖必须留记录。"""
    top = _write_top(tmp_path, ["amber14sb.ff/forcefield.itp"])
    result = core.resolve_forcefield_family(top_path=top, explicit_family="charmm")
    assert result["family"] == "charmm"
    assert result["source"] == "explicit_override"
    assert result["overrode_detection"] == "amber"
    assert result["detection"]["family"] == "amber"


def test_unsupported_forcefield_family_is_rejected(tmp_path):
    top = _write_top(tmp_path, ["oplsaa.ff/forcefield.itp"])
    with pytest.raises(ValueError, match="本轮不支持"):
        core.resolve_forcefield_family(top_path=top)
    with pytest.raises(ValueError, match="本轮不支持"):
        core.resolve_forcefield_family(explicit_family="gromos")


# ---------------------------------------------------------------------------
# §1.3 / B6 色散路线
# ---------------------------------------------------------------------------


def test_soluble_defaults_to_legacy_uniform_lrc_so_behaviour_is_unchanged():
    """§7.7：可溶体系不声明时仍是改动前的 `lrc_coeff/V` 行为。"""
    payload = core.resolve_dispersion_protocol(None)
    assert payload["dispersion_protocol"] == "legacy_uniform_density_lrc"
    assert payload["was_defaulted"] is True
    assert payload["uniform_density_lrc_active"] is True
    assert payload["implemented"] is True


def test_membrane_without_dispersion_protocol_fails_closed():
    """§7.1：membrane + 未指定 dispersion protocol → 失败。"""
    with pytest.raises(ValueError, match="没有选择 dispersion_protocol"):
        core.resolve_dispersion_protocol(None, environment_type="membrane")


def test_membrane_error_message_names_the_expected_protocol_for_the_family():
    """报错要能直接指路，而不是只说"你没选"。"""
    with pytest.raises(ValueError, match="ff_native_isotropic_lrc"):
        core.resolve_dispersion_protocol(
            None, environment_type="membrane", forcefield_family="amber"
        )


def test_membrane_with_legacy_uniform_lrc_fails_closed():
    """§1.3 修正框第 1 条 / §6.4：均匀体相密度假设在膜口袋里直接不成立。"""
    with pytest.raises(ValueError, match="均匀体相密度"):
        core.resolve_dispersion_protocol(
            "legacy_uniform_density_lrc", environment_type="membrane"
        )


def test_membrane_with_amber_native_lrc_passes():
    """⚠️ 注意这是 §1.3 修正框的要点：Amber 脂质**必须**开各向同性 LRC，关掉才是错的。"""
    payload = core.resolve_dispersion_protocol(
        "ff_native_isotropic_lrc",
        environment_type="membrane",
        forcefield_family="amber",
    )
    assert payload["dispersion_protocol"] == "ff_native_isotropic_lrc"
    assert payload["uniform_density_lrc_active"] is False
    assert payload["implemented"] is True


def test_charmm_force_switch_fails_closed_without_quantitative_evidence():
    """OpenMM 只有 potential-switch，没有 force-switch —— 默认卡住。"""
    with pytest.raises(ValueError, match="没有.*force-switch"):
        core.resolve_dispersion_protocol(
            "ff_native_force_switch_no_lrc",
            forcefield_family="charmm",
        )


@pytest.mark.parametrize("dropped", core.FORCE_SWITCH_DEVIATION_EVIDENCE_FIELDS)
def test_charmm_force_switch_needs_every_evidence_field(dropped):
    evidence = {f: "/path" for f in core.FORCE_SWITCH_DEVIATION_EVIDENCE_FIELDS}
    del evidence[dropped]
    with pytest.raises(ValueError, match="缺少"):
        core.resolve_dispersion_protocol(
            "ff_native_force_switch_no_lrc",
            forcefield_family="charmm",
            force_switch_deviation_evidence=evidence,
        )


def test_charmm_force_switch_still_not_membrane_validated_even_with_evidence():
    """有了论证也只是解开 force-switch 那道锁；membrane 已验证集合仍只含 amber 路线。"""
    evidence = {f: "/path" for f in core.FORCE_SWITCH_DEVIATION_EVIDENCE_FIELDS}
    with pytest.raises(ValueError, match="尚未验证"):
        core.resolve_dispersion_protocol(
            "ff_native_force_switch_no_lrc",
            environment_type="membrane",
            forcefield_family="charmm",
            force_switch_deviation_evidence=evidence,
        )


@pytest.mark.parametrize(
    "protocol, route",
    [("lj_pme", "路线 B"), ("membrane_inhomogeneous", "路线 C")],
)
def test_routes_b_and_c_are_recognized_but_not_implemented(protocol, route):
    """声明未实现的路线要报 NotImplementedError，而不是被当成拼错的未知值。

    §1.3 明确写着：路线 B 在四项验收完成前"不能只把基础 NonbondedForce 切成
    LJPME 就宣称支持"。
    """
    with pytest.raises(NotImplementedError, match=route):
        core.resolve_dispersion_protocol(protocol)


def test_family_and_protocol_mismatch_is_rejected():
    """跟随力场原始参数化条件：amber 配 force-switch、charmm 配 isotropic LRC 都是错的。"""
    with pytest.raises(ValueError, match="原始参数化条件"):
        core.resolve_dispersion_protocol(
            "ff_native_force_switch_no_lrc",
            forcefield_family="amber",
            force_switch_deviation_evidence={
                f: "/path" for f in core.FORCE_SWITCH_DEVIATION_EVIDENCE_FIELDS
            },
        )


def test_misspelled_dispersion_protocol_fails():
    with pytest.raises(ValueError, match="非法"):
        core.resolve_dispersion_protocol("no_lrc")


def test_apbs_is_recorded_as_orthogonal_to_dispersion():
    """§5 末条 / §6.5：APBS 只管静电有限尺寸项，不得当成 LJ 修正。"""
    payload = core.resolve_dispersion_protocol(None)
    assert payload["apbs_is_orthogonal_to_dispersion"] is True


# ---------------------------------------------------------------------------
# §13 阈值常量
# ---------------------------------------------------------------------------


def test_coion_runtime_distance_threshold_matches_softcore_cutoff():
    """§13.1 把"全程 ≥ 1.2 nm"定义为 **= softcore cutoff**。

    `abfe_core` 在 `ibs_engine` 下层不能反向 import，所以这个值在两处各有一份。
    这条测试就是防止两处各改一半的唯一防线——改了 `SOFTCORE_CUTOFF_NM` 就会在这里失败。
    """
    import ibs_engine

    assert core.COION_LIGAND_MIN_IMAGE_RUNTIME_NM == pytest.approx(
        ibs_engine.SOFTCORE_CUTOFF_NM
    )


def test_initial_coion_distance_is_stricter_than_runtime():
    """选择时留余量，否则第一帧就贴着下限、稍一抖动就破门。"""
    assert (
        core.COION_LIGAND_MIN_IMAGE_INITIAL_NM
        > core.COION_LIGAND_MIN_IMAGE_RUNTIME_NM
    )


def test_flat_bottom_restraint_replaces_the_legacy_bare_harmonic():
    """§13.1 对比现状 MEM-00d：当前是 k=25 的纯谐振子、无平坦区。"""
    assert core.COION_FLAT_BOTTOM_RADIUS_NM > 0.0
    assert core.COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2 == pytest.approx(100.0)
    assert core.COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2 > 25.0


def test_decoupled_endpoint_tolerance_is_strict_zero_not_merely_small():
    """§13.2：λ=0 端 ligand–environment 能量是**严格零**，不是"很小"。"""
    assert core.DECOUPLED_ENDPOINT_ENERGY_ABS_TOLERANCE_KJ_PER_MOL <= 1.0e-6


def test_acceptance_thresholds_payload_is_json_serializable_and_versioned():
    import json

    payload = core.acceptance_thresholds_payload()
    assert payload["version"] == core.ACCEPTANCE_THRESHOLDS_VERSION
    json.dumps(payload)
    for section in (
        "coion_geometry",
        "numerical_selfconsistency",
        "membrane_quality_gate",
        "result_acceptance",
    ):
        assert payload[section], f"{section} 不能为空"


def test_thresholds_payload_reflects_the_constants_not_a_hardcoded_copy():
    """payload 必须读常量，否则改了常量而 provenance 还报旧值。"""
    payload = core.acceptance_thresholds_payload()
    assert payload["coion_geometry"]["ligand_min_image_runtime_nm"] == (
        core.COION_LIGAND_MIN_IMAGE_RUNTIME_NM
    )
    assert payload["result_acceptance"]["min_independent_repeats"] == (
        core.MIN_INDEPENDENT_REPEATS
    )
    assert payload["numerical_selfconsistency"]["total_charge_conservation_e"] == (
        core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E
    )
