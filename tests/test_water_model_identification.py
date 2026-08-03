"""水模型识别：文件名认不出时按**参数**判，且参数取自 OpenMM 自带 XML。

## 背景：同一类根因的第三次出现

`resolve_water_model_xml()` 原来只看 `#include` 的**文件名词干**。那对
`amber14sb_OL15_fs1.ff/tip3p.itp` 有效，但 CHARMM-GUI 的 AMBER 转换器把 TIP3P
命名为 **`TP3`**（`memtest/toppar/TP3.itp`），于是首跑直接死在：

    ValueError: 在 topol.top 的 #include 里没认出任何水模型；
                已知的有 ['opc','opc3','spce','tip3p','tip3pfb','tip4pew','tip4pfb']

前两次是同一个毛病：脂质按残基名计数（POPC = PA+PC+OL，数成 3 倍）、
水/离子按残基名计数（`TP3`/`Na+`/`Cl-` 静默为 0）。**靠名字判身份在换一套体系时
就会错**，所以这里改成按参数判。

## 为什么不硬编码参数表

候选参数直接从 OpenMM 自带的水模型 XML 读出来（`openmm/app/data/amber14/*.xml`）。
这样"最终选出的 XML"与"用于比对的参数"构造性地来自同一处——不会出现表抄错、
也不会因 OpenMM 升级而漂移。

## 实测（本文件断言的就是这些数）

`TP3` 的参数：3 位点，q_O = −0.834，q_H = +0.417，
σ_O = 0.315075240658 nm，ε_O = 0.635968 kJ/mol
→ 在 7 个候选里**唯一**匹配 `amber14/tip3p.xml`（σ 逐位吻合）。
最接近的竞争者 SPC/E 的 σ 差 1.5e-3 nm，是容差 1e-6 的 1500 倍，不会误判。
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

pytest.importorskip("openmm")

import abfe_core as core

ROOT = Path(__file__).absolute().parents[1]
MEMBRANE_TOP = ROOT / "memtest" / "topol.top"
SOLUBLE_TOP = ROOT / "topol.top"


def _skip_if_missing(path: Path):
    if not path.is_file():
        pytest.skip(f"{path} 不在，跳过（该文件应随仓库提供）")


# ---------------------------------------------------------------------------
# 从拓扑取参数
# ---------------------------------------------------------------------------


def test_tp3_parameters_are_extracted_from_the_real_topology():
    _skip_if_missing(MEMBRANE_TOP)
    parsed = core.parse_gromacs_topology(str(MEMBRANE_TOP))
    params = core.water_model_parameters_from_topology(parsed)
    assert params is not None
    assert params["moleculetype"] == "TP3"
    assert params["n_sites"] == 3
    assert params["charge_o_e"] == pytest.approx(-0.834)
    assert params["charge_h_e"] == pytest.approx(0.417)
    assert params["sigma_o_nm"] == pytest.approx(0.315075240658, abs=1e-12)
    assert params["epsilon_o_kj_mol"] == pytest.approx(0.635968, abs=1e-9)


def test_atomtypes_are_now_parsed_from_the_topology_tree():
    """σ/ε 在 `forcefield.itp` 的 `[ atomtypes ]` 里，必须递归拿到。"""
    _skip_if_missing(MEMBRANE_TOP)
    parsed = core.parse_gromacs_topology(str(MEMBRANE_TOP))
    atomtypes = parsed["atomtypes"]
    assert "OW" in atomtypes and "HW" in atomtypes
    assert atomtypes["OW"]["sigma_nm"] == pytest.approx(0.315075240658, abs=1e-12)
    assert atomtypes["OW"]["epsilon_kj_mol"] == pytest.approx(0.635968, abs=1e-9)
    # 氢无 LJ。
    assert atomtypes["HW"]["sigma_nm"] == pytest.approx(0.0)
    assert atomtypes["HW"]["epsilon_kj_mol"] == pytest.approx(0.0)


def test_moleculetype_atoms_now_carry_type_and_charge():
    """按参数识别水需要每个原子的 atomtype，不只是残基名。"""
    _skip_if_missing(MEMBRANE_TOP)
    parsed = core.parse_gromacs_topology(str(MEMBRANE_TOP))
    atoms = parsed["moleculetypes"]["TP3"]["atoms"]
    assert [a["type"] for a in atoms] == ["OW", "HW", "HW"]
    assert [a["atom_name"] for a in atoms] == ["O", "H1", "H2"]
    assert atoms[0]["charge"] == pytest.approx(-0.834)


# ---------------------------------------------------------------------------
# 候选参数来自 OpenMM XML，不是硬编码表
# ---------------------------------------------------------------------------


def test_candidate_parameters_are_read_from_openmm_data_files():
    params = core.openmm_water_model_parameters("amber14/tip3p.xml")
    assert params is not None
    assert params["n_sites"] == 3
    assert params["charge_o_e"] == pytest.approx(-0.834)
    assert params["sigma_o_nm"] == pytest.approx(0.31507524065751241, abs=1e-12)
    assert params["epsilon_o_kj_mol"] == pytest.approx(0.635968, abs=1e-9)


def test_missing_xml_returns_none_rather_than_raising():
    assert core.openmm_water_model_parameters("amber14/does_not_exist.xml") is None


def test_all_supported_models_expose_readable_parameters():
    """整张映射表里的 XML 都必须能读出参数，否则匹配会静默漏掉候选。"""
    unreadable = [
        key
        for key, relative in core.GMX_TO_OPENMM_WATER_XML.items()
        if core.openmm_water_model_parameters(relative) is None
    ]
    assert unreadable == [], f"这些水模型 XML 读不出参数：{unreadable}"


# ---------------------------------------------------------------------------
# 匹配
# ---------------------------------------------------------------------------


def test_tp3_is_identified_as_tip3p_by_parameters():
    """核心断言：`TP3` 这个名字认不出，但参数唯一匹配 TIP3P。"""
    _skip_if_missing(MEMBRANE_TOP)
    result = core.identify_water_model_by_parameters(str(MEMBRANE_TOP))
    assert result["matched"] is True
    assert result["xml"] == "amber14/tip3p.xml"
    assert result["model_key"] == "tip3p"
    assert result["reason"] == "parameter_fingerprint"


def test_the_match_is_unique_not_merely_first_hit():
    """必须**唯一**匹配。若有多个候选都落在容差内就不能选——那说明容差太松。

    实测最接近的竞争者是 SPC/E，σ 差 1.5e-3 nm ≈ 容差(1e-6) 的 1500 倍。
    """
    _skip_if_missing(MEMBRANE_TOP)
    parsed = core.parse_gromacs_topology(str(MEMBRANE_TOP))
    topology_params = core.water_model_parameters_from_topology(parsed)
    matches = [
        key
        for key, relative in core.GMX_TO_OPENMM_WATER_XML.items()
        if (xml := core.openmm_water_model_parameters(relative))
        and core._water_models_match(topology_params, xml)
    ]
    assert matches == ["tip3p"]

    spce = core.openmm_water_model_parameters("amber14/spce.xml")
    margin = abs(topology_params["sigma_o_nm"] - spce["sigma_o_nm"])
    assert margin > 100 * core.WATER_MODEL_MATCH_SIGMA_TOLERANCE_NM, (
        f"最近竞争者的 σ 余量只有 {margin:.3g} nm，容差可能太松"
    )


def test_resolve_water_model_xml_falls_back_to_parameters_for_the_membrane_system():
    """端到端：这就是首跑挂掉的那个调用，现在应当返回 tip3p。"""
    _skip_if_missing(MEMBRANE_TOP)
    xml, source = core.resolve_water_model_xml(str(MEMBRANE_TOP))
    assert xml == "amber14/tip3p.xml"
    # 来源要标明是参数匹配，而不是伪装成文件名命中。
    assert source.startswith("parameter_match:")
    assert "TP3" in source


def test_filename_path_still_wins_for_the_existing_soluble_system():
    """回归：现有可溶体系仍走文件名词干，行为不变。"""
    _skip_if_missing(SOLUBLE_TOP)
    xml, source = core.resolve_water_model_xml(str(SOLUBLE_TOP))
    assert xml == "amber14/tip3p.xml"
    assert source.endswith("tip3p.itp")
    assert not source.startswith("parameter_match:")


# ---------------------------------------------------------------------------
# fail closed
# ---------------------------------------------------------------------------


def _write_water_topology(tmp_path, q_o, q_h, sigma, epsilon, moltype="TP3"):
    top = tmp_path / "topol.top"
    top.write_text(
        "[ defaults ]\n1 2 yes 0.5 0.8333\n"
        "[ atomtypes ]\n"
        f"OW 8 16.0000 {q_o} A {sigma} {epsilon}\n"
        f"HW 1 1.0080 {q_h} A 0.00000000000e+00 0.000000e+00\n"
        "[ moleculetype ]\n" + moltype + " 1\n"
        "[ atoms ]\n"
        f"1 OW 1 {moltype} O 1 {q_o} 16.0000\n"
        f"2 HW 1 {moltype} H1 2 {q_h} 1.0080\n"
        f"3 HW 1 {moltype} H2 3 {q_h} 1.0080\n"
        "[ molecules ]\n" + moltype + " 1\n",
        encoding="utf-8",
    )
    return top


def test_unknown_water_parameters_fail_closed_with_the_measured_numbers(tmp_path):
    """参数对不上任何已知模型时报错，并把实测数字写进错误信息。

    否则用户只知道"认不出"，不知道自己的水到底是什么参数。
    """
    top = _write_water_topology(
        tmp_path, "-0.700000", "0.350000", "3.00000000000e-01", "5.000000e-01"
    )
    with pytest.raises(ValueError, match="按参数也没匹配上"):
        core.resolve_water_model_xml(str(top))
    with pytest.raises(ValueError, match=r"σ_O=0\.300000000 nm"):
        core.resolve_water_model_xml(str(top))


def test_spce_named_oddly_is_still_identified(tmp_path):
    """不只对 TIP3P 有效：把 SPC/E 起个怪名字也应当按参数认出来。"""
    spce = core.openmm_water_model_parameters("amber14/spce.xml")
    top = _write_water_topology(
        tmp_path,
        f"{spce['charge_o_e']:.6f}",
        f"{spce['charge_h_e']:.6f}",
        f"{spce['sigma_o_nm']:.12e}",
        f"{spce['epsilon_o_kj_mol']:.6e}",
        moltype="SOL",
    )
    xml, source = core.resolve_water_model_xml(str(top))
    assert xml == "amber14/spce.xml"
    assert source.startswith("parameter_match:")


def test_topology_without_water_returns_no_parameters(tmp_path):
    top = tmp_path / "topol.top"
    top.write_text(
        "[ defaults ]\n1 2 yes 0.5 0.8333\n"
        "[ atomtypes ]\nCT 6 12.01 0.0 A 0.34 0.45\n"
        "[ moleculetype ]\nFOO 3\n"
        "[ atoms ]\n1 CT 1 ALA CA 1 0.0 12.01\n"
        "[ molecules ]\nFOO 1\n",
        encoding="utf-8",
    )
    parsed = core.parse_gromacs_topology(str(top))
    assert core.water_model_parameters_from_topology(parsed) is None
    result = core.identify_water_model_by_parameters(str(top))
    assert result["matched"] is False
    assert result["reason"] == "topology_water_parameters_unavailable"
