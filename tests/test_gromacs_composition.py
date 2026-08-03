"""组成驱动的身份判定：`.top` 是权威，残基名不是（memtodolist §1.1 / §3.3 / §9）。

## 为什么要有这一层

`memtest/` 那套 CHARMM-GUI FF-Converter 产的 **AMBER** 膜体系，把之前所有基于
残基名的判据全打穿了，而且**四处都是静默出错、不是报错**：

| 实际 | 旧判据 | 后果 |
| --- | --- | --- |
| include 全是 `toppar/*.itp`，不含 `amber` 字样 | 只看 include 路径 token | 力场族 fail closed，跑不起来 |
| 一个 `POPC` = `PA` + `PC` + `OL` **三个残基** | 按残基计数 | 脂质数 ×3、**APL 错 3 倍**；尾链无磷原子 → 直接 raise |
| 水叫 `TP3`（mdtraj 水表只有 `TIP3`） | `select("water")` | 疏水核内水、co-ion 首层水**静默为 0** |
| 离子叫 `Na+` / `Cl-`（带符号） | 名字集合无 `NA+`/`CL-` | 离子计数**静默为 0** |
| 蛋白含 `HID`/`ASH`/`NTRP`/`CCYS` | mdtraj `protein`（表里只有 `HIP`） | 骨架原子**静默少选 85 个** |

根因不是"少了几个名字"——往硬编码集合里补名字是改产物不改生成器，换一套体系又挂。
`.top` 里本来就写着权威答案（`[ molecules ]` + `[ moleculetype ]`），所以身份一律
从那里来。

## 本文件用真实文件做断言

`memtest/topol.top`（AMBER 膜体系）与仓库根的 `topol.top`（现有可溶体系）都在版本库里，
纯文本解析、无 GPU、秒级。用真实文件而不是全靠合成拓扑，是因为上面那五条恰恰都是
"合成拓扑想不到要造"的情形。
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

pytest.importorskip("openmm")

import abfe_core as core

ROOT = Path(__file__).absolute().parents[1]
MEMBRANE_TOP = ROOT / "memtest" / "topol.top"
SOLUBLE_TOP = ROOT / "topol.top"

# 实测基准（`memtest/step7_production.gro` 共 45354 原子）
MEMBRANE_EXPECTED_MOLECULES = {
    "PROA": 1,
    "POPC": 90,
    "Na+": 25,
    "Cl-": 36,
    "TP3": 9542,
    "Atenolol-rank11": 1,
}
MEMBRANE_TOTAL_ATOMS = 45354


def _skip_if_missing(path: Path):
    if not path.is_file():
        pytest.skip(f"{path} 不在，跳过（该文件应随仓库提供）")


# ---------------------------------------------------------------------------
# 解析器
# ---------------------------------------------------------------------------


def test_membrane_top_molecules_and_moleculetypes_are_parsed():
    _skip_if_missing(MEMBRANE_TOP)
    parsed = core.parse_gromacs_topology(str(MEMBRANE_TOP))

    assert dict(parsed["molecules"]) == MEMBRANE_EXPECTED_MOLECULES
    # 一个 POPC 是**一个 moleculetype、三个残基**——这正是按残基计数会错 3 倍的原因。
    popc = parsed["moleculetypes"]["POPC"]
    assert popc["residue_names"] == ["PA", "PC", "OL"]
    assert popc["n_atoms"] == 134
    assert 134 * 90 == 12060  # 与 .gro 里 PA+PC+OL 的 4140+3420+4500 一致
    assert parsed["moleculetypes"]["TP3"]["n_atoms"] == 3
    assert parsed["moleculetypes"]["Na+"]["n_atoms"] == 1


def test_defaults_is_found_by_following_includes_not_only_in_the_top_level_file():
    """`topol.top` 自己没有 `[ defaults ]`，它在 `toppar/forcefield.itp` 里。"""
    _skip_if_missing(MEMBRANE_TOP)
    parsed = core.parse_gromacs_topology(str(MEMBRANE_TOP))
    defaults = parsed["defaults"]
    assert defaults is not None, "递归跟随 include 才能找到 [ defaults ]"
    assert defaults["comb_rule"] == 2
    assert defaults["fudge_lj"] == pytest.approx(0.5)
    assert defaults["fudge_qq"] == pytest.approx(0.833333, abs=1e-5)
    assert "forcefield.itp" in defaults["source"]


def test_molecule_atom_ranges_align_with_the_real_gro_file():
    """原子区间必须与 `.gro` 的实际原子顺序逐一对齐。

    `PROA` 有 4566 个原子，所以第一个 POPC 原子应当落在 index 4566 ——
    实测 `step7_production.gro` 里第一个 `PA` 原子正是第 4567 行（0-based 4566）。
    这条是整套按索引选择能否成立的地基：错位就会静默选错原子。
    """
    _skip_if_missing(MEMBRANE_TOP)
    parsed = core.parse_gromacs_topology(str(MEMBRANE_TOP))
    ranges = core.molecule_atom_ranges(parsed)

    assert ranges[0]["molecule_name"] == "PROA"
    assert (ranges[0]["start"], ranges[0]["stop"]) == (0, 4566)
    first_lipid = next(r for r in ranges if r["molecule_name"] == "POPC")
    assert (first_lipid["start"], first_lipid["stop"]) == (4566, 4700)
    assert ranges[-1]["stop"] == MEMBRANE_TOTAL_ATOMS


def test_total_atom_count_matches_the_coordinate_file():
    _skip_if_missing(MEMBRANE_TOP)
    gro = ROOT / "memtest" / "step7_production.gro"
    _skip_if_missing(gro)
    with open(gro, encoding="utf-8") as handle:
        handle.readline()
        n_atoms = int(handle.readline())
    parsed = core.parse_gromacs_topology(str(MEMBRANE_TOP))
    assert core.molecule_atom_ranges(parsed)[-1]["stop"] == n_atoms


def test_unknown_molecule_in_molecules_section_fails_closed(tmp_path):
    top = tmp_path / "topol.top"
    top.write_text(
        "[ defaults ]\n1 2 yes 0.5 0.8333\n"
        "[ moleculetype ]\nFOO 3\n[ atoms ]\n1 X 1 FOO A 1 0.0 1.0\n"
        "[ molecules ]\nBAR 1\n",
        encoding="utf-8",
    )
    parsed = core.parse_gromacs_topology(str(top))
    with pytest.raises(ValueError, match="找不到对应的 \\[ moleculetype \\]"):
        core.molecule_atom_ranges(parsed)


# ---------------------------------------------------------------------------
# §1.1 力场族：`[ defaults ]` 为主判据
# ---------------------------------------------------------------------------


def test_amber_membrane_system_is_detected_despite_no_amber_in_include_paths():
    """这条是 memtest 体系当初跑不起来的直接原因。

    include 路径全是 `toppar/*.itp`，一个 `amber` 字样都没有；唯一可靠证据是
    `[ defaults ]` 的 1-4 缩放（0.5 / 0.8333 = Amber；CHARMM 是 1.0 / 1.0）。
    """
    _skip_if_missing(MEMBRANE_TOP)
    result = core.detect_forcefield_family_from_top(str(MEMBRANE_TOP))
    assert result["family"] == "amber"
    assert result["defaults_family"] == "amber"
    assert result["reason"] == "defaults_1_4_scaling_signature"
    # 路径判据在这套体系里确实给不出答案——这正是需要 [ defaults ] 的原因。
    assert result["include_family"] is None

    resolved = core.resolve_forcefield_family(top_path=str(MEMBRANE_TOP))
    assert resolved["family"] == "amber"
    assert resolved["source"] == "auto_detected"


def test_existing_soluble_system_still_detected_as_amber():
    """回归：现有可溶体系走 include 路径判据（它的 `[ defaults ]` 在 .ff 目录里）。"""
    _skip_if_missing(SOLUBLE_TOP)
    result = core.detect_forcefield_family_from_top(str(SOLUBLE_TOP))
    assert result["family"] == "amber"
    assert result["include_family"] == "amber"


def test_charmm_defaults_signature_is_recognized(tmp_path):
    top = tmp_path / "topol.top"
    top.write_text(
        "[ defaults ]\n; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ\n"
        "1 2 yes 1.0 1.0\n"
        "[ moleculetype ]\nTP3 2\n[ atoms ]\n1 OW 1 TP3 OH2 1 -0.834 16.0\n"
        "[ molecules ]\nTP3 1\n",
        encoding="utf-8",
    )
    result = core.detect_forcefield_family_from_top(str(top))
    assert result["defaults_family"] == "charmm"
    assert result["family"] == "charmm"


def test_conflicting_defaults_and_include_evidence_fails_closed(tmp_path):
    """`[ defaults ]` 说 charmm、include 路径说 amber → 必须人工裁决，不许择一。"""
    ff = tmp_path / "amber99.ff"
    ff.mkdir()
    (ff / "forcefield.itp").write_text(
        "[ defaults ]\n1 2 yes 1.0 1.0\n", encoding="utf-8"
    )
    top = tmp_path / "topol.top"
    top.write_text(
        '#include "amber99.ff/forcefield.itp"\n'
        "[ moleculetype ]\nFOO 3\n[ atoms ]\n1 X 1 ALA CA 1 0.0 12.0\n"
        "[ molecules ]\nFOO 1\n",
        encoding="utf-8",
    )
    result = core.detect_forcefield_family_from_top(str(top))
    assert result["family"] is None
    assert result["reason"].startswith("conflict:")
    with pytest.raises(ValueError, match="无法从"):
        core.resolve_forcefield_family(top_path=str(top))


# ---------------------------------------------------------------------------
# 角色判定
# ---------------------------------------------------------------------------


def test_real_membrane_system_roles_are_all_resolved():
    _skip_if_missing(MEMBRANE_TOP)
    parsed = core.parse_gromacs_topology(str(MEMBRANE_TOP))
    comp = core.classify_system_composition(
        parsed, ligand_molecule_name="Atenolol-rank11"
    )

    assert comp["roles"] == {
        "Atenolol-rank11": "ligand",
        "PROA": "protein",
        "POPC": "lipid",
        "Na+": "ion",
        "Cl-": "ion",
        "TP3": "water",
    }
    # 判据要可审计，不能只给结论。
    assert comp["role_evidence"]["POPC"] == "all_residues_are_lipid_residues"
    assert comp["role_evidence"]["TP3"] == "water_molecule_name"
    assert comp["role_evidence"]["Na+"] == "monoatomic_ion_name"
    assert comp["role_evidence"]["PROA"] == "contains_amino_acid_residues"


def test_counts_that_used_to_be_silently_zero_are_now_correct():
    """水与离子：旧判据在这套命名上是 0，现在必须是真实数量。"""
    _skip_if_missing(MEMBRANE_TOP)
    parsed = core.parse_gromacs_topology(str(MEMBRANE_TOP))
    comp = core.classify_system_composition(
        parsed, ligand_molecule_name="Atenolol-rank11"
    )
    atoms = comp["atom_indices_by_role"]
    assert len(atoms["water"]) == 9542 * 3      # 旧判据：0
    assert len(atoms["ion"]) == 25 + 36         # 旧判据：0
    assert len(atoms["protein"]) == 4566
    assert len(atoms["ligand"]) == 41
    assert len(atoms["lipid"]) == 90 * 134
    # 脂质是 **90 个分子**，不是 270 个残基。
    assert len(comp["molecules_by_role"]["lipid"]) == 90


def test_unclassifiable_moleculetype_fails_closed_instead_of_becoming_other(tmp_path):
    """判不出角色必须报错——静默归入 "other" 等于让它从所有原子选择里消失。"""
    top = tmp_path / "topol.top"
    top.write_text(
        "[ defaults ]\n1 2 yes 0.5 0.8333\n"
        "[ moleculetype ]\nWEIRD 3\n[ atoms ]\n1 X 1 ZZZ XX 1 0.0 12.0\n"
        "[ molecules ]\nWEIRD 1\n",
        encoding="utf-8",
    )
    parsed = core.parse_gromacs_topology(str(top))
    with pytest.raises(ValueError, match="无法判定 moleculetype"):
        core.classify_system_composition(parsed)


def test_declared_roles_can_override_and_are_recorded(tmp_path):
    top = tmp_path / "topol.top"
    top.write_text(
        "[ defaults ]\n1 2 yes 0.5 0.8333\n"
        "[ moleculetype ]\nWEIRD 3\n[ atoms ]\n1 X 1 ZZZ XX 1 0.0 12.0\n"
        "[ molecules ]\nWEIRD 2\n",
        encoding="utf-8",
    )
    parsed = core.parse_gromacs_topology(str(top))
    comp = core.classify_system_composition(
        parsed, declared_roles={"WEIRD": "lipid"}
    )
    assert comp["roles"]["WEIRD"] == "lipid"
    assert comp["role_evidence"]["WEIRD"] == "declared_override"
    assert comp["declared_roles"] == {"WEIRD": "lipid"}


def test_illegal_declared_role_is_rejected(tmp_path):
    top = tmp_path / "topol.top"
    top.write_text(
        "[ moleculetype ]\nFOO 3\n[ atoms ]\n1 X 1 ALA CA 1 0.0 12.0\n"
        "[ molecules ]\nFOO 1\n",
        encoding="utf-8",
    )
    parsed = core.parse_gromacs_topology(str(top))
    with pytest.raises(ValueError, match="非法角色"):
        core.classify_system_composition(parsed, declared_roles={"FOO": "membrane"})


# ---------------------------------------------------------------------------
# 蛋白残基名归一化（mdtraj 的表不够用）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("NTRP", "TRP"),   # N 端变体
        ("CCYS", "CYS"),   # C 端变体
        ("HID", "HID"),    # Amber 组氨酸质子化态
        ("HIE", "HIE"),
        ("ASH", "ASH"),
        ("CYX", "CYX"),
        ("ALA", "ALA"),
        ("TP3", None),     # 水不是氨基酸
        ("POPC", None),
        ("Na+", None),
    ],
)
def test_protein_residue_normalization_covers_amber_variants(name, expected):
    """mdtraj 的 `_AMINO_ACID_CODES` 里只有 `HIP`，HID/HIE/ASH/CYX/N-/C- 端全不在，
    直接用它的 `protein` 关键字会静默少选骨架原子。
    """
    assert core.normalize_protein_residue_name(name) == expected


def test_real_membrane_protein_residues_are_all_recognized():
    """memtest 的 PROA 含 NTRP / CCYS / HID / ASH —— 一个都不能漏。"""
    _skip_if_missing(MEMBRANE_TOP)
    parsed = core.parse_gromacs_topology(str(MEMBRANE_TOP))
    residues = parsed["moleculetypes"]["PROA"]["residue_names"]
    for special in ("NTRP", "CCYS", "HID", "ASH"):
        assert special in residues, f"{special} 应出现在 PROA 里（实测如此）"
    unrecognized = [
        r for r in residues if core.normalize_protein_residue_name(r) is None
    ]
    assert unrecognized == [], f"这些蛋白残基名没被认出来：{unrecognized}"


# ---------------------------------------------------------------------------
# 名表本身
# ---------------------------------------------------------------------------


def test_water_and_ion_name_tables_cover_the_real_system():
    assert "TP3" in core.WATER_MOLECULE_NAMES, "CHARMM-GUI AMBER 转换器的水就叫 TP3"
    assert "TIP3" in core.WATER_MOLECULE_NAMES
    assert core._normalize_ion_name("Na+") == "NA"
    assert core._normalize_ion_name("Cl-") == "CL"
    assert core._normalize_ion_name("Na+") in core.MONOATOMIC_ION_NAMES
    assert core._normalize_ion_name("Cl-") in core.MONOATOMIC_ION_NAMES


def test_amber_modular_lipid_residues_are_all_in_the_lipid_table():
    """`PA` / `PC` / `OL` 必须都在脂质名表里，否则 POPC 判不成 lipid。"""
    for residue in ("PA", "PC", "OL"):
        assert residue in core.KNOWN_LIPID_RESIDUE_NAMES
    # 但它们**不**在无歧义表里——那是拦"soluble 却有大量脂质"用的，
    # 用短 token 拦人会误伤合法的可溶体系。
    for residue in ("PA", "PC", "OL"):
        assert residue not in core.LIPID_RESIDUE_NAMES_UNAMBIGUOUS
