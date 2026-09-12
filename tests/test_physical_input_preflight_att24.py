"""物理输入预检（ATT-24 / GitHub issue #64）。

## 为什么要在开跑前判

这五类问题原来都要等到**建完 System、甚至跑起来之后**才以别的面目暴露：

| 真实问题 | 原来的表现 |
|---|---|
| `.top` 有未解析的 `#include` | "某个 moleculetype 找不到" |
| `--ligand` 打错 | "ligand_indices 为空" |
| `.gro` 与 `.top` 原子数不符 | 所有按序号取的原子选择静默错位（配体、co-ion、锚点） |
| 盒子 < 2×cutoff | **PME 不报错**，只是给出错误的静电 |
| 配体/受体太小 | Boresch 六个锚点定义不出来 |

其中盒子那条是唯一"不查就永远查不出来"的 —— 同一对原子被自己的周期镜像重复
作用，能量有限、力有限、所有收敛门都通过，只是数不对。

## 覆盖

用仓库自带的真实 vendored fixture（45354 原子的 CHARMM-GUI 膜体系 +
41 原子的配体单体），不造假体系 —— 造出来的小体系测不到真实格式里的坑
（例如那个配体 `.gro` 的**全零盒子**会让 `app.GromacsGroFile` 抛
`ZeroDivisionError`，这正是预检要如实报告的一种输入）。
"""
from __future__ import annotations

import pathlib

import pytest

import abfe_core as core

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "memtest"
COMPLEX_GRO = FIXTURES / "complex.gro"
COMPLEX_TOP = FIXTURES / "topol.top"
LIGAND_GRO = FIXTURES / "Atenolol-rank11.gro"
LIGAND_TOP = FIXTURES / "Atenolol-rank11.top"

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        not (COMPLEX_GRO.is_file() and COMPLEX_TOP.is_file()),
        reason="需要 memtest/complex.gro + topol.top 这对 vendored fixture",
    ),
]


def _quiet(_message):
    return None


def test_real_complex_inputs_pass_and_report_the_reconciled_counts():
    report = core.preflight_physical_inputs(
        str(COMPLEX_GRO), str(COMPLEX_TOP), "PROA", cutoff_nm=1.0, log=_quiet
    )
    assert report["n_atoms_topology"] == report["n_atoms_gro"] == 45354
    assert report["ligand_n_atoms"] == 4566
    assert report["min_box_edge_nm"] == pytest.approx(6.53673, abs=1e-5)


def test_unknown_ligand_name_lists_the_real_candidates():
    """打错名字时要直接给候选，而不是让它一路走到"配体 0 个原子"。"""
    with pytest.raises(ValueError) as excinfo:
        core.preflight_physical_inputs(
            str(COMPLEX_GRO), str(COMPLEX_TOP), "NOT-A-MOLECULE", cutoff_nm=1.0, log=_quiet
        )
    message = str(excinfo.value)
    assert "没有对应的 [ moleculetype ]" in message
    # 候选里必须排掉水和离子，否则等于没提示。
    assert "PROA" in message
    assert "TP3" not in message and "Na+" not in message


def test_box_smaller_than_twice_cutoff_is_rejected():
    """这条是唯一"不查就永远查不出来"的 —— PME 在小盒子上不报错，只是算错。"""
    with pytest.raises(ValueError, match="minimum image"):
        core.preflight_physical_inputs(
            str(COMPLEX_GRO), str(COMPLEX_TOP), "PROA", cutoff_nm=4.0, log=_quiet
        )


def test_box_exactly_two_times_cutoff_is_accepted():
    """边界必须是 `<` 而不是 `<=`：恰好 2×cutoff 是合法的。"""
    core.preflight_physical_inputs(
        str(COMPLEX_GRO), str(COMPLEX_TOP), "PROA",
        cutoff_nm=6.53673 / 2.0, log=_quiet,
    )


@pytest.mark.skipif(
    not (LIGAND_GRO.is_file() and LIGAND_TOP.is_file()),
    reason="需要 memtest 的配体单体 fixture",
)
def test_all_zero_box_is_reported_as_such_not_as_a_division_error():
    """配体单体 `.gro` 的盒子是全零。

    `app.GromacsGroFile` 在这种文件上抛 `ZeroDivisionError: float division by
    zero` —— 一条谁也看不懂的错。预检自己读 `.gro` 的第 2 行和末行，
    所以能如实说"这个坐标文件没有周期盒子，多半是未溶剂化的配体单体"。
    """
    with pytest.raises(ValueError) as excinfo:
        core.preflight_physical_inputs(
            str(LIGAND_GRO), str(LIGAND_TOP), "Atenolol-rank11",
            cutoff_nm=0.5, log=_quiet,
        )
    message = str(excinfo.value)
    assert "全零" in message and "没有周期盒子" in message
    assert "ZeroDivisionError" not in message


def test_missing_files_fail_before_anything_else_is_attempted():
    with pytest.raises(ValueError, match="文件不存在或不可读"):
        core.preflight_physical_inputs(
            "/nonexistent.gro", str(COMPLEX_TOP), "PROA", cutoff_nm=1.0, log=_quiet
        )


def test_all_problems_are_reported_in_one_go():
    """输入错误常常成组出现；一次改完比来回五次好。"""
    with pytest.raises(ValueError) as excinfo:
        core.preflight_physical_inputs(
            str(COMPLEX_GRO), str(COMPLEX_TOP), "NOT-A-MOLECULE",
            cutoff_nm=4.0, log=_quiet,
        )
    message = str(excinfo.value)
    assert "没有对应的 [ moleculetype ]" in message
    assert "minimum image" in message, "第二个问题被第一个挡住了 —— 应当一次列全"


@pytest.mark.skipif(
    not (LIGAND_GRO.is_file() and LIGAND_TOP.is_file()),
    reason="需要 memtest 的配体单体 fixture",
)
def test_boresch_requires_a_receptor_besides_the_ligand():
    """只有配体的拓扑不该能声明 Boresch —— 三个受体锚点无从取。

    用配体单体 fixture：它的 `[ molecules ]` 里只有 `Atenolol-rank11` 一项。
    该输入同时还有全零盒子的问题，正好一并验证"一次列全"。
    """
    with pytest.raises(ValueError) as excinfo:
        core.preflight_physical_inputs(
            str(LIGAND_GRO), str(LIGAND_TOP), "Atenolol-rank11",
            cutoff_nm=0.5, require_boresch=True, log=_quiet,
        )
    message = str(excinfo.value)
    assert "Boresch" in message and "找不到任何 ≥3 原子的非水非离子分子" in message
    assert "没有周期盒子" in message, "两个问题应当一次列全"


def test_no_false_positive_on_the_repository_membrane_fixture():
    """回归护栏：真实体系必须**通过**。

    预检误报会直接拦下生产运行，比漏报更难受。本条钉住 vendored 的
    45354 原子膜体系在生产口径（cutoff=1.0nm）下无条件通过。
    """
    core.preflight_physical_inputs(
        str(COMPLEX_GRO), str(COMPLEX_TOP), "PROA",
        cutoff_nm=1.0, require_boresch=True, log=_quiet,
    )


# ---------------------------------------------------------------------------
# `--ligand` 的两种写法都必须认（2026-09-10）
# ---------------------------------------------------------------------------
#
# 预检最初只按 `[ moleculetype ]` 名查，而全仓其余 7 处消费 `--ligand` 的地方
# 都按**残基名**。两种命名在 memtest/Atenolol 这条线上恰好不同 ⟹ 传残基名被
# 预检打死、传 moleculetype 名又在建系时"未找到配体残基" —— **两条路都走不通**，
# 预检把一条本来能跑的线 100% 挡死了。
#
# 这正是"预检误报比漏报更难受"那条的实例：它不是没抓到问题，它是造了个问题。


def test_ligand_accepts_a_moleculetype_name():
    report = core.preflight_physical_inputs(
        str(COMPLEX_GRO), str(COMPLEX_TOP), "PROA", cutoff_nm=1.0, log=_quiet
    )
    assert report["ligand_n_atoms"] == 4566


def test_ligand_accepts_a_residue_name_and_counts_only_that_residue():
    """`MOL` 是残基名、不是 moleculetype 名 —— 生产其余 7 处消费者用的就是它。

    原子数必须只数**该残基**的原子（41），不能退化成整个 moleculetype。
    """
    report = core.preflight_physical_inputs(
        str(COMPLEX_GRO), str(COMPLEX_TOP), "MOL", cutoff_nm=1.0, log=_quiet
    )
    assert report["ligand_n_atoms"] == 41, (
        "按残基名匹配时应只数该残基的原子；数成 4566 说明退化成了整个 moleculetype"
    )


def test_candidate_list_does_not_dump_every_amino_acid():
    """名字打错时给的候选必须能用。

    倒出 20 个氨基酸残基名 = 等于没提示。只列 moleculetype 名 +
    单残基 moleculetype 的残基名。
    """
    with pytest.raises(ValueError) as excinfo:
        core.preflight_physical_inputs(
            str(COMPLEX_GRO), str(COMPLEX_TOP), "NOT-A-MOLECULE",
            cutoff_nm=1.0, log=_quiet,
        )
    message = str(excinfo.value)
    amino_acids = ("ALA", "GLY", "LEU", "SER", "VAL", "LYS", "GLU", "ASP")
    leaked = [name for name in amino_acids if name in message]
    assert not leaked, f"候选里混进了氨基酸残基名，等于没提示: {leaked}"
