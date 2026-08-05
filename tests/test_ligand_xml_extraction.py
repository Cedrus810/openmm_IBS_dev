"""P0-13：从 GROMACS .top 抽配体参数时，**不许对同类型力做单例假设**。

对应 `docs/TODO.md` 的 P0-13，以及它引起的 P0-12（溶剂腿构象塌缩）。

## 缺陷是什么（2026-08-04 实测事故）

`runabfe.generate_ligand_xml_from_top` 原先这样取力：

    angle_force = next((f for f in extracted_system.getForces()
                        if isinstance(f, openmm.HarmonicAngleForce)), None)

而 `memtest` 那个膜体系的 System 里有**两个** HarmonicAngleForce：

    force[2]  31401 个角，配体 0 个    ← next() 抓到的是这个
    force[4]     71 个角，配体 71 个   ← 配体的角全在这里

于是 `ligand_only.xml` 的 `<HarmonicAngleForce>` 段是**空的**，用它建出来的溶剂腿
配体**没有任何键角项** —— 分子是软的。后果一路滚下去：

    预平衡里配体从 0.996 塌到 0.660 nm 且 12 个 replica 再没恢复（σ=0.005 nm）
      ⟹ 极性基团聚拢，配体–水静电耦合强 3 倍（⟨U⟩ −569 vs −190 kJ/mol）
      ⟹ 溶剂腿去电荷 62.80 → 191.05 kJ/mol
      ⟹ ΔG_bind = +23.27 kcal/mol（本该是 −5 左右）

可溶体系只有**一个**角力（配体那 71 个混在里面），所以这个 bug 一直侥幸没被踩到，
07-29 的可溶基线 181.00 / 157.84 / −5.535906 不受影响。

## 不要这样让本文件变绿

把对账（写出项数 vs 源体系项数）改成 warning、或者把"多原子配体 0 个键角"那条
放行 —— 那正是让一个**软分子**悄悄跑完 7 小时的原因。
"""

import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

pytest.importorskip("openmm")
import openmm  # noqa: E402

ROOT = Path(__file__).absolute().parents[1]
MEMTEST_TOP = ROOT / "memtest" / "topol.top"


# ---------------------------------------------------------------------------
# 1. 真体系回归：那 71 个键角必须出现在 XML 里
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not MEMTEST_TOP.is_file(), reason="需要 memtest/topol.top")
def test_membrane_ligand_angles_survive_the_extraction(tmp_path):
    """本文件存在的理由：同一份输入，修前 <Angle> = 0，修后必须是 71。"""
    runabfe = pytest.importorskip("runabfe")
    path = runabfe.generate_ligand_xml_from_top(
        str(MEMTEST_TOP), "MOL", str(tmp_path), gmx_include_dir=None
    )
    xml = Path(path).read_text(encoding="utf-8")
    assert xml.count("<Angle ") == 71, (
        "配体键角又被丢了。源 .itp 里有 71 行 [angles]，"
        "抽取时必须聚合**所有** HarmonicAngleForce，不能 next() 取第一个"
    )
    # 其余项一并钉住，防止"修了角、漏了别的"。
    assert xml.count("<Bond ") == 82          # 41 个成键项 + 41 个 Residue 连接
    assert xml.count("<Proper ") == 96
    assert xml.count("<Improper ") == 8


@pytest.mark.skipif(not MEMTEST_TOP.is_file(), reason="需要 memtest/topol.top")
def test_membrane_system_really_has_more_than_one_angle_force(tmp_path):
    """把"为什么 next() 会错"钉成事实，而不是留在注释里。

    这条是上面那条的前提：如果哪天上游拓扑变成只有一个角力，本测试会失败，
    提醒维护者「聚合逻辑现在没有被真实体系覆盖了」——那时该找一个仍会触发的输入，
    而不是把聚合改回 next()。
    """
    from abfe_core import load_gromacs_topology_for_openmm
    from openmm import app

    top = load_gromacs_topology_for_openmm(str(MEMTEST_TOP), includeDir=None)
    system = top.createSystem(
        nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    angle_forces = [
        f for f in system.getForces() if isinstance(f, openmm.HarmonicAngleForce)
    ]
    assert len(angle_forces) >= 2, (
        "这个体系不再有多个 HarmonicAngleForce —— P0-13 的聚合逻辑失去真实覆盖"
    )
    lig = {a.index for a in top.topology.residues() if a.name == "MOL" for a in a.atoms()}
    per_force = [
        sum(
            1
            for i in range(f.getNumAngles())
            if all(int(x) in lig for x in f.getAngleParameters(i)[:3])
        )
        for f in angle_forces
    ]
    assert sum(per_force) == 71
    assert 0 in per_force, (
        "事故的形态是「第一个角力里配体一个都没有」；这个特征消失了就说明输入变了"
    )


# ---------------------------------------------------------------------------
# 2. 对账门：静默丢项必须 fail closed
# ---------------------------------------------------------------------------


def _minimal_top_without_angles(tmp_path: Path) -> Path:
    """一个 3 原子配体、**只有键没有键角**的合法 .top。

    这正是"抽取出来的参数让分子变软"的最小复现：多原子分子写出 0 个键角，
    下游没有任何环节会注意到，除了这道对账。
    """
    top = tmp_path / "floppy.top"
    top.write_text(
        textwrap.dedent(
            """\
            [ defaults ]
            1 2 yes 0.5 0.8333

            [ atomtypes ]
            CT 6 12.011 0.0000 A 0.339967 0.457730
            HC 1  1.008 0.0000 A 0.264953 0.065689

            [ moleculetype ]
            MOL 3

            [ atoms ]
            1 CT 1 MOL C1 1 -0.100 12.011
            2 HC 1 MOL H1 1  0.050  1.008
            3 HC 1 MOL H2 1  0.050  1.008

            [ bonds ]
            1 2 1 0.109 284512.0
            1 3 1 0.109 284512.0

            [ system ]
            floppy

            [ molecules ]
            MOL 1
            """
        ),
        encoding="utf-8",
    )
    return top


def test_polyatomic_ligand_with_zero_angles_fails_closed(tmp_path):
    """3 个原子、2 根键、0 个键角 ⟹ 必须 raise，不许产出软分子的 XML。"""
    runabfe = pytest.importorskip("runabfe")
    top = _minimal_top_without_angles(tmp_path)
    with pytest.raises(RuntimeError, match="一个键角项都没有|P0-13"):
        runabfe.generate_ligand_xml_from_top(
            str(top), "MOL", str(tmp_path / "out"), gmx_include_dir=None
        )


def test_extraction_reconciles_written_terms_against_the_source_system():
    """源码契约：写出的项数必须与源体系逐项对账，且不许降级成 warning。"""
    src = (ROOT / "runabfe.py").read_text(encoding="utf-8")
    assert "配体 XML 抽取对账失败（P0-13）" in src
    assert "n_written != expected" in src
    # 三类力都必须聚合（列表），不能再出现单例假设。
    for name in ("bond_forces", "angle_forces", "torsion_forces"):
        assert f"{name} = _forces_of(" in src, f"{name} 没有走聚合"
    assert "angle_force = next(" not in src, "单例假设又回来了（P0-13）"


def test_nonbonded_force_multiplicity_is_rejected_rather_than_guessed():
    """非键力是另一回事：多于一个时不许猜，直接报错让人收口拓扑。"""
    src = (ROOT / "runabfe.py").read_text(encoding="utf-8")
    assert "个 NonbondedForce" in src and "不要让本函数猜" in src
