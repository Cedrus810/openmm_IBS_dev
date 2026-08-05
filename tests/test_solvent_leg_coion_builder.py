"""B4：溶剂腿 reserved co-ion dummy 插入。

对应 `memtodolist.md` §4.1/§4.4，MEM 会话 2026-08-05。

## 缺陷是什么（已修，本文件现在是契约测试）

`runabfe.build_and_cache_solvent_leg` 之前对
`charge_treatment=co_alchemical_charge_transfer` 无条件 fail closed
（`NotImplementedError`）：溶剂腿盒子里没有建系时预留的中性 ion-shaped dummy，
`ibs_engine._identify_reserved_neutral_co_ions` 找不到任何"残基名在离子集合里
且电荷严格为 0"的粒子，热力学循环闭不上。

## 修法（2026-08-05，B4）

新增 `runabfe._insert_reserved_coalchemical_ion_dummies()`：摘掉离配体质心
minimum-image 距离最远的 N 个水分子，把 N 个 ion-shaped dummy（Na⁺/Cl⁻ 模板）
放在被摘掉的氧原子位置。调用方在 `createSystem()` 之后把这些粒子的电荷显式清零
——身份选择/restraint/charging 电荷映射全部复用既有的
`ibs_engine.select_co_alchemical_ion_once` / `abfe_core.build_co_alchemical_ion_identity`
（那两个已经是 leg-agnostic 的实现，本文件不重复测它们，只测新增的插入逻辑本身）。

## 不要这样让本文件变绿

不要把"随便摘一个水"当成通过——放置策略必须优先摘**最远**的水（§4.4 的安全边距
依赖这一点），任何改成"摘第一个匹配的水"的写法都要让下面的距离断言失败。
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

openmm = pytest.importorskip("openmm")
from openmm import Vec3, app, unit

import runabfe

BOX_NM = 4.0


def _build_synthetic_modeller(n_waters: int = 6):
    """LIG(1 原子，充当配体) + n_waters 个水，摆在不同 minimum-image 距离上。

    水按 index 0..n_waters-1 摆在沿 x 轴递增的距离上，所以"最远的水"永远是
    index 最大的那个（在没有 wrap 的情况下）——断言时直接按这个规律核对。
    """
    topology = app.Topology()
    topology.setPeriodicBoxVectors(
        (
            Vec3(BOX_NM, 0.0, 0.0),
            Vec3(0.0, BOX_NM, 0.0),
            Vec3(0.0, 0.0, BOX_NM),
        )
        * unit.nanometer
    )
    chain = topology.addChain()

    ligand_res = topology.addResidue("LIG", chain)
    topology.addAtom("C1", app.element.carbon, ligand_res)

    positions_nm = [[2.0, 2.0, 2.0]]  # 配体在盒子中心

    # 水的氧原子沿 x 轴远离配体，距离严格递增：index 越大离配体越远。
    for i in range(n_waters):
        water_res = topology.addResidue("HOH", chain)
        topology.addAtom("O", app.element.oxygen, water_res)
        topology.addAtom("H1", app.element.hydrogen, water_res)
        topology.addAtom("H2", app.element.hydrogen, water_res)
        x = 2.0 + 0.3 * (i + 1)
        positions_nm.append([x, 2.0, 2.0])
        positions_nm.append([x + 0.05, 2.05, 2.0])
        positions_nm.append([x + 0.05, 1.95, 2.0])

    positions = np.asarray(positions_nm, dtype=float) * unit.nanometer
    modeller = app.Modeller(topology, positions)
    return modeller


def test_insert_single_dummy_replaces_farthest_water():
    modeller = _build_synthetic_modeller(n_waters=6)
    n_atoms_before = sum(1 for _ in modeller.topology.atoms())

    farthest_water_o_pos_nm = np.asarray(
        modeller.positions.value_in_unit(unit.nanometer)[1 + 3 * 5]
    )  # 第 6 个水（index 5）的氧原子，摆在最远处

    indices = runabfe._insert_reserved_coalchemical_ion_dummies(
        modeller, count=1, cation=True, ligand_atom_indices=[0]
    )

    assert len(indices) == 1
    n_atoms_after = sum(1 for _ in modeller.topology.atoms())
    # 摘掉 1 个水（3 原子）+ 加入 1 个 dummy（1 原子）：净减 2。
    assert n_atoms_after == n_atoms_before - 3 + 1

    atoms = list(modeller.topology.atoms())
    dummy_atom = atoms[indices[0]]
    assert dummy_atom.residue.name == "NA"
    assert dummy_atom.element == app.element.sodium

    dummy_pos_nm = np.asarray(
        modeller.positions.value_in_unit(unit.nanometer)[indices[0]]
    )
    np.testing.assert_allclose(dummy_pos_nm, farthest_water_o_pos_nm, atol=1e-9)

    # 只剩 5 个水（6 - 1 个被摘掉的）。
    remaining_water_residues = [
        res for res in modeller.topology.residues() if res.name == "HOH"
    ]
    assert len(remaining_water_residues) == 5


def test_insert_anion_dummy_uses_chlorine_template():
    modeller = _build_synthetic_modeller(n_waters=4)
    indices = runabfe._insert_reserved_coalchemical_ion_dummies(
        modeller, count=1, cation=False, ligand_atom_indices=[0]
    )
    atoms = list(modeller.topology.atoms())
    dummy_atom = atoms[indices[0]]
    assert dummy_atom.residue.name == "CL"
    assert dummy_atom.element == app.element.chlorine


def test_insert_multiple_dummies_picks_farthest_n_without_index_shift_bug():
    """|q_L| > 1 时必须一次性摘完全部选中的水再一次性加入 dummy——交替
    delete/add 会让先加入的 dummy 被后面的 delete 悄悄移位（见函数 docstring）。
    """
    modeller = _build_synthetic_modeller(n_waters=6)
    pos_before = np.asarray(modeller.positions.value_in_unit(unit.nanometer))
    # 最远的两个水是 index 5 和 index 4（按构造规律，氧原子分别在
    # 1 + 3*5 和 1 + 3*4）。
    expected_positions = {
        tuple(np.round(pos_before[1 + 3 * 5], 6)),
        tuple(np.round(pos_before[1 + 3 * 4], 6)),
    }

    indices = runabfe._insert_reserved_coalchemical_ion_dummies(
        modeller, count=2, cation=True, ligand_atom_indices=[0]
    )
    assert len(indices) == 2
    assert len(set(indices)) == 2  # 没有重复/移位导致的索引冲突

    atoms = list(modeller.topology.atoms())
    for idx in indices:
        assert atoms[idx].residue.name == "NA"

    got_positions = {
        tuple(
            np.round(
                np.asarray(modeller.positions.value_in_unit(unit.nanometer)[idx]), 6
            )
        )
        for idx in indices
    }
    assert got_positions == expected_positions

    remaining_water_residues = [
        res for res in modeller.topology.residues() if res.name == "HOH"
    ]
    assert len(remaining_water_residues) == 4


def test_insert_zero_count_is_noop():
    modeller = _build_synthetic_modeller(n_waters=3)
    n_before = sum(1 for _ in modeller.topology.atoms())
    indices = runabfe._insert_reserved_coalchemical_ion_dummies(
        modeller, count=0, cation=True, ligand_atom_indices=[0]
    )
    assert indices == []
    assert sum(1 for _ in modeller.topology.atoms()) == n_before


def test_insufficient_water_fails_closed():
    modeller = _build_synthetic_modeller(n_waters=2)
    with pytest.raises(RuntimeError, match="不够替换"):
        runabfe._insert_reserved_coalchemical_ion_dummies(
            modeller, count=5, cation=True, ligand_atom_indices=[0]
        )
