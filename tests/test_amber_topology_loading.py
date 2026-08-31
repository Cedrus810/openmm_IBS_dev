"""AMBER 最小加载层：`load_amber_topology_for_openmm()`（memtodolist：GMX 路径可配置化 + AMBER 可行性探索）。

## 为什么要有这一层

仓库生产主链（`runabfe.py`/`abfe_pipeline.py`/`ibs_engine.py`）迄今为止只解析
GROMACS 文本格式（`.top`/`.gro`），没有任何代码路径读取 `.prmtop`/`.inpcrd`
（`tools/validation/validate_charge_transfer_lipid_slab.py` 里有过明确记录）。

这是"能不能把 AMBER 格式塞进来"的可行性验证：只要求 OpenMM 原生的
`AmberPrmtopFile`/`AmberInpcrdFile` 能把一个真实 prmtop/inpcrd 加载成可用的
`Topology`/坐标——不要求（也没有）GROMACS 那条路径上的 composition 分类、
funct-2 校验、co-ion 插入这些 GROMACS 专属能力。

## 用真实文件，不是合成拓扑

`charmm-gui-8600905442/openmm/step5_input.parm7` + `.rst7` 是仓库里已经存在的、
CHARMM-GUI 生成的真实 AMBER 格式膜体系文件（21525 原子），不是这次新加的测试
夹具。用真实文件而不是手搓一个最小 prmtop，是因为 AMBER 段结构（FLAG/FORMAT
分段、POINTERS 数组等）复杂到手搓极易"测试自己骗自己"。
"""

from pathlib import Path

import pytest

pytest.importorskip("openmm")

import abfe_core as core

ROOT = Path(__file__).absolute().parents[1]
AMBER_PRMTOP = ROOT / "tests/fixtures/charmm-gui-8600905442" / "openmm" / "step5_input.parm7"
AMBER_INPCRD = ROOT / "tests/fixtures/charmm-gui-8600905442" / "openmm" / "step5_input.rst7"

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        not (AMBER_PRMTOP.exists() and AMBER_INPCRD.exists()),
        reason="vendored AMBER fixture (charmm-gui-8600905442/openmm/step5_input.{parm7,rst7}) 不存在",
    ),
]


def test_load_prmtop_only_returns_topology_and_no_inpcrd():
    prmtop, inpcrd = core.load_amber_topology_for_openmm(str(AMBER_PRMTOP))

    assert inpcrd is None
    n_atoms = prmtop.topology.getNumAtoms()
    assert n_atoms == 21525


def test_load_prmtop_and_inpcrd_round_trip():
    prmtop, inpcrd = core.load_amber_topology_for_openmm(
        str(AMBER_PRMTOP), str(AMBER_INPCRD)
    )

    n_atoms = prmtop.topology.getNumAtoms()
    assert n_atoms == 21525
    # inpcrd 的坐标数必须和 prmtop 的原子数一致，否则后面 createSystem/设置坐标会错位。
    assert len(inpcrd.positions) == n_atoms
    assert inpcrd.boxVectors is not None


def test_missing_prmtop_fails_closed():
    with pytest.raises(FileNotFoundError):
        core.load_amber_topology_for_openmm(str(ROOT / "does_not_exist.parm7"))


def test_missing_inpcrd_fails_closed():
    with pytest.raises(FileNotFoundError):
        core.load_amber_topology_for_openmm(
            str(AMBER_PRMTOP), str(ROOT / "does_not_exist.rst7")
        )
