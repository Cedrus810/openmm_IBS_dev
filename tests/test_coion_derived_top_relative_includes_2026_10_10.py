"""派生带 reserved co-ion 的 .top 时丢了同目录 #include（2026-10-10）。

真机症状（fragment-opt `MCL-1_ligand-1/rep1`，配体 −1 e，启动 2 秒退出）：

    ERROR 拓扑解析失败: Could not locate #include file: ligand_atomtypes.hmr.itp

原始 .top 用相对 include 引同目录的 itp；派生件写进 `coion_reserved/` 后
include 行原样照抄，而 OpenMM 先搜 .top 所在目录 ⟹ 找不到。修法：把原 .top
目录里的每一项软链进派生目录。

第二条测试钉住 fragment-opt 的真实布局：`topologies_neutral/.../complex.hmr.top`
是指向另一目录的软链，而同目录的 `ligand.hmr.itp` 是**真文件**（与软链目标
目录里的同名文件不同）。OpenMM 按 .top 路径本身的目录解析，所以派生件必须
继续命中这份真文件，而不是 realpath 那边的。
"""
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

import runabfe as R  # noqa: E402
from test_complex_leg_reserved_coion_2026_09_16 import (  # noqa: E402
    MINI_TOP,
    _write_mini_system,
)

_LIG_HEAD = "[ moleculetype ]\nLIG 3\n"
_LIG_BLOCK = MINI_TOP[MINI_TOP.index(_LIG_HEAD):
                      MINI_TOP.index("[ moleculetype ]\nHOH")]
_SPLIT_TOP = MINI_TOP.replace(_LIG_BLOCK, '#include "lig.itp"\n\n')


def _lig_itp(q1, q2):
    assert "0.5000  12.011" in _LIG_BLOCK and "0.5000   1.008" in _LIG_BLOCK
    return (_LIG_BLOCK
            .replace("0.5000  12.011", "%.4f  12.011" % q1)
            .replace("0.5000   1.008", "%.4f   1.008" % q2))


def _derive(gro, top, out):
    system, topology, positions, box_v, lig = R.build_system_from_gromacs(
        gro, top, "LIG", None)
    return R._ensure_complex_reserved_coions(
        system=system, topology=topology, positions=positions,
        box_vectors=box_v, ligand_indices=lig, gro_file=gro, top_file=top,
        ligand_resname="LIG", gmx_include_dir=None, output_dir=str(out),
        ligand_net_charge_e=1)


def _lig_charges(system, lig):
    nb = next(f for f in system.getForces()
              if isinstance(f, openmm.NonbondedForce))
    return [nb.getParticleParameters(int(i))[0].value_in_unit(
        unit.elementary_charge) for i in lig]


def test_same_dir_include_survives_derivation(tmp_path):
    gro, top = _write_mini_system(tmp_path)
    (tmp_path / "mini.top").write_text(_SPLIT_TOP)
    (tmp_path / "lig.itp").write_text(_lig_itp(0.5, 0.5))

    system, _t, _p, _b, lig, report = _derive(gro, top, tmp_path / "run")

    assert report["count"] == 1
    assert "lig.itp" in report["linked_source_dir_entries"]
    assert _lig_charges(system, lig) == pytest.approx([0.5, 0.5])
    # 运行时拓扑（Boresch 等下游读的那份）也要能独立重新解析
    rt_top = R.runtime_gromacs_top(str(tmp_path / "run"), top)
    assert rt_top == report["derived_top"]
    R.build_system_from_gromacs(report["derived_gro"], rt_top, "LIG", None)


def test_symlinked_top_resolves_includes_from_its_own_dir(tmp_path):
    real = tmp_path / "topologies_and_structure"
    view = tmp_path / "topologies_neutral"
    real.mkdir()
    view.mkdir()
    gro, _ = _write_mini_system(real)
    (real / "mini.top").write_text(_SPLIT_TOP)
    (real / "lig.itp").write_text(_lig_itp(0.5, 0.5))
    os.symlink(real / "mini.top", view / "mini.top")
    os.symlink(real / "mini.gro", view / "mini.gro")
    (view / "lig.itp").write_text(_lig_itp(0.6, 0.4))   # 真文件，必须用这份

    sys0, _t, _p, _b, lig0 = R.build_system_from_gromacs(
        str(view / "mini.gro"), str(view / "mini.top"), "LIG", None)
    assert _lig_charges(sys0, lig0) == pytest.approx([0.6, 0.4])

    system, _t, _p, _b, lig, _rep = _derive(
        str(view / "mini.gro"), str(view / "mini.top"), tmp_path / "run")
    assert _lig_charges(system, lig) == pytest.approx([0.6, 0.4])
