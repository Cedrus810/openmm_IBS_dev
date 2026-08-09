"""C2：`tools/validation/validate_charge_transfer_lipid_slab.py` 的 CPU 契约测试。

覆盖用户 code-review 里列的 11 条（见对话记录 / `memtodolist.md` C2 一节）：

  1. Na build 后 TP3 -3 / Na+ +2 / Cl- +1（`_edit_top_molecules_block` 纯文本层）。
  2. dummy 清零后基础总电荷严格为 0。
  3. 11 个 λ 总电荷都为 0。
  4. ordinary Cl 不会被识别为 dummy。
  5. frozen spec 只识别一个 dummy。
  6. restraint force 只存在一份（钉住 v1 的重复注入 bug）。
  7. `ukn` 用原始 System 唯一配置 offsets（钉住 v1 的重复配置 bug）。
  8. thick slab（`_edit_top_molecules_block` 的水专用编辑）只改水计数，不碰其它行。
  9. 缺 timeseries 或质量门时 report 必须失败（status=incomplete）。
  10. 人工制造 co-ion 换侧 / 距离越门 / NaN 时 gate 必须失败。
  11. compare 必须同时满足 `≤2σ` 和 `≤1 kcal/mol`。

第 1/8/9/11 组不需要 OpenMM（纯文本/JSON 逻辑），任何环境都能跑。
第 2/3/4/5/6/7/10 组需要 `openmm`（+ 10 组另需 `mdtraj`），用
`pytest.importorskip` 优雅跳过——本仓库既有测试（如
`tests/test_charge_transfer_hamiltonian.py`）就是这个约定，不是本文件新发明的。

v3→v4（2026-08-09）新增三组，钉住之前只手工验证过的三处修复（不需要真实
GROMACS 拓扑/GPU，都是从 `cmd_*` 里拆出来的纯函数/小型合成体系）：

  12. `_pack_water_slab`：目标格点数按**完整**新增体积算，不是按扣掉 buffer
      的 usable_volume 算（2026-08-09 review 抓到的第二轮 bug）；buffer 只
      影响格点摆放位置，不影响摆多少个。
  13. `_z_number_density_profile`：数密度真的除以了 bin 体积，不是只除帧数。
  14. `_run_equilibration_segment` + `--n-steps-nvt`：NVT→NPT 两段的
      `step`/`phase` 列正确、`system.addForce(barostat)` +
      `Context.reinitialize(preserveState=True)` 这个技巧本身能跑通；
      `cmd_equilibrate_base` 对 `--n-steps-nvt` 的参数校验（负数/
      `>= --n-steps`）能挡住。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

_MODULE_PATH = ROOT / "tools" / "validation" / "validate_charge_transfer_lipid_slab.py"
_spec = importlib.util.spec_from_file_location("validate_charge_transfer_lipid_slab", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)  # type: ignore[union-attr]

import abfe_core as core


# ============================================================================
# 第 1/8 组：`_edit_top_molecules_block`（纯文本，不需要 OpenMM）
# ============================================================================

_MINIMAL_TOP = """\
; minimal synthetic topol.top for _edit_top_molecules_block unit tests
#include "toppar/forcefield.itp"
#include "toppar/POPC.itp"
#include "toppar/K+.itp"
#include "toppar/Na+.itp"
#include "toppar/Cl-.itp"
#include "toppar/TP3.itp"

[ system ]
; Name
Title

[ molecules ]
; Compound\t#mols
POPC  \t          80
K+    \t           8
Na+   \t           8
Cl-   \t          16
TP3   \t        3591
"""


def _parse_molecules_block(top_text: str) -> list:
    lines = top_text.splitlines()
    out = []
    in_molecules = False
    for line in lines:
        stripped = line.split(";", 1)[0].strip()
        lowered = stripped.lower()
        if lowered in ("[ molecules ]", "[molecules]"):
            in_molecules = True
            continue
        if in_molecules and stripped.startswith("[") and lowered not in ("[ molecules ]", "[molecules]"):
            in_molecules = False
        if in_molecules and stripped:
            parts = stripped.split()
            out.append((parts[0], int(parts[1])))
    return out


def test_edit_top_molecules_block_na_probe_counts(tmp_path):
    """用例 1：Na 探针 build 后 TP3 -3 / Na+ +2（probe+dummy） / Cl- +1（普通反离子）。"""
    top_path = tmp_path / "topol.top"
    top_path.write_text(_MINIMAL_TOP, encoding="utf-8")
    out_path = tmp_path / "topol_edited.top"

    mod._edit_top_molecules_block(
        str(top_path), str(out_path), water_moleculetype="TP3", water_delta=-3,
        appended_blocks=[("Cl-", 1), ("Na+", 1), ("Na+", 1)],
    )

    before = dict(_parse_molecules_block(_MINIMAL_TOP))
    after_blocks = _parse_molecules_block(out_path.read_text(encoding="utf-8"))

    after_totals: dict = {}
    for name, count in after_blocks:
        after_totals[name] = after_totals.get(name, 0) + count

    assert after_totals["TP3"] == before["TP3"] - 3
    assert after_totals["Na+"] == before["Na+"] + 2
    assert after_totals["Cl-"] == before["Cl-"] + 1
    assert after_totals["POPC"] == before["POPC"]
    assert after_totals["K+"] == before["K+"]
    # 顺序即插入顺序：普通反离子块在前，探针配体块，dummy 块在最后。
    tail = after_blocks[-3:]
    assert tail == [("Cl-", 1), ("Na+", 1), ("Na+", 1)]


def test_edit_top_molecules_block_thick_variant_only_touches_water(tmp_path):
    """用例 8：`extend-water` 用的水专用编辑（无 appended_blocks）只改 TP3 一行，
    其余每一行（含 POPC/K+/Na+/Cl-）必须逐字节不变。"""
    top_path = tmp_path / "topol.top"
    top_path.write_text(_MINIMAL_TOP, encoding="utf-8")
    out_path = tmp_path / "topol_thick.top"

    mod._edit_top_molecules_block(
        str(top_path), str(out_path), water_moleculetype="TP3", water_delta=+120,
    )

    before_lines = _MINIMAL_TOP.splitlines()
    after_lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(before_lines) == len(after_lines), "水专用编辑不应增删任何行"
    n_diff = sum(1 for a, b in zip(before_lines, after_lines) if a != b)
    assert n_diff == 1, f"应恰好只有 TP3 那一行变化，实际改了 {n_diff} 行"

    before = dict(_parse_molecules_block(_MINIMAL_TOP))
    after = dict(_parse_molecules_block(out_path.read_text(encoding="utf-8")))
    assert after["TP3"] == before["TP3"] + 120
    assert after["POPC"] == before["POPC"]
    assert after["K+"] == before["K+"]
    assert after["Na+"] == before["Na+"]
    assert after["Cl-"] == before["Cl-"]


def test_edit_top_molecules_block_negative_count_rejected(tmp_path):
    top_path = tmp_path / "topol.top"
    top_path.write_text(_MINIMAL_TOP, encoding="utf-8")
    with pytest.raises(ValueError):
        mod._edit_top_molecules_block(
            str(top_path), str(tmp_path / "out.top"),
            water_moleculetype="TP3", water_delta=-999999,
        )


# ============================================================================
# 第 12 组：`_pack_water_slab` 完整体积密度（v4 二次修法，纯 numpy，不需要 OpenMM）
# ============================================================================


def test_pack_water_slab_achieves_full_volume_density_not_just_usable_subvolume():
    """钉住 2026-08-09 review 抓到的 bug：第一版 v4 修复按 buffer 扣除后的
    `usable_volume` 算目标格点数，摊到**完整**新增体积上密度仍然只有声称
    目标的六成左右（真实 C2 盒子实测：1024 个水 / 约 49 nm³ ≈ 20.9 nm⁻³，
    偏离 33.33 nm⁻³ 约 37%）。用同一量级的真实盒子边长验证修复后的偏差落在
    个位数百分比。
    """
    lx = ly = 4.9942007  # base_thin_v3_extend1 实测平衡后的盒子边长（v3 GPU 产物）
    z_min, z_max = 0.0, 1.0  # 单侧新增水层厚度 1.0 nm（--extra-water-nm 2.0 的一半）
    rng = np.random.default_rng(2026)

    molecules, diag = mod._pack_water_slab(lx, ly, z_min, z_max, rng)

    assert molecules.shape == (diag["actual_water_count"], 3, 3)
    assert diag["actual_water_count"] == diag["n_x"] * diag["n_y"] * diag["n_z"]
    assert diag["full_added_volume_nm3"] == pytest.approx(lx * ly * (z_max - z_min))
    assert diag["target_water_count_full_volume"] == pytest.approx(
        mod.BULK_WATER_NUMBER_DENSITY_PER_NM3 * diag["full_added_volume_nm3"], rel=0.02,
    )
    deviation = abs(
        diag["achieved_density_full_volume_nm3"] - mod.BULK_WATER_NUMBER_DENSITY_PER_NM3
    ) / mod.BULK_WATER_NUMBER_DENSITY_PER_NM3
    # 修复前这个偏差约 37%；修复后单侧应该落在 15% 以内——留够裕度，不是卡着
    # 当前实测值的小数点，但足以钉住"退回第一轮 v4 bug"这种回归。
    assert deviation < 0.15, f"完整体积密度偏差 {deviation:.1%} 太大，像是退回了第一轮 v4 的 bug"


def test_pack_water_slab_n_z_uses_full_depth_not_usable_depth():
    """`n_z` 必须按**完整** `depth` 算，不是扣掉 buffer 的 `usable_depth`——
    buffer 只应该影响格点摆在哪（都落在 `[z_min+buffer, z_max-buffer]` 子
    区间内），不影响摆几层。"""
    lx = ly = 5.0
    z_min, z_max = 0.0, 1.0
    rng = np.random.default_rng(1)
    molecules, diag = mod._pack_water_slab(lx, ly, z_min, z_max, rng)

    target_spacing_nm = mod.BULK_WATER_NUMBER_DENSITY_PER_NM3 ** (-1.0 / 3.0)
    expected_n_z = max(1, round((z_max - z_min) / target_spacing_nm))
    assert diag["n_z"] == expected_n_z

    oxygen_z = molecules[:, 0, 2]  # 每个水分子的 O 原子 z 坐标
    z_lo = z_min + diag["buffer_nm"]
    z_hi = z_max - diag["buffer_nm"]
    assert np.all(oxygen_z >= z_lo - 1e-9)
    assert np.all(oxygen_z <= z_hi + 1e-9)


def test_pack_water_slab_diagnostics_are_self_consistent():
    lx, ly = 4.0, 4.0
    z_min, z_max = 0.0, 1.5
    rng = np.random.default_rng(7)
    _molecules, diag = mod._pack_water_slab(lx, ly, z_min, z_max, rng)

    assert diag["usable_volume_nm3"] == pytest.approx(lx * ly * diag["usable_depth_nm"])
    assert diag["achieved_density_full_volume_nm3"] == pytest.approx(
        diag["actual_water_count"] / diag["full_added_volume_nm3"]
    )


# ============================================================================
# 第 13 组：`_z_number_density_profile` 除以 bin 体积（v3→v4 §4，纯 numpy）
# ============================================================================


def test_z_number_density_profile_divides_by_bin_volume():
    """钉住 v3→v4 §4：之前只除帧数、没除 bin 体积。构造已知：2 帧、每帧同一个
    bin 里各有 3 个原子，bin 体积 2.0 nm³——期望密度 = (3 个/帧) / 2.0 nm³
    = 1.5 nm⁻³，不是"3 个原子/帧"这个原始计数。"""
    bins = np.array([0.0, 1.0, 2.0, 3.0])
    z_values = np.array([1.2, 1.3, 1.4] * 2)  # 2 帧、每帧 3 个原子落在 [1,2) 这个 bin
    n_frames = 2
    bin_volume_nm3 = 2.0

    profile = mod._z_number_density_profile(z_values, n_frames, bins, bin_volume_nm3)

    assert profile[0] == pytest.approx(0.0)
    assert profile[1] == pytest.approx(1.5)  # (3 个/2 帧) / 2.0 nm^3
    assert profile[2] == pytest.approx(0.0)


def test_z_number_density_profile_empty_indices_returns_all_zero():
    bins = np.array([0.0, 1.0, 2.0])
    profile = mod._z_number_density_profile(np.asarray([]), 5, bins, 1.0)
    assert profile == [0.0, 0.0]


# ============================================================================
# 第 15 组：`_required_ligand_coion_min_image_nm`（v5→v6，纯数值，不需要 OpenMM）
# ============================================================================


def test_required_ligand_coion_min_image_nm_matches_runtime_formula():
    """钉住 v5→v6：`insert_ions_into_gromacs_files` 挑候选点的门槛必须跟
    `abfe_core.validate_co_alchemical_ion_placement` 真正用的 runtime 判据
    一致，不是更松的 `COION_LIGAND_MIN_IMAGE_INITIAL_NM`（1.6 nm）——真实
    thin base 上 `build` 就是拿 1.968 nm（满足 1.6 nm 松判据）的候选点在
    这道 runtime 判据上炸掉的（需要约 2.02 nm）。
    """
    expected_default = (
        core.COION_LIGAND_MIN_IMAGE_RUNTIME_NM
        + core.COION_FLAT_BOTTOM_RADIUS_NM
        + core.co_alchemical_ion_restraint_wall_margin_nm(None)
    )
    got_default = mod._required_ligand_coion_min_image_nm(None, None)
    assert got_default == pytest.approx(expected_default)
    # 默认参数下这个值必须明显比更松的 initial 判据大——如果两者相等或更小，
    # 说明常量改了但这条判据没跟着更新，候选点筛选又会退回脱节状态。
    assert got_default > core.COION_LIGAND_MIN_IMAGE_INITIAL_NM

    # 自定义 restraint 参数（`--restraint-k`/`--restraint-r0-nm`）也要真的
    # 影响结果，不是悄悄硬编码了默认值。
    got_custom = mod._required_ligand_coion_min_image_nm(50.0, 0.3)
    expected_custom = (
        core.COION_LIGAND_MIN_IMAGE_RUNTIME_NM + 0.3
        + core.co_alchemical_ion_restraint_wall_margin_nm(50.0)
    )
    assert got_custom == pytest.approx(expected_custom)
    assert got_custom != pytest.approx(got_default)


# ============================================================================
# 第 16 组：`_minimum_image_z_delta_nm` / `_find_bulk_water_candidates` 的
# z 轴周期折叠（v6→v7，纯数值，不需要 OpenMM）
# ============================================================================


def test_minimum_image_z_delta_nm_matches_real_bug_reproduction():
    """钉住 v6→v7：数值直接取自真实复现——`base_thin_v3_extend1/equilibrated.gro`
    里一个候选水的 raw z=9.623 nm，膜中面=4.175387500 nm，box_z=8.33498 nm。
    非周期性差值算出 |Δz|=5.448 nm（几何上不可能，超过 box_z/2≈4.17 nm）；
    折叠后的真实 minimum-image 差值应该是 -2.887 nm 左右（这个候选其实只是
    中等深度，没有声称的那么深）。
    """
    z, midplane, box_z = 9.623, 4.1753875, 8.33498
    naive_abs_dz = abs(z - midplane)
    assert naive_abs_dz == pytest.approx(5.4476125)  # 修复前会算出的（错误）值

    dz = mod._minimum_image_z_delta_nm(z, midplane, box_z)
    assert dz == pytest.approx(-2.8873675, abs=1e-6)
    assert abs(dz) < naive_abs_dz  # 折叠后的真实距离必须比未折叠的虚高值小
    assert abs(dz) <= box_z / 2.0 + 1e-9  # 任何点到参考点的周期最短距离不可能超过半个盒高


def test_minimum_image_z_delta_nm_no_wrap_needed_is_unchanged():
    """z 本来就在参考点附近（不需要折叠）时，结果必须和朴素差值一致——
    折叠只应该在真的跨越了周期边界时才改变结果。"""
    z, midplane, box_z = 5.0, 4.2, 8.3
    dz = mod._minimum_image_z_delta_nm(z, midplane, box_z)
    assert dz == pytest.approx(z - midplane)


def test_find_bulk_water_candidates_rejects_unwrapped_far_side_water():
    """钉住 v6→v7 的核心场景：一个 raw z 坐标"跑出盒子"的候选水，折叠前
    看起来是深度 bulk water（远超 3.0 nm 下限），折叠后其实只有约 2.89 nm
    ——低于下限，必须被拒绝，不能被"farthest-first"贪心选中。
    """
    box_z_nm = 8.33498
    midplane_z_nm = 4.1753875
    box_nm = np.diag([5.0, 5.0, box_z_nm])
    # 真实复现用的候选：raw z=9.623（unwrapped，超出 [0, box_z)），
    # 折叠后真实位置约 1.288 nm，离中面约 2.887 nm——低于 3.0 nm 下限。
    unwrapped_z = 9.623
    positions_nm = np.zeros((2, 3))
    positions_nm[0] = [2.5, 2.5, midplane_z_nm]  # 占位磷原子，随便摆在别处
    positions_nm[1] = [2.5, 2.5, unwrapped_z]  # 候选水氧
    water_oxygens = [(1, None)]  # `residue` 字段在函数内部只是原样存回去，用 None 占位即可

    candidates = mod._find_bulk_water_candidates(
        positions_nm, box_nm, midplane_z_nm, phosphorus_indices=[0], water_oxygens=water_oxygens,
    )
    assert candidates == [], (
        "这个候选折叠后离膜中面只有约 2.89 nm，低于 3.0 nm 安全下限，"
        "必须被拒绝——如果这里不是空列表，说明 z 轴周期折叠又被绕过了"
    )


def test_find_bulk_water_candidates_side_uses_wrapped_z_not_raw_z():
    """`side`（upper/lower）判断必须用折叠后的 z，不是原始 z——一个真实
    "下叶"（折叠后 z 在中面之下）的候选，如果只看未折叠的 raw z（在中面
    之上），会被错误标成 "upper"。"""
    box_z_nm = 10.0
    midplane_z_nm = 5.0
    box_nm = np.diag([5.0, 5.0, box_z_nm])
    # raw z=11.0 未折叠时在中面(5.0)之上（看起来是 "upper"，且未折叠
    # |Δz|=6.0 nm 远超 3.0 nm 下限），但折叠后 (11.0-5.0)-10.0=-4.0，
    # 真实在中面之下、真实距离 4.0 nm——应该判成 "lower"，不是 "upper"。
    raw_z = 11.0
    # 用一个真正远离候选的磷原子占位，避免被 nearest-phosphorus 下限拦掉。
    positions_nm = np.zeros((2, 3))
    positions_nm[0] = [0.0, 0.0, 0.0]
    positions_nm[1] = [2.5, 2.5, raw_z]
    water_oxygens = [(1, None)]

    candidates = mod._find_bulk_water_candidates(
        positions_nm, box_nm, midplane_z_nm, phosphorus_indices=[0], water_oxygens=water_oxygens,
    )
    assert len(candidates) == 1
    assert candidates[0]["side"] == "lower"


openmm = pytest.importorskip("openmm")
from openmm import NonbondedForce, Vec3, app, unit  # noqa: E402


# ============================================================================
# 第 14 组：`_run_equilibration_segment` + `--n-steps-nvt` 分阶段松弛
# （v3→v4 §6，合成 20 粒子 argon-like 体系，CPU platform，不需要真实拓扑）
# ============================================================================


def test_run_equilibration_segment_phase_column_and_step_continuity(tmp_path):
    """钉住新的 NVT→NPT 分段机制：
      1. 两段调用的 `step`/`time_ps` 连续累加，不在切阶段时从 0 重开；
      2. CSV 的 `phase` 列正确区分 `nvt`/`npt`；
      3. `system.addForce(barostat)` + `Context.reinitialize(preserveState=True)`
         这个技巧本身不炸——加完 barostat 后续步照常写数据。
    """
    n_particles = 20
    box_nm = 3.0
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(Vec3(box_nm, 0, 0), Vec3(0, box_nm, 0), Vec3(0, 0, box_nm))
    nb = NonbondedForce()
    nb.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.0 * unit.nanometer)

    topology = app.Topology()
    chain = topology.addChain()
    rng = np.random.default_rng(0)
    positions = rng.uniform(0.3, box_nm - 0.3, size=(n_particles, 3))
    for _ in range(n_particles):
        res = topology.addResidue("AR", chain)
        topology.addAtom("AR", app.element.argon, res)
        system.addParticle(39.95 * unit.amu)
        nb.addParticle(0.0 * unit.elementary_charge, 0.34 * unit.nanometer, 0.5 * unit.kilojoule_per_mole)
    system.addForce(nb)
    topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())

    integrator = openmm.LangevinMiddleIntegrator(
        120 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond
    )
    integrator.setRandomNumberSeed(3)
    platform = openmm.Platform.getPlatformByName("CPU")
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions * unit.nanometer)
    simulation.minimizeEnergy(maxIterations=200)
    simulation.context.setVelocitiesToTemperature(120 * unit.kelvin)

    dcd_path = tmp_path / "seg.dcd"
    csv_path = tmp_path / "seg.csv"
    n_degrees_of_freedom = 3 * n_particles - system.getNumConstraints()

    with open(dcd_path, "wb") as dcd_fh, open(csv_path, "w") as csv_fh:
        dcd = app.DCDFile(dcd_fh, topology, dt=0.002 * unit.picosecond, interval=10)
        csv_fh.write("step,time_ps,potential_kJ_mol,volume_nm3,temperature_K,phase\n")

        step_after_nvt = mod._run_equilibration_segment(
            simulation, dcd, csv_fh, 20, 10, 0.002, n_degrees_of_freedom, 0, "nvt",
        )
        assert step_after_nvt == 20

        # 与 cmd_equilibrate_base 完全相同的技巧：先往 System 里加 barostat，
        # 再 reinitialize(preserveState=True)——不重建 Context、位置/速度/
        # 盒矢量原样保留，NVT 段跑出来的状态直接接着往下跑 NPT。
        barostat = openmm.MonteCarloBarostat(1.0 * unit.bar, 120 * unit.kelvin, 5)
        system.addForce(barostat)
        simulation.context.reinitialize(preserveState=True)

        step_after_npt = mod._run_equilibration_segment(
            simulation, dcd, csv_fh, 20, 10, 0.002, n_degrees_of_freedom, step_after_nvt, "npt",
        )
        assert step_after_npt == 40

    rows = list(csv.reader(open(csv_path)))
    header, data = rows[0], rows[1:]
    phase_idx, step_idx = header.index("phase"), header.index("step")
    phases = [r[phase_idx] for r in data]
    steps = [int(r[step_idx]) for r in data]
    assert phases == ["nvt", "nvt", "npt", "npt"]
    assert steps == [10, 20, 30, 40]  # NPT 段接着 NVT 段的累计步数继续，不从 0 重开


def test_run_equilibration_segment_rejects_non_divisible_step_count():
    # 校验发生在函数最开头（触碰 simulation/dcd/csv_fh 之前），传 None 也安全。
    with pytest.raises(SystemExit):
        mod._run_equilibration_segment(
            None, None, None, 15, 10, 0.002, 10, 0, "nvt",
        )


def test_add_barostat_and_activate_actually_changes_volume():
    """钉住 v4→v5 的严重 bug：`cmd_equilibrate_base` 曾经有一条路径（`--n-steps-nvt=0`
    分支，也就是默认值！）只对 `system.addForce(barostat)`，没调用
    `Context.reinitialize`——`simulation.context` 早就用不带 barostat 的
    `system` 建好了，Python 端往 `system` 里加 Force 不会让已经建好的 Context
    知道，新加的 barostat 完全是摆设。真实 GPU 实测复现过：某次续跑用
    `--n-steps-nvt 0` 跑了 8 ns「NPT」，box_z/APL 从头到尾逐帧 bit-for-bit
    原样不变——一步体积试探移动都没真的发生过。

    `_add_barostat_and_activate` 现在把"加 barostat"和"reinitialize"绑死在
    一个函数里，`cmd_equilibrate_base` 的两个分支都通过它加 barostat。这里
    直接验证：调用完之后继续跑几十步，高压 barostat 必须让 box 体积产生
    可测量的变化——不能像 bug 复现的那样保持 bit-for-bit 不变。
    """
    n_particles = 40
    box_nm = 3.0
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(Vec3(box_nm, 0, 0), Vec3(0, box_nm, 0), Vec3(0, 0, box_nm))
    nb = NonbondedForce()
    nb.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.0 * unit.nanometer)

    topology = app.Topology()
    chain = topology.addChain()
    rng = np.random.default_rng(2)
    positions = rng.uniform(0.3, box_nm - 0.3, size=(n_particles, 3))
    for _ in range(n_particles):
        res = topology.addResidue("AR", chain)
        topology.addAtom("AR", app.element.argon, res)
        system.addParticle(39.95 * unit.amu)
        nb.addParticle(0.0 * unit.elementary_charge, 0.34 * unit.nanometer, 0.5 * unit.kilojoule_per_mole)
    system.addForce(nb)
    topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())

    integrator = openmm.LangevinMiddleIntegrator(
        120 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond
    )
    integrator.setRandomNumberSeed(5)
    platform = openmm.Platform.getPlatformByName("CPU")
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions * unit.nanometer)
    simulation.minimizeEnergy(maxIterations=200)
    simulation.context.setVelocitiesToTemperature(120 * unit.kelvin)

    box_before = simulation.context.getState().getPeriodicBoxVectors()
    vol_before = mod._box_volume_nm3(
        np.asarray([v.value_in_unit(unit.nanometer) for v in box_before])
    )

    # 高压（300 bar）+ 每步都试探（frequency=1），几十步内必须能看到体积变化，
    # 不需要跑到真正平衡就足以钉住"完全没生效"这种退化情形。
    membrane_protocol = {"barostat_class": "MonteCarloBarostat", "barostat_frequency": 1}
    mod._add_barostat_and_activate(system, membrane_protocol, 120.0, 300.0, simulation)

    for _ in range(60):
        simulation.step(1)

    box_after = simulation.context.getState().getPeriodicBoxVectors()
    vol_after = mod._box_volume_nm3(
        np.asarray([v.value_in_unit(unit.nanometer) for v in box_after])
    )
    assert vol_after != pytest.approx(vol_before, rel=1e-9), (
        "加 barostat 后体积 60 步内 bit-for-bit 没变化——像是 reinitialize 又被漏掉了"
    )


def _minimal_equilibrate_base_args(tmp_path, n_steps: int, n_steps_nvt: int) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=str(tmp_path / "out"),
        top="unused.top", gro="unused.gro", gmx_include_dir=None,
        water_thickness_label="thick",
        n_steps=n_steps, n_steps_nvt=n_steps_nvt, n_steps_minimize=10,
        report_interval_steps=10, timestep_ps=0.002, temperature_kelvin=303.15,
        friction_per_ps=1.0, seed=1, platform="CPU", precision="mixed",
        allow_cpu_fallback=False,
    )


def test_equilibrate_base_rejects_negative_n_steps_nvt(tmp_path):
    """校验发生在读 top/gro 文件之前（只是变量赋值），用不存在的路径也安全。"""
    args = _minimal_equilibrate_base_args(tmp_path, n_steps=100, n_steps_nvt=-1)
    with pytest.raises(SystemExit, match="不能为负数"):
        mod.cmd_equilibrate_base(args)


def test_equilibrate_base_rejects_n_steps_nvt_not_less_than_n_steps(tmp_path):
    args = _minimal_equilibrate_base_args(tmp_path, n_steps=100, n_steps_nvt=100)
    with pytest.raises(SystemExit, match="必须小于"):
        mod.cmd_equilibrate_base(args)


# ============================================================================
# 第 2-7 组：OpenMM 合成 charge-transfer 系统（复用
# tests/test_charge_transfer_hamiltonian.py 已验证过的搭建思路：
# ligand(+1) + reserved 中性 dummy + 配平 Cl- + 水，Reference/CPU platform）
# ============================================================================

BOX_NM = np.diag([6.0, 6.0, 12.0])
LIGAND_INDICES = [0]
DUMMY_INDEX = 1
ORDINARY_CL_INDEX = 2


def _build_synthetic_charge_transfer_system(*, n_dummies: int = 1, dummy_charge_e: float = 0.0):
    """单原子 Na+ 配体(+1) + reserved 中性 Na 形 dummy + 配平 Cl- + 2 个水。

    与 C2 script 的 v2 三粒子插入顺序一致（普通反离子、探针、dummy），只是这里
    直接手搭 System（不经过 GROMACS .top/.gro 文本层），专门用来测
    Hamiltonian/restraint/report 逻辑，不测 GROMACS 文本编辑（那部分见上面
    第 1/8 组，以及 `insert_ions_into_gromacs_files` 走真实 slab 的验收由用户
    在 GPU 节点上用 static-check 跑，见执行清单）。
    """
    topology = app.Topology()
    chain = topology.addChain()

    ligand_res = topology.addResidue("Na+", chain)
    topology.addAtom("Na+", app.element.sodium, ligand_res)

    dummy_indices = []
    for _ in range(int(n_dummies)):
        dummy_res = topology.addResidue("Na+", chain)
        topology.addAtom("Na+", app.element.sodium, dummy_res)
        dummy_indices.append(topology.getNumAtoms() - 1)

    ordinary_res = topology.addResidue("Cl-", chain)
    topology.addAtom("Cl-", app.element.chlorine, ordinary_res)

    for _ in range(2):
        water_res = topology.addResidue("TP3", chain)
        topology.addAtom("O", app.element.oxygen, water_res)
        topology.addAtom("H1", app.element.hydrogen, water_res)
        topology.addAtom("H2", app.element.hydrogen, water_res)

    charges = [1.0] + [float(dummy_charge_e)] * int(n_dummies) + [-1.0]
    charges += [-0.834, 0.417, 0.417, -0.834, 0.417, 0.417]
    sigmas = [0.2439] * (2 + int(n_dummies)) + [0.4478] + [0.3151, 0.1, 0.1, 0.3151, 0.1, 0.1]
    epsilons = [0.3658] * (1 + int(n_dummies)) + [0.1489] + [0.6364, 0.0, 0.0, 0.6364, 0.0, 0.0]
    masses = [22.99] * (1 + int(n_dummies)) + [35.45, 15.999, 1.008, 1.008, 15.999, 1.008, 1.008]

    force = NonbondedForce()
    force.setNonbondedMethod(NonbondedForce.PME)
    force.setCutoffDistance(1.0 * unit.nanometer)
    for q, sigma, epsilon in zip(charges, sigmas, epsilons):
        force.addParticle(q * unit.elementary_charge, sigma * unit.nanometer, epsilon * unit.kilojoule_per_mole)

    positions = [[3.0, 3.0, 2.0]]
    for i in range(int(n_dummies)):
        positions.append([3.0, 3.0, 8.0 + 0.6 * i])
    positions.append([1.0, 1.0, 5.0])
    positions += [
        [4.5, 3.0, 6.0], [4.6, 3.0, 6.0], [4.4, 3.0, 6.0],
        [2.0, 4.5, 9.0], [2.1, 4.5, 9.0], [1.9, 4.5, 9.0],
    ]

    system = openmm.System()
    for mass in masses:
        system.addParticle(mass * unit.dalton)
    system.setDefaultPeriodicBoxVectors(*(BOX_NM * unit.nanometer))
    system.addForce(force)

    total = sum(charges)
    if float(dummy_charge_e) == 0.0:
        assert abs(total) < 1e-9, f"fixture 自身不电中性：Σq = {total:+.6f} e"
    return system, topology, np.asarray(positions, dtype=float) * unit.nanometer, BOX_NM


def _freeze_spec(system, topology, positions, box):
    import ibs_engine as ie

    return ie.select_co_alchemical_ion_once(
        system, LIGAND_INDICES, topology, positions, box,
        charge_treatment=core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
    )


def test_dummy_zeroed_total_charge_exactly_zero():
    """用例 2：dummy 清零后（fixture 构造本身就要求）基础总电荷严格为 0。"""
    system, topology, positions, box = _build_synthetic_charge_transfer_system()
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    total = sum(
        nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        for i in range(nb.getNumParticles())
    )
    assert abs(total) <= core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E


def test_all_eleven_lambda_total_charge_zero():
    """用例 3：11 个 λ 点，全盒总电荷都严格为 0（不是只查两端）。"""
    import ibs_engine as ie

    system, topology, positions, box = _build_synthetic_charge_transfer_system()
    spec = _freeze_spec(system, topology, positions, box)
    ie.configure_pme_ligand_charge_offsets(
        system, LIGAND_INDICES, lambda_name="lambda_coul", allow_charged_ligand=True,
        topology=topology, positions=positions, box_vectors=box, co_alchemical_ion_spec=spec,
    )
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    lambdas = [round(x, 2) for x in np.arange(1.0, -0.001, -0.1)]
    assert len(lambdas) == 11
    report = ie.charging_charge_conservation_report(
        nb, "lambda_coul", ligand_indices=LIGAND_INDICES,
        co_ion_indices=[int(i["atom_index"]) for i in spec["ions"]],
        ligand_net_charge_e=1, lambdas=lambdas,
    )
    assert report["total_charge_is_lambda_independent"]
    assert abs(report["base_sum_e"]) <= core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E
    for lam_key, total in report["total_charge_by_lambda_e"].items():
        assert abs(total) <= core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E, f"λ={lam_key}: 总电荷={total}"


def test_ordinary_counterion_not_identified_as_dummy():
    """用例 4：普通（带电）反离子 Cl- 不会被 `_identify_reserved_neutral_co_ions`
    错认成 reserved dummy——判据是残基名在离子集合里 **且电荷严格为 0**，
    普通反离子电荷是 -1，不满足第二条。"""
    system, topology, positions, box = _build_synthetic_charge_transfer_system()
    spec = _freeze_spec(system, topology, positions, box)
    ion_indices = [int(i["atom_index"]) for i in spec["ions"]]
    assert ion_indices == [DUMMY_INDEX]
    assert ORDINARY_CL_INDEX not in ion_indices


def test_frozen_spec_identifies_exactly_one_dummy():
    """用例 5：frozen spec 只识别一个 dummy；多于一个零电荷 ion-shaped 粒子时
    `_identify_reserved_neutral_co_ions` 必须 fail closed（数量不等于 |q_L|）。"""
    system, topology, positions, box = _build_synthetic_charge_transfer_system()
    spec = _freeze_spec(system, topology, positions, box)
    assert len(spec["ions"]) == 1

    # 故意多留一个零电荷 Na+ 形粒子（模拟"建系时留了不止一个 dummy"的错误输入）。
    system2, topology2, positions2, box2 = _build_synthetic_charge_transfer_system(n_dummies=2)
    import ibs_engine as ie
    with pytest.raises(RuntimeError, match="预留"):
        ie.select_co_alchemical_ion_once(
            system2, LIGAND_INDICES, topology2, positions2, box2,
            charge_treatment=core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        )


def test_restraint_force_only_injected_once():
    """用例 6：钉住 v1 的重复注入 bug——只调用一次
    `configure_pme_ligand_charge_offsets` 时，restraint force（`CustomCompoundBondForce`，
    force group = `CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP`）必须恰好一份；
    v1 那种"外面先手动调 `_inject_co_alchemical_ion_restraints`，函数内部又调一次"
    的写法必须被 `mod._assert_single_restraint_force` 拦下来。
    """
    import ibs_engine as ie

    system, topology, positions, box = _build_synthetic_charge_transfer_system()
    spec = _freeze_spec(system, topology, positions, box)
    mod._assert_single_restraint_force(system, n_expected=0)

    ie.configure_pme_ligand_charge_offsets(
        system, LIGAND_INDICES, lambda_name="lambda_coul", allow_charged_ligand=True,
        topology=topology, positions=positions, box_vectors=box, co_alchemical_ion_spec=spec,
    )
    mod._assert_single_restraint_force(system, n_expected=1)  # 不 raise = 通过

    # 复现 v1 的 bug：再手动注入一次，现在应该有 2 份，断言必须失败。
    ie._inject_co_alchemical_ion_restraints(system, spec)
    with pytest.raises(SystemExit):
        mod._assert_single_restraint_force(system, n_expected=1)


def test_ukn_configures_hamiltonian_exactly_once_from_raw_system(tmp_path):
    """用例 7：`ukn` 必须从**原始**（未配置过 offset/restraint 的）System 出发，
    `compute_u_kn` 内部唯一配置一次。这里直接验证 v2 的关键不变量：
    把一个**已经配置过**的 system 传给 `compute_u_kn` 风格的二次配置会被
    `_assert_single_restraint_force` 拦住（对应脚本里 `cmd_ukn` 开头的
    `_assert_single_restraint_force(raw_system, n_expected=0)` 断言）。
    """
    import ibs_engine as ie

    system, topology, positions, box = _build_synthetic_charge_transfer_system()
    spec = _freeze_spec(system, topology, positions, box)

    # 模拟 v1 的错误用法：先配置一次（模拟 dynamics 产出的 system_prepared.xml），
    # 如果 `ukn` 把这个"已配置"的 system 当"原始" system 传进去，
    # `_assert_single_restraint_force(..., n_expected=0)` 必须拦住它。
    ie.configure_pme_ligand_charge_offsets(
        system, LIGAND_INDICES, lambda_name="lambda_coul", allow_charged_ligand=True,
        topology=topology, positions=positions, box_vectors=box, co_alchemical_ion_spec=spec,
    )
    with pytest.raises(SystemExit):
        mod._assert_single_restraint_force(system, n_expected=0)

    # 而真正的原始 system（build 阶段的 system.xml，从未配置过）必须通过同一断言。
    raw_system, raw_topology, raw_positions, raw_box = _build_synthetic_charge_transfer_system()
    mod._assert_single_restraint_force(raw_system, n_expected=0)  # 不 raise = 通过


# ============================================================================
# 第 9 组：report 缺产物必须 incomplete（不需要 OpenMM）
# ============================================================================


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_incomplete_when_dynamics_missing(tmp_path):
    out_dir = tmp_path / "case"
    out_dir.mkdir()
    _write_json(out_dir / "build_manifest.json", {
        "protocol_version": mod.PROTOCOL_VERSION, "case": "Na_thin_pos0", "ion": "Na",
        "ligand_net_charge_e": 1, "water_thickness_label": "thin", "position_variant": 0,
        "ligand_coion_min_image_distance_nm": 2.0, "total_charge_at_build_e": 0.0,
    })
    _write_json(out_dir / "static_check_report.json", {"passed": True})
    # 故意不写 dynamics_manifest.json / charging_delta_G.json / slab_quality_gate.json

    args = argparse.Namespace(output_dir=str(out_dir))
    mod.cmd_report(args)

    with open(out_dir / "report.json") as fh:
        report = json.load(fh)
    assert report["status"] == "incomplete"
    assert report["passed"] is False
    assert "dynamics_manifest.json" in report["missing_artifacts"]
    assert "charging_delta_G.json" in report["missing_artifacts"]
    assert "slab_quality_gate.json" in report["missing_artifacts"]


def test_report_incomplete_when_slab_quality_gate_missing_but_rest_present(tmp_path):
    out_dir = tmp_path / "case"
    out_dir.mkdir()
    _write_json(out_dir / "build_manifest.json", {
        "protocol_version": mod.PROTOCOL_VERSION, "case": "Na_thin_pos0", "ion": "Na",
        "ligand_net_charge_e": 1, "water_thickness_label": "thin", "position_variant": 0,
        "ligand_coion_min_image_distance_nm": 2.0, "total_charge_at_build_e": 0.0,
    })
    _write_json(out_dir / "static_check_report.json", {"passed": True})
    _write_json(out_dir / "dynamics_manifest.json", {
        "dcd_paths": [f"traj_{i}.dcd" for i in range(11)],
        "lambdas_coul": [round(1.0 - 0.1 * i, 2) for i in range(11)],
    })
    _write_json(out_dir / "charging_delta_G.json", {
        "delta_G_charging_kJ_mol": 10.0, "uncertainty_kJ_mol": 1.0,
        "delta_G_charging_kcal_mol": 2.39, "uncertainty_kcal_mol": 0.24, "converged": True,
    })
    # 故意不写 slab_quality_gate.json

    args = argparse.Namespace(output_dir=str(out_dir))
    mod.cmd_report(args)
    with open(out_dir / "report.json") as fh:
        report = json.load(fh)
    assert report["status"] == "incomplete"
    assert report["passed"] is False
    assert report["missing_artifacts"] == ["slab_quality_gate.json"]


def test_report_passes_when_all_artifacts_present_and_consistent(tmp_path):
    out_dir = tmp_path / "case"
    out_dir.mkdir()
    _write_json(out_dir / "build_manifest.json", {
        "protocol_version": mod.PROTOCOL_VERSION, "case": "Na_thin_pos0", "ion": "Na",
        "ligand_net_charge_e": 1, "water_thickness_label": "thin", "position_variant": 0,
        "ligand_coion_min_image_distance_nm": 2.0, "total_charge_at_build_e": 0.0,
    })
    _write_json(out_dir / "static_check_report.json", {"passed": True})
    _write_json(out_dir / "dynamics_manifest.json", {
        "dcd_paths": [f"traj_{i}.dcd" for i in range(11)],
        "lambdas_coul": [round(1.0 - 0.1 * i, 2) for i in range(11)],
    })
    _write_json(out_dir / "charging_delta_G.json", {
        "delta_G_charging_kJ_mol": 10.0, "uncertainty_kJ_mol": 1.0,
        "delta_G_charging_kcal_mol": 2.39, "uncertainty_kcal_mol": 0.24, "converged": True,
    })
    _write_json(out_dir / "slab_quality_gate.json", {"passed": True, "checks": {}, "failure_reasons": []})

    args = argparse.Namespace(output_dir=str(out_dir))
    mod.cmd_report(args)
    with open(out_dir / "report.json") as fh:
        report = json.load(fh)
    assert report["status"] == "complete"
    assert report["passed"] is True
    assert report["missing_artifacts"] == []


# ============================================================================
# 第 11 组：compare 双阈值门（不需要 OpenMM）
# ============================================================================


def _fake_report(case: str, dg_kcal: float, err_kcal: float) -> dict:
    return {
        "case": case,
        "charging_delta_G": {
            "delta_G_charging_kcal_mol": dg_kcal, "uncertainty_kcal_mol": err_kcal,
        },
    }


def test_compare_requires_both_thresholds(tmp_path):
    report_a = tmp_path / "a.json"
    report_b = tmp_path / "b.json"

    # 情形 1：差值很小，两条阈值都满足 → passed=True。
    _write_json(report_a, _fake_report("thin_pos0", 0.0, 0.1))
    _write_json(report_b, _fake_report("thin_pos1", 0.2, 0.1))
    args = argparse.Namespace(report_a=str(report_a), report_b=str(report_b), label="t", output=None)
    result = _run_compare_capture(args)
    assert result["passed"] is True

    # 情形 2：差值绝对值超过 1 kcal/mol，即使 2σ 很宽也必须 fail。
    _write_json(report_a, _fake_report("thin_pos0", 0.0, 5.0))
    _write_json(report_b, _fake_report("thin_pos1", 1.5, 5.0))
    result = _run_compare_capture(args)
    assert abs(result["delta_delta_G_kcal_mol"] - 1.5) < 1e-9
    assert result["passed"] is False  # 1.5 kcal/mol > 1.0 kcal/mol 硬阈值

    # 情形 3：差值绝对值 < 1 kcal/mol，但 2σ_combined 更严格，也必须 fail。
    _write_json(report_a, _fake_report("thin_pos0", 0.0, 0.05))
    _write_json(report_b, _fake_report("thin_pos1", 0.3, 0.05))
    result = _run_compare_capture(args)
    assert result["passed"] is False  # 0.3 > 2*sqrt(0.05^2+0.05^2) ≈ 0.141


def _run_compare_capture(args) -> dict:
    """`cmd_compare` 只 print + 可选写文件，这里加一个临时 --output 抓返回值。"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as tmp:
        out_path = tmp.name
    ns = argparse.Namespace(report_a=args.report_a, report_b=args.report_b, label=args.label, output=out_path)
    mod.cmd_compare(ns)
    with open(out_path) as fh:
        result = json.load(fh)
    os.unlink(out_path)
    return result


# ============================================================================
# 第 10 组：slab-quality-gate 必须能抓住人为制造的违规（需要 openmm + mdtraj）
# ============================================================================

md = pytest.importorskip("mdtraj")


def _minimal_slab_topology():
    """4 个单原子"P31"脂质头（2 上叶/2 下叶）+ 1 ligand(Na+) + 1 coion(Na+) + 少量水。

    只用来喂 `mdtraj.Topology`，不需要真实 Lipid21 全原子结构——
    `core._coion_observables_from_trajectory`/`_p31_indices` 只认原子名。
    """
    top = app.Topology()
    chain = top.addChain()
    p31_indices = []
    for i, z in enumerate([1.0, 1.2, 9.0, 9.2]):  # 上叶 z~1, 下叶 z~9（中面~5）
        res = top.addResidue("POPC", chain)
        atom = top.addAtom("P31", app.element.phosphorus, res)
        p31_indices.append(atom.index)
    lig_res = top.addResidue("Na+", chain)
    lig_atom = top.addAtom("Na+", app.element.sodium, lig_res)
    coion_res = top.addResidue("Na+", chain)
    coion_atom = top.addAtom("Na+", app.element.sodium, coion_res)
    for _ in range(4):
        water_res = top.addResidue("TP3", chain)
        top.addAtom("O", app.element.oxygen, water_res)
        top.addAtom("H1", app.element.hydrogen, water_res)
        top.addAtom("H2", app.element.hydrogen, water_res)
    return top, p31_indices, lig_atom.index, coion_atom.index


def _write_synthetic_dcd(path: str, topology, xyz_frames: np.ndarray, box_nm: np.ndarray) -> None:
    md_top = md.Topology.from_openmm(topology)
    unitcell_lengths = np.tile(box_nm.diagonal(), (xyz_frames.shape[0], 1))
    unitcell_angles = np.tile([90.0, 90.0, 90.0], (xyz_frames.shape[0], 1))
    traj = md.Trajectory(
        xyz=xyz_frames, topology=md_top, unitcell_lengths=unitcell_lengths, unitcell_angles=unitcell_angles,
    )
    traj.save_dcd(path)


def _setup_case_dir(tmp_path, coion_z_by_frame, n_frames=4):
    top, p31_indices, lig_idx, coion_idx = _minimal_slab_topology()
    n_atoms = top.getNumAtoms()
    box_nm = np.diag([4.0, 4.0, 10.0])

    base_xyz = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    base_xyz[:, p31_indices[0]] = [2.0, 2.0, 1.0]
    base_xyz[:, p31_indices[1]] = [2.2, 2.0, 1.2]
    base_xyz[:, p31_indices[2]] = [2.0, 2.0, 9.0]
    base_xyz[:, p31_indices[3]] = [2.2, 2.0, 9.2]
    base_xyz[:, lig_idx] = [1.0, 1.0, 1.0]  # 配体固定在上叶一侧 bulk water
    for f in range(n_frames):
        base_xyz[f, coion_idx] = [1.0, 1.0, coion_z_by_frame[f]]
    # 水原子随便摆在盒子中间，避免与其它原子重叠即可
    water_start = coion_idx + 1
    for w in range(4):
        o = water_start + 3 * w
        base_xyz[:, o] = [3.0 + 0.1 * w, 3.0, 5.0]
        base_xyz[:, o + 1] = [3.05 + 0.1 * w, 3.0, 5.0]
        base_xyz[:, o + 2] = [3.0 + 0.1 * w, 3.05, 5.0]

    case_dir = tmp_path / "case"
    case_dir.mkdir()
    dynamics_dir = case_dir / "dynamics"
    dynamics_dir.mkdir()
    dcd_path = str(dynamics_dir / "traj_state00_lam1.00.dcd")
    _write_synthetic_dcd(dcd_path, top, base_xyz, box_nm)

    save_interval_steps = 500
    timestep_ps = 0.002
    csv_path = dynamics_dir / "timeseries.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "lambda_state_index", "lambda_coul", "step", "time_ps", "total_charge_e",
            "ligand_charge_e", "coion_charge_e", "potential_kJ_mol", "max_force_kJ_mol_nm",
            "ligand_coion_distance_nm", "coion_water_coordination", "restraint_energy_kJ_mol",
            "box_x_nm", "box_y_nm", "box_z_nm", "box_volume_nm3",
        ])
        for f in range(n_frames):
            dist = float(np.linalg.norm(base_xyz[f, lig_idx] - base_xyz[f, coion_idx]))
            writer.writerow([
                0, 1.0, (f + 1) * save_interval_steps, (f + 1) * save_interval_steps * timestep_ps,
                0.0, 1.0, 0.0, -1000.0, 100.0, dist, 5, 1.0, 4.0, 4.0, 10.0, 160.0,
            ])

    dyn_manifest = {
        "protocol_version": mod.PROTOCOL_VERSION, "case": "synthetic",
        "lambdas_coul": [1.0], "dcd_paths": [dcd_path], "timeseries_csv": str(csv_path),
        "expected_frames_per_state": n_frames, "save_interval_steps": save_interval_steps,
        "timestep_ps": timestep_ps,
    }
    _write_json(case_dir / "dynamics_manifest.json", dyn_manifest)

    spec = {"ions": [{"atom_index": coion_idx, "element": "sodium",
                       "restraint": {"force_group": core.CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP}}]}
    _write_json(case_dir / "coalchemical_ion_spec.json", spec)
    _write_json(case_dir / "ligand_indices.json", {"ligand_indices": [lig_idx]})
    manifest = {
        "protocol_version": mod.PROTOCOL_VERSION, "case": "synthetic", "ligand_net_charge_e": 1,
    }
    _write_json(case_dir / "build_manifest.json", manifest)

    # system.xml：随便一个空 System 即可，`cmd_slab_quality_gate` 不用它算能量。
    empty_system = openmm.System()
    for _ in range(n_atoms):
        empty_system.addParticle(1.0 * unit.dalton)
    with open(case_dir / "system.xml", "w") as fh:
        fh.write(openmm.XmlSerializer.serialize(empty_system))
    from openmm.app import PDBxFile
    positions = base_xyz[0] * unit.nanometer
    top.setPeriodicBoxVectors(box_nm * unit.nanometer)
    PDBxFile.writeFile(top, positions, str(case_dir / "topology.cif"))
    np.save(case_dir / "positions_nm.npy", base_xyz[0])
    np.save(case_dir / "box_vectors_nm.npy", box_nm)

    return case_dir


def test_slab_quality_gate_passes_when_coion_stays_put(tmp_path):
    # co-ion 全程停留在上叶一侧、|Δz| 与中面距离足够远（中面≈5.0，coion z=1.0 → Δz=4.0 ≥ 3.0）。
    case_dir = _setup_case_dir(tmp_path, coion_z_by_frame=[1.0, 1.0, 1.0, 1.0])
    mod.cmd_slab_quality_gate(argparse.Namespace(output_dir=str(case_dir)))
    with open(case_dir / "slab_quality_gate.json") as fh:
        result = json.load(fh)
    assert result["checks"]["coion_never_flips_membrane_side"] is True


def test_slab_quality_gate_fails_when_coion_flips_membrane_side(tmp_path):
    # co-ion 从上叶一侧（z=1.0，中面上方）漂到下叶一侧（z=9.0，中面下方）——必须被拦。
    case_dir = _setup_case_dir(tmp_path, coion_z_by_frame=[1.0, 1.0, 9.0, 9.0])
    mod.cmd_slab_quality_gate(argparse.Namespace(output_dir=str(case_dir)))
    with open(case_dir / "slab_quality_gate.json") as fh:
        result = json.load(fh)
    assert result["passed"] is False
    assert result["checks"]["coion_never_flips_membrane_side"] is False


def test_slab_quality_gate_fails_on_nonfinite_energy_in_timeseries(tmp_path):
    case_dir = _setup_case_dir(tmp_path, coion_z_by_frame=[1.0, 1.0, 1.0, 1.0])
    csv_path = case_dir / "dynamics" / "timeseries.csv"
    rows = list(csv.reader(open(csv_path)))
    header, data = rows[0], rows[1:]
    data[1][header.index("potential_kJ_mol")] = "nan"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(data)

    mod.cmd_slab_quality_gate(argparse.Namespace(output_dir=str(case_dir)))
    with open(case_dir / "slab_quality_gate.json") as fh:
        result = json.load(fh)
    assert result["passed"] is False
    assert result["checks"]["energy_force_restraint_finite_every_frame"] is False


def test_slab_quality_gate_fails_when_ligand_coion_distance_too_small(tmp_path):
    case_dir = _setup_case_dir(tmp_path, coion_z_by_frame=[1.0, 1.0, 1.0, 1.0])
    csv_path = case_dir / "dynamics" / "timeseries.csv"
    rows = list(csv.reader(open(csv_path)))
    header, data = rows[0], rows[1:]
    idx = header.index("ligand_coion_distance_nm")
    data[2][idx] = "0.1"  # 远小于 COION_LIGAND_MIN_IMAGE_RUNTIME_NM
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(data)

    mod.cmd_slab_quality_gate(argparse.Namespace(output_dir=str(case_dir)))
    with open(case_dir / "slab_quality_gate.json") as fh:
        result = json.load(fh)
    assert result["passed"] is False
    assert result["checks"]["ligand_coion_distance_ge_1p2nm"] is False
