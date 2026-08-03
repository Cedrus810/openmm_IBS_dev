"""MEM-00c：共炼金反离子身份在动力学与 `compute_u_kn` 之间可能漂移。

对应 `memtodolist.md` §0.5.1 MEM-00c、§3.4、§14 R3，以及 `docs/TODO.md` 的 MEM-00c 条目。

## 被钉住的事实

`_select_bulk_water_counterion`（`ibs_engine.py:766`）**按传入坐标当场排序挑离子**，
排序主键是"到最近溶质的 minimum-image 距离"这个连续量。它没有任何持久化身份，
也没有"只读不选"的模式。

而这条选择路径在三个地方各自独立地被调用一次，三处都传 `allow_charged_ligand=True`：

| 用途 | 位置 | 传入坐标 |
| --- | --- | --- |
| 动力学 System 构建 | `_prepare_pme_mixed_alchemical_system`（`ibs_engine.py:1576`） | 调用方的 `positions` |
| REMD 副本构建 | `_build_replicas`（`ibs_engine.py:13347`） | `self.positions` |
| 能量重算 | `compute_u_kn`（`ibs_engine.py:14763`） | `reference_positions` |

三处在**同一进程内**拿到的是同一个 `self.positions`，所以单进程内不会漂。
真正的漂移入口是**跨进程 resume**（`abfe_pipeline.py`）：

- 首跑：`pre_equilibrate()` → `self.positions = equil_data["positions"]`（:5575）
  → 再叠一次 2000 步快速最小化 `self.positions = state.getPositions()`（:5594）；
- resume 且 `skip_equil=True`：`self.positions = traj.xyz[-1]`（:5561），
  直接读 `pre_equilibration.dcd` 末帧，**不再做那 2000 步最小化**。

即：首跑用坐标 P₁ 跑完动力学，resume 进程用坐标 P₂ ≠ P₁ 重算 u_kn。
两者喂给同一个选择器，选出的离子可能不是同一个粒子 →
u_kn 与动力学 Hamiltonian 静默不一致。2000 步最小化的原子位移量级是
0.01–0.1 nm，下面 `test_..._flips_under_minimization_scale_perturbation`
证明 0.05 nm 就足以翻转选择结果。

## 触发条件与影响面

只在**配体净电荷 ≠ 0** 时触发（`lig_net_charge == 0` 直接 return 空）。
当前生产体系 Atenolol 是中性，所以 07-29 落盘的
181.00 / 157.84 / −5.535906 kcal/mol 基线**不受影响**。

## 这些测试何时该被改写

`memtodolist.md` §3.4 要求"选中后身份写入 manifest""resume 时逐项核对，任何身份
漂移都拒绝旧缓存"。等 Phase B3 的 charge-transfer 实现落地并带上钉身份的 manifest 后：

- `test_selection_is_position_dependent_*` 这两条是**缺陷表征测试**（characterization），
  届时必须替换为"给定 manifest 时选择器只读不选 / 身份漂移则 fail closed"的契约测试；
- `test_selector_exposes_no_pinned_identity_parameter` 会因为新参数出现而失败，
  这是**故意的**——它就是提醒改写这个文件的钩子。

不要为了让本文件变绿而放宽任何判据。
"""

import inspect
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

openmm = pytest.importorskip("openmm")
from openmm import NonbondedForce, app, unit

import ibs_engine as ie

# 与 test_audit_protocol_regressions.py 同口径：这些网络文件系统上 Path.resolve()
# 可能抛错，只取词法绝对路径。
ROOT = Path(__file__).absolute().parents[1]

BOX_NM = 4.0


def _build_two_candidate_system(cl_b_z_nm: float):
    """LIG(+1) + 蛋白重原子 + 2 个 CL⁻ + 2 个水，两个 CL 的排序主键接近相持。

    几何全部沿 z 轴摆开，唯一变量是 CL_B 的 z 坐标：

        ligand C1  z = 0.30   (q = +1)
        protein CA z = 1.50   (q =  0，唯一的非配体重原子溶质)
        CL_A       z = 2.50   → 到最近溶质(CA)距离 1.00 nm
        CL_B       z = cl_b_z → 到最近溶质(CA)距离 |cl_b_z - 1.50| nm

    `_select_bulk_water_counterion` 按 solute_dist **降序**取第一个，所以
    cl_b_z > 2.50 时选 CL_B，cl_b_z < 2.50 时选 CL_A。

    水放在 x 方向远处，使两个 CL 的 water_coordination 都为 0——次级排序键不参与
    本测试，保证翻转完全由主键（距离）驱动。
    """
    topology = app.Topology()
    chain = topology.addChain()

    ligand_res = topology.addResidue("LIG", chain)
    topology.addAtom("C1", app.element.carbon, ligand_res)

    protein_res = topology.addResidue("ALA", chain)
    topology.addAtom("CA", app.element.carbon, protein_res)

    cl_a_res = topology.addResidue("CL", chain)
    topology.addAtom("CL", app.element.chlorine, cl_a_res)
    cl_b_res = topology.addResidue("CL", chain)
    topology.addAtom("CL", app.element.chlorine, cl_b_res)

    for _ in range(2):
        water_res = topology.addResidue("HOH", chain)
        topology.addAtom("O", app.element.oxygen, water_res)
        topology.addAtom("H1", app.element.hydrogen, water_res)
        topology.addAtom("H2", app.element.hydrogen, water_res)

    force = NonbondedForce()
    charges = [
        1.0,   # ligand C1
        0.0,   # protein CA
        -1.0,  # CL_A
        -1.0,  # CL_B
        -0.834, 0.417, 0.417,  # water 1
        -0.834, 0.417, 0.417,  # water 2
    ]
    for charge in charges:
        force.addParticle(
            charge * unit.elementary_charge,
            0.3 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )

    positions_nm = np.asarray(
        [
            [0.5, 0.5, 0.30],       # ligand
            [0.5, 0.5, 1.50],       # protein heavy atom
            [0.5, 0.5, 2.50],       # CL_A
            [0.5, 0.5, cl_b_z_nm],  # CL_B
            [3.0, 0.5, 0.50], [3.1, 0.5, 0.50], [2.9, 0.5, 0.50],
            [3.0, 2.0, 0.50], [3.1, 2.0, 0.50], [2.9, 2.0, 0.50],
        ],
        dtype=float,
    )
    box = np.eye(3) * BOX_NM
    return force, topology, positions_nm * unit.nanometer, box


# ---------------------------------------------------------------------------
# 1. 同一输入必须给同一答案（防止将来引入集合/字典序带来的不确定性）
# ---------------------------------------------------------------------------


def test_selection_is_deterministic_for_identical_inputs():
    force, topology, positions, box = _build_two_candidate_system(2.52)

    first, first_refs, first_meta = ie._select_bulk_water_counterion(
        force, [0], topology, positions, box
    )
    second, second_refs, second_meta = ie._select_bulk_water_counterion(
        force, [0], topology, positions, box
    )

    assert first == second, "同一坐标两次调用必须选出同一个粒子"
    np.testing.assert_allclose(np.asarray(first_refs), np.asarray(second_refs))
    assert first_meta["selected"] == second_meta["selected"]


# ---------------------------------------------------------------------------
# 2. MEM-00c 本体：身份由坐标决定，最小化量级的位移就能翻转
# ---------------------------------------------------------------------------


def test_selection_is_position_dependent_and_flips_under_minimization_scale_perturbation():
    """0.05 nm 的位移翻转被选中的离子——这就是 MEM-00c 的可复现证据。

    0.05 nm 远小于 `abfe_pipeline.py:5583` 那 2000 步快速最小化的典型原子位移，
    也小于 NPT 预平衡末段的热运动幅度。因此首跑（做了最小化）与 resume
    （`skip_equil` 分支直接读 DCD 末帧、不做最小化）完全可能选到不同粒子。
    """
    force_far, topology_far, pos_far, box = _build_two_candidate_system(2.52)
    selected_far, refs_far, _ = ie._select_bulk_water_counterion(
        force_far, [0], topology_far, pos_far, box
    )

    force_near, topology_near, pos_near, _ = _build_two_candidate_system(2.47)
    selected_near, refs_near, _ = ie._select_bulk_water_counterion(
        force_near, [0], topology_near, pos_near, box
    )

    # 配体 +1 → 只需要 1 个 −1 e 反离子。
    assert len(selected_far) == 1
    assert len(selected_near) == 1

    assert selected_far == [3], "CL_B 更远时应当被选中"
    assert selected_near == [2], "CL_B 被挪近后应当改选 CL_A"
    assert selected_far != selected_near, (
        "MEM-00c 已被修复？那么本表征测试必须替换为 §3.4 的 manifest 钉身份契约测试，"
        "不要直接删掉。"
    )

    # restraint 参考点跟着身份一起漂——不只是索引换了，锚定位置也换了。
    assert not np.allclose(np.asarray(refs_far), np.asarray(refs_near))


def test_selection_metadata_records_no_persistent_identity():
    """metadata 里只有本次排序结果，没有任何可跨进程核对的身份指纹。"""
    force, topology, positions, box = _build_two_candidate_system(2.52)
    _, _, metadata = ie._select_bulk_water_counterion(
        force, [0], topology, positions, box
    )

    assert metadata["selection"] == (
        "max_minimum_image_distance_to_nearest_solute_then_water_coordination"
    )
    # §3.4 要求写入 manifest 的字段目前一个都不在：residue index/name、元素、
    # 端点电荷、sigma/epsilon/mass、restraint 参数、protocol version。
    for absent in (
        "residue_index",
        "residue_name",
        "element",
        "charge_endpoints_e",
        "restraint",
        "protocol_version",
    ):
        assert absent not in metadata, (
            f"metadata 已包含 {absent!r}——身份持久化正在落地，"
            "本文件需要按 §3.4 改写为契约测试"
        )


def test_neutral_ligand_selects_nothing_so_production_baseline_is_unaffected():
    """净电荷为 0 时直接短路返回——当前 Atenolol 生产基线不经过这条路径。"""
    force, topology, positions, box = _build_two_candidate_system(2.52)
    charge, sigma, epsilon = force.getParticleParameters(0)
    force.setParticleParameters(0, 0.0 * unit.elementary_charge, sigma, epsilon)

    selected, refs, metadata = ie._select_bulk_water_counterion(
        force, [0], topology, positions, box
    )
    assert selected == []
    assert refs == []
    assert metadata == {}


# ---------------------------------------------------------------------------
# 3. 结构性证据：三个调用点都在"重新选"，没有任何只读入口
# ---------------------------------------------------------------------------

_PINNED_IDENTITY_PARAM_NAMES = (
    "coalchemical_ion_indices",
    "co_alchemical_ion_indices",
    "coion_indices",
    "coion_spec",
    "pinned_ion_indices",
    "ion_indices",
)


@pytest.mark.parametrize(
    "func_name",
    [
        "_select_bulk_water_counterion",
        "configure_coalchemical_neutral_decharging",
        "configure_pme_ligand_charge_offsets",
        "_prepare_pme_coulomb_leg_system",
        "_prepare_pme_mixed_alchemical_system",
    ],
)
def test_selector_exposes_no_pinned_identity_parameter(func_name):
    """整条链路上没有"传入已选定身份、只核对不重选"的参数。

    这是 MEM-00c 之所以是**结构性**缺陷而不是偶发 bug 的原因：调用方即便知道
    首跑选了哪个离子，也无处可传。B3 落地钉身份参数后本测试会失败——那是提醒信号。
    """
    func = getattr(ie, func_name)
    params = set(inspect.signature(func).parameters)
    found = params & set(_PINNED_IDENTITY_PARAM_NAMES)
    assert not found, (
        f"{func_name} 已出现钉身份参数 {sorted(found)}；"
        "请按 memtodolist.md §3.4 把本文件改写为 manifest 核对契约测试"
    )


def test_three_call_sites_all_enable_charged_ligand_reselection():
    """动力学 / REMD 副本 / u_kn 重算三处都传 `allow_charged_ligand=True`。

    只要这三处存在且都会走进选择器，跨进程坐标差异就会变成身份差异。
    """
    engine_src = (ROOT / "ibs_engine.py").read_text(encoding="utf-8")
    assert engine_src.count("allow_charged_ligand=True") == 3, (
        "`allow_charged_ligand=True` 的调用点数量变了；MEM-00c 的影响面随之改变，"
        "请重新核对 memtodolist.md §0.5.1 的 file:line 表"
    )


def test_resume_and_fresh_paths_feed_different_coordinates_to_the_selector():
    """`abfe_pipeline.py` 的首跑分支与 resume 分支写入的 `self.positions` 来源不同。

    这条把"坐标会变"从推断变成源码事实：resume 分支读 DCD 末帧，首跑分支在
    预平衡输出之上又叠了一次最小化。两者都会成为 `compute_u_kn` 的
    `reference_positions`。
    """
    pipeline_src = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")

    # resume + skip_equil：直接取轨迹末帧。
    assert "self.positions = traj.xyz[-1] * unit.nanometer" in pipeline_src
    # 首跑：预平衡结果之上再做 2000 步最小化。
    assert 'equil_data = self.pre_equilibrate(resume=resume)' in pipeline_src
    assert "sim.minimizeEnergy(maxIterations=2000)" in pipeline_src
    # 两条分支的产物最终都被当作 u_kn 重算的参考坐标。
    assert pipeline_src.count("reference_positions=self.positions") == 3
