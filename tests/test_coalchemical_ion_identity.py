"""MEM-00c：co-ion 身份必须选一次、冻结、之后处处只读核对。

对应 `memtodolist.md` §0.5.1 MEM-00c、§3.4、§14 R3，以及 `docs/TODO.md` 的 MEM-00c 条目。

## 缺陷是什么（已修，本文件现在是契约测试）

`ibs_engine._select_bulk_water_counterion` 按**传入坐标**当场排序挑离子，排序主键是
"到最近溶质的 minimum-image 距离"这个连续量。而喂给它的坐标在**跨进程 resume**
时本来就不同：

- 首跑：`pre_equilibrate()` 的输出**再叠 2000 步快速最小化**；
- resume（`skip_equil`）：直接读 `pre_equilibration.dcd` **末帧**，不做最小化。

两者差 0.01–0.1 nm，而 `test_frozen_identity_survives_...` 里量到 **0.05 nm 就足以
翻转选择结果**。修复前动力学 / REMD 副本 / `compute_u_kn` 三处各自调一次选择器，
于是首跑用粒子 A 跑动力学、resume 进程用粒子 B 重算 u_kn ——
**u_kn 与动力学 Hamiltonian 静默不一致，ΔG 会错而没有任何异常现象。**

## 修法（2026-08-04，B3 的前置条件）

    选一次（ibs_engine.select_co_alchemical_ion_once）
        ↓
    落成带指纹的 spec（abfe_core.build_co_alchemical_ion_identity）
        ↓
    落盘 checkpoints/coalchemical_ion_spec.json（跨进程钉住身份的唯一办法）
        ↓
    dynamics / replicas / u_kn / resume 全部经
    abfe_core.verify_co_alchemical_ion_identity 只读消费
        ↓
    带电配体而没有 spec ⟹ fail closed（不再有"自动重选"这条路）

## 不要这样让本文件变绿

任何"放宽核对""按当前坐标重新选一个能对上的""把字段从指纹里拿掉"的改法都是把
不一致藏起来。身份对不上的正解是**重跑这条腿并重新选择**。
"""

import inspect
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

openmm = pytest.importorskip("openmm")
from openmm import NonbondedForce, app, unit

import abfe_core as core
import ibs_engine as ie

# 与 test_audit_protocol_regressions.py 同口径：这些网络文件系统上 Path.resolve()
# 可能抛错，只取词法绝对路径。
ROOT = Path(__file__).absolute().parents[1]

BOX_NM = 4.0

# 与 _build_two_candidate_system 的粒子顺序一一对应。
_MASSES_AMU = (
    12.011,  # ligand C1
    12.011,  # protein CA
    35.45,   # CL_A
    35.45,   # CL_B
    15.999, 1.008, 1.008,  # water 1
    15.999, 1.008, 1.008,  # water 2
)

# `_build_two_candidate_system` 的 z 坐标 → 新鲜选择会挑中的粒子 index。
# CL_A 固定在 z=2.50（到最近溶质 CA 距离 1.00 nm），选择器按距离**降序**取第一个。
_CL_B_FAR_NM = 2.52    # CL_B 更远 → 选 CL_B(index 3)
_CL_B_NEAR_NM = 2.47   # CL_B 更近 → 改选 CL_A(index 2)
_FLIP_DISPLACEMENT_NM = abs(_CL_B_FAR_NM - _CL_B_NEAR_NM)  # 0.05 nm


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


def _build_system(cl_b_z_nm: float):
    """同一个体系，但包成 `openmm.System`——新 API 需要质量与 System 级信息。"""
    force, topology, positions, box = _build_two_candidate_system(cl_b_z_nm)
    system = openmm.System()
    for mass in _MASSES_AMU:
        system.addParticle(mass * unit.dalton)
    system.addForce(force)
    system.setDefaultPeriodicBoxVectors(*(box * unit.nanometer))
    return system, topology, positions, box


def _fresh_selection(cl_b_z_nm: float):
    """不经冻结、直接调选择器 —— 只用来证明"坐标确实会翻转选择结果"。"""
    force, topology, positions, box = _build_two_candidate_system(cl_b_z_nm)
    indices, _, _ = ie._select_bulk_water_counterion(
        force, [0], topology, positions, box
    )
    return indices


# ---------------------------------------------------------------------------
# 1. 选择只发生一次：唯一入口 + 下游 fail closed
# ---------------------------------------------------------------------------


def test_selector_has_exactly_one_caller_in_the_engine():
    """`_select_bulk_water_counterion` 只允许被 `select_co_alchemical_ion_once` 调用。

    多一个调用点就等于多一次"按当时坐标重新选"的机会，MEM-00c 就回来了。
    """
    src = (ROOT / "ibs_engine.py").read_text(encoding="utf-8")
    # 定义 1 次 + 调用 1 次；docstring/注释里的提及不带左括号，不会被算进来。
    assert src.count("_select_bulk_water_counterion(") == 2, (
        "`_select_bulk_water_counterion(` 的出现次数变了。它必须只有"
        "`def` 与 `select_co_alchemical_ion_once` 里那一处调用；"
        "任何新增调用点都会重新引入 MEM-00c。"
    )

    one_time_entry = inspect.getsource(ie.select_co_alchemical_ion_once)
    assert "_select_bulk_water_counterion(" in one_time_entry, (
        "唯一选择入口不再调用选择器 —— 选择逻辑被搬走了？请同步本测试与 MEM-00c 记录。"
    )


def test_charged_ligand_without_frozen_spec_fails_closed():
    """带电配体 + 没有 spec ⟹ raise，而不是"顺手自己选一个"。"""
    system, topology, positions, box = _build_system(_CL_B_FAR_NM)

    with pytest.raises(RuntimeError) as excinfo:
        ie.configure_coalchemical_neutral_decharging(
            system, [0], topology, positions, box_vectors=box
        )
    message = str(excinfo.value)
    assert "MEM-00c" in message
    assert "co_alchemical_ion_spec" in message


def test_neutral_ligand_needs_no_spec_at_all():
    """净电荷为 0 ⟹ 唯一入口返回 None，整条 co-ion 路径不被触发。

    当前生产体系（Atenolol，`[ atoms ]` Σq = 0.000000 e）走的正是这一支，
    所以 07-29 落盘的 181.00 / 157.84 / −5.535906 kcal/mol 基线不受本改动影响。
    """
    system, topology, positions, box = _build_system(_CL_B_FAR_NM)
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    _, sigma, epsilon = nb.getParticleParameters(0)
    nb.setParticleParameters(0, 0.0 * unit.elementary_charge, sigma, epsilon)

    assert ie.select_co_alchemical_ion_once(system, [0], topology, positions, box) is None

    # 中性支不需要 spec，也不会 fail closed。
    original_charges, ion_indices = ie.configure_coalchemical_neutral_decharging(
        system, [0], topology, positions, box_vectors=box
    )
    assert ion_indices == []
    assert set(original_charges) == {0}


# ---------------------------------------------------------------------------
# 2. MEM-00c 的修复本体：坐标变了，身份不变
# ---------------------------------------------------------------------------


def test_frozen_identity_survives_minimization_scale_coordinate_change():
    """0.05 nm 位移会翻转**新鲜选择**，但**不会**动摇已冻结的身份。

    这就是 MEM-00c 修好了的证据：0.05 nm 远小于 `abfe_pipeline` 那 2000 步快速
    最小化的典型原子位移，也小于 NPT 预平衡末段的热运动幅度。
    """
    # (a) 先证明坐标真的会翻转选择结果——否则下面的断言是空的。
    assert _fresh_selection(_CL_B_FAR_NM) == [3], "CL_B 更远时新鲜选择应当选 CL_B"
    assert _fresh_selection(_CL_B_NEAR_NM) == [2], "CL_B 挪近后新鲜选择应当改选 CL_A"
    assert _FLIP_DISPLACEMENT_NM == pytest.approx(0.05)

    # (b) 首跑：在 P₁ 上选一次并冻结。
    system_first, topology_first, positions_first, box = _build_system(_CL_B_FAR_NM)
    spec = ie.select_co_alchemical_ion_once(
        system_first, [0], topology_first, positions_first, box
    )
    assert [ion["atom_index"] for ion in spec["ions"]] == [3]

    # (c) resume 进程：坐标已经是 P₂（新鲜选择会给 index 2），但只读核对必须仍给 3。
    system_resume, topology_resume, _, _ = _build_system(_CL_B_NEAR_NM)
    pinned = core.verify_co_alchemical_ion_identity(
        spec, system=system_resume, topology=topology_resume
    )
    assert pinned == [3], (
        "冻结的身份在 resume 坐标下被改掉了 —— MEM-00c 回归了："
        "u_kn 会用与动力学不同的粒子。"
    )

    # (d) 真正的消费路径（注入 offset + restraint）也必须落在被钉住的粒子上。
    _, ion_indices = ie.configure_coalchemical_neutral_decharging(
        system_resume,
        [0],
        topology_resume,
        positions_first,  # 故意喂 P₁/P₂ 之外的坐标：它已经不参与身份决定
        box_vectors=box,
        co_alchemical_ion_spec=spec,
    )
    assert ion_indices == [3]


def test_restraint_reference_comes_from_the_spec_not_current_coordinates():
    """restraint 参考量也被冻结：参考点漂了，离子被拖去的地方就变了（MEM-00d）。

    ⚠️ 2026-08-04 起 restraint 是 **flat-bottom + 锚点相对**（MEM-00d 已修）：
    被冻结的量从"绝对笛卡尔参考点"换成了"锚点原子 index + minimum-image 位移 d0"。
    本测试的意图不变 —— 消费 spec 时**不许**按当前坐标重算这些量。
    """
    system_first, topology_first, positions_first, box = _build_system(_CL_B_FAR_NM)
    spec = ie.select_co_alchemical_ion_once(
        system_first, [0], topology_first, positions_first, box
    )
    restraint = spec["ions"][0]["restraint"]
    # 配体只有一个重原子（C1，index 0），所以锚点必然是它。
    assert restraint["anchor_atom_index"] == 0
    # d0 = 锚点(z=0.30) → CL_B(z=2.52) 的 minimum-image 位移；盒长 4.0 nm 时
    # 2.22 nm 的直线距离会被回卷成 −1.78 nm，这正是必须用 minimum-image 的原因。
    assert restraint["reference_displacement_nm"] == pytest.approx(
        [0.0, 0.0, _CL_B_FAR_NM - 0.30 - BOX_NM]
    )
    # 退役的绝对参考点只留作审计记录，且改了键名（还在读旧键的消费者会 KeyError）。
    assert "reference_position_nm" not in restraint
    assert restraint["selection_time_absolute_position_nm"] == pytest.approx(
        [0.5, 0.5, _CL_B_FAR_NM]
    )

    # 用一份"离子已经挪了 0.3 nm"的坐标去消费 spec，参考量必须还是 spec 里那些。
    system_moved, topology_moved, positions_moved, _ = _build_system(_CL_B_FAR_NM + 0.3)
    n_forces_before = system_moved.getNumForces()
    ie.configure_coalchemical_neutral_decharging(
        system_moved,
        [0],
        topology_moved,
        positions_moved,
        box_vectors=box,
        co_alchemical_ion_spec=spec,
    )
    added = [
        system_moved.getForce(i)
        for i in range(n_forces_before, system_moved.getNumForces())
    ]
    restraint_forces = [
        f for f in added if isinstance(f, openmm.CustomCompoundBondForce)
    ]
    assert len(restraint_forces) == 1, "co-ion flat-bottom restraint 没有被注入"
    particles, params = restraint_forces[0].getBondParameters(0)
    assert list(particles) == [3, 0], "restraint 必须挂在 (co-ion, 锚点) 这一对上"
    assert list(params) == pytest.approx(restraint["reference_displacement_nm"]), (
        "restraint 参考位移跟着当前坐标漂了 —— 它必须来自冻结的 spec"
    )


# ---------------------------------------------------------------------------
# 3. 指纹必须覆盖"变了就要作废旧缓存"的全部字段
# ---------------------------------------------------------------------------


def test_fingerprint_covers_the_full_invalidation_list():
    """B5 那张作废清单里的每一项都必须在指纹里。

    少一项就等于允许一次静默不一致的 resume：例如只钉 atom_index 而不钉
    sigma/epsilon，换了离子类型却仍复用旧 u_kn。
    """
    for field in (
        "atom_index",       # co-ion particle index
        "residue_name",     # ion type
        "element",          # ion type
        "mass_amu",         # mass
        "sigma_nm",         # sigma
        "epsilon_kj_mol",   # epsilon
        "charge_at_lambda1_e",  # endpoint charges
        "charge_at_lambda0_e",
        "restraint",        # restraint definition + reference position
    ):
        assert field in core.CO_ALCHEMICAL_ION_IDENTITY_FINGERPRINT_FIELDS, field

    system, topology, positions, box = _build_system(_CL_B_FAR_NM)
    spec = ie.select_co_alchemical_ion_once(system, [0], topology, positions, box)
    baseline = core.co_alchemical_ion_identity_fingerprint(spec)

    # 逐字段扰动：任何一项变化都必须改变指纹。
    mutations = {
        "atom_index": 2,
        "residue_name": "BR",
        "element": "Br",
        "mass_amu": 79.9,
        "sigma_nm": 0.4,
        "epsilon_kj_mol": 1.0,
        "charge_at_lambda1_e": -0.5,
        "charge_at_lambda0_e": -1.0,
    }
    for field, mutated in mutations.items():
        import copy

        perturbed = copy.deepcopy(spec)
        perturbed["ions"][0][field] = mutated
        assert core.co_alchemical_ion_identity_fingerprint(perturbed) != baseline, field

    # restraint（含参考位置与力常数）同样进指纹。
    import copy

    for key, mutated in (
        ("k_kj_per_mol_nm2", 50.0),
        ("form", "harmonic_periodicdistance"),   # 换回退役形式也必须让指纹变
        ("flat_bottom_radius_nm", 0.8),
        ("anchor_atom_index", 1),
        ("reference_displacement_nm", [0.5, 0.5, 3.0]),
    ):
        perturbed = copy.deepcopy(spec)
        perturbed["ions"][0]["restraint"][key] = mutated
        assert core.co_alchemical_ion_identity_fingerprint(perturbed) != baseline, key

    # 协议版本与 λ 方向也在里面。
    for key, mutated in (
        ("protocol_version", core.CO_ALCHEMICAL_ION_IDENTITY_PROTOCOL_VERSION + 1),
        ("lambda_direction", "lam_coul_0_to_1"),
        ("charge_treatment", core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER),
    ):
        perturbed = copy.deepcopy(spec)
        perturbed[key] = mutated
        assert core.co_alchemical_ion_identity_fingerprint(perturbed) != baseline, key

    # 诊断量（当时的排序距离/水配位数）刻意**不**进指纹：它们每次读坐标都会变一点，
    # 进指纹会让每次 resume 都误判成身份漂移。
    perturbed = copy.deepcopy(spec)
    perturbed["selection_provenance"] = {"selected": [{"solute_distance_nm": 99.0}]}
    assert core.co_alchemical_ion_identity_fingerprint(perturbed) == baseline


@pytest.mark.parametrize(
    "field, mutated",
    [
        ("residue_name", "BR"),
        ("element", "Br"),
        ("mass_amu", 79.9),
        ("sigma_nm", 0.4),
        ("epsilon_kj_mol", 1.0),
    ],
)
def test_verify_rejects_drift_of_each_identity_field(field, mutated):
    """spec 与当前 System/拓扑不符 ⟹ raise，且指明是 MEM-00c 那类不一致。"""
    system, topology, positions, box = _build_system(_CL_B_FAR_NM)
    spec = ie.select_co_alchemical_ion_once(system, [0], topology, positions, box)
    spec["ions"][0][field] = mutated
    spec["fingerprint"] = core.co_alchemical_ion_identity_fingerprint(spec)

    with pytest.raises(ValueError, match="身份漂移"):
        core.verify_co_alchemical_ion_identity(spec, system=system, topology=topology)


def test_verify_rejects_a_hand_edited_spec():
    """改了字段却不重算指纹 ⟹ 自洽性检查先拦下来。"""
    system, topology, positions, box = _build_system(_CL_B_FAR_NM)
    spec = ie.select_co_alchemical_ion_once(system, [0], topology, positions, box)
    spec["ions"][0]["atom_index"] = 2  # 指纹故意不重算

    with pytest.raises(ValueError, match="自身指纹不符"):
        core.verify_co_alchemical_ion_identity(spec, system=system, topology=topology)


def test_spec_is_not_reusable_across_charge_treatments():
    """co-annihilation 的 spec 不能被声明 charge-transfer 的运行拿去用。

    两条路线的端点电荷不同（q_phys→0 vs 0→q_L），混用等于"声明了一种哈密顿量、
    实际跑另一种"。
    """
    system, topology, positions, box = _build_system(_CL_B_FAR_NM)
    spec = ie.select_co_alchemical_ion_once(system, [0], topology, positions, box)
    assert spec["charge_treatment"] == core.CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL
    assert spec["ions"][0]["charge_at_lambda1_e"] == pytest.approx(-1.0)
    assert spec["ions"][0]["charge_at_lambda0_e"] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="不可跨路线复用"):
        core.verify_co_alchemical_ion_identity(
            spec,
            system=system,
            topology=topology,
            charge_treatment=core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        )


def test_charge_transfer_endpoints_satisfy_the_b2_declaration_validator():
    """按 charge-transfer 造出来的 spec 必须能过 §3.4/§1.2 那套声明校验。

    这是"冻结层"与"B2 校验层"的接缝。B3 已落地（2026-08-04），完整的 charge-transfer
    哈密顿量测试在 `tests/test_charge_transfer_hamiltonian.py`；本条只钉这个接缝。

    ⚠️ 这里必须用一个**大盒 + 摆得够远**的构造：charge-transfer 会强制 §13.1 的几何
    余量（flat-bottom 井要能保证 co-ion 全程离配体 ≥ 1.2 nm），上面那个 4 nm 小盒
    fixture 过不了 —— 那不是判据太严，是 1.78 nm 的 minimum-image 距离在 4 nm 盒里
    本来就装不下这个余量。
    """
    box_nm = 10.0
    topology = app.Topology()
    chain = topology.addChain()
    ligand_res = topology.addResidue("MOL", chain)
    topology.addAtom("C1", app.element.carbon, ligand_res)
    dummy_res = topology.addResidue("NA", chain)
    topology.addAtom("NA", app.element.sodium, dummy_res)
    ion_res = topology.addResidue("CL", chain)
    topology.addAtom("CL", app.element.chlorine, ion_res)

    force = NonbondedForce()
    for charge in (1.0, 0.0, -1.0):   # 配体 +1、预留中性 dummy、配平 Cl⁻
        force.addParticle(
            charge * unit.elementary_charge,
            0.3 * unit.nanometer,
            0.1 * unit.kilojoule_per_mole,
        )
    system = openmm.System()
    for mass in (12.011, 22.99, 35.45):
        system.addParticle(mass * unit.dalton)
    box = np.eye(3) * box_nm
    system.setDefaultPeriodicBoxVectors(*(box * unit.nanometer))
    system.addForce(force)
    positions = np.asarray(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 4.0], [5.0, 5.0, 5.0]], dtype=float
    ) * unit.nanometer

    spec = ie.select_co_alchemical_ion_once(
        system,
        [0],
        topology,
        positions,
        box,
        charge_treatment=core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
    )
    ion = spec["ions"][0]
    assert ion["atom_index"] == 1, "charge-transfer 必须认出那个中性 dummy，而不是 Cl⁻"
    assert ion["charge_at_lambda1_e"] == pytest.approx(0.0), "λ=1 必须是中性 dummy"
    assert ion["charge_at_lambda0_e"] == pytest.approx(1.0), "λ=0 必须接过配体的 +1"

    # B2 的声明校验（`abfe_core._validate_co_alchemical_ion_spec`）必须通过。
    core._validate_co_alchemical_ion_spec(spec["ions"], 1)


# ---------------------------------------------------------------------------
# 4. 结构性契约：每一个消费者都必须能吃到冻结的 spec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func_name",
    [
        "configure_coalchemical_neutral_decharging",
        "configure_pme_ligand_charge_offsets",
        "_prepare_pme_coulomb_leg_system",
        "_prepare_pme_mixed_alchemical_system",
    ],
)
def test_every_consumer_accepts_the_frozen_spec(func_name):
    func = getattr(ie, func_name)
    assert "co_alchemical_ion_spec" in inspect.signature(func).parameters, (
        f"{func_name} 收不到冻结身份 —— 它只能自己重选，MEM-00c 就回来了"
    )


def test_replicas_and_u_kn_both_consume_the_frozen_spec():
    """REMD 副本构建与 u_kn 重算是漂移影响最大的两处，单独钉住。"""
    assert (
        "co_alchemical_ion_spec"
        in inspect.signature(ie.REMDManager.__init__).parameters
    )
    assert (
        "co_alchemical_ion_spec"
        in inspect.signature(ie.TraditionalMBARAnalyzer.compute_u_kn).parameters
    )

    build_src = inspect.getsource(ie.REMDManager._build_replicas)
    assert build_src.count("co_alchemical_ion_spec=self.co_alchemical_ion_spec") == 2, (
        "PME decharging 与 mixed alchemical 两条 replica 分支都必须透传冻结身份"
    )


def test_pipeline_freezes_once_persists_and_reverifies_on_resume():
    """`abfe_pipeline` 侧的三件事：只选一次、落盘、resume 只读核对。"""
    src = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")

    # 唯一选择入口在 pipeline 里只被调用一次（在 resolve_co_alchemical_ion_spec 内部）。
    # import 那行是 `    select_co_alchemical_ion_once,`，不带左括号，不计入。
    assert src.count("select_co_alchemical_ion_once(") == 1, (
        "`select_co_alchemical_ion_once(` 在 pipeline 里应当只有一处调用；"
        "多一处就是多一次重选机会"
    )
    # 落盘 + resume 核对。
    assert "coalchemical_ion_spec.json" in src
    assert "verify_co_alchemical_ion_identity(" in src
    # 每个消费点都走同一个 resolve（而不是各自读盘/各自选）。
    assert src.count("co_alchemical_ion_spec=self.resolve_co_alchemical_ion_spec()") == 6, (
        "消费点数量变了：3 处 REMDManager + 3 处 compute_u_kn。"
        "新增消费点必须同样走 resolve_co_alchemical_ion_spec()"
    )
