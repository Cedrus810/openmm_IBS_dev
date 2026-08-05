"""B3：PME co-alchemical **charge-transfer** charging Hamiltonian + MEM-00d restraint。

对应 `memtodolist.md` §2.1（λ 定义）、§2.2（co-ion 粒子模型）、§2.3 + MEM-00d
（restraint 形式）、§2.4（charging Hamiltonian）、§7.2（电荷守恒）、§7.3（co-ion 物理）、
§13.1（几何阈值）、§13.2（端点容差），以及 §11 Phase B3。

## 这条路线是什么（别和 co-annihilation 搞混）

    charge-transfer（生产）:  ligand: q_L → 0   与   co-ion: 0 → q_L
    co-annihilation（对照）:  ligand: q_L → 0   与   异号反离子: −q_L → 0

两者的总电荷都逐 λ 守恒，但物理完全不同：前者把电荷从结合位点**搬到**体相水，
后者**同时湮灭一对**异号电荷（两个离子处在介电环境完全不同的位置，消失自由能不能
可靠抵消 —— 这正是 Wu & Biggin 在膜体系里推荐 charge-transfer 的理由）。

## 为什么 λ=1 端必须是一个**中性** ion-shaped dummy

体系总电荷在 λ=1 必须等于物理体系的总电荷。设配体 q_L=+1、建系时按 §4.3 用普通离子
把盒子配平（普通离子合计 −1）：

    λ=1:  ligand +1, ordinary −1, co-ion  0  ⟹ 总 0（= 物理体系）
    λ=0:  ligand  0, ordinary −1, co-ion +1  ⟹ 总 0

所以 co-ion 是建系时**额外预留**的一个电荷为 0 的 ion-shaped 粒子，不能拿一个已经
带电的物理盐离子来顶（那样 λ=1 端总电荷就变成 −1 了）。本文件里
`test_a_charged_physical_ion_cannot_be_used_as_the_coion` 就是钉这一条。

## 不要这样让本文件变绿

放宽 §13.1 的几何余量、缩小 flat-bottom 半径、把 restraint 换回绝对笛卡尔参考点、
或者在溶剂腿里临时挑一个盐离子当 co-ion —— 每一条都是把已知缺陷藏起来。
"""

import inspect
import math
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

openmm = pytest.importorskip("openmm")
from openmm import NonbondedForce, app, unit

import abfe_core as core
import ibs_engine as ie

ROOT = Path(__file__).absolute().parents[1]

# 膜体系式的各向异性长方体盒（法向 z），用来同时覆盖 §7.3 的
# "restraint 在三斜/各向异性盒中 minimum-image 正确"。
BOX_NM = np.diag([6.0, 6.0, 12.0])
CUTOFF_NM = 1.0

# 配体三个重原子的物理电荷，净 +1。故意不是等分，这样"逐原子 q_i(λ)=λ·q_i"
# 与"总量对得上"是两件互相独立的断言。
LIGAND_CHARGES_E = (0.5, 0.3, 0.2)
LIGAND_MASSES_AMU = (12.011, 12.011, 14.007)
LIGAND_SIGMA_NM = 0.34
LIGAND_EPSILON_KJ = 0.36

# 建系时预留的中性 ion-shaped dummy（Na 形状、电荷 0）与配平用的普通 Cl⁻。
DUMMY_Z_NM = 8.0        # 距配体足够远，满足 §13.1 的初始与全程余量
LIGAND_Z_NM = 2.0


def _build_charge_transfer_system(
    *,
    n_dummies: int = 1,
    dummy_charge_e: float = 0.0,
    dummy_z_nm: float = DUMMY_Z_NM,
    ligand_net_charge_e: int = 1,
    nonbonded_method=None,
):
    """ligand(净 +q) + reserved 中性 dummy + 配平 Cl⁻ + 两个水。

    返回 `(system, topology, positions(Quantity), box_nm)`。
    盒总电荷 = q_L + Σ(dummy) + Σ(ordinary) = 0，与 §4.3 的 λ=1 账目一致。
    """
    topology = app.Topology()
    chain = topology.addChain()

    ligand_res = topology.addResidue("MOL", chain)
    for name in ("C1", "C2", "N1"):
        topology.addAtom(
            name,
            app.element.nitrogen if name.startswith("N") else app.element.carbon,
            ligand_res,
        )

    for _ in range(int(n_dummies)):
        dummy_res = topology.addResidue("NA", chain)
        topology.addAtom("NA", app.element.sodium, dummy_res)

    # 普通离子：把配体的形式电荷配平掉（§4.3 的"普通离子"那一类）。
    for _ in range(abs(int(ligand_net_charge_e))):
        ion_res = topology.addResidue("CL", chain)
        topology.addAtom("CL", app.element.chlorine, ion_res)

    for _ in range(2):
        water_res = topology.addResidue("HOH", chain)
        topology.addAtom("O", app.element.oxygen, water_res)
        topology.addAtom("H1", app.element.hydrogen, water_res)
        topology.addAtom("H2", app.element.hydrogen, water_res)

    scale = float(ligand_net_charge_e)
    ligand_charges = [q * scale for q in LIGAND_CHARGES_E]
    ordinary_sign = -1.0 if ligand_net_charge_e > 0 else 1.0

    charges = list(ligand_charges)
    charges += [float(dummy_charge_e)] * int(n_dummies)
    charges += [ordinary_sign] * abs(int(ligand_net_charge_e))
    charges += [-0.834, 0.417, 0.417, -0.834, 0.417, 0.417]

    sigmas = [LIGAND_SIGMA_NM] * 3
    sigmas += [0.2439] * int(n_dummies)                      # Na⁺
    sigmas += [0.4478] * abs(int(ligand_net_charge_e))       # Cl⁻
    sigmas += [0.3151, 0.1, 0.1, 0.3151, 0.1, 0.1]

    epsilons = [LIGAND_EPSILON_KJ] * 3
    epsilons += [0.3658] * int(n_dummies)
    epsilons += [0.1489] * abs(int(ligand_net_charge_e))
    epsilons += [0.6364, 0.0, 0.0, 0.6364, 0.0, 0.0]

    masses = list(LIGAND_MASSES_AMU)
    masses += [22.99] * int(n_dummies)
    masses += [35.45] * abs(int(ligand_net_charge_e))
    masses += [15.999, 1.008, 1.008, 15.999, 1.008, 1.008]

    force = NonbondedForce()
    force.setNonbondedMethod(
        NonbondedForce.PME if nonbonded_method is None else nonbonded_method
    )
    force.setCutoffDistance(CUTOFF_NM * unit.nanometer)
    for q, sigma, epsilon in zip(charges, sigmas, epsilons):
        force.addParticle(
            q * unit.elementary_charge,
            sigma * unit.nanometer,
            epsilon * unit.kilojoule_per_mole,
        )

    positions = [
        [3.00, 3.00, LIGAND_Z_NM],
        [3.15, 3.00, LIGAND_Z_NM],
        [3.00, 3.15, LIGAND_Z_NM + 0.14],
    ]
    for i in range(int(n_dummies)):
        positions.append([3.00, 3.00, float(dummy_z_nm) + 0.6 * i])
    for i in range(abs(int(ligand_net_charge_e))):
        positions.append([1.00, 1.00, 5.0 + 0.5 * i])
    positions += [
        [4.50, 3.00, 6.00], [4.60, 3.00, 6.00], [4.40, 3.00, 6.00],
        [2.00, 4.50, 9.00], [2.10, 4.50, 9.00], [1.90, 4.50, 9.00],
    ]

    system = openmm.System()
    for mass in masses:
        system.addParticle(mass * unit.dalton)
    system.setDefaultPeriodicBoxVectors(*(BOX_NM * unit.nanometer))
    system.addForce(force)

    total = sum(charges)
    if float(dummy_charge_e) == 0.0:
        # 正常构造必须电中性（§4.3 的 λ=1 账目）。`dummy_charge_e != 0` 是**故意造坏**
        # 的输入（"拿一个带电物理离子当 co-ion"），那种情况下盒子本来就不中性 ——
        # 这正是要被 fail closed 拦住的东西，不该在 fixture 里先断言掉。
        assert abs(total) < 1e-9, f"fixture 自身不电中性：Σq = {total:+.6f} e"
    return (
        system,
        topology,
        np.asarray(positions, dtype=float) * unit.nanometer,
        BOX_NM,
    )


LIGAND_INDICES = [0, 1, 2]
DUMMY_INDEX = 3


def _freeze_spec(**kwargs):
    system, topology, positions, box = _build_charge_transfer_system(**kwargs)
    spec = ie.select_co_alchemical_ion_once(
        system,
        LIGAND_INDICES,
        topology,
        positions,
        box,
        charge_treatment=core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
    )
    return system, topology, positions, box, spec


def _configured(**kwargs):
    """按 charge-transfer 配置好的 charging System（走生产入口，不手搭）。"""
    system, topology, positions, box, spec = _freeze_spec(**kwargs)
    info = ie.configure_pme_ligand_charge_offsets(
        system,
        LIGAND_INDICES,
        lambda_name="lam_coul",
        allow_charged_ligand=True,
        topology=topology,
        positions=positions,
        box_vectors=box,
        co_alchemical_ion_spec=spec,
    )
    return system, topology, positions, box, spec, info


def _energy_and_forces(system, positions, lam=None):
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(positions)
    context.setPeriodicBoxVectors(*(BOX_NM * unit.nanometer))
    if lam is not None:
        context.setParameter("lam_coul", float(lam))
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = np.asarray(
        state.getForces().value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    )
    del context, integrator
    return energy, forces


def _reference_system_at_lambda(lam: float, *, coion_index: int = DUMMY_INDEX):
    """把 §2.1 的电荷映射**手动**写进一个干净 System，作为独立参照。

    这不是把被测代码抄一遍：被测路径用的是 OpenMM 的 ParameterOffset 机制（电荷在
    Context 内部按 λ 缩放），这里是直接把 `λ·q_i` / `(1−λ)·q_L` 写成粒子电荷。
    两条路线只有在"offset 的语义确实是 q(λ)=base+λ·scale、且我们填对了 base/scale"
    时才会给出同一个能量与同一组力。
    """
    system, topology, positions, box = _build_charge_transfer_system()
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    params = [nb.getParticleParameters(i) for i in range(nb.getNumParticles())]

    for idx in LIGAND_INDICES:
        q, sigma, epsilon = params[idx]
        nb.setParticleParameters(
            idx,
            q.value_in_unit(unit.elementary_charge) * float(lam) * unit.elementary_charge,
            sigma,
            epsilon,
        )
    _q, sigma, epsilon = params[coion_index]
    nb.setParticleParameters(
        coion_index, (1.0 - float(lam)) * unit.elementary_charge, sigma, epsilon
    )

    # 配体内部静电逐 λ 恒定：charging builder 把 L-L 对冻结成显式 exception，
    # 参照体系必须做同样的事，否则比的是两个不同的哈密顿量。
    for i, p1 in enumerate(LIGAND_INDICES):
        q1, sig1, eps1 = params[p1]
        for p2 in LIGAND_INDICES[i + 1:]:
            q2, sig2, eps2 = params[p2]
            nb.addException(
                p1,
                p2,
                (
                    q1.value_in_unit(unit.elementary_charge)
                    * q2.value_in_unit(unit.elementary_charge)
                )
                * unit.elementary_charge**2,
                0.5
                * (
                    sig1.value_in_unit(unit.nanometer)
                    + sig2.value_in_unit(unit.nanometer)
                )
                * unit.nanometer,
                math.sqrt(
                    eps1.value_in_unit(unit.kilojoule_per_mole)
                    * eps2.value_in_unit(unit.kilojoule_per_mole)
                )
                * unit.kilojoule_per_mole,
                True,
            )
    return system, positions


# ---------------------------------------------------------------------------
# 1. §7.2 电荷守恒：端点 + 中间态
# ---------------------------------------------------------------------------


def test_total_charge_is_lambda_independent_including_intermediate_lambdas():
    """Σq(λ) 逐 λ 恒定，且等于物理体系的总电荷（这里是 0）。

    判据是代数的：`Σq(λ) = Σq_base + λ·Σq_scale`，所以 Σq_scale = 0 就覆盖了**所有** λ，
    不是只抽查几个点。逐 λ 数值一并断言，因为它是最容易读懂的证据。
    """
    system, _topology, _positions, _box, _spec, _info = _configured()
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    report = ie.charging_charge_conservation_report(
        nb,
        "lam_coul",
        ligand_indices=LIGAND_INDICES,
        co_ion_indices=[DUMMY_INDEX],
        ligand_net_charge_e=1,
        lambdas=np.linspace(0.0, 1.0, 11),
    )
    assert report["total_charge_is_lambda_independent"]
    assert abs(report["scale_sum_e"]) <= core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E
    for lam, total in report["total_charge_by_lambda_e"].items():
        assert abs(total) <= core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E, lam


def test_ligand_and_coion_charges_follow_the_lambda_definition():
    """§2.1：`Σq_lig(λ) = λ·q_L`，`q_coion(λ) = (1−λ)·q_L`，两者之和恒为 q_L。"""
    system, _topology, _positions, _box, _spec, _info = _configured()
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    lambdas = np.linspace(0.0, 1.0, 11)
    report = ie.charging_charge_conservation_report(
        nb,
        "lam_coul",
        ligand_indices=LIGAND_INDICES,
        co_ion_indices=[DUMMY_INDEX],
        ligand_net_charge_e=1,
        lambdas=lambdas,
    )
    assert report["ligand_charge_matches_lambda_times_qL"]
    for lam in lambdas:
        key = f"{float(lam):.6g}"
        lig = report["ligand_charge_by_lambda_e"][key]
        ion = report["co_ion_charge_by_lambda_e"][key]
        assert lig == pytest.approx(float(lam), abs=core.LIGAND_CHARGE_LAMBDA_TOLERANCE_E)
        assert ion == pytest.approx(
            1.0 - float(lam), abs=core.LIGAND_CHARGE_LAMBDA_TOLERANCE_E
        )
        assert lig + ion == pytest.approx(1.0, abs=1e-9)


def test_negative_ligand_charge_uses_an_anionic_coion():
    """§2.2：`q_L = −1` 要用 `0 → −1` 的阴离子型 co-ion（share 取 sign(q_L)）。"""
    _s, _t, _p, _b, spec = _freeze_spec(ligand_net_charge_e=-1)
    ion = spec["ions"][0]
    assert ion["charge_at_lambda1_e"] == pytest.approx(0.0)
    assert ion["charge_at_lambda0_e"] == pytest.approx(-1.0)
    core._validate_co_alchemical_ion_spec(spec["ions"], -1)


def test_stage2_holds_the_ligand_at_zero_and_the_coion_fully_charged():
    """§2.4 末两条：vanishing 阶段 λ_coul ≡ 0 ⟹ ligand 0、co-ion 满电 q_L。

    这是"不能让 co-ion 在 stage 1 之后被错误恢复成中性"那一条的机械保证：
    co-ion 的电荷是 `(1−λ_coul)·q_L`，vanishing 阶段 λ_coul 恒为 0，所以它**必然**
    停在满电端点，不需要额外记一笔状态。
    """
    system, _topology, _positions, _box, _spec, _info = _configured()
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    report = ie.charging_charge_conservation_report(
        nb,
        "lam_coul",
        ligand_indices=LIGAND_INDICES,
        co_ion_indices=[DUMMY_INDEX],
        ligand_net_charge_e=1,
        lambdas=(0.0,),
    )
    assert report["ligand_charge_by_lambda_e"]["0"] == pytest.approx(0.0, abs=1e-9)
    assert report["co_ion_charge_by_lambda_e"]["0"] == pytest.approx(1.0, abs=1e-9)


def test_a_lambda_path_that_breaks_conservation_is_rejected():
    """Σscale ≠ 0 必须当场 raise，而不是让 PME 用中和背景电荷把它糊过去。"""
    with pytest.raises(ValueError, match="Σscale"):
        core.co_alchemical_charge_offset_plan(
            charge_treatment=core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
            ligand_net_charge_e=1,
            # 配体只交出 0.5 e，co-ion 却接过 1 e —— 账不平。
            ligand_charges_e={0: 0.5},
            co_ion_physical_charges_e={3: 0.0},
        )


# ---------------------------------------------------------------------------
# 2. §13.2 端点与中间态：与独立构造的参照体系逐位对照
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lam", [1.0, 0.0, 0.37])
def test_charging_system_matches_an_independently_built_reference(lam):
    """λ=1 / λ=0 / 中间态的能量与逐原子力都必须与手写参照一致（§13.2 容差）。

    λ=1 参照就是**物理体系**（配体满电、dummy 中性），所以这一条同时证明了
    "charging 体系在 λ=1 回到物理哈密顿量"。
    """
    system, _topology, positions, _box, _spec, _info = _configured()
    got_e, got_f = _energy_and_forces(system, positions, lam=lam)

    ref_system, ref_positions = _reference_system_at_lambda(lam)
    want_e, want_f = _energy_and_forces(ref_system, ref_positions)

    assert got_e == pytest.approx(
        want_e, rel=core.ENDPOINT_ENERGY_RELATIVE_TOLERANCE
    ), f"λ={lam}: 能量 {got_e:.6f} vs 参照 {want_e:.6f} kJ/mol"
    max_diff = float(np.max(np.abs(got_f - want_f)))
    assert max_diff <= core.ENDPOINT_FORCE_MAX_ABS_TOLERANCE_KJ_PER_MOL_NM, (
        f"λ={lam}: 逐原子力最大偏差 {max_diff:.3e} kJ/mol/nm"
    )


def test_every_ligand_pair_is_frozen_as_an_exception_without_a_lambda_offset():
    """配体内部静电必须逐 λ 恒定 —— 去电荷腿只该动 ligand–environment。

    这里用**逐对断言**而不是比能量：配体每一对都必须有一条显式 exception，
    chargeProd 等于物理 `q_i·q_j`，且**没有**挂 `lam_coul` 的 ExceptionParameterOffset。
    这是精确判据，而且与非键方法无关。

    ⚠️ 别改成"把其它电荷置零再比 λ=1 与 λ=0 的总能" —— PME 下那样比不出来：
    Ewald 自能项 −(α/√π)Σq_i² 与非电中性体系的中和背景项**本身就随 λ 变**
    （实测差 20.89 kJ/mol，全部来自这两项，与 L-L 对无关）。
    `ibs_engine` 里那段"PME 自能是真实哈密顿量项、不做修正"的注释说的就是这件事。
    """
    system, _topology, _positions, _box, _spec, _info = _configured()
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))

    physical = {0: 0.5, 1: 0.3, 2: 0.2}
    offsets_by_exception = {
        int(nb.getExceptionParameterOffset(i)[1])
        for i in range(nb.getNumExceptionParameterOffsets())
        if str(nb.getExceptionParameterOffset(i)[0]) == "lam_coul"
    }
    seen = set()
    for exc_idx in range(nb.getNumExceptions()):
        p1, p2, charge_prod, _sigma, _eps = nb.getExceptionParameters(exc_idx)
        p1, p2 = int(p1), int(p2)
        if p1 not in LIGAND_INDICES or p2 not in LIGAND_INDICES:
            continue
        seen.add((min(p1, p2), max(p1, p2)))
        assert charge_prod.value_in_unit(
            unit.elementary_charge**2
        ) == pytest.approx(physical[p1] * physical[p2], abs=1e-12), (
            f"L-L 对 ({p1},{p2}) 的 chargeProd 不是物理值 —— 配体内部库仑被 λ 缩放了"
        )
        assert exc_idx not in offsets_by_exception, (
            f"L-L 对 ({p1},{p2}) 挂上了 lam_coul 的 exception offset —— 它必须逐 λ 恒定"
        )
    assert seen == {(0, 1), (0, 2), (1, 2)}, (
        f"配体三个原子只冻结了 {sorted(seen)}；漏掉的那些对会跟着粒子电荷一起被缩放"
    )


def test_ligand_internal_energy_is_lambda_independent_without_pme_bookkeeping():
    """同一件事的能量侧证据，在 NoCutoff 下量（没有自能/中和背景项来搅局）。"""
    system, topology, positions, box = _build_charge_transfer_system(
        nonbonded_method=NonbondedForce.NoCutoff
    )
    spec = ie.select_co_alchemical_ion_once(
        system,
        LIGAND_INDICES,
        topology,
        positions,
        box,
        charge_treatment=core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
    )
    ie.configure_charge_transfer_decharging(
        system, LIGAND_INDICES, topology, lambda_name="lam_coul",
        co_alchemical_ion_spec=spec,
    )
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    # 把配体以外的所有电荷（含 co-ion 的基电荷与 offset）清零，只留配体内部静电。
    for idx in range(nb.getNumParticles()):
        if idx in LIGAND_INDICES:
            continue
        _q, sigma, epsilon = nb.getParticleParameters(idx)
        nb.setParticleParameters(idx, 0.0 * unit.elementary_charge, sigma, epsilon)
    for offset_idx in range(nb.getNumParticleParameterOffsets()):
        param, particle, _q, _s, _e = nb.getParticleParameterOffset(offset_idx)
        if str(param) == "lam_coul" and int(particle) not in LIGAND_INDICES:
            nb.setParticleParameterOffset(offset_idx, param, int(particle), 0.0, 0.0, 0.0)

    e1, _ = _energy_and_forces(system, positions, lam=1.0)
    e0, _ = _energy_and_forces(system, positions, lam=0.0)
    assert e1 == pytest.approx(e0, abs=1e-9), (
        f"配体内部静电随 λ 变了：λ=1 {e1:.9f} vs λ=0 {e0:.9f} kJ/mol。"
        "L-L 对没有被冻结成显式 exception。"
    )


# ---------------------------------------------------------------------------
# 3. §7.3 co-ion 物理
# ---------------------------------------------------------------------------


def test_coion_mass_and_lj_are_identical_at_every_lambda():
    """§2.2 第一版只改 charge：mass / sigma / epsilon 逐 λ 不变。"""
    system, _topology, _positions, _box, _spec, _info = _configured()
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    _q, sigma, epsilon = nb.getParticleParameters(DUMMY_INDEX)
    assert sigma.value_in_unit(unit.nanometer) == pytest.approx(0.2439)
    assert epsilon.value_in_unit(unit.kilojoule_per_mole) == pytest.approx(0.3658)
    assert system.getParticleMass(DUMMY_INDEX).value_in_unit(unit.dalton) == pytest.approx(
        22.99
    )
    for offset_idx in range(nb.getNumParticleParameterOffsets()):
        param, particle, _q, sigma_scale, eps_scale = nb.getParticleParameterOffset(
            offset_idx
        )
        if int(particle) != DUMMY_INDEX:
            continue
        assert float(sigma_scale) == 0.0 and float(eps_scale) == 0.0, (
            "co-ion 的 sigma/epsilon 挂上了 λ offset —— §2.2 要求它们逐 λ 不变"
        )


def test_coion_charge_goes_through_pme_not_a_cutoff_ghost_force():
    """§2.4：co-ion 静电必须走 PME 的 NonbondedForce offset，不许用 cutoff 自定义力。"""
    system, _topology, _positions, _box, _spec, _info = _configured()
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    assert nb.getNonbondedMethod() == NonbondedForce.PME

    offsets = [
        int(nb.getParticleParameterOffset(i)[1])
        for i in range(nb.getNumParticleParameterOffsets())
        if str(nb.getParticleParameterOffset(i)[0]) == "lam_coul"
    ]
    assert DUMMY_INDEX in offsets, "co-ion 没有进 PME 的 particle parameter offset"

    for force in system.getForces():
        if isinstance(force, openmm.CustomNonbondedForce):
            groups = [
                set(force.getInteractionGroupParameters(i)[0])
                | set(force.getInteractionGroupParameters(i)[1])
                for i in range(force.getNumInteractionGroups())
            ]
            for group in groups:
                assert DUMMY_INDEX not in group, (
                    "co-ion 出现在 CustomNonbondedForce 的相互作用组里 —— "
                    "那会把它的静电截断在 cutoff（§2.4 禁止）"
                )


def test_ghost_ion_handler_is_not_used_for_charge_transfer():
    """MEM-00a-5：`GhostIonHandler` 已退役，不得成为 charge-transfer 的实现。"""
    src = (ROOT / "ibs_engine.py").read_text(encoding="utf-8")
    assert "GhostIonHandler" not in src
    transfer_src = inspect.getsource(ie.configure_charge_transfer_decharging)
    assert "GhostIonHandler" not in transfer_src


# ---------------------------------------------------------------------------
# 4. MEM-00d：flat-bottom + 锚点相对的 restraint
# ---------------------------------------------------------------------------


def test_restraint_is_flat_bottom_anchor_relative_and_recorded_in_the_spec():
    _system, _topology, _positions, _box, spec = _freeze_spec()
    restraint = spec["ions"][0]["restraint"]
    assert restraint["form"] == core.CO_ALCHEMICAL_ION_RESTRAINT_FORM_FLAT_BOTTOM
    assert restraint["reference_frame"] == "anchor_atom_relative_displacement"
    assert restraint["k_kj_per_mol_nm2"] == core.COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2
    assert restraint["flat_bottom_radius_nm"] == core.COION_FLAT_BOTTOM_RADIUS_NM
    assert restraint["anchor_atom_index"] in LIGAND_INDICES
    # d0 = 锚点 → co-ion 的 minimum-image 位移。
    assert restraint["reference_displacement_nm"][2] == pytest.approx(
        DUMMY_Z_NM - LIGAND_Z_NM, abs=0.2
    )
    # 退役的绝对参考点只作审计记录，且**改了名字**，让还在读旧键的消费者 KeyError。
    assert "reference_position_nm" not in restraint
    assert restraint["selection_time_absolute_position_nm"][2] == pytest.approx(DUMMY_Z_NM)


def test_injected_restraint_expression_is_exactly_the_one_recorded():
    """spec 记录的表达式必须与实际注入的逐字符相同（否则 spec 描述的是另一个哈密顿量）。"""
    system, _topology, _positions, _box, spec, _info = _configured()
    forces = [
        f for f in system.getForces() if isinstance(f, openmm.CustomCompoundBondForce)
    ]
    assert len(forces) == 1, "co-ion flat-bottom restraint 没有被注入（或注入了多个）"
    force = forces[0]
    assert force.getEnergyFunction() == core.CO_ALCHEMICAL_ION_RESTRAINT_EXPRESSION
    assert force.getEnergyFunction() == spec["ions"][0]["restraint"]["expression"]
    assert force.getForceGroup() == core.CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP
    assert force.usesPeriodicBoundaryConditions()
    particles, params = force.getBondParameters(0)
    assert list(particles) == [
        spec["ions"][0]["atom_index"],
        spec["ions"][0]["restraint"]["anchor_atom_index"],
    ]
    assert list(params) == pytest.approx(
        spec["ions"][0]["restraint"]["reference_displacement_nm"]
    )


def _restraint_only_system(spec, system_template):
    """只留 restraint 的 System，用来单独量它的能量。"""
    system = openmm.System()
    for i in range(system_template.getNumParticles()):
        system.addParticle(system_template.getParticleMass(i))
    system.setDefaultPeriodicBoxVectors(*(BOX_NM * unit.nanometer))
    ie._inject_co_alchemical_ion_restraints(system, spec)
    return system


def _flat_bottom_energy(ion_xyz, anchor_xyz, d0, box=BOX_NM, k=None, r0=None):
    k = core.COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2 if k is None else k
    r0 = core.COION_FLAT_BOTTOM_RADIUS_NM if r0 is None else r0
    delta = np.asarray(ion_xyz) - (np.asarray(anchor_xyz) + np.asarray(d0))
    r = float(np.linalg.norm(core.minimum_image_displacement_nm(delta, box)))
    return 0.5 * k * max(0.0, r - r0) ** 2, r


def test_restraint_is_flat_inside_the_well_and_harmonic_outside():
    system_t, _topology, positions, _box, spec = _freeze_spec()
    restraint_system = _restraint_only_system(spec, system_t)
    ion = spec["ions"][0]["atom_index"]
    anchor = spec["ions"][0]["restraint"]["anchor_atom_index"]
    d0 = spec["ions"][0]["restraint"]["reference_displacement_nm"]

    pos = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
    for shift, label in ((0.0, "井心"), (0.3, "平坦区内"), (0.5, "刚到墙"), (1.1, "墙外")):
        moved = pos.copy()
        moved[ion] = moved[ion] + np.array([0.0, 0.0, shift])
        want, r = _flat_bottom_energy(moved[ion], moved[anchor], d0)
        got, _ = _energy_and_forces(restraint_system, moved * unit.nanometer)
        assert got == pytest.approx(want, abs=1e-6, rel=1e-9), f"{label}: r={r:.3f} nm"
        if shift <= core.COION_FLAT_BOTTOM_RADIUS_NM:
            assert got == pytest.approx(0.0, abs=1e-9), f"{label} 竟然有力"


def test_restraint_uses_minimum_image_in_the_anisotropic_box():
    """离子与锚点分处周期镜像两侧时，restraint 必须走 minimum-image 而不是炸掉。"""
    system_t, _topology, positions, _box, spec = _freeze_spec()
    restraint_system = _restraint_only_system(spec, system_t)
    ion = spec["ions"][0]["atom_index"]
    anchor = spec["ions"][0]["restraint"]["anchor_atom_index"]
    d0 = spec["ions"][0]["restraint"]["reference_displacement_nm"]

    pos = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
    wrapped = pos.copy()
    # 把 co-ion 挪到盒子另一头（等价于被回卷过一次）。井心也随之落在镜像里。
    wrapped[ion] = wrapped[ion] - np.array([0.0, 0.0, BOX_NM[2, 2]])
    want, r = _flat_bottom_energy(wrapped[ion], wrapped[anchor], d0)
    got, _ = _energy_and_forces(restraint_system, wrapped * unit.nanometer)
    assert r < 1.0, "构造有误：回卷后 minimum-image 距离应当仍然很小"
    assert got == pytest.approx(want, abs=1e-6)


def test_box_scaling_does_not_drag_the_coion_toward_the_membrane():
    """MEM-00d 的回归：barostat 缩放坐标后，井心跟着锚点走，能量仍为 0。

    旧形式（绝对笛卡尔参考点的纯谐振子）在这里会产生一个把离子往回拉的力，
    而"往回"在膜半各向异性 NPT 里就是往膜方向 —— 这正是要退役它的原因。
    对照量一并算出来，免得这条测试退化成"反正都是 0"。
    """
    system_t, _topology, positions, _box, spec = _freeze_spec()
    restraint_system = _restraint_only_system(spec, system_t)
    ion = spec["ions"][0]["atom_index"]
    anchor = spec["ions"][0]["restraint"]["anchor_atom_index"]
    d0 = spec["ions"][0]["restraint"]["reference_displacement_nm"]

    pos = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
    before, _ = _energy_and_forces(restraint_system, pos * unit.nanometer)
    assert before == pytest.approx(0.0, abs=1e-9)

    # MC 膜 barostat 式的 z 方向缩放：所有坐标乘 1.05（盒也一起变）。
    scaled = pos.copy()
    scaled[:, 2] *= 1.05
    got, _ = _energy_and_forces(restraint_system, scaled * unit.nanometer)
    assert got == pytest.approx(0.0, abs=1e-6), (
        "缩放后 flat-bottom 井里出现了能量 —— 参考点没有跟着体系走（MEM-00d 回归）"
    )

    # 对照：同样的缩放下，退役的绝对参考点形式会有多大的伪能量。
    retired_shift_nm = abs(pos[ion, 2] * 0.05)
    retired_energy = 0.5 * 25.0 * retired_shift_nm**2
    assert retired_shift_nm > 0.3 and retired_energy > 1.0, (
        "这条对照失去意义了（缩放太小），请调整构造使旧形式确实会产生可观的伪能量"
    )


def test_restraint_energy_is_identical_at_every_lambda():
    """§2.3 / §6.4：restraint 势逐 λ 完全相同，且待在自己的 force group 里。"""
    system, _topology, positions, _box, _spec, _info = _configured()
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(positions)
    context.setPeriodicBoxVectors(*(BOX_NM * unit.nanometer))
    energies = []
    for lam in (1.0, 0.5, 0.0):
        context.setParameter("lam_coul", lam)
        energies.append(
            context.getState(
                getEnergy=True,
                groups={core.CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP},
            )
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
    del context, integrator
    assert energies[0] == pytest.approx(energies[1], abs=1e-12)
    assert energies[0] == pytest.approx(energies[2], abs=1e-12)


def test_placement_that_cannot_guarantee_the_runtime_threshold_fails_closed():
    """§13.1：flat-bottom 井必须**构造性地**保证 co-ion 全程离配体 ≥ 1.2 nm。"""
    with pytest.raises(ValueError, match="§13.1"):
        _freeze_spec(dummy_z_nm=LIGAND_Z_NM + 1.5)  # 1.5 nm：减掉 r₀ 与软墙余量后不够

    # 余量刚好够的摆放必须放行，并把可保证距离如实记下来。
    # 2.3 nm 是这个构造的临界值附近：2.3 − r₀(0.5) − 软墙(0.316) − 配体外缘(0.205)
    # = 1.28 nm ≥ 1.2 nm。摆到 2.2 nm 就只有 1.18 nm，会（正确地）被拦下。
    _s, _t, _p, _b, spec = _freeze_spec(dummy_z_nm=LIGAND_Z_NM + 2.3)
    placement = spec["ions"][0]["placement_diagnostics"]
    assert placement["satisfies_runtime_threshold"]
    assert placement["guaranteed_min_ligand_distance_nm"] >= (
        core.COION_LIGAND_MIN_IMAGE_RUNTIME_NM
    )
    assert placement["wall_margin_nm"] == pytest.approx(
        core.co_alchemical_ion_restraint_wall_margin_nm(), rel=1e-12
    )


# ---------------------------------------------------------------------------
# 5. fail-closed：不许把别的东西当 co-ion，不许跨路线复用
# ---------------------------------------------------------------------------


def test_a_charged_physical_ion_cannot_be_used_as_the_coion():
    """拿一个已经带电的物理离子当 co-ion ⟹ λ=1 端总电荷不再是物理体系的，必须 raise。"""
    with pytest.raises(RuntimeError, match="reserved"):
        # dummy 带 +1：于是体系里一个"电荷为 0 的离子"都没有 → 认不出 reserved co-ion。
        _freeze_spec(dummy_charge_e=1.0)

    with pytest.raises(ValueError, match="中性但保留 LJ"):
        core.co_alchemical_charge_offset_plan(
            charge_treatment=core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
            ligand_net_charge_e=1,
            ligand_charges_e={0: 0.5, 1: 0.3, 2: 0.2},
            co_ion_physical_charges_e={3: 1.0},
        )


def test_wrong_number_of_reserved_dummies_fails_closed():
    """预留 0 个或 2 个都不行：多了就得靠坐标去挑，MEM-00c 的漂移风险原地复活。"""
    with pytest.raises(RuntimeError, match="reserved"):
        _freeze_spec(n_dummies=2)


def test_multivalent_ligand_needs_one_monovalent_coion_per_unit_charge():
    """§2.2：|q_L| > 1 时必须用 |q_L| 个单价 co-ion，每个只接过一个单位电荷。"""
    with pytest.raises(RuntimeError, match="reserved"):
        _freeze_spec(ligand_net_charge_e=2, n_dummies=1)

    _s, _t, _p, _b, spec = _freeze_spec(ligand_net_charge_e=2, n_dummies=2)
    assert len(spec["ions"]) == 2
    for ion in spec["ions"]:
        assert ion["charge_at_lambda1_e"] == pytest.approx(0.0)
        assert ion["charge_at_lambda0_e"] == pytest.approx(1.0)
    # B2 的声明校验（每粒子 ≤ 1 单位电荷、总量配平）必须通过。
    core._validate_co_alchemical_ion_spec(spec["ions"], 2)


def test_missing_spec_fails_closed():
    system, topology, _positions, _box = _build_charge_transfer_system()
    with pytest.raises(RuntimeError, match="MEM-00c"):
        ie.configure_charge_transfer_decharging(
            system, LIGAND_INDICES, topology, co_alchemical_ion_spec=None
        )


def test_the_two_routes_cannot_consume_each_others_spec():
    """路线与 spec 必须对得上：端点电荷相反，混用等于跑了另一个哈密顿量。"""
    system, topology, positions, box, spec = _freeze_spec()
    with pytest.raises(RuntimeError, match="charge-transfer"):
        ie.configure_coalchemical_neutral_decharging(
            system,
            LIGAND_INDICES,
            topology,
            positions,
            box_vectors=box,
            co_alchemical_ion_spec=spec,
        )

    # 反方向：co-annihilation 的 spec 喂给 charge-transfer builder。
    system2, topology2, _p2, _b2, spec2 = _freeze_spec()
    spec2["charge_treatment"] = core.CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL
    spec2["fingerprint"] = core.co_alchemical_ion_identity_fingerprint(spec2)
    with pytest.raises(ValueError, match="不可跨路线复用"):
        ie.configure_charge_transfer_decharging(
            system2, LIGAND_INDICES, topology2, co_alchemical_ion_spec=spec2
        )


def test_dispatch_is_driven_by_the_frozen_spec_not_by_a_second_guess():
    """哪条路线由 spec 里记录的 `charge_treatment` 决定，且只有一个分派点。"""
    _system, _topology, _positions, _box, _spec, info = _configured()
    assert info["mode"] == "co_alchemical_charge_transfer"
    assert info["charge_treatment"] == core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER

    src = (ROOT / "ibs_engine.py").read_text(encoding="utf-8")
    assert src.count("configure_charge_transfer_decharging(") == 2, (
        "`configure_charge_transfer_decharging(` 必须只有 def 与"
        "`configure_pme_ligand_charge_offsets` 里那一处分派；多一处就是多一条并行路径"
    )


def test_solvent_leg_builder_inserts_reserved_dummy_for_charge_transfer():
    """B4 已落地（2026-08-05）：溶剂腿不再对 charge-transfer 无条件 fail closed。

    真正的插入逻辑（摘最远水、造 dummy、清零电荷、多 dummy 不错位、水不够时仍
    fail closed）由 `tests/test_solvent_leg_coion_builder.py` 逐项覆盖——那些测试
    直接调用新增的 `runabfe._insert_reserved_coalchemical_ion_dummies`，不需要
    真实 GROMACS 输入。这里只钉住"入口不再无条件拒绝"这个契约，防止有人把 B4 的
    `NotImplementedError` 悄悄加回去。
    """
    runabfe = pytest.importorskip("runabfe")
    assert core.CHARGE_TRANSFER_SOLVENT_LEG_IMPLEMENTED is True
    src = (ROOT / "runabfe.py").read_text(encoding="utf-8")
    assert "_insert_reserved_coalchemical_ion_dummies" in src
    assert "溶剂腿 builder 尚未实现" not in src


def test_charge_treatment_payload_reports_the_closed_cycle():
    """声明 charge-transfer 时，解析结果必须如实说"循环闭得上"（B4 已落地）。"""
    ion = {
        "atom_index": 3,
        "residue_index": 1,
        "residue_name": "NA",
        "element": "Na",
        "charge_at_lambda1_e": 0.0,
        "charge_at_lambda0_e": 1.0,
        "sigma_nm": 0.2439,
        "epsilon_kj_mol": 0.3658,
        "mass_amu": 22.99,
        "restraint": {"form": core.CO_ALCHEMICAL_ION_RESTRAINT_FORM_FLAT_BOTTOM},
    }
    payload = core.resolve_charge_treatment(
        core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        ligand_net_charge_e=1.0,
        co_alchemical_ion=ion,
    )
    assert payload["charging_hamiltonian_implemented"] is True
    assert payload["solvent_leg_builder_implemented"] is True
    assert payload["closes_thermodynamic_cycle"] is True
    assert payload["apbs_applicable"] is False
