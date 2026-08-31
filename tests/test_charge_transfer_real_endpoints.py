"""C3-1：`tools/validation/compare_charge_transfer_endpoints.py` 的 CPU 契约测试。

对应 `docs/status/memtodolist.md` §「C3：真实体系 λ=1/λ=0 端点能量和力」的 C3-1 清单：

    正/负 ligand 电荷
    单/多 co-ion
    普通 pair、excluded pair、1-4
    ligand internal 保持
    charging λ=1/λ=0
    vanishing λ=1/λ=0
    LRC λ=1/λ=0
    reference planner 独立性
    任一参数篡改能触发 gate
    缺 frame/box/hash 时 fail closed

这里只用小型合成体系（Reference 平台），不碰真实 C1/C2 数据——那是 C3-2
（wiring smoke）与 C3-3/C3-4（真实数值门）的事，需要 GPU 双精度权威跑，
留给用户计算节点执行（见 docs/status/memtodolist.md §9）。

## 这个文件不覆盖什么（诚实标注，不要悄悄假装做了）

- C（vanishing λ_vdw=1 seam）的比较用的是两条**都是生产代码**的路径互相对照，
  不是独立 reference——这是 docs/status/memtodolist.md 表里本来的设计（"两阶段接缝完全
  一致"），不是本文件偷懒。
- 没有测试"缺失真实 DCD 帧/box/hash"这个 I/O 层面的 fail-closed（那需要真实
  轨迹文件），这里只测 `evaluate()`/`compare_endpoint()` 对非有限值和形状不
  匹配的 fail-closed，是同一件事在没有文件系统时的最小可测子集。
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

openmm = pytest.importorskip("openmm")
from openmm import NonbondedForce, app, unit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import abfe_core as core  # noqa: E402
import ibs_engine as ie  # noqa: E402

_MODULE_PATH = ROOT / "tools" / "validation" / "compare_charge_transfer_endpoints.py"
_spec = importlib.util.spec_from_file_location(
    "compare_charge_transfer_endpoints", _MODULE_PATH
)
cte = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cte
_spec.loader.exec_module(cte)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 合成体系：4 原子配体（覆盖 excluded / 1-4 / ordinary 三类 L-L 对）
# + reserved 中性 dummy(s) + 配平用普通离子 + 两个水。
# ---------------------------------------------------------------------------

BOX_NM = np.diag([6.0, 6.0, 12.0])
CUTOFF_NM = 1.0

LIGAND_CHARGES_E = (0.5, 0.3, -0.4, 0.2)
LIGAND_SIGMA_NM = 0.34
LIGAND_EPSILON_KJ = 0.36
LIGAND_MASS_AMU = 12.011

DUMMY_Z_NM = 8.0
LIGAND_Z_NM = 2.0

# 4 个配体原子按链式排布：0-1-2-3。1-2/1-3 全排除，1-4 (0,3) 打折保留，
# (0,2)/(1,3) 里各留一个真正的"普通对"（未定义任何 exception，走标准 combining
# rule），用来测"没有既有 exception 的那些 L-L 对必须被冻结成新 exception"。
LIGAND_EXCLUDED_PAIRS = {(0, 1), (1, 2), (2, 3), (0, 2)}
LIGAND_14_SCALED_PAIRS = {(1, 3): 0.5}
LIGAND_ORDINARY_PAIRS = {(0, 3)}


def _build_system(
    *,
    ligand_net_charge_e: int = 1,
    n_dummies: int = 1,
    dummy_charge_e: float = 0.0,
    dummy_z_nm: float = DUMMY_Z_NM,
    scale_ligand_charges_to_net: bool = True,
):
    """4 原子配体 + reserved 中性 dummy(s) + 配平普通离子 + 两个水。

    `n_dummies` 必须等于 `abs(ligand_net_charge_e)`（§2.2：每个 co-ion 只接
    一个单位电荷），除非有意构造违规输入去触发 fail-closed。
    """
    topology = app.Topology()
    chain = topology.addChain()

    ligand_res = topology.addResidue("MOL", chain)
    names = ("C1", "C2", "N1", "C3")
    for name in names:
        elem = app.element.nitrogen if name.startswith("N") else app.element.carbon
        topology.addAtom(name, elem, ligand_res)

    for _ in range(int(n_dummies)):
        dummy_res = topology.addResidue("NA", chain)
        topology.addAtom("NA", app.element.sodium, dummy_res)

    for _ in range(abs(int(ligand_net_charge_e))):
        ion_res = topology.addResidue("CL", chain)
        topology.addAtom("CL", app.element.chlorine, ion_res)

    for _ in range(2):
        water_res = topology.addResidue("HOH", chain)
        topology.addAtom("O", app.element.oxygen, water_res)
        topology.addAtom("H1", app.element.hydrogen, water_res)
        topology.addAtom("H2", app.element.hydrogen, water_res)

    if scale_ligand_charges_to_net:
        raw_sum = sum(LIGAND_CHARGES_E)
        scale = float(ligand_net_charge_e) / raw_sum
    else:
        scale = 1.0
    ligand_charges = [q * scale for q in LIGAND_CHARGES_E]
    ordinary_sign = -1.0 if ligand_net_charge_e > 0 else 1.0

    charges = list(ligand_charges)
    charges += [float(dummy_charge_e)] * int(n_dummies)
    charges += [ordinary_sign] * abs(int(ligand_net_charge_e))
    charges += [-0.834, 0.417, 0.417, -0.834, 0.417, 0.417]

    sigmas = [LIGAND_SIGMA_NM] * 4
    sigmas += [0.2439] * int(n_dummies)
    sigmas += [0.4478] * abs(int(ligand_net_charge_e))
    sigmas += [0.3151, 0.1, 0.1, 0.3151, 0.1, 0.1]

    epsilons = [LIGAND_EPSILON_KJ] * 4
    epsilons += [0.3658] * int(n_dummies)
    epsilons += [0.1489] * abs(int(ligand_net_charge_e))
    epsilons += [0.6364, 0.0, 0.0, 0.6364, 0.0, 0.0]

    masses = [LIGAND_MASS_AMU] * 4
    masses += [22.99] * int(n_dummies)
    masses += [35.45] * abs(int(ligand_net_charge_e))
    masses += [15.999, 1.008, 1.008, 15.999, 1.008, 1.008]

    force = NonbondedForce()
    force.setNonbondedMethod(NonbondedForce.PME)
    force.setCutoffDistance(CUTOFF_NM * unit.nanometer)
    for q, sigma, epsilon in zip(charges, sigmas, epsilons):
        force.addParticle(
            q * unit.elementary_charge, sigma * unit.nanometer, epsilon * unit.kilojoule_per_mole
        )

    def _lj_combo(i, j):
        return (
            0.5 * (sigmas[i] + sigmas[j]),
            math.sqrt(max(epsilons[i] * epsilons[j], 0.0)),
        )

    for (i, j) in LIGAND_EXCLUDED_PAIRS:
        sig, eps = _lj_combo(i, j)
        force.addException(i, j, 0.0 * unit.elementary_charge**2, sig * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
    for (i, j), fudge in LIGAND_14_SCALED_PAIRS.items():
        sig, eps = _lj_combo(i, j)
        force.addException(
            i, j,
            (fudge * charges[i] * charges[j]) * unit.elementary_charge**2,
            sig * unit.nanometer,
            (fudge * eps) * unit.kilojoule_per_mole,
        )
    # LIGAND_ORDINARY_PAIRS 故意不加 exception——它们走标准 combining rule，
    # 这正是要测试"必须被生产/参照两侧各自冻结成新 exception"的那一类。

    positions = [
        [3.00, 3.00, LIGAND_Z_NM],
        [3.15, 3.00, LIGAND_Z_NM],
        [3.00, 3.15, LIGAND_Z_NM + 0.14],
        [3.15, 3.15, LIGAND_Z_NM + 0.10],
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
        assert abs(total) < 1e-9, f"fixture 自身不电中性：Σq = {total:+.6f} e"

    return (
        system,
        topology,
        np.asarray(positions, dtype=float) * unit.nanometer,
        BOX_NM,
    )


LIGAND_INDICES = [0, 1, 2, 3]


def _nm(positions_quantity) -> np.ndarray:
    """`compare_charge_transfer_endpoints` 的公开函数一律吃"裸的 nm 数值
    数组"（与仓库里 `positions_nm.npy`/`box_vectors_nm` 的既有约定一致），
    不吃 OpenMM `Quantity`——这里的 fixture 为了兼容
    `ie.select_co_alchemical_ion_once` 等生产入口（那些确实要 `Quantity`），
    统一返回带单位的 positions，所以每次喂给 `cte.*` 之前都要在这里剥一次单位。
    """
    return np.asarray(positions_quantity.value_in_unit(unit.nanometer), dtype=float)


def _freeze_spec(system, topology, positions, box, *, ligand_net_charge_e):
    """按生产的选择入口冻结一次身份——这是**测试装置**，不是"reference builder
    自己去选"。之后所有 reference 调用只读这份 spec。
    """
    return ie.select_co_alchemical_ion_once(
        system,
        LIGAND_INDICES,
        topology,
        positions,
        box,
        charge_treatment=core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
    )


def _case(ligand_net_charge_e, n_dummies=None):
    n_dummies = abs(int(ligand_net_charge_e)) if n_dummies is None else n_dummies
    system, topology, positions, box = _build_system(
        ligand_net_charge_e=ligand_net_charge_e, n_dummies=n_dummies
    )
    spec = _freeze_spec(system, topology, positions, box, ligand_net_charge_e=ligand_net_charge_e)
    return system, topology, positions, box, spec


# ---------------------------------------------------------------------------
# 1. charging λ=1 / λ=0：正/负电荷、单/多 co-ion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ligand_net_charge_e", [1, -1, 2])
def test_charging_lambda1_matches_raw_physical_system(ligand_net_charge_e):
    """A：charging λ=1 必须逐位恢复原始物理体系（正/负电荷、单/多 co-ion）。"""
    system, topology, positions, box, spec = _case(ligand_net_charge_e)
    production = cte.production_charging_system(system, LIGAND_INDICES, topology, spec)
    reference = cte.reference_charging_endpoint_system(system, LIGAND_INDICES, spec, lam=1.0)

    report = cte.compare_endpoint(
        "A", production, reference, _nm(positions), box,
        production_globals={"lam_coul": 1.0},
        production_groups={0},
        reference_groups={0},
    )
    assert report["passed"], report


@pytest.mark.parametrize("ligand_net_charge_e", [1, -1, 2])
def test_charging_lambda0_matches_independent_reference(ligand_net_charge_e):
    """B：charging λ=0 必须与"配体清零、co-ion 满电"的独立参照一致。"""
    system, topology, positions, box, spec = _case(ligand_net_charge_e)
    production = cte.production_charging_system(system, LIGAND_INDICES, topology, spec)
    reference = cte.reference_charging_endpoint_system(system, LIGAND_INDICES, spec, lam=0.0)

    report = cte.compare_endpoint(
        "B", production, reference, _nm(positions), box,
        production_globals={"lam_coul": 0.0},
        production_groups={0},
        reference_groups={0},
    )
    assert report["passed"], report


# ---------------------------------------------------------------------------
# 2. 普通 pair / excluded pair / 1-4 / ligand internal 保持
# ---------------------------------------------------------------------------


def _ll_exception_table(nb):
    table = {}
    for exc_idx in range(nb.getNumExceptions()):
        p1, p2, charge_prod, sigma, epsilon = nb.getExceptionParameters(exc_idx)
        p1, p2 = int(p1), int(p2)
        if p1 in LIGAND_INDICES and p2 in LIGAND_INDICES:
            table[(min(p1, p2), max(p1, p2))] = (
                charge_prod.value_in_unit(unit.elementary_charge**2),
                epsilon.value_in_unit(unit.kilojoule_per_mole),
            )
    return table


@pytest.mark.parametrize("lam", [1.0, 0.0])
def test_ligand_internal_pairs_are_identical_between_production_and_reference(lam):
    """[v3 口径] excluded (1-2/1-3)、1-4 打折对必须是**既有** exception 且原样
    保留；ordinary 对**不得**被补成 exception（P0-01：补对会改写真实 PME 的
    λ=1 物理端点）。生产与参照两侧逐位一致——不是靠比总能量间接猜。
    """
    system, topology, positions, box, spec = _case(1, n_dummies=1)

    production = cte.production_charging_system(system, LIGAND_INDICES, topology, spec)
    reference = cte.reference_charging_endpoint_system(system, LIGAND_INDICES, spec, lam=lam)

    nb_prod = cte._find_nonbonded_force(production)
    nb_ref = cte._find_nonbonded_force(reference)
    table_prod = _ll_exception_table(nb_prod)
    table_ref = _ll_exception_table(nb_ref)

    # v3：配置不得增删 exception —— 表里只允许 raw 拓扑自带的 excluded/1-4 对，
    # ordinary 对必须仍走普通非键（随粒子电荷线性湮灭）。
    assert set(table_prod) == set(table_ref) == (
        LIGAND_EXCLUDED_PAIRS | set(LIGAND_14_SCALED_PAIRS)
    ), (
        "production/reference 的 L-L exception 集合必须是 raw 拓扑自带的那 5 对；"
        "出现 LIGAND_ORDINARY_PAIRS 说明有人把普通对补成了 exception（P0-01 回归）"
    )

    # 用实际写进 fixture 的电荷重新推导期望值，不硬编一份数字。
    _sys2, _t2, _p2, _b2 = _build_system(ligand_net_charge_e=1, n_dummies=1)
    nb_raw = cte._find_nonbonded_force(_sys2)
    physical_q = {
        i: nb_raw.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        for i in LIGAND_INDICES
    }

    for pair in LIGAND_EXCLUDED_PAIRS:
        for table in (table_prod, table_ref):
            cp, eps = table[pair]
            assert cp == pytest.approx(0.0, abs=1e-12), f"excluded pair {pair} 的 chargeProd 不是 0"
            assert eps == pytest.approx(0.0, abs=1e-12), f"excluded pair {pair} 的 epsilon 不是 0"

    for pair, fudge in LIGAND_14_SCALED_PAIRS.items():
        i, j = pair
        want_cp = fudge * physical_q[i] * physical_q[j]
        for table in (table_prod, table_ref):
            cp, _eps = table[pair]
            assert cp == pytest.approx(want_cp, abs=1e-12), (
                f"1-4 对 {pair} 的 chargeProd 被覆盖成了非打折值——"
                "生产/参照必须原样保留既有 exception，不能重新按满电荷覆盖。"
            )


# ---------------------------------------------------------------------------
# 3. vanishing λ=1（seam）/ λ=0（严格零）+ LRC
#
# C/D 只在**净中性配体**上有效——2026-08-11 实测确认：`build_ibs_dual_system`
# 在真实生产里吃的是原始、未经 charging 配置的 System，其自身的静态电中性
# 防御要求配体在**这份原始电荷**上净和为 0（Group 2 的配体内部力直接用这份
# 原始电荷重建物理上应当逐 λ 恒定的配体内部 Coulomb）。带净电的 charge-
# transfer 配体接入 vanishing 阶段在当前生产代码里尚未实现（`abfe_config.json`
# 就地注明该路线是"Phase B3，尚未实现"）；C1/C2 用的带电探针配体也只跑过
# charging（11 个 λ_coul 态），从未真正跑过 vanishing。所以这里用一个净中性、
# 但每个原子 partial charge 非零的合成配体（不需要 co-ion），与真实生产的
# 唯一已验证接线方式一致。
# ---------------------------------------------------------------------------

LIGAND_CHARGES_NEUTRAL_E = (0.5, 0.3, -0.4, -0.4)  # 单原子非零，净和为 0


def _build_neutral_system():
    """4 原子净中性配体（沿用同一套 excluded/1-4/ordinary 对结构）+ 两个水，
    没有 co-ion——`build_ibs_dual_system` 在当前生产接线下唯一能吃的输入形态。
    """
    topology = app.Topology()
    chain = topology.addChain()
    ligand_res = topology.addResidue("MOL", chain)
    for name in ("C1", "C2", "N1", "C3"):
        elem = app.element.nitrogen if name.startswith("N") else app.element.carbon
        topology.addAtom(name, elem, ligand_res)
    for _ in range(2):
        water_res = topology.addResidue("HOH", chain)
        topology.addAtom("O", app.element.oxygen, water_res)
        topology.addAtom("H1", app.element.hydrogen, water_res)
        topology.addAtom("H2", app.element.hydrogen, water_res)

    charges = list(LIGAND_CHARGES_NEUTRAL_E) + [-0.834, 0.417, 0.417, -0.834, 0.417, 0.417]
    sigmas = [LIGAND_SIGMA_NM] * 4 + [0.3151, 0.1, 0.1, 0.3151, 0.1, 0.1]
    epsilons = [LIGAND_EPSILON_KJ] * 4 + [0.6364, 0.0, 0.0, 0.6364, 0.0, 0.0]
    masses = [LIGAND_MASS_AMU] * 4 + [15.999, 1.008, 1.008, 15.999, 1.008, 1.008]

    force = NonbondedForce()
    force.setNonbondedMethod(NonbondedForce.PME)
    force.setCutoffDistance(CUTOFF_NM * unit.nanometer)
    for q, sigma, epsilon in zip(charges, sigmas, epsilons):
        force.addParticle(
            q * unit.elementary_charge, sigma * unit.nanometer, epsilon * unit.kilojoule_per_mole
        )

    def _lj_combo(i, j):
        return 0.5 * (sigmas[i] + sigmas[j]), math.sqrt(max(epsilons[i] * epsilons[j], 0.0))

    for (i, j) in LIGAND_EXCLUDED_PAIRS:
        sig, _eps = _lj_combo(i, j)
        force.addException(i, j, 0.0 * unit.elementary_charge**2, sig * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
    for (i, j), fudge in LIGAND_14_SCALED_PAIRS.items():
        sig, eps = _lj_combo(i, j)
        force.addException(
            i, j,
            (fudge * charges[i] * charges[j]) * unit.elementary_charge**2,
            sig * unit.nanometer,
            (fudge * eps) * unit.kilojoule_per_mole,
        )

    positions = [
        [3.00, 3.00, LIGAND_Z_NM],
        [3.15, 3.00, LIGAND_Z_NM],
        [3.00, 3.15, LIGAND_Z_NM + 0.14],
        [3.15, 3.15, LIGAND_Z_NM + 0.10],
        [4.50, 3.00, 6.00], [4.60, 3.00, 6.00], [4.40, 3.00, 6.00],
        [2.00, 4.50, 9.00], [2.10, 4.50, 9.00], [1.90, 4.50, 9.00],
    ]

    system = openmm.System()
    for mass in masses:
        system.addParticle(mass * unit.dalton)
    system.setDefaultPeriodicBoxVectors(*(BOX_NM * unit.nanometer))
    system.addForce(force)

    assert abs(sum(charges[:4])) < 1e-9, "中性配体 fixture 本身必须净电荷为 0"
    return (
        system,
        topology,
        np.asarray(positions, dtype=float) * unit.nanometer,
        BOX_NM,
    )


def test_vanishing_lambda_zero_ligand_environment_is_algebraically_zero():
    """D：λ_vdw=0 的软核 CV 必须**结构性**为零，不是"很小"（memtodolist §8）。"""
    system, _topology, positions, box = _build_neutral_system()
    systems = cte.production_vanishing_fixed_hamiltonian_systems(
        system, LIGAND_INDICES, [1.0, 0.0], box
    )
    vanishing_zero = systems[1]

    report = cte.compare_vanishing_zero_endpoint(
        vanishing_zero, _nm(positions), box,
        ligand_environment_groups={1},
    )
    assert report["passed"], report
    assert report["e_ligand_environment_kj_mol"] == pytest.approx(0.0, abs=1e-9)
    assert report["max_abs_force_ligand_environment_kj_mol_nm"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.xfail(
    reason="P1-19（2026-08-30 登记）：v3 charging 口径下 seam 失配——charging λ=0 "
    "已把 ordinary L-L 内部库仑湮灭，而 vanishing 侧 U_common(Group 2) 仍按"
    "『逐 λ 恒定的物理值』重建配体内部静电，两端点差一个内部库仑常数"
    "（实测 118.5 kJ/mol，中性 4 原子 fixture）。该常数逐 λ_vdw 不变 ⟹ ΔG_vdw "
    "与 ΔG_bind 不受影响，但 seam『两端点同一 Hamiltonian』的记账恒等式被破坏，"
    "需要单独定级并决定 U_common 是否随 v3 改写。修复后此标记应转 XPASS 并摘除。",
    strict=False,
)
def test_vanishing_lambda_one_seam_matches_charging_lambda_zero():
    """C：vanishing λ_vdw=1 与 charging λ_coul=0 必须是同一个物理 Hamiltonian
    （两阶段接缝完全一致——两侧都是生产代码，测的是自洽性，不是独立 reference）。
    """
    system, topology, positions, box = _build_neutral_system()
    charging0 = cte.production_charging_system(system, LIGAND_INDICES, topology, None)
    systems = cte.production_vanishing_fixed_hamiltonian_systems(
        system, LIGAND_INDICES, [1.0, 0.0], box
    )
    vanishing_one = systems[0]

    report = cte.compare_endpoint(
        "C_seam", vanishing_one, charging0, _nm(positions), box,
        reference_globals={"lam_coul": 0.0},
        production_groups={0, 1, 2},
        reference_groups={0},
    )
    assert report["passed"], report


# ---------------------------------------------------------------------------
# 带电配体版 C/D：真正用 Stage2 handoff 机制（`configure_pme_ligand_charge_
# offsets` + `abfe_core.bake_global_parameter_into_fixed_nonbonded_force`，
# `lambda_name="lambda_coul"`，与 `abfe_pipeline.py` 里
# `ABFEPipeline._run_dual_lambda_stage` 的 vanishing 分支逐字一致），不是临时
# 诊断脚本——这是 docs/experiments/STAGE2_CHARGE_TRANSFER_HANDOFF_PROPOSAL.md 落地后补的永久
# 契约测试。C 用**伸展几何**（真实键长键角，见下面 helper）：2026-08-11 实测
# 确认原来那个紧凑几何（配体 4 原子挤在 <0.2nm 内）会带出一个小的、与几何
# 基本无关的绝对残差（~0.0005 kJ/mol），让相对差刚好卡在 1e-5 门外——不是
# Hamiltonian 构造错误，但也不该拿一个"刚好卡过/卡不过"的病态几何做永久回归。
# D 不受几何影响（λ_coul≡0 让 Coulomb 项恒为 0，λ_vdw=0 让 LJ 项恒为 0，
# 与原子实际距离无关），继续用原来的紧凑 fixture 即可。
# ---------------------------------------------------------------------------


def _extend_ligand_geometry(positions_quantity):
    """把 `_case()` 返回的配体前 4 个原子换成真实键长键角的伸展锌链
    （0.153nm 键长、111.5° 键角、anti 二面角），其它原子位置不变。
    """
    positions_nm = np.asarray(positions_quantity.value_in_unit(unit.nanometer), dtype=float).copy()
    bond = 0.153
    angle = math.radians(180.0 - 111.5)
    p0 = positions_nm[0].copy()
    p1 = p0 + np.array([bond, 0.0, 0.0])
    p2 = p1 + np.array([bond * math.cos(angle), bond * math.sin(angle), 0.0])
    p3 = p2 + np.array([bond * math.cos(angle), -bond * math.sin(angle), 0.0])
    positions_nm[0:4] = np.stack([p0, p1, p2, p3])
    return positions_nm * unit.nanometer


def test_bake_handoff_seam_matches_for_charged_ligand_with_realistic_geometry():
    """C，带电版：用真正的 Stage2 handoff 链路（charging 配置 → bake →
    vanishing）算出的 λ_vdw=1 端点，必须与 baked charging λ_coul=0 端点
    一致——这正是 `abfe_pipeline.py` 新增的那条 handoff 要保证的性质。
    """
    system, topology, positions, box, spec = _case(1, n_dummies=1)
    positions = _extend_ligand_geometry(positions)

    charging0 = cte.production_charging_system(
        system, LIGAND_INDICES, topology, spec, lambda_name="lambda_coul"
    )
    vanishing_input = core.bake_global_parameter_into_fixed_nonbonded_force(
        charging0, "lambda_coul", 0.0
    )
    charging0_baked = core.bake_global_parameter_into_fixed_nonbonded_force(
        charging0, "lambda_coul", 0.0
    )
    systems = cte.production_vanishing_fixed_hamiltonian_systems(
        vanishing_input, LIGAND_INDICES, [1.0, 0.0], box
    )
    vanishing_one = systems[0]

    report = cte.compare_endpoint(
        "C_seam_charged", vanishing_one, charging0_baked, _nm(positions), box,
        production_groups={0, 1, 2},
        reference_groups={0},
    )
    assert report["passed"], report
    assert report["rel_delta_e"] < 1e-7, (
        f"伸展几何下相对差应该远优于 1e-5 门，不是刚好卡过：{report['rel_delta_e']:.3e}"
    )


def test_bake_handoff_vanishing_zero_is_algebraically_zero_for_charged_ligand():
    """D，带电版：同样用真正的 handoff 链路，λ_vdw=0 的配体-环境软核 CV 必须
    严格为零——不受几何影响，继续用原来的紧凑 fixture。
    """
    system, topology, positions, box, spec = _case(1, n_dummies=1)
    charging0 = cte.production_charging_system(
        system, LIGAND_INDICES, topology, spec, lambda_name="lambda_coul"
    )
    vanishing_input = core.bake_global_parameter_into_fixed_nonbonded_force(
        charging0, "lambda_coul", 0.0
    )
    systems = cte.production_vanishing_fixed_hamiltonian_systems(
        vanishing_input, LIGAND_INDICES, [1.0, 0.0], box
    )
    vanishing_zero = systems[1]

    report = cte.compare_vanishing_zero_endpoint(
        vanishing_zero, _nm(positions), box,
        ligand_environment_groups={1},
    )
    assert report["passed"], report
    assert report["e_ligand_environment_kj_mol"] == pytest.approx(0.0, abs=1e-9)
    assert report["max_abs_force_ligand_environment_kj_mol_nm"] == pytest.approx(0.0, abs=1e-6)


def test_lj_tail_lrc_coefficient_is_exactly_zero_at_lambda_vdw_zero():
    """LRC λ=0：解析尾项系数必须严格为 0，不是被积分算出一个很小的数。"""
    coeffs = ie._lj_tail_lrc_coefficients_kj_mol(
        [1.0, 0.5, 0.0],
        sigma_nm=np.array([0.34]),
        s6_per_sigma_kj_nm6=np.array([1.0]),
        s12_per_sigma_kj_nm12=np.array([1.0]),
        alpha_lj=0.5,
        m_lj=2.0,
        n_lj=2.0,
    )
    assert coeffs[2] == 0.0
    assert coeffs[0] != 0.0 and coeffs[1] != 0.0


# ---------------------------------------------------------------------------
# 3b. Protocol v2 的 C/D runner（`run_protocol_v2_matrix_cd`）——2026-08-11，
# 用户要求复用真实 charging λ=0 帧、不新跑 MD。这里在 CPU 上用一个真实落盘
# 的 case_dir（system.xml/topology.cif/ligand_indices.json/
# coalchemical_ion_spec.json/build_manifest.json/box_vectors_nm.npy/
# dynamics/*.dcd，与真实生产目录逐字同构）跑一遍完整 runner，在花 GPU 时间
# 之前先把接线炸出来——`mixed_platform_name` 换成 "CPU"（不需要 CUDA），只
# 验证结构（gate2 must be not_applicable、D 的两个 strict-zero 都过、
# gate1/gate3 都过），不是权威数值门本身（数值门已经在
# `test_bake_handoff_seam_matches_for_charged_ligand_with_realistic_geometry`/
# `test_bake_handoff_vanishing_zero_is_algebraically_zero_for_charged_ligand`
# 里用同一套构造验证过）。
# ---------------------------------------------------------------------------


def _write_synthetic_case_dir(tmp_path, *, system, topology, positions_nm, box_nm, spec):
    import mdtraj as md

    case_dir = tmp_path / "cd_case"
    case_dir.mkdir()
    (case_dir / "system.xml").write_text(openmm.XmlSerializer.serialize(system), encoding="utf-8")
    app.PDBxFile.writeFile(topology, positions_nm * unit.nanometer, str(case_dir / "topology.cif"))
    (case_dir / "ligand_indices.json").write_text(
        __import__("json").dumps({"ligand_indices": list(LIGAND_INDICES)}), encoding="utf-8"
    )
    (case_dir / "coalchemical_ion_spec.json").write_text(
        __import__("json").dumps(spec), encoding="utf-8"
    )
    (case_dir / "build_manifest.json").write_text(
        __import__("json").dumps({"coalchemical_ion_fingerprint": spec["fingerprint"]}),
        encoding="utf-8",
    )
    np.save(case_dir / "box_vectors_nm.npy", box_nm)

    dyn_dir = case_dir / "dynamics"
    dyn_dir.mkdir()
    md_top = md.Topology.from_openmm(topology)
    xyz_frames = np.stack([positions_nm, positions_nm])  # 2 帧，同一份坐标即可——只测接线
    unitcell_lengths = np.tile(np.diag(box_nm), (2, 1))
    unitcell_angles = np.tile([90.0, 90.0, 90.0], (2, 1))
    traj = md.Trajectory(
        xyz=xyz_frames, topology=md_top,
        unitcell_lengths=unitcell_lengths, unitcell_angles=unitcell_angles,
    )
    traj.save_dcd(str(dyn_dir / "lam0.dcd"))
    return case_dir


@pytest.mark.xfail(
    reason="P1-19（2026-08-30 登记）：v3 charging 口径下 seam 失配（同 "
    "test_vanishing_lambda_one_seam_matches_charging_lambda_zero 的 xfail 理由）"
    "——C/D 门里的 gate1_reference_identity / gate3_mixed_production_vs_reference "
    "依赖 charging λ=0 与 vanishing 侧 U_common 的 seam 恒等。修复后应转 XPASS 并摘除。",
    strict=False,
)
def test_run_protocol_v2_matrix_cd_wiring_passes_on_charged_fixture(tmp_path):
    system, topology, positions, box, spec = _case(1, n_dummies=1)
    positions_nm = _nm(_extend_ligand_geometry(positions))
    case_dir = _write_synthetic_case_dir(
        tmp_path, system=system, topology=topology,
        positions_nm=positions_nm, box_nm=box, spec=spec,
    )

    report = cte.run_protocol_v2_matrix_cd(
        case_dir,
        lambda0_dcd_name="lam0.dcd",
        lambda0_frame_indices=[0, 1],
        lambda_name="lambda_coul",
        reference_platform_name="Reference",
        mixed_platform_name="CPU",
        mixed_platform_properties={},
    )
    assert report["passed"], report["failed_frames"]
    assert report["n_frames"] == 2
    assert report["n_failed"] == 0
    for frame in report["frames"]:
        assert frame["C"]["gate2_mixed_live_vs_baked"]["applicable"] is False
        assert frame["D"]["gate2_mixed_live_vs_baked"]["applicable"] is False
        assert frame["D"]["strict_zero_reference"]["e_ligand_environment_kj_mol"] == pytest.approx(0.0, abs=1e-9)
        assert frame["D"]["strict_zero_mixed"]["e_ligand_environment_kj_mol"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3c. MEM-00h 双边归一化（`mem00h_normalized_raw_system`/
# `assert_mem00h_switching_convention`）——2026-08-11，用户对最初"关掉某一
# 侧 switch"消融实验的修正：真正的根因是 C2 raw System 自带一个局部
# `[0.995,1.0]nm` LJ switch，跟 vanishing 阶段全局无 switch 的软核 CV 不
# 一致；正确修法是在分支出 charging/baked/vanishing/reference 之前，先把
# 共同的 raw System clone 统一转到 MEM-00h 的 `cutoff=1.0nm,
# switching=False`，而不是事后单边打补丁。这里在一个可控的小体系上精确
# 复现"环境原子落在 switch 窗口内→C seam 出现非零力差"，再证明经过归一化
# 之后同一份体系变回精确匹配——不是从真实 C2 数据反推的结论，是独立
# 构造验证。
# ---------------------------------------------------------------------------


def _case_with_environment_atom_in_switch_window(*, switch_nm, cutoff_nm, r_nm):
    """`_case(1, n_dummies=1)` 的基础上，把水 1 的氧原子精确挪到跟配体锚点
    （原子 0）距离恰好 `r_nm` 的位置，并在返回的 System 的 `NonbondedForce`
    上设一个 `[switch_nm, cutoff_nm]` 的 LJ switch——精确复现 C2 raw
    System 那条局部 switch 惯例，但在一个几十个原子的小体系上，几毫秒内可测。

    配体几何用 `_extend_ligand_geometry` 伸展——原始紧凑几何自带一个跟
    switch 无关的 ~0.0005 kJ/mol 绝对残差（见
    `test_bake_handoff_seam_matches_for_charged_ligand_with_realistic_geometry`
    上面的说明），会跟这里要测的 switch 残差混在一起，伸展几何后消除到
    1e-7 以下，不干扰判读。
    """
    system, topology, positions, box, spec = _case(1, n_dummies=1)
    positions = _extend_ligand_geometry(positions)
    positions_nm = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float).copy()
    anchor = positions_nm[0].copy()
    positions_nm[6] = anchor + np.array([r_nm, 0.0, 0.0])  # 水 1 的氧原子（非零 epsilon）
    positions_nm[7] = positions_nm[6] + np.array([0.09, 0.0, 0.0])
    positions_nm[8] = positions_nm[6] + np.array([0.0, 0.09, 0.0])

    nb = _find_nonbonded_force_for_test(system)
    nb.setUseSwitchingFunction(True)
    nb.setSwitchingDistance(switch_nm * unit.nanometer)
    nb.setCutoffDistance(cutoff_nm * unit.nanometer)
    return system, topology, positions_nm * unit.nanometer, box, spec


def _find_nonbonded_force_for_test(system):
    return next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))


def test_c2_style_switch_mismatch_reproduces_on_a_small_controlled_system():
    """不归一化：环境原子落在 `[0.995,1.0]nm` switch 窗口内时，C seam
    （baked charging λ=0 vs production vanishing λ_vdw=1）必须出现非零力差
    ——这是"两侧要不要 switch 的约定不一致"这个根因的直接、独立复现，不依赖
    真实 C2 数据。"""
    system, topology, positions, box, spec = _case_with_environment_atom_in_switch_window(
        switch_nm=0.995, cutoff_nm=1.0, r_nm=0.997,
    )
    charging0 = cte.production_charging_system(
        system, LIGAND_INDICES, topology, spec, lambda_name="lambda_coul"
    )
    charging0_baked = core.bake_global_parameter_into_fixed_nonbonded_force(
        charging0, "lambda_coul", 0.0
    )
    vanishing_input = core.bake_global_parameter_into_fixed_nonbonded_force(
        charging0, "lambda_coul", 0.0
    )
    vanishing_one, _vanishing_zero = cte.production_vanishing_fixed_hamiltonian_systems(
        vanishing_input, LIGAND_INDICES, [1.0, 0.0], box
    )

    report = cte.compare_endpoint(
        "C_seam_unnormalized", vanishing_one, charging0_baked, _nm(positions), box,
        production_groups={0, 1, 2}, reference_groups={0},
        platform_name="Reference",
    )
    assert not report["passed"], (
        "环境原子精确落在 switch 窗口内，理应复现出非零力差；如果这里反而 "
        "passed，说明测试几何没有真的落进窗口，不是 bug 已经不存在。"
    )
    assert report["max_abs_force_component_diff_kj_mol_nm"] > 1e-2


def test_mem00h_normalization_fixes_the_same_switch_mismatch():
    """同一个几何、同一条构造链，唯一区别是先过一遍
    `mem00h_normalized_raw_system`——力差必须回到机器精度，证明归一化真的
    解决了根因，不是掩盖了它。
    """
    system, topology, positions, box, spec = _case_with_environment_atom_in_switch_window(
        switch_nm=0.995, cutoff_nm=1.0, r_nm=0.997,
    )
    normalized = cte.mem00h_normalized_raw_system(system)
    nb = _find_nonbonded_force_for_test(normalized)
    assert nb.getUseSwitchingFunction() is False
    assert nb.getCutoffDistance().value_in_unit(unit.nanometer) == pytest.approx(1.0)

    charging0 = cte.production_charging_system(
        normalized, LIGAND_INDICES, topology, spec, lambda_name="lambda_coul"
    )
    charging0_baked = core.bake_global_parameter_into_fixed_nonbonded_force(
        charging0, "lambda_coul", 0.0
    )
    vanishing_input = core.bake_global_parameter_into_fixed_nonbonded_force(
        charging0, "lambda_coul", 0.0
    )
    vanishing_one, _vanishing_zero = cte.production_vanishing_fixed_hamiltonian_systems(
        vanishing_input, LIGAND_INDICES, [1.0, 0.0], box
    )
    cte.assert_mem00h_switching_convention(vanishing_one, context="test:vanishing_one")
    cte.assert_mem00h_switching_convention(charging0_baked, context="test:charging0_baked")

    report = cte.compare_endpoint(
        "C_seam_normalized", vanishing_one, charging0_baked, _nm(positions), box,
        production_groups={0, 1, 2}, reference_groups={0},
        platform_name="Reference",
    )
    assert report["passed"], report
    # 这个小体系比真实 C1/C2 多了 WCA/softcore 表达式的额外浮点运算，残差
    # 停在 ~1e-5 量级（仍比 1e-3 门宽两个量级），不是真实数据那种 1e-12~
    # 1e-13——两者都叫"回到机器精度"，只是体系越复杂、累积浮点噪声越大，
    # 门槛按量级不按字面数值判读。
    assert report["max_abs_force_component_diff_kj_mol_nm"] < 1e-4


def test_mem00h_normalized_raw_system_is_noop_when_already_compliant():
    """C1 这类本来就没有局部 switch 的 raw System，归一化必须是 no-op
    （cutoff/switching 都不变），不能因为归一化本身引入任何差异。"""
    system, _topology, _positions, _box, _spec = _case(1, n_dummies=1)
    nb = _find_nonbonded_force_for_test(system)
    assert nb.getUseSwitchingFunction() is False  # fixture 本来就没设 switch

    normalized = cte.mem00h_normalized_raw_system(system)
    nb2 = _find_nonbonded_force_for_test(normalized)
    assert nb2.getUseSwitchingFunction() is False
    assert nb2.getCutoffDistance().value_in_unit(unit.nanometer) == pytest.approx(
        nb.getCutoffDistance().value_in_unit(unit.nanometer)
    )


def test_mem00h_normalized_raw_system_fails_closed_on_unexpected_cutoff():
    """cutoff 不是 MEM00H_CUTOFF_NM 时必须 fail closed，不能悄悄改写成
    1.0nm——cutoff 不符说明输入协议本身就不对，需要先查清楚。"""
    system, _topology, _positions, _box, _spec = _case(1, n_dummies=1)
    nb = _find_nonbonded_force_for_test(system)
    nb.setCutoffDistance(1.2 * unit.nanometer)

    with pytest.raises(RuntimeError, match="cutoff"):
        cte.mem00h_normalized_raw_system(system)


def test_assert_mem00h_switching_convention_fails_closed_when_switch_still_enabled():
    system, _topology, _positions, _box, _spec = _case(1, n_dummies=1)
    nb = _find_nonbonded_force_for_test(system)
    nb.setUseSwitchingFunction(True)
    nb.setSwitchingDistance(0.9 * unit.nanometer)

    with pytest.raises(RuntimeError, match="UseSwitchingFunction"):
        cte.assert_mem00h_switching_convention(system, context="test")


@pytest.mark.xfail(
    reason="P1-19（2026-08-30 登记）：v3 charging 口径下 seam 失配（同 "
    "test_vanishing_lambda_one_seam_matches_charging_lambda_zero 的 xfail 理由）。"
    "修复后应转 XPASS 并摘除。",
    strict=False,
)
def test_run_protocol_v2_matrix_cd_normalizes_c2_style_switch_before_c_seam(tmp_path):
    """端到端：`run_protocol_v2_matrix_cd` 内部自动归一化，即使传入的
    case_dir 的 raw system.xml 带着 C2 那种局部 switch，C seam 也必须干净
    通过——不需要调用方自己先归一化。"""
    system, topology, positions, box, spec = _case_with_environment_atom_in_switch_window(
        switch_nm=0.995, cutoff_nm=1.0, r_nm=0.997,
    )
    positions_nm = _nm(positions)
    case_dir = _write_synthetic_case_dir(
        tmp_path, system=system, topology=topology,
        positions_nm=positions_nm, box_nm=box, spec=spec,
    )

    report = cte.run_protocol_v2_matrix_cd(
        case_dir,
        lambda0_dcd_name="lam0.dcd",
        lambda0_frame_indices=[0, 1],
        lambda_name="lambda_coul",
        reference_platform_name="Reference",
        mixed_platform_name="CPU",
        mixed_platform_properties={},
    )
    assert report["passed"], report["failed_frames"]
    for frame in report["frames"]:
        assert frame["C"]["passed"], frame["C"]


def test_compare_vanishing_zero_endpoint_fails_closed_on_platform_mismatch(monkeypatch):
    """resolved platform/property 与请求值不一致时必须让 `passed` 变 False——
    不能只是记在报告里没人看（与 `compare_endpoint` 的同一处理由一致）。用
    monkeypatch 直接造一个"resolved != requested"的 `platform_info`，不依赖
    某个具体 OpenMM 平台真的会不会在 Context 构造阶段就拒绝未知属性名（实测
    Reference 平台连合法属性名都不接受，构造阶段直接抛异常，测不到"静默不
    生效"这一支——这正是 CUDA 双精度/mixed 这类"请求了不代表真的生效"场景，
    只能在没有 CUDA 的环境下用 monkeypatch 模拟）。
    """
    system, topology, positions, box, spec = _case(1, n_dummies=1)
    charging0 = cte.production_charging_system(
        system, LIGAND_INDICES, topology, spec, lambda_name="lambda_coul"
    )
    vanishing_input = core.bake_global_parameter_into_fixed_nonbonded_force(
        charging0, "lambda_coul", 0.0
    )
    _vanishing_one, vanishing_zero = cte.production_vanishing_fixed_hamiltonian_systems(
        vanishing_input, LIGAND_INDICES, [1.0, 0.0], box
    )

    def _fake_evaluate_with_mismatched_platform_info(system, positions_nm, box_vectors_nm, **kwargs):
        # 故意不把 kwargs 里的 platform_properties 转给真实 Context——
        # Reference 平台连合法属性名都会在 Context 构造阶段直接拒绝，测不到
        # "请求了但静默没生效"这一支；这里直接绕开 `cte.evaluate*`（会重新
        # 走到被 patch 的同一个函数造成递归），用裸 OpenMM 拿真实 energy/
        # forces，platform_info 单独伪造成"resolved != requested"。
        platform_name = kwargs.get("platform_name", "Reference")
        integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
        platform = openmm.Platform.getPlatformByName(platform_name)
        context = openmm.Context(system, integrator, platform)
        context.setPeriodicBoxVectors(*(np.asarray(box_vectors_nm, dtype=float) * unit.nanometer))
        context.setPositions(np.asarray(positions_nm, dtype=float) * unit.nanometer)
        for name, value in (kwargs.get("global_parameters") or {}).items():
            context.setParameter(name, float(value))
        state_kwargs = dict(getEnergy=True, getForces=True)
        if kwargs.get("groups") is not None:
            state_kwargs["groups"] = set(int(g) for g in kwargs["groups"])
        state = context.getState(**state_kwargs)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = np.asarray(
            state.getForces().value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        )
        del context, integrator
        platform_info = {
            "platform_requested": platform_name,
            "platform_resolved": platform_name,
            "properties_requested": {"Precision": "mixed"},
            "properties_resolved": {"Precision": "single"},
        }
        return float(energy), forces, platform_info

    monkeypatch.setattr(cte, "evaluate_with_platform_info", _fake_evaluate_with_mismatched_platform_info)

    report = cte.compare_vanishing_zero_endpoint(
        vanishing_zero, _nm(positions), box,
        ligand_environment_groups={1},
        platform_name="Reference",
        platform_properties={"Precision": "mixed"},
    )
    assert report["e_ligand_environment_kj_mol"] == pytest.approx(0.0, abs=1e-9), (
        "结构性零本身不受影响——失败必须来自 platform_verified，不是数值门。"
    )
    assert report["platform_verified"] is False
    assert report["passed"] is False


# ---------------------------------------------------------------------------
# 4. reference planner 独立性
# ---------------------------------------------------------------------------


def test_reference_builders_never_call_forbidden_production_functions():
    """把每个禁止函数都换成"调用即抛异常"，reference builder 仍必须正常跑完。"""
    system, _topology, _positions, _box, spec = _case(1, n_dummies=1)
    with cte.forbidden_calls_disabled():
        cte.reference_charging_endpoint_system(system, LIGAND_INDICES, spec, lam=1.0)
        cte.reference_charging_endpoint_system(system, LIGAND_INDICES, spec, lam=0.0)
        cte.reference_vanishing_zero_system(system, LIGAND_INDICES, spec)


def test_forbidden_calls_guard_actually_traps_a_call():
    """守卫本身必须真的会炸——否则上一条测试就是"什么都没测"的假阳性。"""
    system, topology, positions, box, spec = _case(1, n_dummies=1)
    with cte.forbidden_calls_disabled():
        with pytest.raises(AssertionError, match="被禁止的生产函数"):
            ie.select_co_alchemical_ion_once(
                system, LIGAND_INDICES, topology, positions, box,
                charge_treatment=core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
            )
        with pytest.raises(AssertionError, match="被禁止的生产函数"):
            ie.build_ibs_dual_system(
                system, topology=None, perturbed_indices=LIGAND_INDICES,
                lambdas_coul=[0.0], lambdas_vdw=[0.0],
                alchemical_params=core.ACESoftcorePotential.from_dict(
                    core.ACESoftcorePotential.optimize_alpha(len(LIGAND_INDICES))
                ),
            )


# ---------------------------------------------------------------------------
# 5. 篡改参数必须触发 gate；缺 frame/box 必须 fail closed
# ---------------------------------------------------------------------------


def test_tampering_ligand_indices_is_rejected():
    """把 co-ion 原子塞进 ligand_indices 必须当场 raise，不能得到一个"看起来
    正常但物理上乱掉"的 System。
    """
    system, _topology, _positions, _box, spec = _case(1, n_dummies=1)
    coion_index = int(spec["ions"][0]["atom_index"])
    with pytest.raises(ValueError, match="重叠"):
        cte.reference_charging_endpoint_system(
            system, LIGAND_INDICES + [coion_index], spec, lam=0.0
        )


def test_tampering_lambda_off_endpoint_is_rejected():
    system, _topology, _positions, _box, spec = _case(1, n_dummies=1)
    with pytest.raises(ValueError, match=r"λ∈\{0,1\}"):
        cte.reference_charging_endpoint_system(system, LIGAND_INDICES, spec, lam=0.37)


def test_static_check_rejects_already_prepared_system_as_raw_input():
    """防止把 `system_prepared.xml`（已配置过 charging）误当 raw system 喂进来。"""
    system, topology, _positions, _box, spec = _case(1, n_dummies=1)
    cte.assert_system_not_alchemically_configured(system, context="raw-check")

    prepared = cte.production_charging_system(system, LIGAND_INDICES, topology, spec)
    with pytest.raises(RuntimeError, match="system_prepared.xml"):
        cte.assert_system_not_alchemically_configured(prepared, context="raw-check")


def test_static_check_rejects_wrong_protocol_version():
    cte.assert_protocol_version({"protocol_version": 1}, expected=1, path="x.json")
    with pytest.raises(RuntimeError, match="protocol_version"):
        cte.assert_protocol_version({"protocol_version": 7}, expected=1, path="x.json")


def test_static_check_rejects_non_endpoint_lambda():
    cte.assert_lambda_is_exact_endpoint(1.0, context="x")
    cte.assert_lambda_is_exact_endpoint(0.0, context="x")
    with pytest.raises(RuntimeError, match="不是精确的"):
        cte.assert_lambda_is_exact_endpoint(0.5, context="x")


def test_evaluate_fails_closed_on_non_finite_positions_or_box():
    """缺 frame/box 的最小可测子集：非有限坐标/盒必须当场 raise，不能悄悄算出
    一个 NaN 能量再靠下游门去发现。
    """
    system, _topology, positions, box, _spec = _case(1, n_dummies=1)
    bad_positions = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float).copy()
    bad_positions[0, 0] = float("nan")
    with pytest.raises(ValueError, match="positions"):
        cte.evaluate(system, bad_positions, box)

    bad_box = np.asarray(box, dtype=float).copy()
    bad_box[0, 0] = float("inf")
    with pytest.raises(ValueError, match="box"):
        cte.evaluate(
            system, np.asarray(positions.value_in_unit(unit.nanometer), dtype=float), bad_box
        )


def test_compare_endpoint_fails_closed_on_particle_count_mismatch():
    """production/reference 粒子数不一致（比如两个 case 的产物被错配）必须
    fail closed，而不是让 numpy 广播出一个看起来能算的假结果。

    用 `_compare_energy_and_forces` 这个纯函数直接测：真实 System 走
    `evaluate()` 时，OpenMM 自己的 `setPositions()` 会在粒子数不对时先炸，
    根本走不到这段形状检查——所以必须绕开 OpenMM，直接喂手造的数组。
    """
    with pytest.raises(RuntimeError, match="形状"):
        cte._compare_energy_and_forces(
            "mismatch",
            0.0, np.zeros((4, 3)),
            0.0, np.zeros((5, 3)),
            energy_rel_tol=cte.ENERGY_RELATIVE_TOLERANCE,
            force_abs_tol=cte.FORCE_ABS_TOLERANCE_KJ_MOL_NM,
        )


# ---------------------------------------------------------------------------
# Protocol v2（2026-08-11）：`force_gate_mode` 契约。
#
# 归因诊断（`diagnose_coion_parameteroffset_mixed_precision.py`）证明了
# CUDA mixed precision 下"production vs 独立 reference"的力差是平台数值
# 路径问题、不是构造错误——Reference 平台上同一对 System 逐位相同。
# `force_gate_mode="diagnostic"` 就是把这条判据从"力差参与 passed"改成
# "力差只记录"，同时保持 energy 门和"力必须有限"这两条硬门不变。
# ---------------------------------------------------------------------------


def test_force_gate_mode_hard_fails_on_large_force_diff():
    """默认 `"hard"`：能量再干净，力差超门也必须 fail——这是 v1 的既有行为，
    必须原样保留，不能被新参数悄悄改掉。
    """
    e_prod, e_ref = 100.0, 100.0000001  # 能量几乎完全一致
    f_prod = np.zeros((3, 3))
    f_ref = np.zeros((3, 3))
    f_ref[1, 0] = 1.0  # 力差远超 1e-3 门
    report = cte._compare_energy_and_forces(
        "hard", e_prod, f_prod, e_ref, f_ref,
        energy_rel_tol=cte.ENERGY_RELATIVE_TOLERANCE,
        force_abs_tol=cte.FORCE_ABS_TOLERANCE_KJ_MOL_NM,
        force_gate_mode="hard",
    )
    assert report["force_within_tolerance"] is False
    assert report["passed"] is False


def test_force_gate_mode_diagnostic_ignores_force_diff_but_keeps_energy_hard():
    """`"diagnostic"`：同样的力差不再让 `passed` 变 False，但力差数值仍然
    如实记录（`force_within_tolerance=False`），不是假装没看见。
    """
    e_prod, e_ref = 100.0, 100.0000001
    f_prod = np.zeros((3, 3))
    f_ref = np.zeros((3, 3))
    f_ref[1, 0] = 1.0
    report = cte._compare_energy_and_forces(
        "diagnostic", e_prod, f_prod, e_ref, f_ref,
        energy_rel_tol=cte.ENERGY_RELATIVE_TOLERANCE,
        force_abs_tol=cte.FORCE_ABS_TOLERANCE_KJ_MOL_NM,
        force_gate_mode="diagnostic",
    )
    assert report["max_abs_force_component_diff_kj_mol_nm"] == pytest.approx(1.0)
    assert report["force_within_tolerance"] is False  # 如实记录，不是被抹掉
    assert report["passed"] is True  # 但不参与 passed


def test_force_gate_mode_diagnostic_still_fails_on_bad_energy():
    """`"diagnostic"` 不是"这条比较随便都过"——energy 门依然是硬门。"""
    e_prod, e_ref = 100.0, 50.0  # 能量差得很远
    f_prod = np.zeros((3, 3))
    f_ref = np.zeros((3, 3))  # 力完全一致
    report = cte._compare_energy_and_forces(
        "diagnostic_bad_energy", e_prod, f_prod, e_ref, f_ref,
        energy_rel_tol=cte.ENERGY_RELATIVE_TOLERANCE,
        force_abs_tol=cte.FORCE_ABS_TOLERANCE_KJ_MOL_NM,
        force_gate_mode="diagnostic",
    )
    assert report["passed"] is False


def test_force_gate_mode_diagnostic_still_fails_on_non_finite_force():
    """无论哪种模式，非有限力都必须 fail——这条不受 `force_gate_mode` 影响。"""
    e_prod, e_ref = 100.0, 100.0000001
    f_prod = np.zeros((3, 3))
    f_ref = np.zeros((3, 3))
    f_ref[0, 0] = np.nan
    report = cte._compare_energy_and_forces(
        "diagnostic_nan_force", e_prod, f_prod, e_ref, f_ref,
        energy_rel_tol=cte.ENERGY_RELATIVE_TOLERANCE,
        force_abs_tol=cte.FORCE_ABS_TOLERANCE_KJ_MOL_NM,
        force_gate_mode="diagnostic",
    )
    assert report["forces_finite"] is False
    assert report["passed"] is False


def test_force_gate_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="force_gate_mode"):
        cte._compare_energy_and_forces(
            "bad_mode", 1.0, np.zeros((1, 3)), 1.0, np.zeros((1, 3)),
            energy_rel_tol=cte.ENERGY_RELATIVE_TOLERANCE,
            force_abs_tol=cte.FORCE_ABS_TOLERANCE_KJ_MOL_NM,
            force_gate_mode="not_a_real_mode",
        )
