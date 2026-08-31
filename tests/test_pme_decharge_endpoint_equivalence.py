"""P0-01：PME decharging 的 lambda=1 物理端点必须与原始 PME System 等价。

## 缺陷是什么（2026-08-30 定级，先在目标 OpenMM 上复现后修复）

`configure_pme_ligand_charge_offsets`（以及同构的
`configure_coalchemical_neutral_decharging` / `configure_charge_transfer_decharging`）
为了让"配体内部库仑不随 λ 变"，把所有**原本走普通非键（无 exception）**的 L–L
原子对补成了显式 exception：

    nb_force.addException(p1, p2, q1*q2, σ_comb, ε_comb, True)

但 OpenMM 的 exception 从不使用普通非键的 cutoff/PME 处理：它把这对相互作用
整体替换成一条直接对（direct pair），不再进入 real-space/reciprocal-space 的
Ewald 分解（OpenMM API 文档明确 "cutoffs are never applied to exceptions"，
OpenMM #2310 也记录过把普通 pair 以相同参数改成 exception 时 PME 能量/力会变）。
于是 **lambda=1 的"物理端点"已经不是原始 PME System** —— 错误是有限但系统性的，
并且会同时进入 REMD replica、u_kn 端点与最终 ΔG。旧测试都在 NoCutoff 下检查
参数表，所以抓不到它。

## 修法（协议身份 PME_DECHARGE_MODEL_VERSION → v3）

不再把普通 L–L 对转成 exception（撤销对物理端点的改写）。配体内部静电的 λ 口径
变为：

- 已有 L–L exception（1-2/1-3 排除、1-4 缩放）的 chargeProd 是独立参数、不读
  粒子电荷 → 天然逐 λ 恒定，维持冻结；
- 普通 L–L 对（≥1-5）随粒子 offset 线性缩放 → 内部库仑随 λ 湮灭（annihilation）。

两条腿（complex/solvent）的配体内部 Hamiltonian 完全相同，湮灭项在结合自由能里
严格相消，热力学循环仍闭合。**lambda=1 时粒子电荷被 offset 精确还原、没有任何
exception 增删 → 与原始 System 逐位等价。**

## 不要这样让本文件变绿

- 放宽比较容差去掩盖 λ=1 的能量/力差异；
- 只在 NoCutoff 下检查参数表、不碰真实 PME 的能量与力；
- 用"参数看起来一样"论证端点等价 —— exception 与普通 pair 参数相同不代表
  物理相同。
"""

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

openmm = pytest.importorskip("openmm")
from openmm import NonbondedForce, unit  # noqa: E402

import ibs_engine as ie  # noqa: E402

# 预注册容差：Reference 平台双精度下，λ=1 与原始 System 必须是同一个 Hamiltonian，
# 数值上只允许求值器路径差异级别的噪声。
ENERGY_TOL_KJ_MOL = 1.0e-6
FORCE_TOL_KJ_MOL_NM = 1.0e-6

BOX_NM = 3.0
CUTOFF_NM = 0.9

# 8 原子中性配体（净电荷精确为 0），后面 6 个"水"。
_LIGAND_Q_E = (-0.6, 0.4, -0.5, 0.3, 0.15, -0.45, 0.2, 0.5)
_N_LIG = len(_LIGAND_Q_E)
_N_WATER = 6
_N_PARTICLES = _N_LIG + 3 * _N_WATER

_LIGAND_SIGMA_NM = 0.32
_LIGAND_EPS_KJ_MOL = 0.45


def _build_ligand_positions_nm() -> np.ndarray:
    """配体摆成一条紧凑折线链，保证 ≥1-5 的普通 L–L 对大量落在 cutoff 之内。"""
    coords = []
    x, y, z = 1.0, 1.0, 1.0
    step = 0.18
    for i in range(_N_LIG):
        coords.append([x + step * i, y + (step * (i % 2)), z + 0.02 * (i % 3)])
    return np.asarray(coords, dtype=float)


def _build_reference_system(
    method=NonbondedForce.PME,
    ligand_charge_scale: float = 1.0,
):
    """手建最小周期体系：8 原子中性配体 + 6 个 TIP3P 型水。

    配体拓扑按 Amber 惯例：1-2/1-3 电荷排除（chargeProd=0），1-4 缩放
    （×0.833333），≥1-5 走普通非键（PME）。这正是缺陷触发的拓扑形态。

    `ligand_charge_scale` 是独立参考实现的 λ 口径：直接把配体**粒子**电荷乘 λ、
    不动任何 exception（exception 的 chargeProd 是独立参数、不读粒子电荷，
    天然逐 λ 恒定）—— 这正是 v3 协议"粒子 offset 缩放 + 既有 exception 冻结"
    应当在每个 λ 下逐位重现的 Hamiltonian。
    """
    system = openmm.System()
    for i in range(_N_LIG):
        system.addParticle((12.011 if i % 2 == 0 else 14.007) * unit.dalton)
    for _ in range(_N_WATER):
        system.addParticle(15.999 * unit.dalton)
        system.addParticle(1.008 * unit.dalton)
        system.addParticle(1.008 * unit.dalton)

    nb = NonbondedForce()
    nb.setNonbondedMethod(method)
    nb.setCutoffDistance(CUTOFF_NM * unit.nanometer)
    nb.setUseDispersionCorrection(True)

    def _add_particle(q_e, sigma_nm, eps_kj_mol):
        nb.addParticle(
            q_e * unit.elementary_charge,
            sigma_nm * unit.nanometer,
            eps_kj_mol * unit.kilojoule_per_mole,
        )

    for q in _LIGAND_Q_E:
        _add_particle(q * ligand_charge_scale, _LIGAND_SIGMA_NM, _LIGAND_EPS_KJ_MOL)
    for _ in range(_N_WATER):
        _add_particle(-0.834, 0.315, 0.636)
        _add_particle(0.417, 0.10, 0.0)
        _add_particle(0.417, 0.10, 0.0)

    # 水内部：O-H、H-H 全排除（chargeProd=0）。
    for w in range(_N_WATER):
        o = _N_LIG + 3 * w
        nb.addException(o, o + 1, 0.0 * unit.elementary_charge**2,
                        0.10 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
        nb.addException(o, o + 2, 0.0 * unit.elementary_charge**2,
                        0.10 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
        nb.addException(o + 1, o + 2, 0.0 * unit.elementary_charge**2,
                        0.10 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)

    # 配体内部：1-2 / 1-3 排除，1-4 缩放；≥1-5 保持普通非键。
    scaled = 0.8333333333
    lig_q = np.asarray(_LIGAND_Q_E, dtype=float)
    for i in range(_N_LIG):
        for j in range(i + 1, _N_LIG):
            sep = j - i
            if sep in (1, 2):
                nb.addException(i, j, 0.0 * unit.elementary_charge**2,
                                _LIGAND_SIGMA_NM * unit.nanometer,
                                _LIGAND_EPS_KJ_MOL * unit.kilojoule_per_mole)
            elif sep == 3:
                # exception 的 chargeProd 不乘 ligand_charge_scale：它是独立参数，
                # 不读粒子电荷 —— 配置后体系的冻结语义与此一致。
                cp = lig_q[i] * lig_q[j] * scaled
                nb.addException(i, j, cp * unit.elementary_charge**2,
                                _LIGAND_SIGMA_NM * unit.nanometer,
                                _LIGAND_EPS_KJ_MOL * unit.kilojoule_per_mole)

    system.addForce(nb)
    box = np.eye(3) * BOX_NM
    system.setDefaultPeriodicBoxVectors(*(box * unit.nanometer))

    positions_nm = np.zeros((_N_PARTICLES, 3), dtype=float)
    positions_nm[: _N_LIG] = _build_ligand_positions_nm()
    rng = np.random.default_rng(20260830)
    grid = np.arange(0.4, BOX_NM - 0.4, 0.5)
    w = 0
    for gx in grid:
        for gy in grid:
            if w >= _N_WATER:
                break
            base = np.array([gx, gy, 2.0 + 0.05 * w])
            positions_nm[_N_LIG + 3 * w: _N_LIG + 3 * w + 3] = (
                base + rng.uniform(-0.02, 0.02, (3, 3))
            )
            w += 1
    assert w == _N_WATER
    return system, positions_nm * unit.nanometer


def _n_normal_ll_pairs(system: openmm.System, n_lig: int = _N_LIG) -> int:
    """普通非键（无 exception 覆盖）的 L–L 对数 —— 这些正是被转成 exception 的对。"""
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    excepted = set()
    for i in range(nb.getNumExceptions()):
        p1, p2 = (int(x) for x in nb.getExceptionParameters(i)[:2])
        excepted.add((min(p1, p2), max(p1, p2)))
    lig = list(range(n_lig))
    return sum(
        1
        for a in range(len(lig))
        for b in range(a + 1, len(lig))
        if (lig[a], lig[b]) not in excepted
    )


def _energy_and_forces(system: openmm.System, positions, lam: float = None):
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    if lam is not None:
        context.setParameter("lambda_coul", lam)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    del context, integrator
    return energy, forces


# ---------------------------------------------------------------------------
# 1. 缺陷本体：λ=1 端点等价性（真实 PME）
# ---------------------------------------------------------------------------


def test_pme_lambda1_energy_and_forces_match_original_system():
    """配置后的 decharging System 在 λ=1 必须等于原始 PME System（能量 + 力）。"""
    assert _n_normal_ll_pairs(_build_reference_system()[0]) > 0, (
        "测试体系必须含有普通 L–L 非键对，否则根本覆盖不到缺陷形态"
    )

    original, positions = _build_reference_system()
    e_ref, f_ref = _energy_and_forces(original, positions)

    configured, positions2 = _build_reference_system()
    ie.configure_pme_ligand_charge_offsets(configured, list(range(_N_LIG)))
    e_cfg, f_cfg = _energy_and_forces(configured, positions2, lam=1.0)

    assert abs(e_cfg - e_ref) < ENERGY_TOL_KJ_MOL, (
        f"λ=1 总能量偏离原始 PME System：{e_cfg - e_ref:+.6e} kJ/mol"
    )
    max_dforce = float(np.max(np.abs(f_cfg - f_ref)))
    assert max_dforce < FORCE_TOL_KJ_MOL_NM, (
        f"λ=1 逐原子力最大偏差 {max_dforce:.3e} kJ/mol/nm"
    )


def test_configuration_must_not_add_or_remove_exceptions():
    """λ=1 端点等价的最小必要条件：不增删任何 exception（参数改写除外）。"""
    original, _ = _build_reference_system()
    nb0 = next(f for f in original.getForces() if isinstance(f, NonbondedForce))
    n_exc0 = nb0.getNumExceptions()

    configured, _ = _build_reference_system()
    ie.configure_pme_ligand_charge_offsets(configured, list(range(_N_LIG)))
    nb1 = next(f for f in configured.getForces() if isinstance(f, NonbondedForce))

    assert nb1.getNumExceptions() == n_exc0, (
        "decharging 配置增删了 exception；在 PME 下 exception 与普通非键物理不同，"
        "这会改写 λ=1 物理端点（P0-01）"
    )


# ---------------------------------------------------------------------------
# 2. 中间 λ 与 λ=0：对照独立参考实现（粒子电荷直接缩放）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lam", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_every_lambda_matches_direct_charge_scaling_reference(lam):
    """配置后 System(λ) ≡ 粒子电荷直接乘 λ 的参考 System，对所有 λ 成立。

    这同时钉住 λ=0 与中间态的 ligand-internal 口径：普通 L–L 库仑随 λ 线性
    湮灭、既有 exception 冻结 —— 正是参考实现所做的事。
    """
    ref_sys, positions = _build_reference_system(ligand_charge_scale=lam)
    e_ref, f_ref = _energy_and_forces(ref_sys, positions)

    cfg_sys, positions2 = _build_reference_system()
    ie.configure_pme_ligand_charge_offsets(cfg_sys, list(range(_N_LIG)))
    e_cfg, f_cfg = _energy_and_forces(cfg_sys, positions2, lam=lam)

    assert abs(e_cfg - e_ref) < ENERGY_TOL_KJ_MOL, (
        f"λ={lam} 能量偏离独立参考实现：{e_cfg - e_ref:+.6e} kJ/mol"
    )
    max_dforce = float(np.max(np.abs(f_cfg - f_ref)))
    assert max_dforce < FORCE_TOL_KJ_MOL_NM, (
        f"λ={lam} 逐原子力最大偏差 {max_dforce:.3e} kJ/mol/nm"
    )


# ---------------------------------------------------------------------------
# 3. 带电路线（co-annihilation / charge-transfer）不得再有同类改写
# ---------------------------------------------------------------------------


def test_no_decharging_builder_converts_normal_ll_pairs_to_exceptions():
    """三个 decharging builder 都不得再把普通 L–L 对补成 exception。

    静态守护：配体内部静电的 λ 口径只能靠"既有 exception 冻结 + 粒子 offset"，
    任何 addException 补对行为都会在 PME 下改写 λ=1 端点（P0-01 本体）。
    """
    import inspect
    from pathlib import Path

    src = (Path(__file__).absolute().parents[1] / "ibs_engine.py").read_text(
        encoding="utf-8"
    )
    for builder in (
        "configure_charge_transfer_decharging",
        "configure_coalchemical_neutral_decharging",
        "configure_pme_ligand_charge_offsets",
    ):
        start = src.index(f"def {builder}(")
        end = src.find("\ndef ", start + 1)
        body = src[start: end if end != -1 else None]
        assert "nb_force.addException(" not in body, (
            f"{builder} 仍在把普通 L–L 对补成 exception —— 这会在真实 PME 下"
            "改写 λ=1 物理端点（P0-01）。配体内部静电请改用"
            "「既有 exception 冻结 + 粒子 charge offset」表达。"
        )
