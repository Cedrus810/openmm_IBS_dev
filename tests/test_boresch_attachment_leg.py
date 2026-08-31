"""[P1-17] Boresch attachment 腿 ΔG(A′→A) 的回归测试。

分两类：

* **纯逻辑**（毫秒级）：λ 阶梯方向、力组冲突、全局参数注册、循环合成时的符号门。
  这几条守的是最容易悄悄坏掉的东西——本仓库有过符号反转的前科
  （v21/v22 那次反号本身是错的，v27 才回滚），所以符号相关的断言必须有测试压着。
* **短 MD**（`cpu_only`，几秒）：6 粒子玩具体系上跑完整条腿，验证
  `ΔG(A′→A) ≥ 0` 这个严格数学下界，以及力常数 →0 时 ΔG →0。
"""

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

import ibs_engine  # noqa: E402
from abfe_core import boresch_dihedral_rad  # noqa: E402


# ---------------------------------------------------------------------------
# 玩具体系：3 个"受体"锚点 + 3 个"配体"锚点，外加一个弱简谐笼子把它们关在
# 有限区域里。没有笼子的话 λ=0 端粒子会自由扩散，ΔG 巨大且相邻态重叠极差——
# 那是物理上正确但数值上没必要的难题，测试只需要一个良态的正值。
#
# ⚠️ 几何必须非退化。第一版把 6 个原子摆成一条直线，于是
# `angle(R1, R0, L0) = 0°` 正好落在 angle()/dihedral() 的 1/sinθ 梯度奇点上
# （`abfe_core.LambdaDependentBoreschForce` 的注释里明确警告过这一点），
# 力炸到发散、u_kn 全是 NaN。下面这组坐标的四个"闸门角"都在 80–125°：
#     angle(R2,R1,R0)=119.9°  angle(R1,R0,L0)=81.0°
#     angle(R0,L0,L1)= 94.0°  angle(L0,L1,L2)=125.0°
# `_toy_restraint` 里有断言把这条守住，几何被人改坏时会直接报错而不是出 NaN。
# ---------------------------------------------------------------------------
_REF_POSITIONS_NM = np.array(
    [
        [1.00, 1.00, 1.00],   # R0
        [1.00, 1.15, 1.02],   # R1
        [1.12, 1.22, 1.10],   # R2
        [1.16, 1.02, 1.05],   # L0
        [1.21, 1.06, 0.91],   # L1
        [1.34, 1.10, 0.88],   # L2
    ],
    dtype=float,
)
_BOX_NM = 4.0


def _angle_rad(p, v, q):
    a, b = p - v, q - v
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


# 🚨 不要在这里重写二面角公式。这份 fixture 生成的 phiA0/phiB0/phiC0 会被喂进
# `LambdaDependentBoreschForce`，符号必须与 OpenMM `dihedral()` 一致；本文件此前
# 有一份返回 −φ 的手写副本，它与当时同样错号的生产代码"自洽"，因此对
# 2026-07-29 那次参考几何镜像事故完全没有分辨力。直接复用生产实现。
_dihedral_rad = boresch_dihedral_rad


def _toy_system(cage_k: float = 200.0):
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(_BOX_NM, 0, 0) * unit.nanometer,
        openmm.Vec3(0, _BOX_NM, 0) * unit.nanometer,
        openmm.Vec3(0, 0, _BOX_NM) * unit.nanometer,
    )
    for _ in range(len(_REF_POSITIONS_NM)):
        system.addParticle(12.0 * unit.amu)

    cage = openmm.CustomExternalForce("0.5*kcage*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    cage.addGlobalParameter("kcage", cage_k)
    for name in ("x0", "y0", "z0"):
        cage.addPerParticleParameter(name)
    for i, pos in enumerate(_REF_POSITIONS_NM):
        cage.addParticle(i, [float(pos[0]), float(pos[1]), float(pos[2])])
    cage.setForceGroup(0)
    system.addForce(cage)

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("TOY", chain)
    for i in range(len(_REF_POSITIONS_NM)):
        topology.addAtom(f"C{i}", app.Element.getBySymbol("C"), residue)
    topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())

    positions = _REF_POSITIONS_NM * unit.nanometer
    return system, topology, positions, system.getDefaultPeriodicBoxVectors()


def _toy_restraint(scale: float = 1.0):
    """限制参数。平衡值**从参考几何算出来**，不硬编码。

    硬编码会让限制的极小值和起始构象对不上——第一版就是那样（thetaA0 写死
    2.0 rad = 114.6°，而实际几何是 0°），限制一上来就往奇点方向猛拽。
    这里用与 `LambdaDependentBoreschForce` 完全相同的原子顺序约定：
        r0      = |R0−L0|
        thetaA0 = angle(R1, R0, L0)      顶点 R0
        thetaB0 = angle(R0, L0, L1)      顶点 L0
        phiA0   = dihedral(R2, R1, R0, L0)
        phiB0   = dihedral(R1, R0, L0, L1)
        phiC0   = dihedral(R0, L0, L1, L2)

    scale 缩放全部力常数；scale→0 就是"几乎没有限制"的极限。
    """
    p = _REF_POSITIONS_NM
    R0, R1, R2, L0, L1, L2 = (p[i] for i in range(6))

    # 四个"闸门角"必须远离 0/180——它们是 dihedral 解析梯度里 1/sinθ 的那个 θ。
    gates = {
        "angle(R2,R1,R0)": _angle_rad(R2, R1, R0),
        "angle(R1,R0,L0)": _angle_rad(R1, R0, L0),
        "angle(R0,L0,L1)": _angle_rad(R0, L0, L1),
        "angle(L0,L1,L2)": _angle_rad(L0, L1, L2),
    }
    for name, val in gates.items():
        deg = np.degrees(val)
        assert 30.0 < deg < 150.0, (
            f"玩具几何退化：{name} = {deg:.2f}°，落在 angle()/dihedral() 的 "
            f"1/sinθ 梯度奇点附近，力会发散、u_kn 出 NaN。请改坐标。"
        )

    return {
        "receptor_indices": [0, 1, 2],
        "ligand_indices": [3, 4, 5],
        "equilibrium_values": {
            "r0": float(np.linalg.norm(L0 - R0)),
            "thetaA0": gates["angle(R1,R0,L0)"],
            "thetaB0": gates["angle(R0,L0,L1)"],
            "phiA0": _dihedral_rad(R2, R1, R0, L0),
            "phiB0": _dihedral_rad(R1, R0, L0, L1),
            "phiC0": _dihedral_rad(R0, L0, L1, L2),
        },
        "force_constants": {
            "kr": 500.0 * scale,
            "kthetaA": 50.0 * scale,
            "kthetaB": 50.0 * scale,
            "kphiA": 50.0 * scale,
            "kphiB": 50.0 * scale,
            "kphiC": 50.0 * scale,
        },
    }


# ---------------------------------------------------------------------------
# 纯逻辑
# ---------------------------------------------------------------------------


def test_scalable_variant_registers_global_parameter_but_fixed_does_not():
    """`fixed_lam=1.0` 会把 λ 编译进表达式、不注册全局参数——attachment 腿不能用它。

    这正是生产 `_add_physical_boresch_restraint` 的形态，也是本条腿必须走
    `add_scalable_boresch_restraint` 的原因。
    """
    system_fixed, _, _, _ = _toy_system()
    ibs_engine._add_physical_boresch_restraint(system_fixed, _toy_restraint())
    fixed_force = system_fixed.getForce(system_fixed.getNumForces() - 1)
    fixed_params = {
        fixed_force.getGlobalParameterName(i)
        for i in range(fixed_force.getNumGlobalParameters())
    }
    assert ibs_engine.BORESCH_ATTACHMENT_LAMBDA_NAME not in fixed_params

    system_scalable, _, _, _ = _toy_system()
    assert ibs_engine.add_scalable_boresch_restraint(system_scalable, _toy_restraint())
    scalable_force = system_scalable.getForce(system_scalable.getNumForces() - 1)
    scalable_params = {
        scalable_force.getGlobalParameterName(i)
        for i in range(scalable_force.getNumGlobalParameters())
    }
    assert ibs_engine.BORESCH_ATTACHMENT_LAMBDA_NAME in scalable_params


@pytest.mark.parametrize(
    "bad_lambdas",
    [
        [1.0, 0.5, 0.0],          # 降序：delta_G 的符号含义会反过来
        [0.0, 0.5, 0.9],          # 不到 1
        [0.1, 0.5, 1.0],          # 不从 0 起
        [0.5],                    # 只有一个态：单向 FEP，不允许
    ],
)
def test_lambda_ladder_must_be_ascending_zero_to_one(bad_lambdas):
    system, topology, positions, box = _toy_system()
    with pytest.raises(ValueError):
        ibs_engine.run_boresch_attachment_leg(
            system, topology, positions, box, _toy_restraint(),
            lambdas=bad_lambdas, platform_name="Reference",
        )


def test_occupied_force_group_fails_closed():
    """力组被别人占着时必须报错——否则拆出来的"限制能量"里混着别的力。"""
    system, topology, positions, box = _toy_system()
    intruder = openmm.CustomExternalForce("0.0*x")
    intruder.setForceGroup(ibs_engine.BORESCH_ATTACHMENT_FORCE_GROUP)
    system.addForce(intruder)
    with pytest.raises(RuntimeError, match="力组"):
        ibs_engine.run_boresch_attachment_leg(
            system, topology, positions, box, _toy_restraint(),
            platform_name="Reference",
        )


def test_degenerate_anchor_geometry_fails_closed_before_nan():
    """退化几何必须在起跑前被拦下，而不是最后以「u_kn 含 NaN」收场。

    这是 2026-07-28 本测试自身踩过的坑：6 个原子摆成一条直线，
    `angle(R1,R0,L0)=0°` 正落在 1/sinθ 梯度奇点上，力发散、能量变 NaN，
    而报错要等到 MBAR 那一步才出现，离病因隔了十万八千里。
    """
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(_BOX_NM, 0, 0) * unit.nanometer,
        openmm.Vec3(0, _BOX_NM, 0) * unit.nanometer,
        openmm.Vec3(0, 0, _BOX_NM) * unit.nanometer,
    )
    for _ in range(6):
        system.addParticle(12.0 * unit.amu)
    topology = app.Topology()
    residue = topology.addResidue("LIN", topology.addChain())
    for i in range(6):
        topology.addAtom(f"C{i}", app.Element.getBySymbol("C"), residue)
    topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())
    collinear = np.array([[1.00 + 0.15 * i, 1.00, 1.00] for i in range(6)]) * unit.nanometer

    restraint = {
        "receptor_indices": [0, 1, 2],
        "ligand_indices": [3, 4, 5],
        "equilibrium_values": {
            "r0": 0.45, "thetaA0": 2.0, "thetaB0": 2.0,
            "phiA0": 0.0, "phiB0": 0.0, "phiC0": 0.0,
        },
        "force_constants": {
            "kr": 500.0, "kthetaA": 50.0, "kthetaB": 50.0,
            "kphiA": 50.0, "kphiB": 50.0, "kphiC": 50.0,
        },
    }
    with pytest.raises(RuntimeError, match="几何不安全"):
        ibs_engine.run_boresch_attachment_leg(
            system, topology, collinear, system.getDefaultPeriodicBoxVectors(),
            restraint, platform_name="Reference",
            n_steps_per_state=100, equil_steps_per_state=10, steps_per_sample=10,
            log=lambda *a, **k: None,
        )


def test_cycle_rejects_negative_attachment_term():
    """合成阶段也要有符号门：stage0 为负必须拒绝，而不是照单相加。"""
    from abfe_pipeline import ABFEPipeline

    pipeline = ABFEPipeline.__new__(ABFEPipeline)
    pipeline._log = lambda *a, **k: None
    pipeline.ligand_indices = []
    pipeline.temperature = 300.0 * unit.kelvin
    pipeline._last_run_config = {"potential_type": "softcore"}
    sampling = {
        "stage0": {"attachment_delta_G_kJ_mol": -1.0, "attachment_error_kJ_mol": 0.1, "converged": True},
        "stage1": {"total_delta_G": 60.0, "total_error": 1.0},
        "stage2": {"total_delta_G": 100.0, "total_error": 1.0},
    }
    with pytest.raises(RuntimeError, match="< 0"):
        pipeline.compute_final_results(
            sampling, {"delta_g_rest": 0.0, "error": 0.0}, system=None
        )


# ---------------------------------------------------------------------------
# 短 MD
# ---------------------------------------------------------------------------

# 这三条短 MD 测的是**正确性**（ΔG≥0、弱限制极限、力常数单调性），不是收敛性。
# 5000 步的玩具轨迹本来就不收敛，BAR/TI 与 split-half 两道**收敛**门理应响——
# 所以显式关掉它们。`ΔG≥0`、力组占用、几何奇点这些正确性门仍然全开。
# 门本身由下面的 test_*_gate_* 单独覆盖。
_SHORT_MD = dict(
    n_steps_per_state=5_000,
    equil_steps_per_state=1_000,
    steps_per_sample=25,
    platform_name="Reference",
    seed=20260728,
    enforce_convergence_gates=False,
)


@pytest.mark.cpu_only
def test_attachment_delta_g_is_nonnegative(tmp_path):
    """严格下界：U_Boresch ≥ 0 ⟹ ΔG(A′→A) = −kT·ln⟨exp(−βU)⟩ ≥ 0。"""
    pytest.importorskip("pymbar")
    system, topology, positions, box = _toy_system()
    result = ibs_engine.run_boresch_attachment_leg(
        system, topology, positions, box, _toy_restraint(),
        log=lambda *a, **k: None, output_dir=str(tmp_path / "leg"), **_SHORT_MD,
    )
    dg = result["attachment_delta_G_kJ_mol"]
    assert np.isfinite(dg)
    assert dg >= 0.0, f"ΔG(A′→A) = {dg} < 0，违反严格下界"
    assert result["direction"].startswith("A_prime_to_A")


@pytest.mark.cpu_only
def test_weak_restraint_limit_approaches_zero(tmp_path):
    """力常数 →0 时限制势 →0，ΔG(A′→A) 也必须 →0，且仍不为负。"""
    pytest.importorskip("pymbar")
    system, topology, positions, box = _toy_system()
    weak = ibs_engine.run_boresch_attachment_leg(
        system, topology, positions, box, _toy_restraint(scale=1.0e-6),
        log=lambda *a, **k: None, output_dir=str(tmp_path / "weak"), **_SHORT_MD,
    )
    dg_weak = weak["attachment_delta_G_kJ_mol"]
    assert dg_weak >= 0.0
    # kT ≈ 2.494 kJ/mol；力常数缩到百万分之一后该项应远小于 kT。
    assert dg_weak < 0.5, f"弱限制极限下 ΔG={dg_weak}，本应趋近 0"


def test_bar_ti_gate_catches_single_frame_domination():
    """人造一帧二面角反转（U_B≈1189 kJ/mol），BAR/TI 一致性门必须拦下。

    这是 HREMD 那轮 38.6±110 的失效模式：`k(1−cosΔ)` 反转时取 2k，三个二面角
    全反转 1189 kJ/mol = 477 kT。TI 是普通均值会被拉高，BAR 走指数平均、那一帧
    权重 e^-477 几乎为零所以纹丝不动——两者一分歧，门就该响。
    """
    K, N = 4, 30
    kt = 0.008314462618 * 300.0
    lam = np.array([0.0, 0.1, 0.35, 1.0])
    rng = np.random.default_rng(20260728)
    u_b = np.abs(rng.normal(5.0, 1.0, size=(K, N)))
    u_b[0, 0] = 1189.0                      # λ=0 采到一次全反转
    u_kn = np.zeros((K, K * N))
    for k in range(K):
        cols = slice(k * N, (k + 1) * N)
        for j in range(K):
            u_kn[j, cols] = lam[j] * u_b[k] / kt

    n_k = np.full(K, N, dtype=int)
    dg_bar, err, _ = ibs_engine._attachment_bar_chain(u_kn, n_k, kt)
    dg_ti = ibs_engine._attachment_ti(lam, [float(np.mean(u_b[k])) for k in range(K)])
    ok, tol, msg = ibs_engine.attachment_bar_ti_gate(dg_bar, err, dg_ti)
    assert not ok, f"单帧反转没有被门拦下：{msg}"


def test_gates_do_not_fire_on_numerically_zero_drift():
    """σ→0 时，纯 z 判据会把数值噪声判成大偏离——必须有绝对下限兜底。

    实测踩过：弱限制极限下所有边的 ΔG 都≈0、BAR 误差也≈0，
    结果报出「漂移 +0.0000 kJ/mol = 7.1×2σ」并拒绝返回。
    """
    ok, tol, msg = ibs_engine.attachment_split_half_gate(drift=1.0e-5, err=1.0e-9)
    assert ok, f"数值零漂移不该失败：{msg}"
    assert tol >= ibs_engine.ATTACHMENT_SPLIT_HALF_ABS_TOL_KJ

    ok2, _, _ = ibs_engine.attachment_bar_ti_gate(dg_bar=1.0e-6, err=1.0e-9, dg_ti=2.0e-6)
    assert ok2, "数值零分歧不该失败"

    # 真正的大漂移仍然要被拦下
    bad, _, _ = ibs_engine.attachment_split_half_gate(drift=5.0, err=0.1)
    assert not bad


@pytest.mark.cpu_only
def test_stronger_restraint_costs_more(tmp_path):
    """限制越硬，打开它越贵——单调性是这条腿最基本的物理自洽。"""
    pytest.importorskip("pymbar")
    results = []
    for scale in (0.05, 1.0):
        system, topology, positions, box = _toy_system()
        results.append(
            ibs_engine.run_boresch_attachment_leg(
                system, topology, positions, box, _toy_restraint(scale=scale),
                log=lambda *a, **k: None,
                output_dir=str(tmp_path / f"scale{scale}"), **_SHORT_MD,
            )["attachment_delta_G_kJ_mol"]
        )
    assert results[1] > results[0], f"硬限制 {results[1]} 不比软限制 {results[0]} 贵"
