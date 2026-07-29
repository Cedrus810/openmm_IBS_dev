"""把 `boresch_dihedral_rad` 的**符号约定**钉死在 OpenMM 自己的 `dihedral()` 上。

真实事故（2026-07-29 定位，`output_lrc_fix`）
--------------------------------------------
`abfe_core.py` 里有**四份**手写二面角副本，都写成

    m1 = n1 × b2̂ ;  φ = atan2(m1·n2, n1·n2)

而 `(n1×b2̂)·n2 = −(n1×n2)·b2̂`，所以它们返回的是 **−φ**。距离和键角不受影响
（`arccos` 无符号），只有三个二面角整体反号 —— 即限制势参考几何的**镜像**。

Boresch 参考值 phiA0/phiB0/phiC0 是喂给 `LambdaDependentBoreschForce` 表达式里
OpenMM 的 `dihedral(p1,p2,p3,p4)` 的，而 OpenMM 与 `mdtraj.compute_dihedrals`
用的都是标准（IUPAC）约定。于是：

1. `boresch_simple.json` 用 mdtraj 算出正确的系综均值参考值；
2. 带 Boresch 的 rebalance 用它把配体稳稳按在自己的 pose 上；
3. 紧接着 `update_boresch_from_last_frame` 用反号的副本重算并**覆盖**参考值：

       phiA0  +1.6696 → −1.7168      phiB0  −1.8045 → +1.8136
       phiC0  −0.6839 → +0.6163      (r0/θA/θB 只差 <0.03，因为它们无符号)

4. attachment 腿的 λ=1 参考态因此变成当前 pose 的镜像，每个二面角都坐在
   `k(1−cosΔ)` 的势壁顶上（Δ≈π ⟹ ≈2k）：λ=0 实测 ⟨U_B⟩=777 kJ/mol、
   max=1115 ≈ Σ2k_φ=1140，ΔG(A′→A) 从应有的 ~5.5 涨到 98.8 kJ/mol，
   BAR/TI 一致性门失败（8.27 > 8.13）。

为什么以前的测试抓不住：`test_boresch_attachment_leg.py` 里的 fixture 自己也有
一份**同样反号**的手写副本，它与当时同样反号的生产代码"自洽"，对镜像事故零分辨
力。所以本文件的原则是——**绝不自己写二面角公式**，只拿 OpenMM / mdtraj 当基准。

全部 `cpu_only`，毫秒级（Reference platform，6 个粒子，不跑动力学）。
"""

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from abfe_core import (  # noqa: E402
    LambdaDependentBoreschForce,
    boresch_dihedral_rad,
    calc_boresch_from_last_frame,
)

pytestmark = pytest.mark.cpu_only


# ---------------------------------------------------------------------------
# 测试构象
# ---------------------------------------------------------------------------
# 6 个锚点原子（nm），顺序即全仓库约定的 [R0(离配体最近), R1, R2, L0, L1, L2]。
# 选取约束（改动这组坐标前请确认 test_fixture_is_discriminating 仍通过）：
#   · |R0−L0| ∈ [0.3, 2.0] nm      —— calc_boresch_from_last_frame 的硬门
#   · θA, θB ∈ [40°, 140°]          —— update_boresch_from_last_frame 的硬门
#   · 三个二面角都远离 0 和 ±π      —— 否则镜像 −φ 与 φ 重合，测试失去分辨力
#
# 手算得到的目标值（test_fixture_is_discriminating 会实际校验）：
#   r0 = 0.4848 nm   θA = 63.4°   θB = 109.0°
#   φA ≈ −1.864 rad (|sin|=0.96)   φB ≈ +1.020 (0.85)   φC ≈ +1.200 (0.93)
#   三个 φ 全反号的代价 ≈ 398 + 324 + 226 = 948 kJ/mol
# —— 与真实事故现场（output_lrc_fix 的 λ=0 ⟨U_B⟩=777、max=1115）同一量级。
#
# ⚠️ L2 不是随手写的：第一版取 (0.55,−0.38,0.33) 时 φC = −3.105 rad，离 ±π 只有
# 2°，镜像代价塌到 0.36 kJ/mol，整条 φC 断言形同空转。现在这个 L2 = L1 + b3，
# b3 = 0.08·b2̂ + 0.1398·n̂1 − 0.0544·ê（b2̂ = (L1−L0)̂，n̂1 = ((L0−R0)×(L1−L0))̂，
# ê = b2̂×n̂1），按 φ = atan2(q, −s) 反解出 φC = +1.2 rad。
_POS_NM = np.array(
    [
        [0.00, 0.00, 0.00],       # R0
        [0.15, 0.02, -0.03],      # R1
        [0.28, 0.14, 0.05],       # R2
        [0.30, -0.35, 0.15],      # L0
        [0.42, -0.28, 0.26],      # L1
        [0.3669, -0.2253, 0.4120],  # L2  (见上面的构造，勿随手改)
    ],
    dtype=np.float64,
)
_REC = [0, 1, 2]
_LIG = [3, 4, 5]

# 与 LambdaDependentBoreschForce 表达式逐项对齐（p1..p6 = R0,R1,R2,L0,L1,L2）：
#   phiA0 = dihedral(p3,p2,p1,p4)   phiB0 = dihedral(p2,p1,p4,p5)
#   phiC0 = dihedral(p1,p4,p5,p6)
_QUADS = {
    "phiA0": (2, 1, 0, 3),
    "phiB0": (1, 0, 3, 4),
    "phiC0": (0, 3, 4, 5),
}

# 量级取自真实体系（output_lrc_fix/boresch_simple.json），好让"镜像要付 ≈2k"这条
# 断言的数值尺度和事故现场一致。
_FC = {
    "kr": 2000.0,
    "kthetaA": 282.4571,
    "kthetaB": 273.2941,
    "kphiA": 216.7324,
    "kphiB": 222.9795,
    "kphiC": 130.0998,
}


def _reference_context(system):
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    return openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference")
    ), integrator


def _openmm_dihedral_rad(pos_nm, quad):
    """用 OpenMM **自己的** `dihedral()` 量一个四元组。

    手法：把 `CustomCompoundBondForce` 的能量表达式直接写成那个角度本身，于是
    势能的数值（kJ/mol）就是角度的数值（rad）。这样基准完全来自 OpenMM，不掺任何
    本仓库的几何代码——正是这一点让本测试对"两边同时错号"免疫。
    """
    system = openmm.System()
    for _ in range(len(pos_nm)):
        system.addParticle(12.0 * unit.amu)
    force = openmm.CustomCompoundBondForce(4, "dihedral(p1,p2,p3,p4)")
    force.addBond([int(i) for i in quad])
    system.addForce(force)

    context, integrator = _reference_context(system)
    try:
        context.setPositions(pos_nm * unit.nanometer)
        return float(
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
    finally:
        del context, integrator


def _boresch_energy_kj(pos_nm, eq, fc):
    """在给定构象上量 U_Boresch（λ=1）。System 里只有这一条力。"""
    system = openmm.System()
    for _ in range(len(pos_nm)):
        system.addParticle(12.0 * unit.amu)
    system.addForce(
        LambdaDependentBoreschForce(
            rec_idx=_REC, lig_idx=_LIG, eq=eq, fc=fc, fixed_lam=1.0, use_pbc=False
        )
    )
    context, integrator = _reference_context(system)
    try:
        context.setPositions(pos_nm * unit.nanometer)
        return float(
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
    finally:
        del context, integrator


def _wrap_to_pi(x):
    return float((float(x) + np.pi) % (2.0 * np.pi) - np.pi)


# ---------------------------------------------------------------------------
# 1) 手算基准：一个可以纸笔复核的构型
# ---------------------------------------------------------------------------
def test_hand_computed_case_is_plus_half_pi():
    """a=(0,1,0) b=(0,0,0) c=(1,0,0) d=(1,0,1) ⟹ φ = +π/2（IUPAC）。

    纸笔复核：b1=(0,−1,0) b2=(1,0,0) b3=(0,0,1) ⟹ n1=(0,0,1) n2=(0,−1,0)，
    n1·n2=0，(n1×n2)·b2̂=(1,0,0)·(1,0,0)=+1 ⟹ atan2(+1, 0)=+π/2。
    反号的老公式在这里给 −π/2。
    """
    phi = boresch_dihedral_rad(
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 1.0]),
    )
    assert phi == pytest.approx(np.pi / 2.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 2) 主判据：与 OpenMM 的 dihedral() 同号同值
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(_QUADS))
def test_matches_openmm_dihedral(name):
    quad = _QUADS[name]
    ours = boresch_dihedral_rad(*[_POS_NM[i] for i in quad])
    theirs = _openmm_dihedral_rad(_POS_NM, quad)
    assert ours == pytest.approx(theirs, abs=1e-6), (
        f"{name}: boresch_dihedral_rad={ours:+.6f} 与 OpenMM dihedral()={theirs:+.6f} 不符。"
        f"差 {ours - theirs:+.6f} rad；若接近 {-2 * theirs:+.6f} 说明符号又反了。"
    )


def test_matches_mdtraj():
    """也必须与 mdtraj 同号：`boresch_simple.json` 的参考值是 mdtraj 算的，两者
    不同号就等于估算器和末帧重锚给出互为镜像的参考几何（正是 2026-07-29 事故）。
    """
    md = pytest.importorskip("mdtraj")
    top = md.Topology()
    residue = top.add_residue("XXX", top.add_chain())
    for i in range(len(_POS_NM)):
        top.add_atom(f"C{i}", md.element.carbon, residue)
    traj = md.Trajectory(
        xyz=_POS_NM.reshape(1, len(_POS_NM), 3).astype(np.float32), topology=top
    )

    names = sorted(_QUADS)
    theirs = md.compute_dihedrals(traj, [list(_QUADS[n]) for n in names])[0]
    for name, ref in zip(names, theirs):
        ours = boresch_dihedral_rad(*[_POS_NM[i] for i in _QUADS[name]])
        assert ours == pytest.approx(float(ref), abs=1e-5), (
            f"{name}: boresch_dihedral_rad={ours:+.6f} 与 mdtraj={float(ref):+.6f} 不符"
        )


# ---------------------------------------------------------------------------
# 3) fixture 自检：这组坐标必须真的能区分 φ 和 −φ
# ---------------------------------------------------------------------------
def test_fixture_is_discriminating():
    """若某个 φ 落在 0 或 ±π 附近，镜像与原值重合，上面的断言就成了空转。

    顺带确认这组坐标落在 calc_boresch_from_last_frame /
    update_boresch_from_last_frame 的硬门内，否则失败信息会变成难懂的 ValueError。
    """
    eq = calc_boresch_from_last_frame(_POS_NM, _REC, _LIG)

    assert 0.3 <= eq["r0"] <= 2.0, f"r0={eq['r0']:.4f} nm 超出 [0.3, 2.0] 硬门"
    for key in ("thetaA0", "thetaB0"):
        deg = np.degrees(eq[key])
        assert 40.0 <= deg <= 140.0, f"{key}={deg:.1f}° 超出 [40°, 140°] 硬门"

    for key in ("phiA0", "phiB0", "phiC0"):
        assert abs(np.sin(eq[key])) > 0.2, (
            f"{key}={eq[key]:+.4f} rad 太接近 0/±π（|sin|={abs(np.sin(eq[key])):.3f}），"
            "镜像与原值几乎重合，本文件的符号断言会失去分辨力。请换一组 _POS_NM。"
        )


# ---------------------------------------------------------------------------
# 4) 端到端：参考值取自某构象 ⟹ 该构象上 U_Boresch ≈ 0
# ---------------------------------------------------------------------------
def test_reference_derived_from_pose_has_zero_restraint_energy():
    """这条与事故一一对应：`calc_boresch_from_last_frame` 从一个构象导出参考值，
    把它喂回 `LambdaDependentBoreschForce`，在**同一个构象**上限制能必须为 0。

    符号一错，λ=1 的参考态就是该构象的镜像，这里会立刻变成几百 kJ/mol。
    """
    eq = calc_boresch_from_last_frame(_POS_NM, _REC, _LIG)
    u = _boresch_energy_kj(_POS_NM, eq, _FC)
    assert u == pytest.approx(0.0, abs=1e-6), (
        f"参考值来自本构象，U_Boresch 却是 {u:.3f} kJ/mol。"
        "几乎肯定是 calc_boresch_from_last_frame 与 OpenMM dihedral() 约定不一致"
        "（2026-07-29 反号事故的复现）。"
    )


def test_mirrored_reference_costs_about_two_k_per_dihedral():
    """反过来钉住失败模式的量级：只把三个 φ0 取反，U_Boresch 必须暴涨到 ~Σ2k_φ。

    这解释了现场为什么会看到 λ=0 的 ⟨U_B⟩=777、max=1115 kJ/mol —— 不是"少数稀有
    帧支配指数平均"，而是**整个系综**都坐在 k(1−cosΔ)、Δ≈π 的势壁顶上。
    """
    eq = calc_boresch_from_last_frame(_POS_NM, _REC, _LIG)
    mirrored = dict(eq)
    for key in ("phiA0", "phiB0", "phiC0"):
        mirrored[key] = -eq[key]

    u = _boresch_energy_kj(_POS_NM, mirrored, _FC)

    # 逐项解析值：偏离是 Δ=φ−(−φ)=2φ（折回 [−π,π)），代价 k(1−cosΔ)。
    expected = sum(
        _FC[k] * (1.0 - np.cos(_wrap_to_pi(2.0 * eq[e])))
        for e, k in (("phiA0", "kphiA"), ("phiB0", "kphiB"), ("phiC0", "kphiC"))
    )
    assert u == pytest.approx(expected, rel=1e-6), (
        f"镜像参考值的 U_Boresch={u:.3f}，解析预期 {expected:.3f} kJ/mol"
    )
    assert u > 200.0, (
        f"镜像代价只有 {u:.3f} kJ/mol，说明这组 _POS_NM 的二面角太接近 0/±π，"
        "本测试没有分辨力（应由 test_fixture_is_discriminating 先拦住）"
    )
