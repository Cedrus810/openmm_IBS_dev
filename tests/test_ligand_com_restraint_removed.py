"""LIGAND_COM_RESTRAINT_PROTOCOL_VERSION=2 回归：Group 5 配体 COM 限制力必须不存在。

背景 4W53/STAGE2_GROUP5_CUDA_PBC_ROOT_CAUSE_2026-08-29.md：
旧实现是非周期绝对锚点的 CustomCentroidBondForce。CUDA 动力学中 centroid 被折叠进
主盒而锚点没有，绝对距离把两个不同周期像当成真实距离 ⟹ 永久激活、跨边界不连续的
外力，实测让配体在 110 ps 内定向漂移 30.9 nm（自由扩散 RMS 仅约 3.0 nm），
使轨迹不满足 MBAR 的平衡采样前提。

⚠️ 这个缺陷 **CPU 上不复现**（CPU 的 centroid 未折叠，与锚点成像一致），
**静态 setPositions() 测试也检不出**（力用的就是调用方给的坐标）。所以本文件锁的是
"这个力压根不存在"这一结构性事实，而不是它的能量值——后者在 CPU 上永远是 0。
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_protocol_version_is_two():
    from ibs_engine import LIGAND_COM_RESTRAINT_PROTOCOL_VERSION

    assert LIGAND_COM_RESTRAINT_PROTOCOL_VERSION == 2


def test_no_force_is_ever_assigned_to_group_five():
    """整个 engine 里不得再有任何 setForceGroup(5)。

    Group 5 现为空组。`e_base = groups{0,2,3,5}` 的记账口径
    (WCA_ACCOUNTING_VERSION=2) 不受影响——该组只是贡献 0。
    """
    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    assert "setForceGroup(5)" not in src


def test_com_restraint_globals_are_gone_from_the_builder():
    """k_com / r0_com / x0,y0,z0 这组全局参数不得再被添加。"""
    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    assert 'addGlobalParameter("k_com"' not in src
    assert 'addGlobalParameter("r0_com"' not in src
    assert "CustomCentroidBondForce(1, com_expr)" not in src


def test_reference_com_helper_has_no_caller():
    """`_compute_reference_com` 唯一的调用点已随该力移除；保留但必须无调用方。

    若将来重新出现调用，必须先确认不是又把它当成**非周期绝对锚点**——那正是被移除
    的 P0 缺陷。可用的周期写法见 `build_co_alchemical_ion_restraint` 的注释：
    `periodicdistance` 只存在于 CustomExternalForce；CustomCompoundBondForce 打开
    PBC 后其中的 `pointdistance` 就是 minimum-image 距离。
    """
    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    calls = re.findall(r"(?<!def )_compute_reference_com\s*\(", src)
    assert calls == [], f"意外出现调用点: {len(calls)} 处"


def test_protocol_version_is_in_the_stage_fingerprint_unconditionally():
    """该力的有无改变采样 Hamiltonian，且对 stage1/stage2 同时生效
    （它只在没有 Boresch 时添加，即溶剂腿两个 stage 都有），
    所以必须无条件进 stage 协议指纹，不能只挂在 vanishing 上。"""
    src = (REPO / "abfe_pipeline.py").read_text(encoding="utf-8")
    i = src.index('"kind": "dual_lambda_stage"')
    j = src.index('if stage_name == "vanishing":', i)
    payload = src[i:j]
    assert "ligand_com_restraint_protocol_version" in payload, (
        "必须在 vanishing 专属分支**之前**、即无条件的 payload 里"
    )


def test_built_window_system_has_no_group_five_force():
    """端到端：实际组装出来的窗口系统里不得有 force group == 5 的力对象。"""
    openmm = pytest.importorskip("openmm")
    from openmm import unit

    system = openmm.System()
    for _ in range(6):
        system.addParticle(12.0)
    L = 3.0
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(L, 0, 0), openmm.Vec3(0, L, 0), openmm.Vec3(0, 0, L)
    )
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    for i in range(6):
        nb.addParticle(0.0, 0.3 * unit.nanometer, 0.1 * unit.kilojoule_per_mole)
    system.addForce(nb)

    from abfe_pipeline import _resolve_alchemical_params
    from ibs_engine import build_ibs_dual_system

    positions = [openmm.Vec3(0.5 + 0.1 * i, 0.5, 0.5) for i in range(6)] * unit.nanometer
    try:
        built, _wrapper = build_ibs_dual_system(
            system=system,
            topology=None,
            perturbed_indices=[0, 1],
            lambdas_coul=[0.0, 0.0],
            lambdas_vdw=[1.0, 0.5],
            # 用生产自己的解析器构造，避免手写 dict 漏掉 alpha_convention 之类的
            # 协议字段（builder 对它是 fail-closed 的）。
            alchemical_params=_resolve_alchemical_params("softcore", None, [0, 1]),
            potential_type="softcore",
            temperature=298.15 * unit.kelvin,
            reference_positions=positions,     # 旧实现正是靠它设绝对锚点
            restraint_params=None,             # 无 Boresch ⟹ 旧实现会添加 Group 5
            dispersion_protocol="legacy_uniform_density_lrc",
            environment_type="soluble",
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        pytest.skip(f"最小体系不满足 builder 的前置条件，跳过端到端检查: {exc}")

    groups = {built.getForce(i).getForceGroup() for i in range(built.getNumForces())}
    assert 5 not in groups, f"仍存在 Group 5 力，force groups = {sorted(groups)}"
