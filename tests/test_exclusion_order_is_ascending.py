"""排除表必须按 (min, max) 升序 —— 纯静态断言，不需要 GPU。

## 为什么需要这条测试

`CustomNonbondedForce` 对 `addExclusion()` 的**调用顺序**敏感。同一个集合、同样的
条数，只把顺序从 Python `set` 的哈希序改成升序，实测整步从 2.06 降到 1.71 ms/step
（真 Atenolol 膜体系 45354 原子、K=16、CUDA/mixed；三次独立测量给出 1.17-1.21×），
而随机打乱比 set 序还慢。**能量与力逐比特不变** —— 所以这纯粹是白烧的时间，
没有任何物理上的补偿，而且不会被任何数值断言发现。

历史：`abfe_core.sync_all_exclusions()` 曾经直接迭代 `missing = union_excl - existing`
这个 `set`，于是给"自己只带局部排除表"的力（典型是 Group-2 那个恢复配体内部非键的
力：interaction group 只有 41x41，却被灌进全系统 120895 条排除）追加了一条**乱序
尾巴**。它的产地本身是 `sorted()` 的，但有序前缀 + 乱序尾巴仍然是乱序 —— 所以
**在各产地加 `sorted()` 挡不住这个问题**，必须在追加处修，也必须在这里断言。

完整调查见 `docs/EXP-031_GPU_OPTIMIZATION_2026-09-09.md`（原始记录在 CUDA 沙箱的
`experiments/EXP-031_ibs_bias_fusion/PROPOSAL_mainline_integration.md` §2/§5.1，不在本仓）。

## 这条断言盯的是什么

不是"某一行代码长什么样"（那种断言一重构就假失败），而是**产物的可观测性质**：
任何建成的 `CustomNonbondedForce`，其排除表读出来必须**基本有序**（逆序位置占比 ≤ 5%）。
它同时覆盖所有产地与所有追加点，包括将来新写的。

判据是"基本有序"而不是"全局升序"，理由见 `MAX_DESCENT_FRACTION` 上方的说明 ——
后者在真体系上会假失败，而那段无序前缀的代价实测 ≈ 0。
"""
from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

openmm = pytest.importorskip("openmm")

# 不需要 GPU：只建 System、读排除表 / CV 名单。不标就进不了 `-m cpu_only`
# 那道常规门，防复发的意义会归零。
pytestmark = pytest.mark.cpu_only
from openmm import unit  # noqa: E402

import abfe_core as core  # noqa: E402


def _exclusions(force):
    return [
        tuple(sorted(int(x) for x in force.getExclusionParticles(i)))
        for i in range(force.getNumExclusions())
    ]


# 判据用「逆序位置占比」而不是「必须全局升序」。
#
# ⚠️ 这条曾经写成 `got == sorted(got)`，在小合成体系上通过、在**真体系上会假失败**
# ——因为一个力自己带的那段排除（例如 Group-2 力的 206 条配体内部排除，索引在
# 45313-45353 附近且彼此无序）排在前面，`sync_all_exclusions` 追加的升序尾巴跟在
# 后面，整体因此不是全局升序。真体系实测：206 条无序前缀 + 120689 条升序尾巴。
#
# 而那段前缀无关紧要：把整表再全局排一次只多拿 1.006x（≈0），修复本身已经拿到
# 1.162x。所以要断言的是**尾巴有序**，不是整表有序。
#
# 「逆序位置占比」正好分得开这两种情况：
#     修复后（有序尾巴）  : 205 / 120895 = 0.17%
#     哈希序（修复前）    : 约 50%
# 阈值取 5%，两边各有一个数量级以上的余量。
MAX_DESCENT_FRACTION = 0.05


def _descent_fraction(pairs):
    if len(pairs) < 2:
        return 0.0
    descents = sum(1 for i in range(1, len(pairs)) if pairs[i] < pairs[i - 1])
    return descents / (len(pairs) - 1)


def _assert_mostly_ascending(force, label):
    got = _exclusions(force)
    frac = _descent_fraction(got)
    assert frac <= MAX_DESCENT_FRACTION, (
        f"{label} 的排除表基本上是乱序的（逆序位置占比 {frac:.1%}，阈值 "
        f"{MAX_DESCENT_FRACTION:.0%}）。这不是整洁问题：CustomNonbondedForce 对 "
        f"addExclusion 的调用顺序敏感，乱序会让该力每步白烧时间而能量逐比特不变"
        f"（真体系实测 1.162x，见 PROPOSAL §2/§5.1）。"
        f"共 {len(got)} 条，前 8 条 = {got[:8]}"
    )


def _iter_custom_nonbonded(system):
    """遍历 System 里所有 CustomNonbondedForce，**含嵌在 CustomCVForce 内部的**。

    嵌套那一层很重要：IBS 的 16 个软核 CV 就在 CustomCVForce 里面，
    `sync_all_exclusions` 遍历 `system.getForces()` 时根本扫不到它们。
    """
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, openmm.CustomNonbondedForce):
            yield f, f"force[{i}] (group {f.getForceGroup()})"
        elif isinstance(f, openmm.CustomCVForce):
            for k in range(f.getNumCollectiveVariables()):
                cv = f.getCollectiveVariable(k)
                if isinstance(cv, openmm.CustomNonbondedForce):
                    yield cv, f"force[{i}].CV[{k}] {f.getCollectiveVariableName(k)}"


def _tiny_system(n_particles=12):
    """一个足够小、但能让 sync_all_exclusions 真的有活干的体系。"""
    system = openmm.System()
    L = 4.0
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(L, 0, 0), openmm.Vec3(0, L, 0), openmm.Vec3(0, 0, L)
    )
    for _ in range(n_particles):
        system.addParticle(12.0)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(1.0 * unit.nanometer)
    for _ in range(n_particles):
        nb.addParticle(0.0, 0.3 * unit.nanometer, 0.1 * unit.kilojoule_per_mole)
    # 刻意用**降序**加 exception，好让 union 的迭代顺序与"碰巧有序"区分开
    for i in range(n_particles - 2, -1, -1):
        nb.addException(i, i + 1, 0.0, 0.3 * unit.nanometer, 0.0)
    system.addForce(nb)
    return system


def test_sync_all_exclusions_emits_ascending_order():
    """`sync_all_exclusions` 追加的那条尾巴必须是升序。

    这是曾经出问题的那一处：`missing` 是个 set，直接迭代 = 哈希序。
    """
    system = _tiny_system()
    n = system.getNumParticles()
    custom = openmm.CustomNonbondedForce("0")
    for _ in range(n):
        custom.addParticle([])
    custom.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    custom.setCutoffDistance(1.0 * unit.nanometer)
    # 只带一条"自己的"排除，其余全靠 sync 追加 —— 正是会中招的形态
    custom.addExclusion(0, 1)
    custom.setForceGroup(2)
    system.addForce(custom)

    synced = core.sync_all_exclusions(system)
    assert synced > 0, "这个测例本该有排除对需要同步；没有的话它什么都没测到"
    _assert_mostly_ascending(custom, "sync_all_exclusions 之后的 CustomNonbondedForce")


def test_sync_all_exclusions_preserves_the_exclusion_set():
    """排序只改顺序，不改集合 —— 这是它安全的全部理由。"""
    system = _tiny_system()
    n = system.getNumParticles()
    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    expected = {
        tuple(sorted(int(x) for x in nb.getExceptionParameters(i)[:2]))
        for i in range(nb.getNumExceptions())
    }
    custom = openmm.CustomNonbondedForce("0")
    for _ in range(n):
        custom.addParticle([])
    custom.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    custom.setCutoffDistance(1.0 * unit.nanometer)
    custom.setForceGroup(2)
    system.addForce(custom)

    core.sync_all_exclusions(system)
    assert set(_exclusions(custom)) == expected


def _build_ibs_dual_or_skip(n=8):
    from abfe_pipeline import _resolve_alchemical_params
    from ibs_engine import build_ibs_dual_system

    system = _tiny_system(n)
    positions = [openmm.Vec3(0.4 + 0.12 * i, 0.5, 0.5) for i in range(n)] * unit.nanometer
    ligand = [0, 1]
    # 2026-09-09：这里原来是
    #   except (TypeError, ValueError, RuntimeError): pytest.skip(...)
    # 而下面两条端到端测试（覆盖嵌套软核 CV，本文件的全部价值所在）都走这个
    # helper。**签名变更正是最可能重新引入排除表顺序问题的重构**，被 TypeError
    # 吞成 skip 之后整套防护会静默消失、而套件仍然全绿。实测本 helper 在当前
    # 代码下正常返回，那个 except 是死代码。现在不再捕获：builder 出问题就红。
    built, _wrapper = build_ibs_dual_system(
        system=system,
        topology=None,
        perturbed_indices=ligand,
        lambdas_coul=[0.0, 0.0, 0.0],
        lambdas_vdw=[1.0, 0.5, 0.0],
        alchemical_params=_resolve_alchemical_params("softcore", None, ligand),
        potential_type="softcore",
        temperature=300.0 * unit.kelvin,
        box_vectors=system.getDefaultPeriodicBoxVectors(),
        reference_positions=positions,
        restraint_params=None,
        dispersion_protocol="legacy_uniform_density_lrc",
        environment_type="soluble",
    )
    return built


def test_every_custom_nonbonded_force_the_ibs_builder_makes_is_ascending():
    """端到端：`build_ibs_dual_system` 的产物里每一个 CustomNonbondedForce 都升序。

    覆盖顶层的 Group-2 力**和**嵌在 CustomCVForce 内部的软核 CV —— 后者
    `sync_all_exclusions` 扫不到，靠的是 `full_softcore_excl` 自己 `sorted()`。
    两条路径都要盯住。
    """
    built = _build_ibs_dual_or_skip()

    seen = 0
    for force, label in _iter_custom_nonbonded(built):
        if force.getNumExclusions() == 0:
            continue
        _assert_mostly_ascending(force, label)
        seen += 1
    assert seen > 0, (
        "builder 产物里没有任何带排除表的 CustomNonbondedForce —— "
        "这条测试因此什么都没验到，说明测例或 builder 的形态变了"
    )


def test_sync_all_exclusions_never_touches_the_nested_softcore_cvs():
    """机制断言：嵌套在 CustomCVForce 里的软核 CV，同步前后排除表**逐条不变**。

    ## 为什么上一条测试盖不住这个

    上一条断言的是**结果**（每个 CustomNonbondedForce 都升序）。可是自从
    `sync_all_exclusions` 改成 `sorted(missing)` 之后，就算软核 CV 哪天开始
    收到追加，追加进去的也**必然是升序** —— 上一条照样绿。它对"软核 CV 开始
    被 sync 波及"这个回归是**瞎的**。

    而那恰好是最可能发生的回归：`sync_all_exclusions` 只挑
    `system.getForces()` 里的 `CustomNonbondedForce`，软核 CV 嵌在 Group-1 的
    `CustomCVForce` 内部（`ibs_wrapper.get_force()` 的返回类型就是它），
    所以**结构上不可达**。谁把某个 CV 提到顶层、或新增一个粒子数匹配的顶层
    `CustomNonbondedForce`，这条保护就没了 —— 那个力会开始收到全系统的追加尾巴。

    ⚠️ 曾经有个流传的说法是"软核 CV 免疫是因为 `full_softcore_excl` 本来就是
    全系统并集 ⟹ `missing` 为空集"。**那是错的**：对它们 `missing` 从未被
    计算过。照那个理由，把 CV 提到顶层会被误判为安全。本测试钉的是真机制。
    """
    built = _build_ibs_dual_or_skip()

    nested = [
        (label, _exclusions(force))
        for force, label in _iter_custom_nonbonded(built)
        if ".CV[" in label
    ]
    if not nested:
        pytest.skip("这个最小体系的 builder 产物里没有嵌套 CV，本条无从验起")

    core.sync_all_exclusions(built)

    after = {label: _exclusions(force)
             for force, label in _iter_custom_nonbonded(built) if ".CV[" in label}
    for label, before in nested:
        assert after[label] == before, (
            f"{label} 的排除表被 sync_all_exclusions 改了 —— 嵌套 CV 本该"
            f"结构上不可达。要么有 CV 被提到了顶层，要么 sync 开始递归进 "
            f"CustomCVForce。前者会让那个力每步多背全系统的排除表；"
            f"后者会破坏 `current_excl != template_excl` 那道正确性检查。"
            f"（{len(before)} 条 → {len(after[label])} 条）"
        )
