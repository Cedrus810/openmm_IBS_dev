"""两条腿的配体约束/HMR 身份比较：质量带容差，约束不带。

起因（4W53，2026-09-01）：两条腿是同一个配体、约束对逐位相同，但质量表来源不同 ——
complex 腿走 GROMACS 拓扑（LIG.itp，AMBER 表）C=12.010736/H=1.007941，solvent 腿走
OpenMM 元素表 C=12.010780/H=1.007947，相对差 ~5e-6。`comparison_sha256` 是逐位哈希，
于是**两条腿都算完之后**整个热力学循环在最后一步被拦下。

容差 5e-4 amu 的依据：这道门要抓的是同位素/HMR 级改动（最小 1 amu 量级），
而质量表来源差异是 5e-5 amu 量级，两者差 4 个数量级。
"""
import openmm
from openmm import unit

import runabfe as R


def _identity(masses, constraints, indices=None, tag=""):
    """手搓一份 constraint_identity（字段名与 constraint_identity_fingerprint 一致）。"""
    idx = list(indices if indices is not None else range(len(masses)))
    return {
        "version": 1,
        "ligand_indices": idx,
        "ligand_masses_amu": [[idx[k], m] for k, m in enumerate(masses)],
        "ligand_constraints": [[idx[a], idx[b], d] for a, b, d in constraints],
        "comparison_sha256": f"hash-{tag}",
        "sha256": f"exact-{tag}",
    }


GROMACS_MASSES = [12.010736, 12.010736, 1.007941, 1.007941]
OPENMM_MASSES = [12.010780, 12.010780, 1.007947, 1.007947]
CONSTRAINTS = [(0, 2, 0.109948), (1, 3, 0.109774)]


def test_mass_table_provenance_difference_is_accepted():
    """GROMACS 表 vs OpenMM 元素表：同一分子，必须放行。"""
    c = _identity(GROMACS_MASSES, CONSTRAINTS, tag="c")
    # 全局编号也不同——这正是 comparison 视图要抹平的东西
    s = _identity(OPENMM_MASSES, CONSTRAINTS, indices=[100, 101, 102, 103], tag="s")
    assert R._ligand_constraint_comparable_view(c) == R._ligand_constraint_comparable_view(s)
    assert R._assert_matching_result_constraint_identity(
        {"constraint_identity": c}, {"constraint_identity": s}, context="t"
    ) is c


def test_hmr_is_still_rejected():
    """HMR 把 H 抬到 3.024（Δ≈2 amu），必须仍然 fail-closed。"""
    c = _identity(GROMACS_MASSES, CONSTRAINTS, tag="c")
    hmr = [12.010736 - 2 * 2.016, 12.010736 - 2 * 2.016, 3.024, 3.024]
    s = _identity(hmr, CONSTRAINTS, tag="s")
    try:
        R._assert_matching_result_constraint_identity(
            {"constraint_identity": c}, {"constraint_identity": s}, context="t"
        )
    except RuntimeError as exc:
        assert "3.024" in str(exc)  # 报错要说清差在哪，不能只打两个 sha256
        return
    raise AssertionError("HMR 必须被拦下")


def test_deuteration_is_still_rejected():
    """氘代 Δ≈1 amu，是这道门要抓的最小真实改动。"""
    c = _identity(GROMACS_MASSES, CONSTRAINTS, tag="c")
    s = _identity([12.010736, 12.010736, 2.014, 2.014], CONSTRAINTS, tag="s")
    assert R._ligand_constraint_comparable_view(c) != R._ligand_constraint_comparable_view(s)


def test_changed_constraint_distance_is_rejected_with_no_tolerance():
    """约束距离不给任何容差：约束集合变了就是不同的 constrained Hamiltonian。"""
    c = _identity(GROMACS_MASSES, CONSTRAINTS, tag="c")
    s = _identity(
        GROMACS_MASSES, [(0, 2, 0.109948 + 1e-6), (1, 3, 0.109774)], tag="s"
    )
    assert R._ligand_constraint_comparable_view(c) != R._ligand_constraint_comparable_view(s)


def test_missing_constraint_pair_is_rejected():
    c = _identity(GROMACS_MASSES, CONSTRAINTS, tag="c")
    s = _identity(GROMACS_MASSES, CONSTRAINTS[:1], tag="s")
    assert R._ligand_constraint_comparable_view(c) != R._ligand_constraint_comparable_view(s)


def test_tolerance_sits_between_provenance_noise_and_real_changes():
    """容差必须夹在「质量表噪声」和「最小真实改动」之间，两边都留出量级余量。"""
    tol = 10 ** -R._LIGAND_MASS_COMPARISON_DECIMALS
    provenance_noise = 6.0e-06   # 实测 4W53 H 原子
    smallest_real = 1.0          # 氘代
    assert provenance_noise * 10 < tol < smallest_real / 100


def test_live_systems_agree_when_only_the_mass_table_differs():
    """端到端：只有质量表不同的两个真实 OpenMM System 必须放行，改了 HMR 必须拦。"""
    def _mk(mass_h):
        sysm = openmm.System()
        for m in (12.010736, 12.010736, mass_h, mass_h):
            sysm.addParticle(m * unit.dalton)
        sysm.addConstraint(0, 2, 0.109948 * unit.nanometer)
        sysm.addConstraint(1, 3, 0.109774 * unit.nanometer)
        return sysm

    lig = [0, 1, 2, 3]
    R._assert_matching_ligand_constraint_identity(
        _mk(1.007941), lig, _mk(1.007947), lig, context="t"
    )
    try:
        R._assert_matching_ligand_constraint_identity(
            _mk(1.007941), lig, _mk(3.024), lig, context="t"
        )
    except RuntimeError:
        return
    raise AssertionError("HMR 必须被拦下")
