"""RBFE R1a：分子图、片段分解与 A→B 原子映射。

设计依据：`docs/design/PLAN_rbfe_interface_and_implementation.md` §5.2 与
`docs/design/PROPOSAL_rbfe_r1_fragment_mapping.md`（路线 **A+B**，2026-09-03 拍板）。

计划 §8 的 R1 验收标准第一条：**映射可审计**。本文件按 §5.2 的五条硬要求组织：

  1. 分子内 / 全局 / hybrid 索引分别记录；
  2. 一一对应、索引范围、化学一致性、**核心连通性**；
  3. 识别对称等价映射与姿势歧义；
  4. 两腿共用同一份冻结映射，各自投影为全局索引；
  5. 映射评分只用于候选排序，不代替化学与几何验收。

不 import openmm、不建 System、不启动 GPU。rdkit 只在标了
`requires_rdkit` 的用例里出现——没装 rdkit 时那些用例 skip，其余全部照跑，
因为保守降级路径本身也必须被测到。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import rbfe_core as rc
import runrbfe

REPO_ROOT = Path(__file__).resolve().parent.parent
LIGAND_ITP = REPO_ROOT / "tests" / "fixtures" / "memtest" / "Atenolol-rank11.itp"
MOLECULETYPE = "Atenolol-rank11"

try:  # pragma: no cover - 取决于运行环境
    import rdkit  # noqa: F401

    HAVE_RDKIT = True
except ImportError:  # pragma: no cover
    HAVE_RDKIT = False

requires_rdkit = pytest.mark.skipif(not HAVE_RDKIT, reason="需要 rdkit（路线 A+B 的 M4 步）")


# ---------------------------------------------------------------------------
# 夹具与小工具
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ligand_a() -> rc.MolecularGraph:
    """仓库里那份**真实**配体（Atenolol），不是构造出来的玩具图。"""
    return rc.MolecularGraph.from_gromacs_itp(str(LIGAND_ITP), MOLECULETYPE)


def _methylate(graph: rc.MolecularGraph, heavy_atom: int):
    """图手术：把 `heavy_atom` 上的一个氢换成甲基，得到配体 B。

    这是 RBFE 的经典测例（H→CH3）：公共核心应当是除那个氢以外的**全部**原子，
    A-only 恰好一个氢，B-only 恰好一个碳三个氢。它同时是 dummy 处理的最小压力
    测试——两侧都有 dummy。

    只动**图**，不编造力场参数：本层的输入就是键图，凭空写一份假参数反而会让
    人误以为这份 B 可以直接拿去跑（真实的 B 需要重新给电荷与成键参数，那是
    R1b/R3 的事）。
    """
    hydrogens = [i for i in graph.neighbors(heavy_atom) if not graph.atom(i).is_heavy]
    assert hydrogens, f"原子 {heavy_atom} 上没有氢"
    dropped = hydrogens[0]
    top = max(graph.indices)
    atoms = [a for a in graph.atoms if a.index != dropped]
    atoms.append(rc.AtomNode(index=top + 1, atomic_number=6, name="C_new", residue_name="MOL"))
    atoms.extend(
        rc.AtomNode(index=top + 1 + k, atomic_number=1, name=f"H_new{k}", residue_name="MOL")
        for k in (1, 2, 3)
    )
    bonds = [b for b in graph.bonds if dropped not in b]
    bonds.append((heavy_atom, top + 1))
    bonds.extend((top + 1, top + 1 + k) for k in (1, 2, 3))
    return rc.MolecularGraph.from_atoms_and_bonds(atoms, bonds), dropped


def _chain(elements, extra_bonds=()):
    """一条直链小分子，用于负例。elements 是原子序数序列。"""
    atoms = [
        rc.AtomNode(index=i, atomic_number=z, name=f"X{i}") for i, z in enumerate(elements)
    ]
    bonds = [(i, i + 1) for i in range(len(elements) - 1)] + list(extra_bonds)
    return rc.MolecularGraph.from_atoms_and_bonds(atoms, bonds)


# ---------------------------------------------------------------------------
# M0：建图
# ---------------------------------------------------------------------------


def test_real_ligand_graph_parses(ligand_a):
    assert ligand_a.n_atoms == 41
    assert len(ligand_a.bonds) == 41
    assert ligand_a.element_counts() == {"C": 14, "H": 22, "N": 2, "O": 3}


def test_local_index_is_dense_and_ordered(ligand_a):
    """分子内索引必须是 0..n-1 且按来源索引升序——它是映射对外的唯一坐标系。"""
    locals_ = [ligand_a.local_index(i) for i in ligand_a.indices]
    assert locals_ == list(range(ligand_a.n_atoms))


def test_bonds_are_canonicalised(ligand_a):
    for left, right in ligand_a.bonds:
        assert left < right
    assert list(ligand_a.bonds) == sorted(ligand_a.bonds)


def test_disconnected_graph_is_rejected():
    """不连通 = 共价配体或 ligand_indices 划错，首版拒绝（计划 §2）。"""
    atoms = [rc.AtomNode(index=i, atomic_number=6) for i in range(4)]
    with pytest.raises(rc.RBFEMappingError, match="不连通"):
        rc.MolecularGraph.from_atoms_and_bonds(atoms, [(0, 1), (2, 3)])


def test_bond_to_outside_atom_is_rejected():
    atoms = [rc.AtomNode(index=i, atomic_number=6) for i in range(3)]
    with pytest.raises(rc.RBFEMappingError, match="端点不在本图"):
        rc.MolecularGraph.from_atoms_and_bonds(atoms, [(0, 1), (1, 99)])


def test_duplicate_atom_index_is_rejected():
    atoms = [rc.AtomNode(index=0, atomic_number=6), rc.AtomNode(index=0, atomic_number=1)]
    with pytest.raises(rc.RBFEMappingError, match="重复原子索引"):
        rc.MolecularGraph.from_atoms_and_bonds(atoms, [])


def test_self_bond_is_rejected():
    atoms = [rc.AtomNode(index=i, atomic_number=6) for i in range(2)]
    with pytest.raises(rc.RBFEMappingError, match="自环"):
        rc.MolecularGraph.from_atoms_and_bonds(atoms, [(0, 0), (0, 1)])


def test_unknown_element_raises_instead_of_guessing():
    node = rc.AtomNode(index=0, atomic_number=118, name="Og")
    with pytest.raises(rc.RBFEMappingError, match="不在已知元素表"):
        _ = node.element


def test_itp_without_atomtypes_is_rejected(tmp_path):
    """拿不到 at.num 就报错，**不按质量或原子名猜元素**。"""
    path = tmp_path / "no_atomtypes.itp"
    path.write_text(
        "[ moleculetype ]\nLIG 3\n\n[ atoms ]\n"
        "1 zz 1 LIG C1 1 0.0 12.011\n2 zz 1 LIG C2 2 0.0 12.011\n\n"
        "[ bonds ]\n1 2 1\n",
        encoding="utf-8",
    )
    with pytest.raises(rc.RBFEMappingError, match="at.num"):
        rc.MolecularGraph.from_gromacs_itp(str(path))


def test_itp_missing_moleculetype_is_rejected(tmp_path):
    path = tmp_path / "empty.itp"
    path.write_text("[ atomtypes ]\nc3 6 12.0 0.0 A 0.3 0.4\n", encoding="utf-8")
    with pytest.raises(rc.RBFEMappingError, match="没有找到"):
        rc.MolecularGraph.from_gromacs_itp(str(path))


# ---------------------------------------------------------------------------
# M1：环分析（R0 的 unchecked 就是在等这一层）
# ---------------------------------------------------------------------------


def test_benzene_ring_detected_on_real_ligand(ligand_a):
    assert len(ligand_a.ring_bonds()) == 6
    assert ligand_a.ring_size_profile() == (6,) * 6
    assert len(ligand_a.ring_atoms()) == 6


def test_acyclic_molecule_has_no_ring(ligand_a):
    chain = _chain([6, 6, 6, 6])
    assert chain.ring_bonds() == frozenset()
    assert chain.ring_size_profile() == ()


def test_ring_size_is_the_smallest_cycle():
    """五元环、六元环要能被区分开——环尺寸变化靠这个判据。"""
    five = _chain([6] * 5, extra_bonds=[(0, 4)])
    six = _chain([6] * 6, extra_bonds=[(0, 5)])
    assert five.ring_size_profile() == (5,) * 5
    assert six.ring_size_profile() == (6,) * 6


# ---------------------------------------------------------------------------
# M2：片段分解
# ---------------------------------------------------------------------------


def test_fragment_decomposition_of_real_ligand(ligand_a):
    decomposition = rc.decompose_into_fragments(ligand_a)
    # 片段数是个位数 —— 提案 §3 路线 A 的核心卖点：人眼能审
    assert 1 < len(decomposition.fragments) < 20
    covered = sorted(
        index for fragment in decomposition.fragments for index in fragment.atom_indices
    )
    assert covered == sorted(ligand_a.indices), "片段必须是原子集合的一个划分"


def test_ring_bonds_are_never_cut(ligand_a):
    decomposition = rc.decompose_into_fragments(ligand_a)
    assert not (set(decomposition.cut_bonds) & ligand_a.ring_bonds())


def test_terminal_bonds_are_never_cut(ligand_a):
    """切末端键只会得到一堆单原子片段，人眼没法审。"""
    for left, right in rc.decompose_into_fragments(ligand_a).cut_bonds:
        assert len([i for i in ligand_a.heavy_neighbors(left) if i != right]) >= 1
        assert len([i for i in ligand_a.heavy_neighbors(right) if i != left]) >= 1


def test_hydrogen_bonds_are_never_cut(ligand_a):
    for left, right in rc.decompose_into_fragments(ligand_a).cut_bonds:
        assert ligand_a.atom(left).is_heavy and ligand_a.atom(right).is_heavy


def test_decomposition_is_deterministic(ligand_a):
    """片段编号进映射身份指纹，非确定性会让指纹漂移。"""
    first = rc.decompose_into_fragments(ligand_a).to_dict()
    second = rc.decompose_into_fragments(ligand_a).to_dict()
    assert first == second


def test_whole_benzene_stays_in_one_fragment(ligand_a):
    decomposition = rc.decompose_into_fragments(ligand_a)
    ring = ligand_a.ring_atoms()
    owners = {decomposition.fragment_of_atom[i] for i in ring}
    assert len(owners) == 1


# ---------------------------------------------------------------------------
# M3-M5：映射
# ---------------------------------------------------------------------------


def test_self_mapping_is_identity(ligand_a):
    """A→A 必须是恒等映射、零 dummy。

    这是计划 §8 里 R3「A→A 为零」那条验收在**映射层**的前提：映射都不恒等，
    自由能不可能为零。
    """
    mapping = rc.map_atoms(ligand_a, ligand_a)
    assert mapping.n_core == ligand_a.n_atoms
    assert mapping.a_only == () and mapping.b_only == ()
    assert all(a == b for a, b in mapping.core_pairs)
    assert mapping.method == "fragment_isomorphism"


def test_self_mapping_passes_validation(ligand_a):
    mapping = rc.map_atoms(ligand_a, ligand_a)
    assert rc.validate_mapping(mapping, ligand_a, ligand_a).ok


@requires_rdkit
def test_h_to_methyl_gives_minimal_dummies(ligand_a):
    """H→CH3：只有被换掉的那个氢是 A-only，只有新甲基是 B-only。

    这条曾经**失败过**，而且失败得很隐蔽：第一版按片段签名生长，苯环换了取代基
    之后签名变了，生长就在苯环处断掉，苯环**后面**那一整段（酰胺尾，A/B 完全
    相同）被整段判成 dummy——公共核心 32 而不是 40，验证还照样 PASS。
    生长改成按接点原子对应之后才对。别把这条测试放宽。
    """
    graph_b, dropped = _methylate(ligand_a, 13)
    mapping = rc.map_atoms(ligand_a, graph_b)

    assert mapping.method == "fragment_isomorphism+rdkit_mcs"
    assert mapping.n_core == ligand_a.n_atoms - 1 == 40
    assert len(mapping.a_only) == 1
    assert len(mapping.b_only) == 4

    dropped_local = ligand_a.local_index(dropped)
    assert mapping.a_only == (dropped_local,)
    b_index = {graph_b.local_index(i): i for i in graph_b.indices}
    b_only_elements = sorted(graph_b.atom(b_index[i]).element for i in mapping.b_only)
    assert b_only_elements == ["C", "H", "H", "H"]


@requires_rdkit
def test_h_to_methyl_mapping_passes_validation(ligand_a):
    graph_b, _ = _methylate(ligand_a, 13)
    mapping = rc.map_atoms(ligand_a, graph_b)
    report = rc.validate_mapping(mapping, ligand_a, graph_b)
    assert report.ok, report.render()


def test_conservative_fallback_is_labelled_not_silent(ligand_a):
    """关掉 rdkit 必须**写进 method**。静默降级会让两次运行给出不同映射却看不出来。"""
    graph_b, _ = _methylate(ligand_a, 13)
    mapping = rc.map_atoms(ligand_a, graph_b, allow_mcs=False)

    assert "conservative" in mapping.method
    assert mapping.n_core < ligand_a.n_atoms - 1
    assert any("整块进 dummy" in note for note in mapping.ambiguities)


def test_conservative_fallback_on_mid_molecule_change_is_rejected(ligand_a):
    """差异在分子**中部**时，保守路径给不出可用映射——这条是实测踩出来的。

    苯环上换取代基 → 关掉 MCS 时整个苯环片段进 dummy → 公共核心被从中间切断
    （前半段和酰胺尾各成一块）。断开的核心让 dummy 贡献无法证明抵消，
    `validate_mapping` 必须拒绝，而不是给一份"能跑但错"的映射。

    结论：差异基团不在末端时，rdkit MCS 不是优化项而是**必需**。
    """
    graph_b, _ = _methylate(ligand_a, 13)
    mapping = rc.map_atoms(ligand_a, graph_b, allow_mcs=False)
    report = rc.validate_mapping(mapping, ligand_a, graph_b)

    assert not report.ok
    assert any("不连通" in e for e in report.errors)
    assert any("启用 rdkit MCS" in e for e in report.errors), "错误信息要告诉人怎么办"


def test_conservative_fallback_works_when_change_is_terminal(ligand_a):
    """差异在**末端**片段时，保守路径本身就是一份合法映射（核心仍连通）。"""
    graph_b, _ = _methylate(ligand_a, 1)     # 异丙基末端甲基，片段图上是叶子
    mapping = rc.map_atoms(ligand_a, graph_b, allow_mcs=False)
    report = rc.validate_mapping(mapping, ligand_a, graph_b)

    assert "conservative" in mapping.method
    assert report.ok, report.render()
    assert any("保守降级" in w for w in report.warnings)


def test_conservative_fallback_still_partitions_all_atoms(ligand_a):
    graph_b, _ = _methylate(ligand_a, 1)
    mapping = rc.map_atoms(ligand_a, graph_b, allow_mcs=False)
    assert len(mapping.core_pairs) + len(mapping.a_only) == mapping.n_atoms_a
    assert len(mapping.core_pairs) + len(mapping.b_only) == mapping.n_atoms_b


def test_mapping_is_deterministic(ligand_a):
    graph_b, _ = _methylate(ligand_a, 13)
    first = rc.map_atoms(ligand_a, graph_b)
    second = rc.map_atoms(ligand_a, graph_b)
    assert first.fingerprint() == second.fingerprint()
    assert first.core_pairs == second.core_pairs


def test_fingerprint_changes_when_mapping_changes(ligand_a):
    """换了映射就必须换指纹——否则旧采样产物会被错误复用（计划 §7）。"""
    graph_b, _ = _methylate(ligand_a, 13)
    with_mcs = rc.map_atoms(ligand_a, graph_b, allow_mcs=True)
    without = rc.map_atoms(ligand_a, graph_b, allow_mcs=False)
    if with_mcs.core_pairs != without.core_pairs:
        assert with_mcs.fingerprint() != without.fingerprint()


def test_fingerprint_ignores_cosmetic_atom_names(ligand_a):
    """改原子名不该让已经跑完的采样作废——它是审计信息，不是身份。"""
    renamed_atoms = [
        rc.AtomNode(
            index=a.index,
            atomic_number=a.atomic_number,
            name=a.name + "_renamed",
            residue_name=a.residue_name,
            residue_index=a.residue_index,
            chain_index=a.chain_index,
        )
        for a in ligand_a.atoms
    ]
    renamed = rc.MolecularGraph.from_atoms_and_bonds(renamed_atoms, ligand_a.bonds)
    assert rc.map_atoms(ligand_a, ligand_a).fingerprint() == rc.map_atoms(
        renamed, renamed
    ).fingerprint()


def test_symmetry_equivalent_solutions_are_reported(ligand_a):
    """对称等价解要被**识别并记录**，不是识别出来再静默挑一个（计划 §5.2）。"""
    mapping = rc.map_atoms(ligand_a, ligand_a)
    assert mapping.symmetry_solution_counts, "Atenolol 有甲基/苯环对称，不可能一个都没有"
    for _, _, count in mapping.symmetry_solution_counts:
        assert count > 1


def test_no_seed_fragment_is_fail_closed():
    """找不到唯一种子就拒绝，不从多个候选里猜一个。"""
    graph = _chain([6] * 6, extra_bonds=[(0, 5)])   # 苯环骨架：只有一个片段且高度对称
    other = _chain([6] * 5, extra_bonds=[(0, 4)])
    with pytest.raises(rc.RBFEMappingError):
        rc.map_atoms(graph, other)


# ---------------------------------------------------------------------------
# 索引：分子内 / 全局 / hybrid 分别记录（计划 §5.2）
# ---------------------------------------------------------------------------


def test_hybrid_indices_are_contiguous_core_first(ligand_a):
    graph_b, _ = _methylate(ligand_a, 13)
    mapping = rc.map_atoms(ligand_a, graph_b, allow_mcs=False)
    hybrid = mapping.hybrid_indices()
    total = mapping.n_core + len(mapping.a_only) + len(mapping.b_only)
    assert hybrid["n_hybrid_atoms"] == total
    assert sorted(set(hybrid["a"].values()) | set(hybrid["b"].values())) == list(range(total))
    for local_a, local_b in mapping.core_pairs:
        assert hybrid["a"][local_a] == hybrid["b"][local_b], "core 原子在 hybrid 里是同一个粒子"


def test_projection_to_two_legs_uses_one_frozen_mapping(ligand_a):
    """两腿共用同一份分子级映射，只是全局索引不同（计划 §5.2）。"""
    mapping = rc.map_atoms(ligand_a, ligand_a)
    n = mapping.n_atoms_a
    complex_leg = mapping.project(range(5000, 5000 + n), range(5000, 5000 + n))
    solvent_leg = mapping.project(range(10, 10 + n), range(10, 10 + n))

    assert complex_leg["core_pairs"][0] == (5000, 5000)
    assert solvent_leg["core_pairs"][0] == (10, 10)
    # 全局索引不同，但**分子级对应关系**逐位相同
    assert [
        (a - 5000, b - 5000) for a, b in complex_leg["core_pairs"]
    ] == [(a - 10, b - 10) for a, b in solvent_leg["core_pairs"]]


def test_projection_rejects_wrong_atom_count(ligand_a):
    mapping = rc.map_atoms(ligand_a, ligand_a)
    with pytest.raises(rc.RBFEMappingError, match="投影失败"):
        mapping.project(range(10), range(mapping.n_atoms_b))


def test_projection_rejects_duplicate_global_indices(ligand_a):
    mapping = rc.map_atoms(ligand_a, ligand_a)
    n = mapping.n_atoms_a
    with pytest.raises(rc.RBFEMappingError, match="重复"):
        mapping.project([0] * n, range(n))


def test_atom_mapping_json_is_auditable(ligand_a):
    """`atom_mapping.json` 要能被人读懂——这是 R1 验收「映射可审计」的落点。"""
    graph_b, _ = _methylate(ligand_a, 13)
    payload = rc.map_atoms(ligand_a, graph_b, allow_mcs=False).to_dict()
    for key in (
        "rbfe_mapping_protocol_version",
        "method",
        "core_pairs_molecule_local",
        "A_only_molecule_local",
        "B_only_molecule_local",
        "fragment_pairs",
        "symmetry_solution_counts",
        "ambiguities",
        "hybrid_indices",
        "atom_identity_A",
    ):
        assert key in payload
    json.dumps(payload)  # 必须可序列化
    assert payload["atom_identity_A"][0]["element"] in {"C", "H", "N", "O"}


# ---------------------------------------------------------------------------
# M6：映射验证
# ---------------------------------------------------------------------------


def _mapping(core_pairs, n_a, n_b, a_only=(), b_only=(), method="fragment_isomorphism"):
    return rc.AtomMapping(
        protocol_version=rc.RBFE_MAPPING_PROTOCOL_VERSION,
        n_atoms_a=n_a,
        n_atoms_b=n_b,
        core_pairs=tuple(core_pairs),
        a_only=tuple(a_only),
        b_only=tuple(b_only),
        fragment_pairs=(),
        method=method,
        symmetry_solution_counts=(),
        ambiguities=(),
        atom_identity_a=(),
        atom_identity_b=(),
    )


def test_element_change_is_rejected():
    """映射元素改变：首版拒绝（计划 §2）。"""
    graph_a = _chain([6, 6, 6, 1])
    graph_b = _chain([6, 6, 7, 1])
    mapping = _mapping([(0, 0), (1, 1), (2, 2), (3, 3)], 4, 4)
    report = rc.validate_mapping(mapping, graph_a, graph_b)
    assert not report.ok
    assert any("元素改变" in e for e in report.errors)


def test_disconnected_core_is_rejected():
    """核心不连通 ⇒ dummy 处理无法证明抵消（计划 §5.2/§5.3）。"""
    graph = _chain([6, 6, 6, 6, 6])
    mapping = _mapping([(0, 0), (4, 4)], 5, 5, a_only=(1, 2, 3), b_only=(1, 2, 3))
    report = rc.validate_mapping(mapping, graph, graph)
    assert any("不连通" in e for e in report.errors)


def test_ring_status_change_is_rejected():
    """环内原子被映到环外原子 = 环断裂/闭合。R0 把这条挂在 unchecked，这里真查掉。"""
    ring = _chain([6] * 6, extra_bonds=[(0, 5)])
    chain = _chain([6] * 6)
    mapping = _mapping([(i, i) for i in range(6)], 6, 6)
    report = rc.validate_mapping(mapping, ring, chain)
    assert any("环断裂/闭合" in e for e in report.errors)


def test_incomplete_partition_is_rejected():
    graph = _chain([6, 6, 6])
    mapping = _mapping([(0, 0)], 3, 3)          # 漏了 1、2，且没进 a_only/b_only
    report = rc.validate_mapping(mapping, graph, graph)
    assert any("完全划分" in e for e in report.errors)


def test_empty_core_is_rejected():
    graph = _chain([6, 6, 6])
    mapping = _mapping([], 3, 3, a_only=(0, 1, 2), b_only=(0, 1, 2))
    report = rc.validate_mapping(mapping, graph, graph)
    assert any("公共核心为空" in e for e in report.errors)


def test_non_injective_mapping_is_rejected():
    graph = _chain([6, 6, 6])
    mapping = _mapping([(0, 0), (1, 0), (2, 2)], 3, 3)
    report = rc.validate_mapping(mapping, graph, graph)
    assert any("一一对应" in e for e in report.errors)


def test_out_of_range_index_is_rejected():
    graph = _chain([6, 6, 6])
    mapping = _mapping([(0, 0), (1, 1), (2, 99)], 3, 3)
    report = rc.validate_mapping(mapping, graph, graph)
    assert any("越界" in e for e in report.errors)


def test_validate_mapping_keeps_an_honest_unchecked_list(ligand_a):
    """手性、姿势、互变异构本层查不了——PASS 不等于全都查过了。"""
    mapping = rc.map_atoms(ligand_a, ligand_a)
    report = rc.validate_mapping(mapping, ligand_a, ligand_a)
    assert report.ok
    assert any("手性" in u for u in report.unchecked)
    assert any("姿势" in u for u in report.unchecked)


# ---------------------------------------------------------------------------
# 与 R0 的 validate_edge 对接
# ---------------------------------------------------------------------------


def _edge_spec():
    def endpoint(name):
        return rc.LigandEndpoint(
            name=name,
            structure="[H]c1ccccc1",
            formal_charge=0,
            input_path=f"/tmp/{name}.itp",
            input_sha256=("a" if name == "A" else "b") * 64,
            protonation_state="neutral_pH7",
            stereochemistry="S",
            partial_charge_source="from_input_topology",
        )

    return rc.EdgeSpec(
        edge_id="A_to_B",
        ligand_a=endpoint("A"),
        ligand_b=endpoint("B"),
        environment=rc.EnvironmentSpec(
            receptor_name="4W53",
            receptor_path="/tmp/rec.pdb",
            receptor_sha256="c" * 64,
            force_field="amber14sb",
            water_model="tip3p",
            ion_model="joung_cheatham",
        ),
        protocol=rc.ProtocolSpec(
            temperature_kelvin=300.0,
            pressure_bar=1.0,
            n_lambda_states=12,
            n_steps_per_state=250000,
            seed=20260903,
            lambda_schedule_name="rbfe_linear_v1",
        ),
        output_dir="/tmp/out",
    )


def test_validate_edge_without_mapping_admits_it_did_not_check(ligand_a):
    report = rc.validate_edge(_edge_spec())
    assert any("环断裂/闭合" in u for u in report.unchecked)
    assert any("需要原子映射" in u for u in report.unchecked)


def test_validate_edge_with_mapping_really_checks_rings(ligand_a):
    """给了映射，环那几条就从"没查"变成真查了。"""
    mapping = rc.map_atoms(ligand_a, ligand_a)
    report = rc.validate_edge(
        _edge_spec(), mapping=mapping, graph_a=ligand_a, graph_b=ligand_a
    )
    assert report.ok, report.render()
    assert not any("需要原子映射" in u for u in report.unchecked)


def test_validate_edge_with_bad_mapping_rejects():
    ring = _chain([6] * 6, extra_bonds=[(0, 5)])
    chain = _chain([6] * 6)
    mapping = _mapping([(i, i) for i in range(6)], 6, 6)
    report = rc.validate_edge(_edge_spec(), mapping=mapping, graph_a=ring, graph_b=chain)
    assert not report.ok
    assert any("环断裂/闭合" in e for e in report.errors)


def test_validate_edge_rejects_partial_mapping_evidence(ligand_a):
    """只给一半证据是配置错误，直接报错，不悄悄退回"没查"。"""
    mapping = rc.map_atoms(ligand_a, ligand_a)
    with pytest.raises(rc.RBFEValidationError, match="必须一起给"):
        rc.validate_edge(_edge_spec(), mapping=mapping, graph_a=ligand_a)


def test_validate_edge_detects_mapping_from_a_different_graph(ligand_a):
    small = _chain([6, 6, 6])
    mapping = rc.map_atoms(small, small)
    report = rc.validate_edge(
        _edge_spec(), mapping=mapping, graph_a=ligand_a, graph_b=ligand_a
    )
    assert any("不是从这两个图算出来的" in e for e in report.errors)


# ---------------------------------------------------------------------------
# CLI：runrbfe.py map
# ---------------------------------------------------------------------------


def test_cli_map_on_real_ligand_succeeds(tmp_path, capsys):
    out = tmp_path / "atom_mapping.json"
    code = runrbfe.main(
        [
            "map",
            "--ligand-a", str(LIGAND_ITP),
            "--ligand-b", str(LIGAND_ITP),
            "--moleculetype-a", MOLECULETYPE,
            "--moleculetype-b", MOLECULETYPE,
            "--out", str(out),
        ]
    )
    assert code == runrbfe.EXIT_OK
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_core"] == 41
    assert payload["validation"]["ok"] is True
    assert len(payload["fingerprint"]) == 64


def test_cli_map_json_stdout_is_pure_json(tmp_path, capsys):
    """`--json` 时 stdout 只能有 JSON，人类可读文本走 stderr（不然 | jq 直接炸）。"""
    runrbfe.main(
        [
            "map",
            "--ligand-a", str(LIGAND_ITP),
            "--ligand-b", str(LIGAND_ITP),
            "--moleculetype-a", MOLECULETYPE,
            "--moleculetype-b", MOLECULETYPE,
            "--json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["method"] == "fragment_isomorphism"
    assert captured.err.strip(), "人类可读文本必须存在，只是不能在 stdout"


def test_cli_map_rejects_unparameterised_input(tmp_path, capsys):
    """SDF 等需要自动参数化的路线首版拒绝（计划 §5.1），不悄悄转换。"""
    sdf = tmp_path / "ligand.sdf"
    sdf.write_text("dummy", encoding="utf-8")
    code = runrbfe.main(["map", "--ligand-a", str(sdf), "--ligand-b", str(sdf)])
    assert code == runrbfe.EXIT_REJECTED
    assert "已参数化的 GROMACS 输入" in capsys.readouterr().err


def test_cli_map_is_registered_and_documented():
    parser = runrbfe.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["map"])       # 缺必需参数
    args = parser.parse_args(
        ["map", "--ligand-a", "a.itp", "--ligand-b", "b.itp"]
    )
    assert args.func is runrbfe.cmd_map
    assert args.no_mcs is False


def test_missing_rdkit_is_reported_as_unavailable_not_disabled(ligand_a, monkeypatch):
    """rdkit 装不上和主动关掉是两回事，method 必须能分辨。

    两者都会退回保守路径，但一个是环境问题（装上 rdkit 就好）、一个是显式选择。
    混成同一个字符串，事后看产物就分不清当时到底发生了什么。
    """
    monkeypatch.setattr(rc, "_mcs_align", lambda *a, **k: {})
    graph_b, _ = _methylate(ligand_a, 1)
    mapping = rc.map_atoms(ligand_a, graph_b, allow_mcs=True)
    assert mapping.method == "fragment_isomorphism_only__mcs_unavailable_conservative"
