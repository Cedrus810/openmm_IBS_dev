"""ATT-11 / P1-13 / P1-14 回归。

这三条都曾以"会毁掉已采数据"为由押后。2026-07-27 定位到 P0-10（Boresch 已提交
平衡值陈旧）之后，复合物腿本来就必须整条重跑，那个理由不再成立，于是一并修掉。
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent


# ===========================================================================
# ATT-11：Boresch 锚点必须用真实成键关系，不能用 0.22 nm 几何阈值冒充
# ===========================================================================
#
# 原实现 `距离 <= 0.22 nm` 有三个问题：
#   1. 区分不了成键与非键近接——蛋白侧 haystack 是预筛的锚点名子集
#      （CA/CB/C/N/O），CA-CB≈0.153 / CA-C≈0.152 是真键，但**非键**的
#      i/i+1 残基间 C-N≈0.133 / CA…N≈0.146 同样落在阈值内；
#   2. 漏掉长键（S-S≈0.205 贴边，金属配位普遍超过 0.22）；
#   3. 只看第 0 帧，一次热涨落就翻转键拓扑。
# 锚点三元组由这张邻接表枚举，选错直接改变解析释放修正。


class _FakeAtom:
    def __init__(self, index):
        self.index = index


class _FakeTopologyPropertyBonds:
    """mdtraj 风格：bonds 是 property。"""
    def __init__(self, pairs):
        self._pairs = [(_FakeAtom(i), _FakeAtom(j)) for i, j in pairs]

    @property
    def bonds(self):
        return iter(self._pairs)


class _FakeTopologyMethodBonds:
    """OpenMM 风格：bonds 是方法。"""
    def __init__(self, pairs):
        self._pairs = [(_FakeAtom(i), _FakeAtom(j)) for i, j in pairs]

    def bonds(self):
        return iter(self._pairs)


def _estimator():
    from abfe_core import GeometricRestraintEstimator
    return GeometricRestraintEstimator()


@pytest.mark.parametrize("topo_cls", [_FakeTopologyPropertyBonds, _FakeTopologyMethodBonds])
def test_bond_adjacency_reads_both_mdtraj_and_openmm_shapes(topo_cls):
    """mdtraj 的 Topology.bonds 是 property，OpenMM 的是方法，两种都要能读。"""
    adj = _estimator()._build_bond_adjacency(topo_cls([(0, 1), (1, 2)]))
    assert adj is not None
    assert adj[1] == {0, 2}
    assert adj[0] == {1}


def test_bond_adjacency_is_none_when_topology_has_no_bonds():
    assert _estimator()._build_bond_adjacency(_FakeTopologyPropertyBonds([])) is None
    assert _estimator()._build_bond_adjacency(None) is None


def test_topology_bonds_beat_the_geometric_threshold():
    """核心断言：真实成键关系必须压过几何距离。

    构造：原子 0 与 1 相距 0.30 nm 但**成键**（长键，几何阈值会漏）；
          原子 0 与 2 相距 0.13 nm 但**不成键**（残基间 C-N 的典型距离，
          几何阈值会误判成键）。
    """
    import numpy as np

    est = _estimator()
    ref_xyz = np.array([[0.0, 0.0, 0.0], [0.30, 0.0, 0.0], [0.13, 0.0, 0.0]])
    haystack = np.array([0, 1, 2])
    adj = est._build_bond_adjacency(_FakeTopologyPropertyBonds([(0, 1)]))

    by_topology = est._find_bonded_neighbors(0, haystack, ref_xyz, adj)
    assert by_topology == [1], "长键漏了 / 非键近接被当成键"

    by_geometry = est._find_bonded_neighbors(0, haystack, ref_xyz, None)
    assert by_geometry == [2], (
        "这条描述的是**旧行为**：几何阈值恰好把真键(0.30)漏掉、把非键(0.13)收进来。"
        "它存在的意义是证明两条路径确实不同——如果这条开始失败，说明回退路径被改了"
    )


def test_geometric_fallback_can_be_disabled():
    from abfe_core import GeometricRestraintEstimator
    est = GeometricRestraintEstimator(allow_geometric_bond_fallback=False)
    assert est.allow_geometric_bond_fallback is False


def test_bond_source_is_recorded_for_audit():
    """诊断里必须能看出用的是真实键还是几何近似。"""
    src = (REPO_ROOT / "abfe_core.py").read_text(encoding="utf-8")
    assert '"bond_source": self.bond_source' in src, (
        "锚点是靠成键关系枚举的，用了哪套判据必须落盘，否则事后无法复核"
    )


def test_anchor_enumeration_checks_per_side_bond_coverage():
    """拓扑有键但配体一侧没有键（.gro + 非标准残基）必须被发现。

    否则 lig_nei 全空、枚举不出组合，最后报成一句无关的"未找到锚点-配体接触对"。
    """
    src = (REPO_ROOT / "abfe_core.py").read_text(encoding="utf-8")
    tree = ast.parse(src, filename="abfe_core.py")
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_generate_anchor_combos"
    )
    body = ast.get_source_segment(src, fn) or ""
    assert "受体锚点" in body and "配体重原子" in body, "缺逐侧键覆盖度检查"


# ===========================================================================
# P1-13：灾难回退必须同步截断三份 history
# ===========================================================================


def _sampler(n):
    return SimpleNamespace(
        energy_history=list(range(n)),
        bias_history=list(range(n)),
        base_energy_history=list(range(n)),
    )


def test_history_lengths_requires_all_three_equal():
    from ibs_engine import _production_history_lengths

    assert _production_history_lengths(_sampler(7)) == 7

    bad = _sampler(7)
    bad.bias_history.append(999)
    with pytest.raises(RuntimeError, match="三份长度不一致"):
        _production_history_lengths(bad)


def test_truncate_drops_the_abandoned_branch_from_all_three():
    from ibs_engine import _production_history_lengths, _truncate_production_history

    s = _sampler(10)
    dropped = _truncate_production_history(s, 4)
    assert dropped == 6
    assert _production_history_lengths(s) == 4
    assert s.energy_history == [0, 1, 2, 3]
    assert s.bias_history == [0, 1, 2, 3]
    assert s.base_energy_history == [0, 1, 2, 3]


@pytest.mark.parametrize("keep", [10, 11, 999])
def test_truncate_is_a_noop_when_nothing_to_drop(keep):
    from ibs_engine import _truncate_production_history

    s = _sampler(10)
    assert _truncate_production_history(s, keep) == 0
    assert len(s.energy_history) == 10


def test_truncate_to_zero_is_allowed():
    from ibs_engine import _production_history_lengths, _truncate_production_history

    s = _sampler(5)
    assert _truncate_production_history(s, 0) == 5
    assert _production_history_lengths(s) == 0


def test_backup_length_is_refreshed_wherever_the_position_backup_is():
    """坐标备份与 history 长度必须成对更新，否则回退会截到错的分叉点。"""
    src = (REPO_ROOT / "ibs_engine.py").read_text(encoding="utf-8")
    lines = src.splitlines()
    assign_pos = [
        i for i, l in enumerate(lines)
        if "production_pos_backup =" in l and "pos_backup = production_pos_backup" not in l
    ]
    assign_len = [
        i for i, l in enumerate(lines) if "production_history_backup_len =" in l
    ]
    assert len(assign_pos) >= 3, f"坐标备份赋值点只找到 {len(assign_pos)} 处"
    assert len(assign_len) == len(assign_pos), (
        f"坐标备份赋值 {len(assign_pos)} 处、history 长度赋值 {len(assign_len)} 处，"
        "两者必须一一对应"
    )
    # 容差放到 12 行：两条赋值之间隔着解释性注释是正常的（初始化那处就隔了 7 行），
    # 这条要抓的是"新增/挪动了坐标备份点却忘了配套刷新长度"，不是行距本身。
    for p in assign_pos:
        assert any(abs(p - a) <= 12 for a in assign_len), (
            f"ibs_engine.py:{p + 1} 附近刷新了 production_pos_backup 却没在近处同步刷新 "
            "production_history_backup_len"
        )


def test_both_rollback_sites_truncate():
    src = (REPO_ROOT / "ibs_engine.py").read_text(encoding="utf-8")
    assert src.count("_truncate_production_history(") >= 3, (
        "两处灾难回退（主循环 + 余数补齐）都必须调用截断，加上定义共 ≥3 次出现"
    )


# ===========================================================================
# P1-14：整分子 PBC 修复必须在第一次建 Context 之前，且失败要 fail closed
# ===========================================================================


def test_repair_runs_before_pre_equilibration():
    """`repair_pbc_molecule_integrity` 必须在 pre_equilibrate 体内、且在建 Context 前。"""
    src = (REPO_ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(src, filename="abfe_pipeline.py")
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "pre_equilibrate"
    )
    calls = [
        n.func.attr for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert "repair_pbc_molecule_integrity" in calls, (
        "pre_equilibrate 里没有调用 PBC 整分子修复——跨盒断裂的分子会先进最小化/NPT，"
        "被最小化固化进相对坐标（P1-14）"
    )


def test_repair_fails_closed_instead_of_degrading_to_a_com_shift():
    """失败必须 raise，不能回退到 _wrap_ligand_to_box（那只是质心平移）。"""
    src = (REPO_ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(src, filename="abfe_pipeline.py")
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "repair_pbc_molecule_integrity"
    )
    handlers = [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]
    assert handlers, "没有异常处理？"
    for h in handlers:
        assert any(isinstance(n, ast.Raise) for n in ast.walk(h)), (
            "except 分支没有 raise——修复失败会被静默降级"
        )
        called = {
            n.func.attr for n in ast.walk(h)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert "_wrap_ligand_to_box" not in called, (
            "又退回 _wrap_ligand_to_box 了；它自己的 docstring 就写着"
            "'仅做整体刚性平移'，修不了跨盒断裂"
        )


def test_the_false_molecular_integrity_claim_is_gone():
    """runabfe 不得再**声称**修复了分子完整性——center_system_rigidly 只平移质心。

    用 AST 只看真正被打印出去的字符串字面量，不做裸子串匹配：源码注释里解释
    "原来这行写的是……" 是应该允许的，被匹配到就成了误报。
    """
    src = (REPO_ROOT / "runabfe.py").read_text(encoding="utf-8")
    tree = ast.parse(src, filename="runabfe.py")

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        if name not in {"info", "warning", "error", "debug", "print", "_log"}:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if "分子完整性修复完毕" in arg.value:
                    offenders.append(node.lineno)
    assert not offenders, (
        f"runabfe.py 第 {offenders} 行仍在声称'分子完整性修复完毕'，"
        "但那里只调用了 center_system_rigidly（整体质心平移），修不了跨盒断裂"
    )
