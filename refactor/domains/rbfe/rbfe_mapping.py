# ---------------------------------------------------------------------------
# R1a：分子图、环分析与片段分解
# ---------------------------------------------------------------------------
#
# 计划 §5.2 对映射的硬要求（逐条）：
#   - 分子内索引、complex 全局索引、solvent 全局索引和 hybrid 索引分别记录；
#   - 验证一对一、索引范围、化学一致性和**映射核心的连通性**；
#   - 识别对称等价映射与姿势歧义；
#   - 两腿使用同一份**冻结**的分子级 A→B 映射，再各自投影为全局原子索引；
#   - 映射评分只用于候选排序，不能代替化学与几何验收。
#
# 路线：`docs/design/PROPOSAL_rbfe_r1_fragment_mapping.md` §3 的 **A+B 混合**
# （2026-09-03 用户拍板）——在非环可旋转键处切开，先做片段级匹配（可人眼审），
# 片段内部再做原子级对齐。
#
# ## 为什么这一层不 import openmm、也不 import rdkit
#
# 计划 §5.1 的红线：「**不能为了调用第三方 builder 悄悄把当前力场重参数化。**」
# 本层只吃**键图**——原子序数 + 键连接——不做任何化学感知：不推断键级、不重新
# 分配电荷、不重新参数化。用户给的那份参数从头到尾原样不动。
#
# rdkit 只在一个地方可能被用到（片段内 MCS，见 `map_atoms` 的 M4 步），且：
#   - 只在**已配对的片段内部**、只在两个片段不同构时才启用；
#   - 建 RWMol 时全部键按单键建、比较时忽略键级（`BondCompareAny`），
#     所以仍然没有引入一次化学感知；
#   - 用完即弃，不参与建系。
#
# ## 与 ABFE 那边的关系：只借设计，不 import
#
# `outer_lambda_neural_basis.py::discover_ligand_rotatable_torsions`（:5637）里有
# 一份等价的键图 + 环判定实现，本节的 `_alternate_path_exists` 与「非环可旋转键」
# 判据与它**语义一致**（同一套定义：重原子、非环、两端各至少还有一个重原子邻居）。
# 但那份实现是 `discover_ligand_rotatable_torsions` 内部的闭包，import 不出来；
# 而且它是 ABFE 侧模块，本模块按依赖方向不得 import ABFE。因此这里是**重写**，
# 不是复制粘贴，也不改 ABFE 那边任何一行。

#: 原子映射层的协议版本。进 `atom_mapping.json` 与边身份指纹。
#: 任何改变图构建、环判定、切键判据或匹配算法的改动都必须 +1——
#: 换了映射就必须拒绝复用旧产物（计划 §7）。
RBFE_MAPPING_PROTOCOL_VERSION = 1


class RBFEMappingError(RBFEValidationError):
    """原子映射无法在**确定性**下完成时抛出。

    映射歧义一律 fail closed：宁可拒绝，也不在多个等价解里随手挑一个——
    计划 §5.2 要求「识别对称等价映射与姿势歧义」，识别出来却静默选一个
    等于没识别。
    """


#: 只列本层实际会遇到的元素；缺的宁可报错也不猜。
_ELEMENT_SYMBOL_BY_Z = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 12: "Mg",
    14: "Si", 15: "P", 16: "S", 17: "Cl", 19: "K", 20: "Ca", 26: "Fe",
    30: "Zn", 35: "Br", 53: "I",
}


@dataclass(frozen=True)
class AtomNode:
    """图里的一个原子。

    `index` 是**该原子在其来源容器里的索引**（OpenMM topology 的全局索引，或
    GROMACS `.itp` 里的 1-based 序号）。映射层内部一律用它作键；分子内索引
    （0..n-1）另行记录，见 `MolecularGraph.local_index`。计划 §5.2 要求这几套
    索引**分别**记录，不能混。
    """

    index: int
    atomic_number: int
    name: str = ""
    residue_name: str = ""
    residue_index: int = 0
    chain_index: int = 0

    @property
    def element(self) -> str:
        symbol = _ELEMENT_SYMBOL_BY_Z.get(int(self.atomic_number))
        if symbol is None:
            raise RBFEMappingError(
                f"原子 index={self.index} name={self.name!r} 的原子序数 "
                f"{self.atomic_number} 不在已知元素表里。"
                "映射依赖元素身份做化学一致性检查，猜不得——请补 _ELEMENT_SYMBOL_BY_Z。"
            )
        return symbol

    @property
    def is_heavy(self) -> bool:
        return int(self.atomic_number) > 1

    def identity(self) -> dict:
        """跨 System 重建仍可审计的原子身份（计划 §5.2）。"""
        return {
            "index": int(self.index),
            "atomic_number": int(self.atomic_number),
            "element": self.element,
            "name": self.name,
            "residue_name": self.residue_name,
            "residue_index": int(self.residue_index),
            "chain_index": int(self.chain_index),
        }


@dataclass(frozen=True)
class MolecularGraph:
    """一个配体分子的键图。**只含配体原子**，不含受体和溶剂。

    构造后不可变：映射一旦基于某个图算出来，图本身就是身份的一部分。
    """

    atoms: tuple
    bonds: tuple

    def __post_init__(self) -> None:
        if not self.atoms:
            raise RBFEMappingError("分子图为空——没有原子")
        indices = [int(a.index) for a in self.atoms]
        if len(set(indices)) != len(indices):
            dupes = sorted({i for i in indices if indices.count(i) > 1})
            raise RBFEMappingError(f"分子图存在重复原子索引：{dupes}")
        index_set = set(indices)

        canonical = set()
        for pair in self.bonds:
            left, right = int(pair[0]), int(pair[1])
            if left == right:
                raise RBFEMappingError(f"自环键：atom {left} 连到自己")
            if left not in index_set or right not in index_set:
                raise RBFEMappingError(
                    f"键 ({left}, {right}) 的端点不在本图的原子集合里——"
                    "配体子图必须自洽：跨配体边界的键说明 ligand_indices 划错了"
                )
            canonical.add((min(left, right), max(left, right)))

        object.__setattr__(self, "atoms", tuple(sorted(self.atoms, key=lambda a: a.index)))
        object.__setattr__(self, "bonds", tuple(sorted(canonical)))

        neighbors = {i: set() for i in index_set}
        for left, right in self.bonds:
            neighbors[left].add(right)
            neighbors[right].add(left)
        object.__setattr__(self, "_neighbors", {i: tuple(sorted(v)) for i, v in neighbors.items()})
        object.__setattr__(self, "_by_index", {int(a.index): a for a in self.atoms})
        object.__setattr__(
            self, "_local_index", {int(a.index): n for n, a in enumerate(self.atoms)}
        )
        object.__setattr__(self, "_ring_bonds_cache", None)

        if not self._is_connected():
            raise RBFEMappingError(
                "配体子图不连通——它被拆成了多个互不相连的片段。"
                "共价配体、或者 ligand_indices 混进了别的分子，都会长这样。"
                "首版拒绝（计划 §2 的共价配体条款）。"
            )

    # -- 基本访问 ----------------------------------------------------------

    @property
    def indices(self) -> tuple:
        return tuple(int(a.index) for a in self.atoms)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    def atom(self, index: int):
        try:
            return self._by_index[int(index)]
        except KeyError:
            raise RBFEMappingError(f"原子 {index} 不在本图里") from None

    def local_index(self, index: int) -> int:
        """全局索引 → 分子内索引（0..n-1，按全局索引升序）。"""
        try:
            return self._local_index[int(index)]
        except KeyError:
            raise RBFEMappingError(f"原子 {index} 不在本图里") from None

    def neighbors(self, index: int) -> tuple:
        try:
            return self._neighbors[int(index)]
        except KeyError:
            raise RBFEMappingError(f"原子 {index} 不在本图里") from None

    def heavy_neighbors(self, index: int) -> tuple:
        return tuple(i for i in self.neighbors(index) if self.atom(i).is_heavy)

    def degree(self, index: int) -> int:
        return len(self.neighbors(index))

    def _is_connected(self) -> bool:
        start = int(self.atoms[0].index)
        seen = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in self._neighbors[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return len(seen) == len(self.atoms)

    # -- M1：环分析 --------------------------------------------------------

    def _alternate_path_exists(self, left: int, right: int) -> bool:
        """去掉 (left, right) 这条键之后，left 还能不能走到 right。

        能走到 ⇒ 这条键在环上。与 `discover_ligand_rotatable_torsions` 里的
        `central_bond_is_in_ring` 语义一致（见本节顶部说明）。
        """
        stack = [left]
        visited = {left}
        while stack:
            current = stack.pop()
            for neighbor in self._neighbors[current]:
                if {current, neighbor} == {left, right}:
                    continue
                if neighbor == right:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return False

    def ring_bonds(self) -> frozenset:
        """所有处于环上的键。"""
        if self._ring_bonds_cache is None:
            found = frozenset(
                bond for bond in self.bonds if self._alternate_path_exists(bond[0], bond[1])
            )
            object.__setattr__(self, "_ring_bonds_cache", found)
        return self._ring_bonds_cache

    def ring_atoms(self) -> frozenset:
        """所有处于环上的原子。

        一个原子在环上 ⟺ 它至少连着一条环键。（若键 (X,a) 在环上，按定义
        去掉它之后 X 仍能走到 a，即 X 与 a 同处一个圈里。）
        """
        found = set()
        for left, right in self.ring_bonds():
            found.add(left)
            found.add(right)
        return frozenset(found)

    def smallest_ring_size(self, left: int, right: int):
        """经过键 (left, right) 的最小环的大小；不是环键则返回 None。

        做法：删掉这条键后做 BFS 求最短路，环大小 = 最短路长度 + 1。
        """
        key = (min(left, right), max(left, right))
        if key not in self.ring_bonds():
            return None
        source, target = key
        distance = {source: 0}
        queue = [source]
        head = 0
        while head < len(queue):
            current = queue[head]
            head += 1
            for neighbor in self._neighbors[current]:
                if {current, neighbor} == {source, target}:
                    continue
                if neighbor in distance:
                    continue
                distance[neighbor] = distance[current] + 1
                if neighbor == target:
                    return distance[neighbor] + 1
                queue.append(neighbor)
        # ring_bonds() 说它在环上，BFS 却走不通 —— 只可能是内部状态被破坏了。
        raise RBFEMappingError(
            f"内部不一致：键 {key} 被判为环键但找不到替代通路"
        )  # pragma: no cover - 防御性

    def ring_size_profile(self) -> tuple:
        """全图环键的最小环大小的**排序多重集**。

        这是一个廉价、确定性的环指纹：环断裂/闭合会改变环键数量，环尺寸变化
        会改变其中的数值。R0 把这两类挂在 `unchecked` 里等的就是它。
        """
        sizes = [self.smallest_ring_size(l, r) for l, r in sorted(self.ring_bonds())]
        return tuple(sorted(s for s in sizes if s is not None))

    def element_counts(self) -> dict:
        counts = {}
        for atom in self.atoms:
            counts[atom.element] = counts.get(atom.element, 0) + 1
        return dict(sorted(counts.items()))

    # -- 构造入口 ----------------------------------------------------------

    @classmethod
    def from_atoms_and_bonds(cls, atoms, bonds) -> "MolecularGraph":
        return cls(atoms=tuple(atoms), bonds=tuple(tuple(b) for b in bonds))

    @classmethod
    def from_openmm_topology(cls, topology, ligand_indices) -> "MolecularGraph":
        """从 OpenMM topology 抽出配体子图。

        openmm 在这里**惰性**使用（只读 topology 的属性，不 import 模块），
        所以 `rbfe_core` 仍然可以在没装 openmm 的环境里 import——R0 的
        「不启动 GPU」性质不因为 R1 而丢掉。
        """
        wanted = {int(i) for i in ligand_indices}
        if len(wanted) < 2:
            raise RBFEMappingError(f"ligand_indices 至少需要两个原子：收到 {len(wanted)}")
        all_atoms = list(topology.atoms())
        out_of_range = sorted(i for i in wanted if i < 0 or i >= len(all_atoms))
        if out_of_range:
            raise RBFEMappingError(
                f"ligand_indices 超出 topology（共 {len(all_atoms)} 个原子）：{out_of_range}"
            )
        nodes = []
        for index in sorted(wanted):
            atom = all_atoms[index]
            element = getattr(atom, "element", None)
            atomic_number = getattr(element, "atomic_number", None) if element is not None else None
            if atomic_number is None:
                raise RBFEMappingError(
                    f"topology 原子 {index}（{getattr(atom, 'name', '?')}）没有 element——"
                    "映射需要元素身份，不能从原子名猜"
                )
            residue = atom.residue
            nodes.append(
                AtomNode(
                    index=int(index),
                    atomic_number=int(atomic_number),
                    name=str(getattr(atom, "name", "")),
                    residue_name=str(residue.name),
                    residue_index=int(residue.index),
                    chain_index=int(residue.chain.index),
                )
            )
        bonds = [
            (int(a.index), int(b.index))
            for a, b in topology.bonds()
            if int(a.index) in wanted and int(b.index) in wanted
        ]
        return cls.from_atoms_and_bonds(nodes, bonds)

    @classmethod
    def from_gromacs_itp(cls, path: str, moleculetype: Optional[str] = None) -> "MolecularGraph":
        """从 GROMACS `.itp` 的 `[ atoms ]` / `[ bonds ]` 建图。

        首版锁定的输入路线是**已参数化的 GROMACS／OpenMM 输入**（计划 §5.1），
        这是其中 GROMACS 那半边的入口。只读 `[ atomtypes ]`（拿 at.num）、
        `[ atoms ]`、`[ bonds ]` 三段——不解析力场参数、不做任何转换。

        元素来自 `[ atomtypes ]` 的 at.num 列。拿不到就**报错**，不按质量或
        原子名猜：猜错一个元素，整个映射的化学一致性检查就是假的。
        """
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        section = None
        atomic_number_by_type = {}
        rows = []
        bonds = []
        current_moleculetype = None
        for raw in text.splitlines():
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("#"):
                continue          # #include / #ifdef：本层不展开
            if line.startswith("["):
                section = line.strip("[] \t").lower()
                continue
            fields = line.split()
            if section == "atomtypes":
                # name at.num mass charge ptype sigma epsilon
                if len(fields) >= 2 and fields[1].isdigit():
                    atomic_number_by_type[fields[0]] = int(fields[1])
            elif section == "moleculetype":
                current_moleculetype = fields[0]
            elif section == "atoms":
                if moleculetype is not None and current_moleculetype != moleculetype:
                    continue
                if len(fields) < 5:
                    continue
                rows.append(fields)
            elif section == "bonds":
                if moleculetype is not None and current_moleculetype != moleculetype:
                    continue
                if len(fields) >= 2:
                    bonds.append((int(fields[0]), int(fields[1])))

        if not rows:
            where = f"（moleculetype={moleculetype!r}）" if moleculetype else ""
            raise RBFEMappingError(f"{path} 里没有找到 [ atoms ] 段{where}")

        nodes = []
        for fields in rows:
            index, atom_type, resnr, resname, atom_name = fields[:5]
            atomic_number = atomic_number_by_type.get(atom_type)
            if atomic_number is None:
                raise RBFEMappingError(
                    f"{path}：原子类型 {atom_type!r} 在 [ atomtypes ] 里没有 at.num，"
                    "无法确定元素。不按质量/原子名猜——猜错元素会让化学一致性检查失效。"
                )
            nodes.append(
                AtomNode(
                    index=int(index),
                    atomic_number=int(atomic_number),
                    name=str(atom_name),
                    residue_name=str(resname),
                    residue_index=int(resnr),
                    chain_index=0,
                )
            )
        return cls.from_atoms_and_bonds(nodes, bonds)


# -- M2：片段分解 ----------------------------------------------------------


def is_fragmentation_bond(graph: MolecularGraph, left: int, right: int) -> bool:
    """这条键能不能作为片段切点：**非环、重原子—重原子、两端都不是末端**。

    与 `discover_ligand_rotatable_torsions` 的「可旋转中央键」判据同一套定义。
    末端键（甲基、羟基等）不切——切了只会得到一堆单原子片段，人眼没法审。
    """
    key = (min(left, right), max(left, right))
    if key not in set(graph.bonds):
        return False
    if key in graph.ring_bonds():
        return False
    if not (graph.atom(key[0]).is_heavy and graph.atom(key[1]).is_heavy):
        return False
    outer_left = [i for i in graph.heavy_neighbors(key[0]) if i != key[1]]
    outer_right = [i for i in graph.heavy_neighbors(key[1]) if i != key[0]]
    return bool(outer_left) and bool(outer_right)


@dataclass(frozen=True)
class Fragment:
    """一个片段：切键之后的一个连通块。"""

    fragment_id: int
    atom_indices: tuple
    #: 该片段里参与切键的原子（连到别的片段去的那些），排序后的元组。
    attachment_atoms: tuple
    element_counts: dict
    ring_size_profile: tuple
    n_internal_bonds: int

    @property
    def n_atoms(self) -> int:
        return len(self.atom_indices)

    def signature(self) -> tuple:
        """片段的**结构签名**：组成 + 环 + 内部键数 + 接点数。

        签名相同不等于同构；它只用来把候选缩到很小的范围，真正的判定在
        M4 的原子级对齐。计划 §5.2：「映射评分只用于候选排序，不能代替
        化学与几何验收。」
        """
        return (
            tuple(sorted(self.element_counts.items())),
            self.ring_size_profile,
            int(self.n_internal_bonds),
            len(self.attachment_atoms),
        )


@dataclass(frozen=True)
class FragmentDecomposition:
    """M2 的产物：片段集合 + 片段之间的连接图。"""

    graph: MolecularGraph
    fragments: tuple
    cut_bonds: tuple
    #: 原子全局索引 -> fragment_id
    fragment_of_atom: dict
    #: fragment_id -> 与之相连的 (邻居 fragment_id, 本侧接点原子, 对侧接点原子) 元组
    adjacency: dict

    def fragment(self, fragment_id: int) -> Fragment:
        return self.fragments[int(fragment_id)]

    def to_dict(self) -> dict:
        return {
            "n_fragments": len(self.fragments),
            "cut_bonds": [list(b) for b in self.cut_bonds],
            "fragments": [
                {
                    "fragment_id": f.fragment_id,
                    "atom_indices": list(f.atom_indices),
                    "attachment_atoms": list(f.attachment_atoms),
                    "element_counts": f.element_counts,
                    "ring_size_profile": list(f.ring_size_profile),
                    "n_internal_bonds": f.n_internal_bonds,
                }
                for f in self.fragments
            ],
        }


def decompose_into_fragments(graph: MolecularGraph) -> FragmentDecomposition:
    """M2：在非环可旋转键处切开，得到片段集合与片段连接图。

    片段编号是**确定性**的：按片段内最小原子索引升序编号。同一个图重复分解
    必然得到同一份编号——映射要能冻结进身份指纹，非确定性的编号会让指纹漂移。
    """
    cut_bonds = tuple(
        bond for bond in graph.bonds if is_fragmentation_bond(graph, bond[0], bond[1])
    )
    cut_set = set(cut_bonds)

    remaining = {}
    for index in graph.indices:
        remaining[index] = [
            n for n in graph.neighbors(index)
            if (min(index, n), max(index, n)) not in cut_set
        ]

    unassigned = set(graph.indices)
    blocks = []
    while unassigned:
        seed = min(unassigned)
        block = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in remaining[current]:
                if neighbor not in block:
                    block.add(neighbor)
                    stack.append(neighbor)
        blocks.append(tuple(sorted(block)))
        unassigned -= block
    blocks.sort(key=lambda b: b[0])

    fragment_of_atom = {}
    for fragment_id, block in enumerate(blocks):
        for index in block:
            fragment_of_atom[index] = fragment_id

    fragments = []
    for fragment_id, block in enumerate(blocks):
        members = set(block)
        internal_bonds = [
            b for b in graph.bonds if b[0] in members and b[1] in members and b not in cut_set
        ]
        ring_sizes = tuple(
            sorted(
                s for s in (graph.smallest_ring_size(l, r) for l, r in internal_bonds)
                if s is not None
            )
        )
        counts = {}
        for index in block:
            symbol = graph.atom(index).element
            counts[symbol] = counts.get(symbol, 0) + 1
        attachments = tuple(
            sorted({b[0] if b[0] in members else b[1] for b in cut_set
                    if b[0] in members or b[1] in members})
        )
        fragments.append(
            Fragment(
                fragment_id=fragment_id,
                atom_indices=block,
                attachment_atoms=attachments,
                element_counts=dict(sorted(counts.items())),
                ring_size_profile=ring_sizes,
                n_internal_bonds=len(internal_bonds),
            )
        )

    adjacency = {f.fragment_id: [] for f in fragments}
    for left, right in cut_bonds:
        fl, fr = fragment_of_atom[left], fragment_of_atom[right]
        adjacency[fl].append((fr, left, right))
        adjacency[fr].append((fl, right, left))
    adjacency = {k: tuple(sorted(v)) for k, v in adjacency.items()}

    return FragmentDecomposition(
        graph=graph,
        fragments=tuple(fragments),
        cut_bonds=cut_bonds,
        fragment_of_atom=dict(sorted(fragment_of_atom.items())),
        adjacency=adjacency,
    )


# -- M3/M4：片段匹配与片段内原子对齐 -------------------------------------
#
# 整体流程（提案 §4 的 M0-M6）：
#
#   M0/M1/M2  建图 → 环分析 → 片段分解            （上面）
#   M3        片段级匹配：种子 + 沿片段图生长      （本节）
#   M4        已配对片段内部的原子级对齐（图同构）  （本节）
#   M3.5/M4b  位置对应但签名不同的片段 → rdkit MCS  （本节，路线 B 的那一半）
#   M5        组装 AtomMapping，冻结、落盘          （本节）
#   M6        映射验证                              （本节）
#
# 全程 fail closed：任何一步出现无法确定性消解的歧义，就抛 `RBFEMappingError`，
# 不在等价解里随手挑一个。

#: 同构搜索最多枚举多少个解。只用来**识别**对称等价（>1 即存在对称），
#: 不需要枚举完；到顶就停并如实记录。
_MAX_ISOMORPHISM_SOLUTIONS = 64


def _fragment_signature(
    decomposition: FragmentDecomposition, fragment_id: int
) -> tuple:
    """片段签名 + 它在片段图里的度。

    度必须进签名：组成完全相同但一个是末端基团、一个是中间连接子，化学处境
    不同，不该被当成同一个候选。
    """
    fragment = decomposition.fragment(fragment_id)
    return fragment.signature() + (len(decomposition.adjacency[fragment_id]),)


def _find_fragment_isomorphisms(
    graph_a: MolecularGraph,
    atoms_a: Sequence[int],
    graph_b: MolecularGraph,
    atoms_b: Sequence[int],
    anchors: Optional[dict] = None,
    limit: int = _MAX_ISOMORPHISM_SOLUTIONS,
) -> list:
    """枚举两个片段之间的**诱导子图同构**（全覆盖，一一对应）。

    约束（缺一不可）：
      - 元素相同——化学一致性，计划 §5.2；
      - **在整图里的度**相同——这条把片段的对外接点也一起对上了，不然会出现
        「片段内部长得一样、但一个接着苯环一个接着甲基」的错配；
      - 环原子身份相同；
      - 片段内部键关系保持（诱导，不只是保边）；
      - `anchors` 给定的原子对必须成立。

    返回 A 全局索引 -> B 全局索引 的字典列表，按确定性顺序。空列表 = 不同构。
    """
    atoms_a = sorted(int(i) for i in atoms_a)
    atoms_b = sorted(int(i) for i in atoms_b)
    if len(atoms_a) != len(atoms_b):
        return []

    ring_a, ring_b = graph_a.ring_atoms(), graph_b.ring_atoms()
    set_a, set_b = set(atoms_a), set(atoms_b)
    bonds_a = {
        (l, r) for l, r in graph_a.bonds if l in set_a and r in set_a
    }
    bonds_b = {
        (l, r) for l, r in graph_b.bonds if l in set_b and r in set_b
    }
    anchors = dict(anchors or {})

    def compatible(a: int, b: int) -> bool:
        if graph_a.atom(a).atomic_number != graph_b.atom(b).atomic_number:
            return False
        if graph_a.degree(a) != graph_b.degree(b):
            return False
        if (a in ring_a) != (b in ring_b):
            return False
        if a in anchors and anchors[a] != b:
            return False
        if b in anchors.values() and anchors.get(a) != b:
            return False
        return True

    # 先按"候选数最少"排序可以剪掉大量分支，但会让枚举顺序依赖数据。
    # 这里坚持**按全局索引升序**：解的顺序必须可复现，才谈得上"规范解"。
    solutions = []

    def backtrack(position: int, mapping: dict, used: set) -> None:
        if len(solutions) >= limit:
            return
        if position == len(atoms_a):
            solutions.append(dict(mapping))
            return
        a = atoms_a[position]
        for b in atoms_b:
            if b in used or not compatible(a, b):
                continue
            ok = True
            for mapped_a, mapped_b in mapping.items():
                bond_in_a = (min(a, mapped_a), max(a, mapped_a)) in bonds_a
                bond_in_b = (min(b, mapped_b), max(b, mapped_b)) in bonds_b
                if bond_in_a != bond_in_b:
                    ok = False
                    break
            if not ok:
                continue
            mapping[a] = b
            used.add(b)
            backtrack(position + 1, mapping, used)
            del mapping[a]
            used.discard(b)

    backtrack(0, {}, set())
    return solutions


def _canonical_solution(solutions: list) -> dict:
    """在多个等价解里选**规范解**：按 (A 索引升序下的 B 索引序列) 字典序最小。

    这只是"选一个可复现的"，不代表其它解是错的。对称等价解的数量会被单独
    记进映射产物，不藏起来。
    """
    return min(solutions, key=lambda s: tuple(s[k] for k in sorted(s)))


def _select_seed(
    da: FragmentDecomposition,
    db: FragmentDecomposition,
    sig_a: dict,
    sig_b: dict,
) -> tuple:
    """M3 的种子：一个在 A 和 B 里**签名都只出现一次**的片段对。

    种子必须唯一，否则从第一步就在猜。优先取原子数最多、环最多的那个——
    骨架片段最不容易撞签名，也最不容易在首版范围（相近骨架）里被改掉。

    找不到唯一种子就 fail closed：这通常意味着两个分子要么高度对称、要么差异
    已经超出「同一受体、相近骨架」的首版范围（计划 §2）。
    """

    def unique_signatures(sigs: dict) -> set:
        counts = {}
        for value in sigs.values():
            counts[value] = counts.get(value, 0) + 1
        return {k for k, v in counts.items() if v == 1}

    shared = unique_signatures(sig_a) & unique_signatures(sig_b)
    if not shared:
        raise RBFEMappingError(
            "找不到可作为种子的片段：没有任何片段签名在 A 和 B 里都只出现一次。"
            "这说明两个分子的片段分解高度对称或差异过大，首版不猜——"
            "请检查是不是超出了「同一受体、相近骨架」的首版范围（计划 §2）。"
        )

    def seed_key(signature) -> tuple:
        fragment_id = next(k for k, v in sig_a.items() if v == signature)
        fragment = da.fragment(fragment_id)
        return (-fragment.n_atoms, -len(fragment.ring_size_profile), fragment_id)

    signature = min(shared, key=seed_key)
    seed_a = next(k for k, v in sig_a.items() if v == signature)
    seed_b = next(k for k, v in sig_b.items() if v == signature)
    return seed_a, seed_b


def _anchor_constraints(
    da: FragmentDecomposition,
    db: FragmentDecomposition,
    fragment_pairs: dict,
    fa: int,
    fb: int,
    known_pairs: dict,
) -> dict:
    """已配对片段 (fa, fb) 内部对齐时的锚点：接点原子必须对上接点原子。

    锚点有两个来源，强度不同：

    1. **切键对端已定**（强）：fa 经切键连到已配对的邻居 nfa，且对端原子
       `other_a` 已经在 `known_pairs` 里——那么本侧接点 `own_a` 只能对到 B 侧
       那条"对端正好是 core[other_a]"的切键的接点。多重连接也能唯一确定。
    2. **一对一连接**（弱，兜底）：fa↔nfa 与 fb↔nfb 之间各恰好一条切键时，
       两个接点直接对上。

    都定不下来就**不加锚**，交给同构自己解；解不唯一会被如实记成对称等价。
    """
    anchors = {}
    for neighbor_a, own_a, other_a in da.adjacency[fa]:
        if neighbor_a not in fragment_pairs:
            continue
        neighbor_b = fragment_pairs[neighbor_a]
        links_b = [e for e in db.adjacency[fb] if e[0] == neighbor_b]
        mapped_other = known_pairs.get(other_a)
        if mapped_other is not None:
            exact = [e for e in links_b if e[2] == mapped_other]
            if len(exact) == 1:
                anchors[own_a] = exact[0][1]
                continue
        links_a = [e for e in da.adjacency[fa] if e[0] == neighbor_a]
        if len(links_a) == 1 and len(links_b) == 1:
            anchors[own_a] = links_b[0][1]
    # 已经在别处确定下来的原子对（例如上一轮 MCS 定下的）同样是锚
    for atom_a in da.fragment(fa).atom_indices:
        if atom_a in known_pairs:
            anchors[atom_a] = known_pairs[atom_a]
    return anchors


def _build_rdkit_fragment(graph: MolecularGraph, atom_indices: Sequence[int]):
    """把一个片段建成 rdkit RWMol：**全部单键、不做化学感知**。

    这是路线 A+B 里 rdkit 唯一出场的地方。刻意不推断键级、不 sanitize 芳香性：
    计划 §5.1 不允许为了调第三方库把输入重新解释一遍。MCS 那边配套用
    `BondCompare.CompareAny` 忽略键级，所以丢掉键级不影响匹配。
    """
    from rdkit import Chem  # 惰性 import：没装 rdkit 也能 import rbfe_core

    order = sorted(int(i) for i in atom_indices)
    position = {index: n for n, index in enumerate(order)}
    mol = Chem.RWMol()
    for index in order:
        atom = Chem.Atom(int(graph.atom(index).atomic_number))
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    members = set(order)
    for left, right in graph.bonds:
        if left in members and right in members:
            mol.AddBond(position[left], position[right], Chem.BondType.SINGLE)
    result = mol.GetMol()
    result.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(result)
    return result, order


def _mcs_align(
    graph_a: MolecularGraph,
    atoms_a: Sequence[int],
    graph_b: MolecularGraph,
    atoms_b: Sequence[int],
    anchors: dict,
) -> dict:
    """M4b：两个**签名不同但位置对应**的片段之间求最大公共子结构。

    用途：A→B 的差异基团往往落在某一个片段里。没有这一步，整个片段会整块变成
    dummy——正确但浪费（公共核心变小、收敛更差）。有了它，只有真正变掉的那几个
    原子才是 dummy。

    rdkit 缺失时返回空字典，由调用方降级为"整块 dummy"并**在 method 里记下来**
    （降级会改变映射，必须可见，不能静默）。
    """
    try:
        from rdkit.Chem import rdFMCS
    except ImportError:
        return {}

    mol_a, order_a = _build_rdkit_fragment(graph_a, atoms_a)
    mol_b, order_b = _build_rdkit_fragment(graph_b, atoms_b)
    result = rdFMCS.FindMCS(
        [mol_a, mol_b],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,      # 不允许 MCS 只吃半个环——那正是"环断裂"
        matchValences=False,
        timeout=30,
    )
    if result.canceled or result.numAtoms == 0:
        return {}

    from rdkit import Chem

    query = Chem.MolFromSmarts(result.smartsString)
    if query is None:
        return {}
    matches_a = mol_a.GetSubstructMatches(query, uniquify=False, maxMatches=256)
    matches_b = mol_b.GetSubstructMatches(query, uniquify=False, maxMatches=256)
    if not matches_a or not matches_b:
        return {}

    best = None
    for match_a in matches_a:
        for match_b in matches_b:
            candidate = {
                order_a[pa]: order_b[pb] for pa, pb in zip(match_a, match_b)
            }
            if any(
                candidate.get(k) not in (None, v) for k, v in anchors.items()
            ):
                continue
            if any(
                graph_a.atom(k).atomic_number != graph_b.atom(v).atomic_number
                for k, v in candidate.items()
            ):
                continue
            key = tuple(candidate[k] for k in sorted(candidate))
            if best is None or key < best[0]:
                best = (key, candidate)
    return best[1] if best is not None else {}


@dataclass(frozen=True)
class AtomMapping:
    """冻结的分子级 A→B 映射（计划 §5.2）。

    **索引一律是分子内索引**（0..n-1，按来源索引升序）。计划 §5.2 要求两腿共用
    同一份分子级映射、再各自投影为全局索引——所以这个对象里不存 complex/solvent
    的全局索引，投影由 `project()` 在每条腿上单独做。这样物理上不可能出现
    "两腿用了不同映射"。
    """

    protocol_version: int
    n_atoms_a: int
    n_atoms_b: int
    #: ((local_a, local_b), ...)，按 local_a 升序。公共核心。
    core_pairs: tuple
    #: 只存在于 A 的原子（分子内索引）——B 侧为 dummy。
    a_only: tuple
    #: 只存在于 B 的原子——A 侧为 dummy。
    b_only: tuple
    #: ((fragment_id_a, fragment_id_b), ...)
    fragment_pairs: tuple
    method: str
    #: ((fragment_id_a, fragment_id_b, n_solutions), ...)，n>1 即存在对称等价解
    symmetry_solution_counts: tuple
    ambiguities: tuple
    atom_identity_a: tuple
    atom_identity_b: tuple

    @property
    def n_core(self) -> int:
        return len(self.core_pairs)

    def a_to_b(self) -> dict:
        return {int(a): int(b) for a, b in self.core_pairs}

    def b_to_a(self) -> dict:
        return {int(b): int(a) for a, b in self.core_pairs}

    def hybrid_indices(self) -> dict:
        """hybrid 体系里的原子编号：core → A-only → B-only，各段内保序。

        返回 {"a": {local_a: hybrid}, "b": {local_b: hybrid}, "n_hybrid_atoms": n}。
        计划 §5.3 要求"保持跨 λ 相同的粒子、质量和约束结构"——hybrid 体系的粒子
        集合因此是三段的并集，两个端点各自只是其中的一个子集。
        """
        map_a, map_b = {}, {}
        cursor = 0
        for local_a, local_b in self.core_pairs:
            map_a[int(local_a)] = cursor
            map_b[int(local_b)] = cursor
            cursor += 1
        for local_a in self.a_only:
            map_a[int(local_a)] = cursor
            cursor += 1
        for local_b in self.b_only:
            map_b[int(local_b)] = cursor
            cursor += 1
        return {"a": map_a, "b": map_b, "n_hybrid_atoms": cursor}

    def project(self, global_indices_a: Sequence[int], global_indices_b: Sequence[int]) -> dict:
        """把分子级映射投影到某一条腿的全局原子索引上（计划 §5.2）。

        `global_indices_*` 是该配体在这条腿的体系里的全局索引，**按分子内顺序**
        给出（即第 i 个元素就是分子内索引 i 的那个原子）。长度对不上直接报错——
        投影错位是那种能一路跑完、结果全错的失败模式。
        """
        list_a = [int(i) for i in global_indices_a]
        list_b = [int(i) for i in global_indices_b]
        if len(list_a) != self.n_atoms_a:
            raise RBFEMappingError(
                f"投影失败：给了 {len(list_a)} 个 A 的全局索引，映射里 A 有 "
                f"{self.n_atoms_a} 个原子"
            )
        if len(list_b) != self.n_atoms_b:
            raise RBFEMappingError(
                f"投影失败：给了 {len(list_b)} 个 B 的全局索引，映射里 B 有 "
                f"{self.n_atoms_b} 个原子"
            )
        for label, values in (("A", list_a), ("B", list_b)):
            if len(set(values)) != len(values):
                raise RBFEMappingError(f"投影失败：{label} 的全局索引有重复")
        return {
            "core_pairs": tuple((list_a[a], list_b[b]) for a, b in self.core_pairs),
            "a_only": tuple(list_a[i] for i in self.a_only),
            "b_only": tuple(list_b[i] for i in self.b_only),
        }

    def to_dict(self) -> dict:
        """写进 `atom_mapping.json` 的内容（计划 §7）。"""
        return {
            "rbfe_mapping_protocol_version": int(self.protocol_version),
            "method": self.method,
            "n_atoms_A": int(self.n_atoms_a),
            "n_atoms_B": int(self.n_atoms_b),
            "n_core": self.n_core,
            "core_pairs_molecule_local": [[int(a), int(b)] for a, b in self.core_pairs],
            "A_only_molecule_local": [int(i) for i in self.a_only],
            "B_only_molecule_local": [int(i) for i in self.b_only],
            "fragment_pairs": [[int(a), int(b)] for a, b in self.fragment_pairs],
            "symmetry_solution_counts": [
                [int(a), int(b), int(n)] for a, b, n in self.symmetry_solution_counts
            ],
            "ambiguities": list(self.ambiguities),
            "hybrid_indices": {
                "A": {str(k): v for k, v in self.hybrid_indices()["a"].items()},
                "B": {str(k): v for k, v in self.hybrid_indices()["b"].items()},
                "n_hybrid_atoms": self.hybrid_indices()["n_hybrid_atoms"],
            },
            "atom_identity_A": list(self.atom_identity_a),
            "atom_identity_B": list(self.atom_identity_b),
        }

    def fingerprint(self) -> str:
        """映射的身份哈希，供 `rbfe_pipeline.edge_identity(atom_mapping_hash=...)`。

        换了映射 ⇒ 哈希变 ⇒ 旧产物不可复用（计划 §7 的硬要求）。
        故意**不含** `atom_identity_*`：那是给人看的审计信息，改个原子名不该
        让已经跑完的采样作废。

        同理故意**不含** `method`：它是"走了哪条代码路径"的标签，不是身份。
        `..._mcs_unavailable_conservative`（这台机器没装 rdkit）与
        `..._mcs_disabled_conservative`（用户给了 --no-mcs）会给出**逐位相同**的
        core/a_only/b_only，却曾经算出两个不同的指纹 ⟹ 换台机器重跑就被
        `assert_reusable` 判成"换了映射"、整条边的 GPU 采样作废。
        映射的内容已经完整地在下面的 payload 里了，method 一个 bit 都没加。
        """
        payload = {
            "v": int(self.protocol_version),
            "core": [[int(a), int(b)] for a, b in self.core_pairs],
            "a_only": [int(i) for i in self.a_only],
            "b_only": [int(i) for i in self.b_only],
            "n_a": int(self.n_atoms_a),
            "n_b": int(self.n_atoms_b),
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def map_atoms(
    graph_a: MolecularGraph,
    graph_b: MolecularGraph,
    *,
    allow_mcs: bool = True,
) -> AtomMapping:
    """M3-M5：从两个配体的键图算出冻结的 A→B 原子映射。

    路线 A+B（提案 §3）：片段级匹配定骨架对应，片段内部做原子级对齐；只有
    「位置对应但组成不同」的那一对片段才动用 rdkit MCS。

    ## 生长为什么按接点原子走，而不是按片段签名走

    第一版按签名生长，在真实 Atenolol 上就错了：把苯环换掉一个取代基之后，
    苯环片段的签名变了 → 生长在苯环处断掉 → **苯环后面那一整段（酰胺尾）
    明明 A、B 一模一样，却被整段判成 A-only/B-only**。

    现在的规则是：先把当前片段对齐（原子级），于是它的接点原子已经进 core；
    再沿切键找邻居——**B 侧候选限定在"挂在对应接点原子上"的那些未配对片段**。
    接点对应是比签名强得多的约束，而且它不因为取代基变化而失效，所以生长能
    穿过差异片段继续往后走。签名只在同一接点上有多个候选时用来消歧。

    `allow_mcs=False` 时完全不碰 rdkit，差异片段整块进 dummy——结果仍然正确，
    只是公共核心更小。这个降级会写进 `method`，不静默。
    """
    da = decompose_into_fragments(graph_a)
    db = decompose_into_fragments(graph_b)
    sig_a = {f.fragment_id: _fragment_signature(da, f.fragment_id) for f in da.fragments}
    sig_b = {f.fragment_id: _fragment_signature(db, f.fragment_id) for f in db.fragments}

    seed_a, seed_b = _select_seed(da, db, sig_a, sig_b)

    pairs = {seed_a: seed_b}
    used_b = {seed_b}
    core: dict = {}
    symmetry_counts = []
    ambiguities = []
    mcs_used = False
    mcs_skipped = False

    queue = [(seed_a, seed_b)]
    head = 0
    while head < len(queue):
        fa, fb = queue[head]
        head += 1

        # --- M4：片段内原子级对齐 ---
        anchors = _anchor_constraints(da, db, pairs, fa, fb, core)
        solutions = _find_fragment_isomorphisms(
            graph_a,
            da.fragment(fa).atom_indices,
            graph_b,
            db.fragment(fb).atom_indices,
            anchors=anchors,
        )
        if solutions:
            chosen = _canonical_solution(solutions)
            if len(solutions) > 1:
                symmetry_counts.append((fa, fb, len(solutions)))
            core.update(chosen)
        else:
            # 不同构（组成不同，或组成相同但连接方式不同）→ M4b：MCS 求最大公共子结构
            partial = {}
            if allow_mcs:
                partial = _mcs_align(
                    graph_a,
                    da.fragment(fa).atom_indices,
                    graph_b,
                    db.fragment(fb).atom_indices,
                    anchors,
                )
            if partial:
                mcs_used = True
                core.update(partial)
            else:
                mcs_skipped = True
                ambiguities.append(
                    f"片段 A#{fa}↔B#{fb} 不同构"
                    + ("且 MCS 未能给出公共子结构" if allow_mcs else "，MCS 被关闭")
                    + "——整块进 dummy"
                )

        # --- M3：沿切键生长，按接点原子对应找 B 侧候选 ---
        for neighbor_a, own_a, _ in sorted(da.adjacency[fa]):
            if neighbor_a in pairs:
                continue
            own_b = core.get(own_a)
            candidates = [
                (nb, ob) for nb, ob, _ in db.adjacency[fb]
                if nb not in used_b and (own_b is None or ob == own_b)
            ]
            candidates = sorted({nb for nb, _ in candidates})
            if len(candidates) > 1:
                narrowed = [nb for nb in candidates if sig_b[nb] == sig_a[neighbor_a]]
                if len(narrowed) == 1:
                    candidates = narrowed
                else:
                    ambiguities.append(
                        f"片段 A#{neighbor_a}（挂在 A#{fa} 的原子 {own_a} 上）在 B 侧有 "
                        f"{len(candidates)} 个候选，签名也消不掉歧义——不猜，留作未配对"
                    )
                    continue
            if not candidates:
                continue
            neighbor_b = candidates[0]
            pairs[neighbor_a] = neighbor_b
            used_b.add(neighbor_b)
            queue.append((neighbor_a, neighbor_b))

    # --- M5：组装 ---
    local_a = {i: graph_a.local_index(i) for i in graph_a.indices}
    local_b = {i: graph_b.local_index(i) for i in graph_b.indices}
    core_pairs = tuple(sorted((local_a[a], local_b[b]) for a, b in core.items()))
    mapped_a = {a for a, _ in core_pairs}
    mapped_b = {b for _, b in core_pairs}

    unpaired_a = sorted(set(sig_a) - set(pairs))
    unpaired_b = sorted(set(sig_b) - used_b)
    for fragment_id in unpaired_a:
        ambiguities.append(f"片段 A#{fragment_id} 未能配对——整块进 A-only（dummy）")
    for fragment_id in unpaired_b:
        ambiguities.append(f"片段 B#{fragment_id} 未能配对——整块进 B-only（dummy）")

    if mcs_used and not mcs_skipped:
        method = "fragment_isomorphism+rdkit_mcs"
    elif mcs_used and mcs_skipped:
        method = "fragment_isomorphism+rdkit_mcs__partial_conservative"
    elif mcs_skipped and allow_mcs:
        method = "fragment_isomorphism_only__mcs_unavailable_conservative"
    elif mcs_skipped:
        method = "fragment_isomorphism_only__mcs_disabled_conservative"
    else:
        method = "fragment_isomorphism"

    return AtomMapping(
        protocol_version=RBFE_MAPPING_PROTOCOL_VERSION,
        n_atoms_a=graph_a.n_atoms,
        n_atoms_b=graph_b.n_atoms,
        core_pairs=core_pairs,
        a_only=tuple(sorted(set(local_a.values()) - mapped_a)),
        b_only=tuple(sorted(set(local_b.values()) - mapped_b)),
        fragment_pairs=tuple(sorted(pairs.items())),
        method=method,
        symmetry_solution_counts=tuple(symmetry_counts),
        ambiguities=tuple(ambiguities),
        atom_identity_a=tuple(a.identity() for a in graph_a.atoms),
        atom_identity_b=tuple(b.identity() for b in graph_b.atoms),
    )


# -- M6：映射验证 ----------------------------------------------------------


def validate_mapping(
    mapping: AtomMapping,
    graph_a: MolecularGraph,
    graph_b: MolecularGraph,
    *,
    edge_id: str = "atom_mapping",
) -> ValidationReport:
    """M6：一一对应、索引范围、化学一致性、核心连通性、环与元素变化。

    这一步把 R0 挂在 `unchecked` 里的几条**真正查掉**：环断裂/闭合、环尺寸变化、
    映射元素改变、共价配体（后者在建图时就被连通性挡掉了）。
    """
    report = ValidationReport(edge_id=edge_id)
    a_to_b = {}
    b_to_a = {}

    for local_a, local_b in mapping.core_pairs:
        if local_a in a_to_b:
            report.errors.append(f"映射不是一一对应：A 的原子 {local_a} 出现多次")
        if local_b in b_to_a:
            report.errors.append(f"映射不是一一对应：B 的原子 {local_b} 出现多次")
        a_to_b[local_a] = local_b
        b_to_a[local_b] = local_a

    for label, values, limit in (
        ("A", list(a_to_b) + list(mapping.a_only), mapping.n_atoms_a),
        ("B", list(b_to_a) + list(mapping.b_only), mapping.n_atoms_b),
    ):
        bad = sorted(i for i in values if i < 0 or i >= limit)
        if bad:
            report.errors.append(f"{label} 侧索引越界（合法范围 0..{limit - 1}）：{bad}")

    covered_a = set(a_to_b) | set(mapping.a_only)
    covered_b = set(b_to_a) | set(mapping.b_only)
    if covered_a != set(range(mapping.n_atoms_a)):
        missing = sorted(set(range(mapping.n_atoms_a)) - covered_a)
        report.errors.append(
            f"A 的原子没有被 core/A-only 完全划分，漏了 {missing}——"
            "hybrid 体系会少粒子"
        )
    if covered_b != set(range(mapping.n_atoms_b)):
        missing = sorted(set(range(mapping.n_atoms_b)) - covered_b)
        report.errors.append(f"B 的原子没有被 core/B-only 完全划分，漏了 {missing}")

    if not mapping.core_pairs:
        report.errors.append(
            "公共核心为空——A 与 B 之间没有任何对应原子。"
            "这不是一条 RBFE 边（那等于把 A 整个消掉再把 B 整个长出来）。"
        )

    index_a = {graph_a.local_index(i): i for i in graph_a.indices}
    index_b = {graph_b.local_index(i): i for i in graph_b.indices}

    # 🔑 下面两条判据（元素改变、环断裂）都靠 index_a/index_b 把分子内索引翻回
    # 图里的原子。原来对翻不出来的 core pair 直接 `continue`——映射与图对不上时，
    # 这两条"首版明令拒绝"的判据就被静默跳过、报告照样 PASS。
    # 先在这里 fail-closed 一次，两个循环的 `continue` 就再也不会真的发生。
    orphan = [
        (int(a), int(b))
        for a, b in mapping.core_pairs
        if a not in index_a or b not in index_b
    ]
    if orphan:
        report.errors.append(
            f"core_pairs 里有 {len(orphan)} 对索引在所给的图里不存在（例如 {orphan[:4]}）"
            "——这份映射不是从这两个图算出来的，元素改变与环断裂判据无法执行"
        )

    # 化学一致性：元素必须相同（计划 §2 的"映射元素改变"）
    element_changes = []
    for local_a, local_b in mapping.core_pairs:
        if local_a not in index_a or local_b not in index_b:
            continue
        atom_a, atom_b = graph_a.atom(index_a[local_a]), graph_b.atom(index_b[local_b])
        if atom_a.atomic_number != atom_b.atomic_number:
            element_changes.append(
                f"{atom_a.name}({atom_a.element}) -> {atom_b.name}({atom_b.element})"
            )
    if element_changes:
        report.errors.append(
            "映射元素改变（首版拒绝，计划 §2）：" + "、".join(element_changes[:8])
            + ("…" if len(element_changes) > 8 else "")
        )

    # 核心连通性（计划 §5.2 明写）
    for label, graph, locals_, index_map in (
        ("A", graph_a, set(a_to_b), index_a),
        ("B", graph_b, set(b_to_a), index_b),
    ):
        members = {index_map[i] for i in locals_ if i in index_map}
        if len(members) <= 1:
            continue
        seed = min(members)
        seen = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in graph.neighbors(current):
                if neighbor in members and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if seen != members:
            hint = ""
            if "conservative" in mapping.method:
                # 实测会踩：差异基团若在分子**中部**（例如苯环上换取代基），保守路径
                # 把整个片段丢进 dummy，核心就被这个片段从中间切断了。此时保守路径
                # 根本给不出可用映射——必须让 MCS 在片段内部保住公共骨架。
                hint = (
                    "。注意 method 是保守降级（{}）：差异片段被整块丢进 dummy。"
                    "差异基团在分子中部时这必然切断核心——请启用 rdkit MCS "
                    "（去掉 --no-mcs / allow_mcs=True），保守路径只在差异位于末端时可用"
                ).format(mapping.method)
            report.errors.append(
                f"{label} 侧的公共核心不连通（{len(seen)}/{len(members)} 个原子在同一块）——"
                f"断开的核心会让 dummy 处理无法证明抵消，首版拒绝{hint}"
            )

    # 环：断裂/闭合与尺寸变化（R0 的 unchecked 在这里被真正查掉）
    ring_a, ring_b = graph_a.ring_atoms(), graph_b.ring_atoms()
    ring_status_changes = []
    for local_a, local_b in mapping.core_pairs:
        if local_a not in index_a or local_b not in index_b:
            continue
        in_ring_a = index_a[local_a] in ring_a
        in_ring_b = index_b[local_b] in ring_b
        if in_ring_a != in_ring_b:
            ring_status_changes.append(
                f"{graph_a.atom(index_a[local_a]).name}"
                f"({'环内' if in_ring_a else '环外'} -> {'环内' if in_ring_b else '环外'})"
            )
    if ring_status_changes:
        report.errors.append(
            "环断裂/闭合：映射把环内原子对到了环外原子（首版拒绝，计划 §2）："
            + "、".join(ring_status_changes[:8])
        )

    profile_a, profile_b = graph_a.ring_size_profile(), graph_b.ring_size_profile()
    if profile_a != profile_b:
        report.warnings.append(
            f"A 与 B 的环指纹不同（{profile_a} vs {profile_b}）。"
            "若差异发生在公共核心上，上面的环断裂判据会报错；"
            "若只发生在 A-only/B-only 部分，那是被整块消掉/长出来的环，需人工确认。"
        )

    for fa, fb, count in mapping.symmetry_solution_counts:
        # 枚举到上限就停了，所以到顶时报的是**下界**，不能写成确切数字。
        how_many = (
            f"≥{count} 个（已达枚举上限 {_MAX_ISOMORPHISM_SOLUTIONS}）"
            if count >= _MAX_ISOMORPHISM_SOLUTIONS
            else f"{count} 个"
        )
        report.warnings.append(
            f"片段 A#{fa}↔B#{fb} 存在 {how_many}对称等价映射，已取规范解"
            "（按 A 索引升序下的 B 索引序列字典序最小）"
        )
    for note in mapping.ambiguities:
        report.warnings.append(note)

    if "conservative" in mapping.method:
        report.warnings.append(
            f"映射走了保守降级路径（method={mapping.method}）：差异片段整块进 dummy，"
            "公共核心比可达到的更小。结果正确但收敛更差。"
        )

    report.unchecked.extend(
        [
            "手性反转（需要坐标/立体化学感知，本层只有键图，R1 后续）",
            "结合姿势歧义（需要几何比较，R1 后续）",
            "互变异构状态改变（需要质子位置比较，R1 后续）",
            "虚拟位点/约束/自定义 Force 是否在已验证 builder 范围内（需要建系，R1b）",
        ]
    )
    return report


