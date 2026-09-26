

# ============================================================================
# §1.1 力场族自动识别 + §1.3 dispersion_protocol（B6）
#
# 两者绑在一起，因为 §1.1 的裁决直接决定 §1.3 走哪条路线：
#   amber 系  → Amber 脂质（Lipid21/Lipid17）+ ff_native_isotropic_lrc
#   charmm 系 → CHARMM36 脂质 + ff_native_force_switch_no_lrc
#
# ⚠️ §1.3 的修正框：原稿"膜体系一律禁用 LRC"是错的。判据是**跟随所选脂质力场的
# 原始参数化条件**——Amber Lipid21 就是在开着各向同性 vdW 长程修正下拟合的，
# 对它关掉 LRC 才是错的；CHARMM36 是 force-switch 且不加 LRC，对它开 LRC 才是错的。
#
# ⚠️ OpenMM 的 `NonbondedForce` 只有 potential-switch（`setUseSwitchingFunction`），
# **没有 force-switch**。所以 charmm 分支无法复现原始 Hamiltonian，默认 fail closed，
# 只有给出定量偏差论证（APL / 膜厚 / 单点能对照）后才允许放行。amber 是首选路径。
# ============================================================================

FORCEFIELD_FAMILY_AMBER = "amber"
FORCEFIELD_FAMILY_CHARMM = "charmm"
FORCEFIELD_FAMILIES = (FORCEFIELD_FAMILY_AMBER, FORCEFIELD_FAMILY_CHARMM)

# 识别得出但本轮不支持的族：明确报错好过"识别不出"这种含糊结论。
FORCEFIELD_FAMILIES_UNSUPPORTED = ("opls", "gromos")

# `#include` 路径里出现这些 token 即判定为对应族。按最长匹配优先，避免
# "charmm36" 被 "charm" 之类的前缀误伤。
_FORCEFIELD_FAMILY_TOKENS = (
    ("charmm", FORCEFIELD_FAMILY_CHARMM),
    ("amber", FORCEFIELD_FAMILY_AMBER),
    ("opls", "opls"),
    ("oplsaa", "opls"),
    ("gromos", "gromos"),
)

DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC = "legacy_uniform_density_lrc"
DISPERSION_PROTOCOL_FF_NATIVE_ISOTROPIC_LRC = "ff_native_isotropic_lrc"
DISPERSION_PROTOCOL_FF_NATIVE_FORCE_SWITCH_NO_LRC = "ff_native_force_switch_no_lrc"
DISPERSION_PROTOCOL_LJ_PME = "lj_pme"
DISPERSION_PROTOCOL_MEMBRANE_INHOMOGENEOUS = "membrane_inhomogeneous"

DISPERSION_PROTOCOLS = (
    DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC,
    DISPERSION_PROTOCOL_FF_NATIVE_ISOTROPIC_LRC,
    DISPERSION_PROTOCOL_FF_NATIVE_FORCE_SWITCH_NO_LRC,
    DISPERSION_PROTOCOL_LJ_PME,
    DISPERSION_PROTOCOL_MEMBRANE_INHOMOGENEOUS,
)

# 已实现（有代码支撑）的路线。LJ-PME 是 §1.3 路线 B、非均匀色散修正是路线 C，
# 两者都只有验收条件、没有实现——声明它们必须 NotImplementedError，
# 而不是被当成拼错的未知值。
DISPERSION_PROTOCOLS_IMPLEMENTED = (
    DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC,
    DISPERSION_PROTOCOL_FF_NATIVE_ISOTROPIC_LRC,
)

# §1.3：`system_type=membrane` 且未选择**已验证**的 dispersion_protocol → fail closed。
# 目前只有 amber 分支这一条算已验证路径。
MEMBRANE_VALIDATED_DISPERSION_PROTOCOLS = (
    DISPERSION_PROTOCOL_FF_NATIVE_ISOTROPIC_LRC,
)

FORCEFIELD_FAMILY_DISPERSION_PROTOCOL = {
    FORCEFIELD_FAMILY_AMBER: DISPERSION_PROTOCOL_FF_NATIVE_ISOTROPIC_LRC,
    FORCEFIELD_FAMILY_CHARMM: DISPERSION_PROTOCOL_FF_NATIVE_FORCE_SWITCH_NO_LRC,
}

# charmm 分支放行所需的定量论证字段（§1.1 最后一条）。
FORCE_SWITCH_DEVIATION_EVIDENCE_FIELDS = (
    "apl_comparison",
    "bilayer_thickness_comparison",
    "single_point_energy_comparison",
)


# `[ defaults ]` 的 1-4 缩放约定是比 include 路径**更可靠**的力场族判据：
#   Amber  : comb-rule 2, gen-pairs yes, fudgeLJ 0.5, fudgeQQ 0.8333
#   CHARMM : comb-rule 2, gen-pairs yes, fudgeLJ 1.0, fudgeQQ 1.0
# 实测 memtest/ 那套 CHARMM-GUI FF-Converter 产的 AMBER 体系里，include 路径全是
# `toppar/*.itp`（**不含 amber 字样**），唯一可靠证据就在
# `toppar/forcefield.itp` 的 `[ defaults ]`：`1 2 yes 0.500000 0.833333`。
# 所以 §1.1 原文写的是"从 `#include` **与** `[ defaults ]` 判定"——两者都要。
_FORCEFIELD_FAMILY_FUDGE_SIGNATURES = (
    (FORCEFIELD_FAMILY_AMBER, 0.5, 1.0 / 1.2),
    (FORCEFIELD_FAMILY_CHARMM, 1.0, 1.0),
)
_FUDGE_MATCH_TOLERANCE = 1.0e-3


def resolve_gromacs_include(
    include_name: str,
    including_file: str,
    gmx_include_dir: Optional[str] = None,
) -> str:
    """解析一个 GROMACS `#include` 到真实路径。

    这是**唯一实现**：`runabfe._resolve_gromacs_include_path` 是它的薄包装，
    避免两处各写一份解析顺序（相对包含文件的目录优先，其次 GROMACS include 目录）。
    """
    candidates = [os.path.join(os.path.dirname(including_file), include_name)]
    if gmx_include_dir:
        candidates.append(os.path.join(gmx_include_dir, include_name))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.realpath(candidate)
    raise FileNotFoundError(
        f"GROMACS include 无法解析: {include_name!r}（来自 {including_file}）"
    )


def _iter_topology_lines(top_path: str, gmx_include_dir: Optional[str] = None):
    """按 include 展开顺序产出 (来源文件, 去注释后的行)。

    深度优先、就地展开，与 GROMACS 预处理器语义一致——这一点对 `[ molecules ]`
    这类"位置有意义"的段落是必需的。重复 include 只展开一次。
    """
    seen = set()

    def _walk(path: str):
        real = os.path.realpath(path)
        if real in seen:
            return
        seen.add(real)
        # 🔑 [2026-09-10] 只为"这条 #include 是否可选"记一个 #ifdef 栈。
        #
        # 本解析器不求完整地展开预处理（见 parse_gromacs_topology），但
        # `#ifdef POSRES / #include "posre.itp" / #endif` 是 `gmx pdb2gmx` 的
        # **默认**产物，posre.itp 常常不在 .top 旁边。OpenMM 自己是
        # `ignore = not all(self._ifStack)`、且 defines 默认为空 ⟹ 它根本不会去
        # 展开这条 include。我们把"未解析的 include"当致命错误（preflight），
        # 所以必须用同一套口径，否则一份 OpenMM 读得好好的拓扑会被预检打死。
        #
        # 口径与 OpenMM 对齐：没有任何 define ⟹ `#ifdef X` 一律不激活、
        # `#ifndef X` 一律激活。只影响"要不要把未解析的 include 记下来"，
        # 其余行照旧原样流出，不改任何原子计数。
        if_stack: List[bool] = []
        with open(real, encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.split(";", 1)[0].strip()
                if not line:
                    continue
                if line.startswith("#ifdef"):
                    if_stack.append(False)
                elif line.startswith("#ifndef"):
                    if_stack.append(True)
                elif line.startswith("#else"):
                    if if_stack:
                        if_stack[-1] = not if_stack[-1]
                elif line.startswith("#endif"):
                    if if_stack:
                        if_stack.pop()
                if line.startswith("#include"):
                    parts = line.split('"')
                    if len(parts) >= 2:
                        try:
                            resolved = resolve_gromacs_include(
                                parts[1], real, gmx_include_dir
                            )
                        except FileNotFoundError:
                            if not all(if_stack):
                                # 在一个不激活的 #ifdef 分支里 —— OpenMM 也不会读它，
                                # 不是"拓扑不完整"，不记。
                                continue
                            # 解析不到的 include 记录下来，由调用方决定是否致命：
                            # 有些体系的 posre.itp 是可选的。
                            yield (real, f"#unresolved_include {parts[1]}")
                            continue
                        yield from _walk(resolved)
                    continue
                yield (real, line)

    yield from _walk(top_path)


def parse_gromacs_topology(
    top_path: str,
    gmx_include_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """解析 GROMACS 拓扑，返回**权威**的体系组成。

    为什么需要它：靠残基名硬编码集合判身份，在换一套体系时会静默判错。实测
    `memtest/` 那套体系里——

      * 一个 POPC 是**一个 `[ moleculetype ]`、三个残基**（PA + PC + OL），
        按残基计数会数出 270 个"脂质"，APL 直接错 3 倍；
      * 水叫 `TP3`（mdtraj 的水表里只有 `TIP3`），离子叫 `Na+` / `Cl-`（带符号），
        两者都会让计数**静默变成 0**。

    而 `.top` 里本来就写着权威答案：

        [ molecules ]
        PROA 1 / POPC 90 / Na+ 25 / Cl- 36 / TP3 9542 / Atenolol-rank11 1

    所以身份一律以 `[ molecules ]` + `[ moleculetype ]` 为准。
    """
    defaults: Optional[Dict[str, Any]] = None
    moleculetypes: "OrderedDictType[str, Dict[str, Any]]" = {}
    molecules: List[Tuple[str, int]] = []
    atomtypes: Dict[str, Dict[str, Any]] = {}
    unresolved_includes: List[str] = []
    files: List[str] = []

    section = None
    current_moltype: Optional[str] = None
    for source, line in _iter_topology_lines(top_path, gmx_include_dir):
        if source not in files:
            files.append(source)
        if line.startswith("#unresolved_include"):
            unresolved_includes.append(line.split(None, 1)[1])
            continue
        if line.startswith("#"):
            continue  # #define / #ifdef 等预处理指令，本解析器不求完整
        if line.startswith("["):
            section = line.strip("[] ").strip().lower()
            if section == "moleculetype":
                current_moltype = None
            continue

        fields = line.split()
        if section == "defaults" and defaults is None:
            # nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ
            defaults = {
                "raw": line,
                "nbfunc": int(fields[0]) if len(fields) > 0 else None,
                "comb_rule": int(fields[1]) if len(fields) > 1 else None,
                "gen_pairs": fields[2] if len(fields) > 2 else None,
                "fudge_lj": float(fields[3]) if len(fields) > 3 else None,
                "fudge_qq": float(fields[4]) if len(fields) > 4 else None,
                "source": source,
            }
        elif section == "atomtypes" and len(fields) >= 7:
            # name at.num mass charge ptype sigma epsilon
            # （CHARMM-GUI 还会在注释里附 sigma_14/epsilon_14，注释已被剥掉。）
            try:
                atomtypes[fields[0]] = {
                    "atomic_number": int(float(fields[1])),
                    "mass": float(fields[2]),
                    "charge": float(fields[3]),
                    "ptype": fields[4],
                    "sigma_nm": float(fields[5]),
                    "epsilon_kj_mol": float(fields[6]),
                }
            except ValueError:
                # 少数力场的 [ atomtypes ] 列序不同（例如省略 at.num）。跳过而不是猜，
                # 由用它的地方 fail closed。
                pass
        elif section == "moleculetype" and current_moltype is None:
            current_moltype = fields[0]
            moleculetypes.setdefault(
                current_moltype,
                {
                    "name": current_moltype,
                    "n_atoms": 0,
                    "residue_names": [],
                    "atoms": [],
                },
            )
        elif section == "atoms" and current_moltype:
            # nr type resnr residue atom cgnr charge mass
            if len(fields) >= 5:
                entry = moleculetypes[current_moltype]
                entry["n_atoms"] += 1
                resname = fields[3]
                if resname not in entry["residue_names"]:
                    entry["residue_names"].append(resname)
                entry.setdefault("atoms", []).append(
                    {
                        "nr": int(fields[0]),
                        "type": fields[1],
                        "residue_name": resname,
                        "atom_name": fields[4],
                        "charge": float(fields[6]) if len(fields) >= 7 else None,
                        "mass": float(fields[7]) if len(fields) >= 8 else None,
                    }
                )
        elif section == "molecules":
            if len(fields) >= 2:
                try:
                    molecules.append((fields[0], int(fields[1])))
                except ValueError:
                    continue

    return {
        "top_path": str(top_path),
        "files": files,
        "defaults": defaults,
        "moleculetypes": moleculetypes,
        "molecules": molecules,
        "atomtypes": atomtypes,
        "unresolved_includes": unresolved_includes,
    }


# ---------------------------------------------------------------------------
# 组成驱动的身份判定（替代残基名硬编码集合）
# ---------------------------------------------------------------------------

# 水的 moleculetype 名。加 `TP3` 是因为 CHARMM-GUI 的 AMBER 转换器就叫它 TP3，
# 而 mdtraj 的 `_WATER_RESIDUES` 只有 `TIP3` —— 靠 mdtraj 的 `water` 关键字会
# **静默选出 0 个水**，疏水核内水与 co-ion 首层水配位数就都变成 0 而不报错。
WATER_MOLECULE_NAMES = frozenset(
    {
        "SOL", "HOH", "WAT", "H2O", "OH2",
        "TIP", "TIP3", "TIP3P", "TP3", "T3P",
        "TIP4", "TIP4P", "TIP4PEW", "TP4",
        "SPC", "SPCE", "OPC", "OPC3",
    }
)

# 单原子离子的 moleculetype 名（已归一化：去掉 +/-/数字后再比）。
MONOATOMIC_ION_NAMES = frozenset(
    {"NA", "CL", "K", "SOD", "CLA", "POT", "MG", "CA", "ZN", "LI", "BR", "I", "CS", "RB", "F"}
)

# 标准 + Amber 常见变体氨基酸残基名。自带一份而不是用 mdtraj 的
# `_AMINO_ACID_CODES`：实测那张表里 `HID` / `HIE` / `ASH` / `CYX` / `NTRP` /
# `CCYS` 全都**不在**（只有 `HIP` 在），于是 `select("protein")` 会静默漏掉这些残基。
STANDARD_AMINO_ACID_CODES = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        # Amber 质子化/氧化态变体
        "HID", "HIE", "HIP", "ASH", "GLH", "CYX", "CYM", "LYN", "TYM", "ARN",
        # 封端基团
        "ACE", "NME", "NHE", "NH2",
    }
)


def normalize_protein_residue_name(name: str) -> Optional[str]:
    """把 `NTRP` / `CCYS` 这类 N-/C-端变体归一到标准三字母码；不是氨基酸则返回 None。"""
    upper = str(name).strip().upper()
    if upper in STANDARD_AMINO_ACID_CODES:
        return upper
    if len(upper) == 4 and upper[0] in ("N", "C") and upper[1:] in STANDARD_AMINO_ACID_CODES:
        return upper[1:]
    return None


def _normalize_ion_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().upper() if ch.isalpha())


def molecule_atom_ranges(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """由 `[ molecules ]` 顺序与各 moleculetype 的原子数算出每个分子的原子区间。

    GROMACS 的原子顺序就是 `[ molecules ]` 的展开顺序，所以这个映射是**精确**的，
    不需要靠键连通性或残基编号（实测那套体系里所有脂质的 `.gro` 残基编号都是 1，
    根本没法用残基编号分子）。

    已用 memtest/ 实测对齐：`PROA` 4566 个原子 → 第一个 POPC 原子在 index 4566，
    与 `step7_production.gro` 里第一个 `PA` 出现的位置逐一致。
    """
    ranges: List[Dict[str, Any]] = []
    cursor = 0
    for name, count in parsed["molecules"]:
        moltype = parsed["moleculetypes"].get(name)
        if moltype is None:
            raise ValueError(
                f"[ molecules ] 里的 {name!r} 找不到对应的 [ moleculetype ]，"
                "拓扑不自洽或有未解析的 include："
                f"{parsed.get('unresolved_includes')}"
            )
        size = int(moltype["n_atoms"])
        if size <= 0:
            raise ValueError(f"moleculetype {name!r} 的原子数为 {size}")
        for ordinal in range(int(count)):
            ranges.append(
                {
                    "molecule_name": name,
                    "ordinal": ordinal,
                    "start": cursor,
                    "stop": cursor + size,
                }
            )
            cursor += size
    return ranges


def preflight_physical_inputs(
    gro_path: str,
    top_path: str,
    ligand_molecule_name: str,
    *,
    gmx_include_dir: Optional[str] = None,
    cutoff_nm: float = 1.0,
    require_boresch: bool = False,
    log=print,
) -> Dict[str, Any]:
    """开跑前把 `.gro`/`.top` 的物理输入过一遍（ATT-24 / GitHub issue #64）。

    ## 为什么在这里判

    这五类问题原来都要等到**建完 System、甚至跑起来之后**才以别的面目暴露：
    未解析的 `#include` 变成"某个 moleculetype 找不到"、配体名打错变成
    "ligand_indices 为空"、盒子太小变成一个没人报错的**错误静电**。
    预检的意义就是把它们变成一条开跑前的、说得清怎么改的错误。

    ## 判什么

    1. **未解析的 `#include`** —— 拓扑不完整，后面一切分子计数都不可信。
    2. **配体 moleculetype 存在** —— `--ligand` 打错时给出候选名，而不是让它
       一路走到"配体 0 个原子"。
    3. **原子数对账** —— `[ molecules ]` 展开出来的总数必须等于 `.gro` 的原子数。
       对不上说明两个文件不是同一次建系的产物。
    4. **盒子 ≥ 2×cutoff** —— minimum image 约定的硬性下限。违反时 PME 会算出
       一个**不报错的错误静电**（同一对原子被自己的镜像重复作用），这是本函数
       里唯一"不查就永远查不出来"的一条。
    5. **Boresch 可构造性**（`require_boresch=True` 时）—— 配体至少 3 个原子、
       且存在一个非水非离子的受体分子，否则六个锚点根本定义不出来。

    **一次报全部**，不是遇到第一个就抛：输入错误往往成组出现，一次改完比来回五次好。

    Returns
    -------
    dict
        `{"n_atoms_topology", "n_atoms_gro", "box_nm", "min_box_edge_nm",
          "ligand_n_atoms", "molecule_names"}`
    """
    problems: List[str] = []

    for label, path in (("--gro", gro_path), ("--top", top_path)):
        if not path or not os.path.isfile(path):
            problems.append(f"{label} 文件不存在或不可读: {path!r}")
    if problems:
        raise ValueError("输入预检失败：\n  - " + "\n  - ".join(problems))

    parsed = parse_gromacs_topology(top_path, gmx_include_dir=gmx_include_dir)

    unresolved = parsed.get("unresolved_includes") or []
    if unresolved:
        problems.append(
            f"拓扑里有 {len(unresolved)} 条未解析的 #include: {unresolved[:5]}。"
            "缺力场目录时后面所有分子计数都不可信 —— 给 --gmx-path（GROMACS 安装"
            "前缀，不是 share/gromacs/top），或设置 GMXLIB / GMXDATA。"
        )

    moleculetypes = parsed.get("moleculetypes") or {}
    declared = [name for name, _count in (parsed.get("molecules") or [])]

    # 🔑 [2026-09-10] `--ligand` 是**残基名**，不是 [ moleculetype ] 名。
    #
    # 本仓其余全部消费点都按残基名取（runabfe 的 7 处都是
    # `atom.residue.name == ligand_resname`），而出厂 abfe_config.json 与所有
    # 文档示例写的是 `MOL`。真实体系里这两个名字经常不同：memtest 那套的
    # [ moleculetype ] 叫 `Atenolol-rank11`，[ atoms ] 第 4 列的 resname 才是 `MOL`。
    # 预检原来只查 moleculetype 名 ⟹ 传 MOL 被预检打死、传 Atenolol-rank11 又会在
    # 建系时"未在拓扑中找到配体残基"，两个值没有一个能同时过这两道门。
    # 这里两种都认：先按 moleculetype 名，再按残基名唯一匹配。
    ligand_moltype_name: Optional[str] = None
    if ligand_molecule_name in moleculetypes:
        ligand_moltype_name = ligand_molecule_name
    else:
        by_residue = [
            name
            for name, entry in moleculetypes.items()
            if ligand_molecule_name in (entry.get("residue_names") or [])
        ]
        if len(by_residue) == 1:
            ligand_moltype_name = by_residue[0]
        elif len(by_residue) > 1:
            problems.append(
                f"--ligand {ligand_molecule_name!r} 作为残基名同时出现在多个 "
                f"[ moleculetype ] 里：{sorted(by_residue)[:12]} —— 按序号取的原子"
                "选择会指向哪一个是不确定的，请改用唯一的那个 moleculetype 名。"
            )

    ligand_n_atoms = None
    if ligand_moltype_name is None:
        if not problems or "--ligand" not in problems[-1]:
            # 候选只列 moleculetype 名，外加**单残基** moleculetype 的残基名。
            # 多残基的（蛋白、POPC=PA+PC+OL）的残基名不是合法的 --ligand 取值，
            # 全列出来只会把真正的候选淹掉（一条蛋白链就是 20 个氨基酸名）。
            candidates = sorted(
                {
                    str(nm)
                    for name, entry in moleculetypes.items()
                    for nm in (
                        [name]
                        + list(
                            entry.get("residue_names") or []
                            if len(entry.get("residue_names") or []) == 1
                            else []
                        )
                    )
                    if str(nm).strip().upper() not in WATER_MOLECULE_NAMES
                    and _normalize_ion_name(nm) not in MONOATOMIC_ION_NAMES
                }
            )
            problems.append(
                f"--ligand {ligand_molecule_name!r} 在拓扑里既没有对应的 "
                f"[ moleculetype ]、也没有对应的残基名。非水非离子的候选："
                f"{candidates[:12]}"
            )
    elif ligand_moltype_name not in declared:
        problems.append(
            f"--ligand {ligand_molecule_name!r}（[ moleculetype ] "
            f"{ligand_moltype_name!r}）有定义，"
            "但没有出现在 [ molecules ] 里 —— 体系里其实没有这个分子。"
        )
    else:
        entry = moleculetypes[ligand_moltype_name]
        if ligand_moltype_name == ligand_molecule_name:
            ligand_n_atoms = int(entry["n_atoms"])
        else:
            # 按残基名匹配时只数该残基的原子：一个 moleculetype 可以含多个残基
            # （POPC = PA + PC + OL），整份 n_atoms 不是配体的原子数。
            ligand_n_atoms = sum(
                1
                for atom in (entry.get("atoms") or [])
                if atom.get("residue_name") == ligand_molecule_name
            )

    n_atoms_topology = None
    if not unresolved:
        try:
            ranges = molecule_atom_ranges(parsed)
            n_atoms_topology = ranges[-1]["stop"] if ranges else 0
        except ValueError as exc:
            problems.append(f"无法从拓扑展开原子区间: {exc}")

    # 只读 `.gro` 的两行（第 2 行 = 原子数，末行 = 盒矢量）。
    #
    # 不用 `app.GromacsGroFile`：它在**全零盒子**的 `.gro` 上抛
    # `ZeroDivisionError: float division by zero`（实测
    # tests/fixtures/memtest/Atenolol-rank11.gro），而"盒子是全零"恰恰是本函数
    # 要如实报告的一种输入，不该表现成一条看不懂的除零。格式本身是固定的，
    # 自己读这两行比绕开那个异常更短也更准。
    n_atoms_gro = None
    box_nm = None
    try:
        with open(gro_path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
        # 🔑 [2026-09-10] 末尾空行不是错。以 "\n\n" 结尾的 .gro（gmx 工具链、
        # 编辑器、cat 拼接都会产生）在这里会让 lines[-1] 变成 ""、进而报
        # ".gro 末行有 0 个数"，而 OpenMM 的 GromacsGroFile 读同一个文件毫无问题。
        while lines and not lines[-1].strip():
            lines.pop()
        if len(lines) < 3:
            raise ValueError(f".gro 只有 {len(lines)} 行，不是合法的 GROMACS 坐标文件")
        n_atoms_gro = int(lines[1].strip())
        box_fields = [float(x) for x in lines[-1].split()]
        if len(box_fields) not in (3, 9):
            raise ValueError(
                f".gro 末行有 {len(box_fields)} 个数，GROMACS 盒矢量应当是 3 或 9 个"
            )
        if len(box_fields) == 3:
            box_nm = [
                [box_fields[0], 0.0, 0.0],
                [0.0, box_fields[1], 0.0],
                [0.0, 0.0, box_fields[2]],
            ]
        else:
            # v1x v2y v3z v1y v1z v2x v2z v3x v3y
            xx, yy, zz, xy, xz, yx, yz, zx, zy = box_fields
            box_nm = [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]]
    except Exception as exc:  # noqa: BLE001 —— 解析失败本身就是要报的问题
        problems.append(f"读取 {gro_path} 失败: {type(exc).__name__}: {exc}")

    if n_atoms_topology is not None and n_atoms_gro is not None:
        if n_atoms_topology != n_atoms_gro:
            problems.append(
                f"原子数对不上：拓扑展开 {n_atoms_topology} 个，"
                f".gro 里 {n_atoms_gro} 个。两个文件不是同一次建系的产物 —— "
                "继续下去所有按序号取的原子选择（配体、co-ion、锚点）都会错位。"
            )

    min_edge = None
    if box_nm is not None:
        min_edge = min(
            float(np.linalg.norm(np.asarray(row, dtype=float))) for row in box_nm
        )
        required = 2.0 * float(cutoff_nm)
        if min_edge <= 0.0:
            problems.append(
                f"{gro_path} 的盒矢量是全零（末行 {box_nm}）—— 这个坐标文件没有"
                "周期盒子，不能直接用于 PME/ABFE。它多半是**未溶剂化的配体单体**，"
                "需要先建系（溶剂化 + 加离子）再跑。"
            )
        elif min_edge < required:
            problems.append(
                f"盒子最短边 {min_edge:.3f} nm < 2×cutoff = {required:.3f} nm，"
                "违反 minimum image 约定。PME 在这种盒子上**不会报错**，"
                "只会给出一个错误的静电（同一对原子被自己的周期镜像重复作用）。"
                "要么加大盒子重新建系，要么调小 cutoff。"
            )

    if require_boresch:
        if ligand_n_atoms is not None and ligand_n_atoms < 3:
            problems.append(
                f"配体只有 {ligand_n_atoms} 个原子，定义不出 Boresch 的三个配体锚点。"
            )
        receptor_like = [
            name
            for name, moltype in moleculetypes.items()
            if name != (ligand_moltype_name or ligand_molecule_name)
            and str(name).strip().upper() not in WATER_MOLECULE_NAMES
            and _normalize_ion_name(name) not in MONOATOMIC_ION_NAMES
            and int(moltype["n_atoms"]) >= 3
            and name in declared
        ]
        if not receptor_like:
            problems.append(
                "体系里除配体之外找不到任何 ≥3 原子的非水非离子分子 —— "
                "定义不出 Boresch 的三个受体锚点。溶剂腿不该传 --boresch。"
            )

    if problems:
        raise ValueError(
            "输入预检失败（一次列全，改完再跑）：\n  - " + "\n  - ".join(problems)
        )

    log(
        f"  [输入预检] 通过：拓扑 {n_atoms_topology} 原子 = .gro {n_atoms_gro} 原子；"
        f"配体 {ligand_molecule_name} {ligand_n_atoms} 原子；"
        + (
            f"盒子最短边 {min_edge:.3f} nm ≥ 2×cutoff {2.0 * float(cutoff_nm):.3f} nm。"
            if min_edge is not None
            else "（.gro 未声明盒矢量，跳过盒子检查）"
        )
    )
    return {
        "n_atoms_topology": n_atoms_topology,
        "n_atoms_gro": n_atoms_gro,
        "box_nm": box_nm,
        "min_box_edge_nm": min_edge,
        "ligand_n_atoms": ligand_n_atoms,
        "molecule_names": sorted(moleculetypes),
    }


def classify_system_composition(
    parsed: Dict[str, Any],
    ligand_molecule_name: Optional[str] = None,
    declared_roles: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """把每个 moleculetype 判定为 protein / lipid / water / ion / ligand。

    判据一律来自 `.top` 本身（moleculetype 的残基构成与原子数），不用残基名硬编码
    集合去猜整个体系。任何**判不出角色**的 moleculetype 一律 fail closed——
    静默归入 "other" 就等于让它在所有选择里消失。

    `declared_roles` 允许显式覆盖（例如非常规命名的脂质），覆盖会留记录。
    """
    declared_roles = {
        str(k): str(v).strip().lower() for k, v in (declared_roles or {}).items()
    }
    valid_roles = {"protein", "lipid", "water", "ion", "ligand"}
    bad = {k: v for k, v in declared_roles.items() if v not in valid_roles}
    if bad:
        raise ValueError(f"declared_roles 出现非法角色 {bad}；允许 {sorted(valid_roles)}")

    roles: Dict[str, str] = {}
    evidence: Dict[str, str] = {}
    for name, moltype in parsed["moleculetypes"].items():
        if name in declared_roles:
            roles[name] = declared_roles[name]
            evidence[name] = "declared_override"
            continue
        residues = [str(r).strip().upper() for r in moltype["residue_names"]]
        upper = str(name).strip().upper()

        if ligand_molecule_name and name == ligand_molecule_name:
            roles[name] = "ligand"
            evidence[name] = "matches_ligand_molecule_name"
        elif upper in WATER_MOLECULE_NAMES or (
            residues and all(r in WATER_MOLECULE_NAMES for r in residues)
        ):
            roles[name] = "water"
            evidence[name] = "water_molecule_name"
        elif moltype["n_atoms"] == 1 and _normalize_ion_name(name) in MONOATOMIC_ION_NAMES:
            roles[name] = "ion"
            evidence[name] = "monoatomic_ion_name"
        elif residues and all(r in KNOWN_LIPID_RESIDUE_NAMES for r in residues):
            # POPC = PA + PC + OL：三个残基全在脂质名表里 → 整个分子是脂质。
            roles[name] = "lipid"
            evidence[name] = "all_residues_are_lipid_residues"
        elif residues and any(
            normalize_protein_residue_name(r) is not None for r in residues
        ):
            roles[name] = "protein"
            evidence[name] = "contains_amino_acid_residues"
        else:
            raise ValueError(
                f"无法判定 moleculetype {name!r} 的角色"
                f"（残基 {residues[:8]}，{moltype['n_atoms']} 原子）。"
                "按 memtodolist 的纪律这里 fail closed，不静默归入 other——"
                "那会让它从所有原子选择里消失。请用 declared_roles 显式指定"
                "（protein / lipid / water / ion / ligand）。"
            )

    ranges = molecule_atom_ranges(parsed)
    by_role: Dict[str, List[int]] = {role: [] for role in valid_roles}
    molecules_by_role: Dict[str, List[Dict[str, Any]]] = {
        role: [] for role in valid_roles
    }
    for entry in ranges:
        role = roles[entry["molecule_name"]]
        by_role[role].extend(range(entry["start"], entry["stop"]))
        molecules_by_role[role].append(entry)

    counts = {}
    for name, count in parsed["molecules"]:
        counts[name] = counts.get(name, 0) + int(count)

    return {
        "roles": roles,
        "role_evidence": evidence,
        "molecule_counts": counts,
        "atom_indices_by_role": {k: sorted(v) for k, v in by_role.items()},
        "molecules_by_role": molecules_by_role,
        "n_atoms_total": ranges[-1]["stop"] if ranges else 0,
        "declared_roles": declared_roles,
    }


# ---------------------------------------------------------------------------
# `[ pairs ]` funct 2 → funct 1（OpenMM 兼容性转换）
# ---------------------------------------------------------------------------

GROMACS_PAIRS_FUNCT2_CONVERSION_VERSION = 1


def gromacs_topology_files(
    top_path: str,
    gmx_include_dir: Optional[str] = None,
) -> List[str]:
    """按 include 展开顺序返回拓扑树里全部文件的真实路径（去重、深度优先）。"""
    seen: List[str] = []
    include_re = re.compile(r'^\s*#\s*include\s+"([^"]+)"')

    def _walk(path: str):
        real = os.path.realpath(path)
        if real in seen:
            return
        seen.append(real)
        with open(real, encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                match = include_re.match(raw.split(";", 1)[0])
                if not match:
                    continue
                try:
                    _walk(resolve_gromacs_include(match.group(1), real, gmx_include_dir))
                except FileNotFoundError:
                    continue

    _walk(top_path)
    return seen


def _scan_moleculetype_charges(text: str) -> Dict[str, Dict[int, float]]:
    """逐 moleculetype 收集 `[ atoms ]` 的电荷（按该 moleculetype 内的局部原子号）。"""
    charges: Dict[str, Dict[int, float]] = {}
    section = None
    current: Optional[str] = None
    expecting_name = False
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            section = line.strip("[] ").strip().lower()
            expecting_name = section == "moleculetype"
            continue
        fields = line.split()
        if expecting_name:
            current = fields[0]
            charges.setdefault(current, {})
            expecting_name = False
        elif section == "atoms" and current and len(fields) >= 7:
            charges[current][int(fields[0])] = float(fields[6])
    return charges


def convert_gromacs_pairs_funct2(
    top_path: str,
    out_dir: str,
    gmx_include_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """把 `[ pairs ]` 的 funct 2 改写成 funct 1，写到 `out_dir`，**不改原始输入**。

    ## 为什么需要

    OpenMM 的 `app.GromacsTopFile` 只接受 `[ pairs ]` funct 1
    （`gromacstopfile.py::_processPair` 显式 `if fields[2] != '1': raise`）。
    CHARMM-GUI 的 AMBER FF-Converter 会对**部分**对写 funct 2：

        ai  aj  2  fudgeQQ  q1  q2  sigma14  epsilon14

    多出来的三列（fudgeQQ / q1 / q2）是**冗余重述**——OpenMM 算 1-4 exception 的电荷
    用的是「粒子电荷 × 全局 fudgeQQ」（`gromacstopfile.py` 里
    `atom1params[0]*atom2params[0]*fudgeQQ`），所以只要

        1. 逐对 fudgeQQ == `[ defaults ]` 的全局 fudgeQQ；
        2. q1/q2 == 该 moleculetype `[ atoms ]` 里的真实电荷，

    funct 2 与 `ai aj 1 sigma14 epsilon14` 就**严格等价**：sigma/eps 正好落在 OpenMM
    读取的 `fields[3:5]` 上。**这两个条件逐对校验，任一不成立就 fail closed**——
    不成立意味着该对真的覆盖了电荷或缩放因子，硬转会静默改变哈密顿量。

    ## 做法

    把整棵 include 树**逐文件拷贝**到 `out_dir`（保持相对路径，所以 `#include` 仍能
    解析），只改写含 funct-2 的文件里那几行。其余内容逐字节原样保留——包括
    `#ifdef POSRES` / `#ifdef DIHRES` 这类未激活的预处理块，不做展开、不做丢弃。

    原始输入一个字节都不动；缓存指纹仍应基于原始文件（本函数返回两侧的 SHA256
    以便一并落进 provenance）。
    """
    import hashlib
    import shutil

    parsed = parse_gromacs_topology(top_path, gmx_include_dir)
    defaults = parsed["defaults"] or {}
    global_fudge_qq = defaults.get("fudge_qq")
    if global_fudge_qq is None:
        raise ValueError(
            f"{top_path} 的拓扑树里找不到 `[ defaults ]` 的 fudgeQQ，"
            "无法校验 funct-2 pairs 是否与全局缩放一致 —— 拒绝盲转。"
        )

    files = gromacs_topology_files(top_path, gmx_include_dir)
    top_dir = os.path.dirname(os.path.realpath(top_path))
    outside = [f for f in files if not os.path.realpath(f).startswith(top_dir + os.sep)
               and os.path.realpath(f) != os.path.realpath(top_path)]
    if outside:
        raise ValueError(
            f"拓扑树里有文件位于 {top_dir} 之外：{outside[:3]}。"
            "逐文件拷贝无法保持它们的相对 include 路径，拒绝转换（需要先把它们收拢到"
            "拓扑目录下，或改用不含 funct-2 pairs 的拓扑）。"
        )

    os.makedirs(out_dir, exist_ok=True)
    patched: List[Dict[str, Any]] = []
    total_converted = 0

    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    for path in files:
        real = os.path.realpath(path)
        relative = (
            os.path.basename(real)
            if real == os.path.realpath(top_path)
            else os.path.relpath(real, top_dir)
        )
        destination = os.path.join(out_dir, relative)
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)

        text = open(real, encoding="utf-8", errors="replace").read()
        charges_by_moltype = _scan_moleculetype_charges(text)

        section = None
        current_moltype: Optional[str] = None
        expecting_name = False
        converted_here = 0
        out_lines: List[str] = []
        for raw in text.splitlines(keepends=True):
            body, _, comment = raw.partition(";")
            stripped = body.strip()
            if stripped.startswith("["):
                section = stripped.strip("[] ").strip().lower()
                expecting_name = section == "moleculetype"
                out_lines.append(raw)
                continue
            if not stripped or stripped.startswith("#"):
                out_lines.append(raw)
                continue
            fields = stripped.split()
            if expecting_name:
                current_moltype = fields[0]
                expecting_name = False
                out_lines.append(raw)
                continue
            if section != "pairs" or len(fields) < 3 or fields[2] != "2":
                out_lines.append(raw)
                continue

            if len(fields) < 8:
                raise ValueError(
                    f"{real} 的 funct-2 `[ pairs ]` 行字段不足 8 个，无法转换：{stripped!r}。"
                    "预期 `ai aj 2 fudgeQQ q1 q2 sigma14 epsilon14`。"
                )
            ai, aj = int(fields[0]), int(fields[1])
            pair_fudge, q1, q2 = float(fields[3]), float(fields[4]), float(fields[5])
            sigma14, epsilon14 = fields[6], fields[7]

            # 条件 1：逐对 fudgeQQ 必须等于全局值。
            if abs(pair_fudge - float(global_fudge_qq)) > 1.0e-6:
                raise ValueError(
                    f"{real} 的 pair ({ai},{aj}) 覆盖了 fudgeQQ："
                    f"逐对 {pair_fudge} vs 全局 {global_fudge_qq}。"
                    "OpenMM 的 1-4 exception 只用全局 fudgeQQ，硬转会**静默改变**"
                    "该对的静电缩放，拒绝转换。"
                )
            # 条件 2：q1/q2 必须等于该 moleculetype 的真实原子电荷。
            table = charges_by_moltype.get(current_moltype or "", {})
            for index, declared_q in ((ai, q1), (aj, q2)):
                actual = table.get(index)
                if actual is None:
                    raise ValueError(
                        f"{real} 的 moleculetype {current_moltype!r} 里找不到原子 {index} 的电荷，"
                        "无法校验 funct-2 pair 的等价性。"
                    )
                if abs(declared_q - actual) > 1.0e-6:
                    raise ValueError(
                        f"{real} 的 pair ({ai},{aj}) 覆盖了原子 {index} 的电荷："
                        f"pair 里写 {declared_q} 而 `[ atoms ]` 是 {actual}。"
                        "OpenMM 的 exception 用的是粒子电荷，硬转会静默改变哈密顿量，拒绝转换。"
                    )

            new_body = f"{fields[0]:>5s} {fields[1]:>5s}     1  {sigma14}  {epsilon14}"
            note = (
                "; [pairs funct 2 -> 1] 原始: "
                f"funct=2 fudgeQQ={fields[3]} q1={fields[4]} q2={fields[5]}"
                "（已校验与全局 fudgeQQ 及 [ atoms ] 电荷一致，故等价）"
            )
            out_lines.append(f"{new_body} {note}\n")
            converted_here += 1

        if converted_here:
            with open(destination, "w", encoding="utf-8") as handle:
                handle.write("".join(out_lines))
            patched.append(
                {
                    "source": real,
                    "destination": destination,
                    "n_pairs_converted": converted_here,
                    "source_sha256": _sha256(real),
                    "destination_sha256": _sha256(destination),
                }
            )
            total_converted += converted_here
        else:
            shutil.copy2(real, destination)

    converted_top = os.path.join(out_dir, os.path.basename(os.path.realpath(top_path)))
    return {
        "conversion_version": GROMACS_PAIRS_FUNCT2_CONVERSION_VERSION,
        "converted_top_path": converted_top,
        "n_pairs_converted": total_converted,
        "patched_files": patched,
        "global_fudge_qq": float(global_fudge_qq),
        "n_files_copied": len(files),
        "note": (
            "funct 2 的 fudgeQQ/q1/q2 三列经逐对校验与全局 fudgeQQ 及 [ atoms ] 电荷一致，"
            "属冗余重述，转成 funct 1 后 sigma14/epsilon14 落在 OpenMM 读取的 fields[3:5]，"
            "哈密顿量不变。原始输入未被修改。"
        ),
    }


def gromacs_file_has_funct2_pairs(path: str) -> bool:
    """**单个文件**自身是否含 `[ pairs ]` funct 2（不跟随 include）。

    与 `gromacs_topology_has_funct2_pairs`（整棵树）分开：定位"哪个 .itp 带"要用
    这个，判断"要不要走兼容性转换"要用那个。两者共用同一份扫描逻辑，只是作用域
    不同——不要各写一份，否则口径会分叉。
    """
    section = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.split(";", 1)[0].strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                section = line.strip("[] ").strip().lower()
                continue
            if section == "pairs":
                fields = line.split()
                if len(fields) >= 3 and fields[2] == "2":
                    return True
    return False


def gromacs_topology_has_funct2_pairs(
    top_path: str,
    gmx_include_dir: Optional[str] = None,
) -> bool:
    """**整棵**拓扑树里是否存在 `[ pairs ]` funct 2（决定要不要走兼容性转换）。

    ⚠️ 递归语义：对顶层 `.top` 调用它会因为 include 了带 funct-2 的 `.itp` 而返回
    True。想知道"哪个文件自己带"请用 `gromacs_file_has_funct2_pairs`。
    """
    return any(
        gromacs_file_has_funct2_pairs(path)
        for path in gromacs_topology_files(top_path, gmx_include_dir)
    )


def openmm_compatible_gromacs_top(
    top_file: str,
    gmx_include_dir: Optional[str] = None,
    compat_dir: Optional[str] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """返回 `(可直接交给 OpenMM 的 .top 路径, 转换信息或 None)`。**幂等、按内容缓存。**

    没有 funct-2 pairs 时原路返回，`None` 表示"未做任何转换"。

    `compat_dir` 不给时用系统临时目录下的**内容寻址**缓存
    （key = 整棵拓扑树各文件 sha256 + 转换版本号），所以：
      - 同一份输入多次调用只转换一次；
      - 输入变了 key 就变，不会复用过期的转换产物；
      - 不往用户的输入目录里写任何东西。

    `runabfe` 主路径会显式传 `output_dir/gromacs_openmm_compat` 以便审计。
    """
    import hashlib
    import tempfile

    if not gromacs_topology_has_funct2_pairs(top_file, gmx_include_dir):
        return top_file, None

    if compat_dir is None:
        digest = hashlib.sha256()
        digest.update(f"v{GROMACS_PAIRS_FUNCT2_CONVERSION_VERSION}\n".encode())
        top_dir = os.path.dirname(os.path.realpath(top_file))
        for path in gromacs_topology_files(top_file, gmx_include_dir):
            with open(path, "rb") as handle:
                file_digest = hashlib.sha256(handle.read()).hexdigest()
            relative = os.path.relpath(os.path.realpath(path), top_dir)
            digest.update(f"{relative}:{file_digest}\n".encode())
        compat_dir = os.path.join(
            tempfile.gettempdir(),
            "abfe_gromacs_openmm_compat",
            digest.hexdigest()[:16],
        )

    candidate = os.path.join(
        compat_dir, os.path.basename(os.path.realpath(top_file))
    )
    manifest_path = os.path.join(compat_dir, "conversion_manifest.json")

    # 复用之前必须核对**整棵树**是否完整且与当前输入一致。
    #
    # ⚠️ 早先这里只检查"顶层 top 存在 && 树里没有 funct-2"就直接复用——那是
    # fail-open：一次被中断的转换会留下半成品（部分文件已拷、部分没拷，或某个
    # `.itp` 被截断），而这两个条件仍可能成立。于是 OpenMM 读到一棵**不自洽**的
    # 拓扑，建出一个参数错乱的 System，最小化后 PE 到 1e13、几千步后变成一个
    # 没有上下文的 `Particle coordinate is NaN`。
    # 所以改成按 manifest 逐文件核对 sha256，任何不符就重转。
    if os.path.isfile(candidate) and os.path.isfile(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception:
            manifest = None
        if manifest and _conversion_manifest_matches(
            manifest, top_file, gmx_include_dir, compat_dir
        ):
            return candidate, {
                "reused_existing_conversion": True,
                "converted_top_path": candidate,
                "conversion_version": GROMACS_PAIRS_FUNCT2_CONVERSION_VERSION,
                "manifest_path": manifest_path,
            }
        logger.warning(
            "[WARN] %s 里已有的 funct-2 转换与当前输入不符（或不完整），重新转换。",
            compat_dir,
        )

    result = convert_gromacs_pairs_funct2(top_file, compat_dir, gmx_include_dir)
    _write_conversion_manifest(
        manifest_path, top_file, gmx_include_dir, compat_dir, result
    )
    result["manifest_path"] = manifest_path
    return result["converted_top_path"], result


def _conversion_tree_fingerprint(
    top_file: str,
    gmx_include_dir: Optional[str],
    compat_dir: str,
) -> Dict[str, Any]:
    """源树与转换产物的逐文件 sha256，用于核对复用是否安全。"""
    import hashlib

    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    top_dir = os.path.dirname(os.path.realpath(top_file))
    source: Dict[str, str] = {}
    converted: Dict[str, Optional[str]] = {}
    for path in gromacs_topology_files(top_file, gmx_include_dir):
        real = os.path.realpath(path)
        relative = (
            os.path.basename(real)
            if real == os.path.realpath(top_file)
            else os.path.relpath(real, top_dir)
        )
        source[relative] = _sha256(real)
        destination = os.path.join(compat_dir, relative)
        converted[relative] = (
            _sha256(destination) if os.path.isfile(destination) else None
        )
    return {
        "conversion_version": GROMACS_PAIRS_FUNCT2_CONVERSION_VERSION,
        "source": source,
        "converted": converted,
    }


def _write_conversion_manifest(
    manifest_path: str,
    top_file: str,
    gmx_include_dir: Optional[str],
    compat_dir: str,
    result: Dict[str, Any],
) -> None:
    fingerprint = _conversion_tree_fingerprint(top_file, gmx_include_dir, compat_dir)
    fingerprint["n_pairs_converted"] = int(result.get("n_pairs_converted", 0))
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(fingerprint, handle, indent=2, ensure_ascii=False)


def _conversion_manifest_matches(
    manifest: Dict[str, Any],
    top_file: str,
    gmx_include_dir: Optional[str],
    compat_dir: str,
) -> bool:
    if int(manifest.get("conversion_version", -1)) != (
        GROMACS_PAIRS_FUNCT2_CONVERSION_VERSION
    ):
        return False
    current = _conversion_tree_fingerprint(top_file, gmx_include_dir, compat_dir)
    # 源文件必须逐一相同（输入变了就不能复用旧转换）。
    if manifest.get("source") != current["source"]:
        return False
    # 转换产物必须**全部存在**且与当初写下的一致（挡住半成品 / 被改过的产物）。
    recorded = manifest.get("converted") or {}
    if set(recorded) != set(current["converted"]):
        return False
    for relative, digest in recorded.items():
        if digest is None or current["converted"].get(relative) != digest:
            return False
    return True


def load_gromacs_topology_for_openmm(
    top_file: str,
    includeDir: Optional[str] = None,
    compat_dir: Optional[str] = None,
    **kwargs,
):
    """**唯一**的 GROMACS 拓扑加载入口：需要时先做 funct-2 等价转换，再交给 OpenMM。

    ## 为什么必须只有一个入口

    OpenMM 的 `GromacsTopFile` 不支持 `[ pairs ]` funct 2。首次修这个问题时只在
    `build_system_from_gromacs` 一处接了转换，结果溶剂腿的
    `build_and_cache_solvent_leg`（另一个直接调 `app.GromacsTopFile` 的地方）
    照样炸——**这与 B1 当初只接了 1 个 `ABFEPipeline` 构造点是同一个毛病**：
    同一件事有多个入口，补一个漏一片。

    现在全仓所有加载点都走这里，并有契约测试禁止裸调 `app.GromacsTopFile`
    （见 `tests/test_gromacs_pairs_funct2_conversion.py`）。
    """
    resolved, _ = openmm_compatible_gromacs_top(top_file, includeDir, compat_dir)
    return app.GromacsTopFile(resolved, includeDir=includeDir, **kwargs)


def load_amber_topology_for_openmm(
    prmtop_file: str,
    inpcrd_file: Optional[str] = None,
    **kwargs,
):
    """加载 AMBER 拓扑（`.prmtop`/`.parm7` + 可选 `.inpcrd`/`.rst7`）为 OpenMM 对象。

    ## 范围：最小加载层，不是 GROMACS 路径的对等物

    这里只做"把一个真实 AMBER 拓扑/坐标文件交给 OpenMM 原生的
    `AmberPrmtopFile`/`AmberInpcrdFile`，让它能被 `createSystem()` 用起来"这一步。
    不做 `load_gromacs_topology_for_openmm` 那一整套 GROMACS 专属能力：

    - 没有 funct-2 pairs 等价转换——AMBER prmtop 本身就是 OpenMM 原生支持的
      二进制式段结构，不存在 `.top` 文本格式那种"某个 funct 变体 OpenMM 不认"
      的可转换性问题。
    - 没有 `classify_system_composition()` 式的角色分类 / co-ion 插入——这些
      是围绕 GROMACS `[ moleculetype ]`/`[ molecules ]` 文本结构写的，AMBER
      侧目前没有对应实现，也没有生产代码路径需要它。

    截至写这个函数时，全仓生产主链（`runabfe.py`/`abfe_pipeline.py`/
    `ibs_engine.py`）没有任何调用点读取 `.prmtop`/`.inpcrd`——这是为可行性
    探索新增的能力，尚未接入 CLI/生产流程。

    fail-closed：文件不存在直接抛 `FileNotFoundError`，不像 GROMACS include
    解析那样允许"找不到就返回 None、后面再报错"。

    Returns
    -------
    (prmtop, inpcrd) : tuple[app.AmberPrmtopFile, Optional[app.AmberInpcrdFile]]
        `prmtop`：`.topology` / `.createSystem(**kwargs)`。
        `inpcrd`：`.positions` / `.boxVectors`；未传 `inpcrd_file` 时为 `None`。
    """
    if not os.path.exists(prmtop_file):
        raise FileNotFoundError(f"AMBER prmtop 文件不存在: {prmtop_file}")
    prmtop = app.AmberPrmtopFile(prmtop_file, **kwargs)

    inpcrd = None
    if inpcrd_file is not None:
        if not os.path.exists(inpcrd_file):
            raise FileNotFoundError(f"AMBER inpcrd 文件不存在: {inpcrd_file}")
        inpcrd = app.AmberInpcrdFile(inpcrd_file)

    return prmtop, inpcrd


def _family_from_defaults(defaults: Optional[Dict[str, Any]]) -> Optional[str]:
    if not defaults:
        return None
    fudge_lj = defaults.get("fudge_lj")
    fudge_qq = defaults.get("fudge_qq")
    if fudge_lj is None or fudge_qq is None:
        return None
    for family, expect_lj, expect_qq in _FORCEFIELD_FAMILY_FUDGE_SIGNATURES:
        if (
            abs(float(fudge_lj) - expect_lj) <= _FUDGE_MATCH_TOLERANCE
            and abs(float(fudge_qq) - expect_qq) <= _FUDGE_MATCH_TOLERANCE
        ):
            return family
    return None


def detect_forcefield_family_from_top(
    top_path: str,
    gmx_include_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """从 GROMACS `.top` 判定力场族（§1.1）：`[ defaults ]` 为主，`#include` 为辅。

    **主判据是 `[ defaults ]` 的 1-4 缩放约定**（Amber 0.5/0.8333 vs CHARMM 1.0/1.0），
    递归跟随 include 去找它——`[ defaults ]` 通常不在顶层 `.top` 里，而在
    `forcefield.itp` 内。实测 CHARMM-GUI FF-Converter 产出的 AMBER 体系其 include
    路径完全不含 `amber` 字样，只有 `[ defaults ]` 能判对。

    include 路径 token 作为**次要**信号：两者一致则 `agree`，只有其一则用它，
    冲突则返回 `family=None` 让上层 fail closed（这种情况必须人工裁决）。
    """
    parsed = parse_gromacs_topology(top_path, gmx_include_dir)

    includes: List[str] = []
    for source, line in _iter_topology_lines(top_path, gmx_include_dir):
        if line.startswith("#unresolved_include"):
            includes.append(line.split(None, 1)[1])
    # 顶层与各级文件里出现过的 include 名（用于路径 token 判据）。
    with open(top_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            stripped = raw.split(";", 1)[0].strip()
            if stripped.startswith("#include"):
                parts = stripped.split('"')
                if len(parts) >= 2:
                    includes.append(parts[1])
    for path in parsed["files"]:
        includes.append(os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path))

    families: Dict[str, List[str]] = {}
    for include in includes:
        lowered = include.lower()
        for token, family in sorted(
            _FORCEFIELD_FAMILY_TOKENS, key=lambda kv: -len(kv[0])
        ):
            if token in lowered:
                families.setdefault(family, []).append(include)
                break

    path_families = sorted(families)
    path_family = path_families[0] if len(path_families) == 1 else None
    defaults_family = _family_from_defaults(parsed["defaults"])

    if defaults_family and path_family:
        if defaults_family == path_family:
            detected, reason = defaults_family, "defaults_and_include_agree"
        else:
            detected, reason = None, (
                f"conflict:defaults={defaults_family},include={path_family}"
            )
    elif defaults_family:
        detected, reason = defaults_family, "defaults_1_4_scaling_signature"
    elif path_family:
        detected, reason = path_family, "single_family_include"
    elif len(path_families) > 1:
        detected, reason = None, f"mixed_family_includes:{path_families}"
    else:
        detected, reason = None, "no_defaults_signature_and_no_recognized_include"

    return {
        "family": detected,
        "reason": reason,
        "includes": sorted(set(includes)),
        "family_evidence": {k: sorted(set(v)) for k, v in families.items()},
        "defaults": parsed["defaults"],
        "defaults_row": (parsed["defaults"] or {}).get("raw"),
        "defaults_family": defaults_family,
        "include_family": path_family,
        "top_path": str(top_path),
    }


def resolve_forcefield_family(
    top_path: Optional[str] = None,
    explicit_family: Optional[str] = None,
    gmx_include_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """确定力场族；识别不出即 fail closed，允许显式覆盖但必须留记录（§1.1）。"""
    detection: Dict[str, Any] = {"family": None, "reason": "not_attempted"}
    if top_path:
        detection = detect_forcefield_family_from_top(top_path, gmx_include_dir)
    detection["gmx_include_dir"] = (
        os.path.abspath(gmx_include_dir) if gmx_include_dir else None
    )

    if explicit_family is not None and str(explicit_family).strip():
        override = str(explicit_family).strip().lower()
        if override in FORCEFIELD_FAMILIES_UNSUPPORTED:
            raise ValueError(
                f"forcefield_family={override!r} 本轮不支持（只支持 "
                f"{list(FORCEFIELD_FAMILIES)}）。"
            )
        if override not in FORCEFIELD_FAMILIES:
            raise ValueError(
                f"forcefield_family={override!r} 非法；允许 {list(FORCEFIELD_FAMILIES)}。"
            )
        # §1.1：覆盖必须留记录，不能静默。
        if detection.get("family") and detection["family"] != override:
            logger.warning(
                "[WARN] forcefield_family 被显式覆盖：自动识别为 %r（依据 %s），"
                "但用户指定 %r。覆盖已记入 provenance。",
                detection["family"], detection["reason"], override,
            )
        return {
            "family": override,
            "source": "explicit_override",
            "overrode_detection": detection.get("family"),
            "detection": detection,
        }

    family = detection.get("family")
    if family in FORCEFIELD_FAMILIES_UNSUPPORTED:
        raise ValueError(
            f"从 {top_path} 识别出力场族 {family!r}，本轮不支持"
            f"（只支持 {list(FORCEFIELD_FAMILIES)}）。"
        )
    if family is None:
        raise ValueError(
            f"无法从 {top_path!r} 识别力场族（原因：{detection.get('reason')}；"
            f"include 列表 {detection.get('includes')}）。"
            "按 memtodolist §1.1 这里 fail closed：**不许猜、不许默认回落到 amber**。"
            "请用 --forcefield-family 显式指定（会记入 provenance）。"
        )
    return {
        "family": family,
        "source": "auto_detected",
        "overrode_detection": None,
        "detection": detection,
    }


def resolve_dispersion_protocol(
    dispersion_protocol: Optional[str],
    environment_type: Optional[str] = None,
    forcefield_family: Optional[str] = None,
    force_switch_deviation_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """校验 LJ/色散路线（§1.3 / B6）。

    默认值刻意**按环境类型分叉**，以保证 §7.7：
      - soluble 未声明 → `legacy_uniform_density_lrc`，即改动前的
        `lrc_coeff[k]/V(t)` 行为，可溶路径逐位不变；
      - membrane 未声明 → **fail closed**（§1.3 明文要求）。
    """
    resolved_environment = resolve_environment_type(environment_type)
    is_membrane = resolved_environment == ENVIRONMENT_TYPE_MEMBRANE

    if dispersion_protocol is None or str(dispersion_protocol).strip() == "":
        if is_membrane:
            expected = (
                FORCEFIELD_FAMILY_DISPERSION_PROTOCOL.get(forcefield_family)
                if forcefield_family
                else None
            )
            raise ValueError(
                "system_type=membrane 但没有选择 dispersion_protocol —— fail closed（§1.3）。"
                f"已验证可用的只有 {list(MEMBRANE_VALIDATED_DISPERSION_PROTOCOLS)}"
                + (f"；按识别到的力场族 {forcefield_family!r} 应选 {expected!r}。" if expected else "。")
            )
        resolved = DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC
        was_defaulted = True
    else:
        resolved = str(dispersion_protocol).strip().lower()
        was_defaulted = False

    if resolved not in DISPERSION_PROTOCOLS:
        raise ValueError(
            f"dispersion_protocol={dispersion_protocol!r} 非法；"
            f"允许 {list(DISPERSION_PROTOCOLS)}。"
        )

    if resolved in (
        DISPERSION_PROTOCOL_LJ_PME,
        DISPERSION_PROTOCOL_MEMBRANE_INHOMOGENEOUS,
    ):
        route = "B（LJ-PME）" if resolved == DISPERSION_PROTOCOL_LJ_PME else "C（膜非均匀色散修正）"
        raise NotImplementedError(
            f"dispersion_protocol={resolved} 属 §1.3 路线 {route}，尚未实现。"
            "该路线的四项/三项验收条件清单里已写明，但代码不存在；"
            "在完成之前不能只把基础 NonbondedForce 切过去就宣称支持。"
        )

    # §6.4 / §1.3：membrane + legacy uniform-density LRC 必须 fail closed。
    if is_membrane and resolved == DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC:
        raise ValueError(
            "system_type=membrane 不得使用 legacy_uniform_density_lrc。"
            "现有 `lj_tail_lrc_coeff[k]/V(t)` 假设配体周围是**均匀体相密度**；"
            "配体埋在脂双层口袋里时这个假设直接不成立——局域密度既不是水也不是体相脂质。"
            "这是膜体系下的真实缺陷，不是保守选项（§1.3 修正框第 1 条、§6.4）。"
        )

    if is_membrane and resolved not in MEMBRANE_VALIDATED_DISPERSION_PROTOCOLS:
        raise ValueError(
            f"system_type=membrane + dispersion_protocol={resolved} 尚未验证。"
            f"已验证的只有 {list(MEMBRANE_VALIDATED_DISPERSION_PROTOCOLS)}（§1.3）。"
        )

    # charmm 的 force-switch 无法在 OpenMM 复现，默认卡住（§1.1 / §1.3 路线 A 末条）。
    if resolved == DISPERSION_PROTOCOL_FF_NATIVE_FORCE_SWITCH_NO_LRC:
        missing = [
            field
            for field in FORCE_SWITCH_DEVIATION_EVIDENCE_FIELDS
            if not (force_switch_deviation_evidence or {}).get(field)
        ]
        if missing:
            raise ValueError(
                "dispersion_protocol=ff_native_force_switch_no_lrc 默认 fail closed："
                "OpenMM 的 NonbondedForce 只有 potential-switch"
                "（setUseSwitchingFunction），**没有 force-switch**，"
                "无法复现 CHARMM36 脂质的原始 Hamiltonian；用 potential-switch 顶替会"
                "移动 APL 与膜厚，且不会报错。"
                f"放行需要定量偏差论证，当前缺少 {missing}（§1.1 / §1.3）。"
                "首选路径是改用 Amber 系脂质力场（与现有 amber14sb 同族）。"
            )

    # 与力场族交叉核对：族与路线不匹配就是用错了参数化条件。
    if forcefield_family:
        family = str(forcefield_family).strip().lower()
        if family not in FORCEFIELD_FAMILIES:
            raise ValueError(
                f"forcefield_family={forcefield_family!r} 非法；允许 {list(FORCEFIELD_FAMILIES)}。"
            )
        expected = FORCEFIELD_FAMILY_DISPERSION_PROTOCOL[family]
        if resolved not in (expected, DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC):
            raise ValueError(
                f"力场族 {family!r} 的原始参数化条件对应 dispersion_protocol={expected!r}，"
                f"但声明的是 {resolved!r}。§1.3 的判据是**跟随所选力场的原始参数化条件**："
                "Amber Lipid21 是在开着各向同性 LRC 下拟合的，关掉 LRC 才是错的；"
                "CHARMM36 是 force-switch 且不加 LRC，开 LRC 才是错的。"
            )

    return {
        "dispersion_protocol": resolved,
        "was_defaulted": bool(was_defaulted),
        "environment_type": resolved_environment,
        "forcefield_family": (
            str(forcefield_family).strip().lower() if forcefield_family else None
        ),
        "implemented": resolved in DISPERSION_PROTOCOLS_IMPLEMENTED,
        # ⚠️ 这个键说的是"**目标**是不是 legacy 那条均匀密度 LRC 路线"，
        # **不是**"某一条腿实际加不加解析尾项"。后者要按腿的环境判，见
        # `resolve_leg_dispersion_implementation()`。两者曾被混为一谈（B6-FIX）。
        "uniform_density_lrc_active": (
            resolved == DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC
        ),
        # §5 最后一条 / §6.5：APBS 与 LJ 色散正交，不得互相顶替。
        "apbs_is_orthogonal_to_dispersion": True,
    }
