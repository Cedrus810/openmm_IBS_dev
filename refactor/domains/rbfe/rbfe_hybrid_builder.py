# ---------------------------------------------------------------------------
# R1b：受限 hybrid builder
# ---------------------------------------------------------------------------
#
# 计划 §5.3 的要求，逐条：
#   - 明确 common core / A-only / B-only，以及各自的 charge、LJ、bonded、
#     exception/1-4、约束和 dummy 处理；
#   - 路径至少描述 charge、sterics 和 bonded 项；
#   - **核对每个 λ 的有效总电荷**；
#   - 验证 λ 端点的物理相互作用恢复；
#   - 保持跨 λ 相同的粒子、质量和约束结构。
#
# ## 与 ABFE 引擎的关系（2026-09-03 用户决定）
#
# 用户明确：**直接 import `abfe_core` 复用，但不改 ABFE 一行代码。**
# 所以本节 import 是**只读**的，且一律**惰性**（在函数体内 import）——
# `abfe_core` 会拉进 openmm，放模块顶部会让 R0/R1a 那两层「不 import openmm、
# 不启动 GPU」的性质连同它们的全部测试一起失效。
#
# 复用清单（都在 ABFE 那边久经生产验证，重写必错）：
#   - `abfe_core.create_ligand_internal_force`：分子内非键 + 1-4 恢复，排除表
#     全覆盖 1-2/1-3/1-4 + **系统级约束** + exception 表
#   - `abfe_core.BeutlerSoftcoreBuilder`：softcore 表达式与 cutoff/switching 约定
#   - `ibs_engine._collect_softcore_exclusions`：跨组排除对收集
#
# ## 从 ABFE 抄 exception offset 时必须改掉的那一处
#
# `ibs_engine.py:3583` 那段的判据是 `(p1 in alchemical) ^ (p2 in alchemical)`——
# **异或**，即只处理「单端炼金」的 exception，配套还有
# `_assert_frozen_ligand_ligand_exceptions` 去断言这个前提。系数写的是
# `base=0, scale=chargeProd`，也就是插到**零**（ABFE 的端点）。
#
# RBFE 的端点不是零，是 B。所以：
#   - `addParticleParameterOffset` / `addExceptionParameterOffset` 这个**机制**
#     照用（它保 PME，是这条路上唯一正确的做法）；
#   - **系数必须换成** `base = 值ᴬ, scale = 值ᴮ − 值ᴬ`；
#   - 那条「单端」前提在 RBFE 里不成立（core–core exception 两端都在变），
#     所以 `_assert_frozen_ligand_ligand_exceptions` 不能照搬。
#
# 顺带澄清一个容易搞错的点：exception 的 `chargeProd` 是**独立参数**，不读粒子
# 电荷。所以「chargeProd 随 λ 线性插值」与「q1(λ)·q2(λ) 是 λ 的二次式」并不矛盾——
# 前者只是选了另一条路径，两个端点仍然精确。自由能只取决于端点，路径可以自选，
# 前提是平滑且无奇点。真正会出错的是**照抄系数**（那会插到零而不是插到 B）。

#: hybrid builder 的协议版本。进 hybrid System 的身份指纹。
#: 任何改变力的构造、λ 语义或分组规则的改动都必须 +1。
RBFE_HYBRID_PROTOCOL_VERSION = 1

#: 三条 λ 的全局参数名。**故意跟 ABFE 的 `lambda_coul` / `lambda_vdw` 不同名**：
#: 同名会让一个体系里两套炼金语义悄悄共享同一个全局参数。
LAMBDA_CHARGE = "lambda_rbfe_charge"
LAMBDA_STERICS = "lambda_rbfe_sterics"
LAMBDA_BONDED = "lambda_rbfe_bonded"
RBFE_LAMBDA_NAMES = (LAMBDA_CHARGE, LAMBDA_STERICS, LAMBDA_BONDED)


class RBFEHybridBuildError(RBFEValidationError):
    """hybrid 体系无法在**可证明正确**的前提下构建时抛出。

    首版受限 builder 的原则：碰到没验证过的输入就拒绝，不"尽力而为"。
    计划 §5.3 最后一句——「无法证明约束／dummy 贡献正确的变换不放行」。
    """


def _all_forces_of_type(system, force_type) -> list:
    """取出**全部**该类型的力，不是第一个。

    🔑 [P0-13] OpenMM 的 System 里同一类型的力可以有多个（GROMACS 拓扑经常
    产生两个 HarmonicBondForce）。ABFE 那边踩过一次 `next()` 只取第一个、
    配体键角被静默丢掉、分子变软的坑。这里一次都不许再踩。
    """
    return [f for f in system.getForces() if type(f) is force_type]


@dataclass(frozen=True)
class HybridTopologyLayout:
    """hybrid 体系的原子布局（计划 §5.2 的第四套索引）。

    索引约定：**环境与配体 A 的原子保持它们在 `system_a` 里的全局索引不变**，
    B-only 原子追加到末尾。这样 A 的坐标、A 的所有力项都不需要重映射，
    出错面小；代价只是 B-only 原子在编号上不与 B 的原始顺序一致（无所谓，
    映射对象本来就是身份来源）。
    """

    n_particles: int
    #: 配体 A 的分子内索引 -> hybrid 全局索引
    a_local_to_hybrid: dict
    #: 配体 B 的分子内索引 -> hybrid 全局索引
    b_local_to_hybrid: dict
    core: tuple
    a_only: tuple
    b_only: tuple
    environment: tuple

    @property
    def alchemical(self) -> tuple:
        return tuple(sorted(set(self.core) | set(self.a_only) | set(self.b_only)))

    def to_dict(self) -> dict:
        return {
            "n_particles": int(self.n_particles),
            "n_core": len(self.core),
            "n_A_only": len(self.a_only),
            "n_B_only": len(self.b_only),
            "n_environment": len(self.environment),
            "core_hybrid_indices": list(self.core),
            "A_only_hybrid_indices": list(self.a_only),
            "B_only_hybrid_indices": list(self.b_only),
        }


def _nb_force_of(system, label: str):
    """取唯一的 NonbondedForce。多于一个就拒绝——首版不处理拆分静电的体系。"""
    import openmm

    forces = _all_forces_of_type(system, openmm.NonbondedForce)
    if len(forces) != 1:
        raise RBFEHybridBuildError(
            f"{label} 有 {len(forces)} 个 NonbondedForce，首版只支持恰好 1 个。"
            "（0 个说明不是可用的力场体系；多个说明静电被拆过，"
            "本 builder 没验证过那种输入。）"
        )
    return forces[0]


def _assert_supported_forces(system, label: str) -> None:
    """首版受限 builder 只认这几种力，其余一律拒绝（计划 §2 最后一条）。"""
    import openmm

    supported = (
        openmm.HarmonicBondForce,
        openmm.HarmonicAngleForce,
        openmm.PeriodicTorsionForce,
        openmm.NonbondedForce,
        openmm.CMMotionRemover,
        openmm.MonteCarloBarostat,
    )
    unsupported = sorted(
        {type(f).__name__ for f in system.getForces() if not isinstance(f, supported)}
    )
    if unsupported:
        raise RBFEHybridBuildError(
            f"{label} 含首版 hybrid builder 未验证的力：{unsupported}。"
            "计划 §2：力场或自定义 Force 不在已验证 builder 支持范围内的输入一律拒绝。"
            "（这是本项目首版的范围限制，不是 RBFE 方法不支持。）"
        )
    for i in range(system.getNumParticles()):
        if system.isVirtualSite(i):
            raise RBFEHybridBuildError(
                f"{label} 的粒子 {i} 是虚拟位点，首版 hybrid builder 未验证（计划 §2）"
            )


def _quantity_value(value):
    """把 openmm Quantity 转成裸浮点（按其自身单位系统的默认单位）。"""
    from openmm import unit

    if hasattr(value, "value_in_unit_system"):
        return value.value_in_unit_system(unit.md_unit_system)
    return float(value)


def _assert_environment_matches(
    system_a, ligand_indices_a, system_b, ligand_indices_b
) -> tuple:
    """A、B 两个体系的**环境必须逐位相同**，否则两腿的 ΔG 差里混进了环境差异。

    比对：粒子数、质量、非键参数、环境内部的成键项、以及约束。
    返回 (env_a, env_b)：两侧环境原子索引，按顺序一一对应。

    为什么这么严：RBFE 的全部意义是「同一个环境里 A 换成 B」。环境只要有一位
    不同，ΔΔG 里就掺了一个不属于这条边的量，而且它在两腿里**不抵消**。
    """
    from openmm import unit

    set_a = {int(i) for i in ligand_indices_a}
    set_b = {int(i) for i in ligand_indices_b}
    env_a = [i for i in range(system_a.getNumParticles()) if i not in set_a]
    env_b = [i for i in range(system_b.getNumParticles()) if i not in set_b]
    if len(env_a) != len(env_b):
        raise RBFEHybridBuildError(
            f"两个体系的环境原子数不同：A 有 {len(env_a)}、B 有 {len(env_b)}。"
            "RBFE 要求同一个环境里 A 换成 B；环境不同的话 ΔΔG 里会混进环境差异，"
            "而且它在两腿里不抵消。"
        )

    for k, (ia, ib) in enumerate(zip(env_a, env_b)):
        ma = _quantity_value(system_a.getParticleMass(ia))
        mb = _quantity_value(system_b.getParticleMass(ib))
        if abs(ma - mb) > 1e-9:
            raise RBFEHybridBuildError(
                f"环境第 {k} 个原子（A#{ia} / B#{ib}）质量不同：{ma} vs {mb}"
            )

    nb_a, nb_b = _nb_force_of(system_a, "配体 A 体系"), _nb_force_of(system_b, "配体 B 体系")
    for k, (ia, ib) in enumerate(zip(env_a, env_b)):
        pa = [_quantity_value(v) for v in nb_a.getParticleParameters(ia)]
        pb = [_quantity_value(v) for v in nb_b.getParticleParameters(ib)]
        if max(abs(x - y) for x, y in zip(pa, pb)) > 1e-9:
            raise RBFEHybridBuildError(
                f"环境第 {k} 个原子（A#{ia} / B#{ib}）非键参数不同：{pa} vs {pb}"
            )

    map_a = {index: k for k, index in enumerate(env_a)}
    map_b = {index: k for k, index in enumerate(env_b)}

    def env_bonded_signature(system, env_map, ligand_set) -> tuple:
        import openmm

        rows = []
        for force in _all_forces_of_type(system, openmm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                p1, p2, length, k = force.getBondParameters(i)
                if p1 in ligand_set or p2 in ligand_set:
                    continue
                key = tuple(sorted((env_map[p1], env_map[p2])))
                rows.append(("bond", key, round(_quantity_value(length), 12), round(_quantity_value(k), 9)))
        for force in _all_forces_of_type(system, openmm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                p1, p2, p3, theta, k = force.getAngleParameters(i)
                if {p1, p2, p3} & ligand_set:
                    continue
                ends = tuple(sorted((env_map[p1], env_map[p3])))
                rows.append(("angle", (ends, env_map[p2]), round(_quantity_value(theta), 12), round(_quantity_value(k), 9)))
        for force in _all_forces_of_type(system, openmm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, n, phase, k = force.getTorsionParameters(i)
                if {p1, p2, p3, p4} & ligand_set:
                    continue
                rows.append(
                    ("torsion", tuple(env_map[p] for p in (p1, p2, p3, p4)), int(n),
                     round(_quantity_value(phase), 12), round(_quantity_value(k), 9))
                )
        return tuple(sorted(rows, key=repr))

    sig_a = env_bonded_signature(system_a, map_a, set_a)
    sig_b = env_bonded_signature(system_b, map_b, set_b)
    if sig_a != sig_b:
        only_a = [r for r in sig_a if r not in set(sig_b)]
        only_b = [r for r in sig_b if r not in set(sig_a)]
        raise RBFEHybridBuildError(
            f"两个体系的环境成键项不同：只在 A 里的 {len(only_a)} 项、"
            f"只在 B 里的 {len(only_b)} 项。示例 A={only_a[:2]} B={only_b[:2]}"
        )

    def env_constraints(system, env_map, ligand_set) -> tuple:
        rows = []
        for i in range(system.getNumConstraints()):
            p1, p2, distance = system.getConstraintParameters(i)
            if p1 in ligand_set or p2 in ligand_set:
                continue
            rows.append((tuple(sorted((env_map[p1], env_map[p2]))), round(_quantity_value(distance), 12)))
        return tuple(sorted(rows))

    if env_constraints(system_a, map_a, set_a) != env_constraints(system_b, map_b, set_b):
        raise RBFEHybridBuildError("两个体系的环境约束不同")

    return tuple(env_a), tuple(env_b)


def build_hybrid_layout(
    system_a,
    ligand_indices_a,
    system_b,
    ligand_indices_b,
    mapping: AtomMapping,
) -> HybridTopologyLayout:
    """确定 hybrid 体系的原子布局，并把所有前置条件查干净。

    这一步不建任何力，只定索引——但**所有 fail-closed 判据都在这里**，
    后面建力的代码可以假定输入已经合法。
    """
    from openmm import unit

    _assert_supported_forces(system_a, "配体 A 体系")
    _assert_supported_forces(system_b, "配体 B 体系")

    list_a = [int(i) for i in ligand_indices_a]
    list_b = [int(i) for i in ligand_indices_b]
    if len(list_a) != mapping.n_atoms_a:
        raise RBFEHybridBuildError(
            f"ligand_indices_a 有 {len(list_a)} 个原子，映射里 A 有 {mapping.n_atoms_a} 个"
        )
    if len(list_b) != mapping.n_atoms_b:
        raise RBFEHybridBuildError(
            f"ligand_indices_b 有 {len(list_b)} 个原子，映射里 B 有 {mapping.n_atoms_b} 个"
        )
    if len(set(list_a)) != len(list_a) or len(set(list_b)) != len(list_b):
        raise RBFEHybridBuildError("ligand_indices 有重复")

    env_a, _env_b = _assert_environment_matches(system_a, list_a, system_b, list_b)

    a_local_to_hybrid = {local: list_a[local] for local in range(len(list_a))}
    core_b_to_a = {int(b): int(a) for a, b in mapping.core_pairs}

    b_local_to_hybrid = {}
    next_index = system_a.getNumParticles()
    b_only_hybrid = []
    for local_b in range(len(list_b)):
        if local_b in core_b_to_a:
            b_local_to_hybrid[local_b] = a_local_to_hybrid[core_b_to_a[local_b]]
        else:
            b_local_to_hybrid[local_b] = next_index
            b_only_hybrid.append(next_index)
            next_index += 1

    # 🔑 计划 §5.3：「保持跨 λ 相同的粒子、质量和约束结构」。core 原子在两个体系
    # 里必须是同一个粒子，质量不同就不是同一个粒子——元素已由映射验证保证相同，
    # 质量还不同说明输入的力场里有同位素或质量重分配（HMR），首版不处理。
    for local_a, local_b in mapping.core_pairs:
        ma = _quantity_value(system_a.getParticleMass(list_a[local_a]))
        mb = _quantity_value(system_b.getParticleMass(list_b[local_b]))
        if abs(ma - mb) > 1e-6:
            raise RBFEHybridBuildError(
                f"core 原子对（A 分子内 {local_a} / B 分子内 {local_b}）质量不同："
                f"{ma} vs {mb}。同位素或质量重分配（HMR）首版不处理——"
                "跨 λ 粒子质量必须相同（计划 §5.3）。"
            )

    return HybridTopologyLayout(
        n_particles=next_index,
        a_local_to_hybrid=dict(sorted(a_local_to_hybrid.items())),
        b_local_to_hybrid=dict(sorted(b_local_to_hybrid.items())),
        core=tuple(sorted(a_local_to_hybrid[a] for a, _ in mapping.core_pairs)),
        a_only=tuple(sorted(a_local_to_hybrid[i] for i in mapping.a_only)),
        b_only=tuple(sorted(b_only_hybrid)),
        environment=tuple(env_a),
    )




def _b_index_to_hybrid(system_a, ligand_indices_a, system_b, ligand_indices_b, layout) -> dict:
    """`system_b` 的**每个**粒子索引 -> hybrid 索引。

    环境按顺序对应（`_assert_environment_matches` 已保证两侧环境逐位相同），
    配体按冻结映射走。
    """
    set_a = {int(i) for i in ligand_indices_a}
    set_b = {int(i) for i in ligand_indices_b}
    env_a = [i for i in range(system_a.getNumParticles()) if i not in set_a]
    env_b = [i for i in range(system_b.getNumParticles()) if i not in set_b]
    result = {int(ib): int(ia) for ia, ib in zip(env_a, env_b)}
    for local_b, index_b in enumerate(int(i) for i in ligand_indices_b):
        result[index_b] = layout.b_local_to_hybrid[local_b]
    return result


def _bond_rows(system, index_map, alchemical_only=None) -> dict:
    """{(i,j) hybrid 对: (r0, k)}。同一对出现多次即拒绝——不知道该插值哪一个。"""
    import openmm

    rows = {}
    for force in _all_forces_of_type(system, openmm.HarmonicBondForce):
        for i in range(force.getNumBonds()):
            p1, p2, length, k = force.getBondParameters(i)
            key = tuple(sorted((index_map[int(p1)], index_map[int(p2)])))
            value = (_quantity_value(length), _quantity_value(k))
            if key in rows and rows[key] != value:
                raise RBFEHybridBuildError(
                    f"键 {key} 出现多次且参数不同：{rows[key]} vs {value}。"
                    "重复成键项的插值语义未定义，首版拒绝。"
                )
            rows[key] = value
    return rows


def _angle_rows(system, index_map) -> dict:
    import openmm

    rows = {}
    for force in _all_forces_of_type(system, openmm.HarmonicAngleForce):
        for i in range(force.getNumAngles()):
            p1, p2, p3, theta, k = force.getAngleParameters(i)
            a, b, c = (index_map[int(p)] for p in (p1, p2, p3))
            key = (min(a, c), b, max(a, c))
            value = (_quantity_value(theta), _quantity_value(k))
            if key in rows and rows[key] != value:
                raise RBFEHybridBuildError(
                    f"键角 {key} 出现多次且参数不同：{rows[key]} vs {value}，首版拒绝。"
                )
            rows[key] = value
    return rows


def _torsion_rows(system, index_map) -> list:
    """[(四元组, periodicity, phase, k)]。二面角**不去重**：同一四元组的多个
    周期项是正常的力场写法。"""
    import openmm

    rows = []
    for force in _all_forces_of_type(system, openmm.PeriodicTorsionForce):
        for i in range(force.getNumTorsions()):
            p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
            quad = tuple(index_map[int(p)] for p in (p1, p2, p3, p4))
            if quad[0] > quad[-1]:
                quad = tuple(reversed(quad))
            rows.append((quad, int(periodicity), _quantity_value(phase), _quantity_value(k)))
    return rows


#: 力组约定。dummy 的成键项**必须**在自己的力组里，否则端点等价性无法精确验证：
#: λ=0 时 hybrid 比纯 A 体系多出来的，恰好就是 B 侧 dummy 的成键项（可分离因子）。
RBFE_FORCE_GROUP_DEFAULT = 0
RBFE_FORCE_GROUP_DUMMY_A = 1
RBFE_FORCE_GROUP_DUMMY_B = 2


def _add_bonded_forces(hybrid, system_a, system_b, layout, a_map, b_map) -> dict:
    """把成键项装进 hybrid 体系。

    ## 分类规则（首版受限 builder 的核心约定）

    | 力项涉及的原子 | 处理 | 力组 |
    |---|---|---|
    | 纯环境 | 从 A 原样搬 | 0 |
    | 纯 core，A、B 参数相同 | 原样搬 | 0 |
    | 纯 core，A、B 参数不同 | `Custom*Force` 按 `lambda_rbfe_bonded` 插值参数 | 0 |
    | 涉及 A-only（dummy） | **全强度保留，永不缩放** | 1 |
    | 涉及 B-only（dummy） | **全强度保留，永不缩放** | 2 |

    最后两条是 dummy 能抵消的**前提**：dummy 原子的键、角、二面角在两个端点都保持
    全强度，于是它的构型积分是一个与 λ 无关的可分离因子，在 complex 与 solvent
    两腿之间严格相消（计划 §5.3 要求「明确可分离因子／修正及其在热力学循环中的
    抵消条件」）。一旦把 dummy 的成键项也跟着 λ 关掉，这个因子就变成 λ 相关的，
    抵消条件不再成立——OpenFE 官方文档警告的「dummy bonded 处理不抵消导致系统
    误差」正是这个。

    分到独立力组还有一个直接用处：`verify_hybrid_endpoints` 靠扣掉对侧 dummy 力组
    来做**精确**的端点比对，不用估计容差。

    ## 二面角为什么不插值参数而是双项缩放

    键/角的参数是 (k, r0)，A、B 一一对应，插值有明确意义。二面角不同：同一个四元组
    在 A、B 里可能有不同的周期项集合、不同的 phase，硬要配对再插值需要先解决
    「哪个周期项对哪个」，而且 phase 插值本身没有物理意义。所以纯 core 的二面角走
    **双项缩放**：A 的项乘 (1-λ)、B 的项乘 λ。λ=0 只剩 A 的、λ=1 只剩 B 的，
    端点严格正确，中间平滑，且完全不需要配对。
    """
    import openmm

    core = set(layout.core)
    a_only, b_only = set(layout.a_only), set(layout.b_only)
    alchemical = core | a_only | b_only

    plain_bond = openmm.HarmonicBondForce()
    plain_angle = openmm.HarmonicAngleForce()
    plain_torsion = openmm.PeriodicTorsionForce()
    dummy_forces = {}
    for side, group in (("A", RBFE_FORCE_GROUP_DUMMY_A), ("B", RBFE_FORCE_GROUP_DUMMY_B)):
        bond = openmm.HarmonicBondForce()
        angle = openmm.HarmonicAngleForce()
        torsion = openmm.PeriodicTorsionForce()
        for force in (bond, angle, torsion):
            force.setForceGroup(group)
        dummy_forces[side] = {"bond": bond, "angle": angle, "torsion": torsion}

    interp_bond = openmm.CustomBondForce(
        f"0.5*k*(r-r0)^2;"
        f" k=(1-{LAMBDA_BONDED})*kA + {LAMBDA_BONDED}*kB;"
        f" r0=(1-{LAMBDA_BONDED})*r0A + {LAMBDA_BONDED}*r0B"
    )
    for name in ("r0A", "kA", "r0B", "kB"):
        interp_bond.addPerBondParameter(name)
    interp_bond.addGlobalParameter(LAMBDA_BONDED, 0.0)

    interp_angle = openmm.CustomAngleForce(
        f"0.5*k*(theta-theta0)^2;"
        f" k=(1-{LAMBDA_BONDED})*kA + {LAMBDA_BONDED}*kB;"
        f" theta0=(1-{LAMBDA_BONDED})*t0A + {LAMBDA_BONDED}*t0B"
    )
    for name in ("t0A", "kA", "t0B", "kB"):
        interp_angle.addPerAngleParameter(name)
    interp_angle.addGlobalParameter(LAMBDA_BONDED, 0.0)

    torsion_a = openmm.CustomTorsionForce(
        f"(1-{LAMBDA_BONDED})*k*(1+cos(periodicity*theta-phase))"
    )
    torsion_b = openmm.CustomTorsionForce(
        f"{LAMBDA_BONDED}*k*(1+cos(periodicity*theta-phase))"
    )
    for force in (torsion_a, torsion_b):
        for name in ("periodicity", "phase", "k"):
            force.addPerTorsionParameter(name)
        force.addGlobalParameter(LAMBDA_BONDED, 0.0)

    bonds_a, bonds_b = _bond_rows(system_a, a_map), _bond_rows(system_b, b_map)
    angles_a, angles_b = _angle_rows(system_a, a_map), _angle_rows(system_b, b_map)

    # 🔑 前置条件：core–core 的**键连接必须完全一致**。不一致意味着公共核心内部有
    # 键生成/断裂——`validate_mapping` 的连通性判据管不到这个（它只查 core 是一整块，
    # 不查 A、B 的 core 内部连法是否相同）。首版拒绝。
    core_bonds_a = {key for key in bonds_a if set(key) <= core}
    core_bonds_b = {key for key in bonds_b if set(key) <= core}
    if core_bonds_a != core_bonds_b:
        raise RBFEHybridBuildError(
            "公共核心内部的键连接在 A、B 里不同："
            f"只在 A 里的 {sorted(core_bonds_a - core_bonds_b)}、"
            f"只在 B 里的 {sorted(core_bonds_b - core_bonds_a)}。"
            "这是 core 内部的成键/断键，首版拒绝（计划 §2）。"
        )

    stats = {"plain_bond": 0, "interp_bond": 0, "dummy_bond_A": 0, "dummy_bond_B": 0,
             "plain_angle": 0, "interp_angle": 0, "dummy_angle_A": 0, "dummy_angle_B": 0,
             "plain_torsion": 0, "torsion_A_scaled": 0, "torsion_B_scaled": 0,
             "dummy_torsion_A": 0, "dummy_torsion_B": 0, "env_torsion": 0}

    def dummy_side(atoms) -> Optional[str]:
        if atoms & a_only:
            return "A"
        if atoms & b_only:
            return "B"
        return None

    # --- 键 ---
    for key in sorted(set(bonds_a) | set(bonds_b)):
        atoms = set(key)
        side = dummy_side(atoms)
        if not (atoms & alchemical):
            r0, k = bonds_a[key]
            plain_bond.addBond(key[0], key[1], r0, k)
            stats["plain_bond"] += 1
        elif side is not None:
            source = bonds_a.get(key, bonds_b.get(key))
            dummy_forces[side]["bond"].addBond(key[0], key[1], source[0], source[1])
            stats[f"dummy_bond_{side}"] += 1
        else:
            row_a, row_b = bonds_a[key], bonds_b[key]
            if row_a == row_b:
                plain_bond.addBond(key[0], key[1], row_a[0], row_a[1])
                stats["plain_bond"] += 1
            else:
                interp_bond.addBond(key[0], key[1], [row_a[0], row_a[1], row_b[0], row_b[1]])
                stats["interp_bond"] += 1

    # --- 键角 ---
    for key in sorted(set(angles_a) | set(angles_b)):
        atoms = {key[0], key[1], key[2]}
        side = dummy_side(atoms)
        if not (atoms & alchemical):
            t0, k = angles_a[key]
            plain_angle.addAngle(key[0], key[1], key[2], t0, k)
            stats["plain_angle"] += 1
        elif side is not None:
            source = angles_a.get(key, angles_b.get(key))
            dummy_forces[side]["angle"].addAngle(key[0], key[1], key[2], source[0], source[1])
            stats[f"dummy_angle_{side}"] += 1
        else:
            row_a, row_b = angles_a.get(key), angles_b.get(key)
            if row_a is None or row_b is None:
                missing = "B" if row_a is not None else "A"
                raise RBFEHybridBuildError(
                    f"纯 core 的键角 {key} 在 {missing} 侧不存在。core 内部连接已校验"
                    "相同，出现这种情况说明力场对同一连接给了不同的键角项集合——"
                    "首版不猜该插到什么，拒绝。"
                )
            if row_a == row_b:
                plain_angle.addAngle(key[0], key[1], key[2], row_a[0], row_a[1])
                stats["plain_angle"] += 1
            else:
                interp_angle.addAngle(
                    key[0], key[1], key[2], [row_a[0], row_a[1], row_b[0], row_b[1]]
                )
                stats["interp_angle"] += 1

    # --- 二面角 ---
    #
    # 🔑 纯 core 且 A、B **逐位相同**的二面角走 native `PeriodicTorsionForce`，
    # 不塞进 `CustomTorsionForce`。原因不是省力：OpenMM 内建的二面角力对**退化几何**
    # （四个原子共线，二面角没有定义）是安全的，而 `CustomTorsionForce` 的解析导数
    # 在那里给 NaN。生产轨迹里出现瞬时近共线不是不可能，所以能用 native 的就用
    # native，把 Custom 的暴露面压到"真正随 λ 变的那几个"。
    rows_a = _torsion_rows(system_a, a_map)
    rows_b = _torsion_rows(system_b, b_map)
    core_torsions_a = {}
    core_torsions_b = {}

    def bucket(rows, sink, other_side_marker):
        for quad, periodicity, phase, k in rows:
            atoms = set(quad)
            side = dummy_side(atoms)
            if not (atoms & alchemical):
                if other_side_marker == "A":            # 环境项只从 A 搬一次
                    plain_torsion.addTorsion(*quad, periodicity, phase, k)
                    stats["env_torsion"] += 1
                continue
            if side is not None:
                dummy_forces[side]["torsion"].addTorsion(*quad, periodicity, phase, k)
                stats[f"dummy_torsion_{side}"] += 1
                continue
            key = (quad, int(periodicity), round(float(phase), 12))
            sink.setdefault(key, []).append(float(k))

    bucket(rows_a, core_torsions_a, "A")
    bucket(rows_b, core_torsions_b, "B")

    for key in sorted(set(core_torsions_a) | set(core_torsions_b), key=repr):
        quad, periodicity, phase = key
        ks_a = sorted(core_torsions_a.get(key, []))
        ks_b = sorted(core_torsions_b.get(key, []))
        shared = 0
        if ks_a == ks_b:                                # A、B 完全一样 → native
            for k in ks_a:
                plain_torsion.addTorsion(*quad, periodicity, phase, k)
                stats["plain_torsion"] += 1
            continue
        for k in ks_a:
            torsion_a.addTorsion(*quad, [float(periodicity), phase, k])
            stats["torsion_A_scaled"] += 1
        for k in ks_b:
            torsion_b.addTorsion(*quad, [float(periodicity), phase, k])
            stats["torsion_B_scaled"] += 1

    for force in (plain_bond, plain_angle, plain_torsion,
                  interp_bond, interp_angle, torsion_a, torsion_b):
        hybrid.addForce(force)
    for side in ("A", "B"):
        for force in dummy_forces[side].values():
            hybrid.addForce(force)
    return stats


def _add_constraints(hybrid, system_a, system_b, layout, a_map, b_map) -> int:
    """约束：环境与 A 侧照搬，B-only 的从 B 搬；core–core 约束两边必须一致。

    计划 §5.3：「保持跨 λ 相同的粒子、质量和**约束结构**」。约束距离没法随 λ
    插值（约束是硬的），所以 core–core 约束只要 A、B 不同就拒绝。
    """
    core = set(layout.core)
    b_only = set(layout.b_only)

    rows_a = {}
    for i in range(system_a.getNumConstraints()):
        p1, p2, distance = system_a.getConstraintParameters(i)
        rows_a[tuple(sorted((a_map[int(p1)], a_map[int(p2)])))] = _quantity_value(distance)
    rows_b = {}
    for i in range(system_b.getNumConstraints()):
        p1, p2, distance = system_b.getConstraintParameters(i)
        rows_b[tuple(sorted((b_map[int(p1)], b_map[int(p2)])))] = _quantity_value(distance)

    for key in set(rows_a) & set(rows_b):
        if set(key) <= core and abs(rows_a[key] - rows_b[key]) > 1e-9:
            raise RBFEHybridBuildError(
                f"core–core 约束 {key} 的距离在 A、B 里不同：{rows_a[key]} vs {rows_b[key]}。"
                "约束是硬的，没法随 λ 插值，首版拒绝（计划 §5.3）。"
            )
    merged = dict(rows_a)
    for key, distance in rows_b.items():
        if key in merged:
            continue
        if not (set(key) & b_only):
            raise RBFEHybridBuildError(
                f"约束 {key} 只存在于 B 侧，但它不涉及任何 B-only 原子。"
                "跨 λ 约束结构必须一致（计划 §5.3），首版拒绝。"
            )
        merged[key] = distance

    for (p1, p2), distance in sorted(merged.items()):
        hybrid.addConstraint(p1, p2, distance)
    return len(merged)




#: Beutler softcore 的无量纲 α，与 `abfe_core.BeutlerSoftcoreBuilder` 同一约定
#: （表达式内乘 sigma^6，不是绝对 nm^6）。
RBFE_SOFTCORE_ALPHA = 0.5


def _particle_endpoint_params(system_a, system_b, layout, a_map, b_map) -> dict:
    """每个 hybrid 粒子在两个端点的 (q, sigma, epsilon)。

    dummy 侧一律取 (0, 对侧的 sigma, 0)：**sigma 不取 0**——softcore 的分母里
    有 sigma^6，取 0 会让软化项整个消失，dummy 在消失过程中反而变成硬球。
    """
    nb_a, nb_b = _nb_force_of(system_a, "配体 A 体系"), _nb_force_of(system_b, "配体 B 体系")
    hybrid_a = {a_map[i]: i for i in range(system_a.getNumParticles())}
    hybrid_b = {b_map[i]: i for i in range(system_b.getNumParticles())}

    params = {}
    for h in range(layout.n_particles):
        side_a = hybrid_a.get(h)
        side_b = hybrid_b.get(h)
        if side_a is not None:
            qa, sa, ea = (_quantity_value(v) for v in nb_a.getParticleParameters(side_a))
        else:
            qa, sa, ea = None, None, None
        if side_b is not None:
            qb, sb, eb = (_quantity_value(v) for v in nb_b.getParticleParameters(side_b))
        else:
            qb, sb, eb = None, None, None
        if side_a is None:                       # B-only
            qa, sa, ea = 0.0, sb, 0.0
        if side_b is None:                       # A-only
            qb, sb, eb = 0.0, sa, 0.0
        params[h] = {"qA": qa, "sigA": sa, "epsA": ea, "qB": qb, "sigB": sb, "epsB": eb}
    return params


def _exception_rows(system, index_map) -> dict:
    import openmm

    nb = _nb_force_of(system, "体系")
    rows = {}
    for i in range(nb.getNumExceptions()):
        p1, p2, charge_prod, sigma, epsilon = nb.getExceptionParameters(i)
        key = tuple(sorted((index_map[int(p1)], index_map[int(p2)])))
        rows[key] = (
            _quantity_value(charge_prod),
            _quantity_value(sigma),
            _quantity_value(epsilon),
        )
    return rows


def _add_nonbonded_forces(hybrid, system_a, system_b, layout, a_map, b_map) -> dict:
    """非键层。

    ## 静电：全部留在 native `NonbondedForce`，用参数 offset 走 λ

    这是从 ABFE 抄的最关键一条（`ibs_engine.configure_pme_ligand_charge_offsets`
    的做法）：**不要**把配体静电搬进 `CustomNonbondedForce`——那会把库仑在 cutoff
    处截断，PME 倒空间就不对了。`addParticleParameterOffset` 让 OpenMM 自己在
    正空间和倒空间里都按 q(λ)=base+λ·scale 处理。

    系数按 RBFE 端点写：core 是 `base=qᴬ, scale=qᴮ−qᴬ`；A-only 是 `base=qᴬ,
    scale=−qᴬ`；B-only 是 `base=0, scale=qᴮ`。（ABFE 那边写的是 `base=0,
    scale=q`，插到零——那是 ABFE 的端点，照抄就错。）

    ## LJ：从 native 力上摘掉，拆成三个 CustomNonbondedForce

    | 力 | 覆盖的对 | λ 语义 |
    |---|---|---|
    | core | core×core、core×env | 参数插值，**不加 softcore**（两端都是实原子） |
    | A-only | A-only×(自身/core/env) | 前因子 (1−λ)，softcore lift λ |
    | B-only | B-only×(自身/core/env) | 前因子 λ，softcore lift (1−λ) |

    **A-only × B-only 不在任何 interaction group 里，也显式加了排除**——两组
    dummy 永远不能相互看见，这是 hybrid 拓扑的硬约束，ABFE 里没有对应概念。

    env×env 的 LJ 仍然留在 native 力上（连同它的长程校正），所以炼金原子在 native
    力上的 epsilon 被清零。代价见返回值里的 `alchemical_lj_lrc_included=False`。
    """
    import openmm
    from openmm import unit

    nb_a = _nb_force_of(system_a, "配体 A 体系")
    endpoint = _particle_endpoint_params(system_a, system_b, layout, a_map, b_map)
    core, a_only, b_only = set(layout.core), set(layout.a_only), set(layout.b_only)
    env = set(layout.environment)
    alchemical = core | a_only | b_only

    # ---------------- native NonbondedForce（静电 + env LJ） ----------------
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(nb_a.getNonbondedMethod())
    nb.setCutoffDistance(nb_a.getCutoffDistance())
    nb.setUseSwitchingFunction(nb_a.getUseSwitchingFunction())
    nb.setSwitchingDistance(nb_a.getSwitchingDistance())
    nb.setUseDispersionCorrection(nb_a.getUseDispersionCorrection())
    nb.setEwaldErrorTolerance(nb_a.getEwaldErrorTolerance())
    nb.setReactionFieldDielectric(nb_a.getReactionFieldDielectric())
    alpha, nx, ny, nz = nb_a.getPMEParameters()
    nb.setPMEParameters(alpha, nx, ny, nz)
    nb.addGlobalParameter(LAMBDA_CHARGE, 0.0)

    for h in range(layout.n_particles):
        row = endpoint[h]
        epsilon = 0.0 if h in alchemical else row["epsA"]
        nb.addParticle(
            row["qA"] * unit.elementary_charge,
            row["sigA"] * unit.nanometer,
            epsilon * unit.kilojoule_per_mole,
        )
        if h in alchemical:
            delta_q = row["qB"] - row["qA"]
            nb.addParticleParameterOffset(
                LAMBDA_CHARGE,
                h,
                delta_q * unit.elementary_charge,
                0.0 * unit.nanometer,
                0.0 * unit.kilojoule_per_mole,
            )

    # ---------------- exception / exclusion ----------------
    exc_a = _exception_rows(system_a, a_map)
    exc_b = _exception_rows(system_b, b_map)

    core_exc_a = {k for k in exc_a if set(k) <= core}
    core_exc_b = {k for k in exc_b if set(k) <= core}
    if core_exc_a != core_exc_b:
        raise RBFEHybridBuildError(
            "公共核心内部的 exception（1-2/1-3 排除与 1-4）在 A、B 里不一致："
            f"只在 A 里的 {sorted(core_exc_a - core_exc_b)}、"
            f"只在 B 里的 {sorted(core_exc_b - core_exc_a)}。"
            "同一份 core 连接却给出不同的排除表，首版拒绝。"
        )

    exception_pairs = sorted(set(exc_a) | set(exc_b))
    # A-only × B-only：两组 dummy 永不相互作用
    forbidden_pairs = sorted(
        (min(x, y), max(x, y)) for x in a_only for y in b_only
    )
    all_excluded = sorted(set(exception_pairs) | set(forbidden_pairs))

    pair14 = []
    for key in exception_pairs:
        row_a, row_b = exc_a.get(key), exc_b.get(key)
        cp_a, sig_a, eps_a = row_a if row_a else (0.0, (row_b or (0, 0.3, 0))[1], 0.0)
        cp_b, sig_b, eps_b = row_b if row_b else (0.0, sig_a, 0.0)
        is_alchemical = bool(set(key) & alchemical)
        native_eps = 0.0 if is_alchemical else eps_a
        index = nb.addException(
            key[0],
            key[1],
            cp_a * unit.elementary_charge**2,
            sig_a * unit.nanometer,
            native_eps * unit.kilojoule_per_mole,
        )
        if is_alchemical:
            nb.addExceptionParameterOffset(
                LAMBDA_CHARGE,
                index,
                (cp_b - cp_a) * unit.elementary_charge**2,
                0.0 * unit.nanometer,
                0.0 * unit.kilojoule_per_mole,
            )
            if eps_a != 0.0 or eps_b != 0.0:
                pair14.append((key, sig_a or 0.1, eps_a, sig_b or 0.1, eps_b))
    for key in forbidden_pairs:
        nb.addException(
            key[0], key[1],
            0.0 * unit.elementary_charge**2,
            0.3 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )
    hybrid.addForce(nb)

    # ---------------- LJ：三个 CustomNonbondedForce ----------------
    def new_lj_force(expression) -> "openmm.CustomNonbondedForce":
        force = openmm.CustomNonbondedForce(expression)
        for name in ("sigA", "epsA", "sigB", "epsB"):
            force.addPerParticleParameter(name)
        force.addGlobalParameter(LAMBDA_STERICS, 0.0)
        for h in range(layout.n_particles):
            row = endpoint[h]
            force.addParticle([row["sigA"], row["epsA"], row["sigB"], row["epsB"]])
        method = (
            openmm.CustomNonbondedForce.CutoffPeriodic
            if nb_a.getNonbondedMethod() in (
                openmm.NonbondedForce.PME,
                openmm.NonbondedForce.Ewald,
                openmm.NonbondedForce.CutoffPeriodic,
            )
            else openmm.CustomNonbondedForce.CutoffNonPeriodic
        )
        force.setNonbondedMethod(method)
        force.setCutoffDistance(nb_a.getCutoffDistance())
        force.setUseSwitchingFunction(nb_a.getUseSwitchingFunction())
        if nb_a.getUseSwitchingFunction():
            force.setSwitchingDistance(nb_a.getSwitchingDistance())
        # 长程校正一律关掉：softcore/插值参数下 OpenMM 的解析尾项公式不成立。
        # ABFE 那边是另算 `_lj_softcore_tail_radial_integrals`；R1b v1 不做，
        # 如实记在 provenance 里。
        force.setUseLongRangeCorrection(False)
        for p1, p2 in all_excluded:
            force.addExclusion(p1, p2)
        return force

    ls = LAMBDA_STERICS
    force_core = new_lj_force(
        "4*epsij*((sigij/r)^12-(sigij/r)^6);"
        " sigij=0.5*(si+sj);"
        " epsij=sqrt(ei*ej);"
        f" si=(1-{ls})*sigA1+{ls}*sigB1;"
        f" sj=(1-{ls})*sigA2+{ls}*sigB2;"
        f" ei=(1-{ls})*epsA1+{ls}*epsB1;"
        f" ej=(1-{ls})*epsA2+{ls}*epsB2"
    )
    force_a = new_lj_force(
        f"(1-{ls})*4*epsij*(sigij^12/D^2 - sigij^6/D);"
        f" D=r^6 + {RBFE_SOFTCORE_ALPHA}*sigij^6*{ls};"
        " sigij=0.5*(sigA1+sigA2);"
        " epsij=sqrt(epsA1*epsA2)"
    )
    force_b = new_lj_force(
        f"{ls}*4*epsij*(sigij^12/D^2 - sigij^6/D);"
        f" D=r^6 + {RBFE_SOFTCORE_ALPHA}*sigij^6*(1-{ls});"
        " sigij=0.5*(sigB1+sigB2);"
        " epsij=sqrt(epsB1*epsB2)"
    )

    core_and_env = core | env
    group_counts = {"core": 0, "a_only": 0, "b_only": 0}
    if core:
        force_core.addInteractionGroup(core, core)
        group_counts["core"] += 1
        if env:
            force_core.addInteractionGroup(core, env)
            group_counts["core"] += 1
    if a_only:
        force_a.addInteractionGroup(a_only, a_only)
        group_counts["a_only"] += 1
        if core_and_env:
            force_a.addInteractionGroup(a_only, core_and_env)
            group_counts["a_only"] += 1
    if b_only:
        force_b.addInteractionGroup(b_only, b_only)
        group_counts["b_only"] += 1
        if core_and_env:
            force_b.addInteractionGroup(b_only, core_and_env)
            group_counts["b_only"] += 1

    # 🔑 **一个 interaction group 都没有的 CustomNonbondedForce 会计算全体粒子对。**
    # 这是 OpenMM 的默认行为，不是"什么都不算"。所以 A-only 或 B-only 为空时
    # （比如 A→A 自边，或者任何"只改参数、不增删原子"的变换——那是很常见的一类边），
    # 空着的那个力会退化成对全部粒子求和，与 force_core 和 native 力**重复计数**。
    #
    # 这个 bug 一度被"正确答案"掩盖：A→A 的 ΔG 仍然算出 0，因为 λ 表对称、两个端点
    # 上 softcore lift 都为 0，两个力恰好互为镜像而抵消——**端点对了，中间态全错**。
    # 是 `test_self_edge_u_kn_rows_are_identical`（验中间态而不是验端点）抓到的。
    for label, force in (("core", force_core), ("a_only", force_a), ("b_only", force_b)):
        if group_counts[label] > 0:
            hybrid.addForce(force)
    nonbonded_group_counts = dict(group_counts)

    # ---------------- 1-4 的 LJ（从 native 摘出来的那部分） ----------------
    if pair14:
        f14 = openmm.CustomBondForce(
            f"(1-{ls})*4*epsA*((sigA/r)^12-(sigA/r)^6)"
            f" + {ls}*4*epsB*((sigB/r)^12-(sigB/r)^6)"
        )
        for name in ("sigA", "epsA", "sigB", "epsB"):
            f14.addPerBondParameter(name)
        f14.addGlobalParameter(LAMBDA_STERICS, 0.0)
        for (p1, p2), sig_a, eps_a, sig_b, eps_b in pair14:
            f14.addBond(p1, p2, [sig_a, eps_a, sig_b, eps_b])
        hybrid.addForce(f14)

    return {
        "n_exceptions": len(exception_pairs),
        "n_forbidden_dummy_pairs": len(forbidden_pairs),
        "lj_interaction_groups": nonbonded_group_counts,
        "lj_forces_added": [
            label for label, count in nonbonded_group_counts.items() if count > 0
        ],
        "n_lj14_terms": len(pair14),
        "native_dispersion_correction": bool(nb_a.getUseDispersionCorrection()),
        "alchemical_lj_lrc_included": False,
        "softcore_alpha": RBFE_SOFTCORE_ALPHA,
    }




@dataclass(frozen=True)
class HybridLambdaSchedule:
    """λ 路径。计划 §2 要求 **λ 协议显式配置，不沿用隐藏默认值**——所以这个对象
    必须由调用方显式构造，builder 不提供"没给就用默认"的行为。

    三条 λ 分开是因为它们的物理含义不同（计划 §5.3：「路径至少描述 charge、
    sterics 和 bonded 项」），而且分段切换会改变中间态的有效总电荷——那正是
    `hybrid_charge_ledger` 要盯的东西。
    """

    name: str
    charge: tuple
    sterics: tuple
    bonded: tuple

    def __post_init__(self) -> None:
        lengths = {len(self.charge), len(self.sterics), len(self.bonded)}
        if len(lengths) != 1:
            raise RBFEHybridBuildError(
                f"三条 λ 的态数不一致：charge={len(self.charge)}、"
                f"sterics={len(self.sterics)}、bonded={len(self.bonded)}"
            )
        if self.n_states < 2:
            raise RBFEHybridBuildError(f"λ 表至少需要 2 个态（端点 A 与 B）：收到 {self.n_states}")
        for label, values in (("charge", self.charge), ("sterics", self.sterics), ("bonded", self.bonded)):
            if any(not (0.0 <= float(v) <= 1.0) for v in values):
                raise RBFEHybridBuildError(f"λ_{label} 有值不在 [0, 1] 内：{values}")
            if float(values[0]) != 0.0 or float(values[-1]) != 1.0:
                raise RBFEHybridBuildError(
                    f"λ_{label} 的端点必须是 0 和 1（λ=0 对应 A、λ=1 对应 B，计划 §3）："
                    f"收到 {values[0]} … {values[-1]}"
                )
            if any(float(b) < float(a) for a, b in zip(values, values[1:])):
                raise RBFEHybridBuildError(f"λ_{label} 必须单调不减：{values}")

    @property
    def n_states(self) -> int:
        return len(self.charge)

    def state(self, index: int) -> dict:
        return {
            LAMBDA_CHARGE: float(self.charge[index]),
            LAMBDA_STERICS: float(self.sterics[index]),
            LAMBDA_BONDED: float(self.bonded[index]),
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_states": self.n_states,
            LAMBDA_CHARGE: [float(v) for v in self.charge],
            LAMBDA_STERICS: [float(v) for v in self.sterics],
            LAMBDA_BONDED: [float(v) for v in self.bonded],
        }

    @classmethod
    def linear(cls, n_states: int) -> "HybridLambdaSchedule":
        """三条 λ 同步线性推进。最简单、也最容易在端点附近出现采样困难的一条。"""
        if n_states < 2:
            raise RBFEHybridBuildError(f"n_states 至少为 2：收到 {n_states}")
        values = tuple(i / (n_states - 1) for i in range(n_states))
        return cls(name=f"rbfe_linear_{n_states}", charge=values, sterics=values, bonded=values)

    @classmethod
    def charge_then_sterics(cls, n_charge: int, n_sterics: int) -> "HybridLambdaSchedule":
        """先走完电荷，再走 sterics 与 bonded。

        ⚠ 这条路径的中间态**有效总电荷不恒定**（电荷已经变成 B 的，而 B-only 原子
        的 sterics 还没打开）。这不是错误——路径可以自选——但它正是计划 §5.3 点名
        要核对的情形，`hybrid_charge_ledger` 会把逐 λ 的总电荷列出来。
        """
        if n_charge < 2 or n_sterics < 2:
            raise RBFEHybridBuildError("n_charge 与 n_sterics 都至少为 2")
        charge = tuple(i / (n_charge - 1) for i in range(n_charge)) + (1.0,) * (n_sterics - 1)
        rest = (0.0,) * (n_charge - 1) + tuple(i / (n_sterics - 1) for i in range(n_sterics))
        return cls(
            name=f"rbfe_charge{n_charge}_then_sterics{n_sterics}",
            charge=charge, sterics=rest, bonded=rest,
        )


@dataclass(frozen=True)
class HybridSystemBundle:
    """`build_hybrid_system` 的产物：System + 布局 + λ 表 + 可审计的溯源。"""

    system: object
    layout: HybridTopologyLayout
    schedule: HybridLambdaSchedule
    mapping_fingerprint: str
    protocol_version: int
    provenance: dict

    def apply_lambda_state(self, context, index: int) -> dict:
        """把第 index 个 λ 态写进 Context。返回实际写入的值，便于落盘对账。"""
        values = self.schedule.state(index)
        for name, value in values.items():
            context.setParameter(name, value)
        return values

    def fingerprint(self) -> str:
        payload = {
            "hybrid_protocol_version": int(self.protocol_version),
            "mapping_fingerprint": self.mapping_fingerprint,
            "schedule": self.schedule.to_dict(),
            "layout": self.layout.to_dict(),
            "bonded": self.provenance.get("bonded"),
            "nonbonded": self.provenance.get("nonbonded"),
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "hybrid_protocol_version": int(self.protocol_version),
            "fingerprint": self.fingerprint(),
            "mapping_fingerprint": self.mapping_fingerprint,
            "layout": self.layout.to_dict(),
            "schedule": self.schedule.to_dict(),
            "provenance": self.provenance,
        }


def build_hybrid_system(
    system_a,
    ligand_indices_a,
    system_b,
    ligand_indices_b,
    mapping: AtomMapping,
    schedule: HybridLambdaSchedule,
) -> HybridSystemBundle:
    """R1b：由 A、B 两个体系 + 冻结映射构建 hybrid System。

    输入约定：`system_a` 与 `system_b` 是**同一个环境**里分别装了配体 A、B 的
    完整体系（`_assert_environment_matches` 会逐位核对）。`ligand_indices_*` 按
    **分子内顺序**给出，即第 i 个元素就是映射里分子内索引 i 的那个原子。

    产物的原子编号：环境与配体 A 保持 `system_a` 的原索引，B-only 追加在末尾。
    """
    import openmm

    layout = build_hybrid_layout(system_a, ligand_indices_a, system_b, ligand_indices_b, mapping)
    a_map = {i: i for i in range(system_a.getNumParticles())}
    b_map = _b_index_to_hybrid(system_a, ligand_indices_a, system_b, ligand_indices_b, layout)

    hybrid = openmm.System()
    for i in range(system_a.getNumParticles()):
        hybrid.addParticle(system_a.getParticleMass(i))
    hybrid_to_b = {v: k for k, v in b_map.items()}
    for h in layout.b_only:
        hybrid.addParticle(system_b.getParticleMass(hybrid_to_b[h]))
    hybrid.setDefaultPeriodicBoxVectors(*system_a.getDefaultPeriodicBoxVectors())

    n_constraints = _add_constraints(hybrid, system_a, system_b, layout, a_map, b_map)
    bonded_stats = _add_bonded_forces(hybrid, system_a, system_b, layout, a_map, b_map)
    nonbonded_stats = _add_nonbonded_forces(hybrid, system_a, system_b, layout, a_map, b_map)

    # 🔑 [2026-09-09] barostat 也要搬。原来只搬 `CMMotionRemover`，而
    # `_assert_supported_forces` 明确把 `MonteCarloBarostat` 列为**支持的输入**，
    # 于是一个 NPT 输入会静默产出 NVT 杂合 System：
    #   * `verify_hybrid_endpoints` 查不出来（barostat 对势能贡献恒为 0），
    #     所以 R1b 的验收面完全无感；
    #   * `free_energy_engine` 那条路径是 fail-closed 的（"请求了 NPT 但 System
    #     里没有 barostat"），会报错而不是采错系综；
    #   * 但任何直接用 `build_hybrid_system` + 自己的 integrator 的调用方，
    #     会在定容下采样，而 `compute_hybrid_u_kn` 照样加 βpV。
    # 只搬 A 侧的（两侧 barostat 若不一致，_assert_supported_forces 之外还有
    # env_bonded_signature 对账，这里不重复判）。
    for force in system_a.getForces():
        if isinstance(force, openmm.CMMotionRemover):
            hybrid.addForce(openmm.CMMotionRemover(force.getFrequency()))
        elif isinstance(force, openmm.MonteCarloBarostat):
            _copied_barostat = openmm.MonteCarloBarostat(
                force.getDefaultPressure(),
                force.getDefaultTemperature(),
                force.getFrequency(),
            )
            _copied_barostat.setRandomNumberSeed(force.getRandomNumberSeed())
            hybrid.addForce(_copied_barostat)

    provenance = {
        "builder": "rbfe_core.build_hybrid_system",
        "restricted_builder_version": int(RBFE_HYBRID_PROTOCOL_VERSION),
        "lambda_parameter_names": list(RBFE_LAMBDA_NAMES),
        "n_constraints": int(n_constraints),
        "bonded": bonded_stats,
        "nonbonded": nonbonded_stats,
        "force_groups": {
            "default": RBFE_FORCE_GROUP_DEFAULT,
            "dummy_bonded_A": RBFE_FORCE_GROUP_DUMMY_A,
            "dummy_bonded_B": RBFE_FORCE_GROUP_DUMMY_B,
        },
        "known_gaps": [
            "炼金区的 LJ 长程校正（LRC）未计入：三个 custom LJ 力都关了 "
            "setUseLongRangeCorrection，native 力上炼金原子的 epsilon 已清零。"
            "ABFE 侧有现成的 ibs_engine._lj_softcore_tail_radial_integrals 可接，R1b v1 未接。",
            "手性/结合姿势/互变异构仍未检查（需要坐标，见 validate_mapping 的 unchecked）。",
        ],
    }
    return HybridSystemBundle(
        system=hybrid,
        layout=layout,
        schedule=schedule,
        mapping_fingerprint=mapping.fingerprint(),
        protocol_version=RBFE_HYBRID_PROTOCOL_VERSION,
        provenance=provenance,
    )


# -- R1b 的验收：端点等价 / 有限差分力 / 逐 λ 电荷 -------------------------


def _energy_and_forces(system, positions, parameters=None, *, platform_name="Reference", groups=None):
    import openmm
    from openmm import unit

    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    for name, value in (parameters or {}).items():
        context.setParameter(name, value)
    kwargs = {"getEnergy": True, "getForces": True}
    if groups is not None:
        kwargs["groups"] = groups
    state = context.getState(**kwargs)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    del context, integrator
    energy = float(energy)
    forces = np.asarray(forces, dtype=float)

    # 🔑 NaN 必须在这里炸掉。踩过一次：`max(0.0, nan)` 在 Python 里返回 **0.0**
    # （因为 `nan > 0.0` 是 False），于是端点比对里一个 NaN 力偏差被静默当成"零偏差"
    # 通过了验收。fail-closed 的检查器自己漏 NaN，比没有检查器更危险。
    if not math.isfinite(energy) or not np.all(np.isfinite(forces)):
        bad = sorted(set(np.nonzero(~np.isfinite(forces))[0].tolist()))
        raise RBFEHybridBuildError(
            f"能量或力出现非有限值：energy={energy}，非有限力的原子 {bad[:12]}"
            f"{'…' if len(bad) > 12 else ''}。"
            "常见原因是几何退化（例如二面角的四个原子共线，CustomTorsionForce "
            "的解析导数在那里没有定义），或 softcore 分母被写成 0。"
        )
    return energy, forces


def _without_dispersion_correction(system):
    """深拷一份关掉色散长程校正的体系，只用于**受控比对**。

    LJ 的长程校正是一个只依赖盒体积与粒子参数的**常数**（力恒为 0）。炼金原子的
    epsilon 在 hybrid 的 native 力上被清零之后，native 的 LRC 就不再包含配体那部分
    ——于是 hybrid 与纯 A 体系之间会差一个常数。这个差值是**已知缺口**，不是接线
    错误：力逐原子精确相等就是证据。

    所以端点等价性分两步报：关掉 LRC 做**精确**比对（这才测得到 builder 自己写的
    东西），再单独把 LRC 缺口**量化成一个数**报出来，不留一句含糊的 caveat。
    """
    import openmm

    copy = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    copy.thisown = 1
    for force in copy.getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.setUseDispersionCorrection(False)
        elif isinstance(force, openmm.CustomNonbondedForce):
            force.setUseLongRangeCorrection(False)
    return copy


def verify_hybrid_endpoints(
    bundle: HybridSystemBundle,
    system_a,
    ligand_indices_a,
    system_b,
    ligand_indices_b,
    positions_hybrid,
    *,
    platform_name: str = "Reference",
    energy_tolerance_kj_per_mol: float = 1e-4,
    force_tolerance_kj_per_mol_nm: float = 1e-3,
) -> dict:
    """计划 §5.3：「验证 λ 端点的物理相互作用恢复」。

    ## 判据是精确的，不是"看着像"

    λ=0 时 hybrid 与纯 A 体系相比，**多出来的恰好是 B 侧 dummy 的成键项**——
    B-only 原子此时电荷为 0、sterics 前因子为 0、且与 A-only 之间已被排除，
    所以它对体系的唯一贡献就是自己那几个全强度成键项。那些项被单独放在力组
    `RBFE_FORCE_GROUP_DUMMY_B` 里，于是判据可以写成等式：

        E_hybrid(λ=0) − E[力组 dummy_B] == E_A
        E_hybrid(λ=1) − E[力组 dummy_A] == E_B

    力也一样，逐原子比。这就是计划 §5.3 说的「明确可分离因子」——这里它不仅被
    明确了，还被独立成一个可以直接读数的力组。

    ## 为什么比对时把 LJ 长程校正关掉

    炼金原子的 epsilon 在 hybrid 的 native 力上被清零，native 的色散长程校正因此
    不再包含配体那部分——hybrid 与纯 A 体系之间会差一个**常数**（LRC 只依赖盒体积
    与粒子参数，力恒为 0）。那是**已知缺口**，不是接线错误，所以：等式判据在关掉
    LRC 的副本上做（这才测得到 builder 自己写的东西），缺口本身单独量化成
    `lj_lrc_gap_kJ_per_mol` 报出来。
    """
    from openmm import unit

    layout = bundle.layout
    a_map = {i: i for i in range(system_a.getNumParticles())}
    b_map = _b_index_to_hybrid(system_a, ligand_indices_a, system_b, ligand_indices_b, layout)

    positions = np.asarray(
        [[v.x, v.y, v.z] if hasattr(v, "x") else list(v) for v in positions_hybrid], dtype=float
    )
    if positions.shape[0] != layout.n_particles:
        raise RBFEHybridBuildError(
            f"positions_hybrid 有 {positions.shape[0]} 个原子，hybrid 体系有 "
            f"{layout.n_particles} 个"
        )
    positions_q = positions * unit.nanometer

    positions_a = positions[: system_a.getNumParticles()] * unit.nanometer
    order_b = [b_map[i] for i in range(system_b.getNumParticles())]
    positions_b = positions[order_b] * unit.nanometer

    hybrid_no_lrc = _without_dispersion_correction(bundle.system)

    report = {"endpoints": {}, "passed": True}
    for label, lam, other_group, reference_system, reference_positions, index_map in (
        ("lambda=0 (A)", 0.0, RBFE_FORCE_GROUP_DUMMY_B, system_a, positions_a, a_map),
        ("lambda=1 (B)", 1.0, RBFE_FORCE_GROUP_DUMMY_A, system_b, positions_b, b_map),
    ):
        parameters = {name: lam for name in RBFE_LAMBDA_NAMES}
        reference_no_lrc = _without_dispersion_correction(reference_system)

        total_e, total_f = _energy_and_forces(
            hybrid_no_lrc, positions_q, parameters, platform_name=platform_name
        )
        dummy_e, dummy_f = _energy_and_forces(
            hybrid_no_lrc, positions_q, parameters,
            platform_name=platform_name, groups={other_group},
        )
        ref_e, ref_f = _energy_and_forces(
            reference_no_lrc, reference_positions, platform_name=platform_name
        )
        separable_e = total_e - dummy_e

        # LRC 缺口：两边"开 LRC 减关 LRC"的差，就是 hybrid 少掉的那部分炼金 LJ 尾项。
        hybrid_with_lrc, _ = _energy_and_forces(
            bundle.system, positions_q, parameters, platform_name=platform_name
        )
        reference_with_lrc, _ = _energy_and_forces(
            reference_system, reference_positions, platform_name=platform_name
        )
        lrc_gap = (hybrid_with_lrc - total_e) - (reference_with_lrc - ref_e)

        force_deviation = 0.0
        for source_index in range(reference_system.getNumParticles()):
            hybrid_index = index_map[source_index]
            delta = (total_f[hybrid_index] - dummy_f[hybrid_index]) - ref_f[source_index]
            force_deviation = max(force_deviation, float(np.max(np.abs(delta))))

        energy_deviation = abs(separable_e - ref_e)
        ok = (
            energy_deviation <= energy_tolerance_kj_per_mol
            and force_deviation <= force_tolerance_kj_per_mol_nm
        )
        report["endpoints"][label] = {
            "hybrid_total_kJ_per_mol": total_e,
            "separable_dummy_kJ_per_mol": dummy_e,
            "hybrid_minus_dummy_kJ_per_mol": separable_e,
            "reference_kJ_per_mol": ref_e,
            "energy_deviation_kJ_per_mol": energy_deviation,
            "max_force_deviation_kJ_per_mol_nm": force_deviation,
            "lj_lrc_gap_kJ_per_mol": float(lrc_gap),
            "passed": bool(ok),
        }
        report["passed"] = report["passed"] and ok
    return report


def verify_hybrid_forces_finite_difference(
    bundle: HybridSystemBundle,
    positions_hybrid,
    lambda_state_index: int,
    *,
    platform_name: str = "Reference",
    delta_nm: float = 2.0e-5,
    n_samples: int = 12,
    seed: int = 20260903,
    relative_tolerance: float = 5.0e-4,
    absolute_tolerance_kj_per_mol_nm: float = 1.0e-2,
) -> dict:
    """计划 §8 的 R1 验收：**有限差分力与解析力一致**。

    对随机抽取的 (原子, 分量) 做中心差分 −dU/dx，与解析力逐个比。抽样是**确定性**
    的（固定 seed），同一个体系跑两次抽到同一批点，否则"通过"不可复现。

    这条测的是 builder 写的表达式与它自己声称的势是否自洽——尤其是三个 custom
    LJ 力那几个手写的 softcore 分母，以及成键插值项。
    """
    from openmm import unit

    positions = np.asarray(
        [[v.x, v.y, v.z] if hasattr(v, "x") else list(v) for v in positions_hybrid], dtype=float
    )
    parameters = bundle.schedule.state(lambda_state_index)
    _, analytic = _energy_and_forces(
        bundle.system, positions * unit.nanometer, parameters, platform_name=platform_name
    )

    rng = np.random.default_rng(seed)
    n_particles = positions.shape[0]
    n_samples = int(n_samples)

    # 🔑 抽样必须**结构上**覆盖炼金区，不能靠运气。原来是在全体粒子上均匀抽：
    # 真实溶剂化体系里炼金原子只有几十个、总粒子七万个，12 个点抽中一个的概率
    # ≈1%——而本函数 docstring 声称测的正是那几个手写的 softcore 分母。
    # 小体系的单元测试（fixture 几乎全是炼金原子）永远看不出这一点。
    # 现在前一半的点在 dummy 与 core 之间交替抽（两类手写力各自覆盖），
    # 后一半仍抽全体（环境侧的 native 力也要测）。
    dummies = sorted(set(bundle.layout.a_only) | set(bundle.layout.b_only))
    core = sorted(set(bundle.layout.core))
    alchemical = sorted(set(bundle.layout.alchemical))
    n_alch = min(n_samples // 2, len(alchemical))
    picks = []
    for i in range(n_alch):
        pool = dummies if (dummies and (i % 2 == 0 or not core)) else core
        pool = pool or alchemical
        picks.append((int(pool[int(rng.integers(0, len(pool)))]), int(rng.integers(0, 3))))
    picks += [
        (int(rng.integers(0, n_particles)), int(rng.integers(0, 3)))
        for _ in range(n_samples - n_alch)
    ]

    rows = []
    worst = 0.0
    for atom, axis in picks:
        shifted_plus = positions.copy()
        shifted_minus = positions.copy()
        shifted_plus[atom, axis] += delta_nm
        shifted_minus[atom, axis] -= delta_nm
        e_plus, _ = _energy_and_forces(
            bundle.system, shifted_plus * unit.nanometer, parameters, platform_name=platform_name
        )
        e_minus, _ = _energy_and_forces(
            bundle.system, shifted_minus * unit.nanometer, parameters, platform_name=platform_name
        )
        numeric = -(e_plus - e_minus) / (2.0 * delta_nm)
        exact = float(analytic[atom, axis])
        deviation = abs(numeric - exact)
        scale = max(abs(exact), abs(numeric), 1.0)
        relative = deviation / scale
        worst = max(worst, relative if deviation > absolute_tolerance_kj_per_mol_nm else 0.0)
        rows.append({
            "atom": atom, "axis": axis,
            "analytic_kJ_per_mol_nm": exact,
            "finite_difference_kJ_per_mol_nm": numeric,
            "abs_deviation": deviation,
            "relative_deviation": relative,
            "passed": bool(
                deviation <= absolute_tolerance_kj_per_mol_nm
                or relative <= relative_tolerance
            ),
        })

    return {
        "lambda_state_index": int(lambda_state_index),
        "lambda_values": parameters,
        "delta_nm": float(delta_nm),
        "n_samples": len(rows),
        #: 其中有多少个点落在炼金区（core/dummy）——"测了多少"要能被读出来。
        "n_alchemical_samples": int(n_alch),
        "worst_relative_deviation": float(worst),
        "samples": rows,
        "passed": all(row["passed"] for row in rows),
    }


def hybrid_charge_ledger(bundle: HybridSystemBundle, *, tolerance_e: float = 1e-9) -> dict:
    """计划 §5.3：「**核对每个 λ 的有效总电荷**。即使 A、B 净电荷相同，也不能因
    不同电荷分段切换而忽略中间态的电荷变化。」

    直接读**建好的** NonbondedForce 的粒子电荷与 offset，按 λ 表逐态求和——
    不是把 builder 的意图重算一遍，而是核对它真正写进去的东西。
    """
    import openmm
    from openmm import unit

    nb = _nb_force_of(bundle.system, "hybrid 体系")
    alchemical = set(bundle.layout.alchemical)

    base = {}
    for index in sorted(alchemical):
        charge, _sigma, _epsilon = nb.getParticleParameters(index)
        base[index] = _quantity_value(charge)

    offsets = {}
    for i in range(nb.getNumParticleParameterOffsets()):
        name, index, charge_scale, _sigma_scale, _eps_scale = nb.getParticleParameterOffset(i)
        if name != LAMBDA_CHARGE:
            continue
        offsets[int(index)] = offsets.get(int(index), 0.0) + _quantity_value(charge_scale)

    stray = sorted(set(offsets) - alchemical)
    if stray:
        raise RBFEHybridBuildError(
            f"非炼金原子 {stray} 上挂着 {LAMBDA_CHARGE} 的电荷 offset——环境电荷不该随 λ 变"
        )

    rows = []
    for index in range(bundle.schedule.n_states):
        lam = bundle.schedule.state(index)[LAMBDA_CHARGE]
        total = sum(base[i] + lam * offsets.get(i, 0.0) for i in sorted(alchemical))
        rows.append({"state": index, LAMBDA_CHARGE: lam, "alchemical_net_charge_e": total})

    endpoints = (rows[0]["alchemical_net_charge_e"], rows[-1]["alchemical_net_charge_e"])
    max_excursion = max(
        abs(row["alchemical_net_charge_e"] - endpoints[0]) for row in rows
    )
    return {
        "per_state": rows,
        "net_charge_A_e": endpoints[0],
        "net_charge_B_e": endpoints[1],
        "endpoints_match": bool(abs(endpoints[0] - endpoints[1]) <= tolerance_e),
        "max_intermediate_excursion_e": float(max_excursion),
        "constant_across_lambda": bool(max_excursion <= tolerance_e),
    }


