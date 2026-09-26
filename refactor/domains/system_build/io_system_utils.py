

# ============================================================================
# 7. 单位格式化器 (I/O 层专用：内核计算绝不碰单位转换)
# ============================================================================
class UnitFormatter:
    """
    单位格式化器：内部计算不碰，仅用于 I/O 转换
    【三层单位规范】
    1. 内核层 (OpenMM): nm, kJ/mol, ps, rad, e (裸数值)
    2. 数据交换层 (JSON): key 带单位后缀，如 r0_nm, kr_kJ_mol_nm2
    3. 人类可读层 (LOG/Report): Å, kcal/mol, ns, °
    """
    # === 基础转换函数 ===
    @staticmethod
    def nm_to_A(val): return val * 10.0
    @staticmethod
    def A_to_nm(val): return val * 0.1
    @staticmethod
    def kJ_to_kcal(val): return val / 4.184
    @staticmethod
    def kcal_to_kJ(val): return val * 4.184
    @staticmethod
    def rad_to_deg(val): return np.degrees(val)
    @staticmethod
    def deg_to_rad(val): return np.radians(val)
    @staticmethod
    def ps_to_ns(val): return val / 1000.0
    @staticmethod
    def ns_to_ps(val): return val * 1000.0

    # === Boresch 参数格式化 (人类可读) ===
    @classmethod
    def format_boresch_human(cls, boresch_dict: dict) -> str:
        """格式化 Boresch 参数为化学家常用单位"""
        eq = boresch_dict["equilibrium_values"]
        fc = boresch_dict["force_constants"]
        return (
            f"r0={cls.nm_to_A(eq['r0']):.2f} Å | "
            f"θA={cls.rad_to_deg(eq['thetaA0']):.1f}° | "
            f"φA={cls.rad_to_deg(eq['phiA0']):.1f}°\n"
            f"kr={cls.kJ_to_kcal(fc['kr']) / 100.0:.2f} kcal/mol/Å²| "
            f"kθA={cls.kJ_to_kcal(fc['kthetaA']):.2f} kcal/mol/rad²"
        )

    # 本函数**自己输出**的带后缀键名 → 裸键名。双向接受是为了让
    # `format_boresch_json` 幂等，见下面 docstring 里的 2026-08-31 说明。
    # 只收本函数自己会写出的这套后缀；`_deg` 之类需要换算的后缀一律不认，
    # 免得把角度当弧度读进来（那会让释放项错约 57 倍）。
    _BORESCH_EQ_KEY_ALIASES = {
        "r0": "r0_nm",
        "thetaA0": "thetaA0_rad",
        "thetaB0": "thetaB0_rad",
        "phiA0": "phiA0_rad",
        "phiB0": "phiB0_rad",
        "phiC0": "phiC0_rad",
    }
    _BORESCH_FC_KEY_ALIASES = {
        "kr": "kr_kJ_mol_nm2",
        "kthetaA": "kthetaA_kJ_mol_rad2",
        "kthetaB": "kthetaB_kJ_mol_rad2",
        "kphiA": "kphiA_kJ_mol_rad2",
        "kphiB": "kphiB_kJ_mol_rad2",
        "kphiC": "kphiC_kJ_mol_rad2",
    }

    @staticmethod
    def _boresch_scalar(mapping: dict, bare: str, suffixed: str, kind: str) -> float:
        """按"裸键优先、带后缀键兜底"取一个 Boresch 标量，缺失即 fail-closed。

        旧实现是 `mapping.get(bare, 0)`：缺键静默变 0。配合
        `apply_boresch_correction` 的"格式化后回写同一路径"，这会把已经带后缀
        的参数（autoload 回来的那份）读成全 0 再写回磁盘。见 docstring。
        """
        for key in (bare, suffixed):
            if key in mapping and mapping[key] is not None:
                return float(mapping[key])
        raise ValueError(
            f"Boresch {kind} 缺少 `{bare}`（带单位后缀的 `{suffixed}` 也没有）。"
            "拒绝按 0 兜底：0 会通过 `_has_valid_boresch_restraint` 的 3+3 锚点"
            "检查、却让限制力变成空操作或让解析释放项报错。"
        )

    # === Boresch 参数序列化 (JSON，key 带单位后缀) ===
    @classmethod
    def format_boresch_json(cls, boresch_dict: dict) -> dict:
        """
        序列化 Boresch 参数为 JSON 安全格式 (兼容扁平/嵌套/混合结构)
        🔑 智能路由提取逻辑，彻底解决 KeyError: 'equilibrium_values'

        🔑 [2026-08-31 P1] **本函数必须幂等**。它输出的键带单位后缀
        （`r0_nm`/`kr_kJ_mol_nm2`…），而旧实现只用裸键 `eq.get("r0", 0)` 读取，
        于是 `format_boresch_json(format_boresch_json(x))` 把 6 个平衡值与
        6 个力常数全部变成 0.0——锚点索引却保留，文件结构看起来仍然合法。
        唯一生产调用方 `abfe_pipeline.apply_boresch_correction` 正是"格式化后
        写回同一个 boresch_params.json"，所以只要 complex 腿走 autoload
        （`boresch_params=None` 且盘上已有该文件，默认允许），生产 Boresch
        工件就会被静默清零；之后 `_boresch_analytic_release` 的
        `50 <= kr <= 5000` 断言才会崩，而文件已经毁了。
        修法是双向接受裸键与本函数自己的后缀键，并且缺键 fail-closed。
        """
        # 1. 智能提取：优先从嵌套层取，若为空则降级到顶层
        anchors = boresch_dict.get("boresch_anchors", boresch_dict)
        eq = anchors.get("equilibrium_values") or boresch_dict.get("equilibrium_values", {})
        fc = anchors.get("force_constants") or boresch_dict.get("force_constants", {})
        rec_idx = anchors.get("receptor_indices") or boresch_dict.get("receptor_indices", [])
        lig_idx = anchors.get("ligand_indices") or boresch_dict.get("ligand_indices", [])

        if not eq or not fc:
            raise ValueError("Boresch 参数字典结构异常：缺失 equilibrium_values 或 force_constants")

        # 2. 构建标准嵌套输出 (严格带单位后缀)
        return {
            "boresch_anchors": {
                # Anchor indices may originate from OpenMM/NumPy arrays.  Cast
                # them here so the formatter's documented JSON output is
                # directly serializable even before the final write boundary.
                "receptor_indices": [int(i) for i in rec_idx],
                "ligand_indices": [int(i) for i in lig_idx],
                "equilibrium_values": {
                    suffixed: cls._boresch_scalar(eq, bare, suffixed, "equilibrium_values")
                    for bare, suffixed in cls._BORESCH_EQ_KEY_ALIASES.items()
                },
                "force_constants": {
                    suffixed: cls._boresch_scalar(fc, bare, suffixed, "force_constants")
                    for bare, suffixed in cls._BORESCH_FC_KEY_ALIASES.items()
                },
            },
            "is_fallback": boresch_dict.get("is_fallback", False),
            "total_score": boresch_dict.get("total_score", None),
            "diagnostics": boresch_dict.get("diagnostics", None),
        }

    # === 结果格式化 (人类可读) ===
    @classmethod
    def format_results_human(cls, results: dict) -> str:
        """格式化最终结果报告"""
        err_kj = results.get("total_error_kJ_mol", results.get("total_error", 0.0))
        # [P1-19] `total_delta_G_complex_kJ_mol` 这个键名是历史遗留——
        # abfe_pipeline.compute_final_results 对 complex 腿和 solvent 腿都用
        # 同一个键存"这条腿自己的总 ΔG"，不分腿。这里原来只看键名存不存在，
        # 于是 solvent 腿也被打上"复合物总自由能"的标题。现在 results 里若带
        # `system_type`（2026-08-25 起新写入 final_results.json 的字段）就按
        # 它判腿别；旧的、没有这个字段的产物保持原样（默认当 complex 处理，
        # 不改变既有行为）。
        # 只有两个真实取值（"complex"/"solvent"，见
        # abfe_pipeline.run_full_pipeline 的 system_type 形参默认值与
        # run_full_abfe_loop 里对两条腿的显式赋值）；空/缺失按旧行为当
        # complex 处理，不用枚举去匹配未来可能出现的新腿名。
        _system_type = str(results.get("system_type") or "complex").lower()
        if "delta_G_bind_kJ_mol" in results:
            dg_kj = results.get("delta_G_bind_kJ_mol", 0.0)
            title = "[OK] 结合自由能 ΔG_bind"
        elif "total_delta_G_complex_kJ_mol" in results and _system_type == "complex":
            dg_kj = results.get("total_delta_G_complex_kJ_mol", 0.0)
            title = "[OK] 复合物总自由能 ΔG_complex"
        elif "total_delta_G_complex_kJ_mol" in results:
            dg_kj = results.get("total_delta_G_complex_kJ_mol", 0.0)
            title = f"[OK] 解耦腿自由能 ΔG_leg（system_type={results.get('system_type')}）"
        elif "decoupling_delta_G_kJ_mol" in results:
            dg_kj = results.get("decoupling_delta_G_kJ_mol", 0.0)
            title = "[OK] 解耦腿自由能 ΔG_leg"
        else:
            dg_kj = results.get("total_delta_G_complex_kJ_mol", results.get("total_delta_G_complex", 0.0))
            title = "[OK] 自由能结果 ΔG"
        return (
            f"\n{'='*50}\n"
            f"{title} = {cls.kJ_to_kcal(dg_kj):.2f} ± {cls.kJ_to_kcal(err_kj):.2f} kcal/mol\n"
            f"   ( = {dg_kj:.2f} ± {err_kj:.2f} kJ/mol )\n"
            f"{'='*50}"
        )

    # === 采样元数据格式化 (JSON) ===
    @classmethod
    def format_sampling_metadata_json(cls, config: dict) -> dict:
        """序列化采样元数据为 JSON"""
        return {
            "sampling_metadata": {
                "dt_ps": config.get("timestep_ps", 0.002),
                "temperature_K": config.get("temperature", 300.0),
                "n_steps_per_window": config.get("n_steps_per_window", 0),
                "friction_ps": config.get("friction", 1.0),
            }
        }

# ============================================================================
# 通用工具函数：System 管理与配体内部力构建
# ============================================================================
from openmm import XmlSerializer

def ensure_owned_system(system: openmm.System) -> openmm.System:
    """强制获取 System 的 Python 所有权，防止 SWIG GC"""
    if system is None:
        raise ValueError("System 对象为 None")
    if getattr(system, 'thisown', 0) == 1:
        try:
            _ = system.getNumParticles()
        except Exception as exc:
            raise RuntimeError(
                "System 声称由 Python 持有，但底层 OpenMM 对象已不可访问。"
            ) from exc
        return system
    new_sys = XmlSerializer.clone(system)
    new_sys.thisown = 1
    _ = new_sys.getNumParticles()
    return new_sys


def sync_all_exclusions(system: openmm.System) -> int:
    """
    生产级排除表同步。

    🚨 关键修复：OpenMM 要求同一 System 里所有共享同一套邻居表的
    NonbondedForce/CustomNonbondedForce（粒子数相同）拥有完全相同的排除表——
    "All Forces must have identical exclusions" 就是这个要求被违反时抛出的。
    旧版本按每个 CustomNonbondedForce 的 interaction group 范围"按需"补齐排除表
    （例如只给 L-E 力补 L-E 相关的对），这在物理上没问题（interaction group
    之外的对本来就不会被计算），但 OpenMM 底层邻居表校验比较的是排除表本身
    是否逐对相同，不管 interaction group——所以只要 NonbondedForce 里有任何
    一个不落在某个 CustomNonbondedForce interaction group 内的排除对
    （典型情况：环境蛋白/水分子自身的 1-2/1-3/1-4 排除，跟只处理 L-E 的软核力
    毫不相关），旧逻辑就会让两者的排除表数量对不上，从而在生产采样阶段
    （通常是第一次真正调用 minimizeEnergy/getState 触发底层邻居表构建时）报错。
    这里改为无差别地把"并集"灌给每一个粒子数匹配的力，牺牲一点点冗余排除对，
    换来严格逐对相同——interaction group 之外的排除对本来就不会被该力用到，
    是纯粹的账本对齐，不改变任何物理量。
    """
    nb_forces = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    custom_forces = [f for f in system.getForces() if isinstance(f, openmm.CustomNonbondedForce)]
    if not nb_forces or not custom_forces:
        return 0
    nb_force = nb_forces[0]
    n_particles = nb_force.getNumParticles()

    union_excl = set()
    for i in range(nb_force.getNumExceptions()):
        p1, p2, _, _, _ = nb_force.getExceptionParameters(i)
        p1, p2 = int(p1), int(p2)
        if p1 != p2:
            union_excl.add((min(p1, p2), max(p1, p2)))

    eligible_forces = []
    existing_per_force = []
    for c_force in custom_forces:
        if c_force.getNumParticles() != n_particles:
            continue
        existing = set()
        for i in range(c_force.getNumExclusions()):
            p1, p2 = c_force.getExclusionParticles(i)
            existing.add((min(int(p1), int(p2)), max(int(p1), int(p2))))
        union_excl |= existing
        eligible_forces.append(c_force)
        existing_per_force.append(existing)

    total_synced = 0
    for c_force, existing in zip(eligible_forces, existing_per_force):
        missing = union_excl - existing
        # 🔑 [EXP-031 路线 A] `sorted()` 不是整洁癖，是这一行本身值 1.162×。
        #
        # `missing` 是 set，迭代序由哈希决定，基本是随机的。OpenMM 的
        # CustomNonbondedForce 对**排除表的写入顺序**敏感：乱序灌进去，它内部
        # 按粒子组织的排除结构与邻居表就退化，每步都要多付一笔。
        #
        # 而且**随机打乱比 set 的哈希序还更慢** —— 所以要紧的是"升序"本身，
        # 不只是"别用哈希序"；换成任何一种确定性顺序都不行。
        #
        # 受害的是那些**自己只带局部排除表**的力 —— 典型是 Group-2「恢复配体
        # 内部非键」：interaction group 只有 41×41，却要被灌进全系统 120895 条
        # 排除（本函数的全部意义就是"账本对齐"，见上面的 docstring）。
        #
        # ⚠️ IBS 的软核 CV 确实免疫，但**原因是结构性的，不是"missing 为空集"**：
        # 它们嵌在 Group-1 的 CustomCVForce 内部（`ibs_engine` 加进 System 的是
        # `ibs_wrapper.get_force()`，返回类型就是 `CustomCVForce`），而本函数只挑
        # `system.getForces()` 里的 `CustomNonbondedForce` —— 嵌套的 CV 根本不在
        # 候选集里，对它们 `missing` **从未被计算过**。同样的话 `ibs_engine.py`
        # 里那段（"软核 CV…sync_all_exclusions 扫不到"）早就写对了。
        #
        # 不要写成"`full_softcore_excl` 本来就是全系统并集所以 missing 为空"。
        # 那个说法会害两次：① 谁把某个 CV 提到顶层、或新增一个粒子数匹配的顶层
        # CustomNonbondedForce，会以为"反正是全集所以安全"，实际它立刻开始收到
        # 这条追加尾巴；② 它暗示"别动 full_softcore_excl 的全集性质，否则排序税
        # 回来" —— 也是错的，动了它坏的是**正确性**（`current_excl != template_excl`
        # 那个 raise 挡的东西），跟排序税无关。
        #
        # 为什么修在这里而不是各产地：各产地都已经 sorted() 过了，但那保护不了
        # 本函数在后面**追加**的这条尾巴。
        #
        # 实测（EXP-031，abfe-ibs-cuda-d5）：真 Atenolol 膜体系 45354 原子、
        # 三次独立测量 **1.162×**。⚠️ 早期文档里的 **1.45× 是水盒工作点的数字，
        # 不可引用**（见 docs/archive/EXP-031_GPU_OPTIMIZATION_2026-09-09.md）。
        # 排除对的集合与条数完全不变，能量逐比特不变；变的只有写入顺序 ——
        # 但那会改 System XML 里排除表的书写顺序 ⟹ `system_xml_sha256` 变，
        # 既有窗口产物/resume 失配。"能量不变"≠"缓存能用"。
        for p1, p2 in sorted(missing):
            c_force.addExclusion(p1, p2)
        total_synced += len(missing)
    return total_synced


_NONBONDED_FORCE_SCALAR_PROPERTIES = (
    # (getter, setter) —— 与电荷/λ 无关的 NonbondedForce 配置，烘焙时必须
    # 原样保留，否则烘焙本身就悄悄改了 Hamiltonian（2026-08-11 用户审阅指出）。
    ("getName", "setName"),
    ("getForceGroup", "setForceGroup"),
    ("getReciprocalSpaceForceGroup", "setReciprocalSpaceForceGroup"),
    ("getNonbondedMethod", "setNonbondedMethod"),
    ("getCutoffDistance", "setCutoffDistance"),
    ("getUseSwitchingFunction", "setUseSwitchingFunction"),
    ("getSwitchingDistance", "setSwitchingDistance"),
    ("getUseDispersionCorrection", "setUseDispersionCorrection"),
    ("getReactionFieldDielectric", "setReactionFieldDielectric"),
    ("getEwaldErrorTolerance", "setEwaldErrorTolerance"),
    ("getExceptionsUsePeriodicBoundaryConditions", "setExceptionsUsePeriodicBoundaryConditions"),
    ("getIncludeDirectSpace", "setIncludeDirectSpace"),
)
_NONBONDED_FORCE_TUPLE_PROPERTIES = (
    # 这两个 getter 返回一个元组（alpha, nx, ny, nz），要展开传给对应 setter。
    ("getPMEParameters", "setPMEParameters"),
    ("getLJPMEParameters", "setLJPMEParameters"),
)


def _copy_nonbonded_force_settings(
    src: openmm.NonbondedForce, dst: openmm.NonbondedForce
) -> List[str]:
    """把 `src` 上与 particle/exception 参数无关的所有配置复制到 `dst`。

    用 `getattr` 逐个探测而不是硬编码假设全部存在——不同 OpenMM 版本这组
    属性不完全一样；缺哪个就跳过并如实报告，不假装复制了。返回跳过的
    属性名列表（供调用方决定要不要打日志/报错）。
    """
    skipped: List[str] = []
    for getter_name, setter_name in _NONBONDED_FORCE_SCALAR_PROPERTIES:
        getter = getattr(src, getter_name, None)
        setter = getattr(dst, setter_name, None)
        if getter is None or setter is None:
            skipped.append(getter_name)
            continue
        setter(getter())
    for getter_name, setter_name in _NONBONDED_FORCE_TUPLE_PROPERTIES:
        getter = getattr(src, getter_name, None)
        setter = getattr(dst, setter_name, None)
        if getter is None or setter is None:
            skipped.append(getter_name)
            continue
        setter(*getter())
    if skipped:
        print(
            "  [WARN] [bake_global_parameter_into_fixed_nonbonded_force] 当前 OpenMM "
            f"版本缺少以下 NonbondedForce 属性，未复制（版本差异，不是烘焙逻辑本身"
            f"的缺陷）：{skipped}"
        )
    return skipped


def _scan_forces_referencing_global_parameter(
    system: openmm.System, parameter_name: str
) -> List[Tuple[int, Any]]:
    """返回 System 里所有把 `parameter_name` 声明为自己 GlobalParameter 的
    `(force_index, force)`——不只看 NonbondedForce，任何 Custom*Force 都可能
    引用同名参数。
    """
    hits: List[Tuple[int, Any]] = []
    for idx in range(system.getNumForces()):
        force = system.getForce(idx)
        get_num = getattr(force, "getNumGlobalParameters", None)
        get_name = getattr(force, "getGlobalParameterName", None)
        if get_num is None or get_name is None:
            continue
        for p in range(get_num()):
            if get_name(p) == parameter_name:
                hits.append((idx, force))
                break
    return hits


def _bake_global_parameter_into_custom_bond_force(
    force: "openmm.CustomBondForce",
    parameter_name: str,
    lambda_value: float,
) -> "openmm.CustomBondForce":
    """把 `parameter_name` 的取值代进 CustomBondForce 的能量表达式，并重建一个
    不再声明该 GlobalParameter 的力（OpenMM 没有 removeGlobalParameter）。

    只做**整词**替换，替换后再断言表达式里确实不含该标识符 —— 宁可报错也不留一个
    看起来烘焙过、实际还带着活参数的力。
    """
    import re

    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", str(parameter_name)):
        raise ValueError(
            f"{parameter_name!r} 不是合法标识符，拒绝在能量表达式里做整词替换。"
        )
    expr = force.getEnergyFunction()
    pattern = re.compile(rf"\b{re.escape(str(parameter_name))}\b")
    if not pattern.search(expr):
        raise RuntimeError(
            f"{parameter_name!r} 被登记为该 CustomBondForce 的 GlobalParameter，"
            f"但没出现在能量表达式里：{expr!r}。语义不明，拒绝烘焙。"
        )
    baked_expr = pattern.sub(f"({float(lambda_value):.17g})", expr)
    if pattern.search(baked_expr):
        raise RuntimeError(f"整词替换后表达式仍含 {parameter_name!r}：{baked_expr!r}")

    new_force = openmm.CustomBondForce(baked_expr)
    new_force.setName(force.getName())
    new_force.setForceGroup(force.getForceGroup())
    new_force.setUsesPeriodicBoundaryConditions(
        force.usesPeriodicBoundaryConditions()
    )
    for i in range(force.getNumPerBondParameters()):
        new_force.addPerBondParameter(force.getPerBondParameterName(i))
    for i in range(force.getNumGlobalParameters()):
        name = force.getGlobalParameterName(i)
        if str(name) == str(parameter_name):
            continue
        new_force.addGlobalParameter(name, force.getGlobalParameterDefaultValue(i))
    for i in range(force.getNumBonds()):
        p1, p2, params = force.getBondParameters(i)
        new_force.addBond(int(p1), int(p2), params)
    if new_force.getNumBonds() != force.getNumBonds():
        raise RuntimeError("烘焙 CustomBondForce 时键数对不上")
    return new_force


def bake_global_parameter_into_fixed_nonbonded_force(
    system: openmm.System,
    parameter_name: str,
    lambda_value: float,
) -> openmm.System:
    """把某个 GlobalParameter 在给定端点上的取值烘焙成 `NonbondedForce` 的
    静态参数，并把这个 GlobalParameter 从 System 里彻底删除——不是靠"调用方
    记得把它设成这个值"这种纪律来保证安全，是让"忘了设"这条路径在结构上
    就不存在。

    背景（`docs/experiments/STAGE2_CHARGE_TRANSFER_HANDOFF_PROPOSAL.md`）：C3-1 会话诊断
    发现的真实 bug——`ibs_engine.build_ibs_dual_system` 会把 charging 配置
    完成后 System 上的 `lam_coul`（默认值 **1.0**）连同它的
    ParticleParameterOffset/ExceptionParameterOffset 原样克隆过去；后续
    求值只要忘了显式 `context.setParameter("lam_coul", 0.0)`，OpenMM 就用
    默认值 1.0，把"配体 0 电荷、co-ion 满电"的 λ=0 端点悄悄翻成"配体满电、
    co-ion 中性"。这个函数就是那条缺失的、结构性的"安全交接"步骤。

    契约（2026-08-11 用户审阅后钉死）：

    1. 只删除 `parameter_name` 这一个 GlobalParameter；`NonbondedForce` 上
       其它 GlobalParameter、挂在其它参数名下的 offset 原样保留。
    2. 同一个粒子/exception 上如果有**多个**挂在 `parameter_name` 上的
       offset，fail closed——**不是**"先把 scale 累加再烘焙一次"。这一条
       在审阅时最初写反了：`parameter = base + Σ(global_i × scale_i)` 里的
       Σ 说的是"同一个粒子上挂了多个不同 GlobalParameter 各自的 offset 要
       相加"，不是"同一个 (parameter, particle) 重复挂多条也相加"。用真实
       Context 实测过（2026-08-11）：对同一个 (parameter, particle) 追加两条
       offset，`getNumParticleParameterOffsets()` 确实报出两条，但 Context
       求值时只认**最后一条**，不是两条的和。这是没有文档、容易被误用的
       OpenMM 行为；真实生产代码从不对同一个粒子重复调用
       `addParticleParameterOffset`，所以这里选择不去复现"取最后一条"这个
       隐藏规则，遇到真的重复就直接报错，不猜语义。
    3. 新建的 `NonbondedForce` 完整复制原 force 的非 particle/exception 配置
       （见 `_copy_nonbonded_force_settings`），当前 OpenMM 版本缺哪个属性
       就跳过并打印警告，不假装复制了。
    4. 若 `parameter_name` 还被 System 里其它（非目标）Force 引用，fail
       closed——本函数只烘焙 `NonbondedForce`，不能宣称整个 System 已经不再
       有这个活参数。
    5. charge、sigma、epsilon 三个分量分别用同一个通用公式
       `base + lambda_value * scale` 烘焙，不借用 `charge_at_lambda`
       （那是电荷专用命名，这里是通用工具，同一个公式对三个分量都适用）。
    6. `lambda_value` 必须是精确的 `0.0` 或 `1.0`——C3 的容差体系只在端点上
       有意义，非端点值拒绝。
    """
    lambda_value = float(lambda_value)
    if lambda_value not in (0.0, 1.0):
        raise ValueError(
            f"lambda_value={lambda_value} 不是精确的 0.0 或 1.0——烘焙只定义在端点上。"
        )

    system = ensure_owned_system(
        XmlSerializer.clone(system)
    )

    hits = _scan_forces_referencing_global_parameter(system, parameter_name)
    nb_hits = [(idx, f) for idx, f in hits if isinstance(f, openmm.NonbondedForce)]
    other_hits = [(idx, f) for idx, f in hits if not isinstance(f, openmm.NonbondedForce)]
    if not nb_hits:
        raise RuntimeError(
            f"{parameter_name!r} 不是这个 System 里任何 NonbondedForce 的 "
            "GlobalParameter，无法烘焙（可能是名字拼错了，或者这个参数根本不在"
            "这个 System 上）。"
        )
    if len(nb_hits) > 1:
        raise RuntimeError(
            f"{parameter_name!r} 出现在 {len(nb_hits)} 个 NonbondedForce 上——"
            "当前实现只支持单个 NonbondedForce，多个的语义未定义，拒绝继续。"
        )
    unbakeable = [
        (idx, f) for idx, f in other_hits if not isinstance(f, openmm.CustomBondForce)
    ]
    if unbakeable:
        other_types = [type(f).__name__ for _idx, f in unbakeable]
        raise RuntimeError(
            f"{parameter_name!r} 还被以下 Force 引用：{other_types}——"
            "本函数只烘焙 NonbondedForce 的 offset 与 CustomBondForce 的表达式，"
            "其它 Force 里这个参数不会被处理。不能假装整个 System 已经不再有这个活"
            "参数；请先确认这些 Force 是否也需要烘焙（当前未实现），或者改用别的"
            "参数名把它们隔离开。"
        )
    # CustomBondForce（当前只有配体内部库仑补偿项 `LigandInternalCoulomb` 走这里）：
    # 参数只出现在能量表达式里，没有 per-bond offset，所以把数值代进表达式、重建
    # 一个不再声明该 GlobalParameter 的力即可 —— 与 NonbondedForce 那侧同样满足
    # 「参数从 System 上彻底消失」这条契约（忘了设 λ 这条路径在结构上不存在）。
    # ⚠️ 必须**先建后删**：`system.getForce(i)` 返回的是 System 持有的引用，
    # `removeForce(i)` 会把它析构掉，之后再读那个 Python 句柄拿到的是垃圾内存
    # （实测能量表达式变成乱码字节）。本仓库栽过同类 SWIG 所有权的坑不止一次。
    # ⚠️ 同样**必须先全建完再统一删**：`removeForce(i)` 会把索引 > i 的力整体前移，
    # 先删 CustomBondForce 就会让下面 `nb_index` 失效（CustomBondForce 排在 NB 前面
    # 时，最后那一刀会砍到别的力上；末尾的 `remaining` 自检会 fail closed，但那已经
    # 是白跑一趟）。所以这里只建不删，替换统一在函数末尾按索引降序一次做完。
    replacements: List[Tuple[int, Any]] = [
        (idx, _bake_global_parameter_into_custom_bond_force(
            force, parameter_name, lambda_value
        ))
        for idx, force in other_hits
    ]

    nb_index, nb = nb_hits[0]
    num_particles = nb.getNumParticles()

    def _value(x, target_unit):
        """SWIG 对"恰好是 0"的 offset scale 有时会返回裸 `float` 而不是
        `Quantity`（已实测：`addParticleParameterOffset(..., 0.0*nm, ...)` 传
        进去，读回来可能就是裸 `0.0`），返回值类型不稳定。这里统一转换成给定
        单位下的裸 float，后续全部用 float 做累加，只在最后写回 System 时
        才重新套上单位——不依赖 OpenMM 返回值本身的类型。
        """
        if unit.is_quantity(x):
            return x.value_in_unit(target_unit)
        return float(x)

    # ---- particle offsets：同一 (parameter, particle) 上重复出现就 fail closed ----
    #
    # 2026-08-11 用户审阅时给的契约原本要求"多条 offset 先求和"（引用 OpenMM
    # 文档里 `parameter = base + Σ(global_i × scale_i)` 的公式）。实测直接
    # 用 Context 验证发现：**OpenMM 对同一个 (parameter, particle) 上的多条
    # `ParticleParameterOffset` 并不求和**——`getNumParticleParameterOffsets()`
    # 确实报出两条，但 Context 求值时只认最后一条（0.3 与 0.2 两条追加，
    # 结果对应的是 0.2，不是 0.5；exception 同理，0.01+0.02 两条追加，结果
    # 对应 0.02，不是 0.03）。那条公式里的 Σ 说的是"同一个粒子上挂了多个
    # 不同 GlobalParameter 各自的 offset 要相加"，不是"同一个 (parameter,
    # particle) 重复挂多条"。真实生产代码（`configure_charge_transfer_
    # decharging` 等）从不对同一个粒子重复调用
    # `addParticleParameterOffset`——每个粒子只出现一次。所以这里选择**不**
    # 复现这个没有文档、容易被误用的"后者覆盖前者"行为，改成 fail closed：
    # 一旦真的出现重复，说明调用方的假设已经出了问题，直接报错比"悄悄按
    # OpenMM 的隐藏规则取最后一条"更安全。
    target_particle_scale: Dict[int, Tuple[Any, Any, Any]] = {}
    kept_particle_offsets: List[Tuple[str, int, Any, Any, Any]] = []
    for i in range(nb.getNumParticleParameterOffsets()):
        pname, particle, q_scale, sigma_scale, eps_scale = nb.getParticleParameterOffset(i)
        particle = int(particle)
        if pname != parameter_name:
            kept_particle_offsets.append((pname, particle, q_scale, sigma_scale, eps_scale))
            continue
        if particle in target_particle_scale:
            raise RuntimeError(
                f"粒子 {particle} 上有多条挂在 {parameter_name!r} 下的 "
                "ParticleParameterOffset——OpenMM 对这种重复不做加法（实测确认，"
                "只认最后一条），这个函数拒绝猜测应该按哪种语义处理，请先在源头"
                "去重。"
            )
        target_particle_scale[particle] = (
            _value(q_scale, unit.elementary_charge),
            _value(sigma_scale, unit.nanometer),
            _value(eps_scale, unit.kilojoule_per_mole),
        )

    # ---- exception offsets，同理：重复就 fail closed，不猜语义 ----
    target_exception_scale: Dict[int, Tuple[Any, Any, Any]] = {}
    kept_exception_offsets: List[Tuple[str, int, Any, Any, Any]] = []
    for i in range(nb.getNumExceptionParameterOffsets()):
        pname, exc_index, cp_scale, sigma_scale, eps_scale = nb.getExceptionParameterOffset(i)
        exc_index = int(exc_index)
        if pname != parameter_name:
            kept_exception_offsets.append((pname, exc_index, cp_scale, sigma_scale, eps_scale))
            continue
        if exc_index in target_exception_scale:
            raise RuntimeError(
                f"exception {exc_index} 上有多条挂在 {parameter_name!r} 下的 "
                "ExceptionParameterOffset——同上，OpenMM 对这种重复不做加法，"
                "拒绝猜测语义，请先在源头去重。"
            )
        target_exception_scale[exc_index] = (
            _value(cp_scale, unit.elementary_charge**2),
            _value(sigma_scale, unit.nanometer),
            _value(eps_scale, unit.kilojoule_per_mole),
        )

    # ---- 建一个干净的新 NonbondedForce，完整复制配置 ----
    new_nb = openmm.NonbondedForce()
    _copy_nonbonded_force_settings(nb, new_nb)

    # ---- 逐粒子烘焙：没有目标 offset 的粒子原样复制 ----
    for idx in range(num_particles):
        q, sigma, epsilon = nb.getParticleParameters(idx)
        if idx in target_particle_scale:
            q_scale, sigma_scale, eps_scale = target_particle_scale[idx]
            q = q + (lambda_value * q_scale) * unit.elementary_charge
            sigma = sigma + (lambda_value * sigma_scale) * unit.nanometer
            epsilon = epsilon + (lambda_value * eps_scale) * unit.kilojoule_per_mole
        new_nb.addParticle(q, sigma, epsilon)

    # ---- 逐 exception 烘焙 ----
    for idx in range(nb.getNumExceptions()):
        p1, p2, charge_prod, sigma, epsilon = nb.getExceptionParameters(idx)
        if idx in target_exception_scale:
            cp_scale, sigma_scale, eps_scale = target_exception_scale[idx]
            charge_prod = charge_prod + (lambda_value * cp_scale) * unit.elementary_charge**2
            sigma = sigma + (lambda_value * sigma_scale) * unit.nanometer
            epsilon = epsilon + (lambda_value * eps_scale) * unit.kilojoule_per_mole
        new_nb.addException(int(p1), int(p2), charge_prod, sigma, epsilon)

    # ---- 重新声明其它 GlobalParameter，重新挂回其它 offset（原样保留） ----
    # kept_*_offsets 里的 scale 值同样可能是裸 float（同一个 SWIG 问题），
    # 重新写回前用 _value 统一转成明确单位的 Quantity，不能假设它们已经是
    # Quantity——那正是刚才炸掉的那个假设。
    for gi in range(nb.getNumGlobalParameters()):
        gname = nb.getGlobalParameterName(gi)
        if gname == parameter_name:
            continue
        new_nb.addGlobalParameter(gname, nb.getGlobalParameterDefaultValue(gi))
    for pname, particle, q_scale, sigma_scale, eps_scale in kept_particle_offsets:
        new_nb.addParticleParameterOffset(
            pname, particle,
            _value(q_scale, unit.elementary_charge) * unit.elementary_charge,
            _value(sigma_scale, unit.nanometer) * unit.nanometer,
            _value(eps_scale, unit.kilojoule_per_mole) * unit.kilojoule_per_mole,
        )
    for pname, exc_index, cp_scale, sigma_scale, eps_scale in kept_exception_offsets:
        new_nb.addExceptionParameterOffset(
            pname, exc_index,
            _value(cp_scale, unit.elementary_charge**2) * unit.elementary_charge**2,
            _value(sigma_scale, unit.nanometer) * unit.nanometer,
            _value(eps_scale, unit.kilojoule_per_mole) * unit.kilojoule_per_mole,
        )

    replacements.append((nb_index, new_nb))
    for idx, replacement in sorted(replacements, key=lambda t: -t[0]):
        system.removeForce(idx)
        system.addForce(replacement)

    remaining = _scan_forces_referencing_global_parameter(system, parameter_name)
    if remaining:
        raise RuntimeError(
            f"内部错误：烘焙后 {parameter_name!r} 仍出现在 "
            f"{[type(f).__name__ for _idx, f in remaining]} 上——烘焙没有做干净。"
        )
    return system


# 1/(4πε₀)，kJ·nm/(mol·e²)。OpenMM 的 `unit` 包不暴露 ε₀，所以只能写成字面量；
# 本文件里凡是配体内部非键表达式都引这一个名字，不再各写一遍字面量。
# ⚠️ 本仓库其它地方还有同值的副本（`ibs_engine._SHADOW_ONE_4PI_EPS0`）与一个**精度
# 不同**的副本（`apbs_correction.COULOMB_FACTOR_KJ_NM_PER_MOL_E2 = 138.93545585`）。
# 那是存量，不在本次改动范围内；**不要再新增第 N 份**。
ONE_4PI_EPS0_KJ_NM_PER_MOL_E2 = 138.935456


def collect_ligand_internal_exclusions(
    nb_force: openmm.NonbondedForce,
    perturbed_indices,
    reference_exclusions=None,
    system: openmm.System = None,
):
    """收集配体内部 1-2/1-3/1-4 排除对（键 / 角 / 刚性约束 / NB exception / 参考表）。

    `create_ligand_internal_force`（Group 2 的 CustomNonbondedForce）与
    `create_ligand_internal_coulomb_force`（去电荷腿的 λ 补偿项）**必须**用同一份
    排除表：两边一旦不一致，Stage-1/Stage-2 接缝上配体内部对就会漏算或重复算，
    而且是个静默的自由能偏差。所以这里只留一份实现。
    """
    perturbed_set = set(int(i) for i in perturbed_indices)
    exclusion_pairs = set()

    if system is not None:
        # === 1. 谐波键 (1-2 排除) ===
        for f in system.getForces():
            if isinstance(f, openmm.HarmonicBondForce):
                for i in range(f.getNumBonds()):
                    p1, p2, _, _ = f.getBondParameters(i)
                    if p1 in perturbed_set and p2 in perturbed_set:
                        exclusion_pairs.add((min(p1, p2), max(p1, p2)))
            
            # === 2. 谐波角 (1-3 排除，取首尾原子) ===
            elif isinstance(f, openmm.HarmonicAngleForce):
                for i in range(f.getNumAngles()):
                    p1, p2, p3, _, _ = f.getAngleParameters(i)
                    # ✅ 仅当首尾原子都在配体内才排除 (1-3)
                    if p1 in perturbed_set and p3 in perturbed_set:
                        exclusion_pairs.add((min(p1, p3), max(p1, p3)))
        
        # === 3. 刚性约束 (1-2 排除，GROMACS 常将含 H 键转为约束) ===
        # 🔑 核心修复：独立于力遍历，直接扫描系统级约束
        for i in range(system.getNumConstraints()):
            p1, p2, _ = system.getConstraintParameters(i)
            if p1 in perturbed_set and p2 in perturbed_set:
                exclusion_pairs.add((min(p1, p2), max(p1, p2)))
        
        # === 4. NonbondedForce 异常表 (1-4 排除) ===
        nb_forces = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
        if nb_forces:
            nb = nb_forces[0]
            for i in range(nb.getNumExceptions()):
                p1, p2, _, _, _ = nb.getExceptionParameters(i)
                p1, p2 = int(p1), int(p2)
                if p1 in perturbed_set and p2 in perturbed_set:
                    exclusion_pairs.add((min(p1, p2), max(p1, p2)))

    # === 5. 合并参考排除表 (来自原始 NonbondedForce 的 exceptions) ===
    if reference_exclusions:
        for p1, p2 in reference_exclusions:
            p1, p2 = int(p1), int(p2)
            if p1 in perturbed_set and p2 in perturbed_set:
                exclusion_pairs.add((min(p1, p2), max(p1, p2)))

    return exclusion_pairs


def create_ligand_internal_force(
    nb_force: openmm.NonbondedForce,
    perturbed_indices: List[int],
    particle_params,
    reference_exclusions=None,
    num_particles: int = None,
    system: openmm.System = None
):
    """
    构建配体-配体内部非键力 (Standard LJ + Coulomb) 和 1-4 恢复力。
    注意：此函数不分配 ForceGroup，调用者需自行设置并添加至 System。
    """
    if num_particles is None:
        num_particles = nb_force.getNumParticles()
    perturbed_set = set(perturbed_indices)

    expr = (
        "4*sqrt(epsilon1*epsilon2)*((sigma12/r)^12 - (sigma12/r)^6) + "
        f"{ONE_4PI_EPS0_KJ_NM_PER_MOL_E2}*q1*q2/r; sigma12 = 0.5*(sigma1+sigma2)"
    )
    ll_force = openmm.CustomNonbondedForce(expr)
    ll_force.addPerParticleParameter('q')
    ll_force.addPerParticleParameter('sigma')
    ll_force.addPerParticleParameter('epsilon')

    for i in range(num_particles):
        if particle_params and i < len(particle_params):
            q, sig, eps = particle_params[i]
        else:
            q, sig, eps = nb_force.getParticleParameters(i)
        ll_force.addParticle([
            q.value_in_unit(unit.elementary_charge),
            sig.value_in_unit(unit.nanometer),
            eps.value_in_unit(unit.kilojoule_per_mole)
        ])

    ll_force.addInteractionGroup(perturbed_set, perturbed_set)
    ll_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    # [MEM-00h，2026-08-06] 统一到基础力场的 1.0 nm、无 switching——理由见
    # BEUTLER_SOFTCORE_CUTOFF_NM 定义处；这个力此前用的是跟 softcore CV 一样的
    # 1.2nm+switch，跟基础 NonbondedForce 的 1.0nm 不一致，是同一个 MEM-00h
    # 存量问题的一部分。
    ll_force.setCutoffDistance(BEUTLER_SOFTCORE_CUTOFF_NM * unit.nanometer)
    ll_force.setUseLongRangeCorrection(False)

    # 排除表由 `collect_ligand_internal_exclusions` 单点收集（1-2/1-3/1-4）。
    exclusion_pairs = collect_ligand_internal_exclusions(
        nb_force, perturbed_set, reference_exclusions, system
    )

    # === 6. 执行排除添加 (严格去重) ===
    for p1, p2 in exclusion_pairs:
        ll_force.addExclusion(p1, p2)

    # 🔍 诊断输出 (可选，生产环境可注释)
    # 去重键由消息文本派生：对数一变就重新打一行（一次 4W53 run 里同一行重复 52 次）。
    _msg = f"  [Group2 排除表] 共收集 {len(exclusion_pairs)} 对配体内部排除 (1-2/1-3/1-4)"
    print_once(f"group2_exclusions|{_msg}", _msg)

    # 1-4 恢复力
    ll_14_force = None
    exceptions_14 = []
    for i in range(nb_force.getNumExceptions()):
        p1, p2, chargeProd, sigma, epsilon = nb_force.getExceptionParameters(i)
        p1, p2 = int(p1), int(p2)
        if p1 in perturbed_set and p2 in perturbed_set:
            has_charge = chargeProd.value_in_unit(unit.elementary_charge**2) != 0
            has_lj = epsilon.value_in_unit(unit.kilojoule_per_mole) != 0
            if has_charge or has_lj:
                exceptions_14.append((p1, p2, chargeProd, sigma, epsilon))

    if exceptions_14:
        expr_14 = (
            "4*epsilon*((sigma/r)^12 - (sigma/r)^6) + "
            f"{ONE_4PI_EPS0_KJ_NM_PER_MOL_E2}*chargeProd/r"
        )
        ll_14_force = openmm.CustomBondForce(expr_14)
        ll_14_force.addPerBondParameter('chargeProd')
        ll_14_force.addPerBondParameter('sigma')
        ll_14_force.addPerBondParameter('epsilon')
        for p1, p2, cp, sig, eps in exceptions_14:
            ll_14_force.addBond(p1, p2, [
                cp.value_in_unit(unit.elementary_charge**2),
                sig.value_in_unit(unit.nanometer),
                eps.value_in_unit(unit.kilojoule_per_mole)
            ])

    ll_force.setUseSwitchingFunction(False)
    ll_force.setSwitchingDistance(BEUTLER_SOFTCORE_CUTOFF_NM * unit.nanometer)
    return ll_force, ll_14_force


LIGAND_INTERNAL_COULOMB_FORCE_NAME = "LigandInternalCoulomb"


def create_ligand_internal_coulomb_force(
    nb_force: openmm.NonbondedForce,
    perturbed_indices,
    ligand_charges_e,
    system: openmm.System = None,
    reference_exclusions=None,
    scale_expr: str = "1",
    global_parameters=None,
):
    """配体内部 ≥1-5 普通对的库仑项，显式逐对、**无截断**。返回 CustomBondForce 或 None。

    **为什么必须是一个独立的力**：`NonbondedForce` 里一个粒子只有一份电荷。一旦让
    它随 λ 缩放，配体–环境对按 λ 走、配体–配体对就必然按 λ² 走 —— 没有任何写法能
    在同一个力里把两者分开。所以「配体内部库仑保持 λ 无关」只能靠一个附加项。
    这不是风格选择。

    **两个 stage 用同一份对表**（`collect_ligand_internal_exclusions` 的补集），
    Stage-1/Stage-2 接缝因此按构造恒等：

      * 去电荷腿：``scale_expr=f"(1 - {lambda_name}^2)"``。主 NB 力给出
        λ²·U_intra，本力补上 (1-λ²)·U_intra，合计恒为 U_intra。λ=1 时本力**恒等
        于 0**，物理端点逐位等于原 System —— 正是 P0-01 要保护的不变量。
      * vanishing 腿：``scale_expr="1"``。配体已整体从主 NB 力剥离，本力就是全部。

    **为什么是 CustomBondForce 而不是 CustomNonbondedForce**：后者必须带 cutoff，
    而 Group 2 用的是 1.0 nm（`BEUTLER_SOFTCORE_CUTOFF_NM`）。跨度超过 1 nm 的配体
    会被截掉最长的内部对（实测某体系最大内部距离 1.46 nm），而去电荷腿那侧的
    PME 是不截断的 —— 两边口径就不一致了。逐对展开既无截断、两个 stage 又逐位一致。

    **为什么是 CustomBondForce 而不是 CustomNonbondedForce**：后者必须带 cutoff，
    而 Group 2 用的是 1.0 nm（`BEUTLER_SOFTCORE_CUTOFF_NM`）。跨度超过 1 nm 的配体
    会被截掉最长的内部对（实测某体系最大内部距离 1.46 nm），而去电荷腿那侧的
    PME 是不截断的 —— 两边口径就不一致了。逐对展开既无截断、两个 stage 又逐位一致。

    ⚠️ **刻意不启用 PBC（不走最小镜像）**，与 `NonbondedForce` 的 exception 同口径。

    这条来回改过一次，把理由记死免得再来：**OpenMM 的 `NonbondedForce` exception
    本身就不做最小镜像**，它假定成键原子总在同一个周期像里。这不是断言，是可运行的
    判据，见 `tests/test_intramolecular_coulomb_is_lambda_independent.py::
    test_nonbonded_exceptions_ignore_minimum_image`（那里现场建体系、现场算，
    不抄任何数字）。

    所以配体内部这一段的**唯一正确前提是"坐标到达时分子是完整的"**（靠平移补全，
    见 PBC-01）。在这个前提下最小镜像与直接距离逐位相同，加 PBC 不解决任何问题；
    而在前提被破坏时，加 PBC 只会让分子内的两半口径不一致（本项"对"、1-2/1-3/1-4
    仍然错），整体照样是错的 —— 等于把上游的坐标问题掩盖成一个更难查的形态。
    额外的坏处：最小镜像在配体跨度 > L/2 时会静默返回错误的距离。

    真正的保证在上游：分子完整性。不要在这里加 PBC。

    Args:
        ligand_charges_e: {粒子索引: 电荷(e)}，必须是**原始**电荷，不是被清零/
            缩放之后的。
        global_parameters: `scale_expr` 里用到的 global parameter，形如
            ``[(name, default)]``。OpenMM 要求**每个 Force 各自注册**它用到的
            global parameter（`NonbondedForce` 上注册过不算数），漏了会在建
            Context 时抛 "Unknown variable"。所以这里收成 builder 的入参，而不是
            留给调用方 —— 「注册是 caller 的活」这个形状在本仓库已经出过静默算错的事故。
    """
    perturbed = sorted(set(int(i) for i in perturbed_indices))
    perturbed_set = set(perturbed)
    charges = {int(k): float(v) for k, v in dict(ligand_charges_e).items()}
    missing = perturbed_set - set(charges)
    if missing:
        raise ValueError(
            f"create_ligand_internal_coulomb_force 缺少 {len(missing)} 个配体原子的原始电荷："
            f"{sorted(missing)[:8]}… 逐对展开不允许缺项（缺一对就是静默少算一段内部库仑）。"
        )

    excluded = collect_ligand_internal_exclusions(
        nb_force, perturbed_set, reference_exclusions, system
    )

    force = openmm.CustomBondForce(
        f"({scale_expr})*{ONE_4PI_EPS0_KJ_NM_PER_MOL_E2}*chargeProd/r"
    )
    # 名字是身份：`build_ibs_dual_system` 拿到的 vanishing 模板可能已经带着一份
    # （charge-transfer 的烘焙交接会把它带过 stage 边界），靠名字显式去重，
    # 不靠"烘焙后配体电荷为 0、所以新建的那份自然是空"这种隐式巧合。
    force.setName(LIGAND_INTERNAL_COULOMB_FORCE_NAME)
    force.addPerBondParameter("chargeProd")
    for name, default in (global_parameters or []):
        force.addGlobalParameter(str(name), float(default))
    n_pairs = 0
    for a in range(len(perturbed)):
        for b in range(a + 1, len(perturbed)):
            p1, p2 = perturbed[a], perturbed[b]
            if (min(p1, p2), max(p1, p2)) in excluded:
                continue
            n_pairs += 1
            qq = charges[p1] * charges[p2]
            if qq == 0.0:
                continue
            force.addBond(p1, p2, [qq])

    n_total = len(perturbed) * (len(perturbed) - 1) // 2
    n_excluded_internal = sum(
        1 for (p1, p2) in excluded if p1 in perturbed_set and p2 in perturbed_set
    )
    if n_pairs + n_excluded_internal != n_total:
        raise RuntimeError(
            f"配体内部对账不平：普通对 {n_pairs} + 排除对 {n_excluded_internal} "
            f"≠ 全部对 {n_total}。排除表与逐对枚举用的不是同一个口径。"
        )
    if force.getNumBonds() == 0:
        return None
    return force
