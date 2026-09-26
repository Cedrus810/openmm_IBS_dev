

# ============================================================================
# B6-FIX（2026-08-04）：把"目标 Hamiltonian"与"每条腿实现该目标的环境专用长程处理"
# 拆开。
#
# 起因是一条实测出来的自相矛盾：膜运行的**溶剂腿**（4.05 nm 纯水盒）在
# `final_results.json` 里带着这句理由——
#
#     disabled_by_membrane_forcefield_protocol: …配体所在口袋的局域密度既不是水
#     也不是体相脂质
#
# 而那条腿里配体周围**恰恰就是**均匀体相水。根因是判据只有一个全局布尔：
#
#     apply_lrc = (dispersion_protocol == "legacy_uniform_density_lrc")
#
# 它没有环境维度，于是"复合物腿口袋里不均匀"这个正确理由被原样套到了水盒上，
# 把一条**合法**的 bulk-water 尾项修正一起关掉了。实测代价：同一个配体
# （Atenolol，41 原子）的溶剂腿 vanishing 从 96.96 变成 83.83 kJ/mol（−13.1）。
#
# 正确的分层：
#
#     dispersion_protocol  = **目标**：所选力场原始参数化时的色散条件
#                            （Amber Lipid21 = 开各向同性 LRC；CHARMM36 = force-switch 不加 LRC）
#     ↓  每条腿在**自己的环境**里怎么达成这个目标
#     实现              = f(目标, 该腿配体所处环境)
#
# 环境维度取 `system_type`，那是**用户在输入文件里显式声明**的值（B1 的规矩是
# 「不许按残基名猜 system_type」，不是「不许用声明出来的 system_type 分派」），
# 所以按它自动切换实现是合法的、可审计的，不是运行时猜测。
#
# 溶剂腿天然落在 soluble 一侧：`runabfe` 构造溶剂腿 pipeline 时**刻意不传**
# `environment_type`/`membrane`（B1 的接线契约测试钉着这一点），于是它解析出来
# 就是 soluble —— 不需要为这条腿新造任何标记。
# ============================================================================

# 实现名（进 provenance / final_results.json，机器可读，别改措辞）。
DISPERSION_IMPL_UNIFORM_BULK_ANALYTIC_TAIL = "uniform_bulk_density_analytic_tail"
DISPERSION_IMPL_TRUNCATED_NO_TAIL = "truncated_no_analytic_tail"
DISPERSION_IMPL_FORCE_SWITCH_NO_TAIL = "force_switch_no_tail_by_forcefield_design"


# 🔑 [2026-09-15] 这条腿里配体**待在哪**。与 `environment_type`（soluble/membrane）
# 正交：可溶体系的复合物腿，配体照样埋在蛋白口袋里。
LIGAND_SURROUNDING_BULK = "bulk"       # 纯溶剂腿：配体周围就是均匀体相
LIGAND_SURROUNDING_POCKET = "pocket"   # 复合物腿：配体埋在口袋里


def resolve_leg_dispersion_implementation(
    dispersion_protocol: Optional[str],
    environment_type: Optional[str] = None,
    ligand_surrounding: Optional[str] = None,
) -> Dict[str, Any]:
    """给**一条腿**定"炼金 ligand–environment 的长程色散怎么处理"。

    返回里 `alchemical_uniform_density_lrc` 就是那个唯一的布尔判据；
    `target_met` 说明这条腿有没有真正达成力场参数化条件所要求的目标
    —— 膜复合物腿目前**达不成**（需要 §1.3 路线 C 的非均匀修正，未实现），
    这一点必须如实写进结果，而不是把"关掉了"记成"处理好了"。

    | 目标 dispersion_protocol | 该腿环境 | 炼金 LRC | target_met |
    | --- | --- | --- | --- |
    | `legacy_uniform_density_lrc` | soluble | 开 | 是（改动前唯一行为，逐位不变）|
    | `ff_native_isotropic_lrc`    | soluble（含膜运行的溶剂腿）| **开** | 是 |
    | `ff_native_isotropic_lrc`    | membrane | 关 | **否**（路线 C 未实现）|
    | `ff_native_force_switch_no_lrc` | 任意 | 关 | 是（力场本身就不加 LRC）|

    ⚠️ 非 legacy 目标**必须显式给出 `environment_type`**：这一维决定加不加修正，
    缺省猜哪一边都会静默出错（猜 soluble → 膜口袋里加上无效修正；猜 membrane →
    又把水盒的合法修正关掉，即本次要修的 bug）。所以缺就 raise。
    """
    protocol = str(dispersion_protocol or "").strip().lower()
    if not protocol:
        protocol = DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC

    if protocol == DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC:
        # legacy 是本改动之前唯一存在的路线，且 membrane+legacy 在
        # `resolve_dispersion_protocol` 就已 fail closed，所以这里不需要环境维度。
        #
        # 🔑🔑 [2026-09-15] **但"不是膜"不等于"配体周围是均匀体相"。**
        #
        # `resolve_dispersion_protocol` 那道 fail-closed 的判据是 `is_membrane`，
        # 而它引用的论证是：
        #     「`lj_tail_lrc_coeff[k]/V(t)` 假设配体周围是**均匀体相密度**；
        #       配体埋在口袋里时这个假设直接不成立」
        # 这段话对**可溶蛋白的口袋**同样成立 —— 复合物腿里配体的第一/第二壳层是
        # 蛋白，而 `coeff/V` 用的是**盒平均**密度。判据卡的是环境类型，
        # 论证讲的是配体待在哪，两者不是一回事。
        #
        # ⚠️ **不改数值行为**：`alchemical_uniform_density_lrc` 仍然是 True。
        # 实测量级很小 —— brd4_ligand1 两条腿离线复算：
        #     complex  N_env=39633 V=416.9 ρ=95.07  E_LRC=−13.2983 kJ/mol
        #     solvent  N_env= 5914 V= 62.3 ρ=94.87  E_LRC=−12.7079 kJ/mol
        #     两腿差 −0.5904 kJ/mol（÷56 个有 LJ 的配体原子 = −0.0105/原子）
        # 体积差 6.7 倍被 `coeff ∝ N_env` 抵消掉了（两边 ρ_env 只差 0.2%），
        # 所以关掉它反而是引入一个更大的、没验证过的改动。
        #
        # 改的只是**如实记账**：这条腿的 `target_met` 不再无条件报 `True`。
        # 本函数 docstring 自己的规矩：「必须如实写进结果，而不是把'关掉了'
        # 记成'处理好了'」—— 同理，也不许把"没验证过"记成"达成了"。
        # `ligand_surrounding` 不声明时**逐字保持原行为**（老调用方不受影响）。
        _pocket = (
            str(ligand_surrounding or "").strip().lower()
            == LIGAND_SURROUNDING_POCKET
        )
        return {
            "dispersion_protocol": protocol,
            "environment_type": resolve_environment_type(environment_type),
            "ligand_surrounding": (
                str(ligand_surrounding).strip().lower()
                if ligand_surrounding else None
            ),
            "ligand_environment_is_uniform_bulk": not _pocket,
            "alchemical_uniform_density_lrc": True,
            "implementation": DISPERSION_IMPL_UNIFORM_BULK_ANALYTIC_TAIL,
            "target_met": not _pocket,
            "reason": (
                "" if not _pocket else
                "uniform_bulk_density_assumed_but_ligand_is_in_a_pocket: "
                "解析尾项按 `coeff[k]/V` 用的是**盒平均**密度，而这条腿里配体埋在"
                "口袋里、局域密度不是体相。修正仍然施加（实测两腿差仅 ~0.6 kJ/mol，"
                "关掉它是更大的未验证改动），但**不声称达成**力场参数化条件。"
                "正解与膜体系同属 §1.3 路线 C（非均匀色散修正），尚未实现。"
            ),
        }

    if environment_type is None:
        raise ValueError(
            f"dispersion_protocol={protocol!r} 是非 legacy 路线，必须显式给出这条腿的 "
            "environment_type。它决定炼金 ligand–environment 的均匀密度尾项加不加：\n"
            "    · 纯水/可溶腿：配体周围就是均匀体相 ⟹ 该修正**成立**，关掉是错的；\n"
            "    · 膜复合物腿：口袋局域密度既不是水也不是体相脂质 ⟹ 该修正不成立。\n"
            "    缺省猜任何一边都会静默产出错误的 ΔG（B6-FIX 修的正是猜错方向那一版）。"
        )
    resolved_env = resolve_environment_type(environment_type)
    is_membrane = resolved_env == ENVIRONMENT_TYPE_MEMBRANE

    if protocol == DISPERSION_PROTOCOL_FF_NATIVE_FORCE_SWITCH_NO_LRC:
        return {
            "dispersion_protocol": protocol,
            "environment_type": resolved_env,
            "ligand_environment_is_uniform_bulk": not is_membrane,
            "alchemical_uniform_density_lrc": False,
            "implementation": DISPERSION_IMPL_FORCE_SWITCH_NO_TAIL,
            # 力场原始参数化就不带 LRC，所以"不加"正是达成目标。
            "target_met": True,
            "reason": (
                "no_analytic_tail_by_forcefield_design: "
                f"dispersion_protocol={protocol!r} 的力场原始参数化条件是 force-switch "
                "且**不加**长程色散修正，因此炼金 ligand–environment 也不加解析尾项"
            ),
        }

    if protocol == DISPERSION_PROTOCOL_FF_NATIVE_ISOTROPIC_LRC:
        if is_membrane:
            return {
                "dispersion_protocol": protocol,
                "environment_type": resolved_env,
                "ligand_environment_is_uniform_bulk": False,
                "alchemical_uniform_density_lrc": False,
                "implementation": DISPERSION_IMPL_TRUNCATED_NO_TAIL,
                # ⚠️ 达不成目标：力场是在开着各向同性色散修正的条件下拟合的，
                # 而这条腿的炼金 ligand–environment 项现在是**截断**的。
                # 正解是 §1.3 路线 C（膜非均匀色散修正），尚未实现。
                "target_met": False,
                # §1.3 指定的机器可读字符串，措辞不要改。
                "reason": (
                    "disabled_by_membrane_forcefield_protocol: "
                    f"dispersion_protocol={protocol!r} 且该腿 environment_type='membrane' 时，"
                    "炼金 ligand–environment 的均匀体相密度 LRC 不适用"
                    "（配体所在口袋的局域密度既不是水也不是体相脂质）；"
                    "环境–环境色散仍按所选力场的原始参数化条件由基础 NonbondedForce 处理。"
                    "达成目标需要 §1.3 路线 C 的非均匀色散修正（未实现），"
                    "所以本腿记为 target_met=false 而不是"
                    "「已正确处理」"
                ),
            }
        return {
            "dispersion_protocol": protocol,
            "environment_type": resolved_env,
            "ligand_environment_is_uniform_bulk": True,
            "alchemical_uniform_density_lrc": True,
            "implementation": DISPERSION_IMPL_UNIFORM_BULK_ANALYTIC_TAIL,
            "target_met": True,
            "reason": "",
        }

    raise ValueError(
        f"dispersion_protocol={protocol!r} 没有对应的每腿实现映射；"
        f"合法值 {list(DISPERSION_PROTOCOLS)}，其中路线 B/C 在 "
        "`resolve_dispersion_protocol` 就已 NotImplementedError。"
    )


# ============================================================================
# §3.3 膜输入核对 + §9 膜质量门（docs/status/memtodolist.md §3.3 / §9 / §13.3）
#
# 这两节的共同前提，清单说得很直接：
#   §3.3 输入必须是**已经完成膜构建和主要平衡**的体系，
#        不依赖当前通用 10 ns 预平衡去完成脂质重排或蛋白插膜。
#   §9   通用 10 ns **不是**膜平衡充分性的证明；每个量都要给时间序列与
#        末段窗口的漂移斜率，不能只报平均值；判据统一为
#        "末段 ≥ 20 ns 内线性漂移小于阈值"（阈值见 §13.3）。
#        质量门失败时回到膜体系平衡，**不允许靠增加 ABFE 窗口掩盖**。
#
# 归档部分（.top / 全部 .itp / 位置限制 / 力场 include 的递归 SHA256）复用
# `runabfe._gromacs_dependency_hashes()`，本节不再造第二套。
# ============================================================================

MEMBRANE_INPUT_PROTOCOL_VERSION = 1
# v2（2026-08-02）：
#   * 新增 `apl_protein_corrected_nm2` 观测量，§13.3 的 APL 绝对值门改为比校正后的值
#     （含蛋白膜拿 raw APL 比纯脂文献值必然偏大，实测 0.826 vs 0.645）；
#   * 时间轴改为由 `frame_interval_ps` 显式重建，不再吃 mdtraj 给 DCD 的整数帧号
#     （v1 下那条 10 ns 轨迹被当成 0.499 ns，所有 per-ns 判据都错 20 倍）。
# v1 的报告与 v2 的**不可直接比较**：per-ns 斜率与 APL 绝对值口径都变了。
# v3（2026-08-02，同日）：
#   * MEM-10：`superpose` 原地污染修掉。此前倾角、蛋白横截面/校正后 APL、疏水核内水、
#     水层间隙、密度分布、脂质横向弛豫都在"对齐到蛋白骨架"的坐标上算，而 slab 边界
#     取自对齐前坐标。实测 τ 因此被放大 12 倍（11.57 → 139.36 ns）、倾角漂移被压掉
#     （0.477 → 1.274 °/window）。**v2 的这些数字全部作废。**
#   * MEM-11：弛豫时间尺度改时间平均 MSD + 声明 lag 窗口；该判据从 `checks`
#     降级为 `statistics.equilibration_vs_relaxation` 诊断。
#   * MEM-13：口袋/配体 RMSD 改成"对齐骨架后不重拟合"的 pose 漂移
#     （0.0760 → 0.0857、0.0493 → 0.0833 nm）。
#   * MEM-12：新增 `statistics.apl_vs_pure_lipid_literature` 诊断（不判门）。
MEMBRANE_QUALITY_GATE_PROTOCOL_VERSION = 3

# §9 质量门的执行模式。
#
#   "enforce"（默认）—— 门未过即阻断。这是 §9 的原意：
#       "质量门失败时回到膜体系平衡，不允许靠增加 ABFE 窗口掩盖。"
#   "advisory"        —— 照样计算、照样落盘报告、失败照样大声 WARNING 并写进
#       provenance，但**不阻断**。用于"先把管路跑通"的探索阶段。
#
# 为什么提供 advisory 而不是让人注释掉调用：门被注释掉就**没有记录**，
# 事后无从知道当时到底过没过。advisory 下报告仍然完整落盘、
# `membrane_quality_gate_mode` 进 provenance，所以"当时是放行跑的"这件事赖不掉。
#
# ⚠️ advisory **不是**生产资格。任何要报出的 ΔG_bind 都必须在 enforce 下通过。
MEMBRANE_QUALITY_GATE_MODE_ENFORCE = "enforce"
MEMBRANE_QUALITY_GATE_MODE_ADVISORY = "advisory"
MEMBRANE_QUALITY_GATE_MODES = (
    MEMBRANE_QUALITY_GATE_MODE_ENFORCE,
    MEMBRANE_QUALITY_GATE_MODE_ADVISORY,
)


def resolve_membrane_quality_gate_mode(value: Optional[str]) -> str:
    """规范化 §9 质量门模式；未声明即 `enforce`（默认严格）。"""
    if value is None or str(value).strip() == "":
        return MEMBRANE_QUALITY_GATE_MODE_ENFORCE
    normalized = str(value).strip().lower()
    if normalized not in MEMBRANE_QUALITY_GATE_MODES:
        raise ValueError(
            f"membrane_quality_gate={value!r} 非法；允许 {list(MEMBRANE_QUALITY_GATE_MODES)}。"
            "不会静默回落——拼错的值被当成 enforce（或反之）都会让人误判结果的资格。"
        )
    return normalized

# §3.3：这些是"记录"类要求，缺一项就说明输入来源不可追溯 → fail closed。
# §9/§15 的预平衡时长门是对**总平衡时长**的要求，不是"本流程必须自己再跑 100 ns"。
# 输入若已由上游（CHARMM-GUI step6/step7、外部 slurm 作业）平衡过，那段时长应当计入。
#
# 但实践里常见的情形是"上游确实跑完了生产，可时长没记下来"——例如手上只有一个
# `step7_production.gro` 末帧。为此声明里提供两条**互斥**的合法表述，二者必居其一：
#
#   1. `upstream_equilibration_ns`: 正数 —— 明确的上游平衡时长，计入总时长；
#   2. `upstream_equilibration_status = "completed_length_unrecorded"` —— 上游生产已
#      完成但时长不可考。此时**跳过标称时长预检**，并要求 `final_equilibration_job`
#      指向证据（作业 ID / 路径），理由进 provenance。
#
# 两者都不给就 fail closed：沉默等于让未充分平衡的体系混进来。
#
# ⚠️ 无论走哪条，§9 那道**实测**质量门（末段漂移 + 脂质横向弛豫时间尺度）都不受影响，
# 它才是真正的判据；标称时长只是一道便宜的早期护栏。
MEMBRANE_UPSTREAM_EQUILIBRATION_FIELD = "upstream_equilibration_ns"
MEMBRANE_UPSTREAM_STATUS_FIELD = "upstream_equilibration_status"
MEMBRANE_UPSTREAM_STATUS_COMPLETED_UNRECORDED = "completed_length_unrecorded"

MEMBRANE_INPUT_REQUIRED_PROVENANCE_FIELDS = (
    "build_tool",
    "build_parameters",
    "final_equilibration_job",
    "source_structure_id",
    # §1.1：SERT 这类转运体必须记录构象态，不同构象态的 S1 可及性不同、不可混用。
    "conformational_state",
    "membrane_composition",
    # §3.0：结合位点溶剂暴露程度决定空腔填充迟滞的风险等级，必须在选体系时写死。
    "binding_site_solvent_exposure",
    # §9：上下叶脂质数"如何确定"必须有依据，不是随手对半分。
    "leaflet_assignment_basis",
)

# §1.1 要求记录构象态，理由是"不同构象态的 S1 可及性和水化程度不同，不可混用"。
# 但那条是针对转运体（SERT 的 outward-open / occluded / inward-open）写的；换成
# GPCR 或作者本来就不区分构象态的体系时，硬要填一个值只会得到编造的记录。
#
# 所以接受这个显式哨兵：字段仍然**必填**（不能静默缺失），但可以填 `unspecified`，
# provenance 里会明确记着"未声明"而不是假装填过。跨构象态比较的可追溯性由输入文件
# 的递归 SHA256（`runabfe._gromacs_dependency_hashes`）兜底——同一份坐标/拓扑就是
# 同一个构象，换了输入指纹就变。
MEMBRANE_CONFORMATIONAL_STATE_UNSPECIFIED = "unspecified"

MEMBRANE_BINDING_SITE_EXPOSURE_LEVELS = (
    "solvent_accessible",
    "interfacial",
    "lipid_exposed",
    "buried_in_hydrophobic_core",
)

# 常见磷脂头基参考原子（判定脂质分子属于上叶还是下叶）。胆固醇无 P，
# 用其羟基氧兜底。识别不到参考原子的脂质会被单独报出来，不静默丢弃。
LIPID_HEAD_REFERENCE_ATOM_NAMES = ("P", "P1", "P8", "P31", "PO4")
LIPID_HEAD_FALLBACK_ATOM_NAMES = ("O3", "O1", "OH", "ROH")

# §9 要求"保存并审查"且 §13.3 给了数值阈值的量 —— 缺一个就 fail closed。
REQUIRED_MEMBRANE_QUALITY_OBSERVABLES = (
    "apl_nm2",
    "bilayer_thickness_nm",
    "lipid_tail_order_parameter",
    "protein_backbone_rmsd_nm",
    "transmembrane_tilt_deg",
    "pocket_rmsd_nm",
    "ligand_heavy_atom_rmsd_nm",
    "box_xy_area_nm2",
    "box_z_nm",
    "box_volume_nm3",
)

# §9 同样要求保存、但属于分布/计数类（§13.3 没给标量阈值）的诊断量。
# 不判阈值不等于可以不存——缺了就没法回答"膜到底平衡没平衡"。
REQUIRED_MEMBRANE_QUALITY_DIAGNOSTICS = (
    "density_profile_along_normal",
    "leaflet_composition",
    "anomalous_pocket_water_count",
    "membrane_periodic_image_contacts",
    "membrane_undulation_or_residual_tension",
    "lipid_lateral_relaxation_timescale_ns",
    "equilibration_length_ns",
)

# 只在共炼金离子路线下额外要求（§9 末两条）。
COION_MEMBRANE_QUALITY_OBSERVABLES = (
    "coion_abs_z_from_midplane_nm",
    "coion_ligand_min_image_distance_nm",
    "coion_protein_heavy_atom_distance_nm",
    "coion_nearest_phosphorus_distance_nm",
    "coion_first_shell_water_count",
)
COION_MEMBRANE_QUALITY_DIAGNOSTICS = ("coion_z_histogram",)

# §9 要求用脂质横向弛豫时间尺度**论证**预平衡时长够。§13 没给这个倍数，
# 这里取 1.0 作为最低门槛（预平衡至少覆盖一个横向弛豫时间），并明确标注它是
# 本实现补的、不是清单原有阈值——改它要同时改这条注释。
MEMBRANE_EQUILIBRATION_MIN_RELAXATION_MULTIPLE = 1.0

# 膜体系预平衡时长下限。依据 §9（"通用 10 ns 几乎肯定不够"）与 §15
# （"膜体系这几个数都会变大：预平衡 ≥ 100 ns"）。用于**配置时**就把
# "拿可溶体系的 10 ns 默认值去跑膜"挡住，而不是等质量门事后否掉——
# 后者要先烧掉整轮采样才知道。
MEMBRANE_MIN_EQUILIBRATION_NS = 100.0

# 膜与其周期镜像之间至少要留多厚的水层。§9 只要求"检查是否存在膜与周期镜像的
# 异常接触"、没给数；这里取 2.0 nm（≈ 两倍 1.0 nm cutoff），并标注为本实现补的。
MEMBRANE_MIN_WATER_SLAB_NM = 2.0

# 判定"脂质尾链进入疏水核"的头基裕度：|z − 中面| < 半膜厚 − 该裕度即算核内。
MEMBRANE_HYDROPHOBIC_CORE_HEADGROUP_MARGIN_NM = 0.5

# 估计脂质横向弛豫时间尺度时使用的特征位移（约一个脂质直径）。
LIPID_LATERAL_RELAXATION_REFERENCE_DISPLACEMENT_NM = 0.8

# 时间平均 MSD 的线性拟合 lag 窗口（ns）。
#
# 依据是实测（memtest 100 ns 膜蛋白体系）：1–30 ns 区间 MSD ~ t^0.80，
# 也就是短 lag 仍处在亚扩散/笼振区，从 5 ns 起才接近线性；上限 30 ns 是为了
# 保留足够多的独立时间原点（100 ns 轨迹在 30 ns lag 上还有约 70 ns 的原点可用）。
# 换窗口会改变 D 与 τ，所以实际用的窗口**必须**随报告落盘（见 details 里的
# `fit_lag_window_ns` / `fit_lag_window_source`）——"换窗口把数调好看"藏不住。
LIPID_LATERAL_MSD_FIT_LAG_MIN_NS = 5.0
LIPID_LATERAL_MSD_FIT_LAG_MAX_NS = 30.0

# 纯 POPC 双层的横向扩散系数参考值（nm²/ns），用于判读上面那个 τ 的量级：
# 文献量级 D ≈ 0.008 nm²/ns（≈ 0.8e-7 cm²/s）→ 位移 0.8 nm 需 τ ≈ 20 ns。
# ⚠️ 仅作**诊断锚点**，不是阈值：含蛋白膜的脂质会被蛋白减速（annular lipid），
# 小膜片还有有限尺寸效应，所以偏离它不构成"体系有问题"的证据。
PURE_POPC_REFERENCE_LATERAL_DIFFUSION_NM2_PER_NS = 0.008


def linear_drift_per_ns(times_ns, values) -> Dict[str, Any]:
    """对时间序列做一次线性拟合，返回每 ns 漂移斜率。

    §9 明确要求"每个量都必须给出时间序列和末段窗口的漂移斜率，不能只报平均值"。
    """
    t = np.asarray(times_ns, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.shape != y.shape or t.ndim != 1:
        raise ValueError(f"时间与数值序列形状不匹配：{t.shape} vs {y.shape}")
    if t.size < 2:
        raise ValueError("线性漂移至少需要 2 个点")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))):
        raise ValueError("时间序列含非有限值，拒绝拟合")
    span = float(t[-1] - t[0])
    if span <= 0.0:
        raise ValueError(f"时间序列跨度非正：{span}")
    slope, intercept = np.polyfit(t, y, 1)
    return {
        "slope_per_ns": float(slope),
        "intercept": float(intercept),
        "mean": float(np.mean(y)),
        "std": float(np.std(y, ddof=1)) if y.size > 1 else 0.0,
        "n_points": int(y.size),
        "span_ns": span,
        "t_start_ns": float(t[0]),
        "t_end_ns": float(t[-1]),
    }


def _tail_window(times_ns, values, window_ns: float):
    """取末段 `window_ns` 的子序列；覆盖不足则抛错（由调用方转成 fail closed）。"""
    t = np.asarray(times_ns, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.size != y.size:
        raise ValueError(f"时间与数值长度不一致：{t.size} vs {y.size}")
    if t.size < 2:
        raise ValueError("时间序列至少需要 2 个点")
    total_span = float(t[-1] - t[0])
    if total_span + 1.0e-9 < window_ns:
        raise ValueError(
            f"序列总跨度 {total_span:.3f} ns 覆盖不了要求的末段窗口 {window_ns:.3f} ns。"
            "§9：通用 10 ns 不是膜平衡充分性的证明——请延长预平衡，"
            "不要缩短判据窗口来让门变绿。"
        )
    cutoff = float(t[-1]) - float(window_ns)
    mask = t >= cutoff - 1.0e-9
    if int(np.count_nonzero(mask)) < 2:
        raise ValueError(
            f"末段 {window_ns:.3f} ns 内只有 {int(np.count_nonzero(mask))} 个采样点，"
            "无法拟合漂移斜率。请提高该量的保存频率。"
        )
    return t[mask], y[mask]


def _series_pair(name: str, series: Any):
    if not isinstance(series, dict):
        raise ValueError(
            f"观测量 {name!r} 必须是 {{'times_ns': [...], 'values': [...]}} 形式的时间序列，"
            f"收到 {type(series).__name__}。§9 不接受只报平均值。"
        )
    for key in ("times_ns", "values"):
        if key not in series:
            raise ValueError(f"观测量 {name!r} 缺少 {key!r}")
    return series["times_ns"], series["values"]


def evaluate_membrane_quality_gate(
    observables: Dict[str, Any],
    diagnostics: Optional[Dict[str, Any]] = None,
    literature_apl_nm2: Optional[float] = None,
    require_coion: bool = False,
    tail_window_ns: float = MEMBRANE_QUALITY_GATE_TAIL_WINDOW_NS,
    pure_lipid_reference_apl_nm2: Optional[float] = None,
) -> Dict[str, Any]:
    """按 §9 + §13.3 判膜质量门。

    返回结构化报告；`passed=False` 时**唯一正解是回到膜体系平衡**，
    不允许靠增加 ABFE 窗口或放宽阈值掩盖（§9 末句，已写进 `remediation`）。
    """
    diagnostics = dict(diagnostics or {})
    required_obs = list(REQUIRED_MEMBRANE_QUALITY_OBSERVABLES)
    required_diag = list(REQUIRED_MEMBRANE_QUALITY_DIAGNOSTICS)
    if require_coion:
        required_obs += list(COION_MEMBRANE_QUALITY_OBSERVABLES)
        required_diag += list(COION_MEMBRANE_QUALITY_DIAGNOSTICS)

    missing_obs = [name for name in required_obs if name not in (observables or {})]
    missing_diag = [name for name in required_diag if diagnostics.get(name) is None]
    if missing_obs or missing_diag:
        raise ValueError(
            "膜质量门缺少必须保存的量 —— fail closed（§9）。"
            f"缺时间序列 {missing_obs}；缺诊断量 {missing_diag}。"
        )

    # 阈值判据表：名字 → (判什么, 阈值, 说明)
    checks: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}
    optional_obs = [
        name
        for name in (APL_PROTEIN_CORRECTED_OBSERVABLE,)
        if name in (observables or {})
    ]
    for name in required_obs + optional_obs:
        times, values = _series_pair(name, observables[name])
        full = linear_drift_per_ns(times, values)
        tail_t, tail_y = _tail_window(times, values, tail_window_ns)
        tail = linear_drift_per_ns(tail_t, tail_y)
        stats[name] = {"full": full, "tail": tail}

    def _add(name, ok, measured, threshold, criterion):
        checks.append(
            {
                "observable": name,
                "criterion": criterion,
                "measured": float(measured),
                "threshold": float(threshold),
                "passed": bool(ok),
            }
        )

    # ---- APL：漂移 ≤ 0.2 %/ns，且与该脂质力场文献值差 ≤ 3% ----
    apl = stats["apl_nm2"]["tail"]
    apl_drift_pct = abs(apl["slope_per_ns"]) / abs(apl["mean"]) * 100.0
    _add(
        "apl_nm2",
        apl_drift_pct <= APL_MAX_DRIFT_PERCENT_PER_NS,
        apl_drift_pct,
        APL_MAX_DRIFT_PERCENT_PER_NS,
        "tail_drift_percent_per_ns",
    )
    # 与文献值比**必须用蛋白横截面校正后的 APL**（见
    # `PROTEIN_CROSS_SECTION_GRID_NM` 的注释）：raw APL 把蛋白占掉的横向
    # 面积也摊给了脂质，含蛋白膜拿它比纯脂文献值必然偏大。
    # 漂移判据仍留在 raw APL 上——那测的是盒面积有没有平衡，掺进蛋白面积的逐帧
    # 噪声只会让它变糊。
    if literature_apl_nm2 is not None:
        corrected = stats.get(APL_PROTEIN_CORRECTED_OBSERVABLE)
        if corrected is not None:
            reference_name = APL_PROTEIN_CORRECTED_OBSERVABLE
            criterion = "deviation_from_literature_percent"
            measured_mean = corrected["tail"]["mean"]
        else:
            # 提取器没给校正序列（老报告 / 手工构造的观测量）——照样判，但把
            # "用的是未校正值"写进 criterion，否则事后分不清这道门当时比的是什么。
            reference_name = "apl_nm2"
            criterion = "deviation_from_literature_percent_uncorrected"
            measured_mean = apl["mean"]
        deviation = abs(measured_mean - float(literature_apl_nm2)) / abs(
            float(literature_apl_nm2)
        ) * 100.0
        _add(
            reference_name,
            deviation <= APL_MAX_DEVIATION_FROM_LITERATURE_PERCENT,
            deviation,
            APL_MAX_DEVIATION_FROM_LITERATURE_PERCENT,
            criterion,
        )

    # ---- 与纯脂文献 APL 的对照：**诊断，不判**（MEM-12，2026-08-02）----
    #
    # `pure_lipid_reference_apl_nm2` 与 `literature_apl_nm2` **刻意是两个不同的字段**：
    # 后者是"要判这道门"的开关，前者只是"记下参考值"。名字分开是为了避免有人
    # 顺手把诊断值填进开关里，从而给含蛋白膜套上一道物理上站不住的门。
    #
    # 为什么含蛋白膜不判与纯脂文献值的 3% 偏差（实测 memtest：校正后 APL 0.5907
    # vs POPC 0.645，低 8.4%）：
    #   * annular lipid 被跨膜蛋白减速并重排，其面积本就不等于体相脂质；
    #   * 蛋白在本体系占约 24% 的横向面积（8.5–9.2 nm² / 36.3 nm²），
    #     "扣掉蛋白面积再除以脂质数"无论怎么定义都残留方法依赖；
    #   * 90 脂小膜片还有有限尺寸效应。
    # 也就是说差百分之几**不构成"体系有问题"的证据**，拿它当门只会制造假阴性。
    #
    # 这道 3% 门应当在**无蛋白 POPC slab** 上启用（memtodolist §8.2 的 lipid slab
    # 工作），那里它才有定义 —— 这不是把门删掉，是把它放到能判的地方。
    if pure_lipid_reference_apl_nm2 is not None:
        corrected = stats.get(APL_PROTEIN_CORRECTED_OBSERVABLE)
        measured_apl = (corrected or {}).get("tail", apl)["mean"]
        reference = float(pure_lipid_reference_apl_nm2)
        stats["apl_vs_pure_lipid_literature"] = {
            "measured_apl_nm2": float(measured_apl),
            "apl_caliber": (
                APL_PROTEIN_CORRECTED_OBSERVABLE if corrected else "apl_nm2"
            ),
            "pure_lipid_reference_apl_nm2": reference,
            "deviation_percent": float(
                abs(measured_apl - reference) / abs(reference) * 100.0
            ),
            "is_gate": False,
            "not_judged_reason": (
                "含蛋白膜与纯脂文献 APL 的偏差不构成体系缺陷证据："
                "annular lipid 被蛋白减速重排、蛋白占本体系约 24% 横向面积、"
                "小膜片有限尺寸效应。这道 3% 门留给无蛋白 POPC slab（§8.2）。"
            ),
        }

    # ---- 双层厚度：漂移 ≤ 0.05 nm / 末段窗口 ----
    thick = stats["bilayer_thickness_nm"]["tail"]
    thick_drift = abs(thick["slope_per_ns"]) * float(tail_window_ns)
    _add(
        "bilayer_thickness_nm",
        thick_drift <= BILAYER_THICKNESS_MAX_DRIFT_NM_PER_TAIL_WINDOW,
        thick_drift,
        BILAYER_THICKNESS_MAX_DRIFT_NM_PER_TAIL_WINDOW,
        "tail_drift_nm_per_window",
    )

    # ---- 跨膜倾角：漂移 ≤ 5° / 末段窗口 ----
    tilt = stats["transmembrane_tilt_deg"]["tail"]
    tilt_drift = abs(tilt["slope_per_ns"]) * float(tail_window_ns)
    _add(
        "transmembrane_tilt_deg",
        tilt_drift <= TRANSMEMBRANE_TILT_MAX_DRIFT_DEG,
        tilt_drift,
        TRANSMEMBRANE_TILT_MAX_DRIFT_DEG,
        "tail_drift_deg_per_window",
    )

    # ---- RMSD 三项：末段均值上限 ----
    for name, limit in (
        ("protein_backbone_rmsd_nm", PROTEIN_BACKBONE_MAX_RMSD_NM),
        ("pocket_rmsd_nm", POCKET_MAX_RMSD_NM),
        ("ligand_heavy_atom_rmsd_nm", LIGAND_HEAVY_ATOM_MAX_RMSD_NM),
    ):
        mean_tail = stats[name]["tail"]["mean"]
        _add(name, mean_tail <= limit, mean_tail, limit, "tail_mean_nm")

    # ---- co-ion 几何（§13.1）----
    # 这几项 §13.1 写的是"**全程** ≥ ..."，所以判的是整条序列的最小值，不是末段、
    # 更不是均值。只看均值会漏掉"末段掉进膜里"这种致命情形（均值仍然漂亮）。
    if require_coion:
        for name, limit in (
            ("coion_ligand_min_image_distance_nm", COION_LIGAND_MIN_IMAGE_RUNTIME_NM),
            ("coion_protein_heavy_atom_distance_nm", COION_PROTEIN_HEAVY_ATOM_MIN_NM),
            ("coion_abs_z_from_midplane_nm", COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM),
            ("coion_nearest_phosphorus_distance_nm", COION_NEAREST_PHOSPHORUS_MIN_NM),
        ):
            _, values = _series_pair(name, observables[name])
            worst = float(np.min(np.asarray(values, dtype=float)))
            _add(name, worst >= limit, worst, limit, "full_series_min_nm_lower_bound")

    # ---- §9：用脂质横向弛豫时间尺度**论证**预平衡时长（诊断，不是门）----
    #
    # ⚠️ **这一项刻意不进 `checks`**（MEM-11，2026-08-02）。理由，按重要性排序：
    #
    # 1. **§9 原文只要求"记录…并用它论证"**，没有要求"≥ 1 倍"。那个倍数
    #    （`MEMBRANE_EQUILIBRATION_MIN_RELAXATION_MULTIPLE`）是本实现自己加的，
    #    当时的注释里就写着"§13 未给此倍数"。
    # 2. **常规膜蛋白平衡的判据不是这个**：看的是 APL / 膜厚 / 序参量 / RMSD 走平
    #    （上面那些 check 就是干这个的，本体系实测余量 2–6 倍），
    #    而不是要求脂质完成一次横向扩散位移。
    # 3. **它是方法依赖量**：τ 由 MSD 拟合窗口、体系拥挤度、盒尺寸共同决定
    #    （见 `_lipid_lateral_relaxation_timescale_ns` 的 docstring：同一条 100 ns
    #    轨迹在旧估计器下 τ 从 30 跳到 11 ns）。用一个方法依赖量当硬门，
    #    会出现"体系没问题、方法波动把它挡住"的假阴性。
    #
    # 与本仓库把 ESS `min_occupancy_normalized` 退役为 diagnostics-only 是同一先例
    # （`docs/TODO.md` TEST-GATE-01）。
    #
    # 🚫 **降级不等于不记录**：下面照样把 τ、比值、与纯 POPC 文献 D 的对照全部落盘，
    #    "永不弛豫的膜"会在这里以极大的 τ 与 < 1 的比值被如实报出（有专门测试钉住）。
    # 🚫 **不得为了让某次运行通过而把它重新塞回 `checks`。** 要重新当门，必须先给出
    #    "这个 τ 估计在跨体系/跨轨迹长度上稳定"的证据，并在 §13 里写下倍数的依据。
    relax_ns = float(diagnostics["lipid_lateral_relaxation_timescale_ns"])
    equil_ns = float(diagnostics["equilibration_length_ns"])
    stats["equilibration_vs_relaxation"] = {
        "equilibration_length_ns": equil_ns,
        "lipid_lateral_relaxation_timescale_ns": relax_ns,
        "ratio_equilibration_over_relaxation": (
            float(equil_ns / relax_ns) if relax_ns > 0 else float("inf")
        ),
        "is_gate": False,
        "retired_reason": (
            "§9 只要求记录并论证，未给倍数；常规膜平衡判据是观测量走平；"
            "τ 是方法依赖量（拟合窗口/拥挤度/盒尺寸），当硬门会产生假阴性。"
            "退役为 diagnostics-only，不得为让运行变绿而塞回 checks。"
        ),
        "lipid_lateral_diffusion": diagnostics.get("lipid_lateral_diffusion"),
    }

    # ---- §9：膜与周期镜像的异常接触必须为 0 ----
    contacts = int(diagnostics["membrane_periodic_image_contacts"])
    checks.append(
        {
            "observable": "membrane_periodic_image_contacts",
            "criterion": "must_be_zero",
            "measured": float(contacts),
            "threshold": 0.0,
            "passed": contacts == 0,
        }
    )

    failed = [c for c in checks if not c["passed"]]
    return {
        "protocol_version": MEMBRANE_QUALITY_GATE_PROTOCOL_VERSION,
        "thresholds_version": ACCEPTANCE_THRESHOLDS_VERSION,
        "tail_window_ns": float(tail_window_ns),
        "passed": not failed,
        "checks": checks,
        "failed_checks": [c["observable"] for c in failed],
        "statistics": stats,
        "diagnostics": diagnostics,
        "coion_required": bool(require_coion),
        "remediation": (
            "质量门失败时回到膜体系平衡（延长预平衡 / 检查建系组成与叶片数），"
            "不允许靠增加 ABFE 窗口、放宽阈值或缩短末段窗口掩盖（§9 末句）。"
        ),
    }


MEMBRANE_QUALITY_GATE_REPORT_FILENAME = "membrane_quality_gate.json"


def run_membrane_quality_gate(
    traj_path: str,
    openmm_topology,
    membrane_quality_inputs: Dict[str, Any],
    *,
    mode: str,
    normal_axis: str = "z",
    ligand_indices=None,
    output_dir: Optional[str] = None,
    frame_interval_ps: Optional[float] = None,
    log=None,
) -> Dict[str, Any]:
    """轨迹 → §9 观测量 → 判定 → 落盘，**唯一实现**（§9 / §6.2）。

    ## 为什么是模块级函数而不是 `ABFEPipeline` 的方法

    原先这段接线只存在于 `ABFEPipeline._evaluate_membrane_quality_gate_after_equilibration`
    里，于是"验证质量门"唯一的办法是**重烧一遍预平衡**（膜体系 10–100 ns，
    5 h 起）。§0.5.7 已经因为"离线重建与生产路径不一致"白花过好几轮，
    所以这里不允许再出现第二份接线：`ABFEPipeline` 与
    `tools/diagnostics/evaluate_membrane_quality_gate.py` 都调这一个函数。

    ## 两种模式（`mode`）

    * `enforce`（默认）——门没过就 raise。这是 §9 末句的原意：
      "质量门失败时回到膜体系平衡，不允许靠增加 ABFE 窗口掩盖。"
    * `advisory`——照样计算、照样落盘报告、失败照样大声 WARNING，但**不阻断**。
      用于"先把管路跑通"的探索阶段。

    为什么提供 advisory 而不是让人注释掉调用：门被注释掉就**没有记录**，
    事后无从知道当时到底过没过。advisory 下报告仍完整落盘、模式进 provenance。
    ⚠️ advisory **不是**生产资格；要报出的 ΔG_bind 必须在 `enforce` 下通过。

    需要的口袋定义 / co-ion 索引由 `membrane_quality_inputs` 显式给出，
    不做运行时推断——同一体系两次跑必须用同一个口袋定义（MEM-00c 那类漂移的教训）。

    `frame_interval_ps` 默认取 `pre_equilibration_frame_interval_ps()`（= 生产设置
    10000 步 × 2 fs = 20 ps/帧）。**必须显式重建时间轴**：mdtraj 读 DCD 给的
    `traj.time` 是整数帧号，直接用会让 §9 的时间轴错 20 倍（详见提取器的 docstring）。
    """
    import mdtraj as md

    emit = log if callable(log) else (lambda message: None)
    mode = resolve_membrane_quality_gate_mode(mode)
    advisory = mode == MEMBRANE_QUALITY_GATE_MODE_ADVISORY

    def _write(report: Dict[str, Any], observables) -> Optional[str]:
        if not output_dir:
            return None
        summary_path = os.path.join(output_dir, MEMBRANE_QUALITY_GATE_REPORT_FILENAME)
        payload: Dict[str, Any] = {"report": report}
        if observables is not None:
            payload["observables"] = observables
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        emit(f"  [OK] 膜质量门摘要已保存: {summary_path}")
        return summary_path

    def _blocked(message: str, observables=None) -> Dict[str, Any]:
        """enforce 下 raise；advisory 下大声记录并落一个"未评估"的报告。

        `observables` 已经算出来时**一定要一起落盘**：观测量是烧了几小时 GPU 才有的，
        判不了门不等于这些数字没价值（例如"跨度覆盖不了末段窗口"时，APL / 膜厚
        的实测值仍然是判断要不要延长平衡的唯一依据）。
        """
        if not advisory:
            raise RuntimeError(message)
        emit(f"  [WARN] [膜质量门 advisory] 未能完成评估：{message}")
        report = {
            "protocol_version": MEMBRANE_QUALITY_GATE_PROTOCOL_VERSION,
            "mode": MEMBRANE_QUALITY_GATE_MODE_ADVISORY,
            "evaluated": False,
            "passed": None,
            "blocked_reason": message,
        }
        _write(report, observables=observables)
        return report

    inputs = dict(membrane_quality_inputs or {})
    pocket = inputs.get("pocket_atom_indices")
    if not pocket:
        return _blocked(
            "膜体系必须在 membrane_quality_inputs 里给出 pocket_atom_indices："
            "口袋定义直接决定 §9 的 pocket_rmsd 这一道门，不接受运行时推断。"
        )
    ligand_resname = inputs.get("ligand_resname")
    if not ligand_resname:
        if not ligand_indices:
            return _blocked(
                "既没有 ligand_resname，也没有 ligand_indices，无法定位配体。"
                "请在膜输入声明里填 ligand_resname。"
            )
        ligand_resname = openmm_topology.atom(int(ligand_indices[0])).residue.name

    emit(f"\n[膜质量门 · {mode}] 读取预平衡轨迹并计算 §9 观测量...")
    try:
        traj = md.load(traj_path, top=md.Topology.from_openmm(openmm_topology))
        # [MEM-17 已移除] 曾经在这里对账「帧数 == n_steps // reporter_interval」，
        # resume 追加的重复帧会让它 fail closed。判据本身算得没错（重复帧是真的），
        # 但它拦的是主线而不是根因：根因在 `abfe_pipeline.pre_equilibrate` 里
        # `DCDReporter(append=True)` 不截断到 checkpoint 帧边界。
        # 2026-08-03 用户决定删掉这道对账，**不要再加回来**；要修就修上游截断。
        observables, diagnostics = membrane_observables_from_trajectory(
            traj,
            ligand_resname=ligand_resname,
            normal_axis=normal_axis,
            pocket_atom_indices=[int(i) for i in pocket],
            coion_atom_index=inputs.get("coion_atom_index"),
            equilibration_length_ns=inputs.get("equilibration_length_ns"),
            # 身份以 `.top` 组成为准：脂质按分子分叶、水/离子/蛋白用权威原子集合，
            # 不靠残基名（TP3 / Na+ / Cl- / HID / NTRP 都会被残基名判据漏掉）。
            composition=inputs.get("composition"),
            # 时间轴显式重建，不吃 mdtraj 给 DCD 的整数帧号。
            frame_interval_ps=(
                pre_equilibration_frame_interval_ps()
                if frame_interval_ps is None
                else frame_interval_ps
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return _blocked(f"{type(exc).__name__}: {exc}")

    try:
        report = evaluate_membrane_quality_gate(
            observables,
            diagnostics,
            literature_apl_nm2=inputs.get("literature_apl_nm2"),
            require_coion=inputs.get("coion_atom_index") is not None,
            # 诊断专用参考值（不判门），见 evaluate_membrane_quality_gate 里的说明。
            pure_lipid_reference_apl_nm2=inputs.get("pure_lipid_reference_apl_nm2"),
        )
    except Exception as exc:  # noqa: BLE001
        # 判定层报错（最典型：轨迹跨度覆盖不了 20 ns 末段窗口）。观测量已经算出来了，
        # 连同"为什么判不了"一起落盘 —— 那些数字是决定要不要延长平衡的依据。
        return _blocked(f"{type(exc).__name__}: {exc}", observables=observables)
    report["mode"] = mode
    report["evaluated"] = True
    summary_path = _write(report, observables)

    for check in report["checks"]:
        flag = "[OK]" if check["passed"] else "[ERR]"
        emit(
            f"    {flag} {check['observable']} [{check['criterion']}] "
            f"{check['measured']:.6g} vs 阈值 {check['threshold']:.6g}"
        )

    if not report["passed"]:
        message = (
            f"膜质量门未通过，失败项：{report['failed_checks']}。"
            f"{report['remediation']}"
            + (f" 详情见 {summary_path}。" if summary_path else "")
        )
        if not advisory:
            raise RuntimeError(message)
        emit(
            f"  [WARN] [膜质量门 advisory] {message}\n"
            "     以 advisory 模式放行继续 —— **这不是生产资格**，"
            "要报出的 ΔG_bind 必须在 enforce 下通过。"
        )
        return report
    emit(f"  [OK] 膜质量门通过（模式 {mode}）")
    return report


# ============================================================================
# §9 观测量提取器：轨迹 → evaluate_membrane_quality_gate() 的输入
#
# 判定层（evaluate_membrane_quality_gate）此前是就绪但拿不到输入的。这一节负责把
# 一条膜轨迹算成它要的时间序列与诊断量。
#
# 设计原则：**宁可 fail closed 也不猜**。
#   - 序参量需要脂质尾链的成键关系。拓扑里没有键就直接报错，**不**用距离阈值
#     冒充共价键——仓库已经在 ATT-11 上吃过这个亏
#     （`GeometricRestraintEstimator._build_bond_adjacency` 的 docstring 记着）。
#   - 口袋定义、co-ion 索引这类无法从轨迹自动可靠推断的，一律要求调用方显式给，
#     给不出就报错，不用"离配体最近的 N 个残基"之类的默契。
#
# ⚠️ 本节的数值口径需要用真实膜体系复核一遍再上生产：APL / 膜厚 / 盒序列是
# 直接几何量，风险低；序参量（用 C–C 键向量而非 C–H，见下）、口袋异常水、
# 横向弛豫时间尺度都是"等价结构指标"，量级对不对必须实测对照文献值。
# ============================================================================


def _require_mdtraj(what: str):
    if not HAS_MDTRAJ:
        raise RuntimeError(
            f"{what} 需要 mdtraj，但当前环境没装。请在 openmm_dev 环境里运行。"
        )
    import mdtraj

    return mdtraj


def _lipid_residue_indices(topology) -> List[int]:
    return [
        int(residue.index)
        for residue in topology.residues
        if str(residue.name).strip().upper() in KNOWN_LIPID_RESIDUE_NAMES
    ]


def _lipid_head_atom_indices_by_residue(topology) -> Dict[int, int]:
    """每个脂质残基的头基参考原子 index（mdtraj topology）。"""
    result: Dict[int, int] = {}
    for residue in topology.residues:
        if str(residue.name).strip().upper() not in KNOWN_LIPID_RESIDUE_NAMES:
            continue
        by_name = {str(a.name).strip().upper(): int(a.index) for a in residue.atoms}
        for candidate in (
            LIPID_HEAD_REFERENCE_ATOM_NAMES + LIPID_HEAD_FALLBACK_ATOM_NAMES
        ):
            if candidate in by_name:
                result[int(residue.index)] = by_name[candidate]
                break
    return result


def _lipid_carbon_carbon_bond_pairs(topology) -> List[Tuple[int, int]]:
    """脂质残基内部的 C–C 成键对，用于序参量。

    只用**拓扑里真实的键**。没有键信息时返回空列表，由调用方 fail closed——
    不用距离阈值冒充共价键。
    """
    lipid_residues = set(_lipid_residue_indices(topology))
    pairs: List[Tuple[int, int]] = []
    for atom_a, atom_b in topology.bonds:
        if int(atom_a.residue.index) not in lipid_residues:
            continue
        if int(atom_b.residue.index) != int(atom_a.residue.index):
            continue
        sym_a = getattr(atom_a.element, "symbol", "") or ""
        sym_b = getattr(atom_b.element, "symbol", "") or ""
        if sym_a.upper() == "C" and sym_b.upper() == "C":
            pairs.append((int(atom_a.index), int(atom_b.index)))
    return pairs


def _series(times_ns, values) -> Dict[str, List[float]]:
    return {
        "times_ns": [float(t) for t in np.asarray(times_ns, dtype=float)],
        "values": [float(v) for v in np.asarray(values, dtype=float)],
    }


def membrane_observables_from_trajectory(
    traj,
    ligand_resname: Optional[str] = None,
    normal_axis: str = "z",
    pocket_atom_indices: Optional[List[int]] = None,
    coion_atom_index: Optional[int] = None,
    equilibration_length_ns: Optional[float] = None,
    composition: Optional[Dict[str, Any]] = None,
    frame_interval_ps: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """把一条膜轨迹算成 `evaluate_membrane_quality_gate()` 需要的观测量与诊断量。

    `traj` 是一个已载入的 mdtraj Trajectory（带 unitcell 与键信息）。
    返回 `(observables, diagnostics)`，可直接喂给判定层。

    ## 时间轴：DCD 必须显式给 `frame_interval_ps`

    §9 的判据全部定义在真实时间轴上（"末段 ≥ 20 ns"、"漂移 ≤ x/ns"），所以时间轴
    错一个倍数，两道门会往**相反**方向坏掉：末段窗口变得过严，而"预平衡 ≥ 一个脂质
    横向弛豫时间"变得过松（MSD 拟合出的 D 被同一个倍数放大）。

    ⚠️ **mdtraj 读 DCD 时不传播真实步长**：`DCDTrajectoryFile.read_as_traj` 给出的
    `traj.time` 是**整数帧号** `[0, 1, 2, …]`。实测 memtest 那条 10 ns / 500 帧
    （10000 步 × 2 fs = 20 ps/帧）的轨迹，`traj.time` 就是 `[0…499]`，
    于是时间轴被当成 0.499 ns —— 比真实值小 20 倍。
    原先这里只校验"存在且单调递增"，帧号完全满足，所以这条守卫对 DCD 是 fail-open。

    因此：
      * 传了 `frame_interval_ps` → 时间轴由它构造（`i × interval`），忽略 `traj.time`。
        这是生产路径的做法，数值来自 `pre_equilibration_frame_interval_ps()`，
        与写轨迹的 reporter/integrator 共用同一组常量。
      * 没传 → 用 `traj.time`，但**整数 dtype 一律拒绝**（那就是上面那个帧号签名）。
    实际采用了哪一条会写进 `diagnostics["time_axis_source"]`，事后查得出来。
    """
    md = _require_mdtraj("膜质量门观测量提取")
    axis = {"x": 0, "y": 1, "z": 2}[str(normal_axis).strip().lower()]
    lateral_axes = [i for i in (0, 1, 2) if i != axis]
    top = traj.topology

    if traj.unitcell_lengths is None:
        raise ValueError(
            "轨迹没有 unitcell 信息，无法计算 APL / 盒序列。\n"
            "    `app.DCDFile` 写每帧 unitcell 时读的是**topology**的盒矢量"
            "（`dcdfile.py:155`：`boxVectors = self._topology.getPeriodicBoxVectors()`），"
            "为 None 就整段不写、header 的 boxFlag 也是 0。\n"
            "    所以这不是轨迹格式问题，而是**写轨迹时那个 topology 没有盒矢量**。"
            "最常见原因：从 `.top` 重建拓扑时没传 `periodicBoxVectors`"
            "（`GromacsTopFile` 只在显式传参时才设盒）。\n"
            "    修法：加载/重建拓扑后 `topology.setPeriodicBoxVectors(box)`，然后重跑预平衡。"
        )
    if frame_interval_ps is not None:
        interval_ps = float(frame_interval_ps)
        if not np.isfinite(interval_ps) or interval_ps <= 0.0:
            raise ValueError(f"frame_interval_ps 必须是正有限值，收到 {frame_interval_ps!r}")
        times = np.arange(traj.n_frames, dtype=float) * interval_ps
        time_axis_source = "declared_frame_interval"
    else:
        raw_time = np.asarray(traj.time)
        if np.issubdtype(raw_time.dtype, np.integer):
            raise ValueError(
                "轨迹的 time 数组是**整数**，这正是 mdtraj 读 DCD 时给出的**帧号**"
                f"（实测形如 [0, 1, 2, …]，本条 {raw_time.size} 帧）。"
                "§9 的判据定义在真实时间轴上（末段 ≥ 20 ns、漂移 ≤ x/ns），"
                "用帧号冒充 ps 会让末段窗口过严、同时让"
                "「预平衡 ≥ 一个脂质横向弛豫时间」过松（D 被同一倍数放大）。\n"
                "    修法：传 `frame_interval_ps`（= reporter 保存间隔 × integrator 步长，"
                "生产路径用 `pre_equilibration_frame_interval_ps()`），不要改判据窗口。"
            )
        times = raw_time.astype(float)
        time_axis_source = "trajectory_time_field"
        interval_ps = None
    if times.size != traj.n_frames or not np.all(np.diff(times) > 0):
        raise ValueError(
            "轨迹的 time 数组缺失或非单调递增。§9 的判据定义在真实时间轴上"
            "（末段 ≥ 20 ns），不允许用帧号冒充 ns。"
        )
    times_ns = times / 1000.0

    lengths = np.asarray(traj.unitcell_lengths, dtype=float)  # (n_frames, 3) nm
    lateral_area = lengths[:, lateral_axes[0]] * lengths[:, lateral_axes[1]]
    normal_length = lengths[:, axis]
    volume = lateral_area * normal_length

    # ---- 叶片划分（用第 0 帧，身份此后固定，不逐帧重选）----
    # 有 `.top` 组成时**按分子**分叶。Amber Lipid21 把一个 POPC 拆成
    # PA + PC + OL 三个残基，按残基会数出 3 倍脂质数、APL 错 3 倍，且尾链残基
    # 没有磷原子会直接报"找不到头基参考原子"（实测 memtest 体系正是如此）。
    #
    # ⚠️ 两条分支**必须**收敛到同一个 `head_units`：`(单元标签, 头基原子 index)`。
    # 下游（`head_indices`、`leaflet_composition`）只许从它派生，不许再有分支专属变量。
    # 原先分子分支产 `head_list`、残基分支产 `head_by_residue`，而下面的
    # `leaflet_composition` 只读后者 —— 分子分支下它从未绑定，于是真实膜体系上
    # 整个 §9 质量门崩在 `UnboundLocalError`（memtest 2026-07-31 与 08-02 各一次，
    # 报告落成 `{"evaluated": false}`）。根因是"身份口径改成按分子了，但有个消费点
    # 没跟着改"，与 §0.5.5 记的"同一件事多个入口，补一个漏一片"同源。
    lipid_molecules = (composition or {}).get("molecules_by_role", {}).get("lipid")
    head_candidates = LIPID_HEAD_REFERENCE_ATOM_NAMES + LIPID_HEAD_FALLBACK_ATOM_NAMES
    head_units: List[Tuple[str, int]] = []
    if lipid_molecules:
        all_atoms = _topology_atoms(top)
        missing_units: List[Dict[str, Any]] = []
        for entry in lipid_molecules:
            start, stop = int(entry["start"]), int(entry["stop"])
            by_name = {}
            for atom in all_atoms[start:stop]:
                by_name.setdefault(str(atom.name).strip().upper(), int(atom.index))
            ref = next((by_name[c] for c in head_candidates if c in by_name), None)
            if ref is None:
                missing_units.append({"atom_range": [start, stop]})
            else:
                # 标签取 moleculetype 名（`POPC`），**不是**构成残基名（PA/PC/OL）——
                # 用残基名会让 leaflet_composition 又数出 3 倍脂质数（§0.5.4 那个坑）。
                head_units.append(
                    (str(entry["molecule_name"]).strip().upper(), ref)
                )
        if missing_units:
            raise ValueError(
                f"{len(missing_units)} 个脂质分子找不到头基参考原子"
                f"（示例原子区间 {missing_units[:3]}）。请补充原子命名，不要静默不计入。"
            )
        lipid_unit_label = "molecule"
    else:
        head_by_residue = _lipid_head_atom_indices_by_residue(top)
        lipid_residues = _lipid_residue_indices(top)
        missing_heads = sorted(set(lipid_residues) - set(head_by_residue))
        if not lipid_residues:
            raise ValueError("轨迹里找不到任何已知脂质残基，这不是膜体系。")
        if missing_heads:
            raise ValueError(
                f"{len(missing_heads)} 个脂质残基找不到头基参考原子"
                f"（残基 index 示例 {missing_heads[:5]}）。请补充原子命名，不要让它们静默不计入。"
            )
        head_units = [
            (str(top.residue(r).name).strip().upper(), head_by_residue[r])
            for r in sorted(head_by_residue)
        ]
        lipid_unit_label = "residue"
    head_indices = np.asarray([idx for _, idx in head_units], dtype=int)
    if head_indices.size == 0:
        raise ValueError("找不到任何脂质头基参考原子，这不是膜体系。")

    # A wrapped first frame can straddle the periodic boundary (e.g. one
    # leaflet at z≈0 and the other at z≈L).  Choosing the arithmetic mean of
    # those raw coordinates would split a leaflet and can make the quality gate
    # reject an otherwise healthy bilayer.  Cut the circular coordinate at its
    # largest empty arc and use that common image for the initial leaflet
    # labels; the same per-atom image offsets are then applied to the unwrapped
    # trajectory below.
    head_z_wrapped = np.asarray(traj.xyz[0][head_indices][:, axis], dtype=float)
    first_box_length = float(lengths[0, axis])
    if (
        head_z_wrapped.ndim != 1
        or not np.all(np.isfinite(head_z_wrapped))
        or not np.isfinite(first_box_length)
        or first_box_length <= 0.0
    ):
        raise ValueError("脂质头基首帧法向坐标或盒长非法")
    z_mod = np.mod(head_z_wrapped, first_box_length)
    order = np.argsort(z_mod)
    sorted_z = z_mod[order]
    gaps = np.diff(np.concatenate([sorted_z, sorted_z[:1] + first_box_length]))
    cut = float(sorted_z[(int(np.argmax(gaps)) + 1) % sorted_z.size])
    head_z_cluster = z_mod - cut
    head_z_cluster[head_z_cluster < 0.0] += first_box_length
    # Restore the coordinate origin: only integer box-image shifts may be
    # applied to heads alone. Water/ions/protein retain the trajectory origin.
    head_z_cluster += cut
    head_coords_frame0 = head_z_cluster
    midplane0 = float(np.mean(head_coords_frame0))
    upper_mask = head_coords_frame0 > midplane0
    n_upper = int(np.count_nonzero(upper_mask))
    n_lower = int(head_indices.size - n_upper)
    if n_upper == 0 or n_lower == 0:
        raise ValueError(
            f"叶片划分退化：上叶 {n_upper} / 下叶 {n_lower}。"
            "第 0 帧可能没有居中到膜中面，或者体系不是双层。"
        )

    # ---- 逐帧：中面、APL、膜厚 ----
    # Use a continuous head trajectory.  With wrapped coordinates a rigid
    # membrane translated through z=0 would split one leaflet across the two
    # ends of the box and make the arithmetic means/thickness meaningless.
    head_xyz_unwrapped = _unwrap_trajectory_subset(traj, head_indices)
    # Align every head's continuous trajectory to the common circular image
    # selected above.  The offsets are constant per atom and therefore do not
    # alter any frame-to-frame displacement/MSD.
    head_xyz_unwrapped[:, :, axis] += (
        head_z_cluster - head_xyz_unwrapped[0, :, axis]
    )[None, :]
    head_z_all = head_xyz_unwrapped[:, :, axis]  # (n_frames, n_heads)
    midplane = head_z_all.mean(axis=1)
    upper_z = head_z_all[:, upper_mask].mean(axis=1)
    lower_z = head_z_all[:, ~upper_mask].mean(axis=1)
    thickness = upper_z - lower_z
    # 每叶面积每脂：横向面积 / 该叶脂质数。上下叶不等时按各自计数分别算再取均值，
    # 不用"总数/2"（§9：叶片数不是随手对半分）。
    apl = 0.5 * (lateral_area / n_upper + lateral_area / n_lower)

    # ---- 序参量：脂质残基内 C–C 键向量相对膜法向 ----
    # ⚠️ 这是 §9 允许的"等价结构指标"，**不是** S_CD。S_CD 需要 C–H 向量，
    # 而 Amber/GROMACS 拓扑里氢是显式的但尾链 C–H 配对需要额外规则；
    # C–C 键向量序参量与 S_CD 同向、量级不同，跨力场比较必须用同一定义。
    cc_pairs = _lipid_carbon_carbon_bond_pairs(top)
    if not cc_pairs:
        raise ValueError(
            "拓扑里没有脂质残基内部的 C–C 键，无法计算尾链序参量。"
            "请提供带键信息的拓扑（GromacsTopFile 产出的 OpenMM topology 有键）；"
            "**不会**用距离阈值冒充共价键（见 ATT-11 的教训）。"
        )
    pair_a = np.asarray([p[0] for p in cc_pairs], dtype=int)
    pair_b = np.asarray([p[1] for p in cc_pairs], dtype=int)
    cc_atom_indices = np.unique(np.concatenate([pair_a, pair_b]))
    cc_unwrapped = _unwrap_trajectory_subset(traj, cc_atom_indices)
    cc_lookup = {int(atom): i for i, atom in enumerate(cc_atom_indices)}
    vectors = np.asarray(
        [
            cc_unwrapped[:, cc_lookup[int(b)], :]
            - cc_unwrapped[:, cc_lookup[int(a)], :]
            for a, b in zip(pair_a, pair_b)
        ],
        dtype=np.float64,
    ).transpose(1, 0, 2)
    norms = np.linalg.norm(vectors, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_theta = vectors[:, :, axis] / norms
    order_parameter = 0.5 * (3.0 * np.nanmean(cos_theta**2, axis=1) - 1.0)

    # ---- RMSD 三项 + 倾角 ----
    ligand_from_composition = (
        (composition or {}).get("atom_indices_by_role", {}).get("ligand")
    )
    if ligand_from_composition:
        ligand_set = set(int(i) for i in ligand_from_composition)
        ligand_heavy = np.asarray(
            sorted(
                int(atom.index)
                for atom in _topology_atoms(top)
                if int(atom.index) in ligand_set
                and (getattr(atom.element, "symbol", "") or "").upper() != "H"
            ),
            dtype=int,
        )
    else:
        if not ligand_resname:
            raise ValueError("未提供 composition 时必须给 ligand_resname。")
        ligand_heavy = top.select(f"resname {ligand_resname} and not element H")
    if ligand_heavy.size == 0:
        raise ValueError(f"找不到配体 {ligand_resname!r} 的重原子。")

    # mdtraj 的 `protein` 关键字按残基名查表，而它的表里 **没有** HID / HIE /
    # ASH / CYX / NTRP / CCYS（只有 HIP），实测 memtest 体系会静默漏掉 85 个原子。
    # 有 `.top` 组成时用组成给的蛋白原子集合，并只取骨架原子名。
    protein_atoms_from_composition = (
        (composition or {}).get("atom_indices_by_role", {}).get("protein")
    )
    protein_heavy = np.asarray([], dtype=int)
    if protein_atoms_from_composition:
        protein_set = set(int(i) for i in protein_atoms_from_composition)
        backbone_names = {"N", "CA", "C", "O"}
        protein_backbone = np.asarray(
            sorted(
                int(atom.index)
                for atom in _topology_atoms(top)
                if int(atom.index) in protein_set
                and str(atom.name).strip().upper() in backbone_names
            ),
            dtype=int,
        )
        # 蛋白横截面用**全部重原子**（不只骨架）：占掉脂质面积的是侧链外表面。
        protein_heavy = np.asarray(
            sorted(
                int(atom.index)
                for atom in _topology_atoms(top)
                if int(atom.index) in protein_set
                and (getattr(atom.element, "symbol", "") or "").upper() != "H"
            ),
            dtype=int,
        )
    else:
        protein_backbone = top.select("protein and backbone")
        # 没有组成兜底时，至少不能静默少选：把"既不是水/脂质/离子/配体、
        # 又没被 mdtraj 认成蛋白"的残基报出来。
        omitted = sorted(
            {
                str(res.name).strip().upper()
                for res in top.residues
                if str(res.name).strip().upper() not in WATER_MOLECULE_NAMES
                and str(res.name).strip().upper() not in KNOWN_LIPID_RESIDUE_NAMES
                and _normalize_ion_name(res.name) not in MONOATOMIC_ION_NAMES
                and str(res.name).strip().upper() != str(ligand_resname or "").upper()
                and not res.is_protein
            }
        )
        if omitted:
            raise ValueError(
                f"这些残基既不是水/脂质/离子/配体，也没被 mdtraj 认成蛋白：{omitted}。"
                "mdtraj 的 protein 残基名表不含 HID/HIE/ASH/CYX/N-/C- 端变体，"
                "直接用它会静默少选骨架原子。请传入 composition"
                "（`classify_system_composition()` 的产出）以使用权威的蛋白原子集合。"
            )
        protein_heavy = np.asarray(
            sorted(int(i) for i in top.select("protein and not element H")), dtype=int
        )
    if protein_backbone.size == 0:
        raise ValueError("找不到蛋白骨架原子，无法计算骨架 RMSD 与跨膜倾角。")
    if pocket_atom_indices is None:
        raise ValueError(
            "必须显式提供 pocket_atom_indices。口袋定义直接决定 pocket_rmsd 这一道门，"
            "不接受「离配体最近的若干残基」之类的运行时默契——那会让同一体系两次跑"
            "用不同的口袋定义（这正是 MEM-00c 那类静默漂移的成因）。"
        )
    pocket_indices = np.asarray(sorted(int(i) for i in pocket_atom_indices), dtype=int)
    if pocket_indices.size == 0:
        raise ValueError("pocket_atom_indices 为空。")

    # ---- RMSD 三项：在**子集副本**上做骨架对齐，绝不动传入的 traj（MEM-10）----
    #
    # ⚠️ `mdtraj.Trajectory.superpose()` **原地修改 `traj.xyz` 并返回 self**。
    # 原先这里写的是 `aligned = traj.superpose(traj, 0, atom_indices=protein_backbone)`，
    # 于是这一行之后所有读 `traj.xyz` 的量都在用"对齐到蛋白骨架"的坐标，而
    # `midplane` / `upper_z` / `lower_z` 是在上面、对齐**之前**算的 —— 两者不在同一
    # 坐标系。实测后果（memtest 100 ns，2026-08-02）：
    #   * 脂质横向弛豫时间尺度 τ 从 11.57 ns 被放大到 **139.36 ns**（12 倍），
    #     直接把 §9 质量门判失败（门里报的正是 139.362）；
    #   * 跨膜倾角在对齐帧里测 → 漂移被系统性压掉；
    #   * 蛋白横截面 / 校正后 APL、疏水核内水、水层间隙、密度分布全部口径错配。
    #
    # 而那次 superpose 对它本来要服务的三个 RMSD **毫无作用**：
    # `md.rmsd(..., atom_indices=X)` 内部会自己在 X 上重新做最优拟合，所以先对齐
    # 不改变返回值（实测 pocket 0.069400 / 0.069400、ligand 0.050201 / 0.050201）。
    # 也就是说它是**纯有害**的一行。
    #
    # 现在只对 backbone ∪ pocket ∪ ligand 这个子集建副本（`atom_slice` 返回新对象），
    # 旋转矩阵只由骨架坐标决定，所以骨架 RMSD 与改前数值等价；内存约为全轨迹的
    # 1/20（实测 1128 + 148 + 41 vs 45354 原子）。
    rmsd_subset = np.unique(
        np.concatenate([protein_backbone, pocket_indices, ligand_heavy])
    )
    _subset_position = {int(g): i for i, g in enumerate(rmsd_subset)}
    _backbone_in_subset = np.asarray(
        [_subset_position[int(i)] for i in protein_backbone], dtype=int
    )
    _pocket_in_subset = np.asarray(
        [_subset_position[int(i)] for i in pocket_indices], dtype=int
    )
    _ligand_in_subset = np.asarray(
        [_subset_position[int(i)] for i in ligand_heavy], dtype=int
    )
    ref = traj.atom_slice(rmsd_subset)
    backbone_rmsd = md.rmsd(ref, ref, 0, atom_indices=_backbone_in_subset)
    ref.superpose(ref, 0, atom_indices=_backbone_in_subset)  # 只改副本

    # 口袋 / 配体是 **pose 漂移**（§9："口袋 RMSD、配体 RMSD/关键相互作用"），
    # 所以对齐蛋白骨架之后**不再重新拟合**，直接量位移（MEM-13）。
    # 原先用 `md.rmsd(..., atom_indices=pocket)`，它会在口袋/配体自身上再做一次
    # 最优拟合 —— 测到的是内部构象变化，而不是相对受体的 pose 漂移。
    # 实测差别（memtest 100 ns 末段 20 ns）：不重拟合 0.0857 / 0.0833 nm，
    # 重拟合 0.0760 / 0.0493 nm（配体差 1.7 倍）。阈值 0.20 / 0.25 不变。
    def _pose_drift_nm(subset_indices):
        delta = ref.xyz[:, subset_indices, :] - ref.xyz[0, subset_indices, :][None]
        return np.sqrt(np.mean(np.sum(delta**2, axis=2), axis=1))

    pocket_rmsd = _pose_drift_nm(_pocket_in_subset)
    ligand_rmsd = _pose_drift_nm(_ligand_in_subset)

    # 跨膜倾角：蛋白骨架坐标的第一主轴与膜法向的夹角（0–90°）。
    tilt_deg = np.empty(traj.n_frames, dtype=float)
    protein_unwrapped = _unwrap_trajectory_subset(traj, protein_backbone)
    normal_vector = np.zeros(3)
    normal_vector[axis] = 1.0
    for frame in range(traj.n_frames):
        # Build a continuous local protein frame so a whole protein crossing a
        # periodic boundary cannot spuriously inflate the principal-axis tilt.
        coords = protein_unwrapped[frame]
        centered = coords - coords.mean(axis=0)
        # 最大奇异值对应的右奇异向量 = 第一主轴。
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        principal = vt[0]
        cosine = abs(float(np.dot(principal, normal_vector)))
        tilt_deg[frame] = math.degrees(math.acos(min(1.0, max(0.0, cosine))))

    # ---- APL 的蛋白横截面校正（§13.3 的绝对值门只有校正后才有意义）----
    # raw APL 把蛋白占掉的横向面积也摊到脂质头上，含蛋白膜因此系统性偏大
    # （实测 memtest 0.826 vs POPC 纯脂文献 0.645）。校正 = 先从横向面积里扣掉
    # 该叶片 slab 内的蛋白占据面积，再除以该叶脂质数。
    # 脂质原子集合：组成优先，名字兜底。下面的"膜与周期镜像水层间隙"复用同一集合。
    lipid_from_composition = (
        (composition or {}).get("atom_indices_by_role", {}).get("lipid")
    )
    if lipid_from_composition:
        lipid_atoms = np.asarray(sorted(int(i) for i in lipid_from_composition), dtype=int)
    else:
        lipid_atoms = np.asarray(
            [
                int(atom.index)
                for atom in _topology_atoms(top)
                if str(atom.residue.name).strip().upper() in KNOWN_LIPID_RESIDUE_NAMES
            ],
            dtype=int,
        )
    if lipid_atoms.size == 0:
        raise ValueError("找不到任何脂质原子，无法判断膜与周期镜像的水层间隙。")
    _lipid_atom_set = set(int(i) for i in lipid_atoms.tolist())
    lipid_heavy = np.asarray(
        [
            int(atom.index)
            for atom in _topology_atoms(top)
            if int(atom.index) in _lipid_atom_set
            and (getattr(atom.element, "symbol", "") or "").upper() != "H"
        ],
        dtype=int,
    )
    cross_upper, cross_lower = _protein_leaflet_cross_sections_nm2(
        traj, protein_heavy, lipid_heavy, axis, lateral_axes, midplane, upper_z, lower_z
    )
    apl_corrected = 0.5 * (
        (lateral_area - cross_upper) / n_upper
        + (lateral_area - cross_lower) / n_lower
    )
    # 栅格边长是**唯一**的方法参数（最近原子归属没有探针半径），所以敏感性检查
    # 就是"换成 2× 粗栅格复算若干帧"。判读绝对值门必须连它一起看。
    sens_frames = np.unique(
        np.linspace(
            0,
            traj.n_frames - 1,
            min(traj.n_frames, PROTEIN_CROSS_SECTION_SENSITIVITY_MAX_FRAMES),
        ).astype(int)
    )
    apl_sensitivity: Dict[str, Any] = {
        "method": "nearest_reference_atom_partition",
        "grid_nm": PROTEIN_CROSS_SECTION_GRID_NM,
        "coarse_grid_nm": (
            PROTEIN_CROSS_SECTION_GRID_NM
            * PROTEIN_CROSS_SECTION_GRID_SENSITIVITY_FACTOR
        ),
        "n_frames_sampled": int(sens_frames.size),
        "n_protein_reference_atoms": int(protein_heavy.size),
        "n_lipid_reference_atoms": int(lipid_heavy.size),
    }
    up_s, lo_s = _protein_leaflet_cross_sections_nm2(
        traj,
        protein_heavy,
        lipid_heavy,
        axis,
        lateral_axes,
        midplane,
        upper_z,
        lower_z,
        grid_nm=(
            PROTEIN_CROSS_SECTION_GRID_NM
            * PROTEIN_CROSS_SECTION_GRID_SENSITIVITY_FACTOR
        ),
        frames=sens_frames,
    )
    area_s = lateral_area[sens_frames]
    apl_sensitivity["apl_nm2_fine_grid"] = float(
        np.mean(apl_corrected[sens_frames])
    )
    apl_sensitivity["apl_nm2_coarse_grid"] = float(
        np.mean(0.5 * ((area_s - up_s) / n_upper + (area_s - lo_s) / n_lower))
    )
    apl_sensitivity["protein_cross_section_nm2_fine_grid"] = float(
        np.mean(0.5 * (cross_upper[sens_frames] + cross_lower[sens_frames]))
    )
    apl_sensitivity["protein_cross_section_nm2_coarse_grid"] = float(
        np.mean(0.5 * (up_s + lo_s))
    )

    observables: Dict[str, Any] = {
        "apl_nm2": _series(times_ns, apl),
        "apl_protein_corrected_nm2": _series(times_ns, apl_corrected),
        "bilayer_thickness_nm": _series(times_ns, thickness),
        "lipid_tail_order_parameter": _series(times_ns, order_parameter),
        "protein_backbone_rmsd_nm": _series(times_ns, backbone_rmsd),
        "transmembrane_tilt_deg": _series(times_ns, tilt_deg),
        "pocket_rmsd_nm": _series(times_ns, pocket_rmsd),
        "ligand_heavy_atom_rmsd_nm": _series(times_ns, ligand_rmsd),
        "box_xy_area_nm2": _series(times_ns, lateral_area),
        "box_z_nm": _series(times_ns, normal_length),
        "box_volume_nm3": _series(times_ns, volume),
    }

    # ---- 疏水核内异常水 ----
    water_oxygens = _resolve_water_oxygen_indices(top, composition)
    half_core = np.maximum(
        0.0, 0.5 * thickness - MEMBRANE_HYDROPHOBIC_CORE_HEADGROUP_MARGIN_NM
    )
    water_z = traj.xyz[:, water_oxygens, axis]
    normal_lengths = np.asarray(traj.unitcell_lengths, dtype=np.float64)[:, axis]
    water_relative_z = water_z - midplane[:, None]
    water_relative_z -= normal_lengths[:, None] * np.floor(
        water_relative_z / normal_lengths[:, None] + 0.5
    )
    inside = np.abs(water_relative_z) < half_core[:, None]
    core_water = inside.sum(axis=1)

    # ---- 膜与周期镜像的水层间隙 ----
    # `lipid_atoms` 在上面（APL 蛋白横截面校正处）已经解析过，两处共用同一集合。
    lipid_z = traj.xyz[:, lipid_atoms, axis]
    lipid_relative_z = lipid_z - midplane[:, None]
    lipid_relative_z -= normal_lengths[:, None] * np.floor(
        lipid_relative_z / normal_lengths[:, None] + 0.5
    )
    water_gap = normal_length - (
        lipid_relative_z.max(axis=1) - lipid_relative_z.min(axis=1)
    )
    image_contact_frames = int(np.count_nonzero(water_gap < MEMBRANE_MIN_WATER_SLAB_NM))

    # ---- 沿法向的质量密度分布（按组分）----
    density_profile = _density_profile_along_normal(traj, axis, midplane)

    # ---- 脂质横向弛豫时间尺度（由头基横向 MSD 估计）----
    relaxation_ns, relaxation_details = _lipid_lateral_relaxation_timescale_ns(
        traj, head_indices, lateral_axes, times_ns
    )

    # 从 `head_units` 派生，与 `head_indices` / 叶片 mask 同源。
    # 不要在这里重新按残基名找头基——那正是本函数崩过两次的地方。
    leaflet_composition = {"upper": {}, "lower": {}}
    head_cluster_by_atom = {
        int(atom_index): float(head_z_cluster[pos])
        for pos, (_label, atom_index) in enumerate(head_units)
    }
    for label, atom_index in head_units:
        bucket = (
            "upper"
            if head_cluster_by_atom[int(atom_index)] > midplane0
            else "lower"
        )
        leaflet_composition[bucket][label] = (
            leaflet_composition[bucket].get(label, 0) + 1
        )

    diagnostics: Dict[str, Any] = {
        "density_profile_along_normal": density_profile,
        "leaflet_composition": leaflet_composition,
        "anomalous_pocket_water_count": int(np.max(core_water)),
        "membrane_periodic_image_contacts": image_contact_frames,
        "membrane_undulation_or_residual_tension": {
            "midplane_std_nm": float(np.std(midplane)),
            "min_water_slab_nm": float(np.min(water_gap)),
            "water_slab_threshold_nm": MEMBRANE_MIN_WATER_SLAB_NM,
        },
        "lipid_lateral_relaxation_timescale_ns": relaxation_ns,
        "lipid_lateral_diffusion": relaxation_details,
        "equilibration_length_ns": (
            float(equilibration_length_ns)
            if equilibration_length_ns is not None
            else float(times_ns[-1] - times_ns[0])
        ),
        "n_upper": n_upper,
        "n_lower": n_lower,
        "core_water_per_frame_mean": float(np.mean(core_water)),
        "lipid_unit": lipid_unit_label,
        # 时间轴来自哪里必须可追溯：mdtraj 读 DCD 给的是帧号，用错了两道门会往
        # 相反方向坏（末段窗口过严 / 弛豫时间过松），事后只看数字分辨不出来。
        "time_axis_source": time_axis_source,
        "frame_interval_ps": interval_ps,
        "trajectory_span_ns": float(times_ns[-1] - times_ns[0]),
        "protein_cross_section_upper_nm2_mean": float(np.mean(cross_upper)),
        "protein_cross_section_lower_nm2_mean": float(np.mean(cross_lower)),
        "n_protein_heavy_atoms": int(protein_heavy.size),
        "apl_protein_cross_section_sensitivity": apl_sensitivity,
        "apl_correction_definition": (
            "apl_protein_corrected_nm2 = mean over leaflets of "
            "(lateral box area - protein-owned lateral area in that leaflet slab) "
            "/ (number of lipid units in that leaflet). The protein-owned area is a "
            "Voronoi-style nearest-reference-atom partition of the lateral plane on a "
            f"{PROTEIN_CROSS_SECTION_GRID_NM} nm periodic grid: every grid cell is "
            "assigned to the closest heavy atom among the protein and lipid heavy "
            "atoms whose normal-axis coordinate lies inside that leaflet slab. There "
            "is NO probe radius - an outward-dilation definition adds a rim along the "
            "protein perimeter and measurably overestimates the protein area. The grid "
            "spacing is the only method parameter; see "
            "apl_protein_cross_section_sensitivity for the coarse-grid cross-check. "
            "Never compare a corrected APL against an uncorrected one, nor values "
            "produced by different partition definitions."
        ),
        "order_parameter_definition": (
            "intra-lipid C–C bond-vector order parameter relative to the membrane "
            "normal; an §9-permitted equivalent structural indicator, NOT S_CD "
            "(which needs C–H vectors). Do not compare across definitions."
        ),
    }

    if coion_atom_index is not None:
        observables.update(
            _coion_observables_from_trajectory(
                traj, int(coion_atom_index), axis, midplane, times_ns, ligand_heavy,
                composition=composition,
            )
        )
        diagnostics["coion_z_histogram"] = _coion_z_histogram(
            traj, int(coion_atom_index), axis, midplane
        )

    return observables, diagnostics


def _resolve_water_oxygen_indices(topology, composition: Optional[Dict[str, Any]]):
    """水氧原子索引；解析不到就报错，**绝不返回空集**。

    实测 memtest 体系的水叫 `TP3`，而 mdtraj 的 `_WATER_RESIDUES` 只有 `TIP3`——
    `select("water and element O")` 会**静默返回 0 个原子**，于是疏水核内异常水
    与 co-ion 首层水配位数都变成 0，两道门形同虚设且不报错。
    """
    from_composition = (
        (composition or {}).get("atom_indices_by_role", {}).get("water")
    )
    atoms = _topology_atoms(topology)
    if from_composition:
        water_set = set(int(i) for i in from_composition)
        indices = [
            int(atom.index)
            for atom in atoms
            if int(atom.index) in water_set
            and (getattr(atom.element, "symbol", "") or "").upper() == "O"
        ]
    else:
        indices = [
            int(atom.index)
            for atom in atoms
            if str(atom.residue.name).strip().upper() in WATER_MOLECULE_NAMES
            and (getattr(atom.element, "symbol", "") or "").upper() == "O"
        ]
    if not indices:
        present = sorted({str(a.residue.name).strip().upper() for a in atoms})
        raise ValueError(
            "找不到任何水氧原子。膜体系必然有水，所以这是识别失败而不是事实——"
            f"拓扑里出现的残基名：{present[:20]}。"
            f"已知水名：{sorted(WATER_MOLECULE_NAMES)}。"
            "请传入 composition（`classify_system_composition()` 的产出）"
            "或把该水模型名加进 WATER_MOLECULE_NAMES。"
        )
    return np.asarray(sorted(indices), dtype=int)


def _nearest_owner_area_nm2(
    protein_xy,
    lipid_xy,
    lateral_lengths,
    grid_nm: float = PROTEIN_CROSS_SECTION_GRID_NM,
) -> float:
    """横向平面上"离蛋白原子比离任何脂质原子都近"的面积，nm²（周期性）。

    Voronoi 式划分：把横向平面打成边长 `grid_nm` 的栅格，每个格心归给最近的那个
    参考原子（蛋白重原子 ∪ 脂质重原子），返回归给蛋白的面积。
    **没有探针半径**——边界自动落在两类原子中间，所以不存在"外扩多少"这个可调量
    （见 `PROTEIN_CROSS_SECTION_GRID_NM` 的注释：外扩法会沿周长多算一圈）。

    周期性靠把参考原子在横向 3×3 复制一遍实现（格心只取主胞），
    这样贴边的蛋白/脂质在最近邻判断里不会被盒边截断。
    """
    from scipy.spatial import cKDTree

    protein_xy = np.asarray(protein_xy, dtype=float).reshape(-1, 2)
    lipid_xy = np.asarray(lipid_xy, dtype=float).reshape(-1, 2)
    if protein_xy.shape[0] == 0:
        return 0.0
    lx = float(lateral_lengths[0])
    ly = float(lateral_lengths[1])
    nx = max(1, int(round(lx / grid_nm)))
    ny = max(1, int(round(ly / grid_nm)))
    cell_x, cell_y = lx / nx, ly / ny
    if lipid_xy.shape[0] == 0:
        # 该 slab 里没有脂质原子 → 整个横向面积都不归脂质。这不是正常膜体系，
        # 但也不该在这里猜；调用方会因为 APL 明显异常而察觉。
        return lx * ly

    refs = np.vstack([protein_xy, lipid_xy])
    is_protein = np.zeros(refs.shape[0], dtype=bool)
    is_protein[: protein_xy.shape[0]] = True
    # 横向 3×3 周期镜像
    shifts = np.array(
        [(i * lx, j * ly) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float
    )
    refs_periodic = (refs[None, :, :] + shifts[:, None, :]).reshape(-1, 2)
    owner_periodic = np.tile(is_protein, shifts.shape[0])

    centers_x = (np.arange(nx) + 0.5) * cell_x
    centers_y = (np.arange(ny) + 0.5) * cell_y
    grid = np.stack(
        np.meshgrid(centers_x, centers_y, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    _, nearest = cKDTree(refs_periodic).query(grid, k=1)
    return float(owner_periodic[nearest].sum()) * cell_x * cell_y


def _protein_leaflet_cross_sections_nm2(
    traj,
    protein_heavy,
    lipid_heavy,
    axis: int,
    lateral_axes,
    midplane,
    upper_z,
    lower_z,
    grid_nm: float = PROTEIN_CROSS_SECTION_GRID_NM,
    frames=None,
):
    """逐帧、逐叶给出蛋白在该叶片 slab 内占掉的横向面积（nm²）。

    上叶 slab = [中面, 上叶头基平面]，下叶 slab = [下叶头基平面, 中面]。
    只取落在该 slab 内的原子——蛋白在膜外的胞内/胞外结构域不占脂质面积，
    把它们算进来会高估蛋白横截面。
    """
    protein_heavy = np.asarray(protein_heavy, dtype=int)
    lipid_heavy = np.asarray(lipid_heavy, dtype=int)
    lengths = np.asarray(traj.unitcell_lengths, dtype=float)
    frame_list = (
        list(range(traj.n_frames)) if frames is None else [int(f) for f in frames]
    )
    upper = np.zeros(len(frame_list), dtype=float)
    lower = np.zeros(len(frame_list), dtype=float)
    if protein_heavy.size == 0:
        return upper, lower
    for out_i, frame in enumerate(frame_list):
        cell = lengths[frame][lateral_axes]
        p_coords = traj.xyz[frame][protein_heavy]
        l_coords = (
            traj.xyz[frame][lipid_heavy]
            if lipid_heavy.size
            else np.empty((0, 3), dtype=float)
        )
        p_z, l_z = p_coords[:, axis], l_coords[:, axis]
        for out, lo, hi in (
            (upper, midplane[frame], upper_z[frame]),
            (lower, lower_z[frame], midplane[frame]),
        ):
            out[out_i] = _nearest_owner_area_nm2(
                p_coords[(p_z >= lo) & (p_z <= hi)][:, lateral_axes],
                l_coords[(l_z >= lo) & (l_z <= hi)][:, lateral_axes]
                if l_coords.size
                else np.empty((0, 2), dtype=float),
                cell,
                grid_nm=grid_nm,
            )
    return upper, lower


def _density_profile_along_normal(traj, axis: int, midplane) -> Dict[str, Any]:
    """按组分给出沿膜法向的质量密度分布（Da/nm³，相对膜中面）。

    The old implementation accumulated masses and divided only by the number
    of frames.  That quantity was a per-frame mass histogram, not a density;
    its value changed with the lateral box area and bin width.  Normalize each
    frame by its actual slab volume (``A_xy * dz``) before averaging so NPT
    trajectories and different histogram resolutions remain comparable.
    """
    top = traj.topology
    groups: Dict[str, List[int]] = {"water": [], "lipid": [], "protein": [], "ion": []}
    # 用与其它地方相同的名表/归一化，避免这里又出现第三套判据
    # （`TP3` 不在旧的 water 集合、`Na+`/`Cl-` 带符号不在旧的 ion 集合，
    #  HID/NTRP/CCYS 不在 mdtraj 的蛋白表——三处都会静默漏掉）。
    ion_names = MONOATOMIC_ION_NAMES
    water_names = WATER_MOLECULE_NAMES
    for atom in _topology_atoms(top):
        name = str(atom.residue.name).strip().upper()
        if name in water_names:
            groups["water"].append(int(atom.index))
        elif name in KNOWN_LIPID_RESIDUE_NAMES:
            groups["lipid"].append(int(atom.index))
        elif _normalize_ion_name(name) in ion_names:
            groups["ion"].append(int(atom.index))
        elif normalize_protein_residue_name(name) is not None:
            groups["protein"].append(int(atom.index))

    bins = np.linspace(-5.0, 5.0, 101)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width_nm = float(bins[1] - bins[0])
    lengths = np.asarray(traj.unitcell_lengths, dtype=np.float64)
    lateral_axes = [i for i in (0, 1, 2) if i != axis]
    lateral_area = lengths[:, lateral_axes[0]] * lengths[:, lateral_axes[1]]
    if (
        lengths.shape != (traj.n_frames, 3)
        or not np.all(np.isfinite(lateral_area))
        or np.any(lateral_area <= 0.0)
    ):
        raise ValueError("密度剖面需要每帧正有限的横向盒面积")
    profile: Dict[str, Any] = {
        "bin_centers_nm": centers.tolist(),
        "density_units": "dalton_per_nm3",
        "bin_width_nm": bin_width_nm,
        "normalization": "mass / (A_xy * bin_width), averaged over frames",
    }

    for group, indices in groups.items():
        if not indices:
            profile[group] = [0.0] * centers.size
            continue
        idx = np.asarray(indices, dtype=int)
        masses = np.asarray(
            [float(getattr(top.atom(int(i)).element, "mass", 0.0) or 0.0) for i in idx]
        )
        relative_z = traj.xyz[:, idx, axis] - midplane[:, None]
        # Keep atoms in the nearest periodic image of the membrane midplane.
        # The membrane protocol uses a rectangular cell; for a general cell,
        # the normal component is still well-defined when the normal axis is a
        # box edge.  This prevents a water/lipid crossing at +/-L/2 from moving
        # to the opposite end of the density profile.
        normal_lengths = lengths[:, axis]
        relative_z -= normal_lengths[:, None] * np.floor(
            relative_z / normal_lengths[:, None] + 0.5
        )
        hist = np.zeros(centers.size, dtype=float)
        for frame in range(traj.n_frames):
            counts, _ = np.histogram(relative_z[frame], bins=bins, weights=masses)
            slab_volume_nm3 = lateral_area[frame] * bin_width_nm
            hist += counts / slab_volume_nm3
        profile[group] = (hist / max(1, traj.n_frames)).tolist()
    return profile


def _unwrap_trajectory_subset(traj, atom_indices: Sequence[int]) -> np.ndarray:
    """Unwrap selected atoms with the actual per-frame periodic cell.

    MDTraj DCD/XTC coordinates are normally wrapped into the primary cell.  A
    direct frame-to-frame difference therefore turns a boundary crossing into
    a box-length jump.  Reconstruct continuous coordinates from consecutive
    minimum-image increments; the helper also handles triclinic unitcell
    vectors and falls back to diagonal lengths when a reader did not expose
    vectors explicitly.
    """
    indices = np.asarray(list(atom_indices), dtype=int)
    wrapped = np.asarray(traj.xyz[:, indices, :], dtype=np.float64)
    if wrapped.ndim != 3 or wrapped.shape[-1] != 3:
        raise ValueError("轨迹坐标必须是 (n_frames, n_atoms, 3)")
    if wrapped.shape[0] == 0 or wrapped.shape[1] == 0:
        return np.array(wrapped, copy=True)
    lengths = np.asarray(traj.unitcell_lengths, dtype=np.float64)
    if lengths.shape != (wrapped.shape[0], 3) or not np.all(np.isfinite(lengths)):
        raise ValueError("轨迹每帧必须提供有限的 unitcell_lengths")
    if np.any(lengths <= 0.0):
        raise ValueError("轨迹 unitcell_lengths 必须为正")
    vectors = getattr(traj, "unitcell_vectors", None)
    if vectors is not None:
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.shape != (wrapped.shape[0], 3, 3):
            vectors = None
    out = np.empty_like(wrapped)
    out[0] = wrapped[0]
    for frame in range(1, wrapped.shape[0]):
        box = vectors[frame] if vectors is not None and np.all(np.isfinite(vectors[frame])) else np.diag(lengths[frame])
        out[frame] = out[frame - 1] + minimum_image_displacement_nm(
            wrapped[frame] - wrapped[frame - 1], box
        )
    return out


def _lipid_lateral_relaxation_timescale_ns(
    traj, head_indices, lateral_axes, times_ns
) -> Tuple[float, Dict[str, Any]]:
    """由头基横向 MSD 估计脂质横向弛豫时间尺度（§9）。返回 `(tau_ns, details)`。

    做法：**时间平均** MSD —— 对每个 lag `L` 用**所有时间原点**
    `msd[L] = mean_t |x(t+L) − x(t)|²`，再在一个**声明的 lag 窗口**内做带截距的
    线性拟合得 2D 扩散系数（MSD = 4·D·Δt），最后换算成"位移一个脂质直径所需时间"。
    这是一个**量级估计**，用于论证预平衡时长，不是精确扩散系数测量。

    ## 为什么不能用单一参考帧 + 过原点拟合全部 lag（MEM-11）

    原实现取 `reference = lateral[0]`（每个 lag 只有**一个**样本）并用过原点最小
    二乘拟合**全部** lag（权重 ∝ lag²，长 lag 主导）。两个问题：

    * 脂质横向 MSD 在短 lag 是**亚扩散**的（实测 memtest 100 ns，1–30 ns 区间
      MSD ~ t^0.80），过原点拟合把这段一起吃进去必然偏；
    * 长 lag 只有极少数独立样本，却被 lag² 权重放大。

    实测后果：同一条 100 ns 轨迹，只改"用到前多少 ns"，τ 就是
    30.1 → 38.0 → 24.1 → 13.2 → 10.8 → 11.6 ns（10/20/40/60/80/100 ns），
    **非单调乱跳** —— 这样的量不该当硬门（见 `evaluate_membrane_quality_gate`
    里 `equilibration_vs_relaxation` 那段说明）。
    改成时间平均 + 5–30 ns 窗口后同一条轨迹给 D = 0.008664 nm²/ns → τ = 18.467 ns，
    与 POPC 文献 D ≈ 0.008 nm²/ns 给出的 τ ≈ 20 ns 吻合。
    """
    if times_ns.size < 3:
        raise ValueError("估计脂质横向弛豫时间尺度至少需要 3 帧")
    diffs = np.diff(times_ns)
    if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-12):
        raise ValueError(
            "时间轴不是等间隔的，时间平均 MSD 需要等间隔帧。"
            "请提供等间隔轨迹或显式给出 frame_interval_ps。"
        )
    dt_ns = float(diffs[0])
    span_ns = float(times_ns[-1] - times_ns[0])

    lag_lo, lag_hi = (
        LIPID_LATERAL_MSD_FIT_LAG_MIN_NS,
        LIPID_LATERAL_MSD_FIT_LAG_MAX_NS,
    )
    window_source = "declared"
    if span_ns < 2.0 * lag_lo:
        # 轨迹太短装不下声明窗口：按跨度缩放，并**如实记录**用了哪个窗口。
        # 不静默套用声明窗口（那会拟合到根本没有的 lag），也不报错（短轨迹在
        # 合成测试与早期诊断里是合法用法）。
        lag_lo, lag_hi = 0.1 * span_ns, 0.6 * span_ns
        window_source = "scaled_to_trajectory_span"

    lag_min_frames = max(1, int(round(lag_lo / dt_ns)))
    lag_max_frames = min(traj.n_frames - 1, int(round(lag_hi / dt_ns)))
    if lag_max_frames <= lag_min_frames:
        lag_min_frames = 1
        lag_max_frames = max(2, traj.n_frames - 1)
        window_source = "degenerate_span_used_all_lags"
    # 拟合点数上限：MSD 是 O(n_frames × n_heads) 一个 lag，30 个点足够定斜率。
    lag_frames = np.unique(
        np.linspace(lag_min_frames, lag_max_frames, 30).astype(int)
    )

    # Coordinates in an XTC/DCD are wrapped into the primary cell.  Taking
    # ``lateral[t+lag] - lateral[t]`` directly therefore turns one ordinary
    # boundary crossing into an O(box-length) jump and inflates D (or makes a
    # slowly diffusing membrane appear to relax instantly).  Reconstruct a
    # continuous trajectory from consecutive minimum-image increments first.
    wrapped_head = np.asarray(traj.xyz[:, head_indices, :], dtype=np.float64)
    unwrapped_head = np.empty_like(wrapped_head)
    unwrapped_head[0] = wrapped_head[0]
    cell_vectors = getattr(traj, "unitcell_vectors", None)
    if cell_vectors is not None:
        cell_vectors = np.asarray(cell_vectors, dtype=np.float64)
        if cell_vectors.shape != (traj.n_frames, 3, 3):
            cell_vectors = None
    cell_lengths = np.asarray(traj.unitcell_lengths, dtype=np.float64)
    for frame in range(1, traj.n_frames):
        if cell_vectors is not None and np.all(np.isfinite(cell_vectors[frame])):
            box = cell_vectors[frame]
        else:
            box = np.diag(cell_lengths[frame])
        step = minimum_image_displacement_nm(
            wrapped_head[frame] - wrapped_head[frame - 1], box
        )
        unwrapped_head[frame] = unwrapped_head[frame - 1] + step
    lateral = unwrapped_head[:, :, lateral_axes]
    msd = np.empty(lag_frames.size, dtype=float)
    for i, lag in enumerate(lag_frames):
        delta = lateral[int(lag):] - lateral[: -int(lag)]
        msd[i] = float(np.mean(np.sum(delta**2, axis=2)))
    lag_ns = lag_frames * dt_ns

    if lag_frames.size >= 2:
        slope, intercept = np.polyfit(lag_ns, msd, 1)
    else:
        slope, intercept = msd[0] / lag_ns[0], 0.0
    # 亚扩散指数（诊断用）：MSD ~ t^alpha，纯扩散 alpha = 1。
    positive = msd > 0
    if int(np.count_nonzero(positive)) >= 2:
        alpha = float(
            np.polyfit(np.log(lag_ns[positive]), np.log(msd[positive]), 1)[0]
        )
    else:
        alpha = float("nan")

    d_lateral = max(float(slope) / 4.0, 1.0e-12)  # nm²/ns
    reference_displacement = LIPID_LATERAL_RELAXATION_REFERENCE_DISPLACEMENT_NM
    tau_ns = float(reference_displacement**2 / (4.0 * d_lateral))
    details = {
        "method": "time_averaged_msd_multiple_origins",
        "lateral_diffusion_nm2_per_ns": float(d_lateral),
        "msd_power_law_exponent": alpha,
        "fit_lag_window_ns": [float(lag_ns[0]), float(lag_ns[-1])],
        "fit_lag_window_source": window_source,
        "n_fit_points": int(lag_frames.size),
        "fit_intercept_nm2": float(intercept),
        "reference_displacement_nm": float(reference_displacement),
        "trajectory_span_ns": span_ns,
        # 判读锚点：POPC 纯脂文献 D ≈ 0.008 nm²/ns → τ ≈ 20 ns。
        "pure_popc_reference_diffusion_nm2_per_ns": (
            PURE_POPC_REFERENCE_LATERAL_DIFFUSION_NM2_PER_NS
        ),
    }
    return tau_ns, details


def _coion_observables_from_trajectory(
    traj, coion_index: int, axis: int, midplane, times_ns, ligand_heavy,
    composition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """co-ion 的 §13.1 几何量时间序列。"""
    top = traj.topology
    coion_xyz = traj.xyz[:, coion_index, :]
    lengths = np.asarray(traj.unitcell_lengths, dtype=np.float64)
    normal_lengths = lengths[:, axis]
    coion_dz = coion_xyz[:, axis] - np.asarray(midplane, dtype=np.float64)
    coion_dz -= normal_lengths * np.floor(coion_dz / normal_lengths + 0.5)
    abs_z = np.abs(coion_dz)

    cell_vectors = getattr(traj, "unitcell_vectors", None)
    if cell_vectors is not None:
        cell_vectors = np.asarray(cell_vectors, dtype=np.float64)
        if cell_vectors.shape != (traj.n_frames, 3, 3):
            cell_vectors = None

    def _frame_minimum_image(delta):
        out = np.empty_like(delta, dtype=np.float64)
        for frame in range(traj.n_frames):
            box = (
                cell_vectors[frame]
                if cell_vectors is not None and np.all(np.isfinite(cell_vectors[frame]))
                else np.diag(lengths[frame])
            )
            out[frame] = minimum_image_displacement_nm(delta[frame], box)
        return out

    def _min_distance(indices):
        if len(indices) == 0:
            return np.full(traj.n_frames, np.inf)
        delta = traj.xyz[:, np.asarray(indices, dtype=int), :] - coion_xyz[:, None, :]
        delta = _frame_minimum_image(delta)
        return np.min(np.linalg.norm(delta, axis=2), axis=1)

    protein_heavy = top.select("protein and not element H")
    phosphorus = top.select("element P")
    water_oxygens = _resolve_water_oxygen_indices(top, composition)

    if water_oxygens.size:
        delta = traj.xyz[:, water_oxygens, :] - coion_xyz[:, None, :]
        delta = _frame_minimum_image(delta)
        distances = np.linalg.norm(delta, axis=2)
        first_shell = (distances <= COION_FIRST_SHELL_WATER_CUTOFF_NM).sum(axis=1)
    else:
        first_shell = np.zeros(traj.n_frames, dtype=int)

    return {
        "coion_abs_z_from_midplane_nm": _series(times_ns, abs_z),
        "coion_ligand_min_image_distance_nm": _series(
            times_ns, _min_distance(ligand_heavy)
        ),
        "coion_protein_heavy_atom_distance_nm": _series(
            times_ns, _min_distance(protein_heavy)
        ),
        "coion_nearest_phosphorus_distance_nm": _series(
            times_ns, _min_distance(phosphorus)
        ),
        "coion_first_shell_water_count": _series(times_ns, first_shell),
    }


def _coion_z_histogram(traj, coion_index: int, axis: int, midplane) -> Dict[str, Any]:
    """§9 末条：co-ion 的 z 分布直方图，不只是瞬时距离。"""
    relative_z = traj.xyz[:, coion_index, axis] - np.asarray(midplane, dtype=float)
    lengths = np.asarray(traj.unitcell_lengths, dtype=float)
    normal_length = lengths[:, axis]
    relative_z -= normal_length * np.floor(relative_z / normal_length + 0.5)
    bins = np.linspace(-6.0, 6.0, 61)
    counts, edges = np.histogram(relative_z, bins=bins)
    return {
        "bin_centers_nm": (0.5 * (edges[:-1] + edges[1:])).tolist(),
        "counts": counts.astype(int).tolist(),
        "min_abs_z_nm": float(np.min(np.abs(relative_z))),
    }


def _validate_minimum_image(box_vectors, cutoff_nm: float) -> None:
    """Validate the triclinic minimum-image condition using plane spacings."""
    vectors = box_vectors
    if hasattr(vectors, "value_in_unit"):
        vectors = vectors.value_in_unit(unit.nanometer)
    else:
        vectors = [
            vector.value_in_unit(unit.nanometer)
            if hasattr(vector, "value_in_unit")
            else vector
            for vector in vectors
        ]
    matrix = np.asarray(vectors, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"Invalid periodic box vectors: shape={matrix.shape}")
    volume = float(abs(np.linalg.det(matrix)))
    if volume <= 0.0:
        raise ValueError(f"Periodic box has non-positive volume: {volume}")

    spacings = []
    for axis in range(3):
        other = [idx for idx in range(3) if idx != axis]
        area = float(np.linalg.norm(np.cross(matrix[other[0]], matrix[other[1]])))
        if area <= 0.0:
            raise ValueError("Periodic box contains collinear vectors")
        spacings.append(volume / area)
    shortest = min(spacings)
    if shortest <= 2.0 * float(cutoff_nm):
        raise ValueError(
            "Minimum-image violation: shortest box plane spacing "
            f"{shortest:.6f} nm must exceed 2*cutoff="
            f"{2.0 * float(cutoff_nm):.6f} nm"
        )


def _build_openmmml_kwargs(
    device: Optional[str] = None,
    precision: Optional[str] = None,
    return_energy_type: Optional[str] = None,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
) -> Dict[str, Any]:
    """
    统一按 openmm-ml 官方接口组织 MLPotential.createSystem() 参数。
    是否显式传 precision 由调用侧决定；若不传则遵循 openmm-ml 的模型默认精度。
    """
    kwargs: Dict[str, Any] = {}
    if return_energy_type is not None:
        kwargs["returnEnergyType"] = return_energy_type
    if device is not None:
        kwargs["device"] = device
    if precision in ("single", "double"):
        kwargs["precision"] = precision
    if charge is not None:
        kwargs["charge"] = charge
    if multiplicity is not None:
        kwargs["multiplicity"] = multiplicity
    return kwargs


_MACE_LOCAL_MODEL_PATHS = {
    "mace-off24-medium": os.path.expanduser("~/.cache/mace/MACE-OFF24_medium.model"),
}


def _build_mace_potential(model_name: str):
    """
    部分 openmm-ml 版本的模型注册表里没有 mace-off24-medium 等较新的预训练名，
    但支持用 model_name="mace" + modelPath=本地缓存权重文件的方式加载同一个模型。
    这里对已知的本地缓存做名字 -> 路径映射；找不到对应缓存文件时退回原始按名加载
    （交给 openmm-ml 自己报错或解析）。
    """
    cached_path = _MACE_LOCAL_MODEL_PATHS.get(model_name)
    if cached_path and os.path.isfile(cached_path):
        return _require_mlpotential()("mace", modelPath=cached_path)
    return _require_mlpotential()(model_name)


def _select_env_indices_from_mdtraj_frame(frame, lig_idx: np.ndarray, env_radius_nm: float, max_env_atoms: Optional[int] = None) -> np.ndarray:
    """
    先做半径近邻筛选，再按“到配体最近距离”的 Top-K 排序裁剪环境原子。
    这里裁的是原子，不是整盒水，也不是全环境残基。
    """
    import mdtraj as md

    raw_env = md.compute_neighbors(frame, env_radius_nm, lig_idx)[0]
    env_idx = np.setdiff1d(raw_env, lig_idx, assume_unique=True)
    if max_env_atoms is None or len(env_idx) <= max_env_atoms:
        return np.asarray(env_idx, dtype=int)

    pos_nm = np.asarray(frame.xyz[0], dtype=np.float64)
    box_vecs = None
    # MDTraj may expose only lengths for some trajectory formats. Treat that
    # as an orthorhombic periodic box instead of silently sorting by raw
    # wrapped coordinates, which can put a boundary-near atom at the wrong end
    # of the neighbour ranking.
    try:
        if frame.unitcell_vectors is not None:
            candidate = np.asarray(frame.unitcell_vectors[0], dtype=np.float64)
            if candidate.shape == (3, 3) and np.all(np.isfinite(candidate)):
                box_vecs = candidate
    except Exception:
        box_vecs = None
    if box_vecs is None:
        try:
            lengths = np.asarray(frame.unitcell_lengths[0], dtype=np.float64)
            if lengths.shape == (3,) and np.all(np.isfinite(lengths)) and np.all(lengths > 0.0):
                box_vecs = np.diag(lengths)
        except Exception:
            box_vecs = None

    delta = pos_nm[lig_idx][:, None, :] - pos_nm[env_idx][None, :, :]
    if box_vecs is not None:
        delta = minimum_image_displacement_nm(delta, box_vecs)
    dists = np.linalg.norm(delta, axis=-1)
    min_dists = np.min(dists, axis=0)
    keep_order = np.argsort(min_dists)[:max_env_atoms]
    return np.sort(np.asarray(env_idx[keep_order], dtype=int))


print = _log_print


