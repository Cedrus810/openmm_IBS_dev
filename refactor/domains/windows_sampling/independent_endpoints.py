# ============================================================================
# 3. 双λ窗口管理器
# ============================================================================
# ============================================================================
# [INDEPENDENT_ENDPOINT_PROTOCOL_VERSION=1] 端点态独立固定-λ 生产采样
#
# 机制见 docs/archive/STAGE2_ROOT_CAUSE_2026-08-28.md。IBS 的 stage2 每个窗口只跑**一条**
# 轨迹，窗口内所有 λ 态都从这一条轨迹重加权出来。那条轨迹里配体（哪怕软核）
# 几乎总占着体积，"水填满配体空腔"的构型概率 ≈ 0——**重加权造不出没采到的
# 构型，再多帧也造不出来**。
#
# ⚠️ [2026-09-02 归因更正] 这段原文写的是"根因见……"，并用 4W53 溶剂腿
# window 2 的 -14.85 真值 / 生产 +4.64 / 错 +19.49 当定量依据。**那个幅度归因
# 是错的**：docs/archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md（2026-09-02
# 结案）把该误差的 **98% 定罪为 λ-WCA 防护壳（Group 4）**——壳退役
# （`WCA_SHIELD_RETIRED = True`）后溶剂腿 stage2 从 +38.59 变成 ≈ -8.3
# （独立参考真值 -6.58 ± 0.26，no-LRC 口径），ΔG_bind 从 +12.75 变成
# -21.36 ± 0.93。上面那些逐窗口数字都是**壳还在时**测的，已被污染。
#
# ⚠️ [2026-09-09 再更正] 上一版这里写"只是残余 1.7~4.2 kJ/mol（≈4%）的候选
# 解释之一，另一个候选是 LRC 口径"——**两句都不对**。LRC 不是候选，它是**掩护**：
# 生产含 LJ 尾项、参考真值不含，尾项 +2.823 与真实误差 -4.318 符号相反、相消成
# -1.494，把 5.5σ 伪装成 1.9σ。口径对齐后（尾项减在生产侧，数值从
# `energies.npy - sampling_states.npy.T` 直接读，那是逐 λ 态严格常数）：
# 生产 -10.898 vs 真值 -6.581±0.256 ⟹ 残差 **-4.318 kJ/mol，5.5σ**。
# 拆分：**密度差 -0.86 ± 0.13（20%）+ 采样协议 ≈-3.46（80%）**。
#   · 密度差：生产 stage2 走 NVT，用**自己 NPT 预平衡末帧冻结的盒**（42.747 nm^3），
#     而参考臂固定用建系盒（43.950）且拒绝恒压器 ⟹ 生产密度高 2.81%。
#     实测 ∂ΔA_LJ/∂V = P_coup - P_dec = +0.71 ± 0.11 kJ/mol/nm^3（≈12 bar）。
#     ⚠️ 用水的压缩率粗估会大 5~8 倍，别用。
#   · 其余十一项 Hamiltonian/估计器级候选已逐条实测排除（估计器、ESS 门、σ、链式接缝、
#     端点、系综、cutoff、软核约定、residual 臂、窗口内漂移、有限样本偏差），
#     **剩下 80% 就是本模块描述的这个机制**：残差全在 λ<0.578 的下降支（空腔塌缩区），
#     win4 一个窗口占 50%。
# 所以保留这条独立端点采样路径的理由不只是"机制成立"，
# 而是**它针对的正是壳退役后剩下误差的主体**。见 docs/archive/STAGE2_SOLVENT_LEG_ERROR_BUDGET.md。
#
# 决定性的一点：window 2 的相邻 <ΔU> 只有 0.4~0.6 kT，**任何基于能量的重叠
# 判据都会说"完美"**。所以这不是"窗口太宽/重叠不足/统计噪声"，加窗、插 λ、
# 延长采样、乃至任何形式的重加权诊断都治不了它。
#
# 本模块实现的修法：对 λ_vdw 接近 0 的关键态**各自建立真正独立的固定-λ 生产
# 轨迹**——每个态有自己的 System、自己的 Context、自己的平衡和采样，绝不从
# 某条窗口混合轨迹重加权出来。这样 λ=0 附近的态从一开始就在"配体不与环境
# 相互作用"的哈密顿量下演化，水本来就能进空腔，塌缩系综是被**采到**的而不是
# 被重加权"造"出来的。
#
# 五条必须同时成立的口径要求（少一条这套东西就没有意义）：
#
#   1. **正式采样不得保留 Group-4 WCA 防护壳。** 传进来的 system XML 必须是
#      `_serialize_ibs_common_system` 的产物（group 1 与 group 4 都已移除），
#      `_build_fixed_state_simulation(require_group4=False)` 会硬断言系统里
#      不存在 group 4。WCA 只准用于建初始构型/预平衡，不准进生产动力学——
#      它把水挡在空腔外，正是本模块要采的那个构型。
#   2. **其余一切与生产完全同一套。** system XML 直接来自已组装好的生产窗口
#      系统，所以 ACE softcore 形式、PME、cutoff/switching 全部逐字节相同；
#      每个态的 CV 力用的是生产自己的 `ibs_wrapper._int_cv_force_xmls[k]`；
#      LRC 用的是 `ibs_wrapper.lj_tail_lrc_coeff_kj_mol`，逐帧按
#      `lrc_coeff[k] / V(t)` 加上——与 `IBSSampler.collect_energies` 里
#      `target_energies = softcore_energies + lrc_energies` 是同一个式子、
#      同一组系数。绝不在这里另算一份。
#   3. **湿空腔与干空腔分别起 walker。** 同一个态至少两个独立起点：干空腔
#      （配体耦合端平衡出来的构型，空腔里没有水）和湿空腔（在完全解耦端长
#      平衡、让水灌进空腔之后的构型）。两组各自跑多个 walker。若两组停在
#      各自的模态里出不来，MBAR 的统计误差可以很小而结果是错的——所以
#      两组的 ΔF 之差必须过 2σ 门（见 `endpoint_wet_dry_hysteresis_gate`）。
#   4. **逐态、逐 segment 时间去相关**（`_decorrelate_per_segment`），MBAR 吃
#      的是有效独立样本数，不是原始帧数。
#   5. **结构性证据必须落盘**：每帧的空腔水分子数、湿/干占比、水进出空腔的
#      转换次数、每个 walker 各自的占据分布。统计量对"该采的构型一次都没采
#      到"这个失效模式是失明的（4W53 的 split-half 前后半程一致地错），只有
#      结构量能直接回答"水到底进没进去过"。
#
# 与既有 fixed-H 探针的关系：`probe_adjacent_path_overlap_bank` 早就在用
# `_build_fixed_state_simulation` 做按态独立采样，但它是**诊断探针**（回答
# "λ 网格连不连得上"），采样长度和产物都不用于生产自由能。本模块复用同一套
# 底层构件，但产出的是**生产数据**：能量矩阵直接喂 MBAR 当 stage2 端点段的
# 主值。两者的 checkpoint 目录、指纹、采样长度都彼此独立，不共享缓存。
# ============================================================================
INDEPENDENT_ENDPOINT_PROTOCOL_VERSION = 1

# 空腔水判据：水的氧原子到**任一配体重原子**的最小像距离小于这个半径，就算它
# 侵入了配体本该占据的体积。0.24 nm 明显小于 O–O 第一峰(0.28 nm)，配体真正
# 耦合时任何水都不可能待在这个距离上——所以计数 > 0 是"空腔被水占据"的直接
# 结构证据，而不是一个需要调参的软指标。
CAVITY_PROBE_RADIUS_NM = 0.24
# 一帧里有多少个这样的水就算"湿"。默认 1：一个水进到配体体积里就已经是那个
# 被 IBS 单轨迹漏掉的构型。
CAVITY_WET_MIN_WATERS = 1
# 湿/干双起点 ΔF 之差的验收门，单位是合并 σ 的倍数。
# ⚠️ 直接**别名** abfe_core.STAGE2_HYSTERESIS_MAX_SIGMA，不另立数值：
# 该常量此前只写进 provenance、从未被任何代码执行过（死常量）。本模块是它的第一个
# 真实执行点。若这里写成独立的 2.0，就会出现"provenance 记录的阈值"与"实际生效的
# 阈值"两份定义，改一个忘另一个 —— 那正是它最初变成死常量的同一类问题。
ENDPOINT_WET_DRY_MAX_SIGMA = STAGE2_HYSTERESIS_MAX_SIGMA

# ============================================================================
# 干/湿双起点诊断：**已退役（2026-09-01）**，在生产路径上是死代码。
# ----------------------------------------------------------------------------
# 退役理由不是"太贵"，是**这个观测量本身是错的**：`count_cavity_waters` 的探针
# 锚在**配体重原子**上，而配体在 λ_vdw→0 只被 Boresch 松松拴着、会整体漂移。
# 于是"空腔水数"在复合物腿上**不是状态函数**——同一个热力学态、同一个 λ，
# 只因配体漂到哪里就能给出不同的湿/干读数。一个不是状态函数的量，无论门槛怎么
# 定都没有判别力，`|ΔF_wet - ΔF_dry| <= 2σ` 这条判据也就无从成立。
# （4W53 复合物腿实测：连续三天每轮都跑满 50 万步湿盆搜索、每轮都"没找到"。）
#
# 退役方式**刻意选择保留函数定义、只切断生产调用**（死代码而非删除）：
#   · `endpoint_wet_dry_hysteresis_gate` / `prepare_wet_cavity_seed` /
#     `build_wet_cavity_seed_key` 等仍然可导入、行为不变，历史结果仍解释得通，
#     现有测试仍然按原样通过；
#   · 生产路径（`ABFEPipeline._run_independent_endpoint_segment`）不再构造湿臂、
#     不再做 wet/dry 拆分求解、不再调用这道门，改用下面的退役记录。
#
# ⚠️ 绝对不要因此升 `INDEPENDENT_ENDPOINT_PROTOCOL_VERSION`：它进了
# `abfe_pipeline._remd_sampling_fingerprint`，升一次就让所有已完成的端点 bank
# 失效、逼 GPU 重跑。而退役湿臂**不改变干臂采的是哪个分布**——`init_modes`
# 属于 manifest 的**预算**键（`_INDEPENDENT_ENDPOINT_BUDGET_KEYS`）而非身份键，
# 所以盘上已有的干记录原样有效。这正是身份/预算二分要保护的场景。
WET_DRY_DIAGNOSTIC_RETIRED = True
#: 退役记录的 reason，供下游区分"退役"与历史上的"没跑"/"跑了没找到"。
WET_DRY_RETIRED_REASON = "wet_dry_diagnostic_retired"


def retired_wet_dry_gate_record() -> Dict[str, Any]:
    """生产路径上代替 `endpoint_wet_dry_hysteresis_gate` 的固定退役记录。

    `passed=None` 是**必须**的，不是随手选的：
    `combine_ibs_and_independent_endpoint` 里写的是
    `_wd_passed = _wd.get("passed", False)`，随后 `wet_dry_blocks = (_wd_passed is False)`。
    所以把这道门整个从结果里删掉、或让它缺 `passed` 键，都会被读成
    **False → 阻塞 converged**，等于用"退役"把整条腿判死。`None` 才是
    "不参与合取"的那个值。
    """
    return {
        "evaluated": False,
        "passed": None,
        "retired": True,
        "reason": WET_DRY_RETIRED_REASON,
        "retired_note": (
            "干/湿双起点诊断已退役：空腔水数探针锚在会漂移的配体上，"
            "在复合物腿上不是状态函数，故无判别力。生产路径只跑干起点。"
        ),
        "max_sigma": float(ENDPOINT_WET_DRY_MAX_SIGMA),
        "delta_F_wet_kJ_mol": None,
        "delta_F_dry_kJ_mol": None,
        "n_wet_dry_transitions": None,
    }


INDEPENDENT_ENDPOINT_INTEGRATOR_SEED_BASE = 611_501
INDEPENDENT_ENDPOINT_VELOCITY_SEED_BASE = 733_907
# 质数步长，避免 (态, 模态, walker) 三重循环里不同组合撞到同一个种子。
_INDEPENDENT_ENDPOINT_SEED_STRIDE = 97
_INDEPENDENT_ENDPOINT_MODE_STRIDE = 10_007
_INDEPENDENT_ENDPOINT_WALKER_STRIDE = 1_009

# 🔑 [2026-09-01] 干/湿双起点**已从生产路径移除**，只保留干起点。
#
# 用户决定：不再为它烧 GPU（4W53 连跑三天，每轮都得出"没找到"）。
# 依据不只是成本：`count_cavity_waters` 的探针锚在**配体重原子**上，而 λ_vdw→0 时
# 幽灵配体与环境无 LJ、只被 Boresch 松松拴着、会漂 —— 所以它测的是「幽灵当前漂到哪
# 0.24 nm 内有几个水」，**不是空腔水合度**，在复合物腿上根本不是状态函数。
# v3 实测把这点钉死：同一个 λ 上 s1|w1 转变 181 次、s1|w0 零次，
# 同分布下单段干驻留 ≥500 帧的概率 ≈ exp(-500/1.54) ≈ 1e-141。
# 再多采样、再换种子都修不了 —— 那是观测量的问题，不是采样量的问题。
#
# ⚠️ **保留的是独立端点采样本身**（每个 λ 各自跑独立轨迹），那是
# docs/archive/STAGE2_ROOT_CAUSE_2026-08-28.md §8.2 提出的采样设计层修法，与干/湿无关。
# ⚠️ [2026-09-02 归因更正] 原文写"针对 **+32 kJ/mol 根因**的修复"——**错了**。
# 那 +32 的 98% 是 λ-WCA 防护壳，已于 2026-09-02 退役定案（见
# docs/archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md）。独立端点采样
# 针对的是单系综重加权这个**机制**。
# ⚠️ [2026-09-09 更正] 上一版这里写"只对残余 ≈4% 负责"——**作废**（那个 4% 带
# LRC 口径错误且是带壳测的）。壳退役后该机制对应 **-2.7 ~ -3.5 kJ/mol，
# 即残差的约 80%**（另 20% 是生产/参考的盒体积差 2.81%，实测 -0.86 ± 0.13）。见 docs/archive/STAGE2_SOLVENT_LEG_ERROR_BUDGET.md。
# 去掉的只是"从湿构型再起一组"这个诊断装置。
# 要恢复双起点，先把探针锚点从配体换成蛋白腔壁参考原子，让它重新成为状态函数；
# 在那之前恢复它只会再烧一轮 GPU 得到同样无判别力的结果。
INDEPENDENT_ENDPOINT_INIT_MODES = ("dry",)
# 历史全集：`_independent_endpoint_seed` 用 index() 派生种子，湿记录的种子必须与
# 移除前逐位一致，否则盘上已有的湿记录（若有）会对不上。所以索引表保持不变。
INDEPENDENT_ENDPOINT_ALL_INIT_MODES = ("dry", "wet")


def water_oxygen_indices(topology) -> np.ndarray:
    """拓扑里所有水氧原子的索引。

    只认真正的水残基（HOH/WAT/TIP3/SOL/H2O）里的氧，不靠"元素是 O"来猜——
    蛋白骨架和配体上也全是氧。
    """
    water_resnames = {"HOH", "WAT", "TIP3", "TIP3P", "TIP4", "SOL", "H2O", "T3P"}
    indices = []
    for residue in topology.residues():
        if str(residue.name).upper() not in water_resnames:
            continue
        for atom in residue.atoms():
            element = getattr(atom, "element", None)
            symbol = getattr(element, "symbol", None) if element is not None else None
            if symbol == "O":
                indices.append(int(atom.index))
    return np.asarray(indices, dtype=int)


def ligand_heavy_atom_indices(topology, ligand_indices) -> np.ndarray:
    """配体重原子（非氢）索引。空腔判据只用重原子——氢的范德华半径太小，
    把它算进去会让"侵入"判据依赖于配体的具体质子化细节。"""
    atoms = list(topology.atoms())
    heavy = []
    for idx in np.asarray(ligand_indices, dtype=int).ravel():
        atom = atoms[int(idx)]
        element = getattr(atom, "element", None)
        symbol = getattr(element, "symbol", None) if element is not None else None
        if symbol != "H":
            heavy.append(int(idx))
    if not heavy:
        raise ValueError("配体没有任何重原子，无法定义空腔水判据")
    return np.asarray(heavy, dtype=int)


def _minimum_image_displacements(delta_nm: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    """OpenMM 归约形式（reduced form）三斜盒子的最小像位移。

    必须按 c→b→a 的顺序依次减，这是归约形式成立的前提；对正交盒子退化成
    逐轴 round，结果与常见的 `dr -= L*round(dr/L)` 完全一致。
    """
    return minimum_image_displacement_nm(delta_nm, box_nm)


def count_cavity_waters(
    positions_nm: np.ndarray,
    box_nm: np.ndarray,
    ligand_heavy_idx: np.ndarray,
    water_oxygen_idx: np.ndarray,
    probe_radius_nm: float = CAVITY_PROBE_RADIUS_NM,
) -> int:
    """一帧里"侵入配体体积"的水分子数：氧到任一配体重原子的最小像距离 < 半径。

    这是 stage2 唯一能直接回答"水到底进没进空腔"的量。所有基于能量的统计门
    对这件事都是失明的（4W53 实测：g≈1.4、split-half 一致，稳定地收敛到错值）。
    """
    pos = np.asarray(positions_nm, dtype=np.float64)
    lig = pos[np.asarray(ligand_heavy_idx, dtype=int)]
    wat = pos[np.asarray(water_oxygen_idx, dtype=int)]
    if wat.size == 0:
        return 0
    delta = wat[:, None, :] - lig[None, :, :]
    delta = _minimum_image_displacements(delta, box_nm)
    dist2 = np.sum(delta * delta, axis=-1)
    r2 = float(probe_radius_nm) ** 2
    return int(np.count_nonzero(np.min(dist2, axis=1) < r2))


def cavity_occupancy_diagnostics(
    counts,
    wet_min_waters: int = CAVITY_WET_MIN_WATERS,
) -> Dict[str, Any]:
    """把一条轨迹的逐帧空腔水数压成验收用的结构性指标。

    `n_wet_dry_transitions` 是最关键的一项：它 > 0 才说明这条轨迹**真的**在
    湿态和干态之间来回过。如果湿起点的 walker 全程 wet_fraction=1、干起点的
    全程 =0、两边转换次数都是 0，那就是两个互不连通的模态各自被采了一遍——
    此时 MBAR 报出来的 σ 再小也不可信，必须由 `endpoint_wet_dry_hysteresis_gate`
    判失败。
    """
    arr = np.asarray(counts, dtype=np.float64).ravel()
    if arr.size == 0:
        return {
            "n_frames": 0, "mean_cavity_waters": None, "max_cavity_waters": None,
            "wet_fraction": None, "dry_fraction": None,
            "n_wet_dry_transitions": None, "wet_min_waters": int(wet_min_waters),
        }
    wet = arr >= float(wet_min_waters)
    transitions = int(np.count_nonzero(wet[1:] != wet[:-1])) if wet.size > 1 else 0
    return {
        "n_frames": int(arr.size),
        "mean_cavity_waters": float(np.mean(arr)),
        "max_cavity_waters": float(np.max(arr)),
        "wet_fraction": float(np.mean(wet)),
        "dry_fraction": float(1.0 - np.mean(wet)),
        "n_wet_dry_transitions": transitions,
        "wet_min_waters": int(wet_min_waters),
    }


def _independent_endpoint_bank_dir(checkpoint_dir: str, stage_type: str) -> str:
    return os.path.join(checkpoint_dir, "independent_endpoint", str(stage_type))


def _independent_endpoint_record_path(
    bank_dir: str, global_k: int, mode: str, walker: int
) -> str:
    return os.path.join(bank_dir, f"state_{int(global_k)}_{mode}_w{int(walker)}.npz")


# ============================================================================
# [2026-08-31] 独立端点库 manifest 的**身份 / 预算**二分
# ----------------------------------------------------------------------------
# 旧实现把整个 manifest 拿去做全字典 `==`，任何一项不符就 `shutil.rmtree(bank_dir)`。
# 后果：想给某个态**补采**（加 walker）时，`n_walkers_per_mode` 一变就整库被删 ——
# 而那正是"沿用已有记录、只追加新记录"这件事本身要做的。真 bug。
#
# 二分依据：
#   · **身份**（identity）决定每条记录**采的是哪个分布**。改了 ⇒ 已有记录不再描述
#     当前要求的分布 ⇒ 必须清库重采。
#   · **预算**（budget）只决定**有多少条记录**。用全新、不碰撞的种子追加同分布的
#     记录是合法的，是同一个估计量加样本，不该清库。
#
# `n_walkers_per_mode` 归预算：`_independent_endpoint_seed` 是
# `base + 97*global_k + 10007*mode_index + 1009*walker`，**与 n_walkers_per_mode 无关**，
# 所以 walker 2 拿到的是全新不碰撞的种子，walker 0/1 的记录不受任何扰动。
# `init_modes` 也归预算：这样"这次没找到湿盆、只跑干起点"不会连坐删掉已有的干记录
# （4W53 实测 wet_basin_found=False，旧口径下每次都要连干记录一起重采）。
#
# ⚠️ 预算只允许**增长**。缩小（例如从 3 walker 退回 2）意味着当前请求覆盖不了盘上
# 已有的记录，此时不清库、只按请求的子集取用 —— 见 `_independent_endpoint_budget_ok`。
#
# 向后兼容：本次**不新增也不重命名任何键**，只是把已有键分成两类。旧的扁平
# manifest 因此在身份字段上原样匹配，拆分本身不会删掉任何已有 bank。
# ============================================================================
_INDEPENDENT_ENDPOINT_BUDGET_KEYS = frozenset({"n_walkers_per_mode", "init_modes"})


def _independent_endpoint_identity(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """取 manifest 的身份部分（决定"采的是哪个分布"的那些键）。"""
    return {
        k: v for k, v in manifest.items()
        if k not in _INDEPENDENT_ENDPOINT_BUDGET_KEYS
    }


def build_independent_endpoint_manifest(
    stage_type: str,
    state_indices: Sequence[int],
    common_system_xml: str,
    cv_xmls: Sequence[str],
    temperature_K: float,
    sample_interval: int,
    sample_steps: int,
    burn_in_steps: int,
    n_walkers_per_mode: int,
    platform_name: str,
    cavity_probe_radius_nm: float,
    cavity_wet_min_waters: int,
    init_modes: Optional[Sequence[str]] = None,
    minimize_iterations: int = 2_000,
    integrator_seed_base: Optional[int] = None,
    velocity_seed_base: Optional[int] = None,
    seed_identity: Optional[Dict[str, Any]] = None,
    coion_identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """独立端点采样库的内容指纹。

    任何一项不匹配 → 整个目录判定不可信、全部重采（与 fixed-H 探针库同一套
    "全对才信、错一项就整体拒绝"哲学）。`common_system_xml` 的哈希覆盖了 ACE
    softcore 形式、PME/cutoff/switching 以及"有没有 Group-4 WCA"——所以把
    WCA 重新塞回生产动力学会立刻让所有已采数据失效，不可能被静默复用。
    """
    return {
        "kind": "independent_endpoint_states",
        "protocol_version": int(INDEPENDENT_ENDPOINT_PROTOCOL_VERSION),
        "stage_type": str(stage_type),
        "state_indices": [int(x) for x in state_indices],
        "common_system_sha256": hashlib.sha256(
            common_system_xml.encode("utf-8")
        ).hexdigest(),
        "cv_xml_sha256": [
            hashlib.sha256(x.encode("utf-8")).hexdigest() for x in cv_xmls
        ],
        "temperature_K": float(temperature_K),
        "sample_interval": int(sample_interval),
        "sample_steps": int(sample_steps),
        "burn_in_steps": int(burn_in_steps),
        "n_walkers_per_mode": int(n_walkers_per_mode),
        # 实际使用的起点模态。干-only 与 干+湿 是两份不同的数据集，缓存不得混用。
        "init_modes": list(init_modes if init_modes is not None
                           else INDEPENDENT_ENDPOINT_INIT_MODES),
        "platform_name": str(platform_name),
        "cavity_probe_radius_nm": float(cavity_probe_radius_nm),
        "cavity_wet_min_waters": int(cavity_wet_min_waters),
        # 起始构型的制备方式变了，已采数据就不该被复用。
        "minimize_iterations": int(minimize_iterations),
        "wet_seeding_scheme": "wet_cavity_ladder_v1",
        # 🔑 [P1] 种子身份进指纹：换了 repeat_seed 就必须重采，不能复用旧 bank。
        "integrator_seed_base": (
            int(integrator_seed_base) if integrator_seed_base is not None else None
        ),
        "velocity_seed_base": (
            int(velocity_seed_base) if velocity_seed_base is not None else None
        ),
        "seed_identity": seed_identity,
        "coion_identity": coion_identity,
    }


def _independent_endpoint_manifest_matches(bank_dir: str, expected: Dict[str, Any]) -> bool:
    """**身份**是否一致。不一致 ⇒ 已有记录描述的不是当前要求的分布 ⇒ 必须清库。

    只比身份部分；预算差异（walker 数、init_modes）交给
    `_independent_endpoint_budget_ok` 单独判断，见上方二分说明。
    读不出/缺文件一律判不匹配（fail-closed）。
    """
    stored = _read_independent_endpoint_manifest(bank_dir)
    if stored is None:
        return False
    return _independent_endpoint_identity(stored) == _independent_endpoint_identity(
        expected
    )


def _read_independent_endpoint_manifest(bank_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(bank_dir, "manifest.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded if isinstance(loaded, dict) else None
    except Exception:
        return None


def _independent_endpoint_budget_delta(
    bank_dir: str, expected: Dict[str, Any]
) -> Dict[str, Any]:
    """身份已匹配的前提下，报告预算差异。**不**决定是否清库。

    `grew` 为真表示这次请求的记录数多于盘上已有的（合法追加）；`shrank` 为真表示
    请求覆盖不了已有记录 —— 此时同样不清库，只按请求的子集取用，多出来的记录
    留在盘上当孤儿（既不销毁数据，也不静默把陈旧数据算进结果）。
    """
    stored = _read_independent_endpoint_manifest(bank_dir) or {}
    old_w = stored.get("n_walkers_per_mode")
    new_w = expected.get("n_walkers_per_mode")
    old_modes = list(stored.get("init_modes") or [])
    new_modes = list(expected.get("init_modes") or [])
    return {
        "n_walkers_per_mode_stored": old_w,
        "n_walkers_per_mode_requested": new_w,
        "init_modes_stored": old_modes,
        "init_modes_requested": new_modes,
        "grew": bool(
            (old_w is not None and new_w is not None and int(new_w) > int(old_w))
            or (set(new_modes) - set(old_modes))
        ),
        "shrank": bool(
            (old_w is not None and new_w is not None and int(new_w) < int(old_w))
            or (set(old_modes) - set(new_modes))
        ),
    }


# ============================================================================
# [2026-08-31] 湿空腔种子搜索的**落盘 + 种子身份**（0831issue.md P1 / HANDOFF D3）
# ----------------------------------------------------------------------------
# 旧实现两个毛病，一次修法解决两面：
#
#   (a) **随机性**：`prepare_wet_cavity_seed` 的两个 seed 默认值是写死常量
#       （`INDEPENDENT_ENDPOINT_*_SEED_BASE - 1`），而唯一调用方从不传参。于是
#       **不同 repeat_seed 产生逐字节相同的 50 万步随机流** —— 各 repeat 的湿盆
#       起始构型完全相同、湿臂样本跨 repeat 相关，重复间离散度被系统性低估。
#       它也不进任何 manifest，违背 EXP-019 种子契约。
#       同一作业实测过结论翻转（12:41 找到、16:19 没找到）。
#
#   (b) **成本**：该搜索在 manifest 校验**之前**无条件执行，且结论不落盘。所以
#       每次 resume 都要重跑 50 万步，即使 bank 完全可复用。
#
# 落盘 + key 带种子同时解决：命中缓存就不重跑（治 b），换种子必然不命中、
# 重新搜索（治 a——绝不让新种子继承上一颗种子"永远是干的"结论）。
#
# ⚠️ **负结论也必须落盘**。只存"找到了"会让"没找到"每次都重搜 50 万步；而如果
# 不把种子放进 key，负结论又会被错误地跨种子继承——那等于把一次随机搜索的失败
# 固化成体系的物理属性。两者缺一不可。
#
# key 只包含搜索**实际依赖**的量。刻意**不含** `n_walkers_per_mode` / `init_modes`：
# 湿种子是一个与 walker 数无关的起始构型，把预算塞进 key 只会造成无谓重搜。
# 缓存目录与 bank 目录**分开**：bank 在身份不符时会被 rmtree，而湿搜索的依赖
# 集合与 bank 的身份集合不同（例如 walker 数变化不该让湿搜索失效）。
# ============================================================================
WET_CAVITY_SEED_CACHE_PROTOCOL_VERSION = 1


def build_wet_cavity_seed_key(
    stage_type: str,
    common_system_xml: str,
    cv_xml_decoupled: str,
    temperature_K: float,
    platform_name: str,
    equilibration_steps: int,
    check_interval: int,
    cavity_probe_radius_nm: float,
    cavity_wet_min_waters: int,
    integrator_seed: int,
    velocity_seed: int,
    seed_identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """湿空腔种子搜索的身份。**种子在里面**——换 repeat_seed 必须重新搜索。"""
    return {
        "kind": "wet_cavity_seed_search",
        "protocol_version": int(WET_CAVITY_SEED_CACHE_PROTOCOL_VERSION),
        "stage_type": str(stage_type),
        "common_system_sha256": hashlib.sha256(
            common_system_xml.encode("utf-8")
        ).hexdigest(),
        "cv_xml_decoupled_sha256": hashlib.sha256(
            cv_xml_decoupled.encode("utf-8")
        ).hexdigest(),
        "temperature_K": float(temperature_K),
        "platform_name": str(platform_name),
        "equilibration_steps": int(equilibration_steps),
        "check_interval": int(check_interval),
        "cavity_probe_radius_nm": float(cavity_probe_radius_nm),
        "cavity_wet_min_waters": int(cavity_wet_min_waters),
        "integrator_seed": int(integrator_seed),
        "velocity_seed": int(velocity_seed),
        "seed_identity": seed_identity,
    }


def load_wet_cavity_seed_cache(
    cache_dir: str, expected_key: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """命中则返回与 `prepare_wet_cavity_seed` 同形状的结果，否则 None（fail-closed）。

    正负结论都能命中：`reached_wet=False` 也是一个有效的、可复用的结论 —— 只要
    key（含种子）没变。key 变了一律不命中，绝不跨种子继承。
    """
    key_path = os.path.join(cache_dir, "wet_seed_key.json")
    npz_path = os.path.join(cache_dir, "wet_seed_state.npz")
    if not (os.path.isfile(key_path) and os.path.isfile(npz_path)):
        return None
    try:
        with open(key_path, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
        if stored != expected_key:
            return None
        with np.load(npz_path, allow_pickle=False) as data:
            reached = bool(data["reached_wet"])
            _diag_raw = str(data["cavity_diagnostics_json"])
            result: Dict[str, Any] = {
                "reached_wet": reached,
                "cavity_water_counts": [int(x) for x in np.asarray(data["counts"])],
                # ⚠️ `cavity_diagnostics` 必须一起落盘/还原。它被
                # `abfe_pipeline` 的 `wet_seed_cavity` 与本文件的
                # `wet_seed_diagnostics` 读取，而两处都用 `.get()` —— 不落盘不会崩，
                # 但**缓存命中时会静默记成 None**，于是续跑产出的证据比首跑少。
                # 那正是"改动让另一个字段开始说谎"这类 bug。
                "cavity_diagnostics": (
                    json.loads(_diag_raw) if _diag_raw not in ("", "null") else None
                ),
                "equilibration_steps": int(data["equilibration_steps"]),
                "dynamics_hamiltonian": str(data["dynamics_hamiltonian"]),
                "positions": (
                    np.asarray(data["positions_nm"]) * unit.nanometer
                ),
                "box_vectors": np.asarray(data["box_nm"]) * unit.nanometer,
                "cache": {"hit": True, "protocol_version": int(
                    WET_CAVITY_SEED_CACHE_PROTOCOL_VERSION
                )},
            }
        return result
    except Exception:
        return None


def save_wet_cavity_seed_cache(
    cache_dir: str, key: Dict[str, Any], result: Dict[str, Any]
) -> None:
    """把搜索结论（含负结论）落盘。先写 npz、后写 key，避免命中半成品。"""
    os.makedirs(cache_dir, exist_ok=True)
    pos = result.get("positions")
    box = result.get("box_vectors")
    pos_nm = np.asarray(
        pos.value_in_unit(unit.nanometer) if hasattr(pos, "value_in_unit") else pos,
        dtype=np.float64,
    )
    box_nm = np.asarray(
        box.value_in_unit(unit.nanometer) if hasattr(box, "value_in_unit") else box,
        dtype=np.float64,
    )
    # [0831issue P2] 同上：原子写，避免截断的 npz 卡死后续 resume。
    _atomic_save_npz(
        os.path.join(cache_dir, "wet_seed_state.npz"),
        reached_wet=bool(result.get("reached_wet")),
        counts=np.asarray(result.get("cavity_water_counts") or [], dtype=np.int64),
        equilibration_steps=int(result.get("equilibration_steps") or 0),
        dynamics_hamiltonian=str(result.get("dynamics_hamiltonian") or ""),
        cavity_diagnostics_json=json.dumps(
            result.get("cavity_diagnostics"), sort_keys=True, default=str
        ),
        positions_nm=pos_nm,
        box_nm=box_nm,
    )
    _atomic_write_json(os.path.join(cache_dir, "wet_seed_key.json"), key)


def prepare_wet_cavity_seed(
    topology,
    common_system_xml: str,
    cv_xml_decoupled: str,
    positions,
    box_vectors,
    temperature,
    platform_name: str,
    ligand_heavy_idx: np.ndarray,
    water_oxygen_idx: np.ndarray,
    equilibration_steps: int = 500_000,
    check_interval: int = 5_000,
    cavity_probe_radius_nm: float = CAVITY_PROBE_RADIUS_NM,
    cavity_wet_min_waters: int = CAVITY_WET_MIN_WATERS,
    integrator_seed: int = INDEPENDENT_ENDPOINT_INTEGRATOR_SEED_BASE - 1,
    velocity_seed: int = INDEPENDENT_ENDPOINT_VELOCITY_SEED_BASE - 1,
) -> Dict[str, Any]:
    """在**完全解耦**态上长平衡出一个"湿空腔"起始构型。

    这是湿起点 walker 的来源。在 λ_vdw=0 上配体与环境没有相互作用，水本来就
    可以自由进入配体占据的体积——所以这一步是在一个水**能**进空腔的哈密顿量
    上等它进去，而不是靠重加权去"造"那个构型。返回的构型随后被喂给**所有**
    端点态（包括还有相互作用的那些）当作湿起点。

    动力学同样不含 Group-4 WCA（`require_group4=False` 会硬断言）——防护壳的
    全部作用就是把水挡在外面，用它来平衡湿空腔是自相矛盾的。

    返回值里带上逐次检查的空腔水数序列：若到最后 `reached_wet` 仍是 False，
    调用方必须当作**失败**处理，而不是拿一个干构型冒充湿起点——那会让湿/干
    双起点退化成两组干起点，2σ 门看起来通过、实际什么都没验证。
    """
    temperature_q = (
        temperature if hasattr(temperature, "value_in_unit")
        else float(temperature) * unit.kelvin
    )
    simulation = _build_fixed_state_simulation(
        topology=topology,
        system_xml=common_system_xml,
        cv_xml=cv_xml_decoupled,
        require_group4=False,
        platform_name=platform_name,
        temperature_q=temperature_q,
        positions=positions,
        box_vectors=box_vectors,
        integrator_seed=int(integrator_seed),
        velocity_seed=int(velocity_seed),
    )
    try:
        counts = []
        steps_done = 0
        best_state = None
        while steps_done < int(equilibration_steps):
            chunk = min(int(check_interval), int(equilibration_steps) - steps_done)
            guarded_step(simulation, int(chunk), "湿腔种子预平衡")
            steps_done += chunk
            state = simulation.context.getState(getPositions=True, getVelocities=True)
            pos_nm = np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=np.float64,
            )
            box_nm = _box_vectors_to_nm_array(state.getPeriodicBoxVectors())
            n_wat = count_cavity_waters(
                pos_nm, box_nm, ligand_heavy_idx, water_oxygen_idx,
                probe_radius_nm=cavity_probe_radius_nm,
            )
            counts.append(int(n_wat))
            # 保留**最湿**的那一帧，而不是最后一帧：最后一帧可能恰好处在水
            # 短暂退出的瞬间，用它当湿起点会白白浪费这次长平衡。
            if best_state is None or n_wat >= max(counts[:-1] or [0]):
                if n_wat >= int(cavity_wet_min_waters) or best_state is None:
                    best_state = state
        diagnostics = cavity_occupancy_diagnostics(counts, cavity_wet_min_waters)
        reached = bool(counts) and max(counts) >= int(cavity_wet_min_waters)
        final = best_state if best_state is not None else simulation.context.getState(
            getPositions=True, getVelocities=True
        )
        return {
            "reached_wet": reached,
            "positions": final.getPositions(),
            "box_vectors": final.getPeriodicBoxVectors(),
            "cavity_water_counts": [int(x) for x in counts],
            "cavity_diagnostics": diagnostics,
            "equilibration_steps": int(steps_done),
            "dynamics_hamiltonian": "U_common_plus_single_cv_int_no_wca",
        }
    finally:
        del simulation
        gc.collect()


def _independent_endpoint_seed(global_k: int, mode: str, walker: int, base: int) -> int:
    # 用**历史全集**取索引：生产已不跑 wet，但索引表必须与移除前一致，
    # 否则 dry 的种子会因为表变短而漂移（dry 仍是 0，这里显式钉住）。
    mode_index = INDEPENDENT_ENDPOINT_ALL_INIT_MODES.index(str(mode))
    return int(
        base
        + _INDEPENDENT_ENDPOINT_SEED_STRIDE * int(global_k)
        + _INDEPENDENT_ENDPOINT_MODE_STRIDE * int(mode_index)
        + _INDEPENDENT_ENDPOINT_WALKER_STRIDE * int(walker)
    )


def run_independent_endpoint_states(
    topology,
    common_system_xml: str,
    ibs_wrapper,
    state_indices: Sequence[int],
    dry_seed: Dict[str, Any],
    wet_seed: Dict[str, Any],
    temperature,
    platform_name: str,
    checkpoint_dir: str,
    stage_type: str,
    ligand_indices,
    global_state_indices: Optional[Sequence[int]] = None,
    lambdas_vdw_for_states: Optional[Sequence[float]] = None,
    # 🔑 [P1] 种子身份必须由调用方（pipeline 的 _seed_for / seed_ledger）派生并
    # 传入。写死常量基数意味着：不同 repeat_seed 的两次运行会产生**逐字节相同**的
    # 端点随机流（等于把独立重复变成同一条轨迹跑两遍），而且改了 repeat_seed 之后
    # 旧的采样库仍然指纹匹配、被直接复用。两者都必须堵死。
    integrator_seed_base: Optional[int] = None,
    velocity_seed_base: Optional[int] = None,
    seed_identity: Optional[Dict[str, Any]] = None,
    burn_in_steps: int = 100_000,
    sample_steps: int = 500_000,
    sample_interval: int = 1_000,
    n_walkers_per_mode: int = 2,
    cavity_probe_radius_nm: float = CAVITY_PROBE_RADIUS_NM,
    cavity_wet_min_waters: int = CAVITY_WET_MIN_WATERS,
    minimize_iterations: int = 2_000,
    coion_identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """为 `state_indices` 里的每一个 λ 态建立**真正独立**的固定-λ 生产轨迹。

    每个 (态, 起点模态, walker) 组合都有自己的 System、自己的 Context、自己的
    平衡和自己的采样——没有任何一条轨迹的构型是从别的态重加权出来的。这是
    与 IBS "一条窗口轨迹重加权出全窗"的根本区别，也是本模块存在的全部理由。

    `common_system_xml` 必须已经移除 Group 1（IBS 混合偏置）与 Group 4
    （λ-WCA 防护壳），即 `_serialize_ibs_common_system` 的产物；
    `_build_fixed_state_simulation(require_group4=False)` 会在建每一个 Context
    时硬断言系统里不存在 Group 4，所以"生产里还藏着 WCA"这件事不可能悄悄
    发生——它会直接 RuntimeError。

    同一时刻最多只有一个动力学 Context 存活（外加一个共享的轻量能量探针
    Context）：每个 walker 建好、跑完、落盘后立即释放。

    **湿空腔阶梯（wet cavity ladder）**：湿构型是在**完全解耦**态上平衡出来的，
    空腔里坐着好几个水。把它直接丢给一个强耦合的 λ 态，水正好压在完全显形的
    配体上——2026-08-29 实测在 λ_vdw=1.0 上 `Particle coordinate is NaN`，
    minimize 2000 步也救不回来（水氧到碳只有零点几埃，LJ 能量太大，L-BFGS 出不去）。
    加大最小化步数是治标；真正的问题是**在 λ_vdw≈1 上"湿空腔"根本不是一个亚稳态**，
    硬塞一个湿构型不是物理检验，只是碰撞。

    所以湿 walker 按 λ_vdw **从小到大**（最解耦 → 最耦合）依次跑，每个态用**上一个
    更解耦的态**的末构型作起点（只取 walker 0 的末构型，保证阶梯与 walker 数无关）。
    相邻两态的哈密顿量差别很小，湿空腔要么被平滑地带上去，要么在某个 λ 上自然干掉——
    后者正是**物理答案**，会如实反映在 `per_walker_cavity` 的 wet_fraction 上。

    干 walker **不**走阶梯：干构型对任何 λ 都相容（配体占着体积本来就是耦合端的常态），
    每个干 walker 都从同一个独立的干种子起。所以湿/干双起点门比的仍然是"两条来源
    完全不同的路径"——阶梯只影响湿侧的**起始构型**，每个 walker 的 Context、速度种子、
    burn-in 和采样都还是各自独立的。

    ⚠️ **每个 walker 在 burn-in 之前还会先在它自己的态上最小化**（
    `minimize_iterations`）。湿起点是在**完全解耦**态上平衡出来的，空腔里坐着
    好几个水；把那个构型直接丢给一个强耦合的 λ 态（λ_vdw→1），水正好压在完全
    显形的配体上，第一步积分就 `Particle coordinate is NaN`——2026-08-29 实测在
    λ_vdw=1.0 上确实炸了。这与 `_probe_vdw_window_fixed_overlap` 里记的那段
    live crash 是同一类错误（在一个势能面上弛豫、到另一个势能面上跑），只是方向
    相反。阶梯负责把跨度变小，最小化负责挤掉残余的不相容，两者缺一不可。

    返回的 `records` 里每条都带逐帧空腔水数；`per_walker_cavity` 汇总每个
    walker 自己的湿/干占比与转换次数——这是判断"湿/干两组是不是各自卡在自己
    的模态里"的第一手证据，`endpoint_wet_dry_hysteresis_gate` 要用它。
    """
    state_indices = [int(x) for x in state_indices]
    # `state_indices` 索引的是 `ibs_wrapper._int_cv_force_xmls`（即"这个窗口
    # 系统里的第几个 λ 态"）。当调用方只为端点块建了一个窗口系统时，这些是
    # **局部**索引，与整条 λ 路径上的全局索引不同——`global_state_indices` 用来
    # 记录后者，拼接时要靠它对齐接点。两者混用会让拼接在错误的 λ 上接合，而
    # 接错的自由能看起来完全正常。
    global_state_indices = (
        [int(x) for x in global_state_indices]
        if global_state_indices is not None else list(state_indices)
    )
    if len(global_state_indices) != len(state_indices):
        raise ValueError("global_state_indices 与 state_indices 长度必须一致")
    if len(state_indices) < 2:
        raise ValueError("独立端点采样至少需要两个 λ 态")
    if sorted(set(state_indices)) != sorted(state_indices):
        raise ValueError("state_indices 不能重复")
    if int(n_walkers_per_mode) < 1:
        raise ValueError("每个起点模态至少需要一个 walker")
    if sample_interval <= 0 or sample_steps < sample_interval:
        raise ValueError("独立端点采样长度不足一个 sample_interval")

    cv_xmls = list(getattr(ibs_wrapper, "_int_cv_force_xmls", []))
    if max(state_indices) >= len(cv_xmls):
        raise RuntimeError("IBS wrapper 缺少独立端点采样所需的 CV XML")

    temperature_q = (
        temperature if hasattr(temperature, "value_in_unit")
        else float(temperature) * unit.kelvin
    )
    ligand_heavy_idx = ligand_heavy_atom_indices(topology, ligand_indices)
    water_oxygen_idx = water_oxygen_indices(topology)
    if water_oxygen_idx.size == 0:
        raise RuntimeError(
            "拓扑里找不到任何水分子；空腔水结构诊断是本模块的验收前提，拒绝"
            "在没有结构证据的情况下产出端点自由能"
        )

    if not dry_seed or dry_seed.get("positions") is None:
        raise ValueError("缺少 dry 起点构型")
    # ------------------------------------------------------------------
    # 🔑 [2026-08-31] 湿起点是**条件性诊断**，不是前置条件。
    #
    # 这里曾经在 reached_wet=False 时直接 raise。那是错的：湿/干双起点是为溶剂腿
    # 的空腔灌水问题引入的遍历性检验手段，而 T4 L99A 这类**本来就干的埋藏疏水腔**
    # 根本不存在湿盆——要求必须找到它，逻辑上不成立，且会让整条腿崩掉
    # （实测：复合物腿 λ_vdw=0 平衡 1 ns，空腔水数 100 次检查全为 0）。
    #
    # 不能把「水」这个针对溶剂腿引入的概念，变成整个 ABFE 管线的普遍物理假设。
    # 真正通用的硬门与具体物理盆无关：raw ESS、top-1% 权重、独立 walker/种子
    # 一致性、BAR–MBAR 一致性、样本数与去相关要求。
    #
    # 现在的行为：构造不到湿起点就只跑干起点，并如实记录「该诊断未评估」。
    # 不 raise、不要求调用方预先知道体系是干腔还是湿腔。
    # ------------------------------------------------------------------
    wet_available = bool(
        wet_seed and wet_seed.get("positions") is not None
        and wet_seed.get("reached_wet") is not False
    )
    active_modes = tuple(INDEPENDENT_ENDPOINT_INIT_MODES) if wet_available else ("dry",)
    seeds = {"dry": dry_seed}
    if wet_available:
        seeds["wet"] = wet_seed

    bank_dir = _independent_endpoint_bank_dir(checkpoint_dir, stage_type)
    expected_manifest = build_independent_endpoint_manifest(
        stage_type=stage_type,
        state_indices=state_indices,
        common_system_xml=common_system_xml,
        cv_xmls=[cv_xmls[k] for k in state_indices],
        temperature_K=temperature_q.value_in_unit(unit.kelvin),
        sample_interval=sample_interval,
        sample_steps=sample_steps,
        burn_in_steps=burn_in_steps,
        n_walkers_per_mode=n_walkers_per_mode,
        platform_name=platform_name,
        cavity_probe_radius_nm=cavity_probe_radius_nm,
        cavity_wet_min_waters=cavity_wet_min_waters,
        init_modes=list(active_modes),
        minimize_iterations=minimize_iterations,
        integrator_seed_base=integrator_seed_base,
        velocity_seed_base=velocity_seed_base,
        seed_identity=seed_identity,
        coion_identity=coion_identity,
    )
    # 身份不符 ⇒ 已有记录描述的不是当前要求的分布 ⇒ 清库。
    # 预算变化（walker 数 / init_modes）⇒ **不清库**，只把 manifest 更新成新预算：
    # 种子公式与 n_walkers_per_mode 无关，新 walker 拿全新不碰撞的种子，
    # 已有记录逐字节不受扰动。见文件上方"身份 / 预算二分"。
    if not _independent_endpoint_manifest_matches(bank_dir, expected_manifest):
        if os.path.isdir(bank_dir):
            shutil.rmtree(bank_dir, ignore_errors=True)
        os.makedirs(bank_dir, exist_ok=True)
        _atomic_write_json(os.path.join(bank_dir, "manifest.json"), expected_manifest)
    else:
        _budget = _independent_endpoint_budget_delta(bank_dir, expected_manifest)
        if _budget["grew"] or _budget["shrank"]:
            print(
                "  [独立端点段] manifest 身份一致、预算变化："
                f"walker {_budget['n_walkers_per_mode_stored']}→"
                f"{_budget['n_walkers_per_mode_requested']}，"
                f"init_modes {_budget['init_modes_stored']}→"
                f"{_budget['init_modes_requested']}。"
                "沿用已有记录，只追加缺的；不清库。"
            )
        os.makedirs(bank_dir, exist_ok=True)
        _atomic_write_json(os.path.join(bank_dir, "manifest.json"), expected_manifest)
    os.makedirs(bank_dir, exist_ok=True)

    records: Dict[str, Dict[str, Any]] = {}
    evaluator = None
    n_all_states = len(cv_xmls)

    # ------------------------------------------------------------------
    # 湿起点必须沿 λ 阶梯走（见 docstring 的"湿空腔阶梯"一段）。
    # 干起点不需要：干构型对任何 λ 都是相容的（配体占着体积本来就是耦合端的
    # 常态），所以每个干 walker 都从同一个独立的干种子起。
    # ------------------------------------------------------------------
    if lambdas_vdw_for_states is not None:
        lam_for = {
            int(k): float(v)
            for k, v in zip(state_indices, lambdas_vdw_for_states)
        }
        wet_order = sorted(state_indices, key=lambda k: lam_for[int(k)])
    else:
        # 缺 λ 值时退回本仓库的约定：state_indices 升序 == λ_vdw 降序，
        # 所以倒序就是"从最解耦走向最耦合"。
        wet_order = list(reversed(state_indices))

    visit: List[Tuple[int, str, int]] = []
    for mode in active_modes:
        order = list(state_indices) if mode == "dry" else wet_order
        for gk in order:
            for w in range(int(n_walkers_per_mode)):
                visit.append((int(gk), str(mode), int(w)))

    # 每个模态"当前的"起始构型。干的自始至终不变；湿的沿阶梯逐态往前传。
    current_seed: Dict[str, Dict[str, Any]] = {"dry": dict(dry_seed)}
    if wet_available:
        current_seed["wet"] = dict(wet_seed)

    for global_k, mode, walker in visit:
        key = f"{global_k}|{mode}|{walker}"
        path = _independent_endpoint_record_path(bank_dir, global_k, mode, walker)
        # 🔑 [0831issue P2] 损坏容错：写入侧现在是原子写（见 _atomic_save_npz 的
        # 调用点），但**改动之前**留下的截断 npz 仍可能躺在盘上，而这里原来没有
        # try/except —— 每次 resume 都在 np.load 崩，且报错完全指不出该删哪个文件。
        # 现在把不可读的记录当"这一格还没算过"处理：删掉它、走下面的重算分支。
        # 这不会掩盖真实数据问题：这些 npz 是**可重算**的中间产物，重算一遍即可，
        # 且下游 build_independent_endpoint_manifest / 各道硬门照常校验内容。
        if os.path.isfile(path):
            try:
                with np.load(path, allow_pickle=False) as _probe:
                    _ = _probe["sampled_steps"]
                _record_readable = True
            except Exception as exc:
                print(
                    f"  [WARN] 独立端点 walker 记录不可读（{os.path.basename(path)}: {exc}）；"
                    "按未完成处理，删除后重算该格。"
                )
                try:
                    os.remove(path)
                except OSError:
                    pass
                _record_readable = False
        else:
            _record_readable = False
        if _record_readable:
            with np.load(path, allow_pickle=False) as data:
                records[key] = {
                    "u_cv_kj_mol": np.asarray(data["u_cv_kj_mol"]),
                    "volume_nm3": np.asarray(data["volume_nm3"]),
                    "cavity_waters": np.asarray(data["cavity_waters"]),
                    "energy_column_indices": [
                        int(x) for x in np.asarray(data["energy_column_indices"])
                    ],
                    "segments": [{
                        "burn_in_steps": int(burn_in_steps),
                        "sample_steps": int(data["sampled_steps"]),
                        "n_frames": int(np.asarray(data["volume_nm3"]).size),
                        "reason": "resumed_from_disk",
                    }],
                    "sampled_steps": int(data["sampled_steps"]),
                    "global_state": int(global_k),
                    "init_mode": str(mode),
                    "walker": int(walker),
                }
                # 续跑时也要把湿阶梯接上，否则后面那些更耦合的态又会拿到最
                # 解耦端的湿构型（也就是这次修掉的那个 NaN）。
                if mode == "wet" and walker == 0 and "final_positions_nm" in data:
                    current_seed["wet"] = {
                        "positions": np.asarray(data["final_positions_nm"]) * unit.nanometer,
                        "box_vectors": np.asarray(data["final_box_nm"]) * unit.nanometer,
                        "origin": f"wet_ladder_resumed_from_state_{global_k}",
                    }
            continue

        seed = current_seed[mode]
        sim = _build_fixed_state_simulation(
            topology=topology,
            system_xml=common_system_xml,
            cv_xml=cv_xmls[global_k],
            require_group4=False,
            platform_name=platform_name,
            temperature_q=temperature_q,
            positions=seed["positions"],
            box_vectors=seed.get("box_vectors"),
            integrator_seed=_independent_endpoint_seed(
                global_k, mode, walker,
                INDEPENDENT_ENDPOINT_INTEGRATOR_SEED_BASE
                if integrator_seed_base is None else int(integrator_seed_base),
            ),
            velocity_seed=_independent_endpoint_seed(
                global_k, mode, walker,
                INDEPENDENT_ENDPOINT_VELOCITY_SEED_BASE
                if velocity_seed_base is None else int(velocity_seed_base),
            ),
        )
        try:
            if evaluator is None:
                evaluator = IBSSampler(
                    sim.context, n_all_states, temperature_q,
                    prefix=getattr(ibs_wrapper, "prefix", "abfe"),
                    ibs_wrapper=ibs_wrapper,
                )
                # 同 probe_adjacent_path_overlap_bank：evaluator 只用它自己的
                # 探针 Context，立刻丢掉对本 walker 动力学 Context 的引用，
                # 避免多个 Context 同时占显存。
                evaluator.context = None

            def _observe(frame_positions, frame_box):
                pos_nm = np.asarray(
                    frame_positions.value_in_unit(unit.nanometer)
                    if hasattr(frame_positions, "value_in_unit")
                    else frame_positions,
                    dtype=np.float64,
                )
                return count_cavity_waters(
                    pos_nm,
                    _box_vectors_to_nm_array(frame_box),
                    ligand_heavy_idx,
                    water_oxygen_idx,
                    probe_radius_nm=cavity_probe_radius_nm,
                )

            # 阶梯之后相邻两态的哈密顿量差别已经很小，但仍然先在**本态自己的**
            # 势能面上弛豫一下，把残余的不相容挤掉。
            if int(minimize_iterations) > 0:
                sim.minimizeEnergy(maxIterations=int(minimize_iterations))
            record = {
                "u_cv_kj_mol": None, "volume_nm3": None,
                "segments": [], "sampled_steps": 0,
            }
            try:
                _extend_state_trajectory(
                    sim, record, evaluator,
                    target_steps=int(sample_steps),
                    sample_interval=int(sample_interval),
                    burn_in_steps=int(burn_in_steps),
                    needs_burn_in=True,
                    segment_reason=f"independent_endpoint_{mode}_w{walker}",
                    frame_observer=_observe,
                )
            except openmm.OpenMMException as exc:
                raise RuntimeError(
                    f"独立端点采样在 λ 态 {global_k}（{mode} 起点, walker "
                    f"{walker}）上积分失败: {exc}。起始构型与本态的哈密顿量不相容。"
                    "湿起点已经沿 λ 阶梯逐态传递（wet ladder），若仍然炸，通常说明"
                    "相邻两个 λ 态之间跨度太大、或 minimize_iterations 不足。请加密"
                    "该段 λ 布点、增大 minimize_iterations，或把独立采样范围限制在"
                    "软核足够软的那一段。"
                ) from exc
            record["global_state"] = int(global_k)
            record["init_mode"] = str(mode)
            record["walker"] = int(walker)
            # evaluator 是用**全部**态建的，所以每帧能量含所有 λ 态的列。
            # 显式落盘这个布局，下游按物理 λ 索引取列，不靠位置约定。
            record["energy_column_indices"] = list(range(n_all_states))
            record["cavity_waters"] = np.asarray(
                record.pop("observer_values"), dtype=np.float64
            )
            final_state = sim.context.getState(getPositions=True)
            final_pos_nm = np.asarray(
                final_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=np.float64,
            )
            final_box_nm = _box_vectors_to_nm_array(
                final_state.getPeriodicBoxVectors()
            )
            # 🔑 [0831issue P2] 原子写：以前是 `np.savez(path, ...)` 直写目标路径，
            # 写入中途被 kill / 掉电会留下一份截断的 npz，之后**每次 resume 都在
            # np.load 那里崩**，必须人工删文件才能继续。同文件的 fixed-H 探针库早就
            # 用 _atomic_save_npz（temp+os.replace），这里口径统一。
            _atomic_save_npz(
                path,
                u_cv_kj_mol=record["u_cv_kj_mol"],
                volume_nm3=record["volume_nm3"],
                cavity_waters=record["cavity_waters"],
                energy_column_indices=np.asarray(
                    record["energy_column_indices"], dtype=np.int64
                ),
                sampled_steps=np.int64(record["sampled_steps"]),
                final_positions_nm=final_pos_nm,
                final_box_nm=final_box_nm,
            )
            records[key] = record
            # 只用 walker 0 的末构型往前传，保证阶梯是确定性的（walker 数量
            # 变化不改变后面每个态拿到的湿起点）。
            if mode == "wet" and walker == 0:
                current_seed["wet"] = {
                    "positions": final_state.getPositions(),
                    "box_vectors": final_state.getPeriodicBoxVectors(),
                    "origin": f"wet_ladder_from_state_{global_k}",
                }
        finally:
            del sim
            gc.collect()

    per_walker_cavity = {
        key: dict(
            cavity_occupancy_diagnostics(rec["cavity_waters"], cavity_wet_min_waters),
            global_state=int(rec["global_state"]),
            init_mode=str(rec["init_mode"]),
            walker=int(rec["walker"]),
        )
        for key, rec in records.items()
    }
    return {
        "protocol_version": int(INDEPENDENT_ENDPOINT_PROTOCOL_VERSION),
        "state_indices": list(state_indices),
        "global_state_indices": list(global_state_indices),
        "records": records,
        "per_walker_cavity": per_walker_cavity,
        "bank_dir": bank_dir,
        "dynamics_hamiltonian": "U_common_plus_single_cv_int_no_wca",
        "wca_present_in_production_dynamics": False,
        "n_walkers_per_mode": int(n_walkers_per_mode),
        "init_modes": list(active_modes),
        # 湿盆是否在预算内被观察到。False 时湿/干门是**未评估**，不是失败。
        "wet_basin_found": bool(wet_available),
        "wet_seed_diagnostics": (wet_seed.get("cavity_diagnostics") if wet_seed else None),
        "wet_seed_equilibration_steps": (
            wet_seed.get("equilibration_steps") if wet_seed else None),
    }


def _reduced_energies_for_record(
    record: Dict[str, Any],
    state_indices: Sequence[int],
    kt: float,
    lrc_coeff: Optional[np.ndarray],
) -> np.ndarray:
    """一条独立轨迹在 `state_indices` 全部态上的约化能量 (n_frames, K)。

    列的物理身份由 `record["energy_column_indices"]` **显式**声明，绝不靠
    "列号恰好等于全局 λ 索引"这种隐式约定去猜——那种约定在能量矩阵只存了窗口
    子集时会静默取错列，而取错列的自由能看起来完全正常。

    LRC 用 `lrc_coeff[k] / V(t)` 逐帧加上——与 `IBSSampler.collect_energies` 的
    `target_energies = softcore_energies + lrc_energies` 是同一个式子、同一组
    系数（`ibs_wrapper.lj_tail_lrc_coeff_kj_mol`）。这里绝不另算一份 LRC，
    否则独立端点段和 IBS 段就不是同一个热力学量，拼接无意义。
    """
    columns = [int(c) for c in record["energy_column_indices"]]
    position_of = {c: i for i, c in enumerate(columns)}
    missing = [int(k) for k in state_indices if int(k) not in position_of]
    if missing:
        raise KeyError(
            f"独立端点轨迹的能量矩阵没有这些 λ 态的列: {missing}；"
            f"该记录记录的列布局是 {columns}"
        )
    idx = np.asarray([position_of[int(k)] for k in state_indices], dtype=int)
    e_kj = np.asarray(record["u_cv_kj_mol"], dtype=np.float64)[:, idx].copy()
    if lrc_coeff is not None:
        volume = np.asarray(record["volume_nm3"], dtype=np.float64)
        if volume.shape[0] != e_kj.shape[0]:
            raise RuntimeError("独立端点轨迹的体积序列与能量帧数不一致")
        # 🔑 [0831issue P2] LRC 系数按**物理 λ 态**编址，不能用 `idx`。
        # `idx` 是"该记录的列布局中的位置"（position_of[k]），只有在布局恰好是恒等
        # 排列时才与物理态号相等。当前唯一的生产者确实写恒等布局（range(n_all_states)），
        # 所以这处此前不可达；但只要将来有任何非恒等布局落盘，`lrc_coeff[idx]` 就会
        # 把尾项系数静默配到错的 λ 态上——自由能会错而不会报错，正是本函数 docstring
        # 明令禁止的那类"隐式列号约定"。这里显式用物理态号取系数。
        lrc_arr = np.asarray(lrc_coeff, dtype=np.float64)
        lrc_idx = np.asarray([int(k) for k in state_indices], dtype=int)
        if lrc_idx.size and int(np.max(lrc_idx)) >= lrc_arr.shape[0]:
            raise IndexError(
                f"LJ 尾项系数只有 {lrc_arr.shape[0]} 个态，取不到 λ 态 "
                f"{[int(k) for k in state_indices]}；拒绝把尾项配到错的 λ 上"
            )
        e_kj = e_kj + lrc_arr[lrc_idx][None, :] / volume[:, None]
    return e_kj / float(kt)


# ============================================================================
# [ENDPOINT_CAVITY_SAMPLING_GATE_PROTOCOL_VERSION=1] 独立端点段的"慢坐标到底被
# 采到了没有"硬门
# ----------------------------------------------------------------------------
# 本模块存在的全部理由（docs/archive/STAGE2_ROOT_CAUSE_2026-08-28.md §3.3）是：IBS 每窗口单轨迹
# 重加权造不出"水塌进空腔"这个构型，所以端点态改成每个 λ 各自独立采样。
# 那么"这条独立轨迹到底有没有采到空腔的湿/干转换"就是该段结果可信度的**前提**，
# 而不是一项可选诊断。
#
# 旧实现的漏洞：`_decorrelate_independent_record` 只在 `std(cavity) > 0` 时才把空腔
# 序列放进候选表。std == 0 时候选表里只剩能量序列，"取更差者"这道保险无从取起，
# 该记录拿到能量序列的满额样本数——而 std == 0 恰恰就是"walker 在慢坐标上一步没动"
# 的签名。最需要保险的情形，保险自己关了。
#
# 4W53 output_v2 实测（n_k=[342, 18, 962, 508]，阈值 20）：
#     state_1 w0/w1  cav_g 65.2 / 50.2   转变 141/68   → 8 + 10   ← 唯一真在采慢模态
#     state_2 w0/w1  cav_g  1.0 /  1.0   转变   4/ 2   → 462 + 500
#     state_3 w0     cav_g 65.8          转变  90      → 8
#     state_3 w1     cav_std = 0.000     转变   0      → 500（保险静默失效）
# 也就是说门卡在唯一一个诚实报出慢自相关的态上，却放行了三个慢模态被冻住的态。
# 那三个大 n_k 不携带"慢模态是否平衡"的任何信息。
#
# 处置：fail-closed，且**不提供配置开关**。
# 「这条轨迹在慢坐标上没动」有两种可能：(a) walker 卡住没爬出来；(b) 该 λ 下真实
# 平衡态确实单相（例如 T4 L99A 那个包埋疏水腔本身就以脱湿著称）。**现有数据分不清
# 这两者**，而它们对 ΔG 的含义完全相反。分不清的时候不能默认成 (b) 放行。
# 要判定 (b) 需要独立证据（双起点湿/干对照、或该 λ 的解析/文献参考），那是一次
# 显式的协议决定，不该由一个 bool 配置项顺手完成——加开关只会变成"为了让门变绿"。
# ============================================================================
ENDPOINT_CAVITY_SAMPLING_GATE_PROTOCOL_VERSION = 1


def _decorrelate_independent_record(
    record: Dict[str, Any],
    reduced: np.ndarray,
    own_column: int,
    min_frames_for_subsampling: int = 20,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """对一条独立轨迹做逐 segment 时间去相关，并在两个候选序列里取**更差**的那个。

    候选一是这个态自己的约化势能；候选二是逐帧空腔水数。这不是可有可无的
    保险：本模块要采的慢模态是**结构性**的（水进出配体空腔），而
    docs/archive/STAGE2_ROOT_CAUSE_2026-08-28.md §3.2 已经实测出这个模态在能量上几乎不
    可见（window 2 相邻 <ΔU> 仅 0.4~0.6 kT）。⚠️ [2026-09-02] 原文这里还写着
    该窗口"贡献 ~21 kJ/mol 的 TΔS"——那个幅度是**壳还在时**推出来的，已随
    λ-WCA 壳退役作废（见 docs/archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md）。
    但本函数的取舍与幅度无关：慢模态在能量上不可见这一点仍然成立，所以只用
    能量序列估自相关时间会系统性低估 g，从而高估独立样本数——正是"稳定地
    收敛到错值"那类失效的统计学版本。取两者中样本更少的那个，宁可保守。
    """
    candidates = {"reduced_potential": np.asarray(reduced[:, own_column], dtype=np.float64)}
    # 🔑 [ENDPOINT_CAVITY_SAMPLING_GATE_PROTOCOL_VERSION=1, 2026-08-31] 空腔候选
    # 是否进候选表，必须**显式记录**，不能像以前那样静默跳过。
    # 旧写法是 `if size 对 and std > 0: candidates[...] = cavity`，两种落空都不留痕：
    #   · `std == 0` → 空腔序列全平 → 候选表里只剩能量序列 → "取更差者"这道保险
    #     **无从取起**，该记录直接拿到能量序列的满额样本数。
    #   · 而 std == 0 恰恰就是"walker 在慢坐标上一步没动"的签名 —— 最需要这道保险的
    #     情形，保险自己关了。本函数 docstring 自己写着只用能量序列会**系统性低估 g**。
    # 4W53 实测：`state_3_dry_w1` 的 cavity std 恰为 0.000、湿/干转变 0 次，
    # 却报出 n_decorrelated=500（满额）；同一个态的 w0 转变 90 次、g≈65.8、只有 8 个。
    # 于是 n_k=[342, 18, 962, 508] 里 508 的绝大部分来自一条**没采到慢模态**的轨迹。
    cavity_status = "absent"
    cavity = record.get("cavity_waters")
    if cavity is not None:
        cavity = np.asarray(cavity, dtype=np.float64).ravel()
        if cavity.size != reduced.shape[0]:
            cavity_status = "length_mismatch"
        elif not np.all(np.isfinite(cavity)):
            cavity_status = "nonfinite"
        elif float(np.std(cavity)) <= 0.0:
            # 不构造候选（对全平序列估自相关无意义），但**必须留痕**并让上层
            # fail-closed —— 不能把"没观察到该坐标变化"当成"该坐标已充分采样"。
            cavity_status = "degenerate_zero_variance"
        else:
            cavity_status = "used"
            candidates["cavity_water_count"] = cavity

    best_key = None
    best_indices = None
    report: Dict[str, Any] = {}
    for key, series in candidates.items():
        indices, g_list, short_segs = _decorrelate_per_segment(
            series, record["segments"],
            min_frames_for_subsampling=min_frames_for_subsampling,
        )
        report[key] = {
            "n_decorrelated": int(indices.size),
            "statistical_inefficiency": [float(x) for x in g_list],
            "short_segments_diagnostic_only": short_segs,
        }
        if best_indices is None or indices.size < best_indices.size:
            best_indices, best_key = indices, key
    report["selected_series"] = best_key
    report["selected_reason"] = "两个候选序列里去相关后样本更少（自相关更慢）的那个"
    # 🔑 [2026-08-31] ACF 型 g 用在**稀有事件指示量**上会系统性虚高样本数，必须把
    # 转变次数一起报出来供对照。4W53 实测（转变次数 vs pymbar g 的 Pearson r = 0.851）：
    #     转变 141 次 → g=62.76（惩罚最重）   转变 4 次 → g=1.00（几乎不惩罚）
    # **探索越充分 g 越大、有效样本越少；卡得越死 g 越小、被认定的独立样本越多。**
    # 原因是几乎恒为 0、偶发几个 1 的稀疏序列方差极小、自相关一个 lag 内衰完，
    # ACF 估计器看不到相关性。于是 state_2（500 帧只变 4 次）拿到 n_dec=500。
    #
    # `cavity_mode_observation_bound` 是一个**严格上界**：双态坐标被 k 次变化切成 k+1
    # 段、段内恒定，所以关于"空腔是湿还是干"至多有 k+1 个独立观测。
    #
    # ⚠️⚠️ **它约束的不是 `n_k`，别拿它 cap 任何东西。** 名字刻意不叫
    # `n_decorrelated_*`，就是为了避免这个误用：
    #   · 本上界约束的是**该记录对「空腔坐标」的独立观测数**；
    #   · 而 MBAR 的 `n_k` 计的是**约化势能**的独立样本，相关自由度是全部坐标，
    #     不只是湿/干指示量。
    # 两者混起来会得到一个「对单相态假阳性、对能量估计过严」的门。
    # 它的正确用途只有一个：**慢模态覆盖度诊断** —— "这条记录最多提供 k+1 个关于
    # 空腔是否湿的独立观测"。空腔水的 TΔS 是单系综机制所依赖的量
    # （docs/archive/STAGE2_ROOT_CAUSE_2026-08-28.md §3.2/§3.3）。
    # ⚠️ [2026-09-02 归因更正] 原文写"正是 **+32 kJ/mol 根因**所依赖的量"——
    # 那 +32 的 98% 已定罪为 λ-WCA 防护壳（见
    # docs/archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md）。这个诊断量
    # 仍然有意义（残余 ≈4% 的候选解释），只是不再对应那个幅度。
    #
    # ⚠️ 同样**只报告、不设门**：0 次转变既可能是"卡住"、也可能是"该 λ 真是单相"
    # （λ_vdw≈1 端配体完全占据空腔就是后者），**在单个 λ 态内部原理上分不清这两者**。
    # 拿它当门会把合法单相态一并拦掉 —— 与"std==0 一刀切"同一个错误。
    if cavity_status == "used" or cavity_status == "degenerate_zero_variance":
        _wet = (np.asarray(cavity, dtype=np.float64) > 0.0).astype(np.float64)
        _trans = int(np.sum(np.abs(np.diff(_wet)))) if _wet.size > 1 else 0
        report["cavity_transitions"] = _trans
        report["cavity_wet_fraction"] = float(_wet.mean()) if _wet.size else None
        report["cavity_mode_observation_bound"] = int(_trans + 1)
        _sel = report.get("cavity_water_count") or {}
        _acf_n = _sel.get("n_decorrelated")
        if _acf_n is not None:
            report["cavity_acf_exceeds_transition_bound"] = bool(
                int(_acf_n) > _trans + 1
            )
    report["cavity_series_status"] = cavity_status
    report["cavity_guard_active"] = bool(cavity_status == "used")
    if cavity_status != "used":
        report["cavity_guard_inactive_reason"] = {
            "absent": "记录里没有 cavity_waters 序列",
            "length_mismatch": "cavity_waters 长度与能量帧数不一致",
            "nonfinite": "cavity_waters 含非有限值",
            "degenerate_zero_variance": (
                "cavity_waters 方差为 0：这条轨迹在慢坐标（空腔水占据）上一步没动，"
                "因此无法从它估计该坐标的自相关时间。能量序列给出的样本数不能代表"
                "慢模态的独立样本数。"
            ),
        }[cavity_status]
    return best_indices, report


def solve_independent_endpoint_states(
    bank: Dict[str, Any],
    kt: float,
    lrc_coeff: Optional[np.ndarray] = None,
    init_modes: Optional[Sequence[str]] = None,
    min_decorrelated_samples: int = 20,
    min_frames_for_subsampling: int = 20,
) -> Dict[str, Any]:
    """把独立端点轨迹解成 λ 态之间的自由能曲线。

    与 IBS 的局部 TMBAR 有一个**结构性**区别：这里每一个物理态都有 `n_k > 0`
    的真实样本，不存在"一个采样分布 + 一堆零样本目标态"的单向重加权。所以
    `raw` 支撑度天然就是健康的，TARGET_SUPPORT_GATE 那道门在这条路径上是
    自然满足而不是被绕过——这正是修复的意义所在。

    `init_modes` 限定只用某一组起点（"wet" 或 "dry"）的 walker 求解，供
    `endpoint_wet_dry_hysteresis_gate` 做双起点对照。默认用全部。
    """
    state_indices = [int(x) for x in bank["state_indices"]]
    # 🔑 [2026-08-31] 全局 λ 身份必须出现在**每一条**返回路径上，失败分支也要带。
    # 之前只有成功路径带 `global_state_indices`，提前返回的失败分支只带局部
    # `state_indices`；`combine_ibs_and_independent_endpoint` 回落到局部索引后把
    # 接缝算成 0，于是**任何**端点求解失败都被伪装成 `combine_join_mismatch`
    # 索引错误，真实原因（样本不足等）完全看不到。
    global_state_indices = [
        int(x) for x in (bank.get("global_state_indices") or state_indices)
    ]
    K = len(state_indices)
    wanted_modes = (
        set(str(m) for m in init_modes) if init_modes is not None
        else set(INDEPENDENT_ENDPOINT_INIT_MODES)
    )
    position_of = {k: i for i, k in enumerate(state_indices)}

    blocks: List[List[np.ndarray]] = [[] for _ in range(K)]
    decorrelation_report: Dict[str, Any] = {}
    used_records: List[Dict[str, Any]] = []
    for key, record in sorted(bank["records"].items()):
        if str(record["init_mode"]) not in wanted_modes:
            continue
        own = position_of[int(record["global_state"])]
        reduced = _reduced_energies_for_record(record, state_indices, kt, lrc_coeff)
        indices, report = _decorrelate_independent_record(
            record, reduced, own, min_frames_for_subsampling=min_frames_for_subsampling
        )
        decorrelation_report[key] = report
        if indices.size == 0:
            continue
        blocks[own].append(reduced[indices, :].T)  # (K, n_decorrelated)
        used_records.append(record)

    # 🔑 [ENDPOINT_CAVITY_SAMPLING_GATE_PROTOCOL_VERSION=1] 慢坐标未被采到的记录，
    # 其能量序列样本数不能充当独立样本证据。见文件上方该常量的长注释。
    cavity_guard = {
        "protocol_version": int(ENDPOINT_CAVITY_SAMPLING_GATE_PROTOCOL_VERSION),
        "records_total": len(decorrelation_report),
        "records_with_active_guard": sum(
            1 for r in decorrelation_report.values()
            if r.get("cavity_guard_active") is True
        ),
        "degenerate_records": {
            key: {
                "status": r.get("cavity_series_status"),
                "reason": r.get("cavity_guard_inactive_reason"),
                "n_decorrelated_reported": (
                    r.get(r.get("selected_series") or "", {}) or {}
                ).get("n_decorrelated"),
            }
            for key, r in sorted(decorrelation_report.items())
            if r.get("cavity_guard_active") is not True
        },
    }
    # 判据必须**不依赖物理假设**。"std(cavity)==0 就拦"是错的：λ_vdw 接近 1 时配体
    # 完全占据空腔，干且零方差是**正确**的平衡行为；而在 λ_vdw→0 端同样的观测则可能是
    # walker 卡住。现有数据分不清这两者，所以不能按 std==0 一刀切。
    #
    # 可判的是**同一个态的兄弟 walker 互相矛盾**：若该态某条 walker 观察到了湿/干来回、
    # 另一条一步没动，则两条轨迹处在不同盆地，无论真实平衡态是什么样，这个态都没收敛。
    # 这条不需要知道"应该湿还是应该干"。
    # 4W53 实测正是此形：state_3 的 w0 转变 90 次(wet_fraction 0.312)、w1 零次(0.000)。
    #
    # 数据缺陷（长度不符/非有限）与此无关，一律 fail-closed。
    # 🔑 [2026-09-01 修] 分组必须是 (态, **模态**)，不能只按态。
    # 记录 key 是 `{global_k}|{mode}|{walker}`，旧写法 `key.split("|")[0]` 把模态丢了，
    # 于是同一个 λ 上的**干起点** walker 与**湿起点** walker 会被拿来互相比较。
    # 那是错的：干/湿双起点**按设计就从不同盆地出发**，它们之间的分歧是预期的初始
    # 条件，不是"没收敛"的证据 —— 判断两个模态是否收敛到一起是
    # `endpoint_wet_dry_hysteresis_gate` 的职责，不是这道门的。
    # 本门要抓的是「**同起点**的两条轨迹却待在不同盆地」，那才是无歧义的非平衡证据。
    # v3 那次是 dry-only 所以没咬到；一旦湿臂真的跑起来，旧写法必然误报，
    # 恰好卡死我们正想启用的东西。
    _by_state: Dict[Tuple[int, str], Dict[str, List[str]]] = {}
    for key, rep in decorrelation_report.items():
        _parts = key.split("|")
        gk = (int(_parts[0]), str(_parts[1]) if len(_parts) > 1 else "")
        slot = _by_state.setdefault(gk, {"active": [], "degenerate": []})
        if rep.get("cavity_guard_active") is True:
            slot["active"].append(key)
        elif rep.get("cavity_series_status") == "degenerate_zero_variance":
            slot["degenerate"].append(key)
    _malformed = sorted(
        key for key, rep in decorrelation_report.items()
        if rep.get("cavity_series_status") in ("length_mismatch", "nonfinite")
    )
    # 键写成 "state|mode"，让报告里一眼看出是哪个模态内部不自洽
    _inconsistent = {
        f"{gk[0]}|{gk[1]}": dict(slot)
        for gk, slot in sorted(_by_state.items())
        if slot["active"] and slot["degenerate"]
    }
    cavity_guard["malformed_records"] = _malformed
    cavity_guard["states_with_walker_disagreement"] = _inconsistent
    # 🔑 [2026-08-31] 跨 λ 诊断：**唯一与物理假设无关的判别方向**。
    # 态内任何判据（std==0 / 转变次数 / ACF g）都分不清"单相平衡"与"卡住"。
    # 但 wet_fraction 作为 λ_vdw 的函数应当光滑单调（λ_vdw→0 配体解耦、水逐步填入），
    # 相邻 λ 之间不可能来回跳。4W53 实测非单调：
    #     λ=0.229 wet=0.062 → λ=0.192 wet=0.321 → λ=0.126 wet=0.003 → λ=0.000 wet=0.156
    # 而 λ=0.126 那个态是**内部最一致**的（两条 walker 差 0.002）却最反常 ——
    # 注意所有 walker 都是 dry 起点，所以"walker 一致"本身不构成收敛证据。
    # ⚠️ **只报告不设门**：需要 ≥3 个 λ 点，且只能说"这批里有卡住的"、不能定位到哪个态；
    # 假阳性率还没在别的体系上验证过。升成硬门之前必须先看假阳性。
    _wet_by_state: Dict[int, List[float]] = {}
    for key, rep_ in decorrelation_report.items():
        wf = rep_.get("cavity_wet_fraction")
        if wf is not None:
            _wet_by_state.setdefault(int(key.split("|")[0]), []).append(float(wf))
    _wet_profile = [
        {"local_state": gk, "wet_fraction_mean": float(np.mean(v)),
         "wet_fraction_spread": float(np.max(v) - np.min(v)), "n_walkers": len(v)}
        for gk, v in sorted(_wet_by_state.items())
    ]
    _means = [p["wet_fraction_mean"] for p in _wet_profile]
    _monotonic = None
    if len(_means) >= 3:
        _diffs = np.diff(_means)
        _monotonic = bool(np.all(_diffs >= -1e-12) or np.all(_diffs <= 1e-12))
    cavity_guard["cross_lambda_wet_profile"] = _wet_profile
    cavity_guard["cross_lambda_wet_fraction_monotonic"] = _monotonic
    cavity_guard["cross_lambda_note"] = (
        "wet_fraction 沿 λ 应光滑单调；非单调是「有 walker 卡住」的、与物理假设无关的"
        "证据（但不能定位到具体态）。**当前仅诊断，不参与 converged 判定**——"
        "假阳性率未在其它体系上验证。"
    )
    cavity_guard["decision_rule"] = (
        "fail-closed 只在两种情形：(1) cavity 序列本身是坏数据（长度不符/非有限）；"
        "(2) 同一个态**且同一个起点模态**内，有 walker 观察到湿/干来回、另有 walker "
        "一步没动 —— 同起点却待在不同盆地。跨模态**不比较**：干/湿双起点按设计就从"
        "不同盆地出发，那由 endpoint_wet_dry_hysteresis_gate 判。"
        "**不**因为单纯 std==0 就拦：λ_vdw≈1 端干且零方差是正确的。"
    )
    # ⚠️ [2026-09-01] **本项全部降级为诊断，不参与 converged 判定。**
    #
    # 干/湿双起点的空腔占据是一个**诊断装置**，不是生产判据 —— 不该由它决定生产跑
    # 不跑得出数。它此前会 fail-closed，实测把一次本来完整的复合物腿运行整个挡下
    # （v3/seed20260908：6 窗口全采完、端点段 8 条记录全产出，摘掉本门后
    # n_k=[742,643,972,1000] 全部 ≥20，端点段 ΔG=+1.344±0.031 kJ/mol）。
    #
    # 而且这个观测量在复合物腿上**不是状态函数**：`count_cavity_waters` 的探针锚在
    # **配体**上，而空腔是**蛋白**的性质；λ_vdw→0 时配体与环境无 LJ、只被 Boresch
    # 松松拴着，会漂 —— 漂进侧链则恒为 0，漂到可及水袋则狂跳。实测佐证：
    #   · 同探针内蛋白重原子远多于水，最近蛋白 0.095–0.259 nm（幽灵嵌在侧链里）
    #   · 8 条 walker 里 7 条 cavity 取值恒 [0]，1 条跳 181 次
    #   · 换 repeat_seed 后整个 wet_fraction 分布变样（物理性质不会如此）
    # 用一个非状态函数的量去 gate 生产，拦住的是噪声。
    #
    # 诊断本身**全部保留并继续落盘**（同模态内不自洽、坏数据、跨 λ 单调性、
    # ACF 与转变数上界的背离），它们仍然是判断端点段遍历性的第一手证据；
    # 只是由人来读，不再自动否决。
    # 真要恢复成硬门，先把探针锚点从配体换成蛋白腔壁参考原子，让它重新成为
    # 蛋白构型的状态函数 —— 在那之前它没有 gate 的资格。
    cavity_guard["gates_converged"] = False
    cavity_guard["demoted_note"] = (
        "干/湿空腔占据是诊断装置，不是生产判据；且该观测量锚在会漂移的幽灵配体上，"
        "在复合物腿上不是状态函数。全部只报告，不参与 converged。"
    )

    n_k = np.asarray([sum(b.shape[1] for b in blk) for blk in blocks], dtype=int)
    if np.any(n_k <= 0):
        return {
            "error": "independent_endpoint_state_without_samples",
            "n_k": n_k.tolist(),
            "state_indices": state_indices,
            "global_state_indices": list(global_state_indices),
            # 🔑 [2026-09-01] 失败路径必须带上去相关报告。没有它，读者只看到
            # "n_k 不够"，却看不到**为什么**不够（是 g 大、还是帧本来就少、
            # 选中的是能量序列还是空腔序列）——而那正是决定下一步怎么办的信息。
            "decorrelation": decorrelation_report,
            "init_modes_used": sorted({str(r["init_mode"]) for r in used_records}),
            "converged": False,
        }
    if int(np.min(n_k)) < int(min_decorrelated_samples):
        return {
            "error": "independent_endpoint_insufficient_decorrelated_samples",
            "n_k": n_k.tolist(),
            "min_decorrelated_samples_threshold": int(min_decorrelated_samples),
            "state_indices": state_indices,
            "global_state_indices": list(global_state_indices),
            # 🔑 [2026-09-01] 失败路径必须带上去相关报告。没有它，读者只看到
            # "n_k 不够"，却看不到**为什么**不够（是 g 大、还是帧本来就少、
            # 选中的是能量序列还是空腔序列）——而那正是决定下一步怎么办的信息。
            "decorrelation": decorrelation_report,
            "init_modes_used": sorted({str(r["init_mode"]) for r in used_records}),
            "converged": False,
        }
    if not has_pymbar():
        return {
            "error": "pymbar_not_installed",
            "state_indices": state_indices,
            "global_state_indices": list(global_state_indices),
            "converged": False,
        }

    u_kn = np.concatenate([np.concatenate(blk, axis=1) for blk in blocks], axis=1)
    u_kn = u_kn - np.min(u_kn, axis=0, keepdims=True)
    mbar = _build_mbar_compatible(
        u_kn, n_k, relative_tolerance=1e-7, initialize="BAR", solver_protocol="default"
    )
    res = _compute_free_energy_result_compatible(mbar, compute_uncertainty=True)
    df_matrix, ddf_matrix = _extract_free_energy_arrays(res, require_uncertainty=True)

    f_kj = (df_matrix[0, :] * float(kt)).astype(float)
    df_kj = (ddf_matrix[0, :] * float(kt)).astype(float)
    delta_g = float(df_matrix[0, K - 1] * float(kt))
    delta_g_sigma = float(ddf_matrix[0, K - 1] * float(kt))

    neff = np.asarray(mbar.compute_effective_sample_number(), dtype=float)
    cavity_by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for record in used_records:
        cavity_by_mode.setdefault(str(record["init_mode"]), []).append(
            cavity_occupancy_diagnostics(record["cavity_waters"])
        )
    return {
        "protocol_version": int(INDEPENDENT_ENDPOINT_PROTOCOL_VERSION),
        "state_indices": state_indices,
        "global_state_indices": list(global_state_indices),
        # 🔑 [2026-08-31] 必须取自**实际贡献了样本的记录**，不能取自"请求"。
        # 这里原来是 `sorted(wanted_modes)` —— 在 `init_modes` 还是 manifest 身份键时
        # 无害（请求恒等于磁盘内容）；但 `init_modes` 已归入**预算**键（增长不清库），
        # 请求与磁盘不再等价。于是"请求了 ['dry','wet'] 但湿记录没落盘、或全被去相关
        # 滤空"的运行会谎报 `init_modes_used = ['dry','wet']`。
        # 而这正是读者用来判断"湿/干门到底有没有真的湿臂"的字段，配合
        # 「walker 一致不构成收敛证据」这条约束，说谎的后果比看起来严重。
        "init_modes_used": sorted(
            {str(r["init_mode"]) for r in used_records}
        ),
        "init_modes_requested": sorted(wanted_modes),
        "f_kJ_mol": f_kj.tolist(),
        "df_kJ_mol": df_kj.tolist(),
        "delta_G_kJ_mol": delta_g,
        "delta_G_sigma_kJ_mol": delta_g_sigma,
        "n_k_decorrelated": n_k.tolist(),
        "min_decorrelated_samples": int(np.min(n_k)),
        "min_decorrelated_samples_threshold": int(min_decorrelated_samples),
        "cavity_guard": cavity_guard,
        "effective_sample_number": neff.tolist(),
        "min_effective_sample_number": float(np.min(neff)),
        "decorrelation": decorrelation_report,
        "cavity_by_init_mode": cavity_by_mode,
        "estimator": "multi_state_MBAR_all_states_sampled",
        "sampling_note": (
            "每个 λ 态都有自己的独立固定-λ 轨迹（n_k>0），不存在单向重加权；"
            "动力学不含 Group-1 IBS 偏置与 Group-4 WCA 防护壳。"
        ),
        "converged": True,
    }


def endpoint_wet_dry_hysteresis_gate(
    result_wet: Dict[str, Any],
    result_dry: Dict[str, Any],
    per_walker_cavity: Dict[str, Any],
    max_sigma: float = ENDPOINT_WET_DRY_MAX_SIGMA,
    wet_arm_available: Optional[bool] = None,
    wet_seed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """湿/干双起点验收门。abfe_core.STAGE2_HYSTERESIS_MAX_SIGMA 的第一个真实执行点。

    两道**都必须过**的判据：

      (1) 数值一致：|ΔF_wet - ΔF_dry| <= max_sigma * sqrt(σ_wet² + σ_dry²)。
          两组来自各自独立的种子与起始构型，样本互不重叠，所以 σ 按独立量
          平方相加；若将来改成共享 walker，必须把协方差补进来。
      (2) 模态确实连通：至少要有一个 walker 真的在湿态和干态之间转换过
          （n_wet_dry_transitions > 0），**或者**湿组与干组的 wet_fraction
          区间有重叠。若湿组全程 wet_fraction=1、干组全程 =0、两边转换次数
          都是 0，那就是两个互不连通的模态各自被采了一遍——此时 (1) 即使
          通过也毫无意义（两个都错、且错得一样，差值照样是 0）。这一条是
          docs/archive/STAGE2_ROOT_CAUSE_2026-08-28.md §4 "所有门测的都是已采到的构型之间
          的散布" 那个教训的直接产物：必须有一项直接证明"该采的构型采到了"。
    """
    # ------------------------------------------------------------------
    # 🔑 [2026-08-31] 湿/干双起点是**条件性诊断**，不是普遍适用的硬门。
    # 只有真的构造出两类起点时才评估；构造不到（例如 T4 L99A 这类本来就干的
    # 埋藏疏水腔，预算内根本观察不到湿盆）就返回 evaluated=False / passed=None。
    #   passed=None 既不算失败、也不算通过，**不参与 converged 合取**，
    #   但必须把「该诊断未评估」如实带进结果与 provenance。
    # 真正通用的硬门与具体物理盆无关：raw ESS、top-1% 权重、独立 walker/种子
    # 一致性、BAR–MBAR 一致性、样本数与去相关要求。
    # ------------------------------------------------------------------
    if wet_arm_available is False or not result_wet:
        # 🔑 [2026-09-01] **"没跑" 与 "跑了没找到" 必须分开记。**
        # 后者是一个关于体系的**物理断言**（该 λ 下不存在可及的湿盆）；前者只是
        # 这次没执行搜索。把两者压成同一条 reason，会让"我们从没找过"读起来像
        # "我们找过、确认没有" —— 这正是本项目反复出现的"字段说谎"。
        # 湿盆搜索自 2026-09-01 起默认不执行（诊断装置，非生产判据），所以
        # 绝大多数运行走的是 not_attempted 这一支。
        _skipped = bool(isinstance(wet_seed, dict) and wet_seed.get("search_skipped"))
        if _skipped:
            return {
                "evaluated": False,
                "passed": None,
                "reason": "wet_search_not_attempted",
                "note": ("本次**未执行**湿盆搜索（干/湿双起点是诊断装置、不是生产判据，"
                         "默认关闭；开启用 independent_endpoint_wet_search=True）。"
                         "因此只跑了干起点。[WARN] 这**不是**「该体系没有湿盆」的证据——"
                         "我们这次根本没找。湿/干双起点门未评估，不参与 converged 判定；"
                         "该端点段的遍历性未经双起点检验。"),
                "skip_reason": wet_seed.get("skip_reason"),
                "max_sigma": float(max_sigma),
            }
        return {
            "evaluated": False,
            "passed": None,
            "reason": "wet_start_not_observed_within_budget",
            "note": ("预算内未观察到湿盆，已只跑干起点。湿/干双起点门未评估——"
                     "既不算通过也不算失败，不参与 converged 判定。"
                     "该端点段的遍历性未经双起点检验。"),
            "max_sigma": float(max_sigma),
        }
    failures: List[str] = []
    for label, result in (("wet", result_wet), ("dry", result_dry)):
        if not result or result.get("converged") is not True:
            failures.append(f"{label}_solve_failed")
    if failures:
        return {
            "evaluated": True,
            "passed": False, "failed_checks": failures,
            "failure_reason": "wet_dry_hysteresis_unavailable",
            "max_sigma": float(max_sigma),
        }

    dg_wet = float(result_wet["delta_G_kJ_mol"])
    dg_dry = float(result_dry["delta_G_kJ_mol"])
    s_wet = float(result_wet["delta_G_sigma_kJ_mol"])
    s_dry = float(result_dry["delta_G_sigma_kJ_mol"])
    delta = abs(dg_wet - dg_dry)
    sigma_combined = float(np.hypot(s_wet, s_dry))
    numeric_ok = bool(
        np.isfinite(delta) and np.isfinite(sigma_combined) and sigma_combined > 0.0
        and delta <= float(max_sigma) * sigma_combined
    )
    if not numeric_ok:
        failures.append("wet_dry_delta_exceeds_sigma_gate")

    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for entry in per_walker_cavity.values():
        by_mode.setdefault(str(entry.get("init_mode")), []).append(entry)
    transitions = [
        int(e.get("n_wet_dry_transitions") or 0)
        for entries in by_mode.values() for e in entries
    ]
    wet_fracs = {
        mode: [float(e["wet_fraction"]) for e in entries
               if e.get("wet_fraction") is not None]
        for mode, entries in by_mode.items()
    }
    any_transition = bool(transitions) and max(transitions) > 0
    overlap = False
    if wet_fracs.get("wet") and wet_fracs.get("dry"):
        overlap = bool(
            min(wet_fracs["wet"]) <= max(wet_fracs["dry"])
            and min(wet_fracs["dry"]) <= max(wet_fracs["wet"])
        )
    ergodicity_ok = bool(any_transition or overlap)
    if not ergodicity_ok:
        failures.append("wet_and_dry_walkers_never_exchanged_modes")

    return {
        "evaluated": True,
        "passed": not failures,
        "failed_checks": failures,
        "failure_reason": (
            None if not failures else "stage2_wet_dry_hysteresis_failed"
        ),
        "delta_G_wet_kJ_mol": dg_wet,
        "delta_G_dry_kJ_mol": dg_dry,
        "delta_kJ_mol": delta,
        "sigma_wet_kJ_mol": s_wet,
        "sigma_dry_kJ_mol": s_dry,
        "sigma_combined_kJ_mol": sigma_combined,
        "max_sigma": float(max_sigma),
        "allowed_delta_kJ_mol": float(max_sigma) * sigma_combined,
        "max_wet_dry_transitions": max(transitions) if transitions else 0,
        "wet_fraction_by_init_mode": wet_fracs,
        "modes_are_connected": ergodicity_ok,
        "sigma_combination": "independent_seeds_quadrature",
    }


def combine_ibs_and_independent_endpoint(
    ibs_result: Dict[str, Any],
    endpoint_result: Dict[str, Any],
    wet_dry_gate: Dict[str, Any],
    n_states: int,
    # 默认值在**调用时**解析：TARGET_SUPPORT_MIN_ABSOLUTE_ESS 定义在本文件更靠后
    # 的位置（紧邻它所服务的 ESS 门），def 时求值会 NameError。
    target_support_min_absolute_ess: Optional[float] = None,
) -> Dict[str, Any]:
    """把 IBS 段与独立端点段拼成一个 stage2 结果。

    λ 路径被切成两段：低 λ 索引那段仍由 IBS 窗口重加权求解（那里配体还实实在在
    占着体积，不存在"水塌进空腔"这个未被采到的构型），高 λ 索引那段（λ_vdw→0）
    改由每个态自己的独立固定-λ 轨迹求解。两段在**同一个** λ 索引上相接。

    方向约定必须一致才能相加：`solve_stage_integrated` 的 `total_delta_G` 是沿
    λ 索引**升序**逐段累加的（见协方差链那段），而 `solve_independent_endpoint_
    states` 的 `delta_G_kJ_mol` 是 `f[K-1] - f[0]`、`state_indices` 同样升序——
    两者同向，接点处的 f 抵消，直接相加即为全程。

    σ 按独立量平方相加：两段来自**完全不相干**的轨迹（不同 System 变体、不同
    Context、不同种子），样本没有任何重叠，所以没有需要补的协方差项。这与同一
    窗口内两个端点之间必须直接读 dDelta_f（有协方差）是两回事，不要混淆。

    `converged` 是三者的合取：IBS 段自己的全部硬门、独立端点段解出且样本足够、
    以及湿/干双起点门。任何一项没有明确通过都判失败——尤其 `wet_dry_gate` 缺失
    时绝不当作"没测到就放行"。
    """
    if target_support_min_absolute_ess is None:
        target_support_min_absolute_ess = TARGET_SUPPORT_MIN_ABSOLUTE_ESS
    # 🔑 [2026-08-31] 端点段自己报的失败原因必须**先**于任何拼接几何检查被读出来。
    # 旧顺序是先查接缝、后查 `endpoint_result["error"]`：端点求解失败时返回的 dict
    # 没有可用的 λ 索引，拼接检查于是拿局部索引算出 join=0，把每一种端点失败都
    # 误报成 `combine_join_mismatch: IBS 段止于 19，独立端点段起于 0`。4W53
    # output_v2 就是这样把 `independent_endpoint_insufficient_decorrelated_samples`
    # （n_k=[342, 18, 962, 508]，阈值 20，只差 2 个样本）伪装成索引错误的。
    # 仍然 fail-closed：这里只改"报什么原因"，不改"是否放行"。
    endpoint_error = endpoint_result.get("error")
    if endpoint_error:
        return {
            "error": f"independent_endpoint_segment_failed: {endpoint_error}",
            "endpoint_error": str(endpoint_error),
            "endpoint_n_k": endpoint_result.get("n_k"),
            "endpoint_min_decorrelated_samples_threshold": endpoint_result.get(
                "min_decorrelated_samples_threshold"
            ),
            "converged": False, "total_delta_G": 0.0, "total_error": 999.9,
        }
    ibs_lams = [int(x) for x in (ibs_result.get("lambdas") or [])]
    # 拼接必须用**全局** λ 索引；`state_indices` 在只为端点块建窗口时是局部的。
    endpoint_states = [
        int(x) for x in (
            endpoint_result.get("global_state_indices")
            or endpoint_result.get("state_indices")
            or []
        )
    ]
    if not ibs_lams or not endpoint_states:
        return {
            "error": "combine_missing_segment_lambdas",
            "converged": False, "total_delta_G": 0.0, "total_error": 999.9,
        }
    if ibs_lams != sorted(ibs_lams) or endpoint_states != sorted(endpoint_states):
        return {
            "error": "combine_segments_not_ascending",
            "converged": False, "total_delta_G": 0.0, "total_error": 999.9,
        }
    join = endpoint_states[0]
    if ibs_lams[-1] != join:
        return {
            "error": (
                f"combine_join_mismatch: IBS 段止于 λ 索引 {ibs_lams[-1]}，"
                f"独立端点段起于 {join}，两段必须在同一个 λ 索引相接"
            ),
            "converged": False, "total_delta_G": 0.0, "total_error": 999.9,
        }
    covered = sorted(set(ibs_lams) | set(endpoint_states))
    if covered != list(range(int(n_states))):
        return {
            "error": (
                f"combine_incomplete_lambda_coverage: 两段合起来覆盖 {covered}，"
                f"期望 [0,{int(n_states)})"
            ),
            "converged": False, "total_delta_G": 0.0, "total_error": 999.9,
        }

    ibs_dg = float(ibs_result.get("total_delta_G", 0.0))
    ibs_sigma = float(ibs_result.get("total_error", 0.0))
    end_dg = float(endpoint_result.get("delta_G_kJ_mol", 0.0))
    end_sigma = float(endpoint_result.get("delta_G_sigma_kJ_mol", 0.0))
    if not all(np.isfinite(x) for x in (ibs_dg, ibs_sigma, end_dg, end_sigma)):
        return {
            "error": "combine_nonfinite_segment",
            "converged": False, "total_delta_G": 0.0, "total_error": 999.9,
        }

    endpoint_min_ess = endpoint_result.get("min_effective_sample_number")
    endpoint_support_ok = bool(
        endpoint_min_ess is not None
        and np.isfinite(endpoint_min_ess)
        and _meets_minimum_with_roundoff(
            float(endpoint_min_ess), float(target_support_min_absolute_ess)
        )
    )
    ibs_gate = ibs_result.get("target_support_gate")
    ibs_support_ok = bool(isinstance(ibs_gate, dict) and ibs_gate.get("passed") is True)
    failed_checks: List[str] = []
    if not ibs_support_ok:
        failed_checks.append("ibs_segment_target_support")
    if not endpoint_support_ok:
        failed_checks.append("independent_endpoint_effective_sample_number")
    combined_target_support = {
        "protocol_version": int(TARGET_SUPPORT_GATE_PROTOCOL_VERSION),
        "passed": not failed_checks,
        "failure_reason": (
            "insufficient_target_support" if failed_checks else None
        ),
        "failed_checks": failed_checks,
        "ibs_segment_gate": ibs_gate,
        "independent_endpoint_min_effective_sample_number": (
            float(endpoint_min_ess) if endpoint_min_ess is not None else None
        ),
        "raw_min_absolute_ess_threshold": float(target_support_min_absolute_ess),
        "segment_note": (
            "独立端点段每个物理态都有 n_k>0 的真实样本，不存在单向重加权，"
            "所以它的支撑度用 MBAR 自己的有效样本数直接判定；IBS 段仍按"
            "raw 重加权支撑度判定。"
        ),
    }

    # 🔑 [2026-08-31] 湿/干门是条件性诊断：
    #   passed is True  -> 通过
    #   passed is False -> 失败，参与合取（真的检出了迟滞）
    #   passed is None  -> **未评估**（预算内没观察到湿盆），不参与合取，
    #                      但必须在结果里留下明确警告。
    _wd = wet_dry_gate if isinstance(wet_dry_gate, dict) else {}
    _wd_passed = _wd.get("passed", False) if _wd else False
    wet_dry_blocks = (_wd_passed is False)      # None 不阻塞
    wet_dry_unevaluated = (_wd_passed is None)
    # 🔑 [2026-09-15] `ibs_result` 来自 `solve_stage_integrated`，它的 `converged`
    # 键已被删除（**不是改义**），改读 `analysis_status`。这里如果还写
    # `ibs_result.get("converged") is True`，拿到的永远是 None ⇒ 恒 False，那才是
    # 真正的静默失效。本函数自己的 `converged` 输出键属于 abfe_pipeline 的契约，
    # 不在本次改动范围内，原样保留。
    ibs_analysis_complete = ibs_result.get("analysis_status") == "ANALYSIS_COMPLETE"
    converged = bool(
        ibs_analysis_complete
        and endpoint_result.get("converged") is True
        and combined_target_support["passed"]
        and not wet_dry_blocks
    )
    return {
        "stage": ibs_result.get("stage", "vanishing"),
        "total_delta_G": ibs_dg + end_dg,
        "total_error": float(np.hypot(ibs_sigma, end_sigma)),
        "total_error_method": "independent_segments_quadrature_ibs_plus_endpoint",
        "converged": converged,
        "join_lambda_index": int(join),
        "n_states": int(n_states),
        "ibs_segment": {
            "lambda_indices": ibs_lams,
            "delta_G_kJ_mol": ibs_dg,
            "uncertainty_kJ_mol": ibs_sigma,
            "analysis_status": ibs_result.get("analysis_status"),
            "analysis_incomplete_reasons": ibs_result.get("analysis_incomplete_reasons"),
            "precision_status": ibs_result.get("precision_status"),
        },
        "independent_endpoint_segment": {
            "lambda_indices": endpoint_states,
            "delta_G_kJ_mol": end_dg,
            "uncertainty_kJ_mol": end_sigma,
            "converged": endpoint_result.get("converged"),
            "n_k_decorrelated": endpoint_result.get("n_k_decorrelated"),
            "min_effective_sample_number": endpoint_min_ess,
        },
        "target_support_gate": combined_target_support,
        "target_support_gate_protocol_version": int(
            TARGET_SUPPORT_GATE_PROTOCOL_VERSION
        ),
        "wet_dry_hysteresis_gate": wet_dry_gate,
        "wet_dry_hysteresis_evaluated": (not wet_dry_unevaluated),
        # 未评估时必须有一条显式警告随结果走，不能只是 passed=None 一个静默的值。
        # 🔑 [2026-09-01] 原因**从门的 reason 派生**，不能写死「预算内未观察到湿盆」——
        # 那是一个物理断言，而搜索默认已不执行（诊断装置、非生产判据），绝大多数运行
        # 根本没找过。这条 warning 会随结果落盘，写错比日志写错更持久。
        # 🔑 [2026-09-01 退役] 退役必须有**自己的**措辞。沿用"未评估/未经检验"那套
        # 会把已经作废的诊断说成一个仍然欠着的检查项，等于让每份结果都背一条
        # 假欠账；而沿用"本次未执行湿盆搜索"更糟——那读起来像"下次跑就有了"。
        "warnings": ([
            (
                "湿/干双起点诊断已退役，未评估：空腔水数探针锚在会漂移的配体上，"
                "在复合物腿上不是状态函数 ⇒ 该判据无判别力，已从生产路径移除"
                "（不参与 converged）。端点段的遍历性由 raw ESS、top-1% 权重、"
                "独立 walker/种子一致性与去相关样本数判定。"
                if (wet_dry_gate or {}).get("reason") == WET_DRY_RETIRED_REASON
                else "湿/干双起点门未评估（"
                + (
                    "本次未执行湿盆搜索——这不是「该体系没有湿盆」的证据"
                    if (wet_dry_gate or {}).get("reason") == "wet_search_not_attempted"
                    else "预算内未观察到湿盆，已只跑干起点"
                )
                + "）——该端点段的遍历性未经双起点检验。"
            )
        ] if wet_dry_unevaluated else []),
        "independent_endpoint_protocol_version": int(
            INDEPENDENT_ENDPOINT_PROTOCOL_VERSION
        ),
        "min_overlap": ibs_result.get("min_overlap"),
        "min_overlap_threshold": ibs_result.get("min_overlap_threshold"),
        "raw_min_absolute_ess": ibs_result.get("raw_min_absolute_ess"),
        "raw_min_absolute_ess_threshold": ibs_result.get(
            "raw_min_absolute_ess_threshold"
        ),
        "max_top1pct_raw_weight": ibs_result.get("max_top1pct_raw_weight"),
        "max_top1pct_raw_weight_threshold": ibs_result.get(
            "max_top1pct_raw_weight_threshold"
        ),
        "window_overlap_diagnostics": ibs_result.get("window_overlap_diagnostics"),
        "min_decorrelated_samples": ibs_result.get("min_decorrelated_samples"),
        "min_decorrelated_samples_threshold": ibs_result.get(
            "min_decorrelated_samples_threshold"
        ),
        "max_endpoint_uncertainty_kJ_mol": ibs_result.get(
            "max_endpoint_uncertainty_kJ_mol"
        ),
        "max_endpoint_uncertainty_kJ_mol_threshold": ibs_result.get(
            "max_endpoint_uncertainty_kJ_mol_threshold"
        ),
        "coverage_diagnostics": {
            "covered_lambda_indices": covered,
            "n_covered_lambda_indices": len(covered),
            "ibs_lambda_indices": ibs_lams,
            "independent_endpoint_lambda_indices": endpoint_states,
            "note": (
                "stage2 由两段组成：IBS 重加权段 + 独立固定-λ 端点段，"
                f"在 λ 索引 {int(join)} 相接。"
            ),
        },
        "method": "IBS-TMBAR + independent fixed-lambda endpoint states",
    }



