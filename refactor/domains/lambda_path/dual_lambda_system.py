# ============================================================================
# 1. 双λ微扰系统构建 (核心修复版)
# ============================================================================
# create_ligand_internal_force 已从 abfe_core 导入
def _lj_tail_correction_sigma_resolved_moments(
    all_params,
    ligand_indices: List[int],
    environment_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """配体<->环境 LJ 长程尾项的【按 sigma_ij 分组】几何求和矩。

    返回 (sigma_nm, s6_per_sigma, s12_per_sigma)，其中对第 b 组

        s6_per_sigma[b]  = sum_{pairs with sigma_ij == sigma_nm[b]} epsilon_ij * sigma_ij^6
        s12_per_sigma[b] = sum_{pairs with sigma_ij == sigma_nm[b]} epsilon_ij * sigma_ij^12

    混合规则与软核表达式一致 (sigma_ij=0.5*(sigma_i+sigma_j),
    epsilon_ij=sqrt(epsilon_i*epsilon_j))。全部是固定几何量，跟 lambda、switching、
    盒子体积无关（那些在径向积分/每帧体积换算里另外处理）。

    🔑 [TRADITIONAL_LJ_LRC_PROTOCOL_VERSION=3] v2 只返回两个【全局标量】
    S6=sum(eps*sigma^6)、S12=sum(eps*sigma^12)，因为当时软核分母
    D(r)=alpha_lj*(1-lambda)^m + r^6 与 pair 无关，I6/I12 可以提到求和号外面、每个
    lambda 只积一次。软核 alpha 改为无量纲、乘 sigma_ij^6 之后
    （SOFTCORE_ALPHA_CONVENTION=dimensionless_sigma_scaled_v2），
    D_ij(r)=alpha_lj*sigma_ij^6*(1-lambda)^m + r^6 变成 pair-specific，那个分解不再
    成立，尾修正必须按 sigma 分组后逐组积分再求和，否则尾修正对应的哈密顿量与实际
    采样的哈密顿量不一致。力场里不同的 sigma 取值很少（远少于 pair 数），分组后每个
    lambda 的积分次数是 O(distinct sigma) 而非 O(pairs)，且只在建系时算一次。
    """
    sigma_lig = np.array(
        [all_params[i][1].value_in_unit(unit.nanometer) for i in ligand_indices], dtype=np.float64
    )
    eps_lig = np.array(
        [all_params[i][2].value_in_unit(unit.kilojoule_per_mole) for i in ligand_indices], dtype=np.float64
    )
    sigma_env = np.array(
        [all_params[j][1].value_in_unit(unit.nanometer) for j in environment_indices], dtype=np.float64
    )
    eps_env = np.array(
        [all_params[j][2].value_in_unit(unit.kilojoule_per_mole) for j in environment_indices], dtype=np.float64
    )
    # 🔑 [2026-09-09] 逐 ligand 行累加，不再物化 n_ligand × n_environment 的稠密外积。
    #
    # 原来一次开出六个全尺寸数组：`sigma_ij`、`eps_ij`、`sigma_key`、`inverse`
    # 以及两个权重数组。41 原子配体 × 73k 环境实测进程峰值约 219 MiB
    # （基线 34 MiB）；100 原子 × 100k 膜盒按同比例约 700 MiB。而真正需要的只是
    # 每个 distinct sigma 的两个矩。逐行版实测峰值约 37 MiB、耗时相同。
    #
    # **这段改写与旧实现逐位等价**，`tests/test_lj_tail_sigma_moments_bit_identical.py`
    # 冻结了旧 dense 实现当 oracle、用 `.tobytes()` 比对。等价性依赖下面这些顺序，
    # 动任何一条都会破：
    #   * ligand 外层、environment 内层 —— 决定累加次序；
    #   * 分组键仍是 `round(..., 9)`；
    #   * 权重用**未 round** 的 sigma；
    #   * bin 仍是 `np.unique` 的升序输出；
    #   * 按原始 pair 次序累加（`np.bincount` 就是按输入下标顺序累加的，
    #     逐行 `np.add.at` 只有在行序与行内序都保持时才等价）。
    #
    # bin 集合不需要遍历所有 pair：sigma_ij 只取决于 (distinct sigma_lig,
    # distinct sigma_env) 的外和，力场里这两个集合都很小。每个候选都被某个真实
    # pair 取到、每个真实键也都是候选，所以与"对全量 key 取 unique"同集合同序。
    unique_sigma_lig = np.unique(sigma_lig)
    unique_sigma_env = np.unique(sigma_env)
    sigma_nm = np.unique(
        np.round(
            0.5 * (unique_sigma_lig[:, None] + unique_sigma_env[None, :]), 9
        ).ravel()
    )
    s6_per_sigma = np.zeros(sigma_nm.shape[0], dtype=np.float64)
    s12_per_sigma = np.zeros(sigma_nm.shape[0], dtype=np.float64)
    for row in range(sigma_lig.shape[0]):
        sigma_row = 0.5 * (sigma_lig[row] + sigma_env)
        eps_row = np.sqrt(eps_lig[row] * eps_env)
        # 1e-9 nm 的分组容差远小于任何物理 sigma 差异，只用来把浮点上完全等价的取值合并。
        bin_index = np.searchsorted(sigma_nm, np.round(sigma_row, 9))
        np.add.at(s6_per_sigma, bin_index, eps_row * sigma_row ** 6)
        np.add.at(s12_per_sigma, bin_index, eps_row * sigma_row ** 12)
    return (
        sigma_nm.astype(np.float64),
        s6_per_sigma.astype(np.float64),
        s12_per_sigma.astype(np.float64),
    )


# 🔑 LRC 协议版本 1→2：v1 只补 r_cutoff（1.2 nm）之外的标准 r^-6 尾项，完全忽略了
# 软核力实际启用的 1.0-1.2 nm switching 区间——OpenMM 的 switching function 会在这段
# 区间把能量从满强度削到 0，真正缺失的修正是 [switching 区间被削弱的部分] +
# [cutoff 之外的标准尾项]（见 CustomNonbondedForce LRC 文档），v1 只补了后者。v1 还
# 把 λ<1 时的 softcore 分母 D(r)=alpha_lj*(1-λ)^m + r^6 当成纯 r^6 处理，等于假装
# softcore 在尾区间不存在。v2 改为对每个 λ 数值积分真实的
# switching-aware + softcore-aware 径向积分（见 _lj_softcore_tail_radial_integrals /
# _lj_tail_lrc_coefficients_kj_mol），同时补上排斥项 r^-12 的尾贡献（之前只有吸引项
# r^-6）。ACE/dual_lambda 路径（IBSSampler._lj_tail_correction_kj_mol /
# probe_bidirectional_overlap 的 fixed-H 探针）和传统 Beutler REMD 路径
# （TraditionalMBARAnalyzer.compute_u_kn 的离线 u_kn 重算）现在共用同一套
# lrc_coeff 计算逻辑，不再各自维护一份。旧版本号下产出的能量文件/u_kn 缓存一律
# fail closed，不能和 v2 的结果混用。
# 🔑 LRC 协议版本 2→3：配合 SOFTCORE_ALPHA_CONVENTION=dimensionless_sigma_scaled_v2，
# 软核分母从 pair 无关的 D(r)=alpha_lj*(1-λ)^m + r^6 变成 pair-specific 的
# D_ij(r)=alpha_lj*sigma_ij^6*(1-λ)^m + r^6。v2 把 S6/S12 先汇总成两个标量、每个 λ
# 只积一次的分解因此失效，v3 改为按 sigma_ij 分组逐组积分再求和（见
# _lj_tail_correction_sigma_resolved_moments / _lj_tail_lrc_coefficients_kj_mol）。
# 两处变化的量级不同，别混为一谈：alpha 本身缩小 ~685 倍会让低 λ 的尾修正系数发生
# 可观改变（旧 α 在 λ→0 时偏移 0.5 nm^6，与 r_switch=1.0 nm 处的 r^6=1.0 同量级，
# 真的会压低积分）；而在此基础上再做 sigma 分辨，因为 α·sigma^6 ≤ 0.5·(0.4)^6≈2e-3
# 相对 r^6≥1 只有 ≲0.2%，属于小修正——但它是让尾修正与采样哈密顿量严格一致所必需的。
# v2 及更早版本号下产出的能量文件/u_kn 缓存对 v3 一律 fail closed。
TRADITIONAL_LJ_LRC_PROTOCOL_VERSION = 3

# 🔑 [MEM-00h，2026-08-06] vdW/softcore 非键协议版本——单独一个版本号，只覆盖
# softcore cutoff/switching（1.2nm+switch → 1.0nm 无switch）这一次改动本身，
# 不复用 TRADITIONAL_LJ_LRC_PROTOCOL_VERSION。
#
# 为什么不直接 bump TRADITIONAL_LJ_LRC_PROTOCOL_VERSION：那个字段目前是
# `stage_type in {"coul","vdw"}` 都写、都校验（见 convergence 字典构造与
# _resume_cached_window_gate_status）——即便 LRC 概念上只跟 vdW 有关。沿用它会
# 把 Stage 1 charging 的缓存也一起判废，而 Stage 1 charging 完全不受这次改动
# 影响（它不构造软核 CV，SOFTCORE_CUTOFF_NM/switching 跟它无关）。这是本仓库
# 明确要求的边界：只作废 Stage 2 vdW window 与对应 u_kn，Stage 1 charging、
# Boresch attachment、预平衡、C1 的 charging 轨迹一律不许重跑。
#
# 因此这个新版本号只在 stage_type=="vdw" 时写入 convergence.json、只在
# stage_type=="vdw" 时参与 resume 门判定（见 _resume_cached_window_gate_status
# 的 stage_type 形参与 vdw_nb_version_match）；stage_type=="coul" 的窗口完全
# 不受影响，缺这个字段也不会被判无效。
VDW_NONBONDED_PROTOCOL_VERSION = 1

# ============================================================================
# [LIGAND_COM_RESTRAINT_PROTOCOL_VERSION]
#   version 1: Group 5 = CustomCentroidBondForce，非周期绝对锚点
#              r_com=sqrt((x1-x0)^2+(y1-y0)^2+(z1-z0)^2)，usesPBC=False，
#              锚点取自输入构型的配体 COM。**已确认为 P0 缺陷，见
#              4W53/STAGE2_GROUP5_CUDA_PBC_ROOT_CAUSE_2026-08-29.md。**
#   version 2: 移除该力（当前）。
#
# 缺陷的准确定义：固定锚点使用未折叠周期像，而 CUDA 动力学中的 centroid 被折叠到
# 主盒；非周期绝对距离把两个不同周期像当成真实距离，形成永久激活、跨边界不连续的
# 错误外力。注意"锚点在主盒外"本身不是错误——CPU 上锚点与 centroid 成像一致，
# E5 恒为 0、F5 恒为 0，行为完全正常。错的是**两侧成像规则不一致**。
#
# 实测（同一生产 System，λ_vdw=0，初始化后不再 setPositions，enforcePeriodicBox=False）：
#     CUDA  err_fold 恒为 0.000，|F5| 12–256 kJ/mol/nm，COM 未折叠位移 110 ps 内
#           3.9 → 30.9 nm 单调增长（自由扩散 RMS 仅约 3.0 nm），跨盒 10 次且力方向跳变
#     CPU   err_raw  恒为 0.000，|F5| 恒为 0.00，COM 位移与自由扩散一致
# 后果：CUDA 轨迹是带定向粒子流的非平衡过程，不满足 MBAR 的"样本来自已知平衡分布"
# 前提。静态 setPositions() 测试与任何 CPU 测试都**检不出**它。
#
# 为什么是移除而不是改成周期最小像：该力只在**没有 Boresch 锚定**时才添加，即只存在
# 于溶剂腿；而均匀溶剂中它不提供任何必要约束。B/C 受控对照（4 个态 × 湿干 × 3 种子）
# 显示：删除组与最小像修复组的湿/干结构统计与局部 BAR 自由能在 ±0.25 kJ/mol 预设
# 等价界内一致（6→8 局部总自由能只差 0.001 kJ/mol），而最小像组在 6→7 边的正向 ESS
# 显著更低（24.8 vs 177.1）。即：保留它没有热力学收益，还可能降低有限时间混合效率。
# 复合物腿本来就跳过它（依赖 Boresch），因此本次改动不影响复合物腿的 Hamiltonian。
#
# ⚠️ 若将来确需 COM 限制，**不要**再用绝对锚点。本文件 `build_co_alchemical_ion_restraint`
# 的注释里已记录可用写法：`periodicdistance` 只存在于 CustomExternalForce；
# `CustomCompoundBondForce` 打开 PBC 后其中的 `pointdistance` 就是 minimum-image 距离。
# 复合物腿还必须先判断它是否与 Boresch 限制重复计入。
# ============================================================================
LIGAND_COM_RESTRAINT_PROTOCOL_VERSION = 2

# ============================================================================
# λ-WCA 防护壳（Group 4）：**已退役（2026-09-02）**，生产采样里是死代码。
# ----------------------------------------------------------------------------
# 退役依据是实测的因果归因，不是判断：4W53 甲苯溶剂腿 stage2（λ_vdw 解耦段）
# 与独立参考真值差 +45.30 kJ/mol，2×2 实验（GPU、逐态独立采样、生产真实 23 态
# λ 表、softcore m=n=2）把它拆开：
#
#                     | 逐态独立采样                | 单混合分布
#     ----------------+-----------------------------+-----------------
#       无壳          | -6.581±0.256 / -7.324±0.258 |   —
#       有壳          | +36.889±0.068               | +38.724（生产）
#
#     壳         +44.21   ← 干净：两条臂都不含 LRC，相减无口径问题
#     单混合     + 1.83   ← ⚠️ 2026-09-09 更正：**错的**，正确是 -0.99
#     统计残差   - 0.74
#
# ⚠️ [2026-09-09 更正] 上面 `单混合 +1.83 (4%)` 有 **LRC 口径错误**：它是
# `38.724（生产，含 LJ 尾项）- 36.889（臂，不含）`。对齐后（尾项减在生产侧）
# 是 **-0.99**；`total_error` 45.30 同理应为 42.48。壳那 +44.21 不受影响
# （两条臂都不含尾项）⟹ **本常量的退役依据完全成立，不要因为这条去翻它**。
#
# 两次独立实现的"无壳+独立采样"互差 0.744 = 2.05σ ⟹ 归因可信，且**无壳能正常采样**。
# 逐窗口壳贡献 1.20/2.17/5.40/7.90/16.53/11.02 随 `bias_to_signal_ratio`
# (1.8/2.0/3.3/5.1/11.6/46.3) 单调上升 ⟹ 那条 Spearman +0.886 是因果。
#
# 机制：壳幅度 4λ_s(1−λ_s) 在 λ→0 只**线性**衰减，而物理配体↔环境 LJ 是
# λ^n_lj（生产 n_lj=2，**二次**衰减），比值 4(1−λ)/λ 在 λ→0 发散。
# rc=0.244 nm 比 O–O 第一接触(0.28 nm)还短，是原子级贴近护栏；λ_vdw→0 时配体自己
# 的 LJ 只剩 λ²≈2%，水贴到 0.244 nm 内，(rc/r)¹² 猛发火（bias 出现 +17~+20 尖峰）。
# 目标态**不含**壳 ⟹ 生产在"带壳系综"上重加权到"无壳目标"，而那些水贴近的构型
# 系综里根本没有。生产那份形式正确的壳修正在数值上几乎完全失效
# （win5 要把权重放大 e^(17/2.48)≈1000 倍去捞没采到的构型）。
#
# 这与 `ibs_engine.py` 约 10978-11020 的要求 #1 完全一致 ——
# 「**正式采样不得保留 Group-4 WCA 防护壳** …… 它把水挡在空腔外，正是本模块要采的
# 那个构型」。区别只是：那里的正解是 `stage2_independent_endpoint`（逐态独立轨迹，
# ~90 分钟/腿，2026-09-01 因成本关掉），而本实验当时的结论是**独立采样只值 4%**
# ⟹ 保留 IBS 单混合分布、只移除壳。
#
# ⚠️ [2026-09-09 更正] 「独立采样只值 4%」**不成立**，两处：(1) 那个 +1.83 带
# LRC 口径错误，对齐后是 -0.99；(2) 它是**带壳**测的。壳退役后同一个量是
# **-3.57 kJ/mol**（放大 3.6 倍），而且它就是壳退役后剩下的**全部**残差
# （生产 -10.898 vs 逐态独立 -7.324 / 真值 -6.581，即 -3.57 ~ -4.32，5.5σ）。
# ⟹ 「只移除壳」这一步是对的且已完成，但**不能**据此认为独立采样没有价值。
# 逐项消元证据见 docs/archive/STAGE2_SOLVENT_LEG_ERROR_BUDGET.md。
#
# 退役方式是**死代码而非删除**：`build_ibs_dual_system` 里构造壳的整段留在一个
# 永久为假的分支里；`_estimate_wca_shield_parameters` 等函数体一字未改、仍可导入。
# 把本常量翻 `False` 即整段复活。
#
# ⚠️ 副作用（已一并处理）：
#   · bias 校准探针原本**硬要求** Group 4 存在（它必须采与生产相同的系综）。
#     生产去壳 ⟹ 探针也必须去壳，那三处断言方向随之反转，见各自处的注释。
#   · `lambda_shield` 全局参数不再被注册。所有 setParameter 点本来就已用
#     `_system_has_global_parameter` 守卫（:9353/:9552/:13602/:13640/:13674/:15858），
#     无需改动。
#   · `WCA_ACCOUNTING_VERSION` 2→3：Group 4 现为**恒空组**，e_bias 实际只有 {1}。
#     `getState(groups={1,4})` 保持不变（空组贡献 0，且复活时仍正确）。
#
# 完整证据：docs/archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md §2.10、
# 数据 docs/reference_data/stage2_vanishing_truth_toluene_2026-09-02.json。
# ============================================================================
WCA_SHIELD_RETIRED = True

# 两条路径（ACE `_create_softcore_force` 与传统 `BeutlerSoftcoreBuilder.build`）
# 目前都硬编码用这组 switching/cutoff 距离构造软核 CustomNonbondedForce，LRC 积分
# 必须用完全相同的边界，否则修正的是一个跟实际采样哈密顿量不匹配的积分区间。
#
# [MEM-00h，2026-08-06] 统一到 1.0 nm、无 switching 后，switch==cutoff，下面积分
# 公式里 [r_switch, r_cutoff] 那段"switching 壳层补偿"区间宽度精确为 0（自动贡献
# 0，不需要为"switching 已关闭"另写一条积分分支），只剩 [r_cutoff, ∞) 的标准
# r^-6/r^-12 尾项——这正是"解析 LRC 不再补偿 1.0-1.2nm switching 壳层，只积分
# 1.0nm→∞"这句话在代码里的落地方式。这两个值仅作函数默认参数；实际调用（见
# build_ibs_dual_system）都会传入从活的 softcore force 读回的 template_cutoff/
# template_switch，因此哪怕只改了 _create_softcore_force 忘了改这里，行为也不会
# 悄悄分叉——但两处都改是为了让默认值本身也如实反映现状，不留一个说谎的常量。
LJ_TAIL_LRC_R_SWITCH_NM = 1.0
LJ_TAIL_LRC_R_CUTOFF_NM = 1.0


def _lj_switching_function_value(r_nm: float, r_switch_nm: float, r_cutoff_nm: float) -> float:
    """OpenMM's standard quintic switching polynomial S(r): 1 at r<=r_switch,
    0 at r>=r_cutoff, smoothly (C1-continuous) interpolating in between."""
    if r_nm <= r_switch_nm:
        return 1.0
    if r_nm >= r_cutoff_nm:
        return 0.0
    x = (r_nm - r_switch_nm) / (r_cutoff_nm - r_switch_nm)
    return 1.0 - 10.0 * x ** 3 + 15.0 * x ** 4 - 6.0 * x ** 5


@functools.lru_cache(maxsize=4096)
def _lj_softcore_tail_radial_integrals(
    lambda_vdw: float,
    alpha_lj: float,
    m_lj: float,
    sigma_nm: float,
    r_switch_nm: float = LJ_TAIL_LRC_R_SWITCH_NM,
    r_cutoff_nm: float = LJ_TAIL_LRC_R_CUTOFF_NM,
) -> Tuple[float, float]:
    """Real switching-aware + softcore-aware radial integrals for one
    (lambda_vdw, sigma_ij) pair:

        I6  = integral_{r_switch}^{r_cutoff} (1-S(r)) r^2 / D(r)   dr
            + integral_{r_cutoff}^{infinity}            r^2 / D(r)   dr
        I12 = integral_{r_switch}^{r_cutoff} (1-S(r)) r^2 / D(r)^2 dr
            + integral_{r_cutoff}^{infinity}            r^2 / D(r)^2 dr

    where D(r) = alpha_lj*sigma_nm^6*(1-lambda_vdw)^m_lj + r^6 is the exact
    softcore denominator used by both AlchemicalPotentialFactory/
    ACESoftcorePotential and BeutlerSoftcoreBuilder under
    SOFTCORE_ALPHA_CONVENTION=dimensionless_sigma_scaled_v2.  It is now
    sigma-dependent, which is why the caller must integrate per sigma group
    rather than once per lambda -- see
    _lj_tail_correction_sigma_resolved_moments.

    (BeutlerSoftcoreBuilder additionally carries a +1e-4*sigma12^6*(1-lambda)
    numerical safety floor near r=0; at r>=r_switch=1.0nm that term is
    <=1e-4*sigma^6 against a D(r)>=1 from the r^6 piece alone, i.e. completely
    negligible for this tail integral, so both paths can safely share this one
    D(r).)

    lru_cache: the same (lambda, sigma) combinations recur across the three
    consumers (IBS production build, fixed-H overlap probe, offline traditional
    MBAR u_kn recompute) and across sigma groups that share a value; the cache
    keeps the per-sigma refactor from multiplying quad() calls.

    No closed form exists once the switching polynomial and the
    lambda-shifted r^6 denominator are both present, so this integrates
    numerically (scipy.integrate.quad, handles the improper r_cutoff->inf
    integral directly). Called once per distinct lambda_vdw at system-build
    time, never per-frame, so the extra cost is negligible.
    """
    lam = float(lambda_vdw)
    alpha = float(alpha_lj)
    m = float(m_lj)
    sigma6 = float(sigma_nm) ** 6

    def _softcore_D(r):
        return alpha * sigma6 * (1.0 - lam) ** m + r ** 6

    def _integrand6_switch(r):
        return (1.0 - _lj_switching_function_value(r, r_switch_nm, r_cutoff_nm)) * r ** 2 / _softcore_D(r)

    def _integrand6_tail(r):
        return r ** 2 / _softcore_D(r)

    def _integrand12_switch(r):
        return (1.0 - _lj_switching_function_value(r, r_switch_nm, r_cutoff_nm)) * r ** 2 / _softcore_D(r) ** 2

    def _integrand12_tail(r):
        return r ** 2 / _softcore_D(r) ** 2

    i6_switch, _ = _scipy_quad(_integrand6_switch, r_switch_nm, r_cutoff_nm)
    i6_tail, _ = _scipy_quad(_integrand6_tail, r_cutoff_nm, np.inf)
    i12_switch, _ = _scipy_quad(_integrand12_switch, r_switch_nm, r_cutoff_nm)
    i12_tail, _ = _scipy_quad(_integrand12_tail, r_cutoff_nm, np.inf)
    return float(i6_switch + i6_tail), float(i12_switch + i12_tail)


def _lj_tail_lrc_coefficients_kj_mol(
    lambdas_vdw,
    sigma_nm,
    s6_per_sigma_kj_nm6,
    s12_per_sigma_kj_nm12,
    alpha_lj: float,
    m_lj: float,
    n_lj: float,
    r_switch_nm: float = LJ_TAIL_LRC_R_SWITCH_NM,
    r_cutoff_nm: float = LJ_TAIL_LRC_R_CUTOFF_NM,
) -> np.ndarray:
    """Per-lambda LJ tail correction coefficients (kJ*nm^3/mol); the actual
    per-frame energy is ``coeff[k] / V(t)`` (V in nm^3). Shared by every
    consumer (IBSSampler production sampling, the fixed-H overlap probe, and
    the offline traditional MBAR u_kn recompute) so all three always apply
    the exact same correction to the exact same lambda -- see call sites for
    the one-time construction and the per-frame `/ V(t)` division.

    🔑 [TRADITIONAL_LJ_LRC_PROTOCOL_VERSION=3] sigma-resolved sum:

        coeff[k] = 16*pi * lambda_vdw[k]**n_lj
                   * sum_b [ S12_b*I12(lambda_vdw[k], sigma_b)
                             - S6_b*I6(lambda_vdw[k], sigma_b) ]

    v2 was ``16*pi*lam**n_lj*(S12*I12(lam) - S6*I6(lam))`` with two global
    scalars, valid only while D(r) was pair-independent.  Under
    SOFTCORE_ALPHA_CONVENTION=dimensionless_sigma_scaled_v2 the softcore
    denominator is D_ij(r)=alpha_lj*sigma_ij^6*(1-lam)^m + r^6, so I6/I12 can
    no longer be factored out of the pair sum; they are evaluated per distinct
    sigma_ij group instead.  sum_b S6_b and sum_b S12_b still reproduce v2's
    S6/S12 exactly, which is what the call sites log.

    At lambda_vdw=0, lambda_vdw**n_lj is exactly 0.0 in floating point (not
    just close to it) for any n_lj>0, so coeff is exactly 0 regardless of
    I6/I12 -- the integral is skipped entirely for that state as a cheap
    optimization, not because it would otherwise be wrong.
    """
    lambdas_arr = np.asarray(lambdas_vdw, dtype=np.float64)
    sigma_arr = np.asarray(sigma_nm, dtype=np.float64).ravel()
    s6_arr = np.asarray(s6_per_sigma_kj_nm6, dtype=np.float64).ravel()
    s12_arr = np.asarray(s12_per_sigma_kj_nm12, dtype=np.float64).ravel()
    if not (sigma_arr.shape == s6_arr.shape == s12_arr.shape):
        raise ValueError(
            "sigma_nm / s6_per_sigma / s12_per_sigma 长度必须一致："
            f"{sigma_arr.shape} / {s6_arr.shape} / {s12_arr.shape}"
        )
    coeffs = np.zeros(lambdas_arr.shape[0], dtype=np.float64)
    for k, lam in enumerate(lambdas_arr):
        lam = float(lam)
        if lam == 0.0:
            continue
        acc = 0.0
        for sigma_b, s6_b, s12_b in zip(sigma_arr, s6_arr, s12_arr):
            i6, i12 = _lj_softcore_tail_radial_integrals(
                lam, float(alpha_lj), float(m_lj), float(sigma_b),
                r_switch_nm, r_cutoff_nm,
            )
            acc += s12_b * i12 - s6_b * i6
        coeffs[k] = 16.0 * math.pi * (lam ** float(n_lj)) * acc
    return coeffs


def _periodic_box_volume_nm3(box_vectors) -> float:
    """Return a validated triclinic box volume in nm^3."""
    arr = np.asarray(box_vectors, dtype=np.float64)
    if arr.shape != (3, 3) or not np.all(np.isfinite(arr)):
        raise ValueError(
            f"周期盒向量必须是有限的 (3, 3) nm 数组，实际 shape={arr.shape}。"
        )
    volume = float(abs(np.linalg.det(arr)))
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError(f"周期盒体积无效: {volume!r} nm^3")
    return volume


def _create_softcore_force(
    nb_force: openmm.NonbondedForce,
    perturbed_indices: List[int],
    environment_indices: List[int],
    lam_coul: float,
    lam_vdw: float,
    alchemical_params,
    potential_type: str = "softcore",
    reference_exclusions=None,
    particle_params_override=None,
    num_particles=None,
) -> openmm.CustomNonbondedForce:
    """
    🔑 修复版：构建 L-E 软核力 (用于 IBS CV)
    严格遵循论文 Eq.8：每个 CV 必须是独立的、无参数的势能函数。
    将 λ 值直接硬编码进表达式字符串，彻底移除 GlobalParameter 依赖。
    """
    total_particles = nb_force.getNumParticles()
    perturbed_set, env_set = set(perturbed_indices), set(environment_indices)
    
    # 🔑 核心：将 λ 格式化为高精度字符串，直接注入表达式
    lam_c_str = f"{lam_coul:.8f}"
    lam_v_str = f"{lam_vdw:.8f}"
    
    # 调用工厂生成完整软核表达式 (含 Coulomb + VdW)
    expr, resolved_params = AlchemicalPotentialFactory.build(
        potential_type, alchemical_params, lam_c_str, lam_v_str
    )
    
    sc_force = openmm.CustomNonbondedForce(expr)
    for p in ["q", "sigma", "epsilon"]:
        sc_force.addPerParticleParameter(p)
        
    # 🔑 移除 addGlobalParameter，λ 已硬编码，CV 成为纯坐标函数 U'_k(x)
    
    for i in range(total_particles):
        if particle_params_override:
            if isinstance(particle_params_override, dict):
                q, sig, eps = particle_params_override.get(i, nb_force.getParticleParameters(i))
            elif i < len(particle_params_override):
                q, sig, eps = particle_params_override[i]
            else:
                q, sig, eps = nb_force.getParticleParameters(i)
        else:
            q, sig, eps = nb_force.getParticleParameters(i)
            
        sc_force.addParticle([
            q.value_in_unit(unit.elementary_charge),
            sig.value_in_unit(unit.nanometer),
            eps.value_in_unit(unit.kilojoule_per_mole)
        ])
        
    sc_force.addInteractionGroup(list(perturbed_set), list(env_set))
    sc_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    is_dexp = str(potential_type or "").strip().lower() == "dexp"
    if is_dexp:
        cutoff_nm = DEXP_VDW_CUTOFF_NM
        switch_width_nm = DEXP_VDW_SWITCH_WIDTH_NM
        switch_nm = cutoff_nm - switch_width_nm
        if not (cutoff_nm > switch_nm > 0.0):
            raise ValueError(
                "DEXP cutoff/switch 配置无效："
                f"cutoff={cutoff_nm}, switch_width={switch_width_nm}, "
                f"derived_switch={switch_nm} nm"
            )
        use_switching = True
    else:
        # [MEM-00h] 统一到基础力场的 1.0 nm、无 switching——见 SOFTCORE_CUTOFF_NM
        # 定义处的理由。switch_nm 特意设成等于 cutoff_nm（而不是干脆不设）：
        # 下游 build_ibs_dual_system 会读 cv_template.getSwitchingDistance() 去算
        # LJ 解析尾项的积分下限（见 _lj_softcore_tail_radial_integrals），switch==
        # cutoff 让那段"switching 壳层"积分区间宽度精确为 0，自动退化成纯
        # cutoff→∞ 的标准尾项，不需要为"switching 已关闭"另开一条积分公式分支。
        cutoff_nm = SOFTCORE_CUTOFF_NM
        switch_nm = SOFTCORE_CUTOFF_NM
        use_switching = False
    sc_force.setCutoffDistance(cutoff_nm * unit.nanometer)
    # 为什么禁用原生开关：LJ+Coulomb 拼在同一个表达式里，OpenMM 的原生解析修正对
    # Coulomb 的 1/r 尾项会发散，实测直接让 CUDA 崩溃（见 AUDIT_STATUS.md）。这不代表
    # LJ tail 不补偿——默认 softcore/ACE 路径下，build_ibs_dual_system 会在此后为每个
    # λ_vdw 态单独算出手写解析尾项系数挂在 ibs_wrapper 上，由 IBSSampler 每帧加进
    # target_energies；是否施加、按哪个协议版本，都记在 final_results.json 的
    # `lj_long_range_dispersion_correction` / `traditional_lj_lrc_protocol_version` 里。
    # APBS 外部项只管静电/连续介质类长程修正，跟这个 LJ tail 项无关。
    # 以上是纯静态设计理由，不再打进日志（本函数每建一次软核力就调一次）。
    if is_dexp:
        # dexp 路线例外：尾项**真的没补**，这是运行时的实质缺口，必须留在日志里。
        print_once(
            "vdw_softcore_lj_tail_dexp",
            "  [WARN] [VDW softcore] potential_type='dexp'：手写解析 LJ 尾项尚未验证，会显式跳过"
            "——本次 run 的 LJ tail 完全不补偿。",
        )
    sc_force.setUseLongRangeCorrection(False)
    
    if reference_exclusions:
        for p1, p2 in reference_exclusions:
            p1, p2 = int(p1), int(p2)
            if p1 < total_particles and p2 < total_particles:
                sc_force.addExclusion(p1, p2)
                
    sc_force.setUseSwitchingFunction(use_switching)
    sc_force.setSwitchingDistance(switch_nm * unit.nanometer)
    return sc_force

def _normalize_softcore_params(
    softcore_params: ACESoftcorePotential,
    n_perturbed: int,
    explicit: bool = False,
) -> ACESoftcorePotential:
    requested_lj = float(getattr(softcore_params, "alpha_lj", float("nan")))
    requested_coul = float(getattr(softcore_params, "alpha_coul", float("nan")))
    if explicit:
        # 🔑 [0831issue P2 顺带发现的真 bug] 这里原来写
        # `power_lj=getattr(softcore_params, "power_lj", (2, 2))`，但
        # `ACESoftcorePotential.__init__` 把 `power_lj=(m, n)` **拆成** `m_lj`/`n_lj`
        # 两个属性存起来，**没有** `power_lj` 属性 —— 于是这个 getattr 永远落到默认值
        # `(2, 2)`，用户显式给的 λ 指数在 explicit 路径上被**静默重置**。
        # 这与本函数 provenance 里那句 "Softcore parameters are no longer silently
        # overwritten by a fixed production default" 直接矛盾。
        # 生产配置恰好就是 (2,2)/(1,1)，所以当前数值无变化；但任何非默认指数此前都
        # 传不进来（也因此下面那道 n_lj 校验会永远看到 2）。改为读真实属性。
        _m_lj = int(getattr(softcore_params, "m_lj", 2))
        _n_lj_in = int(getattr(softcore_params, "n_lj", 2))
        _m_coul = int(getattr(softcore_params, "m_coul", 1))
        _n_coul = int(getattr(softcore_params, "n_coul", 1))
        normalized = ACESoftcorePotential(
            alpha_lj=requested_lj,
            alpha_coul=requested_coul,
            power_lj=(_m_lj, _n_lj_in),
            power_coul=(_m_coul, _n_coul),
        )
        source = "explicit_user_or_cached"
    else:
        adaptive = ACESoftcorePotential.optimize_alpha(max(int(n_perturbed), 1))
        normalized = ACESoftcorePotential.from_dict(adaptive)
        source = "adaptive_by_perturbed_atom_count"
    # 🔑 [0831issue P2] `power_lj = (m, n_lj)` 的 `n_lj` 必须为正。
    # `_lj_tail_lrc_coefficients_kj_mol` 里那句 `if lam == 0.0: continue` 的合法性
    # **完全依赖 n_lj > 0**（λ=0 时 λ^n_lj = 0，尾项系数确实为 0）；n_lj == 0 时
    # λ^0 = 1，λ=0 态的尾项系数不该是 0，那个 continue 会把它静默置零。
    # `n_lj` 由用户经 `power_lj=(m_lj, n_lj)` 传入（`ACESoftcorePotential` 把它拆成
    # `m_lj`/`n_lj` 两个属性），而 `from_dict` 不做范围校验，所以这里 fail closed。
    # 读法与真正的消费点 `build_ibs_dual_system` 完全一致：`getattr(..., "n_lj", 2)`。
    # 生产配置是 power_lj=[2, 2]，本校验对它无影响（实测 power_lj=[m, 0] 时
    # COEF([0.0, 0.5]) 返回 [0.0, -0.1575]，λ=0 的正确值应为 -0.1575）。
    _n_lj_raw = getattr(normalized, "n_lj", 2)
    try:
        _n_lj = int(_n_lj_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"softcore n_lj={_n_lj_raw!r} 无法解析成整数指数"
        ) from exc
    if _n_lj <= 0:
        raise ValueError(
            f"softcore 的 λ 指数 n_lj={_n_lj} <= 0（来自 power_lj 的第二个分量）。"
            "LJ 长程尾项系数的 λ=0 短路（_lj_tail_lrc_coefficients_kj_mol 里的 "
            "`if lam == 0.0: continue`）只在 n_lj > 0 时成立，n_lj<=0 会让 λ=0 态的"
            "尾项被静默置零。拒绝以静默错值继续。"
        )
    normalized.provenance = {
        "source": source,
        "n_perturbed_atoms": int(n_perturbed),
        "lambda_exponent_n_lj": int(_n_lj),
        "input_alpha_lj": requested_lj,
        "input_alpha_coul": requested_coul,
        "alpha_lj": float(normalized.alpha_lj),
        "alpha_coul": float(normalized.alpha_coul),
        "alpha_convention": ACESoftcorePotential.ALPHA_CONVENTION,
        "note": "Softcore parameters are no longer silently overwritten by a fixed production default.",
    }
    # 去重键由消息文本本身派生：键和值不可能漂移，只压掉逐字节相同的重复。参数一变
    # 就重新打一行。本函数每次建系都被调用，一次 4W53 run 里同一行重复了 50 次。
    _msg = (
        f"  [Softcore 参数] {source}: "
        f"alpha_lj={normalized.alpha_lj:.3f} nm^6, alpha_coul={normalized.alpha_coul:.3f} nm^2 "
        f"(输入值: LJ={requested_lj:.3f}, Coul={requested_coul:.3f})"
    )
    print_once(f"softcore_params|{_msg}", _msg)
    return normalized


def _normalize_alchemical_params(alchemical_params, potential_type: str, n_perturbed: int):
    if potential_type == "dexp":
        if isinstance(alchemical_params, DEXPSurrogatePotential):
            return alchemical_params
        return DEXPSurrogatePotential.from_dict(alchemical_params or {})
    if isinstance(alchemical_params, ACESoftcorePotential):
        return _normalize_softcore_params(alchemical_params, n_perturbed, explicit=True)
    explicit = bool(
        isinstance(alchemical_params, dict)
        and ("alpha_lj" in alchemical_params or "alpha_coul" in alchemical_params)
    )
    return _normalize_softcore_params(
        ACESoftcorePotential.from_dict(alchemical_params or {}),
        n_perturbed,
        explicit=explicit,
    )


def _compute_reference_com(
    positions,
    system: openmm.System,
    atom_indices: List[int],
) -> Tuple[float, float, float]:
    """质量加权 COM。

    ⚠️ [LIGAND_COM_RESTRAINT_PROTOCOL_VERSION=2] 唯一的调用点（Group 5 COM 限制力）
    已被移除，本函数目前无调用方，保留仅为将来可能需要的**周期**COM 限制。
    若要重新启用：绝不可再把它的返回值当作非周期绝对锚点使用——那正是被移除的
    P0 缺陷（见该常量的注释）。可用写法见 `build_co_alchemical_ion_restraint`。
    """
    pos_nm = positions.value_in_unit(unit.nanometer) if hasattr(positions, "value_in_unit") else np.asarray(positions, dtype=float)
    masses = np.array(
        [system.getParticleMass(int(idx)).value_in_unit(unit.dalton) for idx in atom_indices],
        dtype=float,
    )
    if np.any(masses <= 0.0):
        masses = np.ones(len(atom_indices), dtype=float)
    coords = np.asarray([pos_nm[int(idx)] for idx in atom_indices], dtype=float)
    ref_com = np.average(coords, axis=0, weights=masses)
    return tuple(float(x) for x in ref_com)


def _resolve_periodic_box_vectors(box_vectors, topology=None, system: Optional[openmm.System] = None):
    """
    解析并校验周期性盒子。

    优先级：
    1. 显式传入的 box_vectors
    2. topology 上的盒子
    3. system 默认盒子
    """
    candidates = []
    if box_vectors is not None:
        candidates.append(box_vectors)
    if topology is not None and hasattr(topology, "getPeriodicBoxVectors"):
        try:
            topo_box = topology.getPeriodicBoxVectors()
        except Exception:
            topo_box = None
        if topo_box is not None:
            candidates.append(topo_box)
    if system is not None:
        try:
            sys_box = system.getDefaultPeriodicBoxVectors()
        except Exception:
            sys_box = None
        if sys_box is not None:
            candidates.append(sys_box)

    for candidate in candidates:
        try:
            ax = candidate[0][0].value_in_unit(unit.nanometer)
            by = candidate[1][1].value_in_unit(unit.nanometer)
            cz = candidate[2][2].value_in_unit(unit.nanometer)
        except Exception:
            continue
        if ax > 0.0 and by > 0.0 and cz > 0.0:
            return candidate
    return None


def _system_requires_periodic_box(system: openmm.System) -> bool:
    """判断系统内是否存在依赖有效周期性盒子的力。"""
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            if force.getNonbondedMethod() in (
                openmm.NonbondedForce.CutoffPeriodic,
                openmm.NonbondedForce.PME,
                openmm.NonbondedForce.LJPME,
                openmm.NonbondedForce.Ewald,
            ):
                return True
        if isinstance(force, openmm.CustomNonbondedForce):
            if force.getNonbondedMethod() == openmm.CustomNonbondedForce.CutoffPeriodic:
                return True
    return False


def _system_has_global_parameter(system: openmm.System, name: str) -> bool:
    """检查 System 里任意 Force 是否已注册某个 global parameter。

    用于窗口管理器的不同系统变体（例如 build_ibs_dual_system 的 VDW softcore
    路径 vs. build_shadow_coul_ibs_system 的 Shadow-Coulomb 路径）共享同一段
    窗口调度代码时，对只在某些变体里存在的全局参数（如 lambda_shield）做
    存在性检查，避免对不含该参数的系统调用 context.setParameter 直接报错。
    """
    for force in system.getForces():
        if not hasattr(force, "getNumGlobalParameters"):
            continue
        for i in range(force.getNumGlobalParameters()):
            if force.getGlobalParameterName(i) == name:
                return True
    return False


def _collect_softcore_exclusions(
    nb_force: openmm.NonbondedForce,
    system: openmm.System,
    perturbed_indices: List[int],
    environment_indices: List[int],
    reference_exclusions=None,
) -> List[Tuple[int, int]]:
    """
    为配体-环境软核相互作用收集完整排除表。
    仅保留跨组(L-E)对，覆盖 1-2/1-3/1-4 与约束对，避免 CV 重复计算。
    """
    perturbed_set = set(int(i) for i in perturbed_indices)
    env_set = set(int(i) for i in environment_indices)
    exclusion_pairs = set()

    def _maybe_add_pair(p1, p2):
        p1, p2 = int(p1), int(p2)
        if p1 == p2:
            return
        in_cross_group = (
            (p1 in perturbed_set and p2 in env_set) or
            (p2 in perturbed_set and p1 in env_set)
        )
        if in_cross_group:
            exclusion_pairs.add((min(p1, p2), max(p1, p2)))

    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                p1, p2, _, _ = force.getBondParameters(i)
                _maybe_add_pair(p1, p2)
        elif isinstance(force, openmm.CustomBondForce):
            for i in range(force.getNumBonds()):
                p1, p2, _ = force.getBondParameters(i)
                _maybe_add_pair(p1, p2)
        elif isinstance(force, openmm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                p1, _, p3, _, _ = force.getAngleParameters(i)
                _maybe_add_pair(p1, p3)
        elif isinstance(force, openmm.CustomAngleForce):
            for i in range(force.getNumAngles()):
                p1, _, p3, _ = force.getAngleParameters(i)
                _maybe_add_pair(p1, p3)
        elif isinstance(force, openmm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, _, _, p4, _, _, _ = force.getTorsionParameters(i)
                _maybe_add_pair(p1, p4)
        elif isinstance(force, openmm.CustomTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, _, _, p4, _ = force.getTorsionParameters(i)
                _maybe_add_pair(p1, p4)
        elif hasattr(openmm, "RBTorsionForce") and isinstance(force, openmm.RBTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, _, _, p4, *_ = force.getTorsionParameters(i)
                _maybe_add_pair(p1, p4)

    for i in range(system.getNumConstraints()):
        p1, p2, _ = system.getConstraintParameters(i)
        _maybe_add_pair(p1, p2)

    for i in range(nb_force.getNumExceptions()):
        p1, p2, _, _, _ = nb_force.getExceptionParameters(i)
        _maybe_add_pair(p1, p2)

    if reference_exclusions:
        for p1, p2 in reference_exclusions:
            _maybe_add_pair(p1, p2)

    return sorted(exclusion_pairs)


def _estimate_wca_shield_parameters(
    particle_params,
    perturbed_indices: List[int],
    environment_indices: List[int],
) -> Dict[str, float]:
    """Estimate conservative lambda-WCA shield parameters from LJ sigma values."""
    lig_sigmas = []
    env_sigmas = []
    for idx in perturbed_indices:
        _q, sig, eps = particle_params[int(idx)]
        eps_val = eps.value_in_unit(unit.kilojoule_per_mole)
        sig_val = sig.value_in_unit(unit.nanometer)
        if eps_val > 1.0e-8 and sig_val > 1.0e-8:
            lig_sigmas.append(sig_val)
    for idx in environment_indices:
        _q, sig, eps = particle_params[int(idx)]
        eps_val = eps.value_in_unit(unit.kilojoule_per_mole)
        sig_val = sig.value_in_unit(unit.nanometer)
        if eps_val > 1.0e-8 and sig_val > 1.0e-8:
            env_sigmas.append(sig_val)

    if lig_sigmas and env_sigmas:
        mixed = []
        env_sample = np.asarray(env_sigmas, dtype=float)
        if env_sample.size > 5000:
            env_sample = np.quantile(env_sample, np.linspace(0.05, 0.95, 64))
        for sig_l in lig_sigmas:
            mixed.extend((0.5 * (sig_l + env_sample)).tolist())
        sigma_ref = float(np.percentile(np.asarray(mixed, dtype=float), 10.0))
        source = "lj_sigma_10th_percentile"
    else:
        sigma_ref = 0.22
        source = "fallback_default_no_lj_sigmas"

    rc = float(np.clip(0.85 * sigma_ref, 0.18, 0.32))
    eps_wca = float(np.clip(1.0 + 0.015 * max(len(perturbed_indices) - 20, 0), 1.0, 2.5))
    return {
        "rc_nm": rc,
        "eps_wca_kJ_mol": eps_wca,
        "sigma_reference_nm": sigma_ref,
        "source": source,
        "n_ligand_lj_atoms": int(len(lig_sigmas)),
        "n_environment_lj_atoms": int(len(env_sigmas)),
    }


def diagnose_softcore_cv_values(
    context: openmm.Context,
    ibs_wrapper: "IBSBiasForce",
    lambdas_coul: List[float],
    lambdas_vdw: List[float],
    prefix: str = "",
    sampler: "IBSSampler" = None,
):
    """打印当前窗口所有软核 CV / Boresch CV 的原始数值，帮助定位异常背景。"""
    print(f"\n🔍 [{prefix}] 软核 CV 数值诊断:")
    try:
        state_base = context.getState(getEnergy=True, groups=set(IBS_E_BASE_FORCE_GROUPS))
        e_base = state_base.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    except Exception as e:
        e_base = float("nan")
        print(f"  e_base 读取失败: {e}")

    try:
        state_bias = context.getState(getEnergy=True, groups=set(IBS_E_BIAS_FORCE_GROUPS))
        e_bias = state_bias.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    except Exception:
        e_bias = float("nan")

    print(
        f"  e_base(Group0+2+3+5)={e_base:.3f} kJ/mol | "
        f"e_bias(Group1+4)={e_bias:.3f} kJ/mol"
    )

    sampled_interactions = None
    if sampler is not None:
        try:
            sampled_interactions = sampler.get_raw_interaction_energies()
        except Exception as e:
            print(f"  CV 探针读取失败: {e}")
            return

    interaction_values = []
    for k, (lc, lv) in enumerate(zip(lambdas_coul, lambdas_vdw)):
        if sampler is not None:
            e_int = sampled_interactions[k] if k < len(sampled_interactions) else float("nan")
        else:
            try:
                cv_vals = ibs_wrapper.get_force().getCollectiveVariableValues(context)
            except Exception as e:
                print(f"  CV 读取失败: {e}")
                return
            # [EXP-031 S1] 不再按 2*k 硬算 —— 那个假设在 residual_enabled（前面先
            # 注册了 exp025_residual_basis，整体偏移 1）和 omit 形态（每态只有 1 个
            # CV）下都是错的，而 CustomCVForce 读错索引不会报错、只会静默给错数。
            # `_int_cv_indices` 是 addCollectiveVariable() 的**真实返回值**表，且与
            # `_evaluate_interaction_energies_live()` 用的是同一份，形态无关。
            int_cv_indices = getattr(ibs_wrapper, "_int_cv_indices", None) or []
            if k < len(int_cv_indices):
                idx_int = int(int_cv_indices[k])
                e_int = cv_vals[idx_int] if idx_int < len(cv_vals) else float("nan")
            else:
                e_int = float("nan")
        e_rest = 0.0
        interaction_values.append(float(e_int))
        e_tot = e_base + e_int + e_rest if np.isfinite(e_base) else float("nan")
        print(
            f"  state {k:>2d} | lam_c={lc:7.4f} lam_v={lv:7.4f} | "
            f"e_int={e_int:14.3f} | e_rest={e_rest:10.3f} | e_total={e_tot:14.3f}"
        )

    interaction_values = np.asarray(interaction_values, dtype=float)
    finite = interaction_values[np.isfinite(interaction_values)]
    if finite.size >= 2:
        adjacent = np.abs(np.diff(finite))
        max_adjacent = float(np.max(adjacent))
        total_span = float(np.max(finite) - np.min(finite))
        print(f"  ΔU诊断: 窗口跨度={total_span:.1f} kJ/mol | 最大相邻ΔU={max_adjacent:.1f} kJ/mol")
        if max_adjacent > 50.0:
            print("  [WARN] 相邻 λ 能量差远超 50 kJ/mol，IBS 权重很可能塌缩；建议显著增加该阶段 λ 状态数。")

def _serialize_ibs_common_system(window_system: openmm.System) -> str:
    """Return U_common from an assembled VDW-IBS window.

    ``system_template`` is the original fully interacting system, so cloning it
    and adding a state CV would double-count ligand-environment interactions.
    The assembled window is the authoritative source: its main NB interaction
    is already zeroed and groups 0/2/3/5 are exactly U_common.  Work on a clone
    and remove only the top-level IBS mixture (group 1) and WCA sampling bias
    (group 4), with strict structural assertions.

    This WCA-less system is for the *path/lambda-grid* overlap probe
    (``probe_bidirectional_overlap``) only -- it deliberately answers "is the
    underlying physical free energy landscape connected enough", not "what
    does production actually sample". Do not reuse it for bias-CV
    calibration purposes; see ``_serialize_ibs_common_plus_wca_system`` for
    that (production's real sampled ensemble includes the WCA shield).
    """
    common = XmlSerializer.clone(window_system)
    removed = {1: 0, 4: 0}
    for force_index in reversed(range(common.getNumForces())):
        group = int(common.getForce(force_index).getForceGroup())
        if group in removed:
            common.removeForce(force_index)
            removed[group] += 1
    # 🔑 [WCA_SHIELD_RETIRED，2026-09-02] Group-4 力数的期望从"恰好 1"放宽到"0 或 1"。
    #
    # 本函数的职责是**剥掉**壳并返回 U_common —— 无论壳在不在都该能剥：
    #   · 退役后的生产系统没有 Group 4（0 个）；
    #   · 复活防护壳、或喂进历史/测试系统时会有 1 个。
    # 原来的 `== {1:1, 4:1}` 想守的是"壳该在却没建出来"，而壳退役后这个失效模式
    # 已不存在。真正要守的反向失效（"退役了却还建了壳"）由另外两处断言抓：
    # `_serialize_ibs_common_plus_wca_system` 与 `_build_fixed_state_simulation`。
    # Group 1 仍然必须**恰好 1 个**——那条唯一性没有放宽。
    if removed.get(1) != 1 or removed.get(4, 0) not in (0, 1):
        raise RuntimeError(
            f"无法从 VDW-IBS 窗口唯一导出 U_common：移除的顶层 force 数={removed}"
            f"（期望 group1 恰好 1、group4 为 0 或 1；"
            f"WCA_SHIELD_RETIRED={WCA_SHIELD_RETIRED}）"
        )
    residual = [
        int(common.getForce(i).getForceGroup())
        for i in range(common.getNumForces())
        if int(common.getForce(i).getForceGroup()) in (1, 4)
    ]
    if residual:
        raise RuntimeError(f"U_common 中仍残留 IBS/WCA force group: {residual}")
    return XmlSerializer.serialize(common)


def _serialize_ibs_common_plus_wca_system(window_system: openmm.System) -> str:
    """Return U_common + WCA_window (Group 4 kept; only Group 1 IBS mixture
    removed) from an assembled VDW-IBS window.

    Production actually samples ``U_common + WCA_window(lambda_shield) +
    CV_k`` (Group 1), not ``U_common + CV_k`` alone -- the WCA shield only
    cancels out of *same-frame, same-window* energy differences between
    states (it has the same lambda_shield for every k in a window), it does
    not disappear from the sampled *trajectory*. A fixed-H probe built on
    ``_serialize_ibs_common_system`` (which strips Group 4 entirely) samples
    a genuinely different conformational ensemble than production, so a
    delta_f measured on it cannot be used to calibrate the bias CV's f_k --
    only for the separate path/lambda-grid overlap question that function
    already serves. This variant keeps Group 4 so the bias-calibration probe
    (``probe_bidirectional_overlap_for_bias_calibration``) samples the exact
    ensemble it needs to reweight, once the caller sets ``lambda_shield`` to
    the same value production used for this window.
    """
    common_plus_wca = XmlSerializer.clone(window_system)
    removed = {1: 0}
    for force_index in reversed(range(common_plus_wca.getNumForces())):
        group = int(common_plus_wca.getForce(force_index).getForceGroup())
        if group in removed:
            common_plus_wca.removeForce(force_index)
            removed[group] += 1
    if removed != {1: 1}:
        raise RuntimeError(
            f"无法从 VDW-IBS 窗口唯一导出 U_common+WCA：移除的顶层 force 数={removed}"
        )
    residual = [
        int(common_plus_wca.getForce(i).getForceGroup())
        for i in range(common_plus_wca.getNumForces())
        if int(common_plus_wca.getForce(i).getForceGroup()) == 1
    ]
    if residual:
        raise RuntimeError(f"U_common+WCA 中仍残留 IBS force group: {residual}")
    has_group4 = any(
        int(common_plus_wca.getForce(i).getForceGroup()) == 4
        for i in range(common_plus_wca.getNumForces())
    )
    # 🔑 [WCA_SHIELD_RETIRED，2026-09-02] 断言方向随退役反转。
    # 本函数的全部意义是"让 bias 校准探针采到与生产**完全相同**的系综"。
    # 生产已不含 Group 4（壳退役，占 stage2 溶剂腿误差 98%），所以探针里出现
    # Group 4 才是错的 —— 那会让探针采一个生产从不采的系综，而这正是本函数
    # docstring 里警告过的那个失效模式（"samples a genuinely different
    # conformational ensemble than production"），只是方向反过来了。
    # 退役状态下本函数与 `_serialize_ibs_common_system` 等价，**刻意不删**：
    # 调用点、缓存键名与复活路径都保持原样。
    if WCA_SHIELD_RETIRED:
        if has_group4:
            raise RuntimeError(
                "WCA_SHIELD_RETIRED=True 但导出的 U_common(+WCA) 里仍存在 Group 4 —— "
                "生产采样已不含防护壳，探针含壳会采到生产从不采的系综，拒绝继续。"
                "若要复活防护壳，翻 WCA_SHIELD_RETIRED 而不是绕过本断言。"
            )
    elif not has_group4:
        raise RuntimeError("U_common+WCA 导出后缺少 Group 4 WCA 防护壳，无法用于 bias 校准探针。")
    return XmlSerializer.serialize(common_plus_wca)


OPENMM_CUSTOM_CV_MAX_VARIABLES = 32
IBS_DUAL_CVS_PER_LAMBDA_STATE = 2
IBS_DUAL_MAX_LAMBDA_STATES = (
    OPENMM_CUSTOM_CV_MAX_VARIABLES // IBS_DUAL_CVS_PER_LAMBDA_STATE
)


# 🔑 [EXP-031 S1, 2026-09-04] Group-1 偏置力形态开关：是否**不再注册**那 K 个恒零的
# `cv_k_rest` 占位 CV。
#
# 为什么它们恒零：`cv_k_rest` 是 `openmm.CustomExternalForce("0")`，**一个粒子都没
# 往里加过**（见 build_ibs_dual_system / build_shadow_coul_ibs_system 的注册处）。
# 它只是论文 Eq.8 里"限制项"的占位槽；真正的物理 Boresch 限制在 Group 3 的独立力上
# （`_add_physical_boresch_restraint`），`diagnose_softcore_cv_values` 里的
# `e_rest = 0.0` 也是写死的。也就是说这 K 个 CV 对能量和力的贡献严格为 0。
#
# 为什么值得摘：`CustomCVForce` 是**每个 CV 一次串行的 inner-context 力求值**，每个
# CV 恒定花 21-25 µs，且**与体系大小无关**（EXP-031 S0 实测 6282 / 24498 / 71262
# 三个尺寸：21.3 / 21.4 / 25.2 µs 每 CV）。生产 K=16 时 32 个 CV 里有一半是纯占位，
# 实测占整步 15%：71262 原子水盒、PME 1.0 nm、2 fs、CUDA/mixed 下，16 个恒零
# `cv_k_rest` 单独就要 +0.414 ms/step（生产形态 2.691 ms/step，无偏置基线
# 0.753 ms/step）。证据：experiments/EXP-031_ibs_bias_fusion/results/s0_b1_cv_scaling.log
#
# ⚠️ 翻这个开关会改变 Group-1 的 CustomCVForce **表达式字符串与 CV 集合**，因此
# system XML 指纹变化 → `dual_window_*` / resume 产物全部失配。默认保持 False
# （legacy 形态），既有测试与"与主线逐字节相同"的回归因此全部不受影响。真要在生产
# 里打开，必须同时升 `IBS_BIAS_PROTOCOL_VERSION` 并显式写出缓存作废代价。
#
# ⚠️ 打开它**不等于**把单窗口 λ 态数上限从 16 抬到 32：CV 预算确实从 2K≤32 松到
# K≤32，但探针路径的 `PROBE_MAX_LAMBDA_STATES = 16` 来自**力组**上限
# （`OPENMM_MAX_FORCE_GROUP - PROBE_FORCE_GROUP_BASE + 1`），是另一条**独立**约束，
# 本开关碰不到它。生产实际可用态数是两者的较小值。
#
# 🔑 [2026-09-05] 默认值 False → **True**（用户决定）。依据：
#   收益  真 Atenolol 膜体系（45354 原子、41 原子真配体、K=16、CUDA/mixed）实测
#         整步 2.086 → 1.709 ms/step，**1.2206×**（三轮 1.2128/1.2206/1.2348，
#         轮间散度 1.80%，闸门 5%）。省下的 0.377 ms 与 71262 原子水盒上省的
#         0.377 ms **完全一致** —— 印证它就是"每 CV 21-25 µs 的调度税"，与体系
#         大小无关。
#   物理  Group-1 能量在**三条独立路径**上逐比特相同：真配体膜体系、
#         `build_shadow_coul_ibs_system`（`tests/test_shadow_coul_ibs_builder.py`）、
#         以及 71262 原子水盒（S1 原验证，Reference/CUDA 三档 24 行全 bitwise）。
#   测试  legacy 与 omit 两形态各 1830 passed；翻默认值前先解除了两条零覆盖：
#         `tests/test_core_physics_numerics.py::_build_bias_context` 改成形态感知、
#         shadow-coul builder 补齐首份覆盖。
#   代价  见下面 IBS_BIAS_PROTOCOL_VERSION 的 v33 说明：system XML 变 ⟹ 现有全部
#         dual_window_*/resume 产物失配、需重跑 GPU。
# 完整记录：experiments/EXP-031_ibs_bias_fusion/PLAN_EXP-031_ibs_bias_fusion.md §10/§12
IBS_BIAS_OMIT_ZERO_REST_CVS = True

def ibs_dual_cvs_per_lambda_state(omit_zero_rest_cvs: Optional[bool] = None) -> int:
    """每个 λ 态在 Group-1 `CustomCVForce` 里真正占掉的 CV 个数。

    `omit_zero_rest_cvs=None` 表示"按当前模块级形态开关取值"。传显式布尔值是为了让
    调用方（和测试）能在不改全局开关的前提下问"另一种形态下是多少"。

    注意：`IBS_DUAL_CVS_PER_LAMBDA_STATE` / `IBS_DUAL_MAX_LAMBDA_STATES` 这两个
    **模块常量的默认值刻意不随开关变化**（`tests/test_probe_and_hysteresis_limits.py`
    断言 `PROBE_MAX_LAMBDA_STATES == IBS_DUAL_MAX_LAMBDA_STATES == 16`，那条断言检查
    的是"两条数值巧合相同的独立限制没有被混为一谈"）。需要按形态取值的地方一律走这
    两个 helper。
    """
    if omit_zero_rest_cvs is None:
        omit_zero_rest_cvs = IBS_BIAS_OMIT_ZERO_REST_CVS
    return 1 if omit_zero_rest_cvs else IBS_DUAL_CVS_PER_LAMBDA_STATE


def ibs_dual_max_lambda_states(omit_zero_rest_cvs: Optional[bool] = None) -> int:
    """当前（或指定）形态下，单个 IBS ensemble 的 CV 预算允许的最大 λ 态数。

    这**只是** `CustomCVForce` 的 32-CV 预算算出来的上限，不是 λ 态数的物理上限，
    也不是探针路径的上限（见 `IBS_BIAS_OMIT_ZERO_REST_CVS` 的第二条 ⚠️）。
    """
    return OPENMM_CUSTOM_CV_MAX_VARIABLES // ibs_dual_cvs_per_lambda_state(
        omit_zero_rest_cvs
    )



def build_ibs_dual_system(
    system: openmm.System,
    topology,
    perturbed_indices: List[int],
    lambdas_coul: List[float],
    lambdas_vdw: List[float],
    alchemical_params,
    potential_type: str = "softcore",
    restraint_params: Optional[Dict] = None,
    temperature: openmm.unit.Quantity = 300 * openmm.unit.kelvin,
    prefix: str = "abfe_dual",
    box_vectors=None,
    reference_positions=None,
    dispersion_protocol: Optional[str] = None,
    # [B6-FIX] 这条腿的环境（soluble/membrane）。非 legacy 色散路线必须给，
    # 否则 `resolve_leg_dispersion_implementation` 会 raise —— 刻意不设可猜的默认值。
    environment_type: Optional[str] = None,
    *,
    # EXP-025 G4 Layer-2 (2026-08-13): optional, disabled-by-default native
    # shared residual basis, threaded straight into IBSBiasForce (see that
    # class's docstring for the exact contract). Every EXISTING call site
    # that never passes these keeps IBSBiasForce.residual_enabled=False and
    # therefore the byte-identical Group-1 expression -- verified by
    # scripts/test_exp025_g4_ibsbiasforce_native_residual.py, not just
    # asserted here.
    residual_basis_force: Optional[openmm.Force] = None,
    residual_state_coefficients: Optional[Sequence[float]] = None,
    residual_energy_offset_kj_mol: float = 0.0,
) -> Tuple[openmm.System, 'IBSBiasForce']:
    """
    构建双λ IBS 采样系统 (终极修复版：彻底剥离 Group 0 的 λ 依赖)
    """
    if len(lambdas_coul) != len(lambdas_vdw):
        raise ValueError("lambdas_coul 与 lambdas_vdw 必须等长")
    if len(lambdas_coul) < 2:
        raise ValueError("IBS dual-lambda ensemble 至少需要两个状态")
    # [EXP-031 S1] CV 预算按 Group-1 的**当前形态**算：摘掉恒零 cv_k_rest 后每态只占
    # 1 个 CV。legacy 形态下 ibs_dual_* helper 返回的就是 2 / 16，与改动前逐值相同。
    _omit_zero_rest_cvs = IBS_BIAS_OMIT_ZERO_REST_CVS
    _cvs_per_state = ibs_dual_cvs_per_lambda_state(_omit_zero_rest_cvs)
    _max_lambda_states = ibs_dual_max_lambda_states(_omit_zero_rest_cvs)
    if len(lambdas_vdw) > _max_lambda_states:
        raise RuntimeError(
            "单个 IBS ensemble 的 lambda 状态过多："
            f"K={len(lambdas_vdw)}, 每态 {_cvs_per_state} 个 CV，"
            f"将超过 OpenMM CustomCVForce 的 {OPENMM_CUSTOM_CV_MAX_VARIABLES}-CV 上限。"
            "请把 vanishing 域划为多个物理子区间；禁止使用单一 [0:K] ensemble。"
            "\n\n[WARN] 这条限制来自 **Group-1 IBS 混合偏置力**（CustomCVForce 一次最多引用 "
            f"{OPENMM_CUSTOM_CV_MAX_VARIABLES} 个 CV），**不是** λ 态数本身的物理上限。"
            "逐态独立固定-λ 采样（INDEPENDENT_ENDPOINT_PROTOCOL_VERSION）根本不用这个"
            "混合偏置力——它只需要 λ 无关的 U_common 和逐态的 _int_cv_force_xmls[k]。"
            "若你要的是独立采样而非 IBS，请按 <=%d 态分块调用本函数、再把各块的 "
            "CV/LRC 拼起来（重叠态可用来校验同一 λ 在不同块建出的 CV 逐字节相同），"
            "不要为了绕过这条限制去改 IBS 偏置力的结构。" % _max_lambda_states
        )
    system = ensure_owned_system(system)
    new_sys = ensure_owned_system(XmlSerializer.clone(system))
    resolved_box = _resolve_periodic_box_vectors(box_vectors, topology=topology, system=new_sys)
    if _system_requires_periodic_box(new_sys):
        if resolved_box is None:
            raise ValueError("系统包含周期性非键力，但未提供有效的周期性盒子。")
        new_sys.setDefaultPeriodicBoxVectors(*resolved_box)

    num_atoms = new_sys.getNumParticles()
    perturbed_set = set(perturbed_indices)
    env_indices = [i for i in range(num_atoms) if i not in perturbed_set]
    alchemical_params = _normalize_alchemical_params(
        alchemical_params, potential_type, len(perturbed_indices)
    )

    lambda_coul_arr = np.asarray(lambdas_coul, dtype=float)
    lambda_vdw_arr = np.asarray(lambdas_vdw, dtype=float)
    if (
        lambda_coul_arr.ndim != 1
        or lambda_vdw_arr.ndim != 1
        or not np.all(np.isfinite(lambda_coul_arr))
        or not np.all(np.isfinite(lambda_vdw_arr))
    ):
        raise ValueError("IBS lambda 数组必须是一维且全部为有限数")
    if (
        np.any((lambda_coul_arr < 0.0) | (lambda_coul_arr > 1.0))
        or np.any((lambda_vdw_arr < 0.0) | (lambda_vdw_arr > 1.0))
    ):
        raise ValueError("IBS lambda 必须位于 [0, 1]")
    if np.any(np.abs(lambda_coul_arr) > 1e-8):
        raise RuntimeError(
            "IBS/OpenMM CustomNonbondedForce 路径不能用于非零 λ_coul。"
            "CustomNonbondedForce 不支持 PME，若把配体-环境 Coulomb 放进 IBS CV，"
            "静电会被截断成 cutoff 相互作用，导致物理 Hamiltonian 和 MBAR 均不可信。"
            "请将 Coulomb 去电荷阶段改为保留 PME 的 NonbondedForce/ParameterOffset 或独立传统窗口；"
            "当前 IBS 路径仅允许 λ_coul=0 的 VDW 短程阶段。"
        )

    # ---------- 1. 提取 NonbondedForce ----------
    nb_forces = [f for f in new_sys.getForces() if isinstance(f, openmm.NonbondedForce)]
    if not nb_forces:
        raise ValueError("未找到 NonbondedForce")
    nb = nb_forces[0]

    # ---------- 2. 保存原始参数快照 (供 Group 1 & 2 使用) ----------
    all_params = [nb.getParticleParameters(i) for i in range(num_atoms)]
    ref_excl = []
    for i in range(nb.getNumExceptions()):
        p1, p2, cp, sig, eps = nb.getExceptionParameters(i)
        ref_excl.append((int(p1), int(p2)))
    
    softcore_excl = _collect_softcore_exclusions(nb, new_sys, perturbed_indices, env_indices, ref_excl)
    # 🚨 关键修复：OpenMM 要求同一 System 内所有粒子数相同的 NonbondedForce/
    # CustomNonbondedForce 拥有完全相同的排除表 ("All Forces must have identical
    # exclusions")。IBS 的软核 CV (Group 1，嵌在 CustomCVForce 内部，
    # sync_all_exclusions 扫不到) 之前只带了 softcore_excl（仅 L-E 跨组对），
    # 跟 nb 的完整排除表（含大量环境蛋白/水分子自身的 1-2/1-3/1-4 对）数量对不
    # 上，会在第一次真正构建邻居表时（通常是 Stage2 去VDW 阶段首次 minimize）
    # 报 "All Forces must have identical exclusions"。这里把 nb 的完整排除对
    # 也并进来，统一喂给下面的软核 CV，跟 wca_force/nb 保持逐对一致；
    # interaction group 之外的这些对本来就不会被软核 CV 计算，纯粹是账本对齐。
    full_softcore_excl = sorted(
        {(min(int(p1), int(p2)), max(int(p1), int(p2))) for p1, p2 in ref_excl if int(p1) != int(p2)}
        | set(softcore_excl)
    )

    # 构建原始 nb 副本 (供 Group 2 配体内部力使用)
    original_nb = openmm.NonbondedForce()
    for q, sig, eps in all_params:
        original_nb.addParticle(q, sig, eps)
    for p1, p2, cp, sig, eps in [nb.getExceptionParameters(i) for i in range(nb.getNumExceptions())]:
        original_nb.addException(int(p1), int(p2), cp, sig, eps)
    original_nb.setNonbondedMethod(nb.getNonbondedMethod())
    original_nb.setCutoffDistance(nb.getCutoffDistance())
    if nb.getUseSwitchingFunction():
        original_nb.setUseSwitchingFunction(True)
        original_nb.setSwitchingDistance(nb.getSwitchingDistance())

    # ---------- 2.5 🔑 静态电中性防御 (替代 ParameterOffset，彻底消除 base 的 λ 依赖) ----------
    lig_net_charge_raw = 0.0
    for idx in perturbed_indices:
        q, _, _ = all_params[idx]
        lig_net_charge_raw += q.value_in_unit(unit.elementary_charge)
    if not np.isfinite(lig_net_charge_raw):
        raise RuntimeError("配体净电荷为 NaN/Inf，拒绝构建 IBS Hamiltonian")
    lig_net_charge = int(round(lig_net_charge_raw))
    if (
        abs(lig_net_charge_raw - lig_net_charge)
        > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E
        or lig_net_charge != 0
    ):
        raise RuntimeError(
            f"检测到带净电或部分净电配体 ({lig_net_charge_raw:+.6f} e)。"
            "净电荷必须先接近整数并且为 0；"
            "IBS 的 VDW 阶段不再静态改写反离子电荷；"
            "请先用保留 PME 的去电荷阶段处理净电荷，或显式实现经过验证的离子炼金方案。"
        )

    # ---------- 3. 🔑 永久关闭主 NB 力中的配体相互作用 (确保 base 严格 λ 无关) ----------
    for idx in perturbed_set:
        # 电荷与 VdW 同时归零，彻底切断主 NB 力对配体的贡献
        nb.setParticleParameters(idx, 0.0*unit.elementary_charge, 0.1*unit.nanometer, 0.0*unit.kilojoule_per_mole)
    
    for i in range(nb.getNumExceptions()):
        p1, p2, cp, sig, eps = nb.getExceptionParameters(i)
        if p1 in perturbed_set or p2 in perturbed_set:
            # 跨组或配体内部异常对全部归零（内部作用已由 Group 2 接管）
            nb.setExceptionParameters(i, int(p1), int(p2), 0.0*unit.elementary_charge**2, sig, 0.0*unit.kilojoule_per_mole)

    # ---------- 4. 配体内部力 (Group 2) ----------
    # 去电荷腿（v5，couple-intramol=no）在 λ_coul=0 端点上：粒子电荷为 0，1-4
    # exception 的 chargeProd 冻结在全强度，普通 ≥1-5 对的库仑由一个 (1-λ²) 的
    # CustomBondForce 补偿项承载、在 λ=0 时取满。这里必须**逐项复刻同一个端点约定**，
    # 否则 Stage-1/Stage-2 接缝上会留一个常数但真实的哈密顿量跳变。
    #   - `ll_f`（CustomNonbondedForce）继续用清零电荷 → 只负责 ≥1-5 的 LJ；
    #     它的 cutoff 是 1.0 nm，跨度大于 1 nm 的配体的最长内部对会被截断，所以
    #     **不能**靠把电荷塞回这里来恢复内部库仑（那样两个 stage 的口径又不一致了）。
    #   - `ll_14_f` 直接从原始 NB 的 exception 读 chargeProd → 1-4 全强度，与去电荷
    #     腿的冻结 exception 一致。
    #   - ≥1-5 库仑改由下面的 `create_ligand_internal_coulomb_force`（逐对、无截断、
    #     与去电荷腿共用同一份排除表）承载，前缀取 1。
    group2_params = list(all_params)
    for idx in perturbed_set:
        _q, _sig, _eps = group2_params[idx]
        group2_params[idx] = (0.0 * unit.elementary_charge, _sig, _eps)
    internal_ref_excl = [(p1, p2) for p1, p2 in ref_excl if p1 in perturbed_set and p2 in perturbed_set]
    ll_f, ll_14_f = create_ligand_internal_force(
        original_nb, perturbed_indices, group2_params, internal_ref_excl, num_atoms, system=new_sys
    )
    ll_f.setForceGroup(2)
    if ll_14_f: ll_14_f.setForceGroup(2)
    new_sys.addForce(ll_f)
    if ll_14_f: new_sys.addForce(ll_14_f)
    # 模板是 `clone(system)`，力会原样带过来。charge-transfer 的烘焙交接
    # （`bake_global_parameter_into_fixed_nonbonded_force`，λ_coul=0）会把已经固化
    # 成满强度的同一个力带过 stage 边界 —— 靠名字显式去重，不靠"烘焙后配体电荷为 0、
    # 所以这里新建的那份自然是空"这种隐式巧合。
    if any(
        f.getName() == LIGAND_INTERNAL_COULOMB_FORCE_NAME for f in new_sys.getForces()
    ):
        print(
            f"  [couple-intramol=no] 模板已带 {LIGAND_INTERNAL_COULOMB_FORCE_NAME}"
            "（stage-1 烘焙交接固化的那份），不重复添加。"
        )
        ll_coul_f = None
    else:
        ll_coul_f = create_ligand_internal_coulomb_force(
            original_nb,
            perturbed_indices,
            {
                int(idx): _value_in_elementary_charge(all_params[int(idx)][0])
                for idx in perturbed_set
            },
            system=new_sys,
            reference_exclusions=internal_ref_excl,
            scale_expr="1",
        )
    if ll_coul_f is not None:
        # Group 2 是 IBS_E_BASE_FORCE_GROUPS 的一员（λ 无关的基态项），这一项正属于它。
        ll_coul_f.setForceGroup(2)
        new_sys.addForce(ll_coul_f)

    # ---------- 5. 物理 Boresch 限制力 (Group 3) ----------
    if _has_valid_boresch_restraint(restraint_params):
        rest_f_phys = LambdaDependentBoreschForce(
            rec_idx=restraint_params["receptor_indices"], lig_idx=restraint_params["ligand_indices"],
            eq=restraint_params["equilibrium_values"], fc=restraint_params["force_constants"],
            lam_name="lambda_boresch_scale", fixed_lam=None, sign=1.0, use_pbc=True
        )
        rest_f_phys.setForceGroup(3)
        new_sys.addForce(rest_f_phys)

    # ---------- 6. λ-WCA 防护壳（Group 4） & COM 限制（Group 5） ----------
    # 🔑 [WCA_SHIELD_RETIRED，2026-09-02] 整段**死代码**（保留、永不执行）。
    # 退役依据是实测因果归因：这个壳解释了 4W53 甲苯溶剂腿 stage2 误差的 **98%**
    # （+44.21 / +45.30 kJ/mol）——壳幅度 4λ(1−λ) 在 λ→0 只线性衰减，而物理 LJ 是
    # λ²，比值 4(1−λ)/λ 发散 ⟹ 目标态无壳而系综带壳，水贴近配体的构型从未被采到。
    # 见该常量处的长注释与
    # docs/archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md §2.10。
    # 翻 `WCA_SHIELD_RETIRED = False` 即整段复活。
    #
    # 已退役 ⟹ **不打任何日志**。壳根本不进生产采样系统，每次建系再报一遍"它没参与"
    # 是纯噪声；退役理由属于代码史，留在上面这段注释里就够了。Group 4 现为空组，
    # e_base = groups{0,2,3,5} 的记账口径（WCA_ACCOUNTING_VERSION）不受影响。
    if not WCA_SHIELD_RETIRED:
        wca_expr = (
            "4.0*lambda_shield*(1.0-lambda_shield)*step(rc-r)*eps_wca*"
            "(((rc/max(r, 1e-6))^6)^2 - 2*((rc/max(r, 1e-6))^6) + 1)"
        )
        wca_params = _estimate_wca_shield_parameters(all_params, list(perturbed_set), env_indices)
        wca_force = openmm.CustomNonbondedForce(wca_expr)
        wca_force.addGlobalParameter("lambda_shield", 0.0)
        wca_force.addGlobalParameter("rc", wca_params["rc_nm"])
        wca_force.addGlobalParameter("eps_wca", wca_params["eps_wca_kJ_mol"])
        for _ in range(num_atoms): wca_force.addParticle([])
        wca_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        wca_force.setCutoffDistance(wca_params["rc_nm"] * openmm.unit.nanometer)
        wca_force.addInteractionGroup(list(perturbed_set), env_indices)
        for p1, p2 in softcore_excl: wca_force.addExclusion(int(p1), int(p2))
        wca_force.setForceGroup(4)
        new_sys.addForce(wca_force)
        _msg = (
            "  [λ-WCA 防护壳] "
            f"rc={wca_params['rc_nm']:.3f} nm, eps={wca_params['eps_wca_kJ_mol']:.3f} kJ/mol "
            f"({wca_params['source']}, sigma_ref={wca_params['sigma_reference_nm']:.3f} nm)"
        )
        print_once(f"wca_shield_params|{_msg}", _msg)

    # ---------- Group 5（配体 COM 限制）已移除 ----------
    # [LIGAND_COM_RESTRAINT_PROTOCOL_VERSION=2] 详见文件上方该常量的长注释。
    # 旧实现是非周期绝对锚点的 CustomCentroidBondForce；在 CUDA 上 centroid 被折叠进
    # 主盒而锚点没有，绝对距离把两个周期像当成真实距离，产生永久激活、跨边界跳变的
    # 外力，实测让配体以远超自由扩散的速度定向漂移（110 ps 内 30.9 nm），使轨迹不再
    # 满足 MBAR 的平衡采样前提。CPU 上不复现，静态 setPositions() 测试也检不出。
    #
    # 该力本来就只在**没有 Boresch 锚定**时才添加（复合物腿一直跳过它），所以它只存在
    # 于溶剂腿，而均匀溶剂中它不提供任何必要约束。B/C 受控对照证明删除它与改成周期
    # 最小像在结构统计与局部自由能上实用等价，且删除的单向重加权效率更好。
    #
    # 结论：不再添加任何 Group 5 力。**Group 5 现为空组**——`e_base = groups{0,2,3,5}`
    # 的记账口径（WCA_ACCOUNTING_VERSION=2）不受影响，该组只是贡献 0。
    if _has_valid_boresch_restraint(restraint_params):
        print_once(
            "group5_com_removed_complex",
            "  复合物腿由 Boresch 锚定定位配体；Group 5 COM 限制力已于"
            " LIGAND_COM_RESTRAINT_PROTOCOL_VERSION=2 全局移除。",
        )
    else:
        print_once(
            "group5_com_removed_solvent",
            "  溶剂腿不添加 Group 5 COM 限制力"
            " (LIGAND_COM_RESTRAINT_PROTOCOL_VERSION=2)：旧的非周期绝对锚点实现在 CUDA 上"
            "产生定向拖拽，均匀溶剂中该限制亦无必要。",
        )

    # ---------- 7. IBS 偏置力与纯 VDW 软核 CV (Group 1) ----------
    if residual_basis_force is None:
        ibs_wrapper = IBSBiasForce(len(lambdas_coul), temperature, prefix=prefix)
    else:
        ibs_wrapper = IBSBiasForce.with_residual_basis(
            len(lambdas_coul),
            temperature,
            prefix=prefix,
            residual_basis_force=residual_basis_force,
            residual_state_coefficients=residual_state_coefficients,
            residual_energy_offset_kj_mol=residual_energy_offset_kj_mol,
        )
    original_params_fresh = [original_nb.getParticleParameters(i) for i in range(num_atoms)]

    cv_template = _create_softcore_force(
        nb,
        perturbed_indices,
        env_indices,
        lam_coul=0.0,
        lam_vdw=0.0,
        alchemical_params=alchemical_params,
        potential_type=potential_type,
        reference_exclusions=full_softcore_excl,
        particle_params_override=original_params_fresh,
        num_particles=num_atoms,
    )
    legacy_softcore_signature = "+ 1e-4)^2"
    if legacy_softcore_signature in cv_template.getEnergyFunction():
        print("  [WARN] [IBS CV] 检测到旧版 VDW softcore 表达式签名；请检查 CV 构造路径。")
    template_cutoff = cv_template.getCutoffDistance()
    template_switch = cv_template.getSwitchingDistance()
    template_use_switching = cv_template.getUseSwitchingFunction()
    template_method = cv_template.getNonbondedMethod()
    template_excl = {
        tuple(sorted(map(int, cv_template.getExclusionParticles(i))))
        for i in range(cv_template.getNumExclusions())
    }

    # ---------- 7.5 🔑 LJ 长程色散尾项：手算解析修正（不走 OpenMM 内建 LRC） ----------
    # 已用 tools/diagnostics/probe_lrc_interaction_group_compat.py 实测确认：CustomNonbondedForce.
    # setUseLongRangeCorrection(True) 配合 addInteractionGroup 在电荷=0 的纯 LJ
    # 情形下工作正常（截断误差消除 ~88%，lambda_vdw=1.0 与 0.5 结果一致）；但本
    # 项目的软核表达式把 LJ 和 Coulomb 拼在同一个 CustomNonbondedForce 里，真实
    # 非零电荷会让 OpenMM 对 1/r 尾项的解析积分发散，直接让 CUDA 后端崩溃
    # ("terminate called recursively" / core dump，已实测复现)。因此改为在
    # Python 侧手算解析修正（不触碰 CustomNonbondedForce/GPU kernel；每帧只是
    # 一次除法）。
    # 🔑 [TRADITIONAL_LJ_LRC_PROTOCOL_VERSION=2] 之前这里只补 cutoff（1.2nm）
    # 之外的标准 r^-6 尾项，完全忽略了这个软核力实际启用的 1.0-1.2nm switching
    # 区间——OpenMM 在这段区间把能量从满强度削到 0，真正缺失的是"switching 削弱
    # 的部分 + cutoff 之外的标准尾项"（见 CustomNonbondedForce LRC 文档），且
    # λ<1 时软核分母 D(r)=alpha_lj*(1-λ)^m_lj + r^6 也需要真实积分，而不是当成
    # 纯 r^6 处理。现在对每个 λ_vdw 数值积分真实的 switching-aware +
    # softcore-aware 径向积分（_lj_tail_lrc_coefficients_kj_mol），同时补上
    # 排斥项 r^-12 的尾贡献（之前只有吸引项 r^-6）。每帧修正 = coeff[k] / V(t)。
    ibs_wrapper.lj_tail_lrc_coeff_kj_mol = None
    if not ibs_lj_tail_lrc_is_applicable(
        potential_type, dispersion_protocol, environment_type
    ):
        print(
            "  [WARN] [LJ LRC] "
            f"{ibs_lj_tail_lrc_inapplicable_reason(potential_type, dispersion_protocol, environment_type)}"
            "，本次不附加修正。"
        )
    else:
        rc_nm = template_cutoff.value_in_unit(unit.nanometer)
        rs_nm = template_switch.value_in_unit(unit.nanometer)
        tail_sigma, tail_s6_per_sigma, tail_s12_per_sigma = (
            _lj_tail_correction_sigma_resolved_moments(
                all_params, perturbed_indices, env_indices
            )
        )
        n_lj_exp = float(getattr(alchemical_params, "n_lj", 2))
        alpha_lj = float(getattr(alchemical_params, "alpha_lj", 0.5))
        m_lj = float(getattr(alchemical_params, "m_lj", 2))
        ibs_wrapper.lj_tail_lrc_coeff_kj_mol = _lj_tail_lrc_coefficients_kj_mol(
            lambdas_vdw, tail_sigma, tail_s6_per_sigma, tail_s12_per_sigma,
            alpha_lj, m_lj, n_lj_exp, rs_nm, rc_nm,
        )
        # 这里原来打一行 S6/S12/sigma 分组/switch/cutoff/alpha_lj/m_lj/n_lj 的汇总。
        # 去掉：这些数是已落盘输入（拓扑 + softcore 参数 + cutoff）的确定性函数，可复算；
        # 而"尾项到底有没有施加、按哪个协议版本"这两个真正需要审计的事实，记在
        # final_results.json 的 `lj_long_range_dispersion_correction`（applicable/applied/
        # dispersion_protocol/environment_type）和 `traditional_lj_lrc_protocol_version` 里。

    for k, (_lc, lv) in enumerate(zip(lambdas_coul, lambdas_vdw)):
        # IBS/OpenMM 只允许处理短程 VDW CV；Coulomb 已禁止进入 CustomNonbondedForce，
        # 避免把 PME 静电截断为 cutoff 相互作用。
        int_f_cv = _create_softcore_force(
            nb,
            perturbed_indices,
            env_indices,
            lam_coul=0.0,
            lam_vdw=float(lv),
            alchemical_params=alchemical_params,
            potential_type=potential_type,
            reference_exclusions=full_softcore_excl,
            particle_params_override=original_params_fresh,
            num_particles=num_atoms,
        )
        int_f_cv.setCutoffDistance(template_cutoff)
        int_f_cv.setSwitchingDistance(template_switch)
        int_f_cv.setNonbondedMethod(template_method)
        # 🔑 [0831issue P2] 这条告警以前在**每个 λ 态**都打一遍（K+1 遍），K=23 时
        # 单次建系就刷 23 行同样的文字，把真正的告警淹掉。它描述的是这个 CV 力族的
        # 一个**共同属性**，与 k 无关，所以只在第一个态打一次。
        if k == 0:
            # 那次 P2 修的是"单次建系刷 K+1 行"；但 build_ibs_dual_system 本身在一次
            # run 里被调几十次，所以 k==0 的守卫之外还需要跨调用去重（λ 态数变了会重打）。
            _msg = (
                "  [WARN] [IBS CV] VDW custom softcore CV 不含 LJ 长程修正（全部 "
                f"{len(lambdas_coul)} 个 λ 态同此）；请勿直接等同于已含 dispersion "
                "correction 的 PME/LJPME 循环；APBS 修正应作为最终外部项记录。"
            )
            print_once(f"ibs_cv_no_lrc|{_msg}", _msg)
        int_f_cv.setUseLongRangeCorrection(False)
        # [MEM-00h，2026-08-06] 之前这里无条件写 True，会把 _create_softcore_force
        # 按 potential_type 已经算好的 use_switching 悄悄覆盖回"总是开 switching"——
        # 对当前生产的 softcore（非 dexp）路径来说，那会让每个 λ_vdw 态实际使用的
        # CV 力switching 状态跟 cv_template（同一份构造逻辑理应产出同样的状态）不
        # 一致。改成跟随 template 的真实状态，而不是硬编码常量。
        int_f_cv.setUseSwitchingFunction(template_use_switching)
        current_excl = {
            tuple(sorted(map(int, int_f_cv.getExclusionParticles(i))))
            for i in range(int_f_cv.getNumExclusions())
        }
        if current_excl != template_excl:
            raise RuntimeError(f"CV {k} 排除表不一致，将破坏 VDW-IBS 邻居表复用条件。")
        ibs_wrapper.addCollectiveVariable(f"cv_{k}_int", int_f_cv)

        # Boresch CV 保持零力（物理限制已在 Group 3）
        # [EXP-031 S1] omit 形态下整槽不建：这个 CustomExternalForce("0") 从未被加过
        # 任何粒子，却要每步付一次 inner-context 力求值（实测 21-25 µs/CV，K=16 时
        # 合计 +0.414 ms/step = 整步 15%）。见 IBS_BIAS_OMIT_ZERO_REST_CVS。
        if not ibs_wrapper.omit_zero_rest_cvs:
            rest_f_cv = openmm.CustomExternalForce("0")
            ibs_wrapper.addCollectiveVariable(f"cv_{k}_rest", rest_f_cv)

    # EXP-025 G4 Layer-2: fail closed HERE, before the Force ever enters the
    # System/before any Context can be created, if the per-state loop above
    # ever left a cv_k_int/cv_k_rest (or, when residual_basis_force is set,
    # the shared exp025_residual_basis) unregistered or duplicated. This is
    # not a hypothetical -- see g4_layer1_oracle_report.json's
    # real_bug_found_and_fixed_during_this_gate for the exact silent-wrong-
    # energy failure mode this closes.
    ibs_wrapper.validate_wiring()
    new_sys.addForce(ibs_wrapper.get_force())

    # ---------- 8. 统一排除表同步 ----------
    sync_all_exclusions(new_sys)
    # Export the exact pre-bias Hamiltonian for fixed-state overlap probes.
    # Do this only after exclusion synchronization so the added state CV can
    # share the same nonbonded topology as the main NonbondedForce.
    ibs_wrapper._common_system_xml = _serialize_ibs_common_system(new_sys)
    # 供 bias 校准探针使用（保留 Group 4 WCA，见该函数 docstring 里对两种用途的
    # 区分说明）——不能用上面那份缺 WCA 的 U_common 去校准 f_k。
    #
    # 🔑 [2026-09-09] 防护壳退役后，两个序列化器的产物**逐字节相同**：
    # `_serialize_ibs_common_system` 剥 Group 1 与 4，
    # `_serialize_ibs_common_plus_wca_system` 只剥 Group 1，而 Group 4 已经不存在。
    # 原来这里照样做**第二次**完整 serialize→deserialize→serialize 往返，再把同样
    # 的字节当成第二份常驻字符串挂在 wrapper 上。被序列化的是**已装配的窗口
    # System**（Group-1 的 CustomCVForce 里装着 K 个 softcore CustomNonbondedForce，
    # 每个都带完整的 per-particle 参数和整张排除表），大体系上是上百 MB 的瞬时
    # 峰值加一份等长常驻。
    #
    # 等价性在这里**当场验证**、不是假设：直接扫 new_sys 的 force group，
    # 确认没有 Group 4 才复用；有的话说明壳被复活了而这段没跟上，fail closed。
    # 原来那条断言在 `_serialize_ibs_common_plus_wca_system` 里，语义一并搬过来。
    if WCA_SHIELD_RETIRED:
        _stray_group4 = [
            int(i)
            for i in range(new_sys.getNumForces())
            if int(new_sys.getForce(i).getForceGroup()) == 4
        ]
        if _stray_group4:
            raise RuntimeError(
                "WCA_SHIELD_RETIRED=True 但装配好的窗口 System 里仍存在 Group 4 "
                f"(force index {_stray_group4}) —— 生产采样已不含防护壳，"
                "探针含壳会采到生产从不采的系综，拒绝继续。"
                "若要复活防护壳，翻 WCA_SHIELD_RETIRED 而不是绕过本断言。"
            )
        ibs_wrapper._common_plus_wca_system_xml = ibs_wrapper._common_system_xml
    else:
        ibs_wrapper._common_plus_wca_system_xml = _serialize_ibs_common_plus_wca_system(new_sys)
    new_sys.thisown = 1
    _ = new_sys.getNumParticles()
    print(f"[IBS System] 构建完成，共 {new_sys.getNumForces()} 个力对象 (Group 0 已彻底剥离 λ 依赖)")
    return new_sys, ibs_wrapper

