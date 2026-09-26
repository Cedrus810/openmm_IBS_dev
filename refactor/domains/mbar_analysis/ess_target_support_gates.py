# ============================================================================
# 🔑 [ESS_GATE_PROTOCOL_VERSION=2] IBS 重加权质量诊断：区分"混合分布覆盖度"
# 与"共模采样偏置代价"
# ----------------------------------------------------------------------------
# 单参考增广 MBAR（n_k=[N,0,...,0]）的 compute_effective_sample_number() 算的是
#
#     W_nk ∝ exp[ (V_bias(x_n) - U'_k(x_n)) / kT ]
#
# 而生产实际施加的 V_bias 读的是 OpenMM force groups {1, 4}（见
# IBSSampler.collect_energies）——Group 1 是 IBS 混合偏置，Group 4 是 λ-WCA 防护壳；
# 目标态能量 U'_k 是 softcore + 解析 LRC。防护壳在窗口内对所有 k 用同一个
# lambda_shield（见 _serialize_ibs_common_plus_wca_system 的 docstring），LRC 是
# coeff[k]/V(t)，所以两者合起来在 log 权重里是一个**逐帧共模因子** r_n：
#
#     log W_nk = r_n / kT + log p_k(x_n) + const_k,
#     p_k(x) = exp[-(U'_k - f_k)/kT] / Σ_j exp[-(U'_j - f_j)/kT]
#
# r_n 在 occupancy ⟨p_k⟩、相邻 ΔF、warmup loose gate、以及物理态↔物理态自由能差
# 里都（大部分）抵消——而后者正是本阶段真正报出去的量。但它在 W_nk 里不抵消，
# 于是把 ESS 按 exp(σ_r²) 整体压掉：真实测到 σ_r 从 λ_vdw≈1 端的 0.95 kT 涨到
# λ_vdw→0 端的 2.40 kT，对应 ESS/N 上限 0.40 → 0.003，而同一批数据的 ⟨p_k⟩ 是
# 平坦的 0.249-0.251。也就是说旧的 min_ess_ratio 门衡量的是**防护壳的重加权代价**，
# 不是采样质量，更不是输出精度：实测 window 5 的 raw ESS 最差(0.0029)但端点
# 不确定度最好(0.199 kJ/mol)，window 3 的 σ_r 几乎一样(2.35 kT)而端点不确定度
# 是 2.17 kJ/mol。用它当收敛门会让 rescue 循环去追一个数学上不可满足、且与
# 被估计量无关的目标。
#
# 现在拆成三项互相独立的证据：
#   (1) min_ess_ratio（**受门**）：去掉共模因子后的混合分布覆盖度 ESS(p_k)/N_decorr。
#       这才是"采样到的混合分布对物理态 k 覆盖够不够"。
#   (2) min_decorrelated_samples（**受门**，已有字段）：独立样本数本身。
#   (3) max_endpoint_uncertainty_kJ_mol（**受门**，已有字段）：MBAR 自己带全协方差
#       算出的端点 ΔF 精度，即输出精度的直接度量。
# raw_min_ess_ratio / raw_absolute_ess / common_mode_log_sigma_kT 继续算、继续落盘，
# 但**只报告不设门**——它们是"防护壳收了多少重加权税"的诚实诊断。
#
# 注意 ESS(p_k) 必须用该窗口真正冻结进生产的 f_k。试过用逐帧
# softmax 归一化（r_n=mean_k 或 logsumexp_k，无需 f_k、且 gauge-free）来省掉
# f_k 的传参：对相邻 ΔU≲2.5 kT 的窄窗口（3/4/5）与真 f_k 版本一致，但对
# window 0（相邻 ΔU≈10 kT）给出 0.014 vs 真值 0.500，差 36 倍——因为 f_k 加权的
# logsumexp 与无权算术平均在谱宽大时是完全不同的逐帧函数。所以这条捷径不成立，
# f_k 必须真的传进来（从 checkpoint 的 ibs_state_*.json 读，见
# get_stage_data_for_analysis）。
# ============================================================================
# v3: occupancy 与 warmup 协议统一为诊断项，不再在全部生产完成后反向否决 stage。
# v4（2026-09-01，0831issue）：混合覆盖度 ESS / 占据 / 共模 σ 改用 **sampling-state**
# 能量 S_k = U_k^softcore + s_residual·A_k·(B_φ − offset) 计算，而不是物理目标
# u_kn（softcore+LRC，不含残差）。理由：受门的 f_k 是在 S_k 上学出来的
# （IBS_BIAS_PROTOCOL_VERSION=31），混合成员概率 p_k ∝ exp(−(S_k−f_k)/kT) 只有同口径
# 才成立。residual 臂上两者相差逐态 rank-1 项 A_k·B_φ(x)，在逐帧 softmax 里不抵消，
# 旧写法算出的既不是采样混合分布也不是物理混合分布，还把这个逐态项漏进了
# common_mode_log_sigma_kT（该量按定义只应含 WCA 防护壳 + softcore-vs-LRC 失配）。
# 只报告、不进任何 resume 指纹（abfe_pipeline 只导入 TARGET_SUPPORT_GATE_PROTOCOL_
# VERSION），所以升版不作废任何缓存。baseline 臂 residual_enabled=False 时
# S_k ≡ U_k^softcore，此改动逐位不变；落盘新增 ess_gate_mixture_gauge 记录实际口径。
# 🔑 v5（2026-09-01）：受门的 raw 权重退化量（`raw_min_absolute_ess` /
# `top1pct_raw_weight`）改在**去相关之前**的帧集上计算。v4 及更早算在
# `_decorrelate_by_worst_target_state` 抽稀后的子集上，而那个抽稀用的序列
# **就是权重的指数** —— 拿权重挑样本、再用挑出的样本量同一批权重的退化程度，
# 循环论证。实测 4W53 甲苯溶剂腿 window 0：ESS/N 0.0072 → 0.2000（28 倍），
# min 绝对 ESS 3.58 → 47.01，门槛 20，**判定被直接翻转**。
# 去相关本身保留（MBAR 的 n_k 需要它），只是不再用来算这道门。
ESS_GATE_PROTOCOL_VERSION = 5

# ---------------------------------------------------------------------------
# 探针 Context 的 force-group 预算。IBSSampler._build_probe_context 给每个 λ 态
# 分配一个独立 force group，以便一次 setPositions 后逐组读出每个态的相互作用能。
# 起始号必须避开生产 Hamiltonian 已占用的 0/1/2/3/4/5。
# ---------------------------------------------------------------------------
OPENMM_MAX_FORCE_GROUP = 31
PROBE_FORCE_GROUP_BASE = 16
PROBE_MAX_LAMBDA_STATES = OPENMM_MAX_FORCE_GROUP - PROBE_FORCE_GROUP_BASE + 1  # = 16

# ============================================================================
# [TARGET_SUPPORT_GATE_PROTOCOL_VERSION=1] 面向"无防护壳物理目标"的硬门
#
# 上面那套受门的 min_ess_ratio 是**扣掉共模因子之后**的混合覆盖度。它回答的是
# "已经采到的这批帧，在同一个窗口的各个 λ 态之间分得开不开"。它**不**回答
# "这批被 Group-4 λ-WCA 防护壳偏置过、且整窗只有一条轨迹的采样，能不能重加权
# 到没有防护壳的真实物理系综"——共模因子恰恰就是在那一步被除掉的。
#
# 4W53 实测（docs/archive/STAGE2_ROOT_CAUSE_2026-08-28.md）是这个盲点的直接证据：
#     溶剂腿  mixture min_overlap = 0.4684  → 判定通过
#             raw     min_overlap = 0.0196  → 对真实物理目标几乎不能重加权
#     converged=True，stage2 报 +35.61 kJ/mol，独立端点参考是 -6.29 kJ/mol。
# ⚠️ [2026-09-02] 这组数字是**壳退役前**的，而这个盲点本身在 09-02 得到正面
# 验证：壳一去掉（`WCA_SHIELD_RETIRED = True`），溶剂腿 stage2 的 raw ESS 从
# 2.93 跳到 173.33、top1% 权重从 0.828 掉到 0.047 —— "raw 支撑度崩掉"确实是被
# 壳偏置直接引起的。参考真值的权威口径在 docs/reference_data/README.md
# （与生产逐窗口可比的是 m2n2 的 -6.581±0.256，no-LRC）。
# 逐窗口的 raw 绝对 ESS 是 [85.9, 8.5, 9.6]（N_decorr=[351, 432, 332]），
# 复合物腿是 [10.5, 8.1, 4.2]——两条腿都只剩个位数到十位数的等效独立样本在
# 支撑物理目标态，而 mixture 门报的是 0.46/0.89 这种"健康"数字。
#
# 所以 raw 量不能继续"只报告不设门"。但也不能把 raw_min_ess_ratio 直接拿去和
# 0.05 比：raw ratio 的上限被共模因子 exp(sigma_r^2) 整体压住（实测 sigma_r
# 0.48~1.24 kT），它是一个跟采样长度无关的比例量，用固定比例阈值卡它既可能
# 数学上不可满足、又会被"多采几帧"稀释。这里改用两项与"物理目标态到底有多少
# 真实支撑"直接对应、且彼此正交的量：
#
#   (1) raw_min_absolute_ess（**受门**）：重加权到无防护壳物理目标之后还剩多少
#       个等效独立样本。绝对数，不随总帧数自动变好；它低就是"目标系综没被采
#       到"。延长采样是能让它变好的正确方向，与 ratio 阈值"样本越少门越严"的
#       病理相反。阈值沿用 final_min_decorrelated_samples 的 20 这个口径：
#       支撑一个目标态的等效独立样本数不该低于我们对采样态本身的要求。
#   (2) max_top1pct_raw_weight（**受门**）：最重的 1% 帧占了全部 raw 权重的多少。
#       ESS 是二阶矩，权重塌缩到个别帧时它掉得慢；这一项直接看权重集中度，是
#       "少数几帧撑起整个估计"这一失效模式的第一手证据。理想值 ~0.01；实测坏
#       结果里是 0.31~0.55。0.35 = 比理想集中 35 倍，再高就没有可信的重加权。
#
# 任一项不达标、或算不出来（证据不全）→ converged=False，
# failure_reason = "insufficient_target_support"。
#
# ⚠️ 这道门只**拦得住**错值，治不了病。docs/archive/STAGE2_ROOT_CAUSE_2026-08-28.md §3.2 的
# window 2 是决定性反例：它相邻 <ΔU> 只有 0.4~0.6 kT，任何基于能量的重叠判据
# 都会说"完美"，却错得最多——因为失效模式是结构性的（"该采的构型一次都没采到"），
# 不是"采到的构型之间散布不够"。这个教训与幅度无关，仍然成立。
#
# ⚠️ [2026-09-02 更正两点]（见 docs/archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md）
#   1. 原文把该窗口的错量写成 +19.49 kJ/mol 并当作单系综重加权的定量依据。
#      那是**壳还在时**测的：该误差的 98% 已定罪为 λ-WCA 防护壳。
#   2. 原文列的第三条修法"把 lambda_WCA 变成显式热力学维度并让 lambda_WCA=0
#      真的有样本"**已作废**——实际采取的处置是把整个防护壳**退役**
#      （`WCA_SHIELD_RETIRED = True`），不是把它升成一个 λ 维度。
#      仍然有效的修法只剩：端点态独立采样 / 窗口内真实 replica exchange。
# 这道门的作用是让那类结果 fail-closed，而不是静默流入 ΔG_bind。
# 反过来说，它的因果性在 09-02 得到了正面验证：壳一去掉，溶剂腿 stage2 的
# raw ESS 从 2.93 跳到 173.33、top1% 权重从 0.828 掉到 0.047。
# ============================================================================
TARGET_SUPPORT_GATE_PROTOCOL_VERSION = 1
TARGET_SUPPORT_MIN_ABSOLUTE_ESS = 20.0
TARGET_SUPPORT_MAX_TOP1PCT_WEIGHT = 0.35


def _top1pct_weight_by_empirical_cdf(w: np.ndarray) -> float:
    """最重 1% 帧的权重占比，按**经验 CDF 线性插值**取（唯一实现）。

    🔑 [2026-09-14 / 审计 #48] 提成模块级 helper 是因为同一个量原来有**两套**实现：
    生产支撑度门这一处已按 CDF 插值改好（理由见下面调用点的长注释），而自检的
    `top1_veto` 与 join 诊断还在用 `max(1, round(0.01N))` 数整帧 —— 那个旧式在
    N<100 时退化成「**最大单帧**占比」，随 N 减小系统性变大，却共用按 N≈330–430
    校准的阈值 0.35（N=40 时被放大约 2.5 倍），而 `top1_veto` 会直接改写 verdict。
    一个量只许有一个实现，否则改一处漏一处。

    判据（可验证）：权重全相等时任意 N 都恰好返回 0.01；n 为 100 的整数倍时插值项
    为 0，与旧实现逐位相同。``w`` 是**未归一化**的非负权重；总和非正返回 nan。
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    total = float(np.sum(w))
    if not (total > 0.0):
        return float("nan")
    n_top_exact = float(w.size) * 0.01
    n_full = int(np.floor(n_top_exact))
    frac = n_top_exact - n_full
    desc = np.sort(w)[::-1]
    mass = float(np.sum(desc[:n_full]))
    if frac > 0.0 and n_full < desc.size:
        mass += float(frac) * float(desc[n_full])
    return mass / total


def _ibs_reweighting_quality_diagnostics(
    u_kj_raw: np.ndarray,
    bias_kj: np.ndarray,
    f_k: Optional[np.ndarray],
    kt: float,
    sampling_kj: Optional[np.ndarray] = None,
    sampling_gauge_required: bool = False,
) -> Dict[str, Any]:
    """Split IBS single-reference reweighting quality into the mixture-coverage
    part (gated) and the common-mode sampling-bias part (reported only).

    ``u_kj_raw`` is (K, N) per-state interaction energy in kJ/mol, ``bias_kj``
    the (N,) recorded sampling bias (force groups {1,4}) in kJ/mol, ``f_k`` the
    (K,) frozen production bias weights in kJ/mol.

    Returns ``mixture_ess`` (absolute, per state), ``mixture_ess_ratio``
    (per state, relative to N), ``raw_ess`` / ``raw_ess_ratio`` (the old
    single-reference quantity, for reference), ``top1pct_raw_weight`` (fraction
    of total raw weight carried by the heaviest 1% of frames -- the most direct
    statement of weight degeneracy) and ``common_mode_log_sigma_kT``.

    ``f_k`` of ``None`` (or a length/finiteness mismatch) yields ``None`` for
    every mixture-derived field and an explicit ``error`` -- callers must treat
    that as "cannot evaluate the gate", never as a pass.

    🔑 [0831issue / ESS_GATE_PROTOCOL_VERSION=4] ``sampling_kj`` is the (K, N)
    **sampling-state** energy, i.e. the same quantity the deployed Group-1 bias
    is built from: ``S_k = U_k^softcore + s_residual*A_k*(B_phi - offset)``.
    ``f_k`` is learned on exactly that quantity (see
    ``IBS_BIAS_PROTOCOL_VERSION=31``), so the mixture membership
    ``p_k ∝ exp(-(S_k - f_k)/kT)`` -- and therefore every mixture/occupancy/
    common-mode field below -- must be built from ``S_k``, not from the
    physical target ``u_kj_raw``.  With the residual arm enabled the two differ
    by the per-state rank-1 term ``A_k*B_phi(x)``, which does **not** cancel in
    the per-frame softmax; mixing the two gauges yields neither the sampled nor
    the physical mixture, and it also leaks that state-dependent term into
    ``common_mode_log_sigma_kT`` (which is documented to carry only the WCA
    guard shell plus the softcore-vs-LRC mismatch).  For the baseline arm
    ``residual_enabled=False`` makes ``S_k == U_k^softcore``, so passing
    ``sampling_kj=None`` there keeps the old numbers bit-for-bit.

    ``sampling_gauge_required=True`` says "this window is known to sample a
    residual-augmented Hamiltonian": a missing or malformed ``sampling_kj``
    then yields ``None`` mixture fields plus an ``error`` (fail closed) instead
    of silently falling back to the physical gauge.

    ``u_kj_raw`` stays the physical target for the ``raw_*`` /
    ``top1pct_raw_weight`` block: those quantify reweighting onto the real
    physical ensemble and are correctly paired with the recorded ``bias_kj``.
    """
    u_kj_raw = np.asarray(u_kj_raw, dtype=np.float64)
    bias_kj = np.asarray(bias_kj, dtype=np.float64).ravel()
    n_states, n_frames = u_kj_raw.shape
    out: Dict[str, Any] = {
        "n_frames": int(n_frames),
        "mixture_gauge": None,
        "raw_ess": None,
        "raw_ess_ratio": None,
        "top1pct_raw_weight": None,
        "mixture_ess": None,
        "mixture_ess_ratio": None,
        "mixture_occupancy": None,
        "mixture_occupancy_normalized": None,
        "common_mode_log_sigma_kT": None,
        "common_mode_log_mean_kJ_mol": None,
        "error": None,
    }
    if n_frames == 0 or bias_kj.size != n_frames:
        out["error"] = "frame_count_mismatch"
        return out

    # ---- raw single-reference weights (reported only) ----
    log_w_raw = (bias_kj[None, :] - u_kj_raw) / float(kt)
    raw_ess = [_ess_from_log_weights(log_w_raw[k]) for k in range(n_states)]
    out["raw_ess"] = [float(x) for x in raw_ess]
    out["raw_ess_ratio"] = [float(x) / float(n_frames) for x in raw_ess]
    # 🔑 [2026-09-01] 「最重 1% 帧的权重占比」必须按**经验 CDF 线性插值**取，
    # 不能用 `max(1, n_frames // 100)` 数整帧。
    #
    # 旧写法在 N < 100 时 `n_frames // 100 == 0`、被 `max(1, ...)` 抬到 1，于是该量
    # 从"最重 1% 帧占多少权重"退化成"**最大单帧**占多少权重" —— 一个远更噪、且随 N
    # 减小系统性变大的量。而 TARGET_SUPPORT_MAX_TOP1PCT_WEIGHT = 0.35 是按 N≈330–430
    # 校准的，用同一个阈值卡小样本窗口会系统性偏严。
    # 这与 `TraditionalMBARAnalyzer` 那条重叠门是**同一个失效类**：小样本上会崩的
    # 估计量被当门用（那边实测 600 帧 0.0834 → 去相关 398 帧 0.0205，把生产卡死）。
    #
    # 判据（可验证）：权重全相等时，无论 N 多少，该量都应**恰好等于 0.01**。
    #   旧实现 N=50 → w/(50w) = 0.02（放大 2 倍）；N=600 → 6w/600w = 0.01。
    #   新实现 任意 N → 0.01。
    # 且当 n_frames 是 100 的整数倍时，插值项为 0，与旧实现**逐位相同**（向后兼容）。
    # [2026-09-14 / 审计 #48] 实现已提成模块级 `_top1pct_weight_by_empirical_cdf`，
    # 自检的 top1_veto 与 join 诊断共用同一份 —— 数值逐位不变，只是不再有三份拷贝。
    _n_top_exact = float(n_frames) * 0.01
    top1 = [
        _top1pct_weight_by_empirical_cdf(
            np.exp(log_w_raw[k] - np.max(log_w_raw[k]))
        )
        for k in range(n_states)
    ]
    out["top1pct_raw_weight"] = top1
    out["top1pct_raw_weight_n_top_exact"] = float(_n_top_exact)
    # [0831issue P2 的原标注保留为审计线索] 这两个字段记录的是**旧实现**会取几个整帧、
    # 以及它是否已退化成 max-single-frame。上面改成 CDF 插值后退化本身已经消除，
    # 但字段继续落盘：历史产物里这两个值是判断"那批数字是否受该缺陷影响"的唯一依据。
    out["top1pct_raw_weight_n_top_frames"] = int(max(1, n_frames // 100))
    out["top1pct_raw_weight_degenerate_max_single_frame"] = bool(n_frames < 100)

    # ========================================================================
    # ---- 采样偏置涨落 / 物理信号涨落 之比（**仅诊断，不设门**）----
    # ------------------------------------------------------------------------
    # 🔑 [2026-09-02] 这一项是 4W53 甲苯溶剂腿实测出来的、目前**唯一**能预测
    # "这个窗口的 ΔF 会错多少"的量。它不是又一个重叠度代理，测的是另一件事：
    # **必须被重加权掉的偏置，比它不该淹没的物理信号大多少倍。**
    #
    # `bias_kj` 是力组 {1,4}（IBS 混合偏置 + λ-WCA 防护壳，见
    # WCA_ACCOUNTING_VERSION=2）。重加权要把它整个除掉才能到无壳物理目标。
    # 而各态之间真正要分辨的差异只有 `u_kj_raw` 的态间差。当
    #     sd(bias) >> sd(U_k)
    # 时，权重由偏置涨落支配、物理 λ 依赖被埋在下面，重加权拿不回该拿的构型。
    #
    # 实测（6 个窗口，靶子是 docs/reference_data/ 里的独立参考真值）：
    #   win        0     1     2     3      4      5
    #   ratio    1.8   2.0   3.3   5.1   11.6   46.3
    #   |误差|   3.0   2.5   5.4   7.5   15.9   11.0   kJ/mol
    #   ESS      3.6  50.9  37.0   8.4    3.4   24.7
    #
    #   ratio vs |误差|  Spearman = **+0.886**
    #   ESS   vs |误差|  Spearman = **−0.600（反相关！）**
    #
    # 也就是说：**低 ESS 不等于算错，过 ESS 门也不等于算对。**
    # window 5 的 ESS 24.7 过门却错 +11.0；window 0 的 ESS 3.6 最差却只错 +3.0。
    # 这个比值抓得住、ESS 抓不住。
    #
    # 解析对照：壳幅度 4λ(1−λ)、物理 LJ 缩放 λ^n_lj（生产 n_lj=2），
    # 比值 4(1−λ)/λ 在 λ→0 发散。该解析式与上面实测 ratio 的
    # Pearson r = 0.9836、Spearman = 1.0000 —— 所以这不是拟合出来的巧合，
    # 是 λ 缩放不匹配的直接后果。根因与正解见
    # `ibs_engine.py` 约 10978-11020 行的长注释（生产动力学不得保留 Group-4）
    # 与 docs/archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md §2.8。
    #
    # ⚠️ **刻意不设门。** 阈值该定在哪没有实测依据（只有 6 个窗口、一个体系），
    # 而本项目的历史教训是"把一个只在某些假设下成立的度量提成硬门"会反复咬人
    # （见 ESS_GATE_PROTOCOL_VERSION 的版本史）。先落盘、先积累跨体系数据。
    # 谁要把它提成门，必须先有多体系标定，并说明阈值怎么来的。
    # ========================================================================
    _sd_bias = float(np.std(bias_kj))
    _sd_u = np.std(u_kj_raw, axis=1)
    out["bias_fluctuation_kJ_mol"] = _sd_bias
    out["target_energy_fluctuation_kJ_mol"] = [float(x) for x in _sd_u]
    _sd_u_pos = _sd_u[_sd_u > 0.0]
    out["bias_to_signal_ratio"] = (
        float(_sd_bias / float(np.min(_sd_u_pos))) if _sd_u_pos.size else None
    )
    out["bias_to_signal_ratio_is_gated"] = False
    out["bias_to_signal_ratio_note"] = (
        "sd(force groups {1,4}) / min_k sd(U_k_int)：必须被重加权掉的偏置"
        "相对不该被淹没的物理信号。仅诊断、不设门（阈值无跨体系标定）。"
        "4W53 甲苯溶剂腿实测与 |ΔF 误差| Spearman=+0.886，"
        "同批数据 ESS 与误差 Spearman=−0.600。"
    )

    # ---- mixture coverage, common-mode factor divided out (gated) ----
    if f_k is None:
        out["error"] = "missing_f_k"
        return out
    f_arr = np.asarray(f_k, dtype=np.float64).ravel()
    if f_arr.size != n_states or not np.all(np.isfinite(f_arr)):
        out["error"] = "f_k_shape_or_finiteness_mismatch"
        return out

    # ---- pick the gauge the mixture must be built in (see docstring) ----
    if sampling_kj is None:
        if sampling_gauge_required:
            out["error"] = "missing_sampling_state_energies"
            return out
        mix_src = u_kj_raw
        out["mixture_gauge"] = "physical_targets"
    else:
        mix_src = np.asarray(sampling_kj, dtype=np.float64)
        if mix_src.shape != u_kj_raw.shape or not np.all(np.isfinite(mix_src)):
            out["error"] = "sampling_state_energy_shape_or_finiteness_mismatch"
            return out
        out["mixture_gauge"] = "sampling_states"

    logits = -(mix_src - f_arr[:, None]) / float(kt)
    log_norm = _logsumexp_rows(logits, axis=0)
    log_p = logits - log_norm[None, :]
    mix_ess = [_ess_from_log_weights(log_p[k]) for k in range(n_states)]
    out["mixture_ess"] = [float(x) for x in mix_ess]
    out["mixture_ess_ratio"] = [float(x) / float(n_frames) for x in mix_ess]

    # 🔑 ESS 单独用**不够**，必须配上占据 <p_k>：ESS 对每个态是尺度不变的
    # ((sum w)^2/sum w^2 在 w -> c*w 下不变)，所以一个"均匀地"被饿死的态——比如
    # f_k 没补偿、U'_k 比别人高 80 kT——它的 p_k 逐帧都是 ~e^-80，但**相对**逐帧
    # 起伏很小，ESS(p_k)/N 照样能报 0.34 这种健康值。实测确认过这个盲点。
    # ESS 查的是二阶矩（权重是否集中在少数帧），<p_k> 查的是一阶矩（这个态到底
    # 有没有拿到权重），两个失效模式必须各由一项覆盖。归一化成 K*<p_k>（理想=1），
    # 沿用 warmup 那边同一套占据判据的口径（IBS_LOCAL_MBAR_GATE_OCC_MIN_FRACTION）。
    occ = np.exp(log_p).mean(axis=1)
    occ_sum = float(np.sum(occ))
    if occ_sum > 0.0 and np.all(np.isfinite(occ)):
        occ = occ / occ_sum
        out["mixture_occupancy"] = [float(x) for x in occ]
        out["mixture_occupancy_normalized"] = [
            float(x) * float(n_states) for x in occ
        ]

    # r_n: the part of the applied sampling bias that no (U'_k, f_k) pair can
    # reproduce -- i.e. the WCA guard shell (Group 4) plus the softcore-vs-LRC
    # Hamiltonian mismatch. -kt*log_norm is the IBS bias the recorded f_k/U'_k
    # imply; the difference is the common-mode nuisance factor.
    residual_kj = bias_kj + float(kt) * log_norm
    if np.all(np.isfinite(residual_kj)):
        out["common_mode_log_sigma_kT"] = float(np.std(residual_kj) / float(kt))
        out["common_mode_log_mean_kJ_mol"] = float(np.mean(residual_kj))
    return out


# v2 [2026-09-16]：把 `N_eff/g` 的两个档位从函数体里的字面量提成常量，并**写进产物**。
# 纯附加键，不进任何指纹（全仓只有这个文件读写它），旧产物用 `.get()` 读仍然安全。
# 动机：控制器要判「加帧还能不能把这个窗口推过门」就必须知道门在哪；先前它只能
# 把 10.0 抄一份 —— 那正是本仓最贵的那类 bug（同一不变量两份实现）。
WINDOW_SELF_SUPPORT_PROTOCOL_VERSION = 2

# 协议安全下限（老板定，见 `window_self_support_check` 里那段推导）：
#     min(N_eff/g) < 1      HARD_INSUFFICIENT
#     1 <= ... < 10         INSUFFICIENT_DATA
#     >= 10                 ANALYSIS_ELIGIBLE
WINDOW_SELF_SUPPORT_N_EFF_OVER_G_HARD_FLOOR = 1.0
WINDOW_SELF_SUPPORT_N_EFF_OVER_G_ELIGIBLE = 10.0


def window_self_support_path(output_dir: str, stage_type: str, idx: int) -> str:
    return os.path.join(
        output_dir, f"dual_window_{int(idx)}_{stage_type}_self_support.json"
    )


def _load_self_support_decorrelation_provenance(w) -> Optional[Dict[str, Any]]:
    """从窗口自检产物里取出**自检那一侧**的去相关 provenance；取不到就 None。

    读不到（没跑自检 / 旧产物没有这个字段 / 文件坏了）一律返回 None ——
    这是诊断输入，缺了只是"这个窗口没法做对照"，绝不能把分析带崩。
    """
    path = (w or {}).get("self_support_path")
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("decorrelation_provenance")
    except (OSError, ValueError):
        return None


def window_self_support_check(
    output_dir: str,
    stage_type: str,
    window_idx: int,
    kt: float,
    *,
    min_frames_per_window: int = 10,
) -> Optional[Dict[str, Any]]:
    """窗口**自己**的支撑够不够 —— 在它刚跑完那一刻就判，不等全部窗口跑完。

    设计依据：``docs/archive/PLAN_PATH_REPAIR_2026-09-11.md`` P2-9c。
    老板原话：「每个窗口跑完就判一次，这不是更早判定吗」。

    **为什么值得做** —— 实测浪费（cyclod_ligand2/rep1 的 launch.log 顺序）::

        [窗口 4] ... 生产采样 (250000 步)
        [OK] 所有窗口采样完成                  ← 5×250k = 125 万步全部烧完
          [WARN] 窗口 4 去相关后 7 帧 < 10，跳过
          [WARN] 窗口 3 ... 5 帧，跳过
          [WARN] 窗口 2 ... 5 帧，跳过
          rescue round 1/2: 仅追加窗口 [0,1,2,3,4] → 全部 500000

    判断依据（该窗口去相关后剩几帧）在**它自己刚跑完那一刻**就算得出来，不需要
    任何别的窗口。win0 在第 4、第 5 个窗口跑的时候，它自己那份数据早就确定不够了，
    只是没有任何东西去看。

    **与事后判定逐字同量**，否则"提前判"和"事后判"会给不同答案：
      · 数组用 ``energies.npy``（分析器 `load_ibs_window_outputs_from_dir` 读的就是它，
        作为 ``u_kj_raw``），**不是** ``sampling_states.npy``。g 本身对逐态常数不变，
        但口径要跟判跳过的那一处对齐、可审计。
      · 分段用 ``convergence.json`` 的 ``production_segments``（分析器
        ``_decorr_segments = w.get("production_segments")``）。
      · 门槛默认 10 = ``solve_stage_integrated`` 的 ``min_frames_per_window``。

    **只判不动手。** 结果只落盘 + 打印，**不**改 ``production_step_overrides``、
    **不**碰 ``converged`` 或 rescue 的任何既有判据。把预算加给谁是 `decide()` 的活，
    而 `decide()` 还不存在；在钩子里塞一条会自己改预算的旁路只会让归因更难。

    **"帧数不足"不是终态**（老板更正：``剩 6 帧 → 整窗丢掉`` ✗、``剩 6 帧 → +250k`` ✓）。
    所以 verdict 用 ``INSUFFICIENT_DATA``（≠ ``FAIL``：样本不够是"还没测够"，
    不是"测出来不合格"）。补多少是待定口径（加法一个 ``n_steps_per_window`` 基础单位
    vs 现在 rescue 的 ``growth=2.0`` 乘法，PLAN P3-9f），本函数**不给数字**。

    读不到文件等情况返回 None —— 这是诊断，不许因此中断采样。
    """
    conv_path = os.path.join(
        output_dir, f"dual_window_{int(window_idx)}_{stage_type}_convergence.json"
    )
    try:
        with open(conv_path, "r", encoding="utf-8") as fh:
            conv = json.load(fh)
    except (OSError, ValueError):
        return None
    try:
        u = np.asarray(np.load(os.path.join(
            output_dir, f"dual_window_{int(window_idx)}_{stage_type}_energies.npy"
        )), dtype=np.float64)
        bias = np.asarray(np.load(os.path.join(
            output_dir, f"dual_window_{int(window_idx)}_{stage_type}_bias.npy"
        )), dtype=np.float64).ravel()
    except (OSError, ValueError):
        return None
    if u.ndim != 2 or u.shape[1] != bias.size:
        return None

    segments = conv.get("production_segments") or None
    seg_diag: List[Dict[str, Any]] = []
    g_per_state: List[float] = []
    try:
        sub_idx, g_val, g_worst = _decorrelate_by_worst_target_state(
            u, bias, float(kt), segments=segments,
            segment_diagnostics=seg_diag, per_state_g_out=g_per_state,
        )
    except Exception:
        return None

    # 🔑 [2026-09-11 / 老板方法论定位] **验收量是 `N_eff,k / g_k`，不是 occupancy。**
    #
    # 能不能恢复 p_k，问的是 ``w_kn ∝ e^{-βU_k(x_n)} / q_IBS(x_n)`` 有没有足够
    # support，**不是 IBS mixture 内部的 component occupancy 漂不漂亮**。
    # 判例：win1 段1 ``occupancy_collapsed=True`` 但支撑健康 ⟹ occupancy 是**假阳性**。
    # ⟹ occupancy 是 **f_k 的训练目标**，不是**结果的验收判据**。
    #
    # 而 ``N_eff,k`` 单独也不够 —— 它把帧当独立的。真正的量是 ``N_eff,k / g_k``
    # （未抽稀帧上的逐目标态 support ÷ 该态的时间自相关）。实测差距极大：
    #     win2  853/50  ≈ 17        （健康）
    #     win0  27.5/126 ≈ **0.2**  （几乎没有独立的有效样本）
    #
    # 🔑 [2026-09-14 / 审计 #53] **分子和分母必须来自同一批帧。**
    # `_decorrelate_by_worst_target_state` 会把 <20 帧的短重启段整段剔掉再逐态取
    # 最大 g，而这里原来在**全部**帧（含被剔段）上算 N_eff ⟹ 主验收量
    # `min_n_eff_over_g` 成了「全段分子 ÷ 最坏段分母」，重启多的窗口被系统性压低。
    # 现在只用**保留下来的那些段**的帧（段的取舍直接读 seg_diag，不在这里重抄
    # <20 帧那条规则，否则两边迟早分叉）。
    _excluded_segments = [d for d in seg_diag if d.get("reason_excluded")]
    if segments:
        _kept_mask = np.zeros(int(bias.size), dtype=bool)
        for _d in seg_diag:
            if not _d.get("reason_excluded"):
                _kept_mask[int(_d["start_frame"]):int(_d["end_frame"])] = True
        _u_fs, _bias_fs = u[:, _kept_mask], bias[_kept_mask]
    else:
        _u_fs, _bias_fs = u, bias
    _n_frames_frame_set = int(_bias_fs.size)

    # 这里在**未抽稀**帧上逐态算 ——  抽稀之后再算等于把 g 除了两遍。
    n_eff_per_state: List[Optional[float]] = []
    top1_per_state: List[Optional[float]] = []
    # 🔑🔑 [2026-09-17] **逐态记下「真正进了 N_eff 的帧数」。**
    # `_bias_fs.size` 是 finite mask **之前**的数；而下面每个态各自过一次
    # `np.isfinite`（不同态的非有限帧可以不同）。效率 η = N_eff/N 的分母必须是
    # **这个态自己**的有效帧数，否则 η 会被非有限帧系统性压低 —— 而那是个
    # 隐蔽误差：它只在有非有限帧时出现，且永远偏向"看起来更差"。
    n_eff_input_n_frames_per_state: List[Optional[int]] = []
    for _k in range(_u_fs.shape[0]):
        _ser = (_u_fs[_k] - _bias_fs) / float(kt)
        _fin = np.isfinite(_ser)
        if not _fin.any():
            n_eff_per_state.append(None)
            top1_per_state.append(None)
            n_eff_input_n_frames_per_state.append(0)
            continue
        _sr = _ser[_fin]
        n_eff_input_n_frames_per_state.append(int(_sr.size))
        # 减最小值只为数值稳定；ESS 对 w 的公共因子不变。
        _w = np.exp(-(_sr - _sr.min()))
        _sw, _sw2 = float(_w.sum()), float((_w * _w).sum())
        n_eff_per_state.append((_sw * _sw / _sw2) if _sw2 > 0.0 else None)
        # [审计 #48] 与生产支撑度门共用同一份 CDF 插值实现；旧的整帧计数在 N<100
        # 时退化成「最大单帧占比」，却和这里的 top1_veto 共用阈值 0.35。
        top1_per_state.append(
            float(_top1pct_weight_by_empirical_cdf(_w)) if _sw > 0.0 else None
        )
    n_eff_over_g: List[Optional[float]] = []
    for _k, _ne in enumerate(n_eff_per_state):
        _g = g_per_state[_k] if _k < len(g_per_state) else None
        n_eff_over_g.append(
            float(_ne) / float(_g) if (_ne is not None and _g and _g > 0.0) else None
        )
    _finite_ratio = [x for x in n_eff_over_g if x is not None]

    # 🔑 [2026-09-11] **分块 N_eff 序列 + 脱轨点** —— 早判的判据本身。
    #
    # 🔑🔑 [2026-09-14 / 审计 #1] **判据量已从「前缀差」换成「块内独立 ESS」。**
    # 原来 `marginal[b] = ESS(prefix_b) − ESS(prefix_{b−1})`，而 ESS 是比值泛函
    # `(Σw)²/Σw²`、**不可加**：当某一帧的权重 W ≫ 其余时，前缀 ESS 反而塌向 1，
    # 于是「第一次采到主导构型」这件好事被算成一个大负增量、判成
    # `CONFIRMED_DERAILMENT` ⟹ 允许关掉 Epoch ⟹ 丢掉那批高权重帧 ⟹ vanishing
    # 腿欠采样 ⟹ **低估解耦代价**。（下面那串实测数字 `73.3 23.1 2.3 3.0` 正是
    # 这个签名，不是脱轨签名。）
    # 现在每个 block 只用**它自己的帧**算一次 ESS，块与块之间互不相减；前缀 ESS
    # 仍然落盘（`prefix_ess_by_block`），但只作诊断，名字里写明它不是边际量。
    #
    # 冻结的 f_k 会随轨迹越采越偏（"乒乓球效应"）；健康时各块的 block-local ESS
    # 大致持平，脱轨后断崖。这种落差不需要精细阈值。
    # 判据（口径不变）：**本块 block-local ESS < 前半程中位数的 1/4**。
    #
    # ⚠️ 早判比固定节奏**更准**：实测脱轨点跨 0.40~1.80 ns（**4.5 倍**）——
    # 固定 1 ns 对段2 win1（0.4 ns 就该换）太晚、对段2 win0（1.8 ns 才需要）偏早。
    # 两者不冲突：固定节奏当**上限兜底**，边际断崖当**提前触发**，谁先到用谁。
    #
    # 定义与同僚手算口径逐条对齐（对不上先查这四条）：
    #   · w_kn = exp(-(u_kn - bias_n)/kT)，u 用 energies.npy、bias 用 bias.npy
    #   · **未抽稀**，不走 _decorrelate_by_worst_target_state
    #   · N_eff,k = (Σw)²/Σw²；最差态 = 整段 N_eff 最小的那个 k
    #   · 数值上先减 max(log w) 再 exp —— 这里的 exp(-(s - s.min())) 就是它
    #     （log w = −s ⟹ max(log w) = −min(s)），不是另一个式子。
    n_blocks = 10
    block_local: List[Optional[float]] = []
    prefix_ess: List[Optional[float]] = []
    derail_block = None
    derail_block_single = None
    derail_threshold = None
    worst_k = (
        int(min(range(len(n_eff_per_state)),
                key=lambda i: (n_eff_per_state[i] is None, n_eff_per_state[i])))
        if any(x is not None for x in n_eff_per_state) else None
    )
    if worst_k is not None and bias.size >= n_blocks:
        _ser = (u[worst_k] - bias) / float(kt)

        def _ess_of(_s: np.ndarray) -> Optional[float]:
            """一段序列自己的 Kish ESS。块与块之间**不相减** —— ESS 不可加。"""
            _s = _s[np.isfinite(_s)]
            if _s.size == 0:
                return None
            _w = np.exp(-(_s - _s.min()))
            _sw, _sw2 = float(_w.sum()), float((_w * _w).sum())
            return (_sw * _sw / _sw2) if _sw2 > 0.0 else None

        for _b in range(1, n_blocks + 1):
            _lo = int(round(bias.size * (_b - 1) / n_blocks))
            _cut = int(round(bias.size * _b / n_blocks))
            prefix_ess.append(_ess_of(_ser[:_cut]))          # 诊断，非边际量
            block_local.append(_ess_of(_ser[_lo:_cut]))      # 判据量：块内独立
        _early = [x for x in block_local[: n_blocks // 2] if x is not None]
        if _early:
            _med = float(np.median(_early))
            _thr = _med / 4.0
            # 老板定的触发规格是「**连续两个** block < 前半程中位数的 1/4」——
            # 比"第一个低于即判"严，能挡掉单块噪声造成的假阳性。
            # 扫描范围与改动前一致（全 10 块），只是被扫的量换成了 block-local。
            for _i in range(len(block_local) - 1):
                _a, _b2 = block_local[_i], block_local[_i + 1]
                if (_a is not None and _a < _thr
                        and _b2 is not None and _b2 < _thr):
                    derail_block = _i
                    break
            # 单块口径**同时保留**（仅报告）：两者在实测数据上会分歧，需要并排看。
            for _i, _m in enumerate(block_local):
                if _m is not None and _m < _thr:
                    derail_block_single = _i
                    break
            derail_threshold = _thr

    n_dec = int(np.asarray(sub_idx).size)
    floor = int(min_frames_per_window)

    # 🔑🔑 [2026-09-11 / 老板定案 A+H] **验收真源是 `N_eff/g`，不是 `n_decorr >= 10`。**
    #
    # 先前这里用 `n_dec >= 10` 判 `VALID_PASS` —— 那是"求解器有没有资格尝试"的
    # 门槛，被当成了"目标态能不能被恢复"的验收，两件事。四种量的分工现在写死：
    #
    #   n_decorr            求解器**是否有资格尝试**（够不够喂进 MBAR）
    #   N_eff,k / g_k       目标态能否被恢复 —— **主验收量**
    #   top1% weight        重尾 / 单帧支配的**否决警报**
    #   occupancy、coverage/mixture ESS
    #                       **只诊断 f_k 训练**，不决定最终 ΔG 是否有效
    #
    # 协议安全下限（老板定；五个窗口不足以拟合"科学最优阈值"，但足够定下限）::
    #
    #     min(N_eff/g) < 1     HARD_INSUFFICIENT     物理上不到一个独立有效贡献
    #     1 <= ... < 10        INSUFFICIENT_DATA
    #     >= 10                ANALYSIS_ELIGIBLE
    #
    # ⚠️ `>= 10` 只表示"**可以进入分析**"，**不是 `VALID_PASS`**。最终 PASS 仍需
    #    endpoint CI、block 稳定性、全路径完整性 —— 那些不在本函数的视野里。
    # ⚠️ **低支撑永远不叫 FAIL。** 它是"尚不可测"（INSUFFICIENT_DATA）。只有支撑
    #    充分之后结果明确违反统计门，才轮得到 FAIL。把两者混起来会退化成
    #    "再测一次直到碰巧通过"。
    _ratio = min(_finite_ratio) if _finite_ratio else None
    _top1_worst = max((x for x in top1_per_state if x is not None), default=None)
    if _ratio is None:
        verdict = "INSUFFICIENT_DATA"      # 连主验收量都算不出来
    elif _ratio < WINDOW_SELF_SUPPORT_N_EFF_OVER_G_HARD_FLOOR:
        verdict = "HARD_INSUFFICIENT"
    elif _ratio < WINDOW_SELF_SUPPORT_N_EFF_OVER_G_ELIGIBLE:
        verdict = "INSUFFICIENT_DATA"
    else:
        verdict = "ANALYSIS_ELIGIBLE"
    # top1% 是**否决警报**，不是主判据：支撑看着够、却被单帧支配时压回不可分析。
    # 阈值用既有的 TARGET_SUPPORT_MAX_TOP1PCT_WEIGHT（生产支撑度门本来就用它），
    # 不新发明。
    top1_veto = bool(
        _top1_worst is not None
        and _top1_worst > float(TARGET_SUPPORT_MAX_TOP1PCT_WEIGHT)
    )
    if top1_veto and verdict == "ANALYSIS_ELIGIBLE":
        verdict = "INSUFFICIENT_DATA"
        verdict_source = "top1pct_veto"
    else:
        verdict_source = "min_n_eff_over_g"

    # 🔑 [2026-09-11 补] **`n_decorr` 的角色必须真正参与，不能只落盘不生效。**
    #
    # 四量分工里 `n_decorr` = "**求解器是否有资格尝试**"，这是比 `N_eff/g` **更前置**
    # 的一层：连喂进 MBAR 的去相关样本都不够时，`N_eff/g` 算出来的任何数都不该被
    # 当成结论。先前这里只把 `solver_eligible` 落了盘、没让它影响 verdict，实测
    # `min_frames_per_window` 从 1 调到 10000 对 verdict **零影响** —— 两者碰巧同向
    # 所以没暴露（tight 组 n_decorr=3 不够，但 verdict 是 N_eff/g=2.28 给的）。
    #
    # 处置：不够资格 ⟹ `HARD_INSUFFICIENT`（这一档本来就是留给"连一个独立有效
    # 贡献都没有"的），并且**它压过 N_eff/g 的分档** —— 更前置的失效优先。
    # 但 **`verdict_source` 要把"为什么"分开记**：帧数不够 ≠ 支撑不够，
    # 补救方向虽然都是加采样，事后归因却不能混。
    # ⚠️ 仍然**不叫 FAIL**：没资格尝试是"尚不可测"，不是"测出来不合格"。
    solver_eligible = n_dec >= floor
    if not solver_eligible:
        verdict = "HARD_INSUFFICIENT"
        verdict_source = "solver_eligibility"
    sufficient = verdict == "ANALYSIS_ELIGIBLE"
    lam = [float(x) for x in (conv.get("lambdas_vdw") or [])]
    return {
        "protocol_version": WINDOW_SELF_SUPPORT_PROTOCOL_VERSION,
        "window_idx": int(window_idx),
        "gauge": "energies",
        "checked_at": "window_production_complete",
        # [DECORR-01 裁决 4] 这一侧去相关计算的**输入身份 + 输出**。
        # 与求解器那一份放在一起才能做同输入对照（先比输入构造，再重放）。
        "decorrelation_provenance": decorrelation_provenance(
            caller="window_self_support_check",
            energy_array_source=f"dual_window_{int(window_idx)}_{stage_type}_energies.npy",
            observable="delta_u_k_over_kT_worst_target_state",
            n_frames_in=int(bias.size), segments=segments, kt=float(kt),
            g=(None if not np.isfinite(g_val) else float(g_val)),
            worst_state=g_worst, subsample_indices=sub_idx,
            window_idx=int(window_idx),
            # [CTL-09] 未抽稀的原始输入 —— 对照要比的就是"喂进去的是不是同一批数"。
            energy_array=u, bias_array=bias,
        ),
        "n_frames_total": int(bias.size),
        "n_frames_decorrelated": n_dec,
        "min_frames_per_window": floor,
        "statistical_inefficiency": (
            None if not np.isfinite(g_val) else float(g_val)
        ),
        "worst_target_state_local": int(g_worst),
        "worst_target_lambda_vdw": (
            float(lam[g_worst]) if 0 <= g_worst < len(lam) else None
        ),
        # 逐态 g 剖面：远端单调衰减（窗口跨度太大）与整体偏低（采样不够）在单个 g
        # 上长得一模一样，而两者的对症动作完全不同。
        "statistical_inefficiency_per_lambda": [float(x) for x in g_per_state],
        # 🔑 验收量（未抽稀帧上逐目标态）。occupancy **不在**这里 —— 它是 f_k 的
        # 训练目标，不是结果的验收判据（win1 的 occupancy_collapsed 是假阳性）。
        "n_eff_per_state": [None if x is None else float(x) for x in n_eff_per_state],
        # [审计 #53] N_eff 与 g 现在同帧集：短段（<20 帧，建立不了自相关）被 g 那边
        # 剔掉，这里也剔掉。字段落盘是为了事后能判「这批数字是哪套帧集算的」。
        "n_eff_frame_set": (
            "segments_kept_by_decorrelation" if segments else "all_frames"
        ),
        # ⚠️ 这是 finite mask **之前**的帧数（整个 frame set 的大小）。
        # 算 η 用的是下面那份**逐态**的，别拿这个当分母。
        "n_eff_frame_set_n_frames_raw": _n_frames_frame_set,
        # 保留旧键名，老读侧不炸；新代码一律用 `_raw` 那个（语义写在名字里）。
        "n_eff_frame_set_n_frames": _n_frames_frame_set,
        # 帧集身份：哪些 segment 被 decorrelation 判定为有效并保留下来。
        "n_eff_frame_set": "segments_kept_by_decorrelation",
        # 🔑 逐态**实际进了 N_eff 的帧数**（已排除无效 segment、**未**按 sub_idx
        # 去相关稀疏、且对该态应用过 finite mask）。η 的唯一合法分母。
        "n_eff_input_n_frames_per_state": [
            None if x is None else int(x) for x in n_eff_input_n_frames_per_state
        ],
        "n_eff_frame_set_excluded_segments": len(_excluded_segments),
        "top1pct_weight_per_state": [
            None if x is None else float(x) for x in top1_per_state
        ],
        "n_eff_over_g_per_state": [
            None if x is None else float(x) for x in n_eff_over_g
        ],
        "min_n_eff_over_g": (min(_finite_ratio) if _finite_ratio else None),
        # [v2] 门槛随产物一起交出去：读侧（控制器）判"加帧还能不能推过门"时
        # 必须用**这一次**实际生效的档位，不许自己抄一份常量。
        "n_eff_over_g_eligible_threshold": float(
            WINDOW_SELF_SUPPORT_N_EFF_OVER_G_ELIGIBLE
        ),
        "n_eff_over_g_hard_floor": float(
            WINDOW_SELF_SUPPORT_N_EFF_OVER_G_HARD_FLOOR
        ),
        "worst_state_by_n_eff_over_g": (
            int(min(range(len(n_eff_over_g)),
                    key=lambda i: (n_eff_over_g[i] is None, n_eff_over_g[i])))
            if _finite_ratio else None
        ),
        # 分块序列 = N_eff,k 的时间分辨版本，也是"早判"的判据本身。
        "worst_state_by_n_eff": worst_k,
        # [审计 #1] 判据量：每块只用自己的帧算 ESS（块内独立、可比）。
        "block_local_ess_by_block": block_local,
        # [审计 #1] 前缀 ESS：**只作诊断**。名字里写死 "prefix" 是因为它原来叫
        # cumulative、被拿去做差当"边际量"用 —— ESS 不可加，那个差没有意义。
        "prefix_ess_by_block": prefix_ess,
        # 向后兼容别名：控制器（abfe_preoptimizer）仍按这个键读。值已换成
        # block-local（判据量本身），**不再**是前缀差。
        "n_eff_marginal_by_block": block_local,
        "block_metric": "block_local_ess",
        "n_blocks": n_blocks,
        "derail_block_index": derail_block,
        # 脱轨点表示为**轨迹比例**（该块的末端）；不换算成 ns，因为这里拿不到
        # 可靠的 dt，换算留给读的人。None = 整段未脱轨。
        "derail_at_trajectory_fraction": (
            None if derail_block is None else (derail_block + 1) / float(n_blocks)
        ),
        "derail_criterion": (
            "**连续两个** block 的 **block-local ESS** < 前半程中位数的 1/4"
            "（老板定的触发规格；比单块严，挡掉单块噪声假阳性）。"
            "[2026-09-14 审计 #1] 判据量已从前缀 ESS 之差换成块内独立 ESS —— "
            "ESS 是比值泛函、不可加，前缀差会把「第一次采到主导构型」判成脱轨。"
        ),
        "derail_threshold": derail_threshold,
        # 🔑 [2026-09-11 老板定案] **脱轨判据是两档，不是二选一。**
        #
        #   单块 < 1/4      → SUSPECTED_DERAILMENT：触发**非变异**探针 / 候选计算
        #   连续两块 < 1/4  → CONFIRMED_DERAILMENT：才允许关闭 Epoch、进入动作决策
        #
        # ⚠️ **在 held-out 反事实验收接好之前，单块不得直接触发重标定。**
        # 理由（老板原话）：这样段2 win1 虽然正式动作会晚到 0.90，但 **0.40 时
        # 已经启动廉价诊断**；不会再因一个震荡低块贸然切换到更差的 f_k。
        # 待 held-out 接通后，单块预警可以立即评估候选，只有确认支撑改善才切换。
        "derail_block_index_single_block": derail_block_single,
        "derailment_status": (
            "CONFIRMED_DERAILMENT" if derail_block is not None
            else "SUSPECTED_DERAILMENT" if derail_block_single is not None
            else "NONE"
        ),
        "derailment_tiers": {
            "SUSPECTED_DERAILMENT": "单块 < 前半程中位数 1/4 ⟹ 只触发非变异探针/候选计算",
            "CONFIRMED_DERAILMENT": "连续两块 < 1/4 ⟹ 才允许关闭 Epoch、进入动作决策",
        },
        "n_segments": len(segments) if segments else 1,
        "segment_diagnostics": seg_diag,
        # `sufficient` 现在的含义是 **ANALYSIS_ELIGIBLE**（可以进入分析），
        # **不是** "通过验收"。
        "sufficient": bool(sufficient),
        "solver_eligible": bool(solver_eligible),
        "solver_eligibility_criterion": "n_decorrelated >= min_frames_per_window",
        # ⚠️ INSUFFICIENT_DATA / HARD_INSUFFICIENT **都不是 FAIL**。低支撑是
        # "尚不可测"，加预算；FAIL 是"支撑够了但违反统计门"，换 Epoch。
        "verdict": verdict,
        # 为什么是这个 verdict：solver_eligibility（帧数不够，连尝试的资格都没有）/
        # min_n_eff_over_g（主验收量）/ top1pct_veto（单帧支配）。三者补救方向都是
        # 加采样，但归因不同，不能混。
        "verdict_source": verdict_source,
        "verdict_thresholds": {"hard_insufficient_below": 1.0,
                               "analysis_eligible_at_or_above": 10.0},
        "top1pct_veto": top1_veto,
        "top1pct_veto_threshold": float(TARGET_SUPPORT_MAX_TOP1PCT_WEIGHT),
        "worst_top1pct_weight": _top1_worst,
        "quantity_roles": {
            "n_decorrelated": "求解器是否有资格尝试",
            "n_eff_over_g": "目标态能否被恢复 —— 主验收量",
            "top1pct_weight": "重尾/单帧支配的否决警报",
            "occupancy_and_coverage_ess": "只诊断 f_k 训练，不决定 ΔG 是否有效",
        },
        # ⚠️ 这是**去相关帧数**的缺口，只在去相关这一关真的没过时才有意义。
        # 去相关过了、却因 min N_eff/g 不达标而 insufficient 时它是 0 ——
        # 真机印出过「还差 0 帧」这种自相矛盾的话。所以那种情况给 None，
        # 让调用方明确区分"缺帧"和"缺有效样本"。
        "frames_short_by": (max(0, floor - n_dec) if n_dec < floor else None),
        "remedy": None if sufficient else (
            "这个窗口还需要加采样（它的 checkpoint 此刻还热）。加多少是待定口径："
            "加法一个 n_steps_per_window 基础单位 vs 现在 rescue 的 growth=2.0 乘法"
            "（PLAN P3-9f）。本检查**只判不动手**，由既有 rescue 通路去加。"
        ),
        "note": "只报告、不设门；不改 production_step_overrides、不碰 converged。",
    }


JOIN_SUPPORT_PROTOCOL_VERSION = 1


def join_lambda_two_sided_support(
    output_dir: str,
    stage_type: str,
    upstream_idx: int,
    downstream_idx: int,
    kt: float,
    *,
    gauge: str = "sampling_states",
    lambda_atol: float = 1e-9,
) -> Optional[Dict[str, Any]]:
    """相邻两个窗口对它们**共享的那一个 λ** 各自给出的重要性支撑。

    设计依据：``docs/archive/PLAN_PATH_REPAIR_2026-09-11.md`` §3bis（跨窗累加）+ P2-9a。

    **为什么需要它**：IBS 窗口的样本本来就能跨窗累加（每个窗口是一个已知的采样
    分布，MBAR 天生处理多个采样分布），所以"某个 λ 到底采够没有"这个问题**不该只问
    拥有它的那一个窗口**。上一跑实测的不对称是系统性的 —— 每个 join 上，**结束**于
    该 λ 的窗口总是比**起始**于它的那个差（rawESS）::

        join λ    上游    下游
        0.6079     16      332
        0.4847    163      270
        0.3852    205      838
        0.3032     91      138

    用单窗判据问，答案是"整个 win0 不可用"（它因去相关后只剩 6 帧被整窗踢出
    ΔG 求和）；用两侧判据问，答案是"win0 的**内部**态缺支撑，它的**端点**没问题
    （下游采得好 20 倍）"。后者才是可执行的诊断。

    **只报告、不设门。** 放行判据一律不动 —— 这一条是证据，不是裁决。

    **递推单元是滑动相邻对，不是累积前缀**：相邻窗口只共享**一个** λ，非相邻窗口的
    λ 集合完全不相交，所以 ``win0+win1+win2`` 相比两次 pairwise join 不多任何信息
    （§3bis.3）。⚠️ 这**不是**被否决的那个设计（把相邻 λ 态之间的 fixed-H overlap
    当 IBS 正确性仲裁）：这里比的是两个**独立系综**对**同一个** λ 态的重要性支撑，
    是同一个量的两份独立估计。

    **口径（gauge）**：``sampling_states.npy`` 是 (N, K)、与 ``bias.npy`` 同口径；
    ``energies.npy`` 是 (K, N) 的**物理**能量，两者差一个**逐态常数**（LRC，实测
    win0 是 −8.08…−2.99、std=0.000）。⚠️ 注意两份数组**转置方向相反**。
    ESS 与 g 对逐态常数都是不变量 —— ``ESS=(Σw)²/Σw²`` 在 ``w→aw`` 下不变、
    自相关对平移不变（同僚实测 g 逐位相同）—— 所以用哪份都行，**但不许混用**，
    而且落盘必须写清用的是哪一份。默认用 ``sampling_states``（它才是 bias 的口径）。
    **绝不**拿 ``energies`` 去重建 bias（那样会差 7.5 kJ/mol）。

    读不到文件、两窗不共享 λ 等情况一律返回 None（这是诊断，不许因此中断采样）。
    """
    if gauge not in ("sampling_states", "energies"):
        raise ValueError(f"未知口径 {gauge!r}；只接受 sampling_states / energies")

    def _load(idx: int):
        conv_path = os.path.join(
            output_dir, f"dual_window_{idx}_{stage_type}_convergence.json"
        )
        try:
            with open(conv_path, "r", encoding="utf-8") as fh:
                lam = [float(x) for x in (json.load(fh).get("lambdas_vdw") or [])]
        except (OSError, ValueError):
            return None
        try:
            bias = np.load(
                os.path.join(output_dir, f"dual_window_{idx}_{stage_type}_bias.npy")
            ).ravel()
            if gauge == "sampling_states":
                # (N, K) —— 与 bias 同口径
                u = np.load(os.path.join(
                    output_dir, f"dual_window_{idx}_{stage_type}_sampling_states.npy"
                )).T
            else:
                # (K, N) —— 物理能量
                u = np.load(os.path.join(
                    output_dir, f"dual_window_{idx}_{stage_type}_energies.npy"
                ))
        except (OSError, ValueError):
            return None
        u = np.asarray(u, dtype=np.float64)
        if u.ndim != 2 or u.shape[0] != len(lam) or u.shape[1] != bias.size:
            return None
        return lam, u, bias

    up, down = _load(int(upstream_idx)), _load(int(downstream_idx))
    if up is None or down is None:
        return None
    lam_up, u_up, bias_up = up
    lam_dn, u_dn, bias_dn = down
    if not lam_up or not lam_dn:
        return None
    # 共享态：上游的**末**态 == 下游的**首**态（单一共享边界约定）
    if abs(lam_up[-1] - lam_dn[0]) > float(lambda_atol):
        return None
    join_lambda = float(lam_dn[0])

    def _side(u: np.ndarray, bias: np.ndarray, k: int) -> Dict[str, Any]:
        series = (u[k] - bias) / float(kt)
        finite = np.isfinite(series)
        n_total = int(series.size)
        if not finite.any():
            return {"n_frames": n_total, "n_finite": 0, "raw_ess": None,
                    "raw_ess_ratio": None, "statistical_inefficiency_g": None,
                    "top1pct_weight": None}
        sr = series[finite]
        # 减最小值只为数值稳定；ESS 对 w 的公共因子不变，结果不受影响。
        w = np.exp(-(sr - sr.min()))
        sw, sw2 = float(w.sum()), float((w * w).sum())
        ess = (sw * sw / sw2) if sw2 > 0.0 else None
        try:
            _idx, g = subsample_series_by_autocorrelation(sr)
            g = float(g)
        except Exception:
            g = None
        # [审计 #48] 与生产支撑度门 / 自检共用同一份经验 CDF 插值实现。
        top = float(_top1pct_weight_by_empirical_cdf(w)) if sw > 0.0 else None
        return {
            "n_frames": n_total,
            "n_finite": int(finite.sum()),
            "raw_ess": None if ess is None else float(ess),
            "raw_ess_ratio": None if ess is None else float(ess) / float(sr.size),
            # 🔑 [2026-09-14 / 审计 #54] `subsample_series_by_autocorrelation`
            # 返回的是**统计低效率 g**，不是 τ_int（`g = 1 + 2τ_int`，差约 2 倍）。
            # 落盘键改名成它真正是的东西，并且**不再写 `tau_int`**。
            #
            # ⚠️ 为什么不留一个"名字不变、值改成真 τ"的兼容别名：那是最难发现的一类
            # 改动 —— grep 找得到键，但语义对不对要看调用方的标签，漏改的读点会静默
            # 差一倍且不抛任何异常（已抓到活体：控制器 render 的 join 表表头标 `g`、
            # 读的却是 `tau_int`）。
            # **历史产物里的 `tau_int` 装的一直是 g（名字错、值对）**，所以读点统一写
            # `d.get("statistical_inefficiency_g", d.get("tau_int"))`：新产物走新键、
            # 旧产物回退旧键，两边拿到的都是 g；漏改的读点在新产物上拿到 `None`
            # （显式缺失，看得见）。若将来真需要 τ_int 本身，用**新名字**
            # （如 `tau_int_from_g` = (g−1)/2），绝不复用 `tau_int`。
            "statistical_inefficiency_g": g,
            "top1pct_weight": top,
        }

    side_up = _side(u_up, bias_up, len(lam_up) - 1)
    side_dn = _side(u_dn, bias_dn, 0)
    ratio = None
    if side_up["raw_ess"] and side_dn["raw_ess"]:
        ratio = float(side_dn["raw_ess"]) / float(side_up["raw_ess"])
    return {
        "protocol_version": JOIN_SUPPORT_PROTOCOL_VERSION,
        "gauge": gauge,
        "note": (
            "只报告、不设门。ESS/g 对逐态常数（LRC）是不变量，所以 gauge 选哪个"
            "都行；但不许混用，也绝不拿 energies 去重建 bias。"
        ),
        # 🔑🔑 [2026-09-11 老板纠正] **这份证据的作用范围，别越界。**
        "scope_limitation": (
            "本产物只能证明**共享态自身**采得好不好，**不能**作为上游窗口放行的依据。"
            "只保存了两窗共享的那一个 λ、没有把下游帧在上游其余 λ 上的交叉能量落盘 ⟹ "
            "下游的强支撑**进不了上游的 local MBAR**、改善不了上游的相对自由能。"
            "所以「上游端点弱、下游强」**不构成**上游可以 PASS。"
            "要真正兜住需二选一：给下游帧补齐上游各态的 reduced potentials；"
            "或保存相邻窗口 union 的交叉能量做多系综 MBAR。在那之前上游保持 "
            "PENDING / INSUFFICIENT，不许提前宣布健康。"
        ),
        "may_be_used_to_pass_upstream": False,
        "join_lambda_vdw": join_lambda,
        "upstream_window": int(upstream_idx),
        "downstream_window": int(downstream_idx),
        "upstream_state_index": len(lam_up) - 1,
        "downstream_state_index": 0,
        "upstream": side_up,
        "downstream": side_dn,
        "downstream_over_upstream_raw_ess": ratio,
        "asymmetry_direction": (
            None if ratio is None
            else ("downstream_better" if ratio > 1.0 else "upstream_better")
        ),
    }


def join_support_path(output_dir: str, stage_type: str, up: int, down: int) -> str:
    return os.path.join(
        output_dir, f"dual_join_{int(up)}_{int(down)}_{stage_type}_support.json"
    )


DECORRELATION_PROVENANCE_VERSION = 1


def _array_content_fingerprint(arr) -> Optional[Dict[str, Any]]:
    """数组**内容**的指纹（shape + sha256 + 有限值统计量）。

    ⚠️⚠️ **仅用于 DECORR-01 的对照诊断，不进任何缓存身份 / resume 指纹。**
    本仓库已经因为"拿自产数组的 sha256 当缓存键"复发过 4 次（code_sha256、
    system_xml/positions、preopt_cache_sha256……）：数组被**合法**重算一次
    （改坐标、换盒、重解 f_k）哈希就变，缓存全失配、5M 步 GPU 白烧。规则是
    "只有用户输入才配做身份"。这里回答的是完全另一个问题 ——"两侧喂进去的是
    不是同一批数" —— 只进诊断产物，不参与任何 resume/缓存判定。

    sha256 与统计量一起给：sha256 判"逐比特相同"，统计量在**不同**时告诉你差在
    哪（形状不同 / 整体偏移 / 只差几帧），否则只能看到两串不相等的十六进制、
    还是不知道该往哪查。
    """
    if arr is None:
        return None
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    fin = a[np.isfinite(a)]
    return {
        "shape": [int(x) for x in a.shape],
        "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
        "n_nonfinite": int(a.size - fin.size),
        "min": (None if fin.size == 0 else round(float(fin.min()), 6)),
        "max": (None if fin.size == 0 else round(float(fin.max()), 6)),
        "mean": (None if fin.size == 0 else round(float(fin.mean()), 6)),
    }


def decorrelation_provenance(
    *, caller: str, energy_array_source: str, observable: str,
    n_frames_in: int, segments, kt: float, g: float, worst_state,
    subsample_indices, window_idx,
    energy_array=None, bias_array=None,
) -> Dict[str, Any]:
    """**同一个去相关计算的输入身份 + 输出**，两个调用点各存一份。

    [DECORR-01 裁决 4，2026-09-14] 逐窗自检与全局求解器对"去相关帧数"实测差 2–8 倍
    （rep2 win3：自检 56 帧 `sufficient=True`，求解器 9/7 帧并跳窗）。
    **差异来源尚未证明**，所以：

      · 两侧**分别记录**，不许改任一侧的数去追平另一侧；
      · 记的是**输入构造**（哪个数组、什么观测量、怎么分段、kT、帧数）加上
        **实际抽样索引** —— 去相关抽样本来就依赖输入时间序列与所选 g，
        拿另一份输出的帧数反推它是错的（pymbar timeseries 的语义）；
      · 有了这两份才能先比输入、再在**完全相同的输入**上重放。
    """
    idx = np.asarray(subsample_indices).ravel() if subsample_indices is not None else None
    segs = [
        {k: seg.get(k) for k in ("start_frame", "end_frame", "n_frames", "session_id")}
        for seg in (segments or [])
    ]
    return {
        "provenance_version": DECORRELATION_PROVENANCE_VERSION,
        "caller": caller,
        "window_idx": (None if window_idx is None else int(window_idx)),
        # ---- 输入构造（先比这些）----
        "energy_array_source": energy_array_source,
        # [CTL-09] 上一行只是**人写的来源名称**（自检写 `energies.npy`、求解器写
        # `u_kn`），名字不同却完全可能是同一批数。真正能回答"两侧是不是同一份
        # 输入"的只有内容指纹，所以它必须跟着 provenance 一起落盘。
        # ⚠️ 见 `_array_content_fingerprint`：**仅用于对照诊断，不进任何缓存身份**。
        "energy_content_fingerprint": _array_content_fingerprint(energy_array),
        "bias_content_fingerprint": _array_content_fingerprint(bias_array),
        "observable": observable,
        "n_frames_in": int(n_frames_in),
        "segments": segs,
        "n_segments": len(segs),
        "kt_kJ_mol": float(kt),
        # ---- 输出 ----
        "statistical_inefficiency": (None if g is None else float(g)),
        "worst_state": (None if worst_state is None else int(worst_state)),
        "n_decorrelated": (None if idx is None else int(idx.size)),
        # 实际抽样索引：首尾 + 步长足够复现判断，全量太大不落盘。
        "subsample_indices_head": (None if idx is None else
                                   [int(x) for x in idx[:16]]),
        "subsample_indices_tail": (None if idx is None else
                                   [int(x) for x in idx[-16:]]),
    }


# [CTL-09] **判定"输入是否相同"只看这一组。** `energy_array_source` 被**移出**
# 判据：它是人写的来源名称，自检写 `energies.npy`、求解器写 `u_kn`，两边名字
# 天生不同却可能是同一批数 —— 拿它当判据 ⟹ 对照永远停在"输入不同"，
# `replay_decorrelation` 那一步一次都跑不到，DECORR-01 的 2–8 倍差异也就永远
# 查不下去。反方向同样致命：名称相同、内容不同（换了段 / 换了帧集）会被判成
# "输入相同"然后去重放，造出第三个互相不可比的数。
DECORRELATION_DECISIVE_INPUT_FIELDS = (
    "energy_content_fingerprint", "bias_content_fingerprint",
    "observable", "n_frames_in", "n_segments", "segments", "kt_kJ_mol",
)
# 只作为**溯源线索**一起列出来（差异照报），不参与 `inputs_identical` 的判定。
DECORRELATION_TRACE_INPUT_FIELDS = ("energy_array_source",)
DECORRELATION_INPUT_FIELDS = (
    DECORRELATION_TRACE_INPUT_FIELDS + DECORRELATION_DECISIVE_INPUT_FIELDS
)


def compare_decorrelation_provenance(a, b) -> Dict[str, Any]:
    """**先比较输入构造**，再看输出差多少。纯函数。

    输入不同 ⟹ 两侧的数字本来就不该一样，**先修输入的口径，不许改数字去追平**。
    输入完全相同却输出不同 ⟹ 才轮到"重放去相关函数"这一步。
    """
    a, b = (a or {}), (b or {})
    input_diffs = {
        f: {"a": a.get(f), "b": b.get(f)}
        for f in DECORRELATION_INPUT_FIELDS
        if a.get(f) != b.get(f)
    }
    decisive_diffs = {
        f: v for f, v in input_diffs.items()
        if f in DECORRELATION_DECISIVE_INPUT_FIELDS
    }
    # 两侧都拿不到内容指纹 = **没有证据**，不等于"证明了相同"。仍按其余可判字段
    # 给结论（否则历史 provenance 全变成"输入不同"，对照直接废掉），但把这件事
    # 显式标出来 —— 下游不许把"没测"读成"测过且一致"。
    content_available = bool(
        a.get("energy_content_fingerprint") and b.get("energy_content_fingerprint")
    )
    return {
        "inputs_identical": not decisive_diffs,
        "content_fingerprints_available": content_available,
        "input_differences": input_diffs,
        "decisive_input_differences": decisive_diffs,
        # 名称不同、内容相同 ⟹ 判"输入相同"，这些差异只留作溯源线索。
        "name_only_differences": {
            f: v for f, v in input_diffs.items()
            if f in DECORRELATION_TRACE_INPUT_FIELDS
        },
        "next_step": (
            "输入一致 ⟹ 在**完全相同的输入**上重放 `_decorrelate_by_worst_target_state`"
            if not decisive_diffs else
            "输入不同 ⟹ 差异先归到输入构造上；**不得**改任一侧的数字去追平另一侧"
        ),
        "output_delta": {
            "statistical_inefficiency": (a.get("statistical_inefficiency"),
                                         b.get("statistical_inefficiency")),
            "n_decorrelated": (a.get("n_decorrelated"), b.get("n_decorrelated")),
            "worst_state": (a.get("worst_state"), b.get("worst_state")),
        },
    }


def replay_decorrelation(u_kj_raw, bias_kj, kt, segments=None) -> Dict[str, Any]:
    """在给定输入上**重放一次**去相关，返回与 provenance 同构的输出。

    这是对照实验的第二步：只有 `compare_decorrelation_provenance` 判定输入一致时
    才有意义 —— 输入不同就重放，等于又造了第三个不可比的数。
    """
    g_out: List[float] = []
    idx, g, worst = _decorrelate_by_worst_target_state(
        np.asarray(u_kj_raw, dtype=np.float64),
        np.asarray(bias_kj, dtype=np.float64).ravel(),
        float(kt), segments=segments, per_state_g_out=g_out,
    )
    return {
        "statistical_inefficiency": float(g),
        "worst_state": int(worst),
        "n_decorrelated": int(np.asarray(idx).size),
        "per_state_g": [float(x) for x in g_out],
    }


def _decorrelate_by_worst_target_state(
    u_kj_raw: np.ndarray,
    bias_kj: np.ndarray,
    kt: float,
    segments: Optional[List[Dict[str, Any]]] = None,
    segment_diagnostics: Optional[List[Dict[str, Any]]] = None,
    per_state_g_out: Optional[List[float]] = None,
) -> Tuple[np.ndarray, float, int]:
    """Subsample using the slowest-decorrelating *reweighting* series.

    🔑 [ESS_GATE_PROTOCOL_VERSION=2] The previous series was the total sampled
    potential ``base_kj + bias_kj``. That is dominated by bulk solvent
    fluctuation, which has nothing to do with how fast the importance weights
    decorrelate, and it measurably *underestimates* g here: on the live
    Atenolol vdw stage it gave g=1.76/2.58/2.44 for windows 0/2/5 where the
    per-target-state reduced-potential difference gives g=19.6/26.5/7.7. An
    understated g inflates n_k fed to MBAR and shrinks the reported error bars
    -- the exact failure ``subsample_series_by_autocorrelation`` exists to
    prevent.

    The right series is the one whose exponential *is* the weight,
    ``Delta_u_k = (U'_k - V_bias)/kT`` (``base`` cancels). One g per target
    state, then the most conservative (largest g) state drives the shared
    subsampling, so no state is left correlated. Mirrors the existing
    per-state pattern at the offline ``u_kn[k, start:end]`` call site.

    Returns ``(indices, g, worst_state_index)``.

    ``per_state_g_out``（可选）被就地填成**逐目标态**的 g 向量。函数内部本来就
    逐态算了 ``g_k``、只留最大的那个；把整条剖面交出来才能区分"远端单调衰减"
    （窗口跨度太大）与"整体偏低"（采样不够）—— 只看一个整窗 g 会把这两种
    完全不同的失效混成一个数。用 out 参数而不是加第 4 个返回值，是为了不动
    既有 10 余处 3 元组解包的调用点。
    """
    u_kj_raw = np.asarray(u_kj_raw, dtype=np.float64)
    bias_kj = np.asarray(bias_kj, dtype=np.float64).ravel()
    if segments is not None:
        _validate_production_segments(segments, u_kj_raw.shape[1])
        indices, g_values, worst_states = [], [], []
        for segment in segments:
            start, end = segment["start_frame"], segment["end_frame"]
            # Short restart fragments cannot establish their autocorrelation.
            if end - start < 20:
                if segment_diagnostics is not None:
                    segment_diagnostics.append(dict(segment, n_decorrelated=0,
                        statistical_inefficiency=None, reason_excluded="too_short"))
                continue
            _seg_g: List[float] = []
            idx, g, worst = _decorrelate_by_worst_target_state(
                u_kj_raw[:, start:end], bias_kj[start:end], kt,
                per_state_g_out=_seg_g if per_state_g_out is not None else None,
            )
            if per_state_g_out is not None and _seg_g:
                # 逐态取各段的最大 g —— 跟"最保守的那个态驱动共享抽稀"同一口径。
                if not per_state_g_out:
                    per_state_g_out.extend(_seg_g)
                else:
                    for _i, _v in enumerate(_seg_g[:len(per_state_g_out)]):
                        per_state_g_out[_i] = max(per_state_g_out[_i], _v)
            if segment_diagnostics is not None:
                segment_diagnostics.append(dict(segment, n_decorrelated=int(idx.size),
                    statistical_inefficiency=float(g), worst_target_state=int(worst)))
            indices.append(idx + start)
            g_values.append(g)
            worst_states.append(worst)
        if not indices:
            return np.array([], dtype=int), float("inf"), -1
        worst = int(np.argmax(g_values))
        return np.concatenate(indices), float(g_values[worst]), worst_states[worst]
    best_idx = np.arange(u_kj_raw.shape[1])
    best_g = 1.0
    worst_k = -1
    for k in range(u_kj_raw.shape[0]):
        series = (u_kj_raw[k] - bias_kj) / float(kt)
        idx_k, g_k = subsample_series_by_autocorrelation(series)
        if per_state_g_out is not None:
            per_state_g_out.append(float(g_k))
        if float(g_k) > best_g:
            best_g = float(g_k)
            best_idx = np.asarray(idx_k, dtype=int)
            worst_k = int(k)
    return best_idx, best_g, worst_k


def recalibrate_f_k_from_production(
    u_kj_raw: np.ndarray,
    bias_kj: np.ndarray,
    base_kj: np.ndarray,
    win_lams: List[int],
    kt: float,
    f_k_current: np.ndarray,
    *,
    w_idx: int = 0,
    min_frames: int = IBS_LOCAL_MBAR_GATE_MIN_FRAMES,
    sampling_kj: Optional[np.ndarray] = None,
    max_consistency_sd_kJ_mol: float = 1.0e-3,
) -> Dict[str, Any]:
    """拿**已落盘的生产帧**重解这个窗口的 f_k，供"重置偏置后再采一段"使用。

    为什么需要它：warmup 判 f_k 用的是 loose gate —— 只要求
    ``max_k|Δf_k − ΔF^MBAR| < 10 kJ/mol ≈ 4 kT``，而且只看 200 帧
    （5 批 × IBS_TMBAR_LEARNING_MINIBATCH_FRAMES）。差几 kJ/mol 完全合法，却足以
    让混合系综塌向少数态：实测 window 3 的 ``top1%_raw_weight`` 到 0.38–0.49，
    单帧扛掉四成权重。在这种 f_k 下继续加帧只是往同一个偏斜分布里加更多帧 ——
    绝对样本数涨、ESS **比值**不动，正是两轮 rescue 实测到的形状。

    生产帧比 warmup 那 200 帧多几个数量级，用它重解出来的 f_k 准得多。

    **目标 f_k 就是各态自由能本身**：混合权重 ∝ exp(−(F_k − f_k)/kT)，占据平坦
    ⟺ f_k = F_k + 常数。所以直接取局部 MBAR 解出的 F，再按跟
    `_apply_pairwise_cap` 同一个规范（均值归零）定规范。

    这里**刻意不加步长上限**：`IBS_MAX_APPLIED_PAIRWISE_STEP_KT` 那类 cap 是给
    在线学习用的（防止低重叠 MBAR 的一大步打崩占据），而这里是拿远多于 warmup
    的数据做一次性重标定，之后要重新平衡再采样，不存在"打崩当前轨迹"的问题。

    ⚠️ 用它换来的 f_k 采出的帧，跟原来那段**不是同一个采样分布**。两段要么各自
    独立分析，要么合并时必须把交叉能量补齐（用 u_kn 对两个 f_k 各算一次
    mixture 能量）——直接把两段记录的 `bias_energies` 摞起来当一个采样态是错的，
    合成实验实测偏差 +0.12 kJ/mol、RMSE 2.6 倍。
    """
    f_current = np.asarray(f_k_current, dtype=np.float64).ravel()

    # 🔑 规范自洽核对，**fail-closed**。落盘的 bias_energies 是 OpenMM 逐帧读出的
    # 真实 Group-1 能量，所以下面这条恒等式必须成立到机器精度：
    #     bias(x) == -kT * logsumexp_k(-(u_k(x) - f_k)/kT)
    #
    # ⚠️ 这里的 u 必须是**采样规范**的能量（sampling_states），不是 energies。两者相差
    # 一个**逐 λ 态的常数**（LJ 长程尾项：它在分析侧的目标能量里，不在采样哈密顿量里）。
    # 逐态常数不是共模，会改变 logsumexp 的形状。真实产物实测（cyclod rep1，4 个窗口）：
    #     用 sampling_states → sd = 0.0000（逐比特成立）
    #     用 energies       → sd = 0.7351 / 0.3734 / 0.1816 / 0.0941（随解耦单调变小，
    #                              那正是尾项随配体消失而缩小）
    # 而 `_load_ibs_window_outputs_from_dir` 只在 residual 臂返回 sampling_state_energies，
    # baseline 臂拿不到 —— 于是很容易在错误规范下解出一个"看起来正常"的 f_k。
    # 所以这条不再只是诊断：对不上就拒绝返回 f_k。宁可不重标定，也不拿一个错规范下解出
    # 的偏置去驱动几十万步采样。
    _u_for_bias = np.asarray(
        sampling_kj if sampling_kj is not None else u_kj_raw, dtype=np.float64
    )
    consistency_sd = None
    try:
        if _u_for_bias.ndim == 2 and _u_for_bias.shape[0] == f_current.size:
            recon = -float(kt) * _logsumexp_rows(
                -(_u_for_bias - f_current[:, None]) / float(kt), axis=0
            )
            resid = np.asarray(bias_kj, dtype=np.float64).ravel() - recon
            if resid.size:
                # 公共常数不重要（规范自由度），只看逐帧离散度。
                consistency_sd = float(np.std(resid - float(np.mean(resid))))
    except Exception:
        consistency_sd = None

    if consistency_sd is None or not np.isfinite(consistency_sd) or (
        consistency_sd > float(max_consistency_sd_kJ_mol)
    ):
        return {
            "error": "f_k_bias_gauge_mismatch",
            "f_k": None,
            "f_k_consistency_sd_kJ_mol": consistency_sd,
            "max_consistency_sd_kJ_mol": float(max_consistency_sd_kJ_mol),
            "hint": (
                "传进来的 u 与 bias_energies 不同规范。生产侧必须传 sampling_states "
                "（采样规范），不能传 energies（含逐 λ 态的 LJ 尾项常数）。"
            ),
        }

    solved = _solve_single_window_local_mbar(
        u_kj_raw, bias_kj, base_kj, list(win_lams), float(kt),
        f_k=f_current, sampled_distribution_row=0, w_idx=int(w_idx),
        min_frames=int(min_frames), sampling_kj=sampling_kj,
    )
    if solved.get("error"):
        return {"error": solved["error"], "f_k": None}
    f_new = np.asarray(solved.get("f"), dtype=np.float64).ravel()
    if f_new.size != f_current.size or not np.all(np.isfinite(f_new)):
        return {"error": "nan_or_shape_mismatch_in_local_mbar_f", "f_k": None}

    f_new = f_new - float(np.mean(f_new))
    f_ref = f_current - float(np.mean(f_current))
    delta = f_new - f_ref
    adjacent_shift = (
        float(np.max(np.abs(np.diff(delta)))) if delta.size > 1 else 0.0
    )
    return {
        "error": None,
        "f_k": f_new,
        "f_k_previous_centered": f_ref,
        "delta_f_k_kJ_mol": delta,
        # gauge 无关的"这次重标定改动了多少"：相邻差的最大变化。
        "max_adjacent_shift_kJ_mol": adjacent_shift,
        "max_abs_shift_kJ_mol": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "n_frames_used": int(solved.get("n_frames_used", 0) or 0),
        "statistical_inefficiency": solved.get("statistical_inefficiency"),
        "min_ess_ratio": solved.get("min_ess_ratio"),
        # 传入的 f_k 与落盘 bias_energies 的逐帧自洽度（sd，kJ/mol）。接近 0 说明
        # 这份 f_k 确实是采样时生效的那份；明显非零说明它过期了，重标定的参照错了。
        "f_k_consistency_sd_kJ_mol": consistency_sd,
    }


HELDOUT_CANDIDATE_PROTOCOL_VERSION = 1


def heldout_candidate_verdict(
    sampling_kj: np.ndarray,
    bias_kj: np.ndarray,
    base_kj: np.ndarray,
    win_lams: Sequence[int],
    kt: float,
    f_current: np.ndarray,
    *,
    n_blocks: int = 4,
    train_blocks: int = 2,
    w_idx: int = 0,
) -> Dict[str, Any]:
    """候选 f_k 的**反事实验收**：在**未参与拟合**的块上，它真的改善了吗？

    🔑 [2026-09-11] 老板的规格：离线生成候选 f_k，**只有候选满足以下条件才真正重标定**
    —— 在 held-out blocks 上提高最差态的 `N_eff/g`、不显著伤害原本健康的状态、
    口径一致、**优化目标是 target support 而不是 occupancy 平坦**。

    ⚠️ **必须用连续时间块，不能随机拆帧。** 帧是时间相关的，随机拆会让 held-out
    和 train 共享同一段慢模态 ⟹ 验收虚高（这跟"短轨迹低估慢相关"是同一个陷阱）。
    前 `train_blocks` 块训练，其余块 held-out。

    ⚠️ 口径与累计残差一致：**只能是 `sampling_states`**。

    返回 verdict：
      · `ACCEPT`      held-out 上最差态 `N_eff/g` 改善，且没有健康态被显著伤害
      · `REJECT`      没改善，或伤到了原本健康的态
      · `UNMEASURED`  held-out 上算不出来（帧太少 / MBAR 解不动）—— **不是拒绝**，
                      它意味着"这份候选在现有数据上判不了"，应转 PROBE_REANCHOR_EPOCH
    """
    u = np.asarray(sampling_kj, dtype=np.float64)
    bias = np.asarray(bias_kj, dtype=np.float64).ravel()
    base = np.asarray(base_kj, dtype=np.float64).ravel()
    K, N = u.shape
    if N < n_blocks * 4 or K < 2:
        return {"verdict": "UNMEASURED", "error": "too_few_frames_or_states"}

    edges = [int(round(N * i / n_blocks)) for i in range(n_blocks + 1)]
    tr = slice(edges[0], edges[int(train_blocks)])
    ho = slice(edges[int(train_blocks)], edges[-1])
    if ho.stop - ho.start < 4:
        return {"verdict": "UNMEASURED", "error": "heldout_too_small"}

    # ---- 训练：只用前面的连续块重解候选 ----
    cand = recalibrate_f_k_from_production(
        u[:, tr], bias[tr], base[tr], list(win_lams), float(kt),
        np.asarray(f_current, dtype=np.float64), w_idx=int(w_idx),
        sampling_kj=u[:, tr],
    )
    if cand.get("error") or cand.get("f_k") is None:
        return {"verdict": "UNMEASURED", "error": cand.get("error") or "no_candidate"}
    f_new = np.asarray(cand["f_k"], dtype=np.float64).ravel()

    # 🔑🔑 [2026-09-14 / 审计 #45] **候选 f_k 必须经由重建的混合偏置进入权重。**
    # 原来写的是 `ser = (uu[k] − bb − f_k[k])/kT`：候选 f_k 只以**逐态常数**进入
    # 序列，而后面的 `exp(-(sr − sr.min()))` 与自相关抽稀对常数平移都是不变量
    # ⟹ `_support(f_current)` 与 `_support(f_new)` **逐位相同**、`improved` 恒为
    # False ⟹ 这条验收在数学上**从来不可能给出 ACCEPT**。
    # 正解：f_k 定义的是混合系综 `V = −kT·logsumexp_j(−(u_j − f_j)/kT)`，权重是
    # `w ∝ exp(−(u_k − V)/kT)`。`multi_segment_analysis.mixture_energy` 就是这个式子
    # （也是落盘 bias_energies 的定义式），两侧用同一份实现、同一批 held-out 帧。
    # 恒等性：`f_new == f_current` 时两侧仍逐位相同 ⟹ improved=False ⟹ REJECT，
    # 与改动前的退化行为一致（回归已验）。
    from multi_segment_analysis import mixture_energy as _mixture_energy  # 惰性

    def _support(f_k: np.ndarray, sl: slice) -> Optional[List[float]]:
        """held-out 块上逐态 N_eff/g。**这就是验收目标：target support，不是占据平坦。**"""
        out = []
        uu = u[:, sl]
        # 该候选 f_k 所定义的混合偏置，在 held-out 帧上重建（**不是**落盘 bias）。
        bb = _mixture_energy(uu, np.asarray(f_k, dtype=np.float64).ravel(), float(kt))
        for k in range(K):
            ser = (uu[k] - bb) / float(kt)
            fin = np.isfinite(ser)
            if not fin.any():
                return None
            sr = ser[fin]
            w = np.exp(-(sr - sr.min()))
            sw, sw2 = float(w.sum()), float((w * w).sum())
            if sw2 <= 0.0:
                return None
            ne = sw * sw / sw2
            try:
                _i, g = subsample_series_by_autocorrelation(sr)
                g = float(g)
            except Exception:
                g = None
            out.append(float(ne) / g if (g and g > 0.0) else None)
        return out

    old_s = _support(np.asarray(f_current, dtype=np.float64), ho)
    new_s = _support(f_new, ho)
    if old_s is None or new_s is None or any(x is None for x in old_s + new_s):
        return {"verdict": "UNMEASURED", "error": "heldout_support_not_computable",
                "candidate_f_k": [float(x) for x in f_new]}

    old_worst, new_worst = min(old_s), min(new_s)
    # "不显著伤害原本健康的状态"：健康 = 旧值 >= 10（协议安全下限），
    # 伤害 = 掉到下限以下。不用比例阈值 —— 那又要拍一个数。
    harmed = [
        k for k in range(K)
        if old_s[k] >= 10.0 and new_s[k] < 10.0
    ]
    improved = new_worst > old_worst
    verdict = "ACCEPT" if (improved and not harmed) else "REJECT"
    return {
        "protocol_version": HELDOUT_CANDIDATE_PROTOCOL_VERSION,
        "verdict": verdict,
        "error": None,
        "energy_gauge": "sampling_states",
        "split": "contiguous_time_blocks",
        "n_blocks": int(n_blocks), "train_blocks": int(train_blocks),
        "train_frames": [int(edges[0]), int(edges[int(train_blocks)])],
        "heldout_frames": [int(edges[int(train_blocks)]), int(edges[-1])],
        # [审计 #45] 两侧权重都用**各自 f_k 重建的混合偏置**，不是落盘 bias、
        # 也不是把 f_k 当逐态常数减掉（那样候选恒等于当前，验收永远 REJECT）。
        "support_weight_definition": (
            "w_k ∝ exp(-(u_k - V(f))/kT), V = -kT·logsumexp_j(-(u_j - f_j)/kT)"
            " —— multi_segment_analysis.mixture_energy"
        ),
        "heldout_n_eff_over_g_before": old_s,
        "heldout_n_eff_over_g_after": new_s,
        "worst_before": float(old_worst), "worst_after": float(new_worst),
        "improved_worst_state": bool(improved),
        "harmed_states": harmed,
        "candidate_f_k": [float(x) for x in f_new],
        "candidate_max_adjacent_shift_kJ_mol": cand.get("max_adjacent_shift_kJ_mol"),
        "note": (
            "验收目标是 **target support**（held-out 最差态 N_eff/g），"
            "**不是 occupancy 平坦** —— 后者正是 win1 被『用支撑换占据』的原因。"
            "UNMEASURED ≠ REJECT：判不了就转 PROBE_REANCHOR_EPOCH 拿独立证据。"
        ),
    }


CUMULATIVE_FK_RESIDUAL_PROTOCOL_VERSION = 1


def cumulative_fk_span_verdict(
    c: np.ndarray,
    ddf_physical_matrix_kJ_mol: Any,
    threshold: float = IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL,
) -> Dict[str, Any]:
    """累计 f_k 残差 `C_j` 的 span 判据 —— **全仓唯一实现**。

    🔑🔑 [2026-09-14 / 审计 #51] 这段数学原来有**两份**拷贝（`cumulative_fk_residual`
    和在线门里的内联块），它们会对同一份数据给出不同结论、又都落进报告里。
    「同一个不变量的 N 份实现」是本仓最贵的那类 bug，所以提成一个函数、两处共用。

    判的是**标准化 span**：零假设 ``r ≡ 0`` 下 ``C`` 是 K_edges 步随机游走，
    ``E[span] ≈ σ_adj·√(2·K_edges/π)`` **随边数增长**，而门槛是刻意做成 K-无关的
    ⟹ 不先除掉它，边最多的窗口（末窗是溢出槽）就被系统性误判 FAIL。
    ``σ_adj`` 取**全部相邻对** ``ddf[i, i+1]`` 的 RMS —— 不用 ``ddf[argmin, argmax]``
    那个预选对（从同一批数据里挑出来的极值对，CI 系统性偏窄）。

    ⚠️ **阈值没换，只是换了刻度**：仍是同一个数字，现在当作"span 是零假设期望的
    多少倍"的门。三档：``std + band < thr`` PASS；``std − band >= thr`` FAIL；
    否则 UNMEASURED。

    ⚠️ ``σ_adj`` 拿不到（没有 ddf / 形状不符 / 全非有限）时显式返回
    ``standardized=None`` + ``verdict="UNMEASURED"`` + ``unmeasured_reason``，
    **绝不退回旧刻度偷偷算一个数** —— 那正是两份实现分叉的来路。

    ``verdict`` 用中性的 ``"FAIL"``；调用方要 ``FAIL_CUMULATIVE_FK`` 自己映射。
    """
    c = np.asarray(c, dtype=np.float64).ravel()
    K = int(c.size)
    k_edges = int(K - 1)
    span = float(np.max(c) - np.min(c)) if K else float("nan")
    a_i, b_i = (int(np.argmax(c)), int(np.argmin(c))) if K else (-1, -1)

    m = None
    if ddf_physical_matrix_kJ_mol is not None:
        m = np.asarray(ddf_physical_matrix_kJ_mol, dtype=float)
        if m.shape != (K, K):
            m = None
    # 预选对的 σ：**只作报告**，不参与判据（保留是为了跟历史产物对账）。
    sigma = (
        float(m[b_i, a_i])
        if (m is not None and K and np.isfinite(m[b_i, a_i])) else None
    )
    sigma_adj = None
    if m is not None and k_edges >= 1:
        adj = np.array([m[i, i + 1] for i in range(k_edges)], dtype=float)
        adj = adj[np.isfinite(adj)]
        if adj.size:
            sigma_adj = float(np.sqrt(np.mean(adj * adj)))

    null_expectation = (
        None if (sigma_adj is None or not (sigma_adj > 0.0))
        else float(sigma_adj * np.sqrt(2.0 * k_edges / np.pi))
    )
    standardized = (
        None if (null_expectation is None or not (null_expectation > 0.0))
        else float(span / null_expectation)
    )
    # CI 半宽换到同一刻度，**同样用 σ_adj**（不用预选对）。
    band = (
        None if (null_expectation is None or sigma_adj is None)
        else float(2.0 * sigma_adj / null_expectation)
    )

    thr = float(threshold)
    unmeasured_reason = None
    if standardized is None or band is None:
        verdict = "UNMEASURED"
        unmeasured_reason = (
            "σ_adj 不可得（无 ddf_physical 矩阵 / 形状不符 / 相邻对全非有限），"
            "标准化 span 算不出来；**不退回旧的 kJ/mol 刻度**"
        )
    elif standardized + band < thr:
        verdict = "PASS"
    elif standardized - band >= thr:
        verdict = "FAIL"
    else:
        verdict = "UNMEASURED"
        unmeasured_reason = "CI 跨过门槛，或 MBAR 精度不足"

    return {
        "span_kJ_mol": span,
        "argmax_state": a_i,
        "argmin_state": b_i,
        "sigma_preselected_pair_kJ_mol": sigma,
        "sigma_adj_kJ_mol": sigma_adj,
        "n_edges": k_edges,
        "null_expectation_kJ_mol": null_expectation,
        "span_standardized": standardized,
        "band_standardized": band,
        "threshold": thr,
        "verdict": verdict,
        "unmeasured_reason": unmeasured_reason,
    }


def _frameset_identity(n_frames: int, segments) -> str:
    """这份残差记录是从**哪批帧**算出来的：帧数 + 分段起点。

    够用来判"两条记录是不是同一批帧"，而且**不把自产数组的 sha256 当身份**
    （那是本仓库已复发 4 次的坑）。提成函数是因为成功路径和 CTL-08 的跳窗
    UNMEASURED 记录都要写它 —— 同一个不变量写两份实现，迟早一边改一边不改，
    下游按 frameset_id 配对时就静默配错。
    """
    return "n%d_segs%s" % (
        int(n_frames),
        "-".join(str(int(x.get("start_frame", -1))) for x in (segments or []))
        or "single",
    )


def _per_segment_cumulative_fk_residual(
    sampling_kj: np.ndarray,
    bias_kj: np.ndarray,
    base_kj: np.ndarray,
    win_lams,
    kt: float,
    *,
    source_ids,
    segment_f_ks,
    provenance,
    w_idx: int = 0,
) -> Dict[str, Any]:
    """**S2-A 逐段门**：每个采样段绑到**它自己的**冻结 f_k，各算各的残差。

    [2026-09-14 裁决 3] 多段窗口的帧采自两份不同偏置，所以：
      · 帧切片由 `sampling_source_id` 给出（**有序帧映射**，不是猜边界）；
      · 每段用 `sampling_segment_f_k[i]`，**不许**拿某一段的 f_k 冒充全体帧；
      · **绝不**拿合并后的 ΔF 作逐段望远镜相消 —— 每段自己解一次 local MBAR。

    窗口级结论（老板定的三态合成）::

        任一**有效**段 FAIL            → FAIL
        否则只要有**缺证据**的段        → UNMEASURED
        全部段 PASS                    → PASS

    "有效段"= 真的解出了 verdict 的段；帧太少/解不出来的段记 `UNMEASURED` 并带上
    自己的理由。每段的身份（段号、来源目录）、帧范围和结论都留在
    `segments` 里，**可审计**。
    """
    ids = None if source_ids is None else np.asarray(source_ids).ravel()
    f_ks = list(segment_f_ks or [])
    if ids is None or not f_ks:
        return {"error": "no_effective_f_k_for_these_frames"}
    prov = list(provenance or [])
    n_frames = int(np.asarray(bias_kj).ravel().size)
    if ids.size != n_frames:
        return {"error": "sampling_source_id_length_mismatch"}

    segments: List[Dict[str, Any]] = []
    for i, f_seg in enumerate(f_ks):
        mask = ids == i
        idx = np.flatnonzero(mask)
        rec: Dict[str, Any] = {
            "segment_position": int(i),
            "segment_index": (prov[i].get("segment_index") if i < len(prov) else None),
            "source_dir": (prov[i].get("source_dir") if i < len(prov) else None),
            "n_frames": int(idx.size),
            # 帧范围按**有序**帧映射给出，便于审计（连续段就是 [first, last]）。
            "frame_range": ([int(idx[0]), int(idx[-1])] if idx.size else None),
            "f_k_fingerprint": (
                None if f_seg is None
                else frozen_candidate_fingerprint(
                    np.asarray(f_seg, dtype=np.float64).ravel())
            ),
        }
        if idx.size == 0 or f_seg is None:
            rec.update({"verdict": "UNMEASURED",
                        "reason": "segment_has_no_frames_or_no_frozen_f_k"})
            segments.append(rec)
            continue
        try:
            sub = cumulative_fk_residual(
                np.asarray(sampling_kj)[:, mask],
                np.asarray(bias_kj).ravel()[mask],
                np.asarray(base_kj).ravel()[mask],
                win_lams, kt,
                np.asarray(f_seg, dtype=np.float64).ravel(),
                w_idx=int(w_idx),
            )
        except Exception as err:  # noqa: BLE001 —— 单段解不出来不该毁掉整个诊断
            rec.update({"verdict": "UNMEASURED", "reason": repr(err)[:200]})
            segments.append(rec)
            continue
        if sub.get("error"):
            rec.update({"verdict": "UNMEASURED", "reason": sub["error"]})
        else:
            rec.update({k: sub.get(k) for k in (
                "verdict", "cumulative_residual_span_kJ_mol",
                "cumulative_residual_span_sigma_kJ_mol", "threshold_kJ_mol",
                # [审计 #51] 判据量与它的输入一并带到段记录里，否则事后只能看到
                # 一个 kJ/mol 的 span 却不知道门是按哪把尺子判的。
                "cumulative_residual_span_standardized",
                "cumulative_residual_sigma_adj_kJ_mol",
                "cumulative_residual_n_edges",
                "cumulative_residual_span_null_expectation_kJ_mol",
                "signed_cumulative_residual_from_first_kJ_mol",
                "endpoint_signed_residual_kJ_mol", "all_same_sign",
                "n_frames_used", "statistical_inefficiency",
            )})
        segments.append(rec)

    verdicts = [r.get("verdict") for r in segments]
    if "FAIL_CUMULATIVE_FK" in verdicts:
        window_verdict = "FAIL_CUMULATIVE_FK"
    elif all(v == "PASS" for v in verdicts) and verdicts:
        window_verdict = "PASS"
    else:
        window_verdict = "UNMEASURED"
    _spans = [r.get("cumulative_residual_span_kJ_mol") for r in segments
              if r.get("cumulative_residual_span_kJ_mol") is not None]
    return {
        "verdict": window_verdict,
        "scope_note": (
            "逐段门（S2-A）：每段绑自己的冻结 f_k、在自己的帧切片上独立解一次；"
            "**没有**跨段望远镜相消。窗口级结论 = 任一有效段 FAIL 则 FAIL，"
            "否则有缺证据段则 UNMEASURED，全部通过才 PASS。"
        ),
        "per_segment_gate": True,
        "n_segments": len(segments),
        "segments": segments,
        # 只做展示：**不是**把逐段 span 合成一个窗口级 span 去判门。
        "max_segment_span_kJ_mol_REPORT_ONLY": (max(_spans) if _spans else None),
    }


def cumulative_fk_residual(
    sampling_kj: np.ndarray,
    bias_kj: np.ndarray,
    base_kj: np.ndarray,
    win_lams: Sequence[int],
    kt: float,
    f_frozen: np.ndarray,
    *,
    energy_gauge: str = "sampling_states",
    threshold_kJ_mol: float = IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL,
    w_idx: int = 0,
) -> Dict[str, Any]:
    """**端到端累计 f_k 偏差**（老板规格 v1）。只算，不设门、不改任何执行。

    ``r_i = (f_{i+1} − f_i) − (F̂_{i+1} − F̂_i)``，``C_j = Σ_{i<j} r_i``。
    望远镜求和 ⟹ ``C_j = (f_j − f_0) − (F̂_j − F̂_0)``；``f`` 是**冻结**的（无误差），
    所以 ``σ(C_j) = σ(F̂_j − F̂_0) = ddf[0, j]``，**直接读同一次 MBAR 拟合的协方差
    矩阵元，不做任何平方相加** —— 相邻差高度相关，平方相加会系统性算错
    （仓库在 ``endpoint_diff_uncertainty`` 处已经记过这个坑一次）。

    门看 **span = max(C) − min(C)**：它 **gauge-free**、不依赖选哪个端点作锚，
    而**只看终点会漏掉「中间先偏后抵消」**。

    [2026-09-14 审计 #51] 判的是**标准化 span**：零假设 ``r ≡ 0`` 下
    ``E[span] ≈ σ_adj·√(2·K_edges/π)`` **随边数增长**，所以先除掉它，门槛才真的
    K-无关（阈值数字没变，只是换了刻度）。``σ_adj`` = 全部相邻对 ``ddf`` 的 RMS，
    **不用** ``ddf[argmin, argmax]``（预选对 ⟹ CI 系统性偏窄）。三档::

        span/E[span] + 2σ_adj/E[span] <  threshold   PASS
        span/E[span] − 2σ_adj/E[span] >= threshold   FAIL
        否则                                          UNMEASURED（CI 跨过门槛，
                                                     或 MBAR 精度不足）

    ⚠️ **口径只能是 sampling_states，传 energies 直接拒绝**（见
    ``assert_sampling_gauge_for_fk_residual`` 的长注释：energies 多一个逐 λ 态常数，
    其**逐边差**会凭空伪造出「全负号、单调」的残差形状，实测把某窗 span 抬高 2.2 倍）。
    """
    assert_sampling_gauge_for_fk_residual(energy_gauge)
    f_frozen = np.asarray(f_frozen, dtype=np.float64).ravel()
    K = len(list(win_lams))
    if f_frozen.size != K or K < 2:
        return {"error": f"f_k 长度 {f_frozen.size} 与窗口 {K} 态不匹配，或 K<2"}
    solved = _solve_single_window_local_mbar(
        np.asarray(sampling_kj, dtype=np.float64),
        np.asarray(bias_kj, dtype=np.float64),
        np.asarray(base_kj, dtype=np.float64),
        list(win_lams), float(kt), f_k=f_frozen, w_idx=int(w_idx),
    )
    if solved.get("error") or solved.get("f") is None:
        return {"error": solved.get("error") or "no_f"}
    f_mbar = np.asarray(solved["f"], dtype=np.float64).ravel()
    if f_mbar.size != K or not np.all(np.isfinite(f_mbar)):
        return {"error": "mbar_f_shape_or_nan"}

    # 两侧都减均值 ⟹ gauge-free（f_k 的公共常数不改变采样分布，也不该改变残差）
    f_ref = f_frozen - float(np.mean(f_frozen))
    f_hat = f_mbar - float(np.mean(f_mbar))
    r = np.diff(f_ref) - np.diff(f_hat)
    c = np.concatenate(([0.0], np.cumsum(r)))
    # [2026-09-14 审计 #51] 判据数学在 `cumulative_fk_span_verdict` 里，全仓唯一实现
    # （在线门那份内联块调的是同一个函数）。这里只负责映射 verdict 词汇。
    _sv = cumulative_fk_span_verdict(
        c, solved.get("ddf_physical_matrix_kJ_mol"), float(threshold_kJ_mol)
    )
    span = _sv["span_kJ_mol"]
    a_i, b_i = _sv["argmax_state"], _sv["argmin_state"]
    sigma = _sv["sigma_preselected_pair_kJ_mol"]
    sigma_adj = _sv["sigma_adj_kJ_mol"]
    k_edges = _sv["n_edges"]
    null_expectation = _sv["null_expectation_kJ_mol"]
    span_standardized = _sv["span_standardized"]
    thr = _sv["threshold"]
    verdict = (
        "FAIL_CUMULATIVE_FK" if _sv["verdict"] == "FAIL" else _sv["verdict"]
    )

    return {
        "protocol_version": CUMULATIVE_FK_RESIDUAL_PROTOCOL_VERSION,
        "error": None,
        "energy_gauge": energy_gauge,
        "signed_adjacent_residual_kJ_mol": [float(x) for x in r],
        "signed_cumulative_residual_from_first_kJ_mol": [float(x) for x in c],
        "cumulative_residual_span_kJ_mol": span,
        # 预选对 ddf[argmin, argmax]：**只作报告**，不再参与判据（CI 系统性偏窄）。
        "cumulative_residual_span_sigma_kJ_mol": sigma,
        "cumulative_residual_span_argmax_state": a_i,
        "cumulative_residual_span_argmin_state": b_i,
        # [审计 #51] 判据量与它的三个输入，四个量一起落盘便于事后换刻度对账。
        "cumulative_residual_span_standardized": span_standardized,
        "cumulative_residual_sigma_adj_kJ_mol": sigma_adj,
        "cumulative_residual_n_edges": k_edges,
        "cumulative_residual_span_null_expectation_kJ_mol": null_expectation,
        "cumulative_residual_span_band_standardized": _sv["band_standardized"],
        "cumulative_residual_unmeasured_reason": _sv["unmeasured_reason"],
        "endpoint_signed_residual_kJ_mol": float(c[-1]),
        "all_same_sign": bool(np.all(r > 0) or np.all(r < 0)) if r.size else None,
        "abs_sum_kJ_mol": float(np.sum(np.abs(r))),
        "verdict": verdict,
        "threshold_kJ_mol": thr,
        "threshold_note": (
            "沿用现有 loose tolerance 那个**同一个数**，但 [2026-09-14 审计 #51] 起"
            "它卡的是**标准化 span**（span / 零假设期望），不是 kJ/mol 的 span —— "
            "同一个门换了刻度，不是新阈值。这样门本身天然 K-无关，"
            "而原始 span 的零假设期望随边数增长这件事被放进了分母。"
        ),
        "threshold_applies_to": "cumulative_residual_span_standardized",
        "sigma_method": (
            "σ_adj = **全部相邻对** ddf_physical[i, i+1] 的 RMS（不用 argmin/argmax "
            "那个预选对 —— 从同一批数据里挑出来的极值对，CI 系统性偏窄）；"
            "零假设期望 E[span] ≈ σ_adj·√(2·K_edges/π)。"
            "通用 max(C)−min(C) 的 block bootstrap CI 仍未实现。"
        ),
        "n_frames_used": solved.get("n_frames_used"),
        "statistical_inefficiency": solved.get("statistical_inefficiency"),
    }


def assert_sampling_gauge_for_fk_residual(gauge: str) -> None:
    """累计 f_k 残差的口径守卫：**不是 sampling_states 就直接拒绝**（fail-closed）。

    🔑 [2026-09-11] 老板要求：「若输入是 `energies`，累计 f_k gate **直接拒绝**」。

    `energies` 比 `sampling_states` 多一个**逐 λ 态常数**（LRC 长程尾项；实测
    两者之差的 std **精确为 0**）。逐态常数本身无害，但它的**逐边差是正的、
    单调递减** ⟹ 会凭空伪造出「全负号、单调」的残差形状。实测污染：win0 的
    cumulative span 被抬高 **2.2 倍**（3.27 → 7.19），而 win3 几乎没变
    （11.47 → 11.92）—— 于是还会伪造出「span 越大末/首支撑比越小」的假单调关系。

    ⚠️ 与支撑量的区别要记清：**`N_eff/g` 不受影响**（逐态常数在归一化里被除掉，
    两口径**逐位相同**）。**只有 f_k 残差这一类会被污染。**
    """
    if str(gauge) != "sampling_states":
        raise ValueError(
            f"累计 f_k 残差的口径必须是 sampling_states，收到 {gauge!r}。"
            "energies 多一个逐 λ 态常数（LRC），其逐边差会伪造出单调同号的残差形状"
            "（实测 win0 的 span 被抬高 2.2 倍）。fail-closed，拒绝计算。"
        )


def _solve_single_window_local_mbar(
    u_kj_raw: np.ndarray,
    bias_kj: np.ndarray,
    base_kj: np.ndarray,
    win_lams: List[int],
    kt: float,
    f_k: Optional[np.ndarray] = None,
    sampled_distribution_row: int = 0,
    w_idx: int = 0,
    min_frames: int = IBS_LOCAL_MBAR_GATE_MIN_FRAMES,
    production_segments: Optional[List[Dict[str, Any]]] = None,
    sampling_kj: Optional[np.ndarray] = None,
    sampling_gauge_required: bool = False,
) -> Dict[str, Any]:
    """Solve one IBS window's local augmented-MBAR problem from raw per-frame
    data (autocorrelation subsampling -> global energy offset -> augmented
    sampled+physical-state matrix -> kT reduction -> column-wise numerical
    stabilization -> MBAR -> effective-sample-number overlap diagnostic).

    This mirrors the per-window construction inside
    ``GlobalMBARAnalyzer.solve_stage_integrated`` (deliberately a separate,
    independent implementation rather than a refactor-and-share: that
    end-of-stage solver has been through multiple audit rounds and touching
    it to extract a shared helper would risk regressing already-verified
    behavior for a lower-value DRY win). Used by the production-time online
    early-stop monitor in ``run_all_windows`` to evaluate a window's
    convergence from its in-progress (not yet finished) sample history,
    without waiting for the stage to complete. If the two implementations
    ever need to diverge (e.g. the stage-level solver gains a correction this
    one doesn't need, or vice versa), that's fine -- they answer different
    questions (final stage assembly vs. "should we stop sampling now") even
    though today they share the same math.

    Returns a dict with ``error`` set if this window's data isn't solvable
    yet (too few frames, pymbar missing, MBAR failure), otherwise
    ``lambdas``/``f`` (kJ/mol, relative to sampled reference)/``df`` (kJ/mol
    uncertainty)/``n_frames_used``/``global_offset``/``min_ess_ratio``/
    ``ess_ratio_per_lambda``/``statistical_inefficiency``.

    ``sampling_kj`` / ``sampling_gauge_required``: forwarded verbatim to
    ``_ibs_reweighting_quality_diagnostics`` (see its docstring) after being put
    through the *same* decorrelation subsampling and finite-column mask as
    ``u_kj_raw``, so the mixture gate never mixes two different frame subsets.
    """
    u_kj_raw = np.asarray(u_kj_raw, dtype=np.float64)
    bias_kj = np.asarray(bias_kj, dtype=np.float64)
    base_kj = np.asarray(base_kj, dtype=np.float64)
    if sampling_kj is not None:
        sampling_kj = np.asarray(sampling_kj, dtype=np.float64)
        if sampling_kj.shape != u_kj_raw.shape:
            return {
                "error": (
                    f"窗口 {w_idx} sampling_state_energies 形状 {sampling_kj.shape} "
                    f"与物理 u_kn {u_kj_raw.shape} 不一致；拒绝用错位数据算混合覆盖度门"
                )
            }
    if u_kj_raw.ndim != 2 or len(win_lams) != u_kj_raw.shape[0]:
        return {"error": f"窗口 {w_idx} 数据维度不匹配: u_kn={u_kj_raw.shape}, lambdas={len(win_lams)}"}

    n_frames = u_kj_raw.shape[1]
    if bias_kj.ndim != 1 or base_kj.ndim != 1:
        return {
            "error": (
                f"窗口 {w_idx} bias/base 必须是一维数组: "
                f"bias={bias_kj.shape}, base={base_kj.shape}"
            )
        }
    if len(bias_kj) != n_frames or len(base_kj) != n_frames:
        return {
            "error": (
                f"窗口 {w_idx} energies/bias/base 帧数不一致: "
                f"energies={n_frames}, bias={len(bias_kj)}, base={len(base_kj)}"
            )
        }
    if (
        not np.all(np.isfinite(u_kj_raw))
        or not np.all(np.isfinite(bias_kj))
        or not np.all(np.isfinite(base_kj))
    ):
        return {"error": f"窗口 {w_idx} energies/bias/base 含 NaN/Inf"}
    if n_frames < min_frames:
        return {"error": "insufficient_frames", "n_frames": int(n_frames)}

    # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] 去相关序列改为"权重本身的指数"，见
    # _decorrelate_by_worst_target_state 的 docstring（旧的 base+bias 会低估 g）。
    #
    # 🔑 [ESS_GATE_PROTOCOL_VERSION=5 / 2026-09-01] 与 solve_stage_integrated 同一处
    # 循环论证修复。本函数比那边更要紧：它同时驱动
    #   · `:14721` 的**冻结验证 loose gate**（VALIDATE，唯一的生产放行证明）
    #   · `:16450` 的在线 early-stop
    # 所以这里的 raw 诊断被抬高，等于让"能不能进生产"这个判断建在被权重自己
    # 抽稀过的帧集上。
    #
    # 4W53 甲苯溶剂腿 window 0 就是这样进生产的：loose gate 报
    # `validation_sample_count=200`、`statistical_inefficiency=2.63`、
    # `n_frames_used=76`（=200/2.63）、`min_absolute_ess=55.75`、
    # `min_ess_ratio=0.734` —— 一片健康；而同一个窗口的生产重加权在未抽稀的
    # 500 帧上 raw ESS 只有 3.58、top-1% 权重 0.601。
    # 见 docs/archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md §2.5/§2.6。
    _u_kj_pre_decorr = u_kj_raw
    _bias_kj_pre_decorr = bias_kj
    _sampling_kj_pre_decorr = sampling_kj

    sub_idx, g_val, g_worst_state = _decorrelate_by_worst_target_state(
        u_kj_raw, bias_kj, kt, segments=production_segments
    )
    if sub_idx.size < n_frames:
        u_kj_raw = u_kj_raw[:, sub_idx]
        bias_kj = bias_kj[sub_idx]
        base_kj = base_kj[sub_idx]
        if sampling_kj is not None:
            sampling_kj = sampling_kj[:, sub_idx]
        n_frames = int(sub_idx.size)
    if n_frames < min_frames:
        return {
            "error": "insufficient_frames_after_decorrelation",
            "n_frames": int(n_frames),
            "statistical_inefficiency": float(g_val),
        }

    u_phys_kj = base_kj[None, :] + u_kj_raw
    mean_energies = np.mean(u_phys_kj, axis=1)
    global_offset = float(np.min(mean_energies))
    u_kj_shifted = u_phys_kj - global_offset
    u_sampled_eff = base_kj + bias_kj - global_offset
    u_mbar = np.vstack([u_sampled_eff, u_kj_shifted])
    u_mbar = u_mbar * (1.0 / float(kt))

    # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] Row 0 是"实际 IBS 采样分布"、row 1..K 是
    # 物理目标态，这个布局是 u_mbar 上面那几行硬编码出来的。旧代码允许把样本数放
    # 到任意 sampled_distribution_row，然后又用 `i != sampled_row` 挑目标行、再和
    # win_lams 直接 zip——sampled_row != 0 时 (a) MBAR 会被告知样本来自某个*物理*
    # 态（物理错误，不只是索引错位），(b) ESS→λ 映射整体错位，win_lams[0] 会被配到
    # 采样分布那一行的 ESS。这个不变量原先只写在下面的注释里、没有 enforce，而对
    # "非法值"的容错回退（越界→0）恰好掩盖了真正危险的"合法但非零"值。改为 fail
    # closed。
    sampled_row = int(sampled_distribution_row)
    if sampled_row != 0:
        return {
            "error": (
                f"窗口 {w_idx} sampled_distribution_row={sampled_row} != 0；"
                "本函数的增广矩阵 row 0 固定是 IBS 采样分布、row 1..K 是物理态，"
                "非零采样行会同时破坏 MBAR 的采样态声明与 ESS→λ 映射，拒绝继续。"
            )
        }
    n_k_local = np.zeros(len(win_lams) + 1, dtype=np.int32)
    n_k_local[sampled_row] = n_frames

    u_min_col = np.min(u_mbar, axis=0, keepdims=True)
    u_mbar_stable = u_mbar - u_min_col
    valid_mask = np.isfinite(u_mbar_stable).all(axis=0)
    u_mbar_final = u_mbar_stable[:, valid_mask]
    n_k_local[sampled_row] = int(np.sum(valid_mask))
    if n_k_local[sampled_row] < min_frames:
        return {"error": "insufficient_valid_frames", "n_frames": int(n_k_local[sampled_row])}

    if not has_pymbar():
        return {"error": "pymbar_not_installed"}

    try:
        mbar = _build_mbar_compatible(
            u_mbar_final, n_k_local, relative_tolerance=1e-7,
            initialize="BAR", solver_protocol="default",
        )
        res = _compute_free_energy_result_compatible(mbar, compute_uncertainty=True)
        df_matrix, ddf_matrix = _extract_free_energy_arrays(res, require_uncertainty=True)
    except Exception as e:
        return {"error": f"mbar_solve_failed: {e}"}

    # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] min_ess_ratio 从"单参考 raw ESS"换成
    # "扣掉共模因子后的混合覆盖度 ESS(p_k)"，raw 量降级为只报告不设门。
    # 详见 _ibs_reweighting_quality_diagnostics 顶部的长注释。
    # 必须用 valid_mask 之后的那批帧：denom 是 n_k_local[sampled_row]=sum(valid_mask)，
    # 若 quality 用未过滤的全部帧，比例的分子分母就来自不同样本集。
    quality = _ibs_reweighting_quality_diagnostics(
        u_kj_raw[:, valid_mask],
        bias_kj[valid_mask],
        f_k,
        kt,
        # 🔑 [0831issue] residual 臂的 f_k 是在 sampling-state 能量上学的，混合
        # 覆盖度必须用同一口径，否则门算的既不是采样混合分布也不是物理混合分布。
        sampling_kj=(
            None if sampling_kj is None else sampling_kj[:, valid_mask]
        ),
        sampling_gauge_required=bool(sampling_gauge_required),
    )
    denom = max(int(n_k_local[sampled_row]), 1)
    min_ess_ratio = None
    ess_ratio_per_lambda = None
    # ---- [ESS_GATE_PROTOCOL_VERSION=5] raw 权重退化诊断算在**去相关之前**的帧集上 ----
    # 与 solve_stage_integrated 完全同一处理；旧口径保留为 `*_post_decorrelation`。
    raw_min_ess_ratio = None
    raw_ess_ratio_per_lambda = None
    raw_min_absolute_ess = None
    raw_gate_n_frames_pre_decorrelation = None
    top1pct_raw_weight_gated = None
    _pre_finite = (
        np.isfinite(_u_kj_pre_decorr).all(axis=0) & np.isfinite(_bias_kj_pre_decorr)
    )
    quality_pre = _ibs_reweighting_quality_diagnostics(
        _u_kj_pre_decorr[:, _pre_finite],
        _bias_kj_pre_decorr[_pre_finite],
        f_k,
        kt,
        sampling_kj=(
            None if _sampling_kj_pre_decorr is None
            else _sampling_kj_pre_decorr[:, _pre_finite]
        ),
        sampling_gauge_required=bool(sampling_gauge_required),
    )
    _pre_denom = max(int(np.count_nonzero(_pre_finite)), 1)
    raw_gate_n_frames_pre_decorrelation = int(_pre_denom)
    if quality_pre.get("raw_ess") is not None:
        _raw_pre = np.asarray(quality_pre["raw_ess"], dtype=float)
        raw_min_absolute_ess = float(np.min(_raw_pre))
        raw_min_ess_ratio = float(np.min(_raw_pre / _pre_denom))
        raw_ess_ratio_per_lambda = {
            int(lam): float(r) for lam, r in zip(win_lams, _raw_pre / _pre_denom)
        }
    top1pct_raw_weight_gated = quality_pre.get("top1pct_raw_weight")

    raw_min_ess_ratio_post_decorrelation = None
    raw_ess_ratio_per_lambda_post_decorrelation = None
    if quality.get("raw_ess") is not None:
        raw_ratio = np.asarray(quality["raw_ess"], dtype=float) / denom
        raw_min_ess_ratio_post_decorrelation = float(np.min(raw_ratio))
        raw_ess_ratio_per_lambda_post_decorrelation = {
            int(lam): float(r) for lam, r in zip(win_lams, raw_ratio)
        }
    min_occupancy_normalized = None
    if quality.get("mixture_ess") is not None:
        mix_ratio = np.asarray(quality["mixture_ess"], dtype=float) / denom
        min_ess_ratio = float(np.min(mix_ratio))
        ess_ratio_per_lambda = {
            int(lam): float(r) for lam, r in zip(win_lams, mix_ratio)
        }
        # 一阶矩伴随量，见 _ibs_reweighting_quality_diagnostics 里的说明：ESS 尺度
        # 不变，抓不到"均匀被饿死"的态，两项必须一起看。
        occ_norm = quality.get("mixture_occupancy_normalized")
        if occ_norm:
            min_occupancy_normalized = float(np.min(occ_norm))

    f_phys_kt = df_matrix[0, 1:]
    df_phys_kt = ddf_matrix[0, 1:]
    f_phys_kj = (f_phys_kt * float(kt)).astype(float)
    df_phys_kj = (df_phys_kt * float(kt)).astype(float)

    # 🔑 端点（第一个 vs 最后一个物理态）自由能差的不确定度，必须直接读
    # ddf_matrix 里这两个物理态之间的元素，而不是把各自相对采样态的
    # 边际不确定度 df_phys_kj[0]/df_phys_kj[-1] 用 sqrt(a^2+b^2) 合并——
    # 两者都是同一次 MBAR 拟合、同一批样本估计出来的，彼此有协方差，
    # 把它们当独立量平方相加会系统性算错端点差的不确定度（可能偏大也可能
    # 偏小，取决于协方差符号），进而让在线 early-stop 判据错误地提前/永远
    # 不停止。物理态在增广矩阵里占据第 1..len(win_lams) 行（第 0 行是采样
    # 分布），第一个物理态是第 1 行，最后一个是第 len(win_lams) 行。
    n_lams = len(win_lams)
    endpoint_diff_uncertainty_kj = (
        float(ddf_matrix[1, n_lams] * float(kt)) if n_lams > 1 else 0.0
    )

    return {
        "lambdas": list(int(x) for x in win_lams),
        "f": f_phys_kj,
        "df": df_phys_kj,
        # 🔑 [2026-09-11] **物理态之间**的不确定度子阵（kJ/mol）。
        # `df` 是各物理态**相对采样态**的边际不确定度，合并它们会系统性算错
        # （同一次 MBAR 拟合、同一批样本，彼此有协方差）—— 这个坑上面
        # endpoint_diff_uncertainty 的注释已经记过一次。
        # 端到端累计偏差门需要 σ(F̂_j − F̂_0) 和 σ(F̂_a − F̂_b)，**直接读这个矩阵**，
        # 不做任何平方相加。物理态在增广矩阵里占 1..n_lams 行。
        "ddf_physical_matrix_kJ_mol": (
            np.asarray(ddf_matrix[1:n_lams + 1, 1:n_lams + 1], dtype=float) * float(kt)
        ),
        "endpoint_diff_uncertainty_kJ_mol": endpoint_diff_uncertainty_kj,
        "n_frames_used": int(n_k_local[sampled_row]),
        "global_offset": global_offset,
        "min_ess_ratio": min_ess_ratio,
        "ess_ratio_per_lambda": ess_ratio_per_lambda,
        "statistical_inefficiency": float(g_val),
        # ---- [ESS_GATE_PROTOCOL_VERSION=2] 只报告、不设门的诚实诊断 ----
        "ess_gate_protocol_version": int(ESS_GATE_PROTOCOL_VERSION),
        "ess_gate_metric": "mixture_coverage_ess_common_mode_removed",
        "ess_gate_mixture_gauge": quality.get("mixture_gauge"),
        "min_occupancy_normalized": min_occupancy_normalized,
        "min_occupancy_normalized_threshold": float(
            IBS_LOCAL_MBAR_GATE_OCC_MIN_FRACTION
        ),
        "mixture_occupancy_normalized": quality.get("mixture_occupancy_normalized"),
        # 🔑 [ESS_GATE_PROTOCOL_VERSION=5] 这几项算在**去相关之前**的帧集上。
        "raw_min_ess_ratio": raw_min_ess_ratio,
        "raw_min_absolute_ess": raw_min_absolute_ess,
        "raw_ess_ratio_per_lambda": raw_ess_ratio_per_lambda,
        "top1pct_raw_weight": top1pct_raw_weight_gated,
        "raw_gate_n_frames_pre_decorrelation": raw_gate_n_frames_pre_decorrelation,
        # ---- 旧（抽稀）口径：仅诊断，供历史产物对账 ----
        "raw_min_ess_ratio_post_decorrelation": raw_min_ess_ratio_post_decorrelation,
        "raw_ess_ratio_per_lambda_post_decorrelation": (
            raw_ess_ratio_per_lambda_post_decorrelation
        ),
        "top1pct_raw_weight_post_decorrelation": quality.get("top1pct_raw_weight"),
        # 🔑 [2026-09-02] 偏置涨落/物理信号 之比（仅诊断，不设门）——见其计算处注释。
        # 本函数同时驱动冻结验证 loose gate（唯一生产放行证明）与在线 early-stop，
        # 所以这一项在这里落盘尤其有价值：它是 ESS 抓不到的那个失效维度。
        "bias_to_signal_ratio": quality_pre.get("bias_to_signal_ratio"),
        "bias_fluctuation_kJ_mol": quality_pre.get("bias_fluctuation_kJ_mol"),
        "target_energy_fluctuation_kJ_mol": quality_pre.get(
            "target_energy_fluctuation_kJ_mol"
        ),
        "common_mode_log_sigma_kT": quality.get("common_mode_log_sigma_kT"),
        "common_mode_log_mean_kJ_mol": quality.get("common_mode_log_mean_kJ_mol"),
        "reweighting_quality_error": quality.get("error"),
        "statistical_inefficiency_worst_state": int(g_worst_state),
    }


