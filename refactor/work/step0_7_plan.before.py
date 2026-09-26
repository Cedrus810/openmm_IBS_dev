# ---------------------------------------------------------------------------
# stage 级质量门的**报告**（不是动作依据）
# ---------------------------------------------------------------------------
# 🔑🔑 [2026-09-15 老板定案] **这些门一条都不许再驱动自动补帧。**
#
# 它们原来是 `solve_stage_integrated` 的 `converged` 那个合取的分项，而
# `converged` 已经被删除（「算出了数」被误报成「精度已验收」）。合取里的四个
# 判据全是**未经标定**的拟合阈值，其中 `max_endpoint_uncertainty_kJ_mol ≤ 1.0`
# 的来历是抄自一个**默认关闭**的 early-stop 启发式的默认形参、全仓无任何依据；
# 而且它是**逐段**量，科学目标是两腿合成的 `σ_bind`（实测各段 `[0.3×5, 2.0]`
# ⟹ σ_bind = 2.98 已达标，门却判未收敛并命令补帧；把那段从 2.0 压到 1.0 要
# 4× 采样量、精度提升为零）。更根本：MBAR 的 σ 是渐近估计，对「该采的构型一次
# 都没采到」结构性失明 —— 4W53 (+32 kJ/mol) 与 decharging (−88 kJ/mol) 两次大错
# σ 都很小、都没报警。
#
# 所以本函数**只剩报告能力**：逐条给出实测值 / 阈值 / 最差窗口，对人工审查有用。
# 每个条目都带 `drives_action: False`，`decide()` 里**没有任何分支读它**。
# 下面三个词是**科学分类**（这道门在物理上属于哪一类失效），不是「对症动作」：
#   · 支撑/偏斜类（目标态支撑度、mixture 重叠）：同分布加帧治不了
#     （§5.1 实测 250k→1M 让 top1% 从 0.545 涨到 0.762、ESS 比值反而变差）；
#   · 样本量类（去相关样本数、端点 σ）：语义是「尚不可测」而不是「不合格」（§3）；
#   · 读不出来的 ⟹ **不归类**，标 `unattributed`。
# ⚠️ 不得据此在 `decide()` 里发 `RUN_PRODUCTION` / `INSERT_LAMBDA` /
#   `SPLIT_TAIL_WINDOW` 等花 GPU 的动作 —— 那正是被删掉的东西。
STAGE_GATE_MORE_SAMPLING = "sample_size_diagnostic"
STAGE_GATE_NARROW_SPAN = "span_diagnostic"
STAGE_GATE_UNATTRIBUTED = "unattributed"

# 🔑🔑 [审计 #47，2026-09-14] **边际增益刹车的三个常量，刻意保守。**
#
# 先前两处判据（`_frames_admission` 的 `max(fin[1:]) <= fin[0]`、`_no_gain` 的
# `max(h[1:]) <= h[0]`）都把 `min N_eff/g` / `solver_n_decorrelated` 当成**随采样
# 单调增**的量。它们不是：
#   · 分子 Kish ESS `(Σw)²/Σw²` 在小 N 时**乐观偏高**（少数几帧的权重还没分化开）；
#   · 分母 ĝ 在 N ≫ τ 之前**单调上涨**（自相关时间被系统性低估，随样本增加才现形）。
# 两个偏差**同向**（都让早期点偏高），而旧判据的基准还偏偏取 `h[0]` —— 偏高最严重
# 的那一点。于是「没涨」几乎必然成立，刹车对正常窗口误触发。
# 这与本文件 `_unreach` 那段**自己否决**「周期内按 n_eff 外推提前判死」的论证
# 直接矛盾：那里承认 g 只在 N ≫ τ 之后才稳。
#
# 改成单侧、带余量的形式：
#   · 至少 `MIN_POINTS` 个点才判（两点根本分不出趋势与噪声）；
#   · 基准是**除最后一点外全部点的中位数**（中位数对早期那个偏高点稳健）；
#   · 只有最后一点**低于基准的 `STALL_RATIO`** 才算"证伪了加帧"。
# ⚠️ 这是**刹车**不是精确判据：它只回答"要不要停止自动补帧"，
# 不回答"这个窗口收敛了没有"。**不要**在这里发明新的统计检验。
MARGINAL_GAIN_MIN_POINTS = 3
MARGINAL_GAIN_STALL_RATIO = 0.9

# 🔑🔑 [审计 #46，2026-09-14] **「加帧治不了」的归因只能有一份判定。**
#
# `window_self_support_check` 写侧特意把三个来源分开：
#   · `top1pct_veto`      权重塌缩（单帧支配）⟹ **加帧治不了**，要缩跨度
#   · `min_n_eff_over_g`  支撑比值偏低       ⟹ 样本量问题，加帧对症
#   · `solver_eligibility` 去相关帧数不够资格 ⟹ 纯样本量，加帧**正是**对症动作
# 而它同时把 `solver_eligibility` 这一档的 verdict **强制置成 `HARD_INSUFFICIENT`**
# （"更前置的失效优先"）。控制器两处判据（`_read_single_stage._support_failed`、
# `decide._is_skew`）只认 verdict、不看来源 ⟹ 长 τ 的解耦端窗口（帧数确实不够）
# 被判成"偏斜、加帧治不了"，送去插 λ —— 而插 λ **不缩短构象慢模态的 τ_int**。
# 写侧自己的注释就是「帧数不够 ≠ 支撑不够」，读侧必须同口径。
# 🔑🔑 [2026-09-15 真机 brd4_ligand1/rep2] **`min_n_eff_over_g` 从这张表里拿掉。**
#
# 上面第 2099 行原来写「`min_n_eff_over_g` 支撑比值偏低 ⟹ 样本量问题，加帧对症」。
# 反例就在同一个 run 里：
#     win1  n_decorr = 888（地板 10 的 **88 倍**）、min N_eff/g = 8.68
# **帧一点都不缺**，缺的是权重压不到目标态上。同分布再加一块，只会得到权重剖面
# 一模一样的更多帧 —— 这正是本仓 §5.1 实测过的：250k→1M 让 top1% 从 0.545 涨到
# 0.762、ESS 比值**反而更差**；而重标定一次 rawESS 27.5→503。
# 真机后果：控制器连发 4 块 `RUN_PRODUCTION` 给 win1，配额烧光 → NO_FEASIBLE_ACTION，
# 而真正缺窗的 win4 全程排在 `blocked_by_upstream` 里、一次都没被看过。
#
# ⚠️ **`solver_eligibility` 必须留下**，理由见上一段（长 τ 的解耦端窗口是真的帧不够，
# 插 λ 不缩短构象慢模态的 τ_int）。这次只动 `min_n_eff_over_g` 一个。
#
# ⚠️ 写侧有个硬不变量让这条改动是安全的：`window_self_support_check` 里
#       solver_eligible = n_dec >= floor
#       if not solver_eligible: verdict_source = "solver_eligibility"   # 覆盖
#   这一步排在比值分档**之后** ⟹ `verdict_source == "min_n_eff_over_g"` 出现时，
#   `n_dec >= floor` **必然成立**（帧数已经够了）。即便如此，下面的
#   `support_failure_is_skew` 仍然接受可选的帧数并自己再验一次 —— 不把正确性
#   押在另一个文件的不变量上。
_SAMPLE_SIZE_VERDICT_SOURCES = ("solver_eligibility",)

# 🔑 [审计 #58] rewindow 子窗在**求解器命名空间**里的索引基数。
# 执行器建窗时按 `10_000 + 100 * n` 钉死（`_immutable_rewindow_step`）；
# 控制器只用它区分「这个 `window_index` 是物理窗口还是子窗」。
SOLVER_UNIT_INDEX_BASE = 10_000


def frames_growth_headroom(view, window_idx=None, *, unit_id=None):
    """把**剩下的补帧配额全花掉**，帧数还能涨几倍。读不出来返回 `None`。

    `(1 + cap) / (1 + used)`：每块补帧加的是一个初始生产块，所以用掉 `used` 块的
    窗口现在有 `(1+used)` 份帧，配额见底时最多 `(1+cap)` 份。
    ponytail: 「一块 = 一个初始块」是实测值（brd4_ligand2/rep1 win2：4 块把
    250k 走到 1.25M），不是从 config 读的。哪天补帧块长度可变，就改成按
    `production_steps` 与本轮块步数算，别在这里加分支。
    """
    cap = view.get("max_production_blocks_per_window")
    if cap is None:
        return None                     # 上限未知 ⟹ 射程算不出来，不猜
    try:
        cap = int(cap)
    except (TypeError, ValueError):
        return None
    if cap < 0:
        return None
    if unit_id is not None:
        rows = (view.get("production_blocks_total_by_unit")
                or view.get("production_blocks_by_unit") or {}).get(str(unit_id))
    else:
        rows = (view.get("production_blocks_total_by_window")
                or view.get("production_blocks_by_window") or {}).get(int(window_idx))
    used = len(rows or [])
    if used >= cap:
        return 1.0                      # 配额已见底 ⟹ 一帧都加不了了
    return (1.0 + cap) / (1.0 + used)


def n_eff_over_g_reachable_by_frames(ratio, target, headroom):
    """**最乐观**假设下，把剩余补帧配额全花掉能不能把 `N_eff/g` 推过门。

    判据是恒等式，不是新阈值：

        N_eff/g = n_decorr × (N_eff/N)

    右边第二项是权重剖面的**效率**，是强度量 —— 同分布再采只放大 n_decorr，
    不改效率。所以「加帧的射程」就是 `ratio × headroom`。

    返回 `None` = 判不了（缺数），调用方按既有行为处理。
    ⚠️ 这是**上界**：实测中效率会随帧数变差（§5.1，250k→1M 让 top1% 0.545→0.762），
    所以 `True` 只表示"值得一试"，不表示"一定推得过"。真正的刹车是事后的
    `marginal_gain_stalled()` —— 两者一前一后，缺一不可。
    """
    if ratio is None or target is None or headroom is None:
        return None
    try:
        r, t, h = float(ratio), float(target), float(headroom)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(r) and np.isfinite(t) and np.isfinite(h)):
        return None
    if h < 1.0 or t <= 0.0:
        return None
    return bool(r * h >= t)


def eta_lever_can_reach(n_eff_input_n_frames, g, target):
    """**只改权重剖面**的动作（η 杠杆）有没有可能把 `N_eff/g` 推过门。

    🔑🔑🔑 [2026-09-18] **这是一个上界，不是阈值；它由定义推出来，不需要标定。**

    验收量（`ibs_engine.window_self_support_check`）::

        R = N_eff/g,   N_eff = (Σw)² / Σw²   （Kish ESS，算在**未抽稀**帧上）

    记 N = 真正进 N_eff 的帧数（`n_eff_input_n_frames_per_state`），
    η = N_eff/N。则 ``R = η · N / g``，而 Cauchy–Schwarz 给

        (Σw)² ≤ N · Σw²   ⟹   N_eff ≤ N   ⟹   **η ≤ 1**

    （等号仅当所有权重相等，即"完美偏置"。）于是

        **R ≤ N / g**

    ⟹ `N/g < target` 时，**任何只改 η 的动作都不可能过门**：插 λ（缩跨度改重叠）、
    重标定 / 重学 f_k（改采样混合）、有界重窗 —— 它们全部只动 η，而 η 已经被 1 封顶。
    此时唯一还能动的是分母那一侧：把 **N** 加上去（同分布补帧），或者把 **g** 降下来
    （换独立初态 / 处理慢模态）。

    ⚠️ 与 `n_eff_over_g_reachable_by_frames()` 是**两个不同的问题**，别混：
      · 那个问的是「**加帧**够不够得着」，判据 `R_now × headroom`（R_now 已含 η）；
      · 这个问的是「**改 η** 够不够得着」，判据 `N/g`（把 η 拿掉、取它的上界 1）。
    两个都可能为假 —— 那才是真的没路走，而不是"再试一个动作"。

    ⚠️ **只在明确算出来时给答案。** `None` = 判不了（缺 N 或 g、非有限、g ≤ 0），
    调用方按"未知不授权"处理 —— 与本模块别处的三态口径一致。

    返回 `True`（η 杠杆**有可能**够到，不保证一定够到）/ `False`（**证明**够不到）
    / `None`（判不了）。
    """
    if n_eff_input_n_frames is None or g is None or target is None:
        return None
    try:
        n, gg, t = float(n_eff_input_n_frames), float(g), float(target)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(n) and np.isfinite(gg) and np.isfinite(t)):
        return None
    if gg <= 0.0 or t <= 0.0 or n <= 0.0:
        return None
    return bool(n / gg >= t)


def support_failure_is_skew(
    verdict, verdict_source, *, n_decorrelated=None, min_frames=None,
    min_n_eff_over_g=None, n_eff_over_g_target=None, frames_headroom=None,
    n_eff_input_n_frames=None, bottleneck_g=None
) -> bool:
    """这次自检失败是**支撑/偏斜类**（加帧治不了）还是**样本量类**（加帧对症）。

    两处调用点（物理窗口 / rewindow 子窗）共用这一份，见上面的长注释。

    `n_decorrelated` / `min_frames` 可选：给了就**自己再验一次**帧数够不够，
    不把正确性押在写侧的覆盖顺序上（见 `_SAMPLE_SIZE_VERDICT_SOURCES` 那段
    ⚠️）。读不到就退回只看 verdict + 来源，行为与先前一致。

    🔑🔑 [2026-09-16] **`min_n_eff_over_g` 偏低不再无条件判成"加帧治不了"。**

    低比值**不蕴含**加帧无用：稳定分布下 `N_eff ∝ N`、`g` 不变 ⟹ 比值随帧数线性
    涨，5 可以长到 10。先前那条规则是从一个真反例（brd4_ligand1/rep2 win1：
    n_decorr=888、比值 8.68 ⟹ 帧一点不缺）推广出来的，但推广过头了：它对
    「帧确实少、比值因此低」的窗口同样成立，于是把一个补帧能治的窗口送去改布局。

    改法不是换个阈值，而是**把射程算出来**（`n_eff_over_g_reachable_by_frames`）：
    剩余配额全花掉仍够不着门 ⟹ 才叫"加帧治不了"。三个数缺任何一个就判不了射程，
    此时保持既有判法（保守当偏斜）—— 老产物、子窗那类拿不到块账的调用点因此
    逐位不变。
    """
    return support_failure_attribution(
        verdict, verdict_source, n_decorrelated=n_decorrelated,
        min_frames=min_frames, min_n_eff_over_g=min_n_eff_over_g,
        n_eff_over_g_target=n_eff_over_g_target,
        frames_headroom=frames_headroom,
        n_eff_input_n_frames=n_eff_input_n_frames,
        bottleneck_g=bottleneck_g) == SUPPORT_FAILURE_STRUCTURAL


# 归因四态。⚠️ `UNKNOWN` **不是**"两者之间"，它是「**还没判出来**」。
SUPPORT_FAILURE_SAMPLE_SIZE = "SAMPLE_SIZE"   # 帧确实少 ⟹ 加帧对症
SUPPORT_FAILURE_STRUCTURAL = "STRUCTURAL"     # 权重塌缩/跨度过大 ⟹ 加帧治不了
SUPPORT_FAILURE_UNKNOWN = "UNKNOWN"           # 判不出来 ⟹ **不得**授权任何覆盖
# 🔑🔑🔑 [2026-09-18] **第四态：瓶颈在 `N/g` 这个分母上，不在权重剖面上。**
# 由 `eta_lever_can_reach()` 的 `R ≤ N/g`（η ≤ 1，Cauchy–Schwarz）判出来：
# `N/g < 门` 时**任何只改 η 的动作都被证明够不到门** —— 插 λ、重标定/重学 f_k、
# 有界重窗全在此列。它**不是** `STRUCTURAL`：那一档授权的正是缩跨度，而缩跨度
# 在这里是被证伪的。剩下能动的只有 N（同分布补帧）与 g（换独立初态 / 治慢模态）。
SUPPORT_FAILURE_N_OVER_G_BOUND = "N_OVER_G_BOUND"


def support_failure_attribution(
    verdict, verdict_source, *, n_decorrelated=None, min_frames=None,
    min_n_eff_over_g=None, n_eff_over_g_target=None, frames_headroom=None,
    n_eff_input_n_frames=None, bottleneck_g=None
) -> Optional[str]:
    """支撑失败的归因，**四态**。非失败返回 `None`。

    🔑🔑 [2026-09-17，用户拍板 A] **「乐观上界说也许够得着」曾被升级成「这是样本量
    问题」，那是本条链的真错误。**

    真实关系是

        R_future = R_now × H × (η_future / η_now) × (g_now / g_future)，  η = N_eff/N

    而 `n_eff_over_g_reachable_by_frames` 只留下 `R_now × H`，**同时假定 η 与 g 恒定**，
    两个假设实测都朝不利方向走（g 实测 12.25→17.63 = 1.44×；η 另有恶化，
    §5.1 250k→1M 让 top1% 0.545→0.762）。所以它的 `True` 只够说「**还没被证伪**」。

    ⚠️ **不做 `R_now × H ÷ 1.44` 那种"已观测 g 惩罚"**，两个理由（用户核实）：
      · `R_now` 已经是用**当前** g 算出来的，再乘 `g_prev/g_now` 是**重复扣除**；
      · 拿过去的 1.44 用于未来 = 假设未来 g 还会再涨同样比例，**那仍是外推**，
        只是不叫"斜率"。
      · 而且 1.44 只解释 1.44×，文档里的约 2.5× 还含 η 恶化 ⟹ 说"共同根因只有 g"是错的。
    要做数值修正，先得有数据契约：production 的 g 与 η 目前**没有**按
    `path_version/f_k/segment/window` 落进历史（ledger 只有步数、去相关帧数、ratio）。

    三态的判法（严格化的那一条在 `min_n_eff_over_g`）：
      · `SAMPLE_SIZE` 只由**硬证据**给出 —— `solver_eligibility` 那族来源，
        或明确的 `n_decorrelated < min_frames`；
      · 射程 `ratio × headroom ≥ target` 只给 `UNKNOWN`（"可能够得着"），
        **不再**算作"这是样本量问题"；
      · 其余（top1pct 否决、射程都够不着、老产物 + HARD_INSUFFICIENT）= `STRUCTURAL`。

    ⚠️ `support_failure_is_skew()` 的布尔值**逐位不变**（它问的是"是不是 STRUCTURAL"），
    所以既有调用点行为一字不改。变的只有 O1：它以前拿 `not is_skew` 当"样本量类"，
    那等于把 `UNKNOWN` 也算进去了 —— 见 `plan()` 里 O1 那段。
    """
    src = str(verdict_source or "")
    if verdict not in ("HARD_INSUFFICIENT", "INSUFFICIENT_DATA"):
        return None           # 通过的窗口同样带 verdict_source，不能只看来源
    if src in _SAMPLE_SIZE_VERDICT_SOURCES:
        return SUPPORT_FAILURE_SAMPLE_SIZE    # 样本量不够 ⟹ 加帧就是对症动作
    # 🔑🔑🔑 [2026-09-18] **先判「改 η 到底够不够得着」，再谈是哪一类 η 失败。**
    #
    # `R = η·N/g` 且 `η ≤ 1`（Kish ESS 的 Cauchy–Schwarz，见
    # `eta_lever_can_reach` 的推导）⟹ `R ≤ N/g`。所以 `N/g < 门` 时，**插 λ /
    # 重标定 / 重学 f_k / 有界重窗这一整族（它们全都只动 η）被证明够不到门**。
    # 把这种窗口判成 `STRUCTURAL` 是错的 —— 那一档授权的恰好是缩跨度。
    #
    # ⚠️ 排在 `top1pct_veto` **之前**：权重塌缩确实是 η 的直接证据，但"η 坏"
    # 与"改 η 能不能救回来"是两个问题，后者由这个上界回答，与 verdict_source 无关。
    # ⚠️ 只在**明确算出 False** 时才用（`None` = 缺 N 或 g ⟹ 照旧往下走），
    # 与本函数别处"未知不授权"的口径一致。
    if eta_lever_can_reach(
            n_eff_input_n_frames, bottleneck_g, n_eff_over_g_target) is False:
        return SUPPORT_FAILURE_N_OVER_G_BOUND
    if src == "min_n_eff_over_g":
        # 帧数本身就没到地板 ⟹ **硬证据**的样本量问题（先补帧，比值以后再说）。
        # 写侧的覆盖顺序保证这种情形会写成 `solver_eligibility`，这里只是不依赖它。
        if (n_decorrelated is not None and min_frames is not None
                and int(n_decorrelated) < int(min_frames)):
            return SUPPORT_FAILURE_SAMPLE_SIZE
        # 🔑🔑 [2026-09-17 P0] **只有明确算出 `False` 才是 STRUCTURAL。**
        # 先前写成「`True` ⟹ UNKNOWN，其余 ⟹ STRUCTURAL」，于是 `None`
        # （三个输入缺任何一个、门缺失、读数非有限）落进了 STRUCTURAL ——
        # **数据缺口被当成了结构性结论**，而 STRUCTURAL 会授权缩跨度类动作
        # （插 λ / 拆窗 / 有界重窗 / D3 停机）。那是拿"没测出来"当"测出来是坏的"。
        _reach = n_eff_over_g_reachable_by_frames(
            min_n_eff_over_g, n_eff_over_g_target, frames_headroom)
        if _reach is False:
            return SUPPORT_FAILURE_STRUCTURAL   # 完整数据 + 连乐观上界都够不着
        # `True`（只是还没被证伪，见上面的 R_future 展开）与 `None`（判不了）
        # **都不授权任何结构动作**。
        return SUPPORT_FAILURE_UNKNOWN
    if src == "top1pct_veto":
        return SUPPORT_FAILURE_STRUCTURAL       # 权重塌缩是直接证据
    # 🔑 [2026-09-17 P0] **老产物缺来源 ⟹ UNKNOWN，不再按 verdict 兜成 STRUCTURAL。**
    # 那条兜底的原意是"保守、不拿加帧顶替"，但它选错了保守方向：STRUCTURAL
    # **授权**缩跨度，而缩跨度同样是要花 GPU、要改布局的动作。真正保守的是
    # UNKNOWN —— 它两边都不授权（既不补帧也不改布局）。同类的"数据缺口授权"。
    return SUPPORT_FAILURE_UNKNOWN


# 🔑🔑🔑 [S2-H + S2-F，2026-09-17] **「总是改变盘面」的动作既没有次数预算、
# 也没有事后有效性判定。**
#
# `RECALIBRATE_FK` `probe_only=False`、**开一个新采样段**：
#   · 停滞保护判的是「同一个动作 + **盘面未变**」(`seen[key] >= 3`) ⟹ 它每轮都改盘面
#     ⟹ `seen[key]` 恒重置成 1 ⟹ **永远到不了 3**；
#   · `action_noop_fingerprint()` 含段维度 ⟹ 新段 = 新指纹 ⟹ 记录永不匹配。
# ⟹ **对唯一的通用刹车结构性免疫。** 而插 λ 有 `max_path_insertions`、补帧有块数
# 硬上限，**只有这一族什么预算都没有**。
#
# 真机 `cmet_ligand1/rep1`（31/40 轮里 17 轮花在窗口 4）：
#     20 RECALIBRATE_FK  seg=vanishing    steps=750000  ratio=9.645  ← 离门 10 差 0.355
#     21 RECALIBRATE_FK  seg=vanishing_4  steps=250000  ratio=4.182  ← 新段，750k 步作废
#     22 RECALIBRATE_FK  seg=vanishing_5  steps=250000  ratio=0.837
#     23 RECALIBRATE_FK  seg=vanishing_6  steps=250000  ratio=7.017
# 第 20 轮再补一块几乎必过；三次重标定把它推到 0.837，盘上留下 7 个采样段，全程无人察觉。
#
# 判据**不发明新阈值**：次数用与 `max_path_insertions` 同性质的终身预算；
# 「有没有变好」复用 `marginal_gain_stalled()`（≥3 点、基准取前面各点的中位数、
# 末点明显低于基准才算被证伪）。两者都只读**已经在写**的自治历史。
STRUCTURAL_ACTION_MAX_PER_WINDOW = 3

# 会**开新采样段**的动作：它们每次都改盘面，所以必须靠次数+效果管，不能靠停滞保护。
_SEGMENT_OPENING_ACTIONS = (
    "RECALIBRATE_FK", "RELEARN_FK_EPOCH", "PROBE_REANCHOR_EPOCH",
)


def structural_action_history(iterations, action, window_idx, path_version):
    """某个窗口在**当前布局**下被施加过几次这个动作，以及每次施加时的验收量。

    读的是已经在写的 `stage2_autonomous_history.json`，不新造产物。
    返回 `(次数, [每次施加时该窗口的 min_n_eff_over_g, ...])`。

    ⚠️ 按 `path_version` 过滤：布局一变就是另一套几何，重来一次是对的。
    ⚠️ 取的是**控制器发出的**动作（`action`），不是执行器实际跑的那个
    （停滞降级会把实际动作另记在 `action_executed` 里，审计 #62a）——
    这里要的是"控制器试过几次"。
    """
    n, series = 0, []
    for it in (iterations or []):
        if path_version is not None and it.get("path_version") != path_version:
            continue
        if str(it.get("action")) != str(action):
            continue
        if int(window_idx) not in [int(x) for x in (it.get("windows") or [])]:
            continue
        n += 1
        for sn in (it.get("snapshot") or []):
            if int(sn.get("window_idx", -1)) == int(window_idx):
                v = sn.get("min_n_eff_over_g")
                if v is not None:
                    series.append(float(v))
                break
    return n, series


def structural_action_refused(view, action, window_idx):
    """这个**改盘面**的动作对这个窗口该不该再发。`None` = 可以发，否则给不可行理由。

    两道，缺一不可（见上面那段长注释）：
      · **次数**：同一布局下每窗口至多 `STRUCTURAL_ACTION_MAX_PER_WINDOW` 次
        —— 这一道专治「它改盘面所以停滞保护永远数不到 3」；
      · **效果**：把逐次施加时的验收量串成序列，交给共享的 `marginal_gain_stalled()`
        —— 这一道专治「三次全变差却无人察觉」。
    """
    if str(action) not in _SEGMENT_OPENING_ACTIONS:
        return None
    _n, _series = structural_action_history(
        ((view.get("autonomous_history") or {}).get("iterations") or []),
        action, int(window_idx), view.get("path_version"))
    # ⚠️ **效果闸排在次数闸前面。** 默认配置下 `STRUCTURAL_ACTION_MAX_PER_WINDOW == 3`
    # 而 `MARGINAL_GAIN_MIN_POINTS == 3`，两道在同一点触发；次数闸若排在前面，
    # 效果闸就**永远不可达**（死代码）。排在前面还让理由更有信息量：
    # 「它在把这个窗口推得更差」比「次数到了」更能指导人工判断。
    _stalled, _d = marginal_gain_stalled(_series)
    if _stalled:
        return (
            f"动作 `{action}` 对窗口 {int(window_idx)} **没有把验收量推上去**："
            f"逐次施加 {_d['series']}，基准（除末点外各点的中位数）="
            f"{_d['baseline_median_of_earlier_points']:.3g}，末点 {_d['last']:.3g}"
            f"（比值 {_d['last_over_baseline']:.2f} < 刹车阈值 {_d['stall_ratio']}）"
            " ⟹ 再发一次没有依据。"
            "⚠️ 它每次都**开新段**，上一段累计的生产步数不再参与这一段 —— "
            "真机实测过 750k 步在第 21 轮被丢掉、验收量从 9.645 掉到 4.182。"
        )
    if _n >= STRUCTURAL_ACTION_MAX_PER_WINDOW:
        return (
            f"动作 `{action}` 在当前布局下已经对窗口 {int(window_idx)} 发过 {_n} 次"
            f"（上限 {STRUCTURAL_ACTION_MAX_PER_WINDOW}）⟹ 不再自动发。"
            "⚠️ 这一族动作**每次都开一个新采样段**，所以主循环的停滞保护"
            "（同一动作 + 盘面未变）对它**结构性免疫** —— 盘面每次都变，计数恒为 1。"
            f"逐次施加时该窗口的验收量：{[round(x, 3) for x in _series]}。"
            "要继续需要人工判断：它是不是真的在把这个窗口推向合格。"
        )
    return None


def marginal_gain_stalled(series) -> Tuple[bool, Dict[str, Any]]:
    """加帧还有没有用。返回 `(是否已被证伪, 诊断)`。见上面三个常量的长注释。"""
    _raw = list(series or [])
    vals = [float(x) for x in _raw if x is not None]
    if len(vals) < MARGINAL_GAIN_MIN_POINTS:
        # 🔑🔑🔑 [S2-A，2026-09-17] **「点数不够」和「一个输入都没有」不是一回事。**
        #
        # 先前两者都报 `NOT_ENOUGH_POINTS` —— 读起来像「才跑了 1-2 块，再跑跑看」，
        # 而真相可能是「这个判据**一个输入都没有**，它从头到尾没有带过电」。
        # 两者处置完全相反：前者继续跑就会有数据，后者**再跑多少轮也不会有**。
        #
        # 真机（`cyclod_ligand2/rep1` w5 连批 3 块、`cmet_ligand1/rep2` w0 连批 3 块、
        # `cmet_ligand2` w4 批满 4 块）：判据量全程是 `[None, None, None]` ——
        # 因为 `solver_n_frames_decorrelated` 来自 stage 结果的
        # `window_overlap_diagnostics`，而 stage 结果**只在 `ANALYZE` 执行时落盘**，
        # `ANALYZE`（分支 8）又排在补帧那一族之后 ⟹ 整个采样阶段读不到
        # ⟹ 过滤掉 None 之后是空列表 ⟹ 恒 `NOT_ENOUGH_POINTS` ⟹ **刹车永不触发**。
        # 那几个窗口的块是被「块数硬上限」停的，不是被「没增益」停的。
        #
        # ⚠️ 这里**只把失效说出来，不改任何行为**（仍然返回"没停滞"）。
        # 换一个替代判据（比如改用自检侧的 `min_n_eff_over_g`）是另一回事：
        # 那个量单次读数噪声 34×，拿它当**停止**判据会误杀 ——
        # 而误杀的代价今天已经实测过（两次生产跑连崩）。
        # 真正的修法是 `AUDIT-S2-02` 的重构，见 docs/TODO_P2.md `S2-A`。
        _present = sum(1 for x in _raw if x is not None)
        return False, {
            "verdict": ("NO_CRITERION_INPUT_AT_ALL" if (_raw and _present == 0)
                        else "NOT_ENOUGH_POINTS"),
            "n_points": len(vals),
            "n_rows_seen": len(_raw),
            "min_points": MARGINAL_GAIN_MIN_POINTS,
            "series": vals,
            **({"inoperative_reason":
                "判据量逐块全是 None ⟹ 这道刹车在本轮**没有带电**。"
                "`solver_n_frames_decorrelated` 来自 stage 结果的 "
                "`window_overlap_diagnostics`，而 stage 结果只在 `ANALYZE` 执行时落盘，"
                "`ANALYZE` 又排在补帧那一族之后 ⟹ 整个采样阶段读不到。"
                "见 docs/TODO_P2.md `S2-A`。"}
               if (_raw and _present == 0) else {}),
        }
    head = sorted(vals[:-1])
    m = len(head)
    baseline = head[m // 2] if m % 2 else 0.5 * (head[m // 2 - 1] + head[m // 2])
    last = vals[-1]
    ratio = (last / baseline) if baseline > 0 else None
    stalled = bool(baseline > 0 and last < baseline * MARGINAL_GAIN_STALL_RATIO)
    return stalled, {
        "verdict": "STALLED" if stalled else "STILL_GAINING_OR_FLAT",
        "series": vals,
        "baseline_median_of_earlier_points": baseline,
        "last": last,
        "last_over_baseline": ratio,
        "stall_ratio": MARGINAL_GAIN_STALL_RATIO,
        "n_points": len(vals),
    }


def _worst_window_by(records, key, *, largest=False):
    """从逐窗记录里挑最差的那个窗口下标；读不出来就返回 None（不猜）。

    🔑🔑 [2026-09-15 真机] **逐窗记录里有一半的指标是「逐 λ 态」的 list。**

    `float(r[key])` 直接假定标量 ⟹ 遇到 list 当场 `TypeError` **炸穿 `decide()`**。
    真机 11 个 run 里 **5 个**这样崩（`top1pct_raw_weight` 是 `list[7]`；
    该窗口记录里**根本没有**标量版本，受门的标量 `max_top1pct_raw_weight`
    在 stage 顶层、不在逐窗记录里）。

    ⚠️ 这是**预先就存在**的 bug（HEAD 上同样崩），先前一直没暴露是因为
    stage 结果从不落盘（`ANALYZE` 轮不到 ⟹ `window_overlap_diagnostics` 读不到）
    ⟹ 这段代码走不到。「退出前必跑一次全路径 ANALYZE」把它变成**每轮必经**。

    归约口径：逐态 list ⟹ 取该窗口**最差的那个态**，方向与选窗一致
    （`largest=True` 时越大越差 ⟹ 取 `max`；否则取 `min`）。这与写侧
    `max_top1pct_raw_weight = max(逐态)` 的口径一致，不新发明。
    读不成数（dict / 字符串 / 全是 NaN / 空 list）⟹ 该窗口**不进候选**，
    与 docstring 的"读不出来就不猜"一致 —— 绝不在这里抛。
    """
    def _scalar(v):
        vals = v if isinstance(v, (list, tuple)) else [v]
        out = []
        for x in vals:
            try:
                fx = float(x)
            except (TypeError, ValueError):
                continue
            if fx == fx and abs(fx) != float("inf"):   # 排掉 NaN / ±inf
                out.append(fx)
        if not out:
            return None
        return max(out) if largest else min(out)

    cand = [
        (_scalar(r[key]), int(r["window_index"]))
        for r in (records or [])
        if isinstance(r, dict)
        and r.get(key) is not None
        and r.get("window_index") is not None
        and _scalar(r[key]) is not None
    ]
    if not cand:
        return None
    return (max(cand) if largest else min(cand))[1]



def _bottleneck_observation(selfchk) -> Dict[str, Any]:
    """从一份 `window_self_support_check` 产物里抽出瓶颈态的 (g, η, ratio) 观测。

    🔑🔑 [2026-09-17] **这是 `n_eff_over_g` 可达性判据的数据契约。**
    现在的射程判据 `R_now × H` 假定 η 与 g 恒定，两个假设实测都朝不利方向走
    （g 实测 12.25→17.63）。要把实测增长纳进来，前提是先**逐块记下** g 与 η ——
    而历史台账原来只有步数、去相关帧数、ratio，连 g 都没有。本函数补的就是这个缺口。

    ⚠️ **本轮只记录，不参与任何决策**（用户拍板：先记、可回填、再 shadow 对比，
    数据够了才谈启用）。所以这里一个阈值都没有。

    ⚠️ 老产物缺键 ⟹ 全部 `None`。**不填 0**：0 会被下游读成"效率为零"，
    而真相是"没测"。三态里这是 UNKNOWN。
    """
    _empty = {
        "bottleneck_state": None, "bottleneck_g": None, "bottleneck_n_eff": None,
        "bottleneck_eta": None, "bottleneck_ratio": None,
        "bottleneck_n_eff_input_n_frames": None,
        "n_eff_frame_set": None, "n_eff_frame_set_n_frames_raw": None,
    }
    if not isinstance(selfchk, dict):
        return dict(_empty)
    k = selfchk.get("worst_state_by_n_eff_over_g")
    if k is None:
        return dict(_empty)
    k = int(k)

    def _at(key):
        arr = selfchk.get(key)
        if not isinstance(arr, (list, tuple)) or not (0 <= k < len(arr)):
            return None
        return arr[k]

    g = _at("statistical_inefficiency_per_lambda")
    ne = _at("n_eff_per_state")
    nin = _at("n_eff_input_n_frames_per_state")
    ratio = _at("n_eff_over_g_per_state")
    eta = None
    if ne is not None and nin:
        try:
            eta = float(ne) / float(nin)
        except (TypeError, ValueError, ZeroDivisionError):
            eta = None
    return {
        "bottleneck_state": k,
        "bottleneck_g": None if g is None else float(g),
        "bottleneck_n_eff": None if ne is None else float(ne),
        "bottleneck_eta": eta,
        "bottleneck_ratio": None if ratio is None else float(ratio),
        "bottleneck_n_eff_input_n_frames": None if nin is None else int(nin),
        "n_eff_frame_set": selfchk.get("n_eff_frame_set"),
        "n_eff_frame_set_n_frames_raw": selfchk.get("n_eff_frame_set_n_frames_raw"),
    }


def stage_quality_gate_failures(stage):
    """stage 结果里**哪几道门没过**，逐条带实测值/阈值/科学分类/最差窗口。

    只读、纯函数。**fail-closed**：值或阈值读不出来时记成
    `unattributed` 失败，绝不当成「过了」—— 「缺证据 ≠ 通过」。

    ⚠️⚠️ [2026-09-15] **纯报告。** 每个条目带 `drives_action: False`，
    `report_category` 是这道门在物理上属于哪一类失效，**不是**"对症动作"。
    `decide()` 不读本函数的任何一条去发动作（见模块顶上那段长注释）。
    原来的 `remedy` 键已删除，不做同名改义 —— 谁还在读它，就让它 KeyError。
    """
    if not isinstance(stage, dict):
        return []
    out = []

    def add(gate, value, threshold, category, worst=None, note=None):
        out.append({
            "gate": gate, "value": value, "threshold": threshold,
            "report_category": category, "worst_window": worst, "note": note,
            # 🔑 这条诊断**不驱动任何动作**。两侧一致：`decide()` 只把整份清单
            # 原样放进返回值供人读，不按它分岔。
            "drives_action": False,
        })

    # 🔑🔑 [审计 #59，2026-09-14] **docstring 说 fail-closed，代码是 fail-open。**
    # 先前每道门都包在 `if <threshold> is not None:` 里 ⟹ **阈值读不出来就静默
    # 当成「这道门过了」**，与紧邻上面那句「缺证据 ≠ 通过」直接矛盾。
    # 而 NaN 更隐蔽：`float("nan") > x` 恒为 False ⟹ 一个 NaN 的端点 σ 会给出
    # `converged=False` + **零条归因** ⟹ `decide()` 落到分支 9d 的
    # `NO_FEASIBLE_ACTION`（「归因不到任何一道具体的门」），而真相是「这道门的
    # 读数是 NaN」。`ibs_engine` 算 `converged` 时显式 `np.isfinite(...)` 守过，
    # 这里没有 —— 同一道门两份实现，只有一份认得 NaN。
    def _num(x):
        """能比较的有限数 → float；None / 非数 / NaN / inf → None（= 没测出来）。"""
        if x is None or isinstance(x, bool):
            return None
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return v if bool(np.isfinite(v)) else None

    def add_numeric_gate(gate, raw_value, raw_threshold, category, worst_key=None,
                         *, largest_is_worse=False, fails_when_greater=False):
        """一道数值门的统一判读，**三态**：过 / 没过 / 没测出来。

        「没测出来」含**阈值缺失**与**值不是有限数**两种，一律记成
        `unattributed` 失败 —— 没有哪一类会改路由（本函数整体不驱动动作），
        但它让报告说出真话：「这道门没判成」而不是「没有任何门失败」。
        """
        # ⚠️ **「这道门不适用」与「这道门判不了」是两回事，别混。**
        # 值和阈值**都**不在 ⟹ 这份 stage 结果压根没带这道门（不同 stage / 旧产物
        # 都会这样）⟹ 跳过，不记失败。只报一条「阈值缺了」的 unattributed 会把
        # 每一份不带全部门的结果都污染成有失败，9d 的理由文本反而更难读。
        # 只要**有一侧在**，这道门就算在场 ⟹ 另一侧缺了就是「判不了」，记 unattributed。
        if raw_value is None and raw_threshold is None:
            return
        v, thr = _num(raw_value), _num(raw_threshold)
        if thr is None or v is None:
            add(gate, raw_value, raw_threshold, STAGE_GATE_UNATTRIBUTED,
                note=("阈值读不出来" if thr is None else
                      f"实测值不是有限数（{raw_value!r}）⟹ 这道门没判成，"
                      "**不得当成通过**"))
            return
        failed = (v > thr) if fails_when_greater else (v < thr)
        if failed:
            add(gate, v, thr, category,
                _worst_window_by(recs, worst_key, largest=largest_is_worse)
                if worst_key else None)

    # 0) 求解器自己就报了错（端点段帧数不够、拼接不上、pymbar 缺失……）。
    #    错误字符串里带 `decorrelated_samples` 的是样本量问题，其余不归类。
    err = stage.get("error")
    if err:
        add("solver_error", str(err), None,
            STAGE_GATE_MORE_SAMPLING if "decorrelated_samples" in str(err)
            else STAGE_GATE_UNATTRIBUTED)

    # 1) 求解器把窗口丢了（`len(local_results) == len(valid_windows)` 那一条）。
    #    被跳过的窗口有**上游专门的分支**在处理；走到这里说明那条没抓住，
    #    所以不归类。**这里不发任何动作** —— 本函数只报告。
    _in = stage.get("input_window_indices")
    _solved = stage.get("solved_window_indices")
    if isinstance(_in, list) and isinstance(_solved, list) and len(_solved) < len(_in):
        add("windows_dropped_by_solver",
            sorted(set(map(int, _in)) - set(map(int, _solved))), None,
            STAGE_GATE_UNATTRIBUTED)

    recs = stage.get("window_overlap_diagnostics") or []

    # 2) mixture 重叠：**跨度类**诊断（同分布加帧治不了）。仅报告。
    add_numeric_gate("min_overlap", stage.get("min_overlap"),
                     stage.get("min_overlap_threshold"),
                     STAGE_GATE_NARROW_SPAN, "min_ess_ratio")

    # 3) 物理目标态支撑度（raw ESS / top1% 权重集中度）：同样是**跨度类**。仅报告。
    tsg = stage.get("target_support_gate")
    if isinstance(tsg, dict) and tsg.get("passed") is False:
        # ⚠️ 最差窗口必须按**这道门自己判的那个量**挑。先前挑的是 `absolute_ess`
        # —— 那是去相关后的 mixture 覆盖度（共模已扣掉、且早已降级为纯诊断），
        # 与本门判的 raw 单参考量是两把尺子：实测同一批窗口 mixture 0.4684 /
        # raw 0.0196，排序可以完全不同 ⟹ 会把 λ 插到不是元凶的窗口上。
        # （TODO.md「四个量不许混用」说的就是这件事。）
        _ibs_gate = tsg.get("ibs_segment_gate") or {}
        # stage 级 `failed_checks` 只到「IBS 段/端点段」这一层（`ibs_segment_target_support`），
        # 具体是 raw ESS 还是 top1% 写在嵌套的 IBS 段门里 —— 两层都要看。
        _checks = [str(c) for c in ((tsg.get("failed_checks") or [])
                                    + (_ibs_gate.get("failed_checks") or []))]
        # 🔑🔑 [审计 #60，2026-09-14] **top1% 先前是「无条件」覆盖 raw ESS 的归因。**
        # 先前条件是 `any("top1pct" in c) or _worst is None` —— 只要失败清单里
        # 提到 top1%，就把 `worst_window` 换成 top1% 最大的那个窗口，**即使 raw ESS
        # 那条也失败、并且指了另一个窗口**。这与紧邻上面那段注释（「最差窗口必须按
        # 这道门自己判的那个量挑」「两把尺子排序可以完全不同」）自相矛盾：两条子门
        # 同时失败时，它单方面让 top1% 那把尺子赢。
        # 现在：raw ESS 那条失败且归因得出窗口时，**保留 raw ESS 的归因**（它是这道
        # 门的主判据）；只有 raw 那条没失败、或它归因不出窗口时才改用 top1%。
        # 两把尺子的读数**都**带进 note，归因信息不丢 —— 不合并、不加权、不排序。
        _raw_failed = any(("absolute_ess" in c) or ("raw" in c and "ess" in c.lower())
                          for c in _checks)
        _top1_failed = any("top1pct" in c for c in _checks)
        _worst_raw = _worst_window_by(recs, "raw_min_absolute_ess")
        _t1 = _worst_window_by(recs, "top1pct_raw_weight", largest=True)
        _worst, _worst_src = _worst_raw, "raw_min_absolute_ess"
        if ((_top1_failed and not _raw_failed) or _worst_raw is None) and _t1 is not None:
            _worst, _worst_src = _t1, "top1pct_raw_weight"
        add("target_support_gate", tsg.get("failed_checks"),
            {"raw_min_absolute_ess": tsg.get("raw_min_absolute_ess_threshold"),
             # 合并后的 stage 级门只带 raw ESS 那个阈值，top1% 的在 IBS 段的门里。
             "max_top1pct_raw_weight": (
                 tsg.get("max_top1pct_raw_weight_threshold")
                 or _ibs_gate.get("max_top1pct_raw_weight_threshold"))},
            STAGE_GATE_NARROW_SPAN, _worst,
            note={"worst_window_selected_by": _worst_src,
                  "worst_by_raw_min_absolute_ess": _worst,
                  "worst_by_top1pct_raw_weight": _t1,
                  "raw_ess_check_failed": _raw_failed,
                  "top1pct_check_failed": _top1_failed,
                  "do_not_merge": "两把尺子排序可以完全不同（实测 mixture 0.4684 / "
                                  "raw 0.0196），不得合并、加权或互相替代"})

    # 4) 去相关样本数：**样本量类**（语义是「尚不可测」而不是「不合格」）。仅报告。
    add_numeric_gate("min_decorrelated_samples",
                     stage.get("min_decorrelated_samples"),
                     stage.get("min_decorrelated_samples_threshold"),
                     STAGE_GATE_MORE_SAMPLING, "n_frames_decorrelated")

    # 5) 端点不确定度：**样本量类**，无逐窗归因。仅报告。
    #    ⚠️ 这道门的阈值（1.0 kJ/mol）抄自一个默认关闭的 early-stop 启发式的默认
    #    形参、全仓无依据，而且它是**逐段**量而目标是合成的 σ_bind ——
    #    **绝不得**再据此命令补帧（2026-09-15 定案，见模块顶上那段长注释）。
    add_numeric_gate("max_endpoint_uncertainty_kJ_mol",
                     stage.get("max_endpoint_uncertainty_kJ_mol"),
                     stage.get("max_endpoint_uncertainty_kJ_mol_threshold"),
                     STAGE_GATE_MORE_SAMPLING, None, fails_when_greater=True)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 出口注册表：**唯一**一张表，`EXITS` / `TERMINAL_EXITS` 全部从它派生。
#
# 🔑🔑 [2026-09-17，用户拍板] **`_RETIRABLE_EXITS` 删除，可退役性升成 plan 的一等语义。**
#
# 旧设计把**三个正交概念**混进字符串名单，分散在三张手写表里：
#   ① 为什么停（exit reason）
#   ② 停的是**谁**（窗口 / stage / 路径）
#   ③ 主循环终不终止（terminal）
# 必然继续漂。实证四次：
#   · `HALT_LAMBDA_BUDGET_INSUFFICIENT` 只在 `EXITS`、不在 `TERMINAL_EXITS`
#     ⟹「看起来像终止、实际不终止」（审计 #14）；
#   · `_RETIRABLE_EXITS` 漏了 `SUPPORT_ATTRIBUTION_UNKNOWN`（2026-09-17 补）；
#   · **还漏了 `D3_REWINDOW_DEPTH_EXHAUSTED`** —— 它同样是窗口局部卡死，
#     却会停掉一个**仍有其他工作**的 stage（用户 2026-09-17 抓出，本次热修）；
#   · `HALT_FRAMES_ADMISSION_CAP` 在两张表里的注释**互相矛盾**：`EXITS` 那边写
#     「刻意**不**进 `TERMINAL_EXITS`」，而它就在 `TERMINAL_EXITS` 里。
#
# 新契约：`plan()` 返回 `halt_scope` / `blocking_window` / `blocking_unit`，
# `_retirable_window()` **只看 scope、不认 exit 字符串**。
# 以后新增一个卡住原因，只在这张表里声明一次；漏了立刻报错，不会再静默停整跑。
#
#   TARGET_LOCAL  —— 当前窗口/单元无路，**允许换窗**（退役换窗的唯一入口）
#   STAGE_GLOBAL  —— 整条 stage 无路，换谁都没用
#   PATH_INVALID  —— 身份/布局/证据不成立，**不得绕过**
#                    （绕过去采下游 = 拿一条已知不成立的路径继续烧 GPU）
HALT_SCOPES = ("TARGET_LOCAL", "STAGE_GLOBAL", "PATH_INVALID")


class ExitSpec(NamedTuple):
    terminal: bool
    # `None` = **不给默认值，发出点必须显式传**。只有 `NO_FEASIBLE_ACTION` 是这样：
    # 它同时承载"这个窗口无路"和"整条 stage 无路"两种语义，给任何默认都会猜错一半。
    default_scope: Optional[str]
    why: str


EXIT_SPECS: Dict[str, ExitSpec] = {
    # ── 终态 ────────────────────────────────────────────────────────────────
    "DONE": ExitSpec(True, "STAGE_GLOBAL", "跨重复精度达标；单次 run 里恒不成立"),
    "DONE_UNTRUSTED": ExitSpec(True, "STAGE_GLOBAL", "门未过但调用方显式放行"),
    "ANALYSIS_COMPLETE_PRECISION_UNMEASURED": ExitSpec(
        True, "STAGE_GLOBAL",
        "硬不变量全过但精度从未被测过 ⟹ 这一跑没有别的事可做，但**不是 DONE**，"
        "结果不得作为已验收结果发布"),
    "GLOBAL_BUDGET_EXHAUSTED": ExitSpec(
        True, "STAGE_GLOBAL", "全局生产预算真的没了（**局部**耗尽不算）"),
    "HALT_FRAMES_ADMISSION_CAP": ExitSpec(
        True, "STAGE_GLOBAL",
        "**所有**窗口的补帧块数配额都用尽。它只在全窗满额时才发（部分满额走"
        "「剔掉满额的、只补剩下的」）⟹ 换谁都没用，是 stage 级。"
        "⚠️ 它先前在 `_RETIRABLE_EXITS` 里，那是**冗余且误导**：按定义此时没有"
        "别的窗口可退役到，退役判定的最后一道守卫必然返回 None。"),
    "HALT_LAMBDA_BUDGET_INSUFFICIENT": ExitSpec(
        True, "STAGE_GLOBAL",
        "插点预算耗尽 + 没有别的可行动作 ⟹ λ 总数不够是**输入问题**，交人工改输入"),
    "D3_REWINDOW_DEPTH_EXHAUSTED": ExitSpec(
        True, "TARGET_LOCAL",
        "有界重窗这条路对**这个单元**用尽（数据模型只能表达一次物理父窗替代）。"
        "🔑 [2026-09-17 热修] 它是**窗口局部**卡死，先前不在 `_RETIRABLE_EXITS` 里 ⟹ "
        "会停掉一个仍有其他工作的 stage。"),
    "SUPPORT_ATTRIBUTION_UNKNOWN": ExitSpec(
        True, "TARGET_LOCAL",
        "这个单元的失败分不出是样本量还是结构性 ⟹ 两边都不授权。"
        "别处还有活就绕过它继续调度（用户规格：不能让它停掉整跑）"),
    "HALT_INVALID_INPUT": ExitSpec(
        True, "PATH_INVALID",
        "身份/输入不一致 ⟹ 盘上的结论属于**另一个系综**，绕过去采下游无意义"),
    "HALT_EVIDENCE_CONTRADICTS_DONE": ExitSpec(
        True, "PATH_INVALID",
        "`DONE` 的证据不是 `CONVERGED` ⟹ 这份证据本身不成立"),
    "NO_FEASIBLE_ACTION": ExitSpec(
        True, None,
        "**没有默认 scope**：它同时承载「这个窗口无路」（可换窗）与「整条 stage "
        "无路」（换谁都没用）。每个发出点必须显式传 `halt_scope=`，漏传 `plan()` 抛错 —— "
        "静默按某个默认处理，要么停掉还有活干的 stage、要么绕过一条不成立的路径。"),
    # ── 路由信号（**必须**被主循环消费，不得终止）────────────────────────────
    "HALT_BUDGET": ExitSpec(False, None, "全局预算耗尽但动作本身可行 ⟹ 换动作继续"),
    "HALT_LOCAL_VALIDATION_CAP": ExitSpec(
        False, None, "撞单周期验证**批次**上限，而全局预算还有钱（≠ HALT_BUDGET）"),
    "HALT_VALIDATION_BUDGET_UNREACHABLE": ExitSpec(
        False, None,
        "算术上证明剩余预算内凑不够去相关帧数。**只改路由、不改 verdict** —— "
        "证据仍是 UNMEASURED，绝不变成 REJECTED"),
    "HALT_NO_ATTRIBUTION": ExitSpec(
        False, None, "测不动且归因不出来 —— stage 级「没有任何门失败」，合法结局"),
    "HALT_FK_REFUTED": ExitSpec(
        False, None,
        "f_k 被有统计功效地驳回 ⟹ 换 Epoch。**是路由**：发它的分支给的是一个"
        "**要执行的动作**（`RECALIBRATE_FK`），进终态集会让那次重标定根本不执行"),
}


def classify_layout_evidence(segments, path) -> Dict[str, Any]:
    """按**当前布局**给逐段窗口证据分类。两个视图构造器共用的唯一实现。

    🔑🔑 [2026-09-17] **先前这套判定只活在 `read_aggregated()` 的一个闭包里**
    （`_layout_matches`），于是单段视图（直接构造 `Stage2RepairController(run, stage)`）
    产不出 `out_of_range_windows` / `unverifiable_layout_evidence` /
    `window_provenance` / `aggregated_segments` 四个键，而：
      · `_coverage_incomplete()` 读 `out_of_range_windows` ⟹ 单段上「证据里混进了
        当前布局根本没有的窗口号」这道闸**恒不触发**；
      · 单段的 `stale_layout_evidence` 是按 `stale_layout_evidence_only` 标记算的，
        而那个标记**只有聚合视图的占位记录才会设** ⟹ 单段上它恒为 `{}`，
        「证据被布局变更作废」这道闸同样恒不触发。
    受影响的是直接构造器 / `decision_trace()` / 非 `--replay` 的 CLI 路径 / 离线脚本；
    `replay()` 走 `for_physical_stage()`，不受影响。

    ⚠️ **不能靠给单段补四个空默认值了事** —— 那只让 schema 表面一致，守卫照样静默
    失效。判定逻辑必须是同一份，否则两边规则还会再漂一次。

    `segments`：`[(段名, [窗口记录, ...]), ...]`，段号升序。
    `path`：`_read_path()` 的返回（要 `lambdas_vdw` / `window_ranges`）。

    四态（顺序有意义，照抄原闭包）：
      · `OUT_OF_RANGE`  当前布局里根本没有这个下标（旧布局遗留产物）。
        **越界检查不依赖 λ 值**，只要有 `window_ranges` 就判得了，所以排在最前 ——
        放到 `not cur_lam` 之后会被它整个吞掉（老产物最常见的形状就是没落 λ 值）。
      · `UNCHECKABLE_NO_CURRENT_LAMBDAS`  连当前 λ 表都没有，无从核对。不记账。
      · `UNVERIFIABLE`  证据自己没带 λ 列表 ⟹ 核不了（≠「核对过、不匹配」）。
      · `STALE`         核对过、与当前布局不符。
      · `MATCH`         身份一致。
    ⚠️ 只有 `STALE` 表示"这份证据不能用"；另外三种**照旧参与合并**（判死会重现
    2026-09-15 那次 win4 占位记录死锁），但必须**记下来**让下游有得判。
    """
    cur_lam = list((path or {}).get("lambdas_vdw") or [])
    cur_rng = [tuple(r) for r in ((path or {}).get("window_ranges") or [])]
    verdict: Dict[Tuple[str, int], str] = {}
    stale: Dict[int, List[str]] = {}
    unverifiable: Dict[int, List[str]] = {}
    out_of_range: Dict[int, List[str]] = {}
    for seg, wins in segments:
        for w in (wins or []):
            i = int(w["window_idx"])
            if cur_rng and i >= len(cur_rng):
                out_of_range.setdefault(i, []).append(seg)
                verdict[(seg, i)] = "OUT_OF_RANGE"
                continue
            if not cur_lam:
                verdict[(seg, i)] = "UNCHECKABLE_NO_CURRENT_LAMBDAS"
                continue
            got = w.get("lambdas_vdw")
            if not got:
                unverifiable.setdefault(i, []).append(seg)
                verdict[(seg, i)] = "UNVERIFIABLE"
                continue
            a, b = cur_rng[i]
            want = cur_lam[a:b]
            if len(want) != len(got):
                stale.setdefault(i, []).append(seg)
                verdict[(seg, i)] = "STALE"
                continue
            # 🔑 **必须跟引擎用同一个栅格。** 先前某版写 `abs(x-y) <= 1e-9`，
            # 而 `ibs_engine` 判 λ 身份是量化到 LAMBDA_GRID_DECIMALS 位后精确比较；
            # 路径落盘量化 / 采样未量化之间天然差 5e-9 ⟹ 引擎说"匹配"、这里说"过期"。
            from ibs_engine import LAMBDA_GRID_DECIMALS as _GD
            _q = lambda vs: [round(float(x), _GD) + 0.0 for x in vs]   # noqa: E731
            if _q(want) != _q(got):
                stale.setdefault(i, []).append(seg)
                verdict[(seg, i)] = "STALE"
                continue
            verdict[(seg, i)] = "MATCH"
    return {
        "verdict_by": verdict,
        "stale": stale,
        "unverifiable": unverifiable,
        "out_of_range": out_of_range,
    }


class Stage2RepairController:
    """Stage-2 的统一控制器：**看状态 → 选一个动作 → 交执行器 → 回来重新判断。**

    设计依据：``docs/archive/PLAN_PATH_REPAIR_2026-09-11.md``（定位、三类修补、判别、可行性、
    停止标准）。放在 `abfe_preoptimizer` 里、而且是**一个类**，是刻意的：

      · **决策不许散开。** 现状是三层各用一种机制 —— `ibs_engine.run_all_windows`
        尾部 4200 行函数里的 if/else（窗口内 f_k）、`abfe_pipeline` 的异常展开
        （插 λ）、`abfe_pipeline:13027` 的 while 循环（补采）；状态写点 25 处、
        分三个区，其中 resume 那一段是**第二套**从盘上推状态的规则。本类把这些
        判断收到一处。
      · 修补的判据（`insert_lambda_in_failed_ibs_window`、`feasible_repair_actions`）
        本来就在本模块，决策跟它们同处一室，不必跨文件对齐。
      · **状态唯一真源是盘。** 本类只读盘 + 纯判断，**不执行任何动作**；
        唯一的例外是 `write_comparison_manifest()`（把纯聚合结果落一份盘），
        它不改任何**决策输入**。[审计 #63] 原话「不改任何文件」与该方法直接矛盾，
        照字面读会让人以为控制器绝不写盘、从而放心地并发调用它。
        于是 resume 就等于"拿同一份盘上状态再 new 一次、再 decide 一次"，结构上
        不可能长出第二套状态重建规则。

    **本类不做**：任何数值计算。逐态 ESS / g 剖面、mixture 覆盖度这些量已经由
    `ibs_engine` 落在 `*_convergence.json` / stage 结果里；这里只读汇总值。需要
    重新算数的诊断另开脚本，别塞进控制器 —— 控制器一旦自己算数，就没法"拿历史
    run 的产物离线重放决策"了。
    """

    # 动作：执行器能做的事。分两类 —— 不改结构的（前三个）与改结构的（后两个）。
    ACTIONS = (
        "CONTINUE_WARMUP",      # 继续学习 / 继续验证（同一个 Epoch，加预算）
        "RUN_PRODUCTION",       # 跑/补生产帧（同一个 f_k，接着原段）
        "RECALIBRATE_FK",       # 用生产帧重解 f_k → 新 Epoch，旧段保留
        "INSERT_LAMBDA",        # 补 λ 缩窗跨度（model B；溢出落末窗）
        "SPLIT_TAIL_WINDOW",    # 从 anchor 起重分**整个尾段**（需 tail anchor）
        # 🗑️ [S2-J，2026-09-17] 这里曾加过一个 `SPLIT_LAST_WINDOW_IN_TWO` 动作，**已撤**。
        # 「末窗一分为二」**不需要新动作** —— `abfe_pipeline._legalize_tail_window()`
        # 早就完整实现了它，包括死区那一步：
        #     split_lo = 2*lo − 1
        #     if tail_k <= hi: return                  # ← 没超 hi 就不管
        #     while hi < tail_k < split_lo: 补 1 个 λ   # 死区：补到可拆
        #     然后按中点拆成两个共享边界态的子窗
        # 它之所以在真机上"从没发生过"，**原因是配置 `hi=8` 而不是缺动作**：
        #     hi=8 ⟹ split_lo=7 ⟹ 死区 (8,7) 是**空集**，且 `K ≤ 8` 直接 return
        #            ⟹ K=6/7/8 全是"合法单窗"，末窗永远不拆（= 缺口 S2-I）
        #     hi=5 ⟹ 死区 = {6} ⟹ 末窗 K=6 → 补 1 → K=7 → 唯一拆法 4+4 ✅
        # ⟹ 正解是**配置 `stage2_window_max_states = 5`**，不是往控制器里加第二份实现。
        # 加动作等于「同一个不变量两份实现」——本仓最贵的那个形状。
        "PROBE_CANDIDATE_FK",   # **非变异**：离线算候选 f_k + 评估，不切换
        "PROBE_REANCHOR_EPOCH", # held-out **判不了**时：候选 f_k + 独立 burn-in + 一块
        # 🔑 **与 RECALIBRATE_FK 科学语义不同，绝不合并。**
        # RECALIBRATE_FK 是拿**已有生产帧**重解 f_k（信息来自旧轨迹）；
        # RELEARN_FK_EPOCH 是**从头 LEARN**一份全新候选（fresh learn → 新 f_k →
        # burn-in → 独立 held-out 验证），因为旧那份已经被统计驳回、永不续验。
        "RELEARN_FK_EPOCH",     # 候选被驳回/不可达 ⟹ 全新 Epoch 从头学一份 f_k
        # （"候选也救不了 ⟹ 布局动作" 用的就是上面那个 `INSERT_LAMBDA`，
        #   2026-09-13 前这里重复列了一次，同一个词在清单里出现两遍。）
        # [2026-09-14 裁决] 固定 λ 表上的**有界**重窗：一次只修一个被归因的窗口，
        # 建可完整覆盖其原区间的重叠子系综，只采一个预算块，然后立刻
        # 读盘 → 求解 → 读回求解器帧数与跳窗 → 回到 decide()。
        # 它**不等同于** INSERT_LAMBDA / SPLIT_TAIL_WINDOW：那两个修不了固定 λ 表
        # 下的中间窗。也**不是**已退役的 bridge rescue（批量建系综、一次重解、
        # 直接撞门）—— 那条不许搬回来。
        "IMMUTABLE_REWINDOW",
        # 🔑 [2026-09-15] 单周期验证**批次上限**打满、Δf−ΔF 从未被求出（无结论、
        # **未被驳回**）⟹ 用这份冻结 f_k 跑**一块**诊断生产。
        # **与 RUN_PRODUCTION 绝不合并**，两点不同：
        #   · RUN_PRODUCTION 的前提是这个窗口的 f_k **已经通过**冻结验证；
        #     这个动作恰恰相反 —— f_k 从未被验证过，执行器必须显式授权引擎
        #     （`provisional_production_windows`），否则引擎会抛路由信号。
        #   · 它的产物 `bias_status=provisional_production` / 证据仍是
        #     `indeterminate`，**不是可信 PASS**；warmup_failure.json 保留为证据。
        # 没有这个动作时，这样的窗口进不了生产、又拿不到新证据 ⟹ 整跑停在
        # NO_FEASIBLE_ACTION（真机 cyclod_ligand2/rep2 win4）。
        "PROVISIONAL_PRODUCTION",
        "ANALYZE",              # 只读：跑 stage 分析（MBAR + 生产质量门）
        "DONE",
        # 🔑 [2026-09-13] **「没有可做的动作」必须有自己的词**，不能借 `DONE`。
        # `DONE` 的定义是「全路径证据齐备且达标」（设计 §2）；归因成功但动作都不可行、
        # 或干脆归因不出来时，结果**没有达标**，写 `DONE` 既与 §2 冲突，又会让
        # `plan()` 把 `execution_status` 算成 `COMPLETE`（它按 `action == "DONE"` 判）。
        # 配 `exit_="NO_FEASIBLE_ACTION"` 使用 ⟹ terminal=True、execution_status=HALTED。
        # 终态在主循环里**先于**执行器分发被 break，所以执行器不需要认识它。
        "NO_ACTION",
    )
    # 🔑🔑 [2026-09-11 老板改目标] **控制器是驱动，不是影子。**
    # 「控制器必须自己诊断、自己换动作、自己继续跑，最终自动产出结果。
    #   任何 LOCAL_* 都必须被主循环消费，**不能炸出流水线**。」
    #
    #     while not DONE:
    #         evidence = read()
    #         action   = decide(evidence, feasible_actions, global_budget)
    #         execute(action)
    #
    # ⟹ 出口要分成两类。**只有这三种允许真正终止**，其余一律是**路由信号**
    #    （被主循环消费、换个动作继续跑）：
    # 🔑🔑 [2026-09-17，用户拍板] **两张表都从 `EXIT_SPECS` 派生，不再手写。**
    # 手写两张表的代价已经实证四次（漏项、互相矛盾的注释、静默停整跑）——
    # 理由与逐个出口的语义全部写在模块级 `EXIT_SPECS` 那张表里，只此一份。
    EXITS = tuple(EXIT_SPECS)
    TERMINAL_EXITS = tuple(k for k, v in EXIT_SPECS.items() if v.terminal)
    # 🗑️ [2026-09-13 清理] 删掉两个**从未被发出过**的死词：
    #   · `HALT_NO_FEASIBLE_ACTION` —— 真正在用的是 `NO_FEASIBLE_ACTION`（无 HALT_
    #     前缀）。它不只是冗余：**它不在 `TERMINAL_EXITS` 里**，谁照着这份清单发它，
    #     `terminal` 就不成立、主循环不 break —— 一个看起来像终止、实际不终止的词。
    #   · `HALT_TRUNCATED_PATH` —— 缺窗口现在走 `decide()` 分支 6b（`RUN_PRODUCTION`，
    #     无出口、是路由），因为「缺窗口」的正确位置在因果顺序**之后**而不是终止。
    # 两者全仓（含 docs/）零引用，删除是纯清理。由
    # `tests/test_stage2_controller_vocabulary.py` 钉住不再漂移。

    # `bias_status` 六个值混了"还在流程中"（前三）与"已有裁决"（后三）。
    # 拆开才知道是"在跑"还是"有结论了"。
    _PHASE = {
        "unconverged": "WARMUP_LEARN",
        "calibrated_pending_validation": "WARMUP_VALIDATE",
        "frozen_validation_indeterminate": "WARMUP_VALIDATE",
        "converged": "PRODUCTION",
        # 临时生产：已经在生产里（Epoch 已花掉、产物在盘上），但 f_k 从未验证过。
        # 强度差异由 `f_k_evidence_status`（= indeterminate）表达，不在这里混。
        "provisional_production": "PRODUCTION",
        "failed": "TERMINAL",
        "calibrated_validation_failed": "TERMINAL",
    }
    # f_k 证据 → verdict 词汇表。**三值不够**：只有 PASS/FAIL 时，"FAIL 但有预算
    # → 再测一次"会退化成重试到碰巧通过。有统计功效的否决必须立刻换 Epoch。
    # ⚠️ 低支撑**永远不是 FAIL**。"尚不可测"（INSUFFICIENT_DATA / HARD_INSUFFICIENT）
    # 加预算；FAIL（STATISTICALLY_REJECTED）换 Epoch。混起来会退化成"再测一次直到
    # 碰巧通过"。`ANALYSIS_ELIGIBLE` 只表示**可以进入分析**，不是通过验收 ——
    # 最终 PASS 还要 endpoint CI、block 稳定性、全路径完整性。
    _VERDICT = {
        "verified": "VALID_PASS",
        "calibrated": "INSUFFICIENT_DATA",
        "indeterminate": "INSUFFICIENT_DATA",
        "refuted": "STATISTICALLY_REJECTED",
        "none": "INSUFFICIENT_DATA",
    }

    def __init__(
        self,
        run_dir: str,
        stage_name: str = "vanishing",
        stage_type: str = "vdw",
        *,
        min_states_per_window: Optional[int] = None,
        max_states_per_window: Optional[int] = None,
        max_path_insertions: Optional[int] = None,
        allow_untrusted_stage_results: bool = False,
        effective_config: Optional[Dict[str, Any]] = None,
    ):
        self.run_dir = os.path.abspath(run_dir)
        self.stage_name = str(stage_name)
        self.stage_type = str(stage_type)
        # 🔑 lo/hi **默认从 run 自己的 run_provenance.json 读**，不要求调用方记得传。
        # 手动传错的后果很实在：可拆区间是 [2lo−1, 2hi−1]，4/5 是 7..9、4/8 是
        # 7..15，判出来的"可不可行"会完全不同。run 自己记了它跑的是什么，就用那个。
        #
        # 🔑🔑 [2026-09-15 审计 #1] **但"run 自己记了"这件事只对复合物腿成立。**
        # `_write_run_provenance` 只往**总**输出目录写一份 `run_provenance.json`，
        # 而溶剂腿的 run_dir 是 `output_dir/solvent_leg`（`runabfe.py` 约 7515），
        # 那底下没有这个文件 ⟹ `_cfg` 为空 ⟹ 分窗判据、插点上限、块数上限、
        # 生产预算**全部退回默认值**，与复合物腿不是同一套。真机实测两腿读出：
        #     分窗判据 metric_integral vs arclength
        #     每块生产步数 500000 vs 250000
        #     生产总预算 2000000 vs 未知
        #     插点上限 8 vs 3；每窗块数上限 7 vs 4
        # 同一次 run 的两条腿用不同预算和不同修复策略，而这个差异只在日志里
        # 露出一行 `config_source`。
        #
        # 修法：调用方**显式传**解析后的有效配置（`effective_config`），它优先于
        # 盘上那份。不传时行为逐字不变（仍读 provenance）—— 离线 replay 与旧
        # 产物不受影响。绝不改成"去父目录找一找"：那是靠目录结构猜配置，
        # 换个布局就又静默错一次。
        # 🔑 **合并，不是替换。** 三层优先级：调用方显式给的 > run 自己记的 >
        # 读侧默认值。
        # ⚠️ 写成"给了 effective_config 就不读 provenance"会**倒退**：调用方只传
        # 得出它手上有的那几个键，而 provenance 里可能记着更多（复合物腿就是），
        # 于是没被显式传的键从"provenance 里的真值"掉回"默认值" —— 修溶剂腿的洞
        # 反而把复合物腿弄降级。
        # ⚠️ 值为 `None` 的键**不算给过**：读侧一律 `_cfg.get(k, <默认>)`，而
        # `{"k": None}.get("k", 4)` 返回 None 不是 4。「未知」用**缺键**表达。
        # 🔑 [2026-09-16] **两边都要滤 None，不是只滤调用方那半边。**
        # 原来只有 `_explicit` 滤了，而 `run_provenance.json` 的 config 是
        # `argparse` 的完整命名空间落盘的 —— 未给的开关一律记成 `null`。真机
        # 13 个 run 的 provenance 每一份都带 `stage2_production_budget_steps: null`
        # 等 17~18 个 None 键。今天不炸只是因为那几个键恰好没走 `int()`；
        # `stage2_max_production_blocks_per_window` 走了 `int()`，一旦它哪天以
        # `null` 进 provenance 就是同一个 `TypeError: int(None)`（09-15 已踩过一次）。
        # 「未知」只有一种表达：**缺键**。
        _disk = {
            k: v for k, v in (
                (self._json(os.path.join(self.run_dir, "run_provenance.json")) or {})
                .get("config") or {}
            ).items()
            if v is not None
        }
        _explicit = {k: v for k, v in (effective_config or {}).items()
                     if v is not None}
        _cfg = dict(_disk)
        _cfg.update(_explicit)
        # 🔑🔑 [2026-09-15] **留着它，因为 `read_aggregated()` 要造子控制器。**
        # 段级子控制器不带这份配置的话，合并视图的生产账是从**子控制器**的
        # `_production_budget_inputs` 生成的（见 read_aggregated 的
        # `production_budget`）⟹ cap/块大小又退回默认，父控制器收到什么都没用。
        # 实测：父 cap=2,000,000 → 视图 cap_known=False；父块 500k → 视图 250k。
        self._explicit_config = dict(_explicit)
        self.config_source = (
            "caller:effective_config+run_provenance.json"
            if (_explicit and _disk) else
            "caller:effective_config" if _explicit else
            "run_provenance.json" if _disk else "caller/default"
        )
        self.lo = int(min_states_per_window if min_states_per_window is not None
                      else _cfg.get("stage2_window_min_states", 4))
        self.hi = int(max_states_per_window if max_states_per_window is not None
                      else _cfg.get("stage2_window_max_states", 5))
        self.max_path_insertions = int(
            max_path_insertions if max_path_insertions is not None
            else _cfg.get("max_path_insertions", 3)
        )
        # 🔑 [2026-09-15] 尾段重分必须用**与生产同一个**分窗判据；从 run 自己记的
        # config 读，缺省与 `_run_dual_lambda_stage` 那侧的缺省一致。
        self.partition_criterion = str(
            _cfg.get("stage2_window_partition", "arclength")
        ).lower()
        self.allow_untrusted = bool(allow_untrusted_stage_results)
        # 🔑🔑 [裁决 2，2026-09-14] **生产补帧预算是独立的一本账。**
        # 先前控制器只有 `warmup_steps_left`（`per_window_budget_remaining` 也是它），
        # 于是「还能不能补帧」根本判不出来，只好拿预热余额去终止一个只花生产帧的
        # 动作。现在上限从 config 读；**读不到就是 unknown，不是 0，也不许用预热
        # 余额代替** —— 上限未知时"预算耗尽"这个结论本身不成立。
        # 🔑 [审计 #31] **这两个键的缺省语义（读侧口径，写侧由 CLI/config 提供）：**
        #   · `stage2_production_budget_steps` 缺省 = **unknown**（不是 0、不是无限）
        #     ⟹ `cap_known=False` ⟹ `plan()` 的生产预算闸**不拦**。
        #     「预算耗尽」是个断言，上限未知时这个结论本身不成立。
        #   · `stage2_max_production_blocks_per_window` 缺省 = **4**，而且
        #     **必须有界** —— cap 缺省既然是 unknown，块数上限就是唯一的刹车；
        #     默认无限正是 BUD-03 要修的那个行为（真机单窗烧到 150 万步）。
        _cap = _cfg.get("stage2_production_budget_steps")
        self.production_cap_steps = (
            int(_cap) if isinstance(_cap, (int, float)) and int(_cap) > 0 else None
        )
        self.production_cap_source = (
            "config:stage2_production_budget_steps"
            if self.production_cap_steps is not None else "unknown"
        )
        # 🔑 **cap 的范围定死为「生产步数」**（名字就叫 production）。
        # 先前建窗准入拿「预热 + 生产」去比这个 cap、完成后却只计生产，
        # 计量范围两头不一致 ⟹ 准入比实际严、账又对不上。
        # 预热/验证走**另一本账**（`warmup_steps_left`），不进这里。
        self.production_block_steps = int(_cfg.get("n_steps_per_window", 250_000))
        # 🔑 **每个采样单元的补帧块数硬上限。** 没有它，补帧就没有停止条件
        # （真机走到过 150 万步）。config 没给就用一个**有界**的默认值 ——
        # 绝不能默认无限，那正是要修的那个行为。
        self.max_blocks_per_window = max(1, int(
            _cfg.get("stage2_max_production_blocks_per_window", 4)))
        # 新系综（rewindow 子窗）要预留的生产额度：**每个子窗一个首块**。
        self.new_ensemble_reserve_steps = int(self.production_block_steps)
        self.stage_dir = os.path.join(self.run_dir, self.stage_name)
        # 🔑 **多采样段有自己的 checkpoint 命名空间。** 约定见
        # `abfe_pipeline._recalibrate_fk_and_resample_segment`：
        #   段输出目录 = f"{stage_dir}_{segment_index}"       → `vanishing_2`
        #   段 checkpoint = checkpoints/f"segment_{index}"     → `checkpoints/segment_2`
        # 读错子目录的后果很具体：段 2 会读到段 1 的 ibs_state，把段 1 的
        # frozen_validation_cumulative_steps（实测 205000）当成段 2 的
        # （真值 0）报出来。路径版本链仍在顶层 checkpoints（λ 路径是全局的）。
        _base_ckpt = os.path.join(self.run_dir, "checkpoints")
        _m = re.match(r"^(.+)_(\d+)$", self.stage_name)
        self.segment_index = int(_m.group(2)) if _m else None
        self.checkpoint_dir = (
            os.path.join(_base_ckpt, f"segment_{self.segment_index}")
            if self.segment_index is not None else _base_ckpt
        )
        self.path_checkpoint_dir = _base_ckpt

    # ---------------------------------------------------------------- 读盘

    @staticmethod
    def _json(path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def _read_path(self) -> Dict[str, Any]:
        """λ 路径版本 + 演化事件计数。事件必须数**整条链**，那才是跨 resume 的
        累计轮数（`_run_stage2_with_path_evolution` 的 rounds_done 同口径）。"""
        # λ 路径是**全局**的（不分段），版本链永远在顶层 checkpoints。
        _pd = self.path_checkpoint_dir
        cur = self._json(os.path.join(_pd, "path_current.json")) or {}
        version = cur.get("version")
        record = None
        if version is not None:
            record = self._json(
                os.path.join(_pd, "path_versions", f"v{int(version)}.json")
            )
        events: Dict[str, int] = {}
        # 🔑 [审计 #57，2026-09-14] **事件要沿祖先链数，不是 glob 数所有版本文件。**
        # 先前 `glob("path_versions/v*.json")` 把**孤儿版本**（写了文件但没推进
        # `path_current.json` 指针 —— 那是设计内的正常残留）也算进插点预算，于是
        # 预算被永久占用、插 λ 提前判不可行；而且绕过了 `lambda_path_versions._validate()`
        # 的 `content_sha256` 校验（读到被改过的版本文件也不会发现）。
        # docstring 自己写的就是「必须数**整条链**」，实现与它相反。
        # 链式回溯已经有权威实现（`lambda_path_versions.history()`，带环检测 +
        # `_validate`），直接用它，不在这里再写一遍。
        try:
            import lambda_path_versions as _lpv_events
            for _rec in _lpv_events.history(_pd):
                kind = (_rec.get("event") or {}).get("kind")
                if kind:
                    events[kind] = events.get(kind, 0) + 1
        except Exception:  # noqa: BLE001
            # 读不到 / 链损坏 ⟹ 退回逐文件扫描（保守：宁可多算也不少算插点数，
            # 少算会让插点预算失效，那正是原来的失败形状）。
            for vf in sorted(glob.glob(os.path.join(_pd, "path_versions", "v*.json"))):
                # 🔑 kind 在 `record["event"]["kind"]`，不在顶层。读错键 ⟹ `events`
                # 恒为 {} ⟹ `path_insertions_left` 恒等于满额预算。
                kind = ((self._json(vf) or {}).get("event") or {}).get("kind")
                if kind:
                    events[kind] = events.get(kind, 0) + 1
        ranges = (record or {}).get("window_ranges")
        return {
            "version": version,
            "events": events,
            "n_states": len((record or {}).get("states", [])) or None,
            "window_ranges": [tuple(int(x) for x in r) for r in ranges] if ranges else None,
            # 当前布局的 λ 表。判"某份证据是不是这个布局产出的"要靠它。
            "lambdas_vdw": [
                float(st.get("lambda_vdw"))
                for st in ((record or {}).get("states") or [])
                if st.get("lambda_vdw") is not None
            ] or None,
        }

    def layout_change_invalidated_segments(self) -> List[str]:
        """布局变更**之前**就已经存在的采样段 —— 它们描述的是另一个窗口几何。

        🔑 [2026-09-17] 与 `classify_layout_evidence` 同一个理由抽出来：先前它只在
        `read_aggregated()` 里算，单段视图没有这个键。它是**全局事实**（读路径版本
        链上当前那条事件），跟合并没有关系，两个视图构造器算出来必须一样。

        真机 15:59 实证过为什么需要它：插 λ 之后旧段的 min N_eff/g 描述的是另一个
        跨度，拿它判「加帧有没有用」会让判据读出一个不存在的趋势 —— 那次连插两次
        把末窗顶到 K=6 越界炸掉。
        """
        try:
            import lambda_path_versions as _lpv_inv
            _ev = (_lpv_inv.load_current(self.path_checkpoint_dir)
                   or {}).get("event") or {}
            if _ev.get("kind") not in ("insert_lambda", "tail_repartition"):
                return []
            # `note` 是 2026-09-14 之后的落点（不进 event_id）；`detail` 是之前的
            # 旧记录，必须继续读得到，否则老 checkpoint 的过期段保护会静默失效。
            return sorted({
                str(x) for x in (
                    (_ev.get("note") or {}).get("segments_before_change")
                    or (_ev.get("detail") or {}).get("segments_before_change")
                    or [])
            })
        except Exception:  # noqa: BLE001 —— 读不到就当没有变更过，与原实现一致
            return []

    def comparison_manifest(self, view=None) -> Dict[str, Any]:
        """把 A/B 对比要用的量**拍平到一份**。纯聚合，不新算任何东西。

        动机：outer-λ 增强采样要跟 baseline 对照，而现在这些数散在
        6 个窗口 × 多个采样段的几十个文件里（`*_self_support.json` /
        `*_convergence.json` / `dual_join_*` / 版本链 / 自治历史 / 探针）。
        对比时一个个去捞既慢又容易捞错段（段号 ≠ 新旧，见 read_aggregated 注释）。

        三块内容，缺一不可：
          · **身份** —— 这是哪条臂、哪套 λ、哪些协议版本。不钉住身份的对比是空的。
          · **验收量** —— 逐窗 min N_eff/g 及其逐态剖面、ΔF±σ、门的结论。
          · **代价** —— 步数与开过几个 Epoch。outer 臂每步更贵，
            只比 ΔG 不比代价等于没比。
        """
        view = view if view is not None else self.read()
        path = self._read_path()
        stage = self._read_stage_result() or {}
        hist = self._json(os.path.join(
            self.path_checkpoint_dir, "stage2_autonomous_history.json")) or {}
        probe = view.get("fk_probe") or {}

        windows = []
        total_prod = 0
        # 🔑 [2026-09-14] **读不到步数的窗口不按 0 步计。**
        # 先前 `int(... or 0)` 把未知压成 0 再累加 ⟹ A/B 的代价栏系统性偏低，
        # 而代价栏是本函数 docstring 里点名的三大必看之一
        # （「只比 ΔG 不比代价等于没比」）。未知不参与求和，并显式报出有几个未知。
        n_unknown_prod = 0
        # 🔑🔑 [2026-09-16] **"无结论"必须和"不存在"分开。**
        # 固定预算模式（自治控制器关闭）下，冻结验证在单周期批数上限内始终求不出
        # Δf−ΔF 的窗口被记成 `indeterminate`（见 abfe_pipeline `_guarded_once`）。
        # 它**没有** self_support 产物（生产就没跑完）⟹ 下面每个验收量都是 None
        # ⟹ 报告里显示成 `·`，和"这个窗口根本不在布局里"长得一模一样。
        # 那正是"静默从配对分析里删掉"，是明令禁止的。所以这里把 stage 结果里的
        # `indeterminate_windows` 读出来，逐窗打标 + 带上原因。
        _indet = {}
        for _x in (stage.get("indeterminate_windows") or []):
            try:
                _indet[int(_x["window_idx"])] = _x
            except (KeyError, TypeError, ValueError):
                continue
        for w in view.get("windows") or []:
            i = int(w["window_idx"])
            prod = w.get("production_steps")
            if prod is None:
                n_unknown_prod += 1
            else:
                total_prod += int(prod)
            windows.append({
                "window_idx": i,
                "segment": w.get("segment"),
                "lambdas_vdw": w.get("lambdas_vdw"),
                "lambda_span": w.get("lambda_span"),
                "n_states": w.get("n_states"),
                # —— 验收量 ——
                "verdict": w.get("self_verdict"),
                "min_n_eff_over_g": w.get("min_n_eff_over_g"),
                "worst_state_by_n_eff": w.get("worst_state_by_n_eff"),
                "n_decorrelated": w.get("self_n_frames_decorrelated"),
                "min_frames_floor": w.get("self_min_frames"),
                "solver_eligible": w.get("self_sufficient"),
                "verdict_source": w.get("self_verdict_source"),
                # None = 有结论（通过或不通过）；非 None = **这个窗口没测出来**。
                # 下游必须据此拒绝为它计算任何 A/B 差值，也不得把它当 0 或失败。
                "indeterminate": (
                    None if i not in _indet
                    else {"reason": (_indet[i] or {}).get(
                              "reason", "frozen_validation_budget_indeterminate"),
                          "detail": str((_indet[i] or {}).get("detail") or "")[:400]}),
                # 判据量是**块内独立 ESS**，不是前缀差（见 `_read_window` 的长注释）。
                "block_local_ess_by_block": w.get("block_local_ess_by_block"),
                "derailment_status": w.get("derailment_status"),
                "derail_at_trajectory_fraction": w.get("derail_at_trajectory_fraction"),
                "cumulative_fk_residual_span_kJ_mol": w.get("cum_fk_span"),
                "cumulative_fk_residual_verdict": w.get("cum_fk_verdict"),
                # —— 代价 ——
                "production_steps": prod,
                "n_frames": w.get("n_frames"),
                "warmup_steps_spent": w.get("warmup_steps_spent"),
                "warmup_steps_cap": w.get("warmup_steps_cap"),
            })

        actions = [
            {"iteration": it.get("iteration"), "action": it.get("action"),
             "windows": it.get("windows"), "exit": it.get("exit"),
             "routing_signal": it.get("routing_signal")}
            for it in (hist.get("iterations") or [])
        ]
        return {
            "manifest_version": 1,
            "generated_for": "baseline_vs_outer_lambda_ab_comparison",
            # ── 身份：不钉住这些，两条臂的数字不可比 ──
            "identity": {
                "run_dir": self.run_dir,
                "stage": getattr(self, "_stage_base", self.stage_name),
                "stage_type": self.stage_type,
                "path_version": path.get("version"),
                "path_events": path.get("events"),
                "n_states": path.get("n_states"),
                "lambdas_vdw": path.get("lambdas_vdw"),
                "window_ranges": [list(r) for r in (path.get("window_ranges") or [])],
                "aggregated_segments": view.get("aggregated_segments"),
                "min_states_per_window": self.lo,
                "max_states_per_window": self.hi,
            },
            # ── 路径级结果 ──
            "path_result": {
                "total_delta_G_kJ_mol": stage.get("total_delta_G"),
                "total_error_kJ_mol": stage.get("total_error"),
                # [2026-09-15] `converged` 已删除（不做兼容别名）。导出新契约两维，
                # 缺键 fail-closed：读不到 `analysis_status` 一律记 INCOMPLETE。
                "analysis_status": (stage.get("analysis_status")
                                    if stage.get("analysis_status") in
                                    ("ANALYSIS_COMPLETE", "ANALYSIS_INCOMPLETE")
                                    else "ANALYSIS_INCOMPLETE"),
                "analysis_incomplete_reasons":
                    stage.get("analysis_incomplete_reasons") or [],
                "precision_status": stage.get("precision_status") or "UNMEASURED",
                "precision_evidence": stage.get("precision_evidence") or {},
                "path_is_complete": stage.get("path_is_complete"),
                "analysis_mode": stage.get("analysis_mode"),
                "skipped_windows": view.get("skipped_windows"),
                # [审计 #58] 子窗与物理窗口是两个命名空间，导出层也必须两个都带 ——
                # 只搬前一个会让所有后分析层（`stage2_ab_report.py` 等）对子窗
                # 结构性失明（写侧有、导出层漏，是「读一个没人写的键」的镜像）。
                "skipped_sampling_units": view.get("skipped_sampling_units"),
                "missing_windows": view.get("missing_windows"),
            },
            "windows": windows,
            # ── 代价：只比 ΔG 不比代价等于没比 ──
            "cost": {
                "total_production_steps": total_prod,
                # 未知不参与求和；有未知时这个总数是**下界**，如实标出来。
                "n_windows_with_unknown_production_steps": int(n_unknown_prod),
                "total_production_steps_is_lower_bound": bool(n_unknown_prod),
                "n_sampling_segments": len(view.get("aggregated_segments") or []),
                "n_autonomous_iterations": len(actions),
                "per_window_budget_remaining": view.get("per_window_budget_remaining"),
            },
            # ── 控制器都做了什么（对比时解释差异从哪来）──
            "controller": {
                "actions": actions,
                "fk_probe_verdict": probe.get("verdict"),
                "fk_probe_recommended_windows": probe.get(
                    "recalibration_recommended_windows"),
                "stale_layout_evidence": view.get("stale_layout_evidence"),
            },
        }

    def write_comparison_manifest(self, view=None, filename=None) -> str:
        """把 `comparison_manifest()` 落到 checkpoints，返回路径。

        ⚠️ **消费者是 `stage2_ab_report.py`**（`manifest_for()`）—— 改名必须同时改
        它，否则 A/B 报告静默读到 None。2026-09-14 改名时它已加了新旧两级回退。

        🔑 [审计 #63] 文件名**不得以 `stage2_` 开头**：`_read_stage_result()` 的
        兜底分支 glob 的正是 `stage2_*.json`，把控制器自己的产物放进那个命名空间，
        等于让「读 stage 结果」有机会读到「控制器对 stage 结果的摘要」。
        （今天它恰好被 `analysis_status`/`total_delta_G` 的内容检查挡住 —— 那是**巧合**，
        不是设计；manifest 以后多一个同名键就会真的命中。）
        """
        payload = self.comparison_manifest(view)
        out = os.path.join(
            self.path_checkpoint_dir,
            filename or "controller_comparison_manifest.json",
        )
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, out)
        return out

    # 候选 stage 结果文件，**按权威性排序**（前面的优先）。
    # ⚠️ 名字顺序 ≠ 新旧：`stage2_vanishing.json`（stage 缓存完成标记，可能是**几轮
    # 之前**写的）字典序排在 `stage2_vanishing_autonomous_inprogress.json 之前。
    _STAGE_RESULT_CANDIDATES = (
        "stage2_{stage}_autonomous_inprogress.json",   # 自治循环每轮 ANALYZE/rewindow 后写
        "stage2_{stage}.json",                          # stage 缓存完成标记
    )

    def _production_blocks_ledger(self, windows, path_version, units=None):
        """逐窗补帧块账：**已经批过几块、每块换来了什么**。

        🔑 [审计 #8/#9，契约 B] 返回 `{"by_window": {...}, "by_unit": {...}}` ——
        **两本账**。子窗（sampling unit）有自己的目录/checkpoint/冻结 f_k，它的块
        必须记在自己名下；先前只有 window 那本，而子窗一个都不在 `view["windows"]`
        里 ⟹ 子窗的块数硬上限与边际刹车**双双失效**（BUD-03 修过的「单窗烧 150 万步」
        在子窗这个新位置原样复发）。

        没有这本账，补帧就没有停止条件：真机实测可以一路 25→50→…→150 万步 ——
        早期分支（求解器跳窗 ⟹ 补帧）走不到后面的边际收益刹车，而停滞保护又把
        "多跑了一块"算成盘面进展、把重复计数清零。
        **「又产生了帧」不等于「获得了有用证据」。**

        判据量取**求解器侧**的去相关帧数（不是自检那份，两者实测差 2–8 倍，
        更不是步数）。

        可比性同 S2-C：同 `path_version`、同段才放进同一条轨迹。
        ⚠️ 这条「同段」是**边际增益曲线**的口径，由
        `test_frames_admission_and_stop.py::test_blocks_from_another_segment_do_not_count`
        钉住，**不要动**：跨 f_k epoch 比数字是没有意义的（设计 §6.10）。
        「这个窗口一共烧过几块 GPU」是**另一个量**，走
        `_production_blocks_total_by_window()`，见那里的 BUD-03 注释。
        """
        return self._production_blocks_scan(windows, path_version,
                                            same_segment_only=True, units=units)

    def _production_blocks_total_by_window(self, windows, path_version, units=None):
        """逐窗**跨全部采样段**的补帧块数。只给「块数硬上限」用。
        同上，返回 `{"by_window": ..., "by_unit": ...}` 两本账。

        🔑🔑 [BUD-03，2026-09-14] **换段不该把已经烧掉的 GPU 退回去。**

        先前「块数硬上限」和「边际增益」共用同一本账，而那本账只留**同段**的行。
        换段正是循环自己的动作（换 Epoch 就换段）⟹ 窗口每换一次段，配额清零、
        重新发满 4 块。真机 cyclod_ligand2/rep1 win5：history 里记的是 `vanishing`，
        当前视图是 `vanishing_7`，于是全部被过滤掉、`production_blocks_by_window`
        直接是空 `{}` —— 硬上限与边际刹车**一次都没触发过**，实测单窗烧到
        1,500,000 步（cyclod_ligand1/rep3 win4、cyclod_ligand2/rep1 win4）。

        两者的可比性要求本来就不同，所以分成两个量、各自过滤：
          · **硬上限** = 资源账，跨段累计（本函数）；
          · **边际增益** = 同一条曲线上的趋势，必须同段（`_production_blocks_ledger`）。

        🔑🔑🔑 [S2-L，2026-09-17 真机] **资源账也不该因为「换布局」就退钱。**

        BUD-03 修掉了「换**段**清零」，但两本账都仍然按 `path_version` 过滤
        （`_production_blocks_scan` 里那句 `path_version != it.path_version: continue`）
        ⟹ **插一次 λ / 拆一次窗，补帧的块数硬上限就清零、重新发满 4 块**。
        这跟 BUD-03 是**同一个 bug 换了一个维度**：那次是"换段退钱"，这次是"换布局退钱"，
        而本函数的定义白纸黑字写着「硬上限 = **资源账**」—— 烧掉的 GPU 不会因为布局
        变了就回来。
        真机证据（`abfe-benchmark-cb` 快照）：`brd4_ligand1/rep2` w3 累计 **1,250,000 步
        = 5 块**，而上限是 4 块；`cmet_ligand1/rep2` w0、`cmet_ligand2/rep2` w3 各 1,000k。
        叠加 `S2-A`（边际增益刹车全程没有输入、恒不触发），补帧在布局反复演化的 run 上
        **实际没有有效上限**。

        ⟹ 硬上限这本账**不按 `path_version` 过滤**（传 `None`）。
        ⚠️ **边际增益那本仍然要过滤**：跨布局比 `min N_eff/g` 没有意义（窗口几何都变了），
        那是判据的可比性要求，不是资源账。两者的过滤维度本来就不同，这正是 BUD-03
        把它们拆成两个函数的理由。
        """
        # ⚠️ 形参 `path_version` 保留但**不再传给扫描**：调用方（两个视图构造器）
        # 传的是同一个值，而这两本账对它的需求相反。留着形参是为了让调用点保持对称、
        # 一眼看出"这里本来有个版本维度、本函数刻意不用"。
        return self._production_blocks_scan(windows, None,
                                            same_segment_only=False, units=units)

    # 🔑 [审计 #40②] **哪些动作确实花掉一个生产块。**
    # 先前只数 `RUN_PRODUCTION`。`PROBE_REANCHOR_EPOCH` 明确「候选 f_k + 独立 burn-in
    # + 一个 +250k 块」、`RECALIBRATE_FK` 派发时 `probe_only=False` 会**开新段跑满**
    # `n_steps_per_window` —— 两个都实打实烧 GPU，却一块都没记进块账 ⟹ 块数硬上限
    # 和边际刹车都看不见它们烧掉的量。
    # [2026-09-15] `PROVISIONAL_PRODUCTION` 采的就是一个实打实的生产块，
    # 不记进块账 = 给它开一条绕过块数硬上限与边际刹车的旁路。
    _BLOCK_CHARGING_ACTIONS = (
        "RUN_PRODUCTION", "PROBE_REANCHOR_EPOCH", "RECALIBRATE_FK",
        "PROVISIONAL_PRODUCTION",
    )

    # 🔑🔑 [2026-09-18] **记账面 ⊋ 准入面。** 两张表刻意不同，别合并：
    #   · `_BLOCK_CHARGING_ACTIONS` 是**准入**用的（`plan()` 里问"这个动作批不批"）；
    #   · 下面这张是**记账**用的（扫历史，数"这个窗口烧过几块"）。
    # `IMMUTABLE_REWINDOW` 只进记账那张：它确实烧生产块（执行器给每个子窗采一个
    # `base_unit`，`plan()` 的 `_PRODUCTION_CHARGED` 也按 `n_children ×
    # new_ensemble_reserve_steps` 预留预算），所以资源账必须看得见它。
    # 但**不进准入**：换不起 Epoch 时有界重窗正是唯一对症的那条路（见 `plan()` 里
    # O1 那段），拿父窗已经吃满的块配额去否决它，等于掐掉唯一的出路 —— 与
    # `_frames_admission` 的 `hard_cap_only` 是同一条理由，方向别搞反。
    # ⚠️ 记进父窗名下不会误伤父窗自己的补帧：父窗一旦被子系综接管就进
    # `parents_done`，`_is_routable_candidate` 对它直接返回 False，不会再收到补帧。
    _BLOCK_LEDGER_ACTIONS = _BLOCK_CHARGING_ACTIONS + ("IMMUTABLE_REWINDOW",)

    def _production_blocks_scan(self, windows, path_version, *,
                                same_segment_only, units=None):
        out: Dict[int, List[Any]] = {}
        out_u: Dict[str, List[Any]] = {}
        try:
            hist = self._json(os.path.join(
                self.path_checkpoint_dir, "stage2_autonomous_history.json")) or {}
            # 单段视图的窗口记录没有 `segment` 字段（那是合并视图加的）⟹ 用本 stage
            # 的名字兜底，两侧同口径，否则账永远是空的、准入门形同虚设。
            _base = getattr(self, "_stage_base", self.stage_name)
            seg_of = {int(w["window_idx"]): (w.get("segment") or _base)
                      for w in (windows or [])}
            # ⚠️ 子窗没有 `segment`：可比性维度是 `identity`（每个子系综一份 f_k）。
            id_of_unit = {str(u["unit_id"]): u.get("identity")
                          for u in (units or []) if u.get("unit_id")}
            for it in (hist.get("iterations") or []):
                if path_version is not None and it.get("path_version") != path_version:
                    continue
                # 🔑🔑🔑 [2026-09-18] **账要跟着「实际跑了什么」，不是「控制器发了什么」。**
                #
                # 主循环的停滞降级会把动作改写成 `PROBE_REANCHOR_EPOCH`（真开新段、
                # 真烧一个 +250k 块），但**刻意保留** `action` 为控制器发的那个，
                # 把实际动作另记成 `action_executed`。而 `action_executed` 在本仓
                # **零个读者** —— 于是凡是降级前的动作不在这张表里的
                # （`CONTINUE_WARMUP` / `INSERT_LAMBDA` / `SPLIT_TAIL_WINDOW` /
                # `RELEARN_FK_EPOCH` / `IMMUTABLE_REWINDOW`），那一块帧**一格都不扣**，
                # 块数硬上限直接漏掉。与 S2-N（写侧 4 个动作、读侧只认 1 个）同一个
                # 形状，只是上移了一层。
                # ⚠️ `structural_action_history()` 仍然只看 `action`，那是**故意**的：
                # 它问的是「控制器试过几次」，不是「烧了几块 GPU」。两个量不同源。
                if (it.get("action_executed") or it.get("action")) \
                        not in self._BLOCK_LEDGER_ACTIONS:
                    continue
                # 🔑🔑 [审计 #40①] **只给这一轮实际点名的窗口记账。**
                # 先前给快照里**每个**窗口都记一行 —— 而快照是全窗的。于是一轮只补
                # win0 的迭代，会把 win1..win5 的配额也各扣一格；换段导致步数变化的
                # 窗口更是白占（去重键含 `production_steps`）。配额被别人的动作吃掉，
                # 真正在烧 GPU 的那个窗口反而提前撞上限。
                _named = it.get("windows")
                _named_set = ({int(x) for x in _named}
                              if isinstance(_named, (list, tuple)) and _named else None)
                for sn in (it.get("snapshot") or []):
                    i = int(sn.get("window_idx", -1))
                    if _named_set is not None and i not in _named_set:
                        continue
                    _seg = sn.get("segment") or _base
                    # 🔑 [审计 #41] **「这个窗口现在在哪个段」读不到时，不做同段过滤。**
                    # 先前 `_seg != seg_of.get(i)`：窗口不在当前视图里（证据被布局变更
                    # 作废、或该窗口这一轮没有产物）时 `seg_of.get` 返回 `None`，与任何
                    # 段名恒不相等 ⟹ **全部行被丢掉** ⟹ 边际增益刹车对「证据残缺」的
                    # 窗口静默失效 —— 而那正是最需要刹车的那一类窗口。
                    # 未知就是未知：保留全部行，并在行里标注段口径不可判。
                    _seg_known = int(i) in seg_of
                    if same_segment_only and _seg_known and _seg != seg_of.get(i):
                        continue          # 换段 = 换 f_k，不是同一条曲线
                    out.setdefault(i, []).append({
                        "iteration": it.get("iteration"),
                        "segment": _seg,
                        "segment_comparability": (
                            "same_segment" if _seg_known else "segment_unknown"),
                        "production_steps": sn.get("production_steps"),
                        "solver_n_decorrelated": sn.get("solver_n_decorrelated"),
                        "min_n_eff_over_g": sn.get("min_n_eff_over_g"),
                    })
                # ── 第二本账：子窗（契约 B）──────────────────────────────
                # 老 history 没有 `sampling_units_snapshot` ⟹ 「没有记录」，
                # 第一块照常批。**但**有些历史把子窗行直接写在 `snapshot` 里
                # （带 `unit_id`），那也是真实烧过的块，不能漏账 —— 按数组优先、
                # 不并用，避免同一块被两种形状各记一次（多记会提前刹车）。
                _uid_named = str(it.get("unit_id") or "") or None
                _u_rows = it.get("sampling_units_snapshot")
                if not _u_rows:
                    _u_rows = [r for r in (it.get("snapshot") or [])
                               if isinstance(r, dict) and r.get("unit_id")]
                for su in _u_rows:
                    uid = str(su.get("unit_id") or "")
                    if not uid:
                        continue
                    # 与窗口侧同一条规矩：只记这一轮实际点名的那个单元。
                    # 迭代没写 `unit_id`（老格式）⟹ 无从点名，全记（保守：宁可
                    # 多记也不少记，少记就是配额白送）。
                    if _uid_named is not None and uid != _uid_named:
                        continue
                    # 子窗的"段"是 `identity`；写在 `snapshot` 里的老形状只有
                    # `segment`（子系综目录名），两者都能承担"换系综就不可比"。
                    _uid_ident = su.get("identity") or su.get("segment")
                    _ident_known = (uid in id_of_unit
                                    and id_of_unit.get(uid) is not None
                                    and su.get("identity") is not None)
                    if (same_segment_only and _ident_known
                            and _uid_ident != id_of_unit.get(uid)):
                        continue
                    out_u.setdefault(uid, []).append({
                        "iteration": it.get("iteration"),
                        # 子窗这本账的"段"就是 `identity`；去重/同段过滤都用它。
                        "segment": _uid_ident,
                        "identity": _uid_ident,
                        "segment_comparability": (
                            "same_identity" if _ident_known else "identity_unknown"),
                        "production_steps": su.get("production_steps"),
                        # 执行器 snapshot 的输出键名沿用块账消费侧现有读法：
                        # 源是 `u["solver_n_frames_decorrelated"]`，落盘键是
                        # `solver_n_decorrelated`（同名不同源，别混）。
                        "solver_n_decorrelated": su.get("solver_n_decorrelated"),
                        "min_n_eff_over_g": su.get("min_n_eff_over_g"),
                    })

            def _dedup(rows):
                # 去重键含段：跨段扫描时，不同段的同一个 production_steps 是**两块**
                # 帧，只按步数去重会把跨段的块吞掉（那正是 BUD-03 要修的漏账）。
                # ⚠️ 步数未知的行**各算一块**（键里保留 `None`，不压成 0）——
                # 压成 0 会把多块"读不到步数"的行去重成一块，块数硬上限跟着少算。
                def _k(r):
                    _s = r.get("production_steps")
                    if _s is None:
                        return (str(r.get("segment") or ""), "unknown",
                                int(r.get("iteration") or 0))
                    return (str(r.get("segment") or ""), int(_s), 0)

                seen, uniq = set(), []
                for r in sorted(rows, key=lambda x: (
                        str(x.get("segment") or ""),
                        int(x.get("production_steps") or 0),
                        int(x.get("iteration") or 0))):
                    k = _k(r)
                    if k in seen:
                        continue
                    seen.add(k)
                    uniq.append(r)
                return uniq

            for i, rows in out.items():
                out[i] = _dedup(rows)
            for uid, rows in out_u.items():
                out_u[uid] = _dedup(rows)
        except Exception:  # noqa: BLE001 —— 读不到就当没有历史（第一块照常批）
            return {"by_window": {}, "by_unit": {}}
        return {"by_window": out, "by_unit": out_u}

    def _read_stage_result(self) -> Optional[Dict[str, Any]]:
        """stage 级结果（含生产质量门、缺窗口清单）。**只在 stage 跑完才存在。**

        🔑🔑 [CTL-01，2026-09-14] **不靠文件名顺序选权威。**
        先前这里 `sorted(glob("stage2_*.json"))` 取第一个带 `converged` 的 ——
        而 `.`(46) < `_`(95) ⟹ `stage2_vanishing.json` **永远排在**
        `stage2_vanishing_autonomous_inprogress.json 之前。于是自治循环每轮
        （ANALYZE / rewindow 之后）刚写出的新结果**根本读不到**，子窗的 solver
        帧数与跳窗状态全从一份过期结果里读 —— 11/20 的新证据进不了 `decide()`。
        （那句"真缓存一旦存在就排在前面、优先被采纳"的注释描述的是**写盘**时的
        意图，读盘这一侧照抄它就反了。）

        现在：**显式候选顺序** + **路径版本必须匹配当前布局**。
        带 `path_version` 且与当前不符的结果是**另一条布局**的结论，直接跳过；
        不带该字段的老产物按"无法判定版本"放行（保持既有行为，不制造新的 fail）。
        """
        _cur_pv = (self._read_path() or {}).get("version")
        stage = getattr(self, "_stage_base", self.stage_name)
        # [审计 #63] 原来这里有个 `seen` 列表：只被 append、从不被读，删掉。
        for pat in self._STAGE_RESULT_CANDIDATES:
            f = os.path.join(self.path_checkpoint_dir, pat.format(stage=stage))
            d = self._json(f)
            # [2026-09-15] `converged` 已从 stage 结果里删除 ⟹ 嗅探键换成
            # `analysis_status`。仍然认 `total_delta_G`，**不是**为了兼容旧语义，
            # 而是为了让老产物**能被读到**、随后在 `read()` 里 fail-closed 判成
            # `ANALYSIS_INCOMPLETE`（读不到 = 不得当成通过）。
            if not (isinstance(d, dict)
                    and ("analysis_status" in d or "total_delta_G" in d)):
                continue
            _pv = d.get("path_version")
            if _pv is not None and _cur_pv is not None and int(_pv) != int(_cur_pv):
                continue          # 另一条布局的结论，不是"旧一点"而是**另一个量**
            return d
        # 兜底：候选名都没命中（改过命名 / 非标准 stage 名）时回到全扫，
        # 但仍然按**修改时间**取最新的那份，而不是按文件名。
        for f in sorted(
            glob.glob(os.path.join(self.path_checkpoint_dir, "stage2_*.json")),
            key=lambda x: os.path.getmtime(x), reverse=True,
        ):
            d = self._json(f)
            if isinstance(d, dict) and ("analysis_status" in d
                                        or "total_delta_G" in d):
                _pv = d.get("path_version")
                if _pv is not None and _cur_pv is not None and int(_pv) != int(_cur_pv):
                    continue
                return d
        return None

    def _read_window(self, idx: int, stage_dir: Optional[str] = None,
                     checkpoint_dir: Optional[str] = None) -> Dict[str, Any]:
        # `stage_dir` / `checkpoint_dir` 可覆盖：immutable rewindow 的子窗住在
        # 自己的目录里（局部下标从 0 起，与父窗索引**同名不同义**），读法一样。
        stage_dir = stage_dir or self.stage_dir
        checkpoint_dir = checkpoint_dir or self.checkpoint_dir
        conv = self._json(os.path.join(
            stage_dir, f"dual_window_{idx}_{self.stage_type}_convergence.json"))
        fail = self._json(os.path.join(
            stage_dir, f"dual_window_{idx}_{self.stage_type}_warmup_failure.json"))
        state = self._json(os.path.join(
            checkpoint_dir, f"ibs_state_{self.stage_type}_window_{idx}.json"))
        lam = (conv or {}).get("lambdas_vdw") or (state or {}).get("lambdas_vdw") or []
        # 🔑🔑 [审计 #56] **`warm` 本身也必须取"消耗更多"的那一份，不是优先取
        # convergence。** 上一次修复（见下面 `spent` 的长注释）只落在 `spent` /
        # `_engine_left` 两个量上，`warm = conv.bias_warmup or fail...` 一字未动 ⟹
        # 同一条窗口记录里新旧两种口径混用：`spent` 来自 warmup_failure（新），
        # 而 gate 读数 / `validation_indeterminate` / `best_effort` 来自 convergence（旧）。
        # 口径只能有一套：`warm` 与 `ledger` / `spent` / `cap` 全部取同一份。
        _conv_warm = (conv or {}).get("bias_warmup") or {}
        _fail_warm = (fail or {}).get("bias_warmup") or fail or {}
        _conv_ledger = _conv_warm.get("warmup_budget_ledger") or {}
        _fail_ledger = _fail_warm.get("warmup_budget_ledger") or {}

        def _spent_of(_led):
            if not _led:
                return None
            return sum(int(_led.get(k, 0) or 0) for k in
                       ("learning_steps", "freeze_burn_in_steps",
                        "frozen_validation_steps"))

        def _cap_of(_led):
            # 🔑 [审计 #36] **`cumulative_cap_steps == 0` 是真实的「上限 0」，不是未知。**
            # 先前 `cap = (...) or (... or None)` 用真值判，而 `new_warmup_budget_ledger`
            # 的默认值恰恰就是 0 ⟹ 真上限 0 被读成未知 ⟹ `warmup_steps_left=None` ⟹
            # `all_windows_budget_exhausted` 恒假（#15 的反方向）。只能用 `is not None`。
            _c = (_led or {}).get("cumulative_cap_steps")
            return int(_c) if _c is not None else None

        _conv_spent, _fail_spent = _spent_of(_conv_ledger), _spent_of(_fail_ledger)
        # 🔑🔑 [审计 #35] **`cap` 与 `spent` 必须取同一份 ledger。**
        # 先前 `spent` 取两份里**较新**（消耗更多）那份、`cap` 却优先取 convergence
        # 那份**较旧**的 ⟹ 用户抬高 `max_bias_warmup_steps` 之后，新 cap 在
        # warmup_failure 里、旧 spent 已经追平旧 cap ⟹ `cap − spent ≡ 0`，
        # 而且这个假零还会把引擎的权威 `warmup_budget_remaining_steps` 挡掉
        # （旧逻辑要求 engine_left ≤ cap−spent 才采信）⟹ **显式升档永不生效**。
        # 现在成对取：消耗更多的那一份的 cap 与 spent 一起用。
        _use_fail_ledger = (_fail_spent if _fail_spent is not None else -1) > \
                           (_conv_spent if _conv_spent is not None else -1)
        warm = (_fail_warm if _use_fail_ledger else _conv_warm) \
            or _conv_warm or _fail_warm or {}
        ledger = _fail_ledger if _use_fail_ledger else _conv_ledger
        spent = _fail_spent if _use_fail_ledger else _conv_spent
        cap = _cap_of(ledger)
        _cap_conv, _cap_fail = _cap_of(_conv_ledger), _cap_of(_fail_ledger)
        # 两份 cap **都在场且不一致** = 有人改过上限（或跨段继承了）⟹ 只有引擎
        # 知道真实剩余，无条件采信它；一致时才谈"取更保守的那个"。
        _caps_disagree = (_cap_conv is not None and _cap_fail is not None
                          and _cap_conv != _cap_fail)
        # 🔑 [P2-9a+ / 老板第 3 条] **warmup 剖面是一等证据** ——
        # 「哪个窗弱，就可以提前增加采样了」。上一跑的实测支持这条：两个
        # occupancy 塌的窗口正是 coverage_ess 最低的两个，而 win3 拿到 2× 预算活了、
        # win0 拿默认 1× 死了（去相关后只剩 6 帧）。win3/win4 那 2×/4× 是上一轮
        # rescue **事后**补的，不是按 warmup 信号**提前**给的 —— 所以 win0 从来没
        # 进过那个名单。⟹ 生产预算本可以在**生产开始之前**按该窗口自己的剖面配。
        # ⚠️ 这里**只落盘 + 只排序，不给公式**：五个点不足以定"coverage_ess < X
        # 就给 N× 预算"，拍一个就是引入未验证阈值（同 P3-9c 那条线）。
        gate = (warm.get("local_mbar_loose_gate") or {}).get("last") or {}
        # [P2-9c] 窗口跑完那一刻的自检：它自己的去相关帧数够不够。
        # 与 `solve_stage_integrated` 判跳过用的**同一个量**（energies + 门槛 10），
        # 只是提前到窗口刚跑完就算。**只报告**，不改预算、不碰放行判据。
        selfchk = self._json(os.path.join(
            stage_dir, f"dual_window_{idx}_{self.stage_type}_self_support.json"
        )) or {}
        bias_status = (state or {}).get("bias_status")
        evid = (state or {}).get("f_k_evidence_status")
        # 🔑🔑 [2026-09-14 真机根因] **「还剩多少预热预算」有两份记录，必须取权威那份。**
        #
        # `convergence.json` 的 ledger 是**生产跑完那一刻**写的；而 warmup 之后还会
        # 继续消耗（续跑再进一次预热、跨段继承），最新值写在
        # `*_warmup_failure.json` 里，引擎自己还算好了 `warmup_budget_remaining_steps`。
        # 先前 `warm = conv.bias_warmup or fail...` **优先取 convergence** ⟹ 控制器
        # 读到的是过期的那份。
        #
        # 实测 rep2 win4：convergence 说 spent=815000 / left=140000，
        # warmup_failure 说 spent=955000 / **remaining=0**。于是控制器以为"预算有余"
        # 一直发补帧，而引擎每次都以「累计预算已用尽，本次可用 0 步」把它弹回来 ——
        # 连发 40 轮、盘面一字节没变。
        #
        # 口径：**引擎自己算的 `warmup_budget_remaining_steps` 是权威**；拿不到才
        # 退回逐项相加。`warm` / `ledger` / `spent` / `cap` 的成对选取见上面
        # `_use_fail_ledger` 那一段（审计 #35/#36/#56）。
        # 引擎的权威剩余量（它才知道跨段继承了多少）。
        _engine_left = _fail_warm.get("warmup_budget_remaining_steps")
        if _engine_left is None:
            _engine_left = (fail or {}).get("warmup_budget_remaining_steps")
        _ledger_left = (max(0, int(cap) - int(spent))
                        if (cap is not None and spent is not None) else None)
        # 🔑 [审计 #43] **来源标签必须如实反映实际采用的那个数。**
        # 先前只判 `_engine_left is not None` 就写 `"engine:..."`，而引擎值可能被
        # 上面的三元式否决、实际用的是 ledger —— 一个会撒谎的标签比没有标签更坏
        # （`tests/test_ctl10_warmup_budget_authority.py` 钉的正是这个字符串）。
        if _engine_left is not None and (_caps_disagree or _ledger_left is None):
            _left_val, _left_src = int(_engine_left), "engine:warmup_budget_remaining_steps"
        elif _engine_left is not None and int(_engine_left) <= _ledger_left:
            _left_val, _left_src = int(_engine_left), "engine:warmup_budget_remaining_steps"
        elif _ledger_left is not None:
            _left_val, _left_src = int(_ledger_left), "ledger:cap-spent"
        else:
            _left_val, _left_src = None, "unknown"
        prod = (conv or {}).get("cumulative_production_steps")
        if prod is None:
            prod = (conv or {}).get("actual_production_steps")
        return {
            "window_idx": idx,
            "has_convergence": conv is not None,
            "has_warmup_failure": fail is not None,
            "n_states": len(lam) or None,
            # 这份证据**是在哪套 λ 上产出的**。布局一变它就不再描述这个窗口 ——
            # 用它做身份，比拿段号/mtime 猜新旧可靠。
            "lambdas_vdw": [float(x) for x in lam] or None,
            "lambda_span": (max(lam) - min(lam)) if lam else None,
            "bias_status": bias_status,
            "phase": (
                # 身份对不上时，盘上的 converged 不算数 —— 那是另一个系综的结论。
                "IDENTITY_MISMATCH"
                if (
                    (state or {}).get("stage_protocol_key") is not None
                    and (conv or {}).get("stage_protocol_key") is not None
                    and (state or {}).get("stage_protocol_key")
                    != (conv or {}).get("stage_protocol_key")
                )
                else self._PHASE.get(bias_status or "", "UNKNOWN")
            ),
            "f_k_evidence_status": evid,
            "verdict": self._VERDICT.get(evid or "", "INSUFFICIENT_DATA"),
            "last_failure_reason": (state or {}).get("last_failure_reason"),
            "last_gate_error": warm.get("last_gate_error") or (fail or {}).get("last_gate_error"),
            "best_effort_acceptance": bool(warm.get("best_effort_acceptance")),
            "best_effort_reason": warm.get("best_effort_acceptance_reason"),
            "bias_update_count": warm.get("bias_update_count"),
            # warmup 剖面（只报告；预算映射未定，见 PLAN P3）
            "warmup_g": gate.get("statistical_inefficiency"),
            "warmup_n_frames_used": gate.get("n_frames_used"),
            "warmup_min_absolute_ess": gate.get("min_absolute_ess"),
            "warmup_min_ess_ratio": gate.get("min_ess_ratio"),
            "warmup_max_adjacent_delta_kJ_mol": gate.get("max_adjacent_delta_kJ_mol"),
            "warmup_gate_threshold_kJ_mol": gate.get("gate_threshold_kJ_mol"),
            # 生产后自检（P2-9c）
            "self_n_frames_decorrelated": selfchk.get("n_frames_decorrelated"),
            "self_min_frames": selfchk.get("min_frames_per_window"),
            "self_sufficient": selfchk.get("sufficient"),
            "self_verdict": selfchk.get("verdict"),
            # 为什么是这个 verdict：solver_eligibility / min_n_eff_over_g /
            # top1pct_veto。三者补救方向都是加采样，但归因不同，对账时要分得开。
            "self_verdict_source": selfchk.get("verdict_source"),
            "self_frames_short_by": selfchk.get("frames_short_by"),
            # 验收量：N_eff,k / g_k（未抽稀帧上逐目标态 support ÷ 时间自相关）。
            # ⚠️ occupancy **不是**验收判据 —— 它是 f_k 的训练目标；win1 段1 的
            # occupancy_collapsed=True 但支撑健康，是实测假阳性。
            "min_n_eff_over_g": selfchk.get("min_n_eff_over_g"),
            # [2026-09-16] 门槛跟着读数一起进视图：判"加帧还能不能推过门"必须用
            # **写侧这一次实际生效**的档位（`window_self_support_check` v2 起落盘）。
            # 老产物没有这个键 ⟹ None ⟹ `support_failure_is_skew` 保持既有判法。
            "self_n_eff_over_g_eligible": selfchk.get(
                "n_eff_over_g_eligible_threshold"),
            # 生产侧累计 f_k 偏差（scope=production），由 solve_stage_integrated 落在
            # stage 结果的 `cumulative_fk_residual_production` 里。
            "cum_fk_span": None, "cum_fk_verdict": None,  # 在 read() 里按窗口填
            "worst_state_by_n_eff": selfchk.get("worst_state_by_n_eff"),
            # 🔑🔑 [2026-09-17，用户拍板 #3] **瓶颈态的 g / η / ratio 观测量。**
            # 目的只有一个：把 `n_eff_over_g` 的**可达性**从"假定 η、g 恒定"的
            # 乐观上界，换成有数据支撑的判断。**本轮只记录，不改任何门**。
            #
            # 三个量必须来自**同一个 k\*、同一个 frame set**，否则记了也没法用：
            #   · `k* = worst_state_by_n_eff_over_g`（不是 `worst_state_by_n_eff`，
            #     那是另一个量 —— 分子最小 ≠ 比值最小）；
            #   · `g[k*]`  取 `statistical_inefficiency_per_lambda[k*]`；
            #   · `η[k*] = n_eff_per_state[k*] / n_eff_input_n_frames_per_state[k*]`，
            #     分母是**逐态、finite mask 之后、未按 sub_idx 去相关稀疏**的帧数。
            #     ⚠️ 绝不能用 `n_frames_decorrelated`（自检侧）或
            #     `solver_n_frames_decorrelated`（求解器侧）—— 那是 solver eligibility
            #     的量，与 η 差好几倍；也不能用 `n_eff_frame_set_n_frames_raw`，
            #     那是 finite mask **之前**的数。
            # 只认 `window_self_support_check` 这一份；solver g / warmup g /
            # validation g 一概不得代替。读不出来一律 `None`（= UNKNOWN），
            # **不填 0、不当作可达** —— 老产物没有这些键，必须如实说"不知道"。
            **_bottleneck_observation(selfchk),
            # 早判：边际 N_eff 断崖（比固定节奏更准，实测脱轨点跨 4.5 倍）
            "derail_at_trajectory_fraction": selfchk.get("derail_at_trajectory_fraction"),
            "derailment_status": selfchk.get("derailment_status"),
            "derail_block_index_single_block": selfchk.get("derail_block_index_single_block"),
            # 身份：state 里持久化的采样身份 vs 本窗产物的身份。
            # 不一致 ⟹ 盘上那个 bias_status=converged 是**另一个系综**的旧结论，
            # 控制器不得据此报"已在生产"（同僚今天真机踩到过：能量缓存因身份不符
            # 被拒、整窗重采，而 bias_converged=True 照样粘过去）。
            "state_protocol_key": (state or {}).get("stage_protocol_key"),
            "product_protocol_key": (conv or {}).get("stage_protocol_key"),
            # 🔑 [审计 #1 的读侧] **语义变了，键名跟着变。**
            # 原来落盘的 `n_eff_marginal_by_block` 是前缀差
            # `ESS(prefix_b) − ESS(prefix_{b−1})` —— ESS 是非可加的比值泛函，
            # 做差不是"边际量"（加一帧高权重帧 ESS 反而塌向 1，被误判 CONFIRMED_DERAILMENT）。
            # 权威判据量现在是**块内独立 ESS** `block_local_ess_by_block`；
            # 前缀差改名 `prefix_ess_by_block` 只作诊断。
            # 视图键跟着改名，否则 `n_eff_marginal_by_block` 这个名字会长期骗人。
            "block_local_ess_by_block": (
                selfchk.get("block_local_ess_by_block")
                or selfchk.get("n_eff_marginal_by_block")   # 旧产物回退
            ),
            "prefix_ess_by_block_REPORT_ONLY": selfchk.get("prefix_ess_by_block"),
            "warmup_steps_spent": spent,
            "warmup_steps_cap": cap,
            # 🔑 [2026-09-14 真机] **账本必须原样带出来。**
            # `relearn_epoch_required_steps()` 读 `window_record["warmup_budget_ledger"]`
            # 来自校准"开一个新 Epoch 要多少步"，而这个键此前**根本没被放进窗口记录**
            # ⟹ 它永远走 80000+10000+50000 的兜底常量。真机 cyclod_ligand1/rep2 win3
            # 的真实账本是 learning=230000 / burn_in=10000 ⟹ 真实需求 290000，
            # 被低估了一半还多。方向是**危险**的那一侧：会批准一个跑不完的 Epoch，
            # 把预算烧光再半路死掉 —— 正是那段注释声称要防止的事。
            # 取消耗更多的那一份（与上面 `spent` 同口径，保守）。
            # 与上面 `spent` / `cap` 同一份 ledger（审计 #35：不许两个量取不同来源）。
            "warmup_budget_ledger": ledger,
            "warmup_steps_left": _left_val,
            "warmup_steps_left_source": _left_src,
            # [2026-09-12] 验证**可达性**预检的原料。同一候选、连续数据上的
            # 多个检查点（`insufficient_attempts` 每次都记一个 g），用来算保守
            # 下界 g_L；再配上 T / Ncap / 剩余预算就能判"在算术上还可不可能"。
            "validation_indeterminate": warm.get("validation_indeterminate"),
            # 🔑🔑 [审计 #55] **嵌套层级读错了。** 引擎把 `validation_g_history`
            # 写在 `bias_warmup["validation_indeterminate"]` **内部**
            # （`ibs_engine.py` 的 `validation_indeterminate_diag`），控制器却在
            # `bias_warmup` **顶层**读 ⟹ 恒为空 ⟹ 每次都退回那个单点兜底，
            # 而引擎自己的注释明写「可达性预检要的是多个检查点，不是单点：
            # 单点在小样本下会误杀」。可达性预检因此一直在用它被警告过的那个口径。
            # 正确层级优先，顶层留作兜底（万一以后被提上去）。
            "validation_g_checkpoints": [
                float(x) for x in (
                    (warm.get("validation_indeterminate") or {}).get(
                        "validation_g_history")
                    or warm.get("validation_g_history")
                    or [(warm.get("validation_indeterminate") or {}).get(
                        "statistical_inefficiency")]
                ) if x
            ],
            # 🔑 T = **去相关**帧数下限（触发 insufficient_frames_after_decorrelation
            # 的那个），不是 `minimum_complete_validation_frames`（原始帧完整性要求）。
            # 混掉会把 gcrit 算小 20 倍：win4 只差 26% 帧数却被判成差 7.5 倍不可达。
            # 老产物没落这个字段时回退到常量，绝不回退到 200。
            "validation_required_frames": (
                (warm.get("validation_indeterminate") or {}).get(
                    "decorrelated_frames_required")
                or _ie_min_frames()
            ),
            "validation_completeness_frames_REPORT_ONLY": warm.get(
                "minimum_complete_validation_frames"),
            "validation_sample_count": (
                (warm.get("validation_indeterminate") or {}).get("validation_sample_count")
            ),
            "frozen_validation_steps": (state or {}).get("frozen_validation_cumulative_steps"),
            "frozen_validation_batches": (state or {}).get("frozen_validation_batches_done"),
            "production_steps": prod,
            "production_steps_target": (conv or {}).get("n_steps_per_window_effective"),
            "n_production_segments": len((conv or {}).get("production_segments") or []) or None,
            "n_frames": ((conv or {}).get("window_data") or {}).get("n_frames"),
        }

    def _make_production_budget(self, *, cap, cap_source, per_window_used,
                                per_unit_used, unknown_usage, note):
        """生产预算账的**唯一**构造器（审计 #39）。

        单段视图和合并视图先前各写一遍，差别不只是写法：单段那份用 `or 0`、没有
        `usage_complete` / `unknown_usage`、还无条件算 `stage_remaining_steps` 和
        `exhausted` ⟹ 同一个 stage 换个视图读，"还剩多少 / 有没有耗尽"会得到两个
        答案，而 `plan()` 的预算闸就靠这两个字段。

        三条规矩（都是"未知不是零"的实例）：
          · **上限未知 ⟹ 余额未知**，「预算耗尽」这个结论不成立，绝不因此终止；
          · **用量不完整 ⟹ 余额未知**，不许拿一个少算过的数当余量去准入；
          · `exhausted` 只有在「上限已知 **且** 用量完整 **且** 确实超了」时才为 True。
        """
        _unknown = list(unknown_usage or [])
        _used = (sum(int(v) for v in (per_window_used or {}).values() if v is not None)
                 + sum(int(v) for v in (per_unit_used or {}).values() if v is not None))
        _cap_known = cap is not None
        return {
            "stage_cap_steps": cap,
            "cap_source": cap_source,
            "cap_known": _cap_known,
            "stage_used_steps": int(_used),
            "usage_complete": not _unknown,
            "unknown_usage": _unknown,
            "stage_remaining_steps": (
                None if (not _cap_known or _unknown) else int(cap) - int(_used)
            ),
            "per_window_used_steps": dict(per_window_used or {}),
            "per_sampling_unit_used_steps": dict(per_unit_used or {}),
            "new_ensemble_reserve_steps": int(self.new_ensemble_reserve_steps),
            "production_block_steps": int(self.production_block_steps),
            "cap_scope": "production_steps_only",
            "exhausted": bool(_cap_known and not _unknown and _used >= int(cap)),
            "note": note,
        }

    def _read_single_stage(self) -> Dict[str, Any]:
        """**单个段**的状态 + 证据 + 剩余预算。只读盘。

        ⚠️ 物理 stage 的决策**不要**直接用它 —— 走 `for_physical_stage()` 的聚合
        视图，否则同一个 stage 会被当成多个独立 stage 各判一次。
        """
        found: List[int] = []
        for pat, rx in (
            (f"dual_window_*_{self.stage_type}_convergence.json", r"dual_window_(\d+)_"),
            (f"dual_window_*_{self.stage_type}_warmup_failure.json", r"dual_window_(\d+)_"),
        ):
            for f in glob.glob(os.path.join(self.stage_dir, pat)):
                m = re.search(rx, os.path.basename(f))
                if m and int(m.group(1)) not in found:
                    found.append(int(m.group(1)))
        # [P2-9a] join λ 两侧支撑：相邻窗口对**共享的那一个 λ** 各自的重要性支撑。
        # 由 `ibs_engine.join_lambda_two_sided_support` 在每个窗口落盘后自动算并
        # 落成 `dual_join_{up}_{down}_{type}_support.json`。**只报告、不参与放行。**
        joins = []
        for f in sorted(glob.glob(os.path.join(
            self.stage_dir, f"dual_join_*_{self.stage_type}_support.json"
        ))):
            d = self._json(f)
            if isinstance(d, dict):
                joins.append(d)
        joins.sort(key=lambda d: int(d.get("upstream_window", -1)))
        # [P2-9h] f_k 重标定探针：在 rescue **之前**判"该重标定还是该加帧"。
        # 位移（`max_adjacent_shift_kJ_mol`）是直接证据，偏斜（top1%）只是症状。
        fk_probe = self._json(os.path.join(
            self.path_checkpoint_dir, "stage2_fk_recalibration_probe.json"
        )) or {}
        # [2026-09-14] 有界 rewindow 的台账：哪一个父窗口、什么身份、采了几块。
        # 进停滞签名（新建子系综 = 盘上状态确实变了），并让 `decide()` 知道
        # 这个父窗口已经建过子系综 —— 别重复建第二个。
        _rw_ledger = self._json(os.path.join(
            self.path_checkpoint_dir, "stage2_rewindow_ledger.json")) or {}
        # 「某个动作在当前盘面上是 no-op」的记账（执行器写，控制器读）。
        noop_actions = self._json(os.path.join(
            self.path_checkpoint_dir, "stage2_noop_actions.json")) or {}
        # `_rw_latest` 的赋值挪到 `_rw_entry_is_current()` 定义之后（审计 #42），
        # 因为选"最新"必须先能判"是不是当前布局的"。
        # 🔑🔑 [2026-09-14 裁决 1] **子窗是控制器可调度的采样单元。**
        # `windows` 继续表示**物理窗口**；子窗另立一张表 `sampling_units`：
        # 它有自己的目录、自己的 checkpoint、自己的冻结 f_k，局部下标从 0 起
        # （与父窗索引**同名不同义**，所以绝不能靠放宽 `_segment_stage_names()`
        # 把它并进段发现），`solver_index` 只是求解器的命名空间偏移。
        # 🔑🔑 [CTL-05，2026-09-14] **ledger 必须按当前 `path_version` 隔离。**
        # 路径演化（插 λ / 拆末窗）之后，旧布局下建的子系综描述的是**另一套 λ 区间**；
        # 拿它们参与当前求解会混进不可比的帧，而旧的 `parents_done` 还会把新的修补
        # 挡在门外（"这个父窗已经建过子系综了"——那是上一条布局的事）。
        # 不带 `path_version` 的老条目按"无法判定版本"放行，不制造新的 fail。
        # ⚠️ 不能写 `path` —— 它在本函数里要到后面才赋值（用了就是 UnboundLocalError，
        # 而且包在 try 里会被静默吞掉）。直接读。
        _cur_pv_for_rw = (self._read_path() or {}).get("version")

        def _rw_entry_is_current(_e):
            _pv = (_e or {}).get("path_version")
            if _pv is None or _cur_pv_for_rw is None:
                return True
            return int(_pv) == int(_cur_pv_for_rw)

        # 🔑🔑 [审计 #42] **"最新的那条 rewindow 记录"不能按 identity 字典序取。**
        # 先前 `for _k in sorted(_rw_ledger): _rw_latest = ...` —— identity 是内容
        # 哈希，字典序与时间毫无关系；而且**不过 `_rw_entry_is_current()`** ⟹
        # `immutable_rewindow` 报出来的 identity / child_ranges 可能描述**另一条
        # 布局**下的子系综。后果不止是显示错：`identity` 进停滞签名，会把一次真实
        # 推进记成 no-op（签名没变 ⟹ 判"推不动" ⟹ 提前放弃）。
        # 正解：只在当前 `path_version` 的条目里选，按条目自带的时间戳/序号取最新，
        # 都没有就按插入序取最后一个 current 的（dict 保插入序）。
        _rw_current = [(_k, _v) for _k, _v in _rw_ledger.items()
                       if isinstance(_v, dict) and _rw_entry_is_current(_v)]

        def _rw_recency(_kv):
            # `solver_index_base` 是执行器建窗时按**台账已有条目数**递增钉死的
            # （`10000 + 100 * n`），一旦写入不再变 ⟹ 它就是这份台账里唯一真实的
            # 序号。没有它的老条目退回插入序。
            _b = (_kv[1] or {}).get("solver_index_base")
            try:
                return (1, float(_b)) if _b is not None else (0, 0.0)
            except (TypeError, ValueError):
                return (0, 0.0)

        _rw_latest = None
        if _rw_current:
            _stamped = [kv for kv in _rw_current if _rw_recency(kv)[0]]
            _rw_latest = (max(_stamped, key=_rw_recency)[1] if _stamped
                          else _rw_current[-1][1])

        sampling_units = []
        for _rid in sorted(_rw_ledger):
            _e = _rw_ledger[_rid]
            if not isinstance(_e, dict) or not _rw_entry_is_current(_e):
                continue
            # 🔑🔑 [审计 #7，2026-09-14] **`status` 必须参与过滤。**
            # 先前只按 `path_version` 过滤。而求解侧
            # （`_solve_with_rewindow_children`）**只采信 `SAMPLED`**，
            # `_topup_rewindow_child` 又从不把 status 推回 `SAMPLED` ⟹
            # 一条 `ABANDONED_NO_PRODUCT`（或卡住的 `SAMPLING_INTENT`）条目会每轮
            # 被判成「缺帧 ⟹ 补一块」，烧到 40 轮上限，而**一帧都进不了 ΔG**；
            # 停滞保护还看不见它（签名里它的 `production_steps` 每轮都在变）。
            # 兜底：老条目缺 `status` 按 `SAMPLED`（与
            # `abfe_pipeline._rewindow_entry_status` 同一条既有先例，别删 ——
            # 删了会把所有既有 run 的子系综一次性判废）。
            _e_status = str(_e.get("status") or "SAMPLED")
            _schedulable = (_e_status == "SAMPLED")
            _od, _cd = _e.get("output_dir"), _e.get("checkpoint_dir")
            for _li, _rng in enumerate(_e.get("child_ranges") or []):
                _w = self._read_window(_li, stage_dir=_od, checkpoint_dir=_cd) \
                    if (_od and _cd) else {}
                sampling_units.append({
                    "unit_id": f"rw:{_e.get('identity')}:{_li}",
                    "kind": "rewindow_child",
                    # 台账状态原样带出来供审计；`schedulable=False` 的单元不产生
                    # 任何动作（见下面 `needs_frames` / `complete` 的赋值）。
                    "status": _e_status,
                    "schedulable": bool(_schedulable),
                    # ⚠️ [契约 B] 子窗**没有 `segment`**。「换段 = 换 f_k」这一维在
                    # 子窗这边由 `identity` 承担（每个子系综自己锁一份 f_k），
                    # 块账的可比性与 no-op 指纹都用它，别再造一个 `segment` 出来。
                    "parent_window": _e.get("parent_window"),
                    "local_index": int(_li),
                    "range": [int(x) for x in _rng],
                    # 续跑这个子窗要用整套子布局（执行器传 window_ranges）
                    "all_child_ranges": [[int(a), int(b)]
                                         for a, b in (_e.get("child_ranges") or [])],
                    "identity": _e.get("identity"),
                    "parent_range": [int(x) for x in (_e.get("parent_range") or [])],
                    "output_dir": _od,
                    "checkpoint_dir": _cd,
                    # 求解器命名空间，**不是**调度身份。
                    # 🔑 基数从台账读（建窗时钉死），**不再**所有 identity 共用
                    # `10000/10001` —— 那会让两组子窗串号、solver 的跳窗记录被
                    # 误分配给另一组。
                    "solver_index": int(_e.get("solver_index_base") or 10_000) + int(_li),
                    "production_steps": _w.get("production_steps"),
                    "self_verdict": _w.get("self_verdict"),
                    # 归因来源要跟着单元走：下面那个循环里 `_w` 是**上一轮遗留**的
                    # 变量（永远指向最后一个子窗），从它读等于全体子窗共用一份归因。
                    "self_verdict_source": _w.get("self_verdict_source"),
                    "self_sufficient": _w.get("self_sufficient"),
                    "self_n_frames_decorrelated": _w.get("self_n_frames_decorrelated"),
                    "min_n_eff_over_g": _w.get("min_n_eff_over_g"),
                    # 🔑🔑 [2026-09-17] **下面这 6 个键先前没被抄进来，而有 3 个消费者在读。**
                    # `sampling_units` 是 `_read_window()` 记录的**手抄子集**，抄漏了不报错，
                    # 行为静默退化成「那个条件永远不成立」—— 与
                    # `tests/test_no_phantom_view_keys.py` 开头列的 5 次是同一形状，
                    # 只是发生在数据面的**第二层**（unit 记录），而那个测试只盯 view 顶层。
                    #   · `self_min_frames`            → `support_failure_is_skew(min_frames=None)`
                    #     ⟹ 「n_decorrelated < min_frames」这条**样本量硬证据**对子窗恒不成立
                    #     ⟹ 子窗的支撑失败一律被归成偏斜/UNKNOWN，走 D3 停机而不是补帧；
                    #   · `self_n_eff_over_g_eligible` → `_frames_admission` 第三道闸的门槛恒 None
                    #     ⟹ `n_eff_over_g_reachable_by_frames` 返回 None ⟹ **该闸对子窗恒放行**；
                    #   · `bottleneck_*` 四个         → D3 终态诊断里恒 None，而这四个正是
                    #     2026-09-17 拍板 #3 专门加来把可达性从「假定 η、g 恒定的乐观上界」
                    #     换成有数据支撑的判断的量。
                    "self_min_frames": _w.get("self_min_frames"),
                    "self_n_eff_over_g_eligible": _w.get("self_n_eff_over_g_eligible"),
                    "bottleneck_state": _w.get("bottleneck_state"),
                    "bottleneck_g": _w.get("bottleneck_g"),
                    "bottleneck_eta": _w.get("bottleneck_eta"),
                    "bottleneck_ratio": _w.get("bottleneck_ratio"),
                    # 🔑 [2026-09-18] η≤1 硬上界（`eta_lever_can_reach`）要的分子。
                    # 少了它，`N/g` 判据对子窗恒 `None` ⟹ 第四态
                    # `N_OVER_G_BOUND` 在子窗侧恒不成立 —— 与上面那四个键当初被
                    # 补上的理由逐字相同，别再漏第五个。
                    "bottleneck_n_eff_input_n_frames": _w.get(
                        "bottleneck_n_eff_input_n_frames"),
                    "warmup_steps_left": _w.get("warmup_steps_left"),
                    "phase": _w.get("phase"),
                    "has_convergence": bool(_w.get("has_convergence")),
                    # 求解器跳窗按 solver_index 映回来（见下）
                    "solver_skip": None,
                })
        # 把 stage 结果的 solver 读数按 solver_index 映射回子窗。
        _sr = self._read_stage_result() or {}
        _sk_by_solver = {
            int(x.get("window_index", -1)): x
            for x in (_sr.get("skipped_windows") or [])
        }
        _ov_by_solver = {
            int(x.get("window_index", -1)): x
            for x in (_sr.get("window_overlap_diagnostics") or [])
        }
        # 🔑🔑 [2026-09-14] **完成 = 过最终的 solver 门，不是"过了入场下限"。**
        # 先前只看"自检通过 + 没被跳窗"，而跳窗门是**入场**下限（10 帧）；
        # 求解器一旦让它进了协方差链，子窗就从待补列表消失 —— 可最终门要的是
        # `min_decorrelated_samples`（20）。于是 11/20 的子窗被判"已完成"，
        # 而最终门报 `worst_window = solver 索引`，9c 发的是**不带 `unit_id`** 的
        # 普通补帧，执行器把一个越界的窗口号交给原窗口路径。
        _final_floor = _sr.get("min_decorrelated_samples_threshold")
        for _u in sampling_units:
            _si = int(_u["solver_index"])
            _u["solver_skip"] = _sk_by_solver.get(_si)
            _ov = _ov_by_solver.get(_si) or {}
            _u["solver_n_frames_decorrelated"] = _ov.get("n_frames_decorrelated")
            _u["solver_min_decorrelated_samples_threshold"] = _final_floor
            _nd = _ov.get("n_frames_decorrelated")
            _meets_final = (
                None if (_final_floor is None or _nd is None)
                else int(_nd) >= int(_final_floor)
            )
            _u["meets_final_solver_gate"] = _meets_final
            # 🔑🔑 [CTL-02，2026-09-14] **调度状态 ≠ 失败归因。**
            #
            # 先前只有一个 `complete` 布尔，而它要求 `self_sufficient is True` ——
            # 那个量**混合**了帧数、`N_eff/g`、top1% 三项。分支 1c 对所有
            # `complete=False` 一律发 `RUN_PRODUCTION` ⟹ **偏斜类**失败
            # （top1% / raw ESS）的子窗被**反复加帧**，而加帧治不了偏斜
            # （§5.1 实测 250k→1M 让 top1% 从 0.545 涨到 0.762、ESS 比值反而更差）。
            #
            # 现在拆成两个正交的量：
            #   · `needs_frames`  —— **样本量**不够（还没测够）⟹ 动作是加帧；
            #   · `support_failed` —— **支撑/偏斜**不合格 ⟹ 加帧治不了，要缩跨度；
            #     没有有界的缩跨度动作时，如实停在 NO_FEASIBLE_ACTION，
            #     **不许拿加帧顶替**。
            # `complete` 保留，但只表示"两者都没问题"，不再被当成动作依据。
            _self_src = str(_u.get("self_verdict_source") or "")
            # ⚠️ `verdict_source` 只说明"这个结论由哪个判据得出"，**通过**的窗口
            # 同样带 `min_n_eff_over_g` —— 只看来源会把健康窗口也标成支撑失败。
            # 必须先确认 verdict 本身是**失败**。
            # 归因口径（TODO「四个量不许混用」）：
            #   · `top1pct_veto`       = 权重塌缩，**否决警报** ⟹ 加帧治不了
            #   · `HARD_INSUFFICIENT`  = 支撑低到测不出来   ⟹ 同上
            #   · `solver_eligibility` = 纯帧数不够 ⟹ 按样本量处理，加帧**正是**对症
            # [审计 #46] 归因判定收敛到 `support_failure_is_skew()` 这一份实现 ——
            # `solver_eligibility`（纯帧数不够，写侧被强制置成 `HARD_INSUFFICIENT`）
            # 原来在这里被当成"加帧治不了"，与写侧「帧数不够 ≠ 支撑不够」直接冲突。
            # 🔑 [2026-09-15] `min_n_eff_over_g` 从"样本量"改判到**偏斜**（真机
            # win1：n_decorr=888、比值 8.68 —— 帧一点不缺，是权重压不到目标态上）。
            # 见 `_SAMPLE_SIZE_VERDICT_SOURCES` 那段。帧数一并传进去自验。
            # 🔑🔑 [2026-09-17 P0] **子窗直接存三态，不许再压成布尔。**
            # 先前 `_support_failed = support_failure_is_skew(...)` 把三态压回两态，
            # 于是 `UNKNOWN` 被读成"不是 structural" ⟹ 走进下面的 `needs_frames`
            # ⟹ 从「错误授权 D3 停机」换成「错误授权补帧」。同一个语义丢两次。
            # ⚠️ 而且原调用**只传了 n_decorrelated/min_frames**，没传 ratio/target/
            # headroom ⟹ 射程恒判不了 ⟹ 恒 `None`。三个参数必须真正接进来，
            # 否则三态里只会出现两态。
            # ⚠️ 射程要块账，而块账在本循环**之后**才算出来 ⟹ 这里先按"没有射程"
            # 归因（⟹ `min_n_eff_over_g` 那支恒 UNKNOWN），**块账就位后由下面的
            # 第二遍重算**。别在这里省掉第二遍：省了就等于永远判不出射程。
            _attr = support_failure_attribution(
                _u.get("self_verdict"), _self_src,
                n_decorrelated=_u.get("self_n_frames_decorrelated"),
                min_frames=_u.get("self_min_frames"))
            _support_failed = (_attr == SUPPORT_FAILURE_STRUCTURAL)
            # 🔑 [2026-09-17] **三条「独立于归因」的补帧理由**，单独存 ——
            # 它们说的是"证据本身还没产出来"，跟这次失败是样本量还是结构性无关。
            # 第二遍重算归因时**不许**把它们一起 AND 掉（实测那样会让一个还没跑完
            # 的子窗拿不到帧，`test_an_incomplete_child_gets_a_run_production…` 变红）。
            _needs_frames_independent = bool(
                not _u["has_convergence"]
                or _u.get("solver_skip") is not None
                # 读不到最终门的读数时**不算过**（缺证据 ≠ 通过）⟹ 先补帧拿证据。
                or _meets_final is not True
            )
            _u["needs_frames_independent"] = _needs_frames_independent
            _needs_frames = bool(
                _needs_frames_independent
                # 🔑 自检判不够 **且归因明确是样本量类** ⟹ 才补帧。
                # `UNKNOWN` 不授权补帧（先前 `not _support_failed` 把它放行了）。
                or (_u.get("self_sufficient") is False
                    and _attr == SUPPORT_FAILURE_SAMPLE_SIZE)
            )
            # [审计 #7] 不可调度的条目（ABANDONED_NO_PRODUCT / 卡住的
            # SAMPLING_INTENT）**一律不产生动作**：它的帧进不了求解，补给它的每一块
            # 都是纯烧 GPU。留在视图里只为审计。
            if not _u.get("schedulable", True):
                _needs_frames = False
                _support_failed = False
                _attr = None
            _u["needs_frames"] = _needs_frames
            _u["support_failed"] = bool(_support_failed)
            _u["support_failure_source"] = _self_src or None
            # 三态原样带到调度层。`UNKNOWN` 既不是"要补帧"也不是"结构性失败"，
            # 它是**归因判不出来** —— 不得授权 RUN_PRODUCTION / D3 停机 /
            # IMMUTABLE_REWINDOW / SPLIT_TAIL_WINDOW / INSERT_LAMBDA 任何一个。
            _u["support_attribution"] = _attr
            _u["attribution_unknown"] = bool(_attr == SUPPORT_FAILURE_UNKNOWN)
            # `complete` 只在**没有任何失败归因**时成立（`_attr is None` = 自检通过）。
            _u["complete"] = bool(not _needs_frames and _attr is None)

        # solver 索引 → unit_id 的反查表：最终门报的是 solver 索引，
        # 而调度身份是 `unit_id` —— 没有这张表就只能把越界的窗口号交给原窗口路径。
        solver_index_to_unit = {
            int(u["solver_index"]): str(u["unit_id"]) for u in sampling_units
        }

        immutable_rewindow = {
            "identity": (_rw_latest or {}).get("identity"),
            "parent_window": (_rw_latest or {}).get("parent_window"),
            "child_ranges": (_rw_latest or {}).get("child_ranges"),
            "n_blocks": len(((_rw_latest or {}).get("blocks") or [])),
            # 只有**当前布局**下建过的才算"已替代"；旧布局的记录不得挡住新修补。
            # 🔑🔑 [2026-09-17，用户拍板 D3] **`status` 必须参与过滤，否则会静默丢窗。**
            #
            # 这里先前只看 `path_version`。而 `parents_done` 有两个消费者，
            # 第二个的后果比第一个重得多：
            #   · `rewindow_feasible()` 拿它判「这个父窗已经切过一层」⟹ 不再切；
            #   · **CTL-03 拿它把父窗整个踢出 `earliest` 排序**（理由是"它已退出
            #     求解覆盖、子系综接管了那段 λ"）。
            # 于是一条 `ABANDONED_NO_PRODUCT`（子系综建了但一帧都没产出）或卡住的
            # `SAMPLING_INTENT` 记录，会让那个物理窗口**永久隐身**：当不上 earliest、
            # 退役轮不到它、也不能再建第二个 rewindow —— 而它那段 λ **根本没有人接管**。
            # 这是"静默丢窗"那一族（同 `_solve_with_rewindow_children` 只采信
            # `SAMPLED` 那条既有先例，见 `sampling_units` 里 `_schedulable` 的注释）。
            #
            # 判据与那一处**同一份**：只有 `SAMPLED` 才算"这个父窗真的被接管了"。
            # 兜底同样是「老条目缺 `status` 按 `SAMPLED`」—— 删了会把所有既有 run 的
            # 子系综一次性判废。
            "parents_done": sorted({
                int(v["parent_window"]) for v in _rw_ledger.values()
                if isinstance(v, dict) and v.get("parent_window") is not None
                and _rw_entry_is_current(v)
                and str(v.get("status") or "SAMPLED") == "SAMPLED"
                and len(v.get("child_ranges") or []) >= 2
            }),
            # 同一父窗在当前布局下的**总尝试次数**（不分 status）——
            # `rewindow_feasible` 拿它封"建→废→再建"的无限循环。
            "attempts_by_parent": {
                str(_pw): sum(
                    1 for v in _rw_ledger.values()
                    if isinstance(v, dict) and _rw_entry_is_current(v)
                    and v.get("parent_window") is not None
                    and int(v["parent_window"]) == int(_pw))
                for _pw in {
                    int(v["parent_window"]) for v in _rw_ledger.values()
                    if isinstance(v, dict) and v.get("parent_window") is not None
                    and _rw_entry_is_current(v)}
            },
            # 建过但**没被接管**的父窗（status 非 SAMPLED / 子窗不全）。
            # 它们**不**进 `parents_done` ⟹ 照常参与 earliest 排序、照常可以再修。
            # 单独列出来只为让人看得见「这里试过一次、没成」。
            "parents_abandoned": sorted({
                int(v["parent_window"]) for v in _rw_ledger.values()
                if isinstance(v, dict) and v.get("parent_window") is not None
                and _rw_entry_is_current(v)
                and (str(v.get("status") or "SAMPLED") != "SAMPLED"
                     or len(v.get("child_ranges") or []) < 2)
            }),
            "entries_from_other_path_versions": sorted({
                str(v.get("identity")) for v in _rw_ledger.values()
                if isinstance(v, dict) and not _rw_entry_is_current(v)
            }),
            # 子系综各自预热、各自锁一份 f_k ⟹ 与基准段**不是同一份**账，
            # 残差/支撑度证据分开记，不许混进基准段。
            "f_k_scope": (_rw_latest or {}).get("f_k_scope"),
        }
        path = self._read_path()
        expected = len(path["window_ranges"] or []) or None
        windows = [self._read_window(i) for i in sorted(found)]
        # 🔑🔑 [2026-09-17] **单段视图跑与聚合视图同一份布局判定。**
        # 先前这四个量只有 `read_aggregated()` 算，单段视图整套都没有 ⟹
        # `_coverage_incomplete()` 的 `out_of_range_windows` 与分支 0a /
        # `_evidence_status` 的 `stale_layout_evidence` 两道闸在单段上**恒不触发**
        # （后者更隐蔽：它按 `stale_layout_evidence_only` 标记算，而那个标记只有
        # 聚合视图的占位记录才会设 ⟹ 恒 `{}`）。
        # 补空默认值不行 —— 那只让 schema 表面一致、守卫照样失效，所以判定共用
        # `classify_layout_evidence()`。
        _cls_single = classify_layout_evidence([(self.stage_name, windows)], path)
        # ⚠️⚠️ **只出视图键，不动逐窗记录。** 我一度顺手给 STALE 的窗口打上
        # `stale_layout_evidence_only`，理由是"让 `_window_routing_state` 也看见"。
        # 那是**超范围**的改动，而且有真实爆炸半径：那个标记被
        # `first_untrusted_window()` 读，它决定 tail anchor ——
        # 实测四个窗口被判 stale 之后 anchor 从 win3 变成 win0，
        # `SPLIT_TAIL_WINDOW` 整条链的四个测试当场红。
        # 逐窗标记是**聚合视图的占位记录机制**（它把 STALE 记录换成一条清空证据的
        # 占位行，因为别的段可能有有效证据）；单段没有"别的段"、也没有占位这一步，
        # 就不该借用那个标记。
        # 这里要补的那道闸（stale ⟹ 不得判 DONE）本来就走**视图键**：
        # 分支 0a 与 `_evidence_status` 读 `view.get("stale_layout_evidence")`，
        # `stale_layout_windows()` 也同时读视图键和逐窗标记 ⟹ 只出视图键，
        # 1d-0 那条路由照样生效，而路由顺序一个字节不变。
        # 生产侧累计 f_k 偏差在 **stage 结果**里（scope=production），逐窗填回。
        # ⚠️ 必须在 `stage` 读出来**之后**做 —— 顺序写反过一次，UnboundLocalError。
        _stage_for_cum = self._read_stage_result()
        _cum_by_win = {
            int(x.get("window_index", -1)): x
            for x in ((_stage_for_cum or {}).get(
                "cumulative_fk_residual_production") or [])
        }
        for _w in windows:
            _c = _cum_by_win.get(int(_w["window_idx"])) or {}
            _w["cum_fk_span"] = _c.get("cumulative_residual_span_kJ_mol")
            # 🔑🔑 [S2-A，2026-09-14] **算不出来 ≠ 没这回事，要显式记 `UNMEASURED`。**
            # 多段窗口的帧采自**两份不同偏置**，`_load_ibs_window_outputs_merged`
            # 因此显式 `base.pop("f_k")` ⟹ 求解器给的是
            # `{"error": "no_effective_f_k_for_these_frames"}`、没有 `verdict`。
            # 先前这里落成 `None`，而 `None` 在 5a-1 的语义是「证据还没产出 ⟹ 去跑
            # ANALYZE 把它算出来」—— 可这个窗口**结构上**就算不出来，于是又是一个
            # 无限 ANALYZE。`UNMEASURED` 才是真话，5a-2 已经按它分岔。
            # ⚠️ 有界 rewindow 的子系综**各自锁一份自己的 f_k**，所以它**不能**
            # 替父窗口把这道残差门宣称通过 —— 缺段证据仍然是 UNMEASURED。
            # 父窗口被有界 rewindow 的子系综取代之后，它**整个不在求解覆盖里**
            # （子系综的 window_index 带 10000 偏移），于是连一条残差记录都没有。
            # 那同样是 `UNMEASURED` 而不是 `None` —— 子系综各自锁自己的 f_k，
            # **不能**替父窗口把这道门宣称通过，也不该把控制器送去空跑 ANALYZE。
            _rewindowed = int(_w["window_idx"]) in set(
                immutable_rewindow.get("parents_done") or [])
            _w["cum_fk_verdict"] = (
                _c.get("verdict")
                or ("UNMEASURED" if (_c.get("error") or _rewindowed) else None)
            )
            # [S2-A 逐段门] 多段窗口给的是**逐段**结论，没有窗口级 span（那会是
            # 跨段望远镜相消）。逐段身份/帧范围/结论原样带上，供审计。
            _w["cum_fk_per_segment"] = _c.get("segments")
            _w["cum_fk_gate_is_per_segment"] = bool(_c.get("per_segment_gate"))
            _w["cum_fk_unmeasured_reason"] = (
                None if _c.get("verdict") else (
                    _c.get("error")
                    or ("replaced_by_immutable_rewindow_children_own_f_k"
                        if _rewindowed else None)
                )
            )
            _h = self._json(os.path.join(
                self.stage_dir,
                f"dual_window_{_w['window_idx']}_{self.stage_type}_heldout.json",
            )) or {}
            _w["heldout_verdict"] = _h.get("verdict")
            _w["heldout_worst_before"] = _h.get("worst_before")
            _w["heldout_worst_after"] = _h.get("worst_after")
        stage = self._read_stage_result()
        # 🔑🔑 [2026-09-15] **新契约的两维读法，fail-closed。**
        # `analysis_status` 只回答「分析跑完整了没有」（硬不变量），
        # `precision_status` 只回答「精度测过没有」（跨重复目标）。
        # 读不到 `analysis_status`（老产物、或写侧还没升级）⟹ **不得当成通过**，
        # 一律记 `ANALYSIS_INCOMPLETE` 并把理由写明。
        _stage_incomplete_reasons: List[str] = []
        if stage is None:
            _stage_analysis_status = "ANALYSIS_INCOMPLETE"
            _stage_incomplete_reasons = ["stage 结果尚未落盘"]
        else:
            _as = stage.get("analysis_status")
            if _as == "ANALYSIS_COMPLETE":
                _stage_analysis_status = "ANALYSIS_COMPLETE"
            elif _as == "ANALYSIS_INCOMPLETE":
                _stage_analysis_status = "ANALYSIS_INCOMPLETE"
                _stage_incomplete_reasons = [
                    str(x) for x in (stage.get("analysis_incomplete_reasons") or [])
                ] or ["写侧判了 ANALYSIS_INCOMPLETE 但没给理由"]
            else:
                _stage_analysis_status = "ANALYSIS_INCOMPLETE"
                _stage_incomplete_reasons = [
                    f"stage 结果里没有 `analysis_status`（读到 {_as!r}）⟹ "
                    "**缺键 fail-closed**，不得当成通过。"
                    "（`converged` 已于 2026-09-15 删除，不做兼容别名。）"
                ]
        # 精度同样 fail-closed：读不到就是 `UNMEASURED`（没测 ≠ 达标）。
        _stage_precision_status = str(
            ((stage or {}).get("precision_status") or "UNMEASURED")
        )
        if _stage_precision_status not in (
                "UNMEASURED", "MEETS_CROSS_REPEAT_TARGET",
                "EXCEEDS_CROSS_REPEAT_TARGET"):
            _stage_precision_status = "UNMEASURED"
        # 缺窗口：布局里有、产物里没有。这是"截断的 ΔG"这类失效的直接信号，
        # 现在只在日志里出现一次 WARN。stage 结果里的 skipped_windows 是另一种
        # （产物在、但去相关后帧数不足被踢出协方差链），两者都要算进来。
        missing = [i for i in range(expected) if i not in found] if expected else []
        # 🔑🔑 [审计 #58，2026-09-14] **`skipped_windows` 混了两个命名空间。**
        # 求解器报的 `window_index` 对物理窗口是下标、对 rewindow 子窗是
        # **solver 命名空间**的偏移索引（≥ `SOLVER_UNIT_INDEX_BASE`）。先前两者
        # 一起塞进 `skipped_windows`，后果：
        #   · 分支 0a 的 `DONE` 与 `_evidence_status` 的 `CONVERGED` 都查
        #     `view["skipped_windows"]` 非空 ⟹ 一个被跳的**子窗**会以一个**根本
        #     不存在的物理窗口号**（10000+）永久封死 DONE；
        #   · `render` 还把 `10000` 当窗口号打印给人看。
        # 拆成两个键：`skipped_windows` 只放物理窗口，`skipped_sampling_units`
        # 放翻成 `unit_id` 的子窗（翻不出来的保留 solver 索引，如实标出来）。
        _skip_all = {
            int(x.get("window_index", -1)): x
            for x in ((stage or {}).get("skipped_windows") or [])
        }
        _skip_recs = {i: r for i, r in _skip_all.items()
                      if i < SOLVER_UNIT_INDEX_BASE}
        skipped = sorted(_skip_recs)
        skipped_units = sorted(
            str(solver_index_to_unit.get(i) or f"solver_index:{i}")
            for i in _skip_all if i >= SOLVER_UNIT_INDEX_BASE
        )
        # 🔑🔑 [DECORR-01，2026-09-14 裁决] **求解器对「能否进入求解、是否跳窗」
        # 有操作权威。** 它实际用的去相关帧与 `skipped_windows` 决定这个窗口进不进
        # MBAR 与协方差链；自检的 `n_decorr` 只是**早期 target-support 诊断**，而且
        # `self_sufficient` 里还混着 `N_eff/g` 与 top1% 的条件，**不能**被读成
        # "求解器帧数够"。
        #
        # 实测（cyclod_ligand2/rep2）：win3 自检 56 帧、`sufficient=True`，求解器
        # 报 9 / 7 帧并**跳窗**；win0 自检 21 帧、g=24.5、`sufficient=False`，
        # 求解器报 9。两个数**门着不同的东西**，差 2–8 倍。
        #
        # ⚠️ **不让两边数字强行看齐**：去相关抽样依赖输入时间序列与所选 g，
        # 拿另一份输出的帧数反推它是错的（pymbar timeseries 的语义）。这里只做
        # 一件事 —— 把两侧**分别命名、分别展示**，并让 solver skip 优先。
        _ov_by_win = {
            int(x.get("window_index", -1)): x
            for x in ((stage or {}).get("window_overlap_diagnostics") or [])
        }
        _final_floor_w = (stage or {}).get("min_decorrelated_samples_threshold")
        for _w in windows:
            _sk = _skip_recs.get(int(_w["window_idx"]))
            _w["solver_skip"] = _sk            # None = 求解器没跳它
            # 🔑 [2026-09-14] **求解器侧的去相关帧数**（不是自检那份，两者差 2–8 倍）。
            # 这是"再给一块帧到底有没有用"的**判据量** —— 补帧准入看它的增长，
            # 而不是看"又跑了 25 万步"。
            _w["solver_n_frames_decorrelated"] = (
                (_ov_by_win.get(int(_w["window_idx"])) or {}).get("n_frames_decorrelated")
                if _sk is None
                else _sk.get("n_frames_after_decorrelation")
            )
            _w["solver_min_decorrelated_samples_threshold"] = _final_floor_w
            _w["evidence_decorrelation"] = {
                # —— 求解器侧：**操作权威**，决定进不进求解 ——
                "solver": None if _sk is None else {
                    "n_frames_after_decorrelation": _sk.get(
                        "n_frames_after_decorrelation"),
                    "min_frames_per_window": _sk.get("min_frames_per_window"),
                    "statistical_inefficiency": _sk.get("statistical_inefficiency"),
                    "statistical_inefficiency_per_lambda": _sk.get(
                        "statistical_inefficiency_per_lambda"),
                    "lambda_indices": _sk.get("lambda_indices"),
                    "reason": _sk.get("reason"),
                    # 段名在**这一层**才是确定的（合并视图里会被覆盖成胜出段）。
                    "segment": self.stage_name,
                    "observable": "u_kn_local(sampled_row)",
                },
                # —— 自检侧：早期诊断，**不是**求解器资格 ——
                "self_check": {
                    "n_frames_decorrelated": _w.get("self_n_frames_decorrelated"),
                    "min_frames": _w.get("self_min_frames"),
                    "verdict": _w.get("self_verdict"),
                    "verdict_source": _w.get("self_verdict_source"),
                    "sufficient_INCLUDES_N_eff_AND_top1pct": _w.get("self_sufficient"),
                    "min_n_eff_over_g": _w.get("min_n_eff_over_g"),
                    "observable": "window_self_support_check",
                },
                "authority": "solver_decides_eligibility_self_check_is_diagnostic",
                "do_not_reconcile": (
                    "两侧的去相关抽样依赖各自的输入帧集与所选 g；"
                    "不得用一侧的帧数反推另一侧，也不得为了让数字一致而改任一侧。"
                    "要诊断差异请用**完全相同的输入**另跑一次对照。"
                ),
                "protocol_version": STAGE2_CONTROLLER_PROTOCOL_VERSION,
            }
        # ---- 生产预算账（与预热账**分开**）----
        # 🔑🔑 [审计 #38/#39] **未知不是零，而且单段视图与合并视图必须同一份实现。**
        # 先前这里 `int(... or 0)`：读不到步数与"确实 0 步"压成同一个数，
        # `usage_complete` / `unknown_usage` 两个字段根本不存在 ⟹ 单段视图与合并
        # 视图对**同一个量**给两个答案。现在两处都走 `_make_production_budget()`。
        _pw_used: Dict[int, Any] = {}
        _unknown: List[Any] = []
        for w in windows:
            _ps = w.get("production_steps")
            if _ps is None:
                if w.get("has_convergence"):
                    _unknown.append({"window_idx": int(w["window_idx"]),
                                     "segment": self.stage_name})
                continue
            _pw_used[int(w["window_idx"])] = int(_ps)
        _unit_used: Dict[str, Any] = {}
        for u in sampling_units:
            _ps = u.get("production_steps")
            # 读不到就**登记为未知**（None），不写 0 —— 读侧的未知检测靠它才成立。
            _unit_used[str(u["unit_id"])] = None if _ps is None else int(_ps)
            if _ps is None and u.get("has_convergence"):
                _unknown.append({"unit_id": str(u["unit_id"])})
        _pv_now = (path or {}).get("version")
        _blocks_same_seg = self._production_blocks_ledger(
            windows, _pv_now, units=sampling_units)
        _blocks_all_seg = self._production_blocks_total_by_window(
            windows, _pv_now, units=sampling_units)
        # 🔑🔑 [2026-09-17 P0] **子窗归因第二遍：块账就位后把射程补上。**
        # 上面第一遍跑在块账之前，`frames_headroom` 拿不到 ⟹ `min_n_eff_over_g`
        # 那支恒 `None` ⟹ 恒 UNKNOWN。三态里只会出现两态。
        # 这里用**真实的子窗块账**重算一次；三个输入缺任何一个仍然是 UNKNOWN
        # （那是对的：判不了就是判不了，不许兜成 STRUCTURAL 去授权缩跨度）。
        _hr_view = {
            "max_production_blocks_per_window": int(self.max_blocks_per_window),
            "production_blocks_total_by_unit": (_blocks_all_seg or {}).get("by_unit") or {},
            "production_blocks_by_unit": (_blocks_same_seg or {}).get("by_unit") or {},
        }
        for _u in sampling_units:
            if not _u.get("schedulable", True):
                continue
            _attr2 = support_failure_attribution(
                _u.get("self_verdict"), _u.get("support_failure_source"),
                n_decorrelated=_u.get("self_n_frames_decorrelated"),
                min_frames=_u.get("self_min_frames"),
                min_n_eff_over_g=_u.get("min_n_eff_over_g"),
                n_eff_over_g_target=_u.get("self_n_eff_over_g_eligible"),
                frames_headroom=frames_growth_headroom(
                    _hr_view, None, unit_id=str(_u.get("unit_id"))),
                # [2026-09-18] η≤1 硬上界（子窗与物理窗口同一判据）
                n_eff_input_n_frames=_u.get("bottleneck_n_eff_input_n_frames"),
                bottleneck_g=_u.get("bottleneck_g"))
            _u["support_attribution"] = _attr2
            _u["support_failed"] = bool(_attr2 == SUPPORT_FAILURE_STRUCTURAL)
            _u["attribution_unknown"] = bool(_attr2 == SUPPORT_FAILURE_UNKNOWN)
            # [2026-09-18] 第四态：瓶颈在 N/g，缩跨度类被 η≤1 证伪 ⟹ 既不是
            # `support_failed`（那一档授权缩跨度）也不是 `attribution_unknown`
            # （它判得出来，只是判出的是"改 η 没用"）。单独标，别被压回三态。
            _u["n_over_g_bound"] = bool(_attr2 == SUPPORT_FAILURE_N_OVER_G_BOUND)
            # ⚠️ **整式重算**，不是在旧值上 AND 掩码：三条独立理由与归因是**或**的
            # 关系，掩码会把它们一起抹掉。
            _u["needs_frames"] = bool(
                _u.get("needs_frames_independent")
                or (_u.get("self_sufficient") is False
                    and _attr2 == SUPPORT_FAILURE_SAMPLE_SIZE))
            _u["complete"] = bool(not _u["needs_frames"] and _attr2 is None)

        production_budget = self._make_production_budget(
            cap=self.production_cap_steps,
            cap_source=self.production_cap_source,
            per_window_used=_pw_used,
            per_unit_used=_unit_used,
            unknown_usage=_unknown,
            note=(
                "单段视图：只算本段的生产步数（跨段累计见合并视图）。"
                "生产补帧花的是这本账；换 Epoch 的最低验证额度花的是 warmup 账"
                "（`warmup_steps_left`）。两者不可互相代替 —— 拿预热余额去终止一个"
                "只花生产帧的动作，正是 2026-09-14 修掉的那个缺陷。"
            ),
        )

        return {
            "protocol_version": STAGE2_CONTROLLER_PROTOCOL_VERSION,
            # 🔑🔑 [2026-09-17] **两个视图构造器的键集必须一致。**
            # 这两个键先前**只在 `read_aggregated()` 里产出**。生产走
            # `for_physical_stage` ⟹ 聚合视图 ⟹ 生产没事；但单段视图
            # （`Stage2RepairController(run, stage)` 直接构造，测试与离线回放都走它）
            # 缺了它们，而 `_decide_once` 照读不误：
            #   · `min_n_eff_over_g_history` 缺 ⟹ `_hist = {}` ⟹ **加帧刹车恒不触发**；
            #   · `stale_layout_evidence` 缺 ⟹ 分支 0a 与 `_evidence_status` 的
            #     「证据被布局作废」守卫 **恒放行** ⟹ 可以带着过期证据判 DONE。
            # 两条都是**静默**失效：不报错、不留痕，只是判据悄悄不生效。
            # 这里给出单段的**退化值**（数据本来就读了，不重写那份跨段逻辑）：
            #   · 历史 = 本段块账里逐块的 min N_eff/g（跨段合并是聚合视图的事）；
            #   · stale = 逐窗 `stale_layout_evidence_only` 标记（单段没有"别的段"
            #     可比，所以它就是全部）。
            # 由 `tests/test_no_phantom_view_keys.py` 的键集对账钉住不再分叉。
            "min_n_eff_over_g_history": {
                int(_i): [(r.get("segment"), r.get("min_n_eff_over_g"))
                          for r in (_rows or [])
                          if r.get("min_n_eff_over_g") is not None]
                for _i, _rows in ((_blocks_same_seg or {}).get("by_window") or {}).items()
            },
            "stale_layout_evidence": {
                int(k): list(v) for k, v in _cls_single["stale"].items()},
            # 下面四个键先前**只有聚合视图产出**，单段缺 ⟹ 读侧静默失效
            # （见 `classify_layout_evidence` 的长注释）。给的是单段的等价语义，
            # 不是空默认值。
            "unverifiable_layout_evidence": {
                int(k): list(v) for k, v in _cls_single["unverifiable"].items()},
            "out_of_range_windows": sorted(_cls_single["out_of_range"]),
            "layout_change_invalidated_segments":
                self.layout_change_invalidated_segments(),
            "window_provenance": {
                int(w["window_idx"]): self.stage_name for w in windows},
            # 单段就是"只有自己这一段"。执行器的 `_disk_signature` 读它，
            # 缺了停滞签名少一维。
            "aggregated_segments": [self.stage_name],
            "immutable_rewindow": immutable_rewindow,
            "sampling_units": sampling_units,
            "solver_index_to_unit": solver_index_to_unit,
            "noop_actions": noop_actions,
            # 🔑🔑 [2026-09-14 真机，同形状第 4 次] **这个键此前根本不存在。**
            # `action_noop_fingerprint()` 的两个调用方（控制器判读 / 执行器记账）
            # 都写 `view.get("path_version")` —— 而路径版本在 `view["path"]["version"]`。
            # 两侧一致地拿到 None ⟹ 指纹里的版本位恒为 0 ⟹ **布局一变就该失效的
            # no-op 记录，永远不会因布局变化而失效**，尽管它的 docstring 明写
            # 「记的是『在这个盘面上没用』，不是『这个动作永远没用』」。
            # 插 λ / 拆末窗恰恰是「让一个先前没用的动作重新变得有用」的操作。
            "path_version": (path or {}).get("version"),
            "production_blocks_by_window": _blocks_same_seg["by_window"],
            "production_blocks_by_unit": _blocks_same_seg["by_unit"],
            # 🔑 [BUD-03] 硬上限用的是**跨段累计**的那本（换段不重置配额）。
            "production_blocks_total_by_window": _blocks_all_seg["by_window"],
            "production_blocks_total_by_unit": _blocks_all_seg["by_unit"],
            "max_production_blocks_per_window": int(self.max_blocks_per_window),
            "production_budget": production_budget,
            "_production_budget_inputs": {
                "cap": self.production_cap_steps,
                "cap_source": self.production_cap_source,
                "reserve": int(self.new_ensemble_reserve_steps),
                "block": int(self.production_block_steps),
                # [审计 #38] **未知就写 None**，不许 `int(... or 0)` —— 读侧
                # （`read_aggregated` 的 `if _v is None`）就是靠它才不是死代码。
                "unit_used": dict(_unit_used),
            },
            "run_dir": self.run_dir,
            "stage_name": self.stage_name,
            "stage_type": self.stage_type,
            "min_states_per_window": self.lo,
            "max_states_per_window": self.hi,
            "config_source": self.config_source,
            "segment_index": self.segment_index,
            "checkpoint_dir_used": self.checkpoint_dir,
            # 插点是有**终身**预算的（rounds_done 从版本链累计，跨 resume 有效）。
            # 🔑 [S2-H/F] 自治历史原样带进视图：`structural_action_refused()` 要用它
            # 数「这个改盘面的动作对这个窗口发过几次、每次的验收量是多少」。
            # 已经在写的产物（`_production_blocks_scan` 也读它），不新造。
            "autonomous_history": self._json(os.path.join(
                self.path_checkpoint_dir, "stage2_autonomous_history.json")) or {},
            "path_insertions_done": int(path["events"].get("insert_lambda", 0)),
            # 尾段重分过几次 —— 崩溃恢复靠它判「已经重分过没有」，
            # 没有它第二次启动会重新重分、把刚跑的新尾段作废。
            "tail_repartitions_done": int(path["events"].get("tail_repartition", 0)),
            "path_insertions_budget": self.max_path_insertions,
            # 🔑🔑 [2026-09-11 更正] **不许把各窗余量求和当"全局预算"。**
            #
            # `cumulative_cap_steps` 是**逐窗**的（实测每窗 955k = max_bias_warmup
            # 900k + burn-in 5k + 验证预留 50k），**没有任何全局池** ——
            # 你不能拿 win0 的余额去给 win4 花。求和（实测 4.29M）是个**无意义的量**，
            # 而且后果严重：它会让"全局还有钱"几乎永真 ⟹ 所有 HALT_BUDGET 都被判成
            # **路由** ⟹ 主循环**永不终止**。
            #
            # 正确口径：
            #   · 路由/终止用**相关窗口自己的**余量；
            #   · "全局耗尽"在双层预算（PLAN 第 5 步 F）做出来之前**没有可靠数据源**，
            #     这里实现成保守占位：**所有**窗口余量都为 0 才算全局耗尽。
            # 🔑🔑 [BUD-05，2026-09-14] **未知就写 None，不许写 0。**
            # 先前 `int(... or 0)` 把「账本读不到」写成「余量 0」= 耗尽 —— 而同一个
            # 量在 `_epoch_validation_unaffordable` 里是 fail-closed、在 `decide()`
            # 分支 1e 里是 `left is not None and <=0`（未知 = 有钱）。同一个未知，
            # 三套语义。实测 cyclod_ligand1/rep3 win4 的 ledger 确实读不到，
            # 在这里被显示成 0。规矩（TODO §1）：**未知不是零，两个方向都不是。**
            "per_window_budget_remaining": {
                int(w["window_idx"]): (
                    None if w.get("warmup_steps_left") is None
                    else int(w["warmup_steps_left"])
                )
                for w in windows
            },
            # ⚠️ 这是**预热/验证**账，不是生产账。名字里没写"warmup"是历史遗留；
            # 换 Epoch 类动作查它，补生产帧**绝不**查它（见 production_budget）。
            # 🔑 [BUD-05] 「全部耗尽」是个**断言**，账不完整就不成立：
            # 只要有一个窗口的余量未知，就不得宣称全局耗尽（否则是拿少算过的账
            # 去论证一个终止性结论）。
            "all_windows_budget_exhausted": bool(
                windows
                and all(w.get("warmup_steps_left") is not None for w in windows)
                and all(int(w["warmup_steps_left"]) <= 0 for w in windows)
            ),
            "global_budget_source": "per-window warmup cap (NOT the production budget)",
            "path_insertions_left": max(
                0, self.max_path_insertions - int(path["events"].get("insert_lambda", 0))
            ),
            "path": path,
            "n_windows_found": len(windows),
            "n_windows_expected": expected,
            "missing_windows": missing,
            # [审计 #58] 只放**物理窗口**；被跳的子窗在下面那个键里（unit_id 命名空间）。
            "skipped_windows": skipped,
            "skipped_sampling_units": skipped_units,
            "production_rescue_targets": (stage or {}).get("production_rescue_targets") or {},
            # 🔑🔑 [2026-09-15 老板定案] `converged` 已删除，**不做同名改义、
            # 不做兼容别名**。新契约两维分开：
            #   · `analysis_status`  —— 只含**硬不变量**（求解跑完了没有、
            #     窗口丢没丢）。它 **≠ 可以判 DONE**。
            #   · `precision_status` —— 跨重复精度。单次 run 里恒为 `UNMEASURED`：
            #     一个 run 本来就无法自证精度。
            # **fail-closed**：读不到 `analysis_status` 的老产物一律当
            # `ANALYSIS_INCOMPLETE`，绝不当成通过。
            "stage_analysis_status": _stage_analysis_status,
            "stage_analysis_incomplete_reasons": _stage_incomplete_reasons,
            "stage_precision_status": _stage_precision_status,
            "stage_precision_evidence": (stage or {}).get("precision_evidence") or {},
            "stage_path_is_complete": (stage or {}).get("path_is_complete"),
            "stage_total_delta_G": (stage or {}).get("total_delta_G"),
            "has_stage_result": stage is not None,
            # 🔑🔑 [2026-09-14 自查] **这份 stage 结果的版本到底核实过没有。**
            # `_read_stage_result()` 对**不带 `path_version` 的老产物**是"无法判定
            # 版本 ⟹ 放行"（保持既有行为、不制造新 fail）。而 stage 缓存
            # `stage2_vanishing.json` 恰恰不盖版本号——只有
            # `_persist_inprogress_stage_result` 写的中间产物盖。
            # ⟹ 「布局 v1 收敛过、后来插了 λ 变 v2」时，中间产物因版本不符被跳过，
            #    老缓存被放行，`converged=True` 描述的却是**另一条布局**。
            # 任何**据此终止**的判断（分支 0a 的 DONE）必须先问这一条；
            # 只用来路由的读法不受影响。
            "stage_result_path_version_verified": bool(
                stage is not None
                and stage.get("path_version") is not None
                and path.get("version") is not None
                and int(stage["path_version"]) == int(path["version"])
            ),
            "joins": joins,
            "fk_probe": fk_probe,
            "windows": windows,
        }

    # ------------------------------------------------------------ 可行性

    def feasible(self, view: Optional[Dict[str, Any]] = None, n_insert: int = 1) -> Dict[str, Optional[str]]:
        """结构性动作在当前布局下可不可行。

        复用模块级的 `feasible_repair_actions` —— 那条规则
        （可拆区间 `[2lo−1, 2hi−1]`、溢出槽上界）必须与
        `insert_lambda_in_failed_ibs_window` 里的守卫**同一份**，不许各写一遍。
        """
        view = view or self.read()
        ranges = view["path"]["window_ranges"]
        n_states = view["path"]["n_states"]
        if not ranges or not n_states:
            return {
                "insert_lambda": "读不到路径版本链，无法判断布局可行性",
                "split_tail_window": "读不到路径版本链，无法判断布局可行性",
            }
        # 🔑🔑 [审计 #24，2026-09-14] **必须把重分起点传下去。**
        # `SPLIT_TAIL_WINDOW` 落到执行器是 `repartition_tail_from_anchor`：
        # 从 `first_untrusted_window` 的**首态**起重分整个尾段。不传这个参数时
        # `feasible_repair_actions` 会如实返回「判不了」（**不再拿「末窗能否一分为二」
        # 冒充**），于是这个动作永远不可行 —— 所以这里必须算出来传进去。
        # 取不到（没有不可信窗口 / 读不到 λ 表）就不传，保持 fail-closed。
        _t_idx = self.first_untrusted_window(view)
        _t_start = None
        if _t_idx is not None and 0 <= int(_t_idx) < len(ranges):
            _t_start = int(ranges[int(_t_idx)][0])
        out = feasible_repair_actions(
            ranges, int(n_states),
            min_states_per_window=self.lo, max_states_per_window=self.hi,
            n_insert=int(n_insert),
            tail_repartition_start_state=_t_start,
        )
        # 🔑🔑 [2026-09-14 真机] **「结构上拆得开」不等于「执行器拆得了」。**
        # ⚠️ 这道检查与上面那条是**互补**的，两道都要留：
        #   · `feasible_repair_actions` 问「布局上切不切得开」（能否切成更细的合法分窗）；
        #   · 这里问「**执行器拿不拿得到起点**」（anchor 是不是现有共享态）。
        # 上面那份只问布局：K 在不在可拆区间 [2lo−1, 2hi−1]。但执行器还需要一个
        # **anchor**（tail-only 重分的起点），取不到就只能跳过 —— 于是控制器发一个
        # 注定落空的动作，盘面不变，靠通用停滞探测连发 4 次才停。那是失败形状 ①
        # （动作对该窗口的状态在结构上不可能）被当成"再试一次"。
        # 可行性判据必须包含执行器真正需要的每一个前提，就在这一处。
        _tail_anchor = self.tail_repartition_anchor(view)
        if out.get("split_tail_window") is None and _tail_anchor is None:
            out["split_tail_window"] = (
                "取不到 tail anchor（没有 window_idx > 0 的不可信窗口，或读不到 λ 表）"
                " ⟹ 执行器无从下刀。这不是「再试一次」能变的。"
            )
        # 🔑🔑 [2026-09-16 真机] **插 λ 的可行性必须包含「插完以后还能不能合法化」。**
        #
        # 末窗豁免 `max_states_per_window` 只活在 path-version 层：插 λ 让末窗吸收
        # 溢出，允许它涨到可拆上限 `2*hi−1`。但 stage 的权威 `window_ranges` 校验
        # 对**所有**窗口一律要求 K ∈ [lo, hi]，所以末窗一旦超过 `hi` 就**必须**先拆。
        # 而拆末窗要 tail anchor，anchor 取自 `first_untrusted_window`，
        # `tail_repartition_anchor` 在 `idx <= 0` 时恒为 None ——
        # **卡住的窗口是 window 0 时，拆末窗在构造上永远不可行。**
        #
        # ⚠️⚠️ [2026-09-17] **下面这段别叫它「死区」** —— 仓里已经有一个叫死区的东西，
        # 而且是**另一件事**：`abfe_pipeline._legalize_tail_window` 的 docstring 把死区定义为
        # `K ∈ (hi, 2*lo−1)`——既当不了单窗又**根本拆不开**（落在可拆区间
        # `[2lo−1, 2hi−1]` 之外），那是一个**纯区间性质**，与有没有 anchor 无关。
        # 而下面这一段 `hi < K_tail ≤ 2*hi−1` 恰恰**全部落在可拆区间里**：它机械上
        # 拆得开，只是**取不到下刀的点**。两件事共用一个词已经造成过一次误诊
        # （2026-09-17，K=9/lo=4/hi=8 被读成「落在死区」，而真实死区 (8,7) 是空集）。
        # 下面叫它**anchor 缺口**：吸收得进、拆得开、就是没人告诉它从哪切。真机
        # cyclod_ligand3/rep1 连插三次（窗口 0 每次都是失败窗口），末窗
        # K=6→7→8→9，第三次插完 K=9 > hi=8 ⟹ 执行器 fail-closed
        # `RuntimeError('末窗 K=9 超执行层上限 8 但取不到 tail anchor ⟹ 无法拆')`，
        # 而那个非法布局**已经落盘**（见 `_legalize_tail_window` 的调用顺序）。
        # cmet_ligand1/rep1、jnk1_ligand1/rep3、p38_ligand2/rep1 是同一个死局的
        # 「插 λ 预算先用完所以没崩」版本。
        #
        # 判据放在可行性这一处：插完会越过 `hi` 且拆不了 ⟹ 这个动作不可行，
        # 如实说出来，别发一个注定把布局搞成非法的动作。
        if out.get("insert_lambda") is None and int(n_insert) > 0:
            _tail_k_now = int(ranges[-1][1]) - int(ranges[-1][0])
            _tail_k_after = _tail_k_now + int(n_insert)
            if _tail_k_after > int(self.hi) and _tail_anchor is None:
                out["insert_lambda"] = (
                    f"插 {int(n_insert)} 个 λ 会让末窗从 K={_tail_k_now} 涨到 "
                    f"K={_tail_k_after} > 执行层上限 {self.hi}，而合法化只能靠拆末窗、"
                    "拆末窗又取不到 tail anchor（没有 window_idx > 0 的不可信窗口）"
                    " ⟹ 插完就是一个**拆不开的非法布局**。这不是「再试一次」能变的："
                    "要么先让 window 0 之后的某个窗口成为不可信窗口（anchor 才存在），"
                    "要么加大输入 λ 总数让末窗有余量。"
                )
        # 🔑🔑 [2026-09-15] **可行性 = 执行器的 dry-run，不是第二套闭式判据。**
        #
        # 上面 `feasible_repair_actions` 用的是闭式不等式
        # `m(lo−1)+1 ≤ T ≤ m(hi−1)+1` 且要求存在 m ≥ m0+1（更细），而执行器
        # `repartition_tail_from_anchor` 先前调的分窗器内部取 `n_windows =
        # min_windows`（最少窗口数，即最宽的窗）⟹ **判据要更细、执行器给更宽**。
        # 纸面复现见 `repartition_tail_from_anchor` 的参数注释。
        # 这正是本仓库反复复发的那个形状：同一个不变量两份实现
        # （`docs/archive/STAGE2_CONTROLLER_DESIGN_2026-09-12.md` 把它记成最贵的 bug）。
        #
        # 现在改成**直接把执行器跑一遍**：`_dry_run_tail_repartition` 是纯函数
        # （不落盘、不采样，微秒级），切得出来才叫可行。执行器落地时调**同一个**
        # 方法、同样的入参，确定性保证拿到同一份 ranges —— 两边不再各切一遍。
        # 方向/缩跨度的硬断言在执行器内部，一并在这里就被验过。
        # ⚠️ 不往这个 dict 里塞 ranges/diag：它的契约是 `{动作名: None | 不可行理由}`，
        # 调用方按 `v is None` 取可行集，混进非动作键迟早出事。
        if out.get("split_tail_window") is None:
            try:
                self._dry_run_tail_repartition(view)
            except Exception as _e:  # noqa: BLE001 —— 切不出来就是不可行，不是崩
                out["split_tail_window"] = (
                    f"执行器 dry-run 切不出更细的尾段：{_e}"
                )
        # 同理，插 λ 有**跨 resume 的累计上限**。原来这个数只出现在一行诊断打印里，
        # `decide()` 和执行器都不看 ⟹ 自治循环无限插点：cyclod_ligand1/rep2 在
        # max_path_insertions=3 下插了 6 次，末窗 K=6→12。失败形状 ⑥（没有停止条件）。
        if out.get("insert_lambda") is None:
            _left = int(view.get("path_insertions_left") or 0)
            if _left <= 0:
                out["insert_lambda"] = (
                    f"插 λ 的跨 resume 累计预算已用尽"
                    f"（max_path_insertions={self.max_path_insertions}，"
                    f"版本链上已插 "
                    f"{int(((view.get('path') or {}).get('events') or {}).get('insert_lambda', 0))} 次）"
                    " ⟹ 继续插就是无限循环。λ 总数不够是**输入问题**，"
                    "应判 HALT_LAMBDA_BUDGET_INSUFFICIENT 由人工改输入。"
                )
        return out

    def rewindow_feasible(self, view: Dict[str, Any],
                          window_idx: Optional[int]) -> Optional[str]:
        """有界重窗（`IMMUTABLE_REWINDOW`）在这个窗口上可不可行。`None` = 可行。

        🔑🔑 [REWIND-01，2026-09-17] 它是**唯一对任意窗口都够得着**的缩跨度动作：
        不动 λ 表、不需要 tail anchor、不吃 `max_path_insertions`
        ⟹ `feasible()` 里挡住 `SPLIT_TAIL_WINDOW` / `INSERT_LAMBDA` 的四个条件
        （取不到 anchor、dry-run 切不出更细、插完越过 `hi`、插点预算用尽）一个都不碰。
        失败窗口是 window 0（本方法的必然坏窗口）时，那两个动作在构造上永远不可行，
        而这一个可行 —— 这就是接它回来的全部理由。

        前提只有两条：
          · **切得开**：子窗划分走 `vanishing_rescue_ranges`（与执行器**同一份**），
            切不出两个各 ≥2 态的重叠子窗就是不可行（执行器那边是 fail-closed 抛错）；
          · **这个父窗还没建过子系综**：一个父窗只切**一层**。子窗再失败不再递归 ——
            不设深度上限就是一个新的无限循环，而子窗的失败另有出口（见死线 D3）。
        """
        if window_idx is None:
            return "没有目标窗口"
        _i = int(window_idx)
        _rw_view = view.get("immutable_rewindow") or {}
        _done = {int(x) for x in (_rw_view.get("parents_done") or [])}
        if _i in _done:
            return (f"窗口 {_i} 在**当前布局**下已经建过有界重窗的子系综 ⟹ 不再切第二层"
                    "（递归重窗没有停止条件）。子窗自己的失败由子窗那条路由处理。")
        # 🔑🔑 [2026-09-17] **放弃过的父窗可以再试一次，但只有一次。**
        # `parents_done` 现在只认 `SAMPLED`（见 `read()` 里那段：不过滤 status 会让
        # 一条 `ABANDONED_NO_PRODUCT` 把物理窗口永久隐身）。代价是"建了又废"的父窗
        # 重新变得可切 ⟹ **建→废→再建** 就是一个新的无限循环。
        # 这里按**总尝试次数**封顶：同一父窗在当前布局下最多两条台账条目
        # （一次正常 + 一次重试）。这与 `max_path_insertions` 同性质，是终身预算。
        # ⚠️ 用**总数**而不是"放弃数"：status 未来若新增别的非 SAMPLED 值，
        # 这道闸不需要跟着改。
        _abandoned = {int(x) for x in (_rw_view.get("parents_abandoned") or [])}
        if _i in _abandoned and int(_rw_view.get("attempts_by_parent", {}).get(
                str(_i), 0) or 0) >= 2:
            return (f"窗口 {_i} 在当前布局下已经建过 2 次有界重窗且都没产出可用子系综 "
                    "⟹ 不再重试（建→废→再建是无限循环）。这是**输入/预算问题**，"
                    "要继续需要人工看那两次为什么 ABANDONED。")
        ranges = ((view.get("path") or {}).get("window_ranges") or [])
        if not (0 <= _i < len(ranges)):
            return f"读不到窗口 {_i} 的 λ 区间（布局里只有 {len(ranges)} 个窗口）"
        _children = vanishing_rescue_ranges([_i], ranges)
        if len(_children) < 2:
            _a, _b = (int(x) for x in ranges[_i])
            return (f"窗口 {_i} 区间 ({_a}, {_b}) 只有 {_b - _a} 个态，"
                    "拆不出两个各自 ≥2 态的重叠子系综 ⟹ 有界重窗不适用。")
        return None

    def rewindow_children(self, view: Dict[str, Any],
                          window_idx: Optional[int]) -> List[Tuple[int, int]]:
        """这个窗口切出来的子系综区间。空 = 切不开。执行器用同一份实现。"""
        ranges = ((view.get("path") or {}).get("window_ranges") or [])
        if window_idx is None or not (0 <= int(window_idx) < len(ranges)):
            return []
        _c = vanishing_rescue_ranges([int(window_idx)], ranges)
        return _c if len(_c) >= 2 else []

    # -------------------------------------------------------------- 决策

    # 🔑🔑 [2026-09-16] **一个窗口修不动了 ≠ 整个 stage 没动作可做。**
    #
    # 真机（brd4_ligand2/rep1、cyclod_ligand2/rep2）：`earliest` 连吃 4 块帧撞上
    # 补帧配额 ⟹ `NO_FEASIBLE_ACTION` ⟹ 主循环 break，而末窗**一块都没批过**、
    # 预热预算还剩 37 万 / 87 万步，全程躺在 `blocked_by_upstream` 里一次都没被看过。
    # 两个 run 的整腿半程漂移（+8.01 / +11.14 kJ/mol）几乎全部来自那个末窗。
    #
    # 这些出口的语义是「**对这个窗口**没有可行动作」，不是「对这条 stage 没有」——
    # `plan()` 里 [审计 #30] 那段注释早就写明「满额是路由信号，不是终态」，但它只
    # 处理了"同一个动作的多个目标窗口"，没处理"换一个窗口重新决策"。
    #
    # 退役条件苛刻，因为 `earliest` 的排序有物理理由（上游重锚会作废下游的 warmup
    # lineage）：**只有当这个窗口已经没有任何动作可做时**，那条理由才自动失效 ——
    # 不会再有重锚，下游的 lineage 就此固定。所以退役只在终态出口上发生，且要求
    # 别处确实还有补帧预算，否则原样返回原来的终态。
    # [2026-09-17 P0] `SUPPORT_ATTRIBUTION_UNKNOWN` 也可退役：用户规格原话
    # 「若还有其他窗口可做，则绕过该单元继续调度，不能让它停掉整跑」——
    # 退役换窗正是这个机制。别处确实没活干时它仍然是真终态。
    # 🗑️ [2026-09-17，用户拍板] **`_RETIRABLE_EXITS` 已删除。**
    # 可退役性不再是一张要记得维护的出口码名单，而是 `plan()` 的一等语义字段
    # `halt_scope`（见模块级 `EXIT_SPECS`）。`_retirable_window()` 只看 scope。

    def _retirable_window(self, view, plan, retired):
        """这一轮的终态该不该退役、换个窗口重来。返回窗口号或 `None`。

        🔑🔑 [2026-09-17，用户拍板] **只看 `halt_scope`，不认 exit 字符串。**
        先前靠 `_RETIRABLE_EXITS` 这张手写名单，漏一个就是「停掉一个仍有活干的
        stage」——实证漏过两次（`SUPPORT_ATTRIBUTION_UNKNOWN`、
        `D3_REWINDOW_DEPTH_EXHAUSTED`）。现在语义由 `EXIT_SPECS` 一处声明、
        由 `plan()` 强制落到每个终态上，漏了会抛而不是静默停。
        `exit` 只负责解释**为什么**停。
        """
        if plan.get("halt_scope") != "TARGET_LOCAL":
            return None
        e = plan.get("blocking_window")
        if e is None or int(e) in {int(x) for x in retired}:
            return None
        recs = {int(w["window_idx"]): w for w in (view.get("windows") or [])}
        w = recs.get(int(e)) or {}
        # ⚠️ 身份不符 / 终态相 / 证据被布局作废 ⟹ **不退役**：这三种不是"这个窗口
        # 修不动"，而是"这条 stage 的证据不成立"。跳过它去采下游等于拿一条已知
        # 不成立的路径继续烧 GPU。
        if w.get("phase") in ("IDENTITY_MISMATCH", "TERMINAL"):
            return None
        if w.get("stale_layout_evidence_only"):
            return None
        # 别处还有补帧预算才谈得上"换一个窗口"；一个都没有时原样交出终态
        # （`HALT_FRAMES_ADMISSION_CAP` 的定义就是全窗满额 ⟹ 这里必然返回 None）。
        _skipped = {
            int(x if not isinstance(x, dict) else x.get("window_index", -1))
            for x in (view.get("skipped_windows") or [])
        } - {-1}
        _done = {int(e)} | {int(x) for x in retired}
        _replaced = {int(x) for x in
                     ((view.get("immutable_rewindow") or {}).get("parents_done") or [])}
        for idx, rec in recs.items():
            if idx in _done:
                continue
            if self._is_routable_candidate(view, rec, _skipped, _replaced):
                return int(e)
        # 🔑🔑🔑 [S2-G，2026-09-17 真机 `cmet_ligand2`] **「别处还有活干」必须把
        # 从没采过的窗口算进来 —— 它们不在 `view["windows"]` 里。**
        #
        # `recs` 只收**盘上有产物**的窗口；布局里有、一步都没跑过的窗口在
        # `view["missing_windows"]`，**不在这张表里**。于是上面那个循环对一个
        # **整窗未采、块配额分文未动**的窗口**恒答否** ⟹ 不退役 ⟹ 交出终态。
        #
        # 真机崩溃链条：w4 跨全部段已批 4 块（上限 4）⟹ `plan()` 补帧准入拒 ⟹
        # `NO_ACTION` + `NO_FEASIBLE_ACTION`（`halt_scope=TARGET_LOCAL`，本该可退役）
        # ⟹ 这里找不到候选（w5 从没采过、不在 `recs`）⟹ 不退役 ⟹ 终态 ⟹
        # 退出前 ANALYZE 也跳过（`missing_window_5`）⟹ `_assert_stage_result_sane`
        # 抛 `RuntimeError`，**整跑失败，而 w5 一步都没跑过**。
        # 分支 6b（缺窗 → 补采）在结构上就在下游，`plan()` 里的准入闸先一步把整轮
        # 变成终态，根本落不到它。
        #
        # 判据不用问预算：**布局里有、产物里一个都没有**的窗口，按定义就是
        # 「未解决 + 预算分文未动」——它比任何有产物的候选都更明确地"还有活干"。
        # ⚠️ 仍要排除被子系综接管的父窗（修它没有意义）与已退役的。
        for _mi in (view.get("missing_windows") or []):
            _mi = int(_mi)
            if _mi in _done or _mi in _replaced:
                continue
            return int(e)
        return None

    @staticmethod
    def unrouted_window_signals(view: Dict[str, Any]) -> List[Dict[str, Any]]:
        """本轮**不会被路由、而且不会自己成为 `earliest`** 的窗口信号。唯一实现。

        🔑🔑 [2026-09-17] **`_pick()` 对所有分支一视同仁，但分支分两类。**
        `_pick()` 只服务 `earliest`，目标窗口不在候选里就返空。对「挑哪个窗口干活」
        那类分支，押后无害 —— 上游解决后该窗口自己会成为 `earliest`。
        但对**路由态是 `ELIGIBLE` 的窗口**，押后 = **永远轮不到**：
        `earliest` 取的是第一个 `PROBLEM`，`ELIGIBLE` 永远不当选；而全窗合格时
        `earliest is None` ⟹ `_pick()` 恒返空 ⟹ **每一条分支都不触发**。

        实测（2026-09-17，造盘面逐条验的 15 个 `_pick` 调用点）：15 条里
        13 条的触发条件都会把窗口判成 `PROBLEM`（押后、会轮到），只有 2 条不会：
          · 生产步数未达标（自检 ELIGIBLE）⟹ **永远补不到目标步数**，
            而全窗合格时 run 直接判完成；
          · 疑似脱轨（单块边际 N_eff 塌，自检仍 ELIGIBLE）⟹ 早期预警丢掉。

        ⚠️ 本函数**不改路由**。放宽 `_pick()` 是另一件事（它的注释写明「刻意不放宽」：
        11 条分支靠它的返空维持各自未成文的前提）。这里只保证**不静默**：
        这两类信号恒常出现在 plan 里，谁都能看见它们没被处理、以及为什么。
        """
        out = []
        for w in (view.get("windows") or []):
            if Stage2RepairController._window_routing_state(w) != "ELIGIBLE":
                continue          # 非 ELIGIBLE ⟹ 会自己成为 earliest ⟹ 押后不丢
            idx = int(w["window_idx"])
            ps, tgt = w.get("production_steps"), w.get("production_steps_target")
            if ps is not None and tgt is not None and int(ps) < int(tgt):
                out.append({
                    "window_idx": idx, "signal": "PRODUCTION_BELOW_TARGET",
                    "production_steps": int(ps), "target": int(tgt),
                    "why_never_routed":
                        "自检判 ELIGIBLE ⟹ 路由态不是 PROBLEM ⟹ 永远不会成为 "
                        "`earliest`；而全窗合格时 `earliest is None`，`_pick()` 恒返空 "
                        "⟹ 补足生产步数那条分支一次都不触发。",
                })
            # ⚠️ [2026-09-17] 脱轨那一条**已经修了**（`_suspected` 在 `earliest is None`
            # 时也发 —— 探针 `probe_only=True` 零采样、不重锚，是 `_pick` 的合法例外）。
            # 这里只在**仍然发不出**时报：别处有 PROBLEM 窗口占着 `earliest`，
            # 而本窗口是 ELIGIBLE、永远排不上。那才是真的还丢着。
            if (w.get("derailment_status") not in (None, "OK", "NONE")
                    and any(Stage2RepairController._window_routing_state(o) == "PROBLEM"
                            for o in (view.get("windows") or []))):
                out.append({
                    "window_idx": idx, "signal": "DERAILMENT_SUSPECTED",
                    "derailment_status": w.get("derailment_status"),
                    "why_never_routed":
                        "本窗口自检仍 ELIGIBLE ⟹ 永远不会成为 `earliest`，而别处有 "
                        "PROBLEM 窗口占着它 ⟹ `PROBE_CANDIDATE_FK` 排不上。"
                        "（全窗合格时这条已由 `earliest is None` 的例外发出。）",
                })
            # 🔑🔑 [2026-09-17] **另外两条「能与 ELIGIBLE 共存」的信号，
            # 判定为「不该开例外、但必须可见」。**
            #
            # 与上面两条同属 `_pick` 的第四类（触发条件能与 `ELIGIBLE` 共存 ⟹
            # 该窗口永远不会成为 `earliest` ⟹ 分支一次都不触发），但**结论相反**：
            #   · 上面两条开了例外 —— 它们的动作是**零成本**（`probe_only=True`
            #     离线算候选）或**补足已承诺的配额**，不重锚上游、不改布局；
            #   · 下面两条**不开** —— `RECALIBRATE_FK` 是 `probe_only=False`、
            #     **开新段跑满** `n_steps_per_window`（成本表里"最贵的那个"）；
            #     `INSERT_LAMBDA` 改**全局 λ 表**、影响所有窗口。
            #     为一个**自检合格**的窗口花这个钱没有依据。
            # 同一把尺子（零成本/已承诺、不重锚上游）量出两个方向 —— 这比
            # 「两条都开」或「两条都不开」可信。
            # ⟹ 对这两条，"永远不触发"很可能**就是对的行为**；真正的缺陷只剩
            # **静默**：探针给了建议、没人知道它被丢了。所以只报告，不改路由。
            if idx in {int(x) for x in
                       ((view.get("fk_probe") or {}).get(
                           "recalibration_recommended_windows") or [])}:
                out.append({
                    "window_idx": idx, "signal": "FK_RECALIBRATION_RECOMMENDED",
                    "why_never_routed":
                        "f_k 探针建议重标定，但本窗口自检 ELIGIBLE ⟹ 永远不会成为 "
                        "`earliest` ⟹ `RECALIBRATE_FK` 一次都不触发。"
                        "**这不是待修的 bug**：重标定会开新段跑满 "
                        "`n_steps_per_window`，为一个自检合格的窗口花这个钱没有依据。"
                        "报出来只是为了让「探针的建议被丢了」这件事**不静默**。",
                })
            _h = [float(v) for _nm, v in
                  ((view.get("min_n_eff_over_g_history") or {}).get(idx) or [])]
            _st, _d = marginal_gain_stalled(_h)
            if _st:
                out.append({
                    "window_idx": idx, "signal": "MARGINAL_GAIN_STALLED",
                    "series": _h, "diagnosis": _d,
                    "why_never_routed":
                        "主验收量不随采样上升，但本窗口自检 ELIGIBLE ⟹ 永远不会成为 "
                        "`earliest` ⟹ 那条分支（插 λ 缩跨度）一次都不触发。"
                        "**这不是待修的 bug**：`INSERT_LAMBDA` 改全局 λ 表、影响所有"
                        "窗口，而一个自检合格的窗口**没有要补救的失败**。"
                        "报出来只是为了不静默。",
                })
        return out

    @staticmethod
    def local_validation_cap_hits(view: Dict[str, Any]) -> List[Dict[str, Any]]:
        """打满单周期验证批次上限、但**全局预热预算还有钱**的窗口。唯一实现。

        🔑🔑 [2026-09-17] **`LOCAL_VALIDATION_CAP` 是一个「提问」，不是失败。**
        引擎打这个信号时日志里写的是「交上层决定是否延长预算」「交顶层自治控制器
        重判」—— 它明确把一个决定**上交**给控制器。而控制器对它**有**答案
        （分支 3a：`PROVISIONAL_PRODUCTION`，不扩 15 批上限、用未验证的冻结 f_k
        采一个 +250k 诊断块）。

        真机 (2026-09-17)：窗口 4 打满 15/15 批、烧掉 17.5 万步、报"无结论"，
        而 `earliest` 是窗口 0 ⟹ `_pick()` 判 `0 not in [4]` ⟹ **分支 3a 整条不触发**
        ⟹ 那个提问既没被回答、也没被记下，只换来一句 `blocked_by_upstream`。
        下一轮同样、再下一轮同样。**不是拒绝，是连问题都没被看见。**

        本函数把"谁在提问"抽成共享谓词，`plan()` 据此**恒常**把未被回答的提问
        落进 `local_validation_cap_pending` —— 哪怕本轮的动作属于别的窗口。
        """
        try:
            import ibs_engine as _ie
            cap = int(_ie.IBS_LOCAL_MBAR_GATE_MAX_BATCHES)
        except Exception:      # noqa: BLE001 —— 读不到上限就谈不上"打满"
            return []
        out = []
        for w in (view.get("windows") or []):
            if w.get("phase") not in ("WARMUP_LEARN", "WARMUP_VALIDATE"):
                continue
            if (w.get("warmup_steps_left") or 0) <= 0:
                continue
            if (w.get("frozen_validation_batches") or 0) < cap:
                continue
            # 已经在用临时生产取证的窗口不再算"提问"——它的答案已经给过了。
            if str(w.get("bias_status") or "") == "provisional_production":
                continue
            out.append({
                "window_idx": int(w["window_idx"]),
                "frozen_validation_batches": w.get("frozen_validation_batches"),
                "batch_cap": cap,
                "warmup_steps_left": w.get("warmup_steps_left"),
                "bias_status": w.get("bias_status"),
            })
        return out

    @staticmethod
    def _window_routing_state(x: Dict[str, Any]) -> str:
        """一个窗口的**路由态**：`ELIGIBLE` / `PROBLEM` / `UNKNOWN`。唯一实现。

        [2026-09-17，用户拍板 #1] **从 `_decide_once` 的闭包提到类上。**
        `_is_routable_candidate`（退役判定）原来在旁边**又写了一份**「什么算未解决」
        （`self_verdict in {...} or solver_skip or skipped`），比这一份少了整整两态，
        于是真机那个「还在预热 / 没有 self-support 产物」的末窗两边判得不一样：
        `earliest` 排序认得它，退役判定不认 ⟹ `earliest` 无路可走时也不退役
        ⟹ 末窗全程躺在 `blocked_by_upstream` 里一次没被看过。
        同一个不变量两份实现，本仓最贵的那个形状。现在只有这一份。

        ⚠️ **三态不是两态。** 缺证据（`self_verdict is None`）= `UNKNOWN` =「不知道」，
        对它的正确动作是**去产出证据**、不是重标定 —— 但它**同样是「还有活可干」**，
        所以退役判定只排除 `ELIGIBLE`。
        """
        if x.get("phase") in ("IDENTITY_MISMATCH", "TERMINAL"):
            return "PROBLEM"
        if x.get("verdict") == "STATISTICALLY_REJECTED":
            return "PROBLEM"
        # 🔑 **"证据被我们自己作废" ≠ "还不知道"。**
        # 布局一变（插 λ / 拆末窗），旧证据描述的是另一个窗口几何、已被剔除。
        # 那是**确定的未解决**，不是 UNKNOWN —— 必须挡住下游，否则就会
        # 「跳过前面的窗口去跑后面的」，正是要防的那个失败模式。
        if x.get("stale_layout_evidence_only"):
            return "PROBLEM"
        # 🔑🔑 [DECORR-01] **求解器跳窗优先于自检通过。**
        # 求解器对"能否进入求解"有操作权威：它把这个窗口踢出了 MBAR 与协方差链，
        # 那这个窗口就是**未解决**，哪怕自检说 `ANALYSIS_ELIGIBLE`。
        # 实测 win3：自检 56 帧 `sufficient=True`，求解器报 9/7 帧并跳窗 ——
        # 先前这里只看自检，于是它被判 ELIGIBLE、earliest 越过它去处理下游，
        # 而它的帧一次都没补上。必须放在 `ANALYSIS_ELIGIBLE` **之前**。
        if x.get("solver_skip"):
            return "PROBLEM"
        v = x.get("self_verdict")
        if v in ("HARD_INSUFFICIENT", "INSUFFICIENT_DATA"):
            return "PROBLEM"
        if v == "ANALYSIS_ELIGIBLE":
            return "ELIGIBLE"
        # 还在预热/验证、或根本没有自检产物 ⟹ 不知道
        if x.get("phase") in ("WARMUP_LEARN", "WARMUP_VALIDATE"):
            return "PROBLEM"     # 在跑但没跑完，属于"这个窗口还没解决"
        return "UNKNOWN"

    # 一个窗口「还有活可干」的**统一**判据。退役判定与 earliest 排序共用
    # `_window_routing_state`，这里只再加"预算花不花得出去"。
    @staticmethod
    def _is_routable_candidate(view, rec, skipped, replaced) -> bool:
        """这个窗口**既未解决、又确实还有自己的预算**可花。

        🔑🔑 [2026-09-17，用户拍板 #1] **「未解决」原来漏掉了最要命的那一形态。**

        先前判据是 `self_verdict ∈ {HARD_INSUFFICIENT, INSUFFICIENT_DATA}`
        或被求解器跳过。而真机 brd4_ligand2/rep1、cyclod_ligand2/rep2 的末窗是：
          · 还在 `WARMUP_LEARN` / `WARMUP_VALIDATE`；
          · **没有 self-support 产物**（那份是生产跑完才写的）⟹ `self_verdict is None`；
          · `has_convergence=False`；
          · 预热预算还剩 37 万 / 87 万步。
        ⟹ 三个条件一条都不命中 ⟹ **不算"未解决"** ⟹ 不构成退役理由 ⟹
        `earliest` 死在原地、这个末窗全程躺在 `blocked_by_upstream` 里一次没被看过，
        而整腿半程漂移（+8.01 / +11.14 kJ/mol）几乎全部来自它。

        ⚠️ **这不是 round-robin**：上游重锚 / rewindow 确实会作废下游的 warmup
        lineage，所以 `earliest` 的因果顺序**保留**。这里只回答一个问题：
        「earliest 已经无路可走时，别处到底还有没有活干」——答错会让整跑提前收摊。

        ⚠️ **预算必须是这个窗口自己的**，而且 `None` = 未知 ⟹ **不放行**（fail-closed）：
        先前只问生产块 headroom，于是预热期窗口（生产账为空）一律判成"没预算"。
        现在预热账与生产账任一有余量即可 —— 两者是两本账，互不代替。
        """
        idx = int(rec["window_idx"])
        # 这三种不是"还没修"，是"这条证据不成立" ⟹ 绕过去采下游等于拿一条已知
        # 不成立的路径继续烧 GPU（与 `_retirable_window` 开头那三道同一口径）。
        if rec.get("phase") in ("IDENTITY_MISMATCH", "TERMINAL"):
            return False
        if rec.get("stale_layout_evidence_only"):
            return False
        if idx in replaced:
            return False          # 已被子系综接管 ⟹ 修父窗没有意义
        # 「未解决」走**与 earliest 排序同一份**判据（`_window_routing_state`），
        # 不在这里另写。`ELIGIBLE` 之外的两态（`PROBLEM` / `UNKNOWN`）都是还有活可干：
        # 前者有明确问题，后者「不知道」—— 而对「不知道」的正确动作是去产出证据。
        # 被求解器跳过的窗口另加一条（它可能带着 ELIGIBLE 的自检却根本没进 MBAR）。
        if (Stage2RepairController._window_routing_state(rec) == "ELIGIBLE"
                and rec.get("solver_skip") is None and idx not in skipped):
            return False
        # 预算：这个窗口**花得出去**的余量。None（读不到）不算有 —— fail-closed。
        #
        # ⚠️ **"账上有钱"不等于"这笔钱它能花"。** 预热账只对**还在花它**的窗口算数：
        # 仍在 `WARMUP_*` 相，或 f_k 还没 verified（那笔钱正是用来验它的）。
        # 一个已经进 `PRODUCTION` 相、f_k 已 verified、生产配额却烧光的窗口，
        # 预热余额再多也推不动它一步 —— 把它算成"还有活干"会让退役循环空转一圈，
        # 而且会让「别处确实还有预算」这个退役前提变成假的。
        # （这条是实测出来的：`test_nothing_left_anywhere_still_terminates` 的盘面
        #   正是"生产烧光、预热账面还剩 50.5 万步"。）
        _wl = rec.get("warmup_steps_left")
        _warmup_spendable = (
            rec.get("phase") in ("WARMUP_LEARN", "WARMUP_VALIDATE")
            or str(rec.get("f_k_evidence_status") or "") != "verified"
        )
        if _warmup_spendable and _wl is not None and int(_wl) > 0:
            return True
        _hr = frames_growth_headroom(view, idx)
        return _hr is not None and float(_hr) > 1.0

    def decide(self, view: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """**纯函数式判断：下一步做什么。** 不执行、不落盘。

        这一层只做一件事：`_decide_once` 给出「对 `earliest` 没有可行动作」的终态时，
        把那个窗口退役、换下一个还有预算的未解决窗口重新决策（见上面那段）。
        每轮至多退役一个窗口、退役过的不再参选 ⟹ 必然终止。
        """
        view = view or self.read()
        retired: List[int] = []
        while True:
            plan = self._decide_once(view, retired=tuple(retired))
            nxt = self._retirable_window(view, plan, retired)
            if nxt is None:
                if retired:
                    plan["retired_windows"] = list(retired)
                    plan["reason"] = (
                        f"（窗口 {retired} 本轮已无任何可行动作、**已退役出路由顺序**："
                        "它们不会再被重锚，下游窗口的 lineage 就此固定 ⟹ 继续修下一个"
                        "还有预算的窗口。退役**不等于合格**，它们的 verdict 原样保留、"
                        "质量门照判。）"
                    ) + str(plan.get("reason") or "")
                return plan
            retired.append(nxt)

    def _decide_once(
        self, view: Optional[Dict[str, Any]] = None, retired: Sequence[int] = (),
    ) -> Dict[str, Any]:
        """一轮决策。`retired` 里的窗口不参与 `earliest` 排序，其余逐字不变。

        优先级是刻意排的，理由写在每一条里。核心一条：**`UNMEASURED`（没测出来）
        永远不当 `FAIL`（测出来不合格）用** —— 前者加预算，后者换 Epoch；混起来
        就会退化成"再测一次直到碰巧通过"。
        """
        view = view or self.read()
        W = view["windows"]
        feas = self.feasible(view)

        # 🔑🔑 [2026-09-17] **本轮的路由上下文。** `earliest` / `blocked` 一算出来就
        # 存进这里，`plan()` 在调用点没传时从这里取。
        #
        # 先前的约定是「**必须显式传**，不许从闭包里取」，理由是闭包引用会在**全新
        # 运行**上炸 NameError（分支 0 在 `earliest` 被绑定**之前**就 return）。那个
        # 理由只对**裸闭包引用**成立；代价是同一对参数要在 **44 个**调用点各写一遍，
        # 而漏一个是静默的 —— 实测漏了 6 个（分支 2 refuted / 3a 验证不可达的两个出口 /
        # 3a PROVISIONAL_PRODUCTION / 5a-0 PROBE_CANDIDATE_FK / 边际停滞的 INSERT_LAMBDA）。
        #
        # 后果不是"少了一行诊断"：`decide()` 的**退役换窗**（2026-09-16 那条，专治
        # 「末窗全程躺在 `blocked_by_upstream` 里一次没被看过」）靠
        # `plan["earliest_unresolved_window"]` 找退役目标，`None` 就直接 `return None`
        # ⟹ 这六条路径上退役**永不触发**。而它们恰恰都通向 `plan()` 内部的两道改写闸
        # （换 Epoch 付不起 → `RUN_PRODUCTION` → 补帧准入 → `NO_FEASIBLE_ACTION` /
        # `HALT_FRAMES_ADMISSION_CAP`），也就是 `_RETIRABLE_EXITS` 的**全部**两个值。
        #
        # 用一个**字典**而不是裸闭包：分支 0 / 0a 在它被填之前 return，`.get()` 返回
        # `None`，行为与先前逐字相同 —— NameError 那条理由被保住，重复那 44 遍被删掉。
        # 显式传参仍然优先（调用点想覆盖就覆盖）。

        _ctx: Dict[str, Any] = {}

        def plan(action, reason, *, exit_=None, windows=None, missing=None,
                 blocked=None, earliest=None, unit_id=None, extra=None,
                 halt_scope=None, blocking_window=None, blocking_unit=None):
            """`blocked` / `earliest` 不传时从 `_ctx` 取（见它上面那段）。

            `halt_scope`：**停的是谁** —— `TARGET_LOCAL` / `STAGE_GLOBAL` /
            `PATH_INVALID`。终态出口不传时从 `EXIT_SPECS` 取默认；
            `NO_FEASIBLE_ACTION` 没有默认（它同时承载局部与全局两种语义），
            漏传直接抛 —— 见 `EXIT_SPECS` 里那条的 `why`。
            """
            if earliest is None:
                earliest = _ctx.get("earliest")
            blocked = list((blocked if blocked is not None
                            else _ctx.get("blocked")) or [])
            # 🔑🔑 [审计 #20，2026-09-14] **换 Epoch 的预算预检只在真要换 Epoch 时查。**
            #
            # 先前它是一道**无条件**的前置闸：`decide()` 一算出 `earliest` 就对它跑
            # `_epoch_validation_unaffordable`，不分接下来要发什么动作，而且对
            # `warmup_steps_left is None` 是 fail-closed ⟹
            #   · 账本读不到的窗口被**永久钉死在 `RUN_PRODUCTION`**，分支
            #     1c/1d/3/3a/4/5/5a/5b/6 全部不可达（#21/#22/#23 三条都被它盖住）；
            #   · `left` 介于 0 和 floor 之间时，还会在一份**尚未冻结/验证**的 f_k 上
            #     直接开生产。
            # 现在挪进 `plan()`：只有动作**真的是换 Epoch 那一族**才查，付不起就改发
            # `RUN_PRODUCTION`（保持原来的 `HALT_BUDGET` 路由语义与理由口径）。
            # ⚠️ 必须排在下面那道补帧准入**之前** —— 它改出来的 `RUN_PRODUCTION`
            # 同样要过块数上限，否则就成了一条绕过刹车的旁路。
            _EPOCH_ACTIONS = (
                "RECALIBRATE_FK", "RELEARN_FK_EPOCH", "PROBE_REANCHOR_EPOCH")
            if action in _EPOCH_ACTIONS and windows:
                _ep_ws = [int(x) for x in windows]
                _ep_broke = self._epoch_validation_unaffordable(
                    view, _ep_ws, new_epoch=(action == "RELEARN_FK_EPOCH"))
                if _ep_broke:
                    _orig = action
                    # 🔑🔑 [2026-09-15] **降级到补帧，只对「样本量类」失败成立。**
                    #
                    # 这条降级的原意是「换 Epoch 付不起 ⟹ 退而做一件便宜且有用的事」。
                    # 但对**偏斜类**窗口，补帧不是便宜且有用，是**便宜且无用**：
                    # 它拿的正是那份需要被换掉的冻结 f_k，采出来的帧权重剖面一模一样。
                    # 本仓 §5.1 实测：同分布 250k→1M 让 top1% 从 0.545 涨到 0.762、
                    # ESS 比值**反而更差**；重标定一次 rawESS 27.5→503。
                    #
                    # 真机后果（`采样问题_2026-09-15.md` 上半部分那条链的落点）：
                    #     warmup 账干 → 换 Epoch 付不起 → 降级补帧 → 生产预算判不了它
                    #     不可行 → 连补至 max_production_blocks=4 → NO_FEASIBLE_ACTION
                    # 三个 rep **一次都没重标定过**，各自把 1.25M 步砸在一份冻结 f_k 上。
                    #
                    # ⚠️ 归因用与别处**同一份**实现（`support_failure_is_skew`），
                    # 不在这里另写判据。`_is_skew` 是 `decide()` 里的闭包、此刻可能
                    # 还没绑定（`plan()` 的调用点早于它的定义），所以直接调模块级函数。
                    _ep_recs = [w for w in view["windows"]
                                if int(w["window_idx"]) in set(_ep_ws)]
                    # 🔑🔑 [2026-09-17，用户拍板 A] **降级到补帧要「样本量类」的**硬证据**，
                    # 「还没被证伪」不算。**
                    #
                    # 先前这里用的是 `not is_skew`，而 `is_skew` 只区分两态 ⟹
                    # `UNKNOWN`（射程说"也许够得着"）被算进了"样本量类"，于是一个
                    # **乐观上界**就足以让 `plan()` 把分支刚判出的「f_k 不对，换 Epoch」
                    # 静默改写成「补帧」——把一个**还没被证伪**的猜测升级成了**确定归因**。
                    # 而那个上界假定 η 与 g 恒定，两个假设实测都朝不利方向走
                    # （见 `support_failure_attribution` 的长注释）。
                    # 现在只有 `SAMPLE_SIZE`（`solver_eligibility` 族来源，或明确的
                    # `n_decorrelated < min_frames`）才授权降级。
                    _ep_attr = {
                        int(w["window_idx"]): support_failure_attribution(
                            w.get("self_verdict"), w.get("self_verdict_source"),
                            n_decorrelated=w.get("self_n_frames_decorrelated"),
                            min_frames=w.get("self_min_frames"),
                            min_n_eff_over_g=w.get("min_n_eff_over_g"),
                            n_eff_over_g_target=w.get("self_n_eff_over_g_eligible"),
                            frames_headroom=frames_growth_headroom(
                                view, int(w["window_idx"])),
                            n_eff_input_n_frames=w.get(
                                "bottleneck_n_eff_input_n_frames"),
                            bottleneck_g=w.get("bottleneck_g"))
                        for w in _ep_recs
                    }
                    _ep_skew = [i for i, a in _ep_attr.items()
                                if a == SUPPORT_FAILURE_STRUCTURAL]
                    _ep_unknown = [i for i, a in _ep_attr.items()
                                   if a == SUPPORT_FAILURE_UNKNOWN]
                    # 🔑 换不起 Epoch 时**优先走有界重窗**，不是退回补帧。
                    # 它不动 λ 表、不需要 tail anchor、不吃插点预算，而且子系综**各自
                    # 锁一份自己的 f_k** ⟹ 它正面回答了「这份 f_k 不对」这个诊断，
                    # 而补帧恰恰是拿那份要被换掉的 f_k 继续采。
                    _ep_rw = None
                    if _ep_skew or _ep_unknown:
                        _ep_tgt = (_ep_skew or _ep_unknown)[0]
                        if self.rewindow_feasible(view, _ep_tgt) is None:
                            _ep_rw = int(_ep_tgt)
                    if _ep_rw is not None:
                        _why_attr = (f"{_ep_skew} 的自检是**支撑/偏斜类**失败"
                                     if _ep_skew else
                                     f"{_ep_unknown} 的归因是 `UNKNOWN`"
                                     "（射程只说明「还没被证伪」，不是「帧不够」）")
                        action, exit_ = "IMMUTABLE_REWINDOW", None
                        windows = [_ep_rw]
                        reason = (
                            f"原动作 `{_orig}` 要换 Epoch，但窗口 {_ep_ws} **付不起新 "
                            f"Epoch 的最低验证额度**（{_ep_broke}）。而 {_why_attr} ⟹ "
                            "**不降级去补帧**（补帧用的正是那份需要被换掉的冻结 f_k，"
                            "采出来的帧权重剖面一模一样）。改走**有界重窗**："
                            f"在窗口 {_ep_rw} 内建重叠子系综，**各自锁一份自己的 f_k** ⟹ "
                            "它正面回答了「这份 f_k 不对」这个诊断，而且不花验证额度、"
                            "不动 λ 表。原动作与理由：" + reason
                        )
                    elif _ep_skew or _ep_unknown:
                        # 🔑🔑🔑 [2026-09-18] **这里原来配的是 `HALT_BUDGET`，
                        # 而它在 `EXIT_SPECS` 里 `terminal=False`（它本来就是路由信号）。**
                        # `NO_ACTION` + 非终态出口 ⟹ `plan["terminal"]` 为 False ⟹
                        # 主循环不 break ⟹ 分发段没有 `NO_ACTION` 的执行器 ⟹ 掉进
                        # 「动作没有执行器，退出循环交人工」⟹
                        # `outcome = {status: HALTED_NO_EXECUTOR, exit: None}`。
                        # 而下游每一道门（`abfe_pipeline` 的放行判、收尾 ANALYZE 的
                        # 判据、2026-09-18 那六条验收）**全部按 `exit` 分类** ——
                        # 这一跑连出口码都没有，事后分不清它是怎么停的。
                        # `tests/test_stage2_controller_vocabulary.py::
                        # test_no_action_is_only_ever_paired_with_a_terminal_exit`
                        # 钉的正是这条不变量，但它的 docstring 自己写明了盲区：
                        # 「`plan()` 内部把 action 改写成 NO_ACTION 的那几道闸在这里
                        # 一条都看不见」。五处内部改写里只有这一处配错，就是它漏的。
                        # 语义本来就是终态：这个窗口换不起 Epoch、加帧治不了、
                        # 有界重窗也不可行 —— 和下面那段理由说的「如实停下」一致。
                        # `TARGET_LOCAL`：卡死的是**这个窗口**，别处照常可修
                        # （`_retirable_window` 据此退役换窗）。
                        action, exit_ = "NO_ACTION", "NO_FEASIBLE_ACTION"
                        halt_scope = "TARGET_LOCAL"
                        reason = (
                            f"原动作 `{_orig}` 要换 Epoch，但窗口 {_ep_ws} **付不起新 "
                            f"Epoch 的最低验证额度**（{_ep_broke}）。"
                            + (f"而其中 {_ep_skew} 的自检是**支撑/偏斜类**失败 ⟹ "
                               if _ep_skew else
                               f"而 {_ep_unknown} 的归因是 `UNKNOWN` —— 射程只说明"
                               "「还没被证伪」（它假定 η 与 g 恒定，两者实测都朝不利"
                               "方向走），**不足以断定「帧不够」** ⟹ ")
                            + "**不降级去补帧**：补帧用的正是那份需要被换掉的冻结 f_k，"
                            "采出来的帧权重剖面一模一样（§5.1 实测同分布 250k→1M 让 "
                            "top1% 0.545→0.762、ESS 比值反而更差）。"
                            "便宜且无用的动作不该顶替一个做不起的对症动作；"
                            f"有界重窗也不可行（{self.rewindow_feasible(view, (_ep_skew or _ep_unknown)[0])}）"
                            " ⟹ 如实停下。**要继续必须显式提高该窗口的预热/验证预算**，"
                            "或改走布局动作（缩跨度）。原动作与理由：" + reason
                        )
                    else:
                        action, exit_ = "RUN_PRODUCTION", "HALT_BUDGET"
                        reason = (
                            f"原动作 `{_orig}` 要换 Epoch，但窗口 {_ep_ws} **付不起新 "
                            f"Epoch 的最低验证额度**（{_ep_broke}）⟹ **不启动**。"
                            "启动之后才发现没预算验，正是 win2 连死三次的形状。"
                            "低支撑/没预算永远是「尚不可测」，不是 FAIL ⟹ 退而补生产帧"
                            "（用的是已冻结的 f_k，一步验证预算都不花，两本账互不代替）。"
                            "⚠️ 这条降级只对**样本量类**失败成立，而且要的是**硬证据**"
                            "（`solver_eligibility` 族来源，或明确的 "
                            "`n_decorrelated < min_frames`）—— 射程判据说的"
                            "「也许够得着」是 `UNKNOWN`，**不授权**这条降级"
                            "（2026-09-17 收紧；先前 `not is_skew` 把 UNKNOWN 也算了进来，"
                            "等于拿一个乐观上界覆盖掉分支刚做出的对症诊断）。"
                            "偏斜类与 UNKNOWN 走上面那两条。"
                            "原动作与理由：" + reason
                        )
            # 🔑🔑🔑 [S2-H + S2-F，2026-09-17] **会开新段的动作：次数 + 事后效果两道闸。**
            # 挂在 `plan()` 里而不是逐个分支 —— 与补帧准入、换 Epoch 预检同一个理由：
            # 逐个挂必然漏（实测漏过 15/16 个出口）。
            # 这一族动作**每次都改盘面**，所以主循环的停滞保护（同一动作 + 盘面未变）
            # 对它结构性免疫；而插 λ 有 `max_path_insertions`、补帧有块数硬上限，
            # **只有它什么预算都没有** ⟹ 真机连发 4 次、丢掉 750k 步、验收量 9.645→0.837。
            if action in _SEGMENT_OPENING_ACTIONS and windows:
                _sa_bad = {}
                for _w in windows:
                    _why_sa = structural_action_refused(view, action, int(_w))
                    if _why_sa:
                        _sa_bad[int(_w)] = _why_sa
                if _sa_bad and len(_sa_bad) == len(windows):
                    _orig_sa = action
                    action, exit_ = "NO_ACTION", "NO_FEASIBLE_ACTION"
                    halt_scope = "TARGET_LOCAL"      # 局部无路，别处照常可修
                    reason = (
                        f"原动作 `{_orig_sa}` 对 {sorted(_sa_bad)} 不再发："
                        + "；".join(f"w{k}: {v}" for k, v in sorted(_sa_bad.items()))
                        + " 原动作与理由：" + reason
                    )
            # 🔑🔑 [裁决 2] **只有生产预算「已知且确实耗尽」才终止。**
            # 花 GPU 的动作（除 ANALYZE / DONE / NO_ACTION 之外全都花）在这里统一
            # 被拦一道 —— 放在 `plan()` 里而不是逐个分支，是因为逐个分支必然漏。
            # ⚠️ 上限 unknown 时**不拦**：未知不是零，"预算耗尽"这个结论不成立。
            # ⚠️ 预热/验证预算（`warmup_steps_left`）**不参与**这道判断，它是另一本账。
            # 🔑🔑 [2026-09-14] **补帧准入：放在 `plan()` 里，不逐个分支挂。**
            # `decide()` 有 **16 个**发 `RUN_PRODUCTION` 的出口；逐个挂门必然漏 ——
            # 实测就漏了 15 个（其中两处的改动还因为同一个脚本里后面的 assert 抛错
            # 而整份写入被中止、悄悄丢掉）。这里一次挂住全部。
            # 🔑🔑🔑 [S2-N，2026-09-18 真机] **写侧记账的动作，读侧都要过这道闸。**
            # 先前这里是 `if action == "RUN_PRODUCTION"`，而 `_BLOCK_CHARGING_ACTIONS`
            # 有 4 个 ⟹ 另外三个扣配额却从不被拦（真机两例：`cmet_ligand2/rep3` w4
            # 吃到 5 块、`cyclod_ligand3/rep3` w1 同形）。同一张表管两侧，别再分叉。
            # ⚠️ 非 `RUN_PRODUCTION` 的三个只过**第一道**（块数硬上限），理由见
            # `_frames_admission` 的 `hard_cap_only` 那段：拿"加帧没用"去否决重标定，
            # 正好掐掉唯一对症的那条路。
            if action in self._BLOCK_CHARGING_ACTIONS:
                _is_topup = (action == "RUN_PRODUCTION")
                _what = "补帧" if _is_topup else f"动作 `{action}`（它要花一个生产块）"
                # 🔑🔑🔑 [S2-B，2026-09-17 真机 `cmet_ligand1/rep2`] **第一遍覆盖优先于重复补帧。**
                #
                # 盘面：布局 5 窗、产物 3 个 ⟹ `missing=[3,4]` **从没采过**；
                # 而 w0 已吃 3/4 块仍在继续，w2 还在 `WARMUP_VALIDATE` 一起被挡。
                # 全部预算喂给了一个窗口，两个窗口一步都没跑过。
                #
                # 挡住它们的是「按因果顺序处理 earliest」，而那条的**物理理由**是
                # 「上游**重锚**会作废下游的 warmup lineage」。但 `RUN_PRODUCTION`
                # 是**同一份冻结 f_k、接着原段** —— 不重锚、不改布局、**作废不了任何
                # 下游**。拿它挡住两个从没采过的窗口没有物理依据。
                #
                # 判据收敛成一句：**从没采过的窗口应该先拿到第一块，再让别的窗口拿
                # 第 N 块（N ≥ 2）**。
                #   · 只对**重复**补帧让路（目标窗口已经有过块）——第一遍本身不让；
                #   · 只让给**从没采过**的窗口（`missing_windows`：布局里有、产物里没有），
                #     它按定义预算分文未动、也没有任何 lineage 可被作废；
                #   · **不碰**重锚/换 Epoch/布局动作的因果顺序 —— 那些确实会作废下游。
                # ⚠️ 代价是"先采的下游窗口可能被之后的重锚作废"。那是**浪费**，
                # 而现状是**整条路径缺窗**（缺窗口的和不是 ΔG，是另一个量）——
                # 后者是硬约束，前者不是。
                # ⚠️ 这条只对**补帧**成立：它的论证是「同一份冻结 f_k、接着原段 ⟹
                # 不重锚、作废不了任何下游」。重标定/换 Epoch **会**重锚，不适用。
                if (_is_topup and not unit_id and windows
                        and view.get("missing_windows")):
                    _tot_b = view.get("production_blocks_total_by_window") or {}
                    _repeat = [int(w) for w in windows
                               if len(_tot_b.get(int(w)) or []) >= 1]
                    _never = sorted(
                        int(x) for x in view["missing_windows"]
                        if int(x) not in {int(y) for y in windows})
                    if _repeat and _never and len(_repeat) == len(windows):
                        _first = _never[0]
                        reason = (
                            f"（**第一遍覆盖优先**：窗口 {_repeat} 已经批过 "
                            f"{[len(_tot_b.get(int(w)) or []) for w in _repeat]} 块，"
                            f"而窗口 {_never} **一步都没跑过**。补帧是同一份冻结 f_k、"
                            "接着原段 —— 不重锚、不改布局、**作废不了任何下游**，"
                            "所以用它挡住从没采过的窗口没有物理依据；而缺窗口的和"
                            "**不是 ΔG**。⟹ 本轮改跑 "
                            f"win{_first}。原目标与理由：）" + reason
                        )
                        windows = [_first]
                # 🔑🔑 [CTL-13，2026-09-14] **准入要逐个目标窗口判，不能只看第一个。**
                # 先前是 `_tgt_w = int(windows[0])` —— 只拿第一个窗口的块账去问准入。
                # 而分支 9c 在「端点 σ 归因不到具体窗口」时发的是
                # `windows=sorted(全部窗口)`（那是有意的：归因不到就全窗各补一块）。
                # ⟹ win0 块数一满就把**整批窗口**的补帧一起毙掉；反过来 win0 还有
                # 额度时，其余已超额的窗口也照批。两个方向都错。
                # 现在：逐窗判，**只保留还能批的那些**；一个都不剩才是 NO_ACTION。
                _tgt_ws = []
                if unit_id:
                    # 🔑 [契约 B] 子窗查**子窗自己那本账** —— 先前拿父窗的账去问，
                    # 而父窗的步数已经冻结 ⟹ 去重后恒 1 行 ⟹ 准入恒放行。
                    # 🔑🔑 [2026-09-17] **`or 0` 会把「父窗未知」记到窗口 0 的块账上。**
                    # 「未知不是零」——这里尤其致命：块账决定补帧准入，把一个
                    # 未知父窗的子窗算进 window 0，会**替 window 0 消耗配额**
                    # （它是本方法的必然坏窗口，配额最金贵），同时让真正的父窗
                    # 一块都不记、准入恒放行。实测目前 `parent_window` 一直有值
                    # （执行器两处都写 `int(window_idx)`），但这不是不修的理由：
                    # 它是**静默**错，出事时没有任何迹象。
                    _u_pw = next(
                        (u.get("parent_window")
                         for u in (view.get("sampling_units") or [])
                         if str(u.get("unit_id")) == str(unit_id)), None)
                    if _u_pw is not None:
                        _tgt_ws = [int(_u_pw)]
                elif windows:
                    _tgt_ws = [int(x) for x in windows]
                _adm = {w: _frames_admission(view, _blocks, _blocks_total, _blocks_total_u, _blocks_u, _max_blocks, w, unit_id=unit_id if unit_id else None,
                                             hard_cap_only=not _is_topup)
                        for w in _tgt_ws}
                _ok_ws = [w for w in _tgt_ws if _adm[w][0]]
                if _tgt_ws and _ok_ws and len(_ok_ws) < len(_tgt_ws) and not unit_id:
                    # 部分窗口还能批 ⟹ **只补那几个**，别因为一个窗口满额就整批作废。
                    _dropped = [w for w in _tgt_ws if not _adm[w][0]]
                    windows = list(_ok_ws)
                    reason = (
                        f"（窗口 {_dropped} 已达块数准入上限、本轮剔除（{_what}）："
                        + "；".join(f"w{w}: {_adm[w][1]}" for w in _dropped)
                        + f"）仍然补 {_ok_ws}。原理由：" + reason
                    )
                if _tgt_ws and not _ok_ws:
                    _adm_why = "；".join(f"w{w}: {_adm[w][1]}" for w in _tgt_ws)
                    # 🔑🔑 [审计 #30，2026-09-14] **满额是路由信号，不是终态。**
                    # 这道闸在 `plan()` 内部，而那时分支**已经锁定了单个窗口** ⟹
                    # 先前发 `NO_FEASIBLE_ACTION`（终态）等于「一个窗口满额就终止
                    # 整跑」，别的窗口和全部布局动作一个都没试过。
                    # 只有**所有**窗口都满额（按跨段累计那本判）才谈得上"没动作可做"。
                    _tot_all = view.get("production_blocks_total_by_window") or {}
                    _all_w = [int(w["window_idx"]) for w in view.get("windows") or []]
                    _all_full = bool(_all_w) and all(
                        len(_tot_all.get(int(i)) or []) >= int(
                            view.get("max_production_blocks_per_window") or 4)
                        for i in _all_w
                    )
                    # 两个都是终态（见 `TERMINAL_EXITS` 里那段注释）：这道闸只在
                    # 目标窗口**一个都批不了**时走到这里，而部分满额早就在上面被
                    # 「剔掉满额的、只补剩下的」处理掉了。`HALT_FRAMES_ADMISSION_CAP`
                    # 只是比笼统的 `NO_FEASIBLE_ACTION` 说得更具体：补帧配额用尽。
                    action = "NO_ACTION"
                    exit_ = ("HALT_FRAMES_ADMISSION_CAP" if _all_full
                             else "NO_FEASIBLE_ACTION")
                    # [2026-09-17] 全窗满额 ⟹ stage 级（`EXIT_SPECS` 给默认）；
                    # 否则只是**本轮点名的这些窗口**批不过 ⟹ 局部，允许换窗。
                    if not _all_full:
                        halt_scope = "TARGET_LOCAL"
                    reason = (
                        f"原动作是对 {_tgt_ws}{f'（子窗 {unit_id}）' if unit_id else ''}"
                        f" 做{_what}，但**不再自动发**：{_adm_why}"
                        "「又产生了帧」不等于「获得了有用证据」。"
                        + ("**所有**窗口的补帧块数配额都已用尽 ⟹ 没有可做的补帧动作了"
                           "（`HALT_FRAMES_ADMISSION_CAP`）。"
                           if _all_full else
                           "本轮点名的窗口一个都批不过（部分满额时上面已经剔掉满额的、"
                           "只补剩下的，走不到这里）。")
                        + "要对这个窗口继续花生产块（补帧或重标定/换 Epoch）"
                        "必须显式提高它的块数上限，"
                        "或改走布局动作。原动作与理由：" + reason
                    )

            # 🔑🔑 [CTL-12，2026-09-14] **no-op 台账是执行器对**每一个**动作记的，
            # 判读却只有三个动作在做 —— 差的那几个正是真机上把循环转死的那几个。**
            #
            # `_record_noop_action` 是通用的（执行前后各取一次盘面指纹，一样就记账），
            # 而 `decide()` 里 `_is_noop()` 只在 `CONTINUE_WARMUP` / `RECALIBRATE_FK` /
            # `PROBE_REANCHOR_EPOCH` 三个动作上被查过。`SPLIT_TAIL_WINDOW` /
            # `INSERT_LAMBDA` / `RUN_PRODUCTION` / `IMMUTABLE_REWINDOW` 一概不查。
            # 真机 cyclod_ligand1/rep2 与 cyclod_ligand2/rep1 都死在同一条路上：
            # `SPLIT_TAIL_WINDOW` 连发 4 次，执行器每次都打了「记为当前盘面上的
            # no-op」，控制器每次都看不见，最后由停滞保护给出 NO_FEASIBLE_ACTION ——
            # 白烧三轮，而且退出理由说成「推不动」而不是「这个动作在这个盘面上
            # 已经被证明什么也不做」。
            #
            # 这里统一挂一道：**已记账为 no-op 的动作不再发**，如实给出口。
            # ⚠️ 指纹带路径版本与生产步数 ⟹ 补过帧 / 换过段 / 插过 λ 之后记录自动
            # 失效、动作重新可选。所以这不是"永久封杀一个动作"。
            # ⚠️ 上面那三个已有判读点**更强**（它们会改发一个对症动作而不是停下），
            # 它们在调用 `plan()` 之前就换好了动作，不会被这道闸碰到。
            # ⚠️ `ANALYZE` 不挂：它是产出证据的动作，判它 no-op 的风险大于收益。
            _NOOP_GUARDED = (
                "RUN_PRODUCTION", "INSERT_LAMBDA", "SPLIT_TAIL_WINDOW",
                "IMMUTABLE_REWINDOW", "RELEARN_FK_EPOCH", "PROBE_CANDIDATE_FK",
                "PROVISIONAL_PRODUCTION",
            )
            # [审计 #8，契约 A] `and not unit_id` 删掉 —— 子窗动作先前**显式被排除**
            # 在这道闸之外，于是子窗那边一条 no-op 刹车都没有。
            if action in _NOOP_GUARDED and (windows or unit_id):
                if unit_id:
                    _dead = ([str(unit_id)]
                             if _is_noop(view, _by_idx_for_noop, _by_uid_for_noop, _noop_led, action, None, unit_id=unit_id) else [])
                    _n_tgt_noop = 1
                else:
                    _dead = [int(w) for w in windows if _is_noop(view, _by_idx_for_noop, _by_uid_for_noop, _noop_led, action, int(w))]
                    _n_tgt_noop = len(windows)
                if _dead and len(_dead) == _n_tgt_noop:
                    _dead_action = action
                    action, exit_ = "NO_ACTION", "NO_FEASIBLE_ACTION"
                    # 这个动作对**这些目标**是 no-op ⟹ 局部无路，别处照常可修。
                    halt_scope = "TARGET_LOCAL"
                    reason = (
                        f"原动作 `{_dead_action}` "
                        f"对 {_dead} 在**当前盘面**上已被执行器记账为 no-op"
                        "（跑过一次、盘面逐项未变）⟹ 再发一次仍然什么也不做。"
                        "盘面一变（补帧 / 换段 / 插 λ）这条记录自动失效、动作重新可选。"
                        "原动作与理由：" + reason
                    )

            # 🔑🔑 [2026-09-15 审计②③] **终态不许和自己的证据打架。**
            #
            # 两条 DONE 分支各自判一遍前提，而 `evidence_status` 在**下面**才算：
            #   · 0a（4331）查了 `_layout_trustworthy` / `stale` / `coverage`，
            #     但**没查逐窗 `STATISTICALLY_REJECTED` / `phase == TERMINAL`**
            #     ⟹ 一份 f_k 已被统计驳回的窗口仍可被宣布 DONE；
            #   · 分支 7（5603）**一道都没查** —— 而 0a 的注释写着「两条都不满足
            #     就往下走路由，由分支 7 兜底（那条路径上保护仍然成立）」，
            #     **那句话是假的**。真机形状：path v2 + stage 缓存不盖 path_version
            #     ⟹ 0a 跳过 ⟹ 分支 7 直接 DONE，发布一个描述**另一条布局**的 ΔG。
            #
            # 逐条去补每个分支 = 又一次"同一不变量 N 份实现"。改在**唯一的出口**：
            # `DONE` 的证据必须是 `CONVERGED`；不是就不是终态，如实降级并说清楚
            # 是哪一维在反对。`allow_untrusted` 只改 `trust_level`，不救这里 ——
            # 它的语义是"门没过但我放行"，不是"把反面证据改写成正面"。
            _ev_now = self._evidence_status(view, action, exit_)
            if action == "DONE" and _ev_now != "CONVERGED":
                _bad_w = sorted(
                    int(w["window_idx"]) for w in (view.get("windows") or [])
                    if w.get("verdict") == "STATISTICALLY_REJECTED"
                    or w.get("phase") == "TERMINAL"
                )
                action, exit_ = "NO_ACTION", "HALT_EVIDENCE_CONTRADICTS_DONE"
                reason = (
                    f"原动作 `DONE` 的证据是 `{_ev_now}`（不是 `CONVERGED`）⟹ "
                    "**这不是完成**。"
                    + (f"逐窗反面证据：窗口 {_bad_w} 的 f_k 被统计驳回或已进终态。"
                       if _bad_w else
                       "布局版本未核实 / 证据被布局变更作废 / 覆盖有缺口"
                       "（见 `missing_evidence`）。")
                    + "原理由：" + reason
                )

            _pb = view.get("production_budget") or {}
            # 🔑🔑 [审计 #34，2026-09-14] **只有真正产生生产帧的动作才走生产账。**
            # 先前 `CONTINUE_WARMUP` / `RELEARN_FK_EPOCH` / `INSERT_LAMBDA` /
            # `SPLIT_TAIL_WINDOW` 也被按一整块**生产**预算收费，付不起就发
            # `GLOBAL_BUDGET_EXHAUSTED`（终态）—— 正是这套双账要防的
            # 「拿 A 账本余额终止只花 B 账本的动作」，方向还反了。
            #   · `CONTINUE_WARMUP` / `RELEARN_FK_EPOCH` 花的是**预热账**
            #     （`warmup_steps_left`），由 `_epoch_validation_unaffordable` 那条路管；
            #   · `INSERT_LAMBDA` / `SPLIT_TAIL_WINDOW` 是**布局动作**，本身不直接花
            #     采样预算（它们引发的重采由下一轮的 `RUN_PRODUCTION` 计费）。
            _PRODUCTION_CHARGED = (
                "RUN_PRODUCTION", "PROBE_REANCHOR_EPOCH",
                "RECALIBRATE_FK", "IMMUTABLE_REWINDOW",
                # [2026-09-15] 临时生产花的是**生产账**（它采生产帧），
                # 不是预热账 —— 与上面 `_BLOCK_CHARGING_ACTIONS` 同一理由。
                "PROVISIONAL_PRODUCTION",
            )
            if action in _PRODUCTION_CHARGED and _pb.get("cap_known"):
                # 🔑 [2026-09-14] **按下一动作的完整成本准入，不是只问"已经耗尽"。**
                # 先前剩 100k 也照发 250k 块 —— 那等于允许超支一整块。
                _blk = int(_pb.get("production_block_steps") or 0)
                _n_tgt = max(1, len(windows or [1]))
                # 🔑🔑 [审计 #32] **`_NON_SAMPLING` 原来选错了动作对。**
                # 实际派发（`abfe_pipeline`）：
                #   · `RECALIBRATE_FK`      `probe_only=False`，**开新段跑满**
                #     `n_steps_per_window` ⟹ 最贵的那个，先前却被判零成本；
                #   · `PROBE_CANDIDATE_FK`  `probe_only=True`，离线算候选、零采样
                #     ⟹ 它才是真正的非变异探针（连这道闸都进不到，见上面的动作表）；
                #   · `PROBE_REANCHOR_EPOCH` 明确「候选 f_k + 独立 burn-in +
                #     **一个 +250k 块**」⟹ 按一块收。
                if action == "PROBE_REANCHOR_EPOCH":
                    _cost = _blk
                elif action == "IMMUTABLE_REWINDOW":
                    # 🔑 [审计 #44] 建新系综：每个子窗一个首块。子窗数从 model B 的
                    # 实际拆法算 —— 先前硬编码 2，而执行器按 `len(children) × base_unit`
                    # 收费，两边一旦不等就是「预留了 2 块、实际花 3 块」。
                    # 子窗数的权威是 `vanishing_rescue_ranges`（执行器
                    # `_build_vanishing_rescue_ranges` 转发到同一份）。
                    # [REWIND-01，2026-09-17] 这里原来自己写 `2 if (b-a) > 2 else 1`
                    # 复算了一遍 —— 那份实现搬到模块级之后直接调，两边不再各写一遍。
                    _n_children = None
                    try:
                        _n_children = len(self.rewindow_children(
                            view, int(windows[0]) if windows else None)) or None
                    except Exception:  # noqa: BLE001
                        _n_children = None
                    if not _n_children:
                        _n_children = 2      # 拿不到就保守取 model B 的最小拆法
                    _cost = int(_n_children) * int(
                        _pb.get("new_ensemble_reserve_steps") or _blk)
                else:
                    _cost = _blk * _n_tgt
                _left = _pb.get("stage_remaining_steps")
                # ⚠️ 余量**未知**（上限未知，或用量账不完整）⟹ 拦不拦都没有依据。
                # 先前 `or 0` 把 None 当成 0 ⟹ 一律拦死，等于"未知 = 耗尽"。
                if _left is not None and int(_left) < _cost:
                    action, exit_ = "NO_ACTION", "GLOBAL_BUDGET_EXHAUSTED"
                    reason = (
                        f"原动作需要 {_cost} 步生产预算，而 stage 只剩 {int(_left)} 步"
                        f"（已用 {_pb.get('stage_used_steps')} / 上限 "
                        f"{_pb.get('stage_cap_steps')}，范围 {_pb.get('cap_scope')}，"
                        f"来源 {_pb.get('cap_source')}）⟹ **不发一个付不起的动作**。"
                        "这与预热/验证预算是两本账，互不代替。原动作与理由：" + reason
                    )
            # 🔑🔑 [2026-09-17，用户拍板] **停的是谁，必须说清楚。**
            # 终态出口没有 scope 就抛 —— 静默按某个默认处理的两个方向都错：
            # 按局部处理会绕过一条已知不成立的路径继续烧 GPU；按全局处理会停掉
            # 一个仍有活干的 stage（`D3_REWINDOW_DEPTH_EXHAUSTED` 就是这么漏的）。
            _spec = EXIT_SPECS.get(str(exit_)) if exit_ else None
            if halt_scope is None and _spec is not None:
                halt_scope = _spec.default_scope
            if halt_scope is not None and halt_scope not in HALT_SCOPES:
                raise ValueError(
                    f"未知 halt_scope {halt_scope!r}；只接受 {HALT_SCOPES}")
            if exit_ and _spec is None:
                raise ValueError(
                    f"出口 {exit_!r} 不在唯一注册表 `EXIT_SPECS` 里 ⟹ 它的 terminal /"
                    " scope 语义没有任何地方声明过。先去那张表里加一条。")
            if _spec is not None and _spec.terminal and halt_scope is None:
                raise ValueError(
                    f"终态出口 {exit_!r} 缺 `halt_scope`：它在 `EXIT_SPECS` 里没有"
                    f"默认 scope（{_spec.why}），发出点必须显式传 "
                    "`halt_scope=\"TARGET_LOCAL\"` 或 `\"STAGE_GLOBAL\"` 或 "
                    "`\"PATH_INVALID\"`。漏传不再静默停整跑。")
            # 🔑🔑🔑 [2026-09-18] **`NO_ACTION` 必须配终态出口 —— 在唯一出口处强制。**
            #
            # 这条不变量本来只有 `tests/test_stage2_controller_vocabulary.py` 的 AST
            # 扫描在钉，而它只看 `plan(...)` **调用点上字面写出来的** `action`/`exit_`；
            # `plan()` **内部**把 action 改写成 `NO_ACTION` 的五道闸（换 Epoch 预检 /
            # 会开新段的动作 / 补帧准入 / no-op 台账 / 生产预算）在它眼里一条都看不见
            # —— 那个测试的 docstring 自己写着这个盲区。实测就漏掉了换 Epoch 预检
            # 那一处（配了非终态的 `HALT_BUDGET`，真机表现是 `outcome.exit=None` +
            # `HALTED_NO_EXECUTOR`）。
            #
            # 检查放这里而不是再补一条 AST 规则：这是**运行期的唯一出口**，五道闸
            # 改完 action 都要经过它，往后再加第六道也躲不掉。主循环按 `terminal`
            # **先于**执行器分发 break，这正是执行器不需要认识 `NO_ACTION` 的前提；
            # 前提破了就该响，不该让执行器收到一个不认识的词再以 `exit=None` 收场。
            if action == "NO_ACTION" and not (_spec is not None and _spec.terminal):
                raise ValueError(
                    f"`NO_ACTION` 配了非终态出口 {exit_!r} ⟹ 主循环不会 break，"
                    "执行器会收到一个它不认识的动作，整跑以 "
                    "`status=HALTED_NO_EXECUTOR` / `exit=None` 收场（下游每一道门都按 "
                    "`exit` 分类，拿不到出口码就分不清这一跑是怎么停的）。"
                    "`NO_ACTION` 的语义是「没有动作」，它**必须**终止 —— 请改配 "
                    "`EXIT_SPECS` 里 terminal=True 的出口，可用："
                    f"{sorted(k for k, v in EXIT_SPECS.items() if v.terminal)}。")
            return {
                "protocol_version": STAGE2_CONTROLLER_PROTOCOL_VERSION,
                "action": action,
                "exit": exit_,
                # 停的是谁（`None` = 不是终态，没有"停"这回事）
                "halt_scope": halt_scope,
                "blocking_window": (
                    int(blocking_window) if blocking_window is not None
                    else (int(earliest) if earliest is not None else None)),
                "blocking_unit": (
                    str(blocking_unit) if blocking_unit is not None
                    else (str(unit_id) if unit_id else None)),
                "reason": reason,
                "windows": windows or [],
                # 🔑 [2026-09-14] **采样单元 ID。** 物理窗口不是唯一可调度的采样
                # 单元：immutable rewindow 的子窗有自己的目录、自己的 checkpoint、
                # 自己的冻结 f_k。给子窗补帧必须点名 `unit_id`，执行器据此续跑
                # **那一个**子窗的 checkpoint，而不是回头动父窗。
                # 普通窗口的动作不带它（None）。
                "unit_id": unit_id,
                "missing_evidence": missing or [],
                "feasible_structural_actions": [k for k, v in feas.items() if v is None],
                "infeasible_structural_actions": {k: v for k, v in feas.items() if v},
                # 停止标准三维：执行状态 / 证据状态 / 可信级别，三件不同的事。
                # `allow_untrusted_stage_results` **只能改 trust_level**，
                # 不得把 evidence_status 改写成 CONVERGED。
                "execution_status": (
                    # 只有**真终止**才是 HALTED；路由信号仍然是 IN_PROGRESS ——
                    # 控制器换个动作继续跑，流水线没有停。
                    "HALTED" if (exit_ in self.TERMINAL_EXITS and action != "DONE")
                    else ("COMPLETE" if action == "DONE" else "IN_PROGRESS")
                ),
                "evidence_status": self._evidence_status(view, action, exit_),
                "trust_level": (
                    "OVERRIDDEN_UNTRUSTED" if self.allow_untrusted else "STATISTICAL_ONLY"
                ),
                # **终止 vs 路由**：路由信号必须被主循环消费，不得炸出流水线。
                # 被上游阻塞的窗口：不是"排队等"，而是**它们的证据前提可能已作废**
                # （上游重锚会换掉共享态起点 / 末帧 lineage）。
                "blocked_by_upstream": blocked if windows else [],
                "blocked_by": (
                    int(earliest) if (blocked and windows and earliest is not None)
                    else None
                ),
                # [2026-09-16] 本轮路由锁定的那个窗口。`blocked_by` 不能代替它：
                # 那个键在"没有下游被挡住"时是 None，而退役判定需要知道**这一轮
                # 到底在修谁**（见 `_retirable_window`）。
                "earliest_unresolved_window": (
                    int(earliest) if earliest is not None else None
                ),
                "terminal": bool(exit_ in self.TERMINAL_EXITS),
                "routing": bool(exit_ is not None and exit_ not in self.TERMINAL_EXITS),
                # ⚠️ **只是提示，不是动作。** 影子模式报"按 warmup 剖面谁最弱"，
                # 好让"提前配预算"这件事先被看见；**不给阈值、不给倍数**，
                # 剖面→预算的映射是未定项（PLAN P3）。
                # 🔑🔑 [S2-A，2026-09-17] **哪些刹车这一轮根本没带电，恒常报出来。**
                # 「判据没触发」和「判据没有输入」在日志里长得一样，而后者意味着
                # 那道刹车从头到尾没起过作用。实测三个 run 的补帧边际增益刹车全程
                # 如此（判据量逐块 `[None, None, None]`）。**只报告、不改行为。**
                "inoperative_criteria": [
                    {"criterion": "frames_marginal_gain_brake",
                     "window_idx": int(_iw),
                     "diagnosis": _idiag,
                     "why": "补帧的边际增益刹车在这个窗口上没有输入 ⟹ 本轮没有带电。"
                            "见 docs/TODO_P2.md `S2-A`。"}
                    for _iw, _irows in sorted(
                        (view.get("production_blocks_total_by_window") or {}).items())
                    for _idiag in [marginal_gain_stalled(
                        [r.get("solver_n_decorrelated") for r in (_irows or [])])[1]]
                    if _idiag.get("verdict") == "NO_CRITERION_INPUT_AT_ALL"
                ],
                "warmup_weakness_ranking": self._warmup_weakness_ranking(view),
                # 🔑🔑 [2026-09-15] **逐段质量门：报告，不是动作依据。**
                # 每条带 `drives_action: False`。`decide()` 的任何分支都**不**按
                # 它分岔（老板定案：这些未经标定的阈值不得再驱动自动补帧）。
                # 保留它是因为「实测值 / 阈值 / 最差窗口」对人工审查确实有用。
                "stage_quality_gates": stage_quality_gate_failures(
                    self._read_stage_result()),
                # 新契约的两维，原样透出（`converged` 已删除，不做兼容别名）。
                "stage_analysis_status": view.get("stage_analysis_status"),
                "stage_analysis_incomplete_reasons":
                    view.get("stage_analysis_incomplete_reasons") or [],
                "stage_precision_status": view.get("stage_precision_status"),
                "stage_precision_evidence": view.get("stage_precision_evidence") or {},
                # 🔑🔑 [2026-09-17] **被上交的提问必须有回应，哪怕答案是"押后"。**
                # `LOCAL_VALIDATION_CAP` 是引擎明确交给控制器的决定（见
                # `local_validation_cap_hits` 的长注释）。分支 3a 有答案，但它走
                # `_pick()` ⟹ 只对 `earliest` 生效 ⟹ 提问窗口不是 earliest 时
                # **整条分支不触发、提问被静默丢掉**。
                # 这里恒常落一份：本轮回答了谁、押后了谁、押后的理由是什么。
                # ⚠️ 这里**不改路由**：押后的理由是真的（上游可能重锚，那会让这一块
                # 取证落在即将作废的布局上 —— 分支 3a 自己的注释就警告了这一点）。
                # 押后不会变成永远：`earliest` 无路可走时会被退役，届时提问窗口
                # 自己成为 `earliest`，分支 3a 就触发了。
                "local_validation_cap_pending": [
                    {**h,
                     "answered_this_round": bool(
                         action == "PROVISIONAL_PRODUCTION"
                         and int(h["window_idx"]) in [int(x) for x in (windows or [])]),
                     "deferred_because": (
                         None if (action == "PROVISIONAL_PRODUCTION"
                                  and int(h["window_idx"]) in
                                  [int(x) for x in (windows or [])])
                         else (
                             f"本轮路由锁定在更早的未解决窗口 {earliest}"
                             "（上游重锚会作废下游 lineage，所以先修最早的）；"
                             "在它解决或被退役之前，对本窗口取证有落在"
                             "**即将作废的布局**上的风险 ⟹ 押后，不是拒绝。"
                             if earliest is not None and int(earliest) != int(h["window_idx"])
                             else "本轮的动作不是 PROVISIONAL_PRODUCTION（见 reason）。"
                         )),
                     }
                    for h in self.local_validation_cap_hits(view)
                ],
                # 🔑🔑 [2026-09-17] **路由态 ELIGIBLE 的窗口带着信号 ⟹ 永远轮不到。**
                # 见 `unrouted_window_signals` 的长注释：15 个 `_pick` 调用点里
                # 13 条押后无害（窗口会自己成为 earliest），2 条会**永久丢失**。
                # 这里恒常把它们摆出来 —— **不改路由，只保证不静默**。
                "unrouted_window_signals": self.unrouted_window_signals(view),
                # 出口专属的结构化诊断（目前只有 D3 用）。**不是**通用逃生舱：
                # 加新键之前先问"它能不能进上面某个既有字段"。
                **(dict(extra) if extra else {}),
                # 🔑 [2026-09-17，死线 D1] **终态窗口的失败信息提到 run 级。**
                # 它们原来只出现在 D1 那条分支的 `reason` 字符串里 ⟹ 换一条出口
                # （退役换窗、或 TERMINAL 窗口不是 `earliest`）就整个看不见了，
                # 而这是人工接手唯一需要的东西。D1 本身**不是控制器能修的死线**
                # （要人改输入 λ 表 / 升档预热预算 / 接受失败），所以对它能做的
                # 全部改进就是**报清楚**。与 `earliest` 是谁无关，恒常输出。
                "terminal_window_failures": [
                    {"window_idx": int(w["window_idx"]),
                     "bias_status": w.get("bias_status"),
                     "verdict": w.get("verdict"),
                     "last_failure_reason": w.get("last_failure_reason"),
                     "last_gate_error": w.get("last_gate_error")}
                    for w in (view.get("windows") or [])
                    if w.get("phase") == "TERMINAL"
                ],
            }

        # 0) 读不到任何窗口 —— 还没开跑，或者路径/目录给错了。别猜。
        if not W:
            return plan(
                "CONTINUE_WARMUP",
                f"{self.stage_dir} 下没有任何窗口产物 —— stage 还没开始，或 "
                "run_dir/stage_name 给错了。",
                missing=["任一窗口的 convergence/warmup_failure 产物"],
            )

        # 0a) 🔑🔑 [CTL-11，2026-09-14] **stage 级判据已经通过 ⟹ DONE，不许再修。**
        #
        # 先前 `DONE` 只在分支 7 判，排在「最早未解决窗口」路由**之后**；而
        # `earliest` 来自逐窗**自检**的 `self_verdict`。两者是同一个量的两份实现：
        #   · stage 级 `converged` 由 `solve_stage_integrated` 在**合并后的全部段**
        #     的帧上算（五条合取），是权威；
        #   · 自检 `min N_eff/g` 只看该窗口**一个段**的帧，对多段窗口系统性偏悲观
        #     （同 DECORR-01：自检是早期诊断，不是求解器资格）。
        # ⟹ 任一窗口自检 < 10 就永远轮不到分支 7。真机 cyclod_ligand2/rep2：
        # stage `converged=True`、无缺窗无跳窗，而 win0/win3 自检 4.37/4.81，
        # `decide()` 给出的是 `PROBE_REANCHOR_EPOCH` —— **一个已经跑完的 stage，
        # 每次 resume 都会重开一个 Epoch 烧 GPU。**
        #
        # 缺窗 / 跳窗仍然一票否决（缺窗口的和不是 ΔG，是另一个量），所以这里
        # 显式再查一遍，不只信 `converged` 一个布尔。
        # ⚠️⚠️ [2026-09-14 自查，这一条是本次改动自己引入的 regression 的补丁]
        # 把 DONE 提到路由之前，就**同时拆掉了原来由路由提供的保护**：
        # 分支 7 在 `1d-0`（产物与布局对不上）/ `6`（跳窗）/ `6b`（缺窗）之后，
        # 那几条会先拦住一个"布局已经变了、`converged` 却是旧布局结论"的盘面。
        # 而 `_read_stage_result()` 对**不带 `path_version` 的老产物**是放行的，
        # stage 缓存 `stage2_vanishing.json` 恰恰不盖版本号 ⟹
        #   布局 v1 收敛 → 后来插 λ 变 v2 → 中间产物因版本不符被跳过
        #   → 老缓存被放行 → 这里直接 DONE。**发布一个描述另一条布局的 ΔG。**
        # 所以提前判 DONE 必须额外要求：① 结果的版本**核实过**；
        # ② 没有任何窗口的证据被布局变更作废。两条都不满足就老老实实往下走路由，
        # 由分支 7 在路由之后兜底（那条路径上保护仍然成立）。
        # [2026-09-14] 版本前置抽成 `_layout_trustworthy()`，与 `_evidence_status`
        # 共用一份 —— 两处不同步正是 `DONE` + `INCONCLUSIVE` 那条自相矛盾的成因。
        # `skipped_windows` 现在只含**物理窗口**（审计 #58）：被跳的子窗不再以
        # 一个不存在的窗口号（10000+）永久封死 DONE；它们由 1c/1c-2 单独路由。
        # 🔑🔑 [2026-09-15 老板定案] **`ANALYSIS_COMPLETE` ≠ 可以判 `DONE`。**
        # 结构与身份检查通过只记「分析完整」；跨重复精度记 `UNMEASURED`，
        # **不得作为已验收结果发布**。所以这里分两个出口：
        #   · `precision_status == MEETS_CROSS_REPEAT_TARGET` ⟹ `DONE`
        #     （单次 run 里**永远不成立** —— 一个 run 无法自证精度，有意如此）；
        #   · 否则 ⟹ `NO_ACTION` + `ANALYSIS_COMPLETE_PRECISION_UNMEASURED`，
        #     **终态**（该停就停、不继续烧 GPU），但**不是 `DONE`**。
        # 🔑🔑 [2026-09-17] **已承诺的生产步数没跑完，不许判终态。**
        # 实测：全窗自检合格时 `earliest is None` ⟹ `_pick()` 恒返空 ⟹ 补足生产那条
        # 分支一次都不触发 ⟹ 一个跑了 10 万 / 目标 25 万步的窗口跟着整条 stage 一起
        # 被判 `terminal=True`。这里只**挡住终态**；真正的补帧由下面分支 5
        # （`short`）发，它在 `_frames_admission` 定义之后，会照常过块数上限与射程闸。
        # ⚠️ 补的是**已承诺的配额**（`n_steps_per_window_effective`，调用方配的目标），
        # 不是任何质量门 —— 与「未经标定的阈值不驱动补帧」不冲突：那条禁的是拿阈值
        # 要更多帧，这条只是把说好要跑的跑完。
        _below_target = [
            int(w["window_idx"]) for w in W
            if w.get("production_steps_target")
            and (w.get("production_steps") or 0) < int(w["production_steps_target"])
        ]

        if (view.get("stage_analysis_status") == "ANALYSIS_COMPLETE"
                and self._layout_trustworthy(view)
                and not view.get("stale_layout_evidence")
                and not self._coverage_incomplete(view)
                and not _below_target):          # [2026-09-17] 见上
            _common = (
                "stage 分析的**硬不变量**已通过（求解跑完、无缺窗、无跳窗、无跳子窗、"
                "布局版本核实过）⟹ `analysis_status = ANALYSIS_COMPLETE`。"
                "⚠️ 逐窗自检的 `min N_eff/g` 只看单段帧、对多段窗口系统性偏悲观，"
                "**它不是权威**，不得用它挡住一个已跑完的 stage —— 否则每次 resume "
                "都会在一个跑完的 stage 上重开修复动作（CTL-11）。"
            )
            if view.get("stage_precision_status") == "MEETS_CROSS_REPEAT_TARGET":
                return plan(
                    "DONE",
                    _common + "且 `precision_status = MEETS_CROSS_REPEAT_TARGET`"
                    "（跨重复精度已达标）⟹ 完成。"
                    "⚠️ 这仍不等于答案正确 —— docs/archive/STAGE2_ROOT_CAUSE_2026-08-28.md §2"
                    "「所有收敛门对该失效模式失明」仍然有效。",
                    exit_="DONE_UNTRUSTED" if self.allow_untrusted else "DONE",
                )
            return plan(
                "NO_ACTION",
                _common + f"但 `precision_status = "
                f"{view.get('stage_precision_status')!r}` —— **精度从未被测过**。"
                "跨重复精度只能由多次独立重复给出，**单次 run 无法自证**，"
                "所以这里不是 `DONE`，而是「分析完整 / 精度未测」这个终态："
                "这一跑没有别的事可做（继续补帧只会烧 GPU 而不提高可验证性），"
                "但结果**不得作为已验收结果发布**。"
                "逐段质量门的读数仍在 `stage_quality_gates` 里供人审查 —— "
                "它们是**未经标定**的阈值，已被明令不得驱动自动补帧。",
                exit_="ANALYSIS_COMPLETE_PRECISION_UNMEASURED",
            )

        # 🔑🔑 [2026-09-11 老板定案] **按因果依赖顺序处理"最早的未解决窗口"**，
        # **不是**按错误类型设全局优先级、也不是"看见缺窗就补"。
        #
        #     for window_idx in stage_order:
        #         if window is not ANALYSIS_ELIGIBLE:
        #             return route_this_window(window_idx)
        #     return RUN_NEXT_MISSING_WINDOW
        #
        # 实测 mismatch（replay 基线照出来的）：rep1 里 win3 支撑不足、win4 卡在
        # local cap，而"缺窗口"分支优先级最高 ⟹ 目标被判成 **win5**，把 win3/win4
        # 全盖住了。
        #
        # **物理理由，不只是调度洁癖**：win3 重锚会产生**新的末帧 / 共享态起点**，
        # win4 当前的 warmup 属于**旧上游 lineage**。win3 成功之后才谈得上 win4 能
        # 不能复用；若它的初始化依赖 win3，就得**重新开始 win4**，不能直接接旧 warmup。
        # （这跟"所有窗口共用同一份起始坐标"是同一件事的两面。）
        # ⚠️ **三态，不是两态。** 先前写成 `self_verdict != "ANALYSIS_ELIGIBLE"`，
        # 于是**没有自检产物**的老 run（`self_verdict is None`）里每个窗口都被判成
        # "未解决" ⟹ earliest 永远是 win0，动作永远落在第一个窗口上。
        #
        # "缺证据不是通过"是对的，但**缺证据也不等于「这个窗口有问题」** ——
        # 它等于「不知道」。对「不知道」的正确动作是**去产出证据**（分析/自检），
        # 不是重标定。混成一态会让控制器在毫无根据的窗口上换 Epoch。
        _order = sorted(W, key=lambda x: int(x["window_idx"]))

        _window_state = self._window_routing_state
        _states = {int(x["window_idx"]): _window_state(x) for x in _order}
        # 🔑🔑 [CTL-03，2026-09-14] **被 rewindow 取代的父窗不参与 earliest 排序。**
        # 它已经退出求解覆盖（子系综接管了它那段 λ），它的旧 warmup / f_k 状态
        # 只是历史记录。先前它照样能当 `earliest`，于是分支 1e（父窗预热预算耗尽
        # ⟹ 发 INSERT_LAMBDA / NO_ACTION）会排在未完成子窗的路由**之前**，
        # 用一个已经不在覆盖里的窗口的旧状态，挡住子窗自己的补帧。
        _replaced_parents = {
            int(x) for x in
            ((view.get("immutable_rewindow") or {}).get("parents_done") or [])
        }
        # 🔑🔑 [2026-09-15 真机 brd4_ligand1/rep2] **被求解器跳掉的窗口优先于
        # "自检说支撑不足"的窗口。**
        #
        # 两者不是同一强度的信号：
        #   · `skipped_windows` 是**求解器的操作权威** —— 那个窗口真的没进 MBAR，
        #     整条路径因此**缺窗**，`analysis_status` 判 `ANALYSIS_INCOMPLETE`，
        #     交出去的和**不是 ΔG**（硬不变量，本仓不许放宽）；
        #   · `self_verdict` 只是逐窗自检的诊断，说的是"这个窗还没测够"。
        #
        # 先前 `earliest` 只按下标排序、两者一视同仁 ⟹ 一个**加帧治不好**的窗口
        # 会把真正让整条路径不成立的那个无限期挡在后面。真机：
        #     win1 self_verdict=INSUFFICIENT_DATA（n_decorr=888 够得离谱、
        #          min N_eff/g=8.68 ⟹ 是**偏斜**，同分布加帧治不了）
        #     win4 HARD_INSUFFICIENT，n_decorr=7 ⟹ 被求解器跳掉，skipped_windows=[4]
        # 6 轮全部路由到 win1（4 块帧烧光配额），win4 一次都没被看过；
        # 整条路径照常采完，最后死在缺窗那道身份门上。
        # 而 win4 恰恰有现成的对症动作（`INSERT_LAMBDA` 缩跨度），只是轮不到。
        #
        # ⚠️ 这**不放宽任何判据**：只改"先修哪一个"的顺序。被跳的窗口仍然要过
        # 同一套可行性与预算闸；自检类窗口一个都没被跳过，只是排在后面。
        # ⚠️ `_replaced_parents` 的排除照旧（父窗已退出求解覆盖，修它没有意义）。
        _skipped_now = {
            int(x if not isinstance(x, dict) else x.get("window_index", -1))
            for x in (view.get("skipped_windows") or [])
        } - {-1}
        # 🔑🔑 [2026-09-16] `_retired` = **本轮已经判定"对它没有任何动作可做"的窗口**。
        # 它们退出 `earliest` 排序（因此也不再挡住下游），但**不被当成合格**：
        # 窗口状态、`analysis_status`、质量门一个都不放宽。语义只有一句：
        # 「这个窗口修不动了，别再让它把还能修的窗口一起拖死。」
        # 谁进这个集合由 `decide()` 那层决定（只有终态 + 别处确实还有预算时才退役）。
        _retired = {int(x) for x in (retired or ())}
        earliest = next(
            (i for i in sorted(_states)
             if i in _skipped_now and i not in _replaced_parents
             and i not in _retired),
            None,
        )
        if earliest is None:
            earliest = next(
                (i for i in sorted(_states)
                 if _states[i] == "PROBLEM" and i not in _replaced_parents
                 and i not in _retired),
                None,
            )
        unknown = [i for i in sorted(_states) if _states[i] == "UNKNOWN"]
        # 🔑 [2026-09-17] **已退役的窗口不算"被上游挡住"。**
        # `blocked_by_upstream` 的语义是「它们的证据前提可能被 `earliest` 的重锚
        # 作废，所以先别动」。而**已退役**的窗口恰恰是「本轮已判定对它没有任何
        # 动作可做」——它不在等 `earliest`，它是被绕过去的那个。两件事混在一张
        # 表里会让人读成"下游还有 N 个在排队"，而其中几个其实已经放弃了。
        # 纯报告字段，不改任何路由。
        blocked = (
            [int(x["window_idx"]) for x in _order
             if int(x["window_idx"]) > earliest
             and int(x["window_idx"]) not in _retired]
            if earliest is not None else []
        )
        # 本轮路由上下文就位 ⟹ `plan()` 从这里取默认值（见 `_ctx` 的长注释）。
        _ctx.update(earliest=earliest, blocked=blocked)
        # 🔑🔑 [REWIND-01，2026-09-17] **λ 表上的两个动作都不可行 ≠ 没有对症动作。**
        #
        # `SPLIT_TAIL_WINDOW` / `INSERT_LAMBDA` 是仅有的两个缩跨度动作，而它们都锚在
        # 尾部：拆末窗要 tail anchor（`first_untrusted_window` 在 `idx <= 0` 时恒为
        # None），插 λ 在末窗顶到 `hi` 且没有 anchor 时也不可行，再加一个跨 resume 的
        # `max_path_insertions`。**失败窗口是 window 0 时两个恒不可行** —— 而 window 0
        # （解耦端点）正是本方法的必然坏窗口。于是四条独立的分支（1e 预热预算耗尽、
        # 5a-2 累计 f_k + held-out REJECT、边际增益停滞、5b 偏斜类自检失败）全都汇到
        # 同一句 `NO_FEASIBLE_ACTION`：控制器**有**对症归因，却没有一个够得着的动作。
        #
        # `IMMUTABLE_REWINDOW` 恰好绕开全部四个封锁条件（见 `rewindow_feasible`），
        # 成本表 / 执行器 / 子窗调度 / no-op 台账早就齐了，唯独缺这一个发出点
        # （`docs/TODO.md` 的 REWIND-01：13 个声明动作里唯一发不出的那个）。
        # ⚠️ 接在**每条死线各自的兜底之前**，不是在链尾加第 33 条分支 ——
        # 否则又是「按书写顺序决定语义」。
        def _bounded_rewindow(win, why):
            """缩跨度都不可行时退到有界重窗。不可行返回 None，由原来的兜底接手。"""
            if self.rewindow_feasible(view, win) is not None:
                return None
            _c = self.rewindow_children(view, int(win))
            return plan(
                "IMMUTABLE_REWINDOW",
                why
                + f"⟹ 退到**固定 λ 表上的有界重窗**：在窗口 {int(win)} 内按中点建 "
                f"{len(_c)} 个重叠子系综 {_c}，各自预热、各自锁一份 f_k。"
                "它**不动 λ 表**（不需要 tail anchor、不吃插点预算、末窗一个态都不涨）"
                "⟹ 上面那些封锁条件一个都不碰；子系综是**新系综**、有自己的预热预算"
                "⟹ 也绕开「重进父窗预热被弹回」。原数据一个字节不改"
                "（独立目录 `<stage>_rewindow_<identity>/`）。"
                "**一个父窗只切一层**：子窗再失败不再递归。",
                windows=[int(win)], blocked=blocked, earliest=earliest,
            )

        # 🗑️ [2026-09-15] 这里原有 `_sidx_to_unit` / `_unit_by_id` / `_route_target`
        # 三样，**唯一**的用途是把 stage 质量门报的 worst_window（solver 索引）翻成
        # 调度身份，好让分支 9b/9c 据此发补帧 / 缩跨度动作。那两个分支已删除
        # （未经标定的阈值不得驱动自动补帧），这三样随之无人使用 ⟹ 一并删掉，
        # 不留"以后也许用得上"的死代码。子窗的路由仍由分支 1c 自己做。

        # 🔑🔑 [2026-09-14] **补帧准入：批下一块之前，先看上一块换来了什么。**
        #
        # 真机可以一路 25→50→…→150 万步：早期分支（求解器跳窗 ⟹ 补帧）走不到
        # 后面的边际收益刹车，而停滞保护把"多跑了一块"算成盘面进展、把重复计数
        # 清零。**"又产生了帧"不等于"获得了有用证据"。**
        #
        # 两条准入，缺一不可：
        #   ① **块数硬上限**（每单元，config 可调，缺省 4 块）——
        #      生产 cap 缺省是 unknown，拦不住，必须有这道；
        #   ② **上一块必须有实质增益** —— 判据量是**求解器侧**的去相关帧数
        #      （不是自检那份，两者差 2–8 倍；也不是步数）。没涨就停，并说清楚。
        _blocks = view.get("production_blocks_by_window") or {}
        # 🔑 [BUD-03] 硬上限数的是**跨全部采样段**的块（换段不重置配额），
        # 边际增益仍然只在**同段**内比（跨 f_k epoch 比数字没有意义，设计 §6.10）。
        # 先前两者共用同段那本 ⟹ 换一次 Epoch 配额清零，实测单窗烧到 150 万步。
        _blocks_total = view.get("production_blocks_total_by_window") or _blocks
        # 🔑🔑 [契约 B] **子窗（sampling unit）有自己的两本账。**
        # 先前块账只扫 `view["windows"]`，而 `sampling_units` 一个都不在里面，
        # `plan()` 又拿**父窗**的账去问准入（父窗步数已冻结 ⟹ 去重后恒 1 行）⟹
        # 子窗的块数硬上限与边际刹车**双双失效**（BUD-03 修过的「单窗烧 150 万步」
        # 在子窗这个新位置原样复发）。
        _blocks_u = view.get("production_blocks_by_unit") or {}
        _blocks_total_u = view.get("production_blocks_total_by_unit") or _blocks_u
        _max_blocks = int(view.get("max_production_blocks_per_window") or 4)


        _noop_led = view.get("noop_actions") or {}
        _by_idx_for_noop = {int(w["window_idx"]): w for w in W}
        _by_uid_for_noop = {str(u["unit_id"]): u
                            for u in (view.get("sampling_units") or [])}



        # 0b) **身份不一致**：盘上的 converged 是另一个系综的结论，不得当依据。
        # 🔑🔑 [2026-09-14 第三轮复核] **这一条以前物理上排在分支 5 之后。**
        # 编号写着 `0b`、意图是"最早"，位置却在 `5)（生产帧没攒够）` 后面 ——
        # 而 `IDENTITY_MISMATCH` 的窗口 `production_steps` 往往就是没到目标，
        # 于是分支 5 先命中、对着一个**另一个系综**的产物发 RUN_PRODUCTION，
        # `HALT_INVALID_INPUT` 永远轮不到。身份不合法时任何动作都不该发，
        # 所以它必须排在全部路由之前。
        _stale_id = _pick(earliest, [w for w in W if w.get("phase") == "IDENTITY_MISMATCH"])
        if _stale_id:
            return plan(
                "CONTINUE_WARMUP",
                f"窗口 {_stale_id} 的 `stage_protocol_key` 在 ibs_state 与产物之间不一致 ⟹ "
                "盘上那个 `bias_status=converged` 是**另一个系综**的旧结论，"
                "不得据此声称已在生产。λ / Hamiltonian / box / 规范版本任一变化，"
                "旧 PASS 都不能继续粘住。",
                exit_="HALT_INVALID_INPUT", windows=_stale_id,
                blocked=blocked, earliest=earliest,
            )

        # 0c) 🔑🔑 [审计 #23，2026-09-14] **`phase == "TERMINAL"` 全仓没有任何分支处理。**
        #
        # `bias_status ∈ {failed, calibrated_validation_failed}` 映成 `TERMINAL`，
        # `_window_state()` 判它 PROBLEM ⟹ 它**当上 `earliest`** 并挡住所有下游窗口；
        # 而后面每一条分支的候选集合（warming / short / skipped / …）都不含它 ——
        # `_pick` 于是对每一条都返回 []，最后掉到 9d 以一个**与根因完全无关**的理由
        # （"归因不到任何一道具体的门"）退出。真正的根因（这个窗口的 f_k 已经被判死）
        # 一个字都没出现在结局里。
        # 这里显式分流：f_k 被驳回走已有的 refuted 路径（分支 2），其余情形如实发
        # `NO_ACTION` + `NO_FEASIBLE_ACTION`，并把 `last_failure_reason` /
        # `last_gate_error` 原样带出去 —— 它们就是人工接手所需要的全部信息。
        if earliest is not None and int(earliest) not in _replaced_parents:
            _tw = next((w for w in W if int(w["window_idx"]) == int(earliest)), {}) or {}
            if _tw.get("phase") == "TERMINAL":
                if _tw.get("verdict") == "STATISTICALLY_REJECTED":
                    pass          # 有统计功效的驳回 ⟹ 交给分支 2 的既有路径
                else:
                    return plan(
                        "NO_ACTION",
                        f"窗口 {earliest} 的 `bias_status` 是终态"
                        f"（{_tw.get('bias_status')!r} ⟹ phase=TERMINAL）：这份 f_k 的"
                        "预热/验证**已经判死**，不是「尚不可测」。"
                        f"失败原因 `{_tw.get('last_failure_reason')}`，"
                        f"最后一次门错误 `{_tw.get('last_gate_error')}`。"
                        "控制器没有能推动它的动作 —— 重解/换 Epoch 都要先重新进入这个"
                        "窗口，而它的预热在入口就会被弹回；加帧同理。"
                        "要继续需要人工判断（改输入 λ 表 / 显式升档预热预算 / "
                        "接受这个窗口失败）。**不是 DONE** —— 结果没有达标。",
                        exit_="NO_FEASIBLE_ACTION", windows=[int(earliest)],
                        blocked=blocked, earliest=earliest,
                        # [2026-09-17] phase=TERMINAL：这份 f_k 已判死 ⟹ 证据不成立，**不得**绕过去采下游
                        halt_scope="PATH_INVALID",
                    )

        # 1d-0) 🔑🔑 [2026-09-14 真机] **产物与当前布局对不上的窗口，只能重采。**
        #
        # 真机 `cyclod_ligand1/rep2` 17:57 崩在这上面：
        #     ValueError('窗口 3 状态数与 window_ranges 不符')  ← 炸出流水线
        # 实测盘面：路径已演化到 v2（win3 从 6 态涨到 7 态），`ibs_state` 里的
        # `lambdas_vdw` / `f_k` 已经是 7，而该窗口的**生产产物**
        # （`convergence.json` / `energies.npy`）还是**6 态的旧帧**。
        # 控制器知道证据过期（`stale_layout_evidence`），却仍然选了
        # `PROBE_REANCHOR_EPOCH` —— 那是"拿已有帧重解"那一族，而这些帧描述的是
        # **另一个窗口几何**，解出来必然维度不符。
        #
        # 与 1d（被求解器跳掉）、1e（预热预算耗尽）是**同一个形状**的第三例：
        # **一个在构造上不可能成功的动作被发了出去**。插过 λ 之后唯一能推动的
        # 只有重采。
        _stale_now = self.stale_layout_windows(view)
        if earliest is not None and int(earliest) in _stale_now:
            # 🔑🔑🔑 [S2-M，2026-09-17 真机] **「只能重采」还得问一句：它进得去吗？**
            #
            # 本分支排在 1e（预热进不去）**之前**，而它无条件发 `RUN_PRODUCTION`。
            # 于是「布局过期 **且** 冻结验证批次打满」的窗口进了死锁：
            #     1d-0 判「只能重采」→ 重采要**重新进入**窗口 → 预热门看到
            #     15/15 批打满 → 抛 `LOCAL_VALIDATION_CAP` 路由信号 → 弹回，盘面未变
            #     → no-op 还记不下（stale 窗口 `production_steps` 读不到 ⟹ 指纹
            #        没有可比身份，2026-09-15 那条保护刻意不落账）
            #     → 连发 3 次 → 停滞保护 → 降级又被 `_stale_for_escalation` 挡住
            #     → NO_FEASIBLE_ACTION，整跑终止。
            #
            # 而对「验证批次打满」这件事控制器**有**答案 —— 分支 3a 的
            # `PROVISIONAL_PRODUCTION`：显式授权引擎用这份**未验证**的冻结 f_k 采一块。
            # 它同样产出**新布局下的**帧，所以一并解决了"证据过期"；
            # 而普通 `RUN_PRODUCTION` 会在预热门上被弹回，一帧都产不出来。
            # 只是 3a 排在本分支后面，够不着。
            #
            # ⟹ 这里分流：过期 **且** 正卡在验证批次上限 ⟹ 发 `PROVISIONAL_PRODUCTION`；
            # 其余情形照旧发 `RUN_PRODUCTION`。
            # ⚠️ 不放宽任何判据：`PROVISIONAL_PRODUCTION` 的语义、额度（只给一个块）、
            # 证据口径（f_k 仍是 `indeterminate`、**不是**可信 PASS）全部照 3a 那套，
            # 判据也复用**同一份** `local_validation_cap_hits()`，不在这里另写。
            _cap_now = {int(h["window_idx"])
                        for h in self.local_validation_cap_hits(view)}
            if int(earliest) in _cap_now:
                return plan(
                    "PROVISIONAL_PRODUCTION",
                    f"窗口 {earliest} **同时**满足两件事：产物描述的是另一套 λ 布局"
                    "（只能重采），**而且**单周期冻结验证批次已打满 ⟹ "
                    "普通 `RUN_PRODUCTION` 会在预热门上被 `LOCAL_VALIDATION_CAP` "
                    "原样弹回、一帧都产不出来（真机实测连发 3 次、盘面一字节没变，"
                    "最后以 NO_FEASIBLE_ACTION 终止整跑）。"
                    "⟹ 走 `PROVISIONAL_PRODUCTION`：显式授权引擎用这份**未验证**的"
                    "冻结 f_k 采一个诊断块 —— 它产出的是**新布局下**的帧，"
                    "一并解决了证据过期。"
                    "⚠️ **不是可信 PASS**：f_k 证据保持 `indeterminate`，"
                    "`warmup_failure.json` 留作证据，额度仍然只有一个块。",
                    exit_="HALT_LOCAL_VALIDATION_CAP",
                    windows=[int(earliest)], blocked=blocked, earliest=earliest,
                )
            return plan(
                "RUN_PRODUCTION",
                f"窗口 {earliest} 的生产产物描述的是**另一套 λ 布局**"
                "（插 λ / 拆窗之后，旧帧的态数与当前 `window_ranges` 不符）⟹ "
                "**任何拿已有帧重解的动作（重标定 / 换 Epoch / 累计残差）在构造上都会"
                "维度不符**，真机实测是直接 `ValueError` 炸出流水线。"
                "唯一能推动它的动作是**按新布局重采**。",
                windows=[int(earliest)], blocked=blocked, earliest=earliest,
            )

        # 1e) 🔑🔑 [2026-09-14 真机] **预热预算耗尽且 f_k 未判定的窗口，根本进不去。**
        #
        # 要给这种窗口补生产帧，流水线必须**重新进入**这个窗口；而入口处的 warmup
        # 门会先看它的累计预算 —— 已耗尽就抛 `LOCAL_VALIDATION_CAP`
        # （"累计预算已用尽（上限 555000、已耗 555000），本次可用 0 步…要继续必须
        # 显式升档"）。于是 `RUN_PRODUCTION` 在**构造上**推不动这个窗口。
        # 真机：win0 连发 40 轮、每轮都被路由信号弹回来，而路由那条 `continue`
        # 还绕过了 no-op 记账，连"这条路走不通"都留不下。
        #
        # 对症动作只有两个：**缩跨度**（布局动作，不需要重进预热），
        # 或**显式升档 warmup 预算**（调用方的决定，控制器不替它做）。
        if earliest is not None and int(earliest) not in _replaced_parents:
            _ew = next((w for w in W if int(w["window_idx"]) == int(earliest)), {}) or {}
            _left = _ew.get("warmup_steps_left")
            _unenterable = (
                _left is not None and int(_left) <= 0
                and str(_ew.get("f_k_evidence_status") or "") not in ("verified",)
            )
            if _unenterable:
                _can_ins = feas.get("insert_lambda") is None
                if _can_ins:
                    return plan(
                        "INSERT_LAMBDA",
                        f"窗口 {earliest} 的 warmup 累计预算已耗尽（剩 {_left} 步）"
                        f"且 f_k 仍是 `{_ew.get('f_k_evidence_status')}` ⟹ "
                        "**重新进入该窗口采样会在预热门上被直接弹回**"
                        "（LOCAL_VALIDATION_CAP），补帧类动作在构造上推不动它。"
                        "不需要重进预热的对症动作是**缩跨度**：插 λ。",
                        windows=[int(earliest)], blocked=blocked, earliest=earliest,
                    )
                _rw = _bounded_rewindow(
                    int(earliest),
                    f"窗口 {earliest} 的 warmup 累计预算已耗尽（剩 {_left} 步）且 f_k 仍是 "
                    f"`{_ew.get('f_k_evidence_status')}` ⟹ 补帧会在预热门上被弹回；"
                    f"而插 λ 也不可行（{feas.get('insert_lambda')}）",
                )
                if _rw is not None:
                    return _rw
                return plan(
                    "NO_ACTION",
                    f"窗口 {earliest} 的 warmup 累计预算已耗尽（剩 {_left} 步）且 f_k 仍是 "
                    f"`{_ew.get('f_k_evidence_status')}` ⟹ 补帧会在预热门上被弹回；"
                    f"而缩跨度也不可行（插 λ：{feas.get('insert_lambda')}；"
                    f"有界重窗：{self.rewindow_feasible(view, earliest)}）。"
                    "**要继续必须显式升档该窗口的 warmup 预算** —— 那是调用方的决定，"
                    "控制器不替它做，也不假装还有别的动作。",
                    exit_="NO_FEASIBLE_ACTION", windows=[int(earliest)],
                    blocked=blocked, earliest=earliest,
                    # [2026-09-17] 这个窗口进不去且缩不了跨度；别的窗口不受影响
                    halt_scope="TARGET_LOCAL",
                )

        # 🔑🔑 [2026-09-11 老板定的决策顺序] 选动作**之前先消费预算可行性**：
        #
        #     聚合同一物理 stage 的全部 Segment
        #     → 找 earliest unresolved window
        #     → **检查该窗口动作是否有预算**      ← 这一步
        #     → 按证据选择 RECALIBRATE / PROBE / RESCUE
        #     → 只有完整前缀全部 eligible，才运行下一个缺失窗口
        #
        # 实测 mismatch：`run2/vanishing_2` 在 555k/555k（零余量）时仍被判
        # `RECALIBRATE_FK` —— 动作可行性没有先消费预算。零预算窗口开新 Epoch
        # 只会得到一份**永远验不了**的 f_k（win2 连死三次就是这个形状）。
        #
        # 🔑🔑 [审计 #20] **这道闸已经挪进 `plan()`**（见那里的长注释）。
        # 留在这里的版本是**无条件**跑的：不分接下来要发什么动作，一律先判
        # "这个窗口付不付得起换 Epoch"，付不起就 return `RUN_PRODUCTION` ——
        # 于是账本读不到的窗口被永久钉死，后面所有分支（1c/1d/3/3a/4/5/5a/5b/6）
        # 一条都到不了。预算消费的**顺序**没变（仍在选定动作之后、返回之前），
        # 变的只是"只对真要花这本账的动作消费"。

        # 1c) 🔑🔑 [2026-09-14 裁决 1] **未完成的子窗优先于对父窗口的任何动作。**
        #     父窗口已经被子系综取代（它整个不在求解覆盖里），所以对它本身做什么
        #     都没有意义；能推动结果的只有把子窗补完。按因果顺序挑
        #     （父窗号、局部下标），并且**不越过更早的未解决物理窗口**。
        #     动作带 `unit_id` —— 执行器据此续跑**那一个子窗自己的 checkpoint**，
        #     绝不回头动父窗（子窗局部下标 0/1 与父窗索引同名不同义）。
        # 🔑 [CTL-02] **只有"缺帧"才走补帧这条分支。** 支撑/偏斜类失败的子窗
        # 由下面的 1c-2 单独处理 —— 拿加帧顶替它正是要修的那个错。
        # [2026-09-18] 父窗已退役的子窗本轮不再路由（理由同 1c-2 / 1c-3）：
        # 本分支发的 `RUN_PRODUCTION` 会被 `plan()` 的补帧准入闸改写成终态，
        # 而它不看 `retired` ⟹ 退役之后原样再触发、再交出同一个终态，
        # 退役循环空转一圈就收摊。
        _pending_units = sorted(
            [u for u in (view.get("sampling_units") or [])
             if u.get("needs_frames")
             and int(_pw_or_last(u)) not in _retired],
            # [2026-09-17] 父窗未知 ⟹ 排到**最后**（大数），不冒充 window 0。
            # 先前 `or 0` 让它插到队首、并被当成 window 0 的子窗路由。
            key=lambda u: (_pw_or_last(u), int(u.get("local_index") or 0)),
        )
        if _pending_units:
            _u = _pending_units[0]
            _pw = _u.get("parent_window")
            if _pw is None:
                # 父窗未知 ⟹ 不知道它在因果顺序里排哪儿，路由它可能越过未解决的
                # 上游窗口。fail-closed：跳过，并由 `plan()` 的常设字段报出来。
                _pending_units = []
            elif earliest is None or int(_pw) <= int(earliest):
                _why = (
                    "还没有生产产物" if not _u.get("has_convergence")
                    else ("被求解器踢出协方差链"
                          f"（去相关后 {(_u.get('solver_skip') or {}).get('n_frames_after_decorrelation')} 帧）"
                          if _u.get("solver_skip")
                          else f"自检判帧数不足（verdict={_u.get('self_verdict')}）")
                )
                return plan(
                    "RUN_PRODUCTION",
                    f"父窗口 {_pw} 已被 immutable rewindow 的子系综取代，子窗 "
                    f"`{_u['unit_id']}`（局部下标 {_u.get('local_index')}，λ 区间 "
                    f"{_u.get('range')}）{_why} ⟹ **续跑这个子窗自己的 checkpoint**"
                    f"（已采 {_u.get('production_steps')} 步）。"
                    "语义是「尚不可测」（INSUFFICIENT_DATA ≠ FAIL）；"
                    "对父窗口本身做任何动作都推不动结果 —— 它整个不在求解覆盖里。",
                    windows=[_pw], unit_id=_u["unit_id"],
                    blocked=blocked, earliest=earliest,
                    # 🔑🔑 [2026-09-18] **动作虽然是 `RUN_PRODUCTION`，也必须点名。**
                    # 它会被 `plan()` 的补帧准入闸 / no-op 台账改写成
                    # `NO_ACTION` + 终态（`TARGET_LOCAL`）—— 到那时 `blocking_window`
                    # 兜底成 `earliest`，而卡死的是**这个子窗的父窗**。
                    # 本分支的守卫只要求 `_pw <= earliest`：`_pw < earliest` 时退役
                    # 会退掉无辜窗口；`earliest is None` 时它是 `None`，
                    # `_retirable_window()` 第一行就返回 ⟹ 连「别处还有没有活干」
                    # 都不问。与 1c-2 / 1c-3 同一形状的第三处。
                    blocking_window=int(_pw),
                )

        # 1d) 🔑🔑 [2026-09-14 真机] **被求解器跳掉的窗口只能加帧，不能重解。**
        #
        # 「重标定 f_k」「换 Epoch 拿独立证据」都要**拿这个窗口已有的生产帧重解**。
        # 而求解器把它跳掉的意思恰恰是「这些帧去相关之后不够解」（9 帧 < 门 10）
        # ⟹ 那一整族动作在**构造上**就是 no-op。
        #
        # 真机 cyclod：win0 被跳（9 帧），f_k 探针按**节奏**（已采 ≥ 500k 步）把它
        # 列进建议重标定名单，而执行器按**相邻位移 > 0.5 kJ/mol** 判 —— 同一个决定
        # 两套判据，且 win0 **根本没有位移读数**（它解不出来，位移表里只有
        # w1/w2/w4/w5）。于是 `RECALIBRATE_FK[0]` 连发 4 次、停滞保护降级到
        # `PROBE_REANCHOR_EPOCH` 又是同一个 no-op，最后 NO_FEASIBLE_ACTION 退出。
        # 全程盘面一个字节没变，而它真正需要的动作（加帧）就在分支 6。
        #
        # 「低支撑永远不是 FAIL，是『尚不可测』，动作是加预算」——
        # 这条在这里就该生效，不该等 f_k 那一整族先空转一遍。
        _solver_skipped_now = (
            {int(x) for x in (view.get("skipped_windows") or [])}
            | {int(w["window_idx"]) for w in W if w.get("solver_skip")}
        )
        if (earliest is not None and int(earliest) in _solver_skipped_now
                and int(earliest) not in _replaced_parents):
            _sk = next((w.get("solver_skip") for w in W
                        if int(w["window_idx"]) == int(earliest)), None) or {}
            return plan(
                "RUN_PRODUCTION",
                f"窗口 {earliest} 去相关帧数不足、被**求解器**踢出协方差链"
                + (f"（去相关后 {_sk.get('n_frames_after_decorrelation')} 帧 < "
                   f"{_sk.get('min_frames_per_window')}）"
                   if _sk.get("n_frames_after_decorrelation") is not None else "")
                + " ⟹ 它的帧不够解，**任何拿已有帧重解的动作（重标定 / 换 Epoch）"
                "在构造上都是 no-op**。对症动作是**加帧**："
                "低支撑永远是「尚不可测」（INSUFFICIENT_DATA ≠ FAIL），动作是加预算。",
                windows=[int(earliest)], blocked=blocked, earliest=earliest,
            )

        # 1c-2) 🔑🔑 [CTL-02] **支撑/偏斜类失败的子窗：加帧治不了，别拿它顶替。**
        #
        # 这条与上面 1c 是**两种科学问题**，不是同一件事的两个程度：
        #   · 1c   缺样本量   ⟹ 加帧（INSUFFICIENT_DATA，"尚不可测"）
        #   · 1c-2 支撑/偏斜 ⟹ 缩跨度；没有有界的缩跨度动作就**如实停下**
        # 先前两者都落进 `complete=False` 这一个布尔、都发 `RUN_PRODUCTION`，
        # 于是 top1%/raw ESS 失败的子窗被反复加帧（§5.1 实测越加越差）。
        # 🔑 [2026-09-17 P0] `support_failed` 现在**只**由 `STRUCTURAL` 置位
        # （见 `read()` 的第二遍归因）。`UNKNOWN` 不进这个选择器 —— D3 停机是
        # 「有界重窗这条路用尽」的结论，归因判不出来时说不出这句话。
        # 🔑🔑 [2026-09-18] **父窗已退役 ⟹ 它的子窗本轮也不再路由。**
        # 退役的语义就是「对这个目标已经没有任何可行动作，别让它把还能修的
        # 窗口一起拖死」。不在这里排除的话，本分支不看 `retired` ⟹ 退役之后
        # `_decide_once` 重跑时它原样再触发、再交出同一个终态，退役等于只在
        # `retired_windows` 里留了条记录，**下游该跑的窗口仍然一步没跑**。
        _skew_units = sorted(
            [u for u in (view.get("sampling_units") or [])
             if u.get("support_attribution") == SUPPORT_FAILURE_STRUCTURAL
             and not u.get("needs_frames")
             and int(_pw_or_last(u)) not in _retired],
            key=lambda u: (_pw_or_last(u), int(u.get("local_index") or 0)),
        )
        if _skew_units:
            _su = _skew_units[0]
            _spw = _su.get("parent_window")
            if _spw is not None and (earliest is None or int(_spw) <= int(earliest)):
                # 🔑🔑 [2026-09-17，用户拍板 D3] **专门的终止码，而不是笼统的
                # `NO_FEASIBLE_ACTION`。**
                #
                # 这条死线（DEAD_LINES 的 D3）在 2026-09-17 之前**结构上不可达**
                # （子窗只由 `IMMUTABLE_REWINDOW` 产生，而它发不出来）。REWIND-01
                # 接通之后它第一次真正会被走到，所以它值一个说得清的名字：
                # 走到这里意味着「**有界重窗这条路已经用尽**」——不是"没动作可做"
                # 那种笼统无解，而是一个具体的、可以拿去做决定的结论。
                #
                # ⚠️ **不开放递归。** 现在的数据模型**只能表达"一次物理父窗替代"**：
                # `rewindow_feasible()` 只收物理 `window_idx`；执行器与合并器都没有
                # `parent_unit_id` / ancestry / 局部替代 / 递归预算语义。直接放开会
                # 产生重复区间覆盖、identity 冲突和预算失真。要开放必须先设计
                # 层级契约，而且门槛是：**至少三个独立 seed 稳定复现 D3，并证明
                # 二次切分优于停机。**
                # ⚠️ 也**不再补帧**、**不创建第二个 rewindow entry**。
                #
                # 诊断一次给全（unit / range / identity / 失败来源 / g / η / 已用块数）
                # —— 这条终态的全部价值就是让人不用再去翻盘面。
                _d3_blocks = len(
                    (view.get("production_blocks_total_by_unit") or {}).get(
                        str(_su.get("unit_id")), []) or [])
                _d3 = {
                    "unit_id": str(_su.get("unit_id")),
                    "parent_window": _spw,
                    "range": _su.get("range"),
                    "identity": _su.get("identity"),
                    "support_failure_source": _su.get("support_failure_source"),
                    "self_verdict": _su.get("self_verdict"),
                    "blocks_used": _d3_blocks,
                    # 瓶颈态观测（同 k*、同 frame set；读不到是 None = UNKNOWN）
                    "bottleneck_state": _su.get("bottleneck_state"),
                    "bottleneck_g": _su.get("bottleneck_g"),
                    "bottleneck_eta": _su.get("bottleneck_eta"),
                    "bottleneck_ratio": _su.get("bottleneck_ratio"),
                }
                return plan(
                    "NO_ACTION",
                    f"子窗 `{_su['unit_id']}`（父窗 {_spw}，λ 区间 {_su.get('range')}，"
                    f"identity {_su.get('identity')}）的失败是**支撑/偏斜类**"
                    f"（归因 {_su.get('support_failure_source')}，"
                    f"verdict={_su.get('self_verdict')}，已批 {_d3_blocks} 块；"
                    f"瓶颈态 k*={_su.get('bottleneck_state')}，"
                    f"g={_su.get('bottleneck_g')}，η={_su.get('bottleneck_eta')}，"
                    f"N_eff/g={_su.get('bottleneck_ratio')}）——**不是样本量不足**。"
                    "同分布加帧治不了偏斜（§5.1 实测 250k→1M 让 top1% 从 0.545 涨到 "
                    "0.762、ESS 比值反而更差），而它已经是**固定 λ 表上的有界重窗**、"
                    "**有界重窗这条路到此为止**：当前数据模型只能表达「一次物理父窗"
                    "替代」（执行器/合并器没有 `parent_unit_id`、没有 ancestry、"
                    "没有递归预算语义），强行再切一层会产生重复区间覆盖、identity "
                    "冲突和预算失真 ⟹ **如实停下**，不递归、不补帧、不再建第二个"
                    "子系综。要继续需要显式换布局或升档预算；要开放二次切分，"
                    "先拿**至少三个独立 seed** 稳定复现这条终态、并证明二次切分优于停机。",
                    exit_="D3_REWINDOW_DEPTH_EXHAUSTED", windows=[_spw],
                    unit_id=_su["unit_id"], blocked=blocked, earliest=earliest,
                    # 🔑🔑 [2026-09-18] **必须显式点名卡死的是谁。**
                    # 不传时 `plan()` 拿 `earliest` 兜底，而本分支的触发条件只要求
                    # `_spw <= earliest` —— `_spw < earliest` 时 `blocking_window`
                    # 就指向一个**与 D3 无关**的窗口。出口的 scope 是 `TARGET_LOCAL`
                    # ⟹ `decide()` 的退役循环照着它退役 `earliest`；而 `_skew_units`
                    # 不随 `retired` 变化 ⟹ 下一轮同一条分支再触发、再退役下一个，
                    # 直到全窗退役。返回的 plan 于是声称那些窗口「已无任何可行动作」
                    # —— 它们一次都没被判过。
                    # 正确的退役目标就是 `_spw`：它的子系综这条路已经用尽，绕过它
                    # 继续调度别处，正是这条出口的语义（用户规格「不能让它停掉整跑」）。
                    blocking_window=int(_spw),
                    extra={"rewindow_depth_exhausted": _d3},
                )

        # 1c-3) 🔑🔑 [2026-09-17 P0] **归因判不出来的子窗：两边都不授权。**
        #
        # `UNKNOWN` 不是"介于两者之间"，是「**还没判出来**」。它不得授权
        # `RUN_PRODUCTION`（可能是结构性的，加帧越加越差）、也不得授权
        # `D3 停机` / `IMMUTABLE_REWINDOW` / `SPLIT_TAIL_WINDOW` / `INSERT_LAMBDA`
        # （可能只是数据缺口，改布局是拿"没测出来"当"测出来是坏的"）。
        # ⚠️ **但它不该停掉整跑**：别处还有活干就绕过它继续调度。所以这里只在
        # 「没有 earliest、也没有别的可做单元」时才发终态。
        # [2026-09-18] 父窗已退役的子窗本轮不再路由（理由同 1c-2 那条）。
        _unk_units = [u for u in (view.get("sampling_units") or [])
                      if u.get("attribution_unknown")
                      and not u.get("needs_frames")
                      and u.get("schedulable", True)
                      and int(_pw_or_last(u)) not in _retired]
        if _unk_units and earliest is None and not _pending_units:
            _uu = _unk_units[0]
            return plan(
                "NO_ACTION",
                f"子窗 `{_uu['unit_id']}`（父窗 {_uu.get('parent_window')}，"
                f"λ 区间 {_uu.get('range')}）自检判失败，但**归因判不出来**"
                f"（verdict={_uu.get('self_verdict')}，来源 "
                f"{_uu.get('support_failure_source')}；射程三输入 "
                f"ratio={_uu.get('min_n_eff_over_g')} / "
                f"target={_uu.get('self_n_eff_over_g_eligible')} / "
                f"headroom 缺任一 ⟹ 判不了）。"
                "`UNKNOWN` **两边都不授权**：补帧可能是"
                "越加越差（若其实是结构性），改布局则是拿「没测出来」当"
                "「测出来是坏的」（若其实只是数据缺口）。"
                "⟹ 停在诊断态。要往下走，先把缺的读数补出来"
                "（`n_eff_over_g_eligible_threshold` / 块账 / 自检来源），"
                "**不是**挑一个动作试试看。",
                exit_="SUPPORT_ATTRIBUTION_UNKNOWN",
                windows=([int(_uu["parent_window"])]
                         if _uu.get("parent_window") is not None else []),
                unit_id=str(_uu["unit_id"]),
                missing=[f"子窗 {_uu['unit_id']} 的支撑失败归因输入"],
                blocked=blocked, earliest=earliest,
                # 🔑🔑 [2026-09-18] **不点名就等于不可退役。**
                # 本分支的守卫是 `earliest is None` ⟹ `plan()` 的兜底把
                # `blocking_window` 也算成 `None` ⟹ `_retirable_window()` 第一行
                # （`e is None ⟹ return None`）直接返回，于是它下面那条 S2-G 兜底
                # （「布局里有、产物里一个都没有的窗口 = 还有活干」）**永不执行**。
                # 盘面：一个 UNKNOWN 子窗 + 一个从没采过的末窗 ⟹ 整跑判终态，
                # 而收尾那次全路径 ANALYZE 又因 `missing_windows` 非空被跳过
                # ⟹ 一份 stage 结果都拿不到。这正是 S2-G 写来修的崩溃形状，
                # 换了个出口原样复发。
                # 点名父窗之后，退役判定才能去问「别处还有没有活干」。
                blocking_window=(int(_uu["parent_window"])
                                 if _uu.get("parent_window") is not None else None),
            )

        # 1b) **救援已经跑过、仍有窗口被踢出 ⟹ 预算耗尽，不是"结果差一点"。**
        #     rescue 循环只在两种情况退出：converged，或轮数用完。所以
        #     "有 rescue 目标 **且** 仍有 skipped_windows" = 轮数用完仍不够。
        #     此时**必须**落成 HALT_BUDGET + INSUFFICIENT_DATA，**禁止静默缺窗**
        #     然后把部分和端出去 —— 那正是老板划 ✗ 的结局。
        _sel = _pick(earliest, view["skipped_windows"], key=None)
        if _sel and view.get("production_rescue_targets"):
            return plan(
                "RUN_PRODUCTION",
                f"窗口 {view['skipped_windows']} 在 rescue 跑过之后**仍然**被踢出"
                f"协方差链（rescue 目标 {view['production_rescue_targets']}）⟹ "
                "轮数已用完而支撑仍不足。这是**预算耗尽**，不是终态失败：低支撑永远是"
                "「尚不可测」（INSUFFICIENT_DATA），不是 FAIL。"
                "⚠️ **在此之前产出的总和缺窗口，不得当作 ΔG 使用。**"
                "要继续必须显式增加预算（或按 §4 改走重标定）。",  # noqa: E501
                # 🔑🔑 [2026-09-14] **这里恒为路由，绝不发终态。**
                # 这四处（本处 / skipped+rescue / f_k 被驳回 / 探针建议重标定）
                # 形状完全一样：「新 Epoch 付不起验证额度 ⟹ 退而补生产帧」。
                # 而 `all_windows_budget_exhausted` 判的是
                # **所有窗口的 `warmup_steps_left` ≤ 0** —— 纯预热账本。
                # `RUN_PRODUCTION` 用的是**已冻结的 f_k**，一步验证预算都不花。
                # 先前在这里发 `GLOBAL_BUDGET_EXHAUSTED`（它在 `TERMINAL_EXITS` 里）
                # ⟹ 主循环在分发**之前**就 break，那个补帧动作**根本不执行**：
                # 拿 A 账本的余额，终止一个只花 B 账本的动作。
                # ⚠️ 控制器里目前**没有"生产补帧预算"这个量**
                # （`per_window_budget_remaining` 也是 `warmup_steps_left`），
                # 所以这里无法判定补帧可不可行 —— 唯一能判的地方是主循环的停滞
                # 保护（"补了一轮，盘上没变"），`GLOBAL_BUDGET_EXHAUSTED` 已改由
                # 它发出。造一个真正的双层预算是 PLAN 第 5 步 F，尚未实现。
                exit_="HALT_BUDGET",
                windows=_sel,
                blocked=blocked, earliest=earliest,
            )

        # 2) f_k 被**有统计功效地**驳回 ⟹ 终态。不许"再测一次"。
        refuted = _pick(earliest, [w for w in W if w["verdict"] == "STATISTICALLY_REJECTED"])
        if refuted:
            # 🔑 [审计 #20] 这里原来又抄了一遍「付不起就退而补生产帧」的分岔。
            # 那道闸现在统一在 `plan()` 里（只对换 Epoch 类动作跑），这一处是
            # **死代码**：`refuted` 里的窗口必然已经被上面那道无条件前置闸截走。
            # 直接发 `RECALIBRATE_FK`，付不付得起由 `plan()` 判并按同样的口径改发
            # `RUN_PRODUCTION` + `HALT_BUDGET`。
            return plan(
                "RECALIBRATE_FK",
                f"窗口 {refuted} 的 f_k 证据是 refuted（有统计功效的否决）⟹ 必须换 "
                "Epoch 重标定，**不能**再加验证预算。"
                # [2026-09-14 更正] 原来这里写「抛 IBSFrozenCalibrationValidationError "
                # 而全仓库没有任何 except 捕获它、会直接炸穿整个 run」——**已经不成立**：
                # `abfe_pipeline._run_stage2_autonomous` 有
                # `except _ie_exc.IBSFrozenCalibrationValidationError`，会封存候选
                # （`seal_refuted_candidate`）、落 `stage2_fk_refuted.json`、再 `continue`
                # 回到顶层重判。留着旧话会让人以为这条路是断的。
                "出口 `HALT_FK_REFUTED` 是**路由**：这个动作要执行，不是终止。",
                exit_="HALT_FK_REFUTED", windows=refuted,
            )

        # 3) 还在预热、且预算有余 ⟹ 继续（学习或验证，按 phase 分）。
        #    这是 UNMEASURED 的正确回应：加同类预算，**不换轴**。
        warming = [w for w in W if w["phase"] in ("WARMUP_LEARN", "WARMUP_VALIDATE")]
        with_budget = [w for w in warming if (w["warmup_steps_left"] or 0) > 0]

        # 3a) **批次上限打满、但全局预算还有钱** ⟹ 这不是 HALT_BUDGET。
        #     老板定的口径：`LOCAL_VALIDATION_CAP_EXHAUSTED`，
        #     evidence = INSUFFICIENT_DATA，**不是** F_K_REFUTED（没有任何证据
        #     驳回这份 f_k，只是这一轮没测出来）。
        #     处置（老板给的，win4 是第一个真实用例）：
        #       · **不扩大旧 warmup ladder**（15 批上限不动）
        #       · 进 `PROVISIONAL_PRODUCTION`，**不是可信 PASS**
        #       · **只给一个 +250k 诊断块**
        #       · 块后立即判：N_eff/g 明显增长并达到 10 → 继续；
        #         边际停滞/下降 或 far-end support 单调塌陷 → 关闭 Epoch、tail rewindow；
        #         top1% 灾难性集中 → 停止同分布加帧
        #     ⚠️ 实测块大小：win4 现在 N/g=4.19，+250k ≈ 1369 帧 ⟹ N/g ≈ 9.6，
        #     **恰好达不到 10**。这不是坏事 —— 它让这一块成为**决定性诊断**：
        #     g 若真已平台会稳步走到 ~9.6（再给一块即可）；g 若继续涨会明显低于 9.6
        #     （转 tail rewindow）。两种结局数值上分得很开。
        #     **别因为"差一点到 10"就自作主张给 +500k。**
        try:
            import ibs_engine as _ie_caps
            _batch_cap = int(_ie_caps.IBS_LOCAL_MBAR_GATE_MAX_BATCHES)
        except Exception:
            _batch_cap = None
        # [2026-09-17] 判据搬到 `local_validation_cap_hits()`（与 `plan()` 共用一份）——
        # 先前这里和 plan 各判各的，正是"同一个不变量两份实现"。
        _cap_idx_now = {int(h["window_idx"]) for h in self.local_validation_cap_hits(view)}
        _cap_hit = [w for w in with_budget if int(w["window_idx"]) in _cap_idx_now]
        # ── 验证**可达性**预检：在剩余预算内还有没有可能凑够去相关帧数 ──────
        # ⚠️ 边界很重要。2026-09-11 已经否决过「周期**内**按 n_eff 外推提前判死」
        # （见 ibs_engine 那段：g 只在 N ≫ τ 后才稳，区间内外推会把本来能测出来的
        # 窗口提前判死）。这里**不违反**那条：
        #   · 判定发生在**周期用尽之后**，决定要不要开**下一个**周期，不掐断本周期；
        #   · `g_L` 取多检查点最小值，不是外推；
        #   · 而且 g 还在随 N 涨这件事**加强**不可达的结论 —— 真 g ≥ 实测 g，
        #     需要的帧数只会更多，缺口只会更大。
        # ⚠️⚠️ **只改路由，不改 verdict**：证据仍是 UNMEASURED，绝不变成 REJECTED，
        # 也绝不仅凭高 g 去插 λ / 拆窗。
        _unreach = []
        for w in _cap_hit:
            gs = w.get("validation_g_checkpoints") or []
            T = w.get("validation_required_frames")
            ncap = w.get("validation_sample_count")
            left = w.get("warmup_steps_left")
            if not gs or not T or not ncap or left is None:
                continue
            r = _ie_reach(
                gs,
                required_decorrelated_frames=int(T),
                cycle_frame_cap=int(ncap),
                budget_remaining_steps=int(left),
                frames_already=int(ncap),
            )
            if r.get("verdict") == "UNREACHABLE":
                _unreach.append((w, r))
        _un_sel = _pick(earliest, [w for w, _ in _unreach])
        if _un_sel:
            _w, _r = next((w, r) for w, r in _unreach
                          if int(w["window_idx"]) in _un_sel)
            _wi = int(_w["window_idx"])
            _relearned = relearn_epoch_used(
                self.checkpoint_dir, int(view.get("path_version") or 0), _wi)
            _need = int(_r["projected_steps_needed"])
            _why = (
                f"窗口 {_un_sel} 的冻结验证在剩余预算内**算术上不可达**："
                f"g 检查点 {[round(x, 1) for x in _r['g_measurements']]} ⟹ "
                f"保守下界 g_L={_r['g_lower_bound']:.1f}，"
                f"而本周期阈值 gcrit_cycle={_r['gcrit_cycle']:.2f}、"
                f"整预算阈值 gcrit_budget={_r['gcrit_budget']:.2f}。"
                f"凑够 {int(_r['required_decorrelated_frames'])} 个去相关帧**从零算起**"
                f"需要约 {int(_r['projected_raw_frames_needed'])} 原始帧 = {_need} 步；"
                f"该窗口已攒 {_w.get('validation_sample_count')} 帧，"
                f"**还差约 {max(0, _need - int(_w.get('warmup_steps_spent') or 0))} 步**，"
                f"而剩余预算 {_w.get('warmup_steps_left')} 步。"
                # 🔑 [审计 #52，2026-09-14] **`projected_steps_needed` 是从零算起的
                # 总步数，不是缺口。** 先前把它与 `warmup_steps_left` 并排打印成
                # 「还需 X 步 / 剩余仅 Y 步」，读者会把 X 当缺口 ⟹ 报出去的差距被
                # 系统性夸大（已攒的那部分被重复计了一次）。
                # ⚠️ **判定本身不受影响**：判定用的是 `gcrit_budget`，它已经含
                # `frames_already`。这里只是把措辞改对，别顺手去改判据。
                "（判定用的是已含 `frames_already` 的 `gcrit_budget`，不是上面这个总数。）"
                "⚠️ 这是**预算**结论不是 f_k 结论：证据仍是 UNMEASURED，"
                "**不得**改写成 REJECTED，**不得**因此插 λ / 拆窗，"
                "**也不得**因为高 g 就接受一份未验证的 f_k。"
            )
            if not _relearned:
                return plan(
                    "RELEARN_FK_EPOCH",
                    _why + "处置：这个窗口还没用过那**唯一一次**替代候选 ⟹ "
                    "开一个全新 Epoch，从头 LEARN 一份 f_k（不是拿旧生产帧重解）。",
                    exit_="HALT_VALIDATION_BUDGET_UNREACHABLE", windows=_un_sel,
                )
            # 🔑 [2026-09-14 梳理] 这里原来写的是 `action="DONE"` 配
            # `exit="NO_FEASIBLE_ACTION"` —— 自相矛盾：`plan()` 按 `action == "DONE"`
            # 算 `execution_status=COMPLETE`，于是一个"无路可走"的结局被记成
            # "执行完毕"。与 09-13 修过的那处是同一个毛病，这一处漏了。
            return plan(
                "NO_ACTION",
                _why + "处置：替代候选**已经用过一次**（一个窗口只给一次，"
                "否则就是反复试到偶然通过）⟹ NO_FEASIBLE_ACTION，"
                "留完整诊断终止。**不是 DONE** —— 结果没有达标。",
                exit_="NO_FEASIBLE_ACTION", windows=_un_sel,
                # [2026-09-17] 这个窗口的唯一一次替代候选已用掉；别处照常
                halt_scope="TARGET_LOCAL",
            )

        # 🔑🔑 [2026-09-15] **「只给一个 +250k 诊断块」在这里兑现。**
        # 已经用掉那一块的窗口（`bias_status == "provisional_production"`）不再
        # 发这个动作 —— 它的处置改由块后的 N_eff/g 判据接手（达到 10 则继续；
        # 边际停滞/下降或 far-end support 塌陷 ⟹ 关 Epoch 走 tail rewindow；
        # top1% 灾难性集中 ⟹ 停止同分布加帧），那些分支在下面本来就有。
        # ⚠️ 再发一次 = 「反复试到偶然通过」，正是 4818 那段明令禁止的。
        _cap_hit = [
            w for w in _cap_hit
            if str(w.get("bias_status") or "") != "provisional_production"
        ]
        _cap_sel = _pick(earliest, _cap_hit)
        if _cap_sel:
            _cap_hit = [w for w in _cap_hit if int(w["window_idx"]) in _cap_sel]
            _idx = _cap_sel
            return plan(
                "PROVISIONAL_PRODUCTION",
                f"窗口 {_idx} 的单周期验证**批次上限**已打满"
                f"（{[w.get('frozen_validation_batches') for w in _cap_hit]}/{_batch_cap} 批），"
                f"而全局预算**还有钱**（剩 {[w.get('warmup_steps_left') for w in _cap_hit]} 步）"
                " ⟹ 这是 `LOCAL_VALIDATION_CAP_EXHAUSTED`，**不是 HALT_BUDGET、"
                "更不是 F_K_REFUTED**（没有任何证据驳回这份 f_k，只是这一轮没测出来）。"
                "处置：**不扩大 15 批上限**；进 PROVISIONAL_PRODUCTION（**不是可信 PASS**）"
                "—— 执行器显式授权引擎用这份**未验证**的冻结 f_k 采一块，"
                "引擎把 `bias_status` 写成 `provisional_production`、证据保持 "
                "`indeterminate`、warmup_failure.json 留作证据（不删）；"
                "**只给一个 +250k 诊断块**，块后立即用 N_eff/g 的**边际增长**判 —— "
                "达到 10 则继续；边际停滞/下降或 far-end support 单调塌陷则关闭 Epoch 走 "
                "tail rewindow；top1% 灾难性集中则停止同分布加帧。"
                "⚠️ 一块 +250k 预计到 N/g≈9.6、**恰好达不到 10**，这是有意的诊断设计，"
                "**别因此改成 +500k**。"
                "⚠️⚠️ **前提：当前布局不会被重分。** 若尾段即将 tail-only 重分，"
                "这一块就是在**即将作废的布局上取证** —— 应**先重分再取证**，"
                "把预算留给新尾段。",
                exit_="HALT_LOCAL_VALIDATION_CAP", windows=_idx,
            )

        _wb_sel = _pick(earliest, with_budget)
        if _wb_sel and all(_is_noop(view, _by_idx_for_noop, _by_uid_for_noop, _noop_led, "CONTINUE_WARMUP", i) for i in _wb_sel):
            # 🔑🔑 [2026-09-14 真机] **续预热对"生产缓存已完整"的窗口是结构性 no-op。**
            # 窗口级 resume 缓存门在**走到预热之前**就 `continue` 掉整个窗口
            # （"已有有效缓存能量 … resume 模式下跳过重新采样"），所以
            # `CONTINUE_WARMUP` 一步也跑不了。真机 rep2：win4 连发 **40 次**、
            # `production_steps` 与 `warmup_steps_left` 一动不动，
            # `repeat_count` 在 1→2→3 之间循环（停滞保护每次降级到探针、探针又
            # 改了一点盘面把计数清零），永远走不到退出。
            # 它真正缺的是**生产帧**（端点 σ），那才是对症动作。
            return plan(
                "RUN_PRODUCTION",
                f"窗口 {_wb_sel} 仍在预热阶段且 warmup 预算有余，但**续预热在当前盘面"
                "上已被证明什么也没做**（窗口级 resume 缓存在走到预热之前就跳过了整个"
                "窗口）⟹ 换对症动作：补生产帧。"
                "语义仍是「尚不可测」（INSUFFICIENT_DATA ≠ FAIL）。",
                windows=list(_wb_sel), blocked=blocked, earliest=earliest,
            )
        if _wb_sel:
            with_budget = [w for w in with_budget if int(w["window_idx"]) in _wb_sel]
            idxs = _wb_sel
            return plan(
                "CONTINUE_WARMUP",
                f"窗口 {idxs} 仍在 "
                f"{sorted({w['phase'] for w in with_budget})}，且 warmup 预算有余"
                f"（剩 {[w['warmup_steps_left'] for w in with_budget]} 步）。"
                "证据是 INSUFFICIENT_DATA（没测出来），不是 FAIL —— 加同类预算，不换轴。",
                windows=idxs,
                blocked=blocked, earliest=earliest,
            )

        # 4) 预热预算耗尽、仍未收敛 ⟹ 需要归因。
        # 🔑 [审计 #21，2026-09-14 复核] **这一段先前是死代码，现在可达了。**
        # 原因是 #20：换 Epoch 的预算预检当时是一道**无条件**前置闸，`stuck` 里的
        # 窗口必然先被它（或 1e）截走 ⟹ `HALT_NO_ATTRIBUTION` 在生产里从不执行。
        # #20 把那道闸挪进 `plan()`、只对换 Epoch 类动作跑之后，这条路打开了：
        # 1e 只处理「预热预算耗尽 **且** f_k 未判定（`f_k_evidence_status != verified`）
        # **且** 确实进不去」的窗口；「预算耗尽但 f_k 已 verified、仍卡在 WARMUP_*」
        # 的窗口落到这里做归因 —— 两条分支的前提互补、不重叠，不需要再改 1e。
        # 归因要的三个量
        #    （逐态 g 剖面 / mixture coverage / 逐边 δ）本控制器**不算数**，
        #    只在 stage 结果里才有。没有就如实说"归因不出来"。
        stuck = [w for w in warming if not (w["warmup_steps_left"] or 0) > 0]
        _st_sel = _pick(earliest, stuck)
        if _st_sel:
            idxs = _st_sel
            if not view["has_stage_result"]:
                return plan(
                    "CONTINUE_WARMUP",
                    f"窗口 {idxs} 的 warmup 预算已耗尽且未收敛，但 stage 结果还没落盘 ⟹ "
                    "归因需要的逐态 g 剖面 / mixture 覆盖度 / 逐边 δ 一个都读不到。"
                    "**归因不出来是合法结局**，不许为了凑一个动作而编理由。"
                    "要继续必须显式升档 warmup 预算，或先把 stage 分析跑完。",
                    # [老板] 只有三种允许终止 ⟹ "归因不出来"本身**不是**终止条件。
                    # 全局还有预算、或还有可行动作时，它是**路由信号**：主循环应该
                    # 去跑最小诊断动作（tail-only 重分 / +250k 诊断块）。
                    # 真的无路可走才 NO_FEASIBLE_ACTION。
                    exit_=(
                        "HALT_NO_ATTRIBUTION"
                        if (not view.get("all_windows_budget_exhausted")
                            or any(v is None for v in feas.values()))
                        else "NO_FEASIBLE_ACTION"
                    ),
                    windows=idxs,
                    missing=["stage2_*.json（逐窗 overlap/ESS 诊断）",
                             "逐态 statistical_inefficiency 剖面",
                             "逐边热力学长度 δ"],
                    # [2026-09-17] 与上面 exit_ **同一个**条件：只有全窗预算耗尽且无可行结构动作才是终态（stage 级）；否则是路由、没有 scope
                    halt_scope=("STAGE_GLOBAL" if (view.get("all_windows_budget_exhausted") and not any(v is None for v in feas.values())) else None),
                )
            # 重解在当前盘面已被证明是 no-op ⟹ 直接走布局动作，别再空转一轮。
            # 🔑🔑 [审计 #14，2026-09-14] **三元式与它上面的注释正好相反。**
            # 注释（也就是 PLAN §3ter.4 的优先级）是：**先重标定 f_k，仍压不住才补 λ**。
            # 代码写的却是「插 λ 可行就插、不可行才重标定」—— 顺序整个反过来。
            # 而且 `_recal_is_noop` 为真时 `or` 会**短路**发一个 `INSERT_LAMBDA`，
            # 哪怕插 λ 此刻不可行；同一条理由文本还在打印「补 λ 当前不可行」，
            # exit 词也配着 λ 预算耗尽 —— 动作、理由、出口三样互相矛盾。
            # 按注释写的优先级重写：
            #   默认 RECALIBRATE_FK；只有「重标定在当前盘面已被记为 no-op」**且**
            #   「插 λ 可行」才改发 INSERT_LAMBDA；两者都不行就如实停下。
            # `HALT_LAMBDA_BUDGET_INSUFFICIENT` 只在**真的因为 λ 预算耗尽**而无路可走
            # 时才用（λ 总数不够是输入问题，交人工改输入）。
            _recal_is_noop = all(_is_noop(view, _by_idx_for_noop, _by_uid_for_noop, _noop_led, "RECALIBRATE_FK", i) for i in idxs)
            _ins_ok = feas.get("insert_lambda") is None
            _why4 = (
                f"窗口 {idxs} 的 warmup 预算已耗尽且未收敛。按 PLAN §3ter.4，窗口级"
                "缺陷的对症动作依次是：**重标定 f_k → 仍压不住才补 λ 缩跨度**（model B）。"
            )
            if not _recal_is_noop:
                return plan(
                    "RECALIBRATE_FK", _why4 + "重标定尚未在本盘面上被证伪 ⟹ 先重标定。",
                    windows=idxs, blocked=blocked, earliest=earliest,
                )
            if _ins_ok:
                return plan(
                    "INSERT_LAMBDA",
                    _why4 + "重标定在**当前盘面**上已被执行器记账为 no-op"
                    "（跑过一次、盘面逐项未变）⟹ 按序进到下一档：补 λ 缩跨度。",
                    windows=idxs, blocked=blocked, earliest=earliest,
                )
            return plan(
                "NO_ACTION",
                _why4 + "重标定在当前盘面上已被记为 no-op，"
                f"而补 λ 也不可行（{feas.get('insert_lambda')}）⟹ 如实停下。"
                # 归因（λ 预算耗尽 vs 布局顶到上限）写进理由文本，**不进 exit 词**：
                # `NO_ACTION` 必须配终态出口，否则主循环不 break、执行器会收到一个
                # 它不认识的动作（`test_stage2_controller_vocabulary` 钉的就是这条）。
                # 先前这里配 `HALT_LAMBDA_BUDGET_INSUFFICIENT`（非终态）正是那个毛病。
                + ("归因：λ 总数不够是**输入问题**，应由人工改输入 λ 表。"
                   if "预算已用尽" in str(feas.get("insert_lambda") or "")
                   else "归因：布局本身顶到了可拆上限，不是预算问题。"),
                # 两个出口都在 `TERMINAL_EXITS` 里（`NO_ACTION` 必须配终态出口，
                # 否则主循环不 break、执行器会收到一个它不认识的动作）。
                # 只有**真的因为 λ 预算耗尽**才用那个更具体的词。
                exit_=("HALT_LAMBDA_BUDGET_INSUFFICIENT"
                       if "预算已用尽" in str(feas.get("insert_lambda") or "")
                       else "NO_FEASIBLE_ACTION"),
                windows=idxs, blocked=blocked, earliest=earliest,
                # [2026-09-17] 与 exit_ 同源：λ 总数不够是**输入问题**（stage 级，换窗也没用）；布局顶到可拆上限则只是这个窗口没路走
                halt_scope=("STAGE_GLOBAL" if "预算已用尽" in str(feas.get("insert_lambda") or "") else "TARGET_LOCAL"),
            )

        # 5) 生产帧没攒够 ⟹ 接着跑（同一个 f_k、接着原段，不改任何结构）。
        short = [
            w for w in W
            # ⚠️ **「未知不是零」这条通则在这一行是有意的例外，别按通则"修"它。**
            # 别处压扁 `None → 0` 都会造出错误决策（虚报余量 / 虚报耗尽 / 把未知
            # 固化成事实落盘）。**这里不会**：未知与 0 的**处置完全相同** ——
            # 两者都该去跑生产。而改成「未知就不判它短」会压制掉「真的一步没跑过」
            # 的窗口的正常生产（那正是 `production_steps=None` 最常见的成因：
            # 还没有 convergence 产物）。
            if w["production_steps_target"] and (w["production_steps"] or 0)
            < int(w["production_steps_target"])
        ]
        # 🔑 [2026-09-17] `earliest is None`（全窗自检合格）时 `_pick()` 恒返空，
        # 而"把已承诺的步数跑完"这件事**不重锚任何上游**、也不改布局 ⟹ 它是
        # `_pick` 的合法例外。没有这条例外，上面那道"不许判终态"的闸会让控制器
        # 卡在兜底上空转，而不是去把帧补上。
        _sh_sel = _pick(earliest, short) or (
            [int(w["window_idx"]) for w in short] if earliest is None else [])
        # 🔑🔑 [审计 #22，2026-09-14] **「生产帧没攒够」本来就不是 PROBLEM 态，
        # 所以它必须有自己的取窗口逻辑，不能借 `earliest`。**
        #
        # `earliest is None` 的意思是「没有任何窗口处于 PROBLEM 态」；而生产帧没
        # 攒够的窗口是 `UNKNOWN`（还没有自检产物）或 `ELIGIBLE`。先前这一条完全
        # 靠 `_pick(short)`，于是**全窗 UNKNOWN 的盘面**（老 run / 刚起步的 run，
        # 最常见）上它一次都触发不了，`decide()` 只发得出 6a 的 `ANALYZE` ——
        # 而窗口根本还没跑完，分析是空转，靠停滞保护退出。
        # 定点修：`earliest is None` 时按窗口序取 `short` 里最早的那一个。
        # （**不放宽 `_pick`** —— 那会一次性拆掉另外 10 条分支的隐式前提。）
        # ⚠️ 已被 rewindow 取代的父窗不参与（CTL-03：它整个不在求解覆盖里）。
        if not _sh_sel and earliest is None and short:
            _cand = sorted(int(w["window_idx"]) for w in short
                           if int(w["window_idx"]) not in _replaced_parents)
            _sh_sel = _cand[:1]
        if _sh_sel:
            short = [w for w in short if int(w["window_idx"]) in _sh_sel]
            idxs = _sh_sel
            return plan(
                "RUN_PRODUCTION",
                f"窗口 {idxs} 的生产步数未达目标"
                f"（{[(w['production_steps'], w['production_steps_target']) for w in short]}）。"
                "同一个冻结 f_k、接着原段继续，不改结构、不丢已有帧。",
                windows=idxs,
                blocked=blocked, earliest=earliest,
            )

        # 🗑️ [2026-09-17] **这里曾加过一条「第一遍跑完先做 ANALYZE」的分支，已回退。**
        #
        # 动机是真的（见下）：`_read_stage_result()` 在整个采样阶段恒为 `None`，
        # 因为 stage 结果**只在 `ANALYZE` 执行时**落盘，而 `ANALYZE`（分支 8）排在
        # 5/5a/5b/6/6a/6b 之后，只要还有窗口缺帧就永远轮不到。后果是三条判据的
        # 输入全空、而且全部**静默放行**：
        #   · `solver_n_frames_decorrelated` 全窗 `None` ⟹ 补帧块账判据量是
        #     `[None, None, None]` ⟹ `marginal_gain_stalled()` 恒 `NOT_ENOUGH_POINTS`
        #     ⟹ **边际增益刹车永不触发**（真机 cyclod_ligand2/rep1 w5 连批 3 块，
        #     三次判据量都是 None；cmet_ligand1/rep2 w0 连批 3 块同样）；
        #   · `cum_fk_verdict` 全 `None` ⟹ 累计 f_k 偏差链（5a-1/5a-2）永不进入；
        #   · solver 侧 `skipped_windows` 恒空 ⟹ 跳窗分支（1d / 6）永不触发。
        #
        # **但把 ANALYZE 提前到这里是错的杠杆。** 实测两次：
        #   · 条件「无 stage 结果 + 全窗已产出 + 第一遍跑完」⟹ 几乎每个盘面都满足
        #     （stage 结果本来就要等第一次 ANALYZE 才有）⟹ 挡住退役/终态整族分支，
        #     全量 **25 红**；
        #   · 收窄成「某窗已吃到第 2 块」⟹ 仍然抢在刹车/上限/退役那一族测试前面，
        #     全量 **11 红** —— 因为多块盘面正是生产的主要工作区。
        # 每收窄一次打红另一批 = 这条分支在热路径上，动它就是动全局语义。
        #
        # ⟹ 正确的修法不是插一条分支，而是 `AUDIT-S2-02` 那次重构（把判断按证据
        # 类型分派），用户已明确押后。**在那之前这个缺口是已知的、被记录的**：
        # 补帧只剩块数硬上限与射程闸两道闸，边际增益那道在采样阶段不带电。
        # 别再试图靠挪 `ANALYZE` 的位置来补它。

        # 5a) **f_k 明确不符 ⟹ 先重标定，不要先加帧。**
        #     PLAN §4：「I 与 II 之间**不排序**：按判别结果直接选类型，不做『先便宜
        #     后贵』的阶梯（证据表明 f_k 明显不符时先加帧是浪费）」。
        #     加帧治不了偏斜是仓库自己的结论，本轮实测复现（win0 250k→1M：
        #     N_decorrelated 40→182 变好，但 ESS_ratio 0.037→0.0264、
        #     top1% 0.545→0.604 **变差**）。
        #     ⚠️ 这一条**故意排在 5b/6（加帧）之前** —— 生产代码现在的顺序是
        #     "rescue 加帧两轮之后才重标定"，所以影子在这里会与生产分歧，
        #     那正是要被记录下来的对账数据。
        # 5a-0) **SUSPECTED_DERAILMENT（单块）⟹ 只做非变异的候选计算，不切换。**
        #   老板定案：单块只产生"疑似"，触发廉价诊断；**连续两块**才允许关闭 Epoch。
        #   ⚠️ 而且在 **held-out 反事实验收**接好之前，单块**一律不得**升级成真重标定 ——
        #   否则就是"因一个震荡低块贸然切换到更差的 f_k"（win1 已经吃过这个亏）。
        #   held-out 验收目前**未实现**，所以这一档现在只能到"算候选 + 报告"为止。
        # 🔑🔑 [2026-09-17] **`earliest is None` 时也要发 —— 这是 `_pick` 的第二个
        # 合法例外（第一个是分支 5 的"补足已承诺步数"）。**
        #
        # 脱轨探针的目标窗口自检往往**仍然 ELIGIBLE**（脱轨是单块边际 N_eff 塌，
        # 不是整窗不合格）⟹ 路由态不是 PROBLEM ⟹ 它**永远不会**成为 `earliest`
        # ⟹ `_pick` 恒返空 ⟹ 这条分支一次都不触发。注意这不是"押后"：
        # `earliest` 只从 PROBLEM 里选，退役只会把窗口**移出**排序、永远不会把一个
        # ELIGIBLE 的加进来 —— 所以它是**永久**的丢失。
        #
        # 开这个例外的理由与分支 5 同一条：`PROBE_CANDIDATE_FK` 是 `probe_only=True`，
        # **离线算候选、零采样**（见 `_PRODUCTION_CHARGED` 那段：它是唯一真正的
        # 非变异探针），不重锚、不改布局、不花 GPU。零成本的取证没有理由被
        # 「先修最早的」挡住。
        _derailed = [w for w in W
                     if w.get("derailment_status") == "SUSPECTED_DERAILMENT"]
        _suspected = _pick(earliest, _derailed) or (
            [int(w["window_idx"]) for w in _derailed] if earliest is None else [])
        _confirmed = [
            w["window_idx"] for w in W
            if w.get("derailment_status") == "CONFIRMED_DERAILMENT"
        ]
        # 🔑 [2026-09-12] **探过一次就得往下走。**
        # 探针是**非变异**的，按定义不改盘 ⟹ "盘上状态没变"对它永远成立 ⟹
        # 主循环的停滞保护必然把它判成"推不动"，连探三次后 NO_FEASIBLE_ACTION
        # 退出（真机 16:46 就是这么把 win5 放弃的）。
        # 它已经给出结论了（`stage2_fk_recalibration_probe.json`），
        # 所以探过就得往下走，让后面的分支按那个结论选动作。
        #
        # 🔑🔑 [2026-09-15] **但"探过"必须是「在**这个盘面**上探过」，不是"探过一次"。**
        # 先前这里用 `_probed`（探针产物的 `windows` 列表）判，而那份列表含**全部**
        # 窗口（探针一次就把所有窗口都写进去），于是**第一次探针之后全仓永久不再重探**。
        # 致命之处在于探针的结论是**步数的函数**（固定节奏重锚：已采步数 ≥ cadence），
        # 每补一块帧都可能翻面 —— 只在第一轮算一次等于把这条判据钉死在初始步数上。
        # 真机 4W53 vanishing：第 5 轮探针在 250k 步判"未到节奏"，此后 win0 补到
        # 1.25M 步（早已超过 500k 的节奏）却再也没被重探，最后 NO_FEASIBLE_ACTION。
        # 换成 `_is_noop`：它的指纹**带生产步数与路径版本**，补过帧/换过段/插过 λ
        # 之后记录自动失效、动作重新可选，正好是这里要的语义；而"同一盘面上重复发"
        # 仍然被挡住，防停滞的原意一个字没丢。
        if _suspected and all(
            _is_noop(view, _by_idx_for_noop, _by_uid_for_noop, _noop_led, "PROBE_CANDIDATE_FK", int(i)) for i in _suspected
        ):
            _suspected = []
        if _suspected and not _confirmed:
            return plan(
                "PROBE_CANDIDATE_FK",
                f"窗口 {_suspected} 单块边际 N_eff 低于前半程中位数 1/4 ⟹ "
                "**SUSPECTED_DERAILMENT**：启动廉价的**非变异**诊断（离线算候选 f_k 并评估），"
                "**不关闭 Epoch、不切换 f_k**。只有连续两块才允许进入动作决策。"
                "⚠️ 并且 held-out 反事实验收（候选必须在未参与拟合的 block 上改善"
                "最差态 N_eff/g）**尚未实现**，所以这一档现在到「算候选 + 报告」为止 —— "
                "在它接通之前，单块预警不得升级成真重标定。",
                windows=_suspected,
                missing=["held-out 反事实验收（候选是否真的改善最差态 N_eff/g）"],
            )

        _probe = view.get("fk_probe") or {}
        # 被求解器跳掉的窗口在上面 1d 就已经被路由走了（重解类动作对它们是 no-op），
        # 这里不会再看到它们。
        _recal = [
            i for i in _pick(earliest, 
                [int(x) for x in
                 (_probe.get("recalibration_recommended_windows") or [])],
                key=None,
            )
            # 🔑 [2026-09-14 真机] 执行器已经在**这个盘面**上试过一次、什么也没做
            # （"重标定未产生新段：没有窗口超过 0.5 kJ/mol 相邻位移阈值"）⟹ 不再发。
            # 探针按**节奏**推荐、执行器按**位移**放行，而且两者读的还不是同一份
            # 数据（执行器从"有这个窗口数据的最新段"重解，真机那次是只含 win0 的
            # `vanishing_9`；探针读的是全量）。同一个决定两套判据 ⟹ 死循环。
            if not _is_noop(view, _by_idx_for_noop, _by_uid_for_noop, _noop_led, "RECALIBRATE_FK", i)
        ]
        if _recal:
            # 🔑 [审计 #20] 同分支 2：这里原来又抄了一遍「付不起 ⟹ 退而补生产帧」。
            # 那道闸已统一到 `plan()`（只对换 Epoch 类动作跑，口径与理由文本不变），
            # 这一处是死代码 —— 无条件前置闸必然先把这些窗口截走。
            return plan(
                "RECALIBRATE_FK",
                f"f_k 探针判定窗口 {_recal} 的相邻位移超过 "
                f"{_probe.get('min_adjacent_shift_kJ_mol')} kJ/mol ⟹ **先重标定 f_k，"
                "不要先加帧**（PLAN §4：f_k 明显不符时先加帧是浪费；加帧只会往同一个"
                "偏斜分布里加更多帧，绝对样本数涨、ESS 比值不动）。"
                "⚠️ 生产代码当前把重标定挂在 rescue 两轮**之后**"
                "（`stage2_recalibrate_f_k_on_rescue`），所以此处与生产分歧 —— "
                "这是影子要记录的对账点，不是控制器在开车。",
                windows=_recal,
                blocked=blocked, earliest=earliest,
            )

        # 5a-1) **支撑不足时，先判累计 f_k 偏差，不要先加帧。**
        #   老板给的路径第一步就是「win3 支撑不足 → **判断累计 f_k 偏差** → 生成候选」。
        #   而"已经脱轨就禁止继续加帧"也是同一条：加帧治不了 f_k 偏斜
        #   （实测 250k→1M 让 top1% 从 0.545 涨到 0.762）。
        #   证据缺失时的正确动作是**把证据做出来**（跑分析落 cumulative 残差），
        #   不是先烧 250k 去加帧再说。
        # ⚠️ [2026-09-14 真机] **「零额外采样就能算」有前提：这批帧求解器得能用。**
        #   窗口去相关后帧数低于求解器下限（`sufficient=False` / 被踢出协方差链）时，
        #   分析**根本跑不到**这个窗口 —— 它在生成任何诊断之前就被 `continue` 掉，
        #   于是 `cumulative_fk_residual_production` 永远不会出现。原来这里不看这一条，
        #   结果是对这种窗口无限返回 ANALYZE：真机 win0（9 帧 < 10）连发 4 次
        #   ANALYZE、盘上一个字节没变，靠停滞保护降级到 PROBE_REANCHOR_EPOCH
        #   （探针判"f_k 不是瓶颈"什么也没做），最后 NO_FEASIBLE_ACTION 退出——
        #   而它真正需要的动作（5b/6 的补采）就在下面几行，永远轮不到。
        #   这类窗口的正确动作是**先把帧补到求解器能用**，再谈 f_k 偏差证据。
        _unsolvable = set(int(x) for x in (view.get("skipped_windows") or []))
        _need_cum = [
            w["window_idx"] for w in W
            if earliest is not None and int(w["window_idx"]) == earliest
            and w.get("self_verdict") in ("HARD_INSUFFICIENT", "INSUFFICIENT_DATA")
            # `UNMEASURED` 是**已产出的证据**（"这个窗口结构上测不了"），不是"还没算"。
            # 拿它去跑 ANALYZE 是空转 —— 见 `cum_fk_verdict` 赋值处的长注释。
            and w.get("cum_fk_verdict") is None
            and w.get("self_sufficient") is not False
            and int(w["window_idx"]) not in _unsolvable
        ]
        if _need_cum:
            return plan(
                "ANALYZE",
                f"窗口 {_need_cum} 支撑不足，但**累计 f_k 偏差证据还不存在**"
                "（`cumulative_fk_residual_production` 缺失）⟹ 先把它算出来，"
                "**不要先加帧**：加帧治不了 f_k 偏斜（实测 250k→1M 让 top1% 从 "
                "0.545 涨到 0.762、ESS 比值反而变差），而累计偏差判据在**现有帧上**"
                "就能算，零额外采样。",
                windows=_need_cum, blocked=blocked, earliest=earliest,
                missing=["cumulative_fk_residual_production（生产侧累计 f_k 残差）"],
            )

        # 5a-2) **累计 f_k 偏差已判 ⟹ 按 held-out 的三种结局分岔。**
        #   老板给的链：判累计偏差 → 生成候选 → **held-out 可测则验收** →
        #   **不可测则自动开 PROBE_REANCHOR_EPOCH** → 支撑改善则继续 →
        #   **不改善则自动 tail-repartition / 插 λ**。
        #   三种结局的处置**互不相同**，混起来就回到"用支撑换占据"那个老错：
        #     ACCEPT     → 换 Epoch（候选确实在未参与拟合的块上改善了最差态）
        #     UNMEASURED → **不是拒绝**，是"判不了" ⟹ 开有界的 PROBE_REANCHOR_EPOCH
        #                  拿独立证据；**绝不回去给旧 f_k 加帧**
        #     REJECT     → 候选救不了 ⟹ 转布局动作（拆末窗 / 插 λ）
        _cum_bad = [
            w for w in W
            if earliest is not None and int(w["window_idx"]) == earliest
            and w.get("cum_fk_verdict") in ("FAIL_CUMULATIVE_FK", "UNMEASURED")
        ]
        if _cum_bad:
            _w0 = _cum_bad[0]
            _hv = _w0.get("heldout_verdict")
            if _hv == "ACCEPT":
                return plan(
                    "RECALIBRATE_FK",
                    f"窗口 {earliest} 累计 f_k 偏差 span={_w0.get('cum_fk_span')}"
                    f"（{_w0.get('cum_fk_verdict')}），且候选在 **held-out 连续时间块**上"
                    f"把最差态 N_eff/g 从 {_w0.get('heldout_worst_before')} 提到 "
                    f"{_w0.get('heldout_worst_after')}、没伤到健康态 ⟹ **换 Epoch**。",
                    windows=[earliest], blocked=blocked, earliest=earliest,
                )
            if _hv == "REJECT":
                # 🔑 [CTL-07，2026-09-14] **拆末窗只对末窗有意义。**
                # 这里先前只问「拆末窗在结构上可不可行」，不问「失败的是不是末窗」——
                # 与 9b 已经修过的是同一个毛病，这一处漏了：中间窗的 f_k 被 held-out
                # 驳回，却换来一次**与它无关**的尾段重分（烧 GPU、改布局、作废下游
                # 证据，元凶窗口一个字节没动）。判据与 9b 同一份。
                _tail_i = max((int(w["window_idx"]) for w in W), default=None)
                _is_tail = (_tail_i is not None and int(earliest) == int(_tail_i))
                _feas_split = feas.get("split_tail_window") is None and _is_tail
                # 🔑🔑 [审计 #13，2026-09-14] **这里原来完全不查插 λ 的可行性。**
                # 它是全仓最后一处「发布局动作却不问可不可行」的地方（其余每一处
                # ——`_no_gain` 分支、9b —— 都先问过 `feas.get("insert_lambda")`）。
                # 插点预算耗尽 / 末窗顶到可拆上限时，它发出的是一个**在构造上不可能
                # 成功**的动作：执行器要么抛错炸穿流水线，要么绕过 `max_path_insertions`
                # 无限插点（真机末窗被从 K=6 顶到 K=12 就是这个形状）。
                # 归因（f_k 救不了它、要缩跨度）是**对的**，所以不可行时**不改归因**，
                # 只改动作：如实给 NO_ACTION + NO_FEASIBLE_ACTION，并带上 `feas` 的原因。
                _feas_ins = feas.get("insert_lambda") is None
                if not _feas_split and not _feas_ins:
                    _rw = _bounded_rewindow(
                        int(earliest),
                        f"窗口 {earliest} 累计 f_k 偏差 span={_w0.get('cum_fk_span')}，"
                        "但候选在 held-out 上**没有改善最差态**（或伤到了健康态）⟹ "
                        "**f_k 救不了它，对症动作是缩跨度**；而 λ 表上的两个缩跨度动作"
                        f"都不可行（拆末窗：{feas.get('split_tail_window')}"
                        f"{'（失败的不是末窗）' if not _is_tail else ''}；"
                        f"插 λ：{feas.get('insert_lambda')}）",
                    )
                    if _rw is not None:
                        return _rw
                    return plan(
                        "NO_ACTION",
                        f"窗口 {earliest} 累计 f_k 偏差 span={_w0.get('cum_fk_span')}，"
                        "但候选在 held-out 上**没有改善最差态**（或伤到了健康态）⟹ "
                        "**f_k 救不了它，对症动作是缩跨度**；而三个缩跨度动作都不可行"
                        f"（拆末窗：{feas.get('split_tail_window')}"
                        f"{'（失败的不是末窗）' if not _is_tail else ''}；"
                        f"插 λ：{feas.get('insert_lambda')}；"
                        f"有界重窗：{self.rewindow_feasible(view, earliest)}）⟹ 如实停下。"
                        "要继续需要显式加插点预算或改输入 λ 表。**不是 DONE** —— "
                        "结果没有达标。",
                        exit_="NO_FEASIBLE_ACTION", windows=[earliest],
                        blocked=blocked, earliest=earliest,
                        # [2026-09-17] 这个窗口三个缩跨度动作都不可行；别处照常
                        halt_scope="TARGET_LOCAL",
                    )
                return plan(
                    "SPLIT_TAIL_WINDOW" if _feas_split else "INSERT_LAMBDA",
                    f"窗口 {earliest} 累计 f_k 偏差 span={_w0.get('cum_fk_span')}，"
                    "但候选在 held-out 上**没有改善最差态**（或伤到了健康态）⟹ "
                    "**f_k 救不了它，转布局动作**。"
                    + ("拆末窗。" if _feas_split
                       else (f"失败的不是末窗（末窗是 {_tail_i}）⟹ 在它自己身上插 λ 缩跨度。"
                             if not _is_tail
                             else f"拆窗不可行（{feas.get('split_tail_window')}）⟹ 插 λ 缩跨度。")),
                    windows=[earliest], blocked=blocked, earliest=earliest,
                )
            # None（还没算）或 UNMEASURED（算了但判不了）：都走有界探针 ——
            # 但探针**也是**"拿已有帧重解"那一族，执行器已经在这个盘面上试过且
            # 什么也没做时，再发一次仍然是 no-op（真机停滞保护降级过去、依旧空转）。
            if _is_noop(view, _by_idx_for_noop, _by_uid_for_noop, _noop_led, "PROBE_REANCHOR_EPOCH", earliest) or _is_noop(view, _by_idx_for_noop, _by_uid_for_noop, _noop_led, 
                    "RECALIBRATE_FK", earliest):
                return plan(
                    "RUN_PRODUCTION",
                    f"窗口 {earliest} 的 held-out 判不了，按序该开有界探针；"
                    "但执行器已经在**当前盘面**上试过重解/换 Epoch、什么也没做"
                    "（未产生新段）⟹ 再发一次仍是 no-op。先补一块帧换取新证据："
                    "盘面一变，那条路自动重新可选。",
                    windows=[earliest], blocked=blocked, earliest=earliest,
                )
            return plan(
                "PROBE_REANCHOR_EPOCH",
                f"窗口 {earliest} 累计 f_k 偏差 span={_w0.get('cum_fk_span')}"
                f"（{_w0.get('cum_fk_verdict')}），而 held-out 验收"
                + ("尚未进行" if _hv is None else "判不了（UNMEASURED）")
                + " ⟹ 开一个**有界的** PROBE_REANCHOR_EPOCH：候选 f_k + 独立 burn-in"
                " + 一个 +250k 块；新 Epoch 支撑改善才晋升，恶化则拒绝候选、走下一动作。"
                "⚠️ **绝不回去给旧 f_k 加帧** —— 加帧治不了偏斜，那只会把同一个偏斜"
                "分布采得更久。",
                windows=[earliest], blocked=blocked, earliest=earliest,
            )

        # ── 边际增长判据：同分布加帧已经被证伪就别再加 ──────────────────
        # cap 分支的原话：「达到 10 则继续；**边际停滞/下降**或 far-end support
        # 单调塌陷则关闭 Epoch」。这里只落**无歧义的那一半**：主验收量
        # `min N_eff/g` **没有上升**。不自造"停滞"阈值。
        # 真机 win4：3.44 → 2.29 → 2.01 → 1.62，g 15.1 → 168.8，
        # 每加一次帧/换一次 Epoch 都更糟，而控制器仍在返回 RUN_PRODUCTION。
        # 🔑 [审计 #47] 判据改用共享的 `marginal_gain_stalled()`：至少 3 个点、
        # 基准取**前面各点的中位数**（不是最偏高的 `h[0]`）、末点明显低于基准才算
        # 被证伪。理由见那三个模块级常量的长注释：`min N_eff/g` 不是随采样单调增的量
        # （分子 Kish ESS 小 N 偏高、分母 ĝ 在 N≫τ 前还在涨，两个偏差同向），
        # 旧判据 `max(h[1:]) <= h[0]` 因此对正常窗口误触发，而且与本文件
        # `_unreach` 那段自己否决「按 n_eff 外推提前判死」的论证直接矛盾。
        _hist = view.get("min_n_eff_over_g_history") or {}
        _no_gain = []
        _no_gain_diag: Dict[int, Any] = {}
        for w in W:
            h = [float(v) for _nm, v in (_hist.get(int(w["window_idx"])) or [])]
            _stall, _d = marginal_gain_stalled(h)
            if _stall:
                _no_gain.append(w)
                _no_gain_diag[int(w["window_idx"])] = _d
        _ng_sel = _pick(earliest, _no_gain)
        if _ng_sel:
            _h = _hist.get(int(_ng_sel[0])) or []
            # [审计 #47④] 理由里把三个量都带上，让人看得出是**分子没涨**还是
            # **分母 ĝ 在涨** —— `min N_eff/g` 是个比值，只报比值分不出这两种。
            _d = _no_gain_diag.get(int(_ng_sel[0])) or {}
            _why_ng = (
                f"窗口 {_ng_sel} 的主验收量 min N_eff/g **没有随采样上升**"
                f"（逐段 {[(nm, round(float(v), 2)) for nm, v in _h]}；"
                f"基准=前面各点的中位数 "
                f"{_d.get('baseline_median_of_earlier_points')}，末点 "
                f"{_d.get('last')}，比值 {_d.get('last_over_baseline')} < 刹车阈值 "
                f"{_d.get('stall_ratio')}；"
                f"同窗求解器去相关帧数 {_by_idx_for_noop.get(int(_ng_sel[0]), {}).get('solver_n_frames_decorrelated')}，"
                f"求解器侧 g="
                f"{((_by_idx_for_noop.get(int(_ng_sel[0]), {}).get('evidence_decorrelation') or {}).get('solver') or {}).get('statistical_inefficiency')}"
                f"）⟹ "
            )
            # 🔑🔑 [2026-09-14 第三轮复核] **这是唯一一处不查可行性就发 INSERT_LAMBDA 的分支。**
            # 其余每一处发布局动作的地方都先问 `feas.get("insert_lambda") is None`，
            # 只有这里直接发。插点预算耗尽 / 末窗顶到可拆上限时，它发出的是一个
            # **在构造上不可能成功**的动作 —— 执行器要么抛错，要么造出执行层非法的
            # 布局（真机末窗被顶到 K=10~12 就是这个形状）。
            # 归因（加帧已被证伪）是对的，所以不可行时不改归因、只改动作：
            # 如实给 NO_FEASIBLE_ACTION，别假装还有路走。
            if feas.get("insert_lambda") is not None:
                _rw = _bounded_rewindow(
                    int(_ng_sel[0]),
                    _why_ng
                    + "同分布加帧已被本窗口自己的数据证伪，**不得再加**；"
                    f"而插 λ 也不可行（{feas['insert_lambda']}）",
                )
                if _rw is not None:
                    return _rw
                return plan(
                    "NO_ACTION",
                    _why_ng
                    + "同分布加帧已被本窗口自己的数据证伪，**不得再加**；"
                    f"而对症的缩跨度动作也不可行（插 λ：{feas['insert_lambda']}；"
                    f"有界重窗：{self.rewindow_feasible(view, int(_ng_sel[0]))}）⟹ "
                    "如实停下。要继续需要显式加插点预算或改输入 λ 表。",
                    exit_="NO_FEASIBLE_ACTION", windows=_ng_sel,
                    blocked=blocked, earliest=earliest,
                    # [2026-09-17] 加帧被证伪 + 缩跨度不可行，局部
                    halt_scope="TARGET_LOCAL",
                )
            return plan(
                "INSERT_LAMBDA",
                _why_ng
                + "同分布加帧已被本窗口自己的数据证伪，**不得再加**。"
                "瓶颈不是帧数而是这个窗口的跨度：轨迹在不同时间块占据不同 λ 区域，"
                "连续采样再多也摊不平（窗口内不遍历）。"
                "对症动作是**缩小跨度**（model B 插 λ，溢出落末窗），"
                "不是更多帧、也不是再换一份 f_k —— 换 Epoch 同样在同一个跨度上采样。"
                f"⚠️ 这条只在**至少 {MARGINAL_GAIN_MIN_POINTS} 个可比点**时触发"
                "（两点分不出趋势与噪声）。",
                windows=_ng_sel,
            )

        # 5b) 窗口自检已经判出"帧数不够" ⟹ 补采。**这是 (6) 的提前版**：同一个量、
        #     同一个门槛，只是在该窗口刚跑完那一刻就知道了，不用等全部窗口跑完
        #     （实测浪费：5×250k=125 万步烧完才做第一次预算判断）。
        #     verdict 是 INSUFFICIENT_DATA（还没测够）⟹ 加预算，不是换 Epoch。
        # 🔑🔑 [2026-09-14] **物理窗口这一侧也要分"缺帧"和"偏斜"。**
        # `sufficient=False` 混了三项（帧数 / `N_eff/g` / top1%），先前一律补帧 ——
        # 与子窗那边（CTL-02）是同一个毛病，这一处漏了。归因口径复用同一份：
        #   · `top1pct_veto` 或 `HARD_INSUFFICIENT` ⟹ 权重塌缩，**加帧治不了**
        #   · 其余 ⟹ 样本量问题，补帧对症
        # 🔑 [审计 #46] 归因判定与子窗那一侧**共用同一份实现**
        # （`support_failure_is_skew`）。先前这里和 `_read_single_stage` 各写一遍，
        # 而且都只认 verdict、不看 `verdict_source` ⟹ `solver_eligibility`
        # （纯帧数不够，写侧被强制置成 `HARD_INSUFFICIENT`）被判成"加帧治不了"，
        # 于是长 τ 的解耦端窗口被送去插 λ —— 而插 λ 不缩短构象慢模态的 τ_int。


        _short_all = [w for w in W if w.get("self_sufficient") is False]
        _skew_sel = _pick(earliest, [w for w in _short_all if _is_skew(view, w)])
        if _skew_sel:
            _sw = next(w for w in _short_all if int(w["window_idx"]) in _skew_sel)
            # 🔑🔑🔑 [2026-09-18] **权重塌缩先换 f_k，再谈缩跨度。**
            #
            # `R = η·N/g`。走到这里说明 `N/g ≥ 门`（够不着的那档已被 5c 截走），
            # 所以缺口确实在 **η** —— 权重剖面。而能动 η 的动作有两族：
            #   · **换 f_k**（重学）：直接改采样混合，**不动布局**；
            #   · **缩跨度**（插 λ / 拆窗）：改重叠也能改 η，但 model B 的后置断言
            #     写死了 `new_ranges = ranges[:-1] + [(tail_lo, tail_hi + n)]`
            #     ⟹ **末窗必须吸收全部溢出** ⟹ 失败窗口的 η 是从末窗身上借来的。
            # 两个同类杠杆，一个无副作用、一个有 —— 没有理由先用有副作用那个。
            #
            # `top1pct_veto` 是权重塌缩的**直接证据**（单帧/少数帧支配整个权重和），
            # 它正是"f_k 没把占据压平"的读数，所以对它 f_k 是对症的那一个。
            # ⚠️ 用 `RELEARN_FK_EPOCH`（从头 LEARN）而不是 `RECALIBRATE_FK`
            # （拿已有生产帧重解）：塌缩窗口的生产帧**就是**那批塌缩的帧，拿它们
            # 重解等于用坏样本估新参考系。两者的语义差别是 `ACTIONS` 里写死的。
            # ⚠️ 一个窗口只给一次（`relearn_epoch_used`），用过就落回下面的缩跨度；
            # 付不付得起由 `plan()` 的换 Epoch 预检判，不在这里另写。
            _sw_src = str(_sw.get("self_verdict_source") or "")
            if _sw_src == "top1pct_veto" and not relearn_epoch_used(
                    self.checkpoint_dir, int(view.get("path_version") or 0),
                    int(_sw["window_idx"])):
                return plan(
                    "RELEARN_FK_EPOCH",
                    f"窗口 {_skew_sel} 的自检失败来源是 **top1pct_veto**（权重塌缩："
                    "少数帧支配整个权重和）—— 这是 **η**（权重剖面）坏了的直接证据，"
                    f"而 `N/g` 这一侧够得着门（否则已被上一档 `N_OVER_G_BOUND` 截走）。"
                    "能改 η 的有两族动作：**换 f_k** 与 **缩跨度**。"
                    "缩跨度（插 λ）在 model B 下 **必然把溢出推给末窗**"
                    "（后置断言：`new_ranges = ranges[:-1] + [(tail_lo, tail_hi+n)]`）"
                    "⟹ 它是拿末窗的 η 去补这个窗口的 η。换 f_k 没有这个代价，"
                    "而且 top1% 塌缩本来就是「f_k 没把占据压平」的读数 ⟹ **先换 f_k**。"
                    "用 `RELEARN_FK_EPOCH` 而不是 `RECALIBRATE_FK`：塌缩窗口的生产帧"
                    "**就是**那批塌缩的帧，拿它们重解等于用坏样本估新参考系。"
                    "一个窗口只给一次；用过、或付不起验证额度，就落回缩跨度。",
                    windows=list(_skew_sel), blocked=blocked, earliest=earliest,
                )
            _can_split_skew = feas.get("split_tail_window") is None and (
                int(_sw["window_idx"]) == max((int(x["window_idx"]) for x in W),
                                              default=-1))
            _can_ins_skew = feas.get("insert_lambda") is None
            if _can_split_skew or _can_ins_skew:
                return plan(
                    "SPLIT_TAIL_WINDOW" if _can_split_skew else "INSERT_LAMBDA",
                    f"窗口 {_skew_sel} 的自检是**支撑/偏斜类**失败"
                    f"（verdict={_sw.get('self_verdict')}，归因 "
                    f"{_sw.get('self_verdict_source')}）—— **不是样本量不足**。"
                    "同分布加帧治不了偏斜（§5.1 实测 250k→1M 让 top1% 从 0.545 涨到 "
                    "0.762、ESS 比值反而更差）⟹ 对症动作是**缩跨度**。",
                    windows=list(_skew_sel), blocked=blocked, earliest=earliest,
                )
            # 🔑🔑 [2026-09-16 真机 cmet_ligand2/rep1 w4] **停下之前，先看证据是不是
            # 根本没做出来过。**
            #
            # 老板给的链是「判累计 f_k 偏差 → 生成候选 → held-out 验收 → 换 Epoch /
            # 缩跨度」，落在 5a-1（ANALYZE）+ 5a-2。但 5a-1 的 guard 里有一条
            # `self_sufficient is not False` —— 那是**自检侧**的量，而 CTL-11 已裁定
            # 逐窗自检「只看单段帧、对多段窗口系统性偏悲观，**它不是权威**」。
            # 于是一个偏斜类窗口必然 `sufficient=False` ⟹ 拿不到 ANALYZE ⟹
            # `cumulative_fk_residual_production` 永不出现 ⟹ 5a-2 的 guard（要求
            # `cum_fk_verdict ∈ {FAIL, UNMEASURED}`）也永不匹配 ⟹ **整条链对它结构上
            # 不可达**，只能掉到这里，布局动作一不可行就 NO_FEASIBLE_ACTION 收摊。
            #
            # 真机读数：w4 `skipped_windows=[]`、去相关 **69** 帧（min_frames=10）——
            # 求解器**确实**把它算进了 MBAR，证据完全做得出来，只是没人去做。
            #
            # 为什么补在**这里**而不是放宽 5a-1 的 guard：5a-1 排在边际增长判据和所有
            # 布局动作**之前**，放宽它等于把「先 ANALYZE」插到全仓每一条路由前面
            # （实测打断 `加帧被证伪 ⟹ 插 λ` 等既有优先级）。而这里是**死胡同本身**：
            # 已经确认没有任何布局动作可发，ANALYZE 零额外采样、只在现有帧上算，
            # 它要么产出证据让 5a-2 接手，要么算不出来 —— 后者由 `_unsolvable` 与
            # 停滞保护兜底，不会空转。
            if (earliest is not None and int(earliest) in set(int(i) for i in _skew_sel)
                    and _sw.get("cum_fk_verdict") is None
                    and int(earliest) not in set(
                        int(x) for x in (view.get("skipped_windows") or []))):
                return plan(
                    "ANALYZE",
                    f"窗口 {_skew_sel} 是**支撑/偏斜类**失败，加帧治不了；缩跨度也不可行"
                    f"（拆窗：{feas.get('split_tail_window')}；插 λ：{feas.get('insert_lambda')}）。"
                    "但**累计 f_k 偏差证据从未做出来过**（`cumulative_fk_residual_production`"
                    "缺失），而求解器并没有跳过这个窗口（`skipped_windows` 里没有它）"
                    "⟹ 证据算得出来，只是没算。先在**现有帧**上把它算出来（零额外采样），"
                    "再由 held-out 决定是换 f_k 还是缩跨度 —— 在证据缺失时直接判"
                    "`NO_FEASIBLE_ACTION` 是把「没查」说成「无路可走」。",
                    windows=list(_skew_sel), blocked=blocked, earliest=earliest,
                )
            _rw = _bounded_rewindow(
                int(_sw["window_idx"]),
                f"窗口 {_skew_sel} 是**支撑/偏斜类**失败，加帧治不了；而 λ 表上的缩跨度"
                f"也不可行（拆窗：{feas.get('split_tail_window')}；"
                f"插 λ：{feas.get('insert_lambda')}）",
            )
            if _rw is not None:
                return _rw
            return plan(
                "NO_ACTION",
                f"窗口 {_skew_sel} 是**支撑/偏斜类**失败，加帧治不了；而缩跨度也不可行"
                f"（拆窗：{feas.get('split_tail_window')}；插 λ：{feas.get('insert_lambda')}；"
                f"有界重窗：{self.rewindow_feasible(view, int(_sw['window_idx']))}）"
                "⟹ 如实停下，**不拿加帧顶替**。",
                exit_="NO_FEASIBLE_ACTION", windows=list(_skew_sel),
                blocked=blocked, earliest=earliest,
                # [2026-09-17] 偏斜类 + 缩跨度不可行，局部
                halt_scope="TARGET_LOCAL",
            )

        # 🔑🔑 [2026-09-17 P0] **`not _is_skew` ⟹ 改成只认 SAMPLE_SIZE 硬证据。**
        # 先前这里是 `not _is_skew(w)`，而 `is_skew` 只区分两态 ⟹ `UNKNOWN`
        # （归因判不出来：射程三输入缺项 / 老产物缺来源）被算成"样本量类"⟹ 补帧。
        # 这正是用户点名的那句：**只改归因函数不够，下一层会把语义重新丢掉** ——
        # 从「错误授权 D3 停机」换成「错误授权补帧」。
        # `plan()` 的 O1 已在同日因同一理由收紧成只认 SAMPLE_SIZE，这里照同一口径。
        # `UNKNOWN` 的窗口两边都不授权，落到下面的诊断态（见 1c-3 / 兜底）。
        short_self = _pick(earliest, [w for w in _short_all
                            if _attr_of(view, w) == SUPPORT_FAILURE_SAMPLE_SIZE])
        if short_self:
            return plan(
                "RUN_PRODUCTION",
                # ⚠️ `sufficient=False` **不只是**"去相关帧数不足"：它同时含
                # `N_eff/g` 与 top1% 两个条件（见 `window_self_support_check`）。
                # 措辞写死成帧数会把一个偏斜问题说成样本量问题。
                f"窗口 {short_self} 的生产后自检判定支撑不足"
                f"（`sufficient=False`；该判据含去相关帧数、N_eff/g 与 top1% 三项，"
                f"逐项读数见自检产物）"
                f"（{[(w.get('self_n_frames_decorrelated'), w.get('self_min_frames')) for w in _short_all]}）。"
                "这不是终态 —— 语义是「这个窗口还需要加采样」（INSUFFICIENT_DATA ≠ FAIL）。"
                "它们的 checkpoint 此刻还热，补采不作废任何已有帧。",
                windows=short_self,
                blocked=blocked, earliest=earliest,
            )

        # 5c) 🔑🔑🔑 [2026-09-18] **瓶颈在 `N/g`：缩跨度那一整族被证明够不着。**
        #
        # `R = η·N/g` 且 `η ≤ 1`（Kish ESS 的 Cauchy–Schwarz）⟹ `R ≤ N/g`。
        # `N/g < 门` 时，插 λ / 重标定 / 重学 f_k / 有界重窗 —— 它们**全部只动 η**
        # —— 无论怎么改都过不了门。所以这种窗口**不该**进 `_skew_sel`（那一档
        # 授权的正是缩跨度）；它有自己的一档 `N_OVER_G_BOUND`。
        #
        # 还能动的只有两个分母项：
        #   · **N**：同分布补帧。`R ∝ N`（η、g 稳定时），所以它**在数学上是可行的**，
        #     只是可能很贵 —— 贵不贵由 `plan()` 那道唯一的补帧准入闸判（块数硬上限 /
        #     边际增益 / 射程），不在这里另写判据。
        #   · **g**：换独立初态 / 处理慢模态。**本仓目前没有这个动作**（全仓没有
        #     任何生成独立起始构型的执行器）。
        #
        # ⟹ 这里发补帧，让准入闸判。批得动就补；批不动时准入闸会把动作改写成终态
        # 并附上它自己的拒绝理由 —— 那时「加帧够不着（射程闸）」与「改 η 也够不着
        # （本档）」两条同时在案，结论就是**必须降 g**，而那个动作还不存在。
        # 这比静默掉到兜底、报一句"归因不到任何一道具体的门"强：后者把一个**已经
        # 算出来的结论**说成了没结论。
        _nog_sel = _pick(earliest, [w for w in _short_all
                          if _attr_of(view, w) == SUPPORT_FAILURE_N_OVER_G_BOUND])
        if _nog_sel:
            _nw = next(w for w in _short_all
                       if int(w["window_idx"]) in _nog_sel)
            _nog_N = _nw.get("bottleneck_n_eff_input_n_frames")
            _nog_g = _nw.get("bottleneck_g")
            _nog_t = _nw.get("self_n_eff_over_g_eligible")
            return plan(
                "RUN_PRODUCTION",
                f"窗口 {_nog_sel} 的瓶颈在 **N/g**，不在权重剖面上："
                f"瓶颈态 k*={_nw.get('bottleneck_state')}，进 N_eff 的帧数 N={_nog_N}，"
                f"g={_nog_g} ⟹ **N/g = "
                + (f"{float(_nog_N) / float(_nog_g):.3g}"
                   if (_nog_N and _nog_g) else "?")
                + f"** < 门 {_nog_t}。"
                "而 `R = N_eff/g = η·N/g` 且 **η ≤ 1**（Kish ESS 的 Cauchy–Schwarz，"
                "等号仅当权重全等）⟹ **R ≤ N/g** ⟹ 插 λ / 重标定 / 重学 f_k / "
                "有界重窗这一整族（它们只动 η）**被证明够不到门**，不是"
                "「还没试过」。⟹ 不发缩跨度。"
                "还能动的只有分母：**加 N**（同分布补帧，`R ∝ N`，数学上可行）"
                "或**降 g**（换独立初态 / 治慢模态 —— **本仓目前没有这个动作**）。"
                "这里先发补帧，够不够得着由补帧准入闸判；它若拒绝，那条拒绝理由"
                "加上本档就同时钉住了两件事：加帧够不着、改 η 也够不着 ⟹ "
                "剩下的唯一出路是降 g。",
                windows=list(_nog_sel), blocked=blocked, earliest=earliest,
            )

        # 1c-4) 🔑🔑 [2026-09-17 P0] **物理窗口侧的 UNKNOWN 出口。**
        #
        # 三态收紧之后，`_skew_sel`（只收 STRUCTURAL）与 `short_self`（只收
        # SAMPLE_SIZE）都不收 `UNKNOWN` ⟹ 一个自检判失败、归因判不出来的**物理
        # 窗口**两条都不命中，一路掉穿到后面的兜底，并报出一句**假话**：
        # 「所有窗口的预热与生产都已完成」。实测：`earliest=1`、w1/w3 都是 PROBLEM，
        # 而 plan 的 `windows` 是空的。**窗口既没被处理、也没被报出来。**
        # 这正是用户规格那句「不能让它停掉整跑」的反面 —— 它没停跑，它把窗口弄丢了。
        #
        # ⚠️ 与子窗那条（1c-3）同一形状、同一出口。`UNKNOWN` 两边都不授权：
        # 补帧可能越加越差（若其实是结构性），改布局是拿「没测出来」当「测出来是
        # 坏的」（若其实只是数据缺口）。
        # ⚠️ 出口进了 `_RETIRABLE_EXITS`：别处还有活干时退役换窗继续调度，
        # 不因为一个判不出归因的窗口停掉整跑（用户规格原话）。
        # ⚠️⚠️ **只在「没有其他独立硬证据」时才拦。**（用户规格原话）
        # 被求解器跳掉 / 还没有产物 —— 这两类是**独立于归因的硬证据**，它们各自
        # 有更靠后的路由（缺窗、跳窗那几条）。UNKNOWN 分支放在它们前面，不排除
        # 就会**截胡**：实测一个被跳掉的窗口被判成"归因不出来"而退役，路由跑去
        # 修一个下标更小、却没被跳的窗口 —— 正是这条优先级要防的反面。
        # （与子窗那侧 `needs_frames_independent` 同一个道理，我在那边做对了、
        #  在这边漏了。）
        _skipped_idx_now = {
            int(x if not isinstance(x, dict) else x.get("window_index", -1))
            for x in (view.get("skipped_windows") or [])} - {-1}
        _unk_wins = [w for w in _short_all
                     if _attr_of(view, w) == SUPPORT_FAILURE_UNKNOWN
                     and w.get("solver_skip") is None
                     and int(w["window_idx"]) not in _skipped_idx_now
                     and w.get("has_convergence")]
        # 🔑🔑🔑 [2026-09-17 P0 真机] **`UNKNOWN` 不授权**结构**动作；它不该挡住补帧。**
        #
        # 两次真机连崩都死在这一条：
        #   run A：窗口 2 ratio=7.09 / target=10 / headroom=5 ⟹ 射程 35.5 ≥ 10
        #   run B：窗口 4 ratio=4.05 / target=10 / headroom=5 ⟹ 射程 20.3 ≥ 10
        # 三个输入**一个都不缺**、射程**远超门**，`reachable is True` ——
        # 也就是「加帧很可能救得回来」的那一类，却被判 UNKNOWN、一帧不给、
        # 第 2/40 和第 3/40 轮直接终止整跑。而上面那段理由还写着「缺任一就判不了」，
        # 在这种盘面上**是假话**：什么都不缺，是射程说"还没被证伪"。
        #
        # 收紧的原意是对的，但范围划错了：`UNKNOWN` 不该授权的是**结构动作**
        # （插 λ / 有界重窗 / D3 停机）—— 那些要花大钱、改全局布局、且一旦按
        # 「没测出来」当「测出来是坏的」就不可逆。**补帧不是那一类**：
        #   · 它是同分布、可逆、最便宜的动作；
        #   · 它**已经**被 `_frames_admission` 三道闸限住了 —— 块数硬上限、
        #     边际增益刹车、以及**射程闸本身**（剩余配额全花掉也够不着就拦）。
        # 把补帧也砍掉等于把那三道刹车变成不可达，同一个保守被记了两遍，
        # 代价是每一个 ratio 低于门、但够得着的窗口都**停掉整跑**。
        #
        # 判据不新造：直接问**执行器那道准入闸自己**批不批。批 ⟹ 发补帧
        # （语义仍是「尚不可测」，不是「已归因为样本量问题」）；不批 ⟹ 才是真没路走，
        # 走下面的 UNKNOWN 终态。
        # 判据不新造、也**不在这里再问一遍准入闸**（`_frames_admission` 只许挂在
        # `plan()` 里一处 —— 逐个挂必然漏，那条结构性守卫是对的）：
        # 直接发补帧，由 `plan()` 那一道唯一的闸判。批 ⟹ 补一块换新证据；
        # 不批 ⟹ 它会原地改写成 `NO_ACTION` + 终态，并把**准入闸真实的拒绝理由**
        # 接在下面这段理由后面 —— 那才是"真没路走"，而且说得比 UNKNOWN 更具体。
        _unk_frames = _pick(earliest, _unk_wins)
        if _unk_frames:
            _fw = next(w for w in _unk_wins
                       if int(w["window_idx"]) in _unk_frames)
            return plan(
                "RUN_PRODUCTION",
                f"窗口 {_unk_frames} 自检判支撑不足、归因判不出来"
                f"（verdict={_fw.get('self_verdict')}，来源 "
                f"{_fw.get('self_verdict_source')}；ratio={_fw.get('min_n_eff_over_g')}"
                f" / target={_fw.get('self_n_eff_over_g_eligible')}"
                f" / headroom={frames_growth_headroom(view, int(_fw['window_idx']))}）"
                "—— `UNKNOWN` **不授权结构动作**（插 λ / 有界重窗 / D3 停机：那是拿"
                "「没测出来」当「测出来是坏的」），**但它不挡补帧**："
                "补帧是同分布、可逆、最便宜的动作，而且**已经**被补帧准入的三道闸"
                "（块数硬上限 / 边际增益刹车 / 射程闸本身）限住了。"
                "连它也砍掉等于把那三道刹车变成不可达，同一个保守记两遍 —— "
                "真机代价是每个「ratio 低于门但够得着」的窗口都停掉整跑"
                "（2026-09-17 两次连崩：ratio 7.09 和 4.05，射程 35.5 / 20.3，门 10）。"
                "语义是「尚不可测」（INSUFFICIENT_DATA ≠ FAIL），"
                "**不是**「已归因为样本量问题」——下一块之后重新归因。",
                windows=list(_unk_frames),
            )

        _unk_sel = _pick(earliest, _unk_wins)
        if _unk_sel:
            _uw = next(w for w in _unk_wins if int(w["window_idx"]) in _unk_sel)
            return plan(
                "NO_ACTION",
                f"窗口 {_unk_sel} 自检判支撑不足，但**归因判不出来**"
                f"（verdict={_uw.get('self_verdict')}，来源 "
                f"{_uw.get('self_verdict_source')}；射程三输入 "
                f"ratio={_uw.get('min_n_eff_over_g')} / "
                f"target={_uw.get('self_n_eff_over_g_eligible')} / "
                f"headroom={frames_growth_headroom(view, int(_uw['window_idx']))}"
                "）。"
                "`UNKNOWN` **不授权结构动作**（插 λ / 有界重窗 / D3 停机：那是拿"
                "「没测出来」当「测出来是坏的」）。"
                "⚠️ 走到这一条说明**补帧那条路也没走成** —— 上面那个分支会先发 "
                "`RUN_PRODUCTION`，只有当这个窗口连 `_pick` 都选不中"
                "（它不是本轮的 `earliest`）时才落到这里；真的批不出帧时，"
                "`plan()` 里那道唯一的准入闸会把动作改写成终态并附上它自己的拒绝理由。"
                " ⟹ 停在诊断态。"
                "要往下走：把缺的读数补出来（写侧 `n_eff_over_g_eligible_threshold` / "
                "块账 / 自检来源），或显式提高该窗口的块数上限，"
                "**不是**挑一个动作试试看。",
                exit_="SUPPORT_ATTRIBUTION_UNKNOWN",
                windows=list(_unk_sel),
                missing=[f"窗口 {int(_uw['window_idx'])} 的支撑失败归因输入"],
            )


        # 6) 被踢出协方差链的窗口（去相关后帧数不足）⟹ 补采。
        #    这条路径以前不可达：被跳过的窗口在生成 overlap 诊断**之前**就 continue 了，
        #    于是进不了 rescue 候选名单（P0-2b 修的就是这个）。
        _sk_sel = _pick(earliest, view["skipped_windows"], key=None)
        if _sk_sel:
            return plan(
                "RUN_PRODUCTION",
                f"窗口 {_sk_sel} 去相关后有效帧数不足、被踢出协方差链。"
                "先补采（同一分布、接着原段）；这是最便宜且不作废任何已有帧的动作。",
                windows=_sk_sel,
                blocked=blocked, earliest=earliest,
            )

        # 6a) 没有任何 PROBLEM，但有窗口**证据缺失**（UNKNOWN）⟹ 先把证据做出来。
        #     绝不能因为"不知道"就去换 Epoch —— 那是在毫无根据的窗口上烧 GPU。
        if earliest is None and unknown:
            return plan(
                "ANALYZE",
                f"窗口 {unknown} 没有自检证据（`self_verdict` 不存在）⟹ 状态是"
                "**UNKNOWN，不是有问题**。对「不知道」的正确动作是**产出证据**"
                "（跑 stage 分析、落自检产物），不是重标定。"
                "⚠️ 缺证据同样**不等于通过** —— 在证据补齐之前不得判 DONE。",
                windows=unknown, blocked=blocked, earliest=earliest,
                missing=["逐窗自检产物 dual_window_*_self_support.json"],
            )

        # 6b) **前缀全部合格**之后，才轮到"缺窗口"。它此前是全局最高优先级，
        #     会把上游未解决的窗口整个盖住（实测把 win3/win4 盖成 win5）。
        #     缺窗**仍然**是硬约束——缺窗口的和不是 ΔG——只是正确位置在因果顺序之后。
        if view["missing_windows"]:
            _next_missing = min(view["missing_windows"])
            return plan(
                "RUN_PRODUCTION",
                f"前缀窗口全部合格；布局里有 {view['n_windows_expected']} 个窗口、"
                f"产物里只有 {view['n_windows_found']} 个，缺 {view['missing_windows']}。"
                "**缺窗口的和不是 ΔG，禁止当结果使用**（不是「精度差一点」，是**另一个量**）。"
                f"按因果顺序先跑 win{_next_missing}。",
                windows=[_next_missing],
                blocked=blocked, earliest=earliest,
            )

        # 7) 分析的硬不变量已通过 ⟹ 终态。**注意它不是 `DONE`。**
        #    与分支 0a 同一套口径（那条在路由之前、这条在路由之后兜底）：
        #    `DONE` 的充要条件是 `precision_status == MEETS_CROSS_REPEAT_TARGET`，
        #    单次 run 里恒不成立 ⟹ 实际出口是「分析完整 / 精度未测」。
        if view["stage_analysis_status"] == "ANALYSIS_COMPLETE":
            if view.get("stage_precision_status") == "MEETS_CROSS_REPEAT_TARGET":
                return plan(
                    "DONE",
                    "分析的硬不变量全部通过，且跨重复精度已达标"
                    "（`precision_status = MEETS_CROSS_REPEAT_TARGET`）。"
                    "⚠️ 这不等于答案正确 —— docs/archive/STAGE2_ROOT_CAUSE_2026-08-28.md §2"
                    "「所有收敛门对该失效模式失明」仍然有效（该节被其自身的超越"
                    "声明明确标为仍然有效）。",
                    exit_="DONE_UNTRUSTED" if self.allow_untrusted else "DONE",
                    blocked=blocked, earliest=earliest,
                )
            return plan(
                "NO_ACTION",
                "分析的硬不变量全部通过（`analysis_status = ANALYSIS_COMPLETE`），"
                f"但 `precision_status = {view.get('stage_precision_status')!r}` —— "
                "**精度从未被测过**，只有跨重复才测得出来 ⟹ 这一跑到此为止，"
                "结果**不得作为已验收结果发布**。逐段质量门的读数在 "
                "`stage_quality_gates` 里供人审查（纯报告，不驱动动作）。",
                exit_="ANALYSIS_COMPLETE_PRECISION_UNMEASURED",
                blocked=blocked, earliest=earliest,
            )

        # 8) 窗口都跑完了、stage 分析还没跑 ⟹ 只差分析。
        if not view["has_stage_result"]:
            return plan(
                "ANALYZE",
                "所有窗口的预热与生产都已完成，但 stage 分析还没跑"
                "（stage2_*.json 不存在）⟹ 先跑分析。",
                # ANALYZE 无出口 —— 它是路由，不是终态。
                missing=["stage2_*.json"],
            )

        # 9) 分析没跑完整（`analysis_status = ANALYSIS_INCOMPLETE`）⟹ **如实停下**。
        #
        # 🔑🔑 [2026-09-15 老板定案] **原来的 9b / 9c 已删除，不得重建。**
        # 它们把逐段质量门（`min_overlap` / `target_support_gate` /
        # `min_decorrelated_samples` / `max_endpoint_uncertainty_kJ_mol`）映射成
        # 「缩跨度」「加采样」两类**命令**，由本分支直接发 `SPLIT_TAIL_WINDOW` /
        # `INSERT_LAMBDA` / `IMMUTABLE_REWINDOW` / `RUN_PRODUCTION` —— **烧 GPU**。
        # 那四个阈值全是**未经标定**的拟合值；`max_endpoint_uncertainty_kJ_mol ≤ 1.0`
        # 更是抄自一个**默认关闭**的 early-stop 启发式的默认形参，全仓无任何依据。
        # 而且它是**逐段**量，科学目标却是两腿合成的 `σ_bind`：实测各段
        # `[0.3×5, 2.0]` ⟹ σ_bind = 2.98 已达标，门却判未收敛并命令补帧；
        # 把那一段从 2.0 压到 1.0 要 **4× 采样量、精度提升为零**。
        # 更根本：MBAR 的 σ 是**渐近**估计，对「该采的构型一次都没采到」结构性失明
        # —— 4W53（+32 kJ/mol）与 decharging（−88 kJ/mol）两次大错 σ 都很小、
        # 都没报警。⟹ **未经标定的阈值不得驱动自动补帧。**
        #
        # ⚠️ **不在这里发明新的补帧依据去填这个空。** 走到分支 9 说明：分析的硬
        # 不变量没过（求解器报错 / 丢窗口之类），而所有**有对症动作**的失效形状
        # —— 缺窗（6b）、跳窗（6）、跳子窗与子窗缺帧（1c）、逐窗自检支撑不足
        # （5x 各支）—— 都在**上游**分支，它们一个都没命中。所以这里只剩一个
        # 诚实的终态：如实停下，把全部诊断一起交出去。
        _gate_failures = stage_quality_gate_failures(self._read_stage_result())
        _named = "、".join(
            f"{f['gate']}（实测 {f['value']}，门 {f['threshold']}，"
            f"分类 {f['report_category']}，**仅报告**）"
            for f in _gate_failures
        ) or "（读不出任何具体门的读数）"
        _why = "；".join(view.get("stage_analysis_incomplete_reasons") or []) \
            or "（写侧判了 ANALYSIS_INCOMPLETE 但没给理由）"

        # 🗑️ [2026-09-17，用户拍板 (a)] **这里原有一条「`allow_untrusted` ⟹ DONE_UNTRUSTED」
        # 的分支，是死代码，连同它那段「这是发布策略」的注释一起删掉。**
        #
        # 它**一次都没发出过** `DONE`：`plan()` 的 O4（`action == "DONE"` 且
        # `_evidence_status != CONVERGED` ⟹ 改写成 `NO_ACTION`/`HALT_EVIDENCE_CONTRADICTS_DONE`）
        # **没有 `allow_untrusted` 豁免**，而走到这里 `analysis_status` 必然是
        # `ANALYSIS_INCOMPLETE`（上游已排除 `ANALYSIS_COMPLETE` 与「没有 stage 结果」）
        # ⟹ `_evidence_status` 不可能是 `CONVERGED` ⟹ 必被改写。
        #
        # **不给 O4 开豁免**的理由不止"O4 的判断对"：`ANALYSIS_INCOMPLETE` 表示
        # 缺窗 / 跳窗 / 求解失败这类**硬不变量**没满足，不是"质量门偏低"。即使放开
        # O4，下游 `abfe_pipeline` 的 `analysis_status != ANALYSIS_COMPLETE ⟹ RuntimeError
        # （拒绝标记为 completed）`照样拦死 ⟹ 只会变成"控制器先报完成、再被下游打回"。
        #
        # ⚠️ **`allow_untrusted` 本身没有被削弱**：它仍然改 `trust_level`
        # （→ `OVERRIDDEN_UNTRUSTED`）、仍然在分析**完整**时给出 `DONE_UNTRUSTED`
        # （分支 0a 与分支 7 两处 `exit_="DONE_UNTRUSTED" if self.allow_untrusted else "DONE"`），
        # 也仍然管跳过 rescue / 不进指纹。死掉的只有"分析不完整也放行"这一条。

        # 9a) 唯一的出口：如实停下并报告。**不是 DONE**，也不发任何花 GPU 的动作。
        # ⚠️ 停之前先分清「没有对症动作」与「归因读不出数」（见 `_attribution_blind_spots`）：
        # 两者的措辞一样、处置相反，前者交人工，后者是证据缺失、该去补产物。
        _corrupt, _absent = self._attribution_blind_spots(view)
        _blind_note = (
            ("⚠️⚠️ **但先别当成「无路可走」：有归因读数是非有限值**"
             f"（{_corrupt}）。上游分支拿 NaN 做的每一个比较都是 False ⟹ "
             "「一条都没命中」**在这里不是结论，是读到了坏数**。"
             "该做的是查这些读数为什么算成 NaN/inf，不是交人工判定无路可走。"
             if _corrupt else "")
            + (f"（另有尚未算出的归因读数，属正常：{_absent}）" if _absent else "")
        )
        return plan(
            "NO_ACTION",
            f"分析未跑完整（`analysis_status = ANALYSIS_INCOMPLETE`）：{_why}。"
            "有对症动作的失效形状都在上游分支、一个都没命中 ⟹ 这里**没有**"
            "一个在科学上站得住的下一步动作。编一个出来会退化成「再跑一次直到"
            "碰巧通过」，回去跑 ANALYZE 则是空转（分析已经跑过）⟹ 交人工。"
            + _blind_note
            + "⚠️ 逐段质量门**不再**是动作依据（未经标定的阈值，2026-09-15 定案），"
            f"它们的读数只作报告：{_named}。"
            "完整条目见返回值的 `stage_quality_gates`（每条带 `drives_action: False`）。",
            exit_="NO_FEASIBLE_ACTION",
            missing=(
                [f"窗口 {r['window_idx']} 的归因读数是非有限值："
                 f"{'、'.join(r['non_finite'])}" for r in _corrupt]
                or (view.get("stage_analysis_incomplete_reasons") or
                    ["stage 分析的硬不变量读数"])
            ),
            blocked=blocked, earliest=earliest,
            # [2026-09-17] 分析未跑完整，且上游每一条有对症动作的分支都没命中 ⟹ 整条 stage 无路，换窗也没用
            halt_scope="STAGE_GLOBAL",
        )

    # 兜底分支要区分「没有对症动作」和「归因函数读不出数」时看的那几个读数 ——
    # 就是上游各分支真正据以分岔的量，不是随便挑的。
    _ATTRIBUTION_INPUTS = (
        ("phase", "窗口相 phase"),
        ("self_verdict", "逐窗自检结论 self_verdict"),
        ("cum_fk_verdict", "累计 f_k 残差判定 cum_fk_verdict"),
        ("min_n_eff_over_g", "主验收量 min N_eff/g"),
        ("warmup_steps_left", "预热预算余额 warmup_steps_left"),
    )

    @classmethod
    def _attribution_blind_spots(cls, view: Dict[str, Any]):
        """落到兜底时，逐窗把归因读数分成**坏数**和**没算**。返回 `(corrupt, absent)`。

        🔑🔑 [2026-09-17，死线 D8] **「没有对症动作」与「归因读不出数」是两件事，
        处置相反。** 兜底的措辞说的是前者（交人工），而真机出现过后者 ——
        `converged=False` + 零条归因，真相是某道门的读数是 NaN（审计 #59 已修
        `stage_quality_gate_failures` 的 fail-open，但**兜底本身仍然分不出这两种**）。

        两类要分开，别混成一个"读不出来"：
          · **corrupt**（非有限：NaN / inf）⟹ **真缺陷**。上游分支拿它做比较时
            每一个比较都是 False（`nan >= x`、`nan < x` 全假）⟹ 静默地一条都不命中，
            长得和"确实没有对症动作"一模一样。必须大声报。
          · **absent**（None）⟹ **正常**。很多窗口本来就还没有 `cum_fk_verdict`
            （证据没产出），那恰恰是上游 5a-1 发 ANALYZE 的依据。只作附注。

        ⚠️ 不做「全部读数都读不出来 ⟹ 全盲」那种判据：有产物的窗口 `phase` 恒可导出，
        全盲在构造上不可达，写了就是死支。
        ⚠️ 这**不新增任何动作**：D8 是设计上正确的兜底，给它配动作会退化成
        「再跑一次直到碰巧通过」。这里只改报告 —— 让停下来的理由说的是真相。
        """
        corrupt, absent = [], []
        for w in (view.get("windows") or []):
            _bad, _none = [], []
            for key, label in cls._ATTRIBUTION_INPUTS:
                v = w.get(key)
                if v is None:
                    _none.append(label)
                elif isinstance(v, float) and not np.isfinite(v):
                    _bad.append(f"{label}={v}")
            if _bad:
                corrupt.append({"window_idx": int(w["window_idx"]),
                                "non_finite": _bad})
            if _none:
                absent.append({"window_idx": int(w["window_idx"]),
                               "not_yet_computed": _none})
        return corrupt, absent

    @staticmethod
    def first_untrusted_window(view: Dict[str, Any]) -> Optional[int]:
        """**第一个不可信窗口**的下标 —— tail-only 重分的起点，由控制器自己算。

        [老板] 「当前从 `first_untrusted_window` 开始重分；**不是让用户手选 anchor**。」
        判据就是主验收量：verdict 不是 `ANALYSIS_ELIGIBLE` 的第一个窗口。
        （本例是 win3，也是第一个 `min(N_eff/g) < 10` 的 —— 人选的 anchor 恰好同解，
        但**机制必须是自动的**。）

        🔑🔑 [2026-09-14 第三轮复核] **「证据被我们自己作废」也是不可信，不是"还不知道"。**

        先前判据只有 `v is not None and v != ANALYSIS_ELIGIBLE` —— 那个
        `v is not None` 是对的（没跑过自检 ≠ 有问题，设计 §4 的三态），但它把
        **布局变更作废掉证据**的窗口一起漏掉了。设计 §4 同一节明写：插 λ / 拆窗
        之后旧证据描述的是另一个窗口几何、已被剔除，那是**确定的未解决**，
        必须判 PROBLEM —— `_window_state()` 就是这么写的，本函数却另写了一套。

        真机 cyclod_ligand1/rep3：win0–3 全 ANALYSIS_ELIGIBLE，win4 的产物被布局
        变更作废、一份证据都没有（`self_verdict=None`、`stale_layout_evidence_only`）
        ⟹ 本函数返回 None ⟹ anchor 取不到 ⟹ 末窗 K=11 > hi=8，
        `_legalize_tail_window` 在启动布局校验处 fail-closed 抛错，
        **decide() 一次都轮不到、整个 run 起不来**。
        判据与 `_window_state()` 对齐即可，不新发明规则。
        """
        for w in sorted(view.get("windows") or [], key=lambda x: int(x["window_idx"])):
            if w.get("stale_layout_evidence_only"):
                return int(w["window_idx"])
            v = w.get("self_verdict")
            if v is not None and v != "ANALYSIS_ELIGIBLE":
                return int(w["window_idx"])
        return None

    @staticmethod
    def stale_layout_windows(view: Dict[str, Any]) -> set:
        """产物描述的是**另一套 λ 布局**的窗口号。

        🔑 [2026-09-16] 这个集合有**两个**消费者（`decide()` 的 1d-0 分支、
        主循环停滞保护的降级闸），所以只能有一份实现 —— 它判的是
        「拿已有帧重解会不会维度不符」，而 `ibs_engine` 的 loader 对同一件事
        是 fail-closed 抛 `ValueError`。两边一旦分岔，就是一个在构造上不可能
        成功的动作被发出去、炸穿流水线（真机 7 个 run）。
        """
        return {
            int(x) for x in (view.get("stale_layout_evidence") or {})
        } | {
            int(w["window_idx"]) for w in (view.get("windows") or [])
            if w.get("stale_layout_evidence_only")
        }

    def tail_repartition_anchor(self, view: Dict[str, Any]) -> Optional[float]:
        """`first_untrusted_window` 的**首态** λ —— 它就是与前一窗共享的那个节点。

        冻结的是「anchor 之前」的窗口，所以从**不可信窗口自己的首态**切，
        才能把这个已知弱的窗口一起重分掉；若从它的**末态**切，会把它永久保留。
        """
        idx = self.first_untrusted_window(view)
        if idx is None or idx <= 0:
            return None
        path = view.get("path") or {}
        ranges = path.get("window_ranges")
        if not ranges or idx >= len(ranges):
            return None
        # 🔑🔑 [2026-09-14 真机] 这里原来读 `w["lambda_vdw_hi"]` —— 全仓**零处写**
        # 这个键（只有这一行读它）。于是 anchor 恒为 None，执行器每次都走
        # 「取不到 tail anchor，跳过本动作」：`SPLIT_TAIL_WINDOW` 自诞生起
        # **一次都没有真正执行过**。cyclod_ligand1/rep2 发了 4 次、版本链上
        # 一条 tail_repartition 都没有，最后靠通用停滞探测判 NO_FEASIBLE_ACTION。
        # anchor 的定义本来就是「不可信窗口自己的首态」= `lam[ranges[idx][0]]`，
        # 布局和 λ 表都在 `view["path"]` 里，不需要第二个字段来搬运它。
        lam = path.get("lambdas_vdw") or []
        start = int(ranges[idx][0])
        if start >= len(lam):
            return None
        return float(lam[start])

    def _pilot_for_partition(self):
        """(pilot_lambdas, metric_g)；读不到返回 (None, None)。

        与 `abfe_pipeline._load_pilot_for_path_evolution` 读的是同一份文件的
        同一批字段 —— 尾段重分要和生产用**同一个**分窗判据，就得拿到同一份 pilot。
        """
        diag = (self._json(os.path.join(
            self.path_checkpoint_dir, "preopt_dual_vanishing.json"
        )) or {}).get("path_diagnostics") or {}
        pl, mg = diag.get("pilot_lambdas"), diag.get("metric_g")
        return (list(pl) if pl else None, list(mg) if mg else None)

    def _dry_run_tail_repartition(self, view: Dict[str, Any]):
        """把 `SPLIT_TAIL_WINDOW` 的执行器跑一遍（纯计算，不落盘）。

        可行性与执行共用**这一次调用的结果**：`feasible()` 用它判可不可行，
        执行器用它返回的 ranges 落地。两边不再各切一遍。
        """
        path = view.get("path") or {}
        lam = [float(x) for x in (path.get("lambdas_vdw") or [])]
        ranges = [tuple(int(i) for i in r) for r in (path.get("window_ranges") or [])]
        anchor = self.tail_repartition_anchor(view)
        if not lam or not ranges or anchor is None:
            raise RuntimeError("读不到 λ 表 / 布局 / anchor")
        a_idx = int(ranges[int(self.first_untrusted_window(view))][0])
        m0 = sum(1 for a, _b in ranges if a >= a_idx)
        pl, mg = self._pilot_for_partition()
        new_ranges, diag = repartition_tail_from_anchor(
            lam, ranges, float(anchor),
            min_states_per_window=self.lo,
            max_states_per_window=self.hi,
            # 🔑 目标窗口数 = 原尾段窗口数 + 1。这就是"拆细"的定义，
            # 也是可行性判据问的那个 m ≥ m0+1 里最小的那个 m。
            n_windows=m0 + 1,
            pilot_lambdas=pl,
            metric_g=mg,
            partition_criterion=self.partition_criterion,
        )
        return [list(r) for r in new_ranges], diag

    @staticmethod
    def _epoch_validation_unaffordable(
        view: Dict[str, Any], windows: Sequence[int], *, new_epoch: bool = False
    ) -> Optional[Dict[str, Any]]:
        """**决定重标定之前，先确认新 Epoch 还付得起最低验证额度。**

        [老板定案 F] 预算要拆两本账：
            global_consumed_budget        永远继承，防止新段白送额度
            epoch_validation_reservation  每个新候选**单独预留**

        「**没钱验证就不启动重标定**」—— 不能启动动作之后才发现新 f_k 没预算验。
        win2 **连续三次**死在这：位移触发把它拖进新 Epoch，新 f_k 要过验证门，
        而账本已经 555000/555000、剩 0，于是循环一次都没进就被标成
        `f_k 不收敛`（第三个变体：预算为零）。

        最低额度用既有的冻结验证阶梯**第一档**，不新发明阈值。
        返回 None = 付得起；否则返回一份说明（调用方据此走 HALT_BUDGET）。

        🔑🔑 [审计 #33，2026-09-14] **两类动作的门槛不是同一个数。**

        `new_epoch=False`（`RECALIBRATE_FK` / `PROBE_REANCHOR_EPOCH`）：只**重解**
        已有帧、不重学 f_k ⟹ 要的就是首档冻结验证额度（50k）。

        `new_epoch=True`（`RELEARN_FK_EPOCH`）：执行器要从头 **LEARN**
        ⟹ 真实需求是 `relearn_epoch_required_steps()` = learn + burn-in + 首档，
        ≥ 140k、真机 cyclod_ligand1/rep2 win3 实测 **290k**。先前这里一律拿 50k
        当门槛 ⟹ **控制器批准、执行器拒绝并终止整个循环**（而不是换个动作），
        方向是危险那一侧：批准一个跑不完的 Epoch，把预算烧光再半路死掉。
        两个数都写进返回值，便于事后对账。
        """
        try:
            import ibs_engine as _ie
            floor_first_rung = int(_ie.FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS[0])
        except Exception:
            return None  # 拿不到阈值就不假装判断（fail-open：只影响诊断）
        by_idx = {int(w["window_idx"]): w for w in (view.get("windows") or [])}
        broke = []
        for i in windows:
            w = by_idx.get(int(i))
            if w is None:
                continue
            # 全新 Epoch 的门槛按**这个窗口自己**的账本自校准（learn 步数逐窗不同）。
            floor = (int(relearn_epoch_required_steps(w)) if new_epoch
                     else int(floor_first_rung))
            left = w.get("warmup_steps_left")
            # ⚠️ **预算未知 = 不可行**（fail-closed）。以前 `left is None` 会跳过检查，
            # 于是"账本读不到"的窗口能一路走到重标定 —— 而真实零预算窗口
            # （555k/555k）正是这样漏过去的。不知道有没有钱，就不许开新 Epoch。
            if left is None:
                broke.append({
                    "window_idx": int(i), "warmup_steps_left": None,
                    "min_epoch_validation_reservation": floor,
                    "first_rung_only_reservation": int(floor_first_rung),
                    "relearn_epoch_required_steps": int(
                        relearn_epoch_required_steps(w)),
                    "reason": "budget_unknown_fail_closed",
                })
            elif int(left) < floor:
                broke.append({
                    "window_idx": int(i),
                    "warmup_steps_left": int(left),
                    "min_epoch_validation_reservation": floor,
                    "first_rung_only_reservation": int(floor_first_rung),
                    "relearn_epoch_required_steps": int(
                        relearn_epoch_required_steps(w)),
                })
        if not broke:
            return None
        return {
            "requires_new_learning": bool(new_epoch),
            "windows": broke,
            "source": (
                "relearn_epoch_required_steps()（learn+burn-in+首档）"
                if new_epoch
                else "ibs_engine.FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS[0]（只重解）"
            ),
        }

    @staticmethod
    def _warmup_weakness_ranking(view: Dict[str, Any]) -> Dict[str, Any]:
        """按 warmup 剖面给窗口**排序**（谁最可能在生产里撑不住）。

        **刻意只排序、不打分、不给阈值。** 三个口径各排一次、不做加权合成 ——
        合成就等于凭空发明一个权重。谁在多个口径上都排在最弱端，才是真信号。

        口径与方向：
          · ``warmup_g``（统计低效率）越大越弱
          · ``warmup_n_frames_used``（gate 实际用上的去相关帧数）越小越弱
          · ``warmup_min_absolute_ess`` 越小越弱
        """
        W = [w for w in view.get("windows") or []]
        out: Dict[str, Any] = {
            "note": ("只排序、不打分、不给阈值；剖面→生产预算的映射未定"
                     "（PLAN P3）。三个口径不做加权合成。"),
            "by_metric": {},
            "weakest_overall": None,
        }
        specs = (("warmup_g", True), ("warmup_n_frames_used", False),
                 ("warmup_min_absolute_ess", False))
        score: Dict[int, int] = {}
        for key, bigger_is_worse in specs:
            vals = [(w["window_idx"], w.get(key)) for w in W if w.get(key) is not None]
            if not vals:
                continue
            vals.sort(key=lambda t: float(t[1]), reverse=bigger_is_worse)
            out["by_metric"][key] = [
                {"window_idx": i, "value": float(v)} for i, v in vals
            ]
            for rank, (i, _v) in enumerate(vals):
                score[i] = score.get(i, 0) + rank
        if score:
            out["weakest_overall"] = [
                i for i, _ in sorted(score.items(), key=lambda t: t[1])
            ]
        return out

    @staticmethod
    def _coverage_incomplete(view) -> List[Any]:
        """**求解覆盖有没有缺口**：缺窗 / 跳窗 / 跳子窗，返回缺口清单（空 = 完整）。

        🔑🔑 [2026-09-14] **子窗被跳与物理窗口被跳是同一条硬约束，不是类比。**
        `skipped_windows` 拆成物理窗口 + `skipped_sampling_units` 之后，所有拿它
        当 DONE 闸的读点就只覆盖物理窗口了 —— 那是**漏网**。被求解器踢出协方差链的
        子窗意味着**它那段 λ 没进最终的 MBAR**，交出去的是缺了一段的和，而
        「缺窗口的和不是 ΔG，是另一个量」在本仓是硬约束。
        子窗缺了还**更严重**：子窗是 immutable rewindow 用来**接管父窗那段 λ** 的，
        父窗已经退出求解覆盖、不会自动顶回来 ⟹ 那段 λ 直接没人管。

        ⚠️ 抽成一份的理由与 `_layout_trustworthy` 完全相同：分支 0a 与
        `_evidence_status` 各写一份、不同步，已经造成过一次真 regression。

        ⚠️ **`production_rescue_targets` 维持只看物理窗口**（别"修"它）：
        production rescue 是按物理窗口目录组织的旧机制（`vanishing_rescue/<plan_id>`），
        对子窗**在构造上跑不动**。给子窗补帧的对症动作是带 `unit_id` 的
        `RUN_PRODUCTION`（走 `_topup_rewindow_child` 续跑它自己的 checkpoint），
        由分支 1c 接住 —— 两条路，不是一条路的两半。
        """
        out: List[Any] = []
        # 🔑 [2026-09-15 审计④] `out_of_range_windows` 一并算缺口：证据里混进了
        # 当前布局根本没有的窗口 ⟹ 这份"完成"描述的不是当前这条路径。
        for k in ("missing_windows", "skipped_windows", "skipped_sampling_units",
                  "out_of_range_windows"):
            for x in ((view or {}).get(k) or []):
                out.append((k, x))
        return out

    @staticmethod
    def _layout_trustworthy(view) -> bool:
        """这份 stage 结论（`analysis_status` / `precision_status`）描述的**是不是
        当前布局**。

        🔑🔑 [2026-09-14] **两处共用一份，别各写一遍。**
        分支 0a 与 `_evidence_status` 原来各自写了一份版本前置，而分支 7 一份都没有
        ⟹ 老产物上走出 `action=DONE` + `evidence_status=INCONCLUSIVE` 这种自相矛盾
        的结局（`tests/test_stage2_repair_controller.py::
        test_converged_stage_is_done_but_not_claimed_correct` 钉的就是它）。

        判据放宽的理由：版本核实只在**布局演化过**时才是必要条件。路径链从没演化过
        （当前版本 ≤ 1，没插过 λ 也没拆过窗）时根本不存在"另一条布局"，而 stage 缓存
        `stage2_vanishing.json` 历史上不盖 `path_version` ⟹ **所有既有 run 恒为
        False**，CTL-11 对它们完全是死的。
        「布局演化过 + 结果没盖版本号」仍然 fail-closed —— 那才是真风险。
        """
        _pv_now = int((view or {}).get("path_version") or 1)
        return bool((view or {}).get("stage_result_path_version_verified")
                    or _pv_now <= 1)

    def _evidence_status(self, view, action, exit_) -> str:
        """证据状态跟执行状态是两件事。`allow_untrusted` 不得影响这一维。

        [老板定案 I] **缺窗与救援耗尽必须是 `INSUFFICIENT_DATA`，不是含糊的
        `INCONCLUSIVE`** —— 前者明确说"证据不够、还能补"，后者容易被读成
        "结果差一点"。禁止部分和冒充结果。
        """
        if any(w["verdict"] == "STATISTICALLY_REJECTED" for w in view["windows"]):
            # REJECTED 优先：它是关于**某份 f_k** 的更强、更具体的结论，
            # 即便同时还缺预算。
            return "REJECTED"
        # 预算耗尽**永远**是"还没测够" —— 老板把 HALT_BUDGET 与
        # evidence_status=INSUFFICIENT_DATA 绑定。绝不是 FAIL。
        # 🔑 [审计 #28] 先前只映了前两个，漏了后两个 —— 而它们同样是"预算耗尽"，
        # 于是被报成 `INCONCLUSIVE`（"结果差一点"），与三行之上自己写的规则
        # 「预算耗尽**永远**是还没测够」直接矛盾。
        # `HALT_FRAMES_ADMISSION_CAP`（审计 #30 新增）同理：块数配额用完 =
        # 还没测够，不是测出来不合格。
        if exit_ in ("HALT_BUDGET", "HALT_LOCAL_VALIDATION_CAP",
                     "GLOBAL_BUDGET_EXHAUSTED",
                     "HALT_VALIDATION_BUDGET_UNREACHABLE",
                     "HALT_FRAMES_ADMISSION_CAP"):
            return "INSUFFICIENT_DATA"
        # [审计 #58] 缺口判据与分支 0a 的 DONE 闸**共用一份**（见
        # `_coverage_incomplete` 的长注释）：缺窗 / 跳窗 / 跳子窗都是覆盖有缺口。
        if self._coverage_incomplete(view):
            return "INSUFFICIENT_DATA"
        # 🔑🔑 [2026-09-15 老板定案] **`CONVERGED` 的充要条件是跨重复精度达标。**
        # 先前这里是 `stage_converged is True ⟹ CONVERGED` —— 那个 `converged`
        # 只是四个**未经标定**阈值的合取，把「算出了数」误报成「精度已验收」。
        # 现在两维分开判：
        #   · `analysis_status == ANALYSIS_COMPLETE` 只说**分析跑完整了**
        #     （硬不变量），**不得**据此报 CONVERGED；
        #   · 只有 `precision_status == MEETS_CROSS_REPEAT_TARGET` 才是
        #     CONVERGED —— 单次 run 里恒不成立（一个 run 无法自证精度）。
        # ⚠️ 版本没核实过 / 有证据被布局变更作废时仍然一票否决（那份结论描述的
        #   可能是另一条布局；同分支 0a 的补丁，理由写在那里）。
        if (self._layout_trustworthy(view)
                and not view.get("stale_layout_evidence")
                and view.get("stage_analysis_status") == "ANALYSIS_COMPLETE"):
            if view.get("stage_precision_status") == "MEETS_CROSS_REPEAT_TARGET":
                return "CONVERGED"
            # 分析完整、精度未测 ⟹ **自己的词**。既不是 CONVERGED（没验收过），
            # 也不是 INSUFFICIENT_DATA（不缺数据，是这一跑测不了精度），
            # 更不是 INCONCLUSIVE（不是"差一点"）。下游拿它跟 "CONVERGED"
            # 比字符串会**明确地**不相等 —— 这正是要的 fail-loud。
            return "PRECISION_UNMEASURED"
        # 任一窗口的主验收量（N_eff/g）判不可分析 ⟹ 整条路径的证据就不够。
        if any(
            w.get("self_verdict") in ("HARD_INSUFFICIENT", "INSUFFICIENT_DATA")
            for w in view["windows"]
        ):
            return "INSUFFICIENT_DATA"
        # 🔑 [2026-09-14 新发现] **f_k 证据本身「还没测出来」也是 INSUFFICIENT_DATA。**
        # 先前这里只看**自检**的 `self_verdict`，于是一个还在预热/验证、
        # `f_k_evidence_status ∈ {calibrated, indeterminate, none}` 的窗口
        # （`_VERDICT` 把它们映成 `INSUFFICIENT_DATA`，字面意思就是"尚不可测"）
        # 会被报成 `INCONCLUSIVE`（"结果差一点"）。这与 `_VERDICT` 那张表自己的
        # 口径矛盾，而且 INCONCLUSIVE 容易被下游读成"测出来不合格"。
        # 先前它没暴露，只是因为这类窗口总会顺带拿到一个 `HALT_BUDGET` 出口。
        if any(w.get("verdict") == "INSUFFICIENT_DATA" for w in view["windows"]):
            return "INSUFFICIENT_DATA"
        return "INCONCLUSIVE"

    # ----------------------------------------------------------- Segment 聚合

    @classmethod
    def for_physical_stage(
        cls,
        run_dir: str,
        stage_base: str = "vanishing",
        stage_type: str = "vdw",
        **kwargs: Any,
    ) -> "Stage2RepairController":
        # `**kwargs` 原样透传给 `__init__`，`effective_config` 也走这条路
        # （审计 #1：溶剂腿的 run_dir 底下没有 run_provenance.json）。
        """同一**物理 stage** 的控制器（把 `vanishing` / `vanishing_2` / … 合起来）。

        🔑 [2026-09-11 老板定案] **Segment 不是独立的 stage。**
        先前每个段各起一个控制器各判一次 ⟹ 同一个物理 stage 出两个互相矛盾的动作，
        而且 `vanishing_2` 缺 win0 会被判成"缺窗口"，其实 win0 在段 1 里是好的。

        正确顺序的第一步就是它：

            **聚合同一物理 stage 的全部 Segment**
            → 找 earliest unresolved window
            → 检查该窗口动作是否有预算
            → 按证据选择动作
            → 只有完整前缀全部 eligible，才运行下一个缺失窗口

        本类只在 `read()` 上分叉：聚合视图逐窗取**最新有产物的那个段**的状态
        （段号大的更新），其余（路径版本链、stage 结果、join、f_k 探针）本来就是
        全局的，仍从基准 checkpoint 读。
        """
        obj = cls(run_dir, stage_base, stage_type, **kwargs)
        obj._aggregate_segments = True
        obj._stage_base = str(stage_base)
        return obj

    def _segment_stage_names(self) -> List[str]:
        """同一物理 stage 的所有段目录名，按**段号**升序（基准段在最前）。

        [审计 #63] 「什么算一个段」原来在三处各写一遍、规则互不一致
        （一处不过滤空壳 + 字符串序，一处过滤空壳 + 整数序，一处用
        `rsplit("_",1)[-1]` 会把 rewindow 目录误判成段）。现在段号解析统一走
        `segment_index_of_dir()`，排序一律按整数；**是否过滤空壳**由调用点决定，
        两个调用点的理由分别写在各自位置：
          · 本函数（合并视图）**要过滤** —— 空转循环每轮建一个新段目录，真机
            rep2 留下 39 个只有占位文件、一份 convergence 都没有的目录；把它们
            当采样段会让每次 read() 白扫一遍，也会让"证据来自哪个段"被搅浑。
          · `existing_segment_names`（写路径版本记录用）**不过滤** —— 它记的是
            「变更前盘上有哪些段目录」，那是一份事实清单，不是证据来源。
        """
        base = getattr(self, "_stage_base", self.stage_name)
        stage_dir = os.path.join(self.run_dir, base)
        names = []
        for d in glob.glob(stage_dir + "*"):
            if not os.path.isdir(d):
                continue
            n = segment_index_of_dir(d, stage_dir)
            if n is None:
                continue
            if n > 1 and not glob.glob(os.path.join(
                    d, f"dual_window_*_{self.stage_type}_convergence.json")):
                continue          # 空壳段不算段（理由见 docstring）
            names.append((int(n), os.path.basename(d)))
        return [nm for _i, nm in sorted(names)]

    def read_aggregated(self) -> Dict[str, Any]:
        """把同一物理 stage 的全部段合成**一个**视图。

        合并规则（刻意保守）：
          · 逐窗取**段号最大且有 convergence 产物**的那份；都没有就取有任何产物的最新一份。
          · `missing_windows` = 布局里有、但**所有段**都没有的窗口。
            （先前按单段判，于是段 2 缺 win0 被当成缺窗口 —— 其实段 1 有。）
          · 路径版本链 / stage 结果 / join / f_k 探针是全局的，从基准 checkpoint 读。
        """
        names = self._segment_stage_names() or [self.stage_name]
        views = []
        for nm in names:
            sub = Stage2RepairController(
                self.run_dir, nm, self.stage_type,
                min_states_per_window=self.lo, max_states_per_window=self.hi,
                max_path_insertions=self.max_path_insertions,
                allow_untrusted_stage_results=self.allow_untrusted,
                # 见 __init__ 里 `_explicit_config` 的说明：不传这份，
                # 合并视图的 cap / 块大小 / 分窗判据全部退回默认。
                effective_config=getattr(self, "_explicit_config", None),
            )
            # 🔑🔑 [审计 #25] **段级子控制器必须知道自己的物理 stage 名。**
            # 不设 `_stage_base`，`_read_stage_result()` 就用 `self.stage_name`
            # （= `vanishing_3`）去格式化候选名 ⟹ 找 `stage2_vanishing_3_*.json`，
            # 而写侧硬编码写的是 `stage2_vanishing_*` ⟹ 两个候选**全 miss**，
            # 落到 mtime 兜底 glob，CTL-01 的「路径版本必须匹配当前布局」这层隔离
            # 对所有非基准段是关着的。段号只改窗口产物目录，stage 结果是全局的。
            sub._stage_base = getattr(self, "_stage_base", self.stage_name)
            views.append((nm, sub.read()))
        base_view = dict(views[0][1])

        # 🔑 [2026-09-12] **先按 λ 身份剔除过期证据，再谈段号。**
        # 原来只按"段号最大且有 convergence"取，等于拿**段号当新旧的代理**。
        # 那在"段只往前开、基准段从不重跑"时成立；但循环现在会往基准段写
        # （补采按证据来源路由）、布局变更时整段重跑 —— 基准段反而常常最新。
        # 真机后果：win4 在基准段已经 43.25 通过，却被 45 分钟前废弃 Epoch 里
        # 那份 1.62 盖住，于是循环接着去"修"一个已经好了的窗口。
        #
        # 判据不是 mtime（脆弱），是**语义身份**：这份证据是在哪套 λ 上产出的。
        # 对不上当前布局 ⟹ 它描述的是另一个窗口几何，直接丢弃。
        _path_now = self._read_path()
        _cur_lam = list(_path_now.get("lambdas_vdw") or [])
        _cur_rng = [tuple(r) for r in (_path_now.get("window_ranges") or [])]

        # 🔑 [2026-09-17] 判定搬进模块级 `classify_layout_evidence()`，与单段视图
        # **同一份**。先前它是本函数的一个闭包，于是单段视图整套判定都没有。
        _cls = classify_layout_evidence(
            [(nm, v.get("windows") or []) for nm, v in views], _path_now)
        unverifiable = _cls["unverifiable"]   # 有证据但核不了布局（缺 λ）
        out_of_range = _cls["out_of_range"]   # 当前布局里根本没有的窗口号

        merged: Dict[int, Dict[str, Any]] = {}
        provenance: Dict[int, str] = {}
        stale: Dict[int, List[str]] = _cls["stale"]
        stale_tpl: Dict[int, Dict[str, Any]] = {}
        for nm, v in views:          # 段号升序 ⟹ 后面的覆盖前面的
            for w in v.get("windows") or []:
                i = int(w["window_idx"])
                # 只有 `STALE` 不参与合并；`OUT_OF_RANGE` / `UNVERIFIABLE` 照旧合并
                # 并已被 helper 记账（判死会重现 win4 占位记录死锁）。
                if _cls["verdict_by"].get((nm, i)) == "STALE":
                    stale_tpl[i] = w          # 留作占位模板：键齐全
                    continue
                if i not in merged or w.get("has_convergence"):
                    merged[i] = dict(w, segment=nm)
                    provenance[i] = nm
        # 🔑🔑 [审计 #18，2026-09-14] **`stale` 表必须清除，不然 DONE 永远到不了。**
        #
        # 先前两张表互不排斥：窗口在新段里**重采成功并进了 `merged`** 之后，它在旧段
        # 留下的那条过期证据仍然躺在 `stale` 里。于是插过一次 λ 之后
        # `view["stale_layout_evidence"]` **恒非空**，而分支 0a 的 `DONE` 与
        # `_evidence_status` 的 `CONVERGED` 都要求它为空 ⟹ **两条收敛路径被永久封死**，
        # 一个已经修好的 stage 每次 resume 都会重新开修复动作烧 GPU。
        # `stale` 的语义只能是「这个窗口**当前没有任何有效证据**」——
        # 有了有效证据就不再是 stale，旧段那份只是历史。
        # （`stale_layout_evidence_only` 占位记录的语义不变：它恰恰是"清不掉"的那些。）
        for _i in list(stale):
            if _i in merged:
                del stale[_i]
                stale_tpl.pop(_i, None)
        # ⚠️ **证据全过期 ≠ 窗口不存在。** 直接把它从列表里剔掉，窗口就"消失"了，
        # 决策会越过它去处理下游 —— 正是要防的那个失败模式（跳过前面的窗口）。
        # 正确语义是「这个窗口目前**没有有效证据**」：留一条占位记录，
        # `self_verdict=None` ⟹ 三态分类判 UNKNOWN ⟹ 动作是**产出证据**，
        # 而且它仍然占着自己的位置，earliest 排序不会跳过它。
        for i, segs in stale.items():
            if i in merged:
                continue
            # 用被剔掉的那份当模板（键齐全），把**所有证据字段清空** ——
            # 留结构、不留结论。
            tpl = dict(stale_tpl.get(i) or {})
            for k in list(tpl):
                if k not in ("window_idx",):
                    tpl[k] = None
            tpl.update({
                "window_idx": int(i),
                "segment": None,
                "has_convergence": False,
                "has_warmup_failure": False,
                "stale_layout_evidence_only": True,
                "stale_segments": list(segs),
            })
            merged[i] = tpl
            provenance[i] = "(证据已被布局变更作废)"
        windows = [merged[i] for i in sorted(merged)]

        expected = base_view.get("n_windows_expected")
        missing = (
            [i for i in range(int(expected)) if i not in merged] if expected else []
        )
        # [审计 #58] 物理窗口与子窗分两个键；单段视图已经拆好，这里只合并。
        skipped = sorted({
            int(x) for _nm, v in views for x in (v.get("skipped_windows") or [])
        })
        skipped_units = sorted({
            str(x) for _nm, v in views
            for x in (v.get("skipped_sampling_units") or [])
        })
        # [2026-09-12] 同一窗口**跨段**的主验收量轨迹。cap 分支早就写明
        # 「块后立即用 N_eff/g 的边际增长判 —— 边际停滞/下降则停止同分布加帧」，
        # 但判这件事需要的历史此前根本没进视图，于是规则一直是死的。
        # ⚠️ **布局变更会作废这段历史。** 插 λ / 拆末窗之后，旧段描述的是**另一个
        # 窗口几何**，它们的 min N_eff/g 不能再拿来判"加帧有没有用"。真机 15:59
        # 实证：第一次插完还没产出任何新证据，判据读着旧轨迹又插了一次，
        # 两次把末窗顶到 K=6 越界炸掉。
        # 当前路径版本记录里的 `segments_before_change` 就是变更前已有的段。
        # [2026-09-17] 判定搬进 `layout_change_invalidated_segments()`，与单段视图
        # 同一份 —— 它是全局事实，两个构造器算出来必须一样。
        _stale_segs = set(self.layout_change_invalidated_segments())
        # ⚠️ [契约 B] `_production_blocks_*` 现在返回 `{"by_window", "by_unit"}`
        # **两本账**，必须在这里拆开放进两个视图键 —— 直接把整个双层字典塞进
        # `production_blocks_by_window`，消费侧的 `.get(int(w_idx))` 会恒取到
        # None，块数硬上限对**所有**窗口静默失效（比原 bug 更糟）。
        _agg_units = base_view.get("sampling_units") or []
        _agg_same = self._production_blocks_ledger(
            windows, (_path_now or {}).get("version"), units=_agg_units)
        # 🔑 [BUD-03] 跨段累计的块数（硬上限用），与上面那本同源不同过滤。
        _agg_all = self._production_blocks_total_by_window(
            windows, (_path_now or {}).get("version"), units=_agg_units)
        blocks_by_window = _agg_same["by_window"]
        blocks_by_unit = _agg_same["by_unit"]
        blocks_total_by_window = _agg_all["by_window"]
        blocks_total_by_unit = _agg_all["by_unit"]

        # 🔑🔑 [S2-C，2026-09-14] **边际增长判据需要两个可比点，而按段建史给不出。**
        #
        # 原来这里一个采样段只贡献**一个**点 ⟹ 单段窗口永远只有 1 个点，
        # `len(h) >= 2` 从不成立 ⟹「同分布加帧已被证伪就别再加」这道唯一的刹车
        # **对单段窗口从不触发**（而单段正是最常见的情形）。布局一变旧段被标 stale
        # 丢掉，多段窗口也会退回 1 个点。
        #
        # 补法是读**已经在写的**逐轮快照（`stage2_autonomous_history.json` 的
        # `snapshot`，主循环每轮决策时落的盘），它天然是**逐块**的。
        # 可比性由两个身份保证，缺一不可：
        #   · `path_version` 相同 —— 布局变了跨度就变了，前后不是同一个量；
        #   · `segment` 相同     —— 换段 = 换了一份 f_k，重加权口径不同。
        # 排序按 `production_steps`（采样量单调），去重也按它。
        support_history: Dict[int, List[Any]] = {}
        _hist_pts: Dict[int, Dict[Any, Any]] = {}

        def _add_pt(i, seg, steps, value, label):
            # ⚠️ 可比性由 (segment, production_steps) 共同决定：**换段 = 换 f_k**，
            # 不同段的 min N_eff/g 不是同一把尺子，不能放进同一条边际增长曲线。
            # 下面排序/比较只在**同一段**内部才有意义 —— 混段的点会让
            # "加帧有没有用"这个判据读出一个根本不存在的趋势。
            if value is None or seg is None:
                return
            key = (seg, int(steps or 0))
            _hist_pts.setdefault(int(i), {}).setdefault(
                key, (int(steps or 0), label, float(value))
            )

        for nm, vv in views:
            if nm in _stale_segs:
                continue          # 变更前的几何，不参与边际增长判定
            for w in vv.get("windows") or []:
                _add_pt(w["window_idx"], nm, w.get("production_steps"),
                        w.get("min_n_eff_over_g"), nm)

        try:
            _auto = self._json(os.path.join(
                self.path_checkpoint_dir, "stage2_autonomous_history.json")) or {}
            _cur_pv = (_path_now or {}).get("version")
            for _it in (_auto.get("iterations") or []):
                if _cur_pv is not None and _it.get("path_version") != _cur_pv:
                    continue      # 别的布局下的点，不可比
                for _sn in (_it.get("snapshot") or []):
                    _seg = _sn.get("segment")
                    if _seg in _stale_segs:
                        continue
                    _add_pt(_sn.get("window_idx"), _seg,
                            _sn.get("production_steps"),
                            _sn.get("min_n_eff_over_g"),
                            f"{_seg or 'base'}@{int(_sn.get('production_steps') or 0)}")
        except Exception:          # noqa: BLE001 —— 历史读不到只是少几个点，不阻断决策
            pass

        for i, pts in _hist_pts.items():
            # **按段分组，只在段内按采样量排序**；把各段首尾串起来会把
            # "换了一份 f_k" 伪装成 "加帧的效果"。
            # 🔑 [CTL-06，2026-09-14] 判据来源必须是**当前段**（这个窗口证据实际
            # 所在的那一段，也就是最新那份 f_k），**不是"点数最多的段"** ——
            # 旧段点数多时会拿旧 f_k 的趋势去裁决新段该不该继续加帧。
            # 当前段不足两点就不判（`len(h) >= 2` 自然不成立），不许拿别的段凑。
            by_seg: Dict[Any, List[Any]] = {}
            for (seg, steps), (_st, label, value) in pts.items():
                by_seg.setdefault(seg, []).append((steps, label, value))
            _cur_seg = (merged.get(int(i)) or {}).get("segment")
            best = by_seg.get(_cur_seg)
            if best is None:
                # 读不到当前段（证据全过期等）⟹ 宁可不判，也不拿别的段代替。
                best = []
            support_history[i] = [
                (label, value) for _steps, label, value in sorted(best, key=lambda t: t[0])
            ]
        # 跨段实际生产消耗：每个窗口在**每个段**里都烧过 GPU，只取胜出段会漏账。
        # 🔑 [CTL-04②，2026-09-14] **消耗未知不得记作零。**
        # 先前 `int(... or 0)` 把"读不到步数"和"确实是 0 步"压成同一个数 ——
        # 与"上限未知 ≠ 零"是同一个错、方向相反：一个会虚报余量、一个会虚报耗尽。
        # 读不到就**登记为未知**，账本据此把整个 stage 的用量标成 unknown，
        # 而不是悄悄少算一块。
        _per_win_all_seg: Dict[int, int] = {}
        _unknown_usage: List[Any] = []
        for _nm, _vv in views:
            for _w in (_vv.get("windows") or []):
                _i = int(_w["window_idx"])
                _ps = _w.get("production_steps")
                if _ps is None:
                    # 有这个窗口的证据、却读不到它烧了多少步 ⟹ 未知，不是 0。
                    if _w.get("has_convergence"):
                        _unknown_usage.append({"window_idx": _i, "segment": _nm})
                    continue
                _per_win_all_seg[_i] = _per_win_all_seg.get(_i, 0) + int(_ps)
        _unit_used_raw = (
            (base_view.get("_production_budget_inputs") or {}).get("unit_used") or {})
        for _uid, _v in _unit_used_raw.items():
            if _v is None:
                _unknown_usage.append({"unit_id": _uid})
        _all_seg_used = (
            sum(_per_win_all_seg.values())
            + sum(int(v) for v in _unit_used_raw.values() if v is not None)
        )

        base_view.update({
            "stage_name": getattr(self, "_stage_base", self.stage_name),
            "aggregated_segments": names,
            "min_n_eff_over_g_history": support_history,
            "production_blocks_by_window": blocks_by_window,
            "production_blocks_by_unit": blocks_by_unit,
            "production_blocks_total_by_window": blocks_total_by_window,
            "production_blocks_total_by_unit": blocks_total_by_unit,
            "max_production_blocks_per_window": int(self.max_blocks_per_window),
            "layout_change_invalidated_segments": sorted(_stale_segs),
            "window_provenance": provenance,
            # 布局变更后作废的逐窗证据（λ 对不上当前布局）。
            "stale_layout_evidence": {int(k): v for k, v in stale.items()},
            # 🔑 [2026-09-15 审计①] 有证据、但**无从核对**它描述的是不是当前布局
            # （那一段的 convergence/ibs_state 没有 λ 列表）。与 `stale` 不同：
            # stale 是"核对过、不匹配"，这个是"核不了"。合并照旧（判死会重现
            # win4 占位记录死锁），但必须**说出来**。
            "unverifiable_layout_evidence": {
                int(k): v for k, v in unverifiable.items()},
            # 🔑 [2026-09-15 审计④] 盘上多出来的窗口号：当前布局里**根本没有**
            # 这个下标（旧布局的遗留产物）。`missing_windows` 只算
            # `range(expected)`、从不查多 ⟹ 先前它一路进 merged 且无人报告。
            "out_of_range_windows": sorted(out_of_range),
            "windows": windows,
            "n_windows_found": len(windows),
            "missing_windows": missing,
            "skipped_windows": skipped,
            # [审计 #58] 两个视图必须给出**同一组键**，否则同一个不变量又变成两份
            # 实现（单段视图只传物理窗口、合并视图双传，下游读哪个全看运气）。
            "skipped_sampling_units": skipped_units,
            # 🔑🔑 [BUD-05，2026-09-14] **未知就写 None，不许写 0。**
            # 先前 `int(... or 0)` 把「账本读不到」写成「余量 0」= 耗尽 —— 而同一个
            # 量在 `_epoch_validation_unaffordable` 里是 fail-closed、在 `decide()`
            # 分支 1e 里是 `left is not None and <=0`（未知 = 有钱）。同一个未知，
            # 三套语义。实测 cyclod_ligand1/rep3 win4 的 ledger 确实读不到，
            # 在这里被显示成 0。规矩（TODO §1）：**未知不是零，两个方向都不是。**
            "per_window_budget_remaining": {
                int(w["window_idx"]): (
                    None if w.get("warmup_steps_left") is None
                    else int(w["warmup_steps_left"])
                )
                for w in windows
            },
            # 🔑 [BUD-05] 「全部耗尽」是个**断言**，账不完整就不成立：
            # 只要有一个窗口的余量未知，就不得宣称全局耗尽（否则是拿少算过的账
            # 去论证一个终止性结论）。
            "all_windows_budget_exhausted": bool(
                windows
                and all(w.get("warmup_steps_left") is not None for w in windows)
                and all(int(w["warmup_steps_left"]) <= 0 for w in windows)
            ),
            # 合并视图的生产账：按**合并后**的窗口 + 全部子窗重算。
            # 跨**所有段**的实际生产消耗（含子窗），不是只算胜出段。
            "production_budget": (lambda _inp, _per=_per_win_all_seg,
                                  _tot=_all_seg_used,
                                  _unknown=list(_unknown_usage): {
                "stage_cap_steps": _inp.get("cap"),
                "cap_source": _inp.get("cap_source"),
                "cap_known": _inp.get("cap") is not None,
                "stage_used_steps": _tot,
                # 有未知消耗时**余量不可知** —— 不许拿一个少算过的数当余量去准入。
                "usage_complete": not _unknown,
                "unknown_usage": _unknown,
                "stage_remaining_steps": (
                    None if (_inp.get("cap") is None or _unknown)
                    else int(_inp["cap"]) - _tot
                ),
                # ⚠️ 逐窗按**所有段**累加，不是只取胜出段 —— 旧段的帧是真烧过的
                # GPU，漏掉它账就永远比实际宽。
                "per_window_used_steps": _per,
                "per_sampling_unit_used_steps": dict(_inp.get("unit_used") or {}),
                "new_ensemble_reserve_steps": int(_inp.get("reserve") or 0),
                # 用量不完整时**不得**宣称耗尽（少算过的账不能拿来判终态）。
                "exhausted": bool(
                    _inp.get("cap") is not None and not _unknown
                    and _tot >= int(_inp["cap"])
                ),
                "production_block_steps": _inp.get("block"),
                "cap_scope": "production_steps_only",
                "note": (
                    "合并视图：逐窗生产步数按**所有采样段**累加（不是只取胜出段），"
                    "子窗另计。cap 只管生产步数，预热/验证走另一本账。"
                ),
            })(base_view.get("_production_budget_inputs") or {}),
        })
        return base_view

    def read(self) -> Dict[str, Any]:  # noqa: F811 —— 覆盖：聚合模式走合并视图
        if getattr(self, "_aggregate_segments", False):
            return self.read_aggregated()
        return self._read_single_stage()

    # ------------------------------------------------------- 影子对账 replay

    # 每类证据在产物里的身份：字段名 → 它来自哪、缺了意味着什么。
    # 历史 run **没有**这些新字段，replay 必须如实标 ABSENT_IN_ARTIFACT，
    # **绝不把缺失当通过**。
    _EVIDENCE_KEYS = (
        ("self_verdict", "窗口自检 verdict（N_eff/g 主验收量）"),
        ("min_n_eff_over_g", "主验收量数值"),
        ("derailment_status", "脱轨两档（单块=疑似/连续两块=确认）"),
        ("warmup_g", "warmup 剖面 g"),
        ("warmup_min_absolute_ess", "warmup 剖面 minESS"),
        ("state_protocol_key", "采样身份（旧 PASS 能不能粘住）"),
    )

    def decision_trace(self, view: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """一次**只读**的决策轨迹：控制器**会**选什么、依据什么、缺什么证据。

        🔑 [2026-09-11] 影子对账用。**不执行动作、不改预算、不写 run** ——
        它复用同一个 `read()` / `decide()`，所以 replay 和生产**不会各判各的**
        （分开实现就等于对账两个不同的东西，那对账本身就没意义了）。

        老板指定的输出口径：run / window / evidence scope+gauge / verdict /
        reason / selected action / target windows / budget view。
        """
        view = view or self.read()
        plan = self.decide(view)
        windows = []
        for w in view.get("windows") or []:
            avail = {}
            for key, what in self._EVIDENCE_KEYS:
                v = w.get(key)
                avail[key] = {
                    "present": v is not None,
                    "value": v,
                    "what": what,
                    # ⚠️ 缺失**不是**通过。历史产物没有新字段是正常的，但那意味着
                    # "这条证据不存在"，不意味着"这条证据判了通过"。
                    "status": "present" if v is not None else "ABSENT_IN_ARTIFACT",
                }
            windows.append({
                "window_idx": w["window_idx"],
                # 别名：生产侧产物用的是 `window_index`，两个名字并存本身就是坑，
                # 这里两个都给，读的人不必记住哪边用哪个。
                "window_index": w["window_idx"],
                "phase": w.get("phase"),
                "verdict": w.get("self_verdict"),
                "verdict_source": w.get("self_verdict_source"),
                "min_n_eff_over_g": w.get("min_n_eff_over_g"),
                "evidence_availability": avail,
                "budget_view": {
                    "warmup_spent": w.get("warmup_steps_spent"),
                    "warmup_cap": w.get("warmup_steps_cap"),
                    "warmup_left": w.get("warmup_steps_left"),
                    "production_steps": w.get("production_steps"),
                },
            })
        # 老板指定的措辞：「产物里有 5 个」不准确 —— 有 warmup 失败状态但没有
        # production convergence 的窗口是"**已知但未完成**"，不是普通完成产物。
        _known = len(view.get("windows") or [])
        _prod_done = sum(
            1 for w in (view.get("windows") or []) if w.get("has_convergence")
        )
        _never = list(view.get("missing_windows") or [])
        n_absent = sum(
            1 for w in windows for e in w["evidence_availability"].values()
            if not e["present"]
        )
        return {
            "protocol_version": STAGE2_CONTROLLER_PROTOCOL_VERSION,
            "readonly": True,
            "run_dir": view["run_dir"],
            "stage_name": view["stage_name"],
            # evidence scope + gauge：口径必须跟着轨迹走，否则对账时分不清
            # "两次判得不同"是策略变了还是口径变了。
            "evidence_scope": "on_disk_artifacts_only",
            "evidence_gauge": {
                "fk_residual": "sampling_states",
                "support_metrics": "gauge_invariant",
            },
            "selected_action": plan["action"],
            "exit": plan.get("exit"),
            "terminal": plan.get("terminal"),
            "routing": plan.get("routing"),
            "target_windows": plan.get("windows"),
            "reason": plan.get("reason"),
            "missing_evidence": plan.get("missing_evidence"),
            "execution_status": plan.get("execution_status"),
            "evidence_status": plan.get("evidence_status"),
            "trust_level": plan.get("trust_level"),
            "budget_view": {
                "per_window_remaining": view.get("per_window_budget_remaining"),
                "all_windows_exhausted": view.get("all_windows_budget_exhausted"),
                "source": view.get("global_budget_source"),
            },
            "path": {
                "version": (view.get("path") or {}).get("version"),
                "events": (view.get("path") or {}).get("events"),
                "tail_repartitions_done": view.get("tail_repartitions_done"),
            },
            "known_window_states": _known,
            "production_complete": _prod_done,
            "never_started": _never,
            "earliest_unresolved_window": self.first_untrusted_window(view),
            "blocked_by_upstream": plan.get("blocked_by_upstream"),
            "windows": windows,
            "n_absent_evidence_fields": n_absent,
        }

    @classmethod
    def replay(
        cls,
        run_dirs: Sequence[str],
        stage_names: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """在一批历史 run 上离线重放决策。**只读、零 GPU。**

        缺字段的老产物必须**跑得起来**并如实标记，不能崩、也不能把缺失当通过 ——
        所以每个 run 单独 try/except，失败只记错误、不打断整批。
        """
        out: List[Dict[str, Any]] = []
        for rd in run_dirs:
            # **按物理 stage 重放，不是按段** —— 段是同一个 stage 的多次采样，
            # 分开判会得到两个互相矛盾的动作（实测：段 2 缺 win0 被判成"缺窗口"，
            # 其实 win0 在段 1 里是好的）。
            names = list(stage_names) if stage_names else ["vanishing"]
            for st in names:
                try:
                    out.append(
                        cls.for_physical_stage(rd, st, **kwargs).decision_trace()
                    )
                except Exception as err:  # noqa: BLE001 —— 一个 run 崩不该毁掉整批
                    out.append({
                        "readonly": True, "run_dir": os.path.abspath(rd),
                        "stage_name": st, "error": repr(err),
                        "selected_action": None,
                    })
        return out

    @staticmethod
    def render_replay(traces: Sequence[Dict[str, Any]]) -> str:
        """人读的 mismatch 基线表。"""
        out = ["决策轨迹 replay（**只读**：不执行动作、不改预算、不写 run）", ""]
        hdr = (f"{'run':<26} {'stage':<13} {'action':<20} {'exit':<28} "
               f"{'exec':<12} {'evidence':<18} {'缺证据':>6}")
        out.append(hdr)
        out.append("-" * len(hdr))
        for t in traces:
            if t.get("error"):
                out.append(f"{os.path.basename(t['run_dir']):<26} "
                           f"{t.get('stage_name',''):<13} **ERROR** {t['error'][:60]}")
                continue
            out.append(
                f"{os.path.basename(t['run_dir']):<26} {t['stage_name']:<13} "
                f"{str(t['selected_action']):<20} {str(t.get('exit') or '-'):<28} "
                f"{str(t.get('execution_status')):<12} "
                f"{str(t.get('evidence_status')):<18} "
                f"{t.get('n_absent_evidence_fields', 0):>6}"
            )
        out.append("")
        out.append("⚠️ 「缺证据」= 该 run 的产物里不存在的证据字段数。"
                   "**缺失不是通过** —— 历史 run 没有新字段是正常的，但那意味着"
                   "「这条证据不存在」，对账时必须当成一类 mismatch，不能读成「判了通过」。")
        return "\n".join(out)

    # -------------------------------------------------------------- 人读

    def render(self, view: Optional[Dict[str, Any]] = None) -> str:
        view = view or self.read()
        plan = self.decide(view)
        p = view["path"]
        out: List[str] = []
        out.append(f"run    : {view['run_dir']}")
        out.append(f"stage  : {view['stage_name']} ({view['stage_type']})  "
                   f"lo/hi={view['min_states_per_window']}/{view['max_states_per_window']}"
                   f"  (来自 {view['config_source']})  可拆区间="
                   f"[{2 * view['min_states_per_window'] - 1},"
                   f"{2 * view['max_states_per_window'] - 1}]")
        out.append(f"插点   : 已用 {view['path_insertions_done']}/"
                   f"{view['path_insertions_budget']}  剩 {view['path_insertions_left']}"
                   "（终身预算，从版本链累计，跨 resume 有效）")
        out.append(f"path   : v{p['version']}  n_states={p['n_states']}  "
                   f"events={p['events'] or '{}'}")
        if view["segment_index"] is not None:
            out.append(f"段     : segment_{view['segment_index']}  "
                       f"ibs_state 读自 {view['checkpoint_dir_used']}")
        line = f"windows: 找到 {view['n_windows_found']}"
        if view["n_windows_expected"]:
            line += f" / 布局 {view['n_windows_expected']}"
        if view["missing_windows"]:
            line += f"   ⚠️ 缺 {view['missing_windows']} —— 缺窗口的总和不是完整 ΔG"
        if view["skipped_windows"]:
            line += f"   ⚠️ 被踢出协方差链 {view['skipped_windows']}"
        # [审计 #58] 子窗用 `unit_id` 打印。先前它们的 solver 索引（10000+）混在
        # `skipped_windows` 里被当窗口号打印出来，人读到一个不存在的"窗口 10000"。
        if view.get("skipped_sampling_units"):
            line += f"   ⚠️ 子窗被踢出协方差链 {view['skipped_sampling_units']}"
        out.append(line)
        out.append(f"stage  : analysis={view['stage_analysis_status']}  "
                   f"precision={view['stage_precision_status']}  "
                   f"ΔG={view['stage_total_delta_G']}  "
                   f"(stage 结果{'已' if view['has_stage_result'] else '**未**'}落盘)")
        if view.get("stage_analysis_incomplete_reasons"):
            out.append("         分析未完整：" +
                       "；".join(view["stage_analysis_incomplete_reasons"]))
        out.append("")
        hdr = (f"{'win':>3} {'K':>2} {'λ跨度':>8} {'phase':<16} {'verdict':<22} "
               f"{'warmup':>15} {'验证':>8} {'生产':>9} {'段':>3} {'帧':>6}  note")
        out.append(hdr)
        out.append("-" * len(hdr))
        for w in view["windows"]:
            note = []
            if w["best_effort_acceptance"]:
                note.append(f"best-effort({w['best_effort_reason']})")
            if w["last_gate_error"]:
                note.append(f"gate={w['last_gate_error']}")
            if w["last_failure_reason"]:
                note.append(f"fail={w['last_failure_reason']}")
            if w["has_warmup_failure"] and not w["has_convergence"]:
                note.append("只有 warmup_failure，未产出 convergence")
            if w.get("derail_at_trajectory_fraction") is not None:
                note.append(
                    f"f_k 在 {w['derail_at_trajectory_fraction']:.0%} 轨迹处脱轨"
                    "（边际 N_eff 断崖）"
                )
            if w.get("min_n_eff_over_g") is not None:
                note.append(f"N_eff/g={float(w['min_n_eff_over_g']):.2f}")
            if w.get("self_verdict") and w.get("self_verdict") != "ANALYSIS_ELIGIBLE":
                note.append(
                    f"{w.get('self_verdict')}"
                    f"(N_eff/g={w.get('min_n_eff_over_g')}, "
                    f"n_decorr={w.get('self_n_frames_decorrelated')})"
                )
            wm = "-" if w["warmup_steps_spent"] is None else str(w["warmup_steps_spent"])
            if w["warmup_steps_cap"]:
                wm = f"{wm}/{w['warmup_steps_cap']}"
            span = "-" if w["lambda_span"] is None else f"{w['lambda_span']:.4f}"
            out.append(
                f"{w['window_idx']:>3} {str(w['n_states'] or '-'):>2} {span:>8} "
                f"{w['phase']:<16} {w['verdict']:<22} {wm:>15} "
                f"{('-' if w['frozen_validation_steps'] is None else w['frozen_validation_steps']):>8} "
                f"{('-' if w['production_steps'] is None else w['production_steps']):>9} "
                f"{str(w['n_production_segments'] or '-'):>3} "
                f"{('-' if w['n_frames'] is None else w['n_frames']):>6}  " + "; ".join(note)
            )
        _fp = view.get("fk_probe") or {}
        if _fp:
            out.append("")
            out.append(
                f"f_k 探针 : verdict={_fp.get('verdict')} "
                f"建议重标定={_fp.get('recalibration_recommended_windows')} "
                f"（阈值 {_fp.get('min_adjacent_shift_kJ_mol')} kJ/mol 相邻位移）"
            )
            for r in (_fp.get("windows") or []):
                if r.get("max_adjacent_shift_kJ_mol") is None:
                    out.append(f"           w{r.get('window')}: 跳过 {r.get('skipped')}")
                else:
                    out.append(
                        f"           w{r.get('window')}: 相邻位移="
                        f"{float(r['max_adjacent_shift_kJ_mol']):.3f} "
                        f"绝对位移={float(r.get('max_abs_shift_kJ_mol') or 0):.3f} "
                        f"n_used={r.get('n_frames_used')} "
                        f"规范对账sd={r.get('f_k_consistency_sd_kJ_mol')}"
                    )
        _rank = plan.get("warmup_weakness_ranking") or {}
        if _rank.get("by_metric"):
            out.append("")
            out.append("warmup 剖面（**只报告**：哪个窗口在生产前就显出弱相）"
                       " —— 剖面→预算的映射未定，不据此做任何事")
            wh = (f"{'win':>3} {'g':>8} {'n_used':>7} {'minESS':>8} "
                  f"{'max_adj':>8} {'门槛':>6}")
            out.append(wh)
            out.append("-" * len(wh))
            for w in view["windows"]:
                f3 = lambda x, wd, pr=2: ("-".rjust(wd) if x is None
                                          else f"{float(x):.{pr}f}".rjust(wd))
                out.append(
                    f"{w['window_idx']:>3} {f3(w.get('warmup_g'), 8)} "
                    f"{str(w.get('warmup_n_frames_used') or '-'):>7} "
                    f"{f3(w.get('warmup_min_absolute_ess'), 8)} "
                    f"{f3(w.get('warmup_max_adjacent_delta_kJ_mol'), 8, 3)} "
                    f"{f3(w.get('warmup_gate_threshold_kJ_mol'), 6, 1)}"
                )
            out.append(f"弱→强（三口径名次和，不加权）: {_rank.get('weakest_overall')}")
        if view.get("joins"):
            out.append("")
            out.append("join λ 两侧支撑（相邻窗口对共享的那一个 λ 各自的 rawESS）"
                       " —— **只报告，不参与放行**")
            out.append("           ⚠️ 只证明**共享态自身**采得好；缺跨窗交叉能量 ⟹ "
                       "下游强支撑进不了上游 local MBAR，**不构成上游 PASS**")
            jh = (f"{'join λ':>8} {'上游':>5} {'rawESS':>9} {'g':>7} {'top1%':>6}  "
                  f"{'下游':>5} {'rawESS':>9} {'g':>7} {'top1%':>6}  {'下/上':>7}  判读")
            out.append(jh)
            out.append("-" * len(jh))
            for j in view["joins"]:
                u, d = j.get("upstream") or {}, j.get("downstream") or {}
                f2 = lambda x, w, p=1: ("-".rjust(w) if x is None
                                        else f"{float(x):.{p}f}".rjust(w))
                r = j.get("downstream_over_upstream_raw_ess")
                # ⚠️ 判读只写**事实**，绝不写"可以兜底/可以参考下游"。
                # 老板纠正：只保存了共享的那一个 λ、没有跨窗交叉能量 ⟹ 下游的强支撑
                # **进不了上游的 local MBAR**，改善不了上游的相对自由能。所以
                # 「上游端点弱、下游强」**不构成**上游可以 PASS —— 上游保持
                # PENDING / INSUFFICIENT。（这句以前写成"端点值可参考下游"，是误用。）
                verdict = "-"
                if r is not None:
                    verdict = ("上游端点弱、下游强 %.1f×（仅诊断：**不**构成上游放行）" % r
                               if r > 2.0 else
                               "下游端点弱、上游强 %.1f×（仅诊断）" % (1.0 / r) if r < 0.5 else
                               "两侧相当（仅诊断）")
                out.append(
                    f"{f2(j.get('join_lambda_vdw'), 8, 4)} "
                    f"{('w%d' % j.get('upstream_window', -1)):>5} "
                    # 🔑 [2026-09-14] **列名是 `g`，就必须读 g 那个键。**
                    # `join_lambda_two_sided_support` 现在落 `statistical_inefficiency_g`
                    # = g；`tau_int` 保留为别名但**值已改成真正的 τ_int=(g−1)/2**
                    # ⟹ 照旧读它会静默打印出**一半**的数字，表头却还写着 g。
                    # 旧产物没有新键时才回退到 `tau_int`（那时它还是 g）。
                    f"{f2(u.get('raw_ess'), 9)} "
                    f"{f2(u.get('statistical_inefficiency_g', u.get('tau_int')), 7, 2)} "
                    f"{f2(u.get('top1pct_weight'), 6, 3)}  "
                    f"{('w%d' % j.get('downstream_window', -1)):>5} "
                    f"{f2(d.get('raw_ess'), 9)} "
                    f"{f2(d.get('statistical_inefficiency_g', d.get('tau_int')), 7, 2)} "
                    f"{f2(d.get('top1pct_weight'), 6, 3)}  "
                    f"{f2(r, 7, 2)}  {verdict}"
                )
        out.append("")
        out.append("=" * 78)
        out.append(f"下一步动作 : {plan['action']}" + (f"   出口: {plan['exit']}" if plan["exit"] else ""))
        out.append(f"涉及窗口   : {plan['windows'] or '-'}")
        out.append(f"理由       : {plan['reason']}")
        if plan["missing_evidence"]:
            out.append(f"缺什么证据 : {plan['missing_evidence']}")
        out.append(f"结构动作   : 可行={plan['feasible_structural_actions']}")
        for k, v in plan["infeasible_structural_actions"].items():
            out.append(f"             {k} 不可行: {v}")
        out.append(f"三维状态   : execution={plan['execution_status']}  "
                   f"evidence={plan['evidence_status']}  trust={plan['trust_level']}")
        return "\n".join(out)


TAIL_REPARTITION_PROTOCOL_VERSION = 1





def _pw_or_last(u):
    """子窗的父窗号；**读不出来就排到最后**，不冒充 window 0。

    [2026-09-17] 先前全仓五处写 `int(u.get("parent_window") or 0)`。
    「未知不是零」在这里有两个后果：排序上插到队首、块账上记到 window 0
    （它是本方法的必然坏窗口、配额最金贵）。目前执行器两处都写
    `int(window_idx)` 所以不会触发，但那是**静默**错，出事时没有迹象。
    """
    _v = u.get("parent_window")
    return 1 << 30 if _v is None else int(_v)


def _attr_of(view, w):
    """物理窗口的**三态**归因。与子窗那一侧、与 O1 同一份实现。

    🔑🔑 [2026-09-17 P0] **三态必须贯穿到动作选择层。**
    `support_failure_is_skew()` 把三态压成布尔（`== STRUCTURAL`），于是
    `not is_skew` 会把 `UNKNOWN` 一起算成"样本量类"⟹ 补帧。
    那只是把错误从「错误授权 D3」换成「错误授权补帧」，语义仍然在下一层丢掉。
    """
    return support_failure_attribution(
        w.get("self_verdict"), w.get("self_verdict_source"),
        n_decorrelated=w.get("self_n_frames_decorrelated"),
        min_frames=w.get("self_min_frames"),
        min_n_eff_over_g=w.get("min_n_eff_over_g"),
        n_eff_over_g_target=w.get("self_n_eff_over_g_eligible"),
        frames_headroom=frames_growth_headroom(
            view, int(w["window_idx"])),
        # [2026-09-18] η≤1 硬上界要的两个读数（写侧早就在落盘，此前无人判）
        n_eff_input_n_frames=w.get("bottleneck_n_eff_input_n_frames"),
        bottleneck_g=w.get("bottleneck_g"))


def _is_skew(view, w):
    # 帧数一并传进去：判据自己再验一次，不押在写侧的覆盖顺序上。
    # [2026-09-16] 再加「加帧的射程」：比值 × 剩余配额够得着门 ⟹ 这是
    # 样本量问题而不是偏斜，该补帧就补帧（见 `support_failure_is_skew`）。
    return support_failure_is_skew(
        w.get("self_verdict"), w.get("self_verdict_source"),
        n_decorrelated=w.get("self_n_frames_decorrelated"),
        min_frames=w.get("self_min_frames"),
        min_n_eff_over_g=w.get("min_n_eff_over_g"),
        n_eff_over_g_target=w.get("self_n_eff_over_g_eligible"),
        frames_headroom=frames_growth_headroom(
            view, int(w["window_idx"])),
        # [2026-09-18] η≤1 硬上界要的两个读数（写侧早就在落盘，此前无人判）
        n_eff_input_n_frames=w.get("bottleneck_n_eff_input_n_frames"),
        bottleneck_g=w.get("bottleneck_g"))


def _pick(earliest, items, key: Optional[str] = "window_idx"):
    """**只路由 earliest 这一个窗口**：它不在本分支的候选里就不触发。

    ⚠️ [审计 #22，2026-09-14] **这里刻意不放宽。** `earliest is None` 时
    返空看起来把一批分支（1b/2/3/3a/4/5/0b/5a-*/5b/6）都打掉了，但那些分支
    **没有一条写过自己「全窗合格时我不该跑」的前提** —— 它们靠的就是这个
    返空。全局放宽等于一次性拆掉 11 条分支的隐式前提，而它们各自该不该在
    「全窗合格」盘面上触发，没有任何一处有成文依据。
    #22 的真实失效面（全窗 UNKNOWN + 生产帧没攒够 ⟹ 只发得出空转的
    `ANALYZE`）改在**分支 5 里定点修**，理由写在那里。
    """
    if earliest is None:
        return []
    idxs = [int(i) if key is None else int(i[key]) for i in items]
    return [earliest] if earliest in idxs else []


def _frames_admission(view, _blocks, _blocks_total, _blocks_total_u, _blocks_u, _max_blocks, w_idx, unit_id=None, *, hard_cap_only=False):
    """能不能再批一块帧。返回 `(True, None)` 或 `(False, 原因)`。

    带 `unit_id` ⟹ 查**子窗那本账**（父窗的账描述的不是它）。

    🔑🔑🔑 [S2-N，2026-09-18 真机] **`hard_cap_only` 给「非补帧」的花块动作用。**

    块账的写侧（`_BLOCK_CHARGING_ACTIONS`）有 4 个动作，而这道闸先前只挂在
    `RUN_PRODUCTION` 上 ⟹ 重标定 / 探针 / 临时生产**扣配额却从不过闸**。
    真机 `cmet_ligand2/rep3` w4：2 块补帧 + 3 块 `RECALIBRATE_FK` = **5 块**（上限 4）；
    `cyclod_ligand3/rep3` w1 同形。它们吃掉的正是那个窗口稍后重采要用的配额。

    但**三道闸不能整套套给它们**，只有第一道（块数硬上限）适用：
      · 第二道「上一块补帧没带来增益」与第三道「剩余配额够不着门」问的都是
        **同分布加帧还有没有用**；而 `RECALIBRATE_FK` 恰恰是「加帧没用」时的
        **对症动作** —— 拿"加帧没用"去否决它，正好把唯一对症的那条路掐掉，
        方向完全反了。
      · 块数硬上限是**资源账**（烧掉的 GPU 不会回来），与动作是什么无关 ⟹
        四个动作一视同仁。
    """
    if unit_id:
        _who = f"子窗 `{unit_id}`"
        rows = _blocks_u.get(str(unit_id)) or []
        rows_all = _blocks_total_u.get(str(unit_id)) or []
        _dim = "子系综 identity"
    else:
        _who = f"窗口 {int(w_idx)}"
        rows = _blocks.get(int(w_idx)) or []
        rows_all = _blocks_total.get(int(w_idx)) or []
        _dim = "段"
    if len(rows_all) >= _max_blocks:
        return False, (
            f"{_who}在当前布局下**跨全部采样段**已经批过 {len(rows_all)} 块帧"
            f"（上限 {_max_blocks}；换段/换 Epoch 不重置配额）⟹ **不再自动补**。"
            f"逐块判据量（{_dim}, 求解器去相关帧数）="
            f"{[(r.get('segment'), r.get('solver_n_decorrelated')) for r in rows_all]}。"
        )
    if hard_cap_only:
        # 非补帧的花块动作：只受资源账约束，不受"加帧还有没有用"那两道约束。
        return True, None
    # [审计 #47] 边际增益刹车改用共享的 `marginal_gain_stalled()`：
    # 至少 3 个点、基准取前面各点的中位数、只有明显低于基准才算被证伪。
    # 旧判据 `max(fin[1:]) <= fin[0]` 把 `solver_n_decorrelated` 当成随采样
    # 单调增的量（分子 Kish ESS 小 N 偏高、分母 ĝ 在 N≫τ 前还在涨，两个偏差
    # 同向），而基准恰好取偏高最严重的 `fin[0]` ⟹ 对正常窗口误触发。
    _stalled, _diag = marginal_gain_stalled(
        [r.get("solver_n_decorrelated") for r in rows])
    if _stalled:
        return False, (
            f"{_who}的补帧**没有带来实质增益**：求解器侧去相关帧数逐块 "
            f"{_diag['series']}，基准（除末点外各点的中位数）="
            f"{_diag['baseline_median_of_earlier_points']:.1f}，"
            f"末点 {_diag['last']:.1f}（比值 "
            f"{_diag['last_over_baseline']:.2f} < 刹车阈值 "
            f"{_diag['stall_ratio']}）。"
            f"同期 min N_eff/g 逐块="
            f"{[r.get('min_n_eff_over_g') for r in rows]}"
            "（去相关帧数是 N/ĝ 那一族：它不涨可能是分子没涨，也可能是分母 ĝ "
            "还在随样本上涨 —— 两种都说明同分布再加一块治不了）⟹ "
            "**停止自动补帧**。"
        )
    # 🔑🔑 [2026-09-17，DEAD_LINES §3.1+§3.2] **第三道：剩余配额的乐观上界
    # 还够不够得着门。**
    #
    # `n_eff_over_g_reachable_by_frames` 的 docstring 自己写着它是**上界**、
    # 「`True` 只表示值得一试」，并且「真正的刹车是事后的
    # `marginal_gain_stalled()` —— 两者一前一后，**缺一不可**」。
    # 而那个刹车的判据是 `末点 < 前面各点中位数 × 0.9`：**单调上升但渐近在
    # 门以下的序列永不触发**。真机 cyclod_ligand1_outer/rep1 win0 逐块
    # `4.459 / 4.553 / 5.287`（门 10）⟹ 判「还在涨」、刹车不响，按 +0.41/块
    # 要 11 块才够。⟹ **这一对少了一半**：进的时候乐观，出的时候没人管。
    #
    # 补的这一道**不引入任何新阈值、也不拟合斜率**（那个量单次噪声 34×，
    # 3 个点拟斜率同样在量噪声），也**不做 `÷1.44` 那种"已观测 g 惩罚"**
    # （`R_now` 本来就是用当前 g 算的 ⟹ 重复扣除；拿过去的比例用于未来
    # ⟹ 那仍是外推，只是不叫"斜率"）—— 它就是**同一个前向判据**，只是拿
    # **现在的读数**和**剩下的配额**再问一次：
    #     当前 ratio × 剩余 headroom < 门  ⟹ 连最乐观的情形都够不着 ⟹ 停。
    # `frames_growth_headroom` 本来就是 `(1+cap)/(1+used)`，用一块少一块，
    # 所以"剩余"是它的原义，不用新写。
    # 上面那三个真机读数代进去：used=1 ⟹ 4.459×2.5=11.1 ≥ 10 放行（此刻确实
    # 还可能够得着）；used=2 ⟹ 4.553×1.67=7.6 < 10 ⟹ **停**。少烧至少一块。
    #
    # ⚠️ **只在明确判 False 时拦**。`None` 是「判不了」（缺数 / 门缺失 /
    # 非有限），一律放行 —— 未知不是"够不着"，这与本文件别处的口径一致。
    # ⚠️ 归因用的判据与 `support_failure_is_skew` **同一个函数**，所以
    # 「加帧够不够得着」在准入侧与归因侧不会给出两个答案。
    _rec = None
    if unit_id:
        _rec = next((u for u in (view.get("sampling_units") or [])
                     if str(u.get("unit_id")) == str(unit_id)), None)
    else:
        _rec = next((w for w in (view.get("windows") or [])
                     if int(w["window_idx"]) == int(w_idx)), None)
    # ⚠️⚠️ [2026-09-17] **帧数没到下限时，射程闸不适用。**
    # 射程算的是 `N_eff/g` 这道**后一道**判据；而一个去相关帧数还没到
    # `min_frames` 下限的窗口，连被求解器尝试的资格都没有 —— 它那个 ratio
    # 是从 7 帧算出来的，**本来就没有意义**，拿它否决补帧等于用一个不可信
    # 的读数掐掉唯一的出路。实测：被求解器跳掉的 w3（7 帧 < 10、归因
    # `SAMPLE_SIZE` 硬证据）被这道闸拒绝补帧 ⟹ 退役 ⟹ 路由跑去修一个
    # 下标更小、却没被跳的窗口，正是「跳窗优先」这条要防的反面。
    if (_rec is not None
            and support_failure_attribution(
                _rec.get("self_verdict"), _rec.get("self_verdict_source"),
                n_decorrelated=_rec.get("self_n_frames_decorrelated"),
                min_frames=_rec.get("self_min_frames"),
            ) == SUPPORT_FAILURE_SAMPLE_SIZE):
        return True, None
    if _rec is not None:
        _hr = frames_growth_headroom(
            view, None if unit_id else int(w_idx),
            unit_id=str(unit_id) if unit_id else None)
        _ratio = _rec.get("min_n_eff_over_g")
        # ⚠️ 这里原来还有一个 `or _rec.get("n_eff_over_g_target")` 候补别名，
        # 而那个键**窗口侧与子窗侧都没有任何人写**（唯一写侧的键名是
        # `self_n_eff_over_g_eligible`）⟹ 纯死分支，删掉，别留着假装有兜底。
        _target = _rec.get("self_n_eff_over_g_eligible")
        if n_eff_over_g_reachable_by_frames(
                _ratio, _target, _hr) is False:
            return False, (
                f"{_who}把**剩余补帧配额全花掉**也够不着门："
                f"当前 min N_eff/g={_ratio}，剩余帧数倍率={_hr:.3g}"
                f"（已批 {len(rows_all)}/{_max_blocks} 块），"
                f"最乐观射程 {float(_ratio) * float(_hr):.3g} < 门 {_target}。"
                "⚠️ 这**不是「证明够不着」**，是「连一个**刻意乐观**的估计都"
                "够不着」：真实关系是 "
                "`R_future = R_now × H × (η_future/η_now) × (g_now/g_future)`"
                "（`η = N_eff/N`），而这里只留 `R_now × H`、**假定 η 与 g 恒定** —— "
                "两个假设实测都朝不利方向走（g 实测 12.25→17.63；η 见 §5.1："
                "250k→1M 让 top1% 0.545→0.762）⟹ 它是**上界**。"
                "上界都够不着就不是「再试一次」的事。"
                "对症动作是缩跨度（插 λ / 有界重窗），不是更多帧。"
            )
    return True, None


def _is_noop(view, _by_idx_for_noop, _by_uid_for_noop, _noop_led, action: str, w_idx, unit_id=None) -> bool:
    """这个动作在**当前盘面**对这个窗口/子窗已经被证明什么也不做。

    执行器写的账（`stage2_noop_actions.json`），指纹由
    `action_noop_fingerprint()` 算 —— 两侧同一份实现。补过帧/换过段
    之后指纹就变了，这条记录自动失效、动作重新可选。

    🔑🔑 [审计 #9，契约 A] **key 有两套命名空间，必须都查。**
    执行器给子窗动作写的是 `f"{action}:unit:{unit_id}"`，而这里先前**只查**
    `f"{action}:{idx}"` ⟹ 子窗的 no-op 账**全仓无人读**，写了等于没写；
    反过来执行器的通用 no-op 检测只要 `plan["unit_id"]` 非空就一律走 unit
    分支 ⟹ 那一轮连普通 key 也不写。两边一起漏 = 子窗的 no-op 刹车从来
    没生效过。带 `unit_id` 时**优先查 unit key，并用该 unit 自己的记录算指纹**
    （父窗的步数是冻结的，拿它算等于把"补过帧"这一维关掉）。
    """
    # 🔑 [2026-09-15] `action_noop_fingerprint` 返回 None = **这个盘面
    # 没有可比身份**（生产步数读不到）。None 不匹配任何东西 —— 包括
    # 台账里那条同样是 None/`?` 的旧记录。否则「从未生产过」的窗口
    # 指纹恒定，no-op 记录在结构上永不失效（真机 win4 死锁的第二环）。
    def _match(rec, record) -> bool:
        if not isinstance(rec, dict):
            return False
        fp = action_noop_fingerprint(record, view.get("path_version"))
        return fp is not None and rec.get("fingerprint") == fp

    if unit_id:
        return _match(_noop_led.get(f"{action}:unit:{unit_id}"),
                      _by_uid_for_noop.get(str(unit_id)))
    return _match(_noop_led.get(f"{action}:{int(w_idx)}"),
                  _by_idx_for_noop.get(int(w_idx)))
