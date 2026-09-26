# =============================================================================
# Stage-2 自治控制器：布局/证据侧的纯函数（S2-D，2026-09-12 从 abfe_pipeline 迁入）
# =============================================================================
# 为什么在这里：这四个是**决策同源**的东西 —— 新 Epoch 预留量、证据来源段解析、
# λ 路径版本读回、段名枚举。判据跟本模块的 `insert_lambda_in_failed_ibs_window` /
# `repartition_tail_from_anchor` / `feasible_repair_actions` 是同一套。跨文件放
# 就是在"同一个不变量的 N 份实现"这条老路上又走一步（设计文档 §6.1）。
#
# ⚠️ **末窗合法化（`_legalize_tail_window`）不在这里**，它留在 `abfe_pipeline`。
# 早先这段注释把它列进来过，是错的：它会 `append_version` 和
# `record_tail_repartition_version` —— **它写盘**。本模块提供的是它用到的那些
# 纯函数（插 λ / 尾段重分 / 版本记录），合法化这个**动作**本身是执行器的事。
#
# ⚠️ **执行器不在这里**：`_run_stage2_autonomous` 主循环留在 `abfe_pipeline`。
# 控制器只读、执行器写盘，这条边界（`test_controller_never_writes_anything`）
# 不因为收拢代码而模糊。`_solve_merged_segments_if_any` 同理留在 pipeline ——
# 它跑的是合并求解，那是求解不是决策。
#
# 这四个都不打日志（迁入前用 `self._log` 的那段是末窗合法化，已留在 pipeline）。

def action_noop_fingerprint(record, path_version) -> Optional[str]:
    """「这个窗口当前的盘面」指纹，用来记住某个动作在此盘面上是 **no-op**。

    返回 ``None`` = **这个盘面没有可比身份**（生产步数读不到）。写侧不得落账、
    读侧不得认作匹配 —— 见下面 `_steps is None` 那段。

    🔑 [2026-09-14 真机] 控制器会反复发一个执行器什么也不做的动作：
    真机两次都是 `RECALIBRATE_FK[0]` 连发 4 次、盘面一字节没变，靠停滞保护
    退出 —— 而退出前**没有**去试那个真正对症的动作。原因是执行器知道"我没做事"
    （日志里明写"重标定未产生新段"），控制器却无从得知。

    指纹只取**加帧/换段就会变**的量：路径版本 + 该窗口的生产步数 + 证据所在段。
    所以补过一块帧之后这条 no-op 记录自动失效、动作重新可选 —— 记的是
    「在这个盘面上没用」，不是「这个动作永远没用」。

    ⚠️ 两侧（执行器记账 / 控制器判读）必须用**同一份**实现，否则又是一个
    「同一个不变量的 N 份实现」。
    """
    w = record or {}
    # 🔑🔑 [审计 #64 后续，2026-09-14] **形参从 `window_record` 改名成 `record`。**
    # 本函数**同时**吃窗口记录和子窗（sampling unit）记录，而这两张表的字段集合
    # 不同（`identity` / `local_index` / `output_dir` 只在子窗上，`segment` /
    # `n_production_segments` 只在窗口记录上）。叫 `window_record` 是在说谎，
    # 而且会让 `tests/test_no_phantom_view_keys.py` 的扫描器（它按形参名认表）
    # 把子窗专属的 `identity` 报成"窗口记录上的幽灵键"—— 它报得对：
    # **按那个名字，那确实是个幽灵键。** 改名让名字与契约一致，扫描器的前提也恢复成立。
    #
    # ⚠️⚠️ **「哪些键在哪张表上」的权威名单在测试里，不在这里，也不许在这里复制一份。**
    #   `tests/test_audit_2026_09_14_controller_budget.py::
    #    test_the_window_table_and_the_unit_table_do_not_share_each_other_s_keys`
    # 它是**从真实 `read()` 打差集**建出来的（单段 + 合并两个构造各跑一遍），
    # 键集一漂就红。在源码里再写一份注释表 = 同一个不变量两份实现，而注释那一份
    # 会先过期 —— 这正是本仓最贵的那类 bug。要改名单，先跑一次 `read()` 打差集。
    #
    # 这一对表的字段差异是 2026-09-14 当天重复出错最多的地方，**四次**：
    #   · `segment` —— 子窗上**从来没有**；窗口记录上也只有**合并视图**才加
    #     （单段视图没有）⟹ 「窗口记录必有 segment」这个直觉是错的；
    #   · `solver_n_decorrelated` —— **两张表上都不存在**，真名
    #     `solver_n_frames_decorrelated`（它只是 history snapshot 的输出键名）；
    #   · `skipped_windows` —— 控制器视图与求解器结果**同名不同源**（元素一个是
    #     int 下标、一个是带 `window_index` 的 dict）；
    #   · `identity` —— 子窗专属，读到窗口记录上就是幽灵键（本函数的形参先前
    #     叫 `window_record`，那个名字本身在说谎，已改名 `record`）。
    # 🔑 [契约 A] **本函数要同时吃窗口记录和子窗（sampling unit）记录。**
    # 三种记录各自的"换段就变"那一维不同名，按可得性依次取：
    #   · 子窗记录 → `identity`（每个子系综自己锁一份 f_k，identity 变就是换了系综）
    #   · 合并视图的窗口记录 → `segment`（段名）
    #   · 单段视图的窗口记录 → `n_production_segments`（段**数**，开了新段就变）
    # 缺这一维的后果是把"换段"这一维悄悄关掉 ⟹ no-op 记录在换段之后仍然有效，
    # 而换段恰恰是"让一个先前没用的动作重新变得有用"的操作。
    seg = w.get("identity")
    if seg is None:
        seg = w.get("segment")
    if seg is None:
        seg = w.get("n_production_segments")
    # ⚠️ **未知步数不等于 0 步。** 先前 `int(... or 0)` 让两者给出同一个指纹 ⟹
    # 一个窗口从"读不到"变成"确实 0 步"（或反过来）指纹不变，no-op 记录不失效。
    #
    # 🔑🔑 [2026-09-15 真机 cyclod_ligand2/rep2] **先前写 `?` 的那版做反了。**
    # 原注释说「未知写成 `?`：它与任何真实步数都不同，记录因此自动失效」——
    # 这是错的：`?` 等于 `?`。**从未生产过**的窗口 `production_steps` 恒为 None
    # ⟹ 指纹恒为 `1|?|vanishing` ⟹ 挂在它上面的 no-op 记录**在结构上永不失效**。
    # 真机后果：win4 的 `RUN_PRODUCTION:4` 被一个 bug 误记成 no-op 之后，
    # 想让它失效必须先产帧，而产帧恰恰被这条记录挡着 —— 死锁，
    # 整条流水线停在 NO_FEASIBLE_ACTION。
    #
    # 现在回到注释本来声明的那个**保守方向**：**未知 ⟹ 没有可比身份 ⟹ 返回
    # None**。两侧都按「None 不匹配任何东西」处理（写侧不记、读侧不认），
    # 宁可多试一次，也绝不粘住一条无法失效的记录。
    _steps = w.get("production_steps")
    if _steps is None:
        return None
    return "|".join(str(x) for x in (
        int(path_version or 0),
        int(_steps),
        str(seg if seg is not None else ""),
    ))


def relearn_epoch_required_steps(window_record):
    """开一个全新 f_k Epoch 至少要预留多少步：LEARN + burn-in + **首档**验证。

    量从这个窗口**自己**上一轮的实际消耗估（自校准），估不到才退保守常量 ——
    跨体系拍一个固定数字是引入未验证阈值。
    """
    led = (window_record or {}).get("warmup_budget_ledger") or {}
    learn = int(led.get("learning_steps") or 0) or 80000
    burn = int(led.get("freeze_burn_in_steps") or 0) or 10000
    # 首档验证预留：不是整份验证预算，只要够走完第一档。
    # ⚠️ 这里原来读 `window_record["validation_attempt_budget_steps"]` —— 窗口记录里
    # **没有这个键**，于是恒取 50000 兜底。首档的权威值就在冻结验证阶梯里，直接用它，
    # 不再要求窗口记录搬运一个它没有的数。
    try:
        import ibs_engine as _ie
        first_rung = int(_ie.FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS[0])
    except Exception:  # noqa: BLE001
        first_rung = 50000
    return int(learn + burn + first_rung)


def segment_dirs_for_evidence(segments, stage_dir: str, checkpoint_dir: str):
    """把证据来源的段名翻成 (输出目录, checkpoint 目录)；基准段返回 (None, None)。

    `segments` 是窗口记录里的 `segment` 值（stage 目录名，如 `vanishing` /
    `vanishing_3`）。多个窗口落在不同段时**fail-closed 抛错**，不静默回落到
    基准段 —— 回落正是本来那个 bug。
    """
    names = {str(x) for x in segments if x}
    suffixes = set()
    for nm in names:
        # [审计 #63] 段号解析统一走 `segment_index_of_dir()`（同 `_segment_stage_names`
        # / `existing_segment_names`）：完整后缀纯数字才算段号，rewindow 目录不会被
        # 误判。解析不出来仍然 **fail-closed 抛错**，绝不静默回落到基准段 ——
        # 回落正是本函数原来那个 bug。
        n = segment_index_of_dir(nm, stage_dir)
        if n is None:
            raise ValueError(
                f"无法解析证据来源段名 {nm!r}"
                f"（基准 {os.path.basename(os.path.normpath(stage_dir))!r}）"
            )
        suffixes.add(int(n))
    if len(suffixes) > 1:
        raise ValueError(
            f"补采的窗口跨越多个采样段 {sorted(suffixes)}，无法落在单一段里"
        )
    n = next(iter(suffixes)) if suffixes else 1
    # ⚠️ **基准 stage 目录隐含就是段 1**：新采样段从 **2** 起
    # （`_recalibrate_f_k_and_resample_segment` 默认 `segment_index=2`，
    # 段号分配是 `max([1] + existing) + 1`）。所以 `n <= 1` 覆盖基准段是**正确**的，
    # 不是"`vanishing_1` 被静默重定向"的漏洞 —— `vanishing_1` 这个目录名按约定
    # 根本不会被创建。
    if n <= 1:
        return None, None          # 基准段：走默认目录
    return (
        f"{stage_dir.rstrip(os.sep)}_{n}",
        os.path.join(checkpoint_dir, f"segment_{n}"),
    )



def windows_by_segment(window_indices, window_records, added_steps):
    """把要补采的窗口按**它们证据所在的段**分组：`{段名: {窗口: 新的目标步数}}`。

    目标步数为 ``None`` 表示「这个窗口盘上烧了多少步读不到 ⟹ 不给目标、按
    调用方原目标跑」。**窗口本身一定在分组里** —— 见下面那段长注释。

    多窗补采天然会跨段（真机 win0-3 在 `vanishing_2`、win4 在基准段），而
    `segment_dirs_for_evidence` 对跨段集合是 fail-closed 的 ⟹ 执行器必须
    **一段一次**，不能把整批丢进去。目标步数是"这个窗口已有的 + 一块"，
    所以分组和算步数是同一件事，放在一起才不会漂开。
    """
    by_seg: Dict[str, Dict[int, Optional[int]]] = {}
    for w in window_indices:
        rec = next(
            (x for x in (window_records or []) if int(x["window_idx"]) == int(w)),
            None,
        ) or {}
        # 🔑 [2026-09-14] **这是全仓唯一把「未知步数」写回盘的地方，最该修。**
        # 目标步数会被执行器当成 `_production_step_overrides` 落进 run —— 先前
        # `int(... or 0) + added` 把「读不到这个窗口烧了多少步」固化成
        # 「它烧过 0 步、目标就是一块」，未知从此变成事实，而这个窗口盘上其实
        # 可能已经有几十万步。fail-closed：读不到就**不给目标**，让调用方按原目标
        # 跑（绝不编一个比现有帧还小的目标去覆盖它）。
        #
        # 🔑🔑 [2026-09-15 真机 cyclod_ligand2/rep2] **「不给目标」≠「不给窗口」。**
        # 上面那条 fail-closed 原来写的是 `continue` —— 把窗口整个从分组里**删掉**，
        # 于是执行器的 `for _seg, overrides in ...` 空转，`run_once` **一次都没调**，
        # 1 秒返回、盘面当然逐项未变 ⟹ 通用 no-op 记账把一个**从没被执行过**的
        # 动作记成「执行过且没用」⟹ 下一轮 NO_FEASIBLE_ACTION ⟹
        # `_assert_stage_result_sane` 抛 ANALYSIS_INCOMPLETE 打死整条流水线。
        # 真机盘面：win4 从未生产（只有 warmup_failure.json），`production_steps`
        # 恒 None，no-op 台账里那条 `RUN_PRODUCTION:4` 的指纹 `1|?|vanishing`
        # 里的 `?` 就是它。`CONTINUE_WARMUP`（同一个函数、added_steps=0）被同一行
        # 静默吞掉，日志里「win4 连发 40 次、盘面一动不动」也是这个，不是窗口级
        # resume 缓存门。
        # 正解就是注释本来说的那句：**不给目标、让调用方按原目标跑** ⟹ 窗口留在
        # 分组里、值为 None。引擎侧 `if window_idx in production_step_overrides`
        # 本来就按「不在表里 = 用默认 n_steps_per_window」处理，调用方把 None
        # 过滤掉即可（见 `_run_stage2_autonomous` 的 RUN_PRODUCTION 分支）。
        _have = rec.get("production_steps")
        by_seg.setdefault(str(rec.get("segment") or ""), {})[int(w)] = (
            None if _have is None else int(_have) + int(added_steps)
        )
    return by_seg


def lambdas_from_version_record(record, fallback):
    """🔑 **落盘的路径版本是 λ 的唯一权威。**

    `lambda_path_versions._q()` 写盘时把 λ 量化到 `LAMBDA_DECIMALS` 位；
    如果继续拿内存里那份**未量化**的去采样，每插一个 λ 就永久制造一次错位。

    真机（2026-09-12 17:05）实测 win4：
        路径版本 0.31005333   vs   实际采样 0.310053335   差 5e-9
    后果是死锁 —— 采样侧的 resume 门用 `np.allclose(atol=1e-9)`（含默认
    rtol=1e-5 ⟹ 实际容差 ~3e-6）判"λ 匹配、跳过重采"，而分析侧
    `load_ibs_window_outputs_from_dir` 用**精确相等**判"λ 不匹配"直接抛
    ValueError。于是那个窗口**永远采不了也永远读不了**，每次启动必崩。

    🔑🔑 [审计 #4，2026-09-14] **原来的长度兜底是 fail-open，恰好在最要命的
    那一种情形下放行。**

    先前是 `return out if len(out) == len(fallback) else [float(x) for x in fallback]`
    —— 长度对不上就**静默退回内存里那份未量化的 λ**，也就是这个函数存在的全部
    理由所要防的那个东西。而长度对不上正是**唯一危险**的情形：
    `append_version` 是幂等的，事件 id 已在祖先链里时它**返回现有的当前版本**
    （见 `lambda_path_versions.append_version`）。所以「刚算出的 `new_l` 与返回
    记录的态数不同」= 这次插点其实没被应用、盘上的权威布局是另一套。此时拿
    `new_l` 去采样，采出来的帧与已发布的路径版本**描述的不是同一条布局**，正是
    上面那段死锁的成因。

    现在：长度不符 ⟹ **fail-closed 抛错**，把「盘上权威布局 ≠ 调用方以为的布局」
    这件事当场暴露，而不是挑一个看起来能跑的继续跑。
    记录里根本没有 λ（老记录 / 空 states）是**另一回事** —— 那是「无从核对」，
    保持原来的回退行为，不制造新的 fail。
    """
    states = (record or {}).get("states") or []
    out = [
        float(st.get("lambda_vdw")) for st in states
        if st.get("lambda_vdw") is not None
    ]
    if not out:
        # 记录里没有 λ（老格式 / 空 states）⟹ 无从核对，保持既有回退。
        return [float(x) for x in fallback]
    if len(out) != len(fallback):
        raise ValueError(
            f"路径版本记录里的 λ 态数（{len(out)}）与调用方要采样的那份"
            f"（{len(fallback)}）不符。`append_version` 幂等 ⟹ 这通常意味着"
            "**这次布局变更并没有被应用**（同一个 event id 已在祖先链里），"
            "盘上的权威布局是另一套。继续按调用方那份采样会产出一批与已发布路径"
            "版本对不上的帧（分析侧按精确相等判 λ，会直接拒绝，窗口从此既采不了"
            "也读不了）。请先核对 `path_current.json` 与 `path_versions/`，"
            "不要绕过这道检查。"
        )
    return out



def segment_index_of_dir(path: str, stage_dir: str):
    """`<stage_dir>_<N>` → N；基准段 → 1；不是采样段 → None。

    判据是**完整后缀**纯数字，不是 `rsplit("_", 1)[-1]`。immutable rewindow 的
    目录叫 `<stage>_rewindow_<sha256[:12]>`，那 12 位十六进制有 (10/16)**12
    ≈ 0.34% 的概率全是数字 —— 按末段判就会把子系综目录当成采样段合并，
    子窗的局部下标 0/1 被当成物理窗口 0/1，静默产出错误 ΔG。

    ⚠️ **别把那个后缀简化成裸 sha12。** 现在安全是**构造上**安全，不是概率上：
    完整后缀是 `rewindow_<sha12>`，含字面量 `rewindow_` ⟹ 即使 sha12 十二位
    全是数字，`"rewindow_0123456789ab".isdigit()` 仍是 False。去掉那个前缀
    就把构造保证降级成上面那个 0.34%。
    """
    base = os.path.basename(os.path.normpath(stage_dir))
    name = os.path.basename(os.path.normpath(path))
    if name == base:
        return 1
    if not name.startswith(base + "_"):
        return None
    suf = name[len(base) + 1:]
    return int(suf) if suf.isdigit() else None


def existing_segment_names(stage_dir: str):
    """盘上已有的采样段目录名（含基准段），按**段号**排序。

    [审计 #63] 先前按**名字**排序（字符串序 ⟹ `vanishing_10` 排在 `vanishing_2`
    之前），且用 `rsplit("_", 1)[-1]` 判段号 ⟹ rewindow 目录有 0.34% 概率被当成
    采样段。两处都改走 `segment_index_of_dir()`，与 `_segment_stage_names` /
    `segment_dirs_for_evidence` 同一份规则。
    """
    base = os.path.basename(os.path.normpath(stage_dir))
    found = []
    for d in glob.glob(stage_dir.rstrip(os.sep) + "_*"):
        n = segment_index_of_dir(d, stage_dir)
        if n is not None and n > 1 and os.path.isdir(d):
            found.append((n, os.path.basename(d)))
    return [base] + [nm for _n, nm in sorted(found)]


def repartition_tail_from_anchor(
    lambdas_var: Sequence[float],
    window_ranges: Sequence[Sequence[int]],
    anchor: float,
    *,
    min_states_per_window: int,
    max_states_per_window: int,
    anchor_atol: float = 1e-9,
    # 🔑🔑 [2026-09-15] 下面三个参数修的是**同一个 bug 的两半**。
    #
    # (1) **方向反了。** 控制器侧的可行性（`feasible_repair_actions` 的
    #     `split_tail_window`）问的是「存不存在 m ≥ 当前尾段窗口数 **+1** 的合法
    #     分窗」——要更**细**。而这里先前调 `vanishing_subdomain_ranges_from_lambdas`
    #     且不传窗口数，那个分窗器内部是 `n_windows = min_windows`，取**最少**窗口数
    #     ⟹ 拆窗动作可能反过来**合并**窗口、把跨度**扩大**。
    #     纸面复现（lo=4, hi=8）：[(0,4),(3,8),(7,12),(11,16)]、anchor=3、尾段 3 个窗
    #     ⟹ 尾段 13 态、总间隔 12、min_windows=ceil(12/7)=2 ⟹ [(3,10),(9,16)]，
    #     三个 5 态窗变成两个 7 态窗。可行性说"可以拆细"，执行器交付"合并了"。
    #     ⟹ `n_windows` 现在是**必须显式给**的目标窗口数（调用方传 m0+1）。
    #
    # (2) **判据被换掉了。** 生产分窗是 `stage2_window_partition=metric_integral`
    #     （按 ∫g dλ 均衡），而尾段重分调的是等状态数贪心、**根本不读 metric_g**。
    #     于是重分之后：冻结前缀按 ∫g 切、新尾段按边数切，同一条路径上两套判据。
    #     ⟹ 给了 `pilot_lambdas`/`metric_g` 就用 `partition_windows_by_metric_integral`，
    #     与生产同一个分窗器；`partition_criterion="metric_integral"` 而数据缺失时
    #     **fail-closed**，不静默退回等弧长（与 `_load_pilot_for_path_evolution`
    #     那几个调用点的口径一致）。
    n_windows: Optional[int] = None,
    pilot_lambdas: Optional[Sequence[float]] = None,
    metric_g: Optional[Sequence[float]] = None,
    partition_criterion: str = "arclength",
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """**只重分尾段**：anchor 之前的窗口逐字冻结，只有 anchor 起的重新分配。

    设计依据：``docs/archive/PLAN_PATH_REPAIR_2026-09-11.md``。老板的原话是
    「若当前实现要求全部从头重跑，那是路径版本/reuse 规则仍过粗，**不是物理上的
    必要代价**」。

    **为什么原来做不到**（查过，归因不在指纹）：
      · ``stage2_window_max_states`` 在 ``_PREOPT_DERIVED_PATH_KEYS`` 里，属于
        **第 2 层（派生路径）**；两层拆分的设计意图本来就是「第 1 层若匹配，
        **仍可离线重算**」—— 指纹层不是阻塞点，改它**不需要重跑 pilot**。
      · 真正作废前缀的是：窗口产物的复用键是**每个窗口自己的 λ 集合**
        （``_window_lambda_key`` / ``_resume_cached_window_gate_status`` 的
        ``np.allclose``），而改 max_states 会走**全局**分窗器
        ``vanishing_subdomain_ranges_from_lambdas`` ⟹ **所有**窗口边界重排 ⟹
        连健康的前缀也对不上。
    ⟹ 缺的就是本函数：一个**只动尾段**的分窗操作。

    老板给的验收条件，逐条在下面用断言钉住：
      1. 不重新跑 pilot、**不改变 λ 表**（本函数只返回 ranges，λ 原样带回并断言相等）
      2. anchor 必须是**现有共享态**
      3. anchor 以前的 ranges **逐字相等**
      4. 仅 anchor 起的尾段重新分配
      5. **完整覆盖、无缺口**，相邻窗口恰好共享一个态
      6. **确定性、幂等**（同输入同输出；对自身输出再跑一次不变）
      7. 输出明确的 ``changed_windows`` 与**复用清单**
      8. **只作废 anchor 以后的产物**
      9. execution 层的 K + 难度检查 → 见 ``feasible_repair_actions(layer="execution")``
     10. **先 shadow/preview，不自动启动 GPU** → 本函数是**纯函数**，不落盘、不采样

    ``anchor`` 是 λ 值（不是下标）—— 调用方看到的是 λ，下标随布局变。
    """
    lam = [float(x) for x in lambdas_var]
    ranges = [(int(a), int(b)) for a, b in window_ranges]
    lo_n, hi_n = int(min_states_per_window), int(max_states_per_window)
    if ranges != sorted(ranges):
        raise ValueError(f"window_ranges 必须按起点升序: {ranges}")
    if not ranges or ranges[-1][1] != len(lam):
        raise ValueError(f"末窗 {ranges[-1] if ranges else None} 必须覆盖到最后一个态")

    # ---- 验收 2：anchor 必须是现有的**共享**态 ----
    shared_pre = sorted({b - 1 for _a, b in ranges[:-1]} & {a for a, _b in ranges[1:]})
    cand = [i for i, v in enumerate(lam) if abs(v - float(anchor)) <= float(anchor_atol)]
    if len(cand) != 1:
        # 调用方最容易踩的是"传了四舍五入过的 λ" ⟹ 匹配 0 个。把**合法 anchor 全列出来**，
        # 免得还要自己去翻 λ 表数小数位。
        raise ValueError(
            f"anchor λ={anchor!r} 在 λ 表里匹配到 {len(cand)} 个态（需恰好 1 个，"
            f"容差 {anchor_atol}）。合法的 anchor（现有共享态）是："
            + ", ".join(f"win{i+1}起点 λ={lam[j]!r}" for i, j in enumerate(shared_pre))
        )
    a_idx = cand[0]
    shared = {b - 1 for _a, b in ranges[:-1]} & {a for a, _b in ranges[1:]}
    if a_idx not in shared:
        raise ValueError(
            f"anchor λ={anchor}（下标 {a_idx}）不是现有共享态；"
            f"现有共享态下标={sorted(shared)}。只允许在共享节点上冻结。"
        )

    prefix = [(a, b) for a, b in ranges if b - 1 <= a_idx]
    if not prefix or prefix[-1][1] - 1 != a_idx:
        raise ValueError(
            f"anchor 下标 {a_idx} 不是任何窗口的末态；前缀={prefix}"
        )

    # ---- 尾段重分：复用既有分窗器，只喂给它 anchor 起的那段 λ ----
    tail_lam = np.asarray(lam[a_idx:], dtype=float)
    if tail_lam.size < lo_n:
        raise RuntimeError(
            f"anchor 之后只有 {tail_lam.size} 个态，不足 min_states_per_window={lo_n}"
        )
    _crit = str(partition_criterion).lower()
    _n_tail_before = sum(1 for a, _b in ranges if a >= a_idx)
    _n_target = int(n_windows) if n_windows is not None else None
    if _crit in ("metric_integral", "state_count"):
        if not pilot_lambdas or not metric_g:
            raise RuntimeError(
                f"partition_criterion={_crit} 需要 pilot_lambdas/metric_g，"
                "但调用方没给。拒绝静默退回等状态数贪心 —— 那会让冻结前缀按 ∫g 切、"
                "新尾段按边数切，同一条路径上两套分窗判据。"
            )
        _tail_ranges_local, _mi_diag = partition_windows_by_metric_integral(
            [float(x) for x in tail_lam],
            [float(x) for x in pilot_lambdas],
            [float(x) for x in metric_g],
            min_states_per_window=lo_n,
            max_states_per_window=hi_n,
            n_windows=_n_target,
            objective=_crit,
        )
        tail_ranges_local = [(int(x), int(y)) for x, y in _tail_ranges_local]
    else:
        _mi_diag = None
        if _n_target is not None:
            # 等状态数贪心没有"目标窗口数"这个入口（它内部恒取 min_windows）。
            # 直接用它的构造规则按 `_n_target` 铺：总间隔均分，小窗在前，
            # 与 `vanishing_subdomain_ranges_from_lambdas` 的分配逻辑逐字一致。
            _iv = int(tail_lam.size) - 1
            if not (_n_target * (lo_n - 1) <= _iv <= _n_target * (hi_n - 1)):
                raise RuntimeError(
                    f"尾段 {tail_lam.size} 个态切不出 {_n_target} 个窗口"
                    f"（每窗 ∈[{lo_n},{hi_n}]）"
                )
            _base, _extra = divmod(_iv, _n_target)
            _spans = [_base] * (_n_target - _extra) + [_base + 1] * _extra
            tail_ranges_local, _st = [], 0
            for _sp in _spans:
                tail_ranges_local.append((_st, _st + _sp + 1))
                _st += _sp
        else:
            tail_ranges_local = [
                (int(x), int(y)) for x, y in vanishing_subdomain_ranges_from_lambdas(
                    tail_lam,
                    min_states_per_window=lo_n,
                    max_states_per_window=hi_n,
                    # 🔑🔑 [审计 #50，2026-09-14] **尾段是子路径，不许套全路径的手工固定表。**
                    # `vanishing_subdomain_ranges_from_lambdas` 在 `size == 23` 时会
                    # 短路返回 `VANISHING_FIXED_WINDOW_RANGES` —— 那张表是**全路径**
                    # 手工调出来的，含专为 λ=1 耦合端做的 window-0 收窄，而尾段的起点
                    # 根本不是 λ=1；它还会**完全忽略** min/max_states_per_window，
                    # 而尾段重分恰恰要按调用方给的 lo/hi 切。更糟的是那张表的尺寸是
                    # 5,4,5,5,5,4，后置的 `validate_single_shared_boundary_ranges`
                    # 照样通过 ⟹ 替换发生了却没有任何人报告。
                    allow_fixed_23_state_table=False,
                )
            ]
    tail_ranges = [(x + a_idx, y + a_idx) for x, y in tail_ranges_local]
    new_ranges = list(prefix) + tail_ranges

    # ---- 验收 1/3/4/5：λ 未变、前缀逐字相等、覆盖完整、单一共享边界 ----
    if [float(x) for x in lambdas_var] != lam:
        raise RuntimeError("λ 表被改动了 —— 本函数只重分窗口，不动 λ")
    if new_ranges[: len(prefix)] != list(prefix):
        raise RuntimeError("前缀 ranges 不再逐字相等")
    validate_single_shared_boundary_ranges(new_ranges, len(lam))
    # 🔑 ``validate_single_shared_boundary_ranges`` **只**校验覆盖与单一共享边界，
    # **不**校验 [lo, hi]。少了这一条就可能返回一个"结构合法但每窗超限"的尾段
    # （验证时实测踩到：尾段被并成 K=10 而 hi=8，居然一路通过）。
    # ⚠️ 只校验**新产生的尾窗**：前缀是冻结带回来的历史布局，它可能是在别的
    # min/max 下定的，不该因为本次参数变了就被判非法。
    for a, b in tail_ranges:
        if not (lo_n <= b - a <= hi_n):
            raise RuntimeError(
                f"重分后的尾窗 {(a, b)} 有 {b - a} 个态，越出 [{lo_n},{hi_n}]；"
                "分窗器给了非法布局，拒绝返回。"
            )

    # ---- 🔑🔑 [2026-09-15] 验收 9/10：**动作的方向必须被证明，不能假定** ----
    # 这两条是这次 bug 的直接回归钉子。先前这里的验收全是**结构性**的（覆盖 /
    # 共享边界 / [lo,hi]），而 `[7,7]` 这种「三窗合成两窗」在结构上完全合法 ——
    # 于是一个方向完全反了的结果一路通过、被 `record_tail_repartition_version`
    # 记成一次成功的拆窗。`SPLIT_TAIL_WINDOW` 的语义是**缩跨度**，缩没缩得由
    # 数字说了算。任何一条不满足就 fail-closed，宁可让控制器看见"这个动作做不到"。
    # ⚠️ 这两条**只在调用方显式要求拆细时**生效（`n_windows` 非空）。
    # 另一个调用点是"末窗 K > hi 的布局合法化"，它的目标是**变合法**、不是变细：
    # 例如 lo/hi=4/8、尾段 [9,4] ⟹ [7,6] 就是一次完全正确的合法化，窗口数没变。
    # 把"必须更细"套在它头上会把一个正确结果判成失败。
    _require_finer = _n_target is not None
    _n_tail_after = len(tail_ranges)
    if _require_finer and _n_tail_after <= _n_tail_before:
        raise RuntimeError(
            f"尾段重分没有把窗口切细：anchor 之后原有 {_n_tail_before} 个窗口、"
            f"重分后 {_n_tail_after} 个（sizes={[b - a for a, b in tail_ranges]}）。"
            "`SPLIT_TAIL_WINDOW` 的语义是缩跨度；窗口数没增加说明分窗器取的是"
            "**最少窗口数**（`vanishing_subdomain_ranges_from_lambdas` 内部的 "
            "`n_windows = min_windows`），方向与可行性判据相反。"
            "调用方必须显式给 `n_windows`（= 原尾段窗口数 + 1）。"
        )
    # 跨度的口径**跟分窗器的目标函数同一个**：metric_integral 时比 ∫g，
    # 否则比态数。比"失败窗口重分后的 span"是不成立的 —— 边界整体移动之后，
    # 原来那个窗口的下标已经不指向同一段 λ 了。
    _tail_old = [(a, b) for a, b in ranges if a >= a_idx]
    if _mi_diag is not None:
        _gc = metric_integral_cumulative(
            lam, [float(x) for x in pilot_lambdas], [float(x) for x in metric_g]
        )
        _span = lambda rs: max(abs(float(_gc[b - 1] - _gc[a])) for a, b in rs)
        _before, _after, _unit = _span(_tail_old), _span(tail_ranges), "∫g dλ"
    else:
        _span = lambda rs: max(b - a for a, b in rs)
        _before, _after, _unit = _span(_tail_old), _span(tail_ranges), "态数"
    if _require_finer and not (_after < _before):
        raise RuntimeError(
            f"尾段重分没有缩小跨度：最大窗口 {_unit} {_before:.4g} → {_after:.4g}"
            f"（尾段 sizes {[b - a for a, b in _tail_old]} → "
            f"{[b - a for a, b in tail_ranges]}）。拒绝把一次扩大跨度的重分"
            "记成成功的 `SPLIT_TAIL_WINDOW`。"
        )

    # ---- 验收 7/8：changed / reusable，按**λ 集合**判（复用真正的键就是它）----
    def _key(rng):
        a, b = rng
        return tuple(round(float(x), 12) for x in lam[a:b])

    old_keys = {i: _key(r) for i, r in enumerate(ranges)}
    new_keys = {i: _key(r) for i, r in enumerate(new_ranges)}
    reusable, changed = [], []
    for ni, nk in new_keys.items():
        hit = next((oi for oi, ok in old_keys.items() if ok == nk), None)
        (reusable if hit is not None else changed).append(
            {"new_window": ni, "old_window": hit, "n_states": len(nk)}
        )
    invalidated = sorted(
        oi for oi, ok in old_keys.items() if ok not in set(new_keys.values())
    )
    if any(i <= len(prefix) - 1 for i in invalidated):
        raise RuntimeError(
            f"前缀窗口被作废了（{invalidated}），违反「只作废 anchor 以后产物」"
        )

    return new_ranges, {
        "protocol_version": TAIL_REPARTITION_PROTOCOL_VERSION,
        "source": "tail_only_repartition_from_shared_anchor",
        "anchor_lambda_vdw": float(lam[a_idx]),
        "anchor_state_index": int(a_idx),
        "frozen_prefix_windows": [list(r) for r in prefix],
        "old_ranges": [list(r) for r in ranges],
        "new_ranges": [list(r) for r in new_ranges],
        "n_windows_before": len(ranges),
        "n_windows_after": len(new_ranges),
        "tail_windows_before": int(_n_tail_before),
        "tail_windows_after": int(_n_tail_after),
        "partition_criterion": _crit,
        "tail_max_window_span_before": float(_before),
        "tail_max_window_span_after": float(_after),
        "tail_span_unit": _unit,
        "reusable_windows": reusable,
        "changed_windows": changed,
        "invalidated_old_windows": invalidated,
        "min_states_per_window": lo_n,
        "max_states_per_window": hi_n,
        "lambda_table_unchanged": True,
        "note": (
            "纯函数：不落盘、不采样、不启动 GPU。执行前还要过 "
            "feasible_repair_actions(layer='execution')，它同时检查 K 与物理难度，"
            "**缺难度证据时 fail-closed**。"
        ),
    }


def record_tail_repartition_version(
    checkpoint_dir: str,
    lambdas_var: Sequence[float],
    new_ranges: Sequence[Sequence[int]],
    diag: Dict[str, Any],
    *,
    first_untrusted_window: Optional[int] = None,
    reason: str = "tail_only_repartition_from_first_untrusted_window",
) -> Dict[str, Any]:
    """把一次尾段重分登记成**路径版本链事件**。这是 execute 通路的**前置**。

    🔑 [2026-09-11] 为什么它不能等 J 一起做：没有版本链事件，**崩溃恢复分不清
    「已经重分过没有」** —— 第二次启动会再重分一次，把刚跑完的新尾段又作废。
    老板的话：「不能等 win4/win5 再次失败后才补。」

    幂等由 `lambda_path_versions.append_version` 的确定性 event_id 保证：
    同一个 (λ表, ranges, kind, detail) 重复登记不会产生第二个版本。

    ⚠️ 本函数**只登记，不采样**。λ 表原样带回（尾段重分**不改 λ**）。
    """
    import lambda_path_versions as _lpv

    lam = [float(x) for x in lambdas_var]
    detail = {
        "anchor_lambda_vdw": diag.get("anchor_lambda_vdw"),
        "anchor_state_index": diag.get("anchor_state_index"),
        "first_untrusted_window": (
            None if first_untrusted_window is None else int(first_untrusted_window)
        ),
        "frozen_prefix_windows": diag.get("frozen_prefix_windows"),
        "old_ranges": diag.get("old_ranges"),
        "new_ranges": diag.get("new_ranges"),
        "changed_windows": diag.get("changed_windows"),
        "reused_windows": diag.get("reusable_windows"),
        "invalidated_old_windows": diag.get("invalidated_old_windows"),
        "n_windows_before": diag.get("n_windows_before"),
        "n_windows_after": diag.get("n_windows_after"),
        "min_states_per_window": diag.get("min_states_per_window"),
        "max_states_per_window": diag.get("max_states_per_window"),
        # 尾段重分**不动 λ 表** —— 登记这一条，好让读版本链的人不必去 diff λ。
        "lambda_table_unchanged": True,
    }
    return _lpv.append_version(
        checkpoint_dir,
        [0.0] * len(lam), lam, [list(r) for r in new_ranges],
        kind="tail_repartition",
        reason=reason,
        detail=detail,
    )


def render_tail_repartition_preview(diag: Dict[str, Any], lambdas_var: Sequence[float]) -> str:
    """人读的 preview。**先看这个，再决定要不要跑 GPU。**"""
    lam = [float(x) for x in lambdas_var]
    out = [
        "尾段重分预览（**纯计算，未启动任何采样**）",
        f"  anchor : λ={diag['anchor_lambda_vdw']:.4f}（下标 {diag['anchor_state_index']}，现有共享态）",
        f"  λ 表   : {len(lam)} 态，**未改动**",
        f"  窗口数 : {diag['n_windows_before']} → {diag['n_windows_after']}",
        "",
        f"  {'窗口':>6} {'范围':>12} {'K':>3} {'λ 区间':>20}  处置",
        "  " + "-" * 62,
    ]
    old = [tuple(r) for r in diag["old_ranges"]]
    reuse_new = {d["new_window"]: d["old_window"] for d in diag["reusable_windows"]}
    for i, r in enumerate(diag["new_ranges"]):
        a, b = int(r[0]), int(r[1])
        oi = reuse_new.get(i)
        tag = (f"**复用**旧 win{oi}（λ 逐位相同）" if oi is not None
               else "**重采**（新 λ 集合）")
        out.append(
            f"  {('win%d' % i):>6} {str((a, b)):>12} {b - a:>3} "
            f"{f'{lam[a]:.4f}→{lam[b-1]:.4f}':>20}  {tag}"
        )
    out.append("")
    out.append(f"  作废的旧窗口 : {diag['invalidated_old_windows']}（anchor 之前的一个都没有）")
    out.append(f"  冻结的前缀   : {diag['frozen_prefix_windows']}")
    out.append("  ⚠️ 执行前必须过 feasible_repair_actions(layer='execution')："
               "同时检查 K 与物理难度，缺难度证据 fail-closed。")
    return "\n".join(out)


REPAIR_ACTIONS = ("insert_lambda", "split_tail_window")


def tail_overflow_exemption_active(path_events) -> bool:
    """末窗的「溢出槽豁免」还在不在。

    🔑🔑 [审计 #29，2026-09-14] **`feasible_repair_actions(tail_exempt_from_max=…)`
    先前是个死参数** —— 它的 docstring 明写「调用方应从路径版本链里数
    ``split_tail_window`` 事件来决定这个标志」，而**全仓没有任何调用方传过它**
    ⟹ 恒为 `True` ⟹「末窗被拆过之后豁免作废、此后插点无处可去 ⟹
    `HALT_LAMBDA_BUDGET_INSUFFICIENT`」那条分支**结构上不可达**。
    后果是插 λ 在溢出槽已经用尽之后仍被判成可行。

    规则（就是那句 docstring 说的）：末窗是 model B 的溢出槽，**在被拆过之前**
    豁免 ``max_states_per_window``；拆过之后两个孩子都受约束、溢出槽消失。

    ⚠️ 两种拆窗事件**都**算数：路径演化分支写 ``split_tail_window``，
    自治分支写 ``tail_repartition``（`record_tail_repartition_version`）。
    先前那句 docstring 只提了前者 —— 只数一种等于漏掉自治循环拆的每一次。

    ``path_events``：``{事件名: 次数}``（`Stage2RepairController._read_path()`
    的 `events`，或 `lambda_path_versions` 数出来的同形 dict）。
    """
    ev = dict(path_events or {})
    return not (int(ev.get("split_tail_window", 0) or 0)
                + int(ev.get("tail_repartition", 0) or 0))


def vanishing_rescue_ranges(
    failing_window_indices: Sequence[int],
    base_ranges: Sequence[Sequence[int]],
) -> List[Tuple[int, int]]:
    """把一个失败的系综换成若干**更小的重叠子系综**（λ 表一个态都不动）。

    🔑 [REWIND-01，2026-09-17] **这是有界重窗子系综划分的唯一权威。**
    先前它只活在 `abfe_pipeline.ABFEPipeline._build_vanishing_rescue_ranges`
    （静态、纯函数），而控制器那侧要判可行性 / 算成本却 import 不动 abfe_pipeline
    （循环依赖 + 它 import openmm）⟹ 成本表只好自己写一份
    `2 if (b-a) > 2 else 1`。同一个不变量两份实现，正是本仓最贵的那个形状。
    现在搬到这里，`abfe_pipeline` 那个静态方法转成一行转发。

    切法：按中点切左右两个**共享中点**的子窗（覆盖完整、接缝有共享节点，
    两条都是执行器的 fail-closed 断言）。K ≤ 2 切不出两个各 ≥2 态的子窗 ⟹
    原样返回，由执行器的 `len(children) < 2` 拒掉。
    """
    rescue_ranges: List[Tuple[int, int]] = []
    for window_idx in sorted(set(int(x) for x in failing_window_indices)):
        start, end = (int(x) for x in base_ranges[window_idx])
        n_states = end - start
        if n_states <= 2:
            rescue_ranges.append((start, end))
            continue
        midpoint = start + (n_states - 1) // 2
        left = (start, midpoint + 1)
        right = (midpoint, end)
        for child in (left, right):
            if child[1] - child[0] >= 2 and child not in rescue_ranges:
                rescue_ranges.append(child)
    return rescue_ranges


def feasible_repair_actions(
    window_ranges: Sequence[Sequence[int]],
    n_states: int,
    *,
    min_states_per_window: int,
    max_states_per_window: int,
    tail_exempt_from_max: bool = True,
    n_insert: int = 1,
    layer: str = "path_version",
    tail_difficulty_ok: Optional[bool] = None,
    # 🔑🔑 [审计 #24，2026-09-14] `SPLIT_TAIL_WINDOW` 这个**动作**实际做的是
    # 「从 anchor 起重分整个尾段」（`repartition_tail_from_anchor`），不是
    # 「把末窗一分为二」。判它需要重分起点 = anchor 所在窗口的**首态全局下标**。
    # 给不出就如实返回「判不了」，**不拿末窗那条冒充**（见函数体里的长注释）。
    tail_repartition_start_state: Optional[int] = None,
) -> Dict[str, Optional[str]]:
    """哪些**结构性**修补动作在当前布局下可行。纯计算，零 GPU。

    设计依据：``docs/archive/PLAN_PATH_REPAIR_2026-09-11.md`` §2「可行性规则」。

    返回 ``{动作名: None 表示可行 | str 表示不可行的理由}``。
    调用方取可行集：``[k for k, v in d.items() if v is None]``。

    **为什么要单独一个纯函数**：动作"不可行"必须以"不在候选集里"的形式呈现给
    控制器，而不是以"调用下去抛异常"的形式。后者会让控制器看不见自己少了一档，
    于是被迫用另一类动作去回答（这就是历史上"覆盖不足 → 插 λ"那个 type error
    的机制）。

    ``layer``：**豁免分两层，别混**（2026-09-11 老板定案）::

        "path_version"  路径版本层：末窗**可以**暂时充当溢出槽（豁免 max）
        "execution"     采样执行层：任何**实际要跑**的窗口都必须满足与普通窗口
                        相同的物理难度约束 —— 末窗不例外

    ⚠️ 执行层**不能只检查 ``K <= hi``**。实测反例：末窗 ``K=4`` 已经满足 max，
    却仍然是全场最难的那个（预热就停了、g 22.8→143.3）。真正缺的是
    ``max_states`` **加上** 热力学/动力学**难度上限**（Fisher / metric integral /
    预测 support / 已有的 ``N_eff/g`` 剖面）。所以执行层要求调用方通过
    ``tail_difficulty_ok`` 显式给出难度判定；**没给就 fail-closed**，
    而不是拿 ``K <= hi`` 冒充"可执行"。
    末窗超过难度上限时，应在**采样前**把 overflow materialize 成两个尾窗。

    ``tail_exempt_from_max``：末窗是 model B 的溢出槽，**在被拆过之前豁免
    ``max_states_per_window``**（仅路径版本层）。拆过之后两个孩子都受约束，溢出槽消失 ——
    此后插点无处可去，唯一出口是报「输入 λ 数目不够」并退出进程（控制器
    **不得**自行加总态数重跑 preopt：总态数是输入，属于 prescribed path 的定义）。
    调用方应从路径版本链里数 ``split_tail_window`` 事件来决定这个标志。

    只覆盖**结构性**动作。"延长采样 / 重标定 f_k"不改结构、永远可行，不在此列。
    """
    ranges = [(int(a), int(b)) for a, b in window_ranges]
    if not ranges:
        raise ValueError("window_ranges 不能为空")
    if ranges != sorted(ranges):
        raise ValueError(f"window_ranges 必须按起点升序: {ranges}")
    lo_n = int(min_states_per_window)
    hi_n = int(max_states_per_window)
    n_ins = int(n_insert)
    if n_ins < 1:
        raise ValueError(f"n_insert 必须 >= 1，收到 {n_insert}")
    if int(ranges[-1][1]) != int(n_states):
        raise ValueError(
            f"末窗 {ranges[-1]} 必须覆盖到最后一个态（n_states={n_states}）"
        )

    if layer not in ("path_version", "execution"):
        raise ValueError(f"未知 layer {layer!r}；只接受 path_version / execution")
    tail_lo, tail_hi = ranges[-1]
    k_tail = tail_hi - tail_lo
    out: Dict[str, Optional[str]] = {}

    # ---- 执行层：末窗**取消**豁免，且必须另有难度判据 ----
    # [老板] 「execution 层取消末窗豁免；overflow 只允许作为**内部路径记账状态**，
    #        不能直接拿去采样。」
    if layer == "execution":
        # 执行层一律不豁免，调用方传什么都不算数。
        tail_exempt_from_max = False
        if k_tail > hi_n:
            out["run_tail_window"] = (
                f"执行层不豁免上限：末窗 K={k_tail} > max_states_per_window={hi_n}。"
                "采样前必须先把 overflow materialize 成两个尾窗。"
            )
        elif tail_difficulty_ok is None:
            # **不拿 K<=hi 冒充可执行。** 实测 K=4 的末窗满足 max 却仍是最难的那个。
            out["run_tail_window"] = (
                f"末窗 K={k_tail} 满足 max_states={hi_n}，但**没有提供难度判据**"
                "（tail_difficulty_ok=None）⟹ fail-closed。K<=max 不足以说明可执行："
                "实测 K=4 的末窗满足 max、却是全场最难的那个（预热就停了，"
                "g 22.8→143.3）。执行层需要 max_states **加上** 热力学/动力学难度上限"
                "（Fisher / metric integral / 预测 support / N_eff/g 剖面）。"
            )
        elif not tail_difficulty_ok:
            out["run_tail_window"] = (
                f"末窗 K={k_tail} 满足 max_states，但**超过难度上限** ⟹ "
                "采样前把 overflow materialize 成两个尾窗。"
            )
        else:
            out["run_tail_window"] = None

    # ---- 插 λ（model B）：ranges 不动、末窗吸收 n 个溢出态 ----
    tail_k_split_max = 2 * hi_n - 1
    if tail_exempt_from_max and k_tail + n_ins > tail_k_split_max:
        # 豁免上限 ≠ 可以无限长：超过 2·hi−1 之后末窗永远拆不开，而拆它正是
        # 唯一的收尾动作。见 insert_lambda_in_failed_ibs_window 里同名守卫。
        out["insert_lambda"] = (
            f"末窗再吸收 {n_ins} 个态会到 K={k_tail + n_ins}，超过可拆上限 "
            f"2·max−1={tail_k_split_max} ⟹ 此后永远切不出合法子窗，溢出槽变死胡同。"
            f"先拆末窗（K 现在 {k_tail}，可拆区间 {2 * lo_n - 1}..{tail_k_split_max}），"
            "或判 HALT_LAMBDA_BUDGET_INSUFFICIENT。"
        )
    elif tail_exempt_from_max:
        out["insert_lambda"] = None
    elif k_tail + n_ins <= hi_n:
        out["insert_lambda"] = None
    else:
        out["insert_lambda"] = (
            f"末窗已被拆过（不再豁免 max_states_per_window={hi_n}），"
            f"当前 K_tail={k_tail}，再吸收 {n_ins} 个态会到 {k_tail + n_ins} > {hi_n}"
            " —— 溢出槽已耗尽。这不是「再想个办法」的问题：**输入的 λ 总数从一开始"
            "就不够**。应退出进程、由人工改输入文件的 λ 数目后重跑"
            "（HALT_LAMBDA_BUDGET_INSUFFICIENT）。"
        )

    # ---- 拆**末窗本身**：两子窗共享一个边界态 ⟹ p + q − 1 = K，两侧都落在 [lo, hi] ----
    # 这一条问的是「溢出槽长到能一分为二了没有」，是 model B 插点的收尾条件。
    split_ok = any(
        p + q - 1 == k_tail and lo_n <= p <= hi_n and lo_n <= q <= hi_n
        for p in range(lo_n, hi_n + 1)
        for q in range(lo_n, hi_n + 1)
    )
    out["split_last_window_in_two"] = None if split_ok else (
        f"末窗 K_tail={k_tail} 在 [{lo_n},{hi_n}] 下切不出两个合法子窗"
        f"（需 p+q−1={k_tail} 且两侧都在区间内 ⟹ 最小可拆 K = {2 * lo_n - 1}）。"
        "继续插 λ 让末窗长大即可到达；拆窗只是末窗溢出压不住时的收尾动作。"
    )

    # ---- `SPLIT_TAIL_WINDOW` 这个**动作**的可行性 ----
    # 🔑🔑 [审计 #24，2026-09-14] **判据问的问题和动作做的事不是一回事。**
    #
    # 控制器的 `SPLIT_TAIL_WINDOW` 落到执行器是 `repartition_tail_from_anchor`：
    # 从 `first_untrusted_window` 的**首态**起，把**整个尾段**重新分窗
    # （anchor 之前逐字冻结）。而先前这里给的可行性是「**末窗**能不能一分为二」
    # （K_tail ∈ [2lo−1, 2hi−1]）—— 两个完全不同的问题。
    #
    # 后果是双向的：
    #   · 标准 23 态布局末窗 K=4 < 7 ⟹ 这个动作**永久不可行**，
    #     而 `repartition_tail_from_anchor` 明明能处理中段重分 ⟹ 设计里那条
    #     「从第一个不可信窗口起重分」的主力修复路径**一次都没被选中过**；
    #   · 反过来 K_tail ∈ [7,9] 被放行时，实际改动范围远大于「拆末窗」——
    #     anchor 之后所有窗口都被重写、下游证据全部作废，而可行性检查对此一无所知。
    #
    # 现在：调用方给出重分起点（`tail_repartition_start_state`，即 anchor 所在窗口
    # 的首态全局下标）时，按**真正要做的那件事**判；给不出就如实说「判不了」，
    # **不再拿末窗那条冒充**。
    # 数学：相邻窗口共享一个边界态 ⟹ m 个窗口覆盖 T 个态需 Σk_i −(m−1) = T，
    # 每个 k_i ∈ [lo,hi] ⟹ m(lo−1)+1 ≤ T ≤ m(hi−1)+1。要「缩跨度」还得比现有
    # 窗口数**更细**，所以要求存在 m ≥ m0+1 满足它（m0 = 当前尾段窗口数）。
    _t_start = tail_repartition_start_state
    if _t_start is None:
        out["split_tail_window"] = (
            "判不了：`SPLIT_TAIL_WINDOW` 实际做的是**从 anchor 起重分整个尾段**"
            "（`repartition_tail_from_anchor`），而调用方没给重分起点 "
            "`tail_repartition_start_state`。**不拿「末窗能否一分为二」冒充** —— "
            f"那是另一个问题（它的答案在 `split_last_window_in_two`："
            f"{'可行' if split_ok else '不可行'}）。"
        )
    else:
        _t_start = int(_t_start)
        _starts = [a for a, _b in ranges]
        if _t_start not in _starts:
            out["split_tail_window"] = (
                f"重分起点 {_t_start} 不是任何现有窗口的首态（窗口起点 {_starts}）⟹ "
                "它不是一个共享边界态，`repartition_tail_from_anchor` 会 fail-closed 抛错。"
            )
        else:
            _t_states = int(n_states) - _t_start
            _m0 = sum(1 for a in _starts if a >= _t_start)
            _ok_ms = [m for m in range(_m0 + 1, _t_states + 2)
                      if m * (lo_n - 1) + 1 <= _t_states <= m * (hi_n - 1) + 1]
            out["split_tail_window"] = None if _ok_ms else (
                f"从态 {_t_start} 起的尾段有 {_t_states} 个态、现为 {_m0} 个窗口，"
                f"在 [{lo_n},{hi_n}] 下切不出**更细**的合法分窗"
                f"（需存在 m ≥ {_m0 + 1} 使 m·{lo_n - 1}+1 ≤ {_t_states} ≤ m·{hi_n - 1}+1）。"
                "缩跨度得另找动作（插 λ / 固定 λ 表上的有界重窗）。"
            )
    return out


def insert_lambda_in_failed_ibs_window(
    lambdas_var: List[float],
    window_ranges: List[Tuple[int, int]],
    failed_range: Tuple[int, int],
    pilot_lambdas: List[float],
    pilot_cumulative_length: List[float],
    *,
    min_states_per_window: int = 4,
    max_states_per_window: int = 5,
    partition_criterion: str = "arclength",
    pilot_metric_g: Optional[Sequence[float]] = None,
    n_insert: Optional[int] = None,
) -> Tuple[List[float], List[Tuple[int, int]], Dict]:
    """在失败窗口内插 λ，**缩小该窗口的 λ 跨度**；多出来的态由**末窗**吸收。

    ⚠️ 2026-09-11 重写。设计依据：``docs/archive/PLAN_PATH_REPAIR_2026-09-11.md`` §2 更正
    与 §3ter。两条关键改动，改前先读那份文档。

    **记账口径 = model B：窗口的下标区间一律不动，只有末窗上界 += n。**

    插一个 λ 之后，窗口区间"跟不跟着走"是两套记账，物理效果相反::

        原始           win0 = (0,4) 装 {λ0,λ1,λ2,λ3}      跨度 λ0→λ3
        model A        win0 = (0,5) 装 {λ0,λ1,λnew,λ2,λ3} 跨度 λ0→λ3 **不变**
        model B(本函数) win0 = (0,4) 装 {λ0,λ1,λnew,λ2}    跨度 λ0→λ2 **缩了**

    ``ΔF(λ0→λ3)`` 是物理量，与中间放几个点无关 —— 所以 **model A 插点之后 bias
    要压平的总落差一分没少，对"窗口太宽"是无效动作**。只有 model B 真的缩小了
    窗口要跨的自由能落差，这才是"补 λ 治窗口太宽"的机制。

    代价与安全性：

      · **前缀（插入点之前的窗口）λ 逐位不变** —— 已采完的窗口全部复用，后置断言保证。
      · 下游窗口的 λ 内容左移一格。窗口是按 0,1,2… 顺序跑的，win_i 失败时
        i+1..N **还没采**，所以顺序跑时这是零成本。
      · resume 安全：``lambda_path_versions.append_version`` 同时记 λ 表与
        window_ranges，``resolve_path`` 取回当前版本，``changed_windows()`` 给
        可复用清单，插点事件带确定性 ID 防重复插。
      · **末窗吸收溢出，因此豁免 ``max_states_per_window``。** 非末窗的态数永远
        是布局期定的那个值（本函数不改，后置断言保证）。

    **本函数不拆窗。** 拆窗只发生在末窗、只由"末窗 K 累积到 f_k 重标定后仍压不住"
    触发，是另一条路径的事。旧实现在这里无条件就地拆窗，而且为了凑拆窗门槛
    ``n = max(1, (2*lo−1) − K)`` 强行插点 —— 那些 λ 没有任何边级证据支持，纯粹是
    成本（多几个态要采）。插点是**边级**工具，不该为窗口级簿记门槛服务。

    ``n_insert``：

      · 给了就用它（调用方按窗口级判据决定插几个，这是控制器的职责）。
      · 不给则走**边级**判据：``n = Σ_边 max(0, ⌈L_edge / L_target⌉ − 1)``，
        ``L_target`` = 全路径平均边长。窗内没有任何边超过 L_target 时**明着报错**，
        不静默插 1 个 —— "边级证据不支持插点"是一个有意义的结论。

    插点**位置** = 窗内最长边在**配置度量**下的中点（pilot 反插值）。
    ``partition_criterion`` 贯穿"定 n / 选边 / 定位"三处：初始布局按哪个度量
    定的布点，补救就按哪个 —— 旧实现里它只影响拆窗切点，拆窗一去掉就成了
    无效参数，而"缺 metric_g 就 fail-closed"那条守卫正是为了防止这种悄悄换判据。

    ⚠️ 措辞：这是**按 pilot 实测长度插值得到的候选加密位置**，不是本次 VALIDATE
    已证实的故障边。调用方应把它标成"学习失败后的补救尝试"。
    """
    lambdas = [float(x) for x in lambdas_var]
    ranges = [(int(a), int(b)) for a, b in window_ranges]
    failed = (int(failed_range[0]), int(failed_range[1]))
    if failed not in ranges:
        raise RuntimeError(f"失败窗口 {failed} 不在当前窗口列表 {ranges} 中")
    if ranges != sorted(ranges):
        raise RuntimeError(f"window_ranges 必须按起点升序（model B 依赖末窗是最后一项）: {ranges}")
    if ranges[-1][1] != len(lambdas):
        raise RuntimeError(
            f"末窗 {ranges[-1]} 必须覆盖到最后一个态（n_states={len(lambdas)}）"
        )
    lo_n, hi_n = int(min_states_per_window), int(max_states_per_window)
    if failed[1] - failed[0] < 2:
        raise RuntimeError(f"失败窗口 {failed} 不足两个态，无法在其中插点")

    pilot_lam = np.asarray(pilot_lambdas, dtype=float).ravel()
    pilot_s = np.asarray(pilot_cumulative_length, dtype=float).ravel()
    if pilot_lam.size != pilot_s.size or pilot_lam.size < 2:
        raise ValueError("pilot lambda 与累计热力学长度必须等长且至少含两个点")
    if not np.all(np.diff(pilot_s) > 0.0):
        raise ValueError("pilot 累计热力学长度必须严格递增")

    criterion = str(partition_criterion).lower()
    if criterion not in ("arclength", "metric_integral", "state_count"):
        raise ValueError(f"未知分窗判据 {partition_criterion!r}")
    if criterion in ("metric_integral", "state_count") and pilot_metric_g is None:
        # 初始布局按 ∫g 定的度量，补救却按 ∫√g 定 n，等于第一次自动补救就换了判据。
        # 拿不到 metric_g 就明着拒绝，不静默退回等弧长。
        raise ValueError(
            f"partition_criterion={criterion!r} 必须同时提供 pilot_metric_g；"
            "拒绝静默退回等弧长——那会让补救悄悄换掉度量。"
        )

    start, end = failed
    is_last = (failed == ranges[-1])
    original_span = abs(lambdas[end - 1] - lambdas[start])
    n_states_before = len(lambdas)
    w_before = len(ranges)

    # 🔑 **判据必须贯穿到底。** 旧实现里 partition_criterion 只影响拆窗切点，
    # 选边与定位一律按弧长；拆窗一去掉，判据就变成完全无效的参数了 —— 而那条
    # "缺 metric_g 就 fail-closed"的守卫存在的全部意义就是防止补救悄悄换判据。
    # 所以这里把 pilot 的累计度量换成配置的那一个，选边（比较）和定位（反插值）
    # 都用它。两个方向都只是对 (pilot_lam, pilot_measure) 做 np.interp。
    if criterion == "metric_integral":
        pilot_measure = np.asarray(
            metric_integral_cumulative(list(pilot_lam), pilot_lambdas, pilot_metric_g),
            dtype=float,
        )
        if not np.all(np.diff(pilot_measure) > 0.0):
            raise ValueError(
                "pilot 的 ∫g 累计度量不是严格递增，无法用于选边/反插值"
            )
    else:
        pilot_measure = pilot_s

    def _measure(vals) -> np.ndarray:
        """把一串 λ 换算成配置度量下的累计坐标（用于比较边长）。"""
        return np.asarray(
            [_pilot_arclength_of(x, pilot_lam, pilot_measure) for x in vals],
            dtype=float,
        )

    # ---- 插几个 ----
    if n_insert is None:
        cum_all = _measure(lambdas)
        all_edges = np.abs(np.diff(cum_all))
        if all_edges.size == 0 or not np.all(np.isfinite(all_edges)):
            raise RuntimeError("全路径边长不可用，无法定 L_target")
        l_target = float(np.mean(all_edges))
        if not (l_target > 0.0):
            raise RuntimeError(f"L_target={l_target} 非正，无法定插点数")
        win_edges = np.abs(np.diff(cum_all[start:end]))
        # 🔑 [2026-09-11] 必须留相对容差。L_target 是**全路径均值**，而生产布局
        # 本来就是按热力学长度等分的 ⟹ 每条边都恰好≈均值，`e/L_target` 在 1 的
        # 两侧抖 ~1e-15。裸 ceil 会把高出 1e-15 的那半数边判成"要插点"：实测
        # 16 个等距 λ、边长逐位不等仅 1.5e-15，就算出 n_needed=3（应为 0）。
        # 于是"边级证据不支持插点"这道守卫被浮点噪声整个击穿。
        _rel_tol = 1e-9
        n_needed = int(
            sum(
                max(0, int(np.ceil(float(e) / l_target - _rel_tol)) - 1)
                for e in win_edges
            )
        )
        if n_needed == 0:
            raise RuntimeError(
                f"失败窗口 {failed} 内没有任何一条边超过全路径平均热力学长度 "
                f"L_target={l_target:.4f}（窗内各边 "
                f"{[round(float(e), 4) for e in win_edges]}）—— **边级证据不支持插点**。"
                "这个窗口的缺陷是窗口级的（跨度 / 混合 / f_k），插几个点必须由调用方"
                "按窗口级判据给出 n_insert；本函数不替它猜，也不静默插 1 个。"
                "见 docs/archive/PLAN_PATH_REPAIR_2026-09-11.md §3ter.4。"
            )
    else:
        n_needed = int(n_insert)
        if n_needed < 1:
            raise ValueError(f"n_insert 必须 >= 1，收到 {n_insert}")

    # ---- 插在哪：窗内最长边的 pilot 弧长中点（与重写前逐字一致） ----
    inserted: List[float] = []
    inserted_edges: List[List[int]] = []
    for _ in range(n_needed):
        arc = [_pilot_arclength_of(lambdas[i], pilot_lam, pilot_measure)
               for i in range(start, end)]
        edges = [abs(arc[i + 1] - arc[i]) for i in range(len(arc) - 1)]
        if not edges or max(edges) <= 0.0:
            raise RuntimeError(f"窗口 {(start, end)} 内各边度量长度为零，无法选插点")
        worst = int(np.argmax(edges))
        left = start + worst
        lam_mid = float(
            np.interp(0.5 * (arc[worst] + arc[worst + 1]), pilot_measure, pilot_lam)
        )
        a_lo, a_hi = sorted((lambdas[left], lambdas[left + 1]))
        if not a_lo < lam_mid < a_hi:
            raise RuntimeError(
                f"热力学中点未严格落在边内部：{lambdas[left]}, {lam_mid}, {lambdas[left + 1]}"
            )
        lambdas.insert(left + 1, lam_mid)
        inserted.append(lam_mid)
        inserted_edges.append([int(left), int(left + 1)])
        # 🔑 **ranges 不动。** 插入之后 lambdas[start:end] 的内容整体左移一格，
        # 失败窗口的跨度因此缩小；被挤出去的那个态归下一个窗口。

    # ---- 末窗吸收溢出；其余区间逐字不变 ----
    # 🔑 [2026-09-11] **溢出槽有上界，别让它变成死胡同。**
    # 末窗豁免 max_states_per_window，但"能一分为二"这件事本身有区间：
    # 两子窗共享一个边界态 ⟹ p+q−1=K，两侧都要落在 [lo,hi] ⟹ 可拆的 K 只有
    # ``2*lo−1 .. 2*hi−1``（4/5 配置下就是 7..9，只有三个值宽）。末窗一旦被插到
    # K > 2*hi−1 就**永远拆不开**了，溢出槽从此是死胡同：既不能再吸收（迟早
    # 压不住），也不能拆（切不出合法子窗），而唯一的收尾动作恰恰是拆它。
    # 当前默认 max_insertions=3 只是**巧合**地把 K 压在 5+3=8 ≤ 9 以内
    # （`_run_stage2_with_path_evolution` 的 rounds_done 是跨 resume 累计的），
    # 那是巧合不是守卫 —— 谁把 max_insertions 调大就会踩进去。所以这里 fail-closed。
    _tail_k_after = (ranges[-1][1] + n_needed) - ranges[-1][0]
    _tail_k_split_max = 2 * hi_n - 1
    if _tail_k_after > _tail_k_split_max:
        raise RuntimeError(
            f"末窗 {ranges[-1]}（当前 {ranges[-1][1] - ranges[-1][0]} 态）再吸收 "
            f"{n_needed} 个溢出态会到 K={_tail_k_after}，超过**可拆上限** "
            f"2·max−1={_tail_k_split_max}（可拆区间 "
            f"{2 * lo_n - 1}..{_tail_k_split_max}）—— 此后末窗永远切不出两个落在 "
            f"[{lo_n},{hi_n}] 的子窗，溢出槽变成死胡同。"
            + (
                f"⚠️ 这次的 n={n_needed} 是**边级判据**算出来的（窗内有边长到需要"
                f"这么多点），而溢出预算只有 {_tail_k_split_max - (ranges[-1][1] - ranges[-1][0])} "
                "个态 —— 两者之间没有任何约束保证前者 ≤ 后者。一条边长到平均的几倍，"
                "本身说明**初始布点**就不对（度规布点不该产生这种边），那是布局期的"
                "问题，不是运行时修补能补的。"
                if n_insert is None else
                "这次的 n 是调用方给的（窗口级判据）。"
            )
            + "正确处置二选一：(a) 趁 K 还在可拆区间内**先拆末窗**，拆完的右孩子重新"
            "成为溢出槽（注意：孩子受 hi 约束，只能再腾出很少几个）；"
            "(b) 承认这是 HALT_LAMBDA_BUDGET_INSUFFICIENT —— 输入的 λ 总数从一开始"
            "就不够，退出进程、人工改输入文件的 λ 数目。**不要把 n 封顶到预算**，"
            "那会静默产出一个不满足触发它那份证据的布局；理由见 "
            "docs/archive/PLAN_PATH_REPAIR_2026-09-11.md §2。"
        )
    new_ranges = list(ranges[:-1]) + [(ranges[-1][0], ranges[-1][1] + n_needed)]
    validate_single_shared_boundary_ranges(new_ranges, len(lambdas))

    # ---- 后置断言：model B 的三条不变量 ----
    if len(lambdas) != n_states_before + n_needed:
        raise RuntimeError(
            f"插点后态数应为 {n_states_before + n_needed}，实得 {len(lambdas)}"
        )
    # (1) 前缀（插入点之前、可能已经采完的窗口）λ 必须逐位不变。
    for (a0, b0) in ranges:
        if b0 <= start + 1:
            before_vals = [round(float(x), 12) for x in list(lambdas_var)[a0:b0]]
            after_vals = [round(float(x), 12) for x in lambdas[a0:b0]]
            if before_vals != after_vals:
                raise RuntimeError(
                    f"前缀窗口 {(a0, b0)} 装的 λ 被改动了（model B 要求逐位不变）：\n"
                    f"  before={before_vals}\n  after ={after_vals}"
                )
    # (2) 非末窗的态数必须一个都没变（溢出只许落末窗）。
    for (a0, b0), (a1, b1) in zip(ranges[:-1], new_ranges[:-1]):
        if (b1 - a1) != (b0 - a0):
            raise RuntimeError(
                f"非末窗 {(a0, b0)} → {(a1, b1)} 的态数变了；溢出只允许由末窗吸收"
            )
        if (b1 - a1) > hi_n:
            raise RuntimeError(
                f"非末窗 {(a1, b1)} 有 {b1 - a1} 个态，超过 max_states_per_window={hi_n}"
            )
    if (new_ranges[-1][1] - new_ranges[-1][0]) != (ranges[-1][1] - ranges[-1][0]) + n_needed:
        raise RuntimeError("末窗没有吸收全部溢出态")
    # (3) 失败窗口的 λ 跨度必须真的变小 —— 除非它本身就是末窗（末窗是溢出槽，
    #     它的正常行为是**变大**，缩不了；那种情况下该做的是拆末窗，不是插点）。
    new_span = abs(lambdas[new_ranges[-1][1] - 1] - lambdas[start]) if is_last \
        else abs(lambdas[end - 1] - lambdas[start])
    if not is_last and not (new_span < original_span - 1e-12):
        raise RuntimeError(
            f"插点没有缩小失败窗口的 λ 跨度（{original_span:.6f} → {new_span:.6f}），"
            "拒绝返回。"
        )

    return lambdas, new_ranges, {
        "source": "ibs_failed_window_pilot_interpolated_midpoint_insertion",
        "accounting": "model_B_fixed_ranges_tail_absorbs_overflow",
        "note": "pilot 插值候选加密位置，非本次 VALIDATE 证实的故障边",
        "failed_global_state_range": [int(failed[0]), int(failed[1])],
        "failed_window_is_last": bool(is_last),
        "n_inserted": int(n_needed),
        "n_insert_source": "caller" if n_insert is not None else "edge_length",
        "inserted_lambdas": [float(x) for x in inserted],
        "inserted_lambda": float(inserted[-1]) if inserted else None,
        "inserted_global_edges": inserted_edges,
        "failed_global_edge": list(inserted_edges[0]) if inserted_edges else None,
        "partition_criterion": criterion,
        "n_windows_before": int(w_before),
        "n_windows_after": int(len(new_ranges)),
        "failed_window_span_before": float(original_span),
        "failed_window_span_after": float(new_span),
        "tail_window_before": list(ranges[-1]),
        "tail_window_after": list(new_ranges[-1]),
        "frozen_prefix_windows": [list(r) for r in ranges if r[1] <= start + 1],
        "min_states_per_window": lo_n,
        "max_states_per_window": hi_n,
        "max_states_per_window_exempt": "tail_window",
    }


def insert_thermodynamic_midpoint_from_ibs_lse_failure(
    lambdas_var: List[float],
    window_ranges: List[Tuple[int, int]],
    warmup_diagnostics: Dict,
    pilot_lambdas: List[float],
    pilot_cumulative_length: List[float],
) -> Tuple[List[float], List[Tuple[int, int]], Dict]:
    """Bridge an irreducible two-state LSE failure with one measured midpoint.

    The inserted coordinate is halfway in the pilot thermodynamic coordinate,
    not the arithmetic lambda midpoint.  The failed [a,b] ensemble is replaced
    by [a,m] and [m,b], and both must later pass a fresh IBS LSE design probe.
    """
    lambdas = [float(x) for x in lambdas_var]
    ranges = [(int(s), int(e)) for s, e in window_ranges]
    failed = tuple(int(x) for x in warmup_diagnostics["global_state_range"])
    if failed not in ranges:
        raise RuntimeError(f"LSE 失败窗口 {failed} 不在当前窗口列表 {ranges} 中")
    start, end = failed
    if end - start != 2:
        raise RuntimeError("只有不可再拆的两态 IBS 窗口允许插入 lambda")

    pilot_lam = np.asarray(pilot_lambdas, dtype=float).ravel()
    pilot_s = np.asarray(pilot_cumulative_length, dtype=float).ravel()
    if pilot_lam.size != pilot_s.size or pilot_lam.size < 2:
        raise ValueError("pilot lambda 与累计热力学长度必须等长且至少含两个点")
    if not np.all(np.diff(pilot_s) > 0.0):
        raise ValueError("pilot 累计热力学长度必须严格递增")

    lambda_left, lambda_right = lambdas[start], lambdas[start + 1]
    # pilot lambda descends while np.interp requires ascending xp.
    s_left = float(np.interp(lambda_left, pilot_lam[::-1], pilot_s[::-1]))
    s_right = float(np.interp(lambda_right, pilot_lam[::-1], pilot_s[::-1]))
    s_mid = 0.5 * (s_left + s_right)
    lambda_mid = float(np.interp(s_mid, pilot_s, pilot_lam))
    lo, hi = sorted((lambda_left, lambda_right))
    if not lo < lambda_mid < hi:
        raise RuntimeError(
            f"热力学中点未严格位于失败边内部: {lambda_left}, {lambda_mid}, {lambda_right}"
        )

    insert_at = start + 1
    new_lambdas = list(lambdas)
    new_lambdas.insert(insert_at, lambda_mid)
    shifted_end = end + 1
    new_ranges = []
    for current_start, current_end in ranges:
        if (current_start, current_end) == failed:
            new_ranges.extend([(start, insert_at + 1), (insert_at, shifted_end)])
        elif current_end <= insert_at:
            new_ranges.append((current_start, current_end))
        elif current_start >= insert_at:
            new_ranges.append((current_start + 1, current_end + 1))
        else:
            new_ranges.append((current_start, current_end + 1))
    new_ranges = canonicalize_window_ranges(new_ranges, len(new_lambdas))
    return new_lambdas, new_ranges, {
        "source": "ibs_lse_design_thermodynamic_midpoint_insertion",
        "failed_global_state_range": [start, end],
        "failed_global_edge": [start, start + 1],
        "failed_lambdas": [lambda_left, lambda_right],
        "inserted_global_state": int(insert_at),
        "inserted_lambda": lambda_mid,
        "inserted_thermodynamic_coordinate": s_mid,
        "replacement_ranges": [[start, insert_at + 1], [insert_at, shifted_end]],
        "lse_balance": warmup_diagnostics.get("lse_balance"),
    }


def split_window_from_warmup_failure(
    lambdas_var: List[float],
    window_ranges: List[Tuple[int, int]],
    warmup_diagnostics: Dict,
    min_states_before_split: int = 5,
    min_states_per_window_floor: int = 3,
) -> Tuple[List[float], List[Tuple[int, int]], Dict]:
    """Split a failed IBS window without inventing a new thermodynamic state.

    Warmup probabilities are measured in a time-dependent IBS mixture, so
    ``std(beta Delta-u)`` from that mixture is not a valid estimate of the
    fixed-lambda thermodynamic metric.  A warmup coverage failure therefore
    changes only the window partition.  The two children share exactly one
    existing lambda state, which is sufficient to stitch their free energies.

    The initial pilot layout intentionally uses a wide cumulative-overlap
    budget (``pilot_overlap_thermodynamic_length``), so neighboring windows
    can legitimately share several states -- e.g. a 6-state parent sharing 3
    states with its next neighbor. That is by design, not a bug. But once this
    parent SPLITS, only ``start`` is preserved on the left child and only
    ``end`` on the right child -- the right child's ``end`` is identical to
    the parent's, so it still inherits the *same* multi-state overlap with
    whatever untouched neighbor came after the parent. For the right child
    (now much smaller than the original parent) that overlap can become a
    large fraction of its own span (observed: a 4-state child sharing 3 states
    -- 75% -- with an untouched 6-state neighbor it used to share only 3-of-6
    with). ``canonicalize_window_ranges`` deliberately does not touch this
    (partial, non-containing overlap is its own legitimate case), so this must
    be fixed here, right after the split: reduce the immediate next
    neighbor's overlap with the new right child down to exactly one shared
    state, the same convention used everywhere else new windows get stitched
    together. Nothing else is re-laid-out -- only ``start`` moves on that one
    neighbor, which cannot change its own overlap with whatever follows IT
    (that overlap is governed by its unchanged ``end``), so this does not
    cascade any further down the path.
    """
    lambdas = [float(x) for x in lambdas_var]
    ranges = [(int(s), int(e)) for s, e in window_ranges]
    failed = tuple(int(x) for x in warmup_diagnostics["global_state_range"])
    if failed not in ranges:
        raise RuntimeError(f"warmup 失败窗口 {failed} 不在当前窗口列表 {ranges} 中")
    start, end = failed
    if end - start < int(min_states_before_split):
        raise RuntimeError(
            f"warmup 失败窗口只有 {end-start} 个态，不能再盲拆；必须使用 fixed-lambda overlap 探针"
        )

    # m is an existing global state.  [start:m+1) and [m:end) share only m.
    # A 2-state IBS window is statistically fragile (see run_all_windows'
    # own comment to that effect), so each child must have >= 3 states;
    # sharing exactly 1 state means the parent needs >= 3+3-1=5 states for
    # this to even be possible -- enforced by min_states_before_split's
    # default above, not just by this floor check.
    middle = (start + end - 1) // 2
    children = [(start, middle + 1), (middle, end)]
    if min(e - s for s, e in children) < 3:
        raise RuntimeError(f"拆分会产生少于三个态的窗口: {children}")
    right_child_end = children[-1][1]

    # 找到失败窗口在原列表中的位置，只调整紧随其后的那一个邻窗（如果存在且
    # 目前跟新右孩子共享超过一个态）；不触碰失败窗口左侧的邻窗——左孩子的
    # start 跟原失败窗口完全相同，它与左侧邻窗的重叠（由左侧邻窗的 end 决定）
    # 不受这次拆分影响。
    failed_pos = ranges.index(failed)
    next_start_override: Optional[int] = None
    if failed_pos + 1 < len(ranges):
        next_s, next_e = ranges[failed_pos + 1]
        if next_s < right_child_end - 1:
            candidate_start = right_child_end - 1
            if next_e - candidate_start >= int(min_states_per_window_floor):
                next_start_override = candidate_start
            # 否则调整后邻窗会小于最小态数下限，保留原样，不强行压缩——
            # 这种情形应该很少见（邻窗本身已经接近最小尺寸）。

    new_ranges: List[Tuple[int, int]] = []
    neighbor_adjustment = None
    for idx, current in enumerate(ranges):
        if current == failed:
            new_ranges.extend(children)
        elif idx == failed_pos + 1 and next_start_override is not None:
            adjusted = (next_start_override, current[1])
            neighbor_adjustment = {
                "old_range": list(current),
                "new_range": list(adjusted),
            }
            new_ranges.append(adjusted)
        else:
            new_ranges.append(current)
    covered = sorted({i for s, e in new_ranges for i in range(s, e)})
    if covered != list(range(len(lambdas))):
        raise RuntimeError(f"拆窗后未完整覆盖 lambda 路径: {covered}")

    feedback = {
        "source": "warmup_window_split_only",
        "failed_window": int(warmup_diagnostics.get("window_index", -1)),
        "failed_global_state_range": [start, end],
        "child_ranges": [list(r) for r in children],
        "shared_global_state": int(middle),
        "next_neighbor_reflowed_to_single_state_overlap": neighbor_adjustment,
        "inserted_lambda": None,
    }
    return lambdas, new_ranges, feedback


def insert_lambda_from_overlap_failure(
    lambdas_var: List[float],
    window_ranges: List[Tuple[int, int]],
    warmup_diagnostics: Dict,
) -> Tuple[List[float], List[Tuple[int, int]], Dict]:
    """Insert one state only after a real bidirectional fixed-H overlap failure.

    The coordinate midpoint is merely the next point to *measure*.  No child
    thermodynamic lengths are fabricated; the pilot cache is explicitly
    invalidated by the caller and the new edges must be measured next round.
    """
    overlap = warmup_diagnostics.get("bidirectional_overlap_probe", {})
    pairs = overlap.get("pairs", [])
    failed_pairs = [p for p in pairs if not bool(p.get("passed", False))]

    asymmetric = overlap.get("passed_but_asymmetric_bottleneck")
    preserve_expanded_parent = False

    if failed_pairs:
        worst = min(failed_pairs, key=lambda p: float(p.get("min_bidirectional_overlap", np.inf)))
        source = "fixed_hamiltonian_bidirectional_overlap"
    elif asymmetric and asymmetric.get("qualified"):
        worst = dict(asymmetric["pair"])
        preserve_expanded_parent = True
        source = "fixed_hamiltonian_passed_but_asymmetric_bottleneck"
    else:
        raise RuntimeError("没有 fixed-H 失败边或合格的通过但不对称瓶颈边，拒绝插点")

    global_edge = int(worst["global_edge"][0])
    lambdas = [float(x) for x in lambdas_var]
    if not 0 <= global_edge < len(lambdas) - 1:
        raise RuntimeError(f"fixed-lambda overlap 失败边索引越界: {global_edge}")

    midpoint = 0.5 * (lambdas[global_edge] + lambdas[global_edge + 1])
    insert_at = global_edge + 1
    new_lambdas = list(lambdas)
    new_lambdas.insert(insert_at, float(midpoint))

    failed_range = tuple(int(x) for x in warmup_diagnostics["global_state_range"])
    ranges = [(int(s), int(e)) for s, e in window_ranges]
    if failed_range not in ranges:
        raise RuntimeError(f"overlap 失败窗口 {failed_range} 不在当前窗口列表 {ranges} 中")

    # K<=4 是 fixed-H overlap 探针/MBAR 校准通道自身的准入上限
    # (ibs_engine.py: `K <= 4 and stage_type == "vdw"`)。合并成单一父窗口
    # 只在结果仍落在这个上限内时才安全——否则父窗口会静默失去重新进入该
    # 通道的资格，比制造一个两态脆弱子窗口更糟。
    if preserve_expanded_parent and (failed_range[1] + 1 - failed_range[0]) > 4:
        preserve_expanded_parent = False

    new_ranges: List[Tuple[int, int]] = []
    for start, end in ranges:
        if (start, end) == failed_range:
            shifted_end = end + 1
            if preserve_expanded_parent:
                # 三态父窗口插点后保留为四态窗口，避免产生 3态+2态。
                new_ranges.append((start, shifted_end))
            else:
                # Both children contain the new state at insert_at.
                new_ranges.extend([(start, insert_at + 1), (insert_at, shifted_end)])
        elif end <= insert_at:
            new_ranges.append((start, end))
        elif start >= insert_at:
            new_ranges.append((start + 1, end + 1))
        else:
            new_ranges.append((start, end + 1))

    covered = sorted({i for s, e in new_ranges for i in range(s, e)})
    if covered != list(range(len(new_lambdas))):
        raise RuntimeError(f"fixed-overlap 插点后未完整覆盖 lambda 路径: {covered}")
    feedback = {
        "source": source,
        "failed_window": int(warmup_diagnostics.get("window_index", -1)),
        "failed_global_edge": [global_edge, global_edge + 1],
        "selected_global_edge": [global_edge, global_edge + 1],
        "inserted_lambda": float(midpoint),
        "measured_min_bidirectional_overlap": float(worst["min_bidirectional_overlap"]),
        "overlap_threshold": float(worst["threshold"]),
        "preserved_expanded_parent_window": preserve_expanded_parent,
        "asymmetry_diagnostics": asymmetric if preserve_expanded_parent else None,
        "thermodynamic_lengths_invalidated": True,
    }
    return new_lambdas, new_ranges, feedback


def plan_vdw_overlap_repair_targets(
    window_ranges: List[Tuple[int, int]],
    window_overlap_diagnostics: List[Dict],
    min_overlap_threshold: float,
    min_states_before_split: int = 5,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Classify production-time low-ESS windows, without touching any lambda.

    Mirrors the warmup-failure split-first-then-probe policy
    (``split_window_from_warmup_failure`` / ``insert_lambda_from_overlap_failure``)
    instead of the old worst-per-lambda-state + arithmetic-midpoint path in
    ``refine_stage_lambda_path_by_overlap``: a whole window reporting low ESS is
    not evidence that one particular lambda edge is too wide (a saturated IBS
    bias or slow conformational relaxation depresses every state's ESS equally,
    which is exactly why the old code's "worst per-lambda state" pick could be
    pure noise -- see ``window 3`` in the reported case, where every state sat
    at min_ess_ratio~=0.0035). So a failing window is only ever split here
    (sharing one existing state, no new lambda), never bisected directly. The
    caller is expected to run a real fixed-Hamiltonian bidirectional overlap
    probe on any window this returns as un-splittable (already at or below
    ``min_states_before_split`` states) before allowing a lambda insertion.

    Returns ``(windows_to_split, windows_needing_probe)``, both lists of
    ``(start, end)`` tuples drawn verbatim from ``window_ranges`` (not
    reconstructed from the diagnostics' own ``lambdas`` field beyond using it
    to locate the match, so downstream code can keep operating on the caller's
    own range objects).
    """
    ranges = [(int(s), int(e)) for s, e in window_ranges]
    ranges_set = set(ranges)
    to_split: List[Tuple[int, int]] = []
    to_probe: List[Tuple[int, int]] = []
    seen = set()
    for rec in window_overlap_diagnostics or []:
        ratio = rec.get("min_ess_ratio")
        lambdas_idx = rec.get("lambdas")
        if ratio is None or not np.isfinite(ratio) or ratio >= min_overlap_threshold:
            continue
        if not lambdas_idx:
            continue
        start, end = int(min(lambdas_idx)), int(max(lambdas_idx)) + 1
        if (start, end) not in ranges_set or (start, end) in seen:
            continue
        seen.add((start, end))
        if (end - start) >= int(min_states_before_split):
            to_split.append((start, end))
        else:
            to_probe.append((start, end))
    return to_split, to_probe


def canonicalize_window_ranges(
    window_ranges: List[Tuple[int, int]],
    n_states: int,
) -> List[Tuple[int, int]]:
    """Remove exact duplicates and strictly-contained windows after a batch
    split, then verify the survivors still fully cover [0, n_states) with
    every adjacent pair (sorted by start) sharing at least one state.

    Splitting several overlapping *parent* windows independently -- one
    ``split_window_from_warmup_failure`` call per failing parent -- can
    produce a child that lands entirely inside a NEIGHBORING parent's span,
    because IBS windows overlap by design. Concretely: parents (0,6) and
    (3,9) (sharing states 3,4,5) each split independently via
    ``middle=(s+e-1)//2``: (0,6) -> (0,3),(2,6); (3,9) -> (3,6),(5,9). The
    child (3,6) is then a strict subset of the child (2,6) (both cover a
    span within {2,3,4,5}), a real case observed when 5 overlapping 6-state
    parents were all split in one round, producing 10 windows instead of the
    minimal connected 6-window chain. Coverage is never actually at risk from
    this (a contained window adds no lambda index its superset doesn't
    already have), but the redundant windows get sampled anyway -- wasted
    GPU time now, and unbounded window-count growth if a later round splits
    them again. This does NOT merge partially-overlapping-but-not-nested
    windows (neither contains the other) -- those provide genuine additional
    overlap and are kept as-is.
    """
    ranges = sorted({(int(s), int(e)) for s, e in window_ranges})
    kept: List[Tuple[int, int]] = []
    for s, e in ranges:
        # A strict subset of an already-kept window contributes no new
        # coverage/adjacency; skip it entirely.
        if any(ks <= s and e <= ke and (ks, ke) != (s, e) for ks, ke in kept):
            continue
        # A previously-kept window can only be a strict subset of this one
        # when they share the same start (sort order guarantees any kept
        # window with a smaller start cannot be contained in this one).
        kept = [
            (ks, ke) for ks, ke in kept
            if not (s <= ks and ke <= e and (ks, ke) != (s, e))
        ]
        kept.append((s, e))

    kept.sort()
    covered = sorted({i for s, e in kept for i in range(s, e)})
    if covered != list(range(n_states)):
        raise RuntimeError(
            f"窗口归约后未完整覆盖 [0,{n_states})，覆盖到 {covered}"
        )
    for (s0, e0), (s1, e1) in zip(kept, kept[1:]):
        if s1 >= e0:
            raise RuntimeError(
                f"窗口归约后相邻窗口不再共享任何状态: {(s0, e0)} 与 {(s1, e1)}"
            )
    for i, (si, ei) in enumerate(kept):
        for j, (sj, ej) in enumerate(kept):
            if i != j and sj <= si and ei <= ej:
                raise RuntimeError(
                    f"窗口归约后仍残留嵌套窗口: {(si, ei)} 被 {(sj, ej)} 严格包含"
                )
    return kept


# #[P1 FIX] 抽取共享方差归一化逻辑，消除重复代码
# abfe_preoptimizer.py 顶部 (约第 15 行)
def _normalize_variance_weights(std_dev_clipped, max_ratio=0.15):
    """共享方差归一化函数，消除代码重复"""
    density_weight = np.log1p(std_dev_clipped) + 0.1
    max_weight = np.sum(density_weight) * max_ratio
    clipped = np.clip(density_weight, None, max_weight)
    # ✅ 修复：Clip 后必须重新归一化，保证 ∑w = 1（等熵长度分布前提）
    return clipped / (np.sum(clipped) + 1e-10)


def finalize_descending_lambda_path(
    optimized_lambdas: np.ndarray,
    target_n_states: int,
    min_spacing: Optional[float] = None,
) -> Tuple[np.ndarray, float, bool]:
    """Shared post-interpolation invariant enforcement for a descending
    (1.0 -> 0.0) CDF-interpolated lambda path.

    Both ``ABFEPreOptimizer.optimize_lambda_path_adaptive`` (single-lambda
    vdw path) and ``DualLambdaPreOptimizer.optimize_stage1_decharging``
    (dual-lambda decharging path) build their own ``optimized_lambdas`` array
    from CDF interpolation, using different density-weight formulas -- that
    physics-specific weighting is intentionally left to each caller. But the
    *invariants* the result must satisfy afterward (finite, bounded to
    [0,1], strictly descending with a minimum spacing, deduplicated, and a
    fail-closed fallback to a linear path if too few distinct states survive)
    are identical, and used to be duplicated only in the single-lambda path;
    ``optimize_stage1_decharging`` only clipped/sorted/pinned endpoints with
    no min-spacing, no dedup, and no fail-closed fallback -- a valid-looking
    CDF interpolation could silently hand back two states with (numerically)
    the same lambda, which breaks MBAR's distinct-state assumption for that
    edge without ever raising or logging.

    Returns ``(lambdas, min_spacing_used, fell_back_to_linear)`` so a caller
    can log/record what was actually applied.
    """
    optimized_lambdas = np.asarray(optimized_lambdas, dtype=float).ravel()
    target_n_states = int(target_n_states)
    if not np.all(np.isfinite(optimized_lambdas)):
        optimized_lambdas = np.linspace(1.0, 0.0, target_n_states)
    optimized_lambdas = np.clip(optimized_lambdas, 0.0, 1.0)
    optimized_lambdas = np.sort(optimized_lambdas)[::-1]
    optimized_lambdas = np.minimum.accumulate(optimized_lambdas)
    optimized_lambdas[0], optimized_lambdas[-1] = 1.0, 0.0

    if min_spacing is None:
        min_spacing = max(0.02, 0.9 / max(target_n_states - 1, 1))
    min_spacing = float(min_spacing)

    for i in range(1, len(optimized_lambdas)):
        if optimized_lambdas[i] < 0.0:
            optimized_lambdas[i] = 0.0
        if optimized_lambdas[i - 1] - optimized_lambdas[i] < min_spacing:
            optimized_lambdas[i] = max(0.0, optimized_lambdas[i - 1] - min_spacing)

    unique_lambdas = []
    spacing_eps = 1e-9
    for lam in optimized_lambdas:
        lam_val = float(lam)
        if not unique_lambdas or (unique_lambdas[-1] - lam_val) >= (min_spacing - spacing_eps):
            unique_lambdas.append(lam_val)
    if unique_lambdas:
        unique_lambdas[0] = 1.0
        unique_lambdas[-1] = 0.0

    if len(unique_lambdas) < target_n_states:
        return np.linspace(1.0, 0.0, target_n_states), min_spacing, True
    return np.array(unique_lambdas), min_spacing, False


def redistribute_lambda_by_delta_f(
    lambdas_in_order: np.ndarray,
    f_k_in_order: np.ndarray,
    n_states: Optional[int] = None,
) -> np.ndarray:
    """
    按累积 |Δf|（真实自由能曲线的弧长）重新分布 λ 点，而不是等 λ 间距，也不是
    `_normalize_variance_weights` 那种基于短程试探采样、又被 log1p 压缩过的方差代理。

    输入的 f_k_in_order 必须是"已经实测、已修正单位"的自由能曲线（比如
    solve_stage_integrated 的输出），跟 lambdas_in_order 按同一顺序对齐。
    端点 λ 值保持不变，中间点按累积 |Δf| 等分，使每一步的自由能变化量大致相等。
    """
    lambdas_in_order = np.asarray(lambdas_in_order, dtype=float)
    f_k_in_order = np.asarray(f_k_in_order, dtype=float)
    if n_states is None:
        n_states = len(lambdas_in_order)
    if n_states < 2:
        raise ValueError("n_states 必须至少为 2（保留两个端点）")

    abs_steps = np.abs(np.diff(f_k_in_order))
    cum = np.concatenate([[0.0], np.cumsum(abs_steps)])
    total = float(cum[-1])
    if total <= 1e-8:
        # 曲线几乎平坦，退化为等 λ 间距。
        return np.linspace(lambdas_in_order[0], lambdas_in_order[-1], n_states)

    targets = np.linspace(0.0, total, n_states)
    new_lambdas = np.interp(targets, cum, lambdas_in_order)
    new_lambdas[0] = lambdas_in_order[0]
    new_lambdas[-1] = lambdas_in_order[-1]
    return new_lambdas


def _pilot_ti_cumulative_f(lam_sorted: np.ndarray, grad_sorted: np.ndarray) -> np.ndarray:
    """Trapezoidal thermodynamic integration of a pilot's measured mean
    gradient <dU/dlambda> into a raw F(lambda) curve, gauge-referenced to
    F(lam_sorted[0]) = 0. Inputs must already be sorted ascending in lambda.

    Shared by ``estimate_f_k_from_pilot_ti`` (bias-seed use case: mean-centers
    the physical free-energy curve into the IBS bias-parameter convention) and
    ``redistribute_vanishing_lambda_subdomains`` (lambda-spacing use case:
    only needs real |Delta F| magnitudes, no sign/gauge convention) -- kept
    as one function so both stay derived from the same integration, not two
    independently-maintained copies of the same trapezoidal rule.
    """
    seg = 0.5 * (grad_sorted[:-1] + grad_sorted[1:]) * np.diff(lam_sorted)
    return np.concatenate(([0.0], np.cumsum(seg)))


def estimate_f_k_from_pilot_ti(
    pilot_lambdas: Optional[List[float]],
    pilot_mean_dU_dlambda: Optional[List[float]],
    target_lambdas: List[float],
) -> Optional[np.ndarray]:
    """[IBS_BIAS_PROTOCOL_VERSION warm-start] Estimate a mean-centered f_k seed
    for ``target_lambdas`` via thermodynamic integration of the pilot's own
    measured mean gradient, instead of cold-starting online learning at
    f_k=0.0 for every state.  The returned array is already in the IBS
    bias-parameter convention.  For

    ``V_IBS = -kT log sum_k exp[-beta (U_k - f_k)]``,

    the integrated contribution of state ``k`` is proportional to
    ``exp(beta*f_k) Z_k = exp[beta*(f_k - F_k)]``.  Flat state weights therefore
    require ``f_k = F_k + constant``: the physical TI curve is mean-centered,
    not sign-inverted.  Occupancy feedback still has the complementary rule
    that an *observed overrepresented* state must have its ``f_k`` lowered.
    Confusing physical free energy with observed occupancy previously inverted
    this warm-start seed and made the online feedback spend most of its budget
    undoing the initialization.

    Five independent real GPU attempts at fixing vanishing window 0 by
    reshaping the lambda grid (6/4/3-state regrouping, adaptive pilot-grid
    refinement, real-Delta_f-equalized placement) all failed with occupation
    pinned at 96-99% on state 0 -- the online SGD/TMBAR loop must *discover*
    the needed bias purely from ~20-frame batches, and if the underlying
    transition is itself hard to sample, batches see almost no evidence from
    the underrepresented states, starving the learner of what it needs to
    grow the bias further (a genuine bootstrap problem, independent of how
    the lambda grid is spaced). ``_sample_scalar_metric`` already records
    ``mean_dU_dlambda_kJ_mol`` (the mean gradient, not just the variance proxy
    ``metric_g`` the rest of this file uses for spacing) at every pilot
    point -- integrating it via the trapezoidal rule gives a real F(lambda)
    estimate, available *before* any window is ever sampled, that can seed
    f_k with roughly the right scale from the first learning update instead
    of requiring the SGD loop to bootstrap that scale from scratch under
    exactly the sampling conditions that make bootstrapping hard.

    Deliberately conservative: returns ``None`` (meaning "no seed; caller
    falls back to today's implicit f_k=0.0") rather than raising or silently
    fabricating a value, in every case where the estimate would not be
    trustworthy:
      - ``pilot_lambdas``/``pilot_mean_dU_dlambda`` missing or empty (old,
        pre-this-feature preopt cache with no ``pilot_points`` data).
      - Fewer than 2 pilot points (trapezoidal integration needs at least 2).
      - Any non-finite value in either input array (a failed/corrupt pilot
        sample must not silently poison the seed).

    ``target_lambdas`` outside ``pilot_lambdas``'s actual measured range are
    NOT extrapolated (unreliable) -- they are clamped to the nearest boundary
    F(lambda) value via ``np.interp``'s ``left``/``right`` parameters, and a
    warning is printed so this isn't silently mistaken for a real estimate.
    """
    if not pilot_lambdas or not pilot_mean_dU_dlambda:
        return None
    try:
        pilot_lambdas = np.asarray(pilot_lambdas, dtype=float).ravel()
        grad = np.asarray(pilot_mean_dU_dlambda, dtype=float).ravel()
    except (TypeError, ValueError):
        # e.g. a None entry from an older/partial pilot_points record that's
        # missing mean_dU_dlambda_kJ_mol for some point -- can't safely cast,
        # not a real estimate either way.
        print("  [WARN] [pilot TI 热启动] pilot_lambdas/mean_dU_dlambda 无法转换为数值数组，放弃热启动，回退 f_k=0.0")
        return None
    if pilot_lambdas.size < 2 or grad.size != pilot_lambdas.size:
        return None
    if not np.all(np.isfinite(pilot_lambdas)) or not np.all(np.isfinite(grad)):
        print("  [WARN] [pilot TI 热启动] pilot_lambdas/mean_dU_dlambda 含非有限值，放弃热启动，回退 f_k=0.0")
        return None

    order = np.argsort(pilot_lambdas)
    lam_sorted = pilot_lambdas[order]
    grad_sorted = grad[order]
    # F(lambda): trapezoidal TI, referenced to lam_sorted[0] (arbitrary gauge
    # -- mean-centering below removes it anyway).
    f_at_pilot = _pilot_ti_cumulative_f(lam_sorted, grad_sorted)

    target = np.asarray(target_lambdas, dtype=float).ravel()
    if target.size == 0 or not np.all(np.isfinite(target)):
        return None
    lo, hi = float(lam_sorted[0]), float(lam_sorted[-1])
    if np.any(target < lo) or np.any(target > hi):
        print(
            f"  [WARN] [pilot TI 热启动] target_lambdas 超出 pilot 实测范围 "
            f"[{lo:.4f}, {hi:.4f}]，越界部分钳位到边界值，不做外推"
        )
    f_at_target = np.interp(target, lam_sorted, f_at_pilot, left=f_at_pilot[0], right=f_at_pilot[-1])
    # [IBS_BIAS_PROTOCOL_VERSION=27] Keep the physical TI sign.  Since the IBS
    # mixture uses exp[-beta*(U_k-f_k)], equal integrated state weights require
    # f_k=F_k+constant.  The former sign inversion produced the exact opposite
    # seed and forced the bounded occupancy feedback to undo it online.
    f_at_target = f_at_target - float(np.mean(f_at_target))
    return f_at_target


def pilot_ti_seed_trust_diagnostics(
    pilot_lambdas: Optional[List[float]],
    pilot_mean_dU_dlambda: Optional[List[float]],
    pilot_std_dU_dlambda: Optional[List[float]],
    pilot_n_dU_dlambda_samples: Optional[List[int]],
    target_lambdas: List[float],
    max_sem_kJ_mol: float = 2.0,
    max_propagated_uncertainty_kJ_mol: float = 5.0,
) -> Dict[str, Any]:
    """评估 `estimate_f_k_from_pilot_ti()` 给出的 pilot TI 种子，对某个具体
    窗口（`target_lambdas`）是否**精度**足够，可以被上游当作"跳过在线学习、
    直接尝试冻结验证"（pilot-first）的候选。

    ⚠️ 这只是精度判断（pilot 网格自己的 TI 积分测得多准），不是准确性判断
    （pilot 探针系统的物理环境——通常跟真实窗口环境不完全一样——测到的
    dU/dlambda 是否真的能代表这个窗口）。后一半必须由调用方另外用同一个
    窗口的独立自举 TI 估计（真实 Hamiltonian 下采样）做交叉验证；这个函数
    单独返回 `trustworthy=True` **不足以**允许 pilot-first，只是必要条件
    之一。见 memtodolist 里"窗口预热状态机重构"计划的风险复核结论。

    纯 Python，不依赖 OpenMM，可离线单元测试。永不抛异常——精度数据缺失、
    形状不对、含非有限值时一律 `trustworthy=False`，不当作调用方的 bug，
    也不当作"数据没问题只是精度不够"（旧的、本次改动之前生成的 preopt
    cache 就没有 `std_dU_dlambda_kJ_mol`/`n_derivative_samples` 这两个字段，
    必须能安全地退化成"不可信"而不是报错）。

    Returns
    -------
    dict，键固定为：
      - ``trustworthy``: bool，下面全部检查通过才是 True。
      - ``reason``: str，第一个未通过的检查名；`trustworthy=True` 时是 "ok"。
      - ``propagated_uncertainty_kJ_mol``: float，覆盖这个窗口 λ 跨度的
        pilot 点子集上，对 F(target_hi)-F(target_lo) 做的粗略 trapezoidal
        误差传播估计（`sqrt(sum((0.5*dlambda)^2 * (sem_i^2+sem_{i+1}^2)))`）。
        更早的检查失败时是 ``nan``。
      - ``max_sem_kJ_mol``: float，同一个局部子集里最差的标准误
        （`std_dU_dlambda_kJ_mol / sqrt(n_derivative_samples)`）。同样，
        更早失败时是 ``nan``。
    """
    nan = float("nan")

    def _fail(reason: str) -> Dict[str, Any]:
        return {
            "trustworthy": False,
            "reason": reason,
            "propagated_uncertainty_kJ_mol": nan,
            "max_sem_kJ_mol": nan,
        }

    if not pilot_lambdas or not pilot_mean_dU_dlambda:
        return _fail("missing_pilot_data")
    if not pilot_std_dU_dlambda or not pilot_n_dU_dlambda_samples:
        return _fail("missing_pilot_precision_fields")

    try:
        lam = np.asarray(pilot_lambdas, dtype=float).ravel()
        grad = np.asarray(pilot_mean_dU_dlambda, dtype=float).ravel()
        std = np.asarray(pilot_std_dU_dlambda, dtype=float).ravel()
        n_samples = np.asarray(pilot_n_dU_dlambda_samples, dtype=float).ravel()
    except (TypeError, ValueError):
        return _fail("non_numeric_pilot_data")

    if not (lam.size == grad.size == std.size == n_samples.size) or lam.size < 2:
        return _fail("shape_mismatch_or_too_few_points")
    if not (
        np.all(np.isfinite(lam))
        and np.all(np.isfinite(grad))
        and np.all(np.isfinite(std))
        and np.all(np.isfinite(n_samples))
    ):
        return _fail("non_finite_pilot_data")
    if np.any(n_samples < 1):
        return _fail("zero_sample_pilot_point")

    order = np.argsort(lam)
    lam_sorted = lam[order]
    std_sorted = std[order]
    n_sorted = n_samples[order]

    target = np.asarray(target_lambdas, dtype=float).ravel()
    if target.size == 0 or not np.all(np.isfinite(target)):
        return _fail("invalid_target_lambdas")

    lo, hi = float(lam_sorted[0]), float(lam_sorted[-1])
    target_lo, target_hi = float(np.min(target)), float(np.max(target))
    if target_lo < lo or target_hi > hi:
        # estimate_f_k_from_pilot_ti() 在这种情况下会钳位到边界值当近似——
        # 对"热启动初值"这种用途足够了；但对"直接当冻结候选"，钳位意味着
        # 这段窗口跨度里根本没有真实 pilot 测量，不能算可信。
        return _fail("target_lambdas_require_extrapolation")

    sem = std_sorted / np.sqrt(n_sorted)

    # 取覆盖这个窗口 λ 跨度的最小 pilot 点子集（跨度两端之外各留一个相邻
    # 点，保证跨度边界所在的那一段梯形也被计入），只在这个局部子集上做
    # 误差传播——关心的是这一个窗口自己的 F(target_hi)-F(target_lo) 有多
    # 不确定，不是整条 pilot 曲线的全局不确定度。
    lo_idx = max(0, int(np.searchsorted(lam_sorted, target_lo, side="right")) - 1)
    hi_idx = min(lam_sorted.size - 1, int(np.searchsorted(lam_sorted, target_hi, side="left")))
    if hi_idx <= lo_idx:
        hi_idx = min(lam_sorted.size - 1, lo_idx + 1)

    local_sem = sem[lo_idx : hi_idx + 1]
    d_lam = np.diff(lam_sorted[lo_idx : hi_idx + 1])
    if local_sem.size < 2:
        return _fail("insufficient_local_pilot_coverage")

    max_local_sem = float(np.max(local_sem))
    variance_terms = (0.5 * d_lam) ** 2 * (local_sem[:-1] ** 2 + local_sem[1:] ** 2)
    propagated_uncertainty = float(np.sqrt(np.sum(variance_terms)))

    if max_local_sem > float(max_sem_kJ_mol):
        return {
            "trustworthy": False,
            "reason": "pilot_sem_too_large",
            "propagated_uncertainty_kJ_mol": propagated_uncertainty,
            "max_sem_kJ_mol": max_local_sem,
        }
    if propagated_uncertainty > float(max_propagated_uncertainty_kJ_mol):
        return {
            "trustworthy": False,
            "reason": "propagated_uncertainty_too_large",
            "propagated_uncertainty_kJ_mol": propagated_uncertainty,
            "max_sem_kJ_mol": max_local_sem,
        }

    return {
        "trustworthy": True,
        "reason": "ok",
        "propagated_uncertainty_kJ_mol": propagated_uncertainty,
        "max_sem_kJ_mol": max_local_sem,
    }


def partition_windows_by_delta_f_budget(
    f_k_in_order: np.ndarray,
    max_window_span_kJ: float,
    overlap: int = 2,
) -> List[Tuple[int, int]]:
    """
    按累积 |Δf| 预算切分窗口边界，而不是按 state 数等分（`generate_overlapping_windows`
    那种纯按索引切分完全不知道每一段 λ 实际有多"陡"）。

    贪心地从每个窗口起点尽量往后扩，直到"再加一个点"就会让该窗口跨度超过
    max_window_span_kJ 才停止（提前判断下一步会不会超标，而不是超标之后才发现），
    确保每个窗口自身的能量跨度尽量贴着预算、不会系统性超支。如果单独一步的
    |Δf| 本身就已经超过预算（说明这里的 λ 点还不够密），该窗口会退化为只包含这
    一对相邻点，不会被强行拉宽掩盖问题。

    🔑 用的是逐步 |Δf| 的累积和（总变差/弧长），不是"终点减起点"的净位移——
    f_k(λ) 在软核/WCA shield 存在时不保证单调（尤其是精修探针步数较短、噪声
    较大时更容易出现局部反复），如果用净位移判断，一段先涨后跌又绕回起点附近
    的区间会被误判成"几乎没变化"，导致该窗口被贪心地拉得异常宽（曾实测出现单
    个窗口吞掉 7 个态、跟前后 3 态一组的窗口极不协调）。用累积和可以保证任何
    真实的往返波动都会被如实计入预算，不会被净位移抵消掩盖。
    """
    f_k_in_order = np.asarray(f_k_in_order, dtype=float)
    n = len(f_k_in_order)
    if n <= 2:
        return [(0, n)]

    overlap = max(1, int(overlap))
    # cum[i] = 从 f_k[0] 到 f_k[i] 逐步 |Δf| 的累积和（总变差），cum[j] - cum[i]
    # 即区间 [i, j] 内实际"走过"的能量距离，而不是端点净位移。
    cum = np.concatenate(([0.0], np.cumsum(np.abs(np.diff(f_k_in_order)))))
    windows = []
    start = 0
    while start < n - 1:
        end = start + 1
        while end + 1 < n and (cum[end + 1] - cum[start]) <= max_window_span_kJ:
            end += 1
        windows.append((start, end + 1))
        if end >= n - 1:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1  # 保证每轮都严格前进，避免死循环
        start = next_start

    # 清理没有带来新覆盖范围的冗余窗口：如果某个窗口因为一开始就撞到预算上限
    # 而提前收尾、右端点没有超过前一个窗口的右端点，它对拼接毫无帮助（完全被
    # 前一个窗口包含），直接丢弃，避免因为固定的 overlap 步长在预算吃紧的区域
    # 里反复产生"原地踏步"的窗口。
    merged = [windows[0]]
    for s, e in windows[1:]:
        if e <= merged[-1][1]:
            continue
        merged.append((s, e))
    return merged


def refine_stage_lambda_path_by_overlap(
    lambdas_var: List[float],
    window_ranges: List[Tuple[int, int]],
    window_overlap_diagnostics: List[Dict],
    min_overlap_threshold: float,
    pts_per_window: int = 6,
    overlap: int = 2,
) -> Tuple[Optional[List[float]], Optional[List[Tuple[int, int]]]]:
    """
    数据驱动地在重叠不足的地方加密 λ 点 —— 用的是这次采样*已经算出来*的
    per-window ESS (有效样本数) 重叠诊断，不是拍脑袋的固定间距或手写 λ 值。

    为什么不能复用 partition_windows_by_delta_f_budget/redistribute_lambda_by_delta_f：
    那条路径把"Δf 曲线陡不陡"当成重叠的代理指标，但 GlobalMBARAnalyzer.
    solve_stage_integrated 自己的审查报告已经指出这只是代理、不是真重叠——一个
    窗口可以 Δf 很平滑但仍然因为 IBS 偏置没收敛/构象弛豫慢等原因导致真实的
    reweight 有效样本比例（ess_ratio）很差。abfe_pipeline._assert_stage_result_sane
    用的正是后者（min_overlap/min_overlap_threshold），所以这里的加密逻辑也必须
    直接读同一个 ess_ratio 诊断，而不是去看 Δf 曲线。

    做法：对每个 min_ess_ratio < 阈值 的窗口，从它自带的 ess_ratio_per_lambda
    （每个目标 λ 态各自的有效样本比例，见 ibs_engine.py solve_stage_integrated）
    里找出全窗口最差的那个 λ 态，在它两侧（窗口内)较宽的那个物理 λ 间隔上插入
    一个新的中点 —— 更宽的间隔更可能是重叠瓶颈。多个窗口同时不达标时会分别
    处理、去重合并。窗口边界不手工指定，插入新点后统一交给
    generate_overlapping_windows 按现有约定（pts_per_window/overlap，跟这个流水线
    别处用的常量一致）重新切分，避免手工窗口边界产生索引错位。

    返回 (None, None) 表示诊断里找不到任何低于阈值、且带有 ess_ratio_per_lambda
    明细的窗口 —— 调用方应该把这当成"自动修复无法定位问题"，而不是继续盲目重试。
    """
    lambdas_var = list(lambdas_var)
    gaps_to_bisect = set()

    for rec in window_overlap_diagnostics or []:
        ratio = rec.get("min_ess_ratio")
        per_lambda = rec.get("ess_ratio_per_lambda")
        win_lams = rec.get("lambdas") or []
        if ratio is None or ratio >= min_overlap_threshold:
            continue
        if not per_lambda or len(win_lams) < 2:
            continue

        worst_lambda_idx = min(per_lambda, key=lambda k: per_lambda[k])
        worst_lambda_idx = int(worst_lambda_idx)
        if worst_lambda_idx not in win_lams:
            continue
        pos = win_lams.index(worst_lambda_idx)

        candidates = []
        if pos > 0:
            candidates.append((win_lams[pos - 1], win_lams[pos]))
        if pos < len(win_lams) - 1:
            candidates.append((win_lams[pos], win_lams[pos + 1]))
        if not candidates:
            continue

        # 两个相邻间隔里，物理 λ 跨度更宽的那个更可能是重叠瓶颈。
        lo, hi = max(
            candidates,
            key=lambda ab: abs(lambdas_var[ab[0]] - lambdas_var[ab[1]]),
        )
        gaps_to_bisect.add((min(lo, hi), max(lo, hi)))

    if not gaps_to_bisect:
        return None, None

    new_lambdas = list(lambdas_var)
    # 从高索引往低索引插入，这样前面插入不会打乱还没处理的间隔的索引。
    for lo, hi in sorted(gaps_to_bisect, key=lambda ab: -ab[0]):
        if hi != lo + 1:
            # 不是相邻的一对（例如上一轮已经在中间插过点导致索引偏移），跳过，
            # 交给下一轮基于新诊断重新定位，而不是插到错误的位置。
            continue
        midpoint = (lambdas_var[lo] + lambdas_var[hi]) / 2.0
        new_lambdas.insert(hi, midpoint)

    if len(new_lambdas) == len(lambdas_var):
        return None, None

    new_window_ranges = generate_overlapping_windows(
        n_states=len(new_lambdas), pts_per_window=pts_per_window, overlap=overlap
    )
    return new_lambdas, new_window_ranges


def refine_stage_lambda_path_from_data(
    stage_dir: str,
    preopt_path: str,
    temperature_k: float = 300.0,
    n_states: Optional[int] = None,
    max_window_span_kJ: float = 35.0,
    overlap: int = 2,
    stage_type: str = "vdw",
) -> Dict:
    """
    用该 stage 已经真实采集到的窗口能量数据，重新设计 λ 分布与窗口边界：
    - λ 点按累积 |Δf|（实测自由能曲线弧长）等分，不是等 λ 间距，也不是被压缩过的
      方差代理。
    - 窗口边界按累积 |Δf| 预算切分，不是按 state 数等分。
    所有数字都从这次真实采样数据现场算出来，不手写任何"魔法数字"。

    直接读取 preopt_path 里现有的 lambdas_var/window_ranges 去定位、加载已有窗口
    能量文件，用 GlobalMBARAnalyzer.solve_stage_integrated（已修复 β 换算）求出当前
    真实 f(λ) 曲线，再基于这条曲线重新设计。旧文件会先备份为 `<preopt_path>.bak`，
    新方案覆盖写回原路径。

    注意：这一步只能"重新规划下一轮该怎么采样"，不能凭空补全还没跑过的数据——
    重新规划后的窗口大多数会跟旧窗口边界不一致，下次 resume 时会被判定为形状不
    匹配、重新采样，这是预期行为，不是 bug。
    """
    from ibs_engine import solve_stage_integrated

    with open(preopt_path, "r", encoding="utf-8") as f:
        preopt = json.load(f)
    lambdas_var = preopt["lambdas_var"]
    window_ranges = preopt["window_ranges"]

    # 🔑 [P1-15] 从文件名解析**真实**窗口编号，按数值排序——此前
    # `sorted(glob.glob(...))` 是字典序，窗口数达到两位数时 window_10/window_11
    # 会排在 window_2 之前，再用 enumerate 的位置当窗口编号就会把 u_kn/bias/
    # base 与 window_ranges 错配，写出错误的新 λ 路径。与
    # runabfe._analyze_dual_leg / abfe_pipeline 清理窗口产物用的是同一套正则
    # `dual_window_(\d+)_{stage_type}_energies\.npy`；编号必须从 0 连续到 N-1，
    # 重复或缺失一律拒绝（不能悄悄错配）。
    _window_idx_re = re.compile(rf"dual_window_(\d+)_{stage_type}_energies\.npy$")
    indexed_e_files = []
    for e_file in glob.glob(os.path.join(stage_dir, f"dual_window_*_{stage_type}_energies.npy")):
        match = _window_idx_re.search(os.path.basename(e_file))
        if not match:
            raise RuntimeError(
                f"无法从文件名解析窗口编号（期望 dual_window_<int>_{stage_type}_energies.npy）: "
                f"{e_file}"
            )
        indexed_e_files.append((int(match.group(1)), e_file))
    indexed_e_files.sort(key=lambda pair: pair[0])
    parsed_indices = [idx for idx, _ in indexed_e_files]
    if parsed_indices != list(range(len(window_ranges))):
        raise RuntimeError(
            f"窗口能量文件编号（解析得到 {parsed_indices}）与 preopt 缓存里的 "
            f"window_ranges 数 ({len(window_ranges)}) 不一致（要求从 0 连续编号），"
            "无法基于现有数据重新设计路径；"
            "请先确认该 stage 的采样已经完整跑完（每个窗口都有对应的 "
            f"dual_window_<int>_{stage_type}_energies.npy，且没有重复/缺失编号）。"
        )

    window_data = []
    for w_idx, (_parsed_idx, e_file) in enumerate(indexed_e_files):
        u_kn = np.load(e_file)
        bias = np.load(e_file.replace("_energies.npy", "_bias.npy"))
        base = np.load(e_file.replace("_energies.npy", "_base.npy"))
        start, end = window_ranges[w_idx]
        window_data.append({
            "u_kn": u_kn,
            "bias_energies": bias,
            "base_energies": base,
            "lambda_indices": list(range(start, end)),
        })

    kt = 0.008314462618 * float(temperature_k)
    res = solve_stage_integrated(window_data, kt, stage_name=stage_type)
    if res.get("error"):
        raise RuntimeError(f"基于现有数据求解当前 f(λ) 曲线失败: {res['error']}")

    lambdas_sorted = res["lambdas"]
    f_k = np.asarray(res["f_k"], dtype=float)
    lam_in_order = np.asarray([lambdas_var[i] for i in lambdas_sorted], dtype=float)

    n_new = int(n_states or len(lambdas_var))
    new_lambdas = redistribute_lambda_by_delta_f(lam_in_order, f_k, n_new)

    # 用旧曲线插值出新 λ 点对应的 f 值，仅用于指导窗口切分。
    interp_order = np.argsort(lam_in_order)
    f_at_new = np.interp(new_lambdas, lam_in_order[interp_order], f_k[interp_order])
    new_windows = partition_windows_by_delta_f_budget(f_at_new, max_window_span_kJ, overlap=overlap)

    covered = sorted({i for s, e in new_windows for i in range(s, e)})
    if covered != list(range(n_new)):
        raise RuntimeError(
            f"内部错误：新窗口划分未能完整覆盖 [0,{n_new})，拒绝写出（覆盖到 {covered}）。"
        )

    new_preopt = {
        "lambdas_var": [float(x) for x in new_lambdas],
        "window_ranges": [[int(s), int(e)] for s, e in new_windows],
        "n_states": n_new,
        "provenance": {
            "source": "refine_stage_lambda_path_from_data",
            "based_on_measured_f_curve": True,
            "max_window_span_kJ_mol": float(max_window_span_kJ),
            "prior_n_states": len(lambdas_var),
            "prior_window_ranges": [list(w) for w in window_ranges],
            "prior_total_delta_G_kJ_mol": float(res.get("total_delta_G", float("nan"))),
            "prior_min_overlap": res.get("min_overlap"),
        },
    }

    backup_path = preopt_path + ".bak"
    shutil.copy(preopt_path, backup_path)
    with open(preopt_path, "w", encoding="utf-8") as f:
        json.dump(new_preopt, f, indent=2)

    return new_preopt


def _sample_group1_energies(context, total_steps, sample_interval=50):
    """批量推进积分器，保留固定采样间隔，减少 Python/C++ 边界往返。"""
    if total_steps <= 0:
        return []

    integrator = context.getIntegrator()
    energies = []
    full_batches, remainder = divmod(int(total_steps), int(sample_interval))

    for _ in range(full_batches):
        guarded_step(integrator, sample_interval, "Group1 能量采样")
        state = context.getState(getEnergy=True, groups={1})
        energies.append(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )

    if remainder:
        guarded_step(integrator, remainder, "Group1 能量采样（余数段）")
        state = context.getState(getEnergy=True, groups={1})
        energies.append(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )

    return energies

