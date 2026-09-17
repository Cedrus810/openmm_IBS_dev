# Stage-2 控制器：八条死线与各自的修复逻辑

> ## 🗂 2026-09-17 归档 —— **不是待办**
>
> 本文是一次性的静态抽取（20 条终态出口 → 8 条死线），主题当天收口，原文一字未改。
>
> | 本文的哪一段 | 结论去了哪 |
> |---|---|
> | §2 一次开四条：接 `IMMUTABLE_REWINDOW` | ✅ 当天落地 = `REWIND-01`，整段在 [TODO_closed_2026-09-17.md](TODO_closed_2026-09-17.md) |
> | §3.1 加帧刹车永不触发 | ✅ 当天修（**没走本节建议的拟斜率**，走的是同一个前向判据问剩余配额），见 [CHANGELOG](../CHANGELOG.md) 09-17 |
> | §3.2 `plan()` 把换 f_k 静默改写成补帧 | ⚖️ **改写本身仍在**（O1 已收紧成只认 `SAMPLE_SIZE`）；它与 §3.1 的**共同根因**——射程判据高估——同日补上第三道准入 |
> | §1 D1 / D8 | ✅ 报告侧改进当天落地（`terminal_window_failures` 恒常输出、`_attribution_blind_spots()` 区分 corrupt/absent），见 [CHANGELOG](../CHANGELOG.md) 09-17 |
> | §1 D3 子窗失败 | ⬜ **第一次真正可达**（rewindow 接通之后）。默认**不递归**；要不要放宽按 §1 D3 再定 |
>
> ⚠️ 本文开头那条警告**仍然成立**，别因为归档就忘了：判据本身噪声 34×
> （[AUDIT_GATES_AND_CRITERIA_2026-09-17.md](AUDIT_GATES_AND_CRITERIA_2026-09-17.md)）⟹
> 死线打开只意味着循环能继续走，**不意味着答案更准**。


> **同批次的另外两份**：控制流全景 → `STAGE2_CONTROLLER_FLOW_2026-09-17.md`；
> 判据/门的数据审查 → `AUDIT_GATES_AND_CRITERIA_2026-09-17.md`。
> ⚠️ 本文的死线分类基于当前判据；而那份审查的结论是**主验收量在当前预算下
> 不可分类**（同配置同 seed 三次 50.33/10.74/1.46）—— 所以「这个窗口有没有问题」
> 本身就不可靠，八条死线里以「窗口失败」为前提的那几条要连带重估。

> ✅ **2026-09-17 晚：§2 的修复已落地（`REWIND-01` 关闭）。** D2/D5/D6/D7 四条现在
> 各自在兜底**之前**先问一次有界重窗；`rewindow_feasible()` = K≥3 且父窗不在
> `parents_done`（深度上限 **1 层**，与本文 §2 的建议一致）。子窗划分搬到
> `abfe_preoptimizer.vanishing_rescue_ranges()`，pipeline 侧转发，成本表不再复算。
> 三条 strict-xfail 已删、静态审计死动作 1→0、全量 2653 passed。
> **D3 现在第一次真正可达**（子窗失败），届时按 §1 D3 再定要不要允许递归 —— 默认不递归。
> ⚠️ 本文开头那条警告仍然成立：判据本身噪声 34×，rewindow 会被误触发在没问题的窗口上。

2026-09-17。从 `Stage2RepairController._decide_once` 的 AST 抽出全部 20 条终态出口，
其中 8 条是 `NO_FEASIBLE_ACTION`（真死局：盘面有问题、控制器发不出任何动作）。

**不写行号**（本仓库的行号会过期，见 `STAGE2_CONTROLLER_DESIGN_2026-09-12.md` §9），
每条按**守卫谓词**定位。

---

## 0. 总图：四条汇到同一个闸

```
   D2 预热预算耗尽 ──┐
   D5 累计f_k+heldout─┤   feas["insert_lambda"]        ┌─ a) 取不到 tail anchor
   D6 边际增益停滞 ──┤   feas["split_tail_window"]  ←──┤  b) dry-run 切不出更细
   D7 自检偏斜类 ────┘        ↓                        │  c) 插完末窗越过 hi 且无 anchor
                        NO_FEASIBLE_ACTION             └─ d) max_path_insertions 用尽
```

a 与 c 的 anchor 来自 `tail_repartition_anchor` → `first_untrusted_window`，
而它在 `idx is None or idx <= 0` 时**恒返回 None**。⟹ **失败窗口是 window 0 时，
a 与 c 恒成立。**

---

## 1. 八条死线

### D1 `phase == "TERMINAL"` 且非统计驳回
- **守卫**：`earliest` 的 `_tw.get("phase") == "TERMINAL"` 且 `verdict != "STATISTICALLY_REJECTED"`
- **含义**：这份 f_k 的预热/验证已被判死，不是「尚不可测」。
- **为什么无动作**：重解 / 换 Epoch / 加帧都要先重新进入这个窗口，而它的预热在入口就会被弹回。
- **修复逻辑**：**不是控制器能修的。** 需要人工决定三选一 ——
  改输入 λ 表 / 显式升档该窗口预热预算 / 接受该窗口失败。
- **状态**：合理的死线，保留。要改的只是让它**报得更清楚**（把 `last_failure_reason`
  与 `last_gate_error` 提到 run 级摘要，不要只埋在 reason 字符串里）。

### D2 预热预算耗尽 + f_k 未验证 + 插 λ 不可行
- **守卫**：`warmup_steps_left <= 0` 且 `f_k_evidence_status != "verified"` 且 `feas["insert_lambda"] is not None`
- **含义**：补帧会在预热门上被弹回；唯一不需重进预热的动作是缩跨度，而它不可行。
- **修复逻辑**：接 `IMMUTABLE_REWINDOW` —— 它**不动 λ 表**、**不需要 tail anchor**、
  在窗内建重叠子系综，四个封锁条件一个都不碰。子窗是**新系综**，有自己的预热预算，
  所以也绕开了「重进父窗预热被弹回」。
- **状态**：**可修**，见 §2。

### D3 子窗支撑/偏斜类失败
- **守卫**：`sampling_units` 里某个 `support_failed and not needs_frames`
- **含义**：rewindow 的子窗在固定 λ 表上已是有界重窗，没有更细的缩跨度动作。
- **修复逻辑**：递归再 rewindow 一层（子窗 K≥3 时仍可切），或如实停下。
- **状态**：⚠️ **2026-09-17 起第一次真正可达** —— `IMMUTABLE_REWINDOW` 已接通
  （abfe-ibs-db），子窗现在真的会出现。当前实现**一个父窗只切一层、子窗不再递归**。
  要不要允许递归**是用户决定，尚未拍板**；在拍板前不得放宽。
  ⚠️ 允许无限递归 = 新的无限循环，必须有深度上限。

### D4 冻结验证算术不可达 + 替代 Epoch 已用过
- **守卫**：可达性预检判 `UNREACHABLE` 且 `relearn_epoch_used(...)` 为真
- **含义**：一个窗口只给一次替代候选，否则就是反复试到偶然通过。
- **修复逻辑**：**不是控制器能修的**，是预算问题。人工升档预热/验证预算。
- **状态**：合理的死线，保留。

### D5 累计 f_k 偏差 FAIL + held-out REJECT + 两个缩跨度都不可行
- **守卫**：`cum_fk_verdict in (FAIL_CUMULATIVE_FK, UNMEASURED)` 且 `heldout_verdict == "REJECT"`
  且 `not _feas_split and not _feas_ins`
- **含义**：已经证明「f_k 救不了它，对症动作是缩跨度」，而缩跨度不可行。
- **额外闸**：`_feas_split` 还要求 `earliest == 末窗`（`_is_tail`）——
  非末窗的失败窗口连 b 都轮不到，直接判不可拆。
- **修复逻辑**：接 `IMMUTABLE_REWINDOW`。这是**归因最明确**的一条死线 ——
  诊断已经指向缩跨度，只是没有一个对任意窗口都可行的缩跨度动作。
- **状态**：**可修**，见 §2。

### D6 `min N_eff/g` 不随采样上升 + 插 λ 不可行
- **守卫**：`marginal_gain_stalled(history)` 为真 且 `feas["insert_lambda"] is not None`
- **含义**：同分布加帧已被本窗口自己的数据证伪，而对症的缩跨度不可行。
- **修复逻辑**：接 `IMMUTABLE_REWINDOW`。
- **⚠️ 附带缺陷（独立于死线）**：`marginal_gain_stalled` 的判据是
  `last < median(前面各点) × 0.9` —— **单调上升但渐近在门以下的序列永不触发**。
  真机 cyclod_ligand1_outer/rep1 窗口 0：`4.459 / 4.553 / 5.287`，
  比值 1.173 > 0.9 ⟹ 判「还在涨」，刹车不响，而门是 10、按 +0.41/块 要 11 块才够。
  两处独立核实（abfe-ibs-26、abfe-ibs-61）。**这条不是本文档的死线，是刹车失灵。**
- **状态**：死线**可修**；刹车缺陷另记，见 §3。

### D7 自检偏斜类失败 + 两个缩跨度都不可行 + 累计 f_k 证据已存在
- **守卫**：`support_failure_is_skew(...)` 为真 且 两个缩跨度不可行 且 `cum_fk_verdict is not None`
- **含义**：与 D5 同形，只是归因来自逐窗自检而非累计残差。
- **修复逻辑**：接 `IMMUTABLE_REWINDOW`。
- **状态**：**可修**，见 §2。

### D8 分析不完整 + 归因不出任何一道门
- **守卫**：链条末尾兜底（`analysis_status == ANALYSIS_INCOMPLETE` 且上游一条都没命中）
- **含义**：没有科学上站得住的下一步动作。编一个出来会退化成「再跑一次直到碰巧通过」。
- **修复逻辑**：**不该修。** 这是设计上正确的兜底。
- **⚠️ 但要查**：真机出现过「`converged=False` + 零条归因」而真相是某道门的读数是 NaN
  （审计 #59 已修 `stage_quality_gate_failures` 的 fail-open）。落到 D8 时要能区分
  「确实没有对症动作」与「归因函数读不出数」。
- **状态**：保留。

---

## 2. 一次开四条：接 `IMMUTABLE_REWINDOW`

**D2 / D5 / D6 / D7 四条共用同一个闸**，而 `IMMUTABLE_REWINDOW` 绕开全部四个封锁条件：

| 封锁条件 | rewindow 为什么不受影响 |
|---|---|
| a) 取不到 tail anchor | 它按 `window_idx` 工作，不需要 anchor |
| b) dry-run 切不出更细 | 它在**窗内**按中点切两个重叠子系综，不重分尾段 |
| c) 插完末窗越过 hi | 它**不动 λ 表**，末窗一个态都不涨 |
| d) 插点预算用尽 | 同上，不消耗 `max_path_insertions` |

**现成的部分**（都已在仓库里、已被测试覆盖）：
- 执行器 `abfe_pipeline._immutable_rewindow_step`：覆盖完整性 / 接缝共享节点 / 身份锁定
  三道 fail-closed，独立目录 `<stage>_rewindow_<identity>/`，原数据一字节不动；
- `plan()` 的成本表（`_PRODUCTION_CHARGED` 里按 `n_children × new_ensemble_reserve_steps` 计价）；
- no-op 台账（`_NOOP_GUARDED` 已含它）；
- 子窗调度（`sampling_units` / `needs_frames` / `blocks_by_unit` 全套）。

**缺的只有一句** `plan("IMMUTABLE_REWINDOW", ...)`，外加一条可行性谓词。

**可行性谓词**：`_build_vanishing_rescue_ranges` 按中点切两个重叠子窗，
要求 **K ≥ 3**（切出 left/right 各 ≥2 态且共享中点）。

> 🔑 **与 2026-09-17 分窗改动的正向耦合**：`stage2_window_partition` 从 `metric_integral`
> 改成 `state_count`（min-max K）之后，23 个蛋白 rep 的布局全部变成 `[5,5,5,5,5]`
> （原来是 `[7,6,4,4,4]` / `[8,5,4,4,4]`）。**每个窗口 K=5 ⟹ 全部满足 K≥3 ⟹
> rewindow 对每一个窗口都可行。** 老布局里 K=4 的中间窗切成 3+2 也可行，但 K=5
> 切成 3+3 更均衡。

**必须同时定的两件事**：
1. **递归深度上限**：子窗还失败时允许不允许再切一层？不设上限 = 新的无限循环。
   建议 **1 层**（与 `max_path_insertions` 同性质的终身预算）。
2. **接在链条哪个位置**：在四条死线**各自的 `if not feasible` 之前**插一个
   「rewindow 可行吗」的分支，而不是在链尾加第 33 条 —— 否则又是「按书写顺序决定语义」。

**接通后**：`tests/test_stage2_controller_vocabulary.py::test_no_dead_actions_left_in_the_declaration`
与 `tests/test_stage2_repair_controller.py::test_fixed_lambda_table_middle_window_falls_back_to_bounded_rewindow`
两条 strict-xfail 会变红，按仓库规矩同时删掉 xfail 标记。TODO 条目是 `REWIND-01`（P1）。

---

## 3. 与死线相邻、但不是死线的两个缺陷

这两条不会让循环停机，会让它**白跑**，一并记下免得下次又重查。

### 3.1 加帧刹车永不触发  —— ✅ **2026-09-17 已修（但不是按本节建议的方式）**

> 本节建议「拿已有各块拟 ratio 对块数的斜率外推」，同时自己警告「3 个点拟斜率
> 同样在量噪声」。**两条都对，所以没走那条路。**
> 实际修法：`_frames_admission` 补第三道 —— **同一个前向判据**
> （`n_eff_over_g_reachable_by_frames`），拿**现在的读数**和**剩下的配额**再问一次。
> 不引入新阈值、不拟合斜率。真机三个数代入：used=1 放行、used=2 停。
> 挂在**全部 16 个 `RUN_PRODUCTION` 出口的唯一咽喉**上，不是挂在归因上。

### 3.1 加帧刹车永不触发
`marginal_gain_stalled`：`last < median(前面各点) × 0.9`。
单调上升但渐近在门以下的序列恒判「还在涨」。真机窗口 0 实测 `4.459/4.553/5.287`
（比值 1.173）⟹ 刹车不响，连补 3 块约 1.25M 步才走到真正管用的动作。
**对症判据应是射程外推而不是是否下降**：拿已有各块拟 ratio 对块数的斜率，
外推到剩余配额用完够不到门就停。
⚠️ 但这个量的单次噪声很大（同 λ 同 seed 同步数，实测 50.33 vs 18.23 = 2.8×，
跨节点 4.7×），**3 个点拟斜率同样在量噪声**。要先解决「测得动吗」。

### 3.2 `plan()` 会把换 f_k 静默改写成补帧
`plan()` 的第一层否决：Epoch 类动作付不起新 Epoch 的验证额度时，
若该窗口失败归因是「样本量类」⟹ 改发 `RUN_PRODUCTION`。
于是分支判「f_k 不对，换 Epoch」，打包时变成「补帧」——
而归因靠的是 `n_eff_over_g_reachable_by_frames`，它假设 `g` 恒定，
真机实测 g 在纯加帧下从 12.25 涨到 17.63（+44%），射程高估约 2.5 倍。

---

## 4. 一句话

八条死线里，**四条是同一个闸**（缩跨度动作对非末窗/window 0 不可行），
**三条是合理的**（TERMINAL、验证预算、兜底），**一条当前不可达**（子窗，等 rewindow 接通）。
唯一的结构性修复是把已经造好的 `IMMUTABLE_REWINDOW` 接回 `decide()`。
