# TODO 取证存档：Stage-2 控制器 / 预算 / benchmark（截至 2026-09-16）

[文档导航](../README.md) · [在推的待办](../TODO.md) · [已关闭条目](TODO_closed_2026-09-16.md)

> **2026-09-16 归档，不是待办。** 这些整段来自 `docs/TODO.md` 09-16 之前的正文：
> 已关闭条目留下的结论、以及**本仓唯一一次成规模的离线取证**
> （6 个真 run 的盘面 + 当前代码 `decide()` 重放、预算实测表、28 个报错 run 的归因）。
>
> 留下来是因为那几张表还有引用价值；**表里的数字都带日期，都不是当前值** ——
> 判当前行为要重新重放，别照抄。逐条关闭证据在
> [TODO_closed_2026-09-16.md](TODO_closed_2026-09-16.md)。

---

## 1. Stage-2 已关闭条目留下的结论（`S2-*` / `DECORR-01` / `AUDIT-S2-01`）

### 剩余缺口（源：控制器 design §7）

> `S2-D`（代码收拢）**2026-09-12 已关闭**，整段进
> [archive/TODO_closed_2026-09-12.md](TODO_closed_2026-09-12.md)。
> 一句话结论：不是"都塞进一个文件"，是**决策同源的进 `abfe_preoptimizer`、
> 写盘的留 `abfe_pipeline`**。逐符号现状见
> [设计文档 §9.4](../STAGE2_CONTROLLER_DESIGN_2026-09-12.md)。

> `S2-B`（完整性要求随预算膨胀）**2026-09-13 已修**，整段进
> [archive/TODO_closed_2026-09-13.md](TODO_closed_2026-09-13.md)。
> 一句话结论：**原条目的后果描述是错的** —— 「可达性判据第 2 档因此失效」从没
> 发生过，那个量在本仓自第一个 commit 起就只进报告、不当门。真实危害是它会骗
> 读它的人（可达性的 T 一度就被错取成它，gcrit 算小 20 倍）。修法仍按原定正解：
> 去掉预算那一支。

> `S2-F`（10 条旧断言对齐新语义）**2026-09-13 已关闭**，整段进
> [archive/TODO_closed_2026-09-13.md](TODO_closed_2026-09-13.md)。
> 一句话结论：**10 条里只有 3 条真的是断言旧了，另外 7 条是 fixture 旧了** ——
> 造的 run 目录缺自检产物/缺预算台账，于是 fail-closed 的前置门把请求兜住，
> 测试想钉的分支一条都走不到。**照着实际输出改断言会让 7 条退化成同一个测试。**

> `DECORR-01`（去相关帧数两份实现）**2026-09-14 已裁决并接通**：**求解器对「能否
> 进入求解、是否跳窗」有操作权威**，自检的 `n_decorr` 保留为早期 target-support
> 诊断（`self_sufficient` 里还混着 `N_eff/g` 与 top1%，不得读成「求解器帧数够」）。
> **两侧数字不强行看齐** —— 分别命名、分别展示（`evidence_decorrelation`），
> 要诊断差异请用**完全相同的输入**另跑一次对照。

> `S2-A`（累计 f_k 残差门对多段窗口结构性不适用）**2026-09-14 已关闭**：
> `_per_segment_cumulative_fk_residual()` 按 `sampling_source_id` 的有序帧映射
> 把每段绑到**它自己的**冻结 f_k、在自己的帧切片上独立解一次，**没有**跨段
> 望远镜相消；窗口级 = 任一有效段 FAIL 则 FAIL，否则有缺证据段则 `UNMEASURED`，
> 全过才 PASS。逐段身份/帧范围/f_k 指纹/结论可审计。

> `DECORR-01`（去相关帧数两份实现）**2026-09-14 接线完成**：两侧各记一份
> `decorrelation_provenance`（输入身份 + 实际抽样索引），`compare_…()` 先比输入
> 构造、`replay_decorrelation()` 在完全相同的输入上重放。
> ⚠️ **分歧来源仍未证明** —— 这套装置是用来证明它的，不是用来消掉它的；
> 在证明之前**不许改任一侧的数去追平另一侧**。
  多段窗口的帧采自两份不同偏置，`_load_ibs_window_outputs_merged` 因此显式
  `base.pop("f_k")`。⟹ 循环每用换 Epoch 修好一个窗口，那个窗口就**永久失去**
  这项证据（那一跑 win0–3 已全丢）。逐段残差不与合并后的 ΔF 直接望远镜相消，
  **需要单独定口径**。
> `S2-C`（边际增长判据要两段历史）**2026-09-14 已修**：`support_history` 改为叠加
> 主循环已经在写的逐轮 `snapshot`（逐块），可比性由 `path_version` + `segment`
> （换段 = 换 f_k）双身份保证、按 `production_steps` 排序去重。一句话结论：
> **按段建史给不出第二个点，而单段正是最常见的情形** —— 那道唯一的"加帧已被
> 证伪就别再加"的刹车对单段窗口从不触发。

> `DECORR-01` 的 `- [ ]` **2026-09-16 摘除**：同一条在它上面已经有两段「已裁决 /
> 已接线」的结论，条目本身是规则 2 说的那种"`[x]` 尸体"。整段进
> [archive/TODO_closed_2026-09-16.md](TODO_closed_2026-09-16.md)。**裁决与禁令一字未变**（权威是求解器；两侧数字不许互相看齐）。

> `AUDIT-S2-01`（`decide()` 从不发 `SPLIT_TAIL_WINDOW`）**2026-09-16 前提证伪**：
> `abfe_preoptimizer.py:5814`（`_feas_split`）与 `5955`（`_can_split_skew`）两处都发得出，
> 且本文 09-14 自己的重放表里 `cyclod_ligand2/rep1` 拿到的就是 `SPLIT_TAIL_WINDOW[5]`。
> 整段进 [archive/TODO_closed_2026-09-16.md](TODO_closed_2026-09-16.md)。
> ⚠️ 真正该盯的不是「发不发得出」而是**发出来做的是不是同一件事** —— 见审计 #24
> （可行性问的是「末窗能否一分为二」，执行器做的是「从 `first_untrusted_window` 起全部重分」）。

> **当天已修（梳理抓到的两条）**：
> · 补帧准入门只挂住 1/16 个出口 ⟹ 改为在 `plan()` 里挂**一次**，并加 AST 守卫
>   钉住"只许一处、且必须在 `plan()` 内"；
> · `action=DONE` 配 `exit=NO_FEASIBLE_ACTION`（`plan()` 会算成 `execution_status=COMPLETE`，
>   把"无路可走"记成"执行完毕"）⟹ 改 `NO_ACTION`，并加 AST 守卫扫全部出口的动作/出口自洽。

---

## 2. 预算系统第三轮复核的取证与两张实测表（2026-09-14）

### 2026-09-14 第三轮复核：预算系统（`BUD-01`~`07`）+ 控制器（`CTL-11`~`15`）
#### —— 2026-09-16 对账后**只剩 `BUD-06` 一条，且已改判为待拍板**

> 这一节写下来的时候是**只读查出、一条未修**的。**今天不是了** —— 见下面的对账块。
> 保留整节是因为**取证方式和那两张实测表仍然是本文最有用的东西**：它们是
> 「6 个真 run 的盘面 + 当前代码重放」这唯一一次成规模的离线取证。
> ⚠️ **两张表的数字停在 2026-09-14**，`CTL-11` / `CTL-12` / `CTL-15` 那几列的备注
> 描述的是**当时**的代码 —— 修完之后的重放见本节末尾 09-14 收尾表与 09-16 的 BM 重放表。
>
> **取证方式**：用**当前工作树的代码**对 6 个 benchmark run 的盘面调
> `Stage2RepairController.for_physical_stage(...).read()` / `.decide()`（控制器只读），
> 另读它们的 `stage2_autonomous_history.json` / `path_versions/v*.json` / `pipeline.log`。
> ⚠️ 盘上那 6 个 run 是**旧代码**跑出来的，它们的 history 不能直接当现在代码的证据；
> 但 `decide()` 的重放是当前代码，可以。下面分开标注。
>
> ⚠️ **下表备注里的 `CTL-11` / `CTL-12` / `CTL-15` 现在都在
> [archive/TODO_closed_2026-09-16.md](TODO_closed_2026-09-16.md) 里（2026-09-16 已关闭）。**
> 表本身是 **09-14 那天**的重放快照，**别当成当前行为** —— 修完之后的重放见
> 本节末尾 09-14 收尾表，以及 `BM-B`。
>
> **当前代码在 6 个 run 上的重放**：
>
> | run | `decide()` 现在给什么 | stage `converged` | 备注 |
> |---|---|---|---|
> | `brd4_ligand2/rep1` | `RUN_PRODUCTION[4]` | None | 末窗 K=10，布局已非法（`CTL-15`） |
> | `cyclod_ligand1/rep2` | `RECALIBRATE_FK[3]` | None | win3 自检 0.386 |
> | `cyclod_ligand1/rep3` | `RUN_PRODUCTION[4]` | None | win4 无任何产物 |
> | `cyclod_ligand2/rep1` | `SPLIT_TAIL_WINDOW[5]` | False | 旧代码在这个动作上连发 4 次死掉（`CTL-12`） |
> | `cyclod_ligand2/rep2` | `PROBE_REANCHOR_EPOCH[0]` | **True** | ⚠️ `CTL-11` |
> | `cyclod_ligand2/rep3` | `DONE` | True | 6 个里唯一正确终止的 |
>
> **同一次重放的生产预算账**（`cap` 全部为 `None`，块账全部为 `{}`）：
>
> | run | 累计已用生产步 | 单窗最高 |
> |---|---|---|
> | brd4_ligand2/rep1 | 1,450,000 | win4 450,000 |
> | cyclod_ligand1/rep2 | 1,250,000 | win3 500,000 |
> | cyclod_ligand1/rep3 | 2,500,000 | **win4 1,500,000** |
> | cyclod_ligand2/rep1 | 4,000,000 | **win4 1,500,000** |
> | cyclod_ligand2/rep2 | 2,500,000 | win4 1,250,000 |
> | cyclod_ligand2/rep3 | 2,500,000 | win1 750,000 |

> ### ✅ 2026-09-16 对账：`BUD-01`~`05`、`BUD-07`、`CTL-11`~`14` **全部已修**
>
> 这七条 + 四条当时是**只读查出、一条未修**地写进来的，后来在 09-14/09-15 的修复批次里
> 陆续修掉了，**但没人回来打勾** —— 于是它们以 `- [ ]` 的样子在这里挂了两天，
> 让整个预算系统看起来还没接上。**2026-09-16 逐条回源码核实，缺陷都已不在**，
> 整段（含逐条证据行号）进 [archive/TODO_closed_2026-09-16.md](TODO_closed_2026-09-16.md)。
>
> 同一天连带纠正了 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](../CONTROLLER_BUDGET_AUDIT_2026-09-14.md)
> 的状态列：那份文档 65 条里有 20 条还标着 `OPEN`，实测 **65 条全部落地**。
>
> **留下的规矩**：「某条缺陷修没修」这个事实**只有一份权威，是源码**。
> 状态栏是缓存，缓存会过期。⚠️ 但它只说明缺陷不在了，**不说明修得对** ——
> 这批改动仍然是真机零验证（`AUDIT-S2-03`）。
>
> ⚠️ **`BUD-06` 没有被这一轮关掉**，它在下面，且已从「bug」改判为「待拍板」。



> `CTL-15` ~ `CTL-19` **2026-09-14 当天已关闭**（`CTL-15` 是**原判断错误**被作废，
> 其余四条已修），整段进 [archive/TODO_closed_2026-09-16.md](TODO_closed_2026-09-16.md)。
> 留下一条规矩：**`lo/hi` 是逐 run 从 `run_provenance.json` 读的**，
> 跨 run 引用可拆区间前必须先看那个 run 自己的 `lo/hi`，别拿手边那个体系的数去套 ——
> `CTL-15` 的错判正是拿 cyclod 的 `4/5` 去套 brd4 的 `8`。

> ### ✅ 2026-09-14 收尾：6 个 benchmark run 的 resume 可用性（当前代码只读重放）
>
> | run | 末窗 K | 启动布局 | `decide()` |
> |---|---|---|---|
> | brd4_ligand2/rep1 | 10 | 先拆 (16,26)→(16,21)+(20,26) | `RUN_PRODUCTION[4]` |
> | cyclod_ligand1/rep2 | 12 | 先拆 (15,27)→(15,21)+(20,27) | `RECALIBRATE_FK[3]` |
> | cyclod_ligand1/rep3 | 11 | 先拆 (16,27)→(16,22)+(21,27) | `RUN_PRODUCTION[4]` |
> | cyclod_ligand2/rep1 | 7 | 先拆 (17,24)→(17,21)+(20,24) | `SPLIT_TAIL_WINDOW[5]` |
> | cyclod_ligand2/rep2 | 4 | 合法 | **`DONE`** |
> | cyclod_ligand2/rep3 | 5 | 合法 | **`DONE`** |
>
> 修之前：rep2 给的是 `PROBE_REANCHOR_EPOCH`（跑完的 stage 每次 resume 重开 Epoch）、
> rep3 在启动布局校验处直接抛错、rep1/ligand1-rep2 在 `SPLIT_TAIL_WINDOW` 上空转到
> `NO_FEASIBLE_ACTION`。
>
> ⚠️ **块账不会追溯**：既有 history 的 `path_version` 全是 `null`（旧代码写的），
> 所以 `BUD-03/04` 的 4 块配额对这几个 run 是**从下次 resume 重新开始数**的，
> 已经烧掉的 1.5M 步不计入。要让它们立刻受限，只能手工改 config 降
> `stage2_max_production_blocks_per_window`。
>
> 离线全套：**2455 passed / 3 skipped / 0 failed**（2026-09-14，当天那次选择）。
> ⚠️ **别拿这个「0 failed」跟本文别处的「74 failed」对账** —— 两处跑的不是同一个选择：
> 09-16 的回归差分是 `-m cpu_only`，基线与改动后**都是 74 failed / 2322→2330 passed**，
> 那 74 条是**存量**（见「benchmark vdw 漂移」那组，是别人删 `converged` 留下的），
> 不是这两批改动造成的。三处数字（2383 / 2455 / 2330）各自带日期与选择，**都不是当前值**。

> **本轮还没查的**（被叫停时正在做的，留给下次）：
> · `BUD-07` 的验证；
> · `read_aggregated()` 的段胜出逻辑、`sampling_units` / `solver_skip` 的构造；
> · `stage_quality_gate_failures()` 五道门的读数与 9b/9c 归因的对应关系；
> · `decide()` 分支顺序：`5a`（f_k 探针建议重标定）排在 `5b`（支撑/偏斜归因）**之前**，
>   所以 cyclod_ligand1/rep2 里一个 `HARD_INSUFFICIENT`（自检 0.386）的窗口拿到的是
>   `RECALIBRATE_FK` 而不是缩跨度 —— 是不是 bug 需要判据侧确认。

---

## 3. 28 个报错 run 的分类与修复后重放（2026-09-16）

### 2026-09-16 benchmark 报错 run 的分类（`BM-01` ~ `BM-05`）

> ⚠️ **本节说的「28 个报错 run」与 `BM-B` 说的「39 rep / 30 个未完成」是同一批数据的两次统计，
> 口径不同，不要相加**：本节按**报错签名**数（一个 run 可以既报错又最终跑完），
> `BM-B` 按**最后一次启动之后是否跑完**数。冲突以 `BM-B` 为准 —— 它用的取证口径更严
> （启动横幅时刻 / 最后一次启动之后的异常 / 归档目录），本节写下来时那三条口径还没定。

> **来源**：13 个体系 × 3 rep 的 benchmark 跑完后，28 个 run 带错误退出。
> 逐个查了 `pipeline.log` / `launch.log` / `checkpoints/path_versions/` / 落盘产物。
> 归属结论：
>
> | 类别 | run 数 | 归属 |
> |---|---|---|
> | 布局变更后发出构造上不可能成功的动作（崩溃） | 7 | **本仓库 bug** → `BM-01`/`02`/`03`，已修 |
> | 末窗被插成拆不开的非法布局（崩溃 + 非法布局落盘） | 2 | **本仓库 bug** → `BM-03` 已修表层；死局本身 → `BM-04`，**2026-09-16 已拍板关闭** |
> | `LOCAL_VALIDATION_CAP` 撞单周期上限 | 3 | 2 个是 window-0 死局 → `BM-04`（已拍板）；`jnk1_ligand1/rep3` 修完后能继续（见下方重放表） |
> | 插件 `0.1 Å` 几何下限误报 | 3 | **本仓库 bug** → 已由 [`LR-06`](../TODO.md#2-local-residual--exp-033) 立项，别在本节重复 |
> | 带电配体复合物腿缺 reserved co-ion | 3 | **本仓库缺功能，2026-09-16 已修**（`runabfe.py` 派生 `.gro`/`.top`，`[molecules]` 末尾追加；`tests/test_complex_leg_reserved_coion_2026_09_16.py` **17 passed**，含 thrombin_ligand1 端到端）。**重跑即可。** |
> | 求解器缺窗（去相关后帧数不足） | 4 | **不是 bug** —— 真采样不足，见本节末「`BM-A` 不是 bug」 |
> | 其它单发（3 条旧 run + 1 条已修） | 6 | **不是 bug / 已修**，见本节末 |
>
> ⚠️ **第一轮我有四处归因是错的，写在这里免得下一个人重走：**
> 1. ~~C 类是 09-15 换分窗协议作废了 09-11 的旧产物~~ —— **证伪**：
>    `brd4_ligand1/rep1` 五个窗口产物全是 09-16 02:31–02:51 的新文件、形状
>    `(7,6,4,4,4)` 与当前布局逐字一致。真因是**本次运行内自己插的 λ**。
> 2. ~~`g=75.4` 却报 `106/1000` 是记账自相矛盾~~ —— **不是**。多段路径返回
>    `concatenate(indices)`（各段去相关数**求和**）配 `g_values[worst]`（各段 g 的
>    **最大值**），自洽。见 `ibs_engine.py:21473`。
> 3. ~~500 帧卡在判据线上、过不过基本抛硬币~~ —— **过强**。772 个实测样本：
>    去相关帧数中位数 47，低于门槛 10 的只占 6.5%。不是系统性的。
> 4. ~~`LOCAL_VALIDATION_CAP` 炸到 `_assert_stage_result_sane` 是契约越界~~ ——
>    **不是**。自治循环确实把它当路由信号处理过（重判、重试、`NO_FEASIBLE_ACTION`
>    退出），顶层拒绝一条不完整路径是设计。**这条别再"修"。**

> **`BM-01`~`03`、`BM-05` 于 2026-09-16 一并修完。** 回归差分（HEAD 不能当基线 ——
> 工作区本来就带着别的会话的未提交改动，所以是**在当前工作区上只回退本次改动**
> 跑同一套 `-m cpu_only`）：
>
> | | failed | passed |
> |---|---|---|
> | 回退本次改动 | 74 | 2322 |
> | 带本次改动 | 74 | 2330 |
>
> 失败集合逐条一致（那 74 条是存量，见 `benchmark vdw 漂移` 那组），多出的 8 个
> pass 正好是新增的 `tests/test_layout_change_does_not_crash_the_loop_2026_09_16.py`。
>
> ### 修完之后这些 run 走得到哪（2026-09-16，当前代码只读重放 `decide()`）
>
> | run | `decide()` | 说明 |
> |---|---|---|
> | brd4_ligand1/rep1 | `PROBE_CANDIDATE_FK[3]` | 不再崩，可重跑 |
> | jnk1_ligand1/rep2 | `RUN_PRODUCTION[4]` | 可重跑 |
> | jnk1_ligand2/rep1 | `RUN_PRODUCTION[3]` | 可重跑 |
> | jnk1_ligand2/rep2 | `PROBE_CANDIDATE_FK[2]` | 可重跑 |
> | jnk1_ligand2/rep3 | `RUN_PRODUCTION[4]` | 可重跑 |
> | cyclod_ligand3/rep2 | `PROBE_CANDIDATE_FK[0]` | 可重跑 |
> | p38_ligand1/rep3 | `PROBE_CANDIDATE_FK[1]` | 可重跑 |
> | jnk1_ligand1/rep3 | `RECALIBRATE_FK[4]` | 可重跑（**原以为是 window-0 死局，重放证伪**） |
> | p38_ligand1/rep2 | `RUN_PRODUCTION[5]` | 可重跑（win5 g=100.1，要加采样才有用） |
> | cyclod_ligand3/rep1,rep3 | ⚠️ `RUN_PRODUCTION[0]` 是**假象** | 启动布局校验先抛 `RuntimeError` ⟹ **起不来**，见 `DATA-02` |
> | cmet_ligand1/rep1 | `NO_FEASIBLE_ACTION[0]` | `BM-04`（window 0 + 末窗已顶满 K=8） |
> | p38_ligand2/rep1 | `NO_FEASIBLE_ACTION[4]` | warmup 预算耗尽 + `BM-04` |
> | cmet_ligand1/rep2 | `NO_FEASIBLE_ACTION[0]` | λ 总数不够，见 `DATA-03` |
> | cmet_ligand2/rep1 | `NO_FEASIBLE_ACTION[4]` | λ 总数不够，见 `DATA-03` |
> | p38_ligand2/rep3 | `NO_FEASIBLE_ACTION[2]` | λ 总数不够，见 `DATA-03` |
>
> ⚠️ **「不再崩」不等于「能跑完」** —— 上表只说 `decide()` 现在给得出可执行动作，
> 没有任何一个 run 真机验证过。

> **留下一条规矩**：`INSERT_LAMBDA` 明写「本动作不采样，受影响窗口的重采由下一轮
> 逐块发 `RUN_PRODUCTION`」，所以**从插 λ 到下游重采完成之间，下游产物描述的是
> 上一套布局 —— 这是设计内的合法中间态，不是要消灭的状态**。凡是消费窗口产物的
> 代码都必须能在这个中间态下工作；`ibs_engine` loader 的 fail-closed 是对的，
> 错的永远是"在这个中间态里去载它不需要的窗口"。

> `BM-01` / `BM-02` / `BM-03` / `BM-05` **2026-09-16 已修**，整段进
> [archive/TODO_closed_2026-09-16.md](TODO_closed_2026-09-16.md)（含逐条位置、判据测试名、回归差分）。
> 留下三条规矩，**别重新论证**：
> 1. **局部动作只许载它要动的窗口** —— loader 的 fail-closed 布局校验是对的，
>    错的永远是「在插 λ 的合法中间态里去载它不需要的窗口」；
> 2. **停滞保护的降级必须走 `decide()` 的守卫**，不许直接改写 `act`；
> 3. **可行性与可落性是两问**（「该不该发」/「发了能不能落」），插 λ 两侧都要判 ——
>    只判一侧就会把非法布局写进版本链，`DATA-02` 那两个 run 就是这么卡死的。
>
> ⚠️ `BM-04`（window 0 死局）**没有**被这四条解决 —— 它已于 2026-09-16 由维护者拍板关闭，
> 结论与理由见本节末尾的拍板块与 [archive/TODO_closed_2026-09-16.md](TODO_closed_2026-09-16.md)。

---

## 4. 维护者拍板：`BM-04` / `DATA-02` / `DATA-03`

### 📌 与代码无关的数据状态问题（不是 bug，但会挡住 run）

> `DATA-01`（`cyclod_ligand2/rep3` win4 的 production manifest 与 f_k 对不上）
> **已消解**（维护者清空三个 run 的 stage-2 产物重跑），整段进
> [archive/TODO_closed_2026-09-16.md](TODO_closed_2026-09-16.md)。**判据没变**：分析 loader 仍然对「manifest 与两份 f_k 都对不上」
> fail-closed，下次再出现照样拦。

> ### ✅ 2026-09-16 维护者拍板：`BM-04` / `DATA-02` / `DATA-03` 三条同日关闭
>
> 覆盖 benchmark 里 **7 个走不动的 run**。全文（含逐条理由、已核实的源码证据、
> 验收判据、GPU 代价）进 [archive/TODO_closed_2026-09-16.md](TODO_closed_2026-09-16.md)。一句话结论：
> **4 个旧/非法 Stage-2 干净重建，3 个预算不足 run 用 `21/4` 有界续跑。**
>
> | 条目 | 结论 |
> |---|---|
> | `BM-04` | **不开放 `anchor = window 0`**（win0 无前置共享态 ⟹ 那等于新增「整条路径重分」，不是放宽判断）。`cmet_ligand1/rep1`、`p38_ligand2/rep1` 保留 pilot/preopt、**新建 Stage-2 命名空间**、按首窗 `cap=4` 重建，**不**加 `stage2_final_n_states` |
> | `DATA-02` | **选 ②**：`cyclod_ligand3/rep1,rep3` 整体归档后重建。不回退 v3（`_publish()` 明文「只前进不后退」）、不伪造 v5。⚠️ 重建**必须同样带 cap=4**，否则重演 |
> | `DATA-03` | `max_path_insertions` **3 → 4**，`stage2_final_n_states` **保持 21**，原地 resume。**不许再加到 5** —— 第 4 次仍不足就说明初始网格偏小，改走 23 干净重跑 |
>
> **拍板前堵上的缺口**：`stage2_first_window_max_states` **此前从未在任何一次真实运行里生效过**
> —— 3/3 预设、14/14 benchmark config、CLI 全都没有它；`abfe_config.json` 里那个 `4`
> **不会被自动加载**（`--config` 默认 `None`，合并基底是 `PRESET_CONFIGS`）。
> 已加进 `runabfe.py::PRESET_CONFIGS` 三个预设。
>
> 📌 **`abfe_config.json` 不是「仓库默认配置」，只是一份样例。**
> 判某个键在某次运行里是什么值，**只能看该 run 的 `run_provenance.json`**。
>
> 🔑 **本次确立的决策原则**：**「算对之前，沉没成本一律不计。」**
> 代理本轮两次把已有缓存的价值算高了（都被纠正）。实际那批缓存里只有 2 个 run
> 出自修复后的代码，其余完成的全是旧码、本来就不可引用。
> **一个还没被证明算得对的结果，它的缓存没有保护价值** ——
> 同类权衡**先问「这条路径算得对吗」，再问「重算要多少 GPU」，顺序不能反。**
