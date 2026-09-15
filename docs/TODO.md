# 当前行动清单

[项目入口](../README.md) · [文档导航](README.md) · [当前科学状态](STATUS.md) · [变更记录](CHANGELOG.md)

> **本文是本仓库唯一的待办清单**（2026-09-12 整理）。一条待办只在这里出现一次；
> 专题文档负责讲"为什么"，本文只负责"还欠什么、欠在哪一行"。
>
> **项目处于开发最末期**，不是在筹备一次首发：核心流水线、Stage-2 自治控制器
> （2026-09-12 17:22 首次独立跑完整条）、闭式重训都已跑通。剩下的是收尾 ——
>
> ⚠️ **2026-09-14 更正**：那次"首次跑通"之后，控制器在四个 benchmark run 上
> 连续暴露了十余个会断掉闭环的缺陷（真机 5 起 + 静态复核两轮共 15 条），
> 当天全部修完、离线 **2383 passed / 0 failed**，但**真机零验证**。
> 在拿到一次完整闭环之前，「Stage-2 自治闭环已跑通」这句话只对 09-12 那个
> 单体系单 run 成立，**不要外推**。
> 所以本文里没有"要建什么新能力"，只有"哪一处还没收干净"。
> [RELEASE_READINESS](RELEASE_READINESS_2026-08-31.md) 里的「预览版前」「首发支持范围」
> 是 08-31 的措辞，**按末期收尾读**，别当成一次尚未开始的发布筹备。
>
> **已关闭的条目不留在这里**，整段移进 `archive/`：
>
> | 归档 | 内容 |
> |---|---|
> | [archive/TODO_closed_2026-09-09.md](archive/TODO_closed_2026-09-09.md) | `MIGRATE-01` / `PBC-01` / `XFAIL-01` / `XFAIL-02` / `CACHE-01` / `CFG-01` 六条已关闭缺陷的完整记录 |
> | [archive/TODO_closed_2026-09-12.md](archive/TODO_closed_2026-09-12.md) | `BOR-01`（Boresch 几何 minimum-image 口径）与 `S2-D`（控制器代码收拢）两条已关闭条目的完整记录 + 关闭验收。留下的规矩：**同一个不变量只能有一份实现**；**收拢的判据是“写盘的留 pipeline”，不是“塞进一个文件”** |
> | [archive/TODO_closed_2026-09-13.md](archive/TODO_closed_2026-09-13.md) | `LR-01`（残差开关进 stage 指纹的作用域）一条已关闭条目 + 只读核实证据。留下的规矩：**一个开关有几条进指纹的路径就得收窄几条**；且**顶层 run 指纹与 stage 指纹口径不同**，顶层无条件进是对的，别跟着"统一" |
| [archive/TODO_2026-08-06_unreconciled.md](archive/TODO_2026-08-06_unreconciled.md) | 2026-08-06 的主表（1350 行，`ATT-xx`/`MEM-xx`/`P0-9~13` 编号）。**那里面的 `- [ ]` 只表示"当时未完成"**，要人逐条对账才能重新变成待办 |
>
> ⚠️ 不要按编号跨文档机械对账：`P1-19` 在 08-29 那份交接里是"v4 charging 接缝内静电失配"（已修），
> 在 08-06 主表里是"per-window σ 系统性低估 2–4 倍"（未完成）。**同名不同义。**

## 优先级

| | 在推 | 内容 |
|---|---|---|
| **1** | 🔴 **代码侧没有收口** | [Stage-2 自治闭环](#1-stage-2-自治闭环) —— `S2-A`~`S2-F`、`CTL-01`~`CTL-10` 已关闭，但 2026-09-14 第三轮复核又开出 `BUD-01`~`07`（**预算系统整条没接上**：生产 cap 从不存在、补帧块账恒空、单窗实测烧到 1.5M 步）与 `CTL-11`~`15`（converged 了也判不出 `DONE`、no-op 台账只有 3/12 个动作会读）。6 个 benchmark run 用**当前代码**重放 `decide()`：**只有 1 个给出 `DONE`**。另剩 `S2-E`、`AUDIT-S2-01`~`03`。**2026-09-14 另有六路并行全面审计（63+2 条），见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md)** —— 下面 `BUD-*` / `CTL-*` 已被它覆盖的逐条标在各条目下，**原条目一律不删**。⚠️ 上一条「2383 passed」是离线测试，**它对上面这些一条都没覆盖到** |
| **2** | 排在后面 | [local-residual / EXP-033](#2-local-residual--exp-033) —— P2 是唯一还开着的，P1 真机没跑过 |
| **3** | 收尾 | [发布工程门](#3-发布工程门) —— 判据是 clone 下来 import 得动、跑得动 |
| **4** | 不修 | [已定位、判定不改](#4-已定位判定不改) —— `AUDIT-01`~`07`，理由别重新论证。**故意不用 checkbox**，它们不会被打勾 |
| **5** | 暂停 | [膜受体–配体路线](#5-膜受体配体路线暂停) —— 停在 C4，当前主线是可溶体系。**这一节占全文 62 个框（约 2/3），是暂停中的将来工作，不计入当前工作量** —— 数框之前先看这一行 |
| **6** | 条件性阻塞 | [PHY-03](#6-phy-03带电路线的条件性阻塞) —— 带电配体路线 |

---

## 1. Stage-2 自治闭环

> ### ✅ 验收口径已首次达成：2026-09-12 17:22
>
> 自治控制器独立走完整条 Stage-2（六窗全部 `ANALYSIS_ELIGIBLE` → `DONE`），
> 体系 `cyclod_ligand2/rep1`（环糊精主客体，可溶），
> ΔG_bind = **−3.48 ± 0.47 kcal/mol** vs 实验 −4.04，差 **1.19σ**。
>
> **设计、实证与十条陷阱全部在
> [STAGE2_CONTROLLER_DESIGN_2026-09-12.md](STAGE2_CONTROLLER_DESIGN_2026-09-12.md)
> —— 接手先读那份，别重推。** 本节只登记它 §7 列的缺口。
>
> ⚠️ 一次跑通 ≠ 通用。这是**单体系单 run**，没有独立重复。

### 剩余缺口（源：控制器 design §7）

> `S2-D`（代码收拢）**2026-09-12 已关闭**，整段进
> [archive/TODO_closed_2026-09-12.md](archive/TODO_closed_2026-09-12.md)。
> 一句话结论：不是"都塞进一个文件"，是**决策同源的进 `abfe_preoptimizer`、
> 写盘的留 `abfe_pipeline`**。逐符号现状见
> [设计文档 §9.4](STAGE2_CONTROLLER_DESIGN_2026-09-12.md)。

> `S2-B`（完整性要求随预算膨胀）**2026-09-13 已修**，整段进
> [archive/TODO_closed_2026-09-13.md](archive/TODO_closed_2026-09-13.md)。
> 一句话结论：**原条目的后果描述是错的** —— 「可达性判据第 2 档因此失效」从没
> 发生过，那个量在本仓自第一个 commit 起就只进报告、不当门。真实危害是它会骗
> 读它的人（可达性的 T 一度就被错取成它，gcrit 算小 20 倍）。修法仍按原定正解：
> 去掉预算那一支。

> `S2-F`（10 条旧断言对齐新语义）**2026-09-13 已关闭**，整段进
> [archive/TODO_closed_2026-09-13.md](archive/TODO_closed_2026-09-13.md)。
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

- [ ] **DECORR-01 「去相关帧数」有两份实现，口径差 2–8 倍。**
  逐窗自检（`window_self_support_check`）与全局求解器各算一次：`cyclod_ligand2/rep2`
  实测 win3 自检 **56** 帧、`sufficient=True`，求解器报 **9 / 7** 帧并把它**跳过**；
  win0 自检 21 帧 (g=24.5)、求解器报 9。两个数**门着不同的东西**（自检喂控制器的
  `self_sufficient`，求解器的决定跳不跳），于是控制器会认为一个被跳过的窗口"帧数
  够"。这是"同一个不变量的 N 份实现"的又一例。**谁是权威没定** —— 定之前别把任何
  一侧改成向另一侧看齐。

> `CTL-01` ~ `CTL-10`（2026-09-14 第二轮静态复核的**全部**未收口项）
> ⚠️ 同日第三轮的六路并行审计把这一片重新完整扫了一遍，见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md)（#20/#22/#23/#25/#27/#30 等）。
> **2026-09-14 当天十条全部关闭**，逐条改动与判据见
> [CHANGELOG](CHANGELOG.md) 同日那条（很长，是本组的完整记录）。
>
> 留下四条规矩，**别重新论证**：
>
> 1. **「同一个量两份实现」是本项目最贵的复发模式。** 这一轮又栽了一次
>    （`CTL-10` 的 `warmup_steps_left`），前四例是去相关帧数、f_k 字段、
>    跳窗清单、生产计账。**加任何"从盘上读一个量"的代码之前，先问它有没有
>    第二个来源、哪个是权威。**
> 2. **调度状态不能代替失败归因**（`CTL-02`）。「这个单元还没做完」和
>    「它为什么不合格」是两件事，压进一个布尔就会让偏斜类失败被当成样本量
>    不足反复加帧。这是本组里**唯一的思路错误**，其余九条都是接线错误。
> 3. **未知不是零，两个方向都不是。** 上限未知 ≠ 上限为零（会虚报耗尽）；
>    消耗未知 ≠ 消耗为零（会虚报余量）。账不完整时余量和"耗尽"都必须是 unknown。
> 4. **贵动作要在采样之前登记意图**（`CTL-04③`）。先跑后记账 ⟹ 中途退出
>    留下已烧 GPU 却未登记的孤儿，下次会重建一遍。
>
> ⚠️ **十条全部只有离线验证**（每条都做过还原变异确认会红），**真机零验证** ——
> 下一次完整 run 就是它们的第一次上机。

### 🟡 2026-09-14 控制器梳理：剩余项（`AUDIT-S2-01` ~ `03`）

> 当天做了一次**逐出口的系统梳理**（45 个出口、12 个动作全扫）。
> 抓到三条，两条当天修完；下面是**剩下的**与**结论**。
>
> **梳理的方法本身值得留下**：按"今天已知的失效形状"逐个出口对账 ——
> ① 动作在构造上不可能成功；② 调度状态代替失败归因；③ 同一个量两份实现；
> ④ 未知当成零/通过；⑤ 部分和冒充完整；⑥ 没有停止条件；
> ⑦ 动作与出口自相矛盾。七条里每一条今天都真实发生过至少一次。

- [ ] **AUDIT-S2-01 `SPLIT_TAIL_WINDOW` 声明了、执行器也认，但 `decide()` 从不发。**
  它在 `ACTIONS` 里、`_run_stage2_autonomous` 有对应分支，但现在所有发它的位置都被
  `_tgt_is_tail` / `_can_split_skew` 之类的条件收窄掉了。
  **可能是对的**（`CTL-07` 正是在修"中间窗失败却拆末窗"），**也可能收得太死** ——
  末窗真的溢出到可拆区间时还有没有路径走到拆窗？
  ⚠️ **需要维护者拍板**，不要自行放宽：放宽的方向正是 `CTL-07` 刚修掉的那个错。
  判据建议：构造一个"末窗 K 落在 `[2lo−1, 2hi−1]` 且末窗自己是最差窗口"的盘面，
  看 `decide()` 给不给 `SPLIT_TAIL_WINDOW`；给不出就是收过头了。

- [ ] **AUDIT-S2-02 `decide()` 已经 1319 行 / 45 个出口 —— 这是今天所有"漏改一处"的共同成因。**
  实测漏过的：`1c`/`1d` 的准入门（脚本中途抛错、**整份写入被中止**而悄悄丢掉，
  当时测试照样绿）、`5b` 的偏斜归因、`9b` 的子窗补帧、held-out `REJECT` 的末窗判据、
  一处 `DONE` 配 `NO_FEASIBLE_ACTION`。
  **每次复核都能再挖出一条，不是因为复核的人厉害，是因为一个 1319 行的函数没人能一次看全。**
  当天的应对是**把判据往 `plan()` 收**（预算门、补帧准入门都收进去了，各自带结构性测试
  钉住"只许挂一处"）—— 这挡得住"漏挂"，**挡不住"分支顺序错"**。
  真要治得把 `decide()` 拆成按证据类型分派的几段，**那是一次重构，需要单独立项**。

- [ ] **AUDIT-S2-03 控制器真机零验证。**
  2026-09-14 一天约 **30 处**修复（两轮静态复核 + 五起真机事故 + 一次梳理）
  全部只有离线测试与还原变异验证。三个 benchmark run 当时还在**另一台节点上跑旧代码**。
  **在拿到一次完整真机闭环之前**：不得声称「Stage-2 自治闭环可用」；
  不得删除任何旧修复路径（`S2-E`）；控制器产出的任何 ΔG 都不是可引用结果。

> **当天已修（梳理抓到的两条）**：
> · 补帧准入门只挂住 1/16 个出口 ⟹ 改为在 `plan()` 里挂**一次**，并加 AST 守卫
>   钉住"只许一处、且必须在 `plan()` 内"；
> · `action=DONE` 配 `exit=NO_FEASIBLE_ACTION`（`plan()` 会算成 `execution_status=COMPLETE`，
>   把"无路可走"记成"执行完毕"）⟹ 改 `NO_ACTION`，并加 AST 守卫扫全部出口的动作/出口自洽。

### 🔴 2026-09-14 第三轮复核：预算系统（`BUD-01`~`07`）+ 控制器（`CTL-11`~`15`）

> **全部只读查出，一条未修，代码零改动。** 编号**接着** `CTL-01`~`CTL-10` 排，
> 别跟那十条（已关闭）混为一谈。
>
> **取证方式**：用**当前工作树的代码**对 6 个 benchmark run 的盘面调
> `Stage2RepairController.for_physical_stage(...).read()` / `.decide()`（控制器只读），
> 另读它们的 `stage2_autonomous_history.json` / `path_versions/v*.json` / `pipeline.log`。
> ⚠️ 盘上那 6 个 run 是**旧代码**跑出来的，它们的 history 不能直接当现在代码的证据；
> 但 `decide()` 的重放是当前代码，可以。下面分开标注。
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

- [ ] **BUD-01 生产预算的 cap 从来不存在 ⟹ `plan()` 的预算闸全程短路。**
  `Stage2RepairController.__init__` 读 config 键 `stage2_production_budget_steps`，
  **这个键不在 `abfe_config.json`，也不在任何一个 run 的 `run_provenance.json` 里**。
  6/6 实测 `cap=None / cap_source=unknown / cap_known=False / stage_remaining_steps=None`。
  连带三条：① `plan()` 里那道统一生产预算闸写成 `... and _pb.get("cap_known")` ⟹ **永远不进**；
  ② `GLOBAL_BUDGET_EXHAUSTED` 是 `TERMINAL_EXITS` 三大真终态之一，而**唯一发出点**就是那道闸
  ⟹ **真机上永不可能发出**，设计 §2 说的三个真终态实际只有两个；
  ③ `IMMUTABLE_REWINDOW` 的预留门 `(not cap_known) or 剩余 >= 预留` ⟹ 恒 True，
  「留不出来就别开系综」从未生效。
  ⚠️ **别只补 config**：先看 `BUD-06`，cap 一旦生效第一次 resume 就会误判耗尽。
  ✅ **已被审计 #31 覆盖并修复**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md) **#31**）：`stage2_production_budget_steps` / `stage2_max_production_blocks_per_window` 已在 **config + preset + CLI 三处**补齐，默认值严格等于原硬编码兜底 ⟹ 行为逐位不变；前者默认 **`null` = 上限未知，不是 0**（控制器写死「未知时不拦」），正是为了不触发 `BUD-06` 那个误判。`max_path_insertions` **刻意只补 CLI**（它进 stage 协议指纹，给默认值会让所有既有 run 的 Stage 1/2 结果缓存全部失配重跑）。**`BUD-06` 本身未被审计覆盖，仍然开着。**

- [ ] **BUD-02 `view["path_version"]` 这个键不存在 —— 一处打穿四个机制。**
  路径版本只在 `view["path"]["version"]`（实测 cyclod_ligand2/rep1 = 4）；顶层**没有**
  `path_version`，而代码里到处 `view.get("path_version")`，**恒为 `None`**：
  · `_run_stage2_autonomous` 写 history 的 `"path_version"` —— 实测 6 个 run 每一轮都是 `null`；
  · `_production_blocks_ledger` 拿它过滤 `it["path_version"] != path_version` ⟹ 真实版本 ≠ `None`
    ⟹ **每条 iteration 都被跳过，补帧块账恒空**；
  · `action_noop_fingerprint(w, view.get("path_version"))`（控制器与执行器两侧都传它）⟹
    指纹缺路径版本这一维 ⟹ **插 λ / 拆窗换了布局之后，旧的 no-op 记录不失效**；
  · `decide()` 里 `relearn_epoch_used(ckpt, int(view.get("path_version") or 0), w)` 读键用 `0`，
    而执行器 `mark_relearn_epoch_consumed` 写的是 `lambda_path_versions.load_current()` 的真实
    版本号 ⟹ **读写不同键，「一个窗口只给一次 fresh Epoch」在控制器侧永远判成「还没用过」**。

- [ ] **BUD-03 块账的第二重失效：换段即清零。**
  即使 `BUD-02` 修好，`_production_blocks_ledger` 还要求
  `snapshot[i]["segment"] == 当前视图里该窗口的 segment`，而**换段正是循环自己的动作**。
  实测 cyclod_ligand2/rep1 win5：history snapshot 里是 `vanishing`，当前视图里是 `vanishing_7`
  ⟹ 全部 `continue`。**窗口每换一次段，它烧过的帧账清零、重新发 4 块。**
  ✅ **已被审计 #40/#41 覆盖并修复**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md) **#40** / **#41**）：#41 修的正是`same_segment_only` 下 `seg_of.get(i)` 对不在当前视图里的窗口返 `None`、与字符串恒不等 ⟹ 刹车静默失效；#40 修的是块账双向错（每轮给视图里**每个**窗口白记一行 / `PROBE_REANCHOR_EPOCH` 确实花一块却被过滤掉）。

- [ ] **BUD-04 `BUD-02`+`BUD-03` 的合计后果：补帧完全没有刹车。**
  `_frames_admission` 的两道判据 —— ① 每窗 `max_production_blocks_per_window`（缺省 4，
  config 里也没有 `stage2_max_production_blocks_per_window`）；② 「上一块必须有实质增益
  （求解器侧去相关帧数）」（需 `len(rows) >= 2`）—— **都读 `production_blocks_by_window`，
  而它 6/6 run 全是 `{}`** ⟹ **两道判据一次都没触发过**。直接对应单窗烧到 1,500,000 步。
  ⚠️ 「补帧没有停止条件」这条被记成已修，实际**生产侧从头到尾没有任何刹车**。
  📌 **口径更新（2026-09-14）**：块账已**不再是单层**，现在是四个平键 ——
  `production_blocks_by_window` / `production_blocks_by_unit` /
  `production_blocks_total_by_window` / `production_blocks_total_by_unit`
  （**同段账 vs 跨段累计** × **物理窗口 vs 采样单元**）。读它的地方要先确认自己要的是哪一格。
  ✅ **已被审计 #8/#30/#31/#40/#41 覆盖并修复**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md) **#8**「子窗两道刹车同时失效」、
  **#30**「块数满额是路由信号不是终态」、**#31** 硬上限进 config/CLI、**#40/#41** 块账双向错）。

- [ ] **BUD-05 同一个「预热余量」有三套 unknown 语义。**
  `warmup_steps_left` 读不到（`None`）时：`per_window_budget_remaining` /
  `all_windows_budget_exhausted` 用 `int(... or 0)` ⟹ **0 = 耗尽**；
  `_epoch_validation_unaffordable` ⟹ **不可行**（fail-closed）；
  `decide()` 分支 1e 的 `left is not None and int(left) <= 0` ⟹ **有钱**。
  实测 cyclod_ligand1/rep3 win4 的 ledger 确实读不到，在第一处被显示成 `0`。
  这是本文上面那条规矩「未知不是零，两个方向都不是」的再次违反。
  ✅ **已被审计根因 ② 覆盖并基本修复**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md) **#15/#36/#38** 已修；**#37 仍 `OPEN`** ——「cap 已知但用量未知」那一路还在 `or 0`）。

- [ ] **BUD-06 `stage_used_steps` 是 lifetime 累计，不是「从这本预算里花掉的」。**
  聚合视图按**所有采样段**累加逐窗生产步数（注释明写「旧段的帧是真烧过的 GPU」），
  插 λ / 拆窗之后**已经作废的旧段**照样计入。⟹ 一旦真给
  `stage2_production_budget_steps` 配上值，cyclod_ligand2/rep1 第一次 resume 就会拿
  **4,000,000** 去比 cap 立刻判耗尽。**「配上 cap」和「cap 生效后行为正确」是两件事。**

- [ ] **BUD-07（未验证，查到一半被叫停）跨段预热预算继承可能只认基准段。**
  `ibs_engine.inherit_warmup_ledger_across_segments` 靠 `base_checkpoint_dir_for()` 找
  「上一层」，而那个函数的锚点是 `path_current.json` —— **只有基准目录有它**。
  所以 `checkpoints/segment_3` 找回的是 `checkpoints/`，**不是 `segment_2`**：
  第 3 个 Epoch 继承基准段的消耗、丢掉段 2 烧掉的量。
  形状与已归档的「多 Epoch 链永远从基准段重学 f_k」一致（那条当时的结论是「第一次换
  Epoch 完全正确，连换两次才踩到」）。⚠️ **只读了函数、没有验证**，需要一个有
  `segment_3` 及以上的 run 对账。

- [ ] **CTL-11 stage 级判据已经通过，`decide()` 仍然不判 `DONE`。**
  实测 cyclod_ligand2/rep2：stage `converged=True`、`missing_windows=[]`、`skipped_windows=[]`，
  当前代码给出 **`PROBE_REANCHOR_EPOCH[0]`**。
  根因：`DONE` 在分支 7，排在「最早未解决窗口」路由**之后**；而 `earliest` 来自逐窗**自检**
  `self_verdict`（该 run win0=4.37、win3=4.81，门 10）。两者是同一个量的两份实现 ——
  stage 级 `converged` 由 `solve_stage_integrated` 在**合并后的全部段**的帧上算（五条合取），
  是权威；自检 `min N_eff/g` 只看该窗口**一个段**的帧，对多段窗口系统性偏悲观。
  ⟹ 任一窗口自检 < 10 就永远轮不到分支 7，**一个已经跑完的 stage，每次 resume 都会
  重开一个 Epoch 烧 GPU**。同形状的第二处：`_evidence_status()` 里「任一窗口 self_verdict
  不合格 ⟹ INSUFFICIENT_DATA」排在 `stage_converged is True ⟹ CONVERGED` **之前**。
  ✅ **部分被审计覆盖**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md)）：**#27** 已修（`_immutable_rewindow_step` 绕过统一入口写盘、不盖 `path_version` ⟹ `stage_result_path_version_verified` 恒 False ⟹ 分支 0a 的 `DONE` **结构上不可达**，这是 `DONE` 判不出来的第二条封锁）；**#18(a) 仍 `OPEN`** —— `stale` 表永不清除 ⟹ `stale_layout_evidence` 恒非空 ⟹ `DONE` / `CONVERGED` 同样被封死。**三条封锁要一起解开，只解一条仍然判不出 `DONE`。**

- [ ] **CTL-12 no-op 台账写 12 个动作，只有 3 个动作会去读。**
  执行器 `_record_noop_action` 对**任何**动作都记账（通用盘面指纹比对），但 `decide()` 里
  `_is_noop()` 只在四个位置被调用、覆盖三个动作：`CONTINUE_WARMUP`、`RECALIBRATE_FK`（两处）、
  `PROBE_REANCHOR_EPOCH`。**`SPLIT_TAIL_WINDOW` / `INSERT_LAMBDA` / `RUN_PRODUCTION` /
  `IMMUTABLE_REWINDOW` / `ANALYZE` 全都不读。**
  后果（旧代码日志实测，形状在当前代码里未变）：cyclod_ligand1/rep2 与 cyclod_ligand2/rep1
  都死在同一条路上 —— `SPLIT_TAIL_WINDOW` 连发 4 次、执行器每次都打了
  「记为当前盘面上的 no-op」、控制器每次都看不见，最后由停滞保护给出 `NO_FEASIBLE_ACTION`。
  ✅ **同形状的子窗半边已被审计 #9 覆盖并修复**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md) **#9**：执行器写 `f"{{action}}:unit:{{uid}}"` 而 `_is_noop` 只查 `f"{{action}}:{{idx}}"` ⟹ 全仓无人读）。**「12 个动作写、只有 3 个动作读」这一半未被审计覆盖，仍然开着。**

- [ ] **CTL-13 补帧准入只查 `windows[0]`。**
  `plan()` 里 `_tgt_w = int(windows[0])`，只拿这一个窗口去问 `_frames_admission`；
  而分支 9c 在「端点 σ 归因不到具体窗口」时发的是 `windows=sorted(全部窗口)`（注释明写是
  有意的）。⟹ win0 块数一满就把**整批窗口**的补帧一起毙掉；反过来 win0 有额度时，
  其余窗口即使已超额也照批。

- [ ] **CTL-14 `RELEARN_FK_EPOCH` 的两处 `break` 没写 `outcome`。**
  `_run_stage2_autonomous` 里该分支的两个提前 `break`（「替代候选已用过一次」「剩余 warmup
  预算 < 新 Epoch 所需」）只写了 `history[-1]["exit"]`，**没有调 `_finish()`** ⟹ `outcome`
  停在 `{"status": "RUNNING", "exit": null, "iterations_used": 0}`。实测 brd4_ligand2/rep1
  的 history 现在就是这个状态 —— 正是 2026-09-14 加 `outcome` 要解决的那个问题。

- [x] ~~**CTL-15 三个 run 的布局已进入无恢复状态。**~~
  🔴 **本条原始判断是错的，已作废（2026-09-14 同日更正）。**
  原文说「brd4_ligand2/rep1 末窗 K=10 > 可拆上限 `2*hi−1 = 9`」—— 那是**拿
  cyclod 的 `lo/hi=4/5` 去套 brd4**。brd4 与 cyclod_ligand1 的
  `stage2_window_max_states` 是 **8**，可拆区间是 `[7, 15]`，K=10/11/12 **全都在
  区间内、可以拆**。实测三条全部能合法化：

  ```
  brd4_ligand2/rep1    (16,26) → (16,21)+(20,26)
  cyclod_ligand1/rep2  (15,27) → (15,21)+(20,27)
  cyclod_ligand1/rep3  (16,27) → (16,22)+(21,27)
  ```

  **留下的规矩**：`lo/hi` 是**逐 run** 从 `run_provenance.json` 读的
  （`Stage2RepairController.__init__` 就是这么写的，理由也写在那里），
  跨 run 引用可拆区间前必须先看那个 run 自己的 `lo/hi`，别拿手边那个体系的数去套。

  插 λ 超预算这件事本身仍然为真（版本链上 5/6 次 vs 预算 3），成因是旧代码
  `_read_path` 读错 `kind` 键、已于 2026-09-14 修好；但它**没有**造成不可恢复的布局。

- [x] ~~**CTL-16 `first_untrusted_window()` 把「证据被布局变更作废」当成「还不知道」。**~~
  **2026-09-14 已修。** 判据只有 `v is not None and v != ANALYSIS_ELIGIBLE`：
  `v is not None` 是对的（没跑过自检 ≠ 有问题，设计 §4 三态），但把
  `stale_layout_evidence_only` 的窗口一起漏掉了 —— 而设计 §4 同一节明写那是
  **确定的未解决**，`_window_state()` 也正是这么判的。**同一个不变量两份实现。**
  真机 cyclod_ligand1/rep3：win0–3 全 ANALYSIS_ELIGIBLE、win4 证据被作废
  ⟹ 返回 None ⟹ 取不到 tail anchor ⟹ 末窗 K=11 > hi=8 时
  `_legalize_tail_window` 在**启动布局校验**处 fail-closed 抛错，
  **`decide()` 一次都轮不到、整个 run 起不来**。判据已与 `_window_state()` 对齐。

- [x] ~~**CTL-17 `0b`（身份不一致）物理位置在分支 5 之后。**~~
  **2026-09-14 已修。** 编号写 `0b`、意图是"最早"，位置却在
  `5) 生产帧没攒够` 后面。而 `IDENTITY_MISMATCH` 的窗口 `production_steps`
  通常正好没到目标，分支 5 又只打 `earliest`（正是这个窗口）⟹ **先发
  `RUN_PRODUCTION`、对着另一个系综的产物补帧**，`HALT_INVALID_INPUT` 永远走不到。
  已移到全部路由之前，判据一字未改。
  ⚠️ `_window_state()` 里的 `IDENTITY_MISMATCH → PROBLEM` 是**另一条**修复
  （它让该窗口成为 `earliest`），必要但不充分，两条互补、别当重复删掉。

- [x] ~~**CTL-18 `_no_gain` 分支不查可行性就发 `INSERT_LAMBDA`。**~~
  **2026-09-14 已修。** 全代码唯一一处发布局动作却不问
  `feas.get("insert_lambda") is None` 的分支；插点预算耗尽 / 末窗顶满时它发出的是
  **构造上不可能成功**的动作。归因（加帧已被证伪）是对的，所以不可行时只改动作、
  不改归因：如实 `NO_ACTION` + `NO_FEASIBLE_ACTION`。

- [x] ~~**CTL-19 9b 的 `SPLIT_TAIL_WINDOW` 传 `windows=[]`，三重失效。**~~
  **2026-09-14 已修。** ① 执行器 `_record_noop_action` 是 `for w in (windows or [])`
  ⟹ 空列表**一条账都不记**；② `plan()` 的 no-op 闸要求 windows 非空 ⟹ 拦不到；
  ③ 主循环停滞保护的降级条件是 `act != PROBE... and wins` ⟹ wins 空就跳过降级、
  直接 `NO_FEASIBLE_ACTION`。真机 cyclod_ligand2/rep1 与 cyclod_ligand1/rep2
  都是这么死的。已改为点名末窗号。

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
> 离线全套：**2455 passed / 3 skipped / 0 failed**。真机仍然零验证。

> **本轮还没查的**（被叫停时正在做的，留给下次）：
> · `BUD-07` 的验证；
> · `read_aggregated()` 的段胜出逻辑、`sampling_units` / `solver_skip` 的构造；
> · `stage_quality_gate_failures()` 五道门的读数与 9b/9c 归因的对应关系；
> · `decide()` 分支顺序：`5a`（f_k 探针建议重标定）排在 `5b`（支撑/偏斜归因）**之前**，
>   所以 cyclod_ligand1/rep2 里一个 `HARD_INSUFFICIENT`（自检 0.386）的窗口拿到的是
>   `RECALIBRATE_FK` 而不是缩跨度 —— 是不是 bug 需要判据侧确认。

### 📌 与代码无关的数据状态问题（不是 bug，但会挡住 run）

- [x] ~~**DATA-01 `cyclod_ligand2/rep3` win4 的 production manifest 与 f_k 全都对不上。**~~
  **2026-09-14 已消解**：维护者把三个 run 的整个 stage-2 产物（`vanishing/` +
  全部段目录 + `ibs_state_*` + `production_window/`）清空重跑，`checkpoints/` 只留
  Stage 0/1。那份 stale manifest 随之不存在了。**判据本身没有变**：分析 loader
  仍然对「manifest 与两份 f_k 都对不上」fail-closed，下次再出现照样拦。
  下面保留原始诊断，供再次出现时对照 ——

- [ ] ~~**DATA-01（原始诊断，留档）**~~
  实测哈希：`manifest == live f_k` **False**、`manifest == production_entry_f_k` **False**
  —— 那份 production checkpoint 是在**第三个** f_k 下写的，比 state 文件还老。
  分析 loader 因此 fail-closed（`窗口 4 冻结 f_k 与 production manifest 不一致`）。
  **这不是 2026-09-14 改出来的**：该窗口 `live == entry`，走的是改动前同一条路。
  对照：rep2 win4 与 brd4 win4 都是 `manifest == entry True`，那两个已由当天的
  `_resolve_analysis_f_k` 修复。
  ~~**待定**：这个窗口是重采、还是把那份 stale checkpoint 作废，需要维护者拍板。~~
  → 维护者选了**清空重跑**。

- [ ] **S2-E 旧修复机制是「关掉」不是「删掉」** —— path_evolution 修复分支 /
  production rescue / rescue 后重标定。**等自治这条再跑通几次再删**，别现在删。

### 不许回退的约定

- **一个 stage 只许有一个控制器。** 不是风格偏好 —— 两套机制各判各的会直接打架。
- **只有三个真终态**：`DONE` / `GLOBAL_BUDGET_EXHAUSTED` / `NO_FEASIBLE_ACTION`。
  `LOCAL_VALIDATION_CAP` / `INSUFFICIENT_DATA` / `CUMULATIVE_FK_MISALIGNMENT` /
  `SKIPPED_WINDOW` / `HALT_BUDGET` **全是路由信号**，不得炸出流水线。
  异常同理：只有 `IBSFrozenCalibrationValidationError`（f_k 被**统计驳回**）是有功效的
  否决，且它也只封存该候选、最多换一次 Epoch，仍不终止 Stage-2。
- **四个量不许混用**：`n_decorr` 是求解器**资格**不是验收量；`N_eff,k / g_k` 是
  **主验收量**（门 = 10）；`top1% weight` 是否决**警报**；occupancy / coverage ESS
  **只诊断 f_k 训练**，永不决定 ΔG 是否有效。
- **低支撑永远不是 `FAIL`**，是「尚不可测」（`INSUFFICIENT_DATA` / `HARD_INSUFFICIENT`），
  动作是加预算。`FAIL` 是支撑够了却违反统计门，动作是换 Epoch。
- **缺证据 ≠ 通过，但缺证据 ≠ 有问题** —— 它是 `UNKNOWN`，动作是去产出证据。
- **累计 f_k 残差只能用 `sampling_states`** —— `energies` 多一个逐 λ 态常数（LRC），
  其逐边差会伪造出「全负号单调」的形状（实测把某窗 span 抬高 2.2 倍）。
  `N_eff/g` 则对它**不变**，两者不变性不同，别"统一"。
- **不得为了得到期待的标签而换误差估计器**（estimator shopping）。新 Epoch 才是拿
  独立证据的地方。
- **`(K−1)×门槛` 不许写** —— 累计偏差门不随边数增长。
- **控制器只读**；执行由独立的 execute 层落盘。

### 设计依据（live，不是待办）

- [STAGE2_CONTROLLER_DESIGN_2026-09-12.md](STAGE2_CONTROLLER_DESIGN_2026-09-12.md)
  —— **权威**：设计 + 实证 + 十条真机咬过的陷阱。最贵的一条是
  **「同一个不变量的 N 份实现」**（λ 身份有四份）。
- [AUTONOMOUS_STAGE2_LOOP_SPEC_2026-09-11.md](AUTONOMOUS_STAGE2_LOOP_SPEC_2026-09-11.md)
  —— 老板定的核心设计指标（唯一验收指标）。
- [STAGE2_AUTONOMOUS_LOOP_STATUS_2026-09-11.md](STAGE2_AUTONOMOUS_LOOP_STATUS_2026-09-11.md)
  —— 逐个 bug 的修复流水账（历史，不是待办）。
- [PLAN_PATH_REPAIR_2026-09-11.md](PLAN_PATH_REPAIR_2026-09-11.md) —— 设计要求的来源，
  分支编号（5a/5b/6）出自这里；代码里 10 余处引它作设计依据。
- [STAGE2_THREE_AXES_COUPLING_2026-09-11.md](STAGE2_THREE_AXES_COUPLING_2026-09-11.md)
  —— 分窗口/分 λ/分采样量三轴当前怎么耦合的。
- [STAGE2_ROOT_CAUSE_2026-08-28.md](STAGE2_ROOT_CAUSE_2026-08-28.md) —— **所有收敛门
  对单轨迹重加权这个失效模式是瞎的**：实测门全绿时 ΔG 错 42 kJ/mol。别拿质量门当验收。

---

## 2. local-residual / EXP-033

背景：`B_φ` **不是配体的物理模型**，是提高 λ 态混合的**采样增强项**；按体系在线学的只有 `f_k`。
方案见 [EXP-033_LOCAL_RESIDUAL_REFIT_2026-09-10.md](EXP-033_LOCAL_RESIDUAL_REFIT_2026-09-10.md)，
P1 落地见 [EXP-033_P1_LANDED_2026-09-12.md](EXP-033_P1_LANDED_2026-09-12.md)。
**`LR-01`（残差开关的指纹作用域）2026-09-13 核实已修，整段进
[archive/TODO_closed_2026-09-13.md](archive/TODO_closed_2026-09-13.md)，别再列。**
**P3（跨配体通用权重）已于 2026-09-12 由用户拍板划掉，前提没了，别重新提。**

- [ ] **EXP-033-P1-GPU（先做这个）P1 闭式重训真机一次没跑过。**
  `closed_form_refit()` 里 probe 那段要建 OpenMM Context，只过了静态校验。
  最小验证：4W53 开 `--outer-lambda-local-residual-ibs`（配体指纹对不上冻结的 Atenolol
  ⟹ 会走重训），跑到预平衡结束看它出不出 manifest。
- [ ] **EXP-033-P2（EXP-033 唯一还开着的一条）** ridge 臂 vs 出厂非线性臂，**U3 口径上机**：
  window-0 utility + ΔG 一致性，且**两臂各自独立标定并冻结自己的 `f_k`**。
  U4 就是栽在候选臂复用 baseline 的 `f_k`，被封为 `INVALID_FOR_PROMOTION`。
  ⚠️ 做 A/B **不要**走 P1 的自动重训（两臂各自重训 ⟹ `sampling_score_sha256` 变成
  run-dependent，两臂不再共用同一把尺子）。要先离线冻一份两臂共用
  （`tools/retrain_local_residual_offline.py`）。
- [ ] **LR-02 `skip_unsupported_frames` 该撤或反转** —— 支撑域外的帧正是"模型没覆盖这个
  体系"的证据，跳过它等于把本该触发停止的信号变成拟合时看不见的样本。
- [ ] **LR-03 `sample-hard-window-scratch` 在主线里是死的** ——
  实现在发布清理时被移出的 `archive/` 里（`outer_lambda_neural_basis.py` 有 **13 处**
  `from archive import`，全是空壳）。而且那份 legacy 实现读 `manifest["lambda_shield"]`，
  WCA 壳退役后该字段是 `None` → `TypeError`。**要么补实现，要么把入口一起删掉。**
- [ ] **LR-04 没有 solvent-only 入口** —— `--only-complex-charging` /
  `--only-boresch-attachment` 与 residual 互斥。
- [ ] **LR-05 重训用的 λ 表是默认值** `linspace(1.0, 0.5, 8)`
  （可用 `outer_lambda_refit_lambda_max/_min/_n_states` 改）。因为重训发生在预优化算出
  真实 stage-2 λ 路径**之前**。按 `B_φ` 的定位这不影响对错；要用真实窗口 λ 表得把重训
  往后挪一个位置 —— 那要改的是**时序**，不是这个模块。

> 🛑 **别在 run 内为了 A/B 重训**，三条理由：违反预注册；`sampling_score_sha256` 变成
> run-dependent；**拿缺构型的帧拟合会把缺掉的态焊进模型**，然后 ESS / overlap /
> split-half / 三方一致全都会更绿——它们只问"这批样本内部自洽吗"。
> 完整的坑清单见 [archive/HANDOFF_LOCAL_RESIDUAL_2026-09-11.md](archive/HANDOFF_LOCAL_RESIDUAL_2026-09-11.md) §6。

---

## 3. 发布工程门

发布定位是 **clone-and-run**（不打包、不发科学结论）。判据只有一条：
`pytest tests/test_fresh_clone_imports.py` 绿。完整论证见
[RELEASE_READINESS_2026-08-31.md](RELEASE_READINESS_2026-08-31.md)。

> ✅ **`REL-01`（预编译 `.so` 随仓库分发）2026-09-12 已执行**，故不再列：
> `.gitignore` 加了三条例外放行 `build` 符号链接 + `build_exp026_a2/*.so`，
> 并单独排掉没有扩展名的 gtest 可执行文件。`git add -An plugins/` 应当**只有 4 条**。
> 文档（[GETTING_STARTED.md](GETTING_STARTED.md)《CUDA 插件》/
> [TROUBLESHOOTING.md](TROUBLESHOOTING.md)）已同步成"随仓库分发、按环境文件建环境不用编"。

- [ ] **REL-05 代码内的中文要切成英文**（**末期收尾项，现在不做**）。
  2026-09-12 实测（已跟踪的 223 个 `.py`、157,528 行）：

  | 类别 | 行数 | 性质 |
  |---|---|---|
  | 含中文的行（合计） | **30,642**（19.5%），分布在 **190** 个文件 | |
  | 纯注释（`#` 开头） | 13,023 | 只影响维护者 |
  | docstring / 日志 / 异常 / 行尾注释 | 17,619 | 其中一部分用户看得见 |
  | **同行含 `raise`/`logger`/`print` 的** | **1,563 行、54 个文件** | **用户直接看得见**；多行拼接没算进去，真实数只多不少 |

  **优先级不是一刀切**：真正非切不可的是那 1,563 行**用户可见的异常与日志文本**
  （fail-closed 报错是这套流水线的主要交互面，报错看不懂等于没有 fail-closed）；
  注释与 docstring 可以最后再动、甚至不动。
  ⚠️ **不要现在做**：异常文本在大量测试里被 `match=` 断言，批量改动会跟正在进行的
  `S2-D` 重构直接撞车。**等重构落地、方法定稿再排。**

- [ ] **REL-06 主页挂一张流程图。** 三份 README 的 `![` 计数都是 **0**；
  `docs/current-pipeline.svg` 是现成的，但只有 `docs/README.md` 引它。
  ⚠️ **不要现在做**：Stage-2 控制器正在重构（`S2-D`），现在挂等于挂一张要过期的图。
  **等重构落地后重新出图再挂。**

- [ ] **REL-07 「本仓库目前没有可以作为最终结论引用的结果」这句话的口径。**
  字面没错（单 seed、无独立重复），但读起来像"什么都没验证过"——
  实际上 4W53 差 **1.83σ**、环糊精差 **1.19σ** 两个闭环都在。
  ⚠️ 这条碰**科学结论口径**，`docs/README.md` 维护规则第 1 条要求科学结论只写
  [STATUS.md](STATUS.md)、别在 README 复制。所以**改法必须是"一行指路"而不是"搬数字"**，
  且**要维护者本人拍板**，不得由代理自行改写。

> ### 🛑 两条**明确不做**，理由别重新论证
>
> | | 为什么不做 |
> |---|---|
> | **`CITATION.cff` / 引用信息** | 项目还在 dev，**方法本身没做完**。现在写引用就是承诺一个还不存在的东西。等方法定稿再写 |
> | **英文文档追平中文** | **中文是主文档，这是为开发方便的既定选择**，不是疏漏。`docs/` 下的教程以中文为准；`README_en.md` 覆盖完整流程即可，不做逐字对等 |

- [ ] **REL-02 `abfe_core.py` 分片没审完** —— 第九轮审查里它是唯一没有分片正文的
  （5 条 P2 只有汇总行）。而《五个文件分别应补什么》恰恰把它的职责定为"集中最终结果资格
  与协议登记"。补审属于预览版前的工作。
- [ ] **REL-03 到目前为止全部是 CPU / 静态验证，零 GPU 复验。** 最需要真机的：
  - residual 臂混合覆盖度门换口径后，EXP-030 candidate 臂的门读数（`ess_gate_mixture_gauge` 应为 `sampling_states`）；
  - 三个 decharging builder 新增的 `frozen_ll_pairs` 断言（真实体系上触发 ⟹ P0-01 的前提本来就不成立，那是新发现不是回归）；
  - 两条腿同进程时的 `pipeline.log` 分离（已用最小复现验证，未在真实两腿运行上确认）。
- [ ] **REL-04 2026-09-09 全仓审计那 52 处改动无一上过 GPU。**
  最需要复验的三处：偏置爬坡补 1.0 档、preopt 探针的 force group 重划、加密点的采样语义变更。
  复跑命令见 [archive/AUDIT_2026-09-09_full_repo.md](archive/AUDIT_2026-09-09_full_repo.md) §4。

> 下面三条 2026-09-13 从 [RELEASE_READINESS_2026-08-31.md](RELEASE_READINESS_2026-08-31.md)
> 并入。它们在那份表里是"仍欠"，但本文没有对应条目 —— 违反本文规则 1「一条待办只在本文出现
> 一次」的反面：**一条待办一次都没出现**。判据栏原文保留在 RELEASE_READINESS 的同名行。

- [ ] **REL-08 基础环境与 GPU/ML 可选环境未分离** ——
  [RELEASE_READINESS:173](RELEASE_READINESS_2026-08-31.md)。Python 版本已于 2026-09-12
  统一成 3.12；`environment.yml` 的个人 `prefix` **2026-09-13 核实已无**（全文件 grep
  不到 `/home/`），那半条已过期。**仍欠的只有拆分**：把 CUDA 12.9 开发工具链 / torch / MACE
  这类只有 GPU 才用得上的依赖从基础环境里拆出去。
  判据：干净机器按文档只装基础环境能 import、能跑 CPU 测试子集。
- [ ] **REL-09 输出目录与长任务保护** ——
  [RELEASE_READINESS:177](RELEASE_READINESS_2026-08-31.md)。现有的是 checkpoint + 短时
  pipeline state lock；**欠覆盖整个作业生命周期的输出目录独占锁、SIGTERM 处理、磁盘预检**。
  判据：重复启动同一输出目录被明确拒绝；调度器中断后可从一致边界续跑；磁盘不足的失败可解释。
- [ ] **REL-10 软件版本号与变更说明** ——
  [RELEASE_READINESS:178](RELEASE_READINESS_2026-08-31.md)。LICENSE（MIT + NOTICE）已有，
  `CITATION` 已有意押后（见上方"明确不做"），**只欠版本号本身**：冻一个版本、写清支持范围。
  ⚠️ 与 `REL-05`/`REL-06` 同理，**方法定稿前不急**；登记在这里是为了不再从 RELEASE_READINESS
  里被重新"发现"一遍。

---

## 4. 已定位、判定不改

> 完整记录见 [`archive/AUDIT_2026-09-09_full_repo.md`](archive/AUDIT_2026-09-09_full_repo.md)。
> 那轮审计共 62 条候选，**52 条已修**（含 NaN 根因、stdout 消失、离线重算挂死、
> 一批宿主内存放大、preopt 缓存两层拆分）。下面 7 条是**有意不修**的，
> 登记在这里是因为它们确实"已定位、未修" —— 下一个人有权知道，
> 也免得被当成新发现重查一遍。**理由都别重新论证。**

> 📌 **这 7 条故意不用 `- [ ]`** —— checkbox 的语义是"待办、将来会打勾"，而这些永远不会
> 被打勾。用 checkbox 写会让任何 `grep -c '^- \[ \]'` 得到一个虚高的待办数。

- **AUDIT-01** `abfe_core.minimum_image_displacement_nm` —— 候选立方体随长宽比增长，
  `radius > 64` 的 guard 在尝试 `N × 2.1e6 × 3 × 8` 字节之后才触发。
  *不修*：真实触发需要极端长宽比 + 大批量输入；`docs/design` 的盒型识别提案覆盖这一片。
- **AUDIT-02** `abfe_core` 膜 leaflet 的 wrapped/unwrapped 混用
  （`_protein_leaflet_cross_sections_nm2`、`assign_lipid_leaflets`、
  `verify_membrane_normal_axis`）。
  *不修*：只影响膜路径，且 `membrane_observables_from_trajectory` 已按最大空隙弧修过一次；
  当前生产是可溶体系，留给膜线单独一轮。
- **AUDIT-03** `free_energy_engine.run_independent_windows` 保留全部帧
  （73k × 1000 帧 × 8 态 ≈ 14 GB）。
  *不修*：当前无生产调用者，是"接线即爆"而不是现在就爆。接线前必须先改。
- **AUDIT-04** `apbs_correction._read_dx_values` —— 257³ 网格约 1 GB 峰值
  （原文本 + 切片副本 + Python float list 三份）。
  *不修*：APBS 修正当前不在主线路径上。改法是 `np.frombuffer`，可降到 ~136 MB。
- **AUDIT-05** `abfe_core.OnlineConvergenceMonitor` 的 K==1 会抛、
  `n_k_array` 与 `u_kn` 列数不一致。
  *不修*：`abfe_core` 之外无调用者。
- **AUDIT-06** `TraditionalABFEPipeline.pre_equilibration_identity_fingerprint`
  引用 `self.pressure` / `self.barostat_protocol`，该类 `__init__` 从未赋值。
  *不修*：全仓无活的调用点（runabfe 里那两个 baseline 都是 `ABFEPipeline` 实例）。
- **AUDIT-07** `abfe_pipeline._rebalance_fingerprint` 没有 System 身份绑定
  ⟹ System 变了而 Boresch 锚点没变时，`rebalance.chk` 会被复用。
  *🛑 明确不修*：唯一修法是把自产产物的 sha256 放进缓存身份，而那是**用户否决过
  4 次**的做法（`code_sha256` 2026-08-24、`system_xml_hash`+`positions_sha256`
  2026-09-09、`preopt_cache_sha256` 2026-09-09）。这类 payload 本来就有显式协议
  版本号承担"算法变了"的信号。**不要再提这个方案。**

> ⚠️ 那 52 处**没有一处上过 GPU**。最需要真机复验的：偏置爬坡补 1.0 档、
> preopt 探针的 force group 重划、加密点的采样语义变更。
> 复跑命令见审计文档 §4。

---

## 5. 膜受体–配体路线（暂停）

> **2026-09-12 状态：整条线暂停**，停在 C4。当前主线是**可溶体系 4W53**，
> 膜线不阻塞任何在推的工作。本节内容的时间戳是 **2026-08-11**，之后没有人动过——
> 重启这条线之前先对着源码核一遍，别直接照着勾。
> 带电配体那一半另见本文 [PHY-03](#6-phy-03带电路线的条件性阻塞)。

> 2026-08-31 发布整理并入，原文件 `docs/status/memtodolist.md`（在 `Atenolol-rank11`，**不在本仓**）。
>
> ⚠️ 本仓库的 `docs/status/` 已于 2026-09-02 撤销（见 [README](README.md)《运行期发现往哪写》）。
> `memtodolist*.md` 在 `Atenolol-rank11`，**别在本仓里找**。


更新日期：2026-08-11（**C3 与 MEM-00h 已正式关闭（用户确认），进入 C4**。
C3-0~C3-4 全部跑过一轮；co-ion/ParameterOffset 归因诊断完成；C3 protocol v2
双层门重设计已实现；C2 的 C-seam switch 不一致已用"MEM-00h 双边归一化"
修复并在全部真实 GPU 数据上验证——A/B 100/100 + C/D 50/50，全部 150 帧一次
通过，C2 的 C-seam 力差回落到机器精度；`summary.json`/`mem00h_report.json`
两份 fail-closed 汇总产物已生成，均 `status=complete, passed=true`）  
状态：Phase B 工程实现基本完成；B5 已关闭。C1、C2、C3 已关闭；MEM-00h 已
关闭。当前进入 C4。

**已关闭事项的完整过程、失败证据和验收记录均已原文迁移到 `docs/status/memtodolist_archive.md`（在 `Atenolol-rank11`，**不在本仓**）。**


---

### 1. 当前做到哪里

已完成的工程能力不再逐项放在本清单中，完整证据见归档。当前状态摘要：

- B1：膜体系识别和 `MonteCarloMembraneBarostat` 已实现。
- B2：`charge_treatment` 配置和双计数 fail-closed 已实现。
- B3：PME co-alchemical charge-transfer Hamiltonian 已实现。
- B4：溶剂腿 reserved co-ion dummy builder 已实现。
- B5 已关闭（2026-08-09）：cache、resume、provenance 全套离线测试 0 failed，
  co-ion 隔离/缓存拒绝/resume 一致性逐项复核通过。
- 中性 Atenolol 膜体系 complex/solvent 双腿工程 smoke test 已跑通。
- C1 已关闭：Na/Cl 硬性验收通过；采用单 seed pilot，不补 seed；Ca 为已知统计限制且不阻塞。

当前主线：

```text
B5、C1、C2、C3、MEM-00h（已关闭）
    ↓
C4 带电膜双腿 smoke test（当前，尚未开始）
    ↓
C5 co-ion 位置/restraint 敏感性
    ↓
Phase D 生产资格
```

---

### 2. Phase C：当前验证

#### C4：带电膜 complex/solvent 双腿 smoke test

前置：B5、C1、C2、C3 全部通过。**2026-08-11：确认 C3 与 MEM-00h 正式关闭，
C4 已解锁。**

**C4 是接线 smoke test，不是生产自由能计算**——不追求收敛，不出最终
ΔG；全部产物必须标 `production_qualified=false`（第 6 步）。C2 的纯脂质
slab（无蛋白）不能代替这里的真实 receptor–ligand complex；C4 第一次真正
需要"膜 + 蛋白 + 带净电配体"这套完整组合。

用户指定的执行顺序（2026-08-11 登记，按顺序执行，不并行跳步）：

**受体/配体组合——阻塞第 1 步，待用户决定，本文档不擅自选择**（2026-08-11
现状普查，只读，未改任何文件）：

- **已有、可复用的**：`memtest/` 下有一个真实的 283 残基 GPCR 样受体
  （`Atenolol-rank1apo.pdb`/`Atenolol-rank1.pdb`，含 TM3 的 `DRY`、TM7 的
  `NPxxY` 保守基序，疑似热稳定化突变体，ICL3 可能被截短）已经嵌入真实
  POPC 膜（`memtest/step7_production.gro`：`PROA 1 / POPC 90 / Na+ 25 /
  Cl- 36 / TP3 9542 / Atenolol-rank11 1`，45354 原子），配上中性 Atenolol
  （`Atenolol-rank1.gjf` 的 QM 电荷计算用的是 `Charge=0`，即去质子化的
  仲胺；`memtest/Atenolol-rank11.itp` 41 个原子电荷加总 Σq≈0），
  `memtest/README_MEMTEST.md` 记录了这套中性体系已经跑通的完整
  complex/solvent 双腿工程 smoke test（膜恒压器、quality gate、诊断脚本
  全部现成）。`abfe_core.py`/`runabfe.py` 的 charge-transfer + 膜恒压器
  通用接线（`--only-complex-charging`、`--membrane-input-declaration`、
  co-ion dummy 插入）已经用这套中性体系验证过，从未在带电配体上跑过。
- **真正的冲突**：`docs/status/memtodolist_archive.md`（在 `Atenolol-rank11`，**不在本仓**）（2026-07-29）记录过一条决定——
  **"首个体系 = SERT（血清素转运体），配体默认净电荷 +1"**。但实际建出来
  并跑通的是上面这个 GPCR + 中性 Atenolol，跟当年那条决定不是同一个体系：
  SERT 从未真正建过膜体系（没有对应的 CHARMM-GUI 产物、没有嵌膜、没有跑过
  任何 smoke）。
- **配体电荷缺口，跟选哪个受体无关，两条路都要补**：仓库里没有任何带电
  （质子化、净 +1）的 Atenolol 参数——所有现成拓扑（根目录
  `Atenolol-rank1.itp`、`memtest/Atenolol-rank11.itp`）都是从
  `Charge=0` 的 QM 计算导出的中性形式。要走"配体带净电"这条路，不管配哪个
  受体，都需要重新做一次质子化仲胺的 QM 电荷推导（Gaussian）+ 重新生成
  GAFF 拓扑——不是挪文件就能解决的工作量。
- **受体身份记录缺口**：`memtest/membrane_input.json` 明确写着
  "未记录上游 PDB ID"、构象态"unspecified"——呼应 §A5"记录受体结构 ID、
  构象状态、突变、缺失残基和质子化态"这条从未打勾的要求；C4 定位是接线
  smoke（`production_qualified=false`），这个记录缺口是否必须先补齐、
  还是可以先如实标注"未知"往前走，也需要用户决定。

**用户 2026-08-11 明确表示：这个选择稍后告诉我，现在只要求把决策点和现状
写清楚——不要自己选受体/配体组合，也不要开始任何构建。**

1. **准备真实带电膜 complex，以及匹配的 solvent leg**
   - [ ] ligand 必须带净电荷（不是 C1/C2 用的中性探针或单原子简化）；
   - [ ] build 时显式插入 reserved neutral ion-shaped dummy；
   - [ ] 排除结构性离子、孔道离子、口袋/膜头基/疏水核中的候选（呼应
     §A5 已经列出但从未做过的排除清单）；
   - [ ] complex 与 solvent 两腿冻结**同一个** co-ion identity 和 restraint
     定义（不能两腿各自独立选一次）。
2. **零步静态预检**（不积分，只建 Context 查一次）
   - [ ] charging 全部 λ 态总电荷恒定；
   - [ ] `λ_coul=1`：ligand 满电、co-ion 中性；`λ_coul=0`：ligand 去电、
     co-ion fully charged；
   - [ ] Stage2 输入已经 baking 完成，System 里不存在活的 `lam_coul`
     GlobalParameter；
   - [ ] complex 用膜恒压器（`MonteCarloMembraneBarostat`），solvent 用
     各向同性恒压器；
   - [ ] handoff protocol/version 和 co-ion fingerprint 都已经进入
     cache identity。
3. **最短 GPU smoke**（不追求自由能收敛，只要能跑）
   - [ ] complex charging 能建 Context、积分、写 checkpoint；
   - [ ] complex Stage2 能接上 charging 端点（真正走一次 Stage2 handoff）；
   - [ ] solvent charging/Stage2 同样可运行；
   - [ ] 全程 energy/force finite；无 NaN、PME error、粒子逃逸或
     restraint runaway；
   - [ ] Stage2 全程 co-ion 保持 fully charged。
4. **相同命令立即 resume 第二次**
   - [ ] 命中相同 co-ion identity；
   - [ ] 已完成窗口被复用，不重跑；
   - [ ] 不重复插入 dummy/offset/restraint；
   - [ ] handoff/cache protocol 字段一致。
5. **复制一份 co-ion spec、故意篡改**（atom index / fingerprint /
   endpoint charge 任选一种）
   - [ ] 必须在建 Context **之前** fail closed；
   - [ ] 原始产物不能被这次篡改测试覆盖/污染。
6. **所有 C4 输出统一标注**
   ```json
   {"production_qualified": false}
   ```
   C4 只是接线 smoke，即使全部 PASS 也不能当生产结果用。

**当前最先要做的是第 1 步**：确定并预检真实带电膜 complex/solvent 输入。
§A5"目标膜输入"下的清单（受体结构 ID、构象状态、配体质子化态/形式电荷、
结构性离子排除等）到目前为止都还没做过，是这一步要补的作业，不是重复劳动。

#### C5：co-ion 位置与 restraint 敏感性

前置：C4 通过。

- [ ] 至少 3 个合法 bulk-water 位置和 1 个故意违规位置。
- [ ] restraint 基线：k=100、r0=0.5。
- [ ] 弱/宽：k=50、r0=0.7。
- [ ] 强/窄：k=200、r0=0.3。
- [ ] 每个合法组合跑 complex/solvent 两腿和至少 3 seeds。
- [ ] 检查 dummy 吸附、charged endpoint 水合、触壁比例和 restraint 能量。
- [ ] 净 `ΔΔG_bind` 同时满足 2σ 和 1 kcal/mol 门。
- [ ] 若两腿 restraint 自由能不抵消，给出显式修正或判定路线失败。

---

### 3. 膜输入与科学协议仍缺

#### A5：目标膜输入

- [ ] 准备并验证真正用于带电生产的已平衡膜输入。
- [ ] 记录受体结构 ID、构象状态、突变、缺失残基和质子化态。
- [ ] 记录配体质子化态、互变异构体、形式电荷和参数来源。
- [ ] 核对结构性 Na⁺/Cl⁻，从 co-ion 候选中显式排除。
- [ ] 排除蛋白孔道、结合口袋、膜头基层和疏水核中的 co-ion 候选。
- [ ] 核对蛋白插膜方向、配体 pose、结构水、辅因子和二硫键。
- [ ] 记录膜组成、上下叶组成、胆固醇比例、盐浓度和温度。

#### 热力学循环和 restraint 账目

- [ ] 写清 co-ion restraint 在 complex/solvent 两腿是否抵消。
- [ ] 若可用体积不同，推导并实现显式修正。
- [ ] charge-transfer 路线最终报告必须明确 `APBS/Rocklin = 0`。
- [ ] co-annihilation 只允许实验对照，禁止进入膜生产 preset。
- [ ] `shadow_ibs` 对带电配体明确 fail closed，或完整实现同一 co-ion 路线。

#### 膜生产协议

- [ ] 明确炼金生产阶段使用 NPT 还是 NVT。
- [ ] 若使用 NVT，记录固定盒矢量来自哪一帧。
- [ ] 明确时间步、约束和是否使用 HMR。
- [ ] 明确膜位置限制的分级释放方案。
- [ ] 记录结合位点是水相可及、界面、脂质暴露还是疏水深埋。
- [ ] 对脂质暴露/空腔填充做正反向或双初态迟滞验证。

---

### 4. 生产资格 Phase D

- [ ] D1：关闭 P1-19/P1-19b 的跨运行不确定度问题。
- [ ] D1：对 P1-22 的 Stage 2 帧选择和 σ 口径形成正式结论。
- [ ] D2：完成 Boresch 真实键拓扑和二面角更新门。
- [ ] D3：至少 3 个独立生产重复一致。
- [ ] D4：至少一个公开或可追溯膜受体 benchmark 通过。
- [ ] D5：完整 provenance、运行命令、环境、seed、输入 SHA256 和复现实验脚本。
- [ ] 膜质量门通过。
- [ ] overlap/ESS 和修正后的不确定度门通过。

---

### 5. Definition of Done

只有以下项目全部完成，才能声明支持生产级膜受体–配体 ABFE：

- [ ] C2–C5 全部通过（C1、C2、C3 已关闭并归档；C4 已解锁，C5 未开始）。
- [ ] co-ion 两腿显式存在、进入 PME、受控并进入全部缓存指纹。
- [ ] 全部 λ 总电荷恒定，且未重复应用 APBS/Rocklin。
- [ ] 膜恒压和平衡质量门通过。
- [ ] Boresch、co-ion restraint 和标准态修正闭环。
- [ ] 至少 3 个独立重复一致。
- [ ] 公开 benchmark 通过。
- [ ] 最终结果可审计、可恢复、可复现。

---

## 6. PHY-03（带电路线的条件性阻塞）

**P1，实验路线。**charge-transfer 的 tethered charge carrier 不能按当前论证严格跨腿抵消

- 位置：`abfe_core.py` 的 co-ion restraint 说明与表达式（约 1088–1117 行）；
  `ibs_engine.py::_create_co_alchemical_ion_restraint`（约 807–848 行）；
  `abfe_core.py::resolve_charge_treatment` 的 `closes_thermodynamic_cycle`（约 926–947 行）。
- 触发：带净电配体使用 `co_alchemical_charge_transfer`。
- 问题一（配分函数）：代码以“两腿同一锚点规则、同一 k/r0”推断 restraint 自由能严格
  抵消。实际受限 charge carrier 的配分函数包含
  `integral exp[-beta*(U_env(r)+U_rest(r-r_anchor))] dr`。complex 与纯水腿的
  `U_env`、排除体积、anchor 系综均不同；lambda=0 时 carrier 还带真实电荷并与环境
  相互作用，因此 restraint 与 carrier 溶剂化不能分离成一个两腿相同的常数。
- 问题二（barostat）：`dx0/dy0/dz0` 是冻结的笛卡尔 nm per-bond 参数。barostat
  缩放盒矢量和粒子坐标时，`d0` 不缩放；“井心随体系/盒一起缩放”的注释不成立，
  半各向异性/三斜 NPT 下尤其明显。
- 影响：decoupled complex/solvent 端点未必共享可严格消掉的 reservoir 状态，最终差值
  可能含 carrier 位置、盒大小、蛋白排除体积和 restraint 的非物理贡献。项目当前已经把
  charge-transfer 标为 `production_qualified=False`，这一边界必须保留；但同时写
  `closes_thermodynamic_cycle=True` 仍过度承诺。
- 要求：给出包含 carrier restraint/标准态/环境项的完整热力学循环推导；若不能证明解析
  抵消，就显式计算两腿 restraint/reservoir correction。参考位移需要采用真正随盒变化的
  分数坐标定义，或改成不依赖冻结笛卡尔井心且有解析标准态修正的相对约束。
- 验收：carrier 平移、anchor 选择、盒尺寸、各向异性缩放和 restraint 强度扫描后，修正后
  ΔG 在统计误差内不变；complex/solvent reservoir 端点有独立 free-energy closure test。
  C4/C5 未通过前不得把数值提升为生产结果。

---

## 往这里加条目的规则

1. **一条待办只在本文出现一次。** 专题文档讲"为什么"，本文讲"还欠什么、欠在哪一行"。
2. 关闭一条就**整段移进 `archive/`**，页首写清结论去了哪 —— 不要在本文留 `[x]` 尸体。
3. 新条目必须带**位置**（`文件:行号` 或函数名）和**判据**（怎样算做完）。
   写不出判据的不是待办，是想法，去 `design/`。
4. 标"不修"的要写明理由，并注明**理由别重新论证** —— 否则下一个人会花一天重查一遍。
5. 科学结论不进本文，进 [STATUS.md](STATUS.md)；协议/代码变更进 [CHANGELOG.md](CHANGELOG.md)。
