# 当前行动清单 —— P1（挡住在推的工作）

[项目入口](../README.md) · [文档导航](README.md) · [当前科学状态](STATUS.md) · [变更记录](CHANGELOG.md)

> **本文是本仓库唯一的待办入口。** 2026-09-16 按**优先级**拆成四份 ——
> 原来 1161 行里只有 3 条是 P1（09-16 测试对账后 +2 条 `COMBINE-01`/`REWIND-01`，共 5 条；
> `REWIND-01` 已于 09-17 关闭并归档 ⟹ **今天开着的是 4 条**），其余是取证、暂停线和判定不改，
> 挤在一起让人数不清到底欠几件事。

> **项目处于开发最末期**，不是在筹备一次首发：核心流水线、Stage-2 自治控制器、
> 闭式重训都已跑通。所以三份 TODO 里没有"要建什么新能力"，只有"哪一处还没收干净"。

| 文件 | 收什么 | 什么时候看 |
|---|---|---|
| **本文** `TODO.md` | **P1 · 挡住在推的工作** + 不许回退的约定 + 设计依据 | **每天** |
| [TODO_P2.md](TODO_P2.md) | **P2 · 该做，不挡路**：重构立项、两条待拍板、EXP-033、发布工程门 | 手上没 P1 的时候 |
| [TODO_P3.md](TODO_P3.md) | **P3 · 现在明确不做**（`AUDIT-01`~`07`、`BM-A`、膜线 62 框、`PHY-03`、押后的三条 `REL-*`） | 只在**想重新论证某条**时 —— 先确认它不在这里 |
| [archive/TODO_evidence_2026-09-16.md](archive/TODO_evidence_2026-09-16.md) | 已关闭条目的取证：重放表、预算实测表、28 个报错 run 的归因 | 要引用实测数字时 |

> **优先级写在编号后的方括号里**（`AUDIT-S2-03 [P1]`），一眼可 grep：`grep -rn '\[P1\]' docs/`。
>
> | | 判据 | 在哪 |
> |---|---|---|
> | **P1** | **挡住在推的工作** —— 不做就拿不到可引用结果 | 本文 |
> | **P2** | 该做，但不挡路（含「要维护者拍板但今天休眠」的） | [TODO_P2.md](TODO_P2.md) |
> | **P3** | **现在明确不做** —— 判定不改 / 已确认不是 bug / 暂停 / 押后到某个触发条件 | [TODO_P3.md](TODO_P3.md) |
>
> 一条待办仍然**只在一处**出现 —— 拆的是优先级，不是拷贝。

> ### 📌 「某条缺陷修没修」的权威是源码，不是这里的 `- [ ]`
>
> 2026-09-16 那次对账：§1 里 16 条未打勾的条目，**13 条描述的缺陷已经不在代码里了**
> —— 09-14/09-15 修掉的，没人回来打勾。危害不是漏修，是**重复排查**。
> ⟹ **关掉一条的动作是「回源码看缺陷在不在」，不是「看上次谁写了什么」**；
> 对账用的 `grep` 命令逐条写在 [archive/TODO_closed.md](archive/TODO_closed.md)。
>
> ⚠️ 对账只证明缺陷不在了，**不证明修得对**。这批改动的真机覆盖仍然只有 2 个 run
> （`AUDIT-S2-03`；「真机零验证」那个旧说法 09-16 已更正，别再照抄）。

> **已关闭的条目不留在这里**，整段移进 `archive/`：
>
> | 归档 | 内容 |
> |---|---|
> | [archive/TODO_closed.md](archive/TODO_closed.md) | `MIGRATE-01` / `PBC-01` / `XFAIL-01` / `XFAIL-02` / `CACHE-01` / `CFG-01` 六条已关闭缺陷的完整记录 |
> | [archive/TODO_closed.md](archive/TODO_closed.md) | `BOR-01`（Boresch 几何 minimum-image 口径）与 `S2-D`（控制器代码收拢）两条已关闭条目的完整记录 + 关闭验收。留下的规矩：**同一个不变量只能有一份实现**；**收拢的判据是“写盘的留 pipeline”，不是“塞进一个文件”** |
> | [archive/TODO_closed.md](archive/TODO_closed.md) | `LR-01`（残差开关进 stage 指纹的作用域）一条已关闭条目 + 只读核实证据。留下的规矩：**一个开关有几条进指纹的路径就得收窄几条**；且**顶层 run 指纹与 stage 指纹口径不同**，顶层无条件进是对的，别跟着"统一" |
> | [archive/TODO_closed.md](archive/TODO_closed.md) | **一次对账，不是一次修复**：`DECORR-01` / `AUDIT-S2-01` / `BUD-01`~`05` / `BUD-07` / `CTL-11`~`19` / `BM-01`~`03`+`05` / `DATA-01` 共 20 余条 —— 其中 13 条是**挂在本文里、但缺陷早已不在代码里**的。含逐条证据行号与对账用的 `grep`。留下的规矩：**「某条缺陷修没修」的权威是源码，`- [ ]` 和状态列都是会过期的缓存** |
> | [archive/TODO_closed.md](archive/TODO_closed.md) | **当天真做掉的 6 条**（与 09-16 那次「对账」不同）：`REWIND-01`（P1）+ `TEST-01`/`LR-03`/`LR-04`/`REL-08`/`REL-09`（P2）。留下的规矩：**死动作要接在各自兜底之前、不是在链尾加一条分支**；**一个父窗只切一层**（不设深度上限就是新的无限循环，且不许放宽）；**测试选择口径的判据是收集数不是 grep 文件** |
> | [archive/TODO_2026-08-06_unreconciled.md](archive/TODO_2026-08-06_unreconciled.md) | 2026-08-06 的主表（1350 行，`ATT-xx`/`MEM-xx`/`P0-9~13` 编号）。**那里面的 `- [ ]` 只表示"当时未完成"**，要人逐条对账才能重新变成待办 |
>
> ⚠️ 不要按编号跨文档机械对账：`P1-19` 在 08-29 那份交接里是"v4 charging 接缝内静电失配"（已修），
> 在 08-06 主表里是"per-window σ 系统性低估 2–4 倍"（未完成）。**同名不同义。**

---

## 1. Stage-2 自治闭环

> ### ✅ 验收口径已首次达成：2026-09-12 17:22
>
> 自治控制器独立走完整条 Stage-2（六窗全部 `ANALYSIS_ELIGIBLE` → `DONE`），
> 体系 `cyclod_ligand2/rep1`（环糊精主客体，可溶），
> ΔG_bind = **−3.48 ± 0.47 kcal/mol** vs 实验 −4.04，差 **1.19σ**。
>
> **设计、实证与十条陷阱全部在
> [STAGE2_CONTROLLER_DESIGN_2026-09-12.md](archive/STAGE2_CONTROLLER_DESIGN_2026-09-12.md)
> —— 接手先读那份，别重推。** 本节只登记它 §7 列的缺口。
>
> ⚠️ 一次跑通 ≠ 通用。这是**单体系单 run**，没有独立重复。

> ### ✅ 已关闭的 8 条整段移进 `archive/`（2026-09-18 搬）
> 
> | 条目 | 一句话 | 去处 |
> |---|---|---|
> | `S2-N` | 块账写侧 4 个动作、读侧只拦 1 个 ⟹ 重标定/探针/临时生产扣配额却从不过闸 | [archive/TODO_closed.md](archive/TODO_closed.md) |
> | `S2-O` | 退出前的全路径 ANALYZE 在有 stale 证据时被整个跳过 ⟹ 整跑零产出 + 自锁 | [archive/TODO_closed.md](archive/TODO_closed.md) |
> | `S2-L` | 补帧的块数硬上限被「换布局」重置 ⟹ 补帧实际没有有效上限 | [archive/TODO_closed.md](archive/TODO_closed.md) |
> | `S2-M` | 「布局过期 + 冻结验证批次打满」= 死锁 ⟹ 整跑终止 | [archive/TODO_closed.md](archive/TODO_closed.md) |
> | `S2-K` | `ANALYZE` 在有过期证据的盘面上炸穿流水线 —— 判据侧知道、执行器侧不看 | [archive/TODO_closed.md](archive/TODO_closed.md) |
> | `S2-J` | model B 的收尾动作「末窗一分为二」在自治控制器下整个不存在 | [archive/TODO_closed.md](archive/TODO_closed.md) |
> | `S2-H` | 「总是改变盘面」的动作对停滞保护与 no-op 台账结构性免疫 | [archive/TODO_closed.md](archive/TODO_closed.md) |
> | `S2-G` | 退役判定看不见「从没采过的窗口」⟹ 整跑崩溃 | [archive/TODO_closed.md](archive/TODO_closed.md) |
> 
> ⚠️ **`S2-N`/`S2-O` 真机零复验** —— 修复落盘 09-18 08:30–08:36，`cyclod_ligand1_outer`
> rep2/rep3 是 09-17 起的吃不到；看 09-18 16:11 之后起的 `p38_ligand1` 那批。

- [ ] **AUDIT-S2-03 [P1] 控制器已有真机，但**旧验收判据（`DONE`）单跑结构性不可达，
  2026-09-18 已换判据**；新判据下的闭环仍未在修复后的代码上拿到。**
  📌 **2026-09-16 更正：本条原文「控制器真机零验证」已不成立。** 全量 benchmark
  （13 体系 ×3 = 39 rep）真上过 GPU，9 个跑出 `final_binding_results.json`。
  但**其中只有 2 个跑的是修复后的代码**（`p38_ligand1/rep2` 启动 09-16 06:29:45、
  `rep3` 07:34:59；判「跑哪版代码」的口径见 `BM-B` 取证①），其余 7 个都是旧码启动。
  **那 2 个的终态是 `ANALYSIS_COMPLETE_PRECISION_UNMEASURED` + `NO_FEASIBLE_ACTION`，
  不是 `DONE`。**（两者都在 `TERMINAL_EXITS` 里，是**合法终态、不是 bug** ——
  但设计 §7 要的验收是「六窗全部 `ANALYSIS_ELIGIBLE` → `DONE`」，那个还没在修复后的代码上出现过。）
  ⚠️ 2026-09-14 那约 **30 处**修复 + 09-16 的 `BM-01`~`03`/`05`，真机覆盖仍然只有上面这 2 个 run。

  📌📌 **2026-09-18 再次更正：旧判据（`decide()` 给 `DONE` 且 `execution_status=COMPLETE`）
  在单跑里结构性不可达，已作废。** 逐条核实（与 `abfe-benchmark-cf` 会话对过 36 份台账）：
  · `DONE` ⟸ `precision_status == MEETS_CROSS_REPEAT_TARGET`
    （`abfe_preoptimizer.py:6038` / `:7805`），而 `MIN_INDEPENDENT_REPEATS = 3`
    （`abfe_core.py:2672`）。
  · Stage-2 IBS **真正计算**该字段的三条生产路径全传空列表
    （`ibs_engine.py:23632` / `:23789` / `:24158` 的 `cross_repeat_precision([])`），
    且仓内**没有跨-run 聚合器**回写它 ⟹ 单跑恒 `UNMEASURED`。
    ⚠️ **「全仓只有三个产出点」的说法不准确，别照抄** —— `"precision_status"` 全仓
    有 22 处（`grep -n '"precision_status"' *.py`），但它们分三类，没有一类能产出
    `MEETS`：① **计算**的就上面三处，全传空列表；② **硬写 `UNMEASURED`** 四处
    （`abfe_pipeline.py:2537` / `:6969` / `:10680` / `:10755`，外加
    `ibs_engine.py:23788` / `:24157`）；③ 其余是**转抄/汇总**
    （`abfe_pipeline.py:6658` / `:7830` / `:7858` / `:14885`、
    `abfe_preoptimizer.py:3388` / `:4450`、`runabfe.py:7125` / `:7137` / `:8804`、
    `ibs_engine.py:14505` / `:23700`），只搬不算，且 `runabfe.py:7125` 的两腿合并
    对缺失值兜底成 `"UNMEASURED"`。⟹ 结论不变，而且比原说法更强。
  · `execution_status=COMPLETE` 不是第二道条件，只是 `action == "DONE"` 的派生值
    （`abfe_preoptimizer.py:5853`）。
  · `--allow-untrusted-stage-results` 救不了 —— 它换出口的那两行写在
    **precision 已达标的 then 分支里面**（`:6045` / `:7813`）。
  · 单跑的合法终态本来就存在：`action=NO_ACTION` + `exit=ANALYSIS_COMPLETE_PRECISION_UNMEASURED`
    （`:6047` / `:7816`，`EXIT_SPECS` 里 `terminal=True`，见 `:2782`）。
  · 管线侧 2026-09-15 已经换过判据：非 `DONE` 出口**不再** raise，只记日志
    （`abfe_pipeline.py:8599`）；真正拦的是分析不完整 / 窗口子集 / λ 覆盖不全 / 非有限结果。
  · 「能不能当已验收结果发布」另有专门闸 `publishable_as_accepted_result`
    （`abfe_pipeline.py:7870`），`MEETS` 的要求在**那里**，控制器层不必再要一遍。
  ⚠️ 真机 36 份台账：`NO_FEASIBLE_ACTION` 20 / `ANALYSIS_COMPLETE_PRECISION_UNMEASURED` 16 /
  **`DONE` 0**；`precision_status=UNMEASURED` 25/25；**没有任何 run 出不来**（全部有
  `outcome.exit`）。上层语义也是接对的：未验收走 WARNING + 零退出码 + 产物齐全。
  ⚠️ 即使凑够 3 个独立重复也大概率拿不到 `MEETS`：按臂实测跨 rep 散布是报出 σ 的
  **3.7× 中位**（10 个臂无一小于 1；三个 n=3 的臂 3.3/3.0/1.9×），而门槛是
  `CROSS_REPEAT_MAX_STDDEV_KCAL_PER_MOL = 1.0`（`abfe_core.py:2671`）。
  把判据改写成「拿到 `MEETS`」只是把一个不可能事件换成另一个。

  **新判据（2026-09-18）—— 任一体系在当前代码上跑出：**
  1. `analysis_status = ANALYSIS_COMPLETE`（两条腿都要）；
  2. `action = NO_ACTION`；
  3. `autonomous_outcome.exit = ANALYSIS_COMPLETE_PRECISION_UNMEASURED`；
  4. `missing_windows` / `skipped_windows` / `skipped_sampling_units` /
     `out_of_range_windows` **全空**；
  5. 不带 `stage_scope = window_subset_no_stage_verdict`，也不带
     `subset_partial_sum_not_delta_G = true`；
  6. `coverage_diagnostics` 覆盖**全部** λ 态且 `dropped_window_indices` 为空。

  `DONE` 需要 ≥3 个独立重复，属 **BM 跨-run 验收**，不属控制器单跑验收 ⟹ 归 `BM-B`。
  **在拿到一次满足上面六条的闭环之前**：不得声称「Stage-2 自治闭环可用」；
  不得删除任何旧修复路径（`S2-E`）；控制器产出的任何 ΔG 都不是可引用结果。

  ⚠️ 本条上面那句「设计 §7 要的验收是『六窗全部 `ANALYSIS_ELIGIBLE` → `DONE`』」
  指的是 2026-09-12 那次真机 —— 那是 **09-15 收紧 `DONE` 判据之前**的事，不能拿它
  当现在的验收口径。设计文档 `STAGE2_CONTROLLER_DESIGN_2026-09-12.md:237` 那句
  「`DONE` 在结构上不可达」说的是**另一件事**（中间结果不落盘的死锁，已修），别混。

- [ ] **BM-B [P1] 全量 benchmark（13 体系 ×3 = 39 rep）：30 个未完成 run 里，最后启动在修复之后的是 0 个。**
  **先取证，再重跑 —— 但有 6 个不能直接重跑，见下。**
  来源：`abfe-benchmark-31` 会话 2026-09-16 的两轮汇总 + 本会话复核。
  阶段完成度 equilibration 36/39、attachment 36/39、decharging 35/39、**vanishing 9/39**。

  ### ⚠️⚠️ 三条取证口径，都是这一轮**栽过之后**才定下来的

  **① 判「跑的是哪版代码」只能用「最后一次启动横幅」（`已合并配置文件`）的时刻，
  不许用 `launch.log` 的 mtime。** Python 在进程启动时加载源码，所以决定版本的是
  **启动时刻**；mtime 是最后一次写日志。两者能差几小时，实测：

  | rep | 最后启动 | mtime | 差 |
  |---|---|---|---|
  | `p38_ligand1/rep1` | 09-16 **04:09:10** | 09-16 06:29 | 2h20m |
  | `brd4_ligand2/rep1` | 09-15 **15:23:38** | 09-16 05:10 | **13h47m** |

  📌 **本条第一版就是这么写错的**：用 mtime 判成「1 个 run 跑过修复后的代码」，
  实际是 **0 个**。修正后结论**方向不变而且更干净** —— 不需要再拿 `BM-A` 第 4 条
  去解释那个"例外"，例外根本不存在。

  **② `launch.log` 跨重启追加，「最后一条异常」可能来自早就被修好的那次启动。**
  实测启动次数：`cyclod_ligand2/rep1` **25 次**、`brd4_ligand2/rep1` 9 次。
  要看**最后一次启动横幅之后**的异常。按这条重扫 30 个：24 个分类不变，**6 个作废**。

  **③ 判「这个 run 跑完没有」时，`final_binding_results.json` 与 `pipeline_state.json`
  对**被人工归档过**的 run **两个都会漏** —— 前者漏在文件被搬走，后者漏在 state 被回退。**
  产物搬到的是 `<体系>/_backup_<rep>_<stage>_<时间戳>/` 与 `_trash_*`（**与 rep 目录同级，
  不在 rep 目录里** —— `ls */rep*/…` 正是这么漏掉的）。
  全仓 `final_binding_results.json`：现役 rep 目录下 **9 份**，含归档共 **20 份**。

  ### 30 个未完成 run 的归属

  | 对方分类 | 归属 |
  |---|---|
  | [11] 窗口/路径布局不一致 | `BM-01`/`02`/`03`（已修）+ `DATA-02`（`cyclod_ligand3/rep1,rep3` 非法布局已落盘，**待拍板**，重跑也起不来） |
  | [7] `ANALYSIS_INCOMPLETE` | `BM-04`（`cmet_ligand1/rep1`、`p38_ligand2/rep1`，**已拍板：重建**）+ `DATA-03`（`cmet_ligand1/rep2`、`cmet_ligand2/rep1`、`p38_ligand2/rep3`，**已拍板：21/4 续跑**）+ `BM-A` 真采样不足 |
  | [5] `TypeError: int() … not 'NoneType'` | **🔴 整桶签名作废，见下面「6 个半状态」** |
  | [3] `cyclod_ligand1_outer/rep1-3` 窗口 0 非有限坐标 | **`LR-06`**，rep 号逐个吻合。**修法已定但没动手 ⟹ 现在重跑还会照样死** |
  | [3] `thrombin_ligand1/rep1-3` co-ion 数量契约 | **已修**：日志 **02:09** vs `runabfe.py` **03:29** ⟹ 日志早 80 分钟，**重跑即可**（本条唯一可以立刻重跑的一批） |
  | [1] `cmet_ligand1/rep3` attachment 漂移 −0.7592 | `BM-A`，**不是 bug**（判据③下末条异常不变） |

  ### 🔴 6 个 cyclod rep 处于「产物被搬走 + state 回退」的半状态 —— 别直接重跑

  `cyclod_ligand1/rep1,rep2,rep3` + `cyclod_ligand2/rep1,rep2,rep3`：
  **最后一次启动之后一条异常都没有**（实测 6/6 异常数 = 0）。
  它们被归进 `[5] TypeError` 那一桶的 6 条 Traceback **全来自更早的启动**。

  实测 `cyclod_ligand2/rep1`：最后一次启动（09-15 17:51:53）**跑完了**，
  18:35:20 打出 `ΔG_bind = −0.61 ± 0.55 kcal/mol` 并保存了 `final_binding_results.json`；
  但现役目录里**没有**那份文件，`pipeline_state.json` 只剩 3 个 stage
  （`equilibration` / `sampling_dual_attachment` / `sampling_dual_decharging`）——
  20:15 那次人工归档把产物搬进了 `cyclod_ligand2/_backup_rep1_vdw_20260915-2015/`
  （含 `final_binding_results.json`、`vanishing`、`vanishing_2/3`、
  `checkpoints_pipeline_state.json.bak`），现役 state 被回退。

  ⚠️ **所以「重跑即可」对这 6 个是错的** —— 本条第一版据那个作废签名写过
  「`BM-05`② 已修 ⟹ 重跑即可」，**撤回**。直接重跑很可能只是把 20:15 那次归档再做一遍。
  **判据：先查清这 6 个现役目录为什么处在半状态、那次归档想达成什么，再决定重跑还是恢复。**
  （`BM-05`② 本身**仍然是已修的** —— 撤回的是"这 6 个 run 死于该 bug"这个取证，不是那条修复。）

  ### ✅ 修复生效的正面证据（比"旧码跑的所以不算数"强）

  | rep | 最后启动 | 结果 |
  |---|---|---|
  | `p38_ligand1/rep2` | 09-16 **06:29:45** | 07:35 出结果 |
  | `p38_ligand1/rep3` | 09-16 **07:34:59** | 08:40 出结果 |
  | `p38_ligand1/rep1` | 09-16 04:09:10（**旧码**） | 卡住 |

  **同一个体系、同一批输入，新码两个跑完、旧码那个卡住。** 这是 `BM-01`~`03`/`05`
  目前唯一的真机正面证据（其余 7 个跑完的都是旧码启动：`brd4_ligand1/rep2`、
  `brd4_ligand2/rep1-3`、`cmet_ligand2/rep3`、`jnk1_ligand1/rep1`、`p38_ligand2/rep2`）。
  ⚠️ `brd4_ligand2/rep1-3` 的**出结果**时间（05:10/05:45/06:09）在修复之后，但**启动**在
  09-15 下午 —— 只看产物 mtime 会误判成新码。

  ### 本次新登记的两条（都不新开条目）

  1. **`输出目录已被另一次运行独占（目录锁）` 历史出现 4 次，全部出自 `cyclod_ligand1`** ——
     `REL-09` 第一次拿到真机证据，已记到那条下面（该条 09-17 关闭，整段进
     [archive/TODO_closed.md](archive/TODO_closed.md)）。
  2. **`thrombin_ligand2` 连 run 目录都没有**（同为 +1 e 配体）。**输入是齐的**
     （`systems.csv`：`status=ok`、`q_lig_e=+1.000`、43764 原子）—— 就是**一次都没启动过**。

  ### 判据

  在**当前代码**上重跑，按上面①②③三条口径重新分类。**重跑之后仍然失败的才是新信息。**
  可以立刻重跑的只有 `thrombin_ligand1/rep1-3`；
  `LR-06` 那 3 个要等插件修完；`DATA-02`/`DATA-03` 那 5 个等拍板；
  6 个 cyclod 先取证。

  📌 科学结论（`results.csv` 11 行 / 6 体系，ME −4.73、MAE 5.91、RMSE 7.35 kcal/mol）
  **不进本文**（规则 5）⟹ 去 [STATUS.md](STATUS.md)。登记时必须写明：
  **11 行的时间戳全部早于 2026-09-16T05:03，即 11/11 出自修复前的代码**（来源列：归档 8 / 现役 3）；
  且 `results.csv` 的 11 行与「9 个现役 `final_binding_results.json`」是**两个不同的集合**
  （前者含归档产物与 `evidence` 目录），别混着数。

- [ ] **COMBINE-01 [P1] `combine_ibs_and_independent_endpoint` 不产出顶层 `analysis_status` ⟹ `--analyze-only` 恢复 vanishing 腿必然抛错（2026-09-16 由测试对账挖出，真机未复现）**

  **症状**：`runabfe._analyze_dual_leg_artifacts` 对 vdw 腿抛
  `vanishing 结果缺少 analysis_status —— 这是 2026-09-15 删除 converged 之前的老产物`。

  **因果链**（三处都已核实）：
  1. 2026-09-15 `converged`→`analysis_status` 改造时，`ibs_engine.py` 约 **14469**
     的注释把 `combine_ibs_and_independent_endpoint` **显式排除在改动范围外**
     （「本函数自己的 `converged` 输出键属于 abfe_pipeline 的契约，不在本次改动范围内」）。
     它的返回 dict 只把 `analysis_status` 埋进 `ibs_segment`，**顶层没有**。
  2. `runabfe.py` 约 **5096**：vdw 腿**强制**走这个拼接 —— 拿不到 endpoint artifact
     直接 `raise「回退分析拒绝用纯 IBS 窗口产出 vanishing 主值」`，没有绕行分支。
  3. 拼完立刻 `_assert_stage_result_sane` ⟹ 顶层缺键 ⟹ fail-closed 抛错。
     同一个洞在生产侧也在：`abfe_pipeline.py` 约 **6618** 用拼接结果**整个替换**
     `stage_result`，把 6590 处刚设好的顶层 `analysis_status` 覆盖掉
     （那条只在 `stage2_independent_endpoint=True` 时触发，默认关 ⟹ 严重性低一档）。

  **为什么不是顺手补一行 passthrough**：合并段的 `analysis_status` 要由两段的**硬不变量**
  合成，而端点段自己的 `converged` 里**混着阈值门**（`min_overlap` 那一类），原样抄过去
  就等于把已退役的拟合阈值重新塞回接受判据 —— 正是 09-15 那次改造要拆开的东西。
  ⚠️ **需要维护者拍板**：端点段的完整性用哪个量表达。

  **判据**：`tests/test_issues_67_27_75_83.py` 的 5 条 +
  `tests/test_independent_endpoint_sampling.py::test_combined_result_passes_the_pipeline_stage_gate`
  共 6 条 `strict-xfail` 变红（修好即自动变红），届时删掉那些标记。

### 不许回退的约定

- **一个 stage 只许有一个控制器。** 不是风格偏好 —— 两套机制各判各的会直接打架。
- **真终态以 `abfe_preoptimizer.py::TERMINAL_EXITS` 为准，今天是 9 个**（2026-09-16 实测）：
  `DONE` / `DONE_UNTRUSTED` / `GLOBAL_BUDGET_EXHAUSTED` / `NO_FEASIBLE_ACTION` /
  `HALT_INVALID_INPUT` / `HALT_EVIDENCE_CONTRADICTS_DONE` /
  `HALT_LAMBDA_BUDGET_INSUFFICIENT` / `HALT_FRAMES_ADMISSION_CAP` /
  `ANALYSIS_COMPLETE_PRECISION_UNMEASURED`。
  ⚠️ **本条原文写的是「只有三个真终态：`DONE`/`GLOBAL_BUDGET_EXHAUSTED`/`NO_FEASIBLE_ACTION`」——
  那是旧代码的清单，2026-09-16 核实已扩到 9 个。** 危险在于本条属于「不许回退的约定」，
  是拿来**否决别人改动**的：照旧清单去审，会把 6 个合法终态当成违规打回。
  ⟹ **不要在本文维护第二份终态清单**（那正是本仓最贵的复发模式）。要查就 `grep TERMINAL_EXITS`。
  **约定的实质不变，是下面这句** ——
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

- **局部动作只许载它要动的窗口** —— loader 的 fail-closed 布局校验是对的，错的永远是
  「在插 λ 的合法中间态里去载它不需要的窗口」。`INSERT_LAMBDA` 明写「本动作不采样，
  受影响窗口的重采由下一轮逐块发 `RUN_PRODUCTION`」⟹ **从插 λ 到下游重采完成之间，
  下游产物描述的是上一套布局，这是设计内的合法中间态，不是要消灭的状态。**
- **停滞保护的降级必须走 `decide()` 的守卫**，不许直接改写 `act`。
- **可行性与可落性是两问**（「该不该发」/「发了能不能落」），插 λ 两侧都要判 ——
  只判一侧就会把非法布局写进版本链。
- **算对之前，沉没成本一律不计。** 先问「这条路径算得对吗」，再问「重算要多少 GPU」，
  顺序不能反 —— 一个还没被证明算得对的结果，它的缓存没有保护价值。
- **`abfe_config.json` 不是「仓库默认配置」，只是一份样例。** 判某个键在某次运行里是
  什么值，只能看该 run 的 `run_provenance.json`。

### 设计依据（live，不是待办）
- **旧修复机制只许「关掉」，不许「删掉」**（原 `S2-E`）—— path_evolution 修复分支 /
  production rescue / rescue 后重标定，**等自治这条再跑通几次再删**。这不是待办，
  是一条禁令：解除它的条件写在 `AUDIT-S2-03` 的判据里。

- [STAGE2_CONTROLLER_DESIGN_2026-09-12.md](archive/STAGE2_CONTROLLER_DESIGN_2026-09-12.md)
  —— **权威**：设计 + 实证 + 十条真机咬过的陷阱。最贵的一条是
  **「同一个不变量的 N 份实现」**（λ 身份有四份）。
- [AUTONOMOUS_STAGE2_LOOP_SPEC_2026-09-11.md](archive/AUTONOMOUS_STAGE2_LOOP_SPEC_2026-09-11.md)
  —— 老板定的核心设计指标（唯一验收指标）。
- [STAGE2_AUTONOMOUS_LOOP_STATUS_2026-09-11.md](archive/STAGE2_AUTONOMOUS_LOOP_STATUS_2026-09-11.md)
  —— 逐个 bug 的修复流水账（历史，不是待办）。
- [PLAN_PATH_REPAIR_2026-09-11.md](archive/PLAN_PATH_REPAIR_2026-09-11.md) —— 设计要求的来源，
  分支编号（5a/5b/6）出自这里；代码里 10 余处引它作设计依据。
- [STAGE2_THREE_AXES_COUPLING_2026-09-11.md](archive/STAGE2_THREE_AXES_COUPLING_2026-09-11.md)
  —— 分窗口/分 λ/分采样量三轴当前怎么耦合的。
- [STAGE2_ROOT_CAUSE_2026-08-28.md](archive/STAGE2_ROOT_CAUSE_2026-08-28.md) —— **所有收敛门
  对单轨迹重加权这个失效模式是瞎的**：实测门全绿时 ΔG 错 42 kJ/mol。别拿质量门当验收。

---

---

## 2. local-residual / EXP-033

> 本节只留 P1 的 `LR-06`。`EXP-033-P1-GPU` / `EXP-033-P2` / `LR-02`~`05` 是 P2，
> 在 [TODO_P2.md](TODO_P2.md) §3。
> ⚠️ 下面正文里的 `P1`/`P3` 是 **EXP-033 自己的阶段编号**，不是优先级；
> 优先级只看方括号 `[P1]`。

背景：`B_φ` **不是配体的物理模型**，是提高 λ 态混合的**采样增强项**；按体系在线学的只有 `f_k`。
方案见 [EXP-033_LOCAL_RESIDUAL_REFIT_2026-09-10.md](archive/EXP-033_LOCAL_RESIDUAL_REFIT_2026-09-10.md)，
P1 落地见 [EXP-033_P1_LANDED_2026-09-12.md](archive/EXP-033_P1_LANDED_2026-09-12.md)。
**`LR-01`（残差开关的指纹作用域）2026-09-13 核实已修，整段进
[archive/TODO_closed.md](archive/TODO_closed.md)，别再列。**
**P3（跨配体通用权重）已于 2026-09-12 由用户拍板划掉，前提没了，别重新提。**

- [ ] **LR-06 [P1] 预热鬼影期 × 插件 0.1 Å 硬门 ⟹ outer 臂在窗口 0 必死。**
  **🟢 2026-09-16：方案 A 已落地并真机验证。2026-09-17：第 6 步真机通过。**
  **⏸️ 剩下的第 5、7 步 2026-09-17 由维护者判为「不着急，等控制器修完再动」** —— 第 7 步换体系复跑会把控制器当前那批未收口的改动一起卷进去，跑了也说明不了插件；第 5 步删脚手架要在有真机复跑兜底时才敢动。

  🔺 **2026-09-18 更新：第 5 步的押后条件（「有真机复跑兜底」）已经满足。**
  `cyclod_ligand1_outer` rep2/rep3 **带着那两个猴子补丁跑完全程**（详见第 6 步）。
  第 7 步（换体系）**仍然没做** —— rep2/rep3 是同一个体系，不是换体系。

  ### 已做完（代码 + 真机）

  | 步 | 状态 |
  |---|---|
  | 1. 4 处硬门改夹取 | ✅ `CudaLocalManyBodyResidualKernels.cpp` **能量侧 2 处夹取**（`rAngstrom = max(r, floor)`，这条边照样计入）+ **力侧 2 处跳过**（下限内 dE/dr ≡ 0，`invRM = 1/r` 因此根本不求值）。两半语义不同，不是同一句抄四遍 |
  | 1b. **Reference 侧同步** | ✅ **原条目漏了这一步**：`g1_math_core.h` 的 `enumerateEdges` 也有同一道门（抛 `MathError`），而 Reference 平台经 `ReferenceLocalManyBodyResidualKernels.h` 用它 ⟹ **只改 CUDA 就是「同一个不变量两份实现」**，G1/G2 钉的正是两者一致。已改成同一套语义并复用同一个常量（`#include "r1_model_layout.h"`，不许再抄一份数字） |
  | 2. 重编 | ✅ `g0_build.sh` 三个 `.so` 全通过 |
  | 3. 回归 | ✅ `exp028_run_regression_suite.sh` **ALL TESTS PASS**（5 个原生 harness）。⚠️ **G1/G2/G3 三个一致性 harness 没跑** —— 它们要 `yyjson`，本机 `$CONDA_PREFIX/include/yyjson.h` 缺失，套件按设计跳过。**改动最需要它们，这是本次验证的已知缺口**，下面用真机 A/B 代偿 |
  | 4. 同步两处 sha256 | ✅ `local_residual/openmm_plugin.py::KNOWN_PLUGIN_SOURCE_SHA256` + `resources/…/manifest.json` 的 `plugin.source_sha256` → `5c74365dffaa…` |
  | 5. 删临时脚手架 | 🔓 **2026-09-18 解除押后**（押后理由是「要有真机复跑兜底」，第 6 步已给出两个完整 run）。两个 `Context` 级补丁**仍在**，但代码里那条「这套东西还没经过大规模验证」（`em_no_residual.py:138`）的注释已经**过期**。⚠️ 两半代价不同，别当一件事做：`patched_set_parameter` 的 9 处调用点全在 `ibs_engine.py`、全是 `context.setParameter(f"{self.prefix}_bias_scale", X)` 同一形状（6786 / 15912 / 15934 / 15953 / 15968 / 16115 / 16152 / 16196 / 19897）⟹ 一个 helper 收掉，顺带消掉「同一不变量 9 份实现」的形状；`patched_get_state` **故意是全局的** —— 它拦的是鬼影期内进程里**任何**不带 `groups` 的 `getState`，不只是 `step_guard.py:104/186` 那两处，改成逐点显式传参就是补丁注释自己写的「逐个去包必然漏」⟹ **建议保留，并把「为什么不合并」写进注释** |
  | 6. `cyclod_ligand1_outer` 复跑 | ✅ **2026-09-17 真机通过**。同一份 `runs/cyclod_ligand1_outer/rep1/launch.log` 里两次启动直接对照（判版本用**启动横幅**不用 mtime，`BM-B` 取证①）：**09-16 02:46（夹取前）** 行 303–357 窗口 0 热化 `fail-closed … pair distance below 0.1 A`，步长降到 0.008 fs 仍炸 → `RuntimeError`；**09-17 01:22（夹取后）** 行 358–1251 `fail-closed` **0 次**，窗口 0–4 全部跑出能量，走到 `[自治 15/40]`，02:29 终止于**人为 SIGTERM**（`TerminationRequested`），不是插件。⟹ 「outer 臂在窗口 0 必死」这个失效模式已消除。<br><br>🔺 **2026-09-18 追加（同僚会话 `abfe-benchmark-cf` 取证）：不只是「不死了」，是跑完了。** `rep2`/`rep3` **全程带着第 5 步那两个猴子补丁**，跑出完整结果：<br>· `rep2` ΔG_bind **-6.28 ± 0.48** kcal/mol，`untrusted=False`（complex 71.20 / solvent 44.94 / Boresch -37.02 kJ/mol），控制器 19 轮，末轮 `NO_ACTION`<br>· `rep3` ΔG_bind **-2.00 ± 0.52** kcal/mol，`untrusted=False`（complex 56.94 / solvent 48.57 / Boresch -32.24 kJ/mol），控制器 14 轮，末轮 `NO_ACTION`<br>· 两 rep 的 complex/solvent **两条腿 `analysis_status` 均为 `ANALYSIS_COMPLETE`**，`precision_status` 均为 `UNMEASURED`，`outcome.exit` 均为 `ANALYSIS_COMPLETE_PRECISION_UNMEASURED`<br>· 版本（按启动横幅，`BM-B` 取证①）：**2026-09-17 14:43:12-04:00** / **15:54:47-04:00**，`abfe_ibs_commit=9a18ded`，RTX 5080 ⟹ 落在「09-17 那批控制器改动之后、09-18 的 `S2-N`/`S2-O` 之前」<br>· `rep1` 无结果，崩在 `ExistingEnsembleRequiresRescueAudit`（IBS state score identity mismatch）—— **这是闸在正常工作，别放宽**：逐 rep 闭式重训 ⟹ `sampling_score_sha256` 逐 rep 不同 ⟹ 拒绝复用别的 rep 的 ensemble；要跑得给它自己的干净 checkpoint 目录<br><br>⚠️⚠️ **`runs/cyclod_ligand1_outer` 与 `runs/cyclod_ligand1` 两臂不能做 A/B。** 两个 config 逐键只差两行（`output` 与 `outer_lambda_local_residual_ibs`），但冻结 manifest 只覆盖 Atenolol、本配体 27 原子对不上指纹 ⟹ 走 EXP-033 P1 的**自动闭式重训**，**三个 rep 各自拟合各自的 B_φ**。所以 outer 均值 -4.14 vs 基线均值 -5.36 这个差**不构成 outer 臂的效果证据**。真要 A/B 得按 `docs/RETRAIN_LOCAL_RESIDUAL.md` 手工冻结一份权重给两臂共用 |
  | 7. 换疏水口袋体系再跑 | ⏸️ **仍然押后，2026-09-18 复核后不变**。`runs/` 下仍然只有 `cyclod_ligand1_outer` 一个 outer 目录，jnk1/p38/brd4 一个都没跑过。第 6 步新增的 rep2/rep3 **是同一个体系**，补的是「补丁能跑完」，不是「换了体系也成立」⟹ 这一步的缺口原样还在 |

  ### 真机 A/B（RTX 2080 Ti，mixed precision，改动前后两份独立编译的 `.so`、两个独立进程）

  | 全局最小 lig–env 距离 | 新旧残差能量 |
  |---|---|
  | **≥ 1.5 Å** | **逐位相同**（`20.874434910728507` / `19.078136849613827` / `19.07729951664752`） |
  | **< 1.5 Å** | 变了（夹取生效） |
  | **0.05 / 0.093 Å** | 旧：`fail-closed`；新：有限值。**`0.093 Å` 正是真机崩点** |

  ⚠️ **判据是「全局最小配体↔环境距离」，不是「被移动那个原子到锚点的距离」** ——
  把一个水原子推到距配体原子 0 为 1.8 Å 处时，它到**另一个**配体原子只有 1.146 Å，
  照样触发夹取。第一版对照表就是这么看岔过一次，结论一度写成"上限之上也变了"。

  ### `r_floor = 1.5 Å` 的依据（不是 0.1）

  `0.1` 的来历只是防 `1/r` 发散。`1.5` 取**训练支撑域下界**：shipped R1 的训练帧实测
  最近配体↔环境距离 **1.517 Å**（水氢）/ **1.524 Å**（带 LJ），中位 1.8 Å。
  📌 **本次新增的实证**（原条目没有这一条，它决定了"夹在 0.1 行不行"）：
  **径向基在小 r 处并不衰减** —— 16 个中心均匀铺在 `[0, 5] Å`、宽 0.333，
  0.1 Å 处基函数值仍有 **0.96**；而落在支撑域下界以下那 5 个中心
  （0.333 / 0.667 / 1.0 / 1.333 Å）上的 pair weight 量级与训练充分的**完全一样**
  （`max|w|` 0.30–0.37 vs 0.32–0.39）。**完全活跃、却从未被数据约束。**
  ⟹ 夹在 0.1 Å 只能止崩，止不住"拿没训过的权重外推"。

  ⚠️ **`r_floor` 目前是编译期常量，绑定 shipped R1 模型。** 换一个支撑域不同的重训模型
  会**静默沿用 1.5**。升级路径写在 `r1_model_layout.h` 的 `ponytail:` 注释里：
  把 `r_floor_angstrom` 放进 payload config + manifest（离线训练器本来就测得到），
  再经 `buildAndLoadKernels()` 的 `defines` 发给运行时编译的内核（和 `NUM_RADIAL_BASIS` 同路）。

  判据：`tests/test_lr06_residual_distance_clamp_2026_09_16.py`（**5 passed**）——
  钉住 ①下限必须是支撑域而非 0.1、②四个站点一个都不许再抛 `MIN_DISTANCE`、
  ③能量侧夹取/力侧跳过各 2 处、④Reference 与 CUDA 同源同语义、⑤两处 sha256 已同步。
  错误码 `EXP025_DEVICE_ERROR_MIN_DISTANCE = 2` **保留**（`exp026_control_plane_layout.h`
  有 `static_assert` 钉 ABI），只改触发策略。

  ⚠️ `B_φ` 在 `r < r_floor` 处的定义变了 ⟹ `sampling_score_sha256` 变 ⟹ 与**已冻结**的
  旧分数不再可比。对 ΔG 正确性无影响（它是偏置，MBAR 对任意冻结偏置无偏），但做 A/B 时
  两臂必须都用新插件。

  <details><summary>原条目（根因分析与方案选择，保留备查）</summary>


  **症状**：`cyclod_ligand1_outer` 三个 rep 全部死在 stage-2 窗口 0：
  `LocalManyBodyResidualForce (CUDA) fail-closed in K1/K6a (computeQ) (code 2):
  pair distance below 0.1 Angstrom`。步长 2 fs→0.02 fs 无效。本节点复现 3/3。

  ### 根因

  `bias_scale` 乘在 **Group-1 整个表达式外面**（`ibs_engine.py` 的
  `f"{prefix}_bias_scale * ({_state_expr(0)} - kt*(...))"`），而 `_state_expr(0)` 里就是
  `cv_0_int + cv_0_rest` —— **配体↔环境的软核相互作用**。所以 `bias_scale = 0` 不是
  "关掉偏置"，是**把配体关成完全的鬼影**：水分子直接穿过配体，`r → 0` 是**必然**。
  三处会进入这个状态（后两处是有意的设计）：

  | 位置 | 何时 |
  |---|---|
  | `local_residual/em_no_residual.py` post-EM | **已修，见下面 ①** |
  | `ibs_engine.py` 时间步长爬坡 | 全程 `bias_scale=0`（注释：避免 dt 与未校准偏置两个不稳定源混在一起） |
  | `ibs_engine.py` 偏置爬坡 | 0 → 0.2 → … → 1.0 |

  而 `CustomCVForce` 会求值它的**每一个** CV（与系数是否为 0 无关），于是插件照样在
  鬼影几何上跑 K1/K6a，撞硬门。

  ### 为什么不能靠"别在鬼影期求值残差"来绕

  一天之内已经找到**三个**求值点，每堵一个就冒出下一个：

  1. 积分器（`setIntegrationForceGroups` 可排除）
  2. 不带 `groups` 的 `getState`（`step_guard.finite_state_check` 每 500 步一次）
  3. **`ibs_engine.py:16033 → 8924` 的 `getCollectiveVariableValues()`** —— 既不走积分
     力组、也不走 `getState`，直接求值全部 CV。**真机就死在这里。**

  鬼影期**照样在产生** <0.1 Å 的构型，排除力组只是让积分器不碰它，构型本身还在往
  重叠里漂。⟹ 逐个堵求值点是打地鼠；两次真机跑谁过谁不过是**随机**的（速度种子不同）。

  ### 已落地（独立成立，与下面的选择无关）

  - **① `local_residual/em_no_residual.py` post-EM 只清 `_s_residual`，不再连
    `_bias_scale` 一起清。** 原来两个一起清，而恢复要等到热化之后 ⟹ 那 10000 步热化
    跑的是鬼影配体。这与 `ibs_engine` 自己的注释直接冲突（它在 EM 前**只**关
    `s_residual`，明写「不像 bias_scale=0 那样连 baseline 也在正常使用的物理
    softcore-state 混合力一起关掉」）。副作用也修掉了：**A/B 两臂不再是一个热化真配体、
    一个热化鬼影**。实测：热化从「3/3 必崩在第 2500 步」变成**稳定通过**。

  ⚠️ **临时脚手架，A 落地后删掉**：`em_no_residual.install()` 里另外两个 `Context` 级
  补丁（`setParameter` 同步积分力组、`getState` 按积分力组取 State）。它们把失败点从
  热化推到了 dt 爬坡之后，但**不是修复**（见上面第 3 个求值点）。主线
  `ibs_engine.py` / `step_guard.py` 目前**零改动**。

  ### 选定修法：**A —— 插件侧夹取**（不是 B）

  把 4 处 `rAngstrom < EXP025_MIN_DISTANCE_ANGSTROM` 的「报错 + `continue`」改成**夹取**：
  能量用 `r_eff = max(r, r_floor)`（下限内是平台、C0 连续、**这条边照样计入**），
  力在下限内取 0（`invRM = 1/r` 那条发散项因此根本不需要求值）。
  位置：`plugins/LocalManyBodyResidual/platforms/cuda/src/CudaLocalManyBodyResidualKernels.cpp`
  的 304 / 447（K1 computeQ / force-scatter）与 766 / 832（K6a 同上）。
  ⚠️ 别改成单纯 `continue`：那会在下限处制造能量跳变、给出冲量。

  **为什么 A 而不是 B**（B = 把 `em_no_residual` 的孪生 System 从"只保护 EM"扩到整个预热期）：
  A **消除失效模式**，B 只是**安排它不被观察到** —— B 的不变量「残差力永远不被喂到鬼影
  几何上」靠控制流维持，而今天已经漏了三次，以后加一个探针就再漏一次。另外 A 不碰
  `run_all_windows` 那 700 行（Context 迁移要搬位置/速度/盒子/全局参数/sampler 的
  context 引用/checkpoint，漏一项是静默错），且对**所有**体系生效。

  **`r_floor` 取训练支撑域下界、跟着模型走**（写进 payload/manifest，重训时从训练帧实测），
  不沿用 0.1 Å —— 那个数的来历只是防 `1/r` 发散。本次训练帧（`pre_equilibration.dcd`
  200 帧，λ=1，**按每帧自己的盒子**算）配体↔环境最小距离：水氢 **1.517 Å** / 带 LJ
  **1.524 Å**，中位都在 1.8 Å；而崩点 0.093 Å **比支撑域近 16 倍**，那一段 typed MLP
  在响应区外、输出与梯度都不受训练约束。⚠️ 水**在**模型环境里（type vocabulary 含
  Z=1/Z=8），所以只能夹取、**不能把水氢剔出环境**。

  ### 前提已验证（2026-09-16，本机）

  | 检查 | 结果 |
  |---|---|
  | nvcc | `/opt/cuda/bin/nvcc` 12.9 ✓ |
  | 构建脚本 | `plugins/LocalManyBodyResidual/g0_build.sh`（手写 g++，非 CMake），只要 `CONDA_PREFIX` 指向 openmm_dev ✓ |
  | 上次构建 | `build_exp026_a2/` 2026-09-10 建于本机，`.o` 还在 ✓ |
  | **CUDA 内核是运行时编译** | 内核源码在 `kDeviceSource` / `kDeviceSourceCSR` 两个 `R"CUDA(...)"` 字符串里，由 OpenMM `createModule` 运行时编译 ⟹ **改内核只需重编宿主 .so，不走 nvcc 路径** ✓ |

  ### 执行顺序与代价

  1. 改 4 处 → 2. `g0_build.sh` 重编 → 3. G0–G3（`exp028_run_regression_suite.sh`）
  → 4. 同步 `local_residual/openmm_plugin.py` 的 `KNOWN_PLUGIN_SOURCE_SHA256`
  **和** `resources/outer_lambda_local_residual/manifest.json` 的 `plugin.source_sha256`
  （run 内重训产出的 manifest 会自动带新值）→ 5. 删掉上面那两个临时 `Context` 补丁
  → 6. `cyclod_ligand1_outer` 真机复跑 → 7. 换一个疏水口袋体系（jnk1 / p38 / brd4）再跑一遍。

  ⚠️ `B_φ` 在 `r < r_floor` 处的定义变了 ⟹ `sampling_score_sha256` 变 ⟹ 与**已冻结**的
  旧分数不再可比。对 ΔG 正确性无影响（它是偏置，MBAR 对任意冻结偏置无偏），但做 A/B 时
  两臂必须都用新插件。
  ⚠️ 错误码 `EXP025_DEVICE_ERROR_MIN_DISTANCE = 2` **保留**（`exp026_control_plane_layout.h`
  有 `static_assert` 钉 ABI），只改触发策略。

  ### 已排除，别重查

  - 不是体系爆炸 —— 失败时逐力组全正常（g0 max|F|=4818、g2=677 kJ/mol/nm），坐标全有限，
    配体内部最近对 0.97 Å（正常 C–H）；
  - **不是合并主线改坏的** —— 插件源码与 EXP-030 实跑那份（`Atenolol-rank11/plugins/…`，
    sha `10afff53…`）`diff` **只差 17 行 MIT 许可证头**；`em_no_residual.py` 两份也都清
    `_bias_scale`；
  - 不是残差力在推 —— 失败坐标上把那个水 H 从 0.12 Å 扫到 4 Å，残差力恒 ~1e-5 kJ/mol/nm
    （本次是闭式重训出的空模型，`relative_improvement=9.2e-07`）；
  - 不是"软核允许重叠所以必然触雷" —— `power_lj=[2,2]`、`alpha_lj=0.5`，窗口 0 最深
    λ_vdw=0.5933 时带 LJ 的对势垒 ~26 kT、λ=0.7054 时 ~118 kT，热涨落进不去。**能进去的
    只有鬼影期。**
  - `candidates=0` 是**死字段** —— `EXP026_STATUS_CANDIDATES` 只被 `exp026ResetSupportStatus`
    置 0、全仓没有任何地方写过。要么补写入，要么从错误信息里删掉。
  </details>


---

## 往这里加条目的规则

1. **一条待办只在本文出现一次。** 专题文档讲"为什么"，本文讲"还欠什么、欠在哪一行"。
2. 关闭一条就**整段移进 `archive/`**，页首写清结论去了哪 —— 不要在本文留 `[x]` 尸体。
3. 新条目必须带**位置**（`文件:行号` 或函数名）和**判据**（怎样算做完）。
   写不出判据的不是待办，是想法，去 `design/`。
4. 标"不修"的要写明理由，并注明**理由别重新论证** —— 否则下一个人会花一天重查一遍。
5. 科学结论不进本文，进 [STATUS.md](STATUS.md)；协议/代码变更进 [CHANGELOG.md](CHANGELOG.md)。

6. **新条目必须带 `[P1]`/`[P2]`/`[P3]`，写在编号之后**，并落到对应文件：
   `[P1]` 进本文、`[P2]` 进 [TODO_P2.md](TODO_P2.md)、`[P3]` 进 [TODO_P3.md](TODO_P3.md)；
   实测表与重放证据进 `archive/`。**改优先级 = 整段搬文件**，不许只改标签。
