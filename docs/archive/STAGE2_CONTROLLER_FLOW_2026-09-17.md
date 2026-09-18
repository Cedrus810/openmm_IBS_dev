# Stage-2 控制器：实际控制流全景（2026-09-17 静态抽取）

> ## 🗂 2026-09-17 归档 —— **不是待办**
>
> 本文是从 `_decide_once` 逐行抽出来的**那一天的**执行顺序快照（行号会过期，本文按守卫谓词定位）。
> 当天的结构性结论已各自归位，原文一字未改：
>
> | 本文的哪一段 | 结论去了哪 |
> |---|---|
> | §3 D1/D2 缩跨度对 window 0 构造上不可行 | ✅ `REWIND-01` 当天接通有界重窗（**不锚在尾部** ⟹ window 0 够得着），见 [TODO_closed.md](TODO_closed.md) |
> | §2.5 静态遍历确认的死代码 | ✅ 死动作 1 → 0（`tools/audit/static_controller_2026-09-17.py` 复核） |
> | §3 D3 分支 2 与分支 29 | ⚖️ **已核实是冗余不是漏洞**（被 O4 拦住，不会漏发 ΔG）；留下的是可维护性 |
> | §3 D4/D5/D6 + §4 结论（32 条分支 × 5 层改写，语义由书写顺序决定） | ⬜ **仍然开着** —— 归 [TODO_P2.md](../TODO_P2.md) `AUDIT-S2-02`（判断函数 1795 行 / 63 个 return 的重构立项） |
> | §3 D2 点名的那批 run | 归 [TODO.md](../TODO.md) `BM-B` |
>
> 📌 **别拿本文的分支编号/条数当现值** —— 它是 09-17 那天的快照，`REWIND-01` 当天就改了分支结构。
> 要现值就重跑 `tools/audit/static_controller_2026-09-17.py`。


> **同批次的另外两份**：死线与修复逻辑 → `STAGE2_DEAD_LINES_2026-09-17.md`；
> 判据/门的数据审查（含一处撤回）→ `AUDIT_GATES_AND_CRITERIA_2026-09-17.md`。

不是设计文档。这是从 `abfe_preoptimizer.Stage2RepairController._decide_once`
（4349–6181）**逐行抽出来的实际执行顺序**，注释一概不采信。

规模：控制器类 4868 行 / `_decide_once` 1832 行 / 45 个 `return plan(...)` /
执行器 `_run_stage2_autonomous` 1322 行 / 动作词汇 12 个 / 出口词汇 14 个。

---

## 0. 全景

```
              decide()
                 │
     ┌───────────▼────────────┐
     │  retire 循环（外层）    │  终态是 NO_FEASIBLE_ACTION /
     │  最多退役 N 个窗口      │  HALT_FRAMES_ADMISSION_CAP 时，
     └───────────┬────────────┘  退役 earliest 换下一个窗口重判
                 │
     ┌───────────▼────────────┐
     │   _decide_once         │  32 条分支的**线性优先级链**
     │   （下面第 2 节）        │  命中即 return，顺序即语义
     └───────────┬────────────┘
                 │  每条分支的结果都要过：
     ┌───────────▼────────────┐
     │  plan() 的 5 层否决     │  ← 第 1 节。**这一层会改写动作本身**
     └───────────┬────────────┘
                 ▼
          {action, exit, windows}
                 │
     ┌───────────▼────────────┐
     │ _run_stage2_autonomous │  执行器：只写盘
     └────────────────────────┘
```

---

## 1. `plan()` 的 5 层否决（对 32 条分支**全部**生效）

`plan()` 名义上是"打包返回值"，实际是第 6 个决策层。按顺序：

| # | 触发 | 原动作 → 改写成 |
|---|---|---|
| O1 | Epoch 类动作 + 该窗口付不起新 Epoch 的验证额度 | **偏斜类**失败 → `NO_ACTION`/`HALT_BUDGET`；**样本量类**失败 → **`RUN_PRODUCTION`**/`HALT_BUDGET` |
| O2 | `RUN_PRODUCTION` + 目标窗口补帧块数满额 / 边际增益停滞 | 部分满额→剔除后继续补；全部满额→`NO_ACTION`/`HALT_FRAMES_ADMISSION_CAP` 或 `NO_FEASIBLE_ACTION` |
| O3 | 动作在 no-op 账本里（同盘面跑过、盘面未变） | `NO_ACTION`/`NO_FEASIBLE_ACTION` |
| O4 | `DONE` 但 `evidence_status != CONVERGED` | `NO_ACTION`/`HALT_EVIDENCE_CONTRADICTS_DONE` |
| O5 | 生产类动作 + stage 剩余步数 < 代价 | `NO_ACTION`/`GLOBAL_BUDGET_EXHAUSTED` |

**O1 是"把错误的采样无限增大"的直接来源**：分支说"f_k 不对，换 Epoch"，
`plan()` 说"换不起 → 那就补帧"。分支的对症判断在打包时被静默替换成它刚刚否决过的动作。

---

## 2. 32 条分支的线性优先级链

命中即返回。`_pick()` 让绝大多数分支**只对 `earliest` 一个窗口**生效。

| # | 判据 | 动作 | 出口 |
|---|---|---|---|
| 1 | 没有任何窗口产物 | `CONTINUE_WARMUP` | — |
| 2 | `ANALYSIS_COMPLETE` **且** 布局可信 **且** 无作废证据 **且** 覆盖完整 | `DONE` / `NO_ACTION` | `DONE` / `ANALYSIS_COMPLETE_PRECISION_UNMEASURED` |
| — | *计算 `earliest` = 第一个 skipped，否则第一个 PROBLEM* | | |
| 3 | `phase == IDENTITY_MISMATCH` | `CONTINUE_WARMUP` | `HALT_INVALID_INPUT` |
| 4 | earliest 是 `TERMINAL` 且非统计驳回 | `NO_ACTION` | `NO_FEASIBLE_ACTION` |
| 5 | earliest 的产物属于旧布局 | **`RUN_PRODUCTION`** | — |
| 6 | earliest warmup 预算耗尽 + f_k 未验证 | `INSERT_LAMBDA` / `NO_ACTION` | — / `NO_FEASIBLE_ACTION` |
| 7 | 有待采子窗（immutable rewindow 的子系综） | **`RUN_PRODUCTION`** | — |
| 8 | earliest 被求解器踢出协方差链 | **`RUN_PRODUCTION`** | — |
| 9 | 子窗是支撑/偏斜类失败 | `NO_ACTION` | `NO_FEASIBLE_ACTION` ⚠️ **2026-09-17 起第一次真正可达**（此前子窗只由 `IMMUTABLE_REWINDOW` 产生，而它发不出来） |
| 10 | skipped + rescue 跑过仍不足 | **`RUN_PRODUCTION`** | `HALT_BUDGET` |
| 11 | f_k 被统计驳回 | `RECALIBRATE_FK` | `HALT_FK_REFUTED` |
| 12 | 冻结验证在剩余预算内算术不可达 | `RELEARN_FK_EPOCH`（一次）/ `NO_ACTION` | `HALT_VALIDATION_BUDGET_UNREACHABLE` / `NO_FEASIBLE_ACTION` |
| 13 | 单周期验证批次打满但全局还有钱 | `PROVISIONAL_PRODUCTION` | `HALT_LOCAL_VALIDATION_CAP` |
| 14 | 预热中有预算，但续预热已被记为 no-op | **`RUN_PRODUCTION`** | — |
| 15 | 预热中有预算 | `CONTINUE_WARMUP` | — |
| 16 | 预热卡住 + 无 stage 结果 | `CONTINUE_WARMUP` | `HALT_NO_ATTRIBUTION` / `NO_FEASIBLE_ACTION` |
| 17 | 预热卡住 + 有 stage 结果 | `RECALIBRATE_FK` → `INSERT_LAMBDA` → `NO_ACTION` | — / `HALT_LAMBDA_BUDGET_INSUFFICIENT` |
| 18 | 生产步数未达目标 | **`RUN_PRODUCTION`** | — |
| 19 | 疑似脱轨（单块边际 N_eff 塌） | `PROBE_CANDIDATE_FK` | — |
| 20 | f_k 探针建议重标定 | `RECALIBRATE_FK` | — |
| 21 | earliest 支撑不足但累计 f_k 残差证据缺失 | `ANALYZE` | — |
| 22 | 累计 f_k 残差 FAIL/UNMEASURED | held-out `ACCEPT`→`RECALIBRATE_FK`；`REJECT`→`SPLIT_TAIL_WINDOW`/`INSERT_LAMBDA`/`NO_ACTION`；否则 `PROBE_REANCHOR_EPOCH`（探针 no-op 则 **`RUN_PRODUCTION`**） | — / `NO_FEASIBLE_ACTION` |
| 23 | `min N_eff/g` 未随采样上升（边际停滞） | `INSERT_LAMBDA` / `NO_ACTION` | — / `NO_FEASIBLE_ACTION` |
| 24 | 自检不足 **且**归因是偏斜类 | `SPLIT_TAIL_WINDOW`/`INSERT_LAMBDA` → `ANALYZE` → `NO_ACTION` | — / `NO_FEASIBLE_ACTION` |
| 25 | 自检不足 **且**归因是样本量类 | **`RUN_PRODUCTION`** | — |
| 26 | 仍有 skipped 窗口 | **`RUN_PRODUCTION`** | — |
| 27 | 无 earliest 但有 UNKNOWN 窗口 | `ANALYZE` | — |
| 28 | 布局里的窗口产物缺失 | **`RUN_PRODUCTION`** | — |
| 29 | `ANALYSIS_COMPLETE`（**不带分支 2 的三道守卫**） | `DONE` / `NO_ACTION` | `DONE` / `ANALYSIS_COMPLETE_PRECISION_UNMEASURED` |
| 30 | 没有 stage 结果 | `ANALYZE` | — |
| 31 | `allow_untrusted` | `DONE` | `DONE_UNTRUSTED` |
| 32 | 兜底 | `NO_ACTION` | `NO_FEASIBLE_ACTION` |

---

## 2.5 静态遍历确认的死代码（2026-09-17，`tools/audit/static_controller_2026-09-17.py`）

| # | 发现 | 性质 |
|---|---|---|
| S1 | `IMMUTABLE_REWINDOW` 是 13 个声明动作里唯一 `decide()` 发不出的 | ✅ **已修**（abfe-ibs-db，2026-09-17）：发出点接在 D2/D5/D6/D7 各自兜底之前；死动作 1→0，全量 2653 passed |
| S2 | **分支 31（`allow_untrusted` → `DONE`）是死代码** | 见下 |
| S3 | `_window_states = dict(_states)` 赋值后全仓无人读 | ✅ **已删**。工具现在只剩 `_` / `_nm` 两个解包占位误报 |
| S4 | 死出口 0 条；13 个动作执行器全都认识；`return` 后无不可达语句 | 干净 |

### S2 详情

分支 31 的守卫是 `analysis_status == ANALYSIS_INCOMPLETE` + `allow_untrusted`，
发 `plan("DONE", exit_="DONE_UNTRUSTED")`。但 2026-09-15 加的 O4
（`action == "DONE" and _evidence_status != "CONVERGED"` ⟹ 改写）**没有 `allow_untrusted` 豁免**
——它的注释写明这是刻意的（「`allow_untrusted` 只改 `trust_level`，不救这里」）。

而在分支 31 那个位置，`analysis_status` 必然是 `ANALYSIS_INCOMPLETE`
⟹ `_evidence_status` 不可能是 `CONVERGED` ⟹ **这条分支永远被改写成
`NO_ACTION`/`HALT_EVIDENCE_CONTRADICTS_DONE`，一次都发不出 `DONE`。**

仓库自己的测试已经记录了新行为
（`tests/test_stage2_repair_controller.py`：「`DONE_UNTRUSTED` 今天只在
『analysis COMPLETE + precision MEETS + 放行』时才出现」），
**但分支和它那段「调用方显式放行 ⟹ DONE_UNTRUSTED。这是发布策略」的注释被留在原地** ——
注释现在是假的，会把人带偏。

⚠️ 这**不**说明 `--allow-untrusted-stage-results` 整个没用：它在别处还有作用
（跳过 rescue、不进指纹）。死的只是控制器里这条 `DONE` 路径。

处置二选一，**要用户拍板**：
- (a) 认为 O4 的判断对 ⟹ **删掉分支 31 和它的注释**，`DONE_UNTRUSTED` 只保留在分支 2/29；
- (b) 认为放行就该出 `DONE_UNTRUSTED` ⟹ 在 O4 里给 `allow_untrusted` 开豁免。

两者语义相反，别自己选。

---

### S5 ⚠️ 「末窗 K=9 死区」是**误诊**（2026-09-17 更正）

本文与 `STAGE2_DEAD_LINES` 早先都写过 cyclod_ligand3 rep1/rep3「落在 K∈(hi, 2hi−1] 的死区、
重跑都起不来」。**那个说法是错的**，源头是 `feasible()` 里的一段注释。

`_legalize_tail_window` 自己的 docstring 给的是正确定义：

> 死区：K ∈ (hi, 2*lo−1) 既不能当单窗（超 hi）又拆不开（可拆区间 [2*lo−1, 2*hi−1]）

lo=4 / hi=8 ⟹ 死区 = (8, 7) = **空集**；可拆区间 = [7, 15]，**K=9 在里面，完全拆得开**。

真正卡住的是**锚点**：`_legalize_tail_window` 调 `repartition_tail_from_anchor`，
而 anchor 来自 `first_untrusted_window` —— 后者在 `idx <= 0`（或没有不可信窗口）时恒返回 `None`。

**这是本文档反复出现的同一族缺陷的又一例**：
「**合法化**」（只要末窗超 hi 就**必须**做，与哪个窗口有问题无关）
借用了「**修复**」的定位谓词（`first_untrusted_window`）当锚点来源。
两个概念被写成同一个数，于是坏窗口是 window 0 时，一个本来完全合法的拆分做不了。

⚠️ `abfe_preoptimizer.feasible()` 里那段把死区写成 `hi < K_tail ≤ 2*hi−1` 的注释
**与 `_legalize_tail_window` 的定义不兼容**，且把锚点问题误标成区间问题。
同一个概念两份定义 —— 引用前先看是哪一份。

（由 abfe-ibs-db 发现并核实；代码未动，已报用户。）

### S6 产物目录三层，别混（2026-09-17 全部核实）

判定同一个 stage 下某个目录属于哪一层，唯一的机械规则在 `segment_index_of_dir`：
**后缀是不是纯数字**。段发现是 `glob(stage_dir + "*")`，**三层都会被 glob 到**，
靠这一条规则筛。

```python
suf = name[len(base) + 1:]
return int(suf) if suf.isdigit() else None
```

| 层 | 路径 | 后缀 | 段发现 |
|---|---|---|---|
| **采样段** | `<run>/vanishing_<N>/` | 纯数字 | ✅ 收。共用**全局** `window_ranges`；段是**相加**不是取代（§6.8） |
| **有界重窗** | `<run>/vanishing_rewindow_<sha12>/` | `rewindow_…` | ❌ 跳过。父窗内的重叠子系综，自带 ranges |
| **production rescue** | `<run>/vanishing_rescue/<plan_id>/` **和** `<ck>/vanishing_rescue/<plan_id>/` | `rescue` | ❌ 跳过（两层各一份） |

构造点：`abfe_pipeline.py` 约 17240–17252。
⚠️ **是硬编码字面量 `"vanishing_rescue"`，不随 stage 名参数化**（别写成 `<stage>_rescue`）。
`plan_id = sha256(json(rescue_ranges, sort_keys=True))[:12]`。

#### 一条构造安全性，别顺手改坏

`rewindow` 的后缀恒含字面量 `rewindow_` ⟹ `"rewindow_0123456789ab".isdigit()` 恒为 `False`
⟹ **即使 sha12 十二位碰巧全是数字（概率 ≈ (10/16)¹² ≈ 0.34%，不可忽略），也不会被误认成段。**

这是**构造上**安全，不是概率上安全。**把后缀简化成裸 sha12 就会把它降级成 0.34% 的隐性冲突。**
（已由 abfe-ibs-db 写进 `segment_index_of_dir` 的 docstring。）

#### 实际后果

把 `vanishing_4` / `vanishing_5` 当成 rescue 产物去解读是**错的** —— 它们是采样段，
描述的是**同一套布局**，必须按窗口合并求解（`_load_ibs_window_outputs_merged`）。
真正自带另一套 ranges 的只有 `rewindow` 那一层。

---

## 3. 从这张表直接读出来的结构性缺陷

### D1 —— 补帧有 11 条入口，缩跨度只有 5 条且全部带闸

`RUN_PRODUCTION` 可达于分支 5、7、8、10、14、18、22、25、26、28 **加上 O1 改写** = **11 条路径**。
`INSERT_LAMBDA` 可达于 6、17、22、23、24 = 5 条，每条都先问 `feas["insert_lambda"]`。
`SPLIT_TAIL_WINDOW` 只可达于 22、24 = 2 条，且都额外要求 `earliest == 末窗`。

链条本身就偏向补帧。

### D2 —— 缩跨度的两个动作都锚在尾部，window 0 在构造上够不着

```
tail_repartition_anchor():  idx = first_untrusted_window(view)
                            if idx is None or idx <= 0: return None
```

`feasible()` 由此判：
- `SPLIT_TAIL_WINDOW` 无 anchor ⟹ 不可行；
- `INSERT_LAMBDA` 末窗会涨过 `hi` 且无 anchor ⟹ 不可行；
- `INSERT_LAMBDA` 还有 `max_path_insertions`（默认 3）硬上限。

**`earliest == 0` 时，两个缩跨度动作全部不可行。** 而这条方法的失败窗口恒是
window 0（解耦端点）。剩下唯一恒可行的动作是补帧 ⟹ 补到 O2 的块数上限 ⟹
`NO_FEASIBLE_ACTION`。

代码注释自己点名的 run：`cyclod_ligand3/rep1`（连插三次，末窗 K=6→7→8→9，
第三次 `RuntimeError` 且非法布局已落盘）、`cmet_ligand1/rep1`、`jnk1_ligand1/rep3`、
`p38_ligand2/rep1`、`cyclod_ligand1/rep2`（插 6 次）。

### D3 —— 分支 2 与分支 29 是同一条规则的两份实现（**已核实：被 O4 拦住，是冗余不是漏洞**）

> 🔑 [2026-09-17 静态核实] `_evidence_status` 只在
> 「布局可信 **且** 无作废证据 **且** `ANALYSIS_COMPLETE` **且** 精度达标」时返回 `CONVERGED`。
> 所以分支 29 即使发出 `DONE`，也必然被 `plan()` 的 O4 改写成
> `NO_ACTION`/`HALT_EVIDENCE_CONTRADICTS_DONE`。**不会漏发 ΔG。**
> 留着的问题是可维护性：同一条规则两份实现，改一处忘另一处。

分支 2：`ANALYSIS_COMPLETE` **且**布局可信**且**无作废证据**且**覆盖完整 → `DONE`。
分支 29：`ANALYSIS_COMPLETE` → `DONE`。**三道守卫一道没有**，相隔 27 条分支。

布局不可信时分支 2 落空，中间 26 条路由若全不命中，分支 29 直接发 `DONE`——
唯一拦住它的是 O4。这正是设计文档 §6.1 称为"最贵的 bug"的同一形状，仍在同一个函数里。

### D4 —— `earliest` 单窗漏斗，没有公平性

`_pick()` 让 11 条以上的分支只对 `earliest` 生效，下游窗口一律 `blocked_by_upstream`。
整条 stage 的进展被一个窗口绑架。`_retirable_window` 的注释记着实测后果：
brd4_ligand2/rep1、cyclod_ligand2/rep2 的末窗**一块帧都没批过**、预热预算还剩
37 万 / 87 万步，"全程躺在 blocked_by_upstream 里一次都没被看过"，
而整腿半程漂移（+8.01 / +11.14 kJ/mol）几乎全部来自那个末窗。

补丁是 `_retirable_window`：终态时退役 earliest 换个窗口重判 ——
于是"终态"变成了条件性非终态。

### D5 —— 同一条物理判断在链条里自相矛盾

"同分布加帧治不了偏斜"（§5.1，实测 250k→1M 让 top1% 0.545→0.762）被分支 9、21、24
当作**不补帧**的理由；而 O1 在付不起 Epoch 时把动作改写成补帧，分支 8/26 对
"求解器踢出协方差链"（同一个低支撑条件）直接补帧。同一份证据，三种处置。

### D6 —— 出口词汇 14 个，设计文档说只有 3 个真终态

`TERMINAL_EXITS` 之外还有 `HALT_*` 系列 8 种被当路由信号，执行器里又按字符串
重判一遍（25 个 token）。`action` / `exit` / `verdict` / `evidence_status` /
`execution_status` / `phase` 六套词汇并存。

---

## 4. 结论

不是某条分支写错了。是**这个形状本身不成立**：

- 32 条分支 × 5 层改写 = 160 种组合，语义完全由书写顺序决定，而顺序没有任何一处成文；
- 动作集合对 window 0 不完备，而 window 0 是本方法的必然坏窗口；
- 因此控制器的稳态行为是：对着一个它修不了的窗口，反复执行它自己判定为无效的动作，
  直到配额耗尽。

三个月来的审计（#20/#23/#24/#30/第三轮复核）全部在调分支顺序，
**没有一次动过动作集合的完备性**。
