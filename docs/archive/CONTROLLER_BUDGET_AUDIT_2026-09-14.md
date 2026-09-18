# Stage-2 控制器 + 预算 全面审计（2026-09-14）

> ## 🗂 2026-09-18 归档 —— **不是待办**
>
> **65 条（63 + 事后 2）全部落地**，2026-09-16 逐条回源码对账确认（见下面那节
> 《2026-09-16 状态列整列纠正》）。回归钉子：
> `tests/test_audit_2026_09_14_controller_budget.py`。
>
> | 本文的哪一段 | 状态 |
> |---|---|
> | 三条贯穿性根因 | **仍然有效** —— 根因 1 还活着的直接证据就是 `AUDIT-S2-02`（[../TODO_P2.md](../TODO_P2.md)） |
> | #18「零调用点 ≠ 忘了接」/ #65「改名/改值不通知消费者」 | **仍然有效**，两条留给下一个人的规矩 |
> | 逐条状态列 | 会过期的缓存。「某条修没修」的权威**只有源码** |
> | 真机验收 | **一条都没跑过** → [../TODO.md](../TODO.md) 的 `AUDIT-S2-03` |

**范围**：`Stage2RepairController`（读盘 / `decide()` / 可行性）、自治执行器
`_run_stage2_autonomous`、两本预算账（预热-验证账 / 生产账）、以及这些判据背后的
物理/统计口径。

**方法**：纯源码审计，零 GPU、零测试运行。6 路独立并行审计后去重合并，共 **63 条**。
标 ⚑ 的是两路以上**独立**发现同一条（可信度最高）。

**2026-09-14 审计后新增 2 条**（不在原 63 条里，编号接着排，见 **S6**）：
**#64** win0 空转重跑前置链（唯一一条**正在实时烧 GPU** 的）、
**#65** 改了落盘名但消费者在另一个文件里没跟。

**修复状态**：⚠️ **状态列已于 2026-09-16 整列纠正为 `FIXED`，先读上面那一段。** 以下是 09-14 首版的原始说明，保留是为了说清这一列的由来 —— 状态列初始一律 `OPEN`。**这一列只有在对应改动真的落盘之后才准改**，
谁改谁负责标，并在 `docs/CHANGELOG.md` 同日条目里留一行。
⚠️ 2026-09-14 首版曾把整列预先写成 `[FIXED]`（当时一条都没修），已纠正 ——
**不要把这一列当成可信输入**，每条都要回源码核实。

**2026-09-14 第二次核实（文档侧）**：逐条 `git diff` 回源码比对，把 **44 条**标成
`FIXED`（判据：源码里有 `[审计 #N]` 标记**且**该标记落在 `git diff` 的**新增行**上
**且**紧邻的是真实代码改动，不是只加注释）。核实基准快照：
`abfe_preoptimizer.py` md5 `1f7349ef`、`abfe_pipeline.py` md5 `fd4fe752`、
`ibs_engine.py` md5 `f23e10ea`。
⚠️ **这三个文件当时正在被并行编辑**（同一次会话内两次抽取相差 2 条），
行号与本次核实结论都有保质期 —— 标 `OPEN` 的条目**可能已经修好只是还没落标记**，
反向不成立（标 `FIXED` 的都验过源码）。

**当前分工**（避免同一文件两个写者）：
- `abfe_preoptimizer.py` / `ibs_engine.py` / `runabfe.py` / `abfe_config.json` / 新增回归测试：本会话
- `abfe_pipeline.py` / `lambda_path_versions.py`：见 CHANGELOG 同日条目
- 两条跨文件契约（no-op 台账 key、history 的 `sampling_units_snapshot`）见 S1 #8/#9 正文，
  两侧必须同一份，不许各写一遍。


---

## ⚠️ 2026-09-16 状态列整列纠正 —— 先读这一段

**这份文档的 `OPEN` 状态列曾经落后源码整整两天。** 2026-09-16 按本文档自己定的判据
（源码里有 `[审计 #N]` 标记）逐条回源码核实：

```
grep -ohE '审计 #[0-9]+' abfe_preoptimizer.py abfe_pipeline.py ibs_engine.py \
    runabfe.py lambda_path_versions.py abfe_config.json | sort -u
```

**#1–#36、#38–#64 全部在场。** 缺的两个：
- **#65** 本文早已自行标 `FIXED`；
- **#37** 没有标记，但**缺陷本身已不在** —— `abfe_preoptimizer.py:4635` 现在是
  `_left = _pb.get("stage_remaining_steps")` 配 `if _left is not None and int(_left) < _cost`，
  紧邻注释逐字写着「先前 `or 0` 把 None 当成 0 ⟹ 一律拦死，等于"未知 = 耗尽"」。

⟹ **65 条全部落地**，状态列已整列改成 `FIXED (2026-09-16 对账)`。

### 但这不是"审计通过"

1. **对账只证明"那处缺陷不在了"，不证明"修得对"。** 全部 65 条仍然是**真机零验证**
   （见 `TODO.md` 的 `AUDIT-S2-03`）。
2. **本文开头那条警告反过来也成立了。** 原文写的是「标 `OPEN` 的可能已经修好只是还没落标记，
   反向不成立」—— 结果 20 条虚挂的 `OPEN` 正是这么来的。**代价不是漏修，是重复排查**：
   `TODO.md` §1 的 `BUD-*` 有一部分就是在已修的地方重新查出来的。
3. **留下的规矩**：「某条缺陷修没修」这个事实**只有一份权威，是源码**；状态列是缓存。
   这就是本仓最贵的复发模式「同一个不变量的 N 份实现」在文档上的版本。
   ⟹ 下一份审计文档**不要**再开一个初始全 `OPEN` 的状态列，
   要么改的时候同步改，要么干脆不写这一列、只留一条回源码的 `grep`。
4. **三条贯穿性根因（下一节）没有过期**，`AUDIT-S2-02`（判断函数已 1795 行 / 63 个 `return`）
   就是根因 1 还活着的直接证据。**读本文请从那三条读起，不要从状态列读起。**

---

## 三条贯穿性根因

1. **路由在「知道这个窗口缺什么」之前就锁定了。**
   `decide()` 先算 `earliest`、再用 `_pick()` 把每个分支限制到那一个窗口，而
   `earliest` 之前还有若干**无条件 `return`**（换 Epoch 预算预检、1e）。任何一条
   早期 return 会把后面所有分支对该窗口静默删掉 ⟹ #20/#21/#22/#23 是同一形状的
   四个实例。

2. **「未知」在四处有四套语义。**
   同一个 `warmup_steps_left is None`：`_epoch_validation_unaffordable` 里
   fail-closed、分支 1e 里当「有钱」、`read_aggregated` 里写 None、执行器里
   `or 0` 当零。⟹ #15/#36/#37/#38 是同一条规矩（「未知不是零，两个方向都不是」）
   的第 4~7 次复发。这条规矩本身已经在仓库记忆里记过。

3. **子窗（sampling unit）是二等公民。**
   不进 history snapshot、不进块账、no-op key 写了没人读、no-op 闸显式排除、
   `status` 不参与过滤、f_k 驳回记到父窗头上 ⟹ #7/#8/#9/#42 全部由此而来。

---

## S0 — 会产出错误的 ΔG，或把已算好的结果弄丢

> **2026-09-14 定级收窄：#3 已从本档下调到 S5**（它污染的是归因表，不是 ΔG 本身）。
> 条目原文与收窄依据见 S5，编号不变、**仍是必修**。

| # | 位置 | 缺陷 | 状态 |
|---|---|---|---|
| 1 | `ibs_engine.py:20240` | `marginal[b] = ESS(prefix_b) − ESS(prefix_{b−1})` **不是边际量**（ESS 是非可加的比值泛函）。加入一帧权重 W≫其余时 ESS 反而塌向 1 —— 注释里贴的 `73.3 23.1 2.3 3.0` 正是「第一次采到主导构型」的签名，却被判 `CONFIRMED_DERAILMENT` 并允许关闭 Epoch。在 vanishing 腿上丢掉高权重帧 = 欠采样 = **低估解耦代价** | **FIXED** |
| 2 | `abfe_pipeline.py:10947` 等 5 处 | 段号 glob 用 `rsplit("_",1)[-1].isdigit()`，而 rewindow 目录名是 `vanishing_rewindow_<sha256[:12]>`，纯数字概率 ≈(10/16)¹²≈0.34%。命中则该目录被当采样段合并，**子窗局部下标 0/1 被当成物理窗口 0/1**，覆盖度检查照样通过 ⟹ 静默错 ΔG | **FIXED (2026-09-16 对账)** |
| 4 | `abfe_preoptimizer.py:5595` | `lambdas_from_version_record` 在长度不符时 **fail-open** 回退到未量化的内存 λ 表 —— 正是它自己 docstring 里说要防的 5e-9 死锁 | **FIXED (2026-09-16 对账)** |
| 5 | `abfe_pipeline.py:11094` + `lambda_path_versions.py:279` | `segments_before_change`（自产的目录扫描结果）进了 `event_id` 哈希，而那是唯一幂等键 ⟹ 重试时只要有新段目录出现，**同一次插点会被应用第二次**。「自产产物进身份」这条老坑的又一次复发 | **FIXED** |
| 6 | `abfe_pipeline.py:10903` + `15393` ⚑ | 多段补帧循环里 `result` 被每段覆盖，而终态在下一轮开头就 break、不再重算 ⟹ 交出去的是窗口子集跑的部分和，还会**覆盖 `_run_stage2_with_path_evolution` 已算好的完整 stage2**；下游 `_assert_stage_result_sane` fail-closed 抛 `subset_partial_sum_not_delta_G` | **FIXED** |

## S1 — 死循环 / 无上限烧 GPU / 整跑崩

| # | 位置 | 缺陷 | 状态 |
|---|---|---|---|
| 7 | `abfe_preoptimizer.py:2670` vs `abfe_pipeline.py:11510` | `sampling_units` 只按 `path_version` 过滤、**不看 `status`**，而 `_solve_with_rewindow_children` 只采信 `SAMPLED`，`_topup_rewindow_child` 又从不把 status 推回 `SAMPLED`。⟹ `ABANDONED_NO_PRODUCT` 子窗每轮烧一个块直到 40 轮上限，**一帧都不进 ΔG**，且停滞保护看不见（签名里它的 `production_steps` 每轮都在变） | **FIXED** |
| 8 | `abfe_preoptimizer.py:3215`/`3163`/`2327` + `abfe_pipeline.py:10797` ⚑⚑ | 子窗**两道刹车同时失效**：no-op 闸写死 `and not unit_id`；块账只扫 `view["windows"]` 而 `sampling_units` 一个不在，`plan()` 又拿**父窗**的账去问准入（父窗步数已冻结 ⟹ 去重后恒 1 行）。BUD-03 修过的「单窗烧 150 万步」在新位置原样复发 | **FIXED** |
| 9 | `abfe_pipeline.py:11648` vs `abfe_preoptimizer.py:3497` ⚑ | 子窗 no-op 写 key `f"{action}:unit:{uid}"`，全仓**无人读**（`_is_noop` 只查 `f"{action}:{idx}"`）；而通用 no-op 检测只要 `plan["unit_id"]` 非空就一律走 unit 分支 ⟹ 那一轮连普通 key 也不写 | **FIXED** |
| 10 | `abfe_pipeline.py:10777` vs `10852` | 停滞降级「只许一次」被签名变化重置 —— 而降级去跑的 `PROBE_REANCHOR_EPOCH` **必然新建采样段并真采样** ⟹ 必然改签名 ⟹ 第 3/6/9… 轮各开一个段。注释声称修掉的「无限开新段、烧 GPU」被这个重置抵消 | **FIXED** |
| 11 | `abfe_preoptimizer.py:3799` vs `1983` | `HALT_FK_REFUTED` **不在 `TERMINAL_EXITS` 里** ⟹ `terminal=False` ⟹ 主循环照样分发 `RECALIBRATE_FK`，而它抛 `IBSFrozenCalibrationValidationError`（裸 `RuntimeError` 子类，不在路由 `except` 里）⟹ **统计驳回以未捕获 traceback 结束整跑**。正是 09-13 那条清理声称已消灭的「看起来像终止、实际不终止」 | **FIXED (2026-09-16 对账)** |
| 12 | `abfe_pipeline.py:12058` + `10762`/`11117` | `_legalize_tail_window` 在死区循环里自己插完 λ 之后，用**插之前的 `view`** 解 anchor。model B 下 `new_lam[i]=old_lam[i−1]`，anchor 值移到 `j+1`，那里永远不是窗口起点 ⟹ `ValueError("anchor 不是现有共享态")` 炸穿。默认 4/5 配置 + `K_tail==6` 时可达 | **FIXED (2026-09-16 对账)** |
| 13 | `abfe_preoptimizer.py:4188` | 5a-2 的 REJECT 分支发 `INSERT_LAMBDA` **完全不查可行性**（全仓唯一漏网的一处）⟹ 要么执行器 `RuntimeError` 炸穿，要么绕过 `max_path_insertions` 无限插点 | **FIXED (2026-09-16 对账)** |
| 14 | `abfe_preoptimizer.py:3986` | 三元式与它上面的注释**正好相反**（注释：重标定 → 仍压不住才插 λ；代码：insert 可行就插、不可行才重标定），且 `_recal_is_noop` 为真时 `or` 短路发 `INSERT_LAMBDA`，而同一条理由文本在打印「补 λ 当前不可行」，还配了个 λ 预算的 exit 词 | **FIXED (2026-09-16 对账)** |
| 15 | `abfe_pipeline.py:11189` ⚑⚑ | `int(_rec.get("warmup_steps_left") or 0)` 把视图**特意**写成 `None` 的未知压成 0，并据此 `break` 整个自治循环，日志还打印一个编造出来的「剩余 0 步」 | **FIXED** |
| 16 | `abfe_pipeline.py:10920` | `CONTINUE_WARMUP` 不传 `_output_dir_override`，**无条件写基准段** —— 而同一函数里 `RUN_PRODUCTION` 已按证据所在段分发。窗口的当前 Epoch 在 `vanishing_N` 时，预热预算烧在基准段的旧 f_k 上，目标段一步不动 | **FIXED** |
| 17 | `abfe_pipeline.py:10858` | 降级条件只排除 `PROBE_REANCHOR_EPOCH`，`ANALYZE` 照样被降级 —— 而发 `ANALYZE` 的场景（6a）白纸黑字写着「绝不能因为不知道就去换 Epoch，那是在毫无根据的窗口上烧 GPU」 | **FIXED** |
| 18 | `abfe_preoptimizer.py:4880` + `3339`/`4757` ⚑ | `stale` 表**永不清除**：窗口在新段重采成功后仍留在 `stale` 里（两张表互不排斥）⟹ 插过一次 λ 之后 `view["stale_layout_evidence"]` 恒非空 ⟹ 分支 0a 的 `DONE` 与 `_evidence_status` 的 `CONVERGED` **两条路被永久封死**。加重因素：本该清旧产物的 `_invalidate_stage_window_files`（`abfe_pipeline.py:8599`）**全仓零调用点**。⚠️ **2026-09-14 已拆成 (a)/(b) 两半分别裁决，见下文《#18 裁决》** | **(a) FIXED (2026-09-16 对账：`abfe_preoptimizer.py:6640` 标记 + `:6652` `del stale[_i]`)** / **(b) WONTFIX（维护者裁决）** |
| 19 | `abfe_pipeline.py:11337`/`11399` + `abfe_preoptimizer.py:2330` | `history` 只在异常和收尾落盘，而**补帧块账正是从这份文件算的** ⟹ 掉节点后块数配额与边际增益刹车双双清零，可以重新发满 4 块 | **FIXED** |

## S2 — 路由结构性失效（分支被遮蔽 / 不可达）

| # | 位置 | 缺陷 | 状态 |
|---|---|---|---|
| 20 | `abfe_preoptimizer.py:3612` | 换 Epoch 预算预检**无条件**对 `earliest` 跑（不分动作类型），且对 `warmup_steps_left is None` fail-closed ⟹ 账本读不到的窗口被永久钉在 `RUN_PRODUCTION`，分支 1c/1d/3/3a/4/5/5a/5b/6 全部不可达；`left=40k < floor=50k` 时还会在**尚未冻结/验证**的 f_k 上开生产 | **FIXED** |
| 21 | `abfe_preoptimizer.py:3957` | 分支 4（预热预算耗尽 ⟹ 归因）**整段死代码**：`stuck` 里的窗口必然已被 1e 或 #20 的预检截走。`HALT_NO_ATTRIBUTION` 在生产里从不执行 | **FIXED (2026-09-16 对账)** |
| 22 | `abfe_preoptimizer.py:3504` | `_pick` 在 `earliest is None` 时返 `[]` ⟹ 分支 2/3/3a/4/5/5b/6 整批打掉。全窗 `UNKNOWN` 时连「生产帧没攒够」都发不出来，只能靠停滞保护退出 | **FIXED** |
| 23 | `abfe_preoptimizer.py:3379` | `phase == "TERMINAL"`（`bias_status ∈ {failed, calibrated_validation_failed}`）**全仓无任何分支处理**。它会当上 `earliest` 并挡住所有下游窗口，最后以一个与根因无关的理由退出 | **FIXED** |
| 24 | `abfe_preoptimizer.py:5965` vs `5610` | `feasible["split_tail_window"]` 问的是「**末窗**能否一分为二」（`K_tail ∈ [2lo−1, 2hi−1]`），而执行器的 `SPLIT_TAIL_WINDOW` 做的是「从 `first_untrusted_window` 起**全部**重分」。标准 23 态布局末窗 `K=4 < 7` ⟹ **这个动作永久不可行**；反之被 granted 时改动范围远大于「拆末窗」，可行性检查完全没算进作废的下游证据 | **FIXED (2026-09-16 对账)** |
| 25 | `abfe_preoptimizer.py:2371` + `4833` | `read_aggregated` 造的段级子控制器不设 `_stage_base` ⟹ 候选名被格式化成 `stage2_vanishing_3_*`，而写侧硬编码 `stage2_vanishing_*` ⟹ 两个候选全 miss ⟹ 落到 mtime 兜底。CTL-01 的版本隔离对非基准段是关的 | **FIXED** |
| 26 | `abfe_pipeline.py:10542` | 影子对账用普通构造函数而非 `for_physical_stage` ⟹ `min_n_eff_over_g_history` / `stale_layout_evidence` / `window_provenance` 只在聚合视图里产出、影子路径上恒为 None ⟹ 边际增益判据与过期证据保护**结构性关闭**。而对账的全部意义就是两边同判 | **FIXED (2026-09-16 对账)** |
| 27 | `abfe_pipeline.py:11894` | `_immutable_rewindow_step` 绕过 `_persist_inprogress_stage_result` 自己写盘，不盖 `path_version` ⟹ `stage_result_path_version_verified` 恒 False ⟹ 分支 0a 的 `DONE` 结构上不可达 | **FIXED** |
| 28 | `abfe_preoptimizer.py:4745` | `_evidence_status` 只把 `HALT_BUDGET` / `HALT_LOCAL_VALIDATION_CAP` 映到 `INSUFFICIENT_DATA`，漏了 `GLOBAL_BUDGET_EXHAUSTED` 与 `HALT_VALIDATION_BUDGET_UNREACHABLE` ⟹ 报成 `INCONCLUSIVE`，与三行之上自己写的规则（「预算耗尽**永远**是还没测够」）矛盾 | **FIXED (2026-09-16 对账)** |
| 29 | `abfe_preoptimizer.py:5884` | `tail_exempt_from_max=False` 与 `layer="execution"` **全仓无人传** ⟹ 「拆过末窗 ⟹ 豁免作废 ⟹ `HALT_LAMBDA_BUDGET_INSUFFICIENT`」分支不可达，执行层难度门（fail-closed 设计）从不评估。且两种拆窗事件分别写 `split_tail_window` / `tail_repartition`，两个都没被计数 | **FIXED (2026-09-16 对账)** |
| 30 | `abfe_preoptimizer.py:3183` | 补帧准入的 `NO_FEASIBLE_ACTION` 是**终态**，而这道闸在 `plan()` 内部、分支已经锁定了单个窗口 ⟹ 一个窗口满额就终止整跑，别的窗口和布局动作一个都没试 | **FIXED** |

## S3 — 预算两本账

| # | 位置 | 缺陷 | 状态 |
|---|---|---|---|
| 31 | `abfe_preoptimizer.py:2081`/`2098` ⚑ | `stage2_production_budget_steps` 与 `stage2_max_production_blocks_per_window` **全仓只有读侧**：`abfe_config.json` 没有、CLI 没有、`run_provenance.json` 不会出现 ⟹ 真实 run 里 `cap_known=False`，`plan()` 的生产预算闸**从未生效过**，唯一的刹车是硬编码的 4 块 | **FIXED** |
| 32 | `abfe_preoptimizer.py:3238` vs `abfe_pipeline.py:10966` | `_NON_SAMPLING` 选错了动作对：`RECALIBRATE_FK` 派发时 `probe_only=False`、会开新段跑满 `n_steps_per_window`，却被判零成本；而注释点名的 `PROBE_REANCHOR_EPOCH`（确实采样）反倒不在名单里 | **FIXED** |
| 33 | `abfe_preoptimizer.py:4661` vs `5504` vs `abfe_pipeline.py:11189` ⚑ | 准入门槛是 `left ≥ FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS[0]`（50k），执行器实际要 `learn+burn+50k` ≥ 140k（实测 290k）⟹ 控制器批准、执行器拒绝并**直接终止循环**，而不是换动作 | **FIXED** |
| 34 | `abfe_preoptimizer.py:3245` ⚑ | `CONTINUE_WARMUP` / `RELEARN_FK_EPOCH` / `INSERT_LAMBDA` / `SPLIT_TAIL_WINDOW` 全被按一整块**生产**预算收费，付不起就 `GLOBAL_BUDGET_EXHAUSTED`（终态）—— 正是这套双账要防的「拿 A 账本余额终止只花 B 账本的动作」，方向相反 | **FIXED** |
| 35 | `abfe_preoptimizer.py:2475`/`2564` | `spent` 取两份 ledger 的**较新**者、`cap` 优先取 convergence 那份**较旧**者 ⟹ 用户抬高 `max_bias_warmup_steps` 之后 `cap−spent ≡ 0`，还会据此把引擎的权威 `warmup_budget_remaining_steps` 挡掉 ⟹ **显式升档预算在控制器里永不生效** | **FIXED** |
| 36 | `abfe_preoptimizer.py:2475` | `cumulative_cap_steps == 0` 被 `or` 当成未知（而 `new_warmup_budget_ledger` 的默认值就是 0）⟹ `warmup_steps_left=None` ⟹ `all_windows_budget_exhausted` 恒假。这是 #15 的反方向 | **FIXED** |
| 37 | `abfe_preoptimizer.py:4513` | `int(_pbud.get("stage_remaining_steps") or 0) >= _need_reserve`：只对 `cap_known` 做了豁免，对「cap 已知但用量未知」的 `None` 直接 `or 0` ⟹ `IMMUTABLE_REWINDOW`（固定 λ 表下唯一对症的动作）被误判为预留不出来 | **FIXED (2026-09-16 对账)** |
| 38 | `abfe_preoptimizer.py:2987` vs `5039` ⚑ | `_production_budget_inputs["unit_used"]` 写侧是 `int(... or 0)`、值恒为 int，读侧 `if _v is None` 是**死代码** ⟹ 子窗步数读不到时静默记 0，`usage_complete` 仍为 True | **FIXED** |
| 39 | `abfe_preoptimizer.py:2930` | 单段视图的 `production_budget` 还是旧写法（`or 0`、无 `usage_complete`/`unknown_usage`、无条件算 `stage_remaining_steps`/`exhausted`），与合并视图对**同一个量**给两个答案 | **FIXED** |
| 40 | `abfe_preoptimizer.py:2327` + `abfe_pipeline.py:10786` | 块账双向错：① 每轮 `RUN_PRODUCTION` 给视图里**每个**窗口都记一行，换段导致步数变化的窗口也白占一格配额；② `PROBE_REANCHOR_EPOCH` 确实花一块却被 `it["action"] != "RUN_PRODUCTION"` 过滤掉 | **FIXED** |
| 41 | `abfe_preoptimizer.py:2334` | `same_segment_only` 模式下 `_seg != seg_of.get(i)`，对不在当前视图里的窗口 `seg_of.get` 返 `None` ⟹ 与字符串恒不等 ⟹ 边际增益刹车对**证据已残缺**的窗口静默失效 | **FIXED** |
| 42 | `abfe_preoptimizer.py:2645` ⚑ | `_rw_latest` 按 **identity 字典序**取最后一条，且**不过 `_rw_entry_is_current()`** ⟹ 可能描述另一条布局下的条目；而 `identity` 进停滞签名，会把真实推进记成 no-op | **FIXED** |
| 43 | `abfe_preoptimizer.py:2562` ⚑ | `warmup_steps_left_source` 只判 `_engine_left is not None`，引擎值被上面的 ternary 否决时仍报 `"engine:warmup_budget_remaining_steps"`。`tests/test_ctl10_warmup_budget_authority.py:56` 钉住的正是这个会撒谎的标签 | **FIXED** |
| 44 | `abfe_preoptimizer.py:3243` | `IMMUTABLE_REWINDOW` 的预留硬编码 `2 × new_ensemble_reserve_steps`，而 `_immutable_rewindow_step` 按 `len(children) × base_unit` 收费 | **FIXED** |

## S4 — 物理 / 统计判据

| # | 位置 | 缺陷 | 状态 |
|---|---|---|---|
| 45 | `ibs_engine.py:21058` | **held-out 验收在数学上恒等于 REJECT**：`ser = (uu[k] − bb − f_k[k])/kT` 里候选 f_k 只以**逐态常数**进入序列，而 `exp(-(sr − sr.min()))` 与自相关抽稀对常数平移都是不变量 ⟹ `_support(f_current)` 与 `_support(f_new)` **逐位相同**、`improved` 恒为 False。正解是用候选重建混合偏置 `V' = −kT·logsumexp_j(−(u_j − f'_j)/kT)`，而 `multi_segment_analysis.mixture_energy` 已经实现了它 | **FIXED** |
| 46 | `abfe_preoptimizer.py:2761` | `solver_eligibility`（纯样本量不够）被归进「加帧治不了」：写侧特意把它与 `top1pct_veto` 分开并注明「帧数不够 ≠ 支撑不够」，但同时又把它强制置成 `HARD_INSUFFICIENT`，而控制器的 `_support_failed` 只认后者（`_self_src` 就在手边没用上）⟹ 长 τ 的解耦端窗口被送去插 λ，而插 λ 不缩短构象慢模态的 τ_int | **FIXED** |
| 47 | `abfe_preoptimizer.py:4239`/`3479` | 两处「边际增益」判据把 `min N_eff/g`、`solver_n_decorrelated` 当成随采样**单调增**的量：分子 Kish ESS 小 N 时乐观偏高、分母 ĝ 在 N≫τ 之前单调上涨，两个偏差同向，而比较基准还取 `h[0]`（偏高最严重那点）。**与 `abfe_preoptimizer.py:3838` 自己否决「周期内按 n_eff 外推提前判死」的论证直接矛盾** | **FIXED** |
| 48 | `ibs_engine.py:20190`/`20575` | top1% 用了**已被判废**的「整帧计数」实现（生产门那一处已改成经验 CDF 线性插值，并写明旧式在 N<100 时退化成「最大单帧占比」），却复用为 N≈330–430 校准的阈值 0.35；N=40 时该量被放大 ≈2.5 倍，且 `top1_veto` 直接改写 verdict | **FIXED** |
| 49 | `abfe_preoptimizer.py:962` | 生产默认分窗名为 `arclength`、**实为等边数**：`vanishing_subdomain_ranges_from_lambdas` 只把 `lambdas.size` 传给贪心分组，不看任何度量。而 λ 表来自混合坐标 `(1−β)ŝ + β(1−λ)`、再被 `densify_lambdas_by_free_energy` 按 \|ΔF\| 插点，所以 docstring 里「λ 位置已编码热力学度规」的前提不成立。真正按 ∫g 均衡的 `partition_windows_by_metric_integral` 已实现但默认不启用 | **FIXED (2026-09-16 对账)** |
| 50 | `abfe_preoptimizer.py:5690` + `946` | 尾段重分把**全路径**分窗器喂给一个子路径：尾段恰好 23 态时命中 `VANISHING_FIXED_WINDOW_RANGES` 这张专为 λ=1 耦合端做的手工固定表（含 window-0 收窄），且 `min/max_states_per_window` 被**完全忽略**，而后置尺寸检查照样通过、无人报告替换发生过 | **FIXED (2026-09-16 对账)** |
| 51 | `ibs_engine.py:21296` | 累计 f_k 的 `span = max(C) − min(C)` 零假设期望 `≈ σ_adj·√(2K/π)` **随 K 增长**，阈值却刻意做成 K-无关；σ 又取 `ddf[argmin, argmax]`（从同一批数据里选出来的预选对 ⟹ CI 系统性偏窄）。末窗是溢出槽、K 最大，最容易被误判 `FAIL_CUMULATIVE_FK` | **FIXED** |
| 52 | `abfe_preoptimizer.py:3851` | `projected_steps_needed = T·g_hi·stride` 是**从零算起**的总步数，却与 `warmup_steps_left` 并排打印成「还需 X 步 / 剩余仅 Y 步」。判定用的是 `gcrit_budget`（已含 `frames_already`）所以结论不受影响，只是报出去的缺口被系统性夸大 | **FIXED (2026-09-16 对账)** |
| 53 | `ibs_engine.py:20154` vs `20176` | 自检里 g 与 N_eff 的帧集不一致：`_decorrelate_by_worst_target_state` 剔除 `<20 帧`的短段并取逐态**最大** g，而 `n_eff_per_state` 在**全部**帧（含被剔段）上算 ⟹ 主验收量是「全段分子 ÷ 最坏段分母」，重启多的窗口被系统性压低 | **FIXED** |
| 54 | `ibs_engine.py:20582` | 把 `subsample_series_by_autocorrelation` 返回的 `g` 落盘成 `"tau_int"`，控制器 `render` 照此打印。`g = 1 + 2τ_int`，读数被当成 τ 用会差约 2 倍 | **FIXED** |

## S5 — 记账 / 一致性（不影响结果，影响可归因）

| # | 位置 | 缺陷 | 状态 |
|---|---|---|---|
| 3 | `abfe_pipeline.py:11918`/`11527` | 🔴 **必修**（2026-09-14 从 S0 下调到本档：**归因错，不是 ΔG 错**）合并采样段用 `sorted(glob(...))` 即**字符串序**：`vanishing_10` 排在 `vanishing_2` 之前，真机见过 39 个段目录。**收窄依据**：λ 链的拼接顺序**不依赖目录顺序** —— `solve_stage_integrated` 内部有 `valid_windows.sort(key=lambda d: min(d["lambda_indices"]))`，按 λ 下标重排一次，所以**ΔG 本身不会拼错**，原来的 S0 定级过重。乱序真正污染的是两处**元数据**：① 元数据捐赠段 —— `base = dict(parts[0])` 取的是「第一段」，字符串序下会选成 `_10` 而不是基准段；② `_segment_index` 写进 `sampling_source_id` 的**归因表** —— 段号与帧的对应整体错位。⟹ 39 段的真机场景下**归因完全不可用**（说不出任何一帧来自哪个 f_k epoch），而归因正是 f_k 重标定/边际增益/停滞保护三套判据的输入。**必修结论不变。** | **FIXED (2026-09-16 对账)** |
| 55 | `abfe_preoptimizer.py:2577` | `validation_g_history` 读错嵌套层级：写侧在 `bias_warmup["validation_indeterminate"]` **内部**，控制器在 `bias_warmup` 顶层读 ⟹ 可达性预检永远只拿到单点 g，而引擎注释明写「单点在小样本下会误杀」 | **FIXED** |
| 56 | `abfe_preoptimizer.py:2428` | `warm = conv.bias_warmup or fail...` 仍**优先取 convergence**（较旧那份）：#35 的修复只落在 `spent`/`_engine_left` 两个量上，`warm` 本身一字未动 ⟹ 同一条窗口记录里新旧两种口径混用 | **FIXED** |
| 57 | `abfe_preoptimizer.py:2140` | `_read_path` 用 glob 数**所有** `path_versions/v*.json` 而非祖先链（与自己的 docstring 相反）⟹ 孤儿版本（写了文件没推指针，设计内的正常残留）永久占用插点预算；且绕过了 `lambda_path_versions._validate()` 的 `content_sha256` 校验 | **FIXED** |
| 58 | `abfe_preoptimizer.py:2861` | `skipped_windows` 混了 solver 命名空间（子窗 `window_index ≥ 10000`）与物理窗口下标 ⟹ 一个被跳的子窗会以父窗之外的身份**永久封死 DONE**；`render` 还把 10000 当窗口号打印给人看 | **FIXED** |
| 59 | `abfe_preoptimizer.py:1855` | `stage_quality_gate_failures` 对**缺阈值**和 **NaN** 都不 fail-closed（docstring 声称 fail-closed）：缺阈值静默当通过；`NaN > x` 为 False ⟹ NaN 端点 σ 给出 `converged=False` + **零归因** ⟹ 落到分支 9d 的 `NO_FEASIBLE_ACTION` | **FIXED (2026-09-16 对账)** |
| 60 | `abfe_preoptimizer.py:1884` | gate-3 归因里，只要任一失败项提到 `top1pct`，`worst_window` 就被**无条件**换成 top1% 最大的那个窗口，即使 raw-ESS 那条也失败并指了另一个窗口 —— 与紧邻上方「不许混用不同尺子」的注释冲突 | **FIXED (2026-09-16 对账)** |
| 61 | `abfe_pipeline.py:11365` | f_k 统计驳回时 `_wi = int(wins[0])`，带 `unit_id` 的动作里那是**父窗**，而被驳回的是子系综自己冻结的那份 f_k ⟹ 父窗被误封「替代候选已用掉」，真正出问题的子窗一条记录都没有；`wins` 为空时还会把 `window_idx=-1` 写进台账 | **FIXED** |
| 62 | `abfe_pipeline.py:10786`/`11404`/`11660` | 三处记账错配：降级后 `act` 变了但 `history[-1]["action"]` 和 no-op key 还是旧的；`outcome.get("status") != "DONE"` 恒真（status 从不写 `"DONE"`，DONE 是 `exit` 的值）；`_record_noop_action` 在 `windows` 为空时空循环后仍写盘 | **FIXED** |
| 63 | `abfe_preoptimizer.py:5551`/`5599`/`4799` + `1929`/`2389`/`6018` | 杂项：`segment_dirs_for_evidence` 的 `n<=1 → (None,None)` 不 fail-closed（`vanishing_1` 一旦存在就被静默重定向到基准段）；三处「什么算一个段」的枚举规则互不一致；类 docstring 声称「不改任何文件」但 `write_comparison_manifest` 写盘、且落点在 `_read_stage_result` 兜底 glob 的命中范围内；`_read_stage_result` 的 `seen` 是死变量；model B「零成本」的前提（下游还没采）在控制器分支 9b 路径上不成立（9b 排在 6b/8 之后，所有窗口都已有产物） | **FIXED** |

---

---

## #18 裁决（2026-09-14，维护者拍板）

本条拆成两半，**状态不同**。

### (a) 控制器半边 —— **`FIXED`**（2026-09-16 对账：`abfe_preoptimizer.py:6640` 的 `[审计 #18]` 标记下 `:6652` 真的 `del stale[_i]`）

`abfe_preoptimizer.read_aggregated` 里 `stale` 表**永不清除**：窗口在新段重采成功
并进了 `merged` 之后，仍然留在 `stale` 里（两张表互不排斥）⟹ 插过一次 λ 之后
`view["stale_layout_evidence"]` **恒非空** ⟹ 分支 0a 的 `DONE` 与 `_evidence_status`
的 `CONVERGED` **被永久封死**。一个已经修好的 stage 每次 resume 都会重开修复动作烧 GPU。

### (b) 执行器半边 —— `WONTFIX`（既不接上，也不删除）

`abfe_pipeline._invalidate_stage_window_files` 全仓零调用点。**不接、不删**，理由如下
（写在这里是为了下一个人不用再翻一遍）：

1. **按审计字面「在布局变更处接上它」是错的。**
   它第二阶段无条件 `os.remove` 已经烧过 GPU 的
   `dual_window_*_{energies,bias,base}.npy` + `convergence.json` + `ibs_state_*.json`，
   直接撞本仓明文规矩 —— `_reconcile_rewindow_intents` 的 docstring 白纸黑字：
   「**不删目录。** 本仓库规矩：不原地删实验产物」。
   而且它**只认基准段目录**，而自治循环的真实盘面是
   `vanishing` + `vanishing_2…N` + `vanishing_rewindow_<id>`：在循环里调它只会削掉基准段，
   段目录和子系综原样留着 ⟹ **制造不对称盘面，比不调更糟**。
   还有一个坑：不传 old/new 四个参数就是**全清**（`reuse_map` 空 ⟹ 第二阶段删光该 stage
   全部窗口产物）。

2. **它要防的失效模式已经被三道语义身份门各自挡住了**，剩下的只是磁盘占用、不是正确性：
   · 控制器读证据侧 —— `read_aggregated._layout_matches`（按 `LAMBDA_GRID_DECIMALS` 栅格
     逐项比 λ，对不上剔成空壳占位）；
   · 采样 resume 侧 —— 缓存门校验 `convergence` 里的真实 λ，不匹配判无效重采；
   · 求解侧 —— `load_ibs_window_outputs_from_dir` 对 `_grid(lambdas_vdw) != _grid(lv)` 直接
     `raise`（fail-closed），且窗口数变少时残留的高 idx 文件进不来
     （loader 只遍历 `range(len(ranges))`）。

3. **为什么也不删。** 它是 `non_mutating_v1` 删掉 overlap autorepair 变异循环之后留下的
   孤儿，本来该删；但爆炸半径偏大：函数本体 139 行、`tests/test_resume_reuse_contracts.py`
   里 7 个测试函数 + 一个专用 `_make_fake_pipeline` helper、`tests/test_non_mutating_policy.py`
   的 mutator 名单一条、`lambda_path_versions.py` 与 `abfe_pipeline.py` 两处文档指路。
   为清一个**不在任何执行路径上**的死函数动这么多测试，收益不抵风险。**保留。**

### 🔑 这条规矩比这条 bug 本身值钱（给下一个人）

> **「某函数零调用点」不等于「忘了接，接上就好」。**
> 接之前先问两件事：① 接上去会不会违反已成文的规矩；
> ② 它要防的失效模式是不是已经被别的机制挡住了。
> **这一条两个方向都栽过。**

---

## S6 — 审计后新增（#64 / #65，不在原 63 条里）

| # | 位置 | 缺陷 | 状态 |
|---|---|---|---|
| 64 | `ibs_engine._resume_cached_window_gate_status` + `run_all_windows` | 🔴🔴 **win0 空转：每涨一次补帧目标，整条前置链白跑一遍**（**唯一一条正在实时烧 GPU 的**）。详见下文《#64 正文》 | **FIXED（P1+P2）** / P3 不做 |
| 65 | `abfe_preoptimizer.write_comparison_manifest` vs `stage2_ab_report.py` | 落盘名改了、唯一消费者在另一个文件里没跟 ⟹ **整份 A/B 报告全空且不报错**。详见下文《#65 正文》 | **FIXED** |

---

## #64 正文 —— win0 空转重跑前置链（2026-09-14 新增）

### 缺陷

`ibs_engine._resume_cached_window_gate_status` 是 **15 个合取项**，其中 14 项是**身份**，
只有第 15 项 `early_stop_ok` 里塞了**两件不同的事**：

* **(15a) 步数预算够不够** —— 缓存的 `n_steps_per_window_effective < effective_target_steps`
  ⟹ 判废。**无条件生效、与 early stop 毫无关系**，是纯粹的**进度**量；
* **(15b) early-stop 身份** —— 只在 `early_stop_triggered is True` 时才走的 `elif`。

而这道门只有**二元输出**：`usable=True` ⟹ `continue` 整窗跳过；`False` ⟹ 走完整窗流程。
**缺「身份全对、只是步数没到 ⟹ 增量续采」这第三档。**

于是补帧目标每涨 250k，就把
**EM + 阶段 2 dt 测试 + Boresch 爬坡 + 冻结 burn-in + 只读复验**
整条前置链重跑一遍，然后 `_try_load_main_window_checkpoint` 立刻把
坐标 / 速度 / 盒子 / 积分器 RNG **全覆盖回上次生产结束那一刻** ——
**整条链的动力学产物逐字被丢弃**。帧没丢，烧掉的是那段前置链。

### 🔑 两条反直觉结论（防止下一个人第三次走错路）

1. **目标步数本来就不在身份键里，而且不该进去。**
   `_stage_window_sampling_identity` 已经显式 `config.pop("n_steps_per_window")` ——
   这是**故意**把它摘出来留在这道门里当**进度**判据的。
   把它「从身份里拿掉」是**错误处方**：那会让 750k 的缓存被当成满足 1000k 目标而**整窗跳过**，
   **抬预算等于白抬**。缺的是**第三档输出**，不是少一个键。

2. **那句 `[WARN]` 文案本身是误导源。**
   它写死「缓存是 early stop 提前停止产出的短样本」，但**触发它的是 (15a)**，
   而 `enable_early_stop` **默认是 False**。这句话先后把两个独立的排查带偏。

### 修法

| 档 | 内容 | 状态 |
|---|---|---|
| **P1** | 拆出第三档 `incremental_topup_only`；**`usable` 语义逐字不变**（`step_budget_ok` 仍在它的与链里，reason 阶梯顺序照旧 ⟹ 步数不足的窗口永远不会被跳过）。WARN 按两种原因**分两条文案**。 | **已落盘** |
| **P2** | top-up 路径上**跳过那次注定会被丢弃的冻结复验**。守卫三条**缺一不可**：`topup_only` + `skip_warmup_entirely` + `restored_from_production_checkpoint`。checkpoint 因任何原因装不上 ⟹ 第三条为 False ⟹ 退回今天的完整 burn-in + 复验，一步不少。 | **已落盘** |
| **P3** | 改用 production checkpoint 省掉 EM / dt 测试 / Boresch 爬坡 | **不做** —— 只是每轮多花一次 EM，**正确性不受影响** |

### ⚠️ P2 是本批**唯一一处主动移除既有保护**

移除的是 `IBS_BIAS_PROTOCOL_VERSION=7` 立的那条规矩：
**「缓存的 `bias_converged=True` 不能证明本次新建 Context 的当前构型已在该固定偏置下平衡过」。**

**为什么这次可以移除**：那条规矩针对的是「EM 出来的**全新**构型」。
而 top-up 恢复的是**上一段生产结束那一刻**的坐标 / 速度 / 盒子 / RNG ——
它本身就是在**同一份冻结 f_k** 下采出来的，且**已经通过过一次冻结验证**。

**代价侧（不移除会怎样）**：现有的「每轮重赚一次 loose gate PASS」在**单调吃 warmup 预算账本**。
真机现场 **195000 / 955000，其中 validation 占 150000**；按 +250k 再走几轮，
**必然**把窗口逼进 best-effort 或判失败 —— 一个**纯粹由重复验证造成的假失败**。

---

## #65 正文 —— 改了落盘名，消费者在另一个文件里没跟（2026-09-14 新增）

`Stage2RepairController.write_comparison_manifest()` 因为 **#63** 把落盘名从
`stage2_comparison_manifest.json` 改成 `controller_comparison_manifest.json`
（**理由正当**：旧名会被 `_read_stage_result()` 的兜底 glob `stage2_*.json` 吃到）。

但**唯一的消费者 `stage2_ab_report.py` 在另一个文件里**，还在读旧名 ⟹ 现状是
**每次跑都生成一份新名的、然后去读一个永远不存在的旧名路径**，
**整份 A/B 报告全空、且不报错**。

已修：`stage2_ab_report._MANIFEST_NAMES` 新旧两级回退。

### 🔑 这条规矩比这条 bug 本身值钱（与 #18 那条并列）

> **改落盘产物的文件名或键名时，消费者几乎总在另一个文件里，而改动方只看自己的文件。**
> 改之前先 `grep` 出**全部**读点；改之后在**写侧 docstring 里点名消费者**。
>
> 尤其危险的是「**键名不变、值变了**」这一类（`tau_int`）——
> **读点不跟改就静默差一倍，连异常都不会抛。**

**这一天之内同一形状已经出现三次**：
① `tau_int` 值变（#54）；② manifest 改名（#65）；③ `skipped_windows` 拆键（#58）。

## 已核实**不是** bug（省得下一个人重查）

- **N_eff/g 没有重复除 g**：`n_eff_per_state` 的 Kish `(Σw)²/Σw²` 明确算在**未抽稀**帧上，再除逐态 `g_k`，只除了一次。
- **`RECALIBRATE_FK` 不引入 MBAR 偏置**：新段作为独立采样态进 S+K MBAR，交叉能量由 `mixture_energy` 在**全部**帧上重建 —— 这正是「帧来自两份不同偏置却要合并重加权」的正确解法。残留只有 O(1/N) 的高阶效应。
- **插点是 pilot 度量下的热力学中点**，不是 λ 线性中点；`partition_criterion` 贯穿「定 n / 选边 / 定位」三处，与初始布局同度量；插点和重分之后都重新跑 `validate_single_shared_boundary_ranges`。
- **`[2lo−1, 2hi−1]` 可拆区间与 `2hi−1` 溢出上界在两处确实是同一份规则**（`feasible_repair_actions` 的 `k_tail + n_ins > 2hi−1` 与 `insert_lambda_in_failed_ibs_window` 的 `(ranges[-1][1] + n_needed) − ranges[-1][0] > 2hi−1` 算术等价），无 off-by-one。
- **`∫|⟨dU/dλ⟩|du` / `∫√g dλ` / `∫g dλ` 三者命名清楚、没有互相冒充**；离散化全是梯形，λ 降序 ↔ u 升序的方向处处一致。
- **`record_tail_repartition_version` 写的 `event.kind` 与 `_read_path` 读的键匹配**。
- **`relearn_epoch_used` 两侧 key 口径一致**：`for_physical_stage(run_dir, "vanishing")` ⟹ `segment_index=None` ⟹ `checkpoint_dir == path_checkpoint_dir`。
- **`seen`/`escalated`/`last_sig` 用的是同一个 3 元组**（类型注解写成 2 元组只是过期，不影响运行）。
- **「求解器 vs 自检去相关帧数差 2–8 倍」不是谜**：求解器按 **f_k epoch**（`sampling_source_id`）切段、自检按**重启**（`production_segments`）切段，后者的序列横跨两个 f_k epoch、中间有一个阶跃 ⟹ ĝ 被撑爆。来源就在 `ibs_engine.py:21845` 与 `20154` 这两行，不是「尚未证明」。
