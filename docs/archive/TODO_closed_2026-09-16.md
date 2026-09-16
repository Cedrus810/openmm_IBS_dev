# 已关闭条目（2026-09-16 对账）

[回到当前清单](../TODO.md)

> **这份归档的由来与前几份不同。** 前几份是「修完一条、搬走一条」；这一份是
> **一次逐条回源码的对账**：`docs/TODO.md` 里挂着一批 `- [ ]`，它们描述的缺陷
> 在工作树里**已经修好了**，只是没人回来打勾。对账方法与证据逐条写在下面。
>
> **为什么这件事本身要记一笔**：本仓已经有两份文档
> （本文的来源 `TODO.md` §1、以及 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](../CONTROLLER_BUDGET_AUDIT_2026-09-14.md)）
> 同时出现过「状态栏落后于源码」。后果不是漏修，是**重复排查** —— 下一个人看到
> 一屏 `- [ ]` 会以为控制器还整条没接上，于是把已经修过的地方再查一遍。
> 这和本仓最贵的那个复发模式（「同一个不变量的 N 份实现」）是同一个形状，
> 只是发生在文档上：**「某条缺陷修没修」这个事实也有两份实现（源码、状态栏），
> 而没人定过谁是权威。** 权威是源码。
>
> ⚠️ **本文只对账「代码里那处缺陷还在不在」，不对账「修得对不对」。**
> 下面每一条都仍然是**真机零验证**（`AUDIT-S2-03` 仍开着）。

## 对账方法

1. **`[审计 #N]` 标记**：`CONTROLLER_BUDGET_AUDIT_2026-09-14.md` 定的约定是
   「源码里有 `[审计 #N]` 标记且落在真实代码改动上 ⟹ 该条已修」。
   实测 `grep -ohE '审计 #[0-9]+' abfe_preoptimizer.py abfe_pipeline.py ibs_engine.py
   runabfe.py lambda_path_versions.py abfe_config.json | sort -u`：
   **#1–#36、#38–#64 全部在场**（缺 #37、#65）。
   · **#65** 文档自己已标 `FIXED`；
   · **#37** 无标记，但缺陷本身已不在 —— `abfe_preoptimizer.py:4635` 现在是
     `_left = _pb.get("stage_remaining_steps")` + `if _left is not None and int(_left) < _cost`，
     紧邻注释逐字写着「先前 `or 0` 把 None 当成 0 ⟹ 一律拦死，等于"未知 = 耗尽"」。
   ⟹ **审计 65 条全部落地**，那份文档的 `OPEN` 状态列已于同日整列纠正。
2. **`[CTL-NN]` / `[BUD-NN]` 标记**：同法。实测在场的有
   `BUD-03 BUD-05 BUD-07 CTL-01~09 CTL-11 CTL-12 CTL-13 CTL-14`。
3. **没有标记的，直接回源码看缺陷在不在**（`BUD-01` / `BUD-02` / `BUD-04` /
   `AUDIT-S2-01` / `DECORR-01`），证据写在各条下面。

---

## A. 逐条对账结论

| 条目 | 原判 | 2026-09-16 实测 | 证据 |
|---|---|---|---|
| `DECORR-01` | 两份实现、谁是权威没定 | **已裁决 + 已接线** | `abfe_preoptimizer.py:3901/3931/4824`、`ibs_engine.py:20920/21271/21340/21386/22658` |
| `AUDIT-S2-01` | `decide()` 从不发 `SPLIT_TAIL_WINDOW` | **前提证伪**：两处发得出 | `abfe_preoptimizer.py:5814`（`_feas_split`）、`5955`（`_can_split_skew`）；且 `TODO.md` 自己的 09-14 重放表里 `cyclod_ligand2/rep1` 拿到的就是 `SPLIT_TAIL_WINDOW[5]` |
| `BUD-01` | cap 键全仓只有读侧 | **已补齐**（config + preset + CLI），默认 `null` = 上限未知 | `abfe_config.json:46/47`、`runabfe.py:333/334/375/376/384` |
| `BUD-02` | `view["path_version"]` 不存在 | **已补**：`read()` 返回值里有这个键 | `abfe_preoptimizer.py:4018` `"path_version": (path or {}).get("version")` |
| `BUD-03` | 块账换段即清零 | **已修**（审计 #40/#41） | `abfe_preoptimizer.py:4021` 注释 `[BUD-03] 硬上限用的是跨段累计的那本` |
| `BUD-04` | 补帧两道刹车都不触发 | **已修**：块账拆成四个平键并真的有值 | `abfe_preoptimizer.py:4019–4024`（`by_window` / `by_unit` / `total_by_window` / `total_by_unit`） |
| `BUD-05` | 同一个「预热余量」三套 unknown 语义 | **已修**（含原先仍 `OPEN` 的 #37） | `abfe_preoptimizer.py:6869`（`None if ... is None else int(...)`）、`6879`（`all_windows_budget_exhausted` 要求账完整）、`4635`（#37） |
| `BUD-07` | 跨段预热预算继承可能只认基准段（未验证） | **已修**，不再是「未验证」 | `ibs_engine.py:8218/8242/8283` 三处 `[BUD-07]` 标记 |
| `CTL-11` | converged 了也判不出 `DONE` | **已修**：`DONE` 提到分支 0a | `abfe_preoptimizer.py:4718` `0a) [CTL-11] stage 级判据已经通过 ⟹ DONE，不许再修` |
| `CTL-12` | 12 个动作写台账、只有 3 个读 | **已修**：`plan()` 里统一挂一道 `_NOOP_GUARDED` | `abfe_preoptimizer.py:4502–4545`，覆盖 `RUN_PRODUCTION`/`INSERT_LAMBDA`/`SPLIT_TAIL_WINDOW`/`IMMUTABLE_REWINDOW`/`RELEARN_FK_EPOCH`/`PROBE_CANDIDATE_FK`/`PROVISIONAL_PRODUCTION`（`ANALYZE` 刻意不挂，理由写在原处） |
| `CTL-13` | 补帧准入只查 `windows[0]` | **已修** | `abfe_preoptimizer.py:4438` 注释 `先前是 _tgt_w = int(windows[0])` |
| `CTL-14` | `RELEARN_FK_EPOCH` 两处 `break` 不写 `outcome` | **已修** | `abfe_pipeline.py:11992`、`12027` 两处 `[CTL-14] 终止必须同时写 outcome` |
| `CTL-15`~`19` | —— | 原文已是 `[x]`，按规则 2 搬走 | —— |
| `BM-01`/`02`/`03`/`05` | —— | 原文已是 `[x]`（2026-09-16 当天修完），按规则 2 搬走 | —— |
| `DATA-01` | —— | 原文已是 `[x]`（维护者选清空重跑），按规则 2 搬走 | —— |

> ⚠️ **仍然开着、没有搬走的**（别以为这一轮把 §1 清空了）：
> `AUDIT-S2-02`（`_decide_once` 现在 **1795 行 / 63 个 return**，比原条目记的 1319/45 **更大**）、
> `AUDIT-S2-03`（真机零验证）、`BUD-06`（已改判为待拍板，理由见 `TODO.md`）、
> `IDENT-01` / `BM-04` / `DATA-02` / `DATA-03`（四条待拍板）、`S2-E`。

---

## B. 原文留档

以下是搬出来的原始条目全文，一字未改。**读它们是为了知道当时踩了什么坑，
不是为了重新排查 —— 上表已经逐条核过缺陷不在了。**


### DECORR-01（去相关帧数两份实现）

- [ ] **DECORR-01 「去相关帧数」有两份实现，口径差 2–8 倍。**
  逐窗自检（`window_self_support_check`）与全局求解器各算一次：`cyclod_ligand2/rep2`
  实测 win3 自检 **56** 帧、`sufficient=True`，求解器报 **9 / 7** 帧并把它**跳过**；
  win0 自检 21 帧 (g=24.5)、求解器报 9。两个数**门着不同的东西**（自检喂控制器的
  `self_sufficient`，求解器的决定跳不跳），于是控制器会认为一个被跳过的窗口"帧数
  够"。这是"同一个不变量的 N 份实现"的又一例。**谁是权威没定** —— 定之前别把任何
  一侧改成向另一侧看齐。

> `CTL-01` ~ `CTL-10`（2026-09-14 第二轮静态复核的**全部**未收口项）
> ⚠️ 同日第三轮的六路并行审计把这一片重新完整扫了一遍，见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](../CONTROLLER_BUDGET_AUDIT_2026-09-14.md)（#20/#22/#23/#25/#27/#30 等）。
> **2026-09-14 当天十条全部关闭**，逐条改动与判据见
> [CHANGELOG](../CHANGELOG.md) 同日那条（很长，是本组的完整记录）。
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

### AUDIT-S2-01（`SPLIT_TAIL_WINDOW` 从不发）

- [ ] **AUDIT-S2-01 `SPLIT_TAIL_WINDOW` 声明了、执行器也认，但 `decide()` 从不发。**
  它在 `ACTIONS` 里、`_run_stage2_autonomous` 有对应分支，但现在所有发它的位置都被
  `_tgt_is_tail` / `_can_split_skew` 之类的条件收窄掉了。
  **可能是对的**（`CTL-07` 正是在修"中间窗失败却拆末窗"），**也可能收得太死** ——
  末窗真的溢出到可拆区间时还有没有路径走到拆窗？
  ⚠️ **需要维护者拍板**，不要自行放宽：放宽的方向正是 `CTL-07` 刚修掉的那个错。
  判据建议：构造一个"末窗 K 落在 `[2lo−1, 2hi−1]` 且末窗自己是最差窗口"的盘面，
  看 `decide()` 给不给 `SPLIT_TAIL_WINDOW`；给不出就是收过头了。

### BUD-01 ~ BUD-05、BUD-07（预算系统）

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
  ✅ **已被审计 #31 覆盖并修复**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](../CONTROLLER_BUDGET_AUDIT_2026-09-14.md) **#31**）：`stage2_production_budget_steps` / `stage2_max_production_blocks_per_window` 已在 **config + preset + CLI 三处**补齐，默认值严格等于原硬编码兜底 ⟹ 行为逐位不变；前者默认 **`null` = 上限未知，不是 0**（控制器写死「未知时不拦」），正是为了不触发 `BUD-06` 那个误判。`max_path_insertions` **刻意只补 CLI**（它进 stage 协议指纹，给默认值会让所有既有 run 的 Stage 1/2 结果缓存全部失配重跑）。**`BUD-06` 本身未被审计覆盖，仍然开着。**

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
  ✅ **已被审计 #40/#41 覆盖并修复**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](../CONTROLLER_BUDGET_AUDIT_2026-09-14.md) **#40** / **#41**）：#41 修的正是`same_segment_only` 下 `seg_of.get(i)` 对不在当前视图里的窗口返 `None`、与字符串恒不等 ⟹ 刹车静默失效；#40 修的是块账双向错（每轮给视图里**每个**窗口白记一行 / `PROBE_REANCHOR_EPOCH` 确实花一块却被过滤掉）。

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
  ✅ **已被审计 #8/#30/#31/#40/#41 覆盖并修复**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](../CONTROLLER_BUDGET_AUDIT_2026-09-14.md) **#8**「子窗两道刹车同时失效」、
  **#30**「块数满额是路由信号不是终态」、**#31** 硬上限进 config/CLI、**#40/#41** 块账双向错）。

- [ ] **BUD-05 同一个「预热余量」有三套 unknown 语义。**
  `warmup_steps_left` 读不到（`None`）时：`per_window_budget_remaining` /
  `all_windows_budget_exhausted` 用 `int(... or 0)` ⟹ **0 = 耗尽**；
  `_epoch_validation_unaffordable` ⟹ **不可行**（fail-closed）；
  `decide()` 分支 1e 的 `left is not None and int(left) <= 0` ⟹ **有钱**。
  实测 cyclod_ligand1/rep3 win4 的 ledger 确实读不到，在第一处被显示成 `0`。
  这是本文上面那条规矩「未知不是零，两个方向都不是」的再次违反。
  ✅ **已被审计根因 ② 覆盖并基本修复**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](../CONTROLLER_BUDGET_AUDIT_2026-09-14.md) **#15/#36/#38** 已修；**#37 仍 `OPEN`** ——「cap 已知但用量未知」那一路还在 `or 0`）。

- [ ] **BUD-07（未验证，查到一半被叫停）跨段预热预算继承可能只认基准段。**
  `ibs_engine.inherit_warmup_ledger_across_segments` 靠 `base_checkpoint_dir_for()` 找
  「上一层」，而那个函数的锚点是 `path_current.json` —— **只有基准目录有它**。
  所以 `checkpoints/segment_3` 找回的是 `checkpoints/`，**不是 `segment_2`**：
  第 3 个 Epoch 继承基准段的消耗、丢掉段 2 烧掉的量。
  形状与已归档的「多 Epoch 链永远从基准段重学 f_k」一致（那条当时的结论是「第一次换
  Epoch 完全正确，连换两次才踩到」）。⚠️ **只读了函数、没有验证**，需要一个有
  `segment_3` 及以上的 run 对账。

### CTL-11 ~ CTL-19（控制器第三轮复核）

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
  ✅ **部分被审计覆盖**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](../CONTROLLER_BUDGET_AUDIT_2026-09-14.md)）：**#27** 已修（`_immutable_rewindow_step` 绕过统一入口写盘、不盖 `path_version` ⟹ `stage_result_path_version_verified` 恒 False ⟹ 分支 0a 的 `DONE` **结构上不可达**，这是 `DONE` 判不出来的第二条封锁）；**#18(a) 仍 `OPEN`** —— `stale` 表永不清除 ⟹ `stale_layout_evidence` 恒非空 ⟹ `DONE` / `CONVERGED` 同样被封死。**三条封锁要一起解开，只解一条仍然判不出 `DONE`。**

- [ ] **CTL-12 no-op 台账写 12 个动作，只有 3 个动作会去读。**
  执行器 `_record_noop_action` 对**任何**动作都记账（通用盘面指纹比对），但 `decide()` 里
  `_is_noop()` 只在四个位置被调用、覆盖三个动作：`CONTINUE_WARMUP`、`RECALIBRATE_FK`（两处）、
  `PROBE_REANCHOR_EPOCH`。**`SPLIT_TAIL_WINDOW` / `INSERT_LAMBDA` / `RUN_PRODUCTION` /
  `IMMUTABLE_REWINDOW` / `ANALYZE` 全都不读。**
  后果（旧代码日志实测，形状在当前代码里未变）：cyclod_ligand1/rep2 与 cyclod_ligand2/rep1
  都死在同一条路上 —— `SPLIT_TAIL_WINDOW` 连发 4 次、执行器每次都打了
  「记为当前盘面上的 no-op」、控制器每次都看不见，最后由停滞保护给出 `NO_FEASIBLE_ACTION`。
  ✅ **同形状的子窗半边已被审计 #9 覆盖并修复**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](../CONTROLLER_BUDGET_AUDIT_2026-09-14.md) **#9**：执行器写 `f"{{action}}:unit:{{uid}}"` 而 `_is_noop` 只查 `f"{{action}}:{{idx}}"` ⟹ 全仓无人读）。**「12 个动作写、只有 3 个动作读」这一半未被审计覆盖，仍然开着。**

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

### BM-01 / BM-02 / BM-03 / BM-05（2026-09-16 benchmark 报错分类）

- [x] ~~**BM-01 局部动作的执行器去载全路径。**~~ **2026-09-16 已修。**
  位置：`abfe_pipeline.py:13360`（`_recalibrate_f_k_and_resample_segment`）。
  `only_windows` 的过滤原来在 loader **之后**的循环里，而 loader 对每个载入的窗口
  做 fail-closed 布局校验 ⟹ 一个只针对窗口 3 的动作崩在窗口 4 上：
  `动作=PROBE_CANDIDATE_FK 窗口=[3]` → `ValueError('窗口 4 状态数与 window_ranges 不符')`。
  真机 7 个 run 同签名（brd4_ligand1/rep1、jnk1_ligand1/rep2、jnk1_ligand2/rep1-3、
  cyclod_ligand3/rep2、p38_ligand1/rep3）。改成在**载入时**把 `only_windows` 之外的
  窗口一并排除；`records` 内容与语义逐字不变（重锚节奏逻辑读它）。
  判据：`test_layout_change_does_not_crash_the_loop_2026_09_16.py::test_recalibrate_only_loads_the_windows_it_acts_on`
  （`only_windows=[1]` ⟹ 排除集合 `{0,2,3}`）+ `…::test_no_only_windows_still_loads_the_whole_path`。

- [x] ~~**BM-02 停滞保护的降级绕过 `decide()` 的全部可行性守卫。**~~ **2026-09-16 已修。**
  位置：`abfe_pipeline.py:11465`（`_run_stage2_autonomous` 的停滞保护）。
  它直接改写 `act = "PROBE_REANCHOR_EPOCH"`，而 `decide()` 里的 1d-0（布局过期）、
  1e（预热进不去）、`TERMINAL` 分流全在上游，一条也没走。真机 jnk1_ligand2/rep1：
  `RUN_PRODUCTION[3]` 被 `LOCAL_VALIDATION_CAP` 连弹 3 次 ⟹ 降级成
  `PROBE_REANCHOR_EPOCH[3]` ⟹ win3 的帧正是插 λ 之后的过期布局 ⟹ 炸穿主循环。
  已加闸：目标窗口在过期集合里 ⟹ 不降级，如实 `NO_FEASIBLE_ACTION`，
  并记 `history[-1]["escalation_blocked_by"]`。
  ⚠️ 顺带把「产物是不是过期布局的」收成**一份实现**
  `Stage2RepairController.stale_layout_windows()`（`abfe_preoptimizer.py:6158`），
  `decide()` 的 1d-0 改成调它 —— 这个判据有两个消费者，而 `ibs_engine` loader 对
  同一件事是 fail-closed 抛 `ValueError`，两边一分岔就是崩溃。
  判据：`…::test_the_escalation_gate_actually_sits_before_the_downgrade`（AST 检查闸的
  位置在降级赋值之前，沿用本仓对这个大循环的既有测法）+ `…::test_stale_layout_windows_is_one_implementation`。

- [x] ~~**BM-03 插 λ 能造出拆不开的末窗，且非法布局在 fail-closed 之前已落盘。**~~
  **2026-09-16 已修（表层；死局本身见 `BM-04`）。**
  真机 `cyclod_ligand3/rep1` 的 `path_versions` 末窗 K：v1=6 → v2=7 → v3=8 → **v4=9**，
  而 `stage2_window_max_states=8`。`append_version` 在 anchor 检查**之前**
  （`abfe_pipeline.py` 插 λ 分支 vs `_legalize_tail_window`），所以 v4 那个非法布局
  **已经写进版本链** —— 后果不是"这一轮崩掉"，是 resume 读到它照样合法化不了，
  这一跑再也走不出来。
  两道都加了，缺一不可：
  · 可行性侧 `abfe_preoptimizer.py:4179`（`feasible()`）——
    插完末窗越过 `hi` 且取不到 tail anchor ⟹ `insert_lambda` 判不可行（回答"该不该发"）；
  · 执行器侧 `abfe_pipeline.py:11802` —— 可落性在 `append_version` **之前**判（回答"发了能不能落"）。
  ⚠️ 闸只在 `末窗K+n_insert > hi` **且** 无 anchor 时触发：cyclod_ligand3/rep1 的
  前两次插点（6→7、7→8）照样放行，挡住的正是把布局搞成非法的第三次。
  判据：`…::test_insert_is_infeasible_when_it_would_strand_the_tail` +
  `…::test_insert_stays_feasible_when_the_tail_has_room`（后者钉的是**不许误伤**）。
  ⚠️ 顺带记一条口径：末窗吸收溢出自己的 fail-closed 门槛是 `K > 2·hi−1`（=15），
  而合法化门槛是 `K > hi`（=8）—— **8..15 是一段"吸收得进、拆不出来"的死区**，
  `BM-03` 只是不让人再走进去，死区是 `BM-04` 的事。

- [x] ~~**BM-05 两处会把排查引向错误对象的记账 bug。**~~ **2026-09-16 已修。**
  ① `ibs_engine.py:22526`（`solve_stage_integrated`）：`w_idx` 是
  `enumerate(valid_windows)` 的**列表位置**，而 `valid_windows` 只含被载入的窗口。
  部分段分析（`[部分窗口段] 本段只采了窗口 [3]`）里它是 0，于是 12 条
  `窗口 {w_idx}` 的日志把物理 win3 报成"窗口 0" —— 真机 brd4_ligand1/rep1 就是这么
  把排查引到错误窗口上的（第一轮归因错 4 条里有 1 条源于此）。结构化字段一直是对的
  （都走 `source_window_idx`）。修法是**在源头改名**：列表位置叫 `_list_pos`，
  `w_idx` 从此就是物理窗口号 —— 一次改对 12 条消息，而不是逐条改 f-string。
  ② `abfe_preoptimizer.py:2706`（`Stage2RepairController.__init__`）：`_explicit` 早就
  滤了 `None`，`_disk`（`run_provenance.json` 的 config，是 argparse 完整命名空间落盘的）
  **没滤**。实测 13 个 run 的 provenance 每份都带 17~18 个 `None` 键。今天不炸只是因为
  那几个键恰好没走 `int()`；09-15 `cyclod_ligand1/rep1` 的
  `TypeError: int() argument … not 'NoneType'` 就是同一个形状。
  **「未知」只有一种表达：缺键。**
  判据：`…::test_null_in_run_provenance_is_treated_as_absent`。

> （`BM-A`「这些不是 bug，别再查」是 **live 清单**，不是已关闭条目 ——
> 它留在 [../TODO.md](../TODO.md) §1，没有搬到这里。）

### DATA-01（`cyclod_ligand2/rep3` win4 manifest 与 f_k 对不上）

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

---

## C. 同日第二批：4 条「与源码不符的陈述」+ 1 条子项关闭

对账做完之后又扫了一遍**陈述**（不只是 checkbox），抓到 4 条。
**它们不是待办，是 `TODO.md` 正文里写着的、与源码不符的话** —— 危害比挂着的 `- [ ]` 大，
因为没有 checkbox 提示读者去核。

| 原文 | 实测 | 处置 |
|---|---|---|
| 「不许回退的约定」：**只有三个真终态** `DONE`/`GLOBAL_BUDGET_EXHAUSTED`/`NO_FEASIBLE_ACTION` | `abfe_preoptimizer.py:2553 TERMINAL_EXITS` 实际 **9 个**（多出 `DONE_UNTRUSTED`、`HALT_INVALID_INPUT`、`HALT_EVIDENCE_CONTRADICTS_DONE`、`HALT_LAMBDA_BUDGET_INSUFFICIENT`、`HALT_FRAMES_ADMISSION_CAP`、`ANALYSIS_COMPLETE_PRECISION_UNMEASURED`） | 改成**指向 `TERMINAL_EXITS`**，不在文档里维护第二份清单 |
| `AUDIT-S2-03`：**控制器真机零验证** | 39 rep 真上过 GPU，9 个出结果 | 改成「**有真机，但没有一次以 `DONE` 收口**」+ 判据 |
| `REL-03`：**到目前为止全部是 CPU / 静态验证，零 GPU 复验** | 同上 | 改成逐子项标状态；③ 关闭 |
| `REL-04`：**52 处改动无一上过 GPU** | 它们在主线里，39 rep 全执行过 | 改成「**跑过，但没有一处被单独复验**」 |

> 🔑 **最危险的是第一条，理由值得单独记：** 它属于「不许回退的约定」——
> 那一节是拿来**否决别人改动**的。一条过期的约定不会像过期的 `- [ ]` 那样只是浪费时间，
> 它会**主动打回正确的改动**：照旧清单去审，6 个合法终态会被当成违规。
> ⟹ **约定类文字比待办类文字更需要对账**，且**能指向源码就别抄一份到文档里**。

### 已关闭的子项：`REL-03` ③ 两条腿同进程的 `pipeline.log` 分离

原条目要求「已用最小复现验证，**未在真实两腿运行上确认**」。
2026-09-16 在 benchmark 的真实两腿运行上确认：`p38_ligand1/rep3` 的
`pipeline.log`（134 KB）与 `solvent_leg/pipeline.log`（51 KB）独立存在；
复合物腿日志里 10 处命中「溶剂腿」全部是**一条 WARN 文案自带的词**
（`[独立端点段] 未启用 … 该路径在溶剂腿上被论证为…`），**不是串日志**。
⟹ ③ 关闭。**①②（residual 臂门读数、`frozen_ll_pairs` 断言）仍开着**，留在 `TODO.md`。

### 顺带修掉的两处排版

- `TODO.md` 归档表第 5 行掉了 `> ` 前缀 ⟹ 表被劈成两半（**旧 bug**，非本次引入）；
- 优先级表第 1/2 行 530 / 362 字符（本次对账时自己写长的）⟹ 压到 ~175。

---

# D. 维护者拍板：`BM-04` / `DATA-02` / `DATA-03` 三条同日关闭（2026-09-16）

> 这三条覆盖 benchmark 里 **7 个走不动的 run**。三条是同一件事的三面，所以一起定。

## 🔑 本次确立的决策原则（比三条结论本身更重要）

> **「算对之前，沉没成本一律不计。」** —— 维护者原话：
> 「再计算正确之前，所有沉没成本都是可以无视的」。
>
> 代理在本轮**两次**把已有缓存的价值算高了，两次都被纠正：
> 一次是建议「别为 2 个 run 作废 39 个 run 的窗口缓存」，
> 一次是把 `PRESET_CONFIGS` 改动的半径估成「39 个 run 从头跑」。
> 实际上：那批缓存里**只有 2 个 run 是修复后代码产出的**（`p38_ligand1/rep2,rep3`），
> 其余 7 个完成的 run 全是旧码、本来就不可引用；而 30 个未完成的 run 无论如何要重做。
> **一个还没被证明算得对的结果，它的缓存没有保护价值。**
> ⟹ 今后同类权衡**先问「这条路径算得对吗」，再问「重算要多少 GPU」**，顺序不能反。

## 拍板前先堵上的一个缺口（否则决定 1 会落空）

`stage2_first_window_max_states` **此前从未在任何一次真实运行里生效过**。四条路全断：

| 来源 | 有没有 |
|---|---|
| `PRESET_CONFIGS` 三个预设（`test`/`production`/`high_accuracy`，各只有 6 个键） | ❌ |
| 14/14 benchmark config（`abfe-benchmark/openmm_IBS/configs/*.json`） | ❌ |
| CLI 开关 | ❌ **不存在** |
| `abfe_config.json`（= 4） | ✅ 有，**但它不会被自动加载**（`--config` 的 argparse 默认是 `None`），且 `runabfe.py:3302` 的合并基底是 `PRESET_CONFIGS[preset]`、**不是**这份文件 |

缺席时 `first_cap is None` ⟹ 键不发出 ⟹ `abfe_preoptimizer` 默认 `first_window_max_states=None` ⟹ **无 cap**，win0 照样能拿 8 个态。

> 📌 **留下的规矩**：`abfe_config.json` **不是"仓库默认配置"，它只是一份样例**。
> 它的 `_comment` 写得像全局默认（「全局 `stage2_window_max_states` 保持 8 不动」），
> 极易被读成已经生效。**判一个键在某次运行里是什么值，只能看
> 该 run 的 `run_provenance.json`，不能看 `abfe_config.json`。**

**处置（已落盘，代码改动）**：`stage2_first_window_max_states: 4` 加进 `runabfe.py::PRESET_CONFIGS`
的**三个**预设。选预设而不是「只改这 2 份 benchmark config」，理由是后者只解开 2 个 run、
其余 12 个体系下次照样踩同一个死局。
**半径**（实测 `_split_preopt_protocol_key` 的两层拆分，`abfe_pipeline.py:3090/3155`）：
本键属**第 2 层**派生路径键 ⟹ 第 1 层（pilot 采样语义）不变 ⟹
**Stage 0/1 与 pilot/preopt 全部保留**，布局离线重算，**只有 Stage-2 窗口轨迹全部重采**。

---

## 决定 1 —— `BM-04`：**不开放 `anchor = window 0`**

**选项：三条出路里都不选，改为「用正确的首窗 cap 重建」。**
适用 `cmet_ligand1/rep1`、`p38_ligand2/rep1`：

1. 保留 pilot/preopt；
2. 给 Stage-2 建**新的** checkpoint/path 命名空间；
3. 按 `stage2_first_window_max_states=4` 重新生成路径并重跑 Stage-2；
4. **不**增加 `stage2_final_n_states`。

**理由（已逐条回源码核实）**：window 0 没有前置共享态，把它塞进 tail-repartition
**实际等于新增「整条路径重分」这个操作**，不是放宽一个判断 ——
`abfe_preoptimizer.py:6174 tail_repartition_anchor` 的 docstring 写明
「冻结的是 anchor **之前**的窗口」，而 win0 之前没有窗口可冻；后续 repartition 契约
也不支持这种语义。原条目出路 1（允许 anchor=win0）因此被否。

**首窗 cap=4 是正确的预防性修复，但它不会改写旧版本链** —— 所以必须重建，不能原地续跑。
`abfe_config.json` 里该键的 `_comment` 独立给出了更强的论证：
**「布局类修复动作一个都够不着 win0」**（`SPLIT_TAIL_WINDOW` 要求 `window_idx>0`；
`INSERT_LAMBDA` 会重排全局边界）⟹ **win0 的跨度只能在分窗这一刻决定，事后没有任何修复路径。**
实测 cap=4 得 `[4,6,4,4,4,5]`：maxK 从 8 降到 6，且那个 6 落在 win1（split-tail 够得着），
代价 +1 个系综。⚠️ 峰值 ∫g 在 cap=8/6/4 三档都是 134.6 —— **封 win0 在 ∫g 账本上一分钱不赚，
它买的是 K 不是 ∫g**，别拿峰值去证明这个开关有效。

## 决定 2 —— `DATA-02`：**选 ②，清理 Stage-2 后重建**

适用 `cyclod_ligand3/rep1`、`rep3`：**不回退 v3、不手工伪造 v5。**
旧的 `path_current.json` / `path_versions/` / 窗口产物**作为整体归档**，随后用全新的
Stage-2 命名空间重建。

**理由（已核实）**：`lambda_path_versions.py:184 _publish()` 注释逐字写着
**「只前进不后退」**，回退直接 `raise ValueError(f"拒绝把当前路径从 v{...} 倒退到 v{...}")`
⟹ 出路 ① 在代码层面就走不通。而「只删窗口文件、保留 v4 指针」同样不行：
`load_current` 读的是指针，启动布局校验仍会读到那个 K=9 的非法布局。
⚠️ 重建时**必须同样带上 cap=4** —— 这两个 run 也是 `first_untrusted_window=0` 的死局，
不带 cap 重建就是重演一遍。

## 决定 3 —— `DATA-03`：**`max_path_insertions` 3 → 4，`stage2_final_n_states` 保持 21，原地 resume**

适用 `cmet_ligand1/rep2`、`cmet_ligand2/rep1`、`p38_ligand2/rep3` ——
这三个是「**结构上仍可插点，只是预算耗尽**」，与上面四个的死局不是一回事。

**理由（已核实）**：`ibs_engine.py:503` 把 `max_path_insertions` 从
**窗口采样身份**里 `pop` 掉了，且紧邻注释写明 stage **结果**缓存不受影响
（仍要完整 protocol key + `lambda_path_fingerprint` 全路径门）
⟹ stage protocol key 会变、要重新分析，但 **λ 未变的已有窗口可以复用，不需要全部重采**。

**⚠️ 暂时不要改成 23**：它会重建初始 λ 网格，导致多数甚至全部 Stage-2 窗口失去复用条件。
**若第 4 次插点后仍报预算不足，就停在 4，不许继续加到 5** —— 那时才说明初始网格确实偏小，
应切换到 23 做一次干净重跑。

**📌 成本里补一项（原估算漏了）**：这三个 run 的 config 里 `max_path_insertions` 本来就是
显式的，所以改值只影响它们；但它进 stage protocol key ⟹ **Stage 1 结果要重新求解，
实测 ~28 分钟/指纹 × 3 个 run**（只重解、不重采样）。不改变结论，但要计进预算。

## 总体策略与 GPU 代价

**4 个旧/非法 Stage-2 干净重建（`cmet_ligand1/rep1`、`p38_ligand2/rep1`、
`cyclod_ligand3/rep1`、`rep3`），3 个预算不足 run 用 21/4 有界续跑。**

- 决定 1 + 2：4 个 run 重跑 Stage-2，**pilot/preopt 与 Stage 0/1 保留**；
- 决定 3：每个约新增 1 个受影响窗口的 production（个别 tail 情况 2 个）+ Stage 1 重解 ~28 min；
  明显低于改成 23 后整段重跑；
- 预设改动的连带：**全部现存 run 的 Stage-2 窗口缓存失配重采**（Stage 0/1 与 pilot 保留）。
  按上面那条决策原则，这个代价**已被接受**。

## 验收判据

- 决定 1/2：四个 run 的新命名空间里，`path_versions/v1.json` 的 `window_ranges` 首窗 `K ≤ 4`，
  且启动布局校验能过、`decide()` 拿得到可执行动作；
- 决定 3：三个 run 能发出第 4 次 `INSERT_LAMBDA` 并继续，不再以
  `HALT_LAMBDA_BUDGET_INSUFFICIENT` 退出；
- 三者共同：重跑后仍失败的才是新信息，按 `BM-B` 的三条取证口径重新分类。

---

## 原文留档（`BM-04`）

- [ ] **🟠 BM-04 卡住的窗口是 window 0 时，「缩跨度」这一族动作在构造上全部不可行。** **需要维护者拍板，代理不要自行改。**
  位置：`abfe_preoptimizer.py::Stage2RepairController.tail_repartition_anchor`
  —— `if idx is None or idx <= 0: return None`。
  链条：拆末窗要 tail anchor → anchor 取自 `first_untrusted_window` 的**首态**
  → window 0 不可信时 `idx == 0` ⟹ anchor 恒 None ⟹ **拆窗永远不可行**
  ⟹ 只剩插 λ ⟹ 插到末窗满（K=hi）就没动作了 ⟹ `NO_FEASIBLE_ACTION`。
  `idx <= 0` 本身不是笔误：docstring 写明 anchor 的语义是「冻结 anchor 之前的窗口」，
  而 window 0 之前没有窗口可冻结。**所以这是设计边界，不是接线错误。**
  真机命中：`cyclod_ligand3/rep1,rep3`（崩）+ `cmet_ligand1/rep1`、`jnk1_ligand1/rep3`、
  `p38_ligand2/rep1`（插 λ 预算先用完所以没崩，终态理由逐字是
  「拆窗：取不到 tail anchor（没有 window_idx > 0 的不可信窗口）」+「插 λ 预算已用尽」）。
  ⚠️ `BM-01`~`03` **不会**让这些 run 跑完 —— 它们只把"崩溃 / 留下非法布局"换成
  "如实停下并说清原因"。**别把 `BM-04` 当成已经解决了。**
  三条出路，代价差很远，**没有默认答案**：
  1. 允许 anchor = window 0 的首态，即**整条路径重分窗** —— 语义上是"全部作废重采"，
     GPU 代价最大；但 window 0 恰恰是最常卡住的窗口（它是耦合端）；
  2. 加大 `stage2_final_n_states` 给末窗留余量 —— 只把死区推后，不消除；
  3. 接受现状，window 0 卡住就人工介入 —— 现在至少会明确说"卡在 window 0 且拆不开"。
  判据：拍板后在本条下写明选了哪条 + 理由，并归档；若选 1，需要一个
  "window 0 不可信 + 末窗顶满"的盘面能走到实际重分窗且不作废已合格窗口的测试。

## 原文留档（`DATA-02` / `DATA-03`）

- [ ] **🟠 DATA-02 `cyclod_ligand3/rep1` 与 `rep3` 的非法布局已经落盘，这两个 run 起不来。** **需要维护者拍板。**
  实测（2026-09-16，当前代码只读）：两者 `checkpoints/path_versions/v4.json` 的
  `window_ranges` 是 `K=[8, 6, 4, 9]`，末窗 9 > `stage2_window_max_states=8`；
  `first_untrusted_window=0` ⟹ `tail_repartition_anchor=None`。
  于是主循环的**启动布局校验**（`abfe_pipeline.py:11306`）看到 `_tk=9 > hi` 就去调
  `_legalize_tail_window`，那里 anchor 取不到 ⟹ `RuntimeError` ⟹ **`decide()` 一次都轮不到**。
  （只读重放 `decide()` 会给 `RUN_PRODUCTION[0]`，那是**假象** —— `decide()` 不做这道校验。）
  ⚠️ `BM-03` 只阻止**再造出**这种布局，**不会**清理已经写进版本链的。这是数据状态问题，
  和 `DATA-01` 同一类，代码侧修不掉。
  两个选项：① 把版本链回退到 `v3`（末窗 K=8，合法）—— 但三次插点都打在 window 0，
  win0 的产物在每一版都是过期的，回退后仍要重采 win0 及下游；
  ② 清空这两个 run 的 stage-2 产物重跑（`DATA-01` 当时维护者选的就是这条）。
  判据：拍板后在本条写明选了哪条，并确认该 run 能走过启动布局校验进入 `decide()`。

- [ ] **🟠 DATA-03 三个 run 的 λ 总数不够，控制器自己判成「输入问题」。** **需要维护者拍板。**
  `cmet_ligand1/rep2`、`cmet_ligand2/rep1`、`p38_ligand2/rep3` 当前 `decide()` 给
  `NO_ACTION` + `NO_FEASIBLE_ACTION`，理由逐字是：
  「插 λ 的跨 resume 累计预算已用尽（`max_path_insertions=3`，版本链上已插 3 次）⟹
  继续插就是无限循环。**λ 总数不够是输入问题，应判 `HALT_LAMBDA_BUDGET_INSUFFICIENT`
  由人工改输入**」。这是**设计内的正确退出**，不是 bug —— 但它就停在这里，
  除非有人改输入。
  当前 benchmark config：`stage2_final_n_states=21`、`max_path_insertions=3`
  （`abfe-benchmark/openmm_IBS/configs/*.json`）。
  ⚠️ 改 config 会动 `stage_protocol_key` ⟹ **作废现有窗口缓存**
  （见 `incident_config_change_invalidated_all_window_caches`）。改之前先把 GPU 代价摆出来。
  判据：拍板后写明改成多少、以及这几个 run 是重跑还是放弃。
