# Stage-2 自治循环：现状与剩余工作

**2026-09-11** · 核心指标（老板定）：

> **一次启动，无人干预；遇到采样/收敛问题，自己读证据、诊断原因、选择动作、执行、复验，直到完整结果。**

验收 = **resume 当前 `rep1` 后人不碰它，自己处理 win3、跑过 win4/win5、产出完整 Stage-2**。
不是 replay 对账、不是测试全绿、不是 trace 好看。

---

## 1. 已接通

`abfe_pipeline._run_stage2_autonomous()` —— `decide → execute → reread`，默认开
（`stage2_autonomous_controller`，异常**不吞、直接抛**）。每轮决策落
`checkpoints/stage2_autonomous_history.json`。

**只有三个真终态**，其余全是路由信号、不得退出循环：
`DONE` / `GLOBAL_BUDGET_EXHAUSTED` / `HALT_INVALID_INPUT`·`NO_FEASIBLE_ACTION`。

动作 → 执行器：

| 动作 | 落到哪 |
|---|---|
| `RUN_PRODUCTION` | `run_once(_only_window_indices, _production_step_overrides={w: 当前+250k})`（**加法**，不是 ×2） |
| `RECALIBRATE_FK` / `PROBE_CANDIDATE_FK` | `_recalibrate_f_k_and_resample_segment()`（新 Epoch，旧段保留） |
| `SPLIT_TAIL_WINDOW` | `repartition_tail_from_anchor` + `record_tail_repartition_version` + 用新 ranges 重跑 |
| `CONTINUE_WARMUP` / `ANALYZE` | `run_once(_resume_override=True)` |

**停滞保护**：同一 `(action, windows)` 连三次推不动 ⟹ 自动降级到换 Epoch，都推不动才
`NO_FEASIBLE_ACTION`。没有它，一个路由信号会把循环**转死**（烧 GPU 且无产出），比炸出去更糟。

### 本轮修掉的（都有实测）

| | 问题 | 结果 |
|---|---|---|
| P0 | `run_once=` 关键字传参 | 位置传参；`RECALIBRATE_FK` 分支不再必崩 |
| P0 | fresh-run `NameError`（`plan()` 闭包引用未构造的 `blocked`/`earliest`） | `plan()` **显式接参**；四个入口（目录不存在/空目录/只有路径版本/只有 win0 部分 warmup）实测不炸 |
| P0 | Segment 当成两个独立 stage | `Stage2RepairController.for_physical_stage()` 聚合；同一物理 stage 只出**一个**动作 |
| P0 | 缺窗分支全局最高优先级，盖住最早未解决窗口 | 改为**按因果顺序取 earliest unresolved**；rep1 目标 win5 → **win3**；缺窗降到"前缀全部合格"之后 |
| P0 | 零/未知预算仍选重标定 | 预算可行性**前置于选动作**，且**预算未知 = 不可行**（fail-closed） |
| P1 | 缺证据被当成"这个窗口有问题" | 三态分开：`ELIGIBLE` / `PROBLEM` / **`UNKNOWN`**。UNKNOWN 的动作是**产出证据**，不是换 Epoch |
| P1 | 支撑不足就先加帧 | 改为**先算累计 f_k 偏差**（现有帧上零额外采样）；加帧治不了 f_k 偏斜（实测 250k→1M 让 top1% 0.545→0.762） |

---

## 2. 剩余工作（4 件，无未解设计问题）

1. **同僚报的 21 failed 清单** —— 含本轮调度改动的预期内旧断言，但**混着真 bug**，必须逐条看。
2. **`PROBE_REANCHOR_EPOCH` + held-out 候选验收** —— 老板路径里「held-out 可测则验收 / **不可测则自动开 PROBE_REANCHOR_EPOCH**」这一步目前**不存在**；候选用**连续时间块**训练、held-out 查 span 与 `N_eff/g`（**不能随机拆帧**）。
3. **插 λ 执行器** —— 「不改善则自动 tail-repartition / **插 λ**」只有前一半。
4. **段号与 ANALYZE 收口** —— 段号 `1+tail+len(history)` 是拍的（可能与已有段目录撞号）；`ANALYZE` 映射到 `run_once`，未必产出全路径 ΔG ⟹ 可能判不到 `DONE`。

---

## 3. 不许回退的约定

- **缺证据 ≠ 通过**，但**缺证据 ≠ 有问题** —— 它是 `UNKNOWN`，动作是去产出证据。
- **低支撑永远不是 `FAIL`** —— 是「尚不可测」（`INSUFFICIENT_DATA` / `HARD_INSUFFICIENT`），加预算；`FAIL` 是支撑够了却违反统计门，换 Epoch。
- **累计 f_k 残差只能用 `sampling_states`** —— `energies` 多一个逐 λ 态常数（LRC），其**逐边差**会伪造出「全负号单调」的形状（实测把某窗 span 抬高 2.2 倍）。`N_eff/g` 则对它**不变**，两者不变性不同，别"统一"。
- **不得为了得到期待的标签而换误差估计器**（estimator shopping）。新 Epoch 才是拿独立证据的地方。
- **`(K−1)×门槛` 不许写** —— 累计偏差门不随边数增长。
- **控制器只读**；执行由独立的 execute 层落盘。

## 2026-09-11 最后一遍扫：五个 bug

在 `_run_stage2_autonomous`（`abfe_pipeline.py`）里扫出来的。前四个已修，第五个未修。

### 1. 停滞计数是「累计」而不是「连续且盘上状态未变」— 已修

日志一直打印「连续第 N 次且盘上状态未变」，但代码从来没读过盘，`seen[key]`
只是这个 key 出现过几次的**累计**计数。

后果直接掐死验收路径：`RUN_PRODUCTION[3]` 第 1、2 块都真的加了 250k、支撑在涨，
第 3 块被当成「推不动」掐掉并强行换 Epoch。补采链最多只能走两块。

修法：加 `_disk_signature(view)` = (路径版本号, 每窗 (window_idx, production_steps,
self_verdict, segment))。指纹变了 ⟹ 动作确实推动了盘 ⟹ `seen[key]` 清零、
`escalated` 也清掉。只有指纹不变时才累加。

### 2. `RUN_PRODUCTION` 跨命名空间串号 — 已修

`cur` 取自**聚合视图**（可能来自 `segment_N` 的记录），却把帧写进**基准 stage 目录**。
两条错：

- 把 segment_N 的 cumulative 当成基准段的起点，override 的绝对步数没有意义；
- 等于回头给**旧 f_k** 加帧，违反「绝不回去给旧 f_k 加帧」。

修法：从窗口记录的 `segment` provenance 推出段目录，通过
`_output_dir_override` / `_checkpoint_dir_override` 把补采落在证据所在的那个段里。
多个窗口落在不同段时 fail-closed，不静默回落到基准段。

### 3. 降级不封顶 → 无限开新段烧 GPU — 已修

原逻辑每轮 `seen[key] >= 3` 都重新降级一次，于是一个推不动的动作**每轮**触发一次
换 Epoch：无限开新段、永不退出。加 `escalated[key]`，降级只许一次，第二次直接
`NO_FEASIBLE_ACTION` 退出并写进 history。

同批还修了段号：`_existing` 从 `[0]` 起会在空目录时算出 `seg_idx=1`，写进基准
stage 自己的命名空间。基准 stage **隐含是段 1**（`_recalibrate_f_k_and_resample_segment`
的默认 `segment_index` 就是 2），改成从 `[1]` 起。

### 4. 两处小的 — 已修

- `INSERT_LAMBDA` 里 `_rng` 第一行把 `window_idx` 当**全局态号**解，被第二行覆盖成
  死代码；一旦第二行条件不成立就静默回落到那个错误解释。删掉，直接索引 `ranges`。
- `run_once` 抛异常时 `raise` 越过了函数末尾的历史落盘，决策轨迹正好在最需要它的
  时候丢掉。抽出 `_write_history()`，异常路径先落盘再 raise。

### 5. 多 Epoch 链永远从基准段重学 f_k — 已修

`_recalibrate_f_k_and_resample_segment` 里源目录写死成 `stage_dir`
（以及 `window_label_prefix="segment1_window"`），所以第 N 次换 Epoch 仍然拿
**段 1** 的帧重解 f_k，把第 2…N−1 段的证据全部丢掉。

影响范围：**第一次**换 Epoch 完全正确，所以当前验收路径
（win3 → 换一次 Epoch → win4 → win5）不受影响。只有需要连换两次 Epoch 时才踩到。

修法：拆开「源目录」和「输出目录」。新增 `source_stage_dir` /
`source_checkpoint_dir` 两个参数，默认 `None` ⟹ 回落到基准段（老调用点行为逐字不变）；
函数内四处源目录引用统一走 `src_dir`/`src_ckpt`
（`_load_ibs_window_outputs_from_dir` 的位置参数与 `checkpoint_dir=`、
`dual_window_*_vdw_sampling_states.npy`、`*_convergence.json`、`*_self_support.json`）。
诊断里加落 `source_stage_dir`：多 Epoch 链「这次 f_k 从哪一段学来的」的唯一可审计凭据。

自治循环的两个换 Epoch 调用点（`RECALIBRATE_FK` / `PROBE_REANCHOR_EPOCH`）都改成传
`_latest_segment_dirs(stage_dir, checkpoint_dir)` —— 盘上段号最大的那个段。

## 新增的两个共用小工具

- `_segment_dirs_for_evidence(segments, stage_dir, checkpoint_dir)` —— 把窗口记录的
  `segment` provenance 翻成 (输出目录, checkpoint 目录)；基准段返回 `(None, None)`；
  **跨段 fail-closed 抛错**，不静默回落（静默回落正是 bug 2 本身）。
- `_latest_segment_dirs(stage_dir, checkpoint_dir)` —— 盘上段号最大的段；只有基准段时
  返回 `(None, None)`。

自检在 `tests/test_stage2_autonomous_segment_routing.py`（7 项，没装 openmm 的机器上
自动回退到 AST 抽函数，`python3` 直接跑）。

### 6. 换 Epoch 无差别作用于全部窗口 — 已修

真机现场（rep1）：

```
[自治] 动作 ANALYZE[3] 连续第 3 次且盘上状态未变 ⟹ 它推不动了。
[自治] 降级到 RECALIBRATE_FK（换 Epoch 拿独立证据）。
[f_k 重标定] 用段1 生产帧重解，窗口 [0,1,2,3,4,5] 的 f_k 位移超过阈值；
            以新 f_k 为热启动种子在 vanishing_2 采第 2 段
```

卡住的是**一个**窗口，重标定却作用在全部 6 个。当时盘上的真实状态：

| 窗口 | min N_eff/g | verdict |
|---|---|---|
| 0 | 12.96 | ANALYSIS_ELIGIBLE |
| 1 | 60.80 | ANALYSIS_ELIGIBLE |
| 2 | 46.98 | ANALYSIS_ELIGIBLE |
| 3 | **1.65** | INSUFFICIENT_DATA |
| 4 | **3.44** | INSUFFICIENT_DATA |
| 5 | 25.93 | ANALYSIS_ELIGIBLE |

4 个已合格的窗口被拖进新段。**不只是白烧 GPU** —— `read_aggregated` 是「逐窗取段号
最大且有 convergence 产物的那份」，所以段 2 的新结果会**顶掉** win0 那份 12.96 的
合格证据；新的更差就等于把已经通过的窗口打回去。

修法：`_recalibrate_f_k_and_resample_segment` 加 `only_windows` 参数（默认 `None`
= 全部，老调用点行为不变）；不在名单里的窗口记 `skipped: not_in_only_windows`。
自治循环的两个 Epoch 执行器都传 `only_windows=sorted(wins)`。

另外附带修了位移判据的表述问题：`[0,1,2,3,4,5] 的 f_k 位移超过阈值` 这行本身也在提示
0.5 kJ/mol 阈值对 50–80 kJ/mol 量程等于谁都超（函数里已有
`displacement_is_not_the_trigger` 的说明，这次没动那部分判据）。

### 7. 降级目标选错 — 已修

`ANALYZE[3]` 推不动 ⟹ 降级到 **`RECALIBRATE_FK`**：一个作用于全路径、没有上界的动作。
按既定判据，`INSUFFICIENT_DATA` + held-out 不可测应当走 **`PROBE_REANCHOR_EPOCH`**
（有界探针：一个窗口、一个 `+250k` 块）。

改成降级到 `PROBE_REANCHOR_EPOCH`。干跑验证：

```
[自治] 降级到 PROBE_REANCHOR_EPOCH（只对窗口 [3] 换 Epoch 拿独立证据，一个块）。
[RECAL] 出=segment_3 限定窗口=[3] 步数={3: 250000}
```

（`segment_3` 是对的：盘上已有 `vanishing_2`，源目录也正确地取到了段 2 而不是段 1。）

## 2026-09-12 真机跑出来的两个 P0

### 8. 路由信号被当致命异常抛穿，整条流水线死 — 已修

现场（14:40:01）：段2 win4 冻结验证 15/15 批仍
`insufficient_frames_after_decorrelation` ⟹ `IBSValidationBudgetIndeterminateError`
⟹ 自治循环 `raise` ⟹ 全跑挂掉。

但这恰恰是既定口径里的**路由信号**，所有证据都指向"不该停"：

- 异常自己写着「不改 f_k、不退回 SGD、不插 λ；resume 会接着验」；
- `warmup_failure.json` 已落盘（`status=validation_budget_exhausted_indeterminate`）；
- 控制器**早就认得它**：`HALT_LOCAL_VALIDATION_CAP` → `RUN_PRODUCTION` +250k 诊断块；
- 该窗口预算还剩 515000 / 955000 步，远没到 `GLOBAL_BUDGET_EXHAUSTED`。

同一个洞的另一半：`_run_stage2_with_path_evolution` 接了
`IBSWarmupConvergenceError` 和 `IBSFrozenCalibrationValidationError`，但**自治循环的
`run_once` 不走那个函数**，这两个在自治路径上一样会炸穿。三类一起补齐：

| 异常 | 处置 |
|---|---|
| `IBSValidationBudgetIndeterminateError` | 路由 `LOCAL_VALIDATION_CAP`，回顶层重判 |
| `IBSWarmupConvergenceError` | 路由 `WARMUP_F_K_NOT_CONVERGED`，回顶层重判 |
| `IBSFrozenCalibrationValidationError` | **仍然上抛**，但先落 `stage2_fk_refuted.json` 不裸炸 |

第三条刻意不路由：f_k 被验证统计驳回是**有功效的否决**，不是"还没测够"。
驳回之后是终态交人工、还是回 LEARN 换一份 f_k 重来，是
`PLAN_PATH_REPAIR_2026-09-11.md` P3 待定 A —— 未定的科学决定不许藏进代码。
**这是全自动路径上唯一剩下的人工闸门。**

顺带把 warmup 预算纳入停滞签名：撞验证批次上限那条路由不产出生产帧，但**确实在烧
该窗口的预算**。不算进来，停滞保护会在预算还剩一半时误判"推不动"退出。

### 9. ANALYZE 把循环自己开出来的采样段全扔了 — 已修

`_vanishing_segment_dirs` / `_load_ibs_window_outputs_merged` 只活在旧的 rescue 路径
（`abfe_pipeline.py:14164+`）。自治循环里 `result, diag = self._recalibrate_...` 的
`diag` **拿到就丢**，段目录从不累积，合并求解从不调用；ANALYZE 走
`run_once(_only_window_indices=None)` ⟹ 基准 stage 目录。

真机代价：段 2 把 win3 的 min N_eff/g 从 **1.65 修到 10.52**
（λ=0.3156 的有效样本 **20 → 247**），而**最终 ΔG 里一点都看不到**。
循环修好了窗口，却报出没修过的那个答案。

修法：新增 `_solve_merged_segments_if_any()` —— 从盘上发现全部段，走既有的
`_load_ibs_window_outputs_merged` + `solve_stage_integrated` 按窗口合并；只有基准段
时返回 `None` 走原路径逐字不变。**数值**失败降级回单段并明确标记，绝不伪装成合并
成功；输入错误（`MultiSegmentInputError`：能量错位 / 身份不一致 / 账本损坏）不捕获，
继续 fail-closed。

段的语义是**相加不是取代**：采纳新段等于把旧段的帧全扔掉（实测 w1/w2/w3 里
2/3 到 4/5 的 ESS）。

## 2026-09-12 老板裁决：两条规则已落地

### 规则 1：`IBSFrozenCalibrationValidationError` 只终止这份候选

PLAN P3 待定 A **已定**。统计驳回不再终止整个 Stage-2：

```
候选 f_k 被统计驳回
→ 封存该候选 + fingerprint，永不续验
→ 若有完整新 Epoch 预算：自动 RELEARN_FK_EPOCH
     fresh LEARN → 新 f_k → burn-in → 独立 held-out 验证
→ 新候选通过：继续生产
→ 新候选也被驳回 / 实质上还是同一份 / 预算不足：
     NO_FEASIBLE_ACTION，终止并留完整诊断
```

新动作 **`RELEARN_FK_EPOCH`**，**不得**混进 `RECALIBRATE_FK`：后者拿**已有生产帧**
重解（信息来自那条已被驳回的轨迹），前者**从头 LEARN**，执行时**不传
`_initial_f_k_by_window`** —— 传了就是热启动，等于把被驳回那份带进新 Epoch，
独立性就没了。

约束落地情况：

| 约束 | 实现 |
|---|---|
| 只允许一次替代候选 | `relearn_epoch_used()` / `mark_relearn_epoch_consumed()`，按 (path_version, window) |
| **两扇门共用配额** | `REJECTED→RELEARN` 与 `UNREACHABLE→RELEARN` 查的是同一条记录 |
| 预留完整 LEARN+burn-in+首档验证 | `_relearn_epoch_required_steps()`，量从该窗口**自己**上轮消耗自校准（win4 实测 140000 步，剩 515000 ⟹ 可启动） |
| 验证进度清零、终身账本不清零 | 新段自带独立验证状态；`inherit_warmup_ledger_across_segments` 保证账本不清零 |
| 驳回记录绑四件套 | `path_version + window_idx + lambda_identity + **f_k 向量**` |
| 不得解释成「λ 太稀」 | 驳回分支不触达任何布局动作 |

**候选身份不用 hash。** 主线在快速反复变动，哈希里放什么一改，所有已封存记录就
全部失配、被驳回的候选会被当成新的重新试一遍 —— 正是"反复试到偶然通过"。而且这是
本仓库**已经复发四次**的同一个坑：自产产物的 sha256 进身份，规则是「只有用户输入
才配做身份」，而 f_k 是我们自己算出来的。正解不是删掉身份，是改成**语义身份**：
f_k 向量本身。`fingerprint` 只作为溯源线索保留，字段名写死
`candidate_fingerprint_PROVENANCE_ONLY`，不参与任何判定。

**顺带修掉设计里一个失效的护栏**：原文「生成出相同 fingerprint ⟹ NO_FEASIBLE_ACTION」
在实现上是**死条件** —— fingerprint 是哈希，fresh LEARN 的 f_k 浮点上永不会逐比特
相同，精确相等永不触发。要挡的是「换了一份但实质还是同一份」，那是 f_k 空间的
**距离**问题。改成 `sealed_candidate_matches()`：mean-center 后逐态比较，容差复用
仓库已有的 `min_adjacent_shift_kJ_mol = 0.5`，不另造阈值。

### 规则 2：验证可达性预检

`validation_reachability_verdict()`（`ibs_engine.py`，纯函数）：

```
gcrit_cycle  = Ncap / T
gcrit_budget = (已有帧 + 剩余预算 / 每帧步数) / T
```

### ⚠️ 修正：T 取错了，win4 其实**可达**

第一版把 `T` 取成 `minimum_complete_validation_frames = 200`。**错了。**
触发 `insufficient_frames_after_decorrelation` 的是
`_solve_single_window_local_mbar` 的 `min_frames`，默认 **10**
（现已提成常量 `IBS_LOCAL_MBAR_GATE_MIN_FRAMES`，两处默认值绑定到它，杜绝漂移）。
200 管的是"这次 attempt 攒够**原始**帧没有"，是完整性要求，两个量差 **20 倍**。

| | 第一版（错） | 实际 |
|---|---|---|
| T | 200 | **10** |
| gcrit_cycle | 3.0 | **60** |
| gcrit_budget | 13.3 | **266** |
| 需要原始帧 | 15462 | **773** = 10 × 77.31 |
| 需要步数 | 3865500 | **193275** |
| 对剩余 515000 | 差 7.5 倍 | **够，富余 2.7 倍** |
| verdict | `UNREACHABLE` | **`REACHABLE_WITHIN_BUDGET`** |

win4 在 600 帧上只差 **173 帧（26%）**，它停下来是因为撞了**15 批的周期上限**，
**不是预算**。这正是控制器本来就写好的 `LOCAL_VALIDATION_CAP` —— 继续攒就行，
也正好印证 2026-09-11 那条「按预算老老实实攒到上限」。

教训：把两个名字相近、量纲相同、差 20 倍的阈值混用，会让一个**只差四分之一帧数**
的窗口被判成「差一个数量级、不可达」，然后白白烧掉那唯一一次替代候选。
**可达性判据的 T 必须从产物里读**（新增落盘 `decorrelated_frames_required`），
读不到只回退到常量，**绝不回退到 200**。

四档路由：`g_L ≤ gcrit_cycle` 继续本周期 / `≤ gcrit_budget` 续累计块 /
`> gcrit_budget` 立即停同候选补采 / 跨阈值 ⟹ 最多一个 +250k 诊断块后必须裁决。

**两条必须写死的边界：**

1. **只改路由，不改 verdict。** 返回值里带 `may_be_used_to_reject_f_k: False`。
   不可达是**预算**结论；证据仍是 `UNMEASURED`，绝不改写成 `REJECTED`，
   也绝不仅凭高 g 去插 λ / 拆窗，更不得因为高 g 就接受一份未验证的 f_k。

2. **不与 2026-09-11 的「不做提前外推」决定冲突。** 那条否决的是**周期内**
   （5→10→15 批）按 n_eff 外推提前掐断，理由是 g 只在 N ≫ τ 后才稳。本预检：
   - 发生在**周期用尽之后**，决定要不要开**下一个**周期，不掐断本周期；
   - `g_L` 取多检查点**最小值**，不是外推；
   - 而且 g 还在随 N 涨这件事**加强**不可达结论 —— 真 g ≥ 实测 g，缺口只会更大。

   落地口径因此是：**`ibs_engine` 只记录 g 历史（新增 `validation_g_history`，
   纯观测零行为改动），判定全在控制器。**

`g_L` 取最小值而非 bootstrap：同一候选上的连续检查点是**嵌套**的（400 帧含 200 帧），
bootstrap 的独立性前提不成立、会低估方差，那样算出来的「保守下界」是假的。

### 待修（不阻塞）：续验路径上验证要求随预算膨胀

`ibs_engine.py:15575` 续验时 `validation_attempt_budget_steps = full_bias_step_budget`，
于是
`minimum_complete_validation_frames = max(200, (515000+249)//250) = 2061` ——
**给的预算越多，要求的帧数越高**，`gcrit_budget` 从 13.3 掉到 2660/2061 ≈ 1.29。
规则 2 的第 2 档（`3 < g_L ≤ 13.3`，开累计验证块）在这个耦合下反而更过不去。

正解是把「完整性要求」与「去相关要求」解耦：200 是统计目标，不该随预算浮动。
未修，因为 win4 当前走的是第 3 档（`UNREACHABLE`），不经过第 2 档。

### 10. 部分段炸掉多段合并 — 已修（resume 前拦下）

Bug 9 接上合并之后，**第一次真实调用就炸**：

```
IBSIncompleteStageCoverageError: 预期窗口 [0,1,2,3,4,5]，实际加载 [0,1,2,3]，
缺失窗口 [4,5]（来源 vanishing_2）
```

根因是 Bug 6 的修复本身：`only_windows` 限定窗口之后，**部分段是常态不是异常**，
而 loader 对**每个段**都要求全窗口齐全。

loader 的报错文本自己指了正道：「必须**显式传入** `excluded_local_windows`，
而不是让文件缺失来隐式决定覆盖范围」。所以：

- `_load_ibs_window_outputs_merged` 的 `excluded_local_windows` 接受 **每段一份**
  （`{段目录: 窗口集合}`）；
- `_solve_merged_segments_if_any` 按段探测缺窗并显式声明；
- ⚠️ **按段放宽了，全体覆盖度不许放宽** —— 缺首/末窗只会让链在更窄的 λ 区间上
  闭合、产出截断的 ΔG 却仍报 `converged=True`。所以合并前断言「每个期望窗口至少
  被某一个段覆盖」，不满足就拒绝合并走单段。

紧接着第二个 fail-closed：多段合并要**采样规范**能量，而
`_load_ibs_window_outputs_from_dir` **只在 residual 臂**返回
`sampling_state_energies`，baseline 臂必须自己从盘上取
（`_recalibrate_f_k_and_resample_segment` 里早就这么绕了，合并 loader 漏了）。
补上同样的读法（`dual_window_{i}_vdw_sampling_states.npy`，落盘 (frames, states)
转成 (states, frames)）。那道 fail-closed **挡得对**：`energies.npy` 比
`sampling_states.npy` 多一个逐 λ 态常数（LJ 长程尾项），逐态常数不是共模、
会改变 logsumexp 的形状。

修完后真实数据实测：

```
[自治] 多段合并：逐段缺窗 vanishing_2缺[4, 5]（已显式声明，非文件缺失隐式决定）
[自治] 2 段合并求解：ΔG=42.9982 ± 1.7879 kJ/mol，converged=False，完整路径=True
```

`converged=False` 正确 —— win4 仍是 `INSUFFICIENT_DATA`，循环应当继续。
