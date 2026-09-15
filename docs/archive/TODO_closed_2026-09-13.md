# 已关闭的待办（2026-09-13 收口）

> 从 [TODO.md](../TODO.md) §2 整段移出的**一条已关闭条目**，**不是待办**。
>
> | 条目 | 留下的规矩 |
> |---|---|
> | `LR-01` | 一个开关往指纹里进，**有几条路径就得收窄几条**。这次是两条（`run_config` 一条、显式 payload 一条），只收一条等于没收。以及：**顶层 run 指纹和 stage 指纹的收窄口径不同** —— 顶层 `final_results` 代表整个 run 的身份，残差开关无条件进是对的，不要跟着一起收窄 |

---

## [x] `LR-01`（已关闭 2026-09-11，2026-09-13 核实归档）`residual_sampling` 无条件进每个 stage 的指纹

**原文**（TODO §2）：

- [ ] **LR-01 `residual_sampling` 无条件进每个 stage 的指纹** ——
  `abfe_pipeline.py:11008`，在 `if stage_name == "vanishing"` 分支**之前**。
  ⟹ 打开开关会让预平衡 / attachment / decharging 的缓存**全部失配、整条链从头重算**，
  而那几段的哈密顿量根本没被残差碰过。紧邻的 MEM-00h 注释写的正是相反的做法。
  正解：挪进 vanishing 作用域（运行时判据是 `stage_name in {"vanishing","vanishing_rescue"}`，`:5489`）。

### 关闭验收（2026-09-13 只读核实）

代码里已按"正解"落地，且**两条路径都收窄了**：

| 位置 | 现状 |
|---|---|
| `abfe_pipeline.py:2752` | `RESIDUAL_SAMPLING_STAGES = frozenset({"vanishing", "vanishing_rescue"})` —— 与原文写的运行时判据逐字一致 |
| `abfe_pipeline.py:11844` | 路径一（`run_config`）：`if stage_name not in RESIDUAL_SAMPLING_STAGES: run_config.pop("residual_sampling", None)` |
| `abfe_pipeline.py:11985` | 路径二（显式 payload）：`if stage_name in RESIDUAL_SAMPLING_STAGES:` 才插 `payload["residual_sampling"]` |
| `abfe_pipeline.py:11836-11845` | 落地注释（2026-09-11）写明"它有**两条**进指纹的路径……两条都得收窄，否则打开开关会让 decharging 的 stage 缓存无谓失配、整段约 28 分钟白重跑"，并确认关着时两条路径都是 no-op ⟹ 既有指纹逐位不变 |

**不是漏网的那一处**：`abfe_pipeline.py:12933`（`_build_top_level_protocol_key`）仍然无条件插
`residual_sampling`。那是顶层 `final_results.json` 的 run 身份指纹，残差开关确实改变了这一整个
run 是什么，**无条件进是正确的**，不要照着 stage 指纹的样子去"统一"。

原文里的行号 `:11008` / `:5489` 是修复前的位置，修复后已位移，按上表的行号读。

---

## [x] `S2-F`（已关闭 2026-09-13）10 条旧断言停在旧语义

**背景**：`decide()` 重构 + §7.5 跨腿构象门裁决落地后，4 个测试文件共 10 条红着。
TODO 原先写的是「别去修，断言会随新语义一起重写」；重构落地后它变成一条真待办。

### 关闭验收

```
改前：10 failed,  76 passed   （4 个文件单跑）
改后：0 failed,   86 passed
全量：./tests/run_offline_tests.sh → 2223 passed, 3 skipped, 0 failed
```

### 留下的规矩：**先分清「断言旧了」和「fixture 旧了」**

10 条里**只有 3 条真的是断言旧了**，另外 7 条是**造数据的 fixture 旧了**。
区别决定改法，弄反了就会把测试做废：

| 类别 | 条数 | 现象 | 正确改法 |
|---|---:|---|---|
| fixture 缺自检产物 | 5 | 窗口全判 `UNKNOWN` ⟹ 「产出证据」分支（`ANALYZE`）在**所有**目标分支之前把请求吞掉 | `_mkrun` 补 `dual_window_*_self_support.json`；**且预热中的窗口不许补**（那份产物是生产跑完才写的，伪造会让 `self_verdict` 判据盖过 `phase`） |
| fixture 缺预算台账 | 1 | `warmup_steps_left is None` ⟹ `budget_unknown_fail_closed`，预算门排在所有动作选择之前 | 补 `bias_warmup.warmup_budget_ledger`。**fail-closed 本身是对的**（不知道付不付得起验证就别开新 Epoch），修 fixture 不是修门 |
| fixture 场景自相矛盾 | 1 | 「被踢出协方差链」的窗口却带着 `ANALYSIS_ELIGIBLE` 自检 | 让场景自洽：自检也判 `INSUFFICIENT_DATA`，并补上 stage 侧 `cumulative_fk_residual_production`（分析既然跑过，这份证据必然在） |
| 断言真的旧了 | 3 | 见下 | 改断言 |

⚠️ **反面教材**：如果照着实际输出把这 7 条的断言改成 `== "ANALYZE"`，它们会全部退化成
同一个「没有自检产物就 ANALYZE」的测试，缺窗 / 短生产 / DONE / skipped 四条分支的覆盖
直接归零。**那是反方向的「为了让它绿」。**

### 3 条真的旧了的断言

| 断言 | 旧 → 新 | 依据 |
|---|---|---|
| 预算耗尽窗口的出口 | `HALT_NO_ATTRIBUTION` → `HALT_BUDGET` | 归因现在是**成功的**：说得出「付不起新 Epoch 的最低验证额度」。预算可行性被提到选动作**之前**（实测 `run2/vanishing_2` 在 555k/555k 零余量时仍被判 `RECALIBRATE_FK`，开出来的 f_k 永远验不了） |
| 健康窗口 `frames_short_by` | `== 0` → `is None` | 它是**去相关帧数**的缺口，只在去相关那关真没过时才有意义。真机据旧语义印出过「还差 0 帧」这种自相矛盾的话 |
| 跨腿构象门 | `pytest.raises(ValueError)` → `gate == "WARN"` 且不阻断 | 设计文档 §7.5 裁决。⚠️ 改断言时**连带钉住三件事**，少一件这道门就退化成「警告了就等于没事」：`cross_leg_conformer_gate` 永远写、WARN 时完整 report 必须带出来、判据一字未改（`passed is False`）。`strict_cross_leg_conformer=True` 仍然硬抛 |

### 顺带更正的两处文档

- `tests/test_stage2_repair_controller.py` 模块 docstring 第 1 条原写「**缺窗口优先于一切**」——
  那正是被实测推翻的口径（缺窗分支排全局最高时，rep1 里目标被判成 win5，把支撑不足的
  win3 和卡在 local cap 的 win4 整个盖住）。已改写为「按因果顺序路由最早的未解决窗口，
  缺窗口的正确位置在因果顺序**之后**」+ 三态说明。
- 两条测试改名以反映它们现在钉的语义：
  `test_insufficient_data_without_budget_halts_on_attribution` →
  `test_no_budget_is_consumed_before_action_selection`；
  `test_combine_refuses_to_report_delta_g_bind_when_ensembles_disagree` →
  `test_combine_warns_but_does_not_block_when_ensembles_disagree`。

---

## [x] `S2-B`（已关闭 2026-09-13）续验路径上验证要求随预算膨胀

**原文**（TODO §1）：

- [ ] **S2-B 续验路径上验证要求随预算膨胀。**
  `ibs_engine.py:15575` 续验时 `validation_attempt_budget_steps = full_bias_step_budget`，
  于是 `minimum_complete_validation_frames = max(200, budget/stride)`：
  **给的预算越多、要求的帧数越高**，可达性判据的第 2 档因此失效。
  正解是把「完整性要求」与「去相关要求」解耦 —— 200 是统计目标，不该随预算浮动。

### 修法（按原定正解）

预算那一支从公式里去掉，只剩统计目标：

```python
minimum_complete_validation_frames = int(required_consecutive_bias_updates) * 20
```

### ⚠️ 但原条目的**后果描述是错的**，别再照抄

「可达性判据的第 2 档因此失效」**从没发生过**：

| 事实 | 证据 |
|---|---|
| 这个量在本仓**从来没有当过门** | 自第一个 commit（`169514e`，2026-08-31）起就只有「赋值 + 写进报告」两处，零条件判断 |
| 可达性预检的 T 读的是**另一个量** | `abfe_preoptimizer.py` 取 `validation_indeterminate.decorrelated_frames_required`（去相关下限 10），回退到 `IBS_LOCAL_MBAR_GATE_MIN_FRAMES`，**绝不回退到 200** —— 那里还留着一段注释专门警告别混 |
| 控制器侧的字段名已自带免责 | `validation_completeness_frames_REPORT_ONLY`（且全仓无人消费） |

**所以修它不改变任何判定。** 真实危害是另外两条，都成立：

1. **这个数会骗读它的人。** 可达性的 T 一度就被错取成它，gcrit 算小 20 倍，把只差
   26% 帧数的 win4 判成「差 7.5 倍、预算内不可达」（见
   [STAGE2_AUTONOMOUS_LOOP_STATUS_2026-09-11.md](../STAGE2_AUTONOMOUS_LOOP_STATUS_2026-09-11.md) §9）。
   一个名叫「要求」的量实际等于「预算」，是这个误读的温床。
2. **随预算浮动的数，两次 run 的报告没法横向比。**

### 留下的规矩：**改之前先确认那个量到底有没有被消费**

`minimum_complete_validation_frames` 和 `truncated_validation_frames_ignored`
（后者恒为 0）都是 write-only。按「它是门」去推导后果，会把修复的理由和验收口径
全写歪 —— 本条目原文就是这么写出来的。**先 grep 消费点，再写后果。**

### 守卫

`tests/test_fk_relearn_and_reachability.py`：
- `test_completeness_requirement_is_decoupled_from_budget` —— 按**源码 AST** 断言
  该赋值右侧不得引用 `validation_attempt_budget_steps` / `full_bias_step_budget` 等
  任何预算量（这个量活在 `run_ibs_bias_warmup` 内部，真跑到它得起 OpenMM + 完整
  预热循环，所以按源码断言）。已用旧公式变异验证过会红。
- `test_reachability_T_is_the_decorrelated_floor_not_the_completeness_target` ——
  钉住两个量不许合并。

### 顺带记下、**没动**的一处

`truncated_validation_frames_ignored` 恒等于 0（只有 `= 0` 和写进报告两处），
却以「被忽略的截断帧数」的名义出现在报告里。要么它该被递增而逻辑丢了，要么
它该删。**不在 S2-B 范围内，未处理。**
