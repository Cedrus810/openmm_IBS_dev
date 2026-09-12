# Stage-2 三轴耦合现状（分窗口 / 分 λ / 分采样量）

**日期**：2026-09-11
**性质**：**现状快照，不是提案。** 为"重新设计一个总控制"做输入用。
只记可核对的事实与出处；凡是推断都显式标成「推断」。

相关但不重复的文档：
- `docs/STAGE2_WINDOW_AND_SEGMENT_REDESIGN_2026-09-11.md` —— 分窗+多采样段重构的落地记录。
- `docs/STAGE2_ROOT_CAUSE_2026-08-28.md` —— 单系综重加权的根因。
- `abfe_config.json` 里 `_comment_stage2_*` 四条 —— 当前各值的实测依据，**别重新论证**。

---

## 1. 三个轴的当前控制点

| 轴 | 谁定初值 | 谁在运行时改 | 出处 |
|---|---|---|---|
| **分 λ**（态数 + 位置） | `stage2_final_n_states=16` + 度规布点 + `stage2_free_energy_densify_points=2`（14+2） | `insert_lambda_in_failed_ibs_window` | `abfe_preoptimizer.py:1648` |
| **分窗口**（每窗装几个态） | `generate_overlapping_windows(n_states, ...)`，受 `stage2_window_min_states=4` / `max_states=5` 约束 → 实测 `[(0,4),(3,8),(7,12),(11,16)]` | 同上那一个函数（尾段重划） | `ibs_engine.py:3905` |
| **分采样量** | config：`pilot_n_steps_per_state=15000`、`n_steps_per_window=250000`、`max_bias_warmup_steps=500000` | 三套互不相通的加码，见 §4 | 见 §4 |

---

## 2. 核心问题：依赖方向在「初始」和「补救」之间反转

```
初始布点：   窗口  ←—— 派生自 ——  λ 数
             generate_overlapping_windows(n_states, pts_per_window, overlap)

补救时：     λ 数  ←—— 派生自 ——  窗口数
             insert_lambda_in_failed_ibs_window 的 docstring 原话：
             "数出尾段实际有几个窗 w_before，把 w_before+1 当作硬目标传给尾段
              分窗器，并插入刚好够满足该目标可行性的 λ"
```

后果：**不存在「只插 λ 不动窗口」的动作，也不存在「只缩窗口不插 λ」的动作。**
运行时只有一个耦合动作，同时移动两个轴。

---

## 3. 五个具体耦合点

### C1 — 补救动作把 λ 轴和窗口轴焊死
`abfe_preoptimizer.py:1648` `insert_lambda_in_failed_ibs_window`。
插几个 λ 由「尾段窗口数 +1」反推，不是由「哪条边太长」决定。
函数内有后置断言保证失败窗口真的变小（历史上按理论最少窗口数推导时，
21 态 [5,5,5,5,5] 的失败窗曾从 5 态变成 **8** 态，比不插还糟）。

### C2 — 窗口级证据触发边级动作（⚠️ 本节 2026-09-11 已更正两次，读到底）

**当前真实状态（AST 核过）**：`_run_stage2_with_path_evolution` 的 handler 集合是
`{_ie.IBSWarmupConvergenceError, RuntimeError}`。后者是插点自身失败的局部兜底，
上抛的是**原**异常，不驱动任何布局动作。

```
IBSWarmupConvergenceError（ibs_engine.py:7845）= "测出来了，f_k 压不平"
        ⟹ 触发插 λ                                    ← 唯一的布局动作触发源

IBSValidationBudgetIndeterminateError（:7853）= "**没测出来**"
        ⟹ 不触发任何布局动作                          ← 老板的裁定
```

**❌ 本节先前写的"两个语义相反的诊断共用同一个动作"是错的，已删。**
那描述的是 2026-09-11 当天一次**存在约一小时、随即被老板否掉并回退**的改动
（原话：「不是 ess 不够拆」）。裁定语义，**别再接回去**：

> **ESS / 帧数不够是采样问题，结构改动治不了。**
> 而插 λ 同样是结构改动 —— 缓存键是 `(window_index, λ 值)`，插点让下游窗口
> 重新编号、λ 对不上、已采产物作废。所以"路由到插 λ"和"路由到拆窗"在代价上
> 是同一类，**不是缓和版本**。

**残余的真问题（这一半仍然成立）**：`IBSWarmupConvergenceError` 本身是**窗口级**
证据（f_k 在整个窗口上压不平），却触发一个历史上按**边级**语义设计的动作。
这一半在 model B 下被消解了 —— model B 的插点是"缩小窗口跨度"，本就是窗口级工具
（见 `DESIGN`/`PLAN_PATH_REPAIR_2026-09-11.md` §2 更正与 §3ter.2）。所以 C2 现在
只剩一条**命名/文档**债，不是行为债。

### C3 — 第三个出口不在任何控制器里
`IBSFrozenCalibrationValidationError`（`ibs_engine.py:7942`，终态 `F_K_EVIDENCE_REFUTED`）
在 `abfe_pipeline.py:93` 只有 `import`，**全仓库零个 `except`** —— 直接炸穿整个 run。
（可能是刻意 fail-closed 交人工；但它不在 §1 的任何一个轴上，那个 import 目前是死的。）

### C4 — 全局重分窗被禁用，且会覆盖手工修复
`abfe_config.json`：`enable_lambda_refine = false`。
原注释：`partition_windows_by_delta_f_budget`（`abfe_preoptimizer.py:2576`）曾有 bug
（净位移代替累积总变差），把 18 态/4 窗重分成畸形 12 窗；bug 已修，但坏结果已写进
`checkpoints/preopt_dual_vanishing.json` 并标成"已精修"，后来用
`repair_stage2_tail_gap_local.py` 手工只修尾段。**再打开这个开关会立刻全局重分、
覆盖手工修复。** 所以存在第二条改窗口的路径，且与 C1 那条互不知道。

### C5 — 跨层只能看见异常，看不见状态
`bias_status` 六态（`unconverged` / `calibrated_pending_validation` /
`frozen_validation_indeterminate` / `calibrated_validation_failed` / `failed` / `converged`）
在 `ibs_engine.py` 之外**零处读取**（grep 证实，只有 tests 引用）。
跨层唯一通道是 2 个异常类型 —— 6 态在层边界上塌成 2 个信号。

---

## 4. 采样量轴：三套互不相通的账

| 机制 | 作用域 | 出处 |
|---|---|---|
| `FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS = (50k, 150k, 300k)` + `WARMUP_LEDGER_BUCKETS`（learning / freeze burn-in / frozen validation） | 窗口内 f_k 验证 | `ibs_engine.py:7699`、`:7711` |
| `production_rescue_targets`（2 轮 × 2 倍步数；实测把 window 5 从 500k 抬到 1M） | 生产阶段补采 | `abfe_pipeline.py:13027` |
| `pilot_n_steps_per_state` / `n_steps_per_window` / `max_bias_warmup_steps` | 初始，config 静态 | `abfe_config.json` |

**没有任何一处能回答"这个窗口总共烧了多少步、重试了几轮"。**
这也是"查看当前状态与证据"那一格做不出来的直接原因。

---

## 5. 三个动作其实绑在同一个物理量上（推断）

三个轴被同一个量约束：**这个窗口装的态数，能否在已投的采样量下被重加权出来**
（去相关有效帧 / local-MBAR 相邻残差，阈值
`IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL = 10.0`，`ibs_engine.py:7629`）。

按代价排序，这是**默认偏好，不是普遍规律**（证据表明 f_k 明显不合适时先加帧是浪费；
只缩窗可行时不必先插 λ）。三档为：

```
有效帧不够 → ① 加采样量   最便宜，已有帧不作废
           → ② 缩窗口     态数少了，同样帧数够用；不动 λ，旧帧保留为独立段
                          ⚠️ 保留 ≠ 可直接复用：缩窗改变了窗内采样分布，旧帧属于
                          另一个混合分布，能否并入 MBAR 要测（见 DESIGN_…_CONTROLLER §5）
           → ③ 插 λ       最贵，缩 λ 跨度，新态从头采
```

**② 在当前配置下是空档。** 生产 `min_states_per_window=4` / `max=5`：
两个子窗共享一个边界态（p+q−1=K），两侧都要 ≥4 就得母窗 ≥7 —— 压不出来。
所以 ① 用完只能直接跳 ③。**这就是 C2 里两个不同诊断被迫共用一个动作的根因** ——
不是设计偷懒，是档位缺了一格。

---

## 6. 重新设计前需要先定的物理判断

**`min_states_per_window` 能不能低于 4？**

- 反对的理由（现有注释）：窗口两端各有一个与邻窗共享的边界态，内部只剩 K−2 个
  自由态。K=4 → 2 个；K=3 → 1 个；K=2 → 0 个。注释同时写明这是**工程下限，
  不是已证明的必要条件**（共享边界并不意味着那两个态的 f_k 不能调）。
- 已有反例：`split_window_from_ibs_lse_failure`（`abfe_preoptimizer.py:1377`）
  允许两态子窗，注释说"那是设计精修档的下限，生产压不齐"。

**这条不定，②档就永远是空的，三个轴怎么重构都还是一个耦合动作换个写法。**

---

## 7. λ 轴自身还有一个零和约束（不是 bug，是物理）

`abfe_config.json` 的 `_comment_stage2_free_energy_densify_points` 记录的实测结论：

- 固定总态数下，**热力学长度均匀**（δ_max）与 **ΔF 均匀** 两个目标互斥：
  "全局改布点权重是零和的（ΔF 砍一半 δ_max 要涨到 1.8）"。
- "重新分窗救不了" —— 穷举过全部合法分窗，最大窗 ΔF 一模一样。
- 14+2 densify 是当前折中：总态数不变（采样成本一分不变），只把 2 个点从平坦中段
  挪到 λ≈1 的陡峭段。实测最大边 ΔF 复合物腿 13.6→7.7、溶剂腿 9.2→6.4。

**推论**：densify 是目前唯一真正解耦的动作 —— 只动 λ 位置，不动态数、不动窗口、
不动采样量。但它只在初始布点时跑一次，不是运行时动作。

---

## 8. 结构性事实（供判断重构代价）

```
run_all_windows          ibs_engine.py:14056     4253 行
run_full_pipeline        abfe_pipeline.py:11014  2965 行
bias_status / bias_converged / f_k_evidence_status 写点   25 处
  分三区：构造 8037-8072、resume 重建 9660-9733、尾部裁决 16804-17001
run_all_windows 内 raise / break / continue / return      50 处
```

`resume 重建`那 70 行是**第二套**从盘上推状态的规则，与尾部裁决那套是两份独立
逻辑，靠人工保持一致。

三层各用一种控制机制：**if/else 尾巴（窗口内） / 异常展开（路径） / while 循环（补采）**。

---

## 9. 别在重新设计时重新论证的既有结论

1. `stage2_n_states=9` 是探针网格，`stage2_final_n_states=16` 才是生产态数 —— 两个不同的量。
2. 探针网格**不能低于 9**（7 点时布点偏离从 0.009 跳到 0.064，L 开始系统性偏）。
3. 相邻态 fixed-H overlap **不是** IBS 收敛的仲裁 —— 这是被否决过的设计错误。
4. 占据（occupancy）已退役为 diagnostics-only，不得再进 `converged`。
5. 改 `stage2_n_states` / `pilot_n_steps_per_state` 会让所有旧 preopt 缓存失配并重跑 pilot。
6. `n_workers` 当前无效，窗口级多 GPU 并行未实现。
