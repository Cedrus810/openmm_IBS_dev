# 巨函数动刀计划（v2）

**2026-09-19。工作区间：`refactor/` 目录内，仅此。**
主线（`abfe_core.py` / `ibs_engine.py` / `abfe_pipeline.py` / `abfe_preoptimizer.py` /
`runabfe.py` / `tests/`）**只读，作对照**。

> v1 的做法是「把支配块整块搬走」。**那是硬切**——5007 行的函数换成 4736 行的方法，
> 除了换个名字什么都没变。v2 改为**按实测的状态流最细处下刀**，并把参数面
> 工程化成数据类。本文件替换 v1。

---

## 0. 目标：让 agent 改得动

**工程化不是为了架构好看，是为了 agent 能安全地改这份代码。**
所有设计取舍以此为准——下面五条是验收判据，不是风格偏好。

| # | 判据 | 现状为什么不行 |
|---|---|---|
| 1 | **一个单元能被一次读完并推理**。目标 **≤400 行** | 4736 行的窗口体、2779 行的 `_decide_once`，读不全就改 = 猜 |
| 2 | **依赖显式且局部** | 125 个变量跨相位存活、隐式模块全局、散落的 `self.*`——agent 看不见自己碰了什么 |
| 3 | **一个关注点一个可 grep 的名字** | 现在只能按行号定位（「L3553 那个 `if bias_converged`」） |
| 4 | **改动影响面可见** | 一坨 4736 行意味着「谁还会受影响」无法回答 |
| 5 | **不变量有机械门钉着** | 已有 343 条源码文本断言在做这事；A1 让它们在重构后仍然有效 |

### 0.1 本仓 agent 实际栽过的三种坑——都跟代码形式直接相关

这三条不是设想，是这个仓库今天/近期的实测事故，每一条都指向一个设计规则：

| 事故 | 形式上的成因 | 规则 |
|---|---|---|
| **09-19 真机 9/9 崩**：`TypeError: vanishing_subdomain_ranges_from_lambdas() got an unexpected keyword argument 'last_window_max_states'` | 往 `_stage_range_kwargs` 塞了一个键，该 dict 被 `**` splat 进不含此形参的函数 | **显式形参，不 splat 裸 dict**。参数不匹配要在构造点炸，不要在展开点炸 |
| **`abfe_pipeline.py:16315` 指纹恒 `None`** | 身份元组 9 个键、旁边的 `derived_path_params={...}` 字面量 8 个键，靠人手同步 | **派生，不并列**。同一份清单只许有一个来源 |
| **「同一不变量四份实现」**（本仓历史最贵的 bug 形状） | 同一个量在四处各写一遍 | **一个不变量一处** |

**冻结数据类对 agent 的价值正是这三条**，不是"现代"：

- `@dataclass` 替代 dict-splat ⟹ 字段不匹配在**构造时**报错，带字段名；而 `**kwargs`
  要到被调函数展开时才炸，且错误信息只有一个键名（就是 09-19 那条）。
- 数据类的字段声明是**唯一来源** ⟹ 不会再出现「元组 9 个、字面量 8 个」。
- `frozen=True` ⟹ agent 不可能误改输入；而且本函数的 40 个入参**实测全只读**，
  这个不变量是可证的，不是约定。
- 有类型标注 ⟹ agent 不用读 4736 行去猜 `production_step_overrides` 是 dict 还是 list。

### 0.2 散落的初始化是 agent 最大的噪声源

相位 2 的 621 行里，**51/107 条是纯字面量初始化**（`is_resumed_ibs = False`、
`steps_at_full_bias = 0`、`validation_batch_history = []` …），散布在 880 行跨度上。

它们不是代码，是**字段声明写成了语句**。归成 `@dataclass class WarmupState` 之后：
agent 要知道「预热阶段有哪些状态、初值是什么」只需读一个 60 行的声明，
而不是在 880 行里找 51 个赋值。这一笔是本计划里**收益/风险比最高**的动作。

---

## 1. 三条原则

**① 刀口选在状态流最细处，不选在行号上。**
每个候选边界都先量「跨边界活跃变量数」，取最小的那个。没量过的边界不下刀。

**② 每刀拆成 A/B 两段，各有各的机械门。**

| 段 | 做什么 | 门 |
|---|---|---|
| **A 纯搬运** | 按相位切开，跨界变量**按原名**当形参 | `verify_extract.py`：AST 逐节点等价 |
| **B 工程化** | 形参归组成冻结数据类，引用改 `cfg.x` | `verify_rename.py`：AST 在「`Name(x)` ↔ `Attribute(cfg,x)`」替换下等价 |

分开做的理由：B 会改动 4000 行里的名字引用，A 的等价门管不住；
拆成两段之后两边都还有可证的门。合成一刀就只能靠眼睛。

**③ `refactor/` 跑不起来（缺 import、无 `__init__.py`），所以没有测试能兜底。**
AST 等价是唯一机械可证的判据，它**证不了运行时行为**。
因此本计划**全部是保结构变换**，零行为改动。真正的行为验证等主线解冻后跑全量，
不在本计划内。

**每刀回滚**：动刀前 `cp <目标文件> <scratchpad>/<名字>.before.py`。没有留底不开刀。

---

## 1. 现状

15 个 ≥800 行的定义 = **50360 行 = 全仓一半**。

| 行数 | 定义 | 文件 | 本计划 |
|---:|---|---|---|
| 15442 | `ABFEPipeline`（类） | pipeline_orchestration/abfe_pipeline_class.py | 经刀2 |
| 6360 | `Stage2RepairController`（类） | stage2_controller/stage2_repair_controller.py | 经刀3 |
| 5405 | `IBSWindowManagerDualLambda`（类） | windows_sampling/window_manager_dual_lambda.py | 经刀1 |
| **5007** | **`run_all_windows`** ← WIRING.md 漏登记 | 同上 | **刀1** |
| 3372 | `run_full_pipeline` | abfe_pipeline_class.py | **刀2** |
| 2779 | `_decide_once` | stage2_repair_controller.py | **刀3** |
| 1769 | `main` | runabfe_main.py | 不在本轮 |
| 1553 | `_run_stage2_autonomous` | abfe_pipeline_class.py | 不在本轮 |

`ABFEPipeline` 的 15246 行里，95 个方法只有 3 个 ≥1000 行；其余 86 个都 <300 行。
**拆 `ABFEPipeline` 实际上是拆 3 个方法。**

---

## 2. 刀1 —— `run_all_windows`（5007 行）

### 2.1 实测相位结构

循环体 `for window_idx, (start, end) in enumerate(self.ranges)` 在 **L554-5290**，
4736 行、270 条顶层语句、嵌套深度 16。按内部大块切成 8 个相位：

| 相位 | 行区间 | 行数 | 语句 | 自产变量 | **需要上游** |
|---|---|---:|---:|---:|---:|
| 1 窗口准备（建系/Simulation/checkpoint 恢复/sampler 构造） | 555-1203 | 469 | 42 | 62 | **0** |
| 2 预热状态初始化（~60 个标量累加器） | 1204-2088 | 621 | 107 | 109 | 10 |
| 3 预热循环（`while` 单条语句） | 2089-2865 | 774 | 1 | 76 | 21 |
| 4 预热收口/诊断（`bias_warmup_diag` 175 行） | 2866-3552 | 605 | 20 | 54 | **51** |
| 5 f_k 冻结/提交（`if bias_converged:` 单条） | 3553-3959 | 405 | 1 | 17 | 32 |
| 6 生产准备 | 3960-4450 | 364 | 65 | 82 | **9** |
| 7 生产循环（`for up in range(n_updates)` 单条） | 4451-4828 | 375 | 1 | 58 | 23 |
| 8 生产收口/落盘（`convergence` 字典 169 行） | 4829-5290 | 392 | 33 | 35 | 22 |

跨相位存活变量共 **125 个**。存活最久的是窗口级对象：
`sampler`(跨 7 相位)、`lc_win`(7)、`lv_win`(6)、`ibs_state_file`(6)、`K`(4)、
`sim` / `win_sys` / `win_sys_xml`(2)。

### 2.2 刀口就在 5/6 之间

「需要上游」这一列就是刀口的价码：

```
相位 6 需要 9 个   ← 最细，刀口在这
相位 2 需要 10 个
相位 3 需要 21 / 相位 8 需要 22 / 相位 7 需要 23
相位 5 需要 32
相位 4 需要 51     ← 最粗，绝不在这切
```

所以刀1 的第一刀是**把窗口体一分为二**，在相位 5/6 之间：

```python
for window_idx, (start, end) in enumerate(self.ranges):
    if _only is not None and window_idx not in _only:
        continue
    ctx = self._prepare_window(window_idx, start, end, ...)      # 相位 1
    warm = self._warmup_window(ctx, ...)                         # 相位 2-5，2874 行
    if warmup_only:
        continue
    self._produce_window(ctx, warm, ...)                         # 相位 6-8，1131 行
```

### 2.2.1 但 2874 行不达标（§0 判据 1 要 ≤400），所以刀1 必须走到第二刀

按老判据（"一刀切掉支配块"）到这里就完了。按 **agent 能改得动**这个判据，
`_warmup_window` 2874 行仍然读不全，等于没改善。所以刀1 是**两刀连做**：

**第二刀**：把相位 2-5 按相位边界再切开。挡路的是相位 4 需要 51 个上游变量——
而这 51 个里的绝大多数，正是 §0.2 那 51 条散落初始化产生的标量。
**先建 `WarmupState` 数据类把它们收拢，那条 51 的边界就自动变细**：
相位 3/4/5 之间传的不再是 51 个裸变量，而是一个 `state` 对象。

```
相位 2 → @dataclass WarmupState 的字段声明（51 条初始化）+ ~20 条真实逻辑
相位 3 → _run_warmup_loop(ctx, cfg, state)          774 行 ← 仍需第三轮
相位 4 → _summarize_warmup(ctx, cfg, state)         605 行 ← 仍需第三轮
相位 5 → _commit_frozen_f_k(ctx, cfg, state)        405 行
相位 6 → _prepare_production(ctx, cfg, state)       364 行 ✅
相位 7 → _run_production_loop(ctx, cfg, prod)       375 行 ✅
相位 8 → _finalize_window(ctx, cfg, prod)           392 行 ✅
```

**顺序很重要**：`WarmupState` 必须**先于**相位 2-5 的切分建好，否则就是在
51 个裸变量上硬切。这也是 §0.2 说它"收益/风险比最高"的原因——它既是
独立的净收益，又是后续每一刀的前置。

### 2.3 A/B 两段

**A 段（纯搬运）**：`_warmup_window` / `_produce_window` 的跨界变量**按原名当形参**，
函数体一字不改 ⟹ `verify_extract.py` 严格 PASS。
签名会很长，但这是**该函数真实耦合面的第一次显式化**，不要用 `**kwargs` 藏回去。

**B 段（工程化）**：`run_all_windows` 的 40 个入参实测成 6 个内聚簇，**0 个未归类**：

| 数据类 | 个数 | 体内引用 | 成员 |
|---|---:|---:|---|
| `EarlyStopConfig` | 9 | 30 | `early_stop_*` ×8 + `enable_early_stop` |
| `WarmupConfig` | 12 | 26 | `warmup_*`、`*_bias_updates`、`enable_gradual_warmup`、`initial_f_k_by_window`、`lse_log_residual_tolerance`、`accept_recalibrated_f_k_without_gate`、`allow_best_effort_warmup` |
| `StepBudget` | 6 | 30 | `n_steps_per_window`、`steps_per_update`、`*_step_overrides`、`mbar_calibration_reserved_steps`、`frozen_validation_is_final_rung` |
| `RuntimeContext` | 5 | 9 | `platform`、`props`、`positions`、`initial_velocities`、`resolved_box` |
| `RepairPolicy` | 3 | 20 | `repair_policy`、`legacy_repair`、`provisional_production_windows` |
| `RunMode` | 4 | 78 | `stage_type`(×57)、`debug_mode`(×16)、`resume`、`_only` |

全部 40 个**实测只读**（唯一同名可疑的 `x` 是各处推导式的局部名），
所以用 `@dataclass(frozen=True, slots=True)`，不可变是**可证的**、不是约定。
新签名一律 keyword-only（`*`），杜绝 40 个位置实参错序。

另建两个**可变**状态对象承载 125 个跨相位变量：
`WindowContext`（相位 1 产出，之后只读：`sampler`/`sim`/`win_sys`/`lc_win`/`lv_win`/`K`/路径们）、
`WarmupState`（相位 2-5 的累加器）。

**B 段的门**：`verify_rename.py`（待写）——把改动后 AST 里的
`Attribute(Name(cfg), x)` 还原成 `Name(x)`，再与改动前逐节点比对。
只允许出现在白名单里的那 40 个名字上，多一个都算失败。

### 2.4 预期与达标情况

按 §0 判据 1（≤400 行）逐个对照：

| 单元 | 行数 | 达标 |
|---|---:|---|
| `run_all_windows`（只剩循环与分派） | ~280 | ✅ |
| `_prepare_window` | ~470 | ⚠️ 略超，可接受 |
| `WarmupState` / 6 个 Config 数据类（声明） | ~150 | ✅ |
| `_run_warmup_loop`（相位 3） | 774 | ❌ **第三轮** |
| `_summarize_warmup`（相位 4） | 605 | ❌ **第三轮**（其中 `bias_warmup_diag` 字典占 175 行，本身是数据不是逻辑） |
| `_commit_frozen_f_k`（相位 5） | 405 | ⚠️ 临界 |
| `_prepare_production`（相位 6） | 364 | ✅ |
| `_run_production_loop`（相位 7） | 375 | ✅ |
| `_finalize_window`（相位 8） | 392 | ✅（其中 `convergence` 字典占 169 行，同上） |

**本计划做到「8 个单元里 5 个达标」**。剩下三个（相位 3/4/5）留第三轮，
因为它们内部还没量过相位——按 §0① 没量过不下刀。

⚠️ 一个诚实的限定：相位 4 和 8 各有一个 175 / 169 行的字典字面量
（`bias_warmup_diag` / `convergence`）。那是**输出契约的数据**，不是逻辑；
把它们抽成构造函数只是搬家，对 agent 的可改性帮助有限。真正该做的是
让它们有 schema（`TypedDict` 或数据类），这样 agent 加字段时知道有谁在读——
但那要先查清读侧，**不在本计划内**。

---

## 3. 刀2 —— `run_full_pipeline`（3372 行）

目标块 `if decoupling_scheme == 'dual_lambda':` 在 **L13053-15442**，体 2389 行。

```
块后语句   0 条   ← 这个 if 一直延伸到函数末尾（也是类末尾）
有 else    是     ← else 也要一起处理，不能只搬 if 体
出流       0 个
入流      28 个   只读 20 / 既读又写 8（cached, f, final, i, k, stage1_states, v, x）
```

**做法**：if 体与 else 体各提一个方法（`_run_dual_lambda_scheme` /
`_run_legacy_scheme`），函数末尾变成二选一分派。
`f/i/k/v/x` 是推导式与循环临时名，提取后自然成为新方法的局部；
**`cached` / `final` / `stage1_states` 三个动刀前单独核**是否块内先写后读。

⚠️ 这刀**未按 §0① 量过内部相位**。下刀前要先做一次与 §2.1 同样的相位测量，
否则就是硬切。**测量结果出来之前，本节只是占位。**

---

## 4. 刀3 —— `_decide_once`：归位到 `AUDIT-S2-02`

**仓库里已有立项**：`docs/TODO_P2.md` §2 的 `AUDIT-S2-02 [P2]`，
并且指定了拆法——**「把 `decide()` 拆成按证据类型分派的几段」**。
`S2-A` 把自己的正解也指向它；`docs/TODO_P3.md` §5 的
`REL-05` / `REL-06` / `REL-10` 三条全押在「Stage-2 重构落地 + 方法定稿」之后。

**所以刀3 不是独立的刀，是 `AUDIT-S2-02` 的第 0 步。**

### 4.1 三个数要更新进 TODO

| | TODO 记的（2026-09-16） | 2026-09-19 实测 |
|---|---:|---:|
| `_decide_once` 行数 | 1795 | **2779**（+984） |
| `return` 出口 | 63 | **72**（自身 54 + 内嵌 def 内 18） |

TODO 自己那句「📌 这个数在变大」又应验一次——09-17 那波控制器修复把它从 1795 推到 2779。
`refactor/` 那份与主线逐字一致（同为 2779 / 72），工作台是当前的。

### 4.2 TODO 没记的结构事实：出口已经收口了

**54 个自身出口里 51 个是 `return plan(...)`。**
也就是说「按证据类型分派」的锚点是现成的：把 51 个 `return plan(...)` 站点
按**驱动它们的证据**归组即可，拆法不用重新发明。

### 4.3 第 0 步（本计划内，机械）

提出 8 个内嵌 `def`，共 886 行，捕获的外层变量改显式形参：

| 行数 | 名字 | 捕获 |
|---:|---|---|
| 646 | `plan` | `_ctx`, `feas`, `view` |
| 134 | `_frames_admission` | `_blocks`, `_blocks_total`, `_blocks_total_u`, `_blocks_u`, `_max_blocks`, `view` |
| 30 | `_is_noop` | `_by_idx_for_noop`, `_by_uid_for_noop`, `_noop_led`, `view`（自带内嵌 `_match`） |
| 19 | `_attr_of` | `view` |
| 17 | `_bounded_rewindow` | `blocked`, `earliest`, `view` |
| 15 | `_pick` | `earliest` |
| 15 | `_is_skew` | `view` |
| 10 | `_pw_or_last` | **0** |

`view` 被 6/8 捕获 ⟹ 统一放第一个形参位。
**顺序**：捕获数从少到多，一个 `def` 一刀，逐刀验。
**做完之后**：`_decide_once` 2779 → ~1893 行，119 条顶层语句、51 个 `return plan(...)` 的
分派骨架露出来——第 1 步才有得做。

### 4.4 第 1 步（**不在本计划内**）

51 个出口按证据类型归组 → 分派几段。**这不是保结构变换**，
`verify_extract.py` 管不住它，需要另立判据（至少要能证明「同一输入下走到同一个
`plan(...)` 出口」）。而且它会动分支顺序——`AUDIT-S2-02` 原文点名
「挡不住分支顺序错」正是这里。**要单独立项、单独定门。**

---

## 5. 刀序

按「收益/风险比」而不是「行数」排。第一刀故意选小的，用来验证门本身可靠。

```
0. 刀3 第0步          886 行，8 次独立小搬运        ← 验门
1. WarmupState 数据类  51 条散落初始化 → 一个声明    ← §0.2，收益/风险比最高，且是后续前置
2. 刀1 A 段            相位 5/6 切开（状态流最细处）
3. 刀1 B 段            6 个冻结 Config + WindowContext + ProductionState
4. 刀1 第二刀          相位 2-5 按边界切开（靠 WarmupState 把 51 的边界压细）
5. 刀2                 先补相位测量，再定
```

⚠️ 第 1 步（`WarmupState`）**不是纯搬运**：把 51 条散落赋值收成数据类字段，
会改变「这些名字在哪个作用域里」。`verify_extract.py` 管不住它。
它的门是 `verify_rename.py` 的变体：证明**每一条被移走的初始化，其变量名与初值
在数据类字段里逐一对应**，且函数体内对这些名字的读写全部改成了 `state.x`。
名单是那 51 条，多一条少一条都算失败。

---

## 6. 本计划**不做**

- 不动主线任何文件（含 `tests/`）。
- 不做 `AUDIT-S2-02` 的第 1 步（证据类型分派）——另立项。
- 不拆 `_warmup_window` 的第二轮（2880 行仍在）。
- 不动文件边界——只改「定义大小」，不改「哪个定义住哪个文件」（域划分是 B 阶段）。
- 不合并重复实现、不删死代码、不补 import、不建包。

---

## 7. 每刀收尾

```
① 留底  cp <目标文件> <scratchpad>/<名字>.before.py
② 切
③ 验    A 段: verify_extract.py <留底> <目标> --src ... --moved ...      必须 PASS
        B 段: verify_rename.py  <留底> <目标> --bind cfg --names <白名单>  必须 PASS
④ 复查  ast.parse 语法合法
        python3 refactor/tools/deps.py    跨域引用数**不应变化**
⑤ 报    报实测行数变化与 verify 输出原文；失败就报失败
```

`deps.py` 那条的意义：拆函数不应该改变跨域引用数。变了说明动到了符号归属，
回滚重来。
