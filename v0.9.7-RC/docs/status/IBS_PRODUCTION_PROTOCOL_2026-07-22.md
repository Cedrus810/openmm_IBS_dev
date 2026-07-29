# IBS 预热、生产与 Coverage Rescue 协议（2026-07-22）

状态：**代码已实现；静态与源码契约测试通过；目标 OpenMM/CUDA 运行验证待完成。**

本文是 2026-07-22 起 Stage 2（vanishing/vdw）IBS 行为的权威说明。它取代
`NON_MUTATING_V1_STATUS.md` 中以下旧结论：

- 冻结占据残差必须严格通过，才允许进入生产；
- 生产质量失败只能终止并交人工；
- 不允许自动创建新的 rescue sampling ensemble。

仍然有效的核心约束是：**生产阶段的 `f_k` 永远只读；旧生产数据不原地改写；
λ 坐标不在 rescue 中自动插值或移动。**

---

## 1. 为什么修改

真实 GPU 日志暴露了两个被混淆的阶段：

1. 预热的任务是为 production 选择一组可用的固定偏置 `f_k`；
2. production 的任务是在该固定 Hamiltonian 下产生正式统计样本。

旧控制流把预热占据平坦度当成最终正确性门，导致一个窗口最多消耗
`570000` 步预热、反复进行四次完整冻结验证，仍在 production 开始前抛出
`IBSWarmupConvergenceError`。这使预热比正式的 `250000` 步 production 更昂贵，
也让真正应该根据 production 样本计算的 overlap/ESS 没有机会运行。

此外，stage-level 失败曾只打印全局 `min_overlap`，例如 `0.006519 < 0.05`，
却不说明具体窗口、目标 λ 态和失败指标，无法执行有针对性的补采。

---

## 2. `f_k` 的符号约定

IBS 偏置下态权重为

```text
p_k ∝ exp[β(f_k - U_k)]
```

平坦占据要求

```text
f_k = F_k + constant
```

因此 pilot TI 得到的物理自由能曲线在 mean-center 后可直接作为 `f_k` 种子，
不需要取反。

2026-07-22 保留的修复：

- `abfe_preoptimizer.py::estimate_f_k_from_pilot_ti()` 删除
  `f_at_target = -f_at_target`；
- `ibs_engine.py::_solve_tmbar_and_recenter()` 使用
  `[f_by_lambda[k] ...]`，不再使用其相反数；
- 在线占据负反馈仍为
  `delta_f = -eta * kT * log(K * p_k)`，其方向本来正确，未取反。

物理解释：自然占据过高的态需要更低的 `f_k`；自然占据过低的态需要更高的
`f_k`。

---

## 3. 新的预热 → 生产状态机

### 3.1 Learning

- bias ramp 后允许 `update_weights()` 调整 `f_k`；
- learning 只产生预热信息，不产生 production 样本；
- 得到候选后进入固定 `f_k` 的 burn-in/validation。

权重控制器采用全历史 TMBAR 自洽绝对更新
（`IBS_WARMUP_UPDATE_PROTOCOL_VERSION=9`）：

- learning 采用统一的定长 minibatch：每累计
  `IBS_TMBAR_LEARNING_MINIBATCH_FRAMES=20` 帧固定 `f_k` 采样后自洽更新一次，
  不再按“严重塌缩 / 中间区 / 稳定区”分成 `40/60/100` 三档；warmup 帧间隔为
  `250 MD steps`。production 的 `steps_per_update=500` 不变；
- 每个新 block 连同此前所有时变偏置 block 一起进入 `tmbar_history`，通过
  `solve_stage_integrated`/MBAR 自洽求出绝对物理自由能候选 `f_target`；
- TMBAR 可解时严格执行
  `f_new = f_old + 0.20 * (f_target - f_old)`，随后 mean-center 去除 gauge。
  若目标在后续迭代中不变，10 次更新走完 `1-0.8^10≈89.3%` 的距离；
- 当前 raw batch 占据、EMA 与 dominant 身份**只作诊断**，不再控制更新方向、
  升档、刹车、重置，也不再作为冻结候选门；累计样本降低的是估计方差，标准误差
  按有效独立样本数约以 `sqrt(20/N_total)` 缩小；
- 若累计历史暂时不足以给出 TMBAR 解，才退回一次固定 `10 kT` pairwise 上限的
  有界占据更新；该上限不随 dominant 升级或重置；
- **冻结候选门** = 当前累计 TMBAR 固定点的剩余阻尼步长已经收缩：即本次应用的
  `0.20*(f_target - f_old)` 的 pairwise spread `≤ IBS_TMBAR_FREEZE_MAX_APPLIED_
  PAIRWISE_STEP_KT = 1.0 kT`（`tmbar_update["converged"]`）。旧判据对每个历史
  minibatch 取最差 ESS，一个早期坏块会把候选永久卡死，现只保留为
  `legacy_per_entry_quality_converged` 质量诊断。raw log 残差不再阻止进入冻结；
  仍需要连续 3 个新的固定块同时通过该步长门，随后由独立的 fixed-`f_k` 验证做
  权威终检。
- 旧控制器尚未完成的 learning checkpoint 不注入 v9；已经完成并锁定的 production
  数据不因纯预热控制器升级而失效。

### 3.2 一次完整 fixed-`f_k` 验证

- 先丢弃 `20000` 步 freeze burn-in；
- 随后完成一个足额 fixed-`f_k` attempt，默认 `50000` 步、100 frames；
- attempt 期间不调用 `update_weights()`；
- 占据残差 `max|log(K·p_k)|` 记录为偏置效率诊断。

### 3.3 有界预热终止

生产运行的默认值为：

```text
max_frozen_validation_cycles_before_accept_best = 1
```

第一个完整 fixed-`f_k` attempt 结束后：

- 无论占据残差是否达到 ideal tolerance，都锁定该 attempt 的 `f_k`；
- 不执行“受限占据负反馈，恢复 learning”；
- 不再进行第 2、3、4 次完整冻结验证；
- 预热残差只表示预计重加权效率，不再作为 production 准入硬门；
- `warmup_only=True` 的设计审计没有后续 production 门兜底，仍要求结果落入
  sane bound。

如果从未得到任何完整 fixed-`f_k` attempt、出现非有限能量/受力、CV 全零或系统
构造失败，仍然 fail closed。

---

## 4. Production 的不可变契约

进入 `# ---------- 生产采样 ----------` 后：

1. 记录 `production_f_k_lock`；
2. 入口 `f_k` 必须与 production manifest 的冻结值一致；
3. 每个 production update 后重新读取全部 `f_k`；
4. 任一分量变化超过 `1e-12 kJ/mol`，立即抛错并停止；
5. production 段禁止调用：
   - `sampler.update_weights()`；
   - `_solve_tmbar_and_recenter()`；
   - `_bounded_log_occupancy_update()`；
   - `Context.setParameter(..._f_k...)`。

生产阶段可以读取 `f_k`，不能更新、重新校准或恢复另一组 `f_k`。

### 4.1 数据边界

- learning、freeze burn-in、fixed-`f_k` validation 的帧全部不计入 production；
- 首次 production 从 0 步开始；
- 生产历史唯一合法的续接来源，是同一 λ、同一 frozen `f_k`、同一 manifest 的
  production checkpoint；
- 新窗口完成 `250000` 步时，默认保存 500 个 production frames
  （`steps_per_update=500`）。

---

## 5. 缓存版本兼容

`IBS_BIAS_PROTOCOL_VERSION` 的规范版本保持为 `27`。

此前误升的 v28 只改变 warmup 的停止/诊断控制，没有改变 production Hamiltonian、
`f_k` 符号或 production 采样分布。因此缓存兼容集合为：

```text
IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS = {27, 28}
```

结果：

- 已完成的 v27 production 不会因为 warmup 控制变化而重跑；
- 已经写出的 v28 production 也不会在撤回误升版后被反向判废；
- v26 及更早、仍可能使用错误 `f_k` 符号或不同生产协议的状态不兼容；
- manifest 比较仍严格检查 λ、系统、WCA/LRC、repair policy、冻结 `f_k` hash、
  生产目标步数等其它字段。

---

## 6. Production 质量失败的精确定位

`min_overlap` 在当前 Stage 2 分析中定义为：

```text
每个 IBS production mixture → 窗口内各目标 λ 态的最小 importance-ESS ratio
```

它不是相邻 fixed-H 窗口之间的双向 overlap。

stage-level 失败现在必须打印每个失败 ensemble 的：

- `window_index` / `window_label`；
- 半开 `window_range`；
- 全局 state indices；
- `lambdas_coul` / `lambdas_vdw`；
- ESS 最差的全局 state 与实际 λ；
- `min_ess_ratio`；
- `absolute_ess`；
- `n_frames_decorrelated`；
- `endpoint_diff_uncertainty_kJ_mol`；
- 失败门列表：`ess_ratio`、`absolute_ess`、`decorrelated_samples`、
  `endpoint_uncertainty`。

不能再只输出一个全局 `min_overlap` 后要求人工猜窗口。

---

## 7. 自动 production coverage 补采

Stage 2 首轮 production 分析失败后，默认执行最多两轮定向补采：

```text
stage2_production_rescue_rounds = 2
stage2_production_rescue_growth = 2.0
```

默认累计目标：

```text
250k → 500k → 1M production steps
```

规则：

- 只扩展失败窗口；通过的窗口继续命中缓存；
- 强制使用 resume，续接现有 production checkpoint；
- 已有 production frames 原样保留，新帧追加；
- 沿用同一组冻结 `f_k`；
- 不重新 learning、不修改 `f_k`、不清空已有 production；
- 每一轮重新计算全部 final gates。

增加样本通常能改善绝对 ESS、去相关样本数和不确定度。ESS ratio 是相对指标，若
加倍样本后仍不改善，说明问题更可能是单个 sampling ensemble 跨度太大，而不是
单纯缺少帧数。

---

## 8. Immutable rescue ensembles

如果定向补采后仍未通过，且 `stage2_enable_bridge_rescue=True`（默认），代码会：

1. 保留原 λ 网格；不插入、不移动 λ；
2. 保留原窗口的全部 production 文件，不删除、不覆盖；
3. 对失败的 `[start,end)` ensemble 使用现有 state 节点生成两个更小、共享一个
   边界 state 的 rescue ranges；
4. 在独立目录创建新的 IBS sampling ensembles：

```text
output/vanishing_rescue/<plan_id>/
checkpoint/vanishing_rescue/<plan_id>/
```

5. 每个新 ensemble 独立预热一次，随后锁定它自己的 `f_k` 并进行 production；
6. combined analysis 保留所有正常原窗口，用 rescue ensembles 替换失败原窗口的
   分析覆盖；失败原窗口数据仍原样留在磁盘，不参与该次最终拼接；
7. 结果记录 `immutable_bridge_rescue`：被替换的原窗口、rescue ranges、目录和
   plan ID。

例如失败范围 `(6,11)`（states 6–10）会生成：

```text
(6,9)  # states 6,7,8
(8,11) # states 8,9,10
```

两个新 ensemble 共享 state 8，并覆盖原范围的所有物理 state。若 rescue 后仍未通过，
pipeline 最终 fail closed，并在异常中列出新的精确瓶颈，不无限创建 ensemble。

### Stage 2 v20 λ 尾部与 coupled-end bridge 覆盖

真实 v18 运行随后证明，仅用已有 state 拆小 ensemble 无法修复原路径本身的
`0.838237→0` 空洞。v19 因此把 17 点生产基础路径固定为
`λ=x², x=linspace(1,0,17)`；Fisher pilot 保留为诊断，但不再有权删除 λ≈0 的
几何覆盖。λ≈1 一端原有的四点增密仍保留，最终精确 21 点路径的最后四点为：

```text
0.03515625, 0.015625, 0.00390625, 0.0
```

v20 保留上述平方锚点，并利用 pilot 的 `∫√g dλ` 在两条最长生产边各插入一个
热力学中点。本体系得到 `0.9803043327` 和 `0.9628845428`，将原第一个窗口拆成
两个共享单一边界的 ensemble；最终为 23 态、6 个窗口、28 个采样槽位。
`THERMODYNAMIC_PATH_PROTOCOL_VERSION=20` 使旧 v18/v19 缓存 fail closed。已经启动的
旧进程不会在运行中改变 Hamiltonian 网格；必须由新进程加载 v20。生产阶段仍不得修改 `f_k`。

配置：

```text
stage2_enable_bridge_rescue = True
stage2_bridge_production_steps = n_steps_per_window  # 默认 250k
```

---

## 9. 主要代码位置

| 行为 | 文件/符号 |
|---|---|
| pilot TI 不取反 | `abfe_preoptimizer.py::estimate_f_k_from_pilot_ti` |
| TMBAR 候选不取反 | `ibs_engine.py::_solve_tmbar_and_recenter` |
| v27/v28 缓存兼容 | `ibs_engine.py::IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS` |
| 一次完整验证后结束预热 | `ibs_engine.py::IBSWindowManagerDualLambda.run_all_windows` |
| 预热/生产数据隔离 | 同上，production history 初始化段 |
| production `f_k` 硬锁 | 同上，`production_f_k_lock` |
| 窗口/λ 精确失败定位 | `abfe_pipeline.py::_stage_quality_failure_details` |
| v20 平方锚点、Fisher bridge 与缓存门 | `abfe_preoptimizer.py::quadratic_vanishing_base_lambdas` / `insert_fisher_bridge_lambdas` / `validate_human_vanishing_anchors_preserved` |
| 定向 production 补采 | `abfe_pipeline.py::run_full_pipeline` Stage 2 rescue loop |
| 新 rescue ranges | `abfe_pipeline.py::_build_vanishing_rescue_ranges` |
| 独立 rescue 数据加载/拼接 | `abfe_pipeline.py::_load_ibs_window_outputs_from_dir` |

---

## 10. 验证状态

本机已完成：

- Python AST 语法解析：通过；
- `test_audit_protocol_regressions.SourceContractTests`：**32/32 通过**；
- 失败定位纯函数测试：能从 `min_overlap=0.006519` 的构造诊断中定位
  `window 2 / worst state 8 / λ_vdw=0.3`；
- rescue range 纯函数测试：`(6,11) → (6,9)+(8,11)` 通过；
- production 源码契约确认无可执行 `update_weights` / `setParameter(f_k)` /
  TMBAR solve / bounded update。

本机未完成：

- 当前 Windows Python 未安装 OpenMM，无法执行真实 Context/GPU 测试；
- 自动补采的真实 checkpoint 追加行为需要在目标 Linux/OpenMM/CUDA 环境确认；
- immutable rescue ensemble 的真实采样和 combined TMBAR 拼接需要目标环境确认。

目标环境最低验收：

1. 用现有 Stage 2 失败输出 `--resume`；
2. 日志明确指出失败 window/state/λ；
3. 只对失败窗口执行 250k→500k→1M 的 production 追加；
4. `f_k` 在原 production 与追加 production 中逐 update 保持完全一致；
5. 已有 `.npy` 帧数单调增加，不覆盖；
6. 若进入 rescue，原目录 hash/mtime 不变，新文件只出现在
   `vanishing_rescue/<plan_id>`；
7. combined result 记录 `immutable_bridge_rescue`；
8. 最终通过全部 quality gates，或带精确窗口/λ 诊断 fail closed。

---

## 11. 不变量摘要

```text
预热：允许学习 f_k；只做一次完整 fixed-f 效率诊断。
生产：f_k 永远只读；正式样本从 0 步独立计数。
补采：只追加同一 frozen-f_k production。
Rescue：新目录、新 ensemble、原 λ 节点；旧生产不改写。
分析：用 production overlap/ESS/不确定度决定最终可用性。
```
