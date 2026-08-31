# EXP-030 合并主线与换体系验证方案（2026-08-30）

## 0. 目标与决定

本方案完成两件事：

1. 将已经确认并修复的 IBS residual sampling 能力、冻结状态记录链路和对应回归测试整理进真正的上游主线。
2. 将下一个体系作为一次新的、前瞻性冻结的验证；不复用 Atenolol 的结果、seed、checkpoint、输出目录或事后窗口选择。

**当前合并决定：**

- 核心能力可以合入主线，但先保持显式启用，不能默认改变已有 baseline。
- Atenolol 五窗敏感性分析中，三个 repeat 的 ESS/实际时间收益为 +9.01%、+81.79%、+9.44%，说明 residual sampling 在所分析的五窗内有一致且有实际意义的正向收益。
- 完整六窗结果及 window_5 覆盖不足仍保留。不能把“排除 window_5”写成主线默认行为。
- retained-residual target 继续作为分析诊断，不替换 production 的 physical target。
- 新体系通过前瞻性完整协议后，再决定是否把 residual sampling 从“可选实验能力”升级成推荐默认值。

本文件是执行方案，不启动 GPU 任务，也不修改已有生产数据。

---

## 1. 当前工作区事实

当前目录：

```text
D:\ABFE_IBS\Atenolol-rank11
```

不是可用的 Git checkout：现有 `.git` 没有有效的 `HEAD/objects/refs`。因此不能在这里诚实地声称已经完成 commit、merge 或 cherry-pick。

“合并主线”必须在一个真实的上游 Git 工作树中执行。当前目录只作为经过验证的文件级来源和实验档案。

### 1.1 合并前必须确定

| 项目 | 要求 |
|---|---|
| 上游仓库地址 | 使用项目真正的 remote，不在本目录臆造 |
| 主分支 | 由上游仓库实际分支确定 |
| 集成分支 | 从最新主分支创建，例如 `integrate/ibs-residual-sampling` |
| 版本策略 | 先检查上游当前协议版本；不得盲目覆盖成 31 |
| 差异来源 | 对当前文件与上游逐段 diff、挑选语义改动，不整文件覆盖 |
| 输出数据 | 永远不进入代码提交 |

---

## 2. 要合并的数学与运行契约

### 2.1 单个 window 内的 sampling score

对 window 内的局部状态 (k)，candidate 的满强度 sampling score 为：

[
X_k(x)=U^{\mathrm{sc}}_k(x)
      +A_k\,[B_\phi(x)-B_0]
      -f_k .
]

baseline 为同一公式的 (A_k=0) 情形。

用于训练 (f_k) 的能量必须与实际部署的 sampling Hamiltonian 一致：

[
U^{\mathrm{train}}_k(x)
=
U^{\mathrm{sc}}_k(x)+A_k[B_\phi(x)-B_0].
]

这正是 v31 修复的核心：不能再用缺少 residual 的 (U^{\mathrm{sc}}_k) 训练 (f_k)，然后把该 (f_k) 部署到含 residual 的动力学中。

### 2.2 windows 之间的 chain 契约

窗口之间仍通过相邻共享物理态和 Stage-2 integrated MBAR/TMBAR 拼接。共享接口必须满足：

- 相同全局 ((\lambda_{\rm coul},\lambda_{\rm vdw})) 映射；
- 相同 (A_k)、offset 和 target 定义；
- 相邻窗口重叠态代表同一个 Hamiltonian；
- 整条路径的全局首态与末态满足 (A=0)；
- **不要求每个局部 window 的两个端点都把 residual 清零。**

窗口内公式和窗口间拼接是两层不同契约，不能把前者的 endpoint 条件错误写到每个局部窗口末端。

### 2.3 target 与 sampling 必须分开

生产物理目标保持：

[
E_k(x)=U^{\mathrm{sc}}_k(x)+E^{\mathrm{LRC}}_k(x),
]

不含 residual。

保留中间 residual 的分析目标：

[
E_k^*(x)=E_k(x)+A_k[B_\phi(x)-B_0]
]

只用于双目标诊断、差额分解和覆盖审计。它不能：

- 覆盖 physical target；
- 追认旧 physical result 为 PASS；
- 进入正式 production decision；
- 因为更接近 baseline 就被称为真值。

---

## 3. 主线文件分类

### 3.1 必须进入主线的生产实现

#### A. `ibs_engine.py`

逐段移植并复核以下语义：

1. candidate residual 项进入真实 sampling state energy。
2. `sampling_state_energies` 进入 (f_k) 学习链；baseline 在 residual 关闭时保持原行为。
3. 协议版本与 cache compatibility fail-closed；上游若已有更高版本，则分配新的下一版本，不能回写成 31。
4. `production_entry_f_k`：
   - `IBSSampler.__init__` 初始化为 `None`；
   - checkpoint save 写入；
   - load 验证长度与 finite 后恢复，缺失或无效时为 `None`；
   - checkpoint restore 和冻结验证完成后、第一步 production 前记录；
   - resume 保留历史 marker，不用“当前时刻”覆盖；
   - production 结束从真实 Context 刷新最终 (f_k)，再次保存 state。
5. production 期间 (f_k) 锁定，入口 marker、最终 state 与 frozen manifest 分段核验。
6. 同步保存 physical target、actual bias、base energy、sampling state energies 和 residual basis；灾难回退时五本逐帧账同步截断。
7. 旧 checkpoint 不伪造新字段；不满足兼容契约时冷启动或 fail-closed。

#### B. 实际生产接线文件

对上游的 `abfe_core.py`、`abfe_pipeline.py` 和实际 residual force/model 入口逐段检查，只移植确实负责以下接线的改动：

- residual basis force 的构造与独立实例；
- (A_k)、offset、cutoff、skin、Bmax 的传递；
- paired positions、box、velocities 的真实消费；
- production Context 与账本使用同一 score identity。

不得因为文件名相同就整体覆盖。如果生产 residual 实现在 `outer_lambda_neural_basis.py`，只合入生产 runtime 所需部分，不把训练工具和实验分析一起带入。

### 3.2 可进入主线的通用契约模块

如果上游希望长期保留这项可选能力，可合入或重命名为非实验专属模块：

- `exp030_protocol.py`：canonical JSON、score identity、phase cost、failure code、fail-closed 契约；
- `exp030_joint_score.py`：实际 sampling mixture 的 responsibility、占据、自相关和 ESS；
- `exp030_analysis.py` 中真正通用的纯函数。

建议不要让生产包 import 名为 `exp030_*` 的实验模块。更稳妥的主线结构是：

```text
ibs/
  residual_sampling.py
  score_contract.py
  sampling_diagnostics.py
analysis/
  dual_target_reweighting.py
experiments/
  exp030/
```

重命名必须是纯搬迁并有测试保护；不要在一次提交中顺便改公式或判据。

### 3.3 仅保留在实验/分析层

以下内容可以归档或作为可选 CLI，但不能成为生产 pipeline 的依赖：

- `scripts/exp030_window_state_machine.py`
- `scripts/exp030_window_state_machine_em_noresidual.py`
- `scripts/exp030_paired_runner.py`
- `scripts/exp030_paired_utility.py`
- `scripts/exp030_dual_target_reweighting_audit.py`
- `scripts/exp030_convergence_vs_steps.py`
- `scripts/exp030_simple_ab_comparison.py`
- `scripts/exp030_minimal_reference_estimator.py`
- `scripts/exp030_extended_sampling_analysis.py`
- `scripts/exp030_backfill_*.py`
- EXP-030 launcher shell scripts。

双目标分析可复用，但必须明确标记 `diagnostics_only`，不能写 production/final decision。

### 3.4 绝不能合入代码主线

- `output/outer_lambda_exp030/`
- `output_lrc_fix*`
- `rerun_archive/`
- checkpoints、DCD、NPY、solver logs、phase-cost、manifest、failure JSON；
- Atenolol 专属 checkpoint hash、seed、路径和模型结果；
- “自动排除 window_5”；
- “coverage 不过就反复重跑到通过”；
- 把五窗结果冒充完整 ABFE ΔG；
- 把 retained-residual target 改成默认 physical estimator。

Markdown 实验记录可放入 `docs/experiments/exp030/` 作为历史证据，但不应成为生产代码的运行配置。

---

## 4. 建议的提交顺序

在真实上游 Git 工作树中按下列顺序提交，每一步可独立 review 和回滚。

### Commit 1：sampler 数学一致性

范围：

- residual sampling state energy；
- (f_k) training 使用完整 sampling state；
- baseline residual-disabled parity；
- 协议版本和 cache compatibility。

验收：

- residual enabled 时训练能量逐帧等于独立 NumPy reference；
- residual disabled 时结果与上游原 baseline 一致；
- shape、finite、单位和 gauge 检查 fail-closed。

### Commit 2：冻结状态生命周期

范围：

- `production_entry_f_k` 初始化、save、load；
- 正确生产入口记录；
- resume 不覆盖历史；
- production 结束 Context 刷新和再次 save；
- reconcile 分段判断。

验收：

- 真正走 `save → JSON文件 → load → reconcile` 的端到端测试；
- 缺失/损坏 marker 不伪造历史；
- 生产期改动 (f_k) 必须 hard fail；
- 合法入口自我修正和生产期漂移可以区分。

### Commit 3：账本和 score identity

范围：

- 五本账同步；
- frozen score、sampling score、lambda、(A_k)、offset identity；
- checkpoint/restart 连续性；
- paired velocity identity；
- phase cost 包含失败尝试。

验收：

- 数组帧数一致；
- 哈希和 manifest 防止数据误混；
- topology/System 序列化字节 hash 只作为 provenance，不把跨环境的历史字节差异重新设成科学 gate；
- source checkpoint 和数据账本完整性继续强校验。

### Commit 4：通用诊断与分析

范围：

- sampling-mixture ESS；
- ESS–steps；
- dual physical/residual target；
- six-window chain contract；
- solver 与独立 closed-form reference 对账。

验收：

- production 不 import 分析模块；
- alternative target 的 solver gate 标记为 diagnostics-only；
- 不使用 `sampling_states.npy` 替换含 LRC 的 physical energies；
- raw entries 只在 solver 内去相关一次；
- shared interface Hamiltonian 不一致时 hard fail。

### Commit 5：文档和可选实验入口

范围：

- EXP-030 运行器/分析器作为 `experiments/exp030`；
- 本方案、数学定义、失败码和运行说明；
- 不提交任何生产输出。

---

## 5. 主线回归测试门

### 5.1 必须纳入的测试

至少包括：

- `tests/test_exp030_frozen_snapshot_reconciliation.py`
- `tests/test_audit_protocol_regressions.py`
- `tests/test_exp030_joint_score.py`
- `tests/test_exp030_protocol.py`
- `tests/test_exp030_analysis.py`
- `tests/test_exp030_dual_target_reweighting_audit.py`
- `tests/test_exp030_ess_diagnostic_distinction.py`

主线重命名后同步改测试 import，但不改变测试语义。

### 5.2 必须新增或确认存在的 portability 测试

1. 不含字符串 `Atenolol`、`rank11`、固定 output root 或固定 seed 的 core test。
2. (K) 和每窗 state 数可配置；若主线仍只支持六窗，必须显式报错，不能静默截断。
3. 相邻窗口共享态的 lambda、(A_k)、offset、target definition 全一致。
4. 仅全局首末态要求 (A=0)，内部接口按 schedule 原值检查。
5. residual model 对新体系的原子类型/元素覆盖、finite energy/force 和单位检查。
6. baseline 不构造 residual force 时不执行 residual kernel。
7. restart 后同一 frozen score 和同一 marker 能精确恢复。
8. production 期间不调用 `update_weights` 或 recenter。
9. 旧 state 缺新 marker 时 fail-closed，而不是补写虚构值。
10. 不把 full chain 的 `total_error` 换成全局 `df_k` 边际误差。

### 5.3 建议命令

在上游标准环境中执行：

```bash
python -m py_compile ibs_engine.py exp030_protocol.py exp030_joint_score.py exp030_analysis.py
pytest -q \
  tests/test_exp030_frozen_snapshot_reconciliation.py \
  tests/test_audit_protocol_regressions.py \
  tests/test_exp030_joint_score.py \
  tests/test_exp030_protocol.py \
  tests/test_exp030_analysis.py \
  tests/test_exp030_dual_target_reweighting_audit.py \
  tests/test_exp030_ess_diagnostic_distinction.py
```

然后运行上游完整测试集。若完整测试集存在历史失败，必须记录基线失败集合，再证明本分支没有新增失败；不能只报“多数通过”。

额外硬编码扫描：

```bash
rg -n "Atenolol|rank11|output_lrc_fix|three_repeat_final_analysis" \
  ibs_engine.py abfe_core.py abfe_pipeline.py ibs analysis
```

### 5.4 代码合并成功判据

满足以下条件即可合入主线的“可选能力”，不要求先完成新体系 GPU production：

- 上述目标测试全部通过；
- 完整测试没有新增回归；
- baseline parity 通过；
- marker 真链路端到端测试通过；
- core 无体系专属路径和 seed；
- production 不依赖实验分析；
- feature/config 默认关闭；
- 文档说明 checkpoint 不兼容边界和 rollback 方法。

---

## 6. 新体系：独立预注册

新体系不能直接改写：

```text
protocols/EXP-030_joint_state_score_preregistration_FROZEN_PRODUCTION.json
```

应创建新的 experiment ID、独立 JSON、独立 output namespace 和独立最终报告。例如：

```text
protocols/EXP-031_<system>_residual_sampling_DRAFT.json
output/outer_lambda_exp031_<system>/
EXP-031_<SYSTEM>_FINAL_STATUS_<date>.md
```

EXP-031 只是示例编号；最终编号以项目登记为准。

### 6.1 运行前冻结清单

| 类别 | 必须冻结 |
|---|---|
| 体系 | topology、System、positions、box、温度、单位、原子映射 |
| 起点 | 每个 repeat × window 的 source checkpoint、positions/box/velocities |
| 路径 | window 顺序、每窗 state 数、全局 lambda index、共享接口 |
| residual | model/runtime identity、(A_k)、offset、cutoff、skin、Bmax、零点规范 |
| arms | baseline residual off；candidate residual on；双方独立 cold-start (f_k) |
| 配对 | 至少3个独立初态、唯一 seed、预定 AB/BA 顺序 |
| 预算 | warmup、validation、production、query、checkpoint cadence |
| 指标 | ESS 定义、ITT 成本范围、ΔG target、coverage、health、decision gate |
| 失败策略 | retry、保留成本、continue/stop 规则 |
| 输出 | 新 namespace，禁止读取 Atenolol output |

体系及模型 hash 用于 provenance 和防误混。不要把跨机器序列化字节差异当作新的科学判据；checkpoint、lambda、账本和 model identity 仍必须可靠核验。

### 6.2 模型适用性预检

在任何 production 前完成：

- 新体系元素、原子类型和局部环境在 residual model 支持域内；
- (B_\phi)、energy 和 force 全部 finite；
- 量纲为 kJ/mol 与 kJ/mol/nm；
- Bmax/saturation 契约与训练时一致；
- cutoff/skin/periodic box 合法；
- (A_k) 长度与全局 state 数一致；
- 相邻共享态使用完全相同的 (A_k) 与 offset；
- 全局首末态 (A=0)；
- baseline residual-off 不构造或调用 residual runtime。

如果模型不覆盖新体系，停止；不能拿 production 结果反向调模型后继续沿用同一预注册。

---

## 7. 新体系固定实验协议

### 7.1 配对设计

建议沿用三个 paired repeat：

| repeat | 预定 arm 顺序 |
|---:|---|
| 1 | baseline → candidate |
| 2 | candidate → baseline |
| 3 | baseline → candidate |

每个 repeat 内，两臂必须使用同一 source positions、box 和 velocities。不同 repeat 使用独立初态和独立 seed。

不得看到结果后：

- 换 seed；
- 换 arm 顺序；
- 把某个 repeat 删除；
- 改 warm start；
- 改预算；
- 只保留收益大的窗口。

### 7.2 预算

如果新体系使用相同六窗 Stage-2 结构，可先冻结为与当前生产一致：

| 阶段 | 预算 |
|---|---:|
| smoke warmup | 20,000 steps |
| smoke production | 2,000 steps/window |
| smoke update/query | 200 steps |
| production warmup cap | 500,000 steps |
| production | 250,000 steps/window |
| production update/query | 500 steps |
| checkpoint interval | 100 updates |

如果新体系的 baseline 已知相关时间尺度明显不同，可以在看 A/B 结果前用独立 pilot 冻结另一预算。pilot 数据不能并入正式 production。

### 7.3 固定门槛

若继续使用当前协议，运行前冻结：

| 门槛 | 数值 |
|---|---:|
| sampling-mixture ESS / raw frames | ≥0.05 |
| sampling-mixture absolute ESS | ≥50 |
| 最少去相关样本 | ≥20 |
| endpoint uncertainty | ≤1.0 kJ/mol |
| energy query failure fraction | ≤1% |
| best-effort warmup | 禁止 |
| candidate health | 全 finite |
| scientific retry | 0 |
| failed attempt cost | 计入 ITT |

正式 promotion 同时要求：

1. 三个 paired repeat 全部存在；
2. 至少 2/3 个 repeat 的 ESS/ITT 增益为正；
3. 三次 ESS/ITT 增益中位数 ≥10%；
4. 每个 repeat 的 shared physical target ΔG consistency (z\le2)；
5. 所有正式 coverage、health、冻结和数据账本检查通过。

ESS 定义固定为：

1. 每个 state 的 responsibility 时间序列；
2. 在当前数据段内估计统计非效率；
3. 去相关后计算 Kish ESS；
4. 每窗对各 state ESS 取调和平均；
5. 正式完整链路对所有预注册窗口求和；
6. 除以实际 ITT 总成本。

ITT 包含初始化、校准、冻结验证、production、能量查询、checkpoint、记录和所有失败尝试。

### 7.4 ESS–steps

预先登记 100k、150k、200k、250k production steps/window 前缀：

- 每个前缀重新估计自相关；
- 不沿用全长 (g)；
- 不强制 ESS 单调；
- 不把累计前缀当成四次独立重复；
- 不按总墙钟时间线性伪造前缀耗时；
- primary 仍是完整预算的 ESS/实际 ITT。

---

## 8. 运行顺序与人工分工

### Phase A：CPU/离线接线验证

执行：

1. NumPy score/LSE reference 对 runtime；
2. residual energy/force finite 和单位检查；
3. (A=0) baseline parity；
4. gauge 平移不变量；
5. save → file → load → reconcile；
6. checkpoint/restart；
7. chain interface identity；
8. dry-run 输出路径和参数；
9. 确认 production 阶段不更新 (f_k)。

通过条件：全部 fail-closed contract 通过。此阶段不能证明采样效率。

### Phase B：smoke

目的只限于：

- 入口能运行；
- 预热/冻结状态机可达；
- 账本、marker、成本和失败报告能落盘；
- baseline/candidate Hamiltonian 接线正确。

smoke **不能证明方法有效或无效**，不能用 smoke ESS/ΔG 做 promotion。

### Phase C：production

运行完整预注册矩阵：

```text
3 repeats × 2 arms × all preregistered windows
```

每窗：

1. 独立 cold-start (f_k) calibration；
2. strict frozen validation；
3. 写 production-entry marker；
4. 固定 (phi,A_k,offset,f_k)；
5. production；
6. Context 刷新、最终 state save；
7. 账本、health、cost 和 identity 验证。

用户负责提交和运行 GPU 任务；合并方案和分析代码不得自动启动任务。

### Phase D：正式分析

固定顺序：

1. 数据完整性和冻结链路；
2. 每窗 sampling coverage；
3. full-chain physical target MBAR/TMBAR；
4. shared-target ΔG consistency；
5. candidate health；
6. ESS–steps；
7. ESS/ITT；
8. preregistered reducer；
9. retained-residual target 仅作诊断附录。

不得先看收益，再决定删窗或改 target。

---

## 9. 单窗口失败策略

如果任意预注册窗口 coverage 未过：

1. 保存全部轨迹、账本、日志和成本；
2. 标记 `EXP030_STOP_TMBAR_OR_COVERAGE_INSUFFICIENT` 或新实验对应失败码；
3. 正式完整链路不得删除该窗口后追认 PASS；
4. 不自动重跑到碰巧通过；
5. 不修改 seed、预算、门槛或 warm-start；
6. 其余未运行窗口是否继续采集，必须在新体系 preregistration 中预先规定；
7. 若继续采集，其目的是完成诊断矩阵，正式 full-chain 状态仍 fail-closed；
8. 任何五窗或其他子集结果只能标记为 sensitivity。

为了避免再次出现“一个窗失败导致其余信息全部缺失”，建议新协议预先写：

```text
continue_collecting_independent_windows_after_scientific_failure = true
formal_decision_remains_fail_closed = true
scientific_retry = 0
```

这允许完成数据采集，但不把失败改成通过。

### 9.1 是否可以预先定义窗口子集

可以，但只能在看 production 结果前，依据 Hamiltonian 结构定义，例如：

- residual-active interior windows；
- endpoint/control windows；
- complete physical chain。

每个子集都必须预先写出：

- 窗口索引；
- 科学含义；
- ESS 与成本如何配对；
- 是 primary、secondary 还是 diagnostic；
- 是否允许生成完整 ΔG。

不能把 Atenolol 的 “window_5 表现差”直接转化成新体系的默认排除规则。

---

## 10. 结果解释标准

### 10.1 可以说

若新体系五项正式条件全部通过：

> 在预注册的新体系、完整窗口集合和固定预算下，candidate 相对 baseline 获得可复现的采样态 ESS/实际时间收益，同时 shared physical target ΔG 一致性和所有 coverage/health 门均通过。

### 10.2 不能混说

- sampling-mixture ESS 提高 ≠ ΔG estimator ESS 提高；
- ESS/时间提高 ≠ ΔG 一定更准确；
- retained-residual target 更接近 baseline ≠ 它是真值；
- 单一 repeat 大幅提高 ≠ 所有重复同样大；
- 五窗 sensitivity ≠ 完整 ABFE PASS；
- smoke 能跑 ≠ 方法有效；
- coverage 失败 ≠ 已证明代码 bug；
- source serialization hash 不同 ≠ 物理体系不同。

---

## 11. Rollout 与回滚

### 11.1 第一阶段：合并为可选能力

默认：

```text
outer_lambda_local_residual_ibs = false

正式用户入口只提供这一项 boolean 开关。旧版 `residual_sampling_enabled` 仅可作为
向后兼容的内部 normalized alias；若与新开关同时出现且取值冲突，必须 fail closed。

正式运行时的冻结 R1 payload/weights 从
`resources/outer_lambda_local_residual/manifest.json` 加载，不从
`output/exp*` 读取。当前冻结模型的适用范围必须明确写成 Atenolol：启用前按配体
局部原子序列和内部键图 fingerprint 校验，global topology atom indices 可以不同，
但不接受仅凭 atom count 将模型静默用于任意新配体。
```

只有显式配置、完整 residual identity 和通过预检时才能开启。

### 11.2 第二阶段：新体系验证

完成一个独立体系的预注册 production。若正式条件未全部通过：

- core 修复仍保留；
- residual sampling 保持可选；
- 不默认推广；
- 失败数据作为新体系证据保存。

### 11.3 第三阶段：默认推广

只有新体系正式通过，并且 baseline parity、性能成本和运维稳定性满足主线要求后，另开单独 review 决定是否推荐开启。不要在本次合并中直接改默认值。

### 11.4 回滚

回滚 residual sampling 功能时：

- 关闭 feature/config；
- 不读取 residual model；
- baseline 路径保持可运行；
- 不降级读取不兼容的 candidate checkpoint；
- 保留所有 ledger 和 manifest 供审计；
- 不删除历史实验输出。

---

## 12. 执行清单

### 主线合并

- [ ] 找到真实上游 Git 仓库和主分支
- [ ] 从最新主分支创建集成分支
- [ ] 逐段 diff `ibs_engine.py`
- [ ] 核对 `abfe_core.py` / `abfe_pipeline.py` / residual runtime 接线
- [ ] 提交 sampler 数学一致性
- [ ] 提交 marker 生命周期
- [ ] 提交五本账和 score identity
- [ ] 搬迁通用诊断模块
- [ ] 合入目标回归测试
- [ ] 新增 portability 测试
- [ ] 运行目标测试与完整测试
- [ ] 确认无体系专属硬编码
- [ ] 确认生产不 import 实验分析
- [ ] 默认保持 residual sampling 关闭
- [ ] code review 后合并

### 换体系

- [ ] 创建新 experiment ID
- [ ] 创建独立 DRAFT preregistration
- [ ] 填写体系、checkpoint、窗口和模型 provenance
- [ ] 冻结三个 paired initial conditions
- [ ] 冻结 seed 与 AB/BA 顺序
- [ ] 冻结预算、门槛、失败后是否继续采集
- [ ] 验证 residual model 支持域
- [ ] CPU algebra/wiring/reference 检查
- [ ] smoke，只验证接线和产物
- [ ] 封存 production-authorized preregistration
- [ ] 用户提交完整 GPU production
- [ ] 按固定顺序运行正式分析
- [ ] 写独立最终报告
- [ ] 决定保持可选或进入默认推广 review

---

## 13. 当前资料索引

- [EXP-030 总体状态](EXP-030_FINAL_STATUS_2026-08-27.md)
- [f_k residual training bug](EXP-030_IBS_FK_RESIDUAL_TRAINING_BUG_2026-08-26.md)
- [冻结快照 timing bug](EXP-030_FROZEN_SNAPSHOT_TIMING_BUG_2026-08-26.md)
- [三个 repeat 总结](output/outer_lambda_exp030/three_repeat_final_analysis_lw5i7uud/OVERALL_CONCLUSION.md)
- [排除 window_5 的敏感性分析](output/outer_lambda_exp030/three_repeat_final_analysis_lw5i7uud/EXCLUDE_WINDOW5_SENSITIVITY.md)
- [完整分析 JSON](output/outer_lambda_exp030/three_repeat_final_analysis_lw5i7uud/analysis.json)
- [五窗敏感性 JSON](output/outer_lambda_exp030/three_repeat_final_analysis_lw5i7uud/exclude_window5_sensitivity.json)

---

## 14. 一句话执行结论

**先把 residual-aware (f_k) 训练、生产入口 marker、冻结/账本契约和测试作为默认关闭的通用能力合入真正的上游主线；EXP-030 数据和五窗选择留在实验档案。然后为新体系创建独立预注册，用固定完整窗口、三个 paired repeat、固定预算和 ESS/ITT + physical ΔG consistency 做前瞻性验证，由用户自行提交运行。**

