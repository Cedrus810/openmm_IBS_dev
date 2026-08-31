# PLAN：Pipeline 速度优化 — 下一批候选（2026-08-26）

本文档**只有设计，没有代码改动**。背景：用户在 `SESSION_CHANGELOG_2026-08-25_performance_and_warmup_redesign.md`
记录的这次会话之后，追问"还有哪里能大幅提速"。本文档记录新发现的候选方向，供下次会话（或用户决定后）实施。
`SESSION_CHANGELOG...` 里已经在做的三条（Candidate-first 状态机、Stage2 pilot early-stop Phase B、A/B 爬坡瘦身）
不在本文档重复，只在优先级排序里引用。

节点约束：`nvidia-smi -L` 确认这台机器只有一张 RTX 2080 Ti，因此多 GPU/窗口并行、双腿并行**不是**候选方向，
除非以后迁到多卡节点。

---

## 候选 0（前置，近零成本）：用已有 `loop_timing_s` 埋点拿真机数据

P0-1 已经在生产主循环里埋了 `integration_s`/`guard_s`/`cv_probe_s`/`ledger_io_s`/`weight_update_s` 计时，
写进 `convergence.json` 的 `loop_timing_s` 字段，但**从未在真机上跑过一次去看实际分布**。

- 下一次真机验证（本来就是"待验证清单"第 3/5 项要做的事）顺带把这份数据摸出来，看 500 步控制面里
  `guard_s`/`cv_probe_s` 占比多少。
- 只有当 `cv_probe_s`（K 个 probe 态查询）或 `guard_s` 占比显著，才值得再投入"降低 500 步频率"这类调整；
  否则这条路线优先级应该往后排——不要凭猜测动它。
- 这条不需要单独设计，只需要在下次真机验证时加一步"读一下这个字段打印出来"。

---

## 候选 1：氢质量重分配（HMR）+ 4fs 步长

### 现状核实（已用 grep 确认，非猜测）

- production 积分器：`LangevinMiddleIntegrator(temperature, 1.0/picosecond, 0.002*picosecond)`（2fs），
  出现在 `ibs_engine.py:10358/15843/16572/16874` 等多处。
- `createSystem(..., constraints=app.HBonds, ...)`（`abfe_core.py:9564`），仓库里搜不到任何
  `hydrogenMass=` 参数——HMR 从未启用。
- `abfe_config.json`：`n_steps_per_window=250000`，2fs 下对应 500ps 物理采样时间/窗口。
- `memtodolist.md` 的 A5（膜生产协议）里本来就有一条悬而未决的待办："明确时间步、约束和是否使用 HMR"——
  这条如果做成，会顺带把膜生产协议那条待办也回答掉，值得在两处互相引用。

### 预期收益

HMR（把氢质量提到 ~4 amu，从所连重原子补偿质量，所有键加约束）是 alchemical FE 领域的标准做法
（FEP+/perses/OpenFE 等默认支持），可以把稳定步长从 2fs 提到 4fs。**关键点**：如果同时把
`n_steps_per_window` 减半（125000），物理采样时长不变（还是 500ps/窗口），但 wall-clock 步数减半——
这是目前唯一一个"不改变任何收敛判据、不改变任何决策逻辑，纯积分器+质量重分配"就能拿到的 ~2x。

### 需要调查/设计的点（按顺序）

1. **审计所有 `createSystem(` 调用点**，不只是 `abfe_core.py:9564` 那一处——EM 阶段的"孪生 System
   不挂残差力"（EXP-030 no-residual patch）、MBAR worker pool 里重建的 System（P1-1）、以及任何
   probe/pilot 用的独立 System，都要跟着改，否则会出现"主动力学用 4fs 质量，EM/probe 用旧质量"的
   不一致。列出完整清单是第一步，不能假设只有一处。
2. **是否所有物种都适合 HMR**：受体蛋白、Atenolol 配体、TIP3P 水都是标准支持场景；膜脂质双分子层
   HMR 也是文献里常见做法但約束方案略有不同（有的实现只对非环内氢质量重分配），如果以后接上 C4 膜体系
   需要单独确认脂质力场（若涉及）是否有已知的 HMR 兼容性问题。**当前 Atenolol-rank11 是中性、非膜体系**，
   这条优先级低，先不深挖。
3. **IBS 残差偏置力 / Boresch restraint 是否隐含高频分量**：`LocalManyBodyResidualForce`（EXP-025~030）
   和 Boresch restraint 是否有对高频振动敏感的项（比如力常数很大的谐振子项），如果有，4fs 步长下可能
   比普通经典力场更容易失稳——需要在改之前明确回答，不能只测试"没有残差力时 4fs 稳定"就下结论。
4. **能量守恒/RMS drift 检查**：挑一个已知稳定的窗口（比如某个中间 λ），2fs vs 4fs+HMR 各跑一段
   NVE（或 NVT 但看温度是否漂移），对比总能量/温度 drift，确认 4fs 下积分误差可接受。
5. **按时间而非步数校准的参数**：`max_bias_learning_steps`、`force_check_interval=10`（按 update 数，
   不是按步数，可能不受影响）、`dt_ramp_steps`（A 候选里提到的爬坡步数）等，如果这些默认值是按"步数"
   经验校准的，换 4fs 后同样步数对应的物理时间翻倍，语义会变——需要明确哪些参数要跟着缩放、哪些不用。
6. **数值等价性验证方案**（仿照 P0-1/P1-1 的"两条路径都在当前代码里"思路）：
   - 先做纯数值一致性检查：同一个窗口，2fs 跑 500ps vs 4fs+HMR 跑 500ps，比较两者给出的 ΔF 是否在
     统计误差范围内一致（不要求逐字节相同，因为轨迹本身不同，但物理量要一致）。
   - 挑一个已知边缘窗口（比如 window_0，历史上出过 EM 崩溃/vanishing endpoint 问题的那个）重点测，
     因为 HMR+更长步长通常在"僵硬"区域更容易先出问题。
7. **协议版本**：物理定义本身没变（还是同一个 System 的同一组力，只是氢原子质量和约束range变了），
   `IBS_BIAS_PROTOCOL_VERSION` 大概率不需要 bump；但如果第 5 点发现有语义相关的默认值联动改变，
   需要单独评估是否要 bump `IBS_WARMUP_UPDATE_PROTOCOL_VERSION`。

### 建议顺序

先做第 1-3 步的静态审计（不需要 GPU，纯读代码），得出"HMR 在这个仓库能不能干净地接进去"的结论，
再决定是否值得排真机验证时间。

---

## 候选 2：production 阶段自适应早停

### 现状核实

- `remaining_production_steps` 的算法（`ibs_engine.py:12735` 附近）目标恒定是
  `effective_n_steps_per_window`（=250000），resume 只是补齐 `cumulative_production_steps` 和目标
  之间的差额，**没有"精度已经够了就提前停"的机制**——不管窗口本身收敛得多快，都要跑满 250000 步。
- warmup/learning 阶段已经有对称的基础设施可以复用：online split-half（P1-19 那批修复之后应该是可信的）、
  local-MBAR loose gate（<10 kJ/mol 的相邻 ΔF 门，v27→v29 的设计）。

### 关键风险（在动手之前必须先回答，不能只做静态论证）

Stage2 pilot 那条历史（30000 步的教训：短 pilot 在 λ≈1 端点系统性低估稀有/发作性事件主导的方差，
导致 `IBSWarmupConvergenceError`）**同样适用于 production**——基于单条短轨迹的收敛判据（split-half、
峰度等）本质上无法可靠区分"真收敛"和"还没等到稀有事件所以看起来稳定"。250000 这个数字本身有没有
类似 30000 那样的历史踩坑背景，需要先做一次尽调（搜代码注释、搜 `SESSION_CHANGELOG`/`memtodolist_archive.md`
里有没有提过"缩短 production 步数导致 XXX 回归"），再决定要不要碰。

### 建议路线（仿照 Stage2 Phase A/Phase B 的两阶段模式）

1. **尽调**：先确认 250000 这个数字的历史来源（是拍脑袋定的还是踩过坑）。如果是后者，本候选的
   优先级和风险都要重新评估。
2. **Phase A（零行为改变，纯插桩）**：production 循环里每跑够一定步数，用已有的 split-half/local-MBAR
   诊断多算一次"如果现在停下,精度够不够"，只记录不生效——完全类似 Stage2 pilot 的 shadow checkpoint 设计。
3. **Phase B（需要真机数据）**：用 Phase A 积累的 shadow 数据回答"有没有窗口曾经被早停判据误判为够了,
   但其实继续跑之后 ΔF 还在明显漂移"——这条必须是**零次**才能真正上线早停,跟 pilot early-stop 的验收
   标准同一个逻辑。
4. 早停判据本身应该同时满足"精度阈值"+"最小步数下限"两个条件（避免早期噪声窗口误判），不能只看阈值——
   这也是从 Candidate-first 状态机和 pilot 早停两处设计里提炼出的共同教训。

### 与候选 1 的关系

如果 HMR 先做成了（production 步数减半），这条早停设计的"最坏情况成本"会跟着减半，两条互不冲突，
可以并行设计，但真机验证阶段建议**不要同时开跑**（避免两条改动的真机结果混在一起,分不清收益来自哪条）。

---

## 优先级建议（仅供参考，最终顺序由用户决定）

1. 候选 0（几乎零成本，下次真机验证顺手做）
2. 候选 1 第 1-3 步静态审计（不需要 GPU）
3. 候选 2 第 1 步尽调（不需要 GPU）
4. 视审计结果决定候选 1/2 谁先进入真机验证阶段——建议避开与 Candidate-first 状态机真机冒烟、
   Stage2 pilot Phase B 同时段跑，防止真机验证矩阵互相污染。

## 开放问题（需要用户决定）

- HMR 要不要走仓库现有的 EXP-XXX 编号track（完整 PASS/FAIL gate、独立文档），还是作为常规性能优化
  直接走本文档的验证清单收尾？目前仓库里的 EXP 系列都是围绕 IBS 残差偏置插件的实验，HMR 性质不同
  （经典 MD 积分设置），个人倾向不占用 EXP 编号，但这个由用户定。
- 候选 2 的"250000 步历史尽调"如果查出来是拍脑袋定的（没有踩坑背景），是否直接跳过尽调进入 Phase A？
