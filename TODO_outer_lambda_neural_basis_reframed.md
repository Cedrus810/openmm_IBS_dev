# TODO：Outer-λ Neural Basis 重构后的研究与实现清单（复核修订版）

> 本清单根据以下三份文档整理：
>
> - `PLAN_outer_lambda_neural_basis.md`
> - `IMPLEMENTATION_PLAN_outer_lambda_neural_basis.md`
> - `EXPERIMENT_LOG_outer_lambda_neural_basis.md`
>
> 当前版本的核心重构是：**研究对象不是“一个可以低频更新的神经笛卡尔力”，而是离线 learned representation / residual 中可能包含的、与相邻 alchemical state `1↔0` 混合瓶颈有关的 candidate signal；研究任务是先验证它是否具有时间持续性和跨轨迹预测力，再把通过验证的信息以最低 online cost 转译为严格、可控、可验证的在线变量或路径项。**
>
> 符号约定：本文的 `state 1↔0` 指相邻 alchemical state index，不等同于物理端点 `λ=1` 与 `λ=0`。具体 state-to-λ 映射必须从冻结协议读取，禁止凭编号推断。

---

## 0. 当前项目定位

### 0.1 主问题

离线分析已经给出候选正向证据：D1 direct-gap `LocalResidualStudent` 的 held-out gap-variance 平均改善为 `13.93%`，按 fold-level 门为 `2/3`，且 `direct_gap_all_folds_improved=false`；这不是全折一致结果。随后 DEC-048 fused 设计的轻量 student 在三次 paired-reseed exploratory 两臂 pilot 中提高了 `mixture_ess_proxy`，但这不是三组独立平衡的 production repeats；逐 MD step 的 TorchForce 部署把计算成本提高到约 `1.81–1.89×`，因此 ESS/GPU-hour 三次均下降。当前证据尚未证明该 signal 是 slow variable，也尚未证明它预测真实的 `state 1↔0` crossing。

接下来不再把主要问题写成：

```text
如何让 full Cartesian neural force 变成 slow force？
```

而应写成：

```text
离线 teacher / residual candidate signal
    → 验证是否存在与 state 1↔0 overlap/crossing 相关的时间持续信息
    → 压缩成低维 slow information 或 cheap CV
    → 每个 MD step 只施加廉价、守恒、可审计的控制项
    → 在 IBS/TMBAR 中作为完整 target Hamiltonian 验证
```

### 0.2 必须保留的概念区分

- `q(t)` 慢，不代表 `-dB/dq · ∇_R q` 这个 Cartesian force 可以按任意 outer interval 冻结。
- 轨迹保存的 `R(t_i)` 是连续动力学过程的取样，不是 IID 的 500 个离散实验。
- learned path term 如果改变了目标 Hamiltonian，就必须进入 target energy、跨态 reduced potential 和 TMBAR/MBAR；它不能混入待消除的 IBS `bias_history`。
- MTS 只回答“冻结某种 Cartesian force 是否仍能保持分布”，不回答“learned slow information 是否有用”。
- EXP-012 的 `D0–D4` 证明的是模型、可导力、部署和短 NVT 资格，不等于 production ABFE 已通过。

### 0.3 当前状态摘要

| 项目 | 当前状态 | 后续处理 |
|---|---|---|
| `EXP-012` latent / student 信号 | 已在 held-out run 上观察到 gap variance 改善；三次 paired-reseed pilot 的 exploratory mixture ESS proxy 均改善 | 保留为 candidate learned signal；不能称为三条独立 production repeat，也不能提前称为 slow information |
| 逐步在线 TorchForce | `DEC-050` 已关闭；成本使 ESS/GPU-hour 变差 | 当前不恢复；本轮不继续搜索新的 online/MTS route |
| `EXP-013-A` exact residual split | 通过数值等价和成本检查 | 仅作为实现/成本事实保留 |
| `EXP-013-B` residual-force MTS | 方案③失败；方案① Qualification gate 未通过；方案② N=1 ESS signal 失败（DEC-059） | `EXP-013_NO_PROMOTION`；不运行方案② MTS、不重调 `c1`；EXP-014 screen 亦未通过（DEC-060） |
| EXP-013/014 后续在线分支 | 三种在线/MTS 方案及 native-compression screen 均未晋级 | 不重调 `c1`、不重选 checkpoint、不继续搜索 MTS 间隔、不直接重开 WP-5；当前分支冻结 |
| `EXP-010` cheap torsion CV | 失败，但有截断残基和高维瞬时能量拟合问题 | 不解释为“torsion 信息无效” |
| `EXP-011` PMF | overlap 不足，已冻结 | 不再补采、不拟合 PMF |
| `EXP-012/WP-5A pilot` | `NO-GO`；`DEC-050` 已关闭当前 real-time TorchForce 路线 | 新 cheap route 必须作为独立新实验重新资格审查，不能恢复旧 candidate |
| `EXP-016` temporal audit | 已完成离线审计；3 条连续轨迹、1500 帧、`Δt_save=1 ps`；物理 crossing 历史不可用；energy-weighted surrogate 仅作探索性结果 | 不晋级 learned slow information；先做独立 physical/overlap event 或另行预注册 cheap offline route |
| `WP-5B`、`WP-6–WP-8` | 阻塞 | 不提前启动跨体系、多基势、DEXP 或完整 production |

---

## 1. P0：先统一状态、协议和研究问题

### 1.1 修正实验日志的状态摘要

- [x] 更新 `EXPERIMENT_LOG_outer_lambda_neural_basis.md` 的总体状态、实验索引、`§11A` EXP-012 正文、WP-5A 结果和实验计数，使其与后部 `DEC-039`–`DEC-057` 一致，并登记 DEC-056 分支裁决。
- [ ] 明确写入：`EXP-012` 的 D0–D4 总体已关闭，但 production ABFE 仍为 `NOT_READY`。
- [ ] 核对实施计划中仍列出的 `(d0-2)`–`(d0-5)`：逐项标为历史子门已关闭、由后续决策 supersede，或明确 reopened；特别是 `DEC-041` baseline v2 已接受，不能继续把 d0-5 写成未测阻塞项。
- [ ] 明确写入：`DEC-050` 关闭的是“逐 MD step 运行当前 student TorchForce 的部署路线”，不是否定 latent / residual 的科学信号。
- [x] 明确写入：`EXP-013-A = PASSED`；`EXP-013-B` 方案③已执行、绝对健康门通过但预注册相对门失败；`DEC-056` 已裁决转方案①。不能再保留“EXP-013 未执行”的旧摘要。
- [ ] 将 `Arm A/B/D` 统一标记为 `not_pursued`，不能写成 `FAILED`；当前只能声称 student 优于 `B=0` 基线，不能声称优于未执行的 A/B/D。
- [x] 保留所有历史数字和失败原因，不覆盖旧结果；当前裁决段明确 supersede 早期状态索引。

### 1.2 冻结当前基线和禁止事项

- [ ] 冻结当前困难窗口：complex vanishing / Stage 2 / window 0 / states `[0,5)`。
- [ ] 冻结当前参考路径：`mode=ibs`、`decoupling=dual_lambda`、`potential=softcore`、`dexp_params=null`。
- [ ] 冻结当前诊断变量仅作为分析工具：primary torsion `[4591,4592,4593,4585]`、secondary torsion `[4593,4585,4594,4595]`、`VAL251 chi1`、ligand hydration coordination。
- [ ] 记录当前 λ schedule 和最小 baseline ESS，后续所有实验必须使用同一版本基线或显式登记变更。
- [ ] 禁止因为新结果不好而事后切换另外 8 个 checkpoint、修改 `c1`、扩展已冻结的 ridge 网格，或重新挑选最优 seed。
- [ ] 禁止恢复当前逐步 TorchForce 路线作为 production 方案。
- [ ] 禁止把 `student_torchscript_d4.pt` 直接用于多态 IBS production：该文件把 `a_k=0.5` 烤入输出；若重新接线，必须重新导出输出未缩放、`a_k=1.0` 的版本。
- [ ] 冻结唯一缩放语义：`TorchScript a_k_baked_in=1.0`、`c1=0.5`、`A_k=sin²(πλ_k)·c1`；禁止在模型输出和 controller 中重复缩放。
- [ ] 冻结历史 checkpoint `hard_window0_run1__direct_gap__seed0.pt` 的 SHA-256：`61abcd1f0d0ff809914003de522f05db66f9dc4b341391bfa0b7f1cb99e6f2e3`；任何新导出 TorchScript 必须另记完整路径和自身 SHA-256。

### 1.3 明确决策门和停止条件

- [ ] 为每条新路线在开始前写出唯一的主指标、统计方法、独立重复数、停止条件和回滚路径。
- [ ] 所有“科学信号”与“计算性能”分开登记：
  - 科学信号：gap variance、overlap、ESS、crossing、autocorrelation、迟滞；
  - 工程性能：ms/step、GPU-hour、显存、Context overhead；
  - 物理正确性：温度、能量漂移、自由能一致性、target/bias 账本。
- [ ] 预先规定：只有在 `ESS/GPU-hour` 改善且物理/统计门都通过时，才允许 promotion。
- [ ] 如果一个方案只提高表面交换率或短 pilot 的 proxy，而不能提高真正的有效样本效率，则不得晋级。

### 1.4 解决实验编号冲突

源实施计划已将 `EXP-014` 预留给 native OpenMM compression contingency，将 `EXP-015` 预留给 post-hoc ranking/reweighting；因此 temporal audit 不能再次使用 `EXP-014`。

- [ ] 保留 `EXP-014`：native `CustomNonbondedForce` typed-pair + radial spline/RBF + cutoff 压缩基势。
- [ ] 保留 `EXP-015`：post-hoc/offline ranking 或 reweighting 候选；启动前仍需单独预注册。
- [x] 将 transition-segment attribution / temporal distillation 登记为 `EXP-016`。
- [x] 在实验日志登记编号映射；禁止出现两个内容不同的 `EXP-014_*` 目录或报告。
- [ ] 实验编号只表示 registry identity，不强制按数字顺序执行。当前主线可以先执行 `EXP-016`，再由结果决定是否启动 `EXP-014`。

---

## 2. P0：EXP-016——transition-segment attribution / temporal distillation

这是当前最重要的新主线。先不新增昂贵 MD；优先审计已经存在的 DEC-048/EXP-012 轨迹和 latent/student cache，验证“candidate signal 是否具有可分辨的时间持续性、是否与未来 overlap/crossing 有关、现有保存频率是否足以回答这些问题”。只有通过本阶段后，才能称其为 learned slow information。

### 2.0 数据可行性硬门

- [x] 已确认现有 IBS 数据没有可追踪的 alchemical state/replica history 或物理 basin crossing；五态 target-energy ledger 不是 physical crossing 轨迹。
- [x] 已区分并分别命名 physical state/basin crossing、IBS dominant-component switch、energy-weighted surrogate event；后者只作为审计标签。
- [x] 已审计标签依赖：surrogate 标签来自 MM ledger 的 energy-weighted argmin；direct-gap student 与 adjacent-gap observables 又是 target-derived，不能充当独立预测证据。
- [x] 因缺少 physical history，本阶段只完成 autocorrelation/attribution 与 surrogate exploratory audit，没有 physical crossing claim。

### 2.1 数据清单和 provenance

- [x] 已盘点三条 run 的 trajectory、实际保存间隔、模拟时间、seed、window/protocol 和输出目录；`EXP-016_data_manifest.json` 记录了可用字段及缺失的 physical state history。
- [x] 未假设 `64 fs`；从 sample metadata 得到 `Δt_save=1 ps`，所有 look-ahead horizon 均为其整数倍。
- [x] 已盘点每条 run 的 `500` 帧 latent cache、student 输出、MM ledger gap/reduced-energy observables 和可用构象诊断量；hydration 仅有 run summary，未伪造逐帧序列。
- [x] manifest 为每条输入登记来源、哈希、run/frame 对齐、protocol/preregistration 和 checkpoint provenance；manifest 变化使 audit 结果失效。
- [x] 三条 run 按独立连续 trajectory 做 LORO；没有随机 frame split。
- [x] 已生成并封存 `output/outer_lambda_exp016_loro/EXP-016_data_manifest.json`。

### 2.2 定义 transition episode，而不是把 frame 当作 IID 样本

- [x] 已冻结本次 audit 的 operational surrogate：`argmin(target_interaction_kj_mol - f_k)` 的相邻 label `0↔1` 改变；明确标为 energy-weighted surrogate，不解释为物理 event。
- [ ] 区分至少四类片段：
  1. λ₁-compatible basin；
  2. crossing 前 pre-transition segment；
  3. overlap / transition segment；
  4. λ₀-compatible basin。
- [ ] 为每次 crossing 生成 episode ID、起止时间、方向、持续时间和是否完整观测。
- [ ] 为每个 crossing 生成 matched non-crossing control segment，匹配 run、时间长度、初始 basin 和必要的构象统计量。
- [ ] 对边界处无法判断未来事件的片段标记为 censored，不要强行作为负样本。
- [x] look-ahead horizon 已冻结为 `1/5/10/25` frames，即 `1/5/10/25 ps`；未分析小于保存间隔的 horizon。

### 2.3 轨迹相关性、ESS 和 block bootstrap

- [x] 已对可用的 teacher/student/gap/torsion/chi1 observables 计算时间自相关；hydration/crossing 的缺失序列明确排除。
- [x] 已报告 IAT、`N_raw` 和有效样本数；例如 student direct-gap 的逐 run `N_eff` 约为 `15.5/258.3/29.0`。
- [x] 使用连续 run 内 block bootstrap；没有把 frame-wise IID z-score 作为主结论。
- [x] 已冻结 circular contiguous block bootstrap：`128 frames = 128 ps`，`2000` replicates；三条 run 仍只算三条独立轨迹。
- [ ] 重叠 look-ahead 标签必须去重；crossing episode 与 matched control 在 bootstrap 中绑定抽样。
- [x] 已逐 run 报告并做 leave-one-run-out；run 内 bootstrap 没有被解释为新增 independent run。
- [x] LORO 按独立 run 执行；held-out student checkpoint 分别匹配 run1/run2/run3，训练只用另外两条 run。
- [x] 主要 attribution 指标输出 block-bootstrap confidence interval；prediction 结果保留为 surrogate exploratory point estimates。
- [ ] 对多时间窗、多 feature、多 λ 邻居比较采用预先登记的多重比较策略，或明确把结果标记为 exploratory。
- [ ] 检查 `mixture_ess_proxy`、pymbar/BAR mutual overlap 和真正 importance ESS 的定义差异；proxy 统一标记为 exploratory，不得用于 production promotion。
- [x] 输出每条 run 的 `N_raw`、block protocol、IAT/`N_eff` 和 `N_independent=1 run` 语义。

### 2.4 测量 slow information 的时间尺度

对每个候选信号分别计算：

- [x] autocorrelation decay；
- [ ] pre-crossing 到 crossing 的条件均值/方差；
- [x] future-event ROC-AUC、AUPRC、Brier score 与 class balance 已计算；结果仅针对 energy-weighted surrogate，未升级为 physical event prediction。
- [ ] control segment 上的 false-positive rate；
- [x] signal 在 `1/5/10/25 ps` horizon 的 surrogate prediction 已计算。
- [x] 当前固定 block length 下的 block-bootstrap attribution 稳定性已报告；未把未预注册的 block-length sweep 当作主结果。
- [x] 已逐 run/LORO 报告；由于没有物理方向/λ-state history，不声称方向或不同 λ 邻居一致性。

候选信号至少包括：

```text
frozen-MACE latent projection
LocalResidualStudent scalar output
student force norm / force projection
gap correction / Δu correction
primary torsion
hydration coordination
ligand-environment contact topology
```

- [ ] 在预注册中为 `τ_information` 选择唯一 operational definition、估计器、截断规则和不确定度；其他定义只作为 sensitivity analysis，不得择优替换。
- [ ] 不把“q(t) 慢”直接等价为“Cartesian force 可以冻结”；同时记录 `∇_R q` 或 force projection 的高频变化。
- [x] 已明确 force(signal) 不可测：现有 cache 没有 teacher coordinate gradient/force series，因此没有把 signal 慢性外推为可冻结 Cartesian force。

### 2.5 交付物和通过标准

- [x] 已生成 `output/outer_lambda_exp016_loro/EXP-016_temporal_audit.json`、machine-readable manifest 和一页结论摘要；本次不生成无法由现有 cache 支撑的 force/physical-crossing 图表。
- [x] 结论已回答可由现有数据支持的部分：physical event 不可用；surrogate prediction 仅 exploratory，按连续 run/LORO 报告；candidate IAT/`N_eff` 已报告。
  - [x] physical event 结论：无可独立定义的 physical `state 1↔0` crossing；surrogate 结果不能升级为 crossing claim。
  - [x] 跨 run 结论：只对可用 surrogate 做三折 LORO；不把三条 run 内 bootstrap 当作更多独立重复。
  - [x] 相关时间：报告 candidate IAT；physical `τ_information` 不定义。
  - [x] 输入形式：本次只审计单帧缓存；没有证据支持在线历史窗口或历史依赖 Hamiltonian。
  - [x] cheap online variable：没有 candidate 通过 physical slow-information 门，暂不压缩/接入 production。
  - [x] 低频更新正确性：force(signal) 未缓存，不能作安全结论，因此不启动低频 online/MTS qualification。
- [ ] 通过条件必须在运行前量化：最小独立 event 数、覆盖的独立 run 数、相对 null/cheap baseline 的最小效应、block-bootstrap CI、校准/假阳性上限和多重比较规则。只有满足预注册阈值的 candidate signal 才能称为 slow information。
- [x] 已按该规则执行：报告 surrogate exploratory prediction，但明确 physical event label unavailable、没有 crossing-prediction claim。
- [x] 未有候选通过“physical learned slow information”门；停止开发 MTS/online bias promotion，保留 teacher attribution/data-quality audit 作为下一允许入口。

---

## 3. P0：把 learned residual 从“瞬时能量”改写成“可验证的慢信息目标”

### 3.1 定义训练目标

- [ ] 明确当前 student 的输出到底要表示：瞬时 gap correction、transition score、future crossing probability、local overlap score，还是某种低维 reaction coordinate。
- [ ] 为每种目标建立单独的 target schema；不能把不同物理含义的输出都称为“slow variable”。
- [ ] 保留 direct-gap student 作为必要对照；teacher 不是无条件 ground truth。
- [ ] 设计至少三类目标的离线比较：
  1. instant frame-wise regression；
  2. block/episode-level target；
  3. future-event / crossing prediction target。
- [ ] 所有训练/验证切分必须按 run 或连续时间块进行；禁止随机 frame split 造成 trajectory leakage。
- [ ] 只使用训练时刻可获得的信息；若使用历史窗口，明确窗口长度和 causal 方向。

### 3.2 低维压缩和可解释性

- [ ] 对 latent、local feature、window correction matrix 和 episode-level feature 做 PCA/SVD/低秩分解。
- [ ] 登记前几个奇异值的解释方差；若 `2–4` 个分量已经覆盖目标变化，再考虑多基势。
- [ ] 若需要十几个以上分量，暂不盲目增加 `M`；评估 continuous λ-conditioned GNN 或放弃低秩假设。
- [ ] 离线阶段可比较 `q_φ(R)` 与 causal `q_φ(history)`，并报告维度、输入 atom set、cutoff、PBC、时间窗口和跨 run 泛化。
- [ ] 第一版静态、守恒 production Hamiltonian 只允许瞬时坐标函数 `q_φ(R)`；`q_φ(history)` 只能用于离线发现/标签。若要在线使用历史依赖，必须另立扩展状态或非平衡重加权推导与资格实验。
- [ ] 检查 q 的坐标梯度是否在近接触、cutoff 边界和动态水环境下有限。
- [ ] 检查 q 是否只识别某一条 run 的构象标签；不得把 run identity 当作 slow variable。
- [ ] 用 attribution / ablation 区分 hydration、torsion、contact topology 和非物理边界伪影。

### 3.3 重新审计 EXP-010 的失败原因

- [ ] 把 EXP-010 的教师环境截断问题写成数据质量风险，而不是 torsion bias 的否定证据。
- [ ] 若重做局部 CV，仅允许使用完整残基/完整相互作用上下文，不能复用含 `26` 个不完整残基的截断环境。
- [ ] 对比固定 `0.5 nm` 单帧球、逐帧两跳闭包和动态 ligand-environment funnel；记录每种支持域的边界伪影。
- [ ] 采用 trajectory-block target，而不是用低维 torsion 直接拟合高维瞬时 interaction energy。
- [ ] 只有在新数据协议、目标和切分全部预注册后，才可重启类似实验。

---

## 4. P1：设计两个主路线与一个探索性 hybrid

### 4.1 路线 A：offline learned slow coordinate → cheap analytic bias

这是优先级最高的终局候选，因为它直接利用“slow information”，而不是冻结 full Cartesian neural force。

- [ ] 训练一个低维、可导、具有明确支持域的 `q_φ`，优先从 `1D` 开始，必要时再到 `2D`。
- [ ] 训练或选择一个廉价解析势：`V_bias(q)` 可使用 spline、tabulated function、低阶多项式或小型 MLP。
- [ ] 如果最终对象可压缩为 typed-pair + radial spline/RBF + cutoff，则在保留编号的 `EXP-014` 中用原生 OpenMM `CustomNonbondedForce` 实现和资格验证。
- [ ] 让外层 λ 包络控制其路径作用：

  ```text
  B_λ(R) = w(λ) · V_bias(q_φ(R))
  w(λ) = sin²(πλ)
  ```

- [ ] 验证 λ=0、1 时能量、力和自由能与基础路径一致。
- [ ] 每个 MD step 只计算 cheap q 和解析偏置；不在 step 内运行 frozen-MACE teacher。
- [ ] 用有限差分验证 `-∇_R [w(λ)V_bias(q_φ(R))]`。
- [ ] 对 cutoff、PBC、动态水、近接触和非参与原子零力建立硬门。
- [ ] 明确该项是 target Hamiltonian path term；不能从 IBS `bias_history` 中整体消去。

### 4.2 路线 B：offline teacher → cheap differentiable student

只有当路线 A 无法保留足够 crossing information 时，才继续扩展 student。

- [ ] 保留当前 `LocalResidualStudent` 架构作为 baseline，不改变冻结 checkpoint 的训练结论。
- [ ] 新版本优先学习 temporal/episode target，而不是继续优化瞬时 energy RMSE。
- [ ] 维持 typed embedding、平滑 radial/contact 特征、`1–2` 个 interaction block、ligand-only pooling 和有界标量能量设计。
- [ ] 固定部署预算：原子数 `≤256/320`、边数 `1536/2048`、单原子邻居 `64/80`；任何预算变化单独登记。
- [ ] 重新导出未缩放 `a_k=1.0` TorchScript，并记录 checkpoint、TorchScript、模型 spec 和 SHA-256。
- [ ] 在 CPU/reference/CUDA 三种后端上验证能量、坐标梯度、端点和性能；无 CUDA 节点时相关测试必须明确 skip。
- [ ] 采用 `DEC-041` 已接受的 `no_student_window0_baseline_v2`：median/P95=`1.3959/1.3968 ms/step`；只有运行条件或协议变化时才重新测量。
- [ ] 记录 `DEC-050` matched-path 性能：baseline/network-only/full=`1.4765/2.7042/3.5678 ms/step`；即使去掉建图，network-only 仍约为 baseline 的 `1.83×`。
- [ ] 旧的约 `95%` 数字只标记为特定 D3 测试中的 step overhead，不称为通用 Context overhead，也不与模型科学信号混为一谈。
- [ ] 只有新 student 让 `ESS/GPU-hour` 改善，才进入 WP-5A；单步 ESS 改善不足以 promotion。

### 4.3 路线 C：blockwise / trajectory-level hybrid

这是探索性路线，不等同于传统 force-MTS。

- [ ] 设计 block protocol：普通 MM/IBS 连续积分；每个 block 结束或低频时刻评估 slow state；下一 block 使用廉价、冻结的 analytic control。
- [ ] block 长度候选必须是实际 `Δt_save` 的整数倍，并由 `EXP-016` 预注册的 `τ_information` 结果支持；不可沿用无法由保存频率解析的 `0.5 ps` 候选。
- [ ] 明确 adaptation 是改变 Hamiltonian、改变 proposal，还是只更新外部 bookkeeping；三者的统计处理不同。
- [ ] 如果在线更新 Hamiltonian，必须写出 detailed balance、time-dependence 和 reweighting protocol；不能把自适应过程当作静态 IBS Hamiltonian。
- [ ] 优先实现“离线先冻结 rule、在线只查询 rule”的版本，避免第一版引入动态自适应。
- [ ] 若无法给出可审计的权重/自由能记账，路线 C 只能作为方法学讨论，不得进入 production。

---

## 5. P1：实现层 TODO——从研发隔离到可验证接口

### 5.1 模型和协议对象

- [ ] 完成 `NeuralBasisModelSpec` / task manifest 的最小实现与接线，必须包含：模型名、类型、绝对路径、模型 SHA-256、training-data version/SHA-256、element set、cutoff、精度、backend/device、atom-selection SHA-256、能量基准、单位、PBC 支持域和安全阈值。
- [ ] 所有 production 模型 hash 由程序重新计算，不能只信 YAML 里的 hash；`verify_files=false` 仅允许测试或 offline dry-run，production 必须 fail closed。
- [ ] 增加 temporal schema：temporal target、history window、time step、future horizon、episode protocol、block protocol 和训练/验证 run 列表；这些字段必须进入 controller/protocol payload。
- [ ] 增加协议版本字段，至少区分：

  ```text
  NEURAL_PATH_PROTOCOL_VERSION
  NEURAL_BASIS_MODEL_PROTOCOL_VERSION
  NEURAL_PATH_ACCOUNTING_VERSION
  TEMPORAL_SLOW_INFO_PROTOCOL_VERSION
  ```

- [ ] 以下任一变化必须使缓存失效：模型内容/顺序/能量基准、atom selection、cutoff、PBC、λ schedule、包络、系数、target/bias ledger、精度、Force backend、temporal target 或 block protocol。
- [ ] production resume gate 必须 fail closed 比较：`neural_path_enabled`、`neural_path_protocol_sha256`、`neural_basis_spec_sha256`、`model_sha256`、`atom_selection_sha256`、`accounting_version`、`temporal_slow_info_protocol_version` 和 `block_protocol_sha256`；启用 neural 而 manifest 缺字段时拒绝 resume。

### 5.2 `OuterLambdaController`

- [ ] 第一版继续固定 `M=1`、`envelope=sin2`、`coefficient_model=constant`。
- [ ] 计算并序列化 `w(λ)`、`c_m(λ)`、`A_km=w(λ_k)c_m(λ_k)`。
- [ ] 检查端点归零、有限性、有界性、平滑性和 `2K+M <= 32`。
- [ ] 检查相邻窗口公共 λ 的系数逐位一致。
- [ ] 记录所有系数、centered constants、模型 hash 和运行配置到结果 manifest。

### 5.3 target/bias/base 账本

每帧严格按以下顺序实现和测试：

```text
1. 读取 λ 无关 base energy
2. 读取 IBS/WCA sampling bias
3. 计算各态 interaction energy
4. 计算共享 basis / cheap CV
5. 由 A_km 生成 neural path energy
6. 形成 target energies
7. 执行同步 finite gate
8. 分开追加 target、bias、base history
```

- [ ] 保证：`target = original interaction + LRC + neural path`。
- [ ] 保证：`bias = IBS log-sum-exp bias + sampling-only WCA`。
- [ ] 任何非有限值触发现有 hard gate；禁止用零替代。
- [ ] 明确 IBS log-sum-exp 的非线性，禁止把 neural force 与 classical bias 做错误的线性拆分。
- [ ] 验证 neural term 进入 cross-state energy、TMBAR/MBAR reduced potential 和 resume cache。
- [ ] 将 `bias_cv`、`neural_path_energy_history`、`basis_energy_history` 与 target/bias/base 作为同一 generation 原子提交；若选择可重建方案，必须预先冻结重建算法、输入和重建 hash。
- [ ] 验证两个真实物理端点、cycle closure、complex/solvent 两条腿和公共 λ 状态一致；standalone wiring smoke 不能替代另一物理端点测试。

### 5.4 代码接入顺序

保持研发隔离，按小提交逐步接入，不把 WP-1–WP-5 合成一次大改。

- [ ] 先完成 `outer_lambda_neural_basis.py`、`local_residual/`、单元测试和离线分析脚本。
- [ ] 再接入 `abfe_core._build_mace_potential` 和 `OrbVacuumContext` 的只读/构建接口。
- [ ] 再验证 `ibs_engine.build_ibs_dual_system`、`IBSBiasForce`、`IBSSampler._build_probe_context`。
- [ ] 再验证 `evaluate_interaction_energies`、`collect_energies`、`IBSWindowManagerDualLambda._build_window_system`。
- [ ] 最后才接入 `abfe_pipeline.ABFEPipeline._run_dual_lambda_stage` 和 CLI `--neural-path-config FILE`。
- [ ] 默认 `enabled=false`；禁用时旧行为、旧缓存语义、旧能量和旧输出必须完全不变。

### 5.5 测试清单

- [ ] CPU 单元测试覆盖 controller、envelope、coefficient、endpoint、ledger、model hash、cache invalidation、disabled compatibility。
- [ ] 新增 `tests/test_exp016_temporal_schema.py` 与 `tests/test_exp016_temporal_audit.py`，覆盖 LORO、连续 block、禁止随机 frame split、look-ahead 去重和 censored episode。
- [ ] 新增/扩展 cache、scale、identity、resume、rollback 测试，覆盖 `a_k=1.0` 与 `c1/A_k` 只缩放一次、manifest 缺 hash 时拒绝 resume、crash 后 generation 不混代，以及 `enabled=false` 的逐位兼容。
- [ ] mock Force 覆盖能量/力和有限门。
- [ ] 现有独立集合保持至少 `80 passed`；任何 CUDA skip 必须说明是环境缺失，不是测试通过。
- [ ] TorchForce/OpenMM Reference/CUDA 的能量和力在容差内一致。
- [ ] checkpoint resume 后能量、state、box vectors 和 manifest 一致。
- [ ] 主 Context 与 probe Context 共存时不发生状态污染。
- [ ] 小体系 `10k–100k` steps 无 NaN、无异常大力、无异常结构。
- [ ] 关闭 neural path 后运行旧基线回归，确认没有隐式改变。

---

## 6. P1：EXP-013 / MTS 旁路实验清单

### 6.1 登记 DEC-055 主结果并执行 DEC-056 分支裁决

- [x] 主结果固定登记：方案③在预注册 `z_threshold=3.0` 下，N=8/16/32 的相对系统性偏移门均未通过；不得事后改写为 pass。
- [ ] 明确现有 z-score 是 raw-snapshot screening statistic，不是经过 serial-correlation correction 的独立样本显著性。
- [x] 绝对温度偏移小于约 `1.3 K` 仅描述物理量级，不构成对预注册失败门的豁免。
- [ ] 若做补充敏感性分析，必须另行冻结 paired block difference、block length、有效样本估计、absolute tolerance、multiple-comparison rule 和最终判据；补充分析不得替换或覆盖主裁决。
- [x] 接受主门失败并转方案①；不并列测试方案①/②，不进入方案③的 013-C，也不事后放宽阈值。

### 6.2 若继续，严格遵循固定顺序

顺序固定为：

```text
③ exact residual split（已失败，DEC-055/056）
    → ① whole fused slow group（当前）
    → ② independent additive student（仅在① Qualification gate 未通过后）
    → EXP-014 native compression（当前冻结 screen 已未通过，DEC-060）
```

- [ ] 方案③仅作为 MTS 数值/成本分支，不宣称它验证了 slow variable。
- [x] 方案①已完成脚本的 `smoke`（固定 `16/32` ticks，仅 backend/健康诊断）和
  `qualification`（固定 `400/2000` ticks，即 `6.4/32 ps`；每 `50` ticks 做 block-mean
  SEM）；Qualification 的 N=2/4/8 系统偏移 gate 未通过，故禁止触碰 `N=16`。当前 `inner_dt=2 fs`
  下 N=16/32 的 outer step 是 `32/64 fs`，不得直接上三重复。
- [x] 方案②只有在方案①结果明确后才考虑；不跳过顺序。
- [x] 方案① Qualification gate 未通过后，方案②按新 Hamiltonian 口径完成 N=1 ESS 信号检查。
- [x] 方案①未通过当前单种子 Qualification gate，故 `N16_NOT_AUTHORIZED`；科学结论为
  `PHYSICAL_SYSTEMATIC_BIAS_INCONCLUSIVE`。转方案② N=1 ESS 信号检查；方案③和方案①
  均不得进入 013-C。
- [x] 方案② N=1 ESS signal 未通过：`mixture_ess_proxy` 从 `47.827779` 降到 `38.798639`
  （相对 `-18.88%`）；两臂 finite/ledger/temperature/force safety 均通过，因此不是
  数值崩溃。按 DEC-059 停止方案② MTS，不重调 `c1`/checkpoint/seed。
- [x] 按 DEC-059 启动独立 EXP-014 native-compression screen；`n_radial=8/16/32` 均未
  通过三折 `R²`/retention 共同门（DEC-060）。不进入 OpenMM force qualification；此前
  `INVALIDATED_OUT_OF_ORDER` 报告不作为证据。
- [ ] 只有后续候选独立通过资格后，才可重新定义 013-C。
- [ ] 每个候选方案至少做 baseline、candidate 和必要的 N=1 reference。
- [ ] 使用 OpenMM State API 在不同 Integrator 间传递状态；禁止直接跨 Integrator `loadCheckpoint()`。
- [ ] 单独记录 PythonForce/CUDA 的 `CUDA_ERROR_INVALID_HANDLE` 风险；不能用 TorchForce 的结果替代后端资格。

### 6.3 MTS 验收门

- [ ] 3 个真正独立重复；固定独立初态/平衡轨迹和 seed 规则并提前登记。仅重抽速度、共享初态或 bootstrap 不增加 repeat 数。
- [ ] 绝对健康：温度、能量漂移、有限能量/力、异常结构率。
- [ ] 相对健康：相对 N=1 reference 的温度/能量/IBS 判别式偏移及 block bootstrap CI。
- [ ] 统计健康：真正的 overlap/ESS、crossing rate、自相关时间、迟滞。
- [ ] 效率健康：正式 BAR/MBAR mutual overlap、importance/absolute ESS 和 ESS/GPU-hour 至少 `2/3` 独立重复优于 baseline；`mixture_ess_proxy_per_gpu_hour` 只能作 exploratory 辅助指标，不能作为 promotion 硬门。
- [ ] 自由能健康：`ΔG_MTS` 与 N=1 student/converged-MM 在统计误差内一致。
- [ ] 任一方案若能量健康但出现系统性 distribution shift，判为不适合 production。
- [ ] 若所有 MTS 方案失败，停止继续机械搜索更新频率；将结论写成“Cartesian-force amortization 不合适”，不写成“slow-information 不合适”。

---

## 7. P2：重新进入 WP-5A production qualification 的条件

只有新 cheap online route 通过离线 temporal audit、物理账本和成本门后，才恢复 WP-5A。

### 7.1 固定对照组

- [ ] 原始基础路径。
- [ ] 仅 λ 重排。
- [ ] 新建的 analytic cheap-CV control（若实现并通过资格门）；不得把历史 `not_pursued` 的 EXP-012 Arm A 写成已复活。
- [ ] 单个冻结 neural basis / cheap student。
- [ ] 必要时再比较 `M=2–4`，不得提前展开。

### 7.2 统计设计

- [ ] 至少 `3` 个真正独立的 production repeats；论文级结论目标为 `5` 个。
- [ ] 独立 repeat 必须具有独立初态/独立平衡轨迹和预登记 seed。baseline 与 candidate 可在 repeat 内配对，但共享初态、仅重抽速度、frame bootstrap 或 episode bootstrap 均不得增加独立 repeat 数。
- [ ] held-out run 不参与模型选择、超参数选择或阈值选择。
- [ ] 训练、选择、验证、production repeat 的边界在 manifest 中明确。
- [ ] 不用 100-frame pilot 宣称 ΔG 收敛；pilot 只能作为混合趋势和工程成本检查。
- [ ] 按 episode/block 统计 crossing、驻留、迟滞和自相关。
- [ ] 输出 raw frames、block count、ESS、IAT 和 uncertainty。

### 7.3 必测指标

- [ ] 相邻 reduced-energy gap variance。
- [ ] BAR mutual overlap，使用严格双向 mutual overlap 定义。
- [ ] MBAR/importance ESS。
- [ ] `ESS/GPU-hour`，作为主 promotion metric。
- [ ] physical round trip、`state 1↔0` event rate、驻留时间和 hysteresis；所有 surrogate metric 必须分栏报告。
- [ ] torsion、hydration、rotamer/contact topology 的独立转换率。
- [ ] 能量/力分位数、最大力、异常结构率和支持域外比例。
- [ ] 与 converged-MM 的 ΔG 一致性。
- [ ] complex/solvent 两条腿、cycle closure、公共 λ 状态和 resume 结果一致。

### 7.4 生产准入门

新方案必须同时满足：

- [ ] endpoint ΔG/force 与基础路径在容差内一致；
- [ ] target/bias/base ledger 审计通过；
- [ ] 至少 `2/3` 独立重复的正式 ESS/GPU-hour 优于 baseline；
- [ ] 不出现不可接受 distribution shift、异常结构或非物理深井；
- [ ] 自由能差在统计误差内一致；
- [ ] 配置、模型、协议、系数和缓存 hash 全部可复现；
- [ ] 明确 go/no-go 结论，而不是只给“看起来有改善”。

---

## 8. P2：多基势、DEXP 和跨体系验证的启动条件

### 8.1 多基势 `M=2–4`

- [ ] 仅在单个 cheap coordinate/basis 已通过 WP-5A 且 residual correction matrix 显示空间变化时启动。
- [ ] 用 SVD/低秩分析证明增加分量的必要性；不能因为单基势效果不稳定就盲目堆到 `M=4`。
- [ ] 检查多基势之间的相关性、条件数、系数有界性和端点归零。
- [ ] 每增加一个 basis 都重新做 ledger、force、cache、性能和 independent repeat 资格。
- [ ] 若需要十几个以上分量，停止多基势路线，评估 continuous λ-conditioned GNN 或重新定义 slow coordinate。

### 8.2 DEXP

- [ ] DEXP 不是当前路线的默认逃生口。
- [ ] 若切换 DEXP，先做无神经项的 complex/solvent ABFE baseline。
- [ ] 先验证 endpoint、cycle closure、long-range correction、tail term 和 independent repeats。
- [ ] 无神经 DEXP baseline 未通过前，不把 neural path 和 DEXP 同时引入，避免无法归因。

### 8.3 跨体系验证

- [ ] 在至少两个额外 ABFE benchmark / 困难窗口上测试已冻结的方法。
- [ ] 不人工指定新的 CV，不修改模型超参数，不为每个体系重新挑 checkpoint。
- [ ] 清楚区分：跨体系复用同一冻结权重，和跨体系复用同一训练/部署管线。
- [ ] 报告失败体系、support-domain violation、显存和成本；不能只报告成功案例。

---

## 9. 回滚、缓存和可复现性

- [ ] 默认配置保持：

  ```yaml
  neural_path:
    enabled: false
  ```

- [ ] 每次运行写入 immutable `run_id/protocol_sha256` 目录，不覆盖基础路径缓存。
- [ ] 所有数组和 manifest 通过 generation pointer 原子提交；失败时只允许切回上一完整 generation，禁止拼接不同 generation。
- [ ] 失败模型、失败日志、失败原因和原始 manifest 全部保留。
- [ ] 任一模型、系数、λ schedule、账本、精度、PBC、Force backend 或 temporal protocol 变化都触发 cache invalidation。
- [ ] 不仅按文件名判断模型相同；必须比较内容 hash 和 spec hash。
- [ ] 运行结束保存：git revision、配置文件、模型 hash、协议 hash、环境版本、GPU 信息、seed、trajectory manifest 和统计脚本版本。
- [ ] 禁止混用不同 protocol 的 target/bias/base/checkpoint；`enabled=false` 只恢复 legacy baseline，不把 neural generation 伪装成 baseline。
- [ ] 回滚后自动运行旧基线 regression、manifest/hash 校验，并确认旧结果可复现到预定容差。
- [ ] 不执行删除基础缓存或覆盖原始实验目录的操作。

---

## 10. 建议执行顺序（短版）

```text
P0 状态/协议审计 + 登记 DEC-055 主门失败事实
        ↓
EXP-016 数据可行性、连续轨迹与 temporal attribution 审计
        ↓
冻结 physical/surrogate event 与 episode-level target schema
        ↓
低维 qφ 与 cheap analytic/student route
        ↓
CPU/reference 力、真实端点、ledger、resume/cache 资格
        ↓
GPU/短 NVT/成本资格
        ↓
真正独立 repeats 的新 WP-5A qualification
        ↓
仅在证据需要时进入 EXP-014 native compression、M=2–4、跨体系或 DEXP
```

EXP-013/MTS 从主线旁路进入，不构成 EXP-016 的前置条件：

```text
DEC-055/056：方案③失败 → DEC-058：方案① Qualification gate 未通过 → DEC-059：方案② N=1 ESS 无信号 → DEC-060：EXP-014 screen 未通过
  └─ 当前冻结 contingency 停止；其它 compression 形式需新决策
```

任何 supplementary 结果都不能覆盖预注册主结果。

---

## 11. 最终研究叙事的验收句

在所有 TODO 完成前，不使用“找到了一个慢神经势”或“只要低频更新神经力就能保持正确采样”这样的表述。

只有在 EXP-016、在线资格和独立 production repeats 全部通过后，目标结论才可以写成：

> 我们在连续、强时间相关的 MD 轨迹上识别出与相邻 alchemical `state 1↔0` 混合瓶颈相关、并在 held-out trajectory 上可复现的 learned slow information。该信息通过冻结的瞬时低维坐标或轻量可导 student 被转译为廉价 online control/path term；其真实端点、守恒力、target/bias 账本、独立重复轨迹统计和 ESS/GPU-hour 均经过单独验证。MTS 只检验 Cartesian-force amortization 是否可行，不作为 slow-information 方法本身的成败判据。

若任一门未通过，只报告对应层级的事实，例如“candidate signal 改善 held-out gap variance”或“Cartesian-force MTS 失败”，不得提升为上述完整结论。

---

## 12. 待填写的决策记录

| 决策 | 负责人/日期 | 选择 | 证据文件 | 状态 |
|---|---|---|---|---|
| DEC-055/056/058/059：EXP-013 方案③→①→② | 2026-08-09 | 方案③未通过 013-B；方案① Qualification gate 未通过，故 `N16_NOT_AUTHORIZED`，且物理偏差结论保持 `INCONCLUSIVE`；方案② N=1 ESS 下降 `18.88%`，不晋级、不运行 MTS | `EXPERIMENT_LOG...`、`output/outer_lambda_exp013_design1_qualification/report.json`、`output/outer_lambda_exp013_design2_n1_ess/report.json` | `EXP013_NO_PROMOTION` |
| EXP-016 event definition | 2026-08-09 | physical history unavailable；登记 energy-weighted surrogate only | `output/outer_lambda_exp016_loro/EXP-016_data_manifest.json` | `SURROGATE_ONLY` |
| EXP-016 `τ_information` | 2026-08-09 | physical `τ_information` 不定义；仅报告 candidate IAT/`N_eff` | `output/outer_lambda_exp016_loro/EXP-016_temporal_audit.json` | `NOT_DEFINED` |
| cheap route | 2026-08-09 | 当前 audit 未通过 physical slow-information 门；不启动 production cheap route | `output/outer_lambda_exp016_loro/EXP-016_summary.md` | `BLOCKED` |
| EXP-014 native compression | 2026-08-09 | 独立离线 screen 未通过；不进入 OpenMM qualification 或 production promotion | `output/outer_lambda_exp014_native_compression_audit_after_exp013/EXP-014_native_compression_audit.json` | `SCREEN_NOT_PASSED / STOP` |
| 新 WP-5A qualification |  | go / no-go |  | `BLOCKED` |
| `M>1` 是否启动 |  |  | SVD/low-rank report | `BLOCKED` |
| DEXP 是否启动 |  |  | no-neural DEXP baseline | `BLOCKED` |
