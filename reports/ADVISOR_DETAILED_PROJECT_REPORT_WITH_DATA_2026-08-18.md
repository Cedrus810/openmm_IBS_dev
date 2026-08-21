# ABFE-IBS 项目详细进展报告：2026-08-18 决策更新版

> 整理日期：2026-08-18  
> 数值证据截止：2026-08-14  
> 前一版：[2026-08-13 数据版](ADVISOR_DETAILED_PROJECT_REPORT_WITH_DATA_2026-08-13.md)  
> 定位：本文件更新项目当前状态，并对两条 MACE/Neural-Operator 路线作出实验身份判定。它不是最终论文结果，也不把设计草案写成已验证结论。

---

## 0. 本次更新回答什么

本版集中回答四个问题：

1. 2026-08-13 之后，主 ABFE 数值和软件路线发生了什么变化；
2. 当前两条 MACE/ML 路线分别在改变什么；
3. 哪条路线是当前项目语义上正确、应继续执行的主线；
4. EDS、λ-EDS 与本项目方案到底相同在哪里、不同在哪里。

未被本版推翻的背景、热力学循环、Stage 1/Stage 2、Boresch、IBS/TMBAR、失败路线和旧实验细节，仍见 2026-08-13 数据版。本版对“当前状态”具有更高时效性；原始 JSON、日志和预注册文件仍高于叙述性报告。

---

## 1. 一页式结论

### 1.1 项目状态

- 旧候选值 `−23.1622 ± 2.5139 kJ/mol` 仍只能作为历史候选，不可作为论文最终值。
- 新的 seed 20260907 完整 artifact 给出 `−49.9024 ± 1.7383 kJ/mol`；它与旧候选值差异很大，说明 run-to-run spread 尚未闭合。
- 因此，新重复不是“结果已确认”，而是更强地证明正式独立重复、统一版本和 between-run uncertainty 仍是首要科学缺口。
- EXP-026 已证明专用 CUDA local-many-body residual 可以通过 correctness 和成本门；A2 对 no-plugin baseline 的规范成本比约为 `1.03206×`。
- EXP-027 原始长轨迹暴露 `addArg()` 累积导致的性能退化；EXP-028 已以固定参数槽 `setArg()` 修复并通过稳定性与回归验证。
- 修复后的 window-0 U3 得到正面 utility 信号，但 U4 复用了 baseline 的 `f_k`，不能用于 candidate 晋级。
- 下一项有效实验不是继续解释 U4，而是 EXP-029：baseline/candidate 各自在自己的 sampling Hamiltonian 下独立校准并冻结 `f_k`，再做完整 ITT A/B。

### 1.2 两条路线的判定

**当前 learned-residual 实验主线应采用 Route 2：固定 target states / fixed target Hamiltonians，只学习 sampling-only bias。**

这里的“实验主线”不等于“已晋级 production”。当前 production 仍是既有 ACE softcore + dual-λ + IBS/TMBAR，不包含已获生产授权的 neural correction。Route 2 目前是 EXP-027/029/030 的研究语义。

理由不是 Route 1 在统计力学上必然错误，而是：

1. Route 2 与当前 EXP-027/029/030 的 candidate、ledger、TMBAR 重加权语义一致；
2. 它不改变当前要计算的 `U_k`，因此可以直接回答“learned bias 是否提高原生产路径的 ESS/GPU-hour”；
3. 它能利用已经完成的 EXP-020 表示、EXP-026 CUDA、EXP-028 修复和现有 IBS 基础设施；
4. 它的下一个证伪实验已经明确，即 EXP-029，而不是重新开一套 path-optimization 工程。

**Route 1：端点固定、重设计中间 Hamiltonian，是概念上成立但尚未资格化的独立研究路线。**

仓库较早的 Outer-lambda neural-path 原型确实包含这种语义：endpoint-zero neural term 进入中间 target energy/discriminant；但它没有 production promotion。它应保留为独立方法学分支，不应被描述为“当前 EXP-029 sampling residual 的真实含义”，也不能与 EXP-029 共用实验身份。若重启，必须新建 protocol、target ledger、端点合同和与 EDS/λ-EDS/thermodynamic-length 的直接基线。

### 1.3 “正确”的精确定义

本报告所说 Route 2 当前正确，是指：

- **方法语义正确**：target thermodynamics 与 sampling machinery 分离；
- **与当前实现一致**：residual 是 sampling-only bias；
- **可被现有 estimator 严格撤销**：实际 bias history 与 target energies 完整记录时，可重加权回原 `U_k`；
- **下一实验可证伪**：EXP-029 可直接测 ITT utility。

这不等于它已经证明比 baseline、EDS 或其他增强采样方法更高效。科学效用仍是 `NOT YET QUALIFIED`。

---

## 2. 2026-08-13 后的主 ABFE 数值更新

### 2.1 新增 seeded 产物

| 产物 | 当前可引用数字 | 状态边界 |
|---|---:|---|
| 旧 `output_lrc_fix` 候选 | `−23.1622 ± 2.5139 kJ/mol` | 历史候选；不是正式重复闭环 |
| seed 20260906 complex leg | `202.6621 ± 1.8244 kJ/mol` | complex 完成；不得伪装成完整 binding result |
| seed 20260907 complex leg | `201.1478 ± 1.2647 kJ/mol` | Stage 2 6/6 windows、23 λ 覆盖 |
| seed 20260907 complete binding artifact | `−49.9024 ± 1.7383 kJ/mol` | 完整单次 seeded artifact；不可单独晋级为最终值 |

seed 20260907 artifact 同时记录：

- solvent：`151.2454 kJ/mol`；
- Boresch：`−35.9300 kJ/mol`，已包含在 complex leg，不能重复扣除；
- OpenMM 8.5.2、CUDA、250k steps/window、500k warmup、IBS v29；
- 时间戳：2026-08-14T00:23:08。

### 2.2 正确解释

新结果不能与旧候选直接取平均，原因包括：

- 代码/provenance 不完全相同；
- 并非所有 repeat 都有同等级完整 binding artifact；
- artifact 内部的 `independent_repeats` 资格标记尚未闭合；
- 两个完整候选之间的差异远大于各自内部误差。

所以本项目当前最诚实的结论是：

> 已经产生新的独立 seed 产物，但正式跨运行重复性仍未建立；内部 estimator uncertainty 明显低估了当前总的不确定性。

### 2.3 对优先级的影响

主 ABFE 线的 Priority 1 保持不变，但措辞应更新为：

1. 统一 code hash、OpenMM/PyMBAR 环境和完整 seed ledger；
2. 至少完成同协议、同版本、完整 complex+solvent 的多次独立运行；
3. 报告 run-level spread、block uncertainty 与 between-run uncertainty；
4. 在上述闭合之前，不发布单一最终 Atenolol 结合自由能。

---

## 3. EXP-025 至 EXP-028：专用 CUDA 路线已经跨过设计阶段

2026-08-13 版把 EXP-025 写成 `DRAFT_DESIGN_ONLY`。这一状态已经过时。

### 3.1 已完成的工程事实

EXP-026 A1.1/A2 已完成专用 local-many-body CUDA backend 的关键资格：

- energy/force parity 与 mixed-precision regression 通过；
- serialization、context update、错误路径与三方等价验证通过；
- A1.1 normative cost：median `1.04140×`，P95 upper `1.04692×`；
- A2 normative cost：median `1.03206×`，P95 upper `1.03665×`；
- 结论：`STOP_OPTIMIZATION_SUCCESS`，允许进入在线效用鉴定，而非直接宣布生产收益。

### 3.2 EXP-027/028 的关键发现

原始 EXP-027 长轨迹发现 14 个 CUDA kernel 每步重复 `addArg()`，使 kernel-launch 参数容器持续增长，形成随步数恶化的控制面成本。

EXP-028 修复为：

- 首次注册固定参数槽；
- 后续只以 `setArg(fixed_idx, ...)` 更新；
- 修复前 30k steps 约从 `2.74` 恶化到 `6.22 ms/step`；
- 修复后从 `2.6537` 到 `2.7406 ms/step`，稳定性 ratio `1.033`；
- RSS 无持续增长；关键 correctness/regression 通过。

这项修复保留 EXP-026 的短程成本结论，同时补上了真实长轨迹未覆盖的缺陷。

### 3.3 U3 与 U4 的边界

修复后的 window-0 U3：

- 2/3 repeats 的 utility difference 为正；
- median relative improvement `+49.4%`；
- candidate/baseline GPU-hour ratio 约 `1.025–1.041`；
- 状态：`EXP027_U3_WINDOW0_UTILITY_PASS`。

但这只能解释为 `BASELINE_FK_TRANSFER / NO_RECALIBRATION` stress test。U4 全六窗口虽然完成计算，却因 candidate 直接复用 baseline `f_k` 而被封存为：

```text
EXP027_U4 = INVALID_FOR_PROMOTION_BASELINE_FK_REUSED_FOR_CANDIDATE
```

它不是 candidate production failure，也不是 candidate utility pass。

---

## 4. 当前实验主线：EXP-029/030 的准确语义

### 4.0 先分清三个并存身份

| 身份 | Neural/residual 如何使用 | 当前状态 |
|---|---|---|
| production ABFE | 不包含已晋级的 neural correction | ACE softcore + dual-λ + IBS/TMBAR 当前生产基线 |
| EXP-027/029/030 candidate | residual 只属于 sampling layer，最终重加权回 baseline target | 当前 learned-residual 实验主线；未 production promotion |
| 旧 Outer-lambda neural-path 原型 | endpoint-zero neural term 进入中间 target Hamiltonian 与 IBS discriminant | Route 1 类研究原型；未 production promotion |

所以 `route2.md` 正确概括的是第二行，不是对仓库所有 outer-λ 代码的追溯性改名。

### 4.1 Target 与 sampling 必须分层

固定的目标状态为：

\[
p_k(R) \propto \exp[-\beta U_k(R)].
\]

candidate 实际采样：

\[
q_{k,\theta}(R)
\propto
\exp\{-\beta[U_k(R)+B_{\theta,k}(R)]\}.
\]

其中：

- `U_k` 是最终要恢复的 target Hamiltonian；
- `B_{θ,k}` 是 frozen representation 驱动的 sampling-only bias；
- `q` 可以改变，但 `p` 不改变；
- 最终 estimator 必须用实际 target energies 和完整 bias history 撤销 sampling bias。

因此，严谨说法不是“增加了 target overlap”，而是：

> 在固定 target states 上改善 bridge sampling、mixing、reweighted overlap efficiency 和 ESS/GPU-hour。

### 4.2 EXP-029 必须比较完整参数集

baseline 与 candidate 应比较：

\[
\Theta_b=\{0,\mathbf f_b^*\},
\qquad
\Theta_c=\{\phi_*,\mathbf f_c^*(\phi_*)\}.
\]

两臂必须：

1. 各自在自己的实际 sampling Hamiltonian 下校准并冻结 `f_k`；
2. 使用相同校准、冻结、rescue 和 production 停止规则；
3. 使用配对初态/速度/seed，但轨迹与 `f_k` 独立演化；
4. 完整记录 target energies、bias history、occupancy、g、ESS 与全部 GPU-hour；
5. 把 warmup、失败尝试、校准、rescue 和 production 全部计入 ITT 成本。

### 4.3 当前判决门

Route 2 只有同时满足以下条件才可晋级：

- ΔG 一致性通过；
- TMBAR/coverage/endpoint uncertainty 门通过；
- 至少 2/3 独立 repeats 的 candidate ITT utility 优于 baseline；
- 中位提升达到预注册门槛；
- 优势不能只来自单一 `min` 统计量或某次偶然 decorrelation；
- fixed bias 的可撤销性和最终 target ledger 经独立检查。

---

## 5. 两条 MACE/Neural-Operator 路线

### 5.1 Route 2：fixed target + learned sampling layer

定义：

\[
\lambda_k\ \text{fixed},
\qquad
U_k(R)\ \text{fixed},
\qquad
\widetilde U_k(R)=U_k(R)+B_{\theta,k}(R).
\]

Frozen MACE 或当前更轻量的 local-many-body representation 用来识别传统一维 λ/energy overlap 没有充分描述的结构慢模；MLP/operator 输出可记录、可冻结、可重加权的 sampling control。

当前已经进入 EXP-027/029 证据链的是 EXP-020/025/026 的 lightweight local-density residual，不是完整在线 Frozen MACE。Frozen MACE encoder 目前应写成后续 representation ablation 候选，不能写成已经资格化的 production 组件。

优点：

- 与当前代码和实验身份一致；
- 不改变原 alchemical target ledger；
- 可直接复用 IBS/TMBAR 与 CUDA backend；
- 失败时只否证 utility，不污染原目标热力学；
- 最接近当前真实瓶颈：固定窗口内的 orthogonal structural relaxation。

主要风险：

- learned bias 可能提高局部 mobility，却降低重加权 ESS；
- MACE latent 距离不自动等于 thermodynamic relevance；
- adaptive training 与 production 若不严格分离，会破坏固定分布与误差分析；
- production 前必须冻结 representation、bias 参数和各 arm/window 的 `f_k`；若在线更新，普通静态 MBAR/TMBAR 语义不再自动成立；
- target、base、Group-1 IBS bias、WCA、LRC、state/source 与 residual 必须保持完整分账；
- `B_θ`、`βB_θ`、`f_k` 的 reduced/physical units 与 gauge 必须明确，防止 double-β 或漏乘 `kBT`；
- 计算开销、校准成本和 rescue 必须计入 ITT；
- 当前 U3 正信号不足以替代完整 EXP-029。

**判决：CURRENT LEARNED-RESIDUAL EXPERIMENT MAINLINE / SEMANTICALLY CORRECT / UTILITY UNPROVEN / NOT PRODUCTION-PROMOTED。**

### 5.2 Route 1：endpoint-constrained path engineering

定义：

\[
H(R;s)=H_0(R;\xi(s))+\sum_a c_a(s)V_a(R),
\qquad
c_a(0)=c_a(1)=0.
\]

或以结构参数化保证：

\[
c_a(s)=s(1-s)\widetilde c_a(s).
\]

只要端点严格等于真实 `H_A/H_B`、所有中间 Hamiltonian 可精确求值并被正确纳入 BAR/MBAR/IBS/TMBAR，端点自由能差原则上与中间路径选择无关。因此该路线的核心统计力学命题成立。

但它不是当前 Route 2 的另一种表述，因为它改变了：

- target intermediate Hamiltonians；
- 生产状态 ledger；
- window placement 与可能的 state topology；
- estimator 输入矩阵及 overlap 定义；
- 必须比较的 prior art 和 baseline。

主要风险：

- endpoint anchoring 只是必要条件，不保证中间态可采、支持域连通或有限方差；
- MACE MMD/latent metric 可能与 reduced-energy overlap、kinetics 或最终 estimator variance 不一致；
- probe-state reweighting 在低 overlap 区域最容易失效，不能靠 operator 外插代替真实 MD；
- 对同一数据反复优化路径再报告收益，会产生 selection bias，必须有外层 held-out repeats；
- neural operator 在单体系、低维 control manifold 上可能没有超过 spline/GP/thermodynamic metric 的必要性；
- 改中间态本身已有 EDS/λ-EDS、minimum-variance path、optimized alchemical path 等直接先例，创新点不能写成“首次学习更平滑路径”。

**判决：CONCEPTUALLY VALID SEPARATE ROUTE / REPRESENTED BY AN UNPROMOTED OUTER-λ PROTOTYPE / NOT THE EXP-029 SEMANTICS / NOT AUTHORIZED。**

### 5.3 Route 1 草案中还必须修正的技术点

1. 草案中的 Gaussian `V_soft` 若以正 `A`、正 `c_soft` 加到 Hamiltonian，会抬高该区域能量；它不自动等于“软化排斥”。必须明确符号，并排除新吸引井、坍缩和 metastable basin。
2. “MACE 误差不会造成 endpoint free-energy bias”只能在端点合同、实际中间 Hamiltonian 完整记账、充分采样和正确 estimator 同时成立时使用。理论端点不漂移，不等于有限样本没有偏差。
3. probe-state 的简单 `exp[-βΔH]` 权重不够描述当前 IBS mixture。必须包含 source mixture、冻结 `f_k`、Group-1 bias、WCA、target/base/LRC ledger，并使用相关样本修正后的 ESS。
4. thermodynamic metric、MACE-MMD metric 与 kinetic penalty 没有天然同尺度；descriptor normalization、kernel bandwidth、组合权重和不确定度必须预注册。
5. endpoint anchoring 只保证两端相等，不保证中间 Hamiltonian 有界、可归一化、力稳定、支持域连通或容易采样。
6. 单体系、少量 windows/probes 不足以证明 neural operator 的 discretization-independent 泛化。首轮应优先 spline/GP，并为 operator 设置 uncertainty 与 trust region。

### 5.4 为什么现在不能合并两条路线

若同时让 `B_θ` 既是 target intermediate correction、又是可撤销 sampling bias，则同一个能量项有两套互相冲突的语义：

- 是否应进入 target reduced-energy matrix；
- 是否应在最终 estimator 中撤销；
- 端点为零是否只是便利条件还是物理合同；
- `f_k` 应对哪一个 Hamiltonian 校准。

因此必须一项一身份：

- EXP-029/030：sampling-only；
- 未来 neural-path 实验：target-path engineering；
- 不允许通过改名或复用 ledger 在两者之间静默切换。

---

## 6. EDS、λ-EDS 与本项目方案的异同

### 6.1 EDS 的核心对象

经典 EDS 构造一个 auxiliary reference/enveloping Hamiltonian，使其 Boltzmann weight 由多个 end-state Boltzmann factors 的加权组合得到。最终端点 target 保持不变，并可从 reference ensemble 重加权恢复，因此经典 EDS 也确实是一种 enhanced/reference-state sampling 方法；不能简单说成“EDS 不属于 sampling enhancement”。

λ-EDS 进一步把这种 enveloping 构造用于 alchemical intermediate states，通过 λ、smoothing parameter 和 energy offset 直接设计连接端点的一系列中间 Hamiltonian。仓库内原文给出的典型形式为：

\[
V_{\lambda\text{-EDS}}(\lambda)
=-\frac{1}{\beta s(\lambda)}
\ln\left[
(1-\lambda)e^{-\beta s(\lambda)V_A}
+\lambda e^{-\beta s(\lambda)(V_B-E(\lambda))}
\right].
\]

所以经典 EDS 可以从“辅助 reference ensemble”角度理解；λ-EDS 及后续 pairwise EDS coupling 更直接属于 path/intermediate-state redesign。二者都通过解析的 enveloping Hamiltonian 改善覆盖，但讨论时不能混成完全相同的实验对象。

### 6.2 与 Route 1 的关系

相同点：

- 两者都承认中间 Hamiltonian 是可设计对象；
- 都保持目标端点的热力学定义；
- 都希望降低 barrier、改善相邻分布连接或减少耗散；
- 最终正确性依赖对实际采样 Hamiltonian 的精确记账与估计。

不同点：

- EDS/λ-EDS 使用有明确统计力学结构的 log-sum-exp/enveloping family；
- Route 1 提议在更一般的 Hamiltonian-control manifold 上，以 frozen MACE descriptor、结构/动力学指标和 operator 重建设计 landscape；
- Route 1 的潜在新意不是“端点固定而中间态可改”，而应是 many-body structural bottleneck 的诊断、跨 control-space 的预测和受约束优化是否提供超出 EDS/thermodynamic metric 的可重复收益。

因此 Route 1 若立项，EDS/λ-EDS 必须是直接 baseline，而不是只放在 related work。

### 6.3 与 Route 2 的关系

相同点：

- 都引入非物理的辅助采样分布；
- 都试图提高困难状态之间的统计连接；
- 都必须保留可复现的 Hamiltonian/bias 参数并进行严格统计处理。

关键不同：

| 问题 | EDS / λ-EDS | Route 2：本项目当前方案 |
|---|---|---|
| 主要改变对象 | 经典 EDS：auxiliary reference；λ-EDS：intermediate Hamiltonian family | fixed `U_k` 之上的 sampling distribution |
| `U_k` 是否保持原样 | 物理端点保持；λ-EDS 中间态重构 | 是，全部离散 target intermediates 保持 |
| target overlap | 可以真实改变所定义中间态之间的 overlap | 原 target overlap 不变，只改善有效/重加权采样效率 |
| 控制形式 | end-state energies 的 enveloping/log-sum-exp 与参数 | configuration-dependent learned bias |
| 最终恢复对象 | EDS 构造下连接的端点自由能 | 原 fixed target states 的自由能 |
| 当前项目状态 | 外部/对照方法 | EXP-029/030 主线 |

一句话概括：

> EDS/λ-EDS 主要是在重构“连接端点的路”；Route 2 保留既定的路和站点，用可撤销的 learned sampling layer 改善车辆在高维结构方向上的通行效率。

### 6.4 与 EDS 比较时不能说什么

当前论文口径不应写：

- “MACE 学习了更好的 intermediate Hamiltonian”；
- “本方法是 EDS 的神经网络推广”；
- “learned bias 增加了 fixed target distributions 的理论 overlap”；
- “端点 bias 为零就自动保证 estimator 正确”。

更准确的表述是：

> We retain the original discrete alchemical target Hamiltonians and introduce a learned, configuration-dependent sampling layer. A frozen many-body representation identifies structural slow modes that limit reweighted connectivity, while the auxiliary bias is removed in the final target-state estimator.

---

## 7. 推荐的实验顺序

### 7.1 现在：完成 Route 2 的最小决定性实验

1. 完成 EXP-029 harness 的 tiny wiring/state-machine smoke；
2. baseline/candidate 独立 calibration 与 freeze；
3. 全 6 windows、3 paired repeats；
4. 统一计算 ΔG consistency、occupancy、decorrelation、ESS 与 ITT GPU-hour；
5. 按预注册门作 `PROMOTE / STOP / INCONCLUSIVE` 判定。

在 EXP-029 之前，不增加 frozen MACE、operator 或新 target-path 自由度。

### 7.2 若 Route 2 通过

再做 representation ablation：

- 当前 local-many-body residual；
- 简单几何/CV bias；
- frozen MACE representation；
- MLP 与 operator 的同预算比较。

只有 frozen MACE 在 held-out repeats 上稳定识别并改善传统指标遗漏的慢模，才建立其方法学贡献。

### 7.3 Route 1 的启动条件

Route 1 只有在独立 protocol 下满足以下条件才建议启动：

- 明确目标体系存在 path/intermediate-state bottleneck，而非主要是 fixed-state orthogonal relaxation；
- 先用低维 control space 验证，不直接上 neural operator；
- baseline 至少包括原路径、手调/thermodynamic-length 路径和 EDS/λ-EDS；
- 路径选择数据与最终评价 repeats 分离；
- 所有中间 Hamiltonian、参数、reweightability 与支持域门预注册；
- 证明收益来自 learned structural metric，而不是额外 probe MD 或更多状态预算。

---

## 8. 当前可以与导师讨论的决策问题

1. 是否同意把 Route 2 固定为 EXP-029/030 唯一当前语义？
2. EXP-029 的 ITT 成本门、中位 utility 提升门和最小 repeat 数是否足够严格？
3. 新 seeded binding result 的巨大 spread 是否要求先暂停方法开发、优先闭合主 ABFE 重复性？
4. Route 1 是否值得作为独立后续项目，而不是并入当前 residual？
5. 若启动 Route 1，第一直接对照选 λ-EDS、thermodynamic length 还是两者都做？
6. frozen MACE 的贡献应定位为 diagnostic representation，还是在证据足够后再升级为 bias/controller？

---

## 9. 当前状态标签

| 项目 | 状态 |
|---|---|
| Atenolol 最终可发表 ΔG | `NOT QUALIFIED` |
| 新独立 seed artifact | `AVAILABLE_BUT_NOT_POOLED` |
| EXP-026 CUDA backend | `CORRECTNESS_AND_COST_QUALIFIED` |
| EXP-028 long-run fix | `FIXED_AND_VALIDATED` |
| EXP-027 U3 window-0 | `POSITIVE_STRESS_TEST_SIGNAL` |
| EXP-027 U4 | `INVALID_FOR_PROMOTION_BASELINE_FK_REUSED_FOR_CANDIDATE` |
| Route 2 / EXP-029 | `CURRENT_EXPERIMENT_MAINLINE / AUTHORIZED_NOT_STARTED / NOT_PRODUCTION_PROMOTED` |
| EXP-030 joint score | `DESIGN_DRAFT_NOT_STARTED` |
| Route 1 neural path | `UNPROMOTED_OUTER_LAMBDA_PROTOTYPE / SEPARATE IDENTITY / NOT AUTHORIZED` |
| EDS/λ-EDS comparison | `REQUIRED_BASELINE_FOR_ROUTE_1` |

---

## 10. 主要证据入口

- [2026-08-13 详细数据版](ADVISOR_DETAILED_PROJECT_REPORT_WITH_DATA_2026-08-13.md)
- [EXP-026 CUDA 控制面优化](../PLAN_EXP-026_cuda_control_plane_optimization.md)
- [EXP-027/028 结果汇总](../exp027_result.md)
- [EXP-027 在线效用计划与过程记录](../PLAN_EXP-027_online_utility.md)
- [EXP-030 joint state score 设计](../exp-30.md)
- [Outer-lambda neural basis 原型](../outer_lambda_neural_basis.py)
- [IBS 原文提取](../references/papers/integrated-boltzmann-sampling.md)
- [λ-EDS 原文/PDF材料](../references/)
- `output_lrc_fix_repeat02_seed20260906/final_results.json`
- `output_lrc_fix_repeat03_seed20260907/final_binding_results.json`
- `output/outer_lambda_exp027_online_utility/exp028_u3_confirmation_report.json`

---

## 11. 最终判断

当前不是“在两个互斥理论里猜一个”。更准确的决策是：

1. **Route 2 是当前 learned-residual 实验应继续执行的路线**，因为它与 EXP-027/029/030 的 candidate、实验身份和可撤销 sampling-bias 语义一致；
2. **当前 production 仍不包含 neural promotion**；Route 2 通过 EXP-029 前不得写成生产方法；
3. **Route 1 在原则上成立，仓库也有未晋级的 Outer-lambda 原型，但必须维持独立 target-path engineering 身份**，不能回写成 EXP-029 residual 的含义；
4. **λ-EDS 与 Route 1 更接近；经典 EDS 与 Route 2 共享辅助/reference sampling 与最终重加权这一高层结构，但构造方式不同**；
5. 在任何新 operator 工作之前，先用 EXP-029 回答当前 local residual 是否真正提高 ITT ESS/GPU-hour；
6. 主 ABFE 新重复的巨大 spread 仍是比方法包装更优先的科学风险。
