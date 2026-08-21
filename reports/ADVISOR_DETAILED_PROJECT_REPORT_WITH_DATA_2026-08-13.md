# ABFE-IBS 项目详细进展报告：方法、数据、失败路线与下一阶段计划

> 用途：导师组会、阶段汇报和论文讨论的中文详细底稿  
> 建议讲述时间：30–50 分钟  
> 整理日期：2026-08-13  
> 项目体系：Atenolol-rank11  
> 文档性质：阶段性技术报告，不是最终论文结果声明

---

## 0. 本报告怎样阅读

本报告试图回答六个问题：

1. 这个项目究竟要解决什么科学问题？
2. 当前 ABFE-IBS 软件已经实现了什么？
3. 当前可以展示哪些定量结果？
4. 哪些结果只是候选值，为什么还不能发表？
5. MACE、Outer-lambda、DEXP、局部多体残差等新路线为什么提出、为什么失败或停止？
6. 下一阶段最值得投入计算资源和开发时间的工作是什么？

全文采用以下证据标签：

| 标签 | 含义 |
|---|---|
| `IMPLEMENTED` | 代码路径已经存在，但不自动等于真实体系已验证 |
| `QUALIFIED` | 在预先定义的对应层级通过资格门 |
| `CANDIDATE` | 可用于内部验收和比较，但不能作为论文最终结果 |
| `FAILED` | 明确执行过，且未通过预注册或冻结门槛 |
| `INCONCLUSIVE` | 数据不足以支持正面或负面科学结论 |
| `STOPPED` | 根据停止规则不再沿同一路线追加资源 |
| `PENDING` | 尚未执行或尚未闭合 |
| `INVALIDATED` | 已确认存在协议、符号、输入或分析错误，仅保留审计价值 |

这里最重要的原则是：

> 计划不等于执行，代码实现不等于验证，单次运行不等于可重复结果，文件名中的 `final` 也不等于可发表。

---

## 1. 一页式结论

### 1.1 当前完成了什么

ABFE-IBS 已经形成一套较完整的软件和方法骨架：

- 读取 GROMACS `.gro/.top` 并构建 OpenMM System；
- 自动建立 complex leg 和 solvent leg；
- 通过 Boresch restraint 处理配体相对构型与标准态修正；
- 将解耦过程拆成 Stage 1 去电荷和 Stage 2 去 van der Waals；
- Stage 2 使用 dual-lambda、ACE softcore、IBS 扩展系综采样和 TMBAR 分析；
- 支持 λ 路径预优化、LJ long-range correction、checkpoint/resume、非突变 rescue 和 provenance；
- 对带电配体、膜体系、APBS 修正和神经残差保留了扩展接口或实验实现。

当前源码和主要候选 artifact 对应的核心协议身份为：

| 协议 | 当前身份 |
|---|---:|
| IBS bias protocol | v29 |
| Thermodynamic path | v21 |
| Traditional/analytic LJ LRC | v3 |
| WCA accounting | v2 |
| ESS gate | v3 |

### 1.2 当前最重要的数值

`output_lrc_fix` 给出的内部候选基线为：

\[
\Delta G_{bind}=-23.1622\pm2.5139\;\mathrm{kJ\,mol^{-1}}
\]

即：

\[
\Delta G_{bind}=-5.5359\pm0.6008\;\mathrm{kcal\,mol^{-1}}.
\]

但该结果在登记表中的状态仍是 `CANDIDATE / citable=NO`，原因包括：

- 没有正式独立重复；
- production seed ledger 为空；
- Boresch radial force constant 有一次裁剪；
- 当前 estimator 与 artifact-era Stage 1 分析口径仍需联合复核；
- 完整 OpenMM/CUDA 验证矩阵和跨运行时间相关不确定度尚未闭合。

因此当前最稳妥的总判断是：

> 软件与方法开发已经可以系统汇报；Atenolol 的最终结合自由能仍不能作为论文结论引用。

### 1.3 新方法路线的主要结论

神经与局部多体方向已经得到一个重要的分层结论：

1. 离线表示层面存在明显信号；
2. 通用部署后端的成本抵消了该信号；
3. 下一步不应继续堆更大的网络，而应测试高度专用的局部 CUDA kernel。

最强的离线证据来自 EXP-020 R1：

- 3 folds × 3 seeds，共 9/9 checkpoints；
- held-out gap-variance mean improvement：`55.5524%`；
- D1 qualification：通过；
- D2 energy/force/invariance qualification：通过；
- 但现有部署均未通过 `≤1.10×` 成本门。

---

## 2. 科学问题与研究动机

### 2.1 为什么做绝对结合自由能

绝对结合自由能 ABFE 的目标，是从分子模拟中估计一个配体从溶液进入受体结合位点时的标准结合自由能。与只比较相似配体的相对自由能不同，ABFE 试图独立闭合 complex 和 solvent 两条热力学腿，因此可以处理更广的化学变化，但采样和误差控制也更困难。

当前项目采用的符号约定为：

\[
\Delta G_{bind}
=\Delta G_{solvent}
-\Delta G_{complex}
+\Delta G_{APBS}.
\]

负值表示结合有利。解析 Boresch release 是否已经计入 complex leg，必须由账本字段决定，不能在最终公式中重复加减。

### 2.2 本项目面对的核心困难

项目的难点不只是“跑一次自由能计算”，而是同时解决：

- 配体消失时的奇异相互作用与端点稳定性；
- 相邻 λ 状态的 overlap；
- 长时间相关性、稀有构象和跨运行漂移；
- restraint 的几何选择、attachment 与标准态 release；
- PME 电荷处理和 LJ 长程尾修正；
- 大体系 GPU 成本；
- checkpoint/resume 后 Hamiltonian 和统计分布是否保持同一身份；
- 单次内部误差是否低估独立重复间的散布。

因此，本项目的研究目标可以概括为：

> 建立一条能够明确记录 Hamiltonian、采样分布、估计器、误差、修正项和失败原因的可审计 ABFE 流程，并探索是否能用低维或局部多体残差提高 Stage 2 的统计效率。

---

## 3. 整体热力学循环

### 3.1 Complex leg

复合物腿包含：

1. Boresch restraint attachment：从未受限态 (A'\) 到受限态 (A\)；
2. Stage 1 decharging：保持 van der Waals 存在，将 ligand electrostatics 从 1 变到 0；
3. Stage 2 vanishing：保持 ligand 静态中性，将 van der Waals 从 1 变到 0；
4. 解析 Boresch standard-state release。

可写成：

\[
\Delta G_{complex}
=\Delta G_{attach}
+\Delta G_{decharge}^{complex}
+\Delta G_{vdW}^{complex}
+\Delta G_{release}.
\]

### 3.2 Solvent leg

溶剂腿不使用 Boresch restraint，主要包含：

\[
\Delta G_{solvent}
=\Delta G_{decharge}^{solvent}
+\Delta G_{vdW}^{solvent}.
\]

### 3.3 Boresch 修正的物理意义

Boresch restraint 使用一个距离、两个角和三个二面角约束配体相对受体的平移与转动。其解析 release 项采用 1 M 标准体积 (V^\circ=1.6605\;\mathrm{nm^3}\)。

重要的是 attachment 和 release 是不同的物理步骤：

- attachment 是采样得到的自由能；
- release 是解析标准态修正；
- 两者遗漏或重复都会破坏热力学循环。

---

## 4. 软件 Pipeline：从输入到最终结果

```text
GROMACS .gro/.top
        |
        v
输入解析、include 发现、System/Topology 缓存、协议指纹
        |
        +---------------------------+
        |                           |
        v                           v
complex leg                    solvent leg builder
        |                           |
预平衡、PBC 修复、Boresch          水盒、同一 charge/dispersion 合同
        |                           |
Stage 0 attachment                 |
        |                           |
Stage 1 PME decharging <-----------+
        |
Stage 2 ACE/IBS vanishing
        |
Local TMBAR + covariance stitching
        |
LRC / Boresch release / optional APBS
        |
cross-leg conformer gate
        |
final_binding_results.json
```

主要模块职责如下：

| 模块 | 主要职责 |
|---|---|
| `runabfe.py` | CLI、配置、输入装配、两腿调度、最终汇总 |
| `abfe_core.py` | 势能、softcore、Boresch、solvent builder、热力学循环 |
| `abfe_preoptimizer.py` | Stage 1/2 λ 路径预优化和协议验证 |
| `ibs_engine.py` | IBS bias、窗口采样、TMBAR/MBAR、LRC、resume 状态 |
| `abfe_pipeline.py` | Stage 0/1/2 orchestration、cache、rescue、腿级结果 |
| `apbs_correction.py` | 外部 APBS/Rocklin/Wu-Biggin 静电修正 |
| `outer_lambda_neural_basis.py` | 隔离的 Outer-lambda/神经路径实验接口 |

---

## 5. 当前生产主线的工作原理

### 5.1 Stage 1：PME decharging

Stage 1 让 (lambda_{coul}:1\rightarrow0\)，同时保持 (lambda_{vdW}=1\)。电荷仍通过 PME-compatible NonbondedForce 参数偏移处理，从而避免在去电荷阶段直接丢失长程静电。

当前估计器合同中，相邻状态 BAR 是主值，finite-difference TI 用于一致性检查，MBAR 是诊断。旧 artifact 的 Stage 1 分析字段可能来自更早口径，因此需要 current-v2 reanalysis 才能宣称完全符合当前源码。

### 5.2 Stage 2：ACE softcore vanishing

Stage 2 让 (lambda_{vdW}:1\rightarrow0\)，且要求 ligand 在进入 IBS custom nonbonded 路径前已经静态中性。

当前默认 ACE softcore 使用无量纲、以 (sigma\) 缩放的 alpha 约定。其目的不是改变物理端点，而是软化中间 λ 状态的短程排斥，避免配体消失时出现数值奇异。

当前主要参数包括：

| 参数 | 当前默认/身份 |
|---|---:|
| (alpha_{LJ}\) | 0.5 |
| (alpha_{Coul}\) | 0.2 |
| alpha convention | `dimensionless_sigma_scaled_v2` |
| ACE cutoff | 1.0 nm |
| switching | 当前默认无 switching |

### 5.3 λ 路径 v21

Stage 2 首先在 17 个 uniform pilot states 上估计 Fisher-like metric：

\[
g(\lambda)=\beta_T^2\operatorname{Var}
\left(\frac{\partial U}{\partial\lambda}\right).
\]

然后依据热力学长度与几何 floor 生成最终 23 个严格递减 λ states。这里：

- 17 是 pilot/base grid；
- 23 是最终生产路径；
- 不能把两者写成配置冲突或两种生产状态数。

最终路径被切成只共享一个边界状态的窗口，每条 λ edge 只归属一个窗口，避免旧 overlapping schedule 的重复记账。

### 5.4 IBS 扩展系综采样

一个 IBS window 同时包含多个物理 λ 状态，其 bias 为：

\[
V_{bias}(x)=-k_BT\log\sum_k
\exp[-\beta(U_k(x)-f_k)].
\]

其中 (f_k\) 用于平衡各状态对采样分布的贡献。计算使用稳定的 max-shift log-sum-exp。

当前 force-group 账本为：

| Group | 内容 | 账本角色 |
|---:|---|---|
| 0 | common/background physical energy | base |
| 1 | IBS log-sum-exp 与 per-state softcore CV | sampling bias |
| 2 | ligand internal nonbonded | base |
| 3 | physical Boresch restraint | base |
| 4 | λ-WCA shield | sampling-only bias |
| 5 | 必要时的 COM restraint | base |

当前 WCA accounting v2 明确：Group 4 属于采样 bias，不能误算成 target Hamiltonian。

### 5.5 (f_k\) 学习、冻结和生产

IBS v29 不是在整段生产中持续更新 bias。其状态机为：

```text
pilot/TI seed
   -> 40-frame minibatch learning
   -> TMBAR update or bounded occupancy fallback
   -> readiness checks
   -> freeze f_k and record SHA-256
   -> fixed-f validation
   -> production with immutable f_k
```

v29 的重要实现点包括：

- minibatch：40 frames；
- TMBAR damping：0.10；
- pairwise applied cap：2 (k_BT\)；
- freeze readiness：连续两批最大调整不超过 1 (k_BT\)；
- validation：5 个 sliding batches；
- frozen bias 与 local MBAR 相邻差异门：10 kJ/mol；
- production 阶段禁止继续更新 (f_k\)。

旧 v27 文档中的 20-frame、0.20 damping 等描述不能代表当前实现。

### 5.6 Local TMBAR 与全阶段拼接

每个窗口的实际采样分布不是某一个纯物理 λ state，因此 Stage 2 不能直接把普通 BAR/TI 当作主值。分析使用 augmented local MBAR：

- row 0 表示实际 sampled mixture；
- rows 1…K 表示物理 target states；
- 使用最差 target reduced-energy sequence 同步去相关；
- 估计 endpoint covariance、mixture coverage ESS 和不确定度。

随后通过共享 λ 边界将窗口串成 covariance chain。最终收敛门包括：

- 所有窗口可解；
- minimum mixture coverage ratio；
- minimum decorrelated samples；
- maximum endpoint uncertainty。

### 5.7 LRC 与非突变 rescue

LJ LRC v3 是 sigma-resolved、softcore-aware 的每帧 (coeff(\lambda)/V(t)\) 修正，并同时处理 (r^{-6}\) 与 (r^{-12}\) 尾部。Custom CV 的 native LRC 被关闭，避免重复修正。

当前 rescue 不再就地改变已经失败窗口的 λ 或 Hamiltonian。允许的动作是：

1. 使用同一冻结 (f_k\) 和 checkpoint 最多延长两轮采样；
2. 仍失败时，在独立 immutable rescue 目录建立 bridge ensemble；
3. 原始窗口和文件全部保留，仅在合并分析中显式替换。

---

## 6. 当前候选 ABFE 数据

### 6.1 最终候选账本

`output_lrc_fix/final_binding_results.json` 记录：

| Quantity | 数值 | 状态 |
|---|---:|---|
| (Delta G_{complex}\) | 180.9981 kJ/mol | artifact component |
| (Delta G_{solvent}\) | 157.8358 kJ/mol | artifact component |
| Boresch correction field | −38.7609 kJ/mol | 已在 complex 账本中处理，不得重复加减 |
| (Delta G_{APBS}\) | 0 | 本 artifact 未启用 APBS |
| (Delta G_{bind}\) | −23.1622 kJ/mol | `CANDIDATE` |
| reported sampling uncertainty | 2.5139 kJ/mol | 尚未包含跨运行散布 |
| (Delta G_{bind}\) | −5.5359 kcal/mol | 同一结果的单位换算 |
| uncertainty | 0.6008 kcal/mol | 同一结果的单位换算 |

注意：Boresch correction 的物理量字段与最终 helper 实际扣除字段不是一回事。该 artifact 中 Boresch 已经进入 complex leg，因此最终结合自由能不能再手工减一次。

### 6.2 Stage 2 诊断

| 指标 | Complex | Solvent | 解释 |
|---|---:|---:|---|
| min overlap | 0.3913 | 0.4438 | 高于当时使用的最低 overlap 门 |
| min decorrelated samples | 96 | 266 | 两腿均有可用去相关样本 |
| min absolute ESS | 37.56 | 145.11 | complex 较弱；该 artifact 的 absolute ESS threshold 为 null |
| max endpoint uncertainty | 0.9249 kJ/mol | 0.9326 kJ/mol | 低于 1.0 kJ/mol 门 |
| λ nodes | 23 | 23 | v21 final path |
| dropped windows | 0 | 0 | 局部窗口均参与拼接 |
| artifact convergence | true | true | 只表示 artifact 内部门通过 |

这里不能写“所有 ESS 均超过 50”，因为 complex 最小 absolute ESS 是 37.56，而且该值在该 artifact 中只作为诊断，不是硬门。

### 6.3 Boresch 诊断

| 指标 | 数值 |
|---|---:|
| analyzed frames | 500 |
| anchor candidates | 562 |
| harmonicity flag | true |
| non-Gaussian count | 0 |
| clipped radial constants | 1 |
| raw (k_r\) | 约 7355.9 kJ/mol/nm² |
| applied (k_r\) | 2000 kJ/mol/nm² |

裁剪不必然意味着计算错误，但说明 restraint 参数处于人为安全上限，必须纳入敏感性和重复性讨论。

### 6.4 为什么仍不能发表

候选值的主要证据缺口如下：

| 缺口 | 当前情况 | 对结论的影响 |
|---|---|---|
| 独立重复 | `performed=false` | 无法估计 run-to-run variation |
| production seeds | 主 ledger 为空 | 无法完整追踪随机性 |
| 时间相关误差 | P1-22 未闭合 | 单次 endpoint σ 可能偏乐观 |
| estimator provenance | Stage 1 artifact 字段偏旧 | 需要 current estimator v2 重分析/重跑 |
| 协议联合验证 | 文档跨 v19/v27/v29 | 不能由旧测试自动证明新协议 |
| Boresch sensitivity | 1 个 (k_r\) 被裁剪 | 需要 restraint 鲁棒性检查 |

---

## 7. 统计不确定度：当前最关键的科学问题

P1-19/P1-22 暴露了一个比“单个窗口是否收敛”更重要的问题：单次内部不确定度可能低估跨运行散布。

已有同协议诊断值为：

```text
5.7726, 5.8623, 6.0786, 6.0880, 7.5216
```

其统计特征约为：

| 统计量 | 数值 |
|---|---:|
| 单次典型内部 σ | 约 0.10 |
| 五次样本 SD | 约 0.716 |
| 去除明显异常运行后的 SD | 约 0.158 |

这些值不是最终结合自由能，而是用于诊断同协议重复散布的量。它们说明：

- 单次 MBAR/TMBAR covariance 只描述当前采样集内部的不确定度；
- 慢构象、初始条件和时间相关性会产生额外 run-level variance；
- 离群运行是否可以剔除，必须由预先定义的物理或统计规则决定；
- 不能为了得到较小误差而事后删除数据。

下一步应采用：

1. 固定 protocol、输入和 padding；
2. 至少追加 1–2 个不同 seed 的独立重复，理想情况为 3 个以上；
3. 对 production time series 做 moving-block/bootstrap；
4. 同时报告 within-run 和 between-run variance；
5. 再决定是否需要扩大盒子、延长窗口或调整路径。

---

## 8. 旧结果为什么失效

### 8.1 2026-07-06 `+40.8362 ± 1.3178 kJ/mol`

状态：`INVALIDATED`。

主要问题：

- artifact 内置的是与当前项目相反的 (complex-solvent\) 符号；
- 使用了旧 PME self/LRC 处理口径；
- endpoint diagnostics 存在问题；
- 不能通过简单取负号把它“修成”当前结果。

### 8.2 2026-07-27 `+16.00 ± 2.20 kJ/mol`

状态：`INVALIDATED`。

主要问题：

- Boresch equilibrium 的 angle/dihedral 对应曾发生错误；
- 结构改变后复用了陈旧 restraint geometry；
- restraint 将配体拉动约 3.42 Å；
- complex leg 因此失去科学有效性。

这些负面结果的价值不是提供“早期预测”，而是推动了：

- Boresch 几何提交和 pose consistency gate；
- 更明确的符号账本；
- 缓存指纹与 fail-closed resume；
- 当前的候选基线重算。

---

## 9. 软件测试与验证状态

静态盘点显示：

| 项目 | 数量/状态 |
|---|---:|
| `test_*.py` 模块 | 88 |
| 顶层 `def test_...` | 1,022 |
| `cpu_only` 标记出现 | 53 |
| `needs_gpu` 标记出现 | 1 |
| parametrize 使用 | 70 |

历史日志曾记录：

| 日期/来源 | 记录结果 | 当前解释 |
|---|---|---|
| 2026-08-09 `memtodolist` B5 | 1161 passed / 3 skipped / 1 deselected / 0 failed | dated documented run |
| 2026-08-11 MEM-00h | 1213 passed / 0 failed | dated documented run |
| 更早 TODO | 956/977/979/1056 passed 等 | 口径不同，不能合并 |

这些不是本次重新运行的结果，而且旧 `VALIDATION_MATRIX` 与若干 TODO 数字存在冲突。因此汇报时可以说“测试体系规模较大且历史记录显示离线测试通过”，不能说“当前 v29 全量验证已经完成”。

当前仍需补齐：

- 一次冻结环境下的完整 offline test rerun；
- 目标 CUDA/OpenMM end-to-end；
- traditional REMD/fixed-box 回归；
- charged-ligand Stage 1→2 真实流程；
- membrane full ABFE；
- 独立 benchmark system set。

---

## 10. 为什么探索 Outer-lambda 与神经残差

Stage 2 的困难可能来自局部环境导致的非线性、慢变量或相邻 λ energy gap 方差。Outer-lambda 设计试图把昂贵、环境依赖的修正写成：

\[
U_\theta(x,\lambda)
=U_{base}(x,\lambda)
+\sum_m \phi_m(\lambda)b_m(x).
\]

当前 v1 使用端点为零的包络：

\[
w(\lambda)=\sin^2(\pi\lambda),
\]

从而确保候选残差不改变 (lambda=0\) 和 (lambda=1\) 的物理端点。

理想收益不是提高机器学习 (R^2\)，而是：

\[
\frac{ESS}{GPU\ hour}\uparrow
\quad\text{或}\quad
\frac{\sigma(\Delta G)}{GPU\ hour}\downarrow.
\]

因此所有神经路线都采用分层资格门：

```text
离线表示信号
 -> energy/force/invariance
 -> OpenMM parity
 -> 真实 CUDA 成本
 -> 短动力学
 -> 独立 online ESS/GPU-hour
```

任一层失败，都不能用前一层的成功代替。

---

## 11. MACE、TorchForce 与 MTS 路线

### 11.1 EXP-006/007：teacher 资格

- EXP-006 仅 path-force gate 失败：最大值约 258.949 kJ/mol/nm，高于 250 门；
- EXP-007 将 coefficient 固定为 0.09 后通过六项 qualification；
- 该通过只证明 teacher 构造可进入离线研究，不等于在线成本获批。

### 11.2 EXP-009：direct MACE-MTS

状态：`FAILED / STOPPED`。

CUDA MTS 在 N=1 就触发 `CUDA_ERROR_INVALID_HANDLE`，涉及 PythonForce/OpenMM-ML 后端。由于最基础频率已经失败，继续扫描 N=2/4/8 没有意义，因此不在同一后端重复投入。

### 11.3 EXP-010：cheap torsion CV

状态：`FAILED`，但只针对该 teacher/target 定义。

- 6 个预注册 LORO 全部没有胜过 intercept；
- intercept RMSE：21.5109；
- 最佳 1D order-2 RMSE：22.1737；
- generalized-force (R^2=-13.59\)。

根因是 protein-only atom-cut teacher 与 per-frame total interaction target 没有形成闭合可学习目标。该结果不能外推为“所有 torsion bias 都无效”。

### 11.4 EXP-011：periodic torsion PMF

状态：`FAILED / STOPPED`。

- formal 24 windows × 3 repeats；
- minimum overlap：0.02353，低于 0.03；
- decorrelated samples：22，低于 25。

按照预注册停止采样和拟合，转向直接 residual/gap 学习。

### 11.5 EXP-012/013：TorchForce 与物理 MTS

EXP-012 离线 direct-gap variance improvement 为 `13.9348%`，但只有 2/3 folds 改善。在线 exploratory 路线出现：

- TorchForce 成本：1.81–1.89×；
- 3 组 ESS/GPU-hour 均下降；
- 理想网络成本估计约 1.83×，高于 1.10× budget。

EXP-013 物理 MTS 的温度偏差：

| MTS 间隔 | temperature z-score |
|---:|---:|
| 2 | 5.61 |
| 4 | 5.79 |
| 8 | 6.83 |
| fused N=8 | 5.62 |

独立 student N=1 虽通过基础健康门，但：

- ESS 下降 18.88%；
- ESS/GPU-hour 从约 932 降到 218。

结论是：通用 TorchForce/MTS 路线没有生产收益，不继续通过调 MTS 间隔或 (c_1\) 参数寻找偶然通过点。

---

## 12. DEXP 路线

DEXP 使用双指数径向核代替或对照标准 LJ：

\[
U(r)=\epsilon\left[
\frac{\beta}{\alpha-\beta}e^{-\alpha x}
-\frac{\alpha}{\alpha-\beta}e^{-\beta x}
\right],
\quad x=\frac{r}{r_0}-1.
\]

默认实验参数 (alpha=14,\beta=5\) 不是普适物理常数，而是特定数据上的经验选择。

正面结果：

- Atenolol 单体系 kernel benchmark 中，DEXP projection 优于 LJ；
- force、torque、Hessian 和环境收敛检查通过。

限制与失败：

- 15 个 V/S/B 多初态 replicas 未充分平衡；
- 12,6 与 14,5 参数组都未通过 convergence；
- 旧“12,6 更接近”的结论被推翻；
- 尚无 8–15 个独立化学体系 benchmark；
- production merge proposal 未实现。

因此 DEXP 目前只能写成“有单体系物理信号的实验势”，不能写成生产 ABFE 优于 LJ。

---

## 13. Overlap、stationarity 与不确定度实验

### 13.1 EXP-016：temporal audit

状态：`INCONCLUSIVE / SURROGATE_ONLY`。

数据只有 3 × 500 frame surrogate trajectories，没有离散 alchemical state history、replica history 或真实 crossing。114 个 energy-weighted surrogate changes 不能证明真实 information timescale 或 MTS slow variable。

### 13.2 EXP-017：overlap-first

状态：`INCONCLUSIVE / STOPPED`。

| 指标 | 数值 |
|---|---:|
| min overlap | 0.3913 |
| min decorrelated | 96 |
| window 5 split-half drift | −0.5587 kJ/mol |
| drift / (2\sigma\) | 4.464 |

虽然 drift 有警告，但没有定位到单一低 overlap λ edge，因此没有理由插 λ，fixed-λ probe、P1/P2 都未启动。

### 13.3 EXP-018：stationarity confirmation

状态：`INCONCLUSIVE / CLOSED`。

- 3 seeds drift z-score：1.134、2.568、1.381；
- 只有 1/3 满足 qualifying negative drift；
- repeat variance ratio：16.7599；
- 追加 seeds 不符合预注册逻辑，因此关闭并要求另立不确定度实验。

### 13.4 EXP-019：baseline reproducibility

状态：`FAILED BEFORE FORMAL REPEATS`。

- v1：无可用 CUDA device；
- v2：5M pre-equil 后出现 `NameError system_type`；
- v3：wiring 修正，但 validate-only endpoint uncertainty 为 1.2481 kJ/mol，高于 1.0 门；
- completed baseline repeats：0；
- diagnostic rescue：159.3165 ± 2.0618 kJ/mol，仅用于诊断，绝不是 endpoint result。

该实验说明正式重复之前，Stage 2 baseline 本身还没有满足资格门。

---

## 14. EXP-020：局部多体残差的最强离线结果

### 14.1 科学假设

R1 grouped-density residual 不直接预测完整势能，而是：

1. 对 41 个 ligand anchors 收集 5 Å 内 ligand-environment edges；
2. 用 16 个 radial basis 和 atom-type weights 聚合局部 density；
3. 每个 anchor 先求邻居和，再经过非线性 readout；
4. 汇总成 bounded residual basis；
5. 由 Outer-lambda 系数 (A_k\) 应用到 IBS target energy。

多体性来自：

\[
\rho\left(\sum_j\phi(r_{ij})\right)
\ne\sum_j\rho(\phi(r_{ij})).
\]

### 14.2 离线资格数据

| 项目 | 结果 |
|---|---:|
| folds × seeds | 3 × 3 |
| checkpoints | 9/9 |
| mean fold-median gap-variance improvement | 55.5524% |
| D1 qualification | true |
| finite-difference absolute error | (4.5321\times10^{-10}\) reduced |
| finite-difference relative error | (2.6301\times10^{-9}\) |
| nonparticipant force | 0 |
| invariance maximum | (2.6021\times10^{-18}\) |
| D2 qualification | true |
| CPU64 reference/export | true |

这证明 pair-additive global representation 是此前的重要表达瓶颈之一，局部 anchor-wise nonlinear density 能恢复强 held-out signal。

### 14.3 部署成本数据

| 后端 | Parity | Cost ratio | 判决 |
|---|---|---:|---|
| N0 full-system CustomGB cost floor | 仅语义/成本 probe | 1.6965× | `STOP_FULL_SYSTEM_CUSTOMGB` |
| N1 per-anchor local CV | qualified | 6.0717× | `COST_FAILED` |
| N2 OpenMM-Torch local Verlet | qualified | 61.2922× | `COST_FAILED` |
| EXP-021 G1 grouped-density skeleton | skeleton | median 1.107419×; P95 upper 1.114105× | `STOP_EXP021_NATIVE_DENSITY` |

预注册上限为 1.10×。EXP-021 的 median 还超过更严格的 1.07 qualification 门，P95 upper 也超过 1.10，因此训练、G2、G4 均未授权。

结论不是“局部 density 没有科学价值”，而是：

> 表示层已经通过，通用 OpenMM/Torch 计算图没有通过真实 CUDA 成本门。

---

## 15. EXP-025：下一代专用 CUDA 路线

EXP-025 当前状态为 `DRAFT_DESIGN_ONLY / production_authorization=false`。它不覆盖 EXP-020/021 的停止结论，而是提出一个新的实现假设：

> 为冻结的 EXP-020 R1 写一个专用 OpenMM C++/CUDA Force，只扫描 41 个 ligand anchors 周围的局部 CSR/Verlet edges，在 GPU 内完成 density reduction、nonlinear readout 和 conservative force。

### 15.1 为什么可能更快

真实体系规模：

| 项目 | 数量 |
|---|---:|
| total atoms | 73,536 |
| ligand anchors | 41 |
| environment atoms | 73,495 |
| ligand × environment Cartesian pairs | 3,013,295 |
| D0 maximum active 5 Å edges | 1,464 |
| observed environment-ID union | 796 |
| model parameters | 3,031 |

R1 只有两遍局部 edge scan、41 个 scalar anchor reductions 和小型 typed MLP，不需要完整 MACE message passing，也不需要 triplets。

### 15.2 必须保持的数学合同

- 保留标准 bonded、LJ 和 PME；
- plugin 只输出 raw local basis；
- (A_k\) 和 offset 只能由 Outer-lambda/IBS 层应用一次；
- 端点 (A=0\) 时必须严格回到 baseline Hamiltonian；
- 力必须是能量的保守梯度；
- 动态 solvent membership 不能冻结，必须用带 skin 的局部 neighbor list；
- correctness 和 cost 两个门都通过前，不允许 production promotion。

### 15.3 建议的停止阶梯

```text
D0 frozen artifact/hash/geometry identity
 -> Reference oracle
 -> CUDA energy/force parity
 -> serialization/context update
 -> real 73,536-atom cost floor
 -> short NVT stability
 -> 3+ paired online repeats
 -> ESS/GPU-hour decision
```

如果专用 kernel 仍不能达到 `≤1.10×`，应停止当前局部 residual online 路线，保留其作为离线 teacher/diagnostic 工具。

---

## 16. ORB 路线

ORB 表示层的离线 LORO 结果较好：

- 3/3 folds 改善；
- fold improvement：28.1%、42.7%、48.3%；
- mean improvement：约 39.7%。

但 matched CUDA scalar 增量为：

| 指标 | 数值 |
|---|---:|
| measured increment | 77.622 ms |
| budget | 0.1–0.2 ms |
| ratio to upper budget | 约 388× |

因此 ORB-003 的结论是 `OFFLINE_TEACHER_ONLY`，ORB-004/005 停止。该结果再次说明：representation quality 与 production efficiency 是两个独立问题。

---

## 17. 带电配体、膜和 APBS

### 17.1 Charge-transfer handoff

带电配体路线通过 co-alchemical ion 保持总电荷守恒。Stage 2 开始前，代码会把 (lambda_{coul}=0\) 的全局参数化 NonbondedForce 烘焙成 fixed force，再交给 IBS vanishing。

当前状态：

- static handoff 已集成；
- CPU contract tests 和 MEM-00h 双侧归一化重处理通过；
- C3 v1 曾出现最大约 (7.47\times10^{-2}\) 的 force mismatch；
- v2 重处理达到约 (10^{-13}\)–(10^{-12}\) 级 energy/force 一致；
- 但 Atenolol 当前为 neutral branch，不触发真实 charged path；
- 尚无真实 charged ligand Stage 1→2 full cycle。

所以可以说“工程 seam 已实现”，不能说“带电配体生产路线已验证”。

### 17.2 膜体系

已有 100 ns NPT 质量门通过，Stage 0 Boresch NaN 问题也已修复。但完整 Stage 0→1→2 膜 ABFE 尚未完成。

旧膜结果 `+23.27 kcal/mol` 因 solvent-leg angles 被静默丢失、检测和 dispersion 问题而无效。

膜生产还需要：

- APL、膜厚、序参量和离子密度；
- restraint 几何稳定性；
- charged/dispersion 合同；
- 膜特定 LRC 或经过验证的替代方案；
- APBS maps 和 cycle sign；
- P1-19/P1-22 不确定度闭合。

### 17.3 APBS

APBS helper 用于 neutralizing-plasma/膜介电环境的外部静电有限尺寸修正。它在 complex/solvent 两腿完成后作为标量进入热力学循环，不改变 IBS 的 λ states。

当前候选 artifact 中 `APBS=0`，表示未启用，不是 APBS 已验证为零。真实膜 APBS 闭环仍是 `PENDING`。

---

## 18. 方法总览：到底有多少种方法

从 CLI 选项做笛卡尔积，可以得到约 576 个表面组合，但其中大量组合物理上不合法、只用于实验或必须 fail-closed，因此不能称为 576 种科学方法。

更合理的归纳是约 18 个方法家族：

| 层级 | 方法家族示例 |
|---|---|
| 输入/体系 | complex、solvent、membrane、charged/co-ion |
| 路径 | Stage 0 attachment、Stage 1 decharge、Stage 2 vanishing |
| 势函数 | ACE/Beutler softcore、experimental DEXP |
| 采样 | IBS dual-lambda、traditional single-lambda、REMD/bridge/shadow |
| 估计器 | adjacent BAR、FD-TI crosscheck、local/global TMBAR |
| 修正 | Boresch release、LJ LRC、APBS |
| 学习扩展 | MACE teacher、LocalResidualStudent、grouped density、ORB、Outer-lambda |
| 运行保障 | resume/cache、non-mutating extension、immutable rescue |

当前只有一条推荐主线：

```text
dual-lambda
 + Stage1 PME decharging
 + Stage2 ACE softcore IBS
 + v21 23-state path
 + v29 fixed-f production
 + Local TMBAR covariance-chain
 + Boresch
 + soluble-system LRC v3
 + non-mutating rescue
```

其余路线应被描述为 control、实验接口、研究分支或已停止路线。

---

## 19. 本项目目前可以主张的创新点

在论文或进度报告中，可以较稳妥地将创新点表述为：

1. **可审计的 dual-lambda ABFE pipeline**：将去电荷、vanishing、restraint、LRC、resume 和最终热力学循环纳入显式协议身份。
2. **IBS learning/freeze/production 分离**：在线学习只用于确定固定 (f_k\)，production 使用不可变 sampling distribution，便于 MBAR/TMBAR 解释。
3. **Stage 2 local TMBAR covariance-chain**：针对扩展系综 mixture 采样而设计的窗口内分析和跨窗口拼接。
4. **非突变故障恢复**：失败窗口不原地改变 Hamiltonian，通过同分布延长或独立 rescue 保留可审计性。
5. **多层资格门研究方法**：将离线信号、物理 correctness、CUDA parity、成本、动力学和 online ESS/GPU-hour 分开判决。
6. **局部多体 residual 的正负证据闭环**：EXP-020 表明科学表示有信号，同时真实 timing 否证了多种通用部署后端，形成专用 kernel 的明确下一步假设。

当前不应主张：

- 已得到最终可信的 Atenolol ABFE；
- 神经 residual 已接入 production；
- DEXP 普遍优于 LJ；
- charged ligand 或 membrane full cycle 已验证；
- APBS/LRC 对任意膜体系已经解决；
- v29 与所有旧协议 artifact 自动兼容。

---

## 20. 下一阶段优先级

### Priority 1：先让主 ABFE 结果具备科学可引用性

| 任务 | 交付物 | 成功标准 |
|---|---|---|
| 冻结环境和输入 | environment、config、hash、seed manifest | 可从零复现 |
| current estimator v2 重分析 | Stage 1 BAR 主值与 FD-TI gate | 分析口径与源码一致 |
| 独立重复 | 至少 2 个新增 seed，理想 3+ | 可估 between-run variance |
| moving-block/bootstrap | 时间相关 uncertainty report | P1-22 闭合 |
| Boresch sensitivity | unclipped/alternative anchors 对照 | restraint 结论稳定 |
| validation matrix | dated CPU/GPU evidence | v29/path21 联合通过 |

### Priority 2：决定局部多体 residual 是否值得在线化

只推进 EXP-025 的最低成本判决路径：

1. 冻结 EXP-020 R1 artifact 和数学合同；
2. 建立 independent Reference oracle；
3. CUDA kernel energy/force parity；
4. 真实 73,536 原子成本测试；
5. 若 `>1.10×`，立即停止；
6. 只有成本与 parity 同时通过，才做短 NVT 和 online repeats。

### Priority 3：扩展体系而不是继续单体系调参

- 选择 8–15 个不同化学类型 benchmark systems；
- 至少包含 neutral/charged、不同柔性和不同 binding-site environment；
- 在同一 protocol 下比较 accuracy、uncertainty、failure rate 和 GPU-hour；
- DEXP、residual 等方法必须在这一层才有资格讨论普适性。

### Priority 4：膜与带电配体

在 soluble baseline 和不确定度闭合后，再执行：

- charged fixture 的 Stage 1→2；
- membrane Stage 0→1→2；
- membrane dispersion/APBS evidence；
- 独立重复与 cycle closure。

---

## 21. 建议与导师讨论的决策问题

1. 当前论文主线是否应聚焦“可审计 IBS-ABFE 方法与失败边界”，而不是急于给出单个 Atenolol 最终数值？
2. 独立重复和跨体系 benchmark 应投入多少 GPU 预算？
3. EXP-025 专用 CUDA plugin 的工程投入，是否值得换取对 EXP-020 强离线信号的一次严格成本判决？
4. DEXP 是否保留为独立方法学小论文/补充材料，而不进入当前生产主线？
5. 膜和带电配体应作为当前论文 extension，还是后续独立工作？
6. 论文中负结果应放在 Results、Discussion 还是 Supplementary Methods？

建议的项目资源分配原则是：

> 首先闭合主线结果的重复性和不确定度，其次用一次严格、可停止的 EXP-025 cost test 决定神经残差是否继续，最后再扩展膜与更多化学体系。

---

## 22. 30–50 分钟讲述顺序建议

| 时间 | 内容 | 目标 |
|---:|---|---|
| 0–5 min | 研究问题与一句话状态 | 先说明“软件已成形，最终数值未闭合” |
| 5–12 min | 热力学循环与 pipeline | 解释 complex/solvent、Stage 0/1/2 |
| 12–20 min | ACE、IBS、λ path、TMBAR | 说明方法核心与创新 |
| 20–26 min | 候选 ABFE 与诊断数据 | 展示 −23.16 候选值及可信度边界 |
| 26–33 min | 统计不确定度和历史无效结果 | 说明为什么不能现在发表 |
| 33–42 min | 神经/DEXP/EXP-020 结果 | 展示正信号与成本失败 |
| 42–46 min | EXP-025、膜和带电路线 | 介绍下一代设计与扩展 |
| 46–50 min | 优先级与导师决策问题 | 获得资源和论文范围决策 |

如果只讲 30 分钟，可压缩 DEXP、ORB、膜和逐实验细节；如果讲 50 分钟，则保留全部数据表并重点讨论 EXP-020/025。

---

## 23. 可直接用于汇报结尾的总结

本项目已经从“能否运行 ABFE”推进到“如何使 Hamiltonian、采样分布、估计器、修正项和失败恢复都可追踪”的阶段。当前主线软件支持 GROMACS→OpenMM、complex/solvent 两腿、Boresch、Stage 1 PME decharging、Stage 2 ACE/IBS、TMBAR、LRC 和非突变 resume/rescue。

`−23.1622 ± 2.5139 kJ/mol` 是当前最完整的 Atenolol 候选基线，但没有独立重复，不能作为最终论文值。旧 `+40.8362` 和 `+16.00 kJ/mol` 已因符号、Boresch 或诊断问题失效。

新方法方面，通用 MACE/TorchForce/MTS、ORB 和多种 OpenMM 原生图均在稳定性或成本门失败；这些失败并非无效工作，而是把问题收敛到了一个更明确的结论：局部多体 residual 具有强离线信号，瓶颈是部署计算图。EXP-025 因此只测试一个高度专用的 CUDA kernel 假设，并保留严格停止规则。

下一阶段的第一优先级不是继续增加方法复杂度，而是通过独立重复、block bootstrap、current estimator 重分析和 v29 联合验证，使主 ABFE 结果具备科学可引用性。

---

## 24. 项目入口与材料分工

| 入口 | 用途 |
|---|---|
| [报告目录](README.md) | 当前五份综合报告和历史里程碑 |
| [运行脚本](../scripts/README.md) | PBS、实验、验证和批处理脚本；脚本存在不代表 production 授权 |
| [测试入口](../tests/README.md) | offline pytest、协议契约、resume、CPU/GPU 验证层级 |
| [诊断与修复工具](../tools/README.md) | diagnostics、repairs 和 plots；不属于生产模块 |
| [外部参考资料](../references/README.md) | 论文、外部实现、GROMACS 对照和引用边界 |
| [整理控制中心](../docs/curation/README.md) | 结果登记、冲突、文档职责、不可变政策和恢复点 |
| [整理版知识库](../curated_project/README.md) | 不含大型输出的阅读副本和当前状态导航 |

## 25. 主要证据入口

- [整理版当前状态](../curated_project/00_从这里开始/CURRENT_STATUS.md)
- [数字与科学结论](../curated_project/00_从这里开始/RESULTS.md)
- [结果登记表](../docs/curation/RESULT_REGISTRY.csv)
- [科学与文档冲突](../docs/curation/CONFLICTS.md)
- [软件进度与技术母稿](SOFTWARE_PROGRESS_AND_TECHNICAL_DRAFT_2026-08-12.md)
- [当前代码与新设计工作原理](CURRENT_CODE_AND_NEW_DESIGNS_WORKING_PRINCIPLES_2026-08-12.md)
- [Pipeline 与方法谱系](PIPELINE_AND_METHODS_LANDSCAPE_2026-08-12.md)
- [失败路线与证据](DEVELOPMENT_FAILURES_AND_EVIDENCE_2026-08-12.md)
- [EXP-025 专用 CUDA 设计](../curated_project/03_当前工作线/03_实验设计/PLAN_EXP-025_local_manybody_cuda.md)
- [Outer-lambda 实验事实日志](../EXPERIMENT_LOG_outer_lambda_neural_basis.md)

大型轨迹、checkpoint 和 JSON artifact 仍保留在原项目输出目录。未来任何结果状态晋级，都应同时更新 `docs/curation/RESULT_REGISTRY.csv`、整理版当前状态和对应 provenance。
