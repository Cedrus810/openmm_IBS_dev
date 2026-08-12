# EXP-021 GroupedDensityCV：成本优先的局部多体路径基函数设计

> 文档状态：`DRAFT_NOT_SEALED`
>
> 实验身份：`EXP-021`（预留；尚未创建或封存 preregistration）
>
> 方法代号：`GroupedDensityCV`
>
> 日期：2026-08-12
>
> 本文是一个独立的新设计。它不修改 EXP-012、EXP-019 或 EXP-020 的封存协议，也不把 EXP-020 的 D1/D2 成功自动解释为生产资格。

---

## 0. 执行摘要

EXP-020 已经证明了一件重要的科学事实：一个很小的局部 density/set 非线性模型能够显著改善 held-out gap variance。seed-fixed R1 在 3 folds × 3 seeds 上全部改善，平均 fold-median 改善为 **55.5524%**；D2 的有限差分、C² cutoff、triclinic PBC 和不变性检查也全部通过。

EXP-020 同时排除了三条部署路线：全体系 CustomGB cost floor 为 **1.6965×**；语义正确的 N1 per-anchor native CV 为 **6.0717×**；使用 2,575-edge 动态 Verlet pool 的 N2 Torch/OpenMM-Torch bridge 为 **61.2922×**。三者都超过预注册的 **1.10×** 上限。N1/N2 的 parity 均通过，所以失败原因不是能量/力语义错误，而是后端分解和 bridge 的实际成本。

EXP-021 不再从“怎样把 R1 塞进 OpenMM”出发，而从“OpenMM 能以低成本原生计算什么”出发：

1. 用 `CustomNonbondedForce.addInteractionGroup()` 只枚举固定配体组与动态环境之间的 cutoff pair；
2. 每个配体组产生一个低维密度 CV；
3. 用 `CustomCVForce` 在 pair 聚合之后施加非线性 readout，从而保留最小的多体表达能力；
4. **在任何模型训练之前**，先在真实体系上测 G1/G2/G4 的空骨架成本；
5. 只有成本门通过的最简单结构才允许训练；失败即停止，不事后堆复杂度。

目标不是复制 MACE，也不是忠实实现原始 DiffLift。目标是检验一个更窄、可证伪的问题：

> 在总开销不超过 baseline 1.10× 的前提下，少量 grouped density CV 能否保留 EXP-020 R1 的主要 gap-variance 收益，并通过 native OpenMM 能量/力等价性检查？

---

## 1. 决策边界与非目标

### 1.1 本实验要回答的问题

- `Q1 / cost`：interaction-group native skeleton 在真实 CUDA 体系上是否足够便宜？
- `Q2 / signal`：1、2 或最多 4 个预定义 ligand groups 是否能保持足够的 held-out gap-variance 改善？
- `Q3 / parity`：训练参考实现能否无歧义地转换为 native OpenMM，并保持能量与力一致？
- `Q4 / utility`：通过 D4 后，它是否真正提高 ESS/GPU-hour，而不只是降低离线 surrogate loss？

### 1.2 明确非目标

- 不再尝试全体系 `CustomGBForce`、masked all-pair scan 或其变体。
- 不重开 N1 的 41-anchor OpenMM decomposition，也不把其 parity qualification 改写成 backend qualification。
- 不重开 N2 Torch/OpenMM-Torch Verlet bridge；缩小 candidate pool 不能抵消已测得的 bridge 成本。
- 不在本实验中实现完整 MACE、environment–environment message passing、动态超图或原论文的 Bernoulli/STE lifting。
- 不把 5.5 Å 描述为 MACE 的绝对感受野；它是单层局部 cutoff，多层会形成多跳感受野。
- 不冻结训练轨迹中出现过的 796 个环境原子 ID；它们只是诊断统计，不是生产邻居表。
- 不把 EXP-020 checkpoint 直接上线；EXP-020 R1 只可作为离线 teacher/参照。
- 不修改已有 IBS 历史、sampling bias 历史、EXP-012/019/020 报告或封存 payload。
- 不因 D1/D2/D3 通过而自动恢复 WP-5 或批准 production。
- 不允许在同一 EXP-021 身份下加入 C++ plugin、CUDA 自定义 kernel 或新的 GNN；那是另一个实验。

---

## 2. 冻结前置证据

以下事实是 EXP-021 的输入证据，不是待重新拟合的自由参数。

### 2.1 系统与数据

| 项目 | 冻结值 |
|---|---:|
| 全体系原子数 | 73,536 |
| 配体原子数 | 41 |
| 环境原子数 | 73,495 |
| 全体系有序 pair 数 | 5,407,469,760 |
| ligand–environment Cartesian candidates | 3,013,295 |
| 训练帧数 | 1,500 |
| 活跃 edge records 总数 | 1,920,801 |
| 单帧最大活跃边数 | 1,464 |
| 单帧最大 unique environment atoms | 255 |
| 单 ligand atom 最大邻居数 | 55 |

数据集：`output/outer_lambda_exp020_softlift/dataset/softlift_dataset_v1.npz`

数据集 SHA-256：`24e5ce7ceb995b67ceddb08e5c7e5991c9a904356be743aecf55dc8d4ae260ab`

三条 run 中出现过的环境原子 ID 并集为 796；这只说明局部口袋具有重复结构，不说明可以静态冻结环境集合。溶剂、离子和边界原子仍可能交换。

### 2.2 EXP-020 科学资格

- D1 seed-fixed 报告：`output/outer_lambda_exp020_softlift_seedfixed001/r1_density/r1__direct_gap__d1_report.json`
- `qualification=true`
- 3/3 folds、9/9 seeds 改善
- 平均 fold-median improvement：`0.5555240525117041`
- R1 参数数：3,031
- 报告自哈希：`693871be4485bc2612b31ad96242cd2160eaece672b4d038e272d584e6c1e37c`

- D2 报告：`output/outer_lambda_exp020_softlift_seedfixed001/d2/r1_d2_report.json`
- `full_d2_qualification=true`
- observed：FD max abs `4.53205e-10`、max rel `2.63000e-9`；C² boundary error `0`；invariance max `2.60209e-18`；triclinic displacement error `3.9968e-15`；no-contact E/F 为 0
- 报告自哈希：`b09221bda06223e20f864d6695f6602e04fcdcf823f8f05dadcd79b22e250688`
- 限制：该证据覆盖 1 checkpoint、1 real frame 和 3 个 FD coordinates；多 checkpoint/frame、cutoff±ε、高 skew、half-box tie 与 support-boundary stress suite 尚未运行。EXP-021 的 §9 必须补齐，不能把 EXP-020 D2 当成 exhaustive proof。

### 2.3 EXP-020 部署边界

- D3 reference/export 报告：`output/outer_lambda_exp020_softlift_seedfixed001/d3/r1_reference_export_report.json`
- `reference_export_qualification=true`
- `full_d3_qualification=false`
- CPU64 的真实帧、synthetic triclinic、no-contact、save/load 均通过
- half-box tie 的 `FAILED_FAIL_CLOSED` 是单独的 pre-active-tie-fix 证据；不得误写成当前正式 report 已包含该失败

### 2.4 N0 终止结论

- N0 报告：`output/outer_lambda_exp020_softlift_seedfixed001/d3/r1_native_n0_cost_floor_report.json`
- CUDA baseline：`0.31607639975845814 ms/step`
- minimal CustomGB probe：`0.53621178958565 ms/step`
- ratio：`1.6964625957376656×`
- 上限：`1.10×`
- 结论：`STOP_FULL_SYSTEM_CUSTOMGB`

该 probe 已是成本下界。完整 R1 只会增加表达式、掩码和 nonlinear readout 成本。因此 EXP-021 禁止用“再优化一次 GB”规避此结论。表中的 5,407,469,760 是 ordered pairs；N0 report 的 logical unordered pairs 为 2,703,734,880，两者不得混用。

### 2.5 N1/N2 parity 成功、成本终止

报告：`output/outer_lambda_exp020_softlift_seedfixed001/d3/r1_n1_n2_local_report.json`

- schema：`exp020-softlift-n1-n2-local-v1`
- status：`COMPLETED_R1_N1_N2_LOCAL_PROTOTYPE_CHECK`
- canonical self-hash：`2856de64b213a283fc2cb21a3c02fe9471af5a2bc4c329663f1cb3240ce7a701`
- raw file SHA-256：`8e7094e64d7204ea05a75def01212327316bb22108a905d6e508eba50ad5be96`
- CUDA、真实体系、frame 0；历史 cost probe 为 5 warmup + 20 timed steps，属于路线淘汰证据，不替代 EXP-021 的严格 D0-COST protocol。

| 路线 | parity | 关键误差 | CUDA cost ratio | 冻结结论 |
|---|---|---|---:|---|
| N1 per-anchor local CV | qualified | energy `1.89705e-6 kJ/mol`；force `5.60070e-5 kJ/mol/nm` | `6.071711×` | `STOP_N1_COST` |
| N2 Torch/OpenMM-Torch Verlet | qualified | OpenMM-Torch energy `5.11555e-5 kJ/mol`；force `8.80640e-4 kJ/mol/nm` | `61.292160×` | `STOP_N2_COST` |

N2 使用动态 Verlet candidate pool：6 Å candidate radius（5 Å cutoff + 1 Å skin），2,575 edges、343 unique environment atoms、最大 82 candidates/anchor；在 skin/2 displacement 或 box change 时重建，未永久冻结 environment atom IDs。Python 与 scripted reference parity 通过，主要边界误差来自 float32 OpenMM-Torch bridge。

由此冻结：

- N1 证明 per-anchor native CV 语义可以正确实现，但现有 41-anchor OpenMM decomposition 成本远超 1.10×；
- N2 证明动态候选池与 Torch/OpenMM-Torch 能量/力链可以正确实现，但 bridge 成本更高；
- 两条路线只获得 parity qualification，不获得 backend qualification；
- 不允许通过减少诊断、冻结水原子、降低 rebuild 频率或忽略 bridge 成本来重新解释 cost failure；
- EXP-020 继续保持 `full_d3_qualification=false`、production blocked；
- EXP-021 的 grouped-CV 不是 N1/N2 的自动晋级，而是一个尚待 D0-COST 证伪的新后端假设。

### 2.6 EXP-021 G1 D0-COST 终止结论

报告：`output/outer_lambda_exp021_grouped_density_cv/d0_cost/g1_skeleton_cost_report.json`

- protocol：`SEALED`，payload SHA `aa471dec24992ee876c444acdd7a5d8da5d49890a97d39dc56a6fa7a2277d7a7`；canonical self-hash `ce638f3e1902dc5640d745248513281e78ff50bb062a7e9b1d28b125a94f9e05`；
- report status：`COMPLETED_D0_COST`；qualification：`false`；conclusion：`STOP_EXP021_NATIVE_DENSITY`；
- report canonical self-hash：`58853be0c3b7fe4b96b0dd261a549ef833ae4c335ce6683bef823e9ede4339ae`；raw file SHA/sidecar：`812d543e44af2ecfa905678a7ec793e5c2807229e43481facac43382443b80ff`；
- 3 个 sealed frames 覆盖 1,116/1,291/1,464 active edges；每 frame 5 paired repeats、100 warmup、1,000 timed steps、2 energy queries/context；真实 RTX 5080 CUDA single precision；
- total-protocol frame median ratios：`1.107419`、`1.109473`、`1.106176`；跨 frame median：`1.107419`，超过 `1.07`；
- 最坏 bootstrap 95% upper ratio：`1.114105`，超过 `1.10`；
- steady-integration ratios 也约为 `1.1088–1.1116`，因此失败不是由 energy-query 统计偶然造成；
- protocol/report/sidecar、六个引用 artifacts、dataset/system/topology identity 以及项目 validator 全部核验通过。

高 occupancy frame 并未比 low/middle frame 更慢；约 10.6–11.1% 的增量更符合“新增一个 CustomNonbondedForce + nested CustomCVForce pass 的固定后端开销”，而不是 edge 数爆炸。这是诊断解释，不改变 sealed gate。

根据封存 stop rule，G1 cost failure 立即终止 EXP-021 native density；G1 training 不获授权，G2/G4 不获授权，D1–D4 保持 `NOT_RUN/BLOCKED`。不得通过放宽 1.07/1.10、改 benchmark 聚合、删除 readout、减少实际 energy queries 或事后重跑挑选较快结果来改写失败。

---

## 3. 为什么不是 GB

`CustomGBForce` 的 ParticlePair computed value 在语义上对体系中所有 cutoff 邻居 pair 工作。`ligand_mask × environment_mask` 可以把无关 pair 的数值乘成零，但不能保证它们在 neighbor/pair kernel 之前被剔除。对于 73,536 原子体系，这与 D2 中约 1,464 条活跃 ligand–environment 边不是同一个成本问题。

`CustomNonbondedForce` 的 interaction group 才能直接表达“只计算集合 A 与集合 B 之间的 pair”。因此新的后端选择是：

```text
固定 ligand group × 全部 environment atoms
                 │
                 ▼  CutoffPeriodic + OpenMM neighbor list
            少量 pair-sum CV
                 │
                 ▼
        CustomCVForce 非线性组合
                 │
                 ▼
             一个共享 B(R)
```

这里“全部 environment atoms”是固定 topology 集合，但每一步进入 5 Å cutoff 的成员由 OpenMM 邻居表动态决定。它既不扫描 environment–environment 相互作用，也不冻结局部水分子的原子 ID。

---

## 4. 模型定义

### 4.1 固定候选与 ligand groups

设 ligand 原子集合为 `L`，全部非 ligand 原子为 `E`。只允许下列预定义层级：

| 层级 | groups | 含义 | 是否允许进入下一层 |
|---|---:|---|---|
| G1 | 1 | 全部 ligand atoms | 首先测试 |
| G2 | 2 | ligand H / ligand heavy | 仅 G1 成本通过但信号失败时 |
| G4 | ≤4 | ligand H/C/N/O；空组省略 | 仅 G2 成本通过但信号失败时 |

元素分组由 topology atomic numbers 决定并写入 sealed protocol。不得在观察 held-out 结果后重新聚类、拆组或按残基手工挑选。

环境元素 vocabulary 继承冻结数据：`[1, 6, 7, 8, 11, 16, 17]`。未知元素或 topology hash 改变时 fail closed，不能映射到“other”。

### 4.2 Pair density

对每个 group `m`，定义：

```math
q_m(R)=\frac{1}{|L_m|}\sum_{i\in L_m}\sum_{j\in E}
c_{C^2}(r_{ij})\sum_{p=1}^{P}w_{m,t_j,p}\,R_p(r_{ij}).
```

其中：

- `r_ij` 使用 periodic minimum image；
- `c_C²(r)` 在 4 Å 内为 1，在 4–5 Å 用 quintic taper，在 `r ≥ 5 Å` 精确为 0；
- pair membership 使用严格 `r < 5 Å`；cutoff 点处值、一阶导和二阶导均为 0；
- `R_p` 是固定中心和宽度的径向基函数；EXP-021 首个 protocol 固定 `P=16`，不得按 held-out 结果改变；
- `w` 是可训练系数；
- `1/|L_m|` 防止 group 大小改变数值尺度；空 group 不创建 Force/CV。

`q_m` 是数值特征，不声称具有物理能量含义。参考实现将其视为无量纲。native OpenMM 中 inner force 必须返回能量标量，因此规定：

```math
U_{q_m}=Q_0 q_m,\qquad Q_0=1\ \mathrm{kJ\ mol^{-1}}.
```

外层表达式按 `q_m = U_qm / Q0` 解释。`Q0` 只用于单位桥接，必须记录且只应用一次。

### 4.3 聚合后非线性

```math
s(R)=\sum_m \left[\rho_m(\bar q_m(R))-\rho_m(0)\right],
```

```math
B(R)=B_{\max}\tanh\left(\frac{s(R)-b_0}{B_{\max}}\right)-B_{\mathrm{nc}}.
```

- `\bar q_m=q_m/Q_m`；`Q_m` 只从该 fold 的训练部分估计并冻结，不接触 validation/held-out run；
- `rho_m` 是固定宽度 `H=8` 的解析一维 readout；训练与 native 使用同一公式：

```math
\rho_m(x)=\sum_{h=1}^{8}v_{mh}\tanh(a_{mh}x+c_{mh}).
```

  export 只把 `a/c/v` 常数写入 OpenMM expression，不进行 spline/table 二次拟合；
- `rho_m(0)` 显式相减；
- `B_nc` 取无接触输入的完整 bounded 输出，使 `B(R)=0` 在所有 ligand–environment pair 都超出 cutoff 时精确成立；
- `b0` 是 gauge/中心化参数，只能由 training split 决定并记录；不得用 held-out 数据调整；
- `Bmax` 使用 reduced、dimensionless 单位并在 protocol 中固定。

非线性发生在 pair sum 之后，因此该模型不是简单 pair-additive potential；但它只保留 1–4 个全局 ligand-group 密度，表达能力显著窄于 EXP-020 的 per-ligand-anchor R1，也远窄于 MACE。

### 4.4 Outer-λ 接线与单位

唯一允许的约定是：

```math
U_B(R)=k_B T\,B(R)\quad [\mathrm{kJ\ mol^{-1}}],
```

```math
H_k^*(R)=H_k^{MM}(R)+A_k\,[U_B(R)-U_{offset}].
```

- 训练 loss 中 `B` 是 reduced dimensionless；ledger 已经应用 beta，loss API 不接受第二个 beta；
- exported basis artifact 固定 `a_k=1.0`，只返回 raw `U_B`；
- `A_k` 仅由现有 OuterLambdaController 应用一次；
- `tanh`、`kBT`、offset 和 `A_k` 各只应用一次；
- `A_k` 使用 sealed global sin² schedule，不能把局部 ledger slice 的两端误当成物理 endpoints；
- target energy history 与 sampling bias history 继续分离，EXP-021 不重写历史 ledger。

---

## 5. Native OpenMM 计算图

### 5.1 每个 group 的 inner force

每个非空 group 创建一个 `CustomNonbondedForce`：

- `addInteractionGroup(L_m, E)`；
- `CutoffPeriodic`，cutoff = 0.5 nm；
- 对全体系每个原子调用一次 `addParticle()`；每个 child force 恰好只有一个 interaction group；
- `L_m` 已排序、两两不交；`E=all_particles−L`，因此 `L_m∩E=∅`，不把 ligand 混入 environment；
- OpenMM 不保证一对粒子中谁是 particle1/particle2。pair expression 必须显式交换对称：

```text
Q0/|L_m| * C2(r) *
[gm1*env2*f(type2,r) + gm2*env1*f(type1,r)]
```

  其中 `gm/env` 是 0/1 per-particle parameters；禁止单独依赖 `type1` 或 `type2` 的非对称公式；
- `CustomNonbondedForce` 不自动继承主 `NonbondedForce` 的 exceptions。本 CV 的冻结语义是包含全部 ligand–environment pair，不添加 exclusions；若未来要排除 1–2/1–3/1–4，必须成为新 protocol；
- 不启用 switching function 或 long-range correction；C² kernel 自己保证 5 Å 处值和前两阶导归零；
- 不创建 environment–environment 或 ligand–ligand interaction group；
- tiny periodic pair enumerator 必须 100% 对齐 pair 数、PBC、交换角色、重复计数和 exclusion 数；任何 orientation 差异直接 block。

### 5.2 Outer readout

一个 `CustomCVForce` 接收 `U_q1...U_qM`：

- 将各 CV 除以 `Q0×Qm`；
- 直接展开冻结的每组 8 个解析 tanh readout；
- 执行中心化、全局 tanh 和 `kBT` 转换；
- 输出 raw `U_B`，不在内部乘 `A_k`。

解析 readout contract：

- reference 与 native 逐项使用同一个 `H=8` tanh 公式，不允许二次拟合或近似转换；
- `Qm`、`a/c/v`、`Bmax`、`b0`、`Bnc` 全部进入 artifact SHA；
- `q_m/Q_m` 不做硬裁剪；资格帧报告训练范围、部署范围、绝对最大值和 tanh saturation fraction；
- 解析式在全部实数上有定义，因此不存在 table-domain 外推。超过预注册的数值支持门仍是 safety failure；
- Reference 与 CUDA 必须验证能量、力、`q_m` 和 global-parameter derivative。

### 5.3 必须由 prototype 证明、不得先验宣称的事项

- 多个 interaction groups 是否各自触发独立 pair pass，以及 G2/G4 的实际增量成本；
- interaction group 对重叠集合、pair 计数、角色顺序和 exclusions 的实际语义；
- `CustomCVForce` 拥有 child force；child 不得再加入 System。wrapper force group 是外部唯一可见组，inner child groups 不参与外部 group mask；
- XML 只保存 global parameter defaults，不保存运行中 Context 参数。restart 必须从 checkpoint/metadata 恢复全部非默认参数，再做 `q/E/F/dE/dp` probe；
- child per-particle 参数更新必须经 `getInnerContext(context)` 后调用 child 的 `updateParametersInContext`；group、expression、cutoff 改变必须重建 Context；
- 同名 global parameters 的 default 必须唯一一致；需总导数的参数在 outer CV 注册并以中心差分审计完整 chain；
- triclinic PBC、每步 box/parameter 向 inner Context 同步和 cutoff 的平台一致性；
- Reference/CUDA 的 `q_m`、能量、力和 global-parameter derivatives 等价性；
- G1/G2/G4 远低于 CustomCVForce 的 32-CV 上限，但 schema 仍固定 `M≤4`；
- 与已有 outer-λ force group、IBS energy collection 和 checkpoint/resume 的兼容性。

---

## 6. 成本优先资格门 D0-COST

这是 EXP-021 与旧设计最重要的区别：**先测部署骨架，再训练。**

N0/N1/N2 并没有直接测量 G1：N1 是 41 个 per-anchor CV，G1 是单个全配体 grouped CV；N2 则包含 TorchForce bridge。因而 G1 仍是一个逻辑上未被直接否定的假设，但不能从 N1 的 6.0717× 按 `1/41` 线性外推，也不能预设它会通过。唯一有效证据是下面定义的真实体系 D0-COST。

### 6.1 基准条件

必须使用与 N0 一致或更严格的真实生产条件：

- 真实 73,536 原子 System、topology 和代表性 trajectory frames；
- CUDA 平台；
- 与 baseline 完全相同的 precision、integrator、constraints、PME/cutoff、force groups 和 reporting；
- baseline 与 candidate 交错测量，避免温度/频率漂移；
- 每个 Context 至少 100 warmup steps；每个 paired repeat 至少 1,000 timed steps；至少 5 个 baseline/candidate 交错 repeat pairs；
- 训练前冻结 benchmark frame/box manifest 及 SHA，覆盖低/中/高 pair occupancy，不得选择便宜帧；
- 同时报告 paired ratios、median、bootstrap 95% upper confidence bound、MAD、GPU memory/OOM、GPU 名称、clock policy、driver、CUDA/OpenMM 版本；
- steady-state integration 计时环禁止 reporter/`getState`；另按冻结的真实 IBS energy-query 频率测 total protocol cost。两者都报告，资格使用 total protocol cost；
- N0 的 5 warmup × 20 timed × 3 repeats 只是历史 cost-floor 证据，不声称复现本节更严格的 EXP-021 protocol。

### 6.2 Skeleton 定义

Skeleton 必须包含最终计算图的结构成本：

- 实际数量的 `CustomNonbondedForce` interaction groups；
- 16 个 radial terms 和 7 种 environment types 的表达式形态；
- 实际 `CustomCVForce`、每组 8 个解析 tanh 和全局 tanh；
- 参数可用固定非零值，但不能把表达式简化成常数或零从而被编译器消除；
- 不需要训练权重，但必须与最终 Force 数量、解析项数和 cutoff 一致。

### 6.3 预留与停止规则

| gate | 条件 | 结论 |
|---|---|---|
| G1 skeleton | median total ratio ≤1.07 且 95% upper bound ≤1.10 | 才允许训练 G1 |
| G1 skeleton fail | median >1.07 或 95% upper bound >1.10 | `STOP_EXP021_NATIVE_DENSITY`；不测试 G2/G4 |
| G2 skeleton | 仅完整、integrity-valid 的 G1 signal FAIL 后；同样 ≤1.07/1.10 | 才允许训练 G2 |
| G4 skeleton | 仅完整、integrity-valid 的 G2 signal FAIL 后；同样 ≤1.07/1.10 | 才允许训练 G4 |
| final model | 完整权重/解析 readout/outer wiring 的 95% upper bound ≤1.10 | 才能 full D3 qualified |

`1.07` 是骨架门，给最终参数、outer wiring 和运行波动保留约 3 个百分点。不得只报告相对 CustomGB 的加速；唯一成本分母是同条件的无 EXP-021 baseline。

---

## 7. 训练与模型选择

### 7.1 数据拆分

- 使用同一冻结 1,500-frame 数据集和 exact frame-to-ledger alignment；
- exact 3-fold leave-one-run-out；
- 每 fold 两条 train runs，第三条 run 全程 held out；
- training runs 的尾部 20% 作为 validation；
- seeds 固定 `[0,1,2]`；
- seed 必须在模型构造、参数初始化和 optimizer 构造之前设置；
- 禁止 random-frame split、held-out normalization、held-out early stopping 或跨 fold teacher fit。

### 7.2 目标函数

主目标复用已审计的 reduced gap contract：

```math
g'_{f,k}=g_{f,k}+\Delta A_k B(R_f),
```

```math
L_{gap}=\frac{1}{2K}\sum_k
\left[\operatorname{Var}_{w^L_k}(g'_{:,k})+
      \operatorname{Var}_{w^R_k}(g'_{:,k})\right].
```

- `adjacent_gap_reduced`、`basis_reduced` 和 `delta_A` 都是 reduced quantities；不再乘 beta；
- importance weights 在训练中使用冻结 ledger 权重，并明确称为 frozen-importance surrogate；
- 权重按现有 `logsumexp` partition contract 归一化，不 clipping；
- held-out 报告除 surrogate 指标外，必须用加入 candidate 后的完整 target weights 重算诊断，不能把 surrogate 称为 self-consistent variance；
- energy/force regularization coefficient、reduction 和 dtype 必须在 preregistration 中给出精确数值。

### 7.3 训练超参数

第一版 preregistration 固定：

- radial basis count `P=16`；
- Adam，learning rate `1e-3`；
- max epochs `500`；
- patience `30`；
- checkpoint 选择只看 training-fold validation objective；
- float64 为资格参考；float32 只作为部署候选，不能改变参考结果；
- 每个 checkpoint 记录 seed wiring、split hash、初始化 hash、epoch 和 optimizer state hash。

### 7.4 可选 teacher 项

直接 gap-only 是主分析。只有在 protocol 预先声明的次分析中，才可使用冻结 EXP-020 seed-fixed R1 作为 teacher：

```math
L=L_{gap}+\lambda_T L_{teacher}+\lambda_E L_E+\lambda_F L_F.
```

- teacher 输出必须预计算并哈希；teacher 不在线运行；
- 投影、归一化和 `lambda_T` 只用 training split 决定；
- direct 与 distilled 使用完全相同的 GroupedDensityCV 架构、参数预算和 seeds；
- distilled 不能替代 direct primary result，也不能在 direct fail 后作为未预注册的救援路线。

---

## 8. D1 离线信号门

对 untouched LORO test 定义：

```math
\Delta_{f,s}=\frac{V^{base}_{f,s}-V^{candidate}_{f,s}}{V^{base}_{f,s}},
\qquad m_f=\operatorname{median}_{s\in\{0,1,2\}}\Delta_{f,s}.
```

`Delta>0` 表示改善；比较使用未四舍五入 float64，`|Delta|≤10^{-12}` 记为 tie、不计改善。每个允许训练的 group level 使用同一判据：

1. 3/3 held-out folds 的 `m_f>0`；
2. 每 fold 至少 2/3 seeds 的 `Delta_f,s>0`；允许第三个 seed 失败，但必须完整报告；
3. 所有 fold×seed 的 9 checkpoints 必须存在、SHA-256 匹配；不同训练单元 checkpoint 不要求数学上必然不同，但重复 hash 必须审计 seed wiring 后才能资格；
4. `mean_f(m_f)≥0.45`；
5. 相对 EXP-020 R1 的 `0.555524`，上述 mean 的绝对降幅 `≤0.105524`；
6. 无 nonfinite、support、shape、hash、data-leakage 或 technical failure。

只有完整 9-run、integrity-valid 的 quantitative signal FAIL 才可进入下一层。缺文件、NaN、seed wiring 或基础设施失败一律为 `UNKNOWN_BLOCKED`，不能当成 G1/G2 信号失败。paired bootstrap CI 作为描述性不确定性报告，不追加事后 significance gate。

晋级顺序严格为：

```text
G1 cost pass → G1 train
  ├─ D1 pass → 停止搜索，进入 D2
  └─ D1 fail → 若 G2 cost skeleton pass，训练 G2
                 ├─ D1 pass → 停止搜索，进入 D2
                 └─ D1 fail → 若 G4 cost skeleton pass，训练 G4
                                ├─ D1 pass → 进入 D2
                                └─ D1 fail → STOP_NO_COMPACT_SIGNAL
```

不允许在 G1 已通过后继续训练 G2/G4 追求更漂亮的数字。这样避免基于 held-out 结果进行无上限的架构搜索。

---

## 9. D2 数值与物理检查

D1 只证明统计信号，不证明可用作势能。被选中的唯一 group level 必须通过：

- float64 autograd 对中心有限差分：能量绝对误差 `≤1e-5` reduced，力相对误差 `≤1e-3`；
- inner/outer cutoff 两侧的能量、力及二阶连续性；
- exact cutoff membership、half-box tie 和 `r≈0` 的 fail-closed 行为；
- reduced triclinic box、OpenMM 合法 box 和高倾斜反例；
- 整体平移、旋转和 atom permutation 不变性；
- no-contact energy/force 精确为零；
- 非参与原子零力；参与 pair 满足作用–反作用与总力闭合；
- 水/离子跨 cutoff 进入和离开的动态 membership；
- G2/G4 空组、单原子组和不同 group size normalization；
- tanh 饱和、`q_m/Q_m` 支持门和 nonfinite fail closed；
- 代表性正常帧、cutoff 边界帧、近接触帧和 synthetic triclinic 帧。

D2 参考实现不得直接调用不可 script/native 的 Python 动态 funnel 后就宣称 deployment equivalence。D2 只给数学参考资格，native 资格属于 D3。

---

## 10. D3 Native 等价性与最终成本

完整 D3 必须同时满足三部分；任何一个缺失都保持 `full_d3_qualification=false`。

### 10.1 D3-A reference/export preflight

- reference module save/load/rerun；
- CPU64 真实帧、triclinic、no-contact 能量/力；
- CPU32/CUDA32 仅在无 half-box ambiguity 的帧上比较；检测到 ambiguity 必须 fail closed 并另建不含 tie 的资格集，不得修改原始失败记录；
- artifact/config/checkpoint/expression/XML/report 的 raw-file SHA 与 canonical self-hash。

### 10.2 D3-B native parity

比较同一模型的 reference 与 native OpenMM：

- OpenMM Reference platform 与目标 CUDA platform；
- 真实轨迹帧 + D2 synthetic frames；
- energy 最大绝对误差 `≤1e-4 kJ/mol`；
- force 最大绝对误差 `≤1e-3 kJ/mol/nm`；
- cutoff crossing 的力曲线、解析 tanh chain 和所有已注册 global-parameter derivatives；
- XML serialize → deserialize → new Context → rerun；
- box vector 更新、global parameter 更新和 checkpoint/resume；
- inner CV、outer tanh、`kBT`、offset、`A_k` 各只应用一次；
- endpoint `A=0` 时 candidate 对总能量与力贡献精确为零。

### 10.3 D3-C final cost

- 使用完整训练参数、解析 readout、outer-λ/IBS 实际调用频率；
- 与 D0-COST 相同的交错 benchmark；
- total protocol cost 的 paired-ratio 95% upper confidence bound `≤1.10`；
- 同时报告绝对 ms/step 和 ESS 计算所需的额外 energy-query cost；
- 若失败，结论为 `STOP_NATIVE_COST`，不得在当前 protocol 内减少表、groups 或检查频率后重测。

只有 A+B+C 全部通过，才可写：

```json
{
  "full_d3_qualification": true,
  "production_promotion": false
}
```

---

## 11. D4 短程动力学与在线判据

D4 仅在 full D3 true 后开始。第一版 EXP-021 限定固定盒 NVT；`q_m` 是未除体积的局部 pair density。NPT、volume normalization 或 barostat 下的动态体积语义不在本 protocol 内。

### 11.1 D4 short NVT

- 至少 3 个独立 paired repeats；
- baseline 与 candidate 使用相同起点配对、不同预注册随机 seed；
- 报告温度、势能、constraint error、最大力、basis/`q_m` 分布、tanh saturation fraction 和数值支持 margin；
- 任一 nonfinite、support violation、resume hash mismatch 或能量爆炸立即终止该 run；
- 禁止在帧内静默禁用 candidate 或回退 baseline。

### 11.2 研究性在线试验

D4 通过仍不等于 production。下一决策需要新的授权和记录，主要指标是：

```math
\text{utility}=\frac{\mathrm{ESS}}{\mathrm{GPU\ hour}}.
```

必须预注册：

- 至少 3 个独立 paired runs；
- primary ESS estimator 和 uncertainty；
- target-state weights 的完整重算；
- cycle closure / complex–solvent 一致性；
- 最低实际提升阈值；
- 不允许用 λ 重排、删除难状态或不同 sampling length 替代方法收益。

只有 ESS/GPU-hour 相对 baseline 改善且安全检查通过，才能另作 production promotion 决策。

---

## 12. Fail-closed 与运行时策略

以下事件全部是硬失败，不是“自动回退”条件：

- topology/system/protocol/model/expression/XML SHA 不匹配；
- 未知元素、空 environment、非法 box、近奇异 box 或不满足 periodic cutoff 几何约束；
- half-box tie、`r≈0` 低于支持门、nonfinite energy/force；
- `|q_m/Q_m|` 或 saturation fraction 超过预注册数值支持门；
- tanh saturation fraction 超出 protocol 上限；
- force/energy safety threshold 超限；
- OpenMM platform、precision 或 force-group wiring 与 artifact 不符；
- checkpoint/resume fingerprint 不匹配。

生产前研究 run 可以在 run 启动时显式选择 `baseline` 或 `candidate`。一旦 run 开始，禁止按帧切换。candidate 失败后只能终止该 run；若要从 baseline 重启，必须使用新的 run ID、清洁初态和独立 provenance。

---

## 13. 参数与计算预算

第一版上限：

| 项目 | G1 | G2 | G4 |
|---|---:|---:|---:|
| interaction-group pair forces | 1 | 2 | ≤4 |
| radial bases/group | 16 | 16 | 16 |
| env types | 7 | 7 | 7 |
| pair weights 上限 | 112 | 224 | 448 |
| total trainable params | ≤2,000 | ≤4,000 | ≤8,000 |
| analytic tanh units/group | 8 | 8 | 8 |
| final cost cap | 1.10× | 1.10× | 1.10× |

这些是上限，不是必须用满。任何扩大 P、增加 ligand groups、引入 pair gate、角矩或 env–env block 的提案都必须成为新实验，不能在 EXP-021 失败后现场添加。

---

## 14. 产物、命名与 provenance

本文件只建立设计，不创建或封存以下未来产物。

```text
protocols/EXP-021_preregistration.json
output/outer_lambda_exp021_grouped_density_cv/
  d0_cost/
    g1_skeleton_cost_report.json
    g2_skeleton_cost_report.json
    g4_skeleton_cost_report.json
  g1/direct_gap/fold_<run>/seed_<seed>/
  g2/direct_gap/fold_<run>/seed_<seed>/
  g4/direct_gap/fold_<run>/seed_<seed>/
  d1/<selected_level>_d1_report.json
  d2/<selected_level>_d2_report.json
  d3/reference_export_report.json
  d3/native_parity_report.json
  d3/final_cost_report.json
  d4/short_nvt_report.json
  decision_log.md
```

每个 report 至少包含：

- schema name/version、experiment ID、stage、status、qualification booleans；
- protocol payload SHA、parent protocol SHA、dataset SHA；
- topology/system/trajectory/frame-index/box hashes；
- source file SHA 和 git commit/dirty state；
- Python/PyTorch/OpenMM/CUDA/driver/GPU/OS 版本；
- dtype、platform、precision、units、cutoffs、radial spec、group membership；
- split/seed wiring、checkpoint/expression/XML SHA；
- `raw_file_sha256` 与排除 self 字段计算的 `canonical_self_sha256` 分开命名；
- exact commands、start/end timestamps、failure reason；
- canonical self-hash，计算时排除 self-hash 字段本身。

封存规则：

- 新 protocol 必须有独立 schema/version 和 immutable payload；
- 封存后任何模型层级、阈值、benchmark 方法或输入 hash 变化都需要 amendment/new experiment ID；
- 失败报告不可覆盖；复跑使用新的 run suffix 并引用原失败；
- `qualification=true` 只能由脚本根据结构化 checks 计算，不能手工编辑；
- gate ledger append-only；任何阈值、代码、数据、fold、seed、hardware 或 benchmark manifest 变化都需 amendment/new ID，并全量重跑，旧结果不可覆盖。

---

## 15. 建议代码边界

```text
local_residual/grouped_density_cv.py
  - group definition validation
  - radial/C2 reference math
  - grouped density + centered bounded readout

local_residual/grouped_density_openmm.py
  - CustomNonbondedForce interaction groups
  - CustomCVForce/analytic readout construction
  - XML/artifact validation

scripts/benchmark_exp021_grouped_density_cost_floor.py
scripts/train_exp021_grouped_density_loro.py
scripts/check_exp021_grouped_density_d2.py
scripts/check_exp021_grouped_density_d3.py
scripts/check_exp021_grouped_density_d4.py

tests/test_exp021_grouped_density_reference.py
tests/test_exp021_grouped_density_interaction_groups.py
tests/test_exp021_grouped_density_loss.py
tests/test_exp021_grouped_density_openmm.py
tests/test_exp021_grouped_density_provenance.py
```

实现顺序必须是：schema/validator → D0 cost skeleton → 训练参考 → D2 → native conversion → D3 parity/cost → D4。不要先写完整训练管线再发现后端成本失败。

---

## 16. 端到端实施清单

### Phase A：冻结设计与 D0-COST

- [ ] 创建并审查 EXP-021 schema；
- [ ] 在观察任何 G1 结果前固定 G1/G2/G4 membership、radial/analytic-readout spec、阈值和全部 hashes；
- [ ] 构建 two-particle interaction-group 语义测试；
- [ ] 构建真实体系 G1 skeleton；
- [ ] 按交错 benchmark 运行成本门；
- [ ] G1 cost fail 时立即封存 `STOP_EXP021_NATIVE_DENSITY`。

### Phase B：仅训练成本通过的最简单层级

- [ ] 构建 reference GroupedDensityCV；
- [ ] 复用冻结 dataset/ledger loss；
- [ ] 检查 seed 在 model construction 前设置；
- [ ] 运行 3 folds × 3 seeds；
- [ ] 验证 9 checkpoint SHA/uniqueness 和 report self-hash；
- [ ] 严格按 G1→G2→G4 stop ladder 决策。

### Phase C：D2 与 native D3

- [ ] 完成全部数值/PBC/no-contact/support 检查；
- [ ] 将同一解析 readout 系数无近似地写入 native expression；
- [ ] 验证 Reference/CUDA native energy-force parity；
- [ ] 验证 XML/new Context/resume；
- [ ] 运行完整最终成本门；
- [ ] A/B/C 全部通过后才设置 full D3 true。

### Phase D：D4 与后续决策

- [ ] 运行 3 paired short NVT repeats；
- [ ] 审计 support、saturation、temperature、constraint、force tails；
- [ ] 另行批准 research online pilot；
- [ ] 用 ESS/GPU-hour 作 primary utility；
- [ ] production promotion 保持独立决策。

---

## 17. 预期失败方式及解释

### 17.1 G1 成本失败

说明即使 interaction-group pruning 也无法在 10% 预算内加入一条 native grouped density 路径。结论是当前 OpenMM force decomposition 不适合该预算，不说明 EXP-020 的科学信号不存在。停止 EXP-021，不测试更昂贵的 G2/G4。

### 17.2 G1 成本通过、信号失败

说明全局 ligand pooling 丢失了 per-anchor 分辨率。只有预注册的 G2、随后 G4 可以继续；它们仍须先过各自成本 skeleton。

### 17.3 G4 仍无信号

结论是“在 ≤4 个固定 grouped density CV 和 ≤10% 成本预算下，无法保留 R1 的主要收益”。保留 EXP-020 作为离线结果，停止；不在当前实验加入 learned groups、angles 或 message passing。

### 17.4 离线通过、native parity 失败

说明 analytic expression、interaction-group 语义或单位/wiring 不等价。保留 reference model 作为研究诊断，禁止 MD。

### 17.5 D3 通过、ESS/GPU-hour 不改善

说明 gap-variance surrogate 的收益没有转化为端到端采样效率。该结果本身是有效负结果，不能用离线 55.55% 改善覆盖。

---

## 18. 最终决策矩阵

| 阶段 | 必须通过 | 失败动作 | 通过后权限 |
|---|---|---|---|
| D0-COST | 真实 CUDA skeleton median≤1.07 且 95% upper≤1.10 | 停止对应及更复杂层级 | 允许训练该层级 |
| D1 | whole-run LORO、9 checkpoints、mean ≥45% | 按固定 ladder 或停止 | 允许 D2 |
| D2 | FD/C²/PBC/invariance/no-contact/support | 停止 native | 允许 D3 native |
| D3-A | reference/export preflight | 修复后新报告，不覆盖 | 不单独授予 native 资格 |
| D3-B | Reference+CUDA native energy/force parity | 停止 native | 等待成本门 |
| D3-C | 完整模型 paired-ratio 95% upper≤1.10 | `STOP_NATIVE_COST` | `full_d3_qualification=true` |
| D4 | 3 paired NVT、安全与支持域 | 停止 online | 可申请 research pilot |
| Online | ESS/GPU-hour 与闭环改善 | 不 promotion | 独立 production 决策 |

---

## 19. 当前状态

截至 G1 D0-COST 报告完成时：

```text
EXP-020 D1                    QUALIFIED
EXP-020 D2                    QUALIFIED
EXP-020 D3 reference/export   PREFLIGHT_QUALIFIED
EXP-020 N1 parity / cost      QUALIFIED / FAILED_6.0717X
EXP-020 N2 parity / cost      QUALIFIED / FAILED_61.2922X
EXP-020 full D3               FALSE
full-system CustomGB          STOPPED_1.6965X

EXP-021 design                HISTORICAL_DESIGN
EXP-021 protocol              SEALED_REVISION_001
EXP-021 G1 D0-COST            FAILED_1.107419_MEDIAN_1.114105_UPPER
EXP-021 conclusion            STOP_EXP021_NATIVE_DENSITY
EXP-021 G1 training           NOT_AUTHORIZED
EXP-021 G2/G4                 NOT_AUTHORIZED
EXP-021 D1-D4                 NOT_RUN_BLOCKED
EXP-021 production promotion  PROHIBITED
```

EXP-021 已达到预注册的停止条件；在该 experiment ID 下没有下一项训练或部署工作。合理的收尾动作只有封存 decision log、保留全部失败产物，并把结论写入实验注册表。

如要研究另一个后端，必须提出新的科学/工程假设并使用新的 experiment ID；不能把它称为 EXP-021 的优化复跑。
