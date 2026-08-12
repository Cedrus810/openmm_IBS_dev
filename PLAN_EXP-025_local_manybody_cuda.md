# EXP-025：Local Many-Body Residual CUDA Plugin

> document_version: `0.1-draft`  
> status: `DRAFT_DESIGN_ONLY`  
> experiment_id: `EXP-025`  
> method_name: `LocalManyBodyResidualCUDA`  
> production_authorization: `false`  
> supersedes: `none`  
> preserves: `EXP-012 / EXP-020 / EXP-021 sealed artifacts and decisions`  
> primary_system: `Atenolol-rank11, 73,536 atoms`  
> target_backend: `OpenMM CUDA, single GPU`  
> target_model: `EXP-020 R1 density residual, 3,031 parameters`

---

## 0. 一句话结论

EXP-025 不再尝试用 `CustomGBForce`、多个 `CustomCVForce`、通用
`CustomNonbondedForce` 或 OpenMM-Torch 去拼装 R1。

本实验只验证一条新路线：

> 把 EXP-020 已经通过 D1/D2 的局部多体残差，写成一个专用 OpenMM CUDA
> Force；只扫描 41 个配体 anchor 周围的局部 CSR/Verlet 边，在 GPU 内完成
> `pair density -> anchor nonlinear readout -> conservative force`，不进行全体系
> pair scan，也不创建 per-anchor CV。

该路线不是 Tersoff，也不是用双体势假装多体势。它保留标准生物力场：

\[
U_{\mathrm{total}}
=U_{\mathrm{bonded}}+U_{\mathrm{LJ}}+U_{\mathrm{PME}}
+A_k\,[U_B(R)-U_{\mathrm{offset}}].
\]

其中 CUDA plugin 只负责输出 raw basis

\[
U_B(R)=k_B T\,B(R),
\]

现有 Outer-lambda/IBS 层负责应用一次且仅一次的 \(A_k\) 和 offset。

---

## 1. 立项依据与历史边界

### 1.1 已冻结的正面证据

EXP-020 seed-fixed R1 已给出可重复的离线科学信号：

- 3 folds x 3 seeds，共 9/9 checkpoints；
- `seed_wiring=FIXED_BEFORE_MODEL_CONSTRUCTION`；
- D1 qualification 为 true；
- 平均 fold-median gap-variance improvement 为 `55.5524%`；
- 9 个 checkpoint SHA-256 匹配且互不相同；
- D2 offline reference qualification 为 true：finite difference、C2 cutoff、
  triclinic PBC、旋转/平移不变性、no-contact zero、非参与原子零力均通过。

冻结证据：

- `output/outer_lambda_exp020_softlift_seedfixed001/r1_density/r1__direct_gap__d1_report.json`
- `output/outer_lambda_exp020_softlift_seedfixed001/d2/r1_d2_report.json`
- `output/outer_lambda_exp020_softlift_seedfixed001/d3/r1_reference_export_report.json`

这些结果证明 R1 值得部署研究，但不等价于 D3 qualified，更不授权 production。

### 1.2 已冻结的部署失败证据

| 路线 | 数值 parity | CUDA cost ratio | 冻结结论 |
|---|---:|---:|---|
| N0 full-system `CustomGBForce` cost floor | probe | `1.6965x` | `STOP_FULL_SYSTEM_CUSTOMGB` |
| N1 per-anchor local CV | pass | `6.0717x` | cost failed |
| N2 OpenMM-Torch Verlet | pass | `61.2922x` | cost failed |
| EXP-021 G1 grouped density CV skeleton | skeleton | median `1.107419x`; bootstrap P95 upper `1.114105x` | cost failed |

相关报告：

- `output/outer_lambda_exp020_softlift_seedfixed001/d3/r1_n1_n2_local_report.json`
- `output/outer_lambda_exp020_softlift_seedfixed001/d3/r1_native_n0_cost_floor_report.json`
- `output/outer_lambda_exp021_grouped_density_cv/d0_cost/g1_skeleton_cost_report.json`

N1/N2 的 parity 通过仅说明数学可以被复现，不改变 cost failure。EXP-021 的
停止结论保持有效；EXP-025 是新的实现假设，不能覆盖、修改或重解释 EXP-021。

### 1.3 为什么专用 kernel 仍值得验证

EXP-020 R1 不是通用消息传递网络。它只有：

- 41 个 ligand anchors；
- 7 个冻结 atom types；
- 16 个 Gaussian radial bases；
- 每条边产生一个 scalar density contribution；
- 每个 anchor 只有一个 scalar \(q_i\)；
- typed `1 -> 16 -> 16 -> 1` SiLU readout；
- 总参数量 3,031。

多体性来自“对邻居求和以后再做非线性”：

\[
\rho\left(\sum_j \phi(r_{ij})\right)
\ne \sum_j \rho(\phi(r_{ij})).
\]

因此无需显式生成 triplets，也不需要
\(O(n_{\mathrm{neighbor}}^2)\) 的 Tersoff-style angular expansion。专用实现只需
两遍局部 edge scan 和一次 41-anchor readout。

---

## 2. Scope 与 non-goals

### 2.1 本实验负责

1. 精确复现冻结 EXP-020 R1 的能量和保守力。
2. 提供独立 Reference oracle 和 OpenMM CUDA backend。
3. 实现只面向 ligand-environment cross edges 的局部 candidate list。
4. 作为 raw basis 接入现有 Outer-lambda/IBS wiring。
5. 在 73,536 原子真实体系上先过 correctness，再过 cost gate。
6. 记录完整 build、model、system、benchmark 和 CUDA provenance。

### 2.2 明确不负责

- 不替换 bonded、LJ、PME、Boresch 或 WCA。
- 不修改物理端点 Hamiltonian。
- 不实现 Tersoff、NEP、完整 MACE 或环境-环境 message passing。
- 不将 pair-additive potential 宣称为 R1 的等价实现。
- 不使用 full-system `CustomGBForce`。
- 不使用 41 个 per-anchor `CustomCVForce`。
- 不使用 OpenMM-Torch/TorchForce 作为 production backend。
- 不在 EXP-025 中训练新 teacher 或改变 EXP-020 D1 数据划分。
- 首版不支持 multi-GPU、NPT/barostat、动态 ligand membership 或在线训练。
- correctness/cost 未同时通过前，不进入 production promotion。

---

## 3. 冻结数学合同

### 3.1 输入与索引

对单帧构型：

- positions: \([N,3]\)，OpenMM 内部单位 nm；
- periodic box: OpenMM restricted triclinic box；
- ligand topology IDs: \([N_L]\)，本体系 \(N_L=41\)；
- environment topology IDs:
  `all atoms - ligand atoms`；
- atom type index: \([N]\)，int32，取值 `[0,6]`；
- candidate edges: 仅 \(i\in L,j\in E\)；
- self、ligand-ligand、environment-environment 不属于本模型。

atom-type vocabulary、41 个 ligand topology IDs 和 environment mask 必须来自冻结
artifact/manifest；不得在 CUDA 代码中凭元素名称重新推断。

### 3.2 Radial density

定义 16 个冻结 Gaussian bases：

\[
G_p(r)=\exp\left[-\frac12
\left(\frac{r-c_p}{\sigma}\right)^2\right],
\qquad p=1,\ldots,16.
\]

centers、width 和排列必须从冻结 checkpoint/config 读取并写入 EXP-025 manifest；
首版不得重调。

inner cutoff \(r_{\mathrm{in}}=0.4\,\mathrm{nm}\)，outer cutoff
\(r_c=0.5\,\mathrm{nm}\)。Quintic C2 envelope：

\[
C_2(r)=
\begin{cases}
1,& r\le r_{\mathrm{in}},\\
1-10s^3+15s^4-6s^5,
&r_{\mathrm{in}}<r<r_c,\\
0,&r\ge r_c,
\end{cases}
\]

其中 \(s=(r-r_{\mathrm{in}})/(r_c-r_{\mathrm{in}})\)。

每条 cross edge：

\[
e_{ij}=C_2(r_{ij})
\sum_{p=1}^{16}W_{t_i,t_j,p}G_p(r_{ij}).
\]

每个 ligand anchor：

\[
q_i=\sum_{j\in E,\ r_{ij}<r_c}e_{ij}.
\]

typed pair weights 的 shape 为 `[7,7,16]`，共 784 个参数。类型顺序是有方向的
`[ligand_type, environment_type]`，但产生的坐标力必须满足作用-反作用。

### 3.3 Typed nonlinear readout

每个 ligand type 拥有一个冻结的小 MLP：

\[
\rho_t:\mathbb{R}\rightarrow\mathbb{R},
\qquad 1\rightarrow16\rightarrow16\rightarrow1,
\]

hidden activation 为 SiLU。7 个 typed MLP 共 2,247 个参数。

定义精确 no-contact centering：

\[
h_i=\rho_{t_i}(q_i)-\rho_{t_i}(0).
\]

ligand aggregate 和 bounded reduced basis（冻结实现使用求和，不除以 \(N_L\)）：

\[
S(R)=\sum_{i=1}^{N_L}h_i,
\]

\[
B(R)=B_{\max}\tanh\left(\frac{S(R)}{B_{\max}}\right),
\qquad B_{\max}=10.
\]

因此无任何 ligand-environment contact 时应有 \(q_i=0\)、\(S=0\)、\(B=0\)
且原子力为零。不得用 bias、padding edge 或 ligand-only embedding 破坏该合同。

### 3.4 Force derivative

设

\[
g_{ij}(r)=\sum_p W_{t_i,t_j,p}G_p(r),
\]

则

\[
\frac{de_{ij}}{dr}=C_2'(r)g_{ij}(r)
+C_2(r)\sum_p W_{t_i,t_j,p}G_p(r)
\left[-\frac{r-c_p}{\sigma^2}\right].
\]

同时

\[
\frac{\partial B}{\partial q_i}
=\operatorname{sech}^2\left(\frac{S}{B_{\max}}\right)
\rho_{t_i}'(q_i).
\]

插件输出物理能量：

\[
U_B=k_BT B.
\]

raw basis force 为：

\[
\mathbf F^{B}_{i\leftarrow j}
=-k_BT\frac{\partial B}{\partial q_i}
\frac{de_{ij}}{dr}
\frac{\mathbf r_i-\mathbf r_j}{r_{ij}},
\]

并对 environment atom 写入严格相反的力。outer path force 由现有 controller 再乘
\(A_k\)。

### 3.5 单位和 exactly-once 规则

| Quantity | Unit / semantics | Applied where |
|---|---|---|
| positions, distance | nm | OpenMM/plugin |
| cutoff | `0.5 nm`，不是 numeric `5.0` | plugin |
| \(B,S,q\) | reduced/dimensionless contract | model |
| \(k_BT\) | kJ/mol | plugin，恰好一次 |
| \(U_B\) | kJ/mol raw basis | plugin output |
| \(A_k\) | dimensionless outer coefficient | existing controller，恰好一次 |
| energy offset | kJ/mol | existing outer expression，恰好一次 |
| beta | ledger already reduced；不得在 plugin 再乘 | loss/ledger contract |

插件 artifact 固定 `a_k=1`、不内置 window-specific coefficient、不减 energy offset。

---

## 4. OpenMM plugin 架构

### 4.1 目录草案

```text
plugins/LocalManyBodyResidual/
|-- CMakeLists.txt
|-- openmmapi/
|   |-- include/openmm/LocalManyBodyResidualForce.h
|   `-- src/
|       |-- LocalManyBodyResidualForce.cpp
|       `-- LocalManyBodyResidualForceImpl.cpp
|-- serialization/
|   `-- LocalManyBodyResidualForceProxy.cpp
|-- platforms/
|   |-- reference/
|   |   `-- ReferenceLocalManyBodyResidualKernel.cpp
|   `-- cuda/
|       |-- CudaLocalManyBodyResidualKernel.cpp
|       `-- kernels/localManyBodyResidual.cu
|-- python/
|-- tests/
`-- README.md
```

遵循 OpenMM 标准层次：

1. public `Force` 保存可序列化的物理/模型合同；
2. `ForceImpl` 负责 group mask、Context 和 Kernel 调度；
3. abstract `KernelImpl` 定义 Reference/CUDA 共同接口；
4. Reference/CUDA `KernelFactory` 分别向平台注册；
5. `SerializationProxy` 只保存稳定主机状态，不保存 GPU 临时状态。

### 4.2 Public Force 最小 API

建议冻结的构造输入：

```cpp
LocalManyBodyResidualForce(
    std::vector<int> ligandTopologyIds,
    std::vector<int> atomTypeIndex,
    double temperatureKelvin,
    double innerCutoffNm,
    double outerCutoffNm,
    double skinNm,
    LocalManyBodyR1Parameters parameters
);
```

必要查询/更新：

- ligand IDs、type vocabulary/hash；
- cutoff/skin/temperature；
- RBF centers/width；
- pair weights；
- typed MLP weights；
- `Bmax`；
- model SHA-256 / protocol SHA-256；
- `updateParametersInContext()` 仅允许 same-shape coefficient update；
- ligand membership、model dimensions、cutoff 或 type map 改变必须重建 Context。

首版不把 \(A_k\) 或 lambda 注册为 plugin global parameter，因为 plugin 只输出 raw
basis。outer wrapper 已负责系数，不应建立第二套 lambda wiring。

### 4.3 与现有 IBS 的连接

推荐结构：

```text
LocalManyBodyResidualForce  --raw U_B-->  existing OuterLambdaIBSBiasForce
                                             |
                                             `-- applies A_k and offset once
```

- plugin Force 作为 shared raw basis child 使用；
- existing OuterLambda/IBS wrapper 保持 top-level Group 1；
- plugin 不单独作为额外 production bias group 重复加入 System；
- base、target、sampling-bias energy histories 继续分离；
- disabled baseline 必须完全不执行 plugin kernel，而不是把 scale 设为 0 仍付成本；
- serialization/restart 必须恢复 outer Context 的当前 globals，不能只依赖 XML defaults。

如果实际 OpenMM ownership 证明 raw plugin child 无法按现有 wrapper 安全持有，必须形成
amendment；不得偷偷增加第二层 `CustomCVForce`，因为那会重新引入 EXP-021 的调度问题。

---

## 5. CUDA 数据布局

### 5.1 持久 device buffers

```text
anchorSystemIds[41]
anchorDeviceIds[41]
atomTypeIndex[paddedN]
environmentMask[paddedN]          # 或固定 environment ID list

cellKeys[N_env]
cellAtoms[N_env]
cellOffsets[nCells+1]

anchorOffsets[42]
edgeAtoms[maxEdges]
edgeCount
overflowFlag

q[41]
dB_dq[41]
energyScratch

lastRebuildPositions[paddedN or participating subset]
lastBoxVectors

pairWeight[7*7*16]
rhoParameters[7 typed MLPs]
rbfCenters[16]
modelScalars
```

`anchorOffsets/edgeAtoms` 构成 anchor-major CSR。首版建议：

- `skin = 0.1 nm`；
- active edges 必须满足冻结 hard ceiling `<=2048`；
- active neighbors per ligand 必须 `<=80`，unique environment `<=320`；
- skin candidate buffer 可初始分配 `4096`、保守分配 `8192`，但这不放宽 active
  support ceilings；
- capacity 不足时 set overflow，当前 step 不得使用截断边继续积分；
- host 在可控同步点扩容并重建，或 fail closed；
- 任何 fallback 和同步成本都计入 benchmark。

### 5.2 Atom reorder

OpenMM CUDA 可动态重排 atom arrays。topology atom ID 不能永久等于 device position
index。

CUDA implementation 必须注册 reorder listener：

1. 读取 OpenMM 当前 atom-index map；
2. 生成 system topology ID 到 current device index 的反向映射；
3. 重新上传 41 个 anchor device IDs；
4. environment/type mapping 与新排列一致；
5. 立即 invalidate candidate CSR 和 last-position state；
6. 下一次执行前强制 rebuild。

缺失该机制属于 P0 correctness failure。

### 5.3 OpenMM force/energy buffers

CUDA 平台的主 force buffer 是 fixed-point `long long`，不是普通 float/double。

force kernel 必须按目标 OpenMM 版本的约定：

- 使用 `realToFixedPoint(force_component)`；
- 对 x/y/z 的 padded offsets 做 atomic add；
- 只累加，不覆盖其他 Force 的贡献；
- energy 写入 OpenMM energy buffer 并走平台 reduction；
- 不做每 step host energy readback。

所有 private CUDA API 都以冻结 OpenMM commit 为 ABI。OpenMM 版本改变必须重编译并重跑
全部 correctness/cost gates。

---

## 6. Local candidate/Verlet 设计

### 6.1 为什么不复用 full-system generic neighbor path

首版不耦合 OpenMM private `CudaNonbondedUtilities`：

- 它面向全体系 pair tiles；
- 不自然暴露 41-anchor 到 environment 的紧凑 CSR；
- R1 在 density aggregate 与 force pass 之间需要全局同步；
- internal API 对 OpenMM commit 敏感；
- generic interaction expression/CV dispatch 已在 EXP-021 显示明显成本。

只有 profiling 证明自建 list 是主要瓶颈，且存在可验证的稳定共享接口时，才另立
amendment 研究复用。

### 6.2 Rebuild pipeline

候选表使用 `r_list = r_c + skin = 0.6 nm`：

1. `computeCellKey`: environment atoms 转到 periodic fractional/cell coordinates；
2. stable sort or deterministic bin by `(cellKey, topologyId)`；
3. `buildCellOffsets`；
4. `queryAnchors`: 每个 anchor 枚举相邻 cells；
5. exact MIC distance `< r_list` 写入 CSR；
6. 保存 rebuild positions 和 box；
7. active forward/force 时再次检查 `< r_c` 并应用 C2。

重建条件：

- 任一 participating atom 的 displacement 超过 `skin/2`；
- box vectors 改变；
- OpenMM atom reorder；
- Context reinitialize / checkpoint restore；
- force group 被跳过期间列表有效性无法证明；
- explicit parameter/topology update。

首版为 fixed-box NVT。NPT/barostat 标记 `UNSUPPORTED`，直到 box-change、最短周期高度和
重建成本全部通过独立 gate。

### 6.3 PBC

- 内部长度全部为 nm；
- 使用 OpenMM CUDA 提供的 periodic delta macros/box pointers；
- 不独立实现一个可能与平台 half-box tie 不同的 inverse-box `round`；
- restricted triclinic box 的最短周期高度必须大于 `2*r_list`；
- 任一 active pair `r < 0.01 nm`（0.1 Å）必须 fail closed；
- half-box tie、near-singular/high-skew box 和 `r -> 0` 必须有明确 fail policy；
- Reference oracle 与 CUDA 使用相同的 OpenMM periodic convention，但分别实现计算，
  避免代码同源掩盖错误。

---

## 7. Kernel launch 设计

### 7.1 正常 step

候选表有效时，每步只需三个阶段：

#### K1: `accumulateAnchorDensity`

- 一个 warp 或一个 small block 对应一个 anchor；
- lanes 遍历 `anchorOffsets[i]:anchorOffsets[i+1]`；
- exact cutoff、C2、RBF16、typed pair weights；
- warp reduction 得到 scalar `q[i]`；
- 避免 edge-wise atomic add 到 q；
- 41 个 anchor 可打包到一个或少数 thread blocks。

#### K2: `evaluateTypedReadout`

- 处理 41 个 `q[i]`；
- typed `1-16-16-1` SiLU；
- 减冻结的 `rho_t(0)`；
- ligand sum（不除以 41）；
- bounded tanh；
- 产生 raw `U_B=kBT*B`；
- 同时写 `dB_dq[i]`。

该 kernel 工作量很小，可与 K1 尾部融合，但只有 profiling 证明 launch overhead 显著且
融合不破坏可读性/测试性时才做。

#### K3: `scatterConservativeForce`

- edge-parallel；
- 重算 C2/RBF 和径向导数，不保存大 edge-feature tensor；
- 读取对应 anchor 的 `dB_dq[i]`；
- 计算 equal/opposite pair force；
- 写入 OpenMM fixed-point force buffer；
- 不对非参与原子产生力。

### 7.2 Rebuild step

在上述 K1 前增加 cell/CSR rebuild。benchmark 必须分别记录：

- cell key/bin/sort；
- CSR query；
- K1 density；
- K2 readout；
- K3 force；
- rebuild frequency 和 rebuild spike P95；
- total OpenMM step time。

### 7.3 首版 correctness kernel

可先实现 41 x environment brute-force CUDA scan 作为独立 correctness milestone，约
3.0 million candidate distance checks/step。它只用于：

- 验证 CUDA math；
- 验证 fixed-point force；
- 验证 atom mapping/PBC/wrapper wiring。

它不是 cost candidate。不得以 brute-force 失败推断局部 CSR kernel 失败，也不得以小体系
brute-force 通过宣称 production cost qualified。

---

## 8. Precision、确定性与数值安全

### 8.1 Precision profiles

- Reference oracle: CPU double；
- CUDA qualification: production CUDA precision profile；
- CUDA double 仅作为诊断（若平台支持且成本可接受）；
- 不要求 CPU/CUDA bitwise equality；
- 要求固定版本、固定模型、固定 edge convention 下满足预注册容差。

建议初始 tolerance，必须在正式 preregistration 中封存：

| Check | double/reference | production mixed |
|---|---:|---:|
| energy max abs | `1e-6 kJ/mol` | `1e-4 kJ/mol` |
| force max abs | `1e-5 kJ/mol/nm` | `1e-3 kJ/mol/nm` |
| finite-difference relative | `1e-5` | `1e-3` |
| net force norm | `1e-8` | `1e-4 kJ/mol/nm` |

absolute floor、relative denominator 和 excluded near-zero components 必须机器化定义。

### 8.2 Fail-closed conditions

以下任一情况不得静默继续 trajectory：

- edge buffer overflow；
- atom/type/index mapping mismatch；
- model SHA/config SHA mismatch；
- nonfinite q、B、energy 或 force；
- unsupported box/multi-GPU/NPT；
- neighbor list validity unknown；
- serialization version unknown；
- cutoff/skin/box safety violation；
- force group/wrapper ownership 不符合 manifest；
- checkpoint restore 后 atom map 或 Context parameters 未恢复。

状态枚举固定为：`PASS / FAIL / UNKNOWN / NOT_RUN`。`UNKNOWN` 不得被解释为 PASS。

---

## 9. Reference oracle 与测试矩阵

### 9.1 Independent Reference implementation

Reference backend 使用 CPU double、41 x all-environment brute force，不使用 CUDA CSR。
它必须独立计算：

- edge membership 和 MIC distance；
- \(q_i\)；
- typed MLP intermediate；
- \(S,B,U_B\)；
- analytic forces；
- per-anchor diagnostics。

同时保留冻结 PyTorch R1 作为第二个 oracle。CUDA 必须同时对齐 Reference 和冻结
PyTorch；两个 oracle 互相不一致时状态为 UNKNOWN/FAIL，不可选择更有利者。

### 9.2 必测 fixtures

1. frozen real frames：覆盖低/中/高 edge occupancy；
2. no-contact frame：energy/force exact zero；
3. single-edge/two-atom hand calculation；
4. multi-neighbor same-anchor，证明非 pair-additive cross term；
5. cutoff `r_in +/- epsilon`、`r_c +/- epsilon`；
6. `r -> 0` guard；
7. orthorhombic PBC；
8. restricted triclinic PBC；
9. cross-boundary anchor-environment pair；
10. half-box tie fail policy；
11. rotation/translation invariance；
12. permutation invariance与固定 ligand local order；
13. participating/nonparticipating atom forces；
14. net force和 pair action-reaction；
15. finite-difference energy-force；
16. CUDA atom reorder listener；
17. force-group mask；
18. XML serialize/deserialise/new Context；
19. checkpoint/restart/nondefault Context globals；
20. edge-cap overflow；
21. repeated-run drift和 precision parity。

### 9.3 Outer-lambda integration checks

至少验证：

- plugin raw basis 不含 \(A_k\)；
- endpoint \(A=0\) 时 path contribution 为零；
- mid-state contribution 为 `A_k*(U_B-offset)`；
- kBT、A、offset、tanh 各应用一次；
- outer force 等于 \(A_k\mathbf F_B\)；
- target history 改变不污染 sampling-bias history；
- disabled baseline bitwise/within platform tolerance 等同于无 plugin System。

---

## 10. EXP-025 gates

### G0 — Build/ABI smoke

目标：证明开发环境和插件生命周期成立。

必须冻结：

- OpenMM exact version/commit；
- CUDA toolkit/driver；
- compiler/CMake；
- GPU model；
- precision/platform properties；
- OpenMM source/private-header SHA；
- plugin binary SHA。

通过条件：Reference/CUDA plugin 均可注册、load、serialize、创建 Context；empty/no-contact
Force 返回零；没有 crash 或 leaked top-level Force。

### G1 — Reference mathematical qualification

通过条件：

- 对冻结 frames 精确复现 PyTorch R1 的 q、B、energy、force；
- no-contact zero；
- analytic/finite-difference force；
- C2、PBC、invariance 全过；
- exactly-once unit/wiring 全过。

### G2 — CUDA brute-force correctness

只验证 CUDA implementation correctness：

- 对齐 Reference/PyTorch；
- fixed-point forces 正确；
- atom reorder、group mask、serialization/restart 全过；
- real 73,536-atom Context 可执行；
- 不将其 throughput 作为最终 gate。

### G3 — Local CSR/Verlet correctness

通过条件：

- candidate set 与 Reference all-pairs 完全一致；
- skin 内运动不漏边；
- rebuild threshold、box change、reorder 均正确；
- cutoff active edge 完全一致；
- overflow 测试 fail closed；
- energy/force 继续满足 G2 容差。

### G4 — Cost-first qualification

G4 是 EXP-025 的核心 go/no-go 门。正式 benchmark 前必须封存完整 R1 operation shape，
不能用会被编译器消除的 zero/dummy expression。

建议硬门：

\[
\operatorname{median}(T_{candidate}/T_{baseline})\le1.07,
\]

且

\[
\operatorname{P95Upper}(T_{candidate}/T_{baseline})\le1.10.
\]

benchmark protocol：

- exact 73,536-atom system；
- production CUDA platform/precision/integrator；
- frozen low/mid/high occupancy frame manifest；
- baseline/candidate paired and interleaved order；
- warmup >= 100 steps；
- measured >= 1,000 steps/repeat；
- >= 5 paired repeats；
- timed integration loop 内无 reporter/getState/q diagnostic；
- 另测 matched production total，按真实 IBS/ledger query cadence；
- pure integration 与 matched production total 都报告，后者为 normative gate；
- 报 median、每次 ratio、bootstrap one-sided P95 upper、peak memory、OOM；
- 记录 edge count、rebuild frequency 和 kernel breakdown；
- 所有失败/超时/OOM 尝试纳入 intention-to-test 记录。

G4 failure 后：

- 状态 `STOP_EXP025_RUNTIME_BACKEND`；
- 不进入 full scientific/online qualification；
- 不改阈值、不换便宜 frame、不减少真实 IBS query 来补 pass；
- 后续架构变化需要 amendment 或新 experiment ID。

### G5 — Full artifact and D3-equivalent qualification

只有 G4 通过才运行：

- 加载冻结 EXP-020 checkpoint 的参数；
- 明确 9 checkpoints 的 canonical deployment selection 规则，禁止按 test 选最优 seed；
- 或按预注册规则全数据重训一个 final artifact；
- 完整 CUDA/PyTorch/Reference parity；
- full-system outer-lambda energy/force/wiring；
- model/system/XML/checkpoint hashes；
- 重启一致性；
- full D3 report。

### G6 — Dynamics and utility

分为两个不可混淆的子门：

1. `G6-SAFETY`: paired short NVT，能量/温度/约束/最大力/finite/neighbor
   rebuild/support/saturation 全部健康；
2. `G6-UTILITY`: 正式 paired online pilot，按既有 ESS/GPU-hour protocol 评估。

short NVT pass 不等于 utility pass。production promotion 仍需至少：

- >= 2/3 independent paired repeats 的 candidate ESS/GPU-hour 优于 baseline；
- median material improvement 达到预注册门；
- TMBAR/overlap/coverage/ledger closure/DeltaG uncertainty 全部通过；
- 总 wall-clock/GPU-hour 包含 warmup、正式采样、在线 ledger 与失败尝试；
- 不出现 lambda state reorder 或 endpoint Hamiltonian 变化。

---

## 11. 成本假设与可证伪预测

### 11.1 主假设

EXP-020 R1 的 runtime 成本主要来自通用 OpenMM Force/CV/Torch 调度与错误的全体系
candidate domain，而不是 R1 本身的算术量。

若使用：

- one local CSR/Verlet list；
- 41-anchor density reduction；
- one tiny typed readout；
- one edge force pass；
- no per-step host synchronization；
- no full-system pair pass；

则完整 R1 有机会满足 G4。

### 11.2 否证条件

以下结果会否证该假设：

- local CSR build/rebuild 本身使 P95 超过 1.10；
- OpenMM inner/raw-basis ownership 必须引入昂贵嵌套 Context；
- fixed-point force atomics或 kernel launches 仍超过预算；
- atom reorder/box/list safety 只能靠高频 host sync 实现；
- 完整 R1 而非 skeleton 才暴露明显吞吐下降；
- correctness 需要退回全体系 scan；
- G4 matched production total 未通过。

EXP-025 允许失败。失败结论比再次包装成另一条 CV 路线更有价值。

---

## 12. 性能优化顺序

只有在 correctness 已通过、profiling 指向明确瓶颈后，才按以下顺序优化：

1. amortize local-list rebuild；
2. anchor-major warp reduction，消除 q atomics；
3. constant/read-only cache 保存 3,031 个参数；
4. K1/K2 有条件融合；
5. edge derivative recomputation vs compact derivative cache 实测比较；
6. stable cell-key radix sort 与 topology-order reproducibility；
7. force atomic contention优化；
8. CUDA graph/launch batching（只有 OpenMM lifecycle 允许时）；
9. 最后才研究复用 OpenMM shared nonbonded list。

禁止的“优化”：

- 丢边、top-k 或 neighbor truncation；
- 减小真实 cutoff；
- 改 RBF 数量、hidden width 或模型数学；
- 停止给 environment 施加反作用力；
- 降低真实 IBS/ledger query cadence后只报告便宜结果；
- 只测零能量/零力 dummy kernel；
- 把 rebuild spike 或失败 run 从统计中删掉。

---

## 13. Build 与环境前置条件

G0 前必须拿到和目标 OpenMM binary 匹配的开发环境：

- exact OpenMM source/build tree；
- public 和目标版本 private CUDA headers；
- CUDA toolkit/nvcc；
- compatible host compiler；
- CMake/build generator；
- Python/OpenMM runtime；
- CUDA device 可见且与 EXP-021 benchmark 平台一致。

EXP-021 证据对应的 OpenMM 版本记录为
`8.5.2.dev-36a30cb`；EXP-025 必须从实际 environment manifest 重新核实，不得只相信
字符串。installed Python wheel 若缺 private platform headers，不足以构建 CUDA plugin。

首版 platform support：

| Platform | Status |
|---|---|
| Reference CPU | required |
| CUDA single GPU | required |
| CPU platform optimized | optional |
| OpenCL/HIP | out of scope |
| multi-GPU CUDA | unsupported/fail closed |
| NPT/barostat | unsupported/fail closed |

---

## 14. Artifacts 与 provenance

建议产物树：

```text
protocols/EXP-025_preregistration.json
output/outer_lambda_exp025_local_manybody_cuda/
|-- g0_build/
|   |-- environment_manifest.json
|   |-- plugin_load_report.json
|   `-- plugin_binary_manifest.json
|-- g1_reference/
|   |-- reference_parity_report.json
|   `-- fixture_manifest.json
|-- g2_cuda_bruteforce/
|   `-- cuda_bruteforce_correctness_report.json
|-- g3_local_csr/
|   |-- candidate_parity_report.json
|   `-- overflow_rebuild_report.json
|-- g4_cost/
|   |-- frame_manifest.json
|   |-- paired_timing_samples.json
|   |-- kernel_profile.json
|   `-- cost_qualification_report.json
|-- g5_full_d3/
|   |-- model_manifest.json
|   |-- full_parity_report.json
|   `-- serialization_restart_report.json
|-- g6_dynamics/
|   |-- short_nvt_report.json
|   `-- ess_gpu_hour_report.json
`-- decision_log.jsonl
```

每个 report 至少保存：

- experiment ID、document/protocol version；
- canonical self-hash 和 raw file SHA-256，字段名不得混用；
- git commit 和 uncommitted diff hash；
- source/build/plugin binary SHA；
- OpenMM/CUDA/driver/compiler/CMake versions；
- GPU model、platform properties、precision；
- system/topology/positions/box/frame-manifest hashes；
- EXP-020 checkpoint/config/type-map hashes；
- ligand/environment index hashes；
- cutoff/skin/capacity/rebuild policy；
- all raw timing/metric samples；
- gate status与失败尝试；
- parent evidence hashes；
- decision ID和amendment chain。

旧报告 append-only，不得覆盖。任何 threshold、数据、frame、模型、atom membership、
OpenMM commit、CUDA math 或 benchmark workload 的改变都需要 amendment；影响方法身份时使用
新的 experiment ID。

---

## 15. 实现伪代码

```cpp
double CudaLocalManyBodyResidualKernel::execute(
    ContextImpl& context,
    bool includeForces,
    bool includeEnergy) {

    validateContextAndAtomMap();

    if (neighborListInvalid(context)) {
        rebuildLocalAnchorCSR(context);       // no truncation
        failIfOverflowOrUnsupported();
    }

    launchAccumulateAnchorDensity(            // q[41]
        positions, box, anchorOffsets, edgeAtoms,
        atomTypes, pairWeights, rbf, cutoff);

    launchTypedReadout(                       // B, U_B, dB/dq[41]
        q, ligandTypes, rhoWeights,
        rhoAtZero, bMax, kBT, energyBuffer);

    if (includeForces) {
        launchScatterConservativeForce(
            positions, box, anchorOffsets, edgeAtoms,
            atomTypes, pairWeights, rbf, cutoff,
            dB_dq, kBT, openmmFixedPointForceBuffer);
    }

    return includeEnergy ? reduceEnergyBuffer() : 0.0;
}
```

实际 OpenMM CUDA kernel 应直接使用平台 energy buffer，避免上面伪代码暗示每 step host
reduction/readback。

---

## 16. Go / Stop 决策表

| Gate | Pass 后 | Fail 后 |
|---|---|---|
| G0 build/ABI | 写 Reference | 修环境；不写 CUDA math |
| G1 Reference | 写 CUDA brute oracle | STOP correctness design |
| G2 CUDA brute | 写 local CSR | 修 CUDA；不得讨论成本合格 |
| G3 local CSR | 运行 G4 cost | STOP neighbor backend |
| G4 cost | 才进入 full artifact/D3 | `STOP_EXP025_RUNTIME_BACKEND` |
| G5 full D3 | paired short NVT | production仍禁止 |
| G6-SAFETY | online utility pilot | STOP dynamics |
| G6-UTILITY | 才可形成 promotion decision | 保持 offline/reference only |

任何 gate 的 PASS 都不自动重新打开 EXP-012 WP-5 或修改 EXP-020/021 的冻结结论。

---

## 17. 当前建议

EXP-025 的第一轮工作只做到 G0-G4：

1. 固定 OpenMM/CUDA build environment；
2. Reference oracle；
3. CUDA brute correctness；
4. local CSR/Verlet；
5. 使用完整 3,031 参数 R1 operation shape 做真实 cost gate。

在 G4 以前不投入完整 Python packaging、多平台支持、NPT、多 GPU 或新一轮训练。

若 G4 通过，这条路线才是一个值得继续工程化的“生物力场 + 局部多体残差”backend；
若 G4 失败，应正式记录专用 CUDA backend 的成本边界，而不是回到 GB、per-anchor CV 或
Torch bridge。

