# 外层 λ 神经基势详细实施计划

> **文档角色：工程执行计划。** 科学原则见
> [`PLAN_outer_lambda_neural_basis.md`](PLAN_outer_lambda_neural_basis.md)，真实运行结果记录在
> [`EXPERIMENT_LOG_outer_lambda_neural_basis.md`](EXPERIMENT_LOG_outer_lambda_neural_basis.md)。
> 本文同时维护任务状态；设计项不等于 production 已接入。
>
> **2026-08-07 归档说明**：所有已完成/已终止工作包的完整时间线、通过门、数值证据和
> 历史决策记录已整体归档至
> [`IMPLEMENTATION_PLAN_outer_lambda_neural_basis_archive.md`](IMPLEMENTATION_PLAN_outer_lambda_neural_basis_archive.md)
> （该文件是归档前本文档的完整快照）。本文档此后只保留：仍在生效的目标/约束/设计规范、
> 当前代码地图、以及**尚未完成或尚未关闭**的工作。任何"已完成"条目如需查证据，去归档文件按
> 章节标题或工作包编号搜索。

## 1. 实施目标

在不改变物理端点的条件下，将少量不显式依赖 λ 的冻结神经基势加入 Stage 2
目标中间 Hamiltonian：

\[
\widetilde H_{\lambda_k}(\mathbf R)
=H_{\lambda_k}^{0}(\mathbf R)
+w(\lambda_k)\sum_{m=1}^{M}c_m(\lambda_k)\overline U_m(\mathbf R)
\]

实施约束：

- 第一版 \(M=1\)，只有证据表明单基势不足时才扩展至 \(M=2\sim4\)。
- 每个 \(U_m\) 在同一步坐标上只计算一次。
- 窗口内全部 λ 状态通过外层系数共享基势。
- 神经项进入 target energy，不进入需要消除的 IBS `bias_history`。
- production 前冻结模型和系数；production 中禁止在线训练。
- 默认关闭；关闭时必须保持当前结果和缓存语义不变。

### 1.1 当前工程状态（2026-08-07 精简，完整证据见归档）

| 工作包 | 状态 | 摘要 |
|---|---|---|
| WP-0 | 完成 | window 0、primary torsion 和两个 secondary/diagnostic 候选已登记 |
| WP-1 | 完成 | 外层包络、系数、端点归零、协议哈希 |
| WP-2 | 完成 | mock Force 与 IBS target/bias/base 账本契约 |
| WP-3 | 完成（通用骨架） | TorchForce、CustomCVForce、序列化、CPU/CUDA/checkpoint 接口 |
| WP-3A | 失败并停止 | EXP-009 的 PythonForce/CUDA MTS 后端在 \(N=1\) 失败 |
| WP-4 | 完成直接 MACE qualification | `coefficient=0.09` 的 EXP-007 通过；不等于可接受的生产成本 |
| WP-4A / EXP-010 | `FAILED` | atom-cut protein MACE 教师六候选跨 run 验证全部失败，教师边界无物理闭合 |
| WP-4B / EXP-011 | `FAILED / STOPPED` | AUG-001 后 mutual overlap `0.02353 < 0.03`；不再补采、不拟合 PMF |
| WP-4C / EXP-012 | D0-D4 全部关闭；**单基势在线部署路线已正式关闭（DEC-050，2026-08-07）** | teacher(`derived_5a`) 已双平台通过 C1；`LocalResidualStudent` direct-gap 路线已通过 (d1)/(d2)/(d3)/(d4)；WP-5A 的三次 paired-reseed exploratory pilot 中 raw `mixture_ess_proxy` 均提升 10.7-27.8%，但这不是三组独立平衡的 production repeats，且当前部署成本使 ESS/GPU-hour 三次均下降（DEC-048/050：授权的性能救援目标 `1.10×baseline` 判定 `TARGET_UNREACHABLE`，即使假设动态建图成本降到零仍差远）；(d0-2/3/4) 设计冻结未完成，现已不再相关 |
| WP-4D / EXP-013 | **未晋级：方案③ `FAILED`（DEC-055/056），方案① Qualification `NOT_MET`（DEC-058），方案② N=1 ESS `FAILED`（DEC-059）** | 方案①禁止 N=16/013-C；方案② `mixture_ess_proxy` 下降 `18.88%`，不运行其 MTS、不重调 `c1`；EXP-014 离线压缩筛选未通过（DEC-060），不进入 OpenMM 资格化或 production promotion |
| WP-5 | **`NOT_PROMOTED / FROZEN`（DEC-048/050/059/060，2026-08-09）** | WP-5A pilot、EXP-013 三种在线/MTS 方案及 EXP-014 compression screen 均未晋级；不直接重开 WP-5，不重调 `c1`/checkpoint，不继续搜索 MTS 间隔 |
| WP-6–8 | 未开始，已阻塞 | 由 WP-5 结果条件触发，WP-5 未晋级故不启动 |

当前所有新增实现均位于独立文件 `outer_lambda_neural_basis.py` 及独立测试/脚本中。
`runabfe.py`、`abfe_core.py`、`abfe_pipeline.py`、`ibs_engine.py` 未接入该模块，这是
有意的研发隔离，不是遗漏。

## 2. 第一轮明确不做

- 连续 λ-conditioned MACE。
- 未处理的全体系 pretrained MACE 总能量直接叠加。
- 把 XED charges 作为第二套静电加入 PME，或创建真实 massless OpenMM particles。
- 把手工 XED sites 固定为 EXP-012 主路线；XED 只允许作为 Arm D 可选消融。
- 把 MACE latent 或仅由 MM gap/ESS 监督的模型解释为真实电子密度、Pauli/exchange 势。
- 用离线 NumPy descriptor 直接驱动 production Force，或忽略 encoder 对坐标的反向传播。
- 同时改变基础势、λ schedule、IBS 权重算法和神经路径。
- 一次性处理水、离子、侧链和 ligand 的所有慢自由度。
- 未通过小窗口验证就启动 complex/solvent 完整 production。

## 3. 推荐基线决策

当前真实产物使用：

```text
mode=ibs
decoupling=dual_lambda
potential=softcore
dexp_params=null
```

推荐顺序：

1. 在当前 `softcore + dual_lambda + IBS` 上完成接口、账本和科学原型。
2. 原型通过后，再迁移至 DEXP。
3. 如果论文必须从 DEXP 起步，则先完成无神经 DEXP 的 ABFE、长程处理和独立重复，
   不同时引入神经路径。

## 4. 当前代码映射

| 文件 | 当前职责 | 计划中的职责 |
|---|---|---|
| `outer_lambda_neural_basis.py` | 独立外层控制器、账本适配、慢变量筛选与历史 CLI 兼容入口；EXP-010/011 实现已迁至 `archive/outer_lambda_exp010_exp011_legacy.py` | 只复用 controller、shared CV、ledger、PBC、TorchForce 和 benchmark 骨架；旧 fragment/PMF 实现不复用 |
| `local_residual/` | EXP-012 v2 主命名空间；`LocalResidualStudent`（direct-gap）、`student_deploy.py`（TorchForce 部署路径）、`teacher_graph.py`（逐帧构图）、`mace_latent.py`（C1 latent adapter）等已实现；production 不导入 | 承载 A/B/C/D、MACE latent adapter、student、训练和部署实现（进行中） |
| `exp012_xed/` | DEC-018 早期兼容/证据命名空间；现有 ledger/schema 实现继续可复现 | 不再定义方法身份；XED 只允许作为 Arm D 可选消融 |
| `protocols/EXP-012_preregistration.json` | `exp012-local-residual-prereg-v2`，`sealed`（DEC-039） | 已冻结的部分不再改；(d0-5) ms/step 缺口补齐后视需要重新哈希 |
| `MaceLatentBasisAdapter`（C1 底层接口） | `derived_5a` CPU+CUDA 均已通过 C1；`original_6a` CPU 通过、CUDA `BLOCKED_ON_VRAM` | 无变化计划，`original_6a` CUDA 视 gradient checkpointing 实现情况决定是否重试 |
| `LocalResidualStudent` | direct-gap 训练与 (d1)/(d2) 资格已通过；distilled 变体未通过相对 direct-gap 的增量门，未采用 | (d3) 部署资格（TorchScript/OpenMM/CUDA/耗时）已 `CLOSED`（DEC-043），(d4) 短 NVT 动力学资格已通过（DEC-044），D0-D4 全部关闭 |
| XED-inspired feature（可选） | 现有 schema 名称中出现，但尚无 feature 实现 | 仅作为 Arm D 消融；不创建 PME 电荷或 OpenMM virtual particles |
| `runabfe.py` | CLI、配置合并、模式路由 | WP-5 独立资格通过后才考虑接收显式神经路径配置 |
| `abfe_core.py` | 基础势、DEXP、MACE 辅助能力 | 当前不修改；后期只合并已冻结接口 |
| `abfe_pipeline.py` | Stage 调度、provenance、缓存 | 当前不修改；后期接收协议指纹 |
| `ibs_engine.py` | IBS 系统、CV、能量探针、TMBAR | 当前不修改；后期接入已验证的 shared Force 和账本 |
| `tests/test_outer_lambda_*.py`、`tests/test_neural_basis_ibs_accounting.py` | 独立端点、账本、TorchForce、GPU、CLI、慢变量和 cheap-CV 测试 | 合并前回归证据 |

重点函数和类：

- `abfe_core._build_mace_potential`
- `abfe_core.OrbVacuumContext`
- `ibs_engine.build_ibs_dual_system`
- `ibs_engine.IBSBiasForce`
- `ibs_engine.IBSSampler._build_probe_context`
- `ibs_engine.IBSSampler.evaluate_interaction_energies`
- `ibs_engine.IBSSampler.collect_energies`
- `ibs_engine.IBSWindowManagerDualLambda._build_window_system`
- `abfe_pipeline.ABFEPipeline._run_dual_lambda_stage`

现有 `OrbVacuumContext` 只能作为依赖加载参考，不能直接充当 production 神经路径。

## 5. 核心对象设计

### 5.1 `NeuralBasisModelSpec`

每个冻结基势至少记录：

- 模型名称、类型和绝对路径；
- 文件 SHA-256；
- 模型来源和训练数据版本；
- 元素集合、cutoff、精度和设备；
- 局部原子选择及其哈希；
- 固定能量基准 \(b_m\)；
- 输出单位；
- 周期盒支持；
- 支持域和安全阈值。

### 5.2 `OuterLambdaController`

职责：

- 计算 \(w(\lambda)\) 和 \(c_m(\lambda)\)；
- 生成全局系数矩阵

  \[
  A_{km}=w(\lambda_k)c_m(\lambda_k)
  \]

- 验证端点、有限性、有界性和平滑性；
- 保证相同 λ 总是生成逐位相同的系数；
- 输出稳定、可序列化的协议 payload。

第一版只支持：

```text
envelope = sin2
coefficient model = constant
M = 1
```

后续才增加非对称包络、Bernstein 或 B-spline。

### 5.3 共享神经 CV

当前每个 IBS 状态可概念化为：

\[
X_k=U_{k,\mathrm{int}}+U_{k,\mathrm{rest}}-f_k
\]

加入神经路径后：

\[
X_k
=U_{k,\mathrm{int}}
+U_{k,\mathrm{rest}}
+\sum_m A_{km}\overline U_m
-f_k
\]

实施规则：

- 每个神经基势只加入一个共享 CV。
- 不为每个 λ 复制相同神经 Force。
- \(A_{km}\) 以只读高精度常数进入表达式。
- 显式检查：

  \[
  2K+M\le32
  \]

- 相邻窗口公共 λ 的 \(A_{km}\) 必须逐位相同。

### 5.4 sampler 能量账本

神经路径启用后：

```text
target_state_energies
    = original_state_interaction
    + LRC_if_applicable
    + neural_path_state_energy
```

采样偏置仍为：

```text
sampling_bias_energy
    = IBS log-sum-exp bias
    + sampling-only WCA
```

每帧处理顺序：

1. 读取 λ 无关 base energy；
2. 读取实际 IBS/WCA sampling bias；
3. 计算现有各态 interaction energy；
4. 计算 \(M\) 个共享神经基势；
5. 用 \(A_{km}\) 得到 \(K\) 个神经路径能量；
6. 形成完整 target energies；
7. 执行同步 finite gate；
8. 分别追加 target、bias、base 历史。

任何分量非有限时，整帧按现有 hard gate 处理，不能用零值替代。

## 6. 建议配置契约

第一版优先提供一个入口：

```text
--neural-path-config FILE
```

示意 YAML：

```yaml
neural_path:
  enabled: false
  protocol_version: 1
  stage: vanishing
  baseline_potential: softcore
  endpoint_tolerance: 1.0e-12

  envelope:
    type: sin2
    parameters: {}

  coefficient_model:
    type: constant
    coefficients: [1.0]
    max_abs_coefficient: 1.0

  bases:
    - name: reorg_basis_0
      backend: torchforce
      model_path: /absolute/path/model.pt
      sha256: required
      energy_offset_kj_mol: 0.0
      atom_selection: fixed_indices
      atom_indices_path: /absolute/path/indices.json
      output_unit: kJ_per_mol
      precision: single
      periodic: true

  safety:
    max_abs_basis_energy_kj_mol: 50.0
    max_abs_path_energy_kj_mol: 20.0
    max_force_norm_kj_mol_nm: 500.0
    fail_on_support_domain_violation: true
```

说明：

- 程序必须重算模型哈希，不能只相信配置文本。
- 安全阈值在成为生产值前必须通过基线数据校准。
- 第一版不开放运行时修改 \(A_{km}\)。

## 7. 工作包

## WP-0：冻结基线

状态：`COMPLETED`。完整冻结结果（体系、torsion、任务清单、通过门）见归档 §WP-0。

## WP-1：外层控制器纯数学测试

状态：`COMPLETED`（独立模块）。任务与通过门见归档 §WP-1。

## WP-2：解析 mock 基势的 IBS 账本

状态：`COMPLETED`（独立模块，尚未合入 production IBS）。8 项验证与通过门见归档 §WP-2。

## WP-3：TorchForce 最小部署

状态：`COMPLETED`（通用独立部署骨架）。验证范围见归档 §WP-3。

### WP-3A：神经 Force 多时间步调度

状态：`FAILED / STOPPED`。EXP-009 在 PythonForce/CUDA MTS 后端 \(N=1\) 已触发
`CUDA_ERROR_INVALID_HANDLE`；不再执行 \(N=2/4/8\)。完整预注册矩阵、性能比较项和
MTS 通过门定义见归档 §WP-3A（保留为历史预注册协议，不能误写成尚待运行）。

## WP-4：单个真实任务化基势

状态：直接完整 MACE 的 EXP-007 qualification 已通过；由于成本和 EXP-009 后端失败，
它只保留为教师。候选范围、模型要求和通过门见归档 §WP-4。

### WP-4A：直接 MACE 与廉价蒸馏的双路线

当前决策：直接 MACE 路线结束；EXP-010 廉价蒸馏路线 `FAILED`——290/300 帧通过支持域门，
intercept-only 能量 RMSE `21.5109 kJ/mol`，最佳候选（1D Fourier order 2）leave-one-run-out
RMSE `22.1737 kJ/mol`、广义力 \(R^2=-13.5934\)，其余候选更差；事后审计还发现 216 个
protein atoms 涉及的 26 个残基无一完整。因此没有候选被冻结，失败原因是教师构造和逐帧
能量目标，不排除 primary torsion 本身。完整协议见归档 §WP-4A。

### WP-4B：EXP-011 完整 MM 条件平均力/PMF

状态：`FAILED / STOPPED`（`FORMAL_RUN1_OVERLAP_FAILED`）。24 centers × 3 replicates 正式
采样已完成（2400 帧全部有限），但 AUG-001 补采 `127.5°` 后严格 MBAR 显示
`112.5°↔127.5°` mutual overlap 仍为 `0.02353 < 0.03` 门，且只有 22 个去相关样本
（门 25）。结论 `COMPLETED_NOT_ACCEPTED`；不再补采、不拟合 PMF、不进入 NVT 或 WP-5。
完整实施顺序、umbrella/reweight 实现、逐窗 overlap 时间线见归档 §WP-4B。

### WP-4C：EXP-012 CV-free 通用局部残差路径势

**已完成/已冻结部分**（完整时间线、DEC 决策编号、SHA-256 证据见归档 §WP-4C）：

- 协议主语义与导入入口已迁移为 `local_residual` A/B/C/D；三条 scratch run 的逐帧五态
  target ledger、CPU/CUDA backend audit 均已通过。
- 路线正式定性为 L2（DEC-030）：`derived_5a`（5 Å，1444 节点/60048 边）teacher 已
  双平台（CPU+CUDA）通过 C1；`original_6a`（6 Å，2135 节点/155624 边）CPU C1 通过，
  CUDA 因 product-layer-1 张量积算子真实峰值需求 `>=~24.07 GiB` 仍 `BLOCKED_ON_VRAM`
  （未完成，见下方开放项）。
- (a) 多帧支持域审计、(b) 逐帧独立构图的离线 latent cache（三条 run 各 500 帧，
  `local_residual/teacher_graph.py::build_teacher_graph_for_frame`）、(c) teacher-side
  cached-latent 线性 readout 的 leave-one-run-out held-out 验证（held-out run1/2/3 相对
  `B=0` 基线分别改善 39.9%/29.1%/64.8%，均值 44.6%，3/3 fold 全部改善）均已完成并通过。
- Arm A/B/D 正式退役为 `not_pursued`（DEC-039）：三者从未实现任何代码，非数值跑输；
  结论收窄为"MACE latent 存在可泛化 gap-variance 信号、值得蒸馏"，不得声称 Arm C 优于
  A/B。
- (d0-1) 在线动态环境表示已由 DEC-038 real-data smoke 解决（动态 cutoff funnel +
  `quintic_c2_cutoff` 平滑权重，与 teacher canonical membership 逐对一致）。
- (d0-5) 模型规模（≤50k 参数目标/≤100k 硬上限）、图规模（S1≤256/320，边≤1536/2048，
  单原子 neighbor≤64/80，1500 帧真实审计）、CPU float64/CUDA float32 funnel 一致性、
  训练 seed（≥3/变体/折）、早停规则（训练 run 内末尾 20% 时间块，`max_epoch=500`、
  `early_stop_patience=30`）、D1 go/no-go 判据均已冻结（DEC-039）；唯一未关闭的缺口是
  ms/step 生产基线重新测量（见下方开放项）。
- (d1) 离线 `LocalResidualStudent` 拟合：direct-gap 路线保留进入 D2；distilled 路线未
  通过相对 direct-gap 的增量门，未采用。
- D1 direct-gap 的最终 held-out 结果须与上述 teacher-side readout 分开解释：平均
  gap-variance 改善为 `13.9348%`，fold-level 门计 `2/3`，且
  `direct_gap_all_folds_improved=false`（原始报告：
  `output/outer_lambda_exp012/student_training_report_direct_gap_only.json`）。因此只能称为
  candidate offline signal，不能称为 direct-gap student 全折一致改善。
- (d2) 坐标/autograd 资格：3 folds × 3 seeds 共 27 组检查（有限差分、非参与原子零力、
  cutoff 平滑性、force tail）全部通过，`COMPLETED_D2_CHECKS`，report SHA
  `75f0e2ca...aeec8c97c`。
- D3-0 provenance gate 已关闭（DEC-041，`CLOSED_STEP3_OPERATIONAL_SEMANTIC_PASS`）：
  System 身份判定为无可检测语义差异，`no_student_window0_baseline_v2.json`
  （median/P95 `1.3959/1.3968` ms/step）可采信为无 student 生产基线。

**当前仍开放的工作**（这是项目当前前沿，完整细节见第 13 节）：

- (d0-2)（最小架构候选冻结）、(d0-3)（teacher-target 协议冻结）、(d0-4)（必需对照实验
  冻结）三项设计冻结尚未完成——即使 (d1)/(d2) 已经用某个具体架构跑过，这三项作为
  正式冻结文档仍是未勾选状态，不得默认已经等价冻结。
- (d0-5) 剩余缺口：当前无 student 的生产 ms/step 基线待重新测量（根因
  `box_vectors.npy` 陈旧已定位并修复诊断脚本，重新测量尚未执行）。
- (d3) 部署资格：**`CLOSED`（DEC-043，2026-08-07）**。四个 sub-item 现状：
  sub-item 1（deployment 一致性）`PASSED_OPERATIONAL_NUMERICAL_EQUIVALENCE`——
  `.pow(2)`→`.square()` 假设已用真实数据重跑（`student_d3_1_2_report_v3.json`）证伪
  （对残差无可测量影响，真正根因未查明、裁定不再追查）；`reference_eager_vs_
  deployable_eager` 残差 `1.69e-7` 比已接受的 CPU64→CPU32 精度包络（`1.32e-5`）小
  约 2 个数量级、比已接受的 CPU32↔CUDA32 设备误差（`7.65e-7`）还小，`correctness_
  cpu64_vs_cpu64=1e-8` 判定为未进入 sealed preregistration 的自定工程阈值，不再拿它
  卡住项目。sub-item 2/3（TorchForce/OpenMM 注入、端点归零）`PASSED`（DEC-042，严格
  `0.000e+00`）。sub-item 4-correctness（cell-list 与 brute-force 等价性）`PASSED`——
  新增 `tests/test_exp012_student_deploy_cell_list.py` 全部通过（重构 `student_deploy.
  py::_DeployableStudent` 抽出 `_energy_from_edges` 消除了初版测试用 box 扰动强制切
  分支引入的物理混淆变量后）。sub-item 4-performance（生产 Context 耗时 `~95%`
  overhead）按 §16 是非阻塞工程目标，如实保留，不再是 D3/D4 前置门。
- (d4) 动力学资格：**`PASSED`（DEC-044，2026-08-07）**，首次执行即通过。新增
  `scripts/check_exp012_student_d4_short_nvt.py`：把 student TorchForce 挂到真实
  `hard_window0` win_sys 独立 force group 上，用同一真实 checkpoint 的 3 个不同
  `LangevinMiddleIntegrator` 种子做独立重复（种子在 `loadCheckpoint()` 之后设置，避免
  checkpoint 内嵌 RNG 状态覆盖种子选择），每个重复成对跑 no-student/with-student；500
  步 warmup + 2000 步监控。report_sha256
  `f06ad7b03ce85ab4ee443fab20e259124e3ea2bc7e41777b0a586f9648783554`：
  `all_finite=true`，student 力贡献 `10.0–39.4 kJ/mol/nm`（阈值 `500`），student 能量
  `-7.52~+0.31 kJ/mol`（与 `a_k=0.5` 理论上界 `±12.47 kJ/mol` 一致），两种配置温度全程
  `298.3–302.5K`（目标 `300K`，sanity 门 `150–600K`），全系统最大力 `~4300–6300
  kJ/mol/nm`（显式溶剂正常键伸缩背景，与 student 无关）。安全阈值是脚本自定的工程
  sanity 门，非 sealed 数值门（D4 从未预注册过）。D4 只回答"积分会不会炸"，不回答
  "采样有没有变好"——ESS/GPU-hour/mutual overlap 资格是 WP-5A 的工作。

## WP-4D / EXP-013：低频在线 student（MTS/rRESPA）

状态：**DEC-060：EXP-013 全顺序不晋级；EXP-014 native-compression screen 未通过**（2026-08-09）。方案③ 013-B `FAILED`；方案① Qualification gate 未通过；方案② N=1 ESS signal 未通过，EXP-013 不晋级。DEC-050 关闭的是"每个 MD
步都调用一次 student TorchForce"这一种在线部署模式；EXP-013 改变的是**评价频率**这个
正交轴——把 student 的（昂贵的）能量/力评价放进独立的慢 force group，用 OpenMM 原生
MTS/rRESPA 积分器（`MTSLangevinIntegrator`）每 \(N\) 个内步才调用一次，而不是优化
`student_deploy.py` 本身的推理成本。**不是重开 EXP-009**：EXP-009 死的是 openmmml
1.6 的完整 MACE + `openmm.PythonForce` 后端，在 `MTSLangevinIntegrator` 的 force-group
内核里于 \(N=1\) 就触发 `CUDA_ERROR_INVALID_HANDLE`（历史代码见
`outer_lambda_neural_basis.py:2779-2789`，明确记录"普通 LangevinMiddleIntegrator 通过
不能证明该组合可用于 MTS"，并预注册了 `start_exp010_cheap_cv_due_to_backend` 的转向）；
当前 `LocalResidualStudent` 走的是 openmm-torch 的 `TorchForce` 后端，是完全独立的一条
通道，此前从未在 MTS 场景下测试过，不受 EXP-009 结论约束。

本节其余内容是已执行方案的历史设计与证据解释，不构成新的实验授权；当前在线/MTS 分支已
封存，不再继续搜索评价间隔或改变 `c1`、checkpoint。

**理想成本下界的算术**（DEC-050 report_sha256 `98496b12...`）：设 \(t_0=1.4765\)、
\(\Delta t=t_{\text{full}}-t_0=2.0913\) ms/step，若 student 每 \(N\) 个内步只评价一次，
理想下界 \(t(N)\approx t_0+\Delta t/N\)：

| \(N\) | 理想成本 / baseline |
|---:|---:|
| 4 | 1.354× |
| 8 | 1.177× |
| 12 | 1.118× |
| 16 | 1.089× |
| 24 | 1.059× |
| 32 | 1.044× |

\(N\approx16\) 才进入经济上有意义的区域；\(N=2/4\) 不值得做正式性能实验。

**关键架构约束**：OpenMM 的 MTS 机制在 force-GROUP 粒度上工作——
`MTSLangevinIntegrator(temperature, friction, dt_outer, [(慢group, 1), (快group, N)])`，
每个 Force 对象只能整体属于一个 group，组间能量/力是**线性加和**的（历史 EXP-009
precedent，`outer_lambda_neural_basis.py:2761-2796`：全部经典力被塞进 group 0，MACE
单独占 group 31，两组线性相加；`dt_outer` 是最外层步长，`N` 是一个 outer step 内快
group 被计算的次数，慢 group 只算 1 次——EXP-009 里 `outer_timestep_fs = mts_ratio *
inner_dt`，`N=mts_ratio`）。但 DEC-048 实测有正向信号的那个设计
（`OuterLambdaIBSBiasForce`，wiring smoke 起）是把 student 的 basis 值作为一个
collective variable **融合进** IBS log-sum-exp 判别式 \(X_k=\text{cv}_k^{int}+
\text{cv}_k^{rest}+A_k\cdot\text{basis}-f_k\) 内部——这是一个**非线性**函数
（`log(sum(exp(...)))`），不能拆成"经典部分的 log-sum-exp + 神经部分的 log-sum-exp"
再线性相加。

**三个候选方案，按优先级排序（2026-08-07 用户提出并核实，DEC-052）**：

- **③ exact residual split（首选）**。定义
  \(V_0(\mathbf R)=-\beta^{-1}\log\sum_k\exp[-\beta(U_k^0(\mathbf R)-f_k)]\)
  （无 student 的经典判别式）与
  \(V_*(\mathbf R)=-\beta^{-1}\log\sum_k\exp[-\beta(U_k^0(\mathbf R)+A_kB_\theta(\mathbf
  R)-f_k)]\)（DEC-048 验证过的融合判别式），取
  \(\Delta V_\theta(\mathbf R)=V_*(\mathbf R)-V_0(\mathbf R)\)，MTS 分组为
  \(V_{\text{sampling}}=V_0^{\text{fast}}+\Delta V_\theta^{\text{slow}}\)。因为
  \(V_0+\Delta V_\theta\equiv V_*\) 是构造性代数恒等式，\(N=1\) 时的 Hamiltonian
  与 DEC-048 的融合设计**严格相同**，不需要为新设计重新证明 ESS 信号，只需要验证
  "\(\Delta V_\theta\) 这个 `CustomCVForce` 写对了没有"——数值等价性检查，沿用 D3
  的方法学，不是物理有效性检查。**唯一真正的风险是计算重复**：`CustomCVForce` 各自
  有自己的 inner Context，\(\Delta V_\theta\) 的慢 Force 必须在自己内部重新算一份
  \(U_k^0\)（`cv_k_int`/`cv_k_rest`），不能免费复用快 group 已经算出的值——这部分
  重复开销是真实成本，必须先测，不能假设"student 部分才贵"。
- **① whole fused Group-1 slow（③工程/成本失败才选）**。整个 Group 1（含经典
  `cv_k_int`/`cv_k_rest` 和 student CV）一起放进慢 group，不改变 \(N=1\) 时的
  Hamiltonian，但经典软核项也变成每 \(N\) 个内步才更新一次——这些项未必是"慢力"，
  \(N=16\) 对应 outer step \(8\) fs（若沿用 EXP-009 的 0.5 fs 内步长），\(N=32\)
  对应 \(16\) fs，经典相互作用在这个尺度上可能先于 student 出问题。013-B 必须专门
  检查这一点，不能只看 student 力/能量分布。
- **② independent additive student（① Qualification gate 未通过才选）**。student 拆成独立的、线性
  加和的额外 Force/group（回到 D3/D4 最初"独立 force group"设计），不再融合进
  log-sum-exp 判别式，配合经典 IBS bias（Group 1，逐步更新，不含 student）。这
  **已经是一个新的 sampling Hamiltonian**——融合判别式下 neural force 实际带一个
  构象依赖系数 \(-\nabla V_*\supset-[\sum_k p_k(\mathbf R)A_k]\nabla B_\theta\)（其中
  \(p_k(\mathbf R)\propto e^{-\beta(U_k^0+A_kB_\theta-f_k)}\)），不能用固定系数
  \(c\,B_\theta\) 代替。选②必须先在 \(N=1\) 补做一次简单 ESS 对照（不需要完整 3
  重复 pilot），确认这个新设计本身还有没有 DEC-048 那种正向信号，否则"MTS 有没有
  用"会被"这个新设计本身有没有用"混进去，重复 DEC-045~050 警告过的"把部署问题和
  模型/设计问题混在一起"的错误。

**决策顺序固定为 ③→①→②，不并列评估、不跳着选**；每一步的失败标准见下方各小节，
失败即进入下一个候选，不回头重试同一个。

### 013-A：exact-residual 数值等价性 + 单次调用成本（首选③的验证）

状态：**`PASSED`（DEC-053，2026-08-07）**。`scripts/check_exp013_residual_split_
equivalence_and_cost.py`，report_sha256
`bc9eb24dcb5d54297664028b2207156efff83b6693d8569e2f3c76d0bcc45519`：数值等价性
成立（相对容差 `1e-3` 内）；`ΔV_θ` 单次调用真实增量 `1.8822ms`；`N=8` 不可行
（预算 `1.1058ms`），**`N=16`（预算 `2.2115ms`，余量约 15%）与 `N=32`（预算
`4.4230ms`）均可行**。方案③不需要转方案①。下一步 013-B。

不是先测 MTS 能不能跑，也不是先测 ESS，是先测两件事：

1. **数值等价性**（D3 方法学，不是新物理）：在真实生产帧上分别构造 \(V_0+\Delta
   V_\theta\)（快 group 用未改动的经典 `IBSBiasForce`，慢 group 用新建的
   \(\Delta V_\theta\) `CustomCVForce`）与 \(V_*\)（wiring smoke 已验证过的
   `OuterLambdaIBSBiasForce`），核对 \(E_{V_0+\Delta V}-E_{V_*}\approx0\) 与
   \(\mathbf F_{V_0+\Delta V}-\mathbf F_{V_*}\approx0\)（严格容差，因为这是同一个
   Hamiltonian 的两种等价写法，不是跨精度/跨方法比较）。
2. **单次调用成本**：直接测 \(\Delta V_\theta\) 作为一个独立 force group 注入真实
   `win_sys` 后单次评价的真实耗时（沿用 DEC-050"同一 Context、同一调用路径"的
   matched-path 方法，不用独立 Python 微基准）。用 DEC-050 的
   \(t_0=1.4765\)ms、目标比例 \(1.10\times\) 给出预算：每个内步允许增加
   \(0.14765\)ms，\(\Delta V_\theta\) 单次调用允许成本 \(C_\Delta\le
   N\times0.14765\)ms：

   | \(N\) | \(C_\Delta\) 预算上限 |
   |---:|---:|
   | 8 | 1.181 ms |
   | 16 | 2.362 ms |
   | 32 | 4.725 ms |

   已知 `full-baseline` 增量约 `2.0913ms`（DEC-050），\(\Delta V_\theta\) 因为要
   重复算一份经典 CV，预期成本 \(\gtrsim2.09\)ms——\(N=16\) 非常紧、可能刚好不够，
   \(N=32\) 有明显余量但 outer interval 更长、动力学资格门槛更严，这正好构成一组
   干净的 go/no-go：**若 \(C_\Delta\) 在 \(N=8\) 甚至 \(N=16\) 都超预算，③在工程上
   即告失败，转①**；若有余量，才进入 013-B 的物理/动力学资格。此外仍需确认
   `MTSLangevinIntegrator` 配 `CustomCVForce`+`TorchForce` 嵌套在 \(N=1\) 能正常跑
   （数值应与①②检查一致，避免 EXP-009 式 backend 报错；出现类似报错 EXP-013 当场
   结束，不重试、不换节点、不调系数）。

### 013-B：时间尺度资格

短程验证 \(N=1,8,16,32\)，逐一对比：student/判别式相关能量分布；力范数与尾部；温度；
结构合理性；相对 \(N=1\) 的能量/构象分布偏移；如可行，shadow-work 或积分误差代理。
\(N=8\) 是物理诊断用，不是经济候选；真正候选是 \(N=16/32\)。通过门：相对 \(N=1\) 参考
没有可分辨的系统性偏移（不是"没崩溃"就算过）。

**状态（DEC-054/055/056，2026-08-09）：首次执行报告已
`INVALIDATED_BY_INITIALIZATION_BUG`；修复后绝对健康门通过，但预注册相对系统性偏移门
失败。DEC-056 已完成裁决：方案③在 013-B `FAILED`，禁止进入 013-C；按冻结顺序转方案①。**

首次执行表面 `all_passed=true`，但复核发现 N=1 参考态本身已从 300 K 崩溃到 ~0.003 K，
四臂共病，相对门（只比较 N vs N=1）结构性看不见这种共同病灶。6 步消除法定位真根因：
`scripts/check_exp013_013b_mts_dynamics_qualification.py` 把生产 checkpoint
（`LangevinMiddleIntegrator` 写入）直接 `loadCheckpoint()` 到 `MTSLangevinIntegrator`
（`CustomIntegrator`）Context——OpenMM 文档明确 checkpoint 绑定写入它的具体
Context/Integrator，跨类型迁移不在支持范围内。已排除的候选（force-group coverage、
逐 group 力求和一致性、热浴 `a/b/kT` 系数、逐步计算程序结构、`ΔV_θ`/TorchForce 残差
项本身）全部正常；换成状态转移（`getState`→`setPositions/setVelocities/
setPeriodicBoxVectors/setParameter`，全程不对 MTS Context 调用 `loadCheckpoint()`）后，
同一 System、同一 integrator、同一初始物理态在整个 0–6.4 ps 窗口稳定在 296.5–302.4 K。
脚本已按此修复，并新增独立于相对比较的**绝对健康门**（`mean_temperature_k`/
`warmup_end_temperature_k` ∈ `[270,330]K`，`relative_energy_drift ≤ 0.10`，
`all_passed` 要求先过 N=1 自身绝对健康门再看相对门）防止未来同类共病再次被判过。
013-A 不受影响，仍 `PASSED`。详见 EXPERIMENT_LOG DEC-054 与
`output/outer_lambda_exp012/exp013_013b_mts_dynamics_qualification_report.json.INVALIDATED.md`。

**修复后重新执行（DEC-055，report_sha256 `64a963626ef36893d440823bd9845ca7c6123cda1b76159e175d9a893810caf3`）**：
`all_absolute_health_passed=true`（四个 N 的 `mean_temperature_k`
`298.50/299.16/299.51/299.78 K`、`relative_energy_drift` `0.0020/0.0020/0.0008/0.0019`，
均在健康门内，`relative_comparison_meaningful=true`）——DEC-054 的修复确认生效，N=1
这次是真实健康的参考态。但预注册的相对系统性偏移门（`z_threshold=3.0`）在 N=8/16/32
**全部失败**：`temperature_k` 随 N 单调升高（+0.66/+1.01/+1.29 K，z=9.57/14.59/18.32，
干净的 dose-response）；`e_v0_plus_dv_kj_mol`（驱动 IBS 判别式的量）也显著偏离参考
（z=10.67/4.85/11.68），但不随 N 单调。温度的单调升高量级虽小（<1.3K），但统计上
极显著，物理解释与此前已预见的风险一致：`ΔV_θ` 依赖构象相关的态权重
\(p_k(\mathbf R)\)，这些权重随快原子运动快速涨落，降频评价会引入 MTS 式共振/离散化
升温——不是新 bug，是方案③在这个系统里的真实统计行为。**这是真实物理结果，不是
初始化 artifact；`<1.3 K` 仅作为物理量级补充报告，不能推翻预注册的 `z>3` 主门。**
详见 EXPERIMENT_LOG DEC-055/056。

### 013-B/①：方案① whole fused Group-1 低成本预检

DEC-056 的下一执行点是新增的
`scripts/check_exp013_design1_mts_precheck.py`（入口
`run_exp013_design1_mts_precheck.sh`）。它把完整的 fused
`OuterLambdaIBSBiasForce` 放在 Group 1 慢组，Group 0/2/3/4/5 作为快组；固定
`inner_dt=2 fs`，只跑 `N=1/2/4/8`，使用相同物理时间的 macro tick。入口分两阶段：

- `smoke` 保留短默认值 `16/32` ticks（`0.256/0.512 ps`），只检查 CUDA/MTS backend、
  State API 初始化、有限值和温度健康；即使短程普通 SEM 出现 `z>3`，也只能作为诊断，
  永远不能设置 `eligible_for_n16_followup=true`；
- `qualification` 固定为 `400/2000` ticks（`6.4/32 ps`，与 DEC-055 时间尺度一致），
  固定每 tick `0.016 ps` 的物理采样间隔，并以连续 `50`-tick block（`0.8 ps`）的
  block-mean SEM 计算相对偏移。只有该阶段四臂绝对健康且无系统偏移，才允许设置
  `eligible_for_n16_followup=true`。

两阶段都同时检查每个 N 的绝对健康门、相对 N=1 的温度/Group-1 fused energy 偏移和
非有限值；资格门只在 qualification 阶段生效。

**Qualification 结果（DEC-058，2026-08-09）**：报告
`output/outer_lambda_exp013_design1_qualification/report.json` 的
`report_sha256=2d96b39e4f6571e131cc16fb98ee4a5b645f35b66455d8d53dfbd442ea3d6d9a`。四臂
绝对健康门通过，但 block-aware SEM 下 temperature 的 N=2/4/8 为 `z=5.61/5.79/6.83`，
fused Group-1 energy 的 N=8 为 `z=5.62`；`systematic_shift_detected_by_n` 三个 N
全部为 `true`，`eligible_for_n16_followup=false`。方案①判定
`DESIGN_1_QUALIFICATION_GATE_NOT_MET` 和 `N16_NOT_AUTHORIZED`；由于这是单种子
Qualification，`PHYSICAL_SYSTEMATIC_BIAS_INCONCLUSIVE`。不运行 N=16/32，不进入 013-C，
随后按 DEC-056 完成方案② N=1 ESS 信号检查；该检查结果见 DEC-059。

这里的“不通过”是程序性 gate 结果，不是“方案①已被证明物理失败”：block-aware SEM
没有跨 seed 重复间变异估计，且不同 MTS 轨迹分叉后不构成真正配对重复。N=2/4 的 fused
energy 未越门，N=8 才越门；因此不能把本次单轨迹结果外推为普遍系统偏差或错误系综。

只有四个 N 均健康且没有可分辨的系统性偏移，才允许另行执行方案①的 N=16；Qualification
gate 未通过则停止，不直接上 N=16/32，更不做三重复。N=16/32 在当前 `inner_dt=2 fs` 下分别是
`32/64 fs` outer step，风险必须先由低成本门筛掉。初始化沿用 DEC-054 修复后的
State API：只有同类 `LangevinMiddleIntegrator` probe Context 调用 `loadCheckpoint()`，
MTS Context 禁止跨 Integrator `loadCheckpoint()`。

### 013-C：WP-5A-mini 三重复

**EXP-013 的方案③、方案①和方案②均未获授权进入 013-C。** 方案① Qualification
gate 未通过，但其物理系统偏差结论仍为 `INCONCLUSIVE`；方案②的 N=1 ESS signal 又未
通过（DEC-059）。因此不运行方案② MTS，也不重新定义 013-C；只有另立新候选并完成
独立资格后，才可重新讨论新的 013-C。
Go/no-go：

- 至少 2/3 重复的 `mixture_ess_proxy_per_gpu_hour` > baseline；
- \(\Delta G_{\text{MTS}}\) 与 \(N=1\) student / converged MM 在预定统计容差内一致。

通过后才重新打开 WP-5B；不通过则按下述 EXP-014 转向，且**同样不重开 EXP-013 本身**
（同一失败即停规则）。

### 若 EXP-013 也失败：EXP-014（原生 OpenMM 压缩）

不是重开 EXP-010（那次失败的是特定的单 torsion Fourier cheap-CV 教师/目标/CV 协议，
不能推广成"局部径向/类型对多体压缩"本身无效；Arm A/B/D 当时是 `not_pursued`，从未
实现，不是数值跑输）。EXP-014 把已经证明有信号的 `LocalResidualStudent` 再做一次
"编译/压缩"：

\[
B_{\text{fast}}(\mathbf R)=\sum_{i\in L}\sum_{j\in E}\sum_p a_{t_i t_j}^{p}\,
\phi_p(r_{ij})\,s(r_{ij})
\]

即 typed pair + radial spline/RBF + cutoff，直接用 `CustomNonbondedForce`/等价原生
OpenMM 表达式执行，不需要每步调用 TorchForce。这是"gap → neural student → 原生解析
局部基"的压缩路径，不是回到 EXP-010 那条路线。

**EXP-014 离线筛选结果（DEC-060，2026-08-09）**：按 DEC-059 在独立目录
`output/outer_lambda_exp014_native_compression_audit_after_exp013/` 完成；`n_radial=8/16/32`
均未满足三折共同的 `held-out R²≥0.90` 与 retained student gap-variance improvement
`≥0.80` 门。`screening_passed=false`，`openmm_force_qualification=NOT_STARTED`，
`production_promotion=STOP`。此前标记 `INVALIDATED_OUT_OF_ORDER` 的报告不作为证据；
本次结果关闭当前冻结的 compression screen，但不外推为所有 analytic compression
形式均无效。

### 路线总览

EXP-012 证明 neural residual 有用（完成）→ DEC-050 证明每步 neural inference 不划算
（完成）→ EXP-013 方案③ 013-B 失败（DEC-055/056）→ 方案①先做
`N=1/2/4/8` 预检，合格后才考虑 `N=16`；方案①的 Qualification gate 未通过才进入
方案②，再失败转 EXP-014；EXP-014 当前冻结 screen 也未通过，停止该路线。当前不再考虑其它
compression、post-hoc/reweighting 或 MTS 间隔搜索；任何未来讨论都必须另立范围与决策，不能
作为本分支的自动后续
（EXP-015 候选，不是现在的优先级——DEC-050 已说明理由：post-hoc 只能改善 estimator，
不能改变已经生成的坐标覆盖，而 DEC-048 观察到的 mixture ESS 提升恰恰来自神经修正
真正参与了动力学）。

## WP-5：CV-free 局部残差的体系内与跨体系验收

状态：D0-D4 已全部关闭（DEC-044，2026-08-07），但 WP-5A pilot 已由 DEC-048/050 判定
`NOT_PROMOTED`；EXP-013 方案③→①→② 及当前 EXP-014 screen 均未形成
production-qualified route（DEC-059/060），
不得直接启动新的 WP-5A 三重复。当前最终边界为：不重调 `c1`、不重选或重训
checkpoint、不继续搜索 MTS 间隔、不直接重开 WP-5；本分支冻结。

**Production 候选冻结（DEC-045，2026-08-07）**：唯一候选 = `output/outer_lambda_exp012/
student_checkpoints/hard_window0_run1__direct_gap__seed0.pt`（SHA-256
`61abcd1f0d0ff809914003de522f05db66f9dc4b341391bfa0b7f1cb99e6f2e3`；D2/D3/D4 全程只用
这一个 checkpoint，未做过跨 checkpoint 挑选）。WP-5A 及以下所有工作**不得**因为
production 结果不理想就切换到其余 8 个已训练 checkpoint（`run2`/`run3`×`seed0/1/2`
等），也**不得**重新训练——重新训练需要重新走 D1→D4 整条资格链。D4 用的
`student_torchscript_d4.pt`（`a_k=0.5` 已烤进模块输出）**不可直接复用**于 WP-5A 的
IBS/TMBAR 接线：真实多态 wiring 需要 `OuterLambdaController` 按每态 `A_k=w(λ_k)·c_m`
在 `OuterLambdaIBSBiasForce` 的 CV 表达式里逐态相乘，喂给它的 basis Force 必须只输出
未缩放的原始模型能量（`a_k=1.0`），否则系数会被重复乘两次；需要为 WP-5A 重新导出一份
`a_k=1.0` 的 TorchScript。D4 运行环境：`torch 2.12.0`（CUDA 12.9）、`openmm
8.5.2.dev-36a30cb`、GPU `NVIDIA GeForce RTX 2080 Ti`（driver `580.173.02`）、conda 环境
`openmm_dev`。

### WP-5A：Atenolol 困难窗口体系内资格

固定四组对照（最终资格判定需要全部四组）：

1. 原始基础路径；
2. 仅 λ 重排；
3. 最简单 Arm A/解析接触基线；
4. EXP-012 选出的单局部残差势。

至少 3 个独立重复。主指标为 BAR mutual overlap、absolute/importance ESS、ESS/GPU-hour、
能量/力分位数、异常结构率和 converged-MM 自由能一致性。晋级要求端点与账本闭合、
稳定性不劣化、统计收益不能由 λ 重排或 Arm A 完全替代。

**当前执行步骤（2026-08-07 起，DEC-047）**：

- step 1（IBS/TMBAR 接线 smoke）：**`CLOSED`**（DEC-047）。`scripts/check_
  exp012_ibs_tmbar_wiring_smoke.py` 修正版全部通过（report_sha256
  `7d8f7ab3d4f98c950be589bbf7020ac1d94c2a1698761b6cdad2c631c97b9e06`）：ledger 闭合、
  target 组合逻辑严格正确（`2.842e-14`）、CUDA-mixed vs 独立 numpy 交叉验证通过（相对
  容差 `1e-4`）、端点 `A_0=0` 精确成立、DEC-041 provenance verdict 采信。
- step 2（production 候选冻结）：**`FROZEN`**（DEC-045/047）。checkpoint
  `hard_window0_run1__direct_gap__seed0.pt`（SHA
  `61abcd1f0d0ff809914003de522f05db66f9dc4b341391bfa0b7f1cb99e6f2e3`）、wiring
  smoke 用的 `a_k=1.0` TorchScript 导出方式、`c1=0.5`、`A_k` 系数矩阵（`sin2` 包络 ×
  常数系数）、cutoff 与运行配置全部冻结，不再因下游 production 结果调整；不重新训练。
- step 3（baseline/student paired-reseed exploratory pilot）：**`COMPLETED`（DEC-048，2026-08-07）**。
  3 次配对重抽全部执行完成（`velocity_draw_matches=true`）；它们是探索性两臂比较，
  不是三组独立平衡的 production repeats，也不是四组对照的完整 WP-5A——"仅 λ 重排"
  和"Arm A 解析接触基线"两组尚未安排。
- step 4（判据评估）：**`pilot_promotion_verdict=FALSE`（DEC-048）**——不晋级，不进入
  WP-5B。关键发现（细节见 EXPERIMENT_LOG DEC-048）：原始 `mixture_ess_proxy`（min-
  over-states 混合覆盖 ESS 代理）在 3/3 次 paired-reseed exploratory pilot 里 student 都比 baseline 高
  （+10.7%/+10.6%/+27.8%）——神经修正对混合覆盖确有真实正向信号；但 student 每步
  计算成本涨了 `1.81×~1.89×`（与 DEC-042/043 已测的 ~95% TorchForce 额外开销/约
  1.95× 单步成本一致），ESS 增益远不足以补偿成本涨幅，故 `mixture_ess_proxy_per_
  gpu_hour` 3/3 全部下降（中位数改善 `-684.22`）。ΔG：3 个重复里 2 个未收敛（100 帧
  pilot 量级下 `solve_stage_integrated` 默认去相关门槛未达标，疑为样本量不足而非
  物理不一致），收敛的那个 repeat `delta_g_z=2.268`（刚过门槛 2.0，原始差值仅
  `2.67 kJ/mol≈0.64 kcal/mol`）。按§16 已预先写明的规则（"若约 1.95× 单步成本没有
  被足够的 ESS 增益补偿，才判该路线性能失败"）判定为**性能失败，不是科学假设失败**。
  按"每个子阶段失败即停"规则，暂停在此，等待用户对三个方向之一做出新的、显式记录的
  决策：(a) 就此关闭单基势路线；(b) 优化 `student_deploy.py` 推理开销；(c) 尝试更大
  的 `c1`（需作为新决策而非事后调参）。不得未经这类明确决策就直接推进 WP-5B。
  上述三个方向是当时的待决选项，已被 DEC-059/DEC-060 及当前最终执行边界 supersede；不再执行。

- **DEC-049（2026-08-07）授权部署性能救援**：用户选择 (b)，并给出精确成功标准——
  仅限部署实现（不改模型权重、不改 `c1`/`A_k`、不改物理路径）；成功标准 = 物理输出
  等价 + 总步耗降到约 `1.10×baseline`（当前 `1.81×~1.89×`）。`1.10×` 的选取依据：
  三个重复里最小的 ESS 增益是 `10.6%`，`1.10×` 成本涨幅卡在增益下限附近留安全边际，
  确保救援成功后 ESS/GPU-hour 是真实净改善而非刚好归零。**未达到 `1.10×` 即正式关闭
  当前单基势在线（real-time TorchForce-during-MD）路线**，不再无限期尝试。
- **DEC-050（2026-08-07）终局判决：`TARGET_UNREACHABLE_CLOSE_ONLINE_PATH`**——正式
  关闭。首次 profiling 尝试（stage 相减）产出物理上不可能的负数"sync overhead"，
  被判定方法论有误后弃用；改为 4 变体（`baseline`/`zero_output`/`network_only`/
  `full`）同一 win_sys、同一调用路径、同一计时方法的终局实验
  （`scripts/measure_exp012_student_matched_path_lower_bound.py`，report_sha256
  `98496b12fa9ca61f74415d732478c4011f1361cecf268d3c761e61036104d9e1`）：
  `baseline=1.4765`、`zero_output=1.6683`（桥接成本 `+0.1918ms`）、
  `network_only=2.7042`（固定边集+真实网络前向反向，`+1.2277ms`）、
  `full=3.5678`（当前真实部署，`+2.0913ms`）ms/step。预算 `0.14765ms`；
  `network_only_delta(1.2277)` 超预算 `8.3×`——**不是边际未达标，是结构性不可达**：
  即便假设动态建图成本能优化到零，剩下的 `network_only` 本身仍是 `1.83×baseline`，
  离 `1.10×` 差距巨大。**关闭范围明确限定为"在线/real-time-during-dynamics"这一种
  部署模式**；不代表关闭"离线/post-hoc reweighting"式应用（完全不同的研究方向，
  需独立设计，本决策不隐含批准或否定）。D0-D4/wiring smoke/pilot 的全部结果（模型
  有真实统计信号、接线力学正确、ledger 闭合）依然作为有效证据保留，不因此被推翻——
  关闭的是"当前部署实现能不能在本项目产生净收益"，不是"模型本身没用"。不进入
  WP-5B/WP-6-8。DEC-045/047/048/049/050 构成完整证据链；重开此路线前必须先说明
  证据链哪里不成立。

### WP-5B：通用训练管线验收

在不人工指定慢 CV、不改变架构、cutoff、loss 权重、训练预算、外层 envelope、数值安全门
和验收门的条件下，将同一训练管线直接应用到至少两个额外 ABFE benchmark 体系或明确
不同的困难 ABFE 窗口。允许每个体系从标准化 pilot ledger 重新训练权重；这验证的是
**通用训练管线**，不是同一冻结权重的零样本迁移。

每个体系至少 3 个独立 production 重复，并同时满足：

- 全局端点严格等于原 MM；
- \(|\Delta G_{\rm residual}-\Delta G_{\rm converged\,MM}|
  <\max(0.5\ {\rm kcal/mol},2\sigma_{\rm combined})\)；
- ESS/GPU-hour 与 mutual overlap 均按 sealed 数值门改善；
- 不增加异常结构、force-tail 或循环不闭合；
- 无体系专属 CV、手调 cutoff、loss 权重或接受阈值。

实验 \(\Delta G\) 误差 `<1.0 kcal/mol` 作为次级报告指标，不作为路径势主正确性门，因为
端点归零的路径残差不能修复原 MM 力场的系统误差。

候选架构（shared trunk + 体系专属 head 等）已于 2026-08-05 设计存档，阻塞于 WP-5A
通过、当前不执行，完整设计见归档 §WP-5B候选架构。

### WP-5C：更强的冻结模型迁移（条件目标）

只有 WP-5B 通过后，才测试同一 frozen encoder/readout 权重跨体系不微调。必须与
“同超参数但逐体系重训”分开报告，不得把两种通用性合并宣称。

TYK2/CDK2 通常属于 RBFE benchmark。它们只能在 RBFE Hamiltonian、mapping、target
ledger、公共状态和端点协议通过独立接口资格后使用；不得直接把当前 ABFE 结果外推为
TYK2/CDK2 通用性证据。

## WP-6：2–4 个基势

只有 WP-5 证明单基势在一部分 λ 有效但低秩表达不足时启动。

- 从 \(M=2\) 开始；
- 最多 \(M=4\)；
- 使用低阶 Bernstein 或少结点 B-spline；
- 约束幅度和二阶导；
- 做 \(M=1,2,4\)、包络和去基势消融。

若 ESS/GPU-hour 没有可重复提升，退回单基势。

## WP-7：DEXP 迁移

前置条件：

- 无神经 DEXP 基线通过；
- DEXP 长程策略明确；
- complex/solvent 有独立重复；
- DEXP 端点和循环定义冻结。

迁移时只替换基础 interaction energy，不同时重训模型或改变外层函数。

## WP-8：完整 production

最低矩阵：

| 路径 | Complex | Solvent | 独立重复 |
|---|---:|---:|---:|
| 基础路径 | 是 | 是 | ≥3 |
| 仅 λ 重排 | 是 | 是 | ≥3 |
| 单神经基势 | 是 | 是 | ≥3 |
| 多神经基势 | 条件启用 | 条件启用 | ≥3 |

论文级结论建议每组 5 个独立重复。

## 8. 测试文件

当前独立测试：

```text
tests/test_outer_lambda_controller.py
tests/test_neural_basis_ibs_accounting.py
tests/test_outer_lambda_torchforce_standalone.py
tests/test_outer_lambda_torchforce_gpu_standalone.py
tests/test_outer_lambda_existing_api_compat.py
tests/test_outer_lambda_cli.py
```

纯 CPU 必跑：

- 包络、系数、端点；
- mock Force 能量和力；
- target/bias/base 账本；
- 公共状态一致；
- CV 上限；
- 配置和哈希；
- 禁用时旧行为不变；
- 缓存失效。

没有 OpenMM-Torch/MACE 时，相关测试必须明确 skip，不得误报通过。

截至 2026-07-31，独立集合最近一次结果为 `80 passed, 1 skipped`；skip 原因为当前
执行节点无 CUDA device，不是测试失败。正式 GPU 节点仍需运行 GPU 项。

GPU 验收：

- 小体系 10k–100k 步；
- 无非有限能量/力；
- 主/probe Context 共存；
- checkpoint 恢复后能量一致；
- 启用/禁用性能基准；
- 一个真实困难窗口短试验。

## 9. 缓存与协议

建议新增：

```text
NEURAL_PATH_PROTOCOL_VERSION
NEURAL_BASIS_MODEL_PROTOCOL_VERSION
NEURAL_PATH_ACCOUNTING_VERSION
```

以下变化必须使 production 缓存失效：

- enabled 开关；
- 模型内容、顺序或能量基准；
- 原子选择；
- 包络或系数；
- λ schedule；
- 基础势；
- target/bias 账本；
- 精度、周期盒或 Force 后端。

禁止只根据文件名判断模型相同。

## 10. 运行诊断

每个窗口至少记录：

- 每个基势能量均值、标准差和分位数；
- 每个状态的神经路径能量；
- 最大附加力和力 RMS；
- 支持域违规次数；
- 神经推理耗时、ns/day 和显存；
- target/bias/base finite gate；
- 端点神经能量最大绝对值；
- 公共 λ 系数哈希；
- ESS、absolute ESS 和 ESS/GPU-hour。
- 神经 Force 的调度方式、force group、MTS ratio 和实际物理评价间隔；
- 当前生产基势属于“完整 MACE”还是“蒸馏 cheap-CV”，以及教师模型身份；
- 相对 \(N=1\) 参考的积分偏差诊断。

## 11. 回滚

- `neural_path.enabled=false` 必须恢复旧系统。
- 神经结果使用独立输出目录。
- 不覆盖或删除基础路径缓存。
- 失败模型和失败原因必须保留记录。
- 回滚后运行旧基线回归。

## 12. 推荐提交批次

1. 外层控制器和纯数学测试；
2. mock Force 与 IBS 账本；
3. 缓存/provenance/旧行为回归；
4. TorchForce 最小部署；
5. 单真实基势和稳定性；
6. 单窗口科学对照；
7. 条件性的多基势；
8. 条件性的 DEXP 迁移；
9. 完整 production。

不得把 1–5 合并为一次大改。

## 13. 开工检查单

约 50 条已完成/已终止条目（WP-0 至 WP-4C 基线、EXP-010/011/EXP-012 前半段全部通过项）
已归档，见归档文件 §13。此处仅保留尚未完成/尚未关闭的项：

- [ ] `original_6a`（6 Å teacher）的 CUDA float32 对照仍 `BLOCKED_ON_VRAM`，等 6 Å 分支
  gradient checkpointing 实现后重试。两臂最终按 held-out gap variance、梯度稳定性、
  显存成本判决，不能仅凭 C1 通过就选 `derived_5a`。
- [ ] (d0-2) 最小架构候选冻结：typed atom embedding + 平滑 ligand–environment
  radial/contact 特征 + 至多 1–2 个轻量 interaction block + ligand-only 不变
  pooling + 有界标量 `B_student`；不是完整张量-equivariant MACE 式 student；只有
  这个最简单候选失败才升级。
- [ ] (d0-3) Teacher-target 协议冻结：与 (c) 同构的 leave-one-run-out（两条 run 拟合/
  训练，第三条 run 评估）；student loss 同时包含直接 gap 优化项和蒸馏项，teacher 不能
  被当成无条件 ground truth。
- [ ] (d0-4) 必需对照实验冻结：同一架构的 direct-gap student（无 teacher target）与
  distilled student（同架构 + teacher loss）都要训练，否则任何增益无法归因于 MACE
  teacher 本身。
- [~] (d0-5) 计算/部署预算冻结：**部分完成（DEC-039，2026-08-05）**。已冻结：最大参数量
  （≤50k 目标/≤100k 硬上限）、图规模（S1≤256/320 原子、边≤1536/2048、单原子
  neighbor≤64/80，1500 帧真实审计，`report_sha256 782e5824...4d33c114cf416f`）、
  CPU float64/CUDA float32 funnel 一致性（`report_sha256
  9671470e...ede385fa93cfcf934b`）、训练 seed（≥3/变体/折，硬下限）、早停规则（改用
  训练 run 内部末尾 20% 连续时间块做早停验证集，不复用被隔离的第三条 run 兼职早停+
  最终评估）、`max_epoch=500`、`early_stop_patience=30`、held-out 改善判据（3 折目标
  全部改善、硬下限 2/3、均值下降 >0%、最差单折不劣化超过 10%、蒸馏相对 direct-gap
  不得更差）。**尚未冻结**：目标每 MD 步毫秒数——生产基线 ms/step 待用户用修复后的
  `scripts/benchmark_exp012_no_student_window0_baseline.py` 重新测量。根因已定位：
  `output_lrc_fix/box_vectors.npy` 只在初次建系统缓存时写一次，真正建窗口 0 用的盒子
  在 NPT 弛豫和 Boresch rebalance 后被内存内重新赋值但从不写回磁盘；脚本已改为两阶段
  构造（先用陈旧盒子 `loadCheckpoint` 读回真实盒子，再用它重建用于哈希校验和计时的
  System），schema 升到 v2；**修复后的重新测量尚未执行**，在拿到
  `win_sys_xml_sha256_matches_manifest=true` 的新报告前，现有 v1 数字
  （median `1.3961`、P95 `1.3988` ms/step）仍只作为参考量级。GPU 显存上限（依赖 D3
  实测）与 student 侧允许 cutoff（待 D1/D2 验证，目前沿用 teacher 侧 5.0 Å）也待定。
- [x] (d3) 部署资格：TorchScript、OpenMM Reference、CUDA 一致性、耗时。**`CLOSED`**
  （DEC-043，2026-08-07）：

  | sub-item | 状态 | 证据 |
  |---|---|---|
  | 1. deployment 一致性（eager vs scripted，CPU64/CPU32/CUDA32） | **`PASSED_OPERATIONAL_NUMERICAL_EQUIVALENCE`** | `student_d3_1_2_report_v3.json`（report_sha256 `c8d940818111f0150c754690e2f519904166587effdfbe60f7383c696b8ca148`）：`.square()` 修复后真实重跑，`reference_eager_vs_deployable_eager` 残差 `1.6908728595e-07`（v2 用 `.pow(2)` 时是 `1.6908728728e-07`，只在第 10 位有效数字有差）——`.pow(2)`/`.square()` 假设被证伪，真正根因未查明、裁定不再追查；残差比已接受的 CPU64→CPU32 精度包络（`1.32e-5`）小约 2 个数量级、比已接受的 CPU32↔CUDA32 设备误差（`7.65e-7`）还小；`1e-8` 绝对阈值未进入 sealed preregistration，不再作为卡点 |
  | 2. TorchForce/OpenMM Reference 注入 | **通过** | `all_passed=true`，`torchforce_consistency` 误差 `0.000e+00`，report_sha256 `d71fe52e...4f66e5b0908c8bf88` |
  | 3. 端点归零 | **通过** | `endpoint_zeroing` 误差 `0.000e+00`（同一报告） |
  | 4. 生产 Context 耗时（correctness 部分） | **通过** | `tests/test_exp012_student_deploy_cell_list.py` 全部通过：cell list 与 brute force 找到完全相同候选边/距离（`<1e-10`）、完整前向+反向能量力一致（重构 `_energy_from_edges` 消除了初版测试用 box 扰动强制切分支引入的物理混淆变量后）、小 box 安全回退、TorchScript 导出后数值一致 |
  | 4. 生产 Context 耗时（performance 部分） | 非阻塞工程目标（§16） | all-pairs 近邻发现原耗时 7.68ms/call；替换为周期性 linked-cell list 后开销从 258% 降到 **95%**（report_sha256 `02ecc152...b93def9181d6abb1`），如实保留，不再是 D3/D4 前置门（`≤50%`/`≤15%` 未进入 sealed preregistration） |

  sub-item 4 的 profiling 脚本自身的 graph-construction 计时函数曾是替换前旧逻辑的手写
  镜像副本、未随 `student_deploy.py` 一起更新（测的是已废弃的旧代码），已修复为直接调用
  真实方法；由于 §16 已把 `95%→50%` 的优化降级为可选工程目标，修复后的 profiling 脚本
  重新定位新瓶颈不再是 D3/D4 前置条件，留作可选后续工作。详见 EXPERIMENT_LOG DEC-043。
- [x] (d4) 动力学资格：短 NVT、稳定性，再做独立重复。**`PASSED`（DEC-044，
  2026-08-07）**，详见第 7 节与 EXPERIMENT_LOG。
- [ ] L1/L2 性能对照及 TorchScript/OpenMM Reference/CUDA、短 NVT 性能资格（L1 目前降级，
  不是当前下一步，见 DEC-030）。
- [ ] WP-5A Atenolol 三重复体系内资格。
- [ ] WP-5B 至少两个额外 ABFE 体系的同协议通用性验收。
- [ ] RBFE 接口资格通过后，才考虑 TYK2/CDK2。

每个子阶段失败即停，不得跳阶段推进到下一个。

## 14. 完成定义

(d3) 部署资格已于 2026-08-07（DEC-043）判定 `CLOSED`：cell-list 等价性单元测试通过，
`.pow(2)`/`.square()` 假设已用真实数据证伪但不再追查根因，`reference_eager_vs_
deployable_eager` 残差裁定为 `PASSED_OPERATIONAL_NUMERICAL_EQUIVALENCE`（理由见第 7
节与 EXPERIMENT_LOG DEC-043，核心是该残差比已接受的 CPU64→CPU32/CPU32↔CUDA32 误差还
小两个数量级，`1e-8` 绝对阈值本身未进入 sealed preregistration）。(d4) 短 NVT 动力学
资格已于同日（DEC-044）首次执行即通过：student TorchForce 真实注入 `hard_window0`
win_sys 并驱动 3 个独立种子重复的实际积分，全程有限值、student 力/温度均在 sanity
门内。EXP-012 的 D0-D4 至此全部关闭。D4 只验证了"积分不会炸"，不验证"采样是否变好"，
后者是 WP-5A 的职责——WP-5A step 1/2（接线 smoke、候选冻结）已通过（DEC-047），
step 3（baseline/student 配对独立重复 pilot）已执行（DEC-048）：模型对 mixture ESS
有真实正向信号（3/3 重复提升 10.7-27.8%），但 ESS/GPU-hour 判据不通过（当前部署
成本涨幅补不上增益）。授权的一次部署性能救援（DEC-049，目标 `1.10×baseline`）经
终局实验（DEC-050）判定结构性不可达，**单基势"每步在线"部署路线已正式关闭**（只
关这一种部署模式，不代表否定模型本身或离线应用）。当前下一执行点是 **WP-4D /
EXP-013**（低频在线 student，MTS/rRESPA，把 student 评价频率而非部署实现本身作为
优化对象）。DEC-053 已确认方案③通过 013-A，但 DEC-055/056 已按预注册主门判定其
013-B `FAILED`；方案③禁止进入 013-C。当前执行方案① whole fused Group-1 slow 的
`N=1/2/4/8` 低成本预检，只有 Qualification gate 通过才考虑 N=16；方案① gate 未
通过才进入方案②，
再失败才转 EXP-014。WP-5B/6-8 仍全部阻塞，不直接推进。
（本段为 EXP-013 启动前的历史计划；已由 DEC-059/DEC-060 及当前最终执行边界取代，不能据此重新启动 EXP-013 或 WP-5。）

EXP-010/EXP-011 已分别终止为 `FAILED`/`FORMAL_RUN1_OVERLAP_FAILED`，不再补采、不再
拟合模型；完整历史见归档。未通过信息、守恒力、稳定性、ESS/GPU-hour 与跨 ABFE 通用性
门前，不得修改 production 模块或宣称模型通用。

只有同时满足以下条件，才可宣称方案在当前项目中“可实现并有效”：

1. 禁用神经路径时旧行为回归；
2. 端点能量和力与基础路径一致；
3. 神经项进入所有目标状态和 TMBAR；
4. IBS/WCA sampling bias 正确分离；
5. 公共 λ 状态跨窗口一致；
6. 模型和系数进入缓存指纹；
7. GPU 短模拟稳定；
8. 至少 3 个独立重复显示 ESS/GPU-hour 改善；
9. 改善不能由简单 λ 重排完全替代；
10. complex/solvent 完整循环与基础路径统计一致。

## 16. 2026-08-07 规范性澄清：D3 性能不作为 D4 的单独阻塞门

本节是当前有效规则，明确覆盖本文较早位置及 DEC-042 中把 `≤50%` 写成“冻结硬淘汰门”
的措辞。核对 sealed `protocols/EXP-012_preregistration.json` 后确认：其中没有 `≤15%`、
`≤50%`、per-step overhead 或同义的 deployment performance 数值门；旧预算表中的这些数字
原本标为 `PROPOSED`，并未进入 sealed preregistration。因此不得把当前约 `95%` overhead
解释为违反预注册硬门，也不得仅凭这一数字阻塞 D4。

当前规则如下：

1. `≤15%` 仅保留为工程优化目标；`≤50%` 仅保留为 preferred engineering ceiling，二者都
   不是 D3 correctness gate，也不是进入 D4 的必要条件。
2. 当前 cell-list 版本相对无 student production Context 的约 `95%` overhead 必须如实记录，
   不得删除或美化；但其是否值得，只能由独立 production 重复中的 ESS/GPU-hour 判决。
3. 最终性能硬门是：计入全部 wall-clock/GPU 成本后，student 路线的 ESS/GPU-hour 必须优于
   基础路径。若约 1.95× 单步成本没有被足够的 ESS 增益补偿，才判该路线性能失败。
4. D3 继续保留真正的正确性门：cell-list 与 canonical/brute-force funnel 的 edge、PBC shift、
   energy、force 等价性；eager/TorchScript 与 CPU/CUDA 的 dtype-aware 一致性；TorchForce/
   OpenMM 注入；端点严格归零；有限值与显存可运行性。这些门未通过仍不得进入 D4。
5. `.square()` 修复后的真实数据重跑和新增 cell-list 单元测试属于必要 correctness regression，
   必须完成；进一步 profiling 和把 overhead 从 95% 优化到 50% 以下是可选工程工作，不再是
   D4 前置条件。
6. 上述 correctness regression 全部通过后即可关闭 D3 并进入 D4 短 NVT；D4/WP-5A 才负责
   测量稳定性、采样改善和最终 ESS/GPU-hour。不得再次以旧的 `≤50%` 文本单独停止推进。

因此，§7/§13/§14 中“95% 高于 ≤50%，故 D3 未通过/不得进入 D4”的旧句均由本节取代；
它们只保留为 DEC-042 当时的历史判断，不再是当前执行规则。D3 当前真正剩余的前置工作仅为：
运行 cell-list 等价性单元测试，并在真实数据上重跑 `.square()` 后的 deployment 一致性检查。
