# 外层 λ 神经基势详细实施计划

> **文档角色：工程执行计划。** 科学原则见
> [`PLAN_outer_lambda_neural_basis.md`](PLAN_outer_lambda_neural_basis.md)，真实运行结果记录在
> [`EXPERIMENT_LOG_outer_lambda_neural_basis.md`](EXPERIMENT_LOG_outer_lambda_neural_basis.md)。
> 本文同时维护任务状态；设计项不等于 production 已接入。

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

### 1.1 截至 2026-08-03 的工程状态

| 工作包 | 状态 | 当前证据/边界 |
|---|---|---|
| WP-0 | 完成 | window 0、primary torsion 和两个 secondary/diagnostic 候选已登记 |
| WP-1 | 完成 | 外层包络、系数、端点归零、协议哈希 |
| WP-2 | 完成 | mock Force 与 IBS target/bias/base 账本契约 |
| WP-3 | 完成（通用骨架） | TorchForce、CustomCVForce、序列化、CPU/CUDA/checkpoint 接口 |
| WP-3A | 失败并停止 | EXP-009 的 PythonForce/CUDA MTS 后端在 \(N=1\) 失败 |
| WP-4 | 完成直接 MACE qualification | `coefficient=0.09` 的 EXP-007 通过；不等于可接受的生产成本 |
| WP-4A / EXP-010 | `FAILED` | atom-cut protein MACE 教师的 290-frame 数据集完成；教师边界无物理闭合且六候选跨 run 验证失败 |
| WP-4B / EXP-011 | `FAILED / STOPPED` | AUG-001 后 mutual overlap `0.02353 < 0.03` 且 22 个去相关样本 `< 25`；不再补采、不拟合 PMF |
| WP-4C / EXP-012 | `PLANNED / PREREG_DRAFT_V2 / L2_FORMALIZED(DEC-030) / derived_5a_C1_CPU+CUDA_PASSED / original_6a_C1_CPU_PASSED_CUDA_BLOCKED_ON_VRAM` | 路线正式定性为 L2（DEC-030）：`original_6a`/`derived_5a` 是离线 teacher，`LocalResidualStudent`（尚不存在代码）是唯一在线模型，L1 降级为非当前下一步。五态 ledger/backend audit 已通过；两个显式候选臂（DEC-027）：`original_6a`（6 Å，2135 节点/155624 边）CPU C1 通过，CUDA 同帧对照三次尝试均在同一 product-layer-1 张量积算子 OOM（真实峰值需求 `>=~24.07 GiB`，已排除碎片化），6 Å 分支 gradient checkpointing 未实现（与 teacher/student 分工无关，独立线程）；`derived_5a`（5 Å，1444 节点/60048 边）CPU 与 CUDA C1 均已通过（DEC-028/029，CPU↔CUDA 相对差 ~1e-7、CUDA 无 OOM），是首个完整通过 C1 的候选臂。两臂最终仍按 held-out gap variance/梯度稳定性/显存成本判决，不能仅凭 C1 通过就选 `derived_5a`。下一步顺序：多帧支持域审计（report-only）→ `derived_5a` 离线 latent cache → cached-latent 线性/ridge readout 的 held-out gap-variance 验证 → 有增益才蒸馏 `LocalResidualStudent` → student 在线力学/性能资格；仍不得训练 |
| WP-5 | 未开始 | EXP-012 增量信息、守恒力、TorchForce/CUDA 和独立重复资格通过前不启动 |
| WP-6–8 | 未开始 | 由 WP-5 结果条件触发 |

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
| `local_residual/` | EXP-012 v2 主命名空间；当前别名复用表示无关的 schema、MM ledger、ledger audit 与 metrics，production 不导入 | 后续承载 A/B/C/D、MACE latent adapter、student、训练和部署实现 |
| `exp012_xed/` | DEC-018 早期兼容/证据命名空间；现有 ledger/schema 实现继续可复现 | 不再定义方法身份；XED 只允许作为 Arm D 可选消融 |
| `protocols/EXP-012_preregistration.json` | `exp012-local-residual-prereg-v2` draft；已登记三条 CUDA ledger/report SHA、统一权重口径、whole-run folds 与 ledger audit | 补齐全局 `A_k`、A/B/C/D 表示、图边界、readout、训练预算和数值判决门后重新哈希并 sealed |
| `MaceLatentBasisAdapter`（C1 底层接口） | 合成 OFF24 合约与 Atenolol 真帧 OMOL CPU smoke 均通过；OMOL 明确 early-stop 于 zero-based product layer 1，输出该层首个 `1024x0e` 标量块 | CUDA 同帧通过后再接 invariant scalar readout |
| `LocalResidualStudent`（计划） | 尚不存在 | 轻量等变 ligand-environment cross encoder；既可独立训练，也可承接 MACE-latent teacher 蒸馏 |
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

状态：`COMPLETED`。

冻结结果：

- complex vanishing Stage 2 window 0，states `[0,5)`；
- primary torsion `[4591,4592,4593,4585]`，三种子 core transitions
  `14/4/10`；
- secondary torsion `[4593,4585,4594,4595]`；
- `VAL251 chi1` 与 hydration 仅保留作诊断。

任务：

- 固定一个小型测试体系和一个真实困难 Stage 2 ensemble。
- 指定唯一首个目标慢自由度。
- 保存基础路径 ESS、转换次数、异常率和 GPU 性能。
- 验证简单 λ 重排是否已经足够。

通过门：

- 基线可重复；
- 瓶颈明确；
- 不是单纯增加采样就能解决。

## WP-1：外层控制器纯数学测试

状态：`COMPLETED`（独立模块）。

任务：

- 实现 `sin2 + constant coefficient`；
- 生成 \(A_{km}\)；
- 验证端点和有限性；
- 生成稳定协议 payload。

必须测试：

- \(A_{0m}=A_{Km}=0\)；
- 相同 λ 的系数逐位相同；
- 非有限参数 fail closed；
- 任一系数变化都会改变协议哈希。

通过门：全部 CPU 测试通过，不加载 MACE/TorchForce。

## WP-2：解析 mock 基势的 IBS 账本

状态：`COMPLETED`（独立模块，尚未合入 production IBS）。

先用简单谐波或原子对势，不使用神经网络。

必须验证：

1. 禁用路径时旧行为不变。
2. λ=0、1 能量和力与基础路径一致。
3. 中间态增量等于解析值。
4. target energies 包含路径项。
5. `bias_history` 不消除路径项。
6. 公共 λ 状态跨窗口完全一致。
7. TMBAR 使用修改后能量。
8. 不同模型/系数不能复用旧缓存。

通过门：无 ML 依赖即可证明端点和统计账本闭合。

## WP-3：TorchForce 最小部署

状态：`COMPLETED`（通用独立部署骨架）。

使用极小 TorchScript 标量模型验证：

- `TorchForce` 单独运行；
- 作为 `CustomCVForce` 的共享 CV；
- 坐标梯度；
- 周期盒；
- XML 序列化或可靠重建；
- 主 Context 与 probe Context 共存；
- CPU/CUDA 一致性；
- checkpoint 恢复；
- 单步时间和显存。

若嵌套、复制或序列化失败，停止 production 接入，先更换明确的重建方案或后端。

### WP-3A：神经 Force 多时间步调度

状态：`FAILED / STOPPED`。EXP-009 在相同 MTS 后端的 \(N=1\) 已触发
`CUDA_ERROR_INVALID_HANDLE`。当前 openmmml 1.6 和实时桥接均依赖
`PythonForce`，因此不再执行 \(N=2/4/8\)，也不通过放宽 coefficient 或换节点重复
同一后端。以下矩阵保留为历史预注册协议，不能误写成尚待运行。

完整 MACE 只有在作为独立 force group 接入 MTS/r-RESPA 后，才允许降低评价频率。
禁止在普通积分循环外“每 \(N\) 步更新一次，然后冻结旧力”。

EXP-007 已冻结外层系数 0.09。降低系数不会减少单次 MACE 推理成本，因此性能实验
不得同时扫描系数和更新时间；第一轮只比较 MTS ratio：

| 外层系数 | MTS ratio \(N\) | 当前 0.5 fs 内步长下的 MACE 间隔 | 用途 |
|---:|---:|---:|---|
| 0.09 | 1 | 0.5 fs | 同一 MTS 实现内的参考 |
| 0.09 | 2 | 1 fs | 第一档降频 |
| 0.09 | 4 | 2 fs | 第二档降频 |

只有 \(N=2,4\) 相对 \(N=1\) 均通过后，才探索 \(N=8,16\)（4、8 fs）。若以后
恢复 2 fs 内步长，必须重新登记物理时间，不得直接复用上述 ratio 结论。若未来重新
改变系数，应作为新的幅度协议先完成独立力学资格，不能混入本性能矩阵。

每个组合至少比较：

- 温度和势能分布；
- path-only 能量与最大原子力；
- 约束错误、NaN 和异常结构；
- 同初态短轨迹的能量漂移或 shadow-work 代理；
- 目标慢变量分布和转换；
- wall time、ns/day、显存和预测 ESS/GPU-hour。

MTS 通过门不能只看“模拟没有崩溃”。相对 \(N=1\) 参考，必须没有可分辨的结构分布
偏移或系统性积分偏差。

## WP-4：单个真实任务化基势

状态：直接完整 MACE 的 EXP-007 qualification 已通过；由于成本和 EXP-009 后端失败，
它只保留为教师。

第一候选范围：

- ligand；
- 固定关键口袋残基；
- 暂不加入交换水和离子；
- 只针对一个 torsion、rotamer 或接触重组。

EXP-010 实际选择为 ligand 41 原子 + 固定 protein environment 216 原子。旧环境中的
39 个水原子涉及 14 个水残基且包含不完整水，已从教师选择中全部移除。该变化生成新的
选择哈希，不复用含水选择的资格结论。

模型要求：

- 标量能量和守恒力；
- 固定粒子映射；
- 元素覆盖明确；
- 平滑有界输出；
- 支持域检测；
- 模型和训练数据哈希固定。

通过门：

- 代表性轨迹上无 NaN 和非物理大力；
- 短 NVT 稳定；
- 推理成本允许进入单窗口试验。

### WP-4A：直接 MACE 与廉价蒸馏的双路线

当前决策：直接 MACE 路线结束；只执行廉价蒸馏路线。

历史直接 MACE 路线（EXP-009 后已关闭）：

1. 完成固定原子身份、PBC、support domain 和能量 offset；
2. 先通过 \(N=1\) 力学资格；
3. 再执行 WP-3A 的 MTS 矩阵；
4. 只有预测 ESS/GPU-hour 仍有竞争力时才进入 WP-5。

廉价蒸馏路线：

1. 用冻结 MACE 给独立训练轨迹标注局部能量、力和 descriptor；
2. 预先指定一个目标慢变量 \(s(\mathbf R)\)，例如 torsion、rotamer、
   coordination number 或 hydration number；
3. 检验 MACE 输出中与该慢变量相关的可解释分量；
4. 拟合 1D/2D spline、tabulated bias、低阶多项式或小型 MLP
   \(V_\phi(s)\)；
5. 冻结模型、训练集哈希、支持区间和外推衰减；
6. cheap bias 每个生产积分步计算，MACE 不再进入每个 MD step。

EXP-010 已实现的具体协议：

1. 从三个独立困难窗口 scratch run 各选 100 帧，共 300 个 source frames；
2. 保持原支持域 `min pair >= 0.07 nm`、`max pair <= 2.5 nm`、
   `Rg <= 0.85 nm`；
3. 支持域外帧在 MACE 前排除并登记，最大允许排除率 5%；预检为 10/300；
4. MACE 教师输出局部 interaction energy、最大原子力和 primary torsion 广义力；
5. energy offset 固定为全部合格训练帧的均值，只移除常数，不调整
   `coefficient=0.09`；
6. 候选矩阵固定为 1D Fourier order 2/4/6 与 2D Fourier order 2/3/4；
7. 验证按整条 run 留一，不允许随机拆帧造成轨迹泄漏；
8. 最终 Force 使用 `CustomTorsionForce` 或 `CustomCompoundBondForce`，不含
   Torch、MACE 或 `PythonForce`。

实际结果：290/300 帧通过支持域门，排除率 3.333%。intercept-only 能量 RMSE 为
`21.5109 kJ/mol`；最佳候选 1D order 2 的 leave-one-run-out 能量 RMSE 为
`22.1737 kJ/mol`，广义力 \(R^2=-13.5934\)。其余 1D 候选更差，2D 候选出现严重
病态拟合。因此没有候选被冻结为最终 Force，EXP-010 记为 `FAILED`。

事后选择完整性审计进一步发现，216 个 protein atoms 涉及 26 个残基，但完整残基为
0；原选择器是单帧 0.5 nm 逐原子半径选择。这一人工断键簇的分解能量只在
冻结原子集的代数定义上闭合，不能当作完整蛋白环境下的物理 interaction energy。
因此 EXP-010 失败不用于排除 primary torsion，而是排除当前教师构造和逐帧能量目标。

### WP-4B：EXP-011 完整 MM 条件平均力/PMF

2026-08-02 准备状态：已新增 `exp011-coverage`、`exp011-fit-pmf`、
`protocols/EXP-011_preregistration.json` 和
`docs/experiments/EXP-011_PREREGISTRATION.md`。历史三 run 覆盖报告位于
`output/outer_lambda_exp011/coverage_report_v2.json`。EXP-011 manifest 状态为
`frozen_for_exp011_complete_mm_pmf`，primary CV 为 `[4591,4592,4593,4585]`，且
`production_approval=false`。覆盖报告未通过硬门，故当前没有
PMF model、OpenMM candidate 或 NVT 资格结果；这不是 EXP-011 正式 PMF 的失败结论。

2026-08-02 已新增：

1. `exp011-umbrella-sample`：在完整 window-0 MM expanded-mixture System 上增加周期
   torsion restraint，使用独立输出目录，逐帧记录角度、restraint energy、中心、力常数、
   seed 和 System/protocol 哈希；
2. `exp011-reweight-umbrella`：汇总各 umbrella window，构造跨窗口 reduced-potential
   矩阵，用 MBAR 输出每帧显式 `log_target_weight` 和 overlap/连通性报告；
3. 周期最短角差、MBAR 显式权重、overlap 连通和 CLI 回归测试。

普通 `sample-hard-window-scratch` 没有 torsion restraint 和重加权账本，不得冒充上述
采样。相关两个测试文件当前为 `58 passed`。

完整体系 smoke 结果：`--minimize-max-iterations 0` 的 Reference 诊断在第一积分步出现
NaN，因此跳过最小化明确不合格；随后在可用 GPU 环境完成 `200` 次最小化的正式 smoke。
`output/outer_lambda_exp011/cuda_smoke_center_m172p5/report.json` 报告 `ok=true`、
`platform=CUDA`、1 个样本、angle `-173.5426°`、umbrella energy `0.01656 kJ/mol`、
temperature `278.03 K`，checkpoint 与 DCD 完整。下一执行点是同一中心的短时稳定性 pilot，
确认多帧均有限且 restraint 后再扩展中心。

同一中心 10 ps 稳定性 pilot 已通过：10/10 帧有限，temperature `298.12–302.11 K`，
angle `-173.12°` 至 `-158.97°`，最大 umbrella energy `2.79 kJ/mol`，报告见
`output/outer_lambda_exp011/pilot_run1_center_m172p5/report.json`。这只批准运行相邻的
`-157.5°` 中心并检查两窗 overlap；不批准直接生成 PMF 或批量运行 24 centers。

`-157.5°` pilot 已通过，两窗 MBAR 的邻窗 overlap 为 `0.3584`（门为 `0.03`），各窗
保留 10/10 样本，局部 overlap 图连通。`qualified_for_pmf_input=true` 的范围仅限这两个
局部窗口，不表示 24-bin 周期覆盖、正式平衡性或 PMF 验收完成。下一执行点为第三个相邻
中心 `-142.5°`，之后重新检查三窗 overlap。

第三窗及三窗 MBAR 已通过：相邻 overlap 为 `0.3105` 与 `0.1864/0.3728`，均超过
`0.03`。第三窗只有 5/10 个去相关样本（`g=1.943`），所以短 pilot 不晋级为正式 PMF
数据。15° spacing 与 `k=100` 已通过局部执行资格；在冻结正式每窗长度前，先运行历史
空白区 `75°–165°` 中部的 `112.5°` 哨兵窗，确认从共同初态到远端中心的可达性。

`112.5°` 哨兵窗已到达目标区：10/10 帧有限，angle `93.22°–112.21°`，最后一帧
`112.21°`，最大 umbrella energy `5.66 kJ/mol`。由于样本分布偏向中心低侧，尚不能冻结
全套正式采样；先运行高侧相邻 `127.5°` 并检查该空白区界面的 MBAR overlap。

空白区界面 overlap 已通过：正反向为 `0.0759/0.0949`，但仅保留 5/10 与 4/10 个
去相关样本，故 pilot 不作为正式 PMF 数据。正式采样机器协议现已冻结在
`protocols/EXP-011_umbrella_sampling_plan.json`（内部 SHA-256
`1cd78aba12f15b52a52f27b5f6c8980544843887ebc2cf84d1b5bc12660c6912`）：24 centers ×
3 replicates，每窗 50 ps burn-in、100 ps sampling、1 ps 报告间隔，并使用三条不同历史
困难窗口轨迹作为 replicate 初态。`scripts/run_exp011_umbrella_grid.py` 默认仅跑一个 pending
window，验证计划哈希、已有报告参数、初态轨迹哈希及非空残缺目录；dry-run 已通过。

主路线不使用局部 MACE 能量标签。实施顺序：

1. 对三条困难窗口 run 生成 primary torsion 周期 bin 覆盖、重叠和有效样本数报告；
2. 覆盖不足时，新建受限/umbrella 或其他增强采样，不用空 bin 拟合；
3. 从完整 MM Hamiltonian 获得条件平均力或 PMF，包含完整蛋白、水和现有约束；
4. 执行整条 run 留一，验证平均力方向、PMF 形状和周期积分闭合；
5. 在查看正式结果前冻结 spline/Fourier 候选、平滑度、幅度与验收门；
6. 只有跨 run 验证通过后，才导出周期 OpenMM Force 并执行独立 NVT qualification。

可选 MACE 教师不属于 EXP-011 主路线。如需重启，必须另立协议，使用完整残基、
backbone buffer 和封端/局部 readout，并对多个环境半径进行能量与 ligand-force 收敛检查。

蒸馏验收不能只看训练 RMSE，还必须检查：

- 沿慢变量的平均力/自由能形状；
- 独立轨迹上的能量排序和力方向；
- 支持区间外是否平滑衰减；
- 端点归零和 IBS target/bias 账本；
- 相对于直接 MACE 的慢变量采样收益与 ESS/GPU-hour。

若直接 MACE 只有在外层间隔大于已验证稳定范围时才有可接受成本，则停止直接生产
路线，保留 MACE 作为教师/分析器。

### WP-4C：EXP-012 CV-free 通用局部残差路径势

状态：`PLANNED / PREREG_DRAFT_V2`。三条 scratch DCD 的逐帧五态完整 target-state ledger
已经用统一 CUDA backend 完成；三条 arrays/report SHA、scratch System SHA、force-group/IBS
闭合以及 run1 CPU/CUDA 公共帧一致性均通过机器审计。协议主语义和导入入口已迁移为
`local_residual` A/B/C/D，`exp012_xed` 保留作 DEC-018 兼容证据。当前仍禁止训练、feature
判决或 production Force，因为 (A_k)、表示细节、MACE layer/图边界、readout、训练预算
和数值判决门尚未冻结，preregistration 仍为 draft。

`ExistingOrbMaceBasisAdapter` / `MaceDecompositionPythonComputation` 继续只作 EXP-010
历史证据。新实现必须建立 `MaceLatentBasisAdapter`，读取 frozen MACE 中间层 node
features，而不是最终 interaction energy 或三次 fragment subtraction。
#### WP-4C.1 数据与训练目标

对相邻状态和 frame \(a\) 定义：

\[
\delta_{ak}(\theta)=\Delta u^0_{ak}
+\beta(A_{k+1}-A_k)B_\theta^{\rm local}(\mathbf R_a).
\]

第一轮固定全局 \(A_k\)，只训练 encoder/readout：

\[
\mathcal L_{\rm gap}
=\sum_k\frac12\left[
\operatorname{Var}_{p_k}(\delta_k)
+\operatorname{Var}_{p_{k+1}}(\delta_k)
\right]
+\lambda_E\langle B_\theta^2\rangle
+\lambda_F\langle\|\nabla B_\theta\|^2\rangle.
\]

必须使用完整 MM target-state ledger、target/MBAR 权重和 whole-run holdout。禁止随机拆帧，
禁止第一轮联合训练 \(A_k\) 与 \(B_\theta\)，禁止以训练集 ESS 选模型。ESS、BAR mutual
overlap、force tails 和 ESS/GPU-hour 均在 held-out/独立运行上判决。

#### WP-4C.2 表示消融

| Arm | 表示 | 判决目的 |
|---|---|---|
| A | typed atom-centered RBF/contact | 最低复杂度和解析成本基线 |
| B | 轻量等变 ligand-environment cross encoder | 判断通用角向/多体表示是否足够 |
| C | frozen-MACE latent + 小型 invariant MLP | 判断 pretrained latent 是否有 held-out 增量 |
| D | XED-inspired field（可选） | 仅判断物理启发特征是否在 A–C 之外有增量 |

Arm C 必须冻结 MACE encoder 权重，明确选择 layer、invariant/equivariant channels、ligand
node pooling 和 readout。环境图需冻结 cutoff、候选原子、message-passing receptive-field
buffer、PBC、元素/电荷支持域和边界失败策略。MACE latent 不能解释为显式孤对电子或
Pauli energy。

#### WP-4C.3 两种 MACE 执行路线

- **L1 在线 encoder：** `FrozenMaceEncoder -> node features -> MLP -> B` 全部位于同一
  Torch autograd 图中，直接产生 ligand 和 environment 守恒力。不能调用返回 NumPy 的
  descriptor API 后把特征当常数。
- **L2 离线 teacher：** frozen MACE latent 用于表示诊断/蒸馏，production 运行
  `LocalResidualStudent`。若 L1 的 ESS/GPU-hour 无竞争力而 L2 保留统计收益，优先 L2。

“小 MLP”不等于“便宜模型”；L1 每步仍包含 MACE forward/backward。两路线必须使用
相同 frame folds、gap loss 和判决口径，并报告 feature 构造、host/device copy 与完整
OpenMM step 成本。

**现状（DEC-030，2026-08-04）**：本轮正式定性为 L2。`original_6a`/`derived_5a`
是离线 teacher（不进 MD 每步，不需要过 OpenMM/NVT/ns-day 门），`LocalResidualStudent`
是唯一计划中的在线模型（尚不存在代码）。这不是绕开上面"必须实际比较 L1/L2
ESS/GPU-hour"的要求：本 session 已测得 teacher 侧单帧离线 latent 提取成本为
`derived_5a` CUDA 19.8s、`original_6a` CPU 172.7s，而每 MD 步预算需要 O(ms) 级，
两者差 3–4 个数量级，L1 在当前证据下已不可行，与 student 成本无关。L1 定义保留
在上面，不删除，只是不再是当前下一步。详见 EXPERIMENT_LOG DEC-030。

#### WP-4C.4 力学、部署和安全门

复用 `OuterLambdaController`、shared-CV/ledger、模型 SHA-256、TorchForce/OpenMM 注入和
benchmark harness。资格顺序：

1. 补齐三条 run 的逐帧五态 ledger，重新校验坐标/能量/状态顺序哈希；
2. 冻结 A/B/C/D、MACE layer/readout、图边界、(A_k)、训练预算和判决门并 seal；
3. 对 ligand/environment 坐标分别做 autograd/finite-difference force check；
4. TorchScript 与 OpenMM Reference 能量/力一致，XML round-trip 和全局端点归零；
5. CPU/CUDA 一致，单困难窗口短 NVT 稳定并记录 ns/day；
6. 至少 3 个独立重复通过后才允许 WP-5 通用性验证。

运行时检查 finite、\(|A_kB|\)、\(|A_k\nabla B|\)、图支持域、公共 λ 一致性和协议哈希。
第一版不使用既有 PythonForce/CUDA MTS 后端，不为每个 λ 复制 encoder。

停止条件：所有表示均不能改善 held-out gap/overlap；Arm C 对 A/B 无增量；在线 MACE
成本抵消 ESS；student 蒸馏不保留收益；图边界/环境半径不收敛；出现非守恒力、端点漂移、
TorchScript/CUDA 不一致、短 NVT 不稳定，或需要未预注册地放宽门限。
## WP-5：CV-free 局部残差的体系内与跨体系验收

### WP-5A：Atenolol 困难窗口体系内资格

固定四组对照：

1. 原始基础路径；
2. 仅 λ 重排；
3. 最简单 Arm A/解析接触基线；
4. EXP-012 选出的单局部残差势。

至少 3 个独立重复。主指标为 BAR mutual overlap、absolute/importance ESS、ESS/GPU-hour、
能量/力分位数、异常结构率和 converged-MM 自由能一致性。晋级要求端点与账本闭合、
稳定性不劣化、统计收益不能由 λ 重排或 Arm A 完全替代。

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

- [x] 已选择 softcore 原型起步。
- [x] 已指定 complex vanishing window 0。
- [x] 已冻结 primary torsion。
- [x] 已保存基础 ESS benchmark 和三条 scratch 慢变量轨迹。
- [x] 已定义教师能量为局部 MACE decomposition interaction energy。
- [x] 已确认不是未经处理的全体系 MACE 总能量。
- [x] 已冻结 protein-only 教师原子集合并生成选择哈希。
- [x] 已定义端点容差。
- [x] 已定义能量、力和支持域安全门。
- [x] 已定义模型/配置/缓存哈希。
- [x] 已准备无 ML 的 mock Force。
- [x] 已准备独立输出目录。
- [x] 已决定完整 MACE 仅作为教师。
- [x] MTS 的内步长、ratio 和失败后端已记录；该路线已停止。
- [x] EXP-010 正式 GPU 教师数据集已完成并通过支持域门。
- [x] 六个预注册 cheap-CV 候选已按整条 run 留一比较；全部失败，未冻结最终模型。
- [x] EXP-010 protein selection 已完成残基完整性审计；26/26 个涉及残基均为部分截断。
- [x] EXP-011 周期 CV 覆盖、目标加权 PMF 定义和跨 run 门已预注册；历史覆盖诊断要求增加受限/增强采样。
- [x] EXP-011 CV manifest 已正确标记为 `frozen_for_exp011_complete_mm_pmf`，内部哈希有效。
- [x] `exp011-umbrella-sample` 周期 restraint、独立输出和逐帧 bias-energy 账本实现。
- [x] `exp011-reweight-umbrella` MBAR 权重、overlap 连通性与 `target_samples.json` 导出。
- [x] 周期 restraint、MBAR 纯数值和 CLI 回归测试（58 passed）。
- [x] 有 GPU 节点上的完整体系单中心 smoke（正式最小化、1 step、报告/checkpoint/DCD 完整）。
- [x] 同一中心短时稳定性 pilot 与多帧有限值检查（10/10 帧有限）。
- [x] 相邻 `-157.5°` 中心 pilot 与两窗 MBAR overlap 检查（overlap `0.3584`）。
- [x] 第三个 `-142.5°` 中心 pilot 与三窗 MBAR overlap 检查。
- [x] 历史空白区 `112.5°` 哨兵窗的可达性与稳定性检查。
- [x] 空白区高侧 `127.5°` 邻窗及 `112.5°/127.5°` MBAR overlap。
- [x] 24 centers × 3 replicates 正式采样计划与 fail-closed 断点续跑入口冻结。
- [x] formal_run1 单窗正式 smoke（100/100 帧有限，初态哈希与 resume 校验通过）。
- [x] formal_run1 剩余 23 窗完成（24/24 窗、2400 帧有限）。
- [x] MBAR 验收语义修正为 mutual overlap + 每个周期邻接接口，schema v2（58 passed）。
- [ ] formal_run1 环形 MBAR overlap：AUG-001 后唯一失败接口仍为 `0.02353 < 0.03`。
- [x] EXP-011-AUG-001 已补采 `127.5°` 500 ps 并重跑严格 MBAR；22 个去相关样本与
  `0.02353` mutual overlap 均未达到冻结门，结论为 `COMPLETED_NOT_ACCEPTED`。
- [x] EXP-011 已在 AUG-001 后冻结为 `FORMAL_RUN1_OVERLAP_FAILED`；不再补采或拟合 PMF。
- [x] EXP-012 三条逐帧五态 target ledger、SHA/System/账本闭合与 CPU/CUDA backend audit。
- [x] C1 合成图 graph/latent/autograd 合约：真实 OFF24、严格 `[512:640]`、两类坐标非零梯度、MACE 参数零梯度；联合回归 `107 passed, 2 skipped`。
- [x] 为 `run1/frame0` CPU smoke 生成并审计 Atenolol 两跳 provisional environment manifest 与 atom mapping。
  第一次尝试（`pocket_cutoff_nm=0.5`、`hard_window0_run1/2/3` 末帧并集，sha `ffa52ebc...254f05`
  / `2c74e5e0...86597a`）已作废：`run1/frame0` CPU smoke 在 `build_mace_graph` 两跳完整性检查处
  失败。DEC-023 将其误诊为“必须覆盖 ligand-centered 12 Å 球”，该结论已由 DEC-024 撤销。
  正确支持域是 6 Å 邻接图的两跳闭包：`S0=L`、`S1=N_6Å(S0)`、`S2=N_6Å(S1)`；12 Å
  只作为几何上界，不能直接用于径向选原子。discovery 与 smoke 现共用
  `topology_n_hop_closure`，多参考帧闭包取并集、触及的环境残基整残基纳入。需要用
  `--edge-cutoff-angstrom 6.0 --interaction-layers 2`，并把 smoke 所用 frame 纳入参考帧后
  重新生成 provisional config/manifest/mapping。当前 frame0-only 产物位于
  `output/outer_lambda_exp012/two_hop_frame0/`：manifest canonical SHA
  `0e399a9e...e223935`、mapping canonical SHA `d84a65f9...88d00f`；它们已通过同帧 CPU smoke，
  但仍是 provisional，不能替代后续多参考帧冻结身份。
  旧错误算法与结论已存档到 `archive/exp012_radial_support_bug_20260803.md`。
- [x] `run1/frame0` 真实 CPU float32 latent/autograd smoke：OMOL extra-large-1024、product layer 1
  early-stop、两层精确闭包、2135 节点/155624 有向边、latent `[41,1024]`；ligand/environment
  梯度 norm `32.2197/22.2791`，repeat 最大差 `0`，MACE 参数梯度数 `0`，报告 SHA
  `ce8fd06c...58db5`。该结果只通过 C1 CPU 可导性，不是性能或科学资格。
- [ ] 同一 frame0、同一 manifest/mapping、同一 product layer 1 的 CUDA float32 对照。
  `BLOCKED_ON_VRAM`：三次尝试（15.47/10.57/23.58 GiB 卡）均在同一 product-layer-1
  张量积 `cat`/`reshape` 算子 OOM；加 `torch.cuda.empty_cache()`（`mace_latent.py`
  no-grad 参考 forward 后）与 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  后已占用显存数字不变（14.57 对 14.59 GiB），排除碎片化；真实峰值需求
  `>=~24.07 GiB`。同时修复代码缺陷：`MaceLatentBasisAdapter.forward` 对不带 index
  的 `--device cuda` 与实际张量具体 device 比较不相等（`mace_latent.py:223-225`
  已归一化修复）。详见 EXPERIMENT_LOG §11A.11、DEC-026。
- [x] DEC-027/028：`original_6a`（6 Å）与 `derived_5a`（5 Å）并列候选臂已双双通过 CPU C1。
  `smoke_exp012_mace_latent.py` 只接受 `--edge-cutoff-angstrom` ∈ `{6.0, 5.0}`，报告含
  `encoder_variant`/`model_r_max_angstrom`/`graph_cutoff_angstrom`/
  `original_encoder_numerically_preserved` 四字段与 `--output` 防覆盖门；`mace_graph.py`
  写死的 “6-Angstrom” 注释/报错已参数化。`derived_5a` CPU smoke（报告 SHA
  `d28be435...9c7a0d`）：两跳精确闭包 974 原子（hop 0/1/2=`41/219/714`），整残基收口后
  1444 节点/60048 边（对比 `original_6a` 的 2135 节点/155624 边，边数降到 38.6%），
  float32 latent `[41,1024]` 有限，ligand/environment 梯度 norm `32.9454/19.9347`，
  repeat 差 `0`，参数梯度数 `0`，CPU 耗时 34.5s（对比 172.7s）。
- [x] `derived_5a` 的 CUDA float32 对照（DEC-029，报告 SHA `ba7d053d...deb37bc8d59`）：
  无 OOM，23.87s；CPU↔CUDA 相对差 latent/ligand-grad/env-grad norm 分别为
  `7.53e-7`/`1.16e-7`/`-3.83e-7`，均为 float32 舍入量级。`derived_5a` 是首个完整通过
  C1（CPU+CUDA）的候选臂。
- [ ] `original_6a` 的 CUDA float32 对照仍 `BLOCKED_ON_VRAM`，等 6 Å 分支 gradient
  checkpointing 实现后重试。两臂最终按 held-out gap variance、梯度稳定性、显存成本判决，
  不能仅凭 C1 通过就选 `derived_5a`。
- [x] DEC-030(a) `derived_5a` 多帧支持域审计，report-only，不设硬门：已针对三条
  真实 run（1500 帧）执行，report_sha256 `a74ea2352263ea9b25e324c9d0930a0199b0fb826d453ab2715b54cd82cf9b69`。
  结果：1499/1500 帧（99.9%）违规，最坏单帧遗漏 122–141/1444 个固定原子。分解：
  遗漏原子约 86–88% 是水（正常——水在两跳壳层内持续扩散进出，非缺陷）；但有约 24 个
  非水残基（ALA50/63/131/168、ARG167、ASN166、ASP246、CYS154/160/227、ILE233、
  LEU57/142/182/229/256、PHE77、PRO230、TRP4/130、TYR155、VAL129/245）与一个钠离子
  在三条独立 run 中都被遗漏——同一批真实残基，不是逐 run 随机噪声，说明它们结构上
  正好卡在 5 Å/两跳边界。结论：frame0-only 的 `derived_5a` manifest 不足以支撑
  (b) 的离线多帧 latent cache。
  第一次尝试（DEC-031）用这 1500 帧闭包的并集重建了 `derived_5a` manifest：
  `--frame-stride-all` 扩展见下方 DEC-031 记录，discovery 报告显示候选池从
  1444 膨胀到 **4874 个环境原子（1136 个残基）**——固定图节点数变成 4915，比
  `original_6a` 那个已经在 CUDA 上 OOM 的 2135 节点图还大一倍以上。DEC-032
  撤销了这个方向：**根本不该为离线 teacher 固定 environment node set**。teacher
  从不进 OpenMM/MD，所以没有理由为一张跨帧共用的固定图付出这个代价；正确做法是
  逐帧独立构造精确两跳闭包 \(S_a=S_{0,a}\cup S_{1,a}\cup S_{2,a}\)（`local_residual/teacher_graph.py`
  新增 `build_teacher_graph_for_frame`，不接受也不需要 environment manifest/atom
  mapping，直接对当前帧调用 `topology_n_hop_closure` 拿到 \(S_a\)，按 topology
  index 升序排列——ligand 41 个原子的相对顺序因此帧帧不变，缓存的 `[41,1024]`
  latent 可以直接跨帧比较/喂线性 probe）。同时不再做整残基收口：EXP-010 的整残基
  要求来自 fragment energy subtraction，当前 ligand latent 读出不算 fragment
  energy，收口是否必要是可以验证而非该假设的问题（`tests/test_exp012_teacher_graph.py`
  锁定：无收口、闭包本身即节点集、ligand 顺序在环境原子数变化时保持相对不变）。
  4874-atom union discovery 报告保留存档，标记
  `FIXED_UNION_POLICY_REJECTED_FOR_OFFLINE_TEACHER`，不晋升为任何运行时 manifest。
  新增三个工具，均未在真实数据上执行（需要 openmm_dev 环境+真实 MACE 模型/GPU）：
  - `scripts/smoke_exp012_teacher_graph_equivalence.py`：frame0 上比较
    974-node 精确闭包（graph A，新策略）与 1444-node 整残基收口图（graph B，
    复用 DEC-028/029 已冻结的 frame0 derived_5a manifest/mapping）——比较完整
    ligand latent 张量（不仅是 norm）、scalar probe 和 ligand 坐标梯度，只报告
    差异，不设通过/失败门（`status=COMPARISON_ONLY_NOT_A_GATE`）。收口是否可以
    正式删除，由这份报告的数值差异决定。
  - `scripts/audit_exp012_per_frame_teacher_graph_geometry.py`：对三条 run
    1500 帧只做几何构图（不跑 MACE），报告每帧 node/edge count、hop 0/1/2
    计数、water/ion/other 环境原子组成，以及 max/mean/P95/P99 汇总，并显式给出
    `overall_max_edge_count_frame`/`overall_max_node_count_frame`（哪个 run
    的哪一帧图最大）——用于替代"猜 frame0 是最坏情况"。
  - `scripts/smoke_exp012_teacher_graph_latent.py`：`smoke_exp012_mace_latent.py`
    的对应版本，改用 `build_teacher_graph_for_frame`（无需 environment
    manifest/atom mapping 参数，改为 `--ligand-indices` JSON），供在
    geometry 扫描选出的最大图那一帧上跑 CPU/CUDA C1。
  下一步顺序（均需在 openmm_dev 环境的真实计算节点执行）：
  ① 跑 equivalence smoke，看 974 vs 1444 的 ligand latent 数值差异，决定是否
  正式删除整残基收口；② 跑 1500 帧 geometry-only 扫描，拿到真实的 node/edge
  count 分布与最大图所在帧；③ 用 `smoke_exp012_teacher_graph_latent.py` 对
  该最大图帧跑 CPU C1，再跑 CUDA C1；④ 通过后才能开始 (b) 的逐帧 latent cache
  生成。
  (b) `derived_5a` 离线多帧 latent cache（逐帧独立构图，不使用固定 manifest）；
  (c) cached-latent 线性/ridge readout 的 held-out（leave-one-run-out）gap-variance
  验证，对比 `B=0` 基线；只有 held-out 上有增益才进入 (d)；
  (d) 蒸馏 `LocalResidualStudent`（尚不存在代码）。
- [ ] Arm A/B/C/D 表示消融、whole-run holdout 与至少 3 个训练 seed（`LocalResidualStudent`
  确定有增益后）。
- [ ] L1/L2 性能对照及 TorchScript/OpenMM Reference/CUDA、短 NVT 性能资格（L1 目前降级，
  不是当前下一步，见 DEC-030）。
- [ ] WP-5A Atenolol 三重复体系内资格。
- [ ] WP-5B 至少两个额外 ABFE 体系的同协议通用性验收。
- [ ] RBFE 接口资格通过后，才考虑 TYK2/CDK2。

## 14. 完成定义

EXP-010 GPU 标注与候选拟合已完成。六个候选均不能优于 intercept-only 基线，且
广义力方向/幅度不稳定，因此已按预注册规则记为 `FAILED`，不能直接进入 WP-5。

EXP-011 已在 AUG-001 后冻结：补采 500 帧均有限，但合并后 `127.5°` 状态只有 22 个
去相关样本（门为 25），`112.5°↔127.5°` mutual overlap 为 `0.02353`（门为 `0.03`）。
严格 v2 报告保持 `qualified_for_pmf_input=false`。不执行第二次补采，不启动
formal_run2/3，不拟合 PMF，也不进入其 NVT、WP-5 或 production。

三条 run 的逐帧五态 target ledger、backend audit 和 `local_residual` v2 语义迁移已经完成。
当前下一执行点是冻结并 seal A/B/C/D 精确表示、MACE layer/图边界、(A_k)、readout、训练
预算和数值门。随后先做 whole-run gap-variance 表示消融，再实现
`MaceLatentBasisAdapter` 的可导在线路线和轻量 student 路线。未通过信息、守恒力、稳定性、
ESS/GPU-hour 与跨 ABFE 通用性门前，不得修改 production 模块或宣称模型通用。

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
