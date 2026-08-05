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
| WP-4C / EXP-012 | `PREREG_SEALED_V2 / L2_FORMALIZED(DEC-030) / ARMS_A_B_D_NOT_PURSUED(DEC-039) / d0-5_PARTIAL(DEC-039) / derived_5a_C1_CPU+CUDA_PASSED / original_6a_C1_CPU_PASSED_CUDA_BLOCKED_ON_VRAM` | 路线正式定性为 L2（DEC-030）：`original_6a`/`derived_5a` 是离线 teacher，`LocalResidualStudent`（尚不存在代码）是唯一在线模型，L1 降级为非当前下一步。五态 ledger/backend audit 已通过；两个显式候选臂（DEC-027）：`original_6a`（6 Å，2135 节点/155624 边）CPU C1 通过，CUDA 同帧对照三次尝试均在同一 product-layer-1 张量积算子 OOM（真实峰值需求 `>=~24.07 GiB`，已排除碎片化），6 Å 分支 gradient checkpointing 未实现（与 teacher/student 分工无关，独立线程）；`derived_5a`（5 Å，1444 节点/60048 边）CPU 与 CUDA C1 均已通过（DEC-028/029，CPU↔CUDA 相对差 ~1e-7、CUDA 无 OOM），是首个完整通过 C1 的候选臂。（2026-08-04/05 更新，DEC-031→039）多帧支持域审计、逐帧 latent cache、held-out gap-variance 验证均已完成且通过（DEC-033/035/036，三 fold 均改善，均值 44.6%）；`LocalResidualStudent` 编码前设计契约（d0）第 1 项"在线动态环境表示"已由 DEC-038 real-data smoke 解决。**DEC-039（2026-08-05）**：Arm A/B/D 正式退役为 `not_pursued`（从未实现，非数值失败；预注册偏离已显式记录，`C_vs_A`/`C_vs_B` 增量比较从未执行，结论收窄为"MACE latent 信号可泛化、值得蒸馏"而非"Arm C 优于 A/B"）；`protocols/EXP-012_preregistration.json` reseal 为 `sealed`（待跑 `scripts/reseal_exp012_preregistration.py` 落实真实 payload_sha256）；(d0-5) 计算/部署预算除 ms/step 生产基线外全部冻结，训练 epoch/seed/早停改为训练 run 内时间块早停验证；ms/step 基线的哈希不匹配根因已定位（`box_vectors.npy` 陈旧）并修复诊断脚本，重新测量待执行。当前下一步是拿到哈希匹配的 ms/step 报告后完整关闭 (d0-5)，随后进入 (d1) 离线 student 拟合；仍不得训练 |
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

#### WP-5B 候选架构（2026-08-05 设计存档，阻塞于 WP-5A 通过，当前不执行）

D1 单体系结果（Atenolol `hard_window0`，DEC-039 训练报告）显示：`direct_gap` 过硬下限
（2/3 折全部 seed 改善，均值 +13.9%），但 `distilled`（对每折 teacher ridge readout 的
scalar 输出做 MSE 蒸馏）没有稳定跟上——三折的 inner-CV 选中的 ridge 系数本身就不稳定
（`0.1/0.001/0.001`），且 distilled 在 held-out run3 上明显劣于 direct_gap（+13.3% 对
+23.1%）。这提示：把 teacher 蒸馏目标定在"每折重新拟合的 scalar ridge 输出"上，这个
目标本身太下游、太体系/fold 特定，不是将来做多体系蒸馏时应该复用的形式。

若未来要做跨体系蒸馏（WP-5B 的候选实现之一，而非替代 WP-5B 本身的验收要求），推荐
形式改为 **shared trunk + 体系专属 head**，而不是让多个体系共享同一个最终标量 \(B\)：

\[
\mathbf h_s(\mathbf R)=f_\theta^{\rm shared}(\text{typed local geometry}),
\qquad
B_s(\mathbf R)=g_{\phi_s}\bigl(\mathbf h_s(\mathbf R)\bigr)
\]

理由：\(B_s\) 依赖该体系的 MM 力场、ligand/pocket、所选 alchemical states、\(A_k\)/
\(\Delta A_k\)、原始 gap 的尺度和方向、pilot 轨迹覆盖的构象分布——两个体系里几何上
相似的局部结构，最优 path correction 未必数值相同。真正可能跨体系共享的是"哪些局部
接触/溶剂重排/化学环境重要、如何把它们编码成一个便宜的表示"，不是"这段局部结构必须
输出固定的 +x kJ/mol"。

蒸馏目标同理应下沉到表示层，而不是每折重拟合的 scalar teacher 输出，候选包括：

- 把 1024 维 MACE latent 投影到低维（如 16–64 维）后让 student 逐维/relational 匹配
  （Gram matrix、frame–frame similarity、子空间），而非只匹配一个标量；
- 训练目标写成 \(\mathcal L_s=\mathcal L_{{\rm gap},s}(g_{\phi_s}(h_\theta))+
  \lambda_T\mathcal L_{\rm latent}(h_\theta,z_s^{\rm MACE})\)——teacher 只监督共享
  trunk 的表示，最终 scalar 完全由该体系自己的 gap loss 决定，不再由 teacher scalar
  直接规定。

三层验证（体系内 held-out run → leave-one-system-out trunk 预训练+体系专属 head 微调
→ zero-shot）与四组必需对照（单体系 direct-gap / 单体系 distilled / 多体系
direct-gap 预训练 / 多体系 MACE-distilled 预训练）与 WP-5B 正文的通用训练管线验收
要求兼容，可以直接复用同一批已冻结的数值门，不需要另立一套判据。

当前架构（typed atom embedding + radial/contact + ligand-only pooling）概念上可以
迁移，前提是 embedding 严格只用可跨体系定义的 atom type/element/role，topology index
只能用于当前帧的身份识别和构边，不得混入 embedding 特征——这一条现有代码已经满足。
但 Atenolol 审计出的图规模上限（S1≤320、边≤2048、单原子 neighbor≤80）是本体系的
部署预算，不是跨体系上限；新体系必须先各自跑一遍相同的 geometry audit，不能直接
沿用这几个数字。

**执行前置条件（明确写清楚，不得绕过）**：本节只是设计存档，不是新的下一步。启动
前必须先满足 WP-5A——即证明当前单体系 direct-gap 原型在**真实生产采样**里
（不只是离线 gap-variance-loss）对 mutual overlap / ESS / ESS/GPU-hour 有可测改善；
在此之前，本节所有内容都不执行、不分配算力。若未来真的启动，必须作为独立预注册的
新实验登记（新 DEC + 新 preregistration 或其修订），不得把本次已完成的 D1 单体系
结果事后包装成"多体系工作一直都是这么规划的"。

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
  正式删除整残基收口——已执行，frame0 上 ligand latent 最大差 `3.33e-16`、
  ligand 梯度最大差 `1.07e-14`、scalar probe 差 `2.84e-14`，均为 float64 舍入
  量级，证实整残基收口对两层 ligand latent 无实际贡献，正式确认删除；
  ② 跑 1500 帧 geometry-only 扫描，拿到真实的 node/edge count 分布与最大图
  所在帧——已执行，最大图为 `hard_window0_run3`/frame343（1066 节点/45656 边）；
  ③ 用 `smoke_exp012_teacher_graph_latent.py` 对该最大图帧跑 CPU C1，再跑
  CUDA C1——均已通过（CPU 72.3s、CUDA 10.4s，CPU↔CUDA 相对差 `2.2e-7`，
  float32 舍入量级，无 OOM）；④ 通过后才能开始 (b) 的逐帧 latent cache 生成——
  已完成，见下。
  中途发现并修复了一个真实的 CPU/CUDA 不一致（非 dtype 问题）：几何审计脚本在
  CPU 上、bulk 脚本在 CUDA 上各自独立判定 graph membership，导致某一帧两侧
  edge count 相差 2（一对原子对在 cutoff 边界因浮点实现差异翻转）。修复为
  DEC-032 Option C：membership 永远在 CPU float64 决定一次
  （`local_residual/teacher_graph.py::compute_canonical_graph_membership`），
  只把离散结果（topology indices/edge_index/unit_shifts）搬到目标 device 执行
  MACE forward（`build_teacher_graph_from_membership`），并新增
  `graph_membership_sha256` 做逐帧硬门核对，而非容忍 count 级别的 ±几条边
  误差。`local_residual/mace_latent.py` 相应放宽了
  `positions.requires_grad` 检查（仅在 `require_coordinate_grad=True` 时强制），
  使 bulk 生成可以老老实实用 `requires_grad=False`+`torch.no_grad()`，不用为
  满足契约假造一个不需要的 autograd leaf。
- [x] (b) `derived_5a` 离线多帧 latent cache（逐帧独立构图，CPU float64 决定
  membership、CUDA float32 执行 MACE forward，不使用固定 manifest）：三条 run
  各 500 帧全部生成完成且通过硬门（node/edge count 与 graph_membership_sha256
  逐帧精确核对，无一帧违规）。`output/outer_lambda_exp012/teacher_latent_cache/`
  下 `latent_cache_hard_window0_run{1,2,3}.npz`（各 `ligand_latent [500,41,1024]`
  float32，~86.16 MB）+ 对应 `_report.json`，report_sha256 分别为
  `bfd1a8ef9df26b111b593ea734f1a8cf76c6658a087cd653d2bcdb3ab3bbb639`（run1）、
  `50138e76b239e39944f2c9e55305b77f66f6c447cb44b20921ae02d15f0b72a8`（run2）、
  `077c8737ef7d9674a9cf7a818ce5b9734ade0b407b3c943b4cf6b2e233476fcc`（run3）；
  npz_sha256 均与文件实际哈希核对一致。三次运行实际耗时共 ~7.7 分钟（单次 no-grad
  前向，远低于早先双前向 smoke 外推的 ~4.3 小时估计）。`ledger_joined=False`，
  纯 representation，未拼 target/gap-variance 数据。
- [x] (c) cached-latent 线性/ridge readout 的 held-out（leave-one-run-out）gap-variance
  验证，对比 `B=0` 基线；只有 held-out 上有增益才进入 (d)。
  **已在真实数据上执行，全部通过**：三个 leave-one-run-out fold 全部相对 `B=0`
  基线改善（held-out run1 39.9%、run2 29.1%、run3 64.8%，均值 44.6%），比
  "至少一个 fold 有增益"的最低门槛更强。join 报告 report_sha256
  `8dfc47e3352534f8b67826ee570f6830de2b618cf935ff9354e74f0082c016ce`，readout
  报告 report_sha256 `d77a8e132780270363abb4a33572912e518c102ef8c1f4ed38d36df92c7b05c3`。
  待办：2/3 fold 选中了 ridge 网格里最小的候选值 `1e-3`，说明内层 CV 可能想要更小的
  正则化，建议后续把网格向下扩展（如 `1e-6/1e-5/1e-4`）复核，但不影响"held-out 确有
  改善"这一定性结论。下一步是 (d)，需要先与用户对齐范围（工具本身如下，只需
  numpy/torch，不需要 MACE/GPU，纯 CPU 线性回归）：
  - `scripts/join_exp012_teacher_latent_cache_with_ledger.py`：拼接 latent cache
    （`pooled_latent`）与 `output/outer_lambda_exp012/mm_ledger_cuda/<run_id>/`
    的 `adjacent_gap_reduced`/`log_importance_unnormalized`。`delta_A`（每条相邻边
    的包络增量）直接从 `protocols/EXP-012_preregistration.json` 的
    `target.global_schedule.A_k`（`sin^2(pi*lambda_vdw)`，`global_state_ids=[0,1,2,3,4]`
    切片）读取并独立用 `sin^2(pi*lambda_vdw)` 公式重新核对，不重新拟合、不猜测——
    对齐 PLAN 文档"第一轮冻结全局 A_k"的明确要求。fail-closed 校验：cache report 的
    `preregistration_sha256` 必须等于当前 preregistration；ledger report 自己的
    `preregistration_payload_sha256` **不**要求相等（ledger 生成时间早于文档后续
    的无关编辑，已核实 `f_k_kj_mol`/`lambdas_vdw` 本身与当前 preregistration 的
    target 部分精确一致），改为直接比对这两个物理相关字段；cache 与 ledger 的
    `frame_index` 数组必须逐位等于 `0..499`，否则拒绝按位置拼接。
  - `scripts/fit_exp012_local_residual_linear_readout.py`：线性 readout
    `basis_reduced = w^T·standardize(pooled_latent)`（无 intercept——`Var(X+c)=Var(X)`，
    intercept 对 `bidirectional_gap_variance_loss` 的贡献恒为零，不是遗漏）；
    ridge 系数用只用两条训练 run 的内层 2-way CV 选择（不碰 held-out run，避免泄漏），
    再在两条训练 run 合并数据上用选中的 ridge 系数重新拟合，最后在真正 held-out
    的第三条 run 上比较 `B=0` 基线与拟合值的 `gap_variance_loss`。`A_k`/`delta_A`
    全程冻结（不联合拟合），复用已测试的
    `local_residual/loss.py::bidirectional_gap_variance_loss`（不重新推导闭式解）；
    由于 `delta_A` 固定后目标函数是 `w` 的凸二次型，用 `torch.optim.LBFGS`
    （strong Wolfe）收敛到唯一全局最优，不需要调学习率。
  - 新增测试 `tests/test_exp012_local_residual_linear_readout.py`：构造一个
    "gap 恰好是 pooled_latent 的线性函数、三条 run 共享同一真实关系"的合成数据集，
    验证三个 leave-one-run-out fold 全部相对 `B=0` 基线改善（`>90%`），以及输出
    policy 明确标注 `a_k_learned=false`/`mace_encoder_trained=false`/
    `local_residual_student_trained=false`（这一步训练的只是线性 readout，
    不是训练 MACE、不是学习 A_k、也不是训练 (d) 的 `LocalResidualStudent`）。
  示例命令（openmm_dev 环境）：
  ```
  python scripts/join_exp012_teacher_latent_cache_with_ledger.py \
    --latent-cache-dir output/outer_lambda_exp012/teacher_latent_cache \
    --output output/outer_lambda_exp012/teacher_latent_ledger_join.npz

  python scripts/fit_exp012_local_residual_linear_readout.py \
    --joined output/outer_lambda_exp012/teacher_latent_ledger_join.npz \
    --output output/outer_lambda_exp012/local_residual_linear_readout_report.json
  ```
- [x] DEC-030(c) 已冻结登记结果（39.9%/29.1%/64.8%，均值 44.6%，全部 fold 改善）；
  不因 2/3 fold 选中 ridge 网格最小值 `1e-3` 而重跑更宽网格替换它——那是看到边界结果后
  才决定加宽网格，事后调整（post-hoc）不能改变已经很强的 go/no-go 结论。后续若要跑更宽
  网格（如 `1e-6/1e-5/1e-4`），必须显式标注 `sensitivity-only`，不得替换 DEC-034/035 的
  登记结果。
- [ ] (d) 蒸馏 `LocalResidualStudent`（尚不存在代码）。编码前必须先完成 DEC-030(d0) 设计
  契约（PLAN 文档同名章节），本轮已冻结契约框架；契约第 1 项（在线动态环境表示）已由
  DEC-038 real-data smoke 解决，其余各项仍待 (d0-5) 完成后才能进入 (d1)：
  - [x] **(d0-1) 在线动态环境表示**（DEC-038，2026-08-05）：不追踪瞬态水分子持久身份，
    用 `local_residual.geometry.ligand_environment_cross_edges` 每步动态重算
    ligand–environment cutoff funnel（无固定 manifest），能量权重用
    `local_residual.geometry.quintic_c2_cutoff` 平滑包络而非硬 0/1 门控。
    `scripts/smoke_exp012_student_environment_funnel.py` 在 `openmm_dev` 环境对
    run1/frame0 与 run3/frame343（teacher 已知最坏图帧）两条真实帧实测，与
    `local_residual/teacher_graph.py` 的已审计 canonical membership 逐对（含周期
    unit shift）完全一致，边界平滑性扫描确认离散候选翻转不产生能量跳变；
    report_sha256 `101e3364f0cebd91694b43bc3b93e239ace49f54a8b5cefbaa960cade411bc7a`
    / `107416dad1bf09c6c371adece36a969143be21610af0ac8254b6f22ea05a7ae4`。只证明设计
    可实现，不代表 student 有统计增益或生产资格，不改变网络结构（d0-2）本身的选择。
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
    neighbor≤64/80，1500 帧真实审计，`report_sha256 782e58242233d3b2153e719dda7685d08f0e65e61f1dc35f4d6d33c114cf416f`）、
    CPU float64/CUDA float32 funnel 一致性（`report_sha256
    9671470e03e12029ae503f1a6f7b5fd31e193d51fc4fd5ede385fa93cfcf934b`）、训练 seed
    （≥3/变体/折，硬下限）、早停规则（**改用训练 run 内部末尾 20% 连续时间块**做早停
    验证集，不复用被隔离的第三条 run 兼职早停+最终评估——原草案"早停判据必须用该折
    被完全隔离的第三条 run"已被用户否决，因为那样第三条 run 同时当验证集和测试集，
    存在乐观偏差）、`max_epoch=500`、`early_stop_patience=30`、held-out 改善判据（3 折
    目标全部改善、硬下限 2/3、均值下降 >0%、最差单折不劣化超过 10%、蒸馏相对
    direct-gap 不得更差）。**尚未冻结**：目标每 MD 步毫秒数（生产基线 ms/step 待用户
    用修复后的 `scripts/benchmark_exp012_no_student_window0_baseline.py` 重新测量，
    `win_sys_xml_sha256_matches_manifest` 根因已定位为 `box_vectors.npy` 陈旧并已修复
    诊断脚本，见 DEC-039，但尚未拿到哈希匹配的新报告）、GPU 显存上限（依赖 D3 实测）、
    允许的 cutoff（沿用 teacher 侧 5.0 Å support cutoff，student 侧待 D1/D2 验证）。
  - [x] (d1) 离线 student 拟合，held-out gap variance 与 teacher fidelity；用真实逐帧
    坐标计算 student 自己的特征，不是只读 teacher 的 cached `pooled_latent`。最终保留
    direct-gap 路线进入 D2；distilled 路线未通过相对 direct-gap 的增量门。
  - [x] (d2) 坐标/autograd 资格：有限差分、cutoff 平滑性、力尾部行为。最终证据为
    `output/outer_lambda_exp012/student_d2_report_v4.json`：3 folds × 3 seeds 的 9 个
    direct-gap checkpoint 在三条 run 上共 27 组检查，`all_checkpoints_passed=true`；
    report SHA-256 `329a98331400f22fe13b76e00f435f4c3a83431441f33bc35af502540d56f08b`。
  - [ ] (d3) 部署资格：TorchScript、OpenMM Reference、CUDA 一致性、耗时（仅在 d1 仍保留
    有意义的 held-out 改善后才启动）。
  - [ ] (d4) 动力学资格：短 NVT、稳定性，再做独立重复（仅在 d3 通过后启动）。
  每个子阶段失败即停，不得跳阶段推进到下一个。

#### (d2) 最终资格结果（2026-08-05，DEC-040）

| 检查 | v4 结果 | 结论 |
|---|---:|---|
| 覆盖 | 9 checkpoints；每个检查 3 runs，共 27 组 | 3 folds × seeds 0/1/2 全覆盖 |
| 有限差分 | 最大绝对误差 `2.4711e-7`；最大相对误差 `1.8242e-5` | 均低于 `1e-4` / `1e-2` 硬门；27/27 通过 |
| 非参与原子 | 27/27 组均为零力 | 通过 |
| Cutoff 平滑性 | 粗/细步长能量跳变缩放比 `22.6405–24.9856`，连续期望值 25；被探测 pair 每组恰好一次 membership flip | 27/27 通过 |
| Force tail / 近接触 | 0.3 Å 合成近接触能量和力全部有限；最大近接触力范数 `5.956` reduced/Å | 27/27 通过 |

报告状态为 `COMPLETED_D2_CHECKS`，dataset report SHA-256 为
`75f0e2ca5a9613ee7cc77964ade9112814ccb136c94a74ec3244d0aaebd8c97c`。报告 policy 明确：
只检查 direct-gap checkpoints，排除 distilled checkpoints；不按 held-out run 选择
checkpoint；未使用 TorchForce、未执行 NVT。故本结果只关闭 D2，下一未完成门为 D3，
不得将其写成 OpenMM/CUDA、短 NVT 或 production 已通过。

#### (d0-5) 预注册提案 → DEC-039（2026-08-05，部分冻结）

> 状态：**除 ms/step 生产基线外已以 DEC-039 正式冻结**（EXPERIMENT_LOG 决策日志表，
> DEC-038 行之后）。ms/step 基线仍待用户用修复后的
> `scripts/benchmark_exp012_no_student_window0_baseline.py` 重新测量并确认
> `win_sys_xml_sha256_matches_manifest=true`，因此上面 `(d0-5)` 复选框标记为
> `[~]`（部分完成）而非 `[x]`。**本提案本身只授权 (d1) 离线 student 拟合；不授权
> TorchScript、OpenMM、CUDA 部署或 NVT——那些仍分别要求 d2/d3/d4 各自通过。**

三类数字：**硬上限**（超过即拒绝训练/部署）、**目标值**（首选设计应达到）、
**淘汰门**（D1/D3 测得不达标时停止，不得静默放宽）。凡标注"PROPOSED"的数字尚无
既有实测支撑，需要用户确认或调整；凡引用真实数字的行都标出了来源。

**1. 模型规模**（2026-08-05 用户确认，替换此前 PROPOSED 数字）

| Item | Target | Hard ceiling | Measurement definition | Evidence/rationale |
|---|---|---|---|---|
| Trainable parameter count | ≤50,000 | ≤100,000 | 对 `requires_grad=True` 的张量求 `numel()` 之和，覆盖 embedding + interaction block(s) + readout + 有界标量头；显式排除冻结 buffer（如固定 cutoff 常数）；**direct-gap 与 distilled 两个 student 各自独立满足此上限，不得合并计算** | 50k 足以覆盖 typed embedding + RBF/contact + 1–2 个小 block + scalar readout；100k（而非此前提议的 500k）是因为更松的上限可能在 edge-wise MLP 上产生明显计算成本；100k float32 本体约 0.4 MB。超过 100k 必须另立设计决策，不得训练时静默放大 |
| Serialized model size on disk | ≤1 MB | ≤2 MB | float32 `torch.save(state_dict())` 文件字节数；与参数量分开报告，因为 dtype/buffer/序列化元数据可能使两者不成比例 | 参数量是主硬门，文件大小是辅助完整性门；100k 参数 + buffers + 元数据，2 MB 留有余量 |

**2. 图规模**（student funnel graph ≠ teacher 两跳闭包图，不能共用同一个数字；2026-08-05 用户确认/修订）

方向语义写死为 **environment sender → ligand receiver**（不是含糊的"ligand→environment"）：
唯一允许的边类型是这一种 bipartite cross edge，不含 reverse edge、不含 ligand–ligand edge、不含
environment–environment edge、不含 self edge。

| Item | Target | Hard ceiling/gate | Measurement definition | Evidence/rationale |
|---|---|---|---|---|
| Unique S1 environment atoms | ≤256 | ≤320；超过即 fail closed，不得截断多余原子 | 5.0 Å cutoff 下 `ligand_environment_cross_edges` 输出中出现的唯一环境侧（receiver）原子数；DEC-038 已证明与 teacher 的 `hop_counts_by_layer[1]` 完全一致 | **真实测量**（1500 帧全集，`output/outer_lambda_exp012/per_frame_teacher_graph_geometry.json`）：min 209 / mean 233.8 / P95 246 / P99 250 / **max 255**。250 已低于实测最大值，不适合作目标；256 覆盖已测最大值并留清晰边界 |
| Directed environment→ligand cross edges | **1536**（已确认） | **2048**（已确认） | 5.0 Å 下 `ligand_environment_cross_edges` 输出 `edge_index` 的列数 | **真实测量，1500 帧全集**：`scripts/audit_exp012_student_environment_funnel_geometry.py`，report_sha256 `782e58242233d3b2153e719dda7685d08f0e65e61f1dc35f4d6d33c114cf416f`。实测最大值 **1464**（`hard_window0_run2` frame202）≤1536，按规则直接冻结 target=1536/ceiling=2048，不需要上调、不触发 redesign。**明确不是**teacher 的 45,656 条两跳闭包边（不同的图） |
| 单个 ligand atom 的最大 neighbor 数 | **64**（已确认） | **80**（已确认） | 每个 ligand atom 作为 receiver 端点被计数的 environment 邻居数（bipartite 邻接矩阵的逐 ligand-atom 度数） | **真实测量，1500 帧全集**：同一份审计报告。实测最大值 **55**（`hard_window0_run1` frame85，ligand topology index 4607，分布 mean 31.23 / P95 42 / P99 46）。按规则 target=向上取整到 16 的倍数=64，hard ceiling=64×1.25 向上取整到 16 的倍数=80 |
| 边的组成 | 硬性要求（非数字）：仅 environment→ligand bipartite | 不含 ligand–ligand、environment–environment、reverse、self edge | — | 对齐 DEC-037 (d0-2) 已冻结的最小架构；审计脚本对全部 1500 帧逐帧结构性断言此不变量，零违反。若未来升级到需要 environment–environment 消息传递的架构，这一行必须重新冻结，不得默默加入 |

**1500 帧几何审计（2026-08-05 已完成，真实执行，非抽样外推）**：
`scripts/audit_exp012_student_environment_funnel_geometry.py`——纯几何、无 MACE、无 GPU、无
student，复用 `local_residual.geometry.ligand_environment_cross_edges`（CPU float64，5.0 Å
cutoff，32 workers，耗时 107.5s）对 `hard_window0_run1/2/3` 全部 1500 帧扫描。三条 run 各自的
edge_count/s1_atom_count 分布（max/mean/P95/P99）：run1 `1445/1281.91/1371.10/1410.07` 与
`249/230.34/242.05/245.00`；run2 `1464/1294.11/1379.05/1412.05` 与
`255/235.76/248.00/251.02`；run3 `1443/1265.59/1344.05/1384.01` 与
`253/235.25/246.00/250.00`。全局 S1 最大值 255 与此前 teacher 两跳闭包审计
（`per_frame_teacher_graph_geometry.json`）独立测得的 hop1 最大值**完全一致**，交叉验证通过。
顺带确认 `system_atom_count=73536`，与此前 grep 估计值精确吻合。
`preregistration_sha256` 一致性核对通过（`aba95cd2a8f42d58172aad71117a646a13557496fec21b7cb59f76bb39ebbd1a`）。

**3. 性能计量口径**（禁止只报"裸网络 forward"；必须逐项 + 求和）

| Item | Proposed target | Hard ceiling/gate | Measurement definition | Evidence/rationale |
|---|---|---|---|---|
| Dynamic neighbor discovery | 待预算内分摊 | — | `ligand_environment_cross_edges` 调用本身的 GPU 同步耗时 | 未在 GPU 上测过——DEC-038 为了与 teacher 精确比对，全程跑在 CPU float64 |
| Triclinic PBC / shifts | 待预算内分摊 | — | 当前实现中此步嵌在 `minimum_image_displacement` 内部，不可分离；d3 前必须决定是并入上一行合并计时，还是加独立子计时器 | 尚未决定 |
| Feature construction | 待预算内分摊 | — | 从 wrapped displacement 到网络输入张量之间的耗时 | 依赖 (d0-2) 最终特征选择，尚未定案 |
| Student forward | 待预算内分摊 | — | 可训练网络本身前向传播耗时 | 无 student 代码 |
| Autograd force | 待预算内分摊 | — | `torch.autograd.grad` 产生 `-∇R B_student` 的耗时 | 无 student 代码 |
| TorchForce/OpenMM 调用开销 | 待预算内分摊 | — | 复用 `outer_lambda_neural_basis.py::benchmark_torchforce_outer_lambda`（WP-3 已有、已验证过 mock/analytic basis 的计时框架），不新造一套 | 已有可复用工具，不必新写 |
| **合计每 MD 步** | PROPOSED：相对当前无 student 的生产步耗时增加 ≤15%；次要绝对参考 ≤5 ms/step（**PROVISIONAL**，未锚定真实基线） | PROPOSED：相对增加 ≤50%（淘汰门：D3 测得超过即停，不得默许进生产） | 端到端测量，嵌在真实 OpenMM 步循环里（对齐生产 `LangevinMiddleIntegrator` 2.0 fs，`ibs_engine.py:1622`），不是网络的独立 microbenchmark | **缺口**：尚未在这块 RTX 2080 Ti 上实测"当前无 student 的生产步耗时/ns-day"基线，相对目标换算成绝对 ms 数之前需要先补这一测——不是靠猜 |

**4. 硬件基线**

| Item | Value | Note |
|---|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti, 11264 MiB, driver 580.173.02 | 本 session `nvidia-smi` 确认 |
| 软件栈 | torch 2.10.0 / OpenMM 8.5.1 / openmm-ml 1.6 / openmmtorch 1.5 / mace_torch 0.3.16 / mdtraj 1.11.1，均在 `openmm_dev`（`/home/ruigengji/mambaforge/envs/openmm_dev`） | 本 session `site-packages` 确认；`omm_torch_126`（torch 2.6.0 / OpenMM 8.2.0 / 有 NNPOps 但无 mdtraj）不是生产执行环境（`docs/handoffs/RESUME_DEXP_SESSION.md:93`，DEC-021/022/032/033 均在 `openmm_dev` 实测），若未来真用 NNPOps 后端必须单独重测，不得挪用这里的数字 |
| Precision | 在线 student forward/backward 用 float32（对齐 DEC-029 teacher CUDA float32 C1）；funnel 直接在 CUDA float32 上执行，**不需要**教师那样的 CPU-float64-决定/GPU-执行拆分 | **已闭合（2026-08-05）**：`scripts/smoke_exp012_student_funnel_cuda_consistency.py`，report_sha256 `9671470e03e12029ae503f1a6f7b5fd31e193d51fc4fd5ede385fa93cfcf934b`。在 run1/frame0（1206 边）与本轮审计确认的真实最坏边数帧 run2/frame202（1464 边）上，CPU float64 与直接 CUDA float32 执行的 `ligand_environment_cross_edges` **edge set 完全一致**（`all_edge_sets_identical=true`），无一处 cutoff 边界分歧；共有 pair 的距离最大绝对差 `3.38e-6` Å、`quintic_c2_cutoff` 权重最大绝对差 `1.55e-6`，均为 float32 舍入量级，不是真实分歧。结论：funnel 不需要 DEC-032/033 Option C 那样的 CPU-float64-决定/GPU-执行拆分，可以直接在 CUDA float32 上跑 |
| 积分步长 | 2.0 fs（`LangevinMiddleIntegrator`） | grep 确认，`ibs_engine.py:1622`，生产真实值，非提议 |
| 体系原子数 | ≈73,536（对 `output_lrc_fix/topology.cif` 的 `ATOM`/`HETATM` 记录行数计数得到） | 近似值，未跑 `mdtraj.load(topology).n_atoms` 精确核实（遵循"不亲自跑"的既有约定）；DEC-039 冻结前应补一个精确值 |

**5. Cutoff / 平滑权重语义**

| Item | Value | Note |
|---|---|---|
| Support cutoff | 5.0 Å（DEC-038，`derived_5a`） | 已用真实数据验证，非本提案新增 |
| 平滑权重族 | `quintic_c2_cutoff`，outer=5.0 Å；inner=4.0 Å | DEC-038 明确记录 `inner_cutoff=4.0Å` 只是 smoke 测试选择，不是已冻结的生产参数——本提案要么显式采纳这个值，要么现在提出别的值，不能继续隐式沿用 smoke 的选择 |
| 成员资格与权重分层 | 硬性要求：离散 neighbor membership（图里有没有这条边）可以逐步变化；能量权重必须在原子离开图之前就已连续降到精确 `0.0`（DEC-038 已用真实边界扫描证实） | 违反此项（把离散 membership 直接当 0/1 能量门）在 d2 视为硬性失败；检查方法与 DEC-038 边界扫描同构，但作用对象是 student 实际能量输出，不只是权重函数本身 |

**6. 训练预算与判决口径**

| Item | Proposed target | Hard ceiling/gate | Measurement definition | Evidence/rationale |
|---|---|---|---|---|
| Outer 验证划分 | 沿用 DEC-034/035/037(第③项) 已冻结的 `hard_window0_run1/2/3` 三折 leave-one-run-out | 不得用单条轨迹内的随机分帧代替 | — | 帧内高度自相关，DEC-011 已因同一理由拒绝过单轨迹结论 |
| 训练 seed 数 | PROPOSED：direct-gap 与 distilled 每种、每折 ≥3 个独立 seed | 硬下限 3 | 模型初始化 + minibatch 顺序均需不同 seed，不是重跑同一 seed | 对齐本项目已有的"至少 3 个独立重复"惯例（DEC-011、§14 完成定义第 8 条） |
| 最大 epoch / 早停 | **PROPOSED — 需要用户给出具体数字，本仓库无先例可套** | 早停判据必须用该折被完全隔离的第三条 run，不得用从两条训练 run 里切出的随机验证集 | — | DEC-034 的 ridge readout 是闭式解（LBFGS 直接收敛到唯一全局最优），没有 epoch/早停概念可以直接借用；这是真正的新数字 |

**7. D1 go/no-go**

| Item | Proposed target | Hard ceiling/gate | Measurement definition | Evidence/rationale |
|---|---|---|---|---|
| 至少几折改善 | 目标：3/3（对齐 DEC-034/035 已登记的线性 readout 结果） | 硬下限：3 折中至少 2 折相对 `B=0` 有 gap-variance 下降 | 与 DEC-034/035 相同的 `bidirectional_gap_variance_loss`，只在该折被隔离的 run 上评估 | 2/3 提议为硬底线（对应 DEC-030(c) 最初的最低门槛，DEC-034/035 后来以 3/3 超额通过）；已登记的线性 readout 结果（3/3，均值 44.6%）是参考基准，不是对不同参数化模型的保证 |
| 平均 gap-variance 下降幅度 | 目标：与已登记线性 readout 同量级（~44.6%） | 硬下限：均值 >0%（严格优于 `B=0`） | 3 折相对下降幅度的均值 | 线性 readout 直接读 teacher 缓存的 `pooled_latent`，是不同的函数族；student 用真实坐标现算特征——达到同量级是目标，不是保证 |
| 最差单折容忍度 | 目标：无单折退化 | **PROPOSED** 硬门：最差单折相对 `B=0` 恶化不超过 10% | — | 防止"2 折很好、1 折严重崩坏"仅凭均值蒙混过关；具体阈值需用户确认 |
| Distillation 相对 direct-gap 对照的增量 | **PROPOSED** 目标：held-out gap variance 相对下降再多 ≥10%（相对值） | 硬门：distilled 不得劣于 direct-gap 对照（劣于即视为 teacher target 起反作用，硬性失败） | 两个变体必须用同架构/同 seed/同数据划分训练（对齐 DEC-037 第③项必需对照），在同一被隔离的 run 上比较 | 落实 DEC-037 第③项"否则任何增益无法归因于 MACE teacher"；没有这个margin，即使数字表面更好也不能宣称"蒸馏有效" |

**冻结 DEC-039 前必须先补齐的缺口（不是数值分歧，是缺失的测量）**：
1. ~~单个 ligand atom 的最大 neighbor 数与 edges 上限~~ **已解决**（1500 帧真实审计，见上，
   report_sha256 `782e58242233d3b2153e719dda7685d08f0e65e61f1dc35f4d6d33c114cf416f`）；
2. ~~funnel 的 CPU float64 vs CUDA float32 一致性 smoke~~ **已解决**（见上，`all_edge_sets_identical=true`，report_sha256 `9671470e03e12029ae503f1a6f7b5fd31e193d51fc4fd5ede385fa93cfcf934b`）；
3. 当前无 student 的生产每步耗时 / ns-day 基线——**根因已定位，等待重新测量**（DEC-039，
   2026-08-05）：v1 report_sha256 `062c63b2b4dfa4e2939504766b5f7692164dbc05527335a5d02b44209e620a57`
   （`1.3902/1.3961/1.3991 ms/step`，median `1.3961`、P95 `1.3988`）里
   `win_sys_xml_sha256_matches_manifest=false` 的根因是 `output_lrc_fix/box_vectors.npy`
   陈旧——它只在初次建系统缓存时写一次（`runabfe.py:880`/`1117`），真正建窗口 0 用的盒子
   取自 `pipeline.box_vectors`，会在 `pre_equilibrate()` NPT 弛豫后（`abfe_pipeline.py:1915`/
   `6294`/`6308`）与 Boresch rebalance 后（`abfe_pipeline.py:2195` → `runabfe.py:4420`）被
   内存内重新赋值但从不写回磁盘；`output_lrc_fix/` 真实文件 mtime（`box_vectors.npy` 01:18 →
   `rebalance.chk` 08:27 → `manifest.json` 09:08）佐证两者相差近 8 小时。窗口 0 生产
   System 无 `MonteCarloBarostat`（v1 报告自带的 `force_groups` 已确认），故盒子一旦建窗口
   即冻结，可以直接从已加载的 `openmm.chk` 读回真实盒子，不需要重放具体走了哪条 pipeline
   分支。`reference_positions=None` 仍确认与此无关（只在
   `not _has_valid_boresch_restraint(...)` 分支使用，`hard_window0` 的 Boresch 约束有效，
   此分支本就不会执行）。`scripts/benchmark_exp012_no_student_window0_baseline.py` 已改为
   两阶段构造（先用陈旧盒子建一次性 probe System/Context 去 `loadCheckpoint`，读回真实盒子，
   再用它重建真正用于哈希校验和计时的 System），schema 升到 v2；**修复后的重新测量尚未
   执行，在拿到 `win_sys_xml_sha256_matches_manifest=true` 的新报告前，这份 ms/step 数字
   仍只作为参考量级，不计入本条已冻结的 DEC-039 生产基线**；
4. 训练 epoch 数与早停判据的具体数字——**已解决**（DEC-039，2026-08-05）：改用训练 run
   内部末尾 20% 连续时间块做早停验证集（不复用被完全隔离的第三条 run 兼职早停+最终评估，
   避免乐观偏差），`max_epoch=500`、`early_stop_patience=30`、`seeds_per_variant_per_fold=3`；
5. ~~体系精确原子数~~ **已解决**：同一份审计报告 `system_atom_count=73536`，与 grep 估计值精确吻合。

- [x] Arm A/B/D 表示消融：**正式退役为 `not_pursued`**（DEC-039，§11A.12，
  `EXPERIMENT_LOG_outer_lambda_neural_basis.md`）——三者从未实现任何代码，不是数值
  跑输。预注册偏离已显式记录：`decision.arm_C_increment_comparisons=["C_vs_A","C_vs_B"]`
  从未执行，实际只做了 `C vs B=0`（无残差项基线）对照，`B=0` 不是 Arm B。结论收窄为
  "MACE latent 存在可泛化 gap-variance 信号、值得蒸馏"，不得声称 Arm C 优于 A/B；
  该论文级结论需另立独立预注册对照实验，不阻塞 D1。Arm C 的 whole-run holdout 与
  3 个 leave-one-run-out fold（教师侧线性 readout）已通过，`LocalResidualStudent`
  自己的 ≥3 seed/变体训练在 D1 中执行。
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
