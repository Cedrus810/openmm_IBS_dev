# 外层 λ 神经基势详细实施计划

> **文档角色：工程执行计划。** 科学原则见
> [`PLAN_outer_lambda_neural_basis.md`](PLAN_outer_lambda_neural_basis.md)，真实运行结果记录在
> [`EXPERIMENT_LOG_outer_lambda_neural_basis.md`](EXPERIMENT_LOG_outer_lambda_neural_basis.md)。
> 本文不表示相关代码已经实现。

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

## 2. 第一轮明确不做

- 连续 λ-conditioned MACE。
- 未处理的全体系 pretrained MACE 总能量直接叠加。
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
| `runabfe.py` | CLI、配置合并、模式路由 | 接收一个显式神经路径配置文件 |
| `abfe_core.py` | 基础势、DEXP、MACE 辅助能力 | 模型规格、模型哈希、外层 λ 控制器 |
| `abfe_pipeline.py` | Stage 调度、provenance、缓存 | 解析配置、生成系数矩阵、写协议指纹 |
| `ibs_engine.py` | IBS 系统、CV、能量探针、TMBAR | 共享基势、目标能量组合和账本 |
| `tests/` | 物理和协议回归 | 端点、账本、缓存、TorchForce/GPU 测试 |

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

## WP-4：单个真实任务化基势

第一候选范围：

- ligand；
- 固定关键口袋残基；
- 暂不加入交换水和离子；
- 只针对一个 torsion、rotamer 或接触重组。

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

## WP-5：单基势小窗口科学试验

固定三组对照：

1. 基础路径；
2. 仅 λ 重排；
3. 基础路径 + 单基势。

主指标：

- importance/absolute ESS；
- ESS/GPU-hour；
- 慢自由度独立转换；
- 自相关时间；
- 神经能量和力分位数；
- 异常结构率；
- 端点 ΔG 一致性。

晋级必须同时满足：

- 端点正确；
- 账本闭合；
- 稳定性不劣化；
- 独立采样指标改善；
- ESS/GPU-hour 不劣于基线；
- 收益不能被简单 λ 重排完全替代。

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

建议新增：

```text
tests/test_outer_lambda_controller.py
tests/test_neural_basis_endpoint_contract.py
tests/test_neural_basis_ibs_accounting.py
tests/test_neural_basis_cache_contract.py
tests/test_neural_basis_shared_evaluation.py
tests/test_neural_basis_torchforce_integration.py
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

- [ ] 已选择 softcore 原型或 DEXP 起步。
- [ ] 已指定一个困难窗口。
- [ ] 已指定一个目标慢自由度。
- [ ] 已保存基础 benchmark。
- [ ] 已定义神经能量的物理含义。
- [ ] 已确认不是未经处理的全体系 MACE 总能量。
- [ ] 已确定固定原子集合。
- [ ] 已定义端点容差。
- [ ] 已定义能量、力和支持域安全门。
- [ ] 已定义模型/配置/缓存哈希。
- [ ] 已准备无 ML 的 mock Force。
- [ ] 已准备独立输出目录。

## 14. 完成定义

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
