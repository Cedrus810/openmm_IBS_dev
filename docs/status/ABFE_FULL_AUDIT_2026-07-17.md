# Atenolol-rank11 ABFE/IBS 项目全量物理与代码审计

- 审计日期：2026-07-17（Asia/Tokyo）
- 审计范围：当前工作区 `K:\ABFE_IBS\Atenolol-rank11`
- 审计对象：输入结构与拓扑、热力学循环、复合物/溶剂体系构建、IBS/MBAR/TMBAR、Boresch、LJ LRC、PME/APBS、缓存与续跑、DEXP 实验分支、脚本/配置与测试；按用户要求排除历史结果和未完成运行的数值判断
- 审计方式：源码静态分析、输入/输出数据一致性检查、JSON/AST 检查、可用的小型数值核对；未修改任何源码或模拟产物

## 1. 一句话结论

**忽略古老的 `output/` 与尚未算完的 `output_lrc_fix/` 后，当前源码和体系构建仍有需要修复的物理/代码问题。**

最重要的是：

用户已经澄清：apo 受体本身就是 `ASH`，游离与结合态 ligand 都是中性，质子在结合前后都留在受体上；因此这里没有配体电荷改变或质子转移，**不列为缺陷**。

剩余问题是：

1. 复合物体系约为 0.15 M NaCl，而自动构建溶剂腿的代码没有传 `ionicStrength`，会生成 0 M 纯水体系；
2. `single_lambda`/传统 REMD 的 LRC 路径调用不存在的函数，且 producer/worker 字段协议不一致；
3. constraint correction、LRC 和最终聚合仍存在 fail-open 路径；
4. DEXP 元数据、production 灾难恢复、随机种子和若干诊断接口仍有确定性缺陷。

## 2. 严重度与证据口径

| 等级 | 含义 |
|---|---|
| P0 | 当前结果不可用、默认工作流可产生科学上无效结果，或存在静默污染核心结果的高风险 |
| P1 | 高影响缺陷；会使受支持分支崩溃、遗漏重要物理项、破坏热力学循环或采样系综 |
| P2 | 中等影响；诊断/不确定度/复现性不可靠，或只在特定条件触发 |
| P3 | 工程、可维护性、可移植性和仓库卫生问题 |

证据状态：

- **已确认**：可从当前源码或落盘数据直接复现/证明；
- **条件性高风险**：结论依赖目标 pH、所选运行模式等尚未在项目中声明的条件；
- **验证缺口**：没有证据证明错误，但当前证据不足以支持生产或发表用途。

## 3. 发现总表

| ID | 严重度 | 类型 | 状态 | 影响范围 | 摘要 |
|---|---:|---|---|---|---|
| P-02 | P1 | 物理/构建 | 已确认 | 默认溶剂腿构建 | 复合物约 0.15 M NaCl，溶剂腿构建代码默认 0 M 纯水 |
| C-01 | P1 | 代码 | 已确认 | 默认否，替代分支是 | `single_lambda` LRC 调用未定义函数且 task 字段错配 |
| C-03 | P1 | 错误处理 | 已确认 | 潜在默认路径 | constraint Jacobian 失败后静默以 0 继续出最终结果 |
| C-04 | P1 | DEXP/LRC | 已确认 | DEXP 分支 | DEXP 明确跳过 LRC，但最终 JSON 无条件声称 `applied=true` |
| C-05 | P1 | 采样 | 已确认，潜在触发 | production 异常恢复路径 | production 中灾难恢复会最小化、重置速度、改变步长后继续混合样本 |
| C-06 | P2 | LRC | 已确认 | 潜在 | dual LRC 取盒子失败时返回全 0，而结果元数据仍声称已应用 |
| C-07 | P2 | 诊断 | 已确认 | 诊断字段 | endpoint 检查只接受 `(1,1)->(0,0)`，会误判正确的两个分阶段路径 |
| C-08 | P2 | 复现性 | 已确认 | 主 IBS 路径 | 未把记录的环境 seed 接到 integrator/速度初始化 |
| C-09 | P2 | 聚合 | 已确认 | 潜在 | 最终聚合对缺失 stage/component 使用 `0.0` 默认值 |
| C-10 | P2 | 构建 | 已确认，通用性 | Atenolol 当前盒子碰巧安全 | 溶剂盒公式少乘了直径/两侧 padding，只靠 3.5 nm 下限兜底 |
| D-01 | P1 | DEXP | 已确认/缺口 | DEXP 分支 | DEXP 仍是单体系实验分支，不具备生产 ABFE 闭环 |
| H-01 | P3 | 仓库 | 已确认 | 间接 | `.git` 不是完整仓库，且源码备份/旧输出混放，版本身份不清 |

## 4. 明确排除的结果目录

按用户要求，本报告不再审计或评价以下数值：

- `output/`：历史版本产物，过于古老；
- `output_lrc_fix/`：仍在计算，未完成。

后续发现只基于当前源码、输入体系、构建逻辑、配置与静态协议，不用这两个目录的自由能、overlap、误差、Boresch diagnostics 或中间失败状态支撑结论。

另外，用户已澄清以下项目均为有意设置，不列为缺陷：

- `rank11` 目录中保留 `rank1` 文件名，是复制到新目录继续处理造成的命名差异；
- `ASH + neutral ligand` 是已确认的游离态与结合态微态；
- 原生 System cache 刻意放宽 raw-input 指纹校验，是为了 debug 时方便快速复用；正式运行由操作者显式重建/`--reset` 管理。

## 5. 物理与建模问题

### 已澄清：ASH/ligand 质子状态不是缺陷

用户确认的热力学状态为：

```text
apo:      RH + L
complex:  RH···L
```

其中受体在游离态本来就是 `ASH`（`RH`），ligand 在游离与结合态都保持中性 `L`；质子在结合前后都属于受体，没有 `LH+ -> L`、`R- -> RH` 或净电荷变化。因此不需要额外的 proton-transfer leg，也不触发带电 ligand 的 PME finite-size correction。

本地结构与这一设定相容：ligand 总电荷约为 0；蛋白 residue 85 为 `ASH`；`ASH OD2-HD2` 约 0.099 nm，并与 ligand 构成紧密氢键网络。用户还已经比较过质子放在两侧的方案并确认受体侧更准确。

唯一仍可记录为方法边界、但不列为 bug 的是：经典固定拓扑把 `HD2` 共价固定在 ASH，表示受体侧定域近似，不描述质子动态离域/跳转。既然目标态就是用户确认的固定微态，这不构成当前 ABFE 热力学循环缺陷。

### P-02 — P1：复合物腿与溶剂腿盐浓度不一致

复合物输入 `solv_ions.gro`：

- 盒长：9.09947 nm，体积约 753.439 nm³；
- 水：22,923 个 TIP3P；
- Na：68；Cl：75；
- 68 对盐对应约 0.1499 M，额外 7 个 Cl 与溶质净正电中和相符。

自动生成的溶剂腿 `output/topology_solvent.cif`：

- 配体：41 原子；
- 水：4,035 原子，即 1,345 个水；
- Na/Cl：**0 个**。

代码证据：[`runabfe.py:570`](../../runabfe.py#L570) 调用：

```python
modeller.addSolvent(ff, boxSize=...)
```

没有传 `ionicStrength`。OpenMM API 的默认值是 0 M，并只在显式指定时加入额外离子对：[OpenMM Modeller.addSolvent 文档](https://docs.openmm.org/development/api-python/generated/openmm.app.modeller.Modeller.html)。

影响：两个腿不是同一个体相化学势条件。即使 ligand 在两腿都中性，离子强度仍可能影响溶剂化、口袋静电屏蔽和结合自由能。

**建议：** 在配置中新增唯一的 `ionic_strength_molar`，复合物与溶剂腿都从该值构建；在 final gate 中比较两个拓扑的离子种类、目标体相盐浓度、水模型和非键参数。不要只比较是否电中性。

## 6. 核心代码缺陷

### C-01 — P1：`single_lambda`/传统 REMD 的 LRC 路径确定性损坏

涉及位置：

- [`ibs_engine.py:568`](../../ibs_engine.py#L568)
- [`ibs_engine.py:9684`](../../ibs_engine.py#L9684)
- [`ibs_engine.py:9841`](../../ibs_engine.py#L9841)

问题有两层：

1. `TraditionalMBARAnalyzer.compute_u_kn()` 调用 `_lj_tail_correction_S_kj_nm6(...)`，但整个项目没有该函数的定义或导入；非 PME Coulomb/VDW 传统路径会在这里 `NameError`。
2. 即使只修函数名，producer 仍传旧字段：

```text
lj_tail_prefactor_kj_nm3_mol
lj_tail_lambda_vdw_power
```

worker `_compute_u_kn_chunk()` 只读取新字段：

```text
lj_tail_lrc_coeff_kj_mol
```

因此 LRC 系数不会到达 worker。metadata 还写旧模型 `analytic_mean_field_r6_tail`，与全局协议 v2 的 switching+softcore-aware 语义不一致。

影响：`--decoupling single_lambda`、传统 2D/mixed/VDW REMD 等受支持路径不可用。默认 `dual_lambda` 的 Stage 1/2 不走此段，所以这不是旧默认结果的直接根因。

**建议：** 删除旧 prefactor 分支，统一调用 `_lj_tail_correction_moments_kj_nm6_nm12()` + `_lj_tail_lrc_coefficients_kj_mol()`，producer/worker 只保留一个 schema；新增一个真实周期小体系测试，断言单 worker/多 worker 的逐帧 `u_kn` 和 LRC 完全一致。

### C-03 — P1：constraint Jacobian 修正失败时按 0 继续

[`abfe_pipeline.py:2905`](../../abfe_pipeline.py#L2905) 中：

```python
cons_correction = 0.0
try:
    cons_correction = calculate_constraint_jacobian_correction(...)
except Exception as e:
    log_warning(...)
```

失败后 pipeline 继续生成 final result，没有 `valid=false`，也不要求用户确认。

**建议：** production 模式下 fail closed；只有显式 `--allow-missing-constraint-correction` 才能以 0 继续，并在顶层结果写 `scientifically_incomplete=true`。

### C-04 — P1：DEXP 跳过 LRC，但最终元数据声称已应用

[`ibs_engine.py:1845`](../../ibs_engine.py#L1845) 对 `potential_type == "dexp"` 明确不计算 LRC，只打印未验证警告；但 [`abfe_pipeline.py:2953`](../../abfe_pipeline.py#L2953) 无条件写：

```json
"lj_long_range_dispersion_correction": {
  "applied": true,
  "status": "implemented_analytic_mean_field_switching_softcore_aware"
}
```

这会把“明确未应用”报告成“已应用”，属于科学 provenance 错误。

**建议：** 从 sampler/leg 返回实际的 LRC applicability/applied/method/delta 字段，聚合层只转发，禁止自行推断；DEXP 在没有经验证的尾项前应 fail closed 或显式标记不完整。

### C-05 — P1：production 中的灾难恢复破坏固定 Hamiltonian/平稳系综

[`ibs_engine.py:7205`](../../ibs_engine.py#L7205) 及余数采样分支在 production 中检测到异常能量/力后：

1. 只恢复 positions；
2. 随机重置 velocities；
3. 做局部能量最小化；
4. 把 timestep 减半；
5. 不重新 equilibration、不丢弃恢复后的过渡段，继续把样本合并到同一 production 集合。

这不再是一个固定动力学协议下的平稳采样。最小化和改变时间步长会改变样本来源；只恢复位置而不恢复速度、box、integrator 随机状态也不是真正 checkpoint rollback。

这是高影响潜在缺陷；是否曾在某次运行中触发不属于本报告的判断范围。

**建议：** production 一旦触发即终止该窗口并标记失败；若要自动恢复，必须恢复完整 checkpoint，使用预先声明的固定 timestep，重新平衡并丢弃 burn-in，且把 recovery 次数写入最终 validity gate。

### C-06 — P2：dual LRC 在盒子读取失败时静默归零

[`ibs_engine.py:2960`](../../ibs_engine.py#L2960) 的 `_lj_tail_correction_kj_mol()` 在读取 box 异常、体积非有限或非正时返回全 0。最终聚合层又无条件声称 LRC 已应用。

对周期 VDW 腿，缺盒子不是“可以当成无修正”的普通情况，应为硬错误。建议在 LRC applicable 时 fail closed，并把每腿实际 correction 统计（均值、范围、端点差）写入 final JSON，而不是 `delta_G_kJ_mol: null`。

### C-07 — P2：lambda endpoint 诊断会误判正确的分阶段路径

[`ibs_engine.py:230`](../../ibs_engine.py#L230) 只把 `(lambda_coul, lambda_vdw)=(1,1)->(0,0)` 认作合法完整路径。

但 dual lambda 两阶段实际是：

- Stage 1：`(1,1)->(0,1)`；
- Stage 2：`(0,1)->(0,0)`。

pipeline 对每个 stage 都调用同一个完整路径诊断，所以正确阶段也得到 `ok=false`。目前它不是 hard gate，因此没有阻断结果，但会污染 diagnostics 并掩盖真正的 endpoint 错误。

**建议：** 让函数接收 `stage_type` 或显式 expected start/end；增加 Stage1、Stage2、single-lambda、2D path 四类单测。

### C-08 — P2：主 IBS 随机种子没有真正接线

provenance 只设计为记录 `OPENMM_RANDOM_SEED`、`ABFE_RANDOM_SEED`、`PYTHONHASHSEED` 环境变量。主 IBS integrator、初始 `setVelocitiesToTemperature()` 和 production recovery 未统一使用这些值，也没有一个正式 `--seed` CLI 控制完整运行。

影响：

- 结果不能 bitwise/轨迹级复现；
- “独立重复使用不同 seed”没有可靠、可审计的操作入口；
- 记录了环境变量也不等于 OpenMM 实际使用了该 seed。

**建议：** 顶层 `master_seed` 经确定性派生生成 integrator、速度、window、probe、replica seed；所有实际 seed 落盘；禁止同一 repeat ID 重用 seed。

### C-09 — P2：最终聚合把缺失组件默认为 0

[`abfe_pipeline.py:2915`](../../abfe_pipeline.py#L2915) 对 `stage1/stage2.total_delta_G`、error 等使用嵌套 `.get(..., 0.0)`。如果某条替代调用路径绕过 stage sanity gate，缺失组件会被解释成“物理贡献恰好为零”。

**建议：** final assembly 对所有必需字段做 schema validation；缺失、null、NaN、Inf 都拒绝出结果。0 只能是上游明确计算并记录的方法学零值。

### C-10 — P2：通用溶剂盒尺寸公式不正确

[`runabfe.py:517`](../../runabfe.py#L517)：

```python
max_r = max(distance(atom, ligand_center))
box_size = max(max_r + 1.5, 3.5)
```

如果 `max_r` 是从中心到最远原子的半径，保证两侧 padding 的立方盒边长应类似：

```text
2*max_r + 2*padding
```

或者直接让 `Modeller.addSolvent(..., padding=...)` 按 bounding box 处理。当前 Atenolol 因 3.5 nm 最小值兜底，实际溶剂盒没有明显越界；但对更大配体公式会低估盒长。

### C-12 — P2：当前测试覆盖没有锁住本次发现的关键接口

现有顶层测试文件只有：

- `test_audit_protocol_regressions.py`
- `test_lrc_interaction_group_compat.py`
- `test_warmup_overlap_protocol.py`

它们没有明显覆盖：

- 复合物/溶剂 ionic strength 必须一致；
- `single_lambda` LRC producer/worker 字段握手；
- DEXP `applied=false` 元数据；
- stage-specific endpoint diagnostics；
- constraint correction 失败必须阻断 final；
- production recovery 不能混入样本；
- binding sign 的落盘回归测试；
- master seed 到所有 OpenMM 随机源的接线。

## 7. DEXP 实验分支评估

### D-01 — P1：DEXP 不是 production-ready ABFE Hamiltonian

项目的 DEXP 调查非常丰富，但证据支持的是“单体系核函数研究”，不是“可用于最终 ABFE 的生产势”。当前限制包括：

1. **没有 DEXP LRC**：主构建函数明确跳过，且元数据还会误报；
2. **没有完整 DEXP ABFE 端到端结果**；
3. **跨体系泛化未验证**：文档计划的 8–15 体系 `--mace-kernel-benchmark` 尚未实现/完成；
4. **平衡性不足**：文档记录初始 15 条 replica 都没有通过平衡性判据；后续 V/S/B 多初态 5 ns 运行中，两个 DEXP 条件仍强烈依赖初态；
5. **结合位点氢键网络改变**：旧分析中酰胺 N–H 原伙伴占有率约从 79% 降到 23–25%，新伙伴从约 9% 升到 39–48%；随后分析又表明这些百分比尚不是可信平衡量；
6. **参考模型不是实验真值**：MACE 局部、截断团簇上的能量/力/Hessian 拟合优于 LJ，不自动等价于蛋白口袋中自由能更准确。

证据：[`DEXP_KERNEL_PHYSICS_ISSUES.md`](../experiments/DEXP_KERNEL_PHYSICS_ISSUES.md)，尤其是单体系范围、氢键伙伴切换、未收敛 occupancy、8–15 体系 benchmark 与端到端 ABFE 待办部分。

**结论：** DEXP 可以继续作为研究分支；在 LRC、跨体系验证、平衡采样和完整 ABFE cycle 都通过前，不应与默认 softcore 结果混作同等级生产结果。

## 8. 仓库与数据管理问题

### H-01 — P3：版本与产物身份不清

- `.git` 目录只有不完整信息，`git status` 报“not a git repository”；无法确定 commit、diff 或源码历史；
- 同目录存在 `.bak`、`.pre_warmup_overlap_patch`、`dexp_experiment1 - 副本.py`、diff 文件和编辑器备份；
- README/AUDIT 文档按日期追加，部分早期“尚未开始/未修复”描述与后续完成状态同时存在，机器和人都容易读错。

**建议：** 恢复完整 Git 仓库；给每次运行使用独立不可变 run ID；final JSON 保存 commit SHA、dirty diff hash、递归输入 hash、protocol schema；旧产物移到只读 archive 并附 manifest，不用普通目录名 `output` 表示“最新”。

## 9. 已通过或没有发现明显问题的检查

以下项目在本次静态/数据检查中表现正常：

- 18 个顶层 Python 文件都能通过 AST 语法解析；
- 扫描到的 227 个 JSON 文件都能正常解析；
- 复合物 `.gro` 头部原子数 73,536 与按 residue/拓扑汇总一致；
- 配体 ITP 为 41 原子，电荷求和数值稳定地给出中性态；
- 复合物约 0.15 M 盐的离子计数与盒体积自洽；
- 复合物盒长 9.10 nm，对 1.2 nm 非键 cutoff 没有明显 minimum-image 尺寸违规；按当前坐标估算最小溶质周期镜像表面间距约 2.46 nm；
- 当前 Stage cache 指纹已经比早期版本完整，包含 system/topology/coordinates/code、run config、Boresch、协议版本和阈值；
- 当前 energy frame gate 使用 `np.isfinite`，早期只拦 NaN、不拦 Inf 的问题已修复；
- 当前预优化异常会向上抛出，不再静默降级为线性路径；
- 当前 overlap failure 分类要求真实 ESS 诊断，不再把任意 solver/warmup 失败错误解释为可自动插 lambda。

## 10. 建议修复顺序

### 第一阶段：先阻止错误结果产生

1. 统一复合物/溶剂 ionic strength；
2. constraint Jacobian 与 applicable LRC 失败都改为 fail closed；
3. 修复 DEXP LRC 元数据误报。

### 第二阶段：修复确定性代码缺陷

1. 重写 `single_lambda` LRC producer/worker 接口并补集成测试；
2. stage-specific endpoint diagnostics；
3. 移除 production 内“最小化后继续混样”的灾难恢复；
4. final result 使用严格 schema，禁止缺失组件默认为 0；
5. 修正溶剂盒尺寸公式；
6. 统一 master seed 并记录所有派生 seed。

### 第三阶段：重新生成科学结果

1. 使用已确认的 `apo ASH + neutral ligand` / `complex ASH···neutral ligand` 状态和统一盐浓度构建两个腿；
2. 正式 production 开始时显式重建需要更新的 debug cache；
3. 在当前 LRC/WCA/IBS 协议下完成复合物与溶剂腿；
4. 所有 warmup、ESS、overlap、decorrelation 和 final gate 通过；
5. 至少 3 个独立 repeat；
6. 报告 sampling、Boresch、repeat、微态和修正项的不确定度；
7. 用独立参考验证 LRC。

### 第四阶段：再考虑 DEXP

1. 实现并验证 DEXP 对应的长程处理；
2. 完成多初态平衡/增强采样，证明 occupancy 不依赖初始态；
3. 完成 8–15 个化学多样体系 benchmark；
4. 至少完成一个标准 softcore 与 DEXP 的配对端到端 ABFE 对照；
5. 在此之前，DEXP 输出必须显式带 `experimental_not_for_production=true`。

## 11. 建议新增的自动门控

每次 run 开始时：

- 配体净电荷必须接近整数，否则报错；
- 复合物和溶剂腿 water model、temperature、pressure、ionic strength 必须一致；
- seed 必须显式给出并落盘。

每个 stage 结束时：

- expected lambda endpoints 精确匹配；
- 所有能量有限；
- warmup bias 已冻结；
- production 中没有 recovery/minimization/timestep change；
- decorrelated samples、ESS、overlap 和 uncertainty 全部过门；
- applicable LRC 实际非空且记录逐态/逐帧统计。

final assembly 时：

- 严格 JSON schema；
- 两腿协议和输入 manifest 配对匹配；
- 必需项禁止默认 0；
- Boresch、constraint、LRC、charge correction 的 applicable/applied 状态一致；
- 结合符号用一个集中定义的函数，并用解析 toy cycle 单测；
- 没有独立 repeats 时明确 `production_valid=false`。

## 12. 本次审计边界

按用户要求，开发机依赖、环境路径、CUDA/conda 配置及当前机器能否执行动态测试均不纳入问题清单。本报告保留当前源码和体系构建可直接确认的问题；开发环境本身不作评价。

## 13. 最终判定

| 对象 | 判定 |
|---|---|
| 历史/未完成运行 | **按用户要求排除，不作判定** |
| 默认 `dual_lambda` 当前源码 | **已有较强 stage fail-closed/指纹基础，但溶剂腿构建和若干 fail-open 路径仍有 P1/P2 问题** |
| `single_lambda`/传统 REMD | **当前 LRC 代码确定性损坏** |
| DEXP | **实验研究分支，不可作为 production ABFE** |
| 整个项目 | **可继续开发；用户澄清项均已排除，剩余重点是盐条件和代码关键分支** |
