# 外层 λ 神经基势实验日志

> **文档角色：实验事实与决策记录。** 方法原则见
> [`PLAN_outer_lambda_neural_basis.md`](PLAN_outer_lambda_neural_basis.md)，实施任务见
> [`IMPLEMENTATION_PLAN_outer_lambda_neural_basis.md`](IMPLEMENTATION_PLAN_outer_lambda_neural_basis.md)。
> 本文件只记录已经实际执行的实验，不把计划或推测写成已完成结果。

## 1. 记录规则

实验使用唯一编号：

```text
EXP-000
EXP-001
EXP-002
...
```

状态只允许：

- `PLANNED`
- `RUNNING`
- `PASSED`
- `FAILED`
- `INCONCLUSIVE`
- `SUPERSEDED`

每次实验必须：

- 保存完整配置和实际命令；
- 保存代码、模型和输入哈希；
- 使用独立输出目录；
- 区分观测事实与解释；
- 给出继续、回滚或停止的明确决定。

禁止：

- 覆盖旧实验目录；
- 只记录成功实验；
- 把接口测试或 warmup 当成 production 证据；
- 隐藏 NaN、异常力、结构异常或性能倒退；
- 没有独立重复就声称“稳定提升”。

## 2. 总体状态

| 项目 | 当前值 |
|---|---|
| 总体阶段 | WP-4 进行中：真实 MACE 基势离线评价层已跑通，MD Force 后端尚未完成 |
| 当前基础势 | `softcore` 原型；生产模块尚未接入神经路径 |
| 当前目标窗口 | 待确定 |
| 当前目标慢自由度 | 待确定 |
| 当前神经基势数量 | 1 个候选：`mace-off24-medium` 局部三 context 分解 |
| 神经路径协议版本 | 独立模块 v1 已实现；production 未接入 |
| 已完成实验数 | 4 |
| 已通过实验数 | 4（均为独立模块/前置验证，不是 production 验收） |
| 当前是否允许 production | 否 |
| 最近更新日期 | 2026-07-30 |

## 3. 决策日志

| 日期 | 编号 | 决策 | 依据 | 影响 |
|---|---|---|---|---|
| 2026-07-29 | DEC-000 | 建立科学总纲、实施计划、实验日志三层文档 | 分离原则、任务和证据 | 后续实验统一按本文件记录 |
| 2026-07-30 | DEC-001 | WP-1/2 和 WP-3 部署骨架先保持在独立模块，不接入生产模块 | 独立测试覆盖端点、账本、TorchForce/OpenMM；`test_production_modules_do_not_import_standalone_neural_module` 明确保证隔离 | production 未导入不是缺陷；成熟后才合并 |
| 2026-07-30 | DEC-002 | 将三批真实 MACE 运行登记为 WP-4 前置可调用性验证 | 1/10/50 帧均完成，能量和力有限、重复一致、路径端点严格归零 | 可以继续修正局部坐标/选择协议和设计每步 OpenMM Force；不能据此宣称 WP-4 qualification 或 WP-5 完成 |

## 4. 基础路径登记

### BASE-000：当前已知生产基线

| 字段 | 值 |
|---|---|
| 状态 | 历史产物，待重新登记为神经路径对照 |
| mode | `ibs` |
| decoupling | `dual_lambda` |
| potential | `softcore` |
| DEXP params | `null` |
| 主要输出目录 | `output_lrc_fix` |
| 可否直接作为完整神经对照 | 否；尚缺性能、慢变量和独立重复登记 |

待补：

- [ ] 代码哈希；
- [ ] 完整运行命令；
- [ ] Stage 2 λ 和窗口；
- [ ] 每窗口生产步数；
- [ ] 每窗口 ESS；
- [ ] GPU 和软件版本；
- [ ] ns/day；
- [ ] 目标慢自由度；
- [ ] 异常结构率；
- [ ] 独立重复。

## 5. 实验索引

| 实验编号 | 日期 | 工作包 | 简述 | 状态 | 结论 |
|---|---|---|---|---|---|
| EXP-000 | 待定 | WP-0 | 冻结基础路径与选择困难窗口 | PLANNED | 待执行 |
| EXP-001 | 2026-07-30 | WP-1 | 外层 λ 控制器、端点归零和协议哈希测试 | PASSED | 独立协议 v1 的数学契约已实现 |
| EXP-002 | 2026-07-30 | WP-2 | 解析 mock 基势、力和 IBS target/bias/base 账本测试 | PASSED | 端点、同步 finite gate 和账本分离已闭合 |
| EXP-003 | 2026-07-30 | WP-3 | TorchForce + CustomCVForce 独立部署测试 | PASSED | 通用部署骨架已跑通；不代表真实模型 production Force 已完成 |
| EXP-004 | 2026-07-30 | WP-4 前置 | 真实 MACE 局部基势 1/10/50 帧离线评价 | PASSED | 真实模型可调用性验证成功；不是 WP-4 qualification |
| EXP-005 | 待定 | WP-4 | 坐标/PBC/固定选择/support-domain/offset 修正与短 NVT | PLANNED | 待执行 |
| EXP-006 | 待定 | WP-5 | 单困难窗口三组 IBS 对照 | PLANNED | 尚未进入 |

---

## 6. EXP-004：真实 MACE 局部基势离线可调用性

### 6.1 基本信息与研究问题

| 字段 | 值 |
|---|---|
| 状态 | PASSED |
| 日期 | 2026-07-30 |
| 对应工作包 | WP-4 前置 |
| 模型 | `mace-off24-medium` |
| 模型 SHA-256 | `e5ccf5837f685899811a68754e7c994393bfd1a81720393b03c643b46c70bc69` |
| 局部选择 | ligand 41 原子 + 固定 environment 255 原子 |
| 分解定义 | `E(complex)-E(ligand)-E(environment)`，力按同一线性组合映射回全坐标 |
| 外层系数 | `0.1` |
| λ schedule | `0,0.25,0.5,0.75,1` |
| energy offset | `0.0 kJ/mol`（尚未标定） |

唯一研究问题：

> 现有真实 MACE 模型能否通过 `ExistingOrbMaceBasisAdapter` 进入独立外层 λ
> 离线评价链路，并给出有限、可重复、端点严格归零的能量和力？

本实验不回答：

- MACE 路径能否改善 IBS、ESS 或 ΔG；
- 当前局部选择和未成像坐标是否已经满足正式支持域；
- 三 context 评价后端能否直接用于每个 MD step；
- 当前系数、offset 或安全阈值是否为 production 值。

### 6.2 输入与代码身份

| 项目 | 路径 | SHA-256 |
|---|---|---|
| 独立模块 | `outer_lambda_neural_basis.py` | `64f6b13a20ab01efa254403f16df3c48f5f9321a88b77f4ddf973c54a83e5a07` |
| topology | `output/topology.cif` | `27f78ec5fe761a27807a76e262a3d5efc7b5faff58d2b6a5a40ee8085768dde1` |
| trajectory | `output/pre_equilibration.dcd` | `1a28f2e076e110af861248345d854a19163f10e293e85b53bec937d79c5fc9f8` |
| 固定选择 meta | `output/dexp_experiment/fit_label_cache_meta.json` | `8f0d8af51f44804f50b7e9e665aa7b0ff1e761df1a9d3b0e88dc1882754c5d23` |
| MACE 模型 | `~/.cache/mace/MACE-OFF24_medium.model` | `e5ccf5837f685899811a68754e7c994393bfd1a81720393b03c643b46c70bc69` |

运行入口：

```bash
CUDA_VISIBLE_DEVICES=0 FRAME_SPEC=last \
  RUN_DIR="$PWD/output/outer_lambda_existing_model/mace_smoke" \
  bash run_outer_lambda_existing_model.sh

CUDA_VISIBLE_DEVICES=0 FRAME_SPEC=400:500:10 \
  RUN_DIR="$PWD/output/outer_lambda_existing_model/mace_10frames" \
  bash run_outer_lambda_existing_model.sh

CUDA_VISIBLE_DEVICES=0 FRAME_SPEC=tail:50 \
  RUN_DIR="$PWD/output/outer_lambda_existing_model/mace_tail50" \
  bash run_outer_lambda_existing_model.sh
```

### 6.3 三批运行结果

| 批次 | frame | 数量 | 总评价时间 s | s/frame | Basis 能量均值 ± SD kJ/mol | 最大 Basis 力 kJ/mol/nm | 端点能量/力 |
|---|---|---:|---:|---:|---:|---:|---|
| smoke | 499 | 1 | 38.741 | 38.741（含初始化） | -301.155 ± 0.000 | 1448.773 | 精确 0 / 精确 0 |
| 10-frame | 400:500:10 | 10 | 46.903 | 4.690 | -278.159 ± 25.831 | 2457.730 | 全部精确 0 / 全部精确 0 |
| tail-50 | 450–499 | 50 | 211.529 | 4.231 | -291.813 ± 26.579 | 2457.730 | 全部精确 0 / 全部精确 0 |

50 帧详细范围：

- Basis energy：`[-344.708, -236.595] kJ/mol`；
- Basis 最大原子力：`[874.800, 2457.730] kJ/mol/nm`；
- 50 帧均无非有限能量或力；
- 第 499 帧在独立 smoke 与 tail-50 中的能量差约
  `7.3e-11 kJ/mol`，最大力完全一致；
- 系数 `0.1` 下观测到的最大路径附加力为
  `245.773 kJ/mol/nm`。

证据目录：

- `output/outer_lambda_existing_model/mace_smoke/`
- `output/outer_lambda_existing_model/mace_10frames/`
- `output/outer_lambda_existing_model/mace_tail50/`

### 6.4 观测限制与解释

直接观测：

- 真实 MACE 局部分解能量和守恒力已成功进入外层 λ 离线评价层；
- 三批结果有限、同帧重复一致，外层 λ=0/1 能量和附加力严格归零；
- 持久 Context 稳态成本约 `4.2–4.7 s/frame`，明显不适合作为每步 MD
  后端；
- 配置未声明 `support_domain`，所以报告中的零 support violation 不能解释为
  正式支持域通过；
- 原始未成像固定选择在 frame 499 的最大两原子距离为 `43.982 nm`，
  明显是 PBC/坐标成像问题，不是合理局部几何尺度；
- `energy_offset=0`，尚未执行路径中心化和幅度标定。

结论边界：

> EXP-004 只证明“真实 MACE 基势评价层可调用、有限、可重复且端点契约正确”。
> 它不构成短 NVT、IBS、自由能、WP-5 或 production 证据。

### 6.5 决策与后续行动

总体状态：`PASSED`（WP-4 前置可调用性门）。

| 后续行动 | 完成条件 | 状态 |
|---|---|---|
| 修复局部选择的 PBC 成像和固定水/环境身份协议 | 局部几何连续，跨帧身份固定，支持 triclinic PBC | PLANNED |
| 用代表性轨迹定义 support domain | 阈值预注册，违规统计有实际含义 | PLANNED |
| 标定 energy offset 和外层幅度 | 能量中心、力分位数和安全门预注册 | PLANNED |
| 将成熟基势变成每步可执行 OpenMM Force | 单一/共享模型后端，不依赖三 context 每帧 probe | PLANNED |
| 接入 IBS 现场采样和 cross-state ledger | 仅在独立 Force/NVT 门通过后执行 | PLANNED |
| WP-5 三臂实验 | baseline / λ relayout / neural path 全部完成 | PLANNED |

---

## 7. 单次实验模板

复制本节，并将标题改为实际实验编号。

## EXP-XXX：实验标题

### 7.1 基本信息

| 字段 | 值 |
|---|---|
| 状态 | PLANNED |
| 开始时间 | |
| 结束时间 | |
| 执行者 | |
| 对应工作包 | |
| 输出目录 | |
| 主机/GPU | |

### 7.2 唯一研究问题

> 待填写。

本实验不回答：

- 待填写；
- 待填写。

### 7.3 预注册假设

成功假设：

> 待填写。

失败假设：

> 待填写。

### 7.4 预注册验收门

| 验收项 | 阈值/条件 | 硬门 |
|---|---|---|
| 端点能量 | | 是 |
| 端点力 | | 是 |
| 非有限能量/力 | 0 次 | 是 |
| 最大附加力 | | 是 |
| ESS/GPU-hour | | 是 |
| 目标慢变量转换 | | 待定 |
| 端点 ΔG 一致性 | | 是 |

验收门必须在运行前填写，不能在看到结果后修改。

### 7.5 输入、模型和版本

| 项目 | 路径/版本 | SHA-256 或标识 |
|---|---|---|
| 代码 | | |
| 配置 | | |
| system XML | | |
| topology | | |
| 初始坐标 | | |
| 神经模型 0 | | |
| 神经模型 1 | | |
| 原子选择 | | |
| λ schedule | | |
| 外层系数 | | |

软件环境：

| 软件 | 版本 |
|---|---|
| Python | |
| OpenMM | |
| OpenMM-Torch | |
| OpenMM-ML | |
| MACE | |
| PyTorch/LibTorch | |
| CUDA driver/runtime | |

### 7.6 实际命令

```bash
# 必须填写实际执行命令。
```

### 7.7 Hamiltonian 定义

基础路径：

\[
H_\lambda^0 =
\]

神经路径：

\[
B_\lambda =
\]

包络：

\[
w(\lambda) =
\]

系数摘要：

| λ | \(A_{\lambda,0}\) | \(A_{\lambda,1}\) | 备注 |
|---:|---:|---:|---|
| | | | |

### 7.8 运行完整性

| 检查 | 结果 | 证据路径 |
|---|---|---|
| 输入哈希匹配 | | |
| 模型哈希匹配 | | |
| checkpoint 恢复 | | |
| production `f_k` 锁定 | | |
| target/bias/base 帧数一致 | | |
| 非有限帧 | | |
| 能量查询失败 | | |
| 轨迹完整 | | |

### 7.9 端点与力学

| 指标 | λ=0 | λ=1 | 阈值 | 结论 |
|---|---:|---:|---:|---|
| 最大能量差 kJ/mol | | | | |
| RMS 力差 kJ/mol/nm | | | | |
| 最大原子力差 kJ/mol/nm | | | | |
| 有限差分相对误差 | | | | |

### 7.10 神经能量与力

| 指标 | Basis 0 | Basis 1 | 总路径项 |
|---|---:|---:|---:|
| 能量均值 kJ/mol | | | |
| 能量标准差 | | | |
| 能量 P95 | | | |
| 最大绝对能量 | | | |
| 力 RMS kJ/mol/nm | | | |
| 力 P95 | | | |
| 最大力 | | | |
| 支持域违规次数 | | | |

### 7.11 采样与自由能

| 指标 | 基础路径 | 仅 λ 重排 | 神经路径 | 单位/备注 |
|---|---:|---:|---:|---|
| ΔG | | | | kJ/mol |
| 报告误差 | | | | kJ/mol |
| importance ESS ratio | | | | |
| absolute ESS | | | | |
| 去相关样本数 | | | | |
| round trip | | | | |
| 慢变量转换次数 | | | | |
| 自相关时间 | | | | |
| ns/day | | | | |
| GPU-hour | | | | |
| ESS/GPU-hour | | | | |

### 7.12 构象与稳定性

| 检查 | 基础路径 | 神经路径 | 结论 |
|---|---:|---:|---|
| 异常键长 | | | |
| 原子重叠 | | | |
| 配体逃逸 | | | |
| 口袋异常塌缩 | | | |
| 水/离子异常占位 | | | |
| NaN/积分器失败 | | | |

### 7.13 性能分解

| 项目 | 时间/显存 |
|---|---:|
| Context 创建时间 | |
| 基础 MD step 时间 | |
| 神经基势推理时间 | |
| probe energy 时间 | |
| 总 step 时间 | |
| 主 Context 显存 | |
| probe Context 显存 | |

### 7.14 观测事实

- 待填写；
- 待填写；
- 待填写。

这里只写直接观测，不写原因推测。

### 7.15 解释

- 待填写；
- 待填写。

必须指出哪些解释仍是推断。

### 7.16 验收结论

| 验收项 | PASS/FAIL | 依据 |
|---|---|---|
| 端点 | | |
| 力学 | | |
| 能量账本 | | |
| 稳定性 | | |
| 采样改善 | | |
| ESS/GPU-hour | | |
| 自由能一致性 | | |

总体状态：

```text
PASSED / FAILED / INCONCLUSIVE
```

### 7.17 决策

- [ ] 进入下一工作包；
- [ ] 保持复杂度，补充独立重复；
- [ ] 调整外层系数后重试；
- [ ] 重新训练同一目标基势；
- [ ] 回滚到基础路径；
- [ ] 停止该方向；
- [ ] 其它：待填写。

决策理由：

> 待填写。

### 7.18 后续行动

| 行动 | 负责人 | 截止条件 | 状态 |
|---|---|---|---|
| | | | |

---

## 8. 跨实验汇总

至少三个可比独立重复完成后填写：

| 方案 | 重复数 | ΔG 均值 | 重复间 SD | ESS/GPU-hour | 慢变量转换 | 异常率 |
|---|---:|---:|---:|---:|---:|---:|
| 基础路径 | | | | | | |
| 仅 λ 重排 | | | | | | |
| 单神经基势 | | | | | | |
| 多神经基势 | | | | | | |

## 9. Production 准入

- [ ] 禁用神经路径时旧行为回归。
- [ ] 端点能量和力严格一致。
- [ ] target/bias/base 账本闭合。
- [ ] 公共 λ 状态跨窗口一致。
- [ ] 模型和系数进入缓存指纹。
- [ ] GPU 稳定性测试通过。
- [ ] 至少 3 个独立重复。
- [ ] ESS/GPU-hour 优于基础路径。
- [ ] 收益不能被仅 λ 重排替代。
- [ ] complex/solvent 端点循环一致。

最终决定：

```text
NOT_READY / READY_FOR_LIMITED_PRODUCTION / READY_FOR_PRODUCTION / STOPPED
```

决定日期：

```text
待填写
```

依据：

> 待填写。
