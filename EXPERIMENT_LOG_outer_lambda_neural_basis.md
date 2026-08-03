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
| 总体阶段 | WP-0 完成；EXP-009/010/011 均已失败并冻结；EXP-012 已升级为 CV-free 通用局部残差路线，preregistration 仍为 draft |
| 当前基础势 | `softcore` 原型；生产模块尚未接入神经路径 |
| 当前目标窗口 | complex vanishing window 0，Stage 2 states `[0,5)` |
| 历史慢变量 | primary ligand torsion `[4591,4592,4593,4585]` 仅保留为 EXP-010/011 诊断证据；EXP-012 不预设单一 torsion CV |
| 当前训练目标 | 最小化完整 MM 相邻态双向 target-state gap variance；不预设慢 CV，frozen-MACE latent 为主要候选表示 |
| 当前生产候选 | 无；EXP-012 三条逐帧五态 ledger 与 backend 审计已完成，但未冻结/执行 A/B/C/D 表示消融、模型、Force 或 NVT 资格 |
| 神经路径协议版本 | 独立模块 v1 已实现；production 未接入且明确保持隔离 |
| 已完成实验数 | 11；EXP-008 与 EXP-012 尚未执行 |
| 已通过实验数 | 7；EXP-006、EXP-009、EXP-010、EXP-011 失败 |
| 当前是否允许 production | 否 |
| 最近更新日期 | 2026-08-03 |

## 3. 决策日志

| 日期 | 编号 | 决策 | 依据 | 影响 |
|---|---|---|---|---|
| 2026-07-29 | DEC-000 | 建立科学总纲、实施计划、实验日志三层文档 | 分离原则、任务和证据 | 后续实验统一按本文件记录 |
| 2026-07-30 | DEC-001 | WP-1/2 和 WP-3 部署骨架先保持在独立模块，不接入生产模块 | 独立测试覆盖端点、账本、TorchForce/OpenMM；`test_production_modules_do_not_import_standalone_neural_module` 明确保证隔离 | production 未导入不是缺陷；成熟后才合并 |
| 2026-07-30 | DEC-002 | 将三批真实 MACE 运行登记为 WP-4 前置可调用性验证 | 1/10/50 帧均完成，能量和力有限、重复一致、路径端点严格归零 | 可以继续修正局部坐标/选择协议和设计每步 OpenMM Force；不能据此宣称 WP-4 qualification 或 WP-5 完成 |
| 2026-07-30 | DEC-003 | 结束短 smoke 循环，将 100-step 修正版运行作为资格阈值标定；下一次直接执行冻结的 1000-step NVT qualification | 每步 CUDA Force、PBC 成像、path-only 力和纯积分性能均已有可用观测 | 旧 smoke 命令仅保留作连通性诊断；正式推进只看 qualification 硬门 |
| 2026-07-30 | DEC-004 | 不事后放宽 250 kJ/mol/nm 路径力硬门；将外层系数从 0.10 降到 0.09 后执行一次正式重试 | EXP-006 仅路径力失败：258.949 kJ/mol/nm，超阈值约 3.6%；其余五项通过 | EXP-006 保持 FAILED；EXP-007 只改变一个预注册参数 |
| 2026-07-30 | DEC-005 | 冻结系数 0.09 的 MACE 路径作为 WP-5 单基势候选 | EXP-007 六个资格硬门全部通过 | 停止 WP-4 幅度和力学阈值调整；后续评价性能与科学收益 |
| 2026-07-30 | DEC-006 | 完整 MACE 不默认作为每个生产积分步的最终 bias；在 EXP-007 通过基础上单独比较 MTS 与 cheap-CV 蒸馏 | EXP-005/006/007 的实时后端约 0.10 s/step，频率成本仍高；完整局部 MACE 还包含高频原子力，不能未经验证每几十步冻结旧力 | MTS 必须使用独立 force group 并以 \(N=1\) 为参考；若性能门不通过，MACE 转为教师模型 |
| 2026-07-30 | DEC-007 | 将 complex vanishing window 0 和 atenolol `C4-N2-C9-C10` 扭转角冻结为第一组困难窗口/慢变量 | window 0 的最终 Stage 2 最小 ESS ratio 最低、绝对 ESS 最低且统计低效最高；预平衡轨迹中该扭转角观察到 trans/gauche-minus 交换 | EXP-009 三臂必须记录该扭转角；后续 WP-5 首先在该窗口检验转换与 ESS |
| 2026-07-30 | DEC-008 | EXP-009 固定使用 OpenMM BAOAB-rRESPA、`coefficient=0.09`、内步长 0.5 fs、\(N=1/2/4\)，正式脚本每臂 50 ps | 同一坐标、温度和种子可隔离调度差异；200 个诊断样本/臂用于首轮分布门 | \(N=4\) 的最低性能门预注册为 1.0 ns/day；不得依据结果修改本次阈值 |
| 2026-07-30 | DEC-009 | 完整 MACE 的 PythonForce CUDA MTS 路线不晋级，启动 EXP-010 | EXP-007 的普通 LangevinMiddleIntegrator 可运行；EXP-009 的 \(N=1\) 已在 CUDA MTS force-group 内核触发 `CUDA_ERROR_INVALID_HANDLE`。环境中的 openmmml 1.6 和独立实时桥接均使用 PythonForce | 不再重跑同一后端、不调 coefficient；完整 MACE 保留为离线教师，生产候选改为周期 torsion cheap-CV |
| 2026-07-30 | DEC-010 | 撤销 DEC-007 中“首个慢变量已冻结”的部分，只保留困难 window 0 已冻结 | 旧 window 能量账本没有坐标；预平衡 torsion 转换不能证明它造成困难 window 的低 ESS | 新增只读困难窗口 scratch 采样和自动 CV 排序；production CV 必须用稳定原子身份并经困难窗口轨迹验证 |
| 2026-07-30 | DEC-011 | 不凭单条困难窗口轨迹冻结 `VAL251 chi1` 或 hydration；要求至少三条独立种子重复 | run1 中 `VAL251 chi1` 只有 1 次 core 转换、\(N_\mathrm{eff}=3.91\)；hydration 的 \(g=82.19\)、\(N_\mathrm{eff}=6.08\)，两者不确定性都很大 | 新增按 `stable_id` 对齐的重复判定；两条轨迹只能比较，三条且转换可重复才可能晋级 |
| 2026-07-30 | DEC-012 | 冻结 ligand torsion `[4591,4592,4593,4585]` 为 EXP-010 的 primary slow variable；不授予 production approval | 三个独立种子的 core 转换为 14/4/10，重复排名 1；相邻 ligand torsion 和 `VAL251 chi1` 分别为候选 2/3 | WP-0 完成，EXP-010 可开始教师标注；secondary/hydration 保留作二维模型和诊断敏感性分析 |
| 2026-07-30 | DEC-013 | EXP-010 教师改用固定 protein-only environment，移除旧选择中的全部固定水；不放宽支持域门 | 旧 255 环境含 39 个水原子、涉及 14 个水残基且有不完整水；scratch frame 的跨度达 7.63 nm。移除后保留 216 个蛋白原子，CPU 一帧跨度 2.342 nm、Rg 0.728 nm 并通过 | 教师选择成为新的带哈希协议；原 EXP-007 含水选择仍保留为历史证据，不冒充适用于 scratch 数据 |
| 2026-07-30 | DEC-014 | 教师标注前按冻结支持域排除 frame，最大允许排除率 5%；不得调大 2.5 nm / 0.85 nm 门 | 三条轨迹每 5 帧预检共 300 帧，290 帧通过；10 帧跨度 2.50–2.60 nm，Rg 均小于 0.747 nm | MACE 不评价外推帧；数据集完整记录排除身份和比例，超过 5% 整批不得拟合 |
| 2026-07-31 | DEC-015 | EXP-010 记为 `FAILED`，不选择最终 Fourier 模型，不执行 cheap-CV NVT 或 WP-5 | 六个预注册候选的 leave-one-run-out 能量 RMSE 均未优于 intercept-only `21.5109 kJ/mol`；最好的 1D order 2 为 `22.1737 kJ/mol` 且广义力 \(R^2=-13.5934\)，2D 候选明显病态 | 不事后增加 Fourier order、不以训练集或 conditional-bin 表现晋级；任何新 CV、目标量、正则或模型族必须另立实验 |
| 2026-07-31 | DEC-016 | 修正 EXP-010 失败归因：它否定当前 atom-cut protein MACE 教师与逐帧总 interaction-energy 教学协议，不否定 torsion bias 本身 | 216 个 protein atoms 由单帧 0.5 nm 逐原子选择得到；实际涉及 26 个残基且完整残基数为 0。primary torsion 对教师 Cartesian force 的平均投影解释比仅 `0.00393`，三条 run 还有显著能量偏移 | 登记 EXP-011：首选从完整 MM Hamiltonian 学习条件平均力/PMF；若以后再用 MACE，必须另立实验验证完整残基/封端和环境半径收敛 |
| 2026-08-02 | DEC-017 | 冻结 EXP-011 目标加权周期 PMF 协议；历史三 run 仅用于覆盖诊断，不用空 bin 产模 | 7/24 pooled bins 为空；三 run 周期有效样本数为 141.41/6.62/3.76；最低 pairwise Bhattacharyya overlap 为 0.328 | 状态为 `PREREGISTERED / SAMPLING_REQUIRED`；下一步受限/增强采样，WP-5 保持未开始 |
| 2026-08-03 | DEC-018 | 冻结 EXP-011 失败并登记 EXP-012：完整 MM 基线 + XED-inspired 局部场 + ligand-only 短程路径残差 | AUG-001 后 mutual overlap 0.02353 < 0.03 且 22 个去相关样本 < 25；EXP-010 fragment-total MACE 边界不闭合 | 不再补采/拟合 EXP-011；EXP-012 先做 atom-centered / +XED / +overlap 三层消融，未证明增量前不训练完整 MACE、不进入 WP-5 |
| 2026-08-03 | DEC-019 | 将 EXP-012 从 XED 专项升级为 CV-free 通用局部残差路径势；frozen-MACE latent 作为主要候选，XED 降为可选 Arm D | MACE 可提供中间层 node descriptors，但在线守恒力必须保留 encoder 对坐标的 autograd；单 CV PMF 已在 EXP-011 失败，gap variance 可直接针对 overlap 瓶颈 | 保留 `exp012_xed` 草案哈希与 schema 证据，sealed 前迁移为 local-residual A/B/C/D 协议；新增 L1 在线 MACE 与 L2 latent-teacher student 对照，WP-5 分体系内和跨 ABFE 通用性验收 |

## 4. 基础路径登记

### BASE-000：当前已知生产基线

| 字段 | 值 |
|---|---|
| 状态 | 历史 production 产物；已用独立 scratch 重建生成同窗口筛选轨迹 |
| mode | `ibs` |
| decoupling | `dual_lambda` |
| potential | `softcore` |
| DEXP params | `null` |
| 主要输出目录 | `output_lrc_fix` |
| 可否直接作为完整神经对照 | 否；scratch 轨迹可用于 CV/教师研发，但不是历史 production checkpoint 的严格续算 |

待补：

- [ ] 代码哈希；
- [ ] 完整运行命令；
- [x] Stage 2 λ 和困难窗口；
- [ ] 每窗口生产步数；
- [x] 每窗口 ESS；
- [ ] GPU 和软件版本；
- [ ] ns/day；
- [x] 目标慢自由度定义和三种子重复；
- [ ] 异常结构率；
- [ ] 独立重复。

## 5. 实验索引

| 实验编号 | 日期 | 工作包 | 简述 | 状态 | 结论 |
|---|---|---|---|---|---|
| EXP-000 | 2026-07-30 | WP-0 | 冻结基础路径、困难窗口并筛选首个慢变量 | PASSED | window 0 与 primary ligand torsion `[4591,4592,4593,4585]` 已由三种子证据冻结 |
| EXP-001 | 2026-07-30 | WP-1 | 外层 λ 控制器、端点归零和协议哈希测试 | PASSED | 独立协议 v1 的数学契约已实现 |
| EXP-002 | 2026-07-30 | WP-2 | 解析 mock 基势、力和 IBS target/bias/base 账本测试 | PASSED | 端点、同步 finite gate 和账本分离已闭合 |
| EXP-003 | 2026-07-30 | WP-3 | TorchForce + CustomCVForce 独立部署测试 | PASSED | 通用部署骨架已跑通；不代表真实模型 production Force 已完成 |
| EXP-004 | 2026-07-30 | WP-4 前置 | 真实 MACE 局部基势 1/10/50 帧离线评价 | PASSED | 真实模型可调用性验证成功；不是 WP-4 qualification |
| EXP-005 | 2026-07-30 | WP-4 标定 | 每步 MACE PythonForce 与复制 System 的 1/10/100-step NVT | PASSED | 每步真实 Force、PBC 和性能口径跑通；用于冻结 EXP-006，不是 qualification |
| EXP-006 | 2026-07-30 | WP-4 | 系数 0.10 的 1000-step MACE NVT qualification | FAILED | 5/6 门通过；路径最大力 258.949 超过 250 |
| EXP-007 | 2026-07-30 | WP-4 | 系数 0.09 的 1000-step MACE NVT qualification | PASSED | 六个硬门全部通过；随后仍需通过调度或蒸馏性能门 |
| EXP-008 | 待定 | WP-5 | 单困难窗口三组 IBS 对照 | PLANNED | 等待 EXP-012 feature、守恒力、NVT 与性能资格通过 |
| EXP-009 | 2026-07-30 | WP-3A/WP-4 | 冻结系数 0.09 的 MTS 调度资格 | FAILED | \(N=1\) 即出现 PythonForce/CUDA MTS 后端错误；没有可用于物理分布比较的 arm |
| EXP-010 | 2026-07-31 | WP-4A | MACE 教师到目标慢变量 cheap-CV bias 的蒸馏 | FAILED | GPU 教师数据集通过支持域门，但六个预注册 Fourier 候选均未通过跨 run 能量/广义力验证 |
| EXP-011 | 2026-08-02 | WP-4B | 完整 MM Hamiltonian 条件平均力/PMF 到周期 torsion bias | FAILED | AUG-001 后 mutual overlap 与去相关样本数仍未过冻结门；`FORMAL_RUN1_OVERLAP_FAILED`，不拟合 PMF |
| EXP-012 | 2026-08-03 | WP-4C | CV-free 通用局部残差路径势：A/B/C/D 表示 + 双向 gap-variance loss | PLANNED | 五态 CUDA ledger、权重账本和 CPU/CUDA audit 已通过，draft 已迁移为 local-residual v2；尚未 seal A/B/C/D、训练模型、建立 Force 或运行 NVT |

---

## 5A. EXP-000：困难窗口与首个慢变量冻结

当前状态：`PASSED`。困难 window 与 EXP-010 primary slow variable 均已冻结；该结论
不授予 production approval。

输入为 `output_lrc_fix/final_results.json`、`pre_equilibration.dcd`、
`rebalance_traj.dcd` 和 `topology.cif`。窗口排序规则在读取结果前固定为：

1. 最小化最终 Stage 2 `min_ess_ratio`；
2. 并列时选择更低的绝对 ESS；
3. 再并列时选择更高的 statistical inefficiency。

实际选择：

| 指标 | 结果 |
|---|---:|
| leg / stage | complex vanishing / Stage 2 |
| window index / state range | 0 / `[0,5)` |
| λvdW | `1.0, 0.923529, 0.854304, 0.790614, 0.731876` |
| minimum ESS ratio | 0.391266 |
| absolute ESS | 37.5616 |
| statistical inefficiency | 5.20498 |
| endpoint-difference uncertainty | 0.924927 kJ/mol |

预平衡阶段候选之一为 atenolol `C4-N2-C9-C10` 周期扭转角：

| 轨迹 | 帧数 | circular mean ± std | basin occupancy | core transitions |
|---|---:|---:|---|---:|
| pre-equilibration | 500 | -169.682° ± 37.800° | trans 0.858；gauche-minus 0.142；gauche-plus 0 | 7 |
| rebalance | 25 | -63.976° ± 7.742° | gauche-minus 1.0 | 0 |

basin 定义为 trans `|phi| >= 120°`、gauche-minus
`-120° <= phi < 0°`、gauche-plus `0° <= phi < 120°`。转换计数另用
trans `|phi| >= 150°`、gauche-minus `[-90,-30]°`、gauche-plus
`[30,90]°` 的 core hysteresis，避免边界抖动被重复计数。

证据：

| 项目 | 路径 | SHA-256 |
|---|---|---|
| WP-0 报告 | `output/outer_lambda_wp0/wp0_selection.json` | `80c6ebafde9394169673a2a2ee8604614270be4356233c9c83eafefa421e4b3e` |
| final results | `output_lrc_fix/final_results.json` | `eac4c9cbf4656df24e92323aa50b94cb32d58b62eb6a4828f0b20026b261b34d` |
| topology | `output_lrc_fix/topology.cif` | `6602f537d13179fc8294bcbaea1c7247fa9148b7d372a6411bc9f705db744ccf` |

限制：现有完成的 IBS window 能量产物没有坐标轨迹，因此不能用预平衡候选替代
困难 window 的慢变量判定。进一步盘点发现 window 0 有原生 Context checkpoint，
但当前代码重建出的 window System SHA
`be34fd38...` 与历史 manifest 的 `f27d2648...` 不同，不能冒充严格续算。

因此新增独立 scratch 路线：只读历史 λ、冻结 f_k、Boresch 和软核参数，从
rebalance 末帧重新 burn-in，在新目录生成坐标；不写回 production checkpoint、
能量或 convergence。CPU 1-step 连通性已完成，`production_data_mutated=false`。

对预平衡轨迹每 10 帧抽样的功能性筛选共发现 8 个 ligand 非环 rotatable torsion
和 27 个 0.6 nm 口袋残基 chi1。该结果只用于验证筛选代码。

正式困难窗口 scratch run1 使用 50 ps burn-in、500 ps sampling、1 ps/帧，共
500 帧；CUDA 实测 116.965 ns/day，且 `production_data_mutated=false`。周期候选
第一为 `VAL251 chi1`：\(g=127.91\)、\(N_\mathrm{eff}=3.91\)、圆周标准差
41.63°，但仅观察到 1 次 core 转换。原先示例 torsion
`[4586,4584,4591,4592]` 排第 19，因此已经排除为首选。

同一 run1 轨迹补算的 ligand 第一水合壳层平滑 coordination 为
均值 2.993、标准差 0.666、\(g=82.19\)、\(N_\mathrm{eff}=6.08\)；它可能有关，
但尚未定义可重复的 wet/dry 状态，不能和周期 torsion 用不同量纲分数直接混排。

| 项目 | 路径 | SHA-256 |
|---|---|---|
| run1 原始筛选 | `output/outer_lambda_slow_variable_screen/hard_window0_run1/candidate_screen.json` | `af27ee07e2056b5500962a72c5ddccf8e29df8a7acd618f5f0d03491d9d68fa8` |
| run1 含 hydration 的 v2 筛选 | `output/outer_lambda_slow_variable_screen/hard_window0_run1/candidate_screen_v2.json` | `5ed88f64f19655e55243cddf0e12d693c99264475df414934e4f103cce62450b` |

run1 当时的结论为 `candidate_ranking_only`；预注册要求至少完成三个独立随机种子，
并按稳定原子身份对齐后，才允许冻结 EXP-010 输入。

三种子重复已经完成。预注册重复门得到三个合格周期候选：

1. ligand torsion `[4591,4592,4593,4585]`，三条轨迹 core 转换
   `14/4/10`，\(g=3.54/75.54/132.89\)；
2. 相邻 ligand torsion `[4593,4585,4594,4595]`，转换 `3/14/15`；
3. `VAL251 chi1` `[4020,4022,4024,4026]`，转换 `1/0/2`。

第一、第二 torsion 的离散盆 NMI 在三个种子中为 0.22、0.27、0.58，说明有关联但
不完全冗余。EXP-010 首先使用排名第一的 torsion；第二 torsion 作为二维模型候选，
`VAL251` 只作诊断。hydration 在三条轨迹中的 \(g\) 为
82.19/126.58/34.61，持续保留，但在定义 wet/dry 状态前不进入 production CV。

| 项目 | 路径 | SHA-256 |
|---|---|---|
| 三种子比较 v2 | `output/outer_lambda_slow_variable_screen/replicate_comparison_v2.json` | `fc339493a672f36ce9b5c4917208ac39ac71bb17364c081b5095ff0320cc5a85` |
| EXP-010 慢变量 manifest | `output/outer_lambda_slow_variable_screen/slow_variable_manifest.json` | `b14819c50149cadd4db3b641ba514d72126cfda049dd96351ab80aaf82d3455d` |

manifest 内部协议 SHA-256 为
`f09ecf3786a7c3f65db19181733d7afc2f9489d49a050012173d2ac9bd541805`，
状态是 `frozen_for_exp010_teacher_distillation`，且明确
`production_approval=false`。

### 5A.1 EXP-010 教师数据准备

独立模块现已实现：

- `exp010-prepare-selection`：删除交换水并冻结 protein-only 选择；
- `exp010-label`：按三条独立 run 生成 MACE 能量、primary/secondary torsion 和
  primary 广义力标签；
- `exp010-fit`：拟合 1D/2D 周期 Fourier 守恒势，并按整条 run 留一验证；
- 解析 `CustomTorsionForce` / `CustomCompoundBondForce` 导出、XML round-trip
  和重叠 torsion 测试。

教师能量 offset 使用所有通过支持域帧的冻结数据集均值，只消除常数分量，不调整
`coefficient=0.09`。周期 Fourier 的支持区间是完整
\([-\pi,\pi]\)，不存在非周期边界外推。

CPU 一帧 protein-only 连通性结果：

| 项目 | 值 |
|---|---|
| 教师选择 | ligand 41 + protein environment 216 |
| MACE raw interaction energy | -175.978129 kJ/mol |
| primary torsion | -149.470817° |
| primary generalized force | -8.295638 kJ/mol/rad |
| 教师最大原子力 | 714.669 kJ/mol/nm |
| 支持域 | max pair 2.341981 nm；Rg 0.727548 nm；通过 |
| safety / support violations | 0 / 0 |

| 项目 | 路径 | SHA-256 |
|---|---|---|
| protein-only 选择 smoke | `output/outer_lambda_exp010/smoke_cpu_protein_only/protein_only_selection_meta.json` | `76d25abe03ac47071b3a03bbde4988d83a9f0bcf0142e472f592b49f71393ae5` |
| CPU 一帧教师数据 | `output/outer_lambda_exp010/smoke_cpu_protein_only/teacher_dataset.json` | `4055fcf8142c82f747acda73f8e08c8b3edcdfd5cc3e5b922fcd0f0474855660` |
| 正式节点脚本 | `run_outer_lambda_exp010_teacher_and_fit.sh` | `1d6ffb3129cbd431d8e63241864faa7a40b0c839ddb6824e4cea9cdde8e5f3db` |

正式节点矩阵使用每 run 100 帧，共 300 个 source frames；支持域实际保留
290 帧。拟合候选固定为 1D order 2/4/6 和 2D order 2/3/4，结果返回前未增删
候选。正式结果和失败判定见第 11 节。

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
| 修复局部选择的 PBC 成像和固定水/环境身份协议 | 局部几何连续，跨帧身份固定，支持 triclinic PBC | COMPLETED；EXP-010 改为 protein-only 教师 |
| 用代表性轨迹定义 support domain | 阈值预注册，违规统计有实际含义 | COMPLETED；2.5 nm / 0.85 nm 门保持不变 |
| 标定 energy offset 和外层幅度 | 能量中心、力分位数和安全门预注册 | COMPLETED for EXP-007；cheap-CV 使用数据集均值 offset |
| 将成熟基势变成每步可执行 OpenMM Force | 单一/共享模型后端，不依赖三 context 每帧 probe | DIRECT MACE STOPPED；解析 cheap-CV Force 机制已实现，但 EXP-010 无合格模型可资格 |
| 接入 IBS 现场采样和 cross-state ledger | 仅在独立 Force/NVT 门通过后执行 | PLANNED |
| WP-5 三臂实验 | baseline / λ relayout / neural path 全部完成 | PLANNED |

---

## 7. EXP-005：每步 MACE Force NVT 标定

当前状态：`PASSED`（WP-4 qualification 前置标定，不是 qualification）。

已完成：

| 批次 | CUDA 步数 | λ | 总墙钟 s | 纯积分 s/step | path energy kJ/mol | path-only 最大力 kJ/mol/nm | 最大闭合误差 kJ/mol | 结果 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `step1` | 1 | 0.5 | 8.274 | 旧口径未分离 | -35.327 | 旧口径未分离 | 0.00236 | PASSED |
| `step10` | 10 | 0.5 | 8.989 | 旧口径未分离 | -35.330 至 -33.633 | 旧口径未分离 | 0.01290 | PASSED |
| `step100_v2` | 100 | 0.5 | 21.329 | 0.10218 | 均值 -34.293 | 186.372 | 0.00508 | PASSED |

证据：

- `output/outer_lambda_mace_nvt/step1/nvt_smoke.json`
- `output/outer_lambda_mace_nvt/step10/nvt_smoke.json`
- `output/outer_lambda_mace_nvt/step100_v2/nvt_smoke.json`

已确认：

- MACE 路径 Force 已在真实 OpenMM Context 中随积分器每步执行；
- 无 NaN、无积分器异常；
- base + path 与 total 能量在数值精度内闭合；
- 调用方 base System 未被修改；
- λ=0/1 的端点 fast path 在运行模型前返回严格零。
- 修正版 100-step 报告中 PBC minimum-image 支持几何稳定：
  最大 pair distance `2.154–2.176 nm`，最小 pair distance
  `0.095719 nm`，回转半径约 `0.731 nm`；
- 100 步内 support violation 为 0，但该轮配置没有预先声明 support domain，
  因此它只能用于冻结下一轮阈值。

本轮边界：

- 100 步不足以作为正式稳定性 qualification；
- support domain 和 `energy_offset` 是看到本轮数据后才标定，不能追认本轮通过；
- 没有 IBS、ESS 或 ΔG 证据。

决定：结束 smoke 扩展，进入 EXP-006。

## 8. EXP-006：冻结协议的 MACE NVT qualification

当前状态：`FAILED`。

预注册配置：

| 项目 | 冻结值 |
|---|---:|
| NVT steps | 1000 |
| report interval | 25 |
| timestep | 0.5 fs |
| λ / coefficient | 0.5 / 0.1 |
| energy offset | -343.0 kJ/mol |
| support min pair | 0.07 nm |
| support max pair | 2.5 nm |
| support max radius of gyration | 0.85 nm |

硬门：

| 检查 | 阈值 |
|---|---:|
| 完整运行 | `passed=true` 且 steps ≥ 1000 |
| support domain | 已配置且违规数为 0 |
| path-only 最大原子力 | ≤ 250 kJ/mol/nm |
| 最大能量闭合误差 | ≤ 0.1 kJ/mol |
| 纯积分耗时 | ≤ 0.2 s/step |

实际命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_DIR="$PWD/output/outer_lambda_mace_qualification/run1" \
bash run_outer_lambda_mace_qualification.sh
```

输出：

- `output/outer_lambda_mace_qualification/run1/prepare_report.json`
- `output/outer_lambda_mace_qualification/run1/nvt_qualification.json`

实际结果：

| 检查 | 观测 | 结论 |
|---|---:|---|
| 完整运行 | 1000 steps，`passed=true` | PASS |
| support domain | 已配置，0 次违规 | PASS |
| path-only 最大原子力 | 258.949 kJ/mol/nm | **FAIL** |
| 最大能量闭合误差 | 0.01135 kJ/mol | PASS |
| 纯积分耗时 | 0.09905 s/step | PASS |
| 总墙钟 | 127.127 s | 记录项 |

路径力 40 个报告点的均值、P95 和最大值分别为
`201.803 / 255.883 / 258.949 kJ/mol/nm`；共有 6 个报告点超过 250。
路径能量范围为 `[-3.499, 2.480] kJ/mol`。支持域实际范围：

- minimum pair distance：`0.0957191–0.0957198 nm`；
- maximum pair distance：`2.1535–2.3269 nm`；
- radius of gyration：`0.7302–0.7355 nm`。

身份记录：

| 项目 | SHA-256 |
|---|---|
| 独立模块 | `362d59e2572eb26942201f6c180ca695aad5a09b6ca66434a416f0270a09ae39` |
| qualification 脚本（EXP-006 实际版本） | `a46a0beb3eda5b62e5e0d6df5bb57355ff2b91601f8dc78d63cec5192a006a6f` |
| 生成配置 | `7de6a5d26046ef86d53dfbded600e8603084c078995959a681362fda9c6054a4` |
| qualification 报告 | `5acf16a1fe5eea16ef3aaf081317c300017edd641d3bb3d404e71a5f707ae9ea` |
| system XML | `e2eb7b94fceec5b4cdf552972fc40fa633d49b8e4385acfc62a357e4bfc01717` |

结论：严格保持 `FAILED`，不修改原报告、不提高验收阈值。失败是幅度门，
不是部署、稳定性、支持域或性能失败。

## 9. EXP-007：系数 0.09 的 MACE NVT qualification

当前状态：`PASSED`。

相对 EXP-006 唯一协议变化：

```text
coefficient: 0.10 -> 0.09
```

支持域、offset、初始 frame、1000 steps、报告间隔、积分参数和全部资格硬门保持
不变。在 EXP-006 的相同坐标上，纯线性缩放会将最大路径力从 258.949 降至约
233.054 kJ/mol/nm；新轨迹仍必须独立通过实际硬门，不能用该推算代替运行。

预注册命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
COEFFICIENT=0.09 \
RUN_DIR="$PWD/output/outer_lambda_mace_qualification/run2_coeff009" \
bash run_outer_lambda_mace_qualification.sh
```

实际结果：

| 检查 | 观测 | 结论 |
|---|---:|---|
| 完整运行 | 1000 steps，`passed=true` | PASS |
| support domain | 已配置，0 次违规 | PASS |
| path-only 最大原子力 | 246.775 kJ/mol/nm | PASS |
| 最大能量闭合误差 | 0.01490 kJ/mol | PASS |
| 纯积分耗时 | 0.09771 s/step | PASS |
| 总墙钟 | 125.490 s | 记录项 |

路径力均值/P95/最大值为 `182.632 / 230.522 / 246.775 kJ/mol/nm`；
路径能量范围 `[-3.120, 2.466] kJ/mol`。support-domain 实际范围全部位于
预注册边界内。

| 项目 | SHA-256 |
|---|---|
| 独立模块 | `362d59e2572eb26942201f6c180ca695aad5a09b6ca66434a416f0270a09ae39` |
| qualification 脚本 | `91e59b7549e95d0c446a2abf5e99146eadd2c6a63c52a8b416f4d1b5e92a8d32` |
| 生成配置 | `8ab035799c190910477f4f28a10b9b02eec8e0f733d590031ea7d9a9c474babd` |
| qualification 报告 | `ce27671f6af95ef8f0a0b0d8095a8503e79fa0ea9c56ff81a84aa4a73fa72d8d` |
| prepare 报告 | `de9aed74c41643c1b4cd9fdb8a821ce6d52936789ea1a4ccb889beea7274423b` |

结论：WP-4 完整 MACE 的 NVT qualification 通过。冻结
`coefficient=0.09`、offset、支持域和模型身份；随后 EXP-009 证明当前直接 MTS
后端不晋级，因此实际路线转入 EXP-010，而不是直接开始 WP-5。不再调整 EXP-007
阈值或重复相同 NVT qualification。

---

## 10. EXP-009：完整 MACE 的多时间步性能资格

当前状态：`FAILED`。EXP-007 已通过并冻结 `coefficient=0.09`；本实验没有改变幅度。

唯一研究问题：

> 在不产生可分辨积分偏差的前提下，完整 MACE 路径 Force 能否通过 MTS 降低到足以
> 进入单窗口 IBS 的成本？

第一阶段预注册矩阵：

| coefficient | MTS ratio \(N\) | 当前 0.5 fs 内步长下的 MACE 间隔 |
|---:|---:|---:|
| 0.09 | 1 | 0.5 fs |
| 0.09 | 2 | 1 fs |
| 0.09 | 4 | 2 fs |

降低 coefficient 会减小路径力，但不会降低一次 MACE forward 的成本。因此系数不属于
本性能矩阵的扫描轴；任何新系数都必须另建幅度资格实验。

执行规则：

- \(N=1\) 是每个系数自己的参考，不得省略；
- 使用 OpenMM force group 与 MTS/r-RESPA；不得在 Python 循环中冻结旧力；
- 三臂使用同一初始坐标、300 K、随机种子 `20260730` 和 50 ps 物理时长；
- 正式入口每 0.25 ps 记录一次诊断，共 200 个样本/臂；
- 只有 \(N\le4\) 全部通过，才允许单独探索 \(N=8\)；
- 本阶段必须同时报告稳定性、分布偏差和 ns/day；ESS/GPU-hour 留到有 IBS
  target-energy 账本的 WP-5，不能由普通 NVT 伪造；
- 仅“没有崩溃”不构成通过。

预注册硬门：

| 检查 | 门限 |
|---|---:|
| 完整运行 / finite | 三臂全部完成，无非有限诊断 |
| support domain | 已配置且违规数为 0 |
| path-only 最大原子力 | `<= 250 kJ/mol/nm` |
| 最大能量闭合误差 | `<= 0.1 kJ/mol` |
| 温度均值相对 N=1 | 绝对差 `<= 5 K` |
| 总势能均值相对 N=1 | 标准化差 `<= 0.25` |
| 扭转角分布相对 N=1 | 24-bin Jensen-Shannon divergence `<= 0.05` |
| ligand Kabsch RMSD 均值相对 N=1 | 绝对差 `<= 0.05 nm` |
| N=4 实测性能 | `>= 1.0 ns/day` |

以上分布门是本次 EXP-009 的首轮工程资格判据，不等同于长时间平衡分布证明。
报告必须同时保存每个采样点，便于发现均值门掩盖的漂移。

实现入口：

```bash
CUDA_VISIBLE_DEVICES=0 \
  RUN_DIR="$PWD/output/outer_lambda_mace_mts/exp009_run2_input_fix" \
  bash run_outer_lambda_mace_mts_qualification.sh
```

脚本固定 `coefficient=0.09`，默认
`N_INNER_STEPS=100000`、`REPORT_INTERVAL_INNER_STEPS=500`。开发连通性短跑必须使用
不同输出目录，并且不得登记为 EXP-009 正式结果。

首次设置故障记录（未进入 MTS 比较，不改变本实验 `PLANNED` 状态）：

- `output/outer_lambda_mace_mts/exp009` 错误使用了
  `output_lrc_fix/pre_equilibration.dcd`，而 offset、固定原子选择和 WP-4
  qualification 来自 `output/` 协议；
- 两条轨迹 SHA-256 不同；该混用帧的离线 MACE 结果为 basis
  `-186.622 kJ/mol`、centered basis `156.378 kJ/mol`、λ=0.5 路径能
  `14.074 kJ/mol`；
- 更早的支持域预检应当拒绝该帧：最大 pair distance `7.473 nm`、
  Rg `1.927 nm`，超过冻结的 `2.5 nm / 0.85 nm`；
- 节点运行随后在 \(N=1\) 触发 `max_abs_path_energy=20 kJ/mol`，因此没有
  产生可用于 MTS 判定的 arm report；
- 修正只拆分输入协议，不改变 coefficient、offset、MTS ratio 或任何资格阈值：
  WP-0 继续读取 `output_lrc_fix/`，MTS 改为严格复用 EXP-007 的
  `output/pre_equilibration.dcd` 与 `output/topology.cif`；
- 独立模块已增加积分前支持域 fail-closed，脚本已增加运行目录防覆盖门。

第二次运行使用修正后的 EXP-007 输入，但 \(N=1\) 在
`MTSLangevinIntegrator` 的 CUDA force-group 执行中返回
`CUDA_ERROR_INVALID_HANDLE (400)`。这发生在任何 \(N=2/4\) 比较之前，故没有
资格声称存在或不存在积分分布偏差。

部署判定：

- OpenMM `8.5.1`；
- openmmml `1.6`，其 MACE 后端为 `openmm.PythonForce`；
- openmmtorch `1.5` 已安装，但当前实时 MACE 分解并非 TorchForce；
- torch `2.10.0`，mace-torch `0.3.16`；
- 普通 LangevinMiddleIntegrator 的 EXP-007 通过，但不能外推到 CUDA
  CustomIntegrator/MTS force-group。

总体结论：`FAILED`（直接完整 MACE MTS 的后端资格失败，不是 MACE 势能物理失败）。
按预注册决策 `start_exp010_cheap_cv_due_to_backend`，禁止通过重试、降低系数或放宽
阈值掩盖部署不兼容。

若只有 \(N\ge8\) 或更大间隔才能获得可接受速度，而该间隔不能通过分布/积分门，
直接 MACE 生产路线判为不晋级，启动 EXP-010。

预注册决策：

- \(N=4\) 物理门和性能门都通过：单独测试 \(N=8\)，再决定是否进入 WP-5；
- \(N=4\) 物理门通过但低于 1.0 ns/day：启动 EXP-010；
- \(N=2\) 或 \(N=4\) 出现积分/分布偏差：完整 MACE 只保留为教师，启动 EXP-010。

## 11. EXP-010：MACE 教师到 cheap-CV bias 的蒸馏

当前状态：`FAILED`。正式 GPU 教师标注成功完成并通过数据支持域门，但六个预注册
Fourier 候选均未通过跨 run 能量和广义力验证，因此没有最终 cheap-CV 模型。

目标不是重新训练连续 λ-conditioned MACE，而是：

\[
U_{\rm MACE}^{\rm local}(\mathbf R)
\longrightarrow
V_\phi(s(\mathbf R))
\]

primary \(s_1\) 已冻结为 `[4591,4592,4593,4585]`，secondary \(s_2\) 为
`[4593,4585,4594,4595]`。当前实现使用完整周期 Fourier：

- 1D：order 2、4、6；
- 2D：order 2、3、4；
- 每个候选都按整条 run 留一验证；
- 解析导出为 `CustomTorsionForce` / `CustomCompoundBondForce`。

生产时 cheap-CV bias 每步计算；完整 MACE 只用于训练/验证标签，不在每个 MD step
运行。实验必须记录教师模型哈希、训练轨迹拆分、慢变量定义、支持区间、外推衰减、
能量/平均力验证和 ESS/GPU-hour。

### 11.1 已冻结输入协议

| 项目 | 值 |
|---|---|
| 教师模型 | `mace-off24-medium` |
| 教师能量 | `E(complex)-E(ligand)-E(protein environment)` |
| ligand / environment | 41 / 216 atoms |
| 水策略 | 移除旧环境中全部 39 个固定水原子；不做动态水选择 |
| source trajectories | 三条独立 window-0 scratch trajectories |
| frame 选择 | 每条每 5 帧取样，100 frames/run，300 source frames |
| 支持域 | min pair 0.07 nm；max pair 2.5 nm；Rg 0.85 nm |
| 支持域违规策略 | MACE 前排除并登记；排除率硬门 5% |
| 实际预检 | 290/300 通过；10 帧因 max pair 2.50–2.60 nm 排除 |
| energy offset | 合格教师帧 raw energy 的冻结均值 |
| 外层 coefficient | 0.09；本实验不再调幅度 |
| 数据拆分 | leave-one-run-out；禁止随机 frame split |

### 11.2 已完成 smoke

protein-only CPU 一帧得到 raw interaction energy `-175.978129 kJ/mol`、
primary generalized force `-8.295638 kJ/mol/rad`、最大原子力
`714.669 kJ/mol/nm`。支持域 max pair `2.341981 nm`、Rg `0.727548 nm`，
support/safety violation 均为 0。

截至 2026-07-31，独立模块 SHA-256 为
`956a4401710c812df7125cf32431b362a75b1a5ec173203b805fbd8a0f0f11f3`；
相关独立测试最近一次为 `80 passed, 1 skipped`，skip 原因为本执行节点无 CUDA
device。

### 11.3 实际运行命令

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_DIR="$PWD/output/outer_lambda_exp010/run1" \
bash run_outer_lambda_exp010_teacher_and_fit.sh
```

该命令已完成，输出目录未覆盖既有实验。

### 11.4 教师数据集结果

| 检查 | 结果 | 判定 |
|---|---:|---|
| source / 合格帧 | 300 / 290 | PASS |
| 支持域排除率 | 3.333%（门限 5%） | PASS |
| 合格帧 support / safety violation | 0 / 0 | PASS |
| CUDA 标注时间 | 1564.997 s | 记录项 |
| dataset mean offset | -168.957841 kJ/mol | 已冻结 |
| centered energy 标准差 / 范围 | 21.510895 / `[-75.0555, 53.3613]` kJ/mol | 记录项 |
| primary generalized force 范围 | `[-20.7254, 5.81678]` kJ/mol/rad | 记录项 |
| teacher 最大原子力范围 | `[370.327, 1746.150]` kJ/mol/nm | 记录项 |

教师数据集 `ok=true`、`qualified_for_fit=true`。这只说明标签可用于拟合，不代表任何
cheap-CV 候选通过。

### 11.5 六候选 leave-one-run-out 结果

intercept-only 能量基线 RMSE 为 `21.510895 kJ/mol`。预注册候选结果如下：

| 候选 | 参数数 | 能量 RMSE kJ/mol | 能量 \(R^2\) | 广义力 RMSE kJ/mol/rad | 广义力 \(R^2\) | 判定 |
|---|---:|---:|---:|---:|---:|---|
| 1D order 2 | 5 | 22.1737 | -0.0626 | 19.8500 | -13.5934 | FAIL |
| 1D order 4 | 9 | 22.3263 | -0.0772 | 28.9483 | -30.0371 | FAIL |
| 1D order 6 | 13 | 22.8989 | -0.1332 | 73.2137 | -197.526 | FAIL |
| 2D order 2 | 25 | 362.122 | -282.395 | 335.942 | -4178.87 | FAIL |
| 2D order 3 | 49 | 1042.67 | -2348.50 | 904.546 | -30302.6 | FAIL |
| 2D order 4 | 81 | 5000.44 | -54037.0 | 3860.46 | -551963 | FAIL |

1D order 2 是六者中最小的留一能量 RMSE，但仍劣于 intercept-only，且广义力方向/
幅度不稳定。2D 候选随阶数增加出现严重病态拟合。部分全数据训练或 conditional-bin
指标较好，不能替代整条 run 留一验证，也不能据此选择模型。

| 证据 | 路径 | SHA-256 |
|---|---|---|
| 正式教师数据集 | `output/outer_lambda_exp010/run1/teacher_dataset.json` | `6cdffc3984302e80f0b342b69c5d7d0f666b7a461e221deb4e2a21a91ebfed35` |
| 1D order 2 | `output/outer_lambda_exp010/run1/fit_1d_order2.json` | `eebaf9fbc1bf2e628d5810f119c1d0da64bfc9fcc32df653ee4462fb541daada` |
| 1D order 4 | `output/outer_lambda_exp010/run1/fit_1d_order4.json` | `2fc3b98479752bd41b4c349dd47b083e5ad21fed71d4cdc1b69deb4cdeb473c0` |
| 1D order 6 | `output/outer_lambda_exp010/run1/fit_1d_order6.json` | `72783aebc84d0c0cba3e7720a500e3dae4bf926aff9bd576278ba13e2527ff81` |
| 2D order 2 | `output/outer_lambda_exp010/run1/fit_2d_order2.json` | `af2a122bfb544e86229623b9448646014f66582f00bb3e03de454ffeebca23ba` |
| 2D order 3 | `output/outer_lambda_exp010/run1/fit_2d_order3.json` | `0501749f2fa9d8884e77f6698836d228cffbc37175b7b4219e2525caa23a9cbc` |
| 2D order 4 | `output/outer_lambda_exp010/run1/fit_2d_order4.json` | `3dcaba470e07ec1eb48156d7efc73e6fd086fbbcbe76e45740b858f93849b57c` |

### 11.6 决定

EXP-010 严格记为 `FAILED`。不冻结六个候选中的任何一个，不执行 cheap-CV NVT
qualification，也不启动 EXP-008/WP-5。不得在看到结果后增加 Fourier order、改变
ridge、改用随机 frame split 或用训练集指标追认通过。

本实验不能将失败归因于 torsion bias 或 Fourier 表达本身。事后选择完整性
审计发现，216 个 protein atoms 是由单个参考帧的 0.5 nm 原子半径选择得到，
共涉及 26 个残基，但完整选中的残基数为 0。因此 MACE 看到的是存在断裂共价
边界和不完整局部邻域的人工簇。对这一冻结原子集，
\(E(L+P_{\rm sel})-E(L)-E(P_{\rm sel})\) 代数上有定义；但它不等于完整蛋白环境下的
物理 interaction energy，常数 offset 也无法修复这一边界依赖。

此外，当前的逐帧目标要求低维 \(V(\phi_1,\phi_2)\) 重现高维瞬时 MACE interaction
energy 和力，理论上只有当教师能量几乎只依赖这两个 CV 时才可能逐帧闭合。
实际 primary torsion 对教师 Cartesian force 的平均投影解释比仅 `0.00393`；三条 run
的 centered-energy 均值为 `+4.67/-12.44/+7.72 kJ/mol`。这些观测同时兼容
教师边界伪影和未显式慢变量，不能单独用来排除 torsion bias。

### 11.7 EXP-011 推进边界

EXP-011 将作为新实验，不是 EXP-010 重试。唯一研究问题是：

> 完整困难窗口 MM Hamiltonian 下 primary torsion 的条件平均力或 PMF，能否在
> 整条 run 留一验证中生成稳定、周期且可积分的 cheap bias？

主路线要求：

- 教师使用完整 MM Hamiltonian，不做局部 protein atom selection；
- 目标是沿 CV 的条件平均力/PMF，不是逐帧完整相互作用能；
- 先检查三条 run 的周期 bin 覆盖和重叠，不足时先增加受限/增强采样；
- 候选形式、平滑度、幅度和跨 run 门限必须在看正式结果前冻结；
- 只有先通过条件平均力/PMF 跨 run 验证，才进入 OpenMM NVT 力学资格。

2026-08-02 已冻结机器可读协议
`protocols/EXP-011_preregistration.json`（内部 SHA-256
`d18a38706b5bcd5aa4c7d713e1c34aa9fd19512398b2582c793b2a097096c596`），并实现
`exp011-coverage` 与 `exp011-fit-pmf`。验收说明见
`docs/experiments/EXP-011_PREREGISTRATION.md`。

重新审计后，EXP-011 专用 manifest 已冻结为
`output/outer_lambda_exp011/slow_variable_manifest.json`，状态为
`frozen_for_exp011_complete_mm_pmf`，primary CV 为 `[4591,4592,4593,4585]`，内部
manifest 哈希有效且 `production_approval=false`。

正式覆盖基线 `output/outer_lambda_exp011/coverage_report_v2.json` 为
`qualified_for_pmf=false`：pooled 空 bin 为 11、17–22；run 2/3 的周期有效样本数仅
6.62/3.76；run1/run2 overlap 为 0.328。决定为
`collect_restrained_or_enhanced_sampling`。因此没有生成 PMF model，也没有启动 NVT、
WP-5 或 production。

2026-08-02 已实现 `exp011-umbrella-sample` 与 `exp011-reweight-umbrella`。周期 restraint
使用最短周期角差；采样报告逐帧保存 angle/umbrella energy；重加权按 source window
去相关，用 MBAR 导出显式 `log_target_weight` 并检查 overlap 图。相关测试为
`58 passed`。

完整体系单中心 CUDA smoke 已通过。失败的 Reference 诊断目录
`output/outer_lambda_exp011/cpu_smoke_center_m172p5` 仍只含 DCD header，不得复用；正式结果
位于 `output/outer_lambda_exp011/cuda_smoke_center_m172p5/report.json`。该运行使用
`center=-172.5°`、`k=100 kJ mol^-1 rad^-2`、200 次最小化和 1 个积分/采样步；报告
`ok=true`、`platform=CUDA`、angle `-173.5426°`、umbrella energy
`0.01656 kJ/mol`、temperature `278.03 K`，且 checkpoint 与 DCD 均已写出。
因此前处理与周期 restraint 的 GPU 执行门已通过。下一动作是同一中心的短时稳定性 pilot；
它通过前仍不批量启动 24 centers、PMF、WP-5 或 production。

同一中心短时稳定性 pilot 随后通过，结果位于
`output/outer_lambda_exp011/pilot_run1_center_m172p5/report.json`：10/10 个诊断帧均为
有限值，temperature 为 `298.12–302.11 K`，angle 为 `-173.12°` 至 `-158.97°`，最大
umbrella energy 为 `2.79 kJ/mol`，10 ps 采样段性能为 `52.78 ns/day`。该结果仅批准相邻
中心 overlap pilot，不作为平衡性证明或正式 PMF 数据。下一中心冻结为 `-157.5°`，保持
相同 `k=100 kJ mol^-1 rad^-2` 和短 pilot 长度。

`-157.5°` 相邻中心 pilot 也通过：10/10 帧有限，temperature `298.12–301.15 K`，
angle `-168.47°` 至 `-151.35°`。两窗 MBAR 报告
`output/outer_lambda_exp011/pilot_run1_two_window_mbar.json` 为 `ok=true`，邻窗 overlap
`0.3584`，高于冻结门 `0.03`；两窗各保留 10 个去相关样本。这里的
`qualified_for_pmf_input=true` 仅表示这两个局部窗口的 overlap 图连通，不能解释为全周期
覆盖、平衡性或正式 PMF 验收通过。下一步只增加第三个相邻中心 `-142.5°`。

第三个 `-142.5°` pilot 通过：10/10 帧有限，temperature `297.07–301.49 K`，angle
`-149.75°` 至 `-131.84°`，最大 umbrella energy `1.73 kJ/mol`。三窗 MBAR 报告
`output/outer_lambda_exp011/pilot_run1_three_window_mbar.json` 为 `ok=true`；相邻 overlap
为 `0.3105` 以及 `0.1864/0.3728`，均高于 `0.03`。第三窗估计 statistical
inefficiency `1.943`，只保留 5/10 个去相关样本，进一步说明 10 ps pilot 不能当正式 PMF
数据。局部 15° spacing 与 `k=100` 已获执行资格；下一测试点改为历史空白区
`75°–165°` 中部的 `112.5°` 哨兵中心，用于检查远离初态时的可达性和数值稳定性。

`112.5°` 哨兵窗通过可达性与数值稳定性门：10/10 帧有限，temperature
`298.66–302.28 K`，angle `93.22°–112.21°`，最后一帧 `112.21°`，最大 umbrella
energy `5.66 kJ/mol`。它证明远端空白区可由当前最小化/burn-in 流程到达，但分布明显偏向
中心低侧，不能单独证明高侧邻接连通。下一步运行 `127.5°` 高侧邻窗并只对
`112.5°/127.5°` 做 MBAR overlap 检查。

空白区两窗 MBAR 通过：`112.5°→127.5°` overlap `0.0759`，反向 `0.0949`，均高于
`0.03`，但去相关后分别只有 5/10 与 4/10 帧。因此 pilot 在此停止，不拼接为 PMF。
正式采样计划已冻结为 `protocols/EXP-011_umbrella_sampling_plan.json`，内部 SHA-256
`1cd78aba12f15b52a52f27b5f6c8980544843887ebc2cf84d1b5bc12660c6912`：24 个 15°
bin 中心、`k=100`、每窗 50 ps burn-in + 100 ps sampling、每 1 ps 报告一帧、3 个
replicate，分别从历史 run1/run2/run3 的不同困难窗口轨迹末帧开始。每窗记录初始轨迹路径
与文件哈希。`scripts/run_exp011_umbrella_grid.py` 提供 fail-closed 断点续跑，默认只运行一个
pending window；formal_run1 dry-run 已通过，相关 CLI/controller 回归为 `21 passed`。
当前只批准 formal_run1 单窗正式 smoke，尚未启动 24×3 批量采样。

formal_run1 首个正式窗 `-172.5°` 已通过：100/100 帧有限，temperature
`297.74–302.74 K`，相对中心的周期角偏差 `-8.82°` 至 `+21.92°`，最大 umbrella
energy `7.32 kJ/mol`；初态轨迹 SHA-256 为
`47d2fca0d4189ec7eb5d5e6743406162494cc1aaf25ed4a6aeae6d0c75df3b11`，与 runner
记录一致。断点续跑 dry-run 正确识别 `completed=1`、`pending=23`。因此批准 formal_run1
继续剩余 23 窗；formal_run2/3 尚未批准启动，PMF/WP-5 仍未开始。

formal_run1 随后完成 24/24 窗，共 2400 个有限原始帧。审计发现旧 reweighter 以有向可达
图判 `qualified_for_pmf_input`，比冻结的逐邻窗 mutual-overlap 口径宽松；实现已改为
`min(O_ij,O_ji)` 并要求每个观测邻接接口通过，报告 schema 升至 v2，回归测试
`58 passed`。旧宽松报告保留为 `formal_run1_mbar_legacy_directed.json`。v2 严格报告
`formal_run1_mbar.json` 判定 `qualified_for_pmf_input=false`：唯一失败接口为
`112.5°↔127.5°`，mutual overlap `0.0110 < 0.03`；`127.5°` 的 100 帧因
`g=15.789` 只保留 7 帧。未查看或拟合正式 PMF，formal_run2/3 继续暂停。

针对该预注册门触发的补采已另行冻结为
`protocols/EXP-011_augmentation_001_p127p5.json`，内部 SHA-256
`4107c6719c68657f7a50ae730e6fdc8a212117456d31c31e8cde96dac82632ed`：仅补
`127.5°`，从原正式窗轨迹末帧开始，50 ps burn-in + 500 ps sampling，seed
`2026091200`。合并后必须同时满足该状态至少 25 个去相关样本、失败接口 mutual overlap
至少 `0.03` 及全部 24 个周期邻接接口通过；否则继续停止。

EXP-011-AUG-001 已执行并完成，但未通过冻结验收。补采报告
`formal_run1_supplement/center_p127p5_v1/report.json` 含 500/500 个有限帧，所有参数、seed
与初态轨迹哈希均匹配；angle 覆盖 `77.50°–169.78°`，temperature
`296.85–303.88 K`。补采自身 `g=33.120`，仅保留 15/500 个去相关样本；与原
`127.5°` 窗的 7 个样本合并后为 22，低于冻结门 25。严格 v2 合并报告
`output/outer_lambda_exp011/formal_umbrella_v1/formal_run1_mbar_post_aug001.json`
（文件 SHA-256 `1522c49af5cd5730506196967f9a05d9202ae4fd43dab6c3492a773bfc97771d`）
仍为 `qualified_for_pmf_input=false`：图整体 mutual-connected，但唯一失败接口
`112.5°→127.5°` 的 mutual overlap 仅从 `0.0110` 提升至 `0.02353`，仍低于 `0.03`；
反向值为 `0.10696`。目标加权数据文件 SHA-256 为
`c51a9cf1142b809bb6ecb35250d2b89dd8a6e31f43dc76a1d3d970a84aa0f9d0`。

最终状态记为 `EXP-011-AUG-001 COMPLETED_NOT_ACCEPTED / FORMAL_RUN1_OVERLAP_FAILED`。
按用户要求在此停止：不执行第二次补采，不启动 formal_run2/3，不拟合或查看正式 PMF，
不进入 NVT、WP-5 或 production。现有数据与失败报告全部保留用于后续重新预注册。

若另行恢复 MACE 教师，必须登记为独立实验，并至少要求：完整残基与 backbone
buffer；所有共价截断边界封端或由明确局部 readout 排除；对多个环境半径执行能量/配体力
收敛检查。该 MACE 支线不与 EXP-011 的主问题混合。

---

## 11A. EXP-012：CV-free 通用局部残差路径势

当前状态：`PLANNED / PREREG_DRAFT`。本节登记的是当前预注册方向，不是完成结果。
DEC-019 supersede DEC-018 的 XED 主路线，但不删除 DEC-018 或既有草案哈希。

### 11A.1 唯一研究问题

> 在保留完整 MM/softcore/PME 基线和物理端点、不人工指定慢 CV 的前提下，一个局部
> 坐标残差势能否降低困难 alchemical 窗口的相邻态双向 energy-gap 方差，并在计算成本
> 计入后提高 mutual overlap 和 ESS/GPU-hour？

本实验不回答 MACE latent 是否是真实电子密度，也不把残差解释为物理 Pauli/exchange 势。

### 11A.2 Hamiltonian 与训练目标

\[
H_k^*=H_k^{\rm MM}+A_kB_\theta^{\rm local},\qquad A_0=A_K=0.
\]

\[
\delta_{ak}(\theta)=\Delta u^0_{ak}
+\beta(A_{k+1}-A_k)B_\theta^{\rm local}(\mathbf R_a),
\]

\[
\mathcal L_{\rm gap}
=\sum_k\frac12\left[
\operatorname{Var}_{p_k}(\delta_k)
+\operatorname{Var}_{p_{k+1}}(\delta_k)
\right]
+\lambda_E\langle B_\theta^2\rangle
+\lambda_F\langle\|\nabla B_\theta\|^2\rangle.
\]

第一轮固定全局 \(A_k\)，禁止与网络联合调节；使用完整 MM target-state ledger、MBAR/target
权重和 whole-run folds。ESS、BAR mutual overlap 与 ESS/GPU-hour 只在 held-out 数据和
独立运行上判决。

### 11A.3 表示消融

| Arm | 输入 | 目的 |
|---|---|---|
| A | typed atom-centered RBF/contact | 最低复杂度基线 |
| B | 轻量等变 ligand-environment cross encoder | 判断通用角向/多体表示是否足够 |
| C | frozen-MACE node latent + invariant MLP | 判断 pretrained latent 是否有增量 |
| D | XED-inspired field（可选） | 仅测量额外物理启发信息，不定义方法身份 |

Arm C 新建 `MaceLatentBasisAdapter`，不得读取最终 interaction energy，不执行
`E(complex)-E(ligand)-E(environment)`。必须冻结 MACE layer、channel 类型、ligand-node
pooling、环境图 cutoff/receptive-field buffer、PBC 和支持域。

### 11A.4 在线与蒸馏路线

- **L1：在线 frozen MACE。** encoder、latent readout 和 \(B\) 位于同一 Torch autograd
  图中；ligand/environment 力均来自 \(-\nabla B\)。离线 NumPy descriptor 不得冒充
  production 可导特征。
- **L2：latent teacher + 轻量 student。** MACE 只离线提供表示诊断或蒸馏目标，production
  运行轻量等变 student。若 L1 的 MACE forward/backward 成本抵消 ESS，优先 L2。

两路线必须使用相同 frame folds、gap loss 和判决门，完整报告 OpenMM step 成本。

### 11A.5 实施和资格顺序

1. 补齐三条 scratch run 逐帧五态 target ledger，并核对坐标、状态顺序和 energy hash；
2. 将现有 `exp012_xed` draft 迁移/别名为通用 `local_residual` schema，保留旧哈希；
3. 冻结 A/B/C/D、MACE layer、图边界、readout、\(A_k\)、训练预算和数值门后 seal；
4. 完成 whole-run holdout 与至少 3 个训练 seed；
5. 对 ligand/environment 坐标分别做 autograd/finite-difference force check；
6. TorchScript/OpenMM Reference、XML round-trip、端点和 CPU/CUDA 一致；
7. 单困难窗口短 NVT 与至少 3 个独立 production 重复。

### 11A.6 WP-5 通用性验收

先在 Atenolol 困难窗口比较基础路径、仅 λ 重排、Arm A 和最优局部残差。通过后，将同一
架构、cutoff、loss 权重、训练预算、外层 envelope、安全门和验收门应用到至少两个额外
ABFE benchmark 体系；不人工指定 CV，但允许按统一 pilot-ledger 流程逐体系重训权重。

每个额外体系至少 3 个独立重复，并要求：

- 全局端点严格等于 MM；
- \(|\Delta G_{\rm residual}-\Delta G_{\rm converged\,MM}|
  <\max(0.5\ {\rm kcal/mol},2\sigma_{\rm combined})\)；
- mutual overlap 与 ESS/GPU-hour 通过 sealed 改善门；
- 不出现体系专属 CV、超参数或验收阈值调整。

实验误差 `<1.0 kcal/mol` 只作次级报告。TYK2/CDK2 属于 RBFE 使用场景，必须先完成 RBFE
Hamiltonian、mapping 和 ledger 接口资格，不能直接作为当前 ABFE EXP-012 的首轮硬门。
“同协议逐体系重训”与“同一冻结权重零样本迁移”必须分开报告。

### 11A.7 晋级与停止

任一情况停止：所有表示均无 held-out gap/overlap 增益；Arm C 对 A/B 无增量；在线 MACE
成本抵消 ESS；student 不保留收益；图边界/环境半径不收敛；非守恒力、端点漂移、CUDA
不一致或短 NVT 不稳定；需要未预注册地修改超参数或门限。

### 11A.8 已完成工程预检与当前下一步

2026-08-03 已将关闭的 EXP-010/011 实现迁入
`archive/outer_lambda_exp010_exp011_legacy.py`，主模块只保留 lazy 兼容入口；新增
`exp012_xed/schema.py` 和 `protocols/EXP-012_preregistration.json`。草案已登记三条
500-frame scratch run、真实 SHA-256、五态 λ/f_k 和三个 whole-run folds，相关 CPU 回归
为 58 passed。

随后已经完成三条 `hard_window0_run1/2/3` 的统一 CUDA ledger，每条 500 帧。三条重建
System SHA 均逐位等于 scratch SHA `be34fd38...8144`，arrays SHA 分别为
`fd2985dd...071fa`、`44c813e1...99ad7`、`4c0e35b8...6e689`；Group 1 IBS LSE 最大闭合误差
均为 `2.84e-14 kJ/mol`，force-group 最大闭合误差不超过 `3.77e-4 kJ/mol`。
run1 的 500-frame CPU/CUDA 审计通过：相邻 gap 最大差 `6.43e-6` reduced，未归一
importance log-weight 最大差 `1.29e-5`。机器报告为
`output/outer_lambda_exp012/mm_ledger_audit.json`，状态 `PASSED`。

draft 已升级为 `exp012-local-residual-prereg-v2`，主导入命名空间为 `local_residual`；旧
`exp012_xed` 保留作 DEC-018 兼容/哈希证据。这些结果只证明输入 ledger 与方法协议骨架，
不证明 feature、模型或 Force。当前唯一允许的下一步是冻结 A/B/C/D 表示、MACE layer、
图边界、readout、全局 (A_k)、训练预算和数值门并 seal；在此之前不得训练正式模型、
跑 production NVT、启动 WP-5 或修改 production 模块。
---

## 12. 单次实验模板

复制本节，并将标题改为实际实验编号。

## EXP-XXX：实验标题

### 12.1 基本信息

| 字段 | 值 |
|---|---|
| 状态 | PLANNED |
| 开始时间 | |
| 结束时间 | |
| 执行者 | |
| 对应工作包 | |
| 输出目录 | |
| 主机/GPU | |

### 12.2 唯一研究问题

> 待填写。

本实验不回答：

- 待填写；
- 待填写。

### 12.3 预注册假设

成功假设：

> 待填写。

失败假设：

> 待填写。

### 12.4 预注册验收门

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

### 12.5 输入、模型和版本

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

### 12.6 实际命令

```bash
# 必须填写实际执行命令。
```

### 12.7 Hamiltonian 定义

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

### 12.8 运行完整性

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

### 12.9 端点与力学

| 指标 | λ=0 | λ=1 | 阈值 | 结论 |
|---|---:|---:|---:|---|
| 最大能量差 kJ/mol | | | | |
| RMS 力差 kJ/mol/nm | | | | |
| 最大原子力差 kJ/mol/nm | | | | |
| 有限差分相对误差 | | | | |

### 12.10 神经能量与力

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

### 12.11 采样与自由能

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

### 12.12 构象与稳定性

| 检查 | 基础路径 | 神经路径 | 结论 |
|---|---:|---:|---|
| 异常键长 | | | |
| 原子重叠 | | | |
| 配体逃逸 | | | |
| 口袋异常塌缩 | | | |
| 水/离子异常占位 | | | |
| NaN/积分器失败 | | | |

### 12.13 性能分解

| 项目 | 时间/显存 |
|---|---:|
| Context 创建时间 | |
| 基础 MD step 时间 | |
| 神经基势推理时间 | |
| probe energy 时间 | |
| 总 step 时间 | |
| 主 Context 显存 | |
| probe Context 显存 | |

### 12.14 观测事实

- 待填写；
- 待填写；
- 待填写。

这里只写直接观测，不写原因推测。

### 12.15 解释

- 待填写；
- 待填写。

必须指出哪些解释仍是推断。

### 12.16 验收结论

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

### 12.17 决策

- [ ] 进入下一工作包；
- [ ] 保持复杂度，补充独立重复；
- [ ] 调整外层系数后重试；
- [ ] 重新训练同一目标基势；
- [ ] 回滚到基础路径；
- [ ] 停止该方向；
- [ ] 其它：待填写。

决策理由：

> 待填写。

### 12.18 后续行动

| 行动 | 负责人 | 截止条件 | 状态 |
|---|---|---|---|
| | | | |

---

## 13. 跨实验汇总

至少三个可比独立重复完成后填写：

| 方案 | 重复数 | ΔG 均值 | 重复间 SD | ESS/GPU-hour | 慢变量转换 | 异常率 |
|---|---:|---:|---:|---:|---:|---:|
| 基础路径 | | | | | | |
| 仅 λ 重排 | | | | | | |
| 单神经基势 | | | | | | |
| 多神经基势 | | | | | | |

## 14. Production 准入

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
NOT_READY
```

决定日期：

```text
2026-08-03
```

依据：

> 独立端点和账本契约已通过，但 EXP-010 fragment teacher、EXP-011 torsion-PMF 均已失败；
> EXP-012 已升级为 CV-free 通用局部残差路线；五态 ledger、输入哈希、whole-run folds 和
> backend audit 已完成，但 preregistration 仍为 draft，尚缺 sealed A/B/C/D 协议、表示消融、
> 守恒力、NVT 和跨体系性能结果。因此没有候选可进入
> WP-5 或 production。
