# OpenMM 8.6+ 官方 REMD 后端接入计划

日期：2026-08-31  
状态：设计计划；本次只新增本文档，不修改运行代码、不升级环境、不启动模拟。

## 1. 结论与目标

可以按版本选择后端，门槛应为 **OpenMM >= 8.6.0**，包含 8.6.0 本身，而不是严格大于 8.6.0。8.6.0 发布说明确认新增官方 `ReplicaExchangeSampler`，支持温度、压力或 Hamiltonian 不同的多状态采样。[官方发布说明](https://github.com/openmm/openmm/releases/tag/8.6.0)

本项目的“传统 REMD”主要是同温度的 λ/Hamiltonian replica exchange，不是温度阶梯 REMD。目标是将已支持的传统交换路径接到官方采样器，同时保留旧版本兼容路径。**版本合格只是必要条件，还需检查实际 API 和当前协议是否已通过适配验证。**

本次不替换 IBS/ACE/EXP-030 算法，不改变 Hamiltonian、λ 调度、自由能符号约定、物理修正或结果验收标准，也不引入 expanded ensemble。

## 2. 当前项目的接入点

以下位置以编写计划时的源码为准，实施时按符号定位：

| 文件／符号 | 当前职责 | 计划处理 |
| --- | --- | --- |
| `ibs_engine.py::REMDManager`（约 18172 行） | 构建副本、预热、相邻交换、按状态写 DCD、交换诊断 | 保留 legacy 实现，抽取可共用的 System 构建和预热逻辑 |
| `abfe_pipeline.py::TraditionalABFEPipeline.run_leg` | 传统采样及离线 MBAR | 用统一工厂选择采样后端，保留分析链 |
| `abfe_pipeline.py` 中直接构造 `REMDManager` 的其他位置 | 包括 dual_lambda 中的 REMD charging/decharging 路径 | 逐调用点接线；不能只处理 `--mode traditional` |
| `abfe_pipeline.py::_remd_sampling_fingerprint` 及传统路径内的采样指纹 | 约束 DCD、能量缓存和 resume 的身份 | 加入实际后端、版本、交换协议和输出布局身份 |
| `runabfe.py::RunConfig`、参数解析及透传 | 配置到两条腿／各阶段的传递 | 增加统一 `remd_backend` 配置 |
| `ibs_engine.py::BoreschAttachmentREMDManager`、`ShadowBridgeREMDManager` | 专用 Hamiltonian 和额外诊断 | 第一阶段明确保留 legacy，另行验证后再放行 |
| `TraditionalMBARAnalyzer.compute_u_kn/solve` | 重算交叉能量、MBAR、LRC 和收敛诊断 | 第一阶段保留，不直接换成官方能量输出 |

现有 `REMDManager` 每轮按 `(0,1), (1,2), ...` 顺序尝试所有相邻边，接受后立即更新状态映射。每个副本目前持有一个常驻 Context。文件名虽为 `{stage}_rep{i}.dcd`，实际内容按**热力学状态 i** 分流，不能在新后端中改成物理 replica 轨迹。

## 3. 后端选择规则

新增配置（计划中的接口，尚未实现）：

```text
--remd-backend auto|legacy|openmm
配置键：remd_backend
目标默认值：auto
```

| 请求 | 运行条件 | 行为 |
| --- | --- | --- |
| `legacy` | 任何原有支持的环境 | 始终使用旧引擎，方便复现与回退 |
| `auto` | 稳定版 >= 8.6.0，所需 API 可用，当前协议已通过适配验证 | 使用官方后端 |
| `auto` | < 8.6.0，或新接口缺失，或当前协议尚未适配 | 使用 legacy，并记录具体原因 |
| `openmm` | 版本、API、协议检查全部通过 | 使用官方后端 |
| `openmm` | 任一检查不通过 | 在创建生产 Context／写轨迹前明确报错，不能默默改用 legacy |

实施要求：

1. 使用 `packaging.version.Version` 做语义版本比较，不用字符串比较或浮点数。环境文件显式声明 `packaging`。
2. 集中解析实际导入的 OpenMM 版本，并记录原始版本字符串。不能仅凭另一环境的包管理器信息选择接口。
3. 新 API 延迟导入，旧环境必须仍能导入本项目。检查适配器需要的类、方法和参数，不能只检查一个类名。
4. 预发布版／开发版／不可解析版本第一阶段不自动启用官方后端；`auto` 记录原因后走 legacy，显式 `openmm` 拒绝并说明支持范围。
5. 高版本若发生 API 不兼容，不得仅因为版本号大就强行调用；增加兼容测试后才能声明支持。
6. 后端只在阶段开始前解析一次；采样开始后遇到 NaN、配置错误、能量错误或 I/O 错误必须停止并保留诊断，禁止捕获所有异常后从头换引擎继续。
7. GPU→CPU 是平台策略，与 official→legacy 是后端策略，分别记录，不能混为同一种回退。

## 4. 已核实的官方接口与适配设计

正确入口是 `openmm.app.ReplicaExchangeSampler` 和 `ReplicaExchangeReporter`。官方状态采用 `list[dict]`，不是 `openmmtools.states.ThermodynamicState`。最小调用关系如下；它只说明 API，不能直接用于本项目生产：

```python
from openmm.app import ReplicaExchangeSampler

# simulation 已由项目完成 System 构建与初始化；
# state_parameters 中每个 dict 的键集合相同，且参数确实存在。
sampler = ReplicaExchangeSampler(
    state_parameters, simulation, stepsPerIteration=exchange_interval
)
sampler.reporters.append(project_reporter)
sampler.simulate(iterations)
```

8.6.0 final tag 的重要差异：

- 共享一个 Simulation/Context，逐副本推进；每轮评估各副本在所有状态下的能量。不能套用当前“每副本一个常驻 Context”的预算逻辑，也不承诺自动多 GPU 并行。
- 默认每轮随机选择副本对，尝试次数为 `K**2`；**没有直接配置相邻交换的开关**。
- 顺序是推进／评估能量 → reporter → exchange；`replicaStateIndex[i]` 表示副本 i 当时所属状态。
- 交换随机数来自 Python `random`，不是现有 NumPy RNG；普通 `Simulation.reporters` 不能代替 `sampler.reporters`。

以上按固定版本源码核实，实施时不以滚动 development 文档替代版本证据。[8.6.0 sampler 源码](https://github.com/openmm/openmm/blob/8.6.0/wrappers/python/openmm/app/replicaexchangesampler.py)

### 4.0 模块边界

按用户确认的大模块化方向，统一在 `free_energy_engine.py` 内放置能力检查、后端选择和官方／legacy 通用采样适配，不另拆 `remd_backends.py`。化学 System 构建仍归现有 ABFE 化学层或未来 `rbfe_core.py`；由 pipeline 将构建结果传入 shared engine。`ibs_engine.py` 的旧类先保留，工厂迁移分阶段进行，shared engine 不反向 import `ibs_engine.py`，避免循环依赖。与 [RBFE 大模块接口设计](PLAN_rbfe_interface_and_implementation.md) 共用同一采样契约；尚未适配的旧类不能直接作为 RBFE fallback。

对上层保持 `run(n_steps, exchange_interval, save_interval, stage_name) -> traj_files` 的语义；审计调用方读取的其他属性和重复调用行为，逐项提供明确契约。

### 4.1 System 与热力学状态

- 复用当前 PME 去电荷、混合 alchemical 和 Beutler softcore 构建逻辑；官方采样器不会自动替项目生成正确的 ABFE Hamiltonian。
- 将当前状态列表映射为同温度、不同全局参数的热力学状态。按实际 System 中存在的参数设置 `lambda_coul`、`lambda_vdw`，不要给某一分支强加不存在的参数。
- 保留 PME、配体内部相互作用、约束、虚拟位点、力组、Boresch 约束和冻结 co-ion 身份；新旧后端使用同一份确定后的输入。
- 在冻结构型上逐 λ 比较势能、力和交叉能量，再开放采样。不得为绕过接口限制而偷偷重定义 Hamiltonian。
- 保留现有 preflight、EM／温和预热、随机源身份和平台属性；为各副本创建独立的初始位置／速度 State，不能把官方默认克隆初始构型直接当作项目预热已完成。
- 将 `max_resident_contexts` 明确解释为实际 Context 预算并记录后端差异；官方共享 Context 路径仍需检查峰值显存、失败清理和平台回退，不能直接沿用 legacy 的副本数比较。

### 4.2 交换和运行步数

- 首版官方后端采用原生随机副本对交换，记录 `exchange_scheme=openmm_random_pairs` 和尝试次数；legacy 记录 `legacy_sequential_neighbors`。两者目标平衡分布可以一致，但交换过程不同，必须分别资格验证。
- 需要严格复现原来的相邻扫描协议时选择 legacy。若将来覆写官方 `exchangeReplicas()` 做相邻交换，应作为单独适配协议测试，不伪称官方内置 neighbor 模式。
- 单位明确区分 MD steps、交换间隔和 sampler iterations；准确处理 `n_steps % exchange_interval` 尾段，不增跑、漏跑或额外交换。
- 核对保存发生在交换前还是交换后，并始终用对应时刻的状态映射写轨迹。
- 第一阶段仅放行已验证的步数／保存间隔组合；无法保持当前输出语义时，在开跑前拒绝显式官方模式或让 `auto` 选择 legacy，不静默四舍五入。
- 官方 RNG 与现有 seed ledger 建立可审计映射：分别记录共享积分器、各副本初始速度及 Python `random` 的交换 seed。同进程其他组件的全局 RNG 不能被静默污染；使用受控的 RNG 状态隔离并验证，不假设官方构造器支持 `random_seed`。
- 相同 seed 不代表两种后端轨迹逐帧相同，原来的“每副本独立积分器 RNG”也不能原样声称已保留。

### 4.3 输出、分析和续跑

- 优先用自定义 sampler reporter 保留现有文件名／保存契约。若启用官方 `ReplicaExchangeReporter`，使用独立空子目录；其原生命名为 `state_i.dcd`／`replica_i.dcd`，通过明确映射桥接，不让它直接接管已有非空阶段目录。
- 继续输出按状态分流的 `{stage}_rep{i}.dcd`、`{stage}_exchange_diagnostics.json` 和 `{stage}_sampling.meta.json`；保持坐标、盒矢量、时间及帧数可被现有分析读取。
- 第一阶段继续从这些轨迹离线生成 `u_kn`、`n_k`，沿用 MBAR 和 LRC 路径。官方报告器输出可作交叉核验，不能未经证明就替代现有分析输入。
- 明确区分交换轮数、尝试边数与接受边数；旧字段语义不能因接口切换悄悄变化。
- 记录 requested/resolved backend、原始与解析后 OpenMM 版本、适配器协议版本、选择原因、交换算法、精度／设备、seed 身份及实际 Context 数。
- 在采样和派生能量缓存身份中传递以上关键信息。旧文件缺少后端字段时，只能按经验证的旧 schema 识别为 legacy，不能当作官方运行产物。
- 官方 reporter 的 `checkpoints=True` 保存 `checkpoint_i.xml`（serialized State），`resume=True` 依赖这些文件及 `log.csv`。它可恢复构型和状态映射，但不能视为包含完整 RNG 的二进制 Context checkpoint；能量 CSV 也是 iteration×replica×state 布局，需要重排后才能与项目数据比较。[8.6.0 reporter 源码](https://github.com/openmm/openmm/blob/8.6.0/wrappers/python/openmm/app/replicaexchangereporter.py)
- 第一阶段保证“已完成且身份匹配的采样／分析可复用”；**阶段中断后的精确动态续跑另行验收**。不能把坐标快照当成包含 RNG 和积分器内部状态的完整 checkpoint。
- 禁止跨后端直接加载动态 checkpoint 或向旧后端的未完成 DCD 追加帧。升级后的 `auto` 若与已有运行身份冲突，应停止复用并提示显式 legacy／另开输出目录，不覆盖原文件。

### 4.4 不随版本升级放宽的限制

- 保留 `runabfe.py::_assert_traditional_protocol_supported` 对膜体系、非默认色散协议和显式力场族的限制。
- 保留 `TraditionalABFEPipeline` 对带电配体的当前限制；dual_lambda 已支持的 PME 去电荷路径继续使用其自身 co-ion 协议，不能混淆两条入口。
- 传统 Beutler 的离线 LRC 保留固定盒 NVT 适用条件及明显 NPT 体积波动时的拒绝逻辑。官方压力交换支持不等于项目这一物理缺口已修复。
- Boresch attachment 的 round-trip 等专用诊断和 Shadow Bridge 的 Hamiltonian 需各自通过资格验证，不能仅修改基类就自动切换。

## 5. 分阶段实施与验收

| 阶段 | 交付 | 放行条件 |
| --- | --- | --- |
| P0：选择器 | 版本／能力检查、配置透传、选择日志；默认仍保留旧运行行为 | 旧版无新 API 时可正常导入；显式官方失败清楚；两条腿各入口参数完整 |
| P1：最小官方适配 | 共用 System 构建、状态映射、短程运行和 DCD 桥接 | 小体系能量／力一致；状态分流、步数和帧数验证通过 |
| P2：生产契约 | preflight、预热、seed、平台预算、指纹、已完成缓存复用 | 不丢现有安全门；换后端／换 seed／换协议不误复用缓存 |
| P3：科学与性能对照 | CPU 快速验证及目标 GPU、真实体系短程对照报告 | 分布／自由能在预先约定的不确定度范围内相容；时间与显存实测可接受 |
| P4：默认启用 auto | 仅对通过资格验证的路径启用 >=8.6 官方选择 | 文档、回退示例、CI 版本矩阵和证据齐备 |
| 后续独立任务 | Boresch attachment、Shadow Bridge、多 GPU、动态精确续跑 | 各自验收，不阻塞首批已支持路径 |

第一版优先保证后端可切换和科学契约一致。共享 Context 可能降低设备显存占用，但状态搬运、逐副本推进及每轮全状态交叉能量评估可能增加成本，不预先承诺必然加速。保留 `legacy` 开关，不在接入时删除旧实现。

## 6. 验证清单

- [ ] 版本边界：8.5.x、8.6.0、8.6.1、8.10.0，以及 rc/dev/未知版本；用模拟版本验证分支，用真实旧版和 8.6.0 验证导入与运行。
- [ ] 能力异常：版本合格但缺类／缺参数；区分“不可用”和真正的运行错误。
- [ ] 2–4 状态解析小体系：逐状态势能／力、交叉能量、采样分布；异常参数与非有限能量必须触发预期处理。
- [ ] 轨迹映射：构造可追踪的状态切换，证明 `rep{i}.dcd` 始终对应状态 i；校验盒矢量、帧数、时间及 `n_k`。
- [ ] 运行边界：零步数、尾段、交换／保存周期不整除及重复调用；继承项目“至少两个状态”等输入校验。
- [ ] 续跑身份：已完成 legacy 产物、已完成官方产物、损坏／缺失元数据、换后端和未完成轨迹均有清晰行为。
- [ ] 保留相关回归：`test_traditional_pipeline_interfaces.py`、`test_traditional_mode_preflight_and_equil.py`、`test_remd_gpu_context_budget.py`、`test_decharging_seed_contract.py`、`test_resume_reuse_contracts.py` 等。
- [ ] GPU：检查设备／精度、Context 数、显存峰值、初始化失败清理、平台回退日志；不把未测试的多 GPU 当成已支持。
- [ ] 新旧后端独立重复：固定物理协议和相同采样预算，比较 ΔG／误差、能量分布、overlap、ESS、交换／混合指标；事先定义容差与统计判据，不要求逐帧一致。
- [ ] 更新 `environment.yml`、`environment-ci.yml` 与 CI 版本矩阵，不把全项目最低 OpenMM 版本强行抬到 8.6。
- [ ] 更新使用文档，说明 auto 适用范围、legacy 复现方式、选择日志和 checkpoint 边界。

## 7. 完成定义

用户无需手工改代码：受支持的 OpenMM >=8.6.0 环境和已验证协议自动使用官方 REMD；旧环境或尚未适配协议明确使用 legacy；显式选择官方时不悄悄降级。两条路径保持项目物理与分析契约，所有后端决定可审计，产物身份不混用。
