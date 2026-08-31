# RBFE 接口与代码实现计划

日期：2026-08-31  
状态：仅设计；按用户要求采用大模块化，给出职责、依赖方向、接口契约和未来验收标准。本次不新增 Python 模块、不修改生产代码，不代表当前已有可运行的 RBFE 引擎。

## 1. 是否需要

如果项目目标扩展到一系列相近配体的亲和力比较，应准备 RBFE；如果只继续做单个配体的绝对结合自由能，RBFE 不是 ABFE 的必需依赖。两条方法保留独立入口，复用经过抽象和验证的底层组件。

本计划采用常规的“两条腿、A→B 配体变换”的 RBFE 路线。它需要配体映射和混合拓扑；官方 REMD 只提供多状态采样，不能从两个配体自动生成正确的变换 Hamiltonian。OpenFE 的相对自由能流程也将映射、体系构建和两条腿明确分开。[OpenFE 官方协议说明](https://docs.openfree.energy/en/stable/guide/protocols/relativehybridtopology.html)

前置兼容工作单独维护：

- OpenMM >=8.6.0 的官方 REMD 选择见 [REMD 后端计划](PLAN_openmm_8_6_remd_backend.md)。
- openmm-ml >=1.7 可退出已修复的 MACE 本地模型 device 补丁；经典力场 RBFE 不依赖 openmm-ml，不应为了运行 RBFE 强制加载 ML。
- 不把 IBS/ACE/EXP-030 直接移植为 RBFE 生产默认值；先建立可验证的传统基线。

## 2. 首版范围

优先完成一个明确的 A→B 边，再扩展为配体网络。

首批资格验证采用：同一受体、相近骨架、固定质子化／互变异构状态、相容结合姿势、两个中性配体、可溶性体系、经典力场，以及可核对的端点结构和参数。lambda 与温度／压力／约束协议显式配置，不沿用 ABFE 的隐藏默认值。

首版在生产 Context 创建前拒绝：

- 净电荷变化，以及尚未验证的同电荷带电配体路线；
- 环断裂／闭合、环尺寸变化、手性反转、映射元素改变、共价配体；
- 膜体系、质子化或互变异构状态改变、大幅骨架或结合模式改变；
- 元素、虚拟位点、约束、力场或自定义 Force 不在已验证 builder 支持范围内的输入。

这些是本项目首版的范围限制，不代表 RBFE 方法或其他软件普遍不支持这些变换。

## 3. 热力学与符号契约

统一定义 A→B、lambda=0 对应 A、lambda=1 对应 B：

```text
ΔG_complex(A→B) = G_complex(B) - G_complex(A)
ΔG_solvent(A→B) = G_solvent(B) - G_solvent(A)

ΔΔG_bind(B-A) = ΔG_complex(A→B) - ΔG_solvent(A→B)
```

因此负的 ΔΔG 表示 B 的结合自由能更低。输出必须同时记录 A、B、方向、单位和每条腿的值。

当前 ABFE 使用 coupled→decoupled 的去耦约定，最终采用 solvent−complex。**RBFE 不可直接调用该符号约定下的 ABFE 汇总函数，也不能只把名称改成 relative。**

RBFE 不机械添加 ABFE 的 Boresch 解析释放项。若采用限制配体构象／位置的 restraints，必须明确其对目标自由能的影响，并证明两腿抵消或计算所需修正。用于标定绝对值的 ABFE／实验锚点是网络分析层的可选输入。

## 4. 代码组织与接口草案

采用四个大模块，和现有 `abfe_core.py / abfe_pipeline.py / runabfe.py` 的分层方式衔接。**不拆成 `schema.py / mapping.py / topology.py / sampling.py / network.py` 等大量小文件**；这些是大模块内的职责区块。

| 大模块 | 内部职责 | 明确不负责 |
| --- | --- | --- |
| `rbfe_core.py` | 输入／结果类型、配体身份、映射、hybrid Hamiltonian、端点验证、RBFE 能量评估与 ΔΔG 汇总 | 不调度网络、不选择硬件、不处理 CLI |
| `rbfe_pipeline.py` | prepare/run/analyze 编排、两腿、独立重复、边集合／网络、运行状态和续跑校验 | 不自行构建相互作用公式、不重写交换算法 |
| `free_energy_engine.py` | ABFE/RBFE 共用采样契约、官方／legacy 后端选择、通用独立窗口与 REMD 推进、reporter、平台资源与运行诊断 | 不识别 ligand A/B、不生成 ABFE 去耦或 RBFE hybrid System、不计算两腿结合自由能 |
| `runrbfe.py` | 参数解析、配置加载、调用 pipeline、用户可见结果与退出码 | 不持有科学算法或第二份计算流程 |

`rbfe_core.py` 内可按“数据契约 → 映射 → hybrid builder → 验证 → 分析”分节；`rbfe_pipeline.py` 内按“单腿 → 双腿边 → 网络”分节。大模块化不意味着复制已有代码或允许循环依赖。

依赖方向固定为：

```text
runrbfe.py
    └── rbfe_pipeline.py
            ├── rbfe_core.py
            └── free_energy_engine.py

runabfe.py
    └── abfe_pipeline.py
            ├── abfe_core.py / ibs_engine.py 的现有化学与 IBS 逻辑
            └── free_energy_engine.py 〔未来逐步接入〕
```

- shared engine 不反向 import pipeline、`rbfe_core` 或 `ibs_engine`。
- 两条 pipeline 分别把化学层输出转成 shared engine 的通用采样输入，并把采样输出交回自己的分析层。
- 现有 ABFE 先保持原有入口；只有已通过回归的公共部分才迁入 shared engine，不以 RBFE 设计为由整批重构 ABFE。
- 先前 REMD 计划中的 `remd_backends.py` 调整为 `free_energy_engine.py` 内部的后端适配区块，避免平行建立两套版本判断与交换引擎。
- CLI/API 均调用同一 pipeline；OpenFE/Perses 等候选 builder 的适配放在 `rbfe_core.py` 内，不额外扩散入口。

拟议 Python 接口如下，仅定义职责，不是可直接运行的现有 API：

```python
def validate_edge(spec: EdgeSpec) -> ValidationReport: ...
def prepare_edge(spec: EdgeSpec) -> PreparedEdge: ...
def build_hybrid_leg(edge: PreparedEdge, phase: str) -> PreparedLeg: ...
def run_leg(leg: PreparedLeg, protocol: ProtocolSpec) -> SamplingArtifacts: ...
def analyze_leg(leg: PreparedLeg, artifacts: SamplingArtifacts) -> LegResult: ...
def combine_rbfe(complex_result: LegResult, solvent_result: LegResult) -> EdgeResult: ...
```

规划中的 CLI：

```text
python runrbfe.py validate --config rbfe_edge.json
python runrbfe.py prepare  --config rbfe_edge.json
python runrbfe.py run      --prepared prepared_edge.json --backend auto
python runrbfe.py analyze  --run-dir output_rbfe/A_to_B/repeat_01
```

### 4.1 共用引擎的边界契约

以下为字段设计，不在本次实现类：

| 契约 | 必需信息 |
| --- | --- |
| `SamplingRequest` | 已构建的 System/Topology、有序 state 参数表、各副本初始构型／速度／盒矢量、积分器与系综设置、采样／交换／保存步数、seed 计划、平台预算、输出目标、调用方协议指纹 |
| `BackendResolution` | requested/resolved backend、实际 OpenMM 版本、能力检查结果、交换算法、选择理由 |
| `SamplingArtifacts` | 按状态的样本位置／索引、MD 步数与保存时点、状态映射、诊断、checkpoint 类型、实际后端、指纹 |
| `LegResult` | phase、A→B 身份、ΔG、单位、误差方法、有效样本与质量门、所用物理修正、产物指纹 |
| `EdgeResult` | 经过身份校验的两腿结果、ΔΔG、传播后的不确定度、方向、qualification 与理由 |

引擎保证的是“按指定 Hamiltonian 和状态表采样”，化学正确性由 core 验证，完整流程身份由 pipeline 验证。未提供某项能力时必须显式报告不支持，不由引擎猜测物理默认值。

不提供“尚未实现但返回成功”的占位 sampler；接口准备完成与可运行科学计算是两个验收阶段。

## 5. 输入、映射与混合拓扑

### 5.1 输入身份

EdgeSpec 至少保存：edge_id、ligand_A/B、受体／环境身份、各端点输入路径及 hash、带显式氢的化学结构、形式电荷、质子化／立体化学身份、原子映射、温度与系综、力场／水／离子模型、部分电荷来源、采样配置、seed 和输出目录。

已参数化的 GROMACS／OpenMM 输入与自动参数化的 SDF 路线分开。**不能为了调用第三方 builder 悄悄把当前力场重参数化。** 首版锁定一条可完整验证的参数化路线，其他输入先拒绝，不自动猜测转换。

### 5.2 原子映射

- 分子内索引、complex 全局索引、solvent 全局索引和 hybrid 索引分别记录。
- 验证一对一、索引范围、化学一致性和映射核心的连通性，识别对称等价映射与姿势歧义。
- 两腿使用同一份冻结的分子级 A→B 映射，再各自投影为全局原子索引。
- 映射评分只用于候选排序，不能代替化学与几何验收；配体构象或输入顺序变化后不能误用旧索引。
- 现有教师／student 的原子映射不能仅因为名字相同就视作 alchemical atom mapper。

### 5.3 Hybrid builder

- 明确 common core、A-only 和 B-only 原子，以及各自的 charge、LJ、bonded、exception／1-4、约束和 dummy 处理。
- 路径至少描述 charge、sterics 和 bonded 项；当前单配体的 `lambda_coul/lambda_vdw` 去耦规则不自动等价于 A→B 变换。
- 核对每个 λ 的有效总电荷。即使 A、B 净电荷相同，也不能因不同电荷分段切换而忽略中间态的电荷变化。
- 验证 λ 端点的物理相互作用恢复；hybrid 端点含 dummy 自由度时，明确可分离因子／修正及其在热力学循环中的抵消条件，不能仅凭端点“看起来像 A/B”判定正确。
- 保持跨 λ 相同的粒子、质量和约束结构；无法证明约束／dummy 贡献正确的变换不放行。

候选工程路线为封装经验证的 hybrid builder，或实现本项目所需的受限 builder。先做小体系对照再决定生产提供者；OpenFE/Perses 可作为参考与候选适配对象，但不代表其默认力场、采样器和全部变换已获本项目认可。官方文档也提醒某些 dummy bonded 处理存在不抵消导致系统误差的风险，应将此项纳入选择标准。[OpenFE hybrid topology 限制](https://docs.openfree.energy/en/stable/guide/protocols/relativehybridtopology.html#the-hybrid-topology-approach)

## 6. 现有代码的复用边界

可评估复用平台选择、seed 派生、通用日志、轨迹写入、数据指纹及独立的 MBAR 数值求解部分；保留这些组件既有的回归测试。

不能直接复用：

- `REMDManager` 的 ABFE System 构建分支。需先抽象为“已构建的 System + states + 初始构型 → 采样产物”的接口。
- `TraditionalMBARAnalyzer.compute_u_kn` 中按单配体去耦和 LRC 假设重建评估系统的逻辑。RBFE 必须对实际 hybrid Hamiltonian 评估。
- `TraditionalABFEPipeline` 的阶段编排及 ABFE 专用 Boresch／APBS／co-ion 自动修正。
- ABFE 的结果字段、采样 fingerprint 及 λ 调度协议版本。

`backend=auto` 复用 OpenMM >=8.6.0 的版本／能力判断原则，但只有 RBFE hybrid states 和 reporter 已完成适配时才能启用。旧版也必须有经过验证的通用 fallback；不能直接回退到一个会重新构建 ABFE 去耦系统的旧类。

第三方完整 RBFE 流程有自己的采样后端，不能假设升级 OpenMM 后它会自动使用 8.6 官方 sampler。

## 7. 输出、分析与恢复

建议每条边、每次重复独立保存：

```text
output_rbfe/A_to_B/repeat_01/
  edge_manifest.json
  atom_mapping.json
  endpoint_validation.json
  complex/  # hybrid System、状态表、样本、交叉能量、检查点、诊断
  solvent/
  rbfe_result.json
```

- 元数据包含 A/B 化学及输入身份、映射和参数 hash、hybrid builder 版本、λ 路径、平台、实际后端、有效 seeds、方向、单位和物理修正。
- 能量矩阵记录维度、状态顺序、样本状态索引、是否约化和是否包含 NPT 所需项；不按文件名猜测。
- 汇总保留两腿 ΔG 与不确定度、ΔΔG、误差估计方法、重复间方差及 qualification 状态；独立两腿时方差相加，有相关性时显式计入协方差。
- 分析结合 overlap、有效样本量、时间稳定性、交换混合和结合姿势诊断；不能只看交换接受率。
- 动态 checkpoint 与已完成采样缓存复用分开；禁止跨 A/B 方向、映射、力场、后端或 λ 路径直接追加。
- 网络阶段检查连通性和闭合环；闭合不通过需定位问题，闭合通过也不是所有系统误差均消失的证明。
- ABFE 锚点仅在受体状态、化学身份、力场、温度和自由能定义一致时用于转换绝对值，并传播锚点误差。

映射、打分、网络规划和两腿 Transformation 分层可参考 OpenFE 的官方教程；实现时固定依赖版本及输入来源。[OpenFE RBFE 网络教程](https://docs.openfree.energy/en/stable/tutorials/rbfe_python_tutorial.html)

## 8. 实施顺序与验收

| 阶段 | 代码交付 | 验收标准 |
| --- | --- | --- |
| R0 | schema、验证接口、方向明确的结果汇总、CLI validate | 错误输入被拒绝；合成两腿数据的符号／单位／误差传播正确；不启动 GPU |
| R1 | 映射和受限 hybrid builder | 映射可审计；端点相互作用及 dummy 处理有验证；有限差分力与解析力一致 |
| R2 | 通用采样接口及 RBFE 单腿 | 独立窗口先跑通，再验证 REMD；真实 hybrid 能量和样本归属一致 |
| R3 | complex+solvent 完整单边 | A→A 为零、A→B 与 B→A 相容；独立重复及参考实现对照通过 |
| R4 | 网络、恢复、生产 qualification | 三角闭合、断点恢复不混样本、既有 ABFE 回归不变 |
| 后续 | 带电／膜／大变换／IBS增强 | 每条路线独立协议和科学验收，不因接口存在自动开放 |

首个实际体系需要提供第二个配体 B 的结构／参数及可核对的结合姿势，或者明确使用的公开基准配体对。不能根据 Atenolol 文件名猜测 B，也不能把接口测试当作真实体系验证。

本次设计完成定义：四个大模块的职责、单向依赖、共享边界、数据契约和验证路线清楚，可直接作为后续实现任务书；不交付运行代码。未来实现从 R0 开始，只有 R3/R4 证据齐全才声明“RBFE 可用于生产”。