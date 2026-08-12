# ABFE-IBS 软件开发进度与技术方法底稿

> 文档性质：进度报告与论文写作母稿，不是 README，也不是最终发表结论。  
> 证据截止日期：2026-08-12（Asia/Tokyo）。  
> 参考体系：Atenolol-rank11。  
> 当前原则：保留成功、失败、未验证和被主动终止的路线；任何结果只有在其预注册门、运行环境和证据范围内成立。

## 0. 如何使用这份底稿

这份文档用于回答四个问题：

1. 软件解决什么科学问题，采用了什么物理与统计方法；
2. 软件由哪些模块组成，一次计算如何从输入走到最终自由能；
3. 当前哪些能力已经实现并获得证据，哪些仍缺真实运行验证；
4. 哪些开发路线失败、为什么失败、失败结论能够推广到什么范围。

后续写进度报告时可以直接保留第 1、3、5、8、9、10 节；写论文时可将第 2、4、5 节改写为 Methods，将第 6、7、8 节改写为 Results/Negative results，将第 9、10 节改写为 Limitations/Future work。

本文统一使用以下状态词，避免把不同性质的“没做成”混为一谈：

| 状态 | 含义 |
|---|---|
| `IMPLEMENTED` | 代码或流程已经存在，但不自动等于科学验证通过 |
| `VALIDATED` | 在明确环境、输入和验收门下获得了可复核证据 |
| `FAILED` | 已执行且未通过预先冻结的门；失败范围必须写清 |
| `INCONCLUSIVE` | 有数据，但不足以支持肯定或否定结论 |
| `NOT_PURSUED` | 主动不继续或从设计范围排除，不等于实验失败 |
| `NOT_STARTED` | 设计或计划存在，但尚未执行 |
| `INVALIDATED/HISTORICAL` | 曾经产生结果，但后来发现协议、实现或输入问题，不能作为当前科学结论 |

## 1. 项目目标与当前总体判断

本项目开发了一个基于 OpenMM 的绝对结合自由能（absolute binding free energy, ABFE）工作流。软件读取 GROMACS `.gro/.top` 体系，分别计算配体在蛋白复合物和水溶液中的解耦自由能，并利用 Integrated Boltzmann Sampling（IBS）、MBAR/TMBAR、Boresch 约束和长程修正形成完整的自由能账本。

项目同时包含两条研究线：

- **生产主线**：以 `dual_lambda + softcore + IBS` 为默认路线，目标是得到可恢复、可审计、失败关闭的 ABFE 计算流程；
- **探索研究线**：DEXP、MACE、ORB、外层 λ 神经基势、解析/分组密度 CV、膜体系和带电配体 charge-transfer 等，用于研究能否改善困难窗口的 overlap、ESS 或单位 GPU 时间的有效采样效率。

截至 2026-08-12，可以作出的总体判断是：

- 生产主线的软件骨架、缓存、续算、能量账本、预优化、IBS 采样、MBAR/TMBAR 分析和多类 fail-closed 检查已经形成；
- 当前源码中的主要协议版本为 `IBS_BIAS_PROTOCOL_VERSION=29`、`THERMODYNAMIC_PATH_PROTOCOL_VERSION=21`、`TRADITIONAL_LJ_LRC_PROTOCOL_VERSION=3`、`WCA_ACCOUNTING_VERSION=2`、`ESS_GATE_PROTOCOL_VERSION=3`；
- 现有测试和历史 GPU 结果提供了大量工程证据，但验证矩阵、独立重复和时间相关不确定度仍未完全闭合；
- 2026-07-27 的旧 Atenolol 结合自由能结果已确认无效，只能用于审计；
- 当前外层神经势、MTS 和在线 ORB 路线均没有获得 production promotion；它们产生了有价值的负结果，而不是可投入生产的模型；
- 最新的 EXP-019 在 Stage 2 端点不确定度门上失败，尚未完成正式 baseline repeats；因此现在适合报告“软件和方法开发进度”，不适合报告“最终可信结合自由能”。

## 2. 科学问题与自由能定义

### 2.1 热力学循环

软件分别计算复合物腿和溶剂腿从完全耦合态到解耦态的自由能：

```text
Delta G_complex : ligand in binding site, coupled -> decoupled
Delta G_solvent : ligand in water,        coupled -> decoupled
```

当前软件的结合自由能符号定义为：

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

其中复合物腿包含 Boresch 约束的附加/释放账本，溶剂腿不使用 Boresch；`Delta G_APBS` 只有在显式提供外部 APBS 修正时才加入。对于真实有利结合，通常应有 `Delta G_complex > Delta G_solvent`，因此 `Delta G_bind < 0`。

### 2.2 双 λ 解耦

默认 `dual_lambda` 路线把解耦分为两个主要阶段：

1. **Stage 1 / decharging**：逐步移除配体与环境的静电耦合；默认生产方法为 PME decharging；
2. **Stage 2 / vanishing**：在电荷处理后逐步移除 van der Waals 相互作用，使用 softcore 势避免端点奇异性。

这两个阶段分别生成 λ 状态、采样窗口和能量矩阵，最终由 BAR/MBAR/TMBAR 类估计器汇总。实验性的 `shadow_ibs`、charge-transfer 和其它 Hamiltonian 不属于当前默认生产主线。

### 2.3 IBS 混合势与权重

IBS 在同一个采样 ensemble 中混合多个目标 λ 态。当前文档采用的核心形式为：

```text
p_k(x) proportional to exp[beta * (f_k - U_k(x))]
V_bias(x) = -kT * log sum_k exp[-beta * (U_k(x) - f_k)]
```

其中 `U_k(x)` 是构型 `x` 在第 `k` 个目标态下的能量，`f_k` 是用于平衡各态贡献的自由能偏置。预热阶段允许更新 `f_k`；一旦进入正式 production，`f_k` 必须冻结并受运行时只读锁保护，预热/验证数据不能混入 production 样本。

TMBAR 用于处理预热期间偏置随时间变化的历史；MBAR/BAR 用于固定 Hamiltonian 或最终能量矩阵的自由能和 overlap 分析。软件同时保存 overlap、importance ESS、去相关样本数、端点误差和 split-half 等诊断，但这些指标的科学含义并不相同，不能互相替代。

### 2.4 λ 路径与窗口划分

Stage 2 当前采用协议 v21 的混合度量路径。路径设计的目标是：

- 保留 coupled 和 decoupled 端点；
- 控制最大 λ 间隔；
- 用热力学度量为高难度区域分配更多状态；
- 将全路径划分为多个只共享单一边界态的 IBS ensemble；
- 生产后如果 coverage 不足，优先只补采失败窗口；必要时创建独立、不可变的 rescue ensemble，而不是原地移动 λ、重写 `f_k` 或改变已有 ensemble。

### 2.5 Boresch 约束、LJ 长程项和 APBS

- **Boresch 约束**用于在复合物腿解耦过程中稳定配体的位置和取向，并通过解析项恢复标准态自由能。锚点、平衡几何、力常数和构象身份必须进入缓存指纹；陈旧参数必须 fail closed。
- **LJ 长程修正**采用 switching-aware、softcore-aware 的解析平均场形式，并同时处理 `r^-6` 与 `r^-12` 项。OpenMM `CustomNonbondedForce` 的内建 LRC 在当前组合势下不直接使用，因为它会把不适用的表达式一起积分并曾造成 CUDA 问题。
- **APBS 修正**由独立工具准备 PQR/DX、运行 APBS 并汇总 snapshot correction。它是最终热力学循环的外部修正项，不能替代 LJ tail correction。

## 3. 软件架构

### 3.1 主入口与配置

`runabfe.py` 是主命令入口，负责：

- 解析命令行与 JSON 配置；
- 从 GROMACS 输入建立或加载 OpenMM System；
- 管理 complex/solvent 两条腿的缓存身份；
- 准备配体参数、Boresch 参数、膜/离子相关选项；
- 启动 IBS 或 traditional 模式；
- 执行 post-analysis，并生成最终结果和 provenance。

当前仓库实际使用 `abfe_config.json`。文档曾提到推荐 `abfe_config.yaml`，但该文件并不存在；JSON 中还保留了机器相关的 GROMACS 路径，因此跨机器复现前必须重新核对。

### 3.2 主要模块职责

| 模块 | 主要职责 | 当前定位 |
|---|---|---|
| `runabfe.py` | CLI、配置、系统构建/缓存、两腿编排、最终合并 | 生产入口 |
| `abfe_pipeline.py` | 阶段编排、窗口运行、质量门、resume、rescue、结果落盘 | 生产流程控制 |
| `ibs_engine.py` | IBS 势、采样器、TMBAR/MBAR、overlap、ESS、Boresch attachment、PME/softcore/LRC | 物理与采样核心 |
| `abfe_preoptimizer.py` | λ 度量、路径生成、窗口划分、pilot TI、geodesic/overlap 预优化 | 路径优化 |
| `abfe_core.py` | 系统处理、约束、膜体系、DEXP/charge-transfer 相关底层对象 | 基础物理组件 |
| `apbs_correction.py` | PQR/DX 准备、APBS 输入、运行与修正汇总 | 外部修正工具 |
| `dexp_experiment.py` | DEXP/MACE surrogate、轨迹/PMF/氢键/多初态分析 | 独立实验模块 |
| `outer_lambda_neural_basis.py` | 外层 λ 控制器、能量账本、TorchForce/MACE/ORB/学生模型和实验 CLI | 隔离的研究模块 |
| `tools/` | 诊断、迁移、修复、绘图和验证工具 | 辅助工具 |
| `scripts/` | Linux/PBS/GPU 实验入口和批处理 | 运行编排 |
| `tests/` | 物理数值、协议、缓存、resume、源码契约和实验模块测试 | 回归证据 |

外层神经模块被有意保持在生产模块之外。测试中存在“生产模块不得导入独立神经模块”的隔离契约；因此“尚未接入 production”是当前设计边界，不是漏接代码。

### 3.3 数据、缓存与 provenance

软件为 System、拓扑、Boresch、预平衡、λ 路径、窗口能量、checkpoint、IBS 状态和最终结果分别保存文件。重要身份包括：

- 输入 `.gro/.top` 及 include 依赖哈希；
- 配体参数和 System XML 身份；
- 坐标、盒向量、Boresch 几何及协议版本；
- λ 数组、窗口范围、势模型、WCA/LRC/ESS 协议；
- 随机种子、平台、软件版本和运行参数；
- 结果文件与实验报告的 SHA-256。

当前工作区的 `.git` 元数据不完整，无法提供 commit SHA 或可靠 dirty-state provenance。因此现阶段报告必须使用“文件哈希 + 协议版本 + 日期 + 运行命令/环境”代替 Git 版本声明。

## 4. 一次完整计算的执行流程

1. **输入和环境检查**：读取 GROMACS 坐标/拓扑、配体残基名、GROMACS include、配置和可选外部参数。
2. **System 建立与缓存**：生成 complex 和 solvent OpenMM System，保存 XML、拓扑和缓存 manifest。
3. **预平衡与构象身份**：处理 PBC、最小化/平衡、保存轨迹；复合物腿确定或加载 Boresch 锚点和几何。
4. **路径预优化**：对 Stage 1/2 进行 pilot 评估，生成 λ 网格和 IBS 窗口；协议身份进入缓存门。
5. **IBS 预热**：收集多态能量，学习候选 `f_k`，执行固定权重 burn-in/validation；不合格时按明确状态停止或进入允许的恢复路径。
6. **Production**：冻结 `f_k`，从独立第 0 步开始正式采样；保存能量、偏置、轨迹和 checkpoint。
7. **质量与 rescue**：计算 overlap、ESS、去相关样本和误差；只对失败窗口追加采样，仍不足时创建独立 rescue ensembles，原数据不覆盖。
8. **阶段估计**：对 decharging、vanishing 和 Boresch attachment/release 分别估计自由能及误差。
9. **两腿合并**：按当前符号约定合并 complex、solvent、Boresch、LJ LRC 和可选 APBS。
10. **结果审计**：写出 `final_results.json`、`final_binding_results.json`、热力学循环说明、diagnostics 和 provenance。

## 5. 当前已经形成的能力

| 能力 | 代码状态 | 证据状态 | 备注 |
|---|---|---|---|
| GROMACS→OpenMM complex/solvent 构建与缓存 | `IMPLEMENTED` | 有历史真实体系证据 | 跨机器需修正 `gmx_path` |
| dual-lambda decharging/vanishing | `IMPLEMENTED` | 有真实运行和结果文件 | 当前整体不确定度尚未闭合 |
| IBS warmup、冻结 `f_k`、production 隔离 | `IMPLEMENTED` | 静态/回归与部分 GPU 证据 | 当前协议已到 v29，旧文档版本不能直接代替当前验证 |
| MBAR/TMBAR/BAR 估计和 overlap/ESS 诊断 | `IMPLEMENTED` | 合成数值测试和真实数据分析存在 | 时间相关与跨运行方差仍需加强 |
| checkpoint/resume 和缓存 fail-closed | `IMPLEMENTED` | 多项定向测试存在 | 验证矩阵仍有完整环境项目待运行 |
| immutable rescue ensembles | `IMPLEMENTED` | 静态/部分运行证据 | 需要当前协议下目标 CUDA 复核 |
| Boresch attachment/release 与谐振性诊断 | `IMPLEMENTED` | 旧 bug 已定位并加身份门 | 当前有力常数裁剪，需要结果中披露 |
| switching/softcore-aware LJ LRC | `IMPLEMENTED` | 当前协议 v3 | traditional fixed-box 小回归仍待跑 |
| APBS 外部修正工具 | `IMPLEMENTED` | 工具级 | 当前主结果中修正值为 0，未形成完整闭环证据 |
| traditional REMD | `IMPLEMENTED` | 历史 12-context CUDA 复测通过 | 固定盒/LRC 回归 V-02 待完成 |
| 膜体系预平衡和质量门 | `IMPLEMENTED` | 100 ns NPT 门有通过记录 | Stage 1/2 ABFE 尚未完成 |
| 带电配体 co-alchemical/charge-transfer | `IMPLEMENTED/EXPERIMENTAL` | 局部力学门修复后通过 | 缺真实 charged-ligand 完整循环 |
| DEXP/MACE/ORB/外层 λ | 独立研究实现 | 多条实验证据 | 无路线获得 production promotion |

## 6. 当前科学结果与可信边界

### 6.1 2026-07-27 Atenolol 结果：无效但必须保留

该轮曾得到：

```text
Delta G_complex = 145.54 +/- 1.64 kJ/mol
Delta G_solvent = 161.54 +/- 1.47 kJ/mol
Delta G_bind    = +16.00 +/- 2.20 kJ/mol
```

这个结果**不得作为科学结论引用**。根因是复合物腿复用了陈旧且错误的 `boresch_equilibrium_committed.json`：两个角度平衡值发生对调，三个二面角也错误；体系后来重新平衡，但旧保护只检查文件是否存在，没有验证其是否仍描述当前构象。约束把配体从自身 pose 拉离约 3.42 Å，而无约束预平衡漂移仅约 0.60 Å，导致方向性氢键丢失、复合物去电荷自由能异常以及 Boresch 释放项错误。

此外，磁盘上一些旧 `thermodynamic_cycle.md`/结果还使用与当前代码相反的结合自由能符号。旧数据只能用于说明 bug 发现和协议演化，不可重新包装为“早期预测值”。

### 6.2 `output_lrc_fix`：当前验收基线，但不是最终发表结果

`output_lrc_fix` 包含修复 Boresch/LRC/符号等问题后的当前候选基线。其 `final_binding_results.json`（2026-07-29）记录 `Delta G_complex=180.9981`、`Delta G_solvent=157.8358`、`Delta G_bind=-23.1622 +/- 2.5139 kJ/mol`（`-5.5359 +/- 0.6014 kcal/mol`）。artifact 协议身份为 IBS v29、path v21、LJ LRC v3、WCA v2；两腿 Stage 2 均标记 converged、使用 23 个 lambda nodes 且无 dropped windows。但独立重复为 false、main production seed ledger 为空，Boresch 有一个 `kr` 从约 7355.9 裁剪至 2000。旧 status snapshot 中另一组两腿值重算的约 `-40.84 kJ/mol` 与该 artifact 不是同一结果，禁止混用。因此 `-23.16 kJ/mol` 只能称为归档候选结果，不能称为最终发表结果。

### 6.3 EXP-017 至 EXP-019：问题从“λ 间距”转向“时间相关不确定度”

- **EXP-017**：P0-A 的账本/TMBAR 门通过，但没有定位到具体低 overlap λ 边；`min overlap≈0.3913`、最小去相关样本约 96。window 5 出现 `-0.5587 kJ/mol`、约 `4.46×2σ` 的 split-half drift。结论为 `INCONCLUSIVE/STOPPED`，不授权 fixed-λ probe、插 λ、P1 或 P2。
- **EXP-018**：对 window 5 做 3 个独立 seed 的 stationarity confirmation。三个 drift z 值约为 `1.134/2.568/1.381`，只有 `1/3` 重现负向漂移；重复间方差比约 `16.76`，出现 MBAR 单次误差低估信号，但裁决仍是 `INCONCLUSIVE/CLOSED`。
- **EXP-019**：零成本归因显示重复差异主要集中在 weighted interaction 的 state 0 描述性方差；随后 v3 在正式 baseline repeat 之前就未通过 Stage 2 端点不确定度门：`1.2481 > 1.0 kJ/mol`，完成的 baseline repeats 为 0。诊断 rescue 值 `159.3165 +/- 2.0618 kJ/mol` 不可晋级为端点结果。下一步只授权只读 Stage 2 rescue/coverage audit。

这一系列结果说明：当前困难不能再简单归因于某条 λ 边过宽；跨时间和跨运行的不确定度模型是下一阶段的核心。

