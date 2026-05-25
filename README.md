# Atenolol-rank1 ABFE 项目说明

本项目是一个基于 OpenMM 的 ABFE（Absolute Binding Free Energy，绝对结合自由能）计算工作目录，当前以 `Atenolol-rank1` 体系为例，围绕 5 个核心 Python 文件组织完整计算流程。

这份 README 不只是“怎么运行”，更重点解释这套代码的逻辑分层、数据流、两条计算腿的组织方式，以及每个脚本在全流程中的职责。

## 1. 项目目标

该项目的目标是计算配体与受体体系的结合自由能 `ΔG_bind`。从主入口 `runabfe.py` 的实现可以看出，程序采用标准 ABFE 双腿思路：

- 先计算复合物腿（complex leg）
  这里是“蛋白/受体 + 配体 + 溶剂”体系中的配体去耦合自由能

- 再计算溶剂腿（solvent leg）
  这里是“纯水中的配体”去耦合自由能

- 如果启用了 Boresch 限制，还会加入解析修正项

最终结果按下面的形式汇总：

```text
ΔG_bind = ΔG_complex - ΔG_solvent + ΔG_restraint
```

在 `runabfe.py` 末尾，这个结果会保存为：

- `output/final_binding_results.json`

## 2. 目录中最重要的文件

### 2.1 核心 Python 文件

这个项目当前的主要逻辑集中在 5 个 Python 文件中：

1. `runabfe.py`
   唯一推荐入口。负责参数解析、配置合并、系统加载、缓存管理、Boresch 处理、复合物腿和溶剂腿调度、最终结果汇总。

2. `abfe_pipeline.py`
   主流程调度器。负责把“预平衡 -> 预优化 -> 分阶段采样 -> MBAR 分析 -> 汇总结果”这条主线串起来。

3. `abfe_preoptimizer.py`
   预优化模块。负责 ACES/IBS 路径预扫描、自适应 lambda 分布优化、双 lambda 阶段路径生成。

4. `ibs_engine.py`
   生产采样引擎。负责真正的 IBS 窗口构建、逐窗口模拟、能量落盘、checkpoint、全局 MBAR/TMBAR 分析。

5. `abfe_core.py`
   物理核心库。负责软核势、DEXP 势、Boresch 限制力、几何估算器、解析修正、单位格式化和系统复制等底层能力。

### 2.2 典型输入文件

- `solv_ions.gro`
- `topol.top`
- `Atenolol-rank1.pdb`
- `Atenolol-rank1.mol2`
- `abfe_config.json`

这些文件提供初始结构、拓扑、配体信息与运行参数。

### 2.3 典型输出目录

- `output/`
- `output/checkpoints/`
- `output/solvent_leg/`

其中 `output/` 是复合物腿默认输出目录，`output/solvent_leg/` 是溶剂腿输出目录。

## 3. 整体架构

从代码结构看，这套程序可以理解为 4 层：

### 第 1 层：命令行与工程组织层

由 `runabfe.py` 负责。

这一层解决的问题是：

- 用户怎么传参
- 怎么从 JSON 配置和命令行合并得到最终运行参数
- 系统是从已有缓存读取，还是从 GROMACS 文件重新构建
- 是否需要自动创建溶剂腿
- 是否启用 Boresch
- 是新跑、续跑，还是只分析已有结果

### 第 2 层：流程调度层

由 `abfe_pipeline.py` 负责。

这一层把一次 ABFE 任务拆成可管理的步骤：

- 预平衡
- 坐标修复与安全松弛
- 路径预优化
- 去电荷阶段采样
- 去范德华阶段采样
- 全局自由能分析
- Boresch 修正与最终结果写出

### 第 3 层：窗口采样与统计分析层

由 `ibs_engine.py` 负责。

这一层关注的是：

- 如何把一条 lambda 路径切成重叠窗口
- 每个窗口如何构建 OpenMM 系统
- 如何进行逐窗口模拟
- 如何保存能量矩阵和偏置历史
- 如何做全局 MBAR / TMBAR 拼接

### 第 4 层：物理模型层

由 `abfe_core.py` 负责。

这一层是底层“物理零件库”，提供：

- ACES softcore 势
- DEXP surrogate 势
- Lambda 依赖的 Boresch 力
- Boresch 解析修正
- 几何约束估算器
- Orb/几何方法的 Boresch 候选生成
- 单位处理与结果格式化

## 4. 主执行逻辑

这里按 `runabfe.py -> abfe_pipeline.py -> ibs_engine.py` 的调用关系，把程序主线梳理一遍。

### 4.1 命令行入口

程序入口在：

- `runabfe.py:1362` 的 `main()`

主入口支持几类模式：

- 默认 ABFE 运行模式
- `prepare` 预处理子命令
- `traditional` 传统模式
- `--analyze-only` 只分析已有结果

其中正常最常用的是默认模式。

### 4.2 配置读取与参数合并

`runabfe.py` 会先解析命令行参数，再构造 `RunConfig`。这意味着：

- `abfe_config.json` 提供默认值
- 命令行参数优先级更高

当前目录自带的 `abfe_config.json` 里已经设置了典型生产参数，例如：

- `preset = production`
- `platform = CUDA`
- `decoupling = dual_lambda`
- `potential = softcore`
- `stage1_n_states = 16`
- `stage2_n_states = 16`
- `n_equil_steps = 5000000`

### 4.3 系统加载策略

主程序优先走缓存加载，而不是每次都重新从 GROMACS 构建。

执行逻辑是：

1. 如果 `output/` 中已有原生缓存，并且没有指定 `--reset`
   直接调用 `load_native_system(...)` 读取缓存

2. 如果没有缓存
   先从 `--gro` 和 `--top` 构建系统

3. 构建完成后
   立即写入原生缓存，再重新从缓存加载一次

这样做的意义是：

- 统一后续对象来源
- 避免 GROMACS 解析与即时对象混用引入副作用
- 让续跑机制更稳定

### 4.4 初始坐标处理

系统加载后，`runabfe.py` 会对坐标做两件事：

1. 转成安全、统一的坐标格式
2. 通过 `center_system_rigidly(...)` 做配体居中与 PBC 完整性修复

这个步骤很关键，因为后面的 Boresch 估计、窗口初始化、限制力检查都依赖合理的几何构型。

### 4.5 自动构建溶剂腿

这套代码不是只算复合物腿，而是一键 ABFE。

因此在复合物腿运行前，主程序会检查溶剂腿缓存是否存在：

- 如果不存在，自动构建 ligand-in-water 体系并缓存
- 如果已经存在，则直接复用

这意味着用户只要给出复合物体系和配体信息，程序就会自动补齐溶剂腿所需缓存。

### 4.6 Boresch 限制处理

主入口会调用：

- `resolve_boresch_restraint(...)`

用来确定复合物腿是否启用 Boresch 以及参数从哪里来。代码中支持几种来源：

- 简单几何法
- Orb ML/估算器路线
- 外部文件指定

如果启用了 Boresch，并且没有 `--skip-rebalance`，程序还会调用：

- `ABFEPipeline._rebalance_with_boresch(...)`

先做一次带限制力再平衡，目的是让当前坐标与限制几何更一致，避免正式采样时刚开始就产生不合理拉扯。

### 4.7 复合物腿计算

复合物腿通过：

- `ABFEPipeline.run_full_pipeline(...)`

启动。

这里会把以下参数传进去：

- decoupling 方案
- 势能模型
- Boresch 参数
- 二面角修正参数
- 续跑标记
- 是否执行预平衡
- 每窗口步数
- 每次更新步数
- 每阶段状态数

### 4.8 溶剂腿计算

复合物腿结束后，程序会从缓存中加载溶剂腿系统，再创建第二个 `ABFEPipeline` 实例。

这里有一个非常重要的硬编码逻辑：

- 溶剂腿强制 `boresch_params = None`

也就是说，溶剂腿绝对不挂 Boresch 限制。这符合 ABFE 常见做法，因为 Boresch 只作用在复合物腿中的受体-配体相对构型约束上。

### 4.9 最终结果汇总

最后 `runabfe.py` 会取出：

- `dg_complex`
- `dg_solvent`
- `dg_boresch`

并计算：

```text
ΔG_bind = (ΔG_complex - ΔG_solvent) + ΔG_boresch
```

误差则用两腿误差平方和开根号合并。

## 5. `abfe_pipeline.py` 的核心逻辑

`abfe_pipeline.py` 是整个项目最重要的流程层文件，建议在阅读代码时重点先看它。

### 5.1 `ABFEPipeline`

主类在：

- `abfe_pipeline.py:405`

它持有一次单腿计算所需的核心状态：

- system
- topology
- positions
- box_vectors
- ligand_indices
- output_dir
- checkpoint_dir
- platform_name

也就是说，`ABFEPipeline` 本质上就是“一条腿的完整运行上下文”。

### 5.2 `run_full_pipeline(...)`

完整入口在：

- `abfe_pipeline.py:1811`

这个函数大体按下面顺序执行。

#### 步骤 1：设备策略检测

程序会根据窗口数和平台名选择 GPU 策略，决定使用哪些设备。

#### 步骤 2：恢复全局状态

如果开启了 `resume`，会读取 pipeline state，检查哪些阶段已经完成，避免重复计算。

#### 步骤 3：应用二面角修正

如果外部传入了 `torsion_params`，会在预平衡前应用到系统里。

#### 步骤 4：预平衡

这是单腿计算的物理准备阶段。

逻辑包括：

- 如果 `resume` 且预平衡轨迹和状态都有效，则尝试跳过
- 如果轨迹有效，直接读取最后一帧作为新的起始构型
- 如果状态不完整或检查失败，则重新执行预平衡
- 新预平衡完成后，再做一次快速最小化，去掉残余应力

这部分逻辑明显做了较多工程加固，目的是提高断点续跑的稳定性。

#### 步骤 5：安全松弛

无论是否跳过预平衡，pipeline 都会执行一次短的 L-E 界面安全弛豫。

从代码注释看，这一步是为了缓解：

- PBC 修复
- 抽帧续跑
- 坐标重载

可能导致的局部水分子与配体接触冲突。

#### 步骤 6：更新 Boresch 平衡几何量

如果 Boresch 参数有效，pipeline 会根据当前最新坐标更新：

- 距离平衡值
- 角度平衡值
- 二面角平衡值

这样可以减少限制项和当前构型的偏差。

#### 步骤 7：阶段级续跑判断

`dual_lambda` 模式下，程序会分别检查：

- Stage 1: decharging
- Stage 2: vanishing

同时也会检查预优化缓存文件是否匹配当前状态数。

如果匹配，则直接复用已有优化路径。

### 5.3 双 lambda 路线

从 `run_full_pipeline(...)`、`DualLambdaPreOptimizer` 和 `IBSWindowManagerDualLambda` 可以看出，当前项目的默认主路线是 `dual_lambda`。

它把去耦合过程拆成两个阶段：

1. Stage 1: 去电荷（decharging）
   一般保持 vdw 端开启，逐步关闭 coulomb 相互作用

2. Stage 2: 去范德华（vanishing）
   在 coulomb 已关闭基础上，继续逐步关闭 vdw 相互作用

这么做的好处是：

- 物理路径更稳定
- 采样更容易控制
- 能把电荷与范德华去耦合分开优化

### 5.4 `TraditionalABFEPipeline`

除了 IBS 双 lambda 路线外，文件中还有：

- `TraditionalABFEPipeline`

位置在：

- `abfe_pipeline.py:2501`

这个类提供传统模式，适合做对照、兼容旧路线或调试。不过从主入口默认参数和整体代码重心来看，当前推荐路线仍然是 `ibs + dual_lambda`。

## 6. `abfe_preoptimizer.py` 的核心逻辑

预优化模块解决的是一个非常实际的问题：

- lambda 状态不是越均匀越好
- 需要根据能量波动和采样难度，把更多状态分配到更“难”的区间

### 6.1 `ABFEPreOptimizer`

主类在：

- `abfe_preoptimizer.py:61`

这个类面向较通用的单路径优化逻辑，负责：

- 探测系统中的 lambda 参数名
- 计算能量波动
- 优化 softcore 参数
- 优化窗口范围
- 自适应调整 lambda 路径

### 6.2 `DualLambdaPreOptimizer`

双 lambda 专用类在：

- `abfe_preoptimizer.py:593`

它会分别为两个阶段做路径优化：

- `optimize_stage1_decharging(...)`
- `optimize_stage2_vanishing(...)`

### 6.3 Stage 1 的优化思想

去电荷阶段的逻辑大致是：

- 固定 `lambda_vdw = 1`
- 让 `lambda_coul` 从 1 逐步走到 0
- 在每个候选点短采样，收集 group 1 能量
- 用能量方差估计该区域的“困难程度”
- 把更多 lambda 点分布到方差更大的区域

换句话说，它不是简单线性切分，而是“哪里难就往哪里多放状态”。

### 6.4 Stage 2 的优化思想

去范德华阶段的逻辑类似：

- 固定 `lambda_coul = 0`
- 让 `lambda_vdw` 从 1 逐步走到 0
- 对每个候选点做短采样
- 依据能量方差重新分布 lambda 点

### 6.5 预优化的意义

这一步的意义不是“计算最终自由能”，而是为正式 IBS 采样准备更好的离散路径：

- 降低相邻状态差异
- 改善窗口重叠
- 提高 MBAR 的可解性
- 减少把采样预算浪费在过于容易的区间

## 7. `ibs_engine.py` 的核心逻辑

这个文件是真正做生产采样的引擎层。

### 7.1 `generate_overlapping_windows(...)`

函数位置：

- `ibs_engine.py:632`

它负责把一串 lambda 状态切分为多个重叠窗口。

之所以要重叠，是因为：

- 相邻窗口之间需要共享一部分状态
- 这样才能把局部采样结果拼接成全局自由能曲线

### 7.2 `IBSWindowManagerDualLambda`

主窗口管理器在：

- `ibs_engine.py:1501`

它负责：

- 为每个窗口构建 IBS 双 lambda 系统
- 配置平台与积分器
- 设置周期性盒子
- 做最小化
- 做测试步进
- 做 Boresch 安全检查
- 启动正式生产采样
- 原子化保存能量、偏置和基础能量数组

### 7.3 为什么窗口管理器里有很多“安全步骤”

从代码看，窗口正式开始前会做很多保护性操作：

- 最小化
- Boresch 几何检查
- 测试性步进
- NaN 检测
- 力分解诊断
- 约束死锁检查
- Boresch 强度渐进恢复

这些逻辑说明项目作者非常关注实际模拟中容易出现的工程问题，例如：

- 坐标炸掉
- 约束不收敛
- 初始限制力过强
- 某些窗口一启动就 NaN

所以这个文件不只是“跑模拟”，更像是“带大量稳定性护栏的生产执行器”。

### 7.4 MBAR / TMBAR 分析

该文件中负责分析的主要对象包括：

- `GlobalMBARAnalyzer`
- `TraditionalMBARAnalyzer`
- `solve_stage_integrated(...)`

它们的职责是把每个窗口输出的局部能量矩阵整合起来，求得：

- 每一阶段的自由能变化
- 不确定度
- 最终可供 pipeline 汇总的阶段结果

## 8. `abfe_core.py` 的核心逻辑

这是底层物理能力文件，代码量也很大，但阅读时不一定要最先啃。

### 8.1 `ACESoftcorePotential`

位置：

- `abfe_core.py:123`

这是默认 softcore 势模型的核心类，用于构建和传递去耦合过程中需要的软核参数。

### 8.2 `DEXPSurrogatePotential`

位置：

- `abfe_core.py:250`

如果用户选择 `--potential dexp`，就会走这条分支。

它的作用是提供一个替代 softcore 的 DEXP 势表示，并支持从字典或 JSON 参数恢复。

### 8.3 Boresch 相关

这个文件中与 Boresch 相关的核心对象有：

- `calculate_boresch_analytical_correction(...)`
- `LambdaDependentBoreschForce`
- `GeometricRestraintEstimator`
- `OrbBoreschEstimator`

它们分别覆盖：

- 解析修正计算
- 可随 lambda 缩放的限制力实现
- 几何候选约束生成
- 机器学习/Orb 路线的候选估计

### 8.4 `UnitFormatter`

位置：

- `abfe_core.py:2617`

主要用来做结果格式化，方便把自由能结果打印成人类可读形式。

### 8.5 `ensure_owned_system(...)`

位置：

- `abfe_core.py:2740`

这个函数是工程上很重要的辅助函数，用于确保系统对象是当前上下文安全拥有的副本，避免窗口构建或多阶段处理中因为共享对象产生副作用。

## 9. 5 个 Python 文件之间如何协作

可以把它们理解成下面这条调用链：

```text
runabfe.py
  -> abfe_pipeline.py
     -> abfe_preoptimizer.py
     -> ibs_engine.py
        -> abfe_core.py
```

更具体一点：

1. `runabfe.py` 决定“跑什么、从哪里开始、哪些参数生效”
2. `abfe_pipeline.py` 决定“一条腿按什么顺序跑”
3. `abfe_preoptimizer.py` 决定“路径怎么分布更合理”
4. `ibs_engine.py` 决定“窗口怎么建、怎么采样、怎么分析”
5. `abfe_core.py` 提供“底层物理对象与通用工具”

## 10. 当前目录里的 `abfe_config.json`

当前项目已经带了一个配置文件：

- [abfe_config.json](/K:/ABFE_IBS/Atenolol-rank1/abfe_config.json:1)

其中值得特别注意的字段有：

- `preset`
  预设强度，当前是 `production`

- `platform`
  当前设置为 `CUDA`

- `decoupling`
  当前设置为 `dual_lambda`

- `potential`
  当前设置为 `softcore`

- `boresch`
  当前默认开启

- `n_equil_steps`
  预平衡步数

- `n_steps_per_window`
  每个窗口的生产步数

- `stage1_n_states` / `stage2_n_states`
  两阶段的状态数

- `gmx_path`
  当前配置里是 Linux 风格路径，如果你在 Windows 或另一台 Linux 机器上运行，通常需要改成你本机实际的 GROMACS 力场目录

## 11. 如何运行

### 11.1 最常用命令

```bash
python runabfe.py --ligand MOL --gro solv_ions.gro --top topol.top --config abfe_config.json
```

### 11.2 续跑

```bash
python runabfe.py --ligand MOL --config abfe_config.json --resume
```

适用场景：

- `output/` 中已经有缓存和 checkpoint
- 希望从中断处继续

### 11.3 强制重跑

```bash
python runabfe.py --ligand MOL --gro solv_ions.gro --top topol.top --config abfe_config.json --reset
```

适用场景：

- 你怀疑缓存已经不可靠
- 改了关键参数，想从头计算

### 11.4 只做分析

```bash
python runabfe.py --ligand MOL --config abfe_config.json --analyze-only
```

前提是对应能量数据和中间结果已经存在。

### 11.5 预处理子命令

代码中还支持：

```bash
python runabfe.py prepare --gro solv_ions.gro --top topol.top --ligand MOL --output-dir ./prep_output
```

这个子命令可用于生成预处理文件，比如：

- Boresch 文件
- DEXP 拟合输入/输出

## 12. 主要命令行参数

从 `runabfe.py` 的参数定义来看，常用参数包括：

- `--gro`
  GROMACS 结构文件，首次构建时需要

- `--top`
  GROMACS 拓扑文件，首次构建时需要

- `--ligand`
  配体残基名，例如 `MOL`

- `--ligand-xml`
  配体力场 XML/FFXML，主要用于溶剂腿构建

- `--gmx-path`
  GROMACS 力场 include 目录

- `--output`
  输出目录，默认 `./output`

- `--config`
  JSON 配置文件

- `--resume`
  从 checkpoint 恢复运行

- `--reset`
  忽略缓存，强制重新开始

- `--mode`
  `ibs` 或 `traditional`

- `--decoupling`
  `dual_lambda`、`single_lambda`、`2d_diagonal`、`2d_geodesic`

- `--potential`
  `softcore` 或 `dexp`

- `--boresch` / `--no-boresch`
  显式开启或关闭 Boresch 限制

- `--boresch-source`
  指定 Boresch 来源

- `--skip-rebalance`
  启用 Boresch 时跳过再平衡

- `--rebalance-steps`
  再平衡步数

- `--n-steps-per-window`
  每窗口采样步数

- `--steps-per-update`
  采样更新频率

- `--n-states-per-stage`
  每阶段状态数

- `--enable-early-stop`
  启用提前停止逻辑

- `--enable-gradual-warmup`
  启用渐进预热

- `--disable-warmup`
  关闭预热

- `--warmup-steps`
  预热步数

- `--n-workers`
  并行 worker 数

- `--parallel-stages`
  去电荷和去 vdW 阶段并行执行

## 13. 输出文件说明

运行结束后，常见输出包括：

- `output/pipeline.log`
  主流程日志

- `output/pre_equilibration.dcd`
  预平衡轨迹

- `output/checkpoints/pre_equil.chk`
  预平衡 checkpoint

- `output/checkpoints/pipeline_state.json`
  全局流程状态

- `output/checkpoints/preopt_dual_decharging.json`
  去电荷阶段预优化缓存

- `output/checkpoints/preopt_dual_vanishing.json`
  去范德华阶段预优化缓存

- `output/final_results.json`
  单腿结果

- `output/final_binding_results.json`
  最终结合自由能结果

- `output/solvent_leg/final_results.json`
  溶剂腿结果

- `output/solvent_leg/checkpoints/`
  溶剂腿断点续跑状态

此外，还会看到大量窗口级文件，例如：

- `dual_window_*_energies.npy`
- `dual_window_*_bias.npy`
- `dual_window_*_base.npy`

它们是 MBAR/TMBAR 分析的原始能量输入。

## 14. 这个项目的工程特点

从代码实现看，这个项目有几个很鲜明的特点。

### 14.1 强调缓存优先

不是每次现构系统，而是尽量从原生缓存恢复。

### 14.2 强调断点续跑

不只是简单 checkpoint，而是：

- 预平衡可续跑
- 阶段采样可续跑
- 预优化可复用
- 结果文件可复用

### 14.3 强调稳定性护栏

例如：

- 坐标居中与 PBC 修复
- 安全松弛
- Boresch 几何检查
- 测试步进
- NaN 诊断
- 约束死锁预警

这说明代码非常偏“生产工程化”，而不仅仅是论文级原型脚本。

### 14.4 双 lambda 是主路线

虽然代码支持其它模式，但当前默认配置、主要类实现和大量优化逻辑都明显围绕 `dual_lambda` 展开。

## 15. 建议的阅读顺序

如果你准备继续维护这套代码，推荐按下面顺序阅读：

1. [runabfe.py](/K:/ABFE_IBS/Atenolol-rank1/runabfe.py:1362)
   先理解程序从哪里进、整体做了什么

2. [abfe_pipeline.py](/K:/ABFE_IBS/Atenolol-rank1/abfe_pipeline.py:1811)
   再理解一条腿如何完整运行

3. [abfe_preoptimizer.py](/K:/ABFE_IBS/Atenolol-rank1/abfe_preoptimizer.py:593)
   接着看双 lambda 路径是怎么优化出来的

4. [ibs_engine.py](/K:/ABFE_IBS/Atenolol-rank1/ibs_engine.py:1501)
   然后看窗口是怎么逐个采样和分析的

5. [abfe_core.py](/K:/ABFE_IBS/Atenolol-rank1/abfe_core.py:123)
   最后再深挖具体物理对象和限制力实现

## 16. 当前 README 的适用边界

这份文档是根据当前目录中的源码结构直接整理的，重点反映的是：

- 这套代码现在是怎么组织的
- 默认推荐路线是什么
- 各模块之间怎样配合

它不是 ABFE 理论教材，也不是每个公式的完整推导说明。如果你后面希望，我还可以继续补两类文档：

1. “偏理论版 README”
   重点讲 ABFE、Boresch、dual lambda、IBS、MBAR 的概念与公式

2. “偏开发版 README”
   重点讲类图、关键函数、输入输出结构、常见改代码入口
