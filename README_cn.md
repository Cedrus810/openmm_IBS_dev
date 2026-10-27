# ABFE-IBS：绝对结合自由能工作流

🌐 [**English (项目首页) →**](README.md)　·　[文档导航](docs/README.md)

ABFE-IBS 是一个基于 OpenMM 的绝对结合自由能（ABFE）工作流。它读取 GROMACS
`.gro/.top` 体系，分别计算配体在蛋白复合物和体相溶剂中的去耦自由能，并用 IBS
采样、MBAR/TMBAR、Boresch 约束账本和显式长程修正组合完整热力学循环：

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

管线覆盖 GROMACS→OpenMM 建系、complex/solvent 两腿、dual-lambda 解耦（先去电荷、再去 vdW）、
MBAR/TMBAR 估计、Boresch attachment/release 记账、LJ 长程修正、缓存、续跑和 fail-closed 质量门。
每一步实现的是哪篇文献的方法、以及协议版本，见 [docs/METHODS.md](docs/METHODS.md)。

## AIBS：vdW 阶段的自适应 integrated Boltzmann sampling

integrated Boltzmann sampling（IBS）把相邻 λ 态合进一个混合势。AIBS 里相邻 λ 态分成
**window**；能量偏移固定的混合势叫 **generator**，在它下面跑的一条轨迹叫一个 **segment**。
每个 segment 期间 generator 不变，所以跑完的 segment 可以先互相比较，再决定要不要补采：

- **第一轮**：复合物腿每个 window 跑三个 segment —— 一个中心 generator，加两个向窗口两端
  各加小线性倾斜的 generator（溶剂腿每个 window 只跑中心一段）。
- 每轮结束后对每个 window 做 **segment deletion**（逐个去掉 segment 重新求解）、pooled 逐态 ESS
  和 top-1% 权重占比检查；不过的 window 下一轮补 segment。
- segment 之间的分歧计入报告的不确定度。

AIBS 作用于 `dual_lambda` 解耦的 vdW 阶段（stage 2），在运行配置里开启（**默认关闭**），
并要求固定的 stage 2 布局：均匀 λ 格点（`stage2_uniform_lambda_states` = N 给出
λ_i = 1 − i/(N−1)）和显式窗口（`stage2_window_ranges`，左闭右开的态下标区间，相邻窗口共用
边界态）。论文中性 benchmark 用的 stage 2 设置：

```json
{
  "decoupling": "dual_lambda",
  "n_steps_per_window": 250000,
  "stage2_uniform_lambda_states": 26,
  "stage2_window_ranges": [[0, 6], [5, 11], [10, 16], [15, 21], [20, 26]],
  "stage2_lcr_rounds": true
}
```

`n_steps_per_window` 是一个 segment 的长度（250000 步 = 2 fs 下 0.5 ns）。只开
`stage2_lcr_rounds` 而不给布局键，运行会直接报错退出。

可选参数（默认值在 `stage2_rounds.py`）：

| 键 | 默认 | 含义 |
|---|---|---|
| `stage2_lcr_max_rounds` | 3 | 最多轮数（含第一轮） |
| `stage2_lcr_sigma_tol_kJ` | 1.5 | segment deletion 灵敏度阈值（kJ/mol） |
| `stage2_lcr_tilt_kT` | 2.0 | 第一轮左/右 generator 的首末总倾斜（kT） |
| `stage2_lcr_trust_floor_kT` | 3.0 | 重定中心时信赖域半径下限（kT） |
| `stage2_lcr_burn_in_steps` | 25000 | 每个 segment 开头丢弃的步数 |

pooled ESS（20）和 top-1% 权重（0.35）两个阈值写死在代码里。

## 引用与数据

<!-- TODO 发布前：作者、期刊信息、数据归档 DOI -->
使用本代码请引用：

> *Adaptive Integrated Boltzmann Sampling for the van der Waals Stage of Absolute
> Binding Free-Energy Calculations*（稿件准备中，正式引用待补）。

支撑论文的数据集与分析脚本将存入公开数据归档，DOI 届时在此给出。

## 1. 环境

核心依赖：**Python 3.12**、OpenMM 8.5.2、NumPy、SciPy、MDTraj、PyMBAR（钉在
`pymbar-core=4.2.0`）。GPU 运行还需要 CUDA。`environment.yml` 是完整的生产环境
（`environment-ci.yml` 是 CPU CI 的子集），含 CUDA 相关选择，**换机器前要复核**。

```bash
conda env create -f environment.yml
python -c "import openmm, numpy, scipy, mdtraj, pymbar; print(openmm.__version__)"
```

`python runabfe.py --help` 在 import 时报 `No module named 'openmm'`，说明当前 shell
不在可运行的环境里，不是命令行参数错误。

自检与配置诊断（只读，不建 Context）：

```bash
python runabfe.py doctor
python runabfe.py validate-config --config abfe_config.json
python runabfe.py config-template --out my_system.json
```

## 2. 输入

首次建系通常需要：

| 参数 | 含义 |
|---|---|
| `--gro` | GROMACS 坐标文件 |
| `--top` + `--gmx-path` | GROMACS 拓扑及其 include 树（`--gmx-path` 给 GROMACS 安装前缀；留空时依次探测 `GMXLIB`/`GMXDATA`/`PATH`） |
| `--ligand` | 配体残基名 |
| `--output` | 独立的**新**输出目录 |

`abfe_config.json` 是参考配置，含为早期运行冻结的选项，**不是新体系模板**——用
`config-template` 生成，或读[迁移指南](docs/MIGRATING_TO_A_NEW_SYSTEM.md)。

**不要把一个体系的 checkpoint 拿给另一个体系用。**

## 3. 运行

新体系、显式输入：

```bash
python runabfe.py \
  --config abfe_config.json \
  --gro /path/to/system.gro \
  --top /path/to/topol.top \
  --ligand LIG \
  --gmx-path /path/to/gromacs \
  --output ./output_new_system \
  --boresch --boresch-source simple
```

续跑：

```bash
python runabfe.py --config abfe_config.json --ligand MOL --resume
```

只分析已有能量与 checkpoint（不跑动力学）：

```bash
python runabfe.py --config abfe_config.json --ligand MOL --analyze-only
```

用 `--resume`、`--reset`、`--analyze-only` 之前**先读**[输出与续跑](docs/OUTPUTS_AND_RESUME.md)。

## 4. 读结果

结果落在 `--output` 目录：`final_binding_results.json` 是汇总，`run_provenance.json`
记协议身份与输入指纹，`checkpoints/` 是续跑依据。符号约定、各项定义和续跑语义见
[OUTPUTS_AND_RESUME.md](docs/OUTPUTS_AND_RESUME.md)。

跑完只代表算完，不代表科学上已验证：引用数字之前先看收敛与重复间散布。

## 5. 可选功能（默认关闭）

| 功能 | 状态 | 文档 |
|---|---|---|
| outer-λ 局部残差（`--outer-lambda-local-residual-ibs`） | 实验性，只用于复合物腿。仓库不带冻结权重：先不开它跑一次，从该次的 stage 2 帧重训，再带 `--outer-lambda-resource-manifest` 重跑。使用 `plugins/LocalManyBodyResidual/build/` 里预编译的 CUDA 插件 | [RETRAIN_LOCAL_RESIDUAL.md](docs/RETRAIN_LOCAL_RESIDUAL.md) |
| reference IBS 对照臂（运行配置里 `stage2_online_ibs`，与 `stage2_lcr_rounds` 同开） | 只接管复合物腿 stage 2：偏移在生产中在线学习；复合物 stage 2 采完即结束，不写结合自由能。离线 TMBAR 分析脚本不在本仓库 | — |
| RBFE（`runrbfe.py`） | 独立于 ABFE 主线；契约、原子映射、受限 hybrid builder 已有，端到端的边还没有 | — |
| 口袋水采样器（`tools/pocket_water/`） | 实验性工具，未接入管线 | — |
| 膜体系、charge-transfer、DEXP / MACE / ORB 路线 | 有代码，不在默认生产路径上 | — |

## 6. 目录结构

发布方式是 **clone-and-run**：`git clone` 之后直接 `python runabfe.py …`，不打包。
`pytest tests/test_fresh_clone_imports.py` 检查 clone 下来能否 import。

ABFE 主线：

| 文件 | 作用 |
|---|---|
| `runabfe.py` | ABFE 命令行入口（`doctor` / `validate-config` / `config-template` 也在这） |
| `abfe_core.py` | 体系构建与底层物理组件 |
| `abfe_pipeline.py` | 阶段编排、质量门、续跑、结果落盘 |
| `abfe_preoptimizer.py` | λ 路径与窗口预优化；stage 2 分窗与控制器 |
| `ibs_engine.py` | IBS、MBAR/TMBAR、Boresch、长程修正 |
| `free_energy_engine.py` | ABFE / RBFE 共用的采样引擎 |
| `abfe_diagnostics.py` | 只读诊断子命令的实现 |
| `stage2_rounds.py`、`stage2_rounds_evidence.py` | AIBS 轮次控制器（纯函数）与逐窗证据（segment deletion、pooled ESS、top-1% 权重） |
| `stage2_online_ibs.py` | reference IBS 对照臂：偏移在生产中在线学习、时间聚合 MBAR（不配置就不启用） |
| `step_guard.py`、`lambda_path_versions.py`、`multi_segment_analysis.py` | 入口依赖的小支撑模块 |
| `apbs_correction.py` | Rocklin 有限尺寸静电修正 |
| `outer_lambda_neural_basis.py` | outer-λ 基势（`local_residual.openmm_plugin` 的依赖） |
| `stage2_ab_report.py` | 跑完后的 stage 2 报告 / 两次运行 A/B 对比 |

RBFE：`runrbfe.py`、`rbfe_core.py`、`rbfe_pipeline.py`。

| 目录 | 内容 |
|---|---|
| `local_residual/` | outer-λ 局部残差：运行时加载、重训与训练栈 |
| `plugins/LocalManyBodyResidual/` | 原生 OpenMM 插件源码、构建脚本 `g0_build.sh`、预编译 `.so`（`build/`，对着 `openmm=8.5.2` 编；CUDA kernel 运行时编译，不绑显卡架构） |
| `abfe_scripts/`、`exp012_xed/` | 局部残差链用的离线训练 / 导出脚本 |
| `tools/` | 诊断、验证与重训工具，不是生产入口 |
| `tests/` | 自动化测试（`./tests/run_offline_tests.sh`） |
| `docs/` | 使用文档 |

## 7. 改代码后的最低检查

```bash
./tests/run_offline_tests.sh                                       # 全部（排除 needs_gpu）
./tests/run_offline_tests.sh tests/test_core_physics_numerics.py   # 单个文件
```

测试通过证明的是**软件契约**，不证明新的**科学结果**。

## 8. 文档

**[docs/README.md](docs/README.md) 是文档导航。** `docs/` 下的教程是中文的；
英文 [README.md](README.md) 覆盖完整流程。

## 许可

[MIT License](LICENSE)，Copyright (c) 2026 Ruigeng Ji。

第三方署名与合规说明见 [NOTICE](NOTICE)。**OpenMM 是双授权的**：public API、reference
platform、CPU platform 和 application layer 走 MIT，CUDA、HIP、OpenCL platform 走 LGPL；
`plugins/LocalManyBodyResidual/platforms/cuda/` 建在后者之上。
