# ABFE-IBS：绝对结合自由能工作流

[English](README_en.md) · [简明入口](README.md) · [完整文档](docs/README.md)

ABFE-IBS 是一个基于 OpenMM 的绝对结合自由能（ABFE）工作流。它读取 GROMACS
`.gro/.top` 体系，分别计算配体在蛋白复合物和体相溶剂中的去耦自由能，并用 IBS
采样、MBAR/TMBAR、Boresch 约束账本和显式长程修正组合完整热力学循环：

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

方法学依据与逐项文献见 [docs/METHODS.md](docs/METHODS.md)。

管线覆盖 GROMACS→OpenMM 建系、complex/solvent 两腿、dual-lambda 解耦、
IBS 预热与固定偏置 production、MBAR/TMBAR 估计、Boresch attachment/release 记账、
LJ 长程修正、缓存、续跑和 fail-closed 质量门。

> **当前科学状态见 [docs/STATUS.md](docs/STATUS.md)。** 本仓库目前**没有**可以作为
> 最终结论引用的结果；主线体系、结果登记、协议版本和开放问题全部登记在那一份，
> 引用任何数字前先读它。

## 1. 环境

核心依赖：Python 3.10+、OpenMM、NumPy、SciPy、MDTraj、PyMBAR。GPU 生产运行还需要
匹配的 CUDA 或 OpenCL。仓库提供 `environment.yml`，但其中包含 CUDA 版本和原机器的
环境选择，**安装到其他主机前必须审阅**。

```bash
python -c "import openmm, numpy, scipy, mdtraj, pymbar; print(openmm.__version__)"
```

如果 `python runabfe.py --help` 在导入阶段报 `No module named 'openmm'`，
说明当前 shell 没进可运行环境，不是命令参数写错了。

自检与配置诊断（只读，不建 Context）：

```bash
python runabfe.py doctor
python runabfe.py validate-config --config abfe_config.json
python runabfe.py config-template --out my_system.json
```

## 2. 输入

首次建系通常需要：

| 参数 | 内容 |
|---|---|
| `--gro` | GROMACS 坐标文件 |
| `--top` + `--gmx-path` | GROMACS 拓扑及其 include 依赖（`--gmx-path` 给 GROMACS 安装前缀） |
| `--ligand` | 配体残基名 |
| `--output` | 独立的**新**输出目录 |

`abfe_config.json` 是参考配置，含机器相关的 `gmx_path` 和针对历史运行冻结的选项。
不要未经审阅就当成新体系模板——用 `config-template` 生成，或读
[迁移教程](docs/MIGRATING_TO_A_NEW_SYSTEM.md)。

**不要把任何体系的 checkpoint 复用到另一个体系。**

## 3. 运行

新体系，显式输入：

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

只分析已有能量和 checkpoint（不跑动力学）：

```bash
python runabfe.py --config abfe_config.json --ligand MOL --analyze-only
```

用 `--resume` / `--reset` / `--analyze-only` 前**必须**先读
[输出与续跑](docs/OUTPUTS_AND_RESUME.md)。不要对受保护的历史结果目录执行 `--reset`。

## 4. 读结果

结果落在 `--output` 目录：`final_binding_results.json` 是汇总，
`run_provenance.json` 记录协议身份和输入指纹，`checkpoints/` 是续跑依据。
符号约定、每一项的口径和续跑语义见
[OUTPUTS_AND_RESUME.md](docs/OUTPUTS_AND_RESUME.md)。

**文件名里有 `final` 不代表结果可以引用。** 判据在
[docs/STATUS.md](docs/STATUS.md)。

## 5. 改代码后的最低验证

```bash
./tests/run_offline_tests.sh                                # 全部（排除 needs_gpu）
./tests/run_offline_tests.sh tests/test_core_physics_numerics.py   # 单个文件
```

测试通过证明**软件契约**成立，不自动证明新的**科学结果**已验证。
维护规则见 [MAINTAINING.md](docs/MAINTAINING.md) 和 [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md)。

## 6. 仓库地图

```text
runabfe.py                  主命令入口
abfe_core.py                系统与底层物理组件
abfe_pipeline.py            阶段编排、质量门、resume 和结果落盘
ibs_engine.py               IBS、MBAR/TMBAR、Boresch、LRC 核心
abfe_preoptimizer.py        lambda 路径和窗口预优化
abfe_diagnostics.py         doctor / validate-config / config-template
local_residual/             local-residual 路径势的生产子集
tests/                      回归与协议测试
tools/                      诊断、显式修复和绘图（非生产入口）
plugins/                    原生 OpenMM 插件源码
docs/                       唯一文档集
```

逐条说明见 [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md)。

## 7. 文档

| 目标 | 入口 |
|---|---|
| 当前科学状态、结果登记、协议版本 | [docs/STATUS.md](docs/STATUS.md) |
| 方法学依据与文献引用 | [docs/METHODS.md](docs/METHODS.md) |
| 安装、输入与命令 | [GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| 输出结构、符号、续跑 | [OUTPUTS_AND_RESUME.md](docs/OUTPUTS_AND_RESUME.md) |
| 排障 | [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |
| 迁移到其他体系 | [MIGRATING_TO_A_NEW_SYSTEM.md](docs/MIGRATING_TO_A_NEW_SYSTEM.md) |
| 改代码和验证 | [MAINTAINING.md](docs/MAINTAINING.md) |
| 当前行动 | [docs/TODO.md](docs/TODO.md) |
| 历史材料索引 | [docs/HISTORY_LOG.md](docs/HISTORY_LOG.md) |
| 完整文档地图 | [docs/README.md](docs/README.md) |

本仓库是 ABFE-IBS 的**工程区分支**：只有工作流源码、生产回归测试、诊断工具和使用
文档。参考体系的 `output*` / 轨迹 / checkpoint、开发期实验脚本（`exp0XX_*`）、
失败实验记录和逐条决策历史都在 `Atenolol-rank11` 工作区，本仓库只保留
[历史材料索引](docs/HISTORY_LOG.md)。

## 许可

[MIT License](LICENSE)，Copyright (c) 2026 Ruigeng Ji。

第三方组件署名与合规依据见 [NOTICE](NOTICE)——注意 **OpenMM 是双授权的**：
public API / reference / CPU platform / application layer 是 MIT，
CUDA、HIP、OpenCL platform 是 LGPL。本仓库不 vendor 第三方源码，也不分发第三方二进制。
