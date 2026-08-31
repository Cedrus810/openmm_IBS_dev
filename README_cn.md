# ABFE-IBS：绝对结合自由能工作流

[English](README_en.md) · [简明入口](README.md) · [完整文档](docs/README.md)

ABFE-IBS 是一个基于 OpenMM 的绝对结合自由能（ABFE）研究与生产工作流。它读取 GROMACS
`.gro/.top` 体系，分别计算配体在蛋白复合物和体相溶剂中的去耦自由能，并用 IBS、
MBAR/TMBAR、Boresch 约束账本和显式长程修正组合完整热力学循环。

本仓库同时包含：

- 可迁移到其他蛋白–配体体系的工作流源码、测试、教程和工具；
- Atenolol-rank11 参考体系；
- 生产候选、验证数据、失败实验和历史记录。

参考体系的 `output*`、`validation/`、`memtest/` 等目录是特定运行的证据，不是可直接
复用到新体系的 checkpoint。迁移前请阅读[迁移教程](docs/MIGRATING_TO_A_NEW_SYSTEM.md)。

## 当前科学状态（证据截至 2026-08-12）

生产主线的软件骨架已经形成，包括 GROMACS→OpenMM 系统构建、complex/solvent 两腿、
dual-lambda 解耦、IBS 预热与固定偏置 production、MBAR/TMBAR、Boresch
attachment/release、LJ 长程修正、缓存、续跑和 fail-closed 检查。

当前报告记录的主要协议身份为 IBS v29、thermodynamic path v21、LJ LRC v3、WCA v2。
但是独立重复、随机种子账本、完整验证矩阵和时间相关不确定度尚未闭合。因此当前适合报告
**软件与方法开发进展**，不适合宣称已获得最终可发表的 Atenolol 结合自由能。

| 数值 | 状态 | 能否作为最终结论引用 |
|---|---|---|
| `−23.1622 ± 2.5139 kJ/mol`（`output_lrc_fix`） | `CANDIDATE` | 否：缺独立重复，seed ledger 为空，Boresch `kr` 有裁剪 |
| `+40.8362 ± 1.3178 kJ/mol`（旧 `output`） | `INVALIDATED` | 否：旧符号约定与当前约定相反，且有诊断问题 |
| `+16.00 ± 2.20 kJ/mol`（2026-07-27） | `INVALIDATED` | 否：陈旧且错误的 Boresch 平衡几何 |

当前符号约定：

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

文件名包含 `final` 不代表结果可以引用。完整状态请以
`Atenolol-rank11` 工作区里的结果索引和机器可读结果登记表
（`RESULT_REGISTRY.csv`）为准——它们不在本工程区分支。

## 快速开始

### 1. 准备环境

核心依赖包括 Python 3.10+、OpenMM、NumPy、SciPy、MDTraj 和 PyMBAR；GPU 生产运行还需要
匹配的 CUDA 或 OpenCL。仓库提供 `environment.yml`，但其中包含 CUDA 版本和原机器环境
选择，安装到其他主机前必须审阅。

```bash
python -c "import openmm, numpy, scipy, mdtraj, pymbar; print(openmm.__version__)"
```

如果 `python runabfe.py --help` 在导入阶段报告 `No module named 'openmm'`，说明当前
shell 尚未进入可运行环境，而不是命令参数错误。

### 2. 准备输入

首次构建通常需要：

- GROMACS 坐标文件（`--gro`）；
- GROMACS 拓扑及其 include 依赖（`--top`、`--gmx-path`）；
- 配体残基名（`--ligand`）；
- 独立的新输出目录。

`abfe_config.json` 是 Atenolol 参考配置，包含机器相关的 `gmx_path` 和针对历史运行冻结的
选项。不要未经审阅就把它当成新体系模板。

### 3. 运行

续跑参考计算：

```bash
python runabfe.py --config abfe_config.json --ligand MOL --resume
```

使用显式输入和新输出目录：

```bash
python runabfe.py \
  --config abfe_config.json \
  --gro /path/to/system.gro \
  --top /path/to/topol.top \
  --ligand LIG \
  --gmx-path /path/to/gromacs/share/gromacs/top \
  --output ./output_new_system \
  --boresch --boresch-source simple
```

仅分析已有能量和 checkpoint：

```bash
python runabfe.py --config abfe_config.json --ligand MOL --analyze-only
```

使用 `--resume`、`--reset` 或 `--analyze-only` 前必须阅读
[输出与续跑](docs/OUTPUTS_AND_RESUME.md)。不要对受保护的历史结果目录随意执行 `--reset`。

## 最低验证

修改代码后，在仓库根目录运行：

```bash
./tests/run_offline_tests.sh
```

定向测试：

```bash
./tests/run_offline_tests.sh tests/test_core_physics_numerics.py
```

该入口默认排除标记为 `needs_gpu` 的测试。测试通过证明软件契约成立，不自动证明新的科学结果
已经验证。

## 阅读路线

| 目标 | 入口 |
|---|---|
| 历史材料索引 | [docs/HISTORY_LOG.md](docs/HISTORY_LOG.md) |
| 安装、输入与命令 | [GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| 输出结构、结果口径、续跑 | [OUTPUTS_AND_RESUME.md](docs/OUTPUTS_AND_RESUME.md) |
| 排障 | [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |
| 迁移到其他体系 | [MIGRATING_TO_A_NEW_SYSTEM.md](docs/MIGRATING_TO_A_NEW_SYSTEM.md) |
| 修改代码和验证 | [MAINTAINING.md](docs/MAINTAINING.md) |
| 当前综合进度与论文母稿 | `Atenolol-rank11` 的 2026-08-12 技术底稿（不在本分支） |
| 当前行动 | [docs/TODO.md](docs/TODO.md) |
| 数字冲突与有效性 | `Atenolol-rank11` 的 `CONFLICTS.md`（不在本分支） |

## 仓库地图

```text
runabfe.py                  主命令入口
abfe_core.py                系统与底层物理组件
abfe_pipeline.py            阶段编排、质量门、resume 和结果落盘
ibs_engine.py               IBS、MBAR/TMBAR、Boresch、LRC 核心
abfe_preoptimizer.py        lambda 路径和窗口预优化
tests/                      回归与协议测试
tools/                      诊断、显式修复和绘图
docs/                       唯一文档集：教程、协议、TODO 与历史 log
plugins/                    原生 OpenMM 插件源码
```

## 结果与数据安全

以下证据目录在整理期间不得原地改写、移动或删除：

```text
output/
output_lrc_fix/
output_lrc_fixonly-complex-charging/
validation/
solvent_box_scan/
memtest/output_membrane_100ns/
memtest/output_membrane_5ns/
```

新的算法或协议应写入新的输出目录。旧结果、失败路线和无效结论继续保留，以便复核协议演化和
bug 根因。详细政策见 `Atenolol-rank11` 的 `IMMUTABILITY_POLICY.md`。

## 研究分支与状态词

DEXP、MACE、ORB、outer-lambda 神经基势、膜体系和 charge-transfer 都有代码、计划或实验
记录，但不等于已进入生产主线：

- `IMPLEMENTED`：代码存在；
- `VALIDATED`：在明确输入、环境和验收门下通过；
- `CANDIDATE`：可继续验证但不可宣称最终；
- `FAILED/INVALIDATED`：保留证据但不得作为当前科学结论；
- `PLAN/DESIGN`：计划不是执行结果，也不是生产授权。

## 文档维护约定

- 稳定教程写入 `docs/`；体系与日期相关结论留在 `Atenolol-rank11`，
  本分支只在 [docs/HISTORY_LOG.md](docs/HISTORY_LOG.md) 登记它们存在过。
- 当前解释以源码和 sealed protocol 为准。
- 新数字必须记录来源 artifact、单位、符号、协议、有效性和是否可引用。
- 不通过改写旧报告来“修正历史”；使用登记表和替代关系保留历史。

