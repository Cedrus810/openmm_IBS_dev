# ABFE-IBS：绝对结合自由能工作流

[English](README_en.md) · [简明入口](README.md) · [完整文档](docs/README.md)

ABFE-IBS 是一个基于 OpenMM 的绝对结合自由能（ABFE）研究与生产工作流。它读取 GROMACS
`.gro/.top` 体系，分别计算配体在蛋白复合物和体相溶剂中的去耦自由能，并用 IBS、
MBAR/TMBAR、Boresch 约束账本和显式长程修正组合完整热力学循环。

本仓库是 ABFE-IBS 的**工程区分支**，面向 release，只包含：

- 可迁移到其他蛋白–配体体系的工作流源码（`runabfe.py` 等，见
  [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md)）；
- 生产回归测试（`tests/`）与人工诊断工具（`tools/`）；
- 使用文档（`docs/`，见[文档导航](docs/README.md)）。

**不在这里**：参考体系的 `output*` / 验证轨迹 / checkpoint、开发期实验脚本
（`exp0XX_*`）、失败实验记录和逐条决策历史。原文全部在 `Atenolol-rank11`
工作区，本仓库只保留一份[历史材料索引](docs/HISTORY_LOG.md)。

要迁移到新的蛋白–配体体系，读[迁移教程](docs/MIGRATING_TO_A_NEW_SYSTEM.md)；
**不要**把任何体系的 checkpoint 复用到另一个体系。

## 当前科学状态（证据截至 2026-09-02）

生产主线的软件骨架已经形成，包括 GROMACS→OpenMM 系统构建、complex/solvent 两腿、
dual-lambda 解耦、IBS 预热与固定偏置 production、MBAR/TMBAR、Boresch
attachment/release、LJ 长程修正、缓存、续跑和 fail-closed 检查。

**当前主线体系是 4W53（T4 lysozyme L99A + toluene），不再是 Atenolol。**

当前代码里的主要协议身份（直接读自源码常量，非转述）：

| 协议 | 常量 | 值 |
|---|---|---|
| IBS 偏置 | `ibs_engine.IBS_BIAS_PROTOCOL_VERSION` | **32** |
| 热力学路径 | `abfe_preoptimizer.THERMODYNAMIC_PATH_PROTOCOL_VERSION` | 21 |
| LJ 长程修正 | `ibs_engine.TRADITIONAL_LJ_LRC_PROTOCOL_VERSION` | 3 |
| WCA 记账 | `ibs_engine.WCA_ACCOUNTING_VERSION` | **3**（`WCA_SHIELD_RETIRED = True`） |
| ESS 门 | `ibs_engine.ESS_GATE_PROTOCOL_VERSION` | **5** |
| 配体 COM 约束 | `ibs_engine.LIGAND_COM_RESTRAINT_PROTOCOL_VERSION` | 2 |

### 结果登记

| 数值 | 体系 / 运行 | 状态 | 能否作为最终结论引用 |
|---|---|---|---|
| **`−21.36 ± 0.93 kJ/mol`**（`−5.11 ± 0.22 kcal/mol`） | 4W53，`output_v3_seed20260908`，2026-09-02 | **注册标签待维护者指定** | **否：单 seed（`20260908`），本仓库内无第二个独立重复** |
| `−23.1622 ± 2.5139 kJ/mol`（`output_lrc_fix`） | Atenolol-rank11 | **已作废（2026-08-24 判定）** | 否 |
| `+40.8362 ± 1.3178 kJ/mol`（旧 `output`） | Atenolol-rank11 | `INVALIDATED` | 否：旧符号约定与当前约定相反，且有诊断问题 |
| `+16.00 ± 2.20 kJ/mol`（2026-07-27） | Atenolol-rank11 | `INVALIDATED` | 否：陈旧且错误的 Boresch 平衡几何 |

4W53 那一行的对照与全部证据：实验值 **−23.10 kJ/mol**（`−5.52 ± 0.04 kcal/mol`），
差 **0.41 kcal/mol、1.83σ 内**；质量门同时转健康（溶剂腿 stage2 raw ESS
2.93 → 173.33，top1% 0.828 → 0.047）。逐项见
[docs/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md](docs/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md)。

**仍开放（不阻塞，但必须随数字一起说）**：溶剂腿 stage2 是唯一有独立参考真值的一项，
实测 ≈ **−8.3**，真值 **−6.58 ± 0.26**（no-LRC 口径，见
[docs/reference_data/README.md](docs/reference_data/README.md)），差 1.7~4.2 kJ/mol
尚未归因。独立重复、随机种子账本和时间相关不确定度**仍未闭合**。

当前符号约定：

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

文件名包含 `final` 不代表结果可以引用。**Atenolol 那三行**的原始 artifact、
结果索引和机器可读登记表（`RESULT_REGISTRY.csv`）都在 `Atenolol-rank11`
工作区，不在本工程区分支；4W53 那一行的证据在本仓库 `docs/` 里（上面已链）。

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

