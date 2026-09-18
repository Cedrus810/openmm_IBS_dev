# ABFE-IBS：绝对结合自由能工作流

🌐 [**English (项目首页) →**](README.md)　·　[文档导航](docs/README.md)　·　[目录结构](PROJECT_LAYOUT.md)

ABFE-IBS 是一个基于 OpenMM 的绝对结合自由能（ABFE）工作流。它读取 GROMACS
`.gro/.top` 体系，分别计算配体在蛋白复合物和体相溶剂中的去耦自由能，并用 IBS
采样、MBAR/TMBAR、Boresch 约束账本和显式长程修正组合完整热力学循环：

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

管线覆盖 GROMACS→OpenMM 建系、complex/solvent 两腿、dual-lambda 解耦、
IBS 预热与固定偏置 production、MBAR/TMBAR 估计、Boresch attachment/release 记账、
LJ 长程修正、缓存、续跑和 fail-closed 质量门。
每一步实现的是哪篇文献的方法见 [docs/METHODS.md](docs/METHODS.md)。

仓库里还有一条 **RBFE（相对结合自由能）** 线，入口是 `runrbfe.py`，独立于 ABFE 主线。

> ## ⚠️ 引用任何数字之前
>
> **本仓库目前没有可以作为最终结论引用的结果。** 主线体系（4W53 / T4 lysozyme
> L99A + toluene）、结果登记表、协议版本、开放问题和有效性判据**全部**登记在
> [docs/STATUS.md](docs/STATUS.md) —— 那是本仓库唯一声明科学状态的文档，
> 本文和其它任何地方都不再复制它的表。
>
> **文件名里有 `final` 不代表结果可以引用。** 判据在同一份里。

## 1. 环境

核心依赖：**Python 3.12**、OpenMM、NumPy、SciPy、MDTraj、PyMBAR。GPU 生产运行还需要
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
不要未经审阅就当成新体系模板 —— 用 `config-template` 生成，或读
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

续跑 / 只分析已有能量和 checkpoint（不跑动力学）：

```bash
python runabfe.py --config abfe_config.json --ligand MOL --resume
python runabfe.py --config abfe_config.json --ligand MOL --analyze-only
```

用 `--resume` / `--reset` / `--analyze-only` 前**必须**先读
[输出与续跑](docs/OUTPUTS_AND_RESUME.md)。不要对受保护的历史结果目录执行 `--reset`。

## 4. 读结果

结果落在 `--output` 目录：`final_binding_results.json` 是汇总，
`run_provenance.json` 记录协议身份和输入指纹，`checkpoints/` 是续跑依据。
符号约定、每一项的口径和续跑语义见
[OUTPUTS_AND_RESUME.md](docs/OUTPUTS_AND_RESUME.md)；能不能引用见上面那条。

## 5. 改代码后的最低验证

```bash
./tests/run_offline_tests.sh                                       # 全部（排除 needs_gpu）
./tests/run_offline_tests.sh tests/test_core_physics_numerics.py   # 单个文件
```

测试通过证明**软件契约**成立，不自动证明新的**科学结果**已验证。
完整的三档验证与「改完该更新哪份文档」见 [MAINTAINING.md](docs/MAINTAINING.md)。

## 6. 从哪里开始读代码

- `runabfe.py` —— ABFE 主命令入口（`doctor` / `validate-config` / `config-template` 也在这）
- `runrbfe.py` —— RBFE 命令入口
- `abfe_config.json` —— 参考配置，**不是新体系模板**

逐个模块干什么、哪些是"缺一个 clone 就跑不起来"的支撑模块、文件往哪放，
**全在 [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md)** —— 那是仓库结构的唯一出处，
本文不再复制一份目录树。

## 7. 文档

**[docs/README.md](docs/README.md) 是完整文档地图。** 最常用的四个入口：

| 目标 | 入口 |
|---|---|
| 当前科学状态、结果能不能引用、协议版本 | [docs/STATUS.md](docs/STATUS.md) |
| 安装、输入与首次运行 | [GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| 还欠什么活（带 `[P1]`/`[P2]`/`[P3]` 等级） | [docs/TODO.md](docs/TODO.md) |
| 排障 | [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |

## 8. 这个仓库是什么

**工程区分支**：只有工作流源码、生产回归测试、诊断工具和使用文档。参考体系的
`output*` / 轨迹 / checkpoint、失败实验记录和逐条决策历史都在 `Atenolol-rank11`
工作区，本仓库只保留[历史材料索引](docs/HISTORY_LOG.md)。

**发布定位是 clone-and-run**：`git clone` 之后直接 `python runabfe.py …`，不打包。
冻结的 R1 资源与预编译插件 `.so` 都随仓库分发，不用编、不用另取。判据是
`pytest tests/test_fresh_clone_imports.py`。细节见 [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md)。

**`docs/` 下的教程以中文为准** —— 为开发速度做的既定选择，不是疏漏。
仓库首页 [README.md](README.md) 是英文版，覆盖同样的完整流程；
两份**不做逐字对等**（见 [docs/TODO_P3.md](docs/TODO_P3.md)）。

## 许可

[MIT License](LICENSE)，Copyright (c) 2026 Ruigeng Ji。

第三方组件署名与合规依据见 [NOTICE](NOTICE) —— 注意 **OpenMM 是双授权的**：
public API / reference / CPU platform / application layer 是 MIT，
CUDA、HIP、OpenCL platform 是 LGPL。本仓库不 vendor 第三方源码，也不分发第三方二进制。
