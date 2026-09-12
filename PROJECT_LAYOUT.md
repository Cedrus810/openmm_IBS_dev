# 项目目录导航

本仓库是 ABFE-IBS 的**工程区分支**，面向 release：只保留生产代码、生产回归测试、
诊断工具和使用文档。开发期的原始轨迹、结果 artifact 和过程文档不在这里
（原文在 `Atenolol-rank11`，登记在 [docs/HISTORY_LOG.md](docs/HISTORY_LOG.md)）。

**发布定位是 clone-and-run**：`git clone` 之后直接 `python runabfe.py …`，不打包。
`pyproject.toml` 只承担 linter 配置，没有 `[project]` / `[build-system]`。
判据是 `pytest tests/test_fresh_clone_imports.py` —— 它绿了才说明 clone 下来 import 得动。
预编译插件 `.so`、冻结的 R1 资源都**随仓库分发**，clone 下来不用编、不用另取。

项目处于**开发最末期**，不是在筹备首发；还欠什么按 [docs/TODO.md](docs/TODO.md) 走。

## 从哪里开始

- `runabfe.py`：**ABFE 主命令行入口**（`doctor` / `validate-config` / `config-template` 子命令也在这）。
- `runrbfe.py`：**RBFE 命令行入口**（相对结合自由能，独立于 ABFE 主线）。
- `abfe_config.json`：参考配置。含**机器本地**的 `gmx_path` 和为历史运行冻结的选项，
  **不是新体系模板**——用 `runabfe.py config-template` 生成。

## 顶层模块

ABFE 主线：

| 文件 | 作用 |
|---|---|
| `abfe_core.py` | 体系构建与底层物理组件 |
| `abfe_pipeline.py` | 阶段编排、质量门、resume、结果落盘 |
| `abfe_preoptimizer.py` | λ 路径与窗口预优化；Stage-2 分窗与修补控制器 |
| `ibs_engine.py` | IBS、MBAR/TMBAR、Boresch、LRC 核心 |
| `free_energy_engine.py` | ABFE / RBFE **共用**的自由能采样引擎 |
| `abfe_diagnostics.py` | `doctor` / `validate-config` / `config-template` 的实现。**只读、不建 Context、不是启动硬门**——改之前先读它的模块 docstring 里那三条约束 |

支撑模块（体积小但**缺一个 clone 就跑不起来**，见 `tests/test_fresh_clone_imports.py`）：

| 文件 | 作用 | 引用方式 |
|---|---|---|
| `step_guard.py` | OpenMM 积分步进的统一异常出口 | **模块级** import ×3，缺了它四个入口模块全部 `ModuleNotFoundError` |
| `lambda_path_versions.py` | Stage-2 λ 路径的版本记录（路径演化时不丢已完成的工作） | 惰性 ×6 |
| `multi_segment_analysis.py` | 多采样段分析适配层：同一窗口的多段采样**相加**而不是替换 | 惰性 ×5 |
| `apbs_correction.py` | Rocklin 有限尺寸静电修正（膜 ABFE 用） | 惰性；当前不在主线路径上 |
| `outer_lambda_neural_basis.py` | 外层 λ 神经基势 | `local_residual.openmm_plugin` 的依赖，属于生产启动路径 |

RBFE 线（独立，不影响 ABFE 主线）：`rbfe_core.py`（数据契约、输入验证、ΔΔG 汇总）、
`rbfe_pipeline.py`（运行目录与身份、两腿、独立重复、边网络、续跑校验）。

## 目录

| 目录 | 内容 |
|---|---|
| `local_residual/` | local-residual 路径势。运行时加载那条是**自足**的（`openmm_plugin`、`em_no_residual`、`geometry`）；闭式重训还要 `softlift`、`softlift_dataset`、`refit` |
| `resources/outer_lambda_local_residual/` | 冻结的 R1 模型资源（manifest + payload + 权重）。**2026-09-12 起随仓库分发** |
| `abfe_scripts/` | 离线脚本：local-residual 的训练 / 导出 / manifest 生成 |
| `exp012_xed/` | EXP-012 独立研究代码。**二阶依赖**：只有 `local_residual/{ledger_audit,metrics,mm_ledger,schema}.py` 用 `import *` 吃它，闭式重训链不碰 |
| `tests/` | 全部自动化测试与固定离线测试入口 |
| `tools/` | 人工诊断、修复、验证和绘图工具，**不是生产入口** |
| `plugins/LocalManyBodyResidual/` | 原生 OpenMM 插件：34 个源码文件 + `g0_build.sh` + **随仓库分发的预编译 `.so`**（对着 `environment.yml` 钉的 `openmm=8.5.2` 编的；`build` 是指向 `build_exp026_a2` 的符号链接，两者都已跟踪）。按环境文件建环境**不用编** |
| `docs/` | 唯一文档集，见 [docs/README.md](docs/README.md) |

`LICENSE`：MIT，Copyright (c) 2026 Ruigeng Ji。
`NOTICE`：第三方组件署名。**OpenMM 是双授权的**——public API / reference / CPU platform /
application layer 走 MIT，CUDA、HIP、OpenCL platform 走 LGPL，而
`plugins/LocalManyBodyResidual/platforms/cuda/` 正建在后者之上。改插件的链接方式或
开始分发编译产物之前先读这份。

## 修改代码后的最低检查

在仓库根目录执行：

```bash
./tests/run_offline_tests.sh                                       # 全部（排除 needs_gpu）
./tests/run_offline_tests.sh tests/test_core_physics_numerics.py   # 单个文件
```

测试通过证明**软件契约**成立，不自动证明新的**科学结果**已验证。

## 维护规则

1. 新的自动化测试只放在 `tests/`。
2. 临时诊断脚本放入 `tools/diagnostics/`；验证脚本放入 `tools/validation/`。
   事故结案后诊断脚本应随事故一起移出（见 [docs/MAINTAINING.md](docs/MAINTAINING.md)）。
3. 不要在代码整理中移动已有计算输入或结果。新诊断应通过显式 `--out`/`--output`
   参数写入对应运行目录或用户指定位置。
4. 不得使用"副本""bak""pre_patch"等文件充当当前实现。旧源码副本留在
   `Atenolol-rank11`，不进本分支。
5. **`local_residual/__init__.py` 必须保持空的（不做 `import *` 再导出）。**
   它原本用 `from .softlift import *` 等一串再导出，导致 `runabfe` 一 import
   `local_residual.openmm_plugin` 就把整个 EXP-012 研究栈拖进生产启动路径。
   要用哪个模块就显式 import 哪个。
6. **顶层目录名要先查有没有撞车。** `mace_torch` 装了一个顶层正规包也叫 `scripts`，
   而本仓的原来只是 PEP 420 namespace 目录 ⟹ **正规包永远赢**，闭式重训里那句
   `from scripts.… import main` 在任何装了 MACE 的环境里必炸。
   2026-09-12 改名 `abfe_scripts/` 并加了 `__init__.py`。
   （`tests` 也撞 `mace_torch`，但没有任何 `from tests.` 用法、pytest 不走包名，属潜伏、未改。）
7. **`abfe_core.py` / `ibs_engine.py` 里 torch / openmmml / pymbar 必须保持惰性 import。**
   它们以前是模块级 eager import，让每个入口（`--help`、`self-test`、配置诊断）都白付 2.4 s。
   现在走 `has_orb()` / `has_pymbar()` / `_require_torch()` / `_require_pymbar()`；
   模块内部**不得**写裸 `HAS_ORB` / `HAS_PYMBAR`（模块级 `__getattr__` 只对外部属性访问
   生效，内部裸读会 `NameError`）。由 `tests/test_cli_diagnostics_and_lazy_imports.py` 钉住。
8. 核心模块后续若迁入 `src/`，必须单独进行，并先保证整套测试通过。

## 两条已翻转的旧规则（别按老印象办）

| 旧规则 | 现状 |
|---|---|
| ~~「residual sampling 不随发布」（2026-08-31）~~ | **2026-09-12 翻转**：当时不发是因为「R1 只能跑 Atenolol、永远没法换体系」，EXP-033 P1 把换配体重训压成一次闭式求解后这条前提没了。`resources/outer_lambda_local_residual/` 与预编译 `.so` 都随仓库分发，开关仍默认 `false` |
| ~~「一次性实验脚本（`exp0XX_*`）不进本仓库」~~ | 部分翻转：`exp012_xed/` 与 `abfe_scripts/` 里的 `exp0XX_*` 已跟踪，因为闭式重训链和 `local_residual` 的 ledger 依赖它们。**新的**一次性实验脚本仍然不进本仓 |
