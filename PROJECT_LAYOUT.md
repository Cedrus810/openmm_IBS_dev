# 项目目录导航

本仓库是 ABFE-IBS 的**工程区分支**，面向 release：只保留生产代码、生产回归测试、
诊断工具和使用文档。开发期的实验代码、一次性脚本和过程文档不在这里
（原文在 `Atenolol-rank11`，登记在 [docs/HISTORY_LOG.md](docs/HISTORY_LOG.md)）。

## 从哪里开始

- `runabfe.py`：主要命令行入口。
- `abfe_config.json`：当前示例/生产配置。
- `abfe_core.py`、`abfe_pipeline.py`、`abfe_preoptimizer.py`、`ibs_engine.py`：
  核心实现。暂时保留在仓库根目录，以维持现有导入和生产命令兼容。
- `abfe_diagnostics.py`：`runabfe.py doctor` / `validate-config` / `config-template`
  三个只读诊断命令的实现。只读、不建 Context、不是启动硬门——改之前先读它的
  模块 docstring 里那三条约束。
- `outer_lambda_neural_basis.py`：`local_residual.openmm_plugin` 的依赖，
  生产启动路径的一部分。
- `local_residual/`：local-residual 路径势的**生产子集**（`openmm_plugin`、
  `em_no_residual`、`geometry`）。
- `tests/`：全部自动化测试与固定离线测试入口。
- `tools/`：人工诊断、修复、验证和绘图工具，不属于生产入口。
- `plugins/LocalManyBodyResidual/`：原生 OpenMM 插件源码。
- `docs/`：唯一文档集，见 [docs/README.md](docs/README.md)。
- `LICENSE`：MIT，Copyright (c) 2026 Ruigeng Ji。
- `NOTICE`：第三方组件署名。**OpenMM 是双授权的**——public API / reference /
  CPU platform / application layer 走 MIT，CUDA、HIP、OpenCL platform 走 LGPL，
  而 `plugins/LocalManyBodyResidual/platforms/cuda/` 正建在后者之上。改插件的
  链接方式或开始分发编译产物之前先读这份。

## 修改代码后的最低检查

在仓库根目录执行：

```bash
./tests/run_offline_tests.sh
```

只运行一个测试文件：

```bash
./tests/run_offline_tests.sh tests/test_core_physics_numerics.py
```

## 维护规则

1. 新的自动化测试只放在 `tests/`。
2. 临时诊断脚本放入 `tools/diagnostics/`；可重复使用的修复脚本放入
   `tools/repairs/`；画图脚本放入 `tools/plots/`；验证脚本放入 `tools/validation/`。
3. 不要在代码整理中移动已有计算输入或结果。新诊断应通过显式 `--out`/`--output`
   参数写入对应运行目录或用户指定位置。
4. 不得使用"副本""bak""pre_patch"等文件充当当前实现。旧源码副本留在
   `Atenolol-rank11`，不进本分支。
5. **`local_residual/__init__.py` 必须保持空的（不做 `import *` 再导出）。**
   它原本用 `from .softlift import *` 等一串再导出，导致 `runabfe` 一 import
   `local_residual.openmm_plugin` 就把整个 EXP-012 研究栈拖进生产启动路径。
   要用哪个模块就显式 import 哪个。
6. 一次性实验脚本（`exp0XX_*`）不进本仓库。它们属于实验工作区。
7. **residual sampling（`outer_lambda_local_residual_ibs`）不随首发。** 加载器
   `local_residual/openmm_plugin.py` 和原生 CUDA 插件仍在仓库里、仍可导入，
   但冻结 R1 模型资源和训练/部署栈都不分发：R1 是**按配体训练**的模型，
   只对 Atenolol 有效，换体系必须重训。开关默认 `false`；打开时 fail-closed
   并说明去 `Atenolol-rank11` 取什么。见 `docs/HISTORY_LOG.md`。
8. 核心模块后续若迁入 `src/`，必须单独进行，并先保证整套测试通过。
9. **`abfe_core.py` / `ibs_engine.py` 里 torch / openmmml / pymbar 必须保持惰性
   import。** 它们以前是模块级 eager import，让每个入口（`--help`、`self-test`、
   配置诊断）都白付 2.4 s。现在走 `has_orb()` / `has_pymbar()` /
   `_require_torch()` / `_require_pymbar()`；模块内部**不得**写裸 `HAS_ORB` /
   `HAS_PYMBAR`（模块级 `__getattr__` 只对外部属性访问生效，内部裸读会
   NameError）。由 `tests/test_cli_diagnostics_and_lazy_imports.py` 钉住。
