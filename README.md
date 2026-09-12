# ABFE-IBS

### 🌐 [**English README →**](README_en.md)　|　中文说明见下方，完整版：[README_cn.md](README_cn.md)　|　[文档导航](docs/README.md)

ABFE-IBS 是一个基于 OpenMM、面向 GROMACS 输入的绝对结合自由能（ABFE）工作流。
它读取 `.gro/.top` 体系，分别计算配体在蛋白复合物和体相溶剂中的去耦自由能，
用 IBS 采样、MBAR/TMBAR、Boresch 约束账本和显式长程修正组合完整热力学循环。

```bash
python runabfe.py --config abfe_config.json \
  --gro system.gro --top topol.top --ligand LIG \
  --gmx-path /path/to/gromacs --output ./output_new_system \
  --boresch --boresch-source simple
```

**从这里开始**

- 完整说明：[中文 README](README_cn.md) / [English README](README_en.md)
- 安装、输入与运行：[GETTING_STARTED.md](docs/GETTING_STARTED.md)
- 输出、符号与续跑：[OUTPUTS_AND_RESUME.md](docs/OUTPUTS_AND_RESUME.md)
- 迁移到新体系：[MIGRATING_TO_A_NEW_SYSTEM.md](docs/MIGRATING_TO_A_NEW_SYSTEM.md)
- 还欠什么活：[docs/TODO.md](docs/TODO.md)（唯一待办清单）
- 目录结构：[PROJECT_LAYOUT.md](PROJECT_LAYOUT.md) · 完整文档地图：[docs/README.md](docs/README.md)

仓库里还有一条 **RBFE（相对结合自由能）** 线，入口是 `runrbfe.py`，独立于 ABFE 主线。

修改代码后的最低 CPU 检查：

```bash
./tests/run_offline_tests.sh
```

**当前状态** —— 主线体系是 4W53（T4 lysozyme L99A + toluene）。
**本仓库目前没有可以作为最终结论引用的结果。** 现有数字、协议版本、开放问题和
有效性判据统一登记在 [docs/STATUS.md](docs/STATUS.md)，引用任何数字前先读那一份。

本仓库是**工程区分支**：只保留发布所需的生产代码、生产回归测试和使用文档。
原始轨迹与结果 artifact 在 `Atenolol-rank11` 工作区，本仓库只保留一份
[历史材料 log](docs/HISTORY_LOG.md) 做索引。

**发布定位是 clone-and-run**：`git clone` 之后直接跑，不打包。冻结的 R1 资源与
预编译插件 `.so` 都随仓库分发，不用编、不用另取。判据是
`pytest tests/test_fresh_clone_imports.py` —— clone 下来 import 得动、跑得动。

**许可** —— [MIT License](LICENSE)，Copyright (c) 2026 Ruigeng Ji。第三方组件署名与
合规依据见 [NOTICE](NOTICE)——注意 **OpenMM 是双授权的**：public API / reference /
CPU platform / application layer 是 MIT，而 CUDA、HIP、OpenCL platform 是 LGPL。
本仓库不 vendor 任何第三方源码，也不随包分发任何第三方二进制。
