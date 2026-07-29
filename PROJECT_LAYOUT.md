# 项目目录导航

本仓库以“生产代码、测试、诊断工具、结论证据、历史材料互不混放”为原则。

## 从哪里开始

- `runabfe.py`：主要命令行入口。
- `abfe_config.json`：当前示例/生产配置。
- `abfe_core.py`、`abfe_pipeline.py`、`abfe_preoptimizer.py`、`ibs_engine.py`：
  核心实现。第一轮整理暂时保留在仓库根目录，以维持现有导入和生产命令兼容。
- `tests/`：全部自动化测试与固定离线测试入口。
- `tools/`：人工诊断、修复和绘图工具，不属于生产入口。
- `scripts/`：PBS 与实验运行脚本。
- `reports/project/`：维护记录和阶段性结论文档；已有计算结果保持原位置，不在代码整理中迁移。
- `references/`：论文与外部参考资料。
- `archive/`：旧补丁、源码备份和历史材料；生产代码不得从这里导入。
- `docs/`：协议与维护文档。
- `output_lrc_fix/`：当前 LRC-fix 验收基线。整理期间保持原位，不移动、不改写。

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
   `tools/repairs/`；画图脚本放入 `tools/plots/`。
3. 不要在代码整理中移动已有计算输入或结果。新诊断应通过显式`--out`/`--output` 参数写入对应运行目录或用户指定位置。
4. 旧源码副本和补丁只放入 `archive/`，不得使用“副本”“bak”“pre_patch”等文件
   充当当前实现。
5. `output_lrc_fix/` 是结果证据，不是源码目录。修改算法后应使用它做回归核对，
   不应在其中直接修补代码。
6. 核心模块后续若迁入 `src/`，必须单独进行，并先保证整套测试通过；本轮整理不
   改变核心模块的导入位置。
