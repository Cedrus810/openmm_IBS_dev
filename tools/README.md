# 诊断与验证工具

[项目入口](../README.md) · [排障指南](../docs/TROUBLESHOOTING.md) · [输出与续跑](../docs/OUTPUTS_AND_RESUME.md)

`tools/` 中的程序是维护工具，不是生产工作流入口，也不应被核心模块导入。

## 目录职责

- `diagnostics/`：只读分析、质量检查和问题定位；
- `validation/`：验收脚本，产出可归档的机器可读结论；
- `pocket_water/`：口袋水采样器（实验性，不在默认生产路径上）；
- `refit_outer_residual_from_stage2.py`：outer-λ 局部残差重训，见 [RETRAIN_LOCAL_RESIDUAL.md](../docs/RETRAIN_LOCAL_RESIDUAL.md)；
- `_run_dir.py`：共享的"默认运行目录"解析，见下。

## 默认运行目录

不显式给目录时，`diagnostics/` 下的工具统一走 `tools/_run_dir.py`：

```text
ABFE_OUTPUT_DIR 环境变量  ->  abfe_config.json 的 "output" 键  ->  ./output
```

和 `runabfe.py --output` 是同一个真源。**显式传的参数永远优先**，
这些工具对目标目录的读写语义不受影响。

新增工具需要默认目录时用：

```python
from _run_dir import DEFAULT_RUN_DIR      # 导入时求值，够 argparse default= 用
from _run_dir import default_run_dir      # 需要运行时动态解析时用
```

不要往工具里硬编码具体运行目录名。

## 安全使用

1. 从仓库根目录执行，并显式传入输入和输出路径。
2. 先运行 `--help` 或阅读脚本头部，确认它是否只读。
3. 对已完成运行的受保护输出目录，诊断写 sidecar；不要原地补字段。
4. 任何原地改状态的脚本都应记录修改前 hash、修改内容、输出路径和恢复方式——
   这类脚本一次性用完即移出主线。

## 新增工具

可复用的诊断进入 `diagnostics/`；验收脚本进入 `validation/`。
临时一次性脚本不要放进生产模块。工具应接受参数、避免硬编码体系路径，并在危险操作前
fail closed。事故关闭时一并把它的诊断脚本移出。

