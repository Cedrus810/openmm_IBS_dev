# 诊断与验证工具

[项目入口](../README.md) · [排障指南](../docs/TROUBLESHOOTING.md) · [输出与续跑](../docs/OUTPUTS_AND_RESUME.md)

`tools/` 中的程序是维护工具，不是生产工作流入口，也不应被核心模块导入。

## 目录职责

- `diagnostics/`：只读分析、质量检查和问题定位；
- `validation/`：验收脚本，产出可归档的机器可读结论；
- `_run_dir.py`：共享的"默认运行目录"解析，见下。

## 已结案脚本去哪了

事故结案 = 它的诊断脚本也结案：结论留在 `docs/`，脚本移出主线。
2026-09-07 发布清理移走了 24 个（window0 / WCA 壳 / ESS / Boresch 系综那批诊断、
`repairs/` 全部一次性 checkpoint 修补器、DEXP 退役后失去产物的 `plots/`），去向：

```text
/home/ruigengji/ABFE_IBS/Atenolol-rank11/archive/from_mainline_2026-09-07/
```

在 `docs/` 或代码注释里读到一个 `tools/` 下已不存在的脚本名，就去那里找，
清单和移出理由见该目录的 `README.md`。**不要放回主线。**

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

不要再往工具里硬编码具体运行目录名。2026-08-31 之前有 14 个工具各自把
`output_lrc_fix`（Atenolol-rank11 的 LRC-fix 验收基线目录，不在本工程区分支）
写死成默认值，分散在 `--run-dir` / `--output-dir` / `--output-root` / 位置参数
四种写法下，不带参数直接跑会拿到一个不存在的路径。

## 安全使用

1. 从仓库根目录执行，并显式传入输入和输出路径。
2. 先运行 `--help` 或阅读脚本头部，确认它是否只读。
3. 对已完成运行的受保护输出目录，诊断写 sidecar；不要原地补字段。
4. 任何原地改状态的脚本都应记录修改前 hash、修改内容、输出路径和恢复方式——
   这类脚本一次性用完即移出（见上）。

## 新增工具

可复用的诊断进入 `diagnostics/`；验收脚本进入 `validation/`。
临时一次性脚本不要放进生产模块。工具应接受参数、避免硬编码体系路径，并在危险操作前
fail closed。事故关闭时一并把它的诊断脚本移出。

