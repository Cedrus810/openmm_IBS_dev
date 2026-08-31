#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`tools/` 下诊断与修复脚本共用的"默认运行目录"解析。

## 为什么要有这个文件

2026-08-31 发布整理前，`tools/` 里有 14 个脚本各自把 `output_lrc_fix` 硬编码成
argparse 默认值（`--run-dir` / `--output-dir` / `--output-root` / 位置参数，
四种写法都有）。`output_lrc_fix` 是 Atenolol-rank11 那次 LRC-fix 验收基线的运行
目录，**不在本工程区分支里**——不带参数直接跑这些工具，会拿到一个不存在的路径，
报错还长得像"数据坏了"而不是"你没给目录"。

同一个默认值散在 14 处、四种参数名下，改一次要改 14 个地方，必然漏。这里收成一处。

## 解析顺序

1. 环境变量 `ABFE_OUTPUT_DIR`——临时切目录不用改命令行；
2. 仓库根 `abfe_config.json` 的 `"output"` 键——和 `runabfe.py --output` 同一个真源；
3. `./output`——`abfe_config.json` 自己的默认值，兜底。

**只影响"用户没显式给目录"这一种情况。** 显式传的参数永远优先，
这些工具对目标目录的读写语义一点没变。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: 所有解析都失败时的兜底值，与 `abfe_config.json` 里的 `"output"` 默认值一致。
FALLBACK_RUN_DIR = "./output"

#: 允许在不改命令行的前提下临时切换默认目录。
ENV_VAR = "ABFE_OUTPUT_DIR"

_REPO_ROOT = Path(__file__).resolve().parents[1]


def default_run_dir(config_path: str | os.PathLike[str] | None = None) -> str:
    """返回"用户没显式指定时"应该使用的运行目录。

    只读，不创建目录、不检查目录是否存在——存在性由调用方按它自己的语义判断
    （有的工具要求目录里已有 checkpoint，有的允许新建）。
    """
    env_value = os.environ.get(ENV_VAR, "").strip()
    if env_value:
        return env_value

    path = Path(config_path) if config_path is not None else _REPO_ROOT / "abfe_config.json"
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return FALLBACK_RUN_DIR

    if isinstance(config, dict):
        value = config.get("output")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return FALLBACK_RUN_DIR


#: 供 argparse `default=` 直接引用的模块级常量。
#: 注意这是**导入时**求值的：脚本运行中途改 `ABFE_OUTPUT_DIR` 不会反映到这里，
#: 需要动态解析就调用 `default_run_dir()`。
DEFAULT_RUN_DIR = default_run_dir()


if __name__ == "__main__":  # pragma: no cover - 手工核对用
    print(DEFAULT_RUN_DIR)
