"""ABFE-IBS 离线脚本（local-residual 训练/导出/manifest）。

顶层名字刻意**不叫** `scripts`：`mace_torch` 装了个同名顶层正规包，
而正规包永远赢过 PEP 420 namespace 目录 —— 2026-09-12 实测 `from scripts.X import`
在任何装了 MACE 的环境里必然 ModuleNotFoundError。这里有 `__init__.py`，
是正规包，不依赖 namespace 解析。
"""
