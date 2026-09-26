# ============================================================================
# 8. Orbv3 → DEXP 拟合流水线 (从 DEXP_class.py 合并)
# ============================================================================


# [CLI-01, 2026-09-02] `HAS_ORB` / `HAS_PYMBAR` 现在是惰性探测（见文件开头），
# 模块里不再有这两个全局名。外部读 `abfe_core.HAS_PYMBAR` 的调用方和测试仍然
# 要能拿到布尔值，所以这里用 PEP 562 的模块级 `__getattr__` 补上。
# 注意：模块内部的代码必须调 `has_orb()` / `has_pymbar()`——模块内的全局名查找
# 不会走 `__getattr__`，写成裸 `HAS_PYMBAR` 会直接 NameError。
def __getattr__(name: str):
    if name == "HAS_ORB":
        return has_orb()
    if name == "HAS_PYMBAR":
        return has_pymbar()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
