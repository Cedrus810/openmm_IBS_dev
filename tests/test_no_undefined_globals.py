"""全仓扫描：函数体里引用了一个**模块里根本没有的全局名**。

S2-D 那次搬家踩的就是这个形状 —— 删掉函数内的 `from abfe_preoptimizer import ...`，
却漏改函数体里的裸名字。`_relearn_epoch_required_steps` 那条被单独写了一条 grep
测试钉住，而同一次搬家在 `_legalize_tail_window` 里留下的**四个**同类漏网
（`insert_lambda_in_failed_ibs_window` / `repartition_tail_from_anchor` /
`record_tail_repartition_version` / `_pre`）一条都没被抓住：那条路只有"末窗超上限
必须拆"时才走，离线测试全绿、GPU 上才炸。

所以这里不再逐个 grep 符号名，改成 `symtable` 的通用判据：
函数作用域里被当**全局**读、模块顶层又没有这个名字 ⟹ 红。
"""
import ast
import builtins
import pathlib
import symtable

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__"}


def _module_level_names(tree: ast.Module) -> set:
    """模块顶层绑定的名字（含 if/try/for/with 里的 import 与赋值）。

    ⚠️ **不下潜进函数/类体** —— 那里面的 `from x import y` 是局部名字。下潜过一次，
    结果是把函数内的 import 当成模块级绑定，本测试自己就瞎了（改坏了验证不出来）。
    """
    names = set()

    def visit(node, top):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            return                      # 体内是局部作用域，到此为止
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        for child in ast.iter_child_nodes(node):
            visit(child, False)

    for node in tree.body:
        visit(node, True)
    return names


def _undefined(path: pathlib.Path):
    src = path.read_text(encoding="utf-8")
    known = _module_level_names(ast.parse(src)) | BUILTINS
    found = []

    def walk(table, scope):
        for child in table.get_children():
            walk(child, scope + [child.get_name()])
        if table.get_type() != "function":
            return
        for sym in table.get_symbols():
            name = sym.get_name()
            if sym.is_global() and not sym.is_assigned() and name not in known:
                found.append(f"{'.'.join(scope)}: {name}")

    walk(symtable.symtable(src, str(path), "exec"), [])
    return sorted(set(found))


def test_no_function_reads_a_name_the_module_never_defines():
    bad = {}
    for path in sorted(ROOT.glob("*.py")) + sorted(ROOT.glob("local_residual/*.py")):
        missing = _undefined(path)
        if missing:
            bad[path.name] = missing
    assert not bad, (
        "这些函数读了一个模块里不存在的全局名（多半是搬家时删了 import 没改调用）：\n"
        + "\n".join(f"  {f}:\n    " + "\n    ".join(v) for f, v in bad.items())
    )
