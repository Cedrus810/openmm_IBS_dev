#!/usr/bin/env python3
"""调用点实参必然已绑定门：抽取/提升后，传给新方法的每个裸名实参在调用点**必然已绑定**。

缺口（第四次同形）：verify_extract 只验搬走的块，调用点与签名是新写的代码。
「保守公式」形参 = 段内 Load ∩ 调用方可见绑定——但「可见」若按「某处有 Store」算，
会把两类名字当成实参传出去：
  ① 只在**条件分支/循环/try 体/内嵌 def** 里绑定的名字：原代码只在绑定后的路径上才读，
     改成 x=x 实参后在调用点就求值 ⟹ 未绑定路径上 UnboundLocalError 提前爆
     （scheme_dispatch 的 f/cached/point…；step0 plan 在 _blocks 赋值前的早退出口）；
  ② 闭包提升：内嵌 def 读自由变量是**调用时**读，提升成模块函数后变成**调用点**求值。

判据（保守，宁可误报）：对 module 里每个 `self.<callee>(...)` / `<callee>(...)` 调用，
裸名实参 x 须满足其一：
  - x 是所在函数的局部名，且按「必然赋值」分析在该调用点已绑定；
  - x 不是所在函数（及外层函数）的局部名，而是模块级名或内建名。
所在函数是内嵌 def、x 是外层函数局部（闭包捕获）⟹ 时序不可静态判，一律报。

用法：
    python3 refactor/tools/verify_bound.py <file> --callee NAME [--callee NAME ...]
"""
import argparse
import ast
import builtins
import sys

SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _iter_own(node):
    """遍历 node，但不进入内嵌作用域（def/lambda/class 体、推导式）。"""
    if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return                      # 根本身就是内层作用域（如 `x = [.. for k in ..]`）
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield node                  # 根是内嵌 def：只有名字与装饰器属于本作用域
        for d in node.decorator_list:
            yield from _iter_own(d)
        return
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        for ch in ast.iter_child_nodes(n):
            if isinstance(ch, SCOPES) or isinstance(
                    ch, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    yield ch          # 名字本身绑定在本作用域；装饰器/默认值另算
                    for d in ch.decorator_list:
                        stack.append(d)
                continue
            stack.append(ch)


def _target_names(t):
    return {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}


def own_locals(fn):
    a = fn.args
    loc = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    for v in (a.vararg, a.kwarg):
        if v:
            loc.add(v.arg)
    for s in fn.body:
        for n in _iter_own(s):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                loc.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                loc.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                loc |= {(x.asname or x.name).split(".")[0] for x in n.names}
            elif isinstance(n, ast.ExceptHandler) and n.name:
                loc.add(n.name)
            elif isinstance(n, ast.NamedExpr):
                loc |= _target_names(n.target)
    for n in ast.walk(fn):          # global/nonlocal 声明的不算局部
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            loc -= set(n.names)
    return loc


COMPS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _scoped_calls(node, bound, deferred=False):
    """表达式里的调用 → (call, 当时已绑定集合, 是否延迟执行)。

    推导式当场执行：体内调用按「外层已绑定 ∪ 推导式目标」算（生成器表达式严格说
    是惰性的，但本仓用法都是立即消费，按当场算）；lambda 延迟执行 ⟹ 闭包语义。
    """
    if node is None:
        return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return
    if isinstance(node, ast.Lambda):
        a = node.args
        inner = bound | {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
        for d in a.defaults + [x for x in a.kw_defaults if x]:
            yield from _scoped_calls(d, bound, deferred)
        yield from _scoped_calls(node.body, inner, True)
        return
    if isinstance(node, COMPS):
        b = set(bound)
        for g in node.generators:
            yield from _scoped_calls(g.iter, b, deferred)
            b |= _target_names(g.target)
            for c in g.ifs:
                yield from _scoped_calls(c, b, deferred)
        for part in ([node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]):
            yield from _scoped_calls(part, b, deferred)
        return
    if isinstance(node, ast.Call):
        yield node, frozenset(bound), deferred
    for ch in ast.iter_child_nodes(node):
        yield from _scoped_calls(ch, bound, deferred)


class DefiniteAssignment:
    """保守的必然赋值分析；在目标调用处记录当时的已绑定集合。"""

    def __init__(self, targets):
        self.targets = targets     # id(call) 集合
        self.found = {}

    def expr(self, node, bound):
        if node is None:
            return bound
        walrus = set()
        for c, b, deferred in _scoped_calls(node, bound):
            if id(c) in self.targets:
                self.found[id(c)] = (b, deferred)
        for n in _iter_own(node):
            if isinstance(n, ast.NamedExpr):
                walrus |= _target_names(n.target)
        return bound | walrus

    def block(self, stmts, bound):
        for s in stmts:
            bound, term = self.stmt(s, bound)
            if term:
                return bound, True
        return bound, False

    def stmt(self, s, b):
        E = self.expr
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for d in s.decorator_list:
                b = E(d, b)
            if not isinstance(s, ast.ClassDef):
                for d in s.args.defaults + [x for x in s.args.kw_defaults if x]:
                    b = E(d, b)
            return b | {s.name}, False
        if isinstance(s, (ast.Return, ast.Raise)):
            E(getattr(s, "value", None) or getattr(s, "exc", None), b)
            return b, True
        if isinstance(s, (ast.Break, ast.Continue)):
            return b, True
        if isinstance(s, ast.Assign):
            b = E(s.value, b)
            for t in s.targets:
                b = E(t, b)   # 下标/属性目标里的调用
                b = b | _target_names(t)
            return b, False
        if isinstance(s, ast.AnnAssign):
            b = E(s.value, b)
            return (b | _target_names(s.target)) if s.value is not None else b, False
        if isinstance(s, ast.AugAssign):
            return E(s.value, b), False
        if isinstance(s, ast.Delete):
            return b - {n for t in s.targets for n in _target_names(t)}, False
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            return b | {(x.asname or x.name).split(".")[0] for x in s.names}, False
        if isinstance(s, ast.If):
            b = E(s.test, b)
            b1, t1 = self.block(s.body, b)
            b2, t2 = self.block(s.orelse, b)
            if t1 and t2:
                return b, True
            return (b2 if t1 else b1 if t2 else b1 & b2), False
        if isinstance(s, (ast.For, ast.AsyncFor)):
            b = E(s.iter, b)
            self.block(s.body, b | _target_names(s.target))
            self.block(s.orelse, b)
            return b, False
        if isinstance(s, ast.While):
            b = E(s.test, b)
            self.block(s.body, b)
            self.block(s.orelse, b)
            return b, False
        if isinstance(s, (ast.With, ast.AsyncWith)):
            for it in s.items:
                b = E(it.context_expr, b)
                if it.optional_vars is not None:
                    b = b | _target_names(it.optional_vars)
            return self.block(s.body, b)
        if isinstance(s, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            bb, tb = self.block(s.body, b)
            exits = []
            if not tb:
                bo, to = self.block(s.orelse, bb)
                if not to:
                    exits.append(bo)
            for h in s.handlers:
                E(h.type, b)
                hb, th = self.block(h.body, b | ({h.name} if h.name else set()))
                if not th:
                    exits.append(hb - ({h.name} if h.name else set()))
            term = not exits
            after = frozenset.intersection(*map(frozenset, exits)) if exits else b
            if s.finalbody:
                bf, tf = self.block(s.finalbody, set(after))
                return bf, term or tf
            return set(after), term
        if isinstance(s, ast.Match):
            b = E(s.subject, b)
            for c in s.cases:
                pn = {n.name for n in ast.walk(c.pattern) if getattr(n, "name", None)}
                self.block(c.body, b | pn)
            return b, False
        # Expr / Assert / Pass / Global / Nonlocal …
        for ch in ast.iter_child_nodes(s):
            if isinstance(ch, ast.expr):
                b = E(ch, b)
        return b, False


def _stmt_calls(s):
    """语句里属于本函数作用域执行的调用（含推导式/lambda，不含内嵌 def 体）。"""
    for n in _iter_own(s):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for ch in ast.iter_child_nodes(n):
            if isinstance(ch, COMPS + (ast.Lambda,)):
                yield from _scoped_calls(ch, frozenset())
        if isinstance(n, ast.Call):
            yield n, None, False


def _func_parents(tree):
    """每个函数节点 → 外层函数链（由内到外）。"""
    parents = {}

    def rec(node, chain):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                parents[id(ch)] = chain
                rec(ch, [ch] + chain)
            else:
                rec(ch, chain)
    rec(tree, [])
    return parents


def bound_extra(c, bound):
    """推导式/lambda 目标名：不是函数局部，但在该调用处已绑定。"""
    return set(bound or ())


def check(path, callees, only_args=()):
    src = open(path, encoding="utf-8", newline="").read()
    tree = ast.parse(src)
    module_names = set(dir(builtins))
    for s in tree.body:
        for n in _iter_own(s):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                module_names.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                module_names.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                module_names |= {(x.asname or x.name).split(".")[0] for x in n.names}
    parents = _func_parents(tree)
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    bad, n_calls = [], 0
    for fn in funcs:
        calls = [c for s in fn.body for c, _, _ in _stmt_calls(s) if (
            (isinstance(c.func, ast.Attribute) and c.func.attr in callees
             and isinstance(c.func.value, ast.Name) and c.func.value.id == "self")
            or (isinstance(c.func, ast.Name) and c.func.id in callees))]
        if not calls:
            continue
        n_calls += len(calls)
        loc = own_locals(fn)
        outer = set()
        for o in parents.get(id(fn), []):
            if not isinstance(o, ast.Lambda):
                outer |= own_locals(o)
        da = DefiniteAssignment({id(c) for c in calls})
        a = fn.args
        init = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
        init |= {v.arg for v in (a.vararg, a.kwarg) if v}
        da.block(fn.body, init)
        for c in calls:
            cname = c.func.attr if isinstance(c.func, ast.Attribute) else c.func.id
            bound, deferred = da.found.get(id(c), (None, False))
            for v in list(c.args) + [k.value for k in c.keywords]:
                if isinstance(v, ast.Starred):
                    v = v.value
                if not isinstance(v, ast.Name):
                    continue
                x = v.id
                if only_args and x not in only_args:
                    continue
                if x in bound_extra(c, bound):
                    continue
                if x in loc:
                    if deferred:
                        bad.append((fn.name, c.lineno, cname, x, "lambda 内调用（延迟执行，闭包时序不可静态判）"))
                    elif bound is None:
                        bad.append((fn.name, c.lineno, cname, x, "调用点不可达/未分析"))
                    elif x not in bound:
                        bad.append((fn.name, c.lineno, cname, x, "调用点未必已绑定"))
                elif x in outer:
                    bad.append((fn.name, c.lineno, cname, x, "闭包捕获（调用时序不可静态判）"))
                elif x not in module_names:
                    bad.append((fn.name, c.lineno, cname, x, "所在函数与模块都不绑定它"))
    return bad, n_calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--callee", action="append", required=True)
    ap.add_argument("--arg", action="append", default=[],
                    help="只检查这些实参名（归组刀只新增 bind=bind 一个实参时用）")
    ap.add_argument("--allow", action="append", default=[], metavar="NAME=理由",
                    help="已人工证明必然绑定的名字（分析不了分支条件相关性时用）；必须写理由")
    a = ap.parse_args()
    allow = {}
    for x in a.allow:
        k, _, why = x.partition("=")
        if not why.strip():
            sys.exit(f"FAIL: --allow {k} 没写理由")
        allow[k] = why
    bad, n = check(a.file, set(a.callee), set(a.arg))
    for k, why in allow.items():
        if any(b[3] == k for b in bad):
            print(f"  [豁免] {k}: {why}")
    bad = [b for b in bad if b[3] not in allow]
    if n == 0:
        print(f"FAIL: 找不到 {a.callee} 的任何调用点")
        return 1
    if bad:
        print(f"FAIL: {len(bad)} 个实参在调用点未必已绑定（{n} 个调用点）")
        seen = set()
        for fn, ln, cn, x, why in bad:
            if (fn, cn, x, why) in seen:
                continue
            seen.add((fn, cn, x, why))
            k = sum(1 for b in bad if b[0] == fn and b[2] == cn and b[3] == x)
            print(f"  {fn} → {cn}(…{x}=…) @L{ln}{f' 等 {k} 处' if k > 1 else ''}：{why}")
        return 1
    print(f"PASS: {n} 个调用点，全部裸名实参必然已绑定")
    return 0


if __name__ == "__main__":
    sys.exit(main())
