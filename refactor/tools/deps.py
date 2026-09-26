#!/usr/bin/env python3
"""refactor/ 域依赖体检。纯 AST，不 import 被测代码。

用法：在仓库根 `python3 refactor/tools/deps.py`
每切一刀重跑一次：环数和跨域引用数应当单调下降。

近似说明：把文件里任何位置出现的 Store 名都当作已定义（over-approximate），
所以是**低估**未解析名——报出来的数是下界。
"""
import ast, builtins, collections, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent   # refactor/
BUILTINS = set(dir(builtins))


def bound_names(tree):
    """该文件里任何作用域绑定过的名字（含 import / 形参 / global）。"""
    s = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            s.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            s.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            s.update((a.asname or a.name).split(".")[0] for a in n.names)
        elif isinstance(n, ast.arg):
            s.add(n.arg)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            s.update(n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            s.add(n.name)
    return s


def toplevel_names(tree):
    """模块级定义 —— 决定一个名字归哪个域所有。"""
    s = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            s.add(n.name)
        elif isinstance(n, ast.Assign):
            s.update(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            s.add(n.target.id)
    return s




def _dom_of(p):
    """重组后域代码在 domains/ 下;tools/cuts/work/docs 保持噪音域口径。"""
    parts = p.relative_to(ROOT).parts
    return parts[1] if parts[0] == "domains" and len(parts) > 2 else parts[0]

files = sorted(p for p in ROOT.rglob("*.py")
               if "split_tool" not in str(p) and p.name != "deps.py")
if not files:
    sys.exit("refactor/ 下没有 .py —— 副本已删？那就只剩 WIRING.md 这张地图了，正常。")

trees = {p: ast.parse(p.read_text()) for p in files}
owner = collections.defaultdict(list)
for p in files:
    for n in toplevel_names(trees[p]):
        owner[n].append(p)

edges = collections.Counter()      # (使用方域, 定义方域) -> 引用次数
external = collections.Counter()   # 没有任何域定义 => stdlib / 三方
free_files = 0
for p in files:
    dom = _dom_of(p)
    bound = bound_names(trees[p])
    loads = collections.Counter(
        n.id for n in ast.walk(trees[p])
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    )
    free = {k: v for k, v in loads.items() if k not in bound and k not in BUILTINS}
    free_files += bool(free)
    for name, cnt in free.items():
        homes = {_dom_of(q) for q in owner.get(name, ())}
        if not homes:
            external[name] += cnt
        for h in homes - {dom}:
            edges[(dom, h)] += cnt

doms = sorted({_dom_of(p) for p in files})
cycles = sorted({(a, b) for a, b in edges if (b, a) in edges and a < b})
dupes = {n: sorted({_dom_of(q) for q in v})
         for n, v in owner.items()
         if len({_dom_of(q) for q in v}) > 1}

print(f"文件 {len(files)} | 域 {len(doms)} | 有未解析名的文件 {free_files}")
print(f"跨域一方引用 {sum(edges.values())} 次，横跨 {len(edges)}/{len(doms)*(len(doms)-1)} 个有向域对")
print(f"stdlib/三方待补 import：{len(external)} 个名 / {sum(external.values())} 次引用")
print(f"★ 循环依赖域对 {len(cycles)}  ← 这个数降到 0 之前，不具备直接接线条件")

print("\n每域出/入边（出边 0 = 可独立成模块的真叶子）：")
out = collections.Counter(); inn = collections.Counter(); outd = collections.Counter()
for (a, b), c in edges.items():
    out[a] += c; inn[b] += c; outd[a] += 1
for d in sorted(doms, key=lambda x: out[x]):
    mark = "  ← 叶子" if out[d] == 0 else ""
    print(f"  {d:24s} out={out[d]:5d} ({outd[d]:2d} 域)  in={inn[d]:5d}{mark}")

if cycles:
    print("\n循环依赖域对：")
    for a, b in cycles:
        print(f"  {a} ↔ {b}   ({edges[(a,b)]} / {edges[(b,a)]})")

if dupes:
    print(f"\n跨域重名顶层定义 {len(dupes)} 个（去重目标，见 HOW_TO_CONTINUE.md ❸）：")
    for n, ds in sorted(dupes.items()):
        print(f"  {n:38s} {', '.join(ds)}")
