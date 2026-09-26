#!/usr/bin/env python3
"""refactor/ 分层违反边分析器（B1）。纯 AST，不 import 被测代码。

用法：在仓库根 `python3 refactor/tools/layer_check.py`
思路复用 deps.py：Load 名 → 顶层定义归属域 → 域对有向边，在此基础上加：
  * B1 五层分层草案（HOW_TO_CONTINUE.md）的层号映射
  * 每条有向域对边标注：合法（高层→低层）/ 违反（低层→高层）/ 同层
  * 违反边按引用次数降序，列出每条边贡献最大的名字（名字/次数/定义域文件）
  * 验证 HOW_TO_CONTINUE.md B1 的 707/629/44 与 B2 的 infra 名单
    （707/629/44 为 2026-09-19 infra 抽取前基线；抽取后实测 913/444/45，
    总引用 1357/107，见 2026-09-20 勘误）
  * 模拟「抽 L0 infra 域」后剩余的违反边

口径说明（与 deps.py 完全一致）：
  * 把文件里任何位置出现过的 Store 名都当作已定义（over-approximate），
    所以「未解析名」是低估，报出来的引用数是**下界**。
  * 一个名字若在多个域都有顶层定义（跨域重名），每次 Load 会对每个
    非本域的定义域各记一次边 —— 跨域重名会放大计数，两边口径相同。
  * 同层边口径（实测结论，见输出末尾）：文档的「合法 707」= 严格 高层→低层；
    「违反 629」= 其余全部，即 同层 353 次/21 边 + 向上 276 次/23 边。
    本脚本把三桶分开报：合法(严格高→低) / 同层 / 违反(低→高)，
    并同时给出文档口径（合法 vs 非合法=同层+违反）的对照数。
"""
import ast, builtins, collections, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent   # refactor/
BUILTINS = set(dir(builtins))

# ---- B1 五层分层草案（HOW_TO_CONTINUE.md 第 4 章 B1）----
# [2026-09-19] 第一刀抽出 L0 infra 域（日志/摘要/计时/编码器/原子写 NP），注册为 L0。
LAYERS = {
    "runtime_env": 0, "infra": 0,
    "misc": 1, "result_schema": 1, "orb_surrogate": 1, "rbfe": 1, "outer_lambda_neural": 1,
    "lambda_path": 2, "system_build": 2, "charging": 2, "membrane": 2, "boresch": 2,
    "mbar_analysis": 3, "windows_sampling": 3, "remd": 3, "stage2_controller": 3,
    "pipeline_orchestration": 4,
}

# ---- B2 infra 名单（抽 L0 infra 域的候选收编对象）----
INFRA_NAMES = [
    "log", "logger", "NumpyEncoder", "_sha256_file", "_atomic_write_json",
    "print_once", "_timed", "_atomic_save_npy",
]


def bound_names(tree):
    """该文件里任何作用域绑定过的名字（含 import / 形参 / global）。同 deps.py。"""
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
    """模块级定义 —— 决定一个名字归哪个域所有。同 deps.py。"""
    s = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            s.add(n.name)
        elif isinstance(n, ast.Assign):
            s.update(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            s.add(n.target.id)
    return s


def build(files, trees, owner_override=None):
    """通用边统计。owner_override: {name: set(domain)} 强制改写名字归属（infra 模拟用）。

    返回 (edges, edge_names, external)
      edges:      {(from_dom, to_dom): 总引用次数}
      edge_names: {(from_dom, to_dom): {name: 次数}}
    """
    owner = collections.defaultdict(set)
    for p in files:
        rel = p.relative_to(ROOT)
        dom = rel.parts[1] if rel.parts[0] == "domains" else rel.parts[0]
        for n in toplevel_names(trees[p]):
            owner[n].add(dom)
    if owner_override:
        for n, doms in owner_override.items():
            owner[n] = set(doms)

    edges = collections.Counter()
    edge_names = collections.defaultdict(collections.Counter)
    external = collections.Counter()
    for p in files:
        rel = p.relative_to(ROOT)
        dom = rel.parts[1] if rel.parts[0] == "domains" else rel.parts[0]
        bound = bound_names(trees[p])
        loads = collections.Counter(
            n.id for n in ast.walk(trees[p])
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        )
        for name, cnt in loads.items():
            if name in bound or name in BUILTINS:
                continue
            homes = set(owner.get(name, ()))
            if not homes:
                external[name] += cnt
            for h in homes - {dom}:
                edges[(dom, h)] += cnt
                edge_names[(dom, h)][name] += cnt
    return edges, edge_names, external


def main():
    files = sorted(p for p in ROOT.rglob("*.py")
                   if p.relative_to(ROOT).parts[0] == "domains"   # 2026-09-20 重组：域代码在 domains/ 下，tools/cuts/work/docs 都不是域
                   and len(p.relative_to(ROOT).parts) == 3)
    if not files:
        sys.exit("refactor/ 下没有 .py —— 副本已删？只剩地图，正常。")
    trees = {p: ast.parse(p.read_text()) for p in files}
    doms = sorted({p.relative_to(ROOT).parts[1] for p in files})
    missing = [d for d in doms if d not in LAYERS]
    if missing:
        sys.exit(f"这些域不在 B1 分层表里，先补 LAYERS: {missing}")

    edges, edge_names, _external = build(files, trees)

    legal = {e: c for e, c in edges.items() if LAYERS[e[0]] > LAYERS[e[1]]}
    viol = {e: c for e, c in edges.items() if LAYERS[e[0]] < LAYERS[e[1]]}
    same = {e: c for e, c in edges.items() if LAYERS[e[0]] == LAYERS[e[1]]}
    # 文档口径（实测反推）：合法 = 严格 高→低；「违反」= 同层 + 低→高（一切非严格下降的边）
    doc_viol = {e: c for e, c in edges.items() if LAYERS[e[0]] <= LAYERS[e[1]]}
    total = sum(edges.values())

    print(f"文件 {len(files)} | 域 {len(doms)} | 跨域一方引用 {total} 次 / {len(edges)} 条有向域对边")
    print(f"B1 五层草案：合法(严格高→低) {sum(legal.values())} 次/{len(legal)} 边 | "
          f"违反(低→高) {sum(viol.values())} 次/{len(viol)} 边 | "
          f"同层 {sum(same.values())} 次/{len(same)} 边")
    print(f"文档口径：合法 {sum(legal.values())} / 非合法(同层+违反) "
          f"{sum(doc_viol.values())} 次 / {len(doc_viol)} 边   ← 707/629/44 为 infra 抽取前基线（已过期，现值见上）")

    # ---- 违反边明细（文档口径：同层 + 向上，共 44 条；降序，Top 贡献名字）----
    print("\n=== 非合法边（同层↑ / 违反↑），按引用次数降序 —— B3 逐边决策对象 ===")
    for (a, b), c in sorted(doc_viol.items(), key=lambda kv: (-kv[1], kv[0])):
        tag = "同层" if LAYERS[a] == LAYERS[b] else "违反"
        names = edge_names[(a, b)].most_common()
        top = ", ".join(f"{n}:{k}" for n, k in names[:6])
        more = f" …(+{len(names) - 6}名)" if len(names) > 6 else ""
        print(f"  {tag} L{LAYERS[a]} {a} → L{LAYERS[b]} {b}: {c} 次   [{top}{more}]")

    # ---- B2 infra 名单核验（三种口径都算，看文档 253 用的是哪种）----
    print("\n=== B2 infra 名单核验（三口径：仅向上违反边 / 文档口径非合法边 / 全部跨域边）===")
    def infra_counts_over(edge_set):
        cc = collections.Counter()
        for e in edge_set:
            for n, k in edge_names[e].items():
                if n in INFRA_NAMES:
                    cc[n] += k
        return cc
    c_up = infra_counts_over(viol)
    c_doc = infra_counts_over(doc_viol)
    c_all = infra_counts_over(edges)
    doc = {"log": 104, "logger": 33, "NumpyEncoder": 29, "_sha256_file": 28,
           "_atomic_write_json": 21, "print_once": 14, "_timed": 12, "_atomic_save_npy": 12}
    def homes_of(name):
        out = collections.defaultdict(list)
        for p in files:
            if name in toplevel_names(trees[p]):
                out[p.relative_to(ROOT).parts[1]].append(p.name)
        return out
    for n in INFRA_NAMES:
        hs = homes_of(n)
        where = "; ".join(f"{d}({'/'.join(fs)})" for d, fs in sorted(hs.items())) or "(无顶层定义)"
        print(f"  {n:22s} 向上={c_up[n]:4d}  非合法={c_doc[n]:4d}  全跨域={c_all[n]:4d}"
              f"  文档={doc.get(n)}   定义于 {where}")
    print(f"  合计  向上={sum(c_up.values())}  非合法={sum(c_doc.values())}"
          f"  全跨域={sum(c_all.values())}   文档 253（253/629=40%）")

    # ---- 模拟：抽 L0 infra 域（把 infra 名单强制改归属到虚拟 L0 域）----
    print("\n=== 模拟：抽 L0 infra 域（infra 名单全部改归虚拟 L0 域 __infra__）===")
    LAYERS["__infra__"] = 0
    e2, en2, _ = build(files, trees, owner_override={n: {"__infra__"} for n in INFRA_NAMES})
    doc_viol2 = {e: c for e, c in e2.items()
                 if LAYERS[e[0]] <= LAYERS[e[1]] and "__infra__" not in e}
    viol2 = {e: c for e, c in e2.items()
             if LAYERS[e[0]] < LAYERS[e[1]] and "__infra__" not in e}
    print(f"  文档口径非合法边: {len(doc_viol2)} 条 / {sum(doc_viol2.values())} 次"
          f"（现状 {len(doc_viol)} 条 / {sum(doc_viol.values())} 次）"
          f" ⇒ 消掉 {len(doc_viol) - len(doc_viol2)} 条 / "
          f"{sum(doc_viol.values()) - sum(doc_viol2.values())} 次")
    print(f"  严格口径向上违反边: {len(viol2)} 条 / {sum(viol2.values())} 次"
          f"（现状 {len(viol)} 条 / {sum(viol.values())} 次）"
          f" ⇒ 消掉 {len(viol) - len(viol2)} 条 / "
          f"{sum(viol.values()) - sum(viol2.values())} 次")
    gone = set(doc_viol) - set(doc_viol2)
    newv = set(doc_viol2) - set(doc_viol)
    if gone:
        print("  整条消失的边: " + "; ".join(
            f"{a}→{b}({doc_viol[(a, b)]})" for a, b in sorted(gone)))
    if newv:
        print("  ⚠ 新增的非合法边: " + "; ".join(
            f"{a}→{b}({doc_viol2[(a, b)]})" for a, b in sorted(newv)))
    print("  剩余非合法边（B3 逐条决策对象，infra 抽取后）:")
    for (a, b), c in sorted(doc_viol2.items(), key=lambda kv: (-kv[1], kv[0])):
        tag = "同层" if LAYERS[a] == LAYERS[b] else "违反"
        names = en2[(a, b)].most_common()
        top = ", ".join(f"{n}:{k}" for n, k in names[:6])
        print(f"    {tag} L{LAYERS[a]} {a} → L{LAYERS[b]} {b}: {c} 次   [{top}]")
    del LAYERS["__infra__"]

    # ---- 同层边口径结论 ----
    print("\n=== 707/629/44 口径结论（infra 抽取前基线，仅作口径说明） ===")
    print(f"  实测：合法(严格高→低) {sum(legal.values())} / 同层 {sum(same.values())} / "
          f"向上违反 {sum(viol.values())}；边数 {len(legal)}/{len(same)}/{len(viol)}")
    print(f"  文档基线（infra 抽取前）：合法 707 / 违反 629 / 44 边")
    if (sum(legal.values()) == 707 and sum(doc_viol.values()) == 629
            and len(doc_viol) == 44):
        print("  ⇒ 复现：文档的「合法」= 严格 高层→低层；「违反」= 其余全部"
              "（同层 353 次/21 边 + 向上 276 次/23 边）。")


if __name__ == "__main__":
    main()
