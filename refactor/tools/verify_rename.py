#!/usr/bin/env python3
"""归组变换的等价门：验「只是把散落的名字收进了对象，逻辑一个节点都没动」。

`verify_extract.py` 管的是**纯搬运**（块换个地方，AST 逐节点相同）。
但把 51 条散落初始化收成 `@dataclass WarmupState`、或把 40 个入参归成 6 个 Config，
会改动成千上万处名字引用（`x` → `state.x`），`verify_extract.py` 一概判 FAIL。

本工具补这个缺口：在「`Name(x)` ↔ `Attribute(Name(bind), x)`」这一组**受控替换**下
比对 AST。白名单之外的任何改动都会被抓出来。

它能证：
  ① 每一处 `bind.x` 都还原得回原来的 `x`，且还原后逻辑逐节点相同；
  ② 从函数体里消失的初始化语句，恰好是白名单里那些，不多不少；
  ③ 每条消失的初始化，其初值与数据类字段的默认值 AST 相同。
它证不了：运行时行为。

用法：
    python3 refactor/tools/verify_rename.py <改动前> <改动后> \
        --fn _decide_once --bind state \
        --names a,b,c            # 或 --names-file <每行一个>
        [--dataclass WarmupState]   # 给了就一并做 ②③
"""
import argparse
import ast
import sys


def _strip(node):
    for n in ast.walk(node):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset",
                     "type_comment"):
            if hasattr(n, attr):
                try:
                    delattr(n, attr)
                except AttributeError:
                    setattr(n, attr, None)
    return node


def _dump(nodes):
    return [ast.dump(_strip(n), annotate_fields=True) for n in nodes]


def _find_fn(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    return None


def _find_class(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    return None


def _body(fn):
    b = list(fn.body)
    if (b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant)
            and isinstance(b[0].value.value, str)):
        b = b[1:]
    return b


class _Unbind(ast.NodeTransformer):
    """把 `bind.x` 还原成 `x`（仅限白名单），并记账还原了多少处。"""

    def __init__(self, bind, names, keep=()):
        self.bind = bind
        self.names = names
        self.keep = set(keep)   # 既有 bind.<attr> 访问：不还原、不算 stray
        self.hits = 0
        self.stray = set()   # 白名单外、但也是 bind.<attr> 的访问

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == self.bind:
            if node.attr in self.names:
                self.hits += 1
                return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
            if node.attr not in self.keep:
                self.stray.add(node.attr)
        return node


def _init_targets(stmt):
    """这条语句是不是「给单个名字赋值」？是就返回 (名字, 值节点)。"""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
            and isinstance(stmt.targets[0], ast.Name):
        return stmt.targets[0].id, stmt.value
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id, stmt.value
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--fn", required=True, help="被改写的函数名（两侧同名）")
    ap.add_argument("--bind", required=True, help="承载对象的变量名，如 state / cfg")
    ap.add_argument("--names", default="", help="白名单，逗号分隔")
    ap.add_argument("--names-file", default=None, help="白名单文件，每行一个")
    ap.add_argument("--expand-kwarg", action="store_true",
                    help="after 侧把 `kwarg bind=bind` 展开为白名单 9 名的 kwargs"
                         "（值=同名 Name，按白名单排序序）再比对"
                         "（Config 归组：调用点 9 kwarg 折 1 的还原式比对）")
    ap.add_argument("--keep", default="", help="既有 bind.<attr> 访问名单（逗号分隔）："
                    "不还原、也不算 stray（增量收编场景：白名单只含本次新增名）")
    ap.add_argument("--keep-file", default=None, help="既有访问名单文件，每行一个")
    ap.add_argument("--dataclass", default=None,
                    help="改动后文件里的数据类名；给了就核对②③（初始化搬家对账）")
    ap.add_argument("--block", default="fn",
                    choices=("fn", "for", "if", "while", "with"),
                    help="比对锚点：fn = --fn 整个体（默认，原口径）；"
                         "for/if/while/with = 锚到 --fn 顶层第 --index 个该类块的体"
                         "（WarmupState：51 条初始化在 run_all_windows 的窗口 for 体内）")
    ap.add_argument("--index", type=int, default=0,
                    help="--block 非 fn 时取第几个该类块（默认 0）")
    ap.add_argument("--in-place", action="store_true",
                    help="原地收编口径：白名单名字的初始化语句**保留原地**（变成 "
                         "bind.x = ... 写入），不搬家。此模式下 ② 不做『消失对账』，"
                         "改为校验每个白名单名在改动前有绑定；③(--dataclass) 做"
                         "『字段存在性』校验而非初值比对")
    ap.add_argument("--construction", type=int, default=None,
                    help="锚点内允许的 `<bind> = ...` 构造语句条数，对账后从比对中剔除；"
                         "默认 None = 不允许任何构造语句（旧口径）。WarmupState 用 1")
    a = ap.parse_args()

    names = set(x.strip() for x in a.names.split(",") if x.strip())
    if a.names_file:
        names |= {l.strip() for l in open(a.names_file, encoding="utf-8")
                  if l.strip() and not l.startswith("#")}
    keep = set(x.strip() for x in a.keep.split(",") if x.strip())
    if a.keep_file:
        keep |= {l.strip() for l in open(a.keep_file, encoding="utf-8")
                 if l.strip() and not l.startswith("#")}
    if names & keep:
        sys.exit(f"FAIL: --names 与 --keep 交集非空: {sorted(names & keep)}")
    if not names:
        sys.exit("FAIL: 白名单为空 —— 这个门靠白名单限定改动范围，不能不给")

    tb = ast.parse(open(a.before, encoding="utf-8").read())
    ta = ast.parse(open(a.after, encoding="utf-8").read())
    fb, fa = _find_fn(tb, a.fn), _find_fn(ta, a.fn)
    if fb is None:
        sys.exit(f"FAIL: 改动前没有 {a.fn}")
    if fa is None:
        sys.exit(f"FAIL: 改动后没有 {a.fn}")

    def _anchor(fn_node, side):
        body = _body(fn_node)
        if a.block == "fn":
            return body, None
        kind = {"for": ast.For, "if": ast.If, "while": ast.While,
                "with": ast.With}[a.block]
        blocks = [s for s in body if isinstance(s, kind)]
        if len(blocks) <= a.index:
            sys.exit(f"FAIL: {side}侧 {a.fn} 顶层只有 {len(blocks)} 个 {a.block} 块，"
                     f"取不到第 {a.index} 个")
        blk = blocks[a.index]
        pos = next(i for i, s in enumerate(body) if s is blk)
        return list(blk.body), pos


    def _refix_simple(stmts):
        """_Unbind 把 AnnAssign 目标从 Attribute 还原成 Name 后，simple 按解析器
        口径应为 1（解析器只在目标为非括号 Name 时记 1；Attribute 一律 0）。"""
        fixed = 0
        for st in stmts:
            for n in ast.walk(st):
                if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) \
                        and n.simple == 0:
                    n.simple = 1
                    fixed += 1
        return fixed

    bb, pos_b = _anchor(fb, "改动前")
    ba, pos_a = _anchor(fa, "改动后")

    # --- ② 从函数体消失的初始化，必须恰好是白名单里的 ---
    # 每个名字只吸收**首个**顶层赋值（= 搬走的 init）；同名再赋值（如相位4重建
    # bias_warmup_diag 大字典）是留在原地的普通语句，须留在比对序列里。
    moved = {}          # 名字 -> 初值节点（改动前）
    kept_before = []
    if a.in_place:
        # 原地收编：初始化不搬家。校验每个白名单名在改动前有绑定（Store/形参/例外）
        def _binds_all(stmts):
            b = set()
            for st in stmts:
                for n in ast.walk(st):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                        b.add(n.id)
                    elif isinstance(n, ast.arg):
                        b.add(n.arg)
            return b
        bound = _binds_all(bb)
        bound |= {a.arg for a in fb.args.args + fb.args.kwonlyargs + fb.args.posonlyargs}
        if fb.args.vararg:
            bound.add(fb.args.vararg.arg)
        if fb.args.kwarg:
            bound.add(fb.args.kwarg.arg)
        nobind = sorted(names - bound)
        if nobind:
            print(f"FAIL: 原地收编白名单里 {len(nobind)} 个名字在改动前无绑定：{nobind[:8]}")
            return 1
    for s in bb:
        nm, val = _init_targets(s)
        if not a.in_place and nm in names and val is not None and nm not in moved:
            moved[nm] = val
            continue
        kept_before.append(s)

    # --- ① 还原 bind.x 后逐节点比对 ---
    ub = _Unbind(a.bind, names, keep)
    restored = [ub.visit(s) for s in ba]
    for s in restored:
        ast.fix_missing_locations(s)

    _refix_simple(restored)

    # --- ⓪ 构造语句对账（--construction 显式给出时）：`<bind> = ...` 计数后剔除 ---
    if a.construction is not None:
        def _is_construction(s):
            return (isinstance(s, ast.Assign) and len(s.targets) == 1
                    and isinstance(s.targets[0], ast.Name)
                    and s.targets[0].id == a.bind)
        cons = [s for s in restored if _is_construction(s)]
        if len(cons) != a.construction:
            print(f"FAIL: 锚点内 `<{a.bind}> = ...` 构造语句实测 {len(cons)} 条，"
                  f"--construction 期望 {a.construction}")
            return 1
        restored = [s for s in restored if not _is_construction(s)]
        # before 侧既有构造（上一刀引入、本刀仅改 kwarg）成对剔除；
        # kwarg 增量不在本门射程（执行器双向验证覆盖）
        bcons = [s for s in kept_before if _is_construction(s)]
        if bcons:
            kept_before = [s for s in kept_before if not _is_construction(s)]
        print(f"构造语句 after {len(cons)} 条 / before 既有 {len(bcons)} 条 "
              f"已成对剔除（kwarg 增量归执行器双向验证）")

    if a.expand_kwarg:
        # 按 before 同形调用的实况展开（全树扫描：折叠的调用点可能嵌套在
        # For/If 子树里，只扫顶层会漏——WarmupContext 教训）。
        # 匹配：func 相同 + 非 bind kwargs 与 before 某 Call 逐字相同 → 唯一
        # 候选才展开（采纳 before 的 kwarg 列表）；0/多候选留给 ① FAIL。
        n_exp = [0]
        bcalls = [n for n in ast.walk(ast.Module(body=kept_before, type_ignores=[]))
                  if isinstance(n, ast.Call)]

        def _non_bind_kw_dump(c):
            return [(kw.arg, _dump([kw.value]))
                    for kw in c.keywords if kw.arg != a.bind]

        def _is_self_attr_call(c):
            return isinstance(c.func, ast.Attribute) \
                and isinstance(c.func.value, ast.Name) \
                and c.func.value.id == "self"

        for st in restored:
            for n in ast.walk(st):
                if not (isinstance(n, ast.Call) and _is_self_attr_call(n)):
                    continue
                hit = next((i for i, kw in enumerate(n.keywords)
                            if kw.arg == a.bind and isinstance(kw.value, ast.Name)
                            and kw.value.id == a.bind), None)
                if hit is None:
                    continue
                def _non_bind_non_names_dump(c):
                    return [(kw.arg, _dump([kw.value]))
                            for kw in c.keywords
                            if not (kw.arg == a.bind
                                    or (kw.arg in names
                                        and isinstance(kw.value, ast.Name)
                                        and kw.value.id == kw.arg))]
                ac_non = _non_bind_non_names_dump(n)
                cands = [bc for bc in bcalls
                         if _is_self_attr_call(bc)
                         and bc.func.attr == n.func.attr
                         and _non_bind_non_names_dump(bc) == ac_non
                         and any(kw.arg in names for kw in bc.keywords)]
                if len(cands) != 1:
                    continue
                bc = cands[0]
                n.keywords = [ast.keyword(arg=kw.arg, value=kw.value)
                              for kw in bc.keywords]
                n_exp[0] += 1
        for st in restored:
            ast.fix_missing_locations(st)
        print(f"expand-kwarg: {n_exp[0]} 个调用点已按 before 实况展开")

    print(f"白名单 {len(names)} 个 | 还原 {a.bind}.x 共 {ub.hits} 处 "
          f"| 从函数体移走的初始化 {len(moved)} 条")
    if ub.stray:
        print(f"FAIL: 出现白名单外的 {a.bind}.<attr> 访问：{sorted(ub.stray)}")
        return 1

    never_moved = sorted(names - set(moved))
    if never_moved and a.dataclass:
        print(f"注意: 白名单里有 {len(never_moved)} 个在改动前不是顶层初始化 "
              f"（形参或深层赋值）：{never_moved[:8]}")

    de, da = _dump(kept_before), _dump(restored)
    if len(de) != len(da):
        print(f"FAIL: 语句条数不同 {len(de)} vs {len(da)}（漏搬/多搬/顺序变了）")
        return 1
    bad = next((i for i, (x, y) in enumerate(zip(de, da)) if x != y), None)
    if bad is not None:
        print("FAIL: 还原后 AST 仍不等价")
        print(f"  第 {bad} 条语句起出现差异：")
        print(f"    改动前: {de[bad][:200]}")
        print(f"    改动后: {da[bad][:200]}")
        return 1
    print("PASS ①: 还原后逐节点等价 —— 除名字绑定外逻辑未动")

    # --- ①' 锚点之外的方法体顶层语句必须逐节点等价（其中的 bind.x 一并还原） ---
    if a.block != "fn":
        other_b = [s for i, s in enumerate(_body(fb)) if i != pos_b]
        other_a = [s for i, s in enumerate(_body(fa)) if i != pos_a]
        ub2 = _Unbind(a.bind, names, keep)
        other_a = [ub2.visit(s) for s in other_a]
        for s in other_a:
            ast.fix_missing_locations(s)
        _refix_simple(other_a)
        if ub2.stray:
            print(f"FAIL: 锚点外出现白名单外的 {a.bind}.<attr>：{sorted(ub2.stray)}")
            return 1
        do, da2 = _dump(other_b), _dump(other_a)
        if len(do) != len(da2):
            print(f"FAIL: 锚点外顶层语句条数不同 {len(do)} vs {len(da2)}")
            return 1
        bad2 = next((i for i, (x, y) in enumerate(zip(do, da2)) if x != y), None)
        if bad2 is not None:
            print(f"FAIL: 锚点之外的第 {bad2} 条顶层语句不等价（不许动锚点外代码）")
            print(f"    改动前: {do[bad2][:200]}")
            print(f"    改动后: {da2[bad2][:200]}")
            return 1
        print(f"PASS ①': 锚点外 {len(other_b)} 条顶层语句逐节点等价")

    # --- ③ 初始化搬家对账 ---
    if a.dataclass:
        cls = _find_class(ta, a.dataclass)
        if cls is None:
            print(f"FAIL: 改动后没有数据类 {a.dataclass}")
            return 1
        fields = {}
        for s in cls.body:
            if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
                fields[s.target.id] = s.value
        if a.in_place:
            absent = sorted(names - set(fields))
            if absent:
                print(f"FAIL: 原地收编白名单里 {len(absent)} 个名字不是 {a.dataclass} 字段：{absent[:8]}")
                return 1
            print(f"PASS ②③(原地): {len(names)} 名全部是 {a.dataclass} 字段（初值原地保留，不比对默认值）")
            return 0
        missing = sorted(set(moved) - set(fields))
        if missing:
            print(f"FAIL: 这些初始化从函数体消失了，但数据类里没有对应字段：{missing}")
            return 1
        mismatch = []
        for nm, val in moved.items():
            dv = fields[nm]
            if dv is None:
                mismatch.append((nm, ast.unparse(val), "<无默认值>"))
                continue
            # 可变默认值通常写成 field(default_factory=...)，此时只比对工厂名下的字面量
            if (isinstance(dv, ast.Call) and isinstance(dv.func, ast.Name)
                    and dv.func.id == "field"):
                kw = {k.arg: k.value for k in dv.keywords}
                fac = kw.get("default_factory")
                if fac is not None and isinstance(fac, (ast.Lambda,)):
                    dv = fac.body
                elif fac is not None and isinstance(fac, ast.Name):
                    # list / dict / set 这类内建工厂：拿空容器当等价物
                    dv = {"list": ast.List(elts=[], ctx=ast.Load()),
                          "dict": ast.Dict(keys=[], values=[]),
                          "set": ast.Call(func=ast.Name(id="set", ctx=ast.Load()),
                                          args=[], keywords=[])}.get(fac.id, dv)
                elif "default" in kw:
                    dv = kw["default"]
            if _dump([val]) != _dump([dv]):
                mismatch.append((nm, ast.unparse(val), ast.unparse(dv)))
        if mismatch:
            print(f"FAIL: {len(mismatch)} 个字段的初值与数据类默认值不同：")
            for nm, a_, b_ in mismatch[:10]:
                print(f"    {nm}: 原 {a_}  ->  字段默认 {b_}")
            return 1
        extra = sorted(set(fields) - set(moved) - names)
        if extra:
            print(f"注意: 数据类有 {len(extra)} 个字段不来自本次搬家：{extra[:8]}")
        print(f"PASS ②③: {len(moved)} 条初始化全部落到 {a.dataclass} 字段，初值逐一相同")

    return 0


if __name__ == "__main__":
    sys.exit(main())
