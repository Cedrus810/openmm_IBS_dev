#!/usr/bin/env python3
"""通用切分引擎——取代 16 个特化执行器。

**为什么要它**：`refactor/cuts/` 里 16 个执行器共 3600 行，90% 是同一套样板
（找类/找方法 → 结构锚定位 → 安全检查 → 组装新方法 → 改写调用点 → 双向验证）。
特化的代价不只是重复：每写一个新执行器就要重犯一遍同样的坑
（CRLF 被洗、AnnAssign 漏接、缩进模板误套、调用点嵌套漏扫……都真栽过）。

**设计**：引擎固定，每刀只给一份**声明式 spec**。两种模式对应两道现成的门：

    mode="extract"  语句块搬成新方法      → verify_extract.py
    mode="absorb"   散落名字收进对象      → verify_rename.py

**位置无关**：spec 用 `start`/`end` 两个**结构谓词**定位，不用行号或下标。
主线小修会让下标漂移，但不会改变锚点语句的形状；锚点命中数 ≠ 1 一律
**fail-closed**，让人来看，不让脚本猜。

用法：
    python3 refactor/tools/cutter.py <spec 模块路径> [--root <目标树>] [--dry-run]

`--root` 默认 `refactor/domains`；重放时指向 `refactor/replay`。
"""
import argparse
import ast
import builtins
import importlib.util
import pathlib
import sys

BUILTINS = set(dir(builtins))


# ─────────────────────────── 谓词工具（供 spec 使用） ───────────────────────────

def assign_attr(obj, attr):
    """匹配 `obj.attr = ...`。"""
    def pred(s):
        return (isinstance(s, ast.Assign) and len(s.targets) == 1
                and isinstance(s.targets[0], ast.Attribute)
                and isinstance(s.targets[0].value, ast.Name)
                and s.targets[0].value.id == obj
                and s.targets[0].attr == attr)
    pred.desc = f"`{obj}.{attr} = ...`"
    return pred


def assign_name(name):
    """匹配 `name = ...`（含 AnnAssign）。"""
    def pred(s):
        if isinstance(s, ast.Assign) and len(s.targets) == 1 \
                and isinstance(s.targets[0], ast.Name):
            return s.targets[0].id == name
        return isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) \
            and s.target.id == name
    pred.desc = f"`{name} = ...`"
    return pred


def if_test(*fragments):
    """匹配 `if <test>:`，其 test 源码含全部 fragments。"""
    def pred(s):
        if not isinstance(s, ast.If):
            return False
        txt = ast.unparse(s.test)
        return all(f in txt for f in fragments)
    pred.desc = "`if " + " … ".join(fragments) + ":`"
    return pred


def if_test_exact(text):
    """匹配 `if <test>:` 且 test 源码**恰好等于** text。

    子串匹配不够用的场合：`warm.bias_converged` 在 run_all_windows 循环体里
    出现 6 次，其中 5 次是 `not warm.bias_converged and ...`，只有目标那条
    test 恰好就是它。
    """
    def pred(s):
        return isinstance(s, ast.If) and ast.unparse(s.test) == text
    pred.desc = f"`if {text}:`（test 精确相等）"
    return pred


def call_to(dotted):
    """匹配独立语句 `dotted(...)`。"""
    def pred(s):
        return (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                and ast.unparse(s.value.func) == dotted)
    pred.desc = f"`{dotted}(...)`"
    return pred


def for_target(name):
    """匹配 `for <name> in ...:`（目标为单个裸名）。"""
    def pred(s):
        return (isinstance(s, ast.For) and isinstance(s.target, ast.Name)
                and s.target.id == name)
    pred.desc = f"`for {name} in ...:`"
    return pred


def stmt_type(kind):
    """匹配某种语句类型（如 ast.Try / ast.For）。"""
    def pred(s):
        return isinstance(s, kind)
    pred.desc = f"`{kind.__name__}` 语句"
    return pred


# ─────────────────────────────── 作用域分析 ───────────────────────────────

def bound_names(stmts):
    """这些语句里被绑定过的名字（含 import / 形参 / except as / global）。"""
    b = set()
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                b.add(n.id)
            elif isinstance(n, ast.arg):
                b.add(n.arg)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                b.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    b.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, ast.ExceptHandler) and n.name:
                b.add(n.name)
    return b


def _inner_scopes(stmts):
    """收集嵌套作用域（def/lambda/推导式）及其各自绑定的名字。"""
    scopes = []

    def walk(node):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) \
                    or hasattr(ch, "generators"):
                scopes.append(ch)
            walk(ch)

    for s in stmts:
        walk(s)
    out = []
    for sc in scopes:
        b = {x.arg for x in ast.walk(sc) if isinstance(x, ast.arg)}
        b |= {x.id for x in ast.walk(sc) if isinstance(x, ast.Name)
              and isinstance(x.ctx, (ast.Store, ast.Del))}
        b |= {x.name for x in ast.walk(sc)
              if isinstance(x, ast.ExceptHandler) and x.name}
        out.append((sc, b))
    return out


def outer_loads(stmts):
    """**方法作用域**视角读到的名字——推导式/内嵌 def 的局部名不算（幻影逃逸）。"""
    inner = _inner_scopes(stmts)
    out = set()
    for n in ast.walk(ast.Module(body=list(stmts), type_ignores=[])):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if not any(sc.lineno <= n.lineno <= sc.end_lineno and n.id in b
                       for sc, b in inner):
                out.add(n.id)
    return out


def outer_writes(stmts):
    """方法作用域视角写到的名字（同样排除内层作用域）。"""
    inner_bound = set()
    for sc, b in _inner_scopes(stmts):
        inner_bound |= b
    return bound_names(stmts) - inner_bound


# ─────────────────────────────── 引擎本体 ───────────────────────────────

def read_bytes(p):
    """保字节读（CRLF 保真——`Path.read_text` 会把 \\r\\n 洗成 \\n，栽过三次）。"""
    with open(p, encoding="utf-8", newline="") as fh:
        return fh.read()


def write_bytes(p, text):
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def find_method(tree, klass, fn):
    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == klass), None)
    if cls is None:
        sys.exit(f"FAIL: 找不到类 {klass}")
    m = next((n for n in cls.body
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == fn), None)
    if m is None:
        sys.exit(f"FAIL: 找不到 {klass}.{fn}")
    return cls, m


def append_to_class(text, klass, block):
    """把 block（缩进好的方法文本）追加到类 klass 体末尾。

    类在文件末尾（历史 domains 切片）⟹ 与旧执行器「文件尾 + 空行 + block」逐字节同口径；
    类后还有别的顶层定义（主线 abfe_pipeline.py 里 ABFEPipeline 后跟
    TraditionalABFEPipeline）⟹ 插在类末行之后、原有分隔空行之前。
    """
    lines = text.splitlines(keepends=True)
    cls = next(n for n in ast.parse(text).body
               if isinstance(n, ast.ClassDef) and n.name == klass)
    rest = lines[cls.end_lineno:]
    if not any(ln.strip() for ln in rest):
        if not text.endswith("\n"):
            text += "\n"
        return text + "\n" + block
    nl = "\r\n" if lines[cls.end_lineno - 1].endswith("\r\n") else "\n"
    return "".join(lines[:cls.end_lineno]) + nl + block.rstrip("\r\n") + nl + "".join(rest)


def body_no_doc(node):
    b = list(node.body)
    if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
            and isinstance(b[0].value.value, str):
        b = b[1:]
    return b


def locate(stmts, start_pred, end_pred, spec_name):
    """结构锚定位，返回 (lo, hi)；命中数 ≠ 1 或次序颠倒一律 fail-closed。"""
    starts = [i for i, s in enumerate(stmts) if start_pred(s)]
    if end_pred is None:
        # 无终点锚 = 搬到容器末尾（_finish_production：for-up 起直到方法体尾）
        if len(starts) != 1:
            sys.exit(f"FAIL[{spec_name}]: 起点锚 {getattr(start_pred, 'desc', '?')} "
                     f"命中 {len(starts)} 处（应为 1）")
        return starts[0], len(stmts)
    ends = [i for i, s in enumerate(stmts) if end_pred(s)]
    sd = getattr(start_pred, "desc", "<起点锚>")
    ed = getattr(end_pred, "desc", "<终点锚>")
    if len(starts) != 1:
        sys.exit(f"FAIL[{spec_name}]: 起点锚 {sd} 命中 {len(starts)} 处（应为 1）\n"
                 f"  → 主线结构已变或前置刀未跑。人工核对后更新 spec 的锚点，"
                 f"不要改成按下标定位。")
    if len(ends) != 1:
        sys.exit(f"FAIL[{spec_name}]: 终点锚 {ed} 命中 {len(ends)} 处（应为 1）\n"
                 f"  → 同上。")
    if starts[0] >= ends[0]:
        sys.exit(f"FAIL[{spec_name}]: 终点锚下标 {ends[0]} 不在起点锚 {starts[0]} 之后")
    return starts[0], ends[0]


def _container_body(method, spec):
    """搬移区所在的语句序列。

    spec["container"]:
      "fn"（默认）—— 方法体顶层（剥 docstring）
      "loop"        —— 方法体里**唯一**的 For/While 的体。`run_all_windows`
                       那条线的四刀都切在窗口 for 循环体内，不是方法体顶层。
    """
    body = body_no_doc(method)
    kind = spec.get("container", "fn")
    if kind == "fn":
        return body, None
    loops = [s for s in body if isinstance(s, (ast.For, ast.While))]
    if len(loops) != 1:
        sys.exit(f"FAIL[{spec['name']}]: container='loop' 但 {spec['fn']} 顶层有 "
                 f"{len(loops)} 个循环（应为 1）——主线结构已变")
    return loops[0].body, loops[0]


def check_moved(moved, region, spec):
    """搬移区的 fail-closed 检查（照抄各特化执行器里被实践验证过的那几条）。"""
    name = spec["name"]
    # stmts 可写成集合：历史快照值 + 人工核过的主线漂移值（spec 里逐值写来源），
    # 与 drift_ok 同一口径——历史回归仍只命中历史值。
    want_n = spec.get("stmts")
    if want_n is not None and len(moved) not in (
            set(want_n) if isinstance(want_n, (set, frozenset, tuple, list)) else {want_n}):
        sys.exit(f"FAIL[{name}]: 搬移语句 {len(moved)} 条 ≠ 预期 {want_n}")
    span = moved[-1].end_lineno - moved[0].lineno + 1
    if "span" in spec:
        lo, hi = spec["span"]
        if not (lo <= span <= hi):
            sys.exit(f"FAIL[{name}]: 搬移行数 {span} 偏离实测区间 {spec['span']}")
    want = spec.get("region_indent") or (12 if spec.get("container") == "loop"
                                         else spec.get("indent", 8))
    for ln in region:
        if ln.strip() and not ln.startswith(" " * want):
            sys.exit(f"FAIL[{name}]: 搬移区缩进异常（期望 {want} 空格）：{ln[:60]!r}")
    if any('"""' in ln or "'''" in ln for ln in region):
        sys.exit(f"FAIL[{name}]: 搬移区含三引号——组装 docstring 会打架")
    mod = ast.Module(body=list(moved), type_ignores=[])
    for n in ast.walk(mod):
        if isinstance(n, (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom)):
            sys.exit(f"FAIL[{name}]: {type(n).__name__} @ L{n.lineno}——搬走会改语义")
        if (isinstance(n, ast.Return) and not spec.get("allow_return")
                and not any(sc.lineno <= n.lineno <= sc.end_lineno
                            for sc, _ in _inner_scopes(list(moved)))):
            # 只拦**方法级** Return；内嵌 def / lambda 里的 return 属于它自己，
            # 随整块搬走语义不变。
            sys.exit(f"FAIL[{name}]: 方法级 Return @ L{n.lineno}"
                     f"（如确需 return 协议，spec 里显式开 allow_return 并补相应的门）")


def compute_params(moved, pre, method, spec):
    """重算跨界形参并与 spec 对账——漂移即停。"""
    name = spec["name"]
    exempt = set(spec.get("exempt", ()))
    pparams = {a.arg for a in method.args.args + method.args.kwonlyargs}
    pparams |= set(spec.get("extra_visible", ()))
    visible = bound_names(pre) | pparams
    got = sorted((outer_loads(moved) & visible) - BUILTINS - {"self"} - exempt)
    want = sorted(spec["params"])
    # drift_ok：人工核过的主线漂移名（spec 里逐名写来源）。差异全落在其中才放行，
    # 并打印出来；历史快照上不出现 ⟹ 历史回归不受影响。
    diff = set(got) ^ set(want)
    if diff and diff <= set(spec.get("drift_ok", ())):
        print(f"  [漂移·已核] {name}: 多 {sorted(set(got) - set(want))} "
              f"少 {sorted(set(want) - set(got))}")
        want = got
    if got != want:
        sys.exit(f"FAIL[{name}]: 参数表漂移\n"
                 f"  多出 {sorted(set(got) - set(want))}\n"
                 f"  缺少 {sorted(set(want) - set(got))}\n"
                 f"  → 主线改动影响了搬移区的依赖面，人工核对后更新 spec['params']")
    return got


def check_escapes(moved, post, spec):
    """写逃逸：搬移区写、后段读的名字。非空则必须走 return 协议（本引擎不默认支持）。"""
    exempt = set(spec.get("exempt", ()))
    esc = sorted((outer_writes(moved) & outer_loads(post)) - exempt)
    if esc:
        sys.exit(f"FAIL[{spec['name']}]: 写逃逸 {esc}\n"
                 f"  → 这些名字在搬移区被写、在后段被读，纯搬运会丢值。\n"
                 f"  需要 return 协议 + 对应的门（见 PROGRESS「Cut A 阻塞 2」）。")


def run_extract(spec, root, dry_run):
    """模式 A：把一段语句原样搬成同类的新方法，原位换成 keyword 调用。"""
    name = spec["name"]
    path = pathlib.Path(root) / spec["file"]
    src = read_bytes(path)
    nl = "\r\n" if src.count("\r\n") * 2 > src.count("\n") else "\n"
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    _, method = find_method(tree, spec["klass"], spec["fn"])
    fbody, container = _container_body(method, spec)

    if "start_after" in spec:
        # 起点 = 某个调用语句的**下一条**。用于「前一刀留下的调用点之后」这种
        # 天然锚（比拿 print(...) 之类不唯一的语句当锚可靠）。
        k = [i for i, s in enumerate(fbody) if spec["start_after"](s)]
        if len(k) != 1:
            sys.exit(f"FAIL[{name}]: start_after 锚 "
                     f"{getattr(spec['start_after'], 'desc', '?')} 命中 {len(k)} 处"
                     f"（应为 1）——前置刀未跑或主线结构已变")
        lo = k[0] + 1
        hi = locate(fbody, lambda s: s is fbody[lo], spec["end"], name)[1]
    elif spec.get("single"):
        # 搬移区恰为起点锚那一条语句（如整条 While / 整条 If）。
        k = [i for i, s in enumerate(fbody) if spec["start"](s)]
        if len(k) != 1:
            sys.exit(f"FAIL[{name}]: 起点锚 "
                     f"{getattr(spec['start'], 'desc', '?')} 命中 {len(k)} 处（应为 1）")
        lo, hi = k[0], k[0] + 1
    else:
        lo, hi = locate(fbody, spec["start"], spec.get("end"), name)
    if "slice_expect" in spec and (lo, hi) != tuple(spec["slice_expect"]):
        print(f"  [漂移] {name}: 下标 {tuple(spec['slice_expect'])} -> ({lo}, {hi})"
              f"；锚点仍唯一命中，按新下标继续")

    moved = fbody[lo:hi]
    # 首条语句的**前导注释**是否一并搬走。AST 里注释不是节点，`moved[0].lineno`
    # 指向语句本身，默认切法会把注释块留在调用点。
    # 默认 False（= 历史各执行器的口径）；`produce_window` 那刀显式开 True，
    # 因为那三行注释解释的正是被搬走的生产段。改这个开关会改产物字节。
    first_ln = moved[0].lineno
    if spec.get("take_leading_comments", False):
        col = len(lines[first_ln - 1]) - len(lines[first_ln - 1].lstrip())
        k = first_ln - 1
        while k - 1 >= 0:
            prev = lines[k - 1]
            st = prev.strip()
            if st.startswith("#") and (len(prev) - len(prev.lstrip())) == col:
                k -= 1
            else:
                break
        first_ln = k + 1
    region = lines[first_ln - 1: moved[-1].end_lineno]
    check_moved(moved, region, spec)
    # container='loop' 时，调用方可见的前段 = 方法体前段 + 循环体内前段
    pre = fbody[:lo]
    extra_vis = set(spec.get("extra_visible", ()))
    if container is not None:
        mb = body_no_doc(method)
        pre = mb[:mb.index(container)] + pre
        # 循环的**目标变量**（for window_idx, (start, end) in …）由 For 语句
        # 自身绑定，不在 pre 里；但对循环体是可见的，必须算进形参候选。
        extra_vis |= {n.id for n in ast.walk(container.target)
                      if isinstance(n, ast.Name)}
    spec = dict(spec, extra_visible=tuple(extra_vis))
    params = compute_params(moved, pre, method, spec)
    check_escapes(moved, fbody[hi:], spec)

    # 缩进有两个层级，不能混：
    #   call_indent —— 调用点所在层（= 搬移区在原文里的缩进）
    #   body_indent —— 新方法体所在层（类内方法固定 8）
    # container='loop' 时两者差 4，必须去缩进；container='fn' 时相等，
    # **逐字搬运不去缩进**（_finish_production 那次误套 12→8 模板把文件写坏过）。
    call_indent = region[0][:len(region[0]) - len(region[0].lstrip())]
    body_indent = " " * spec.get("indent", 8)
    dedent = len(call_indent) - len(body_indent)
    if dedent < 0:
        sys.exit(f"FAIL[{name}]: 搬移区缩进 {len(call_indent)} 浅于方法体 "
                 f"{len(body_indent)}——组装会把方法体截断")
    if dedent:
        region = [(ln[dedent:] if ln.strip() else ln) for ln in region]
    indent = body_indent
    # 多行 keyword-only 签名：形参多到 30+ 时单行不可读，且逐参一行让
    # 后续「某参数还在不在」的 diff 只动一行。
    sig = (f"{indent[:4]}def {spec['new']}({nl}"
           f"{indent}self,{nl}{indent}*,{nl}"
           + "".join(f"{indent}{p},{nl}" for p in params)
           + f"{indent[:4]}):{nl}")
    # doc_text：spec 只存 docstring **内容**，缩进与引号由引擎渲染。
    # （旧的 doc= 存的是渲染好的行，容易在 repr 转义上出错，保留兼容。）
    if "doc_text" in spec:
        dl = spec["doc_text"].rstrip("\n").split("\n")
        doc = (f'{indent}"""{dl[0]}{nl}'
               + "".join((f"{indent}{x}{nl}" if x.strip() else nl) for x in dl[1:])
               + f'{indent}"""{nl}')
    else:
        doc = spec.get("doc", "")
    body_text = "".join(region)          # 逐字搬运，不去缩进（缩进模板误套栽过）
    new_method = sig + doc + body_text

    call = (call_indent + f"self.{spec['new']}({nl}"
            + "".join(f"{call_indent}    {p}={p},{nl}" for p in params)
            + call_indent + ")" + nl)

    # ⚠️ 插入文本必须**按行拆开**再进列表——多行塞进单个元素会让后续按行号
    #    索引全错（GUIDE §五坑表第 6 条，本引擎首次跑就栽了一次）。
    out = (lines[:first_ln - 1] + call.splitlines(keepends=True)
           + lines[moved[-1].end_lineno:])

    # 新方法的落点。默认 class_end：**不移动任何既有方法的位置**，
    # 因此同一文件上连续下刀时后续刀的锚点定位不受影响（历史上 16 个特化
    # 执行器用的就是这个口径，改成 after_fn 会与既有产物逐字节不符）。
    tail = (nl if spec.get("blank_line", True) else "") + new_method
    place = spec.get("place", "class_end")
    t_out = ast.parse("".join(out))
    cls2, m2 = find_method(t_out, spec["klass"], spec["fn"])
    at = cls2.end_lineno if place == "class_end" else m2.end_lineno
    if place == "class_end":
        # 越过类末尾紧随的空行——那些空行属于文件尾，新方法要插在它们之后，
        # 否则产物里空行会跑到文件最末（与历史产物差两行）。
        while at < len(out) and not out[at].strip():
            at += 1
    out = out[:at] + tail.splitlines(keepends=True) + out[at:]
    text = "".join(out)

    try:
        ast.parse(text)
    except SyntaxError as e:
        sys.exit(f"FAIL[{name}]: 组装结果语法错误 {e}——未写盘")
    # 结构自检：新方法必须在类内，且类体不得出现浮动语句
    t2 = ast.parse(text)
    cls2, _ = find_method(t2, spec["klass"], spec["fn"])
    names2 = {n.name for n in cls2.body
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if spec["new"] not in names2:
        sys.exit(f"FAIL[{name}]: 新方法 {spec['new']} 不在类 {spec['klass']} 内"
                 f"——多半是缩进模板套错（栽过：方法体比 docstring 还浅，"
                 f"解析器把函数体截断，ast.parse 仍合法）")
    floating = [n for n in cls2.body
                if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.Assign, ast.AnnAssign, ast.Expr, ast.Pass))]
    if floating:
        sys.exit(f"FAIL[{name}]: 类体出现浮动语句 "
                 f"{[type(n).__name__ for n in floating]}——同上")

    if dry_run:
        print(f"  [dry-run] {name}: 搬 {len(moved)} 条 / {len(region)} 行，"
              f"形参 {len(params)}；未写盘")
        return
    write_bytes(path, text)
    print(f"  [写盘] {name}: {spec['fn']} 搬出 {len(moved)} 条 → {spec['new']}"
          f"（{len(params)} 形参）")
    print(f"  验：python3 refactor/tools/verify_extract.py <留底> {path} "
          f"--src {spec['fn']} --moved {spec['new']} --block fn --slice {lo}:{hi}")


def _rewrite_names_in_text(lines, targets, bind, newline):
    """行手术：在每个目标 Name 前插入 `bind.`。

    **为什么是行手术而不是 unparse 整棵树**：unparse 会抹掉注释、空行、
    原有的换行与引号风格——那对一个 5000 行的方法是灾难。行手术只动
    被点名的那几列，其余字节原样不动。

    按 (行, 列) **降序**处理：先改后面的，前面的列偏移才不受影响。

    ⚠️ **`ast` 的 col_offset 是 UTF-8 字节偏移，不是字符偏移。** 本仓库的
    注释与日志里全是中文，一行里只要前面有中文，字节偏移就远大于字符偏移，
    直接拿它索引 str 会插到错误位置（表现为莫名其妙的 SyntaxError，
    且错误行往往离真正被改坏的地方很远）。必须先转成字符偏移。
    """
    def _char_col(row, bcol):
        return len(row.encode("utf-8")[:bcol].decode("utf-8", errors="ignore"))

    out = list(lines)
    for ln, col in sorted(targets, reverse=True):
        row = out[ln - 1]
        c = _char_col(row, col)
        out[ln - 1] = row[:c] + bind + "." + row[c:]
    return out


def _collect_name_sites(node, names, inner_shadowed):
    """方法子树里所有该改写的 Name 节点位置（含 f-string 内——py3.12 是真实节点）。"""
    sites = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in names:
            if (n.lineno, n.col_offset) in inner_shadowed:
                continue
            sites.append((n.lineno, n.col_offset))
    return sites


def run_absorb(spec, root, dry_run):
    """模式 B：把散落的名字收进一个数据类对象。

    两种口径（spec["in_place"]）：
      True  —— 初始化语句留在原地，变成 `bind.x = ...`；字段一律 `Any = None`
               （默认值不可观测，前提是首写都在体顶层无条件执行）。
      False —— 初始化语句被删除，初值成为字段默认值（WarmupState 口径）。
    """
    name = spec["name"]
    path = pathlib.Path(root) / spec["file"]
    src = read_bytes(path)
    # 行尾按**文件主导口径**，不硬编码。历史上 _pipeline_state_absorb 把 40 行
    # CRLF 写进了一个纯 LF 文件，产物混合行尾——那是缺陷，本引擎不复刻。
    # 代价：那几刀与历史产物的对账要用「行尾归一后逐字节相同」，不是裸 cmp。
    crlf = src.count("\r\n")
    newline = "\r\n" if crlf * 2 > src.count("\n") else "\n"
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    bind, dcls = spec["bind"], spec["dataclass"]
    names = set(spec["names"])

    existing = next((n for n in tree.body
                     if isinstance(n, ast.ClassDef) and n.name == dcls), None)
    extend = spec.get("extend", False)
    if existing is not None and not extend:
        sys.exit(f"FAIL[{name}]: 数据类 {dcls} 已存在——重复收编？"
                 f"（若本刀是**往已有数据类追加字段**，spec 里设 extend=True）")
    if existing is None and extend:
        sys.exit(f"FAIL[{name}]: extend=True 但数据类 {dcls} 不存在"
                 f"——前置刀未跑")
    _, method = find_method(tree, spec["klass"], spec["fn"])
    if not extend and any(isinstance(n, ast.Name) and n.id == bind
                          for n in ast.walk(method)):
        sys.exit(f"FAIL[{name}]: 承载名 {bind!r} 在 {spec['fn']} 里已被占用"
                 f"——换一个名字（历史教训：`state` 被 OpenMM getState 占着）")

    body = body_no_doc(method)

    # ① 首写核验：每个名字的首个 Store 必须是**体顶层无条件直赋**
    first_store = {}
    for nm in sorted(names):
        stores = sorted(n.lineno for n in ast.walk(method)
                        if isinstance(n, ast.Name) and n.id == nm
                        and isinstance(n.ctx, ast.Store))
        if not stores:
            sys.exit(f"FAIL[{name}]: {nm} 在 {spec['fn']} 里没有 Store——名单过期")
        fs = stores[0]
        owner = next((s for s in body if s.lineno <= fs <= s.end_lineno), None)
        toplevel = (owner is not None and owner.lineno == fs
                    and isinstance(owner, (ast.Assign, ast.AnnAssign)))
        if not toplevel:
            # 首写在嵌套块里 ⟹ 存在绕过它的路径 ⟹ 字段默认值**可能被观测**。
            # 机械判据到此为止：要放行必须在 spec 里给出**书面理由**，
            # 逼使用者把「为什么这条路径上默认值仍不可观测」写下来，
            # 而不是让脚本默默放过。
            why = (spec.get("proven_conditional") or {}).get(nm)
            if not why:
                sys.exit(
                    f"FAIL[{name}]: {nm} 的首写(L{fs})不是体顶层无条件直赋\n"
                    f"  → 存在绕过首写的路径，原地收编后默认值可被观测"
                    f"（原代码在该路径上是 UnboundLocalError，收编后变成 None）。\n"
                    f"  若已证明该路径不可达或随后必被无条件覆盖，在 spec 里写：\n"
                    f"      proven_conditional={{\"{nm}\": \"<理由>\"}}")
            print(f"  [豁免] {nm}: {why}")
        first_store[nm] = fs

    # ② 遮蔽检查：内层作用域（推导式/内嵌 def）里同名的局部绑定不能改写
    inner_shadowed = set()
    for sc, bound in _inner_scopes(body):
        for n in ast.walk(sc):
            if isinstance(n, ast.Name) and n.id in names and n.id in bound:
                inner_shadowed.add((n.lineno, n.col_offset))
    if inner_shadowed and not spec.get("allow_shadow"):
        sys.exit(f"FAIL[{name}]: {len(inner_shadowed)} 处名字被内层作用域遮蔽"
                 f"——改写会串味。核实后在 spec 里开 allow_shadow。")

    # ③ 行手术
    sites = _collect_name_sites(method, names, inner_shadowed)
    out = _rewrite_names_in_text(lines, sites, bind, newline)

    # ④ 构造语句：插在最早的那条首写之前（原地口径）。
    #    extend 刀不插——`bind = Dcls()` 是前置刀留下的，本刀只追加字段。
    if not extend:
        ctor_at = min(first_store.values())
        indent = out[ctor_at - 1][:len(out[ctor_at - 1])
                                  - len(out[ctor_at - 1].lstrip())]
        out.insert(ctor_at - 1, f"{indent}{bind} = {dcls}(){newline}")

    # ⑤ 数据类声明（模块头部，紧跟 import 区之后）
    fields = spec.get("fields") or {nm: "Any" for nm in sorted(names)}
    defaults = spec.get("defaults", {})
    order = spec.get("field_order") or sorted(names)
    if extend:
        decl = [f"    {nm}: {fields.get(nm, 'Any')} = "
                f"{defaults.get(nm, 'None')}{newline}" for nm in order]
        at = existing.end_lineno           # 追加到已有类体末尾
    else:
        at = _decl_insert_line(tree, spec)
        decl = _render_dataclass(spec, names, fields, newline)
        if at == 0 and decl and decl[0] == newline:
            # 插在文件首行：不要前置空行，改在尾部补一条作与后文的分隔
            # （插在别处时，`out[at:]` 原有的空行就是分隔，不必补）。
            decl = decl[1:] + [newline]
        elif not spec.get("dataclass_after"):
            decl = decl + [newline, newline]   # 与紧随其后的目标类隔两行（主线口径）
    out = out[:at] + decl + out[at:]

    # ⑤' 形参播种：本刀收编的名字里若有**原方法形参**，它进入方法时已经带值，
    #     必须在构造点把该值播进去，否则收编后读到的是字段默认 None。
    #     （`stage1_states` 就是这个形状：P-A 读参数值、P-B 重绑、P-D 读重绑值。）
    seeds = list(spec.get("seed_from_param", ()))
    if seeds:
        pnames = {a.arg for a in method.args.args + method.args.kwonlyargs}
        bad = [s for s in seeds if s not in pnames]
        if bad:
            sys.exit(f"FAIL[{name}]: seed_from_param 里的 {bad} 不是 "
                     f"{spec['fn']} 的形参——名单过期或前置刀已改签名")
        kw = ", ".join(f"{s}={s}" for s in seeds)
        needle = f"{bind} = {dcls}()"
        # 整句精确匹配——数据类 docstring 里往往也写着 `dual = PipelineState()`，
        # 子串匹配会把那行也算上（本引擎第一次跑就撞了）。
        hits = [i for i, ln in enumerate(out) if ln.strip() == needle]
        if len(hits) != 1:
            sys.exit(f"FAIL[{name}]: 构造语句 `{needle}` 命中 {len(hits)} 处（应为 1）")
        out[hits[0]] = out[hits[0]].replace(needle, f"{bind} = {dcls}({kw})")

    text = "".join(out)
    try:
        t2 = ast.parse(text)
    except SyntaxError as e:
        sys.exit(f"FAIL[{name}]: 组装结果语法错误 {e}——未写盘")

    # ⑥ 双向验证：期望树 = 原树深拷贝做同一变换，与行手术结果逐节点比对。
    #    这道自检抓过 CRLF 双回车、AnnAssign simple 镜像等一批坑。
    expect = _expected_tree(src, spec, names, bind, dcls, fields, spec.get("defaults", {}))
    if ast.dump(_norm(t2)) != ast.dump(_norm(expect)):
        sys.exit(f"FAIL[{name}]: 行手术结果与期望树不符——未写盘（引擎自身有 bug，"
                 f"或主线结构与 spec 不匹配）")

    n_cls = len([n for n in t2.body if isinstance(n, ast.ClassDef) and n.name == dcls])
    if n_cls != 1:
        sys.exit(f"FAIL[{name}]: 数据类 {dcls} 出现 {n_cls} 次")

    if dry_run:
        print(f"  [dry-run] {name}: {len(names)} 字段 / 改写 {len(sites)} 处；未写盘")
        return
    write_bytes(path, text)
    print(f"  [写盘] {name}: {dcls} {len(names)} 字段，{spec['fn']} 内改写 "
          f"{len(sites)} 处 → {bind}.*")
    ip = "--in-place " if spec.get("in_place", True) else ""
    print(f"  验：python3 refactor/tools/verify_rename.py <留底> {path} "
          f"--fn {spec['fn']} --bind {bind} --dataclass {dcls} "
          f"--names {','.join(sorted(names))} {ip}--construction 1")


def _render_dataclass(spec, names, fields, newline):
    """渲染数据类声明（含 docstring 与空行布局）。

    布局与历史产物一致：前置空行 → 装饰器 → class → 可选 docstring → 空行 →
    字段逐行。末尾**不**补空行（`out[at:]` 原有的那条空行就是分隔）。
    """
    dcls = spec["dataclass"]
    out = [newline,
           f"{spec.get('decorator', '@dataclass')}{newline}",
           f"class {dcls}:{newline}"]
    doc = spec.get("dataclass_doc")
    if doc:
        body = doc.strip("\n").split("\n")
        out.append(f'    """{body[0]}{newline}')
        for ln in body[1:]:
            out.append((f"    {ln}" if ln.strip() else "") + newline)
        out.append(f'    """{newline}')
        out.append(newline)
    defaults = spec.get("defaults", {})
    for nm in sorted(names):
        out.append(f"    {nm}: {fields.get(nm, 'Any')} = "
                   f"{defaults.get(nm, 'None')}{newline}")
    return out


def _norm(tree):
    for n in ast.walk(tree):
        for a in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(n, a):
                try:
                    delattr(n, a)
                except AttributeError:
                    setattr(n, a, None)
    return tree


class _Bind(ast.NodeTransformer):
    """AST 侧的同一变换，用于造期望树。"""

    def __init__(self, names, bind):
        self.names, self.bind = names, bind

    def visit_Name(self, node):
        if node.id in self.names:
            return ast.copy_location(
                ast.Attribute(value=ast.Name(id=self.bind, ctx=ast.Load()),
                              attr=node.id, ctx=node.ctx), node)
        return node

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        # `x: T = v` → `prod.x: T = v` 之后目标不再是裸名，解析器口径 simple=0
        if isinstance(node.target, ast.Attribute):
            node.simple = 0
        return node


def _expected_tree(src, spec, names, bind, dcls, fields, defaults):
    tree = ast.parse(src)
    _, method = find_method(tree, spec["klass"], spec["fn"])
    body = body_no_doc(method)
    first = min(min(n.lineno for n in ast.walk(method)
                    if isinstance(n, ast.Name) and n.id == nm
                    and isinstance(n.ctx, ast.Store))
                for nm in names)
    _Bind(names, bind).visit(method)
    ast.fix_missing_locations(method)
    if not spec.get("extend"):
        owner_i = next(i for i, s in enumerate(method.body)
                       if s.lineno <= first <= s.end_lineno)
        method.body.insert(owner_i, ast.parse(f"{bind} = {dcls}()").body[0])
    # 与行手术走**同一个渲染函数**——两条路径分别拼装过一次 docstring，
    # 缩进口径不同导致字符串值不等，被这道双向验证抓了个正着。
    if spec.get("extend"):
        cls = next(n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name == dcls)
        order = spec.get("field_order") or sorted(names)
        for nm in order:
            cls.body.append(ast.parse(
                f"{nm}: {fields.get(nm, 'Any')} = "
                f"{defaults.get(nm, 'None')}").body[0])
        ast.fix_missing_locations(cls)
        seeds = list(spec.get("seed_from_param", ()))
        if seeds:
            for n in ast.walk(method):
                if (isinstance(n, ast.Assign) and len(n.targets) == 1
                        and isinstance(n.targets[0], ast.Name)
                        and n.targets[0].id == bind
                        and isinstance(n.value, ast.Call)):
                    n.value.keywords = [
                        ast.keyword(arg=s, value=ast.Name(id=s, ctx=ast.Load()))
                        for s in seeds]
            ast.fix_missing_locations(method)
        return tree
    decl = "".join(_render_dataclass(spec, names, fields, "\n"))
    at = _decl_insert_line(ast.parse(src), spec)
    tree.body.insert(_body_index_for_line(tree, at), ast.parse(decl).body[0])
    return tree


def _decl_insert_line(tree, spec):
    """数据类声明插在哪一行（0-based 行索引）——**结构定位，不写行号**。

    spec["dataclass_after"] 给一个已存在的顶层定义名，声明插在它之后；
    不给则插在**目标类之前**（历史 domains 文件里类在第 1 行 ⟹ 等于文件头；
    主线文件的 import 可能散落到万行之后，「最后一条 import 之后」会落到类后面）。
    """
    after = spec.get("dataclass_after")
    if after:
        node = next((n for n in tree.body
                     if getattr(n, "name", None) == after), None)
        if node is None:
            sys.exit(f"FAIL[{spec['name']}]: dataclass_after 指定的 {after!r} "
                     f"不在模块顶层——前置刀未跑，或主线结构已变")
        return node.end_lineno
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == spec["klass"])
    return min([d.lineno for d in cls.decorator_list] + [cls.lineno]) - 1


def _body_index_for_line(tree, line):
    for i, s in enumerate(tree.body):
        if s.lineno > line:
            return i
    return len(tree.body)


MODES = {"extract": run_extract, "absorb": run_absorb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="spec 模块路径（定义 SPEC 字典）")
    ap.add_argument("--root", default="refactor/domains", help="目标树根目录")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sp = pathlib.Path(a.spec)
    m = importlib.util.spec_from_file_location(sp.stem, sp)
    mod = importlib.util.module_from_spec(m)
    m.loader.exec_module(mod)
    spec = getattr(mod, "SPEC", None)
    if spec is None:
        sys.exit(f"FAIL: {sp} 里没有 SPEC")
    mode = spec.get("mode", "extract")
    if mode not in MODES:
        sys.exit(f"FAIL: 未知模式 {mode}（可用：{sorted(MODES)}）")
    MODES[mode](spec, a.root, a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
