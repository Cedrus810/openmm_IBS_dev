#!/usr/bin/env python3
"""刀3第0步执行器 v2（CUTTING_PLAN.md §4.3）：_decide_once 的内嵌 def 逐个提为模块级。

v2 修订（实测驱动，门拦出来的）：
  - 捕获表按**传递闭包**重算：被调函数提升后签名变长，调用者必须把
    「被调者的捕获形参」收进自己的捕获并在调用点传入。
    plan = §4.3 的 3 个 + self + _frames_admission(6) + _is_noop(4) = 12；
    _bounded_rewindow = 自身 4 + plan 的 12 = 14（view 共享）。
  - 提升顺序改为被调者先行：_frames_admission → _is_noop → plan → _bounded_rewindow
    （§4.3 原顺序撞环：_bounded_rewindow 体内调 plan/self，plan 体内调
    _frames_admission/_is_noop/self —— 原表漏记，门实测拦出）。
  - view 统一首参（§4.3 约定）。
  - verify_extract --strip-call NAME=K：把 after 侧对已提函数调用的前 K 个插入
    实参剥掉再逐节点比对（与 §0② verify_rename 同一「还原式比对」思想）。

保结构变换：函数体逐字不变（仅去 8 格缩进）；捕获变量按原名当形参；
全文件调用点在 "(" 后插实参。fail-closed 不动文件。
"""
import argparse
import ast
import builtins
import os
import pathlib
import sys

# 重放（replay.py）用 CUT_FILE 指向目标文件；缺省＝历史 domains 路径。
FILE = pathlib.Path(os.environ["CUT_FILE"]) if os.environ.get("CUT_FILE") else pathlib.Path(__file__).resolve().parent / "stage2_controller" / "stage2_repair_controller.py"
KLASS = "Stage2RepairController"
DECIDE = "_decide_once"
MAINLINE_SOURCE = "abfe_preoptimizer.py"

# 提升后的新签名形参表（捕获参数，前缀位）。顺序 = 调用点插参顺序。
CAPTURES = {
    "_pw_or_last": [],
    "_attr_of": ["view"],
    "_is_skew": ["view"],
    "_pick": ["earliest"],
    "_frames_admission": ["view", "_blocks", "_blocks_total", "_blocks_total_u",
                          "_blocks_u", "_max_blocks"],
    "_is_noop": ["view", "_by_idx_for_noop", "_by_uid_for_noop", "_noop_led"],
    "plan": ["view", "_ctx", "feas", "self",
             "_blocks", "_blocks_total", "_blocks_total_u", "_blocks_u", "_max_blocks",
             "_by_idx_for_noop", "_by_uid_for_noop", "_noop_led"],
    "_bounded_rewindow": ["view", "blocked", "earliest", "self", "_ctx", "feas",
                          "_blocks", "_blocks_total", "_blocks_total_u", "_blocks_u",
                          "_max_blocks", "_by_idx_for_noop", "_by_uid_for_noop",
                          "_noop_led"],
}


# 主线漂移（人工核过）：只在该 def 体内**确实引用**时才追加到捕获表末尾。
# 历史快照上不引用 ⟹ 历史回归逐字节不变。追加项同时进所有调用点（含传递闭包：
# plan 已捕获 self，不必再动）。
CAPTURES_DRIFT = {
    # 09-22 复现模式：`getattr(self, "repro_frozen_path", False)`（L6481 附近），只读
    "_frames_admission": ["self"],
}


def _read(p):
    with open(p, encoding="utf-8", newline="") as fh:
        return fh.read()


def _write(p, text):
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def free_vars(fn):
    bound = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in n.names)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
    loads = {n.id for n in ast.walk(fn)
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return loads - bound - set(dir(builtins))


def insert_args(lines, call, args_txt):
    """在 call 的 "(" 之后插入 args_txt（fail-closed：括号必须紧随函数名）。"""
    line = lines[call.func.end_lineno - 1]
    # end_col_offset 是 UTF-8 字节偏移；本仓中文注释满屏，必须转字符偏移
    p = len(line.encode("utf-8")[:call.func.end_col_offset].decode("utf-8", errors="ignore"))
    if p >= len(line) or line[p] != "(":
        sys.exit(f"FAIL: 调用点 L{call.func.end_lineno} 括号不在预期位置：{line!r}")
    return (call.func.end_lineno - 1, p + 1, args_txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--def", dest="name", required=True, choices=sorted(CAPTURES))
    ap.add_argument("--fixup", action="append", default=[], metavar="NAME",
                    help="体内对已提函数 NAME 的调用点按其新签名前缀插实参（可多次）")
    args = ap.parse_args()
    name, caps = args.name, CAPTURES[args.name]
    for fx in args.fixup:
        if fx not in CAPTURES:
            sys.exit(f"FAIL: --fixup {fx} 不在 CAPTURES 表里")

    src = _read(FILE)
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)

    klass = next((n for n in tree.body
                  if isinstance(n, ast.ClassDef) and n.name == KLASS), None)
    method = next((n for n in (klass.body if klass else [])
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == DECIDE), None)
    if method is None:
        sys.exit(f"FAIL: 找不到 {KLASS}.{DECIDE}")
    nested = next((n for n in method.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == name), None)
    if nested is None:
        sys.exit(f"FAIL: {DECIDE} 里没有内嵌 def {name}（已提过？）")
    if nested.body[0].lineno == nested.lineno:
        sys.exit("FAIL: 一行式 def，本执行器不支持")

    # ① 捕获核实：自由变量 ∩ _decide_once 局部名 ⊆ 捕获表。
    #    其余自由变量 = 模块级名（本切片顶层 / 主线源文件顶层 / 已提函数），随接线解析。
    def _locals_of_method():
        loc = {a.arg for a in method.args.args + method.args.kwonlyargs}
        if method.args.vararg:
            loc.add(method.args.vararg.arg)
        if method.args.kwarg:
            loc.add(method.args.kwarg.arg)
        for n in ast.walk(method):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                loc.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                loc.add(n.name)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                loc.add(n.name)
        return loc

    fv = free_vars(nested)
    for extra in CAPTURES_DRIFT.get(name, []):
        if extra in fv and extra not in caps:
            print(f"  [漂移·已核] {name}: 捕获追加 {extra}")
            caps = caps + [extra]
            CAPTURES[name] = caps
    unexpected = (fv & _locals_of_method()) - set(caps)
    if unexpected:
        sys.exit(f"FAIL: {name} 有捕获表之外的 _decide_once 局部自由变量 "
                 f"{sorted(unexpected)}——先核 §4.3 传递闭包再动刀")
    unused = set(caps) & (fv - {n.id for n in ast.walk(nested)
                                if isinstance(n, ast.Name)}) if caps else set()
    if unused:
        print(f"⚠️ 捕获 {sorted(unused)} 实测未被体内引用；照加形参（保结构）")

    # ② 裸引用检查（非调用的 Load）。精确到作用域：引用若处于某个**绑定了该名**
    #    的函数作用域内（LEGB 遮蔽），提升后语义不变、放行；只有一路向外都
    #    绑不到、最终落到模块级的裸引用才危险（v1 全文件粗扫把 decide() 里
    #    的局部 `plan = ...` 变量误报了）。
    call_func_ids = {id(n.func) for n in ast.walk(tree)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                     and n.func.id == name}
    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    def _binds(fn):
        b = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        if fn.args.vararg:
            b.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            b.add(fn.args.kwarg.arg)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                b.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                b.add(n.name)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                b.add(n.name)
        return b

    bare = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Name) and n.id == name
                and isinstance(n.ctx, ast.Load) and id(n) not in call_func_ids):
            continue
        ln = n.lineno
        shadowed = any(f.lineno <= ln <= f.end_lineno and name in _binds(f)
                       for f in funcs)
        if not shadowed:
            bare.append(n)
    if bare:
        loc = ", ".join(f"L{n.lineno}" for n in bare[:5])
        sys.exit(f"FAIL: {name} 存在无遮蔽的裸引用（{loc}）——提升改变语义，先人工处理")

    # ③ 调用点必须全部落在 _decide_once 子树内
    decide_ids = {id(n) for n in ast.walk(method)}
    outside = [c for c in ast.walk(tree)
               if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
               and c.func.id == name and id(c) not in decide_ids]
    if outside:
        loc = ", ".join(f"L{c.lineno}" for c in outside[:5])
        sys.exit(f"FAIL: {name} 在 {DECIDE} 之外被调用（{loc}）")

    # ④ 组装模块级新 def
    start = min([d.lineno for d in nested.decorator_list] + [nested.lineno])
    end = nested.end_lineno
    indent = " " * nested.col_offset
    if nested.decorator_list:
        sys.exit("FAIL: 内嵌 def 带装饰器，本执行器未支持")
    header_lines = lines[start - 1: nested.body[0].lineno - 1]
    body_lines = lines[nested.body[0].lineno - 1: end]
    for ln in body_lines:
        if ln.strip() and not ln.startswith(indent):
            sys.exit(f"FAIL: 函数体缩进异常：{ln!r}")
    header = "".join(header_lines)
    if caps:
        p = header.index("(", header.index(f"def {name}"))
        has_args = bool(nested.args.args or nested.args.posonlyargs
                        or nested.args.kwonlyargs or nested.args.vararg
                        or nested.args.kwarg)
        ins = ", ".join(caps) + (", " if has_args else "")
        header = header[:p + 1] + ins + header[p + 1:]

    def dedent(txt):
        return "".join(ln[len(indent):] if ln.startswith(indent) else ln
                       for ln in txt.splitlines(keepends=True))

    new_def = dedent(header) + dedent("".join(body_lines))
    if not new_def.endswith("\n"):
        new_def += "\n"

    # ⑤ 调用点插实参（本 def 的全部调用点）
    calls = [c for c in ast.walk(tree)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
             and c.func.id == name]
    edits = []
    if caps:
        for c in calls:
            txt = ", ".join(caps) + (", " if (c.args or c.keywords) else "")
            edits.append(insert_args(lines, c, txt))
    for (l, c_, txt) in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        line = lines[l]
        lines[l] = line[:c_] + txt + line[c_:]

    # ⑥ fixup：本 def 体内对已提函数的调用点按新签名前缀插实参
    fixup_edits = []
    if args.fixup:
        nested_ids = {id(n) for n in ast.walk(nested)}
        for fx in args.fixup:
            fx_caps = CAPTURES[fx]
            targets = [c for c in ast.walk(nested)
                       if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                       and c.func.id == fx]
            for c in targets:
                txt = ", ".join(fx_caps) + (", " if (c.args or c.keywords) else "")
                fixup_edits.append(insert_args(lines, c, txt))
            print(f"  fixup {fx}: 体内 {len(targets)} 处调用点插 {len(fx_caps)} 实参")
    for (l, c_, txt) in sorted(fixup_edits, key=lambda e: (e[0], e[1]), reverse=True):
        line = lines[l]
        lines[l] = line[:c_] + txt + line[c_:]

    # ⑦ 删原块、尾部追加
    del lines[start - 1:end]
    # 落点：文件尾；主线文件尾若是 `if __name__ == "__main__":` 块，插到它之前
    # （否则脚本式运行时新 def 尚未定义）。历史 domains 文件无此块 ⟹ 口径不变。
    main_blk = [n for n in tree.body if isinstance(n, ast.If)
                and "__name__" in ast.unparse(n.test)]
    if main_blk and main_blk[-1] is tree.body[-1]:
        k = main_blk[-1].lineno - 1 - (end - start + 1)   # 原块已删，行号前移
        head = "".join(lines[:k]).rstrip("\n") + "\n"
        text = head + "\n\n" + new_def + "\n\n" + "".join(lines[k:])
    else:
        text = "".join(lines)
        if not text.endswith("\n"):
            text += "\n"
        text += "\n\n" + new_def
    _write(FILE, text)

    idx = [i for i, n in enumerate(method.body)
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))].index(
        next(i for i, n in enumerate(method.body)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == name))
    print(f"✅ {name}: 捕获形参 {len(caps)} 个，调用点改写 {len(edits)} 处，"
          f"fixup {len(fixup_edits)} 处，原块 L{start}-{end} 提至模块级；"
          f"verify --block def --index {idx}"
          + (f" + --strip-call {' '.join(f'{fx}={len(CAPTURES[fx])}' for fx in args.fixup)}"
             if args.fixup else ""))


if __name__ == "__main__":
    main()
