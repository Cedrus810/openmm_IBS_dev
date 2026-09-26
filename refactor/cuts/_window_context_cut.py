#!/usr/bin/env python3
"""D 线第二刀执行器：WindowContext 归组（6 个窗口级对象）。

实测（2026-09-20）：sampler/sim/win_sys/lc_win/lv_win/win_sys_xml——
  * 六名首写均在任何读之前（循环体内文本序核验），零内层遮蔽；
  * phase-1 两个早退 continue（L693/L830）跳过整窗 → 窗内默认值不可观测
    （每窗重建 wc，不跨窗泄漏）；
  * 清理协议随改写自动转换：del sampler → del wc.sampler（del 位点必然
    已绑定——均在 phase 1 之后）、sim/win_sys = None → wc.sim = None 等，
    gc.collect() 不动；warmup_only 路径的 win_sys = None 同样自动转换。
变换：
  ① 模块级 @dataclass(slots=True) WindowContext（6 字段 Any = None；
     可变——清理协议改写字段；每窗在循环体顶 `wc = WindowContext()` 重建）；
  ② 循环体 + 5 个方法体内 6 名全部引用改写 wc.<field>；
  ③ 5 个方法签名 6→1（kwonly 首删除位插 wc，kw_defaults 平行）；
  ④ 5 个调用点 kwarg 折 1（首删除位插 wc=wc）。
门：执行器双向验证 + 6 道 verify_rename（--in-place / run_all_windows 用
--block for --construction 1，含调用点的四道加 --expand-kwarg）+ py_compile +
layer_check。本文件 CRLF。
"""
import ast
import copy
import os
import pathlib
import sys

# 重放（replay.py）用 CUT_FILE 指向目标文件；缺省＝历史 domains 路径。
FILE = pathlib.Path(os.environ["CUT_FILE"]) if os.environ.get("CUT_FILE") else pathlib.Path(__file__).resolve().parent / "windows_sampling" / "window_manager_dual_lambda.py"
KLASS = "IBSWindowManagerDualLambda"
NAMES = ["lc_win", "lv_win", "sampler", "sim", "win_sys", "win_sys_xml"]
BIND = "wc"
DATACLASS = "WindowContext"
# (方法, 调用者) —— 调用者None=run_all_windows循环体
SIG_EDITS = ["_run_warmup_loop", "_summarize_warmup", "_commit_frozen_f_k",
             "_produce_window", "_finish_production"]
CALL_EDITS = {"_run_warmup_loop": "loop", "_summarize_warmup": "loop",
              "_commit_frozen_f_k": "loop", "_produce_window": "loop",
              "_finish_production": "_produce_window"}
FNS = ["run_all_windows"] + SIG_EDITS

DC_TEXT = '''@dataclass(slots=True)
class WindowContext:
    """窗口级对象簇（D 线归组第二刀，2026-09-20）。

    sampler/sim/win_sys/lc_win/lv_win/win_sys_xml：跨相位存活最久的六个
    窗口对象（A 段相位测量）。每窗在循环体顶 `wc = WindowContext()` 重建，
    字段在 phase-1 渐进绑定（六名首写均先于任何读，默认 None 不可观测）；
    清理协议（del sampler / sim = None / …）随改写转为字段操作，
    gc.collect() 时序不变。
    """

''' + "".join(f"    {n}: Any = None\n" for n in NAMES)


def _read(p):
    with open(p, encoding="utf-8", newline="") as fh:
        return fh.read()


def _strip_dump(node):
    for n in ast.walk(node):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset",
                     "type_comment"):
            if hasattr(n, attr):
                try:
                    delattr(n, attr)
                except AttributeError:
                    setattr(n, attr, None)
    return ast.dump(node)


def main():
    src = _read(FILE)
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    if any(isinstance(n, ast.ClassDef) and n.name == DATACLASS for n in tree.body):
        sys.exit("FAIL: WindowContext 已存在（重复归组？）")
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == KLASS)
    fns = {nm: next(n for n in klass.body if isinstance(n, ast.FunctionDef)
                    and n.name == nm) for nm in FNS}
    loop = next(s for s in fns["run_all_windows"].body if isinstance(s, ast.For))

    # ① 遮蔽核验（全类范围）
    scopes = []

    def collect(node):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) \
                    or hasattr(ch, "generators"):
                scopes.append(ch)
            collect(ch)

    for s in klass.body:
        collect(s)
    for n in ast.walk(klass):
        if isinstance(n, ast.Name) and n.id in NAMES:
            for sc in scopes:
                if sc.lineno <= n.lineno <= sc.end_lineno:
                    b = {x.arg for x in ast.walk(sc) if isinstance(x, ast.arg)}
                    b |= {x.id for x in ast.walk(sc) if isinstance(x, ast.Name)
                          and isinstance(x.ctx, (ast.Store, ast.Del))}
                    if n.id in b:
                        sys.exit(f"FAIL: 内层遮蔽 {n.id} @ L{n.lineno}")

    # ② 调用点定位（loop×4 + produce×1）
    caller_body = {"loop": loop.body,
                   "_produce_window": fns["_produce_window"].body}
    calls = {}
    for cn, where in CALL_EDITS.items():
        cc = [s for s in caller_body[where]
              if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
              and isinstance(s.value.func, ast.Attribute)
              and s.value.func.attr == cn]
        if len(cc) != 1:
            sys.exit(f"FAIL: {cn} 调用点 {len(cc)} 处（预期 1）")
        calls[cn] = cc[0]
        got_kw = sorted(x.arg for x in cc[0].value.keywords if x.arg in NAMES)
        want = sorted(set(NAMES) & {a.arg for a in fns[cn].args.args
                                    + fns[cn].args.kwonlyargs})
        if got_kw != want or not want:
            sys.exit(f"FAIL: {cn} 调用点 wc kwargs {got_kw} ≠ 签名交集 {want}")

    # ③ 行手术 A：引用改写（wc = 循环体 + 5 方法体；排除构造行与调用点行）
    ctor_ln = loop.body[0].lineno          # 构造插在循环体第一条语句处
    call_lines = set()
    for c in calls.values():
        for ln in range(c.lineno, c.end_lineno + 1):
            call_lines.add(ln)
    skip = call_lines | {ctor_ln}
    pts = {}
    # 只改写 FNS 这 6 个方法。旧版 ast.walk(klass) 连 _build_window_system /
    # _safe_boresch_ramp 等 5 个无关方法里**各自的局部** sim/sampler 也改成了
    # wc.*（42 处，运行即 NameError）；6 道 verify_rename 门只看 FNS，没射到。
    for n in (x for fn in fns.values() for x in ast.walk(fn)):
        if isinstance(n, ast.Name) and n.id in NAMES and n.lineno not in skip:
            pts.setdefault(n.lineno, []).append((n.col_offset, n.id))
    total = sum(len(v) for v in pts.values())
    for ln in sorted(pts):
        line = lines[ln - 1]
        raw = line.encode("utf-8")

        def ch(bc, _b=raw):
            return len(_b[:bc].decode("utf-8"))
        for bc, nm2 in sorted(pts[ln], key=lambda t: t[0], reverse=True):
            c = ch(bc)
            if line[c:c + len(nm2)] != nm2:
                sys.exit(f"FAIL: L{ln} 列 {c} 不是 {nm2!r}: {line!r}")
            line = line[:c] + f"{BIND}.{nm2}" + line[c + len(nm2):]
        lines[ln - 1] = line

    # ④ 行手术 B：结构行（降序）
    edits = []
    # wc 名在排序 kwarg/签名里**散布**（非 es 的连续块）：逐行删除 + 首行替换
    for cn, c in calls.items():
        kw_lines = sorted(ln for kw in c.value.keywords if kw.arg in NAMES
                          for ln in [kw.value.lineno])
        edits.append((kw_lines[0], kw_lines[0], [f"            {BIND}={BIND},\n"]))
        for ln in kw_lines[1:]:
            edits.append((ln, ln, []))
    for mn in SIG_EDITS:
        sig_ln = sorted(a.lineno for a in fns[mn].args.kwonlyargs if a.arg in NAMES)
        edits.append((sig_ln[0], sig_ln[0], [f"        {BIND},\n"]))
        for ln in sig_ln[1:]:
            edits.append((ln, ln, []))
    edits.append((ctor_ln, ctor_ln - 1, [f"            {BIND} = {DATACLASS}()\n"]))
    ecs = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "EarlyStopConfig")
    dc_lines = [ln + "\n" for ln in DC_TEXT.rstrip("\n").split("\n")]
    edits.append((ecs.end_lineno + 1, ecs.end_lineno, dc_lines))
    for a_, b_, rep in sorted(edits, key=lambda e: e[0], reverse=True):
        lines[a_ - 1:b_] = rep
    text = "".join(lines)
    new_tree = ast.parse(text)

    # ⑤ 双向验证
    expect = copy.deepcopy(tree)
    exp_k = next(n for n in expect.body if isinstance(n, ast.ClassDef)
                 and n.name == KLASS)
    exp_fns = {nm: next(n for n in exp_k.body if isinstance(n, ast.FunctionDef)
                        and n.name == nm) for nm in FNS}
    exp_loop = next(s for s in exp_fns["run_all_windows"].body
                    if isinstance(s, ast.For))
    exp_caller = {"loop": exp_loop.body,
                  "_produce_window": exp_fns["_produce_window"].body}

    class _Bind(ast.NodeTransformer):
        def visit_Name(self, node):
            self.generic_visit(node)
            if node.id in NAMES and node.lineno not in skip:
                return ast.copy_location(
                    ast.Attribute(value=ast.Name(id=BIND, ctx=ast.Load()),
                                  attr=node.id, ctx=node.ctx), node)
            return node
    for nm in FNS:
        _Bind().visit(exp_fns[nm])

    exp_calls = {}
    for cn, where in CALL_EDITS.items():
        exp_calls[cn] = next(s for s in exp_caller[where]
                             if isinstance(s, ast.Expr)
                             and isinstance(s.value, ast.Call)
                             and isinstance(s.value.func, ast.Attribute)
                             and s.value.func.attr == cn)
    # 构造语句（在 _Bind 之后插入；体首）
    exp_loop.body.insert(0, ast.parse(f"{BIND} = {DATACLASS}()").body[0])
    # 调用点 kwarg 折 1（首删除位）
    for cn, c in exp_calls.items():
        old = c.value.keywords
        kidx = next(i for i, kw in enumerate(old) if kw.arg in NAMES)
        keep = [kw for kw in old if kw.arg not in NAMES]
        keep.insert(kidx, ast.keyword(arg=BIND,
                                      value=ast.Name(id=BIND, ctx=ast.Load())))
        c.value.keywords = keep
    # 签名 6→1（首删除位；kw_defaults 平行）
    for mn in SIG_EDITS:
        fn = exp_fns[mn]
        old = fn.args.kwonlyargs
        idx = next(i for i, a in enumerate(old) if a.arg in NAMES)
        kept = [a for a in old if a.arg not in NAMES]
        kept.insert(idx, ast.arg(arg=BIND))
        fn.args.kwonlyargs = kept
        kd = [d for d, a in zip(fn.args.kw_defaults, old) if a.arg not in NAMES]
        kd.insert(idx, None)
        fn.args.kw_defaults = kd
    # 数据类插入（EarlyStopConfig 之后）
    exp_dc = ast.parse(DC_TEXT).body[0]
    exp_e = next(n for n in expect.body if isinstance(n, ast.ClassDef)
                 and n.name == "EarlyStopConfig")
    expect.body.insert(expect.body.index(exp_e) + 1, exp_dc)

    if _strip_dump(new_tree) != _strip_dump(expect):
        sys.exit("FAIL: 行手术结果与期望树全模块比对不一致——拒绝写盘")

    with open(FILE, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    print(f"✅ {DATACLASS}: 6 字段 slots；构造 @ 循环体顶 L{ctor_ln}；"
          f"引用改写 {total} 处；5 签名 6→1、5 调用点 kwarg 折 1")


if __name__ == "__main__":
    main()
