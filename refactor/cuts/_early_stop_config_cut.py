#!/usr/bin/env python3
"""D 线第一刀执行器：EarlyStopConfig 归组（CUTTING_PLAN.md §2.3 簇 1/6）。

实测影响面（2026-09-20）：恰 3 函数 + 2 调用点，类外零引用——
  run_all_windows      签名 9/9（**保留**：外部 API；入口构造 es）体 18 引用
                       （实测全部是两个调用的 kwarg 值，收编后体内零裸引用）
  _produce_window      签名 9/9 → 折 1；体 11 引用
  _finish_production   签名 9/9 → 折 1；体 19 引用
  调用点 ×2（L2312/L2964）：9 个 es kwargs → es=es
变换：
  ① 模块级 @dataclass(frozen=True, slots=True) EarlyStopConfig（9 字段，
     无默认=必填配置；§2.3 只读性由执行器核验：3 体内零 Store/Del）；
  ② run_all_windows docstring 后构造 es = EarlyStopConfig(<9 kwarg=形参裸名>)；
  ③ 三体内 9 名引用改写 es.<name>（构造行与调用点 kwarg 值除外）；
  ④ 两方法签名去 9 加 1；两调用点 9 kwarg 折 1。
门：执行器内置双向验证（期望树同变换，全模块逐节点比对）+ 三函数
verify_rename --in-place + py_compile + layer_check。
"""
import ast
import copy
import os
import pathlib
import sys

# 重放（replay.py）用 CUT_FILE 指向目标文件；缺省＝历史 domains 路径。
FILE = pathlib.Path(os.environ["CUT_FILE"]) if os.environ.get("CUT_FILE") else pathlib.Path(__file__).resolve().parent / "windows_sampling" / "window_manager_dual_lambda.py"
KLASS = "IBSWindowManagerDualLambda"
NAMES = ["early_stop_check_interval_steps", "early_stop_max_delta_g_drift_kJ_mol",
         "early_stop_max_uncertainty_kJ_mol", "early_stop_min_absolute_ess",
         "early_stop_min_decorrelated_samples", "early_stop_min_ess_ratio",
         "early_stop_min_steps", "early_stop_required_consecutive_passes",
         "enable_early_stop"]
BIND = "es"
DATACLASS = "EarlyStopConfig"
FNS = ["run_all_windows", "_produce_window", "_finish_production"]
CALLS = ["_produce_window", "_finish_production"]   # 类内被改签名的两个方法

DC_TEXT = '''@dataclass(frozen=True, slots=True)
class EarlyStopConfig:
    """early-stop 参数簇（D 线归组第一刀，CUTTING_PLAN.md §2.3；2026-09-20）。

    run_all_windows 入口自形参构造，向 _produce_window/_finish_production
    穿线。只读性实测：三函数体内对 9 名零 Store/Del（执行器核验）。
    字段无默认 = 必填配置（构造点全 kwargs 提供）。
    """

''' + "".join(f"    {n}: Any\n" for n in NAMES)


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
        sys.exit("FAIL: EarlyStopConfig 已存在（重复归组？）")
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == KLASS)
    fns = {}
    for nm in FNS:
        cand = [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == nm]
        if len(cand) != 1:
            sys.exit(f"FAIL: {nm} 有 {len(cand)} 个")
        fns[nm] = cand[0]

    # ① 只读性核验：三体内零 Store/Del
    for nm, fn in fns.items():
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and n.id in NAMES \
                    and isinstance(n.ctx, (ast.Store, ast.Del)):
                sys.exit(f"FAIL: {nm} 体内 {n.id} 被 Store/Del @ L{n.lineno}——非只读")
    # 遮蔽核验
    for nm, fn in fns.items():
        scopes = []

        def collect(node):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) \
                        or hasattr(ch, "generators"):
                    scopes.append(ch)
                collect(ch)
        for s in fn.body:
            collect(s)
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and n.id in NAMES:
                for sc in scopes:
                    if sc.lineno <= n.lineno <= sc.end_lineno:
                        b = {x.arg for x in ast.walk(sc) if isinstance(x, ast.arg)}
                        b |= {x.id for x in ast.walk(sc) if isinstance(x, ast.Name)
                              and isinstance(x.ctx, (ast.Store, ast.Del))}
                        if n.id in b:
                            sys.exit(f"FAIL: {nm} 内层遮蔽 {n.id} @ L{n.lineno}")

    # ② 调用点定位（唯一）
    loop = next(s for s in fns["run_all_windows"].body if isinstance(s, ast.For))
    CALLER = {"_produce_window": loop.body, "_finish_production": fns["_produce_window"].body}
    calls = {}
    for cn in CALLS:
        cc = [s for s in CALLER[cn]
              if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
              and isinstance(s.value.func, ast.Attribute)
              and s.value.func.attr == cn]
        if len(cc) != 1:
            sys.exit(f"FAIL: {cn} 调用点 {len(cc)} 处（预期 1）")
        calls[cn] = cc[0]
        got_kw = sorted(x.arg for x in cc[0].value.keywords if x.arg in NAMES)
        if got_kw != sorted(NAMES):
            sys.exit(f"FAIL: {cn} 调用点 es kwargs {len(got_kw)} ≠ 9")

    # ③ 行手术 A：名字引用改写（排除构造行与两个调用点的全部行）
    ctor_fn = fns["run_all_windows"]
    doc0 = ctor_fn.body[0]
    if not (isinstance(doc0, ast.Expr) and isinstance(doc0.value, ast.Constant)
            and isinstance(doc0.value.value, str)):
        sys.exit("FAIL: run_all_windows 体首不是 docstring")
    ctor_ln = doc0.end_lineno + 1            # 构造行插入位置
    call_lines = set()
    for cn in CALLS:
        c = calls[cn]
        for ln in range(c.lineno, c.end_lineno + 1):
            call_lines.add(ln)
    skip_lines = call_lines | {ctor_ln}
    pts = {}
    for nm in FNS:
        for n in ast.walk(fns[nm]):
            if isinstance(n, ast.Name) and n.id in NAMES \
                    and n.lineno not in skip_lines:
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

    # ④ 行手术 B：结构行编辑（降序）
    edits = []   # (start_line, end_line_inclusive, replacement_lines)
    for cn in CALLS:
        c = calls[cn]
        kw_lines = sorted(ln for kw in c.value.keywords if kw.arg in NAMES
                          for ln in [kw.value.lineno])
        edits.append((kw_lines[0], kw_lines[-1], [f"            {BIND}={BIND},\n"]))
        sig_ln = sorted(a.lineno for a in fns[cn].args.kwonlyargs if a.arg in NAMES)
        edits.append((sig_ln[0], sig_ln[-1], [f"        {BIND},\n"]))
    edits.append((ctor_ln, ctor_ln - 1,
                  ["        es = EarlyStopConfig(\n"]
                  + [f"            {n}={n},\n" for n in NAMES]
                  + ["        )\n"]))
    wcls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == "WarmupState")
    dc_lines = [ln + "\n" for ln in DC_TEXT.rstrip("\n").split("\n")]
    edits.append((wcls.end_lineno + 1, wcls.end_lineno, dc_lines))
    for a_, b_, rep in sorted(edits, key=lambda e: e[0], reverse=True):
        lines[a_ - 1:b_] = rep
    text = "".join(lines)
    new_tree = ast.parse(text)

    # ⑤ 双向验证
    expect = copy.deepcopy(tree)
    exp_dc = ast.parse(DC_TEXT).body[0]
    exp_w = next(n for n in expect.body if isinstance(n, ast.ClassDef)
                 and n.name == "WarmupState")
    expect.body[expect.body.index(exp_w) + 1:expect.body.index(exp_w) + 1] = [exp_dc]
    exp_k = next(n for n in expect.body if isinstance(n, ast.ClassDef)
                 and n.name == KLASS)
    exp_fns = {nm: next(n for n in exp_k.body if isinstance(n, ast.FunctionDef)
                        and n.name == nm) for nm in FNS}

    class _Bind(ast.NodeTransformer):
        def visit_Name(self, node):
            self.generic_visit(node)
            if node.id in NAMES and not (node.lineno == ctor_ln
                                         or node.lineno in call_lines):
                return ast.copy_location(
                    ast.Attribute(value=ast.Name(id=BIND, ctx=ast.Load()),
                                  attr=node.id, ctx=node.ctx), node)
            return node
    for nm in FNS:
        _Bind().visit(exp_fns[nm])
    # 构造语句在 _Bind 之后插入（其 kwarg 值必须保持形参裸名，parse 行号
    # 对不上 ctor_ln，先插会被误改写——absorb 系列同款教训）
    exp_fns["run_all_windows"].body.insert(1, ast.parse(
        "es = EarlyStopConfig(" + ",".join(f"{n}={n}" for n in NAMES) + ")").body[0])
    exp_loop = next(s for s in exp_fns["run_all_windows"].body
                    if isinstance(s, ast.For))
    EXP_CALLER = {"_produce_window": exp_loop.body,
                  "_finish_production": exp_fns["_produce_window"].body}
    for cn in CALLS:
        for s in EXP_CALLER[cn]:
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) \
                    and isinstance(s.value.func, ast.Attribute) \
                    and s.value.func.attr == cn:
                old_kw = s.value.keywords
                kidx = next(i for i, kw in enumerate(old_kw) if kw.arg in NAMES)
                keep = [kw for kw in old_kw if kw.arg not in NAMES]
                keep.insert(kidx, ast.keyword(arg=BIND,
                                              value=ast.Name(id=BIND, ctx=ast.Load())))
                s.value.keywords = keep
                break
        fn = exp_fns[cn]
        old = fn.args.kwonlyargs
        idx = next(i for i, a in enumerate(old) if a.arg in NAMES)
        kept = [a for a in old if a.arg not in NAMES]
        kept.insert(idx, ast.arg(arg=BIND))
        fn.args.kwonlyargs = kept
        kw_def = fn.args.kw_defaults
        kept_d = [d for d, a in zip(kw_def, old) if a.arg not in NAMES]
        kept_d.insert(idx, None)
        fn.args.kw_defaults = kept_d

    if _strip_dump(new_tree) != _strip_dump(expect):
        sys.exit("FAIL: 行手术结果与期望树全模块比对不一致——拒绝写盘")

    with open(FILE, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    per = {nm: sum(1 for ln in pts for _, x in pts[ln] if x == nm) for nm in FNS}
    print(f"✅ {DATACLASS}: 9 字段 frozen+slots；构造 @ L{ctor_ln}；"
          f"引用改写 {per}（调用点 kwarg 值不在内）；2 签名折编、2 调用点 9→1")


if __name__ == "__main__":
    main()
