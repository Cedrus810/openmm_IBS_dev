#!/usr/bin/env python3
"""刀2 内部细分·刀 P-C 执行器：窗口划分段 → _partition_windows。

实测（2026-09-20，第二波收编后）：P-C = _run_dual_lambda_scheme 体语句
[26:41]（15 条，L13909-14101）：partition criterion / vanishing ranges 解析 /
path record / window_ranges_2 / should_run_stage2 门槛。
  * 需要上游 6（保守公式，e/f 已豁免：e 无 Store、handler 局部全覆盖；
    f 为 with 局部——Store/Load 全在自家 with 块内，方法作用域视角的
    "逃逸/形参"均为幻影）；
  * 写逃逸（后段 Load）= 无；
  * 段内方法级 Return = 无（方法在 P-E 尾 Return）。

变换：语句切片原样搬为同类方法 _partition_windows(self, *, 5 形参)
（keyword-only）；调用点 keyword 实参。体缩进本就在方法体层级（8），
**逐字搬运不去缩进**（_finish_production 教训）。本文件 LF。

fail-closed：地标漂移（首条 _partition_criterion / 末条 stage2_status resume）/
条数行数偏离 / 参数表漂移 / 写逃逸非空 / 方法级 Return / Global·Yield /
缩进异常 / 三引号 → 拒写。
"""
import ast
import builtins
import pathlib
import sys

FILE = pathlib.Path(__file__).resolve().parent / "pipeline_orchestration" / "abfe_pipeline_class.py"
KLASS = "ABFEPipeline"
FN = "_run_dual_lambda_scheme"
NEW = "_partition_windows"
SLICE_EXPECT = (26, 41)   # 2026-09-20 快照下的实测下标。**不作定位用**，只作漂移报告；
                          # 真正的定位见 _locate()——主线小修会移动下标，但不动锚点。
MOVED_STMTS = 15
SPAN = (185, 200)
EXEMPT = {"e", "f"}
PARAMS = ["_stage2_protocol_key", "_vanishing_range_kwargs", "dual", "kwargs",
          "resume"]       # self 之外，已排序

DOC = ('        """P-C 窗口划分段—— 刀2 内部细分自 _run_dual_lambda_scheme 原样搬出。\n'
       '\n'
       '        纯搬运（verify_extract --block fn --slice 26:41 逐节点等价）；跨界\n'
       '        变量按原名作 keyword-only 形参（e/f 为段内自覆盖临时名，豁免）。\n'
       '        """\n')


def _locate(fbody):
    """按**结构锚**定位搬移区，返回 (lo, hi)；不依赖语句下标。

    主线只做小 bug 修复时，下标会漂但这两条语句的形状不会变：
      起点锚 = `dual._partition_criterion = ...`（P-C 段的第一条）
      终点锚 = `if dual.should_run_stage1:`（P-C 之后的第一条，不搬）

    任一锚点命中数 ≠ 1，或次序颠倒，一律 **fail-closed 退出**——
    那说明主线动了结构，需要人工看一眼再更新锚点，而不是让脚本猜位置。
    """
    def _is_start(s):
        return (isinstance(s, ast.Assign) and len(s.targets) == 1
                and isinstance(s.targets[0], ast.Attribute)
                and isinstance(s.targets[0].value, ast.Name)
                and s.targets[0].value.id == "dual"
                and s.targets[0].attr == "_partition_criterion")

    def _is_end(s):
        return (isinstance(s, ast.If)
                and "dual.should_run_stage1" in ast.unparse(s.test))

    starts = [i for i, s in enumerate(fbody) if _is_start(s)]
    ends = [i for i, s in enumerate(fbody) if _is_end(s)]
    if len(starts) != 1:
        sys.exit(f"FAIL: 起点锚 `dual._partition_criterion = ...` 命中 {len(starts)} 处"
                 f"（应为 1）—— 主线结构已变，人工核对后更新 _locate 的锚点")
    if len(ends) != 1:
        sys.exit(f"FAIL: 终点锚 `if dual.should_run_stage1:` 命中 {len(ends)} 处"
                 f"（应为 1）—— 同上")
    if starts[0] >= ends[0]:
        sys.exit(f"FAIL: 终点锚下标 {ends[0]} 不在起点锚 {starts[0]} 之后 —— 主线结构已变")
    return starts[0], ends[0]


def _read(p):
    with open(p, encoding="utf-8", newline="") as fh:
        return fh.read()


def binds_ms(stmts):
    b = set()
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                b.add(n.id)
            elif isinstance(n, ast.arg):
                b.add(n.arg)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                b.add(n.name)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                b.add(n.name)
    return b


def ms_loads(stmts):
    out = set()
    scopes = []

    def collect(node):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) \
                    or hasattr(ch, "generators"):
                scopes.append(ch)
            collect(ch)

    for s in stmts:
        collect(s)
    sb = []
    for sc in scopes:
        b = {x.arg for x in ast.walk(sc) if isinstance(x, ast.arg)}
        b |= {x.id for x in ast.walk(sc) if isinstance(x, ast.Name)
              and isinstance(x.ctx, (ast.Store, ast.Del))}
        b |= {x.name for x in ast.walk(sc) if isinstance(x, ast.ExceptHandler) and x.name}
        sb.append((sc, b))
    for n in ast.walk(ast.Module(body=stmts, type_ignores=[])):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if not any(sc.lineno <= n.lineno <= sc.end_lineno and n.id in b
                       for sc, b in sb):
                out.add(n.id)
    return out


def ms_writes(stmts):
    mod = ast.Module(body=stmts, type_ignores=[])
    ib = set()
    for sc in ast.walk(mod):
        if isinstance(sc, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) \
                or hasattr(sc, "generators"):
            ib |= binds_ms([sc])
    return binds_ms(stmts) - ib


def main():
    src = _read(FILE)
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    klass = next((n for n in tree.body if isinstance(n, ast.ClassDef)
                  and n.name == KLASS), None)
    m = next((n for n in klass.body if isinstance(n, ast.FunctionDef)
              and n.name == FN), None) if klass else None
    if m is None:
        sys.exit(f"FAIL: 找不到 {KLASS}.{FN}")
    fbody = m.body[1:]                      # 剥 docstring（fn 体 = 门口径）
    lo, hi = _locate(fbody)                 # 结构锚定位，不用下标（见 _locate 注释）
    if (lo, hi) != SLICE_EXPECT:
        print(f"  [漂移] 搬移区下标 {SLICE_EXPECT} -> ({lo}, {hi})；"
              f"锚点仍唯一命中，按新下标继续")
    moved = fbody[lo:hi]
    if len(moved) != MOVED_STMTS:
        sys.exit(f"FAIL: 搬移语句 {len(moved)} 条 ≠ {MOVED_STMTS}")
    t0 = moved[0].targets[0] if isinstance(moved[0], ast.Assign) else None
    is_pc = (isinstance(t0, ast.Attribute) and t0.attr == "_partition_criterion"
             and isinstance(t0.value, ast.Name) and t0.value.id == "dual")
    if not is_pc:
        sys.exit(f"FAIL: 首条地标漂移：{ast.unparse(moved[0])[:60]}")
    last_txt = ast.unparse(moved[-1])
    if not ("stage2_status" in last_txt and "resume" in last_txt):
        sys.exit(f"FAIL: 末条地标漂移：{last_txt[:60]}")
    span = moved[-1].end_lineno - moved[0].lineno + 1
    if not (SPAN[0] <= span <= SPAN[1]):
        sys.exit(f"FAIL: 搬移行数 {span} 偏离实测区间 {SPAN}")

    region = lines[moved[0].lineno - 1: moved[-1].end_lineno]
    for ln in region:
        if ln.strip() and not ln.startswith(" " * 8):
            sys.exit(f"FAIL: 搬移区缩进异常：{ln!r}")
    if any('"""' in ln or "'''" in ln for ln in region):
        sys.exit("FAIL: 搬移区含三引号")
    for n in ast.walk(ast.Module(body=moved, type_ignores=[])):
        if isinstance(n, (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom)):
            sys.exit(f"FAIL: {type(n).__name__} @ L{n.lineno}")
    for n in ast.walk(ast.Module(body=moved, type_ignores=[])):
        if isinstance(n, ast.Return):
            sys.exit(f"FAIL: 方法级 Return @ L{n.lineno}")

    # 参数重算 + 对账（e/f 豁免）
    B = set(dir(builtins))
    pre = fbody[1:lo]                       # ctor 起的前段
    pparams = {a.arg for a in m.args.args + m.args.kwonlyargs} | {"dual"}
    caller_vis = binds_ms(pre) | pparams
    got = sorted((ms_loads(moved) & caller_vis) - B - {"self"} - EXEMPT)
    if got != PARAMS:
        sys.exit(f"FAIL: 参数表漂移 多{sorted(set(got)-set(PARAMS))} "
                 f"少{sorted(set(PARAMS)-set(got))}")
    post = fbody[hi:]
    esc = sorted(ms_writes(moved) & ms_loads(post) - EXEMPT) \
        if False else sorted((ms_writes(moved) & ms_loads(post)) - EXEMPT)
    if esc:
        sys.exit(f"FAIL: 写逃逸 {esc}")

    # 组装（体 8 缩进逐字搬运；本文件 LF）
    body_text = "".join(region)
    sig = [f"    def {NEW}(", "        self,", "        *,"]
    sig += [f"        {p}," for p in PARAMS]
    sig.append("    ):")
    method = "\n".join(sig) + "\n" + DOC + body_text
    if not method.endswith("\n"):
        method += "\n"

    call = [f"        self.{NEW}("]
    call += [f"            {p}={p}," for p in PARAMS]
    call.append("        )")
    call_lines = [(ln if ln.endswith("\n") else ln + "\n") for ln in call]

    first_ln, last_ln = moved[0].lineno, moved[-1].end_lineno
    out = lines[:first_ln - 1] + call_lines + lines[last_ln:]
    text = "".join(out)
    if not text.endswith("\n"):
        text += "\n"
    text = text + "\n" + method
    ast.parse(text)
    with open(FILE, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)

    print(f"✅ {NEW}: 搬移 {len(moved)} 条语句 / {span} 行（L{first_ln}-{last_ln}），"
          f"形参 {len(PARAMS)} 个（keyword-only），调用点 {len(call_lines)} 行")
    print(f"   verify_extract.py work/partition_windows.before.py {FILE.name} "
          f"--moved {NEW} --src {FN} --block fn --slice {lo}:{hi}")


if __name__ == "__main__":
    main()
