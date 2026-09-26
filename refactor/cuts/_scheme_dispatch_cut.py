#!/usr/bin/env python3
"""刀2 执行器（CUTTING_PLAN.md §3）：run_full_pipeline 尾部三路分派抽双方法。

实测（2026-09-20，作用域感知口径）：run_full_pipeline 尾块为 **elif 链**：
  if decoupling_scheme == 'dual_lambda':       <52 条顶层语句, ~1952 行>
  elif decoupling_scheme == 'single_lambda':   <16 条>
  elif decoupling_scheme == '2d_diagonal':     <306 行链，直至函数尾>

变换（§3 草图的实测修正版——三路 elif 链而非两路）：
  ① if 体 → _run_dual_lambda_scheme(self, *, 26 形参)；
  ② 内层 elif 链整体（单条 If 语句，AST 与 if 同构，首行 elif→if 源级改写）
     → _run_single_lambda_scheme(self, *, 23 形参)；
  ③ 尾部改为 return 分派（保持 if/else 骨架）。
Return 传播精确性：if 体以无条件 Return 收尾；single_lambda 侧 3 个
`return final` + 落空隐式 None——调用点 return 包裹后与原语义逐路径等价。

参数 = 保守公式（段内方法作用域 Load ∩ 调用方可见绑定，含 **kwargs 绑定）。
豁免：orelse 侧 e/exc 为 handler 局部（全覆盖实测）；if 体侧 e 有 L13200 一处
真外层读，是真形参（两侧不对称，分别实测）。5 个方法级内嵌 def 作为函数形参
传入（闭包对象随参传递，语义不变）。

门：verify_extract --block if --index 21（method1 体）+ --orelse（method2 语句）。
fail-closed：地标漂移 / 参数表漂移 / Global·Nonlocal·Yield / 缩进异常 /
三引号 → 拒写。本文件为 LF，新行一律 LF。
"""
import ast
import builtins
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from cutter import append_to_class  # noqa: E402

# 重放（replay.py）用 CUT_FILE 指向目标文件；缺省＝历史 domains 路径。
FILE = pathlib.Path(os.environ["CUT_FILE"]) if os.environ.get("CUT_FILE") else pathlib.Path(__file__).resolve().parent / "pipeline_orchestration" / "abfe_pipeline_class.py"
KLASS = "ABFEPipeline"
FN = "run_full_pipeline"
M1 = "_run_dual_lambda_scheme"
M2 = "_run_single_lambda_scheme"

PARAMS1 = [
    '_build_top_level_protocol_key',
    '_decharge_method',
    '_path_evolution_silenced_for_autonomous',
    '_sampling_repair_policy',
    '_stage1_protocol_key',
    '_stage2_protocol_key',
    '_vanishing_range_kwargs',
    'analysis_context',
    'boresch_params',
    'dexp_params',
    'enable_early_stop',
    'kwargs',
    'n_steps_per_window',
    'potential_type',
    'resume',
    'sampling_key',
    'stage0',
    'stage1_states',
    'stage2_probe_states',
    'stages',
    'steps_per_update',
    'system_type',
]
PARAMS2 = [
    '_build_geodesic_path_protocol_key',
    '_build_sampling_protocol_key',
    '_build_top_level_protocol_key',
    '_load_validated_simple_path',
    '_path_content_fingerprint',
    'boresch_params',
    'decoupling_scheme',
    'dexp_params',
    'enable_early_stop',
    'kwargs',
    'n_states_per_stage',
    'n_steps_per_window',
    'potential_type',
    'resume',
    'stage0',
    'stages',
    'steps_per_update',
    'system_type',
]
assert PARAMS1 == sorted(PARAMS1)
# 段内自覆盖的幻影名（09-23 verify_bound 门抓出）：搬移区里每处读前都有段内先写，
# 但调用方只在条件分支/内嵌 def 里绑定它们——旧版照「某处有 Store」当形参传，
# 在未绑定路径上调用点即 UnboundLocalError（single_lambda 路径实测炸）。
# cached：run_full_pipeline 只在内嵌 def _load_validated_simple_path 里绑定，
#   dual 支路只把它原样转传给 _resume_preoptimized，后者段内先写后读。
EXEMPT1 = {"cached", "e", "f", "final"}
EXEMPT2 = {"e", "exc", "f", "final", "point", "raw_path", "values"}
# 主线漂移（人工核过、只读的新增依赖）：参数表对账时允许这些名字多出/缺少。
# 历史快照上不出现 ⟹ 历史回归不受影响；新增一律写来源与核对结论。
DRIFT_OK = {
    "_repro_tilt_signs",       # 09-22 复现模式：run_full_pipeline 头部赋值，dual 支路只读
    "_repro_uniform_states",   # 同上
}
assert PARAMS2 == sorted(PARAMS2)

DOC1 = ('        """dual_lambda 分派支路—— 刀2 自 run_full_pipeline 原样搬出。\n'
        '\n'
        '        纯搬运（verify_extract --block if --index 21 逐节点等价）；跨界变量\n'
        '        按原名作 keyword-only 形参；方法级 Return 由调用点 return 包裹传播\n'
        '        （CUTTING_PLAN.md §3）。\n'
        '        """\n')
DOC2 = ('        """single_lambda / 2d_diagonal 分派支路—— 刀2 自 run_full_pipeline\n'
        '        原样搬出（整条 elif 链逐字搬运；首行 elif→if 为纯语法改写，\n'
        '        AST 同构）。\n\n'
        '        纯搬运（verify_extract --block if --index 21 --orelse 逐节点等价）；\n'
        '        跨界变量按原名作 keyword-only 形参。\n'
        '        """\n')


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


def _body_no_doc(fn_node):
    b = list(fn_node.body)
    if (b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant)
            and isinstance(b[0].value.value, str)):
        b = b[1:]
    return b


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
    tail = m.body[-1] if m.body else None
    ok_tail = (isinstance(tail, ast.If) and isinstance(tail.test, ast.Compare)
               and isinstance(tail.test.left, ast.Name)
               and tail.test.left.id == "decoupling_scheme"
               and len(tail.orelse) == 1 and isinstance(tail.orelse[0], ast.If)
               and isinstance(tail.orelse[0].test.left, ast.Name)
               and tail.orelse[0].test.left.id == "decoupling_scheme")
    if not ok_tail:
        sys.exit("FAIL: 尾部 elif 链地标漂移")
    inner = tail.orelse[0]
    moved1, moved2 = tail.body, [inner]
    if len(moved1) != 52:
        sys.exit(f"FAIL: if 体 {len(moved1)} 条 ≠ 52")
    if not isinstance(moved1[-1], ast.Return):
        sys.exit("FAIL: if 体末条不是 Return（return 包裹传播的前提）")
    if len(moved2[0].body) != 16:
        sys.exit(f"FAIL: single_lambda 体 {len(moved2[0].body)} 条 ≠ 16")

    for tag, seg in (("if", moved1), ("else", moved2)):
        for n in ast.walk(ast.Module(body=seg, type_ignores=[])):
            if isinstance(n, (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom)):
                sys.exit(f"FAIL: {tag} 体 {type(n).__name__} @ L{n.lineno}")

    B = set(dir(builtins))
    head = m.body[:-1]
    pparams = {a.arg for a in m.args.args + m.args.kwonlyargs + m.args.posonlyargs}
    if m.args.vararg:
        pparams.add(m.args.vararg.arg)
    if m.args.kwarg:
        pparams.add(m.args.kwarg.arg)
    caller_vis = binds_ms(head) | pparams
    got1 = sorted((ms_loads(moved1) & caller_vis) - B - {"self"} - EXEMPT1)
    got2 = sorted((ms_loads(moved2) & caller_vis) - B - {"self"} - EXEMPT2)
    def _accept(got, want, tag):
        extra, miss = set(got) - set(want), set(want) - set(got)
        if (extra | miss) and (extra | miss) <= DRIFT_OK:
            print(f"  [漂移·已核] {tag} 多{sorted(extra)} 少{sorted(miss)}")
            return got
        return want
    PARAMS1[:] = _accept(got1, PARAMS1, "method1")
    PARAMS2[:] = _accept(got2, PARAMS2, "method2")
    if got1 != PARAMS1:
        sys.exit(f"FAIL: method1 参数表漂移 多{sorted(set(got1)-set(PARAMS1))} "
                 f"少{sorted(set(PARAMS1)-set(got1))}")
    if got2 != PARAMS2:
        sys.exit(f"FAIL: method2 参数表漂移 多{sorted(set(got2)-set(PARAMS2))} "
                 f"少{sorted(set(PARAMS2)-set(got2))}")

    TRIPLE = chr(34) * 3

    def region_of(moved):
        return lines[moved[0].lineno - 1: moved[-1].end_lineno]

    def check_common(region):
        for ln in region:
            if ln.strip() and not ln.startswith(" " * 12):
                sys.exit(f"FAIL: 搬移区缩进异常：{ln!r}")
        if any(TRIPLE in ln for ln in region):
            sys.exit("FAIL: 搬移区含三引号")

    body1 = dedent4("".join(region_of(moved1)))
    check_common(region_of(moved1))

    region2 = region_of(moved2)
    first = region2[0]
    if not first.lstrip().startswith("elif "):
        sys.exit(f"FAIL: method2 首行不是 elif：{first!r}")
    out2 = [first.replace("elif ", "if ", 1)]
    for ln in region2[1:]:
        if ln.strip() and not ln.startswith(" " * 8):
            sys.exit(f"FAIL: method2 搬移区缩进异常：{ln!r}")
        out2.append(ln)                      # 链与体逐字搬运（链 8 / 体 12+）
    if any(TRIPLE in ln for ln in region2):
        sys.exit("FAIL: method2 搬移区含三引号")
    body2 = "".join(out2)

    def sig(name, params):
        s = [f"    def {name}(", "        self,", "        *,"]
        s += [f"        {p}," for p in params]
        s.append("    ):")
        return "\n".join(s)

    m1_text = sig(M1, PARAMS1) + "\n" + DOC1 + body1
    m2_text = sig(M2, PARAMS2) + "\n" + DOC2 + body2
    for t in (m1_text, m2_text):
        if not t.endswith("\n"):
            t = t + "\n"

    disp_src = (
        "        if decoupling_scheme == 'dual_lambda':\n"
        "            return self._run_dual_lambda_scheme(\n"
        + "".join(f"                {p}={p},\n" for p in PARAMS1)
        + "            )\n"
        # 第二支必须是 else：single 方法内部自带四条非 dual 路径的 test 与
        # raise 兜底。写成 elif 会让 2d_* 死码、fail-closed 的 raise 静默成 None
        # （刀2 事故，09-20 手修；此处同步进执行器，verify_dispatch 把关）。
        "        else:\n"
        "            return self._run_single_lambda_scheme(\n"
        + "".join(f"                {p}={p},\n" for p in PARAMS2)
        + "            )\n")

    first_ln, last_ln = tail.lineno, tail.end_lineno
    out = lines[:first_ln - 1] + [disp_src] + lines[last_ln:]
    text = append_to_class("".join(out), KLASS, m1_text + "\n" + m2_text)
    ast.parse(text)

    t2 = ast.parse(text)
    k2 = next(n for n in t2.body if isinstance(n, ast.ClassDef) and n.name == KLASS)
    m2f = next(n for n in k2.body if isinstance(n, ast.FunctionDef) and n.name == FN)
    last = m2f.body[-1]
    assert isinstance(last, ast.If) and len(last.body) == 1 \
        and isinstance(last.body[0], ast.Return) and len(last.orelse) == 1 \
        and isinstance(last.orelse[0], ast.Return)
    assert len(next(n for n in k2.body if isinstance(n, ast.FunctionDef)
                    and n.name == M1).body) == 53      # docstring + 52
    assert len(next(n for n in k2.body if isinstance(n, ast.FunctionDef)
                    and n.name == M2).body) == 2       # docstring + 内层 If 链

    with open(FILE, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)

    print(f"✅ 刀2: {M1}（if 体 {len(moved1)} 条/{moved1[-1].end_lineno-moved1[0].lineno+1} 行，"
          f"形参 {len(PARAMS1)}）+ {M2}（elif 链整体，形参 {len(PARAMS2)}）；"
          f"尾部改 if/elif return 分派（{len(disp_src.splitlines())} 行）")


def dedent4(txt):
    return "".join(ln[4:] if ln.startswith(" " * 12) else ln
                   for ln in txt.splitlines(keepends=True))


if __name__ == "__main__":
    main()
