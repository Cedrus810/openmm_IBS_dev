#!/usr/bin/env python3
"""刀1 B 段·第 3 步执行器：[相位7+8] 联合抽取 → _finish_production。

前提：ProductionState 已收编 26 个 6/7 边界名字（_production_state_absorb.py）。
本轮搬 [相位7(for up)+相位8] 联合段（实测 25 条语句，L2964-3767，~804 行）——
**不**单独切 7/8：两者边界还有 9 个相位 7 内嵌首写的名字（fmax/state_n 等），
吸收它们会在 n_updates==0 的 crash 路径上改语义（铁律 4），7/8 细分留下一轮、
等用户对该路径语义拍板。

临时名豁免（流幻影，逐一实测）：e（Store@L3078/3381 先于全部 Load，handler 局部）、
gid/e_g（debug 循环局部，Load 在自己 Store 之后）、i（for 局部，Load 全在自己的
for-i 循环内）——四者的跨段"逃逸"均为名字级分析幻影，不作字段不作形参。

纯搬运：体逐字不变（仅去 4 格缩进），门 = verify_extract.py --block for
--slice 58:83。参数 = 保守公式（段内方法作用域 Load ∩ 调用方可见绑定）− 豁免名。
fail-closed：地标漂移 / 行数偏离 / 参数表漂移 / 写逃逸非空（豁免外）/ 方法级
Return / 缩进异常 / 三引号 → 拒写。新行一律 CRLF。
"""
import ast
import builtins
import pathlib
import sys

FILE = pathlib.Path(__file__).resolve().parent / "windows_sampling" / "window_manager_dual_lambda.py"
KLASS = "IBSWindowManagerDualLambda"
FN = "_produce_window"
NEW = "_finish_production"
CRLF = "\r\n"
EXEMPT = {"e", "gid", "e_g", "i"}
SPAN_EXPECT = (770, 840)
MOVED_STMTS = 25

PARAMS = ["K", "debug_mode", "early_stop_check_interval_steps",
          "early_stop_max_delta_g_drift_kJ_mol", "early_stop_max_uncertainty_kJ_mol",
          "early_stop_min_absolute_ess", "early_stop_min_decorrelated_samples",
          "early_stop_min_ess_ratio", "early_stop_min_steps",
          "early_stop_required_consecutive_passes", "enable_early_stop", "end",
          "ibs_state_file", "ibs_wrap", "lc_win", "lse_log_residual_tolerance",
          "lv_win", "n_steps_per_window", "prod", "repair_policy", "sampler",
          "self", "sim", "stage_type", "start", "steps_per_update", "warm",
          "win_sys", "window_idx"]
assert PARAMS == sorted(PARAMS) and len(PARAMS) == 29 and "self" in PARAMS

DOC = ('        """相位 7+8：生产采样循环 + 收口/落盘—— 刀1 B段第3步自\n'
       '        _produce_window 原样搬出（联合段；7/8 细分等 n_updates==0 路径\n'
       '        语义拍板，见执行器头注释）。纯搬运（verify_extract --slice 逐节点\n'
       '        等价）；跨界变量按原名作 keyword-only 形参。\n'
       '        """\n').replace("\n", CRLF)


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
    ups = [s for s in m.body if isinstance(s, ast.For)
           and isinstance(s.target, ast.Name) and s.target.id == "up"]
    if len(ups) != 1:
        sys.exit(f"FAIL: for-up 地标 {len(ups)} 处（预期 1）")
    node = ups[0]
    idx = m.body.index(node)
    moved = m.body[idx:]
    if len(moved) != MOVED_STMTS:
        sys.exit(f"FAIL: 搬移语句 {len(moved)} 条 ≠ {MOVED_STMTS}")
    span = moved[-1].end_lineno - moved[0].lineno + 1
    if not (SPAN_EXPECT[0] <= span <= SPAN_EXPECT[1]):
        sys.exit(f"FAIL: 搬移行数 {span} 偏离实测区间 {SPAN_EXPECT}")

    region = lines[moved[0].lineno - 1: moved[-1].end_lineno]
    for ln in region:
        if ln.strip() and not ln.startswith(" " * 8):
            sys.exit(f"FAIL: 搬移区缩进异常：{ln!r}")
    if any('"""' in ln or "'''" in ln for ln in region):
        sys.exit("FAIL: 搬移区含三引号")
    if isinstance(moved[0], ast.Expr) and isinstance(moved[0].value, ast.Constant) \
            and isinstance(moved[0].value.value, str):
        sys.exit("FAIL: 首语句为字符串字面量")

    # 参数重算 + 对账
    B = set(dir(builtins))
    rest = m.body[:idx]
    pparams = {a.arg for a in m.args.args + m.args.kwonlyargs + m.args.posonlyargs}
    if m.args.vararg:
        pparams.add(m.args.vararg.arg)
    if m.args.kwarg:
        pparams.add(m.args.kwarg.arg)
    caller_vis = binds_ms(rest) | pparams
    got = sorted((ms_loads(moved) & caller_vis) - B - {"self"} - EXEMPT)
    want = [p for p in PARAMS if p != "self"]
    if got != want:
        sys.exit(f"FAIL: 参数表漂移 多{sorted(set(got) - set(want))} 少{sorted(set(want) - set(got))}")

    # 写逃逸（豁免外）/ 方法级 Return
    rest_loads = ms_loads(rest)
    esc = sorted((ms_writes(moved) & rest_loads) - EXEMPT)
    if esc:
        sys.exit(f"FAIL: 写逃逸 {esc}")
    outs = []
    for n in ast.walk(ast.Module(body=moved, type_ignores=[])):
        if isinstance(n, ast.Return):
            outs.append(n.lineno)
    if outs:
        sys.exit(f"FAIL: 方法级 Return @ {outs}")

    # 组装 + 手术。注意：搬移语句本就在 _produce_window 体层级（8 缩进），
    # 新方法体同为 8 缩进 → **逐字搬运，不去缩进**（12→8 的模板在这里不适用）。
    body_text = "".join(region)
    sig = [f"    def {NEW}(", "        self,", "        *,"]
    sig += [f"        {p}," for p in want]
    sig.append("    ):")
    method = CRLF.join(sig) + CRLF + DOC + body_text
    if not method.endswith(CRLF):
        method += CRLF

    call = [f"        self.{NEW}("]
    call += [f"            {p}={p}," for p in want]
    call.append("        )")
    call_lines = [(ln if ln.endswith(CRLF) else ln + CRLF) for ln in call]

    first_ln, last_ln = moved[0].lineno, moved[-1].end_lineno
    out = lines[:first_ln - 1] + call_lines + lines[last_ln:]
    text = "".join(out)
    if not text.endswith(CRLF):
        text += CRLF
    text = text + CRLF + method
    ast.parse(text)
    with open(FILE, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)

    print(f"✅ {NEW}: 搬移 {len(moved)} 条语句 / {span} 行（L{first_ln}-{last_ln}），"
          f"形参 {len(want)} 个（keyword-only），调用点 {len(call_lines)} 行")
    print(f"   verify_extract.py work/finish_production.before.py {FILE.name} "
          f"--moved {NEW} --src {FN} --block for --index 0 --slice {idx}:{idx + len(moved)}")


if __name__ == "__main__":
    main()
