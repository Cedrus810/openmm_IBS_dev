#!/usr/bin/env python3
"""刀1 第二刀·刀b/刀c 执行器：相位 3/4 抽取（CUTTING_PLAN.md §2.2.1）。

前提：B 段第 1 步已把 frozen_f_k_snapshot / mode / ema_mean_p_values 收编进
WarmupState——本文件头注释记录的 3/4、4/5 边界写逃逸已实测清零。

  --phase 3  相位 3 = 预热循环（唯一 While，实测 774 行，L2110-2883）
             → _run_warmup_loop(self, *, 25 形参)
  --phase 4  相位 4 = 预热收口/诊断（20 条语句，实测 686 行，L2887-3572）
             → _summarize_warmup(self, *, 40 形参)   〔须在 --phase 3 之后跑〕

纯搬运：体逐字不变（仅去 4 格缩进），门 = verify_extract.py --block for
--slice <起:止>。参数表 = 保守公式（段内方法作用域 Load ∩ 调用方可见绑定），
传入值 = 调用方当前值，与原同作用域读严格等价。
fail-closed：地标漂移 / 行数偏离 / 条数不符 / 参数表漂移 / 写逃逸非空 /
方法级 Return / 窗口级 continue·break / 缩进异常 / 三引号 → 拒写。
新行一律 CRLF。
"""
import ast
import builtins
import pathlib
import sys

FILE = pathlib.Path(__file__).resolve().parent / "windows_sampling" / "window_manager_dual_lambda.py"
KLASS = "IBSWindowManagerDualLambda"
FN = "run_all_windows"
CRLF = "\r\n"

PARAMS3 = ["K", "_attempt_ledger", "_nan_diagnostics", "check_chunk",
           "expected_main_manifest", "full_bias_step_budget", "ibs_state_file",
           "k", "lc_win", "lv_win", "main_ckpt_path", "main_manifest_path",
           "max_bias_updates", "prior_cumulative_steps", "provisional_authorized",
           "resumed_calibration_pending", "sampler", "self", "sim",
           "skip_frozen_revalidation", "skip_warmup_entirely", "stage_type",
           "start", "warm", "window_idx"]
PARAMS4 = ["K", "_attempt_ledger", "_ledger", "_nan_diagnostics",
           "accept_recalibrated_f_k_without_gate", "allow_best_effort_warmup",
           "check_chunk", "coverage_ess_threshold",
           "effective_frozen_validation_budget",
           "effective_mbar_calibration_reserved_steps", "end",
           "expected_main_manifest", "frozen_validation_reserved_steps",
           "full_bias_step_budget", "ibs_state_file", "ibs_wrap", "k", "lc_win",
           "lse_log_residual_tolerance", "lv_win", "main_ckpt_path",
           "main_manifest_path", "max_bias_updates", "min_bias_updates",
           "min_probability_threshold", "minimum_complete_validation_frames",
           "required_consecutive_bias_updates", "resume_frozen_validation",
           "resumed_calibration_pending", "sampler", "self", "sgd_step_budget", "sim",
           "stage_type", "start", "target_p", "validation_attempt_budget_steps",
           "warm", "warmup_only", "window_idx"]
for lst, n in ((PARAMS3, 25), (PARAMS4, 40)):
    assert lst == sorted(lst) and len(lst) == n and "self" in lst

DOC3 = ('        """相位 3：预热循环—— 刀1 第二刀·刀b 自 run_all_windows 原样搬出。\n'
        '\n'
        '        纯搬运（verify_extract --block for --slice 逐节点等价）；跨界变量按\n'
        '        原名作 keyword-only 形参（CUTTING_PLAN.md §2.2.1）。\n'
        '        """\n').replace("\n", CRLF)
DOC4 = ('        """相位 4：预热收口/诊断—— 刀1 第二刀·刀c 自 run_all_windows 原样搬出。\n'
        '\n'
        '        纯搬运（verify_extract --block for --slice 逐节点等价）；跨界变量按\n'
        '        原名作 keyword-only 形参（CUTTING_PLAN.md §2.2.1）。\n'
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


def main():
    phase = sys.argv[sys.argv.index("--phase") + 1] if "--phase" in sys.argv else sys.exit(
        "用法: _warmup_split_cuts.py --phase 3|4")
    if phase not in ("3", "4"):
        sys.exit("FAIL: --phase 只能是 3 或 4")

    src = _read(FILE)
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    klass = next((n for n in tree.body if isinstance(n, ast.ClassDef)
                  and n.name == KLASS), None)
    m = next((n for n in klass.body if isinstance(n, ast.FunctionDef)
              and n.name == FN), None) if klass else None
    if m is None:
        sys.exit(f"FAIL: 找不到 {KLASS}.{FN}")
    loop = next((s for s in m.body if isinstance(s, ast.For)), None)
    body = loop.body if loop else None
    if body is None:
        sys.exit("FAIL: run_all_windows 顶层没有 for")

    if phase == "3":
        whiles = [s for s in body if isinstance(s, ast.While)]
        if len(whiles) != 1:
            sys.exit(f"FAIL: While 地标 {len(whiles)} 处（预期 1）")
        node = whiles[0]
        idx = body.index(node)
        moved = [node]
        span_expect = (740, 810)
        new_name, params, doc = "_run_warmup_loop", PARAMS3, DOC3
    else:
        c3 = [s for s in body if isinstance(s, ast.Expr)
              and isinstance(s.value, ast.Call)
              and isinstance(s.value.func, ast.Attribute)
              and s.value.func.attr == "_run_warmup_loop"]
        if len(c3) != 1:
            sys.exit(f"FAIL: _run_warmup_loop 调用地标 {len(c3)} 处（预期 1；先跑 --phase 3）")
        c3_i = body.index(c3[0])
        cc = [s for s in body if isinstance(s, ast.Expr)
              and isinstance(s.value, ast.Call)
              and isinstance(s.value.func, ast.Attribute)
              and s.value.func.attr == "_commit_frozen_f_k"]
        if len(cc) != 1:
            sys.exit(f"FAIL: _commit_frozen_f_k 调用地标 {len(cc)} 处（预期 1）")
        end_i = body.index(cc[0])
        if end_i <= c3_i + 1:
            sys.exit("FAIL: 相位 4 区间为空——结构漂移")
        moved = body[c3_i + 1: end_i]
        node = None
        idx = c3_i + 1
        span_expect = (660, 710)
        new_name, params, doc = "_summarize_warmup", PARAMS4, DOC4
        if len(moved) != 20:
            sys.exit(f"FAIL: 相位 4 语句 {len(moved)} 条 ≠ 20")

    span = moved[-1].end_lineno - moved[0].lineno + 1
    if not (span_expect[0] <= span <= span_expect[1]):
        sys.exit(f"FAIL: 搬移行数 {span} 偏离实测区间 {span_expect}")

    region = lines[moved[0].lineno - 1: moved[-1].end_lineno]
    for ln in region:
        if ln.strip() and not ln.startswith(" " * 12):
            sys.exit(f"FAIL: 搬移区缩进异常：{ln!r}")
    if any('"""' in ln or "'''" in ln for ln in region):
        sys.exit("FAIL: 搬移区含三引号（去缩进会改字符串内容）")
    first = moved[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
            and isinstance(first.value.value, str):
        sys.exit("FAIL: 首语句为字符串字面量")

    # 参数重算（保守公式）+ 对账
    B = set(dir(builtins))
    rest = [s for i, s in enumerate(body) if s not in moved]
    params_m = {a.arg for a in m.args.args + m.args.kwonlyargs + m.args.posonlyargs}
    if m.args.vararg:
        params_m.add(m.args.vararg.arg)
    if m.args.kwarg:
        params_m.add(m.args.kwarg.arg)
    tgt = {n.id for n in ast.walk(loop.target) if isinstance(n, ast.Name)}
    caller_vis = binds_ms(rest) | params_m | tgt

    scopes = []

    def collect(node):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) \
                    or hasattr(ch, "generators"):
                scopes.append(ch)
            collect(ch)
    for s in moved:
        collect(s)
    sb = []
    for sc in scopes:
        b = {x.arg for x in ast.walk(sc) if isinstance(x, ast.arg)}
        b |= {x.id for x in ast.walk(sc) if isinstance(x, ast.Name)
              and isinstance(x.ctx, (ast.Store, ast.Del))}
        b |= {x.name for x in ast.walk(sc) if isinstance(x, ast.ExceptHandler) and x.name}
        sb.append((sc, b))
    ms_loads = set()
    for n in ast.walk(ast.Module(body=moved, type_ignores=[])):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if not any(sc.lineno <= n.lineno <= sc.end_lineno and n.id in b
                       for sc, b in sb):
                ms_loads.add(n.id)
    got = sorted((ms_loads & caller_vis) - B - {"self"})
    want = [p for p in params if p != "self"]
    if got != want:
        sys.exit(f"FAIL: 参数表漂移 多{sorted(set(got) - set(want))} 少{sorted(set(want) - set(got))}")

    # 写逃逸 / Return / 窗口级 continue·break
    fi = next(i for i, s in enumerate(m.body) if s is loop)
    rest_loads = {n.id for s in rest for n in ast.walk(s)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    post_loads = {n.id for s in m.body[fi + 1:] for n in ast.walk(s)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    ib = set()
    mod = ast.Module(body=moved, type_ignores=[])
    for sc in ast.walk(mod):
        if isinstance(sc, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) \
                or hasattr(sc, "generators"):
            ib |= binds_ms([sc])
    w_ms = binds_ms(moved) - ib
    esc = sorted(w_ms & (rest_loads | post_loads | tgt))
    if esc:
        sys.exit(f"FAIL: 写逃逸 {esc}——需返回值协议，本执行器不支持")

    def bad_returns(stmts):
        out = []

        def walk(n_, in_fn):
            for ch in ast.iter_child_nodes(n_):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    walk(ch, True)
                elif isinstance(ch, ast.Return) and not in_fn:
                    out.append(ch.lineno)
                else:
                    walk(ch, in_fn)
        for s in stmts:
            walk(s, False)
        return out
    if bad_returns(moved):
        sys.exit("FAIL: 方法级 Return")
    mv_loops = [(x.lineno, x.end_lineno) for x in ast.walk(mod)
                if isinstance(x, (ast.For, ast.While, ast.AsyncFor))]
    cbs = [(type(n).__name__, n.lineno) for n in ast.walk(mod)
           if isinstance(n, (ast.Continue, ast.Break))
           and not any(a <= n.lineno <= b for a, b in mv_loops)]
    if cbs:
        sys.exit(f"FAIL: 窗口级 continue/break @ {cbs}")

    # 组装 + 手术
    def dedent4(txt):
        return "".join(ln[4:] if ln.startswith(" " * 12) else ln
                       for ln in txt.splitlines(keepends=True))
    body_text = dedent4("".join(region))
    sig = [f"    def {new_name}(", "        self,", "        *,"]
    sig += [f"        {p}," for p in want]
    sig.append("    ):")
    method = CRLF.join(sig) + CRLF + doc + body_text
    if not method.endswith(CRLF):
        method += CRLF

    call = [f"            self.{new_name}("]
    call += [f"                {p}={p}," for p in want]
    call.append("            )")
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

    print(f"✅ {new_name}: 搬移 {len(moved)} 条语句 / {span} 行（L{first_ln}-{last_ln}），"
          f"形参 {len(want)} 个（keyword-only），调用点 {len(call_lines)} 行")
    print(f"   verify_extract.py <留底> {FILE.name} --moved {new_name} --src {FN} "
          f"--block for --index 0 --slice {idx}:{idx + len(moved)}")


if __name__ == "__main__":
    main()
