#!/usr/bin/env python3
"""刀1 第二刀·刀a 执行器（CUTTING_PLAN.md §2.2.1）：_commit_frozen_f_k 抽取。

把 run_all_windows 循环体的相位 5（`if warm.bias_converged:` 单条语句，
f_k 冻结/提交，实测 body[120]，L3570-3974 共 405 行）原样搬为同类方法
_commit_frozen_f_k；调用点 keyword 实参逐名传入。纯搬运：体逐字不变
（仅去 4 格缩进），门 = verify_extract.py --block for --slice 120:121。

边界实测（2026-09-20，五边界测量）：需要上游 24、写逃逸 无、方法级 Return 无、
窗口级 continue/break 无——五条候选边界中唯一与「相位3-5 整块」并列的零逃逸刀。
相位3/4 单独切被 frozen_f_k_snapshot / mode / ema_mean_p_values 三个跨边界标量
拦住（B 段收编 WarmupState 后自动解锁，见 PROGRESS.md）。

fail-closed：地标漂移（bias_converged If 不唯一/行数异常）/ 参数表与权威名单
不符 / 写逃逸非空 / 方法级 Return / 窗口级 continue·break / 缩进异常 /
三引号 / 首语句字符串字面量 → 拒绝写文件。新行一律 CRLF（全文件已统一 CRLF）。
"""
import ast
import builtins
import pathlib
import sys

FILE = pathlib.Path(__file__).resolve().parent / "windows_sampling" / "window_manager_dual_lambda.py"
KLASS = "IBSWindowManagerDualLambda"
FN = "run_all_windows"
NEW = "_commit_frozen_f_k"
CRLF = "\r\n"
EXPECTED_MOVED = 1
EXPECTED_PARAMS = [
    "effective_frozen_validation_budget", "ema_mean_p_values",
    "expected_main_manifest", "frozen_validation_is_final_rung",
    "frozen_validation_reserved_steps", "full_bias_step_budget",
    "ibs_state_file", "lc_win", "lv_win", "main_ckpt_path",
    "main_manifest_path", "max_bias_updates", "mode",
    "prior_cumulative_steps", "resumed_calibration_pending", "sampler",
    "sgd_step_budget", "sim", "skip_frozen_revalidation", "stage_type",
    "warm", "warmup_only", "window_idx",
]
assert len(EXPECTED_PARAMS) == 23 and EXPECTED_PARAMS == sorted(EXPECTED_PARAMS)

DOCSTRING = ('        """相位 5：f_k 冻结/提交—— 刀1 第二刀·刀a 自 run_all_windows 原样搬出。\n'
             '\n'
             '        纯搬运（verify_extract --block for --slice 120:121 逐节点等价）；\n'
             '        跨界变量按原名作 keyword-only 形参（CUTTING_PLAN.md §2.2.1）。\n'
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

    # ① 地标：相位 5 = 唯一的 `if warm.bias_converged:` 顶层 If
    cand = [s for s in body if isinstance(s, ast.If)
            and isinstance(s.test, ast.Attribute) and s.test.attr == "bias_converged"]
    if len(cand) != 1:
        sys.exit(f"FAIL: bias_converged If 地标 {len(cand)} 处（预期 1）")
    node = cand[0]
    idx = body.index(node)
    span = node.end_lineno - node.lineno + 1
    if not (380 <= span <= 430):
        sys.exit(f"FAIL: 相位 5 行数 {span} 偏离实测 405——结构漂移")

    moved = [node]
    if len(moved) != EXPECTED_MOVED:
        sys.exit("FAIL: 搬移语句数不符")

    # ② 区域体检
    region = lines[node.lineno - 1: node.end_lineno]
    for ln in region:
        if ln.strip() and not ln.startswith(" " * 12):
            sys.exit(f"FAIL: 搬移区缩进异常：{ln!r}")
    if any('"""' in ln or "'''" in ln for ln in region):
        sys.exit("FAIL: 搬移区含三引号（去缩进会改字符串内容）")
    if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
            and isinstance(node.body[0].value.value, str):
        sys.exit("FAIL: 首语句为字符串字面量，与 docstring 剥离冲突")

    # ③ 参数重算 + 对账
    B = set(dir(builtins))
    mod = ast.Module(body=moved, type_ignores=[])
    inner_binds = set()
    for sc in ast.walk(mod):
        if isinstance(sc, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) \
                or hasattr(sc, "generators"):
            inner_binds |= binds_ms([sc])
    mv_writes = binds_ms(moved) - inner_binds
    mv_loads = {n.id for s in moved for n in ast.walk(s)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    free = mv_loads - binds_ms(moved) - B - inner_binds
    fi = next(i for i, s in enumerate(m.body) if s is loop)
    caller_binds = binds_ms(body[:idx]) | binds_ms(body[idx + 1:]) \
        | {a.arg for a in m.args.args + m.args.kwonlyargs + m.args.posonlyargs}
    if m.args.vararg:
        caller_binds.add(m.args.vararg.arg)
    if m.args.kwarg:
        caller_binds.add(m.args.kwarg.arg)
    caller_binds |= {n.id for n in ast.walk(loop.target) if isinstance(n, ast.Name)}
    params = sorted((free & caller_binds) - {"self"})
    if params != EXPECTED_PARAMS:
        sys.exit(f"FAIL: 参数表漂移 多{sorted(set(params) - set(EXPECTED_PARAMS))} "
                 f"少{sorted(set(EXPECTED_PARAMS) - set(params))}")

    # ④ 写逃逸复检：清理块 + 循环后 + 循环体其余部分的全部 Load
    rest_reads = {n.id for i, s in enumerate(body) if i != idx
                  for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    post_reads = {n.id for s in m.body[fi + 1:] for n in ast.walk(s)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    esc = mv_writes & (rest_reads | post_reads)
    if esc:
        sys.exit(f"FAIL: 写逃逸 {sorted(esc)}——需返回值协议，本执行器不支持")

    # ⑤ 控制流复检
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
    rets = bad_returns(moved)
    if rets:
        sys.exit(f"FAIL: 方法级 Return @ {rets}")
    mv_loops = [(x.lineno, x.end_lineno) for x in ast.walk(mod)
                if isinstance(x, (ast.For, ast.While, ast.AsyncFor))]
    cbs = [(type(n).__name__, n.lineno) for n in ast.walk(mod)
           if isinstance(n, (ast.Continue, ast.Break))
           and not any(a <= n.lineno <= b for a, b in mv_loops)]
    if cbs:
        sys.exit(f"FAIL: 窗口级 continue/break @ {cbs}")

    # ⑥ 组装（新行一律 CRLF）
    def dedent4(txt):
        return "".join(ln[4:] if ln.startswith(" " * 12) else ln
                       for ln in txt.splitlines(keepends=True))
    body_text = dedent4("".join(region))
    sig = [f"    def {NEW}(", "        self,", "        *,"]
    sig += [f"        {p}," for p in EXPECTED_PARAMS]
    sig.append("    ):")
    method = CRLF.join(sig) + CRLF + DOCSTRING + body_text
    if not method.endswith(CRLF):
        method += CRLF

    call = [f"            self.{NEW}("]
    call += [f"                {p}={p}," for p in EXPECTED_PARAMS]
    call.append("            )")
    call_lines = [(ln if ln.endswith(CRLF) else ln + CRLF) for ln in call]

    # ⑦ 手术
    out = lines[:node.lineno - 1] + call_lines + lines[node.end_lineno:]
    text = "".join(out)
    if not text.endswith(CRLF):
        text += CRLF
    text = text + CRLF + method
    ast.parse(text)
    with open(FILE, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)

    idx_in_loop = idx
    print(f"✅ {NEW}: 搬移 {span} 行（L{node.lineno}-{node.end_lineno}），"
          f"形参 {len(EXPECTED_PARAMS)} 个（keyword-only），调用点 {len(call_lines)} 行")
    print(f"   verify_extract.py work/commit_frozen_f_k.before.py {FILE.name} "
          f"--moved {NEW} --src {FN} --block for --index 0 --slice {idx_in_loop}:{idx_in_loop + 1}")


if __name__ == "__main__":
    main()
