#!/usr/bin/env python3
"""刀1 A 段执行器（CUTTING_PLAN.md §5 序2、§2.2/§2.3）：_produce_window 抽取。

把 run_all_windows 窗口循环体的**生产段**（相位 6-8 主体，实测 body[129..210]
共 82 条语句，L4044-5271 含生产段头注释）原样搬为同类方法 _produce_window；
调用点以 keyword 实参逐名传入（形参 keyword-only，杜绝 30 参错序）。
纯搬运：方法体逐字不变（仅去 4 格缩进），门 = verify_extract.py --block for --tail 82。

与 §2.2 草图的两处实测修正（2026-09-20，量过才下刀，已记 PROGRESS.md）：
  1. `if warmup_only:` 块（L3998-4023，内含绑定窗口 for 的 continue 与引用清理）
     **留在调用方**——continue 搬进方法即语法错误；
  2. 窗口尾引用释放协议（L5274-5307：probe If ×2、del sampler、del sim.context、
     sim = None、ibs_wrap/_nan_diagnostics/win_sys = None、gc.collect()）**留在
     调用方**——源码注释明说这是防「代理悬垂到下一个窗口」的生命周期语义，
     搬走会改变调用方绑定的释放时机。
写逃逸实测为无（搬移区方法作用域写集 ∩ (清理块读集 ∪ 循环后读集) = ∅），无返回值。
循环携带裁决：`e`/`f` 头部各仅 1 处 Load 且都在自身绑定结构内
（L875 except 处理器 f-string、L658 json.load(f)），搬移区内的 e/f 成为方法局部，等价。

fail-closed：地标漂移（warmup_only 块 / probe If / 末条「窗口完成」print）/
参数表与权威名单不符 / 条数漂移 / 方法级 Return / 窗口级 continue·break /
缩进异常 / 区域内三引号 / 首语句为字符串字面量（与 docstring 剥离冲突）→ 拒写。
"""
import ast
import builtins
import pathlib
import sys

FILE = pathlib.Path(__file__).resolve().parent / "windows_sampling" / "window_manager_dual_lambda.py"
KLASS = "IBSWindowManagerDualLambda"
FN = "run_all_windows"
NEW = "_produce_window"
EXPECTED_MOVED = 82
EXPECTED_PARAMS = [
    "K", "debug_mode", "early_stop_check_interval_steps",
    "early_stop_max_delta_g_drift_kJ_mol", "early_stop_max_uncertainty_kJ_mol",
    "early_stop_min_absolute_ess", "early_stop_min_decorrelated_samples",
    "early_stop_min_ess_ratio", "early_stop_min_steps",
    "early_stop_required_consecutive_passes", "enable_early_stop", "end",
    "ibs_state_file", "ibs_wrap", "lc_win", "lse_log_residual_tolerance",
    "lv_win", "n_steps_per_window", "production_step_overrides",
    "repair_policy", "sampler", "sim", "skip_frozen_revalidation",
    "stage_type", "start", "steps_per_update", "warm", "win_sys",
    "win_sys_xml", "window_idx",
]
assert len(EXPECTED_PARAMS) == 30 and EXPECTED_PARAMS == sorted(EXPECTED_PARAMS)

DOCSTRING = '        """相位 6-8（生产准备/生产循环/收口）—— 刀1 A 段自 run_all_windows 原样搬出。\n\n        纯搬运（verify_extract --block for --tail 82 逐节点等价）；跨界变量按原名\n        作 keyword-only 形参。窗口级控制流与引用释放协议（if warmup_only 块、\n        del sampler / sim = None / …）留在 run_all_windows 调用点\n        （CUTTING_PLAN.md §2.2/§2.3，实测修正见本文件头注释）。\n        """\n'


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

    # ① 地标
    wo = [s for s in body if isinstance(s, ast.If)
          and isinstance(s.test, ast.Name) and s.test.id == "warmup_only"
          and s.end_lineno - s.lineno > 20]
    if len(wo) != 1:
        sys.exit(f"FAIL: warmup_only 大块地标 {len(wo)} 处（预期 1）")
    wo_i = body.index(wo[0])

    def is_probe_if(s):
        if not isinstance(s, ast.If):
            return False
        return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "getattr" and len(n.args) >= 2
                   and isinstance(n.args[0], ast.Name) and n.args[0].id == "sampler"
                   and getattr(n.args[1], "value", "") == "_probe_context"
                   for n in ast.walk(s.test))
    probe = [s for s in body if is_probe_if(s)]
    if len(probe) != 1:
        sys.exit(f"FAIL: 清理块地标（probe If）{len(probe)} 处（预期 1）")
    probe_i = body.index(probe[0])
    if probe_i <= wo_i:
        sys.exit("FAIL: 清理块在 warmup_only 块之前——结构漂移")
    last = body[probe_i - 1]
    if not (isinstance(last, ast.Expr) and "完成，能量已保存" in ast.unparse(last)):
        sys.exit(f"FAIL: 搬移区末条不是「窗口完成」print：L{last.lineno} {ast.unparse(last)[:60]}")

    moved = body[wo_i + 1: probe_i]
    if len(moved) != EXPECTED_MOVED:
        sys.exit(f"FAIL: 搬移语句 {len(moved)} 条 ≠ 权威 {EXPECTED_MOVED} 条")
    if isinstance(moved[0], ast.Expr) and isinstance(moved[0].value, ast.Constant) \
            and isinstance(moved[0].value.value, str):
        sys.exit("FAIL: 搬移区首语句是字符串字面量，与 docstring 剥离冲突")

    # ② 行区间（前导注释块归属搬移区）
    first_ln = moved[0].lineno
    j = first_ln - 2
    while j >= wo[0].end_lineno and lines[j].strip().startswith("#"):
        first_ln = j + 1
        j -= 1
    last_ln = moved[-1].end_lineno
    region = lines[first_ln - 1: last_ln]
    for ln in region:
        if ln.strip() and not ln.startswith(" " * 12):
            sys.exit(f"FAIL: 搬移区缩进异常：{ln!r}")
    if any('"""' in ln or "'''" in ln for ln in region):
        sys.exit("FAIL: 搬移区含三引号（去缩进会改字符串内容），需人工处理")

    # ③ 参数重算 + 对账
    mod = ast.Module(body=moved, type_ignores=[])
    inner_binds = set()
    for sc in ast.walk(mod):
        if isinstance(sc, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) \
                or hasattr(sc, "generators"):
            inner_binds |= binds_ms([sc])
    mv_writes = binds_ms(moved) - inner_binds
    mv_loads = {n.id for s in moved for n in ast.walk(s)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    B = set(dir(builtins))
    free = mv_loads - binds_ms(moved) - B - inner_binds
    caller_binds = binds_ms(body[:wo_i + 1]) | binds_ms(body[probe_i:]) \
        | {a.arg for a in m.args.args + m.args.kwonlyargs + m.args.posonlyargs}
    if m.args.vararg:
        caller_binds.add(m.args.vararg.arg)
    if m.args.kwarg:
        caller_binds.add(m.args.kwarg.arg)
    caller_binds |= {n.id for n in ast.walk(loop.target) if isinstance(n, ast.Name)}
    params = sorted(free & caller_binds - {"self"})
    if params != EXPECTED_PARAMS:
        sys.exit(f"FAIL: 参数表漂移 多{sorted(set(params) - set(EXPECTED_PARAMS))} "
                 f"少{sorted(set(EXPECTED_PARAMS) - set(params))}")

    # ④ 写逃逸复检（清理块读 ∪ 循环后读）
    fi = next(i for i, s in enumerate(m.body) if s is loop)
    cleanup_reads = {n.id for s in body[probe_i:] for n in ast.walk(s)
                     if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    post_reads = {n.id for s in m.body[fi + 1:] for n in ast.walk(s)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    esc = mv_writes & (cleanup_reads | post_reads)
    if esc:
        sys.exit(f"FAIL: 写逃逸 {sorted(esc)}——需要返回值协议，本执行器不支持")

    # ⑤ 控制流复检
    def bad_returns(stmts):
        out = []

        def walk(node, in_fn):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    walk(ch, True)
                elif isinstance(ch, ast.Return) and not in_fn:
                    out.append(ch.lineno)
                else:
                    walk(ch, in_fn)
        for s in stmts:
            walk(s, False)
        return out
    mv_loops = [(x.lineno, x.end_lineno) for x in ast.walk(mod)
                if isinstance(x, (ast.For, ast.While, ast.AsyncFor))]
    rets = bad_returns(moved)
    if rets:
        sys.exit(f"FAIL: 方法级 Return @ {rets}")
    cbs = [(type(n).__name__, n.lineno) for n in ast.walk(mod)
           if isinstance(n, (ast.Continue, ast.Break))
           and not any(a <= n.lineno <= b for a, b in mv_loops)]
    if cbs:
        sys.exit(f"FAIL: 窗口级 continue/break @ {cbs}")

    # ⑥ 组装
    def dedent4(txt):
        return "".join(ln[4:] if ln.startswith(" " * 12) else ln
                       for ln in txt.splitlines(keepends=True))
    body_text = dedent4("".join(region))
    sig = [f"    def {NEW}(", "        self,", "        *,"]
    sig += [f"        {p}," for p in EXPECTED_PARAMS]
    sig.append("    ):")
    method = "\n".join(sig) + "\n" + DOCSTRING + body_text
    if not method.endswith("\n"):
        method += "\n"

    call = ["            self._produce_window("]
    call += [f"                {p}={p}," for p in EXPECTED_PARAMS]
    call.append("            )\n")

    # ⑦ 手术：删搬移行、插调用、类尾加方法
    out = lines[:first_ln - 1] + [ln if ln.endswith("\n") else ln + "\n" for ln in call] \
        + lines[last_ln:]
    text = "".join(out)
    if not text.endswith("\n"):
        text += "\n"
    text = text + "\n" + method
    ast.parse(text)

    with open(FILE, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)

    new_len = last_ln - first_ln + 1
    print(f"✅ {NEW}: 搬移 {len(moved)} 条语句 / {new_len} 行（L{first_ln}-{last_ln}），"
          f"形参 {len(EXPECTED_PARAMS)} 个（keyword-only），调用点 {len(call)} 行；"
          f"留调用方：warmup_only 块 + 释放协议块")
    print(f"   verify_extract.py work/produce_window.before.py {FILE.name} "
          f"--moved {NEW} --src {FN} --block for --index 0 --tail {EXPECTED_MOVED}")


if __name__ == "__main__":
    main()
