#!/usr/bin/env python3
"""可重放切分驱动：固定刀序 + 每刀过门 + 失败即停。

    python3 refactor/tools/replay.py history              # 回归：每刀从历史留底出发，对账历史产物 + 过门
    python3 refactor/tools/replay.py mainline             # 对当前主线三文件整链重放 → refactor/replay/out/
    python3 refactor/tools/replay.py mainline --shadow    # 另建影子仓库（切分版三文件 + 其余照主线）供跑测试

一律在仓库根执行。主线只读：mainline 模式把三文件**复制**进 refactor/replay/ 再切。

两类执行体：
  spec —— 通用引擎 cutter.py + cuts/specs/<name>.py（8 刀）
  exec —— 保留的特化执行器 cuts/_*.py（形状独特的 7 刀：warmup_state 名单口径、
          absorb3、两次跨多方法归组、scheme_dispatch 双方法+分派、resume_preoptimized、
          decide_once_step0 ×8），经 CUT_FILE 指向目标文件。

门：每刀跑它在 work/verify/<刀>.verify.sh 里的那组门，下标从产物结构现算（不抄死值）。
行尾：目标文件原本无 CRLF 时，刀后把混入的 CRLF 归一为 LF（历史执行器有硬写 \\r\\n 的）。
"""
import argparse
import ast
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RF = REPO / "refactor"
W = RF / "work"
sys.path.insert(0, str(RF / "tools"))
import cutter  # noqa: E402

# 类 → (主线文件, 历史 domains 文件)
FILES = {
    "pipe": ("abfe_pipeline.py", "pipeline_orchestration/abfe_pipeline_class.py", "ABFEPipeline"),
    "ibs": ("ibs_engine.py", "windows_sampling/window_manager_dual_lambda.py",
            "IBSWindowManagerDualLambda"),
    "ctl": ("abfe_preoptimizer.py", "stage2_controller/stage2_repair_controller.py",
            "Stage2RepairController"),
}

ES = ("early_stop_check_interval_steps,early_stop_max_delta_g_drift_kJ_mol,"
      "early_stop_max_uncertainty_kJ_mol,early_stop_min_absolute_ess,"
      "early_stop_min_decorrelated_samples,early_stop_min_ess_ratio,early_stop_min_steps,"
      "early_stop_required_consecutive_passes,enable_early_stop")
WN = f"{W}/warmup_state_names.txt"
PN = f"{W}/pipeline_state_names.txt"


# ───────────────────────── 门参数：从产物结构现算下标 ─────────────────────────

def _tree(p):
    return ast.parse(cutter.read_bytes(p))


def _m(tree, key, fn):
    return cutter.find_method(tree, FILES[key][2], fn)[1]


def _extract_gate(key, src, new, loop):
    """抽取刀：调用点在容器里的下标 k + 新方法体条数 n ⟹ --slice k:k+n。"""
    def g(before, after):
        t = _tree(after)
        m = _m(t, key, src)
        body = cutter.body_no_doc(m)
        if loop:
            body = next(s for s in body if isinstance(s, (ast.For, ast.While))).body
        k = [i for i, s in enumerate(body)
             if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
             and ast.unparse(s.value.func) == f"self.{new}"]
        if len(k) != 1:
            sys.exit(f"FAIL[gate]: self.{new}(...) 调用点 {len(k)} 处")
        n = len(cutter.body_no_doc(_m(t, key, new)))
        blk = ["--block", "for", "--index", "0"] if loop else ["--block", "fn"]
        return [["verify_extract.py", before, after, "--moved", new, "--src", src,
                 *blk, "--slice", f"{k[0]}:{k[0] + n}"]]
    return g


def _rename_gate(*argsets):
    def g(before, after):
        return [["verify_rename.py", before, after, *a] for a in argsets]
    return g


def _scheme_gate(before, after):
    m = _m(_tree(before), "pipe", "run_full_pipeline")
    ifs = [s for s in cutter.body_no_doc(m) if isinstance(s, ast.If)]
    i = len(ifs) - 1                      # 尾部 elif 链 = 最后一条顶层 If
    pos = len(m.body) - 1
    base = [before, after, "--src", "run_full_pipeline"]
    return [["verify_extract.py", *base, "--moved", "_run_dual_lambda_scheme",
             "--block", "if", "--index", str(i)],
            ["verify_extract.py", *base, "--moved", "_run_single_lambda_scheme",
             "--block", "if", "--index", str(i), "--orelse"],
            ["verify_dispatch.py", *base, "--pos", str(pos)]]


def _step0_gate(name, fixups):
    def g(before, after):
        m = _m(_tree(before), "ctl", "_decide_once")
        defs = [n.name for n in cutter.body_no_doc(m) if isinstance(n, ast.FunctionDef)]
        strip = []
        for fx in fixups:
            strip += ["--strip-call", f"{fx}={len(STEP0_CAPS[fx])}"]
        return [["verify_extract.py", before, after, "--moved", name, "--src", "_decide_once",
                 "--block", "def", "--index", str(defs.index(name)), *strip]]
    return g


def _load_step0_caps():
    p = RF / "cuts" / "_decide_once_step0.py"
    sp = importlib.util.spec_from_file_location("_s0", p)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod.CAPTURES


STEP0_CAPS = _load_step0_caps()


# ───────────────────────────── 刀序（唯一权威） ─────────────────────────────
# hist = (历史留底, 历史产物)；产物 None = 链末，对 domains 当前文件。
# 每条链内顺序即依赖序（后刀的锚点依赖前刀留下的调用点/数据类）。

def S(name, key, kind, gate, hist, **kw):
    return dict(name=name, key=key, kind=kind, gate=gate, hist=hist, **kw)


STEPS = [
    # ── ABFEPipeline（abfe_pipeline.py）──
    S("scheme_dispatch", "pipe", "exec", _scheme_gate,
      ("scheme_dispatch", "pipeline_state"), exe="_scheme_dispatch_cut.py",
      expect_diff="−cached/e/f/final、−f/final/point/raw_path/values（幻影形参，调用点未必已绑定）",
      bound=["_run_dual_lambda_scheme", "_run_single_lambda_scheme"],
      bound_ok={"analysis_context": "run_full_pipeline 在 `if decoupling_scheme == \"dual_lambda\":` "
                "里赋值，dual 分派调用点在同一条件下（门不做分支条件相关性分析）"}),
    S("pipeline_state", "pipe", "spec", _rename_gate(
        ["--fn", "_run_dual_lambda_scheme", "--bind", "dual", "--names-file", PN,
         "--dataclass", "PipelineState", "--in-place", "--construction", "1"]),
      ("pipeline_state", "pipeline_state2")),
    S("pipeline_state2", "pipe", "spec", _rename_gate(
        ["--fn", "_run_dual_lambda_scheme", "--bind", "dual", "--names", "stage1_states,opt_res",
         "--keep-file", PN, "--dataclass", "PipelineState", "--in-place", "--construction", "1"]),
      ("pipeline_state2", "partition_windows")),
    S("partition_windows", "pipe", "spec",
      _extract_gate("pipe", "_run_dual_lambda_scheme", "_partition_windows", False),
      ("partition_windows", "resume_preoptimized"), bound=["_partition_windows"]),
    S("resume_preoptimized", "pipe", "exec",
      _extract_gate("pipe", "_run_dual_lambda_scheme", "_resume_preoptimized", False),
      ("resume_preoptimized", None), exe="_resume_preoptimized_cut.py",
      expect_diff="−cached（幻影）",
      bound=["_resume_preoptimized"]),

    # ── IBSWindowManagerDualLambda（ibs_engine.py）──
    S("warmup_state", "ibs", "exec", _rename_gate(
        ["--fn", "run_all_windows", "--bind", "warm", "--names-file", WN,
         "--dataclass", "WarmupState", "--block", "for", "--index", "0", "--construction", "1"]),
      ("warmup_state", "produce_window"), exe="_warmup_state_cut.py"),
    S("produce_window", "ibs", "spec",
      _extract_gate("ibs", "run_all_windows", "_produce_window", True),
      ("produce_window", "commit_frozen_f_k"), bound=["_produce_window"]),
    S("commit_frozen_f_k", "ibs", "spec",
      _extract_gate("ibs", "run_all_windows", "_commit_frozen_f_k", True),
      ("commit_frozen_f_k", "absorb3"), expect_diff="+legacy_repair 形参（修历史漏传）",
      bound=["_commit_frozen_f_k"]),
    S("absorb3", "ibs", "exec", _rename_gate(
        ["--fn", "run_all_windows", "--bind", "warm",
         "--names", "frozen_f_k_snapshot,mode,ema_mean_p_values",
         "--keep-file", WN, "--dataclass", "WarmupState"]),
      ("absorb3", "run_warmup_loop"), exe="_warmup_state_absorb3.py"),
    S("run_warmup_loop", "ibs", "spec",
      _extract_gate("ibs", "run_all_windows", "_run_warmup_loop", True),
      ("run_warmup_loop", "summarize_warmup"), bound=["_run_warmup_loop"],
      expect_diff="−k 形参（推导式局部幻影，调用点未必已绑定）"),
    S("summarize_warmup", "ibs", "spec",
      _extract_gate("ibs", "run_all_windows", "_summarize_warmup", True),
      ("summarize_warmup", "production_state"), bound=["_summarize_warmup"],
      expect_diff="+legacy_repair 形参（修历史漏传）；−k（幻影）"),
    S("production_state", "ibs", "spec", _rename_gate(
        ["--fn", "_produce_window", "--bind", "prod", "--names-file",
         f"{W}/production_state_names.txt", "--keep-file", WN,
         "--dataclass", "ProductionState", "--in-place", "--construction", "1"]),
      ("production_state", "finish_production")),
    S("finish_production", "ibs", "spec",
      _extract_gate("ibs", "_produce_window", "_finish_production", False),
      ("finish_production", "early_stop_config"), bound=["_finish_production"]),
    S("early_stop_config", "ibs", "exec", _rename_gate(
        ["--fn", "_finish_production", "--bind", "es", "--names", ES,
         "--dataclass", "EarlyStopConfig", "--in-place"],
        ["--fn", "_produce_window", "--bind", "es", "--names", ES,
         "--dataclass", "EarlyStopConfig", "--in-place", "--expand-kwarg"],
        ["--fn", "run_all_windows", "--bind", "es", "--names", ES,
         "--dataclass", "EarlyStopConfig", "--in-place", "--construction", "1",
         "--expand-kwarg"]),
      ("early_stop_config", "window_context"), exe="_early_stop_config_cut.py",
      bound=["_produce_window", "_finish_production"], bound_args=["es"]),
    S("window_context", "ibs", "exec", _rename_gate(
        ["--fn", "run_all_windows", "--bind", "wc", "--names",
         "lc_win,lv_win,sampler,sim,win_sys,win_sys_xml", "--dataclass", "WindowContext",
         "--block", "for", "--in-place", "--construction", "1", "--expand-kwarg"],
        *[["--fn", fn, "--bind", "wc", "--names", "sampler,sim,lc_win,lv_win",
           "--dataclass", "WindowContext", "--in-place"]
          for fn in ("_run_warmup_loop", "_summarize_warmup", "_commit_frozen_f_k")],
        ["--fn", "_produce_window", "--bind", "wc", "--names",
         "lc_win,lv_win,sampler,sim,win_sys,win_sys_xml", "--dataclass", "WindowContext",
         "--in-place", "--expand-kwarg"],
        ["--fn", "_finish_production", "--bind", "wc", "--names",
         "sampler,sim,win_sys,lc_win,lv_win", "--dataclass", "WindowContext", "--in-place"]),
      ("window_context", None), exe="_window_context_cut.py",
      bound=["_run_warmup_loop", "_summarize_warmup", "_commit_frozen_f_k",
             "_produce_window", "_finish_production"], bound_args=["wc"],
      expect_diff="只改写 6 个目标方法（旧版误改 5 个无关方法 42 处 → NameError: wc）"),
]

# ── Stage2RepairController._decide_once 内嵌 def 提升（abfe_preoptimizer.py）──
# 顺序 = 被调者先行（v2）。被调者提升时执行器在**全文件**调用点插实参（含尚未提升的
# 调用者体内），所以 v2 下 --fixup 一次都不需要（它是 v1 刀序的遗留；多开会重复插参）。
_S0 = [("_pw_or_last", [], "step0_1_pw_or_last"), ("_attr_of", [], "step0_2_attr_of"),
       ("_is_skew", [], "step0_3_is_skew"), ("_pick", [], "step0_4_pick"),
       ("_frames_admission", [], "step0_5_frames_admission"), ("_is_noop", [], "step0_6_is_noop"),
       ("plan", [], "step0_7_plan"), ("_bounded_rewindow", [], "step0_8_bounded_rewindow")]
# ⛔ 后 4 刀挂起（09-23 verify_bound 抓出，历史 8/8 的「门 PASS」不成立）：
#   plan —— _decide_once 有 3 个 `return plan(...)` 早退出口在 `_blocks…_noop_led`
#     8 名赋值之前；闭包是调用时才读、且只在用到的分支读，提升成模块函数后调用点
#     当场求值 ⟹ UnboundLocalError（收敛 stage 走 DONE/NO_ACTION 即炸，测试实测 7 处）。
#   _frames_admission / _is_noop —— 仍从内嵌 plan 里调用，捕获名的时序同上，静态判不了。
#   _bounded_rewindow —— 依赖 plan 先提升。
# 解锁需要改刀法（改行为，要用户拍板），见 work/PROGRESS.md「step0 后 4 刀挂起」。
STEP0_BLOCKED = [x[0] for x in _S0[4:]]
_S0 = _S0[:4]
for i, (nm, fx, hb) in enumerate(_S0):
    nxt = _S0[i + 1][2] if i + 1 < len(_S0) else "step0_5_frames_admission"
    STEPS.append(S(f"step0_{nm.lstrip('_')}", "ctl", "exec", _step0_gate(nm, fx), (hb, nxt),
                   exe="_decide_once_step0.py", bound=[nm],
                   argv=["--def", nm] + sum((["--fixup", f] for f in fx), [])))


# ─────────────────────────────── 执行 ───────────────────────────────

def _run_step(st, target):
    """对 target 文件原地执行一刀。返回 None 或失败信息。"""
    had_crlf = "\r\n" in cutter.read_bytes(target)
    if st["kind"] == "spec":
        sp = importlib.util.spec_from_file_location(
            st["name"], RF / "cuts" / "specs" / f"{st['name']}.py")
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        spec = dict(mod.SPEC, file=target.name)
        try:
            cutter.MODES[spec["mode"]](spec, str(target.parent), False)
        except SystemExit as e:
            return str(e)
    else:
        r = subprocess.run([sys.executable, str(RF / "cuts" / st["exe"]), *st.get("argv", [])],
                           env=dict(os.environ, CUT_FILE=str(target)),
                           capture_output=True, text=True, cwd=REPO)
        if r.returncode:
            return (r.stdout + r.stderr).strip()
        for ln in r.stdout.splitlines():
            if "漂移" in ln:
                print(ln)
    if not had_crlf:
        txt = cutter.read_bytes(target)
        if "\r\n" in txt:
            cutter.write_bytes(target, txt.replace("\r\n", "\n"))
    return None


def _run_gates(st, before, after):
    argvs = st["gate"](str(before), str(after))
    if st.get("bound"):
        # 调用点实参必然已绑定（09-23 补）：抽取/提升类的调用点是新写代码，
        # 上面那组门都不验它。
        argvs.append(["verify_bound.py", str(after)]
                     + sum((["--callee", c] for c in st["bound"]), [])
                     + sum((["--arg", x] for x in st.get("bound_args", ())), [])
                     + sum((["--allow", f"{k}={v}"] for k, v in st.get("bound_ok", {}).items()), []))
    for argv in argvs:
        r = subprocess.run([sys.executable, str(RF / "tools" / argv[0]), *argv[1:]],
                           capture_output=True, text=True, cwd=REPO)
        tail = [ln for ln in r.stdout.splitlines() if "PASS" in ln or "FAIL" in ln]
        if r.returncode or not any("PASS" in ln for ln in tail):
            return f"{argv[0]} {' '.join(argv[3:])[:300]}\n" + (r.stdout + r.stderr)[-2000:]
        for ln in r.stdout.splitlines():
            if "[豁免]" in ln:
                print(ln)
    return None


def _norm_ast(p):
    return ast.dump(cutter._norm(_tree(p)))


def cmd_history(a):
    tmp = RF / "replay" / "history"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    bad = 0
    for st in STEPS:
        if a.only and st["name"] not in a.only:
            continue
        hb, ha = st["hist"]
        before = W / f"{hb}.before.py"
        after = (W / f"{ha}.before.py") if ha else RF / "domains" / FILES[st["key"]][1]
        target = tmp / f"{st['name']}.py"
        shutil.copyfile(before, target)
        err = _run_step(st, target) or _run_gates(st, before, target)
        if err:
            print(f"  ❌ {st['name']}\n     " + err.replace("\n", "\n     ")[:3000])
            bad += 1
            continue
        same_b = (cutter.read_bytes(target).replace("\r\n", "\n")
                  == cutter.read_bytes(after).replace("\r\n", "\n"))
        same_a = _norm_ast(target) == _norm_ast(after)
        if same_b and same_a:
            print(f"  ✅ {st['name']:<28} 门 PASS + 与历史产物逐字节(行尾归一)相同")
        elif st.get("expect_diff"):
            print(f"  ✅ {st['name']:<28} 门 PASS；与历史产物不同（预期：{st['expect_diff']}）")
        else:
            print(f"  ❌ {st['name']:<28} 门 PASS 但与历史产物不同（字节{'✅' if same_b else '❌'} "
                  f"AST{'✅' if same_a else '❌'}）；产物留在 {target}")
            bad += 1
    print(f"\n{'全绿' if not bad else f'{bad} 刀失败'}")
    return 1 if bad else 0


def _duplicate_defs(p):
    """模块顶层 / 类体内的重名 def·class（后者静默覆盖前者，py_compile 不报）。"""
    t = _tree(p)
    out = []
    for scope, body in [("<module>", t.body)] + [
            (n.name, n.body) for n in t.body if isinstance(n, ast.ClassDef)]:
        seen = set()
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if n.name in seen:
                    out.append(f"{scope}.{n.name}")
                seen.add(n.name)
    return out


def _ensure_dataclass_import(p):
    """主线接线：数据类用 @dataclass/field，主线三文件原本不 import dataclasses。

    插在第一个数据类声明之前（模块顶层 import，非惰性）。历史 domains 快照是
    跑不起来的切片，从没需要过这一行。
    """
    txt = cutter.read_bytes(p)
    t = ast.parse(txt)
    have = {a.asname or a.name for n in t.body if isinstance(n, ast.ImportFrom)
            and n.module == "dataclasses" for a in n.names}
    if {"dataclass", "field"} <= have:
        return False
    first = next((n for n in t.body if isinstance(n, ast.ClassDef) and any(
        "dataclass" in ast.unparse(d) for d in n.decorator_list)), None)
    if first is None:
        return False
    lines = txt.splitlines(keepends=True)
    at = min(d.lineno for d in first.decorator_list) - 1
    lines.insert(at, "from dataclasses import dataclass, field\n\n\n")
    cutter.write_bytes(p, "".join(lines))
    return True


def cmd_mainline(a):
    out = RF / "replay" / "out"
    snaps = RF / "replay" / "steps"
    for d in (out, snaps):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for key, (main, _, _) in FILES.items():
        shutil.copyfile(REPO / main, out / main)
    for i, st in enumerate(STEPS):
        target = out / FILES[st["key"]][0]
        before = snaps / f"{i:02d}_{st['name']}.before.py"
        shutil.copyfile(target, before)
        err = _run_step(st, target)
        if not err:
            err = _run_gates(st, before, target)
        if err:
            print(f"  ❌ 第 {i} 刀 {st['name']} 失败——链停在这里（{target.name} 保持该刀之前的状态）\n     "
                  + err.replace("\n", "\n     ")[:3000])
            shutil.copyfile(before, target)
            return 1
        print(f"  ✅ {i:02d} {st['name']:<28} 门 PASS")
    for key, (main, _, _) in FILES.items():
        p = out / main
        if _ensure_dataclass_import(p):
            print(f"  [接线] {main}: + from dataclasses import dataclass, field")
        dup = _duplicate_defs(p)
        if dup:
            print(f"  ❌ {main}: 重名定义 {dup}——提升/抽取出的名字覆盖了既有定义")
            return 1
        r = subprocess.run([sys.executable, "-m", "py_compile", str(p)], capture_output=True, text=True)
        if r.returncode:
            print(f"  ❌ py_compile {main}\n{r.stderr}")
            return 1
        n0 = len(cutter.read_bytes(REPO / main).splitlines())
        n1 = len(cutter.read_bytes(p).splitlines())
        print(f"  {main}: {n0} → {n1} 行")
    print(f"\n全链 {len(STEPS)} 刀完成；产物 {out.relative_to(REPO)}/")
    if a.shadow:
        _build_shadow(out)
    return 0


def _build_shadow(out):
    """影子仓库：顶层 .py 与 tests/ 实拷（测试按 __file__ 定位仓库根，符号链接会被
    resolve 回主线），其余符号链接；三文件换成切分版。"""
    sh = RF / "replay" / "shadow"
    shutil.rmtree(sh, ignore_errors=True)
    sh.mkdir(parents=True)
    for p in REPO.iterdir():
        if p.name in ("refactor", ".git", "__pycache__", "output", "tmp"):
            continue
        dst = sh / p.name
        if p.is_file() and p.suffix in (".py", ".ini", ".toml", ".json", ".yml", ".md"):
            shutil.copyfile(p, dst)
        elif p.name == "tests":
            shutil.copytree(p, dst, ignore=shutil.ignore_patterns("__pycache__"), symlinks=True)
        else:
            dst.symlink_to(p)
    for main, _, _ in FILES.values():
        shutil.copyfile(out / main, sh / main)
    print(f"  影子仓库：{sh.relative_to(REPO)}/  →  cd 进去跑 ./tests/run_offline_tests.sh")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("history")
    h.add_argument("--only", nargs="*")
    m = sub.add_parser("mainline")
    m.add_argument("--shadow", action="store_true")
    a = ap.parse_args()
    os.chdir(REPO)
    return {"history": cmd_history, "mainline": cmd_mainline}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
