#!/usr/bin/env python3
"""刀1 第1步执行器（CUTTING_PLAN.md §5 序1、§0.2）：WarmupState 数据类收拢。

51 条散落在 run_all_windows 相位2 的常量型初始化收成模块级 @dataclass WarmupState，
体内全部读写改写为 warm.<field>。承载名用 warm（§2.2 刀1 拆分草图
`warm = self._warmup_window(ctx, ...)` 的返回名）；`state` 已被循环体内
OpenMM getState 局部量占用（L2969），不可用。

变换形状（归组变换，非纯搬运；门 = verify_rename.py --block for）：
  ① 文件头加 import + @dataclass class WarmupState 声明
     （字段顺序 = 原 init 行序；默认值 = 原初值，门③逐条对账）；
  ② 51 条 init 语句删除，原首位插入一行 `warm = WarmupState()`；
  ③ 方法子树内 51 个名字的全部引用改写为 warm.<name>
     （实测零内层遮蔽、零循环外引用；f-string 内引用在 py3.12 下是真实
     AST 节点，一并改写；无 {x=} 调试说明符，输出文本不变）。

fail-closed：名单按下方口径重算并与 EXPECTED 逐一比对（条数+顺序+名字），
任何异常形态（非单行 init / 遮蔽 / walrus / del / global / match / warm 撞名 /
循环外引用 / 模块常量多定义）都拒绝写文件。

名单口径（2026-09-20 实测，复现 §0.2 的 51）：循环体顶层、单名目标
（Assign/AnnAssign）、相位2区间 [L1204, L2088]、RHS ∈ {纯字面量,
float('nan'|'inf'), 全大写模块常量名} 的 52 条候选**排除 check_chunk**——
它在全循环仅 1 Store + 14 Load，是 IBS_WARMUP_FRAME_STRIDE_STEPS 的纯别名、
非累加器，且是 52→51 唯一有语义依据的排除项。0.2 的「51」从未落盘过名单，
本口径为唯一可复现重建，已记入 work/PROGRESS.md。
"""
import ast
import os
import pathlib
import sys

# 重放（replay.py）用 CUT_FILE 指向目标文件；缺省＝历史 domains 路径。
FILE = pathlib.Path(os.environ["CUT_FILE"]) if os.environ.get("CUT_FILE") else pathlib.Path(__file__).resolve().parent / "windows_sampling" / "window_manager_dual_lambda.py"
KLASS = "IBSWindowManagerDualLambda"
FN = "run_all_windows"
BIND = "warm"
DATACLASS = "WarmupState"
# 相位2 区间按**结构锚**定位（原为行号 (1204, 2088)，主线一漂就错）：
#   起 = 循环体内 `is_resumed_ibs = ...`（含）；止 = 其后第一条 While（预热循环，不含）。
REGION_START = "is_resumed_ibs"
EXCLUDE = {"check_chunk"}

# 权威名单（51 条，顺序 = 原 init 行序）。执行器按口径重算后与此逐一比对。
EXPECTED = [
    "is_resumed_ibs", "bias_warmup_diag", "resumed_frozen_f_k", "f_k_warm_started",
    "f_stability_threshold_kJ_mol", "frozen_burn_in_steps", "steps_at_full_bias",
    "bias_update_count", "bias_converged", "restored_from_production_checkpoint",
    "best_effort_acceptance", "best_effort_acceptance_reason",
    "best_effort_reburnin_required", "best_effort_reburnin_steps",
    "truncated_validation_frames_ignored", "safety_cap_best_effort_tmbar_converged",
    "budget_fallback_used", "consecutive_pass_count", "validation_pass_count",
    "last_lse_balance", "lse_residual_history", "learning_to_validation_cycles",
    "validation_feedback_update_count", "validate_direct_retry_pending",
    "ever_completed_a_validate_attempt", "last_failure_reason",
    "frozen_validation_retry_count", "last_f_delta", "freeze_burn_in_done",
    "best_effort_residual", "best_effort_f_k", "best_effort_validation_stats",
    "validation_batch_history", "validation_cumulative_history",
    "last_validation_batch_p", "validation_sample_count", "validation_batch_count",
    "validation_steps_this_freeze", "early_probe_triggered",
    "early_probe_trigger_reason", "frozen_mbar_batches",
    "validation_solve_at_batches", "validation_insufficient_attempts",
    "validation_g_history", "updates_since_freeze", "have_frozen_once",
    "local_mbar_gate_history", "last_local_mbar_gate",
    "validation_indeterminate_diag", "provisional_production", "warmup_timers",
]
assert len(EXPECTED) == 51 and len(set(EXPECTED)) == 51

# Name 形 RHS 的注解（模块常量，实测 ibs_bias_sampler.py L1603: = 5）
CONST_ANN = {"validation_solve_at_batches": "int"}


def _read(p):
    with open(p, encoding="utf-8", newline="") as fh:
        return fh.read()


def is_literal(n):
    if isinstance(n, ast.Constant):
        return True
    if isinstance(n, (ast.List, ast.Tuple, ast.Set)):
        return all(is_literal(e) for e in n.elts)
    if isinstance(n, ast.Dict):
        return all(k is None or is_literal(k) for k in n.keys) \
            and all(is_literal(v) for v in n.values)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        return is_literal(n.operand)
    return False


def is_constish(n):
    if is_literal(n):
        return True
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "float" \
            and len(n.args) == 1 and isinstance(n.args[0], ast.Constant):
        return True
    if isinstance(n, ast.Name) and n.id.replace("_", "").isupper():
        return True
    return False


def main():
    src = _read(FILE)
    tree = ast.parse(src)
    klass = next((n for n in tree.body
                  if isinstance(n, ast.ClassDef) and n.name == KLASS), None)
    m = next((n for n in klass.body if isinstance(n, ast.FunctionDef)
              and n.name == FN), None) if klass else None
    if m is None:
        sys.exit(f"FAIL: 找不到 {KLASS}.{FN}")
    loop = next((s for s in m.body if isinstance(s, ast.For)), None)
    if loop is None:
        sys.exit("FAIL: run_all_windows 顶层没有 for（结构漂移？）")
    body = loop.body

    # ① 名单重算 + 对账
    starts = [i for i, s in enumerate(body)
              if isinstance(s, (ast.Assign, ast.AnnAssign))
              and REGION_START in {getattr(t, "id", None) for t in
                                   (s.targets if isinstance(s, ast.Assign) else [s.target])}]
    if len(starts) != 1:
        sys.exit(f"FAIL: 相位2 起点锚 `{REGION_START} = ...` 命中 {len(starts)} 处（应为 1）")
    region_lo = starts[0]
    whiles = [i for i, s in enumerate(body) if i > region_lo and isinstance(s, ast.While)]
    if not whiles:
        sys.exit("FAIL: 相位2 终点锚（起点后第一条 While）不存在")
    region_hi = whiles[0]
    cands = []
    for s in body:
        if not (region_lo <= body.index(s) < region_hi):
            continue
        if isinstance(s, ast.Assign) and len(s.targets) == 1 \
                and isinstance(s.targets[0], ast.Name) and is_constish(s.value):
            cands.append((s.targets[0].id, s))
        elif isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) \
                and s.value is not None and is_constish(s.value):
            cands.append((s.target.id, s))
    cands = [(nm, s) for nm, s in cands if nm not in EXCLUDE]
    got = [nm for nm, _ in cands]
    if got != EXPECTED:
        only_got = [n for n in got if n not in EXPECTED]
        only_exp = [n for n in EXPECTED if n not in got]
        sys.exit(f"FAIL: 名单重算与权威名单不一致（多 {only_got} / 少 {only_exp} / "
                 f"顺序不同）——结构漂移，先人工核对再动刀")
    names51 = set(EXPECTED)

    # ② 安全检查（全部实测过为零/无，漂移即拒绝）
    params = {a.arg for a in m.args.args + m.args.kwonlyargs + m.args.posonlyargs}
    if m.args.vararg:
        params.add(m.args.vararg.arg)
    if m.args.kwarg:
        params.add(m.args.kwarg.arg)
    if names51 & params:
        sys.exit(f"FAIL: 名单撞形参 {sorted(names51 & params)}")
    for_i = next(i for i, s in enumerate(m.body) if s is loop)
    pre = [n for s in m.body[:for_i] for n in ast.walk(s)
           if isinstance(n, ast.Name) and n.id in names51]
    pre += [n for n in ast.walk(loop.target)
            if isinstance(n, ast.Name) and n.id in names51]
    if pre:
        sys.exit(f"FAIL: 循环前引用 {[(n.id, n.lineno) for n in pre]}")
    post = [n for s in m.body[for_i + 1:] for n in ast.walk(s)
            if isinstance(n, ast.Name) and n.id in names51]
    if post:
        sys.exit(f"FAIL: 循环后引用 {[(n.id, n.lineno) for n in post]}")
    for n in ast.walk(m):
        if isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name) \
                and n.target.id in names51:
            sys.exit(f"FAIL: walrus 目标 {n.target.id} @ L{n.lineno}（state.x := 不合法）")
        if isinstance(n, ast.Name) and n.id in names51 and isinstance(n.ctx, ast.Del):
            sys.exit(f"FAIL: del {n.id} @ L{n.lineno}")
        if isinstance(n, (ast.Global, ast.Nonlocal)) and names51 & set(n.names):
            sys.exit(f"FAIL: {type(n).__name__} @ L{n.lineno}")
        if isinstance(n, ast.Match):
            sys.exit(f"FAIL: match 语句 @ L{n.lineno}（模式绑定未纳入分析）")
    # 内层函数/lambda/推导式作用域绑定名单名 → 遮蔽，拒绝（实测为零）
    scopes = []

    def collect(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) \
                    or hasattr(child, "generators"):
                scopes.append(child)
            collect(child)
    for s in body:
        collect(s)
    shadow = []
    for n in ast.walk(m):
        if isinstance(n, ast.Name) and n.id in names51:
            for sc in scopes:
                if sc.lineno <= n.lineno <= sc.end_lineno:
                    b = {x.arg for x in ast.walk(sc) if isinstance(x, ast.arg)}
                    b |= {x.id for x in ast.walk(sc)
                          if isinstance(x, ast.Name) and isinstance(x.ctx, (ast.Store, ast.Del))}
                    b |= {x.name for x in ast.walk(sc)
                          if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
                    if n.id in b:
                        shadow.append((n.id, n.lineno, type(sc).__name__))
    if shadow:
        sys.exit(f"FAIL: 内层作用域遮蔽 {shadow[:6]}——需人工裁定后扩展执行器")
    hits_warm = [n.lineno for n in ast.walk(m)
                 if isinstance(n, ast.Name) and n.id == BIND]
    if hits_warm:
        sys.exit(f"FAIL: 承载名 {BIND!r} 已被占用 @ {hits_warm}")
    spans = [(s.lineno, s.end_lineno) for _, s in cands]
    if any(a != b for a, b in spans):
        sys.exit(f"FAIL: 存在跨行 init {spans}——本执行器按单行语句删行")
    if len({a for a, _ in spans}) != len(spans):
        sys.exit("FAIL: 多条 init 共享同一行")
    for nm, s in cands:
        first = min((n for n in ast.walk(m)
                     if isinstance(n, ast.Name) and n.id == nm),
                    key=lambda n: (n.lineno, n.col_offset))
        if (first.lineno, first.col_offset) != (s.lineno, s.col_offset):
            sys.exit(f"FAIL: {nm} 首次出现 L{first.lineno} 不是其 init L{s.lineno}")

    # ③ 数据类文本
    fields, need_any = [], False
    for nm, s in cands:
        v = s.value
        if isinstance(s, ast.AnnAssign):
            ann = ast.unparse(s.annotation)
        elif isinstance(v, ast.Constant):
            if v.value is None:
                ann, need_any = "Any", True
            else:
                ann = type(v.value).__name__
        elif isinstance(v, ast.List):
            ann = "list"
        elif isinstance(v, ast.Dict):
            ann = "dict"
        elif isinstance(v, (ast.UnaryOp,)) or (
                isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                and v.func.id == "float"):
            ann = "float"
        elif isinstance(v, ast.Name):
            if nm not in CONST_ANN:
                sys.exit(f"FAIL: Name 形 RHS {nm} 无注解表条目")
            ann = CONST_ANN[nm]
        else:
            sys.exit(f"FAIL: {nm} 的 RHS 形态未覆盖：{ast.unparse(v)[:60]}")
        if isinstance(v, ast.List) and not v.elts:
            dflt = "field(default_factory=list)"
        elif isinstance(v, ast.Dict) and not v.keys:
            dflt = "field(default_factory=dict)"
        elif isinstance(v, ast.Dict):
            dflt = f"field(default_factory=lambda: {ast.unparse(v)})"
        else:
            dflt = ast.unparse(v)
        fields.append(f"    {nm}: {ann} = {dflt}")

    imports = "from dataclasses import dataclass, field\n"
    if need_any:
        imports += "from typing import Any\n"
    header = (imports + "\n\n" + f"@dataclass\nclass {DATACLASS}:\n"
              f'    """单窗预热累加器（CUTTING_PLAN.md §5 序1、§0.2）。\n'
              f"\n"
              f"    51 条散落在 run_all_windows 相位2 的常量型初始化收拢于此：\n"
              f"    字段顺序 = 原 init 行序，默认值 = 原初值（verify_rename ③逐条对账）。\n"
              f"    可变对象：本类承载跨语句累加，不做 frozen；每窗重建一次\n"
              f"    （run_all_windows 窗口循环体内的 `{BIND} = {DATACLASS}()`）。\n"
              f'    """\n\n'
              + "\n".join(fields) + "\n\n\n")

    # ④ 改写点收集（init 行除外；f-string 内引用在 py3.12 是真实节点，一并计入）
    init_lines = {ln for a_, b_ in spans for ln in range(a_, b_ + 1)}
    joined = set()
    for js in ast.walk(m):
        if isinstance(js, ast.JoinedStr):
            for sub in ast.walk(js):
                if isinstance(sub, ast.Name):
                    joined.add(id(sub))
    pts, refs_in_fstr = {}, 0
    for n in ast.walk(m):
        if isinstance(n, ast.Name) and n.id in names51 and n.lineno not in init_lines:
            pts.setdefault(n.lineno, []).append((n.col_offset, n.id))
            if id(n) in joined:
                refs_in_fstr += 1
    total_refs = sum(len(v) for v in pts.values())

    lines = src.splitlines(keepends=True)

    # ⑤ 逐行改写：字节列 → 字符列，右→左
    for ln in sorted(pts):
        line = lines[ln - 1]
        b = line.encode("utf-8")

        def ch(bc, _b=b):
            return len(_b[:bc].decode("utf-8"))
        for bc, nm in sorted(pts[ln], key=lambda t: t[0], reverse=True):
            c = ch(bc)
            if line[c:c + len(nm)] != nm:
                sys.exit(f"FAIL: L{ln} 字符列 {c} 处不是 {nm!r}: {line!r}")
            line = line[:c] + f"{BIND}.{nm}" + line[c + len(nm):]
        lines[ln - 1] = line

    # ⑥ 删 init 行（降序）+ 原首位插入构造
    first_start = min(a_ for a_, _ in spans)
    indent = lines[first_start - 1][:len(lines[first_start - 1])
                                     - len(lines[first_start - 1].lstrip())]
    for a_, b_ in sorted(spans, reverse=True):
        del lines[a_ - 1:b_]
    lines.insert(first_start - 1, f"{indent}{BIND} = {DATACLASS}()\n")

    # ⑦ 文件头插入 + 自检 + 落盘
    # 数据类头插在目标类之前（历史 domains 文件里类在第 1 行 ⟹ 等于文件头；
    # 主线文件头有模块 docstring 与 import，插文件头会把 docstring 挤掉）。
    k_at = min([d.lineno for d in klass.decorator_list] + [klass.lineno]) - 1
    text = "".join(lines[:k_at]) + header + "".join(lines[k_at:])
    ast.parse(text)
    with open(FILE, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)

    idx = next(i for i, s in enumerate(m.body) if s is loop)
    print(f"✅ {DATACLASS}: 字段 {len(fields)} 个（Any 注解 {sum(1 for f in fields if ': Any ' in f)} 个），"
          f"init 删除 {len(spans)} 条，引用改写 {total_refs} 处（其中 f-string 内 {refs_in_fstr}），"
          f"构造插入 @ 原 L{first_start}；文件头 +{len(header.splitlines())} 行")
    print(f"   verify_rename.py work/warmup_state.before.py {FILE.name} "
          f"--fn {FN} --bind {BIND} --names-file work/warmup_state_names.txt "
          f"--dataclass {DATACLASS} --block for --index {idx} --construction 1")


if __name__ == "__main__":
    main()
