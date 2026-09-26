#!/usr/bin/env python3
"""刀1 链复合门：端到端证明「当前文件 ≡ 原始文件 + 且仅 + 刀1 链的 8 个变换」。

链（window_manager_dual_lambda.py，留底全在 work/）：
  WarmupState收拢 → 刀1A(_produce_window) → 刀a(_commit_frozen_f_k) →
  B①3标量收编 → 刀b(_run_warmup_loop) → 刀c(_summarize_warmup) →
  B②ProductionState收编 → 刀d(_finish_production) →
  D①EarlyStopConfig归组（es=es 展开 + es.x 还原，机械精确）→
  D②WindowContext归组（wc=wc 调用随 inline 消费；wc.x 还原 + 循环顶
  wc 构造删除，机械精确）

方法：从当前文件把 run_all_windows 循环体**重建**回原始形态——
  ① 内联 5 个抽出方法体到各自调用点（_finish_production 先内联回
     _produce_window，再整体内联回循环体）；
  ② `prod = ProductionState()` 删除、prod.x 还原为 x（原地收编，精确）；
  ③ `warm = WarmupState()` **替换**为 51 条原始 init（由数据类字段默认值
     经 ③对账的逆映射重建：default_factory=list/dict/λ → 字面量），
     warm.x 还原为 x（54 名）；
然后与最早留底（warmup_state.before.py = 刀1 链前状态）的原始循环体比对。

**唯一公理**（非精确还原、已单独证过、此处复用）：WarmupState 的 51 条 init
从散布原位**提升**到构造点——语义无损性由收拢刀当时的实测背书（51/51 首写
=init、纯字面量无相互依赖、先读后写为零）。故比对口径 = 双方剥离 51 条 init 后：
  (a) init 集合对账：名（含顺序）、值 AST、注解 AST 逐一相同；
  (b) 剥离余部逐节点相同。
其余一切（内联、prod 原地还原、warm 原地还原、调用点 kwarg）必须**精确**。

只读，不写任何文件。PASS = 刀1 链 8 刀被端到端封死。
"""
import ast
import copy
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent   # refactor/
KLASS = "IBSWindowManagerDualLambda"
FN = "run_all_windows"
BEFORE = ROOT / "work" / "warmup_state.before.py"
NAMES51 = (ROOT / "work" / "warmup_state_names.txt").read_text(encoding="utf-8").split()
AFTER = ROOT / "domains" / "windows_sampling" / "window_manager_dual_lambda.py"
BINDDATA = {"warm": "WarmupState", "prod": "ProductionState"}
FNS = ["_run_warmup_loop", "_summarize_warmup", "_commit_frozen_f_k",
       "_produce_window", "_finish_production"]
NAMES54 = set(NAMES51) | {"frozen_f_k_snapshot", "mode", "ema_mean_p_values"}


def _read(p):
    with open(p, encoding="utf-8", newline="") as fh:
        return fh.read()


def _clean_dump(node):
    for n in ast.walk(node):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset",
                     "type_comment"):
            if hasattr(n, attr):
                try:
                    delattr(n, attr)
                except AttributeError:
                    setattr(n, attr, None)
    return ast.dump(node)


def _is_self_call(s, fn_name):
    return (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
            and isinstance(s.value.func, ast.Attribute)
            and s.value.func.attr == fn_name
            and isinstance(s.value.func.value, ast.Name)
            and s.value.func.value.id == "self")


def _body_no_doc(fn_node):
    b = list(fn_node.body)
    if (b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant)
            and isinstance(b[0].value.value, str)):
        b = b[1:]
    return b


def _inline(stmts, fn_name, body):
    out, hits = [], 0
    for s in stmts:
        if _is_self_call(s, fn_name):
            out.extend(copy.deepcopy(body))
            hits += 1
        else:
            out.append(s)
    if hits != 1:
        sys.exit(f"FAIL: {fn_name} 调用点 {hits} 处（预期 1）——重建失败")
    return out


def _binds_all(stmts):
    b = set()
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                b.add(n.id)
            elif isinstance(n, ast.arg):
                b.add(n.arg)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                b.add(n.name)
    return b


def _strip_inits(stmts, names):
    """剥离每个名字的**首个**顶层单名赋值（Assign/AnnAssign），返回 (余部, {名: (值dump, 注解dump|None)}, 顺序)。"""
    rest, imap, order = [], {}, []
    for s in stmts:
        nm, val, ann = None, None, None
        if isinstance(s, ast.Assign) and len(s.targets) == 1 \
                and isinstance(s.targets[0], ast.Name):
            nm, val = s.targets[0].id, s.value
        elif isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) \
                and s.value is not None:
            nm, val, ann = s.target.id, s.value, s.annotation
        if nm in names and nm not in imap and val is not None:
            imap[nm] = (_clean_dump(val), _clean_dump(ann) if ann is not None else None)
            order.append(nm)
        else:
            rest.append(s)
    return rest, imap, order


def main():
    tb = ast.parse(_read(BEFORE))
    ta = ast.parse(_read(AFTER))

    kb = next(n for n in tb.body if isinstance(n, ast.ClassDef) and n.name == KLASS)
    mb = next(n for n in kb.body if isinstance(n, ast.FunctionDef) and n.name == FN)
    before_body = next(s for s in mb.body if isinstance(s, ast.For)).body

    ka = next(n for n in ta.body if isinstance(n, ast.ClassDef) and n.name == KLASS)
    ma = next(n for n in ka.body if isinstance(n, ast.FunctionDef) and n.name == FN)
    after_body = next(s for s in ma.body if isinstance(s, ast.For)).body
    fns = {}
    for nm in FNS:
        cand = [n for n in ka.body if isinstance(n, ast.FunctionDef) and n.name == nm]
        if len(cand) != 1:
            sys.exit(f"FAIL: 抽出方法 {nm} 有 {len(cand)} 个（预期 1）")
        fns[nm] = cand[0]

    # ① 内联（自内向外：_finish_production 先内联回 _produce_window 体，
    #    再把重建后的 _produce_window 体连同其余三个方法体逐一内联回循环体）
    # D① 逆变换（inline 之前作用于三个体：调用语句会被 inline 消费，
    # 之后做会扑空）：es=es kwarg 展开 + es.x 还原
    ES9 = ["early_stop_check_interval_steps", "early_stop_max_delta_g_drift_kJ_mol",
           "early_stop_max_uncertainty_kJ_mol", "early_stop_min_absolute_ess",
           "early_stop_min_decorrelated_samples", "early_stop_min_ess_ratio",
           "early_stop_min_steps", "early_stop_required_consecutive_passes",
           "enable_early_stop"]
    n_exp = [0]

    class _ExpandEs(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            for i, kw in enumerate(node.keywords):
                if (kw.arg == "es" and isinstance(kw.value, ast.Name)
                        and kw.value.id == "es"):
                    exp = [ast.keyword(arg=n2, value=ast.Name(id=n2, ctx=ast.Load()))
                           for n2 in ES9]
                    node.keywords = node.keywords[:i] + exp + node.keywords[i + 1:]
                    n_exp[0] += 1
                    break
            return node

    class _UnbindEs(ast.NodeTransformer):
        def visit_Attribute(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == "es" \
                    and node.attr in ES9:
                return ast.copy_location(
                    ast.Name(id=node.attr, ctx=node.ctx), node)
            return node

    def _undo_es(stmts):
        out = [_ExpandEs().visit(_UnbindEs().visit(st)) for st in stmts]
        for st in out:
            ast.fix_missing_locations(st)
        return out

    WC6 = ["lc_win", "lv_win", "sampler", "sim", "win_sys", "win_sys_xml"]

    class _UnbindWc(ast.NodeTransformer):
        def visit_Attribute(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == "wc" \
                    and node.attr in WC6:
                return ast.copy_location(
                    ast.Name(id=node.attr, ctx=node.ctx), node)
            return node

    def _undo_wc(stmts):
        out = [_UnbindWc().visit(st) for st in stmts]
        for st in out:
            ast.fix_missing_locations(st)
        return out

    fns_u = {nm: _undo_wc(_undo_es(_body_no_doc(fns[nm]))) for nm in
             ("_finish_production", "_produce_window", "_run_warmup_loop",
              "_summarize_warmup", "_commit_frozen_f_k")}
    after_body_u = _undo_wc(_undo_es(after_body))
    # wc 构造语句在循环体顶（body[0]）——删除
    if not (after_body_u and isinstance(after_body_u[0], ast.Assign)
            and isinstance(after_body_u[0].targets[0], ast.Name)
            and after_body_u[0].targets[0].id == "wc"):
        sys.exit("FAIL: 循环体顶不是 wc 构造——重建失败")
    after_body_u = after_body_u[1:]

    pw_body = _inline(fns_u["_produce_window"], "_finish_production",
                      fns_u["_finish_production"])
    recon = _inline(after_body_u, "_produce_window", pw_body)
    recon = _inline(recon, "_run_warmup_loop", fns_u["_run_warmup_loop"])
    recon = _inline(recon, "_summarize_warmup", fns_u["_summarize_warmup"])
    recon = _inline(recon, "_commit_frozen_f_k", fns_u["_commit_frozen_f_k"])
    if n_exp[0] != 2:
        sys.exit(f"FAIL: es=es 调用点展开 {n_exp[0]} 处（预期 2）——重建失败")

    # ② 构造语句处置
    warm_i = [i for i, s in enumerate(recon)
              if isinstance(s, ast.Assign) and len(s.targets) == 1
              and isinstance(s.targets[0], ast.Name) and s.targets[0].id == "warm"
              and isinstance(s.value, ast.Call)
              and getattr(s.value.func, "id", "") == "WarmupState"]
    if len(warm_i) != 1:
        sys.exit(f"FAIL: warm 构造语句 {len(warm_i)} 处（预期 1）")
    prod_i = [i for i, s in enumerate(recon)
              if isinstance(s, ast.Assign) and len(s.targets) == 1
              and isinstance(s.targets[0], ast.Name) and s.targets[0].id == "prod"
              and isinstance(s.value, ast.Call)
              and getattr(s.value.func, "id", "") == "ProductionState"]
    if len(prod_i) != 1:
        sys.exit(f"FAIL: prod 构造语句 {len(prod_i)} 处（预期 1）")

    dcls = next(n for n in ta.body if isinstance(n, ast.ClassDef)
                and n.name == "WarmupState")
    fields, fann = {}, {}
    for s in dcls.body:
        if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
            fields[s.target.id] = s.value
            fann[s.target.id] = s.annotation
    if sorted(fields) != sorted(NAMES54):
        sys.exit("FAIL: WarmupState 字段集与 54 名单不符")

    def invert_default(v):
        """verify_rename ③ 的逆映射：字段默认 → 原 init 字面量。"""
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "field":
            kw = {k.arg: k.value for k in v.keywords}
            fac = kw.get("default_factory")
            if isinstance(fac, ast.Lambda):
                return fac.body
            if isinstance(fac, ast.Name):
                return {"list": ast.List(elts=[], ctx=ast.Load()),
                        "dict": ast.Dict(keys=[], values=[]),
                        "set": ast.Call(func=ast.Name(id="set", ctx=ast.Load()),
                                        args=[], keywords=[])}[fac.id]
            if "default" in kw:
                return kw["default"]
            sys.exit(f"FAIL: field 默认值形态未覆盖: {ast.unparse(v)}")
        return v

    warm_inits = []
    for nm in NAMES51:
        v = fields[nm]
        tgt = ast.Name(id=nm, ctx=ast.Store())
        warm_inits.append(ast.Assign(targets=[tgt], value=invert_default(v)))

    del recon[prod_i[0]]          # 先删高位（prod 构造在相位6，晚于 warm 构造）
    recon[warm_i[0]:warm_i[0] + 1] = warm_inits

    # ③ warm./prod. 还原（先确认无内层遮蔽）
    scopes = []

    def collect(node):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) \
                    or hasattr(ch, "generators"):
                scopes.append(ch)
            collect(ch)
    for s in recon:
        collect(s)
    sb = []
    for sc in scopes:
        b = {x.arg for x in ast.walk(sc) if isinstance(x, ast.arg)}
        b |= {x.id for x in ast.walk(sc) if isinstance(x, ast.Name)
              and isinstance(x.ctx, (ast.Store, ast.Del))}
        b |= {x.name for x in ast.walk(sc) if isinstance(x, ast.ExceptHandler) and x.name}
        sb.append((sc, b))
    hits, shadow = 0, set()

    class _Unbind(ast.NodeTransformer):
        def visit_Attribute(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id in BINDDATA:
                attr = node.attr
                owner_fields = (NAMES54 if node.value.id == "warm"
                                else set(NAMES51) | set())
                if node.value.id == "prod":
                    # prod 字段 = 26 名（ ProductionState）
                    pcls = next(n for n in ta.body if isinstance(n, ast.ClassDef)
                                and n.name == "ProductionState")
                    pfields = {s.target.id for s in pcls.body
                               if isinstance(s, ast.AnnAssign)}
                    if attr not in pfields:
                        shadow.add(f"prod.{attr}")
                        return node
                elif attr not in NAMES54:
                    shadow.add(f"warm.{attr}")
                    return node
                return ast.copy_location(
                    ast.Name(id=attr, ctx=node.ctx), node)
            return node
    recon = [_Unbind().visit(s) for s in recon]
    for s in recon:
        ast.fix_missing_locations(s)
    for s in recon:
        for n in ast.walk(s):
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) \
                    and n.simple == 0:
                n.simple = 1   # 目标从 prod.x/warm.x 还原为 Name,按解析器口径 0→1
    # 遮蔽核查：还原后的名字不得被内层作用域绑定（否则还原改变语义）
    for s in recon:
        for n in ast.walk(s):
            if isinstance(n, ast.Name) and n.id in NAMES54:
                for sc, b in sb:
                    if sc.lineno <= n.lineno <= sc.end_lineno and n.id in b:
                        shadow.add(f"内层遮蔽 {n.id} @L{n.lineno}")
    if shadow:
        sys.exit(f"FAIL: 还原异常 {sorted(shadow)[:8]}")

    # ④ 双方剥 51 init 后比对
    before_rest, before_map, before_order = _strip_inits(before_body, set(NAMES51))
    recon_rest, recon_map, recon_order = _strip_inits(recon, set(NAMES51))

    ok = True
    if before_order != recon_order:
        print(f"FAIL(a1): init 顺序不同\n  原: {before_order}\n  重建: {recon_order}")
        ok = False
    # 注解保留性：原 AnnAssign init 的注解必须原样活在数据类字段注解上
    ann_bad = [nm for nm in before_order
               if before_map[nm][1] is not None
               and (nm not in fann or fann[nm] is None
                    or _clean_dump(fann[nm]) != before_map[nm][1])]
    if ann_bad:
        print(f"FAIL(a0): {len(ann_bad)} 个原 AnnAssign 注解未保留到字段：{ann_bad[:4]}")
        ok = False
    mism = [nm for nm in before_order
            if nm in recon_map and recon_map[nm][0] != before_map[nm][0]]
    if set(before_map) != set(recon_map):
        print(f"FAIL(a2): init 名集合不同 多{sorted(set(recon_map)-set(before_map))} "
              f"少{sorted(set(before_map)-set(recon_map))}")
        ok = False
    elif mism:
        print(f"FAIL(a3): {len(mism)} 个 init 值/注解不同：")
        for nm in mism[:6]:
            print(f"    {nm}: 原 {before_map[nm]}  重建 {recon_map[nm]}")
        ok = False
    if len(before_rest) != len(recon_rest):
        print(f"FAIL(b1): 剥离余部语句数 {len(before_rest)} vs {len(recon_rest)}")
        ok = False
    else:
        for i, (x, y) in enumerate(zip(before_rest, recon_rest)):
            if _clean_dump(x) != _clean_dump(y):
                print(f"FAIL(b2): 余部第 {i} 条起不同：")
                print(f"    原  : {_clean_dump(x)[:200]}")
                print(f"    重建: {_clean_dump(y)[:200]}")
                ok = False
                break

    print(f"\n重建循环体 {len(recon)} 条（原始 {len(before_body)} 条）；"
          f"init 对账 {len(before_order)} 名；剥离余部 {len(before_rest)} 条")
    if ok:
        print("PASS: 刀1 链复合门 —— 当前文件 ≡ 原始 + 10 变换"
              "（唯一公理：51 init 字面量提升，已单独证过）")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
