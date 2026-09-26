#!/usr/bin/env python3
"""刀2 B 段·第二波收编执行器：stage1_states（形参→字段，构造点传初值）+ opt_res。

实测（2026-09-20）：stage1_states 是 _run_dual_lambda_scheme 的**形参**
（run_full_pipeline body[32] 绑定后传入），P-A 读参数值、P-B 重绑、P-D 读
重绑值——收编后构造点传初值 `PipelineState(stage1_states=stage1_states)`，
重绑值走 dual.stage1_states 写入，参数值/重绑值两段流都保真。形参保留在
签名内（暂成 vestigial，D 线 Config 归组时统一处置）。opt_res 为 P-A 顶层
绑定的普通局部，原地收编。

变换：
  ① PipelineState 追加 2 字段（Any = None）；
  ② _run_dual_lambda_scheme 内 2 名字全部引用改写 dual.<field>
     （构造语句行**除外**——其 kwarg 值必须保持形参裸名）；
  ③ 构造语句改写为 PipelineState(stage1_states=stage1_states)。
门：执行器内置双向验证（期望树同变换，全模块逐节点比对）+
verify_rename --in-place --construction 1 + py_compile + layer_check。
"""
import ast
import copy
import pathlib
import sys

FILE = pathlib.Path(__file__).resolve().parent / "pipeline_orchestration" / "abfe_pipeline_class.py"
KLASS = "ABFEPipeline"
FN = "_run_dual_lambda_scheme"
BIND = "dual"
DATACLASS = "PipelineState"
NAMES2 = ["stage1_states", "opt_res"]
CTOR_KW = {"stage1_states": "stage1_states"}   # 构造点 kwarg: 字段名=形参名
NEW_FIELDS = ["stage1_states: Any = None", "opt_res: Any = None"]


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
    dcls = next((n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == DATACLASS), None)
    klass = next((n for n in tree.body if isinstance(n, ast.ClassDef)
                  and n.name == KLASS), None)
    m = next((n for n in klass.body if isinstance(n, ast.FunctionDef)
              and n.name == FN), None) if klass else None
    if dcls is None or m is None:
        sys.exit("FAIL: 找不到 PipelineState 或 _run_dual_lambda_scheme")
    existing = [s.target.id for s in dcls.body if isinstance(s, ast.AnnAssign)]
    for f in NAMES2:
        if f in existing:
            sys.exit(f"FAIL: 字段 {f} 已存在（重复收编？）")
    pparams = {a.arg for a in m.args.args + m.args.kwonlyargs}
    if "stage1_states" not in pparams:
        sys.exit("FAIL: stage1_states 不是方法形参——与实测不符")
    if "opt_res" in pparams:
        sys.exit("FAIL: opt_res 不应是形参")

    # 构造语句（唯一）
    ctor = [s for s in m.body if isinstance(s, ast.Assign)
            and isinstance(s.targets[0], ast.Name) and s.targets[0].id == BIND
            and isinstance(s.value, ast.Call)
            and getattr(s.value.func, "id", "") == DATACLASS]
    if len(ctor) != 1:
        sys.exit(f"FAIL: 构造语句 {len(ctor)} 处（预期 1）")
    ctor_ln = ctor[0].lineno

    # 遮蔽检查
    scopes = []

    def collect(node):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) \
                    or hasattr(ch, "generators"):
                scopes.append(ch)
            collect(ch)
    for s in m.body:
        collect(s)
    for n in ast.walk(m):
        if isinstance(n, ast.Name) and n.id in NAMES2:
            for sc in scopes:
                if sc.lineno <= n.lineno <= sc.end_lineno:
                    b = {x.arg for x in ast.walk(sc) if isinstance(x, ast.arg)}
                    b |= {x.id for x in ast.walk(sc) if isinstance(x, ast.Name)
                          and isinstance(x.ctx, (ast.Store, ast.Del))}
                    if n.id in b:
                        sys.exit(f"FAIL: 内层遮蔽 {n.id} @ L{n.lineno}")

    # 行手术：改写引用（构造语句行除外——kwarg 值必须保持形参裸名）
    pts = {}
    for n in ast.walk(m):
        if isinstance(n, ast.Name) and n.id in NAMES2 and n.lineno != ctor_ln:
            pts.setdefault(n.lineno, []).append((n.col_offset, n.id))
    total = sum(len(v) for v in pts.values())
    for ln in sorted(pts):
        line = lines[ln - 1]
        raw = line.encode("utf-8")

        def ch(bc, _b=raw):
            return len(_b[:bc].decode("utf-8"))
        for bc, nm in sorted(pts[ln], key=lambda t: t[0], reverse=True):
            c = ch(bc)
            if line[c:c + len(nm)] != nm:
                sys.exit(f"FAIL: L{ln} 字符列 {c} 处不是 {nm!r}: {line!r}")
            line = line[:c] + f"{BIND}.{nm}" + line[c + len(nm):]
        lines[ln - 1] = line

    # 构造语句加 kwarg（值 = 形参裸名）
    cl = lines[ctor_ln - 1]
    if "PipelineState()" not in cl:
        sys.exit(f"FAIL: 构造行形态异常：{cl!r}")
    lines[ctor_ln - 1] = cl.replace("PipelineState()",
                                    "PipelineState(stage1_states=stage1_states)")

    # 追加字段到 PipelineState 体尾
    last_field = max(i for i, s in enumerate(dcls.body)
                     if isinstance(s, ast.AnnAssign))
    field_lines = ["    " + f + "\n" for f in NEW_FIELDS]   # 缩进必须自带(absorb3 教训)
    lines[dcls.body[last_field].end_lineno:dcls.body[last_field].end_lineno] = field_lines

    text = "".join(lines)
    new_tree = ast.parse(text)

    # 双向验证
    expect = copy.deepcopy(tree)
    exp_dcls = next(n for n in expect.body if isinstance(n, ast.ClassDef)
                    and n.name == DATACLASS)
    exp_klass = next(n for n in expect.body if isinstance(n, ast.ClassDef)
                     and n.name == KLASS)
    exp_m = next(n for n in exp_klass.body if isinstance(n, ast.FunctionDef)
                 and n.name == FN)

    class _Bind(ast.NodeTransformer):
        def visit_Name(self, node):
            self.generic_visit(node)
            if node.id in NAMES2 and node.lineno != ctor_ln:
                return ast.copy_location(
                    ast.Attribute(value=ast.Name(id=BIND, ctx=ast.Load()),
                                  attr=node.id, ctx=node.ctx), node)
            return node
    exp_m = _Bind().visit(exp_m)
    # 构造语句重建（kwarg 值保持形参裸名）
    for s in exp_m.body:
        if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name) \
                and s.targets[0].id == BIND:
            s.value.keywords = [ast.keyword(arg="stage1_states",
                                            value=ast.Name(id="stage1_states",
                                                           ctx=ast.Load()))]
            break
    for f in NEW_FIELDS:
        stmt = ast.parse("class _T:\n    " + f + "\n").body[0].body[0]
        exp_dcls.body.append(stmt)

    if _strip_dump(new_tree) != _strip_dump(expect):
        sys.exit("FAIL: 行手术结果与期望树全模块比对不一致——拒绝写盘")

    with open(FILE, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    per = {nm: sum(1 for ln in pts for _, x in pts[ln] if x == nm) for nm in NAMES2}
    print(f"✅ 第二波收编: 改写 {per}，字段追加 2 个（总 28）；构造 kwarg 传初值；"
          f"全模块逐节点比对通过")


if __name__ == "__main__":
    main()
