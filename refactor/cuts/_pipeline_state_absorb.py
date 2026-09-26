#!/usr/bin/env python3
"""刀1 B 段·第 2 步执行器：ProductionState 收编 26 个生产段跨界名字。

范围实测（2026-09-20，作用域感知口径）：_produce_window 相位 6/7 边界写逃逸
29 名 − 3 个已裁决段内自覆盖临时名（e=handler 局部、gid/e_g=debug 循环局部，
7 侧 Load 全部在自己段内 Store 之后，生产流不跨段）= 26 名。
26/26 的首写都是 _produce_window 体顶层无条件赋值 → 字段默认 None
**不可观测**，原地收编语义安全（不同于相位 7 内部嵌套首写的名字——那批留给
7/8 边界，需先裁决 n_updates==0 的 crash 路径语义）。

变换：
  ① 模块级追加 @dataclass ProductionState（26 字段，全 Any = None，原地口径）；
  ② _produce_window 子树内 26 名字全部引用改写为 prod.<field>（134 处）；
  ③ 首写点前插入一行 `prod = ProductionState()`。
门：执行器内置双向验证（期望树=原树深拷贝做同一变换，与行手术结果全模块
逐节点比对）+ verify_rename --in-place（三情形自检已过）+ py_compile +
layer_check 不变量。新行一律 CRLF。
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
CRLF = "\r\n"
NAMES = [
    "_partition_criterion", "_path_record", "_stage1_preopt_key",
    "_stage2_preopt_key", "expected_vanishing_ranges", "optimized_lambdas_1",
    "optimized_lambdas_2", "preopt1_file", "preopt2_file",
    "should_run_stage1", "should_run_stage2", "stage1", "stage1_file",
    "stage1_key", "stage1_status", "stage2", "stage2_file", "stage2_key",
    "stage2_pilot_lambdas", "stage2_pilot_mean_dU_dlambda",
    "stage2_pilot_n_dU_dlambda_samples", "stage2_pilot_std_dU_dlambda",
    "stage2_states", "stage2_status", "window_ranges_1", "window_ranges_2",
]
assert len(NAMES) == 26


DCLASS_SRC = '''@dataclass
class PipelineState:
    """dual_lambda 方案跨相位状态（刀2 B 段收编，2026-09-20）。

    26 个跨 P-A~P-E 五相位的名字（键/文件/预算/stage 结果/pilot 统计），
    原地收编：初始化语句保留原位（dual.<field> = ...），每调用经
    `dual = PipelineState()` 重建。字段默认 None 不可观测：21 个首写为
    体顶层无条件直赋；expected_vanishing_ranges 条件首写后由 L13930 顶层
    无条件重写覆盖；stage1/stage2 条件首写由 should_run_stage1/2 ≡ True
    （无条件）保证 P-D 必然绑定——26/26 先写后读，全路径证明见头注释。
    """

'''
FIELD_LINES = [f"    {n}: Any = None" for n in NAMES]


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
    if any(isinstance(n, ast.ClassDef) and n.name == DATACLASS for n in tree.body):
        sys.exit("FAIL: ProductionState 已存在（重复收编？）")
    klass = next((n for n in tree.body if isinstance(n, ast.ClassDef)
                  and n.name == KLASS), None)
    m = next((n for n in klass.body if isinstance(n, ast.FunctionDef)
              and n.name == FN), None) if klass else None
    if m is None:
        sys.exit(f"FAIL: 找不到 {KLASS}.{FN}")
    if any(isinstance(n, ast.Name) and n.id == BIND for n in ast.walk(m)):
        sys.exit(f"FAIL: 承载名 {BIND!r} 已被占用")

    body = m.body

    # ① 首写核验：26/26 体顶层无条件直赋，且记录构造插入位置（最早首写语句之前）
    PROVEN_COND = {"stage1", "stage2", "expected_vanishing_ranges"}
    first_store_ln = {}
    for nm in NAMES:
        ocs = sorted((n.lineno, n.col_offset, type(n.ctx).__name__)
                     for n in ast.walk(m) if isinstance(n, ast.Name) and n.id == nm)
        fs = next((ln for ln, _, c in ocs if c == "Store"), None)
        if fs is None:
            sys.exit(f"FAIL: {nm} 无 Store")
        owner = next((s for s in body if s.lineno <= fs <= s.end_lineno), None)
        toplevel = owner and owner.lineno == fs \
            and isinstance(owner, (ast.Assign, ast.AnnAssign))
        if not toplevel and nm not in PROVEN_COND:
            sys.exit(f"FAIL: {nm} 首写不是体顶层直赋（L{fs}）——结构漂移")
        first_store_ln[nm] = fs
    ctor_at = min(first_store_ln.values())
    ctor_owner = next(s for s in body if s.lineno == ctor_at)
    ctor_indent = lines[ctor_at - 1][:len(lines[ctor_at - 1])
                                     - len(lines[ctor_at - 1].lstrip())]

    # ② 遮蔽检查
    scopes = []

    def collect(node):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) \
                    or hasattr(ch, "generators"):
                scopes.append(ch)
            collect(ch)
    for s in body:
        collect(s)
    sb = []
    for sc in scopes:
        b = {x.arg for x in ast.walk(sc) if isinstance(x, ast.arg)}
        b |= {x.id for x in ast.walk(sc) if isinstance(x, ast.Name)
              and isinstance(x.ctx, (ast.Store, ast.Del))}
        if BIND in b:
            sys.exit(f"FAIL: 内层作用域绑定承载名 {BIND} @ L{sc.lineno}")

    # ③ 行手术：引用改写（右→左，字节列→字符列）
    pts = {}
    for n in ast.walk(m):
        if isinstance(n, ast.Name) and n.id in NAMES:
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

    # ④ 行插入（降序：后位先插，前位索引不受影响）
    #    —— 构造行在 ctor_at-1（文件后部），类块在 WarmupState 尾（文件前部）
    # abfe_pipeline_class.py 模块体只有 ABFEPipeline 一个类 → PipelineState 插文件头
    dcls_text = DCLASS_SRC + "\n".join(FIELD_LINES) + "\n"
    dcls_lines = [ln + CRLF for ln in dcls_text.rstrip("\n").split("\n")] + [CRLF]
    ctor_line = f"{ctor_indent}{BIND} = {DATACLASS}(){CRLF}"
    lines[ctor_at - 1:ctor_at - 1] = [ctor_line]
    lines[0:0] = dcls_lines

    text = "".join(lines)
    new_tree = ast.parse(text)

    # ⑤ 双向验证
    expect = copy.deepcopy(tree)
    exp_klass = next(n for n in expect.body
                     if isinstance(n, ast.ClassDef) and n.name == KLASS)
    exp_m = next(n for n in exp_klass.body
                 if isinstance(n, ast.FunctionDef) and n.name == FN)

    class _Bind(ast.NodeTransformer):
        def visit_Name(self, node):
            self.generic_visit(node)
            if node.id in NAMES:
                return ast.copy_location(
                    ast.Attribute(value=ast.Name(id=BIND, ctx=ast.Load()),
                                  attr=node.id, ctx=node.ctx), node)
            return node
    exp_m = _Bind().visit(exp_m)
    # 目标从 Name 变 Attribute 后，AnnAssign.simple 需按解析器口径重算（1→0）
    for n in ast.walk(exp_m):
        if isinstance(n, ast.AnnAssign) and not isinstance(n.target, ast.Name):
            n.simple = 0

    exp_dcls = ast.parse(dcls_text).body[0]
    expect.body[0:0] = [exp_dcls]
    ctor_stmt = ast.parse(ctor_line.strip()).body[0]
    exp_m_body_i = next(i for i, s in enumerate(exp_m.body) if s.lineno == ctor_at)
    exp_m.body[exp_m_body_i:exp_m_body_i] = [ctor_stmt]

    if _strip_dump(new_tree) != _strip_dump(expect):
        sys.exit("FAIL: 行手术结果与期望树全模块比对不一致——拒绝写盘")

    with open(FILE, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    with open(pathlib.Path(__file__).resolve().parent / "scratchpad" / "pipeline_state_names.txt",
              "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(NAMES) + "\n")
    print(f"✅ {DATACLASS}: 字段 {len(NAMES)} 个，引用改写 {total} 处，"
          f"构造插入 @ 原 L{ctor_at}（{ctor_owner.lineno} 行语句前）")


if __name__ == "__main__":
    main()
