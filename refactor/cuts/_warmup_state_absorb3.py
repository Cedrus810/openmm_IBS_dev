#!/usr/bin/env python3
"""刀1 B 段·第1步执行器：WarmupState 收编 3 个跨界标量。

frozen_f_k_snapshot / mode / ema_mean_p_values 是当年 51 名单的排除项
（首写点在 resume 决策 If 的分支内，非无条件顶层）。实测（2026-09-20）：
  * 三个名字首次文本出现均为 Store，且每条路径先写后读（if/else 两路都赋值：
    L2043-2098 两路覆盖 mode/frozen，L2993-3011 两路覆盖 ema）——
    **字段默认值在所有路径上不可观测**，收编语义安全；
  * run_all_windows 内 41 处引用（14+18+9），零内层遮蔽；
  * _commit_frozen_f_k 以形参读 mode/ema → 本刀零改动，调用点 kwarg 值
    改为 warm.<field>（kwarg 名是字符串节点，改写天然正确）。

变换：
  ① WarmupState 末尾追加 3 个字段（Any = None / str = "learning" / Any = None，
     默认值不可观测，取自然初值）；
  ② run_all_windows 子树内 3 名字全部引用改写为 warm.<field>（作用域感知）。
门：执行器内置双向验证（期望树 = 原树深拷贝做同一 Name→Attribute 变换 + 解析法
插入字段，与行手术结果 parse 后的全模块逐节点比对，过才写盘）
+ verify_rename --keep（既有 51 字段访问不还原也不算 stray）
+ py_compile + layer_check 不变量。
"""
import ast
import copy
import os
import pathlib
import sys

# 重放（replay.py）用 CUT_FILE 指向目标文件；缺省＝历史 domains 路径。
FILE = pathlib.Path(os.environ["CUT_FILE"]) if os.environ.get("CUT_FILE") else pathlib.Path(__file__).resolve().parent / "windows_sampling" / "window_manager_dual_lambda.py"
KLASS = "IBSWindowManagerDualLambda"
FN = "run_all_windows"
BIND = "warm"
DATACLASS = "WarmupState"
NAMES3 = ["frozen_f_k_snapshot", "mode", "ema_mean_p_values"]
FIELDS = [
    "frozen_f_k_snapshot: Any = None",
    'mode: str = "learning"',
    "ema_mean_p_values: Any = None",
]
COMMENT = "    # —— 刀1 B段第1步收编（2026-09-20）：3 个跨界标量，解锁相位3/4边界 ——"


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
        sys.exit("FAIL: 找不到 WarmupState 或 run_all_windows")
    existing = [s.target.id for s in dcls.body if isinstance(s, ast.AnnAssign)]
    for f in NAMES3:
        if f in existing:
            sys.exit(f"FAIL: 字段 {f} 已存在（重复收编？）")
    if len(existing) != 51:
        sys.exit(f"FAIL: WarmupState 现有字段 {len(existing)} ≠ 51——结构漂移")

    # ① 作用域遮蔽检查
    scopes = []

    def collect(node):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) \
                    or hasattr(ch, "generators"):
                scopes.append(ch)
            collect(ch)
    for s in m.body:
        collect(s)
    shadow = []
    for n in ast.walk(m):
        if isinstance(n, ast.Name) and n.id in NAMES3:
            for sc in scopes:
                if sc.lineno <= n.lineno <= sc.end_lineno:
                    b = {x.arg for x in ast.walk(sc) if isinstance(x, ast.arg)}
                    b |= {x.id for x in ast.walk(sc) if isinstance(x, ast.Name)
                          and isinstance(x.ctx, (ast.Store, ast.Del))}
                    if n.id in b:
                        shadow.append((n.id, n.lineno, type(sc).__name__))
    if shadow:
        sys.exit(f"FAIL: 内层遮蔽 {shadow[:6]}——需人工裁定")

    # ② 行手术：逐行、字节列→字符列、右→左
    pts = {}
    for n in ast.walk(m):
        if isinstance(n, ast.Name) and n.id in NAMES3:
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

    # ③ 数据类追加字段（warmup_timers 字段行之后）
    wt = next((s for s in dcls.body if isinstance(s, ast.AnnAssign)
               and isinstance(s.target, ast.Name) and s.target.id == "warmup_timers"), None)
    if wt is None:
        sys.exit("FAIL: WarmupState 里没有 warmup_timers 字段（结构漂移）")
    field_block = [COMMENT + "\r\n"] + ["    " + f + "\r\n" for f in FIELDS]
    lines[wt.end_lineno:wt.end_lineno] = field_block

    text = "".join(lines)
    new_tree = ast.parse(text)

    # ④ 双向验证：期望树 = 原树深拷贝 + Name→warm.attr 变换（仅 run_all_windows）
    #    + 解析法插入 3 字段；与实际 after 全模块逐节点比对。
    expect = copy.deepcopy(tree)
    exp_klass = next(n for n in expect.body
                     if isinstance(n, ast.ClassDef) and n.name == KLASS)
    exp_m = next(n for n in exp_klass.body
                 if isinstance(n, ast.FunctionDef) and n.name == FN)

    class _Bind(ast.NodeTransformer):
        def visit_Name(self, node):
            self.generic_visit(node)
            if node.id in NAMES3:
                return ast.copy_location(
                    ast.Attribute(value=ast.Name(id=BIND, ctx=ast.Load()),
                                  attr=node.id, ctx=node.ctx), node)
            return node
    exp_m = _Bind().visit(exp_m)

    exp_dcls = next(n for n in expect.body
                    if isinstance(n, ast.ClassDef) and n.name == DATACLASS)
    field_src = "class _W:\n    " + "\n    ".join(FIELDS) + "\n"
    new_fields = [s for s in ast.parse(field_src).body[0].body
                  if isinstance(s, ast.AnnAssign)]
    wt_i = next(i for i, s in enumerate(exp_dcls.body)
                if isinstance(s, ast.AnnAssign) and s.target.id == "warmup_timers")
    exp_dcls.body[wt_i + 1:wt_i + 1] = new_fields

    if _strip_dump(new_tree) != _strip_dump(expect):
        sys.exit("FAIL: 行手术结果与期望树全模块比对不一致——拒绝写盘")

    with open(FILE, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    per = {n: sum(1 for _, x in (p for ln in pts.values() for p in ln) if x == n)
           for n in NAMES3}
    print(f"✅ 收编 3 标量: 引用改写 {total} 处 {per}，字段追加 3 个（总 54）；"
          f"全模块逐节点比对通过")
    print(f"   verify_rename.py work/absorb3.before.py {FILE.name} --fn {FN} "
          f"--bind {BIND} --names {','.join(NAMES3)} --keep <51名单> --dataclass {DATACLASS}")


if __name__ == "__main__":
    main()
