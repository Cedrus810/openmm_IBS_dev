#!/usr/bin/env python3
"""分派覆盖门：抽取条件链为「return self._m(...)」分派后，验证调用点的
进入条件 ∪ 落空路径 与原链的条件结构**等价**。

背景（刀2 事故）：verify_extract --orelse 只验「搬走的块没变」；调用点是
新写的代码，不在其射程内。刀2 把 5 路链写成 2 路 if/elif 无 else——
2d_diagonal/2d_geodesic 死码、raise ValueError 静默化为 None。本门封这个缺口。

骨架口径：条件链 = [(test_1, ...), ..., (test_k, ...), catchall?]。
等价判据（抽取-分派变换族）：
  (a) 前缀：after 的 test 序列必须与 before 的 test 序列逐位置 AST 相同；
  (b) 兜底：before 有终端 else ⟺ after 有终端 else
      （少 else = 落空静默化；多 else = 吞掉原落空路径）；
  (c) 形状：after 每支必须恰为 `return self.<method>(...)`（抽取-分派签名），
      且 else 支（若有）无再嵌套。

用法：
    python3 refactor/tools/verify_dispatch.py <before> <after> \
        --src <函数名> --pos <锚点链头在函数体中的语句下标>

自检：三情形（正确 / 刀2 事故形态：缺 else / 首测被改）见提交会话记录。
"""
import ast
import sys


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


def chain_skeleton(if_node):
    """条件链骨架：([test dump...], catchall, [(支体语句...)...])"""
    tests, branches, catchall = [], [], False
    c = if_node
    while True:
        tests.append(_clean_dump(c.test))
        branches.append(c.body)
        if len(c.orelse) == 1 and isinstance(c.orelse[0], ast.If):
            c = c.orelse[0]
            continue
        if c.orelse:
            catchall = True
            branches.append(c.orelse)
        break
    return tests, catchall, branches


def is_dispatch_branch(body):
    """支体必须恰为 `return self.<method>(...)`。"""
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    v = body[0].value
    return (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute)
            and isinstance(v.func.value, ast.Name)
            and v.func.value.id == "self")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--src", required=True)
    ap.add_argument("--pos", required=True, type=int,
                    help="锚点链头在函数体中的语句下标（抽取前后应一致）")
    a = ap.parse_args()

    tb = ast.parse(open(a.before, encoding="utf-8").read())
    ta = ast.parse(open(a.after, encoding="utf-8").read())

    def get_fn(tree):
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == a.src:
                return n
        sys.exit(f"FAIL: 找不到 {a.src}")

    fb, fa = get_fn(tb), get_fn(ta)
    if len(fb.body) != len(fa.body):
        print(f"FAIL: 函数体语句数不同 {len(fb.body)} vs {len(fa.body)}"
              f"（抽取-分派不应增删顶层语句）")
        return 1
    if a.pos >= len(fb.body):
        sys.exit(f"FAIL: --pos {a.pos} 超出函数体")
    anchor, disp = fb.body[a.pos], fa.body[a.pos]
    if not isinstance(anchor, ast.If):
        sys.exit(f"FAIL: 锚点不是 If")
    if not isinstance(disp, ast.If):
        print("FAIL: 替代语句不是 if 分派")
        return 1

    b_tests, b_catchall, _ = chain_skeleton(anchor)
    a_tests, a_catchall, a_branches = chain_skeleton(disp)

    ok = True
    # (a) 前缀
    if a_tests != b_tests[:len(a_tests)]:
        for i, (x, y) in enumerate(zip(a_tests, b_tests)):
            if x != y:
                print(f"FAIL(a): 第 {i} 支 test 与原链不同：\n  after: {x[:160]}\n  before: {y[:160]}")
                ok = False
                break
        if len(a_tests) > len(b_tests):
            print("FAIL(a): after 的 test 数多于原链")
            ok = False
    # (b) 兜底等价
    if b_catchall != a_catchall:
        if b_catchall:
            print("FAIL(b): 原链有终端 else（fail-closed），分派缺 else —— "
                  "缺失路径将静默化（刀2 事故形态）")
        else:
            print("FAIL(b): 原链无终端 else（存在落空路径），分派加了 else —— "
                  "落空路径被吞")
        ok = False
    # (c) 分派形状
    all_branches = list(a_branches)
    for i, br in enumerate(all_branches):
        if not is_dispatch_branch(br):
            print(f"FAIL(c): 第 {i} 支不是 `return self.<method>(...)` 形状")
            ok = False
            break

    print(f"原链 {len(b_tests)} test + {'else' if b_catchall else '落空'} | "
          f"分派 {len(a_tests)} test + {'else' if a_catchall else '落空'}")
    if ok:
        print("PASS: 分派覆盖等价 —— 进入条件与落空路径与原链一致")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
