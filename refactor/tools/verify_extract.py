#!/usr/bin/env python3
"""巨函数拆分的等价门：验「搬走的块只是换了个地方，一个节点都没动」。

`refactor/` 跑不起来（缺 import、无 __init__.py），所以拆巨函数时**没有测试能兜底**。
这个检查器提供机械可证的替代判据：把提取前后的 AST 归一化后逐节点比对。

它能证：搬运没有改动语句结构、没漏块、没多块、缩进变化不影响语义。
它证不了：运行时行为（那要等主线解冻后跑全量）。别把它当成测试。

用法：
    python3 refactor/tools/verify_extract.py <改动前的文件> <改动后的文件> \
        --moved <新函数名> --from <原函数名>

例：
    python3 refactor/tools/verify_extract.py /tmp/before.py \
        refactor/domains/windows_sampling/window_manager_dual_lambda.py \
        --moved _run_one_window --from run_all_windows
"""
import argparse
import ast
import sys


def _strip(node):
    """归一化：抹掉位置信息和 docstring，只留结构。"""
    for n in ast.walk(node):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset",
                     "type_comment"):
            if hasattr(n, attr):
                try:
                    delattr(n, attr)
                except AttributeError:
                    setattr(n, attr, None)
    return node


def _dump(nodes):
    return [ast.dump(_strip(n), annotate_fields=True) for n in nodes]


def _find(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    return None


def _body_without_docstring(fn):
    body = list(fn.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--moved", required=True, help="提取出来的新函数名")
    ap.add_argument("--src", required=True, dest="src", help="原来包住它的函数名")
    ap.add_argument("--block", default="for",
                    choices=("for", "if", "while", "with", "defs", "def", "fn"),
                    help="原函数里被搬走的块类型；defs = 搬走所有内嵌 def；"
                         "def = 搬走第 --index 个内嵌 def（刀3第0步：单 def 提升）；"
                         "fn = 整个函数体（配 --slice 做尾部/中段切片，刀1 B段第3步）")
    ap.add_argument("--index", type=int, default=0,
                    help="原函数顶层同类型块中的第几个（默认第 0 个）")
    ap.add_argument("--strip-call", action="append", default=[], metavar="NAME=K",
                    help="after 侧把对 NAME 的调用的前 K 个位置实参剥掉再比对"
                         "（刀3第0步：提升级联在调用点插入的捕获实参，还原后比对）")
    ap.add_argument("--orelse", action="store_true",
                    help="锚定块的 orelse[0] 整条语句为搬移对象（刀2：elif 链整体抽方法）")
    ap.add_argument("--slice", default=None, metavar="START:END",
                    help="只搬了锚点块体的 [START:END) 语句切片（刀1 A 段：_produce_window"
                         " = 窗口 for 体的 body[129:211]，其后还有留在调用方的释放协议块）；"
                         "0 起始、半开区间")
    a = ap.parse_args()

    tb = ast.parse(open(a.before, encoding="utf-8").read())
    ta = ast.parse(open(a.after, encoding="utf-8").read())

    src_before = _find(tb, a.src)
    if src_before is None:
        sys.exit(f"FAIL: 改动前的文件里没有 {a.src}")
    new_fn = _find(ta, a.moved)
    if new_fn is None:
        sys.exit(f"FAIL: 改动后的文件里没有 {a.moved}")

    # --- 取出改动前「本该被搬走」的那些语句 ---
    if a.block == "fn":
        expected = _body_without_docstring(src_before)
        if a.slice is not None:
            s_, _, e_ = a.slice.partition(":")
            s0, e0 = int(s_), int(e_)
            if not (0 <= s0 < e0 <= len(expected)):
                sys.exit(f"FAIL: --slice {a.slice} 超出函数体语句数 {len(expected)}")
            expected = expected[s0:e0]
        actual = _body_without_docstring(new_fn)
    elif a.block == "defs":
        expected = [n for n in _body_without_docstring(src_before)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        actual = _body_without_docstring(new_fn)
    elif a.block == "def":
        blocks = [n for n in _body_without_docstring(src_before)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(blocks) <= a.index:
            sys.exit(f"FAIL: {a.src} 顶层只有 {len(blocks)} 个内嵌 def，取不到第 {a.index} 个")
        if blocks[a.index].name != a.moved:
            sys.exit(f"FAIL: {a.src} 第 {a.index} 个内嵌 def 是 {blocks[a.index].name}，"
                     f"不是 --moved 的 {a.moved}")
        expected = _body_without_docstring(blocks[a.index])
        actual = _body_without_docstring(new_fn)
    else:
        kind = {"for": ast.For, "if": ast.If, "while": ast.While,
                "with": ast.With}[a.block]
        blocks = [n for n in _body_without_docstring(src_before)
                  if isinstance(n, kind)]
        if len(blocks) <= a.index:
            sys.exit(f"FAIL: {a.src} 顶层只有 {len(blocks)} 个 {a.block} 块，"
                     f"取不到第 {a.index} 个")
        if a.orelse:
            if len(blocks[a.index].orelse) != 1:
                sys.exit(f"FAIL: --orelse 需要锚点块恰有 1 条 orelse，实测 "
                         f"{len(blocks[a.index].orelse)}")
            expected = [blocks[a.index].orelse[0]]
            actual = _body_without_docstring(new_fn)
            de, da = _dump(expected), _dump(actual)
            print(f"原块语句数 {len(de)}  →  新函数体语句数 {len(da)}")
            if de == da:
                print("PASS: 逐节点等价 —— 搬运没有改动任何语句结构")
                return 0
            print("FAIL: AST 不等价")
            for i, (x, y) in enumerate(zip(de, da)):
                if x != y:
                    print(f"  第 {i} 条语句起出现差异：")
                    print(f"    改动前: {x[:180]}")
                    print(f"    改动后: {y[:180]}")
                    break
            return 1
        expected = list(blocks[a.index].body)
        if a.slice is not None:
            s_, _, e_ = a.slice.partition(":")
            s0, e0 = int(s_), int(e_)
            if not (0 <= s0 < e0 <= len(expected)):
                sys.exit(f"FAIL: --slice {a.slice} 超出块体语句数 {len(expected)}")
            expected = expected[s0:e0]
        actual = _body_without_docstring(new_fn)

    de, da = _dump(expected), _dump(actual)

    # --strip-call：after 侧还原（把插入的捕获实参剥掉）后比对
    if a.strip_call:
        for spec in a.strip_call:
            nm, _, k = spec.partition("=")
            k = int(k)

            class _Strip(ast.NodeTransformer):
                def visit_Call(self, node):
                    self.generic_visit(node)
                    if isinstance(node.func, ast.Name) and node.func.id == nm \
                            and len(node.args) >= k:
                        node.args = node.args[k:]
                    return node

            actual = [_Strip().visit(n) for n in actual]
            for n in actual:
                ast.fix_missing_locations(n)
            de, da = _dump(expected), _dump(actual)
        print(f"已按 --strip-call 还原 {len(a.strip_call)} 组调用点后比对")

    print(f"原块语句数 {len(de)}  →  新函数体语句数 {len(da)}")
    if de == da:
        print("PASS: 逐节点等价 —— 搬运没有改动任何语句结构")
        return 0

    print("FAIL: AST 不等价")
    if len(de) != len(da):
        print(f"  语句条数不同：{len(de)} vs {len(da)}"
              f"（漏搬 / 多搬 / 顺序变了）")
    for i, (x, y) in enumerate(zip(de, da)):
        if x != y:
            print(f"  第 {i} 条语句起出现差异：")
            print(f"    改动前: {x[:180]}")
            print(f"    改动后: {y[:180]}")
            break
    return 1


if __name__ == "__main__":
    sys.exit(main())
