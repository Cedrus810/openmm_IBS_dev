#!/usr/bin/env python3
"""A1 测试守卫解耦 —— 一次性转换脚本 + diff 提案生成器。

对应 refactor/HOW_TO_CONTINUE.md 第 4 章 A1：
把 tests/ 里「读主线 .py 文本」的守卫从「绑文件路径」变成「绑代码对象」
（inspect.getsource(模块.类.方法)），断言字面量一字不改。

纯 AST + 少量正则辅助，**不 import 任何主线模块**（主线 import openmm 很慢）。

用法（仓库根执行）：
    python3 refactor/tools/a1_tool.py            # 生成 refactor/work/a1_proposals/
    python3 refactor/tools/a1_tool.py --check    # 只报统计，不写文件

产出（默认模式）：
    refactor/work/a1_proposals/<testfile>.diff   每个受影响测试文件一个 unified diff 提案
    refactor/work/a1_proposals/INDEX.md          汇总表 + not_found / C1 暴露 / ambiguous 明细

分类法（HOW_TO_CONTINUE 2.2）：
    C2  好写法必须存在     assert "lit" in src / count >= 1
    C1  坏写法不得出现     assert "lit" not in src
    C3  唯一性计数         assert src.count("lit") == N
    C4  顺序/定位           src.index("a") < src.index("b")
    C5  其他               re.search / ast.parse / splitlines ...

每条断言的状态（对断言字面量在主线文件里做 AST 反查）：
    auto         字面量唯一命中一个顶层定义 → 提案收窄到 inspect.getsource(那个定义)
    ambiguous    命中多个定义 → 列出全部候选，diff 只给模块级绑定，收窄留人工
    module_level 字面量在模块顶层（import 区 / 模块级语句）→ inspect.getsource(模块)
    not_found    字面量根本不在该主线文件里 → 守卫钉错地方 / 测试已腐烂，重点标记
    c1_no_target C1 守卫（not in 全文件）且字面量当前全文件不存在（守卫生效中的正常态）；
                 收窄无目标可选，提案保底绑模块对象，A2 再升包级 AST
    c1_exposed   C1 守卫但字面量在文件里出现了 → 测试要么已在失败，要么命中的是
                 注释/字符串；「靠扫全文件才成立」的守卫，单独标记
"""
from __future__ import annotations

import argparse
import ast
import difflib
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]   # 仓库根(原 refactor/ 直下,重组后 tools/ 深一层)
TESTS = REPO / "tests"
OUT_DIR = REPO / "refactor" / "work" / "a1_proposals"  # 保持不变

# ─────────────────────────────────────────────────────────────────────────────
# 主线模块识别：仓库根的 *.py（不含 refactor/、tests/）
# ─────────────────────────────────────────────────────────────────────────────

def mainline_modules() -> dict[str, Path]:
    mods = {}
    for p in sorted(REPO.glob("*.py")):
        mods[p.stem] = p
    return mods

MAINLINE = mainline_modules()


# ─────────────────────────────────────────────────────────────────────────────
# 主线文件区域索引（反查字面量落在哪个顶层定义里）
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Region:
    qualname: str            # "abfe_core" / "abfe_core.ABFEPipeline.run_full_pipeline" / "<module>"
    is_module: bool          # 模块顶层代码（import 区、模块级语句）
    start: int               # 1-based 行号（含装饰器）
    end: int
    text: str                # 原始源码段


@dataclass
class ModuleIndex:
    name: str
    regions: list[Region]
    raw: str

    def hit_regions(self, lit: str) -> list[Region]:
        """字面量命中的区域。先全文原样搜（含转义形态），全空才走 AST 常量回退。"""
        forms = _escaped_forms(lit)
        hits = []
        for r in self.regions:
            if any(f in r.text for f in forms):
                hits.append(r)
        if hits:
            return hits
        # AST 常量回退（处理隐式拼接/转义差异）——整树只走一遍
        for node in ast.walk(self._tree()):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value
                if lit in v or any(f in v for f in forms if f != lit):
                    for r in self.regions:
                        if r.start <= node.lineno <= r.end:
                            hits.append(r)
        return hits

    _tree_cache: ast.Module | None = None

    def _tree(self) -> ast.Module:
        if self._tree_cache is None:
            self._tree_cache = ast.parse(self.raw)
        return self._tree_cache


def _escaped_forms(s: str) -> list[str]:
    """测试字面量(已解码)与源码原文之间的常见形态差。"""
    forms = {s}
    forms.add(s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t"))
    forms.add(s.replace('"', '\\"'))
    forms.add(s.replace('"', '\\"').replace("\n", "\\n"))
    return [f for f in forms if f]


def build_module_index(mod: str) -> ModuleIndex:
    path = MAINLINE[mod]
    raw = path.read_text(encoding="utf-8")
    tree = ast.parse(raw)
    lines = raw.splitlines(keepends=True)

    def seg(start: int, end: int) -> str:
        return "".join(lines[start - 1:end])

    regions: list[Region] = []

    def add_module_stmts(stmts):
        for st in stmts:
            regions.append(Region("<module>", True,
                                  st.lineno, st.end_lineno, seg(st.lineno, st.end_lineno)))

    def walk_body(body, prefix):
        for st in body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = st.lineno
                if st.decorator_list:
                    start = min(d.lineno for d in st.decorator_list)
                regions.append(Region(f"{prefix}.{st.name}", False,
                                      start, st.end_lineno, seg(start, st.end_lineno)))
            elif isinstance(st, ast.ClassDef):
                start = st.lineno
                if st.decorator_list:
                    start = min(d.lineno for d in st.decorator_list)
                # 类体里非方法的语句 = 类体区域
                other = [s for s in st.body
                         if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
                if other:
                    s0 = min(s.lineno for s in other)
                    e0 = max(s.end_lineno for s in other)
                    regions.append(Region(f"{prefix}.{st.name}.<classbody>", False,
                                          s0, e0, seg(s0, e0)))
                walk_body(st.body, f"{prefix}.{st.name}")
            else:
                regions.append(Region("<module>", True,
                                      st.lineno, st.end_lineno, seg(st.lineno, st.end_lineno)))

    walk_body(tree.body, mod)
    regions.sort(key=lambda r: (r.start, r.end))
    return ModuleIndex(mod, regions, raw)


MODULE_INDEX_CACHE: dict[str, ModuleIndex] = {}


def module_index(mod: str) -> ModuleIndex | None:
    if mod not in MAINLINE:
        return None
    if mod not in MODULE_INDEX_CACHE:
        MODULE_INDEX_CACHE[mod] = build_module_index(mod)
    return MODULE_INDEX_CACHE[mod]


# ─────────────────────────────────────────────────────────────────────────────
# 测试文件侧：定位「读主线 .py 文本」的绑定与消费点
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Prov:
    """一个变量携带的「源码 provenance」。"""
    kind: str                 # 'src' | 'slice' | 'derived'
    module: str | None        # 主线模块名（None = 解析不了/非主线）
    path: str | None = None   # 原始文件相对路径（tools/ 等非主线文件）
    base: str | None = None   # slice 的基变量
    bounds: list = field(default_factory=list)  # slice 边界里的字面量
    site: object | None = None  # 溯源到哪个 BindingSite（精确归属，防同名变量串味）


@dataclass
class Assertion:
    file: str
    line: int                 # assert 语句行号
    cls: str                  # C1..C5
    op: str                   # in / not_in / count_eq / count_ge / index_cmp / other
    literals: list[str]       # 涉及的断言字面量（C5 可能为空或 pattern）
    var: str                  # 源变量显示名
    module: str | None        # 目标主线模块
    path: str | None          # 目标文件（非主线时）
    scope: str                # whole_file / narrowed / derived
    hits: list = field(default_factory=list)     # [(region)] 每个字面量的命中叶子并集
    module_hits: bool = False
    per_literal: dict = field(default_factory=dict)  # lit -> {"leaves": [...], "module_level": bool}
    in_assert: bool = True
    note: str = ""
    site: object | None = None  # 精确归属的读文本站点


@dataclass
class BindingSite:
    file: str
    line: int
    module: str | None        # 主线模块名；None = 动态/非主线
    path: str | None          # 相对路径
    dynamic: bool = False     # 文件名来自变量/参数，静态解析不了
    spec_load: bool = False   # importlib.util.spec_from_file_location
    var: str = ""             # 绑定到哪个变量（显示用）
    node_ref: object | None = None  # helper 调用点：待替换的 Call 节点
    consumers: list = field(default_factory=list)  # Assertion 列表
    target: str | None = None  # 提案的 inspect.getsource 目标表达式


class TestFileAnalyzer:
    @staticmethod
    def _read_preserve_newlines(path: Path) -> str:
        # newline=""：保留 CRLF。read_text 的默认通用换行会把 \r\n 折成 \n，
        # 生成的 diff 就变成 LF 而应用到 CRLF 文件（tests/ 里有 2 个）必 FAILED。
        with path.open(encoding="utf-8", newline="") as fh:
            return fh.read()

    def __init__(self, path: Path):
        self.path = path
        self.rel = str(path.relative_to(REPO))
        self.raw = self._read_preserve_newlines(path)
        self.lines = self.raw.splitlines(keepends=True)
        self.tree = ast.parse(self.raw)
        _PARENT.clear()
        build_parent_index(self.tree)
        self.imports: dict[str, str] = {}   # 模块名 -> 绑定名（含 asname）
        self.bindings: list[BindingSite] = []
        self.assertions: list[Assertion] = []
        self.spec_loads: list[BindingSite] = []
        self._helpers: dict[str, ast.FunctionDef] = {}
        self._scan()

    # ── import 收集 ──
    def _collect_imports(self):
        # 只登记「模块对象可用」的绑定（import X / import X as y），且只认
        # **模块级**绑定：tree.body 往下钻 If/Try/With（try: import x 的守卫
        # 同样是模块级执行），不进 FunctionDef/ClassDef —— 函数局部
        # `import ibs_engine as ie` 只在那一个函数里可见，拿它当全文件绑定名
        # 会生成 NameError（影子树实跑 13 处失败的根因）。from X import y 不产生
        # 模块绑定 —— diff 生成时会补 `import X`，否则 target_expr 会引用
        # 一个不存在的名字。
        def walk_modlevel(stmts):
            for node in stmts:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(node, ast.Import):
                    for a in node.names:
                        if a.name in MAINLINE:
                            self.imports[a.name] = a.asname or a.name.split(".")[0]
                elif isinstance(node, (ast.If, ast.Try, ast.With, ast.AsyncWith)):
                    walk_modlevel(node.body)
                    for h in getattr(node, "handlers", ()):
                        walk_modlevel(h.body)
                    for extra in ("orelse", "finalbody"):
                        walk_modlevel(getattr(node, extra, ()))
        walk_modlevel(self.tree.body)

    # ── helper 函数识别：返回 (…/f(name)).read_text() 的函数 ──
    def _collect_helpers(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                params = {a.arg for a in node.args.args}
                for ret in ast.walk(node):
                    if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Call):
                        f = ret.value.func
                        if (isinstance(f, ast.Attribute) and f.attr == "read_text"):
                            self._helpers[node.name] = node
                            # helper 里文件名来自参数 → 动态站点
                            recv = f.value
                            if any(isinstance(n, ast.Name) and n.id in params
                                   for n in ast.walk(recv)):
                                self.bindings.append(BindingSite(
                                    file=self.rel, line=ret.lineno, module=None, path=None,
                                    dynamic=True, var=f"{node.name}()"))

    # ── 路径表达式 → 模块名 ──
    def _resolve_path_expr(self, node, env) -> Prov | None:
        """把路径类表达式解析成 Prov（module 或 path），失败返回 None。"""
        # 裸字符串 "x.py"（open("x.py").read() 等）
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value.endswith(".py"):
            mod = Path(node.value).stem
            return Prov("src", mod if mod in MAINLINE else None, path=node.value)
        # Name → env 查
        if isinstance(node, ast.Name):
            return env.get(node.id)
        # a / b / "x.py" 链：找最右的 ".py" 常量
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            parts = _binop_parts(node)
            for p in reversed(parts):
                if isinstance(p, ast.Constant) and isinstance(p.value, str) and p.value.endswith(".py"):
                    rel = p.value
                    mod = Path(rel).stem
                    prov = Prov("src", mod if mod in MAINLINE else None,
                                path=rel if mod not in MAINLINE else rel)
                    return prov
            return None
        # Path(ie.__file__) / ie.__file__
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path":
            return self._resolve_path_expr(node.args[0], env) if node.args else None
        if isinstance(node, ast.Attribute) and node.attr == "__file__":
            base = node.value
            if isinstance(base, ast.Name):
                for m, bind in self.imports.items():
                    if bind == base.id:
                        return Prov("src", m, path=f"{m}.py")
            if isinstance(base, ast.Attribute) and base.attr in MAINLINE:
                return Prov("src", base.attr, path=f"{base.attr}.py")
        # os.path.join(ROOT, "x.py")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "join"):
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.endswith(".py"):
                    mod = Path(a.value).stem
                    return Prov("src", mod if mod in MAINLINE else None, path=a.value)
        return None

    # ── read_text / open().read() 调用识别 ──
    def _read_call_prov(self, node: ast.Call, env) -> Prov | None:
        f = node.func
        # X.read_text(...) / X.read(...) / X.read_bytes(...)
        if isinstance(f, ast.Attribute) and f.attr in ("read_text", "read", "read_bytes"):
            if f.attr == "read":
                # open(path).read() —— 只认 open() 直调
                if not (isinstance(f.value, ast.Call) and isinstance(f.value.func, ast.Name)
                        and f.value.func.id == "open"):
                    return None
                prov = self._resolve_path_expr(f.value.args[0] if f.value.args else None, env)
            else:
                prov = self._resolve_path_expr(f.value, env)
            if prov is None and isinstance(f.value, ast.AST):
                # 文件名走参数/变量的读 —— 记为动态站点（INDEX 里列人工项）
                self._register_binding(node, Prov("src", None), env)
            return prov
        return None

    # ── helper 调用解析：_src("ibs_engine.py") ──
    def _helper_call_prov(self, node: ast.Call, env) -> Prov | None:
        if not (isinstance(node.func, ast.Name) and node.func.id in self._helpers):
            return None
        fn = self._helpers[node.func.id]
        params = [a.arg for a in fn.args.args]
        subst = dict(zip(params, node.args))
        for ret in ast.walk(fn):
            if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Call):
                f = ret.value.func
                if isinstance(f, ast.Attribute) and f.attr in ("read_text", "read"):
                    recv = f.value
                    if isinstance(f.attr, str) and f.attr == "read" and not (
                            isinstance(recv, ast.Call) and getattr(recv.func, "id", "") == "open"):
                        continue
                    recv = recv.args[0] if f.attr == "read" and isinstance(recv, ast.Call) else recv
                    # 表达式里把参数名换成常量
                    resolved = _subst_names(recv, subst)
                    prov = self._resolve_path_expr(resolved, env)
                    if prov is not None:
                        # 纯读 helper：调用点即读文本站点，diff 直接改调用点
                        site = BindingSite(file=self.rel, line=node.lineno,
                                           module=prov.module, path=prov.path,
                                           var=self._host_var(node) or "",
                                           node_ref=node)
                        self.bindings.append(site)
                        prov.site = site
                    return prov
        return None

    # ── 主扫描：递归作用域 ──
    def _scan(self):
        self._collect_imports()
        self._collect_helpers()
        self._walk_scope(self.tree.body, {}, "")

    def _walk_scope(self, body, inherited: dict, scope_name: str):
        env = dict(inherited)
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                saved = getattr(self, "_params", set())
                self._params = {a.arg for a in stmt.args.args} | {a.arg for a in stmt.args.kwonlyargs}
                self._walk_scope(stmt.body, env, stmt.name)
                self._params = saved
            elif isinstance(stmt, ast.ClassDef):
                # 类级：先把类体里任何 self.X / cls.X 赋值（setUpClass 等）收进类环境
                cls_env = dict(env)
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.ClassDef) and sub is not stmt:
                        continue
                    if isinstance(sub, ast.Assign):
                        for t in sub.targets:
                            if (isinstance(t, ast.Attribute)
                                    and isinstance(t.value, ast.Name)
                                    and t.value.id in ("self", "cls")):
                                prov = self._eval_prov(sub.value, env)
                                if prov is not None:
                                    cls_env[f"self.{t.attr}"] = prov
                                    cls_env[f"cls.{t.attr}"] = prov
                self._walk_scope(stmt.body, cls_env, stmt.name)
            elif isinstance(stmt, ast.For) and isinstance(stmt.iter, (ast.List, ast.Tuple)):
                # for marker in ("a", "b"): assert marker in src —— 循环展开
                lits = [_const_str(e) for e in stmt.iter.elts]
                if all(l is not None for l in lits) and lits \
                        and isinstance(stmt.target, ast.Name):
                    env[stmt.target.id] = Prov("lits", None, bounds=lits)
                self._process_stmt(stmt, env, scope_name)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):                # with open("x.py") as fh: ... → fh 带源码 provenance
                for item in stmt.items:
                    ctx = item.context_expr
                    if isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Name) \
                            and ctx.func.id == "open" and ctx.args:
                        prov = self._resolve_path_expr(ctx.args[0], env)
                        if prov is not None or _has_py_string(ctx.args[0]):
                            self._register_binding(ctx, prov or Prov("src", None), env)
                            if item.optional_vars is not None and isinstance(item.optional_vars, ast.Name) and prov is not None:
                                env[item.optional_vars.id] = prov
                self._process_stmt(stmt, env, scope_name)
            else:
                self._process_stmt(stmt, env, scope_name)

    def _process_stmt(self, stmt, env: dict, scope_name: str):
        # 嵌套 def/class 由 _scan 里各自的 scope walk 处理，这里不下去，避免重复计数
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        # 1) 赋值 → provenance
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            value = stmt.value
            targets = []
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
            elif stmt.target is not None:
                targets = [stmt.target]
            prov = self._eval_prov(value, env)
            # i = src.index("lit") / find —— 位置变量，供后续切片归因
            if prov is None and isinstance(value, ast.Call) \
                    and isinstance(value.func, ast.Attribute) \
                    and value.func.attr in ("index", "find") and value.args:
                base = self._src_prov(value.func.value, env)
                lit = _const_str(value.args[0])
                if base is not None and base.kind in ("src", "slice") and lit is not None:
                    prov = Prov("pos", base.module, path=base.path, site=base.site,
                                base=base.base or value.func.value.id if isinstance(value.func.value, ast.Name) else None, bounds=[lit])
            for t in targets:
                if isinstance(t, ast.Name):
                    if prov is not None:
                        env[t.id] = prov
                    else:
                        env.pop(t.id, None)
                elif isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) \
                        and t.value.id in ("self", "cls"):
                    key = f"{t.value.id}.{t.attr}"
                    if prov is not None:
                        env[key] = prov
                    else:
                        env.pop(key, None)
        # 2) 断言/表达式里的源码消费
        self._collect_uses(stmt, env, scope_name)

    def _eval_prov(self, node, env) -> Prov | None:
        if node is None:
            return None
        # read_text / open().read()
        if isinstance(node, ast.Call):
            prov = self._read_call_prov(node, env)
            if prov is not None:
                self._register_binding(node, prov, env)
                return prov
            prov = self._helper_call_prov(node, env)
            if prov is not None:
                return prov
            # spec_from_file_location
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr == "spec_from_file_location"):
                mod = None
                path = None
                if len(node.args) >= 2:
                    p = self._resolve_path_expr(node.args[1], env)
                    if p is not None:
                        mod, path = p.module, p.path
                site = BindingSite(file=self.rel, line=node.lineno, module=mod, path=path,
                                   dynamic=(mod is None and path is None), spec_load=True)
                self.bindings.append(site)
                self.spec_loads.append(site)
                return None
            # 派生：src.splitlines() / src.split("def ") / json.loads(src) 等 —— 带着模块信息
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                base = env.get(node.func.value.id)
                if base is not None and base.module and base.kind in ("src", "slice"):
                    if node.func.attr in ("splitlines", "strip", "replace", "split",
                                          "rsplit", "partition", "lower", "upper"):
                        bounds = [_const_str(a) for a in node.args
                                  if _const_str(a) is not None]
                        return Prov("slice", base.module, path=base.path,
                                    base=node.func.value.id, bounds=bounds, site=base.site)
            return None
        # 别名
        if isinstance(node, ast.Name):
            p = env.get(node.id)
            return p
        # self.xxx / cls.xxx
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in ("self", "cls"):
            return env.get(f"{node.value.id}.{node.attr}")
        # 切片：fn = src[a:b]（边界可以是常量、位置变量、及其算术组合）
            if isinstance(node, ast.Subscript):
                base = self._eval_prov(node.value, env)
                if base is not None and base.kind in ("src", "slice"):
                    bounds = []
                    sl = node.slice
                    if isinstance(sl, ast.Slice):
                        for b in (sl.lower, sl.upper):
                            bounds.extend(_bound_lits(b, env))
                    elif isinstance(sl, ast.Name):
                        bounds.extend(_bound_lits(sl, env))
                    return Prov("slice", base.module, path=base.path, base=node.value.id if isinstance(node.value, ast.Name) else base.base, bounds=bounds, site=base.site)
                return None
        # max(0, i - 3000) 之类包装切片边界的调用
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("max", "min", "len") and node.args:
            for a in node.args:
                p = self._eval_prov(a, env)
                if p is not None:
                    return p
        return None
        # 布尔/其他不追
        return None

    def _register_binding(self, call: ast.Call, prov: Prov, env):
        mod = prov.module
        path = prov.path
        dynamic = False
        # 文件名解析不了 → dynamic
        if mod is None and path is None:
            # receiver 含 ".py" 字面量或当前作用域参数名（文件名走变量）→ 动态站点
            recv = call.func.value if isinstance(call.func, ast.Attribute) else None
            has_param = recv is not None and any(
                isinstance(n, ast.Name) and n.id in getattr(self, "_params", ())
                for n in ast.walk(recv))
            if recv is not None and (_has_py_string(recv) or has_param):
                dynamic = True
            else:
                return  # 不是在读 .py（可能是 json/txt —— 上游已过滤大半）
        site = BindingSite(file=self.rel, line=call.lineno, module=mod, path=path,
                           dynamic=dynamic, var="")
        # 找宿主赋值变量名
        site.var = self._host_var(call) or ""
        prov.site = site
        self.bindings.append(site)

    def _host_var(self, call: ast.Call) -> str | None:
        parent = _PARENT.get(id(call))
        while parent is not None:
            if isinstance(parent, ast.Assign):
                for t in parent.targets:
                    if isinstance(t, ast.Name):
                        return t.id
                    if isinstance(t, ast.Attribute):
                        return _unparse(t)
                return None
            if isinstance(parent, ast.AnnAssign):
                return _unparse(parent.target)
            if isinstance(parent, (ast.Assign, ast.Call, ast.Return, ast.Expr,
                                   ast.Assign, ast.keyword, ast.Subscript, ast.BinOp,
                                   ast.Compare, ast.Attribute, ast.AugAssign)):
                nxt = _PARENT.get(id(parent))
                if nxt is None:
                    return None
                parent = nxt
                continue
            return None
        return None

    # ── 源码消费点收集 ──
    def _collect_uses(self, stmt, env, scope_name: str):
        for node in ast.walk(stmt):
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and node is stmt.value:
                continue  # 赋值右侧已在 _eval_prov 里处理
            if isinstance(node, ast.Compare):
                self._check_compare(node, env, scope_name)
            elif isinstance(node, ast.Call):
                self._check_call(node, env, scope_name)
            elif isinstance(node, ast.Subscript):
                # src[i:j] 用作值（C4 派生已在 _eval_prov 记录 slice）
                pass

    def _src_prov(self, node, env) -> Prov | None:
        """表达式对应的源码 provenance（含 self.engine 形式）。"""
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in ("self", "cls"):
            return env.get(f"{node.value.id}.{node.attr}")
        if isinstance(node, ast.Call):
            return self._eval_prov(node, env)
        if isinstance(node, ast.Subscript):
            return self._eval_prov(node, env)
        return None

    def _add_in_checks(self, cmp, left, right_src: Prov, env, negate: bool):
        """`lit in src` / `lit not in src`，含循环展开与 f-string 兜底。"""
        in_assert = _in_assert(cmp)
        lits = []
        if isinstance(left, ast.Constant) and isinstance(left.value, str):
            lits = [left.value]
        elif isinstance(left, ast.Name):
            p = env.get(left.id)
            if p is not None and p.kind == "lits":
                lits = list(p.bounds)
        if lits:
            for lit in lits:
                self._add_assert(cmp, "C1" if negate else "C2",
                                 "not_in" if negate else "in", [lit], right_src, env,
                                 in_assert=in_assert,
                                 note="循环展开" if len(lits) > 1 else "")
            return
        if isinstance(left, ast.JoinedStr):
            # f-string：静态常量段是求值结果的**保证出现**部分（无占位符时整串
            # 就是字面量）。旧版整条记 C5 人工 ⟹ binding_target 看不见它，
            # 收窄目标可能漏掉住在别的函数里的字面量（test_untrusted_switch:136
            # 那次：`f"allow_untrusted_stage_results=bool("` 住在 main，绑定却被
            # 收窄到 build_parser）。纯常量 → 当普通字面量；混合 → 逐段记。
            parts = [v.value for v in left.values
                     if isinstance(v, ast.Constant) and isinstance(v.value, str)]
            has_ph = any(isinstance(v, ast.FormattedValue) for v in left.values)
            if not has_ph and parts:
                lit = "".join(parts)
                self._add_assert(cmp, "C1" if negate else "C2",
                                 "not_in" if negate else "in", [lit], right_src, env,
                                 in_assert=in_assert, note="f-string 纯常量")
                return
            for lit in parts:
                self._add_assert(cmp, "C1" if negate else "C2",
                                 "not_in" if negate else "in", [lit], right_src, env,
                                 in_assert=in_assert, note="f-string 静态段")
            if not parts:
                self._add_assert(cmp, "C5", "fstring_in", [], right_src, env,
                                 in_assert=in_assert, note="f-string 全动态（需人工）")
            return
        if isinstance(left, ast.Constant):
            return  # 非 str 常量的成员测试，不是源码守卫
        # 字面量来自更复杂的表达式（f(x)、相加等）—— 记为动态
        self._add_assert(cmp, "C5", "dynamic_in", [], right_src, env,
                         in_assert=in_assert, note="动态字面量（需人工）")

    def _check_compare(self, cmp: ast.Compare, env, scope_name):
        for idx, op in enumerate(cmp.ops):
            left = cmp.left if idx == 0 else cmp.comparators[idx - 1]
            right = cmp.comparators[idx]
            left_src = self._src_prov(left, env)
            right_src = self._src_prov(right, env)
            if left_src is not None and left_src.kind == "lits":
                left_src = None
            if right_src is not None and right_src.kind == "lits":
                right_src = None
            if isinstance(op, ast.In) and left_src is None and right_src is not None:
                self._add_in_checks(cmp, left, right_src, env, negate=False)
            elif isinstance(op, ast.NotIn) and left_src is None and right_src is not None:
                self._add_in_checks(cmp, left, right_src, env, negate=True)
            elif isinstance(op, (ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq)):
                # src.count(lit) == N / src.index(a) < src.index(b)
                for a, b in ((left, right), (right, left)):
                    if isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute) \
                            and a.func.attr == "count":
                        base = self._src_prov(a.func.value, env)
                        lit = _const_str(a.args[0]) if a.args else None
                        if base is not None and lit is not None:
                            other = b if a is left else left
                            if isinstance(other, ast.Constant):
                                if isinstance(op, ast.Eq):
                                    self._add_assert(cmp, "C3", "count_eq", [lit], base, env,
                                                     in_assert=_in_assert(cmp), note=f"== {other.value!r}")
                                elif isinstance(op, (ast.GtE, ast.Gt)) and (
                                        isinstance(other.value, int) and other.value >= 1):
                                    self._add_assert(cmp, "C2", "count_ge", [lit], base, env,
                                                     in_assert=_in_assert(cmp),
                                                     note=f"{type(op).__name__} {other.value}")
                            elif isinstance(other, ast.Call) and getattr(other.func, "attr", "") == "count":
                                lit2 = _const_str(other.args[0]) if other.args else None
                                if lit2 is not None:
                                    self._add_assert(cmp, "C3", "count_eq", [lit, lit2], base, env,
                                                     in_assert=_in_assert(cmp), note="count 与 count 比较")
                    if isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute) \
                            and a.func.attr in ("index", "find"):
                        base = self._src_prov(a.func.value, env)
                        lit = _const_str(a.args[0]) if a.args else None
                        if base is not None and lit is not None:
                            other = b if a is left else left
                            if isinstance(other, ast.Call) and getattr(other.func, "attr", "") in ("index", "find"):
                                lit2 = _const_str(other.args[0]) if other.args else None
                                if lit2 is not None:
                                    self._add_assert(cmp, "C4", "index_cmp", [lit, lit2], base, env,
                                                     in_assert=_in_assert(cmp))
                                else:
                                    self._add_assert(cmp, "C4", "index_cmp", [lit], base, env,
                                                     in_assert=_in_assert(cmp))
                            else:
                                self._add_assert(cmp, "C4", "index_cmp", [lit], base, env,
                                                 in_assert=_in_assert(cmp), note="index/find 用于比较")

    def _check_call(self, call: ast.Call, env, scope_name):
        f = call.func
        # re.search/findall/match(pattern, src)
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "re" \
                and f.attr in ("search", "findall", "match", "fullmatch", "sub"):
            for a in call.args:
                if isinstance(a, ast.Name):
                    p = env.get(a.id)
                    if p is not None and p.kind == "src":
                        self._add_assert(call, "C5", "regex", [], p, env,
                                         in_assert=_in_assert(call), note=f"re.{f.attr}")
        # src.count / .index 单独出现（不在 Compare 里）→ C5 记录
        if isinstance(f, ast.Attribute) and f.attr in ("count", "index", "find"):
            base = self._src_prov(f.value, env)
            if base is not None and not isinstance(_PARENT.get(id(call)), ast.Compare):
                lit = _const_str(call.args[0]) if call.args else None
                if lit is not None:
                    cls = "C5"
                    self._add_assert(call, cls, f"{f.attr}_use", [lit], base, env,
                                     in_assert=_in_assert(call),
                                     note="index/count 脱离比较使用（定位辅助）")
        # json.loads / ast.parse / print 等对源变量的其他消费
        if isinstance(f, ast.Name) and f.id in ("json",) :
            pass
        if isinstance(f, ast.Attribute) and f.attr == "parse" and isinstance(f.value, ast.Name) \
                and f.value.id == "ast":
            for a in call.args:
                p = self._src_prov(a, env)
                if p is not None and p.kind in ("src", "slice"):
                    self._add_assert(call, "C5", "ast_parse", [], p, env,
                                     in_assert=_in_assert(call), note="ast.parse(源文本)")

    def _add_assert(self, node, cls, op, literals, prov: Prov, env, in_assert=True, note=""):
        mod = prov.module
        path = prov.path
        scope = "whole_file" if prov.kind == "src" else ("narrowed" if prov.kind == "slice" else "derived")
        a = Assertion(file=self.rel, line=node.lineno, cls=cls, op=op, literals=literals,
                      var=_prov_display(node, prov), module=mod, path=path, scope=scope,
                      in_assert=in_assert, note=note, site=getattr(prov, "site", None))
        # C1-C4 一律反查；C5 里 index/count 的定位字面量也要反查——
        # 它们若落在候选收窄区域之外，收窄后 .index 会当场 ValueError
        if cls in ("C1", "C2", "C3", "C4") or (cls == "C5" and op in ("index_use", "count_use")):
            if mod:
                mi = module_index(mod)
                leaves = []
                module_level = False
                per_lit = {}
                for lit in literals:
                    if mi is None:
                        per_lit[lit] = {"leaves": [], "module_level": False, "no_index": True}
                        continue
                    hits = mi.hit_regions(lit)
                    lv = [r.qualname for r in hits if not r.is_module]
                    ml = any(r.is_module for r in hits)
                    per_lit[lit] = {"leaves": lv, "module_level": ml}
                    leaves.extend(lv)
                    module_level = module_level or ml
                a.per_literal = per_lit
                a.hits = sorted(set(leaves))
                a.module_hits = module_level
        self.assertions.append(a)


def _binop_parts(node):
    parts = []
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
            stack.append(n.right)
            stack.append(n.left)
        else:
            parts.append(n)
    return parts


def _const_str(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _bound_lits(node, env) -> list:
    """切片边界里能静态看出的字面量（内联 .index("...") 或位置变量）。"""
    out = []
    if node is None:
        return out
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr in ("index", "find") and n.args:
            lit = _const_str(n.args[0])
            if lit is not None:
                out.append(lit)
        elif isinstance(n, ast.Name):
            p = env.get(n.id)
            if p is not None and p.kind == "pos":
                out.extend(p.bounds)
    return out


def _has_py_string(node) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith(".py"):
            return True
    return False


def _subst_names(node, subst: dict):
    """把表达式里的 Name 换成调用点常量（浅拷贝）。"""
    if isinstance(node, ast.Name) and node.id in subst and isinstance(subst[node.id], ast.Constant):
        return subst[node.id]
    for child in ast.iter_child_nodes(node):
        _subst_names_inplace(child, subst)
    return node


def _subst_names_inplace(node, subst):
    for fld, val in ast.iter_fields(node):
        if isinstance(val, ast.Name) and val.id in subst:
            setattr(node, fld, subst[val.id])
        elif isinstance(val, ast.AST):
            _subst_names_inplace(val, subst)
        elif isinstance(val, list):
            for i, v in enumerate(val):
                if isinstance(v, ast.Name) and v.id in subst:
                    val[i] = subst[v.id]
                elif isinstance(v, ast.AST):
                    _subst_names_inplace(v, subst)


def _in_assert(node) -> bool:
    p = _PARENT.get(id(node))
    while p is not None:
        if isinstance(p, ast.Assert):
            return True
        p = _PARENT.get(id(p))
    return False


def _prov_display(node, prov: Prov) -> str:
    return _unparse(node)


def _unparse(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<?>"


# 父节点索引（模块级做一次）
_PARENT: dict[int, ast.AST] = {}


def build_parent_index(tree):
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            _PARENT[id(child)] = node


# ─────────────────────────────────────────────────────────────────────────────
# 状态判定 + 提案目标选择
# ─────────────────────────────────────────────────────────────────────────────

def assertion_status(a: Assertion) -> str:
    if a.cls == "C5":
        return "c5"
    if a.module is None:
        return "unresolved"
    if a.cls == "C1":
        if not a.hits and not a.module_hits:
            return "c1_no_target"      # 全文件不存在 —— 守卫生效中的正常态
        return "c1_exposed"            # 字面量在文件里出现 —— 严重标记
    if not a.hits and not a.module_hits:
        return "not_found"
    if a.hits and not a.module_hits and len(a.hits) == 1:
        return "auto"
    if a.module_hits and not a.hits:
        return "module_level"
    return "ambiguous"


# C5 里「位置型」消费：收窄到函数后若字面量不在该函数里，.index/.find 当场 ValueError
POS_OPS = ("index_use", "count_use")
# C5 里对任何目标文本都等价可用的消费（getsource(模块/类/方法) 都能 parse/regex）
SAFE_C5_OPS = ("regex", "ast_parse")


def binding_target(site: BindingSite, assertions: list[Assertion]) -> str | None:
    """该绑定提案换成 inspect.getsource(<target>) 的目标表达式（模块相对名）。

    收窄条件（全部满足才收窄到唯一定义）：
    - 所有 C1-C4 断言都 auto 且命中同一个顶层定义；
    - 所有 C5 位置型消费（.index/.find/.count 定位）的字面量也落在该定义里；
    - 没有 not_found / ambiguous / c1_exposed / module_level / 动态字面量。
    任一不满足 → 保底绑模块对象（getsource(模块) == 整文件文本，逐字节等价）。
    """
    if site.module is None:
        return None
    auto_targets = set()
    for a in assertions:
        st = assertion_status(a)
        if st == "auto":
            auto_targets.add(a.hits[0])
        elif st in ("not_found", "c1_no_target", "c1_exposed", "module_level",
                    "ambiguous", "unresolved"):
            return site.module
    if auto_targets and len(auto_targets) == 1:
        tgt = next(iter(auto_targets))
        region_leaves = _region_leaf_set(tgt)
        for a in assertions:
            if assertion_status(a) == "c5" and a.op in POS_OPS:
                if any(l not in region_leaves for l in a.literals):
                    return site.module
        return tgt
    return site.module


_REGION_LEAF_CACHE: dict[str, set] = {}


def _region_leaf_set(qualname: str) -> set:
    """该区域（含其子定义）覆盖的 qualname 集合 —— 用于位置字面量归属判断。"""
    if qualname in _REGION_LEAF_CACHE:
        return _REGION_LEAF_CACHE[qualname]
    mod = qualname.split(".")[0]
    mi = module_index(mod)
    out = set()
    if mi is not None:
        for r in mi.regions:
            if r.qualname == qualname or r.qualname.startswith(qualname + "."):
                out.add(r.qualname)
        out.add(qualname)
    _REGION_LEAF_CACHE[qualname] = out
    return out


def target_expr(mod: str, rel_target: str | None, imports: dict[str, str]) -> str:
    """rel_target 形如 'abfe_core.ABFEPipeline.run_full_pipeline'；返回测试里的目标表达式。"""
    bind = imports.get(mod, mod)
    if rel_target is None or rel_target == mod:
        return bind
    suffix = rel_target[len(mod):]          # ".ABFEPipeline.run_full_pipeline"
    return bind + suffix


# ─────────────────────────────────────────────────────────────────────────────
# diff 生成
# ─────────────────────────────────────────────────────────────────────────────

def _host_func_of(an: TestFileAnalyzer, lineno: int):
    """包含 lineno 的最内层函数定义；模块级站点返回 None。"""
    best = None
    for n in ast.walk(an.tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and n.lineno <= lineno <= n.end_lineno:
            if best is None or (n.end_lineno - n.lineno) < (best.end_lineno - best.lineno):
                best = n
    return best


def _local_bound_names(fn) -> set:
    """函数作用域里被**局部绑定**的名字（Python 作用域规则：整个函数内 local）。

    嵌套函数/类只登记名字本身，体里的绑定不影响外层作用域。
    """
    bound = set()

    def walk(n):
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(c.name)
                continue
            if isinstance(c, ast.Name) and isinstance(c.ctx, (ast.Store, ast.Del)):
                bound.add(c.id)
            elif isinstance(c, ast.arg):
                bound.add(c.arg)
            elif isinstance(c, (ast.Import, ast.ImportFrom)):
                for a in c.names:
                    bound.add((a.asname or a.name).split(".")[0])
            elif isinstance(c, ast.ExceptHandler) and c.name:
                bound.add(c.name)
            elif isinstance(c, (ast.Global, ast.Nonlocal)):
                bound.update(c.names)
            walk(c)

    walk(fn)
    return bound


def make_diff(an: TestFileAnalyzer) -> tuple[str | None, str | None]:
    """对单个测试文件生成 unified diff 提案。

    返回 (diff, error)。error 非 None 表示改写后语法/字面量校验失败，diff 不可信。
    """
    if not an.bindings:
        return None, None
    # 精确归属：断言在 _add_assert 里已通过 prov.site 绑到读文本站点
    site_asserts = defaultdict(list)
    orphan = []
    for a in an.assertions:
        if a.site is not None:
            site_asserts[id(a.site)].append(a)
        elif a.cls != "C5" and a.in_assert and a.module:
            orphan.append(a)   # 有主线目标但找不到读站点 —— 需人工
    an.orphan_assertions = orphan

    new_lines = list(an.lines)
    edits = []      # (lineno, col0, lineno, col1, text)
    used_nodes = set()
    edited_modules = set()
    for site in an.bindings:
        if site.dynamic or site.module is None:
            continue
        asserts = site_asserts.get(id(site), [])
        tgt_rel = binding_target(site, asserts)
        site.target = tgt_rel
        expr = target_expr(site.module, tgt_rel, an.imports)
        node = None
        if site.node_ref is not None:
            node = site.node_ref
        else:
            node = _find_read_call_node(an, site, used_nodes)
        if node is None:
            continue
        used_nodes.add(id(node))
        edited_modules.add(site.module)
        recv = node.func.value
        # 替换整个 X.read_text(...) 调用（含 receiver）。
        # 括号修正：receiver 常写成 `(REPO / "x.py")` —— 若从 BinOp 内部起替换，
        # 分组左括号会悬空（右括号在 span 内被吃掉）。所以凡 recv 前一字符是 "("，
        # 就把该括号一并纳入 span，替换文本自带配对括号。
        c0, l0 = recv.col_offset, recv.lineno - 1
        # 统一包 textwrap.dedent：收窄到类方法的 getsource 是带缩进的，
        # 11 处喂 ast.parse 的用点里收窄到顶层函数的不缩进能解析、方法的才炸
        # （影子树只炸 1 处是运气不是正确）。模块级 getsource 每行缩进不齐，
        # dedent 是无操作，无害。
        repl = f"textwrap.dedent(inspect.getsource({expr}))"
        # 括号修正：分组左括号可能在 recv **前一行**（`src = (　\n　Path...)`）。
        # 从 recv 起点向前跳过空白找 "("；找到就把它纳入 span，替换文本自带配对。
        # 只吃第一个非空白字符是 "(" 的位置——更远的括号与本表达式无关。
        ll, cc = l0, c0
        while True:
            if cc > 0:
                ch = new_lines[ll][cc - 1]
            elif ll > 0:
                ll -= 1
                cc = len(new_lines[ll].rstrip("\r\n"))
                continue
            else:
                break
            if ch == "(":
                c0, l0 = cc - 1, ll
                repl = f"({repl})"
                break
            if not ch.isspace():
                break
            cc -= 1
        # ── 作用域碰撞检查（影子树实测三类残留之二）──
        # 宿主函数若**局部绑定**了替换表达式引用的名字（inspect / textwrap /
        # 模块绑定名），这个名字在整个函数作用域都是 local —— 模块级补 import
        # 救不了 UnboundLocalError（barostat:572 局部 import inspect 在用点之后、
        # platform_routing:473 `runabfe = ...getsource(runabfe)` 自引用、
        # stage2_retire:178 `pre` 同形状）。处置：语句前插入**别名** import 行，
        # 表达式全走别名 —— 与局部绑定互不干扰，原局部 import 一行不动。
        bind_name = an.imports.get(site.module, site.module)
        host = _host_func_of(an, node.lineno)
        if host is not None:
            colliding = ({bind_name, "inspect", "textwrap"}
                         & _local_bound_names(host))
        else:
            colliding = set()
        if colliding:
            alias = {nm: f"_a1_{nm}" for nm in sorted(colliding)}
            expr2 = expr
            if bind_name in alias:
                expr2 = re.sub(rf"\b{re.escape(bind_name)}\b", alias[bind_name], expr2)
            repl = f"textwrap.dedent({alias.get('inspect', 'inspect')}.getsource({expr2}))"
            if "textwrap" in alias:
                repl = repl.replace("textwrap.dedent", f"{alias['textwrap']}.dedent", 1)
            indent = re.match(r"[ \t]*", new_lines[l0]).group(0)
            ins = "".join(
                f"{indent}import {site.module if nm == bind_name else nm} as {alias[nm]}\n"
                for nm in sorted(colliding))
            edits.append((l0, 0, l0, 0, ins))
        edits.append((l0, c0,
                      node.end_lineno - 1, node.end_col_offset,
                      repl))

    if not edits:
        return None, None

    edits.sort(key=lambda e: (e[0], e[1]), reverse=True)
    for (l0, c0, l1, c1, text) in edits:
        if l0 == l1:
            line = new_lines[l0]
            new_lines[l0] = line[:c0] + text + line[c1:]
        else:
            first = new_lines[l0]
            last = new_lines[l1]
            merged = first[:c0] + text + last[c1:]
            new_lines[l0:l1 + 1] = [merged]
    # 元素内嵌换行（多行 import 插入）拆回逐行 —— 否则 unified diff 的 +/- 行
    # 内部藏 \n，hunk 畸形，patch 拒收。
    new_lines = [seg for ln in new_lines for seg in re.split(r"(?<=\n)", ln) if seg]

    # 补 import inspect / import 模块（只为提案里真正引用到的模块补 ——
    # import 主线模块很慢且有副作用，不因「扫描到」就加）
    new_lines = _ensure_imports(new_lines, an, edited_modules)

    # ── 校验：改写后必须 (a) 语法成立 (b) assert 里的字符串字面量逐字节不变 ──
    new_src = "".join(new_lines)
    err = _validate_rewrite(an, new_src)
    if err:
        return None, err

    diff = difflib.unified_diff(
        an.lines, new_lines,
        fromfile=f"a/{an.rel}", tofile=f"b/{an.rel}", n=3, lineterm="\n")
    return "".join(diff), None


def _assert_string_constants(src: str) -> Counter:
    """assert 语句里出现的全部字符串常量（多重集）—— 用于守卫字面量逐字节校验。"""
    out = Counter()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    out[sub.value] += 1
    return out


def _validate_rewrite(an: TestFileAnalyzer, new_src: str) -> str | None:
    try:
        before = _assert_string_constants(an.raw)
        after = _assert_string_constants(new_src)
    except SyntaxError as e:
        return f"改写后语法错误: {e}"
    if before != after:
        lost = before - after
        gained = after - before
        return (f"assert 字面量发生变化! 丢失 {dict(lost)} 新增 {dict(gained)} "
                f"—— 违反 3.4「断言字面量一字不改」，此 diff 作废")
    return None


def _find_read_call_node(an: TestFileAnalyzer, site: BindingSite, used: set):
    """按行号找回 read_text Call 节点；同一行多个读时按已用集合去重。"""
    cands = [n for n in ast.walk(an.tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "read_text" and n.lineno == site.line
             and id(n) not in used]
    return cands[0] if cands else None


def _ensure_imports(lines: list[str], an: TestFileAnalyzer,
                    edited_modules: set) -> list[str]:
    need = []
    # 只认**模块级** import（tree.body 顶层）。旧写法 `^\s*import inspect` 连函数
    # 局部的都算已导入 —— 单个函数里有局部 import inspect 时，其余用点全部
    # NameError（影子树实跑 38 处失败的根因）。
    def _top_imports():
        for node in an.tree.body:
            if isinstance(node, ast.Import):
                for a in node.names:
                    yield a.name.split(".")[0]
    top_names = set(_top_imports())
    if "inspect" not in top_names:
        need.append("import inspect")
    if "textwrap" not in top_names:
        need.append("import textwrap")
    mods_needed = {m for m in edited_modules if m and m not in an.imports}
    for m in sorted(mods_needed):
        need.append(f"import {m}")
    if not need:
        return lines
    # 插到**最外层**最后一条 import 之后。只认 tree.body 顶层的 import ——
    # 函数局部 import 和 `if __name__` 块里的 import 不是插入点（插进去会
    # 撕裂缩进块，test_fk_relearn_and_reachability 踩过）。没有顶层 import
    # 就插到模块 docstring 之后（保住 __doc__ 语义）。
    last_import = 0
    for node in an.tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import = max(last_import, node.end_lineno)
        elif (last_import == 0 and isinstance(node, ast.Expr)
              and isinstance(node.value, ast.Constant)
              and isinstance(node.value.value, str)):
            last_import = node.end_lineno
    out = list(lines)
    out[last_import:last_import] = [l + "\n" for l in need]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 汇总 / INDEX
# ─────────────────────────────────────────────────────────────────────────────

CLS_ORDER = ["C1", "C2", "C3", "C4", "C5"]
ST_ORDER = ["auto", "ambiguous", "module_level", "not_found",
            "c1_no_target", "c1_exposed", "unresolved", "c5"]


def compute_stats(analyzers):
    cls_counter = Counter()
    st_counter = Counter()
    per_file = {}
    n_pos_uses = 0
    for name, an in sorted(analyzers.items()):
        row = Counter()
        for a in an.assertions:
            if not a.in_assert:
                if a.cls == "C5" and a.op in POS_OPS:
                    n_pos_uses += 1  # 非 assert 的定位辅助使用，单独计数
                continue
            cls_counter[a.cls] += 1
            st_counter[assertion_status(a)] += 1
            row[a.cls] += 1
            row[assertion_status(a)] += 1
        if row:
            per_file[name] = row
    return cls_counter, st_counter, per_file, n_pos_uses


def fmt_status_counts(st_counter) -> str:
    parts = []
    for s in ST_ORDER:
        if st_counter.get(s):
            parts.append(f"{s}={st_counter[s]}")
    return " ".join(parts) or "(none)"


def write_index(analyzers, path: Path, failures: dict | None = None):
    cls_counter, st_counter, per_file, n_pos = compute_stats(analyzers)
    failures = failures or {}
    lines = []
    ap = lines.append
    ap("# A1 测试守卫解耦 —— 提案索引")
    ap("")
    ap("由 `refactor/tools/a1_tool.py` 生成（纯 AST，不 import 主线模块）。")
    ap("每个 `*.diff` 是**提案**：把「读主线 .py 文本」换成 `inspect.getsource(对象)`，")
    ap("**断言字面量一字不改**。落地前逐条人工过（HOW_TO_CONTINUE 3.4）。")
    ap("")
    ap("## 总计")
    ap("")
    ap(f"- 扫描测试文件：{len(analyzers)} 个")
    ap(f"- 断言处（assert 内，C1–C5）：{sum(cls_counter.values())}")
    for c in CLS_ORDER:
        ap(f"  - {c}: {cls_counter.get(c, 0)}")
    ap(f"- 另有 {n_pos} 处非 assert 的 .index/.find/.count 定位辅助使用（已参与收窄安全性判断）")
    ap(f"- 状态分布：{fmt_status_counts(st_counter)}")
    nf = st_counter.get("not_found", 0)
    c1e = st_counter.get("c1_exposed", 0)
    amb = st_counter.get("ambiguous", 0)
    ap(f"- **not_found（守卫钉错地方/已腐烂）: {nf}**")
    ap(f"- **c1_exposed（C1 守卫靠扫全文件才成立/命中注释字符串）: {c1e}**")
    ap(f"- ambiguous: {amb}")
    ap("")
    ap("## 分类说明")
    ap("")
    ap("| 状态 | 含义 |")
    ap("|---|---|")
    ap("| auto | 字面量唯一命中一个顶层定义，提案直接收窄到 `inspect.getsource(模块.类.方法)` |")
    ap("| ambiguous | 命中多个定义，收窄会改变守卫强度；diff 保底绑模块对象，候选见明细 |")
    ap("| module_level | 字面量在模块顶层（import 区/模块级语句），提案 `inspect.getsource(模块)` |")
    ap("| not_found | 字面量不在目标主线文件里 —— 守卫钉错地方或测试已腐烂，**必须逐条报告** |")
    ap("| c1_no_target | C1（not in 全文件）且字面量当前不存在 = 守卫生效中；收窄无目标，保底绑模块 |")
    ap("| c1_exposed | C1 但字面量在文件中出现 —— 守卫靠「扫全文件」的语义与该出现并存，单独标记 |")
    ap("")
    ap("## 汇总表（仅列出有断言的测试文件）")
    ap("")
    ap("| 测试文件 | 断言数 | C1 | C2 | C3 | C4 | C5 | auto | ambiguous | module_level | not_found | c1_no_target | c1_exposed |")
    ap("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, row in sorted(per_file.items()):
        ap(f"| {name} | {sum(row[c] for c in CLS_ORDER)} | "
           + " | ".join(str(row.get(c, 0)) for c in CLS_ORDER) + " | "
           + " | ".join(str(row.get(s, 0)) for s in
                        ["auto", "ambiguous", "module_level", "not_found",
                         "c1_no_target", "c1_exposed"]) + " |")
    ap("")

    # 明细：not_found
    ap("## not_found 明细（守卫钉错地方 / 测试已腐烂 —— 重点）")
    ap("")
    any_nf = False
    for name, an in sorted(analyzers.items()):
        for a in an.assertions:
            if assertion_status(a) == "not_found":
                any_nf = True
                ap(f"- `{a.file}:{a.line}` — C{a.cls[1]}（{a.op}）字面量 "
                   f"`{a.literals[0]!r}` 目标 `{a.module or a.path}` 未命中任何定义。{a.note}")
    if not any_nf:
        ap("（无）")
    ap("")

    # 明细：c1_exposed
    ap("## C1 收窄暴露明细（字面量在目标文件中出现 —— 守卫语义存疑）")
    ap("")
    any_c1 = False
    for name, an in sorted(analyzers.items()):
        for a in an.assertions:
            if assertion_status(a) == "c1_exposed":
                any_c1 = True
                hits = "; ".join(f"{a.module}.{h}" for h in a.hits[:4])
                ap(f"- `{a.file}:{a.line}` — `assert {a.literals[0]!r} not in {a.var}`，"
                   f"但该字面量出现在 `{a.module}` 的：{hits}。"
                   f"整文件守卫此刻要么在失败、要么命中的是注释/字符串——需要人工裁决。")
    if not any_c1:
        ap("（无）")
    ap("")

    # 明细：ambiguous
    ap("## ambiguous 明细（字面量命中多个定义）")
    ap("")
    any_amb = False
    for name, an in sorted(analyzers.items()):
        for a in an.assertions:
            if assertion_status(a) == "ambiguous":
                any_amb = True
                cands = ", ".join(a.hits[:5]) + ("…" if len(a.hits) > 5 else "")
                ap(f"- `{a.file}:{a.line}` — {a.cls}（{a.op}）字面量 `{_trunc(a.literals[0])!r}` "
                   f"在 `{a.module}` 命中 {len(a.hits)} 处：{cands}")
    if not any_amb:
        ap("（无）")
    ap("")

    # 明细：c1_no_target
    ap("## C1 无目标守卫（字面量当前全文件不存在 —— 守卫生效中，收窄需人工选目标）")
    ap("")
    n = 0
    for name, an in sorted(analyzers.items()):
        for a in an.assertions:
            if assertion_status(a) == "c1_no_target":
                n += 1
    ap(f"共 {n} 条。这类守卫的语义是「这个坏写法不许在任何地方回来」，")
    ap("收窄到单个函数会丢掉全文件覆盖面。提案 diff 里保底绑到**模块对象**")
    ap("（`inspect.getsource(模块)`，文本与现 read_text 等价），A2 再升包级 AST 扫描。")
    ap("")
    for name, an in sorted(analyzers.items()):
        for a in an.assertions:
            if assertion_status(a) == "c1_no_target":
                ap(f"- `{a.file}:{a.line}` — `assert {a.literals[0]!r} not in {a.var}`（目标 `{a.module}`）")
    ap("")

    # spec_from_file_location
    ap("## spec_from_file_location 站点")
    ap("")
    n = 0
    for name, an in sorted(analyzers.items()):
        for s in an.spec_loads:
            n += 1
            tgt = s.module or s.path or "（动态，静态解析不了）"
            ap(f"- `{s.file}:{s.line}` → `{tgt}`")
    if not n:
        ap("（无）")
    ap("")

    # 动态绑定
    ap("## 动态/未解析的读文本站点（diff 未覆盖，需人工）")
    ap("")
    n = 0
    for name, an in sorted(analyzers.items()):
        for s in an.bindings:
            if s.dynamic or (s.module is None and not s.spec_load and s.path is None):
                n += 1
                ap(f"- `{s.file}:{s.line}`（{s.var or '匿名'}）")
    if not n:
        ap("（无）")
    ap("")

    # 孤儿断言：有主线目标但静态找不到读文本站点
    ap("## 孤儿断言（断言目标解析到了主线模块，但静态追不到它的读文本站点 —— 需人工）")
    ap("")
    n = 0
    for name, an in sorted(analyzers.items()):
        for a in getattr(an, "orphan_assertions", []):
            n += 1
            ap(f"- `{a.file}:{a.line}` — {a.cls}（{a.op}）`{a.var}` → `{a.module}`")
    if not n:
        ap("（无）")
    ap("")

    # 校验失败（正常应为空）
    if failures:
        ap("## ⚠ 校验失败的 diff（已弃置，未写入提案 —— 工具缺陷信号）")
        ap("")
        for name, err in sorted(failures.items()):
            ap(f"- `{name}`: {err}")
        ap("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _trunc(s: str, n: int = 60) -> str:
    return s if len(s) <= n else s[:n] + "…"


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="只报统计，不写提案")
    args = ap.parse_args()

    # 重建父索引（按文件）
    analyzers = {}
    for p in sorted(TESTS.glob("test_*.py")):
        try:
            an = TestFileAnalyzer(p)
            analyzers[p.name] = an
        except SyntaxError as e:
            print(f"[warn] {p.name} 语法解析失败: {e}", file=sys.stderr)

    cls_counter, st_counter, per_file, n_pos = compute_stats(analyzers)

    print("== A1 扫描统计 ==")
    print(f"测试文件: {len(analyzers)}")
    total = sum(cls_counter.values())
    print(f"断言处(assert 内): {total}")
    for c in CLS_ORDER:
        print(f"  {c}: {cls_counter.get(c, 0)}")
    print(f"  (另有 {n_pos} 处非 assert 的 .index/.find/.count 定位辅助)")
    print(f"状态: {fmt_status_counts(st_counter)}")
    # 绑定站点
    n_sites = sum(len([s for s in an.bindings if not s.spec_load]) for an in analyzers.values())
    n_spec = sum(len(an.spec_loads) for an in analyzers.values())
    n_dyn = sum(len([s for s in an.bindings if s.dynamic and not s.spec_load])
                for an in analyzers.values())
    n_files = len({s.file for an in analyzers.values() for s in an.bindings})
    print(f"读文本绑定站点: {n_sites}（动态 {n_dyn}），spec_from_file_location: {n_spec}，涉及 {n_files} 个测试文件")

    if args.check:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_diff = 0
    failures = {}
    for name, an in sorted(analyzers.items()):
        d, err = make_diff(an)
        if err:
            failures[name] = err
            print(f"[validate] {name}: {err}", file=sys.stderr)
            continue
        if d:
            (OUT_DIR / f"{name}.diff").write_text(d if d.endswith("\n") else d + "\n",
                                                  encoding="utf-8")
            n_diff += 1
    write_index(analyzers, OUT_DIR / "INDEX.md", failures)
    print(f"提案: {n_diff} 个 diff + INDEX.md → {OUT_DIR}"
          + (f"（{len(failures)} 个校验失败已弃置）" if failures else ""))


if __name__ == "__main__":
    main()
