#!/usr/bin/env python3
"""第一刀：抽 L0 infra（纯搬运）。HOW_TO_CONTINUE B2 / b3_edge_decisions §7。

只动 refactor/ 域文件：把 infra 名单的定义块**逐字**搬到新域 refactor/domains/infra/infra.py。
验收：所有域文件行多集合前后完全一致（sort|md5sum 同值）；逐文件 py_compile。
边界：logger（三份 __name__）与 _atomic_write_json（字节不同）不入 infra；
_file_sha256 两份各自原样下沉，不合并不改名。
"""
import ast
import collections
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent

PLAN = {
    "membrane/membrane_gates.py": ["_infer_log_level_from_message", "_log_print",
                                   "_ONCE_EMITTED_KEYS", "print_once"],
    "mbar_analysis/ibs_data_loading_head.py": ["_infer_log_level_from_message", "_log_print",
                                               "_ONCE_EMITTED_KEYS", "print_once",
                                               "_atomic_save_npy", "_sha256_file", "_timed"],
    "pipeline_orchestration/pipeline_head.py": ["_infer_log_level_from_message", "_log_print",
                                                "_file_sha256", "NumpyEncoder"],
    "charging/apbs_correction.py": ["_file_sha256"],
    "pipeline_orchestration/runabfe_head.py": ["log", "_sha256_file"],
    "system_build/units_validators.py": ["NumpyEncoder"],
}

# infra.py 内的块顺序（决定重名副本哪个生效）：ibs 的 print_once 排在 membrane
# 之后（B2：留 ibs_engine 那份 docstring）；abfe_core 的 NumpyEncoder（units_
# validators 里的「全局统一」版）排最后生效。
ORDER = ["membrane/membrane_gates.py", "mbar_analysis/ibs_data_loading_head.py",
         "charging/apbs_correction.py", "pipeline_orchestration/pipeline_head.py:_plain",
         "pipeline_orchestration/runabfe_head.py",
         "pipeline_orchestration/pipeline_head.py:_numpy",
         "system_build/units_validators.py"]


def _read(p):
    # newline=""：字节保真。域文件里有 CRLF 片段（源自主线 ibs_engine.py），
    # read_text 的通用换行会把 \r\n 洗成 \n —— 第一版踩过：Counter 相等而
    # md5 变了，纯搬运被整文件换行归一化破坏，作废重做。
    with open(p, encoding="utf-8", newline="") as fh:
        return fh.read()


def _write(p, text):
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def domain_lines():
    c = collections.Counter()
    for p in sorted(ROOT.rglob("*.py")):
        if p.parent == ROOT:      # 工具脚本不对账
            continue
        for ln in _read(p).splitlines(keepends=True):
            c[ln] += 1
    return c


def spans_for(rel, lines, tree):
    names = PLAN[rel]
    out = []
    for st in tree.body:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nm = st.name
        elif isinstance(st, ast.Assign):
            nm = next((t.id for t in st.targets if isinstance(t, ast.Name)), None)
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
            nm = st.target.id          # `_ONCE_EMITTED_KEYS: set = set()` 是 AnnAssign ——
            if st.value is None:       # 纯声明无值，不动
                continue
        else:
            continue
        if nm not in names:
            continue
        if nm == "log" and 'getLogger("runabfe")' not in lines[st.lineno - 1]:
            continue
        start = min([d.lineno for d in getattr(st, "decorator_list", [])] + [st.lineno])
        while start > 1 and lines[start - 2].lstrip().startswith("#"):
            start -= 1                       # 紧贴的注释行随块走
        end = st.end_lineno
        while end < len(lines) and not lines[end].strip():
            end += 1                         # 块尾空行随块走（保多集合不变）
        out.append((start, end, nm))
    out.sort()
    return out


def main():
    before = domain_lines()
    blocks = collections.defaultdict(list)   # rel -> [(start, end, nm, text)]
    for rel in PLAN:
        p = ROOT / rel
        lines = _read(p).splitlines(keepends=True)
        tree = ast.parse("".join(lines))
        for s, e, nm in spans_for(rel, lines, tree):
            blocks[rel].append((s, e, nm, "".join(lines[s - 1:e])))

    # 顺序展开
    seq = []
    for key in ORDER:
        rel, _, filt = key.partition(":")
        for s, e, nm, text in blocks[rel]:
            if filt == "_plain" and nm == "NumpyEncoder":
                continue
            if filt == "_numpy" and nm != "NumpyEncoder":
                continue
            seq.append((rel, s, e, nm, text))
    n_blocks = sum(len(v) for v in blocks.values())
    assert len(seq) == n_blocks, f"顺序覆盖不全 {len(seq)}/{n_blocks}"

    # 删源（只删块行）
    for rel in PLAN:
        p = ROOT / rel
        lines = _read(p).splitlines(keepends=True)
        kill = set()
        for s, e, nm, _ in blocks[rel]:
            kill.update(range(s, e + 1))
        new = [ln for i, ln in enumerate(lines, 1) if i not in kill]
        _write(p, "".join(new))

    (ROOT / "infra").mkdir(exist_ok=True)
    _write(ROOT / "infra" / "infra.py", "".join(t for *_, t in seq))

    after = domain_lines()
    if before != after:
        diff = (before - after) + (after - before)
        sys.exit(f"❌ 行多集合不一致！差异 {sum(diff.values())} 行：{list(diff)[:5]}")
    # 自检：源文件里不得残留任何 PLAN 名的顶层定义（v1 踩过：AnnAssign 被静默
    # 跳过，md5 照样对账——多集合不变掩盖了「没搬走」，靠 layer_check 的
    # infra→membrane 上升边才暴露。此检查把这类漏网堵在刀内）。
    for rel, names in PLAN.items():
        src = _read(ROOT / rel)
        tree = ast.parse(src)
        for st in tree.body:
            top = None
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                top = st.name
            elif isinstance(st, ast.Assign):
                top = next((t.id for t in st.targets if isinstance(t, ast.Name)), None)
            elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                top = st.target.id
            if top is None or top not in names:
                continue
            if top == "log" and 'getLogger("runabfe")' not in src.splitlines()[st.lineno - 1]:
                continue                    # 别处的 log=，不是要搬的那个
            sys.exit(f"❌ {rel} 仍残留顶层定义 {top}（行 {st.lineno}）")
    for rel in list(PLAN) + ["infra/infra.py"]:
        py_compile.compile(str(ROOT / rel), doraise=True)
    print(f"✅ 搬运 {len(seq)} 个定义块；行多集合一致；无残留定义；py_compile 全过")


if __name__ == "__main__":
    main()
