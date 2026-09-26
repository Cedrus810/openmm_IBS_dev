#!/usr/bin/env python3
"""只读探针 v2：复用 layer_check 的 build()，解释 1365 → 1357。
两个口径 × 三个控制器状态：
  A. 17 域口径（layer_check main 的文件集，scratchpad 排除）
  B. deps.py 噪音口径（scratchpad 计入为域）——再细分：
     B1 当前 scratchpad（8 留底 + 本探针）
     B2 旧测量时点 scratchpad（仅 step0_1..step0_7 七份留底）
不写任何域文件。
"""
import ast
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("lc", ROOT / "layer_check.py")
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)

all_py = sorted(p for p in ROOT.rglob("*.py"))
domain_files = [p for p in all_py
                if len(p.relative_to(ROOT).parts) > 1
                and "split_tool" not in str(p)
                and p.relative_to(ROOT).parts[0] != "scratchpad"]
scratch_now = sorted(p for p in all_py if p.relative_to(ROOT).parts[0] == "scratchpad")
scratch_old = [p for p in scratch_now
               if p.name.startswith("step0_") and p.name != "step0_8_bounded_rewindow.before.py"]

ctrl = next(p for p in domain_files if p.name == "stage2_repair_controller.py")
STATES = [
    ("原始(step0_1留底)", ROOT / "scratchpad" / "step0_1_pw_or_last.before.py"),
    ("刀6后(step0_7留底)", ROOT / "scratchpad" / "step0_7_plan.before.py"),
    ("当前(刀8后)", ctrl),
]

def total(files, ctrl_tree_override=None):
    trees = {p: ast.parse(p.read_text(encoding="utf-8")) for p in files}
    if ctrl_tree_override is not None:
        trees[ctrl] = ctrl_tree_override
    edges, _, ext = lc.build(files, trees)
    return sum(edges.values()), len(edges)

for label, path in STATES:
    override = None if path == ctrl else ast.parse(path.read_text(encoding="utf-8"))
    a = total(domain_files, override)
    b1 = total(domain_files + scratch_now, override)
    b2 = total(domain_files + scratch_old, override)
    print(f"控制器={label:16s} | A 17域: {a[0]:5d}/{a[1]:3d}边"
          f" | B1 当前scratchpad: {b1[0]:5d}/{b1[1]:3d}边"
          f" | B2 旧scratchpad(7留底): {b2[0]:5d}/{b2[1]:3d}边")
