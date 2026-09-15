# -*- coding: utf-8 -*-
"""控制器 view 上**不许读一个没人写的键**（2026-09-14，一天之内同形状 4 次）。

  · `w["lambda_vdw_hi"]`        → anchor 恒 None，`SPLIT_TAIL_WINDOW` 从没执行过
  · `record["kind"]`（应在 event 里）→ 插 λ 的跨 resume 预算从没生效
  · `view["path_version"]`      → no-op 记录不会因布局变化失效
  · `w["segment"]`              → no-op 指纹里"换段"这一维被悄悄关掉
  · `window_record["warmup_budget_ledger"]` / `["validation_attempt_budget_steps"]`
        → 「开一个新 Epoch 要多少步」永远返回兜底常量 140000，真实需求 290000

共同点：**读方和写方对不上，两边都不报错，行为静默退化成「那个条件永远不成立」**。
单元测试抓不到（fixture 也照着读方的想象造），只有把「真 view 有哪些键」和
「源码从 view 上读哪些键」对撞才看得见。

⚠️ 必须用 `for_physical_stage()` 造控制器 —— 那是**自治循环真正用的那条构造路径**
（会聚合采样段）。用普通构造函数会拿到一个更小的 view，凭空多出三个"幽灵键"。
"""
import ast
import os

import pytest

from abfe_preoptimizer import Stage2RepairController

from test_stage2_repair_controller import _mkrun

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = ("abfe_preoptimizer.py", "abfe_pipeline.py")
# 源码里指向控制器 view 的变量名。
RECEIVERS = {"view", "base_view", "sub_view", "_view"}


def _keys_read_from_view():
    out = []
    for name in SOURCES:
        path = os.path.join(ROOT, name)
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in RECEIVERS and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                out.append((node.args[0].value, name, node.lineno))
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id in RECEIVERS
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)
                    and isinstance(node.ctx, ast.Load)):
                out.append((node.slice.value, name, node.lineno))
    return out


@pytest.fixture
def real_view(tmp_path):
    ranges = [(0, 8), (7, 13), (12, 16), (15, 27)]
    run = _mkrun(tmp_path,
                 windows={0: {"K": 8}, 1: {"K": 6}, 2: {"K": 4},
                          3: {"K": 12, "self_verdict": "HARD_INSUFFICIENT"}},
                 ranges=ranges, n_states=27)
    return Stage2RepairController.for_physical_stage(run, "vanishing", "vdw").read()


def test_every_key_read_from_the_view_actually_exists_on_it(real_view):
    reads = _keys_read_from_view()
    assert reads, "扫描器没抓到任何 view 读取 —— 它自己坏了"
    missing = sorted({k for k, _, _ in reads} - set(real_view))
    where = {k: [f"{f}:{l}" for kk, f, l in reads if kk == k][:3] for k in missing}
    assert not missing, (
        "这些键在源码里被从 view 上读，真 view 里却没有 —— 读方拿到的永远是 None，"
        "对应的判断静默失效：\n"
        + "\n".join(f"  · {k}  ← {where[k]}" for k in missing)
    )


def test_the_scanner_would_catch_a_phantom_key(real_view):
    """扫描器自己要能抓到 —— 否则它只是一条永远绿的装饰。"""
    reads = _keys_read_from_view()
    assert ("path_version", "abfe_preoptimizer.py") in {(k, f) for k, f, _ in reads}, \
        "扫描器没看到 view.get('path_version') —— 覆盖面不对"
    assert "path_version" in real_view
    # 把它从 view 里拿掉，扫描器必须报出来。
    shrunk = {k: v for k, v in real_view.items() if k != "path_version"}
    assert "path_version" in ({k for k, _, _ in reads} - set(shrunk))


# --------------------------------------------------------------- 窗口记录
# 只认**明确**指向窗口记录的形参名。`w` / `rec` 这类通用名在源码里还指
# no-op 账本、`window_overlap_diagnostics` 条目等等，放进来全是噪声。
WINDOW_RECEIVERS = {"window_record", "wrec"}


def _recv_name(node):
    """接收者变量名。认 `x.get(...)` 也认 `(x or {}).get(...)`。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.BoolOp) and node.values and isinstance(node.values[0], ast.Name):
        return node.values[0].id
    return None


def _window_aliases(fn):
    """函数里指向 `window_record` 形参的所有名字（含 `w = window_record or {}`）。"""
    args = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    names = (args & WINDOW_RECEIVERS) or set()
    if not names:
        return set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and _recv_name(node.value) in names:
            names.add(node.targets[0].id)
    return names


def _keys_read_from_window_record():
    out = []
    for name in SOURCES:
        with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            recvs = _window_aliases(fn)
            if not recvs:
                continue
            for node in ast.walk(fn):
                recv = key = None
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "get" and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)):
                    recv, key = _recv_name(node.func.value), node.args[0].value
                elif (isinstance(node, ast.Subscript)
                      and isinstance(node.slice, ast.Constant)
                      and isinstance(node.slice.value, str)
                      and isinstance(node.ctx, ast.Load)):
                    recv, key = _recv_name(node.value), node.slice.value
                if recv in recvs:
                    out.append((key, name, node.lineno))
    return out


def test_every_key_read_from_a_window_record_actually_exists_on_it(real_view):
    reads = _keys_read_from_window_record()
    assert reads, "扫描器没抓到任何窗口记录读取 —— 它自己坏了"
    have = set()
    for w in real_view["windows"]:
        have |= set(w)
    missing = sorted({k for k, _, _ in reads} - have)
    where = {k: [f"{f}:{l}" for kk, f, l in reads if kk == k][:3] for k in missing}
    assert not missing, (
        "这些键在源码里被从窗口记录上读，真记录里却没有：\n"
        + "\n".join(f"  · {k}  ← {where[k]}" for k in missing)
    )


def test_relearn_epoch_budget_is_self_calibrated_not_a_constant(real_view):
    """真机 cyclod_ligand1/rep2：账本 learning=230000 ⟹ 需求 290000，不是兜底 140000。

    方向很重要：低估 ⟹ **批准一个跑不完的 Epoch**，烧光预算再半路死 ——
    正是 `relearn_epoch_required_steps` 那段 docstring 声称要防止的事。
    """
    from abfe_preoptimizer import relearn_epoch_required_steps

    w = dict(real_view["windows"][0])
    w["warmup_budget_ledger"] = {"learning_steps": 230000, "freeze_burn_in_steps": 10000}
    need = relearn_epoch_required_steps(w)
    assert need > 240000, f"账本摆在那里却仍然取兜底常量：{need}"
    # 没有账本时才退兜底 —— 而且兜底必须**更小**，好让"有账本"这一支可辨认。
    bare = relearn_epoch_required_steps({k: v for k, v in w.items()
                                         if k != "warmup_budget_ledger"})
    assert bare < need
