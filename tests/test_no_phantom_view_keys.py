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

pytestmark = pytest.mark.cpu_only

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


# ─────────────────────────────────────────────────────────────────────────────
# 🔑🔑 [2026-09-17] **同一个 bug 类的第二层：`sampling_units`（子窗记录）。**
#
# 上面那套只盯 view **顶层**（`RECEIVERS = view/base_view/sub_view/_view`）。
# 而子窗记录是 `_read_window()` 返回值的**手抄子集**，抄漏了同样不报错、同样
# 静默退化成「那个条件永远不成立」。实测一次抄漏 6 个键、3 个消费者中招：
#   · `self_min_frames`            → 子窗归因里「样本量硬证据」恒不成立
#   · `self_n_eff_over_g_eligible` → 补帧准入第三道闸对子窗恒放行
#   · `bottleneck_*` ×4            → D3 终态诊断的四个量恒 None
# 纯静态对撞，不需要 fixture（fixture 也会照着读方的想象造）。
# ─────────────────────────────────────────────────────────────────────────────
_UNIT_RECEIVERS = {"_u", "_su"}


def _sampling_unit_keys_written(tree):
    out = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "sampling_units"
                and node.args and isinstance(node.args[0], ast.Dict)):
            out |= {k.value for k in node.args[0].keys
                    if isinstance(k, ast.Constant)}
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "_u"
                        and isinstance(tgt.slice, ast.Constant)):
                    out.add(tgt.slice.value)
    return out


def _sampling_unit_keys_read(tree):
    out = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in _UNIT_RECEIVERS and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            out.setdefault(node.args[0].value, []).append(node.lineno)
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id in _UNIT_RECEIVERS
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
                and isinstance(node.ctx, ast.Load)):
            out.setdefault(node.slice.value, []).append(node.lineno)
    return out


def test_sampling_unit_records_have_every_key_someone_reads():
    path = os.path.join(ROOT, "abfe_preoptimizer.py")
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    written = _sampling_unit_keys_written(tree)
    assert written, "没找到 sampling_units 的构造点 —— 本测试失效了，先修测试"
    read = _sampling_unit_keys_read(tree)
    phantom = {k: sorted(set(v)) for k, v in read.items() if k not in written}
    assert not phantom, (
        "子窗记录上读了没人写的键 ⟹ 恒 None ⟹ 对应判据静默恒不成立：\n"
        + "\n".join(f"  {k} 读于 abfe_preoptimizer.py:{v}"
                    for k, v in sorted(phantom.items()))
    )


# ---------------------------------------------------------------------------
# [2026-09-17] 两个视图构造器的契约不许分叉
# ---------------------------------------------------------------------------

def test_every_view_key_the_decider_reads_exists_in_both_builders(tmp_path):
    """`_decide_once` 读的每一个 view 键，**两个视图构造器都得产出**。

    `read()` 会分叉：`for_physical_stage`（生产走这条）→ `read_aggregated()`；
    直接构造 → `_read_single_stage()`。先前 `min_n_eff_over_g_history` 与
    `stale_layout_evidence` **只在聚合视图里产出**，而 `_decide_once` 照读不误：

      · 历史缺 ⟹ `_hist = {}` ⟹ **加帧刹车恒不触发**；
      · stale 缺 ⟹ 分支 0a 与 `_evidence_status` 的「证据被布局作废」守卫
        **恒放行** ⟹ 可以带着过期证据判 DONE。

    两条都是**静默**失效：不报错、不留痕，判据只是悄悄不生效。
    生产没踩到纯属走运（它一直用聚合视图），测试和离线回放全在踩。
    """
    import inspect
    import re
    import abfe_preoptimizer as pre

    C = pre.Stage2RepairController
    src = inspect.getsource(C._decide_once)
    read_keys = (set(re.findall(r'view\.get\("([a-z_0-9]+)"', src))
                 | set(re.findall(r'view\["([a-z_0-9]+)"\]', src)))
    assert read_keys, "探针没抓到任何 view 键 —— 正则失效了，这条会静默假绿"

    from test_stage2_repair_controller import _mkrun, R4
    run = _mkrun(tmp_path, windows={i: {"K": 4} for i in range(4)},
                 ranges=R4, n_states=13)
    single = set(C(run, "vanishing").read())
    agg = set(C.for_physical_stage(run, "vanishing").read())

    for label, have in (("单段视图 `_read_single_stage`", single),
                        ("聚合视图 `read_aggregated`", agg)):
        missing = sorted(k for k in read_keys if k not in have)
        assert not missing, (
            f"{label} 缺这些 `_decide_once` 会读的键：{missing}。"
            "缺键不会报错 —— 对应的判据只是**静默失效**。")


# 🔑🔑 [2026-09-17] **正则探针换成明确的键集契约。**
#
# 老版本靠 `re.search(r'view\.get\("<key>"')` 判「这个键有没有被读来做判断」，
# 三处都漏了：
#   · `aggregated_segments` 被写进**豁免名单** —— 而执行器的 `_disk_signature`
#     确实读它（少一维 ⟹ 停滞签名认不出"建了新子系综"）；
#   · `out_of_range_windows` 是通过**动态键** `(view or {}).get(k)` 读的
#     （`_coverage_incomplete` 的 `for k in (...)` 循环），正则抓不到；
#   · `unverifiable_layout_evidence` / `window_provenance` 当时只用于审计/导出。
# 契约改成最强也最简单的那一条：**两个构造器必须产出同一组键。**
# 差一个键就红，不用再判「它是不是报告性的」——那个判断本身就是漏网的来源。
VIEW_SCHEMA_KEYS_BOTH_BUILDERS_MUST_HAVE = {
    "out_of_range_windows", "unverifiable_layout_evidence",
    "stale_layout_evidence", "window_provenance", "aggregated_segments",
    "min_n_eff_over_g_history", "path_version", "sampling_units",
    "skipped_windows", "skipped_sampling_units",
}


def _run_with_real_lambdas(tmp_path, *, windows, ranges, n_states, **kw):
    """造一个**布局自洽**的 run：路径有真 λ，且每个窗口的 λ = 它自己那段切片。

    两个坑，缺一个这套测试就测不到东西：
      · `_mkrun` 的 `states` 只有 `id`、**没有 `lambda_vdw`** ⟹ `_read_path()` 的
        `lambdas_vdw` 恒 None ⟹ 布局身份核对整套是 inert 的。现有每个用例都如此 ——
        这正是本文件这一类缺陷第二层一直没被拦住的原因。
      · `_mkrun` 给**每个**窗口发同一份默认 λ `[1.0, 0.9, 0.8, 0.7]`，与路径对不上
        ⟹ 补上路径 λ 之后，所有窗口一律被判 STALE。拿一个布局本身就不自洽的盘面
        去比两个构造器，比的是混淆项，不是它们的差异。
    """
    import json
    from test_stage2_repair_controller import _mkrun
    lam = [round(1.0 - i / (n_states - 1), 6) for i in range(n_states)]
    wins = {}
    for i, w in windows.items():
        a, b = ranges[i]
        wins[i] = dict(w, lambdas_vdw=lam[a:b])
    run = _mkrun(tmp_path, windows=wins, ranges=ranges, n_states=n_states, **kw)
    v1 = os.path.join(run, "checkpoints", "path_versions", "v1.json")
    with open(v1, encoding="utf-8") as fh:
        rec = json.load(fh)
    rec["states"] = [{"id": f"s{i}", "lambda_vdw": lam[i]} for i in range(n_states)]
    with open(v1, "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    return run


def test_the_two_builders_produce_the_same_view_keys(tmp_path):
    """**两个视图构造器的键集必须逐字相同。**

    只有聚合视图产出的键，在单段视图上恒为 None ⟹ 读它的判据**静默失效**。
    实测漏了四个：`out_of_range_windows`（`_coverage_incomplete` 的越界闸）、
    `aggregated_segments`（执行器停滞签名）、`unverifiable_layout_evidence`、
    `window_provenance`。
    """
    import abfe_preoptimizer as pre
    C = pre.Stage2RepairController
    from test_stage2_repair_controller import R4
    run = _run_with_real_lambdas(
        tmp_path, windows={i: {"K": 4} for i in range(4)}, ranges=R4, n_states=13)
    agg = set(C.for_physical_stage(run, "vanishing").read())
    single = set(C(run, "vanishing").read())
    assert not (agg - single), (
        f"这些键只有聚合视图产出：{sorted(agg - single)}。"
        "单段视图（直接构造器 / decision_trace / 非 --replay 的 CLI）下它们恒为 "
        "None ⟹ 对应判据静默失效。补的必须是**等价语义**，不是空默认值。")
    assert not (single - agg), (
        f"这些键只有单段视图产出：{sorted(single - agg)}")
    for k in VIEW_SCHEMA_KEYS_BOTH_BUILDERS_MUST_HAVE:
        assert k in agg and k in single, f"契约键 {k} 不在视图里"


def test_single_stage_view_catches_an_out_of_range_window(tmp_path):
    """盘上多出一个当前布局里没有的窗口号 ⟹ 单段视图必须报出来、覆盖判缺口。

    `missing_windows` 只算 `range(expected)`、**从不查多** ⟹ 没有这一条，
    一份含越界窗口的证据可以被当成"当前路径已完成"。
    """
    import json
    import abfe_preoptimizer as pre
    C = pre.Stage2RepairController
    from test_stage2_repair_controller import R4
    run = _run_with_real_lambdas(
        tmp_path, windows={i: {"K": 4} for i in range(4)}, ranges=R4, n_states=13)
    # 旧布局遗留的产物：布局里只有 0..3
    with open(os.path.join(run, "vanishing",
                           "dual_window_99_vdw_convergence.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"window_idx": 99, "lambdas_vdw": [0.9, 0.8, 0.7, 0.6],
                   "cumulative_production_steps": 250000}, fh)

    view = C(run, "vanishing").read()
    assert view["out_of_range_windows"] == [99], view["out_of_range_windows"]
    assert C._coverage_incomplete(view), (
        "越界窗口没让 `_coverage_incomplete()` 触发 ⟹ 可以带着它判 DONE")


def test_single_stage_view_reports_unverifiable_layout_evidence(tmp_path):
    """证据自己没带 λ 列表 ⟹ 核不了布局，必须如实记进 `unverifiable_layout_evidence`。

    与 `stale`（核对过、不匹配）是两件事：这条是"核不了"。合并照旧，但必须说出来。
    """
    import json
    import abfe_preoptimizer as pre
    C = pre.Stage2RepairController
    from test_stage2_repair_controller import R4
    run = _run_with_real_lambdas(
        tmp_path, windows={i: {"K": 4} for i in range(4)}, ranges=R4, n_states=13)
    for name in ("vanishing/dual_window_2_vdw_convergence.json",
                 "checkpoints/ibs_state_vdw_window_2.json"):
        path = os.path.join(run, *name.split("/"))
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        d.pop("lambdas_vdw", None)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh)

    view = C(run, "vanishing").read()
    assert 2 in view["unverifiable_layout_evidence"], (
        f"窗口 2 缺 λ 身份却没进 unverifiable：{view['unverifiable_layout_evidence']}")
    assert 2 not in view["stale_layout_evidence"], "「核不了」不得被当成「已过期」"


def test_one_segment_run_agrees_between_direct_and_aggregated_views(tmp_path):
    """只有一段的 run：两个构造器的关键字段与 `decide()` 结果必须一致。"""
    import abfe_preoptimizer as pre
    C = pre.Stage2RepairController
    from test_stage2_repair_controller import R4
    run = _run_with_real_lambdas(
        tmp_path, windows={i: {"K": 4} for i in range(4)}, ranges=R4, n_states=13)
    v_agg = C.for_physical_stage(run, "vanishing").read()
    v_one = C(run, "vanishing").read()
    for k in ("out_of_range_windows", "unverifiable_layout_evidence",
              "stale_layout_evidence", "skipped_windows",
              "missing_windows", "path_version"):
        assert v_agg[k] == v_one[k], f"{k}: 聚合={v_agg[k]!r} 单段={v_one[k]!r}"
    p_agg = C.for_physical_stage(run, "vanishing").decide(v_agg)
    p_one = C(run, "vanishing").decide(v_one)
    assert (p_agg["action"], p_agg["exit"]) == (p_one["action"], p_one["exit"]), (
        f"同一个单段 run，两个构造器给出不同决策："
        f"聚合={p_agg['action']}/{p_agg['exit']} 单段={p_one['action']}/{p_one['exit']}")


def _retired_regex_probe_replaced_by_the_schema_contract(tmp_path):

    """聚合视图独有的键，只许是**报告性**的（类内无人拿 `view.get` 读它做判断）。"""
    import inspect
    import re
    import abfe_preoptimizer as pre

    C = pre.Stage2RepairController
    from test_stage2_repair_controller import _mkrun, R4
    run = _mkrun(tmp_path, windows={i: {"K": 4} for i in range(4)},
                 ranges=R4, n_states=13)
    only_agg = set(C.for_physical_stage(run, "vanishing").read()) - set(
        C(run, "vanishing").read())

    src = inspect.getsource(C)
    # 允许名单：纯报告用途、且读侧有 `or []` / `or {}` 兜底
    ALLOWED = {"aggregated_segments"}
    bad = [k for k in sorted(only_agg)
           if k not in ALLOWED
           and re.search(rf'view\.get\("{k}"|view\["{k}"\]', src)]
    assert not bad, (
        f"这些键只有聚合视图产出、却被类内代码读来做判断：{bad}。"
        "单段视图下它们恒为 None ⟹ 判据静默失效。要么两边都产出，"
        "要么确认是纯报告并加进 ALLOWED。")
