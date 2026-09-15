"""补帧的**准入与停止条件**。

真机可以一路 25→50→…→**150 万步**：
  · 早期分支（求解器跳窗 ⟹ 补帧）**走不到**后面的边际收益刹车；
  · 停滞保护把"多跑了一块"算成盘面进展，**重复计数清零**；
  · 生产 cap 缺省是 unknown，拦不住。

**「又产生了帧」不等于「获得了有用证据」。** 两条准入，缺一不可：
  ① 每窗口块数**硬上限**（config 可调，缺省 4 块）——生产 cap unknown 时的唯一防线；
  ② 上一块必须有**实质增益**，判据量是**求解器侧**的去相关帧数
     （不是自检那份，两者实测差 2–8 倍；也不是步数）。
"""
import json
import os
import pathlib
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abfe_preoptimizer import Stage2RepairController  # noqa: E402

from test_stage2_repair_controller import _mkrun, R4, FULL  # noqa: E402


def _run(tmp_path, *, blocks, cap=None, prod=250000):
    """`blocks` = 逐块的 (production_steps, solver_n_decorrelated)。"""
    cfg = {"stage2_window_min_states": 4, "stage2_window_max_states": 8,
           "max_path_insertions": 3}
    if cap is not None:
        cfg["stage2_max_production_blocks_per_window"] = cap
    w = dict(FULL)
    w[0] = {"K": 4, "self_verdict": "INSUFFICIENT_DATA",
            "self_verdict_source": "solver_eligibility", "prod": prod}
    run = _mkrun(tmp_path, windows=w, ranges=R4, n_states=13, config=cfg,
                 stage_result={"converged": False,
                               "min_decorrelated_samples_threshold": 20,
                               "window_overlap_diagnostics": [
                                   {"window_index": 0, "n_frames_decorrelated": 9}]})
    (pathlib.Path(run) / "checkpoints" / "stage2_autonomous_history.json").write_text(
        json.dumps({"iterations": [
            {"iteration": k + 1, "action": "RUN_PRODUCTION", "path_version": 1,
             "snapshot": [{"window_idx": 0, "segment": "vanishing",
                           "production_steps": st, "solver_n_decorrelated": nd}]}
            for k, (st, nd) in enumerate(blocks)]})
    )
    return run


def test_the_first_block_is_always_granted(tmp_path):
    plan = Stage2RepairController(_run(tmp_path, blocks=[]), "vanishing").decide()
    assert plan["action"] == "RUN_PRODUCTION", plan["reason"]


def test_growth_keeps_the_tap_open(tmp_path):
    """判据量在涨 ⟹ 继续批。"""
    run = _run(tmp_path, blocks=[(250000, 6), (500000, 9)])
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] == "RUN_PRODUCTION", plan["reason"]


def test_no_real_gain_stops_the_automatic_topping_up(tmp_path):
    """**要害**：多跑了一块但判据量没涨 ⟹ 停，并说清楚为什么。

    ⚠️ [审计 #30，2026-09-14] **`NO_FEASIBLE_ACTION` 仍然是终态，别改成路由。**
    #30 的诉求（「一个窗口满额就终止整跑，别的窗口一个都没试」）由 `plan()` 里
    **部分满额就剔掉满额的、只补剩下的**那段解决 —— 那才是真正的修复。
    走到本断言这条路时，本轮点名的窗口**一个都批不过**，那就是真的无路可走。
    做成路由会更糟：`NO_ACTION` 配非终态出口 ⟹ 主循环不 break ⟹ 执行器收到一个
    它不认识的动作，而盘面没变 ⟹ 下一轮返回同一个 plan，只能靠停滞保护收场。
    （**所有**窗口都满额时发的是更具体的 `HALT_FRAMES_ADMISSION_CAP`，同样是终态。）

    ⚠️ [审计 #47] 判据从「没涨」（`max(h[1:]) <= h[0]`）放宽成「末点明显低于
    前面各点的中位数」。9/9/8 仍然触发（基准 9，末点 8 < 9×0.9）。
    """
    run = _run(tmp_path, blocks=[(250000, 9), (500000, 9), (750000, 8)])
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] == "NO_ACTION", plan["reason"]
    # `NO_ACTION` 必须配终态出口，否则主循环不 break（词汇表不变量）
    assert plan["exit"] in ("NO_FEASIBLE_ACTION", "HALT_FRAMES_ADMISSION_CAP")
    assert plan["exit"] in Stage2RepairController.TERMINAL_EXITS
    assert plan["terminal"] is True
    assert "没有带来实质增益" in plan["reason"]


def test_the_hard_block_cap_stops_it_even_while_still_growing(tmp_path):
    """生产 cap 缺省 unknown ⟹ 块数硬上限是唯一防线，涨着也得停。"""
    run = _run(tmp_path, blocks=[(250000, 5), (500000, 8), (750000, 12)], cap=3)
    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] == "NO_ACTION", plan["reason"]
    assert "已经批过 3 块" in plan["reason"] and "上限 3" in plan["reason"]


def test_the_cap_is_bounded_by_default_never_unlimited(tmp_path):
    c = Stage2RepairController(_run(tmp_path, blocks=[]), "vanishing")
    assert 1 <= c.max_blocks_per_window <= 8, (
        "缺省块数上限必须是**有界**的 —— 默认无限正是要修的那个行为"
    )


def test_blocks_from_another_segment_do_not_count(tmp_path):
    """换段 = 换 f_k，不是同一条曲线；旧段的块不该占当前段的额度。"""
    run = _run(tmp_path, blocks=[])
    (pathlib.Path(run) / "checkpoints" / "stage2_autonomous_history.json").write_text(
        json.dumps({"iterations": [
            {"iteration": k, "action": "RUN_PRODUCTION", "path_version": 1,
             "snapshot": [{"window_idx": 0, "segment": "vanishing_9",
                           "production_steps": 250000 * k,
                           "solver_n_decorrelated": 9}]}
            for k in range(1, 6)]})
    )
    v = Stage2RepairController(run, "vanishing").read()
    assert (v["production_blocks_by_window"] or {}).get(0) in (None, [])


def test_the_gate_lives_in_plan_not_scattered_across_branches():
    """准入门必须在 `plan()` 里挂**一次**，不许逐个分支挂。

    `decide()` 有 16 个发 `RUN_PRODUCTION` 的出口。逐个挂必然漏 —— 实测漏了 15 个，
    其中两处的改动还因为同一个脚本里后面的 assert 抛错、整份写入被中止而悄悄丢掉，
    而当时"测试通过"照样绿。
    """
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "abfe_preoptimizer.py").read_text("utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "decide")
    calls = [n.lineno for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "_frames_admission"]
    assert len(calls) == 1, f"准入门被挂在 {len(calls)} 处 —— 逐个挂必然漏：{calls}"

    plan_fn = next(n for n in ast.walk(fn)
                   if isinstance(n, ast.FunctionDef) and n.name == "plan")
    assert plan_fn.lineno <= calls[0] <= plan_fn.end_lineno, (
        "唯一那处不在 `plan()` 里 —— 那就还是逐个分支挂"
    )


def test_no_exit_contradicts_its_action():
    """`action=DONE` 配非 DONE 出口 ⟹ `execution_status` 会算成 COMPLETE。

    一个"无路可走"的结局被记成"执行完毕" —— 09-13 修过一处，梳理时又抓到一处。
    """
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "abfe_preoptimizer.py").read_text("utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "decide")
    bad = []
    for n in ast.walk(fn):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "plan" and n.args):
            continue
        if not isinstance(n.args[0], ast.Constant):
            continue
        act = n.args[0].value
        ex = next((ast.unparse(k.value) for k in n.keywords if k.arg == "exit_"), None)
        if act == "DONE" and ex and "DONE" not in ex:
            bad.append((n.lineno, act, ex))
        if act == "NO_ACTION" and ex and not any(
                t in ex for t in ("NO_FEASIBLE_ACTION", "GLOBAL_BUDGET_EXHAUSTED")):
            bad.append((n.lineno, act, ex))
    assert not bad, f"动作与出口自相矛盾：{bad}"
