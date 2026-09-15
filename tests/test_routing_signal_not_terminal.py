"""`LOCAL_VALIDATION_CAP` 是路由信号，不得炸出流水线。

真机 2026-09-14：窗口 4 的冻结验证在单周期预算内没求出 Δf−ΔF ⟹
`IBSValidationBudgetIndeterminateError` 从 `_run_stage2_with_path_evolution`
一路炸穿 `run_full_pipeline`。而那次调用只是**第一次执行**，顶层自治控制器
（对同一个异常有路由分支）就在它返回之后接管 —— 信号在控制器拿到方向盘之前
把整条管线打死了。

约定见 docs/TODO.md「不许回退的约定」：只有三个真终态，
`LOCAL_VALIDATION_CAP` / `INSUFFICIENT_DATA` / `CUMULATIVE_FK_MISALIGNMENT` /
`SKIPPED_WINDOW` 全是路由。
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ibs_engine as _ie  # noqa: E402
from abfe_pipeline import ABFEPipeline  # noqa: E402


def _call(tmp_path, *, route_to_controller, repair_policy="path_evolution_v1"):
    """跑一次路径演化闭环，`run_once` 第一下就抛路由信号。"""
    logged = []
    stub = types.SimpleNamespace(_log=logged.append)

    def run_once(*_a, **_kw):
        raise _ie.IBSValidationBudgetIndeterminateError(
            "窗口 4 的冻结验证在单周期 15/15 批内始终 insufficient_frames_after_"
            "decorrelation：Δf−ΔF 从未被求出，对这份 f_k **无结论**。",
            {"window_idx": 4, "verdict": "insufficient_frames_after_decorrelation"},
        )

    return ABFEPipeline._run_stage2_with_path_evolution(
        stub, run_once,
        [1.0, 0.75, 0.5, 0.25, 0.0], [(0, 3), (2, 5)],
        checkpoint_dir=str(tmp_path),
        preopt_file=str(tmp_path / "preopt.json"),
        repair_policy=repair_policy,
        route_to_controller=route_to_controller,
    ), logged


def test_routing_signal_is_handed_to_the_controller_not_raised(tmp_path):
    (result, lam, ranges), logged = _call(tmp_path, route_to_controller=True)

    assert result["routing_signal"] == "LOCAL_VALIDATION_CAP"
    assert result["converged"] is False          # 绝不伪装成一次成功的执行
    assert lam and ranges                        # 路径原样交回，没被插点/拆窗动过
    assert any("LOCAL_VALIDATION_CAP" in m for m in logged)


def test_it_still_raises_when_no_controller_will_take_over(tmp_path):
    """没有控制器就没人去延长那个窗口的验证预算 ⟹ 静默返回会把「没结论」
    伪装成执行完毕。那时必须照旧抛。"""
    with pytest.raises(_ie.IBSValidationBudgetIndeterminateError):
        _call(tmp_path, route_to_controller=False)


# ---------------------------------------------------------------------------
# 全局拼接失败：窗口被跳过 ⟹ 路由（SKIPPED_WINDOW），不是终态
# ---------------------------------------------------------------------------
# 真机 2026-09-14（另一组体系）：win0/win3 去相关后只剩 7~9 帧（门 10）被跳过
# ⟹ 拼接链缺共享 λ 节点 ⟹ `window_overlap_broken` ⟹ RuntimeError 炸穿管线。

def test_routing_error_strings_match_what_the_solver_actually_emits():
    """这几个 reason 是**字符串匹配**的，拼错就等于这条路由从来没生效。

    `no_local_tmbAR_results` 的大小写就是源码里那么怪 —— `abfe_pipeline` 的
    注释里一度写成全小写的 `no_local_tmbar_results`，照那个抄必然漏。
    """
    import ast
    import pathlib

    from abfe_pipeline import STAGE_SOLVE_ROUTING_ERRORS

    root = pathlib.Path(__file__).resolve().parents[1]
    emitted = {
        node.args[0].value
        for node in ast.walk(ast.parse((root / "ibs_engine.py").read_text("utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("_fallback", "_incomplete_path_fallback")
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    missing = sorted(STAGE_SOLVE_ROUTING_ERRORS - emitted)
    assert not missing, (
        f"这些 reason 在 ibs_engine 里根本发不出来（拼错或已改名）：{missing}；"
        f"求解器实际会发的是 {sorted(emitted)}"
    )


def test_skipped_window_solve_failure_is_not_fatal_but_stays_unconverged():
    """路由 ≠ 放行：错误原样留档、`converged` 保持 False。"""
    from abfe_pipeline import STAGE_SOLVE_ROUTING_ERRORS

    # 这是求解器 `_fallback()` 的真实形状。
    stage_result = {"error": "window_overlap_broken", "converged": False,
                    "total_delta_G": 0.0, "total_error": 999.9}
    assert stage_result["error"] in STAGE_SOLVE_ROUTING_ERRORS

    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1] / "abfe_pipeline.py"
    ).read_text("utf-8")
    assert "STAGE_SOLVE_ROUTING_ERRORS" in src
    # 绝不能顺手把它标成收敛/可发布
    assert 'stage_result["converged"] = True' not in src


def test_stitching_failure_keeps_the_list_of_skipped_windows():
    """拼接失败的出口必须带着「哪些窗口被跳过」。

    不带的后果不是少了条日志：控制器那条「被踢出协方差链 ⟹ 补采」的分支
    读的就是 `stage["skipped_windows"]`，丢了它这条分支**不可达**，于是控制器
    只能对同一个窗口反复 ANALYZE 到停滞保护退出（真机 2026-09-14）。
    """
    from ibs_engine import GlobalMBARAnalyzer

    analyzer = GlobalMBARAnalyzer.__new__(GlobalMBARAnalyzer)   # 纯函数，不需要状态
    skipped = [{"window_index": 0, "n_frames_after_decorrelation": 9,
                "min_frames_per_window": 10,
                "reason": "insufficient_frames_after_decorrelation"}]

    out = analyzer._incomplete_path_fallback("window_overlap_broken", skipped)
    assert out["error"] == "window_overlap_broken"
    assert out["converged"] is False
    assert out["skipped_windows"] == skipped
    assert out["path_is_complete"] is False

    # 一个窗口都没跳过时仍然是完整路径（那就是另一种失败，别混）
    assert analyzer._incomplete_path_fallback("x", [])["path_is_complete"] is True


def test_every_stitching_exit_carries_the_diagnosis():
    """三个拼接失败出口都必须走带诊断的那条，别有人再用裸 `_fallback`。"""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    fn = next(
        n for n in ast.walk(ast.parse((root / "ibs_engine.py").read_text("utf-8")))
        if isinstance(n, ast.FunctionDef) and n.name == "solve_stage_integrated"
    )
    bare = {
        node.args[0].value
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_fallback"
        and node.args and isinstance(node.args[0], ast.Constant)
    }
    from abfe_pipeline import STAGE_SOLVE_ROUTING_ERRORS
    leaked = sorted(bare & STAGE_SOLVE_ROUTING_ERRORS)
    assert not leaked, f"这些出口还在用裸 _fallback（丢掉 skipped_windows）：{leaked}"


def test_the_non_mutating_early_return_is_guarded_too(tmp_path):
    """自治启用时策略被降级成 `non_mutating_v1` ⟹ **早退那条才是实际走的路**。

    那两个 `return run_once(...)` 不在主 try 的覆盖范围内；不包住它们，
    整个路由修复在默认配置下等于没做。
    """
    (result, lam, _r), logged = _call(
        tmp_path, route_to_controller=True, repair_policy="non_mutating_v1")
    assert result["routing_signal"] == "LOCAL_VALIDATION_CAP"
    assert result["converged"] is False
    assert lam
    assert any("LOCAL_VALIDATION_CAP" in m for m in logged)


def test_autonomous_really_downgrades_path_evolution_not_just_logs_it():
    """「已关闭 path_evolution」先前是一句假日志：只拼字符串、不改变量，
    而且那段代码跑在 `_run_stage2_with_path_evolution` **之后** 150 行。"""
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / "abfe_pipeline.py").read_text("utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_full_pipeline")

    # 1) 真的有一次赋值把它降级，而不是只出现在字符串里
    downgrades = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_sampling_repair_policy"
                for t in n.targets)
        and isinstance(n.value, ast.Constant)
        and n.value.value == "non_mutating_v1"
    ]
    assert downgrades, "自治启用时没有任何一处真的把 path_evolution 降级"

    # 2) 降级必须发生在传给 `_run_stage2_with_path_evolution` **之前**
    call_lines = [
        n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_run_stage2_with_path_evolution"
    ]
    assert call_lines, "找不到 path_evolution 的调用点"
    assert min(d.lineno for d in downgrades) < min(call_lines), (
        "降级发生在调用之后 = 路径已经被改过了，等于没关"
    )

    # 3) 跟另外两样一样，要有硬断言兜底
    assert '_sampling_repair_policy != "path_evolution_v1"' in src


def test_the_warmup_convergence_signal_is_routed_too(tmp_path):
    """`IBSWarmupConvergenceError` 同样是路由信号，早退那条路也得接住。

    真机 `cyclod_ligand2/rep2` win4：它在 `while True` 那条路里本来是被接住并演化
    路径的 —— 但自治控制器启用时 `repair_policy` 已被降级成 `non_mutating_v1`
    （单控制器，布局动作归 `decide()`），**早退这条路成了实际路径**，那个 handler
    变得不可达，异常直接炸穿 `run_full_pipeline`。
    """
    import types

    logged = []
    stub = types.SimpleNamespace(_log=logged.append)

    def run_once(*_a, **_kw):
        raise _ie.IBSWarmupConvergenceError(
            "窗口 4 的 IBS 偏置预热在 2 次权重更新后未收敛（f_not_converged）。",
            {"window_index": 4, "final_mode": "learning"})

    (result, lam, ranges) = ABFEPipeline._run_stage2_with_path_evolution(
        stub, run_once, [1.0, 0.75, 0.5, 0.25, 0.0], [(0, 3), (2, 5)],
        checkpoint_dir=str(tmp_path), preopt_file=str(tmp_path / "p.json"),
        repair_policy="non_mutating_v1",        # ← 自治启用时的实际策略
        route_to_controller=True,
    )
    assert result["routing_signal"] == "WARMUP_F_K_NOT_CONVERGED"
    assert result["converged"] is False
    assert lam and ranges                        # 路径一个字节没动
    assert any("WARMUP_F_K_NOT_CONVERGED" in m for m in logged)


def test_it_still_raises_without_a_controller_downstream(tmp_path):
    import types

    def run_once(*_a, **_kw):
        raise _ie.IBSWarmupConvergenceError("x", {})

    with pytest.raises(_ie.IBSWarmupConvergenceError):
        ABFEPipeline._run_stage2_with_path_evolution(
            types.SimpleNamespace(_log=lambda *a: None), run_once,
            [1.0, 0.5, 0.0], [(0, 3)],
            checkpoint_dir=str(tmp_path), preopt_file=str(tmp_path / "p.json"),
            repair_policy="non_mutating_v1", route_to_controller=False)
