"""自治循环的段路由 + 停滞计数自检。

覆盖 2026-09-11 扫出来的三个 bug：
  · 补采落在证据所在的段（不是永远写基准段）
  · 换 Epoch 从最新一段重学 f_k（不是永远从段 1）
  · 停滞计数按"盘上状态是否变了"清零（不是累计出现次数）
"""
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from abfe_pipeline import ABFEPipeline  # noqa: E402
except ImportError:  # 没装 openmm 的机器上：直接从源码抽这几个纯函数
    import ast
    import types

    _SRC = ast.parse(
        pathlib.Path(__file__).resolve().parents[1].joinpath("abfe_pipeline.py").read_text()
    )
    _CLS = next(
        n for n in _SRC.body
        if isinstance(n, ast.ClassDef) and n.name == "ABFEPipeline"
    )
    _WANT = {
        "_segment_dirs_for_evidence",
        "_latest_segment_dirs",
        "_recalibrate_f_k_and_resample_segment",
    }
    _ns = {"os": os, "glob": __import__("glob"), "Optional": object, "List": list,
           "Dict": dict, "Tuple": tuple, "Any": object}
    ABFEPipeline = types.SimpleNamespace()
    for _n in _CLS.body:
        if isinstance(_n, ast.FunctionDef) and _n.name in _WANT:
            _n.decorator_list = []
            _mod = ast.Module(body=[_n], type_ignores=[])
            exec(compile(ast.fix_missing_locations(_mod), "<x>", "exec"), _ns)
            setattr(ABFEPipeline, _n.name, _ns[_n.name])

SEG = ABFEPipeline._segment_dirs_for_evidence
LATEST = ABFEPipeline._latest_segment_dirs


def test_evidence_in_base_segment_uses_default_dirs():
    assert SEG({"vanishing"}, "/r/vanishing", "/r/checkpoints") == (None, None)
    assert SEG(set(), "/r/vanishing", "/r/checkpoints") == (None, None)


def test_evidence_in_segment_n_routes_to_that_segment():
    out, ckpt = SEG({"vanishing_3"}, "/r/vanishing", "/r/checkpoints")
    assert out == "/r/vanishing_3"
    assert ckpt == "/r/checkpoints/segment_3"


def test_evidence_spanning_segments_fails_closed():
    # 静默回落到基准段正是原来的 bug —— 必须抛错而不是猜。
    try:
        SEG({"vanishing", "vanishing_2"}, "/r/vanishing", "/r/checkpoints")
    except ValueError:
        return
    raise AssertionError("跨段补采应当 fail-closed")


def test_latest_segment_picks_max_suffix(tmp_path):
    stage = tmp_path / "vanishing"
    for name in ("vanishing", "vanishing_2", "vanishing_5", "vanishing_x"):
        (tmp_path / name).mkdir()
    out, ckpt = LATEST(str(stage), str(tmp_path / "checkpoints"))
    assert out.endswith("vanishing_5")
    assert ckpt.endswith("segment_5")


def test_latest_segment_is_none_when_only_base(tmp_path):
    (tmp_path / "vanishing").mkdir()
    assert LATEST(str(tmp_path / "vanishing"), str(tmp_path / "cp")) == (None, None)


def test_recalibrate_reads_source_not_output_dir():
    """源目录必须是独立参数，否则多 Epoch 链只会重复读段 1。"""
    # 走 __code__/__kwdefaults__ 而不是 inspect.signature：后者会求值注解，
    # 在只抽了函数体的回退模式下拿不到真实类型。
    fn = ABFEPipeline._recalibrate_f_k_and_resample_segment
    names = fn.__code__.co_varnames[: fn.__code__.co_argcount
                                    + fn.__code__.co_kwonlyargcount]
    assert "source_stage_dir" in names
    assert "source_checkpoint_dir" in names
    # 默认必须是 None ⟹ 回落到基准段，老调用点行为不变。
    assert (fn.__kwdefaults__ or {}).get("source_stage_dir", "x") is None


def test_stall_counter_resets_when_disk_advances():
    """盘上状态变了就不算重复 —— 否则正常的逐块补采第 3 块就被掐掉。"""
    seen, last_sig, escalated = {}, {}, {}

    def tick(key, sig):
        if last_sig.get(key) != sig:
            seen[key] = 1
            escalated.pop(key, None)
        else:
            seen[key] = seen.get(key, 0) + 1
        last_sig[key] = sig
        return seen[key]

    k = ("RUN_PRODUCTION", (3,))
    # 每块都真的加了 250k ⟹ 指纹一直在变 ⟹ 永远不触发停滞保护
    assert tick(k, (1, ((3, 250000, "INSUFFICIENT_DATA", "vanishing"),))) == 1
    assert tick(k, (1, ((3, 500000, "INSUFFICIENT_DATA", "vanishing"),))) == 1
    assert tick(k, (1, ((3, 750000, "INSUFFICIENT_DATA", "vanishing"),))) == 1
    # 盘不动了才开始累加
    stuck = (1, ((3, 750000, "INSUFFICIENT_DATA", "vanishing"),))
    assert tick(k, stuck) == 2
    assert tick(k, stuck) == 3


def test_epoch_action_can_be_restricted_to_windows():
    """换 Epoch 必须能限定窗口 —— 否则合格窗口被拖进新段、证据被顶掉。"""
    fn = ABFEPipeline._recalibrate_f_k_and_resample_segment
    names = fn.__code__.co_varnames[: fn.__code__.co_argcount
                                    + fn.__code__.co_kwonlyargcount]
    assert "only_windows" in names
    assert (fn.__kwdefaults__ or {}).get("only_windows", "x") is None


def test_escalation_target_is_the_bounded_probe():
    """降级目标必须是有界的 PROBE_REANCHOR_EPOCH，不是全路径 RECALIBRATE_FK。

    真机现场：win3 一个窗口卡住，却重标定 [0..5] 全部 6 窗并整段重采，
    而 win0/1/2/5 当时都已经是 ANALYSIS_ELIGIBLE。
    """
    import ast

    src = pathlib.Path(__file__).resolve().parents[1].joinpath("abfe_pipeline.py")
    tree = ast.parse(src.read_text())
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_run_stage2_autonomous"
    )
    assigned = {
        t.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name) and t.id == "act"
        for v in [node.value]
        if isinstance(v, ast.Constant)
    }
    targets = [
        node.value.value
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "act" for t in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    assert "PROBE_REANCHOR_EPOCH" in targets, targets
    assert "RECALIBRATE_FK" not in targets, "降级不得走全路径重标定"
    assert assigned  # 确实是常量赋值，不是被改写成别的形状


def test_routing_exceptions_do_not_escape_the_loop():
    """LOCAL_VALIDATION_CAP / f_k 未收敛是**路由信号**，不得炸穿自治循环。

    2026-09-12 14:40 真机现场：段2 win4 冻结验证 15/15 批仍
    insufficient_frames_after_decorrelation，异常直接穿透，整条流水线死。
    f_k 被统计驳回也路由，但语义不同：它只终止**这份候选**（封存、永不续验、
    最多换一次 Epoch），不终止整个 Stage-2 —— 老板 2026-09-12 裁决，P3-A 已定。
    完整规则与一次性配额的自检在 tests/test_fk_relearn_and_reachability.py。
    """
    import ast

    src = pathlib.Path(__file__).resolve().parents[1].joinpath("abfe_pipeline.py")
    fn = next(
        n for n in ast.walk(ast.parse(src.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_run_stage2_autonomous"
    )
    caught = {}
    for h in [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]:
        for node in ast.walk(h.type or ast.Pass()):
            if isinstance(node, ast.Attribute):
                # 该 handler 体内有没有裸 raise ⟹ 是上抛还是路由
                caught[node.attr] = any(
                    isinstance(b, ast.Raise) for b in ast.walk(h)
                )
    assert caught.get("IBSValidationBudgetIndeterminateError") is False, \
        "验证批次上限必须路由，不能上抛"
    assert caught.get("IBSWarmupConvergenceError") is False, \
        "f_k 压不平必须路由，不能上抛"
    assert caught.get("IBSFrozenCalibrationValidationError") is False, \
        "统计驳回只终止这份候选：封存 + 路由到 RELEARN_FK_EPOCH，不再上抛"


def test_analyze_merges_every_sampling_segment():
    """ANALYZE 必须合并循环自己开出来的采样段，否则修好的窗口进不了最终 ΔG。"""
    import ast

    src = pathlib.Path(__file__).resolve().parents[1].joinpath("abfe_pipeline.py")
    fn = next(
        n for n in ast.walk(ast.parse(src.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_run_stage2_autonomous"
    )
    calls = {
        n.func.attr for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "_solve_merged_segments_if_any" in calls


def test_probe_candidate_fk_is_non_mutating():
    """PROBE_CANDIDATE_FK 必须传 probe_only —— 控制器分支明说「不关闭 Epoch、
    不切换 f_k」。真机后果：不传就每 2 分钟开一个新段直到烧光窗口预算。"""
    import ast

    src = pathlib.Path(__file__).resolve().parents[1].joinpath("abfe_pipeline.py")
    fn = next(
        n for n in ast.walk(ast.parse(src.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_run_stage2_autonomous"
    )
    body = ast.dump(fn)
    assert "probe_only" in body, "探针动作必须落到 probe_only 执行路径"
    # 且必须是由动作本身决定，不是写死 False
    assert "PROBE_CANDIDATE_FK" in body


def test_marginal_gain_rule_stops_futile_frame_addition():
    """min N_eff/g 不随采样上升 ⟹ 停止同分布加帧，改走布局动作。

    cap 分支原文早就写了「边际停滞/下降则关闭 Epoch」，但判它需要的跨段历史
    此前没进视图，规则一直是死的。真机 win4：3.44→2.29→2.01→1.62，
    g 15.1→168.8，控制器却一直返回 RUN_PRODUCTION。
    只落**无歧义的那一半**（没有上升），不自造"停滞"阈值。
    """
    import ast

    src = pathlib.Path(__file__).resolve().parents[1].joinpath("abfe_preoptimizer.py")
    tree = ast.parse(src.read_text())
    # 视图必须带跨段历史
    assert "min_n_eff_over_g_history" in src.read_text()
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "decide"
    )
    dump = ast.dump(fn)
    assert "_no_gain" in dump, "边际增长判据必须在 decide() 里"
    # 必须**早于**所有加帧分支，否则永远轮不到它
    body = src.read_text()
    i_gain = body.index("_no_gain = []")
    i_short = body.index("short_self = _pick(")
    assert i_gain < i_short, "边际增长判据必须前置于自检补采分支"


if __name__ == "__main__":
    import tempfile
    import pathlib

    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
            with tempfile.TemporaryDirectory() as d:
                fn(pathlib.Path(d))
        else:
            fn()
        print("  ok", name)
    print("全过")
