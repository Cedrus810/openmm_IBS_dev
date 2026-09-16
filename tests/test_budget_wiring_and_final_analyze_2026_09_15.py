"""三处「配置在中间层丢失 / 停止决定在收尾层被绕过」的回归钉子。

都按**完整调用链**钉，不按单分支：前两条的失效方式正是"两端各自都对、
中间那一跳把值丢了"，只测端点永远是绿的。
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_BUDGET_KEYS = ("stage2_production_budget_steps",
                "stage2_max_production_blocks_per_window")


def _kwarg_names(src, func_attr):
    """`x.<func_attr>(...)` 这些调用各自传了哪些关键字。"""
    out = []
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == func_attr):
            out.append({kw.arg for kw in node.keywords if kw.arg})
    return out


def test_cli_passes_production_budget_to_both_legs():
    src = open("runabfe.py", encoding="utf-8").read()
    calls = _kwarg_names(src, "run_full_pipeline")
    legs = [c for c in calls if "system_type" in c]
    assert len(legs) == 2, f"预期复合物腿+溶剂腿两处，实到 {len(legs)}"
    for c in legs:
        for k in _BUDGET_KEYS:
            assert k in c, f"{k} 没传进 run_full_pipeline（下游接了、上游没接）"


def test_budget_keys_are_not_part_of_cache_identity():
    """接上预算不得作废任何既有缓存 —— 它们是执行策略，不是身份。"""
    import abfe_pipeline
    cfg = {"n_steps_per_window": 250_000,
           "kwargs": {"foo": 1, **{k: 7 for k in _BUDGET_KEYS}}}
    stripped = abfe_pipeline._strip_non_identity_kwargs(cfg)
    assert stripped["kwargs"] == {"foo": 1}
    assert cfg["kwargs"].get(_BUDGET_KEYS[0]) == 7, "不许就地改入参"
    # 键不在时必须逐位 no-op（既有指纹不能动）
    plain = {"kwargs": {"foo": 1}}
    assert abfe_pipeline._strip_non_identity_kwargs(plain) is plain


def test_aggregated_view_keeps_caller_config(tmp_path):
    """合并视图的生产账必须还是调用方给的那本，不是子控制器的默认值。"""
    from abfe_preoptimizer import Stage2RepairController

    run = tmp_path / "solvent_leg"          # 故意没有 run_provenance.json
    (run / "vanishing").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    cfg = {"n_steps_per_window": 500_000,
           "stage2_production_budget_steps": 2_000_000,
           "stage2_max_production_blocks_per_window": 7}
    ctl = Stage2RepairController.for_physical_stage(
        str(run), "vanishing", "vdw", effective_config=cfg)
    view = ctl.read()
    pb = view["production_budget"]
    assert pb["cap_known"] is True and pb["stage_cap_steps"] == 2_000_000
    assert pb["production_block_steps"] == 500_000
    assert view["max_production_blocks_per_window"] == 7


def test_final_analyze_is_skipped_when_it_would_sample():
    """控制器停手后，收尾的全路径 ANALYZE 不得在缺产物时启动采样。"""
    src = open("abfe_pipeline.py", encoding="utf-8").read()
    body = src.split("def _run_stage2_autonomous(")[1].split("\n    def ")[0]
    tail = body.split("退出前跑一次")[0][-2500:]
    assert "_blockers" in tail and "missing_windows" in tail, \
        "收尾 ANALYZE 前没有'会不会采样'的判据"
    # 真正的调用必须挂在 blockers 为空的那一支上
    assert "elif outcome.get(\"exit\") not in (" in body.split("_final = run_once(")[0][-800:], \
        "run_once 收尾调用没有被 blockers 分支守住"
