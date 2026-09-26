"""合并/救援路径必须和正常路径用同一套判据口径；split-half 漂移门要接得出来。

来历：09-19 cyclod 重算批的三条 `analysis_mode=merged_segments` 腿，落盘的
`diagnostics` 是空 `{}`、σ 下界 `applied=False`。根因是**一个**：f_k 重标定的合并
求解裸调 `solve_stage_integrated`（六个门阈值没传、`inflate_sigma_from_split_half`
吃了函数默认 False），而填 `diagnostics` 的 `_populate_stage_diagnostics` 又只在
正常路径和 bridge rescue 分支里被调。这是同一个坑的**第三次**（前两次见该方法的
docstring）。所以这里钉契约，不钉某一次的数值。

⚠️ 「六门阈值」那半在真机上**今天无观测**：仓库与 benchmark 的 config 一个门阈值键
都没设，两套值恰好重合（abfe-ibs-b1 核过八条腿落盘阈值，逐位相同）。也就是说只有
显式塞非默认阈值才测得出来 —— 这正是它必须由契约测试而不是回归数据来守的理由。
"""
import ast
import io
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PIPELINE = os.path.join(_ROOT, "abfe_pipeline.py")

_ENTRY = "_solve_stage_with_config_gates"
# 判据口径的全部形参。少传一个 = 那条路径按函数默认值判，而协议指纹记的是 config 那套。
_GATE_KWARGS = {
    "final_min_ess_ratio",
    "final_min_absolute_ess",
    "final_min_decorrelated_samples",
    "final_max_uncertainty_kJ_mol",
    "final_min_target_absolute_ess",
    "final_max_top1pct_raw_weight",
    "split_half_max_z",
    "inflate_sigma_from_split_half",
}


def _pipeline_tree():
    with io.open(_PIPELINE, encoding="utf-8") as fh:
        return ast.parse(fh.read())


def test_only_the_single_entry_calls_the_stage_solver():
    """本文件里只有 `_solve_stage_with_config_gates` 可以直调 stage 求解器。

    这是本轮最贵的一条教训：`solve_stage_integrated` 曾有 6 个调用点，每个都得自己
    记得传 7 个判据形参 + 自己记得填诊断，漏了不报错。而自治控制器的两条合并求解是
    `from ibs_engine import solve_stage_integrated as _solve` **换名**调的 ——
    按函数名做的检查根本搜不到它们，于是"每个调用点都传全了"的契约在真正出事的那
    两条路径上是瞎的。所以这里钉的不是"每个调用点传全"，是"只许有一个调用点"。
    """
    offenders, aliases = [], []
    for fn in ast.walk(_pipeline_tree()):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                nm = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if nm == "solve_stage_integrated" and fn.name != _ENTRY:
                    offenders.append(f"{fn.name}:{node.lineno}")
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.name == "solve_stage_integrated" and a.asname:
                        aliases.append(f"{fn.name}:{node.lineno} as {a.asname}")
    assert not offenders, (
        f"这些地方绕过 {_ENTRY} 直调 stage 求解器：{offenders}。"
        "绕过 = 自己维护一套判据阈值与诊断填充，而漏了不报错。"
    )
    assert not aliases, (
        f"换名导入 stage 求解器：{aliases}。换名 = 让所有按名字做的检查失明，"
        "09-19 那三条 merged 腿正是这么漏掉的。"
    )


def test_the_single_entry_carries_the_full_gate_contract():
    entry = next(
        (n for n in ast.walk(_pipeline_tree())
         if isinstance(n, ast.FunctionDef) and n.name == _ENTRY), None
    )
    assert entry is not None, f"{_ENTRY} 不见了 —— 测试自己过期了"
    call = next(
        n for n in ast.walk(entry)
        if isinstance(n, ast.Call)
        and (getattr(n.func, "id", None) or getattr(n.func, "attr", None))
        == "solve_stage_integrated"
    )
    missing = sorted(_GATE_KWARGS - {k.arg for k in call.keywords})
    assert not missing, f"唯一入口少传 {missing}"
    assert any(
        isinstance(n, ast.Call)
        and (getattr(n.func, "id", None) or getattr(n.func, "attr", None))
        == "_populate_stage_diagnostics"
        for n in ast.walk(entry)
    ), (
        f"{_ENTRY} 没填 `diagnostics`。`solve_stage_integrated` 自己不产出这个键，"
        "不填 ⟹ 落盘的 stage 结果里逐窗 σ/g/overlap/split-half 全空，"
        "质量门拿不到证据只能空过（09-19 那三条 merged 腿就是这样）。"
    )


def test_split_half_gate_requires_total_and_per_window_drift():
    """只按 max_window_z 拦会误杀「窗口间归属重排」。

    实测（09-19 cyclod 8 条腿、阈值 2.0，abfe-ibs-b1 提供）：只看 max_window_z 拦 6 条，
    其中 5 条 total_z < 2 是重排；要求两者都超只拦 1 条，正好是唯一真的没平衡的那条。
    代码自己在 WARN 文案里写着这条规则（"两者都大才是真的前后半程不一致"）。
    """
    src = io.open(os.path.join(_ROOT, "ibs_engine.py"), encoding="utf-8").read()
    block = src[src.index("split_half_max_z is not None"):]
    block = block[:block.index("split_half_gate_failed")]
    assert "_tot_z is not None" in block and "_tot_z >" in block, (
        "split-half 硬门只判了逐窗 z。总量稳而逐窗大 ⟹ 多半是归属重排不是漂移，"
        "这么接出来的是个会误杀 5/6 的门。"
    )
    assert "max_z >" in block, "总量 z 也不能单独判：两者都大才算前后半程不一致"


def test_the_drift_gate_knob_is_reachable_from_config():
    """判据一直在，缺的只是一条从 config 进去的路 —— 那条路现在必须存在。"""
    pipeline_src = io.open(_PIPELINE, encoding="utf-8").read()
    runabfe_src = io.open(os.path.join(_ROOT, "runabfe.py"), encoding="utf-8").read()
    assert 'kwargs.get("split_half_max_z")' in pipeline_src
    assert runabfe_src.count('split_half_max_z=config.get("split_half_max_z")') == 2, (
        "runabfe 是**逐键枚举**往 run_full_pipeline 传参，两条腿都得写 —— "
        "漏一条就是那条腿的阈值永远进不了 kwargs（`stage2_autonomous_controller` "
        "栽过同一个坑）"
    )
    # 默认必须仍是"只诊断不阻断"：不设就不进指纹，存量 run 逐位不变。
    assert 'if kwargs.get("split_half_max_z") is not None:' in pipeline_src, (
        "阈值无条件进 `_final_gate_thresholds` ⟹ 所有存量 stage 结果缓存一次性失配"
    )


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_"):
            _fn()
            print("ok", _name)


def test_the_drift_threshold_never_enters_the_sampling_identity():
    """`split_half_max_z=None` 不得改变 stage 指纹。

    2026-09-20 真机回归：runabfe 逐键枚举传参 ⟹ 不设也会把 `split_half_max_z=None`
    塞进 `kwargs` ⟹ 逐字段进 `_stage_protocol_key` ⟹ 一个 `None` 换掉 sha256 ⟹
    **全仓任何 resume 都重跑 stage 1 + stage 2**。与 09-11
    `max_bias_warmup_steps` 那次是同一个形状：新键进 stage 身份 = 全部缓存作废。

    判据阈值该走 `final_gate_thresholds`（只作废 stage 结果、窗口轨迹照常复用），
    不该走 `run_config.kwargs`。
    """
    import abfe_pipeline as ap

    assert "split_half_max_z" in ap._NON_IDENTITY_KWARGS
    base = {"kwargs": {"final_min_ess_ratio": 0.05}}
    with_knob = {"kwargs": dict(base["kwargs"], split_half_max_z=None)}
    assert (ap._strip_non_identity_kwargs(with_knob)["kwargs"]
            == ap._strip_non_identity_kwargs(base)["kwargs"]), (
        "带不带这个键必须得到逐位相同的身份 payload"
    )
