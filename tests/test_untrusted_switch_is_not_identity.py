"""`allow_untrusted_stage_results` 是执行策略，不得进任何缓存身份键。

背景（本仓库反复出现过的同一类 bug）：`_stage_protocol_key` 把整个
`_last_run_config` 逐字段写进 stage 协议指纹。因此任何新加的顶层键或 kwargs 键
都会让全部 stage 缓存失配、强制重跑 GPU（实测代价：Stage 1 约 28 分钟 +
6 个 vanishing 窗口全部重采样）。

这个开关只决定「质量门没过时是中止、还是标记 results_untrusted 继续」——
它不改变任何被计算出来的数：同一份轨迹在开关两种取值下算出的 ΔG 逐位相同。
所以它必须像 `resume` / `run_equilibration` 一样被剔除出身份。

同时钉住它**确实生效**：默认 fail-closed，显式打开才放行。
"""
import json
from pathlib import Path

import pytest

import abfe_pipeline as ap

REPO = Path(__file__).resolve().parent.parent
FLAG = "allow_untrusted_stage_results"


def _run_config_after_identity_scrub(raw_config):
    """跑 `_stage_protocol_key` 里从 run_config 构造到剔除结束的那一段真实代码。"""
    import inspect

    src = inspect.getsource(ap.ABFEPipeline._stage_protocol_key).split("\n")
    start = next(i for i, l in enumerate(src) if "run_config = dict(" in l)
    end = next(i for i, l in enumerate(src) if "payload = {" in l)
    body = "\n".join(l[8:] for l in src[start:end])

    obj = object.__new__(ap.ABFEPipeline)
    obj._last_run_config = raw_config
    scope = {"self": obj}
    exec(body, {"getattr": getattr, "dict": dict, "isinstance": isinstance}, scope)
    return scope["run_config"]


def test_flag_is_scrubbed_from_stage_protocol_key_both_levels():
    """顶层和 kwargs 两处都要剔除，且不得连带改动其它字段。"""
    baseline = {
        "decoupling_scheme": "dual_lambda",
        "potential_type": "softcore",
        "n_states_per_stage": 16,
        "kwargs": {"decharge_method": "pme", "warmup_steps": 1000},
    }
    with_flag = json.loads(json.dumps(baseline))
    with_flag[FLAG] = True
    with_flag["kwargs"][FLAG] = True
    with_flag["resume"] = True
    with_flag["run_equilibration"] = False

    scrubbed = _run_config_after_identity_scrub(with_flag)
    clean = _run_config_after_identity_scrub(json.loads(json.dumps(baseline)))

    assert FLAG not in scrubbed
    assert FLAG not in scrubbed["kwargs"]
    # 开关不论开关都必须落到**同一个**身份上，否则打开它就等于重跑 GPU
    assert json.dumps(scrubbed, sort_keys=True) == json.dumps(clean, sort_keys=True)


def test_flag_does_not_mutate_the_callers_config_dict():
    """剔除必须作用在副本上：污染 _last_run_config 会让落盘的 provenance 丢掉这个决定。"""
    cfg = {"kwargs": {FLAG: True, "warmup_steps": 1000}, FLAG: True}
    _run_config_after_identity_scrub(cfg)
    assert cfg[FLAG] is True
    assert cfg["kwargs"][FLAG] is True


def _failing_vanishing_result():
    return {
        "stage": "vanishing",
        "total_delta_G": 12.3,
        "total_error": 0.9,
        "target_support_gate": {
            "passed": False,
            "failure_reason": "raw_min_absolute_ess_below_threshold",
            "failed_checks": ["raw_min_absolute_ess"],
            "raw_min_absolute_ess": 2.8,
            "protocol_version": ap.TARGET_SUPPORT_GATE_PROTOCOL_VERSION,
        },
    }


def test_default_is_still_fail_closed():
    obj = object.__new__(ap.ABFEPipeline)
    obj._last_run_config = {}
    with pytest.raises(RuntimeError, match="物理目标支撑度硬门"):
        ap.ABFEPipeline._assert_stage_result_sane(
            obj, "vanishing", _failing_vanishing_result()
        )


def test_explicit_opt_in_continues_but_marks_the_result_untrusted():
    obj = object.__new__(ap.ABFEPipeline)
    obj._last_run_config = {FLAG: True}
    result = _failing_vanishing_result()
    ap.ABFEPipeline._assert_stage_result_sane(obj, "vanishing", result)
    # 放行 ≠ 抹掉证据
    assert result["results_untrusted"] is True
    gates = [
        f.get("gate") if isinstance(f, dict) else f
        for f in (result.get("stage_quality_failures") or [])
    ]
    assert "target_support_gate" in gates


def test_cli_exposes_the_switch_and_defaults_to_off():
    src = (REPO / "runabfe.py").read_text(encoding="utf-8")
    assert '"--allow-untrusted-stage-results"' in src
    assert f"allow_untrusted_stage_results=bool(" in src


def test_explicit_opt_in_also_skips_both_rescue_loops():
    """放行质量门时，两级 rescue 必须一并停掉——否则只是晚几小时走同一条路。

    rescue 的唯一目的是把 `converged` 拱成 True。调用方已经决定「没过也继续」时，
    再跑 2 轮 ×2 倍步数的生产补采 + bridge rescue 不会改变最终走向：补采完照样
    不 converged，照样走放行路径。实测代价：4W53 window 5 的生产目标就是被这个
    循环从 500k 抬到 1M 的。
    """
    src = (REPO / "abfe_pipeline.py").read_text(encoding="utf-8")
    assert "_rescue_disabled_by_untrusted" in src
    i = src.index("_rescue_disabled_by_untrusted = bool(")
    window = src[i : i + 600]
    assert FLAG in window
    # 生产 rescue 轮数被清零
    assert "production_rescue_rounds = 0" in src
    # bridge rescue 也要挂同一个条件
    assert "and not _rescue_disabled_by_untrusted" in src
    # 放弃了什么必须写在日志里，不能静默跳过
    assert "不再自动补救" in src


def test_rescue_skip_is_off_by_default():
    """默认路径不受影响：没打开开关时 rescue 轮数仍来自 kwargs 默认值 2。"""
    src = (REPO / "abfe_pipeline.py").read_text(encoding="utf-8")
    assert 'int(kwargs.get("stage2_production_rescue_rounds", 2))' in src
    i = src.index("if _rescue_disabled_by_untrusted and production_rescue_rounds:")
    assert i > src.index('int(kwargs.get("stage2_production_rescue_rounds", 2))')
