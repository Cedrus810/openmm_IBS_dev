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


pytestmark = pytest.mark.cpu_only

REPO = Path(__file__).resolve().parent.parent
FLAG = "allow_untrusted_stage_results"


def _run_config_after_identity_scrub(raw_config, stage_name="decharging"):
    """跑 `_stage_protocol_key` 里从 run_config 构造到剔除结束的那一段真实代码。"""
    import inspect

    src = inspect.getsource(ap.ABFEPipeline._stage_protocol_key).split("\n")
    start = next(i for i, l in enumerate(src) if "run_config = dict(" in l)
    end = next(i for i, l in enumerate(src) if "payload = {" in l)
    body = "\n".join(l[8:] for l in src[start:end])

    obj = object.__new__(ap.ABFEPipeline)
    obj._last_run_config = raw_config
    scope = {"self": obj, "stage_name": stage_name}
    exec(
        body,
        {
            "getattr": getattr,
            "dict": dict,
            "isinstance": isinstance,
            "RESIDUAL_SAMPLING_STAGES": ap.RESIDUAL_SAMPLING_STAGES,
            "_strip_non_identity_kwargs": ap._strip_non_identity_kwargs,
        },
        scope,
    )
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


def _failing_traditional_result():
    """traditional 腿的求解产物：有限 ΔG，但 `converged` 不是 True。

    🔑 [2026-09-15] 这里**故意**还用 `converged` —— 删键只波及
    `ibs_engine.solve_stage_integrated`，traditional 腿这条路径的 `converged`
    是另一个生产者、语义也不同（见 `_sampling_result_convergence_rejection_reason`
    的 docstring）。别"统一"。
    """
    return {"total_delta_G": 12.3, "total_error": 0.9, "converged": False,
            "min_overlap": 0.004, "min_overlap_threshold": 0.05}


# 🔑🔑 [2026-09-15 重新指向] 下面两条原本打在
# `ABFEPipeline._assert_stage_result_sane` 上，断言「默认 fail-closed / 开关打开
# 才放行」。**那个分支已经不在那里了**：同日 target_support_gate 等四道阈值门被
# 整体降级为只报告（不 raise、不触发补帧），`_assert_stage_result_sane` 里现在
# 一处都不读 `allow_untrusted_stage_results`（实测 grep 0 命中）。
#
# 开关本身**没死，是搬家了**，今天有两个落点：
#   · traditional 腿 —— `_assert_or_warn_sampling_converged(allow_untrusted=…)`，
#     即下面这两条打的地方；
#   · 自治控制器 —— 只改 `trust_level`（`abfe_preoptimizer.py` 约 4663 / 6139）
#     与 rescue 跳过（`abfe_pipeline.py` 约 16840）。
# 本文件的主题（开关不得进缓存身份）由上面四条覆盖，与落点无关。

def test_default_is_still_fail_closed():
    with pytest.raises(RuntimeError, match="未通过收敛 sanity gate"):
        ap._assert_or_warn_sampling_converged(
            _failing_traditional_result(), context="vanishing",
            allow_untrusted=False, log=lambda *a, **k: None,
        )


def test_explicit_opt_in_continues_but_marks_the_result_untrusted():
    logged = []
    reason = ap._assert_or_warn_sampling_converged(
        _failing_traditional_result(), context="vanishing",
        allow_untrusted=True, log=logged.append,
    )
    # 放行 ≠ 抹掉证据：拒绝理由必须**返回给调用方落盘**，并且日志里刺眼。
    assert reason is not None and "converged=False" in reason
    assert any("allow_untrusted_stage_results=True 显式放行" in m for m in logged), logged
    assert any("不得作为可发布结果" in m for m in logged), logged


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


# =============================================================================
# 同型第二例：`residual_sampling` 只对 vanishing 是身份
# =============================================================================
# 残差项只改 vanishing 的 Hamiltonian（`_run_dual_lambda_stage` 的
# `residual_for_stage` 判据就是 `RESIDUAL_SAMPLING_STAGES`），但它曾经**无条件**
# 进 `_stage_protocol_key`：既走 run_config，也走显式插入的 `_residual_payload`。
# 后果与上面那个开关完全同型 —— 打开残差会让压根没被它碰过的 decharging stage
# 缓存失配、整段约 28 分钟白重跑。


def _residual_config(enabled: bool):
    cfg = {
        "decoupling_scheme": "dual_lambda",
        "potential_type": "softcore",
        "n_states_per_stage": 16,
        "kwargs": {"decharge_method": "pme"},
    }
    if enabled:
        cfg["residual_sampling"] = {
            "enabled": True,
            "sampling_score_sha256": "a" * 64,
            "residual_energy_offset_kj_mol": 0.0,
        }
    return cfg


def test_residual_sampling_is_not_identity_for_decharging():
    """开残差不得动 Stage 1 的身份。"""
    on = _run_config_after_identity_scrub(_residual_config(True), "decharging")
    off = _run_config_after_identity_scrub(_residual_config(False), "decharging")
    assert "residual_sampling" not in on
    assert json.dumps(on, sort_keys=True) == json.dumps(off, sort_keys=True)


@pytest.mark.parametrize("stage_name", sorted(ap.RESIDUAL_SAMPLING_STAGES))
def test_residual_sampling_stays_identity_where_it_actually_runs(stage_name):
    """收窄不能收过头：残差真生效的 stage 必须仍然失配，否则是静默串协议。"""
    on = _run_config_after_identity_scrub(_residual_config(True), stage_name)
    off = _run_config_after_identity_scrub(_residual_config(False), stage_name)
    assert on["residual_sampling"]["enabled"] is True
    assert json.dumps(on, sort_keys=True) != json.dumps(off, sort_keys=True)


def test_runtime_and_cache_read_the_same_stage_set():
    """运行期判据与缓存身份判据必须是同一个集合 —— 漂开就是这个 bug 本身。"""
    import inspect

    src = inspect.getsource(ap.ABFEPipeline._run_dual_lambda_stage)
    assert "stage_name in RESIDUAL_SAMPLING_STAGES" in src
    assert ap.RESIDUAL_SAMPLING_STAGES == frozenset({"vanishing", "vanishing_rescue"})
