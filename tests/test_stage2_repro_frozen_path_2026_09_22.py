"""复现模式（等分 λ + 钉死分窗 + 关掉一切布局动作）的契约。

用途：同一体系的不同 rep / 不同机器共用**逐位相同**的 λ 表与窗口，
让 ΔG 的差分只反映采样，不反映布局。
"""

import inspect

import pytest

import abfe_pipeline as pl
import ibs_engine as ie
import runabfe
from test_preopt_protocol_scope import _fake_pipeline, _key


pytestmark = pytest.mark.cpu_only


def test_uniform_lambda_table_is_exact_at_both_ends_and_evenly_spaced():
    lambdas = pl.uniform_vanishing_lambda_path(25)
    assert len(lambdas) == 25
    assert lambdas[0] == 1.0 and lambdas[-1] == 0.0
    # λ_i = 1 − i/24，逐位可复现
    assert lambdas == [1.0 - i / 24 for i in range(25)]
    steps = [a - b for a, b in zip(lambdas, lambdas[1:])]
    assert max(steps) - min(steps) < 1e-12
    with pytest.raises(ValueError):
        pl.uniform_vanishing_lambda_path(1)


def test_repro_mode_is_off_by_default_and_touches_nothing():
    kwargs = {"sampling_repair_policy": "path_evolution_v1"}
    assert pl._freeze_stage2_path_for_repro(kwargs, lambda *_: None) is None
    assert kwargs == {"sampling_repair_policy": "path_evolution_v1"}


def test_repro_mode_requires_explicit_windows():
    with pytest.raises(RuntimeError) as exc:
        pl._freeze_stage2_path_for_repro(
            {"stage2_uniform_lambda_states": 25}, lambda *_: None)
    assert "stage2_window_ranges" in str(exc.value)
    # 25 态每窗 5 的形状要直接写在报错里，不让人自己算
    assert "[0, 5]" in str(exc.value) and "[20, 25]" in str(exc.value)


def test_repro_mode_freezes_every_layout_switch():
    """关键：只关控制器**不冻结布局** —— benchmark 的 config 全预置 path_evolution_v1。"""
    kwargs = {
        "stage2_uniform_lambda_states": 25,
        "stage2_window_ranges": [[0, 5], [4, 9], [8, 13], [12, 17], [16, 21], [20, 25]],
        # 故意摆上 benchmark 的真实取值：复现模式必须把它们改掉
        "sampling_repair_policy": "path_evolution_v1",
        "stage2_autonomous_controller": True,
        "max_path_insertions": 3,
    }
    assert pl._freeze_stage2_path_for_repro(kwargs, lambda *_: None) == 25
    assert kwargs["sampling_repair_policy"] == "non_mutating_v1"
    assert kwargs["max_path_insertions"] == 0
    assert kwargs["stage2_frozen_layout"] is True
    # 控制器关掉：换 f_k 改走固定节奏（跑满一段就换，共 stage2_repro_fk_cycles 轮）。
    # `stage2_frozen_layout` 仍置位，是给"万一有人把控制器打开"兜底。
    assert kwargs["stage2_autonomous_controller"] is False
    # bridge rescue 的开关是 `and not _autonomous_on` ⟹ 关掉控制器反而会打开它，
    # 而它自己那道 min_overlap 判据会抛。复现模式必须显式关掉。
    assert kwargs["stage2_enable_bridge_rescue"] is False
    # 阈值类判据降成 WARN（现成通道，且不进指纹 ⟹ 不作废缓存）
    assert kwargs["allow_untrusted_stage_results"] is True
    # 缓存校验拿 final_n_states 比态数，两个数必须一致
    assert kwargs["stage2_final_n_states"] == 25


def test_repro_mode_rejects_a_conflicting_final_state_count():
    with pytest.raises(RuntimeError) as exc:
        pl._freeze_stage2_path_for_repro({
            "stage2_uniform_lambda_states": 25,
            "stage2_final_n_states": 23,
            "stage2_window_ranges": [[0, 5], [4, 9], [8, 13], [12, 17], [16, 21], [20, 25]],
        }, lambda *_: None)
    assert "stage2_final_n_states" in str(exc.value)


def test_unset_repro_key_adds_no_field_to_the_preopt_fingerprint():
    """不开复现模式时，preopt 指纹必须**逐位不变**（否则全仓 pilot 重烧）。"""
    pipe = _fake_pipeline()
    derived = {k: None for k in pl._PREOPT_DERIVED_PATH_KEYS_ALL}
    key = _key(pipe, derived_path_params=derived)
    assert "stage2_uniform_lambda_states" not in key["payload"]
    derived["stage2_uniform_lambda_states"] = 25
    on = _key(pipe, derived_path_params=derived)
    assert on["payload"]["stage2_uniform_lambda_states"] == 25
    assert on != key


def test_repro_key_is_a_derived_layer_key_so_the_pilot_is_not_reburned():
    """它只决定布点 ⟹ 归派生层 ⟹ 打开它能从旧 pilot 离线重算，不重跑 GPU。"""
    pipe = _fake_pipeline()
    key = _key(pipe, derived_path_params={
        k: None for k in pl._PREOPT_DERIVED_PATH_KEYS_ALL} | {
        "stage2_uniform_lambda_states": 25})
    sampling, derived = pl._split_preopt_protocol_key(key["payload"])
    assert derived["stage2_uniform_lambda_states"] == 25
    assert "stage2_uniform_lambda_states" not in sampling


def test_config_key_is_forwarded_only_when_set():
    assert "stage2_uniform_lambda_states" not in runabfe._path_evolution_kwargs({})
    out = runabfe._path_evolution_kwargs({"stage2_uniform_lambda_states": 25})
    assert out["stage2_uniform_lambda_states"] == 25


# ---------------------------------------------------------------------------
# 控制器侧：布局冻结 + 固定长度采样（控制器仍然开着，只剩 f_k 那半）
# ---------------------------------------------------------------------------

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abfe_preoptimizer import Stage2RepairController  # noqa: E402
from test_stage2_repair_controller import _mkrun, R4, FULL  # noqa: E402

_LAYOUT_ACTIONS = ("INSERT_LAMBDA", "SPLIT_TAIL_WINDOW", "IMMUTABLE_REWINDOW")


def _frozen_board(tmp_path, *, frozen, windows=None, history=None):
    config = {"stage2_window_min_states": 4, "stage2_window_max_states": 8,
              "max_path_insertions": 3}
    if frozen:
        config["stage2_frozen_layout"] = True
    run = _mkrun(tmp_path, windows=windows or dict(FULL), ranges=R4,
                 n_states=13, config=config)
    if history is not None:
        with open(os.path.join(run, "checkpoints",
                               "stage2_autonomous_history.json"), "w") as handle:
            json.dump(history, handle)
    return Stage2RepairController(run, "vanishing")


def test_freeze_is_what_makes_layout_actions_infeasible(tmp_path):
    """两块盘面只差一个开关：冻结的那块三个布局动作全不可行。"""
    loose = _frozen_board(tmp_path / "loose", frozen=False)
    frozen = _frozen_board(tmp_path / "frozen", frozen=True)

    assert not any("复现模式" in (v or "") for v in loose.feasible().values())
    frozen_reasons = frozen.feasible()
    assert set(frozen_reasons) == {"insert_lambda", "split_tail_window"}
    assert all("复现模式" in (v or "") for v in frozen_reasons.values())
    # 有界重窗不吃插点预算、不需要 anchor，是唯一"对任何窗口都够得着"的缩跨度
    # 动作 —— 不单独堵住它，冻结就是空的。
    assert "复现模式" in (frozen.rewindow_feasible(frozen.read(), 0) or "")


def test_frozen_run_never_emits_a_layout_or_top_up_action(tmp_path):
    """已经烧过一块、窗口仍不够 ⟹ 既不许加帧也不许动布局，只剩换 f_k / 停下。"""
    windows = dict(FULL)
    windows[3] = {"K": 4, "self_verdict": "HARD_INSUFFICIENT",
                  "min_n_eff_over_g": 0.96, "n_decorr": 69}
    history = {"iterations": [{
        "iteration": 1, "action": "RUN_PRODUCTION", "path_version": 1,
        "windows": [3],
        "snapshot": [{"window_idx": 3, "production_steps": 250000,
                      "solver_n_decorrelated": 69, "min_n_eff_over_g": 0.96}],
    }]}
    ctl = _frozen_board(tmp_path, frozen=True, windows=windows, history=history)
    view = ctl.read()
    # 先钉住 fixture 本身被读懂了，否则下面测的是"历史压根没记上"。
    assert len((view.get("production_blocks_by_window") or {}).get(3) or []) == 1

    plan = ctl.decide(view)
    assert plan["action"] not in _LAYOUT_ACTIONS, plan["reason"]
    assert plan["action"] != "RUN_PRODUCTION", plan["reason"]


def test_repro_mode_rejects_windows_wider_than_the_configured_max():
    """26 态 / 5 窗 × 6 态 + max=5 ⟹ 开跑就炸，不许等 pilot 烧完才在 stage 2 门口炸。"""
    ranges = [[0, 6], [5, 11], [10, 16], [15, 21], [20, 26]]
    with pytest.raises(RuntimeError) as exc:
        pl._freeze_stage2_path_for_repro({
            "stage2_uniform_lambda_states": 26,
            "stage2_final_n_states": 26,
            "stage2_window_ranges": ranges,
            "stage2_window_min_states": 4,
            "stage2_window_max_states": 5,
        }, lambda *_: None)
    assert "stage2_window_max_states" in str(exc.value)

    ok = {
        "stage2_uniform_lambda_states": 26,
        "stage2_final_n_states": 26,
        "stage2_window_ranges": ranges,
        "stage2_window_min_states": 4,
        "stage2_window_max_states": 6,
    }
    assert pl._freeze_stage2_path_for_repro(ok, lambda *_: None) == 26
    assert pl.uniform_vanishing_lambda_path(26)[1] == pytest.approx(0.96, abs=1e-12)


def test_attachment_convergence_gates_are_wired_and_default_to_repro_off():
    """复现模式下 attachment 腿的收敛门只记录不阻断；显式键能覆盖两个方向。"""
    src = inspect.getsource(pl)
    i = src.index("stage0 = run_boresch_attachment_leg(")
    call = src[i:i + 2500]
    assert 'enforce_convergence_gates=_attach_enforce_gates,' in call, (
        "生产路径必须把这个开关透传下去，否则 config 怎么写都没用"
    )
    j = src.index("_attach_enforce_gates = bool(kwargs.get(")
    definition = src[j:j + 200]
    assert '"attachment_enforce_convergence_gates",' in definition
    assert "not _repro_uniform_states," in definition, (
        "不设键时：复现模式默认不阻断，其余情况默认阻断"
    )
    # CODE_REVIEW 2026-09-24 #1：resume 校验缓存也按**本次**开关，不按缓存自带的标记
    cache_check = src[j:i]
    assert "require_converged=_attach_enforce_gates," in cache_check
    assert 'cached_stage0.get("convergence_gates_enforced"' not in cache_check
    # 结果里必须留痕「这次是拦了还是只记录了」
    assert '"convergence_gates_enforced": bool(enforce_convergence_gates),' in \
        inspect.getsource(ie)
    # config 侧：设了才透传（无条件塞键会让既有缓存失配）
    assert "attachment_enforce_convergence_gates" not in runabfe._path_evolution_kwargs({})
    on = runabfe._path_evolution_kwargs(
        {"attachment_enforce_convergence_gates": True})
    off = runabfe._path_evolution_kwargs(
        {"attachment_enforce_convergence_gates": False})
    assert on["attachment_enforce_convergence_gates"] is True
    assert off["attachment_enforce_convergence_gates"] is False


def test_warmup_failure_fallthrough_is_debug_route_only():
    """预热不收敛改走临时生产，**只在 debug 路线**；标准生产路线逐字不变。"""
    engine_src = inspect.getsource(ie)
    # 引擎侧默认关闭
    assert "debug_provisional_on_warmup_failure: bool = False," in engine_src
    # 且必须与"调用方授权"**同时**成立才放行——不是单独一个开关就能绕过授权
    assert ("if provisional_authorized and debug_provisional_on_warmup_failure:"
            in engine_src)

    pipe_src = inspect.getsource(pl)
    i = pipe_src.index("debug_provisional_on_warmup_failure=bool(")
    assert 'kwargs.get("stage2_frozen_layout", False)' in pipe_src[i:i + 200], (
        "只有复现模式（debug 路线）才置位"
    )
    # 复现模式默认授权全部窗口，否则沿用调用方给的那份
    j = pipe_src.index("provisional_production_windows=(")
    block = pipe_src[j:j + 700]
    assert "list(range(len(_ranges)))" in block
    assert "_repro_uniform_states" in block


def test_softcore_power_lj_switch_defaults_to_production_exponents():
    """软核指数：不设 = (1,1)（09-24 起的生产默认，openfe 口径）；设了就以它为准，且进指纹。"""
    import abfe_core
    assert abfe_core.SOFTCORE_POWER_LJ_OVERRIDE is None
    assert abfe_core.ACESoftcorePotential.optimize_alpha(54)["power_lj"] == [1, 1]
    # 构造默认、from_dict 回退与 optimize_alpha 默认必须是同一份
    assert (abfe_core.ACESoftcorePotential().m_lj, abfe_core.ACESoftcorePotential().n_lj) == (1, 1)
    assert abfe_core.ACESoftcorePotential.from_dict({}).n_lj == 1
    try:
        abfe_core.SOFTCORE_POWER_LJ_OVERRIDE = [2, 2]
        assert abfe_core.ACESoftcorePotential.optimize_alpha(54)["power_lj"] == [2, 2]
    finally:
        abfe_core.SOFTCORE_POWER_LJ_OVERRIDE = None
    with pytest.raises(ValueError):
        abfe_core.ACESoftcorePotential.optimize_alpha(54, power_lj=[1, 9])
    # config 侧：设了才透传
    assert "softcore_power_lj" not in runabfe._path_evolution_kwargs({})
    assert runabfe._path_evolution_kwargs(
        {"softcore_power_lj": [1, 1]})["softcore_power_lj"] == [1, 1]


def test_boresch_force_constants_can_use_the_openfe_fixed_values():
    """Boresch 力常数：默认拟合不变；openfe_fixed 档给固定 4184 / 334.72。"""
    import abfe_core
    est = abfe_core.OrbBoreschEstimator.__new__(abfe_core.OrbBoreschEstimator)
    est.config = {"boresch_force_constant_mode": "openfe_fixed"}
    ks = est._apply_hybrid_filter(None, None)
    assert ks["kr"] == pytest.approx(4184.0)
    for name in ("kthetaA", "kthetaB", "kphiA", "kphiB", "kphiC"):
        assert ks[name] == pytest.approx(334.72)
    # 我们拟合档的角力常数上限只有 100 —— 比 openfe 的固定值还低 3.3 倍，
    # 这条钉住"为什么鬼影摆得动"的量纲事实。
    assert abfe_core.UnitConstants.FORCE_CONSTANT_KANGLE_RANGE[1] >= 100.0


def test_attachment_converged_false_is_carried_not_re_blocked():
    """门自己关掉的那一跑，下游不能再拿 converged=False 把 run 拦死。"""
    import abfe_core
    payload = {"converged": False, "convergence_gates_enforced": False,
               "attachment_delta_G_kJ_mol": 23.97, "attachment_error_kJ_mol": 0.32}
    # [09-23] 默认读 payload 自带的 convergence_gates_enforced；门开着（生产路线）照样硬判
    with pytest.raises(abfe_core.FinalResultValidationError):
        abfe_core.validate_boresch_attachment_result(
            {**payload, "convergence_gates_enforced": True})
    with pytest.raises(abfe_core.FinalResultValidationError):
        abfe_core.validate_boresch_attachment_result(payload, require_converged=True)
    # 显式放宽时只放宽 converged 这一项
    dg, err = abfe_core.validate_boresch_attachment_result(
        payload, require_converged=False)
    assert dg == pytest.approx(23.97) and err == pytest.approx(0.32)
    # ΔG/σ 的有限性照常硬判 —— 这条不是"门太严"，是数不成立
    for bad in ({**payload, "attachment_delta_G_kJ_mol": float("nan")},
                {**payload, "attachment_error_kJ_mol": -1.0}):
        with pytest.raises(abfe_core.FinalResultValidationError):
            abfe_core.validate_boresch_attachment_result(bad, require_converged=False)
    # pipeline 侧按 convergence_gates_enforced 决定放不放宽
    src = inspect.getsource(pl)
    i = src.index("attachment_dg, attachment_err = validate_boresch_attachment_result(")
    assert "require_converged=not _gates_off" in src[i:i + 200]


def test_mid_accumulation_budget_exhaustion_also_goes_provisional_in_repro():
    """[2026-09-24] 分档累计途中步数先耗尽 ⟹ 复现模式也进临时生产（不再跳窗口、
    合并求解缺窗口必炸）；诊断的已耗步数含本次 attempt，不再报 0。"""
    src = inspect.getsource(ie)
    i = src.index("_consumed_now = warmup_ledger_total_steps(_ledger) + sum(")
    block = src[i:i + 5000]
    assert '"cumulative_consumed_steps": _consumed_now,' in block
    assert ("provisional_authorized\n                    and debug_provisional_on_warmup_failure\n"
            "                    and frozen_f_k_snapshot is not None" in block), (
        "授权 + debug 路线 + 有冻结候选，三条同时成立才放行"
    )
    assert "_attempt_ledger.values()" in block[:200]
