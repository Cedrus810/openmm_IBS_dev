"""用生产帧重解 f_k 的离线契约测试（无 GPU）。

背景：warmup 的 loose gate 只要求 max|Δf−ΔF^MBAR| < 10 kJ/mol 且只看 200 帧，
留下的 f_k 可能偏到让混合系综塌向少数态；在这种偏置下加帧只是往同一个偏斜分布里
加更多帧（实测两轮 rescue：绝对样本数涨、ESS 比值不动）。
"""

import inspect

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only
pytest.importorskip("openmm")

import ibs_engine as ie


KT = 2.494
F_TRUE = np.array([0.0, 2.0, 5.0, 9.0, 14.0])
CENTERS = np.arange(F_TRUE.size) * 1.0
WIDTH = 0.7


def _draw(n, f_k, seed):
    """从以 f_k 为偏置的混合系综采 n 帧，返回 (u_kn, bias, base)。"""
    rng = np.random.default_rng(seed)
    w = np.exp(-(F_TRUE - np.asarray(f_k)) / KT)
    w /= w.sum()
    x = rng.normal(CENTERS[rng.choice(F_TRUE.size, size=n, p=w)], WIDTH)
    u = np.array([
        0.5 * ((x - CENTERS[k]) / WIDTH) ** 2 * KT + F_TRUE[k]
        for k in range(F_TRUE.size)
    ])
    bias = -KT * np.log(np.sum(np.exp(-(u - np.asarray(f_k)[:, None]) / KT), axis=0))
    return u, bias, np.zeros(n)


def _centered(v):
    v = np.asarray(v, dtype=float)
    return v - v.mean()


def test_recalibration_moves_f_k_towards_the_true_free_energies():
    """冷启动 f_k=0 下采的帧，应当把 f_k 拉向真实自由能。"""
    f_cold = np.zeros(F_TRUE.size)
    u, bias, base = _draw(4000, f_cold, 11)
    out = ie.recalibrate_f_k_from_production(
        u, bias, base, list(range(F_TRUE.size)), KT, f_cold
    )
    assert out["error"] is None
    before = np.max(np.abs(_centered(f_cold) - _centered(F_TRUE)))
    after = np.max(np.abs(_centered(out["f_k"]) - _centered(F_TRUE)))
    assert after < before / 3.0, f"重标定后仍偏 {after:.2f}（原来 {before:.2f}）"
    assert out["max_adjacent_shift_kJ_mol"] > 1.0, "冷启动下这次重标定应当是实质性的"


def test_recalibration_is_a_near_noop_when_f_k_is_already_right():
    """f_k 已经对了就不该乱动 —— 否则每轮 rescue 都会白白扰动一个好偏置。"""
    u, bias, base = _draw(4000, F_TRUE, 12)
    out = ie.recalibrate_f_k_from_production(
        u, bias, base, list(range(F_TRUE.size)), KT, F_TRUE.copy()
    )
    assert out["error"] is None
    assert out["max_adjacent_shift_kJ_mol"] < 0.5, (
        f"f_k 已正确时改动过大：{out['max_adjacent_shift_kJ_mol']:.3f} kJ/mol"
    )


def test_result_is_mean_centered_so_the_gauge_is_fixed():
    """f_k 的公共常数是规范自由度；跟 _apply_pairwise_cap 取同一个规范（均值零）。"""
    f_cold = np.zeros(F_TRUE.size)
    u, bias, base = _draw(2000, f_cold, 13)
    out = ie.recalibrate_f_k_from_production(
        u, bias, base, list(range(F_TRUE.size)), KT, f_cold
    )
    assert abs(float(np.mean(out["f_k"]))) < 1e-9


def test_unsolvable_input_fails_closed_without_returning_an_f_k():
    """解不出来就不给 f_k —— 绝不能返回一个猜的偏置去驱动几十万步采样。"""
    f_cold = np.zeros(F_TRUE.size)
    u, bias, base = _draw(3, f_cold, 14)          # 帧数远低于 min_frames
    out = ie.recalibrate_f_k_from_production(
        u, bias, base, list(range(F_TRUE.size)), KT, f_cold
    )
    assert out["error"] is not None
    assert out["f_k"] is None


def test_naive_concatenation_of_two_segments_is_biased():
    """两段直接摞 bias_energies 当一个采样态是错的 —— 锁住这条，别有人图省事这么合。

    正确做法是用 u_kn 对两个 f_k 各算一次 mixture 能量把交叉项补齐。
    """
    from scipy.special import logsumexp
    from pymbar import MBAR

    def mixture_energy(u, f):
        return -KT * logsumexp(-(u - np.asarray(f)[:, None]) / KT, axis=0)

    def delta_g(u_aug, N_k, n_sampling_states):
        f = MBAR(u_aug / KT, N_k, verbose=False).compute_free_energy_differences()
        f = f["Delta_f"][0] * KT
        return f[n_sampling_states + F_TRUE.size - 1] - f[n_sampling_states]

    f_cold = np.zeros(F_TRUE.size)
    n = 3000
    u1, _, _ = _draw(n, f_cold, 21)
    u2, _, _ = _draw(n, F_TRUE, 22)
    both = np.hstack([u1, u2])

    naive = np.vstack([
        np.concatenate([mixture_energy(u1, f_cold), mixture_energy(u2, F_TRUE)]),
        both,
    ])
    proper = np.vstack([
        mixture_energy(both, f_cold), mixture_energy(both, F_TRUE), both,
    ])
    truth = F_TRUE[-1] - F_TRUE[0]
    err_naive = abs(delta_g(naive, [2 * n] + [0] * F_TRUE.size, 1) - truth)
    err_proper = abs(delta_g(proper, [n, n] + [0] * F_TRUE.size, 2) - truth)
    assert err_proper < err_naive, (
        f"补齐交叉能量应当更准：proper={err_proper:.3f} naive={err_naive:.3f}"
    )


# ---------------------------------------------------------------------------
# 注入口：重解出的 f_k 作为下一段预热的热启动种子
# ---------------------------------------------------------------------------

def test_run_all_windows_accepts_a_per_window_seed():
    sig = inspect.signature(ie.IBSWindowManagerDualLambda.run_all_windows)
    assert "initial_f_k_by_window" in sig.parameters
    assert sig.parameters["initial_f_k_by_window"].default is None, (
        "必须默认 None —— 不传时行为逐字不变"
    )


def test_explicit_seed_takes_precedence_over_pilot_ti_in_source():
    """显式种子优先于 pilot-TI，且两条分支都走同一个长度校验。"""
    src = inspect.getsource(ie.IBSWindowManagerDualLambda.run_all_windows)
    assert "initial_f_k_by_window.get(int(window_idx))" in src
    explicit = src.index("_explicit_seed is not None")
    pilot = src.index("estimate_f_k_from_pilot_ti(")
    assert explicit < pilot, "显式种子必须先判，pilot-TI 只是 else 分支"
    # 种子只是起点，不得跳过冻结验证：注入点仍在 `if not is_resumed_ibs` 的预热段内。
    assert "f_k_warm_started = True" in src


def test_gauge_mismatch_fails_closed():
    """u 与 bias 不同规范时必须拒绝返回 f_k。

    真实产物实测：`energies.npy` 比 `sampling_states.npy` 多一个**逐 λ 态常数**
    （LJ 长程尾项，在分析侧目标能量里、不在采样哈密顿量里）。逐态常数不是共模，
    会改变 logsumexp 的形状 —— 用 energies 对账得 sd=0.735/0.373/0.182/0.094
    （cyclod rep1 四个窗口，随解耦单调变小就是尾项本身），用 sampling_states 得 0.0000。
    在错规范下照样能解出一个"看起来正常"的 f_k，所以这里必须 fail-closed。
    """
    f_true = F_TRUE.copy()
    u, bias, base = _draw(2000, f_true, 31)

    good = ie.recalibrate_f_k_from_production(
        u, bias, base, list(range(F_TRUE.size)), KT, f_true
    )
    assert good["error"] is None
    assert good["f_k_consistency_sd_kJ_mol"] < 1e-6

    # 给每个态加一个**不同**的常数：模仿 energies 相对 sampling_states 的尾项。
    per_state_tail = np.array([0.0, 0.4, 0.9, 1.6, 2.5])[:, None]
    bad = ie.recalibrate_f_k_from_production(
        u + per_state_tail, bias, base, list(range(F_TRUE.size)), KT, f_true
    )
    assert bad["error"] == "f_k_bias_gauge_mismatch"
    assert bad["f_k"] is None

    # 共模常数是规范自由度，不该被误杀。
    ok_common = ie.recalibrate_f_k_from_production(
        u + 7.0, bias, base, list(range(F_TRUE.size)), KT, f_true + 7.0
    )
    assert ok_common["error"] is None


# ---------------------------------------------------------------------------
# 集成回归：段 2 被采纳后不得被 bridge rescue 悄悄退回段 1
# ---------------------------------------------------------------------------

def test_bridge_rescue_collects_every_sampling_segment():
    """bridge rescue 只重载**一个**目录就会丢段。

    历史：段 2 被采纳后它从写死的 `output_dir/vanishing` 重载（丢段 2）；把路径
    改指段 2 之后又变成丢段 1。正确做法是先按窗口收集全部段，再套窗口替换规则。
    """
    import abfe_pipeline

    src = inspect.getsource(abfe_pipeline.ABFEPipeline.run_full_pipeline)
    load = src.index("original_outputs = self._load_ibs_window_outputs_merged(")
    window = src[load:load + 300]
    assert "_vanishing_segment_dirs" in window, "必须收集全部采样段"
    assert 'os.path.join(self.output_dir, "vanishing")' not in window
    # 段 2 产生时必须登记进段目录清单，否则收集不到它。
    assert "_vanishing_segment_dirs.append((" in src


def test_second_segment_is_merged_not_adopted():
    """采纳段 2 等于扔掉段 1 的帧（实测 w1/w2/w3 里 2/3 到 4/5 的 ESS）。"""
    import abfe_pipeline

    src = inspect.getsource(abfe_pipeline.ABFEPipeline.run_full_pipeline)
    assert "_load_ibs_window_outputs_merged(" in src
    assert '_seg_diag["analysis_mode"] = "merged_segments"' in src
    assert "stage2 = _seg_result" not in src, "不得再直接采纳段 2"


def test_numerical_merge_failure_degrades_but_input_error_does_not():
    """数值失败可降级，输入错误必须继续 fail-closed —— 两类共用一个 except
    就等于把静默错误重新埋回去。"""
    import abfe_pipeline
    import multi_segment_analysis as msa

    import ast
    import textwrap

    src = inspect.getsource(abfe_pipeline.ABFEPipeline.run_full_pipeline)
    tree = ast.parse(textwrap.dedent(src))
    caught = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is not None:
            for t in (node.type.elts if isinstance(node.type, ast.Tuple)
                      else [node.type]):
                caught.add(ast.unparse(t).split(".")[-1])
    assert "MultiSegmentSolveError" in caught, "数值失败必须被捕获以便降级"
    assert "MultiSegmentInputError" not in caught, (
        "输入错误绝不能被捕获降级 —— 那等于把静默错误重新埋回去"
    )
    assert '"independent_segments_fallback"' in src
    assert '_seg_diag["merge_succeeded"] = False' in src
    assert not issubclass(msa.MultiSegmentInputError, msa.MultiSegmentSolveError)


def test_recalibration_diagnostics_survive_the_bridge_rescue_overwrite():
    """bridge rescue 会整个重赋值 stage2；诊断不跟着搬就等于机制不可审计。"""
    import abfe_pipeline

    src = inspect.getsource(abfe_pipeline.ABFEPipeline.run_full_pipeline)
    assert "if _fk_recalibration_diag is not None:" in src
    assert src.index("_fk_recalibration_diag = _seg_diag") < src.index(
        "if _fk_recalibration_diag is not None:"
    )


def test_policy_switches_do_not_invalidate_existing_window_caches():
    """这几个键是执行策略，不决定某个已落盘窗口采的是什么。

    留在窗口采样身份里的后果：任何人一打开开关就让全部已有窗口缓存失配、静默重采。
    """
    base = ie._stage_window_sampling_identity(_stage_key())
    for key, value in (
        ("stage2_recalibrate_f_k_on_rescue", True),
        ("stage2_f_k_recalibration_min_shift", 1.5),
        ("max_path_insertions", 5),
        ("sampling_repair_policy", "path_evolution_v1"),
    ):
        assert ie._stage_window_sampling_identity(_stage_key(**{key: value})) == base, (
            f"{key} 不该让已采好的窗口轨迹失配"
        )


def _stage_key(**overrides):
    kwargs = {"final_min_ess_ratio": 0.1}
    kwargs.update(overrides)
    return {"payload": {
        "stage_name": "vanishing", "potential_type": "aces",
        "run_config": {"n_steps_per_window": 500000, "kwargs": kwargs},
        "code_sha256": "deadbeef",
    }}


def test_segment_two_inherits_the_accumulated_production_budget():
    """段 2 退回基础预算有两个后果，第二个更隐蔽。

    直接的是扔掉段 1 攒的帧；隐蔽的是**更短的序列让新段在 g/ESS 诊断上系统性
    显得更好**——实测段 1 的 w3 全长 1000 帧 g=202.9，自己截到 500 帧变成
    13.9/31.0，与段 2 的 27.8 同一档。不继承预算等于内建一个自我恭维的偏差。
    """
    import abfe_pipeline

    method = inspect.getsource(
        abfe_pipeline.ABFEPipeline._recalibrate_f_k_and_resample_segment
    )
    assert "production_step_overrides" in inspect.signature(
        abfe_pipeline.ABFEPipeline._recalibrate_f_k_and_resample_segment
    ).parameters
    assert "_production_step_overrides=(" in method

    src = inspect.getsource(abfe_pipeline.ABFEPipeline.run_full_pipeline)
    assert "production_step_overrides=dict(production_rescue_targets)," in src, (
        "调用点必须把已累积的 rescue 目标传下去"
    )


def test_segment_two_only_reruns_the_reseeded_windows():
    """判定完就知道是哪几个窗口，没被重播种的不该从零重采。

    第一次真机跑把四个窗口全重跑了，包括 w3（位移没过阈值、根本没重播种）和
    w0（位移 0.579 最小、g=5.9 最好）。同样的 GPU 花在 w3（g=202.9，去相关后
    只剩 5 帧）才是刀刃。
    """
    import abfe_pipeline

    method = inspect.getsource(
        abfe_pipeline.ABFEPipeline._recalibrate_f_k_and_resample_segment
    )
    assert "_only_window_indices=sorted(seeds)" in method


def test_only_window_indices_defaults_to_running_everything():
    sig = inspect.signature(ie.IBSWindowManagerDualLambda.run_all_windows)
    assert sig.parameters["only_window_indices"].default is None
    src = inspect.getsource(ie.IBSWindowManagerDualLambda.run_all_windows)
    assert "if _only is not None and window_idx not in _only:" in src


def test_unknown_window_index_fails_closed():
    """写错窗口号必须报错，不能静默少跑一个窗口。"""
    src = inspect.getsource(ie.IBSWindowManagerDualLambda.run_all_windows)
    assert "only_window_indices 含不存在的窗口" in src
    assert "拒绝静默忽略" in src
