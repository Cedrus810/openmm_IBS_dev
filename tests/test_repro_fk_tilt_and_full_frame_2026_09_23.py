"""复现节奏：左/中/右倾斜 + 重标定点估计用全帧 + REPRO-01 转发（2026-09-23）。"""
import inspect
import os

import numpy as np

os.environ.setdefault("PYMBAR_DISABLE_JAX", "1")

import abfe_pipeline as ap
import ibs_engine as ie


def test_tilt_uses_module_margin_and_moves_endpoint_gap():
    kt = 2.479
    learned = np.array([-6.0, -3.9, -1.7, 0.7, 3.7, 7.2])      # 递增（win3 段1 学到的）
    anchor = np.array([-1.55, 0.11, 1.63, 2.25, 0.99, -3.43])   # 非单调（pilot）
    gap = abs((learned[-1] - learned[0]) - (anchor[-1] - anchor[0])) / kt
    for sign in (+1, -1):
        f, delta, g = ie.repro_fk_tilt(learned, anchor, kt, sign)
        assert np.isclose(g, gap)
        # 断言的是**模块里那个常数**，谁改了余量这里就红。
        assert np.isclose(delta, gap + ie.IBS_REPRO_TILT_MARGIN_KT)
        assert ie.IBS_REPRO_TILT_MARGIN_KT == 2.0
        ep_shift = ((f[-1] - f[0]) - (learned[-1] - learned[0])) / kt
        assert np.isclose(ep_shift, -sign * delta)
        # 均值不动（规范不变），只改形状。
        assert np.isclose(f.mean(), learned.mean())


def test_local_mbar_point_estimate_all_frames_keeps_every_frame():
    rng = np.random.default_rng(0)
    n, K, kt = 400, 3, 2.479
    # AR(1) 相关序列 ⟹ g 明显 > 1，默认路径会抽帧。
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.95 * x[i - 1] + rng.normal()
    u = np.vstack([x * k for k in (0.0, 0.5, 1.0)])
    f = np.zeros(K)
    bias = -kt * np.log(np.exp(-(u - f[:, None]) / kt).sum(0))
    base = np.zeros(n)
    kw = dict(f_k=f, min_frames=5, sampling_kj=u)
    strided = ie._solve_single_window_local_mbar(u, bias, base, [0, 1, 2], kt, **kw)
    full = ie._solve_single_window_local_mbar(
        u, bias, base, [0, 1, 2], kt, point_estimate_all_frames=True, **kw)
    assert strided.get("error") is None and full.get("error") is None
    assert strided["n_frames_used"] < n
    assert full["n_frames_used"] == n


def test_recalibration_forwards_full_frame_flag():
    assert "point_estimate_all_frames" in inspect.signature(
        ie.recalibrate_f_k_from_production).parameters
    src = inspect.getsource(ap.ABFEPipeline._recalibrate_f_k_and_resample_segment)
    assert "point_estimate_all_frames=bool(point_estimate_all_frames)" in src


def test_new_repro_keys_do_not_enter_stage_fingerprint():
    rc = {"kwargs": {"stage2_repro_fk_cycles": 3,
                     "stage2_repro_fk_tilt_signs": [1, 0, -1], "x": 1}}
    assert ap._strip_non_identity_kwargs(rc)["kwargs"] == {"x": 1}


def test_repro01_frozen_layout_and_tilt_are_forwarded():
    src = inspect.getsource(ap.ABFEPipeline)
    # _run_stage2_once 的具名转发清单里必须有这两个键，否则读侧恒取默认值。
    assert 'stage2_frozen_layout=kwargs.get("stage2_frozen_layout", False)' in src
    assert "repro_fk_tilt_sign=_repro_fk_tilt_sign" in src
    assert 'repro_fk_tilt_sign=kwargs.get("repro_fk_tilt_sign")' in src
    assert "repro_fk_tilt_sign" in inspect.signature(
        ie.IBSWindowManagerDualLambda.run_all_windows).parameters


def test_repro_default_is_three_cycles_left_middle_right():
    assert ap.REPRO_FK_CYCLES_DEFAULT == 3
    assert tuple(ap.REPRO_FK_TILT_SIGNS_DEFAULT) == (1, 0, -1)
    assert len(ap.REPRO_FK_TILT_SIGNS_DEFAULT) == ap.REPRO_FK_CYCLES_DEFAULT


def test_debug_provisional_branch_sets_sampler_state():
    # 该分支在 `if bias_converged:` 的 else 里；只改局部变量时，进生产前的
    # `if not sampler.bias_converged: raise` 必炸（真机 09-23 两次）。
    src = inspect.getsource(ie.IBSWindowManagerDualLambda.run_all_windows)
    i = src.index("if provisional_authorized and debug_provisional_on_warmup_failure:")
    branch = src[i:src.index("raise IBSWarmupConvergenceError(", i)]
    for needle in ("sampler.bias_converged = True",
                   'sampler.bias_status = "provisional_production"',
                   "sampler.f_k_evidence_status = F_K_EVIDENCE_INDETERMINATE",
                   "sampler.save_ibs_state("):
        assert needle in branch, needle


def test_tilt_ignores_anchor_on_lambda_zero_window():
    kt = 2.479
    learned = np.array([13.0, 7.0, 0.6, -4.5, -7.8, -9.2])
    lam_end = [0.2, 0.16, 0.12, 0.08, 0.04, 0.0]
    # 锚点给 None 也不崩；Δ 只剩余量。
    f, delta, gap = ie.repro_fk_tilt(learned, None, kt, -1, lambdas_vdw=lam_end)
    assert gap == 0.0 and np.isclose(delta, ie.IBS_REPRO_TILT_MARGIN_KT)
    # 不含 0 的窗口照旧用锚点。
    anchor = learned + np.linspace(0, 10, 6)
    _, delta2, gap2 = ie.repro_fk_tilt(
        learned, anchor, kt, -1, lambdas_vdw=[0.4, 0.36, 0.32, 0.28, 0.24, 0.2])
    assert np.isclose(gap2, 10 / kt) and np.isclose(delta2, gap2 + ie.IBS_REPRO_TILT_MARGIN_KT)
    src = inspect.getsource(ie.IBSWindowManagerDualLambda.run_all_windows)
    assert "lambdas_vdw=lv_win" in src
