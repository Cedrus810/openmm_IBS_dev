"""stage 2 轮次控制器：证据层、执行层（账本 / resume / 失败 / 预算）、身份、透传、引擎钩子。

设计 docs/design/PROPOSAL_stage2_lcr_rounds_2026-09-24.md；计划
docs/design/PLAN_stage2_controller_rewrite_2026-09-24.md（第 2–4 步）。

执行层用一个假引擎：按账本给的 f_k 从**可精确采样**的合成模型里出帧并写成真实文件格式
（`dual_window_*_vdw_{sampling_states,energies}.npy` + `ibs_state_vdw_window_*.json`）。
合成模型：约化能量 u_k(x) = (x − μ_k)²/2 + c_k/kT，各态配分函数相同 ⟹ F_k = c_k + 常数；
IBS 混合 p_f(x) ∝ Σ_k exp(f_k − u_k(x)) 等于「按 exp((f_k − c_k)/kT) 选分量再 x ~ N(μ_k, 1)」。
"""

import inspect
import json
import os

import numpy as np
import pytest

pytest.importorskip("pymbar")

import abfe_pipeline as ap
import runabfe
import stage2_rounds as sr
import stage2_rounds_evidence as sre

KT = 2.494


def _synth(c_kJ, f_kJ, n, rng, spacing=1.0):
    c = np.asarray(c_kJ, dtype=float)
    f = np.asarray(f_kJ, dtype=float)
    k = c.size
    mu = np.arange(k) * spacing
    p = np.exp((f - c) / KT - np.max((f - c) / KT))
    p /= p.sum()
    comp = rng.choice(k, size=n, p=p)
    x = rng.normal(mu[comp], 1.0)
    u_red = 0.5 * (x[None, :] - mu[:, None]) ** 2 + (c / KT)[:, None]
    return u_red * KT                                      # (K, N) kJ


# ---------------------------------------------------------------------------
# 证据层
# ---------------------------------------------------------------------------

def test_evidence_recovers_free_energies_and_is_the_msa_solution():
    rng = np.random.default_rng(1)
    c = np.array([0.0, 1.5, 2.0, 4.0]) * KT
    fs = [sr.tilt_f_k(c, KT, s) for s in (0, 1, -1)]
    segs = [_synth(c, f, 600, rng) for f in fs]
    nums = sre.window_evidence_numbers(segs, segs, fs, KT, first_round_segments=[0, 1, 2])
    assert nums["delta_f_kJ"] == pytest.approx(c[-1] - c[0], abs=4 * nums["sigma_mbar_all_frames_kJ"] + 0.3)
    assert np.allclose(nums["merged_f_k_kJ"], c - c[0], atol=1.0)
    assert len(nums["loo_delta_f_kJ"]) == 3 and all(np.isfinite(nums["loo_delta_f_kJ"]))
    assert min(nums["state_ess"]) > 20
    assert 0.0 < nums["top1pct_weight"] < 0.35
    # 抬首态的那段（下标 1）对首态的 ESS 贡献大于抬末态的那段（下标 2）
    seg_ess = nums["first_round_segment_state_ess"]
    assert seg_ess[1][0] > seg_ess[2][0] and seg_ess[2][-1] > seg_ess[1][-1]
    # 与生产合并求解同一份实现
    import multi_segment_analysis as msa
    cross, src = msa.build_cross_bias(segs, fs, KT)
    ref = msa.collapse_segments(cross, src, np.hstack(segs), KT)
    assert nums["delta_f_kJ"] == pytest.approx(ref["target_delta_f_kJ_mol"][0, -1])


def test_adjacent_overlap_is_small_only_when_states_are_far_apart():
    rng = np.random.default_rng(2)
    c = np.zeros(4)
    far = [_synth(c, c, 800, rng, spacing=8.0)]
    near = [_synth(c, c, 800, rng, spacing=0.3)]
    o_far = sre.window_evidence_numbers(far, far, [c], KT)["adjacent_overlap"]
    o_near = sre.window_evidence_numbers(near, near, [c], KT)["adjacent_overlap"]
    assert max(o_far) < 0.05 and min(o_near) > 0.5


def test_single_segment_window_uses_within_segment_block_jackknife():
    """09-25：溶剂腿第一轮只一段 ⟹ 段内切三块做 jackknife（设计 §11.5）。"""
    rng = np.random.default_rng(3)
    c = np.array([0.0, 1.0, 2.0])
    seg = [_synth(c, c, 300, rng)]
    nums = sre.window_evidence_numbers(seg, seg, [c], KT)
    assert nums["jackknife_kind"] == "within_segment_blocks"
    assert len(nums["loo_delta_f_kJ"]) == sre.WITHIN_SEGMENT_BLOCKS == 3
    assert sr.jackknife_sigma(nums["loo_delta_f_kJ"]) is not None
    two = [_synth(c, c, 300, rng), _synth(c, c, 300, rng)]
    assert sre.window_evidence_numbers(two, two, [c, c], KT)["jackknife_kind"] == "segments"


def test_block_jackknife_sees_a_segment_that_switches_basin_midway():
    """段内换盆（溶剂腿 w3 灌水双稳态那种）⟹ 块间 ΔF 差大、σ_jk 大。"""
    rng = np.random.default_rng(4)
    a = _synth([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], 300, rng)
    b = _synth([0.0, 6.0, 12.0], [0.0, 1.0, 2.0], 300, rng)
    mixed = [np.hstack([a, b[:, :150], a[:, :150]])]
    steady = [np.hstack([a, a])]
    s_mixed = sr.jackknife_sigma(sre.window_evidence_numbers(mixed, mixed, [[0, 1, 2]], KT)["loo_delta_f_kJ"])
    s_steady = sr.jackknife_sigma(sre.window_evidence_numbers(steady, steady, [[0, 1, 2]], KT)["loo_delta_f_kJ"])
    assert s_mixed > 3 * max(s_steady, 0.1)


# ---------------------------------------------------------------------------
# 执行层（假引擎）
# ---------------------------------------------------------------------------

LAM = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
RANGES = [(0, 3), (2, 4), (3, 6)]              # 共享边界态；末窗含 λ=0
PILOT_L = list(np.linspace(1.0, 0.0, 11))
PILOT_M = [float(-10.0 + 20.0 * (1 - x)) for x in PILOT_L]


class _Stub:
    _STAGE2_ROUNDS_LEDGER = ap.ABFEPipeline._STAGE2_ROUNDS_LEDGER
    _STAGE2_ROUNDS_LEDGER_VERSION = ap.ABFEPipeline._STAGE2_ROUNDS_LEDGER_VERSION
    _stage2_segment_dirs = staticmethod(ap.ABFEPipeline._stage2_segment_dirs)
    _stage2_rounds_ledger_save = staticmethod(ap.ABFEPipeline._stage2_rounds_ledger_save)
    _stage2_spec_to_dict = staticmethod(ap.ABFEPipeline._stage2_spec_to_dict)
    _stage2_rounds_ledger_load = ap.ABFEPipeline._stage2_rounds_ledger_load
    _stage2_rounds_load_window = ap.ABFEPipeline._stage2_rounds_load_window
    _run_stage2_rounds = ap.ABFEPipeline._run_stage2_rounds

    def __init__(self, tmp, leg="complex"):
        self.output_dir = str(tmp)
        self.checkpoint_dir = os.path.join(str(tmp), "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.leg_name = leg
        self.logs = []

    def _log(self, msg):
        self.logs.append(msg)

    def _stage2_rounds_finalize(self, ledger, ledger_path, *args):
        return {"ledger": ledger, "ledger_path": ledger_path}


def _true_c(w):
    """每窗的真实 F（kJ）。w1 故意离锚点远，让第一轮有态缺帧。"""
    return {0: [0.0, 1.0, 2.0], 1: [0.0, 25.0], 2: [0.0, 1.0, 1.5]}[w]   # w1 离锚点 ~10 kT


class _FakeEngine:
    def __init__(self, tmp, n=400, fail=(), wrong_f=False):
        self.tmp = str(tmp)
        self.calls = []
        self.n = n
        self.fail = set(fail)
        self.wrong_f = wrong_f

    def __call__(self, n_states, lam, ranges, *, _output_dir_override, _checkpoint_dir_override,
                 _only_window_indices, _authoritative_window_ranges, _resume_override,
                 _fixed_f_k_by_window, _segment_seed_tag):
        (w,) = _only_window_indices
        seg = int(_segment_seed_tag.replace("seg", ""))
        self.calls.append((seg, w))
        if (seg, w) in self.fail:
            raise RuntimeError("NaN in 窗口（假）")
        out = _output_dir_override or os.path.join(self.tmp, "vanishing")
        ck = _checkpoint_dir_override or os.path.join(self.tmp, "checkpoints")
        os.makedirs(out, exist_ok=True)
        os.makedirs(ck, exist_ok=True)
        f = np.asarray(_fixed_f_k_by_window[w], dtype=float)
        rng = np.random.default_rng(1000 * seg + w)
        # w1 两态隔得开（间距 3σ），再加上 F 离锚点 ~10 kT ⟹ 第一轮末态必然缺帧
        u = _synth(_true_c(w), f, self.n, rng, spacing=3.0 if w == 1 else 1.0)
        np.save(os.path.join(out, f"dual_window_{w}_vdw_energies.npy"), u)
        np.save(os.path.join(out, f"dual_window_{w}_vdw_sampling_states.npy"), u.T)
        f_rec = (f + 1.0) if self.wrong_f else f
        with open(os.path.join(ck, f"ibs_state_vdw_window_{w}.json"), "w") as fh:
            json.dump({"production_entry_f_k": [float(x) for x in f_rec]}, fh)
        return {}


def _run(stub, engine, **kw):
    return stub._run_stage2_rounds(
        engine, lambdas_var=LAM, window_ranges=RANGES, kt=KT,
        pilot_lambdas=kw.pop("pilot_lambdas", PILOT_L), pilot_mean_dU_dlambda=kw.pop("pilot_mean", PILOT_M),
        pilot_std_dU_dlambda=None, pilot_n_dU_dlambda_samples=None,
        n_steps_per_window=250000, run_kwargs=kw)


def test_round_one_is_written_to_the_ledger_before_any_sampling(tmp_path):
    stub = _Stub(tmp_path)

    def boom(*a, **k):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        _run(stub, boom)
    with open(os.path.join(stub.checkpoint_dir, "stage2_rounds_ledger.json")) as fh:
        led = json.load(fh)
    r1 = led["rounds"][0]
    assert r1["round"] == 1 and len(r1["specs"]) == 3 * len(RANGES) and r1["status"] == {}
    assert sorted({s["segment"] for s in r1["specs"]}) == [1, 2, 3]
    # 段 1 = 中（不倾斜），2 = 左，3 = 右
    by_seg = {s["segment"]: s["tilt_sign"] for s in r1["specs"] if s["window"] == 0}
    assert by_seg == {1: 0, 2: 1, 3: -1}


def test_end_to_end_window_major_and_later_rounds_untilted(tmp_path):
    stub = _Stub(tmp_path)
    eng = _FakeEngine(tmp_path)
    out = _run(stub, eng)
    led = out["ledger"]
    # 第一轮按窗口顺序：w0 的三段 → w1 的三段 → …
    assert eng.calls[:9] == [(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)]
    assert 2 <= len(led["rounds"]) <= sr.MAX_ROUNDS
    # w1 第一轮缺帧 ⟹ 第二轮被点名，理由是缺帧；w0 / w2 不补
    r2 = led["rounds"][1]
    assert [s["window"] for s in r2["specs"]] == [1]
    assert r2["specs"][0]["reason"].startswith("starved_state_")
    assert led["rounds"][0]["labels"]["1"][0] == sr.LABEL_PROVISIONAL
    assert led["rounds"][0]["labels"]["0"][0] == sr.LABEL_CONVERGED
    for r in led["rounds"][1:]:
        assert all(s["tilt_sign"] == 0 and s["center_source"].startswith("merged") for s in r["specs"])
        assert len({s["segment"] for s in r["specs"]}) == 1          # 每轮一个段
    for r in led["rounds"]:
        assert set(r["labels"]) == {"0", "1", "2"} and r["evidence"] is not None
    with open(os.path.join(str(tmp_path), "vanishing_4", "segment_spec.json")) as fh:
        seg4 = json.load(fh)
    assert seg4["segment"] == 4 and set(seg4["windows"]) == {"1"}


def test_resume_does_not_replan_or_resample(tmp_path):
    stub = _Stub(tmp_path)
    out1 = _run(stub, _FakeEngine(tmp_path))

    def must_not_run(*a, **k):
        raise AssertionError("resume 不应再采任何段")
    out2 = _run(_Stub(tmp_path), must_not_run)
    assert [r["specs"] for r in out2["ledger"]["rounds"]] == [r["specs"] for r in out1["ledger"]["rounds"]]


def test_resume_mid_round_only_runs_what_is_missing(tmp_path):
    stub = _Stub(tmp_path)
    eng = _FakeEngine(tmp_path)
    calls = {"n": 0}

    def crash_after_four(*a, **k):
        calls["n"] += 1
        if calls["n"] > 4:
            raise KeyboardInterrupt
        return eng(*a, **k)
    with pytest.raises(KeyboardInterrupt):
        _run(stub, crash_after_four)
    eng2 = _FakeEngine(tmp_path)
    _run(_Stub(tmp_path), eng2)
    assert eng2.calls[:5] == [(2, 1), (3, 1), (1, 2), (2, 2), (3, 2)]


def test_a_crashed_segment_is_marked_failed_and_the_rest_continue(tmp_path):
    stub = _Stub(tmp_path)
    out = _run(stub, _FakeEngine(tmp_path, fail={(2, 0)}))
    r1 = out["ledger"]["rounds"][0]
    assert r1["status"]["2:0"]["state"] == "FAILED"
    assert r1["evidence"]["0"]["segments"] == [1, 3]


def test_window_with_every_segment_crashed_is_not_retried(tmp_path):
    stub = _Stub(tmp_path)
    eng = _FakeEngine(tmp_path, fail={(1, 0), (2, 0), (3, 0)})
    out = _run(stub, eng)
    assert out["ledger"]["rounds"][0]["evidence"]["0"] is None
    assert not any(w == 0 for seg, w in eng.calls if seg > 3)        # 设计 §12.10 A10


def test_missing_pilot_fails_closed_before_sampling(tmp_path):
    stub = _Stub(tmp_path)
    eng = _FakeEngine(tmp_path)
    with pytest.raises(RuntimeError, match="pilot"):
        _run(stub, eng, pilot_mean=None)
    assert eng.calls == []


def test_budget_exhausted_stops_without_opening_a_round(tmp_path):
    stub = _Stub(tmp_path)
    out = _run(stub, _FakeEngine(tmp_path), stage2_production_budget_steps=9 * 250000)
    assert len(out["ledger"]["rounds"]) == 1                       # 第二轮本来要开（见上一条）
    assert out["ledger"]["stop_reason"] == "GLOBAL_BUDGET_EXHAUSTED"


def test_segment_whose_recorded_f_differs_from_the_ledger_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="不一致"):
        _run(_Stub(tmp_path), _FakeEngine(tmp_path, wrong_f=True))


def test_solvent_leg_runs_one_segment_per_window_in_round_one(tmp_path):
    eng = _FakeEngine(tmp_path)
    out = _run(_Stub(tmp_path, leg="solvent"), eng)
    r1 = out["ledger"]["rounds"][0]
    assert [(s["window"], s["segment"], s["tilt_sign"]) for s in r1["specs"]] == [(0, 1, 0), (1, 1, 0), (2, 1, 0)]
    assert eng.calls[:3] == [(1, 0), (1, 1), (1, 2)]
    assert r1["thresholds"]["min_independent_seeds"] == 1
    assert r1["evidence"]["0"]["jackknife_kind"] == "within_segment_blocks"
    # 简单窗口一段就能收敛；缺帧的 w1 照样被补
    assert r1["labels"]["0"][0] == sr.LABEL_CONVERGED
    assert any(s["window"] == 1 for r in out["ledger"]["rounds"][1:] for s in r["specs"])


def test_zero_window_shift_only_on_complex_leg(tmp_path):
    a = _run(_Stub(tmp_path / "c", leg="complex"), _FakeEngine(tmp_path / "c"))
    b = _run(_Stub(tmp_path / "s", leg="solvent"), _FakeEngine(tmp_path / "s"))
    ca = [s for s in a["ledger"]["rounds"][0]["specs"] if s["window"] == 2 and s["tilt_sign"] == 0][0]
    cb = [s for s in b["ledger"]["rounds"][0]["specs"] if s["window"] == 2 and s["tilt_sign"] == 0][0]
    da = np.array(ca["center_f_k_kJ"])
    db = np.array(cb["center_f_k_kJ"])
    assert (da[-1] - da[0]) - (db[-1] - db[0]) == pytest.approx(2.0 * KT)


def test_layout_change_refuses_to_reuse_the_ledger(tmp_path):
    _run(_Stub(tmp_path), _FakeEngine(tmp_path))
    with pytest.raises(RuntimeError, match="布局"):
        _Stub(tmp_path)._stage2_rounds_ledger_load(
            os.path.join(str(tmp_path), "checkpoints"), LAM, [(0, 3), (2, 6)])


# ---------------------------------------------------------------------------
# 收尾（σ_window / 标签 / indeterminate）
# ---------------------------------------------------------------------------

def test_finalize_replaces_window_sigma_and_flags_untrusted(tmp_path):
    class Fin(_Stub):
        _stage2_rounds_finalize = ap.ABFEPipeline._stage2_rounds_finalize

        def _load_ibs_window_outputs_merged(self, *a, **k):
            return []

        def _solve_stage_with_config_gates(self, **k):
            return {"total_delta_G": 10.0, "total_error": 1.0, "diagnostics": {
                "covariance_chain_segments": [
                    {"window_index": w, "uncertainty_kJ_mol": 0.5} for w in range(3)]}}

        @staticmethod
        def _populate_stage_diagnostics(r):
            return r
    ledger = {"rounds": [{"round": 1, "specs": [
        {"window": w, "segment": s} for w in range(3) for s in (1, 2, 3)],
        "status": {f"{s}:{w}": {"state": "SAMPLED"} for w in range(3) for s in (1, 2, 3)},
        "evidence": {"0": {"sigma_jk_kJ": 2.0}, "1": {"sigma_jk_kJ": 0.1}, "2": {"sigma_jk_kJ": None}},
        "labels": {"0": ["PROVISIONAL", ["sigma_jk_above_tol"]], "1": ["CONVERGED", []],
                   "2": ["CONVERGED", []]}}]}
    stub = Fin(tmp_path)
    res = stub._stage2_rounds_finalize(ledger, "L", LAM, RANGES, "d", stub.checkpoint_dir, KT, {})
    assert res["analysis_mode"] == "stage2_lcr_rounds_v1" and res["results_untrusted"] is True
    assert res["total_error"] == pytest.approx(np.sqrt(1.0 + 2.0 ** 2 - 0.5 ** 2))
    assert res["stage2_rounds"]["sigma_window"]["0"]["sigma_window_kJ"] == pytest.approx(2.0)


def test_finalize_without_data_for_a_window_produces_no_delta_g(tmp_path):
    ledger = {"rounds": [{"round": 1, "specs": [], "status": {},
                          "evidence": {"0": None, "1": {}, "2": {}},
                          "labels": {"0": ["INDETERMINATE", ["no_sampled_segment"]]}}]}
    stub = _Stub(tmp_path)
    res = ap.ABFEPipeline._stage2_rounds_finalize(stub, ledger, "L", LAM, RANGES, "d", "c", KT, {})
    assert "total_delta_G" not in res and res["analysis_status"] == ap.ANALYSIS_INCOMPLETE
    assert res["indeterminate_windows"][0]["window_idx"] == 0


# ---------------------------------------------------------------------------
# 身份 / 透传 / 接线
# ---------------------------------------------------------------------------

def test_switch_enters_only_the_vanishing_identity_and_only_when_true():
    rc = {"kwargs": {"stage2_lcr_rounds": True, "x": 1}}
    assert ap._scope_stage2_lcr_switch(rc, "vanishing")["kwargs"]["stage2_lcr_rounds"] is True
    assert "stage2_lcr_rounds" not in ap._scope_stage2_lcr_switch(rc, "decharging")["kwargs"]
    off = {"kwargs": {"stage2_lcr_rounds": False, "x": 1}}
    assert "stage2_lcr_rounds" not in ap._scope_stage2_lcr_switch(off, "vanishing")["kwargs"]
    assert ap._scope_stage2_lcr_switch({"kwargs": {"x": 1}}, "vanishing") == {"kwargs": {"x": 1}}


def test_planning_parameters_are_not_identity():
    for k in ("stage2_lcr_tilt_kT", "stage2_lcr_trust_floor_kT", "stage2_lcr_trust_min_ess",
              "stage2_lcr_sigma_tol_kJ", "stage2_lcr_max_rounds", "stage2_lcr_burn_in_steps",
              "stage2_lcr_zero_window_end_shift_kT"):
        assert k in ap._NON_IDENTITY_KWARGS
    assert "stage2_lcr_rounds" not in ap._NON_IDENTITY_KWARGS


def test_runabfe_passes_the_switch_only_when_true_and_params_when_set():
    assert "stage2_lcr_rounds" not in runabfe._path_evolution_kwargs({})
    assert "stage2_lcr_rounds" not in runabfe._path_evolution_kwargs({"stage2_lcr_rounds": False})
    out = runabfe._path_evolution_kwargs({"stage2_lcr_rounds": True, "stage2_lcr_sigma_tol_kJ": 2,
                                          "stage2_lcr_max_rounds": 4})
    assert out["stage2_lcr_rounds"] is True
    assert out["stage2_lcr_sigma_tol_kJ"] == 2.0 and out["stage2_lcr_max_rounds"] == 4


def test_old_path_is_kept_and_switch_defaults_off():
    src = inspect.getsource(ap.ABFEPipeline.run_full_pipeline)
    assert '_lcr_on = bool(kwargs.get("stage2_lcr_rounds", False))' in src
    assert "self._run_stage2_with_path_evolution(" in src           # 老路径还在
    assert "and not _lcr_on:" in src                                 # 固定节奏 / 自治控制器让位
    assert "要求冻结布局" in src


def test_fixed_f_k_is_threaded_to_the_engine():
    sig = inspect.signature(ap.ABFEPipeline._run_dual_lambda_stage).parameters
    for k in ("fixed_f_k_by_window", "segment_seed_tag", "fixed_f_k_burn_in_steps"):
        assert k in sig
    import ibs_engine as ie
    esig = inspect.signature(ie.IBSWindowManagerDualLambda.run_all_windows).parameters
    for k in ("fixed_f_k_by_window", "fixed_f_k_burn_in_steps", "fixed_f_k_burn_in_block_steps",
              "segment_seed_tag"):
        assert k in esig
    assert esig["fixed_f_k_by_window"].default is None
    # 09-26：burn-in 必须能分成多块，否则看不出漂移（块长 = 总长时只有 1 块）
    assert esig["fixed_f_k_burn_in_block_steps"].default < esig["fixed_f_k_burn_in_steps"].default
    assert "fixed_f_k_burn_in_block_steps=5000" in inspect.getsource(ap.ABFEPipeline._run_dual_lambda_stage)
    esrc = inspect.getsource(ie.IBSWindowManagerDualLambda.run_all_windows)
    assert "and (not fixed_f_k_mode)" in esrc                        # 跳过 LEARN/VALIDATE
    assert "and not fixed_f_k_mode:" in esrc                         # 不再倾斜
    assert 'sampler.bias_status = "fixed_f_k"' in esrc


def test_segment_tag_changes_the_seed_stream():
    import ibs_engine as ie

    class _Ledger:
        def derive(self, phase, stage, window, stream, attempt):
            return hash((phase, stage, window, stream, attempt)) % (2 ** 31)
    m = ie.IBSWindowManagerDualLambda.__new__(ie.IBSWindowManagerDualLambda)
    m.seed_ledger = _Ledger()
    m._segment_seed_tag = None
    base = m._seed_for("sampling", "vdw", 3, "velocity")
    m._segment_seed_tag = "seg2"
    s2 = m._seed_for("sampling", "vdw", 3, "velocity")
    m._segment_seed_tag = "seg3"
    s3 = m._seed_for("sampling", "vdw", 3, "velocity")
    assert len({base, s2, s3}) == 3


def test_stage_cache_payload_keeps_the_rounds_fields():
    """09-26 真机：stage2_vanishing.json 缺 analysis_mode / stage2_rounds（白名单漏了）。"""
    res = {"total_delta_G": 1.0, "total_error": 0.5, "diagnostics": {},
           "analysis_mode": "stage2_lcr_rounds_v1", "segment_dirs": ["a", "b"],
           "stage2_rounds": {"labels": {"0": ["CONVERGED", []]}, "sigma_window": {}},
           "total_error_before_jackknife": 0.4, "indeterminate_windows": []}
    payload = ap.ABFEPipeline._build_stage_cache_payload("vanishing", res, 26, {"k": 1}, LAM, RANGES)
    for k in ("analysis_mode", "segment_dirs", "stage2_rounds", "total_error_before_jackknife",
              "indeterminate_windows"):
        assert payload[k] == res[k]


def test_runabfe_passthrough_works_on_a_real_runconfig(tmp_path, monkeypatch):
    """09-26：透传用了 config[_k]，而真实入口传的是 RunConfig（不支持下标）⟹ 设了任一
    stage2_lcr_* 计划参数就在启动时 TypeError（abfe-ibs-19 发现并修）。普通 dict 测不出来。"""
    cfg = tmp_path / "lcr.json"
    cfg.write_text(json.dumps({"stage2_lcr_rounds": True, "stage2_lcr_burn_in_steps": 150000,
                               "stage2_lcr_sigma_tol_kJ": 2.0}), encoding="utf-8")
    argv = ["--config", str(cfg)]
    monkeypatch.setattr(runabfe.sys, "argv", ["runabfe.py"] + argv)
    config = runabfe.RunConfig(runabfe.build_parser().parse_args(argv))
    with pytest.raises(TypeError):
        config["stage2_lcr_burn_in_steps"]                      # 这正是原 bug 的形状
    out = runabfe._path_evolution_kwargs(config)
    assert out["stage2_lcr_rounds"] is True
    assert out["stage2_lcr_burn_in_steps"] == 150000 and out["stage2_lcr_sigma_tol_kJ"] == 2.0
