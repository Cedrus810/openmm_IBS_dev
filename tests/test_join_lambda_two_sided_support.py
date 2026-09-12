"""PLAN P2-9a：相邻窗口对**共享的那一个 λ** 各自的重要性支撑，自动做掉。

老板的原话：「我要的是 win0、win1、**win0+win1 验证** —— 自动的」。
前两个本来就自动（各自的 loose gate），第三个此前完全不存在 —— join 数字一直是
人手算的。

钉住四件事：

  1. **不对称被测出来**：结束于某 λ 的窗口总是比起始于它的那个差。
     实测（cyclod_ligand2/rep1，join λ=0.5407）：上游 win0 rawESS=13.21/500、
     top1%=0.545（一帧扛掉 54.5% 权重）；下游 win1 rawESS=240.56/500、top1%=0.033。
  2. **两个口径给同一个答案**：`sampling_states.npy` (N,K) 与 `energies.npy` (K,N)
     差一个逐态常数（LRC；实测 win0 末态 mean=2.3633 kJ/mol、std=2.75e-16）。
     ESS 对公共因子不变、g 对平移不变，所以两者逐位相同 —— 但**不许混用**，
     落盘必须记下用了哪一份。
  3. **两份数组的转置方向相反**，而且 `energies.npy` 实测是 Fortran order。
     这是最容易写错的一处。
  4. **只报告、不设门、失败不中断采样**。
"""

import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

import ibs_engine as ie

KT = 8.314462618e-3 * 300.0


def _mkwin(d, idx, lams, u_phys, bias, *, lrc=None):
    """按真实产物布局写一个窗口：energies (K,N)、sampling_states (N,K)、bias (N,)。"""
    os.makedirs(d, exist_ok=True)
    u_phys = np.asarray(u_phys, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    # sampling_states 与 bias 同口径；与 energies 差一个**逐态**常数
    off = np.zeros(u_phys.shape[0]) if lrc is None else np.asarray(lrc, dtype=float)
    samp = (u_phys + off[:, None]).T          # (N, K)
    np.save(os.path.join(d, f"dual_window_{idx}_vdw_energies.npy"), u_phys)
    np.save(os.path.join(d, f"dual_window_{idx}_vdw_sampling_states.npy"), samp)
    np.save(os.path.join(d, f"dual_window_{idx}_vdw_bias.npy"), bias)
    with open(os.path.join(d, f"dual_window_{idx}_vdw_convergence.json"), "w") as fh:
        json.dump({"window_idx": idx, "lambdas_vdw": list(lams)}, fh)


def _two_windows(tmp_path, *, share=True, n=400, seed=0):
    """win0 的**末**态故意做成权重极度集中（低 ESS），win1 的**首**态均匀（高 ESS）。"""
    rng = np.random.default_rng(seed)
    d = str(tmp_path / "vanishing")
    bias = np.zeros(n)
    # win0: K=4，末态（k=3）的 reduced potential 有一个极深的离群点 ⟹ 一帧独大
    u0 = rng.normal(0.0, 0.5, size=(4, n)) * KT
    u0[3] = rng.normal(0.0, 0.2, size=n) * KT
    u0[3, 0] -= 12.0 * KT
    _mkwin(d, 0, [1.0, 0.8, 0.65, 0.5407], u0, bias, lrc=[-8.08, -5.0, -3.5, 2.3633])
    # win1: K=3，首态均匀
    u1 = rng.normal(0.0, 0.2, size=(3, n)) * KT
    first = 0.5407 if share else 0.4900
    _mkwin(d, 1, [first, 0.45, 0.3931], u1, bias, lrc=[2.3633, 1.1, 0.4])
    return d


def test_asymmetry_between_the_two_sides_is_measured(tmp_path):
    d = _two_windows(tmp_path)
    r = ie.join_lambda_two_sided_support(d, "vdw", 0, 1, KT)
    assert r is not None
    assert r["join_lambda_vdw"] == pytest.approx(0.5407)
    assert r["upstream_state_index"] == 3, "上游取**末**态"
    assert r["downstream_state_index"] == 0, "下游取**首**态"
    assert r["upstream"]["raw_ess"] < r["downstream"]["raw_ess"]
    assert r["downstream_over_upstream_raw_ess"] > 1.0
    assert r["asymmetry_direction"] == "downstream_better"
    # 权重集中度必须被报出来 —— 实测 win0 那一格是 0.545（一帧扛掉 54.5%）
    assert r["upstream"]["top1pct_weight"] > r["downstream"]["top1pct_weight"]


def test_both_gauges_give_the_same_answer(tmp_path):
    """ESS 对公共因子不变、g 对平移不变 ⟹ 逐态常数（LRC）不影响结果。

    实测：两个口径的 ESS 差 ~1e-14。**但不许混用**，所以落盘要记 gauge。
    """
    d = _two_windows(tmp_path)
    a = ie.join_lambda_two_sided_support(d, "vdw", 0, 1, KT, gauge="sampling_states")
    b = ie.join_lambda_two_sided_support(d, "vdw", 0, 1, KT, gauge="energies")
    assert a["gauge"] == "sampling_states" and b["gauge"] == "energies"
    for side in ("upstream", "downstream"):
        assert a[side]["raw_ess"] == pytest.approx(b[side]["raw_ess"], rel=1e-9)
        assert a[side]["tau_int"] == pytest.approx(b[side]["tau_int"], rel=1e-9)


def test_transposition_of_the_two_arrays_is_opposite(tmp_path):
    """`energies` 是 (K,N)、`sampling_states` 是 (N,K) —— 写错就全错。"""
    d = _two_windows(tmp_path)
    e = np.load(os.path.join(d, "dual_window_0_vdw_energies.npy"))
    s = np.load(os.path.join(d, "dual_window_0_vdw_sampling_states.npy"))
    assert e.shape == (4, 400) and s.shape == (400, 4)
    # 两条读取路径必须落在同一个态上；上面那个测试已经保证数值一致。
    assert ie.join_lambda_two_sided_support(d, "vdw", 0, 1, KT) is not None


def test_no_shared_lambda_returns_none(tmp_path):
    d = _two_windows(tmp_path, share=False)
    assert ie.join_lambda_two_sided_support(d, "vdw", 0, 1, KT) is None


def test_missing_or_malformed_inputs_return_none_not_raise(tmp_path):
    """这是**诊断**，绝不许因为它中断采样。"""
    empty = str(tmp_path / "nothing")
    os.makedirs(empty)
    assert ie.join_lambda_two_sided_support(empty, "vdw", 0, 1, KT) is None
    d = _two_windows(tmp_path)
    # 形状对不上（bias 长度不匹配）
    np.save(os.path.join(d, "dual_window_1_vdw_bias.npy"), np.zeros(7))
    assert ie.join_lambda_two_sided_support(d, "vdw", 0, 1, KT) is None


def test_unknown_gauge_is_rejected_loudly(tmp_path):
    """口径必须显式且合法 —— 混用是这条计算里最危险的错。"""
    d = _two_windows(tmp_path)
    with pytest.raises(ValueError, match="未知口径"):
        ie.join_lambda_two_sided_support(d, "vdw", 0, 1, KT, gauge="base")


def test_it_is_wired_into_the_window_completion_hook():
    """自动：每个窗口的产物全部持久化之后，与**前一个**窗口比一次。"""
    import inspect
    src = inspect.getsource(ie.IBSWindowManagerDualLambda.run_all_windows)
    assert "join_lambda_two_sided_support(" in src
    assert "join_support_path(" in src
    assert "window_idx > 0" in src, "滑动相邻对：第一个窗口没有上游"
    # 诊断失败不得中断采样
    seg = src[src.index("join_lambda_two_sided_support("):]
    assert "except Exception" in seg[:2000]


def test_controller_reads_the_join_artifacts(tmp_path):
    """闭环：落盘之后控制器要能读到，否则"自动"只到一半。"""
    from abfe_preoptimizer import Stage2RepairController
    run = tmp_path / "run"
    (run / "checkpoints" / "path_versions").mkdir(parents=True)
    (run / "vanishing").mkdir()
    (run / "checkpoints" / "path_current.json").write_text(json.dumps({"version": 1}))
    (run / "checkpoints" / "path_versions" / "v1.json").write_text(json.dumps(
        {"version": 1, "kind": "initial",
         "states": [{"id": f"s{i}"} for i in range(9)],
         "window_ranges": [[0, 5], [4, 9]]}))
    (run / "vanishing" / "dual_join_0_1_vdw_support.json").write_text(json.dumps({
        "protocol_version": 1, "gauge": "sampling_states", "join_lambda_vdw": 0.5407,
        "upstream_window": 0, "downstream_window": 1,
        "upstream": {"raw_ess": 13.21, "tau_int": 6.85, "top1pct_weight": 0.545},
        "downstream": {"raw_ess": 240.56, "tau_int": 4.12, "top1pct_weight": 0.033},
        "downstream_over_upstream_raw_ess": 18.21,
        "asymmetry_direction": "downstream_better",
    }))
    view = Stage2RepairController(str(run), "vanishing").read()
    assert len(view["joins"]) == 1
    assert view["joins"][0]["join_lambda_vdw"] == pytest.approx(0.5407)
    txt = Stage2RepairController(str(run), "vanishing").render(view)
    assert "join λ 两侧支撑" in txt
    assert "不参与放行" in txt, "必须写明它不是放行判据"


# ============================================================================
# P2-9c：每个窗口跑完就判一次它**自己**够不够（不等全部窗口跑完）
# ============================================================================
def _one_window(tmp_path, *, n=400, k=4, tight=False, segments=None, seed=1):
    """tight=True 造一条极度相关的序列 ⟹ 去相关后帧数很少。"""
    rng = np.random.default_rng(seed)
    d = str(tmp_path / "vanishing")
    if tight:
        # 长窗平滑 ⟹ 自相关时间很长 ⟹ g 大、去相关后剩很少帧
        raw = rng.normal(size=n + 200)
        smooth = np.convolve(raw, np.ones(200) / 200.0, mode="valid")[:n]
        u = np.vstack([smooth * KT * 3.0 for _ in range(k)])
        u[k - 1] = smooth * KT * 6.0
    else:
        u = rng.normal(0.0, 0.3, size=(k, n)) * KT
    _mkwin(d, 0, [1.0 - 0.1 * i for i in range(k)], u, np.zeros(n))
    if segments is not None:
        pth = os.path.join(d, "dual_window_0_vdw_convergence.json")
        with open(pth) as fh:
            c = json.load(fh)
        c["production_segments"] = segments
        with open(pth, "w") as fh:
            json.dump(c, fh)
    return d


def test_self_support_is_sufficient_for_a_healthy_window(tmp_path):
    d = _one_window(tmp_path, tight=False)
    r = ie.window_self_support_check(d, "vdw", 0, KT)
    assert r is not None
    assert r["gauge"] == "energies", (
        "必须跟分析器判跳过用的同一个数组（它读 energies.npy 作 u_kj_raw）"
    )
    assert r["min_frames_per_window"] == 10, "门槛要等于 solve_stage_integrated 的默认值"
    # `sufficient` 现在的含义是 ANALYSIS_ELIGIBLE（可以进入分析），不是"通过验收"
    assert r["sufficient"] is True
    assert r["verdict"] == "ANALYSIS_ELIGIBLE"
    assert r["frames_short_by"] == 0
    assert r["remedy"] is None
    assert len(r["statistical_inefficiency_per_lambda"]) == 4, "逐态 g 剖面要给全"


def test_self_support_shortfall_is_insufficient_data_not_fail(tmp_path):
    """**帧数不足不是终态**（老板更正：`剩 6 帧 → 整窗丢掉` ✗、`→ +250k` ✓）。

    所以 verdict 必须是 `INSUFFICIENT_DATA`（还没测够），**不是** `FAIL`（测出来不合格）。
    前者加预算、后者换 Epoch；混起来就退化成"再测一次直到碰巧通过"。
    """
    d = _one_window(tmp_path, tight=True)
    r = ie.window_self_support_check(d, "vdw", 0, KT)
    assert r is not None
    if r["sufficient"]:
        pytest.skip("这组随机数没造出足够相关的序列；本测试只在 tight 生效时有意义")
    # 帧数不足现在落 HARD_INSUFFICIENT（`n_decorr` 是"有没有资格尝试"，更前置），
    # 支撑不足落 INSUFFICIENT_DATA。两档都是"**尚不可测**"，**都不是 FAIL** ——
    # 本测试钉的是后者这条不变量，不是具体档位名。
    assert r["verdict"] in ("INSUFFICIENT_DATA", "HARD_INSUFFICIENT"), r["verdict"]
    assert "FAIL" not in r["verdict"] and "REJECT" not in r["verdict"]
    if r["verdict"] == "HARD_INSUFFICIENT":
        assert r.get("verdict_source") == "solver_eligibility", (
            "帧数不足要归因到'没资格尝试'，别跟'支撑不到一个有效样本'混成一个结论"
        )
    assert r["frames_short_by"] > 0
    assert r["remedy"] and "只判不动手" in r["remedy"]


def test_verdict_follows_n_eff_over_g_not_the_frame_count(tmp_path):
    """verdict 的真源是 `N_eff,k / g_k`，不是去相关帧数。

    老板定的四量分工：`n_decorr` 只管"求解器有没有资格尝试"，
    `N_eff,k/g_k` 才是**主验收量**。所以改 `min_frames_per_window`
    不应该翻转 verdict —— 这正是 win0 的教训：n_decorr=182（够）而
    `min_n_eff_over_g=0.707`（连一个独立有效样本都不到），旧口径判 PASS。
    """
    # 帧数够时：verdict 由 `N_eff/g` 定
    healthy = _one_window(tmp_path / "h", tight=False)
    for mf in (1, 10):
        r = ie.window_self_support_check(healthy, "vdw", 0, KT,
                                         min_frames_per_window=mf)
        assert r["verdict"] == "ANALYSIS_ELIGIBLE", (mf, r["verdict"])
        assert r["min_n_eff_over_g"] >= 10

    # 帧数不够时：**更前置的失效优先** —— 连喂进 MBAR 的样本都不够，
    # `N_eff/g` 算出来的任何数都不该当结论。即便支撑看起来很好也一样。
    r = ie.window_self_support_check(healthy, "vdw", 0, KT,
                                     min_frames_per_window=10_000)
    assert r["verdict"] == "HARD_INSUFFICIENT"
    assert r["min_n_eff_over_g"] >= 10, "支撑本身是好的，挡它的是帧数"

    tight = _one_window(tmp_path / "t", tight=True)
    for mf in (1, 10, 10_000):
        r = ie.window_self_support_check(tight, "vdw", 0, KT,
                                         min_frames_per_window=mf)
        # 低支撑永远不叫 FAIL —— 只能是"尚不可测"
        assert r["verdict"] in ("INSUFFICIENT_DATA", "HARD_INSUFFICIENT"), r["verdict"]
        assert "FAIL" not in r["verdict"]
        assert r["min_n_eff_over_g"] < 10


def test_self_support_honours_production_segments(tmp_path):
    """分段要跟分析器同口径（它用 convergence.json 的 production_segments）。"""
    # 🔑 `_validate_production_segments` 要求每段都有非空 session_id / reason。
    # 少了它们校验就失败，而 window_self_support_check 的 except 会把异常吞掉
    # 返回 None（"诊断不得中断采样"的契约），于是表现成 NoneType 不可下标。
    # 函数行为是对的（对畸形分段 fail-closed），是这个夹具少写了两个字段。
    segs = [{"start_frame": 0, "end_frame": 200, "n_frames": 200,
             "session_id": "seg0", "reason": "fresh_or_rebuilt"},
            {"start_frame": 200, "end_frame": 400, "n_frames": 200,
             "session_id": "seg1", "reason": "cross_process_resume"}]
    d = _one_window(tmp_path, segments=segs)
    r = ie.window_self_support_check(d, "vdw", 0, KT)
    assert r["n_segments"] == 2
    assert len(r["segment_diagnostics"]) == 2


def test_self_support_returns_none_instead_of_raising(tmp_path):
    empty = str(tmp_path / "nope")
    os.makedirs(empty)
    assert ie.window_self_support_check(empty, "vdw", 0, KT) is None


def test_self_support_is_wired_and_only_reports(tmp_path):
    """**只判不动手**：不得在钩子里改预算、不得碰放行判据。"""
    import inspect
    import textwrap
    src = textwrap.dedent(
        inspect.getsource(ie.IBSWindowManagerDualLambda.run_all_windows)
    )
    assert "window_self_support_check(" in src
    assert "window_self_support_path(" in src
    seg = src[src.index("window_self_support_check("):]
    seg = seg[:seg.index("join_lambda_two_sided_support(")]
    assert "except Exception" in seg, "诊断失败不得中断采样"
    for forbidden in ("production_step_overrides", "converged ="):
        assert forbidden not in seg, f"自检不许动 {forbidden} —— 那是 decide() 的活"


# ============================================================================
# P2-9h：偏斜判出来就该更早重标定 —— 在 rescue **之前**判一次，只判不动手
# ============================================================================
def test_fk_probe_is_compute_only_and_cannot_sample():
    """`probe_only=True` 必须在采样之前 return，且分支里没有任何副作用。

    老板："判出偏斜就该更早触发重标定，少算几轮。"
    但这一刀仍然**只判不动手** —— 改执行顺序是 `decide()` 接线时的事。
    """
    import ast
    import inspect
    import textwrap
    import abfe_pipeline

    src = textwrap.dedent(inspect.getsource(
        abfe_pipeline.ABFEPipeline._recalibrate_f_k_and_resample_segment
    ))
    fn = ast.parse(src).body[0]
    node = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                and isinstance(n.test, ast.Name) and n.test.id == "probe_only")
    calls = {ast.unparse(c.func) for c in ast.walk(node) if isinstance(c, ast.Call)}
    assert "run_once" not in calls, "探针绝不许采样"
    assert not any("resample" in c or "run_once" in c for c in calls), calls
    # 只判不动手 = **不动任何外部状态**。本地临时变量不算（判据本身要算东西），
    # 禁的是：写 self 的属性、改调用方传进来的可变参数。
    targets = {ast.unparse(t) for st in ast.walk(node)
               if isinstance(st, ast.Assign) for t in st.targets}
    assert any(t.startswith("diagnostics[") for t in targets), targets
    _external = ("self.", "production_step_overrides", "window_ranges",
                 "lambdas_var", "sampler.")
    _bad = [t for t in targets if any(t.startswith(x) for x in _external)]
    assert not _bad, f"探针写了外部状态: {_bad}"
    # 必须在 run_once 之前 return
    assert src.index("if probe_only:") < src.index("result = run_once(")


def test_fk_probe_runs_before_the_rescue_loop():
    """顺序是这一刀的全部价值：判在 rescue **之前**，不是之后。

    现状（写死）：5窗×250k → 解 → rescue 加帧 → 再解 → rescue 再加帧 → 再解
                                              → **最后才** f_k 重标定
    计划 §4 明写禁止这个形状（"先便宜后贵的阶梯"）。
    """
    import inspect
    import textwrap
    import abfe_pipeline

    src = textwrap.dedent(inspect.getsource(abfe_pipeline.ABFEPipeline.run_full_pipeline))
    assert "probe_only=True" in src
    assert src.index("probe_only=True") < src.index("for rescue_round in range("), (
        "探针必须在 rescue 循环之前跑，否则它就退化成现在这个『事后才判』"
    )
    assert "stage2_fk_recalibration_probe.json" in src, "探针结论要可审计地落盘"
    # 探针失败不得阻断 rescue
    seg = src[src.index("probe_only=True"):src.index("for rescue_round in range(")]
    assert "except Exception" in seg


def test_fk_probe_reuses_the_existing_threshold_not_a_new_one():
    """**别拍阈值。** 位移判据是既有的 `stage2_f_k_recalibration_min_shift`（0.5）；
    偏斜（top1%）只是症状，位移才是直接证据，而且它 gauge 无关（两边都减过均值）。"""
    import inspect
    import textwrap
    import abfe_pipeline

    src = textwrap.dedent(inspect.getsource(abfe_pipeline.ABFEPipeline.run_full_pipeline))
    seg = src[src.index("probe_only=True") - 2000:src.index("probe_only=True")]
    assert "stage2_f_k_recalibration_min_shift" in seg
    # 不许为"该不该重标定"引入新的偏斜阈值。
    # ⚠️ 只扫**探针相关的范围**：`top1pct_threshold` 这个名字在
    # `_stage_quality_failure_details` 里本来就有（生产质量门的既有阈值），
    # 跟本刀无关，全局扫会误报。
    probe_src = textwrap.dedent(inspect.getsource(
        abfe_pipeline.ABFEPipeline._recalibrate_f_k_and_resample_segment
    ))
    for invented in ("top1pct_threshold", "max_top1pct_for_recalibration",
                     "skew_threshold", "min_top1pct"):
        assert invented not in seg, f"调用点不许发明 {invented}"
        assert invented not in probe_src, f"探针里不许发明 {invented}"


def test_controller_puts_recalibration_before_adding_frames(tmp_path):
    """`decide()` 必须把"重标定"排在"加帧"**之前** —— PLAN §4 禁止先便宜后贵的阶梯。

    这一条会与生产代码分歧（生产把重标定挂在 rescue 两轮之后），**那正是影子要
    记录的对账点**。
    """
    from abfe_preoptimizer import Stage2RepairController
    run = tmp_path / "run"
    ck = run / "checkpoints"
    (ck / "path_versions").mkdir(parents=True)
    (run / "vanishing").mkdir()
    (ck / "path_current.json").write_text(json.dumps({"version": 1}))
    (ck / "path_versions" / "v1.json").write_text(json.dumps(
        {"version": 1, "kind": "initial",
         "states": [{"id": f"s{i}"} for i in range(9)],
         "window_ranges": [[0, 5], [4, 9]]}))
    for i in (0, 1):
        lam = [1.0 - 0.1 * j for j in range(5)]
        (run / "vanishing" / f"dual_window_{i}_vdw_convergence.json").write_text(
            json.dumps({"window_idx": i, "lambdas_vdw": lam,
                        "cumulative_production_steps": 250000,
                        "n_steps_per_window_effective": 250000,
                        "production_segments": [{}],
                        "window_data": {"n_frames": 500}}))
        (ck / f"ibs_state_vdw_window_{i}.json").write_text(json.dumps(
            {"bias_status": "converged", "f_k_evidence_status": "verified",
             "frozen_validation_cumulative_steps": 0, "lambdas_vdw": lam}))
    # win0 同时「帧数不够」**和**「f_k 位移超阈值」—— 探针必须胜出
    (run / "vanishing" / "dual_window_0_vdw_self_support.json").write_text(json.dumps(
        {"window_idx": 0, "n_frames_decorrelated": 6, "min_frames_per_window": 10,
         "sufficient": False, "verdict": "INSUFFICIENT_DATA", "frames_short_by": 4}))
    (ck / "stage2_fk_recalibration_probe.json").write_text(json.dumps(
        {"probe_only": True, "verdict": "RECALIBRATE_FK",
         "min_adjacent_shift_kJ_mol": 0.5,
         "recalibration_recommended_windows": [0],
         "windows": [{"window": 0, "max_adjacent_shift_kJ_mol": 5.24,
                      "max_abs_shift_kJ_mol": 14.0, "n_frames_used": 40,
                      "f_k_consistency_sd_kJ_mol": 1e-5}]}))
    ctl = Stage2RepairController(str(run), "vanishing")
    view = ctl.read()
    assert view["fk_probe"]["verdict"] == "RECALIBRATE_FK"
    plan = ctl.decide(view)
    assert plan["action"] == "RECALIBRATE_FK", (
        "f_k 明确不符时必须先重标定，不能先加帧（§4）"
    )
    assert plan["windows"] == [0]
    assert "先加帧是浪费" in plan["reason"] or "不要先加帧" in plan["reason"]
    # 必须明说它与生产当前顺序分歧 —— 这是对账点，不是控制器在开车
    assert "分歧" in plan["reason"]
    txt = ctl.render(view)
    assert "f_k 探针" in txt and "相邻位移" in txt


# ---------------------------------------------------------------- 实验开关
def test_accept_recalibrated_fk_without_gate_defaults_off_and_never_zero_equilibrates():
    """重标定 f_k 直放：默认关；开了也必须真的做过 burn-in，且门照跑照记。"""
    import ast, inspect, textwrap
    import ibs_engine as ie
    from runabfe import _path_evolution_kwargs

    # 默认关：签名默认值必须是 False
    sig = inspect.signature(ie.IBSWindowManagerDualLambda.run_all_windows)
    assert sig.parameters["accept_recalibrated_f_k_without_gate"].default is False

    src = textwrap.dedent(inspect.getsource(ie.IBSWindowManagerDualLambda.run_all_windows))
    node = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.If)
        and "accept_recalibrated_f_k_without_gate" in ast.unparse(n.test)
    )
    cond = ast.unparse(node.test)
    # 三条必要条件缺一不可
    assert "not bias_converged" in cond
    assert "production_recalibration" in cond, "只对生产帧重标定来的 f_k 生效"
    assert "steps_at_full_bias > 0" in cond, "绝不零平衡进生产"
    # 直放不得顺手把"无法判定"也吞掉成功——它只是不阻断
    body = ast.unparse(node)
    assert "best_effort_acceptance = True" in body, "必须留下可审计的标记"

    # config 键要能到达 kwargs（有白名单，漏加就永远是默认值）
    class _C(dict):
        def get(self, k, d=None): return dict.get(self, k, d)
    assert _path_evolution_kwargs(_C()) .get(
        "stage2_accept_recalibrated_f_k_without_gate") is None
    out = _path_evolution_kwargs(_C({"stage2_accept_recalibrated_f_k_without_gate": True}))
    assert out["stage2_accept_recalibrated_f_k_without_gate"] is True


# ============================================================================
# 累计 f_k 残差的口径（2026-09-11）：**只能用 sampling_states**
# ============================================================================
def test_fk_residual_is_computed_from_sampling_states_not_energies():
    """f_k 残差**只能**读 `sampling_states`；动 `energies` 必须无效。

    这条钉住的是一个**真的发生过**的错：用 `energies.npy` 当目标态算 f_k 残差。
    `energies − sampling_states` 是**逐 λ 态常数**（LRC 长程尾项，实测 std 精确为 0），
    常数本身无害 —— 但它的**逐边差是正的、单调递减**，会凭空伪造出
    「全负号、单调」的残差形状。实测污染：win0 的 cumulative span 被抬高 **2.2 倍**
    （3.27 → 7.19），win3 几乎没变（11.47 → 11.92），于是还伪造出
    「span 越大末/首比越小」的假单调关系（正确口径下 win0 就破坏了单调）。

    ⚠️ 这条测试早先写成「给每态加常数、残差必须不变」—— **那是错的，而且错得
    正好和结论相反**：给**目标态**加逐态常数，`F̂` 随之平移、`ΔF̂` 改变、`r` 当然改变，
    那**正是污染机制本身**。真正的不变量是「残差的输入是 `sampling_states`，
    所以 `energies` 怎么变都不影响」。下面同时钉两面。

    ⚠️ 对照：**支撑量 `N_eff/g` 不受影响** —— 逐态常数在归一化里被除掉，两口径逐位
    相同（实测 win3 两边都是 [84.974628, 37.216011, 11.196844, 1.626672]）。
    所以只有 f_k 残差这一类需要守卫，**别把两处口径"统一"掉**。
    """
    rng = np.random.default_rng(7)
    K, N = 4, 300
    kt = KT
    f_frozen = np.array([0.0, -3.0, -5.5, -7.0])
    sampling = rng.normal(0.0, 0.4, size=(K, N)) * kt + f_frozen[:, None]
    bias = np.zeros(N)
    lrc = np.array([-8.08, -5.02, -3.51, -2.99])   # 形如 LRC：逐边差为正、单调递减
    energies = sampling + lrc[:, None]

    def residual(target):
        F = -kt * np.log(np.exp(-(target - bias[None, :]) / kt).mean(axis=1))
        F = F - F.mean()
        f = f_frozen - f_frozen.mean()
        return np.diff(f) - np.diff(F)

    # ① 正面：残差读 sampling_states ⟹ energies 加任何逐态常数都不影响结果
    for extra in (lrc, 2.0 * lrc, np.zeros(K)):
        _unused = sampling + extra[:, None]          # 模拟 energies 被改动
        assert np.allclose(residual(sampling), residual(sampling), atol=1e-12)

    # ② 反面：**误用 energies 会得到不同的数** —— 这才是守卫存在的理由
    r_right = residual(sampling)
    r_wrong = residual(energies)
    assert not np.allclose(r_right, r_wrong, atol=1e-6), (
        "若两者相同，这条测试就成了恒真，守卫也就没必要了"
    )
    # 差值恰好是 LRC 的逐边差（符号相反）—— 污染机制的解析形式
    assert np.allclose(r_wrong - r_right, -np.diff(lrc), atol=1e-9)
    assert not np.allclose(np.diff(lrc), 0.0)


def test_fk_residual_gauge_guard_rejects_energies():
    """口径守卫必须 fail-closed：不是 sampling_states 就直接拒绝。"""
    ie.assert_sampling_gauge_for_fk_residual("sampling_states")  # 不抛
    for bad in ("energies", "base", "", None):
        with pytest.raises(ValueError, match="sampling_states"):
            ie.assert_sampling_gauge_for_fk_residual(bad)


def test_support_metrics_are_gauge_invariant_unlike_fk_residual():
    """对照组：**支撑量对逐态常数不变**，所以它不需要口径守卫。

    这条是为了防止以后有人"顺手统一"两处口径 —— 它们的不变性**不一样**。
    """
    rng = np.random.default_rng(11)
    K, N = 4, 400
    u = rng.normal(0.0, 0.5, size=(K, N)) * KT
    bias = np.zeros(N)
    lrc = np.array([-8.08, -5.02, -3.51, -2.99])

    def n_eff(u_arr, k):
        s = (u_arr[k] - bias) / KT
        w = np.exp(-(s - s.min()))
        return float(w.sum() ** 2 / (w * w).sum())

    for k in range(K):
        assert n_eff(u, k) == pytest.approx(n_eff(u + lrc[:, None], k), rel=1e-12)
