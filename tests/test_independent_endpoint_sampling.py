"""INDEPENDENT_ENDPOINT_PROTOCOL_VERSION=1 回归。

背景 STAGE2_ROOT_CAUSE_2026-08-28.md：IBS stage2 每个窗口只跑一条轨迹、窗口内
所有 λ 态靠重加权得到，采不到"水塌进配体空腔"这个构型；window 2 的相邻
⟨ΔU⟩ 只有 0.4~0.6 kT（任何能量重叠判据都判优）却错 +19.49 kJ/mol。

本文件锁住修复侧的三件事：
  1. 空腔水结构判据本身算对（含周期性最小像）；
  2. 每个 λ 态都有独立样本时，多态 MBAR 能还原已知的解析自由能差；
  3. 湿/干双起点门在"数值不一致"和"两个模态压根没连通"两种情况下都判失败——
     后者是关键：两组各自卡在自己的模态里时，ΔF 之差可以是 0 而结果全错。
"""

import numpy as np
import pytest

from ibs_engine import (
    CAVITY_PROBE_RADIUS_NM,
    ENDPOINT_WET_DRY_MAX_SIGMA,
    INDEPENDENT_ENDPOINT_PROTOCOL_VERSION,
    _minimum_image_displacements,
    cavity_occupancy_diagnostics,
    count_cavity_waters,
    endpoint_wet_dry_hysteresis_gate,
    solve_independent_endpoint_states,
)

KT = 0.008314462618 * 298.15


# ---------------------------------------------------------------------------
# 空腔水结构判据
# ---------------------------------------------------------------------------

def _cubic_box(length_nm):
    return np.diag([float(length_nm)] * 3)


def test_cavity_water_counts_only_waters_inside_the_probe_radius():
    box = _cubic_box(5.0)
    positions = np.array([
        [2.5, 2.5, 2.5],   # 0: 配体重原子
        [2.5, 2.5, 2.6],   # 1: 水氧，0.10 nm —— 在半径内
        [2.5, 2.5, 2.9],   # 2: 水氧，0.40 nm —— 在半径外
    ])
    n = count_cavity_waters(
        positions, box, ligand_heavy_idx=np.array([0]),
        water_oxygen_idx=np.array([1, 2]), probe_radius_nm=CAVITY_PROBE_RADIUS_NM,
    )
    assert n == 1


def test_cavity_water_count_respects_the_minimum_image_convention():
    """配体在盒子一角、水在对角——直接算差向量是 4.9 nm，最小像下只有 0.1 nm。

    漏掉这个的话，穿过周期边界待在空腔里的水会被整帧漏计，湿态会被误判成干态。
    """
    box = _cubic_box(5.0)
    positions = np.array([
        [0.05, 2.5, 2.5],
        [4.95, 2.5, 2.5],
    ])
    naive = float(np.linalg.norm(positions[1] - positions[0]))
    assert naive > 4.0
    n = count_cavity_waters(
        positions, box, ligand_heavy_idx=np.array([0]),
        water_oxygen_idx=np.array([1]), probe_radius_nm=CAVITY_PROBE_RADIUS_NM,
    )
    assert n == 1


def test_minimum_image_handles_a_triclinic_reduced_box():
    box = np.array([[4.0, 0.0, 0.0], [0.5, 4.0, 0.0], [0.5, 0.5, 4.0]])
    delta = np.array([[3.9, 3.6, 3.7]])
    wrapped = _minimum_image_displacements(delta, box)
    assert float(np.linalg.norm(wrapped)) < float(np.linalg.norm(delta))
    assert np.all(np.abs(wrapped) <= 2.5)


def test_cavity_occupancy_counts_wet_dry_transitions():
    diag = cavity_occupancy_diagnostics([0, 0, 1, 1, 0, 2, 2], wet_min_waters=1)
    assert diag["n_frames"] == 7
    assert diag["wet_fraction"] == pytest.approx(4 / 7)
    assert diag["dry_fraction"] == pytest.approx(3 / 7)
    # 干→湿、湿→干、干→湿 = 3 次
    assert diag["n_wet_dry_transitions"] == 3


def test_cavity_occupancy_reports_zero_transitions_for_a_stuck_trajectory():
    diag = cavity_occupancy_diagnostics([0] * 50, wet_min_waters=1)
    assert diag["wet_fraction"] == 0.0
    assert diag["n_wet_dry_transitions"] == 0


# ---------------------------------------------------------------------------
# 多态 MBAR：每个态都有自己的独立样本
# ---------------------------------------------------------------------------

_SIGMAS = (1.0, 1.3, 1.6)
_CENTERS = (0.0, 0.6, 1.2)


def _analytic_reduced_f():
    """u_k(x) = 0.5*((x-c_k)/s_k)^2 的解析约化自由能（相对 k=0），单位 kT。

    Z_k = s_k*sqrt(2*pi) ⇒ f_k = -ln(s_k) + const。
    """
    return np.array([-np.log(s) for s in _SIGMAS]) - (-np.log(_SIGMAS[0]))


def _synthetic_record(state_pos, global_state, mode, walker, n_frames, seed,
                      cavity_pattern=None):
    rng = np.random.default_rng(seed)
    x = rng.normal(_CENTERS[state_pos], _SIGMAS[state_pos], size=n_frames)
    u = np.stack(
        [0.5 * ((x - c) / s) ** 2 for c, s in zip(_CENTERS, _SIGMAS)], axis=1
    ) * KT
    if cavity_pattern is None:
        cavity = np.zeros(n_frames)
    else:
        cavity = np.asarray(cavity_pattern, dtype=float)[:n_frames]
    return {
        "u_cv_kj_mol": u,
        # 合成记录只存这三个态的列；列身份显式声明，与生产记录（存全部态的
        # 列）走同一条取列逻辑。
        "energy_column_indices": [7, 9, 11],
        "volume_nm3": np.full(n_frames, 30.0),
        "cavity_waters": cavity,
        "segments": [{"burn_in_steps": 0, "sample_steps": n_frames,
                      "n_frames": n_frames, "reason": "synthetic"}],
        "sampled_steps": n_frames,
        "global_state": int(global_state),
        "init_mode": str(mode),
        "walker": int(walker),
    }


def _synthetic_bank(n_frames=1500, seed0=0, cavity_by_mode=None):
    state_indices = [7, 9, 11]
    records = {}
    for pos, global_k in enumerate(state_indices):
        for mode in ("dry", "wet"):
            for walker in range(2):
                pattern = None
                if cavity_by_mode is not None:
                    pattern = cavity_by_mode[mode]
                seed = seed0 + 1000 * pos + 100 * (mode == "wet") + walker
                records[f"{global_k}|{mode}|{walker}"] = _synthetic_record(
                    pos, global_k, mode, walker, n_frames, seed, pattern
                )
    return {"state_indices": state_indices, "records": records,
            "protocol_version": INDEPENDENT_ENDPOINT_PROTOCOL_VERSION}


def test_multistate_mbar_recovers_the_analytic_free_energy_differences():
    """每个 λ 态都有真实样本时，MBAR 必须还原解析值。

    这是 IBS 单轨迹重加权做不到的那件事的正面对照：这里 n_k 全部 > 0。
    """
    bank = _synthetic_bank()
    res = solve_independent_endpoint_states(bank, KT, lrc_coeff=None)
    assert res.get("error") is None, res
    assert res["converged"] is True
    expected_kj = _analytic_reduced_f() * KT
    got = np.asarray(res["f_kJ_mol"], dtype=float)
    got = got - got[0]
    assert np.allclose(got, expected_kj, atol=0.15), (got, expected_kj)
    assert res["delta_G_kJ_mol"] == pytest.approx(expected_kj[-1], abs=0.15)
    # 每个态都被真正采样过——这正是修复的结构性区别。
    assert min(res["n_k_decorrelated"]) > 0
    assert res["estimator"] == "multi_state_MBAR_all_states_sampled"


def test_solver_can_restrict_to_one_init_mode():
    bank = _synthetic_bank()
    wet = solve_independent_endpoint_states(bank, KT, init_modes=["wet"])
    dry = solve_independent_endpoint_states(bank, KT, init_modes=["dry"])
    assert wet["init_modes_used"] == ["wet"]
    assert dry["init_modes_used"] == ["dry"]
    # 各自只用一半 walker
    assert sum(wet["n_k_decorrelated"]) < sum(
        solve_independent_endpoint_states(bank, KT)["n_k_decorrelated"]
    )


def test_solver_fails_closed_when_a_state_has_no_samples():
    bank = _synthetic_bank()
    for key in [k for k in bank["records"] if k.startswith("11|")]:
        del bank["records"][key]
    res = solve_independent_endpoint_states(bank, KT)
    assert res["converged"] is False
    assert res["error"] == "independent_endpoint_state_without_samples"


def test_solver_fails_closed_on_too_few_decorrelated_samples():
    bank = _synthetic_bank(n_frames=40)
    res = solve_independent_endpoint_states(bank, KT, min_decorrelated_samples=10_000)
    assert res["converged"] is False
    assert res["error"] == "independent_endpoint_insufficient_decorrelated_samples"


def test_decorrelation_uses_the_slower_of_energy_and_cavity_series():
    """空腔水数是慢模态时，去相关必须听它的，而不是听能量的。

    §3.2 已经实测出这个模态在能量上几乎不可见；只用能量估自相关会低估 g、
    高估独立样本数。
    """
    n = 1500
    # 极慢的湿/干方波：整段只翻转两次。
    slow = np.concatenate([np.zeros(n // 3), np.ones(n // 3), np.zeros(n - 2 * (n // 3))])
    bank = _synthetic_bank(n_frames=n, cavity_by_mode={"dry": slow, "wet": slow})
    res = solve_independent_endpoint_states(bank, KT)
    picked = [r["selected_series"] for r in res["decorrelation"].values()]
    assert "cavity_water_count" in picked
    flat = _synthetic_bank(n_frames=n)
    res_flat = solve_independent_endpoint_states(flat, KT)
    # 恒定的空腔序列没有信息量，不参与竞争。
    assert all(
        r["selected_series"] == "reduced_potential"
        for r in res_flat["decorrelation"].values()
    )
    assert min(res["n_k_decorrelated"]) < min(res_flat["n_k_decorrelated"])


# ---------------------------------------------------------------------------
# 湿/干双起点验收门
# ---------------------------------------------------------------------------

def _walker_cavity(mode, wet_fraction, transitions, walker=0):
    return {
        "init_mode": mode, "walker": walker, "global_state": 11,
        "wet_fraction": wet_fraction, "dry_fraction": 1.0 - wet_fraction,
        "n_wet_dry_transitions": transitions, "n_frames": 100,
        "mean_cavity_waters": wet_fraction, "max_cavity_waters": 2.0,
        "wet_min_waters": 1,
    }


def _res(dg, sigma):
    return {"converged": True, "delta_G_kJ_mol": dg, "delta_G_sigma_kJ_mol": sigma}


def test_wet_dry_gate_passes_when_consistent_and_modes_are_connected():
    gate = endpoint_wet_dry_hysteresis_gate(
        _res(-6.30, 0.30), _res(-6.10, 0.30),
        {"a": _walker_cavity("wet", 0.6, 12), "b": _walker_cavity("dry", 0.4, 9)},
    )
    assert gate["passed"] is True, gate
    assert gate["modes_are_connected"] is True


def test_wet_dry_gate_fails_when_the_two_start_points_disagree():
    gate = endpoint_wet_dry_hysteresis_gate(
        _res(-6.30, 0.30), _res(+3.20, 0.30),
        {"a": _walker_cavity("wet", 0.6, 12), "b": _walker_cavity("dry", 0.4, 9)},
    )
    assert gate["passed"] is False
    assert "wet_dry_delta_exceeds_sigma_gate" in gate["failed_checks"]
    assert gate["failure_reason"] == "stage2_wet_dry_hysteresis_failed"
    assert gate["max_sigma"] == pytest.approx(ENDPOINT_WET_DRY_MAX_SIGMA)


def test_wet_dry_gate_fails_when_the_modes_never_connect_even_if_numbers_agree():
    """最危险的情形：两组各自卡在自己的模态里，ΔF 之差却是 0。

    数值一致在这里**不是**证据——两个都可能一样地错。必须由结构量否决。
    """
    gate = endpoint_wet_dry_hysteresis_gate(
        _res(-6.30, 0.30), _res(-6.30, 0.30),
        {"a": _walker_cavity("wet", 1.0, 0), "b": _walker_cavity("dry", 0.0, 0)},
    )
    assert gate["passed"] is False
    assert gate["failed_checks"] == ["wet_and_dry_walkers_never_exchanged_modes"]
    assert gate["modes_are_connected"] is False


def test_wet_dry_gate_fails_closed_when_a_solve_is_missing():
    gate = endpoint_wet_dry_hysteresis_gate(
        _res(-6.30, 0.30), {"converged": False}, {},
    )
    assert gate["passed"] is False
    assert gate["failure_reason"] == "wet_dry_hysteresis_unavailable"


# ---------------------------------------------------------------------------
# "生产动力学里不准还藏着 WCA" —— 这条必须是硬断言，不是约定
# ---------------------------------------------------------------------------

def _minimal_topology_with_water(n_waters=3):
    import openmm.app as app
    from openmm.app.element import hydrogen, oxygen, carbon

    topology = app.Topology()
    chain = topology.addChain()
    lig = topology.addResidue("LIG", chain)
    topology.addAtom("C1", carbon, lig)
    topology.addAtom("H1", hydrogen, lig)
    for _ in range(n_waters):
        res = topology.addResidue("HOH", chain)
        topology.addAtom("O", oxygen, res)
        topology.addAtom("H1", hydrogen, res)
        topology.addAtom("H2", hydrogen, res)
    return topology


def test_water_and_ligand_index_helpers_pick_the_right_atoms():
    from ibs_engine import ligand_heavy_atom_indices, water_oxygen_indices

    topology = _minimal_topology_with_water(n_waters=3)
    # 配体是原子 0(C) 和 1(H)；水从 2 开始，每个水的氧在 2, 5, 8。
    assert water_oxygen_indices(topology).tolist() == [2, 5, 8]
    # 氢不参与空腔判据。
    assert ligand_heavy_atom_indices(topology, [0, 1]).tolist() == [0]


def test_fixed_state_simulation_refuses_a_system_that_still_has_group4_wca():
    """`require_group4=False` 必须**硬断言**系统里不存在 Group 4。

    这是"正式采样不能保留隐藏的 Group-4 WCA"这条要求的执行点：把防护壳留在
    生产动力学里会直接 RuntimeError，不可能悄悄发生。
    """
    import openmm
    from openmm import XmlSerializer, unit as omm_unit

    from ibs_engine import _build_fixed_state_simulation

    system = openmm.System()
    system.addParticle(12.0)
    shield = openmm.CustomExternalForce("0.0*x")
    shield.addParticle(0, [])
    shield.setForceGroup(4)          # ← 冒充还留在生产系统里的 WCA 防护壳
    system.addForce(shield)

    cv = openmm.CustomExternalForce("0.0*y")
    cv.addParticle(0, [])

    with pytest.raises(RuntimeError, match="WCA group 4"):
        _build_fixed_state_simulation(
            topology=_minimal_topology_with_water(n_waters=0),
            system_xml=XmlSerializer.serialize(system),
            cv_xml=XmlSerializer.serialize(cv),
            require_group4=False,
            platform_name="Reference",
            temperature_q=298.15 * omm_unit.kelvin,
            positions=[[0.0, 0.0, 0.0]] * 1,
            box_vectors=None,
            integrator_seed=1,
            velocity_seed=2,
        )


def test_sampler_runs_dry_only_when_no_wet_basin_is_observed():
    """[2026-08-31 决定] 湿起点是**条件性诊断**，不是前置条件。

    此处曾断言「制备不出湿起点就 raise」。那是错的：T4 L99A 这类本来就干的埋藏
    疏水腔根本不存在湿盆（实测复合物腿 λ_vdw=0 平衡 1 ns，空腔水数 100 次检查
    全为 0），原行为会把整条腿打断。现在改为只跑干起点、门标记为未评估。
    不能把「水」这个针对溶剂腿引入的概念变成整个 ABFE 管线的普遍物理假设。
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "ibs_engine.py").read_text(encoding="utf-8")
    i = src.index("def run_independent_endpoint_states(")
    j = src.index("def _reduced_energies_for_record(")
    body = src[i:j]
    # 不得再因为缺湿起点而 raise
    assert "拿一个干构型冒充湿起点" not in body
    # 必须走「只跑干起点」的分支并如实记录
    assert "wet_available" in body
    assert 'active_modes = tuple(INDEPENDENT_ENDPOINT_INIT_MODES) if wet_available else ("dry",)' in body
    assert '"wet_basin_found"' in body


def test_manifest_fingerprint_changes_if_wca_comes_back_into_the_system():
    """指纹覆盖 system XML，所以"把 WCA 塞回生产"会让已采数据整体失效。"""
    from ibs_engine import build_independent_endpoint_manifest

    common = dict(
        stage_type="vdw", state_indices=[9, 11], cv_xmls=["<a/>", "<b/>"],
        temperature_K=298.15, sample_interval=1000, sample_steps=500_000,
        burn_in_steps=100_000, n_walkers_per_mode=2, platform_name="CUDA",
        cavity_probe_radius_nm=CAVITY_PROBE_RADIUS_NM, cavity_wet_min_waters=1,
    )
    no_wca = build_independent_endpoint_manifest(common_system_xml="<System/>", **common)
    with_wca = build_independent_endpoint_manifest(
        common_system_xml="<System><WCA/></System>", **common
    )
    assert no_wca["common_system_sha256"] != with_wca["common_system_sha256"]
    assert no_wca != with_wca


# ---------------------------------------------------------------------------
# 两段拼接：IBS 段 + 独立固定-λ 端点段
# ---------------------------------------------------------------------------

def _ibs_segment(dg=20.5, sigma=0.4, lams=(0, 1, 2, 3, 4, 5, 6), passed=True):
    return {
        "stage": "vanishing",
        "lambdas": list(lams),
        "total_delta_G": dg,
        "total_error": sigma,
        "converged": True,
        "min_overlap": 0.47,
        "min_overlap_threshold": 0.05,
        "raw_min_absolute_ess": 85.9,
        "raw_min_absolute_ess_threshold": 20.0,
        "max_top1pct_raw_weight": 0.12,
        "max_top1pct_raw_weight_threshold": 0.35,
        "target_support_gate": {"passed": passed, "failed_checks": []},
        "window_overlap_diagnostics": [],
    }


def _endpoint_segment(dg=-26.8, sigma=0.3, states=(6, 7, 8, 9, 10, 11), min_ess=140.0):
    return {
        "state_indices": list(states),
        "delta_G_kJ_mol": dg,
        "delta_G_sigma_kJ_mol": sigma,
        "converged": True,
        "n_k_decorrelated": [200] * len(states),
        "min_effective_sample_number": min_ess,
    }


def _good_gate():
    return {"passed": True, "failed_checks": [], "modes_are_connected": True}


def test_combiner_adds_the_two_segments_in_the_same_lambda_direction():
    from ibs_engine import combine_ibs_and_independent_endpoint

    out = combine_ibs_and_independent_endpoint(
        _ibs_segment(), _endpoint_segment(), _good_gate(), n_states=12
    )
    assert out.get("error") is None, out
    assert out["total_delta_G"] == pytest.approx(20.5 - 26.8)
    # 两段样本互不重叠 ⇒ σ 平方相加
    assert out["total_error"] == pytest.approx(float(np.hypot(0.4, 0.3)))
    assert out["join_lambda_index"] == 6
    assert out["converged"] is True
    assert out["coverage_diagnostics"]["covered_lambda_indices"] == list(range(12))


def test_combiner_rejects_a_gap_between_the_two_segments():
    from ibs_engine import combine_ibs_and_independent_endpoint

    out = combine_ibs_and_independent_endpoint(
        _ibs_segment(lams=(0, 1, 2, 3, 4)), _endpoint_segment(states=(6, 7, 8, 9, 10, 11)),
        _good_gate(), n_states=12,
    )
    assert out["converged"] is False
    assert "combine_join_mismatch" in out["error"]


def test_combiner_rejects_incomplete_lambda_coverage():
    from ibs_engine import combine_ibs_and_independent_endpoint

    out = combine_ibs_and_independent_endpoint(
        _ibs_segment(), _endpoint_segment(states=(6, 7, 8)), _good_gate(), n_states=12
    )
    assert out["converged"] is False
    assert "combine_incomplete_lambda_coverage" in out["error"]


def test_combiner_fails_closed_without_a_passing_wet_dry_gate():
    from ibs_engine import combine_ibs_and_independent_endpoint

    for gate in (None, {}, {"passed": False, "failed_checks": ["x"]}):
        out = combine_ibs_and_independent_endpoint(
            _ibs_segment(), _endpoint_segment(), gate, n_states=12
        )
        assert out["converged"] is False, gate


def test_combiner_fails_closed_when_the_endpoint_segment_has_too_little_support():
    from ibs_engine import combine_ibs_and_independent_endpoint

    out = combine_ibs_and_independent_endpoint(
        _ibs_segment(), _endpoint_segment(min_ess=3.0), _good_gate(), n_states=12
    )
    assert out["converged"] is False
    assert out["target_support_gate"]["passed"] is False
    assert (
        "independent_endpoint_effective_sample_number"
        in out["target_support_gate"]["failed_checks"]
    )


def test_combined_result_passes_the_pipeline_stage_gate():
    """拼出来的结果必须能直接过 `_assert_stage_result_sane`——包括它对 vanishing
    强制要求的 target_support_gate。"""
    from abfe_pipeline import ABFEPipeline
    from ibs_engine import combine_ibs_and_independent_endpoint

    class _Dummy:
        _stage_quality_failure_details = staticmethod(
            ABFEPipeline._stage_quality_failure_details
        )
        _format_stage_quality_failure_details = staticmethod(
            ABFEPipeline._format_stage_quality_failure_details
        )

    out = combine_ibs_and_independent_endpoint(
        _ibs_segment(), _endpoint_segment(), _good_gate(), n_states=12
    )
    out["min_decorrelated_samples"] = 332
    out["min_decorrelated_samples_threshold"] = 20
    out["max_endpoint_uncertainty_kJ_mol"] = 0.4
    out["max_endpoint_uncertainty_kJ_mol_threshold"] = 1.0
    ABFEPipeline._assert_stage_result_sane(_Dummy(), "Stage 2 (vanishing)", out)


# ---------------------------------------------------------------------------
# 局部列索引 vs 全局 λ 索引：接错点的自由能看起来完全正常，必须锁死
# ---------------------------------------------------------------------------

def test_global_and_local_lambda_indices_are_kept_distinct():
    from ibs_engine import combine_ibs_and_independent_endpoint

    bank = _synthetic_bank()
    # 端点块只为自己建了一个窗口系统 ⇒ CV/列是局部的 0,1,2；
    # 它们在整条 λ 路径上的全局身份是 6,7,8。
    bank["global_state_indices"] = [6, 7, 8]
    res = solve_independent_endpoint_states(bank, KT)
    assert res["state_indices"] == [7, 9, 11]          # 列身份（局部）
    assert res["global_state_indices"] == [6, 7, 8]    # λ 身份（全局）

    out = combine_ibs_and_independent_endpoint(
        _ibs_segment(dg=1.0, sigma=0.1, lams=(0, 1, 2, 3, 4, 5, 6)),
        res, _good_gate(), n_states=9,
    )
    # 拼接必须用全局索引 —— 用局部的 [7,9,11] 会判成 join 不匹配。
    assert out.get("error") is None, out
    assert out["join_lambda_index"] == 6


def test_pipeline_fingerprint_covers_the_endpoint_protocol_version():
    """采样协议的**代码**变化必须让 stage2 缓存失效——配置项进不了这一项。"""
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "abfe_pipeline.py").read_text(
        encoding="utf-8"
    )
    block = re.search(
        r'if stage_name == "vanishing":\n            payload\["vdw_nonbonded_protocol_version"\].*?\n        # ',
        source, re.S,
    )
    assert block is not None
    assert "independent_endpoint_protocol_version" in block.group(0)


def test_stage2_wiring_excludes_the_endpoint_window_from_the_ibs_solve():
    """开关打开时，末窗口的 IBS 数据不得参与拼接（否则那段被算两遍）。"""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "abfe_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "_independent_endpoint_enabled" in source
    assert "excluded_local_windows={_endpoint_window_index}" in source
    assert "combine_ibs_and_independent_endpoint(" in source


def test_manifest_fingerprint_covers_the_seed_relaxation():
    """起始构型的制备方式（本态最小化步数）变了，已采数据不得复用。

    2026-08-29 实测：湿起点来自完全解耦态，直接丢给 λ_vdw=1.0 会
    `Particle coordinate is NaN`；修法是每个 walker 先在本态最小化。既然这一步
    改变了喂进 burn-in 的构型，它就必须进指纹。
    """
    from ibs_engine import build_independent_endpoint_manifest

    common = dict(
        stage_type="vdw", state_indices=[9, 11], common_system_xml="<System/>",
        cv_xmls=["<a/>", "<b/>"], temperature_K=298.15, sample_interval=1000,
        sample_steps=500_000, burn_in_steps=100_000, n_walkers_per_mode=2,
        platform_name="CUDA", cavity_probe_radius_nm=CAVITY_PROBE_RADIUS_NM,
        cavity_wet_min_waters=1,
    )
    a = build_independent_endpoint_manifest(minimize_iterations=2_000, **common)
    b = build_independent_endpoint_manifest(minimize_iterations=10_000, **common)
    assert a["minimize_iterations"] == 2_000
    assert a != b


def test_sampler_reports_which_walker_blew_up():
    """积分炸掉时必须点名是哪个 (λ 态, 起点模态, walker)，并说清最可能的原因。

    裸的 OpenMM `Particle coordinate is NaN` 不说明是哪个 walker，也不提示"起始
    构型与本态哈密顿量不相容"这个真正的原因。
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "ibs_engine.py").read_text(
        encoding="utf-8"
    )
    i = source.index("def run_independent_endpoint_states(")
    j = source.index("def _reduced_energies_for_record(")
    body = source[i:j]
    assert "except openmm.OpenMMException as exc:" in body
    assert "独立端点采样在 λ 态" in body
    assert "minimize_iterations" in body
    # 最小化必须发生在 burn-in 之前
    assert body.index("minimizeEnergy") < body.index("_extend_state_trajectory(")


# ---------------------------------------------------------------------------
# 湿空腔阶梯
# ---------------------------------------------------------------------------

def test_wet_walkers_walk_the_lambda_ladder_dry_walkers_do_not():
    """湿起点必须从最解耦一端逐态往耦合端传；干起点每个 walker 都用同一个种子。

    2026-08-29 实测：把完全解耦态的湿构型直接丢给 λ_vdw=1.0，minimize 2000 步后
    仍然 `Particle coordinate is NaN`——λ_vdw≈1 上"湿空腔"根本不是亚稳态。
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "ibs_engine.py").read_text(
        encoding="utf-8"
    )
    i = source.index("def run_independent_endpoint_states(")
    j = source.index("def _reduced_energies_for_record(")
    body = source[i:j]
    # 湿侧按 λ_vdw 升序（最解耦优先）访问，干侧按原顺序
    assert "wet_order = sorted(state_indices, key=lambda k: lam_for[int(k)])" in body
    assert 'order = list(state_indices) if mode == "dry" else wet_order' in body
    # 只有 walker 0 的末构型往前传，阶梯与 walker 数无关
    assert 'if mode == "wet" and walker == 0:' in body
    # 末构型必须落盘，否则续跑时阶梯会断、后面的态又拿到最解耦端的湿构型
    assert "final_positions_nm=final_pos_nm" in body
    assert '"final_positions_nm" in data' in body


def test_wet_seeding_scheme_is_part_of_the_manifest():
    from ibs_engine import build_independent_endpoint_manifest

    m = build_independent_endpoint_manifest(
        stage_type="vdw", state_indices=[9, 11], common_system_xml="<System/>",
        cv_xmls=["<a/>", "<b/>"], temperature_K=298.15, sample_interval=1000,
        sample_steps=500_000, burn_in_steps=100_000, n_walkers_per_mode=2,
        platform_name="CUDA", cavity_probe_radius_nm=CAVITY_PROBE_RADIUS_NM,
        cavity_wet_min_waters=1,
    )
    assert m["wet_seeding_scheme"] == "wet_cavity_ladder_v1"


def test_ladder_order_falls_back_to_reverse_state_order_without_lambda_values():
    """不给 λ 值时退回仓库约定：state_indices 升序 == λ_vdw 降序，倒序即最解耦优先。"""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "ibs_engine.py").read_text(
        encoding="utf-8"
    )
    i = source.index("def run_independent_endpoint_states(")
    j = source.index("def _reduced_energies_for_record(")
    body = source[i:j]
    assert "wet_order = list(reversed(state_indices))" in body


# ---------------------------------------------------------------------------
# 端点段失败原因不得被拼接层伪装
# （2026-08-31 回归：4W53 output_v2 把 n_k=[342, 18, 962, 508] 差 2 个样本的
#  真实失败，报成了 `combine_join_mismatch: IBS 段止于 19，独立端点段起于 0`）
# ---------------------------------------------------------------------------

def test_insufficient_samples_branch_still_carries_the_global_lambda_identity():
    """失败分支也必须带 `global_state_indices`——否则拼接层会回落到局部索引。"""
    bank = _synthetic_bank(n_frames=40)
    bank["global_state_indices"] = [19, 20, 21]
    res = solve_independent_endpoint_states(bank, KT, min_decorrelated_samples=10_000)
    assert res["error"] == "independent_endpoint_insufficient_decorrelated_samples"
    assert res["converged"] is False
    assert res["global_state_indices"] == [19, 20, 21], res


def test_state_without_samples_branch_also_carries_global_indices():
    bank = _synthetic_bank()
    bank["global_state_indices"] = [19, 20, 21]
    for key in [k for k in bank["records"] if k.startswith("11|")]:
        del bank["records"][key]
    res = solve_independent_endpoint_states(bank, KT)
    assert res["error"] == "independent_endpoint_state_without_samples"
    assert res["global_state_indices"] == [19, 20, 21], res


def test_combiner_reports_the_endpoint_error_not_a_fake_join_mismatch():
    """端点段自报失败时，必须原样透出它的原因，而不是索引错误。"""
    from ibs_engine import combine_ibs_and_independent_endpoint

    failed_endpoint = {
        "error": "independent_endpoint_insufficient_decorrelated_samples",
        "n_k": [342, 18, 962, 508],
        "min_decorrelated_samples_threshold": 20,
        "state_indices": [0, 1, 2, 3],
        "global_state_indices": [19, 20, 21, 22],
        "converged": False,
    }
    out = combine_ibs_and_independent_endpoint(
        _ibs_segment(lams=tuple(range(20))), failed_endpoint, _good_gate(), n_states=23,
    )
    assert out["converged"] is False
    assert "combine_join_mismatch" not in out["error"]
    assert out["endpoint_error"] == (
        "independent_endpoint_insufficient_decorrelated_samples"
    )
    # 诊断数字要能直接看到，不用再去翻 checkpoint json
    assert out["endpoint_n_k"] == [342, 18, 962, 508]
    assert out["endpoint_min_decorrelated_samples_threshold"] == 20


def test_combiner_still_fails_closed_on_a_real_join_mismatch():
    """只改'报什么原因'，不改'是否放行'：端点段无 error 时几何检查照旧。"""
    from ibs_engine import combine_ibs_and_independent_endpoint

    out = combine_ibs_and_independent_endpoint(
        _ibs_segment(lams=(0, 1, 2, 3, 4)), _endpoint_segment(states=(6, 7, 8, 9, 10, 11)),
        _good_gate(), n_states=12,
    )
    assert out["converged"] is False
    assert "combine_join_mismatch" in out["error"]
