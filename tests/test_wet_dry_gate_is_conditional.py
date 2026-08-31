"""湿/干双起点门是**条件性诊断**，不是普遍适用的硬门（2026-08-31 决定）。

背景：这个门是为溶剂腿的空腔灌水问题引入的。但 T4 L99A 这类本来就干的埋藏疏水腔
根本不存在湿盆——实测复合物腿 λ_vdw=0 平衡 1 ns，空腔水数 100 次检查全为 0。
原实现把「必须能制备湿起点」写成硬性前置条件并 raise，直接把整条腿打断。

不能把「水」这个针对溶剂腿引入的概念，变成整个 ABFE 管线的普遍物理假设，
也不该要求调用方预先知道每个体系是干腔还是湿腔。

统一语义：
    wet.reached  -> 跑干+湿，评估门
    否则          -> 只跑干，gate = {evaluated: False, passed: None,
                                     reason: wet_start_not_observed_within_budget}
    passed=None   -> 不算失败也不算通过，**不参与 converged 合取**，但必须留警告
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _res(dg, sig):
    return {"converged": True, "delta_G_kJ_mol": dg, "delta_G_sigma_kJ_mol": sig}


def _cav(mode, wf, tr):
    return {"init_mode": mode, "wet_fraction": wf, "n_wet_dry_transitions": tr}


def test_no_wet_basin_is_not_evaluated_not_failed():
    from ibs_engine import endpoint_wet_dry_hysteresis_gate

    g = endpoint_wet_dry_hysteresis_gate(
        None, _res(38.2, 0.3), {"d": _cav("dry", 0.0, 0)}, wet_arm_available=False
    )
    assert g["evaluated"] is False
    assert g["passed"] is None
    assert g["reason"] == "wet_start_not_observed_within_budget"


def test_wet_basin_present_still_evaluates_normally():
    from ibs_engine import endpoint_wet_dry_hysteresis_gate

    cav = {"a": _cav("wet", 0.6, 10), "b": _cav("dry", 0.4, 9)}
    ok = endpoint_wet_dry_hysteresis_gate(
        _res(0.0, 0.3), _res(0.1, 0.3), cav, wet_arm_available=True)
    assert ok["evaluated"] is True and ok["passed"] is True
    bad = endpoint_wet_dry_hysteresis_gate(
        _res(0.0, 0.3), _res(7.0, 0.3), cav, wet_arm_available=True)
    assert bad["evaluated"] is True and bad["passed"] is False


def _seg():
    return ({"stage": "vanishing", "lambdas": list(range(7)), "total_delta_G": 20.0,
             "total_error": 0.4, "converged": True,
             "target_support_gate": {"passed": True, "failed_checks": []}},
            {"state_indices": list(range(6, 12)), "delta_G_kJ_mol": -26.0,
             "delta_G_sigma_kJ_mol": 0.3, "converged": True,
             "min_effective_sample_number": 140.0})


def test_unevaluated_gate_does_not_block_converged_but_warns():
    from ibs_engine import combine_ibs_and_independent_endpoint

    ibs, ep = _seg()
    out = combine_ibs_and_independent_endpoint(
        ibs, ep, {"evaluated": False, "passed": None,
                  "reason": "wet_start_not_observed_within_budget"}, n_states=12)
    assert out["converged"] is True, "passed=None 不得参与合取"
    assert out["wet_dry_hysteresis_evaluated"] is False
    assert out["warnings"], "未评估必须留下显式警告，不能只有一个静默的 None"
    assert any("未经双起点检验" in w for w in out["warnings"])


def test_failed_gate_still_blocks():
    from ibs_engine import combine_ibs_and_independent_endpoint

    ibs, ep = _seg()
    out = combine_ibs_and_independent_endpoint(
        ibs, ep, {"evaluated": True, "passed": False, "failed_checks": ["x"]},
        n_states=12)
    assert out["converged"] is False, "真的检出迟滞时必须仍然阻塞"
    assert out["warnings"] == []


def test_sampler_no_longer_raises_without_a_wet_seed():
    src = (REPO / "ibs_engine.py").read_text(encoding="utf-8")
    i = src.index("def run_independent_endpoint_states(")
    j = src.index("def _reduced_energies_for_record(")
    body = src[i:j]
    assert "reached_wet=False" not in body or "raise RuntimeError" not in body.split(
        "wet_available")[0][-600:], "湿起点缺失不得再 raise"
    assert "wet_available" in body and 'active_modes' in body
    assert '"wet_basin_found"' in body


def test_dryonly_and_wetdry_banks_do_not_share_cache():
    """干-only 与 干+湿 是两份不同的数据集，manifest 必须区分。"""
    from ibs_engine import build_independent_endpoint_manifest

    common = dict(stage_type="vdw", state_indices=[9, 11], common_system_xml="<S/>",
                  cv_xmls=["<a/>", "<b/>"], temperature_K=298.15, sample_interval=1000,
                  sample_steps=300_000, burn_in_steps=100_000, n_walkers_per_mode=2,
                  platform_name="CUDA", cavity_probe_radius_nm=0.24,
                  cavity_wet_min_waters=1)
    a = build_independent_endpoint_manifest(init_modes=["dry"], **common)
    b = build_independent_endpoint_manifest(init_modes=["dry", "wet"], **common)
    assert a["init_modes"] == ["dry"] and b["init_modes"] == ["dry", "wet"]
    assert a != b


def test_pipeline_falls_back_to_dry_only_without_touching_the_budget():
    """只跑干起点时**不得**调整 walker 数。

    曾经写成「翻倍以保持总采样量」——那会让实际采样预算取决于「这个体系的空腔
    会不会进水」，等于把体系依赖从门里赶出去、又从预算的后门放回来；而且
    n_walkers_per_mode 进 manifest 指纹，临界体系会产生不可预测的缓存失效。
    采样预算必须由 independent_endpoint_walkers_per_mode 显式决定，对所有体系一致。
    """
    src = (REPO / "abfe_pipeline.py").read_text(encoding="utf-8")
    i = src.index('_wet_ok = bool(wet_seed.get("reached_wet"))')
    blk = src[i:i + 900]
    assert "_walkers *= 2" not in blk, "不得因缺少湿盆而改动采样预算"
    assert "raise RuntimeError" not in blk
    assert "未评估" in blk
