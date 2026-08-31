"""P0-12a/b：配体构象诊断 + 跨腿一致性门 + 起始构象进溶剂腿缓存身份。

对应 `docs/status/memtodolist.md` §3.0 末条与 `docs/TODO.md` 的 P0-12。

## 缺陷是什么

§3.0 早就写着「若配体本身亲脂，溶剂腿（纯水）里可能出现构象塌缩或自聚集；记录溶剂腿
配体回转半径与内部氢键随 λ 的变化」——**那条诊断一直没实现**。2026-08-04 那轮膜运行
里它真的发生了：

    配体重原子最大内距（去电荷 12 个 replica、600 帧汇总，实测）
      复合物腿  p5–p95 = 1.338–1.441 nm    ← 口袋撑着，伸展
      溶剂腿    p5–p95 = 0.624–0.707 nm    ← 塌缩，12 个 replica 零涨落
    可溶生产基线（健康）
      复合物腿  p5–p95 = 1.338–1.441 nm
      溶剂腿    p5–p95 = 0.733–1.391 nm    ← 从塌缩到伸展都采到，与复合物腿有重叠

塌缩把极性基团聚拢 ⟹ 配体–水静电耦合强 3 倍（⟨U⟩ = −569 vs −190 kJ/mol）⟹
去电荷 191.05 vs 62.80 kJ/mol。**两条腿在给不同构象族做热力学循环**，
ΔG_bind = ΔG_solv − ΔG_cplx 因此没有意义 —— 而且全程没有任何报警。

## 不要这样让本文件变绿

把重叠判据的百分位放宽成 [p0, p100]、改成"均值差 < 某个 nm"、或者在门失败时改成
warning —— 每一条都是把已知缺陷藏回去。不重叠的正解是修采样
（溶剂腿双起点验证 / 加构象采样维度），不是放宽门。
"""

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

pytest.importorskip("openmm")

import abfe_core as core

ROOT = __import__("pathlib").Path(__file__).absolute().parents[1]

# 2026-08-04 实测值（见模块 docstring）。这些数字是本文件存在的理由，别改成"好看"的。
MEASURED = {
    "membrane_complex": (1.338, 1.391, 1.441, 1.391, 0.030),
    "membrane_solvent": (0.624, 0.660, 0.707, 0.660, 0.005),
    "soluble_complex": (1.338, 1.391, 1.441, 1.391, 0.030),
    "soluble_solvent": (0.733, 1.173, 1.391, 1.100, 0.051),
}


def _summary(key, leg="unknown"):
    p5, p50, p95, mean, std = MEASURED[key]
    return {
        "protocol_version": core.LIGAND_CONFORMER_DIAGNOSTICS_VERSION,
        "leg": leg,
        "source": f"measured:{key}",
        "overlap_percentiles": list(core.LIGAND_CONFORMER_OVERLAP_PERCENTILES),
        "observables": {
            "max_internal_heavy_distance_nm": {
                "n": 600, "mean": mean, "std": std,
                "min": p5, "p5": p5, "p50": p50, "p95": p95, "max": p95,
            }
        },
    }


# ---------------------------------------------------------------------------
# 1. 门本体：用实测分布分开"坏的那次"与"健康的基线"
# ---------------------------------------------------------------------------


def test_the_broken_membrane_run_is_rejected_and_the_healthy_baseline_passes():
    """同一条判据必须同时做到这两件事，否则它要么没用、要么是调出来的。"""
    membrane = core.evaluate_cross_leg_conformer_consistency(
        _summary("membrane_complex", "membrane"), _summary("membrane_solvent", "soluble")
    )
    soluble = core.evaluate_cross_leg_conformer_consistency(
        _summary("soluble_complex", "soluble"), _summary("soluble_solvent", "soluble")
    )
    assert membrane["evaluated"] and membrane["passed"] is False
    assert membrane["overlap_nm"] == pytest.approx(0.624 - 1.338 + 0.083, abs=1e-9) or (
        membrane["overlap_nm"] < 0
    )
    assert soluble["evaluated"] and soluble["passed"] is True
    assert soluble["overlap_nm"] > 0

    with pytest.raises(ValueError, match="构象跨腿一致性门未通过"):
        core.assert_cross_leg_conformer_consistency(membrane)
    core.assert_cross_leg_conformer_consistency(soluble)  # 不得抛


def test_reason_names_the_intervals_and_forbids_widening_the_gate():
    report = core.evaluate_cross_leg_conformer_consistency(
        _summary("membrane_complex"), _summary("membrane_solvent")
    )
    reason = report["reason"]
    assert reason.startswith("cross_leg_conformer_ensembles_do_not_overlap")
    assert "1.338" in reason and "0.707" in reason, "理由里必须带上实测区间"
    assert "不是**放宽本判据" in reason or "不是" in reason


def test_missing_summary_is_recorded_as_not_evaluated_not_as_passed():
    """判不了门与门过了必须区分开（与 §9 膜质量门同一条纪律）。"""
    for cplx, solv in (
        (None, _summary("soluble_solvent")),
        (_summary("soluble_complex"), None),
        (None, None),
    ):
        report = core.evaluate_cross_leg_conformer_consistency(cplx, solv)
        assert report["evaluated"] is False
        assert report["passed"] is None
        assert "not_evaluated" in report["reason"]
        core.assert_cross_leg_conformer_consistency(report)  # 不阻断


def test_gate_is_wired_into_the_single_cycle_closure_and_nowhere_else():
    """ATT-09 的纪律：公式只有一份，门也只能有一份。"""
    src = (ROOT / "abfe_core.py").read_text(encoding="utf-8")
    assert src.count("def assert_cross_leg_conformer_consistency(") == 1
    assert src.count("assert_cross_leg_conformer_consistency(") == 2, (
        "断言函数必须只有 def + `combine_binding_free_energy` 里那一处调用；"
        "多一处就是多一套门"
    )


def test_combine_refuses_to_report_delta_g_bind_when_ensembles_disagree():
    kwargs = dict(
        dg_complex_kJ_mol=175.57,
        dg_solvent_kJ_mol=272.93,
        err_complex_kJ_mol=1.50,
        err_solvent_kJ_mol=1.46,
    )
    with pytest.raises(ValueError, match="构象跨腿一致性门未通过"):
        core.combine_binding_free_energy(
            complex_conformer_summary=_summary("membrane_complex"),
            solvent_conformer_summary=_summary("membrane_solvent"),
            **kwargs,
        )
    # 重叠时正常汇总，且报告被带进结果（事后可审计）。
    payload = core.combine_binding_free_energy(
        complex_conformer_summary=_summary("soluble_complex"),
        solvent_conformer_summary=_summary("soluble_solvent"),
        **kwargs,
    )
    assert payload["ligand_conformer_cross_leg"]["passed"] is True
    # 不给 summary 的老路径（traditional / 后处理）必须照旧能用。
    legacy = core.combine_binding_free_energy(**kwargs)
    assert legacy["ligand_conformer_cross_leg"]["evaluated"] is False
    assert legacy["delta_G_bind_kJ_mol"] == pytest.approx(272.93 - 175.57)


# ---------------------------------------------------------------------------
# 2. 度量本身算对没有
# ---------------------------------------------------------------------------


def _linear_chain(n=5, spacing=0.15, collapsed=False):
    """一条直链（伸展）或对折（塌缩），坐标可手算。"""
    xs = np.arange(n) * spacing
    if collapsed:
        xs = np.where(np.arange(n) < n // 2, xs, (n - 1 - np.arange(n)) * spacing)
    return np.stack([xs, np.zeros(n), np.zeros(n)], axis=-1)[None, :, :]


def test_max_internal_distance_and_rg_match_hand_computation():
    xyz = _linear_chain(n=5, spacing=0.15)
    m = core.ligand_conformer_metrics(xyz, heavy_local_indices=range(5))
    assert m["max_internal_heavy_distance_nm"][0] == pytest.approx(0.6)
    # 等质量直链，坐标 0,0.15,...,0.6，质心 0.3 → Rg² = mean((x-0.3)²)
    xs = np.arange(5) * 0.15
    assert m["radius_of_gyration_nm"][0] == pytest.approx(
        np.sqrt(np.mean((xs - xs.mean()) ** 2))
    )


def test_collapsed_conformer_has_smaller_metrics_than_extended():
    ext = core.ligand_conformer_metrics(_linear_chain(collapsed=False), range(5))
    col = core.ligand_conformer_metrics(_linear_chain(collapsed=True), range(5))
    assert (
        col["max_internal_heavy_distance_nm"][0]
        < ext["max_internal_heavy_distance_nm"][0]
    )
    assert col["radius_of_gyration_nm"][0] < ext["radius_of_gyration_nm"][0]


def test_radius_of_gyration_is_mass_weighted():
    xyz = _linear_chain(n=3, spacing=0.2)
    equal = core.ligand_conformer_metrics(xyz, range(3), masses_amu=[12.0, 12.0, 12.0])
    heavy_end = core.ligand_conformer_metrics(xyz, range(3), masses_amu=[12.0, 12.0, 200.0])
    assert heavy_end["radius_of_gyration_nm"][0] != pytest.approx(
        equal["radius_of_gyration_nm"][0]
    )


def test_internal_polar_contacts_count_only_close_declared_pairs():
    xyz = np.array([[[0.0, 0, 0], [0.30, 0, 0], [0.60, 0, 0]]], dtype=float)
    m = core.ligand_conformer_metrics(
        xyz, heavy_local_indices=range(3), polar_pairs=[(0, 1), (0, 2)]
    )
    # 0.30 nm ≤ 0.35 阈值 → 计 1；0.60 nm → 不计。
    assert m["internal_polar_contact_count"][0] == pytest.approx(1.0)
    none = core.ligand_conformer_metrics(xyz, range(3), polar_pairs=[])
    assert none["internal_polar_contact_count"][0] == pytest.approx(0.0)


def test_metrics_need_at_least_two_heavy_atoms():
    with pytest.raises(ValueError, match="重原子少于 2"):
        core.ligand_conformer_metrics(_linear_chain(), heavy_local_indices=[0])


def test_summary_carries_the_percentiles_used_by_the_gate():
    xyz = np.concatenate([_linear_chain(), _linear_chain(collapsed=True)], axis=0)
    m = core.ligand_conformer_metrics(xyz, range(5))
    s = core.ligand_conformer_summary(m, leg="soluble", source="unit-test")
    obs = s["observables"]["max_internal_heavy_distance_nm"]
    lo, hi = core.LIGAND_CONFORMER_OVERLAP_PERCENTILES
    assert f"p{lo:g}" in obs and f"p{hi:g}" in obs
    assert s["overlap_percentiles"] == [lo, hi]
    assert set(s["observables"]) == {
        "max_internal_heavy_distance_nm",
        "radius_of_gyration_nm",
        "internal_polar_contact_count",
    }, "§3.0 要求的 Rg 与内部氢键代理量都必须记录，不能只留主判据"


# ---------------------------------------------------------------------------
# 3. P0-12b：起始构象进溶剂腿缓存身份
# ---------------------------------------------------------------------------


def test_conformer_fingerprint_is_invariant_under_rigid_motion():
    """刚体平移/旋转不该让盒缓存失效 —— 只有构象变了才该失效。"""
    xyz = _linear_chain(n=6, spacing=0.14)[0]
    base = core.ligand_conformer_fingerprint(xyz, range(6))
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    moved = xyz @ rot.T + np.array([3.1, -2.0, 0.5])
    assert core.ligand_conformer_fingerprint(moved, range(6))["sha256"] == base["sha256"]


def test_conformer_fingerprint_changes_when_the_conformer_changes():
    ext = core.ligand_conformer_fingerprint(_linear_chain(collapsed=False)[0], range(5))
    col = core.ligand_conformer_fingerprint(_linear_chain(collapsed=True)[0], range(5))
    assert ext["sha256"] != col["sha256"]
    assert ext["max_internal_heavy_distance_nm"] > col["max_internal_heavy_distance_nm"]


def test_solvent_cache_identity_includes_the_start_conformer():
    """P0-12b 的本体：旧口径下两个完全不同的起始构象都判"缓存有效"。"""
    runabfe = pytest.importorskip("runabfe")
    # 5 → 6（2026-08-05，B4）：manifest 加了 reserved co-ion 字段，版本号继续往上走，
    # 但本测试关心的仍是"升过版本号"这件事本身，不是具体数值。
    assert runabfe.SOLVENT_CACHE_PROTOCOL_VERSION == 7, (
        "身份口径变了必须升版本号，否则旧缓存会被静默复用"
    )
    src = (ROOT / "runabfe.py").read_text(encoding="utf-8")
    assert '"ligand_start_conformer"' in src
    assert "ligand_conformer_fingerprint(" in src
    # 两个调用点都必须把 positions 传进身份构造，否则那条腿的构象又不进指纹了。
    assert src.count("positions=positions,") >= 2
