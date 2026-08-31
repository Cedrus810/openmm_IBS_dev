"""§3.3 膜输入核对 + §9 膜质量门（docs/status/memtodolist.md §3.3 / §9 / §13.3）。

两节的共同前提：**通用 10 ns 不是膜平衡充分性的证明**。所以这里守的核心不是
"算得对不对"，而是"缺了必须保存的量会不会被放过"、"能不能靠缩短判据窗口把门弄绿"。

判据统一为「末段 ≥ 20 ns 内线性漂移小于阈值」，阈值全部来自 §13.3 的命名常量
（`abfe_core` 的 `ACCEPTANCE_THRESHOLDS_VERSION`），本文件不自带任何魔数阈值——
它只构造能通过/不能通过的合成序列。

归档部分（`.top` / 全部 `.itp` / 位置限制 / 力场 include 的递归 SHA256）复用
`runabfe._gromacs_dependency_hashes()`，不在本文件重复测。

全部 CPU 可跑：合成时间序列 + 最小 Topology，不建 Context、不读轨迹。
"""

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

openmm = pytest.importorskip("openmm")
from openmm import app, unit

import abfe_core as core


TAIL = core.MEMBRANE_QUALITY_GATE_TAIL_WINDOW_NS  # 20.0 ns

# `_membrane_system()` 把上叶头基放在 z=7.0、下叶放在 z=3.0
SYNTHETIC_UPPER_HEAD_Z = 7.0
SYNTHETIC_LOWER_HEAD_Z = 3.0
SYNTHETIC_BILAYER_SEPARATION_NM = SYNTHETIC_UPPER_HEAD_Z - SYNTHETIC_LOWER_HEAD_Z


# ---------------------------------------------------------------------------
# 合成观测量
# ---------------------------------------------------------------------------


def _series(mean, slope_per_ns=0.0, span_ns=40.0, n=81):
    """一条线性时间序列：value = mean + slope*(t - t_mid)。"""
    t = np.linspace(0.0, span_ns, n)
    return {
        "times_ns": t.tolist(),
        "values": (mean + slope_per_ns * (t - t.mean())).tolist(),
    }


def _good_observables(**overrides):
    obs = {
        # APL 0.65 nm²，几乎不漂
        "apl_nm2": _series(0.65, 0.0002),
        # 蛋白横截面校正后的 APL（MEM-03）。文献值那道门比的是**这一条**：
        # raw APL 把跨膜蛋白占掉的横向面积也摊给了脂质。这里让两者相等，
        # 相当于"无蛋白或蛋白不在膜内"，于是既有测试的期望值全部不变。
        "apl_protein_corrected_nm2": _series(0.65, 0.0002),
        "bilayer_thickness_nm": _series(3.85, 0.0001),
        "lipid_tail_order_parameter": _series(0.21),
        "protein_backbone_rmsd_nm": _series(0.18),
        "transmembrane_tilt_deg": _series(4.0, 0.01),
        "pocket_rmsd_nm": _series(0.12),
        "ligand_heavy_atom_rmsd_nm": _series(0.15),
        "box_xy_area_nm2": _series(41.6),
        "box_z_nm": _series(9.5),
        "box_volume_nm3": _series(395.2),
    }
    obs.update(overrides)
    return obs


def _good_diagnostics(**overrides):
    diag = {
        "density_profile_along_normal": {"z_nm": [0.0, 1.0], "water": [1.0, 0.0]},
        "leaflet_composition": {"upper": {"POPC": 64}, "lower": {"POPC": 64}},
        "anomalous_pocket_water_count": 0,
        "membrane_periodic_image_contacts": 0,
        "membrane_undulation_or_residual_tension": "none_detected",
        "lipid_lateral_relaxation_timescale_ns": 30.0,
        "equilibration_length_ns": 120.0,
    }
    diag.update(overrides)
    return diag


def _coion_observables():
    return {
        "coion_abs_z_from_midplane_nm": _series(4.2),
        "coion_ligand_min_image_distance_nm": _series(1.9),
        "coion_protein_heavy_atom_distance_nm": _series(1.6),
        "coion_nearest_phosphorus_distance_nm": _series(1.4),
        "coion_first_shell_water_count": _series(5.8),
    }


# ---------------------------------------------------------------------------
# §9 漂移斜率
# ---------------------------------------------------------------------------


def test_linear_drift_recovers_a_known_slope():
    series = _series(0.65, slope_per_ns=0.003)
    fit = core.linear_drift_per_ns(series["times_ns"], series["values"])
    assert fit["slope_per_ns"] == pytest.approx(0.003, abs=1e-9)
    assert fit["mean"] == pytest.approx(0.65, abs=1e-9)
    assert fit["span_ns"] == pytest.approx(40.0)


@pytest.mark.parametrize(
    "times, values, pattern",
    [
        ([0.0], [1.0], "至少需要 2 个点"),
        ([0.0, 1.0], [1.0], "形状不匹配"),
        ([0.0, 0.0], [1.0, 2.0], "跨度非正"),
        ([0.0, float("nan")], [1.0, 2.0], "非有限值"),
    ],
)
def test_linear_drift_rejects_degenerate_input(times, values, pattern):
    with pytest.raises(ValueError, match=pattern):
        core.linear_drift_per_ns(times, values)


# ---------------------------------------------------------------------------
# §9 质量门：完整性
# ---------------------------------------------------------------------------


def test_well_equilibrated_membrane_passes():
    report = core.evaluate_membrane_quality_gate(
        _good_observables(), _good_diagnostics(), literature_apl_nm2=0.64
    )
    assert report["passed"] is True, report["failed_checks"]
    assert report["failed_checks"] == []
    assert report["thresholds_version"] == core.ACCEPTANCE_THRESHOLDS_VERSION
    assert report["tail_window_ns"] == pytest.approx(TAIL)


@pytest.mark.parametrize("dropped", core.REQUIRED_MEMBRANE_QUALITY_OBSERVABLES)
def test_missing_required_observable_fails_closed(dropped):
    obs = _good_observables()
    del obs[dropped]
    with pytest.raises(ValueError, match="缺少必须保存的量"):
        core.evaluate_membrane_quality_gate(obs, _good_diagnostics())


@pytest.mark.parametrize("dropped", core.REQUIRED_MEMBRANE_QUALITY_DIAGNOSTICS)
def test_missing_required_diagnostic_fails_closed(dropped):
    diag = _good_diagnostics()
    diag[dropped] = None
    with pytest.raises(ValueError, match="缺少必须保存的量"):
        core.evaluate_membrane_quality_gate(_good_observables(), diag)


def test_scalar_instead_of_time_series_is_rejected():
    """§9：不能只报平均值。"""
    obs = _good_observables(apl_nm2=0.65)
    with pytest.raises(ValueError, match="不接受只报平均值"):
        core.evaluate_membrane_quality_gate(obs, _good_diagnostics())


def test_short_trajectory_cannot_satisfy_the_tail_window():
    """10 ns 覆盖不了 20 ns 末段窗口 → fail closed，而不是拿 10 ns 凑。"""
    obs = _good_observables(apl_nm2=_series(0.65, 0.0002, span_ns=10.0, n=21))
    with pytest.raises(ValueError, match="覆盖不了要求的末段窗口"):
        core.evaluate_membrane_quality_gate(obs, _good_diagnostics())


def test_tail_window_cannot_be_shrunk_to_make_the_gate_green():
    """把窗口缩到 5 ns 能让短轨迹"通过"——所以这条路必须是显式的、可见的。

    本测试不禁止传入更小的窗口（诊断时有用），但钉住报告里会记下实际用的窗口，
    使"缩窗口换绿灯"在 provenance 里无法隐藏。
    """
    obs = _good_observables(apl_nm2=_series(0.65, 0.0002, span_ns=10.0, n=21))
    for name in obs:
        obs[name] = _series(
            core.linear_drift_per_ns(obs[name]["times_ns"], obs[name]["values"])["mean"],
            span_ns=10.0,
            n=21,
        )
    report = core.evaluate_membrane_quality_gate(
        obs, _good_diagnostics(), tail_window_ns=5.0
    )
    assert report["tail_window_ns"] == pytest.approx(5.0)
    assert report["tail_window_ns"] != TAIL, "缩过的窗口必须与默认值可区分"


# ---------------------------------------------------------------------------
# §13.3 各项阈值
# ---------------------------------------------------------------------------


def test_apl_drift_above_threshold_fails():
    # 0.65 nm² 上 0.5 %/ns ⇒ slope ≈ 0.00325，超 0.2 %/ns。
    obs = _good_observables(apl_nm2=_series(0.65, 0.65 * 0.005))
    report = core.evaluate_membrane_quality_gate(obs, _good_diagnostics())
    assert report["passed"] is False
    assert "apl_nm2" in report["failed_checks"]


def test_apl_far_from_literature_value_fails():
    """§13.3：与该脂质力场文献值差 ≤ 3%。"""
    report = core.evaluate_membrane_quality_gate(
        _good_observables(), _good_diagnostics(), literature_apl_nm2=0.55
    )
    assert report["passed"] is False
    criteria = {
        c["criterion"] for c in report["checks"] if not c["passed"]
    }
    assert "deviation_from_literature_percent" in criteria


def test_thickness_drift_above_threshold_fails():
    # 0.05 nm / 20 ns ⇒ 阈值斜率 0.0025 nm/ns；给 0.01 nm/ns 必超。
    obs = _good_observables(bilayer_thickness_nm=_series(3.85, 0.01))
    report = core.evaluate_membrane_quality_gate(obs, _good_diagnostics())
    assert report["passed"] is False
    assert "bilayer_thickness_nm" in report["failed_checks"]


def test_tilt_drift_above_threshold_fails():
    obs = _good_observables(transmembrane_tilt_deg=_series(4.0, 1.0))
    report = core.evaluate_membrane_quality_gate(obs, _good_diagnostics())
    assert report["passed"] is False
    assert "transmembrane_tilt_deg" in report["failed_checks"]


@pytest.mark.parametrize(
    "name, bad_mean",
    [
        ("protein_backbone_rmsd_nm", core.PROTEIN_BACKBONE_MAX_RMSD_NM + 0.05),
        ("pocket_rmsd_nm", core.POCKET_MAX_RMSD_NM + 0.05),
        ("ligand_heavy_atom_rmsd_nm", core.LIGAND_HEAVY_ATOM_MAX_RMSD_NM + 0.05),
    ],
)
def test_rmsd_above_threshold_fails(name, bad_mean):
    obs = _good_observables(**{name: _series(bad_mean)})
    report = core.evaluate_membrane_quality_gate(obs, _good_diagnostics())
    assert report["passed"] is False
    assert name in report["failed_checks"]


def test_periodic_image_contact_must_be_zero():
    diag = _good_diagnostics(membrane_periodic_image_contacts=3)
    report = core.evaluate_membrane_quality_gate(_good_observables(), diag)
    assert report["passed"] is False
    assert "membrane_periodic_image_contacts" in report["failed_checks"]


def test_equilibration_shorter_than_lipid_relaxation_is_recorded_not_gated():
    """§9：脂质横向弛豫时间尺度用来**论证**预平衡时长，不当硬门（MEM-11）。

    原先这条断言"弛豫 30 ns 时 10 ns 预平衡一定不过"。2026-08-02 降级理由：
      * §9 原文只要求"记录…并用它论证"，`..._MIN_RELAXATION_MULTIPLE = 1.0`
        这个倍数是本实现自加的（当时注释里就写了"§13 未给此倍数"）；
      * 常规膜蛋白平衡的判据是 APL/膜厚/序参量/RMSD 走平（那些仍是硬门）；
      * τ 是方法依赖量（MSD 拟合窗口/拥挤度/盒尺寸），当硬门会产生假阴性 ——
        实测 memtest 100 ns 上旧估计器让 τ 在 11–38 ns 之间随轨迹长度乱跳。

    ⚠️ 降级**不等于**不记录：比值 < 1 必须在报告里如实出现。
    """
    diag = _good_diagnostics(
        lipid_lateral_relaxation_timescale_ns=30.0, equilibration_length_ns=10.0
    )
    report = core.evaluate_membrane_quality_gate(_good_observables(), diag)
    assert all(
        c["observable"] != "equilibration_length_ns" for c in report["checks"]
    ), "已退役为 diagnostics-only，不得为了让某次运行不过而塞回 checks"
    assert "equilibration_length_ns" not in report["failed_checks"]

    rec = report["statistics"]["equilibration_vs_relaxation"]
    assert rec["is_gate"] is False
    assert rec["equilibration_length_ns"] == pytest.approx(10.0)
    assert rec["lipid_lateral_relaxation_timescale_ns"] == pytest.approx(30.0)
    assert rec["ratio_equilibration_over_relaxation"] == pytest.approx(1.0 / 3.0)
    assert rec["retired_reason"]


def test_report_states_that_more_abfe_windows_is_not_the_fix():
    """§9 末句必须出现在报告里，而不是只写在文档中。"""
    report = core.evaluate_membrane_quality_gate(
        _good_observables(), _good_diagnostics()
    )
    assert "不允许靠增加 ABFE 窗口" in report["remediation"]


# ---------------------------------------------------------------------------
# co-ion 相关（§9 末两条 + §13.1）
# ---------------------------------------------------------------------------


def test_coion_observables_are_required_only_on_the_coion_route():
    # 不要求时缺 co-ion 量也能过。
    core.evaluate_membrane_quality_gate(_good_observables(), _good_diagnostics())
    # 要求时缺就 fail closed。
    with pytest.raises(ValueError, match="缺少必须保存的量"):
        core.evaluate_membrane_quality_gate(
            _good_observables(), _good_diagnostics(), require_coion=True
        )


def test_coion_route_passes_with_full_observables():
    obs = _good_observables(**_coion_observables())
    diag = _good_diagnostics(coion_z_histogram={"bins_nm": [3.0, 4.0], "counts": [5, 9]})
    report = core.evaluate_membrane_quality_gate(obs, diag, require_coion=True)
    assert report["passed"] is True, report["failed_checks"]
    assert report["coion_required"] is True


def test_coion_drifting_toward_the_membrane_fails():
    """§13.1：co-ion 到膜中面的 |z| 必须全程 ≥ 3.0 nm，判的是最坏值不是均值。

    构造一条均值 3.5 nm 但最低点跌到 2.0 nm 的序列——只看均值会漏过。
    """
    coion = _coion_observables()
    t = np.linspace(0.0, 40.0, 81)
    values = np.full_like(t, 4.5)
    values[-3:] = 2.0  # 末段掉进膜里
    coion["coion_abs_z_from_midplane_nm"] = {
        "times_ns": t.tolist(),
        "values": values.tolist(),
    }
    obs = _good_observables(**coion)
    diag = _good_diagnostics(coion_z_histogram={"bins_nm": [2.0], "counts": [3]})
    report = core.evaluate_membrane_quality_gate(obs, diag, require_coion=True)
    assert report["passed"] is False
    assert "coion_abs_z_from_midplane_nm" in report["failed_checks"]
    assert np.mean(values) > core.COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM, (
        "构造前提：均值仍在阈值之上，所以只有判最坏值才能抓到"
    )


def test_coion_too_close_to_ligand_fails():
    coion = _coion_observables()
    coion["coion_ligand_min_image_distance_nm"] = _series(
        core.COION_LIGAND_MIN_IMAGE_RUNTIME_NM - 0.3
    )
    obs = _good_observables(**coion)
    diag = _good_diagnostics(coion_z_histogram={"bins_nm": [4.0], "counts": [9]})
    report = core.evaluate_membrane_quality_gate(obs, diag, require_coion=True)
    assert report["passed"] is False
    assert "coion_ligand_min_image_distance_nm" in report["failed_checks"]


# ---------------------------------------------------------------------------
# §3.3 膜输入核对
# ---------------------------------------------------------------------------


def _membrane_system(n_per_leaflet=4, box=(6.0, 6.0, 10.0), triclinic=False):
    """一个最小膜体系：上下叶各 n 个 POPC（各一个 P 原子）+ 蛋白 + 配体 + 水 + 离子。"""
    topology = app.Topology()
    chain = topology.addChain()
    coords = []

    for leaflet_z in (7.0, 3.0):
        count = n_per_leaflet
        for _ in range(count):
            residue = topology.addResidue("POPC", chain)
            topology.addAtom("P", app.element.phosphorus, residue)
            coords.append([1.0, 1.0, leaflet_z])

    protein = topology.addResidue("ALA", chain)
    topology.addAtom("CA", app.element.carbon, protein)
    coords.append([3.0, 3.0, 5.0])

    ligand = topology.addResidue("MOL", chain)
    topology.addAtom("C1", app.element.carbon, ligand)
    coords.append([3.1, 3.0, 5.0])

    for _ in range(10):
        water = topology.addResidue("SOL", chain)
        topology.addAtom("O", app.element.oxygen, water)
        coords.append([5.0, 5.0, 9.0])

    for name, element in (("NA", app.element.sodium), ("CL", app.element.chlorine)):
        residue = topology.addResidue(name, chain)
        topology.addAtom(name, element, residue)
        coords.append([4.0, 4.0, 9.5])

    box_vectors = np.diag(np.asarray(box, dtype=float))
    if triclinic:
        box_vectors[1][0] = 0.5
    positions = np.asarray(coords, dtype=float) * unit.nanometer
    return topology, positions, box_vectors


def _declared(**overrides):
    declared = {
        "build_tool": "CHARMM-GUI 膜构建器 v3.8",
        "build_parameters": {"lipid": "POPC", "salt_molar": 0.15},
        "final_equilibration_job": "slurm://yayoi/job/123456",
        "source_structure_id": "PDB 5I6X",
        "conformational_state": "outward-open",
        "membrane_composition": {"POPC": 8},
        "binding_site_solvent_exposure": "solvent_accessible",
        "leaflet_assignment_basis": "per-leaflet area matched at 0.65 nm^2/lipid",
        # §9/§15：上游平衡必须有说法。这里给明确时长；
        # 另一条合法表述（已完成但时长不可考）见下面专门的测试。
        "upstream_equilibration_ns": 150.0,
    }
    declared.update(overrides)
    return declared


def test_upstream_equilibration_must_be_declared_one_way_or_another():
    """§9/§15：沉默等于让未充分平衡的体系混进来。"""
    topology, positions, box = _membrane_system()
    declared = _declared()
    del declared["upstream_equilibration_ns"]
    with pytest.raises(ValueError, match="必须说明上游平衡情况"):
        core.validate_membrane_input(topology, positions, box, declared)


def test_completed_but_unrecorded_upstream_equilibration_is_accepted_with_evidence():
    """常见情形：手上只有一个生产末帧（`step7_production.gro`），时长不可考。

    这条合法，但必须显式声明状态并给出证据；此时**标称时长预检不适用**，
    §9 的实测质量门成为唯一判据。
    """
    topology, positions, box = _membrane_system()
    declared = _declared()
    del declared["upstream_equilibration_ns"]
    declared["upstream_equilibration_status"] = "completed_length_unrecorded"
    report = core.validate_membrane_input(topology, positions, box, declared)
    assert report["upstream_equilibration_ns"] is None
    assert report["nominal_equilibration_precheck_applicable"] is False
    assert report["upstream_equilibration_status"] == "completed_length_unrecorded"


def test_completed_but_unrecorded_still_requires_pointing_at_the_evidence():
    topology, positions, box = _membrane_system()
    declared = _declared()
    del declared["upstream_equilibration_ns"]
    declared["upstream_equilibration_status"] = "completed_length_unrecorded"
    declared["final_equilibration_job"] = ""
    with pytest.raises(ValueError, match="final_equilibration_job"):
        core.validate_membrane_input(topology, positions, box, declared)


def test_declared_upstream_length_is_reported_for_the_nominal_precheck():
    topology, positions, box = _membrane_system()
    report = core.validate_membrane_input(topology, positions, box, _declared())
    assert report["upstream_equilibration_ns"] == pytest.approx(150.0)
    assert report["nominal_equilibration_precheck_applicable"] is True


@pytest.mark.parametrize("bad", [-1.0, float("nan"), "很久"])
def test_invalid_upstream_length_is_rejected(bad):
    topology, positions, box = _membrane_system()
    with pytest.raises(ValueError, match="upstream_equilibration_ns"):
        core.validate_membrane_input(
            topology, positions, box, _declared(upstream_equilibration_ns=bad)
        )


def test_valid_membrane_input_passes_and_reports_measured_counts():
    topology, positions, box = _membrane_system()
    report = core.validate_membrane_input(topology, positions, box, _declared())
    assert report["leaflets"]["n_upper"] == 4
    assert report["leaflets"]["n_lower"] == 4
    assert report["leaflets"]["imbalance_fraction"] == pytest.approx(0.0)
    assert report["n_water"] == 10
    assert report["ion_counts"] == {"CL": 1, "NA": 1}
    assert report["box_is_rectangular"] is True
    assert report["protocol_version"] == core.MEMBRANE_INPUT_PROTOCOL_VERSION


@pytest.mark.parametrize("dropped", core.MEMBRANE_INPUT_REQUIRED_PROVENANCE_FIELDS)
def test_missing_provenance_field_fails_closed(dropped):
    """§3.3：来源不可追溯就不许进 ABFE。"""
    topology, positions, box = _membrane_system()
    declared = _declared()
    del declared[dropped]
    with pytest.raises(ValueError, match="缺少 §3.3 要求记录的字段"):
        core.validate_membrane_input(topology, positions, box, declared)


def test_conformational_state_may_be_explicitly_unspecified_but_is_recorded():
    """§1.1 那条是为转运体写的；GPCR 或不区分构象态的体系可以填显式哨兵。

    字段仍必填（不能静默缺失），但 `unspecified` 会被如实记进报告——
    provenance 里看得出"未声明"，而不是被塞了一个编造的构象名。
    """
    topology, positions, box = _membrane_system()
    report = core.validate_membrane_input(
        topology, positions, box, _declared(conformational_state="unspecified")
    )
    assert report["conformational_state"] == "unspecified"
    assert report["conformational_state_declared"] is False

    declared_report = core.validate_membrane_input(
        topology, positions, box, _declared(conformational_state="outward-open")
    )
    assert declared_report["conformational_state_declared"] is True


def test_conformational_state_still_cannot_be_silently_absent():
    """留空/缺失仍然 fail closed —— 显式哨兵与静默缺失是两件事。"""
    topology, positions, box = _membrane_system()
    for bad in ("", None):
        declared = _declared(conformational_state=bad)
        with pytest.raises(ValueError, match="缺少 §3.3 要求记录的字段"):
            core.validate_membrane_input(topology, positions, box, declared)


def test_binding_site_exposure_must_be_one_of_the_declared_levels():
    """§3.0：这一条决定迟滞风险等级，必须在选体系时就写死。"""
    topology, positions, box = _membrane_system()
    with pytest.raises(ValueError, match="binding_site_solvent_exposure"):
        core.validate_membrane_input(
            topology, positions, box, _declared(binding_site_solvent_exposure="有点埋")
        )


def test_membrane_normal_axis_is_measured_not_assumed():
    """§3.3：膜法向必须从坐标实测核对，不能只信盒子形状。

    `MonteCarloMembraneBarostat` 把法向**硬编码为 z**（XY 等比例、Z 独立）。
    若坐标里的双层其实垂直于 x，barostat 会沿膜平面内单独缩放、把法向与一个面内
    方向绑死，膜会被压坏且**不报任何错**。盒子长边与膜法向未必一致，所以只看盒形
    不够。
    """
    topology, positions, box = _membrane_system()
    report = core.verify_membrane_normal_axis(topology, positions, declared_axis="z")
    assert report["agrees"] is True
    assert report["measured_axis"] == "z"
    # 真正的法向应当是"两簇完美平衡"的那个轴。
    assert report["per_axis"]["z"]["balance"] == pytest.approx(1.0)
    assert report["per_axis"]["z"]["separation_nm"] == pytest.approx(
        SYNTHETIC_BILAYER_SEPARATION_NM, abs=1e-6
    )


def test_bilayer_normal_to_x_is_caught_instead_of_silently_squashed():
    """把同一个双层转成垂直于 x，声明 z 时必须报错。"""
    topology, positions, box = _membrane_system()
    coords = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
    # 交换 x 与 z：双层现在垂直于 x。
    swapped = coords[:, [2, 1, 0]] * unit.nanometer
    with pytest.raises(ValueError, match="实测最像双层法向的是 'x'"):
        core.verify_membrane_normal_axis(topology, swapped, declared_axis="z")


def test_normal_axis_report_is_included_in_the_membrane_input_report():
    topology, positions, box = _membrane_system()
    report = core.validate_membrane_input(topology, positions, box, _declared())
    assert report["membrane_normal_axis"]["measured_axis"] == "z"
    assert report["membrane_normal_axis"]["agrees"] is True


def test_too_few_lipids_to_measure_the_normal_fails_closed():
    topology = app.Topology()
    chain = topology.addChain()
    coords = []
    for z in (7.0, 3.0):
        residue = topology.addResidue("POPC", chain)
        topology.addAtom("P", app.element.phosphorus, residue)
        coords.append([1.0, 1.0, z])
    positions = np.asarray(coords, dtype=float) * unit.nanometer
    with pytest.raises(ValueError, match="无法实测膜法向"):
        core.verify_membrane_normal_axis(topology, positions)


def test_triclinic_box_is_rejected_for_membrane():
    """§1.1：膜体系盒型必须是长方体，不得用截角八面体/十二面体。"""
    topology, positions, box = _membrane_system(triclinic=True)
    with pytest.raises(ValueError, match="必须是长方体"):
        core.validate_membrane_input(topology, positions, box, _declared())


def test_atom_count_mismatch_between_coordinates_and_topology_is_rejected():
    topology, positions, box = _membrane_system()
    truncated = np.asarray(positions.value_in_unit(unit.nanometer))[:-1] * unit.nanometer
    with pytest.raises(ValueError, match="与拓扑原子数.*不一致"):
        core.validate_membrane_input(topology, truncated, box, _declared())


def test_declared_leaflet_counts_are_cross_checked_against_measurement():
    """§9：叶片数必须有依据——声明与实测不符即报错，不静默采用声明值。"""
    topology, positions, box = _membrane_system()
    with pytest.raises(ValueError, match="声明 n_upper"):
        core.validate_membrane_input(
            topology, positions, box, _declared(n_upper=7)
        )


def test_declared_water_and_ion_counts_are_cross_checked():
    topology, positions, box = _membrane_system()
    with pytest.raises(ValueError, match="声明水分子数"):
        core.validate_membrane_input(topology, positions, box, _declared(n_water=99))
    with pytest.raises(ValueError, match="声明离子计数"):
        core.validate_membrane_input(
            topology, positions, box, _declared(ion_counts={"NA": 5, "CL": 5})
        )


def test_asymmetric_leaflets_are_reported_not_assumed_to_be_half():
    """不假设对半分：上下叶不等时要如实报出计数与不平衡度。"""
    topology = app.Topology()
    chain = topology.addChain()
    coords = []
    for leaflet_z, count in ((7.0, 5), (3.0, 3)):
        for _ in range(count):
            residue = topology.addResidue("POPC", chain)
            topology.addAtom("P", app.element.phosphorus, residue)
            coords.append([1.0, 1.0, leaflet_z])
    positions = np.asarray(coords, dtype=float) * unit.nanometer
    leaflets = core.assign_lipid_leaflets(topology, positions)
    assert leaflets["n_upper"] == 5
    assert leaflets["n_lower"] == 3
    assert leaflets["imbalance_fraction"] == pytest.approx(2 / 8)


def test_lipids_without_a_head_reference_atom_are_not_silently_dropped():
    topology = app.Topology()
    chain = topology.addChain()
    coords = []
    # 一个有 P 的正常脂质 + 一个原子命名不认识的脂质。
    good = topology.addResidue("POPC", chain)
    topology.addAtom("P", app.element.phosphorus, good)
    coords.append([1.0, 1.0, 7.0])
    weird = topology.addResidue("POPC", chain)
    topology.addAtom("ZZ9", app.element.carbon, weird)
    coords.append([1.0, 1.0, 3.0])
    positions = np.asarray(coords, dtype=float) * unit.nanometer

    leaflets = core.assign_lipid_leaflets(topology, positions)
    assert len(leaflets["unassignable_lipid_residues"]) == 1

    box = np.diag([6.0, 6.0, 10.0])
    # 报错必须是"有脂质找不到头基参考原子"这条更**具体**的，
    # 而不是"头基太少、法向不可测"——后者是前者的下游后果，
    # 先报下游会让人去查错的地方（所以 validate_membrane_input 里
    # 叶片划分排在法向核对之前）。
    with pytest.raises(ValueError, match="找不到头基参考原子"):
        core.validate_membrane_input(topology, positions, box, _declared())


def test_no_lipid_head_atoms_at_all_fails_loudly():
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("POPC", chain)
    topology.addAtom("ZZ9", app.element.carbon, residue)
    positions = np.asarray([[1.0, 1.0, 5.0]], dtype=float) * unit.nanometer
    with pytest.raises(ValueError, match="找不到任何可用的脂质头基参考原子"):
        core.assign_lipid_leaflets(topology, positions)
