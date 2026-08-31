"""B6 + §1.1 + §13：色散路线、力场族识别、验收阈值常量。

对应 docs/status/memtodolist.md：
  §1.1  力场族按输入自动识别（amber 首选，charmm 因 OpenMM 无 force-switch 默认卡住），
        识别不出 fail closed，允许显式覆盖但必须留记录。
  §1.3  `dispersion_protocol` 显式配置；membrane 未选已验证路线 → fail closed；
        membrane + legacy 均匀密度 LRC → fail closed（§6.4 同条）。
  §13   全部"预设阈值"落成命名常量并进 provenance，不许运行时凭感觉判。
  §7.1  「membrane + 未指定 dispersion protocol：失败」这一条。
  §7.7  soluble 不声明时行为不变。

全部 CPU 可跑：纯校验层 + 文本解析，不建 System/Context。
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

openmm = pytest.importorskip("openmm")
from openmm import unit

import abfe_core as core

ROOT = Path(__file__).absolute().parents[1]


# ---------------------------------------------------------------------------
# §1.1 力场族识别
# ---------------------------------------------------------------------------


def test_production_topology_is_detected_as_amber():
    """实测：本仓库 `topol.top` 走 `amber14sb_OL15_fs1.ff` → amber。

    注意它**没有 `[ defaults ]` 段**（那在 forcefield.itp 里），所以主判据只能是
    include 路径。这条测试同时钉住"本地 include 不参与判定"。
    """
    result = core.detect_forcefield_family_from_top(str(ROOT / "tests/fixtures/topol.top"))
    assert result["family"] == "amber"
    assert result["reason"] == "single_family_include"
    assert result["defaults_row"] is None
    # 本地 include 出现在 includes 里但不构成族证据。
    assert "./Atenolol-rank1.itp" in result["includes"]
    assert set(result["family_evidence"]) == {"amber"}


def _write_top(tmp_path, includes, defaults=None):
    lines = ["; test topology", ""]
    if defaults is not None:
        lines += ["[ defaults ]", "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ", defaults]
    lines += [f'#include "{inc}"' for inc in includes]
    path = tmp_path / "topol.top"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_charmm_topology_is_detected(tmp_path):
    top = _write_top(
        tmp_path,
        ["charmm36.ff/forcefield.itp", "./ligand.itp", "charmm36.ff/tip3p.itp"],
        defaults="1               2               yes             1.0     1.0",
    )
    result = core.detect_forcefield_family_from_top(top)
    assert result["family"] == "charmm"
    # [ defaults ] 若存在则记录下来作交叉检查，但不是唯一判据。
    assert result["defaults_row"].startswith("1")


def test_mixed_family_includes_fail_closed(tmp_path):
    top = _write_top(tmp_path, ["amber14sb.ff/forcefield.itp", "charmm36.ff/tip3p.itp"])
    with pytest.raises(ValueError, match="无法从"):
        core.resolve_forcefield_family(top_path=top)


def test_unrecognized_forcefield_dir_fails_closed_and_does_not_default_to_amber(tmp_path):
    """§1.1：识别不出时 **fail closed，不许猜、不许默认回落到 amber**。"""
    top = _write_top(tmp_path, ["my_custom_ff/forcefield.itp", "./ligand.itp"])
    with pytest.raises(ValueError, match="不许猜、不许默认回落到 amber"):
        core.resolve_forcefield_family(top_path=top)


def test_explicit_override_is_recorded_not_silent(tmp_path):
    """§1.1：允许显式覆盖，但覆盖必须留记录。"""
    top = _write_top(tmp_path, ["amber14sb.ff/forcefield.itp"])
    result = core.resolve_forcefield_family(top_path=top, explicit_family="charmm")
    assert result["family"] == "charmm"
    assert result["source"] == "explicit_override"
    assert result["overrode_detection"] == "amber"
    assert result["detection"]["family"] == "amber"


def test_unsupported_forcefield_family_is_rejected(tmp_path):
    top = _write_top(tmp_path, ["oplsaa.ff/forcefield.itp"])
    with pytest.raises(ValueError, match="本轮不支持"):
        core.resolve_forcefield_family(top_path=top)
    with pytest.raises(ValueError, match="本轮不支持"):
        core.resolve_forcefield_family(explicit_family="gromos")


# ---------------------------------------------------------------------------
# §1.3 / B6 色散路线
# ---------------------------------------------------------------------------


def test_soluble_defaults_to_legacy_uniform_lrc_so_behaviour_is_unchanged():
    """§7.7：可溶体系不声明时仍是改动前的 `lrc_coeff/V` 行为。"""
    payload = core.resolve_dispersion_protocol(None)
    assert payload["dispersion_protocol"] == "legacy_uniform_density_lrc"
    assert payload["was_defaulted"] is True
    assert payload["uniform_density_lrc_active"] is True
    assert payload["implemented"] is True


def test_membrane_without_dispersion_protocol_fails_closed():
    """§7.1：membrane + 未指定 dispersion protocol → 失败。"""
    with pytest.raises(ValueError, match="没有选择 dispersion_protocol"):
        core.resolve_dispersion_protocol(None, environment_type="membrane")


def test_membrane_error_message_names_the_expected_protocol_for_the_family():
    """报错要能直接指路，而不是只说"你没选"。"""
    with pytest.raises(ValueError, match="ff_native_isotropic_lrc"):
        core.resolve_dispersion_protocol(
            None, environment_type="membrane", forcefield_family="amber"
        )


def test_membrane_with_legacy_uniform_lrc_fails_closed():
    """§1.3 修正框第 1 条 / §6.4：均匀体相密度假设在膜口袋里直接不成立。"""
    with pytest.raises(ValueError, match="均匀体相密度"):
        core.resolve_dispersion_protocol(
            "legacy_uniform_density_lrc", environment_type="membrane"
        )


def test_membrane_with_amber_native_lrc_passes():
    """⚠️ 注意这是 §1.3 修正框的要点：Amber 脂质**必须**开各向同性 LRC，关掉才是错的。"""
    payload = core.resolve_dispersion_protocol(
        "ff_native_isotropic_lrc",
        environment_type="membrane",
        forcefield_family="amber",
    )
    assert payload["dispersion_protocol"] == "ff_native_isotropic_lrc"
    assert payload["uniform_density_lrc_active"] is False
    assert payload["implemented"] is True


def test_charmm_force_switch_fails_closed_without_quantitative_evidence():
    """OpenMM 只有 potential-switch，没有 force-switch —— 默认卡住。"""
    with pytest.raises(ValueError, match="没有.*force-switch"):
        core.resolve_dispersion_protocol(
            "ff_native_force_switch_no_lrc",
            forcefield_family="charmm",
        )


@pytest.mark.parametrize("dropped", core.FORCE_SWITCH_DEVIATION_EVIDENCE_FIELDS)
def test_charmm_force_switch_needs_every_evidence_field(dropped):
    evidence = {f: "/path" for f in core.FORCE_SWITCH_DEVIATION_EVIDENCE_FIELDS}
    del evidence[dropped]
    with pytest.raises(ValueError, match="缺少"):
        core.resolve_dispersion_protocol(
            "ff_native_force_switch_no_lrc",
            forcefield_family="charmm",
            force_switch_deviation_evidence=evidence,
        )


def test_charmm_force_switch_still_not_membrane_validated_even_with_evidence():
    """有了论证也只是解开 force-switch 那道锁；membrane 已验证集合仍只含 amber 路线。"""
    evidence = {f: "/path" for f in core.FORCE_SWITCH_DEVIATION_EVIDENCE_FIELDS}
    with pytest.raises(ValueError, match="尚未验证"):
        core.resolve_dispersion_protocol(
            "ff_native_force_switch_no_lrc",
            environment_type="membrane",
            forcefield_family="charmm",
            force_switch_deviation_evidence=evidence,
        )


@pytest.mark.parametrize(
    "protocol, route",
    [("lj_pme", "路线 B"), ("membrane_inhomogeneous", "路线 C")],
)
def test_routes_b_and_c_are_recognized_but_not_implemented(protocol, route):
    """声明未实现的路线要报 NotImplementedError，而不是被当成拼错的未知值。

    §1.3 明确写着：路线 B 在四项验收完成前"不能只把基础 NonbondedForce 切成
    LJPME 就宣称支持"。
    """
    with pytest.raises(NotImplementedError, match=route):
        core.resolve_dispersion_protocol(protocol)


def test_family_and_protocol_mismatch_is_rejected():
    """跟随力场原始参数化条件：amber 配 force-switch、charmm 配 isotropic LRC 都是错的。"""
    with pytest.raises(ValueError, match="原始参数化条件"):
        core.resolve_dispersion_protocol(
            "ff_native_force_switch_no_lrc",
            forcefield_family="amber",
            force_switch_deviation_evidence={
                f: "/path" for f in core.FORCE_SWITCH_DEVIATION_EVIDENCE_FIELDS
            },
        )


def test_misspelled_dispersion_protocol_fails():
    with pytest.raises(ValueError, match="非法"):
        core.resolve_dispersion_protocol("no_lrc")


def test_apbs_is_recorded_as_orthogonal_to_dispersion():
    """§5 末条 / §6.5：APBS 只管静电有限尺寸项，不得当成 LJ 修正。"""
    payload = core.resolve_dispersion_protocol(None)
    assert payload["apbs_is_orthogonal_to_dispersion"] is True


# ---------------------------------------------------------------------------
# §13 阈值常量
# ---------------------------------------------------------------------------


def test_coion_runtime_distance_is_independent_of_softcore_cutoff():
    """[MEM-00h，2026-08-06 复核后拍板] 这两个值**不再要求相等**。

    旧版本把 `COION_LIGAND_MIN_IMAGE_RUNTIME_NM` 定义成"= softcore cutoff"的
    派生量；这次统一非键协议时用户明确决定解耦：co-ion 全程距离门是一条独立、
    保守的几何安全门，不用为了让 softcore cutoff 收敛到 1.0 nm 就跟着降到
    1.0——1.2 nm 本来就更保守，没必要降。

    这条测试只钉住"当前这条门槛仍然是 1.2 nm、且≥ softcore cutoff"这两件事，
    不再断言两者相等；`ibs_engine.SOFTCORE_CUTOFF_NM` 以后再变，这条测试也
    不应该因此失败。
    """
    import ibs_engine

    assert core.COION_LIGAND_MIN_IMAGE_RUNTIME_NM == pytest.approx(1.2)
    assert core.COION_LIGAND_MIN_IMAGE_RUNTIME_NM >= ibs_engine.SOFTCORE_CUTOFF_NM


def test_initial_coion_distance_is_stricter_than_runtime():
    """选择时留余量，否则第一帧就贴着下限、稍一抖动就破门。"""
    assert (
        core.COION_LIGAND_MIN_IMAGE_INITIAL_NM
        > core.COION_LIGAND_MIN_IMAGE_RUNTIME_NM
    )


def test_flat_bottom_restraint_replaces_the_legacy_bare_harmonic():
    """§13.1 对比现状 MEM-00d：当前是 k=25 的纯谐振子、无平坦区。"""
    assert core.COION_FLAT_BOTTOM_RADIUS_NM > 0.0
    assert core.COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2 == pytest.approx(100.0)
    assert core.COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2 > 25.0


def test_decoupled_endpoint_tolerance_is_strict_zero_not_merely_small():
    """§13.2：λ=0 端 ligand–environment 能量是**严格零**，不是"很小"。"""
    assert core.DECOUPLED_ENDPOINT_ENERGY_ABS_TOLERANCE_KJ_PER_MOL <= 1.0e-6


def test_acceptance_thresholds_payload_is_json_serializable_and_versioned():
    import json

    payload = core.acceptance_thresholds_payload()
    assert payload["version"] == core.ACCEPTANCE_THRESHOLDS_VERSION
    json.dumps(payload)
    for section in (
        "coion_geometry",
        "numerical_selfconsistency",
        "membrane_quality_gate",
        "result_acceptance",
    ):
        assert payload[section], f"{section} 不能为空"


def test_thresholds_payload_reflects_the_constants_not_a_hardcoded_copy():
    """payload 必须读常量，否则改了常量而 provenance 还报旧值。"""
    payload = core.acceptance_thresholds_payload()
    assert payload["coion_geometry"]["ligand_min_image_runtime_nm"] == (
        core.COION_LIGAND_MIN_IMAGE_RUNTIME_NM
    )
    assert payload["result_acceptance"]["min_independent_repeats"] == (
        core.MIN_INDEPENDENT_REPEATS
    )
    assert payload["numerical_selfconsistency"]["total_charge_conservation_e"] == (
        core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E
    )


# ---------------------------------------------------------------------------
# MEM-00h 物理验收：1.0 nm、无 switching、端点与零力门
# ---------------------------------------------------------------------------


def _two_particle_nb(*, charge_lig=0.0, charge_env=0.0, use_lrc=False):
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.0 * unit.nanometer)
    nb.setUseSwitchingFunction(False)
    nb.setUseDispersionCorrection(bool(use_lrc))
    nb.addParticle(
        charge_lig * unit.elementary_charge,
        0.34 * unit.nanometer,
        0.40 * unit.kilojoule_per_mole,
    )
    nb.addParticle(
        charge_env * unit.elementary_charge,
        0.30 * unit.nanometer,
        0.50 * unit.kilojoule_per_mole,
    )
    return nb


def _evaluate_single_force(force, positions_nm, box_nm=4.0):
    system = openmm.System()
    for _ in range(len(positions_nm)):
        system.addParticle(12.0 * unit.dalton)
    vectors = (
        openmm.Vec3(box_nm, 0.0, 0.0),
        openmm.Vec3(0.0, box_nm, 0.0),
        openmm.Vec3(0.0, 0.0, box_nm),
    ) * unit.nanometer
    system.setDefaultPeriodicBoxVectors(*vectors)
    force.setForceGroup(1)
    system.addForce(force)
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(np.asarray(positions_nm, dtype=float) * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True, groups={1})
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    del context, integrator
    return float(energy), np.asarray(forces, dtype=float)


def _ace_softcore_force(nb, ligand, environment, lambda_vdw):
    import ibs_engine

    params = core.ACESoftcorePotential.from_dict(
        core.ACESoftcorePotential.optimize_alpha(len(ligand))
    )
    return ibs_engine._create_softcore_force(
        nb,
        ligand,
        environment,
        lam_coul=0.0,
        lam_vdw=float(lambda_vdw),
        alchemical_params=params,
        potential_type="softcore",
    )


def test_softcore_and_reconstruction_builders_use_the_physical_1nm_no_switch_protocol():
    nb = _two_particle_nb()
    beutler = core.BeutlerSoftcoreBuilder.build(nb, [0], [1])
    ligand_internal, _ = core.create_ligand_internal_force(
        nb,
        [0, 1],
        [nb.getParticleParameters(i) for i in range(2)],
        num_particles=2,
    )
    for force in (beutler, ligand_internal):
        assert force.getCutoffDistance().value_in_unit(unit.nanometer) == pytest.approx(1.0)
        assert force.getUseSwitchingFunction() is False


def test_lambda_vdw_one_matches_original_nonbonded_energy_and_forces_with_lrc():
    """λvdW=1 的 CV + 项目 LRC 必须回到原始 1.0 nm LJ Hamiltonian。

    OpenMM 的 ``NonbondedForce`` 原生 LRC 是 whole-system correction，而这里的
    alchemical CV 只含 ligand-environment interaction group。用它直接和 cross
    term 的手算 LRC 比，会把非 cross 的 finite-system correction 混进参考值；
    因此短程部分与原始 NonbondedForce 对比，LRC 部分单独按项目的 cross-term
    协议加回。
    """
    import ibs_engine

    positions = [[1.5, 2.0, 2.0], [2.4, 2.0, 2.0]]  # r=0.9 nm
    nb_reference = _two_particle_nb(use_lrc=False)
    ref_energy, ref_forces = _evaluate_single_force(nb_reference, positions)

    nb_for_cv = _two_particle_nb(use_lrc=False)
    cv = _ace_softcore_force(nb_for_cv, [0], [1], lambda_vdw=1.0)
    cv_energy, cv_forces = _evaluate_single_force(cv, positions)

    all_params = [nb_for_cv.getParticleParameters(i) for i in range(2)]
    sigma, s6, s12 = ibs_engine._lj_tail_correction_sigma_resolved_moments(
        all_params, [0], [1]
    )
    alpha = core.ACESoftcorePotential.from_dict(
        core.ACESoftcorePotential.optimize_alpha(1)
    )
    lrc_coeff = ibs_engine._lj_tail_lrc_coefficients_kj_mol(
        [1.0], sigma, s6, s12, alpha.alpha_lj, alpha.m_lj, alpha.n_lj, 1.0, 1.0
    )[0]
    cv_plus_lrc = cv_energy + lrc_coeff / (4.0 ** 3)

    assert cv_energy == pytest.approx(ref_energy, rel=2e-6, abs=2e-6)
    assert cv_plus_lrc == pytest.approx(
        ref_energy + lrc_coeff / (4.0 ** 3), rel=2e-6, abs=2e-6
    )
    # The uniform analytic tail has no coordinate derivative, so the raw CV
    # force must equal the native NonbondedForce force exactly at this point.
    assert cv_forces == pytest.approx(ref_forces, rel=2e-6, abs=2e-6)


def test_softcore_force_is_zero_strictly_outside_1nm_and_at_lambda_zero():
    """Dedicated 0.9/1.05/1.1 nm probes pin the hard 1.0 nm boundary."""
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1.0 * unit.nanometer)
    nb.setUseSwitchingFunction(False)
    for charge in (0.25, -0.30, 0.10, -0.05):
        nb.addParticle(
            charge * unit.elementary_charge,
            0.32 * unit.nanometer,
            0.45 * unit.kilojoule_per_mole,
        )

    positions = [
        [2.0, 2.0, 2.0],
        [2.9, 2.0, 2.0],   # 0.90 nm: inside
        [3.05, 2.0, 2.0],  # 1.05 nm: outside
        [3.10, 2.0, 2.0],  # 1.10 nm: outside
    ]
    force = _ace_softcore_force(nb, [0], [1, 2, 3], lambda_vdw=0.5)
    energy, forces = _evaluate_single_force(force, positions, box_nm=5.0)
    assert energy != pytest.approx(0.0)
    assert np.max(np.abs(forces[0])) > 0.0
    assert np.array_equal(forces[2:], np.zeros_like(forces[2:]))

    # The force carries all particles from ``nb``.  Use a matching two-particle
    # reference for the separate λ=0 endpoint context.
    zero_nb = _two_particle_nb()
    zero_force = _ace_softcore_force(zero_nb, [0], [1], lambda_vdw=0.0)
    zero_energy, zero_forces = _evaluate_single_force(zero_force, positions[:2], box_nm=5.0)
    assert zero_energy == 0.0
    assert np.array_equal(zero_forces, np.zeros_like(zero_forces))


def test_every_production_softcore_cv_state_inherits_cutoff_and_switching_template():
    """Catches the old per-λ unconditional setUseSwitchingFunction(True) bug."""
    import ibs_engine

    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(4.0, 0.0, 0.0) * unit.nanometer,
        openmm.Vec3(0.0, 4.0, 0.0) * unit.nanometer,
        openmm.Vec3(0.0, 0.0, 4.0) * unit.nanometer,
    )
    for _ in range(2):
        system.addParticle(12.0 * unit.dalton)
    nb = _two_particle_nb()
    system.addForce(nb)
    new_system, wrapper = ibs_engine.build_ibs_dual_system(
        system,
        topology=None,
        perturbed_indices=[0],
        lambdas_coul=[0.0, 0.0, 0.0],
        lambdas_vdw=[1.0, 0.5, 0.0],
        alchemical_params=core.ACESoftcorePotential.from_dict(
            core.ACESoftcorePotential.optimize_alpha(1)
        ),
        potential_type="softcore",
        box_vectors=system.getDefaultPeriodicBoxVectors(),
    )
    assert new_system.getNumParticles() == 2
    assert len(wrapper._int_cv_force_xmls) == 3
    for force_xml in wrapper._int_cv_force_xmls:
        force = openmm.XmlSerializer.deserialize(force_xml)
        assert force.getCutoffDistance().value_in_unit(unit.nanometer) == pytest.approx(1.0)
        assert force.getSwitchingDistance().value_in_unit(unit.nanometer) == pytest.approx(1.0)
        assert force.getUseSwitchingFunction() is False
