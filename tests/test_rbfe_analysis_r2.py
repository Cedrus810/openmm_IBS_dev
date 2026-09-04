"""RBFE R2：对真实 hybrid Hamiltonian 求 u_kn + MBAR，以及独立窗口采样。

设计依据：`docs/design/PLAN_rbfe_interface_and_implementation.md` §6 / §7 / §8。

计划 §8 的 R2 交付物是「通用采样接口及 RBFE 单腿」，验收标准：

    **独立窗口先跑通，再验证 REMD；真实 hybrid 能量和样本归属一致。**

§6 同时划了两条线：**不能**复用 `TraditionalMBARAnalyzer.compute_u_kn`
（它按单配体去耦和 LRC 假设重建评估系统），但**可以**复用「独立的 MBAR 数值求解
部分」。本文件两条都测。

## 为什么大部分用例不跑 MD

`A→A` 自边的所有 λ 态 Hamiltonian **逐位相同** ⟹ u_kn 每一行都一样 ⟹ MBAR 恒等于 0。
这个性质对**任意**样本成立，跟采样质量无关。于是"分析层对不对"可以用合成样本
（确定性抖动的几何）验到底，既严格又快；只留一条真 MD 的贯通用例。
"""

from __future__ import annotations

import numpy as np
import openmm
import pytest
from openmm import unit

import free_energy_engine as fe
import rbfe_core as rc
from test_rbfe_hybrid_r1b import POSITIONS, make_mapping, make_system

BOX = [
    openmm.Vec3(3, 0, 0) * unit.nanometer,
    openmm.Vec3(0, 3, 0) * unit.nanometer,
    openmm.Vec3(0, 0, 3) * unit.nanometer,
]
BOX_NM = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


def identity_mapping(n_atoms: int = 5) -> rc.AtomMapping:
    """A→A 用的恒等映射：全是 core，没有 dummy。"""
    return rc.AtomMapping(
        protocol_version=rc.RBFE_MAPPING_PROTOCOL_VERSION,
        n_atoms_a=n_atoms, n_atoms_b=n_atoms,
        core_pairs=tuple((i, i) for i in range(n_atoms)),
        a_only=(), b_only=(),
        fragment_pairs=(), method="identity_self_edge",
        symmetry_solution_counts=(), ambiguities=(),
        atom_identity_a=(), atom_identity_b=(),
    )


@pytest.fixture
def ab_bundle():
    system_a, ligand_a = make_system("A")
    system_b, ligand_b = make_system("B")
    return rc.build_hybrid_system(
        system_a, ligand_a, system_b, ligand_b, make_mapping(),
        rc.HybridLambdaSchedule.linear(4),
    )


@pytest.fixture
def self_bundle():
    system_a, ligand_a = make_system("A")
    system_a2, ligand_a2 = make_system("A")
    return rc.build_hybrid_system(
        system_a, ligand_a, system_a2, ligand_a2, identity_mapping(),
        rc.HybridLambdaSchedule.linear(4),
    )


def synthetic_samples(bundle, *, n_frames: int = 6, pressure_bar=None, jitter: float = 0.01):
    """确定性抖动出来的"样本"。

    MBAR 的正确性判据（A→A 为零、u_kn 对角一致）与样本是否服从玻尔兹曼分布无关，
    所以这里不跑 MD——省下的时间比测试覆盖更值钱，而严格性一点没丢。
    """
    n_atoms = bundle.layout.n_particles
    base = POSITIONS[:n_atoms]
    rng = np.random.default_rng(20260903)
    positions, boxes, parameters = [], [], []
    for state in range(bundle.schedule.n_states):
        frames = np.asarray(
            [base + jitter * rng.standard_normal(base.shape) for _ in range(n_frames)]
        )
        positions.append(frames)
        boxes.append(np.asarray([BOX_NM] * n_frames))
        parameters.append(bundle.schedule.state(state))
    return fe.InMemoryWindowSamples(
        positions_by_state=tuple(positions),
        box_vectors_by_state=tuple(boxes),
        state_parameters=tuple(parameters),
        step_plan=fe.resolve_step_plan(1000, 100, 100),
        temperature_kelvin=300.0,
        pressure_bar=pressure_bar,
        seeds_by_state=tuple(range(bundle.schedule.n_states)),
        provenance={"sampler": "synthetic_deterministic_jitter"},
    )


# ---------------------------------------------------------------------------
# u_kn：真实 hybrid Hamiltonian，样本归属靠下标
# ---------------------------------------------------------------------------


def test_u_kn_shape_and_sample_attribution(ab_bundle):
    samples = synthetic_samples(ab_bundle, n_frames=5)
    matrix = rc.compute_hybrid_u_kn(ab_bundle, samples, temperature_kelvin=300.0)

    n_states = ab_bundle.schedule.n_states
    assert matrix["u_kn"].shape == (n_states, n_states * 5)
    assert matrix["n_k"].tolist() == [5] * n_states
    # 样本归属由下标承载，不靠文件名（计划 §7）
    assert matrix["sample_state_index"].tolist() == sum(([k] * 5 for k in range(n_states)), [])
    assert matrix["reduced"] is True


def test_u_kn_is_evaluated_on_the_hybrid_system_without_rebuilding(ab_bundle):
    """§6 禁止复用 ABFE 的 compute_u_kn——它为每个 λ 重建一个"等效去耦体系"。

    RBFE 这边结构上不可能犯那个错：hybrid System 只有一个，换态就是 setParameter。
    """
    samples = synthetic_samples(ab_bundle, n_frames=3)
    matrix = rc.compute_hybrid_u_kn(ab_bundle, samples, temperature_kelvin=300.0)
    assert matrix["evaluated_on"] == "hybrid_system_via_setParameter__no_system_rebuild"
    assert matrix["hybrid_fingerprint"] == ab_bundle.fingerprint()


def test_u_kn_matches_an_independent_energy_evaluation(ab_bundle):
    """逐格核对：u_kn[l, n] 必须等于 β·U(帧 n, λ 表第 l 态)，独立算一遍。"""
    samples = synthetic_samples(ab_bundle, n_frames=3)
    matrix = rc.compute_hybrid_u_kn(ab_bundle, samples, temperature_kelvin=300.0)
    beta = matrix["beta_mol_per_kJ"]

    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(ab_bundle.system, integrator,
                             openmm.Platform.getPlatformByName("Reference"))
    column = 0
    for k in range(ab_bundle.schedule.n_states):
        for frame in samples.positions_by_state[k]:
            context.setPeriodicBoxVectors(*BOX)
            context.setPositions(frame * unit.nanometer)
            for evaluated in range(ab_bundle.schedule.n_states):
                for name, value in ab_bundle.schedule.state(evaluated).items():
                    context.setParameter(name, value)
                energy = context.getState(getEnergy=True).getPotentialEnergy()
                expected = beta * energy.value_in_unit(unit.kilojoule_per_mole)
                assert matrix["u_kn"][evaluated, column] == pytest.approx(expected, rel=1e-12)
            column += 1
    del context, integrator


def test_u_kn_rejects_samples_taken_on_a_different_lambda_table(ab_bundle):
    """样本采样时的 λ 与评估用的 λ 必须来自同一张表——对不上就拒绝，不迁就。"""
    samples = synthetic_samples(ab_bundle, n_frames=2)
    wrong = list(samples.state_parameters)
    wrong[1] = dict(wrong[1])
    wrong[1][rc.LAMBDA_STERICS] = 0.987654
    broken = fe.InMemoryWindowSamples(
        positions_by_state=samples.positions_by_state,
        box_vectors_by_state=samples.box_vectors_by_state,
        state_parameters=tuple(wrong),
        step_plan=samples.step_plan, temperature_kelvin=300.0, pressure_bar=None,
        seeds_by_state=samples.seeds_by_state, provenance={},
    )
    with pytest.raises(rc.RBFEHybridBuildError, match="λ 与 λ 表不一致"):
        rc.compute_hybrid_u_kn(ab_bundle, broken, temperature_kelvin=300.0)


def test_u_kn_rejects_state_count_mismatch(ab_bundle):
    samples = synthetic_samples(ab_bundle, n_frames=2)
    broken = fe.InMemoryWindowSamples(
        positions_by_state=samples.positions_by_state[:-1],
        box_vectors_by_state=samples.box_vectors_by_state[:-1],
        state_parameters=samples.state_parameters[:-1],
        step_plan=samples.step_plan, temperature_kelvin=300.0, pressure_bar=None,
        seeds_by_state=samples.seeds_by_state[:-1], provenance={},
    )
    with pytest.raises(rc.RBFEHybridBuildError, match="λ 表有"):
        rc.compute_hybrid_u_kn(ab_bundle, broken, temperature_kelvin=300.0)


def test_ensemble_mismatch_is_rejected_in_both_directions(ab_bundle):
    """NPT 样本按 NVT 约化会漏掉 βpV，反过来会凭空多一项。两个方向都要拒。"""
    nvt = synthetic_samples(ab_bundle, n_frames=2, pressure_bar=None)
    npt = synthetic_samples(ab_bundle, n_frames=2, pressure_bar=1.0)
    with pytest.raises(rc.RBFEHybridBuildError, match="系综不一致"):
        rc.compute_hybrid_u_kn(ab_bundle, nvt, temperature_kelvin=300.0, pressure_bar=1.0)
    with pytest.raises(rc.RBFEHybridBuildError, match="系综不一致"):
        rc.compute_hybrid_u_kn(ab_bundle, npt, temperature_kelvin=300.0, pressure_bar=None)


def test_npt_adds_beta_p_v_and_says_so(ab_bundle):
    """计划 §7：「是否约化和**是否包含 NPT 所需项**」必须记录，不让下游猜。"""
    nvt = rc.compute_hybrid_u_kn(
        ab_bundle, synthetic_samples(ab_bundle, n_frames=2), temperature_kelvin=300.0
    )
    npt = rc.compute_hybrid_u_kn(
        ab_bundle, synthetic_samples(ab_bundle, n_frames=2, pressure_bar=1.0),
        temperature_kelvin=300.0, pressure_bar=1.0,
    )
    assert nvt["includes_pV"] is False and npt["includes_pV"] is True

    volume = abs(np.linalg.det(BOX_NM))
    expected = nvt["beta_mol_per_kJ"] * 1.0 * volume * 0.0602214076
    assert float(np.max(npt["u_kn"] - nvt["u_kn"])) == pytest.approx(expected, rel=1e-9)
    assert float(np.min(npt["u_kn"] - nvt["u_kn"])) == pytest.approx(expected, rel=1e-9)


def test_empty_state_is_rejected(ab_bundle):
    samples = synthetic_samples(ab_bundle, n_frames=2)
    positions = list(samples.positions_by_state)
    positions[0] = np.zeros((0, ab_bundle.layout.n_particles, 3))
    broken = fe.InMemoryWindowSamples(
        positions_by_state=tuple(positions),
        box_vectors_by_state=samples.box_vectors_by_state,
        state_parameters=samples.state_parameters,
        step_plan=samples.step_plan, temperature_kelvin=300.0, pressure_bar=None,
        seeds_by_state=samples.seeds_by_state, provenance={},
    )
    with pytest.raises(rc.RBFEHybridBuildError, match="一帧都没有"):
        rc.compute_hybrid_u_kn(ab_bundle, broken, temperature_kelvin=300.0)


# ---------------------------------------------------------------------------
# A→A 自边必须为零（R3 那条验收在分析层的落点）
# ---------------------------------------------------------------------------


def test_self_edge_u_kn_rows_are_identical(self_bundle):
    """A→A 的所有 λ 态 Hamiltonian 逐位相同 ⟹ u_kn 每一行都一样。

    这是"自边为零"的**机制**，先把它钉死；下一条再验最终数字。
    """
    matrix = rc.compute_hybrid_u_kn(
        self_bundle, synthetic_samples(self_bundle, n_frames=4), temperature_kelvin=300.0
    )
    u_kn = matrix["u_kn"]
    for row in range(1, u_kn.shape[0]):
        assert np.allclose(u_kn[row], u_kn[0], rtol=0, atol=1e-9)


def test_self_edge_free_energy_is_exactly_zero(self_bundle):
    """计划 §8 的 R3 验收「**A→A 为零**」在分析层的落点。

    对**任意**样本都成立，与采样质量无关——所以这里用合成样本，不跑 MD。
    """
    result, diagnostics = rc.analyze_leg(
        self_bundle, synthetic_samples(self_bundle, n_frames=6),
        phase="solvent", edge_id="self", ligand_a_name="A", ligand_b_name="A",
        temperature_kelvin=300.0, decorrelate=False,
    )
    assert abs(result.delta_g) < 1e-6, f"A→A 必须为零，得到 {result.delta_g}"
    assert result.stderr < 1e-5
    assert diagnostics["n_k"] == [6] * self_bundle.schedule.n_states


# ---------------------------------------------------------------------------
# analyze_leg → LegResult
# ---------------------------------------------------------------------------


def test_leg_result_direction_matches_the_plan(ab_bundle):
    """方向链：MBAR 的 `delta_G` = f[-1]−f[0] = G(λ=1)−G(λ=0) = G(B)−G(A) = ΔG(A→B)。

    这条链跨了三个模块（λ 表端点 → solve() 的取法 → LegResult 的语义），
    靠记忆维持迟早出错。这里用一个符号已知的构造把它钉死：

    A→A 自边把所有 λ 态做成同一个 Hamiltonian，ΔG 必须是 0；若方向链哪一环反了，
    自边仍然是 0（对称），所以还要一条**非对称**的判据——用 `LegResult.direction`
    与 `interpretation()` 的文字口径一起锁。
    """
    result, _ = rc.analyze_leg(
        ab_bundle, synthetic_samples(ab_bundle, n_frames=5),
        phase="complex", edge_id="e1", ligand_a_name="LIG_A", ligand_b_name="LIG_B",
        temperature_kelvin=300.0, decorrelate=False,
    )
    assert result.direction == rc.RBFE_DIRECTION == "A_to_B"
    assert result.phase == "complex"
    assert result.ligand_a_name == "LIG_A" and result.ligand_b_name == "LIG_B"
    assert np.isfinite(result.delta_g) and result.stderr >= 0


def test_leg_result_can_be_reported_in_kcal(ab_bundle):
    samples = synthetic_samples(ab_bundle, n_frames=5)
    in_kj, _ = rc.analyze_leg(
        ab_bundle, samples, phase="solvent", edge_id="e", ligand_a_name="A",
        ligand_b_name="B", temperature_kelvin=300.0, decorrelate=False,
    )
    in_kcal, _ = rc.analyze_leg(
        ab_bundle, samples, phase="solvent", edge_id="e", ligand_a_name="A",
        ligand_b_name="B", temperature_kelvin=300.0, energy_unit=rc.KCAL_PER_MOL,
        decorrelate=False,
    )
    assert in_kcal.energy_unit == rc.KCAL_PER_MOL
    assert in_kcal.delta_g == pytest.approx(in_kj.delta_g / 4.184, rel=1e-9)
    assert in_kcal.stderr == pytest.approx(in_kj.stderr / 4.184, rel=1e-9)


def test_leg_result_feeds_combine_rbfe(ab_bundle):
    """R2 的产物必须能直接喂进 R0 的 ΔΔG 汇总——两层接得上才算接通。"""
    samples = synthetic_samples(ab_bundle, n_frames=5)
    complex_leg, _ = rc.analyze_leg(
        ab_bundle, samples, phase="complex", edge_id="e", ligand_a_name="A",
        ligand_b_name="B", temperature_kelvin=300.0, decorrelate=False,
    )
    solvent_leg, _ = rc.analyze_leg(
        ab_bundle, samples, phase="solvent", edge_id="e", ligand_a_name="A",
        ligand_b_name="B", temperature_kelvin=300.0, decorrelate=False,
    )
    edge = rc.combine_rbfe(complex_leg, solvent_leg)
    # 两腿是同一份样本 ⟹ ΔG 相同 ⟹ ΔΔG = complex − solvent = 0
    assert edge.ddg_bind == pytest.approx(0.0, abs=1e-12)


def test_artifacts_fingerprint_tracks_the_samples(ab_bundle):
    """换了样本就换指纹——否则旧分析结果会被错误复用（计划 §7）。"""
    first, _ = rc.analyze_leg(
        ab_bundle, synthetic_samples(ab_bundle, n_frames=5), phase="solvent",
        edge_id="e", ligand_a_name="A", ligand_b_name="B",
        temperature_kelvin=300.0, decorrelate=False,
    )
    second, _ = rc.analyze_leg(
        ab_bundle, synthetic_samples(ab_bundle, n_frames=7), phase="solvent",
        edge_id="e", ligand_a_name="A", ligand_b_name="B",
        temperature_kelvin=300.0, decorrelate=False,
    )
    assert first.artifacts_fingerprint != second.artifacts_fingerprint
    assert len(first.artifacts_fingerprint) == 64


def test_analyze_leg_rejects_unknown_phase(ab_bundle):
    with pytest.raises(ValueError, match="phase 必须是"):
        rc.analyze_leg(
            ab_bundle, synthetic_samples(ab_bundle, n_frames=3), phase="vacuum",
            edge_id="e", ligand_a_name="A", ligand_b_name="B", temperature_kelvin=300.0,
        )


# ---------------------------------------------------------------------------
# 独立窗口采样（free_energy_engine）
# ---------------------------------------------------------------------------


def _request(bundle, *, steps=200, save=100, seeds=True, pressure_bar=None, platform="Reference"):
    n_states = bundle.schedule.n_states
    positions = POSITIONS[: bundle.layout.n_particles] * unit.nanometer
    plan = {"integrator_seeds": [20260903 + i for i in range(n_states)]} if seeds else {}
    return fe.SamplingRequest(
        system=bundle.system, topology=None,
        states=[fe.ThermodynamicStateSpec(index=i, parameters=bundle.schedule.state(i))
                for i in range(n_states)],
        initial_positions=[positions] * n_states,
        initial_box_vectors=[BOX] * n_states,
        temperature_kelvin=300.0, pressure_bar=pressure_bar, timestep_fs=1.0,
        total_md_steps=steps, exchange_interval=save, save_interval=save,
        seed_plan=plan, platform_name=platform,
        output_dir="/tmp", stage_name="rbfe_test",
        caller_protocol_fingerprint=bundle.fingerprint(),
    )


def test_engine_never_invents_seeds(ab_bundle):
    """契约写的是「engine 不生成 seed，只转交并登记」——拿不到就报错，不自己随机。"""
    with pytest.raises(fe.SamplingContractError, match="不生成种子"):
        fe.run_independent_windows(_request(ab_bundle, seeds=False))


def test_seed_count_must_match_state_count(ab_bundle):
    request = _request(ab_bundle)
    broken = fe.SamplingRequest(
        **{**request.__dict__, "seed_plan": {"integrator_seeds": [1, 2]}}
    )
    with pytest.raises(fe.SamplingContractError, match="integrator_seeds"):
        fe.run_independent_windows(broken)


def test_npt_without_a_barostat_is_rejected(ab_bundle):
    """engine **不往调用方的 System 里加力**——恒压器属于体系定义。"""
    with pytest.raises(fe.SamplingContractError, match="没有 barostat"):
        fe.run_independent_windows(_request(ab_bundle, pressure_bar=1.0))


def test_independent_windows_produce_the_planned_frames(ab_bundle):
    samples = fe.run_independent_windows(
        _request(ab_bundle, steps=200, save=100), minimize=False
    )
    assert samples.n_states == ab_bundle.schedule.n_states
    assert samples.n_frames_by_state == (2,) * ab_bundle.schedule.n_states
    assert samples.provenance["exchange"] == "none__independent_windows"
    for state in range(samples.n_states):
        assert samples.state_parameters[state] == ab_bundle.schedule.state(state)


def test_independent_windows_are_reproducible(ab_bundle):
    """同样的种子必须给同样的轨迹——否则"通过"不可复现。"""
    first = fe.run_independent_windows(_request(ab_bundle, steps=100, save=100), minimize=False)
    second = fe.run_independent_windows(_request(ab_bundle, steps=100, save=100), minimize=False)
    for state in range(first.n_states):
        assert np.allclose(
            first.positions_by_state[state], second.positions_by_state[state], atol=0, rtol=0
        )


def test_sampler_output_feeds_the_analyzer_end_to_end(self_bundle):
    """贯通用例：真 MD 采样 → u_kn → MBAR。

    用 **A→A 自边**：它的答案与采样质量无关（恒为 0），所以这条用例可以只跑
    很少的步数还保持严格。跑 A→B 的话，几帧样本的 MBAR 结果没有可断言的真值。
    """
    samples = fe.run_independent_windows(
        _request(self_bundle, steps=200, save=100), minimize=False
    )
    result, diagnostics = rc.analyze_leg(
        self_bundle, samples, phase="solvent", edge_id="self",
        ligand_a_name="A", ligand_b_name="A", temperature_kelvin=300.0, decorrelate=False,
    )
    assert abs(result.delta_g) < 1e-6
    assert diagnostics["sampler_provenance"]["sampler"] == \
        "free_energy_engine.run_independent_windows"
