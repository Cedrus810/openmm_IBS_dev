"""WP-2：解析 mock 基势的端点、力和 IBS 能量账本测试。"""

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from outer_lambda_neural_basis import (
    ExistingOpenMMMLBasisEvaluation,
    HarmonicDistanceBasis,
    ExistingOrbMaceBasisAdapter,
    IBSEnergyLedger,
    IBSSamplerNeuralPathAdapter,
    NeuralPathConfigError,
    NeuralPathIntegrityError,
    NeuralPathFrameError,
    TorchForceDeploymentError,
    NeuralBasisTaskManifest,
    MaceDecompositionPythonComputation,
    OuterLambdaIBSBiasForce,
    OuterLambdaController,
    assess_mace_nvt_qualification,
    assess_mace_mts_matrix,
    assess_exp011_periodic_coverage,
    analyze_periodic_torsion_series,
    benchmark_existing_orb_mace_basis,
    build_exp010_protein_only_selection,
    build_exp011_periodic_umbrella_force,
    build_periodic_fourier_openmm_force,
    compare_wp5_arms,
    compare_wp5_replicates,
    compare_slow_variable_screens,
    count_discrete_transitions,
    discover_ligand_rotatable_torsions,
    evaluate_outer_lambda_force_group_states,
    fit_periodic_fourier_distillation,
    fit_exp011_reweighted_periodic_pmf,
    importance_effective_sample_size,
    integrated_autocorrelation_time,
    freeze_slow_variable_manifest,
    project_force_onto_torsion,
    reweight_exp011_umbrella_reports,
    torsion_coordinate_gradient_radians,
    prepare_existing_model_node_config,
    periodic_dihedral_degrees,
    qualify_wp4_basis,
    run_mace_decomposition_mts_arm,
    run_mace_decomposition_nvt_smoke,
    screen_ligand_hydration_coordination,
    screen_periodic_torsion_candidates,
    select_wp0_difficult_window,
    summarize_finite_series,
)


pytestmark = pytest.mark.cpu_only


def _controller(
    tmp_path,
    *,
    enabled=True,
    support_domain=None,
    periodic=False,
    backend="torchforce",
    coefficient=0.5,
):
    model = tmp_path / "mock.pt"
    indices = tmp_path / "indices.json"
    model.write_bytes(b"analytic mock placeholder")
    indices.write_text(json.dumps([0, 1]), encoding="utf-8")
    basis_config = {
        "name": "analytic_harmonic_mock",
        "backend": backend,
        "model_path": str(model),
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "energy_offset_kj_mol": 1.0,
        "atom_selection": "fixed_indices",
        "atom_indices_path": str(indices),
        "output_unit": "kJ_per_mol",
        "precision": "double",
        "periodic": periodic,
    }
    if backend == "existing_openmmml":
        basis_config["model_name"] = "mace-off24-medium"
    if support_domain is not None:
        basis_config["support_domain"] = support_domain
    config = {
        "neural_path": {
            "enabled": enabled,
            "protocol_version": 1,
            "stage": "vanishing",
            "baseline_potential": "softcore",
            "endpoint_tolerance": 1.0e-12,
            "envelope": {"type": "sin2", "parameters": {}},
            "coefficient_model": {
                "type": "constant",
                "coefficients": [coefficient],
                "max_abs_coefficient": 1.0,
            },
            "bases": [basis_config],
            "safety": {
                "max_abs_basis_energy_kj_mol": 50.0,
                "max_abs_path_energy_kj_mol": 20.0,
                "max_force_norm_kj_mol_nm": 500.0,
                "fail_on_support_domain_violation": True,
            },
        }
    }
    return OuterLambdaController.from_mapping(config)


def test_harmonic_mock_force_matches_finite_difference():
    basis = HarmonicDistanceBasis(
        atom_i=0,
        atom_j=1,
        force_constant_kj_mol_nm2=100.0,
        equilibrium_distance_nm=0.2,
    )
    positions = [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]
    evaluated = basis.evaluate(positions)

    step = 1.0e-6
    plus = copy.deepcopy(positions)
    minus = copy.deepcopy(positions)
    plus[0][0] += step
    minus[0][0] -= step
    numerical_force = -(
        basis.evaluate(plus).energy_kj_mol
        - basis.evaluate(minus).energy_kj_mol
    ) / (2.0 * step)

    assert evaluated.energy_kj_mol == pytest.approx(4.5)
    assert evaluated.forces_kj_mol_nm[0][0] == pytest.approx(
        numerical_force, rel=1.0e-9
    )
    assert evaluated.forces_kj_mol_nm[1][0] == pytest.approx(-numerical_force)
    assert tuple(
        evaluated.forces_kj_mol_nm[0][axis]
        + evaluated.forces_kj_mol_nm[1][axis]
        for axis in range(3)
    ) == pytest.approx((0.0, 0.0, 0.0))


def test_existing_mace_adapter_reuses_local_decomposition_and_maps_forces(
    monkeypatch,
):
    import numpy as np

    class FakeQuantity:
        def __init__(self, value):
            self.value = value

        def value_in_unit(self, _unit):
            return self.value

    class FakeState:
        def __init__(self, energy, forces):
            self.energy = energy
            self.forces = forces

        def getPotentialEnergy(self):
            return FakeQuantity(self.energy)

        def getForces(self, asNumpy=False):
            assert asNumpy is True
            return FakeQuantity(np.asarray(self.forces, dtype=float))

    class FakeContext:
        def __init__(self, energy, forces):
            self.state = FakeState(energy, forces)
            self.positions = None

        def setPositions(self, positions):
            self.positions = positions

        def getState(self, **kwargs):
            assert kwargs == {"getEnergy": True, "getForces": True}
            return self.state

    class FakePipeline:
        def __init__(self, model_name, device):
            self.device = device
            self.label_mode = (
                "orbv3_interaction"
                if "orb" in model_name
                else "mace_decomposition"
            )
            self.bundle = {
                "comb_idx": np.asarray([0, 2, 3]),
                "lig_idx": np.asarray([0]),
                "env_idx": np.asarray([2, 3]),
                "contexts": {
                    "cplx": {
                        "context": FakeContext(
                            10.0,
                            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                        )
                    },
                    "lig": {
                        "context": FakeContext(2.0, [[0.5, 0.0, 0.0]])
                    },
                    "env": {
                        "context": FakeContext(
                            3.0,
                            [[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]],
                        )
                    },
                },
            }

        def _preflight_orb_backend(self, *args):
            return None

        def _get_orb_decomposition_bundle(self, *args):
            return self.bundle

        def _clear_orb_context_cache(self):
            self.bundle = {}

    monkeypatch.setitem(
        sys.modules,
        "dexp_退役",
        SimpleNamespace(Orbv3DEXPFittingPipeline=FakePipeline),
    )
    with ExistingOrbMaceBasisAdapter(
        model_name="mace-off24-medium", device="cpu"
    ) as adapter:
        result = adapter.evaluate(
            [[0.0, 0.0, 0.0]] * 4,
            ligand_indices=[0],
            environment_indices=[2, 3],
            atomic_numbers=[6, 1, 7, 8],
        )

    assert result.label_mode == "mace_decomposition"
    assert result.energy_kj_mol == pytest.approx(5.0)
    assert result.forces_kj_mol_nm[0] == pytest.approx((0.5, 0.0, 0.0))
    assert result.forces_kj_mol_nm[1] == pytest.approx((0.0, 0.0, 0.0))
    assert result.forces_kj_mol_nm[2] == pytest.approx((1.75, 0.0, 0.0))
    assert result.forces_kj_mol_nm[3] == pytest.approx((2.25, 0.0, 0.0))


def test_existing_orb_adapter_requires_conservative_model(monkeypatch):
    class FakePipeline:
        def __init__(self, model_name, device):
            self.device = device
            self.label_mode = "orbv3_interaction"
            self._orb_ctx_cache = {}

        def _clear_orb_context_cache(self):
            self._orb_ctx_cache = {}

    monkeypatch.setitem(
        sys.modules,
        "dexp_退役",
        SimpleNamespace(Orbv3DEXPFittingPipeline=FakePipeline),
    )
    with pytest.raises(NeuralPathConfigError, match="conservative ORB"):
        ExistingOrbMaceBasisAdapter(
            model_name="orb-v3-direct-omol", device="cpu"
        )

    with ExistingOrbMaceBasisAdapter(
        model_name="orb-v3-conservative-omol", device="cpu"
    ) as adapter:
        assert adapter.label_mode == "orbv3_decomposition"


def test_mace_python_computation_has_exact_endpoint_fast_path(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"endpoint must not load this")
    computation = MaceDecompositionPythonComputation(
        model_path=str(model),
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        atomic_numbers=[6, 7],
        ligand_indices=[0],
        environment_indices=[1],
        coefficient=0.1,
        energy_offset_kj_mol=5.0,
        lambda_parameter_name="outer_lambda",
        device="cpu",
        precision="single",
        max_abs_basis_energy_kj_mol=100.0,
        max_abs_path_energy_kj_mol=20.0,
        max_force_norm_kj_mol_nm=500.0,
    )

    class EndpointState:
        def getParameters(self):
            return {"outer_lambda": 0.0}

        def getPositions(self, **kwargs):
            raise AssertionError("端点不应读取坐标或运行 MACE")

    energy, forces = computation(EndpointState())
    assert energy == 0.0
    assert forces.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_mace_python_computation_scales_decomposed_energy_and_force(
    tmp_path, monkeypatch
):
    import numpy as np

    model = tmp_path / "model.pt"
    model.write_bytes(b"mocked MACE")
    computation = MaceDecompositionPythonComputation(
        model_path=str(model),
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        atomic_numbers=[6, 7],
        ligand_indices=[0],
        environment_indices=[1],
        coefficient=0.1,
        energy_offset_kj_mol=5.0,
        lambda_parameter_name="outer_lambda",
        device="cpu",
        precision="single",
        max_abs_basis_energy_kj_mol=100.0,
        max_abs_path_energy_kj_mol=20.0,
        max_force_norm_kj_mol_nm=500.0,
    )
    monkeypatch.setattr(
        computation,
        "_evaluate_decomposition",
        lambda positions, box: (
            15.0,
            np.asarray([[-100.0, 0.0, 0.0], [100.0, 0.0, 0.0]]),
        ),
    )

    class FakeQuantity:
        def __init__(self, value):
            self.value = np.asarray(value)

        def value_in_unit(self, _unit):
            return self.value

    class MidpointState:
        def getParameters(self):
            return {"outer_lambda": 0.5}

        def getPositions(self, asNumpy=False):
            assert asNumpy is True
            return FakeQuantity([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])

        def getPeriodicBoxVectors(self, asNumpy=False):
            assert asNumpy is True
            return FakeQuantity(np.identity(3))

    energy, forces = computation(MidpointState())
    assert energy == pytest.approx(1.0)
    assert forces[0] == pytest.approx([-10.0, 0.0, 0.0])
    assert forces[1] == pytest.approx([10.0, 0.0, 0.0])


def test_mace_python_computation_minimum_images_unwrapped_coordinates():
    import numpy as np

    positions = np.asarray([[0.1, 0.0, 0.0], [9.0, 0.0, 0.0]])
    box = np.diag([9.1, 9.1, 9.1])
    imaged = MaceDecompositionPythonComputation._minimum_image_selected(
        positions, [0, 1], box
    )
    assert np.linalg.norm(imaged[1] - imaged[0]) == pytest.approx(0.2)


def test_cross_state_force_group_query_restores_context_parameter():
    class FakeEnergy:
        def __init__(self, value):
            self.value = value

        def value_in_unit(self, _unit):
            return self.value

    class FakeState:
        def __init__(self, value):
            self.value = value

        def getPotentialEnergy(self):
            return FakeEnergy(self.value)

    class FakeContext:
        def __init__(self):
            self.value = 0.3
            self.queries = []

        def getParameter(self, name):
            assert name == "outer_lambda"
            return self.value

        def setParameter(self, name, value):
            assert name == "outer_lambda"
            self.value = value

        def getState(self, **kwargs):
            self.queries.append(kwargs)
            return FakeState(10.0 * self.value)

    context = FakeContext()
    energies = evaluate_outer_lambda_force_group_states(
        context,
        [0.0, 0.5, 1.0],
        force_group=29,
    )

    assert energies == pytest.approx((0.0, 5.0, 10.0))
    assert context.value == pytest.approx(0.3)
    assert context.queries == [
        {"getEnergy": True, "groups": 1 << 29},
        {"getEnergy": True, "groups": 1 << 29},
        {"getEnergy": True, "groups": 1 << 29},
    ]


def test_mace_nvt_runner_uses_system_copy_and_endpoint_fast_path(tmp_path):
    import openmm
    from openmm import unit

    controller = _controller(
        tmp_path,
        backend="existing_openmmml",
    )
    system = openmm.System()
    system.addParticle(12.0 * unit.dalton)
    system.addParticle(14.0 * unit.dalton)
    bond = openmm.HarmonicBondForce()
    bond.addBond(
        0,
        1,
        0.2 * unit.nanometer,
        100.0 * unit.kilojoules_per_mole / unit.nanometer**2,
    )
    system.addForce(bond)
    original_force_count = system.getNumForces()

    report = run_mace_decomposition_nvt_smoke(
        controller,
        system,
        atomic_numbers=[6, 7],
        ligand_indices=[0],
        environment_indices=[1],
        positions_nm=[[0.0, 0.0, 0.0], [0.21, 0.0, 0.0]],
        box_vectors_nm=[
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        lambda_value=0.0,
        n_steps=1,
        report_interval=1,
        device="cpu",
        platform_name="Reference",
    )

    assert report["passed"] is True
    assert report["support_domain_configured"] is False
    assert report["path_energy_kj_mol"]["max_abs"] == 0.0
    assert report["max_energy_closure_error_kj_mol"] == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert system.getNumForces() == original_force_count


def test_outer_lambda_ibs_bias_shares_one_basis_across_states(tmp_path):
    import openmm
    from openmm import unit

    controller = _controller(tmp_path)
    lambdas = [0.0, 0.5, 1.0]
    basis = openmm.HarmonicBondForce()
    basis.addBond(
        0,
        1,
        0.2 * unit.nanometer,
        100.0 * unit.kilojoules_per_mole / unit.nanometer**2,
    )
    wrapper = OuterLambdaIBSBiasForce(
        controller,
        lambdas,
        300.0,
        [basis],
        prefix="test_neural",
    )
    original_state_energies = [1.0, 2.0, 4.0]
    for state_index, energy in enumerate(original_state_energies):
        interaction = openmm.CustomExternalForce(str(energy))
        interaction.addParticle(0, [])
        restraint = openmm.CustomExternalForce("0")
        restraint.addParticle(0, [])
        wrapper.addCollectiveVariable(
            f"cv_{state_index}_int", interaction
        )
        wrapper.addCollectiveVariable(
            f"cv_{state_index}_rest", restraint
        )

    system = openmm.System()
    system.addParticle(12.0)
    system.addParticle(12.0)
    system.addForce(wrapper.get_force())
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(
        system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    try:
        energy = context.getState(
            getEnergy=True
        ).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        basis_energies = wrapper.get_centered_basis_energies_kj_mol(context)
    finally:
        del context, integrator

    assert basis_energies == pytest.approx((0.5,))
    path = tuple(
        math.fsum(
            coefficient * centered_basis
            for coefficient, centered_basis in zip(
                row, basis_energies, strict=True
            )
        )
        for row in controller.coefficient_matrix(lambdas)
    )
    state_energies = [
        original + addition
        for original, addition in zip(
            original_state_energies, path, strict=True
        )
    ]
    kt = (
        unit.MOLAR_GAS_CONSTANT_R * 300.0 * unit.kelvin
    ).value_in_unit(unit.kilojoules_per_mole)
    expected = -kt * math.log(
        sum(math.exp(-value / kt) for value in state_energies)
    )
    assert energy == pytest.approx(expected, abs=1.0e-8)
    assert len(wrapper._basis_cv_indices) == 1
    assert len(wrapper._int_cv_force_xmls) == len(lambdas)


def test_neural_sampler_adapter_keeps_target_and_sampling_bias_separate(
    tmp_path,
):
    import numpy as np
    import openmm
    from openmm import unit

    controller = _controller(tmp_path)
    lambdas = [0.0, 0.5, 1.0]
    basis = openmm.HarmonicBondForce()
    basis.addBond(
        0,
        1,
        0.2 * unit.nanometer,
        100.0 * unit.kilojoules_per_mole / unit.nanometer**2,
    )
    wrapper = OuterLambdaIBSBiasForce(
        controller, lambdas, 300.0, [basis], prefix="adapter_neural"
    )
    original = np.asarray([1.0, 2.0, 4.0])
    lrc = np.asarray([0.1, 0.2, 0.3])
    for state_index, energy in enumerate(original):
        interaction = openmm.CustomExternalForce(str(float(energy)))
        interaction.addParticle(0, [])
        restraint = openmm.CustomExternalForce("0")
        restraint.addParticle(0, [])
        wrapper.addCollectiveVariable(
            f"cv_{state_index}_int", interaction
        )
        wrapper.addCollectiveVariable(
            f"cv_{state_index}_rest", restraint
        )
    base = openmm.CustomExternalForce("5")
    base.addParticle(0, [])
    base.setForceGroup(0)
    system = openmm.System()
    system.addParticle(12.0)
    system.addParticle(12.0)
    system.addForce(base)
    system.addForce(wrapper.get_force())
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(
        system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])

    class FakeSampler:
        n_states = 3

        def __init__(self):
            self.context = context
            self.energy_buffer = []
            self.energy_history = []
            self.bias_history = []
            self.base_energy_history = []
            self.e_offset = 0.0
            self.query_results = []

        def _collect_interaction_energies(self):
            return original.copy()

        def _lj_tail_correction_kj_mol(self):
            return lrc.copy()

        def _record_energy_query_result(self, success, reason=None):
            self.query_results.append((success, reason))

    sampler = FakeSampler()
    adapter = IBSSamplerNeuralPathAdapter(
        sampler, controller, lambdas, wrapper
    )
    try:
        returned = adapter.collect_energies()
    finally:
        del context, integrator

    assert returned == pytest.approx([0.0, 1.25, 3.0])
    assert sampler.energy_buffer[0] == pytest.approx(
        [0.0, 1.25, 3.0]
    )
    assert sampler.energy_history[0] == pytest.approx(
        [1.1, 2.45, 4.3]
    )
    assert sampler.base_energy_history == pytest.approx([5.0])
    assert len(sampler.bias_history) == 1
    assert sampler.bias_history[0] != pytest.approx(0.25)
    assert adapter.neural_path_energy_history[0] == pytest.approx(
        [0.0, 0.25, 0.0]
    )
    assert adapter.basis_energy_history[0] == pytest.approx((1.5,))
    assert sampler.query_results == [(True, None)]


def test_mace_nvt_qualification_accepts_frozen_passing_run():
    run_report = {
        "passed": True,
        "n_steps": 1000,
        "support_domain_configured": True,
        "support_domain_violation_count": 0,
        "max_path_force_kj_mol_nm": {"max": 186.4},
        "max_energy_closure_error_kj_mol": 0.006,
        "integration_seconds_per_step": 0.103,
    }

    report = assess_mace_nvt_qualification(run_report)

    assert report["qualified"] is True
    assert report["failed_checks"] == []
    assert all(report["checks"].values())


def test_mace_nvt_qualification_rejects_calibration_only_run():
    run_report = {
        "passed": True,
        "n_steps": 100,
        "support_domain_configured": False,
        "support_domain_violation_count": 0,
        "max_path_force_kj_mol_nm": {"max": 186.4},
        "max_energy_closure_error_kj_mol": 0.006,
        "integration_seconds_per_step": 0.103,
    }

    report = assess_mace_nvt_qualification(run_report)

    assert report["qualified"] is False
    assert report["failed_checks"] == ["minimum_steps", "support_domain"]


def test_wp0_selects_lowest_final_ess_window_and_torsion_hysteresis():
    final_results = {
        "stage_diagnostics": {
            "stage2": {
                "window_overlap_diagnostics": [
                    {
                        "window_index": 0,
                        "window_range": [0, 5],
                        "lambdas_vdw": [1.0, 0.8],
                        "min_ess_ratio": 0.39,
                        "absolute_ess": 38.0,
                        "statistical_inefficiency": 5.2,
                        "endpoint_diff_uncertainty_kJ_mol": 0.9,
                    },
                    {
                        "window_index": 1,
                        "window_range": [4, 8],
                        "lambdas_vdw": [0.8, 0.6],
                        "min_ess_ratio": 0.58,
                        "absolute_ess": 200.0,
                        "statistical_inefficiency": 1.4,
                        "endpoint_diff_uncertainty_kJ_mol": 0.7,
                    },
                ]
            }
        }
    }

    selected = select_wp0_difficult_window(final_results)
    torsion = analyze_periodic_torsion_series(
        [175.0, -178.0, 140.0, -65.0, -62.0, -120.0, 170.0]
    )

    assert selected["selected_window"]["window_index"] == 0
    assert torsion["basin_occupancy"]["gauche_minus"] == pytest.approx(
        2.0 / 7.0
    )
    assert torsion["core_transition_count"] == 2


def test_slow_variable_screen_discovers_graph_torsion_and_ranks_periodically():
    import mdtraj as md

    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("LIG", chain, resSeq=7)
    atoms = [
        topology.add_atom(f"C{index}", md.element.carbon, residue)
        for index in range(5)
    ]
    for left, right in zip(atoms[:-1], atoms[1:], strict=True):
        topology.add_bond(left, right)

    candidates = discover_ligand_rotatable_torsions(
        topology, list(range(5))
    )
    assert candidates
    assert all(
        candidate["candidate_type"] == "ligand_rotatable_torsion"
        for candidate in candidates
    )
    selected = candidates[0]
    assert all("residue_name" in atom for atom in selected["atoms"])

    frames = []
    for angle in range(0, 360, 18):
        radians = math.radians(angle)
        frames.append(
            [
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.4, 0.1 * math.cos(radians), 0.1 * math.sin(radians)],
                [0.6, 0.1, 0.1],
            ]
        )
    report = screen_periodic_torsion_candidates(
        frames, None, [selected]
    )

    assert report["selection_status"] == "candidate_ranking_only"
    assert report["n_frames"] == 20
    assert report["periodic_torsion_candidates"][0][
        "rank_within_periodic_torsions"
    ] == 1


def test_hydration_screen_uses_water_oxygen_and_periodic_minimum_image():
    import mdtraj as md

    topology = md.Topology()
    chain = topology.add_chain()
    ligand = topology.add_residue("LIG", chain, resSeq=1)
    ligand_atom = topology.add_atom("C1", md.element.carbon, ligand)
    water = topology.add_residue("HOH", chain, resSeq=2)
    water_oxygen = topology.add_atom("O", md.element.oxygen, water)
    topology.add_atom("H1", md.element.hydrogen, water)
    topology.add_atom("H2", md.element.hydrogen, water)
    frames = [
        [
            [0.05, 0.0, 0.0],
            [0.95, 0.0, 0.0],
            [0.95, 0.01, 0.0],
            [0.95, -0.01, 0.0],
        ]
        for _ in range(10)
    ]
    boxes = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 10

    report = screen_ligand_hydration_coordination(
        frames, boxes, topology, [ligand_atom.index]
    )

    assert report["water_oxygen_count"] == 1
    assert report["definition"]["periodic_minimum_image"] is True
    assert report["mean"] > 0.99
    assert report["effective_uncorrelated_samples"] == pytest.approx(10.0)


def test_exp010_selection_removes_exchange_water_but_keeps_protein():
    import mdtraj as md

    topology = md.Topology()
    chain = topology.add_chain()
    ligand = topology.add_residue("MOL", chain)
    ligand_atom = topology.add_atom("C", md.element.carbon, ligand)
    protein = topology.add_residue("VAL", chain)
    protein_atom = topology.add_atom("CA", md.element.carbon, protein)
    water = topology.add_residue("HOH", chain)
    water_atoms = [
        topology.add_atom("O", md.element.oxygen, water),
        topology.add_atom("H1", md.element.hydrogen, water),
        topology.add_atom("H2", md.element.hydrogen, water),
    ]

    selected = build_exp010_protein_only_selection(
        {
            "ligand_indices": [ligand_atom.index],
            "env_indices": [
                protein_atom.index,
                *[atom.index for atom in water_atoms],
            ],
        },
        topology,
    )

    assert selected["env_indices"] == [protein_atom.index]
    policy = selected["outer_lambda_exp010_selection_policy"]
    assert policy["removed_water_atom_count"] == 3
    assert policy["removed_water_residue_count"] == 1
    assert len(selected["selection_sha256"]) == 64


def test_slow_variable_replicate_comparison_requires_three_before_freeze():
    def report(transitions, rank=1):
        return {
            "report_type": "outer_lambda_slow_variable_screen",
            "periodic_torsion_candidates": [
                {
                    "stable_id": "sidechain_chi1:chain0:VAL:251",
                    "candidate_type": "pocket_sidechain_chi1",
                    "atom_indices": [0, 1, 2, 3],
                    "atoms": [
                        {"index": 0},
                        {"index": 1},
                        {"index": 2},
                        {"index": 3},
                    ],
                    "rank_within_periodic_torsions": rank,
                    "periodic_statistical_inefficiency": 20.0,
                    "torsion": {
                        "core_transition_count": transitions,
                        "circular_std_degrees": 35.0,
                    },
                }
            ],
        }

    two = compare_slow_variable_screens([report(1), report(2)])
    three = compare_slow_variable_screens(
        [report(1), report(2), report(1)]
    )

    assert two["production_cv_may_be_frozen"] is False
    assert two["selection_status"] == "more_independent_sampling_required"
    assert three["production_cv_may_be_frozen"] is True
    assert three["qualified_periodic_stable_ids"] == [
        "sidechain_chi1:chain0:VAL:251"
    ]

    manifest = freeze_slow_variable_manifest(
        three,
        {"window_index": 0, "window_range": [0, 5]},
    )
    assert manifest["production_approval"] is False
    assert manifest["target_experiment"] == "EXP-010"
    assert manifest["primary_slow_variable"]["atom_indices"] == [0, 1, 2, 3]
    assert len(manifest["manifest_sha256"]) == 64

    exp011_manifest = freeze_slow_variable_manifest(
        three,
        {"window_index": 0, "window_range": [0, 5]},
        experiment_id="exp011",
    )
    assert exp011_manifest["status"] == "frozen_for_exp011_complete_mm_pmf"
    assert exp011_manifest["target_experiment"] == "EXP-011"


def test_torsion_force_projection_recovers_generalized_force():
    positions = [
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.4, 0.05, 0.08],
    ]
    gradient = torsion_coordinate_gradient_radians(
        positions, [0, 1, 2, 3]
    )
    expected = 7.5
    forces = [
        [expected * component for component in vector]
        for vector in gradient
    ]

    projection = project_force_onto_torsion(forces, gradient)

    assert projection[
        "generalized_force_kj_mol_per_radian"
    ] == pytest.approx(expected)
    assert projection[
        "cartesian_force_projection_fraction"
    ] == pytest.approx(1.0)


def test_periodic_fourier_distillation_uses_whole_run_validation():
    samples = []
    for run in range(3):
        for index in range(24):
            angle = -math.pi + 2.0 * math.pi * (
                index + run / 3.0
            ) / 24.0
            energy = 2.0 + 3.0 * math.cos(angle) - 2.0 * math.sin(
                2.0 * angle
            )
            generalized_force = 3.0 * math.sin(angle) + 4.0 * math.cos(
                2.0 * angle
            )
            samples.append(
                {
                    "run_id": f"run{run}",
                    "primary_angle_radians": angle,
                    "teacher_centered_energy_kj_mol": energy,
                    "teacher_primary_generalized_force_kj_mol_per_radian": (
                        generalized_force
                    ),
                }
            )
    dataset = {
        "report_type": "outer_lambda_exp010_teacher_dataset",
        "primary_slow_variable": {"atom_indices": [0, 1, 2, 3]},
        "secondary_slow_variable": None,
        "samples": samples,
    }

    report = fit_periodic_fourier_distillation(
        dataset, dimensions=1, order=2, ridge=0.0
    )

    assert len(report["folds"]) == 3
    assert report["leave_one_run_out_energy"]["rmse"] < 1.0e-10
    assert report["leave_one_run_out_primary_generalized_force"][
        "rmse"
    ] < 1.0e-9
    assert report["model"]["support_policy"] == (
        "full_periodic_domain_no_extrapolation"
    )


def test_exp011_coverage_fails_closed_on_empty_periodic_bins():
    report = assess_exp011_periodic_coverage(
        {
            "run1": [-170.0] * 500,
            "run2": [-60.0] * 500,
            "run3": [60.0] * 500,
        }
    )

    assert report["qualified_for_pmf"] is False
    assert report["decision"] == "collect_restrained_or_enhanced_sampling"
    assert report["empty_pooled_bin_indices"]
    assert report["gates"]["minimum_effective_samples_per_pooled_bin"] is False


def test_exp011_reweighted_pmf_uses_whole_run_holdout_and_exports_bias():
    samples = []
    for run in range(3):
        centers = []
        remaining = []
        for bin_index in range(24):
            center = -172.5 + 15.0 * bin_index
            count = round(80 + 20 * math.cos(math.radians(center)))
            centers.append(center)
            remaining.append(count)
        ordered = []
        while any(remaining):
            for offset in range(24):
                bin_index = (offset * 7 + run) % 24
                if remaining[bin_index]:
                    ordered.append(centers[bin_index])
                    remaining[bin_index] -= 1
        for angle in ordered:
            samples.append(
                {
                    "run_id": f"run{run + 1}",
                    "angle_degrees": angle,
                    "log_target_weight": 0.0,
                }
            )
    protocol = {
        "coverage": {
            "bins": 24,
            "minimum_runs": 3,
            "minimum_frames_per_run": 500,
            "minimum_effective_samples_per_run": 25.0,
            "minimum_occupied_fraction_per_run": 0.5,
            "minimum_effective_samples_per_pooled_bin": 2.0,
            "minimum_runs_per_bin": 2,
            "minimum_raw_samples_per_run_bin": 3,
            "minimum_pairwise_bhattacharyya": 0.5,
            "minimum_effective_samples_per_basin": 5.0,
        },
        "pmf_fit": {
            "fourier_orders": [2, 4, 6],
            "ridge": 1.0e-6,
            "bias_scale": 0.5,
        },
        "acceptance": {
            "maximum_fold_pmf_rmse_kj_mol": 2.5,
            "minimum_fold_pmf_correlation": 0.8,
            "maximum_fold_barrier_error_kj_mol": 3.0,
            "maximum_bias_peak_to_peak_kj_mol": 12.0,
        },
    }
    dataset = {
        "report_type": "outer_lambda_exp011_target_samples",
        "target_hamiltonian_id": "synthetic_complete_mm_target",
        "temperature_kelvin": 300.0,
        "torsion_atom_indices": [0, 1, 2, 3],
        "samples": samples,
    }

    report = fit_exp011_reweighted_periodic_pmf(dataset, protocol)

    assert report["coverage"]["qualified_for_pmf"] is True
    assert report["qualified"] is True
    assert report["selected_order"] in {2, 4, 6}
    assert report["selected_model"]["bias_definition"].startswith("V_bias")
    assert all(len(candidate["folds"]) == 3 for candidate in report["candidates"])


def test_exp011_umbrella_reweight_exports_explicit_target_weights():
    reports = []
    for run in range(3):
        for center in (-135.0, -45.0, 45.0, 135.0):
            diagnostics = []
            for offset in range(-60, 61, 10):
                angle = ((center + offset + run * 2.0 + 180.0) % 360.0) - 180.0
                delta = math.atan2(
                    math.sin(math.radians(angle - center)),
                    math.cos(math.radians(angle - center)),
                )
                diagnostics.append(
                    {
                        "angle_degrees": angle,
                        "umbrella_energy_kj_mol": 0.5 * 5.0 * delta * delta,
                    }
                )
            reports.append(
                {
                    "report_type": "outer_lambda_exp011_umbrella_window",
                    "temperature_kelvin": 300.0,
                    "umbrella": {
                        "run_id": f"run{run + 1}",
                        "torsion_atom_indices": [0, 1, 2, 3],
                        "center_degrees": center,
                        "force_constant_kj_mol_radian2": 5.0,
                    },
                    "diagnostics": diagnostics,
                }
            )

    result = reweight_exp011_umbrella_reports(
        reports,
        target_hamiltonian_id="synthetic_complete_mm",
        minimum_neighbor_overlap=1.0e-6,
    )

    assert result["qualified_for_pmf_input"] is True
    assert result["mutual_overlap_connected"] is True
    assert result["neighbor_overlap_qualified"] is True
    assert len(result["neighbor_interfaces"]) == 4
    assert all(item["qualified"] for item in result["neighbor_interfaces"])
    assert len(result["state_centers_degrees"]) == 4
    assert result["dataset"]["sampling_method"] == "umbrella_mbar"
    assert result["dataset"]["decorrelated_by_source_window"] is True
    assert all(
        math.isfinite(sample["log_target_weight"])
        for sample in result["dataset"]["samples"]
    )


def test_exp011_umbrella_force_uses_shortest_periodic_angle_difference():
    import openmm
    from openmm import unit

    positions = [
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.4, 0.05, 0.08],
    ]
    angle = periodic_dihedral_degrees(positions, [0, 1, 2, 3])
    force = build_exp011_periodic_umbrella_force(
        [0, 1, 2, 3],
        center_degrees=angle + 359.0,
        force_constant_kj_mol_radian2=100.0,
    )
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0)
    system.addForce(force)
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(
        [openmm.Vec3(*value) for value in positions] * unit.nanometer
    )
    energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )

    assert energy == pytest.approx(
        0.5 * 100.0 * math.radians(1.0) ** 2, rel=1.0e-8
    )


def test_periodic_fourier_model_builds_serializable_openmm_force():
    import openmm
    from openmm import unit

    model = {
        "model_type": "periodic_fourier_cheap_cv",
        "dimensions": 1,
        "intercept_kj_mol": 2.0,
        "primary_atom_indices": [0, 1, 2, 3],
        "terms": [
            {
                "wavevector": [1],
                "cos_coefficient_kj_mol": 3.0,
                "sin_coefficient_kj_mol": -1.0,
            }
        ],
    }
    force = build_periodic_fourier_openmm_force(model, force_group=3)
    xml = openmm.XmlSerializer.serialize(force)
    rebuilt = openmm.XmlSerializer.deserialize(xml)
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0)
    system.addForce(rebuilt)
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(
        system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    positions = [
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.4, 0.05, 0.08],
    ]
    context.setPositions(
        [openmm.Vec3(*value) for value in positions] * unit.nanometer
    )
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    angle = math.radians(
        periodic_dihedral_degrees(positions, [0, 1, 2, 3])
    )

    assert energy == pytest.approx(
        2.0 + 3.0 * math.cos(angle) - math.sin(angle)
    )
    assert all(math.isfinite(float(value)) for row in forces for value in row)
    assert rebuilt.getForceGroup() == 3


def test_two_dimensional_fourier_force_allows_overlapping_torsions():
    import openmm
    from openmm import unit

    model = {
        "model_type": "periodic_fourier_cheap_cv",
        "dimensions": 2,
        "intercept_kj_mol": 0.0,
        "primary_atom_indices": [0, 1, 2, 3],
        "secondary_atom_indices": [2, 3, 4, 5],
        "terms": [
            {
                "wavevector": [1, -1],
                "cos_coefficient_kj_mol": 1.0,
                "sin_coefficient_kj_mol": 0.0,
            }
        ],
    }
    force = build_periodic_fourier_openmm_force(model)
    system = openmm.System()
    for _ in range(6):
        system.addParticle(12.0)
    system.addForce(force)
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(
        system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(
        [
            openmm.Vec3(0.0, 0.1, 0.0),
            openmm.Vec3(0.0, 0.0, 0.0),
            openmm.Vec3(0.2, 0.0, 0.0),
            openmm.Vec3(0.4, 0.1, 0.1),
            openmm.Vec3(0.6, 0.0, 0.1),
            openmm.Vec3(0.8, 0.1, 0.0),
        ]
        * unit.nanometer
    )
    state = context.getState(getEnergy=True, getForces=True)

    assert math.isfinite(
        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    )


def test_mace_mts_arm_uses_outer_step_without_loading_endpoint_model(tmp_path):
    import openmm
    from openmm import unit

    controller = _controller(
        tmp_path,
        backend="existing_openmmml",
        coefficient=0.09,
        support_domain={"max_pair_distance_nm": 1.0},
    )
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0 * unit.dalton)
    bonds = openmm.HarmonicBondForce()
    for left, right in ((0, 1), (1, 2), (2, 3)):
        bonds.addBond(
            left,
            right,
            0.2 * unit.nanometer,
            100.0
            * unit.kilojoules_per_mole
            / unit.nanometer**2,
        )
    system.addForce(bonds)

    report = run_mace_decomposition_mts_arm(
        controller,
        system,
        atomic_numbers=[6, 7, 6, 6],
        ligand_indices=[0],
        environment_indices=[1],
        positions_nm=[
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.17, 0.0],
            [0.4, 0.2, 0.16],
        ],
        box_vectors_nm=[
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        torsion_atom_indices=[0, 1, 2, 3],
        mts_ratio=2,
        lambda_value=0.0,
        n_inner_steps=4,
        report_interval_inner_steps=4,
        device="cpu",
        platform_name="Reference",
    )

    assert report["passed"] is True
    assert report["inner_timestep_fs"] == pytest.approx(0.5)
    assert report["outer_timestep_fs"] == pytest.approx(1.0)
    assert report["n_outer_steps"] == 2
    assert report["path_energy_kj_mol"]["max_abs"] == 0.0
    assert report["max_path_force_kj_mol_nm"]["max_abs"] == 0.0


def test_mace_mts_arm_rejects_unsupported_initial_frame_before_model_load(
    tmp_path,
):
    import openmm
    from openmm import unit

    controller = _controller(
        tmp_path,
        backend="existing_openmmml",
        coefficient=0.09,
        support_domain={"max_pair_distance_nm": 0.1},
    )
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0 * unit.dalton)

    with pytest.raises(
        NeuralPathFrameError,
        match="初始坐标不在冻结支持域，积分未启动",
    ):
        run_mace_decomposition_mts_arm(
            controller,
            system,
            atomic_numbers=[6, 7, 6, 6],
            ligand_indices=[0],
            environment_indices=[1],
            positions_nm=[
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.17, 0.0],
                [0.4, 0.2, 0.16],
            ],
            box_vectors_nm=[
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            torsion_atom_indices=[0, 1, 2, 3],
            mts_ratio=1,
            lambda_value=0.0,
            n_inner_steps=1,
            report_interval_inner_steps=1,
            device="cpu",
            platform_name="Reference",
        )


def test_mace_mts_arm_rejects_pythonforce_cuda_backend_before_context(
    tmp_path,
):
    import openmm
    from openmm import unit

    controller = _controller(
        tmp_path,
        backend="existing_openmmml",
        coefficient=0.09,
        support_domain={"max_pair_distance_nm": 1.0},
    )
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12.0 * unit.dalton)

    with pytest.raises(
        TorchForceDeploymentError,
        match="start_exp010_cheap_cv_due_to_backend",
    ):
        run_mace_decomposition_mts_arm(
            controller,
            system,
            atomic_numbers=[6, 7, 6, 6],
            ligand_indices=[0],
            environment_indices=[1],
            positions_nm=[
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.17, 0.0],
                [0.4, 0.2, 0.16],
            ],
            box_vectors_nm=[
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            torsion_atom_indices=[0, 1, 2, 3],
            mts_ratio=1,
            lambda_value=0.0,
            n_inner_steps=1,
            report_interval_inner_steps=1,
            device="cuda",
            platform_name="CUDA",
        )


def _fake_mts_arm(ratio, *, ns_per_day, force=200.0, torsion_shift=0):
    probabilities = [0.5, 0.5] + [0.0] * 22
    if torsion_shift:
        probabilities = [0.0, 0.0, 0.5, 0.5] + [0.0] * 20
    return {
        "mts_ratio": ratio,
        "passed": True,
        "support_domain_configured": True,
        "support_domain_violation_count": 0,
        "max_path_force_kj_mol_nm": {"max": force},
        "max_energy_closure_error_kj_mol": 0.01,
        "temperature_kelvin": {"mean": 300.0, "std": 5.0},
        "total_energy_kj_mol": {"mean": -1000.0, "std": 20.0},
        "selected_aligned_rmsd_nm": {"mean": 0.1, "std": 0.01},
        "slow_torsion": {
            "histogram": {"probabilities": probabilities}
        },
        "ns_per_day": ns_per_day,
    }


def test_mts_matrix_routes_stable_fast_n4_to_n8():
    report = assess_mace_mts_matrix(
        [
            _fake_mts_arm(1, ns_per_day=0.44),
            _fake_mts_arm(2, ns_per_day=0.8),
            _fake_mts_arm(4, ns_per_day=1.2),
        ],
        minimum_n4_ns_per_day=1.0,
    )

    assert report["qualified"] is True
    assert report["decision"] == "qualified_to_test_n8_before_wp5"


def test_mts_matrix_routes_distribution_bias_to_teacher_only():
    report = assess_mace_mts_matrix(
        [
            _fake_mts_arm(1, ns_per_day=0.44),
            _fake_mts_arm(2, ns_per_day=0.8),
            _fake_mts_arm(4, ns_per_day=1.2, torsion_shift=1),
        ],
        minimum_n4_ns_per_day=1.0,
    )

    assert report["qualified"] is False
    assert report["physics_passed"] is False
    assert report["decision"] == "direct_mace_teacher_only_due_to_mts_bias"


def test_prepare_existing_model_can_freeze_support_domain(tmp_path):
    model_path = tmp_path / "mace.model"
    model_path.write_bytes(b"frozen model identity")
    selection_meta = tmp_path / "selection.json"
    selection_meta.write_text(
        json.dumps({"ligand_indices": [0], "env_indices": [1]}),
        encoding="utf-8",
    )

    report = prepare_existing_model_node_config(
        selection_meta_path=selection_meta,
        model_path=model_path,
        output_dir=tmp_path / "prepared",
        min_pair_distance_nm=0.07,
        max_pair_distance_nm=2.5,
        max_radius_of_gyration_nm=0.85,
    )

    config = json.loads(Path(report["config_path"]).read_text())
    assert config["neural_path"]["bases"][0]["support_domain"] == {
        "min_pair_distance_nm": 0.07,
        "max_pair_distance_nm": 2.5,
        "max_radius_of_gyration_nm": 0.85,
    }


def test_existing_model_batch_benchmark_composes_outer_lambda(
    tmp_path, monkeypatch
):
    class FakeAdapter:
        def __init__(self, model_name, device):
            self.model_name = model_name
            self.device = device
            self.label_mode = "mace_decomposition"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def evaluate(self, positions_nm, **kwargs):
            return ExistingOpenMMMLBasisEvaluation(
                model_name=self.model_name,
                label_mode=self.label_mode,
                energy_kj_mol=5.0,
                forces_kj_mol_nm=(
                    (-1.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                ),
                max_force_norm_kj_mol_nm=1.0,
            )

    monkeypatch.setattr(
        "outer_lambda_neural_basis.ExistingOrbMaceBasisAdapter",
        FakeAdapter,
    )
    controller = _controller(
        tmp_path,
        support_domain={"max_pair_distance_nm": 1.0},
    )
    report = benchmark_existing_orb_mace_basis(
        controller,
        model_name="mace-off24-medium",
        device="cpu",
        lambdas=[0.0, 0.5, 1.0],
        frames_nm=[
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]],
        ],
        ligand_indices=[0],
        environment_indices=[1],
        atomic_numbers=[6, 7],
    )

    assert report["passed"] is True
    assert report["label_mode"] == "mace_decomposition"
    assert report["basis_energy_kj_mol"]["mean"] == pytest.approx(5.0)
    assert report["frames"][0]["path_energy_kj_mol"] == pytest.approx(
        [0.0, 2.0, 0.0]
    )
    assert report["frames"][0]["max_path_force_kj_mol_nm"] == pytest.approx(
        [0.0, 0.5, 0.0]
    )


def test_endpoint_path_energy_and_force_are_exact_zero(tmp_path):
    controller = _controller(tmp_path)
    basis = HarmonicDistanceBasis(0, 1, 100.0, 0.2)
    evaluated = basis.evaluate([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])

    assert controller.neural_path_state_energies(
        [0.0, 1.0], [evaluated.energy_kj_mol]
    ) == (0.0, 0.0)
    assert controller.neural_path_forces(
        0.0, [evaluated.forces_kj_mol_nm]
    ) == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert controller.neural_path_forces(
        1.0, [evaluated.forces_kj_mol_nm]
    ) == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def test_midpoint_increment_matches_analytic_energy_and_force(tmp_path):
    controller = _controller(tmp_path)
    basis = HarmonicDistanceBasis(0, 1, 100.0, 0.2)
    evaluated = basis.evaluate([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])

    # U=4.5, b=1.0, A=sin²(pi/2)*0.5=0.5 => B=1.75。
    assert controller.neural_path_state_energies(
        [0.5], [evaluated.energy_kj_mol]
    ) == pytest.approx((1.75,))
    path_forces = controller.neural_path_forces(
        0.5, [evaluated.forces_kj_mol_nm]
    )
    assert path_forces[0] == pytest.approx(
        tuple(0.5 * value for value in evaluated.forces_kj_mol_nm[0])
    )


def test_target_contains_neural_and_lrc_but_bias_history_does_not(tmp_path):
    controller = _controller(tmp_path)
    ledger = IBSEnergyLedger(controller)
    frame = ledger.append_frame(
        lambdas=[0.0, 0.5, 1.0],
        original_interaction_energies_kj_mol=[10.0, 20.0, 30.0],
        lrc_state_energies_kj_mol=[0.1, 0.2, 0.3],
        basis_energies_kj_mol=[5.0],
        sampling_bias_energy_kj_mol=-7.0,
        base_energy_kj_mol=100.0,
    )

    # Ubar=4, A_mid=0.5 => path=[0,2,0]。
    assert frame.neural_path_state_energies_kj_mol == pytest.approx(
        (0.0, 2.0, 0.0)
    )
    assert frame.bias_cv_state_energies_kj_mol == pytest.approx(
        (10.0, 22.0, 30.0)
    )
    assert frame.target_state_energies_kj_mol == pytest.approx(
        (10.1, 22.2, 30.3)
    )
    assert ledger.sampling_bias_history == [-7.0]
    assert ledger.base_energy_history == [100.0]


def test_nonfinite_component_rejects_whole_frame_without_partial_append(tmp_path):
    ledger = IBSEnergyLedger(_controller(tmp_path))
    valid = {
        "lambdas": [0.0, 0.5, 1.0],
        "original_interaction_energies_kj_mol": [10.0, 20.0, 30.0],
        "lrc_state_energies_kj_mol": [0.1, 0.2, 0.3],
        "basis_energies_kj_mol": [5.0],
        "sampling_bias_energy_kj_mol": -7.0,
        "base_energy_kj_mol": 100.0,
    }
    ledger.append_frame(**valid)
    bad = dict(valid)
    bad["lrc_state_energies_kj_mol"] = [0.1, math.nan, 0.3]

    with pytest.raises((NeuralPathFrameError, ValueError), match="有限"):
        ledger.append_frame(**bad)
    assert len(ledger) == 1
    assert set(ledger.history_lengths().values()) == {1}


def test_disabled_path_preserves_old_accounting_and_skips_basis_evaluation(tmp_path):
    ledger = IBSEnergyLedger(_controller(tmp_path, enabled=False))
    frame = ledger.append_frame(
        lambdas=[0.0, 0.5, 1.0],
        original_interaction_energies_kj_mol=[10.0, 20.0, 30.0],
        lrc_state_energies_kj_mol=[0.1, 0.2, 0.3],
        # 禁用时调用方不应计算基势；空序列必须合法。
        basis_energies_kj_mol=[],
        sampling_bias_energy_kj_mol=-7.0,
        base_energy_kj_mol=100.0,
    )

    assert frame.bias_cv_state_energies_kj_mol == (10.0, 20.0, 30.0)
    assert frame.target_state_energies_kj_mol == pytest.approx(
        (10.1, 20.2, 30.3)
    )
    assert frame.basis_energies_kj_mol == ()


def test_common_lambda_has_identical_path_hamiltonian_across_windows(tmp_path):
    controller = _controller(tmp_path)
    left = controller.neural_path_state_energies([0.0, 0.25, 0.5], [5.0])
    right = controller.neural_path_state_energies([0.5, 0.75, 1.0], [5.0])

    assert left[-1] == right[0]
    assert left[-1].hex() == right[0].hex()


def test_series_summary_reports_population_statistics_and_percentiles():
    summary = summarize_finite_series([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary["count"] == 5
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["std"] == pytest.approx(math.sqrt(2.0))
    assert summary["p05"] == pytest.approx(1.2)
    assert summary["p50"] == pytest.approx(3.0)
    assert summary["p95"] == pytest.approx(4.8)
    assert summary["max_abs"] == pytest.approx(5.0)


def test_fixed_atom_selection_and_support_domain_pass(tmp_path):
    controller = _controller(
        tmp_path,
        support_domain={
            "min_pair_distance_nm": 0.1,
            "max_pair_distance_nm": 1.0,
            "max_radius_of_gyration_nm": 0.4,
        },
    )
    evaluation = controller.evaluate_support_domains(
        [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )[0]

    assert controller.bases[0].atom_indices() == (0, 1)
    assert evaluation.supported is True
    assert evaluation.min_pair_distance_nm == pytest.approx(0.5)
    assert evaluation.max_pair_distance_nm == pytest.approx(0.5)
    assert evaluation.radius_of_gyration_nm == pytest.approx(0.25)
    assert evaluation.violations == ()


def test_support_domain_reports_contact_and_extent_violations(tmp_path):
    controller = _controller(
        tmp_path,
        support_domain={
            "min_pair_distance_nm": 0.2,
            "max_pair_distance_nm": 0.8,
            "max_radius_of_gyration_nm": 0.3,
        },
    )
    too_close = controller.evaluate_support_domains(
        [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )[0]
    too_far = controller.evaluate_support_domains(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )[0]

    assert too_close.supported is False
    assert "min_pair_distance_below_support" in too_close.violations
    assert too_far.supported is False
    assert "max_pair_distance_above_support" in too_far.violations
    assert "radius_of_gyration_above_support" in too_far.violations


def test_periodic_support_domain_uses_minimum_image(tmp_path):
    controller = _controller(
        tmp_path,
        periodic=True,
        support_domain={
            "min_pair_distance_nm": 0.1,
            "max_pair_distance_nm": 0.3,
            "max_radius_of_gyration_nm": 0.2,
        },
    )
    evaluation = controller.evaluate_support_domains(
        [[0.05, 0.0, 0.0], [0.95, 0.0, 0.0]],
        box_vectors_nm=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )[0]

    assert evaluation.supported is True
    assert evaluation.min_pair_distance_nm == pytest.approx(0.1)
    assert evaluation.max_pair_distance_nm == pytest.approx(0.1)
    assert evaluation.radius_of_gyration_nm == pytest.approx(0.05)


def test_periodic_support_domain_requires_box(tmp_path):
    controller = _controller(
        tmp_path,
        periodic=True,
        support_domain={"max_pair_distance_nm": 0.5},
    )
    with pytest.raises(NeuralPathConfigError, match="box_vectors_nm"):
        controller.evaluate_support_domains(
            [[0.05, 0.0, 0.0], [0.95, 0.0, 0.0]]
        )


def _wp4_manifest(tmp_path, basis_name="analytic_harmonic_mock"):
    training_data = tmp_path / "training.extxyz"
    training_data.write_bytes(b"frozen representative training data")
    return NeuralBasisTaskManifest.from_mapping(
        {
            "basis_name": basis_name,
            "target_slow_variable": "ligand torsion C1-C2-C3-C4",
            "atom_elements": ["C", "N"],
            "training_data_path": str(training_data),
            "training_data_sha256": hashlib.sha256(
                training_data.read_bytes()
            ).hexdigest(),
            "includes_exchange_waters": False,
            "includes_ions": False,
        }
    )


def _wp4_reports(controller, *, seconds_per_frame=0.01):
    identity = {
        "basis_model_sha256": [controller.bases[0].sha256],
        "atom_selection_sha256": [
            controller.bases[0].atom_indices_sha256
        ],
        "passed": True,
        "safety_violation_count": 0,
        "support_domain_violation_count": 0,
    }
    benchmark = {
        **identity,
        "report_type": "outer_lambda_torchforce_benchmark",
        "overall_seconds_per_state_frame": seconds_per_frame,
    }
    nvt = {
        **identity,
        "report_type": "outer_lambda_torchforce_nvt_smoke",
    }
    return benchmark, nvt


def test_wp4_qualification_closes_identity_stability_and_cost_gates(tmp_path):
    controller = _controller(
        tmp_path,
        support_domain={"max_pair_distance_nm": 1.0},
    )
    benchmark, nvt = _wp4_reports(controller)
    report = qualify_wp4_basis(
        controller,
        _wp4_manifest(tmp_path),
        benchmark,
        nvt,
        max_seconds_per_frame=0.02,
    )

    assert report["qualified"] is True
    assert report["failed_checks"] == []
    assert report["element_counts"] == {"C": 1, "N": 1}
    assert len(report["task_manifest_sha256"]) == 64


def test_wp4_qualification_reports_cost_and_model_identity_failures(tmp_path):
    controller = _controller(
        tmp_path,
        support_domain={"max_pair_distance_nm": 1.0},
    )
    benchmark, nvt = _wp4_reports(controller, seconds_per_frame=0.03)
    benchmark["basis_model_sha256"] = ["0" * 64]
    report = qualify_wp4_basis(
        controller,
        _wp4_manifest(tmp_path),
        benchmark,
        nvt,
        max_seconds_per_frame=0.02,
    )

    assert report["qualified"] is False
    assert "benchmark_model_identity" in report["failed_checks"]
    assert "inference_cost_within_budget" in report["failed_checks"]


def test_atom_selection_file_change_is_detected_after_controller_creation(tmp_path):
    controller = _controller(tmp_path)
    selection_path = Path(controller.bases[0].atom_indices_path)
    selection_path.write_text(json.dumps([0, 2]), encoding="utf-8")

    with pytest.raises(NeuralPathIntegrityError, match="SHA-256"):
        controller.bases[0].atom_indices()


def test_wp5_metric_primitives_are_deterministic():
    uniform = importance_effective_sample_size([0.0, 0.0, 0.0, 0.0])
    concentrated = importance_effective_sample_size([0.0, -20.0, -20.0])

    assert uniform["absolute_ess"] == pytest.approx(4.0)
    assert uniform["ess_ratio"] == pytest.approx(1.0)
    assert concentrated["absolute_ess"] == pytest.approx(1.0, rel=1.0e-7)
    assert count_discrete_transitions(["a", "a", "b", "b", "a"]) == 2

    constant = integrated_autocorrelation_time([2.0] * 8)
    assert constant["statistical_inefficiency"] == pytest.approx(1.0)
    assert constant["effective_uncorrelated_samples"] == pytest.approx(8.0)


def _wp5_arm(
    name,
    *,
    weights,
    states,
    delta_g=10.0,
    uncertainty=0.5,
    anomaly_count=0,
):
    return {
        "name": name,
        "gpu_hours": 1.0,
        "delta_g_kj_mol": delta_g,
        "uncertainty_kj_mol": uncertainty,
        "log_importance_weights": weights,
        "slow_state_labels": states,
        "slow_variable": [0.0, 0.1, 0.2, 0.1],
        "anomaly_count": anomaly_count,
        "n_frames": 4,
        "endpoint_contract_passed": name == "neural_path",
        "accounting_contract_passed": name == "neural_path",
        "mechanical_stability_passed": name == "neural_path",
    }


def test_wp5_three_arm_promotion_passes_only_with_unique_neural_gain():
    report = compare_wp5_arms(
        [
            _wp5_arm(
                "baseline",
                weights=[0.0, -20.0, -20.0, -20.0],
                states=["a", "a", "b", "b"],
            ),
            _wp5_arm(
                "lambda_relayout",
                weights=[0.0, 0.0, -20.0, -20.0],
                states=["a", "b", "b", "a"],
            ),
            _wp5_arm(
                "neural_path",
                weights=[0.0, 0.0, 0.0, 0.0],
                states=["a", "b", "a", "b"],
                delta_g=10.2,
            ),
        ]
    )

    assert report["promotion_passed"] is True
    assert report["failed_checks"] == []
    assert all(report["checks"].values())


def test_wp5_comparison_reports_failed_gates_instead_of_hiding_them():
    arms = [
        _wp5_arm(
            "baseline",
            weights=[0.0, 0.0, 0.0, 0.0],
            states=["a", "b", "a", "b"],
        ),
        _wp5_arm(
            "lambda_relayout",
            weights=[0.0, 0.0, 0.0, 0.0],
            states=["a", "b", "a", "b"],
        ),
        _wp5_arm(
            "neural_path",
            weights=[0.0, -20.0, -20.0, -20.0],
            states=["a", "a", "a", "a"],
            delta_g=20.0,
            anomaly_count=1,
        ),
    ]
    report = compare_wp5_arms(arms)

    assert report["promotion_passed"] is False
    assert "delta_g_consistent" in report["failed_checks"]
    assert "ess_per_gpu_hour_improved_vs_baseline" in report["failed_checks"]
    assert "slow_transitions_improved" in report["failed_checks"]
    assert "anomaly_rate_not_worse" in report["failed_checks"]


def test_wp5_rejects_duplicate_arm_names():
    arm = _wp5_arm(
        "baseline",
        weights=[0.0, 0.0],
        states=["a", "b"],
    )
    with pytest.raises(NeuralPathConfigError, match="必须且只能包含"):
        compare_wp5_arms(
            [
                arm,
                dict(arm),
                _wp5_arm(
                    "lambda_relayout",
                    weights=[0.0, 0.0],
                    states=["a", "b"],
                ),
            ]
        )


def test_wp5_replicated_comparison_aggregates_three_paired_repeats():
    arms = []
    for replicate_index in range(3):
        replicate_id = f"seed-{replicate_index}"
        for arm in (
            _wp5_arm(
                "baseline",
                weights=[0.0, -20.0, -20.0, -20.0],
                states=["a", "a", "b", "b"],
                delta_g=10.0 + 0.1 * replicate_index,
            ),
            _wp5_arm(
                "lambda_relayout",
                weights=[0.0, 0.0, -20.0, -20.0],
                states=["a", "b", "b", "a"],
                delta_g=10.0 + 0.1 * replicate_index,
            ),
            _wp5_arm(
                "neural_path",
                weights=[0.0, 0.0, 0.0, 0.0],
                states=["a", "b", "a", "b"],
                delta_g=10.1 + 0.1 * replicate_index,
            ),
        ):
            arm["replicate_id"] = replicate_id
            arms.append(arm)

    report = compare_wp5_replicates(arms)

    assert report["promotion_passed"] is True
    assert report["minimum_replicates"] == 3
    assert all(arm["replicate_count"] == 3 for arm in report["arms"])
    assert report["arms"][0][
        "delta_g_between_replicate_sd_kj_mol"
    ] == pytest.approx(0.1)


def test_wp5_replicated_comparison_requires_matched_seed_sets():
    arms = []
    for name in ("baseline", "lambda_relayout", "neural_path"):
        for replicate_index in range(3):
            replicate_id = f"seed-{replicate_index}"
            if name == "neural_path" and replicate_index == 2:
                replicate_id = "different-seed"
            arm = _wp5_arm(
                name,
                weights=[0.0, 0.0, 0.0, 0.0],
                states=["a", "b", "a", "b"],
            )
            arm["replicate_id"] = replicate_id
            arms.append(arm)

    with pytest.raises(NeuralPathConfigError, match="配对 replicate_id"):
        compare_wp5_replicates(arms)
