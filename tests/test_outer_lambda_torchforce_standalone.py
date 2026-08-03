"""WP-3：完全独立于 ABFE 主程序的 OpenMM/TorchForce 最小部署测试。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.filterwarnings(
        "ignore:.*torch\\.jit\\.trace.*deprecated.*:DeprecationWarning"
    ),
]

openmm = pytest.importorskip("openmm")
from openmm import unit

from outer_lambda_neural_basis import (
    OuterLambdaController,
    build_openmm_outer_lambda_force,
    build_torchforce_outer_lambda_force,
    deserialize_openmm_force,
    evaluate_openmm_outer_lambda_force,
    serialize_openmm_force,
)


def _controller(tmp_path, model_path, *, periodic=False):
    indices = tmp_path / "indices.json"
    indices.write_text(json.dumps([0, 1]), encoding="utf-8")
    return OuterLambdaController.from_mapping(
        {
            "neural_path": {
                "enabled": True,
                "protocol_version": 1,
                "stage": "vanishing",
                "baseline_potential": "softcore",
                "endpoint_tolerance": 1.0e-12,
                "envelope": {"type": "sin2", "parameters": {}},
                "coefficient_model": {
                    "type": "constant",
                    "coefficients": [0.5],
                    "max_abs_coefficient": 1.0,
                },
                "bases": [
                    {
                        "name": "standalone_torch_mock",
                        "backend": "torchforce",
                        "model_path": str(model_path),
                        "sha256": hashlib.sha256(
                            model_path.read_bytes()
                        ).hexdigest(),
                        "energy_offset_kj_mol": 0.0,
                        "atom_selection": "fixed_indices",
                        "atom_indices_path": str(indices),
                        "output_unit": "kJ_per_mol",
                        "precision": "double",
                        "periodic": periodic,
                    }
                ],
                "safety": {
                    "max_abs_basis_energy_kj_mol": 50.0,
                    "max_abs_path_energy_kj_mol": 20.0,
                    "max_force_norm_kj_mol_nm": 500.0,
                    "fail_on_support_domain_violation": True,
                },
            }
        }
    )


def _harmonic_openmm_basis():
    basis = openmm.CustomBondForce("0.5*k*(r-r0)^2")
    basis.addGlobalParameter("k", 100.0)
    basis.addGlobalParameter("r0", 0.2)
    basis.addBond(0, 1, [])
    return basis


def test_analytic_openmm_basis_endpoint_energy_and_force_are_zero(tmp_path):
    placeholder = tmp_path / "placeholder.pt"
    placeholder.write_bytes(b"not loaded by analytic test")
    controller = _controller(tmp_path, placeholder)
    positions = [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]

    for endpoint in (0.0, 1.0):
        force = build_openmm_outer_lambda_force(
            controller, endpoint, [_harmonic_openmm_basis()]
        )
        result = evaluate_openmm_outer_lambda_force(
            force, lambda_value=endpoint, positions_nm=positions
        )
        assert result.energy_kj_mol == pytest.approx(0.0, abs=1.0e-12)
        assert result.max_force_norm_kj_mol_nm == pytest.approx(
            0.0, abs=1.0e-12
        )


def test_analytic_openmm_basis_midpoint_matches_outer_coefficient(tmp_path):
    placeholder = tmp_path / "placeholder.pt"
    placeholder.write_bytes(b"not loaded by analytic test")
    controller = _controller(tmp_path, placeholder)
    force = build_openmm_outer_lambda_force(
        controller, 0.5, [_harmonic_openmm_basis()]
    )
    result = evaluate_openmm_outer_lambda_force(
        force,
        lambda_value=0.5,
        positions_nm=[[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )

    # U=0.5*100*(0.5-0.2)^2=4.5；A=0.5。
    assert result.energy_kj_mol == pytest.approx(2.25)
    assert result.forces_kj_mol_nm[0][0] == pytest.approx(-15.0)
    assert result.forces_kj_mol_nm[1][0] == pytest.approx(15.0)


def test_production_modules_do_not_import_standalone_neural_module():
    repo_root = Path(__file__).resolve().parents[1]
    for filename in (
        "runabfe.py",
        "abfe_core.py",
        "abfe_pipeline.py",
        "ibs_engine.py",
    ):
        source = (repo_root / filename).read_text(encoding="utf-8")
        assert "outer_lambda_neural_basis" not in source


def test_torchforce_energy_force_and_xml_rebuild(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("openmmtorch")

    class PositionSquaredEnergy(torch.nn.Module):
        def forward(self, positions):
            return torch.sum(positions**2)

    # trace 避免本地测试函数中定义的类受 TorchScript source lookup 限制。
    model = torch.jit.trace(
        PositionSquaredEnergy(), torch.zeros((2, 3), dtype=torch.float32)
    )
    model_path = tmp_path / "position_squared.pt"
    model.save(str(model_path))
    controller = _controller(tmp_path, model_path)

    force = build_torchforce_outer_lambda_force(controller, 0.5)
    result = evaluate_openmm_outer_lambda_force(
        force,
        lambda_value=0.5,
        positions_nm=[[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    assert result.energy_kj_mol == pytest.approx(0.125, rel=1.0e-6)
    assert result.forces_kj_mol_nm[0][0] == pytest.approx(
        -0.5, rel=1.0e-6
    )

    xml = serialize_openmm_force(
        build_torchforce_outer_lambda_force(controller, 0.5)
    )
    rebuilt = deserialize_openmm_force(xml)
    rebuilt_result = evaluate_openmm_outer_lambda_force(
        rebuilt,
        lambda_value=0.5,
        positions_nm=[[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    assert rebuilt_result.energy_kj_mol == pytest.approx(
        result.energy_kj_mol, rel=1.0e-7
    )
    assert np.allclose(
        np.asarray(rebuilt_result.forces_kj_mol_nm, dtype=np.float64),
        np.asarray(result.forces_kj_mol_nm, dtype=np.float64),
        rtol=1.0e-7,
        atol=1.0e-10,
    )


def _save_position_squared_model(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("openmmtorch")

    class PositionSquaredEnergy(torch.nn.Module):
        def forward(self, positions):
            return torch.sum(positions**2)

    model = torch.jit.trace(
        PositionSquaredEnergy(), torch.zeros((2, 3), dtype=torch.float32)
    )
    path = tmp_path / "position_squared_shared.pt"
    model.save(str(path))
    return path


def _build_context(controller, lambda_value, positions_nm, platform_name):
    force = build_torchforce_outer_lambda_force(controller, lambda_value)
    system = openmm.System()
    system.addParticle(12.0 * unit.dalton)
    system.addParticle(12.0 * unit.dalton)
    system.addForce(force)
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(
        [openmm.Vec3(*xyz) for xyz in positions_nm] * unit.nanometer
    )
    return system, integrator, context


def _context_energy_force(context):
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    return float(energy), np.asarray(forces, dtype=np.float64)


def test_reference_and_cpu_platforms_agree(tmp_path):
    model_path = _save_position_squared_model(tmp_path)
    controller = _controller(tmp_path, model_path)
    positions = [[0.5, 0.1, -0.2], [0.0, 0.0, 0.0]]
    available = {
        openmm.Platform.getPlatform(index).getName()
        for index in range(openmm.Platform.getNumPlatforms())
    }
    if "CPU" not in available:
        pytest.skip("OpenMM CPU platform 不可用")

    reference = evaluate_openmm_outer_lambda_force(
        build_torchforce_outer_lambda_force(controller, 0.5),
        lambda_value=0.5,
        positions_nm=positions,
        platform_name="Reference",
    )
    cpu = evaluate_openmm_outer_lambda_force(
        build_torchforce_outer_lambda_force(controller, 0.5),
        lambda_value=0.5,
        positions_nm=positions,
        platform_name="CPU",
    )

    assert cpu.energy_kj_mol == pytest.approx(
        reference.energy_kj_mol, rel=1.0e-6, abs=1.0e-8
    )
    assert np.allclose(
        np.asarray(cpu.forces_kj_mol_nm),
        np.asarray(reference.forces_kj_mol_nm),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_two_torchforce_contexts_can_coexist(tmp_path):
    """模拟 production Context 与能量 probe Context 同时存活。"""

    model_path = _save_position_squared_model(tmp_path)
    controller = _controller(tmp_path, model_path)
    system_a, integrator_a, context_a = _build_context(
        controller,
        0.5,
        [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "Reference",
    )
    system_b, integrator_b, context_b = _build_context(
        controller,
        0.5,
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "Reference",
    )
    try:
        energy_a, forces_a = _context_energy_force(context_a)
        energy_b, forces_b = _context_energy_force(context_b)
        # 再读一次 A，确保创建/评价 B 没有污染 A。
        energy_a_again, forces_a_again = _context_energy_force(context_a)
        assert energy_a == pytest.approx(0.125)
        assert energy_b == pytest.approx(0.5)
        assert energy_a_again == pytest.approx(energy_a)
        assert np.array_equal(forces_a_again, forces_a)
        assert not np.array_equal(forces_a, forces_b)
    finally:
        del context_b
        del integrator_b
        del system_b
        del context_a
        del integrator_a
        del system_a


def test_torchforce_context_checkpoint_restores_coordinates_and_energy(tmp_path):
    model_path = _save_position_squared_model(tmp_path)
    controller = _controller(tmp_path, model_path)
    system, integrator, context = _build_context(
        controller,
        0.5,
        [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "Reference",
    )
    try:
        initial_energy, initial_forces = _context_energy_force(context)
        checkpoint = context.createCheckpoint()
        assert isinstance(checkpoint, bytes)
        assert checkpoint

        context.setPositions(
            [
                openmm.Vec3(1.0, 0.0, 0.0),
                openmm.Vec3(0.0, 0.0, 0.0),
            ]
            * unit.nanometer
        )
        changed_energy, _ = _context_energy_force(context)
        assert changed_energy != pytest.approx(initial_energy)

        context.loadCheckpoint(checkpoint)
        restored_energy, restored_forces = _context_energy_force(context)
        restored_positions = (
            context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(unit.nanometer)
        )
        assert restored_energy == pytest.approx(initial_energy, rel=1.0e-7)
        assert np.allclose(restored_forces, initial_forces)
        assert np.allclose(
            np.asarray(restored_positions),
            np.asarray([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )
    finally:
        del context
        del integrator
        del system


def test_periodic_torchforce_receives_box_vectors_and_uses_minimum_image(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("openmmtorch")

    class PeriodicPairEnergy(torch.nn.Module):
        def forward(self, positions, boxvectors):
            boxsize = torch.diag(boxvectors)
            delta = positions[0] - positions[1]
            delta = delta - torch.round(delta / boxsize) * boxsize
            return torch.sum(delta**2)

    example_positions = torch.tensor(
        [[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]], dtype=torch.float32
    )
    example_box = torch.eye(3, dtype=torch.float32) * 2.0
    model = torch.jit.trace(
        PeriodicPairEnergy(), (example_positions, example_box)
    )
    model_path = tmp_path / "periodic_pair.pt"
    model.save(str(model_path))
    controller = _controller(tmp_path, model_path, periodic=True)

    result = evaluate_openmm_outer_lambda_force(
        build_torchforce_outer_lambda_force(controller, 0.5),
        lambda_value=0.5,
        positions_nm=[[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]],
        box_vectors_nm=[
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        platform_name="Reference",
    )

    # minimum-image r=0.2；raw U=r²=0.04；外层 A=0.5。
    assert result.energy_kj_mol == pytest.approx(0.02, rel=1.0e-5)
    assert result.forces_kj_mol_nm[0][0] == pytest.approx(
        -0.2, rel=1.0e-5
    )
    assert result.forces_kj_mol_nm[1][0] == pytest.approx(
        0.2, rel=1.0e-5
    )
