"""可选 GPU 验收：无 CUDA/OpenMM-Torch CUDA 支持时明确 skip。"""

import hashlib
import json

import numpy as np
import pytest


pytestmark = [
    pytest.mark.needs_gpu,
    pytest.mark.filterwarnings(
        "ignore:.*torch\\.jit\\.trace.*deprecated.*:DeprecationWarning"
    ),
]

openmm = pytest.importorskip("openmm")
pytest.importorskip("openmmtorch")
torch = pytest.importorskip("torch")

from outer_lambda_neural_basis import (
    OuterLambdaController,
    TorchForceDeploymentError,
    build_torchforce_outer_lambda_force,
    evaluate_openmm_outer_lambda_force,
)


def _cuda_available():
    return any(
        openmm.Platform.getPlatform(index).getName() == "CUDA"
        for index in range(openmm.Platform.getNumPlatforms())
    )


def test_cuda_matches_reference_for_frozen_torchforce(tmp_path):
    if not _cuda_available():
        pytest.skip("OpenMM CUDA platform 不可用")

    class PositionSquaredEnergy(torch.nn.Module):
        def forward(self, positions):
            return torch.sum(positions**2)

    model = torch.jit.trace(
        PositionSquaredEnergy(), torch.zeros((2, 3), dtype=torch.float32)
    )
    model_path = tmp_path / "cuda_position_squared.pt"
    model.save(str(model_path))
    indices = tmp_path / "indices.json"
    indices.write_text(json.dumps([0, 1]), encoding="utf-8")
    controller = OuterLambdaController.from_mapping(
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
                        "name": "cuda_mock",
                        "backend": "torchforce",
                        "model_path": str(model_path),
                        "sha256": hashlib.sha256(
                            model_path.read_bytes()
                        ).hexdigest(),
                        "energy_offset_kj_mol": 0.0,
                        "atom_selection": "fixed_indices",
                        "atom_indices_path": str(indices),
                        "output_unit": "kJ_per_mol",
                        "precision": "single",
                        "periodic": False,
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
    positions = [[0.5, 0.1, -0.2], [0.0, 0.0, 0.0]]
    reference = evaluate_openmm_outer_lambda_force(
        build_torchforce_outer_lambda_force(controller, 0.5),
        lambda_value=0.5,
        positions_nm=positions,
        platform_name="Reference",
    )
    try:
        cuda = evaluate_openmm_outer_lambda_force(
            build_torchforce_outer_lambda_force(controller, 0.5),
            lambda_value=0.5,
            positions_nm=positions,
            platform_name="CUDA",
        )
    except TorchForceDeploymentError as exc:
        pytest.skip(f"CUDA platform 存在但当前节点无法创建 TorchForce Context: {exc}")

    assert cuda.energy_kj_mol == pytest.approx(
        reference.energy_kj_mol, rel=1.0e-5, abs=1.0e-7
    )
    assert np.allclose(
        np.asarray(cuda.forces_kj_mol_nm),
        np.asarray(reference.forces_kj_mol_nm),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
