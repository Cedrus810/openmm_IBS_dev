"""只读导入现有 IBS API，验证独立神经路径产物的文件/估计器兼容性。

本文件不要求、也不允许修改 ``ibs_engine.py``。它把独立模块当作数据生产者，把
现有 IBS 函数当作只读消费者。
"""

import hashlib
import json

import numpy as np
import pytest


pytestmark = pytest.mark.cpu_only

pytest.importorskip("openmm")
pytest.importorskip("pymbar")

import ibs_engine as ie
from outer_lambda_neural_basis import IBSEnergyLedger, OuterLambdaController


def _controller(tmp_path):
    model = tmp_path / "standalone-placeholder.pt"
    indices = tmp_path / "indices.json"
    model.write_bytes(b"standalone accounting model identity")
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
                        "name": "standalone_basis",
                        "backend": "torchforce",
                        "model_path": str(model),
                        "sha256": hashlib.sha256(
                            model.read_bytes()
                        ).hexdigest(),
                        "energy_offset_kj_mol": 1.0,
                        "atom_selection": "fixed_indices",
                        "atom_indices_path": str(indices),
                        "output_unit": "kJ_per_mol",
                        "precision": "double",
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


def test_standalone_ledger_triplet_is_accepted_by_existing_ibs_reader(tmp_path):
    """独立 target/bias/base 可直接交给现有 IBS 三文件读取器。"""

    ledger = IBSEnergyLedger(_controller(tmp_path))
    for frame_index in range(4):
        ledger.append_frame(
            lambdas=[0.0, 0.5, 1.0],
            original_interaction_energies_kj_mol=[
                10.0 + frame_index,
                20.0 + frame_index,
                30.0 + frame_index,
            ],
            lrc_state_energies_kj_mol=[0.1, 0.2, 0.3],
            basis_energies_kj_mol=[5.0 + 0.1 * frame_index],
            sampling_bias_energy_kj_mol=-7.0 + 0.1 * frame_index,
            base_energy_kj_mol=100.0 + frame_index,
        )

    energies = np.asarray(ledger.target_energy_history, dtype=np.float64).T
    bias = np.asarray(ledger.sampling_bias_history, dtype=np.float64)
    base = np.asarray(ledger.base_energy_history, dtype=np.float64)
    energies_path = tmp_path / "energies.npy"
    bias_path = tmp_path / "bias.npy"
    base_path = tmp_path / "base.npy"
    np.save(energies_path, energies)
    np.save(bias_path, bias)
    np.save(base_path, base)

    metadata = ie._window_data_metadata(
        str(energies_path), str(bias_path), str(base_path)
    )
    convergence = {
        "window_data_protocol_version": ie.IBS_WINDOW_DATA_PROTOCOL_VERSION,
        "window_data": metadata,
    }
    loaded = ie._load_validated_window_data_triplet(
        str(energies_path),
        str(bias_path),
        str(base_path),
        convergence,
    )

    assert metadata["n_frames"] == 4
    assert np.array_equal(loaded[0], energies)
    assert np.array_equal(loaded[1], bias)
    assert np.array_equal(loaded[2], base)


def test_existing_history_contract_accepts_external_adapter_object():
    """无需改 IBSSampler；具有同名 histories 的外部对象即可复用现有契约函数。"""

    class ExternalHistoryAdapter:
        energy_history = [(1.0, 2.0), (1.1, 2.1)]
        bias_history = [-1.0, -0.9]
        base_energy_history = [100.0, 101.0]

    assert ie._production_history_lengths(ExternalHistoryAdapter()) == 2


def test_main_ibs_constructor_signature_remains_unmodified():
    """独立研发不得偷偷向主类构造器加入 neural 参数。"""

    import inspect

    parameters = inspect.signature(ie.IBSBiasForce.__init__).parameters
    assert list(parameters) == ["self", "n_states", "temperature", "prefix"]
    assert all("neural" not in name for name in parameters)
