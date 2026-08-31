"""外层 λ 独立 CLI 测试；不导入 ABFE 主程序。"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "outer_lambda_neural_basis.py"


def _config_file(tmp_path):
    model = tmp_path / "model.pt"
    indices = tmp_path / "indices.json"
    model.write_bytes(b"CLI model identity only")
    indices.write_text(json.dumps([0, 1]), encoding="utf-8")
    config = {
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
                    "name": "cli_basis",
                    "backend": "torchforce",
                    "model_path": str(model),
                    "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
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
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _run(*args):
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *map(str, args)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_help_lists_all_subcommands():
    completed = _run("--help")
    assert completed.returncode == 0
    for command in (
        "validate",
        "coefficients",
        "protocol",
        "probe",
        "benchmark",
        "nvt-smoke",
        "label-existing",
        "label-trajectory",
        "mace-nvt-smoke",
        "mace-nvt-qualification",
        "mace-mts-qualification",
        "wp0-select",
        "screen-slow-variables",
        "compare-slow-variable-screens",
        "freeze-slow-variable",
        "exp011-coverage",
        "exp011-fit-pmf",
        "exp011-umbrella-sample",
        "exp011-reweight-umbrella",
        "exp010-label",
        "exp010-fit",
        "exp010-prepare-selection",
        "sample-hard-window-scratch",
        "prepare-existing",
        "compare",
        "compare-replicates",
        "qualify",
    ):
        assert command in completed.stdout


def test_cli_validate_recomputes_hashes(tmp_path):
    config = _config_file(tmp_path)
    completed = _run("validate", "--config", config)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["basis_count"] == 1
    assert payload["basis_names"] == ["cli_basis"]
    assert len(payload["protocol_sha256"]) == 64
    assert len(payload["model_sha256"][0]) == 64
    assert len(payload["atom_selection_sha256"][0]) == 64


def test_cli_coefficients_has_exact_endpoints(tmp_path):
    config = _config_file(tmp_path)
    completed = _run(
        "coefficients",
        "--config",
        config,
        "--lambdas",
        "0,0.5,1",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["coefficient_matrix"] == [[0.0], [0.5], [0.0]]


def test_cli_protocol_can_write_output_file(tmp_path):
    config = _config_file(tmp_path)
    output = tmp_path / "result" / "protocol.json"
    completed = _run(
        "protocol",
        "--config",
        config,
        "--lambdas",
        "0,0.5,1",
        "--output",
        output,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["payload"]["lambda_schedule"] == [0.0, 0.5, 1.0]


def test_cli_bad_model_hash_returns_machine_readable_error(tmp_path):
    config = _config_file(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["neural_path"]["bases"][0]["sha256"] = "0" * 64
    config.write_text(json.dumps(payload), encoding="utf-8")
    completed = _run("validate", "--config", config)

    assert completed.returncode == 2
    error = json.loads(completed.stderr)
    assert error["ok"] is False
    assert error["error_type"] == "NeuralPathIntegrityError"


def test_cli_compare_runs_wp5_three_arm_gate(tmp_path):
    config = _config_file(tmp_path)

    def arm(name, weights, states):
        return {
            "name": name,
            "gpu_hours": 1.0,
            "delta_g_kj_mol": 10.0,
            "uncertainty_kj_mol": 0.5,
            "log_importance_weights": weights,
            "slow_state_labels": states,
            "n_frames": 4,
            "anomaly_count": 0,
            "endpoint_contract_passed": name == "neural_path",
            "accounting_contract_passed": name == "neural_path",
            "mechanical_stability_passed": name == "neural_path",
        }

    comparison = tmp_path / "three_arms.json"
    comparison.write_text(
        json.dumps(
            {
                "arms": [
                    arm(
                        "baseline",
                        [0.0, -20.0, -20.0, -20.0],
                        ["a", "a", "b", "b"],
                    ),
                    arm(
                        "lambda_relayout",
                        [0.0, 0.0, -20.0, -20.0],
                        ["a", "b", "b", "a"],
                    ),
                    arm(
                        "neural_path",
                        [0.0, 0.0, 0.0, 0.0],
                        ["a", "b", "a", "b"],
                    ),
                ]
            }
        ),
        encoding="utf-8",
    )
    completed = _run(
        "compare",
        "--config",
        config,
        "--input",
        comparison,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["promotion_passed"] is True
    assert payload["failed_checks"] == []
    assert len(payload["protocol_sha256"]) == 64
