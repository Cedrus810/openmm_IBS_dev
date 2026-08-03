from __future__ import annotations

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from local_residual.mace_contract import canonical_report_sha256
from local_residual.mace_latent import MaceLatentBasisAdapter, MaceLatentError, load_c0_report


def _report():
    contract = {
        "model_path": "/test/mock.model",
        "model_sha256": "0" * 64,
        "model_class": "test.MockMACE",
        "versions": {"torch": "test", "mace": "0.3.16", "e3nn": "test"},
        "interaction_layer_count": 2,
        "product_layer_count": 2,
        "r_max_angstrom": 6.0,
        "atomic_numbers": [1, 6, 8],
        "node_feats_contract": {
            "tensor_key": "node_feats",
            "source": "torch.cat(product_layer_node_feats, dim=-1)",
            "product_layer_outputs": [],
            "concatenated_dimension": 640,
            "selected_product_layer_index": 1,
            "recommended_invariant_slice": {"start": 512, "stop": 640, "irreps": "128x0e"},
        },
    }
    body = {
        "schema_version": "exp012-mace-model-contract-v1",
        "status": "PASSED_READ_ONLY_ARCHITECTURE_INSPECTION",
        "expected": contract,
        "observed": copy.deepcopy(contract),
        "policy": {
            "model_forward_executed": False,
            "latent_forward_executed": False,
            "numpy_descriptor_path_allowed": False,
            "final_energy_allowed": False,
            "fragment_subtraction_allowed": False,
            "scientific_qualification": False,
        },
    }
    return {**body, "report_sha256": canonical_report_sha256(body)}


class _MockMACE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

    def forward(self, data, training=False, compute_force=False):
        assert training is False and compute_force is False
        positions = data["positions"]
        # Every ligand latent depends on every ligand/environment coordinate.
        coordinate_signal = positions.sum()
        channels = torch.arange(640, dtype=positions.dtype, device=positions.device)
        node_feats = coordinate_signal + positions.sum(dim=1, keepdim=True) + channels
        return {"node_feats": node_feats, "energy": self.weight * coordinate_signal}


class _WrongWidthMACE(_MockMACE):
    def forward(self, data, training=False, compute_force=False):
        return {"node_feats": data["positions"].sum() + torch.zeros((3, 128), dtype=data["positions"].dtype)}


def _graph():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    data = {
        "positions": positions,
        "node_attrs": torch.ones((3, 1), dtype=torch.float64),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "shifts": torch.zeros((2, 3), dtype=torch.float64),
        "unit_shifts": torch.zeros((2, 3), dtype=torch.float64),
        "cell": torch.eye(3, dtype=torch.float64).unsqueeze(0) * 30,
        "batch": torch.zeros(3, dtype=torch.long),
        "ptr": torch.tensor([0, 3]),
        "head": torch.zeros(1, dtype=torch.long),
        "pbc": torch.ones((1, 3), dtype=torch.bool),
        "total_charge": torch.zeros(1, dtype=torch.float64),
        "total_spin": torch.ones(1, dtype=torch.float64),
    }
    return {
        "data": data,
        "ligand_mask": torch.tensor([True, False, True]),
        "environment_mask": torch.tensor([False, True, False]),
        "diagnostics": {"edge_membership_piecewise_fixed": True},
    }


def _adapter(model):
    return MaceLatentBasisAdapter(
        c0_report=_report(), model_path=Path("/test/mock.model"),
        device="cpu", dtype="float64", model_loader=lambda *_: model,
        allow_test_model_identity=True,
    )


def test_exact_512_640_slice_is_differentiable_for_ligand_and_environment():
    graph = _graph()
    model = _MockMACE()
    result = _adapter(model)(graph)
    latent = result["ligand_latent"]
    assert latent.shape == (2, 128)
    assert latent[0, 0].item() == pytest.approx(515.0)
    assert result["tensor_metadata"]["selected_slice"] == [512, 640]
    assert result["tensor_metadata"]["energy_fields_exposed"] is False
    assert "energy" not in result and "node_feats" not in result
    gradient = torch.autograd.grad(latent.sum(), graph["data"]["positions"])[0]
    assert torch.isfinite(gradient).all()
    assert gradient[[0, 2]].abs().sum() > 0
    assert gradient[1].abs().sum() > 0
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_repeat_forward_is_identical_and_latent_not_detached():
    graph = _graph()
    adapter = _adapter(_MockMACE())
    first = adapter(graph)["ligand_latent"]
    second = adapter(graph)["ligand_latent"]
    assert first.grad_fn is not None and second.grad_fn is not None
    assert torch.equal(first, second)


def test_early_stop_uses_second_product_scalar_block_and_skips_third_product():
    report = _report()
    products = [
        {"index": 0, "dimension": 5, "irreps": "2x0e+1x1o", "concatenated_start": 0, "concatenated_stop": 5},
        {"index": 1, "dimension": 5, "irreps": "2x0e+1x1o", "concatenated_start": 5, "concatenated_stop": 10},
        {"index": 2, "dimension": 2, "irreps": "2x0e", "concatenated_start": 10, "concatenated_stop": 12},
    ]
    report["expected"]["interaction_layer_count"] = 3
    report["expected"]["product_layer_count"] = 3
    report["expected"]["node_feats_contract"]["product_layer_outputs"] = products
    report["observed"] = copy.deepcopy(report["expected"])
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = canonical_report_sha256(body)

    class Product(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset

        def forward(self, node_feats):
            return node_feats + self.offset

    class MustNotRun(torch.nn.Module):
        def forward(self, node_feats):
            raise AssertionError("third product executed")

    class ThreeProductModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
            self.products = torch.nn.ModuleList([Product(1.0), Product(2.0), MustNotRun()])

        def forward(self, data, training=False, compute_force=False):
            signal = data["positions"].sum(dim=1, keepdim=True)
            node_feats = signal.expand(-1, 5)
            outputs = []
            for product in self.products:
                node_feats = product(node_feats)
                outputs.append(node_feats)
            return {"node_feats": torch.cat(outputs, dim=-1)}

    adapter = MaceLatentBasisAdapter(
        c0_report=report,
        model_path=Path("/test/mock.model"),
        device="cpu",
        dtype="float64",
        model_loader=lambda *_: ThreeProductModel(),
        allow_test_model_identity=True,
        product_layer_index=1,
    )
    graph = _graph()
    result = adapter(graph)
    assert result["ligand_latent"].shape == (2, 2)
    assert result["tensor_metadata"]["selected_product_layer_index"] == 1
    assert result["tensor_metadata"]["forward_stopped_after_selected_product_layer"] is True
    gradient = torch.autograd.grad(result["ligand_latent"].sum(), graph["data"]["positions"])[0]
    assert gradient.abs().sum() > 0
    with torch.no_grad():
        no_grad = adapter.forward(graph, require_coordinate_grad=False)
    assert no_grad["ligand_latent"].grad_fn is None


def test_invalid_contract_slice_and_wrong_runtime_width_are_rejected():
    report = _report()
    report["expected"]["node_feats_contract"]["recommended_invariant_slice"] = {
        "start": 512, "stop": 641, "irreps": "129x0e"
    }
    report["observed"] = copy.deepcopy(report["expected"])
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = canonical_report_sha256(body)
    with pytest.raises(MaceLatentError, match="invariant slice is invalid"):
        load_c0_report(report)
    with pytest.raises(MaceLatentError, match=r"\[nodes,640\]"):
        _adapter(_WrongWidthMACE())(_graph())


def test_detached_latent_and_bad_c0_hash_fail_closed():
    class Detached(_MockMACE):
        def forward(self, data, training=False, compute_force=False):
            return {"node_feats": torch.zeros((3, 640), dtype=data["positions"].dtype)}

    with pytest.raises(MaceLatentError, match="detached"):
        _adapter(Detached())(_graph())
    report = _report()
    report["report_sha256"] = "f" * 64
    with pytest.raises(MaceLatentError, match="SHA mismatch"):
        load_c0_report(report)


def test_import_is_lazy_and_real_model_requires_explicit_enable(tmp_path):
    # Module import above did not import mace. The real model is exercised only
    # when a dedicated node run opts in; routine tests never load/download it.
    import os
    import subprocess
    import sys
    if os.environ.get("EXP012_RUN_REAL_MACE_LATENT_TEST") != "1":
        pytest.skip("set EXP012_RUN_REAL_MACE_LATENT_TEST=1 for the real model smoke")
    required = {
        "model": "EXP012_MACE_MODEL_PATH",
        "c0": "EXP012_MACE_C0_REPORT",
        "environment": "EXP012_ENVIRONMENT_MANIFEST",
        "mapping": "EXP012_ATOM_MAPPING",
        "topology": "EXP012_FRAME_TOPOLOGY",
        "trajectory": "EXP012_FRAME_TRAJECTORY",
    }
    missing = [variable for variable in required.values() if not os.environ.get(variable)]
    if missing:
        pytest.fail("real latent smoke requested but inputs are missing: " + ", ".join(missing))
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "real_mace_latent_smoke.json"
    result = subprocess.run(
        [
            sys.executable, str(root / "scripts" / "smoke_exp012_mace_latent.py"),
            "--model", os.environ[required["model"]],
            "--c0-report", os.environ[required["c0"]],
            "--environment-manifest", os.environ[required["environment"]],
            "--atom-mapping", os.environ[required["mapping"]],
            "--topology", os.environ[required["topology"]],
            "--trajectory", os.environ[required["trajectory"]],
            "--frame-indices", os.environ.get("EXP012_FRAME_INDEX", "0"),
            "--edge-cutoff-angstrom", "6.0",
            "--geometric-upper-bound-angstrom",
            os.environ.get("EXP012_GEOMETRIC_UPPER_BOUND_ANGSTROM", "12.0"),
            "--device", os.environ.get("EXP012_MACE_DEVICE", "cpu"),
            "--dtype", os.environ.get("EXP012_MACE_DTYPE", "float64"),
            "--output", str(output),
        ],
        cwd=root, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = __import__("json").loads(output.read_text(encoding="utf-8"))
    c0 = __import__("json").loads(Path(os.environ[required["c0"]]).read_text(encoding="utf-8"))
    selected = c0["expected"]["node_feats_contract"]["recommended_invariant_slice"]
    expected_latent_dimension = selected["stop"] - selected["start"]
    assert report["status"] == "COMPLETED_AUTOGRAD_SMOKE_ONLY"
    assert report["frames"][0]["latent"]["shape"][1] == expected_latent_dimension
    assert report["frames"][0]["mace_parameter_grad_count"] == 0
