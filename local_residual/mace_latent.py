"""Frozen, differentiable MACE latent adapter for EXP-012 Arm C smoke tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from local_residual.mace_contract import canonical_report_sha256


# Legacy OFF24 values remain exported for compatibility with older tests and
# reports.  Runtime dimensions and slices are always read from the selected
# C0 report; OMOL and future models must not inherit these constants.
EXP012_MODEL_PATH = "/home/ruigengji/.cache/mace/MACE-OFF24_medium.model"
EXP012_MODEL_SHA256 = "e5ccf5837f685899811a68754e7c994393bfd1a81720393b03c643b46c70bc69"
NODE_FEATS_DIMENSION = 640
LATENT_SLICE = slice(512, 640)
LATENT_IRREPS = "128x0e"


class MaceLatentError(RuntimeError):
    """The C0 identity, graph, model, or differentiable latent contract failed."""


class _MaceProductLayerStop(RuntimeError):
    """Private control-flow exception used to stop MACE after a chosen product."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_c0_report(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        report = dict(value)
    else:
        try:
            report = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MaceLatentError(f"cannot read C0 report: {value}") from exc
    if not isinstance(report, dict):
        raise MaceLatentError("C0 report must be a JSON object")
    report_hash = report.get("report_sha256")
    unhashed = {key: value for key, value in report.items() if key != "report_sha256"}
    if report_hash != canonical_report_sha256(unhashed):
        raise MaceLatentError("C0 report canonical SHA mismatch")
    if report.get("status") != "PASSED_READ_ONLY_ARCHITECTURE_INSPECTION":
        raise MaceLatentError("C0 report did not pass architecture inspection")
    expected = report.get("expected")
    observed = report.get("observed")
    if not isinstance(expected, Mapping) or observed != expected:
        raise MaceLatentError("C0 expected and observed contracts differ")
    node_contract = expected.get("node_feats_contract")
    required = {
        "tensor_key": "node_feats",
        "source": "torch.cat(product_layer_node_feats, dim=-1)",
    }
    if not isinstance(node_contract, Mapping) or any(
        node_contract.get(key) != value for key, value in required.items()
    ):
        raise MaceLatentError("C0 node_feats concatenation contract is invalid")
    try:
        dimension = int(node_contract["concatenated_dimension"])
        selected = node_contract["recommended_invariant_slice"]
        start = int(selected["start"])
        stop = int(selected["stop"])
        irreps = str(selected["irreps"])
        interaction_layers = int(expected["interaction_layer_count"])
        r_max = float(expected["r_max_angstrom"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MaceLatentError("C0 latent/graph contract is incomplete") from exc
    if dimension <= 0 or not 0 <= start < stop <= dimension or not irreps:
        raise MaceLatentError("C0 invariant slice is invalid")
    if interaction_layers <= 0 or r_max <= 0.0:
        raise MaceLatentError("C0 graph layer/cutoff contract is invalid")
    policy = report.get("policy")
    if not isinstance(policy, Mapping) or any(
        policy.get(field) is not False
        for field in (
            "numpy_descriptor_path_allowed",
            "final_energy_allowed",
            "fragment_subtraction_allowed",
        )
    ):
        raise MaceLatentError("C0 forbidden-path policy is missing or permissive")
    return report


class MaceLatentBasisAdapter:
    """Load one frozen model and return its C0-selected ligand invariant slice.

    ``model_loader`` and ``allow_test_model_identity`` are explicit test seams;
    production use verifies that the model path matches the selected C0 report.
    """

    def __init__(
        self,
        *,
        c0_report: str | Path | Mapping[str, Any],
        model_path: str | Path = EXP012_MODEL_PATH,
        device: str = "cpu",
        dtype: str = "float64",
        model_loader: Callable[[Path, str, str], Any] | None = None,
        allow_test_model_identity: bool = False,
        product_layer_index: int | None = None,
    ) -> None:
        self.report = load_c0_report(c0_report)
        self.model_path = Path(model_path).expanduser().resolve()
        self.device_name = device
        self.dtype_name = dtype
        if device != "cpu" and not device.startswith("cuda"):
            raise MaceLatentError("device must be cpu or an explicit cuda device")
        if dtype not in {"float32", "float64"}:
            raise MaceLatentError("dtype must be float32 or float64")
        expected = self.report["expected"]
        node_contract = expected["node_feats_contract"]
        self.product_layer_index = product_layer_index
        if product_layer_index is None:
            selected = node_contract["recommended_invariant_slice"]
            self.node_feats_dimension = int(node_contract["concatenated_dimension"])
            self.latent_slice = slice(int(selected["start"]), int(selected["stop"]))
            self.latent_irreps = str(selected["irreps"])
        else:
            if isinstance(product_layer_index, bool) or not isinstance(product_layer_index, int):
                raise MaceLatentError("product_layer_index must be an integer or None")
            products = node_contract.get("product_layer_outputs")
            if not isinstance(products, list) or not 0 <= product_layer_index < len(products):
                raise MaceLatentError("product_layer_index is outside the C0 product layers")
            product = products[product_layer_index]
            if product.get("index") != product_layer_index:
                raise MaceLatentError("C0 product layer ordering is inconsistent")
            self.node_feats_dimension = int(product["dimension"])
            product_irreps = str(product["irreps"])
            scalar_prefix = re.match(r"^(\d+)x0e(?:\+|$)", product_irreps)
            if scalar_prefix is None:
                raise MaceLatentError(
                    "selected product layer does not begin with an invariant 0e block"
                )
            scalar_dimension = int(scalar_prefix.group(1))
            self.latent_slice = slice(0, scalar_dimension)
            self.latent_irreps = f"{scalar_dimension}x0e"
        self.latent_dimension = self.latent_slice.stop - self.latent_slice.start
        if not allow_test_model_identity:
            expected_path = Path(str(expected.get("model_path", ""))).expanduser().resolve()
            if self.model_path != expected_path:
                raise MaceLatentError("model path differs from the selected C0 identity")
            if not self.model_path.is_file():
                raise MaceLatentError("selected MACE model file is missing")
        self._loader = model_loader
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MaceLatentError("Torch/MACE runtime is unavailable") from exc
        if self.device_name.startswith("cuda") and not torch.cuda.is_available():
            raise MaceLatentError("CUDA was requested but is unavailable")
        torch_dtype = torch.float64 if self.dtype_name == "float64" else torch.float32
        if self._loader is None:
            try:
                model = torch.load(
                    self.model_path,
                    map_location=torch.device(self.device_name),
                    weights_only=False,
                )
            except Exception as exc:
                raise MaceLatentError(f"failed to load identified MACE model: {exc}") from exc
        else:
            model = self._loader(self.model_path, self.device_name, self.dtype_name)
        try:
            model = model.to(device=torch.device(self.device_name), dtype=torch_dtype)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        except Exception as exc:
            raise MaceLatentError(f"failed to freeze/configure MACE model: {exc}") from exc
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise MaceLatentError("MACE parameters are not completely frozen")
        self._model = model
        return model

    def forward(
        self,
        graph_batch: Mapping[str, Any],
        *,
        require_coordinate_grad: bool = True,
    ) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise MaceLatentError("Torch runtime is unavailable") from exc
        if not isinstance(graph_batch, Mapping) or set(graph_batch) < {
            "data",
            "ligand_mask",
            "environment_mask",
            "diagnostics",
        }:
            raise MaceLatentError("graph batch is missing required graph/mask fields")
        data = graph_batch["data"]
        if not isinstance(data, Mapping):
            raise MaceLatentError("graph data must be a mapping")
        required_fields = {
            "positions", "node_attrs", "edge_index", "shifts", "unit_shifts",
            "cell", "batch", "ptr", "head", "pbc", "total_charge", "total_spin",
        }
        missing = sorted(required_fields - set(data))
        if missing:
            raise MaceLatentError(f"graph data is missing fields: {missing}")
        positions = data["positions"]
        if not isinstance(positions, torch.Tensor):
            raise MaceLatentError("graph positions must be a Torch tensor")
        if require_coordinate_grad and not positions.requires_grad:
            raise MaceLatentError(
                "graph positions must be a requires_grad Torch tensor when "
                "require_coordinate_grad is True"
            )
        expected_dtype = torch.float64 if self.dtype_name == "float64" else torch.float32
        expected_device = torch.device(self.device_name)
        if expected_device.type == "cuda" and expected_device.index is None:
            expected_device = torch.device("cuda", torch.cuda.current_device())
        if positions.dtype != expected_dtype or positions.device != expected_device:
            raise MaceLatentError("graph dtype/device differs from the explicit adapter contract")
        model = self._load()
        if self.product_layer_index is None:
            try:
                results = model(data, training=False, compute_force=False)
            except Exception as exc:
                raise MaceLatentError(f"real Tensor MACE forward failed: {exc}") from exc
            if not isinstance(results, Mapping) or "node_feats" not in results:
                raise MaceLatentError("MACE forward did not return node_feats")
            node_feats = results["node_feats"]
        else:
            products = getattr(model, "products", None)
            if products is None or self.product_layer_index >= len(products):
                raise MaceLatentError("loaded MACE model lacks the C0-selected product layer")
            captured: dict[str, Any] = {}

            def stop_after_product(_module, _inputs, output):
                captured["node_feats"] = output
                raise _MaceProductLayerStop()

            handle = products[self.product_layer_index].register_forward_hook(stop_after_product)
            try:
                model(data, training=False, compute_force=False)
            except _MaceProductLayerStop:
                pass
            except Exception as exc:
                raise MaceLatentError(
                    f"early-stop Tensor MACE forward failed: {exc}"
                ) from exc
            finally:
                handle.remove()
            if "node_feats" not in captured:
                raise MaceLatentError("selected MACE product layer hook did not execute")
            node_feats = captured["node_feats"]
        if not isinstance(node_feats, torch.Tensor) or node_feats.ndim != 2:
            raise MaceLatentError("node_feats must be a rank-two Torch tensor")
        if node_feats.shape != (positions.shape[0], self.node_feats_dimension):
            raise MaceLatentError(
                f"node_feats must have exact shape [nodes,{self.node_feats_dimension}]"
            )
        if node_feats.dtype != positions.dtype or node_feats.device != positions.device:
            raise MaceLatentError("node_feats dtype/device changed unexpectedly")
        if not bool(torch.isfinite(node_feats).all().item()):
            raise MaceLatentError("node_feats contains non-finite values")
        ligand_mask = graph_batch["ligand_mask"]
        environment_mask = graph_batch["environment_mask"]
        if (
            not isinstance(ligand_mask, torch.Tensor)
            or ligand_mask.dtype != torch.bool
            or ligand_mask.shape != (positions.shape[0],)
            or not isinstance(environment_mask, torch.Tensor)
            or environment_mask.dtype != torch.bool
            or environment_mask.shape != ligand_mask.shape
            or not bool(torch.equal(environment_mask, ~ligand_mask))
        ):
            raise MaceLatentError("ligand/environment masks are invalid")
        latent = node_feats[ligand_mask, self.latent_slice]
        if latent.shape != (int(ligand_mask.sum().item()), self.latent_dimension):
            raise MaceLatentError("ligand latent slice produced an unexpected shape")
        if require_coordinate_grad and (not latent.requires_grad or latent.grad_fn is None):
            raise MaceLatentError("ligand latent was detached from the coordinate graph")
        return {
            "ligand_latent": latent,
            "support_diagnostics": dict(graph_batch["diagnostics"]),
            "tensor_metadata": {
                "source_key": "node_feats",
                "full_shape": list(node_feats.shape),
                "selected_slice": [self.latent_slice.start, self.latent_slice.stop],
                "selected_irreps": self.latent_irreps,
                "selected_product_layer_index": self.product_layer_index,
                "forward_stopped_after_selected_product_layer": (
                    self.product_layer_index is not None
                ),
                "dtype": str(latent.dtype),
                "device": str(latent.device),
                "mace_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "mace_trainable_parameter_count": sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
                "energy_fields_exposed": False,
            },
        }

    __call__ = forward


__all__ = [
    "EXP012_MODEL_PATH", "EXP012_MODEL_SHA256", "LATENT_IRREPS", "LATENT_SLICE",
    "MaceLatentBasisAdapter", "MaceLatentError", "NODE_FEATS_DIMENSION", "load_c0_report",
]
