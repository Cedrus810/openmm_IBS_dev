"""Read-only architecture inspection for an explicitly identified MACE model.

This module intentionally does not import Torch, MACE, or e3nn at import time.
The actual model is loaded only by :func:`inspect_mace_model_contract`.  C0 is
an identity/architecture gate: it does not execute a model forward, obtain
NumPy descriptors, inspect final energies, or perform fragment subtraction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping


REPORT_SCHEMA_VERSION = "exp012-mace-model-contract-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MaceContractError(RuntimeError):
    """Raised when a model cannot satisfy the explicit C0 contract."""


@dataclass(frozen=True)
class MaceModelContract:
    """Caller-supplied expected identity and architecture for one MACE model."""

    model_path: str
    expected_sha256: str
    expected_class: str
    expected_torch_version: str
    expected_mace_version: str
    expected_e3nn_version: str
    expected_interaction_layer_count: int
    expected_product_layer_count: int
    expected_r_max_angstrom: float
    expected_atomic_numbers: tuple[int, ...]
    expected_product_layer_index: int
    expected_product_layer_irreps: tuple[str, ...]
    expected_node_feats_dimension: int
    expected_invariant_slice_start: int
    expected_invariant_slice_stop: int
    expected_invariant_slice_irreps: str

    def __post_init__(self) -> None:
        if not self.model_path:
            raise MaceContractError("model_path must be explicit and non-empty")
        if not _SHA256_RE.fullmatch(self.expected_sha256):
            raise MaceContractError("expected_sha256 must be 64 lowercase hexadecimal characters")
        for name in (
            "expected_class",
            "expected_torch_version",
            "expected_mace_version",
            "expected_e3nn_version",
            "expected_invariant_slice_irreps",
        ):
            if not getattr(self, name):
                raise MaceContractError(f"{name} must be explicit and non-empty")
        if self.expected_interaction_layer_count <= 0 or self.expected_product_layer_count <= 0:
            raise MaceContractError("expected layer counts must be positive")
        if not math.isfinite(self.expected_r_max_angstrom) or self.expected_r_max_angstrom <= 0:
            raise MaceContractError("expected_r_max_angstrom must be finite and positive")
        if (
            not self.expected_atomic_numbers
            or any(isinstance(z, bool) or not isinstance(z, int) or z <= 0 for z in self.expected_atomic_numbers)
            or tuple(sorted(set(self.expected_atomic_numbers))) != self.expected_atomic_numbers
        ):
            raise MaceContractError(
                "expected_atomic_numbers must be an explicit sorted unique tuple of positive integers"
            )
        if not 0 <= self.expected_product_layer_index < self.expected_product_layer_count:
            raise MaceContractError("expected_product_layer_index is outside expected products")
        if (
            len(self.expected_product_layer_irreps) != self.expected_product_layer_count
            or any(not irreps for irreps in self.expected_product_layer_irreps)
        ):
            raise MaceContractError(
                "expected_product_layer_irreps must explicitly identify every product layer"
            )
        if self.expected_node_feats_dimension <= 0:
            raise MaceContractError("expected_node_feats_dimension must be positive")
        if not 0 <= self.expected_invariant_slice_start < self.expected_invariant_slice_stop:
            raise MaceContractError("expected invariant slice must be non-empty and forward ordered")
        if self.expected_invariant_slice_stop > self.expected_node_feats_dimension:
            raise MaceContractError("expected invariant slice exceeds node_feats dimension")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_report_sha256(report_without_hash: Mapping[str, Any]) -> str:
    """Hash a report using deterministic UTF-8 canonical JSON."""

    payload = json.dumps(
        report_without_hash,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _installed_versions() -> Mapping[str, str]:
    packages = {"torch": "torch", "mace": "mace-torch", "e3nn": "e3nn"}
    found = {}
    for report_name, distribution_name in packages.items():
        try:
            found[report_name] = version(distribution_name)
        except PackageNotFoundError as exc:
            raise MaceContractError(
                f"required distribution is unavailable: {distribution_name}"
            ) from exc
    return found


def _load_torch_model(path: Path):
    try:
        import torch
    except ImportError as exc:
        raise MaceContractError("Torch is unavailable; real MACE inspection cannot run") from exc
    try:
        # A serialized MACE Module requires object loading.  Model identity is
        # verified before this call, so only the explicitly trusted SHA is read.
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise MaceContractError(f"failed to load identified MACE model: {exc}") from exc


def _scalar_float(value: Any, name: str) -> float:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        result = float(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise MaceContractError(f"model {name} must be a scalar") from exc
    if not math.isfinite(result):
        raise MaceContractError(f"model {name} must be finite")
    return result


def _atomic_number_tuple(value: Any) -> tuple[int, ...]:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        numbers = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise MaceContractError("model atomic_numbers must be an integer sequence") from exc
    if not numbers or any(number <= 0 for number in numbers):
        raise MaceContractError("model atomic_numbers must be non-empty and positive")
    return numbers


def _qualified_class_name(model: Any) -> str:
    cls = type(model)
    return f"{cls.__module__}.{cls.__qualname__}"


def _irreps_dimension(irreps: Any) -> int:
    """Return an e3nn irreps dimension without importing e3nn."""

    dimension = getattr(irreps, "dim", None)
    if dimension is not None:
        try:
            result = int(dimension)
        except (TypeError, ValueError) as exc:
            raise MaceContractError("product irreps dimension is not an integer") from exc
        if result <= 0:
            raise MaceContractError("product irreps dimension must be positive")
        return result
    compact = str(irreps).replace(" ", "")
    result = 0
    for term in compact.split("+"):
        match = re.fullmatch(r"(?:(\d+)x)?(\d+)[eo]", term)
        if match is None:
            raise MaceContractError(f"cannot determine product irreps dimension: {irreps}")
        multiplicity = int(match.group(1) or 1)
        angular_momentum = int(match.group(2))
        result += multiplicity * (2 * angular_momentum + 1)
    if result <= 0:
        raise MaceContractError("product irreps dimension must be positive")
    return result


def inspect_mace_model_contract(
    contract: MaceModelContract,
    *,
    model_loader: Callable[[Path], Any] | None = None,
    version_provider: Callable[[], Mapping[str, str]] | None = None,
) -> Mapping[str, Any]:
    """Inspect model identity and architecture, failing closed on any mismatch."""

    path = Path(contract.model_path).expanduser().resolve()
    if not path.is_file():
        raise MaceContractError(f"model file does not exist: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != contract.expected_sha256:
        raise MaceContractError(
            f"model sha256 mismatch: expected {contract.expected_sha256}, observed {actual_sha256}"
        )

    versions = dict((version_provider or _installed_versions)())
    if set(versions) != {"torch", "mace", "e3nn"}:
        raise MaceContractError("version provider must return exactly torch, mace, and e3nn")
    model = (model_loader or _load_torch_model)(path)

    try:
        interactions = model.interactions
        products = model.products
        products[contract.expected_product_layer_index]
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise MaceContractError(
            "model lacks interactions/products or selected product.linear.irreps_out"
        ) from exc

    product_layer_outputs = []
    concatenated_offset = 0
    try:
        for index, product in enumerate(products):
            irreps_object = product.linear.irreps_out
            layer_dimension = _irreps_dimension(irreps_object)
            product_layer_outputs.append(
                {
                    "index": index,
                    "irreps": str(irreps_object),
                    "dimension": layer_dimension,
                    "concatenated_start": concatenated_offset,
                    "concatenated_stop": concatenated_offset + layer_dimension,
                }
            )
            concatenated_offset += layer_dimension
    except AttributeError as exc:
        raise MaceContractError("every product layer must expose linear.irreps_out") from exc
    selected_output = product_layer_outputs[contract.expected_product_layer_index]

    observed = {
        "model_path": str(path),
        "model_sha256": actual_sha256,
        "model_class": _qualified_class_name(model),
        "versions": {
            "torch": str(versions["torch"]),
            "mace": str(versions["mace"]),
            "e3nn": str(versions["e3nn"]),
        },
        "interaction_layer_count": len(interactions),
        "product_layer_count": len(products),
        "r_max_angstrom": _scalar_float(model.r_max, "r_max"),
        "atomic_numbers": list(_atomic_number_tuple(model.atomic_numbers)),
        "node_feats_contract": {
            "tensor_key": "node_feats",
            "source": "torch.cat(product_layer_node_feats, dim=-1)",
            "product_layer_outputs": product_layer_outputs,
            "concatenated_dimension": concatenated_offset,
            "selected_product_layer_index": contract.expected_product_layer_index,
            "recommended_invariant_slice": {
                "start": selected_output["concatenated_start"],
                "stop": selected_output["concatenated_stop"],
                "irreps": selected_output["irreps"],
            },
        },
    }
    expected = {
        "model_path": str(path),
        "model_sha256": contract.expected_sha256,
        "model_class": contract.expected_class,
        "versions": {
            "torch": contract.expected_torch_version,
            "mace": contract.expected_mace_version,
            "e3nn": contract.expected_e3nn_version,
        },
        "interaction_layer_count": contract.expected_interaction_layer_count,
        "product_layer_count": contract.expected_product_layer_count,
        "r_max_angstrom": contract.expected_r_max_angstrom,
        "atomic_numbers": list(contract.expected_atomic_numbers),
        "node_feats_contract": {
            "tensor_key": "node_feats",
            "source": "torch.cat(product_layer_node_feats, dim=-1)",
            "product_layer_outputs": [
                {
                    "index": index,
                    "irreps": irreps,
                    "dimension": _irreps_dimension(irreps),
                    "concatenated_start": sum(
                        _irreps_dimension(previous)
                        for previous in contract.expected_product_layer_irreps[:index]
                    ),
                    "concatenated_stop": sum(
                        _irreps_dimension(previous)
                        for previous in contract.expected_product_layer_irreps[: index + 1]
                    ),
                }
                for index, irreps in enumerate(contract.expected_product_layer_irreps)
            ],
            "concatenated_dimension": contract.expected_node_feats_dimension,
            "selected_product_layer_index": contract.expected_product_layer_index,
            "recommended_invariant_slice": {
                "start": contract.expected_invariant_slice_start,
                "stop": contract.expected_invariant_slice_stop,
                "irreps": contract.expected_invariant_slice_irreps,
            },
        },
    }

    mismatches = []
    for field in (
        "model_class",
        "versions",
        "interaction_layer_count",
        "product_layer_count",
        "atomic_numbers",
        "node_feats_contract",
    ):
        if observed[field] != expected[field]:
            mismatches.append(field)
    if not math.isclose(
        observed["r_max_angstrom"], expected["r_max_angstrom"], rel_tol=0.0, abs_tol=1.0e-12
    ):
        mismatches.append("r_max_angstrom")
    if mismatches:
        raise MaceContractError("MACE contract mismatch: " + ", ".join(mismatches))

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "PASSED_READ_ONLY_ARCHITECTURE_INSPECTION",
        "expected": expected,
        "observed": observed,
        "policy": {
            "model_forward_executed": False,
            "latent_forward_executed": False,
            "numpy_descriptor_path_allowed": False,
            "final_energy_allowed": False,
            "fragment_subtraction_allowed": False,
            "scientific_qualification": False,
        },
    }
    return {**report, "report_sha256": canonical_report_sha256(report)}


__all__ = [
    "MaceContractError",
    "MaceModelContract",
    "REPORT_SCHEMA_VERSION",
    "canonical_report_sha256",
    "inspect_mace_model_contract",
]
