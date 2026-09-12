"""EXP-019 reference models for the ligand-anchored SoftLift ladder.

The reference path consumes an already validated :class:`PackedSoftLiftBatch`.
It deliberately does not build a neighbor list, apply ``A_k``/``beta``, or
convert OpenMM units.  Those boundaries keep the offline scalar basis and the
Outer-:math:`\\lambda` Hamiltonian accounting separate.

R1 is the primary model.  R2 and R3 are included as explicit, finite
ablation-capable reference models; neither is advertised as a native OpenMM
candidate.  All three are rotation/translation invariant because the model
uses distances (and, for R3, invariant contractions of normalized
displacements) only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Literal, Sequence


class SoftLiftError(ValueError):
    """Raised when the EXP-019 model or packed-batch contract is violated."""


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise SoftLiftError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SoftLiftError(f"{name} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise SoftLiftError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class SoftLiftConfig:
    """Frozen tensor and support-domain contract for one SoftLift rung."""

    schema_version: str
    rung: Literal["R1", "R2", "R3"]
    type_vocabulary: tuple[int, ...]
    n_ligand_atoms: int
    n_radial_basis: int
    n_channels: int
    pair_dim: int
    context_dim: int
    inner_cutoff_angstrom: float
    outer_cutoff_angstrom: float
    b_max_reduced: float
    max_environment_atoms: int
    max_edges: int
    max_neighbors_per_ligand: int
    no_contact_output: Literal["exact_zero"]
    protocol_sha256: str

    def __post_init__(self) -> None:
        if not self.schema_version or not isinstance(self.schema_version, str):
            raise SoftLiftError("schema_version must be a non-empty string")
        if self.rung not in {"R1", "R2", "R3"}:
            raise SoftLiftError("rung must be R1, R2, or R3")
        if (
            not isinstance(self.type_vocabulary, tuple)
            or not self.type_vocabulary
            or any(isinstance(value, bool) or not isinstance(value, int) for value in self.type_vocabulary)
            or len(set(self.type_vocabulary)) != len(self.type_vocabulary)
        ):
            raise SoftLiftError("type_vocabulary must be a non-empty tuple of unique integers")
        if any(value <= 0 for value in self.type_vocabulary):
            raise SoftLiftError("type_vocabulary entries must be positive atomic numbers")
        for name in (
            "n_ligand_atoms", "n_radial_basis", "n_channels", "max_environment_atoms",
            "max_edges", "max_neighbors_per_ligand",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SoftLiftError(f"{name} must be a positive integer")
        for name in ("pair_dim", "context_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SoftLiftError(f"{name} must be a non-negative integer")
        inner = _finite_positive(self.inner_cutoff_angstrom, "inner_cutoff_angstrom")
        outer = _finite_positive(self.outer_cutoff_angstrom, "outer_cutoff_angstrom")
        if not inner < outer:
            raise SoftLiftError("cutoffs must satisfy 0 < inner < outer")
        _finite_positive(self.b_max_reduced, "b_max_reduced")
        if self.no_contact_output != "exact_zero":
            raise SoftLiftError("no_contact_output must be exact_zero")
        if not isinstance(self.protocol_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.protocol_sha256):
            raise SoftLiftError("protocol_sha256 must be a lowercase SHA-256 digest")
        if self.rung == "R1" and self.n_channels != 1:
            raise SoftLiftError("R1 has exactly one density channel")
        if self.rung in {"R2", "R3"} and (self.n_channels != 4 or self.pair_dim != 8 or self.context_dim != 8):
            raise SoftLiftError("R2/R3 primary contract is n_channels=4, pair_dim=context_dim=8")


@dataclass(frozen=True)
class PackedSoftLiftBatch:
    """Validated packed ragged edges for a batch of frames.

    ``edge_frame`` and ``edge_ligand_local`` are the only receiver identity
    used by the models.  Global topology indices must be resolved before this
    object is built; this prevents accidental ``index_add_`` into a global
    ligand index space.
    """

    ligand_type_index: Any
    edge_frame: Any
    edge_ligand_local: Any
    edge_ligand_type: Any
    edge_environment_type: Any
    edge_distance_angstrom: Any
    edge_displacement_angstrom: Any | None = None
    edge_unit_shift: Any | None = None

    def __post_init__(self) -> None:
        import torch

        if not isinstance(self.ligand_type_index, torch.Tensor) or self.ligand_type_index.ndim != 2:
            raise SoftLiftError("ligand_type_index must have shape (frames, ligand_atoms)")
        if self.ligand_type_index.dtype == torch.bool or self.ligand_type_index.is_floating_point():
            raise SoftLiftError("ligand_type_index must be integer")
        if self.ligand_type_index.shape[0] <= 0 or self.ligand_type_index.shape[1] <= 0:
            raise SoftLiftError("packed batch must contain at least one frame and ligand atom")
        edge_values = (
            self.edge_frame, self.edge_ligand_local, self.edge_ligand_type,
            self.edge_environment_type,
        )
        for name, value in zip(
            ("edge_frame", "edge_ligand_local", "edge_ligand_type", "edge_environment_type"), edge_values
        ):
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                raise SoftLiftError(f"{name} must be one-dimensional")
            if value.device != self.ligand_type_index.device:
                raise SoftLiftError(f"{name} and ligand_type_index must share a device")
            if value.dtype == torch.bool or value.is_floating_point():
                raise SoftLiftError(f"{name} must be integer")
        distance = self.edge_distance_angstrom
        if not isinstance(distance, torch.Tensor) or distance.ndim != 1 or not distance.is_floating_point():
            raise SoftLiftError("edge_distance_angstrom must be a floating one-dimensional tensor")
        if distance.device != self.ligand_type_index.device:
            raise SoftLiftError("edge_distance_angstrom and ligand_type_index must share a device")
        edge_count = int(distance.shape[0])
        if any(int(value.shape[0]) != edge_count for value in edge_values):
            raise SoftLiftError("all packed edge arrays must have equal length")
        if not bool(torch.isfinite(distance).all().item()) or bool((distance < 0).any().item()):
            raise SoftLiftError("edge distances must be finite and non-negative")
        if edge_count:
            if bool((self.edge_frame < 0).any().item()) or bool((self.edge_frame >= self.ligand_type_index.shape[0]).any().item()):
                raise SoftLiftError("edge_frame is outside the packed frame range")
            if bool((self.edge_ligand_local < 0).any().item()) or bool((self.edge_ligand_local >= self.ligand_type_index.shape[1]).any().item()):
                raise SoftLiftError("edge_ligand_local is outside the ligand-local range")
            expected = self.ligand_type_index[self.edge_frame, self.edge_ligand_local]
            if not bool(torch.equal(expected, self.edge_ligand_type)):
                raise SoftLiftError("edge_ligand_type disagrees with ligand_type_index receiver identity")
        for name, value, shape_tail in (
            ("edge_displacement_angstrom", self.edge_displacement_angstrom, (3,)),
            ("edge_unit_shift", self.edge_unit_shift, (3,)),
        ):
            if value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.shape != (edge_count, *shape_tail):
                raise SoftLiftError(f"{name} must have shape ({edge_count}, 3)")
            if value.device != self.ligand_type_index.device:
                raise SoftLiftError(f"{name} and ligand_type_index must share a device")
            if name == "edge_unit_shift":
                if value.dtype == torch.bool or value.is_floating_point():
                    raise SoftLiftError("edge_unit_shift must be integer")
            elif not value.is_floating_point() or not bool(torch.isfinite(value).all().item()):
                raise SoftLiftError("edge_displacement_angstrom must be finite floating-point")

    @property
    def n_frames(self) -> int:
        return int(self.ligand_type_index.shape[0])

    @property
    def n_ligand_atoms(self) -> int:
        return int(self.ligand_type_index.shape[1])

    @property
    def n_edges(self) -> int:
        return int(self.edge_distance_angstrom.shape[0])


def pack_softlift_frames(frames: Sequence[dict[str, Any]]) -> PackedSoftLiftBatch:
    """Pack per-frame local edge dictionaries without padding or truncation."""

    import torch

    if not frames:
        raise SoftLiftError("frames must be non-empty")
    first_types = frames[0]["ligand_type_index"]
    if not isinstance(first_types, torch.Tensor) or first_types.ndim != 1:
        raise SoftLiftError("each frame must provide one-dimensional ligand_type_index")
    n_ligand = int(first_types.shape[0])
    ligand_types = []
    edge_frames = []
    edge_ligand_local = []
    edge_ligand_type = []
    edge_environment_type = []
    edge_distance = []
    edge_displacement = []
    edge_shifts = []
    have_displacement = all(frame.get("edge_displacement_angstrom") is not None for frame in frames)
    have_shifts = all(frame.get("edge_unit_shift") is not None for frame in frames)
    if any((frame.get("edge_displacement_angstrom") is not None) != have_displacement for frame in frames):
        raise SoftLiftError("edge_displacement_angstrom must be present for every frame or none")
    if any((frame.get("edge_unit_shift") is not None) != have_shifts for frame in frames):
        raise SoftLiftError("edge_unit_shift must be present for every frame or none")
    for frame_index, frame in enumerate(frames):
        types = frame["ligand_type_index"]
        if not isinstance(types, torch.Tensor) or types.shape != (n_ligand,):
            raise SoftLiftError("all frames must share the ligand atom count")
        ligand_types.append(types)
        distance = frame["edge_distance_angstrom"]
        local = frame["edge_ligand_local"]
        env_type = frame["edge_environment_type"]
        if not all(isinstance(value, torch.Tensor) and value.ndim == 1 for value in (distance, local, env_type)):
            raise SoftLiftError("frame edge fields must be one-dimensional tensors")
        if not (distance.shape == local.shape == env_type.shape):
            raise SoftLiftError("frame edge fields must have equal length")
        count = int(distance.shape[0])
        if bool((local < 0).any().item()) or bool((local >= n_ligand).any().item()):
            raise SoftLiftError("frame edge_ligand_local is out of range")
        edge_frames.append(torch.full((count,), frame_index, dtype=torch.int64, device=distance.device))
        edge_ligand_local.append(local.to(torch.int64))
        edge_ligand_type.append(types.to(torch.int64)[local.to(torch.int64)])
        edge_environment_type.append(env_type.to(torch.int64))
        edge_distance.append(distance)
        if have_displacement:
            edge_displacement.append(frame["edge_displacement_angstrom"])
        if have_shifts:
            edge_shifts.append(frame["edge_unit_shift"].to(torch.int64))
    kwargs: dict[str, Any] = {
        "ligand_type_index": torch.stack(ligand_types).to(torch.int64),
        "edge_frame": torch.cat(edge_frames) if edge_frames else torch.empty(0, dtype=torch.int64),
        "edge_ligand_local": torch.cat(edge_ligand_local) if edge_ligand_local else torch.empty(0, dtype=torch.int64),
        "edge_ligand_type": torch.cat(edge_ligand_type) if edge_ligand_type else torch.empty(0, dtype=torch.int64),
        "edge_environment_type": torch.cat(edge_environment_type) if edge_environment_type else torch.empty(0, dtype=torch.int64),
        "edge_distance_angstrom": torch.cat(edge_distance) if edge_distance else torch.empty(0),
    }
    if have_displacement:
        kwargs["edge_displacement_angstrom"] = torch.cat(edge_displacement)
    if have_shifts:
        kwargs["edge_unit_shift"] = torch.cat(edge_shifts)
    return PackedSoftLiftBatch(**kwargs)


def _quintic_c2(distance, inner: float, outer: float):
    import torch

    x = (distance - inner) / (outer - inner)
    transition = 1.0 - 10.0 * x.pow(3) + 15.0 * x.pow(4) - 6.0 * x.pow(5)
    return torch.where(
        distance <= inner,
        torch.ones_like(distance),
        torch.where(distance >= outer, torch.zeros_like(distance), transition),
    )


def _validate_model_batch(batch: PackedSoftLiftBatch, config: SoftLiftConfig) -> None:
    import torch

    if batch.n_ligand_atoms != config.n_ligand_atoms:
        raise SoftLiftError("packed ligand atom count differs from model config")
    if batch.n_edges:
        frame_edge_counts = torch.bincount(batch.edge_frame, minlength=batch.n_frames)
        if int(frame_edge_counts.max().item()) > config.max_edges:
            raise SoftLiftError("a packed frame exceeds the hard EXP-019 edge ceiling")
        receiver = batch.edge_frame * batch.n_ligand_atoms + batch.edge_ligand_local
        receiver_counts = torch.bincount(
            receiver, minlength=batch.n_frames * batch.n_ligand_atoms
        )
        if int(receiver_counts.max().item()) > config.max_neighbors_per_ligand:
            raise SoftLiftError("a packed ligand atom exceeds the hard EXP-019 neighbor ceiling")
    if batch.n_edges:
        for name, values in (("ligand", batch.edge_ligand_type), ("environment", batch.edge_environment_type)):
            if bool((values < 0).any().item()) or bool((values >= len(config.type_vocabulary)).any().item()):
                raise SoftLiftError(f"edge {name} type index is outside the model vocabulary")
        if bool((batch.edge_distance_angstrom >= config.outer_cutoff_angstrom).any().item()):
            raise SoftLiftError("packed edges must use strict distance < outer_cutoff membership")
    if bool((batch.ligand_type_index < 0).any().item()) or bool((batch.ligand_type_index >= len(config.type_vocabulary)).any().item()):
        raise SoftLiftError("ligand type index is outside the model vocabulary")
    if config.rung == "R3" and batch.edge_displacement_angstrom is None:
        raise SoftLiftError("R3 requires edge_displacement_angstrom")
    if batch.edge_displacement_angstrom is not None:
        norm = torch.linalg.vector_norm(batch.edge_displacement_angstrom, dim=-1)
        if not torch.allclose(norm, batch.edge_distance_angstrom, rtol=0.0, atol=1e-8):
            raise SoftLiftError("edge displacement norm does not match edge distance")


def _typed_readout(readouts, features, type_index):
    import torch

    outputs = torch.stack([network(features).squeeze(-1) for network in readouts], dim=-1)
    selector = torch.nn.functional.one_hot(type_index, num_classes=len(readouts)).to(features.dtype)
    return (outputs * selector).sum(dim=-1)


def _mlp(nn, input_dim: int, hidden_dim: int, output_dim: int):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))


class SoftLiftR1:
    """Marker base used for type check documentation; concrete class is built lazily."""


class SoftLiftR2:
    """Marker base for the context-conditioned reference class."""


class SoftLiftR3:
    """Marker base for the normalized-moment reference class."""


_MODEL_CACHE: dict[str, type] = {}


def _build_model_classes() -> tuple[type, type, type]:
    import torch
    import torch.nn as nn

    if _MODEL_CACHE:
        return _MODEL_CACHE["R1"], _MODEL_CACHE["R2"], _MODEL_CACHE["R3"]

    class R1Model(nn.Module, SoftLiftR1):
        def __init__(self, config: SoftLiftConfig, hidden_rho: int = 16):
            super().__init__()
            self.config = config
            self.type_count = len(config.type_vocabulary)
            self.radial_centers = nn.Parameter(
                torch.linspace(0.0, config.outer_cutoff_angstrom, config.n_radial_basis), requires_grad=False
            )
            self.register_buffer("radial_width", torch.tensor(config.outer_cutoff_angstrom / max(config.n_radial_basis - 1, 1)))
            self.pair_weight = nn.Parameter(torch.zeros(self.type_count, self.type_count, config.n_radial_basis))
            self.rho = nn.ModuleList([nn.Sequential(
                nn.Linear(1, hidden_rho), nn.SiLU(), nn.Linear(hidden_rho, hidden_rho), nn.SiLU(), nn.Linear(hidden_rho, 1)
            ) for _ in range(self.type_count)])

        def radial_basis(self, distance):
            diff = distance[:, None] - self.radial_centers[None, :]
            return torch.exp(-0.5 * (diff / self.radial_width).square())

        def forward(self, batch: PackedSoftLiftBatch):
            _validate_model_batch(batch, self.config)
            receiver = batch.edge_frame * self.config.n_ligand_atoms + batch.edge_ligand_local
            q_flat = batch.ligand_type_index.new_zeros((batch.n_frames * batch.n_ligand_atoms,), dtype=batch.edge_distance_angstrom.dtype)
            if batch.n_edges:
                radial = self.radial_basis(batch.edge_distance_angstrom)
                envelope = _quintic_c2(batch.edge_distance_angstrom, self.config.inner_cutoff_angstrom, self.config.outer_cutoff_angstrom)
                weights = self.pair_weight[batch.edge_ligand_type, batch.edge_environment_type]
                edge_q = envelope * (weights * radial).sum(dim=-1)
                q_flat = q_flat.index_add(0, receiver, edge_q)
            q = q_flat.reshape(batch.n_frames, self.config.n_ligand_atoms)
            zeros = torch.zeros_like(q)
            type_index = batch.ligand_type_index.reshape(-1)
            rho_q = _typed_readout(self.rho, q.reshape(-1, 1), type_index).reshape_as(q)
            rho_zero = _typed_readout(self.rho, zeros.reshape(-1, 1), type_index).reshape_as(q)
            per_ligand = rho_q - rho_zero
            raw = per_ligand.sum(dim=-1)
            return self.config.b_max_reduced * torch.tanh(raw / self.config.b_max_reduced)

    class ContextModel(nn.Module):
        def __init__(self, config: SoftLiftConfig, include_moments: bool, context_mode: str = "normal"):
            super().__init__()
            if context_mode not in {"normal", "context_zero", "anchor_shuffle"}:
                raise SoftLiftError("context_mode must be normal, context_zero, or anchor_shuffle")
            self.config = config
            self.type_count = len(config.type_vocabulary)
            self.include_moments = include_moments
            self.context_mode = context_mode
            pair_input_dim = self.type_count * self.type_count + config.n_radial_basis
            self.pair_encoder = _mlp(nn, pair_input_dim, config.pair_dim, config.pair_dim)
            self.context_encoder = nn.Sequential(nn.Linear(config.pair_dim, config.context_dim), nn.SiLU(), nn.Linear(config.context_dim, config.context_dim))
            self.gate_mlp = _mlp(nn, config.pair_dim + config.context_dim, config.pair_dim, config.n_channels)
            self.value_mlp = _mlp(nn, config.pair_dim, config.pair_dim, config.n_channels * config.pair_dim)
            if include_moments:
                self.moment_mlp = _mlp(nn, config.pair_dim, config.pair_dim, config.n_channels)
            feature_dim = config.n_channels * config.pair_dim + config.n_channels
            if include_moments:
                feature_dim += config.n_channels * 3
            self.readout = nn.ModuleList([
                _mlp(nn, feature_dim, 16, 1) for _ in range(self.type_count)
            ])
            self.radial_centers = nn.Parameter(
                torch.linspace(0.0, config.outer_cutoff_angstrom, config.n_radial_basis), requires_grad=False
            )
            self.register_buffer("radial_width", torch.tensor(config.outer_cutoff_angstrom / max(config.n_radial_basis - 1, 1)))

        def radial_basis(self, distance):
            diff = distance[:, None] - self.radial_centers[None, :]
            return torch.exp(-0.5 * (diff / self.radial_width).square())

        def _pair_features(self, batch):
            if batch.n_edges == 0:
                return batch.edge_distance_angstrom.new_empty((0, self.type_count * self.type_count + self.config.n_radial_basis))
            radial = self.radial_basis(batch.edge_distance_angstrom)
            pair_index = batch.edge_ligand_type * self.type_count + batch.edge_environment_type
            one_hot = torch.nn.functional.one_hot(pair_index, num_classes=self.type_count * self.type_count).to(radial.dtype)
            return torch.cat((one_hot, radial), dim=-1)

        def _context(self, batch, pair):
            receiver = batch.edge_frame * self.config.n_ligand_atoms + batch.edge_ligand_local
            total = batch.n_frames * self.config.n_ligand_atoms
            context_num = batch.edge_distance_angstrom.new_zeros((total, self.config.context_dim))
            context_den = batch.edge_distance_angstrom.new_zeros((total,))
            if batch.n_edges:
                envelope = _quintic_c2(batch.edge_distance_angstrom, self.config.inner_cutoff_angstrom, self.config.outer_cutoff_angstrom)
                pair_context = self.context_encoder(self.pair_encoder(pair))
                context_num = context_num.index_add(0, receiver, envelope[:, None] * pair_context)
                context_den = context_den.index_add(0, receiver, envelope)
            context = context_num / (1.0 + context_den[:, None])
            if self.context_mode == "context_zero":
                context = torch.zeros_like(context)
            elif self.context_mode == "anchor_shuffle":
                context = context.reshape(batch.n_frames, self.config.n_ligand_atoms, self.config.context_dim)
                context = torch.roll(context, shifts=1, dims=1).reshape(-1, self.config.context_dim)
            return context, receiver

        def _aggregate(self, batch):
            pair = self._pair_features(batch)
            context, receiver = self._context(batch, pair)
            total = batch.n_frames * self.config.n_ligand_atoms
            n_eff = batch.edge_distance_angstrom.new_zeros((total, self.config.n_channels))
            u_num = batch.edge_distance_angstrom.new_zeros((total, self.config.n_channels, self.config.pair_dim))
            envelope = _quintic_c2(batch.edge_distance_angstrom, self.config.inner_cutoff_angstrom, self.config.outer_cutoff_angstrom)
            if batch.n_edges:
                pair_encoded = self.pair_encoder(pair)
                context_edge = context[receiver]
                gate = envelope[:, None] * torch.sigmoid(self.gate_mlp(torch.cat((pair_encoded, context_edge), dim=-1)))
                values = self.value_mlp(pair_encoded).reshape(batch.n_edges, self.config.n_channels, self.config.pair_dim)
                n_eff = n_eff.index_add(0, receiver, gate)
                u_num = u_num.index_add(0, receiver, gate[:, :, None] * values)
            u = u_num / (1.0 + n_eff[:, :, None])
            return u, n_eff, receiver, envelope, pair

        def _moments(self, batch, receiver, envelope, pair):
            if batch.edge_displacement_angstrom is None:
                raise SoftLiftError("R3 requires edge_displacement_angstrom")
            total = batch.n_frames * self.config.n_ligand_atoms
            pair_encoded = self.pair_encoder(pair)
            context, _ = self._context(batch, pair)
            if batch.n_edges:
                context_edge = context[receiver]
                gate = envelope[:, None] * torch.sigmoid(self.gate_mlp(torch.cat((pair_encoded, context_edge), dim=-1)))
                f = self.moment_mlp(pair_encoded)
                weighted = gate * f
                dbar = batch.edge_displacement_angstrom / self.config.outer_cutoff_angstrom
                m1_num = batch.edge_distance_angstrom.new_zeros((total, self.config.n_channels, 3))
                m2_num = batch.edge_distance_angstrom.new_zeros((total, self.config.n_channels, 3, 3))
                m1_num = m1_num.index_add(0, receiver, weighted[:, :, None] * dbar[:, None, :])
                m2_edges = weighted[:, :, None, None] * dbar[:, None, :, None] * dbar[:, None, None, :]
                m2_num = m2_num.index_add(0, receiver, m2_edges)
                n_eff = gate.new_zeros((total, self.config.n_channels)).index_add(0, receiver, gate)
            else:
                n_eff = batch.edge_distance_angstrom.new_zeros((total, self.config.n_channels))
                m1_num = batch.edge_distance_angstrom.new_zeros((total, self.config.n_channels, 3))
                m2_num = batch.edge_distance_angstrom.new_zeros((total, self.config.n_channels, 3, 3))
            denom = 1.0 + n_eff
            m1 = m1_num / denom[:, :, None]
            m2 = m2_num / denom[:, :, None, None]
            i1 = m1.square().sum(dim=-1)
            i2 = torch.diagonal(m2, dim1=-2, dim2=-1).sum(dim=-1)
            i3 = torch.diagonal(torch.matmul(m2, m2), dim1=-2, dim2=-1).sum(dim=-1)
            return torch.cat((i1, i2, i3), dim=-1), n_eff

        def forward(self, batch: PackedSoftLiftBatch):
            _validate_model_batch(batch, self.config)
            u, n_eff, receiver, envelope, pair = self._aggregate(batch)
            features = torch.cat((u.reshape(-1, self.config.n_channels * self.config.pair_dim), torch.log1p(n_eff)), dim=-1)
            if self.include_moments:
                invariants, n_eff = self._moments(batch, receiver, envelope, pair)
                features = torch.cat((features, invariants), dim=-1)
            type_index = batch.ligand_type_index.reshape(-1)
            zeros = torch.zeros_like(features)
            values = _typed_readout(self.readout, features, type_index)
            zero_values = _typed_readout(self.readout, zeros, type_index)
            per_ligand = values - zero_values
            raw = per_ligand.reshape(batch.n_frames, self.config.n_ligand_atoms).sum(dim=-1)
            return self.config.b_max_reduced * torch.tanh(raw / self.config.b_max_reduced)

    _MODEL_CACHE.update({"R1": R1Model, "R2": ContextModel, "R3": ContextModel})
    return R1Model, ContextModel, ContextModel


def build_softlift_model(config: SoftLiftConfig, *, context_mode: str = "normal"):
    """Build the reference model selected by a frozen config."""

    classes = _build_model_classes()
    if config.rung == "R1":
        if context_mode != "normal":
            raise SoftLiftError("R1 does not have a context control mode")
        return classes[0](config)
    return classes[1](config, include_moments=config.rung == "R3", context_mode=context_mode)


def count_trainable_parameters(model: Any) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)


#: 出厂 R1 的容量常量与 b_max 都是按这个配体尺寸定的（Atenolol）。
R1_REFERENCE_LIGAND_ATOM_COUNT = 41


def primary_r1_config(protocol_sha256: str) -> SoftLiftConfig:
    """Return the preregistered R1 configuration."""

    return SoftLiftConfig(
        schema_version="exp019-softlift-v1",
        rung="R1",
        type_vocabulary=(1, 6, 7, 8, 11, 16, 17),
        n_ligand_atoms=41,
        n_radial_basis=16,
        n_channels=1,
        pair_dim=0,
        context_dim=0,
        inner_cutoff_angstrom=4.0,
        outer_cutoff_angstrom=5.0,
        b_max_reduced=10.0,
        max_environment_atoms=320,
        max_edges=2048,
        max_neighbors_per_ligand=80,
        no_contact_output="exact_zero",
        protocol_sha256=protocol_sha256,
    )


def derived_r1_config(
    protocol_sha256: str,
    *,
    type_vocabulary,
    n_ligand_atoms: int,
    max_environment_atoms: int,
    max_edges: int,
    max_neighbors_per_ligand: int,
) -> SoftLiftConfig:
    """`primary_r1_config` 的参数化版本：同一套 R1 架构，词表/配体原子数/容量按体系给。

    出厂那份是 41 原子的 Atenolol 常量，换配体一定对不上（`_config` 的 R1 分支会
    直接抛）。这里只把"跟体系有关"的四项外提，架构常量（n_channels=1、pair_dim=0、
    4/5 Å 内外截断、exact_zero）逐字保持不变，因此传 Atenolol 那组值进来得到的
    config 与 `primary_r1_config` 逐字段相同。

    `b_max_reduced` 是唯一按尺寸缩放的量：出厂 10.0 是按 41 原子定的，而输出头是
    对原子求和之后再套全局 tanh（EXP-033 §1 第 3 条），大配体用 10.0 会直接进死区。
    小配体不缩（保持 >= 出厂值），让 41 原子这条路径逐位不变。
    """

    vocabulary = tuple(int(value) for value in type_vocabulary)
    if not vocabulary or len(set(vocabulary)) != len(vocabulary):
        raise SoftLiftError("type_vocabulary 必须非空且不含重复元素")
    if int(n_ligand_atoms) <= 0:
        raise SoftLiftError("n_ligand_atoms 必须为正")
    scale = max(1.0, float(n_ligand_atoms) / float(R1_REFERENCE_LIGAND_ATOM_COUNT))
    return SoftLiftConfig(
        schema_version="exp019-softlift-v1",
        rung="R1",
        type_vocabulary=tuple(sorted(vocabulary)),
        n_ligand_atoms=int(n_ligand_atoms),
        n_radial_basis=16,
        n_channels=1,
        pair_dim=0,
        context_dim=0,
        inner_cutoff_angstrom=4.0,
        outer_cutoff_angstrom=5.0,
        b_max_reduced=10.0 * scale,
        max_environment_atoms=int(max_environment_atoms),
        max_edges=int(max_edges),
        max_neighbors_per_ligand=int(max_neighbors_per_ligand),
        no_contact_output="exact_zero",
        protocol_sha256=protocol_sha256,
    )


__all__ = [
    "R1_REFERENCE_LIGAND_ATOM_COUNT",
    "derived_r1_config",
    "PackedSoftLiftBatch",
    "SoftLiftConfig",
    "SoftLiftError",
    "SoftLiftR1",
    "SoftLiftR2",
    "SoftLiftR3",
    "build_softlift_model",
    "count_trainable_parameters",
    "pack_softlift_frames",
    "primary_r1_config",
]
