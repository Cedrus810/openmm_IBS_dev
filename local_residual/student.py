"""DEC-037 (d0-2): the minimal `LocalResidualStudent` architecture candidate.

This is the *first* candidate frozen by the DEC-030(d0) design contract, not a
full tensor-equivariant MACE-style student: a rotation-invariant scalar model
built from typed atom embeddings, a smooth ligand-environment radial/contact
feature, at most a couple of lightweight interaction blocks, ligand-only
invariant pooling, and a bounded scalar head. Only if this simplest candidate
fails does the design escalate to something more expressive (DEC-037).

Everything here operates on interatomic *distances* (never raw displacement
vectors), so the output is rotation- and translation-invariant by
construction; its gradient w.r.t. Cartesian coordinates is therefore already a
proper equivariant force with no additional machinery required.

This module does not itself decide which environment atoms are "in range" of
the ligand this frame -- that is `local_residual.geometry.ligand_environment_cross_edges`
(DEC-038/039, already validated against the teacher's canonical membership and
against direct CUDA float32 execution). `reindex_ligand_environment_edges`
below is the only new piece of plumbing: it turns that function's *global*
topology-index edge list into the compact local index space this model's
embedding lookups need, once per frame.

Units: distances are Angstrom (matching `local_residual.geometry`). The
model's scalar output is a dimensionless *reduced* quantity (no `beta`
multiplication anywhere in this module) -- it is designed to be used directly
as `basis_reduced` in `local_residual.loss.bidirectional_gap_variance_loss`,
the same convention the teacher-side linear readout (DEC-034/035/036) already
uses.

Following this package's existing convention (`geometry.py`, `loss.py`,
`mace_graph.py`, ...), torch is imported lazily inside functions, never at
module import time -- importing `local_residual.student` for its pure
`reindex_ligand_environment_edges` helper (e.g. from a non-torch caller) must
not require torch to be installed. The one exception is
`build_local_residual_student`, which obviously needs torch to build an
`nn.Module`; it defines the module class on first call and reuses the exact
same class object on every later call (not a fresh class per call), so
`isinstance`/pickling stay well-behaved across multiple models in one process.
"""

from __future__ import annotations

from typing import Any, Sequence

from .schema import Exp012ProtocolError


class LocalResidualStudentError(Exp012ProtocolError):
    """Raised when a `LocalResidualStudent` input violates its explicit contract."""


def _require_int_sequence(values: Any, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise LocalResidualStudentError(f"{name} must be a sequence of integers")
    if len(values) == 0:
        raise LocalResidualStudentError(f"{name} must be non-empty")
    result = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int):
            raise LocalResidualStudentError(f"{name} entries must be plain integers")
        result.append(item)
    return tuple(result)


def reindex_ligand_environment_edges(
    ligand_topology_indices: Sequence[int],
    edge_ligand_topology: Any,
    edge_environment_topology: Any,
):
    """Map a frame's global-topology-index edges to this model's local index space.

    ``ligand_topology_indices`` fixes the ligand's local ordering (index ``i``
    in every returned/consumed tensor always means this same topology atom,
    across every frame and every call). ``edge_ligand_topology``/
    ``edge_environment_topology`` are the two rows of
    ``ligand_environment_cross_edges(...)["edge_index"]`` for one frame (ligand
    row first, matching that function's own "ligand-major" contract).

    Returns a dict with ``environment_topology_indices`` (the sorted, unique
    environment atoms this frame actually has an edge to -- this frame's local
    environment index space, smallest first) and ``edge_ligand_local``/
    ``edge_environment_local`` (both shape ``(n_edges,)``, indexing into the
    ligand ordering given and into ``environment_topology_indices``
    respectively). Grouping repeated environment atoms into one local index
    (rather than embedding the same atom once per edge) is exactly the
    ligand-only, environment-shared aggregation the interaction block expects.
    """

    import torch

    ligand_order = _require_int_sequence(ligand_topology_indices, "ligand_topology_indices")
    if len(set(ligand_order)) != len(ligand_order):
        raise LocalResidualStudentError("ligand_topology_indices must not contain duplicates")
    ligand_topology_to_local = {topology_index: local for local, topology_index in enumerate(ligand_order)}

    for name, value in (
        ("edge_ligand_topology", edge_ligand_topology),
        ("edge_environment_topology", edge_environment_topology),
    ):
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise LocalResidualStudentError(f"{name} must be a one-dimensional Torch tensor")
    if edge_ligand_topology.shape != edge_environment_topology.shape:
        raise LocalResidualStudentError(
            "edge_ligand_topology and edge_environment_topology must have equal length"
        )

    edge_count = int(edge_ligand_topology.shape[0])
    if edge_count == 0:
        return {
            "environment_topology_indices": torch.empty((0,), dtype=torch.int64),
            "edge_ligand_local": torch.empty((0,), dtype=torch.int64),
            "edge_environment_local": torch.empty((0,), dtype=torch.int64),
        }

    ligand_topology_list = edge_ligand_topology.tolist()
    try:
        edge_ligand_local = torch.tensor(
            [ligand_topology_to_local[index] for index in ligand_topology_list], dtype=torch.int64
        )
    except KeyError as exc:
        raise LocalResidualStudentError(
            f"edge references ligand topology index {exc.args[0]} outside ligand_topology_indices"
        ) from exc

    unique_environment, inverse = torch.unique(edge_environment_topology, sorted=True, return_inverse=True)
    return {
        "environment_topology_indices": unique_environment.to(torch.int64),
        "edge_ligand_local": edge_ligand_local,
        "edge_environment_local": inverse.to(torch.int64),
    }


def count_trainable_parameters(model) -> int:
    """Sum of `numel()` over `requires_grad=True` tensors -- the (d0-5) size gate."""

    return sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)


_MODEL_CLASS_CACHE: list = []


def _get_local_residual_student_class():
    """Define the `nn.Module` subclass on first use, then always reuse it."""

    if _MODEL_CLASS_CACHE:
        return _MODEL_CLASS_CACHE[0]

    import torch
    import torch.nn as nn

    class _LigandEnvironmentInteractionBlock(nn.Module):
        """CFConv-style message: environment senders -> ligand receivers only.

        Deliberately the only edge direction and the only message-passing step
        in this model (DEC-039 "边的组成": environment→ligand bipartite; no
        ligand-ligand, no environment-environment, no reverse, no self edges).
        """

        def __init__(self, hidden_dim: int, n_radial_basis: int):
            super().__init__()
            self.filter_net = nn.Sequential(
                nn.Linear(n_radial_basis, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
            )
            self.update_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(self, ligand_h, env_h, edge_ligand_local, edge_environment_local, radial_basis, envelope):
            if edge_ligand_local.numel() == 0:
                return ligand_h
            filt = self.filter_net(radial_basis) * envelope[:, None]
            messages = env_h[edge_environment_local] * filt
            aggregated = ligand_h.new_zeros(ligand_h.shape)
            aggregated = aggregated.index_add(0, edge_ligand_local, messages)
            return ligand_h + self.update_net(aggregated)

    class LocalResidualStudent(nn.Module):
        """See module docstring. Construct via `build_local_residual_student(...)`."""

        def __init__(
            self,
            type_vocabulary: Sequence[int],
            *,
            hidden_dim: int = 32,
            n_interaction_blocks: int = 2,
            n_radial_basis: int = 16,
            inner_cutoff_angstrom: float = 4.0,
            outer_cutoff_angstrom: float = 5.0,
            b_max_reduced: float = 10.0,
        ):
            super().__init__()
            vocabulary = _require_int_sequence(type_vocabulary, "type_vocabulary")
            if len(set(vocabulary)) != len(vocabulary):
                raise LocalResidualStudentError("type_vocabulary must not contain duplicates")
            if int(hidden_dim) <= 0 or int(n_interaction_blocks) <= 0 or int(n_radial_basis) <= 0:
                raise LocalResidualStudentError(
                    "hidden_dim, n_interaction_blocks, and n_radial_basis must be positive"
                )
            if not (0.0 < float(inner_cutoff_angstrom) < float(outer_cutoff_angstrom)):
                raise LocalResidualStudentError(
                    "cutoffs must satisfy 0 < inner_cutoff_angstrom < outer_cutoff_angstrom"
                )
            if not float(b_max_reduced) > 0.0:
                raise LocalResidualStudentError("b_max_reduced must be a positive finite scalar")

            self.type_vocabulary = tuple(sorted(vocabulary))
            self._type_to_index = {value: index for index, value in enumerate(self.type_vocabulary)}
            self.hidden_dim = int(hidden_dim)
            self.inner_cutoff_angstrom = float(inner_cutoff_angstrom)
            self.outer_cutoff_angstrom = float(outer_cutoff_angstrom)
            self.b_max_reduced = float(b_max_reduced)

            self.embedding = nn.Embedding(len(self.type_vocabulary), self.hidden_dim)
            centers = torch.linspace(0.0, self.outer_cutoff_angstrom, int(n_radial_basis))
            self.register_buffer("radial_centers", centers)
            width = self.outer_cutoff_angstrom / max(int(n_radial_basis) - 1, 1)
            self.register_buffer("radial_width", torch.tensor(float(width)))
            self.blocks = nn.ModuleList(
                [
                    _LigandEnvironmentInteractionBlock(self.hidden_dim, int(n_radial_basis))
                    for _ in range(int(n_interaction_blocks))
                ]
            )
            self.readout = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, 1),
            )

        def atomic_numbers_to_type_index(self, atomic_numbers: Sequence[int]):
            import torch as _torch

            try:
                return _torch.tensor(
                    [self._type_to_index[int(z)] for z in atomic_numbers], dtype=_torch.int64
                )
            except KeyError as exc:
                raise LocalResidualStudentError(
                    f"atomic number {exc.args[0]} is outside this model's frozen "
                    f"type_vocabulary {self.type_vocabulary}"
                ) from exc

        def _radial_basis(self, distance):
            diff = distance[:, None] - self.radial_centers[None, :]
            return (-0.5 * (diff / self.radial_width).square()).exp()

        def forward(
            self,
            ligand_type_index,
            environment_type_index,
            edge_ligand_local,
            edge_environment_local,
            distance,
        ):
            """Return one bounded scalar (`basis_reduced`) for this single frame.

            All tensors below must share one device/dtype (except the two
            integer index tensors). `distance` may or may not carry an
            autograd link back to Cartesian coordinates -- this model is
            agnostic to that (D1 caches plain distances for training speed;
            D2 recomputes them live from positions for the autograd/
            finite-difference force check).
            """

            for name, value in (
                ("ligand_type_index", ligand_type_index),
                ("environment_type_index", environment_type_index),
                ("edge_ligand_local", edge_ligand_local),
                ("edge_environment_local", edge_environment_local),
            ):
                if not isinstance(value, torch.Tensor) or value.ndim != 1:
                    raise LocalResidualStudentError(f"{name} must be a one-dimensional Torch tensor")
            if not isinstance(distance, torch.Tensor) or distance.ndim != 1:
                raise LocalResidualStudentError("distance must be a one-dimensional Torch tensor")
            if (
                edge_ligand_local.shape != edge_environment_local.shape
                or edge_ligand_local.shape != distance.shape
            ):
                raise LocalResidualStudentError(
                    "edge_ligand_local, edge_environment_local, and distance must have equal length"
                )
            if ligand_type_index.numel() == 0:
                raise LocalResidualStudentError("ligand_type_index must be non-empty")

            ligand_h = self.embedding(ligand_type_index)
            if environment_type_index.numel() == 0 or distance.numel() == 0:
                pooled = ligand_h.mean(dim=0)
            else:
                from .geometry import quintic_c2_cutoff

                env_h = self.embedding(environment_type_index)
                radial = self._radial_basis(distance)
                envelope = quintic_c2_cutoff(
                    distance,
                    inner_cutoff=self.inner_cutoff_angstrom,
                    outer_cutoff=self.outer_cutoff_angstrom,
                )
                for block in self.blocks:
                    ligand_h = block(ligand_h, env_h, edge_ligand_local, edge_environment_local, radial, envelope)
                pooled = ligand_h.mean(dim=0)

            raw = self.readout(pooled).squeeze(-1)
            return self.b_max_reduced * torch.tanh(raw / self.b_max_reduced)

    _MODEL_CLASS_CACHE.append(LocalResidualStudent)
    return LocalResidualStudent


def build_local_residual_student(type_vocabulary: Sequence[int], **kwargs):
    """Construct a `LocalResidualStudent` instance. See the module docstring."""

    model_class = _get_local_residual_student_class()
    return model_class(type_vocabulary, **kwargs)


__all__ = [
    "LocalResidualStudentError",
    "build_local_residual_student",
    "count_trainable_parameters",
    "reindex_ligand_environment_edges",
]
