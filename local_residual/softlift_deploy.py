"""Scriptable R1 coordinate deploy path for EXP-019 D3.

The first deploy path is deliberately an explicit triclinic brute-force
reference.  It is correct and scriptable, but it is not claimed to be an
efficient online neighbor list.  A production/native path remains gated by
the real cost criterion in ``DiffLift.MD``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence


class SoftLiftDeploymentError(ValueError):
    """An R1 deployment identity or support-domain check failed."""


def build_softlift_r1_deploy(
    model,
    *,
    ligand_topology_indices: Sequence[int],
    all_topology_atomic_numbers: Sequence[int],
    temperature_kelvin: float,
    a_k: float = 1.0,
    min_distance_support_angstrom: float = 0.1,
    fixed_edges: tuple[Sequence[int], Sequence[int]] | None = None,
):
    """Create a coordinate-facing R1 module returning ``U_B`` in kJ/mol.

    ``positions`` and ``box`` passed to the returned module are OpenMM-style
    nanometers.  The conversion to Angstrom happens exactly once internally.
    The returned artifact has no state-specific ``A_k``; ``a_k`` is accepted
    only as an identity assertion and must equal ``1.0``.  When ``fixed_edges``
    is supplied, it is a candidate Verlet pool expressed as
    ``(ligand_topology_indices, environment_topology_indices)``.  Live
    distances and strict cutoff membership are still evaluated every call.
    """

    import math
    import torch
    import torch.nn as nn

    if getattr(model.config, "rung", None) != "R1":
        raise SoftLiftDeploymentError("only R1 has a first deploy prototype")
    if not math.isfinite(float(a_k)) or float(a_k) != 1.0:
        raise SoftLiftDeploymentError("deploy artifact a_k is fixed at 1.0; OuterLambda applies A_k")
    if not math.isfinite(float(temperature_kelvin)) or float(temperature_kelvin) <= 0.0:
        raise SoftLiftDeploymentError("temperature_kelvin must be finite and positive")
    if not math.isfinite(float(min_distance_support_angstrom)) or float(min_distance_support_angstrom) <= 0.0:
        raise SoftLiftDeploymentError("min_distance_support_angstrom must be finite and positive")
    ligand = [int(value) for value in ligand_topology_indices]
    all_numbers = [int(value) for value in all_topology_atomic_numbers]
    if not ligand or len(set(ligand)) != len(ligand):
        raise SoftLiftDeploymentError("ligand topology indices must be unique and non-empty")
    if any(value < 0 or value >= len(all_numbers) for value in ligand):
        raise SoftLiftDeploymentError("ligand topology index is out of range")
    type_map = {int(value): index for index, value in enumerate(model.config.type_vocabulary)}
    try:
        ligand_types = [type_map[all_numbers[index]] for index in ligand]
    except KeyError as exc:
        raise SoftLiftDeploymentError(f"atomic number {exc.args[0]} is outside frozen type vocabulary") from exc
    environment = [index for index in range(len(all_numbers)) if index not in set(ligand)]
    try:
        environment_types = [type_map[all_numbers[index]] for index in environment]
    except KeyError as exc:
        raise SoftLiftDeploymentError(f"atomic number {exc.args[0]} is outside frozen type vocabulary") from exc

    fixed_edge_ligand_local: list[int] = []
    fixed_edge_environment: list[int] = []
    ligand_to_local = {topology_index: local_index for local_index, topology_index in enumerate(ligand)}
    if fixed_edges is not None:
        if len(fixed_edges) != 2 or len(fixed_edges[0]) != len(fixed_edges[1]):
            raise SoftLiftDeploymentError("fixed_edges must contain equal-length ligand and environment arrays")
        for ligand_index, environment_index in zip(fixed_edges[0], fixed_edges[1]):
            ligand_index = int(ligand_index)
            environment_index = int(environment_index)
            if ligand_index not in ligand_to_local:
                raise SoftLiftDeploymentError("fixed_edges contains a topology index outside the ligand")
            if environment_index < 0 or environment_index >= len(all_numbers):
                raise SoftLiftDeploymentError("fixed_edges contains an out-of-range environment topology index")
            if environment_index in ligand_to_local:
                raise SoftLiftDeploymentError("fixed_edges must contain ligand-environment pairs")
            fixed_edge_ligand_local.append(ligand_to_local[ligand_index])
            fixed_edge_environment.append(environment_index)
        if len(set(zip(fixed_edge_ligand_local, fixed_edge_environment))) != len(fixed_edge_ligand_local):
            raise SoftLiftDeploymentError("fixed_edges must not contain duplicate pairs")

    class DeployR1(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_ligand_atoms = int(model.config.n_ligand_atoms)
            self.type_count = len(model.config.type_vocabulary)
            self.n_radial_basis = int(model.config.n_radial_basis)
            self.inner_cutoff_angstrom = float(model.config.inner_cutoff_angstrom)
            self.outer_cutoff_angstrom = float(model.config.outer_cutoff_angstrom)
            self.b_max_reduced = float(model.config.b_max_reduced)
            self.min_distance_support_angstrom = float(min_distance_support_angstrom)
            self.temperature_kelvin = float(temperature_kelvin)
            self.register_buffer("ligand_indices", torch.tensor(ligand, dtype=torch.int64))
            self.register_buffer("environment_indices", torch.tensor(environment, dtype=torch.int64))
            self.register_buffer("ligand_type_index", torch.tensor(ligand_types, dtype=torch.int64))
            self.register_buffer("environment_type_index", torch.tensor(environment_types, dtype=torch.int64))
            self.register_buffer(
                "type_index_by_topology",
                torch.tensor([type_map[number] for number in all_numbers], dtype=torch.int64),
            )
            self.register_buffer("fixed_edge_ligand_local", torch.tensor(fixed_edge_ligand_local, dtype=torch.int64))
            self.register_buffer("fixed_edge_environment", torch.tensor(fixed_edge_environment, dtype=torch.int64))
            self.use_fixed_edges = fixed_edges is not None
            self.register_buffer("radial_centers", model.radial_centers.detach().clone())
            self.register_buffer("radial_width", model.radial_width.detach().clone())
            self.pair_weight = nn.Parameter(model.pair_weight.detach().clone())
            self.rho = deepcopy(model.rho)

        def _radial_basis(self, distance):
            difference = distance[:, None] - self.radial_centers[None, :]
            return torch.exp(-0.5 * (difference / self.radial_width).square())

        def _envelope(self, distance):
            x = (distance - self.inner_cutoff_angstrom) / (self.outer_cutoff_angstrom - self.inner_cutoff_angstrom)
            transition = 1.0 - 10.0 * x.pow(3) + 15.0 * x.pow(4) - 6.0 * x.pow(5)
            return torch.where(
                distance <= self.inner_cutoff_angstrom,
                torch.ones_like(distance),
                torch.where(distance >= self.outer_cutoff_angstrom, torch.zeros_like(distance), transition),
            )

        def forward(self, positions_nm, boxvectors):
            positions_angstrom = positions_nm * 10.0
            box_angstrom = boxvectors * 10.0
            if self.use_fixed_edges:
                fixed_ligand = self.ligand_indices.index_select(0, self.fixed_edge_ligand_local)
                source = positions_angstrom.index_select(0, fixed_ligand)
                target = positions_angstrom.index_select(0, self.fixed_edge_environment)
                raw = target - source
                fractional = torch.linalg.solve(
                    box_angstrom.transpose(0, 1), raw.reshape(-1, 3).transpose(0, 1)
                ).transpose(0, 1)
                centered = fractional - torch.round(fractional)
                displacement = torch.matmul(centered, box_angstrom)
                distance = torch.linalg.vector_norm(displacement, dim=-1)
                membership = distance < self.outer_cutoff_angstrom
                active_ligand = self.fixed_edge_ligand_local[membership]
                active_environment = self.fixed_edge_environment[membership]
                active_distance = distance[membership]
                tie = (fractional.abs() - 0.5).abs() <= 1.0e-12
                active_tie = tie.any(dim=-1) & membership
                torch._assert(not bool(active_tie.any()), "active half-box tie detected")
                torch._assert(int(active_distance.shape[0]) <= 2048, "directed edge hard ceiling exceeded")
                torch._assert(
                    bool(torch.all(active_distance >= self.min_distance_support_angstrom)),
                    "minimum-distance support violated",
                )
                torch._assert(
                    int(torch.unique(active_environment).numel()) <= 320,
                    "unique environment hard ceiling exceeded",
                )
                receiver_count = torch.bincount(active_ligand, minlength=self.n_ligand_atoms)
                torch._assert(int(receiver_count.max()) <= 80, "per-ligand neighbor hard ceiling exceeded")
                pair_ligand_type = self.ligand_type_index[active_ligand]
                pair_environment_type = self.type_index_by_topology[active_environment]
                edge_q = self._envelope(active_distance) * (
                    self.pair_weight[pair_ligand_type, pair_environment_type]
                    * self._radial_basis(active_distance)
                ).sum(dim=-1)
                q = positions_nm.new_zeros((self.n_ligand_atoms,)).index_add(0, active_ligand, edge_q)
                zeros = torch.zeros_like(q)
                rho_q = torch.stack([network(q[:, None]).squeeze(-1) for network in self.rho], dim=-1)
                rho_zero = torch.stack([network(zeros[:, None]).squeeze(-1) for network in self.rho], dim=-1)
                selected = torch.nn.functional.one_hot(self.ligand_type_index, num_classes=self.type_count).to(q.dtype)
                raw_reduced = ((rho_q - rho_zero) * selected).sum(dim=-1).sum()
                reduced = self.b_max_reduced * torch.tanh(raw_reduced / self.b_max_reduced)
                return reduced * (0.00831446261815324 * self.temperature_kelvin)
            ligand_positions = positions_angstrom.index_select(0, self.ligand_indices)
            environment_positions = positions_angstrom.index_select(0, self.environment_indices)
            raw = environment_positions[None, :, :] - ligand_positions[:, None, :]
            fractional = torch.linalg.solve(box_angstrom.transpose(0, 1), raw.reshape(-1, 3).transpose(0, 1)).transpose(0, 1).reshape(raw.shape)
            centered = fractional - torch.round(fractional)
            displacement = torch.matmul(centered.reshape(-1, 3), box_angstrom).reshape(raw.shape)
            distance = torch.linalg.vector_norm(displacement, dim=-1)
            membership = distance < self.outer_cutoff_angstrom
            # A half-box tie outside the strict cutoff cannot contribute to
            # the artifact.  Check ties only on active pairs so float32
            # rounding in distant, irrelevant pairs does not create a false
            # failure; active ties remain fail-closed.
            tie = (fractional.abs() - 0.5).abs() <= 1.0e-12
            active_tie = tie.any(dim=-1) & membership
            torch._assert(not bool(active_tie.any()), "active half-box tie detected")
            torch._assert(int(membership.sum()) <= 2048, "directed edge hard ceiling exceeded")
            active_distance = distance[membership]
            torch._assert(bool(torch.all(active_distance >= self.min_distance_support_angstrom)), "minimum-distance support violated")
            ligand_receivers = torch.arange(self.n_ligand_atoms, dtype=torch.int64, device=positions_nm.device)[:, None].expand(-1, self.environment_indices.shape[0]).reshape(-1)[membership.reshape(-1)]
            environment_senders = torch.arange(self.environment_indices.shape[0], dtype=torch.int64, device=positions_nm.device)[None, :].expand(self.n_ligand_atoms, -1).reshape(-1)[membership.reshape(-1)]
            torch._assert(int(torch.unique(environment_senders).numel()) <= 320, "unique environment hard ceiling exceeded")
            receiver_count = torch.bincount(ligand_receivers, minlength=self.n_ligand_atoms)
            torch._assert(int(receiver_count.max()) <= 80, "per-ligand neighbor hard ceiling exceeded")
            pair_ligand_type = self.ligand_type_index[ligand_receivers]
            pair_environment_type = self.environment_type_index[environment_senders]
            edge_q = self._envelope(active_distance) * (
                self.pair_weight[pair_ligand_type, pair_environment_type] * self._radial_basis(active_distance)
            ).sum(dim=-1)
            q = positions_nm.new_zeros((self.n_ligand_atoms,)).index_add(0, ligand_receivers, edge_q)
            zeros = torch.zeros_like(q)
            rho_q = torch.stack([network(q[:, None]).squeeze(-1) for network in self.rho], dim=-1)
            rho_zero = torch.stack([network(zeros[:, None]).squeeze(-1) for network in self.rho], dim=-1)
            selected = torch.nn.functional.one_hot(self.ligand_type_index, num_classes=self.type_count).to(q.dtype)
            raw_reduced = ((rho_q - rho_zero) * selected).sum(dim=-1).sum()
            reduced = self.b_max_reduced * torch.tanh(raw_reduced / self.b_max_reduced)
            return reduced * (0.00831446261815324 * self.temperature_kelvin)

    return DeployR1()


__all__ = ["SoftLiftDeploymentError", "build_softlift_r1_deploy"]
