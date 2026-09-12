"""EXP-020 R1 native OpenMM prototype boundary.

The prototype uses a ``CustomGBForce`` pair reduction to build one scalar
density per ligand atom and a ``CustomCVForce`` to apply the nonlinear global
bounded readout.  It is deliberately a prototype: the force is eligible to
be built after D1/D2, but native parity and the real cost gate remain separate
qualification requirements.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


class NativeSoftLiftNotAuthorized(RuntimeError):
    """The native prototype cannot be built from the supplied identity."""


def native_r1_status() -> dict[str, object]:
    return {
        "backend": "N0_CustomGBForce_CustomCVForce",
        "status": "PROTOTYPE_ELIGIBLE",
        "eligibility": "R1_D1_AND_D2_SATISFIED",
        "pair_semantics": "ParticlePairNoExclusions",
        "state_specific_A_k_in_artifact": False,
        "beta_in_artifact": False,
        "cost_gate": "candidate_ms_per_step <= 1.10 * baseline_ms_per_step",
        "reason_not_constructed": "D1/D2 satisfied; native prototype and parity/cost qualification pending",
    }


def _format(value: float) -> str:
    return format(float(value), ".17g")


def _model_scalar(value: Any) -> float:
    return float(value.detach().cpu().item())


def _add_global(force, name: str, value: float) -> str:
    force.addGlobalParameter(name, float(value))
    return name


def _typed_rho_zero(model, type_index: int) -> float:
    import torch

    network = model.rho[type_index]
    parameter = next(network.parameters())
    zero = torch.zeros((1, 1), dtype=parameter.dtype, device=parameter.device)
    with torch.no_grad():
        return float(network(zero).reshape(-1)[0].detach().cpu().item())


def _linear_layers(network):
    layers = [layer for layer in network if hasattr(layer, "weight") and hasattr(layer, "bias")]
    if len(layers) != 3:
        raise NativeSoftLiftNotAuthorized("R1 rho network must contain exactly three Linear layers")
    return layers


def _build_pair_density_expression(force, model, type_count: int, n_radial_basis: int) -> str:
    import torch

    inner = _format(float(model.config.inner_cutoff_angstrom))
    outer = _format(float(model.config.outer_cutoff_angstrom))
    width = _format(_model_scalar(model.radial_width))
    centers = [_model_scalar(value) for value in model.radial_centers]
    pair_weight = model.pair_weight.detach().cpu().to(dtype=torch.float64)
    scaled_distance = "(10*r)"
    scaled_x = f"(({scaled_distance}-{inner})/({outer}-{inner}))"
    envelope = f"select({inner}-{scaled_distance},1,select({outer}-{scaled_distance},1-10*{scaled_x}^3+15*{scaled_x}^4-6*{scaled_x}^5,0))"
    terms = []
    for ligand_type in range(type_count):
        for environment_type in range(type_count):
            type_factor = f"lt{ligand_type}1*et{environment_type}2"
            for radial_index in range(n_radial_basis):
                weight_name = f"w_{ligand_type}_{environment_type}_{radial_index}"
                _add_global(force, weight_name, float(pair_weight[ligand_type, environment_type, radial_index].item()))
                center = _format(centers[radial_index])
                radial = f"exp(-0.5*((({scaled_distance}-{center})/{width})^2))"
                terms.append(f"{type_factor}*{weight_name}*{radial}")
    if not terms:
        raise NativeSoftLiftNotAuthorized("R1 radial basis must contain at least one basis function")
    return f"lf1*ef2*{envelope}*({' + '.join(terms)})"


def _add_rho_network(force, model, type_index: int) -> str:
    layers = _linear_layers(model.rho[type_index])
    first, second, third = layers
    first_weights = first.weight.detach().cpu().reshape(-1).tolist()
    first_bias = first.bias.detach().cpu().reshape(-1).tolist()
    second_weights = second.weight.detach().cpu().tolist()
    second_bias = second.bias.detach().cpu().reshape(-1).tolist()
    third_weights = third.weight.detach().cpu().reshape(-1).tolist()
    third_bias = float(third.bias.detach().cpu().reshape(-1)[0].item())
    first_names = []
    for hidden_index, (weight, bias) in enumerate(zip(first_weights, first_bias)):
        weight_name = _add_global(force, f"rho_{type_index}_w1_{hidden_index}", weight)
        bias_name = _add_global(force, f"rho_{type_index}_b1_{hidden_index}", bias)
        hidden_name = f"rho_{type_index}_h1_{hidden_index}"
        activation_name = f"rho_{type_index}_a1_{hidden_index}"
        force.addComputedValue(
            hidden_name,
            f"lt{type_index}*({bias_name}+{weight_name}*q)",
            force.SingleParticle,
        )
        force.addComputedValue(
            activation_name,
            f"{hidden_name}/(1+exp(-{hidden_name}))",
            force.SingleParticle,
        )
        first_names.append(activation_name)
    second_names = []
    for hidden_index, bias in enumerate(second_bias):
        bias_name = _add_global(force, f"rho_{type_index}_b2_{hidden_index}", bias)
        terms = []
        for input_index, weight in enumerate(second_weights[hidden_index]):
            weight_name = _add_global(force, f"rho_{type_index}_w2_{hidden_index}_{input_index}", weight)
            terms.append(f"{weight_name}*{first_names[input_index]}")
        hidden_name = f"rho_{type_index}_h2_{hidden_index}"
        activation_name = f"rho_{type_index}_a2_{hidden_index}"
        force.addComputedValue(
            hidden_name,
            f"lt{type_index}*({bias_name}+{' + '.join(terms)})",
            force.SingleParticle,
        )
        force.addComputedValue(
            activation_name,
            f"{hidden_name}/(1+exp(-{hidden_name}))",
            force.SingleParticle,
        )
        second_names.append(activation_name)
    output_terms = []
    for input_index, weight in enumerate(third_weights):
        weight_name = _add_global(force, f"rho_{type_index}_w3_{input_index}", weight)
        output_terms.append(f"{weight_name}*{second_names[input_index]}")
    bias_name = _add_global(force, f"rho_{type_index}_b3", third_bias)
    output_name = f"rho_{type_index}"
    force.addComputedValue(
        output_name,
        f"lt{type_index}*({bias_name}+{' + '.join(output_terms)})",
        force.SingleParticle,
    )
    return output_name


def build_native_r1(
    model,
    *,
    ligand_topology_indices: Sequence[int],
    all_topology_atomic_numbers: Sequence[int],
    temperature_kelvin: float,
    a_k: float = 1.0,
):
    """Build the EXP-020 R1 native prototype force.

    The returned ``CustomCVForce`` produces ``kT * B_theta``.  It contains no
    state-specific ``A_k`` and no beta reapplication.  The pair reduction is
    intentionally a full ``ParticlePairNoExclusions`` scan; the cost gate
    must measure this exact prototype before any promotion decision.
    """

    try:
        import openmm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise NativeSoftLiftNotAuthorized("OpenMM is required for the native R1 prototype") from exc

    config = getattr(model, "config", None)
    if config is None or getattr(config, "rung", None) != "R1":
        raise NativeSoftLiftNotAuthorized("native prototype only accepts an R1 reference model")
    if not math.isfinite(float(temperature_kelvin)) or float(temperature_kelvin) <= 0.0:
        raise NativeSoftLiftNotAuthorized("temperature_kelvin must be finite and positive")
    if not math.isfinite(float(a_k)) or float(a_k) != 1.0:
        raise NativeSoftLiftNotAuthorized("native artifact a_k is fixed at 1.0")
    ligand = [int(value) for value in ligand_topology_indices]
    atomic_numbers = [int(value) for value in all_topology_atomic_numbers]
    if len(ligand) != config.n_ligand_atoms or len(set(ligand)) != len(ligand):
        raise NativeSoftLiftNotAuthorized("ligand topology indices do not match the frozen R1 ligand")
    if any(value < 0 or value >= len(atomic_numbers) for value in ligand):
        raise NativeSoftLiftNotAuthorized("ligand topology index is outside the atom range")
    type_map = {int(value): index for index, value in enumerate(config.type_vocabulary)}
    ligand_set = set(ligand)
    environment = [index for index in range(len(atomic_numbers)) if index not in ligand_set]
    try:
        ligand_types = [type_map[atomic_numbers[index]] for index in ligand]
        environment_types = [type_map[atomic_numbers[index]] for index in environment]
    except KeyError as exc:
        raise NativeSoftLiftNotAuthorized(f"atomic number {exc.args[0]} is outside the frozen R1 vocabulary") from exc
    ligand_type_by_atom = dict(zip(ligand, ligand_types))
    environment_type_by_atom = dict(zip(environment, environment_types))

    force = openmm.CustomGBForce()
    force.setName("EXP020_R1_NativePrototype_Density")
    force.setNonbondedMethod(openmm.CustomGBForce.CutoffPeriodic)
    force.setCutoffDistance(float(config.outer_cutoff_angstrom) / 10.0)
    force.addPerParticleParameter("lf")
    force.addPerParticleParameter("ef")
    for type_index in range(len(config.type_vocabulary)):
        force.addPerParticleParameter(f"lt{type_index}")
    for type_index in range(len(config.type_vocabulary)):
        force.addPerParticleParameter(f"et{type_index}")

    for atom_index, atomic_number in enumerate(atomic_numbers):
        is_ligand = atom_index in ligand_set
        values = [1.0 if is_ligand else 0.0, 0.0 if is_ligand else 1.0]
        values.extend(1.0 if is_ligand and index == ligand_type_by_atom[atom_index] else 0.0 for index in range(len(config.type_vocabulary)))
        values.extend(1.0 if not is_ligand and index == environment_type_by_atom[atom_index] else 0.0 for index in range(len(config.type_vocabulary)))
        force.addParticle(values)

    type_count = len(config.type_vocabulary)
    n_radial_basis = int(config.n_radial_basis)
    pair_expression = _build_pair_density_expression(force, model, type_count, n_radial_basis)
    force.addComputedValue("q", pair_expression, openmm.CustomGBForce.ParticlePairNoExclusions)
    rho_names = [_add_rho_network(force, model, type_index) for type_index in range(type_count)]
    rho_zero_terms = []
    for type_index in range(type_count):
        zero_name = _add_global(force, f"rho_{type_index}_zero", _typed_rho_zero(model, type_index))
        rho_zero_terms.append(f"lt{type_index}*{zero_name}")
    force.addComputedValue("rho_total", "+".join(rho_names), openmm.CustomGBForce.SingleParticle)
    force.addEnergyTerm(
        f"lf*(rho_total-({' + '.join(rho_zero_terms)}))",
        openmm.CustomGBForce.SingleParticle,
    )

    k_t = 0.00831446261815324 * float(temperature_kelvin)
    cv = openmm.CustomCVForce(
        f"{_format(k_t)}*{_format(float(config.b_max_reduced))}*tanh(B/{_format(float(config.b_max_reduced))})"
    )
    cv.setName("EXP020_R1_NativePrototype_BoundedBasis")
    cv.addCollectiveVariable("B", force)
    return cv


__all__ = ["NativeSoftLiftNotAuthorized", "build_native_r1", "native_r1_status"]
