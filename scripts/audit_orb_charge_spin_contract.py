#!/usr/bin/env python
"""Audit the charge/spin information available for an ORB local fragment.

This is deliberately a contract audit, not a prescription of quantum
chemistry.  It reports the charge carried by the selected topology atoms in
the existing OpenMM ``NonbondedForce`` and whether the serialized system
contains any spin/multiplicity field.  A missing spin contract keeps the OMol
arm exploratory even when the local charge is unambiguous.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--system-xml", required=True)
    parser.add_argument("--ligand-indices", required=True, help="comma-separated topology indices")
    parser.add_argument("--charge-tolerance", type=float, default=1e-6)
    parser.add_argument("--contract-id", default="orb-parent-system-charge-spin-v1")
    parser.add_argument("--conditioning-scope", default="parent_full_system")
    parser.add_argument("--total-charge", type=float, default=0.0)
    parser.add_argument("--spin-multiplicity", type=float, default=1.0)
    parser.add_argument(
        "--freeze-parent-contract",
        action="store_true",
        help="freeze the parent-system neutral closed-shell singlet assumption",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    import mdtraj as md
    from openmm import NonbondedForce, XmlSerializer, unit

    output = Path(args.output)
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")
    topology_path = Path(args.topology).resolve()
    system_path = Path(args.system_xml).resolve()
    ligand_indices = [int(value) for value in args.ligand_indices.split(",") if value.strip()]
    if not ligand_indices or len(set(ligand_indices)) != len(ligand_indices):
        parser.error("--ligand-indices must be non-empty and unique")

    topology = md.load(str(topology_path)).topology
    atom_count = topology.n_atoms
    if min(ligand_indices) < 0 or max(ligand_indices) >= atom_count:
        parser.error("ligand topology index is outside the topology")
    system_xml = system_path.read_text(encoding="utf-8")
    system = XmlSerializer.deserialize(system_xml)

    nonbonded = [
        system.getForce(index)
        for index in range(system.getNumForces())
        if isinstance(system.getForce(index), NonbondedForce)
    ]
    if len(nonbonded) != 1:
        raise SystemExit(f"expected exactly one NonbondedForce, found {len(nonbonded)}")
    force = nonbonded[0]
    if force.getNumParticles() != atom_count:
        raise SystemExit(
            f"topology/system particle mismatch: topology={atom_count}, system={force.getNumParticles()}"
        )
    charges = [
        float(force.getParticleParameters(index)[0].value_in_unit(unit.elementary_charge))
        for index in range(force.getNumParticles())
    ]
    total_charge = float(sum(charges))
    ligand_charge = float(sum(charges[index] for index in ligand_indices))
    spin_markers = [marker for marker in ("spin_multiplicity", "spinMultiplicity", "multiplicity") if marker in system_xml]
    charge_is_neutral = abs(ligand_charge) <= args.charge_tolerance
    exact_primary_contract = (
        args.contract_id == "orb-parent-system-charge-spin-v1"
        and args.conditioning_scope == "parent_full_system"
        and args.total_charge == 0.0
        and args.spin_multiplicity == 1.0
    )
    charge_contract_ok = (
        abs(total_charge - args.total_charge) <= args.charge_tolerance
        and abs(ligand_charge - args.total_charge) <= args.charge_tolerance
    )
    if args.freeze_parent_contract and not exact_primary_contract:
        raise SystemExit(
            "--freeze-parent-contract requires the registered parent contract: Q=0, M=1, "
            "orb-parent-system-charge-spin-v1, parent_full_system"
        )
    primary_qualified = bool(args.freeze_parent_contract and charge_contract_ok)
    body = {
        "schema_version": "orb-charge-spin-contract-audit-v1",
        "status": (
            "FROZEN_PARENT_CONDITIONING_CONTRACT"
            if primary_qualified else "UNFROZEN_SPIN_CONTRACT"
        ),
        "primary_qualified": primary_qualified,
        "command": " ".join(sys.argv),
        "inputs": {
            "topology": {"path": str(topology_path), "sha256": _sha256(topology_path)},
            "system_xml": {"path": str(system_path), "sha256": _sha256(system_path)},
            "ligand_indices": ligand_indices,
        },
        "conditioning_contract": {
            "contract_id": args.contract_id,
            "conditioning_scope": args.conditioning_scope,
            "total_charge": args.total_charge,
            "spin_multiplicity": args.spin_multiplicity,
            "interpretation": (
                "closed-shell singlet conditioning inherited from the parent system; "
                "not the electronic multiplicity of the truncated L-hop fragment"
            ),
            "status": (
                "FROZEN_PARENT_NEUTRAL + FROZEN_PARENT_CLOSED_SHELL_SINGLET_ASSUMPTION"
                if primary_qualified else "UNFROZEN"
            ),
        },
        "system": {
            "topology_atom_count": atom_count,
            "nonbonded_particle_count": force.getNumParticles(),
            "total_charge_e": total_charge,
            "ligand_charge_e": ligand_charge,
            "charge_tolerance_e": args.charge_tolerance,
            "ligand_charge_consistent_with_declared_parent_charge": charge_contract_ok,
            "total_charge_consistent_with_declared_parent_charge": (
                abs(total_charge - args.total_charge) <= args.charge_tolerance
            ),
            "ligand_charge_consistent_with_neutral": charge_is_neutral,
        },
        "spin": {
            "markers_found_in_serialized_system": spin_markers,
            "spin_contract_present": bool(spin_markers),
            "status": "present" if spin_markers else "absent_from_openmm_system",
        },
        "conclusion": {
            "charge_contract": "FROZEN_PARENT_NEUTRAL" if primary_qualified else (
                "candidate_neutral" if charge_is_neutral else "non_neutral_or_ambiguous"
            ),
            "spin_contract": (
                "FROZEN_PARENT_CLOSED_SHELL_SINGLET_ASSUMPTION"
                if primary_qualified else "UNFROZEN"
            ),
            "omol_primary_allowed": primary_qualified,
            "reason": (
                "parent-system conditioning contract is explicitly frozen; missing spin in OpenMM XML "
                "is recorded as expected, not treated as a blocker"
                if primary_qualified else
                "spin multiplicity is not supplied by the current OpenMM system serialization"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": body["status"], "ligand_charge_e": ligand_charge}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
