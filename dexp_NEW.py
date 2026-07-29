#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pair-specific LJ-matched DEXP production system builder.

This module is intentionally small.  The research and validation harness remains
in ``dexp_experiment.py``; production code must not depend on its legacy global
Orb fitter or on fitted ``r0_vdw/A_fit/B_fit/offset_c0`` values.

The production model follows ``docs/experiments/DEXP_KERNEL_PHYSICS_ISSUES.md``:

* ``sigma_ij`` and ``epsilon_ij`` come from the original force field through
  Lorentz-Berthelot combining rules;
* ``r0_ij = 2**(1/6) * sigma_ij``;
* only the analytic kernel shape ``alpha_vdw/beta_vdw`` is configurable;
* the current validated Atenolol default is ``(14, 5)``;
* Gaussian Coulomb uses a shifted-force cutoff, while DEXP uses the matching
  0.50--0.70 nm switching shell.

Typical use::

    python dexp_NEW.py \
      --system-xml output/system_native.xml \
      --topology output/topology.cif \
      --ligand-indices output/ligand_indices.json \
      --output-system output/system_dexp_pair_specific.xml
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


PROTOCOL_NAME = "pair_specific_lj_matched_dexp"
PROTOCOL_VERSION = 1

# These keys belong to the retired global Orb fit.  Silently accepting them is
# dangerous because the pair-specific production expression cannot consume them.
LEGACY_FIT_KEYS = frozenset(
    {
        "r0_vdw",
        "A_fit",
        "B_fit",
        "offset_c0",
        "offset_c1",
        "fitting_success",
        "final_cost",
        "fit_target_mode",
        "fit_objective",
    }
)


@dataclass(frozen=True)
class DEXPProductionConfig:
    """Complete parameter contract for the new production DEXP Hamiltonian."""

    alpha_vdw: float = 14.0
    beta_vdw: float = 5.0
    sigma_elec: float = 0.10
    switch_width: float = 0.20
    cutoff_distance: float = 0.70

    def validate(self) -> "DEXPProductionConfig":
        values = asdict(self)
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError(f"DEXP production parameters must be finite: {values}")
        if not self.alpha_vdw > self.beta_vdw > 0.0:
            raise ValueError(
                "DEXP requires alpha_vdw > beta_vdw > 0; "
                f"got alpha={self.alpha_vdw}, beta={self.beta_vdw}"
            )
        if self.sigma_elec <= 0.0:
            raise ValueError(f"sigma_elec must be positive; got {self.sigma_elec}")
        if not 0.0 < self.switch_width < self.cutoff_distance:
            raise ValueError(
                "switch_width must satisfy 0 < switch_width < cutoff_distance; "
                f"got width={self.switch_width}, cutoff={self.cutoff_distance}"
            )
        return self

    @property
    def switching_distance(self) -> float:
        return float(self.cutoff_distance - self.switch_width)

    def to_builder_params(self) -> Dict[str, float]:
        """Return only fields consumed by the pair-specific production class."""
        self.validate()
        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DEXPProductionConfig":
        legacy = sorted(LEGACY_FIT_KEYS.intersection(raw))
        if legacy:
            raise ValueError(
                "Refusing legacy/global DEXP fit fields in the new production "
                f"contract: {', '.join(legacy)}"
            )
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw).difference(allowed))
        if unknown:
            raise ValueError(
                "Unknown DEXP production fields: " + ", ".join(unknown)
            )
        return cls(**{key: float(value) for key, value in raw.items()}).validate()


def load_ligand_indices(path: Path | str) -> Tuple[int, ...]:
    """Load and strictly validate ``{"ligand_indices": [...]}``."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot read ligand index JSON {source}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"ligand_indices"}:
        raise ValueError(
            f"{source} must contain exactly one key: 'ligand_indices'"
        )
    raw_indices = payload["ligand_indices"]
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError(f"{source}: ligand_indices must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_indices):
        raise ValueError(f"{source}: every ligand index must be an integer")
    indices = tuple(int(value) for value in raw_indices)
    if len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError(
            f"{source}: ligand indices must be unique and non-negative"
        )
    return indices


def _box_vectors_nm(system, box_vectors=None):
    from openmm import unit

    vectors = box_vectors
    if vectors is None:
        vectors = system.getDefaultPeriodicBoxVectors()
    if vectors is None:
        raise ValueError("A periodic DEXP production system requires box vectors")
    if hasattr(vectors, "value_in_unit"):
        rows = vectors.value_in_unit(unit.nanometer)
    else:
        rows = []
        for vector in vectors:
            if hasattr(vector, "value_in_unit"):
                rows.append(vector.value_in_unit(unit.nanometer))
            else:
                rows.append(vector)
    return tuple(tuple(float(x) for x in row) for row in rows)


def _validate_minimum_image(box_nm, cutoff_nm: float) -> None:
    import numpy as np

    matrix = np.asarray(box_nm, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"Invalid periodic box vectors: shape={matrix.shape}")
    volume = float(abs(np.linalg.det(matrix)))
    if volume <= 0.0:
        raise ValueError(f"Periodic box has non-positive volume: {volume}")
    # Plane spacings, valid for triclinic cells: h_i = V / |a_j x a_k|.
    spacings = []
    for axis in range(3):
        other = [idx for idx in range(3) if idx != axis]
        area = float(np.linalg.norm(np.cross(matrix[other[0]], matrix[other[1]])))
        if area <= 0.0:
            raise ValueError("Periodic box contains collinear vectors")
        spacings.append(volume / area)
    if min(spacings) <= 2.0 * float(cutoff_nm):
        raise ValueError(
            "Minimum-image violation: shortest box plane spacing "
            f"{min(spacings):.6f} nm must exceed 2*cutoff="
            f"{2.0 * cutoff_nm:.6f} nm"
        )


def build_production_system(
    original_system,
    ligand_indices: Sequence[int],
    *,
    config: DEXPProductionConfig | None = None,
    reference_positions=None,
    box_vectors=None,
    force_group: int = 1,
):
    """Build the new DEXP Hamiltonian without invoking any fitting code."""
    from abfe_core import SurrogateSystemBuilder

    cfg = (config or DEXPProductionConfig()).validate()
    n_particles = int(original_system.getNumParticles())
    ligand = tuple(int(index) for index in ligand_indices)
    if not ligand:
        raise ValueError("ligand_indices is empty")
    if len(set(ligand)) != len(ligand):
        raise ValueError("ligand_indices contains duplicates")
    if min(ligand) < 0 or max(ligand) >= n_particles:
        raise ValueError(
            f"ligand index outside [0, {n_particles}): {sorted(ligand)}"
        )
    ligand_set = set(ligand)
    environment = tuple(index for index in range(n_particles) if index not in ligand_set)
    if not environment:
        raise ValueError("No environment atoms remain after selecting the ligand")

    resolved_box = _box_vectors_nm(original_system, box_vectors)
    _validate_minimum_image(resolved_box, cfg.cutoff_distance)

    builder = SurrogateSystemBuilder(cfg.to_builder_params())
    return builder.build_surrogate_system(
        original_system=original_system,
        ligand_indices=ligand,
        environment_indices=environment,
        lambda_names=("lam_coul", "lam_vdw"),
        force_group=int(force_group),
        reference_positions=reference_positions,
        box_vectors=box_vectors,
    )


def production_manifest(
    *,
    config: DEXPProductionConfig,
    input_system: Path,
    ligand_indices: Sequence[int],
    output_system: Path,
) -> Dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "model": "pair-specific LJ-matched analytic DEXP",
        "parameters": config.to_builder_params(),
        "derived": {
            "switching_distance_nm": config.switching_distance,
            "r0_rule": "r0_ij = 2^(1/6) * 0.5 * (sigma_i + sigma_j)",
            "epsilon_rule": "epsilon_ij = sqrt(epsilon_i * epsilon_j)",
        },
        "legacy_global_fit_fields_consumed": False,
        "input_system": str(input_system),
        "input_system_sha256": _sha256(input_system),
        "ligand_indices": [int(index) for index in ligand_indices],
        "output_system": str(output_system),
        "output_system_sha256": _sha256(output_system),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the pair-specific LJ-matched DEXP production System"
    )
    parser.add_argument(
        "--system-xml", type=_existing_file, default=Path("output/system_native.xml")
    )
    parser.add_argument(
        "--topology", type=_existing_file, default=Path("output/topology.cif")
    )
    parser.add_argument(
        "--ligand-indices",
        type=_existing_file,
        default=Path("output/ligand_indices.json"),
    )
    parser.add_argument(
        "--output-system",
        type=Path,
        default=Path("output/system_dexp_pair_specific.xml"),
    )
    parser.add_argument("--alpha", type=float, default=14.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--sigma-elec-nm", type=float, default=0.10)
    parser.add_argument("--switch-width-nm", type=float, default=0.20)
    parser.add_argument("--cutoff-nm", type=float, default=0.70)
    parser.add_argument("--force-group", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    from openmm import XmlSerializer, app

    args = parse_args(argv)
    input_system = Path(args.system_xml).resolve()
    topology_path = Path(args.topology).resolve()
    ligand_path = Path(args.ligand_indices).resolve()
    output_system = Path(args.output_system).expanduser().resolve()
    if output_system == input_system:
        raise ValueError("Refusing to overwrite the native input System XML")

    config = DEXPProductionConfig(
        alpha_vdw=args.alpha,
        beta_vdw=args.beta,
        sigma_elec=args.sigma_elec_nm,
        switch_width=args.switch_width_nm,
        cutoff_distance=args.cutoff_nm,
    ).validate()
    ligand_indices = load_ligand_indices(ligand_path)

    original_system = XmlSerializer.deserialize(
        input_system.read_text(encoding="utf-8")
    )
    coordinates = app.PDBxFile(str(topology_path))
    if len(coordinates.positions) != original_system.getNumParticles():
        raise ValueError(
            "Topology/System particle count mismatch: "
            f"{len(coordinates.positions)} != {original_system.getNumParticles()}"
        )
    topology_box = coordinates.topology.getPeriodicBoxVectors()
    new_system = build_production_system(
        original_system,
        ligand_indices,
        config=config,
        reference_positions=coordinates.positions,
        box_vectors=topology_box,
        force_group=args.force_group,
    )

    output_system.parent.mkdir(parents=True, exist_ok=True)
    output_system.write_text(XmlSerializer.serialize(new_system), encoding="utf-8")
    manifest_path = output_system.with_suffix(output_system.suffix + ".manifest.json")
    manifest = production_manifest(
        config=config,
        input_system=input_system,
        ligand_indices=ligand_indices,
        output_system=output_system,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"DEXP production System: {output_system}")
    print(f"Protocol manifest: {manifest_path}")
    print(
        "Model: pair-specific LJ-matched DEXP "
        f"(alpha,beta)=({config.alpha_vdw:g},{config.beta_vdw:g}), "
        f"switch/cutoff={config.switching_distance:.2f}/{config.cutoff_distance:.2f} nm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
