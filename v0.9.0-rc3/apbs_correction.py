#!/usr/bin/env python
"""Prepare and collect an external APBS correction for ABFE results.

This helper intentionally starts from PQR files.  The PQR charge/radius model is
part of the physical protocol and should be generated with a documented tool
such as PDB2PQR or a force-field-specific exporter before this script is used.

Typical use:

    python apbs_correction.py prepare \
      --complex-pqr complex.pqr \
      --receptor-pqr receptor.pqr \
      --ligand-pqr ligand.pqr \
      --out-dir output/apbs

    python apbs_correction.py run --out-dir output/apbs --apbs-bin apbs

    python apbs_correction.py collect --out-dir output/apbs

The generated JSON contains the value to pass to runabfe.py with
--apbs-correction-kj-mol.  APBS is an external electrostatic/continuum term; it
does not replace the separate LJ dispersion/tail correction.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


KJ_PER_KCAL = 4.184
DEFAULT_DIME = (161, 161, 161)
MOLECULE_ORDER = ("complex", "receptor", "ligand")


@dataclass(frozen=True)
class PQRGeometry:
    path: Path
    natoms: int
    center: Tuple[float, float, float]
    lengths: Tuple[float, float, float]
    total_charge: float
    min_xyz: Tuple[float, float, float]
    max_xyz: Tuple[float, float, float]


def _as_floats(values: Sequence[str]) -> Tuple[float, ...]:
    return tuple(float(v) for v in values)


def read_pqr_geometry(path: Path) -> PQRGeometry:
    """Read basic geometry from a PQR file.

    The parser uses the last five whitespace-separated fields as
    x, y, z, charge, radius, which is robust to both chain-ID and no-chain-ID
    PQR variants.
    """

    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    charges: List[float] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            fields = line.split()
            if len(fields) < 10:
                continue
            try:
                x, y, z, q, _radius = _as_floats(fields[-5:])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
            zs.append(z)
            charges.append(q)

    if not xs:
        raise ValueError(f"No ATOM/HETATM PQR records with coordinates found in {path}")

    min_xyz = (min(xs), min(ys), min(zs))
    max_xyz = (max(xs), max(ys), max(zs))
    center = tuple((lo + hi) * 0.5 for lo, hi in zip(min_xyz, max_xyz))
    lengths = tuple(hi - lo for lo, hi in zip(min_xyz, max_xyz))
    return PQRGeometry(
        path=path,
        natoms=len(xs),
        center=center,
        lengths=lengths,
        total_charge=sum(charges),
        min_xyz=min_xyz,
        max_xyz=max_xyz,
    )


def _fmt3(values: Sequence[float]) -> str:
    return " ".join(f"{v:.3f}" for v in values)


def _validate_dime(dime: Sequence[int]) -> Tuple[int, int, int]:
    if len(dime) != 3:
        raise ValueError("--dime needs exactly three integers")
    out = tuple(int(v) for v in dime)
    if any(v < 33 for v in out):
        raise ValueError("APBS grid dimensions should be at least 33 in each direction")
    return out


def _padded_lengths(lengths: Sequence[float], padding: float, minimum: float) -> Tuple[float, float, float]:
    return tuple(max(float(v) + 2.0 * padding, minimum) for v in lengths)


def write_apbs_input(
    name: str,
    pqr_path: Path,
    output_path: Path,
    dime: Sequence[int],
    center: Sequence[float],
    coarse_lengths: Sequence[float],
    fine_lengths: Sequence[float],
    pdie: float,
    sdie: float,
    temperature: float,
    ion_strength: float,
    ion_radius: float,
    boundary: str,
    write_potential: bool,
) -> None:
    ion_lines = ""
    if ion_strength > 0.0:
        ion_lines = (
            f"  ion charge 1 conc {ion_strength:.6g} radius {ion_radius:.3f}\n"
            f"  ion charge -1 conc {ion_strength:.6g} radius {ion_radius:.3f}\n"
        )

    write_line = ""
    if write_potential:
        write_line = f"  write pot dx {name}_potential\n"

    text = f"""read
  mol pqr {pqr_path.as_posix()}
end

elec name {name}
  mg-auto
  dime {int(dime[0])} {int(dime[1])} {int(dime[2])}
  cglen {_fmt3(coarse_lengths)}
  fglen {_fmt3(fine_lengths)}
  cgcent {_fmt3(center)}
  fgcent {_fmt3(center)}
  mol 1
  lpbe
  bcfl {boundary}
  pdie {pdie:.6g}
  sdie {sdie:.6g}
  chgm spl2
  srfm smol
  srad 1.400
  swin 0.300
  sdens 10.0
  temp {temperature:.6g}
{ion_lines}  calcenergy total
  calcforce no
{write_line}end

quit
"""
    output_path.write_text(text, encoding="utf-8")


def prepare(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_pqr_paths = {
        "complex": Path(args.complex_pqr).resolve(),
        "receptor": Path(args.receptor_pqr).resolve(),
        "ligand": Path(args.ligand_pqr).resolve(),
    }
    for label, path in source_pqr_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} PQR not found: {path}")

    pqr_paths: Dict[str, Path] = {}
    for label, source_path in source_pqr_paths.items():
        local_path = (out_dir / f"{label}.pqr").resolve()
        if source_path != local_path:
            shutil.copyfile(source_path, local_path)
        pqr_paths[label] = local_path

    geometries = {label: read_pqr_geometry(path) for label, path in pqr_paths.items()}
    dime = _validate_dime(args.dime)

    common_center: Optional[Tuple[float, float, float]] = None
    common_lengths: Optional[Tuple[float, float, float]] = None
    if args.common_grid:
        complex_geom = geometries["complex"]
        common_center = complex_geom.center
        common_lengths = complex_geom.lengths

    manifest: Dict[str, object] = {
        "schema": "apbs_correction_manifest_v1",
        "note": (
            "External APBS electrostatic/continuum correction. "
            "This value is not an LJ dispersion/tail correction."
        ),
        "settings": {
            "dime": list(dime),
            "coarse_padding_A": args.coarse_padding,
            "fine_padding_A": args.fine_padding,
            "minimum_length_A": args.minimum_length,
            "common_grid": bool(args.common_grid),
            "pdie": args.pdie,
            "sdie": args.sdie,
            "temperature_K": args.temperature,
            "ion_strength_M": args.ion_strength,
            "ion_radius_A": args.ion_radius,
            "boundary": args.boundary,
        },
        "molecules": {},
    }

    for label in MOLECULE_ORDER:
        geom = geometries[label]
        center = common_center if common_center is not None else geom.center
        lengths = common_lengths if common_lengths is not None else geom.lengths
        coarse_lengths = _padded_lengths(lengths, args.coarse_padding, args.minimum_length)
        fine_lengths = _padded_lengths(lengths, args.fine_padding, args.minimum_length)

        input_path = out_dir / f"{label}.in"
        log_path = out_dir / f"{label}.log"
        write_apbs_input(
            name=label,
            pqr_path=Path(f"{label}.pqr"),
            output_path=input_path,
            dime=dime,
            center=center,
            coarse_lengths=coarse_lengths,
            fine_lengths=fine_lengths,
            pdie=args.pdie,
            sdie=args.sdie,
            temperature=args.temperature,
            ion_strength=args.ion_strength,
            ion_radius=args.ion_radius,
            boundary=args.boundary,
            write_potential=args.write_potential,
        )
        manifest["molecules"][label] = {
            "pqr": str(geom.path),
            "source_pqr": str(source_pqr_paths[label]),
            "input": str(input_path.resolve()),
            "log": str(log_path.resolve()),
            "natoms": geom.natoms,
            "total_charge_e": geom.total_charge,
            "center_A": list(center),
            "raw_lengths_A": list(geom.lengths),
            "coarse_lengths_A": list(coarse_lengths),
            "fine_lengths_A": list(fine_lengths),
            "min_xyz_A": list(geom.min_xyz),
            "max_xyz_A": list(geom.max_xyz),
        }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote APBS inputs and manifest to {out_dir}")
    return 0


def _load_manifest(out_dir: Path) -> Dict[str, object]:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def run_apbs(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    manifest = _load_manifest(out_dir)
    molecules = manifest.get("molecules", {})
    if not isinstance(molecules, dict):
        raise ValueError("manifest.json has no molecule table")

    for label in MOLECULE_ORDER:
        item = molecules.get(label)
        if not isinstance(item, dict):
            raise ValueError(f"manifest.json missing molecule entry: {label}")
        input_path = Path(str(item["input"]))
        log_path = Path(str(item["log"]))
        cmd = [args.apbs_bin, str(input_path)]
        print("Running", " ".join(cmd))
        completed = subprocess.run(
            cmd,
            cwd=str(out_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"APBS failed for {label}; see {log_path}")
    return 0


ENERGY_PATTERNS = (
    re.compile(
        r"Global\s+net\s+ELEC\s+energy\s*=\s*([-+0-9.eE]+)\s*(kJ/mol|kcal/mol|kT)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:Total\s+)?(?:electrostatic|ELEC)\s+energy\s*[:=]\s*([-+0-9.eE]+)\s*(kJ/mol|kcal/mol|kT)?",
        re.IGNORECASE,
    ),
)


def parse_apbs_energy(log_path: Path) -> Tuple[float, str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches: List[Tuple[float, str]] = []
    for pattern in ENERGY_PATTERNS:
        for match in pattern.finditer(text):
            value = float(match.group(1))
            unit = (match.group(2) or "kJ/mol").lower()
            matches.append((value, unit))
    if not matches:
        raise ValueError(f"Could not find an APBS electrostatic energy in {log_path}")

    value, unit = matches[-1]
    if unit == "kcal/mol":
        return value * KJ_PER_KCAL, unit
    if unit == "kj/mol":
        return value, unit
    if unit == "kt":
        raise ValueError(f"APBS energy in kT needs explicit conversion; log={log_path}")
    return value, unit


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite: {value}")
    return value


def collect(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    manifest = _load_manifest(out_dir)
    molecules = manifest.get("molecules", {})
    if not isinstance(molecules, dict):
        raise ValueError("manifest.json has no molecule table")

    energies: Dict[str, float] = {}
    source_units: Dict[str, str] = {}
    for label in MOLECULE_ORDER:
        item = molecules.get(label)
        if not isinstance(item, dict):
            raise ValueError(f"manifest.json missing molecule entry: {label}")
        log_path = Path(str(item["log"]))
        energy_kj, source_unit = parse_apbs_energy(log_path)
        energies[label] = _finite(energy_kj, label)
        source_units[label] = source_unit

    delta_kj = energies["complex"] - energies["receptor"] - energies["ligand"]
    result = {
        "schema": "apbs_correction_result_v1",
        "definition": "G_APBS(complex) - G_APBS(receptor) - G_APBS(ligand)",
        "delta_G_apbs_kJ_mol": delta_kj,
        "delta_G_apbs_kcal_mol": delta_kj / KJ_PER_KCAL,
        "component_energies_kJ_mol": energies,
        "source_units": source_units,
        "manifest": str((out_dir / "manifest.json").resolve()),
        "runabfe_args": [
            "--apbs-correction-kj-mol",
            f"{delta_kj:.10g}",
            "--apbs-correction-note",
            f"APBS correction from {(out_dir / 'apbs_correction.json').resolve()}",
        ],
        "warning": (
            "Use only if this APBS cycle is part of the validated thermodynamic "
            "protocol. This is not an LJ long-range dispersion correction."
        ),
    }

    output_path = out_dir / "apbs_correction.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, run, and collect an external APBS correction term."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_p = sub.add_parser("prepare", help="Generate APBS input files from PQR files")
    prepare_p.add_argument("--complex-pqr", required=True)
    prepare_p.add_argument("--receptor-pqr", required=True)
    prepare_p.add_argument("--ligand-pqr", required=True)
    prepare_p.add_argument("--out-dir", default="output/apbs")
    prepare_p.add_argument("--dime", nargs=3, type=int, default=DEFAULT_DIME)
    prepare_p.add_argument("--coarse-padding", type=float, default=40.0, help="Angstrom")
    prepare_p.add_argument("--fine-padding", type=float, default=20.0, help="Angstrom")
    prepare_p.add_argument("--minimum-length", type=float, default=40.0, help="Angstrom")
    prepare_p.add_argument("--common-grid", action="store_true", help="Use complex grid for all molecules")
    prepare_p.add_argument("--pdie", type=float, default=2.0)
    prepare_p.add_argument("--sdie", type=float, default=78.54)
    prepare_p.add_argument("--temperature", type=float, default=300.0)
    prepare_p.add_argument("--ion-strength", type=float, default=0.150, help="M")
    prepare_p.add_argument("--ion-radius", type=float, default=2.0, help="Angstrom")
    prepare_p.add_argument("--boundary", default="sdh", choices=("sdh", "mdh", "focus", "zero"))
    prepare_p.add_argument("--write-potential", action="store_true")
    prepare_p.set_defaults(func=prepare)

    run_p = sub.add_parser("run", help="Run APBS for inputs listed in manifest.json")
    run_p.add_argument("--out-dir", default="output/apbs")
    run_p.add_argument("--apbs-bin", default="apbs")
    run_p.set_defaults(func=run_apbs)

    collect_p = sub.add_parser("collect", help="Parse APBS logs and write correction JSON")
    collect_p.add_argument("--out-dir", default="output/apbs")
    collect_p.set_defaults(func=collect)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
