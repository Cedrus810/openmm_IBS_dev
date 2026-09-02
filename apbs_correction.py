#!/usr/bin/env python
"""Rocklin finite-size electrostatic correction for membrane ABFE.

This helper implements the APBS/RIP part of Wu and Biggin, JCTC 2022,
DOI 10.1021/acs.jctc.1c01251, following the public RocklinC reference
implementation.  It is deliberately *not* a continuum solvation cycle.

For each representative complex snapshot, APBS produces three potential grids:

* protein_RIP_het: receptor charged; ligand retained as an uncharged excluded volume;
* ligand_RIP_het: ligand charged; receptor retained as an uncharged excluded volume;
* ligand_RIP_hom: the same ligand field in the homogeneous reference.

Their integrated potentials enter the Rocklin NET/USV, RIP, EMP, and DSC terms.
For lipid bilayers, the heterogeneous calculations can read explicit continuum
dielectric and fixed-charge maps.  A dielectric slab alone is intentionally not
accepted as a membrane production model because the paper shows that omitted
lipid head-group charge gives a material residual error.

This correction applies only to charge-changing alchemical calculations that
used a neutralizing plasma.  Wu and Biggin recommend a co-alchemical ion with
charge transfer for lipid bilayers; do not add this post hoc correction when the
simulation already used that neutral route.  Neutral ligands also require no
Rocklin correction.

This is orthogonal to Lennard-Jones long-range dispersion correction (LRC).
LJ tail corrections must be handled in the explicit alchemical force/thermodynamic
cycle, not in this APBS electrostatic post-processing script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


KJ_PER_KCAL = 4.184
R_KJ_PER_MOL_K = 0.00831446261815324
COULOMB_FACTOR_KJ_NM_PER_MOL_E2 = 138.93545585
XI_LS = -2.837297
XI_CB = -2.38008
CHARGE_TOLERANCE_E = 0.02
ROCKLIN_SOLUTE_DIELECTRIC = 1.0
ROCKLIN_APBS_ION_STRENGTH_M = 0.0

WATER_MODELS = {
    "tip3p": {"epsilon_s": 97.0, "gamma_s_e_nm2": 2.0 * 0.417 * 0.09572**2},
    "tip4p": {"epsilon_s": 51.0, "gamma_s_e_nm2": 2.0 * 0.52 * 0.09572**2 - 1.04 * 0.015**2},
}


@dataclass(frozen=True)
class PQRGeometry:
    path: Path
    natoms: int
    total_charge_e: float
    center_a: Tuple[float, float, float]


@dataclass(frozen=True)
class DXGeometry:
    path: Path
    counts: Tuple[int, int, int]
    deltas_a: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
    origin_a: Tuple[float, float, float]

    @property
    def lengths_a(self) -> Tuple[float, float, float]:
        return tuple(
            (self.counts[i] - 1) * math.sqrt(sum(value * value for value in self.deltas_a[i]))
            for i in range(3)
        )

    @property
    def voxel_volume_a3(self) -> float:
        a, b, c = self.deltas_a
        return abs(
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )


def _fmt3(values: Sequence[float]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def _absolute(path: Path) -> Path:
    """Make a path absolute without resolving Windows network/reparse points."""
    path = path.expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_pqr_atom(line: str) -> Optional[Tuple[List[str], Tuple[float, float, float], float]]:
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return None
    fields = line.split()
    if len(fields) < 10:
        return None
    try:
        xyz = tuple(float(value) for value in fields[-5:-2])
        charge = float(fields[-2])
    except ValueError:
        return None
    return fields, xyz, charge


def read_pqr_geometry(path: Path) -> PQRGeometry:
    natoms = 0
    total_charge = 0.0
    xyz: List[Tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_pqr_atom(line)
            if parsed is None:
                continue
            _fields, position, charge = parsed
            natoms += 1
            total_charge += charge
            xyz.append(position)
    if not xyz:
        raise ValueError(f"No valid ATOM/HETATM PQR records found in {path}")
    center = tuple(sum(point[i] for point in xyz) / len(xyz) for i in range(3))
    return PQRGeometry(path=path, natoms=natoms, total_charge_e=total_charge, center_a=center)


def _write_charge_masked_pqr(destination: Path, charged: Path, uncharged: Path) -> None:
    """Write both components while retaining radii/coordinates as excluded volumes."""
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        for path, zero_charge in ((charged, False), (uncharged, True)):
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    parsed = _parse_pqr_atom(line)
                    if parsed is None:
                        continue
                    fields, _position, _charge = parsed
                    if zero_charge:
                        fields[-2] = "0.000000"
                    output.write(" ".join(fields) + "\n")
        output.write("END\n")


def _copy_pqr(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"PQR file not found: {source}")
    shutil.copyfile(source, destination)


def _parse_dx_geometry(path: Path) -> DXGeometry:
    if not path.is_file():
        raise FileNotFoundError(f"OpenDX map not found: {path}")
    counts: Optional[Tuple[int, int, int]] = None
    origin: Optional[Tuple[float, float, float]] = None
    deltas: List[Tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 8 and fields[:5] == ["object", "1", "class", "gridpositions", "counts"]:
                counts = (int(fields[5]), int(fields[6]), int(fields[7]))
            elif len(fields) == 4 and fields[0] == "origin":
                origin = (float(fields[1]), float(fields[2]), float(fields[3]))
            elif len(fields) == 4 and fields[0] == "delta" and len(deltas) < 3:
                deltas.append((float(fields[1]), float(fields[2]), float(fields[3])))
            if counts is not None and origin is not None and len(deltas) == 3:
                break
    if counts is None or origin is None or len(deltas) != 3:
        raise ValueError(f"Could not parse grid geometry from {path}")
    if min(counts) < 2:
        raise ValueError(f"OpenDX map has invalid counts in {path}: {counts}")
    return DXGeometry(path=path, counts=counts, origin_a=origin, deltas_a=tuple(deltas))


def _validate_map_geometry(
    maps: Iterable[DXGeometry],
    box_a: Sequence[float],
    grid_center_a: Sequence[float],
) -> None:
    maps = list(maps)
    first = maps[0]
    for item in maps[1:]:
        if item.counts != first.counts or item.origin_a != first.origin_a or item.deltas_a != first.deltas_a:
            raise ValueError("All dielectric and lipid-charge maps must use one identical OpenDX grid")
    for actual, expected in zip(first.lengths_a, box_a):
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=0.02):
            raise ValueError(
                f"Map extent {first.lengths_a} A does not match --box {tuple(box_a)} A. "
                "Use the mean MD unit-cell dimensions and maps defined on that same cell."
            )
    map_center = tuple(
        first.origin_a[i] + 0.5 * (first.counts[i] - 1) * first.deltas_a[i][i]
        for i in range(3)
    )
    if any(not math.isclose(actual, expected, rel_tol=0.0, abs_tol=0.02) for actual, expected in zip(map_center, grid_center_a)):
        raise ValueError(
            f"Map center {map_center} A does not match --grid-center {tuple(grid_center_a)} A. "
            "PQR snapshots and maps must share one coordinate frame."
        )


def _stage_maps(args: argparse.Namespace, out_dir: Path) -> Dict[str, object]:
    diel_values = [args.diel_map_x, args.diel_map_y, args.diel_map_z]
    have_diel = [value is not None for value in diel_values]
    if any(have_diel) and not all(have_diel):
        raise ValueError("Provide all of --diel-map-x, --diel-map-y, --diel-map-z, or none")
    if any(have_diel) != (args.lipid_charge_map is not None):
        raise ValueError(
            "Membrane Rocklin mode requires both dielectric maps and --lipid-charge-map. "
            "A dielectric-only slab omits the lipid head-group electrostatic potential."
        )
    if not any(have_diel):
        return {"mode": "homogeneous", "maps": {}}

    named_sources = {
        "diel_x": _absolute(Path(args.diel_map_x)),
        "diel_y": _absolute(Path(args.diel_map_y)),
        "diel_z": _absolute(Path(args.diel_map_z)),
        "lipid_charge": _absolute(Path(args.lipid_charge_map)),
    }
    staged: Dict[str, object] = {}
    geometries: List[DXGeometry] = []
    for key, source in named_sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"APBS map file not found: {source}")
        destination = out_dir / f"membrane_{key}.dx"
        if source != destination:
            shutil.copyfile(source, destination)
        geometry = _parse_dx_geometry(destination)
        geometries.append(geometry)
        staged[key] = {"path": str(destination), "sha256": _file_sha256(destination)}
    _validate_map_geometry(geometries, args.box, args.grid_center)
    return {"mode": "membrane_continuum", "maps": staged}


def _sanitize_label(label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise ValueError(f"Invalid snapshot label {label!r}; use only letters, numbers, '.', '_' or '-'")
    return label


def _load_snapshot_specs(args: argparse.Namespace) -> List[Dict[str, Path]]:
    if args.snapshots_json is None:
        required = (args.complex_pqr, args.receptor_pqr, args.ligand_pqr)
        if any(value is None for value in required):
            raise ValueError(
                "Provide --complex-pqr, --receptor-pqr, and --ligand-pqr, or use --snapshots-json"
            )
        return [{
            "label": "snapshot_000",
            "complex": _absolute(Path(args.complex_pqr)),
            "receptor": _absolute(Path(args.receptor_pqr)),
            "ligand": _absolute(Path(args.ligand_pqr)),
        }]

    payload = json.loads(Path(args.snapshots_json).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("--snapshots-json must be a non-empty JSON list")
    specs: List[Dict[str, Path]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError("Each --snapshots-json entry must be an object")
        label = _sanitize_label(str(item.get("label", f"snapshot_{index:03d}")))
        try:
            spec = {
                "label": label,
                "complex": _absolute(Path(str(item["complex_pqr"]))),
                "receptor": _absolute(Path(str(item["receptor_pqr"]))),
                "ligand": _absolute(Path(str(item["ligand_pqr"]))),
            }
        except KeyError as exc:
            raise ValueError("Each snapshot needs complex_pqr, receptor_pqr, and ligand_pqr") from exc
        specs.append(spec)
    if len({str(spec["label"]) for spec in specs}) != len(specs):
        raise ValueError("Snapshot labels must be unique")
    return specs


def _write_elec_block(
    name: str,
    mol_index: int,
    dime: Sequence[int],
    box_a: Sequence[float],
    center_a: Sequence[float],
    temperature_k: float,
    solute_dielectric: float,
    exterior_dielectric: float,
    use_membrane_maps: bool,
) -> str:
    lines = [
        f"elec name {name}",
        "  mg-manual",
        f"  dime {int(dime[0])} {int(dime[1])} {int(dime[2])}",
        f"  glen {_fmt3(box_a)}",
        f"  gcent {_fmt3(center_a)}",
        f"  mol {mol_index}",
        "  lpbe",
        "  bcfl mdh",
        # Rocklin/RIP APBS grids use an unscreened reference with solute
        # dielectric fixed at 1.0.  Keep this as protocol state, not a CLI knob.
        f"  pdie {solute_dielectric:.8g}",
        f"  sdie {exterior_dielectric:.8g}",
        "  chgm spl4",
        "  srfm smol",
        "  srad 1.4",
        "  swin 0.3",
        "  sdens 40.0",
        f"  temp {temperature_k:.8g}",
    ]
    if use_membrane_maps:
        lines.extend(["  usemap charge 1", "  usemap diel 1"])
    lines.extend([
        "  calcenergy no",
        "  calcforce no",
        f"  write pot dx {name}",
        "end",
    ])
    return "\n".join(lines)


def _write_apbs_input(
    path: Path,
    protein_only: Path,
    ligand_in_protein: Path,
    ligand_only: Path,
    dime: Sequence[int],
    box_a: Sequence[float],
    center_a: Sequence[float],
    temperature_k: float,
    epsilon_s: float,
    maps: Dict[str, object],
) -> None:
    mode = str(maps["mode"])
    map_table = maps["maps"]
    read_lines = [
        "read",
        f"  mol pqr {protein_only.as_posix()}",
        f"  mol pqr {ligand_in_protein.as_posix()}",
        f"  mol pqr {ligand_only.as_posix()}",
    ]
    if mode == "membrane_continuum":
        assert isinstance(map_table, dict)
        read_lines.extend([
            f"  charge dx {Path(str(map_table['lipid_charge']['path'])).as_posix()}",
            "  diel dx "
            f"{Path(str(map_table['diel_x']['path'])).as_posix()} "
            f"{Path(str(map_table['diel_y']['path'])).as_posix()} "
            f"{Path(str(map_table['diel_z']['path'])).as_posix()}",
        ])
    read_lines.append("end")
    body = ["\n".join(read_lines)]
    body.append(_write_elec_block(
        "protein_RIP_het", 1, dime, box_a, center_a, temperature_k,
        ROCKLIN_SOLUTE_DIELECTRIC, epsilon_s,
        use_membrane_maps=(mode == "membrane_continuum"),
    ))
    body.append(_write_elec_block(
        "ligand_RIP_het", 2, dime, box_a, center_a, temperature_k,
        ROCKLIN_SOLUTE_DIELECTRIC, epsilon_s,
        use_membrane_maps=(mode == "membrane_continuum"),
    ))
    body.append(_write_elec_block(
        "ligand_RIP_hom", 3, dime, box_a, center_a, temperature_k,
        ROCKLIN_SOLUTE_DIELECTRIC, ROCKLIN_SOLUTE_DIELECTRIC,
        use_membrane_maps=False,
    ))
    path.write_text("\n\n".join(body + ["quit", ""]), encoding="utf-8")


def _check_charge(name: str, observed: float, expected: float) -> None:
    if not math.isclose(observed, expected, abs_tol=CHARGE_TOLERANCE_E, rel_tol=0.0):
        raise ValueError(
            f"{name} PQR charge is {observed:.6f} e, but the declared alchemical charge is "
            f"{expected:.6f} e.  Use PQR files matching the simulated protonation state."
        )


def prepare(args: argparse.Namespace) -> int:
    if args.charge_treatment != "neutralizing-plasma":
        raise ValueError(
            "This Rocklin post-processing correction is only for neutralizing-plasma simulations. "
            "The supplied paper recommends co-alchemical charge transfer for bilayers; do not double-correct it."
        )
    if abs(args.ligand_net_charge) < 1.0e-8:
        raise ValueError(
            "The ligand net-charge change is zero. Rocklin finite-size correction is not applicable; "
            "leave --apbs-correction-kj-mol at 0.0."
        )
    if args.n_solvent <= 0:
        raise ValueError("--n-solvent must be the positive mean number of explicit water molecules")
    if len(args.box) != 3 or any(value <= 0.0 for value in args.box):
        raise ValueError("--box needs three positive mean MD cell lengths in Angstrom")
    if len(args.dime) != 3 or any(value < 33 for value in args.dime):
        raise ValueError("--dime needs three integers >= 33")

    out_dir = _absolute(Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    model = WATER_MODELS[args.water_model]
    maps = _stage_maps(args, out_dir)
    specs = _load_snapshot_specs(args)
    snapshots: List[Dict[str, object]] = []
    for spec in specs:
        label = str(spec["label"])
        for key in ("complex", "receptor", "ligand"):
            if not spec[key].is_file():
                raise FileNotFoundError(f"{key} PQR not found for {label}: {spec[key]}")
        frame_dir = out_dir / label
        frame_dir.mkdir(parents=True, exist_ok=True)
        complex_pqr = frame_dir / "complex.pqr"
        receptor_pqr = frame_dir / "receptor.pqr"
        ligand_pqr = frame_dir / "ligand.pqr"
        _copy_pqr(spec["complex"], complex_pqr)
        _copy_pqr(spec["receptor"], receptor_pqr)
        _copy_pqr(spec["ligand"], ligand_pqr)
        complex_geom = read_pqr_geometry(complex_pqr)
        receptor_geom = read_pqr_geometry(receptor_pqr)
        ligand_geom = read_pqr_geometry(ligand_pqr)
        _check_charge("ligand", ligand_geom.total_charge_e, args.ligand_net_charge)
        _check_charge("receptor", receptor_geom.total_charge_e, args.receptor_net_charge)
        _check_charge(
            "complex", complex_geom.total_charge_e, args.receptor_net_charge + args.ligand_net_charge
        )
        if complex_geom.natoms != receptor_geom.natoms + ligand_geom.natoms:
            raise ValueError(
                f"{label}: complex PQR has {complex_geom.natoms} atoms but receptor + ligand has "
                f"{receptor_geom.natoms + ligand_geom.natoms}. The three PQRs must describe one complex snapshot."
            )
        protein_only = frame_dir / "protein_only.pqr"
        ligand_in_protein = frame_dir / "ligand_in_protein.pqr"
        _write_charge_masked_pqr(protein_only, receptor_pqr, ligand_pqr)
        _write_charge_masked_pqr(ligand_in_protein, ligand_pqr, receptor_pqr)
        input_path = frame_dir / "apbs.in"
        _write_apbs_input(
            input_path,
            protein_only,
            ligand_in_protein,
            ligand_pqr,
            args.dime,
            args.box,
            args.grid_center,
            args.temperature,
            float(model["epsilon_s"]),
            maps,
        )
        snapshots.append({
            "label": label,
            "input": str(input_path),
            "log": str(frame_dir / "apbs.log"),
            "pqr": {
                "complex": str(complex_pqr),
                "receptor": str(receptor_pqr),
                "ligand": str(ligand_pqr),
                "protein_only": str(protein_only),
                "ligand_in_protein": str(ligand_in_protein),
            },
            "potential_grids": {
                "protein_RIP_het": str(frame_dir / "protein_RIP_het.dx"),
                "ligand_RIP_het": str(frame_dir / "ligand_RIP_het.dx"),
                "ligand_RIP_hom": str(frame_dir / "ligand_RIP_hom.dx"),
            },
            "charges_e": {
                "complex": complex_geom.total_charge_e,
                "receptor": receptor_geom.total_charge_e,
                "ligand": ligand_geom.total_charge_e,
            },
        })

    warnings: List[str] = []
    if len(snapshots) < 5:
        warnings.append(
            "Fewer than 5 snapshots were supplied. This is a diagnostic calculation, not a production "
            "ensemble average; the paper averages APBS potentials across representative MD snapshots."
        )
    if maps["mode"] == "homogeneous":
        warnings.append(
            "No membrane maps supplied: this is the homogeneous Rocklin protocol, not the lipid-continuum variant."
        )
    manifest = {
        "schema": "rocklin_membrane_apbs_manifest_v1",
        "citation": "Wu and Biggin, JCTC 2022, DOI: 10.1021/acs.jctc.1c01251",
        "definition": "Delta G_corr = Delta G_NET+USV + Delta G_RIP + Delta G_EMP + Delta G_DSC",
        "method": "Rocklin finite-size correction using APBS residual integrated potentials",
        "settings": {
            "charge_treatment": args.charge_treatment,
            "box_A": list(args.box),
            "grid_center_A": list(args.grid_center),
            "dime": list(args.dime),
            "temperature_K": args.temperature,
            "solute_dielectric": ROCKLIN_SOLUTE_DIELECTRIC,
            "apbs_mobile_ion_strength_M": ROCKLIN_APBS_ION_STRENGTH_M,
            "water_model": args.water_model,
            "epsilon_s": model["epsilon_s"],
            "gamma_s_e_nm2": model["gamma_s_e_nm2"],
            "ligand_net_charge_e": args.ligand_net_charge,
            "receptor_net_charge_e": args.receptor_net_charge,
            "n_solvent_mean": args.n_solvent,
            "maps": maps,
            "warnings": warnings,
        },
        "snapshots": snapshots,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(snapshots)} Rocklin APBS job(s) to {out_dir}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


def _load_manifest(out_dir: Path) -> Dict[str, object]:
    path = out_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rocklin_membrane_apbs_manifest_v1":
        raise ValueError(
            "This is not a Rocklin membrane manifest. Previous apbs_correction.py solvation-cycle "
            "manifests must not be collected as finite-size corrections. Run prepare again."
        )
    return manifest


def run_apbs(args: argparse.Namespace) -> int:
    out_dir = _absolute(Path(args.out_dir))
    manifest = _load_manifest(out_dir)
    snapshots = manifest.get("snapshots")
    if not isinstance(snapshots, list):
        raise ValueError("manifest.json has no snapshots")
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            raise ValueError("Invalid snapshot entry in manifest")
        input_path = _absolute(Path(str(snapshot["input"])))
        log_path = _absolute(Path(str(snapshot["log"])))
        print("Running", args.apbs_bin, input_path)
        completed = subprocess.run(
            [args.apbs_bin, str(input_path)],
            cwd=str(input_path.parent),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"APBS failed for {snapshot.get('label', input_path.parent.name)}; see {log_path}")
    return 0


def _read_dx_values(path: Path) -> Tuple[DXGeometry, List[float]]:
    geometry = _parse_dx_geometry(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    header = re.search(
        r"object\s+3\s+class\s+array\s+type\s+\S+\s+rank\s+0\s+items\s+(\d+)\s+data\s+follows",
        text,
        flags=re.IGNORECASE,
    )
    if header is None:
        raise ValueError(f"Could not locate scalar data in OpenDX potential grid {path}")
    expected = int(header.group(1))
    values: List[float] = []
    for token in re.finditer(r"[-+]?(?:\d+\.?(?:\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text[header.end():]):
        values.append(float(token.group(0)))
        if len(values) == expected:
            break
    if len(values) != expected:
        raise ValueError(f"OpenDX potential grid {path} has {len(values)} values; expected {expected}")
    if expected != math.prod(geometry.counts):
        raise ValueError(f"OpenDX grid item count disagrees with grid dimensions in {path}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"OpenDX potential grid contains a non-finite value: {path}")
    return geometry, values


def _potential_to_integrated_potential_kj_nm3_per_mol_e(
    path: Path,
    temperature_k: float,
) -> Tuple[float, Dict[str, object]]:
    """Follow RocklinC: mean APBS potential (kT/e) times the grid's geometric volume."""
    geometry, values = _read_dx_values(path)
    average_kT_per_e = statistics.fmean(values)
    # 🔑 之前这里用 len(values)（= nx*ny*nz，网格点总数）乘 voxel_volume_a3，
    # 但一个 nx*ny*nz 的 OpenDX 网格实际围出的几何体积是 (nx-1)*(ny-1)*(nz-1)
    # 个体素——跟 DXGeometry.lengths_a 算边长时用的 (counts[i]-1) 是同一个
    # 约定（见上面 lengths_a 属性）。用点数而不是体素数会系统性高估体积
    # （每边多算了一个格点厚度），且这个体积从未跟 --box 已经校验过的真实
    # 体积做过交叉核对。
    n_voxels = math.prod(count - 1 for count in geometry.counts)
    grid_volume_nm3 = geometry.voxel_volume_a3 * n_voxels / 1000.0
    integrated = average_kT_per_e * R_KJ_PER_MOL_K * temperature_k * grid_volume_nm3
    return integrated, {
        "path": str(path),
        "mean_potential_kT_per_e": average_kT_per_e,
        "grid_volume_nm3": grid_volume_nm3,
        "counts": list(geometry.counts),
        "voxel_volume_A3": geometry.voxel_volume_a3,
    }


def _rocklin_snapshot_correction(snapshot: Dict[str, object], settings: Dict[str, object]) -> Dict[str, object]:
    temperature_k = float(settings["temperature_K"])
    epsilon_s = float(settings["epsilon_s"])
    gamma_s = float(settings["gamma_s_e_nm2"])
    q_ligand = float(settings["ligand_net_charge_e"])
    q_receptor = float(settings["receptor_net_charge_e"])
    n_solvent = float(settings["n_solvent_mean"])
    grids = snapshot.get("potential_grids")
    if not isinstance(grids, dict):
        raise ValueError("Snapshot has no potential grid table")
    ip_raw, protein_meta = _potential_to_integrated_potential_kj_nm3_per_mol_e(
        Path(str(grids["protein_RIP_het"])), temperature_k
    )
    il_het_raw, ligand_het_meta = _potential_to_integrated_potential_kj_nm3_per_mol_e(
        Path(str(grids["ligand_RIP_het"])), temperature_k
    )
    il_hom_raw, ligand_hom_meta = _potential_to_integrated_potential_kj_nm3_per_mol_e(
        Path(str(grids["ligand_RIP_hom"])), temperature_k
    )
    volumes = [
        float(protein_meta["grid_volume_nm3"]),
        float(ligand_het_meta["grid_volume_nm3"]),
        float(ligand_hom_meta["grid_volume_nm3"]),
    ]
    volume_nm3 = volumes[0]
    if any(not math.isclose(value, volume_nm3, rel_tol=1.0e-10, abs_tol=1.0e-10) for value in volumes[1:]):
        raise ValueError("The three APBS potential grids must use one common grid volume")
    # 🔑 三个网格互相一致只说明它们彼此没漂移，不能说明这个体积本身算对了
    # （比如上面 (nx-1)(ny-1)(nz-1) 的体素数写错，三个网格会一起错但仍然
    # "互相一致"）。这里额外拿 prepare 阶段已经用 --box 校验过的真实
    # box_A（_validate_map_geometry 已经确认它跟地图的 lengths_a 一致）算出
    # 的体积做交叉核对，作为这个几何量的独立真值来源。
    box_a = settings.get("box_A")
    if isinstance(box_a, (list, tuple)) and len(box_a) == 3:
        box_a = [float(v) for v in box_a]
        box_volume_nm3 = math.prod(box_a) / 1000.0
        if not math.isclose(volume_nm3, box_volume_nm3, rel_tol=1.0e-3, abs_tol=1.0e-6):
            raise ValueError(
                f"{snapshot.get('label', 'snapshot')}: grid volume {volume_nm3:.6f} nm^3 derived from the "
                f"potential grid's own geometry does not match the --box-derived volume "
                f"{box_volume_nm3:.6f} nm^3 (box_A={tuple(box_a)} A). This grid should describe the same "
                "cell that --box was validated against in _validate_map_geometry; refusing to use a "
                "possibly-corrupted geometric volume in the finite-size correction below."
            )
        # 🔑 下面的 bq_* 自能修正用 XI_CB（立方晶格 Wigner 常数），NET 项用
        # XI_LS（同类立方晶格常数）+ characteristic_length_nm = volume_nm3**(1/3)
        # ——两者都是 Wu & Biggin 论文针对立方周期盒推导的常数/几何量。对强
        # 各向异性（比如膜体系常见的扁长方体）盒子直接套用同一组常数没有物理
        # 依据，且此前从未有任何检查或警告。这里在这类盒子上直接 fail closed，
        # 而不是不做任何提示地悄悄套用一个可能不适用的各向同性近似——把"是否
        # 要为各向异性盒重新推导/验证格点常数"这个决定留给启用该功能的人，
        # 而不是让它在数值上静默发生。10% 的比例阈值只是一个保守起点：真正
        # 近立方的溶剂盒（比如均匀水盒）通常远小于此。
        box_aspect_ratio = max(box_a) / min(box_a)
        if box_aspect_ratio > 1.10:
            raise ValueError(
                f"{snapshot.get('label', 'snapshot')}: --box {tuple(box_a)} A has aspect ratio "
                f"{box_aspect_ratio:.3f} (> 1.10). The finite-size correction below (XI_CB self-energy "
                "terms and the XI_LS NET term with characteristic_length_nm = volume_nm3**(1/3)) uses "
                "cubic-lattice Wigner constants, which are only valid for a near-cubic periodic cell. "
                "This has not been re-derived here for an anisotropic (e.g. membrane) box; using it "
                "anyway would silently produce an unvalidated number. Either run this correction on a "
                "near-cubic solvent cell, or implement and validate the anisotropic (tetragonal/"
                "orthorhombic) Madelung-constant analogue before removing this guard."
            )
    bq_protein = (-XI_CB * COULOMB_FACTOR_KJ_NM_PER_MOL_E2 / epsilon_s) * q_receptor * volume_nm3 ** (2.0 / 3.0)
    bq_ligand_het = (-XI_CB * COULOMB_FACTOR_KJ_NM_PER_MOL_E2 / epsilon_s) * q_ligand * volume_nm3 ** (2.0 / 3.0)
    bq_ligand_hom = (-XI_CB * COULOMB_FACTOR_KJ_NM_PER_MOL_E2) * q_ligand * volume_nm3 ** (2.0 / 3.0)
    ip = ip_raw - bq_protein
    il_het = il_het_raw - bq_ligand_het
    il_hom = il_hom_raw - bq_ligand_hom
    il_slv = il_het - il_hom
    charge_square_difference = (q_receptor + q_ligand) ** 2 - q_receptor**2
    characteristic_length_nm = volume_nm3 ** (1.0 / 3.0)
    dsc = -(2.0 * math.pi / 3.0) * COULOMB_FACTOR_KJ_NM_PER_MOL_E2 * gamma_s * q_ligand * n_solvent / volume_nm3
    net = -XI_LS * COULOMB_FACTOR_KJ_NM_PER_MOL_E2 * 0.5 * charge_square_difference / characteristic_length_nm
    net_usv = net / epsilon_s
    rip = ((ip + il_het) * (q_receptor + q_ligand) - ip * q_receptor) / volume_nm3
    r_ligand_squared_nm2 = il_slv / (
        (2.0 * math.pi / 3.0) * COULOMB_FACTOR_KJ_NM_PER_MOL_E2 * (1.0 - 1.0 / epsilon_s) * q_ligand
    )
    if not math.isfinite(r_ligand_squared_nm2) or r_ligand_squared_nm2 <= 0.0:
        raise ValueError(
            f"{snapshot.get('label', 'snapshot')}: empirical Rocklin radius squared is "
            f"{r_ligand_squared_nm2}; check potential-grid units, PQR charges, and map alignment."
        )
    r_ligand_nm = math.sqrt(r_ligand_squared_nm2)
    empirical = -(
        COULOMB_FACTOR_KJ_NM_PER_MOL_E2
        * 0.5
        * (16.0 * math.pi**2 / 45.0)
        * (1.0 - 1.0 / epsilon_s)
        * charge_square_difference
        * r_ligand_nm**5
        / volume_nm3**2
    )
    analytical = net_usv + rip + empirical
    total = analytical + dsc
    return {
        "label": snapshot.get("label"),
        "delta_G_correction_kJ_mol": total,
        "delta_G_correction_kcal_mol": total / KJ_PER_KCAL,
        "terms_kJ_mol": {
            "NET": net,
            "NET_plus_USV": net_usv,
            "RIP": rip,
            "EMP": empirical,
            "ANA": analytical,
            "DSC": dsc,
        },
        "integrated_potentials_kJ_nm3_per_mol_e": {
            "I_P": ip,
            "I_L_heterogeneous": il_het,
            "I_L_homogeneous": il_hom,
            "I_L_solvation_difference": il_slv,
        },
        "empirical_ligand_radius_nm": r_ligand_nm,
        "grid_metadata": {
            "protein_RIP_het": protein_meta,
            "ligand_RIP_het": ligand_het_meta,
            "ligand_RIP_hom": ligand_hom_meta,
        },
    }


def _mean_and_sem(values: Sequence[float]) -> Tuple[float, Optional[float], Optional[float]]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None, None
    stdev = statistics.stdev(values)
    return mean, stdev, stdev / math.sqrt(len(values))


def collect(args: argparse.Namespace) -> int:
    out_dir = _absolute(Path(args.out_dir))
    manifest = _load_manifest(out_dir)
    settings = manifest.get("settings")
    snapshots = manifest.get("snapshots")
    if not isinstance(settings, dict) or not isinstance(snapshots, list):
        raise ValueError("Malformed Rocklin manifest")
    per_snapshot = [_rocklin_snapshot_correction(snapshot, settings) for snapshot in snapshots if isinstance(snapshot, dict)]
    if len(per_snapshot) != len(snapshots):
        raise ValueError("Malformed snapshot entry in manifest")
    corrections = [float(item["delta_G_correction_kJ_mol"]) for item in per_snapshot]
    mean, stdev, sem = _mean_and_sem(corrections)
    warnings = list(settings.get("warnings", []))
    if len(per_snapshot) == 1:
        warnings.append("One APBS snapshot has no conformational uncertainty estimate; do not treat it as production-grade.")
    result = {
        "schema": "rocklin_membrane_apbs_result_v1",
        "citation": manifest["citation"],
        "definition": manifest["definition"],
        "delta_G_apbs_kJ_mol": mean,
        "delta_G_apbs_kcal_mol": mean / KJ_PER_KCAL,
        "snapshot_standard_deviation_kJ_mol": stdev,
        "snapshot_standard_error_kJ_mol": sem,
        "n_snapshots": len(per_snapshot),
        "per_snapshot": per_snapshot,
        "manifest": str(out_dir / "manifest.json"),
        "runabfe_args": [
            "--apbs-correction-kj-mol",
            f"{mean:.10g}",
            "--apbs-correction-note",
            "Rocklin membrane finite-size correction (Wu & Biggin 2022; APBS RIP ensemble average)",
        ],
        "warnings": warnings,
        "applicability": (
            "Valid only for a charge-changing neutralizing-plasma calculation. Do not apply to a "
            "co-alchemical-ion/charge-transfer calculation or a neutral ligand."
        ),
    }
    path = out_dir / "apbs_correction.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="APBS residual-integrated-potential Rocklin correction for membrane ABFE"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_p = sub.add_parser("prepare", help="Generate Rocklin/RIP APBS inputs")
    prepare_p.add_argument("--complex-pqr")
    prepare_p.add_argument("--receptor-pqr")
    prepare_p.add_argument("--ligand-pqr")
    prepare_p.add_argument(
        "--snapshots-json",
        help="JSON list with label, complex_pqr, receptor_pqr, ligand_pqr for an APBS ensemble",
    )
    prepare_p.add_argument("--out-dir", default="output/apbs")
    prepare_p.add_argument("--box", nargs=3, type=float, required=True, metavar=("LX", "LY", "LZ"), help="Mean MD box lengths in Angstrom")
    prepare_p.add_argument("--grid-center", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"), help="APBS/map center in Angstrom")
    prepare_p.add_argument("--dime", nargs=3, type=int, default=(257, 257, 257))
    prepare_p.add_argument("--temperature", type=float, default=300.0)
    prepare_p.add_argument("--water-model", choices=tuple(WATER_MODELS), default="tip3p")
    prepare_p.add_argument("--ligand-net-charge", type=float, required=True)
    prepare_p.add_argument("--receptor-net-charge", type=float, required=True)
    prepare_p.add_argument("--n-solvent", type=float, required=True, help="Mean explicit water molecule count")
    prepare_p.add_argument(
        "--charge-treatment",
        choices=("neutralizing-plasma", "co-alchemical-ion"),
        required=True,
        help="Electrostatic treatment actually used in the alchemical MD",
    )
    prepare_p.add_argument("--diel-map-x", default=None)
    prepare_p.add_argument("--diel-map-y", default=None)
    prepare_p.add_argument("--diel-map-z", default=None)
    prepare_p.add_argument("--lipid-charge-map", default=None, help="Fixed lipid charge-density OpenDX map")
    prepare_p.set_defaults(func=prepare)

    run_p = sub.add_parser("run", help="Run APBS for each snapshot in the manifest")
    run_p.add_argument("--out-dir", default="output/apbs")
    run_p.add_argument("--apbs-bin", default="apbs")
    run_p.set_defaults(func=run_apbs)

    collect_p = sub.add_parser("collect", help="Integrate RIP grids and assemble Rocklin terms")
    collect_p.add_argument("--out-dir", default="output/apbs")
    collect_p.set_defaults(func=collect)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
