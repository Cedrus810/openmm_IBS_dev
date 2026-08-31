# ABFE-IBS: an absolute binding free-energy workflow

[中文](README_cn.md) · [Concise entry](README.md) · [Documentation](docs/README.md)

ABFE-IBS is an OpenMM-based absolute binding free-energy workflow. It consumes GROMACS
`.gro/.top` systems, computes coupled-to-decoupled free energies for the protein–ligand
complex and bulk-solvent legs, and combines IBS sampling, MBAR/TMBAR analysis, a Boresch
restraint ledger, and explicit long-range corrections in a documented thermodynamic cycle.

This repository contains reusable workflow code, the Atenolol-rank11 reference system, and
the full record of production candidates, validation runs, failed experiments, and historical
decisions. Reference `output*`, `validation`, and `memtest` trees are evidence from specific
runs; they must not be reused as checkpoints for another molecular system.

## Scientific status (evidence through 2026-08-12)

The main software path includes GROMACS-to-OpenMM construction, complex and solvent legs,
dual-lambda decoupling, IBS warmup and frozen-bias production, MBAR/TMBAR, Boresch
attachment/release accounting, LJ long-range correction, caching, resume, and fail-closed
checks. Current reports identify the principal protocol versions as IBS v29, thermodynamic
path v21, LJ LRC v3, and WCA v2.

Independent repeats, the production seed ledger, parts of the validation matrix, and the
time-correlated uncertainty model are not yet closed. The repository is ready to report
software and method-development progress, but not a final publishable Atenolol binding free
energy.

| Result | Registry status | Final/citable? |
|---|---|---|
| `−23.1622 ± 2.5139 kJ/mol` from `output_lrc_fix` | `CANDIDATE` | No: no independent repeat, empty seed ledger, clipped Boresch `kr` |
| `+40.8362 ± 1.3178 kJ/mol` from historical `output` | `INVALIDATED` | No: opposite historical sign convention and diagnostic issues |
| `+16.00 ± 2.20 kJ/mol` from 2026-07-27 | `INVALIDATED` | No: stale and incorrect Boresch equilibrium geometry |

The current sign convention is:

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

A filename containing `final` does not make an artifact citable. Consult the
the result index and machine-readable registry (`RESULT_REGISTRY.csv`) in the
`Atenolol-rank11` workspace — neither ships with this engineering branch.

## Quick start

### 1. Environment

Core dependencies are Python 3.10+, OpenMM, NumPy, SciPy, MDTraj, and PyMBAR. Production GPU
runs additionally require a compatible CUDA or OpenCL stack. The supplied `environment.yml`
contains machine- and CUDA-specific choices and should be reviewed before use on another host.

```bash
python -c "import openmm, numpy, scipy, mdtraj, pymbar; print(openmm.__version__)"
```

If `python runabfe.py --help` fails with `No module named 'openmm'`, the active shell is not
in a runnable ABFE environment.

### 2. Inputs

A first build normally needs a GROMACS coordinate file, topology and include tree, ligand
residue name, GROMACS data path, and a new output directory. `abfe_config.json` is the
Atenolol reference configuration. It contains a machine-specific `gmx_path` and deliberately
frozen historical-run options; do not use it as an unreviewed template for a new system.

### 3. Run

Resume the reference calculation:

```bash
python runabfe.py --config abfe_config.json --ligand MOL --resume
```

Run with explicit inputs and a new output directory:

```bash
python runabfe.py \
  --config abfe_config.json \
  --gro /path/to/system.gro \
  --top /path/to/topol.top \
  --ligand LIG \
  --gmx-path /path/to/gromacs/share/gromacs/top \
  --output ./output_new_system \
  --boresch --boresch-source simple
```

Analyze existing energies and checkpoints only:

```bash
python runabfe.py --config abfe_config.json --ligand MOL --analyze-only
```

Read [Outputs and resume](docs/OUTPUTS_AND_RESUME.md) before using `--resume`, `--reset`,
or `--analyze-only`. Never point `--reset` at a protected historical evidence directory.

## Minimum verification after code changes

From the repository root:

```bash
./tests/run_offline_tests.sh
```

Focused test:

```bash
./tests/run_offline_tests.sh tests/test_core_physics_numerics.py
```

The offline entry excludes tests marked `needs_gpu`. Passing code tests establishes software
contracts; it does not automatically validate a new scientific result.

## Documentation map

| Need | Entry point |
|---|---|
| Index of historical material | [docs/HISTORY_LOG.md](docs/HISTORY_LOG.md) |
| Installation, inputs, and commands | [GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| Outputs, sign convention, and resume | [OUTPUTS_AND_RESUME.md](docs/OUTPUTS_AND_RESUME.md) |
| Troubleshooting | [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |
| Migration to another system | [MIGRATING_TO_A_NEW_SYSTEM.md](docs/MIGRATING_TO_A_NEW_SYSTEM.md) |
| Code maintenance and tests | [MAINTAINING.md](docs/MAINTAINING.md) |
| Current technical master report | 2026-08-12 technical draft in `Atenolol-rank11` (not in this branch) |
| Current global actions | [docs/TODO.md](docs/TODO.md) |
| Numerical and document conflicts | `CONFLICTS.md` in `Atenolol-rank11` (not in this branch) |

Most detailed tutorials are maintained in Chinese; their commands, paths, and status markers
remain directly usable.

## Repository map

```text
runabfe.py                  main CLI entry
abfe_core.py                systems and low-level physical components
abfe_pipeline.py            orchestration, quality gates, resume, result writing
ibs_engine.py               IBS, MBAR/TMBAR, Boresch, and LRC core
abfe_preoptimizer.py        lambda-path and window preoptimization
tests/                      regression and protocol tests
tools/                      diagnostics, explicit repairs, and plotting
docs/                       the single documentation set: guides, TODO, history log
plugins/                    native OpenMM plugin sources
```

## Evidence preservation

The following trees are under preservation hold during documentation and code curation:

```text
output/
output_lrc_fix/
output_lrc_fixonly-complex-charging/
validation/
solvent_box_scan/
memtest/output_membrane_100ns/
memtest/output_membrane_5ns/
```

New algorithms and protocols should write to a new output directory. Historical results,
failed approaches, and invalidated conclusions remain available for audit. See the
the immutability policy in `Atenolol-rank11`.

## Experimental branches and status vocabulary

DEXP, MACE, ORB, outer-lambda neural bases, membrane systems, and charge-transfer work have
code, plans, or experimental evidence, but are not automatically part of the promoted
production path:

- `IMPLEMENTED`: code exists;
- `VALIDATED`: evidence passed stated gates for a defined scope;
- `CANDIDATE`: useful for further validation, not final;
- `FAILED` / `INVALIDATED`: retained evidence, not a current scientific conclusion;
- `PLAN` / `DESIGN`: neither execution evidence nor production authorization.

## Documentation policy

- Stable usage belongs in `docs`; dated system-specific conclusions belong in `reports` or the result registry.
- Current interpretation follows current source, sealed protocols, and the latest reports; older `docs/status` files are snapshots.
- New headline numbers require a source artifact, units, sign convention, protocol identity, validity, and citation status.
- Do not rewrite old reports to erase history; record supersession and the current interpretation separately.

