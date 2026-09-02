# ABFE-IBS: an absolute binding free-energy workflow

[中文](README_cn.md) · [Concise entry](README.md) · [Documentation](docs/README.md)

ABFE-IBS is an OpenMM-based absolute binding free-energy workflow. It consumes GROMACS
`.gro/.top` systems, computes coupled-to-decoupled free energies for the protein–ligand
complex and bulk-solvent legs, and combines IBS sampling, MBAR/TMBAR analysis, a Boresch
restraint ledger, and explicit long-range corrections in a documented thermodynamic cycle.

This repository is the **engineering branch** of ABFE-IBS, aimed at release. It contains
only the reusable workflow source (`runabfe.py` and friends -- see
[PROJECT_LAYOUT.md](PROJECT_LAYOUT.md)), the production regression tests (`tests/`),
manual diagnostic tooling (`tools/`), and the user documentation (`docs/`).

**Not here**: reference-system `output*` trees, validation trajectories and checkpoints,
development-era experiment scripts (`exp0XX_*`), failed-experiment records, and the
per-decision history. Those live in the `Atenolol-rank11` workspace; this repository keeps
only an [index of that material](docs/HISTORY_LOG.md).

To move to a different protein-ligand system, read the
[migration guide](docs/MIGRATING_TO_A_NEW_SYSTEM.md). Never reuse one system's checkpoints
for another.

## Scientific status (evidence through 2026-09-02)

The main software path includes GROMACS-to-OpenMM construction, complex and solvent legs,
dual-lambda decoupling, IBS warmup and frozen-bias production, MBAR/TMBAR, Boresch
attachment/release accounting, LJ long-range correction, caching, resume, and fail-closed
checks.

**The current production system is 4W53 (T4 lysozyme L99A + toluene), no longer Atenolol.**

Principal protocol identities, read directly from the source constants (not restated):

| Protocol | Constant | Value |
|---|---|---|
| IBS bias | `ibs_engine.IBS_BIAS_PROTOCOL_VERSION` | **32** |
| Thermodynamic path | `abfe_preoptimizer.THERMODYNAMIC_PATH_PROTOCOL_VERSION` | 21 |
| LJ long-range correction | `ibs_engine.TRADITIONAL_LJ_LRC_PROTOCOL_VERSION` | 3 |
| WCA accounting | `ibs_engine.WCA_ACCOUNTING_VERSION` | **3** (`WCA_SHIELD_RETIRED = True`) |
| ESS gate | `ibs_engine.ESS_GATE_PROTOCOL_VERSION` | **5** |
| Ligand COM restraint | `ibs_engine.LIGAND_COM_RESTRAINT_PROTOCOL_VERSION` | 2 |

### Result registry

| Result | System / run | Registry status | Final/citable? |
|---|---|---|---|
| **`−21.36 ± 0.93 kJ/mol`** (`−5.11 ± 0.22 kcal/mol`) | 4W53, `output_v3_seed20260908`, 2026-09-02 | **label pending maintainer** | **No: single seed (`20260908`), no second independent repeat in this repository** |
| `−23.1622 ± 2.5139 kJ/mol` from `output_lrc_fix` | Atenolol-rank11 | **VOIDED (declared 2026-08-24)** | No |
| `+40.8362 ± 1.3178 kJ/mol` from historical `output` | Atenolol-rank11 | `INVALIDATED` | No: opposite historical sign convention and diagnostic issues |
| `+16.00 ± 2.20 kJ/mol` from 2026-07-27 | Atenolol-rank11 | `INVALIDATED` | No: stale and incorrect Boresch equilibrium geometry |

For the 4W53 row: experiment is **−23.10 kJ/mol** (`−5.52 ± 0.04 kcal/mol`), so the
discrepancy is **0.41 kcal/mol, within 1.83σ**. The quality gates turned healthy at the same
time (solvent-leg stage2 raw ESS 2.93 → 173.33, top-1% weight 0.828 → 0.047). Per-item
evidence: [docs/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md](docs/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md).

**Still open (not blocking, but must be quoted alongside the number):** solvent-leg stage2 is
the only quantity with an independent reference truth. Measured ≈ **−8.3** against a truth of
**−6.58 ± 0.26** (no-LRC convention, see
[docs/reference_data/README.md](docs/reference_data/README.md)); the 1.7-4.2 kJ/mol gap is not
yet attributed. Independent repeats, the production seed ledger, and the time-correlated
uncertainty model remain **unclosed**.

The current sign convention is:

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

A filename containing `final` does not make an artifact citable. For the **three Atenolol
rows**, the raw artifacts, result index, and machine-readable registry (`RESULT_REGISTRY.csv`)
live in the `Atenolol-rank11` workspace and do not ship with this engineering branch. The
evidence for the 4W53 row is in this repository's `docs/` (linked above).

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

