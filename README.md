# ABFE-IBS: an absolute binding free-energy workflow

🌐 [**中文 →**](README_cn.md)　·　[Documentation](docs/README.md)　·　[Repository layout](PROJECT_LAYOUT.md)

ABFE-IBS is an OpenMM-based absolute binding free-energy (ABFE) workflow. It consumes
GROMACS `.gro/.top` systems, computes decoupling free energies for the protein-ligand
complex and the bulk-solvent leg, and closes the thermodynamic cycle with IBS sampling,
MBAR/TMBAR analysis, a Boresch restraint ledger, and explicit long-range corrections:

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

Per-step methodology and citations: [docs/METHODS.md](docs/METHODS.md).

The pipeline covers GROMACS-to-OpenMM construction, both legs, dual-lambda decoupling,
IBS warmup and frozen-bias production, MBAR/TMBAR estimation, Boresch
attachment/release accounting, LJ long-range correction, caching, resume, and
fail-closed quality gates.

> **Current scientific status lives in [docs/STATUS.md](docs/STATUS.md).** This repository
> currently has **no** result that may be cited as a final conclusion. The production
> system, result registry, protocol versions, and open items are all recorded there --
> read it before quoting any number.

## 1. Environment

Core dependencies: **Python 3.12**, OpenMM, NumPy, SciPy, MDTraj, PyMBAR. Production GPU
runs additionally require a matching CUDA or OpenCL stack. The supplied `environment.yml`
carries machine- and CUDA-specific choices and **must be reviewed before use on another
host**.

```bash
python -c "import openmm, numpy, scipy, mdtraj, pymbar; print(openmm.__version__)"
```

If `python runabfe.py --help` fails at import with `No module named 'openmm'`, the active
shell is not in a runnable environment -- it is not a command-line argument error.

Self-check and configuration diagnostics (read-only, no Context is built):

```bash
python runabfe.py doctor
python runabfe.py validate-config --config abfe_config.json
python runabfe.py config-template --out my_system.json
```

## 2. Inputs

A first build normally needs:

| Flag | Meaning |
|---|---|
| `--gro` | GROMACS coordinate file |
| `--top` + `--gmx-path` | GROMACS topology and its include tree (`--gmx-path` takes the GROMACS install prefix) |
| `--ligand` | ligand residue name |
| `--output` | a separate, **new** output directory |

`abfe_config.json` is a reference configuration containing a machine-specific `gmx_path`
and options deliberately frozen for historical runs. Do not use it as an unreviewed
template for a new system -- generate one with `config-template`, or read the
[migration guide](docs/MIGRATING_TO_A_NEW_SYSTEM.md).

**Never reuse one system's checkpoints for another system.**

## 3. Running

New system, explicit inputs:

```bash
python runabfe.py \
  --config abfe_config.json \
  --gro /path/to/system.gro \
  --top /path/to/topol.top \
  --ligand LIG \
  --gmx-path /path/to/gromacs \
  --output ./output_new_system \
  --boresch --boresch-source simple
```

Resume:

```bash
python runabfe.py --config abfe_config.json --ligand MOL --resume
```

Analyze existing energies and checkpoints only (no dynamics):

```bash
python runabfe.py --config abfe_config.json --ligand MOL --analyze-only
```

Read [Outputs and resume](docs/OUTPUTS_AND_RESUME.md) **before** using `--resume`,
`--reset`, or `--analyze-only`. Never point `--reset` at a protected historical evidence
directory.

## 4. Reading results

Results land in the `--output` directory: `final_binding_results.json` is the summary,
`run_provenance.json` records protocol identities and input fingerprints, and
`checkpoints/` is what resume relies on. Sign conventions, per-term definitions, and
resume semantics are in [OUTPUTS_AND_RESUME.md](docs/OUTPUTS_AND_RESUME.md).

**A filename containing `final` does not make a result citable.** The criteria are in
[docs/STATUS.md](docs/STATUS.md).

## 5. Minimum verification after code changes

```bash
./tests/run_offline_tests.sh                                       # everything except needs_gpu
./tests/run_offline_tests.sh tests/test_core_physics_numerics.py   # one file
```

Passing tests establish **software contracts**; they do not automatically validate a new
**scientific result**. Maintenance rules are in [MAINTAINING.md](docs/MAINTAINING.md) and
[PROJECT_LAYOUT.md](PROJECT_LAYOUT.md).

## 6. Where to start reading the code

- `runabfe.py` -- main ABFE CLI (`doctor` / `validate-config` / `config-template` live here too)
- `runrbfe.py` -- RBFE CLI
- `abfe_config.json` -- reference configuration, **not a template for a new system**

What every module does, which small modules a clone cannot run without, and where new
files belong: **[PROJECT_LAYOUT.md](PROJECT_LAYOUT.md)** -- the single source for
repository structure. This file no longer carries a second copy of the tree.

## 7. Documentation

**[docs/README.md](docs/README.md) is the full documentation map.** The four most-used
entry points:

| Need | Entry point |
|---|---|
| Scientific status, whether a result is citable, protocol versions | [docs/STATUS.md](docs/STATUS.md) |
| Installation, inputs, first run | [GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| Open work (carries `[P1]`/`[P2]`/`[P3]` levels) | [docs/TODO.md](docs/TODO.md) |
| Troubleshooting | [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |

## 8. What this repository is

This is the **engineering branch** of ABFE-IBS: workflow source, production regression
tests, diagnostic tooling, and user documentation only. Reference-system `output*` trees,
trajectories and checkpoints, failed-experiment records, and the per-decision history live
in the `Atenolol-rank11` workspace; this repository keeps only an
[index of that material](docs/HISTORY_LOG.md).

**The release model is clone-and-run**: `git clone`, then `python runabfe.py ...`. There is
no packaging step. The frozen R1 resources and the prebuilt plugin `.so` ship with the
repository -- a clone needs no compilation step and no separate download. The criterion is
`pytest tests/test_fresh_clone_imports.py`. Details in [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md).

**The tutorials under `docs/` are written in Chinese** -- a deliberate choice for
development speed, not an oversight: Chinese is the authoritative version there, and its
commands, paths, and status markers are directly usable as-is. This English README covers
the full workflow; [README_cn.md](README_cn.md) is its Chinese counterpart. The two are
**not** kept at word-for-word parity (see [docs/TODO_P3.md](docs/TODO_P3.md)).

## License

[MIT License](LICENSE), Copyright (c) 2026 Ruigeng Ji.

Third-party attribution and the compliance rationale are in [NOTICE](NOTICE) -- note that
**OpenMM is dual-licensed**: the public API, reference platform, CPU platform, and
application layer are MIT, while the CUDA, HIP, and OpenCL platforms are LGPL. This
repository vendors no third-party source and ships no third-party binary.
