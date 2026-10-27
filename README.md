# ABFE-IBS: an absolute binding free-energy workflow

🌐 [**中文 →**](README_cn.md)　·　[Documentation](docs/README.md)

ABFE-IBS is an OpenMM-based absolute binding free-energy (ABFE) workflow. It consumes
GROMACS `.gro/.top` systems, computes decoupling free energies for the protein-ligand
complex and the bulk-solvent leg, and closes the thermodynamic cycle with IBS sampling,
MBAR/TMBAR analysis, a Boresch restraint ledger, and explicit long-range corrections:

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

Per-step methodology, citations, and protocol versions: [docs/METHODS.md](docs/METHODS.md).

The pipeline covers GROMACS-to-OpenMM construction, both legs, dual-lambda decoupling
(electrostatics removed before van der Waals), MBAR/TMBAR estimation, Boresch
attachment/release accounting, LJ long-range correction, caching, resume, and
fail-closed quality gates.

## AIBS: adaptive integrated Boltzmann sampling for the van der Waals stage

Integrated Boltzmann sampling (IBS) brings neighboring λ states into one mixture
potential. In AIBS, neighboring λ states are grouped into **windows**; a mixture potential
with fixed energy offsets is a **generator**, and one trajectory under it is a
**segment**. The generator stays fixed during each segment, so completed segments can be
compared before more sampling is requested:

- **Round 1** samples each complex-leg window with three segments -- a center generator and
  two with a small linear tilt toward either end of the window (one center segment per
  solvent-leg window).
- After each round, every window is checked by **segment deletion** (removing segments in
  turn and re-solving), the pooled per-state ESS, and the top-1% weight fraction. Windows
  that fail receive additional segments in the next round.
- Segment disagreement is included in the reported uncertainty.

AIBS applies to the van der Waals stage (stage 2) of `dual_lambda` decoupling. It is
switched on in the run configuration (it is **off by default**) and requires a fixed
stage-2 layout: a uniform λ grid (`stage2_uniform_lambda_states` = N gives
λ_i = 1 − i/(N−1)) and explicit windows (`stage2_window_ranges`, half-open state-index
ranges; neighboring windows share their boundary state). The stage-2 settings used for
the manuscript's neutral benchmark:

```json
{
  "decoupling": "dual_lambda",
  "n_steps_per_window": 250000,
  "stage2_uniform_lambda_states": 26,
  "stage2_window_ranges": [[0, 6], [5, 11], [10, 16], [15, 21], [20, 26]],
  "stage2_lcr_rounds": true
}
```

`n_steps_per_window` is the length of one segment (250000 steps = 0.5 ns at 2 fs).
Setting `stage2_lcr_rounds` without the layout keys stops the run with an error.

Optional parameters (defaults in `stage2_rounds.py`):

| Key | Default | Meaning |
|---|---|---|
| `stage2_lcr_max_rounds` | 3 | Maximum number of rounds, including round 1 |
| `stage2_lcr_sigma_tol_kJ` | 1.5 | Segment-deletion sensitivity threshold (kJ/mol) |
| `stage2_lcr_tilt_kT` | 2.0 | End-to-end tilt of the round-1 left/right generators (kT) |
| `stage2_lcr_trust_floor_kT` | 3.0 | Lower bound of the trust region for re-centered offsets (kT) |
| `stage2_lcr_burn_in_steps` | 25000 | Steps discarded at the start of each segment |

The pooled-ESS (20) and top-1% weight (0.35) thresholds are fixed in code.

## Citation and data

<!-- TODO before release: author list, journal reference, and data-archive DOI -->
If you use this code, please cite:

> *Adaptive Integrated Boltzmann Sampling for the van der Waals Stage of Absolute
> Binding Free-Energy Calculations* (manuscript in preparation; reference to be added).

The data sets and analysis scripts supporting the manuscript will be deposited in a
public archive; the DOI will be listed here.

## 1. Environment

Core dependencies: **Python 3.12**, OpenMM 8.5.2, NumPy, SciPy, MDTraj, PyMBAR (pinned to
`pymbar-core=4.2.0`). GPU runs additionally require a CUDA stack. `environment.yml` is the
full production environment (`environment-ci.yml` is the CPU-only CI subset); it carries
CUDA-specific choices and **should be reviewed before use on another host**.

```bash
conda env create -f environment.yml
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
| `--top` + `--gmx-path` | GROMACS topology and its include tree (`--gmx-path` takes the GROMACS install prefix; if empty, `GMXLIB`/`GMXDATA`/`PATH` are searched) |
| `--ligand` | ligand residue name |
| `--output` | a separate, **new** output directory |

`abfe_config.json` is a reference configuration with options frozen for earlier runs. Do
not use it as an unreviewed template for a new system -- generate one with
`config-template`, or read the [migration guide](docs/MIGRATING_TO_A_NEW_SYSTEM.md).

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
`--reset`, or `--analyze-only`.

## 4. Reading results

Results land in the `--output` directory: `final_binding_results.json` is the summary,
`run_provenance.json` records protocol identities and input fingerprints, and
`checkpoints/` is what resume relies on. Sign conventions, per-term definitions, and
resume semantics are in [OUTPUTS_AND_RESUME.md](docs/OUTPUTS_AND_RESUME.md).

A completed run is a computation, not a validated scientific result: check convergence
and repeat-to-repeat spread before quoting a number.

## 5. Optional features (off by default)

| Feature | Status | Docs |
|---|---|---|
| Outer-λ local residual (`--outer-lambda-local-residual-ibs`) | Experimental, complex leg only. No frozen weights ship with the repository: run once without it, refit from that run's stage-2 frames, then rerun with `--outer-lambda-resource-manifest`. Uses the prebuilt CUDA plugin in `plugins/LocalManyBodyResidual/build/` | [RETRAIN_LOCAL_RESIDUAL.md](docs/RETRAIN_LOCAL_RESIDUAL.md) |
| Reference-IBS comparison arm (`stage2_online_ibs` in the run config, together with `stage2_lcr_rounds`) | Complex-leg stage 2 only: offsets are learned during production; the run stops after complex stage-2 sampling and writes no binding result. The offline TMBAR analysis script is not part of this repository | -- |
| RBFE (`runrbfe.py`) | Independent of the ABFE mainline; contracts, atom mapping and a restricted hybrid builder exist, end-to-end edges do not | -- |
| Pocket-water sampler (`tools/pocket_water/`) | Experimental tool, not wired into the pipeline | -- |
| Membrane systems, charge-transfer, DEXP / MACE / ORB paths | Code present, not on the default production path | -- |

## 6. Repository layout

The release model is **clone-and-run**: `git clone`, then `python runabfe.py ...`; there
is no packaging step. `pytest tests/test_fresh_clone_imports.py` checks that a fresh
clone imports.

ABFE mainline:

| File | Role |
|---|---|
| `runabfe.py` | ABFE command-line entry (also `doctor` / `validate-config` / `config-template`) |
| `abfe_core.py` | System construction and low-level physics components |
| `abfe_pipeline.py` | Stage orchestration, quality gates, resume, result output |
| `abfe_preoptimizer.py` | λ-path and window pre-optimization; stage-2 windowing and controller |
| `ibs_engine.py` | IBS, MBAR/TMBAR, Boresch, long-range correction |
| `free_energy_engine.py` | Sampling engine shared by ABFE and RBFE |
| `abfe_diagnostics.py` | Implementation of the read-only diagnostic subcommands |
| `stage2_rounds.py`, `stage2_rounds_evidence.py` | AIBS round controller (pure functions) and its per-window evidence (segment deletion, pooled ESS, top-1% weight) |
| `stage2_online_ibs.py` | Reference-IBS comparison arm: offsets learned online, time-aggregated MBAR (off unless configured) |
| `step_guard.py`, `lambda_path_versions.py`, `multi_segment_analysis.py` | Small support modules required by the entry points |
| `apbs_correction.py` | Rocklin finite-size electrostatic correction |
| `outer_lambda_neural_basis.py` | Outer-λ basis potential (dependency of `local_residual.openmm_plugin`) |
| `stage2_ab_report.py` | Post-run stage-2 report / two-run A/B comparison |

RBFE: `runrbfe.py`, `rbfe_core.py`, `rbfe_pipeline.py`.

| Directory | Contents |
|---|---|
| `local_residual/` | Outer-λ local residual: runtime loader, refit, and training stack |
| `plugins/LocalManyBodyResidual/` | Native OpenMM plugin source, build script `g0_build.sh`, and the prebuilt `.so` (`build/`, built against `openmm=8.5.2`; CUDA kernels are compiled at runtime, so no GPU architecture is baked in) |
| `abfe_scripts/`, `exp012_xed/` | Offline training / export scripts used by the local-residual chain |
| `tools/` | Diagnostics, validation, and refit tools -- not production entry points |
| `tests/` | Automated tests (`./tests/run_offline_tests.sh`) |
| `docs/` | User documentation |

## 7. Minimum verification after code changes

```bash
./tests/run_offline_tests.sh                                       # everything except needs_gpu
./tests/run_offline_tests.sh tests/test_core_physics_numerics.py   # one file
```

Passing tests establish **software contracts**; they do not validate a new **scientific
result**.

## 8. Documentation

**[docs/README.md](docs/README.md) is the documentation map.** The guides under `docs/` are
written in Chinese; this English README covers the full workflow, and
[README_cn.md](README_cn.md) is its Chinese counterpart.

## License

[MIT License](LICENSE), Copyright (c) 2026 Ruigeng Ji.

Third-party attribution and the compliance rationale are in [NOTICE](NOTICE) -- note that
**OpenMM is dual-licensed**: the public API, reference platform, CPU platform, and
application layer are MIT, while the CUDA, HIP, and OpenCL platforms are LGPL. The
plugin under `plugins/LocalManyBodyResidual/platforms/cuda/` builds on the latter.
