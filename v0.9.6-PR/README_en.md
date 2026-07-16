# Atenolol-rank11 ABFE/IBS workflow

[中文](README_cn.md) | [English (current)](README_en.md)

This repository is an ABFE (Absolute Binding Free Energy) working directory for the Atenolol-rank11 system. The code is built around OpenMM: it constructs native OpenMM caches from GROMACS `gro/top` inputs, runs both the complex leg and the solvent leg, and aggregates binding free energies from IBS/MBAR/TMBAR-style window sampling results.

The currently recommended route is:

```text
mode = ibs
decoupling = dual_lambda
potential = softcore
boresch_source = simple
```

`traditional` REMD, DEXP, ORB/MACE Boresch estimation, and external APBS corrections are still present in the code, but they are not the main route for the current results in this directory.

## Current Status

Warning: the values currently stored in `output/final_binding_results.json` were computed with the old sign convention, `Delta G_bind = Delta G_complex - Delta G_solvent`. That is inconsistent with the current code in `runabfe.py`, where `delta_g_bind_uncorrected = dg_solvent - dg_complex`. The calculation must be rerun to obtain values whose sign matches the current formula. See "Interpreting Results" below and `AUDIT_STATUS.md` for details. The following numbers have been recomputed using the current formula while keeping the original `Delta G_complex` and `Delta G_solvent` values unchanged; they are only a reference and do not mean that new sampling has already been performed:

```text
Delta G_complex = 192.8876 kJ/mol
Delta G_solvent = 152.0514 kJ/mol
Boresch correction = -36.5108 kJ/mol
APBS correction = 0.0000 kJ/mol
Delta G_bind = Delta G_solvent - Delta G_complex = -40.8362 kJ/mol = -9.7601 kcal/mol
reported error = 1.3178 kJ/mol
```

A negative value indicates favorable binding: decoupling the ligand in the pocket costs more free energy (`Delta G_complex`) than decoupling it in solution (`Delta G_solvent`), which means the ligand interacts more strongly in the pocket. That is why it binds. The positive-sign result remaining on disk has the wrong physical sign and should not be used for any conclusion.

This is not a fully corrected, closed-cycle result ready to be treated as a final publication value. The most important current physical boundaries are:

- The default ACE/`dual_lambda` VDW/vanishing leg now includes an analytic LJ tail correction (`traditional_lj_lrc_protocol_version=2`): for each λ_vdw it numerically integrates the real switching-aware (restoring the energy the 1.0-1.2nm switching function removes) and softcore-aware (using the real softcore denominator, not a bare `r^6`) radial integral, covering both the attractive `r^-6` and repulsive `r^-12` moments. It intentionally does not enable OpenMM's built-in `CustomNonbondedForce` LRC, which would also integrate the Coulomb term in the combined expression and can crash CUDA. The Beutler `single_lambda`/REMD path now uses the same formula for its offline correction; outputs with `traditional_lj_lrc_protocol_version<2` correspond to the earlier formula, which did not yet account for the switching region, and should not be mixed with the current protocol's results.
- The APBS correction is only added to the final `Delta G_bind` as an external term, `Delta G_APBS`. The current `apbs_correction_kJ_mol = 0.0`, so it is disabled. APBS cannot replace the LJ tail correction.
- Older outputs may still contain historical PME self-correction text in `thermodynamic_cycle.md` and provenance. The current `output/final_binding_results.json` includes exactly such a stale text snapshot under `provenance.thermodynamic_cycle`. Treat `AUDIT_STATUS.md` and current diagnostics as authoritative. The current conclusion is that the manual `+C*lambda^2` PME self-energy "correction" has been withdrawn and should not be used as a production correction term.
- The current Boresch harmonicity check passes (`diagnostics.boresch.boresch_harmonicity_check.harmonic_assumption_ok = true`), but 3 of the 6 force constants (`kr`, `kthetaA`, `kphiA`) were clipped to conservative ranges (`force_constant_clipped`). This must be retained in the result interpretation.
- Stage 2 uses `Local-TMBAR-Stitched`, and the uncertainty propagates window offset variance. The complex leg has `offset_error_contribution approx 0.52 kJ/mol`, and the solvent leg has `approx 0.82 kJ/mol`. However, the uncertainty still does not include full global MBAR covariance, autocorrelation time, or effective sample size corrections. The overlap proxies are low (`min_overlap_proxy approx 0.027` for the complex leg and `approx 0.043` for the solvent leg), so denser windows and/or more sampling are recommended for follow-up confirmation.
- No independent repeat has been performed yet: `diagnostics.independent_repeats.performed = false`.

See `AUDIT_STATUS.md` for a more detailed list of methodological defects, remaining engineering-audit items, and repair status (it has merged and superseded the older `PHYSICS_DEFECTS.md`/`RELEASE_AUDIT_REMAINING.md`/`RE_AUDIT_2026-07-10.md`, which now remain only as compatibility stubs pointing to `AUDIT_STATUS.md`).

Implementation snapshot as of 2026-07-16: the default production path uses
`IBS_BIAS_PROTOCOL_VERSION=12`, `THERMODYNAMIC_PATH_PROTOCOL_VERSION=7`,
`TRADITIONAL_LJ_LRC_PROTOCOL_VERSION=2`, and `WCA_ACCOUNTING_VERSION=2`. v12 adds
a per-state, fingerprinted, checkpointed fixed-H probe trajectory bank. If a
MBAR-calibrated frozen `f_k` has not yet passed independent validation, it is
saved as `calibrated_pending_validation` and resumed with cumulative
50k -> 150k -> 300k validation budgets instead of falling back to SGD,
blindly splitting a window, or rerunning expensive probes. There are currently
no confirmed P0/P1 blockers on the default production path. See `todolist.md`
for unfinished code or engineering actions and `VALIDATION_MATRIX.md` for code-complete fixes
that still require full dependency or GPU evidence.

## Main Files

```text
runabfe.py              Command-line entry point; config merging, cache construction, two-leg scheduling, final aggregation
abfe_pipeline.py        Single-leg ABFE pipeline; pre-equilibration, pre-optimization, sampling, single-leg result writing
abfe_preoptimizer.py    Lambda-path pre-optimization; dual-lambda and 2D path helper logic
ibs_engine.py           IBS sampling, window energies, MBAR/TMBAR post-processing, and checkpoints
abfe_core.py            Softcore/DEXP potentials, Boresch restraints, analytical corrections, and utilities
apbs_correction.py      Optional helper script for external APBS correction; does not handle LJ tail correction
abfe_config.json        Example/compatibility config for the current directory
environment.yml         Example environment
AUDIT_STATUS.md         Current authoritative audit status: physical/modeling issues affecting Delta G + remaining engineering-audit checklist
todolist.md             Unimplemented code changes and outstanding engineering actions
VALIDATION_MATRIX.md    Monitoring table for code-complete fixes awaiting full tests or GPU evidence
PHYSICS_DEFECTS.md      Merged into AUDIT_STATUS.md; now a compatibility stub only
RELEASE_AUDIT_REMAINING.md  Merged into AUDIT_STATUS.md; now a compatibility stub only
```

This directory also contains Atenolol-rank11 input and intermediate files, such as `solv_ions.gro`, `topol.top`, `Atenolol-rank1.*`, `output/`, and related files.

## Dependencies

A conda/mamba environment is recommended. Core dependencies include:

- Python 3.10+
- OpenMM
- NumPy
- SciPy
- MDTraj
- PyMBAR

Optional dependencies:

- CUDA or OpenCL for GPU runs.
- A GROMACS force-field include directory for the first system build from `.top`.
- OpenMM-ML, torch, and MACE/ORB-related dependencies. These are only needed when using `--boresch-source auto`, `orb_simple`, `orb_ml`, or related ML functionality.

The current `output/run_provenance.json` records the runtime environment as Python 3.12.13, OpenMM 8.5.1, NumPy 2.4.3, PyMBAR 4.0.3, and MDTraj 1.10.3.

## Input Requirements

The first cache build usually requires:

- `--gro`: GROMACS coordinate file. The current config uses `solv_ions.gro`.
- `--top`: GROMACS topology file. The current config uses `topol.top`.
- `--ligand`: ligand residue name. The current config uses `MOL`.
- `--gmx-path`: GROMACS force-field include directory. It can also be detected from `GMXDATA` or the local `gmx` installation.
- `--ligand-xml`: optional. This can explicitly specify ligand XML/FFXML when building the solvent leg. If it is not provided, the code tries to extract the ligand from the GROMACS topology and generate `output/ligand_only.xml`.

Command-line arguments take precedence over the config file.

## Quick Start

Resume an existing calculation using the current config:

```bash
python runabfe.py --config abfe_config.json --ligand MOL --resume
```

First run, or rebuild caches:

```bash
python runabfe.py \
  --config abfe_config.json \
  --gro solv_ions.gro \
  --top topol.top \
  --ligand MOL \
  --gmx-path /path/to/gromacs/share/gromacs/top \
  --output ./output \
  --boresch \
  --boresch-source simple
```

Ignore caches and restart:

```bash
python runabfe.py --config abfe_config.json --ligand MOL --reset
```

Only post-process existing window energies and checkpoints:

```bash
python runabfe.py --config abfe_config.json --ligand MOL --analyze-only
```

Example CPU smoke test with a small budget:

```bash
python runabfe.py \
  --config abfe_config.json \
  --platform CPU \
  --n-steps-per-window 1000 \
  --n-states-per-stage 4 \
  --output ./smoke_output \
  --reset
```

## Config Example

The current `abfe_config.json` is a compatibility JSON config. Key fields are:

```json
{
  "preset": "production",
  "platform": "CUDA",
  "output": "./output",
  "temperature": 300.0,

  "gro": "solv_ions.gro",
  "top": "topol.top",
  "ligand": "MOL",
  "gmx_path": "/path/to/gromacs/share/gromacs/top",

  "decoupling": "dual_lambda",
  "potential": "softcore",

  "n_steps_per_window": 250000,
  "steps_per_update": 500,
  "stage1_n_states": 12,
  "stage2_n_states": 18,

  "boresch": true,
  "boresch_source": "simple",
  "boresch_batch": 0,
  "boresch_select": 1,

  "skip_rebalance": false,
  "rebalance_steps": 50000,

  "enable_early_stop": false,
  "enable_gradual_warmup": true,
  "warmup_steps": 500000,
  "min_bias_updates": 12,
  "max_bias_updates": 50,
  "required_consecutive_bias_updates": 3,
  "max_bias_warmup_steps": 500000,

  "pilot_finite_difference_delta": 0.01,
  "pilot_max_window_thermodynamic_length": 6.0,
  "pilot_overlap_thermodynamic_length": 1.5,
  "pilot_max_states_per_window": 6,

  "enable_lambda_refine": false,

  "resume": false,
  "reset": false
}
```

Note: the `gmx_path` in this repository is machine-specific and must be checked before running on another machine. The current config intentionally keeps `enable_lambda_refine=false` so it does not overwrite the existing local Stage 2 tail repair; do not enable it without first reading `_comment_lambda_refine` in `abfe_config.json`.

## Command Entry Points

Main command:

```bash
python runabfe.py [options]
```

Important options:

- `--mode {ibs,traditional}`: sampling engine. Default: `ibs`.
- `--decoupling {dual_lambda,single_lambda,2d_diagonal,2d_geodesic}`: decoupling path. Default: `dual_lambda`.
- `--potential {softcore,dexp}`: potential model. Default: `softcore`.
- `--decharge-method {pme,shadow_ibs}`: only affects Stage 1 (decharging) for `--decoupling dual_lambda`. Default: `pme`, preserving the original behavior. `shadow_ibs` is an experimental Shadow-Coulomb IBS path. It has not been independently physically validated, only supports neutral ligands, and currently does not support `--parallel-stages`. Production results should keep the default `pme`.
- `--preset {test,production,high_accuracy}`: sampling preset.
- `--stage1-n-states`: number of lambda states for the decharging stage. Takes precedence over `--n-states-per-stage`.
- `--stage2-n-states`: number of lambda states for the vanishing stage. Takes precedence over `--n-states-per-stage`.
- `--resume`: reuse caches, checkpoints, window energies, and pre-optimized paths.
- `--reset`: ignore caches and restart.
- `--parallel-stages`: try to run the decharging and vanishing stages in parallel.
- `--n-workers`: number of workers for offline energy recomputation/post-processing.
- `--apbs-correction-kj-mol`: add an external APBS correction to the final `Delta G_bind`.
- `--apbs-correction-note`: record the source of the APBS correction.

Built-in subcommand:

```bash
python runabfe.py self-test
```

Runs lightweight tests and physical-convention checks. Note that the current self-test still contains a historical PME self-correction sign check. If it conflicts with the latest physical conclusion, the code/tests should be fixed first instead of treating that check as production correction evidence.

```bash
python runabfe.py prepare \
  --gro solv_ions.gro \
  --top topol.top \
  --ligand MOL \
  --gmx-path /path/to/gromacs/share/gromacs/top \
  --output-dir ./prep_output \
  --save-boresch boresch.json
```

Generates preprocessing files, such as Boresch parameters or DEXP parameters.

```bash
python runabfe.py refine-lambda-path \
  --stage-dir output/vanishing \
  --preopt-file output/checkpoints/preopt_dual_vanishing.json \
  --stage-type vdw \
  --max-window-span-kj 35.0 \
  --overlap 2
```

Redistributes lambda states and window boundaries from existing window energies. It overwrites `--preopt-file`, so keep a backup before using it.

## Boresch Sources

Allowed values for `--boresch-source`:

- `simple`: pure geometric fluctuation estimate without ML dependencies. This is the current recommendation and the source used by the current results.
- `fluctuation`: a geometric fluctuation route similar to `simple`.
- `traditional`: reads an external Boresch anchor file and requires `--boresch-anchors`.
- `orb_ml`: reads an ORB/ML prediction file and requires `--boresch-orb`.
- `orb_simple`: uses a single-candidate estimate from ORB/MACE pocket-force projection. Requires ML dependencies and model licensing.
- `auto`: uses ORB/MACE multi-candidate enumeration. Requires ML dependencies and model licensing.

Internal estimation routes save parameters and diagnostics in `output/boresch_*.json`. For the current results, inspect these Boresch diagnostics carefully:

- `diagnostics.boresch.analytical_release_reliable`
- `boresch_correction_diagnostics.diagnostics.boresch_harmonicity_check`
- `force_constant_clipped`
- `diagnostics.warnings`

## Output Structure

Common outputs:

```text
output/run_provenance.json              Config, command line, hashes, software versions
output/final_binding_results.json       Final Delta G_bind summary
output/final_results.json               Complex-leg results
output/solvent_leg/final_results.json   Solvent-leg results
output/pipeline.log                     Complex-leg log
output/solvent_leg/pipeline.log         Solvent-leg log
output/system_native.xml                Complex OpenMM System cache
output/system_solvent.xml               Solvent-leg OpenMM System cache
output/topology.cif                     Complex topology cache
output/topology_solvent.cif             Solvent-leg topology cache
output/boresch_simple.json              Current Boresch parameter cache
output/checkpoints/                     Complex-leg checkpoints and stage state
output/solvent_leg/checkpoints/         Solvent-leg checkpoints and stage state
output/decharging/                      Decharging-stage trajectories/energies
output/vanishing/                       Vanishing-stage window energies
output/vanishing/production_fixed_h_overlap.json  Per-edge fixed-H diagnostics used by production ESS repair
output/vanishing/sampling_repair_decisions.json   Automatic recalibration/resampling decisions
output/checkpoints/probes/              Resumable fixed-H path and bias-calibration trajectory banks
```

Window-level files usually include:

```text
dual_window_*_energies.npy
dual_window_*_bias.npy
dual_window_*_base.npy
dual_window_*_convergence.json
decharging_pme_u_kn.npy
decharging_pme_u_kn.meta.json
```

These files are the core inputs for `--analyze-only` and `refine-lambda-path`. Confirm that they are no longer needed before cleaning them up.

## Interpreting Results

The final binding free energy is aggregated as:

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

where:

- `Delta G_complex` and `Delta G_solvent` are both defined as decoupling free energies for their respective legs, from lambda 1 to 0 (coupled to decoupled). Larger values mean decoupling is harder and the interactions are stronger.
- `Delta G_complex` already includes the analytical Boresch release correction for the complex leg.
- The solvent leg does not use Boresch, so `boresch_correction_kJ_mol = 0`.
- For a genuinely binding ligand, `Delta G_complex > Delta G_solvent` because decoupling in the pocket is harder. Therefore, `Delta G_bind` should be negative, indicating favorable binding.
- `Delta G_APBS` defaults to 0. It is only applied when `--apbs-correction-kj-mol` is passed explicitly.
- The default ACE/`dual_lambda` vanishing leg automatically includes a switching-aware, softcore-aware analytic LJ tail term with both `r^-6` and `r^-12` contributions. Traditional Beutler REMD adds the same correction offline for fixed-box NVT trajectories. It fails closed when appreciable NPT volume fluctuations are detected, because appending `1/V` afterward cannot repair a volume distribution sampled under the wrong Hamiltonian.

Before treating a result as ready for the next level of discussion, check at least:

- Whether `output/final_binding_results.json` exists and has a timestamp matching the current run.
- Whether `provenance.hashes.code_sha256` matches the code version you intend to archive.
- Whether `lj_long_range_dispersion_correction.status` is `implemented_analytic_mean_field_switching_softcore_aware` (`protocol_version=2`), with a matching LRC protocol version/fingerprint; do not mix legacy `not_implemented` or `implemented_analytic_mean_field` (v1, missing the switching-region correction) output with the current protocol.
- Whether `stage_diagnostics.stage2.min_overlap_proxy` is too low.
- Whether `stage_diagnostics.*.uncertainty_note` still reports missing full covariance/autocorrelation corrections.
- Whether the Boresch harmonicity check has `ok = true` and `harmonic_assumption_ok = true`.
- Whether many entries in `force_constant_clipped` are `true`.
- Whether independent repeats have been performed. The current result records `independent_repeats.performed = false`.

## Caches and Resume Behavior

`--resume` tries to reuse:

- `system_native.xml` / `system_solvent.xml`
- `ligand_indices*.json`
- `topology*.cif`
- `pre_equilibration.dcd`
- `checkpoints/pre_equil.chk`
- `boresch_*.json`
- `preopt_dual_*.json`
- Stage sampling results and window energy files

Use a new `--output` directory or `--reset` in the following cases:

- `gro/top/ligand` changed.
- Ligand parameters or `gmx_path` changed.
- `decoupling`, `potential`, or DEXP parameters changed.
- The number of lambda states, window density, or sampling budget changed.
- The Boresch source, anchors, or candidate selection changed.
- You want to fully refresh old `thermodynamic_cycle` or provenance text.

## Parallelism and GPU

`--parallel-stages` tries to run decharging and vanishing in parallel. Under CUDA, if both stages share the same GPU, the code may fall back to serial execution to avoid context conflicts.

Linux shell example:

```bash
IBS_STAGE1_CUDA_DEVICE=0 IBS_STAGE2_CUDA_DEVICE=1 \
python runabfe.py --config abfe_config.json --ligand MOL --resume --parallel-stages
```

Windows PowerShell example:

```powershell
$env:IBS_STAGE1_CUDA_DEVICE = "0"
$env:IBS_STAGE2_CUDA_DEVICE = "1"
python runabfe.py --config abfe_config.json --ligand MOL --resume --parallel-stages
```

## FAQ

### `ModuleNotFoundError: No module named 'openmm'`

The current Python environment does not have OpenMM installed, or the correct environment has not been activated:

```bash
python -c "import openmm; print(openmm.__version__)"
```

### GROMACS include files cannot be found

Check that `--gmx-path` points to a directory containing `.ff` folders, for example:

```text
/path/to/gromacs/share/gromacs/top
```

You can also set `GMXDATA` so the code can try `$GMXDATA/top` automatically.

### Ligand residue cannot be found

Confirm that `--ligand MOL` matches the residue name in the `.gro/.top` files. The current directory uses `MOL`.

### Solvent-leg construction fails

A common cause is incomplete ligand XML/FFXML or failed extraction from the GROMACS topology. Try explicitly passing `--ligand-xml`, or check whether `output/ligand_only.xml` was generated.

### Automatic Boresch estimation fails

The current recommendation is to start with the non-ML route:

```bash
--boresch --boresch-source simple
```

If the anchors are unstable, inspect `output/boresch_simple.json`, `pre_equilibration.dcd`, and the Boresch harmonicity diagnostics.

### `--analyze-only` is missing energy files

At minimum, keep the stage checkpoints or window-level `.npy` energy files. Do not delete these casually:

```text
output/checkpoints/stage1_decharging.json
output/checkpoints/stage2_vanishing.json
output/decharging/decharging_pme_u_kn.npy
output/vanishing/dual_window_*_energies.npy
```

### `thermodynamic_cycle` in the results conflicts with the defect list

This is a known issue caused by stale historical provenance text. The current README and `AUDIT_STATUS.md` are authoritative: APBS does not replace the LJ tail correction, the manual PME self `+C*lambda^2` term is not used as a production correction, and `Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS`, not `Delta G_complex - Delta G_solvent`.

If your `output/final_binding_results.json` or `thermodynamic_cycle.md` was generated before these documentation fixes, its `delta_G_bind_kJ_mol` sign may be reversed, and the `thermodynamic_cycle` field text may still be the old version. Rerun the final aggregation for the complex leg and solvent leg to refresh the result to the current convention. New sampling is not required as long as `complex_results` and `solv_results` can be loaded from cache.

## Maintenance Suggestions

After modifying code, run a syntax check first:

```bash
python -c "import ast, pathlib; files=['runabfe.py','abfe_pipeline.py','abfe_preoptimizer.py','ibs_engine.py','abfe_core.py']; [ast.parse(pathlib.Path(f).read_text(encoding='utf-8'), filename=f) for f in files]; print('syntax ok')"
```

Then run:

```bash
python runabfe.py self-test
```

If the self-test disagrees with the latest physical conclusions in `AUDIT_STATUS.md`, update the tests and thermodynamic-cycle documentation so old assumptions do not continue entering new provenance. The full suite also requires OpenMM, PyMBAR, and pytest; a syntax-only pass is not an end-to-end validation when runtime dependencies are absent.

Recommended next priorities:

1. Run the full `python -m pytest -q` suite in the target environment, with particular attention to the fixed-H bank, native checkpoints, LRC, and the v12 frozen-validation state machine.
2. Revalidate the v12 `calibrated_pending_validation` resume path and fixed-H `lambda_shield` synchronization fix on a real GPU.
3. Run at least one independent repeat for the current Atenolol-rank11 config.
4. Use stage diagnostics to decide whether the vanishing-stage windows should be densified or sampled longer; track the remaining source-level P2 items in `todolist.md`.
