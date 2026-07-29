# dexp_experiment.py

`dexp_experiment.py` is the single-file harness for fitting, running, and judging a DEXP surrogate potential.

The important mental model is:

```text
Gaussian Coulomb + DEXP ~= MACE local ligand-environment interaction
```

MM is a baseline and a floor, not the truth target. The DEXP potential should be judged by whether it gives a stable MD Hamiltonian whose sampled structures are still endorsed by MACE.

## What This Script Does

The full pipeline has four stages:

1. Read tail frames from the pre-equilibration trajectory.
2. Label local ligand-environment interaction energies with MACE/OpenMM-ML.
3. Fit a DEXP pair potential to the local MACE residual.
4. Build a surrogate OpenMM system, run DEXP and original-baseline MD, then write diagnostics.

The default fit target is `mace_surrogate_residual`:

```text
DEXP target = E_MACE_local - E_gaussian_coul_region
```

So the fitted surrogate is interpreted as:

```text
E_applied_local = E_gaussian_coul + E_DEXP
```

Older target modes such as `qmmm_residual` and `ml_minus_mm_total` are still present, but they answer a different question and should not be used as the main DEXP validation target.

## Typical Commands

Full fit + 1 ns DEXP/baseline comparison:

```bash
python dexp_experiment.py --platform CUDA
```

Reuse cached MACE labels and only refit/diagnose:

```bash
python dexp_experiment.py --reuse-fit-labels --fit-only
```

Run the optional learned pair-RBF diagnostic without changing the MD potential:

```bash
python dexp_experiment.py --reuse-fit-labels --fit-only --learned-rbf-diagnostic
```

After a DEXP production run, relabel the DEXP trajectory with MACE and ask whether MACE endorses the sampled structures:

```bash
python dexp_experiment.py ^
  --relabel-traj output/dexp_experiment/dexp_surrogate/traj.dcd ^
  --relabel-baseline-traj output/dexp_experiment/original_baseline/traj.dcd
```

`--relabel-traj` is the most important validation mode for the current scientific question. It evaluates the world produced by DEXP, not only old fitting frames.

## Main Inputs

Default paths:

| argument | default | meaning |
| --- | --- | --- |
| `--traj` | `output/pre_equilibration.dcd` | pre-equilibrated trajectory used for fitting |
| `--traj-top` | `output/topology.cif` | topology for the trajectory |
| `--system-xml` | `output/system_native.xml` | original OpenMM system |
| `--ligand-indices` | `output/ligand_indices.json` | ligand atom indices |
| `--gmx-top` | `topol.top` | GROMACS topology used for MM reference components |
| `--output-dir` | `output/dexp_experiment` | output directory |

## Fit Controls

Important fitting options:

| argument | default | note |
| --- | --- | --- |
| `--fit-frames` | `500` | maximum number of tail frames |
| `--fit-last-ns` | `5.0` | only use the final time window |
| `--fit-env-radius` | `0.50 nm` | environment selection radius for the MACE local cluster |
| `--fit-r-min` | `0.20 nm` | lower distance bound for DEXP fitting pairs |
| `--fit-r-max` | `0.45 nm` | upper distance bound for DEXP fitting pairs in this script version |
| `--fit-objective` | `pmf_mean` | fit per-bin mean profile along min-distance instead of raw frame noise |
| `--fit-pmf-bins` | `12` | min-distance bins for PMF matching |
| `--fit-pmf-min-bin-frames` | `10` | sparse bins are excluded from the fit |
| `--reuse-fit-labels` | off | reuse cached MACE/MM labels where compatible |
| `--holdout-fraction` | `0.2` | validation frames held out from fitting |

`pmf_mean` is the preferred objective here because DEXP is a low-dimensional local surrogate. It should reproduce useful mean surfaces and local free-energy shape, not every many-body per-frame fluctuation.

## Output Files

Fitting outputs:

| file | meaning |
| --- | --- |
| `dexp_fitted_params.json` | fitted DEXP parameters and fit metadata |
| `fit_frame_diagnostics.csv` | per-frame labels, target energies, min distances, pair counts |
| `fit_holdout_validation.csv` | holdout per-frame prediction diagnostics |
| `fit_holdout_pmf_profile.csv` | holdout profile along min-distance |
| `fit_all_accepted_mace_surrogate_pmf_1d.csv/png` | MACE-vs-surrogate 1D surface/profile on accepted frames |
| `fit_all_accepted_mace_surrogate_pmf_2d.csv/png` | MACE-vs-surrogate 2D profile using min-distance and contact count |
| `fit_learned_rbf_*` | optional learned pair-RBF diagnostic files |

MD/postprocess outputs:

| file | meaning |
| --- | --- |
| `comparison_summary.json` | machine-readable complete summary |
| `comparison_report.md` | human-readable report |
| `dexp_surrogate/traj.dcd` | DEXP production trajectory |
| `original_baseline/traj.dcd` | original MM baseline trajectory |
| `le_min_distance_comparison.csv/png` | ligand-environment closest-contact distribution |
| `le_rdf_comparison.csv/png` | ligand-environment RDF comparison |
| `le_pmf_1d_comparison.csv/png` | local 1D PMF proxy from production trajectories |
| `lambda_window_*` | fixed-lambda short reruns and contact diagnostics |

Relabel outputs:

| file | meaning |
| --- | --- |
| `relabel_dexp_1d_pmf.csv` | DEXP production frames relabeled by MACE |
| `relabel_mm_baseline_1d_pmf.csv` | optional MM baseline relabel floor |
| `relabel_pmf_summary.json` | MACE endorsement summary |
| `relabel_dexp_1d_pmf.png` | DEXP-world PMF plus MACE-minus-DEXP residual profile |

## How To Judge The DEXP Potential

Do not judge DEXP by asking whether MM was right. Judge it in this order.

### 1. Can It Run MD?

Basic pass/fail:

| sign | desired behavior |
| --- | --- |
| temperature | stable near target temperature |
| total energy | no explosion or runaway drift |
| forces | finite and not pathological |
| ligand RMSD/contact | no immediate collapse or atom crossing |

This only says the Hamiltonian is runnable. It does not prove the surface is good.

### 2. Does It Match MACE On Fitting/Holdout Frames?

Use:

```text
fit_all_accepted_mace_surrogate_pmf_1d.csv
fit_all_accepted_mace_surrogate_pmf_2d.csv
fit_holdout_validation.csv
fit_holdout_pmf_profile.csv
```

The useful quantities are profile shape and bin means along physical CVs:

```text
min ligand-environment distance
min distance + short-range contact count
```

Per-frame RMSE can look bad even when the useful mean surface is acceptable, because MACE contains many-body and environment-dependent effects that a compact DEXP pair form cannot exactly represent.

### 3. Does MACE Endorse DEXP-Sampled Structures?

This is the main test.

Run:

```bash
python dexp_experiment.py ^
  --relabel-traj output/dexp_experiment/dexp_surrogate/traj.dcd ^
  --relabel-baseline-traj output/dexp_experiment/original_baseline/traj.dcd
```

For DEXP frames:

```text
delta = E_MACE_local - (E_gaussian_coul + E_DEXP)
```

For MM baseline frames:

```text
delta = E_MACE_local - (E_mm_coul + E_mm_vdw)
```

The relabel code removes the arbitrary constant offset before judging shape:

```text
delta_shape = delta - mean(delta)
```

Main relabel metrics:

| metric | interpretation |
| --- | --- |
| `shape_profile_rmse_kjmol` | distance-dependent disagreement after removing legal zero offset |
| `mean_residual_within_sem_bins / n_bins` | how many bins are consistent with zero shape residual within SEM |
| `zero_offset_kjmol` | legal constant energy offset, not by itself a failure |
| `too_close_fraction` | fraction of frames below `--relabel-min-dist-floor` |
| `reweight_ess_global_fraction` | reweighting health; often collapses when energy noise is large |

The floor criterion is:

```text
DEXP shape RMSE should be <= MM baseline shape RMSE
DEXP too-close fraction should be <= MM baseline too-close fraction
```

If DEXP fails this, it means DEXP generated a world that MACE likes less than the MM floor.

### 4. Check Contact Distributions And Switch Artifacts

Use:

```text
le_min_distance_comparison.csv/png
le_rdf_comparison.csv/png
le_pmf_1d_comparison.csv/png
lambda_window_rdf.csv/png
lambda_window_min_distance.csv/png
```

Important red flags:

| symptom | likely meaning |
| --- | --- |
| DEXP min-distance much smaller than baseline | short-range attraction too strong or wall too soft |
| RDF spike in the switch shell | cutoff/switch artifact |
| lambda windows change contact distance non-smoothly | alchemical coupling/surrogate seam issue |
| too many frames below relabel floor | atom crossing or MACE OOD geometry |

The script separates core-window RDF from surrogate switch-zone RDF when fitted parameters are available. A peak inside the switch zone should not be interpreted as a real MACE-local structural feature without relabel confirmation.

## Learned Pair-RBF Diagnostic

`--learned-rbf-diagnostic` fits an offline pair-RBF model grouped by ligand/environment element type buckets.

It is diagnostic only:

```text
it does not change the OpenMM DEXP force
it does not run MD
it answers whether a slightly more flexible learned local function would fit the same labels better
```

Use it to decide whether the analytic DEXP form is too stiff. If RBF profile RMSE improves strongly while DEXP MD remains stable, that is evidence for adding a learned local correction later.

## Current Interpretation Rules

Good DEXP:

```text
stable MD
low too-close fraction
MACE relabel shape residual small after removing constant offset
DEXP relabel floor is better than or comparable to MM baseline
RDF/min-distance has no narrow switch-shell artifact
1D/2D MACE-vs-surrogate profiles have the same qualitative shape
```

Bad DEXP:

```text
runs hotter or unstable
samples much shorter ligand-environment contacts than both baseline and fitting data
has a narrow RDF bump near the switch/cutoff shell
MACE relabel says DEXP-sampled structures have large distance-dependent residuals
learned RBF fits well but analytic DEXP does not
```

Ambiguous DEXP:

```text
per-frame holdout RMSE is high but profile shape is reasonable
FEP/reweighting ESS collapses but bin mean residuals are stable
MM and DEXP trajectories differ, but MACE relabel prefers DEXP
```

In this project, the final scientific question is not whether DEXP reproduces MM. The useful question is:

```text
Can DEXP provide a stable local MD potential whose sampled 1D/2D PMF surface is accepted by MACE?
```

