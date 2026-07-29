# BOR-01 Boresch dihedral sign bug archive

Date: 2026-07-29

## Status

Fixed and verified. The active TODO keeps only follow-up work:

- BOR-02: add a committed-geometry deviation gate for `update_boresch_from_last_frame`.
- BOR-05: feed real ligand bond topology into Boresch anchor selection.
- P1-19: continue uncertainty / sampling work for the vanishing leg.

## Bug

Four hand-written Boresch dihedral implementations in `abfe_core.py` returned the
mirror convention, effectively `-phi`, while OpenMM `dihedral()` and mdtraj use
the standard IUPAC convention.

The contaminated path was especially dangerous in
`calc_boresch_from_last_frame`: it could overwrite the correct
`boresch_simple.json` equilibrium values with mirrored dihedral references.
That put the Boresch attachment lambda=0 ensemble near the top of the
`k * (1 - cos(delta))` wall instead of near the intended pose.

## Root Cause

The local formula used the opposite triple-product sign:

```text
(n1 x b2_hat) . n2 = -(n1 x n2) . b2_hat
```

Distances and angles were unaffected. Only the three dihedral equilibrium values
were mirrored.

## Fix

The duplicate formulas were replaced by the single module-level helper
`abfe_core.boresch_dihedral_rad()`.

Known call sites fixed:

- `OrbBoreschEstimator.estimate_from_trajectory`
- `_finalize_candidate`
- `scan_boresch_1d_pes._calc_geom`
- `calc_boresch_from_last_frame`
- Boresch injection / resume validation paths in `abfe_pipeline.py` and
  `runabfe.py`

`BORESCH_GEOMETRY_CONVENTION_VERSION` stayed at 2. The cached
`simple`/`fluctuation` geometry values were numerically correct; the bug was the
last-frame auto/orb overwrite path, now fixed at source.

## Verification

Added convention tests that do not reimplement the dihedral formula:

- compare against OpenMM `dihedral()`
- compare against mdtraj
- assert the fixture has enough sine leverage to catch sign flips
- assert end-to-end Boresch energy is near zero at the committed pose
- assert mirrored dihedrals make the restraint energy jump

The fixed full rerun reported:

```text
complex leg Delta G = 181.00 +/- 1.76 kJ/mol
solvent leg Delta G = 157.84 +/- 1.79 kJ/mol
Boresch attachment = 4.39 +/- 0.08 kJ/mol
analytic correction = -38.76 kJ/mol
Delta G_bind = -23.16 +/- 2.51 kJ/mol = -5.54 +/- 0.60 kcal/mol
```

Against `result.txt` total `-6.279 +/- 0.457 kcal/mol`, the difference is
`+0.74 kcal/mol`, about `0.98 sigma` with the optimistic reported uncertainty.
The old `-9.76 kcal/mol` sign-bug result should no longer be cited.

## Follow-up

BOR-02 remains important because the old last-frame update gate checked only
`r0` and the two angles. It should compare all six committed coordinates,
wrapping dihedrals to pi, and on excess deviation warn while preserving the
original ensemble-derived equilibrium values.
