# PyMBAR uncertainty protocol

## Frozen dependency

The production and CPU-CI environments both require `pymbar-core=4.2.0`.
The version is pinned because reported ABFE uncertainties must not silently
change when a new environment is solved.

## Intended estimator semantics

- The current MBAR compatibility layer does not pass an explicit
  `uncertainty_method`.
- Under the pinned PyMBAR 4.2.0 contract, `None` selects the asymptotic
  `svd-ew` covariance method.
- This pin stabilizes library behavior; it does not establish that asymptotic
  covariance is sufficient for short or slowly drifting runs.
- Split-half drift calibration remains tracked separately by issue #78.
- The vdW/stage2 finite-sample protocol remains separate under issue #87.
  Pinning PyMBAR must not enable BAR, TI, bootstrap, all-frame estimates, or
  `sqrt(g)` inflation in production.

## Evidence and provenance

`env3.txt` records a resolved production environment containing
`pymbar-core 4.2.0`. Runtime evidence should continue to record the imported
`pymbar.__version__`, estimator path, and any explicit uncertainty method.

Any future version change requires a deliberate environment edit, regression
of known MBAR matrices and uncertainty outputs, updated provenance/migration
notes, and confirmation that #78/#87 semantics were not changed implicitly.
