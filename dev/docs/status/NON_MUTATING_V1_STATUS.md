# IBS `non_mutating_v1` — Status & Handoff (2026-07-18)

> **2026-07-22 状态更新：本文关于“冻结占据残差必须严格通过才可进入生产”、
> “production quality 失败只终止、不自动补采”和“禁止新建 rescue ensemble”的描述
> 已被 [IBS_PRODUCTION_PROTOCOL_2026-07-22.md](IBS_PRODUCTION_PROTOCOL_2026-07-22.md)
> 取代。仍然有效的不变量是 production 内 `f_k` 只读、原 production 文件不原地
> 改写、rescue 不移动或插入 λ 坐标。本文其余内容作为 2026-07-18 历史审计保留。

One-line: the pipeline had grafted a **replica-FEP-MBAR adjacent-overlap arbiter** and an overlapping local-window stitcher onto an **IBS** engine. `non_mutating_v1` seals the production-time mutation paths, and thermodynamic-path protocol v12 first places λ nodes by thermodynamic length, then groups roughly 4–5 conventional λ intervals into each few-state IBS ensemble. Every λ interval belongs to exactly one ensemble; neighboring ensembles still reuse their single boundary λ node as a common free-energy reference, just as the paper's `[0,0.5]` and `[0.5,1]` Stage-2 ensembles share `0.5`. There is no fixed λ=0.5 cut and no legacy `overlap=2` rule that shares two nodes and therefore duplicates an interval. Target-environment pytest and GPU validation are still pending, so `output_lrc_fix` remains under an operational resume hold.

GitHub tracking: target-environment fail-closed regressions are
[`#30`](https://github.com/Cedrus810/openmm_IBS_dev/issues/30); the hash-bound,
read-only `output_lrc_fix` rescue audit is
[`#31`](https://github.com/Cedrus810/openmm_IBS_dev/issues/31).

Reference method: **Integrated Boltzmann Sampling (IBS)** — Lin, Xia, Zhang, Gao, *JCTC* 2026, doi 10.1021/acs.jctc.5c01240 (`integrated-boltzmann-sampling.md`). This repo is a **differentiable, converge-then-freeze, λ-only (fixed 300 K)** variant for absolute BFE of Atenolol (Boresch + DEXP/MACE). NOTE: `a-relative-binding-free-energy-framework-…md` (CBFE/ACES) was sent by mistake — different method, reference only.

Full plan: `~/.claude/plans/a-relative-binding-free-energy-framewor-immutable-starlight.md` (non-portable external path; it is not present in this workspace and must be copied into the repo before handoff).

---

## Why (the false premise)

In IBS, each state's free energy is recovered by **reweighting one frozen integrated mixture → each target state** — never adjacent→adjacent. A steep ΔF is exactly absorbed by the weights `a_k = q_k·e^{βf_k}`. Adjacent-state phase-space overlap is a property of **pairwise BAR/MBAR** (the f_k calibrator), *not* of the Boltzmann/IBS identity. The old auto-repair used adjacent fixed-H overlap (`min_overlap ≥ 0.03`) as the *arbiter of IBS path correctness* and mutated the grid to chase it — which also **re-randomized the frozen f_k / λ-schedule / state set**, destroying the deterministic, differentiable frozen reference the method depends on. This burned a week of GPU with no ΔG.

Also settled: in the production analyzer `GlobalMBARAnalyzer.solve_stage_integrated`, `result["min_overlap"]` is the **single-reference importance-ESS ratio** from `mbar.compute_effective_sample_number()` — NOT fixed-H overlap — so it is a **legitimate hard gate and is kept**.

### λ design boundary (2026-07-18)

The initial Stage-2 λ nodes come from the pilot thermodynamic-length metric, so λ→0 can receive more, tighter nodes. Grouping happens only after those physical λ coordinates are fixed. For the configured 18-state path, the 17 thermodynamic intervals are partitioned without duplication as `[0:5],[5:9],[9:13],[13:17]`, i.e. counts `[5,4,4,4]`. The corresponding IBS node ranges are `(0,6),(5,10),(9,14),(13,18)` because boundary nodes `5,9,13` are intentionally present in both neighboring ensembles to provide a common free-energy reference. Thus there are 18 unique λ nodes, 17 unique λ intervals and zero duplicated intervals—not “zero shared nodes.” The old fixed λ=0.5 split, `pilot_overlap_thermodynamic_length`, and legacy `overlap=2` controls do not decide the vanishing layout. There are no recursive schedule probes and no sliding K=2 chain.

For the K-state integrated ensemble, learning implements the paper's Sec. 2.3 time average with the normalization of the time-dependent IBS distribution retained. At iteration `t`, `n_k^(t)=exp(βf_k^(t))`, `Ẑ_t=Σ_j n_j^(t)Q̂_j^TA`, and the common-scale contribution is `Q̂_{k,t}=Ẑ_t·<p_k>_t/n_k^(t)`; these contributions—not separately normalized batch vectors—are averaged over iterations. The update is then `n_k ∝ 1/Q̂_k^TA`. Candidate convergence checks the pre-update balance `n_old,k·Q̂_k^TA`. After freezing, the code discards 20k burn-in steps and accumulates multiple new batches under that unchanged `f_k`; only their cumulative `r_k=log(K·<p_k>)` can pass the final gate. A single 20-frame failure no longer returns immediately to learning. A complete failed validation attempt may join the TA estimator only after it has rejected that candidate, and the next candidate is tested with fresh held-out data. Coverage ESS and adjacent Δu remain diagnostic only. If this ensemble does not converge within its explicit budget, the run stops with the LSE/TA diagnostics; it does not auto-split, insert λ, invoke adjacent fixed-H/BAR/MBAR, or fall back to the local-TMBAR stitcher. Once production sampling starts, λ, state membership and frozen `f_k` remain immutable.

---

## Round 1 — code complete (9/9; compiles; target-environment pytest/GPU pending)

| # | What | Where |
|---|---|---|
| P0-1 | Outer repair loop bypassed: stage runs once → hard gates → return; `while True` loop unreachable, marked `deprecated_non_mutating_policy` | `abfe_pipeline.py::_run_stage_with_overlap_autorepair` |
| P0-2 | Inner bypass: `run_all_windows(repair_policy=…)`; vdw `not bias_converged` branch (fixed-H probe / asymmetric arbiter / calibration probe / in-context `setParameter(f_k)`) gated off → surfaces `IBSWarmupConvergenceError` | `ibs_engine.py::run_all_windows` |
| P0-3 | `sampling_repair_policy` in `_stage_protocol_key`, per-window `convergence.json`, and both reuse checks (stage-level + per-window) | `abfe_pipeline.py`, `ibs_engine.py` |
| P0-4 | No-GPU tests written: single outer `run_once`, zero mutators/probes, hard-gate propagation, old/missing-policy `ibs_state` rejection before `f_k` injection, matching-state positive resume, and old energy-cache rescue propagation with file hash/mtime preservation. **Not yet executed in the target environment.** | `test_non_mutating_policy.py` |
| P0-5 | "increase λ density / insert λ" advice → "preserve data, run rescue/coverage analysis, don't mutate the grid in place" | `abfe_pipeline.py::_assert_stage_result_sane` |
| P0-6 | `sampling_repair_policy` is written to new `ibs_state`, main-window and production checkpoint manifests. `IBSSampler` defaults fail-closed; `load_ibs_state()` now compares cached/current policy before reading/injecting `f_k`, and re-raises `ExistingEnsembleRequiresRescueAudit` instead of swallowing it. | `ibs_engine.py` |
| P0-7 | Old-policy on-disk energies raise `ExistingEnsembleRequiresRescueAudit`; the structured exception now bypasses the generic cache-load handler, so execution cannot fall through to system construction/re-sampling. Regression asserts zero build calls and unchanged hashes/mtimes. | `ibs_engine.py::run_all_windows` reuse branch + test |
| P0-8 | Under non-mutating: `feedback_action = "none_non_mutating_f_not_converged"`; `IBSWarmupConvergenceError` message drops split/insert advice | `ibs_engine.py` |
| P0-9 | `should_run_legacy_repair(policy)`: `non_mutating_v1`→False, `legacy_mutating`→True, else `ValueError` (typo fails closed); single gate in `run_all_windows`; unit-tested | `ibs_engine.py` + test |

New symbols: `ExistingEnsembleRequiresRescueAudit`, `should_run_legacy_repair` (both `ibs_engine.py`).

### Post-review defects and merged fixes (2026-07-17)

#### NM-01 — rescue-audit exception was swallowed (**P0; code fixed, regression written, target pytest pending**)

The per-window energy reuse branch formerly raised `ExistingEnsembleRequiresRescueAudit` inside a broad `except Exception`, which swallowed the structured stop signal and allowed fall-through to `_build_window_system()`. The handler now has an explicit `except ExistingEnsembleRequiresRescueAudit: raise` before the generic load/parse handler. The new regression constructs a legacy energy cache, asserts the exact exception type, proves `_build_window_system` is never called, and compares input file SHA256/mtime snapshots before and after.

Acceptance status:

- code path: **implemented and statically confirmed**;
- exact exception / zero system construction / unchanged fixture hash+mtime: **regression written, not yet run in `openmm_dev`**;
- real `output_lrc_fix` preservation: **not tested and must not be used as a test fixture**.

#### NM-02 — `ibs_state` policy was saved but not enforced on load (**P0; code fixed, regressions written, target pytest pending**)

`load_ibs_state()` now compares cached/current `sampling_repair_policy` immediately after JSON parsing and before reading or injecting `f_k`. Under `non_mutating_v1`, a missing or mismatched cached policy raises `ExistingEnsembleRequiresRescueAudit`; that exception is explicitly re-raised before the generic state-load handler. `IBSSampler.__init__` also defaults the policy to `non_mutating_v1` so direct sampler users fail closed if the caller forgets to stamp it. The positive regression verifies that a matching state still restores normally.

Acceptance status:

- pre-injection policy check and structured exception propagation: **implemented and statically confirmed**;
- missing/legacy policy, zero `setParameter(f_k)`, unchanged state file, and matching-policy positive resume: **regressions written, not yet run in `openmm_dev`**.

> **Operational stop remains:** NM-01/NM-02 are code-fixed, but do **not** run `--resume` against `output_lrc_fix` until the new no-GPU regressions pass in `openmm_dev` and their logs are archived. The legacy ensemble must not be the first live test of these guards.

### Kept as hard gates (in order)
1. **Occupation / IBS fixed point** — learning uses the normalized-IBS cross-iteration TA `Q̂_k` estimator above; after freeze burn-in, cumulative fresh samples under one unchanged `f_k` must satisfy `max|log(K·<p_k>)| <= ibs_lse_log_residual_tolerance`. Raw-occupation EMA, coverage ESS and adjacent Δu are diagnostic only.
2. **Importance-ESS / absolute ESS / decorrelated samples / endpoint uncertainty** on the frozen production (`_assert_stage_result_sane`).

### Demoted to unused diagnostic
Adjacent fixed-H overlap, asymmetric-overlap / ΔF-slope arbiter, and their λ-insertion suggestions (live only inside the now-dead loop + gated-off inner branch).

---

## NOT implemented yet (deferred by plan — nothing dropped)

- **P0-7 alt**: "force-write to a new ensemble-ID directory" (chose preserve+raise instead).
- **Invariant #1** unified immutable ensemble-ID (FP λ values + π_k + f_k + T + WCA/common-H + Boresch + potential version + LRC protocol → one ID). Only `sampling_repair_policy` was added to existing per-field fingerprints.
- **Invariant #2** explicit `a_k = q_k·e^{βf_k}` for non-uniform λ (q_k into responsibility + fingerprint).
- **Part B** stronger occupation uncertainty contract: decorrelation, block-bootstrap CI and validation halves around the current Log-Sum-Exp residual gate. (Learning uses the Sec. 2.3 normalized-IBS cross-iteration TA estimator; the final gate uses cumulative fresh fixed-`f_k` samples after a separate burn-in. Coverage ESS and adjacent Δu are diagnostic only.)
- **Part C** dense/adaptive target-grid importance-ESS map; contiguous-hole confirmation across trajectory halves; human-gated new-ensemble **proposal** emission. (Failures currently surface as `reweighting_quality_failed_at_evaluated_targets`, never `confirmed_effective_support_hole`.)
- **Invariant #4** block-bootstrap uncertainty everywhere; missing-mode via replicas/CV.
- **Invariant #5** immutable analysis products (new `analysis_id` dir for re-analysis).
- **Part E** global-GLS stitching + full covariance + seam χ² + bootstrap. (Current: sequential inverse-variance offset + incomplete covariance, `ibs_engine.py:~8151/~8206`.)
- **Boresch/stage composition** verification (fully-on common Hamiltonian `ibs_engine.py:~439`; 1 M release; cycle `abfe_core.py:~853`).
- **Rescue-audit execution** (classify `output_lrc_fix` windows as clean/tainted/indeterminate; authorize clean files through a hash-bound sidecar manifest or copy them into a new ensemble-ID directory — never re-stamp the originals in place).
- **Full excision** of the dead outer loop + inner legacy branch.
- **Execution and archival of the new resume fail-closed regression suite** for old energy caches and old `ibs_state` files (NM-01/NM-02 above).
- **In-code/documentation cleanup**: `_run_stage_with_overlap_autorepair` and `run_all_windows` docstrings, README files, `AUDIT_STATUS.md`, `VALIDATION_MATRIX.md`, and `todolist.md` still describe fixed-H calibration / split / insert / `reseed_resample` as active production behavior.

---

## Consequence for `output_lrc_fix` + the next step

`output_lrc_fix/vanishing` holds **energy/base/bias arrays but no per-frame coordinates**. Windows **0/1/3/4/7/8/9 have legacy `convergence.json` files whose `bias_warmup.status` says `frozen_validation_converged`**; window 2 has a warmup-failure file; 5/6 are absent. Every existing convergence/state file inspected here has a missing/null `sampling_repair_policy`, so none is currently proven reusable under `non_mutating_v1`. These are **candidate audit artifacts**, not yet `non_mutating_v1`-validated or stage-level scientifically converged results.

Window 9 contains a large `λ_vdw 0.769→0.0` interval. This is a **coverage-risk candidate**, not a confirmed effective-support hole: dense-target ESS mapping and trajectory-half confirmation are Part C and have not been implemented.

NM-01/NM-02 are now fixed in code, but the new guards have not yet been executed in the target dependency environment. A plain rerun remains operationally prohibited until those regressions pass and the rescue audit runs; `output_lrc_fix` must not be used as the first validation fixture.

**Rescue audit (next, read-only):** classify each existing candidate window as `clean_reusable` (pure-SGD frozen `f_k`, never recalibrated), `tainted` (old path rewrote `f_k` or mixed incompatible sampling), or `indeterminate` (insufficient provenance; fail closed). Do not assume that only window 2/5/6/endpoint need reruns until this classification is complete. Off-grid re-gridding is impossible because no per-frame coordinates were saved for `vanishing`.

### Rescue-audit contract

1. Before interpretation, inventory every relevant `.npy`, `.json`, OpenMM checkpoint and manifest with absolute path, size, mtime and SHA256. Preserve that immutable inventory as the audit input.
2. Associate evidence by exact λ content/window range and file hash, **not by `window_idx` alone**: the legacy repair loop split, inserted and renumbered windows. `sampling_repair_decisions.json` already records `recalibrate_f_k` decisions in multiple rounds, and those historical indices must be mapped back to λ ranges before declaring a current file clean.
3. Use three outcomes: `clean_reusable`, `tainted`, `indeterminate`. Absence of an `mbar_calibration` field is supporting evidence, not by itself proof that no later external/outer repair rewrote the state.
4. Do **not** edit original legacy `convergence.json`/`ibs_state` files merely to add `sampling_repair_policy="non_mutating_v1"`. That would replace missing provenance with an assertion and destroy the clean before/after evidence. Emit a read-only sidecar rescue manifest (original hashes + evidence + decision + reviewer + policy version), or copy approved artifacts into a new ensemble-ID directory.
5. A loader may accept rescued data only through an explicit, versioned rescue-manifest contract whose hashes still match. Any changed/missing file, ambiguous index mapping or incomplete evidence becomes `indeterminate` and requires a new ensemble.
6. Only after classification may the rerun set be stated. Preserve the originals even for windows judged tainted.

---

## Verification status and required evidence

Current evidence on this workspace:

- in-memory syntax compilation of `abfe_pipeline.py`, `ibs_engine.py`, and `test_non_mutating_policy.py`: **PASS**;
- existing no-GPU pytest: **WRITTEN, NOT RUN HERE** — the available Python lacks OpenMM and pytest, and `conda` is not on PATH;
- NM-01/NM-02 disk-resume regressions: **WRITTEN, NOT RUN HERE**;
- GPU sanity: **NOT RUN**.

Every verification record should include date/time, host, Python/OpenMM/PyMBAR/pytest versions, exact command, exit code, full log path, source snapshot ID, and hashes of any data used.

```bash
# 1) No-GPU control-flow + enum test
conda run -n openmm_dev python -m pytest test_non_mutating_policy.py -v

# 1b) Required before any output_lrc_fix resume: disk fail-closed tests
#     - old/missing-policy energies -> ExistingEnsembleRequiresRescueAudit
#     - old/missing-policy ibs_state -> ExistingEnsembleRequiresRescueAudit
#     - exact exception propagates; zero Context/sampling/setParameter(f_k)
#     - all fixture file hashes/mtimes unchanged

# 2) One GPU sanity run (SMALL stage/window, NOT full production):
#    expect no fixed-H GPU, no f_k rewrite; either completes via the two hard
#    gates, or raises f_not_converged / ExistingEnsembleRequiresRescueAudit
#    without mutating λ / f_k / ensemble fingerprint.
```

Static check already done: `python3 -m py_compile abfe_pipeline.py ibs_engine.py test_non_mutating_policy.py` → OK; grep confirms no path reaches split/insert/renumber/invalidate/recalibrate/fixed-H probe/`setParameter(f_k)` under `non_mutating_v1`.

The previous grep-only gap for old-state `f_k` injection and swallowed rescue exceptions is now covered by explicit code guards and targeted tests; acceptance still requires executing those tests in `openmm_dev`.

---

## Source snapshot and documentation consistency

This workspace does not currently provide a usable Git commit identity (`.git` contains only an incomplete `info/` directory), so the reviewed source baseline must be identified by content hash until normal version control is restored:

| File | SHA256 reviewed on 2026-07-17 |
|---|---|
| `abfe_pipeline.py` | `FE2C62095DCA078F23159E3C0E79DDB5FE3650D554AAFD343587F293D049176C` |
| `ibs_engine.py` | `2CBBCE11C0085385388537A6203C407B8A6DD123156B5FAB2E5C17E6CAAF4098` |
| `test_non_mutating_policy.py` | `54292313EB2BCDEFB4682B941864882B22DF05519585CCA16CFE57B0BCF387EA` |

Documents that currently conflict with this handoff must be marked superseded or updated before release:

- `VALIDATION_MATRIX.md` still treats fixed-H calibration, automatic insertion and `reseed_resample` as active production validation targets (notably VAL-GPU-001 through VAL-GPU-005);
- `README.md`, `README_cn.md`, and `README_en.md` still advertise automatic recalibration/resampling artifacts and recommend denser windows based on legacy overlap language;
- `AUDIT_STATUS.md` and `todolist.md` retain earlier mutable-policy conclusions;
- `_run_stage_with_overlap_autorepair` and `run_all_windows` docstrings describe behavior that is now unreachable or fail-closed;
- the full plan must be moved from the user-home `~/.claude/plans/...` path to a repository-relative, versioned file.

Until those are synchronized, **this file is the authoritative status for `non_mutating_v1`**, subject to NM-01/NM-02 above.
