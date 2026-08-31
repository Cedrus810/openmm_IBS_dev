# Proposal: let frozen validation run once even if the candidate streak never completes

## Context

Nine real bugs in the online-learning candidate gate got fixed today (TMBAR
sign inversion, two roundoff-tolerance gaps, a hardcoded `10` ignoring
`min_frames_per_window`, an unreachable `candidate_min_decorrelated_samples=5`
threshold) — all confirmed via `diagnose_window0_converged_gate.py` re-running
the real saved `tmbar_history`. The candidate gate itself (per-update
diagnostic used to decide "is this f_k worth freezing") now genuinely returns
`converged=True` on real data.

Separately, the user pushed back hard on the broader design: production/
frozen-validation sampling is comparatively cheap next to the 500,000-step
learning phase that's already been paid for, and the pipeline currently never
even attempts frozen validation unless a noisy per-update heuristic happens to
pass 3 times in a row first.

## What the code actually does (`ibs_engine.py`, `run_all_windows`)

- The only transition from `mode="learning"` into `freeze_burn_in`/
  `validating` is `consecutive_pass_count >= required_consecutive_bias_updates`
  (default 3 **consecutive** candidate-gate passes; lines ~7075-7093). A
  single noisy failure resets the streak to 0.
- If `bias_update_count >= max_bias_updates` (50) while still
  `mode=="learning"` (streak never completed), line ~6997-6998 just `break`s,
  and the window falls straight through to `raise IBSWarmupConvergenceError`
  (line ~7875) — frozen validation is **never attempted**.
- Frozen validation itself (`freeze_burn_in` -> `validating`) is a separate,
  strict, already-existing check: discards a fresh 20k-step burn-in, then
  requires several validation batches of newly-collected samples under the
  truly fixed Hamiltonian to satisfy `max_abs_log_residual <=
  lse_log_residual_tolerance` (0.25). It does not reuse or trust the
  candidate diagnostic at all — it's a completely independent test.
- The 500k (learning) + 20k (burn-in) + 50k (validation) = 570k total step
  budget already reserves the burn-in/validation steps regardless of
  outcome. The failed run only consumed `500000/570000` — the other 70k was
  already paid for and simply never spent.
- `non_mutating_v1`'s explicit bans (verbatim): no fixed-H overlap probes, no
  asymmetric-slope judging, no in-place bias/MBAR recalibration, no reusing
  cross-policy cached f_k. It says nothing about "attempt frozen validation
  once with the current f_k when the learning budget runs out."

## Proposed change

Replace the bare `break` at the `bias_update_count >= max_bias_updates`
exhaustion point with: if still `mode=="learning"`, do the same freeze
transition already implemented for the candidate-streak path (snapshot
current f_k, switch to `freeze_burn_in`, reset burn-in/validation counters) —
guarded so it can only happen once per window attempt — instead of breaking
immediately. Mark the diagnostics so a later failure report distinguishes
"reached via budget-exhaustion fallback" from "reached via genuine candidate
streak." Everything downstream (freeze_burn_in -> validating -> accept/reject)
is completely unchanged; if frozen validation also fails, the window still
fails closed exactly as today, just after getting a real shot first.

Bump `IBS_BIAS_PROTOCOL_VERSION` since this changes when a window can reach
`calibrated_pending_validation`/frozen.

## What this does NOT do

- Does not touch the lambda schedule, window grouping, or ensemble.
- Does not weaken frozen validation's own acceptance thresholds.
- Does not remove fail-closed behavior — if frozen validation also fails,
  same `IBSWarmupConvergenceError`/`IBSFrozenCalibrationValidationError` as
  today.
