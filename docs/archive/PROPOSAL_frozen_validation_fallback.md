# Proposal: let frozen validation run once even if the candidate streak never completes

> **📦 已归档（2026-08-31）。** 本提案已在 `IBS_BIAS_PROTOCOL_VERSION = 25` 实施，
> 其引入的 fallback 与诊断字段至今保留；但它正文描述的「3 次连续候选通过」gate
> 已被 Candidate-first / Validate-or-Learn v1 整体撤掉。
> **不是待办，也不要拿正文第 2 节当作当前 gate 的描述**——当前逻辑读
> `ibs_engine.py` 顶部 v24→v32 版本史。`ibs_engine.py:6306` 的注释按文件名引用本文件。

日期：2026-07-20（撰写）／2026-08-31（状态复核）
状态：**已实施（`IBS_BIAS_PROTOCOL_VERSION = 25`），随后被后续协议改写覆盖。
本文件转为历史提案，不再是待办。**

## 0. 实施状态（2026-08-31 逐条核对源码）

| 提案内容 | 现状 |
|---|---|
| 用「一次性冻结验证 fallback」替换 `max_bias_updates` 耗尽处的裸 `break` | ✅ v25 落地，`budget_fallback_used` guard 保证每个 window attempt 只触发一次（`ibs_engine.py:13662`） |
| 诊断上区分「预算耗尽 fallback」与「真候选连胜」 | ✅ 落盘字段 `reached_freeze_via_budget_exhaustion_fallback`（`ibs_engine.py:14335`） |
| Bump `IBS_BIAS_PROTOCOL_VERSION` | ✅ v25；当前已推进到 **32**（`ibs_engine.py:6427`） |
| 不动 λ 调度／窗口分组／ensemble | ✅ |
| 不放宽冻结验证自身阈值、保持 fail-closed | ✅（但 v25 之后阈值体系整体被重设计，见下） |

### 提案的前提已经不成立（重要）

本文件第 2 节描述的 gate 是「`consecutive_pass_count >= required_consecutive_bias_updates`
（默认 3 次**连续**候选通过）才能离开 learning」。**这套判据后来被整体撤掉了。**

- v25 上线后首次真机 GPU 全流程即暴露问题：5 个窗口全部在
  `learning_to_validation_cycles==2` 被 best-effort 接受，接受时 `best_effort_residual`
  高达 13~123（容差 0.5 的 26~246 倍），Stage 2 最终 `_assert_stage_result_sane`
  以 `min_overlap=0.0047` 失败（见 `ibs_engine.py` v26 版本史）。根因是 v25 只加了重试
  次数上限、没加「接受结果本身是否合理」的下限检查。
- 当前生效的是 **Candidate-first / Validate-or-Learn v1**：LEARN 只调占据度
  （`Δf_k = -η·kT·log(K·p_k)`），**没有 batch 计数器、没有连胜 streak**，
  唯一进生产的证明是真 Hamiltonian 的 local-MBAR loose gate（`ibs_engine.py:13755-13767`）。
- 预算耗尽的处理也变了：现在是 v29 的「预算耗尽即接受当前 f_k、由下游放行进生产」
  （`ibs_engine.py:13743-13748` 注释），正确性门槛完全落在生产后的
  `_assert_stage_result_sane`（`min_overlap`/绝对 ESS/去相关样本数/端点不确定度）。

**结论：提案要求的行为已经存在且更进一步，但不要拿本文件第 2 节当作当前 gate 的描述。**
要读当前逻辑，直接看 `ibs_engine.py` 顶部 `IBS_BIAS_PROTOCOL_VERSION` 版本史
（v24 → v32 那一段）和 `run_all_windows`。

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
