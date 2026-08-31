# GitHub Issues 快照 — `Cedrus810/openmm_IBS_dev`

- 抓取日期：2026-08-31
- 来源：[GitHub Issues API](https://api.github.com/repos/Cedrus810/openmm_IBS_dev/issues?state=all&per_page=100)
- 仓库 issue/PR 条目数：88（包含 GitHub Issues API 返回的 Pull Request 条目）
- 说明：正文完整保留；`comments` 记录评论数量，评论线程正文未展开。

## 本轮代码修复后的关闭结论（2026-08-31）

下面是本轮确认可以关闭的 Issue。这里记录的是代码审查和回归验证结论；上面的状态汇总仍是 2026-08-31 抓取时的 GitHub API 快照，不因本地记录而自动修改远端 Issue 状态。

| Issue | 本轮结论 | 关闭依据 |
|---:|---|---|
| [#27](https://github.com/Cedrus810/openmm_IBS_dev/issues/27) | **关闭**（快照中已关闭） | 旧宽指纹可迁移到新窄指纹；未记录的字段只在明确未启用时补默认值；Hamiltonian、co-ion 和协议版本不匹配仍拒绝。回归覆盖迁移成功、版本升级拒绝和下一次直接命中。 |
| [#67](https://github.com/Cedrus810/openmm_IBS_dev/issues/67) | **关闭**（快照中已关闭） | analyze-only 与主流程共用阶段、窗口、manifest、冻结 `f_k`、收敛和覆盖校验；回退结果保留身份；缺窗、损坏或缺独立端点证据会 fail closed。 |
| [#75](https://github.com/Cedrus810/openmm_IBS_dev/issues/75) | **关闭** | 生产 checkpoint/convergence 持久化 resume/rebuild 边界；回退同步截断历史；主 MBAR、在线检查和 split-half 按段估计自相关，并记录边界处 base 能量跳变。 |
| [#83](https://github.com/Cedrus810/openmm_IBS_dev/issues/83) | **关闭** | Boresch 正常路径从 GROMACS 拓扑接入真实配体键，cache-only 路径使用 System 键/约束；simple/fluctuation 和 prepare 禁用几何猜键，缺少真实键图时明确失败。 |

本轮**只关闭 #27、#67、#75、#83**。其余仍为 Open 的 Issue 不因这四项修复自动关闭；其余 Closed 条目也保持原快照结论。

## 状态汇总

- Open：17
- Closed：71
- Pull Request 条目：1

| 编号 | 类型 | 状态 | 标题 | 标签 | 评论 |
|---:|---|---|---|---|---:|
| [#1](https://github.com/Cedrus810/openmm_IBS_dev/issues/1) | Issue | closed | [P2] Migrate production ESS fixed-H probes to a per-state trajectory bank | enhancement, p2 | 1 |
| [#2](https://github.com/Cedrus810/openmm_IBS_dev/issues/2) | Issue | closed | [Validation] Revalidate IBS v12 frozen-calibration resume on CUDA | p2, validation | 1 |
| [#3](https://github.com/Cedrus810/openmm_IBS_dev/issues/3) | Issue | closed | [Validation] Revalidate fixed-H lambda_shield NaN fix | p2, validation | 1 |
| [#4](https://github.com/Cedrus810/openmm_IBS_dev/issues/4) | Issue | closed | [Validation] Run current non_mutating_v1 OpenMM/PyMBAR regression suite | p2, validation | 2 |
| [#5](https://github.com/Cedrus810/openmm_IBS_dev/issues/5) | Issue | closed | [Validation] Revalidate fixed-H lambda_shield NaN fix | duplicate, p2, validation | 1 |
| [#6](https://github.com/Cedrus810/openmm_IBS_dev/issues/6) | Issue | closed | [Validation] Run full OpenMM and PyMBAR regression suite | duplicate, p2, validation | 1 |
| [#7](https://github.com/Cedrus810/openmm_IBS_dev/issues/7) | Issue | closed | [Audit][Fixed] IBS v10 separates path-overlap and bias-calibration ensembles | p2, audit, fixed | 0 |
| [#8](https://github.com/Cedrus810/openmm_IBS_dev/issues/8) | Issue | closed | [Audit][Fixed] IBS v11 introduces resumable per-state fixed-H probe banks | p2, audit, fixed | 0 |
| [#9](https://github.com/Cedrus810/openmm_IBS_dev/issues/9) | Issue | closed | [Audit][Fixed] IBS v12 exact bias and frozen-calibration resume state machine | p2, audit, fixed | 0 |
| [#10](https://github.com/Cedrus810/openmm_IBS_dev/issues/10) | Issue | closed | [Audit][Fixed] Exclude short fixed-H segments from MBAR | p2, audit, fixed | 1 |
| [#11](https://github.com/Cedrus810/openmm_IBS_dev/issues/11) | Issue | closed | [Audit][Fixed] Release stale fixed-H evaluator dynamics contexts | p2, audit, fixed | 1 |
| [#12](https://github.com/Cedrus810/openmm_IBS_dev/issues/12) | Issue | closed | [Audit][Fixed] Reject missing or invalid fixed-H volume records | p2, audit, fixed | 1 |
| [#13](https://github.com/Cedrus810/openmm_IBS_dev/issues/13) | Issue | closed | [Audit][Fixed] Clear production step overrides after lambda or window reindexing | p2, audit, fixed | 1 |
| [#14](https://github.com/Cedrus810/openmm_IBS_dev/issues/14) | Issue | closed | [Audit][Fixed] Make fixed-H probe and sampling-repair JSON writes atomic | p2, audit, fixed | 1 |
| [#15](https://github.com/Cedrus810/openmm_IBS_dev/issues/15) | Issue | closed | [Audit][Fixed] Fail closed on malformed analyze-only checkpoints and solver failures | p2, audit, fixed | 1 |
| [#16](https://github.com/Cedrus810/openmm_IBS_dev/issues/16) | Issue | closed | [Audit][Fixed] Fail closed when Boresch analytical correction cannot be computed | p2, audit, fixed | 1 |
| [#17](https://github.com/Cedrus810/openmm_IBS_dev/issues/17) | Issue | closed | [Audit][Fixed] Gate Shadow-Bridge and Shadow-IBS sublegs on convergence and overlap | p2, audit, fixed | 1 |
| [#18](https://github.com/Cedrus810/openmm_IBS_dev/issues/18) | Issue | closed | [Audit][Fixed] Unify Boresch force-constant sanitation ranges | p2, audit, fixed | 1 |
| [#19](https://github.com/Cedrus810/openmm_IBS_dev/issues/19) | Issue | closed | [Audit][Fixed] Share descending lambda-path invariant enforcement | p2, audit, fixed | 1 |
| [#20](https://github.com/Cedrus810/openmm_IBS_dev/issues/20) | Issue | closed | [Audit][Fixed] Correct dormant MBAR units, compatibility calls, and anchor RMSF reference | p2, audit, fixed | 1 |
| [#21](https://github.com/Cedrus810/openmm_IBS_dev/issues/21) | Issue | closed | [Audit][Fixed] Correct APBS grid volume and reject unsupported anisotropic boxes | p2, audit, fixed | 1 |
| [#22](https://github.com/Cedrus810/openmm_IBS_dev/issues/22) | Issue | closed | [Audit][Fixed] Integrate 2D geodesic long-move metric costs along the path | p2, audit, fixed | 1 |
| [#23](https://github.com/Cedrus810/openmm_IBS_dev/issues/23) | Issue | closed | [Audit][Fixed] Cache code_sha256 for the lifetime of the Python process | bug, p2, audit, fixed | 1 |
| [#24](https://github.com/Cedrus810/openmm_IBS_dev/issues/24) | Issue | closed | [Validation] Validate IBS v13 fixed-H Boresch Context on CUDA | p2, validation | 1 |
| [#25](https://github.com/Cedrus810/openmm_IBS_dev/issues/25) | Issue | closed | [Validation] Exercise batched production ESS auto-repair | p2, validation | 1 |
| [#26](https://github.com/Cedrus810/openmm_IBS_dev/issues/26) | Issue | closed | [Validation] Verify reseed_resample production checkpoint continuation | p2, validation | 1 |
| [#27](https://github.com/Cedrus810/openmm_IBS_dev/issues/27) | Issue | closed | [Validation] Complete preoptimization cache-key migration validation | p2, validation | 1 |
| [#28](https://github.com/Cedrus810/openmm_IBS_dev/issues/28) | Issue | closed | [Validation] Validate online TMBAR f_k learning (IBS v19) | p2, validation | 1 |
| [#29](https://github.com/Cedrus810/openmm_IBS_dev/issues/29) | Issue | closed | [P2] Design fail-closed adaptive pilot densification for the VDW tail | enhancement, p2 | 1 |
| [#30](https://github.com/Cedrus810/openmm_IBS_dev/issues/30) | Issue | closed | [Validation] Execute non_mutating_v1 fail-closed resume regressions | p2, validation | 1 |
| [#31](https://github.com/Cedrus810/openmm_IBS_dev/issues/31) | Issue | closed | [Audit] Build a hash-bound read-only rescue manifest for output_lrc_fix | p2, audit | 1 |
| [#32](https://github.com/Cedrus810/openmm_IBS_dev/issues/32) | Issue | open | [Validation] Verify traditional REMD v3 LRC delivery and per-frame correction |  | 1 |
| [#33](https://github.com/Cedrus810/openmm_IBS_dev/issues/33) | Issue | closed | [Validation] Validate Stage 2 v21 blended-metric lambda schedule on CUDA |  | 1 |
| [#34](https://github.com/Cedrus810/openmm_IBS_dev/issues/34) | Issue | closed | [Validation] Rebuild solvent leg with explicit 0.15 M NaCl cache |  | 1 |
| [#35](https://github.com/Cedrus810/openmm_IBS_dev/issues/35) | Issue | closed | [P2] Excise unreachable legacy auto-repair logic from production code |  | 0 |
| [#36](https://github.com/Cedrus810/openmm_IBS_dev/issues/36) | Issue | closed | [P2] Support finite-size electrostatic correction for anisotropic boxes |  | 1 |
| [#37](https://github.com/Cedrus810/openmm_IBS_dev/issues/37) | Issue | closed | [Research] Quantify Boresch non-harmonic release-correction error |  | 0 |
| [#38](https://github.com/Cedrus810/openmm_IBS_dev/issues/38) | Issue | open | [Research] Recalibrate Shadow-Coulomb windows and overlap thresholds |  | 1 |
| [#39](https://github.com/Cedrus810/openmm_IBS_dev/issues/39) | Issue | open | [Research] Evaluate point-ESS lambda insertion heuristic off production path |  | 1 |
| [#40](https://github.com/Cedrus810/openmm_IBS_dev/issues/40) | Issue | open | [Research] Assess ACE softcore, pilot-sampling, and PBC geometry sensitivities |  | 1 |
| [#41](https://github.com/Cedrus810/openmm_IBS_dev/issues/41) | Issue | open | [P2] Normalize runtime logging and assess multi-seed DEXP optimization |  | 0 |
| [#42](https://github.com/Cedrus810/openmm_IBS_dev/issues/42) | Issue | closed | [P0 candidate] Fix orphan scan_boresch_1d_pes self signature |  | 0 |
| [#43](https://github.com/Cedrus810/openmm_IBS_dev/issues/43) | Issue | closed | [P0 candidate] Resolve vanishing-subdomain default constant syntax |  | 0 |
| [#44](https://github.com/Cedrus810/openmm_IBS_dev/issues/44) | Issue | closed | [P0 candidate] Make single-GPU REMD replica Context allocation safe |  | 0 |
| [#45](https://github.com/Cedrus810/openmm_IBS_dev/issues/45) | Issue | closed | [P0 candidate] Make parallel stage workers spawn-safe |  | 0 |
| [#46](https://github.com/Cedrus810/openmm_IBS_dev/issues/46) | Issue | closed | [P0 candidate] Fail closed on missing IBS bias/base artifacts |  | 0 |
| [#47](https://github.com/Cedrus810/openmm_IBS_dev/issues/47) | Issue | closed | [P0 candidate] Unify DEXP fitted and production Hamiltonian models |  | 1 |
| [#48](https://github.com/Cedrus810/openmm_IBS_dev/issues/48) | Issue | closed | [P1 candidate] Remove or give semantics to solvent-leg topology argument |  | 0 |
| [#49](https://github.com/Cedrus810/openmm_IBS_dev/issues/49) | Issue | closed | [P1 candidate] Centralize thermodynamic-cycle sign convention |  | 0 |
| [#50](https://github.com/Cedrus810/openmm_IBS_dev/issues/50) | Issue | closed | [P1 candidate] Hard-gate intermittent IBS base-energy failures |  | 0 |
| [#51](https://github.com/Cedrus810/openmm_IBS_dev/issues/51) | Issue | closed | [P1 candidate] Replace single geometric bond-distance cutoff |  | 0 |
| [#52](https://github.com/Cedrus810/openmm_IBS_dev/issues/52) | Issue | closed | [P1 candidate] Reconcile fixed vanishing windows with adaptive schedule API |  | 0 |
| [#53](https://github.com/Cedrus810/openmm_IBS_dev/issues/53) | Issue | closed | [P1 candidate] Bound and observe TMBAR history checkpoint footprint |  | 0 |
| [#54](https://github.com/Cedrus810/openmm_IBS_dev/issues/54) | Issue | closed | [P1 candidate] Preserve configured DEXP cutoff and switch in IBS |  | 0 |
| [#55](https://github.com/Cedrus810/openmm_IBS_dev/issues/55) | Issue | closed | [P1 candidate] Select alchemical counterions with PBC-aware bulk criteria |  | 0 |
| [#56](https://github.com/Cedrus810/openmm_IBS_dev/issues/56) | Issue | closed | [P1 candidate] Propagate covariance into global TMBAR uncertainty |  | 0 |
| [#57](https://github.com/Cedrus810/openmm_IBS_dev/issues/57) | Issue | closed | [P1 candidate] Bind pre-equilibration cache to coordinates, box, and budget |  | 0 |
| [#58](https://github.com/Cedrus810/openmm_IBS_dev/issues/58) | Issue | closed | [P1 candidate] Correct solvent-box sizing from requested padding |  | 0 |
| [#59](https://github.com/Cedrus810/openmm_IBS_dev/issues/59) | Issue | open | [P2] Expand unit-test matrix for core physical contracts |  | 0 |
| [#60](https://github.com/Cedrus810/openmm_IBS_dev/issues/60) | Issue | open | [P2] Add public ABFE benchmark integration suite |  | 0 |
| [#61](https://github.com/Cedrus810/openmm_IBS_dev/issues/61) | Issue | open | [P2] Publish minimum user, input, API, and thermodynamic-cycle docs |  | 1 |
| [#62](https://github.com/Cedrus810/openmm_IBS_dev/issues/62) | Issue | open | [P2] Add layered CI, linting, typing, and formatting |  | 0 |
| [#63](https://github.com/Cedrus810/openmm_IBS_dev/issues/63) | Issue | open | [P2] Add runtime recovery and resource-protection controls |  | 0 |
| [#64](https://github.com/Cedrus810/openmm_IBS_dev/issues/64) | Issue | open | [P2] Strengthen CLI and physical input validation |  | 0 |
| [#65](https://github.com/Cedrus810/openmm_IBS_dev/issues/65) | Issue | open | [P2] Introduce protocol-version registry and migration tooling |  | 0 |
| [#66](https://github.com/Cedrus810/openmm_IBS_dev/issues/66) | Issue | open | [P2] Decompose IBSWindowManagerDualLambda.run_all_windows |  | 0 |
| [#67](https://github.com/Cedrus810/openmm_IBS_dev/issues/67) | Issue | closed | [P0] Harden analyze-only against stale or incomplete IBS stage data |  | 1 |
| [#68](https://github.com/Cedrus810/openmm_IBS_dev/issues/68) | Issue | closed | [P1] Truncate or segment sampling history after production catastrophe rollback |  | 0 |
| [#69](https://github.com/Cedrus810/openmm_IBS_dev/issues/69) | Issue | closed | [P1] Repair molecules across PBC before first minimization or pre-equilibration |  | 0 |
| [#70](https://github.com/Cedrus810/openmm_IBS_dev/issues/70) | Issue | closed | [P1] Preserve convergence, coverage, ESS, and rescue evidence in stage checkpoints |  | 0 |
| [#71](https://github.com/Cedrus810/openmm_IBS_dev/issues/71) | Issue | closed | [P1] Wire traditional-mode resume through run_full and run_leg |  | 0 |
| [#72](https://github.com/Cedrus810/openmm_IBS_dev/issues/72) | Issue | closed | [P2] Correct single-stage lambda endpoint diagnostics semantics |  | 0 |
| [#73](https://github.com/Cedrus810/openmm_IBS_dev/issues/73) | Issue | closed | [Audit][Fixed] Gate committed Boresch equilibrium reuse against current pose | audit, fixed | 0 |
| [#74](https://github.com/Cedrus810/openmm_IBS_dev/issues/74) | Issue | closed | [Validation] Confirm endpoint-sigma diagnosis and archive evidence for VDW rescue result | validation | 0 |
| [#75](https://github.com/Cedrus810/openmm_IBS_dev/issues/75) | Issue | open | [P3] Treat cross-process production resume boundaries as separate trajectory segments |  | 0 |
| [#76](https://github.com/Cedrus810/openmm_IBS_dev/issues/76) | Issue | closed | [P3] Pin pymbar-core version for stable uncertainty semantics |  | 1 |
| [#77](https://github.com/Cedrus810/openmm_IBS_dev/issues/77) | Issue | closed | [Audit][Fixed] Correct solvent-leg box edge and water-model resolution |  | 0 |
| [#78](https://github.com/Cedrus810/openmm_IBS_dev/issues/78) | Issue | open | [P1] Calibrate per-window uncertainty with split-half drift diagnostics |  | 1 |
| [#79](https://github.com/Cedrus810/openmm_IBS_dev/issues/79) | Issue | closed | [P1] Add the missing Boresch restraint-attachment free-energy leg |  | 0 |
| [#80](https://github.com/Cedrus810/openmm_IBS_dev/issues/80) | Issue | closed | [P1] Resolve residual charging discrepancy against reference protocol |  | 1 |
| [#81](https://github.com/Cedrus810/openmm_IBS_dev/issues/81) | Issue | closed | [P1] Remove Stage 2 preoptimization fingerprint fail-open environment bypass |  | 1 |
| [#82](https://github.com/Cedrus810/openmm_IBS_dev/issues/82) | Issue | closed | [P2] Make GROMACS include auto-discovery portable on Windows and POSIX |  | 1 |
| [#83](https://github.com/Cedrus810/openmm_IBS_dev/issues/83) | Issue | open | [P2] Feed real ligand bond topology into Boresch anchor selection |  | 0 |
| [#84](https://github.com/Cedrus810/openmm_IBS_dev/issues/84) | Issue | closed | [P2] Add membrane-aware barostat selection for membrane receptor systems |  | 1 |
| [#85](https://github.com/Cedrus810/openmm_IBS_dev/issues/85) | Issue | closed | [P1] Recompute convergence gates after split-half sigma inflation |  | 1 |
| [#86](https://github.com/Cedrus810/openmm_IBS_dev/issues/86) | Issue | closed | [P1] Gate Boresch last-frame updates on dihedral deviations |  | 1 |
| [#87](https://github.com/Cedrus810/openmm_IBS_dev/issues/87) | Issue | open | [P1] Design vdW/stage2 frame-selection and uncertainty protocol |  | 0 |
| [#88](https://github.com/Cedrus810/openmm_IBS_dev/pull/88) | PR | closed | Integrate experimental charge-transfer mainline support |  | 0 |

## 条目正文

### #1 — [P2] Migrate production ESS fixed-H probes to a per-state trajectory bank（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/1
- 作者：Cedrus810
- 创建：2026-07-16T09:50:06Z
- 更新：2026-07-29T06:38:52Z
- 关闭：2026-07-29T06:38:52Z
- 标签：enhancement, p2
- 评论数：1

**正文：**

> ## Problem
> The production ESS auto-repair call site `abfe_pipeline.py::_probe_vdw_window_fixed_overlap` still uses legacy per-edge path and bias-calibration probes. Interior states are sampled repeatedly and calibration retries repeat burn-in.
> 
> ## Acceptance criteria
> - Preserve result-cache-first behavior.
> - On a cache miss, use a per-state trajectory bank without mixing it with the warmup live-context bank.
> - Fingerprint call-site identity, starting coordinates, box, Hamiltonian, platform, and protocol versions.
> - Add invalidation and resume regression coverage.
> 
> ## Source
> `docs/TODO.md` — remaining performance/recovery hardening item.
> 

### #2 — [Validation] Revalidate IBS v12 frozen-calibration resume on CUDA（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/2
- 作者：Cedrus810
- 创建：2026-07-16T09:50:14Z
- 更新：2026-07-19T08:34:24Z
- 关闭：2026-07-19T08:34:24Z
- 标签：p2, validation
- 评论数：1

**正文：**

> ## Validation target
> Revalidate the IBS v12 `calibrated_pending_validation` state machine and exact log-sum-exp bias on CUDA.
> 
> ## Acceptance criteria
> - Run the previously problematic VDW window with occupancy near `p=[0.8485,0.1502,0.00126]`.
> - Confirm cumulative 50k -> 150k -> 300k budgets run only the remaining delta.
> - Resume frozen `f_k` without SGD learning, window split, or lambda insertion.
> - Confirm native main-window checkpoint restore and fingerprint rejection.
> - Confirm terminal `calibrated_validation_failed` after the final rung.
> - Check exact log-sum-exp on previously healthy windows.
> 
> ## Evidence required
> Job ID, command, code SHA256, IBS state, convergence JSON, checkpoint manifest, logs, and verdict.

### #3 — [Validation] Revalidate fixed-H lambda_shield NaN fix（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/3
- 作者：Cedrus810
- 创建：2026-07-16T09:50:21Z
- 更新：2026-07-19T08:34:29Z
- 关闭：2026-07-19T08:34:29Z
- 标签：p2, validation
- 评论数：1

**正文：**

> ## Validation target
> Revalidate fixed-H bias-calibration `lambda_shield` synchronization and the NaN fix on window `(5,9)`.
> 
> ## Acceptance criteria
> - Minimize under the real `lambda_shield` and state CV Hamiltonian.
> - First burn-in stage must not raise `Particle coordinate is NaN`.
> - Keep numerical NaN distinct from physical low-overlap edges `[5,6]` and `[6,7]`.
> - Record `nvidia-smi` context and memory behavior.
> 
> ## Evidence required
> Window log, probe manifest, per-edge overlap, fixed-H result JSON, GPU trace, and verdict.

### #4 — [Validation] Run current non_mutating_v1 OpenMM/PyMBAR regression suite（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/4
- 作者：Cedrus810
- 创建：2026-07-16T09:50:28Z
- 更新：2026-07-29T07:01:40Z
- 关闭：2026-07-29T07:01:40Z
- 标签：p2, validation
- 评论数：2

**正文：**

> ## Validation target
> Run the complete current regression suite after the `non_mutating_v1` transition.
> 
> ## Command
> ```bash
> python -m pytest -q
> ```
> 
> ## Required coverage
> - NM-01: legacy energy-cache rescue stop propagates; zero system construction; fixture hashes/mtimes remain unchanged.
> - NM-02: missing or mismatched `sampling_repair_policy` is rejected before any `f_k` injection; matching-policy resume succeeds.
> - `should_run_legacy_repair`: `non_mutating_v1` disables split/insert/recalibration paths and invalid policy values fail closed.
> - Main-window native checkpoint roundtrip, content-fingerprint rejection, cumulative validation-step accounting, and cross-process rung recovery.
> - Existing protocol regressions: finite energies, Boresch fail-closed, lambda-path invariants, APBS and 2D-geodesic checks.
> 
> ## Environment
> `openmm_dev` with OpenMM, PyMBAR, NumPy, pytest, and intended platform support.
> 
> ## Evidence required
> Environment export, complete pytest output, platform details, source SHA256, and failed-case logs.
> 
> ## Sources
> `docs/status/NON_MUTATING_V1_STATUS.md`; `docs/status/VALIDATION_MATRIX.md` (`VAL-TEST-001` and the still-relevant parts of `VAL-GPU-006`).
> 

### #5 — [Validation] Revalidate fixed-H lambda_shield NaN fix（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/5
- 作者：Cedrus810
- 创建：2026-07-16T09:51:02Z
- 更新：2026-07-16T09:51:44Z
- 关闭：2026-07-16T09:51:44Z
- 标签：duplicate, p2, validation
- 评论数：1

**正文：**

> ## Validation target
> Revalidate fixed-H bias-calibration `lambda_shield` synchronization and the NaN fix on window `(5,9)`.
> 
> ## Acceptance criteria
> - Minimize under the real `lambda_shield` and state CV Hamiltonian.
> - First burn-in stage must not raise `Particle coordinate is NaN`.
> - Keep numerical NaN distinct from physical low-overlap edges `[5,6]` and `[6,7]`.
> - Record `nvidia-smi` context and memory behavior.
> 
> ## Evidence required
> Window log, probe manifest, per-edge overlap, fixed-H result JSON, GPU trace, and verdict.

### #6 — [Validation] Run full OpenMM and PyMBAR regression suite（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/6
- 作者：Cedrus810
- 创建：2026-07-16T09:51:04Z
- 更新：2026-07-16T09:51:49Z
- 关闭：2026-07-16T09:51:49Z
- 标签：duplicate, p2, validation
- 评论数：1

**正文：**

> ## Validation target
> Run the complete suite for the 13 code-level P2 fixes and main-window native resume.
> 
> ## Command
> python -m pytest -q
> 
> ## Environment
> `openmm_dev` with OpenMM, PyMBAR, NumPy, pytest, and the intended GPU platform where applicable.
> 
> ## Evidence required
> Environment export, complete pytest output, platform details, code SHA256, and failed-case logs.

### #7 — [Audit][Fixed] IBS v10 separates path-overlap and bias-calibration ensembles（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/7
- 作者：Cedrus810
- 创建：2026-07-16T09:52:51Z
- 更新：2026-07-16T09:52:57Z
- 关闭：2026-07-16T09:52:57Z
- 标签：p2, audit, fixed
- 评论数：0

**正文：**

> ## Resolution
> Fixed in `IBS_BIAS_PROTOCOL_VERSION=10`.
> 
> The physical path-overlap probe and production bias-calibration probe now use separate Hamiltonians and result fields. Bias calibration includes the production WCA shield, excludes offline LRC, and enforces decorrelated-sample and uncertainty gates.
> 
> ## Status
> Code-level fix complete. Runtime evidence is tracked by the open validation issues.

### #8 — [Audit][Fixed] IBS v11 introduces resumable per-state fixed-H probe banks（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/8
- 作者：Cedrus810
- 创建：2026-07-16T09:52:59Z
- 更新：2026-07-16T09:53:03Z
- 关闭：2026-07-16T09:53:03Z
- 标签：p2, audit, fixed
- 评论数：0

**正文：**

> ## Resolution
> Fixed in `IBS_BIAS_PROTOCOL_VERSION=11` with `FIXED_H_PROBE_CACHE_PROTOCOL_VERSION=2`.
> 
> Fixed-H probes now share trajectories per state, use fingerprinted manifests and OpenMM native checkpoints, decorrelate restart segments independently, reject damaged volume records, and release stale evaluator contexts.
> 
> ## Status
> Code-level fix complete. GPU memory and NaN evidence remains tracked by validation issues.

### #9 — [Audit][Fixed] IBS v12 exact bias and frozen-calibration resume state machine（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/9
- 作者：Cedrus810
- 创建：2026-07-16T09:53:07Z
- 更新：2026-07-16T09:53:12Z
- 关闭：2026-07-16T09:53:12Z
- 标签：p2, audit, fixed
- 评论数：0

**正文：**

> ## Resolution
> Fixed in `IBS_BIAS_PROTOCOL_VERSION=12` and `MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION=1`.
> 
> The bias uses exact max-shift log-sum-exp. Calibrated frozen `f_k` is persisted separately, cumulative validation budgets are respected, resume never falls back to SGD, terminal failure is explicit, and native main-window checkpoints preserve the real dynamics context.
> 
> ## Status
> Code-level fix complete. CUDA validation remains tracked by #2.

### #10 — [Audit][Fixed] Exclude short fixed-H segments from MBAR（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/10
- 作者：Cedrus810
- 创建：2026-07-16T09:54:21Z
- 更新：2026-07-16T09:55:52Z
- 关闭：2026-07-16T09:55:52Z
- 标签：p2, audit, fixed
- 评论数：1

**正文：**

> ## Resolution
> Short restart segments below the sampling floor are now diagnostic-only and no longer enter decorrelated MBAR indices.
> 
> ## Status
> Code-level fix completed on 2026-07-16. Retained as a retrospective audit record; runtime suite coverage is tracked by #4.

### #11 — [Audit][Fixed] Release stale fixed-H evaluator dynamics contexts（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/11
- 作者：Cedrus810
- 创建：2026-07-16T09:54:23Z
- 更新：2026-07-16T09:56:25Z
- 关闭：2026-07-16T09:56:25Z
- 标签：p2, audit, fixed
- 评论数：1

**正文：**

> ## Resolution
> Per-state probe cleanup now releases evaluator references to old dynamics contexts, reducing peak GPU context retention.
> 
> ## Status
> Code-level fix completed on 2026-07-16. Retained as a retrospective audit record; runtime suite coverage is tracked by #4.

### #12 — [Audit][Fixed] Reject missing or invalid fixed-H volume records（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/12
- 作者：Cedrus810
- 创建：2026-07-16T09:54:25Z
- 更新：2026-07-16T09:56:31Z
- 关闭：2026-07-16T09:56:31Z
- 标签：p2, audit, fixed
- 评论数：1

**正文：**

> ## Resolution
> Missing, non-finite, non-positive, or length-mismatched `volume.npy` records are treated as corrupted state caches and resampled instead of being padded with zeros.
> 
> ## Status
> Code-level fix completed on 2026-07-16. Retained as a retrospective audit record; runtime suite coverage is tracked by #4.

### #13 — [Audit][Fixed] Clear production step overrides after lambda or window reindexing（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/13
- 作者：Cedrus810
- 创建：2026-07-16T09:54:27Z
- 更新：2026-07-16T09:56:36Z
- 关闭：2026-07-16T09:56:36Z
- 标签：p2, audit, fixed
- 评论数：1

**正文：**

> ## Resolution
> `pending_step_overrides` is cleared whenever lambda insertion, split, or canonicalization changes window numbering, preventing budgets from being assigned to the wrong window.
> 
> ## Status
> Code-level fix completed on 2026-07-16. Retained as a retrospective audit record; runtime suite coverage is tracked by #4.

### #14 — [Audit][Fixed] Make fixed-H probe and sampling-repair JSON writes atomic（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/14
- 作者：Cedrus810
- 创建：2026-07-16T09:54:29Z
- 更新：2026-07-16T09:56:41Z
- 关闭：2026-07-16T09:56:41Z
- 标签：p2, audit, fixed
- 评论数：1

**正文：**

> ## Resolution
> Fixed-H edge results and `sampling_repair_decisions.json` now use atomic JSON replacement to survive interrupted writes.
> 
> ## Status
> Code-level fix completed on 2026-07-16. Retained as a retrospective audit record; runtime suite coverage is tracked by #4.

### #15 — [Audit][Fixed] Fail closed on malformed analyze-only checkpoints and solver failures（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/15
- 作者：Cedrus810
- 创建：2026-07-16T09:54:49Z
- 更新：2026-07-16T09:56:47Z
- 关闭：2026-07-16T09:56:47Z
- 标签：p2, audit, fixed
- 评论数：1

**正文：**

> ## Resolution
> `--analyze-only` now validates stage identity, finite typed free energies and errors, continuous numeric window indices, and integrated-solver convergence instead of silently substituting zero or partial results.
> 
> ## Status
> Code-level fix completed on 2026-07-16. Retained as a retrospective audit record; runtime suite coverage is tracked by #4.

### #16 — [Audit][Fixed] Fail closed when Boresch analytical correction cannot be computed（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/16
- 作者：Cedrus810
- 创建：2026-07-16T09:54:51Z
- 更新：2026-07-16T09:56:53Z
- 关闭：2026-07-16T09:56:53Z
- 标签：p2, audit, fixed
- 评论数：1

**正文：**

> ## Resolution
> When Boresch parameters exist, analytical release calculation failure now aborts post-analysis instead of silently continuing with a zero correction.
> 
> ## Status
> Code-level fix completed on 2026-07-16. Retained as a retrospective audit record; runtime suite coverage is tracked by #4.

### #17 — [Audit][Fixed] Gate Shadow-Bridge and Shadow-IBS sublegs on convergence and overlap（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/17
- 作者：Cedrus810
- 创建：2026-07-16T09:54:53Z
- 更新：2026-07-16T09:57:24Z
- 关闭：2026-07-16T09:57:24Z
- 标签：p2, audit, fixed
- 评论数：1

**正文：**

> ## Resolution
> Both experimental decharging sublegs and their combined Stage 1 result now propagate and enforce `converged` and `min_overlap` diagnostics.
> 
> ## Status
> Code-level fix completed on 2026-07-16. Retained as a retrospective audit record; runtime suite coverage is tracked by #4.

### #18 — [Audit][Fixed] Unify Boresch force-constant sanitation ranges（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/18
- 作者：Cedrus810
- 创建：2026-07-16T09:54:56Z
- 更新：2026-07-16T09:57:29Z
- 关闭：2026-07-16T09:57:29Z
- 标签：p2, audit, fixed
- 评论数：1

**正文：**

> ## Resolution
> Final sanitation now matches estimator and analytical validation ranges: `kr=[100,2000]` and angular/dihedral constants `[10,1000]`, removing source-dependent secondary softening.
> 
> ## Status
> Code-level fix completed on 2026-07-16. Retained as a retrospective audit record; runtime suite coverage is tracked by #4.

### #19 — [Audit][Fixed] Share descending lambda-path invariant enforcement（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/19
- 作者：Cedrus810
- 创建：2026-07-16T09:55:21Z
- 更新：2026-07-16T09:57:35Z
- 关闭：2026-07-16T09:57:35Z
- 标签：p2, audit, fixed
- 评论数：1

**正文：**

> ## Resolution
> Stage 1 and single-lambda optimizers now share `finalize_descending_lambda_path` for finite values, endpoints, minimum spacing, deduplication, and fail-closed linear fallback.
> 
> ## Status
> Code-level fix completed on 2026-07-16. Retained as a retrospective audit record; runtime suite coverage is tracked by #4.

### #20 — [Audit][Fixed] Correct dormant MBAR units, compatibility calls, and anchor RMSF reference（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/20
- 作者：Cedrus810
- 创建：2026-07-16T09:55:23Z
- 更新：2026-07-16T09:57:52Z
- 关闭：2026-07-16T09:57:52Z
- 标签：p2, audit, fixed
- 评论数：1

**正文：**

> ## Resolution
> `ChunkedMBARAnalyzer` converts reduced kT values to kJ/mol, online monitoring uses compatibility wrappers, and automatic-anchor RMSF is measured from the mean structure.
> 
> ## Status
> Code-level fix completed on 2026-07-16. Retained as a retrospective audit record; runtime suite coverage is tracked by #4.

### #21 — [Audit][Fixed] Correct APBS grid volume and reject unsupported anisotropic boxes（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/21
- 作者：Cedrus810
- 创建：2026-07-16T09:55:25Z
- 更新：2026-07-16T09:57:59Z
- 关闭：2026-07-16T09:57:59Z
- 标签：p2, audit, fixed
- 评论数：1

**正文：**

> ## Resolution
> APBS integration uses `(nx-1)(ny-1)(nz-1)` voxels, cross-checks against `--box`, and fails closed when cubic-lattice constants are applied to materially anisotropic cells.
> 
> ## Status
> Code-level fix completed on 2026-07-16. Retained as a retrospective audit record; runtime suite coverage is tracked by #4.

### #22 — [Audit][Fixed] Integrate 2D geodesic long-move metric costs along the path（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/22
- 作者：Cedrus810
- 创建：2026-07-16T09:55:27Z
- 更新：2026-07-16T09:58:04Z
- 关闭：2026-07-16T09:58:04Z
- 标签：p2, audit, fixed
- 评论数：1

**正文：**

> ## Resolution
> Long 2D geodesic moves now use segmented bilinear/trapezoidal metric integration instead of endpoint-only averaging that could skip a high-variance ridge.
> 
> ## Status
> Code-level fix completed on 2026-07-16. Retained as a retrospective audit record; runtime suite coverage is tracked by #4.

### #23 — [Audit][Fixed] Cache code_sha256 for the lifetime of the Python process（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/23
- 作者：Cedrus810
- 创建：2026-07-16T11:01:15Z
- 更新：2026-07-16T11:01:21Z
- 关闭：2026-07-16T11:01:21Z
- 标签：bug, p2, audit, fixed
- 评论数：1

**正文：**

> ## Bug
> `abfe_pipeline.py::_code_hash()` reread four source files on every call. A running Python process continues executing code loaded at import time, so live edits could make one process emit different `code_sha256` values at different stages even though its executable bytecode had not changed.
> 
> ## Impact
> Stage/result reuse, PME cache, provenance, and analysis fingerprints could drift within one process, causing false cache invalidation or misleading cross-process comparisons. The run restarted at 16:30:34 was not affected because the four hashed files were not edited afterward.
> 
> ## Resolution
> Added process-level `_CODE_HASH_CACHE`. The first call reads and hashes the source files; all later calls return the cached value. The fingerprint now remains stable for the lifetime of the process.
> 
> ## Verification
> `py_compile` passed. Runtime regression evidence can be added if this path is exercised in a long-lived process with live disk edits.

### #24 — [Validation] Validate IBS v13 fixed-H Boresch Context on CUDA（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/24
- 作者：Cedrus810
- 创建：2026-07-16T13:38:41Z
- 更新：2026-07-19T08:34:34Z
- 关闭：2026-07-19T08:34:34Z
- 标签：p2, validation
- 评论数：1

**正文：**

> ## Validation target
> Validate the IBS v13 fixed-H probe Boresch Context fix on OpenMM + CUDA.
> 
> ## Current evidence
> The rerun correctly invalidated old protocol state and one window completed learning and frozen validation under the new Hamiltonian. The full stage has not completed yet.
> 
> ## Acceptance criteria
> - Confirm IBS v13 rejects old `frozen_f_k_pending` and restarts learning.
> - Confirm frozen validation converges with `lambda_boresch_scale=1.0`.
> - Compare fixed-H overlap and bias-calibration diagnostics with the production Hamiltonian.
> - Complete the stage and archive logs, state JSON, convergence JSON, and probe diagnostics.
> 
> ## Source
> `VALIDATION_MATRIX.md` — `VAL-GPU-003`.

### #25 — [Validation] Exercise batched production ESS auto-repair（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/25
- 作者：Cedrus810
- 创建：2026-07-16T13:38:43Z
- 更新：2026-07-19T08:34:38Z
- 关闭：2026-07-19T08:34:38Z
- 标签：p2, validation
- 评论数：1

**正文：**

> ## Validation target
> Exercise the production ESS auto-repair fixes for batched edge insertion and `already_good` starvation.
> 
> ## Acceptance criteria
> - Multiple failing windows insert all known failing edges within the same repair round.
> - `already_good` windows receive recalibration or resampling decisions in the same round after path remapping.
> - Repair-round accounting does not starve sampling repairs.
> - Archive complete repair-loop logs, `sampling_repair_decisions.json`, and `production_fixed_h_overlap.json`.
> 
> ## Source
> `VALIDATION_MATRIX.md` — `VAL-GPU-004`.

### #26 — [Validation] Verify reseed_resample production checkpoint continuation（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/26
- 作者：Cedrus810
- 创建：2026-07-16T13:38:46Z
- 更新：2026-07-19T08:34:43Z
- 关闭：2026-07-19T08:34:43Z
- 标签：p2, validation
- 评论数：1

**正文：**

> ## Validation target
> Validate resumable production sampling after `reseed_resample` with `PRODUCTION_WINDOW_CHECKPOINT_PROTOCOL_VERSION=1`.
> 
> ## Acceptance criteria
> - Preserve production arrays and a compatible native checkpoint.
> - Resume from cumulative step N and run only the remaining M steps.
> - Append new frames instead of overwriting prior samples.
> - Survive an intentional interruption after a periodic checkpoint.
> - Keep `recalibrate_f_k` as full invalidation rather than checkpoint continuation.
> 
> ## Source
> `VALIDATION_MATRIX.md` — `VAL-GPU-005`.

### #27 — [Validation] Complete preoptimization cache-key migration validation（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/27
- 作者：Cedrus810
- 创建：2026-07-16T13:38:48Z
- 更新：2026-08-12T16:44:27Z
- 关闭：2026-08-12T16:44:27Z
- 标签：p2, validation
- 评论数：1

**正文：**

> ## Status
> 
> Open; partial real-run evidence exists, but the current narrow preoptimization protocol-key migration is not fully validated.
> 
> ## Current implementation
> 
> - The narrow key is controlled by `PREOPT_HAMILTONIAN_PROTOCOL_VERSION` and Hamiltonian-relevant inputs.
> - Unrelated source-code changes should not invalidate preoptimization caches.
> - Compatibility logic can recognize the legacy wide-schema key and migrate it once.
> - Existing `output_lrc_fix` preoptimization JSON files still use the older wide schema, and the current migration event has not been archived.
> 
> ## Remaining acceptance criteria
> 
> - Demonstrate that an actual preoptimization Hamiltonian/protocol-version change invalidates the cache.
> - Demonstrate that unrelated code changes preserve reuse under the narrow key.
> - Run one legacy-wide to current-narrow migration, archive the before/after key diff and log, then confirm the next resume uses direct equality without another migration.
> - Validate `ABFE_DEBUG_FREEZE_CODE_HASH=1` only for the code-hash paths it still controls; do not require the current narrow preoptimization key to contain `preopt_code_sha256`.
> - Archive logs, hashes, protocol-key payloads, and source commit IDs.
> 
> ## Source
> 
> `docs/status/VALIDATION_MATRIX.md` VAL-TEST-002 and the current `_preopt_protocol_key` implementation.
> 

### #28 — [Validation] Validate online TMBAR f_k learning (IBS v19)（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/28
- 作者：Cedrus810
- 创建：2026-07-19T08:34:48Z
- 更新：2026-07-29T06:39:28Z
- 关闭：2026-07-29T06:39:28Z
- 标签：p2, validation
- 评论数：1

**正文：**

> ## Validation target
> Validate the implemented online TMBAR learning path (`IBS_BIAS_PROTOCOL_VERSION=19`) in the target pytest and CUDA environments. The code now persists `IBSSampler.tmbar_history` and reuses `GlobalMBARAnalyzer.solve_stage_integrated` during learning; this issue no longer tracks implementation work.
> 
> ## Acceptance criteria
> - Reproduce the former VDW window-0 case and confirm state0↔state5 no longer shows the pathological ~300+ kJ/mol gap; the result should be consistent in scale with the independent pilot-TI estimate (~16.7 kJ/mol), or fail with explicit TMBAR diagnostics.
> - Run `python -m pytest test_warmup_overlap_protocol.py test_non_mutating_policy.py test_audit_protocol_regressions.py -v` in `openmm_dev`, including the new equal-width-wells and insufficient-history tests.
> - Confirm candidate thresholds (ESS ratio 0.05, absolute ESS 5, decorrelated samples 5, uncertainty 5 kJ/mol) neither freeze candidates pathologically early nor exhaust `max_bias_updates` without candidates.
> - Resume a v18 state and verify protocol mismatch causes a clean restart from `f_k=0`; old TA fields must never be interpreted as empty TMBAR history.
> - Keep the window-3 pilot-density problem separate; it remains tracked by #29.
> 
> ## Evidence required
> Before/after window-0 state and warmup-failure JSON, `learning_tmbar` histories, per-batch TMBAR diagnostics, full pytest output, CUDA/platform metadata, and source hashes.
> 
> ## Source
> `docs/status/VALIDATION_MATRIX.md` — `VAL-GPU-007`; `docs/TODO.md` (v19 implementation moved out of active code TODOs).
> 

### #29 — [P2] Design fail-closed adaptive pilot densification for the VDW tail（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/29
- 作者：Cedrus810
- 创建：2026-07-19T08:34:50Z
- 更新：2026-07-29T06:40:00Z
- 关闭：2026-07-29T06:40:00Z
- 标签：enhancement, p2
- 评论数：1

**正文：**

> ## Problem
> Stage-2 pilot thermodynamic-length sampling uses a single short, uniformly seeded grid with roughly 10,000 steps per point. Rare or bursty tail events can cause the metric to be underestimated. Independent TI diagnostics show the real unresolved density problem is the final VDW tail (window 3, about 210.6 kJ/mol across five states), not window 0.
> 
> ## Design constraints
> - Never reuse the deprecated adjacent fixed-H overlap arbiter.
> - Only propose a new immutable ensemble; never rewrite frozen `f_k`, lambda nodes, or state membership in place.
> - Trigger a longer local pilot remeasurement when downstream learning and pilot-scale diagnostics disagree.
> - Retain a human/rescue-audit gate and record all evidence in a versioned manifest.
> 
> ## Acceptance criteria
> - Synthetic intermittent-metric test demonstrates the short pilot underestimation and longer remeasurement recovery.
> - Window-3 diagnostic proposes denser tail sampling without touching existing artifacts.
> - No production-time split/insert/recalibrate path becomes reachable under `non_mutating_v1`.
> 
> ## Source
> `docs/TODO.md` (2026-07-19 pilot safety densification item).
> 

### #30 — [Validation] Execute non_mutating_v1 fail-closed resume regressions（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/30
- 作者：Cedrus810
- 创建：2026-07-19T08:34:52Z
- 更新：2026-07-29T06:40:03Z
- 关闭：2026-07-29T06:40:03Z
- 标签：p2, validation
- 评论数：1

**正文：**

> ## Validation target
> Execute and archive the new `non_mutating_v1` fail-closed regression suite in the target dependency environment.
> 
> ## Acceptance criteria
> - NM-01 legacy energy cache raises `ExistingEnsembleRequiresRescueAudit`, makes zero system-build calls, and preserves file hashes/mtimes.
> - NM-02 missing/mismatched policy is rejected before `f_k` injection; matching policy resumes normally.
> - One outer run, zero legacy mutators/probes, hard-gate propagation, and invalid policy typo failure are all demonstrated.
> - Archive environment, full pytest output, source hashes, and failure logs.
> 
> ## Operational rule
> Do not use `output_lrc_fix` as the first live validation fixture and do not run plain `--resume` against it before this issue and the rescue audit are complete.
> 
> ## Source
> `docs/status/NON_MUTATING_V1_STATUS.md` — NM-01/NM-02 and Round 1 validation hold.
> 

### #31 — [Audit] Build a hash-bound read-only rescue manifest for output_lrc_fix（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/31
- 作者：Cedrus810
- 创建：2026-07-19T08:34:53Z
- 更新：2026-08-12T16:47:18Z
- 关闭：2026-08-12T16:47:18Z
- 标签：p2, audit
- 评论数：1

**正文：**

> ## Objective
> Classify the legacy `output_lrc_fix/vanishing` artifacts as `clean_reusable`, `tainted`, or `indeterminate` without modifying the originals.
> 
> ## Audit contract
> - Inventory every relevant NPY/JSON/checkpoint/manifest with absolute path, size, mtime, and SHA256.
> - Map evidence by exact lambda content/window range and file hash, not `window_idx` alone.
> - Treat absence of an `mbar_calibration` field as supporting evidence only, not proof.
> - Emit a versioned, hash-bound sidecar rescue manifest; never restamp legacy files in place.
> - Any changed/missing file, ambiguous index mapping, or incomplete provenance is `indeterminate` and requires a new ensemble.
> - Loader acceptance must require the explicit rescue-manifest contract and exact hash matches.
> 
> ## Acceptance criteria
> - Read-only inventory and classification report completed.
> - Reviewer/policy version and evidence links recorded for each decision.
> - Approved reuse path is tested only after the non-mutating regressions pass.
> 
> ## Source
> `docs/status/NON_MUTATING_V1_STATUS.md` — Rescue-audit contract and operational stop.
> 

### #32 — [Validation] Verify traditional REMD v3 LRC delivery and per-frame correction（Issue，open）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/32
- 作者：Cedrus810
- 创建：2026-07-21T11:56:46Z
- 更新：2026-08-12T16:47:24Z
- 关闭：未关闭
- 标签：无
- 评论数：1

**正文：**

> ## Status
> 
> Open; implementation is now protocol v3. Source-contract tests exist, but the required fixed-box runtime regression and evidence archive are still pending.
> 
> ## Validation target
> 
> Validate the traditional `single_lambda`/REMD LJ LRC v3 path with a small fixed-box regression.
> 
> ## Current implementation
> 
> - `TRADITIONAL_LJ_LRC_PROTOCOL_VERSION=3`.
> - Sigma-resolved, switching- and softcore-aware r^-6/r^-12 coefficients.
> - The producer passes `lj_tail_lrc_coeff_kj_mol` to each worker.
> - Each frame applies `coeff[k] / V(t)`.
> - The cache gate is bound to the v3 protocol.
> - Missing box data or unsupported volume behavior fails closed.
> 
> ## Remaining acceptance criteria
> 
> - Run a small fixed-box traditional REMD case in the target OpenMM/PyMBAR environment.
> - Confirm every task receives a finite coefficient array with length equal to the number of lambda states.
> - Exercise switching, softcore coefficients, and the PME-alpha fallback.
> - Archive pytest output, run logs, coefficient arrays, platform metadata, and source hashes.
> 
> ## Source
> 
> `docs/TODO.md` V-02; `docs/status/VALIDATION_MATRIX.md` VAL-TEST-003.
> 

### #33 — [Validation] Validate Stage 2 v21 blended-metric lambda schedule on CUDA（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/33
- 作者：Cedrus810
- 创建：2026-07-21T11:56:47Z
- 更新：2026-07-29T06:41:47Z
- 关闭：2026-07-29T06:41:47Z
- 标签：无
- 评论数：1

**正文：**

> ﻿## Status
> Open; updated on 2026-07-27 to match the current v21 lambda protocol.
> 
> The old issue title/body referred to the retired v18/v20 Fisher/square-anchor schedule. docs/TODO.md now states that current code uses THERMODYNAMIC_PATH_PROTOCOL_VERSION=21 and lended_metric_vanishing_lambdas.
> 
> ## Current Acceptance Conditions
> - path_protocol_version=21.
> - lambda_placement_method=\"fisher_metric_blended_with_geometric_floor_v21\".
> - 23 unique lambda values.
> - Last edge approximately .100049 -> 0.
> - 
> ealized_max_lambda_gap <= max_lambda_gap_bound.
> - Windows [(0,5),(4,8),(7,12),(11,16),(15,20),(19,23)].
> - 28 slots and sliding_overlap_states=0.
> - v18/v19/v20 caches fail closed under the v21 protocol.
> 
> ## Source
> docs/TODO.md - V-03 and design/LAMBDA_SCHEDULE_CONTRACT.md.
> 

### #34 — [Validation] Rebuild solvent leg with explicit 0.15 M NaCl cache（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/34
- 作者：Cedrus810
- 创建：2026-07-21T11:56:50Z
- 更新：2026-07-29T06:41:41Z
- 关闭：2026-07-29T06:41:41Z
- 标签：无
- 评论数：1

**正文：**

> ﻿## Status
> Open as V-04 validation; updated from docs/TODO.md on 2026-07-28.
> 
> The P0-11 solvent box bug itself is fixed and measured: corrected 4.257 nm solvent box changes Delta G_bind negligibly (-2.121 -> -2.111 kcal/mol) in the 3.000 / 4.257 / 6.057 nm scan. The production default remains SOLVENT_PADDING_NM = 1.5 with protocol v4 solvent cache.
> 
> ## Remaining Validation
> This issue remains open only for the operational V-04 cache rebuild validation:
> 
> - Next production --resume should reject/rebuild old protocol v3 / 3.000 nm solvent caches.
> - Confirm solvent_cache_manifest.json records ox_edge_nm, ligand_longest_axis_nm, padding_nm, 
> onbonded_cutoff_nm, and solvent_forcefield.
> - Confirm Na/Cl are nonzero for 0.15 M NaCl.
> 
> ## Source
> docs/TODO.md - P0-11 / V-04.
> 

### #35 — [P2] Excise unreachable legacy auto-repair logic from production code（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/35
- 作者：Cedrus810
- 创建：2026-07-21T11:56:51Z
- 更新：2026-07-27T10:09:15Z
- 关闭：2026-07-27T10:09:15Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after the 2026-07-27 TODO/source review.
> 
> docs/TODO.md now marks ATT-27 / E-03 as fixed. Executable dead code for disproven overlap auto-repair and retired lambda-refinement paths has been removed from production code and archived as non-executable Markdown.
> 
> ## Resolution
> Removed 1143 lines total:
> 
> - 872 lines after the unconditional 
> eturn in _run_stage_with_overlap_autorepair, archived at docs/archive/removed_overlap_autorepair_mutation_loop.md.
> - 83 lines from _refine_lambda_path_with_medium_probe, archived at docs/archive/removed_refine_lambda_path_with_medium_probe.md.
> - 188 lines from _retired_overlapping_vdw_schedule_design, archived at docs/archive/removed_retired_overlapping_vdw_schedule_design.md.
> 
> The enable_lambda_refine guard remains, but has moved near the 
> un_full_pipeline entry so accidental enablement fails early. Regression coverage is in 	est_att27_dead_code_removed.py.
> 
> ## Source
> docs/TODO.md - ATT-27 / E-03.
> 

### #36 — [P2] Support finite-size electrostatic correction for anisotropic boxes（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/36
- 作者：Cedrus810
- 创建：2026-07-21T11:56:53Z
- 更新：2026-07-29T06:40:29Z
- 关闭：2026-07-29T06:40:29Z
- 标签：无
- 评论数：1

**正文：**

> ## Objective`nImplement and validate an applicable finite-size electrostatic correction for membrane/anisotropic boxes.`n`n## Guardrails`n- Until a supported method is validated, `apbs_correction.py` must continue to fail closed when aspect ratio exceeds 1.10.`n- Document assumptions, units, box-shape applicability, numerical verification, and regression coverage.`n`n## Source`n`docs/TODO.md` E-04.

### #37 — [Research] Quantify Boresch non-harmonic release-correction error（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/37
- 作者：Cedrus810
- 创建：2026-07-21T11:56:54Z
- 更新：2026-07-27T10:09:22Z
- 关闭：2026-07-27T10:09:22Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after the 2026-07-27 TODO/source review.
> 
> docs/TODO.md now marks R-01 as complete. The Boresch k(1-cos(delta)) non-harmonic release-correction error has a closed-form estimate and is negligible for this system.
> 
> ## Resolution
> The exact angular partition function is the von Mises integral 2*pi*exp(-x)*I0(x) with x = beta*k; the harmonic approximation gives sqrt(2*pi/x). The ratio has the asymptotic expansion 1 + 1/(8x) + 9/(128x^2) + ..., giving each angular DOF error -kT*log(ratio).
> 
> For the production force constants, the total non-harmonic correction is roughly 0.02-0.03 kJ/mol, about 0.005 kcal/mol. Even at the force-constant lower clip bound, the bound is below 0.107 kcal/mol. This is far below the observed restraint/charging discrepancies, whose root cause was P0-10 rather than non-harmonicity.
> 
> ## Conclusion
> No hard gate is needed for this non-harmonic correction term. Existing harmonicity diagnostics remain useful for distribution-shape warnings, but the release-correction non-harmonicity itself is not a blocking error.
> 
> ## Source
> docs/TODO.md - R-01.
> 

### #38 — [Research] Recalibrate Shadow-Coulomb windows and overlap thresholds（Issue，open）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/38
- 作者：Cedrus810
- 创建：2026-07-21T11:56:56Z
- 更新：2026-08-12T16:47:27Z
- 关闭：未关闭
- 标签：无
- 评论数：1

**正文：**

> Use real bridge data to recalibrate Shadow-Coulomb window count and overlap thresholds. Source: `docs/TODO.md` R-02.

### #39 — [Research] Evaluate point-ESS lambda insertion heuristic off production path（Issue，open）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/39
- 作者：Cedrus810
- 创建：2026-07-21T11:56:57Z
- 更新：2026-08-12T16:47:29Z
- 关闭：未关闭
- 标签：无
- 评论数：1

**正文：**

> Evaluate `refine_stage_lambda_path_by_overlap` point-ESS insertion heuristic. It must remain outside the `non_mutating_v1` production path. Source: `docs/TODO.md` R-03.

### #40 — [Research] Assess ACE softcore, pilot-sampling, and PBC geometry sensitivities（Issue，open）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/40
- 作者：Cedrus810
- 创建：2026-07-21T11:56:59Z
- 更新：2026-08-12T16:47:31Z
- 关闭：未关闭
- 标签：无
- 评论数：1

**正文：**

> Assess the ACE softcore 1e-6 denominator floor, pilot metric minimum sample count, and PBC recentering impact on Boresch geometry. Source: `docs/TODO.md` R-04.

### #41 — [P2] Normalize runtime logging and assess multi-seed DEXP optimization（Issue，open）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/41
- 作者：Cedrus810
- 创建：2026-07-21T11:57:01Z
- 更新：2026-07-28T03:45:06Z
- 关闭：未关闭
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Open; updated from docs/TODO.md on 2026-07-28.
> 
> docs/TODO.md now broadens ATT-28 / R-05 from logging only to logging and shared utility cleanup.
> 
> ## Required Work
> - Unify structured logging entrypoints, levels, and file/console behavior.
> - Evaluate DEXP multi-seed optimization later, per DEXP deferral.
> - Replace module-level print = _log_print style behavior with a cleaner logging path.
> - Unify 5 NumpyEncoder implementations while preserving 
> p.integer -> int and 
> p.bool_ -> bool, avoiding local encoders that coerce integers to floats or reject numpy booleans.
> 
> ## Source
> docs/TODO.md - ATT-28 / R-05.
> 

### #42 — [P0 candidate] Fix orphan scan_boresch_1d_pes self signature（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/42
- 作者：Cedrus810
- 创建：2026-07-22T07:15:18Z
- 更新：2026-07-26T06:58:41Z
- 关闭：2026-07-26T06:58:41Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after local TODO review on 2026-07-26.
> 
> docs/TODO.md marks ATT-01 as verified not applicable: scan_boresch_1d_pes is an OrbScanner instance method, so self is not an orphan top-level parameter.
> 
> ## Source
> docs/TODO.md - ATT-01.
> 

### #43 — [P0 candidate] Resolve vanishing-subdomain default constant syntax（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/43
- 作者：Cedrus810
- 创建：2026-07-22T07:15:20Z
- 更新：2026-07-26T06:58:45Z
- 关闭：2026-07-26T06:58:45Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after local TODO review on 2026-07-26.
> 
> docs/TODO.md marks ATT-02 as verified not applicable: the current constant is VANISHING_TARGET_INTERVALS_PER_ENSEMBLE; the module imports and related paths pass tests.
> 
> ## Source
> docs/TODO.md - ATT-02.
> 

### #44 — [P0 candidate] Make single-GPU REMD replica Context allocation safe（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/44
- 作者：Cedrus810
- 创建：2026-07-22T07:15:21Z
- 更新：2026-07-26T06:58:48Z
- 关闭：2026-07-26T06:58:48Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after local TODO review on 2026-07-26.
> 
> docs/TODO.md marks ATT-03 as fixed: GPU/OpenCL defaults to one resident Context and falls back to CPU before creating extra GPU Contexts; OOM paths release created Contexts before CPU fallback.
> 
> ## Source
> docs/TODO.md - ATT-03.
> 

### #45 — [P0 candidate] Make parallel stage workers spawn-safe（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/45
- 作者：Cedrus810
- 创建：2026-07-22T07:15:22Z
- 更新：2026-07-27T03:49:56Z
- 关闭：2026-07-27T03:49:56Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after the 2026-07-27 TODO/source review.
> 
> docs/TODO.md now marks ATT-04 as fixed. The original review attributed the issue to _run_stage_worker_process() importing bfe_pipeline, but spawn must import that module anyway. The confirmed import-time CUDA side effect was the module-level GLOBAL_DEVICE, SUPPORTS_TF32 = get_optimal_device_settings() in bfe_core.py, which could initialize torch/CUDA during child-process import.
> 
> ## Resolution
> Device settings are now resolved lazily through memoized helpers (_resolve_device_settings, get_global_device, supports_tf32). The MACE/ML consumers now resolve defaults inside function bodies instead of at import time. Regression coverage is in 	est_import_time_side_effects.py, including a clean child-process import check that 	orch.cuda.is_initialized() remains false.
> 
> ## Source
> docs/TODO.md - ATT-04.
> 

### #46 — [P0 candidate] Fail closed on missing IBS bias/base artifacts（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/46
- 作者：Cedrus810
- 创建：2026-07-22T07:15:24Z
- 更新：2026-07-26T07:01:40Z
- 关闭：2026-07-26T07:01:40Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after local TODO review on 2026-07-26.
> 
> `docs/TODO.md` marks ATT-05/P0-5 as fixed: IBS energy/bias/base are now treated as an atomic three-file set with protocol, shape, frame count, SHA-256, manifest matching, finite checks, and fail-closed resume/skip/collection/final-analysis behavior.
> 
> ## Source
> `docs/TODO.md` - ATT-05 / P0-5.
> 

### #47 — [P0 candidate] Unify DEXP fitted and production Hamiltonian models（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/47
- 作者：Cedrus810
- 创建：2026-07-22T07:15:25Z
- 更新：2026-07-28T05:57:17Z
- 关闭：2026-07-28T05:57:17Z
- 标签：无
- 评论数：1

**正文：**

> ﻿## Status
> Closed after the 2026-07-28 DEXP production-contract update.
> 
> Old Orb global DEXP fitting is no longer part of the production protocol. The new production implementation uses the pair-specific LJ-matched analytic kernel frozen in docs/experiments/DEXP_KERNEL_PHYSICS_ISSUES.md: 
> 0_ij and epsilon_ij are generated analytically per pair from the original LJ sigma/epsilon, while only new-kernel configuration such as lpha_vdw / eta_vdw remains.
> 
> The production model no longer contains 
> 0_vdw, A_fit, B_fit, offset_c0, or offset_c1.
> 
> ## Resolution
> - Added the independent production entrypoint dexp_NEW.py.
> - dexp_NEW.DEXPProductionConfig rejects all legacy fitter fields.
> - bfe_core.DEXPSurrogatePotential.from_dict() now fails closed on old JSON fields instead of silently dropping parameters and running a different Hamiltonian.
> 
> ## Validation
> - Legacy-field rejection tests.
> - Two-atom U(r0) = -epsilon_ij test.
> - lambda_vdw = 0 endpoint test.
> - Real Atenolol 73,536-particle System construction smoke test.
> - Full CPU regression recorded as 420 passed.
> 
> ## Source
> docs/TODO.md - P0-6 / P0-7 / ATT-06 / ATT-07.
> 

### #48 — [P1 candidate] Remove or give semantics to solvent-leg topology argument（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/48
- 作者：Cedrus810
- 创建：2026-07-22T07:15:27Z
- 更新：2026-07-26T07:01:45Z
- 关闭：2026-07-26T07:01:45Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after local TODO review on 2026-07-26.
> 
> `docs/TODO.md` marks ATT-08 as verified not applicable: `topology` is used to hard-gate consistency between complex topology ligand atom count and `ligand_indices`.
> 
> ## Source
> `docs/TODO.md` - ATT-08.
> 

### #49 — [P1 candidate] Centralize thermodynamic-cycle sign convention（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/49
- 作者：Cedrus810
- 创建：2026-07-22T07:15:29Z
- 更新：2026-07-27T03:49:59Z
- 关闭：2026-07-27T03:49:59Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after the 2026-07-27 TODO/source review.
> 
> docs/TODO.md now marks ATT-09 as fixed. The original statement that all thermodynamic-cycle formulas were correct was not accurate: 
> un_traditional_mode() and 
> un_full_abfe_loop() did not apply APBS corrections.
> 
> ## Resolution
> The binding free-energy combination is centralized in bfe_core.combine_binding_free_energy(). The four callers now pass explicit oresch_already_included_in_complex and pbs_correction_kJ_mol values. Regression coverage is in 	est_thermodynamic_cycle.py, including analytic toy-cycle checks and AST contracts for the four call sites.
> 
> ## Source
> docs/TODO.md - ATT-09.
> 

### #50 — [P1 candidate] Hard-gate intermittent IBS base-energy failures（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/50
- 作者：Cedrus810
- 创建：2026-07-22T07:15:30Z
- 更新：2026-07-26T07:01:50Z
- 关闭：2026-07-26T07:01:50Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after local TODO review on 2026-07-26.
> 
> `docs/TODO.md` marks ATT-10 as fixed: intermittent IBS base-energy failures now hard-gate on consecutive failures, total failures, and failure rate, with diagnostics recording attempts, failures, reasons, and limits.
> 
> ## Source
> `docs/TODO.md` - ATT-10.
> 

### #51 — [P1 candidate] Replace single geometric bond-distance cutoff（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/51
- 作者：Cedrus810
- 创建：2026-07-22T07:15:32Z
- 更新：2026-07-27T09:06:42Z
- 关闭：2026-07-27T09:06:42Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after the 2026-07-27 TODO/source review.
> 
> docs/TODO.md now marks ATT-11 as fixed. The old single .22 nm geometric bond-distance rule has been replaced with topology-backed bond adjacency for Boresch anchor neighbor discovery.
> 
> ## Resolution
> Implemented _build_bond_adjacency() and extended _find_bonded_neighbors() to use explicit adjacency. The geometric cutoff is now only an explicit fallback (llow_geometric_bond_fallback) and records ond_source / ond_dist_nm_if_geometric in diagnostics. Regression coverage is in 	est_boresch_anchor_and_pbc_fixes.py.
> 
> ## Source
> docs/TODO.md - ATT-11.
> 

### #52 — [P1 candidate] Reconcile fixed vanishing windows with adaptive schedule API（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/52
- 作者：Cedrus810
- 创建：2026-07-22T07:15:33Z
- 更新：2026-07-27T03:50:01Z
- 关闭：2026-07-26T07:01:55Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed, with the 2026-07-27 TODO caveat recorded.
> 
> docs/TODO.md still treats ATT-12 as closed, but corrects the old v20 description: the square-anchor/Fisher-bridge language is retired. Current code uses THERMODYNAMIC_PATH_PROTOCOL_VERSION=21 and lended_metric_vanishing_lambdas, with six single-boundary shared windows.
> 
> ## Current Contract
> See design/LAMBDA_SCHEDULE_CONTRACT.md for the v21 lambda schedule contract.
> 
> ## Source
> docs/TODO.md - ATT-12 / V-03.
> 

### #53 — [P1 candidate] Bound and observe TMBAR history checkpoint footprint（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/53
- 作者：Cedrus810
- 创建：2026-07-22T07:15:35Z
- 更新：2026-07-26T07:01:58Z
- 关闭：2026-07-26T07:01:58Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after local TODO review on 2026-07-26.
> 
> `docs/TODO.md` marks ATT-13 as fixed: TMBAR history is bounded to the latest 200 minibatches, records discarded counts, and resume validates/loads only the bounded suffix.
> 
> ## Source
> `docs/TODO.md` - ATT-13.
> 

### #54 — [P1 candidate] Preserve configured DEXP cutoff and switch in IBS（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/54
- 作者：Cedrus810
- 创建：2026-07-22T07:15:37Z
- 更新：2026-07-28T05:57:21Z
- 关闭：2026-07-28T05:57:21Z
- 标签：无
- 评论数：0

**正文：**

> ﻿
> 

### #55 — [P1 candidate] Select alchemical counterions with PBC-aware bulk criteria（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/55
- 作者：Cedrus810
- 创建：2026-07-22T07:15:38Z
- 更新：2026-07-26T07:02:03Z
- 关闭：2026-07-26T07:02:03Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after local TODO review on 2026-07-26.
> 
> `docs/TODO.md` marks ATT-15/P1-9 as fixed: alchemical counterion selection is now PBC-aware, uses bulk-water criteria, enforces integer ligand charge tolerance, and supports multiple monovalent counterions for multivalent ligands.
> 
> ## Source
> `docs/TODO.md` - ATT-15 / P1-9.
> 

### #56 — [P1 candidate] Propagate covariance into global TMBAR uncertainty（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/56
- 作者：Cedrus810
- 创建：2026-07-22T07:15:40Z
- 更新：2026-07-26T07:02:07Z
- 关闭：2026-07-26T07:02:07Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after local TODO review on 2026-07-26.
> 
> `docs/TODO.md` marks ATT-16/P1-10 as fixed: global TMBAR final uncertainty now uses endpoint-difference covariance from each independent window segment and records the error method in results.
> 
> ## Source
> `docs/TODO.md` - ATT-16 / P1-10.
> 

### #57 — [P1 candidate] Bind pre-equilibration cache to coordinates, box, and budget（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/57
- 作者：Cedrus810
- 创建：2026-07-22T07:15:41Z
- 更新：2026-07-26T07:02:10Z
- 关闭：2026-07-26T07:02:10Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after local TODO review on 2026-07-26.
> 
> `docs/TODO.md` marks ATT-17/P1-11 as fixed: pre-equilibration fingerprints now bind starting coordinates, box, and requested step count across production call sites.
> 
> ## Source
> `docs/TODO.md` - ATT-17 / P1-11.
> 

### #58 — [P1 candidate] Correct solvent-box sizing from requested padding（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/58
- 作者：Cedrus810
- 创建：2026-07-22T07:15:43Z
- 更新：2026-07-26T07:02:13Z
- 关闭：2026-07-26T07:02:13Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after local TODO review on 2026-07-26.
> 
> `docs/TODO.md` marks ATT-18/P1-12 as fixed: solvent-box construction now uses OpenMM `padding=1.5 nm` semantics, while `box_size_nm` remains only as a warning-bearing compatibility parameter.
> 
> ## Source
> `docs/TODO.md` - ATT-18 / P1-12.
> 

### #59 — [P2] Expand unit-test matrix for core physical contracts（Issue，open）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/59
- 作者：Cedrus810
- 创建：2026-07-22T07:15:45Z
- 更新：2026-07-26T07:02:15Z
- 关闭：未关闭
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Open.
> 
> `docs/TODO.md` still tracks ATT-19: core physical unit-test coverage needs a minimum matrix for softcore potential endpoints, DEXP LJ matching, IBS log-sum-exp stability, window stitching, PBC, ion counting, resume, and parallel worker behavior.
> 
> ## Source
> `docs/TODO.md` - ATT-19.
> 

### #60 — [P2] Add public ABFE benchmark integration suite（Issue，open）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/60
- 作者：Cedrus810
- 创建：2026-07-22T07:15:47Z
- 更新：2026-07-26T07:02:16Z
- 关闭：未关闭
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Open.
> 
> `docs/TODO.md` still tracks ATT-20: the project needs reproducible public ABFE benchmark integration coverage for neutral and charged ligands, two-leg cycle closure, and experimental comparison before release-grade accuracy claims.
> 
> ## Source
> `docs/TODO.md` - ATT-20.
> 

### #61 — [P2] Publish minimum user, input, API, and thermodynamic-cycle docs（Issue，open）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/61
- 作者：Cedrus810
- 创建：2026-07-22T07:15:48Z
- 更新：2026-08-12T16:42:34Z
- 关闭：未关闭
- 标签：无
- 评论数：1

**正文：**

> ﻿## Status
> Open.
> 
> `docs/TODO.md` still tracks ATT-21: minimum user-facing docs remain incomplete, including README, installation/deployment, API, input formats, user manual, and independent thermodynamic-cycle derivation.
> 
> ## Source
> `docs/TODO.md` - ATT-21.
> 

### #62 — [P2] Add layered CI, linting, typing, and formatting（Issue，open）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/62
- 作者：Cedrus810
- 创建：2026-07-22T07:15:50Z
- 更新：2026-07-28T03:44:58Z
- 关闭：未关闭
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Open.
> 
> docs/TODO.md still tracks ATT-22 as broader CI/CD/static-check work. The offline pytest entrypoint has been fixed and has recorded 367 passed, but real CI/lint/typing/format gates remain missing.
> 
> ## Current Progress
> - 
> un_offline_tests.sh activates openmm_dev safely and verifies the selected Python.
> - CPU/OpenMM/PyMBAR suite has been recorded as 367 passed.
> 
> ## Remaining Work
> - Add GitHub Actions or equivalent CI.
> - Add ruff/flake8, mypy, black/isort gates.
> - Keep GPU-required jobs out of ordinary CI.
> - Clean the remaining 3 bare except: sites in bfe_core.py, bfe_pipeline.py, and bfe_preoptimizer.py that can catch KeyboardInterrupt / SystemExit.
> 
> ## Source
> docs/TODO.md - ATT-22.
> 

### #63 — [P2] Add runtime recovery and resource-protection controls（Issue，open）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/63
- 作者：Cedrus810
- 创建：2026-07-22T07:15:51Z
- 更新：2026-07-28T03:45:00Z
- 关闭：未关闭
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Open; updated from docs/TODO.md on 2026-07-28.
> 
> docs/TODO.md still tracks ATT-23 as runtime recovery and resource-protection work.
> 
> ## Current Known Gap
> Checkpoint/trajectory validity checks are too shallow:
> 
> - _is_checkpoint_valid() currently checks only file size and seekability.
> - _is_traj_valid() checks rough size, CORD, and the first record length.
> 
> Neither proves that the OpenMM checkpoint can actually be loaded or that the DCD frames are complete. Recovery should rely on real loadCheckpoint / DCD parsing before deciding to append old trajectories.
> 
> ## Source
> docs/TODO.md - ATT-23.
> 

### #64 — [P2] Strengthen CLI and physical input validation（Issue，open）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/64
- 作者：Cedrus810
- 创建：2026-07-22T07:15:53Z
- 更新：2026-07-28T03:45:03Z
- 关闭：未关闭
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Open / partially fixed.
> 
> docs/TODO.md still tracks ATT-24 as broader input validation work. Explicit config/torsion silent downgrade is fixed; DEXP input contracts remain deferred.
> 
> ## Remaining Work
> - Broader ligand residue / TOP include / atom-count / Boresch constructability / box-size preflight diagnostics.
> - Deferred DEXP input-file contract.
> - Low-priority cleanup: remove calc_boresch_from_last_frame() guessed transpose for (3,3) coordinates.
> - Low-priority cleanup: replace ACESoftcorePotential.optimize_alpha() assertions with explicit ValueError even though the method currently has no repository call sites.
> 
> ## Source
> docs/TODO.md - ATT-24.
> 

### #65 — [P2] Introduce protocol-version registry and migration tooling（Issue，open）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/65
- 作者：Cedrus810
- 创建：2026-07-22T07:15:55Z
- 更新：2026-07-26T07:02:24Z
- 关闭：未关闭
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Open.
> 
> `docs/TODO.md` still tracks ATT-25: protocol versions need a unified registry, cache fingerprint composition rules, migration notes, and compatibility tests to prevent missed invalidation.
> 
> ## Source
> `docs/TODO.md` - ATT-25.
> 

### #66 — [P2] Decompose IBSWindowManagerDualLambda.run_all_windows（Issue，open）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/66
- 作者：Cedrus810
- 创建：2026-07-22T07:15:56Z
- 更新：2026-07-28T03:45:04Z
- 关闭：未关闭
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Open; updated from docs/TODO.md on 2026-07-28.
> 
> IBSWindowManagerDualLambda.run_all_windows remains too long and behaviorally dense. The current measured scale is roughly 3055 lines, and a major split is still deferred until end-to-end regression is steadier.
> 
> ## Additional Cleanup Scope
> When this work resumes, also handle three zero-call legacy points now listed in TODO:
> 
> - scan_boresch_1d_pes() contains a second Angstrom-to-nm conversion and unreachable angle branch.
> - ggregate_all_energies() guesses matrix orientation from lengths.
> - A duplicate import of generate_overlapping_windows remains.
> 
> ## Source
> docs/TODO.md - ATT-26.
> 

### #67 — [P0] Harden analyze-only against stale or incomplete IBS stage data（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/67
- 作者：Cedrus810
- 创建：2026-07-27T03:50:11Z
- 更新：2026-08-12T16:47:22Z
- 关闭：2026-08-12T16:47:22Z
- 标签：无
- 评论数：1

**正文：**

> ﻿## Status
> Open; created from docs/TODO.md on 2026-07-27.
> 
> ## Problem
> 
> unabfe.run_post_analysis() can accept existing stage1_decharging.json / stage2_vanishing.json after checking only stage, 	otal_delta_G, and 	otal_error. It does not validate protocol_key, lambda_path_fingerprint, converged, coverage_diagnostics, or window manifests, so old or truncated stage checkpoints can still be treated as authoritative.
> 
> The raw-window fallback also does not inherit the current stage completeness and ESS/f_k contract: it can miss a missing terminal window, does not read frozen _k from checkpoints/ibs_state_*.json, and therefore cannot compute the current ESS gate correctly.
> 
> ## Required Work
> Reuse the main pipeline loader/contract for analyze-only: three-file manifests, expected window coverage, checkpoint/f_k loading, stage checkpoint protocol validation, and coverage diagnostics.
> 
> ## Source
> docs/TODO.md - P0-9.
> 

### #68 — [P1] Truncate or segment sampling history after production catastrophe rollback（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/68
- 作者：Cedrus810
- 创建：2026-07-27T03:50:12Z
- 更新：2026-07-27T09:06:46Z
- 关闭：2026-07-27T09:06:46Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after the 2026-07-27 TODO/source review.
> 
> docs/TODO.md now marks P1-13 as fixed. Production catastrophe rollback now tracks and restores sampling-history lengths together with the coordinate backup.
> 
> ## Resolution
> Added ibs_engine._production_history_lengths() and _truncate_production_history(). Each production position backup now has a paired production_history_backup_len, and both catastrophe rollback paths truncate energy_history, ias_history, and ase_energy_history consistently. Regression coverage checks equal lengths, truncation behavior, no-op behavior, and backup/length assignment pairing.
> 
> ## Note
> The 2026-07-27 V-06 diagnosis found this path did not trigger in the archived output_lrc_fix data; this fix is still correct robustness work, but it does not explain the observed base-energy jumps.
> 
> ## Source
> docs/TODO.md - P1-13 and V-06.
> 

### #69 — [P1] Repair molecules across PBC before first minimization or pre-equilibration（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/69
- 作者：Cedrus810
- 创建：2026-07-27T03:50:14Z
- 更新：2026-07-27T09:06:50Z
- 关闭：2026-07-27T09:06:50Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after the 2026-07-27 TODO/source review.
> 
> docs/TODO.md now marks P1-14 as fixed. Whole-molecule PBC repair has moved before the first Context/minimization/pre-equilibration.
> 
> ## Resolution
> ABFEPipeline.repair_pbc_molecule_integrity() is called at the start of pre_equilibrate(), before minimization/NPT and before _pre_equilibration_fingerprint is computed. Failure now fails closed instead of falling back to ligand-only centroid wrapping.
> 
> ## Source
> docs/TODO.md - P1-14.
> 

### #70 — [P1] Preserve convergence, coverage, ESS, and rescue evidence in stage checkpoints（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/70
- 作者：Cedrus810
- 创建：2026-07-27T03:50:15Z
- 更新：2026-07-27T09:06:53Z
- 关闭：2026-07-27T09:06:53Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after the 2026-07-27 TODO/source review.
> 
> docs/TODO.md now marks P1-15 as fixed. Stage checkpoint payloads now preserve convergence/coverage evidence and are re-validated before reuse.
> 
> ## Resolution
> Added _populate_stage_diagnostics(), made the immutable rescue merge path populate diagnostics too, added JSON-safe payload conversion for numpy-containing diagnostics, persisted converged and coverage_diagnostics, and re-run sanity checks through _assert_reusable_stage_cache_sane() on resume hits.
> 
> ## Source
> docs/TODO.md - P1-15.
> 

### #71 — [P1] Wire traditional-mode resume through run_full and run_leg（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/71
- 作者：Cedrus810
- 创建：2026-07-27T03:50:16Z
- 更新：2026-07-27T09:06:57Z
- 关闭：2026-07-27T09:06:57Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after the 2026-07-27 TODO/source review.
> 
> docs/TODO.md now marks P1-16 as fixed. Traditional mode now propagates resume from CLI/config down through the full pipeline.
> 
> ## Resolution
> TraditionalABFEPipeline.run_full(..., resume=False) passes the same explicit value to both 
> un_leg() calls, and 
> unabfe.run_traditional_mode() passes config.resume and not config.reset.
> 
> ## Source
> docs/TODO.md - P1-16.
> 

### #72 — [P2] Correct single-stage lambda endpoint diagnostics semantics（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/72
- 作者：Cedrus810
- 创建：2026-07-27T03:50:18Z
- 更新：2026-07-27T09:07:00Z
- 关闭：2026-07-27T09:07:00Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after the 2026-07-27 TODO/source review.
> 
> docs/TODO.md now marks P2-15 as fixed. Single-stage endpoint diagnostics now use stage-specific expected endpoints instead of full two-stage path criteria.
> 
> ## Resolution
> lambda_endpoint_diagnostics() keeps the full-path default and now accepts explicit expected_start / expected_end. Stage 1 checks (1,1)->(0,1) and Stage 2 checks (0,1)->(0,0), so legal half-paths no longer report ok=false.
> 
> ## Source
> docs/TODO.md - P2-15.
> 

### #73 — [Audit][Fixed] Gate committed Boresch equilibrium reuse against current pose（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/73
- 作者：Cedrus810
- 创建：2026-07-27T09:07:06Z
- 更新：2026-07-27T09:08:08Z
- 关闭：2026-07-27T09:08:08Z
- 标签：audit, fixed
- 评论数：0

**正文：**

> ﻿## Status
> Closed on creation from docs/TODO.md after the 2026-07-27 fix.
> 
> ## Problem
> The committed Boresch equilibrium file (checkpoints/boresch_equilibrium_committed.json) was reused indefinitely if present, without identity or geometry validation. In the current audit, an old 2026-07-10 committed file contained swapped/misassigned angular/dihedral equilibrium values and was reused after the 2026-07-26 complex re-equilibration.
> 
> ## Impact
> The old complex-leg samples were generated under the wrong restraint equilibrium and must be discarded for the next fresh rerun. Solvent-leg data is unaffected because it has no Boresch restraint.
> 
> ## Resolution
> Added oresch_committed_deviation_sigma() and ABFEPipeline._assert_committed_boresch_still_matches_pose(). Reuse now checks anchor identity and per-DOF sigma deviation, wraps dihedrals through _wrap_to_pi, fails closed above BORESCH_COMMITTED_MAX_DEVIATION_SIGMA=4.0, and writes a schema-versioned committed file with receptor/ligand indices, force constants, temperature, and derivation timestamp.
> 
> Regression coverage is in 	est_boresch_committed_gate.py.
> 
> ## Source
> docs/TODO.md - P0-10.
> 

### #74 — [Validation] Confirm endpoint-sigma diagnosis and archive evidence for VDW rescue result（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/74
- 作者：Cedrus810
- 创建：2026-07-27T09:07:08Z
- 更新：2026-07-27T10:09:18Z
- 关闭：2026-07-27T10:09:18Z
- 标签：validation
- 评论数：0

**正文：**

> ﻿## Status
> Closed after the 2026-07-27 TODO/source review.
> 
> docs/TODO.md now marks V-06 as complete. Endpoint-sigma diagnosis evidence has been moved from /tmp/sigma_diag/ into docs/status/evidence_2026-07-27/.
> 
> ## Archived Evidence
> - endpoint_sigma_diagnosis.json: reproduces 145.90847168207642 / 1.384443322336141, includes segment diagnostics, shows primary_z 鈮?0.89 < 2, and records history scan results.
> - charging_linear_response.json: shows trapezoid TI and MBAR agree within about 1 kJ/mol and records pocket/water charging-energy contrast.
> - pose_drift.json: records 0.60 脜 unconstrained pre-equilibration drift vs 3.42 脜 Boresch rebalance drift, supporting P0-10.
> 
> ## Caveat
> These evidence files describe the now-discarded complex-leg sampling. Estimator/method conclusions remain useful; physical free-energy values from that old run should not be cited as final.
> 
> ## Source
> docs/TODO.md - V-06.
> 

### #75 — [P3] Treat cross-process production resume boundaries as separate trajectory segments（Issue，open）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/75
- 作者：Cedrus810
- 创建：2026-07-27T09:07:09Z
- 更新：2026-07-27T09:07:09Z
- 关闭：未关闭
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Open; created from docs/TODO.md V-06 on 2026-07-27.
> 
> ## Problem
> The V-06 diagnosis found base-energy jumps at cross-process resume/window-rebuild boundaries. In the audited data these jumps cancel out of (u_kn[k] - bias)/kT, so decorrelation and endpoint sigma were not affected. That is a fortunate property of this case, not a clean trajectory contract.
> 
> ## Required Work
> Record production segment boundaries across resume/rebuild events and avoid treating discontinuous segments as one continuous trajectory for autocorrelation estimates. Re-run the diagnostic to confirm ase_jump_frame_indices align with session boundaries.
> 
> ## Source
> docs/TODO.md - V-06 residual item.
> 

### #76 — [P3] Pin pymbar-core version for stable uncertainty semantics（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/76
- 作者：Cedrus810
- 创建：2026-07-27T09:07:11Z
- 更新：2026-08-13T04:41:50Z
- 关闭：2026-08-13T04:41:50Z
- 标签：无
- 评论数：1

**正文：**

> ﻿## Status
> Open; created from docs/TODO.md V-06 on 2026-07-27.
> 
> ## Problem
> environment.yml does not pin pymbar-core. The local openmm_dev environment resolved to 4.0.3, while another environment dump records 4.2.0. The current audit says None -> svd-ew semantics are the same for the checked versions, but final uncertainty semantics should not depend on an unconstrained dependency.
> 
> ## Required Work
> Pin or bound pymbar-core in the environment, document the intended uncertainty method, and keep regression evidence tied to that version range.
> 
> ## Source
> docs/TODO.md - V-06 residual item.
> 

### #77 — [Audit][Fixed] Correct solvent-leg box edge and water-model resolution（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/77
- 作者：Cedrus810
- 创建：2026-07-28T03:45:08Z
- 更新：2026-07-28T03:45:38Z
- 关闭：2026-07-28T03:45:37Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed on creation from docs/TODO.md after the 2026-07-28 fix.
> 
> ## Problem
> P0-11 found that the solvent-leg box was 3.000 nm, exactly 2 * SOLVENT_PADDING_NM, meaning ligand size was not included. For the current ligand longest axis of 1.257 nm, a 1.5 nm per-side padding box should be about 4.257 nm. The 3.000 nm box left only about 0.87 nm from ligand surface to box face along the long axis.
> 
> A same-pass audit also found a live water-model split: one path could use mber14/tip3pfb.xml while the complex leg used ordinary TIP3P from the GROMACS top include.
> 
> ## Resolution
> - Added bfe_core.solvent_box_edge_nm() with gmx editconf -d semantics.
> - Both solvent builders now pass explicit oxSize=.
> - Built boxes are validated against requested edge length and checked against 2 * cutoff.
> - SOLVENT_CACHE_PROTOCOL_VERSION bumped 3 -> 4.
> - Manifest records ox_edge_nm, ligand_longest_axis_nm, padding_nm, and 
> onbonded_cutoff_nm.
> - Added bfe_core.resolve_water_model_xml() to infer the water model from complex .top includes and fail closed if unknown.
> 
> ## Result
> The 2026-07-28 scan over 3.000 / 4.257 / 6.057 nm showed the corrected 4.257 nm box changes Delta G_bind negligibly (-2.121 -> -2.111 kcal/mol), but the cache/protocol fix is still required.
> 
> ## Source
> docs/TODO.md - P0-11.
> 

### #78 — [P1] Calibrate per-window uncertainty with split-half drift diagnostics（Issue，open）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/78
- 作者：Cedrus810
- 创建：2026-07-28T03:45:12Z
- 更新：2026-07-29T06:26:07Z
- 关闭：未关闭
- 标签：无
- 评论数：1

**正文：**

> ﻿## Status
> Open; updated from docs/TODO.md on 2026-07-28.
> 
> P1-19 still tracks per-window / per-leg uncertainty underestimation. The split-half diagnostics remain the proposed way to expose slow drift invisible to ESS/overlap/occupancy gates.
> 
> ## Existing Evidence
> Solvent box scans showed 5 of 18 windows with split-half z > 2, including window 4 cases where ESS/overlap/occupancy looked excellent but actual half-window drift exceeded the asymptotic sigma by 2-4x.
> 
> ## New Supporting Evidence
> The Boresch attachment leg reproduced the same theme:
> 
> - 12-state run BAR = 5.3784 kJ/mol with single-run sigma about 0.083.
> - 4-state run BAR = 5.8238 kJ/mol with single-run sigma about 0.100.
> - Difference is 0.4454 kJ/mol, about 4.4x the single-run sigma.
> 
> The adopted attachment uncertainty is therefore the half-difference between independent measurements, not either single-run asymptotic sigma.
> 
> ## Proposed Fix
> For stage/window reporting, use a drift floor such as:
> 
> sigma_win = max(sigma_MBAR, |split_half_drift| / 2)
> 
> and propagate the larger segment uncertainty through stage/final results.
> 
> ## Source
> docs/TODO.md - P1-19 and P1-17 measurement notes.
> 

### #79 — [P1] Add the missing Boresch restraint-attachment free-energy leg（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/79
- 作者：Cedrus810
- 创建：2026-07-28T03:45:14Z
- 更新：2026-07-28T06:09:39Z
- 关闭：2026-07-28T06:09:39Z
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Closed after the 2026-07-28 Boresch attachment implementation and GPU measurements.
> 
> docs/TODO.md now marks P1-17 complete. The missing Boresch restraint-attachment leg has been added and measured.
> 
> ## Implementation
> - Added ibs_engine.run_boresch_attachment_leg() for sequential independent-window lambda_boresch sampling from 0 -> 1.
> - Added ibs_engine.add_scalable_boresch_restraint() for a scan-capable Boresch force (ixed_lam=None).
> - Main estimator is adjacent BAR; TI is the consistency gate; decorrelated MBAR is diagnostic only.
> - Added strict Delta G_attach >= 0 fail-closed checks.
> - Added force-group occupancy checks, U(lambda) linearity checks, and pre-run anchor geometry gates.
> - bfe_pipeline.compute_final_results now includes dg_attach + dg_decharge + dg_vdw and reports inal["boresch_attachment"].
> - Added 
> unabfe.py --only-boresch-attachment incremental mode with Boresch fingerprint compatibility checks.
> - Updated bfe_core.THERMODYNAMIC_CYCLE_DOC to include Delta G_attach.
> 
> ## Measurement
> Adopted value:
> 
> Delta G_attach = 5.601 +/- 0.223 kJ/mol = 1.339 +/- 0.053 kcal/mol
> 
> Candidate binding free energy becomes:
> 
> Delta G_bind = -3.460 kcal/mol with interval roughly -3.513 ~ -3.406.
> 
> The uncertainty is the half-difference between two independent measurements, not either single-run asymptotic sigma.
> 
> ## Important Notes
> - Hamiltonian REMD for attachment is intentionally not used: Boresch dihedral flips can create enormous U_B values and make exponential estimates single-frame dominated.
> - The attachment leg should stay single-basin to remain consistent with the analytic Boresch release correction.
> 
> ## Source
> docs/TODO.md - P1-17.
> 

### #80 — [P1] Resolve residual charging discrepancy against reference protocol（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/80
- 作者：Cedrus810
- 创建：2026-07-28T03:45:16Z
- 更新：2026-07-29T06:26:24Z
- 关闭：2026-07-29T06:26:24Z
- 标签：无
- 评论数：1

**正文：**

> ﻿## Status
> Open; updated from docs/TODO.md on 2026-07-28.
> 
> P1-18 still tracks the residual charging discrepancy versus 
> esult.txt, but the interpretation has been narrowed.
> 
> ## Current State
> Current charging contribution is about -0.3645 kcal/mol; 
> esult.txt gives -1.680 kcal/mol, leaving about +1.316 kcal/mol residual.
> 
> Evidence still does not support changing MBAR first: trapezoid TI and MBAR agree within roughly 1 kJ/mol for the current u_kn, and REMD exchange rates are around 0.61.
> 
> ## Resolved / Removed Hypotheses
> - P0-11 solvent box length scan is complete and did not eat the charging residual. The old 62.8865 kJ/mol solvent value is no longer merely from an invalid 3.000 nm box; it reflects the current Hamiltonian/protocol within the tested box range.
> - The constrained-vs-unconstrained fully charged ensemble question has been absorbed by P1-17. Once the attachment leg is included, the restrained charging leg is thermodynamically valid and must not be separately corrected.
> 
> ## Remaining Focus
> The remaining hard-evidence hypothesis is ligand-ligand electrostatic treatment / reference protocol mismatch:
> 
> - Current model freezes ligand-ligand Coulomb via explicit exceptions and scales only ligand-environment electrostatics.
> - 
> esult.txt lacks provenance, so its L-L treatment, PME settings, cutoff/tolerance/grid, water/salt, Boresch state, and lambda direction are unknown.
> - Recover reference mdp/scripts/logs if possible; otherwise compare frozen L-L decoupling against annihilation-like L-L treatment on the same frozen coordinates before declaring a code defect.
> 
> ## Source
> docs/TODO.md - P1-18.
> 

### #81 — [P1] Remove Stage 2 preoptimization fingerprint fail-open environment bypass（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/81
- 作者：Cedrus810
- 创建：2026-07-28T03:45:17Z
- 更新：2026-07-29T06:39:03Z
- 关闭：2026-07-29T06:39:03Z
- 标签：无
- 评论数：1

**正文：**

> ﻿## Status
> Open; created from docs/TODO.md on 2026-07-28.
> 
> ## Problem
> P1-20 found that Stage 2 preoptimization cache recovery still honors ABFE_DEBUG_SKIP_STAGE2_FINGERPRINT=1, forcing protocol_match = True on fingerprint mismatch. It is default-off and noisy, so not P0, but any stale job environment with the variable set can reuse lambda/window caches across incompatible code, Hamiltonian, coordinate, or Boresch states.
> 
> ## Required Work
> Remove the production-entry bypass. If one-time migration is needed, implement a separate explicit offline migration tool that validates physical inputs and rewrites the new fingerprint. Add regression coverage proving that setting the old environment variable no longer bypasses mismatch fail-closed behavior.
> 
> ## Source
> docs/TODO.md - P1-20.
> 

### #82 — [P2] Make GROMACS include auto-discovery portable on Windows and POSIX（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/82
- 作者：Cedrus810
- 创建：2026-07-28T03:45:19Z
- 更新：2026-08-12T16:48:29Z
- 关闭：2026-08-12T16:48:29Z
- 标签：无
- 评论数：1

**正文：**

> ﻿## Status
> Open; created from docs/TODO.md on 2026-07-28.
> 
> ## Problem
> P2-16 found that 
> unabfe.find_gmx_include_dir() uses Unix which gmx and falls back to personal /home/ruigengji/... paths. Explicit --gmx-path / GMXDATA still works, so this is not a P0/P1 runtime blocker, but automatic discovery fails on Windows even when gmx.exe is on PATH.
> 
> ## Required Work
> Use shutil.which("gmx") / shutil.which("gmx.exe"), derive the share/include directory from the executable location when possible, remove personal-directory fallbacks, and add Windows/POSIX mock tests.
> 
> ## Source
> docs/TODO.md - P2-16.
> 

### #83 — [P2] Feed real ligand bond topology into Boresch anchor selection（Issue，open）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/83
- 作者：Cedrus810
- 创建：2026-07-28T03:45:20Z
- 更新：2026-07-28T03:45:20Z
- 关闭：未关闭
- 标签：无
- 评论数：0

**正文：**

> ﻿## Status
> Open; created from docs/TODO.md on 2026-07-28.
> 
> ## Problem
> After ATT-11, receptor-side Boresch anchor selection can use real topology bonds, but ligand-side selection still falls back to the 0.22 nm geometric rule when mdtraj reads mmCIF without ligand bond records. The pipeline already has real ligand bonds through GromacsTopFile.topology.bonds() and generate_ligand_xml_from_top.
> 
> ## Required Work
> Provide ligand bond topology to the estimator, either by writing CONECT-aware PDB/topology input or by passing explicit ond_overrides. This should eliminate the ligand-side geometric fallback without changing intramolecular coordinates.
> 
> ## Source
> docs/TODO.md - Boresch topology follow-up.
> 

### #84 — [P2] Add membrane-aware barostat selection for membrane receptor systems（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/84
- 作者：Cedrus810
- 创建：2026-07-28T03:45:22Z
- 更新：2026-08-13T03:36:50Z
- 关闭：2026-08-13T03:36:50Z
- 标签：无
- 评论数：1

**正文：**

> ﻿## Status
> Open; created from docs/TODO.md on 2026-07-28.
> 
> ## Problem
> The current complex-leg pressure control uses isotropic openmm.MonteCarloBarostat, coupling x/y/z scaling. That is inappropriate for bilayer membrane receptors, where membrane area and thickness should not be scaled isotropically.
> 
> ## Required Work
> Add a system_type or equivalent branch to select MonteCarloMembraneBarostat (xy coupled, z independent, surface tension usually 0) or MonteCarloAnisotropicBarostat for membrane systems. Keep solvent leg independent; ligand-in-water solvent leg should not inherit the membrane complex box.
> 
> ## Source
> docs/TODO.md - membrane receptor follow-up.
> 

### #85 — [P1] Recompute convergence gates after split-half sigma inflation（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/85
- 作者：Cedrus810
- 创建：2026-07-29T06:27:23Z
- 更新：2026-08-12T16:42:37Z
- 关闭：2026-08-12T16:42:37Z
- 标签：无
- 评论数：1

**正文：**

> ## Status\nOpen; synced from docs/TODO.md on 2026-07-29.\n\nTracks P1-23.\n\n## Bug\nIn ibs_engine.solve_stage_integrated, inflate_sigma_from_split_half=True can replace total_error and per-segment uncertainty_kJ_mol, but it does not update max_endpoint_uncertainty_kJ_mol and does not re-evaluate converged.\n\nThat means a sigma adoption path can raise the reported uncertainty while convergence gates still read the pre-inflation smaller sigma. The flag is off by default, so this is P1 rather than P0.\n\n## Fix Requirement\nAny accepted sigma-inflation path must recompute endpoint uncertainty gates and final convergence status from the inflated values. This should be designed together with P1-22 / #78 uncertainty work, but the fail-open behavior is a real code bug.\n\n## Source\ndocs/TODO.md P1-23.

### #86 — [P1] Gate Boresch last-frame updates on dihedral deviations（Issue，closed）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/86
- 作者：Cedrus810
- 创建：2026-07-29T06:27:27Z
- 更新：2026-08-12T16:42:41Z
- 关闭：2026-08-12T16:42:41Z
- 标签：无
- 评论数：1

**正文：**

> ## Status\nOpen; synced from docs/TODO.md on 2026-07-29.\n\nTracks BOR-02, follow-up from the fixed BOR-01 Boresch dihedral sign bug.\n\n## Problem\nupdate_boresch_from_last_frame currently has strong validation for r0 and theta, but it does not check the three dihedral equilibrium values. That allowed mirrored dihedrals from the old sign bug to overwrite the correct boresch_simple.json values.\n\n## Fix Requirement\nReuse boresch_committed_deviation_sigma to compare the newly derived equilibrium values against the original committed values. Wrap dihedral differences to pi. Use the existing thresholds BORESCH_COMMITTED_MAX_DEVIATION_SIGMA = 4.0 and WARN = 2.5.\n\nExcess deviation should warn and keep orig_eq, not raise. The original values come from the 500-frame ensemble mean and are more reliable than a single last-frame re-anchor; a hard failure would create false-positive production kills.\n\n## Test\nExtend tests/test_boresch_committed_gate.py near the existing committed-gate tests.\n\n## Source\ndocs/TODO.md BOR-02 and archive/BOR-01-boresch-dihedral-sign-fixed-2026-07-29.md.

### #87 — [P1] Design vdW/stage2 frame-selection and uncertainty protocol（Issue，open）

- 链接：https://github.com/Cedrus810/openmm_IBS_dev/issues/87
- 作者：Cedrus810
- 创建：2026-07-29T06:27:30Z
- 更新：2026-08-12T16:42:31Z
- 关闭：未关闭
- 标签：无
- 评论数：0

**正文：**

> ## Status
> 
> Open; synced with the current P1-22 decision after the 2026-08 documentation update.
> 
> ## Problem
> 
> The finite-sample frame-selection and uncertainty protocol for vdW/stage2 remains unresolved. Historical all-frame observations are evidence for investigation only and are not the current production conclusion.
> 
> ## Current constraints
> 
> - Production remains TMBAR-only for vdW/stage2.
> - Do not introduce BAR or TI: physical lambda rows have `n_k=0`, and softcore vdW does not have the required on-disk derivative data.
> - Do not silently adopt all-frame estimates, sqrt(g) sigma inflation, or bootstrap sigma in the production estimator.
> - Keep this work separate from estimator-layer changes and from P1-19/#78 calibration.
> 
> ## Research direction
> 
> A zero-GPU moving-block/bootstrap study over existing trajectories remains a possible future design. It must be independently specified, validated, and kept off the production path until its point-estimate, block-length, gate, and uncertainty semantics are accepted.
> 
> ## Acceptance criteria
> 
> - Specify a reproducible finite-sample frame-selection protocol.
> - Define conservative uncertainty and convergence gates without treating correlated frames as independent.
> - Demonstrate the proposal on existing trajectories and independent repeats.
> - Record an explicit decision before any production integration.
> 
> ## Source
> 
> `docs/TODO.md` P1-22 and the current TMBAR-only production constraints.
> 

### #88 — Integrate experimental charge-transfer mai