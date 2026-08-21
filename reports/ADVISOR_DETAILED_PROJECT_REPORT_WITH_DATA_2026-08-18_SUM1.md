# ADVISOR_DETAILED_PROJECT_REPORT_WITH_DATA：ABFE-IBS 项目详细进展报告

# Advisor-Facing Detailed Project Report with Data: ABFE-IBS Methods, Results, Validation, Failed Routes, and Next-Stage Decisions

> 中文：导师汇报用详细数据版；综合截至 2026-08-18 的项目方法、软件、数值结果、验证证据与研究路线判断。  
> English: Advisor-facing detailed data report consolidating the project methods, software, numerical results, validation evidence, and research-route decisions available through 2026-08-18.

> 中文：整理日期为 2026-08-18；本版重新盘点了当日工作区可见的代码、文档、机器可读结果与验证产物，各实验的实际运行日期以其 artifact provenance 为准。  
> English: Consolidation date: 2026-08-18. This version re-audits the code, documents, machine-readable results, and validation artifacts visible in the workspace on that date; the actual execution date of each experiment is governed by its artifact provenance.

> 中文：本报告用于向导师完整说明项目已经完成的工作、当前可以支持的结论、尚未闭合的科学风险以及建议的下一阶段实验。它不是最终论文结果；原始 JSON、日志、checkpoint、预注册协议和代码中的可复核事实优先于叙述性文字。  
> English: This report is intended to provide the advisor with a complete account of the work performed, the conclusions currently supported by evidence, the scientific risks that remain unresolved, and the recommended next experiments. It is not a final publication result; auditable facts in raw JSON artifacts, logs, checkpoints, preregistration protocols, and source code take precedence over narrative text.

---

**阅读逻辑 / Reading logic:** 当前结论与优先级 → 热力学循环和数据流 → Stage 0/1/2 与 estimator → 数值结果、失败和总不确定度 → charged/membrane 扩展 → DEXP 独立方法线 → Route 1/Route 2/EDS 决策 → 实现与证据附录 → 最终软件/MACE/local-residual 专题章。

## 1. 执行摘要 / Executive Summary

### 1.1 当前已经完成的核心工作

项目已经形成了一条可运行的绝对结合自由能软件链：输入准备、复杂物与溶剂两条热力学腿、PME decharging、ACE softcore vanishing、dual-λ 路径、IBS 扩展系综采样、冻结权重生产、local TMBAR、LRC、Boresch restraint correction、checkpoint/resume 和结构化 provenance。与此同时，项目建立了分层测试与实验协议，用于区分数学正确性、软件正确性、短程稳定性、真实 GPU 成本和最终 ESS/GPU-hour 效用。

The project now has an executable absolute binding free-energy stack covering input preparation, complex and solvent thermodynamic legs, PME decharging, ACE softcore vanishing, a dual-λ path, IBS expanded-ensemble sampling, frozen-weight production, local TMBAR, long-range correction, Boresch restraint correction, checkpoint/resume, and structured provenance. In parallel, the project has established layered tests and experimental protocols that distinguish mathematical correctness, software correctness, short-run stability, real GPU cost, and final ESS-per-GPU-hour utility.

### 1.2 当前最重要的科学结论

当前最强的重复性证据来自 Seed 20260906 与 Seed 20260907 两条 current-protocol-family complex legs：分别为 `48.437 ± 0.436` 与 `48.075 ± 0.302 kcal/mol`，差值仅 `0.362 kcal/mol`，约为合并内部误差的 `0.68σ`。因此，当前 complex-leg total 表现出非常强的独立运行稳定性信号。这个正面结论必须与完整 binding cycle 分开：Seed06 没有完成 solvent leg，所以尚不能证明 full-cycle ΔG_bind 同样稳定，也不能据此给出最终可发表值。

The strongest repeatability evidence comes from the Seed 20260906 and Seed 20260907 complex legs under the current protocol family: `48.437 ± 0.436` and `48.075 ± 0.302 kcal/mol`, respectively. Their difference is only `0.362 kcal/mol`, approximately `0.68σ` of the combined internal uncertainty. The current complex-leg total therefore shows a remarkably strong independent-run stability signal. This positive result must be separated from the complete binding cycle: Seed06 has no completed solvent leg, so the evidence does not yet establish equally strong full-cycle ΔG_bind repeatability or a publication-ready final value.

mixed-history `output_lrc_fix`（`−5.536 ± 0.601 kcal/mol`）与 Seed07 binding（`−11.927 ± 0.415 kcal/mol`）属于不同 protocol/provenance identities，不应掩盖 Seed06/07 complex-leg 的正面结果。二者 `6.391 kcal/mol` 的间隔只是跨身份 discrepancy，不是 seed-only variance。

The mixed-history `output_lrc_fix` value (`−5.536 ± 0.601 kcal/mol`) and Seed07 binding value (`−11.927 ± 0.415 kcal/mol`) belong to different protocol/provenance identities and must not obscure the Seed06/07 complex-leg result. Their `6.391 kcal/mol` separation is a descriptive cross-identity discrepancy, not a seed-only variance estimate.

### 1.3 核心技术路线与阶段决策 / Core Route and Stage Decisions

**中文**

| 路线或对象 | 当前证据 | 阶段决策 |
|---|---|---|
| Atenolol 最终可发表 ΔG | 两个非严格同条件候选为 `−5.536` 与 `−11.927 kcal/mol`；不可直接合并 | `NOT_PUBLICATION_READY`；不得挑选单值或归因于 seed |
| 可溶液生产基线 | dual-λ + PME + ACE softcore + IBS v29/TMBAR + Boresch + soluble LRC | 保持为当前主线；优先完成同协议独立重复 |
| 当前 complex-leg 可重复性 | Seed06/Seed07 为 `48.437 ± 0.436` / `48.075 ± 0.302 kcal/mol`；差 `0.362 kcal/mol` (`0.68σ`) | **强稳定性正信号**；与完整循环资格化分开 |
| Seed 20260907 artifact | `−11.927 ± 0.415 kcal/mol`；与旧候选 protocol identity 不完全一致 | `AVAILABLE_BUT_NOT_POOLED` |
| 带电配体扩展 | C1/C2/C3 seam 与真实端点验证通过 | 工程资格前进；完整 charged ABFE cycle 仍待完成 |
| 膜扩展 | neutral 5 ns smoke 跑通；旧 100 ns 路径因 topology loss 失效 | 不进入主结论；修复后做 production-length repeat |
| MACE-informed DEXP | 单体系局部 projection、force/Hessian 与 short stability 有正信号 | 独立 potential-family 研究；尚未并入 production ABFE |
| Route 2 fixed-target residual | offline representation 与 CUDA scoped cost/correctness 已有证据 | EXP-029 是下一决定性实验；production entry 尚未接线 |
| Route 1 / EDS / λ-EDS | endpoint-constrained path design 仍是概念/对照空间 | 与 Route 2 分账；不改写当前 target identity |

**English**

| Route or item | Current evidence | Stage decision |
|---|---|---|
| Atenolol publication-ready ΔG | Two candidates under non-identical conditions: `−5.536` and `−11.927 kcal/mol`; they cannot be pooled | `NOT_PUBLICATION_READY`; do not select one value or attribute the discrepancy to the seed |
| Soluble production baseline | dual-λ + PME + ACE softcore + IBS v29/TMBAR + Boresch + soluble LRC | Retain as the current mainline; prioritize an independent repeat under the same protocol |
| Current complex-leg repeatability | Seed06/Seed07 are `48.437 ± 0.436` / `48.075 ± 0.302 kcal/mol`; difference `0.362 kcal/mol` (`0.68σ`) | **Strong positive stability signal**; keep separate from full-cycle qualification |
| Seed 20260907 artifact | `−11.927 ± 0.415 kcal/mol`; not fully identical to the historical candidate’s protocol identity | `AVAILABLE_BUT_NOT_POOLED` |
| Charged-ligand extension | C1/C2/C3 seam and real-endpoint validation passed | Engineering qualification advances; complete charged ABFE cycle remains pending |
| Membrane extension | Neutral 5 ns smoke run completed; historical 100 ns path failed because of topology loss | Excluded from the main conclusion; perform a production-length repeat after repair |
| MACE-informed DEXP | Positive signals from single-system local projection, force/Hessian, and short stability | Independent potential-family study; not yet part of production ABFE |
| Route 2 fixed-target residual | Offline representation and scoped CUDA cost/correctness evidence are available | EXP-029 is the next decisive experiment; production entry is not wired |
| Route 1 / EDS / λ-EDS | Endpoint-constrained path design remains a conceptual/control space | Keep separately accounted from Route 2; do not rewrite the current target identity |

本报告之后严格按照实际 pipeline 顺序展开。机器学习软件、MACE/TorchForce 历史、专用 CUDA backend 与局部多体 residual 不再穿插在主流程中，而统一放在全文最后一章。

The remainder of this report follows the actual pipeline in order. Machine-learning software, the MACE/TorchForce history, the dedicated CUDA backend, and the local many-body residual are no longer interleaved with the main workflow; they are consolidated in the final chapter.

## 2. 理论框架、热力学循环与管线架构 / Thermodynamic Cycle and Pipeline Architecture

### 2.1 项目目标与困难

项目目标是估计 Atenolol-rank11 的绝对结合自由能。软件不等待自然结合/解离事件，而是分别构造受体复合物和水溶液中的 alchemical legs，再通过标准热力学循环得到结合自由能。

The objective is to estimate the absolute binding free energy of Atenolol-rank11. Instead of waiting for spontaneous binding and unbinding, the software constructs separate alchemical legs in the receptor complex and aqueous solvent and combines them through a standard thermodynamic cycle.

当前符号约定为

\[
\Delta G_{\mathrm{bind}}
=\Delta G_{\mathrm{solvent}}
-\Delta G_{\mathrm{complex}}
+\Delta G_{\mathrm{APBS}}.
\]

A negative value denotes favorable binding. The sign convention, inclusion of Boresch release, and APBS scalar must be read from the final ledger rather than reconstructed from memory.

主要困难包括短程端点奇异性、相邻状态 overlap、离子/水/侧链慢重排、Boresch attachment 与 release、PME 电荷处理、LJ 长程修正、resume 后分布身份，以及单次 covariance 是否低估跨运行方差。

The main challenges are short-range endpoint singularities, overlap between neighboring states, slow ion/water/side-chain reorganization, Boresch attachment and release, PME charge treatment, LJ long-range corrections, distribution identity after resume, and underestimation of between-run variance by single-run covariance.

### 2.2 Complex leg

complex leg 包含：Boresch restraint attachment、Stage 1 PME decharging、Stage 2 van der Waals vanishing，以及解析 Boresch standard-state release：

\[
\Delta G_{\mathrm{complex}}
=\Delta G_{\mathrm{attach}}
+\Delta G_{\mathrm{decharge}}^{\mathrm{complex}}
+\Delta G_{\mathrm{vdW}}^{\mathrm{complex}}
+\Delta G_{\mathrm{release}}.
\]

The complex leg contains sampled Boresch attachment, Stage 1 PME decharging, Stage 2 van der Waals vanishing, and analytical standard-state release. Attachment and release are distinct physical operations and must not be merged or counted twice.

Boresch restraint 使用一个距离、两个角和三个二面角限制配体相对于受体的平移和转动；当前标准态体积为 `1.6605 nm³`。旧候选的 Boresch 项约为 `−38.7609 kJ/mol`，seed 20260907 为 `−35.9300 kJ/mol`，两者均已包含在各自 complex ledger 中。

The Boresch restraint uses one distance, two angles, and three dihedrals to control ligand translation and rotation relative to the receptor; the standard-state volume is `1.6605 nm³`. The Boresch terms of approximately `−38.7609 kJ/mol` in the historical candidate and `−35.9300 kJ/mol` in seed 20260907 are already included in their respective complex ledgers.

旧 artifact 中一个径向力常数从约 `7355.9` 被裁剪至 `2000 kJ mol⁻¹ nm⁻²`。这不自动证明结果错误，但意味着 anchor 与 force-constant sensitivity 必须进入正式资格化。

In the earlier artifact, a radial force constant was clipped from approximately `7355.9` to `2000 kJ mol⁻¹ nm⁻²`. This does not automatically invalidate the run, but it requires anchor and force-constant sensitivity analysis before qualification.

### 2.3 Solvent leg

solvent leg 不使用 Boresch restraint：

\[
\Delta G_{\mathrm{solvent}}
=\Delta G_{\mathrm{decharge}}^{\mathrm{solvent}}
+\Delta G_{\mathrm{vdW}}^{\mathrm{solvent}}.
\]

The solvent leg contains ligand decharging and vanishing in water without a Boresch restraint. Its topology, box, energy matrix, and provenance must remain independent from the complex leg.

### 2.4 计算管线、数据流与不可变接口 / Computational Pipeline, Data Flow, and Immutable Interfaces

#### 2.4.1 从输入到最终结果 / From Inputs to the Final Result

```text
GROMACS .gro/.top + ligand identity
  -> include/topology parsing and OpenMM System cache
  -> complex and solvent construction
  -> Boresch setup for complex
  -> Stage 1 PME decharging
  -> static-neutral handoff
  -> Stage 2 ACE-softcore + IBS
  -> local TMBAR/MBAR + LRC
  -> Boresch release + optional APBS
  -> final_binding_results.json + provenance
```

`runabfe.py` 负责 CLI 与两腿调度；`abfe_core.py` 负责基础势、softcore、Boresch 和热力学循环；`abfe_preoptimizer.py` 负责 λ path 预优化；`ibs_engine.py` 负责 IBS、(f_k)、TMBAR、LRC 和统计门；`abfe_pipeline.py` 负责 stage orchestration、checkpoint、resume、rescue 和最终汇总。

`runabfe.py` handles the CLI and two-leg orchestration; `abfe_core.py` defines baseline potentials, softcore interactions, Boresch terms, and the thermodynamic cycle; `abfe_preoptimizer.py` optimizes the λ path; `ibs_engine.py` implements IBS, (f_k), TMBAR, LRC, and statistical gates; and `abfe_pipeline.py` coordinates stages, checkpointing, resume, rescue, and final aggregation.

当前主要协议身份为 IBS v29、thermodynamic path v21、LRC v3、WCA accounting v2 和 ESS gate v3。旧 artifact 只能在自身协议身份下解释，不能自动视为与 v29 兼容。

The principal current identities are IBS v29, thermodynamic path v21, LRC v3, WCA accounting v2, and ESS gate v3. Historical artifacts remain interpretable only under their own protocol identities and are not automatically compatible with v29.

#### 2.4.2 三阶段顺序与不可交换的接口 / Three-Stage Order and Non-Interchangeable Interfaces

**中文**

| 步骤 | 复合物腿 | 溶剂腿 | 关键输入 | 关键输出 |
|---|---|---|---|---|
| Stage 0 | Boresch restraint attachment | 不使用 | 构象、锚点、平衡几何 | 受约束复合物与连接功 |
| Stage 1 | PME 去电荷 | PME 去电荷 | 带电配体、完整 vdW | 静态中性 handoff |
| Stage 2 | ACE-softcore vanishing + IBS | ACE-softcore vanishing + IBS | 中性配体、排斥体积 | 考虑 mixture 的 vdW 自由能 |
| Final analysis | release + LRC + 可选 APBS | LRC + 可选 APBS | 两腿台账 | binding free energy 与不确定度包 |

**English**

| Step | Complex leg | Solvent leg | Key input | Key output |
|---|---|---|---|---|
| Stage 0 | Boresch restraint attachment | Not used | Pose, anchors, equilibrium geometry | Restrained complex and attachment work |
| Stage 1 | PME decharging | PME decharging | Charged ligand with full vdW | Static-neutral handoff |
| Stage 2 | ACE-softcore vanishing + IBS | ACE-softcore vanishing + IBS | Neutral ligand with excluded volume | Mixture-aware vdW free energy |
| Final analysis | Release + LRC + optional APBS | LRC + optional APBS | Both leg ledgers | Binding free energy and uncertainty package |

Stage 0、Stage 1 和 Stage 2 不是三个可任意拼装的算法标签，而是具有明确输入身份和 handoff contract 的顺序流程。任何结构、anchor、charge policy、state mapping 或 protocol fingerprint 的变化，都必须使下游 cache 和 resume fail closed。

Stages 0, 1, and 2 are not freely interchangeable algorithm labels. They are an ordered workflow with explicit input identities and handoff contracts. Any change to the structure, anchors, charge policy, state mapping, or protocol fingerprint must invalidate downstream caches and cause resume to fail closed.

## 3. 主干算子物理验证与工程实现 / Physical Validation and Engineering of Core Operators

在 restraint attachment 之前，物理输入先经过 PBC-integrity repair、minimization 与 Langevin equilibration。通用平衡设置为 `5,000,000` steps、`2 fs`（约 `10 ns`）。已完成的膜体系 `100 ns` equilibration record 显示体系整体稳定，约前 `5 ns` 之后变化已经很小；因此本文采用 `10 ns` 作为实际平衡长度。后续仍结合质量指标与 production evidence 对 equilibration artifact 进行整体判断。

Before restraint attachment, the physical input passes PBC-integrity repair, minimization, and Langevin equilibration. The general equilibration setting is `5,000,000` steps at `2 fs` (approximately `10 ns`). In the completed `100 ns` membrane equilibration record, the system remained stable and changed little after approximately the first `5 ns`; accordingly, `10 ns` is used here as a practical equilibration length. A completed equilibration artifact is still evaluated together with the downstream quality and production evidence.

### 3.1 Stage 0：Boresch restraint attachment / Stage 0: Boresch Restraint Attachment

**中文**

| 维度 | Stage 0 内容 |
|---|---|
| 目标 | 在不破坏结合构象的前提下，逐步固定配体相对受体的 6 个刚体自由度 |
| 输入 | 经 PBC 修复的复合物、明确的受体/配体锚点、未过期的平衡几何 |
| 哈密顿量变化 | 逐步施加 1 个距离、2 个角和 3 个二面角约束；仅属于复合物腿 |
| 采样与估计 | 对约束连接功进行采样；约束释放项在热力学循环末端解析计算 |
| 输出 | 受约束复合物、连接项台账、锚点与力常数来源记录 |
| 核心风险 | 几何过期、角/二面角映射错误、力常数过强或被裁剪、结合构象位移 |
| 当前状态 | 软件路径可运行；锚点与力常数敏感性尚未闭合 |

**English**

| Dimension | Stage 0 content |
|---|---|
| Objective | Gradually restrain the six rigid-body degrees of freedom of the ligand relative to the receptor without disrupting the binding pose |
| Input | PBC-repaired complex, explicit receptor and ligand anchors, and a current equilibrium geometry |
| Hamiltonian change | Gradually apply one distance, two angle, and three dihedral restraints; this term belongs only to the complex leg |
| Sampling and estimation | Sample the restraint-attachment work; calculate the restraint-release term analytically at the end of the thermodynamic cycle |
| Output | Restrained complex, attachment ledger, and anchor and force-constant provenance |
| Main risks | Stale geometry, incorrect angle or dihedral mapping, excessive or clipped force constants, and binding-pose displacement |
| Current status | The software path is operational; anchor and force-constant sensitivity has not yet been closed |

**中文说明**

Stage 0 只出现在 complex leg，用来逐步施加 Boresch restraint。输入结构需要先通过 pose consistency、PBC repair 和 anchor geometry 检查；结构或 anchor identity 改变后，旧 reference geometry 不再代表同一个 Hamiltonian，需要重新生成。

attachment protocol version 为 1。默认 λ schedule 是 `[0, 0.1, 0.35, 1.0]`，每态运行 `250,000` steps，其中 `50,000` steps 用于 equilibration，记录 stride 为 `1,000`，默认 seed 为 `20260728`。报告方向为 A′→A。由于 restraint potential 非负，负的 `ΔG_attach` 被视为异常；BAR/TI 差要求不超过 `1.0 kJ/mol`，split-half 差要求不超过 `0.5 kJ/mol`。

**English notes**

Stage 0 belongs only to the complex leg and gradually attaches the Boresch restraint. The input first passes pose-consistency, PBC-repair, and anchor-geometry checks. If the structure or anchor identity changes, the previous reference geometry no longer defines the same Hamiltonian and is regenerated.

Attachment protocol version 1 uses the default λ schedule `[0, 0.1, 0.35, 1.0]`, with `250,000` steps per state, `50,000` equilibration steps, a stride of `1,000`, and default seed `20260728`. The reported direction is A′→A. Because the restraint potential is non-negative, a negative `ΔG_attach` is treated as an error. BAR/TI disagreement is limited to `1.0 kJ/mol`, and the split-half difference to `0.5 kJ/mol`.

#### 3.1.1 实际 Boresch Hamiltonian、估计量与解析释放 / Implemented Boresch Hamiltonian, Estimator, and Analytical Release

**中文**

受体锚点记为 \(R_0,R_1,R_2\)，配体锚点记为 \(L_0,L_1,L_2\)。本项目实际使用的六个坐标为

\[
\begin{gathered}
r=|R_0-L_0|,\\
\theta_A=\angle(R_1,R_0,L_0),\qquad
\theta_B=\angle(R_0,L_0,L_1),\\
\phi_A=\operatorname{dih}(R_2,R_1,R_0,L_0),\qquad
\phi_B=\operatorname{dih}(R_1,R_0,L_0,L_1),\\
\phi_C=\operatorname{dih}(R_0,L_0,L_1,L_2).
\end{gathered}
\]

源码中的 attachment 势不是把所有坐标都写成全局二次谐势，而是

\[
\begin{aligned}
U_B(x;\lambda_B)=\lambda_B\Big[&\frac{1}{2} k_r(r-r_0)^2
+k_{\theta A}\{1-\cos(\theta_A-\theta_{A0})\}
+k_{\theta B}\{1-\cos(\theta_B-\theta_{B0})\}\\
&+k_{\phi A}\{1-\cos(\phi_A-\phi_{A0})\}
+k_{\phi B}\{1-\cos(\phi_B-\phi_{B0})\}
+k_{\phi C}\{1-\cos(\phi_C-\phi_{C0})\}\Big].
\end{aligned}
\]

距离项是二次谐势，角度和二面角项是周期余弦势。当偏差 \(\delta\) 很小时，周期余弦势可近似写为

\[
k(1-\cos\delta)\approx \frac{1}{2}k\delta^2.
\]

这个等式只在小偏差、单势阱附近成立，不能把完整周期势误写成全局二次势。

端点为 \(\lambda_B=0\)（未施加 restraint 的物理结合态 \(A'\)）与 \(\lambda_B=1\)（完整 restraint 态 \(A\)）。由于 \(U_B\ge0\)，方向 \(A'\to A\) 满足

\[
\Delta G_{\mathrm{attach}}
=-k_BT\ln\left\langle e^{-\beta U_B}\right\rangle_{A'}\ge0.
\]

生产主值由相邻窗口 BAR 链给出；由于 \(U_B(x;\lambda_B)=\lambda_BU_B(x)\)，TI 交叉检查为

\[
\Delta G_{\mathrm{TI}}
=\int_0^1\left\langle\frac{\partial U}{\partial\lambda_B}\right\rangle_{\lambda_B}d\lambda_B
=\int_0^1\langle U_B\rangle_{\lambda_B}d\lambda_B.
\]

解析 standard-state release 单独计算为

\[
\Delta G_{\mathrm{release}}
=-RT\ln\left[
\frac{8\pi^2V^\circ}{r_0^2\sin\theta_{A0}\sin\theta_{B0}}
\frac{\sqrt{k_rk_{\theta A}k_{\theta B}k_{\phi A}k_{\phi B}k_{\phi C}}}
{(2\pi RT)^3}
\right],
\qquad V^\circ=1.6605\ \mathrm{nm^3}.
\]

该 release 公式是六个 restraint coordinates 的局部单势阱 Gaussian approximation。\(r_0\) 使用 nm，角度使用 rad，力常数使用一致的 kJ mol\(^{-1}\) 单位。分母是 \((2\pi RT)^3\)，不是 \((2\pi RT)^{3/2}\)。若二面角跨越不同周期势阱，解析近似失效。attachment 与 release 的方向必须明确，并且在最终 cycle 中各计入一次。

**English**

The receptor anchors are denoted by \(R_0,R_1,R_2\), and the ligand anchors by \(L_0,L_1,L_2\). The six implemented coordinates are

\[
\begin{gathered}
r=|R_0-L_0|,\\
\theta_A=\angle(R_1,R_0,L_0),\qquad
\theta_B=\angle(R_0,L_0,L_1),\\
\phi_A=\operatorname{dih}(R_2,R_1,R_0,L_0),\qquad
\phi_B=\operatorname{dih}(R_1,R_0,L_0,L_1),\\
\phi_C=\operatorname{dih}(R_0,L_0,L_1,L_2).
\end{gathered}
\]

The implemented attachment Hamiltonian is

\[
\begin{aligned}
U_B(x;\lambda_B)=\lambda_B\Big[&\frac{1}{2} k_r(r-r_0)^2
+k_{\theta A}\{1-\cos(\theta_A-\theta_{A0})\}
+k_{\theta B}\{1-\cos(\theta_B-\theta_{B0})\}\\
&+k_{\phi A}\{1-\cos(\phi_A-\phi_{A0})\}
+k_{\phi B}\{1-\cos(\phi_B-\phi_{B0})\}
+k_{\phi C}\{1-\cos(\phi_C-\phi_{C0})\}\Big].
\end{aligned}
\]

The distance term is quadratic, whereas the angle and dihedral terms are periodic cosine potentials. For a small displacement \(\delta\),

\[
k(1-\cos\delta)\approx \frac{1}{2}k\delta^2.
\]

This approximation is local to a single well and must not be interpreted as a globally quadratic potential.

The endpoints are \(\lambda_B=0\), the unrestrained physical bound state \(A'\), and \(\lambda_B=1\), the fully restrained state \(A\). Because \(U_B\ge0\),

\[
\Delta G_{\mathrm{attach}}
=-k_BT\ln\left\langle e^{-\beta U_B}\right\rangle_{A'}\ge0.
\]

Adjacent-window BAR is the production estimator. Since \(U_B(x;\lambda_B)=\lambda_BU_B(x)\), the TI consistency check is

\[
\Delta G_{\mathrm{TI}}
=\int_0^1\left\langle\frac{\partial U}{\partial\lambda_B}\right\rangle_{\lambda_B}d\lambda_B
=\int_0^1\langle U_B\rangle_{\lambda_B}d\lambda_B.
\]

The analytical standard-state release is evaluated separately:

\[
\Delta G_{\mathrm{release}}
=-RT\ln\left[
\frac{8\pi^2V^\circ}{r_0^2\sin\theta_{A0}\sin\theta_{B0}}
\frac{\sqrt{k_rk_{\theta A}k_{\theta B}k_{\phi A}k_{\phi B}k_{\phi C}}}
{(2\pi RT)^3}
\right],
\qquad V^\circ=1.6605\ \mathrm{nm^3}.
\]

This release expression is a local, single-basin Gaussian approximation. It uses nm for \(r_0\), radians for angular coordinates, and consistent kJ mol\(^{-1}\) force constants. The denominator is \((2\pi RT)^3\), not \((2\pi RT)^{3/2}\). Dihedral transitions between periodic wells invalidate the approximation. Attachment and release have explicit directions and enter the final cycle exactly once each.

#### 3.1.2 Boresch 参数来源模块 / Boresch Parameter-Source Modules

**中文**

| 功能分支 | 模块介绍 | 输入与输出 | 状态与分析 |
|---|---|---|---|
| `simple` | 根据受体–配体几何和轨迹涨落选择锚点、参考几何与力常数 | 输入已平衡复合物；输出统一的 Boresch 参数记录 | **当前推荐**；默认不依赖 MACE，仍需通过谐性与结合构象检查 |
| `fluctuation` | 根据几何涨落估计约束参数 | 输入轨迹涨落统计；输出与 `simple` 相同的哈密顿量格式 | **当前推荐/便于诊断**；谐性通过不能替代锚点敏感性分析 |
| `traditional` | 读取用户提供的传统锚点和参数文件 | 输入外部锚点文件；输出标准 Boresch 参数记录 | **可选**；外部文件需自行保证来源可追溯和几何一致性 |
| `orb_ml` | 读取外部 ORB/ML 约束方案 | 输入外部 ORB/ML 参数文件；输出标准 Boresch 参数记录 | **可选/历史兼容输入**；ML 物理曲率不等于人为约束刚度，不能单凭该来源认可力常数 |
| `orb_simple` | 当前代码对单个 ORB/MACE 候选进行口袋局部力投影 | 输入模型、结构和候选；输出标准参数记录 | **实验性/拟重构**；更合理的职责是锚点评分和病态锚点识别，而不是直接预测约束力常数 |
| `auto` | 枚举并评价多个 ORB/MACE 锚点候选 | 输入模型与候选集合；输出选定方案和候选记录 | **实验性/拟重构**；锚点选择与刚度验证需要使用相互独立的数据，避免同轨迹重复使用 |
| `boresch=false` | 不施加 Boresch 约束 | 不执行 Stage 0 连接或解析释放 | 仅用于诊断或特殊对照；不用于标准复合物腿 ABFE |

**English**

| Functional branch | Module description | Input and output | Status and analysis |
|---|---|---|---|
| `simple` | Selects anchors, reference geometry, and force constants from receptor–ligand geometry and trajectory fluctuations | Input: equilibrated complex; output: unified Boresch parameter record | **Currently recommended**; normally MACE-independent and must pass harmonicity and pose gates |
| `fluctuation` | Explicitly estimates restraint parameters from geometric fluctuations | Input: trajectory fluctuation statistics; output: the same Hamiltonian format as `simple` | **Currently recommended / diagnostic-friendly**; harmonicity cannot replace anchor sensitivity |
| `traditional` | Reads user-supplied traditional anchor/parameter files | Input: external anchor file; output: standard Boresch record | **Optional**; provenance and geometric consistency remain the responsibility of the external file |
| `orb_ml` | Reads an external ORB/ML restraint proposal | Input: external ORB/ML parameter file; output: standard Boresch record | **Optional / backward-compatible input**; ML physical curvature is not an artificial restraint stiffness and cannot alone qualify force constants |
| `orb_simple` | Current code applies local pocket-force projection to one ORB/MACE candidate | Input: model, structure, and candidate; output: standard record | **Experimental / redesign planned**; its defensible role is anchor scoring and bad-anchor detection, not direct prediction of restraint constants |
| `auto` | Enumerates, scores, and selects among multiple ORB/MACE restraint proposals | Input: model and candidate set; output: selected proposal and candidate ledger | **Experimental / redesign planned**; anchor selection and stiffness validation require data separation to prevent same-trajectory double dipping |
| `boresch=false` | Applies no Boresch restraint | No Stage 0 attachment/release | Diagnostic or special control only; not for standard complex-leg ABFE |

##### 3.1.2.1 ORB/ML 的根本职责边界：选择 anchor，而不是预测人为 restraint 刚度 / Fundamental ORB/ML Boundary: Select Anchors, Do Not Predict Artificial Restraint Stiffness

**中文**

Boresch restraint 的 \(K_{\mathrm{restraint}}\) 不是需要由 ORB/MACE “预测”的真实化学键刚度。Boresch Hamiltonian 是人为加入的取向约束；它的任务是在 decoupling 时防止配体漂移或翻转、维持选定的相对 orientation、提供可充分采样的 restraint ensemble，并允许在声明的单盆地解析近似下把这项人工自由能补回。ORB/MACE 局部力或 Hessian 所描述的则是未加 restraint 时局部物理势面的 \(K_{\mathrm{physical}}\)。一般不存在

\[
K_{\mathrm{restraint}}=K_{\mathrm{physical}}
\]

这一物理要求。

例如，天然口袋中的某个 torsion 即使只有 \(k_{\mathrm{physical}}=3\)，为了阻止 decoupled ligand 翻转，也完全可能需要 \(k_{\mathrm{restraint}}=40\)。反之，一个物理方向很硬，也不意味着人工 restraint 应机械地取到 500。生产参数应由 sampling stability、single-basin behavior、overlap、解析 correction 的适用条件和预注册 sensitivity protocol 决定，而不是把 ML curvature 当作 restraint constant。

这里还有一个数学上的问题。即使我们真的想估计局部物理刚度，它也应该是耦合的 \(6\times6\) 对象。对六个 Boresch coordinates

\[
\mathbf q=(r,\theta_A,\theta_B,\phi_A,\phi_B,\phi_C)^T,
\]

局部二次模型原则上应写成

\[
U(\mathbf q)\approx U(\mathbf q_0)
+\frac{1}{2}(\mathbf q-\mathbf q_0)^T
\mathbf K_{\mathrm{physical}}
(\mathbf q-\mathbf q_0),
\]

其中 \(\mathbf K_{\mathrm{physical}}\) 一般含有 \(r\leftrightarrow\theta\)、\(\theta\leftrightarrow\phi\) 以及 \(\phi_A\leftrightarrow\phi_B\) 等非零交叉项。当前 `orb_simple` 的逐坐标估计

\[
k_i=-\frac{\operatorname{Cov}(F_i,q_i)}{\operatorname{Var}(q_i)}
\]

等价于六次独立 regression，忽略了这些耦合。即使将 generalized force 修正为声明 metric 下的

\[
\mathbf Q=\mathbf G^{-1}\mathbf J\mathbf F,
\]

比较合理的物理诊断也应是一次联合拟合

\[
\Delta\mathbf Q=-\mathbf K_{\mathrm{physical}}\Delta\mathbf q,
\]

而不是六个互不相关的 scalar regressions。

不过，生产 Boresch restraint 仍适合保持 diagonal form，因为当前解析 correction 是按独立局部项构造的。完整矩阵更适合诊断 anchor 是否退化或强耦合；最终的人工 \(\mathbf K_R\) 则由采样协议单独选择。

另一个不能忽略的区别是：瞬时 microscopic generalized force 不是 PMF 斜率。即使 Hamiltonian、metric、torsion branch 和完整体系 ML force 全部正确，单帧 \(\mathbf Q(\mathbf q,x)\) 也不能直接等同于

\[
-\nabla_{\mathbf q}A(\mathbf q),
\qquad
A(\mathbf q)=-k_BT\ln P(\mathbf q).
\]

距离、角度和二面角是 curvilinear coordinates；其有效自由能还包含 coordinate metric、Jacobian/Fixman 或 entropic contribution，以及被积分掉的蛋白、溶剂和离子自由度。真正与 ensemble 有关的是 PMF curvature \(\nabla_{\mathbf q}^2A\)，而不是一张 snapshot 上 Cartesian potential Hessian 的简单投影。

这也是 trajectory fluctuation 估计

\[
k_i^{\mathrm{fluc}}\approx\frac{k_BT}{\operatorname{Var}(q_i)}
\]

在概念上更接近“有效自由能曲率”的原因。它仍依赖 harmonic、single-basin 与有限采样近似，但至少回答的是 thermal ensemble 中该 coordinate 波动多大。当前 `_apply_hybrid_filter()` 在 force regression 不可信时回退到这一表达式；这个回退比把 snapshot physical force 直接解释为 Boresch stiffness 更接近正确问题，但仍不应自动成为未经 sensitivity 的 production constant。

当前 anchor screening 与 stiffness estimation 还使用了部分相同数据。筛选先偏好低 RMSF atoms，形式上包括

\[
\texttt{rigid\_mask}=\mathrm{RMSF}<\mathrm{RMSF}_{\mathrm{cutoff}},
\]

随后又用同一轨迹的较小 \(\operatorname{Var}(q_i)\) 推出较大的 \(k_i^{\mathrm{fluc}}\)。这会产生 selection bias：算法先挑“这段轨迹里最不动”的 anchor，再用同一证据宣布它们需要更硬的弹簧。`auto` 的 candidate signal score 若也来自同一轨迹，会进一步形成筛选、估计和排序的同数据复用。

最低限度的数据隔离应为：前 50% trajectory 只选 anchors，后 50% 独立测量 covariance、branch stability 与 pose behavior；更稳健的方案是 blocked cross-validation，并在最终 holdout 上报告 anchor diagnostics。正式 sensitivity 则需在预注册的一组 diagonal restraint strengths 上独立比较 attachment work、single-basin behavior、pose stability 与最终 cycle。

\(\mathbf q_0\) 与 \(\mathbf K\) 的数据来源也需要统一。当前 `runabfe.py` 对 `auto`/`orb_simple` 路径可由多帧统计得到 \(k\)，随后却把 `equilibrium_values` 改成最后一帧几何；`simple`/`fluctuation` 已明确保留 ensemble mean。前一种做法造成“多帧 stiffness、单帧中心”的不一致，并让偶然末帧决定人工势阱中心。

更自然的中心为

\[
r_0=\langle r\rangle,
\qquad
\theta_{A0}=\langle\theta_A\rangle,
\qquad
\theta_{B0}=\langle\theta_B\rangle,
\]

以及 circular torsion mean

\[
\phi_0=\operatorname{atan2}
\left(\langle\sin\phi\rangle,\langle\cos\phi\rangle\right).
\]

若必须提交真实构象，应从轨迹中选择最接近六维 mean geometry 的一帧，而不是直接使用最后一帧。reference geometry 一经提交，就必须随 Hamiltonian fingerprint 冻结，resume 不能重锚。

按这个逻辑，ORB/MACE/geometric information 更适合输出 `anchor_quality_score`，评价局部刚性、六维 covariance condition number、Jacobian/Gram condition number、angle singularity margin、torsion branch stability、ligand/receptor anchor spacing，以及对环境扰动的敏感性。对候选 anchor 定义 circularly unwrapped coordinate vector 后，可估计

\[
\mathbf C=
\left\langle
(\mathbf q-\bar{\mathbf q})(\mathbf q-\bar{\mathbf q})^T
\right\rangle.
\]

若 \(\kappa(\mathbf C)\gg1\)，说明六维坐标高度退化、尺度失衡或强耦合，应优先更换 anchor，而不是用更大的 diagonal spring 掩盖问题。\(\kappa(\mathbf J\mathbf J^T)\)、\(\sin\theta_A\)、\(\sin\theta_B\) 和 torsion branch crossings 应作为独立 bad-anchor gates。

最终 production restraint 仍使用人为、可审计的 diagonal \(\mathbf K_R\)。trajectory 可提供初始尺度

\[
k_i^{\mathrm{raw}}=\frac{k_BT}{\sigma_i^2},
\qquad
k_i^R=\alpha_i k_i^{\mathrm{raw}},
\]

但 \(\alpha_i\) 必须明确定义为 protocol-level `restraint_strength_factor`，并配合预注册上下限、attachment overlap、single-basin 与 sensitivity gates；上下限是协议边界，不应再充当错误 estimator 爆炸后的静默补救。最终 provenance 必须分别保存 anchor selection data、validation data、\(\bar{\mathbf q}\)、\(\mathbf C\)、condition numbers、\(\alpha_i\)、未裁剪/应用后的 \(k_i^R\) 与 committed reference frame。

因此，`orb_simple`/`auto` 当前的 force-to-\(k_{\mathrm{Boresch}}\) 输出只能保留为实验性历史分支，不能作为 production 物理依据。ORB/MACE 在此处最有价值的角色是 **anchor selector、geometry-quality estimator 与 bad-anchor detector**；Boresch 的人工 \(k\) 则由真实预平衡 ensemble、采样稳定性、解析近似和 sensitivity protocol 决定。

**English**

The Boresch \(K_{\mathrm{restraint}}\) is not a chemical-bond stiffness that ORB/MACE needs to predict. The restraint is an artificial orientational Hamiltonian introduced to keep the ligand from drifting or flipping during decoupling, maintain a chosen relative orientation, and make the restrained ensemble compatible with the analytical release treatment. ORB/MACE forces or Hessians instead describe the local unrestrained physical surface, \(K_{\mathrm{physical}}\). In general,

\[
K_{\mathrm{restraint}}\ne K_{\mathrm{physical}}.
\]

A naturally soft torsion may still need a stronger artificial restraint to prevent ligand flipping, while a physically stiff direction does not justify an equally stiff restraint. The production constants should therefore be chosen from restrained-sampling behavior, overlap, single-basin stability, and sensitivity of the analytical correction rather than copied from an ML curvature.

Even if the goal were to estimate the physical local stiffness, the correct object would be a coupled matrix. For

\[
\mathbf q=(r,\theta_A,\theta_B,\phi_A,\phi_B,\phi_C)^T,
\]

the local quadratic model is

\[
U(\mathbf q)\approx U(\mathbf q_0)
+\frac{1}{2}(\mathbf q-\mathbf q_0)^T
\mathbf K_{\mathrm{physical}}
(\mathbf q-\mathbf q_0).
\]

The off-diagonal terms describe distance–angle, angle–torsion, and torsion–torsion coupling. The present scalar estimate

\[
k_i=-\frac{\operatorname{Cov}(F_i,q_i)}{\operatorname{Var}(q_i)}
\]

performs six independent regressions and discards those couplings. With a metric-correct generalized force,

\[
\mathbf Q=\mathbf G^{-1}\mathbf J\mathbf F,
\]

the coherent physical diagnostic would instead fit

\[
\Delta\mathbf Q=-\mathbf K_{\mathrm{physical}}\Delta\mathbf q.
\]

The full matrix is useful for diagnosing anchor degeneracy and coupling. The production restraint can still remain diagonal because the current analytical release is constructed from independent local terms; its diagonal \(\mathbf K_R\) is a protocol choice rather than a direct estimate of \(\mathbf K_{\mathrm{physical}}\).

An instantaneous microscopic generalized force is also not identical to the PMF gradient. For curvilinear distance, angle, and dihedral coordinates, the effective free energy

\[
A(\mathbf q)=-k_BT\ln P(\mathbf q)
\]

contains coordinate-metric, Jacobian/Fixman, entropic, and integrated-environment contributions. The relevant ensemble quantity is the curvature of \(A(\mathbf q)\), not a simple projection of one snapshot's Cartesian Hessian. This is why

\[
k_i^{\mathrm{fluc}}\approx
\frac{k_BT}{\operatorname{Var}(q_i)}
\]

is conceptually closer to an effective free-energy curvature. It still assumes adequate sampling and a locally harmonic single basin, so it is a diagnostic and starting scale rather than an automatically qualified production constant.

Selection and validation also need separate data. The current workflow first favors low-RMSF anchors and then uses small variances from the same trajectory to infer large fluctuation stiffnesses. This reuses the same evidence and biases the result toward harder directions. A simple remedy is to select anchors from the first half of the trajectory and evaluate covariance, torsion-branch stability, and pose behavior on the second half. Blocked cross-validation with a final holdout is preferable when enough trajectory is available.

The reference geometry should be estimated consistently from the ensemble used for the restraint analysis. Distance and angle centers use ordinary means,

\[
r_0=\langle r\rangle,
\qquad
\theta_{A0}=\langle\theta_A\rangle,
\qquad
\theta_{B0}=\langle\theta_B\rangle,
\]

while torsions use circular means,

\[
\phi_0=\operatorname{atan2}
\left(\langle\sin\phi\rangle,\langle\cos\phi\rangle\right).
\]

If a real configuration must be committed, the frame closest to the six-dimensional mean geometry is more defensible than the final frame. Once committed, the reference geometry remains fixed under the Hamiltonian fingerprint.

Under the revised design, ORB/MACE and geometric information are used to score anchor quality. For a circularly unwrapped coordinate vector,

\[
\mathbf C=
\left\langle
(\mathbf q-\bar{\mathbf q})(\mathbf q-\bar{\mathbf q})^T
\right\rangle.
\]

Large \(\kappa(\mathbf C)\) or \(\kappa(\mathbf J\mathbf J^T)\), small \(\sin\theta_A\) or \(\sin\theta_B\), unstable torsion branches, poor anchor spacing, or strong environmental sensitivity identify a problematic anchor set. In that case, replacing the anchors is more appropriate than increasing the diagonal springs.

The equilibrated trajectory may provide an initial restraint scale,

\[
k_i^{\mathrm{raw}}=\frac{k_BT}{\sigma_i^2},
\qquad
k_i^R=\alpha_i k_i^{\mathrm{raw}},
\]

where \(\alpha_i\) is an explicit protocol-level `restraint_strength_factor`. Preregistered parameter ranges define the restraint family to be tested; they are not post hoc clamps for a failed estimator. The record should preserve the selection and validation data, \(\bar{\mathbf q}\), \(\mathbf C\), condition numbers, \(\alpha_i\), raw and applied constants, and the committed reference frame.

The current force-to-\(k_{\mathrm{Boresch}}\) output of `orb_simple` and `auto` therefore remains an experimental legacy route. The more defensible role of ORB/MACE is anchor selection, geometry-quality assessment, and bad-anchor detection. The artificial Boresch constants are chosen from the equilibrated ensemble, restrained-sampling behavior, analytical assumptions, and a sensitivity study.

#### 3.1.3 Boresch 执行与分析子模块 / Boresch Execution and Analysis Submodules

**中文**

| 子模块 | 作用 | 主输出 | 资格边界 |
|---|---|---|---|
| anchor validation | 检查 receptor/ligand atom identity、PBC geometry 与退化角 | 已验证的六坐标定义 | 结构或 atom identity 改变必须使 cache 失效 |
| equilibrium submission | 冻结已提交的 reference geometry | `boresch_equilibrium_committed.json` | resume 不得按最后一帧自动重新锚定 |
| sequential attachment | A′→A 逐步打开 restraint | adjacent-BAR `ΔG_attach` | 只用于 complex leg，负 attachment 直接拒绝 |
| FD-TI crosscheck | 用重加权 finite difference 检查 attachment derivative | TI consistency diagnostic | 是 crosscheck，不替代 BAR 主值 |
| MBAR diagnostic | 检查多状态一致性与 overlap | diagnostic free-energy matrix | 不是 Stage 0 主 estimator |
| Boresch rebalance | 在固定 restraint 下重新平衡 complex | reusable starting state | 默认 `50,000` steps；生产不建议跳过 |
| analytic release | 将受约束的 decoupled ligand 释放到 1 M 标准态 | `ΔG_release` | 只加入一次；不得在 final cycle 再次手工扣除 |
| harmonicity/sensitivity | 检查 500-frame fluctuation、clipping 与 alternative anchors | qualification report | harmonicity pass 不等于 sensitivity closure |
| Stage 0 cache | 按 Boresch/Hamiltonian/topology/box/protocol key 复用 attachment | `stage0_attachment.json` | mismatch 必须重算 |

**English**

| Submodule | Function | Primary output | Qualification boundary |
|---|---|---|---|
| anchor validation | Checks receptor/ligand atom identity, PBC geometry, and degenerate angles | Validated six-coordinate definition | A structural or atom-identity change must invalidate the cache |
| equilibrium submission | Freezes the committed reference geometry | `boresch_equilibrium_committed.json` | Resume must not automatically re-anchor from the last frame |
| sequential attachment | Gradually opens the restraint from A′→A | Adjacent-BAR `ΔG_attach` | Complex leg only; negative attachment is rejected |
| FD-TI crosscheck | Checks the attachment derivative using reweighted finite differences | TI consistency diagnostic | A crosscheck; it does not replace the BAR primary value |
| MBAR diagnostic | Checks multistate consistency and overlap | Diagnostic free-energy matrix | Not the Stage 0 primary estimator |
| Boresch rebalance | Re-equilibrates the complex with the restraint fixed | Reusable starting state | Default `50,000` steps; skipping production rebalance is discouraged |
| analytic release | Releases the restrained decoupled ligand to the 1 M standard state | `ΔG_release` | Add only once; do not subtract it manually again in the final cycle |
| harmonicity/sensitivity | Checks 500-frame fluctuations, clipping, and alternative anchors | Qualification report | A harmonicity pass does not close sensitivity |
| Stage 0 cache | Reuses attachment by the Boresch/Hamiltonian/topology/box/protocol key | `stage0_attachment.json` | A mismatch requires recomputation |

#### 3.1.4 500-frame 几何诊断与当前边界 / The 500-Frame Geometry Diagnostic and Present Boundary

**中文**

| 指标 | 数值 |
|---|---:|
| 分析帧数 | `500` |
| 锚点候选数 | `562` |
| 谐性标志 | `true` |
| 非高斯计数 | `0` |
| 被裁剪的径向常数数目 | `1` |
| 原始径向常数 | approximately `7355.9 kJ mol⁻¹ nm⁻²` |
| 应用的径向常数 | `2000 kJ mol⁻¹ nm⁻²` |

**English**

| Metric | Value |
|---|---:|
| Analyzed frames | `500` |
| Anchor candidates | `562` |
| Harmonicity flag | `true` |
| Non-Gaussian count | `0` |
| Clipped radial constants | `1` |
| Raw radial constant | approximately `7355.9 kJ mol⁻¹ nm⁻²` |
| Applied radial constant | `2000 kJ mol⁻¹ nm⁻²` |

这组结果说明局部几何近似 harmonic，但也说明至少一个 restraint parameter 到达人为上限。正式 sensitivity 应比较 unclipped/alternative anchors、pose stability、restraint work 和最终 ΔG 的变化，而不能只引用 harmonicity flag。

These results indicate approximately harmonic local geometry, but also show that at least one restraint parameter reached an imposed ceiling. Formal sensitivity must compare unclipped or alternative anchors, pose stability, restraint work, and final ΔG changes rather than citing only the harmonicity flag.

### 3.2 Stage 1：PME decharging 与静态中性 handoff / Stage 1: PME Decharging and the Static-Neutral Handoff

#### 3.2.1 物理步骤合同 / Physical Step Contract

**中文**

| 维度 | Stage 1 内容 |
|---|---|
| 目标 | 将 ligand–environment electrostatics 从全耦合变为零，同时保留 vdW excluded volume |
| 输入 | Stage 0 restrained complex 或独立 solvent leg；`λ_coul=1, λ_vdW=1` |
| Hamiltonian 变化 | `λ_coul: 1→0`；`λ_vdW=1` 不变；保留 PME reciprocal/self/PBC 合同 |
| 采样与估计 | adjacent BAR 为主，FD-TI crosscheck，MBAR diagnostic |
| 输出 | leg-specific decharging free energy、static-neutral OpenMM System、estimator provenance |
| 核心风险 | 删除电荷替代 PME 路径、净电荷变化、旧 estimator 字段、charged-ion identity 错配 |
| 当前状态 | neutral Atenolol branch 可运行；真实 charged-ligand full cycle 未验证 |

**English**

| Dimension | Stage 1 content |
|---|---|
| Objective | Change ligand–environment electrostatics from fully coupled to zero while retaining vdW excluded volume |
| Input | Stage 0 restrained complex or independent solvent leg; `λ_coul=1, λ_vdW=1` |
| Hamiltonian change | `λ_coul: 1→0`; `λ_vdW=1` unchanged; preserve the PME reciprocal/self/PBC contract |
| Sampling and estimation | Adjacent BAR primary, FD-TI crosscheck, MBAR diagnostic |
| Output | Leg-specific decharging free energy, static-neutral OpenMM System, and estimator provenance |
| Main risk | Replacing the PME path by deleting charges, net-charge changes, stale estimator fields, and charged-ion identity mismatch |
| Current status | Neutral Atenolol branch is operational; a real charged-ligand full cycle is not validated |

**中文说明**

Stage 1 将 ligand electrostatics 从 `λ_coul=1` 变为 `0`，同时保持 `λ_vdW=1`。这条路径保留 PME reciprocal-space、自能、PBC 和总电荷处理，因此不能用简单删除电荷来代替。

当前主估计是相邻状态 BAR；finite-difference TI 用于检查路径一致性，MBAR 用于诊断多状态 overlap。历史 artifact 中的 Stage 1 字段可能来自旧 estimator，只有字段存在并不能说明它符合 current estimator v2。

**English notes**

Stage 1 changes ligand electrostatics from `λ_coul=1` to `0` while retaining `λ_vdW=1`. The path preserves PME reciprocal-space, self-energy, periodic-boundary, and total-charge treatment; simply deleting charges is not equivalent.

Adjacent-state BAR is the primary estimator. Finite-difference TI checks path consistency, and MBAR diagnoses multistate overlap. Historical Stage-1 fields may come from older estimators, so their presence alone does not establish compliance with current estimator v2.

#### 3.2.1.1 PME-preserving λ Hamiltonian 与 Stage 1 estimators / PME-Preserving λ Hamiltonian and Stage-1 Estimators

**中文**

OpenMM parameter offset 的实际电荷路径为

\[
q_i(\lambda)=q_{i,\mathrm{base}}+\lambda q_{i,\mathrm{scale}}.
\]

对中性配体，\(q_{i,\mathrm{base}}=0\)、\(q_{i,\mathrm{scale}}=q_i\)，即

\[
q_i(\lambda)=\lambda q_i,
\qquad
(\lambda_{\mathrm{coul}},\lambda_{\mathrm{vdW}}):(1,1)\to(0,1).
\]

ligand–ligand Coulomb interactions 被显式冻结，只有 ligand–environment electrostatics 随 λ 变化。因此这是 ligand–environment **decoupling**，不是把配体内部静电也一并 annihilate。每个 λ 的 reduced potential 由完整 PME Hamiltonian 重算：

\[
u_k(x_n)=\beta U_k^{\mathrm{PME}}(x_n).
\]

原生 `NonbondedForce` 会在每个 λ 用缩放后的电荷重新计算 real-space、reciprocal-space、PBC 和 Ewald self contribution。因此当前生产路径不再额外加入旧版 \(+C\lambda^2\)；该项已经包含在 \(U_k^{\mathrm{PME}}\) 中，再加一次会重复计数。

对于带电配体，生产设计采用同号、保留 LJ 的 neutral co-ion 做 charge transfer：

\[
q_{\mathrm{lig},i}(\lambda)=\lambda q_i,
\qquad
q_{\mathrm{co},j}(\lambda)=(1-\lambda)s_j.
\]

总电荷合同不是只在端点检查，而是要求

\[
Q_{\mathrm{box}}(\lambda)
=\sum_iq_{i,\mathrm{base}}+\lambda\sum_iq_{i,\mathrm{scale}},
\qquad
\sum_iq_{i,\mathrm{scale}}=0,
\]

从而 \(Q_{\mathrm{box}}(\lambda)\) 对所有中间 λ 恒定。co-ion 的 λ-independent flat-bottom restraint 为

\[
U_{\mathrm{co}}=
\frac{1}{2}k_{\mathrm{co}}
\left[\max\left(0,
\left|\mathbf r_{\mathrm{ion}}-\mathbf r_{\mathrm{anchor}}-\mathbf d_0\right|-r_{\mathrm{flat}}
\right)\right]^2.
\]

Stage 1 主值为相邻 BAR 链；全帧/去相关 MBAR 是诊断。非均匀 λ 网格上的 finite-difference TI 使用真实 λ 值：

\[
\left\langle\frac{\partial U}{\partial\lambda}\right\rangle_k
\approx
\left\langle
\frac{U_{k+1}-U_{k-1}}{\lambda_{k+1}-\lambda_{k-1}}
\right\rangle_k,
\qquad
\Delta G_{\mathrm{FD-TI}}
=\int\left\langle\frac{\partial U}{\partial\lambda}\right\rangle_\lambda d\lambda.
\]

这里的 λ 必须使用实际的非均匀网格值，不能用状态编号代替。

**English**

The OpenMM parameter-offset path is

\[
q_i(\lambda)=q_{i,\mathrm{base}}+\lambda q_{i,\mathrm{scale}}.
\]

For a neutral ligand, \(q_{i,\mathrm{base}}=0\) and \(q_{i,\mathrm{scale}}=q_i\), so

\[
q_i(\lambda)=\lambda q_i,
\qquad
(\lambda_{\mathrm{coul}},\lambda_{\mathrm{vdW}}):(1,1)\to(0,1).
\]

Ligand–ligand Coulomb interactions remain fixed; only ligand–environment electrostatics vary with λ. Stage 1 is therefore a ligand–environment decoupling path rather than a complete annihilation of intraligand electrostatics. Each reduced potential is recomputed from the full PME Hamiltonian:

\[
u_k(x_n)=\beta U_k^{\mathrm{PME}}(x_n).
\]

The native `NonbondedForce` evaluates real-space, reciprocal-space, periodic, and Ewald self contributions with the scaled charges at every λ. The current production path does not add the historical manual \(+C\lambda^2\) correction, because that contribution is already present in \(U_k^{\mathrm{PME}}\).

For a charged ligand, charge is transferred to a same-sign neutral co-ion that retains its LJ interactions:

\[
q_{\mathrm{lig},i}(\lambda)=\lambda q_i,
\qquad
q_{\mathrm{co},j}(\lambda)=(1-\lambda)s_j.
\]

The total-charge condition is enforced algebraically for the entire path:

\[
Q_{\mathrm{box}}(\lambda)
=\sum_iq_{i,\mathrm{base}}+\lambda\sum_iq_{i,\mathrm{scale}},
\qquad
\sum_iq_{i,\mathrm{scale}}=0.
\]

The co-ion is kept near its selected region by a λ-independent flat-bottom restraint:

\[
U_{\mathrm{co}}=
\frac{1}{2}k_{\mathrm{co}}
\left[\max\left(0,
\left|\mathbf r_{\mathrm{ion}}-\mathbf r_{\mathrm{anchor}}-\mathbf d_0\right|-r_{\mathrm{flat}}
\right)\right]^2.
\]

Adjacent-state BAR provides the Stage-1 primary estimate. Full-frame and decorrelated MBAR are diagnostics. The finite-difference TI check uses the actual nonuniform λ coordinates:

\[
\left\langle\frac{\partial U}{\partial\lambda}\right\rangle_k
\approx
\left\langle
\frac{U_{k+1}-U_{k-1}}{\lambda_{k+1}-\lambda_{k-1}}
\right\rangle_k,
\qquad
\Delta G_{\mathrm{FD-TI}}
=\int\left\langle\frac{\partial U}{\partial\lambda}\right\rangle_\lambda d\lambda.
\]

State indices are not substitutes for the actual λ values.

#### 3.2.2 Stage 1 electrostatics 功能分支 / Stage-1 Electrostatics Functional Branches

**中文**

| 功能分支 | 原理与模块作用 | Estimator/修正 | 状态与分析 |
|---|---|---|---|
| `pme` offsets | 使用 `NonbondedForce` parameter offsets 缩放 ligand charge，保留 PME real/reciprocal/self/PBC | adjacent BAR primary；FD-TI crosscheck；MBAR diagnostic | **当前推荐**；neutral baseline 与 charged co-ion route 都以真实 PME 为权威 |
| `shadow_ibs` | 使用 real-space shadow-Coulomb CV 构造 mixture，并通过 real-PME→shadow bridge 连接 | bridge free energy + shadow IBS/TMBAR | **实验性、neutral-only**；不能承担真实 PME，也不能用于 charged production claim |
| static-neutral baker | 将 `λ_coul=0` 的 parameterized NB 烘焙为固定 System | energy/force seam validation | **主 handoff 模块**；Stage 2 禁止保留旧 global-parameter identity |
| Stage-1 BAR analyzer | 相邻 states 的主 estimator | covariance-aware BAR | **主分析模块**；必须报告 estimator provenance |
| reweighted FD-TI | 检查 `∂U/∂λ` 与 BAR 路径一致性 | finite-difference consistency gate | **必须 crosscheck**；不是独立可替换主值 |
| Stage-1 MBAR | 全状态诊断 | MBAR matrix/overlap | **diagnostic**；不得与 Stage-2 global TMBAR 混称 |

**English**

| Functional branch | Principle and module role | Estimator/correction | Status and analysis |
|---|---|---|---|
| `pme` offsets | Scale ligand charge with `NonbondedForce` parameter offsets while preserving PME real/reciprocal/self/PBC | Adjacent BAR primary; FD-TI crosscheck; MBAR diagnostic | **Currently recommended**; real PME is authoritative for both the neutral baseline and charged co-ion route |
| `shadow_ibs` | Constructs a mixture using a real-space shadow-Coulomb CV and connects it through a real-PME→shadow bridge | Bridge free energy + shadow IBS/TMBAR | **Experimental, neutral-only**; cannot represent real PME or support a charged-production claim |
| static-neutral baker | Bakes the `λ_coul=0` parameterized NB into a fixed System | Energy/force seam validation | **Primary handoff module**; Stage 2 must not retain the old global-parameter identity |
| Stage-1 BAR analyzer | Primary estimator for adjacent states | Covariance-aware BAR | **Primary analysis module**; estimator provenance must be reported |
| reweighted FD-TI | Checks consistency between `∂U/∂λ` and the BAR path | Finite-difference consistency gate | **Required crosscheck**; not an independently interchangeable primary value |
| Stage-1 MBAR | Full-state diagnostic | MBAR matrix/overlap | **Diagnostic**; must not be conflated with Stage-2 global TMBAR |

#### 3.2.3 Stage 1→2 handoff 的不可变条件 / Immutable Conditions at the Stage 1→2 Handoff

**中文**

Stage 1 在保持 `λ_vdW=1` 的同时把 `λ_coul` 从 `1` 变到 `0`，因此移除的是 ligand–environment electrostatics，短程 excluded volume 仍然存在。进入 Stage 2 前，带 global parameter 的 `λ_coul=0` `NonbondedForce` 会被固化成静态中性 System。这样 Stage 2 只面对一个明确的固定 Hamiltonian，不会同时混用 parameterized PME 与 baked system。

当前 soluble Atenolol 使用 neutral branch。带电配体还需要保持总电荷与 co-alchemical-ion identity；相关的 charge-transfer、Rocklin/APBS 和 co-annihilation 扩展在第 5 章单独讨论。工程 seam 的 energy/force agreement 只证明 handoff 实现一致，不等于已经完成带电配体 ABFE cycle。

**English**

Stage 1 changes `λ_coul` from `1` to `0` while keeping `λ_vdW=1`. It removes ligand–environment electrostatics but retains short-range excluded volume. Before Stage 2, the global-parameterized `λ_coul=0` `NonbondedForce` is baked into a fixed static-neutral System. Stage 2 then sees one explicit Hamiltonian rather than a mixture of parameterized PME and baked-system identities.

The current soluble Atenolol route uses the neutral branch. A charged-ligand route must additionally preserve total charge and co-alchemical-ion identity; charge transfer, Rocklin/APBS, and co-annihilation extensions are discussed separately in Section 5. Energy/force agreement across the engineering seam validates the handoff implementation but does not constitute a completed charged-ligand ABFE cycle.

### 3.3 Stage 2：ACE-softcore、Path v21 与 IBS v29 / Stage 2: ACE Softcore, Path v21, and IBS v29

#### 3.3.1 物理步骤合同 / Physical Step Contract

**中文**

| 维度 | Stage 2 内容 |
|---|---|
| 目标 | 在静态中性体系中将 ligand–environment vdW 从 1 平滑变为 0 |
| 输入 | Stage 1 static-neutral handoff、v21 path、window partition、IBS protocol fingerprint |
| Hamiltonian 变化 | ACE softcore `λ_vdW: 1→0`；IBS/WCA 只按各自 ledger 身份加入 |
| 采样 | local IBS mixture；learning→freeze→fixed-bias validation→immutable production |
| 输出 | 每个 window 的 target/bias/base histories、frozen `(f_k)` hash、checkpoint 和 provenance |
| 核心风险 | endpoint singularity、低 overlap、错误 mixture identity、production 中更新 `(f_k)`、重复 edge 记账 |
| 当前状态 | v21/v29 是当前 production baseline；独立重复和总不确定度仍未闭合 |

**English**

| Dimension | Stage 2 content |
|---|---|
| Objective | Smoothly change ligand–environment vdW from 1 to 0 in the static-neutral system |
| Input | Stage 1 static-neutral handoff, v21 path, window partition, and IBS protocol fingerprint |
| Hamiltonian change | ACE softcore `λ_vdW: 1→0`; add IBS/WCA only under their respective ledger identities |
| Sampling | Local IBS mixture; `learning→freeze→fixed-bias validation→immutable production` |
| Output | Per-window target/bias/base histories, frozen `(f_k)` hash, checkpoint, and provenance |
| Main risk | Endpoint singularity, low overlap, incorrect mixture identity, updating `(f_k)` during production, and duplicate edge accounting |
| Current status | v21/v29 are the current production baseline; independent repeats and total uncertainty remain unresolved |

**中文说明**

Stage 2 从 Stage 1 的静态中性体系出发，把 ligand–environment van der Waals coupling 从 `1` 变到 `0`。当前参数是 `α_LJ=0.5`、`α_Coul=0.2`、`dimensionless_sigma_scaled_v2` 和 `1.0 nm` ACE cutoff，默认不使用 switching。

这一阶段的轨迹由 IBS mixture 产生，而不是来自单一 target state。因此单态 BAR/TI 不能直接作为完整主值；分析需要同时重建 actual mixture、全部 target states、冻结的 \(f_k\)、WCA sampling bias 和 LRC。

**English notes**

Stage 2 starts from the static-neutral system produced by Stage 1 and changes ligand–environment van der Waals coupling from `1` to `0`. The current parameters are `α_LJ=0.5`, `α_Coul=0.2`, `dimensionless_sigma_scaled_v2`, and a `1.0 nm` ACE cutoff, with switching disabled by default.

The trajectory is generated by an IBS mixture rather than a single target state. A single-state BAR/TI analysis therefore cannot provide the complete primary estimate; the analysis jointly reconstructs the actual mixture, all target states, frozen \(f_k\), WCA sampling bias, and LRC.

#### 3.3.1.1 项目 ACE-like softcore Hamiltonian / Project-Specific ACE-Like Softcore Hamiltonian

**中文**

混合规则为

\[
\sigma_{ij}=\frac{\sigma_i+\sigma_j}{2},
\qquad
\epsilon_{ij}=\sqrt{\epsilon_i\epsilon_j}.
\]

定义 sigma-scaled 软核分母

\[
D_{6,ij}(r,\lambda_v)=
\max\!\left[r^6+\alpha_{\mathrm{LJ}}\sigma_{ij}^6
(1-\lambda_v)^{m_{\mathrm{LJ}}},10^{-6}\right],
\]

\[
D_{2,ij}(r,\lambda_c)=
\max\!\left[r^2+\alpha_{\mathrm C}\sigma_{ij}^2
(1-\lambda_c)^{m_{\mathrm C}},10^{-6}\right].
\]

代码实现的 ACE-like pair potential 为

\[
\begin{aligned}
U_{ij}^{\mathrm{ACE}}=&\lambda_v^{n_{\mathrm{LJ}}}4\epsilon_{ij}
\left(\frac{\sigma_{ij}^{12}}{D_{6,ij}^2}
-\frac{\sigma_{ij}^{6}}{D_{6,ij}}\right)
+\lambda_c^{n_{\mathrm C}}
\frac{138.935456\,q_iq_j}{\sqrt{D_{2,ij}}}.
\end{aligned}
\]

默认参数为

\[
\alpha_{\mathrm{LJ}}=0.5,
\quad(m_{\mathrm{LJ}},n_{\mathrm{LJ}})=(2,2),
\]

\[
\alpha_{\mathrm C}=
\begin{cases}
0.3,&N_{\mathrm{lig}}≤50,\\
0.2,&N_{\mathrm{lig}}>50,
\end{cases}
\qquad(m_{\mathrm C},n_{\mathrm C})=(1,1).
\]

Stage 2 本身强制 \(\lambda_c\equiv0\)，所以生产阶段实际只使用上式的 vdW 部分；Coulomb 已在 Stage 1 处理。\(\lambda_v=1\) 精确恢复普通 12–6 LJ，\(\lambda_v=0\) 则使 ligand–environment ACE interaction 严格为零，同时 ligand intramolecular interactions 由独立项保留。

这里采用的是项目自己的 `dimensionless_sigma_scaled_v2` 约定。\(\alpha_{\mathrm{LJ}}\) 与 \(\alpha_{\mathrm C}\) 都是无量纲参数，分别与 pair-specific \(\sigma_{ij}^6\) 和 \(\sigma_{ij}^2\) 相乘。这与把 α 当作绝对 nm\(^6\)/nm\(^2\) 的旧 softcore 写法并不等价；ACE dual-λ 的 λ powers 也不同于传统 Beutler control。因此这里称为 project-specific ACE-like Hamiltonian，而不是原文公式的逐字复现。

目标态、采样态和公共能量必须分开：

\[
U_k^{\mathrm{target}}(x)=U_{\mathrm{base}}(x)+U_{\mathrm{ACE},k}(x)+U_{\mathrm{LRC},k}(x),
\]

\[
U_s(x)=U_{\mathrm{base}}(x)+U_{\mathrm{bias}}(x)+U_{\mathrm{WCA}}(x).
\]

\(U_{\mathrm{bias}}\) 与 \(U_{\mathrm{WCA}}\) 只改变采样分布，不属于物理 target Hamiltonian。分析时它们进入 actual-mixture row，再通过 reweighting 去除。

**English**

The mixing rules are

\[
\sigma_{ij}=\frac{\sigma_i+\sigma_j}{2},
\qquad
\epsilon_{ij}=\sqrt{\epsilon_i\epsilon_j}.
\]

The sigma-scaled softcore denominators are

\[
D_{6,ij}(r,\lambda_v)=
\max\!\left[r^6+\alpha_{\mathrm{LJ}}\sigma_{ij}^6
(1-\lambda_v)^{m_{\mathrm{LJ}}},10^{-6}\right],
\]

\[
D_{2,ij}(r,\lambda_c)=
\max\!\left[r^2+\alpha_{\mathrm C}\sigma_{ij}^2
(1-\lambda_c)^{m_{\mathrm C}},10^{-6}\right].
\]

The implemented ACE-like pair potential is

\[
\begin{aligned}
U_{ij}^{\mathrm{ACE}}=&\lambda_v^{n_{\mathrm{LJ}}}4\epsilon_{ij}
\left(\frac{\sigma_{ij}^{12}}{D_{6,ij}^2}
-\frac{\sigma_{ij}^{6}}{D_{6,ij}}\right)
+\lambda_c^{n_{\mathrm C}}
\frac{138.935456\,q_iq_j}{\sqrt{D_{2,ij}}}.
\end{aligned}
\]

The default parameters are

\[
\alpha_{\mathrm{LJ}}=0.5,
\quad(m_{\mathrm{LJ}},n_{\mathrm{LJ}})=(2,2),
\]

\[
\alpha_{\mathrm C}=
\begin{cases}
0.3,&N_{\mathrm{lig}}\le50,\\
0.2,&N_{\mathrm{lig}}>50,
\end{cases}
\qquad(m_{\mathrm C},n_{\mathrm C})=(1,1).
\]

Stage 2 enforces \(\lambda_c\equiv0\), so only the van der Waals branch is active after PME decharging. At \(\lambda_v=1\), it reduces exactly to the ordinary 12–6 LJ potential. At \(\lambda_v=0\), the ligand–environment ACE interaction is zero, while ligand intramolecular interactions are retained separately.

This project uses the `dimensionless_sigma_scaled_v2` convention. The dimensionless parameters \(\alpha_{\mathrm{LJ}}\) and \(\alpha_{\mathrm C}\) multiply pair-specific \(\sigma_{ij}^6\) and \(\sigma_{ij}^2\), respectively. This differs from legacy softcore equations that treat alpha as an absolute nm\(^6\)/nm\(^2\) quantity, and the ACE dual-λ powers also differ from the traditional Beutler control. The equation is therefore described as a project-specific ACE-like Hamiltonian rather than a verbatim reproduction of the paper.

The target and sampled energies are kept separate:

\[
U_k^{\mathrm{target}}(x)=U_{\mathrm{base}}(x)+U_{\mathrm{ACE},k}(x)+U_{\mathrm{LRC},k}(x),
\]

\[
U_s(x)=U_{\mathrm{base}}(x)+U_{\mathrm{bias}}(x)+U_{\mathrm{WCA}}(x).
\]

The IBS and WCA terms define the sampled distribution but are not physical target terms. They enter the actual-mixture row and are removed by reweighting.

#### 3.3.2 Decoupling/path 功能分支 / Decoupling and Path-Construction Branches

**中文**

| 分支 | 模块介绍 | 优点 | 状态与限制 |
|---|---|---|---|
| `dual_lambda` | 先独立去 Coulomb，再独立去 vdW；Stage 1/2 可分别设置 states 与 estimator | 易定位 electrostatic/steric bottleneck，账本最清晰 | **当前推荐** |
| `single_lambda` | 一个 λ 同时控制 Coulomb/vdW，或沿固定 Beutler path 变化 | 简单，适合 traditional control | **对照路线**；难以独立优化两个 bottleneck |
| `2d_diagonal` | 在 `(λ_coul,λ_vdw)` 平面沿预设 diagonal 一维序列前进 | 可检查 sequential path dependence | **部分实现/比较**；证据不足以作默认路线 |
| `2d_geodesic` | pilot 估计二维 metric 后，以 monotonic geodesic/Dijkstra 选择低热力学长度路径 | 理论上绕开高方差区域 | **实验性**；pilot 成本、cache 与 provenance 更复杂 |

**English**

| Branch | Module description | Advantage | Status and limitation |
|---|---|---|---|
| `dual_lambda` | Decouple Coulomb independently first, then decouple vdW; Stage 1 and Stage 2 can use separate states and estimators | Makes electrostatic/steric bottlenecks easy to localize and keeps the ledger clearest | **Currently recommended** |
| `single_lambda` | One λ controls Coulomb/vdW simultaneously or follows a fixed Beutler path | Simple and suitable for traditional control | **Control route**; the two bottlenecks cannot be optimized independently |
| `2d_diagonal` | Advances along a preset one-dimensional diagonal sequence in the `(λ_coul,λ_vdw)` plane | Can test sequential path dependence | **Partially implemented/comparative**; insufficient evidence for the default route |
| `2d_geodesic` | Estimates a two-dimensional metric in a pilot and selects a low-thermodynamic-length path with monotonic geodesic/Dijkstra | Theoretically avoids high-variance regions | **Experimental**; pilot cost, cache, and provenance are more complex |

#### 3.3.3 Stage 2 potential 与保护模块 / Stage-2 Potential and Protection Modules

**中文**

| 模块 | Hamiltonian/function | 当前状态 | 分析 |
|---|---|---|---|
| ACE softcore | `dimensionless_sigma_scaled_v2` ligand–environment vdW | **production baseline** | endpoint regularization 与 LRC v3 已按当前主线设计 |
| Beutler softcore | traditional interaction-group softcore | legacy/control | 用于 conventional/single-λ regression，不自动共享 ACE 资格 |
| DEXP pair kernel | finite double-exponential vdW + Gaussian Coulomb contract | experimental | 改变 physical potential family；无已验证 DEXP LRC |
| static-neutral handoff | 强制 Stage 2 `λ_coul=0` fixed System | mainline seam | 避免在 Stage 2 混入 parameterized PME identity |
| WCA shield | 按窗口平均 `λ_vdW` 设置的短程 sampling guard | mainline IBS internal | 只属于 sampling bias，必须从 target reweight 掉 |
| path v21 optimizer | pilot metric→23-state schedule→local window partition | current artifact path | 17 是 pilot/base count，不等于最终 23-state path |
| λ refinement | 对困难区增加/调整 states | default disabled | production 期间禁止移动 λ；新 schedule 必须新 identity |

**English**

| Module | Hamiltonian/function | Current status | Analysis |
|---|---|---|---|
| ACE softcore | `dimensionless_sigma_scaled_v2` ligand–environment vdW | **Production baseline** | Endpoint regularization and LRC v3 are designed for the current mainline |
| Beutler softcore | Traditional interaction-group softcore | Legacy/control | Used for conventional/single-λ regression; does not automatically share ACE qualification |
| DEXP pair kernel | Finite double-exponential vdW + Gaussian Coulomb contract | Experimental | Changes the physical potential family; no validated DEXP LRC |
| static-neutral handoff | Forces a Stage 2 `λ_coul=0` fixed System | Mainline seam | Prevents the parameterized PME identity from entering Stage 2 |
| WCA shield | Short-range sampling guard set from the window-mean `λ_vdW` | Mainline IBS internal | Sampling bias only; it must be reweighted out of the target |
| path v21 optimizer | Pilot metric→23-state schedule→local window partition | Current artifact path | 17 is the pilot/base count, not the final 23-state path |
| λ refinement | Adds/adjusts states in difficult regions | Default disabled | λ must not move during production; a new schedule requires a new identity |

#### 3.3.4 Path v21：pilot metric、23-state path 与 window partition

**中文**

pilot metric 定义为

\[
g(\lambda)=\beta_T^2\operatorname{Var}\left(\frac{\partial U}{\partial\lambda}\right).
\]

它利用 generalized-force fluctuation 描述局部热力学难度，但不直接衡量 kinetic mixing、round trip 或不同重复之间的一致性。

17 个 uniform pilot states 用于估计 metric；23 个严格递减 states 是最终 production path。最终 path 再被划分为多个局部 IBS ensembles。相邻 windows 只共享一个 boundary state，每个 edge 只归属一个 window，以避免 cross-window double counting。

Seventeen uniform pilot states estimate the metric; twenty-three strictly decreasing states form the final production path. The final path is then partitioned into local IBS ensembles. Neighboring windows share only one boundary state, and each edge belongs to one window, preventing cross-window double counting.

困难 complex window 0 的局部 λ 值为：

```text
1.000000, 0.923529, 0.854304, 0.790614, 0.731876
```

这五个数只属于一个困难窗口，不是完整的 23-state path。

**English**

The pilot metric is

\[
g(\lambda)=\beta_T^2\operatorname{Var}\left(\frac{\partial U}{\partial\lambda}\right).
\]

It uses generalized-force fluctuations to describe local thermodynamic difficulty, but it does not directly measure kinetic mixing, round trips, or between-run reproducibility.

Seventeen uniform pilot states estimate the metric, and twenty-three strictly decreasing states define the production path. The path is divided into local IBS ensembles. Neighboring windows share one boundary state, and each edge belongs to only one window, avoiding cross-window double counting.

The difficult complex window 0 uses

```text
1.000000, 0.923529, 0.854304, 0.790614, 0.731876
```

These five values describe one local window rather than the complete 23-state path.

#### 3.3.5 IBS mixture 的数学对象 / Mathematical Object of the IBS Mixture

##### 3.3.5.1 论文中的 integrated ensemble / Integrated Ensemble in the Paper

**中文**

本小节的“论文公式”仅指 Lin et al., *J. Chem. Theory Comput.* (2026), DOI [`10.1021/acs.jctc.5c01240`](https://doi.org/10.1021/acs.jctc.5c01240) 的 Eq. 3–15；下一小节开始改按本项目源码重写，二者不视为同一个 estimator identity。

论文先写离散 integrated ensemble

\[
p_{\mathrm{int}}(R)\propto
\sum_{k=1}^{K}n_k\exp[-\beta U'_k(R)],
\]

其中 \(n_k\) 直接乘在第 \(k\) 个 Boltzmann factor 前；平衡目标是

\[
n_kQ_k=\text{constant},
\qquad
Q_k=\int dR\,e^{-\beta U'_k(R)}.
\]

因此论文路线需要估计各态配分函数 \(Q_k\)，再令 \(n_k\propto Q_k^{-1}\)。论文还讨论了随时间变化的 IBS distributions，并把多个历史批次组成 time-dependent MBAR mixture。这里的 \(n_k\) 是 integrated-ensemble 权重；它不是 MBAR 中表示样本数的 \(N_k\)。

**English**

In this subsection, “paper equations” refers to Eqs. 3–15 of Lin et al., *J. Chem. Theory Comput.* (2026), DOI [`10.1021/acs.jctc.5c01240`](https://doi.org/10.1021/acs.jctc.5c01240). The next subsection switches to equations reconstructed from the project source.

The paper's discrete integrated ensemble is

\[
p_{\mathrm{int}}(R)\propto
\sum_{k=1}^{K}n_k\exp[-\beta U'_k(R)].
\]

The coefficient \(n_k\) multiplies the Boltzmann factor directly, and the balancing condition is

\[
n_kQ_k=\text{constant},
\qquad
Q_k=\int dR\,e^{-\beta U'_k(R)}.
\]

The paper therefore estimates \(Q_k\) and updates \(n_k\propto Q_k^{-1}\). It also discusses a time-dependent MBAR mixture assembled from historical IBS distributions. Here \(n_k\) is an ensemble weight; it is not the MBAR sample count \(N_k\).

##### 3.3.5.2 本项目用 \(f_k\) 重参数化 / Project Reparameterization with \(f_k\)

**中文**

以下不是论文公式的逐字复制，而是依据本项目代码与能量约定重写。项目实现为

\[
V_{\mathrm{IBS}}(R;\mathbf f)
=-k_BT\ln\sum_{k=1}^{K}
\exp\{-\beta[U'_k(R)-f_k]\}.
\]

\[
e^{-\beta(U'_k-f_k)}=e^{\beta f_k}e^{-\beta U'_k}.
\]

因此代码的有效 integrated-ensemble 权重为

\[
n_k\propto e^{\beta f_k},
\]

而不是 \(n_k=f_k\)。若要求各态对 mixture 的积分贡献相同，则

\[
e^{\beta f_k}Q_k=\text{constant}
\quad\Longrightarrow\quad
f_k=-k_BT\ln Q_k+C=F_k+C,
\]

其中 \(F_k=-k_BT\ln Q_k\)，\(C\) 是任意公共 gauge。也就是说，最优 \(f_k\) 等于各态相对自由能（差一个公共常数），初始化可取 \(f_k=0\)，但 \(f_k\) 本身不是可观测物。

每个构象对状态 \(k\) 的 instantaneous responsibility 为

\[
p_k(R)=\frac{\exp[-\beta(U'_k-f_k)]}{\sum_j\exp[-\beta(U'_j-f_j)]}.
\]

代码使用 maximum-pivot log-sum-exp 计算这些指数，避免大能量差造成上溢或下溢。\(f_k\) 用来改善采样，不是最终物理观测量。

**English**

The following equation is reconstructed from the project implementation rather than copied from the paper:

\[
V_{\mathrm{IBS}}(R;\mathbf f)
=-k_BT\ln\sum_{k=1}^{K}
\exp\{-\beta[U'_k(R)-f_k]\}.
\]

Since

\[
e^{-\beta(U'_k-f_k)}=e^{\beta f_k}e^{-\beta U'_k},
\]

the effective ensemble weight is

\[
n_k\propto e^{\beta f_k},
\]

not \(n_k=f_k\). Equal integrated contributions require

\[
e^{\beta f_k}Q_k=\text{constant}
\quad\Longrightarrow\quad
f_k=-k_BT\ln Q_k+C=F_k+C.
\]

Thus \(f_k\) is a relative free-energy offset up to a common gauge. Initializing \(f_k=0\) only sets the starting point; it does not imply equal partition functions. The instantaneous responsibility of state \(k\) is

\[
p_k(R)=\frac{\exp[-\beta(U'_k-f_k)]}{\sum_j\exp[-\beta(U'_j-f_j)]}.
\]

The implementation evaluates the log-sum-exp with a maximum pivot for numerical stability. The offsets improve sampling and are not final physical observables.

##### 3.3.5.3 与论文更新路径的明确区别 / Explicit Difference from the Paper's Update Route

**中文**

论文的叙述是先由重加权估计 \(Q_k\)，再更新显式权重 \(n_k\)。本项目绕过“先单独输出每个 \(Q_k\)”这一步，直接在 energy-offset 参数化中更新 \(f_k\)；正式学习路径由 MBAR/TMBAR self-consistency 给出相对自由能，occupancy-based update 只作为回退机制。生产前冻结 \(\mathbf f\)，之后不再自适应。

本项目还有三条不能与论文原式混同的边界：第一，生产温度固定，不执行论文的联合 β-integration；第二，IBS 只用于 Stage 2 vdW softcore，Stage 1 PME decharging 走独立 BAR/FD-TI 路径；第三，最终主估计是逐 window 的 actual-mixture augmented MBAR，再按共享边界拼接，不是把全部学习历史放进论文式单一全局 time-mixture。

**English**

The paper first estimates \(Q_k\) and then updates explicit weights \(n_k\). This project instead updates the energy offsets \(f_k\) directly from MBAR/TMBAR self-consistency; occupancy feedback is retained only as a fallback. The offsets are frozen before production.

Three additional boundaries distinguish the project implementation from the paper: production is performed at fixed temperature without joint β integration; IBS is used only for the Stage-2 vdW softcore while Stage 1 follows an independent PME BAR/FD-TI route; and the final primary estimate uses per-window actual-mixture augmented MBAR followed by shared-boundary stitching, rather than a single global mixture over the entire adaptive history.

#### 3.3.6 IBS v29 状态机与冻结合同 / IBS v29 State Machine and Freeze Contract

IBS v29 按 `learning → freeze readiness → fixed-bias validation → immutable production` 顺序运行；production 中禁止继续更新 `(f_k)`。minibatch、damping、step cap 和版本差异列入附录 A。

IBS v29 follows `learning → freeze readiness → fixed-bias validation → immutable production`; `(f_k)` updates are forbidden during production. Low-level state-machine parameters are listed in Appendix A.

#### 3.3.7 Force groups 与三本账 / Force Groups and the Three Ledgers

**中文**

当前正文只区分三本账：physical target energy、actual sampling bias 和 common/base energy。WCA 与 learned residual 属于 sampling ledger；Boresch 属于 physical/base identity；LRC 进入 target ledger。具体 force-group 编号列入附录 A。

每帧追加必须是原子的：target、sampling bias、base、WCA、LRC、state/source、path/residual histories 要么一起写入，要么全部拒绝。NaN/Inf 不能被静默替换为 0；不同 history 长度不能错位。

**English**

The analysis keeps three separate ledgers: physical target energy, actual sampling bias, and common/base energy. WCA and a learned residual belong to the sampling ledger; Boresch belongs to the physical/base definition; and LRC enters the target ledger. Appendix A lists the force-group numbers.

Frame appends must be atomic: target, sampling bias, base, WCA, LRC, state/source, and path/residual histories are either all written or all rejected. NaN/Inf values must not be silently replaced with zero, and history lengths must not diverge.

### 3.4 统计估计器、covariance chain、LRC 与 rescue / Estimator, Covariance Chain, LRC, and Rescue

#### 3.4.1 分析合同 / Analysis Contract

**中文**

| 维度 | 分析内容 |
|---|---|
| 输入 | actual IBS mixture、全部 target rows、frozen bias、WCA/LRC histories、state/source identity |
| 去相关 | 使用最差 target reduced-energy sequence，同步抽取所有 rows |
| 主估计 | local TMBAR augmented matrix + shared-boundary covariance stitching |
| 修正 | Boresch release、soluble-system LRC v3、可选 APBS external scalar |
| 输出 | leg ΔG、endpoint covariance、coverage/ESS、跨窗口 covariance chain、final cycle ledger |
| 失败处理 | 同分布有限延长；否则独立 immutable bridge/rescue；禁止就地修改 Hamiltonian |

**English**

| Dimension | Analysis content |
|---|---|
| Input | Actual IBS mixture, all target rows, frozen bias, WCA/LRC histories, and state/source identity |
| Decorrelation | Use the worst target reduced-energy sequence and synchronously extract all rows |
| Primary estimate | Local TMBAR augmented matrix + shared-boundary covariance stitching |
| Corrections | Boresch release, soluble-system LRC v3, and optional APBS external scalar |
| Output | Leg ΔG, endpoint covariance, coverage/ESS, cross-window covariance chain, and final cycle ledger |
| Failure handling | Limited same-distribution extension; otherwise an independent immutable bridge/rescue; no in-place Hamiltonian modification |

#### 3.4.2 Sampling、estimator 与 diagnostics 模块 / Sampling, Estimator, and Diagnostic Modules

**中文**

| 功能模块 | 作用 | 主要产物 | 推荐等级与分析 |
|---|---|---|---|
| IBS log-sum-exp sampler | 一个 local ensemble 覆盖多个 λ target states | mixture trajectory, state probabilities, bias histories | **Stage 2 主线**；必须用真实 mixture row 分析 |
| traditional REMD/state-wise sampler | 每个 state 独立或进行 replica-exchange sampling | state trajectories and energy matrix | **控制/回归**；不替代 IBS mixture identity |
| all-history TMBAR during pilot | 用完整学习历史更新/诊断 `(f_k)` | bias update ledger | **learning-only**；production 后禁止继续更新 |
| fixed-bias validation | 冻结 `(f_k)` 后检查 mixture health | validation batches | **freeze gate**；数据不能混回 learning set |
| local augmented TMBAR | mixture row + target rows 的 window 主 estimator | local covariance/free-energy block | **Stage 2 主值** |
| covariance-chain stitching | 通过共享 boundary states 拼接全路径 | global ΔG and covariance chain | **主线**；不得丢 edge 或重复计数 |
| mixture-coverage ESS | 衡量 mixture 对各 target state 的有效覆盖 | processed coverage ratio | **质量门**；与 raw importance overlap 分开报告 |
| decorrelation module | 选择最差 target sequence，同步抽取所有 rows | decorrelated indices | **质量门**；禁止每 row 各抽各的造成错位 |
| split-half/moving-block | stationarity 与 time-correlation 诊断 | drift/block uncertainty | **必须报告**；不替代 independent repeats |
| fixed-H bidirectional probe | 对单一困难 λ edge 做受控正向/反向检查 | edge diagnostic | 仅作 diagnostic；不能作为整条 path 主值 |
| immutable rescue | 新建 bridge/rescue ensemble 补覆盖 | rescue plan and replacement ledger | **允许恢复策略**；原 window 不修改 |
| APBS external correction | 在两腿完成后加入 electrostatic finite-size scalar | manifest/result/maps/net-charge record | 条件性模块；不是 IBS state，也不替代 LRC |

**English**

| Functional module | Function | Primary artifact | Recommendation and analysis |
|---|---|---|---|
| IBS log-sum-exp sampler | One local ensemble covers multiple λ target states | Mixture trajectory, state probabilities, bias histories | **Stage 2 mainline**; must be analyzed with the actual mixture row |
| traditional REMD/state-wise sampler | Independent sampling or replica exchange for each state | State trajectories and energy matrix | **Control/regression**; does not replace the IBS mixture identity |
| all-history TMBAR during pilot | Updates/diagnoses `(f_k)` using the complete learning history | Bias update ledger | **Learning only**; updates are forbidden after production begins |
| fixed-bias validation | Checks mixture health after freezing `(f_k)` | Validation batches | **Freeze gate**; data must not be mixed back into the learning set |
| local augmented TMBAR | Window primary estimator using the mixture row plus target rows | Local covariance/free-energy block | **Stage 2 primary value** |
| covariance-chain stitching | Joins the full path through shared boundary states | Global ΔG and covariance chain | **Mainline**; edges must not be dropped or counted twice |
| mixture-coverage ESS | Measures effective mixture coverage of each target state | Processed coverage ratio | **Quality gate**; report separately from raw importance overlap |
| decorrelation module | Selects the worst target sequence and synchronously extracts all rows | Decorrelated indices | **Quality gate**; do not independently subsample each row and create misalignment |
| split-half/moving-block | Stationarity and time-correlation diagnostics | Drift/block uncertainty | **Must be reported**; does not replace independent repeats |
| fixed-H bidirectional probe | Controlled forward/reverse check of a single difficult λ edge | Edge diagnostic | Diagnostic only; cannot serve as the primary value for the full path |
| immutable rescue | Creates a new bridge/rescue ensemble to restore coverage | Rescue plan and replacement ledger | **Allowed recovery strategy**; original window is not modified |
| APBS external correction | Adds an electrostatic finite-size scalar after both legs are complete | Manifest/result/maps/net-charge record | Conditional module; not an IBS state and does not replace LRC |

##### 3.4.2.1 Actual-mixture augmented MBAR

**中文**

对第 \(n\) 帧，实际采样能量与第 \(k\) 个物理目标能量分别为

\[
U_{s,n}=E_{\mathrm{base},n}+V_{\mathrm{IBS},n}+V_{\mathrm{WCA},n},
\]

\[
U_{k,n}^{\mathrm{phys}}=E_{\mathrm{base},n}+U_{k,n}^{\mathrm{softcore}}+\frac{C_k}{V_n}.
\]

因此每个 local window 的增广 reduced-potential column 是

\[
\mathbf u_n^{\mathrm{aug}}
=\beta
\begin{bmatrix}
U_{s,n}-c_n\\
U_{0,n}^{\mathrm{phys}}-c_n\\
\vdots\\
U_{K-1,n}^{\mathrm{phys}}-c_n
\end{bmatrix},
\qquad
\mathbf N=(N,0,\ldots,0).
\]

其中 \(c_n\) 只用于数值稳定，逐列减去公共常数不改变自由能差。row 0 对应真正产生轨迹的 IBS/WCA mixture；其余各行只是需要重加权得到的物理 target，因此样本数为 0。

PyMBAR 给出的物理自由能可写为

\[
F_k=k_BT\,\Delta f_{0\to k}.
\]

窗口端点不确定度必须直接读取同一次 MBAR 拟合的差值协方差：

\[
\sigma_{a\to b}=k_BT\,d\Delta f_{a,b}.
\]

两个 endpoint 来自同一次 local MBAR 拟合，彼此相关，所以不能再用 \(\sqrt{\sigma_a^2+\sigma_b^2}\) 代替这个差值协方差。

**English**

For frame \(n\), the sampled and physical-target energies are

\[
U_{s,n}=E_{\mathrm{base},n}+V_{\mathrm{IBS},n}+V_{\mathrm{WCA},n},
\]

\[
U_{k,n}^{\mathrm{phys}}=E_{\mathrm{base},n}+U_{k,n}^{\mathrm{softcore}}+\frac{C_k}{V_n}.
\]

The augmented reduced-potential column is

\[
\mathbf u_n^{\mathrm{aug}}
=\beta
\begin{bmatrix}
U_{s,n}-c_n\\
U_{0,n}^{\mathrm{phys}}-c_n\\
\vdots\\
U_{K-1,n}^{\mathrm{phys}}-c_n
\end{bmatrix},
\qquad
\mathbf N=(N,0,\ldots,0).
\]

The shift \(c_n\) is purely numerical and does not change free-energy differences. Row 0 is the IBS/WCA mixture that generated the trajectory; the remaining rows are unsampled physical targets. PyMBAR returns

\[
F_k=k_BT\,\Delta f_{0\to k},
\qquad
\sigma_{a\to b}=k_BT\,d\Delta f_{a,b}.
\]

The endpoint estimates are correlated within the same local MBAR fit, so their difference uncertainty cannot be replaced by \(\sqrt{\sigma_a^2+\sigma_b^2}\).

##### 3.4.2.2 同步去相关、mixture coverage 与 ESS / Synchronized Decorrelation, Mixture Coverage, and ESS

**中文**

对每个 target 构造重加权序列

\[
\Delta u_{k,n}=\beta\left(U_{k,n}^{\mathrm{phys}}-U_{s,n}\right).
\]

分别估计 statistical inefficiency \(g_k\)，选择最慢的 target state 决定一组公共 decorrelated indices，并用同一组 indices 同步抽取所有能量行。不能逐 row 独立抽样，否则同一列不再代表同一个构象。

冻结 \(f_k\) 后，责任概率与 mixture coverage ESS 为

\[
p_{n k}=
\frac{\exp[-\beta(U'_{k,n}-f_k)]}
{\sum_j\exp[-\beta(U'_{j,n}-f_j)]},
\]

\[
ESS_k^{\mathrm{mix}}=
\frac{\left(\sum_np_{nk}\right)^2}{\sum_np_{nk}^2},
\qquad
R_k^{\mathrm{mix}}=
\frac{ESS_k^{\mathrm{mix}}}{N_{\mathrm{decorrelated}}}.
\]

最终 coverage gate 使用所有 window/state 中最差的 \(R_k^{\mathrm{mix}}\)。occupancy

\[
\mathrm{Occ}_k=K\langle p_k\rangle
\]

是一阶矩诊断，不能替代 ESS；一个状态可能持续以很小但近似恒定的概率出现，从而给出表面较高的 ESS。raw importance ESS、absolute ESS、occupancy 与 common-mode residual 均需保留原名报告，但在 ESS gate v3 中不替代 processed mixture-coverage ratio。

**English**

For each target, the reweighting series is

\[
\Delta u_{k,n}=\beta\left(U_{k,n}^{\mathrm{phys}}-U_{s,n}\right).
\]

The statistical inefficiency \(g_k\) is estimated for every target. The slowest target determines one common set of decorrelated frame indices, which is then applied to all energy rows. Row-wise independent subsampling would break the frame alignment required by MBAR.

After \(f_k\) is frozen, the state responsibilities and mixture-coverage ESS are

\[
p_{n k}=
\frac{\exp[-\beta(U'_{k,n}-f_k)]}
{\sum_j\exp[-\beta(U'_{j,n}-f_j)]},
\]

\[
ESS_k^{\mathrm{mix}}=
\frac{\left(\sum_np_{nk}\right)^2}{\sum_np_{nk}^2},
\qquad
R_k^{\mathrm{mix}}=
\frac{ESS_k^{\mathrm{mix}}}{N_{\mathrm{decorrelated}}}.
\]

The final coverage statistic is the worst \(R_k^{\mathrm{mix}}\) over all states and windows. Occupancy,

\[
\mathrm{Occ}_k=K\langle p_k\rangle,
\]

is a first-moment diagnostic rather than a substitute for ESS. A state can remain uniformly starved and still show an apparently high normalized ESS. Raw importance ESS, absolute ESS, occupancy, and the common-mode residual are therefore reported separately.

##### 3.4.2.3 Shared-boundary covariance chain

**中文**

局部窗口先利用共享态做 inverse-variance offset 对齐。若 \(O_w\) 是窗口 \(w\) 与已拼接路径的重叠态集合，则

\[
o_w=
\frac{\sum_{k\in O_w}
(F_k^{\mathrm{global}}-F_{k,w}^{\mathrm{local}})/\nu_{k,w}}
{\sum_{k\in O_w}1/\nu_{k,w}},
\qquad
\nu_{k,w}=\operatorname{Var}(F_k^{\mathrm{global}})+
\operatorname{Var}(F_{k,w}^{\mathrm{local}}).
\]

最终主值按每个独立窗口从 join state 到新 endpoint 的 segment 累加：

\[
\Delta G_w=F_w(\lambda_{\mathrm{end}})-F_w(\lambda_{\mathrm{join}}),
\qquad
\Delta G_{\mathrm{stage}}=\sum_w\Delta G_w.
\]

每段误差直接取 local MBAR 的 endpoint-difference uncertainty。不同生产窗口相互独立时，才在窗口间相加方差：

\[
\sigma_w=k_BT\,d\Delta f_{w,\mathrm{join},\mathrm{end}},
\qquad
\sigma_{\mathrm{stage}}=\sqrt{\sum_w\sigma_w^2}.
\]

这样每条 edge 只计算一次。同一窗口内的 endpoint covariance 已经包含在 \(d\Delta f\) 中；只有相互独立窗口的方差才在窗口间平方相加。这个实现受论文 Eq. 15 启发，但它是 local augmented TMBAR 加 covariance-chain stitching，并不是论文写出的单一 time-aggregated distribution。

**English**

Local windows are aligned through their shared states. If \(O_w\) is the overlap between window \(w\) and the assembled path, the inverse-variance offset is

\[
o_w=
\frac{\sum_{k\in O_w}
(F_k^{\mathrm{global}}-F_{k,w}^{\mathrm{local}})/\nu_{k,w}}
{\sum_{k\in O_w}1/\nu_{k,w}},
\qquad
\nu_{k,w}=\operatorname{Var}(F_k^{\mathrm{global}})+
\operatorname{Var}(F_{k,w}^{\mathrm{local}}).
\]

Each independent window contributes the segment from its join state to its new endpoint:

\[
\Delta G_w=F_w(\lambda_{\mathrm{end}})-F_w(\lambda_{\mathrm{join}}),
\qquad
\Delta G_{\mathrm{stage}}=\sum_w\Delta G_w.
\]

The segment uncertainty comes directly from the local MBAR endpoint difference:

\[
\sigma_w=k_BT\,d\Delta f_{w,\mathrm{join},\mathrm{end}},
\qquad
\sigma_{\mathrm{stage}}=\sqrt{\sum_w\sigma_w^2}.
\]

Each edge is counted once. Within-window endpoint covariance is already included in \(d\Delta f\); only independent-window variances are added in quadrature. This estimator is inspired by Eq. 15 of the paper but implements local augmented TMBAR with covariance-chain stitching rather than a single time-aggregated distribution.

#### 3.4.3 Recovery 与 fail-closed 子模块 / Recovery and Fail-Closed Submodules

**中文**

| 恢复模块 | 允许动作 | 禁止动作 | 判定 |
|---|---|---|---|
| same-distribution extension | 从同一 checkpoint、同一 Hamiltonian、同一 frozen `(f_k)` 有限延长 | 修改 λ、potential 或重新学习 `(f_k)` | 首选响应 |
| immutable bridge rescue | 新建共享 state 的独立 bridge，例如 `(6,11)→(6,9)+(8,11)` | 覆盖或删除原 `(6,11)` 数据 | 允许，但需显式 replacement ledger |
| checkpoint resume | 验证 status、fingerprints、protocol keys 后继续 | 仅因文件存在就跳过 stage | 主线/封闭失败 |
| cache migration | 只对明确兼容的 v27/28/29 bias cache 使用 versioned contract | 将旧 Hamiltonian cache 当作新 protocol | 受限 |
| mutating rescue | 无 | 原地修改失败 window 的 λ/Hamiltonian/target identity | 禁止/弃用 |
| 核心边界 | within-run covariance 不能代表 between-run variance | — | 边界条件 |

**English**

| Recovery module | Allowed action | Forbidden action | Verdict |
|---|---|---|---|
| same-distribution extension | Extend for a limited duration from the same checkpoint, Hamiltonian, and frozen `(f_k)` | Change λ, change the potential, or relearn `(f_k)` | Allowed first response |
| immutable bridge rescue | Create an independent bridge sharing a state, e.g. `(6,11)→(6,9)+(8,11)` | Overwrite or delete the original `(6,11)` data | Allowed with an explicit replacement ledger |
| checkpoint resume | Continue after validating status, fingerprints, and protocol keys | Skip a stage merely because a file exists | Mainline/fail-closed |
| cache migration | Use a versioned contract only for explicitly compatible v27/28/29 bias caches | Treat an old-Hamiltonian cache as a new protocol | Restricted |
| mutating rescue | None | Modify a failed window’s λ/Hamiltonian/target identity in place | Prohibited/deprecated |
| Main boundary | Within-run covariance cannot represent between-run variance | — | Boundary condition |

#### 3.4.4 Local TMBAR 的完整资格门 / Complete Local-TMBAR Qualification Gates

**中文**

local TMBAR 的结果不能只看一个 ΔG 数字。分析首先确认 actual-mixture row 与所有 target rows 使用同一组构象和同步去相关帧，然后检查每个 local window 是否可解、完整 λ path 是否被覆盖，以及共享边界拼接后有没有遗漏或重复 edge。

当前 Stage 2 的内部通过条件是：最差 mixture-coverage ratio 不低于 `0.05`，同步去相关样本数不少于 `20`，任一 window 的 endpoint uncertainty 不高于 `1.0 kJ/mol`，并且整条 covariance chain 能够闭合。ESS gate v3 中，absolute ESS 与 state occupancy 仍是诊断量，因为 `min_absolute_ess_threshold=null`、`min_occupancy_is_gate=false`。报告因此分别给出 raw ESS、processed mixture coverage、absolute ESS 与 occupancy，不再把它们压成一个含糊的“ESS pass”。

这些检查回答的是单次运行内部的覆盖和统计稳定性。它们不能替代 stationarity、time-correlation 或独立重复；某一个 window 的 BAR 结果也不能代替 mixture-aware Stage 2 主估计。

**English**

A local TMBAR result is not qualified by its ΔG value alone. The analysis first confirms that the actual-mixture row and all target rows use the same configurations and synchronized decorrelated frames. It then checks that every local window is solvable, the complete λ path is covered, and shared-boundary stitching neither drops nor duplicates an edge.

The current Stage-2 internal criteria require a worst mixture-coverage ratio of at least `0.05`, at least `20` synchronized decorrelated samples, no endpoint uncertainty above `1.0 kJ/mol`, and a closed covariance chain. Under ESS gate v3, absolute ESS and state occupancy remain diagnostics because `min_absolute_ess_threshold=null` and `min_occupancy_is_gate=false`. Raw ESS, processed mixture coverage, absolute ESS, and occupancy are therefore reported separately rather than compressed into a single “ESS pass.”

These checks describe within-run coverage and statistical stability. They do not replace stationarity, time-correlation, or independent-repeat evidence, and a BAR result from one window cannot substitute for the mixture-aware Stage-2 primary estimate.

#### 3.4.5 LRC v3 与 non-mutating rescue / LRC v3 and Non-Mutating Rescue

**中文**

LRC v3 同时处理 \(r^{-6}\) 与 \(r^{-12}\) tails，并保持 sigma-resolved softcore denominator。对第 \(b\) 个 \(\sigma\) bin，定义

\[
S_{6,b}=\sum_{(i,j)\in b}\epsilon_{ij}\sigma_{ij}^{6},
\qquad
S_{12,b}=\sum_{(i,j)\in b}\epsilon_{ij}\sigma_{ij}^{12},
\]

\[
D_b(r;\lambda_k)=r^6+\alpha_{\mathrm{LJ}}\sigma_b^6
(1-\lambda_k)^{m_{\mathrm{LJ}}}.
\]

若 \(S(r)\) 为 switching function、\(r_s\) 为 switching 起点、\(r_c\) 为 cutoff，则

\[
I_{6,b}=
\int_{r_s}^{r_c}\frac{[1-S(r)]r^2}{D_b(r;\lambda_k)}dr
+\int_{r_c}^{\infty}\frac{r^2}{D_b(r;\lambda_k)}dr,
\]

\[
I_{12,b}=
\int_{r_s}^{r_c}\frac{[1-S(r)]r^2}{D_b(r;\lambda_k)^2}dr
+\int_{r_c}^{\infty}\frac{r^2}{D_b(r;\lambda_k)^2}dr.
\]

每个状态的系数及逐帧 target correction 为

\[
C_k=16\pi\lambda_k^{n_{\mathrm{LJ}}}
\sum_b\left(S_{12,b}I_{12,b}-S_{6,b}I_{6,b}\right),
\qquad
U_{k,n}^{\mathrm{LRC}}=\frac{C_k}{V_n}.
\]

逐帧修正 \(C_k/V_n\) 在 MBAR 之前加入物理 target row，但不参与 IBS 权重学习。native/custom-CV LRC 保持关闭，避免重复修正。该均匀密度表达式只适用于已经资格化的 soluble environment，不能直接套到非均匀膜环境。

失败窗口的处理也遵循同一个统计原因：一旦修改 λ 或 Hamiltonian，采样分布就变了，新旧数据不能再当成同一个 ensemble 合并。因此先从相同 checkpoint、相同 Hamiltonian 和冻结的 \(f_k\) 有限延长；若覆盖仍不足，再建立独立的 bridge/rescue ensemble。原窗口保留，二者的替换关系只在合并分析中记录。

**English**

LRC v3 treats both \(r^{-6}\) and \(r^{-12}\) tails while retaining a sigma-resolved softcore denominator. For sigma bin \(b\),

\[
S_{6,b}=\sum_{(i,j)\in b}\epsilon_{ij}\sigma_{ij}^{6},
\qquad
S_{12,b}=\sum_{(i,j)\in b}\epsilon_{ij}\sigma_{ij}^{12},
\]

\[
D_b(r;\lambda_k)=r^6+\alpha_{\mathrm{LJ}}\sigma_b^6
(1-\lambda_k)^{m_{\mathrm{LJ}}}.
\]

If \(S(r)\) is the switching function, \(r_s\) its starting point, and \(r_c\) the cutoff,

\[
I_{6,b}=
\int_{r_s}^{r_c}\frac{[1-S(r)]r^2}{D_b(r;\lambda_k)}dr
+\int_{r_c}^{\infty}\frac{r^2}{D_b(r;\lambda_k)}dr,
\]

\[
I_{12,b}=
\int_{r_s}^{r_c}\frac{[1-S(r)]r^2}{D_b(r;\lambda_k)^2}dr
+\int_{r_c}^{\infty}\frac{r^2}{D_b(r;\lambda_k)^2}dr.
\]

The state coefficient and frame-specific correction are

\[
C_k=16\pi\lambda_k^{n_{\mathrm{LJ}}}
\sum_b\left(S_{12,b}I_{12,b}-S_{6,b}I_{6,b}\right),
\qquad
U_{k,n}^{\mathrm{LRC}}=\frac{C_k}{V_n}.
\]

The correction enters the physical target row before MBAR analysis but is not used during IBS weight learning. Native/custom-CV LRC remains disabled to avoid double correction. The uniform-density expression is restricted to qualified soluble environments and is not transferred to an inhomogeneous membrane system.

The rescue logic follows from the same statistical distinction. Changing λ or the Hamiltonian changes the sampled distribution, so the modified data cannot be pooled as if they belonged to the original ensemble. The first response is a limited extension from the same checkpoint with the same Hamiltonian and frozen \(f_k\). If coverage remains inadequate, a separate bridge or rescue ensemble is created. The original window is retained, and the relationship between the two datasets is recorded during merged analysis.

##### 3.4.5.1 Dispersion/LRC 功能分支 / Dispersion and LRC Branches

**中文**

| 分支 | 作用 | 当前状态 |
|---|---|---|
| `legacy_uniform_density_lrc` | soluble ACE 的解析 LJ tail correction | 当前 soluble 默认 |
| `ff_native_isotropic_lrc` | 使用 Amber/native isotropic LRC 条件 | 已实现；Amber 优先 |
| `ff_native_force_switch_no_lrc` | CHARMM force-switch/no-LRC 条件 | 需额外膜/能量证据 |
| `lj_pme` | LJ-PME 长程色散 | 接口存在，未实现 |
| `membrane_inhomogeneous` | 膜非均匀色散修正 | 需要但未实现 |
| DEXP no-LRC branch | DEXP 独立 cutoff/switch，暂不加 LRC | 实验性 |

**English**

| Branch | Function | Current status |
|---|---|---|
| `legacy_uniform_density_lrc` | Analytical LJ tail correction for soluble ACE | Current soluble default |
| `ff_native_isotropic_lrc` | Uses Amber/native isotropic LRC conditions | Implemented; Amber preferred |
| `ff_native_force_switch_no_lrc` | CHARMM force-switch/no-LRC conditions | Requires additional membrane/energy evidence |
| `lj_pme` | LJ-PME long-range dispersion | Interface exists; not implemented |
| `membrane_inhomogeneous` | Inhomogeneous membrane dispersion correction | Needed but not implemented |
| DEXP no-LRC branch | Independent DEXP cutoff/switch with no LRC added yet | Experimental |

#### 3.4.6 APBS 在最终 cycle 中的位置 / Position of APBS in the Final Cycle

**中文**

APBS helper 用于 neutralizing-plasma 和膜介电环境下的外部静电有限尺寸修正。它在 complex/solvent 两腿完成后作为 scalar 进入热力学循环，不改变 IBS λ states。artifact 中 `APBS=0` 只表示未启用，不能解释为 APBS 已验证为零；真实膜 APBS cycle closure 仍为 `PENDING`。

**English**

The APBS helper provides an external electrostatic finite-size correction for neutralizing-plasma and membrane dielectric environments. It enters the thermodynamic cycle as a scalar after the complex and solvent legs are complete and does not alter the IBS λ states. `APBS=0` means disabled, not validated as zero; closure of a real membrane APBS cycle remains `PENDING`.

#### 3.4.7 最终 cycle 与当前误差传播 / Final Cycle and Current Error Propagation

**中文**

按第 2 章的符号，最终结合自由能为

\[
\Delta G_{\mathrm{bind}}^\circ
=\Delta G_{\mathrm{complex}}
-\Delta G_{\mathrm{solvent}}
+\Delta G_{\mathrm{APBS}},
\]

其中 complex ledger 已包含 sampled Boresch attachment 与 analytical release，各腿又分别包含 Stage 1、Stage 2 与适用的 LRC。当前实现把两条腿视为独立随机估计，因此

\[
\sigma_{\mathrm{bind}}
=\sqrt{\sigma_{\mathrm{complex}}^2+
\sigma_{\mathrm{solvent}}^2}.
\]

解析 Boresch release 和外部 APBS scalar 在当前代码中按确定性修正记账，因此没有进入 sampled variance。这个做法只描述现有 uncertainty ledger，并不表示它们在物理上没有不确定度；anchor sensitivity、APBS model uncertainty 和 between-run variance 仍需单独评估。

合格 APBS record 必须包含 manifest、result file、dielectric maps、lipid-charge map 与 net charge；只填 `apbs_correction_kJ_mol` 不能构成证据。neutral 与 co-ion routes 应将 APBS 标为 not applicable，且 APBS 永远不能替代 dispersion/LRC。

**English**

Using the notation of Section 2, the final binding free energy is

\[
\Delta G_{\mathrm{bind}}^\circ
=\Delta G_{\mathrm{complex}}
-\Delta G_{\mathrm{solvent}}
+\Delta G_{\mathrm{APBS}}.
\]

The complex ledger already contains the sampled Boresch attachment and analytical release, while each leg contains Stage 1, Stage 2, and the applicable LRC. The current implementation treats the two legs as independent stochastic estimates:

\[
\sigma_{\mathrm{bind}}
=\sqrt{\sigma_{\mathrm{complex}}^2+
\sigma_{\mathrm{solvent}}^2}.
\]

Analytical Boresch release and an externally supplied APBS scalar are currently recorded as deterministic corrections and do not enter the sampled variance. This describes the implemented uncertainty ledger; it does not imply that anchor sensitivity, APBS model uncertainty, or between-run variance is physically zero.

A qualified APBS record contains the manifest, result file, dielectric maps, lipid-charge map, and recorded net charge. A lone `apbs_correction_kJ_mol` value is not sufficient evidence. Neutral and co-ion routes mark APBS as not applicable, and APBS does not replace dispersion/LRC treatment.

## 4. 数值结果、失效历史与不确定度分析 / Numerical Results, Failure Analysis, and Uncertainty

### 4.1 当前候选结果对比 / Current Candidate Comparison

#### 4.1.1 三条 complex leg 的直接对照 / Direct Comparison of the Three Complex Legs

**中文**

| Complex artifact | Total ΔG_complex | Attachment | Decharge | vdW | Analytical release | Net sampled+analytic Boresch | 证据身份 |
|---|---:|---:|---:|---:|---:|---:|---|
| historical `output_lrc_fix` | `43.260` | `1.049` | `17.799` | `34.137` | `−9.264` | `−8.215` | `PROVENANCE_MIXED_HISTORICAL`；复用目录；旧 code/system |
| Seed 20260906 | `48.437` | `0.929` | `19.405` | `37.603` | `−9.037` | `−8.108` | 当前协议族；仅完成 complex |
| Seed 20260907 | `48.075` | `1.108` | `17.342` | `38.674` | `−8.588` | `−7.480` | 当前协议族；完成 complex 与 binding cycle |

**English**

| Complex artifact | Total ΔG_complex | Attachment | Decharge | vdW | Analytical release | Net sampled+analytic Boresch | Evidence identity |
|---|---:|---:|---:|---:|---:|---:|---|
| historical `output_lrc_fix` | `43.260` | `1.049` | `17.799` | `34.137` | `−9.264` | `−8.215` | `PROVENANCE_MIXED_HISTORICAL`; reused directory; old code/system |
| Seed 20260906 | `48.437` | `0.929` | `19.405` | `37.603` | `−9.037` | `−8.108` | Current protocol family; complete complex only |
| Seed 20260907 | `48.075` | `1.108` | `17.342` | `38.674` | `−8.588` | `−7.480` | Current protocol family; complete complex and binding cycle |

All values are in `kcal/mol`. `Attachment + decharge + vdW` forms the decoupling ledger; analytical release and the common constraint correction (`−0.461 kcal/mol`) then close the complex leg.

所有数值单位均为 `kcal/mol`。`Attachment + decharge + vdW` 构成 decoupling ledger；随后加入 analytical release 与三条腿共同的 constraint correction（`−0.461 kcal/mol`）得到 complex-leg total。

当前磁盘上的 `output_lrc_fix` 不能简单标成“未写入 attachment”，但也不能作为干净的修复后重复。其单一 `pipeline.log` 在 `02:09` 先记录错误的 BAR/TI `98.755/107.023 kJ/mol`，在 `08:09` 又于同一目录记录修复后的 `4.3889/4.4509 kJ/mol`；Stage 1/2 随后以 resume 方式完成。最终 JSON 的代数确实包含后一个 `4.3889 kJ/mol` attachment，但目录混合了失败代和修复代，缺少不可变、独立的 clean post-fix run identity。因此本报告将它降级为 `PROVENANCE_MIXED_HISTORICAL`，不再称作可与 Seed06/07 对等的修复后 baseline。

The current on-disk `output_lrc_fix` cannot simply be described as omitting attachment, but neither is it a clean post-fix repeat. Its single `pipeline.log` records the erroneous BAR/TI values `98.755/107.023 kJ/mol` at 02:09 and the corrected `4.3889/4.4509 kJ/mol` in the same directory at 08:09; Stage 1/2 were then completed through resume. The final JSON algebra does include the latter `4.3889 kJ/mol` attachment, but the directory mixes failed and repaired generations without an immutable, independent clean post-fix run identity. It is therefore downgraded to `PROVENANCE_MIXED_HISTORICAL` rather than treated as a repaired baseline comparable to Seed06/07.

真正可用于判断当前随机重复性的，是后两条 current-protocol-family complex legs。Seed06 与 Seed07 使用相同 code hash、system XML hash、OpenMM 版本以及 IBS-v29/path-v21/LRC-v3/WCA-v2 合同；两者 total complex 只差 `0.362 kcal/mol`，小于合并内部误差 `0.531 kcal/mol`。不过二者独立 equilibration coordinates、Boresch candidate pool 和最终 ligand anchors 不同，因此更准确的称呼是“同一当前协议族的独立 complex runs”，而不是逐字节相同 Hamiltonian 的 seed-only replicas。

The two current-protocol-family complex legs provide the relevant repeat comparison. Seed06 and Seed07 share the code hash, system XML hash, OpenMM version, and IBS-v29/path-v21/LRC-v3/WCA-v2 contract. Their total complex values differ by only `0.362 kcal/mol`, below the combined internal uncertainty of `0.531 kcal/mol`. Their independently equilibrated coordinates, Boresch candidate pools, and final ligand anchors differ, so the precise description is independent complex runs under the same current protocol family, not byte-identical-Hamiltonian seed-only replicas.

在当前两条独立 complex runs 的证据范围内，`ΔG_complex=48.437±0.436` 与 `48.075±0.302 kcal/mol` 表现出**异常稳定的总量重复性**：绝对差仅 `0.362 kcal/mol`、相对约 `0.75%`，标准化差约为 `0.68` 个合并标准误。更重要的是，这种一致性是在独立 equilibration、不同 Boresch candidate pools，且 Seed07 更换 ligand anchors 的情况下仍然出现。因此，当前数据对“修复后协议族的 complex-leg total 具有强重复性”给出明确正证据；旧 `output_lrc_fix` 的低值不应再被用来否定这一点。

Within the present two-run evidence base, `ΔG_complex=48.437±0.436` and `48.075±0.302 kcal/mol` show **exceptionally stable repeatability of the total complex-leg free energy**. The absolute difference is only `0.362 kcal/mol` (approximately `0.75%`), corresponding to about `0.68` combined standard errors. This agreement persists across independent equilibration, different Boresch candidate pools, and a change in ligand anchors for Seed07. The current artifacts therefore provide clear positive evidence that the repaired current protocol family yields a highly repeatable complex-leg total; the lower mixed-history `output_lrc_fix` value should not be used to argue otherwise.

这一判断严格限定在 complex-leg total。Seed06/Seed07 的 decharge 分量相差 `2.063 kcal/mol`，vdW 分量相差 `1.071 kcal/mol`，方向相反并发生部分抵消；net Boresch 也相差 `0.628 kcal/mol`。因此当前证据支持“总 complex leg 异常稳定”，但尚不能证明每个分项分别稳定，更不能替代第二条完整 solvent leg 和完整 binding cycle 的独立重复。

This conclusion is restricted to the total complex leg. The Seed06/Seed07 decharge components differ by `2.063 kcal/mol`, the vdW components by `1.071 kcal/mol` in the opposite direction, with partial compensation, and the net Boresch terms differ by `0.628 kcal/mol`. The evidence therefore supports exceptional stability of the total complex leg, but not independent stability of every component, and it does not replace a second complete solvent leg or complete binding-cycle repeat.

旧 complex leg 比 Seed06/Seed07 分别低 `5.178/4.816 kcal/mol`。这个差异不能由“漏掉显式 attachment”单独解释：旧腿与 Seed06 的 net sampled+analytic Boresch 仅差 `0.107 kcal/mol`；显著差异同时存在于 decharge，尤其是 vdW。由于旧目录的 mixed-history provenance、旧 code/system/OpenMM identity 与不同 Boresch equilibrium geometry，现有数据无法把这部分差异唯一归因于 restraint、实现版本或随机采样。

The historical complex leg is lower than Seed06/Seed07 by `5.178/4.816 kcal/mol`. Omission of the explicit attachment term alone cannot explain the difference: the net sampled-plus-analytic Boresch contribution differs by only `0.107 kcal/mol` between the historical leg and Seed06, while substantial changes occur in decharge and especially vdW. Given the mixed-history provenance, older code/system/OpenMM identity, and different Boresch equilibrium geometry, the present artifacts cannot uniquely attribute this discrepancy to the restraint, implementation version, or stochastic sampling.

#### 4.1.2 两个完整 binding 候选的限定性对照 / Qualified Comparison of the Two Complete Binding Candidates

**中文**

| 完整候选 | ΔG_bind (kcal/mol) | 相对内部误差 | 关键运行身份 | 允许的解释 |
|---|---:|---:|---|---|
| historical `output_lrc_fix` | `−5.536` | `±0.601` | 混合失败/修复 attachment 历史；code `d3b0426d…`；system XML `e2eb7b94…` | `PROVENANCE_MIXED_HISTORICAL`；不是干净的 post-fix repeat |
| Seed 20260907 | `−11.927` | `±0.415` | code `5ece5b4d…`；system XML `21200724…`；OpenMM `8.5.2`；Boresch candidates `578` | 当前 artifact 可用；不合并 |

**English**

| Complete candidate | ΔG_bind (kcal/mol) | Internal uncertainty | Key run identity | Permitted interpretation |
|---|---:|---:|---|---|
| historical `output_lrc_fix` | `−5.536` | `±0.601` | Mixed failed/repaired attachment history; code `d3b0426d…`; system XML `e2eb7b94…` | `PROVENANCE_MIXED_HISTORICAL`; not a clean post-fix repeat |
| Seed 20260907 | `−11.927` | `±0.415` | Code `5ece5b4d…`; system XML `21200724…`; OpenMM `8.5.2`; Boresch candidates `578` | Available current artifact; not pooled |

二者 topology hash（`9b5988f6…`）和 coordinate hash（`fd41926b…`）相同，但运行身份不完全相同。其描述性数值间隔为 `6.391 kcal/mol`；该差值混合了代码/系统构建/软件版本/Boresch 候选集合与随机采样等潜在贡献，不能被命名为“seed effect”，也不能用来估计 same-protocol between-run variance。

The two artifacts share topology hash `9b5988f6…` and coordinate hash `fd41926b…`, but not the complete run identity. Their descriptive numerical separation is `6.391 kcal/mol`; it may combine contributions from code, system construction, software version, Boresch candidate set, and stochastic sampling. It must not be labeled a seed effect or used to estimate same-protocol between-run variance.

Seed 20260906 只有 complex-leg 结果，没有完整 binding cycle，因此不进入本表。早期 `+40.8362` 与 `+16.00 kJ/mol` 已被物理/实现缺陷判定失效，保留在下节 failure table，而不是与当前完整候选并列。

Seed 20260906 contains only a complex-leg result and therefore does not enter this table. The earlier `+40.8362` and `+16.00 kJ/mol` results were invalidated by physical or implementation defects and remain in the failure table below rather than being presented beside the complete candidates.

### 4.2 历史失效归因与改进链 / Post-Mortem and Protocol Evolution

**中文**

| 历史失效 | 根本原因 | 新增门禁或协议 | 当前状态 |
|---|---|---|---|
| `+40.8362 kJ/mol` | complex/solvent 符号相反；旧 PME self/LRC 与 endpoint 口径错误 | 显式 sign ledger、统一 cycle formula、endpoint diagnostics | `INVALIDATED` |
| `+16.00 kJ/mol` | Boresch angle/dihedral 映射错误；过期 geometry；pose 位移约 `3.42 Å` | geometry commit、pose-consistency gate、anchor fingerprint、cache invalidation | `INVALIDATED` |
| Boresch attachment `98.7551/107.0230 kJ/mol` | mirrored-dihedral/sign defect；旧错误势能约 `943 kJ/mol` | 统一 dihedral 实现；修复后 BAR/TI `4.3889/4.4509`，六坐标均 `<0.5σ` | mapping defect closed；sensitivity pending |
| membrane `+97.3579 kJ/mol` | loader 漏读第二个 `HarmonicAngleForce` 中的 `71` 个 ligand angles | 遍历全部 forces；精确核对 `41/71/104` bonds/angles/torsions | result invalid；loader repaired |
| uncertainty underestimation | within-run covariance 远小于 between-run scatter | independent repeats、moving-block/bootstrap、between-run variance | open |

**English**

| Historical failure | Root cause | Added control | Status |
|---|---|---|---|
| `+40.8362 kJ/mol` | Complex/solvent signs were opposite; old PME self/LRC and endpoint conventions were incorrect | Explicit sign ledger, unified cycle formula, endpoint diagnostics | `INVALIDATED` |
| `+16.00 kJ/mol` | Boresch angle/dihedral mapping error; stale geometry; pose displacement of approximately `3.42 Å` | Geometry commit, pose-consistency gate, anchor fingerprint, cache invalidation | `INVALIDATED` |
| Boresch attachment `98.7551/107.0230 kJ/mol` | Mirrored-dihedral/sign defect; old incorrect potential reached approximately `943 kJ/mol` | Unified dihedral implementation; corrected BAR/TI `4.3889/4.4509`, all six coordinates `<0.5σ` | Mapping defect closed; sensitivity pending |
| Membrane `+97.3579 kJ/mol` | Loader failed to read `71` ligand angles in the second `HarmonicAngleForce` | Iterate over all forces; exact `41/71/104` bond/angle/torsion checks | Result invalid; loader repaired |
| Uncertainty underestimation | Within-run covariance was far smaller than between-run scatter | Independent repeats, moving-block/bootstrap, between-run variance | Open |

`−5.536 kcal/mol` 不是可以晋级的修复后 baseline。当前最终 JSON 的代数使用了修复后的 attachment，但其目录混合失败与修复两代、后续阶段又通过 resume 完成，因此状态为 `PROVENANCE_MIXED_HISTORICAL`；现有证据既不足以把它简单恢复为“漏算 attachment”的原始值，也不足以把它当作 clean post-fix run。

这些失效直接促成 Boresch geometry submission、pose-consistency gate、显式 sign ledger、cache fingerprint、fail-closed resume、non-mutating rescue 和当前候选的重新计算。负结果因此属于方法证据，而不是应被删除的噪声。

These failures directly motivated Boresch geometry submission, pose-consistency gates, explicit sign ledgers, cache fingerprints, fail-closed resume, non-mutating rescue, and recalculation of the current candidate. The negative results are methodological evidence rather than noise to be deleted.

### 4.3 当前诊断与跨运行风险 / Current Diagnostics and Cross-Run Risk

新旧完整候选的描述性间隔为 `6.391 kcal/mol`，但由于二者 protocol identity 不完全相同，这个差值不能直接进入 between-run variance 估计。它首先证明的是：当前 artifact 之间存在尚未分解的 protocol-plus-sampling discrepancy。正式不确定度必须来自冻结身份后的独立重复，并分别报告 within-run covariance、moving-block/bootstrap 与 same-protocol between-run variance。

The descriptive separation between the historical and newer complete candidates is `6.391 kcal/mol`, but their differing protocol identities prevent direct use of this value in a between-run variance estimate. It establishes an unresolved protocol-plus-sampling discrepancy across the current artifacts. Formal uncertainty must instead come from independent repeats under a frozen identity, with within-run covariance, moving-block/bootstrap uncertainty, and same-protocol between-run variance reported separately.

---

### 4.4 Mixed-history 历史 artifact 的 component ledger / Component Ledger of the Mixed-History Artifact

**中文**

| 数量 | 数值 | 证据边界 |
|---|---:|---|
| `ΔG_complex` | `180.9981 kJ/mol` | 历史 artifact 分量 |
| `ΔG_solvent` | `157.8358 kJ/mol` | 历史 artifact 分量 |
| Boresch 字段 | `−38.7609 kJ/mol` | 已包含在 complex ledger |
| APBS | `0` | 已禁用，未验证为零 |
| `ΔG_bind` | `−23.1622 kJ/mol` | 历史候选 |
| 报告的 sampling uncertainty | `2.5139 kJ/mol` | 不包括 between-run spread |
| `ΔG_bind` | `−5.5359 kcal/mol` | 单位换算 |
| uncertainty | `0.6008 kcal/mol` | 单位换算 |

**English**

| Quantity | Value | Evidence boundary |
|---|---:|---|
| `ΔG_complex` | `180.9981 kJ/mol` | Historical artifact component |
| `ΔG_solvent` | `157.8358 kJ/mol` | Historical artifact component |
| Boresch field | `−38.7609 kJ/mol` | Already included in complex ledger |
| APBS | `0` | Disabled; not validated as zero |
| `ΔG_bind` | `−23.1622 kJ/mol` | Historical candidate |
| Reported sampling uncertainty | `2.5139 kJ/mol` | Excludes between-run spread |
| `ΔG_bind` | `−5.5359 kcal/mol` | Unit conversion |
| Uncertainty | `0.6008 kcal/mol` | Unit conversion |

Boresch correction 的展示字段与 helper 内部使用的 release quantity 不能凭字段名再次组合。该 artifact 已在 complex leg 中处理 Boresch，因此任何手工二次扣除都会产生 double counting。

The displayed Boresch field and the release quantity used internally by the helper must not be recombined based on field names. Boresch is already included in the complex leg of this artifact, so manual subtraction would double count it.

#### 4.4.1 Stage 2 诊断的逐项解释 / Itemized Interpretation of Stage-2 Diagnostics

**中文**

| 指标 | Complex | Solvent | 解释 |
|---|---:|---:|---|
| 最小 overlap | `0.3913` | `0.4438` | 高于该 artifact 使用的门 |
| raw 最小 importance overlap | `0.01045` | `0.00502` | 更严格的 raw-weight 诊断；不是上面的 processed overlap |
| 最小去相关样本数 | `96` | `266` | 可用内部样本 |
| 最小 absolute ESS | `37.56` | `145.11` | complex 较弱；阈值为空 |
| 最大 endpoint uncertainty | `0.9249` | `0.9326 kJ/mol` | 低于 artifact 的 1.0 gate |
| 最小 state occupancy | `0.7478` | `0.6621` | 所有 states 均有占据，但不均匀 |
| 最大 common-mode log σ | `2.7272` | `1.8189` | covariance/common-mode 诊断 |
| offset-error contribution | `4.7252` | `5.6928 kJ/mol` | 拼接 offset 贡献不可忽略 |
| λ nodes | `23` | `23` | v21 final path |
| dropped windows | `0` | `0` | 所有 local windows 都进入 stitching |
| artifact convergence | `true` | `true` | 仅表示内部 gate |

**English**

| Metric | Complex | Solvent | Interpretation |
|---|---:|---:|---|
| Minimum overlap | `0.3913` | `0.4438` | Above the gate used by that artifact |
| Raw minimum importance overlap | `0.01045` | `0.00502` | Stricter raw-weight diagnostic; not the processed overlap above |
| Minimum decorrelated samples | `96` | `266` | Usable internal samples |
| Minimum absolute ESS | `37.56` | `145.11` | Complex is weaker; threshold was null |
| Maximum endpoint uncertainty | `0.9249` | `0.9326 kJ/mol` | Below the artifact’s 1.0 gate |
| Minimum state occupancy | `0.7478` | `0.6621` | All states are occupied, but not uniformly |
| Maximum common-mode log σ | `2.7272` | `1.8189` | Covariance/common-mode diagnostic |
| Offset-error contribution | `4.7252` | `5.6928 kJ/mol` | Non-negligible stitched-offset contribution |
| λ nodes | `23` | `23` | v21 final path |
| Dropped windows | `0` | `0` | All local windows entered stitching |
| Artifact convergence | `true` | `true` | Internal gate only |

`artifact convergence=true` 只表示该 artifact 的内部协议门通过，不表示 publication qualification、independent reproducibility 或与新协议兼容。

`artifact convergence=true` means only that the internal artifact gates passed. It does not establish publication qualification, independent reproducibility, or compatibility with a newer protocol.

#### 4.4.2 历史候选与 seed 20260907 的 Stage-2 横向比较 / Stage-2 Comparison with Seed 20260907

下表用于定位“哪些 diagnostics 同时发生了变化”，而不是把两个 artifact 当作严格重复。由于 code/system identity 不同，任何指标差异都只能形成后续受控实验的假设，不能单独归因于 seed。

This table identifies which diagnostics changed together; it does not treat the artifacts as strict repeats. Because code/system identities differ, metric differences can generate hypotheses for controlled experiments but cannot be attributed to the seed alone.

**中文**

| 指标 | 历史 complex | 历史 solvent | Seed07 complex | Seed07 solvent |
|---|---:|---:|---:|---:|
| 处理后最小 overlap | `0.3913` | `0.4438` | `0.5127` | `0.3300` |
| raw 最小 importance overlap | `0.01045` | `0.00502` | `0.00818` | `0.00578` |
| 最小去相关样本数 | `96` | `266` | `137` | `123` |
| 最小 absolute ESS | `37.56` | `145.11` | `88.18` | `40.59` |
| 最大 endpoint σ (`kJ/mol`) | `0.9249` | `0.9326` | `0.7257` | `0.6154` |
| 最小 state occupancy | `0.7478` | `0.6621` | `0.9283` | `0.4043` |
| 最大 common-mode log σ | `2.7272` | `1.8189` | `2.1701` | `1.6339` |
| offset-error contribution (`kJ/mol`) | `4.7252` | `5.6928` | `9.1164` | `3.4001` |

**English**

| Metric | Historical complex | Historical solvent | Seed07 complex | Seed07 solvent |
|---|---:|---:|---:|---:|
| Processed minimum overlap | `0.3913` | `0.4438` | `0.5127` | `0.3300` |
| Raw minimum importance overlap | `0.01045` | `0.00502` | `0.00818` | `0.00578` |
| Minimum decorrelated samples | `96` | `266` | `137` | `123` |
| Minimum absolute ESS | `37.56` | `145.11` | `88.18` | `40.59` |
| Maximum endpoint σ (`kJ/mol`) | `0.9249` | `0.9326` | `0.7257` | `0.6154` |
| Minimum state occupancy | `0.7478` | `0.6621` | `0.9283` | `0.4043` |
| Maximum common-mode log σ | `2.7272` | `1.8189` | `2.1701` | `1.6339` |
| Offset-error contribution (`kJ/mol`) | `4.7252` | `5.6928` | `9.1164` | `3.4001` |

The processed overlap and the raw importance-weight overlap answer different questions and must never be silently substituted for one another. Seed07 improves several complex-leg diagnostics, but its complex offset-error contribution is larger and its solvent occupancy/absolute ESS is weaker. Therefore the large binding-value shift cannot be dismissed by pointing to one favorable overlap number.

processed overlap 与 raw importance-weight overlap 回答不同问题，不能互相替代。seed07 的若干 complex 指标更好，但 complex offset-error contribution 更大，solvent occupancy 与 absolute ESS 更弱。因此不能仅凭某个较好的 overlap 数字解释两次 binding 值的巨大差异。

#### 4.4.3 Solvent box 与重复运行证据 / Solvent-Box and Repeat-Run Evidence

**中文**

| 运行 | Decharge | Vanish | Solvent 总值 | 解释 |
|---|---:|---:|---:|---|
| pad 1.5, 2026-07-28 | `63.115` | `101.639` | `162.826 ± 1.559 kJ/mol` | 独立同盒尺寸/运行诊断 |
| main, 2026-07-29 | `62.800` | `96.964` | `157.836 kJ/mol` | 历史候选 solvent leg |
| pad 2.4 | `64.249` | `94.491` | `156.812 ± 1.792 kJ/mol` | 大盒尺寸诊断 |

**English**

| Run | Decharge | Vanish | Solvent total | Interpretation |
|---|---:|---:|---:|---|
| pad 1.5, 2026-07-28 | `63.115` | `101.639` | `162.826 ± 1.559 kJ/mol` | Independent same-size/run diagnostic |
| main, 2026-07-29 | `62.800` | `96.964` | `157.836 kJ/mol` | Historical candidate solvent leg |
| pad 2.4 | `64.249` | `94.491` | `156.812 ± 1.792 kJ/mol` | Larger-box diagnostic |

The same-box vanishing difference is `4.675 kJ/mol` (`2.34σ`), whereas the reported cross-box difference is only about `1.13σ`. The evidence therefore does **not** establish a finite-size effect; run-to-run scatter is at least as plausible and must be quantified first.

同一盒尺寸的 vanishing 差达到 `4.675 kJ/mol`（`2.34σ`），而报告的跨盒差仅约 `1.13σ`。因此现有证据**不能**证明 finite-size effect；必须先量化 run-to-run scatter。

### 4.5 Boresch 与 Stage 2 诊断如何解释 / Interpretation of Boresch and Stage-2 Diagnostics

上表已经限定了 `artifact convergence=true` 的适用范围。与之相同，500-frame Boresch harmonicity flag 只能说明局部几何近似 harmonic，不能替代 anchor/force-constant sensitivity。

The table above already defines the limited scope of `artifact convergence=true`. Likewise, the 500-frame Boresch harmonicity flag indicates only approximately harmonic local geometry and cannot replace anchor and force-constant sensitivity analysis.

The attachment-sign defect provides a quantitative example. Before correction, BAR/TI attachment estimates were approximately `98.7551/107.0230 kJ/mol`; after correction they became `4.3889/4.4509 kJ/mol`, with the corrected BAR estimate `4.3889 ± 0.0779 kJ/mol`. The old mirrored dihedral construction could create energies near `943 kJ/mol`. After correction, all six reference coordinates were within `0.5σ` of their sampled distributions. This closes the sign/mapping defect, but not the independent anchor and force-constant sensitivity requirement.

attachment sign 缺陷给出了定量反例：修复前 BAR/TI 约为 `98.7551/107.0230 kJ/mol`；修复后变为 `4.3889/4.4509 kJ/mol`，其中修复后的 BAR 为 `4.3889 ± 0.0779 kJ/mol`。旧 mirrored dihedral 可产生接近 `943 kJ/mol` 的能量；修复后六个 reference coordinates 均落在采样分布的 `0.5σ` 内。这关闭了 sign/mapping 缺陷，但没有替代独立的 anchor 与 force-constant sensitivity。

### 4.6 为什么 mixed-history artifact 不能晋级 / Why the Mixed-History Artifact Cannot Be Promoted

**中文**

| 缺口 | 当前情况 | 影响 |
|---|---|---|
| independent repeats | `performed=false` | 运行间变异未知 |
| production seed ledger | principal ledger 不完整/为空 | stochastic provenance 不完整 |
| time-correlated uncertainty | P1-22 未闭合 | endpoint σ 可能偏乐观 |
| Stage 1 estimator provenance | 旧字段 | 需要 current-v2 reanalysis |
| joint protocol validation | 文档跨 v19/v27/v29 | 旧测试不能证明新协议 |
| Boresch sensitivity | 一个径向常数被裁剪 | restraint 稳健性未解决 |

**English**

| Gap | Current state | Consequence |
|---|---|---|
| Independent repeats | `performed=false` | Run-to-run variation is unknown |
| Production seed ledger | Incomplete/empty in the principal ledger | Stochastic provenance is incomplete |
| Time-correlated uncertainty | P1-22 not closed | Endpoint σ may be optimistic |
| Stage 1 estimator provenance | Older fields | Current-v2 reanalysis is required |
| Joint protocol validation | Documents span v19/v27/v29 | Old tests cannot prove the new protocol |
| Boresch sensitivity | One radial constant was clipped | Restraint robustness remains unresolved |

#### 4.6.1 P1-19/P1-22 与 run-level variance

同协议诊断值为：

```text
5.7726, 5.8623, 6.0786, 6.0880, 7.5216
```

**中文**

| 统计量 | 数值 |
|---|---:|
| typical internal σ | approximately `0.10` |
| five-run sample SD | approximately `0.716` |
| SD after excluding the obvious outlier | approximately `0.158` |

**English**

| Statistic | Value |
|---|---:|
| Typical internal σ | approximately `0.10` |
| Five-run sample SD | approximately `0.716` |
| SD after excluding the obvious outlier | approximately `0.158` |

这些不是最终 binding free energies，而是同协议散布诊断。它们证明 MBAR/TMBAR covariance 只描述给定样本集内部的不确定度；慢构象、初始条件和时间相关性会贡献额外 run-level variance。

These are not final binding free energies, but diagnostics of same-protocol dispersion. They show that MBAR/TMBAR covariance describes uncertainty within a given sample set, while slow conformations, initial conditions, and temporal correlation contribute additional run-level variance.

离群运行只有在预注册的物理或统计排除规则下才能剔除。为了缩小误差而事后删除是不允许的；正式结果必须并列报告 within-run、block/bootstrap 和 between-run uncertainty。

An outlier may be excluded only under a preregistered physical or statistical rule. Post hoc deletion to reduce the error bar is not allowed; formal reporting must present within-run, block/bootstrap, and between-run uncertainty side by side.

For the historical candidate, a split-half/run-scatter diagnostic gives an uncertainty scale of approximately `±4.23 kJ/mol` (`±1.01 kcal/mol`), larger than the artifact-reported `±2.51 kJ/mol` (`±0.60 kcal/mol`). This is an honest diagnostic bound rather than a frozen formal estimator; it is included to show the direction and likely size of the missing uncertainty component.

对历史候选，split-half/run-scatter 诊断给出约 `±4.23 kJ/mol`（`±1.01 kcal/mol`）的不确定度尺度，大于 artifact 内部报告的 `±2.51 kJ/mol`（`±0.60 kcal/mol`）。这只是诚实的诊断边界，而非已冻结的正式 estimator；其作用是说明缺失误差项的方向和可能量级。

## 5. 体系扩展：带电配体与膜蛋白环境 / Extensions: Charged Ligands and Membrane Systems

### 5.1 扩展分支身份与证据边界 / Extension Identity and Evidence Boundary

**中文**

| 扩展模块 | 作用 | 当前证据 | 当前判决 |
|---|---|---|---|
| co-alchemical ion transfer | 保持 ligand/co-ion 总电荷守恒 | C1/C2/C3 | seam 已通过；完整带电 ABFE 未闭合 |
| static-neutral handoff | 将 `λ_coul=0` PME 系统烘焙为固定系统 | C3 v2、150 帧、0 失败 | 真实体系端点检查通过 |
| C1 water-box | 覆盖离子符号与盒尺寸 | 6 个案例 | 验证通过；不是结合结果 |
| C2 lipid slab | 覆盖厚度、位置与 seeds | 4 个 primary + repeats | 工程门通过；保留 seed 散布 |
| membrane loader | 完整重建配体拓扑 | 修复丢失的 `71` 个 angles | 根因已修复；production 必须重跑 |
| APBS/membrane LRC | 膜静电有限尺寸与非均匀色散 | cycle 未闭合 | 不能宣称膜 ABFE 已完成 |

**English**

| Extension module | Function | Current evidence | Decision |
|---|---|---|---|
| co-alchemical ion transfer | Preserve total ligand/co-ion charge conservation | C1/C2/C3 | Seam passed; complete charged ABFE remains open |
| static-neutral handoff | Bake the PME system at `λ_coul=0` into a fixed system | C3 v2, 150 frames, 0 failures | Real-system endpoint checks passed |
| C1 water-box | Cover ion sign and box size | 6 cases | Validation passed; not a binding result |
| C2 lipid slab | Cover thickness, position, and seeds | 4 primary runs + repeats | Engineering gates passed; seed scatter retained |
| membrane loader | Fully reconstruct ligand topology | Repaired loss of `71` angles | Root cause fixed; production must be rerun |
| APBS/membrane LRC | Finite-size membrane electrostatics and nonuniform dispersion | Cycle not closed | Do not claim membrane ABFE complete |

### 5.2 Charge-transfer handoff / Charge-Transfer Handoff

The charged-ligand route uses a co-alchemical ion to preserve total charge. Before Stage 2, the global parameterized NonbondedForce at `λ_coul=0` is baked into a fixed force and handed to IBS vanishing.

- static handoff is integrated;
- C1 real-waterbox qualification contains six ion/box cases, all of which passed charge conservation, finite-energy/force, geometry, restraint-group, endpoint, dynamics, and `u_kn` completion checks;
- C2 real lipid-slab qualification contains four thickness/position conditions and independent-seed repeats;
- C3 v1 exposed a maximum force mismatch of approximately `7.47×10⁻2`;
- C3 protocol v2 reprocessed five real cases over 150 frames with zero failures; A/B/C/D endpoints, C-seam, D strict-zero, input hashes, and the MEM-00h evaluation convention all passed;
- current Atenolol follows the neutral branch;
- no complete charged-ligand complex-minus-solvent ABFE production cycle has yet been closed.

The defensible statement has therefore advanced from “the seam is implemented” to **“the charged handoff and endpoint seam have passed real-system, multi-environment validation; a complete charged ABFE cycle and reproducibility qualification remain pending.”**

因此可支持的表述已经从“实现了 seam”推进为：**“charged handoff 与 endpoint seam 已通过真实体系、多环境验证；完整 charged ABFE cycle 与重复性资格化仍未完成。”**

#### 5.2.1 C1 水盒数据 / C1 Water-Box Data

**中文**

| 案例 | `ΔG` (`kJ/mol`) | 状态 |
|---|---:|---|
| Ca large | `663.441 ± 8.429` | 通过 |
| Ca small | `665.897 ± 9.398` | 通过 |
| Cl large | `2.617 ± 3.299` | 通过 |
| Cl small | `1.516 ± 3.213` | 通过 |
| Na large | `0.588 ± 3.787` | 通过 |
| Na small | `0.016 ± 3.404` | 通过 |

**English**

| Case | `ΔG` (`kJ/mol`) | Status |
|---|---:|---|
| Ca large | `663.441 ± 8.429` | Passed |
| Ca small | `665.897 ± 9.398` | Passed |
| Cl large | `2.617 ± 3.299` | Passed |
| Cl small | `1.516 ± 3.213` | Passed |
| Na large | `0.588 ± 3.787` | Passed |
| Na small | `0.016 ± 3.404` | Passed |

These are validation free energies, not ligand-binding results. Their role is to exercise charge conservation, co-ion geometry, endpoints, dynamics, and estimator plumbing across ion sign and box size.

这些数值是验证自由能，不是 ligand-binding 结果；它们用于覆盖离子符号、盒尺寸、总电荷守恒、co-ion 几何、端点、动力学和 estimator plumbing。

#### 5.2.2 C2 脂质 slab 数据 / C2 Lipid-Slab Data

**中文**

| 条件 | 主运行 (`kJ/mol`) | 独立 seeds (`kJ/mol`) | 门结果 |
|---|---:|---:|---|
| Na thick pos0 | `−3.172 ± 2.258` | `−2.659 ± 2.650`；`−0.082 ± 2.601` | 全部通过 |
| Na thick pos1 | `−0.756 ± 2.379` | `1.096 ± 2.547`；`0.839 ± 2.623` | 全部通过 |
| Na thin pos0 | `−2.958 ± 2.694` | `1.871 ± 2.614`；`−0.890 ± 2.692` | 全部通过 |
| Na thin pos1 | `−0.643 ± 2.293` | `−3.972 ± 2.176`；`2.555 ± 3.009` | seed2027 未通过旧 equality gate；hydration-v3 非劣性通过 |

**English**

| Condition | Primary run (`kJ/mol`) | Independent seeds (`kJ/mol`) | Gate result |
|---|---:|---:|---|
| Na thick pos0 | `−3.172 ± 2.258` | `−2.659 ± 2.650`; `−0.082 ± 2.601` | All passed |
| Na thick pos1 | `−0.756 ± 2.379` | `1.096 ± 2.547`; `0.839 ± 2.623` | All passed |
| Na thin pos0 | `−2.958 ± 2.694` | `1.871 ± 2.614`; `−0.890 ± 2.692` | All passed |
| Na thin pos1 | `−0.643 ± 2.293` | `−3.972 ± 2.176`; `2.555 ± 3.009` | seed2027 failed a legacy equality gate; hydration-v3 non-inferiority passed |

All four primary cases completed eleven λ states and passed static, dynamics, `u_kn`, co-ion geometry, charge, and endpoint checks. The hydration-v3 non-inferiority summary contains `12/12` passes at a water margin of `0.5`. A legacy equality gate for one seed was false; this is retained as version-evolution evidence rather than mislabeled as a current non-inferiority failure. The broad seed spread still shows that engineering correctness and reproducibility are separate gates.

四个 primary case 均完成 11 个 λ states，并通过 static、dynamics、`u_kn`、co-ion geometry、charge 和 endpoint 检查。hydration-v3 non-inferiority summary 在 water margin `0.5` 下为 `12/12` 通过；一个 seed 的旧 equality gate 为 false，这应保留为版本演化证据，而不是误写成当前 non-inferiority failure。跨 seed 散布仍证明 engineering correctness 与 reproducibility 是两个独立门。

#### 5.2.3 C3 真实端点矩阵 / C3 Real-Endpoint Matrix

Protocol v2 evaluates C1 Na-large plus four C2 slab conditions. A/B contains `100` frames, C/D contains `50`, for `150` total and `0` failures. All A/B/C/D endpoints passed; C-seam and D strict-zero passed. The evaluator used a non-mutating in-memory clone at cutoff `1.0 nm` with switching disabled. Raw C2 systems used switching at `0.995 nm`, while C1 already conformed; the normalization changed only the evaluator clone and preserved all raw systems and trajectories.

Protocol v2 覆盖 C1 Na-large 与四个 C2 slab 条件。A/B 为 `100` 帧，C/D 为 `50` 帧，共 `150` 帧、`0` 失败；A/B/C/D、C-seam 与 D strict-zero 全部通过。评估器只对内存 clone 采用 `1.0 nm` cutoff、关闭 switching；C2 raw systems 的 switching 为 `0.995 nm`，C1 已符合约定。归一化不改写任何 raw system 或 trajectory。

#### 5.2.4 Charged-route fail-closed contract

co-ion route 必须冻结 complex/solvent 两腿的 ion、coordinate 与 restraint fingerprints，并通过 charge、geometry、energy、force 和 endpoint 门。co-ion 与 APBS neutralizing-plasma 互斥；具体数值容差列入附录 A。

### 5.3 Membrane systems / 膜体系

#### 5.3.1 Membrane 运行控制模块 / Membrane Run-Control Modules

**中文**

| 模块/选项 | 作用 | 当前状态 |
|---|---|---|
| `system_type=soluble` | 各向同性水溶环境 | 当前默认 |
| `system_type=membrane` | 启用膜 barostat、质量门与膜 dispersion contract | 工程接口已实现 |
| `xy_mode=isotropic/anisotropic` | 控制膜平面缩放方式 | isotropic 默认 |
| `z_mode=free/fixed/constant_volume` | 控制膜法向盒变化 | free 默认 |
| `normal_axis=z` | 定义膜法向 | 仅 `z` 合法 |
| quality gate `enforce` | 未通过膜质量门即阻断 | 推荐 |
| quality gate `advisory` | 记录失败但继续 | 仅调试/接线 |
| membrane input declaration | 冻结 lipid/protein/ligand/build provenance | membrane 必需 |

**English**

| Module / Option | Function | Current status |
|---|---|---|
| `system_type=soluble` | Isotropic aqueous environment | Default |
| `system_type=membrane` | Enable the membrane barostat, quality gates, and membrane dispersion contract | Engineering interface implemented |
| `xy_mode=isotropic/anisotropic` | Control membrane-plane scaling | Isotropic default |
| `z_mode=free/fixed/constant_volume` | Control normal-direction box changes | Free default |
| `normal_axis=z` | Define the membrane normal | Only `z` is valid |
| quality gate `enforce` | Block when the membrane quality gate is not passed | Recommended |
| quality gate `advisory` | Record failure and continue | Debugging/wiring only |
| membrane input declaration | Freeze lipid/protein/ligand/build provenance | Required for membrane systems |

The membrane system passed a 100 ns NPT quality gate, and a Stage 0 Boresch NaN was corrected. A neutral 5 ns full engineering smoke completed Stage 0→1→2 and produced `−26.485 ± 1.767 kJ/mol` (`−6.330 ± 0.422 kcal/mol`). This demonstrates executable full-pipeline plumbing, but `4.98 ns` is insufficient for the membrane quality/production tail and does not validate the charged co-ion route.

膜体系通过了 100 ns NPT quality gate，Stage 0 Boresch NaN 也已修正。一个 neutral 5 ns 全流程 engineering smoke 已完成 Stage 0→1→2，得到 `−26.485 ± 1.767 kJ/mol`（`−6.330 ± 0.422 kcal/mol`）。它证明全流程可以执行，但 `4.98 ns` 不足以满足膜体系的 quality/production tail，也不能验证 charged co-ion route。

The historical 100 ns-path result, `+97.358 ± 2.092 kJ/mol` (`+23.269 kcal/mol`), is invalid because the solvent-leg loader assumed a single `HarmonicAngleForce`. The membrane input contained two such forces, and all `71` ligand angles were in the second; `ligand_only.xml` therefore contained zero angles. The ligand collapsed from a maximum internal heavy-atom distance of approximately `0.996 nm` to `0.660 nm`, the interaction scale shifted from roughly `−190 ± 34` to `−569 ± 90 kJ/mol`, and decharging moved from `62.80` to `191.05 kJ/mol`. The repair iterates all forces and reconciles exact counts: `41` bonds, `71` angles, and `104` torsions.

历史 100 ns-path 结果 `+97.358 ± 2.092 kJ/mol`（`+23.269 kcal/mol`）无效，原因是 solvent-leg loader 假设系统只有一个 `HarmonicAngleForce`。实际膜输入有两个，该配体全部 `71` 个 angles 位于第二个 force 中，导致 `ligand_only.xml` 的 angle 数为零。配体最大内部重原子距离从约 `0.996 nm` 塌缩到 `0.660 nm`，interaction 从约 `−190 ± 34` 变为 `−569 ± 90 kJ/mol`，decharging 从 `62.80` 变为 `191.05 kJ/mol`。修复后遍历全部 force，并精确核对 `41` bonds、`71` angles、`104` torsions。

Membrane qualification still requires area per lipid, membrane thickness, order parameters, ion density, restraint-geometry stability, charged/dispersion contracts, a membrane-specific LRC or validated alternative, APBS maps and cycle sign, and closure of P1-19/P1-22 uncertainty.

#### 5.3.2 Membrane production gates and the 100 ns artifact boundary

The membrane barostat contract uses normal axis `z`, zero surface tension, isotropic XY, free Z, frequency 25, and `MonteCarloMembraneBarostat`. Production gates include a 20 ns tail window, APL drift below `0.2%/ns`, bilayer-thickness drift no greater than `0.05 nm` per tail window, protein-backbone RMSD no greater than `0.30 nm`, transmembrane-tilt drift no greater than `5°`, pocket RMSD no greater than `0.20 nm`, ligand heavy-atom RMSD no greater than `0.25 nm`, cross-repeat SD no greater than `1 kcal/mol`, at least three independent repeats, and a benchmark of at least five ligands with MAE no greater than `1.5 kcal/mol` and absolute outliers no greater than `3 kcal/mol`.

膜 barostat contract 为 normal axis `z`、zero surface tension、XY isotropic、Z free、frequency 25、`MonteCarloMembraneBarostat`。production gates 包括 20 ns tail、APL drift `<0.2%/ns`、thickness drift `≤0.05 nm/tail`、protein backbone RMSD `≤0.30 nm`、tilt drift `≤5°`、pocket RMSD `≤0.20 nm`、ligand heavy-atom RMSD `≤0.25 nm`、cross-repeat SD `≤1 kcal/mol`、至少三个 independent repeats，以及至少五个 ligands、MAE `≤1.5 kcal/mol`、absolute outlier `≤3 kcal/mol` 的 benchmark。

The 100 ns artifact used an advisory membrane-quality gate, had `independent_repeats.performed=false`, and recorded the ligand–environment membrane LRC target as unmet. Its pure-POPC diagnostic APL was `0.645 nm²`, while the protein-containing system measured about `0.5885 nm²` (approximately `8.8%` lower); that comparison is diagnostic because raw APL from a protein-containing membrane cannot be treated as a pure-lipid literature gate.

100 ns artifact 使用 advisory membrane-quality gate，`independent_repeats.performed=false`，且 ligand–environment membrane LRC 的 `target_met=false`。pure-POPC diagnostic APL 为 `0.645 nm²`，含蛋白体系约 `0.5885 nm²`（低约 `8.8%`）；含蛋白体系的 raw APL 不能直接冒充纯脂 literature gate，因此这里只能作诊断。

### 5.4 当前扩展结论 / Current Extension Verdict

charged-ligand **full production cycle**、production-length membrane Stage 0→1→2、membrane-specific dispersion/LRC 与 APBS cycle closure 均未完成。C1/C2/C3 和 neutral membrane smoke 已显著推进工程资格化，但不应在 soluble Atenolol baseline 和 between-run uncertainty 尚未闭合时被写成最终物理验证。

The complete charged-ligand production cycle, production-length membrane Stage 0→1→2, membrane-specific dispersion/LRC, and APBS cycle closure remain incomplete. C1/C2/C3 and the neutral membrane smoke materially advance engineering qualification, but they must not be presented as final physical validation before the soluble Atenolol baseline and between-run uncertainty are closed.

## 6. 独立方法线：MACE-informed DEXP 势能面投影 / Independent Method Line: MACE-Informed DEXP Projection

### 6.1 方法定位：改变 potential family，而不是增加 sampling bias / Method Identity: Changing the Potential Family, Not Adding a Sampling Bias

DEXP 属于 experimental potential family。它改变 ligand–environment nonbonded potential 的解析函数族，因此既不是 Route 2 sampling-only residual，也不能与固定 ACE target ledger 混为同一个 production identity。当前最准确的名称是 **MACE-informed DEXP analytic projection**：MACE 只在离线阶段提供局部参考势能面，生产 Hamiltonian 中只保留解析 DEXP 与 Gaussian electrostatics。

DEXP is an experimental potential family. It changes the analytic form of the ligand–environment nonbonded potential, so it is neither a Route 2 sampling-only residual nor part of the same production identity as the fixed ACE target ledger. The most accurate name is **MACE-informed DEXP analytic projection**: MACE supplies a local reference potential-energy surface only offline, while the production Hamiltonian contains only analytic DEXP and Gaussian electrostatics.

双指数 van der Waals、Gaussian charge 和“无奇点直接 decoupling”已经由 DEGAUSS 提出，不能作为本项目的新颖性。当前项目可能新增的部分，是在真实蛋白–配体口袋中用 MACE 局部参考、anchor-relative perturbations、LOAO、odd/even、force/torque/Hessian 与 production-equivalence audit 选择和验证低维解析核。

Double-exponential van der Waals interactions, Gaussian charges, and singularity-free direct decoupling were introduced by DEGAUSS and are not novel contributions of this project. The possible contribution here is a protein–ligand analytic-projection protocol that uses a local MACE reference, anchor-relative perturbations, LOAO, odd/even decomposition, force/torque/Hessian validation, and a production-equivalence audit to select a low-dimensional analytic kernel.

#### 6.1.1 DEXP 功能模块表 / DEXP Functional-Module Table

**中文**

| 功能模块 | 作用 | 当前状态 |
|---|---|---|
| MACE local reference | 生成 ligand–environment 局部参考能量/力 | 离线教师 |
| anchor/perturbation generator | 构造 20 anchors × 74 rigid perturbations | 已完成 |
| pair-specific DEXP fit | 比较 `(12,6)`、`(14,5)` 与 LOAO optimum | 单体系正向结果 |
| odd/even analysis | 分别检查 gradient 与 curvature | 已完成 |
| force/torque/Hessian validation | 独立验证方向、大小与二阶响应 | 已完成 |
| environment convergence | 比较 radius/cropping 条件 | 已完成 |
| production-equivalence audit | 检查 OpenMM 能量/力和短程 dynamics | 短范围通过 |
| replica dynamics | 检查 V/S/B 多初态是否汇合 | 尚未平衡 |
| correction routes | radial/angular/Gaussian-width 增量修正 | 已关闭/停止 |
| main ABFE integration | 将 DEXP 纳入 production cycle | 未资格化 |

**English**

| Module | Function | Current status |
|---|---|---|
| MACE local reference | Generate local ligand–environment reference energy/force | Offline teacher |
| anchor/perturbation generator | Construct 20 anchors × 74 rigid perturbations | Completed |
| pair-specific DEXP fit | Compare `(12,6)`, `(14,5)`, and the LOAO optimum | Positive in a single system |
| odd/even analysis | Separately inspect gradient and curvature | Completed |
| force/torque/Hessian validation | Independently validate direction, magnitude, and second-order response | Completed |
| environment convergence | Compare radius/cropping conditions | Completed |
| production-equivalence audit | Check OpenMM energy/force and short dynamics | Short-scoped pass |
| replica dynamics | Check convergence across V/S/B initial states | Not equilibrated |
| correction routes | Incremental radial/angular/Gaussian-width corrections | Closed/stopped |
| main ABFE integration | Integrate DEXP into the production cycle | Not qualified |

### 6.2 Pair-specific 双指数核 / Pair-Specific Double-Exponential Kernel

对 ligand–environment 原子对 `(i,j)`，保留 Lorentz–Berthelot 组合律：

\[
\epsilon_{ij}=\sqrt{\epsilon_i\epsilon_j},\qquad
\sigma_{ij}=\frac{\sigma_i+\sigma_j}{2},\qquad
r_{0,ij}=2^{1/6}\sigma_{ij}.
\]

令 `x_{ij}=r_{ij}/r_{0,ij}-1`，well-matching DEXP 为

\[
U_{ij}(r)=\epsilon_{ij}\left[
\frac{\beta}{\alpha-\beta}e^{-\alpha x_{ij}}
-\frac{\alpha}{\alpha-\beta}e^{-\beta x_{ij}}
\right],\qquad \alpha>\beta>0.
\]

它严格满足 `U(r_0)=-ε`、`U'(r_0)=0`、`U''(r_0)=εαβ/r_0²`，并且在 `r→0` 时保持有限能量和有限力。DEXP(12,6) 与 LJ 12–6 只在井底匹配值、斜率与曲率；一个是指数函数族，一个是幂函数族，二者并不等价。对 `(14,5)`，两项系数为 `5/9=0.5556` 与 `14/9=1.5556`，曲率系数为 `70`；`(12,6)` 的曲率系数为 `72`。

The kernel exactly satisfies `U(r_0)=-ε`, `U'(r_0)=0`, and `U''(r_0)=εαβ/r_0²`, while retaining finite energy and force as `r→0`. DEXP(12,6) matches LJ 12–6 only in value, slope, and curvature at the well minimum; the exponential and inverse-power function families are not equivalent. For `(14,5)`, the two coefficients are `5/9=0.5556` and `14/9=1.5556`, with curvature coefficient `70`; `(12,6)` has curvature coefficient `72`.

The present code contract uses a DEXP cutoff of `0.70 nm`, switch width `0.20 nm`, and Gaussian-Coulomb width `0.10 nm`. It deliberately applies no DEXP LRC because the corresponding formula has not yet been validated. Consequently, selecting `--potential dexp` does not inherit the softcore-LJ LRC v3 qualification and cannot be treated as a drop-in production replacement.

当前代码合同使用 DEXP cutoff `0.70 nm`、switch width `0.20 nm`、Gaussian-Coulomb width `0.10 nm`。DEXP 暂不添加 LRC，因为对应公式尚未验证。因此选择 `--potential dexp` 不会自动继承 softcore-LJ LRC v3 的资格，也不能视为 production drop-in replacement。

### 6.3 MACE reference 与 20×74 perturbation cloud / MACE Reference and the 20×74 Perturbation Cloud

离线参考相互作用定义为

\[
E_{\mathrm{MACE,int}}=E_{\mathrm{MACE}}(L+E)-E_{\mathrm{MACE}}(L)-E_{\mathrm{MACE}}(E).
\]

从平衡轨迹末段选择 20 个 anchors。每个 anchor 对 ligand 构造 74 个不经 relax/minimize 的刚体扰动：平移 `±0.005, ±0.01, ±0.02, ±0.04 nm`，方向覆盖三个主惯性轴与随机方向；转动 `±0.5°, ±1.5°, ±3.0°`，绕三个主惯性轴。总样本数为 `20×74=1480`。

Twenty anchors were selected from the end of an equilibrated trajectory. For each anchor, 74 rigid-body ligand perturbations were generated without relaxation or minimization: translations of `±0.005, ±0.01, ±0.02, and ±0.04 nm` along the three principal inertia axes and random directions, and rotations of `±0.5°, ±1.5°, and ±3.0°` about the three principal axes. The complete dataset contains `20×74=1480` samples.

每个 anchor 内使用 `ΔE=E(x+δ)-E(x)` 消除绝对能量零点。七个 perturbation bins 等权，`translation@0.04 nm` 额外乘 `0.5`，防止最大平移支配目标。交叉验证严格按 anchor 进行 20-fold leave-one-anchor-out；同一 anchor 的 74 条记录不能当作 74 个独立样本。

Within each anchor, `ΔE=E(x+δ)-E(x)` removes the arbitrary absolute-energy zero. Seven perturbation bins receive equal weight, with an additional factor of `0.5` for `translation@0.04 nm` to prevent the largest displacement from dominating. Cross-validation uses strict 20-fold leave-one-anchor-out grouping; the 74 records from one anchor are not treated as 74 independent samples.

### 6.4 Odd/even、力与 Hessian 的验证逻辑 / Odd/Even, Force, and Hessian Validation Logic

对成对扰动 `±δ`：

\[
E_{\mathrm{odd}}(\delta)=\frac{E(+\delta)-E(-\delta)}{2},\qquad
E_{\mathrm{even}}(\delta)=\frac{E(+\delta)+E(-\delta)}{2}.
\]

odd 主要反映局部梯度和平衡位置错配；even 主要反映曲率与更高偶数阶形状。跨幅度拟合 `e_odd(δ)=gδ+c_3δ³` 与 `e_even(δ)=hδ²/2+c_4δ⁴`，用于估计局部 gradient 与完整对称 `3×3` translation Hessian。只看 cosine similarity 不够，必须同时检查 direction、magnitude、held-out residual norm、Hessian Frobenius residual 和 eigenvalue RMSE。

The odd component primarily reflects local gradients and equilibrium-position mismatch, whereas the even component reflects curvature and higher even-order shape. Fits across amplitudes, `e_odd(δ)=gδ+c_3δ³` and `e_even(δ)=hδ²/2+c_4δ⁴`, estimate the local gradient and complete symmetric `3×3` translational Hessian. Cosine similarity alone is insufficient; direction, magnitude, held-out residual norm, Hessian Frobenius residual, and eigenvalue RMSE must all be examined.

### 6.5 解析基线与参数识别 / Analytic Baseline and Parameter Identification

在 DEXP(12,6) 基线下，1480 条样本的总 residual RMSE 为 `7.884 kJ/mol`，bias 为 `2.649 kJ/mol`，`corr(ΔU_DEXP,ΔE_target)=0.944`。旋转样本 `n=360`，RMSE/bias 为 `4.418/1.313 kJ/mol`；平移样本 `n=1120`，为 `8.710/3.078 kJ/mol`。残差随扰动幅度平滑增加：最小扰动约 `1 kJ/mol`，最大 `0.04 nm` 平移达到 `15.925 kJ/mol`。

For the DEXP(12,6) baseline, the 1480 samples gave a total residual RMSE of `7.884 kJ/mol`, bias `2.649 kJ/mol`, and `corr(ΔU_DEXP,ΔE_target)=0.944`. The 360 rotational samples had RMSE/bias `4.418/1.313 kJ/mol`; the 1120 translational samples had `8.710/3.078 kJ/mol`. Residuals grew smoothly with perturbation amplitude, from approximately `1 kJ/mol` at the smallest perturbations to `15.925 kJ/mol` at `0.04 nm` translation.

逐幅度数据表明误差随局部位移连续增长，而不是由少数随机 outliers 主导；完整幅度表登记在附录 C。因此当前 benchmark 的 support domain 仅限结合口袋附近的小扰动，不能外推到 ligand 完全离开口袋后的全局势能面。

Amplitude-resolved data show smooth error growth rather than domination by a few random outliers; the full table is registered in Appendix C. The support domain is therefore restricted to small perturbations near the bound pocket and cannot be extrapolated to the global potential-energy surface after complete ligand displacement.

20 个 LOAO folds 中，19 折选择 `(14,5)`，1 折选择 `(13,6)`；估计为 `α=13.95±0.22, β=5.05±0.22`。held-out weighted RMSE 从 `(12,6)` 的 `5.823` 降至 `5.051 kJ/mol`，改善约 `13%`；全数据 `(14,5)` 为 `5.215 kJ/mol`。

Nineteen of twenty LOAO folds selected `(14,5)`, while one selected `(13,6)`, yielding `α=13.95±0.22, β=5.05±0.22`. Held-out weighted RMSE decreased from `5.823 kJ/mol` for `(12,6)` to `5.051 kJ/mol`, an improvement of approximately `13%`; the full-data `(14,5)` value was `5.215 kJ/mol`.

密网格确认低误差区是狭长 basin，主要约束 `α+β≈19.004`，而不是证明 `(14,5)` 是普适常数；在 even 指标上 `(14,5)` 与连续最优点的差异仅 `3.1%` 且 bootstrap CI 跨零，因此保留可解释的整数默认值。扫描范围、步长和完整 basin 统计下沉至附录 C。

The dense scan identified an elongated low-error basin governed mainly by `α+β≈19.004`, not a universal `(14,5)` constant. On the even metric, `(14,5)` differed from the continuous optimum by only `3.1%`, with a bootstrap confidence interval crossing zero; the interpretable integer default was therefore retained. Full scan limits, spacing, and basin statistics are moved to Appendix C.

### 6.6 LJ、DEXP(12,6) 与 DEXP(14,5) 的横向比较 / Horizontal Comparison of LJ, DEXP(12,6), and DEXP(14,5)

**中文**

| 核 | 平移奇部 | 平移偶部 | 旋转奇部 | 旋转偶部 |
|---|---:|---:|---:|---:|
| K0 LJ | `15.624` | `21.401` | `5.602` | `6.104` |
| K1 DEXP(12,6) | `6.611` | `5.670` | `3.732` | `2.365` |
| K2 DEXP(14,5) | `6.733` | `3.142` | `3.552` | `1.390` |

**English**

| Kernel | Translation odd | Translation even | Rotation odd | Rotation even |
|---|---:|---:|---:|---:|
| K0 LJ | `15.624` | `21.401` | `5.602` | `6.104` |
| K1 DEXP(12,6) | `6.611` | `5.670` | `3.732` | `2.365` |
| K2 DEXP(14,5) | `6.733` | `3.142` | `3.552` | `1.390` |

20 个 anchors 的总体胜场为 `LJ:K1:K2=0:3:17`。K1 与 K2 的 odd 胜场为 `10:10`，even 为 `1:19`，说明 LJ→DEXP 改善同时出现在 gradient 与 curvature，而 DEXP 内部 `(12,6)→(14,5)` 的主要额外收益集中在 even/Hessian。

Across 20 anchors, overall wins were `LJ:K1:K2=0:3:17`. K1 versus K2 wins were `10:10` for odd and `1:19` for even, showing that the LJ→DEXP family change improves both gradient and curvature, whereas the additional `(12,6)→(14,5)` benefit is concentrated in even/Hessian behavior.

weighted/median/10%-trimmed 排序保持不变；MACE conditional-mean profile 落在 pooled SEM 内的 bins 为 `LJ 1/7, K1 2/7, K2 6/7`。switch 前后差异只有 `0.01–0.02 kJ/mol`；按最小接触距离分层或只保留最小幅扰动，DEXP 相对 LJ 的排序也不改变。

Weighted, median, and 10%-trimmed rankings remained unchanged. The number of MACE conditional-mean-profile bins within pooled SEM was `1/7` for LJ, `2/7` for K1, and `6/7` for K2. Switching changed results by only `0.01–0.02 kJ/mol`; stratification by minimum contact distance and restriction to the smallest perturbations also preserved the DEXP-over-LJ ordering.

### 6.7 Force、torque 与 Hessian 的独立证据 / Independent Force, Torque, and Hessian Evidence

force cosine 为 `0.754/0.779/0.779`，torque cosine 为 `0.648/0.622/0.632`，单独看 cosine 无法区分三个 kernels。combined direction-and-magnitude residual 明确区分 DEXP 与 LJ：K1−K0 为 `−188.1`，95% CI `[−320,−68.6]`；K2−K0 为 `−190.8`，`[−301.7,−82.3]`；K2−K1 为 `−2.7`，`[−27,22.5]`。

Force cosine values were `0.754/0.779/0.779` and torque cosines `0.648/0.622/0.632`, so cosine alone did not distinguish the kernels. The combined direction-and-magnitude residual clearly separated DEXP from LJ: K1−K0 was `−188.1`, 95% CI `[−320,−68.6]`; K2−K0 was `−190.8`, `[−301.7,−82.3]`; K2−K1 was `−2.7`, `[−27,22.5]`.

**中文**

| 指标 | K0 LJ | K1 DEXP(12,6) | K2 DEXP(14,5) |
|---|---:|---:|---:|
| 留出力 RMSE | `417.8` | `289.0` | `295.6` |
| 留出力相关系数 | `0.672` | `0.714` | `0.709` |
| Hessian Frobenius 残差 | `56598` | `24823` | `16687` |
| Hessian 特征值 RMSE | `31185` | `12917` | `7189` |

**English**

| Metric | K0 LJ | K1 DEXP(12,6) | K2 DEXP(14,5) |
|---|---:|---:|---:|
| Held-out force RMSE | `417.8` | `289.0` | `295.6` |
| Held-out force correlation | `0.672` | `0.714` | `0.709` |
| Hessian Frobenius residual | `56598` | `24823` | `16687` |
| Hessian eigenvalue RMSE | `31185` | `12917` | `7189` |

DEXP 相对 LJ 的 held-out force RMSE 约降低 `30%`，而两种 DEXP 在力上统计打平。完整 Hessian 则稳定区分 `(14,5)` 与 `(12,6)`：Frobenius 与 eigenvalue 的 anchor 胜场均为 `0:1:19`，K2−K1 的 bootstrap CI 不跨零。

DEXP reduced held-out force RMSE by approximately `30%` relative to LJ, while the two DEXP kernels were statistically tied in force. The complete Hessian consistently separated `(14,5)` from `(12,6)`: anchor wins for both Frobenius and eigenvalue metrics were `0:1:19`, and the K2−K1 bootstrap confidence intervals did not cross zero.

### 6.8 Production equivalence、稳定性与环境收敛 / Production Equivalence, Stability, and Environmental Convergence

真实 `73,536`-atom system 使用 `(14,5)` 后能量和力有限，300 K Langevin `0.4 ps` 无 NaN 或爆炸；总势能由 `−997944.8` 变为 `−998240.6 kJ/mol`。有限差分力相对误差约 `9×10⁻7–1×10⁻5`，低于 `10⁻4` 门。

The real `73,536`-atom system retained finite energies and forces with `(14,5)` and showed no NaN or instability during `0.4 ps` of 300 K Langevin dynamics; total potential energy changed from `−997944.8` to `−998240.6 kJ/mol`. Relative finite-difference force errors were approximately `9×10⁻7–1×10⁻5`, below the `10⁻4` gate.

离线 baseline 缺少 `0.50–0.70 nm` quintic switch 会造成绝对能量约 `8–9 kJ/mol` 差异，但对实际使用的 `ΔU(pert−anchor)` 最大影响仅 `0.385 kJ/mol`，主要幅度为 `0.02–0.24 kJ/mol`，odd/even 结果变化小于 `1%`。

Omission of the `0.50–0.70 nm` quintic switch in the offline baseline caused an absolute-energy difference of approximately `8–9 kJ/mol`, but changed the actual `ΔU(pert−anchor)` values by at most `0.385 kJ/mol`, typically `0.02–0.24 kJ/mol`, with less than `1%` change in odd/even metrics.

环境收敛覆盖 `5 anchors×4 radii×2 cropping modes=520` 次 decomposition、约 1560 次 MACE forwards。半径从 `0.50` 到 `0.90 nm`，8 个 radius/cropping conditions 中 K2 都是最优，排序稳定率 `100%`；`49/60` directions 的 odd 符号稳定，翻转主要发生在 `|odd|<0.3–0.6 kJ/mol` 的近零方向。

Environmental convergence covered `5 anchors×4 radii×2 cropping modes=520` decompositions, approximately 1560 MACE forward passes. From `0.50` to `0.90 nm`, K2 ranked best in all eight radius/cropping conditions, a `100%` ranking-stability rate; odd signs were stable in `49/60` directions, with flips concentrated in near-zero directions where `|odd|<0.3–0.6 kJ/mol`.

### 6.9 被关闭的修正路线 / Closed Correction Routes

固定 `(14,5)` 后，donor–acceptor/fallback radial corrections 的 OOF weighted RMSE 为 `5.215/5.216/5.199 kJ/mol`，最好只改善 `0.3–0.6%`；ridge 全扫描的最好点也只有 `1.1%`，且 rotation odd 恶化。角度诊断的最高跨-anchor sign stability 为 `70%`，低于 `85%` 门；Gaussian-width/charge-penetration 诊断也只有 `50–70%` 稳定性。因此 contact-specific radial、angular 和 Gaussian-width 三条修正均正式关闭。

With `(14,5)` fixed, donor–acceptor/fallback radial corrections produced OOF weighted RMSE values of `5.215/5.216/5.199 kJ/mol`, an improvement of only `0.3–0.6%`; the best point in the complete ridge scan improved only `1.1%` and worsened rotational odd error. The highest cross-anchor sign stability for angular diagnostics was `70%`, below the `85%` gate, while Gaussian-width/charge-penetration diagnostics reached only `50–70%`. Contact-specific radial, angular, and Gaussian-width corrections were therefore formally closed.

### 6.10 Replica dynamics、V/S/B 多初态与不能越界的结论 / Replica Dynamics, V/S/B Initial States, and the Claim Boundary

早期 `3 conditions×5 replicas×2 ns` 轨迹在 committed-state 分析中 `15/15` 都被标记为 not equilibrated。随后完成 `3 conditions×3 initial states(V/S/B)×2 replicas×5 ns=90 ns`。original 在三个初态下均以 V 为主 (`56–88%`)，而 DEXP(12,6) 和 DEXP(14,5) 从 V 初态都漂移到 N 为主 (`63%/68%`)，从 S/B 初态则以 S 为主 (`38–51%`)。

All `15/15` trajectories in the initial `3 conditions×5 replicas×2 ns` study were classified as not equilibrated by committed-state analysis. A subsequent `3 conditions×3 initial states(V/S/B)×2 replicas×5 ns=90 ns` study found original to remain V-dominant (`56–88%`) across initial states, whereas DEXP(12,6) and DEXP(14,5) both shifted from V starts toward N (`63%/68%`) and from S/B starts toward S (`38–51%`).

因此旧的“(12,6) 更接近 original”结论被推翻。两种 DEXP 的动力学差异远小于它们共同相对 original 的差异，且没有跨初态汇合。这不推翻局部解析投影 benchmark，但阻止了 production superiority、结合模式机制和 equilibrium occupancy 的主张。

The earlier claim that `(12,6)` was closer to original was therefore overturned. The dynamical difference between the two DEXP kernels was much smaller than their common difference from original, and convergence across initial states was absent. This does not invalidate the local analytic-projection benchmark, but it blocks claims of production superiority, binding-mode mechanism, or equilibrium occupancy.

### 6.11 当前判决与冻结的多体系协议 / Current Verdict and Frozen Multi-System Protocol

当前 DEXP 判决为 `SINGLE-SYSTEM POSITIVE ANALYTIC-PROJECTION SIGNAL / DYNAMICS NOT EQUILIBRATED / NOT_PRODUCTION_MERGED`。如果在单体系证据下必须选择进一步测试的默认核，`(14,5)` 是唯一仍有独立正证据的整数代表；但它不是已经建立的 universal magic number。

The current DEXP verdict is `SINGLE-SYSTEM POSITIVE ANALYTIC-PROJECTION SIGNAL / DYNAMICS NOT EQUILIBRATED / NOT_PRODUCTION_MERGED`. If one default kernel must be selected for further testing on the present single-system evidence, `(14,5)` is the only integer representative with independent positive support; it is not an established universal magic number.

下一阶段应冻结流程并扩展到 `8–15` 个化学多样体系：每体系从 20 anchors 起步；保持同一 perturbation grid；统一比较 LJ、DEXP(12,6) 与 DEXP(14,5)/per-system LOAO optimum；必须完成逐幅度、逐 anchor、robust、switch、conditional mean、distance strata、small-amplitude、force 和 Hessian 复核。预注册 H1 为至少 75% 体系中 DEXP family 的 odd/even 均优于 LJ；H2 为 `(14,5)` 的 even/Hessian 优势可迁移；若 H1 成立而 H2 不成立，则接受“DEXP family 可迁移、最优 α/β 体系依赖”。

The next stage should freeze the protocol and extend it to `8–15` chemically diverse systems: begin with 20 anchors per system; preserve the perturbation grid; compare LJ, DEXP(12,6), and DEXP(14,5) or the per-system LOAO optimum; and complete per-amplitude, per-anchor, robust, switching, conditional-mean, distance-stratified, small-amplitude, force, and Hessian checks. Preregistered H1 requires DEXP to outperform LJ in both odd and even metrics for at least 75% of systems. H2 requires transfer of the `(14,5)` even/Hessian advantage. If H1 passes while H2 fails, the accepted conclusion is that the DEXP family transfers but the optimal α/β values are system dependent.

`r_0` scaling、参数来源差异、逐幅度 residual、密网格 basin 与 DEXP artifact 年表完整登记在附录 C；它们用于 provenance 和复核，不改变上述单体系判决。

The complete `r_0` scaling diagnostic, parameter-source distinctions, amplitude-resolved residuals, dense-grid basin, and DEXP artifact chronology are registered in Appendix C for provenance and audit; they do not alter the single-system verdict above.

## 7. 研究路线、导师决策与下一阶段计划 / Research Routes, Advisor Decisions, and Next-Stage Plan

### 7.1 为什么必须区分 target 与 sampling

For a fixed physical target state,

\[
p_k(R)=Z_k^{-1}\exp[-\beta U_k(R)].
\]

Route 2 generates samples from

\[
q_{k,\theta}(R)=\widetilde Z_{k,\theta}^{-1}
\exp\{-\beta[U_k(R)+B_{\theta,k}(R)]\}.
\]

目标仍然是 (p_k)，而不是 (q_{k,\theta})。(B_{\theta,k}) 的实际值、state identity 和所有并行 bias 必须被记录，以便最终 estimator 严格恢复 (U_k) 的 thermodynamics。

The target remains (p_k), not (q_{k,\theta}). The realized (B_{\theta,k}), state identity, and every simultaneous bias must be recorded so that the final estimator rigorously recovers the thermodynamics of (U_k).

因为 (U_k) 和 (U_{k+1}) 不变，Route 2 不会改变 target distributions 的数学 overlap。它可能改善的是被实际访问的 bridge configurations、mixing、round trips、reweighted overlap efficiency 和 ESS/GPU-hour。

Because (U_k) and (U_{k+1}) remain unchanged, Route 2 does not change the mathematical overlap of the target distributions. It may improve visited bridge configurations, mixing, round trips, reweighted-overlap efficiency, and ESS/GPU-hour.

### 7.2 Route 2 的完整冻结合同

Route 2 production qualification requires all of the following to be frozen before production:

- representation/model parameters (phi_*);
- state-amplitude schedule (A_{w,k});
- arm/window-specific (f_{w,k});
- unit convention for `(B_\phi)`, `(\beta B_\phi)`, and `(f_k)`;
- state mapping, gauge convention, and protocol fingerprint;
- target/base/bias/WCA/LRC ledger schema;
- seed family and stopping rules.

若 production 中继续训练 representation、重新中心化 residual、调用 `update_weights()` 或修改 (f_k)，普通 static TMBAR/MBAR 语义不再自动成立。

If the representation is trained further, the residual is recentered, `update_weights()` is called, or (f_k) changes during production, ordinary static TMBAR/MBAR semantics no longer apply automatically.

Route 2 的主要风险不是 endpoint drift，而是 importance-weight degeneracy：bias 可能增加构象跳跃，却使回到 target 的权重更加尖锐，最终降低 ESS。因此 mobility、occupancy 和 transition counts 必须与 reweighted ESS、uncertainty 和 GPU cost 同时报告。

The main Route 2 risk is not endpoint drift but importance-weight degeneracy: a bias may increase conformational transitions while sharpening the weights required to recover the target, thereby reducing ESS. Mobility, occupancy, and transition counts must therefore be reported together with reweighted ESS, uncertainty, and GPU cost.

### 7.3 Route 1 的数学身份

Route 1 defines

\[
H(R;s)=H_0(R;\xi(s))+\sum_a c_a(s)V_a(R),
\]

with a structural endpoint constraint such as

\[
c_a(s)=s(1-s)\widetilde c_a(s).
\]

For controls that must be monotonic, a softplus-integrated parameterization can be used:

\[
q(s)=\frac{\int_0^s\operatorname{softplus}[r(t)]dt}
{\int_0^1\operatorname{softplus}[r(t)]dt}.
\]

这保证 endpoint value 或 monotonicity，但不保证中间 Hamiltonian 物理安全。必须另外约束 coefficient、force、energy、support domain、normalizability、PBC、neighbor-list identity 和新 metastable basins。

This guarantees endpoint values or monotonicity, but not physical safety of the intermediate Hamiltonians. Coefficients, forces, energies, support domain, normalizability, PBC behavior, neighbor-list identity, and newly introduced metastable basins require separate constraints.

### 7.4 Frozen MACE 在 Route 1 中能做什么

Frozen MACE should be treated first as a representation

\[
\Phi_{\mathrm{MACE}}:R\mapsto z(R),
\]

not as a replacement for the MM endpoint Hamiltonians.

它可以帮助诊断 ligand、ion、water shell 和 binding-site side chains 的 many-body structural change。可比较的量包括 latent distributions、MMD、TICA/VAMP slow modes、autocorrelation、transition counts 和与 reduced-energy overlap 的关系。

It can diagnose many-body structural changes involving the ligand, ions, water shells, and binding-site side chains. Candidate observables include latent distributions, MMD, TICA/VAMP slow modes, autocorrelation, transition counts, and their relationship to reduced-energy overlap.

仅比较 latent mean 不足以区分多峰与单峰分布。MMD 也不是自动可靠：descriptor normalization、kernel bandwidth、correlated-trajectory bias、block bootstrap 和 role/PBC mapping 必须明确。

Comparing only latent means cannot distinguish multimodal from unimodal distributions. MMD is not automatically reliable either: descriptor normalization, kernel bandwidth, correlated-trajectory bias, block bootstrap, and role/PBC mapping must be explicit.

### 7.5 Neural operator 的证据门

A proposed convergence landscape may contain

\[
\mathcal C(\xi)=\{G_{ab}(\xi),S_{ab}(\xi),K(\xi),ESS(\xi),\ldots\},
\]

where (G) is thermodynamic, (S) structural, and (K) kinetic.

这些量没有天然相同的单位。任何加权和都必须预注册 normalization、weights、uncertainty 和 trust region。否则 optimizer 只是在追逐 representation scale。

These quantities do not share natural units. Any weighted combination must preregister normalization, weights, uncertainty, and a trust region; otherwise the optimizer merely follows representation scale.

单体系的 12–23 个 windows 加少量 probes 只足以支持局部 interpolation，不能证明 discretization-independent operator generalization。第一阶段应优先用 spline 或 Gaussian process；只有跨多个 paths、ligands 和 probe layouts 的 held-out tests 才能证明 GNO/operator 必要。

A single system with 12–23 windows and a few probes supports only local interpolation, not discretization-independent operator generalization. The first stage should prioritize splines or Gaussian processes; only held-out tests across multiple paths, ligands, and probe layouts can establish the need for a GNO/operator.

### 7.6 Route 1 probe reweighting 的限制

Simple probe weights

\[
w_n\propto\exp[-\beta(H_{\xi'}(R_n)-H_{\xi}(R_n))]
\]

are valid only when the source distribution is correctly represented by (H_\xi).

Current IBS samples a mixture with frozen (f_k), Group-1 bias, WCA, and possibly additional histories. A valid probe must use the complete source distribution, preserve target/base/LRC ledgers, and estimate correlated-sample ESS. When ESS is inadequate, real short MD is required; operator extrapolation cannot substitute for missing support.

### 7.7 `V_soft` 符号问题

If the proposed auxiliary term is

\[
H=H_{baseline}+c_{soft}V_{soft}(r),\qquad
V_{soft}(r)=A\exp[-(r-r_0)^2/(2\sigma^2)],
\]

then positive (A) and positive (c_{soft}) raise the energy in that region. This does not automatically “soften a repulsive barrier.” The sign and intended physical action must be explicit, and tests must exclude an artificial attractive well, collapse, or new metastability.

### 7.8 Classical EDS

Classical EDS constructs a reference Hamiltonian

\[
V_R(R;s,\mathbf E^R)
=-\frac{1}{\beta s}
\log\sum_i\exp\{-\beta s[V_i(R)-E_i^R]\}.
\]

The offsets (E_i^R) balance state contributions, while (s) controls smoothing. Endpoint free energies can be recovered from the reference ensemble, for example

\[
\Delta G_{BA}=-\beta^{-1}\log
\frac{\langle e^{-\beta(V_B-V_R)}\rangle_R}
{\langle e^{-\beta(V_A-V_R)}\rangle_R}.
\]

因此 classical EDS 既是 auxiliary/reference sampling，也是 Hamiltonian-level envelope construction。它不应被简化为普通 linear interpolation，也不应被排除在 enhanced sampling 之外。

Classical EDS is both auxiliary/reference sampling and a Hamiltonian-level envelope construction. It should be reduced neither to ordinary linear interpolation nor excluded from enhanced sampling.

EDS parameters are not free. Poor offsets can suppress one endpoint, while an inappropriate smoothing parameter can create an unphysical global minimum or poor reweightability. EDS still requires overlap, parameter qualification, and independent repeats.

### 7.9 λ-EDS

λ-EDS uses the enveloping construction to define intermediate states:

\[
V_{\lambda\mathrm{-EDS}}(R)=
-\frac{1}{\beta s}\log\left[
(1-\lambda)e^{-\beta s(V_A-E_A)}
+\lambda e^{-\beta s(V_B-E_B)}
\right].
\]

At (lambda=0,1), the physical endpoints are recovered up to additive offsets. For (0<\lambda<1), the intermediate Hamiltonians are redesigned.

λ-EDS is therefore closer to Route 1 than Route 2. Both treat intermediate Hamiltonians as design objects. The potential novelty of Route 1 cannot be “fixed endpoints with modifiable intermediates”; it must be demonstrated as added value from many-body structural diagnostics, continuous control-space prediction, or superior held-out path optimization.

### 7.10 完整比较矩阵

**中文**

| 维度 | Classical EDS | λ-EDS | Route 1 | Route 2 |
|---|---|---|---|---|
| 主要对象 | 辅助参考 Hamiltonian | EDS 中间耦合 | 学习得到的目标路径 | 学习得到的采样层 |
| 物理端点 | 在 gauge/offset 下保留 | 保留 | 硬端点合同 | 目标状态不变 |
| 中间目标 | 不要求固定 ladder | 重新设计 | 重新设计 | 不变 |
| 表示 | 解析端点能量 | 解析端点能量 | Frozen MACE/operator 可选 | MACE/MLP/local residual 可选 |
| 最终恢复 | 参考系 → 端点 | EDS 状态 → 物理 ΔG | 实际新状态 → 端点 ΔG | 带偏置样本 → 固定 `U_k` |
| 主要风险 | offset/smoothing 与参考混合 | 重设计路径上的 support | 不安全路径与选择偏差 | 权重退化与成本 |
| 直接基线 | 常规 EDS/RE-EDS | EI 与热力学路径 | EI、热力学长度、EDS/λ-EDS | 无 learned bias 的固定路径 |
| 当前项目状态 | 文献/控制 | Route 1 所需控制 | 尚未推广的原型 | EXP-029/030 主线 |

**English**

| Dimension | Classical EDS | λ-EDS | Route 1 | Route 2 |
|---|---|---|---|---|
| Primary object | Auxiliary reference Hamiltonian | Intermediate EDS coupling | Learned target path | Learned sampling layer |
| Physical endpoints | Retained up to gauge/offset | Retained | Hard endpoint contract | Target states unchanged |
| Intermediate targets | No fixed ladder required | Redesigned | Redesigned | Unchanged |
| Representation | Analytic endpoint energies | Analytic endpoint energies | Frozen MACE/operator optional | MACE/MLP/local residual optional |
| Final recovery | Reference → endpoints | EDS states → physical ΔG | Actual new states → endpoint ΔG | Biased samples → fixed `U_k` |
| Main risk | Offset/smoothing and reference mixing | Support along redesigned path | Unsafe path and selection bias | Weight degeneracy and cost |
| Direct baseline | Conventional EDS/RE-EDS | EI and thermodynamic paths | EI, thermodynamic length, EDS/λ-EDS | Fixed path without learned bias |
| Current project status | Literature/control | Required Route-1 control | Unpromoted prototype | EXP-029/030 mainline |

### 7.11 Route 1 的最小实验顺序

1. Diagnostic-only Frozen MACE: do not alter any Hamiltonian.  
2. Test whether latent/MMD/kinetic metrics predict known ion, water, side-chain, or ligand transitions.  
3. Introduce only one bounded auxiliary basis in a low-dimensional control space.  
4. Compare spline/GP with the original path and thermodynamic-length path.  
5. Add EDS/λ-EDS as direct controls.  
6. Separate path-selection data from final held-out evaluation repeats.  
7. Run energy/force/PBC/endpoint/support and real-cost gates.  
8. Use at least three independent seeds and report ESS/GPU-hour and between-run variance.  
9. Only then test a neural operator and cross-ligand transfer.

### 7.12 Route 2 的最小实验顺序

1. Freeze the EXP-020/026 residual artifact and CUDA binary identity.  
2. Complete a tiny harness/state-machine smoke test.  
3. Independently calibrate and freeze baseline (f_b).  
4. Independently calibrate and freeze candidate (f_c(\phi_*)).  
5. Run all six windows and three paired repeats.  
6. Record all target, bias, base, WCA, LRC, occupancy, and rescue histories.  
7. Require ΔG consistency and Stage-2 TMBAR qualification.  
8. Compare ITT ESS/GPU-hour including calibration and failures.  
9. Promote, stop, or declare inconclusive according to the preregistered gate.

### 7.13 分层验收、可支持主张与导师决策 / Layered Qualification, Defensible Claims, and Advisor Decisions

#### 7.13.1 共同验收门 / Common Qualification Gates

1. 身份/provenance：冻结 topology、positions、box、target policy、state mapping、hash、seed 和 schema。  
2. 代数/ledger：target、sampling、base、WCA、LRC、residual/path 逐帧闭合。  
3. correctness：energy/force、finite difference、PBC、invariance、endpoint 与 conservative force。  
4. cost：calibration、freeze、rescue、production、ledger、neighbor list、kernel launch 与 synchronization 全部计入 ITT。  
5. dynamics：NaN、temperature、drift、RMS displacement、long-run growth、resume 和 seeds。  
6. estimator：完整 energy matrix、actual bias history、coverage、decorrelated ESS、endpoint uncertainty 和 block uncertainty。  
7. utility：至少三个 paired repeats；ΔG consistency、uncertainty 与 ITT utility 同时通过。

1. Identity/provenance: freeze topology, positions, box, target policy, state mapping, hashes, seeds, and schema.  
2. Algebra/ledger: close target, sampling, base, WCA, LRC, and residual/path records frame by frame.  
3. Correctness: energy/force, finite differences, PBC, invariance, endpoints, and conservative forces.  
4. Cost: include calibration, freeze, rescue, production, ledger, neighbor lists, kernel launches, and synchronization in ITT.  
5. Dynamics: check NaNs, temperature, drift, RMS displacement, long-run growth, resume, and seeds.  
6. Estimator: verify the complete energy matrix, actual bias history, coverage, decorrelated ESS, endpoint uncertainty, and block uncertainty.  
7. Utility: at least three paired repeats, with simultaneous ΔG consistency, uncertainty, and ITT utility passes.

#### 7.13.2 当前可支持的创新主张 / Currently Defensible Contributions

1. **Auditable dual-λ ABFE pipeline**: explicit identities for decharging, vanishing, restraints, LRC, resume, and the final thermodynamic cycle.  
2. **IBS learning/freeze/production separation**: adaptive work determines a frozen `(f_k)`, while production uses an immutable distribution.  
3. **Local TMBAR covariance chain**: mixture-aware local analysis and cross-window stitching.  
4. **Non-mutating recovery**: failed windows are extended under the same distribution or replaced by independent immutable rescue ensembles.  
5. **Layered qualification methodology**: physical identity, correctness, cost, dynamics, estimator validity, and final utility are separate gates.  
6. **Explicit separation of sampling-only and target-path semantics**.  
7. **MACE-informed DEXP analytic projection as a single-system method signal**, with universal and production claims explicitly withheld.

#### 7.13.3 当前不能支持的主张 / Currently Unsupported Claims

- a final trustworthy Atenolol ABFE;
- production promotion of a neural residual;
- universal superiority of DEXP over LJ;
- validated full cycles for charged ligands or membranes;
- APBS or LRC solved for arbitrary membrane systems;
- automatic compatibility of v29 with all historical artifacts;
- superiority of Route 2 over EDS or Route 1 before EXP-029;
- neural-operator superiority before simpler path models and EDS controls.

#### 7.13.4 更新后的资源优先级 / Updated Resource Priorities

**中文**

| 优先级 | 任务 | 交付物 | 成功标准 |
|---:|---|---|---|
| 1 | 冻结 environment/input/hash/seed manifest | 可复现运行包 | 从零复现 |
| 1 | 当前 estimator v2 reanalysis | Stage 1 BAR + FD-TI 报告 | source/analysis identity 对齐 |
| 1 | 独立重复 | 至少 2 个新的，理想为 3+ | 已估计 between-run variance |
| 1 | moving-block/bootstrap | 时间相关性报告 | P1-22 已闭合 |
| 1 | Boresch sensitivity | unclipped/alternative-anchor comparison | restraint result 稳定 |
| 2 | EXP-029 | six windows × three paired repeats | ΔG 与 ITT utility gates 通过 |
| 3 | representation ablation | local residual/CV/MACE/MLP/operator | 匹配预算的 held-out comparison |
| 4 | Route 1 pilot | original/TL/EDS/λ-EDS/learned paths | held-out path benefit |
| 5 | charged/membrane extensions | full cycles 与 APBS/dispersion evidence | independent closure |
| 5 | 8–15-system benchmark | 多样化学与环境 | accuracy、uncertainty、failures、GPU-hour |

**English**

| Priority | Task | Deliverable | Success criterion |
|---:|---|---|---|
| 1 | Freeze environment/input/hash/seed manifest | Reproducible run package | Reproduction from zero |
| 1 | Current estimator v2 reanalysis | Stage 1 BAR + FD-TI report | Source/analysis identity aligned |
| 1 | Independent repeats | At least 2 new, ideally 3+ | Between-run variance estimated |
| 1 | Moving-block/bootstrap | Time-correlation report | P1-22 closed |
| 1 | Boresch sensitivity | Unclipped/alternative-anchor comparison | Restraint result stable |
| 2 | EXP-029 | Six windows × three paired repeats | ΔG and ITT utility gates pass |
| 3 | Representation ablation | Local residual/CV/MACE/MLP/operator | Matched-budget held-out comparison |
| 4 | Route 1 pilot | Original/TL/EDS/λ-EDS/learned paths | Held-out path benefit |
| 5 | Charged/membrane extensions | Full cycles and APBS/dispersion evidence | Independent closure |
| 5 | 8–15-system benchmark | Diverse chemistry and environments | Accuracy, uncertainty, failures, GPU-hour |

#### 7.13.5 建议与导师讨论的明确问题 / Explicit Questions for Advisor Discussion

1. Should the paper emphasize an auditable IBS-ABFE framework and its failure boundaries rather than rush to report one Atenolol scalar?  
2. What GPU budget should be reserved for same-protocol independent repeats and an 8–15-system benchmark?  
3. Is EXP-029 accepted as the sole decisive experiment for the current learned residual?  
4. Are three paired repeats and the proposed ITT utility gate sufficiently rigorous?  
5. Should Frozen MACE remain a representation ablation until the lightweight residual passes?  
6. Should Route 1 be developed as a separate methods project with EDS/λ-EDS as direct controls?  
7. Should DEXP remain an independent methods/supplementary line rather than enter production?  
8. Should charged-ligand and membrane ABFE be current-paper extensions or later independent work?  
9. Where should negative results appear: Results, Discussion, or Supplementary Methods?  
10. What evidence threshold should be required before calling any value “publication-ready”?

#### 7.13.6 35–50 分钟讲述顺序 / Suggested 35–50 Minute Presentation Order

**中文**

| 时间 | 部分 | 导师目标 |
|---|---|---|
| 0–5 min | 科学问题与一句话状态 | 软件已成熟；最终数值尚未解决 |
| 5–12 min | 热力学循环与管线 | complex/solvent、Stage 0/1/2 |
| 12–20 min | ACE、IBS v29、path v21、local TMBAR | 解释方法学核心 |
| 20–28 min | 历史与新 ABFE artifacts | 展示数值与证据边界 |
| 28–34 min | 不确定度与无效历史值 | 解释为何尚不宜发表 |
| 34–41 min | MACE/TorchForce failures 与 EXP-020/026/028 | 展示负结果如何收窄路线 |
| 41–46 min | EXP-029 与 Route 2 | 请求决定性的 online-utility 决策 |
| 46–50 min | Route 1、EDS、优先级与导师问题 | 决定未来方法范围与资源 |

**English**

| Time | Section | Advisor objective |
|---|---|---|
| 0–5 min | Scientific problem and one-sentence status | Software mature; final value unresolved |
| 5–12 min | Thermodynamic cycle and pipeline | Complex/solvent, Stage 0/1/2 |
| 12–20 min | ACE, IBS v29, path v21, local TMBAR | Explain methodological core |
| 20–28 min | Historical and new ABFE artifacts | Show numbers and evidence boundaries |
| 28–34 min | Uncertainty and invalid historical values | Explain why publication is premature |
| 34–41 min | MACE/TorchForce failures and EXP-020/026/028 | Show how negative results narrowed the route |
| 41–46 min | EXP-029 and Route 2 | Request a decisive online-utility decision |
| 46–50 min | Route 1, EDS, priorities, and advisor questions | Decide future method scope and resources |

For a 35-minute presentation, move DEXP, ORB, charged ligands, membranes, and detailed EDS equations to backup slides. For a 50-minute presentation, retain the numerical qualification tables and the full EXP-020→029 evidence chain.

#### 7.13.7 主线结论 / Mainline Closing Statement

本项目已经从“ABFE 能否运行”推进到“如何使 Hamiltonian、采样分布、估计器、修正项、软件恢复和研究身份全部可审计”。当前主线软件已经覆盖 complex/solvent 两腿、Boresch、PME decharging、ACE/IBS vanishing、local TMBAR、LRC 和非突变 resume/rescue。

The project has progressed from asking whether ABFE can run to establishing how Hamiltonians, sampling distributions, estimators, corrections, software recovery, and research identities can all remain auditable. The mainline software covers complex and solvent legs, Boresch, PME decharging, ACE/IBS vanishing, local TMBAR, LRC, and non-mutating resume/rescue.

当前数值证据不支持给出唯一的 Atenolol 最终值。mixed-history `output_lrc_fix` 与 Seed 20260907 相隔 `6.391 kcal/mol`，但二者不是严格的同协议重复；该差距只能说明需要受控复核，不能替代同一冻结身份下的独立重复，也不能直接作为 between-run uncertainty。

The current numerical evidence does not support one final Atenolol value. The mixed-history `output_lrc_fix` artifact and Seed 20260907 differ by `6.391 kcal/mol`, but they are not strict same-protocol repeats. Their gap therefore motivates—not replaces—same-protocol independent repeats under a frozen identity; only those repeats can determine between-run uncertainty.

## 附录 A. 实现模块、协议参数与主要证据入口 / Appendix A. Implementation Modules, Protocol Parameters, and Evidence Entry Points

正文按科学判断顺序保留模块职责、关键结果与资格边界；会打断叙事的命令行开关、默认值、force-group 编号和数值容差集中登记于本附录。它们仍属于报告的一部分，但不应与科学结论争夺正文层级。

The main text retains module roles, decisive results, and qualification boundaries in scientific order. Command-line switches, defaults, force-group identifiers, and numerical tolerances that would interrupt the narrative are registered here. They remain part of the report without competing with the main scientific argument.

### A.1 管线实现模块 / Pipeline Implementation Modules

**中文**

| 模块 | 作用 | 当前状态 |
|---|---|---|
| input/system loader | 读取 topology、coordinates、ligand selection 与 environment metadata | 已实现；需要 topology reconciliation |
| complex/solvent leg builder | 从共同端点身份构建两条 thermodynamic legs | 已实现 |
| Boresch parameter source | simple/fluctuation、traditional、`orb_ml`、`orb_simple`、`auto` | simple/fluctuation 推荐；其他可选/实验性 |
| Stage 0 restraint operator | 连接标准态与 restrained complex，并写入解析修正 | 已实现；sensitivity 待定 |
| Stage 1 electrostatics operator | PME decharging、charge-transfer seam 与 static-neutral handoff | neutral baseline 已实现；charged extension 在 seam 层级已资格化 |
| Stage 2 path builder | ACE-softcore、DEXP experimental family 与 λ/path construction | ACE baseline 已实现；DEXP 实验性 |
| IBS controller | learning、freeze、fixed validation 与 immutable production | v29 合同已实现 |
| estimator/covariance chain | local TMBAR、block/bootstrap、cross-window covariance | 已实现；between-run 项待完成 |
| correction ledger | Boresch、LRC、APBS 与 standard-state terms | 已实现，带 environment-specific gates |
| checkpoint/resume/rescue | fingerprint、seed ledger、non-mutating recovery | 已实现；cross-process E2E 仍为 release gate |
| artifact/report writer | machine-readable result、manifest、diagnostics 与 claim status | 已实现 |

**English**

| Module | Function | Current status |
|---|---|---|
| input/system loader | Read topology, coordinates, ligand selection, and environment metadata | Implemented; topology reconciliation required |
| complex/solvent leg builder | Build two thermodynamic legs from shared endpoint identity | Implemented |
| Boresch parameter source | simple/fluctuation, traditional, `orb_ml`, `orb_simple`, `auto` | simple/fluctuation recommended; others optional/experimental |
| Stage 0 restraint operator | Connect the standard state to the restrained complex and write the analytic correction | Implemented; sensitivity open |
| Stage 1 electrostatics operator | PME decharging, charge-transfer seam, and static-neutral handoff | Neutral baseline implemented; charged extension qualified at seam level |
| Stage 2 path builder | ACE-softcore, experimental DEXP family, and λ/path construction | ACE baseline implemented; DEXP experimental |
| IBS controller | Learning, freeze, fixed validation, and immutable production | v29 contract implemented |
| estimator/covariance chain | Local TMBAR, block/bootstrap, and cross-window covariance | Implemented; between-run term open |
| correction ledger | Boresch, LRC, APBS, and standard-state terms | Implemented with environment-specific gates |
| checkpoint/resume/rescue | Fingerprint, seed ledger, and non-mutating recovery | Implemented; cross-process E2E remains a release gate |
| artifact/report writer | Machine-readable result, manifest, diagnostics, and claim status | Implemented |

### A.2 用户可见配置轴 / User-Visible Configuration Axes

**中文**

| 配置轴 | 主要选项 | 本报告中的默认或边界 |
|---|---|---|
| system leg | complex / solvent | binding cycle 两者均必需 |
| environment | soluble / membrane | 先完成 soluble baseline；membrane extension 受门控 |
| charge treatment | neutral / PME decharge / charge-transfer | neutral baseline；charged seam 单独处理 |
| restraint source | simple/fluctuation / traditional / `orb_ml` / `orb_simple` / `auto` | simple/fluctuation 推荐 |
| decoupling path | ACE-softcore / experimental DEXP | ACE 为主线；DEXP 使用独立 claim ledger |
| sampler | IBS / traditional REMD-like branch | IBS 为主线；不同身份不得合并 |
| estimator | local TMBAR / qualified fallback | TMBAR 仅在全部门通过后使用 |
| dispersion treatment | environment-specific LRC decision | 不同 potentials/environments 之间不得静默继承 |
| continuum correction | APBS off/on | 仅在明确 cycle 位置后启用 |
| resume/recovery | fresh / resume / rescue | immutable fingerprint；rescue 不突变 |
| backend | CPU / CUDA / TorchForce or specialized residual backend | 必须满足冻结环境等价性 |

**English**

| Configuration axis | Main options | Default or boundary in this report |
|---|---|---|
| System leg | complex / solvent | Both required for the binding cycle |
| Environment | soluble / membrane | Soluble baseline first; membrane extension gated |
| Charge treatment | neutral / PME decharge / charge-transfer | Neutral baseline; charged seam separate |
| Restraint source | simple/fluctuation / traditional / `orb_ml` / `orb_simple` / `auto` | simple/fluctuation recommended |
| Decoupling path | ACE-softcore / experimental DEXP | ACE is mainline; DEXP has a separate claim ledger |
| Sampler | IBS / traditional REMD-like branch | IBS mainline; identities must not be pooled |
| Estimator | local TMBAR / qualified fallback | TMBAR only after complete gates |
| Dispersion treatment | Environment-specific LRC decision | No silent inheritance across potentials/environments |
| Continuum correction | APBS off/on | Only after explicit cycle placement |
| Resume/recovery | fresh / resume / rescue | Immutable fingerprint; rescue is non-mutating |
| Backend | CPU / CUDA / TorchForce or specialized residual backend | Frozen-environment equivalence required |

### A.3 IBS v29 冻结合同 / IBS v29 Freeze Contract

**中文**

| 阶段 | 关键参数或门 | 允许的动作 |
|---|---|---|
| learning | minibatch `40`；damping `0.10`；per-update cap `2 kBT` | 更新 bias/mixture parameters 并记录 ledger |
| freeze decision | 连续两个 batch 的 change `≤1 kBT` | 冻结一个 immutable parameter set |
| fixed validation | 在 frozen parameters 下运行五个 batch | 仅验证；不得自适应 |
| promotion gate | fixed-validation discrepancy `≤10 kJ/mol` 加 diagnostics | 进入 production 或 fail closed |
| production | frozen parameters、seed 和 fingerprint | 仅采样；禁止更新 |

**English**

| Phase | Parameter or gate | Permitted action |
|---|---|---|
| Learning | Minibatch `40`; damping `0.10`; per-update cap `2 kBT` | Update bias/mixture parameters and record the ledger |
| Freeze decision | Change `≤1 kBT` in two consecutive batches | Freeze one immutable parameter set |
| Fixed validation | Five batches under frozen parameters | Validate only; no adaptation |
| Promotion gate | Fixed-validation discrepancy `≤10 kJ/mol` plus diagnostics | Enter production or fail closed |
| Production | Frozen parameters, seed, and fingerprint | Sampling only; updates prohibited |

### A.4 Force groups 与三本账 / Force Groups and the Three Ledgers

**中文**

| Group | 内容 | Ledger 作用 |
|---:|---|---|
| `0` | bonded 与环境内部 baseline | shared/base |
| `1` | electrostatics 或 PME-related component | physical target component |
| `2` | van der Waals / decoupling component | physical target component |
| `3` | Boresch restraint | restraint ledger |
| `4–6` | path、correction 或 environment-specific terms | explicit physical/correction ledgers |
| `10–13` | IBS/residual/bias bookkeeping | sampling ledger；除非单独 promote |

**English**

| Group | Content | Ledger role |
|---:|---|---|
| `0` | Bonded and environment-internal baseline | Shared/base |
| `1` | Electrostatics or PME-related component | Physical target component |
| `2` | Van der Waals / decoupling component | Physical target component |
| `3` | Boresch restraint | Restraint ledger |
| `4–6` | Path-, correction-, or environment-specific terms | Explicit physical/correction ledgers |
| `10–13` | IBS/residual/bias bookkeeping | Sampling ledger only unless separately promoted |

具体 group 意义由 run manifest 固定；编号本身不是跨版本物理身份。分析必须分别重构 base physical energy、target Hamiltonian 和 sampling/bias ledger，不能用“总能量相同”代替逐项一致性。

Exact meanings are frozen by each run manifest; an identifier alone is not a cross-version physical identity. Analysis must reconstruct the base physical energy, target Hamiltonian, and sampling/bias ledger separately rather than substituting equality of total energy for component-wise equivalence.

### A.5 Charged-route 数值容差 / Charged-Route Numerical Tolerances

**中文**

| 检查 | 容差 | 失败处理 |
|---|---:|---|
| 配体电荷身份 | `1×10⁻3 e` | 停止 |
| 总电荷/λ 电荷守恒 | `1×10⁻6 e` | 停止 |
| 端点相对能量闭合 | `1×10⁻5` | 停止 |
| 力一致性 | `1×10⁻3` | 停止 |
| 严格端点能量闭合 | `1×10⁻6` | 停止 |
| GROMACS/OpenMM 分量匹配 | `1×10⁻4` | 停止 |
| 跨引擎总能量包络 | `0.1 kJ/mol` | 诊断失败；不晋级 |

**English**

| Check | Tolerance | Failure behavior |
|---|---:|---|
| ligand charge identity | `1×10⁻3 e` | stop |
| total/λ charge conservation | `1×10⁻6 e` | stop |
| endpoint relative-energy closure | `1×10⁻5` | stop |
| force consistency | `1×10⁻3` | stop |
| strict endpoint energy closure | `1×10⁻6` | stop |
| GROMACS/OpenMM component match | `1×10⁻4` | stop |
| total-energy cross-engine envelope | `0.1 kJ/mol` | diagnostic failure; no promotion |

### A.6 主要证据入口 / Principal Evidence Entry Points

- [2026-08-13 详细数据版 / Detailed data report](ADVISOR_DETAILED_PROJECT_REPORT_WITH_DATA_2026-08-13.md)
- [2026-08-18 决策更新版 / Decision update](ADVISOR_DETAILED_PROJECT_REPORT_WITH_DATA_2026-08-18.md)
- [EXP-026 CUDA 控制面优化 / CUDA control-plane optimization](../PLAN_EXP-026_cuda_control_plane_optimization.md)
- [EXP-027/028 结果汇总 / Result summary](../exp027_result.md)
- [EXP-027 在线效用计划 / Online utility plan](../PLAN_EXP-027_online_utility.md)
- [EXP-030 joint-state score 设计 / Joint-state-score design](../exp-30.md)
- [Outer-λ neural-basis 原型 / Prototype](../outer_lambda_neural_basis.py)
- [IBS 论文提取文本 / IBS paper extraction](../references/papers/integrated-boltzmann-sampling.md)
- `output_lrc_fix_repeat02_seed20260906/final_results.json`
- `output_lrc_fix_repeat03_seed20260907/final_binding_results.json`
- `output/outer_lambda_exp027_online_utility/exp028_u3_confirmation_report.json`

## 附录 B. 历史材料与当前结论的处理原则 / Appendix B. Treatment of Historical Material and Current Conclusions

### B.1 状态标签 / Status Labels

**中文**

| 标签 | 定义 |
|---|---|
| `RAW` | 原始产物已存在，尚未完成资格判定 |
| `DIAGNOSTIC_ONLY` | 仅用于定位机制或失效，不进入最终估计 |
| `INCONCLUSIVE` | 数据存在，但无法区分竞争解释 |
| `INVALID_FOR_PROMOTION` | 已知违反协议或数据完整性，禁止晋级 |
| `PROVENANCE_MIXED_HISTORICAL` | 同一目录或日志混合失败与修复代；可用于诊断，不可作为 clean repeat |
| `QUALIFIED` | 在明确范围和门限内通过 |
| `AUTHORIZED_NOT_STARTED` | 已批准，但没有完成运行证据 |
| `NOT_PRODUCTION_PROMOTED` | 原型或 scoped test 通过，尚未进入 production |
| `NOT_PUBLICATION_READY` | 仍有会改变主结论的开放不确定性 |

**English**

| Label | Meaning |
|---|---|
| `RAW` | Raw artifact exists; qualification has not yet been completed |
| `DIAGNOSTIC_ONLY` | Used only to localize a mechanism or failure; excluded from the final estimate |
| `INCONCLUSIVE` | Data exist, but competing explanations cannot be distinguished |
| `INVALID_FOR_PROMOTION` | Known protocol or data-integrity violation; promotion is prohibited |
| `PROVENANCE_MIXED_HISTORICAL` | Failed and repaired generations are mixed in one directory or log; diagnostic use only, not a clean repeat |
| `QUALIFIED` | Passed within an explicit scope and threshold |
| `AUTHORIZED_NOT_STARTED` | Approved, but no completed run evidence exists |
| `NOT_PRODUCTION_PROMOTED` | Prototype or scoped test passed; not yet in production |
| `NOT_PUBLICATION_READY` | Open uncertainty can still change the main conclusion |

These labels separate the existence of an artifact from the strength of the claim it can support. A completed file is not automatically a qualified result, and a qualified component is not automatically a publication-ready full cycle.

这些标签把“产物是否存在”与“可以支持多强的主张”分开。文件完成不等于结果合格，单个组件合格也不等于完整 cycle 已达到发表标准。

### B.2 证据优先级 / Evidence Precedence

结论冲突时采用同一顺序：原始 machine-readable artifact；不可变输入、manifest、hash、seed 与环境；run-specific qualification/report；当前 source/test contract；dated plan 或 memo；无明确身份的叙述。计划只能证明工作被提出或授权，不能证明已经完成。

When conclusions conflict, the order is: raw machine-readable artifact; immutable input, manifest, hash, seed, and environment; run-specific qualification/report; current source/test contract; dated plan or memo; and finally prose without an explicit identity. A plan proves intention or authorization, not completion.

### B.3 历史材料保留规则 / Historical-Material Rule

本报告不删除历史失败，不使用新结果追溯覆盖旧实验身份，也不把计划写成已经完成的结果。此前阶段报告中仍然有效的方法解释被保留；被新 artifact 或后续实验推翻的状态，在正文中明确标为历史状态，并由更新后的证据替代。

This report does not delete historical failures, retroactively overwrite old experimental identities with new results, or present plans as completed results. Methodological explanations from the earlier stage report are retained where valid. Status statements superseded by new artifacts or later experiments are explicitly identified as historical and replaced by updated evidence in the main text.

## 附录 C. 工作区实验与产物总账 / Appendix C. Workspace Experiment and Artifact Ledger

### C.1 为什么需要 artifact ledger / Why an Artifact Ledger Is Required

The workspace is no longer a code-only prototype. It contains production-like ABFE outputs, independent repeats, membrane runs, charged-system validation matrices, MACE/outer-λ experiments, source, tests, handoffs, and failure archives. At the time of this audit, approximately `4,770` files were visible, including about `1,556 NPZ`, `1,264 JSON`, `366 NPY`, `303 Python`, `227 Markdown`, `204 CSV`, `93 JPEG`, `85 XML`, `82 log`, `77 C++`, `59 header`, and `57 PNG` files. These counts are a dated inventory, not scientific sample counts.

当前工作区已经不是“只有代码的原型”。它同时包含近 production 的 ABFE 产物、独立 repeats、膜运行、charged-system validation matrices、MACE/outer-λ 实验、source、tests、handoffs 与 failure archives。本次盘点可见约 `4,770` 个文件，包括约 `1,556 NPZ`、`1,264 JSON`、`366 NPY`、`303 Python`、`227 Markdown`、`204 CSV`、`93 JPEG`、`85 XML`、`82 log`、`77 C++`、`59 header` 和 `57 PNG`。这些只是带时间戳的文件盘点，不是科学样本数。

“全部纳入报告”在这里定义为：每类证据进入 ledger；关键数字进入正文或表格；每个主张能追溯到产物；失败结果保留；二进制 array、trajectory 和重复 cache 不逐字嵌入 Markdown，而以路径、角色与读取规则登记。这样既不丢证据，也不会把报告变成不可读的二进制转储。

“Include everything” therefore means that every evidence class is registered, decisive values are tabulated, each claim is traceable, and failed outcomes remain visible. Binary arrays, trajectories, and duplicate caches are indexed by path, role, and interpretation rather than pasted into Markdown.

### C.2 证据权威顺序 / Artifact Authority Order

当不同文档或产物冲突时，按以下顺序判定：

1. completed run 的原始 machine-readable artifact（JSON/NPZ/NPY/CSV/XML 与不可变输入 hash）；
2. 与该 artifact 同目录的 manifest、qualification report 和 run log；
3. 针对该 run 的 handoff 或 result report；
4. source/test 所定义的当前 contract；
5. dated TODO、plan、advisor memo；
6. 无明确版本或来源的叙述性文字。

When records conflict, completed machine-readable artifacts and their immutable inputs outrank prose. A plan is evidence of intended work, not evidence of completion. A historical report is not silently rewritten; it is marked as superseded and linked to the newer artifact.

### C.3 主 ABFE 产物账 / Principal ABFE Artifact Ledger

**中文**

| 路径 | 范围 | 决定性内容 | 报告状态 |
|---|---|---|---|
| `output/final_binding_results.json` | 早期完整 cycle | `+40.8362 ± 1.3178 kJ/mol`; complex `192.8876`，solvent `152.0514`，Boresch `−36.5108` | `INVALIDATED` |
| `output_lrc_fix/final_binding_results.json` | 复用目录的历史完整 cycle | `−5.5359 ± 0.6008 kcal/mol`; 失败与修复后的 attachment generation 共存于同一日志 | `PROVENANCE_MIXED_HISTORICAL` |
| `output_lrc_fix_repeat02_seed20260906/final_results.json` | 独立 repeat 02 | complex `202.6621 ± 1.8244 kJ/mol`; 没有完成的 solvent/binding artifact | `INCOMPLETE` |
| `output_lrc_fix_repeat03_seed20260907/final_binding_results.json` | 完整 Seed 20260907 cycle | `−11.9270 ± 0.4155 kcal/mol`; 身份不同于历史候选 | `AVAILABLE_BUT_NOT_POOLED` |
| `memtest/output_membrane_5ns/final_binding_results.json` | 中性膜工程 smoke | `−26.4852 ± 1.7675 kJ/mol`; `−6.3301 ± 0.4224 kcal/mol` | engineering-only |
| `memtest/output_membrane_100ns/final_binding_results.json` | 溶剂 topology 损坏的膜运行 | `+97.3579 ± 2.0921 kJ/mol`; `+23.2691 kcal/mol` | `INVALIDATED` |

**English**

| Path | Scope | Decisive content | Report status |
|---|---|---|---|
| `output/final_binding_results.json` | early full cycle | `+40.8362 ± 1.3178 kJ/mol`; complex `192.8876`, solvent `152.0514`, Boresch `−36.5108` | `INVALIDATED` |
| `output_lrc_fix/final_binding_results.json` | reused-directory historical full cycle | `−5.5359 ± 0.6008 kcal/mol`; failed and corrected attachment generations coexist in one log | `PROVENANCE_MIXED_HISTORICAL` |
| `output_lrc_fix_repeat02_seed20260906/final_results.json` | independent repeat 02 | complex `202.6621 ± 1.8244 kJ/mol`; no finished solvent/binding artifact | `INCOMPLETE` |
| `output_lrc_fix_repeat03_seed20260907/final_binding_results.json` | complete Seed 20260907 cycle | `−11.9270 ± 0.4155 kcal/mol`; identity differs from historical candidate | `AVAILABLE_BUT_NOT_POOLED` |
| `memtest/output_membrane_5ns/final_binding_results.json` | neutral membrane engineering smoke | `−26.4852 ± 1.7675 kJ/mol`; `−6.3301 ± 0.4224 kcal/mol` | engineering-only |
| `memtest/output_membrane_100ns/final_binding_results.json` | membrane run with corrupted solvent topology | `+97.3579 ± 2.0921 kJ/mol`; `+23.2691 kcal/mol` | `INVALIDATED` |

The two complete soluble candidates are separated by `6.391 kcal/mol`, but they differ in code hash, system XML hash, OpenMM version, and Boresch candidate set. Repeat02 never produced a complete cycle. The workspace therefore establishes an unresolved cross-identity discrepancy—not a quantified same-protocol reproducibility variance—and the two complete values must not be pooled.

两个完整 soluble 候选的描述性间隔为 `6.391 kcal/mol`，但 code hash、system XML hash、OpenMM 版本和 Boresch candidate set 均不完全相同；repeat02 又未完成完整 cycle。因此工作区证明的是尚未分解的跨身份 discrepancy，而不是已经量化的同协议 reproducibility variance；两个完整值禁止直接 pooled。

### C.4 Charged 与 membrane 验证产物账 / Charged and Membrane Validation Ledger

**中文**

| 证据族 | 主要目录/文件 | 已建立内容 | 尚未建立内容 |
|---|---|---|---|
| C1 水盒 | `validation/c1_*` artifacts | 离子符号/大小覆盖；静态、动力学、端点和 `u_kn` 检查 | 配体 ABFE 精度 |
| C2 脂质 slab | `validation/c2_lipid_slab_v11_full11/` 加独立 seed 目录 | 四种 slab geometry、11 个 λ states、多 seed 行为 | production 长度的膜结合 |
| C3 endpoint seam | `validation/c3_real_endpoints_v2/summary.json`、`mem00h_report.json` | 150 个真实 frames、0 次失败、A/B/C/D 和 strict-zero closure | 完整 complex-minus-solvent charged cycle |
| 中性膜 smoke | `memtest/output_membrane_5ns/` | 完整 Stage 0→1→2 可执行性 | production convergence 和 charged branch |
| 无效 100 ns 路径 | `memtest/output_membrane_100ns/` 及 membrane handoff | 诊断价值和 topology-loss 根因 | 有效 binding estimate |

**English**

| Evidence family | Primary directories/files | What is established | What is not established |
|---|---|---|---|
| C1 water boxes | `validation/c1_*` artifacts | ion sign/size coverage; static, dynamics, endpoint, and `u_kn` checks | ligand ABFE accuracy |
| C2 lipid slabs | `validation/c2_lipid_slab_v11_full11/` plus independent-seed directories | four slab geometries, 11 λ states, and multi-seed behavior | production-length membrane binding |
| C3 endpoint seam | `validation/c3_real_endpoints_v2/summary.json`, `mem00h_report.json` | 150 real frames, 0 failures, A/B/C/D, and strict-zero closure | full complex-minus-solvent charged cycle |
| neutral membrane smoke | `memtest/output_membrane_5ns/` | full Stage 0→1→2 executability | production convergence and charged branch |
| invalid 100 ns path | `memtest/output_membrane_100ns/` and membrane handoff | diagnostic value and topology-loss root cause | valid binding estimate |

### C.5 Solvent-size、Boresch 与 protocol-evolution 账 / Solvent-Size, Boresch, and Protocol-Evolution Ledger

**中文**

| 证据 | 关键观察 | 后果 |
|---|---|---|
| pad 1.5 / main / pad 2.4 solvent legs | 相同尺寸的 scatter 超过表观跨 box 信号 | 暂不得宣称有限尺寸效应 |
| 500-frame Boresch diagnostics | historical/new candidate sets 含 562/578 个 anchor candidates，且各有一个 clipped case | 局部 harmonicity 不是 sensitivity closure |
| attachment sign repair | 约 `100 kJ/mol` 的错误 attachment 降至约 `4.4 kJ/mol` | restraint sign/mapping 必须 contract-test |
| v19→v21 path 和 v27→v29 IBS evolution | estimator/path/state-machine identities 已改变 | 不得在没有显式 compatibility analysis 时跨 identity 合并 runs |
| LRC 与 dispersion revisions | 引入了按环境决策和 non-mutating rescue | 旧的 “LRC fixed” 标签不表示符合当前 contract |
| checkpoint/cache revisions | 增加了 fingerprints、seed ledgers 和 fail-closed resume | 仅有文件存在不足以证明 provenance |

**English**

| Evidence | Key observation | Consequence |
|---|---|---|
| pad 1.5 / main / pad 2.4 solvent legs | same-size scatter exceeds the apparent cross-box signal | do not claim a finite-size effect yet |
| 500-frame Boresch diagnostics | historical/new candidate sets contain 562/578 anchor candidates and one clipped case each | local harmonicity is not sensitivity closure |
| attachment sign repair | about `100 kJ/mol` erroneous attachment reduced to about `4.4 kJ/mol` | restraint sign/mapping must be contract-tested |
| v19→v21 path and v27→v29 IBS evolution | estimator/path/state-machine identities changed | never pool runs across identities without an explicit compatibility analysis |
| LRC and dispersion revisions | environment-specific decisions and non-mutating rescue were introduced | old “LRC fixed” labels do not imply current-contract equivalence |
| checkpoint/cache revisions | fingerprints, seed ledgers, and fail-closed resume were added | file presence alone is insufficient provenance |

### C.6 MACE、outer-λ、residual 与 DEXP 产物族 / MACE, Outer-λ, Residual, and DEXP Families

**中文**

| 家族 | 主要证据 | 科学作用 |
|---|---|---|
| outer-λ neural basis | `outer_lambda_neural_basis.py`、EXP-006/007/009–020 reports 和 arrays | teacher signal、basis construction、overlap 与 stationarity probes |
| grouped-density residual | EXP-020 和后续 reports | local many-body correction 的候选 representation |
| 专用 CUDA backend | `PLAN_EXP-025*`、EXP-026 artifacts、C++/CUDA sources 和 qualification logs | 可行性、correctness 与成本边界 |
| long-run online utility | `exp027_result.md`、EXP-027/028 arrays/reports | 揭示 state-candidate mismatch 和 instability mechanisms |
| joint-state-score | `exp-30.md` 以及 EXP-029/030 artifacts | 当前决定性的 online-utility formulation |
| DEXP analytic projection | 中文 manuscript、fit/validation tables、V/S/B replica evidence | 独立的 potential-family 路线；不是仅 sampling residual |
| ORB | manuscript/results 和 representation diagnostics | representation comparison；尚非 production ABFE claim |

**English**

| Family | Principal evidence | Scientific role |
|---|---|---|
| outer-λ neural basis | `outer_lambda_neural_basis.py`, EXP-006/007/009–020 reports and arrays | teacher signal, basis construction, overlap, and stationarity probes |
| grouped-density residual | EXP-020 and downstream reports | candidate representation for local many-body correction |
| specialized CUDA backend | `PLAN_EXP-025*`, EXP-026 artifacts, C++/CUDA sources, and qualification logs | feasibility, correctness, and cost envelope |
| long-run online utility | `exp027_result.md`, EXP-027/028 arrays/reports | reveals state-candidate mismatch and instability mechanisms |
| joint-state-score | `exp-30.md` and EXP-029/030 artifacts | current decisive online-utility formulation |
| DEXP analytic projection | Chinese manuscript, fit/validation tables, V/S/B replica evidence | independent potential-family line; not a sampling-only residual |
| ORB | manuscript/results and representation diagnostics | representation comparison; not yet a production ABFE claim |

These families must remain experimentally connected but scientifically distinct. In particular, successful MACE-to-DEXP projection does not validate Route 1 residual sampling, and an outer-λ offline score does not modify the frozen Route 2 target Hamiltonian.

这些产物族可以共享证据，但科学身份必须分开。尤其是，MACE→DEXP 的成功 projection 不能替代 Route 1 residual sampling 的验证；outer-λ offline score 也不能改写 Route 2 的 frozen target Hamiltonian。

### C.7 原始数据的保存与读取规则 / Raw-Data Preservation and Reading Rules

- JSON/Markdown/CSV contain human-auditable summaries and ledgers; they should be reviewed first.
- NPZ/NPY contain reduced potentials, weights, trajectories, features, scores, and intermediate arrays; interpretation requires the matching manifest and code version.
- XML/CIF/PDB describe system/topology state and must be paired with force-count and atom-index reconciliation.
- Logs establish execution history but do not outrank final artifacts when a run was resumed or repaired.
- Images are diagnostic visualization, not numerical authority unless their underlying arrays are also preserved.
- Duplicate caches are never treated as independent samples merely because they are separate files.

- JSON/Markdown/CSV 是优先阅读的人类可审计 summary/ledger；
- NPZ/NPY 保存 reduced potentials、weights、trajectory/features/scores 等数组，必须配合 manifest 与代码版本解释；
- XML/CIF/PDB 必须与 force count、atom index reconciliation 配套；
- log 证明执行历史，但在 resume/repair 后不高于 final artifact；
- image 是诊断可视化，若无底层数组则不能作为数值权威；
- 重复 cache 即使文件不同，也不能自动当成独立样本。

### C.8 DEXP 详细诊断与 artifact 身份 / DEXP Detailed Diagnostics and Artifact Identity

#### C.8.1 逐幅度 residual / Amplitude-Resolved Residuals

**中文**

| 扰动类型 | 幅度 | n | 偏差 (kJ/mol) | RMSE (kJ/mol) |
|---|---:|---:|---:|---:|
| 旋转 | `0.5°` | 120 | `0.089` | `0.975` |
| 旋转 | `1.5°` | 120 | `0.780` | `3.059` |
| 旋转 | `3.0°` | 120 | `3.069` | `6.946` |
| 平移 | `0.005 nm` | 280 | `0.149` | `1.426` |
| 平移 | `0.010 nm` | 280 | `0.588` | `2.911` |
| 平移 | `0.020 nm` | 280 | `2.327` | `6.273` |
| 平移 | `0.040 nm` | 280 | `9.248` | `15.925` |

**English**

| Perturbation type | Amplitude | n | Bias (kJ/mol) | RMSE (kJ/mol) |
|---|---:|---:|---:|---:|
| rotation | `0.5°` | 120 | `0.089` | `0.975` |
| rotation | `1.5°` | 120 | `0.780` | `3.059` |
| rotation | `3.0°` | 120 | `3.069` | `6.946` |
| translation | `0.005 nm` | 280 | `0.149` | `1.426` |
| translation | `0.010 nm` | 280 | `0.588` | `2.911` |
| translation | `0.020 nm` | 280 | `2.327` | `6.273` |
| translation | `0.040 nm` | 280 | `9.248` | `15.925` |

#### C.8.2 密网格 basin 与 `r_0` scaling / Dense-Grid Basin and `r_0` Scaling

对 `α∈[9,19]`、`β∈[2,10]` 使用 `0.1` 步长，共扫描 8115 组。5% low-error basin 含 194 点，紧约束方向为 `α+β≈19.004`，标准差 `0.134`。even 连续最优点为 `(12.5,6.5)`、值 `2.730`；`(14,5)` 为 `2.818`，仅高 `3.1%`，bootstrap CI 跨零。

The scan covered 8115 points over `α∈[9,19]` and `β∈[2,10]` at `0.1` spacing. The 5% low-error basin contained 194 points, with the tightly constrained direction `α+β≈19.004` and standard deviation `0.134`. The continuous even optimum was `(12.5,6.5)` at `2.730`; `(14,5)` gave `2.818`, only `3.1%` higher, with a bootstrap confidence interval crossing zero.

固定 `(14,5)` 后扫描 `s_r=0.96–1.04`、步长 `0.01`。even 在 `s_r=1.00` 为 `2.82 kJ/mol`，而 `0.96/1.04` 分别为 `8.33/9.88`。odd 在 `0.99` 的微小改善从 `6.11` 到 `6.07`，但 CI `[−0.35,0.16]` 跨零；LOAO `20/20` folds 均选择 `s_r=1.00`，故没有可 promotion 的 `r_0` rescaling。

With `(14,5)` fixed, `s_r=0.96–1.04` was scanned at `0.01` spacing. The even metric was `2.82 kJ/mol` at `s_r=1.00`, versus `8.33/9.88` at `0.96/1.04`. The small odd improvement at `0.99`, from `6.11` to `6.07`, had CI `[−0.35,0.16]`; all `20/20` LOAO folds selected `s_r=1.00`, leaving no promotable rescaling.

#### C.8.3 参数来源不可互换 / Non-Interchangeable Parameter Sources

**中文**

| 参数来源 | α | β | 标定含义 |
|---|---:|---:|---|
| DEGAUSS early | `17.470` | `4.099` | TIP3P O–O LJ 在 `0.9r_m–5.0r_m` 区间上的拟合；水密度约 `0.977 g/cm³` |
| DEGAUSS final | `18.17` | `3.65` | TIP3P O–O LJ 在 `0.7r_m–3.0r_m` 区间上的拟合；一个全局 pair kernel |
| 本项目解析参考 | `12` | `6` | 在 `r_0` 匹配 LJ value/slope/curvature |
| Atenolol 单体系最优 | `14` | `5` | 20-anchor LOAO local MACE projection；`α+β≈19` basin |

**English**

| Parameter source | α | β | Calibration meaning |
|---|---:|---:|---|
| DEGAUSS early | `17.470` | `4.099` | TIP3P O–O LJ over `0.9r_m–5.0r_m`; water density about `0.977 g/cm³` |
| DEGAUSS final | `18.17` | `3.65` | TIP3P O–O LJ over `0.7r_m–3.0r_m`; one global pair kernel |
| analytic reference in this project | `12` | `6` | LJ value/slope/curvature match at `r_0` |
| Atenolol single-system optimum | `14` | `5` | 20-anchor LOAO local MACE projection; `α+β≈19` basin |

这些参数回答不同问题：水模型拟合区间、井底解析匹配和 Atenolol pocket 局部投影不能按数值相近直接互换。

These parameter sets answer different calibration questions. A water-model fitting range, analytic well-minimum matching, and local projection in the Atenolol pocket cannot be interchanged by numerical proximity.

#### C.8.4 DEXP artifact 年表 / DEXP Artifact Chronology

**中文**

| 产物家族 | 代表位置 | 参数/模型身份 | 允许用途 |
|---|---|---|---|
| 当前 pair-specific projection | manuscript-associated fit/validation artifacts | 来自 20-anchor LOAO 的 `(14,5)`；pair-specific well-matched kernel | 当前单体系证据 |
| legacy global fit | `output/dexp_experiment/` | 较旧的全局 α/β，包括接近 `17/15` 或 `14.0004/11.9009` 的值 | 仅历史 development |
| 10 ns diagnostic branch | `output/dexp_experiment_10ns_diag/` | legacy relabel/dynamics identity | 仅历史诊断 |
| OLD archive | `output/dexp_experiment_OLD/` | 更早的、接近 `17.339/14.998` 的值 | 仅 provenance/archive |

**English**

| Artifact family | Representative location | Parameter/model identity | Permitted use |
|---|---|---|---|
| current pair-specific projection | manuscript-associated fit/validation artifacts | `(14,5)` from 20-anchor LOAO; pair-specific well-matched kernel | current single-system evidence |
| legacy global fit | `output/dexp_experiment/` | older global α/β, including values near `17/15` or `14.0004/11.9009` | historical development only |
| 10 ns diagnostic branch | `output/dexp_experiment_10ns_diag/` | legacy relabel/dynamics identity | historical diagnostic only |
| OLD archive | `output/dexp_experiment_OLD/` | earlier values near `17.339/14.998` | provenance/archive only |

文件名 `comparison_summary.json` 或 `dexp_fitted_params.json` 本身不足以确定科学身份。旧 global fit、10 ns diagnostic 与 OLD branch 不是当前 pair-specific projection 的 repeats，禁止合并统计量。

The filenames `comparison_summary.json` and `dexp_fitted_params.json` do not identify the scientific model by themselves. Legacy global-fit, 10 ns diagnostic, and OLD branches are not repeats of the present pair-specific projection and must not be pooled.

## 附录 D. 软件缺陷、测试与未关闭事项账 / Appendix D. Defects, Tests, and Open-Item Ledger

### D.1 已有测试证据 / Existing Test Evidence

The 0813 inventory and later issue-specific records represent different timestamps and scopes. The former counted 88 test modules and 1,022 top-level test functions, with historical full-suite logs of 1,161 and 1,213 passes. Later issue evidence includes: Issue #84, `114 passed` plus a reference CPU smoke of `2 passed`; Issue #75 segment tests, `3 passed, 49 deselected`; layered CI for #62 with Ruff, syntax, Black/isort, mypy and `5 passed, 2 skipped`; and a later #76 environment pin to `pymbar-core=4.2.0`.

0813 盘点与后续 issue-specific records 属于不同时间和范围。前者统计 88 个 test modules、1,022 个顶层 test functions，历史 full-suite log 为 1,161 与 1,213 passes。后续证据包括：Issue #84 的 `114 passed` 与 reference CPU smoke `2 passed`；Issue #75 segment tests 的 `3 passed, 49 deselected`；Issue #62 layered CI 的 Ruff、syntax、Black/isort、mypy 与 `5 passed, 2 skipped`；以及 Issue #76 后续环境将 `pymbar-core` 固定为 `4.2.0`。

These numbers must not be summed. They demonstrate breadth and targeted regression coverage, not one single current-environment all-test pass. A clean frozen-environment rerun remains the authoritative release gate.

这些数字不能相加；它们说明覆盖面与针对性 regression evidence，而不是同一个当前环境的一次全量 pass。冻结环境的 clean rerun 仍是 release authority。

### D.2 缺陷到 protocol contract 的映射 / Defect-to-Contract Mapping

**中文**

| 失败/事项 | 可观察症状 | 已添加或要求的 contract | 当前边界 |
|---|---|---|---|
| Boresch sign/mapping | 约 `100 kJ/mol` attachment、mirrored-dihedral spike | reference-geometry 和 analytic sign tests | 已修复；sensitivity 仍未闭合 |
| 膜 solvent leg 丢失 ligand angles | angles 为零、ligand collapse、decharge 偏大 | 遍历所有 force objects；精确 `41/71/104` reconciliation | 根因已修复；损坏结果无效 |
| C3 v1 seam mismatch | 最大 force mismatch 约 `7.47×10⁻2` | bilateral MEM-00h normalization 和 v2 matrix | v2 在 150 帧上通过 |
| uncertainty underestimation (#78/P1-22) | between-run spread 超过 internal σ | block/bootstrap 加 between-run variance | 尚未闭合 |
| checkpoint/DCD validation (#63) | resume state 过浅或含义不明确 | fingerprint 和 structural integrity checks | partial/historical xfail evidence |
| CLI/input validation (#64) | assert 误用 / matrix transpose 歧义 | explicit exceptions 和 shape contracts | 需要当前 frozen-suite confirmation |
| Boresch anchors (#83) | 可能存在 restraint dependence | independent anchor/force-constant sensitivity | 开放 |
| traditional REMD LRC v3 (#32) | legacy branch runtime uncertainty | fixed-box/traditional regression | 尚未完全闭合 |
| 跨进程 resume (#75) | 只有 segment test | 带 immutable ledger 的真实 cross-process resume | targeted tests 通过；E2E 待完成 |
| dependency drift (#76) | estimator identity 可能随环境移动 | 显式固定 `pymbar-core=4.2.0` | 已实现；仍需 clean rebuild |

**English**

| Failure/issue | Observable symptom | Contract added or required | Current boundary |
|---|---|---|---|
| Boresch sign/mapping | ~`100 kJ/mol` attachment, mirrored-dihedral spike | reference-geometry and analytic sign tests | fixed; sensitivity still open |
| lost ligand angles in membrane solvent leg | zero angles, collapsed ligand, inflated decharge | iterate all force objects; exact `41/71/104` reconciliation | root cause fixed; corrupted result invalid |
| C3 v1 seam mismatch | max force mismatch ~`7.47×10⁻2` | bilateral MEM-00h normalization and v2 matrix | v2 passed on 150 frames |
| underestimated uncertainty (#78/P1-22) | between-run spread exceeds internal σ | block/bootstrap plus between-run variance | not closed |
| checkpoint/DCD validation (#63) | shallow or ambiguous resume state | fingerprint and structural integrity checks | partial/historical xfail evidence |
| CLI/input validation (#64) | assert misuse / matrix transpose ambiguity | explicit exceptions and shape contracts | requires current frozen-suite confirmation |
| Boresch anchors (#83) | possible restraint dependence | independent anchor/force-constant sensitivity | open |
| traditional REMD LRC v3 (#32) | legacy branch runtime uncertainty | fixed-box/traditional regression | not fully closed |
| resume across process (#75) | segment test only | real cross-process resume with immutable ledger | targeted tests pass; E2E pending |
| dependency drift (#76) | estimator identity can move with environment | explicit `pymbar-core=4.2.0` pin | implemented; clean rebuild still required |

### D.3 当前必须保持开放的科学事项 / Scientific Items That Must Remain Open

1. Finish at least three same-protocol, independent, complete soluble cycles with immutable seed and environment ledgers.
2. Freeze and report the formal combination of within-run covariance, block/bootstrap uncertainty, and between-run variance.
3. Complete Boresch anchor and force-constant sensitivity without post hoc selection.
4. Reanalyze Stage 1 under one current estimator/provenance identity and prove compatibility with Stage 2 artifacts.
5. Close a complete charged complex-minus-solvent production cycle after the successful C1/C2/C3 seam qualification.
6. Repeat membrane Stage 0→1→2 at production length with the repaired topology loader, membrane quality metrics, dispersion/LRC decision, and APBS cycle closure.
7. Keep DEXP, Route 1 residual sampling, Route 2 frozen-target sampling, EDS/λ-EDS, and ORB under separate claim ledgers.
8. Require frozen-environment CPU/CUDA end-to-end reruns before a release or publication claim.

1. 完成至少三个同协议、独立、完整的 soluble cycles，并冻结 seed/environment ledger；
2. 冻结 within-run covariance、block/bootstrap 与 between-run variance 的正式组合规则；
3. 完成 Boresch anchor/force-constant sensitivity，禁止事后挑选；
4. 以同一 current estimator/provenance identity 重分析 Stage 1，并证明与 Stage 2 artifact 的兼容性；
5. 在 C1/C2/C3 seam 资格化之后，完成真实 charged complex-minus-solvent production cycle；
6. 用修复后的 topology loader 重做 production-length membrane Stage 0→1→2，并闭合膜质量、dispersion/LRC 与 APBS；
7. DEXP、Route 1 residual sampling、Route 2 frozen-target、EDS/λ-EDS、ORB 必须使用独立 claim ledger；
8. release 或 publication claim 前，必须完成冻结环境 CPU/CUDA end-to-end rerun。

## 8. 最后一章：软件、MACE 与局部多体 residual 的当前结论 / Final Chapter: Software, MACE, and the Current Local-Many-Body Residual Conclusion

本章按要求放在全文最后，并集中回答一个问题：离线 representation signal、软件 correctness、CUDA cost、长程稳定性和 online scientific utility 分别走到了哪一步。这里不再回头改写前述物理 pipeline 的 target identity。

This chapter is intentionally placed at the very end. It answers one question in a single location: how far the project has progressed at the separate levels of offline representation signal, software correctness, CUDA cost, long-run stability, and online scientific utility. It does not retroactively change the target identity of the physical pipeline described above.

### 8.1 软件与 residual 功能模块表 / Software and Residual Functional-Module Table

**中文**

| 模块 | 作用 | 当前状态 |
|---|---|---|
| MACE teacher | 提供离线 local energy/latent/force reference | 仅离线 |
| ORB teacher | 提供 graph representation 与 Boresch candidates | 仅离线/teacher |
| Outer-λ controller | 构造 endpoint-zero λ-dependent coefficient | 独立 harness |
| LocalResidualStudent | 学习 local many-body residual/gap | 离线信号为正 |
| TorchForce deployment | 每步执行 neural energy/force | 成本停止 |
| MTS deployment | 低频更新 residual force | physics/ESS 停止 |
| grouped-density/SoftLift | 压缩 local representation | 离线为正或成本停止 |
| EXP-025/026 CUDA backend | 专用 local residual kernel 与 control-plane optimization | scoped cost/correctness 已合格 |
| neural ledger adapter | 分开记录 target、sampling bias 与 base energy | prototype/harness |
| EXP-027 U3/U4 | online utility stress test | U3 signal；U4 不得晋级 |
| EXP-029 | per-arm calibration + six-window paired utility | 已授权，尚未开始 |
| EXP-030 | frozen state-conditioned score | 设计草案 |
| production pipeline wiring | 将 residual 参数传入正式 `runabfe` Stage 2 | 尚未接线 |

**English**

| Module | Function | Current status |
|---|---|---|
| MACE teacher | provides offline local energy/latent/force reference | offline only |
| ORB teacher | provides graph representation and Boresch candidates | offline/teacher only |
| Outer-λ controller | constructs endpoint-zero λ-dependent coefficient | independent harness |
| LocalResidualStudent | learns local many-body residual/gap | offline signal positive |
| TorchForce deployment | executes neural energy/force at every step | cost-stopped |
| MTS deployment | updates residual force at low frequency | physics/ESS-stopped |
| grouped-density/SoftLift | compresses local representation | offline positive or cost-stopped |
| EXP-025/026 CUDA backend | dedicated local residual kernel and control-plane optimization | scoped cost/correctness qualified |
| neural ledger adapter | records target, sampling bias, and base energy separately | prototype/harness |
| EXP-027 U3/U4 | online utility stress test | U3 signal; U4 invalid for promotion |
| EXP-029 | per-arm calibration + six-window paired utility | authorized, not started |
| EXP-030 | frozen state-conditioned score | design draft |
| production pipeline wiring | passes residual parameters into formal `runabfe` Stage 2 | not wired |

### 8.2 实验状态总表 / Experiment Status Summary

**中文**

| 实验 | 关键结果 | 判决 |
|---|---|---|
| EXP-006 | maximum path force `258.949 > 250` | teacher qualification 未通过 |
| EXP-007 | coefficient `0.09`；six window checks 通过 | 仅授权 offline teacher |
| EXP-009 | N=1 `CUDA_ERROR_INVALID_HANDLE` | direct MACE-MTS 已停止 |
| EXP-010 | `0/6` folds 超过 intercept；generalized `R²=−13.5934` | 此 torsion target 失败 |
| EXP-011 | overlap `0.02353<0.03`；decorrelated `22<25` | 在 PMF 前停止 |
| EXP-012 | offline `+13.9348%`，仅 `2/3` folds；online cost `1.81–1.89×` | 无 online benefit |
| EXP-013 | temperature z `5.61–6.83`；ESS/GPU-hour `932→218` | physical MTS 被拒绝 |
| EXP-014 | native compression screen 未通过 | 已关闭 |
| EXP-016 | `3×500` surrogate frames，无真实 state crossing | `INCONCLUSIVE / SURROGATE_ONLY` |
| EXP-017 | min overlap `0.3913`；window-5 drift/`2σ=4.464` | `INCONCLUSIVE / STOPPED` |
| EXP-018 | drift z `1.134, 2.568, 1.381`；variance ratio `16.7599` | `INCONCLUSIVE / CLOSED` |
| EXP-019 | CUDA/NameError/endpoint uncertainty `1.2481>1.0` | formal repeats `0` |
| EXP-020 | offline improvement `55.5524%`；9/9 checkpoints | representation signal 已资格化 |
| EXP-021 | median `1.107419×`，P95 `1.114105×` | native density 已停止 |
| EXP-025 G0–G3/G4 oracle | ABI、CPU reference、CUDA brute force、CSR/mixed precision 和 equivalence 均通过 | harness 中 correctness 合格 |
| EXP-025 G4 cost | median `1.123398×`，P95 `1.132172×` | runtime backend 已停止 |
| EXP-026 A1.1 | cost `1.04140×`，P95 `1.04692×` | cost 合格 |
| EXP-026 A2 | cost `1.03206×`，P95 `1.03665×` | cost 合格 |
| EXP-028 | long-run `addArg` defect 已修复；stability ratio `1.033` | 已修复并验证 |
| EXP-027 U3 | 2/3 positive；median `+49.4%` | positive stress-test signal |
| EXP-027 U4 | candidate 复用了 baseline `(f_k)` | 不得晋级 |
| EXP-029 | per-arm calibration、6 windows、3 paired repeats | 已授权，尚未开始 |
| EXP-030 | joint state-conditioned frozen score | 设计草案，尚未开始 |
| ORB-001 | fold improvements `28.1/42.7/48.3%`；mean `39.6822%` | offline representation 有前景 |
| ORB-003 | CUDA scalar increment `77.622 ms`，约为 upper budget 的 `388×` | online path 已停止；ORB-004/005 未运行 |

**English**

| Experiment | Key result | Verdict |
|---|---|---|
| EXP-006 | maximum path force `258.949 > 250` | teacher qualification failed |
| EXP-007 | coefficient `0.09`; six window checks passed | offline teacher authorized only |
| EXP-009 | N=1 `CUDA_ERROR_INVALID_HANDLE` | direct MACE-MTS stopped |
| EXP-010 | `0/6` folds beat the intercept; generalized `R²=−13.5934` | this torsion target failed |
| EXP-011 | overlap `0.02353<0.03`; decorrelated `22<25` | stopped before PMF |
| EXP-012 | offline `+13.9348%`, only `2/3` folds; online cost `1.81–1.89×` | no online benefit |
| EXP-013 | temperature z `5.61–6.83`; ESS/GPU-hour `932→218` | physical MTS rejected |
| EXP-014 | native compression screen failed | closed |
| EXP-016 | `3×500` surrogate frames, no real state crossing | `INCONCLUSIVE / SURROGATE_ONLY` |
| EXP-017 | min overlap `0.3913`; window-5 drift/`2σ=4.464` | `INCONCLUSIVE / STOPPED` |
| EXP-018 | drift z `1.134, 2.568, 1.381`; variance ratio `16.7599` | `INCONCLUSIVE / CLOSED` |
| EXP-019 | CUDA/NameError/endpoint uncertainty `1.2481>1.0` | formal repeats `0` |
| EXP-020 | offline improvement `55.5524%`; 9/9 checkpoints | representation signal qualified |
| EXP-021 | median `1.107419×`, P95 `1.114105×` | native density stopped |
| EXP-025 G0–G3/G4 oracle | ABI, CPU reference, CUDA brute force, CSR/mixed precision, and equivalence passed | correctness qualified in harness |
| EXP-025 G4 cost | median `1.123398×`, P95 `1.132172×` | runtime backend stopped |
| EXP-026 A1.1 | cost `1.04140×`, P95 `1.04692×` | cost qualified |
| EXP-026 A2 | cost `1.03206×`, P95 `1.03665×` | cost qualified |
| EXP-028 | long-run `addArg` defect fixed; stability ratio `1.033` | fixed and validated |
| EXP-027 U3 | 2/3 positive; median `+49.4%` | positive stress-test signal |
| EXP-027 U4 | baseline `(f_k)` reused for candidate | invalid for promotion |
| EXP-029 | per-arm calibration, 6 windows, 3 paired repeats | authorized, not started |
| EXP-030 | joint state-conditioned frozen score | design draft, not started |
| ORB-001 | fold improvements `28.1/42.7/48.3%`; mean `39.6822%` | offline representation promising |
| ORB-003 | CUDA scalar increment `77.622 ms`, about `388×` the upper budget | online path stopped; ORB-004/005 not run |

Only experiments with an auditable log, registry row, protocol, or machine-readable artifact are assigned an outcome. Missing numbers in the identifier sequence are not silently interpreted as failed or completed experiments, and `curated_project` copies are not counted as independent repeats.

只有具备可审计 log、registry row、protocol 或机器可读 artifact 的实验才在此赋予结果。编号序列中未出现的编号不会被擅自解释为失败或完成，`curated_project` 中的副本也不会当作独立 repeat。

### 8.3 Outer-λ/MACE 的原始假设 / Original Outer-λ/MACE Hypothesis

Stage 2 的困难不一定只来自 λ spacing，也可能来自局部环境引起的非线性 energy-gap variance 和 orthogonal slow modes。早期设计写成：

\[
U_\theta(x,\lambda)=U_{base}(x,\lambda)+\sum_m\phi_m(\lambda)b_m(x),
\]

with the endpoint-zero envelope

\[
w(\lambda)=\sin^2(\pi\lambda).
\]

The scientific objective was never merely to increase ML (R^2). The operational targets were

\[
ESS/GPU\ hour\uparrow
\quad\text{or}\quad
\sigma(\Delta G)/GPU\ hour\downarrow.
\]

因此所有 neural routes 必须依次通过 representation signal、energy/force/invariance、OpenMM parity、真实 CUDA cost、short dynamics 和 independent online utility。前一层成功不能替代后一层失败。

All neural routes must therefore pass representation signal, energy/force/invariance, OpenMM parity, realistic CUDA cost, short dynamics, and independent online utility in sequence. Success at an earlier layer cannot substitute for failure at a later layer.

### 8.4 EXP-006/007：teacher qualification

EXP-006 只在 path-force gate 失败，maximum force approximately `258.949 kJ mol⁻¹ nm⁻¹`, above the `250` gate. EXP-007 fixed the coefficient at `0.09` and passed six qualification checks.

EXP-006 failed only the path-force gate, with a maximum force of approximately `258.949 kJ mol⁻¹ nm⁻¹`, above the `250` limit. EXP-007 fixed the coefficient at `0.09` and passed all six qualification checks.

该通过只授权 teacher 进入离线研究，不授权 online MACE force、MTS 或 production sampling。teacher qualification 与 deployment qualification 是不同实验身份。

This pass authorized the teacher for offline research only. It did not authorize an online MACE force, MTS, or production sampling. Teacher qualification and deployment qualification are separate experimental identities.

### 8.5 EXP-009：direct MACE-MTS

状态为 `FAILED / STOPPED`。在最保守的 N=1 调用频率下即触发 `CUDA_ERROR_INVALID_HANDLE`，涉及 PythonForce/OpenMM-ML backend。因为基础频率已经失败，继续扫描 N=2/4/8 不会回答有意义的科学问题。

The status is `FAILED / STOPPED`. The most conservative N=1 frequency already triggered `CUDA_ERROR_INVALID_HANDLE` in the PythonForce/OpenMM-ML backend. Since the base frequency failed, scanning N=2/4/8 would not answer a meaningful scientific question.

### 8.6 EXP-010：cheap torsion CV

状态为 `FAILED`，但只否证该 teacher/target construction：

**中文**

| 指标 | 数值 |
|---|---:|
| preregistered LORO folds | `6` |
| 超过 intercept 的 folds | `0/6` |
| intercept RMSE | `21.5109` |
| 最佳一维二阶 RMSE | `22.1737` |
| generalized-force (R²) | `−13.59` |

**English**

| Metric | Value |
|---|---:|
| preregistered LORO folds | `6` |
| folds beating intercept | `0/6` |
| intercept RMSE | `21.5109` |
| best one-dimensional order-2 RMSE | `22.1737` |
| generalized-force (R²) | `−13.59` |

The protein-only atom-cut teacher and per-frame total-interaction target did not form a closed learnable target. This does not prove that every torsional bias is ineffective.

### 8.7 EXP-011：periodic torsion PMF

状态为 `FAILED / STOPPED`。Formal design used `24 windows × 3 repeats`, but minimum overlap was `0.02353 < 0.03` and decorrelated samples were `22 < 25`. The protocol therefore stopped before fitting or claiming a PMF benefit.

该失败推动路线从手工 torsion coordinate 转向直接 residual/gap learning，但不证明 torsion 与慢动力学无关。

The failure motivated a move from a hand-selected torsion coordinate to direct residual/gap learning, but it does not prove that torsions are irrelevant to slow dynamics.

### 8.8 EXP-012/013：TorchForce 与 physical MTS

EXP-012 offline direct-gap variance improvement was `13.9348%`, with only `2/3` folds improving. Online exploratory runs showed:

- TorchForce cost `1.81–1.89×`;
- all three ESS/GPU-hour comparisons worsened;
- idealized network cost approximately `1.83×`, above the `1.10×` budget.

EXP-013 physical MTS temperature diagnostics were:

**中文**

| MTS 间隔 | 温度 z-score |
|---:|---:|
| 2 | `5.61` |
| 4 | `5.79` |
| 8 | `6.83` |
| fused N=8 | `5.62` |

**English**

| MTS interval | Temperature z-score |
|---:|---:|
| 2 | `5.61` |
| 4 | `5.79` |
| 8 | `6.83` |
| fused N=8 | `5.62` |

Independent student N=1 passed basic health checks but reduced ESS by `18.88%`; ESS/GPU-hour dropped from approximately `932` to `218`. The generic TorchForce/MTS route therefore had no production benefit.

#### 8.8.1 EXP-014 native compression screen

EXP-014 tested whether the useful residual signal could be compressed into a sufficiently cheap native representation. It failed its promotion screen. This negative result separates “the teacher contains useful information” from “the information is expressible within the native cost budget,” and it motivated the purpose-built grouped-density representation in EXP-020.

EXP-014 检查有用 residual signal 是否能压缩进足够便宜的 native representation，结果未通过 promotion screen。该负结果把“teacher 含有信息”与“信息能在 native cost budget 内表达”明确分开，并推动后续转向 EXP-020 专用 grouped-density representation。

### 8.9 EXP-016～019：overlap、stationarity 与 reproducibility

**中文**

| 实验 | 数据/证据 | 状态 | 不能主张 |
|---|---|---|---|
| EXP-016 | 3 条连续轨迹、1500 帧、Δt=`1 ps`；114 次 weighted changes；alignment 通过但缺少 physical state/replica history | `INCONCLUSIVE / SURROGATE_ONLY` | true state-crossing 时间尺度 |
| EXP-017 | min overlap `0.3913`；min decorrelated `96`；drift `−0.5587`；drift/`2σ=4.464` | `INCONCLUSIVE / STOPPED` | 单个坏 λ edge 或 automatic λ insertion |
| EXP-018 | drift z `1.134, 2.568, 1.381`；variance ratio `16.7599` | `INCONCLUSIVE / CLOSED` | 已确认的系统性 drift |
| EXP-019 | CUDA 不可用；`NameError system_type`；endpoint uncertainty `1.2481>1.0` | 在 formal repeats 前失败 | baseline reproducibility |

**English**

| Experiment | Evidence | Status | Unsupported claim |
|---|---|---|---|
| EXP-016 | 3 continuous trajectories, 1500 frames, Δt=`1 ps`; 114 weighted changes; alignment passed but physical state/replica history absent | `INCONCLUSIVE / SURROGATE_ONLY` | true state-crossing timescale |
| EXP-017 | min overlap `0.3913`; min decorrelated `96`; drift `−0.5587`; drift/`2σ=4.464` | `INCONCLUSIVE / STOPPED` | one bad λ edge or automatic λ insertion |
| EXP-018 | drift z `1.134, 2.568, 1.381`; variance ratio `16.7599` | `INCONCLUSIVE / CLOSED` | confirmed systematic drift |
| EXP-019 | CUDA unavailable; `NameError system_type`; endpoint uncertainty `1.2481>1.0` | failed before formal repeats | baseline reproducibility |

EXP-019 diagnostic rescue `159.3165 ± 2.0618 kJ/mol` is diagnostic only and must never be presented as an endpoint free energy.

### 8.10 EXP-020：R1 grouped-density residual

R1 does not predict the full potential. It collects ligand–environment edges within `5 Å` of `41` ligand anchors, expands them in `16` radial bases with atom-type weights, performs an anchor-wise nonlinear readout, and aggregates a bounded residual basis.

The many-body character follows from

\[
\rho\left(\sum_j\phi(r_{ij})\right)\neq\sum_j\rho(\phi(r_{ij})).
\]

**中文**

| 资格项 | 结果 |
|---|---:|
| folds × seeds | `3 × 3` |
| checkpoints | `9/9` |
| fold-median gap-variance 平均改进 | `55.5524%` |
| D1 qualification | `true` |
| finite-difference absolute error | 降至 `4.5321×10⁻10` |
| finite-difference relative error | `2.6301×10⁻9` |
| nonparticipant force | `0` |
| maximum invariance error | `2.6021×10⁻18` |
| D2 qualification | `true` |
| CPU64 reference/export | `true` |

**English**

| Qualification item | Result |
|---|---:|
| folds × seeds | `3 × 3` |
| checkpoints | `9/9` |
| mean fold-median gap-variance improvement | `55.5524%` |
| D1 qualification | `true` |
| finite-difference absolute error | reduced to `4.5321×10⁻10` |
| finite-difference relative error | `2.6301×10⁻9` |
| nonparticipant force | `0` |
| maximum invariance error | `2.6021×10⁻18` |
| D2 qualification | `true` |
| CPU64 reference/export | `true` |

This establishes a strong held-out representation signal and suggests that pair-additive global representations were a bottleneck. It does not establish online utility.

### 8.11 通用后端的成本失败 / Cost Failure of Generic Backends

**中文**

| Backend | parity 范围 | 成本比 | 决策 |
|---|---|---:|---|
| N0 full-system CustomGB | semantic/cost probe | `1.6965×` | `STOP_FULL_SYSTEM_CUSTOMGB` |
| N1 per-anchor local CV | qualified | `6.0717×` | `COST_FAILED` |
| N2 OpenMM-Torch local Verlet | qualified | `61.2922×` | `COST_FAILED` |
| EXP-021 grouped-density skeleton | skeleton | median `1.107419×`；P95 `1.114105×` | `STOP_EXP021_NATIVE_DENSITY` |

**English**

| Backend | Parity scope | Cost ratio | Decision |
|---|---|---:|---|
| N0 full-system CustomGB | semantic/cost probe | `1.6965×` | `STOP_FULL_SYSTEM_CUSTOMGB` |
| N1 per-anchor local CV | qualified | `6.0717×` | `COST_FAILED` |
| N2 OpenMM-Torch local Verlet | qualified | `61.2922×` | `COST_FAILED` |
| EXP-021 grouped-density skeleton | skeleton | median `1.107419×`; P95 `1.114105×` | `STOP_EXP021_NATIVE_DENSITY` |

The preregistered upper budget was `1.10×`, with a stricter median qualification target of `1.07×`. The representation layer passed; generic deployment graphs did not.

### 8.12 EXP-025：专用 CUDA 假设、真实规模与冻结合同

EXP-025 proposed an OpenMM C++/CUDA Force that scans only local CSR/Verlet edges around the `41` ligand anchors, performs density reduction and nonlinear readout on the GPU, and returns a conservative force.

**中文**

| 系统量 | 数值 |
|---|---:|
| 总原子数 | `73,536` |
| ligand anchors | `41` |
| environment atoms | `73,495` |
| ligand × environment Cartesian pairs | `3,013,295` |
| maximum active 5 Å edges | `1,464` |
| observed environment-ID union | `796` |
| model parameters | `3,031` |

**English**

| System quantity | Value |
|---|---:|
| total atoms | `73,536` |
| ligand anchors | `41` |
| environment atoms | `73,495` |
| ligand × environment Cartesian pairs | `3,013,295` |
| maximum active 5 Å edges | `1,464` |
| observed environment-ID union | `796` |
| model parameters | `3,031` |

The key insight is that the model needs two local edge scans, 41 scalar anchor reductions, and a small typed MLP rather than full MACE message passing or triplets.

The mathematical contract requires standard bonded/LJ/PME terms to remain intact; the plugin emits only the raw local basis; the outer layer applies the coefficient and offset once; `A=0` returns exactly to baseline; forces remain conservative; dynamic solvent membership uses a neighbor list with skin; and no promotion occurs before correctness and cost both pass.

### 8.13 EXP-026：专用 backend qualification

The original G4 cost probe failed at ratio `1.123398`, with bootstrap P95 `1.132172`. Optimization therefore targeted measurable host/device control-plane overhead without changing the frozen scientific model.

EXP-026 A1.1 passed energy/force parity, mixed precision, serialization, context-update, error-path, and attribution checks. Its normative cost was median `1.04140×` with P95 upper `1.04692×`.

EXP-026 A2 retained correctness and improved the normative result to median `1.03206×` with P95 upper `1.03665×`. The decision `STOP_OPTIMIZATION_SUCCESS` means that optimization could stop and online utility testing could begin; it does not mean that the residual already improves production science.

**中文**

| 资格/归因量 | 数值 |
|---|---:|
| energy absolute error | `7.60572×10⁻5 kJ/mol` |
| G2 force error | `5.38032×10⁻4` |
| G3 force error | `5.38986×10⁻4` |
| G3−G2 force difference | `1.52588×10⁻5` |
| 移除的 H2D 调用次数 | 约 `2,206` |
| 移除的 H2D 字节数 | 约 `324.7 MB` |
| 移除的 D2H 调用次数 | 约 `2,210` |
| 移除的 D2H 字节数 | 约 `325.2 MB` |

**English**

| Qualification/attribution quantity | Value |
|---|---:|
| energy absolute error | `7.60572×10⁻5 kJ/mol` |
| G2 force error | `5.38032×10⁻4` |
| G3 force error | `5.38986×10⁻4` |
| G3−G2 force difference | `1.52588×10⁻5` |
| H2D calls removed | approximately `2,206` |
| H2D bytes removed | approximately `324.7 MB` |
| D2H calls removed | approximately `2,210` |
| D2H bytes removed | approximately `325.2 MB` |

The attribution shows why optimization succeeded: it removed repeated transfer/control-plane work while preserving the model contract. Performance qualification remains separate from scientific promotion.

### 8.14 EXP-027/028：长轨迹缺陷与修复

Fourteen CUDA kernels repeatedly called `addArg()` at every step, causing the kernel-argument container to grow and runtime to deteriorate approximately from `2.74` to `6.22 ms/step` over a long run.

EXP-028 registered parameter slots once and subsequently used `setArg(fixed_idx, ...)`. After the fix, runtime changed only from approximately `2.6537` to `2.7406 ms/step`, the stability ratio was `1.033`, RSS remained approximately constant, and correctness regressions passed.

### 8.15 U3、U4 与 candidate-specific `(f_k)` 错配

The repaired window-0 U3 produced positive utility differences in `2/3` repeats, a median relative improvement of `+49.4%`, and candidate/baseline GPU-hour ratios of approximately `1.025–1.041`.

However, the candidate reused baseline (f_k), so U3 is a `BASELINE_FK_TRANSFER / NO_RECALIBRATION` stress test. U4 completed all six windows but retained the same calibration violation:

```text
EXP027_U4 = INVALID_FOR_PROMOTION_BASELINE_FK_REUSED_FOR_CANDIDATE
```

Its actual budget was `6 windows × 2 arms × 3 repeats × 60,000 steps = 2.16 million steps`; no window was dropped. Nevertheless, every repeat failed the joint requirements `min_decorrelated_samples ≥ 20` and `max_endpoint_uncertainty ≤ 1.0`. The approximate `(r,s,r/s)` values were `(0.7357,1.081,0.68)`, `(0.9294,1.077,0.86)`, and `(4.1762,1.056,3.95)`. Candidate-minus-baseline differences concentrated in high-coefficient windows 1–3 and changed sign across repeats, which is more consistent with poor mixing/high variance than with one fixed path bias.

U4 的实际预算为 `6 windows × 2 arms × 3 repeats × 60,000 steps = 2.16 million steps`，且没有 dropped window；但所有 repeats 都未同时满足 `min_decorrelated_samples ≥ 20` 与 `max_endpoint_uncertainty ≤ 1.0`。三个 repeat 的近似 `(r,s,r/s)` 分别为 `(0.7357,1.081,0.68)`、`(0.9294,1.077,0.86)`、`(4.1762,1.056,3.95)`。candidate−baseline 差主要集中在高系数 windows 1–3，并在 repeats 间变号，更符合 mixing 不足/高方差，而非固定 path bias。

Fixed but mismatched (f_k) values do not automatically create asymptotic bias if the complete bias history and target energies are used. They do reduce mixing efficiency and can inflate finite-budget variance, especially where the residual amplitude is large.

### 8.16 EXP-029/030：当前唯一决定性实验 / The Current Decisive Experiment

EXP-029 compares

\[
\Theta_b=\{0,\mathbf f_b^*\},\qquad
\Theta_c=\{\phi_*,\mathbf f_c^*(\phi_*)\}.
\]

Each arm must independently calibrate and freeze (f_k) under its actual sampling Hamiltonian. The two arms use paired initial states, velocities, and seed families, but their trajectories and (f_k) evolve independently. All six windows and at least three paired repeats are required.

ITT accounting includes warmup, calibration, failed attempts, rescue, production, and ledger overhead. Promotion requires ΔG consistency, TMBAR/coverage/endpoint-uncertainty passes, at least `2/3` candidate utility wins, and the preregistered median improvement.

EXP-030 is a `DESIGN_DRAFT_NOT_STARTED` joint state-conditioned score. It formalizes the complete frozen candidate score but keeps the residual sampling-only. It does not authorize online co-adaptation of (phi) and (f_k).

The proposed score is deliberately rank-1 rather than an unconstrained state-conditioned network:

\[
C_{w,k}(R;\phi,f)=A_{w,k}B_\phi(R)-f_{w,k},
\qquad
g_{w,k}=f_{w,k}-A_{w,k}B_\phi(R),
\]

\[
u^*_{w,k}=u^0_{w,k}+C_{w,k}=u^0_{w,k}-g_{w,k},
\qquad
u_{w,\mathrm{mix}}=-\log\sum_k\exp[g_{w,k}-u^0_{w,k}].
\]

In physical units,

\[
X_{w,k}=U^0+A(U_B-U_{\mathrm{offset}})-f,
\qquad
V_{\mathrm{mix}}=-k_BT\log\sum_k\exp[-\beta X_{w,k}].
\]

Here `f` is an additive free-energy offset for each window/state and must not be placed inside `tanh`. Baseline and candidate must independently calibrate their own `f*`. The immutable ledger must retain `target_state_energies`, `sampling_bias_energy`, and `base_energy` separately.

ITT utility is defined as

\[
\eta=\frac{N_{\mathrm{eff}}}{C_{\mathrm{warmup}}+C_{\mathrm{calib}}+C_{\mathrm{freeze}}+C_{\mathrm{rescue}}+C_{\mathrm{prod}}+C_{\mathrm{ledger}}},
\qquad D_r=\log\eta_c-\log\eta_b.
\]

Promotion requires `D_r>0` in at least `2/3` repeats, median relative improvement of at least `0.10`, inclusion of every cost term, and simultaneous TMBAR, coverage, shared-target ΔG-consistency, and candidate-health passes. Until that experiment is run, EXP-030 remains a design, not a result.

这里的 `f` 是逐 window/state 的 additive free-energy offset，不能放进 `tanh`。baseline 与 candidate 必须分别校准自己的 `f*`，ledger 则必须拆开保存 `target_state_energies`、`sampling_bias_energy` 与 `base_energy`。promotion 要求至少 `2/3` repeats 的 `D_r>0`、median relative improvement 至少 `0.10`、所有成本全部计入，并同时通过 TMBAR、coverage、shared-target ΔG consistency 与 candidate health gates。

### 8.17 ORB 与 representation 边界 / ORB and the Representation Boundary

ORB improved `3/3` offline LORO folds by approximately `28.1%`, `42.7%`, and `48.3%`, with a mean improvement of about `39.7%`.

The matched CUDA scalar increment was `77.622 ms`, while the budget was `0.1–0.2 ms`; this is approximately `388×` the upper budget. ORB-003 is therefore `OFFLINE_TEACHER_ONLY`, and ORB-004/005 were stopped.

### 8.18 Production wiring 边界 / Production-Wiring Boundary

The IBS engine supports `residual_basis_force`, `residual_state_coefficients`, and `residual_energy_offset_kj_mol`, shares an `exp025_residual_basis` CV, and validates the two-CV-per-state layout. With these arguments set to `None`, residual behavior is off and the original IBS path remains compatible.

However, the current `abfe_pipeline.py::_build_window_system` and `runabfe.py` production entry do not pass the residual arguments. The EXP-027 preregistration records zero production-entry hits for `residual_basis_force/LocalManyBodyResidualForce`. Therefore:

**中文**

| 层级 | 当前状态 |
|---|---|
| IBS engine capability | 已实现 |
| standalone candidate harness | 已实现并完成实验验证 |
| dedicated CUDA A2 cost/correctness candidate | 在其 scoped harness 中合格 |
| `runabfe` production-pipeline integration | **未接线** |
| full six-window, two-leg production utility | **尚未展示** |
| production target identity | residual-off physical baseline |

**English**

| Layer | Current status |
|---|---|
| IBS engine capability | implemented |
| standalone candidate harness | implemented and experimentally exercised |
| dedicated CUDA A2 cost/correctness candidate | qualified in its scoped harness |
| `runabfe` production-pipeline integration | **not wired** |
| full six-window, two-leg production utility | **not demonstrated** |
| production target identity | residual-off physical baseline |

IBS engine 已支持 residual basis、state coefficients 与 energy offset，也有 standalone candidate harness；但当前 `abfe_pipeline.py::_build_window_system` 与 `runabfe.py` 的正式生产入口并未传递这些参数。因此 EXP-027/028 是 standalone/experimental sampling-bias 证据，不是当前完整 ABFE production pipeline 的结果。任何 advisor-facing 结论都必须以 residual-off baseline 为当前 production identity。

---

### 8.19 本章最终判决 / Final Verdict of This Chapter

EXP-020 已经证明 lightweight local-many-body representation 的强离线信号；EXP-026 证明专用 CUDA backend 在 scoped correctness 与 cost 上合格；EXP-028 修复并验证长程 argument-slot growth。它们共同授权 **online utility experiment**，不授权 scientific production promotion。

EXP-020 established a strong offline signal for a lightweight local-many-body representation; EXP-026 qualified the dedicated CUDA backend for scoped correctness and cost; and EXP-028 repaired and validated long-run argument-slot growth. Together, these results authorize an **online utility experiment**, not scientific production promotion.

U3/U4 使用 baseline `(f_k)` transfer，因而不能回答 candidate 在自己的 residual-active sampling Hamiltonian 下是否有效。EXP-029 必须让 baseline 与 candidate 分别 calibration、freeze 和 production，并在全部 6 windows、3 个 paired repeats 上同时比较 ΔG consistency、coverage、decorrelation、reweighted ESS、between-run behavior 与 all-inclusive ITT GPU-hour。

U3/U4 transferred baseline `(f_k)` values and therefore cannot determine whether the candidate is useful under its own residual-active sampling Hamiltonian. EXP-029 must calibrate, freeze, and run the baseline and candidate separately, then compare ΔG consistency, coverage, decorrelation, reweighted ESS, between-run behavior, and all-inclusive ITT GPU-hours across all six windows and three paired repeats.

因此当前准确判决是：`SOFTWARE_CORRECTNESS_AND_COST_QUALIFIED / LONG-RUN DEFECT_FIXED / OFFLINE_SIGNAL_POSITIVE / ONLINE_UTILITY_UNPROVEN / NOT_PRODUCTION_PROMOTED`。完整 Frozen MACE 仍是 future representation ablation；Route 1 仍是独立 target-path research program；二者都不能在 EXP-029 之前混入当前 production identity。

The accurate current verdict is `SOFTWARE_CORRECTNESS_AND_COST_QUALIFIED / LONG-RUN DEFECT_FIXED / OFFLINE_SIGNAL_POSITIVE / ONLINE_UTILITY_UNPROVEN / NOT_PRODUCTION_PROMOTED`. Full Frozen MACE remains a future representation ablation, while Route 1 remains a separate target-path research program. Neither may be merged into the current production identity before EXP-029.
