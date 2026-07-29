# 当前行动清单

更新：2026-07-29。完整 2026-07-27 审计长记录已移入 [archive/TODO-2026-07-27-full.md](archive/TODO-2026-07-27-full.md)。历史原文见 [archive/todolist-2026-07-20.md](archive/todolist-2026-07-20.md)，审计证据见 [status/AUDIT_STATUS.md](status/AUDIT_STATUS.md)，运行验证见 [status/VALIDATION_MATRIX.md](status/VALIDATION_MATRIX.md)。

## 当前决策

- **Boresch 二面角符号反号根因已修（BOR-01，2026-07-29）。** `abfe_core.py` 四份手写
  二面角副本都返回 **−φ**，其中 `calc_boresch_from_last_frame` 会用镜像参考值覆盖
  正确的 `boresch_simple.json`。现统一走 `abfe_core.boresch_dihedral_rad()`
  （`abfe_core.py:1142`）。全量记录见
  [handoffs/BORESCH_DIHEDRAL_SIGN_HANDOFF.md](handoffs/BORESCH_DIHEDRAL_SIGN_HANDOFF.md)。
- **当前生产结果（2026-07-29 11:02，符号修复后全量）**：

  ```
  复合物腿 ΔG_cplx = 181.00 ± 1.76    溶剂腿 ΔG_solv = 157.84 ± 1.79
  Boresch attachment = 4.39 ± 0.08    解析修正 = −38.76    APBS = 0.00
  ΔG_bind = −23.16 ± 2.51 kJ/mol = −5.54 ± 0.60 kcal/mol
  参考 result.txt total = −6.279 ± 0.457 → 差 +0.74，0.98σ，1σ 内一致
  ```

  对比 07-06 符号 bug 期的 −9.76 kcal/mol，**2.7 kcal 的改善基本全部来自本次符号修复**
  （复合物腿 192.89 → 181.00）。⚠️ ±0.60 是乐观的，真实约 **±1.0 kcal/mol**。
  这组数取代此前所有 ΔG_bind 候选（−2.121 / −3.460 / −3.4797）与 attachment
  5.601 ± 0.223 —— 那些都是符号 bug 期的值，不得再引用。
- **口径纪律：`result.txt` 只有 total 可比，分项不可比。** 它是旧方法的参考值，本仓库
  实现的是 IBS，分项拆法本来就不同；restraint 与被限制腿的采样还会结构性抵消，把
  charging 的差和 restraint 的差当成两条独立线索是重复计数。**「与 result.txt 差
  2.8 kcal，逐项归因」这条旧结论已整体撤销**，P1-18 因此关闭（见该条）。
- **vdW（stage2）只能用 TMBAR，本轮一字未动。** 不得引入 BAR / TI / 全帧主值 /
  √g σ / bootstrap σ。07-28 曾有一批这样的扩张被整批撤回，见 P1-21 末尾与 P1-22。
  charging（stage1）主值为相邻 BAR（P1-21，2026-07-28）。

- **当前没有尚未修复、会阻挡生产全量重跑的 P0。**
- P0-10 的生产代码已修；旧复合物腿采样作废，07-29 那轮已是修复后的 fresh full rerun。
- P0-11 已结案（2026-07-28）：溶剂腿盒子缺陷已修，生产默认 `SOLVENT_PADDING_NM = 1.5`（盒边 4.257 nm）。
  ⚠️ 当时「三档盒子对 ΔG_bind 零影响」的结论口径已被 07-29 修正：pad1.5→pad2.4 的
  −7.15 kJ/mol **不是**有限尺寸效应，而是 vanishing 腿的跑间散布（同盒子两次独立跑
  差 2.34σ，比跨盒子 1.13σ 还大）。见 P1-19。
- **当前最高优先的物理问题是 vanishing 腿的误差估计偏小（P1-19），不是盒子尺寸。**
  它**不是代码 bug**（渐近协方差在正确地算 within-run 统计误差），而是不确定度量化
  加采样不足；同一带里真正的代码 bug 是 P1-23 的 σ 采纳 fail-open。
- 新 DEXP 已按 `experiments/DEXP_KERNEL_PHYSICS_ISSUES.md` 冻结为
  pair-specific LJ-matched 解析核，并拆出 `dexp_NEW.py` 生产入口；旧 Orb 全局 fitter
  不再属于生产协议，旧参数 JSON fail closed。
- 主 TODO 只保留仍需行动的事项；已修复项目、长表格、诊断过程与 2026-07-27 复审细节都归档。

## P0 / P1 当前待办

- [x] **P0-6 / P0-7 / ATT-06 / ATT-07：旧 DEXP fitter/生产 Hamiltonian 契约分裂。**
  GitHub [#47](https://github.com/Cedrus810/openmm_IBS_dev/issues/47)，2026-07-28 结案。
  新生产协议不再拟合全局 `r0_vdw/A_fit/B_fit`，也不含 `offset_c0/offset_c1`；
  `r0_ij/epsilon_ij` 由原始 LJ `sigma/epsilon` 逐 pair 解析生成，只保留
  `alpha_vdw/beta_vdw` 等新核配置。`dexp_NEW.DEXPProductionConfig` 与
  `abfe_core.DEXPSurrogatePotential.from_dict()` 均拒绝旧字段，旧 JSON 不可能再被
  静默解释成另一套 Hamiltonian。验证：新契约字段测试、旧字段 fail-closed 测试、
  二原子 `U(r0)=-epsilon_ij`/lambda 端点测试及 Atenolol 73,536 粒子真实构建烟测。



- [x] **P0-11：溶剂腿盒子只有 3.000 nm（= 2×padding，溶质尺寸贡献为 0）。已修并实测结案（2026-07-28）。**

  `output_lrc_fix/topology_solvent.cif` 的 `_cell.length_a/b/c` 全是 30.0000 Å，
  而 `SOLVENT_PADDING_NM = 1.5` 的语义是「配体每侧 1.5 nm 溶剂」。
  配体最长轴 1.257 nm，正确盒边应是 1.257 + 3.0 ≈ 4.26 nm；
  落地的 3.000 = 2×1.5 恰好等于溶质尺寸完全没算进去。
  长轴方向配体表面到盒面只剩 0.87 nm，第二水化层直接和周期镜像重叠。
  对照复合物腿 9.09947 nm（继承自 `solv_ions.gro`，那一侧没问题）。
  时间戳确认是 2026-07-27 18:21 本轮新建，不是旧缓存。

  **已修**：新增 `abfe_core.solvent_box_edge_nm()`（`gmx editconf -d` 语义），
  两个 builder 都改成显式传 `boxSize=`，建完后校验 OpenMM 实际盒子等于请求值，
  外加最小镜像检查 `盒边 > 2×cutoff`。`SOLVENT_CACHE_PROTOCOL_VERSION` **3 → 4**，
  所有 3.000 nm 旧缓存整体作废。manifest 新记
  `box_edge_nm` / `ligand_longest_axis_nm` / `padding_nm` / `nonbonded_cutoff_nm`。

  **同批修掉的水模型不一致**：`SolventLegRunner.build_solvent_system` 原本写死
  `amber14/tip3pfb.xml`（TIP3P-FB），而复合物腿走 GROMACS
  `amber14sb_OL15_fs1.ff/tip3p.itp`（普通 TIP3P），σ/ε/电荷都不同。
  本轮实际走的是 `runabfe.build_and_cache_solvent_leg`（manifest 记的是 tip3p），
  **没踩到**，但那是条活的分叉路径。现改为
  `abfe_core.resolve_water_model_xml()` 从复合物 `.top` 的 `#include` 反推，
  认不出来 fail closed，绝不回退默认值；`solvent_forcefield` 也进缓存身份指纹。

  **已测并结案（2026-07-28）**：`tools/diagnostics/diagnose_solvent_box_scan.py` 跑了 3.000 / 4.257 / 6.057 nm：

  | 盒边 | 粒子 | decharging | vanishing | 总计 | ΔG_bind |
  |---|---|---|---|---|---|
  | 3.000 | 2,574 | 62.887 | 101.825 | 162.783 | −2.121 kcal |
  | 4.257 | 7,417 | 63.115 | 101.639 | 162.826 | **−2.111 kcal** |
  | 6.057 | 21,316 | 64.249 | 94.491 | 156.812 | — |

  **结论：盒子修复对 ΔG_bind 零影响（−2.121 → −2.111）。生产默认定
  `SOLVENT_PADDING_NM = 1.5`（盒边 4.257 nm，45 min，vs 3.000 nm 的 37 min）。**

  - stage1 有硬结论：三轮 split-half 的 z 全 < 2，σ≈1.0 可信；量到 6.057 nm
    decharging 才 64.25，而 `result.txt` 反解要 68.1。**盒子填不上 charging 的缺口**
    （见 P1-18）。
  - stage2 无结论也不再追：per-window σ 被低估 2–4 倍（见下），真实精度 ±3–5 kJ/mol，
    在这个精度下 101.8 / 101.6 / 94.5 无显著差异，但也藏得下 ≲1 kcal 的盒子效应。
    按用户 2026-07-28 决定，**不再为此加采样**；6.057 那轮的 −7.33 不作为物理结论。
  - LRC 漏溶剂粒子数的假设被**否掉**：若 `lrc_coeff/V` 漏了 N，4.257 nm 处应已掉
    ~5.4 kJ/mol，实测掉 0.19。

  **注意**：协议号升到 4，下一次生产 `--resume` 会重建溶剂腿缓存并重跑溶剂腿。

  代码改动只过了 `py_compile` 与水模型解析的纯 stdlib 复现（真 `topol.top` 唯一命中
  `tip3p` → `amber14/tip3p.xml`）；盒子逻辑本身已由上述三轮 GPU 运行实跑验证。

- [x] **P1-21：charging（stage1/decharging）主值改为相邻 BAR + FD-TI 一致性门（2026-07-28 结案，已接生产）。**

  `ESTIMATOR_ANALYSIS_PROTOCOL_VERSION = 2`（`ibs_engine.py:73`，与采样协议号严格分离）。
  同一批 `decharging_pme_u_kn.npy` 上四个口径实测：

  | | 去相关 MBAR | 全帧 MBAR | 相邻 BAR | 重加权 FD-TI |
  |---|---|---|---|---|
  | complex | 64.4113 | 65.0032 | **65.0762 ± 0.6148** | 65.1262 |
  | solvent | 62.8865 | 63.4637 | **63.4117 ± 0.6604** | 63.6378 |

  BAR / FD-TI / 全帧 MBAR 三者一致，**去相关 MBAR 两条腿共同偏低 0.5–0.6 kJ/mol**。
  正确命名是**「自相关子采样导致有限样本点估计不稳定」**——出问题的是丢帧之后选中的
  那个有限子集，**不是「MBAR 本身有偏」**，MBAR 没有被否定。BAR 逐边只用相邻两态
  自己的样本，不受全局选帧影响。

  ⚠️ **两条腿的偏移基本抵消：对 ΔG_bind 的净移动只有 −0.140 kJ/mol = −0.033 kcal**
  （−3.4464 → −3.4797）。它是个真问题，但**不是** P1-18 那 1.3 kcal charging 缺口的
  解释，两件事不要混。与 `result.txt` 的缺口因此是 **2.799 kcal**。

  实现：`adjacent_bar_chain()`（attachment 也复用它，`_attachment_bar_chain` 改为
  delegate）、`reweighted_fd_ti()`、`stage1_estimator_crosschecks()`、
  `stage1_ti_consistency_gate()`、`TraditionalMBARAnalyzer.solve(primary_estimator=...)`
  （默认值 = 旧行为），生产接线在 `abfe_pipeline._CHARGING_ESTIMATOR_KWARGS()` +
  decharging 两处调用点。TI 门容差 **0.5 kJ/mol**（实测 |BAR−TI| = 0.050 / 0.226），
  刻意不沿用 attachment 的 1.0。缓存失效由既有 `_code_hash()` 承担，未新增指纹字段。
  回归：`tests/test_charging_estimator_protocol.py` 14 条；全离线套件 434 passed。

  ⛔ **同批教训**：此前一版把全帧主值、√g σ 缩放、独立样本数门改口径、移动块
  bootstrap、σ evidence、crosscheck 接线加进了 `GlobalMBARAnalyzer`（= vdW/stage2），
  而 charging 的参数加了却没人传——**该改的没接上线，不该改的全改了**，已整批撤回。
  **vdW 只能用 TMBAR**，理由与禁令写在 `ESTIMATOR_ANALYSIS_PROTOCOL_VERSION` 注释里。

- [ ] **P1-22：vdW/stage2 的帧选择与 σ 口径（独立课题，不得顺手做进估计量层）。**

  两件已知的事，但都必须单独设计、单独验证，**不要**再在
  `GlobalMBARAnalyzer.solve_stage_integrated` 上顺手扩张：

  1. **点估计**：去相关选帧在有限样本下不稳。曾用（已撤回的）全帧模式量到
     complex 143.1162 / solvent 101.6877，相对去相关分别移动 −3.708 / −0.136 kJ/mol，
     且 complex 的位移集中在 win0 (−1.884) 与 win2 (−2.122)，而全帧值正好落在两个
     半程 142.638 / 143.235 之间。**留作历史观测，不是当前结论。**
  2. **σ**：全帧 MBAR 的渐近协方差是 **naive σ**（把相关帧当独立样本，低估约 √g）；
     换全帧还会让两道门变松——独立样本数门必须吃 N/g（否则"≥20"对 500 帧恒真）、
     端点 σ 必须乘 √g。零 GPU 的正解是**移动块 bootstrap**（每个 replicate 重新执行
     local-TMBAR 与窗口拼接、扫块长取最保守值），而不是块间 SEM（dof 只有 4，
     那个 SEM 本身就是噪声量）。stage1 上实测 bootstrap σ ≈ 渐近 σ 的 1.9 倍。

  **⛔ 无论怎么设计，都不得给 vdW 引入 BAR 或 TI**：BAR 前提是两个端点系综各自
  有样本（IBS 物理 λ 行 n_k=0）；TI 前提是势对 λ 线性（vdW softcore 非线性，且从未
  落盘 ∂U/∂λ）。

- [ ] **P1-23：σ 采纳路径的 fail-open（`ibs_engine.solve_stage_integrated`）。**

  `inflate_sigma_from_split_half=True` 会替换 `total_error` 与逐段
  `uncertainty_kJ_mol`，却**不更新 `max_endpoint_uncertainty_kJ_mol`、也不重判
  `converged`**——等于"抬高了 σ 而门还在读抬高前的小 σ"。该标志默认关闭，所以不是
  P0；但任何 σ 采纳路径都应当重算端点 σ 门并重判收敛。属 vdW 侧，与 P1-22 一并设计。

- [ ] **P1-19：per-window σ 系统性低估 2–4 倍，五道门全都看不见（2026-07-28；2026-07-29 并入原 BOR-04 的跨跑证据，当前最高优先物理项）。**

  **定性：这不是代码 bug。** `segment_error` 取 pymbar 渐近协方差，代码在正确地做它
  声称的事——渐近协方差算的就是「单次运行内、样本 iid 且已收敛」前提下的统计误差。
  缺陷在于**这个数被当成别的东西用**（进 ΔG_bind 误差棒、进收敛门），而它低估真实
  跑间散布 2–3 倍。根子多半在采样侧（vanishing 腿系综仍在慢移），σ 偏小只是盖住了它。
  归类为**不确定度量化 + 采样不足**，不是崩溃级缺陷。
  ⚠️ 与此相邻的 **P1-23 才是真 bug**（σ 采纳路径 fail-open：抬高了 σ 而门还在读小 σ）。

  措辞统一：这是**渐近协方差在"看起来独立、系综仍在慢移"时的低估**，
  与 P1-21 那条「自相关子采样导致有限样本点估计不稳定」是两件不同的事；
  两者都**不构成「MBAR 本身有偏」**。σ 口径的修法见 P1-22（移动块 bootstrap）。

  新增 `ibs_engine.split_half_drift_diagnostics()`：把每个窗口的帧按时间切前后两半
  各解一遍，判据 `z = |后半−前半| / (2σ_win)`（两半各自 SE≈√2σ，其差 SE≈2σ）。
  ⚠️ 两个半程走**与主值相同的帧选择**（各自重新去相关）。2026-07-28 曾把它改成强制
  全帧并被撤回，所以下面那三张表**继续有效、不需要重算**（实测复核：complex stage2
  win1 仍是 5.25×2σ，总 σ 1.5913 → 3.3771、×2.12）。
  `solve_stage_integrated()` 每次解完自动挂 `split_half_diagnostics` 并落盘。
  **默认只诊断不阻断**；传 `split_half_max_z=2.0` 才否掉 `converged`。

  溶剂盒扫描三轮 18 个窗口，**5 个超 2σ**（σ 正确时期望 0.8 个，二项概率 ~0.1%）：

  | | win0 | win1 | win2 | win3 | win4 | win5 |
  |---|---|---|---|---|---|---|
  | 3.000 | 0.69 | **2.05** | 0.85 | 0.88 | 1.70 | 1.34 |
  | 4.257 | **2.29** | 1.40 | 1.48 | 1.02 | **4.34** | 0.40 |
  | 6.057 | 0.20 | **3.05** | 0.16 | 0.60 | **2.93** | 0.64 |

  **window 4 是铁证**：3.000 那轮它在所有现有指标上都是优等生
  （`absolute_ess` 348.6、`n_decorr` 357、g 1.40、每个 λ 的 `ess_ratio` ≥ 0.976、
  `min_occupancy` 0.944），`σ_win` 只有 0.236，实漂 0.80–1.50，z 到 4.34。
  **没有伴随任何 ESS/overlap 退化**——问题不在采样质量的代理量，在 σ 本身。

  根因位置：`GlobalMBARAnalyzer.solve_stage_integrated` 里
  `segment_error = float(local["dDelta_f"][join_idx, end_idx])`（相对 def 偏移 +441），
  直接取 pymbar 的**渐近协方差**。渐近协方差假定样本独立同分布且已收敛，
  而 window 4 恰是「看起来独立、系综仍在慢移」的情形。

  拟议修法（零 GPU 成本，用已算出的量）：`σ_win ← max(σ_MBAR, |漂移|/2)`。
  改后 win4 的 σ：0.236→0.402（3.000）、0.173→0.750（4.257）、0.196→0.575（6.057）。

  **影响**：stage2 总 σ 会从 1.10–1.47 变成 3–5 kJ/mol，ΔG_bind 误差棒从
  ±0.62 kcal/mol 变成 ±1.0–1.5，与 `result.txt` 那 4.16 kcal 的差距性质从
  「约 7σ」变成「约 3σ」。**stage1 不受影响**（三轮 z 全 < 2，已验证）。

  工具：`tools/diagnostics/diagnose_split_half_convergence.py --stage both`（纯离线，秒级）。

  **跨跑证据（2026-07-29 从原 BOR-04 并入，与上面的 split-half 是同一现象的两个视角）：**
  同 padding 1.5、同一个盒子（`box_edge_nm=4.257`、Na=7 Cl=7）两次独立跑，
  vanishing 差 **4.675 kJ/mol = 2.34σ**，**比跨盒子差异（1.13σ）还大**；
  decharging 反而干净（同盒子 0.24σ、跨盒子 1.14σ）。所以问题精确定位在 **vanishing 腿**。

  | 运行 | decharging | vanishing | 总计 |
  |---|---|---|---|
  | pad 1.5 scan（07-28） | 63.115 ± 1.104 | **101.639** ± 1.100 | 162.826 ± 1.559 |
  | pad 1.5 主跑（07-29 11:02） | 62.800 ± 0.671 | **96.964** ± 1.663 | 157.836 ± 1.793 |
  | pad 2.4 scan（07-28） | 64.249 ± 1.078 | **94.491** ± 1.431 | 156.812 ± 1.792 |

  **推论：pad1.5→pad2.4 那 −7.15 kJ/mol 不是有限尺寸效应，就是这个跑间散布。**
  这同时否掉了 P0-11 当时「三档盒子对 ΔG_bind 零影响」的口径。

  **行动顺序（不得跳步）：**

  1. **在固定 padding 1.5 下再跑 1–2 次重复**，把 vanishing 的真实跑间 σ 钉下来
     （现在只有 2 个样本，σ ≈ 3.31 是极粗估计）。这是判断后续任何盒子扫描结果是否
     显著的**唯一基准**。
  2. 重新评估上面那条 `σ_win ← max(σ_MBAR, |漂移|/2)` 下界是否该默认启用
     （目前 `默认未采用`）。
  3. **只有 1 做完之后**才决定要不要为盒子尺寸加档。**现在别再扫盒子**——
     `--padding 3.0` 单跑一档分不清散布与尺寸效应，纯属浪费 2–3 h。
     若最终要改生产默认，改 `runabfe.py:101` 的 `SOLVENT_PADDING_NM`（目前 1.5）；
     改后 `solvent_cache_manifest.json` 的 `identity.padding_nm` 不匹配会自动重建缓存，
     **无需手工删文件，更不要把扫描目录的 `final_results.json` 拷进
     `output_lrc_fix/solvent_leg/`**（那是改产物不改生成器）。

- [x] **P1-8 / ATT-14：IBS 中 DEXP cutoff/switch 配置失效。**
  GitHub [#54](https://github.com/Cedrus810/openmm_IBS_dev/issues/54)，2026-07-28 结案。
  `_create_softcore_force()` 现对 DEXP 从已规范化参数读取 `cutoff_distance` 与
  `switch_width`，并使用 `switch = cutoff - width`；默认得到 0.70/0.50 nm。
  传统 softcore 仍保持 1.20/1.00 nm，二者不再共享硬编码。

- [x] **P1-17：热力学循环缺「打开 Boresch 限制」这条腿。已补并实测（2026-07-28）。**

  **实现**（已在 GPU 上跑通两轮，见下方实测结果）：

  - `ibs_engine.run_boresch_attachment_leg()`：λ_boresch 0→1 的顺序独立窗口。
    配体全程完全耦合，不碰 `lambda_coul`/`lambda_vdw`。
    **主估计量 = 相邻 BAR**（`_attachment_bar_chain`），**TI**（`_attachment_ti`）
    作一致性门，去相关 MBAR 降级为诊断。
    **默认 λ 表 `[0.0, 0.1, 0.35, 1.0]`（4 态）**。初版 12 态是照搬 decoupling 腿的
    直觉加密的，理由写的是「λ→0 相邻态重叠极差」——**实测正好相反**：每条边
    ⟨Δu⟩ ≤ 0.33 kT（λ→0 那几档 0.04/0.08/0.15 反而最好），而 BAR 的合理区间是
    1–2 kT，等于密了 3–6 倍。那条论证的前提是「限制是该坐标的主约束」，
    但配体在口袋里、蛋白本身已经把它按住了，U_B 在所有 λ 下都被压在 1.6–5.6 kT。
  - `ibs_engine.add_scalable_boresch_restraint()`：`fixed_lam=None` 的变体。
    生产用的 `_add_physical_boresch_restraint` 传 `fixed_lam=1.0`，
    `LambdaDependentBoreschForce.__init__` 会把 1.0 编译进表达式字符串、
    **连全局参数都不注册**，扫不了 λ。
  - **符号**：λ 升序 0→1，MBAR 的 `delta_G = f[K-1] − f[0]` 直接就是 ΔG(A′→A)，
    全链路没有任何取负号的地方。字段名 `attachment_delta_G_kJ_mol` 刻意不同于
    `total_delta_G`，防止被当成同类量反号求和。
  - **严格下界 fail closed**：Boresch 势处处 ≥0 ⟹
    `ΔG = −kT·ln⟨exp(−βU)⟩ ≥ 0`。求解器与 `compute_final_results` 两处都拒绝负值。
  - 另有：力组占用检查（靠单独取力组能量拿 U_B，混进别的力会**静默**算错）、
    U(λ) 线性性两点自检、起跑前锚点几何闸（θ 近 0°/180° 时 1/sinθ 梯度奇点，
    否则只表现为 MBAR 那句「u_kn 含 NaN」）。
  - **门**：BAR vs TI 分歧 > max(1.0 kJ/mol, 3σ) → 失败；split-half |漂移|/2σ > 3
    → 失败；ΔG < 0 → 失败；MBAR 偏离只警告；HREMD 若被启用且零 round trip → 失败。
  - `abfe_pipeline.compute_final_results`：`dg_phys = dg_attach + dg_decharge + dg_vdw`，
    并在 `final["boresch_attachment"]` 单列；缺 stage0 时打警告说明循环未闭合。
  - `runabfe.py --only-boresch-attachment`：增量模式。**stage1/stage2 本就是在
    受约束系综里测的，补这一项不改变它们**，所以约 2–3 h，不必重跑 7 h 的整条腿。
    会校验冻结 stage1/stage2 的 Boresch 指纹与当前一致，不同就拒绝拼接。
  - `abfe_core.THERMODYNAMIC_CYCLE_DOC` 已改写为
    `ΔG_complex = ΔG_attach + ΔG_decouple,restrained + ΔG_release_to_1M`。
  - 回归 `tests/test_boresch_attachment_leg.py`：λ 阶梯方向、力组冲突、全局参数注册、
    合成期符号门（纯逻辑，毫秒级）+ 6 粒子玩具体系上的 ΔG≥0 / 弱限制极限 /
    力常数单调性（`cpu_only`，几秒）。

  **实测结果（2026-07-28，采用值）**：

  ```
  ΔG_attach = 5.601 ± 0.223 kJ/mol = 1.339 ± 0.053 kcal/mol
  ΔG_bind   = −3.460 kcal/mol      （区间 −3.513 ~ −3.406；原 −2.121）
  ```

  **两轮独立测量**（不同 λ 表），误差棒取半程差而不是任一单轮的 σ：

  | λ 表 | BAR(主) | TI | MBAR(诊断) | \\|BAR−TI\\| | 报出 σ | ΔG_bind |
  |---|---|---|---|---|---|---|
  | 12 态 | 5.3784 | 5.3867 | 5.7440 | 0.008 | 0.083 | −3.406 |
  | **4 态** `[0,0.1,0.35,1]` | **5.8238** | 6.0118 | 6.1792 | 0.188 | 0.100 | −3.513 |

  **两轮差 0.4454 kJ/mol = 报出 σ 的 4.4 倍**——这是第三次撞上「单轮渐近协方差
  系统性低估」（另两次：P1-19 的 window4、溶剂盒扫描的跨 seed SEM）。
  **任何单轮的 σ 都不要引用。**

  4 态那轮更可信：⟨U_B⟩ 全程单调，而 12 态那轮 12 档里有 3 处倒挂
  （λ=0.35 报 5.411 比 λ=0.5 的 5.518 还低；4 态测同一个 λ=0.35 得 7.008，
  说明 12 态那档卡住了）。而且计算量只有 1/3。**稀疏 λ 表是对的**：
  实测每条边 Δu ≤ 0.33 kT，而 BAR 的合理区间是 1–2 kT，12 态密了 3–6 倍。

  ```
  ```

  同一批样本四个口径：TI **5.3867** / 相邻 BAR **5.3784** / 两端 BAR 5.4317 /
  去相关 MBAR **5.7440**。前三者一致，**去相关 MBAR 是离群值**——它的选帧把
  结果偏出 0.37 kJ/mol（4.4σ），而当时的容差 `max(2.0, 3σ)` 在 σ 被低估时
  只剩 2.0 那个与量级无关的常数兜底，放行了。**主值已改为相邻 BAR，
  TI 作一致性门，MBAR 降级为诊断。**

  可证伪预期通过：1.339 > 参考的 0.442 kcal/mol（3.0 倍）。

  **⛔ Hamiltonian REMD 路径作废（`BoreschAttachmentREMDManager`，保留但默认不可达）**：
  首次启用产出 `38.6006 ± 109.9858`、零 round trip。根因不是实现错，是物理——
  Boresch 的 `k(1−cosΔ)` 项在反转时取 2k：单个二面角 359–469 kJ/mol
  （144–188 kT），三个全反转 1189 kJ/mol = **477 kT**。交换一旦把副本送进翻转态，
  那一帧的天文 U_B 就支配指数平均，σ=110 是它的指纹。
  第一轮顺序窗口 3000 帧里 U_B 最大值只有 64.99 kJ/mol（26.1 kT，是最软那个
  反转的 18%）——**一次反转都没采到**，所以良态。

  **单盆地限制不是缺陷，是必要条件**：配套的解析释放项
  （`calculate_boresch_analytical_correction`，−37.649 kJ/mol）本身假定单一
  简谐盆地。配一个会采到翻转的 attachment 腿反而不自洽。**不要把这当 bug 去"修"。**

  **原可证伪预期（已通过，留档）**：ΔG_attach 应显著 > 参考的 +0.442 kcal/mol。
  依据是 `boresch_onoff_v2` 实测限制压掉了约 10 kJ/mol 的口袋静电
  （受约束 172.79±1.61 → 无约束 181.06±0.94，而参考那条腿 σ 只有 0.010，
  说明它的限制几乎没扰动系综）。**测出来接近 0.442 就说明这套推理错了。**

  **原始记录（发现时）：**

  参考值 `result.txt` 把 restraint 拆成两段，我们只有第二段：

  ```
  restraint            -0.442  0.010   ← 在完全耦合的复合物里用一条真实 λ 腿把限制打开（采样出来的）
  restraint-analytical  7.050  0.000   ← 解析释放到 1 M 标准态
  ----
  restraint             6.608  0.010   ← 两者之和
  ```

  当前实现只有解析释放。已 grep 确认**不是换了算法，是真的没有这条腿**：
  `LambdaDependentBoreschForce`（`abfe_core.py:1177`）的 `lambda_boresch_scale`
  只在 warmup/rebalance 爬坡时用（`ibs_engine.py:5467 / 5691 / 5890 / 7754 / 7806 / 7905`），
  生产采样期间一律钉在 1.0，从来不是一个被估自由能的 alchemical 维度；
  stage1/stage2 里也没有 restraint 阶段。

  **物理表述**：现在 `ΔG_complex = ΔG(A→C)`，其中 A = 「配体耦合 + 限制已打开」的复合物。
  但物理结合态是 A′ = 「配体耦合 + 无限制」。缺的正是 A′→A 这一步。

  **量级**（参考值实测 A′→A = +0.442 kcal/mol）：

  | | kJ/mol | kcal/mol |
  |---|---|---|
  | 现在 ΔG_cplx | 171.658 | 41.027 |
  | 补上 A′→A 后 | 173.51 | 41.47 |
  | ΔG_bind 现在 | −8.875 | **−2.121** |
  | ΔG_bind 补上后 | −10.72 | **−2.563** |

  数值不大，但**符号恒定偏正、每个体系都有，且随限制力常数变硬而变大**，不是可以忽略的随机误差。

  **两条同批查清的相关结论，避免后人重复排查：**

  1. **解析释放项对 P0-10 的错平衡值几乎不敏感**——`RESULT_2026-07-27_atenolol_rank11.md`
     里「解析释放修正也吃这组错值」这句需要更正。修复前后该项只从 −37.3223 变到
     −37.6493 kJ/mol（差 0.33），而这中间锚点整个换过
     （配体 `[4597,4600,4601]` → `[4594,4595,4596]`，r0 0.4739 → 0.4308 nm，
     kthetaB 200 → 71.9，裁剪的力常数 4 个 → 1 个）。
     原因是释放公式只吃 `r0²·sinθA·sinθB·√Πk`：二面角平衡值根本不进公式，
     而 θA/θB 对调让 `sinθA·sinθB` 不变。
     **所以 P0-10 的伤害路径只有采样（把配体拽离 pose），不包括解析项本身。**
     已用 `abfe_core.py:1114` 的公式独立复算：−37.64931 vs JSON 的 −37.64925，差 6e-5，
     公式与参数都无误。

  2. **锚点不同的两次运行，restraint 逐项不可直接对比**。限制势越硬，解析释放越贵，
     但打开它也越贵、被约束的解耦腿也跟着变，三项互相抵消，**只有总和是锚点无关的**。
     本轮解析项 +8.998 vs 参考 +7.050 那 +1.95 kcal 因此不能直接判为缺陷，
     它本该被缺的那条腿和解耦腿吃掉一部分。`RESULT_2026-07-27...md` §6 的逐项表
     在锚点变了之后只有「总计」那一行仍然有效。

- [x] **P1-18：已关闭（2026-07-29）——立项前提被撤销，不要按原计划排查。**

  原命题是「charging 相对 `result.txt` 残留 1.32 kcal/mol」。**它建立在拿 `result.txt`
  的分项当真值这个方法论错误上**：本仓库实现的是 IBS，与参考的分项拆法本就不同，分项
  对不上不构成偏差；而且 restraint 与被限制腿的采样结构性抵消，把 charging 的差和
  restraint 的差当成两条线索是重复计数。07-29 全量结果的 **total 与参考相差 0.98σ，
  1σ 内一致**，没有需要归因的缺口。理由详见
  [handoffs/BORESCH_DIHEDRAL_SIGN_HANDOFF.md](handoffs/BORESCH_DIHEDRAL_SIGN_HANDOFF.md) §7.2 / §9.4。

  真正剩下的物理问题是 vanishing 腿的误差估计（P1-19），它的证据**完全来自内部
  复现性**，不依赖任何与参考的比较。

  下面原文仅供追溯，**不是待办**：

  ---

  当前生产结果为复合物 `64.4113`、溶剂 `62.8865 kJ/mol`，所以 charging 对
  `ΔG_bind` 的贡献仅 `−1.5249 kJ/mol = −0.3645 kcal/mol`；`result.txt` 为
  `−1.680 kcal/mol`，残差 `+1.316 kcal/mol`。现有证据不支持先改 MBAR：
  对当前 `u_kn` 的逐 λ 梯形 TI 为复合物约 `65.22`、溶剂约 `63.54 kJ/mol`，
  分别只比 MBAR 高约 `0.81/0.65 kJ/mol`；REMD 平均交换率也有
  `0.615/0.604`。`tools/diagnostics/diagnose_charging_linear_response.py` 的两端平均会虚高
  20–30 kJ/mol，不能再拿它判 MBAR 错误；这里只认逐 λ 梯形 TI、MBAR 和相邻
  BAR 三口径。

  **前置已完成（2026-07-28），下面两项的前提要按新结果重写：**

  - **P0-11 盒长扫描已做完，没吃掉残差。** 量到 6.057 nm 时 decharging 才
    64.25（参考反解要 68.1），三档盒子对 ΔG_bind 零影响。所以 `62.8865` 这个数
    不再"来自已判无效的盒子"，它就是当前口径下的值。
  - **下面第 1 项已被 P1-17 吸收，不再作为独立问题。** 补上 attachment 腿
    ΔG(A′→A) 之后，循环对任意限制强度严格闭合——受约束系综里测出来的 charging
    **就是对的**，不需要（也**不能**）再单独修正，否则重复计数。
    受约束/无约束对照（`boresch_onoff_v2/`）也已做完，但那轮的 RMSD/质心指标
    因为没扣除蛋白整体转动而不可用（ON 臂在满强度限制下也报 3.76 Å，
    kr=2000 下配体不可能真移动这么多）——**该信的是 attachment 腿在 λ=0 测到的
    ⟨U_B⟩ ≈ 8.8–10.2 kJ/mol，不是那个 RMSD**。
  - **剩下的只有第 2 项（`llfreeze` 口径）。** 它是残差里唯一有硬证据的一条：
    两腿共同偏低 **5.21 kJ/mol**——同一分子、同一套分子内库仑，在两腿里完全相同，
    正是"去电荷 Hamiltonian 定义不同"的指纹。但**那部分在 ΔG_bind 里完全抵消**；
    真正进缺口的是复合物腿特有的 **5.48 kJ/mol = 1.31 kcal**，是否同源尚无证据。

  1. ~~复合物 charging 的真实端全程开启强 Boresch，需做受约束/无约束系综对照。~~
     **（已由 P1-17 解决，保留原文供追溯）**
     当前 `fixed_lam=1.0`，所以 fully charged 的 λ=1 端也在全强度 Boresch 下采样；
     这不会作为显式 λ 能量差直接相加，却会改变配体取向、关键氢键占据和蛋白/水响应。
     本轮 λ=1 的 `⟨U(0)−U(1)⟩` 为口袋 `171.08`、水中 `180.96 kJ/mol`，说明当前
     受约束口袋系综的静电稳定化仍弱于水约 `9.88 kJ/mol`。最小判别实验：从同一
     fully charged 坐标出发，Boresch 开/关各做至少 3 个独立短轨迹；比较
     ligand–environment 静电能、关键氢键占据、配体重原子 RMSD/质心位移。若关闭
     Boresch 后口袋静电显著增强，残差归因于受约束真实态；修法应与 P1-17 一起设计
     （补 attachment 腿或把 restraint 纳入可闭合的 λ 路径），不能单独调力常数追
     `result.txt`。

  2. **核对 `llfreeze` 与参考方法的 annihilation/decoupling 口径。**
     当前模型 `pme_decharge_v2_llfreeze_pmeself_20260523` 通过显式 exception 冻结
     所有 ligand–ligand Coulomb，只缩放 ligand–environment 静电；`result.txt`
     没有 provenance，尚不知道其 charging 是否也冻结 L–L、是否使用相同 PME
     tolerance/grid/cutoff、盒长/水/盐、Boresch 状态和 λ 方向。先找回参考作业的
     mdp/脚本/日志；找不到则分别按「L–L 冻结 decoupling」和「L–L 同步消失
     annihilation」重算同一批冻结坐标的 `u_kn`，比较两条腿与差值。验收要求是先把
     Hamiltonian 和 restraint 口径逐项对齐，再比较自由能；不能仅凭
     `charging-complex/-lig` 两行反推代码缺陷。

- [x] **P1-20：Stage 2 预优化缓存指纹的 fail-open 旁路已删除（2026-07-28 完成；本条 2026-07-29 才勾掉，此前状态记错）。**

  `ABFE_DEBUG_SKIP_STAGE2_FINGERPRINT` 已**不再是旁路**：`abfe_pipeline.py:6208` 现在检测到
  该环境变量就直接 `raise RuntimeError`，报文说明它已于 P1-20 删除、指纹不匹配必须重跑 pilot。
  回归测试在 `tests/test_charging_estimator_protocol.py:348-367`（断言「不得再有任何
  fail-open 通路」）。全仓 grep 确认无其它 `SKIP_*FINGERPRINT` 旁路。

  ✅ 保留的两处 `protocol_match = True`（`abfe_pipeline.py:6041` Stage 1 / `:6229` Stage 2）
  **不是** fail-open：它们是 schema 迁移路径，先由 `_preopt_cache_matches_ignoring_code_hash()`
  把物理输入（potential_type/Boresch/温度/压力/坐标/预优化代码本身）逐项核对一致，
  才原地重盖窄指纹。

  ⚠️ 同类风险仍有一个 opt-in 开关：`ABFE_DEBUG_FREEZE_CODE_HASH=1`
  （`abfe_pipeline.py:497`）会把 `code_sha256/preopt_code_sha256` 冻成常量。它每次运行都打印
  🚨 警告、docstring 也写明「正式出结果前必须取消设置并至少完整跑一次真哈希校验」，
  属于有意识的显式 opt-in，暂不列为缺陷；但**作业环境残留它的后果与被删的那条旁路同级**。

- [ ] **P2-17：`tools/repairs/repair_stage2_window0_real_delta_f.py` 的文档化流程已跑不通（2026-07-29 发现）。**

  该修复工具在三处（`:42`、`:72`、`:230`）指导用户用
  `ABFE_DEBUG_SKIP_STAGE2_FINGERPRINT=1 python runabfe.py ...` 续跑，其中 `:230` 是运行时
  直接打印给用户的下一步命令。但该环境变量自 P1-20 起会让 `runabfe.py` **直接 raise**，
  所以照着这个提示走必然失败。改法：删掉这三处提示，改为说明「指纹不匹配就重跑 pilot」，
  或给该工具补一条真正的离线迁移路径（逐字段核验后重写指纹）。纯文本/提示修改，不动物理逻辑。

## P2 工程 / 发布质量

- [ ] **ATT-19：核心物理单元测试覆盖仍不足。** GitHub [#59](https://github.com/Cedrus810/openmm_IBS_dev/issues/59)。已补一批数值/协议测试，但仍缺软核势端点、DEXP LJ matching、PBC/离子计数更完整覆盖、并行 worker 等最小矩阵。**新增测试一律只放 `tests/`**（2026-07-29 起全部自动化测试已归位该目录，入口 `./tests/run_offline_tests.sh`，单文件 `./tests/run_offline_tests.sh tests/<file>.py`）。

- [ ] **ATT-20：缺少公开 ABFE benchmark 端到端集成验证。** GitHub [#60](https://github.com/Cedrus810/openmm_IBS_dev/issues/60)。需要中性/带电配体、两腿循环闭合、实验对比的可复现脚本。

- [ ] **ATT-21：文档缺口。** GitHub [#61](https://github.com/Cedrus810/openmm_IBS_dev/issues/61)。2026-07-29 已完成仓库与文档整理：目录导航见根 `PROJECT_LAYOUT.md`，文档导航见 `docs/README.md`，教程拆为 `GETTING_STARTED` / `OUTPUTS_AND_RESUME` / `TROUBLESHOOTING` / `MIGRATING_TO_A_NEW_SYSTEM` / `MAINTAINING`，旧 README 状态段迁入 `status/README_STATUS_SNAPSHOT_2026-07-29.md`。仍缺 API 参考、独立热力学循环推导文档、打包元数据。

- [ ] **ATT-22：CI/CD、静态检查与格式化仍缺。** GitHub [#62](https://github.com/Cedrus810/openmm_IBS_dev/issues/62)。`tests/run_offline_tests.sh` 已修并跑出 367 passed（2026-07-29 起测试统一在 `tests/`，CI 配置应指向该入口）；仍需 GitHub Actions、ruff/flake8、mypy、black/isort，并隔离 GPU 作业。顺带清掉 `abfe_core.py`、`abfe_pipeline.py`、`abfe_preoptimizer.py` 中仍会捕获 `KeyboardInterrupt/SystemExit` 的 3 处裸 `except:`。

- [ ] **ATT-23：运行恢复与资源保护能力不足。** GitHub [#63](https://github.com/Cedrus810/openmm_IBS_dev/issues/63)。继续评估 GPU OOM 降级/Context 回收、长任务中断恢复、磁盘空间预检和运行时估计。新增明确缺口：`_is_checkpoint_valid()` 目前只检查文件 ≥512 B 且可 seek，`_is_traj_valid()` 只检查粗略大小、`CORD` 和首个记录长度；二者都不能证明 checkpoint 可加载或 DCD 帧完整。恢复流程应以真实 `loadCheckpoint`/DCD 解析为准，并避免在 checkpoint 加载成功前决定追加旧轨迹。

- [~] **ATT-24：输入验证不足；显式 config/torsion 静默降级已修，DEXP 暂缓。** GitHub [#64](https://github.com/Cedrus810/openmm_IBS_dev/issues/64)。仍需 broader ligand/TOP/Boresch/box-size 前置诊断与 DEXP 输入契约。低优先防御项：移除 `calc_boresch_from_last_frame()` 对 `(3,3)` 坐标的猜测式转置；把 `ACESoftcorePotential.optimize_alpha()` 的 `assert` 改为显式 `ValueError`（该方法当前全仓零调用）。

- [ ] **P2-16：GROMACS include 自动发现不具备 Windows 可移植性。**

  `runabfe.find_gmx_include_dir()` 调用 Unix 命令 `which gmx`，失败后又扫描两条
  特定用户的 `/home/ruigengji/...` 目录。Windows 上仍可通过显式 `--gmx-path`
  或 `GMXDATA` 正常运行，所以原清单的“严重、必失败”评级不成立；但 PATH 中已有
  `gmx.exe` 时无法自动发现，属于真实的 P2 跨平台缺陷。改用 `shutil.which("gmx")`，
  从可执行文件位置推导 share 目录，并删除个人目录回退；补 Windows/POSIX mock 测试。

- [ ] **ATT-25：协议版本矩阵缺少统一注册/迁移工具。** GitHub [#65](https://github.com/Cedrus810/openmm_IBS_dev/issues/65)。需要统一注册表、缓存指纹组合规则、迁移说明和兼容性测试。

- [ ] **ATT-26：`IBSWindowManagerDualLambda.run_all_windows` 过长且职责混杂。** GitHub [#66](https://github.com/Cedrus810/openmm_IBS_dev/issues/66)。实测约 3055 行；端到端回归稳定前暂缓大拆。清理时一并处理 3 个零调用遗留点：`scan_boresch_1d_pes()` 的二次 Å→nm 转换及不可达角度分支、`aggregate_all_energies()` 用矩阵长短猜 `(K,N)` 方向、重复导入同一个 `generate_overlapping_windows`。

- [ ] **ATT-28 / R-05：日志与通用工具实现分裂。** GitHub [#41](https://github.com/Cedrus810/openmm_IBS_dev/issues/41)。统一结构化日志入口、级别和文件/控制台策略；另评估 DEXP 多随机种子优化。`ibs_engine.py` 的模块级 `print = _log_print` 属于本项而非数值 bug；同时统一 5 个 `NumpyEncoder` 实现，保持 `np.integer → int`、`np.bool_ → bool`，避免局部版本把整数写成浮点或拒绝 numpy 布尔值。

## 当前运行验证

- [ ] **V-02：传统 `single_lambda`/REMD 小型固定盒回归。** GitHub [#32](https://github.com/Cedrus810/openmm_IBS_dev/issues/32)。确认每个 task 收到有限、长度等于态数的 v3 LRC 数组，且每帧修正为 `coeff[k]/V(t)`。

- [x] **V-03：Stage 2 v21 blended-metric lambda schedule CUDA 验证 —— 已由 2026-07-29 生产跑满足，结案。**
  GitHub [#33](https://github.com/Cedrus810/openmm_IBS_dev/issues/33)（可关）。
  验收条件见 [design/LAMBDA_SCHEDULE_CONTRACT.md](design/LAMBDA_SCHEDULE_CONTRACT.md)，逐条核对落盘证据：
  - 两条腿 `protocol_key.payload.thermodynamic_path_protocol_version = **21**` → v21 确实在 CUDA 上跑过。
  - `stage_diagnostics.stage2.window_overlap_diagnostics[*].window_range`
    = `[[0,5],[4,8],[7,12],[11,16],[15,20],[19,23]]`，与契约「窗口画线」的半开区间**逐字相同**。
  - **6 个 ensemble** ✓；程序核对相邻窗口只共享一个节点 ✓；覆盖到最终 state **22** ✓。
  - `validate_single_shared_boundary_ranges`（`abfe_preoptimizer.py:362`）已接入
    `abfe_pipeline.py`，回归在 `tests/test_audit_protocol_regressions.py`、
    `tests/test_warmup_overlap_protocol.py`。

- [ ] **V-08：核对 `stage2_n_states = 17` 与实际落地 23 个唯一 λ 的语义（2026-07-29 登记，低优先）。**
  V-03 的窗口契约完全满足，但 `provenance.config.stage2_n_states` 是 **17**，而实际落地的是
  23 个唯一 λ（6 窗口 × 槽位 − 5 次边界复用）。两个数都对得上各自的定义时没问题，
  但配置名叫 `n_states` 却不等于实际态数，容易被下一个人误读成契约违规。
  只需确认语义并在契约文档里写明二者关系，不改数值。

- [x] **V-07：离线全套已重跑通过，ESS 门断言全部确认（2026-07-29 结案）。**
  入口是 `./tests/run_offline_tests.sh` = `pytest -m "not needs_gpu"`。
  ⚠️ 口径说明：`cpu_only` / `needs_gpu` 是 `tests/pytest.ini:30-32` 注册的 marker，
  `cpu_only` 的语义是「只需 OpenMM Reference/CPU platform，可在登录节点跑」——
  **它标注硬件需求，不是测试选择门**。当前 22 个测试文件里**没有一个**打 `needs_gpu`，
  所以离线入口排除不掉任何东西，等于跑全套。
  上一轮 285 项里只有 `ess_gate_protocol_version` 失败（已按 BOR-01 改为 3），该断言短路在
  版本号那一行，后面几条（`min_absolute_ess_threshold is None`、
  `min_absolute_ess_gate_retired_reason` 存在、`raw_*` 四项非 None、`ess_gate_metric` 标签）
  当轮未执行；本次重跑已全部执行并通过，无遗留。

- [x] **V-04：重建溶剂腿显式 0.15 M NaCl 缓存 —— 已满足，结案（2026-07-29 核对）。**
  GitHub [#34](https://github.com/Cedrus810/openmm_IBS_dev/issues/34)（可关）。
  证据全在 `output_lrc_fix/solvent_cache_manifest.json`：
  `protocol_version = 4`、`ionic_strength_molar = 0.15`、**`na_count = 7` / `cl_count = 7`（非零）**、
  `positive_ion = Na+` / `negative_ion = Cl-`、`box_edge_nm = 4.257`、`padding_nm = 1.5`、
  `nonbonded_cutoff_nm = 1.0`。第三条验收（旧 checkpoint 因指纹变化被拒）由
  `SOLVENT_CACHE_PROTOCOL_VERSION` 3→4 达成，且盒边从 3.000 变成 4.257 本身就证明缓存真的重建过。

## Boresch 二面角符号事故（2026-07-29）与后续

全量诊断、证据链、时间线见
[handoffs/BORESCH_DIHEDRAL_SIGN_HANDOFF.md](handoffs/BORESCH_DIHEDRAL_SIGN_HANDOFF.md)。
下面只留仍需行动的部分。

- [x] **BOR-01：四份手写二面角返回 −φ 的根因已修并实测确认（2026-07-29）。**

  `(n1×b2̂)·n2 = −(n1×n2)·b2̂`，所以 `abfe_core.py` 那四份副本（`1962`
  `OrbBoreschEstimator.estimate_from_trajectory`、`2029` `_finalize_candidate`、
  `2837` `scan_boresch_1d_pes._calc_geom`、`3375` `calc_boresch_from_last_frame`）
  返回的都是标准约定的**镜像**。OpenMM `dihedral()` 与 `mdtraj.compute_dihedrals`
  都用 IUPAC 约定，所以只有三个二面角整体反号，`r0/θ` 不受影响（`arccos` 无符号）。

  症状是 attachment 腿 λ=0 整个系综坐在 `k(1−cosΔ)` 势壁顶上（mean=777，std 只有 67），
  BAR/TI 门失败只是表征。**不要因此加密 λ**——同一张 4 态表 07-28 跑出 BAR/TI 差
  0.12 kJ/mol。

  已改：新增模块级 `abfe_core.boresch_dihedral_rad()`（`abfe_core.py:1142`，带完整事故
  docstring），四份副本全部替换，一次修完三个注入点（`abfe_pipeline.py:3338`、
  `runabfe.py:1777`、`abfe_pipeline.py:3263` 的 resume 校验）。
  `tests/test_boresch_attachment_leg.py` 改为复用该函数，不再自带公式。
  新增 `tests/test_boresch_dihedral_convention.py`（8 条 `cpu_only`，6 粒子，不跑动力学）：
  手算基准、与 OpenMM 自己的 `dihedral()` 对比、与 mdtraj 对比、fixture 判别力自检
  （三个 φ 的 `|sin|` > 0.2）、端到端 `U_Boresch ≈ 0`、反号后必须暴涨到 `Σ2k_φ`。
  **原则：测试绝不自己写二面角公式，只拿 OpenMM/mdtraj 当基准。**

  同批顺带修正：`tests/test_core_physics_numerics.py` 的 `ess_gate_protocol_version`
  钉子 2 → 3（`ibs_engine.py:11505` 已在 07-29 05:30 bump 到 3，v3 只取消了最终 stage
  的 occupancy 反向否决，warmup 的 production-entry 占据门未变，与二面角无关）。

- [ ] **BOR-02：`update_boresch_from_last_frame` 的校验门不看二面角。**

  `abfe_pipeline.py:3328` 的两道强校验只看 θ 和 r0，所以这次的反号值畅通无阻。应复用
  现成的 `boresch_committed_deviation_sigma`（`abfe_pipeline.py:197`，同文件模块级可直接调）
  比较新推导的 `new_eq` 与它要覆盖的 `orig_eq`，阈值沿用现成常量
  `BORESCH_COMMITTED_MAX_DEVIATION_SIGMA = 4.0` / `..._WARN_... = 2.5`，二面角先过
  `_wrap_to_pi`。

  **超限行为必须是「告警 + 保留 `orig_eq`」，不是 raise**：与同一函数已有两道门风格一致；
  `orig_eq` 来自 `boresch_simple.json` 的 500 帧系综均值，本来就比单帧重锚可靠，退回它
  是严格更优；4σ 在 6 个自由度上误报率约 2.8%，硬门会以约 1/36 概率无故杀掉一次 9 小时
  生产跑。真正的守门人是 `tests/test_boresch_dihedral_convention.py`。
  测试就近扩进 `tests/test_boresch_committed_gate.py`（已 import 该函数、同套阈值常量）。

- [x] **BOR-03：`BORESCH_GEOMETRY_CONVENTION_VERSION` 2 → 3 —— 决定不做（2026-07-29 关闭）。**

  `runabfe.py:1633` 保持 **2**。理由：v2 缓存的 `simple`/`fluctuation` 文件本身数值没问题
  （实测 `output_lrc_fix/boresch_simple.json` 的 `equilibrium_values` 与
  `diagnostics.fluctuation_distribution.mean` 逐位相等），真正的污染路径是
  `auto`/`orb_simple` 分支用反号值覆盖，**那条路径已由 BOR-01 从根上修掉**。
  升版只会强制重建一批数值本来就正确的缓存，收益为零。
  handoff §9.2 里那条建议按此作废，不要照做。

- [→] **BOR-04：vanishing 腿的报出 σ 偏小 —— 已并入 P1-19，不在本节重复登记。**
  同 padding 两次独立跑差 2.34σ（比跨盒子的 1.13σ 还大）这条跨跑证据，与 P1-19 的
  split-half 是同一现象的两个视角、同一个拟议修法，故合并。**不是代码 bug，是不确定度
  量化 + 采样不足**；真 bug 在 P1-23。行动顺序（先固定 padding 重复跑、别再扫盒子）见 P1-19。

- [ ] **BOR-05：让 mdtraj 拿到带配体键的拓扑（原「Boresch 拓扑后续」）。** 配体侧几何回退是当前唯一可用判据，
  但管线本来就有真实键：`GromacsTopFile` 的 `top.topology.bonds()` 含配体键
  （`generate_ligand_xml_from_top` 就靠它建 `bond_neighbors`）。可选路径是把 OpenMM 拓扑
  连 CONECT 写成 PDB 供 mdtraj 读，或给估计器传 `bond_overrides`。这会让配体侧也用上真实键、
  彻底摆脱 0.22 nm 近似。动的是拓扑缓存格式，属独立改动。

## 低优先 / 后续稳健性

- [ ] **P0-9（延期，不阻挡生产重跑）：补齐 `--analyze-only` 的 stage 完整性与 ESS 契约。**
  GitHub [#67](https://github.com/Cedrus810/openmm_IBS_dev/issues/67)。这是离线分析入口的工程完整性事项；
  以后再复用主 pipeline loader 补齐 manifest、expected windows、checkpoint/f_k、
  stage checkpoint 协议与覆盖验证。

- [ ] **跨进程 production resume/rebuild 边界应作为独立 trajectory segments。** GitHub [#75](https://github.com/Cedrus810/openmm_IBS_dev/issues/75)。本批数据中 base jumps 被 `u_kn - bias` 抵消而无害，但不应依赖运气。

- [ ] **膜受体前置：恒压器目前是各向同性的（2026-07-28 记）。** `abfe_pipeline.py:1341/1345`
  用的是 `openmm.MonteCarloBarostat`，它把 x/y/z 按同一因子缩放。放到双层膜上会把
  面积和厚度绑死，面积每脂（APL）会跑掉。膜体系需要 `MonteCarloMembraneBarostat`
  （xy 耦合、z 独立、表面张力一般取 0）或 `MonteCarloAnisotropicBarostat`，
  且需要一个 `system_type` 分支去选。
  盒子读取侧没问题：`abfe_pipeline.py:1567` 用 `np.linalg.norm(box_nm, axis=1)`，
  各向异性/三斜盒都能正确取边长，没有假设立方。
  另注：膜受体只影响复合物腿；溶剂腿是配体在体相水里，与膜无关，
  仍走 P0-11 修好的独立水盒逻辑，不应继承复合物盒。

- [ ] **锁定 `pymbar-core` 版本。** GitHub [#76](https://github.com/Cedrus810/openmm_IBS_dev/issues/76)。当前 uncertainty semantics 依赖未锁版本，建议 pin/bound 并记录 intended method。

## 长期研究项

- [ ] **R-02：用真实 bridge 数据重新标定 Shadow-Coulomb。** GitHub [#38](https://github.com/Cedrus810/openmm_IBS_dev/issues/38)。
- [ ] **R-03：评估 point-ESS lambda insertion heuristic。** GitHub [#39](https://github.com/Cedrus810/openmm_IBS_dev/issues/39)。
- [ ] **R-04：评估 ACE softcore floor、pilot metric 样本数、PBC 重居中。** GitHub [#40](https://github.com/Cedrus810/openmm_IBS_dev/issues/40)。

## 已归档完成项

以下完成项的详细定位、表格、回归测试和诊断过程已移入 [archive/TODO-2026-07-27-full.md](archive/TODO-2026-07-27-full.md)：

- P0-1 到 P0-5、P0-8、P0-10。
- P1-9 到 P1-16。
- ATT-01 到 ATT-05、ATT-08 到 ATT-13、ATT-15 到 ATT-18。
- P2-13、P2-14、P2-15。
- V-01、V-05 的历史验证与 V-06 诊断过程。
- 2026-07-26/27 本地验证长记录。
- 2026-07-28 移出的完成项、关闭项与 #1–#27 复核记录见 [archive/TODO-2026-07-28-completed.md](archive/TODO-2026-07-28-completed.md)。
