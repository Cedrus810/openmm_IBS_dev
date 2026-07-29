# 当前行动清单

更新：2026-07-29。完整 2026-07-27 审计长记录已移入 [archive/TODO-2026-07-27-full.md](archive/TODO-2026-07-27-full.md)。历史原文见 [archive/todolist-2026-07-20.md](archive/todolist-2026-07-20.md)，审计证据见 [status/AUDIT_STATUS.md](status/AUDIT_STATUS.md)，运行验证见 [status/VALIDATION_MATRIX.md](status/VALIDATION_MATRIX.md)。本轮移出的完成/关闭项见 [../archive/todo-2026-07-29.md](../archive/todo-2026-07-29.md)。

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

- [ ] **V-08：核对 `stage2_n_states = 17` 与实际落地 23 个唯一 λ 的语义（2026-07-29 登记，低优先）。**
  V-03 的窗口契约完全满足，但 `provenance.config.stage2_n_states` 是 **17**，而实际落地的是
  23 个唯一 λ（6 窗口 × 槽位 − 5 次边界复用）。两个数都对得上各自的定义时没问题，
  但配置名叫 `n_states` 却不等于实际态数，容易被下一个人误读成契约违规。
  只需确认语义并在契约文档里写明二者关系，不改数值。

## Boresch 二面角符号事故（2026-07-29）与后续

全量诊断、证据链、时间线见
[handoffs/BORESCH_DIHEDRAL_SIGN_HANDOFF.md](handoffs/BORESCH_DIHEDRAL_SIGN_HANDOFF.md)。
下面只留仍需行动的部分。

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


