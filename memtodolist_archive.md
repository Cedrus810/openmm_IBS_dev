# 膜受体–配体 ABFE 专项行动清单（历史归档）

> 归档日期：2026-08-06  
> 本文件保存重整前的完整历史、已完成事项、设计讨论和验证证据。  
> 当前未完成工作请看 `memtodolist.md`。

## 2026-08-06 完成项追加归档

### C1：带电小水盒 charge-transfer 验证 —— 已关闭

最终裁决：

- Na⁺/Cl⁻ 硬性验收通过。
- 验收口径采用单 seed pilot；按用户决定暂不追加独立 seed。
- 正、负两种电荷符号的 charge-transfer 均完成真实 PME/CUDA 验证。
- small/large 盒长敏感性满足 C1 硬门。
- 全部 λ 状态总电荷守恒，未叠加 APBS/Rocklin。
- Ca²⁺ 结果作为多 co-ion 分摊路线的已知统计限制记录在案。
- Ca²⁺ 不阻塞 C1，不追加采样或分析投入。

结论：C1 从当前行动清单移除，后续主线进入 C2。


## 2026-08-13 完成项原文迁移归档

> 以下内容从 `memtodolist.md` 原文迁移；正文未改写。当前未完成工作仍留在当前清单。

<!-- 迁移块：顶部 C3 完成总结 -->

**C3 本轮做完的事**：

- C3-1（CPU 契约，`tests/test_charge_transfer_real_endpoints.py` 27 项 +
  `tests/test_bake_global_parameter.py` 12 项）、C3-2（单帧 wiring
  smoke）全部完成并通过。
- **Stage2 charge-transfer handoff 已经从"提案"变成"生产代码"**：新增
  `abfe_core.bake_global_parameter_into_fixed_nonbonded_force`（把某个
  GlobalParameter 在给定端点上的取值固化成静态参数、彻底删除该参数），
  并接入 `abfe_pipeline.py` 的 `ABFEPipeline._run_dual_lambda_stage`
  vanishing 分支（新增 `_charge_transfer_vanishing_handoff_active()`
  判据 + `CHARGE_TRANSFER_VANISHING_HANDOFF_PROTOCOL_VERSION` 缓存指纹，
  中性配体路径逐位不变、只对带电 charge-transfer 配体生效，当前 Atenolol
  生产配置从未触发这个分支）。全套离线回归 1196 passed / 0 failed
  （1 处既有测试的期望计数需要同步更新，已更新——见
  `tests/test_coalchemical_ion_identity.py`）。
- **C3-3/C3-4 v1（单一 A-vs-C 门）在真实 C1/C2 数据、用户本机 GPU、生产
  实际用的 CUDA `Precision=mixed` 上跑过一轮完整 20+80 帧 A/B 矩阵**：
  能量门全部干净通过（~1e-7）；力门（1e-3 kJ/mol/nm）大量失败，集中在
  co-ion/配体侧 restraint anchor 原子、呈位置相关模式，**v1 判定未通过**，
  证据见 `validation/c3_real_endpoints_v1/run_matrix_mixed_precision/`
  （已保留，未覆盖）。
- **归因诊断（一次性、限定范围）**证明：production 与独立参照在 Reference
  双精度平台上 B≡C 逐位相同（构造没有问题）；差异只在 CUDA mixed precision
  下出现，取决于两侧添加等价 exception 的顺序，不是 Hamiltonian 构造错误。
  见下方"C3 co-ion/ParameterOffset 归因诊断"小节。
- **据此用户提出并已实现 protocol v2 双层门**（Gate1 Reference 恒等性、
  Gate2 CUDA mixed 活参数 vs 固化生产、Gate3 CUDA mixed production vs 参照
  ——能量硬门、力降级为诊断）。**用现有 100 帧重新后处理（未重跑 MD）**：
  C1 20 帧 + C2 四格各 20 帧，`n_failed=0`，**100/100 全部通过**，结果见
  `validation/c3_real_endpoints_v2/`。见下方"C3 protocol v2"小节的完整表格
  和理由。
- **C/D 用同一套 v2 判据、复用真实 charging λ=0 帧（不重跑 MD）跑了一遍**
  （新增 `run_protocol_v2_matrix_cd`）：D 干净通过；**C（seam）第一轮在
  C2 上大量失败**（力差最坏到 0.64 kJ/mol/nm，能量始终干净）——根因是 C2
  raw System 自己的 `NonbondedForce` 带一个 `[0.995,1.0]nm` 窄 LJ switch
  （2026-08-07 为修膜压缩问题特意加的、当时代码注释已经写明"不改全局
  MEM-00h 无 switch 约定，需要独立决策"），跟 vanishing 阶段全局无 switch
  的软核 CV 不一致。
- **用户当场指出我最初的三个候选方案和第一版消融实验都不对**，指定了正确
  修法——**MEM-00h 双边归一化**：C3 求值前先把 charging/baked/vanishing/
  reference 共同的 raw System clone 统一转到 `cutoff=1.0nm,
  switching=False`，不改 C2 已有的 raw 文件/轨迹本身。新增
  `mem00h_normalized_raw_system()`/`assert_mem00h_switching_convention()`，
  接入两个 v2 runner，7 个新 CPU 契约测试（含独立复现根因+验证修复的可控
  小体系测试）+ 全量离线回归 1213 passed/0 failed。**修复后重跑全部真实
  GPU 数据：A/B 100/100 + C/D 50/50，全部 150 帧一次通过，C2 的 C-seam
  力差从最坏 0.64 kJ/mol/nm 回落到跟 C1 同一量级的机器精度（1e-13~1e-12）**。

- **用户 2026-08-11 确认 C3 实质验收完成**，要求补齐两份 fail-closed 汇总
  产物才能正式关闭：`summary.json`（A/B `100/100`、C/D `50/50`、
  `n_failed=0`、四条端点全过、`status=complete`、`passed=true`）、
  `mem00h_report.json`（evaluation cutoff `1.0nm`/switching `false`、C2
  双边归一化已执行且**如实记录 C2 raw 本身确实带局部 switch**（不能误写成
  无 switch）、production/reference convention 一致、LRC/VDW protocol
  version 一致、C seam/D strict-zero 通过、`status=complete`、
  `passed=true`）。新增 `tools/validation/generate_c3_summary_reports.py`
  （纯汇总/核验脚本，不做任何新的物理计算，只读已有的 10 份真实 case 结果
  JSON + 对 raw System 做只读结构核验）生成两份文件，均通过；2 个新 CPU
  契约测试（含"篡改一份输入结果，汇总必须如实变成 False"的 fail-closed
  验证）。**C3 与 MEM-00h 已正式关闭，进入 C4。**详见下方 C3 小节。历史
  清单、已完成项目和详细论证：`memtodolist_archive.md`

<!-- 迁移块：MEM-00h 已关闭章节 -->

## 2. 立即处理

### MEM-00h：统一 ligand–environment 与 environment–environment LJ cutoff

状态：**已关闭（2026-08-11）。** 最后一条关闭条件（C3 真实端点 λ=1 能量/力
与基础力场一致）已用 `mem00h_normalized_raw_system()` 双边归一化 + 真实
GPU 数据验证通过，见下方最后一条勾选项和 §C3。

当前实现决定：

- 基础 `NonbondedForce` cutoff：1.0 nm。
- ACE/IBS softcore cutoff：1.0 nm。
- 传统 Beutler softcore cutoff：1.0 nm。
- softcore switching：关闭。
- LJ tail/LRC 积分边界：1.0 nm 到无穷远。
- DEXP 保持独立的 0.70 nm cutoff / 0.20 nm switch width，不受本项影响。
- co-ion ↔ ligand 的 1.2 nm 运行时距离门保留；它是独立的保守几何安全阈值，不是非键 cutoff。

关闭条件：

- [x] softcore、Beutler、LRC 三条路径的协议测试通过（2026-08-09 复核，全套
  离线测试 0 failed，含相关协议测试）。
- [x] Stage 2 cache/resume 能识别新的 vdW 非键协议版本并拒绝旧缓存
  （`tests/test_stage_diagnostics_persistence.py::test_vdw_protocol_version_is_written_only_to_stage2_cache_payload`
  + `tests/test_resume_reuse_contracts.py` 的 `lrc_version_match` 系列 gate，
  2026-08-09 复核通过）。
- [x] Stage 1 charging、Boresch attachment、预平衡和 C1 charging 缓存不被误
  作废（`test_resume_keeps_neutral_legacy_cache_compatible`、
  `test_resume_accepts_lse_tolerance_within_roundoff`、
  `test_resume_accepts_cache_produced_under_higher_step_budget` 等
  "接受"系列 gate，2026-08-09 复核通过）。
- [x] 全套离线测试无新增失败（2026-08-09：`./tests/run_offline_tests.sh`，
  1161 passed / 3 skipped（均为显式 opt-in 环境变量跳过）/ 0 failed）。
- [x] 在真实端点测试 C3 中证明 λ=1 能量和力与基础力场一致（2026-08-11）：
  A/B 100/100 + C/D 50/50 全部真实帧通过；C2 的 C-seam 力差经 MEM-00h
  双边归一化后回落到机器精度（1e-13~1e-12）；
  `validation/c3_real_endpoints_v2/summary.json`/`mem00h_report.json`
  两份 fail-closed 汇总产物均 `status=complete, passed=true`。

缓存结论：MEM-00h 会作废旧 Stage 2/vdW 采样与对应 `u_kn`；不应作废 Stage 1 charging、
Boresch attachment、预平衡或 C1 charging 轨迹。

---


<!-- 迁移块：B5 已关闭章节 -->

## 3. Phase B 剩余验收

### B5：cache / resume / provenance

**2026-08-09 复核**（`./tests/run_offline_tests.sh -q`，`openmm_dev`；本仓库
无 git 历史可查，不是本次改动导致——用户确认原本就没有 git）：

- [x] 跑完整离线测试并达到 0 failed——**1161 passed, 3 skipped, 1 deselected
  (needs_gpu), 0 failed**（3 个 skip 都是显式 opt-in 环境变量
  `EXP012_MACE_MODEL_PATH`/`EXP012_RUN_REAL_MACE_LATENT_TEST`/
  `ORB_RUN_REAL_TESTS`，不是静默失败）。
- [x] `test_exp012_schema.py` 的 11 个既有失败：**已修好**（用户确认，
  只是当时忘了同步改本清单）。单独跑 4 遍（含默认序）13/13 全过，文件内
  没有任何 `skip`/`xfail` 装饰器，是真正的通过，不是隐藏跳过。
- [x] 确认 complex/solvent 两腿不能串用 co-ion spec 或 manifest
  （`tests/test_coion_cache_resume_provenance.py::test_missing_charge_transfer_spec_fails_closed_with_leg_and_path`、
  `::test_charge_transfer_final_gate_requires_both_legs_and_matching_protocol`、
  `::test_charge_transfer_solvent_cache_requires_ordinary_salt_counts`）。
- [x] 确认 co-ion index、charge、LJ、restraint、fingerprint 任一变化都会拒绝
  旧缓存（`test_resume_reuse_contracts.py::test_resume_rejects_changed_coion_identity`
  三个 mutation 参数化用例、
  `test_top_level_sampling_cache_identity.py::test_outer_sampling_cache_rejects_changed_coion_identity`、
  `::test_geodesic_path_cache_reoptimizes_on_changed_coion_identity`）。
- [x] 确认 resume 后仍使用同一 co-ion、同一 restraint 和同一 λ 方向
  （`test_resume_reuse_contracts.py::test_resume_coion_identity_gate_accepts_exact_match`、
  `lambdas_match` 系列 gate）。

**B5 五条今天全部确认通过——正式关闭。**

---


<!-- 迁移块：C2/C3 已关闭章节 -->

C1 已完成并移入 `memtodolist_archive.md`。当前从 C2 开始。

### C2：无蛋白 lipid–water slab

脚本：`tools/validation/validate_charge_transfer_lipid_slab.py`（`PROTOCOL_VERSION = 8`，
每次版本号升级都是实测数据逼出来的 Hamiltonian/统计方法修复，不是预先设计；
旧版本产物必须作废重建）。命令序列/执行纪律：
`tools/validation/README_C2_LIPID_SLAB.md`。CPU 契约测试：
`tests/test_c2_lipid_slab_validation.py`（111 项全过）。拓扑：
`charmm-gui-8600905442/gromacs/`（**不是** `openmm/` 目录下的 `.parm7`/`.rst7`，
全仓库没有代码路径读取那份）。

**2026-08-07 发现并修复的三类问题**（详见脚本模块 docstring 与 README，不重复贴代码）：

1. v1→v2：探针配体+reserved co-ion dummy 只删 2 个水，λ=1 端总电荷不为 0
   （需要额外插一个普通反离子配平成 0）；restraint 与 charging Hamiltonian
   被 `configure_pme_ligand_charge_offsets`（内部会自己注入 restraint）和
   外层手动调用重复配置了两次，`ukn` 犯了同一类错误（读已配置过的 System
   又交给 `compute_u_kn` 二次配置）。
2. v2→v3：`MonteCarloMembraneBarostat`（XYIsotropic+ZFree，几何上已是
   semi-isotropic）配合 hard 1.0 nm LJ 截断在 Lipid21 slab 上产生人工面内
   压缩（APL 10 ns 内从 0.683 压到 0.590 nm² 且尾段仍在降）。根因是解析
   色散尾项只是总体积的函数，分不清 MC 试探移动缩放的是各向异性膜结构的
   XY 还是 Z。修法：两处 `NonbondedForce` 都加窄 potential-switch
   （0.995→1.000 nm，outer cutoff 仍 1.0 nm）——**范围只在本脚本自己的
   System 构建里，没有改 `abfe_core.py`/`ibs_engine.py` 的 MEM-00h 全局常量**，
   不牵动已跑通的复合物/溶剂腿。
3. 判"漂移是否显著"原来用 OLS 线性回归的斜率标准误，对 APL 这类几 ns 尺度上
   强自相关（振荡而非白噪声）的量不成立——同一条轨迹换窗口给出过互相矛盾的
   "显著"结论。改成分块法（Flyvbjerg–Petersen：块间方差而不是块内回归残差）。

**2026-08-07 系统性代码审查又修了 7 处**（`code-review` 高强度过一遍全文件，
不影响上面已经跑出来的 thin/thick 数字——都是防护/一致性修复，不改
Hamiltonian）：co-ion 水配位判据认不出离子元素时原来 fail open（悄悄判通过），
改成 fail closed；`equilibration_monitor.csv` 的温度换算原来用 `3N` 自由度，
没扣掉 `rigidWater`/`HBonds` 的约束自由度，报出来的温度系统性偏低（只影响
显示，不影响真实恒温）；`_apl_drift_recommendation` 把"没传
`--literature-apl-nm2`"（`None`）和"方向判错了"（`False`）当同一个分支处理，
拆开；`--literature-apl-nm2 0` 曾被真值判断悄悄吞成 `None`；
`slab-quality-gate` 里 co-ion 换侧检查用逐帧膜中面、但贴近阈值检查用整条
轨迹平均膜中面，两个检查基准不一致，统一成逐帧；`P31` 硬编码字符串改成复用
`abfe_core.LIPID_HEAD_REFERENCE_ATOM_NAMES`；`dynamics` 里逐帧水配位数改成
向量化（原来纯 Python 循环 ~3600 个水氧原子，垒在 GPU 采样循环里有实打实的
墙钟开销）。

**2026-08-09 thick-vs-thin 矛盾查明并修复**（当时 C2 唯一的阻塞项，现已关闭；
`PROTOCOL_VERSION 3→6`，详见脚本 docstring + `README_C2_LIPID_SLAB.md`
changelog，不重复贴代码）——四条候选解释里，命中的是"`extend-water` 自建
水层的起始密度引入了系统偏差"，其余三条（弛豫不够久/switch 对盒高敏感/纯
振荡）均排除：

- v3→v4：`base-quality-gate` 的 `density_profile_along_normal` 少除了一个
  bin 体积（只是诊断读数问题，不影响 APL/膜厚等已经在用的判据）；
  **`extend-water` 声称按 33.33 nm⁻³ 铺水，实测只铺出六成（1024 个水，
  真实密度约 20.9 nm⁻³）**——这就是 thick base 系统性偏低的根因；
  `equilibrate-base` 新增 `--n-steps-nvt` 分阶段松弛（新水层先固定盒扩散，
  再切 NPT，避免 barostat 一上来就冲击欠密度水层）。
- v4→v5：`--n-steps-nvt=0`（默认值）分支加了 barostat 却漏了
  `Context.reinitialize`，导致该分支整段"NPT"实际是 NVT——**任何 v4 产物
  必须作废重建**，本轮受影响的续跑段已用 v5 重跑。
- v5→v6：`build` 阶段 co-ion 候选点筛选门槛（1.6 nm）比下游
  `validate_co_alchemical_ion_placement` 的 runtime 判据（默认 restraint
  参数下约 2.02 nm）松，首次真实 `build` 时炸出来；改成按实际 restraint
  参数反解门槛，未动 §13.1 本身的常量。
- v6→v7（**严重**，第一批真实 4 格 GPU pilot 跑完、`slab-quality-gate`
  首次在真实探针 case 上执行才触发）：`_find_bulk_water_candidates` 算
  候选水离膜中面距离时直接 `abs(z - midplane_z_nm)`，没做 z 轴周期折叠——
  `.gro` 里长时间模拟扩散穿过周期边界的原子坐标不会自动折回
  `[0, box_z)`（实测约 24% 的原子受影响）。`Na_thin_pos0` 选中的三个点
  报的离中面距离都是 5.42-5.69 nm（几何上不可能，超过 `box_z/2≈4.17nm`），
  折叠后真实距离只有 2.65-2.92 nm——低于 3.0 nm 安全下限，本该被过滤掉；
  贪心算法反而优先选中了这些"看起来最深、实际离膜很近"的点。**实测后果**：
  4 个真实 GPU pilot 里，`slab-quality-gate` 在每一个 λ 窗口都测到探针
  40-140 ps 内逼近磷原子到 0.64-1.3 nm、水配位跌到 0。修法：新增
  `_minimum_image_z_delta_nm` 做 z 轴单轴周期折叠。**四格 `build`/
  `static-check`/GPU pilot 全部作废重来**——thin/thick base 本身不受影响。

修复后重新按 extend-water（1536 个水，密度约 31.3 nm⁻³）+ NVT 预弛豫独立
重建 thick base（10 ns：2 ns pilot + 8 ns 续跑），验证结果：

- [x] thin base（`base_thin_v3_extend1`，14 ns）：分块法 `not_significant`，
  APL 尾均值 **0.6209 nm²**（偏差 2.9%），膜厚漂移/叶片计数（40/40）/NaN 全过。
- [x] thick base（重建后，10 ns）：分块法 `not_significant`，APL 尾均值
  **0.6253 nm²**（偏差 **2.17%**，不再是 8.4%），膜厚漂移/叶片计数（40/40）/
  NaN 全过。**thick-vs-thin 矛盾已关闭**：不是弛豫不够久，是初始水层密度
  bug；同一 Hamiltonian 下重建后的 thick 与 thin 一样稳定在接近文献值。
- [x] 至少两个合法 bulk-water co-ion 初始位置：`--position-variant 0/1`
  四格（`Na_thin_pos0/1`、`Na_thick_pos0/1`）已用 v7 重新 `build`+
  `static-check`，全过，最近 P31 距离 1.76–3.05 nm（明显好于 v6 的
  错误站点）。
- [ ] wiring smoke——**v6 站点上跑过的那次已作废，需要在 v7 重建的站点上
  重新跑**（GPU）。
- [ ] 每格先跑 1 seed pilot，再扩展到 3 seeds——**v6 站点上的第一批 4 个
  完整 pilot 已作废**（`slab-quality-gate` 探针逼近磷原子/水配位归零，
  见上方 v6→v7），需要在 v7 重建的站点上重新提交 GPU 跑，是 C2 剩下唯一
  没做的事。
- [x] 保存 APL、膜厚、密度剖面 —— `base-quality-gate`/`slab-quality-gate` 已实现
  并落盘（含逐帧 timeseries.csv）；`density_profile_along_normal` 现在是
  v4 修复后的真实 nm⁻³ 数密度。
- [ ] 盒高和初始位置敏感性通过 2σ / 1 kcal/mol 门（`compare` 子命令已实现，
  需要第 5 步的 4 个 pilot 完成后才有数据可比）。

#### v8 已冻结与 v9 当前状态（2026-08-10）

- [x] v8 coordination gate：只对 `charge_fraction >= 0.9` 的 λ 硬判；平均水配位
  ≥5 且 ≥5 帧比例 ≥95%；其它 λ 只记录诊断。
- [x] v8 bulk-water restraint：`kZ=50`、`rZ=0.5 nm`、PBC-aware pair-center、
  动态 P31 膜中面、wall-hit/能量/pair-center 诊断已实现。
- [x] v8 CPU 回归：相关 111 项全过；bulk restraint periodic-distance CPU smoke 通过。
- [x] v8 `Na_thick_pos0`/`Na_thin_pos1` build + static-check 通过；输出在
  `validation/c2_lipid_slab_v8/`，v7 输出和 generated 输入 hash 未变。
- [x] v8 `Na_thick_pos0` 重新质量门：PBC 符号跳变改判为边界 crossing，不算失败；其它门全过。
- [ ] v8 `Na_thin_pos1` 仍失败：P31=0.967 nm、λ=0.1 配位 88%；结果冻结，不降配位门。
- [x] v9 PBC gate：连续分数坐标/膜核心穿越判定；v8 轨迹可 legacy 重评。
- [x] v9 bulk target：`z_midplane + signed_target_fraction*Lz`，pair-center PBC-aware。
- [x] v9 thin `Na_thin_pos1` build/static-check：target offset 0.20 nm、rZ=0.30 nm、kZ=50；输出在
  `validation/c2_lipid_slab_v9/Na_thin_pos1/`。
- [ ] v9 CUDA 只跑 λ=`0.2,0.1,0.0`；GPU 命令交给用户计算节点执行。
- [ ] v9 逐帧核对 P31、配位、pair-center displacement、bulk energy；配位门保持 mean≥5 且 fraction≥0.95。
- [x] v9 失败帧定位：λ=0 frame 43，P31 atom 2872 (`PA21/P31`)，ligand=0.884 nm；低配位不同步。
- [x] v10 ligand reference safety wall：独立 force group 8、rZ=0.20 nm、kZ=50；静态 ligand 包络通过。
- [x] v10 λ=0 分块 occupancy 与最小样本门实现；λ=0 最少 200 帧。
- [x] v10 CUDA 已完成 `Na_thin_pos1` 的 λ=`0.2,0.1,0.0`，λ=0 采样 100000 步；GPU 由用户计算节点执行。
- [x] v10 λ=0 第 6 块离线审计：coordination=4 最长 5 帧、无 coordination≤3，且未同步接近 P31/膜。
- [x] hydration gate v2（历史）：mean≥5、20 帧 block-bootstrap 下限≥5、与 C1 Na 参考差值不显著为负；`fraction≥5` 降为诊断项；严重脱水统一要求 `coordination≤3 AND r5≥0.4 nm` 连续至少 2 帧。
- [x] hydration gate v3（最终评估规则）：保留 mean/bootstrap/severe/几何硬门；C1 comparison 改为
  non-inferiority，差值 bootstrap CI 下界须 `≥−0.5` 个水分子；统一重评全部 12 个结果。
- [x] v10 现有轨迹用 hydration gate v2 重评：`validation/c2_lipid_slab_v10/Na_thin_pos1/slab_quality_gate_hydration_v2.json` PASS；旧 v10 gate 失败证据保留。
- [x] v10 `Na_thick_pos0` build/static-check：force group 7/8、全盒电荷守恒和 ligand envelope 通过。
- [x] v10 `Na_thick_pos0` pilot + λ=0.1 补充段：joint severe gate、稳定性和 C1 对照通过。
- [x] v10 两个代表 case hydration gate v2 final PASS：thin/thick 均通过，未修改 Hamiltonian/restraint。
- [ ] v10 两个代表 case 完整 11 λ CUDA；GPU 命令交给用户。

#### v10 full-11 冻结与 v11 当前执行段（2026-08-10）

- [x] v10 full-11 `Na_thick_pos0`：slab-quality-gate PASS；CPU `u_kn` 收敛，
  `ΔG_charging=1.1065 kJ/mol=0.2645 kcal/mol`；report PASS。该结果作为
  **v10 thick 成功证据**冻结。
- [x] v10 full-11 `Na_thin_pos1`：slab-quality-gate FAIL，且不是 gate 假阳性：
  λ=0 frame 145 co-ion `|Δz|=2.861 nm<3.0 nm`、最近 P31=`0.794 nm<1.0 nm`。
  其 `charging_delta_G.json` 虽然数值收敛（`0.8714 kcal/mol`），但因几何硬门失败
  **作废，不得用于 ΔG 或 compare**。
- [x] v10 根因收敛：pair-center restraint 只约束整体平移，v10 ligand safety wall
  只保护 ligand；fully charged co-ion 可在 pair 内部相对移动，受到膜头基静电吸引。
  hydration/statistics、PBC gate 和现有 v10 pair/ligand restraint 不再作为当前阻塞点。
- [ ] v11：新增独立 λ-independent、PBC-aware co-ion member safety wall（force group 9），
  动态 target 仍来自 P31 膜中面与当前 Lz，独立记录 co-ion safety energy、wall-hit
  和 fingerprint；设计安全区为膜中面距离 ≥3.0 nm，并保留约 0.2 nm 几何裕量，
  P31 正式 gate 仍为 1.0 nm。
- [ ] v11 CPU：thin_pos1 新 build/static-check 通过后，只跑 λ=`0.2,0.1,0.0`；
  通过后跑 thin_pos1 full-11。
- [x] v11 thin_pos1 pilot：co-ion member containment、几何、P31、PBC、restraint
  和 λ=0 hydration/C1 reference 全部通过；唯一未过是 λ=0.1 前 60 帧水合壳重组
  尚未平衡（block mean=`4.4,4.2,4.7,5.7,5.7`），不是持续 severe dehydration。
- [ ] v11 thin λ=0.1 confirmation：从 pilot `traj_state01_lam0.10.dcd` 最后一帧
  续接，额外平衡 100–200 ps，再采至少 200 帧；合并后检查 supplement 前后半
  block mean 差≤0.5、bootstrap/C1 reference gate 通过。
- [x] v11 thin λ=0.1 confirmation：bootstrap lower=`5.113 > 5`，补充段前后均值
  `5.61/5.59`，severe joint event 无连续两帧；几何/P31/PBC/restraint 和 λ=0/C1
  reference 全部通过。v11 thin 代表性 3-λ pilot 正式 PASS。
- [ ] v11 thin_pos1 full-11：λ=0.1 每态预平衡固定使用 100000 steps（200 ps），
  不能退回原 20000 steps（40 ps）。
- [x] v11 thin_pos1 full-11：11 λ、几何/P31/PBC/restraint、电荷/能量/力和 hydration
  gate 全部 PASS；MBAR 收敛，最小 overlap=`0.087`，ΔG=`-0.154 ± 0.548 kcal/mol`。
  λ=1 reference comparison 仅为中性 dummy 诊断态失败，不属于 hydration 硬门。
- [x] v11 thick_pos0 build/static-check：force group 7/8/9、co-ion safety 静态几何、
  电荷守恒和单点能量/力均通过；输出在 `validation/c2_lipid_slab_v11/Na_thick_pos0/`。
- [x] v11 候选点选择收紧：若 farthest-first 三点组合不满足 pair/member 最坏
  P31 包络，继续在同侧候选中搜索通过 v11 静态包络的组合；未放宽 gate 或修改
  Hamiltonian。该修复后 thin_pos0、thick_pos1 build/static-check 均通过。
- [ ] v11 thick_pos0 full-11：必须使用与 thin 完全相同的
  `n_steps_equil=100000`（200 ps）full-11 协议；不得用 v10 thick 代替。
- [x] v11 两个代表 case full-11 正式通过：thin_pos1 `-0.1537±0.5481 kcal/mol`、
  thick_pos0 `-0.7580±0.5397 kcal/mol`；绝对差=`0.6043 kcal/mol`，合并
  `1σ≈0.769 kcal/mol`，同时满足 `<1 kcal/mol` 和 `<2σ≈1.54 kcal/mol`；
  MBAR overlap 分别约 `0.087/0.104`，所有 gate 通过。
- [x] C2 四格完整验收：新增 `Na_thin_pos0`、`Na_thick_pos1`，两者均使用同一
  v11 build/full-11 协议，分别完成 gate、u_kn、report/summary。
- [x] 四格 compare：检查 thin 内 pos0/pos1、thick 内 pos0/pos1、pos0 内
  thin/thick、pos1 内 thin/thick；每项同时满足 2σ 和 1 kcal/mol 门。
- [x] 四格 compare 全部通过后，扩展到 3 seeds；C2 已完成最终验收。

四格 v11 full-11 结果（2026-08-10）：

- [x] `Na_thin_pos0`: `-0.7071 ± 0.6439 kcal/mol`，overlap=`0.0903`，report PASS。
- [x] `Na_thin_pos1`: `-0.1537 ± 0.5481 kcal/mol`，overlap=`0.0871`，report PASS。
- [x] `Na_thick_pos0`: `-0.7580 ± 0.5397 kcal/mol`，overlap=`0.1036`，report PASS。
- [x] `Na_thick_pos1`: `-0.1806 ± 0.5687 kcal/mol`，overlap=`0.0776`，report PASS。
- [x] 四个方向 compare 全部通过：thin 内差=`0.5534`、thick 内差=`0.5775`、
  pos0 跨厚度差=`0.0510`、pos1 跨厚度差=`0.0269 kcal/mol`；全部同时满足
  `<1 kcal/mol` 与 `<2σ`。C2 四格单 seed 验收完成；3 seeds 仍未开始。
- [x] C2 seed 扩展口径已冻结：当前 seed=`2026` 保留；每格只新增独立
  seed=`2027,2028`，即 `4 cases × 2 additional seeds = 8` 次新 full-11，
  最终 `4 × 3 = 12` 个 case-seed 结果，不是再增加 3 个 seed。
- [x] 8 个新 case-seed 全部固定 v11 Hamiltonian、λ schedule、`n_steps_equil=100000`
  （200 ps）、采样长度和 gate；每个 seed 独立完成 slab gate、MBAR、report/summary，
  输出目录不得覆盖单 seed 产物。
- [x] 最终按 case 的 3 个 seed 统计均值和跨-seed 不确定度；四个 contrast 重新用
  跨-seed统计检查 `<1 kcal/mol` 与 `<2σ`。12 个 case-seed 全部通过后 C2 正式关闭，
  再进入 C3 真实端点能量/力验证。
- [x] v11 CPU/GPU：thin full-11 通过后，用**同一 v11 build/protocol**重建并重跑
  thick_pos0 full-11；不得把 v10 thick 与 v11 thin 混为同一 C2 验收集。
- [x] v11 两个代表 case 同版本通过后，才进入另外两格；随后四格三 seed 验收完成，进入 C3。

### C2 v11 三 seed 最终结果（2026-08-10）

- [x] 12/12 case-seed 的 MBAR/u_kn 收敛；12/12 通过最终 hydration/几何/restraint gate、report/summary。
- [x] C1 reference equality gate 的原始边缘 FAIL（`Na_thin_pos1_seed2027`）未删除；保留原始
  `slab_quality_gate.json`、`report.json`、`summary.json`。新规则写入
  `slab_quality_gate_hydration_v3.json`、`report_hydration_v3.json`、`summary_hydration_v3.json`。
- [x] 该 seed 的 C1 差值 CI=`(-0.415,-0.195,-0.010)`，按预先声明的 `−0.5` 水分子
  non-inferiority margin 通过；absolute hydration、bootstrap、severe-dehydration 和几何门均通过。
- [x] 三 seed case 均值 ± seed 间 SD（kcal/mol）：thin_pos0=`−0.158±0.579`、
  thin_pos1=`−0.164±0.780`、thick_pos0=`−0.471±0.396`、thick_pos1=`+0.094±0.240`。
- [x] 四个跨-seed contrast 均 `<1 kcal/mol` 且按 combined seed SD 均 `<2σ`；审计文件为
  `validation/c2_lipid_slab_v11_hydration_noninferiority_v3_summary.json`。

### C3：真实体系 λ=1/λ=0 端点能量和力

**状态：已正式关闭（2026-08-11，用户确认）。** A/B/C/D 四条端点恒等式在
真实 C1/C2 数据（150 帧）上全部通过；`validation/c3_real_endpoints_v2/`
下 `summary.json`/`mem00h_report.json` 两份 fail-closed 汇总产物均
`status=complete, passed=true`（由 `tools/validation/
generate_c3_summary_reports.py` 从已跑完的 10 份真实 case 结果 JSON 纯汇总/
核验生成，不重跑任何 MD/GPU）。同时关闭 MEM-00h（见上方 §MEM-00h）。
下一步进入 C4。

- [x] **C3-0：协议冻结（2026-08-10，C3 protocol version=1）。** 本阶段只冻结
  验收契约，不代表真实体系端点已经运行或通过。

  - **比较对象与端点定义：**
    - A：production `charging λ_coul=1` vs raw physical System；
    - B：production `charging λ_coul=0` vs 独立直接改粒子电荷的 charging reference；
    - C：production fixed physical Hamiltonian `U_common + CV(λ_vdw=1)` vs charging
      `λ_coul=0`；
    - D：production fixed physical Hamiltonian `U_common + CV(λ_vdw=0)` vs 独立
      decoupled reference。固定物理 Hamiltonian 不含 IBS log-sum bias，也不含
      Group 4 WCA sampling shell。
  - **reference 独立性：** reference 侧只读取 raw `system.xml`、`topology.cif`、
    `ligand_indices.json`、冻结的 `coalchemical_ion_spec.json` 和当前帧的
    positions/完整 box vectors。禁止调用或间接调用：
    `abfe_core.co_alchemical_charge_offset_plan`、
    `abfe_core.create_ligand_internal_force`、
    `ibs_engine.configure_charge_transfer_decharging`、
    `ibs_engine.configure_pme_ligand_charge_offsets`、
    `ibs_engine.configure_coalchemical_neutral_decharging`、
    `ibs_engine.select_co_alchemical_ion_once`/其它 co-ion selector，以及
    `ibs_engine.build_ibs_dual_system`。production 侧正常调用这些被测 builder。
  - **MEM-00h production baseline：** 基础与 softcore Nonbonded cutoff 均为
    `1.0 nm`；switching=`false`；LJ tail/LRC 积分边界为 `1.0 nm → ∞`；
    `TRADITIONAL_LJ_LRC_PROTOCOL_VERSION=3`、
    `VDW_NONBONDED_PROTOCOL_VERSION=1`。C3 评价时 production/reference 两侧
    同时使用该协议；`λ_vdw=1` 必须计入对应 LRC，`λ_vdw=0` 的 LRC 系数必须
    精确为零。co-ion 的 `1.2 nm` 运行时几何门不视为非键 cutoff。
  - **抽帧矩阵：** C1 `validation/c1_waterbox/Na_large` 共 20 帧，
    `λ=1` 与 `λ=0` 各 10 帧，固定索引
    `[10,20,30,40,50,60,70,80,90,99]`；C2 使用
    `validation/c2_lipid_slab_v11/<case>/` 的 raw 输入和
    `validation/c2_lipid_slab_v11_full11/<case>/dynamics/` 的轨迹，四格
    `Na_thin_pos0`、`Na_thin_pos1`、`Na_thick_pos0`、`Na_thick_pos1` 各 20 帧，
    `λ=1` 使用同一组 10 个索引，`λ=0` 使用
    `[20,40,60,80,100,120,140,160,180,199]`。C2 full-11 的
    `protocol version=11`、raw/prepared hash、原子顺序和 co-ion fingerprint
    必须先核对；禁止把 `system_prepared.xml` 再传给 charging configure。
  - **数值门：** 每帧、每比较独立判定；energy relative difference 定义为
    `abs(Eprod-Eref) / max(1 kJ/mol, abs(Eref))`，要求 `≤1e-5`，同时记录
    absolute ΔE；最大逐原子 force-component 差要求
    `≤1e-3 kJ/mol/nm`。vanishing `λ_vdw=0` 另要求
    `|E_ligand-environment|≤1e-6 kJ/mol`、
    `max|F_ligand-environment|≤1e-3 kJ/mol/nm`、`LRC coefficient=0`。
    任意一帧失败即该 case 失败，禁止用多帧平均掩盖失败。

#### C3 执行总计划（冻结）

##### 1. C3 的核心目标

C3 不跑新的自由能，也不重新采样。它是一套真实体系上的 Hamiltonian 端点恒等式测试：

```text
生产 builder 构造出的端点
            vs
完全独立、直接改粒子参数构造的 reference 端点
```

使用 C1/C2 已有轨迹的相同坐标、相同周期盒做单点能量和逐原子力比较。同时关闭 MEM-00h
唯一剩余的真实端点验收。

##### 2. 需要验证的四条恒等式

| 比较 | 生产侧 | 独立 reference | 验证内容 |
|---|---|---|---|
| A | charging `λ_coul=1` | 原始物理体系 | 满电荷端必须恢复基础力场 |
| B | charging `λ_coul=0` | 直接把 ligand 电荷置零、co-ion 充满 | charge-transfer 终点正确 |
| C | vanishing `λ_vdw=1` | charging `λ_coul=0` | 两阶段接缝完全一致 |
| D | vanishing `λ_vdw=0` | 独立构造的 decoupled reference | ligand–environment 静电和 LJ 严格归零 |

每一帧必须独立通过，禁止用多帧平均掩盖失败。

##### 3. 独立 reference builder

建议新增：

- `tools/validation/compare_charge_transfer_endpoints.py`
- `tests/test_charge_transfer_real_endpoints.py`

Reference 侧只允许读取：

- raw `system.xml`；
- `topology.cif`；
- `ligand_indices.json`；
- 冻结的 `coalchemical_ion_spec.json`；
- 当前帧 positions/box。

Reference builder 禁止调用：

- `abfe_core.co_alchemical_charge_offset_plan`；
- `ibs_engine.configure_charge_transfer_decharging`；
- `ibs_engine.configure_pme_ligand_charge_offsets`；
- `ibs_engine.configure_coalchemical_neutral_decharging`；
- co-ion selector；
- `ibs_engine.build_ibs_dual_system`；
- `abfe_core.create_ligand_internal_force`。

生产侧正常调用这些函数，因为它们正是被测对象。

Reference 必须自己完成：

1. 从 raw physical System 深拷贝。
2. 从冻结 spec 读取 ligand/co-ion index 和端点电荷。
3. charging `λ=0`：
   - ligand 基础电荷直接设为 0；
   - co-ion 直接设为最终物理电荷；
   - ligand–environment exception 静电置零；
   - 独立恢复 ligand–ligand 普通非键和 1-4 项。
4. vanishing `λ=0`：
   - ligand–environment Coulomb 和 LJ 全部删除；
   - ligand internal、环境内部、co-ion、Boresch 和安全 restraint 保留。

可以参考已有 synthetic direct-reference 测试的思想，但不能直接共用生产 planner：
`tests/test_charge_transfer_hamiltonian.py`。

##### 4. 生产 vanishing 端点如何取值

不能直接查询 IBS bias System 的总能量，因为 Group 1 是混合态 bias，Group 4 还有 WCA
sampling shell。

C3 必须构造固定物理 Hamiltonian：

```text
U_common + CV(λvdW=k)
```

要求：

- 不包含 IBS log-sum bias；
- 不包含 WCA sampling shell；
- `λ=1` 加入对应的 analytic LRC；
- `λ=0` 的 LRC 系数必须严格为零；
- 使用 `ibs_engine.py` 中已有 fixed-Hamiltonian 思路，但 C3 输出完整能量和力，不只
  查询 CV energy。

##### 5. 抽帧矩阵

###### C1

主案例：

```text
validation/c1_waterbox/Na_large
```

抽 20 帧：

- `λ=1`：10 帧；
- `λ=0`：10 帧。

100 帧轨迹固定索引：

```text
[10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
```

###### C2

使用 v11 full-11 seed2026 的四格：

- `Na_thin_pos0`；
- `Na_thin_pos1`；
- `Na_thick_pos0`；
- `Na_thick_pos1`。

每格抽 20 帧：

- `λ=1`：10 帧，索引同上；
- `λ=0`：10 帧，从 200 帧中取：

```text
[20, 40, 60, 80, 100, 120, 140, 160, 180, 199]
```

合计：

```text
C1：20 帧
C2：4 × 20 = 80 帧
总计：100 帧 × 4 种端点比较 ≈ 400 组单点计算
```

不需要 MD，GPU 成本主要是 PME 单点力计算。

推荐额外做 `Cl_large`，覆盖负电荷方向；`Ca_large` 可作为多 co-ion share 的增强测试，
但不作为 C3 最低关闭条件。

##### 6. C2 输入路径注意

C2 full-11 目录只有 prepared system 和轨迹。Raw build 输入必须来自：

```text
validation/c2_lipid_slab_v11/<case>/
```

轨迹来自：

```text
validation/c2_lipid_slab_v11_full11/<case>/dynamics/
```

必须核对：

- 原子数和顺序；
- raw system/topology hash；
- full-11 manifest 指向的 prepared-system hash；
- co-ion fingerprint；
- `protocol version=11`。

禁止把 `system_prepared.xml` 再传给 charging configure，否则会重复注入 offsets/restraint。

##### 7. MEM-00h 与 C2 专用 switching

这里要分清两层：

- C2 采样脚本为了膜稳定使用了 `0.995→1.000 nm` 窄 switching；
- MEM-00h 的全局生产协议是 `1.0 nm cutoff`、无 switching。

因此 C3 的硬验收基线必须是：

```text
cutoff = 1.0 nm
switching = false
全局 LRC protocol 与生产一致
```

C2 轨迹只提供真实膜构象和周期盒。评价 System 应将生产侧和 reference 侧同时规范到
MEM-00h 协议，并记录唯一发生的协议转换。

可以额外输出“按 C2 原采样 Hamiltonian 求值”的诊断，但它不能用于关闭 MEM-00h。

##### 8. 数值比较规则

每个 frame、每个比较均记录：

```text
E_production
E_reference
absolute ΔE
relative ΔE
max |ΔF_xyz|
force RMSD
最坏 atom index/name/residue
ligand 最大力差
environment 最大力差
```

硬门：

```text
energy relative difference ≤ 1e-5
maximum per-atom force-component difference ≤ 1e-3 kJ/mol/nm
```

vanishing `λ=0` 还需单独检查：

```text
|E_ligand-environment| ≤ 1e-6 kJ/mol
max |F_ligand-environment| ≤ 1e-3 kJ/mol/nm
LRC coefficient = 0
```

能量相对误差定义为：

```text
abs(Eprod-Eref) / max(1 kJ/mol, abs(Eref))
```

同时必须报告绝对误差，防止巨大总能量掩盖局部分量错误。

##### 9. 平台和单点求值纪律

权威运行建议使用：

```text
CUDA
Precision=double
DeterministicForces=true
```

不要使用当前默认的 CUDA mixed precision 直接套 `1e-3` 力阈值。

每帧必须：

1. 从 DCD 读取完整 triclinic box vectors；
2. `setPeriodicBoxVectors()`；
3. `setPositions()`；
4. 若存在 virtual sites，调用 `computeVirtualSites()`；
5. 显式设置所有 global parameters；
6. 零步直接取 energy/forces。

禁止：

- 最小化；
- `applyConstraints()`；
- 积分一步；
- 使用初始 `box_vectors_nm.npy` 代替逐帧盒；
- C2 只读取 `Lx/Ly/Lz` 而丢弃完整盒向量。

C2 动态 safety targets 若保留，必须从对应 timeseries 恢复到该帧，并在生产/reference 两侧
完全一致。更稳妥的是同时报告：

- nonbonded endpoint 分量；
- 包含共同 restraint 的完整总 Hamiltonian。

##### 10. 静态 fail-closed 检查

开始求值前必须拒绝：

- raw system 已含 charging offsets；
- prepared system 被二次配置；
- atom count/order 不一致；
- topology/system/DCD hash 不匹配；
- co-ion fingerprint 不一致；
- λ 不是精确 0 或 1；
- cutoff/switch/LRC 版本不符；
- global parameter 缺失或使用默认值；
- force group 缺失/重复；
- positions/box 非有限；
- reference builder 调用了任何生产 planner；
- LRC 未计入 `λ=1` 或 `λ=0` 非零。

建议测试中 monkeypatch 所有禁止函数为“调用即抛异常”，证明 reference 真正独立。

##### 11. 输出结构

建议：

```text
validation/c3_real_endpoints_v1/
├── protocol_manifest.json
├── frame_manifest.json
├── c1_Na_large/
│   ├── per_frame.csv
│   ├── systems_manifest.json
│   └── report.json
├── c2_Na_thin_pos0/
├── c2_Na_thin_pos1/
├── c2_Na_thick_pos0/
├── c2_Na_thick_pos1/
├── mem00h_report.json
└── summary.json
```

`frame_manifest.json` 固定记录：

- case/seed/source λ/frame index/time；
- coordinates hash；
- box vectors 和 hash；
- system/topology/spec hash。

每份 report 必须包含：

- OpenMM/Python 版本；
- platform properties；
- C3 protocol/schema 版本；
- acceptance-threshold payload；
- production/reference System hash；
- 每个失败 frame、atom 和 force group；
- `status=complete/incomplete`；
- `passed=true/false`。

##### 12. 实施顺序

###### C3-0：冻结协议

- [x] C3 protocol version=1；
- [x] 独立 builder 禁止调用列表；
- [x] energy/force/zero 阈值；
- [x] frame 选择；
- [x] production MEM-00h baseline。

###### C3-1：CPU 契约测试（2026-08-11 完成）

新增 `tools/validation/compare_charge_transfer_endpoints.py`（独立 reference
builder + production builder 薄封装 + `evaluate`/`compare_endpoint` + 静态
fail-closed 检查 + `wiring-smoke` CLI）与
`tests/test_charge_transfer_real_endpoints.py`（20 项，全部通过，合成小体系，
`Reference` 平台）：

- [x] 正/负 ligand 电荷（`ligand_net_charge_e ∈ {1,-1}` 参数化）；
- [x] 单/多 co-ion（额外 `ligand_net_charge_e=2, n_dummies=2` 参数化）；
- [x] 普通 pair、excluded pair、1-4（4 原子配体 fixture 覆盖三类 L-L 对，
  逐 exception 断言 chargeProd/epsilon，不是靠比总能量间接猜）；
- [x] ligand internal 保持（同上，且证明生产侧 `configure_charge_transfer_
  decharging` 无条件冻结 L-L 对——不分 λ=0/1；本轮修复前 reference builder
  曾只在 λ=0 冻结,被这条测试当场钉出结构性差异，已修复为无条件冻结）；
- [x] charging `λ=1/λ=0`（对比 raw 物理体系 / 独立构造的置零+满电参照）；
- [x] vanishing `λ=1/λ=0`（仅对**净中性**配体成立，见下方发现）；
- [x] LRC `λ=0`（`_lj_tail_lrc_coefficients_kj_mol` 直接调用断言系数严格为 0）；
- [x] reference planner 独立性（`forbidden_calls_disabled()` 把 7 个禁止函数
  换成"调用即抛异常"，reference builder 仍能正常跑完；另有一条测试证明守卫
  本身真的会抓到违规调用，不是形同虚设）；
- [x] 任一参数篡改能触发 gate（ligand/co-ion 索引重叠、非端点 λ、
  已配置过 charging 的 system 误当 raw 输入、protocol_version 不符）；
- [x] 缺 frame/box/hash 时 fail closed（非有限坐标/盒当场 `ValueError`；
  production/reference 粒子数不一致当场 `RuntimeError`）。

**2026-08-11 实测发现并修复（第一版归因错误，已用诊断脚本纠正）**：
`production_vanishing_fixed_hamiltonian_systems` 最初错误地假设"喂给
`build_ibs_dual_system` 的应该是 charging 配置完成后 base=0 的 System"，被
C 的 seam 测试实测炸出 0.71 kJ/mol 的差（远超 1e-5 相对容差）。**第一版归因
"Group 2 用调用时刻的粒子电荷重建配体内部 Coulomb，喂错了状态会把这一项
静默腰斩成 0"是错的**——用户要求先诊断再提设计后，补跑了一个逐 force-group
拆账的诊断脚本，证明 Group 2 全程一位不差（`create_ligand_internal_force`
本来就是从 exception 表直接读物理 chargeProd，不看粒子当前电荷）；**真正
根因是把 charging 配置完成后的 System 深拷贝进 `build_ibs_dual_system` 时，
`lam_coul` 这个 GlobalParameter（默认值 1.0！）连同它的 offset 一起被原样
克隆过去，而后续求值没有显式把它设回 0**——用默认值 1.0 求值等于"配体满电、
co-ion 中性"，正好把 charging λ=0 端点的电荷图景**颠倒**了，量级完全对得上
0.71 kJ/mol。显式设 `lam_coul=0.0` 后差值降到 0.0005 kJ/mol（相对差 1.3e-5，
这个残差本身是诊断脚本自己那个原子间距极端紧凑的合成 fixture 的大数相减
浮点噪声，不是 Hamiltonian 问题）。当时的修复（改吃 raw system，绕开这个
问题）本身没错，但没有找到真正根因，且没有解决带电配体的路径。

**由此确认的真实限制，直接挡住 C4**：`build_ibs_dual_system` 自身的静态
电中性防御是在**喂给它的那个 System 当前的**电荷上核对的（代码事实，不是
靠 `abfe_config.json` 里"Phase B3 尚未实现"那条可能过期的注释）。当前生产
代码里**没有**一条把 charge-transfer 的 charging `λ_coul=0` 端点安全交接给
vanishing 阶段的路径——不是"两种电荷需求互相打架"（诊断已经证明 Group 2
不需要额外的原始电荷来源），而是"没有一步把 charging 配置完成后 System 上
那个危险的活 `lam_coul` 参数（默认值 1.0，代表错误的端点）固化/清除掉"。
C1/C2 的带电探针配体也只跑过 charging（11 个 λ_coul 态），从未真正跑过
vanishing。**C4 要求"Stage 2 中 co-ion 保持 fully charged"，这条 handoff
不存在，带电配体的 C4 双腿闭环就搭不起来**——设计文档见仓库根目录
`STAGE2_CHARGE_TRANSFER_HANDOFF_PROPOSAL.md`。

**2026-08-11 用户审阅批准方向后，工具函数已实现并测试（仍未接入生产调用链）**：
`abfe_core.bake_global_parameter_into_fixed_nonbonded_force(system,
parameter_name, lambda_value)`，把某个 GlobalParameter 在给定端点上的取值
固化成静态参数、彻底删除这个 GlobalParameter（不是给 `build_ibs_dual_system`
加新参数）。实现过程中用真实 Context 纠正了设计文档第一版的两处描述：
① OpenMM 对同一 (parameter, particle)/(parameter, exception) 上重复的
offset **不做加法**，只认最后一条（实测：0.3+0.2 两条追加，Context 求值
结果对应 0.2 不是 0.5）——契约相应改成"检测到重复就 fail closed"，不是
"先聚合再烘焙"；② 返回值的 Quantity/裸 float 类型不稳定，需要统一转换。
`tests/test_bake_global_parameter.py`（12 项）覆盖：烘焙结果与显式设
Context 参数逐位相同、结构性删除 GlobalParameter、不相关参数/offset 原样
保留、重复 offset fail closed、完整保留 NonbondedForce 配置（cutoff/
switching/method/dispersion/reaction-field/Ewald/force group/name/
exceptions-PBC）、参数被别的 Force 引用时 fail closed、sigma/epsilon
offset 正确烘焙、非端点 λ 拒绝。**用真实带电 fixture 重新验证过 seam 相对差**：
换成正常键长键角（0.153nm/111.5°/anti 二面角）的伸展几何后，Reference 与
CUDA `Precision=double`+`DeterministicForces=true` 上相对差都是 ~1.2e-9
（比 1e-5 门宽 4 个量级）；原诊断用的紧凑合成 fixture（原子间距<0.2nm，非
真实分子构象）上有一个很小（~0.0005 kJ/mol）、Reference/CUDA-double 互相
吻合、与几何基本无关的绝对残差——**这推翻了"大数相减放大浮点误差"的最初
猜测**（若真是大数相减，换伸展几何后绝对差该跟着降，但它没有），具体来源
未定位，量级足够小不阻塞，如实记录。**接下来仍未做的事**：把这个函数接入
`runabfe.py`/`abfe_pipeline.py` 真正的 Stage1→Stage2 调用链；新增
`CHARGE_TRANSFER_VANISHING_HANDOFF_PROTOCOL_VERSION`；把带电 fixture 的
seam/D 验证正式并入 `tests/test_charge_transfer_real_endpoints.py`（目前
只有独立诊断脚本）；`compare_charge_transfer_endpoints.py` 的
`compare_endpoint` 已经把 resolved platform/property 核对做成硬门（不
匹配直接让 `passed=False`），但 `run_matrix` 仍然只能跑 A/B。C3 的
vanishing 比较（C/D）现状只能在合成中性配体上做契约测试，不能拿真实
C1/C2 数据验证"带电配体 + vanishing"。

**2026-08-11 全套离线回归复核（用户本机跑）**：`1196 passed, 3 skipped,
1 deselected (needs_gpu), 1 failed`——失败项
`tests/test_orb_latent.py::test_cached_omol_path_is_resolved_without_network_when_runtime_is_available`
是本机缺 `orb-v3-conservative-omol` 缓存权重，跟本次改动无关（用户确认），
新增的 `abfe_core.bake_global_parameter_into_fixed_nonbonded_force` 与
`compare_endpoint` 的 resolved-platform 硬门均无新增失败。

###### C3-2：单帧 wiring smoke（2026-08-11 CPU 平台完成，非权威数值门；
**仅证明接线正确，不构成 C3-3/C3-4 的权威数值判定**）

依次跑通（`compare_charge_transfer_endpoints.py wiring-smoke`，CPU 平台）：

1. C1 `Na_large` 一帧（frame_index=10，λ=1/λ=0 各一帧）：[x] 能量相对差
   ≈3e-10、最大力差 ≈8.5e-4 kJ/mol/nm，两者都在门内；
2. C2 `thin_pos1` 一帧（frame_index=20）：[x] 能量相对差 ≈1e-9（几乎精确），
   [ ] 最大力差 1.47e-3 kJ/mol/nm，超出 1e-3 门（约 1.5 倍）；
3. C2 `thick_pos0` 一帧：[x] 能量相对差 ≈4e-10，[ ] 最大力差 1.47-2.2e-3
   kJ/mol/nm，同样超出。

逐 force-group 账目已打印并核对：**三个 case 的所有帧，group 1（IBS 混合
bias）与 group 4（WCA 防护壳）全部严格为 0**——没有把它们算进物理端点。
C2 两个 case 的 restraint groups（6=co-ion flat-bottom、7=bulk-water、
8=ligand safety、9=co-ion safety）读数与"当前帧配体/co-ion 实际有没有贴着
边界"定性相符（非零但量级合理），无异常。

**2026-08-11 三平台归因已完成并 CONFIRMED（此前"判读为浮点噪声"是未经验证
的猜测，已被下面的实测替代，不再作为结论依据）**：给 `evaluate()`/
`compare_endpoint()` 加了 `platform_properties` 与**实际 resolved** 平台/
属性核验（不只记请求值），新增 `run-matrix`（固定帧索引表权威判定，暂只跑
A/B——C/D 见下方 handoff 缺口）与 `attribute`（同一帧在 Reference/CPU/
CUDA 三平台各跑一次完整比较）两个 CLI 子命令。对 `Na_thin_pos1`/
`Na_thick_pos0` 各自的 λ=1/λ=0 帧（`frame_index=20`，共 4 帧）逐一跑了三
平台归因，原始 JSON 落盘在 `validation/c3_real_endpoints_v1/attribution/`：

- [x] **Reference 平台：production 与 reference 的力差严格为 `0.0`**（bit-for-bit，
  4 帧全部如此）；
- [x] **CUDA（`Precision=double`、`DeterministicForces=true`，已核验
  resolved 属性确实生效，不是请求了没生效）：力差同样严格为 `0.0`**，4 帧
  全部如此；
- [ ] **只有 CPU 平台出现非零力差**（1.2e-3~2.4e-3 kJ/mol/nm，超过 1e-3 门），
  且"最坏原子"在 4 帧里分别是 4 个不同、看起来与配体/co-ion 无关的原子
  （索引 13197/13488/10809/17871）——不是同一个物理位置反复出问题。

**结论（有证据支撑，不是猜测）**：Reference 与 CUDA-double 都已是双精度且
彼此独立实现（Reference 是朴素直接 Ewald 求和，CUDA-double 走 cuFFT 双精度
倒空间），两者都严格为 0 而 CPU 平台单独出现非零差，且"最坏原子"逐帧随机
换——这个模式指向 **CPU 平台自身的数值实现细节**（最可能是其 PME 倒空间/FFT
路径的内部精度或求和顺序，CPU 平台没有可设置的 `Precision` 属性，不受我们
控制），**不是** production/reference 两侧构造了不同的 Hamiltonian。**权威
平台（CUDA 双精度）的力差是精确的 0**，比 1e-3 门还严好几个数量级——C2 的
真实数据在权威平台上通过力门槽，C3-2 wiring smoke 里 CPU 平台超门的现象
到此有了确认的归因，不再是悬而未决的猜测。

**仍然尚未实现、不能算已完成的部分**：`run-matrix` 目前只能跑 A/B（C/D 仍
卡在 charging→vanishing handoff 缺口，见上一节），也还没有在真实 C1/C2 的
完整 20/80 帧列表上跑过——只跑了 4 帧的三平台归因诚实核对，**C3-3/C3-4 仍未
关闭**。本仓库所在的这个 sandbox 里实测有一块空闲的 RTX 2080 Ti
（`nvidia-smi` 确认），单点力评估（非长程 MD 生产）体量很小，这次归因已经
用它跑通；后续跑完整 20/80 帧矩阵前需要与用户确认是否继续在本机跑还是转到
专用节点。

###### C3-3：C1 主门（2026-08-11 已在真实生产 CUDA 精度上跑完，未通过，
证据见 `validation/c3_real_endpoints_v1/run_matrix_mixed_precision/`）

**用户 2026-08-11 决定**：C3 应验证真实生产配置（CUDA `Precision=mixed`，
不设 `DeterministicForces`），不是另造一个脱离生产的双精度门；之前一版用
CUDA 双精度+deterministic 的三平台归因是过度的精度研究，不再继续。

`Na_large` 20 帧（`run-matrix`，CUDA `Precision=mixed`，resolved 属性已核验）：

- [x] **能量门全部通过，且非常干净**：相对差稳定在 `1.75e-7` 量级（20 帧
  几乎不随 λ/构象变化），比 1e-5 门宽 4 个量级。
- [ ] **力门 8/20 帧失败**（1e-3 kJ/mol/nm 门，最坏达 1.5e-3）。**不是随机
  噪声**：8 个失败帧里 6 个的最坏原子都是索引 7177——正是这个 case 的
  **co-ion**（`build_manifest.json` 的 `reserved_coion_indices`）。

###### C3-4：C2 主门（2026-08-11 已在真实生产 CUDA 精度上跑完四格 80 帧，
证据同上目录）

四格结果（`run-matrix`，CUDA `Precision=mixed`）：

| case | 能量相对差范围 | 力门失败帧数 | 力差范围 (kJ/mol/nm) | 失败帧最坏原子 |
|---|---|---|---|---|
| `Na_thin_pos0` | 1.25e-7~1.34e-7 | **18/20** | 4.7e-4~**7.47e-2**（75×门槛） | 21517/21518 反复出现 |
| `Na_thin_pos1` | 1.25e-7~1.29e-7 | 7/20 | 2.0e-4~2.1e-2（21×门槛） | 21517/21518 反复出现 |
| `Na_thick_pos0` | 1.40e-7~1.42e-7 | **0/20（干净通过）** | 1.5e-4~4.5e-4 | 无 |
| `Na_thick_pos1` | 1.40e-7~1.44e-7 | **18/20** | 2.3e-4~**6.87e-2**（68×门槛） | 26125/26126 反复出现 |

**跨 case 的一致模式，不是各自独立的巧合**：

- 能量门在全部 100 帧（C1 的 20 + C2 的 80）上都干净通过，且相对差量级
  在每个 case 内部高度稳定（同一 case 不同帧几乎是同一个数）——说明生产/
  参照两侧构造的确实是同一个 Hamiltonian，charging 端点 A/B 本身没有问题。
- 力门失败**集中在 co-ion 或与它紧邻的原子**：C1 是 7177（该 case 唯一
  co-ion），C2 的 `thin_pos0`/`thin_pos1`（共用同一份 thin 拓扑）反复是
  21517/21518，`thick_pos1` 是 26125/26126（`thick` 拓扑下的对应位置）。
  `thick_pos0` 完全干净——同一份 thick 拓扑换一个 co-ion 空间位置就从
  0/20 变成 18/20 失败，说明这不是"这个原子天生数值差"，而是**位置相关**
  的敏感度。
**用户决定（2026-08-11）**：先如实记录这份数据，不撤销 1e-3 力门、不因为
"生产就是这个精度"而悄悄放宽。C3-3/C3-4 按当前门槛判定**未通过**，留给
用户决定后续结构性动作。

###### C3 co-ion/ParameterOffset 归因诊断（2026-08-11，用户限定范围的一次性
诊断；不调整 gate、不重跑 MD、不进入 C4）

方法（脚本：`tools/validation/diagnose_coion_parameteroffset_mixed_precision.py`，
结果：`validation/c3_real_endpoints_v1/coion_parameteroffset_attribution.json`）：
对 5 帧（`thick_pos0` 1 个 PASS 对照、`thick_pos1` 2 个 FAIL、`thin_pos0`
1 个 FAIL、C1 1 个 FAIL）各构造三个数学上应等价的 System——**A**：production，
`lam_coul`/`lambda_coul` GlobalParameter 仍是活的；**B**：把 A 用
`bake_global_parameter_into_fixed_nonbonded_force` 在同一端点烘焙成静态值；
**C**：独立参照。用 `NonbondedForce.setForceGroup`/
`setReciprocalSpaceForceGroup` 把 direct-space 与 reciprocal-space PME
拆成两个 force group 分别求值比较。

**结果，5 帧完全一致**：

- **A vs B（活参数 vs 烘焙成静态值，同一个 production Hamiltonian）**：
  direct-space 在全部 5 帧**精确为 `0.000e+00`**；reciprocal-space 只有
  ~1.3e-4~2.1e-4 的量级、且最坏原子是随机的（不是 co-ion 也不是其近邻）。
  **结论：`bake_global_parameter_into_fixed_nonbonded_force` 本身、以及
  "活 ParameterOffset vs 烘焙成静态值"这条机制，不是问题来源**——这一步
  验证的正是这个函数自己的正确性，干净通过。
- **B vs C（production 构造 vs 独立参照构造）**：direct-space 出现真正的大
  差异（2.97e-2~6.70e-2 kJ/mol/nm），reciprocal-space 很小（1.4e-3~1.9e-3）。
  最坏原子：C2 三个 FAIL 帧都是**配体侧的 restraint anchor 原子**
  （`thin_pos0`→21517，`thick_pos1`→26125）而不是 co-ion 本身（co-ion 自己
  的差值更小：1.76e-3~4.08e-3）；C1（配体只有 1 个原子，本身就是探针）最坏
  原子直接是 co-ion（7177，差值 1.48e-3）。`thick_pos0`（PASS 对照）在
  B vs C 上也保持干净（1.4e-4 量级，跟 A vs B 的平台底噪同量级）。
- **补测（只加了这一步，没有扩大范围）**：把 `thin_pos0_FAIL` 这一帧的
  `B vs C direct-space` 比较额外在 **Reference 平台**（双精度）跑一遍——
  `E_B` 与 `E_C` 精确相同到小数点后六位，**逐原子力差精确为 `0.0`**。同一对
  System 换到 CUDA mixed 就变成 2.97e-2 kJ/mol/nm 的差。

**结论（对应用户给的判读规则）**：B 与 C 在 Reference/双精度上逐位相同，
证明 production 与独立参照构造的确实是**同一个 Hamiltonian**，不存在真实的
构造差异（不是"B≠C 需要修构造"这一支）。差异只在 CUDA mixed precision 下
出现，且集中在配体内部 exception 数量较多的原子（restraint anchor）而不是
均匀分布——指向**CUDA mixed precision 的 direct-space NonbondedForce/
exception 计算核对"数值相同、构造顺序不同"（production 与参照两侧给
NonbondedForce 添加 exception 的顺序天然不同）敏感**，是平台数值路径问题，
不是 Hamiltonian 构造错误，也不是最初怀疑的"reciprocal-space ParameterOffset
力梯度路径"（那一条已经被 A vs B 的干净结果排除）。

**遗留问题（本轮诊断没有回答，留给下一步）**：C3-3/C3-4 里"同一份 thick
拓扑换一个 co-ion 位置就从 0/20 变成 18/20 失败"的位置相关性，本次诊断没有
单独解释——已知"差异集中在配体侧 exception 较多的原子"，但没有确认为什么
`thick_pos1` 的这个原子比 `thick_pos0` 对应原子对 mixed precision 更敏感
（可能与该帧原子间距、局部 PME 网格点分布有关，未验证）。

**仍然没有做的事（按用户指示）**：没有调整 1e-3 力门；没有重跑任何 MD；
没有关闭 C3；没有进入 C4。

**带电配体 C/D（handoff 接通后，2026-08-11）**：D（严格零）在 Reference 与
CUDA `Precision=mixed` 上都精确为 `0.0`——符合预期，D 的零是代数结构性的
（`λ_coul≡0`/`λ_vdw=0` 直接让对应项系数为零），不依赖数值精度。C（seam）在
CUDA mixed 上力差达到 `4.0 kJ/mol/nm`（比 1e-3 门大 4000 倍！），比真实
C1/C2 数据看到的最坏情况（`7.5e-2`）还要严重得多。这个合成 fixture 只有
~15 个原子、盒子很小（6×6×12nm）——小体系在 mixed precision 下的 PME 网格
精度本来就是已知的一般性限制（网格点少，单点误差占比更大），怀疑是这个
原因而不是 handoff 机制本身的问题（Reference 平台上同样的 fixture 相对差
`1.2e-9`，干净），但**没有去证实**，如实记录，不下结论。

###### C3 protocol v2：双层门重设计（2026-08-11，用户提出）

**背景**：上面的归因诊断已经证明——B≡C 在 Reference/双精度上逐位相同
（同一个 Hamiltonian，构造没有问题），差异只在 CUDA mixed precision 下
出现，且取决于两侧给 `NonbondedForce` 添加等价 exception 的**顺序**而非
数值本身。用户据此判断：**单一绝对力容差在这里没有稳定物理含义**，不能从
观测到的最大失败值反推一个新容差（如 `0.07`/`0.1`）去替换 `1e-3`——那是
事后定门，明确被否决。

**v2 设计（`tools/validation/compare_charge_transfer_endpoints.py`
`PROTOCOL_VERSION=2`）**：把原来单一的 A-vs-C 比较拆成三类判据，每帧独立
判定：

| Gate | 比较对象 | 平台 | 能量门 | 力门 |
|---|---|---|---|---|
| **Gate 1** 独立 Hamiltonian 恒等性 | production/baked vs 独立参照 | **Reference**（双精度） | ≤1e-5 相对差（硬） | ≤1e-3 kJ/mol/nm（硬） |
| **Gate 2** 生产一致性 | 活 `ParameterOffset` vs 烘焙成静态值（同一 production Hamiltonian） | **CUDA mixed** | ≤1e-5（硬） | ≤1e-3（硬） |
| **Gate 3** 生产 vs 参照 | production vs 独立参照 | **CUDA mixed** | ≤1e-5（硬） | **只诊断，不作为失败条件**（原因：exception 排列顺序导致的 mixed precision 数值路径差异，与 Hamiltonian 正确性无关，已由 Gate 1 独立证明） |

不变的硬约束：任何 gate 下逐原子力必须是有限值（NaN/Inf 直接判 fail）；
D（严格零）保持硬门。**这个设计没有放宽物理正确性**——Reference 平台仍然
严格验证独立构造是否一致（Gate 1）；CUDA mixed 下"活参数 vs 固化成生产"
仍然严格要求一致（Gate 2，这条本来就干净通过）；只取消了"CUDA mixed 下
两种不同但等价的 exception 排列必须达到 `1e-3`"这一条不成立的要求
（Gate 3 力门降级为诊断）。

**用户明确的执行顺序**：① 在本文件登记 protocol v2 和理由（本节）；
② 保留现有 v1 mixed-force FAIL JSON 不覆盖（`validation/c3_real_endpoints_v1/`
目录未改动，md5 核对一致）；③ 给 runner 加上述三类判据（新增
`run_protocol_v2_matrix` + CLI `run-matrix-v2`，`force_gate_mode` 参数
控制 hard/diagnostic，5 个新 CPU 契约测试覆盖 diagnostic 分支）；④ 用现有
100 帧重新后处理，**不重跑 MD**（复用 `validation/c2_lipid_slab_v11*/` 下
已有的 DCD 轨迹和 `validation/c1_*` 已有数据）；⑤ 结果写入新目录
`validation/c3_real_endpoints_v2/`，与 v1 并存，不覆盖。

**v2 复算结果（100/100 帧，C1 的 20 + C2 四格各 20）**：

| case | Gate1 能量/力 (Reference) | Gate2 能量/力 (mixed，硬) | Gate3 能量 (mixed，硬) | Gate3 力 (mixed，诊断) | 帧结果 |
|---|---|---|---|---|---|
| `C1_Na_large` | 0 / 0（逐位相同） | 1.78e-7 / 1.69e-4 | 1.82e-7 | 最坏 1.41e-2 | **20/20 通过** |
| `Na_thin_pos0` | 0 / 0 | 1.29e-7 / 2.44e-4 | 1.35e-7 | 最坏 7.47e-2 | **20/20 通过** |
| `Na_thin_pos1` | 0 / 0 | 1.29e-7 / 2.90e-4 | 1.29e-7 | 最坏 2.11e-2 | **20/20 通过** |
| `Na_thick_pos0` | 0 / 0 | 1.42e-7 / 2.44e-4 | 1.42e-7 | 最坏 4.55e-4 | **20/20 通过** |
| `Na_thick_pos1` | 0 / 0 | 1.42e-7 / 2.67e-4 | 1.44e-7 | 最坏 6.87e-2 | **20/20 通过** |

**结论**：`n_failed=0`（100/100），`failed_frames=[]`，每个 case 的
`passed=True`。Gate 1 在全部 100 帧上逐位精确相同（0/0），是本轮最强的一条
证据——独立参照与 production/baked 构造在双精度参照平台上没有任何可测差异。
Gate 2（生产实际用的 CUDA mixed，活参数 vs 固化）力差最坏 2.9e-4，比 1e-3
门还留 3 倍余量，干净通过。Gate 3 能量在全部 100 帧稳定在 ~1e-7 相对差，
硬门干净通过；力差诊断值域 4.55e-4~7.47e-2，与此前归因诊断和 v1 结果的量级
完全吻合（例如 `Na_thin_pos0`/`Na_thick_pos1` 仍是最坏的两格），**符合预期
且不作为失败**——这不是新信息，是同一套已诊断清楚的 mixed-precision
exception-ordering 现象在新判据下被正确分类为"诊断"而非"失败"。

A/B 的 v2 复算范围明确限定为"用现有 100 帧重新后处理"，当时未覆盖 C/D——
下一节是 C/D 的 v2 应用结果（同一天完成）。

###### C/D 的 v2 应用（2026-08-11，用户明确指示：复用真实 charging λ=0 帧，
不新跑 vanishing MD；结果**发现一个真实的、此前已知但未解决的 Hamiltonian
不一致，C 未通过，暂停在这里等用户决定**）

**方法**：Stage2 handoff 落地后，带净电的 charge-transfer 配体（C1/C2 的
单原子 Na+ 探针）已经可以走"charging 配置→烘焙→喂给 `build_ibs_dual_
system`"的真实链路，不再局限于合成中性配体。新增
`run_protocol_v2_matrix_cd`：直接复用 A/B 的 B 端点已经采样过的
charging λ_coul=0 真实帧（C1 10 帧 + C2 四格各 10 帧 = 50 帧，不跑新 MD），
对每帧做：

- **C**（seam）：baked charging λ_coul=0 vs production vanishing λ_vdw=1，
  Reference 硬门 + CUDA mixed（energy 硬/force 诊断）。
- **D**（strict zero + vs 独立 reference）：production vanishing λ_vdw=0
  vs 独立构造的 `reference_vanishing_zero_system`，同样两层门；外加
  `compare_vanishing_zero_endpoint` 的严格零门，Reference 和 CUDA mixed
  下**都是硬门**（不受诊断化影响，因为它是代数结构性的零）。
- **gate2**（live-vs-baked）：C/D 在比较发生的时刻两侧都没有活的
  GlobalParameter（vanishing 侧每态 CV 是固定构造，charging 侧已被烘焙），
  标记 `applicable=False`，不参与 `passed`，不制造无意义比较（用户指示）。

**结果（50 帧，Reference 平台/硬门，不涉及 CUDA mixed 精度）**：

| case | C（seam）gate1 力差范围 (kJ/mol/nm) | C 通过 | D gate1/严格零 | D 通过 |
|---|---|---|---|---|
| `C1_Na_large` | 4.5e-13~1.7e-12（精确匹配） | **10/10** | 精确 0.0 | **10/10** |
| `Na_thin_pos0` | 2.3e-13~**3.88e-01** | 4/10 | 精确 0.0 | 10/10 |
| `Na_thin_pos1` | 5.7e-02~**5.62e-01** | 0/10 | 精确 0.0 | 10/10 |
| `Na_thick_pos0` | 2.2e-02~**6.40e-01** | 0/10 | 精确 0.0 | 10/10 |
| `Na_thick_pos1` | 2.3e-13~**5.89e-01** | 1/10 | 精确 0.0 | 10/10 |

能量在全部 50 帧上都干净通过（相对差 ~1e-6，比 1e-5 门宽一个量级）——
只有**力**在 C2 上大量失败。**D 在全部 50 帧上 100% 精确通过**（严格零、
vs 独立 reference 都是），证明带电配体的 D 端点构造完全正确，问题只在 C。

**根因已经用消融实验确定，不是猜测**：C2 raw System 自己的
`NonbondedForce` 带一个 `[0.995, 1.000]nm` 的窄 LJ switching 窗口——这是
`validate_charge_transfer_lipid_slab.py`（C2 PROTOCOL_VERSION v2→v3，
2026-08-07 已有记录）为修复 `MonteCarloMembraneBarostat` 在硬 1.0nm 截断下
造成的膜面内人工压缩（APL 10ns 内从 0.683 压到 0.590 nm²）而特意加的，
**当时的代码注释已经明确写明**"范围只到 C2 自己的 System 构建，不改
`ibs_engine.SOFTCORE_CUTOFF_NM` 这些全局 MEM-00h 常量……需要独立决策，不在
本轮顺手改"——也就是说这个不一致 2026-08-07 就已知，被有意推迟决策，不是
这次才引入的新 bug。真正触发 C3 失败的正是这个被推迟的不一致：vanishing
阶段配体-环境 softcore CV（`CustomNonbondedForce`）从始至终是硬截断
（`useSwitchingFunction=False`，全局 MEM-00h 约定），跟 C2 这个局部窗口
不匹配——**逐帧统计证实是充要条件**：只要该帧配体-环境存在至少一对
LJ-active（非氢）距离落在 `[0.995,1.0]nm` 窗口内，C 就必然出现非零力差
（帧 40 只有 1 对、差值 8.9e-4 刚好卡在门内；`thick_pos0`/`thin_pos1` 几乎
每帧都有这样的对，故几乎全部失败）；零对时力差精确为 `0`（`1e-13` 量级）。
额外用消融验证过：把 charging 侧的 `NonbondedForce.setUseSwitchingFunction`
强制关掉重新比较，结果是**全部帧都变差**（力差跳到 3.7~5.8 kJ/mol/nm）——
证明 switching 本身没错、是两侧"要不要 switching"的约定不一致，不是
switching 实现本身有 bug。这是 Reference/双精度平台上的**结构性**差异，
跟这次诊断出的"CUDA mixed exception 排列"完全是两件独立的事，不能也不该
用 Protocol v2 的 force-diagnostic 机制去掩盖——v2 的 force-diagnostic 只
覆盖"两个等价构造在 mixed precision 下的力差"，这里恰恰是**两个不等价的
构造**（switch vs 无 switch），Gate1 在 Reference 上如实抓出来了，是这个
门正常工作的证据，不是门本身需要调整。

**用户当场指出：我最初给的三个候选选项都不是最合适的处理，也指出了我第一版
消融实验的方法性错误**——只关掉 `charging0_baked` 一侧的 switch，`vanishing_
one` 的 Group0（环境–环境，继承自同一份 raw System）却仍然带着 switch，
制造了一个新的、更大范围的环境–环境不一致，而不是移除原来那个局部的
配体–环境不一致；这解释了"全部帧都变差"这个结果，但**不能**据此反驳
"switch 不一致是根因"这个结论。

**正确修法（用户指定的第四条路，不是三选项里的任何一个）——C3 双边
normalization**：C3 求值时不应该继续背着 C2 采样脚本自己的 switch 惯例；
C2 轨迹只提供构象和周期盒，端点恒等式对任何坐标都应该成立。新增
`mem00h_normalized_raw_system()`：在分支出 charging/baked/vanishing/
reference 之前，先把共同的 raw System clone 统一转到 MEM-00h 的
`cutoff=1.0nm, switching=False`（cutoff 只核验、不强制改写；switching 强制
关闭）；A/B/C/D 全部从这同一份归一化 clone 分别构造。配套
`assert_mem00h_switching_convention()` 在每个关键构造节点核验 cutoff/
switching 确实传导到了最终喂给 Context 的 System 上。**不改 C2 已有的
raw 文件/轨迹本身**——归一化只发生在 C3 评估工具内部的一份内存 clone 上，
C2 生产采样（继续用局部 switch 解决膜压缩问题）完全不受影响。已接入
`run_protocol_v2_matrix`（A/B）和 `run_protocol_v2_matrix_cd`（C/D），两处
都在 `load_case_raw_inputs()` 之后立即调用。7 个新 CPU 契约测试
（`tests/test_charge_transfer_real_endpoints.py`，含在一个可控小体系上
精确复现"环境原子落进 switch 窗口→非零力差"、再证明归一化后回到机器精度
的独立构造验证，不依赖真实 C2 数据）+ 全量离线回归 1213 passed/0 failed。

**修复后重跑全部真实 GPU 数据（A/B 100 帧 + C/D 50 帧，同一批已有轨迹，
未重跑 MD）**：

| case | A/B（100 帧中的 20） | C（seam）力差 | C/D 总体 |
|---|---|---|---|
| `C1_Na_large` | 20/20 | 1.7e-12（不变） | 10/10 |
| `Na_thin_pos0` | 20/20 | **5.97e-13**（原 3.88e-01） | 10/10 |
| `Na_thin_pos1` | 20/20 | **4.55e-13**（原 5.62e-01） | 10/10 |
| `Na_thick_pos0` | 20/20 | **4.55e-13**（原 6.40e-01） | 10/10 |
| `Na_thick_pos1` | 20/20 | **4.55e-13**（原 5.89e-01） | 10/10 |

**A/B 100/100 + C/D 50/50，全部 150 帧真实数据一次通过，C2 的 C-seam 力差
从最坏 0.64 kJ/mol/nm 回落到跟 C1 同一量级的机器精度（1e-13~1e-12）**——
证明这次修复解决的正是根因，不是掩盖症状。原始 v1 FAIL 记录（
`validation/c3_real_endpoints_v1/`）以及本节上方 2026-08-11 早些时候的 v2
失败记录仍保留在本文件里作为诚实的过程记录，不删除、不改写；
`validation/c3_real_endpoints_v2/` 目录下的 JSON 已更新为归一化后的结果
（同名覆盖，因为这些是"当前判据下的最新结果"而不是像 v1 那样的独立历史
快照）。

要求：

- 100/100 frame selection 完整；
- 四种 endpoint comparison 全部逐帧通过；
- 无 NaN；
- 无任何被平均掩盖的失败；
- MEM-00h protocol 字段一致；
- `summary.json=PASS`。

##### 13. 失败定位

- A 失败：charging `λ=1` 未恢复基础体系，重点查 offsets、exceptions 或重复 restraint；
- B 失败：charge-transfer `λ=0` 映射、co-ion 电荷或 ligand internal freeze 错；
- C 失败：charging→vanishing 接缝、LRC 或 Group 2 重构错；
- D 失败：残留 ligand–environment LJ/Coulomb、exclusion 或 `λ=0` LRC 不为零；
- 仅膜帧失败：重点查逐帧盒、PBC、C2 switching 或动态 safety target；
- 仅 CUDA 失败而 Reference 通过：精度/platform 问题，不是物理 builder 问题。

最终只有 C3 和 `mem00h_report.json` 同时 PASS，才能同时关闭 C3 与 MEM-00h，并进入 C4。

**2026-08-11：C3 与 `mem00h_report.json` 同时 PASS——`validation/
c3_real_endpoints_v2/summary.json`/`mem00h_report.json` 均
`status=complete, passed=true`（用户确认）。C3 与 MEM-00h 已正式关闭，
进入 C4。**


<!-- 迁移块：Definition of Done 已完成条目 -->

- [x] MEM-00h 端点能量/力验收通过（2026-08-11，随 C3 关闭）。
- [x] B5 cache/resume/provenance 正式关闭（2026-08-09）。

更新日期：2026-07-29  
状态：设计与实施清单；尚未达到膜体系生产就绪  
目标：在现有 `dual_lambda + softcore + IBS/MBAR + Boresch` 主链上，增加可审计、
可恢复、可验证的膜受体–配体 ABFE。

---

## 0. 先给结论

- [ ] **不是只重写溶剂腿。**
  - 工作量上，溶剂腿的构建、离子计数、dummy/co-alchemical ion 身份与缓存改造可能最大。
  - 物理上，dummy-charge（下文统一称 **co-alchemical charge-transfer ion**）
    必须同时进入复合物腿和溶剂腿的 charging Hamiltonian。
  - 复合物腿还必须增加膜恒压器、膜平衡质量门、膜兼容的 LJ 协议和 co-ion 空间限制。
- [ ] **带净电荷配体的膜 ABFE 默认优先采用 co-alchemical charge-transfer ion。**
  - 配体去电荷时，把同号电荷转移给一个位于体相水区的中性 co-ion：

    ```text
    ligand: q -> 0
    co-ion:  0 -> q
    ```

  - 每个 λ 状态的总电荷保持严格不变。
  - 这是 charge-transfer，不是把一个异号反离子也同时去电荷的 co-annihilation。
  - 对 `|q| > 1`，默认使用多个单价 co-ion 分担电荷，不把多个单位电荷集中到一个
    非物理多价粒子上。
- [ ] **co-alchemical ion 与 APBS/Rocklin 是二选一的净电荷处理路线，禁止重复修正。**
  - 若模拟始终保持总电荷不变：`APBS/Rocklin correction = 0`，并记录
    `not_applicable_co_alchemical_charge_transfer`。
  - 若使用 PME neutralizing plasma、体系总电荷随 λ 改变：才允许使用当前
    `apbs_correction.py` 的 Rocklin/RIP 后处理。
  - 中性配体既不需要 co-ion，也不需要 Rocklin 净电荷修正。
- [ ] **APBS 只处理静电有限尺寸/介电非均匀项，不处理 LJ 色散尾项。**
  LJ 协议必须独立决定。
- [ ] **当前热力学目标默认定义为水相 1 M 标准态结合自由能。**
  当前溶剂腿是 ligand-in-water。若目标改成“膜相配体与受体结合”，必须新增膜分配腿
  或膜相参考腿，不能沿用本清单的两腿闭环。

论文依据：

- Wu Z, Biggin PC. *Correction Schemes for Absolute Binding Free Energies
  Involving Lipid Bilayers*. JCTC 2022,
  DOI: [10.1021/acs.jctc.1c01251](https://doi.org/10.1021/acs.jctc.1c01251)。
  该工作比较了膜体系中的 Rocklin、co-annihilation 和 charge-transfer，并推荐
  co-alchemical charge-transfer ion。
- 后续风险核对：
  [Probing Limitations of Co-Alchemical Charge Changes in Free-Energy Calculations](https://pmc.ncbi.nlm.nih.gov/articles/PMC12159973/)。
  co-ion 位置、距离、局域强电场和把多单位电荷集中到一个粒子上都必须纳入验收。

---

## 0.5 现状核对（2026-07-29 对当前四文件实测，**先读这节再动手**）

本清单原稿是按「仓库尚无任何带电配体处理」写的。实测不是这样：**生产路径上已经
有一条带电配体的共炼金实现在跑，而且它是 co-annihilation，不是本清单第 0 节选定的
charge-transfer。** 下面每条都带 `file:line`，Phase A 必须逐条裁决"改造 / 退役 / 保留"。

### 0.5.1 已存在但与本计划冲突的实现

- [ ] **MEM-00a：`configure_coalchemical_neutral_decharging`（`ibs_engine.py:1784`）是
  co-annihilation，不是 charge-transfer。**
  它挑一个**异号**反离子（`target_ion_charge = -1.0 if lig_net_charge > 0`，
  `ibs_engine.py:788`），用**同一个** `lam_coul` offset 让配体和该反离子**一起**去电荷
  （`ibs_engine.py:1850-1866`）。总电荷确实逐 λ 守恒，但物理是"同时湮灭一对电荷"，
  与第 0 节写的 `ligand: q->0 / co-ion: 0->q` 相反。

  **✅ 已裁决（2026-07-29）：改成 charge-transfer；co-annihilation 降级为实验对照。**

  裁决理由：co-annihilation 同时执行 `ligand: q → 0` 与 `counterion: −q → 0`，
  总电荷虽守恒，但**两个异号离子在膜/水非均匀环境中的消失自由能不能可靠抵消**——
  一个在结合位点、一个在体相水，介电环境完全不同。charge-transfer 只把电荷从结合
  位点搬到体相水区，总电荷全程不变；Wu & Biggin 的膜体系比较中它对盒尺寸和膜组成
  的依赖明显更小，因此作为生产默认。

  落地动作：

  - [x] **MEM-00a-1** ✅ 2026-07-30（B2）：`charge_treatment` 枚举与
    `CHARGE_TRANSFER_PROTOCOL_VERSION = 1` 已落在 `abfe_core.py`，与
    `SOLVENT_CACHE_PROTOCOL_VERSION`、`IBS_BIAS_PROTOCOL_VERSION` 并列、未复用。
  - [ ] **MEM-00a-2**：现有 `configure_coalchemical_neutral_decharging` 改名为
    `configure_co_annihilation_experimental`，对应 `charge_treatment` 值为
    `co_annihilation_experimental`。
    - [ ] **不允许出现在任何膜生产 preset 中**（`system_type=membrane` + 该值 → fail closed）；
    - [ ] 只允许用于水盒、lipid slab 与方法对照；
    - [ ] 其所有输出（manifest / result / provenance）必须带
      `experimental_not_for_production: true`。
  - [ ] **MEM-00a-3**：**先不删旧代码。** 它的用途是给 charge-transfer 当负对照，
    验证盒长依赖与膜偏差（§8.1 末条、§8.2 末条已有对应条目）。
    等 charge-transfer 完成端到端验收（§12 Definition of Done）后再决定是否退役。
  - [ ] **MEM-00a-4**：**旧数据一律作废，禁止迁移。** co-annihilation 产出的
    charging checkpoint、`u_kn`、IBS 状态（f_k / bias）、预优化缓存全部 fail closed，
    不得迁移或"折算"到 charge-transfer 路线。
  - [ ] **MEM-00a-5**：`GhostIonHandler`（`abfe_core.py:2284`）**直接标记退役**，
    不在其上做任何扩展。它是固定空间点的自定义相互作用，不是进入 PME 粒子电荷表的
    共炼金离子，不能用来证明逐 λ 周期体系净电荷恒定。
    （它当前只被 `dexp_退役.py` 引用，退役不影响生产四文件。）
- [ ] **MEM-00b：这条路径是当前默认，不是死代码。**
  `configure_pme_ligand_charge_offsets`（`ibs_engine.py:1907`）在净电荷 ≠ 0 且
  `allow_charged_ligand=True` 时直接调它；三个传 `True` 的调用点是
  `_prepare_pme_mixed_alchemical_system`（`ibs_engine.py:1576`）、
  `_build_replicas`（`ibs_engine.py:13347`）、`compute_u_kn`（`ibs_engine.py:14763`）。
  即：**动力学、REMD 副本、能量重算三处各自独立地走了一次共炼金配置。**
- [x] **MEM-00c：反离子身份运行时重选 —— 已修（2026-08-04）：选一次 + 冻结 + 只读核对。**
  `_select_bulk_water_counterion`（现 `ibs_engine.py:811`）按"到最近溶质的
  minimum-image 距离 + 水配位数"当场排序挑离子。

  ⚠️ 原文这句判断**是错的**，B3 复核时别沿用：*"因为 MEM-00b 的三个调用点各调一次，
  若三处传入的 `positions` 不同（很可能）…"* —— 同一进程内三处拿的是**同一个**
  `self.positions`，不会漂。真正的漂移入口是**跨进程 resume**：首跑的
  `self.positions` 是预平衡输出**再叠 2000 步快速最小化**，resume（`skip_equil`）
  直接读 `pre_equilibration.dcd` **末帧**、不做最小化。两者差 0.01–0.1 nm，
  实测 **0.05 nm 即足以翻转选择结果**。

  修法与证据见 `docs/TODO.md` 的 MEM-00c 条目（选择唯一入口
  `select_co_alchemical_ion_once`、带指纹的 spec 落 `checkpoints/coalchemical_ion_spec.json`、
  6 个消费点全走 `resolve_co_alchemical_ion_spec()`、无 spec 即 fail closed；
  `tests/test_coalchemical_ion_identity.py` 20 条契约测试，全套 977 passed）。
  spec 里记 `charge_treatment`，所以它同时服务本条 co-annihilation 与 B3 的
  charge-transfer，两者不可互相复用。
  ⚠️ 尚未在带电体系上真机验证 —— 当前 Atenolol 净电荷 = 0，这条路径不被触发。
- [x] **MEM-00d：反离子 restraint 形式 —— 已修（2026-08-04，与 B3 同一改动）。**
  旧形式（`0.5*k*periodicdistance(x,y,z,x0,y0,z0)^2`，k = 25，参考点是**选中那一刻的
  绝对笛卡尔坐标**）有两个毛病：没有平坦区；参考点在膜半各向异性 NPT 下不随盒缩放，
  Z 方向盒长变而参考点不动 ⟹ 离子被系统性拖向膜。

  新形式 `flat_bottom_anchor_relative`：
  `0.5*k_ion*max(0, pointdistance(x1,y1,z1, x2+dx0,y2+dy0,z2+dz0) - r0_ion)^2`，
  k = 100 kJ/mol/nm²、r₀ = 0.5 nm（§13.1），force group 6，逐 λ 相同。
  井心 = **锚点原子当前位置 + 冻结位移 d0** ⟹ 随体系一起被 barostat 缩放。
  锚点 = 配体重原子中离配体质心最近的那一个，两条腿同一条规则。

  ⚠️ **实测 API 事实（别再猜）**：`periodicdistance` **只存在于 CustomExternalForce**；
  `CustomCentroidBondForce` / `CustomCompoundBondForce` 都报 `unknown function`。
  而 CustomExternalForce 只能吃绝对参考点，正是要退役的形式。可行组合是
  `CustomCompoundBondForce` + `pointdistance` + `setUsesPeriodicBoundaryConditions(True)`
  —— 打开 PBC 后 bond 内粒子被平移到与第一个粒子同一镜像，于是 `pointdistance`
  **就是** minimum-image 距离（实测：离子 z=0.2、锚点 z=9.4、盒 z=12 → 0.2 nm）。

  如设计所愿：`form` 变了 ⟹ 指纹变 ⟹ 旧 spec 与旧缓存自动作废；
  `CO_ALCHEMICAL_ION_IDENTITY_PROTOCOL_VERSION` 1 → 2。退役的绝对参考点保留为审计字段
  但**改了键名**（`reference_position_nm` → `selection_time_absolute_position_nm`），
  还在读旧键的消费者会 KeyError 而不是静默拿到已退役的参考点。
  证据见 `docs/TODO.md` 的 B3 + MEM-00d 条目与
  `tests/test_charge_transfer_hamiltonian.py`。
- [ ] **MEM-00e：restraint 自由能账目没有写进热力学循环。**
  该力加在 force group 6 且**逐 λ 相同**，所以腿内抵消；但复合物腿与溶剂腿的可用
  体积不同，§2.3 要求的"两腿是否抵消"说明目前在代码和文档里都不存在。
- [ ] **MEM-00f：`shadow_ibs` 路径对带电配体直接报错**（`ibs_engine.py:3222-3236`）。
  它是 `--decharge-method` 的实验性备选（`runabfe.py:1953-1967`，默认 `pme`）。
  若采纳 co-ion 路线，要么在 shadow 侧也实现（驱动变量换成 `lambda_shadow_coul`），
  要么显式声明"shadow 路径不支持带电配体"并让它 fail closed，不许留悬空分支。
- [ ] **MEM-00g：`GhostIonHandler`（`abfe_core.py:2284`）已只被 `dexp_退役.py` 引用，
  不在生产路径上。** §2.2 那条警告仍然成立（不要拿它当 PME co-ion），但**它不是当前
  真正的风险源**，MEM-00a–00e 才是。

### 0.5.4 真实膜体系（`memtest/`）暴露的五处身份判定缺陷（2026-07-30 已修）

`memtest/` 是 CHARMM-GUI FF-Converter 产的 **AMBER** 膜体系
（PROA 1 + POPC 90 + Na⁺ 25 + Cl⁻ 36 + TP3 9542 + 配体 1，共 45354 原子，
盒 6.096 × 6.096 × 11.867 nm 长方体、法向 z）。它把此前所有**基于残基名**的判据
全打穿了，而且**四处是静默出错、不是报错**：

| 实际 | 旧判据 | 后果 |
| --- | --- | --- |
| include 全是 `toppar/*.itp`，**不含 `amber` 字样** | 只看 include 路径 token | 力场族 fail closed，**跑不起来** |
| 一个 `POPC` = `PA` + `PC` + `OL` **三个残基**（一个 moleculetype、134 原子） | 按残基计数 | 脂质数 ×3、**APL 错 3 倍**；尾链残基无磷原子 → 直接 raise |
| 水叫 `TP3`（mdtraj `_WATER_RESIDUES` 只有 `TIP3`） | `select("water")` | 疏水核内水、co-ion 首层水配位数**静默为 0** |
| 离子叫 `Na+` / `Cl-`（带符号） | 名字集合无 `NA+`/`CL-` | 离子计数**静默为 0** |
| 蛋白含 `HID`/`ASH`/`NTRP`/`CCYS` | mdtraj `protein`（表里只有 `HIP`） | 骨架原子**静默少选 85 个** |

**根因不是"少了几个名字"。** 往硬编码集合里补名字是改产物不改生成器，换一套体系
又挂。`.top` 里本来就写着权威答案，所以身份一律从 `[ molecules ]` + `[ moleculetype ]` 来：

- `abfe_core.parse_gromacs_topology()`：递归展开 include，取 `[ defaults ]` /
  `[ moleculetype ]`（含残基构成与原子数）/ `[ molecules ]`。
- `abfe_core.molecule_atom_ranges()`：由 `[ molecules ]` 展开顺序算出每个分子的**精确**
  原子区间。已实测对齐：`PROA` 4566 原子 → 第一个 POPC 原子在 index 4566，
  与 `step7_production.gro` 里第一个 `PA` 的位置逐一致；末端 stop = 45354 = `.gro` 原子数。
- `abfe_core.classify_system_composition()`：判 protein / lipid / water / ion / ligand，
  判不出即 fail closed（静默归入 "other" 等于让它从所有原子选择里消失）；
  允许 `declared_roles` 显式覆盖并留记录。
- 力场族改以 **`[ defaults ]` 的 1-4 缩放**为主判据（Amber 0.5/0.8333 vs CHARMM 1.0/1.0），
  递归跟随 include 去找它（`[ defaults ]` 在 `toppar/forcefield.itp` 里，不在顶层 `.top`）；
  include 路径 token 降为次要信号，两者冲突时 fail closed 要求人工裁决。
  §1.1 原文写的就是"从 `#include` **与** `[ defaults ]` 判定"——此前只实现了一半。
- 叶片划分与 APL 改为**按分子**（`assign_lipid_leaflets(lipid_molecules=…)`）。
- 蛋白残基名自带一份含 Amber 变体的表 + N-/C- 端前缀归一（`NTRP`→`TRP`、`CCYS`→`CYS`），
  不再依赖 mdtraj 的 `protein` 关键字；没有组成兜底时会把 mdtraj 漏掉的残基**报出来**。
- 水氧原子解析**空集即报错**（膜体系必然有水，空集是识别失败而非事实）。
- `runabfe._resolve_gromacs_include_path` 收敛为 `abfe_core.resolve_gromacs_include`
  的薄包装，解析顺序只有一处实现。

证据：`tests/test_gromacs_composition.py`（直接对真实 `memtest/topol.top` 与仓库根
`topol.top` 断言，纯文本、无 GPU）。

---

### 0.5.5 OpenMM 不支持 `[ pairs ]` funct 2（2026-07-30 已修）

`memtest/` 首跑第一个硬错误：

    ValueError: Unsupported function type in [ pairs ] line:
        1  11  2 0.833333  -0.125447  -0.033096  3.39966950842e-01  7.62882666667e-02

OpenMM 的 `app.GromacsTopFile` 只接受 funct 1
（`gromacstopfile.py::_processPair` 里 `if fields[2] != '1': raise`）。
CHARMM-GUI 的 AMBER FF-Converter 对**部分**对写 funct 2，多出 `fudgeQQ q1 q2` 三列。
实测 `toppar/POPC.itp` 有 **21 条**（共 356 条 pairs），其余 7 个文件一条没有。

**为什么 OpenMM 报错是对的**：它读 `fields[3:5]`。对 funct 1 那正好是 `sigma eps`；
对 funct 2 会读成 `fudgeQQ q1`。所以这不是它太严，是那三列真的会让它读错列。

**为什么可以等价转换**：OpenMM 算 1-4 exception 的电荷用的是「粒子电荷 × 全局
fudgeQQ」（`atom1params[0]*atom2params[0]*fudgeQQ`）。所以只要两条成立，那三列就是
冗余重述：

1. 逐对 `fudgeQQ` == `[ defaults ]` 的全局值 —— 实测 21 条全是 `0.833333` ✅
2. `q1`/`q2` == 该 moleculetype `[ atoms ]` 的真实电荷 —— 实测 21 条全相等 ✅

`abfe_core.convert_gromacs_pairs_funct2()` **逐对校验这两条**，任一不成立即
fail closed（不成立意味着该对真的覆盖了静电缩放或电荷，硬转会静默改变哈密顿量）。

实现要点：
- **逐文件拷贝改写**，不做 include 展开：只有含 funct-2 的文件里那几行被改，
  其余文件逐字节拷贝。实测 `POPC.itp` 行数不变、仅 21 行不同，另 7 个文件
  `filecmp` 逐字节相同。
- `#ifdef POSRES` / `#ifdef DIHRES` 原样保留（位置限制就藏在里面，
  丢掉等于悄悄改变了"能不能做位置限制阶梯"）。
- **原始输入一个字节不动**；转换产物写到 `output_dir/gromacs_openmm_compat/`。
- 主 System 缓存指纹仍按**原始**输入算，但加入
  `GROMACS_PAIRS_FUNCT2_CONVERSION_VERSION` —— 改了转换逻辑会正确让缓存失效，
  否则是静默串协议。
- 转换结果（含两侧 SHA256、改写条数）进 `run_provenance.json`。

顺带确认：位置限制材料**在拓扑里就有**（`POPC.itp:1273`、`PROA.itp:42805` 的
`#ifdef POSRES`），不是缺 `posre.itp`；默认不激活。§3.2 的分级释放若要自己做，
走 `GromacsTopFile` 的 `defines` 即可。

**⚠️ 首次修这个问题时犯了和 B1 同样的错：只接了一个入口。**
第一版只在 `build_system_from_gromacs` 处做转换，结果溶剂腿的
`build_and_cache_solvent_leg`（另一个直接调 `app.GromacsTopFile` 的地方）照样炸。
全仓一共有 **6 处**加载点（`runabfe.py` 4 处 + `abfe_core.py` 1 处 +
`abfe_pipeline.py` 1 处）。这与 B1 当初只接了 1 个 `ABFEPipeline` 构造点
（实际 5 个复合物腿）是**同一个毛病：同一件事有多个入口，补一个漏一片**。

现在收敛为**唯一入口** `abfe_core.load_gromacs_topology_for_openmm()`：
- `openmm_compatible_gromacs_top()` 幂等、**内容寻址缓存**
  （key = 整棵树各文件 sha256 + 转换版本号），同一份输入只转一次，
  输入变了 key 就变，且不往用户输入目录里写东西；
- 主路径显式传 `output_dir/gromacs_openmm_compat` 让产物落在输出目录便于审计；
- 6 处加载点全部改走它，并有**契约测试禁止任何生产文件裸调
  `app.GromacsTopFile(`**（唯一豁免是入口函数自己那一行）——新加加载点会在测试里失败。

证据：`tests/test_gromacs_pairs_funct2_conversion.py`（20 条，直接对真实
`memtest/toppar/POPC.itp` 断言 + 合成拓扑验证两条 fail-closed + 唯一入口契约）。

---

### 0.5.6 水模型靠文件名识别（2026-07-30 已修）—— 同一类根因的第三次

`memtest/` 首跑第二个硬错误（过了 `GromacsTopFile` 之后）：

    ValueError: 在 topol.top 的 #include 里没认出任何水模型；
                已知的有 ['opc','opc3','spce','tip3p','tip3pfb','tip4pew','tip4pfb']

`resolve_water_model_xml()` 只看 `#include` 的**文件名词干**。对
`amber14sb_OL15_fs1.ff/tip3p.itp` 有效；但 CHARMM-GUI 的 AMBER 转换器把 TIP3P
命名为 **`TP3`**（`toppar/TP3.itp`），词干匹配必然落空。

**这是同一个根因的第三次出现**（前两次见 §0.5.4）：脂质按残基名计数、
水/离子按残基名计数、水模型按文件名识别——**靠名字判身份，换一套体系就错**。

修法：文件名词干优先（保证现有可溶体系行为不变），认不出时按**实际参数**判——
O/H 电荷 + O 的 σ/ε + 位点数。实测 `TP3` 是

    3 位点, q_O = −0.834, q_H = +0.417,
    σ_O = 0.315075240658 nm, ε_O = 0.635968 kJ/mol

在 7 个候选里**唯一**匹配 `amber14/tip3p.xml`（σ 逐位吻合）。最接近的竞争者
SPC/E 的 σ 差 1.5e-3 nm ≈ 容差(1e-6) 的 1500 倍，不存在误判空间。

关键设计：**候选参数直接从 OpenMM 自带的 XML 读出**
（`openmm/app/data/amber14/*.xml`），不硬编码参数表——"最终选出的 XML"与
"用于比对的参数"构造性来自同一处，不会抄错、不会随 OpenMM 升级漂移。
匹配必须**唯一**，多个候选落在容差内即拒绝（说明容差太松）。
认不出时报错信息里带上实测的四个数字，否则用户只知道"认不出"、
不知道自己的水到底是什么参数。

顺带扩展了 `parse_gromacs_topology()`：新增 `[ atomtypes ]`（σ/ε/质量/电荷）与
每个 moleculetype 的 `atoms`（含 atomtype、原子名、电荷、质量）——σ/ε 在
`forcefield.itp` 里，必须递归拿到。

证据：`tests/test_water_model_identification.py`（15 条，含"起怪名字的 SPC/E
也能按参数认出"、"唯一性余量 > 100× 容差"、"认不出时报错带实测数字"、
以及现有可溶体系仍走文件名的回归）。

---

### 0.5.7 mmCIF 拓扑缓存丢键 → PBC 修复撕开脂质 → NaN（2026-07-30 已修）

`memtest/` 首跑第三个硬错误，也是最难定位的一个：预平衡动力学出
`Particle coordinate is NaN`，而最小化"通过"了。

**根因**：`app.PDBxFile.writeFile` **不写任何键记录**（写入端没有 `struct_conn` /
`chem_comp_bond`），读取端只能靠 `createStandardBonds()` 补**标准残基**
（氨基酸/核酸/水）的键。于是 `topology.cif` 往返之后，
`POPC`（=PA+PC+OL）、配体 `MOL`、离子的键**全部静默丢失**，
而 `load_native_system` 当时只校验**原子数**、不校验键数。

`pre_equilibrate` 之前的「PBC 分子完整性修复」靠 topology 的键判断"什么算一个
分子" —— 键丢了就把跨周期边界的脂质**逐段撕开**。实测对比：

| 拓扑来源 | 最小化后 PE | max\|F\| |
| --- | --- | --- |
| `topology.cif`（丢键） | **4.109e+13** kJ/mol | **3.72e+09** @ `PA334/H8S`（脂质尾链氢） |
| `.top` 重建（有键） | −648536 kJ/mol | 2501 @ `HOH3812/O`（水，正常） |

**为什么绕了好几轮**：`runabfe` 在**全新构建**之后也会立刻
`load_native_system` 重新读回（"确保后续所有对象都来自落盘文件"，
`runabfe.py:3791` 附近），所以**首跑与缓存命中都会中招**；
而我的离线诊断直接用 `build_system_from_gromacs`（`.top` 拓扑，有键）+ 不调 PBC 修复，
所以怎么跑都不炸。**离线重建与生产路径不一致，是白花几轮的直接原因。**

定位靠的是 `memtest/reproduce_production_preequil.py` 的三段 bisect
（原样 / `--from-gro` / `--no-cache`）——它**只调生产函数**，不自己搭东西。

修复：
- `load_native_system(require_bonded_topology=True)`（膜体系必传）**优先从 `.top`
  重建**拓扑，mmCIF 只在没有 `.top` 时兜底；重建失败即 fail closed，不许退回 mmCIF。
- mmCIF 分支加**键数校验**：`topology.getNumBonds() < HarmonicBondForce.getNumBonds()`
  即判定丢键、丢弃该拓扑。
- 环境类型改为在**加载 System 之前**解析（`resolve_environment_type`，只看 config），
  并对两处解析加一致性断言——否则"要不要带键拓扑"与"用哪种 barostat"会基于不同判断。
  （首版把它写在使用点之后，自检直接抓出 `NameError`。）
- `pre_equilibrate` 新增**最小化后受力合理性门**：`max|F| > 1e6 kJ/mol/nm` 当场 raise，
  列出受力最大的 10 个原子、给出参考量级、指出最可能原因与验证方法。
  这条把"跑 10 ps 后一个没有上下文的 NaN"变成"起点就坏，且告诉你坏在哪"。
- `openmm_compatible_gromacs_top` 的复用改为按 `conversion_manifest.json` 逐文件
  核对 sha256（原先只查"顶层 top 存在 && 树里没 funct-2"，是 fail-open：
  半成品/被改过的转换目录会被静默复用）。

⚠️ **可溶路径的同类隐患未动**：溶剂腿与可溶复合物腿的 `MOL` 配体键在 mmCIF 往返里
同样会丢。改动会移动现有生产基线的预平衡起点（§7.7 / R7），所以单独立项评估，
已记入 `docs/TODO.md`。

**⚠️ 这个修复自己引入了下一个坑（同日修）**：`require_bonded_topology` 分支里
调 `load_gromacs_topology_for_openmm(top_file, includeDir=...)` 时**没传
`periodicBoxVectors`**，于是重建的拓扑没有盒矢量。而 `app.DCDFile` 写每帧 unitcell
时读的是 **topology** 的盒矢量（`dcdfile.py:155`，为 None 就整段不写、header 的
`boxFlag` 也是 0）——mmCIF 拓扑自带 `_cell` 所以历史上没暴露。
结果：10 ns 预平衡跑完（DCD 272 MB / 500 帧），却在 §9 质量门读轨迹时才报
「轨迹没有 unitcell 信息」。**烧完才发现**。

修法三处：
- `load_native_system` 的三条拓扑恢复路径**统一补盒矢量**
  （`topology.setPeriodicBoxVectors(box_vectors)`），重建时也直接传参；
- `pre_equilibrate` 在**开跑前**拦住"膜体系 + 拓扑无盒矢量"，不让它跑完才失败；
- 提取器的报错指明根因位置（`dcdfile.py:155`）与修法，避免下次再从头查。

证据：`tests/test_membrane_barostat_protocol.py` 的
`test_membrane_requires_a_bonded_topology_not_the_mmcif_cache` /
`test_topology_must_carry_box_vectors_before_a_membrane_run` /
`test_bad_starting_state_fails_closed_after_minimization`。

---

### 0.5.8 §9 质量门的 enforce / advisory 双模式（2026-07-30）

`memtest/` 的 10 ns 预平衡跑通后卡在质量门：那条 DCD 是在「拓扑缺盒矢量」那一版
写出来的，没有 unitcell，算不了 APL 与盒序列。而轨迹本身很健康
（末段 T 303.6–305.8 K、PE −508k~−510k、体积 438.97–440.37 nm³、密度 1.034 g/mL，
监控 1001 行），不值得为一道门重烧 10 ns。

所以给 §9 质量门加了 `membrane_quality_gate` 模式：

| 模式 | 行为 |
| --- | --- |
| `enforce`（默认） | 门未过即阻断。这是 §9 末句的原意 |
| `advisory` | 照样计算、照样落盘报告、失败大声 WARNING，但**不阻断** |

**为什么是 advisory 而不是让人注释掉调用**：门被注释掉就**没有记录**，
事后无从知道当时到底过没过。advisory 下：

- 报告仍完整落盘（`membrane_quality_gate.json`）；**连"算不出来"也落盘**
  （`{"evaluated": false, "blocked_reason": ...}`）；
- `membrane_quality_gate_mode` 进 `run_provenance.json` —— "这次是放行跑的"赖不掉；
- 模式值拼错直接报错，不静默回落成 enforce（反之亦然，两个方向都会让人误判资格）；
- 开跑前那道「拓扑无盒矢量」守卫也跟着分模式，否则等于用另一条路把门变硬。

⚠️ **advisory 不是生产资格。** 任何要报出的 ΔG_bind 必须在 `enforce` 下通过。
这句话同时写在代码常量注释、`memtest/abfe_config.json` 注释与放行时的 WARNING 里。

`memtest/abfe_config.json` 曾设 `advisory`，理由写在配置注释里。
**2026-08-02 已改回 `enforce`**（见 §0.5.9）：带 unitcell 的 DCD 已拿到，
且门本身的两个缺陷（MEM-01 / MEM-08）已修。

---

### 0.5.9 质量门自己坏了两处，而且一直没人知道（2026-08-02 已修）

08-02 又跑了一轮 10 ns 预平衡（5e6 步 30.2 min → **实测 476 ns/day**），
DCD 这次带 unitcell 了（§0.5.8 挂着的那一项由此完成）。但它把 §9 质量门自身的
两个缺陷顶到了台面上 —— **这道门从来没有在真实膜体系上评估成功过一次**。

#### MEM-01：分子路径下 `head_by_residue` 未绑定（同一根因的第四次）

`abfe_core.membrane_observables_from_trajectory` 里，§0.5.4 把叶片划分从"按残基"
改成"按分子"时分成了两条分支，但下面 `leaflet_composition` 那段仍然只读**残基分支的
局部变量** `head_by_residue`。memtest 有 `.top` 组成 → 走分子分支 → 该变量从未绑定：

    UnboundLocalError: cannot access local variable 'head_by_residue'

07-31 14:51 与 08-02 16:53 两次运行**逐字相同**，`membrane_quality_gate.json`
一直是 `{"evaluated": false}`。advisory 模式把它记成 WARNING 就过去了；
⚠️ 同样的崩溃在 `enforce` 下是 `raise`，所以"直接改成 enforce"当时会死在这里。

**这是 §0.5.4–§0.5.6 那个根因的第四次**，但换了个形态：前三次是"靠名字判身份"，
这次是"**身份口径改了，但有个消费点没跟着改**" —— 与 §0.5.5
"同一件事有多个入口，补一个漏一片"同源。

修法是**改生成器不改产物**：两条分支统一产出
`head_units: List[(单元标签, 头基原子 index)]`，`head_indices` 与
`leaflet_composition` 都只从它派生，分支专属变量整个消失。
分子路径的标签取 moleculetype 名（`POPC`），**不是**构成残基名（PA/PC/OL）——
用残基名会让 leaflet_composition 又数出 3 倍脂质数。

**为什么能漏出来**：`tests/test_membrane_observable_extractor.py` 原有 22 条测试
**没有一条**传 `composition=`，分子分支覆盖率是零。已补 5 条，其中一条构造
Amber Lipid21 式的模块化残基命名（一个 POPC 分子 = `PC` 头基残基 + `PA` 尾链残基），
断言"按残基必须 fail closed、按分子才对"。

#### MEM-08：§9 的时间轴错 20 倍

修掉 MEM-01 之后门跑通了，报出来的是：

    ValueError: 序列总跨度 0.499 ns 覆盖不了要求的末段窗口 20.000 ns

但那条轨迹是 **10 ns**（5e6 步 × 2 fs），500 帧、每帧 10000 步 = **20 ps**。
`0.499` 从哪来？实测：

    >>> DCDTrajectoryFile(...).read_as_traj(top, n_frames=6).time
    array([0, 1, 2, 3, 4, 5])          # 整数 dtype，间隔 1

**mdtraj 读 DCD 不传播真实步长，`traj.time` 是整数帧号。**
提取器原先只校验"时间数组存在且单调递增"——帧号完全满足，
所以那条 docstring 里写着"不允许用帧号冒充 ns"的守卫，对 DCD（管线唯一写的格式）
是 fail-open。

**时间轴错一个倍数，两道门往相反方向坏**，这也是它难被发觉的原因：

| 判据 | 时间轴小 20 倍时 |
| --- | --- |
| 末段 ≥ 20 ns 窗口 | **过严** —— 要 400 ns 真实时间才够 |
| 预平衡 ≥ 一个脂质横向弛豫时间 | **过松** —— MSD 拟合的 D 被放大 20 倍，弛豫时间缩小 20 倍 |
| 各量的 per-ns 漂移斜率 | **大 20 倍** —— APL 0.2 %/ns 阈值会被虚假违反 |

修法：时间轴由「reporter 保存间隔 × integrator 步长」显式重建。
两个常量 `PRE_EQUILIBRATION_TRAJ_INTERVAL_STEPS = 10000` /
`PRE_EQUILIBRATION_TIMESTEP_PS = 0.002` 落在 `abfe_core`，
**写轨迹的 reporter/integrator 与判门的一侧共用它们**（值不变，故行为逐位兼容；
有 AST 契约测试禁止 `DCDReporter` 再写字面量间隔）。
不传 `frame_interval_ps` 时，遇到**整数 dtype** 的时间数组一律拒绝并说明原因。
实际用了哪一条写进 `diagnostics.time_axis_source`。

修好后同一条 DCD 报 **9.980 ns**（499 × 20 ps），与手算一致。

#### 顺带：质量门收敛为唯一实现 + 离线复判工具（MEM-02）

上面这两个 bug 各花了一次 30 min GPU 才被看到，因为质量门的接线只存在于
`ABFEPipeline._evaluate_membrane_quality_gate_after_equilibration` 里 ——
唯一的验证办法就是**重烧一遍预平衡**。§0.5.7 已经因为
"离线重建与生产路径不一致"白花过好几轮，所以这次不允许再有第二份接线：

- 提取 → 判定 → 落盘收敛到 `abfe_core.run_membrane_quality_gate()`；
- `tools/diagnostics/evaluate_membrane_quality_gate.py` 调**同一个**函数，
  拓扑走 `runabfe.load_native_system(require_bonded_topology=True)`、
  组成走 `classify_system_composition()`，与生产逐项对齐；纯 CPU、不建 Context；
- advisory 下"判不了门"时**观测量也一起落盘**：那些数字是烧了 GPU 才有的，
  判不了门不等于没价值（"跨度不够"时 APL/膜厚的实测值正是延长平衡的唯一依据）。

#### 那条 10 ns 轨迹修好后算出来的 §9 观测量

| 观测量 | 均值 | 末段 5 ns | 斜率 /ns | §13.3 阈值 |
| --- | --- | --- | --- | --- |
| `apl_nm2`（raw） | 0.8074 | 0.8058 | −0.0018 | 漂移 ≤ 0.2 %/ns |
| `bilayer_thickness_nm` | 3.9301 | 3.9321 | +0.0051 | ≤ 0.05 nm / 20 ns |
| `lipid_tail_order_parameter` | 0.1507 | 0.1534 | +0.0015 | （无绝对阈值）|
| `protein_backbone_rmsd_nm` | 0.1047 | 0.1101 | +0.0004 | ≤ 0.30 nm |
| `transmembrane_tilt_deg` | 3.4348 | 3.2355 | −0.1505 | 漂移 ≤ 5° |
| `pocket_rmsd_nm` | 0.0643 | 0.0688 | +0.0009 | ≤ 0.20 nm |
| `ligand_heavy_atom_rmsd_nm` | 0.0362 | 0.0370 | −0.0007 | ≤ 0.25 nm |
| `box_volume_nm3` | 440.30 | 440.25 | −0.0414 | — |

RMSD 三项与倾角都远在阈值内。唯一判不了的是"末段 20 ns" —— 轨迹只有 10 ns，
所以 `n_equil_steps` 改成 **5e7（100 ns，≈5.0 h @ 476 ns/day）**。
⚠️ 配置期那道 `MEMBRANE_MIN_EQUILIBRATION_NS = 100.0` 预检**没挡住** 10 ns 那次，
因为本体系声明 `upstream_equilibration_status = completed_length_unrecorded`
→ 标称时长预检不适用 → 整条跳过（`runabfe.py:4030`）。设计本身没错
（上游时长确实不可考），但代价是 §9 实测门成了唯一的门。

#### Stage 0 attachment 腿的 NaN：只加了体检，没有宣布根因

08-02 17:03，`ibs_engine.py:1425` 出 `Particle coordinate is NaN`。

⚠️ **第一个实际跑的态是 λ=1.0（全强度限制力），不是 λ 列表里的第一个 0.0** ——
`order = list(range(K-1, -1, -1))`，从全强度端往下扫。日志此前只打印 λ 列表
`[0.0, 0.1, 0.35, 1.0]`，很容易被误读成"限制力为零时就炸了"。已补一行明说。

本轮只做了**让下次失败带上下文**（MEM-06）：§0.5.7 给 `pre_equilibrate` 加的
"最小化后 max|F| 门"抽成 `abfe_core.assert_starting_state_is_sane()`（唯一一份），
attachment 腿在第一次 `step` 之前调同一个函数，并额外落盘六个力常数、
**起点实测六个几何量 vs 已提交平衡值**（走 BOR-01 之后唯一正确的
`calc_boresch_from_last_frame`，不写第五份二面角副本）、逐 force group 能量。
报错按"g0 已异常 → 起点坐标坏"和"只有 Boresch 力组异常 → 查几何/力常数"分两条路。
这段**只读**，对既有可溶生产路径数值逐位无影响。

同时按用户指示把 `boresch_source` 从 `auto` 换成 **`simple`**（纯几何涨落估算，
不加载 MACE/e3nn；`auto` 的 `OrbBoreschEstimator` 会加载 MACE 并按 kr 大小给候选加分）。
⚠️ 注意 `boresch_source="traditional"` 是另一回事：那个值只从 `--boresch-anchors`
读外部 JSON、不做估算。

#### MEM-09：被中断的预平衡会被当成"已完成"复用

顺着"5 h 的跑中断了能不能续"这个问题查出来的第三个 fail-open。

`runabfe.equilibrium_is_done()`（`runabfe.py:405`）原先只查
「轨迹存在（>10 KB）+ checkpoint 存在 + 指纹相符」。而
`pre_equilibration_fingerprint.json` 是 `pre_equilibrate()` 在**第一步之前**就写的
（`abfe_pipeline.py:1488`，为的是让被中断的运行下次能认出自己的身份），
它记的 `n_steps` 是**目标**步数，不是已完成步数。

于是：**100 ns 跑到 40 ns 被杀掉 → 下次判成"已完成"→ `pre_equilibrate()` 整段跳过
→ 连它内部的 §9 膜质量门一起跳过**，而 provenance 与指纹都写着 100 ns。
短平衡冒充长平衡，且 `enforce` 根本没机会拦（门在被跳过的那段里面）。

修法：追加**完成判定** —— `checkpoints/pipeline_state.json` 的
`equilibration.status` 必须是 `completed`（该字段只在步进真正结束后才写），
且 `total_steps` ≥ 指纹文件记录的目标步数；缺该状态文件时保守视为未完成。
⚠️ 对既有目录零影响：`output_lrc_fix` 与 `memtest/output_membrane` 实测都有
`status=completed` + `total_steps=5000000`，仍判为已完成（§7.7）。

各阶段的续跑粒度（写进 `memtest/README_MEMTEST.md`）：预平衡每 200 ps
一个 checkpoint；Stage 0 attachment 无窗口级 checkpoint（整段重跑约 7 min）；
Stage 1/2 每个 λ 窗口一份 `openmm.chk` + manifest，manifest 任一项不符整份拒绝。
`memtest/abfe_config.json` 的 `resume` 已设为 `true` —— 不设时
`DCDReporter` 用 `append=False`，已跑的部分直接作废。

#### MEM-10：`superpose` 原地改坐标，污染 6 个观测量（100 ns 那轮的直接肇因）

100 ns 预平衡跑完后 `enforce` 门卡在

    ✗ equilibration_length_ns [at_least_one_lipid_lateral_relaxation_time]
      100.04 vs 阈值 139.362

**那个 139.362 本身是错的。** 根因：`abfe_core.py:3607`

    aligned = traj.superpose(traj, 0, atom_indices=protein_backbone)

`mdtraj.Trajectory.superpose()` **原地修改 `traj.xyz` 并返回 self**。于是这一行之后
所有读 `traj.xyz` 的量都在用"对齐到蛋白骨架"的坐标，而 `midplane` / `upper_z` /
`lower_z` 是在 3502–3503 行、对齐**之前**算的 —— 两者不在同一坐标系。

| 观测量 | 污染后 | 修好后 |
| --- | --- | --- |
| 脂质横向弛豫 τ | **139.36 ns** | 11.57 ns（单参考帧口径）|
| 跨膜倾角漂移 | 0.477 °/window（被压掉） | **1.274** °/window |
| 蛋白横截面 / 校正后 APL、疏水核内水、水层间隙、密度分布 | 坐标系错配 | 一致 |

τ 的 12 倍放大**逐位复现**了门里那个 139.362，所以因果链没有猜测成分。

**而那行 superpose 对它本来要服务的三个 RMSD 毫无作用**：
`md.rmsd(..., atom_indices=X)` 内部会自己在 X 上重新做最优拟合，先对齐不改变返回值
（实测 pocket 0.069400 / 0.069400、ligand 0.050201 / 0.050201）。**它是纯有害的一行。**

修法：主 `traj` 一个字节不动；只为三个 RMSD 建 backbone ∪ pocket ∪ ligand 的
**子集副本**（`atom_slice`，内存约为全轨迹 1/20），在副本上对齐。
契约测试直接钉根因：跑完提取器后 `assert_array_equal(traj.xyz, before)`。

#### MEM-11：弛豫时间尺度不该当硬门（用户质疑"是不是比正常 MD 严"是对的）

估计器原先是**单一参考帧**（每个 lag 只有一个样本）+ 过原点最小二乘拟合**全部** lag
（权重 ∝ lag²，长 lag 主导）。而实测 MSD ~ t^**0.80**（亚扩散），这样拟合必然偏：

| 用到的轨迹长度 | 10 | 20 | 40 | 60 | 80 | 100 ns |
| --- | --- | --- | --- | --- | --- | --- |
| τ | 30.1 | 38.0 | 24.1 | 13.2 | 10.8 | 11.6 ns |

**非单调乱跳** —— 这种量当硬门只会制造假阴性。改成**时间平均 MSD**（多时间原点）+
声明 lag 窗口（5–30 ns）带截距拟合后：D = **0.008664** nm²/ns → τ = **18.467 ns**，
与 POPC 文献 D ≈ 0.008 nm²/ns 给出的 20 ns 吻合。

**判据本身也降级**（硬门 → 诊断），三条依据：

1. §9 原文只要求「**记录**弛豫尺度、**用它论证**预平衡时长」，
   `MEMBRANE_EQUILIBRATION_MIN_RELAXATION_MULTIPLE = 1.0` 这个倍数是本实现自加的
   （旧注释里就写着「§13 未给此倍数」）；
2. 常规膜蛋白平衡的判据是 APL / 膜厚 / 序参量 / RMSD 走平 —— 那些仍是硬门，
   本体系实测余量 2–6 倍；不是要求脂质完成一次横向扩散位移；
3. τ 是方法依赖量（拟合窗口、拥挤度、盒尺寸）。

与本仓库把 ESS `min_occupancy_normalized` 退役为 diagnostics-only 同一先例。
⚠️ **降级不等于不记录**：τ、比值、与文献 D 的对照全部落
`statistics.equilibration_vs_relaxation`；"永不弛豫的膜"会以极大 τ、比值 < 1 被如实
报出（有专门测试）。⚠️ **不得为让某次运行通过而塞回 `checks`**。

顺带：合成 fixture 从"沿固定方向确定性位移"改成**真正的二维随机行走** ——
旧构造只让单参考帧 MSD 等于 4Dt，本身不是扩散运动，只是恰好配合了旧估计器
（所以旧那条 `rel=1e-3` 的"高精度"是 fixture 与估计器互相配合出来的假精度）。

#### MEM-12：APL 的 3% 绝对值门，含蛋白膜不启用

**先认账**：§0.5.9 上一版写"§13.3 的绝对值门可以开了"，但**没有**把
`literature_apl_nm2` 写进 `memtest/membrane_input.json`，那道门一直没跑。
现在实测（100 ns）校正后 APL = 0.5907 vs POPC 0.645，差 **8.42%** —— 真设了也不过。

含蛋白膜差百分之几**不构成"体系有问题"的证据**：annular lipid 被跨膜蛋白减速重排、
蛋白占本体系约 **24%** 横向面积（8.47 / 9.22 nm² vs 盒面 36.3 nm²）、
90 脂小膜片还有有限尺寸效应。改为新增**诊断专用**字段
`pure_lipid_reference_apl_nm2`（名字与"要判"的 `literature_apl_nm2` 刻意不同），
判定层落 `statistics.apl_vs_pure_lipid_literature`（含 `is_gate: false` 与不判的理由）。
这道门应在**无蛋白 POPC slab**（§8.2）上启用 —— 不是删掉，是放到能判的地方。

#### MEM-13：口袋/配体 RMSD 测的是内部构象，不是 pose 漂移

`md.rmsd(aligned, aligned, 0, atom_indices=pocket)` 会在口袋/配体**自身**上再做一次
最优拟合，所以它测的是内部构象变化，而 §9 要的是"配体 pose"。改成"对齐蛋白骨架后
**不重拟合**的位移"：实测 0.0857 / 0.0833 nm（重拟合口径 0.0760 / **0.0493**，
配体差 1.7 倍）。阈值 0.20 / 0.25 **不变**，修正后仍有 2.3× / 3.0× 余量。

#### MEM-14：门写在 `pre_equilibrate()` 内部，重跑即绕过

`_update_stage_status("equilibration","completed")` 与预平衡指纹写在门**之前**
（`abfe_pipeline.py:1745-1763`），门在 `pre_equilibrate()` **内部**。于是
**门失败 → 原样重跑 → `equilibrium_is_done()` 为真 → `pre_equilibrate()` 跳过 →
门也一起被跳过 → 直接进 Stage 0**。`enforce` 的语义被控制流击穿，
而它存在的全部意义就是"门没过不许继续烧 λ 窗口"。

修法：幂等的 `ensure_membrane_quality_gate_passed()` 接在**每个消费预平衡的入口**
（`run_full_pipeline` + 两个 `--only-*` 增量重跑入口），`pre_equilibrate()` 里那次保留
（刚产出就 fail fast）。有源码契约测试覆盖三个入口。

#### 修完之后：同一条 100 ns 轨迹在 `enforce` 下通过

    ✓ apl_nm2 [tail_drift_percent_per_ns]        0.0515 vs 0.2
    ✓ bilayer_thickness_nm [tail_drift/window]   0.0144 vs 0.05
    ✓ transmembrane_tilt_deg [tail_drift/window] 1.274  vs 5
    ✓ protein_backbone_rmsd_nm [tail_mean]       0.1331 vs 0.30
    ✓ pocket_rmsd_nm [tail_mean]                 0.0857 vs 0.20
    ✓ ligand_heavy_atom_rmsd_nm [tail_mean]      0.0833 vs 0.25
    ✓ membrane_periodic_image_contacts           0      vs 0
    ✅ 膜质量门通过（模式 enforce）

**没有重烧 GPU** —— 全部修复都在分析层，用现成轨迹离线复判
（`tools/diagnostics/evaluate_membrane_quality_gate.py`，纯 CPU）。
`MEMBRANE_QUALITY_GATE_PROTOCOL_VERSION` 2 → **3**：v2 的倾角/横截面/弛豫/pose
数字全部作废。

证据：`tests/test_membrane_observable_extractor.py`（分子路径 5 条 + 时间轴 6 条 +
APL 校正 8 条）、`tests/test_membrane_barostat_protocol.py` 的
`test_bad_starting_state_fails_closed_after_minimization`（现同时钉住共用实现与
attachment 腿调用点）、
`test_extractor_does_not_mutate_the_caller_trajectory`（MEM-10 根因契约）、
`test_quality_gate_cannot_be_bypassed_by_rerunning`（MEM-14 三个入口）、
`test_interrupted_pre_equilibration_is_not_mistaken_for_a_finished_one`（4 种情形）、
`test_pre_equilibration_checkpoint_interval_bounds_the_work_lost_on_resume`。

---

### 0.5.10 Stage 0 的 NaN：刚性水被 PBC 修复撕开（2026-08-03 已修，MEM-15）

`memtest` 100 ns 那轮在 attachment 腿第一个 λ 态出 `Particle coordinate is NaN`。
**根因与 Boresch 无关**，而且它对当时所有诊断都是隐形的。

#### 因果链（每一步都有实测数字）

1. `.top` 的 TIP3P 用 settles 定义 ⟹ OpenMM 把 O–H / H–H 变成**约束**。
   实测：`topology.bonds()` 里**涉及水的键数 = 0**（非水键 16635），
   而约束里涉及水的 **28626** 个（= 9542 水 × 3）；
   `HarmonicBondForce` 里涉及水的项 = **0**。
2. `ABFEPipeline.repair_pbc_molecule_integrity` 用 `mdtraj.image_molecules()`，
   它**按 topology 的键**判断"什么算一个分子" ⟹ 每个水原子被当成独立分子 ⟹
   跨边界的 **243 个水**被逐原子回卷，O 与 H 落到**不同周期镜像**。
3. **PME 排除修正失效**：729 个 exception 对跨盒，最远 **13.76 nm**
   （OpenMM 要求排除对比 cutoff 近，本体系 cutoff = 1.0 nm）⟹
   虚假的 **−30.9 MJ/mol** 非键能（−612757.78 vs 健康的 −581848.01）。
4. **约束求解器崩掉**：要在相距 5.9–12.4 nm 的 O/H 之间满足 0.0957 nm 的刚性约束
   ⟹ 不收敛 ⟹ **不到 1 ps** 就 NaN。

#### 为什么绕了很多轮：这个损坏对既有诊断全部隐形

| 诊断 | 坏坐标下的读数 | 为什么看不见 |
| --- | --- | --- |
| `HarmonicBondForce` 能量 | 9525.72（与健康坐标**逐位相同**） | 水根本没有键力项 |
| 最大键长 | 0.1905 nm（正常） | 同上 |
| 角 / 二面角能量 | 逐位相同 | 同上 |
| 最小化后 `max|F|` | 5292 kJ/mol/nm（正常量级） | PME 的排除误差是**平滑长程项** |
| MEM-06 的起点体检 | **通过** | 它只看 PE 与受力 |

也正因为如此，离线**忠实重放**（同起点、同种子、同步数，走完整条 λ 序列 2.4 ns）
**不复现** —— 因为它用的是 rebalance 末帧，那份坐标还没经过这步修复，
729 个排除对最远只有 **0.4331 nm**。两份坐标的差别是"整体平移 + 逐原子回卷"，
键合能因此完全一致，只有非键差了 30909.77 kJ/mol。

#### 沿途被逐一排除的假设（都有实测，记下来免得再走一遍）

| 假设 | 判定依据 |
| --- | --- |
| Boresch 参数是旧的 / MACE 出来的 | `boresch_simple.json` 当天新生成，来自 `GeometricRestraintEstimator`（纯几何涨落，不加载力场/ML） |
| 盒矢量不匹配 | rebalance 末帧 5.9417×5.9417×12.4657 nm，与平衡末帧逐位一致（这条我一度判错过，实测推翻） |
| 锚点共线奇点 | 六锚点全重原子，四个角度 103–128° |
| 限制力太硬 / 几何不对 | λ=1 时 `E_Boresch = 0`（起点正在最小点）；λ=0（力乘 0）也活 100 ps |
| 体系被改动 | Force 清单、粒子数 45354、约束数 38349 与 fresh 加载逐项相同 |

#### 修法

1. **改根因**：`repair_pbc_molecule_integrity` 先把 System 的**约束补成键**，
   再交给 `image_molecules()`。实测补 **28626** 个（正好水的约束数；另 9723 个
   约束对本来就是键）。
   ⚠️ 用约束而不是"从 `.top` 的 `[ molecules ]` 取区间"：约束就在 System 里，
   任何输入来源都有、不依赖 `.top` 是否可得，而它补上的正好是缺的那些边。
2. **加守卫**：`abfe_core.assert_starting_state_is_sane` 新增**镜像一致性**检查
   （排除对与约束对是否都在同一镜像内，上限取 cutoff 或半个最短盒边）。
   这不是冗余：上表说明力检查**构造性地**看不见这类损坏。
3. **补现场证据**（这条腿此前一帧轨迹、一行监控都不写）：
   * `attachment/stage0_attachment_start.npz` —— 起点坐标 + 盒矢量，
     **可离线确定性复现**（只存 SHA256 时每查一步都得重跑一次生产）；
   * `attachment/stage0_attachment_inputs.json` —— Force 清单 / 粒子数 / 约束数 /
     盒矢量 / 锚点 raw vs minimum-image 距离 / λ 序列 / 种子 / 各段步数 / 限制力全参数；
   * `attachment/stage0_attachment_monitor.csv` —— 头 1000 步 **50 fs** 一行、
     之后 1 ps 一行的 `PE / T / max|F| / 受力最大原子 / E_Boresch`。

#### 验证（两个方向都实测过）

* 拿生产那份坏坐标喂守卫：力检查照常打印正常的 `max|F| = 5292`，
  **紧接着** raise「729 个 nonbonded_exceptions 对跨了周期镜像（最远 13.760 nm，
  上限 1.000 nm）」；
* 约束补键后重修同一份坐标：排除对最远 **0.433 nm**、约束对 **0.151 nm**，
  守卫放行，且 `PE = −508657` 回到 rebalance 的健康值 —— 那 31 MJ/mol 虚假能量消失。

证据：`tests/test_membrane_barostat_protocol.py::
test_torn_rigid_water_is_caught_before_dynamics`（合成两个刚性水，
`topology.getNumBonds() == 0` 与真实拓扑同构；把一个水的 H2 逐原子回卷 → 守卫 raise）
与 `test_pbc_repair_promotes_constraints_to_bonds_for_grouping`
（源码契约：约束补键必须在 `traj.image_molecules(` 之前）。

#### ✅ 真机端到端确认（2026-08-03 23:38–23:43，`memtest/output_membrane_100ns`）

MEM-15 不再只有合成测试兜着 —— **真实 100 ns 膜体系的 attachment 腿整条跑通、无 NaN**：

* `pipeline.log`：23:38:40 PBC 修复报 `🔗 已把 28626 个约束补成键用于分子归组`
  （= 9542 水 × 3，与上面因果链第 1 步的实测数字逐位一致）；
  23:38:44 起点 `PE = −508657 kJ/mol`、`max|F| = 5292 kJ/mol/nm (idx=3353 GLU208/C)`；
  `✓ 镜像一致性: 排除对 120895 个（最远 0.433 nm）、约束对 38349 个（最远 0.151 nm）`。
* `attachment/stage0_attachment_monitor.csv`：1409 行覆盖 4 个 λ 态，
  **`nan`/`inf` 出现 0 次**；温度 304–310 K，势能 −508657 → −509385 kJ/mol，
  `max_force` 全程 ~5.2×10³ kJ/mol/nm 无发散。旧 bug 是「**不到 1 ps** 就 NaN」，
  这次 4 态 × 300000 步全过。
* 结果：`ΔG(A′→A) = 6.0880 ± 0.1062 kJ/mol = 1.4551 ± 0.0254 kcal/mol`
  （BAR 主口径；TI 6.2551、MBAR 诊断 5.8793；前后半程漂移 −0.0171 kJ/mol，容差 0.6371）。
  产物：`attachment/attachment_meta.json`、`attachment_u_kn.npy`。
* 23:48:20 已进入 Stage 1 去电荷（12 状态线性 λ 路径）。

⚠️ **这是 §0.5.7 那个根因的第二次**：§0.5.7 是 mmCIF 往返丢掉**非标准残基的键**，
导致 PBC 修复撕开脂质；这次是刚性水的键**从来就不在 topology 里**（只在约束里），
导致 PBC 修复撕开水。共同教训：**"按 topology 的键归组分子"这个前提本身要验证**，
而不是假定拓扑里有全部连通性。

---

### 0.5.11 pymbar 4 的 JAX 后端预分配整卡 75% 显存 → REMD 静默退 CPU（2026-08-04 已修）

**这条挡住了膜体系的 Stage 1，而且伪装成"显存不够/体系太大"。**

pymbar 4 的后端是 JAX；JAX 默认 `XLA_PYTHON_CLIENT_PREALLOCATE=true` +
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`，**一碰 GPU 就预分配整卡的 75%**。
attachment 腿末尾要用 pymbar 解 BAR/MBAR —— 于是它把卡吃掉 75%，
紧接着 Stage 1 要建 12 个 replica Context 就建不出来，静默回退 CPU
（慢约两个数量级：第 0 轮交换 29 分钟，500 轮约 10 天，表现得像卡死）。

日志逐行对上（`memtest/output_membrane_100ns/pipeline.log`）：

```
06:09:09  📊 [显存] 预平衡结束（Context 已清理）: used=  269 free=15574 total=16303 MiB
06:15:21  WARNING | ******* JAX 64-bit mode is now on! *******     ← pymbar 解 MBAR
06:15:22  📊 [显存] Stage 0 attachment 结束:        used=12197 free= 3646
06:15:22  📊 [显存] Stage 1 建 replica 之前:        used=12197 free= 3646
          → 建满 11 个、第 12 个 OpenMMException: No compatible CUDA device is available
```

**12197 / 16303 = 74.8%**，就是那个 0.75。12 个 Context 需要 12 × 317 = 3804 MiB，
只剩 3646 MiB —— **差 158 MiB**。

**修法**：`abfe_core.py` 顶部、**在任何 `import pymbar` 之前**
`os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")`
（JAX 只在初始化时读一次环境变量，设晚了等于没设）。只关预分配、**不**把 MBAR
挪到 CPU：JAX 仍在 GPU 上按需申请，数值路径不变，不动已落盘基线。
用 `setdefault` 所以外部可覆盖（更彻底可外部 `export JAX_PLATFORMS=cpu`，
但那会改变 MBAR 的执行设备，属于需单独验证的改动）。

**真机前后对比**（同一台机、同一体系）：

| 时间点 | 修之前 | 修之后 |
| --- | --- | --- |
| 预平衡结束 | used=269 | used=269 |
| **Stage 0 attachment 结束** | **used=12197** | **used=445** |
| Stage 1 建 replica 之前 | used=12197, free=3646 → **退 CPU** | used=445, **free=15398** |
| Stage 1 | 建到第 11 个死 | **06:34:12 → 06:53:26 跑完，全程 CUDA** |

**为什么绕了很多轮（四个假设全被否掉，别再猜一遍）**：
1. 进程内 Context 泄漏 —— 全新进程 resume 照样失败（每个进程都会重新预分配）；
2. 体系太大 / PME 网格 —— 更**大**的可溶体系（73536 原子）12 个 Context 成功过，
   膜只有 45354；
3. OOM（12 个 Context 太多）—— 离线探针
   `memtest/probe_remd_context_capacity.py` **12/12 全建成**、每个 ≈ 317 MiB、
   共 3.8 GB，`--replay-preoptimizer` 也一样。探针成功的原因正是**它不调 pymbar**，
   JAX 从未初始化；
4. fork / 并行 stage —— `--parallel-stages` 2026-07-27 已整体移除，主进程串行。

**教训**：**GPU 显存不是只有 OpenMM 在用。** 分析层（pymbar/JAX）、ML 势
（torch/MACE）都会抓 VRAM，而 JAX 是**按整卡比例预分配**、不是按需。
查这类问题第一件事是打点"开跑前 used 是多少" —— `vram_before` 这一个数
比任何体系大小的推理都值钱。定位工具已留在仓库里：
`REMDManager._build_replicas` 逐 Context 记显存并把失败证据**当场**落盘
`<stage_dir>/remd_platform_fallback.json`（不能靠日志：`logger` 没有 FileHandler，
`pipeline.log` 是 `ABFEPipeline._log` 另一条通路写的，所以回退告警在归档日志里一行都没有）；
`ABFEPipeline._log_vram()` 在预平衡结束 / attachment 结束 / Stage 1 建 replica 之前三处打点。

顺带修掉两个**真实但非元凶**的泄漏（各约一个 Context 的量）：
* `ibs_engine.run_boresch_attachment_leg` 整段原来**没有任何 teardown**
  （唯一的 `app.Simulation` 在 `ibs_engine.py:1494`）→ 已加显式释放。
  ⚠️ 实测是**空操作**（`used 445 → 445`）：引用计数本来就回收了。写显式仍然对，
  但它不是元凶，别把它当成本条的解释。
* λ 预优化原来只 `del context, integrator, probe_sys`，而 `optimizer`
  （`DualLambdaPreOptimizer(probe_sys, context, ...)`）仍持有 `context`
  → 改为 `del optimizer, ...` + `gc.collect()`。实测**确实是漏**：`used 806 → 445`，
  释放了 361 MiB。

证据：`tests/test_import_time_side_effects.py` 两条新测试（源码顺序必须早于
`import pymbar`；干净子进程 import 后标志为 `false`）。全套 979 passed / 2 skipped / 0 failed。

---

### 0.5.12 单次 σ 不代表 attachment 腿的跨运行散布（2026-08-04 观测；已知统计限制，不阻塞开发）

同一体系、同一协议下 attachment 腿（`ΔG(A′→A)`）五次独立运行（kJ/mol）：

```
5.7726 ± 0.0969
5.8623 ± 0.0971
6.0786 ± 0.0976
6.0880 ± 0.1062
7.5216 ± 0.1141   ← 显著离群
```

| 口径 | 数值 | 相对单次 σ ≈ 0.10 |
| --- | --- | --- |
| 五次样本标准差 | **0.716** | ≈ 7× |
| 五次极差 | 1.749 | (17× —— **极差不是标准差，这个比值没有校准意义**) |
| 去掉 7.5216 后四次样本标准差 | **0.158** | ≈ **1.6×** |

⚠️ **本节早前写成"跑间散布比 σ 大约 17 倍"是错的**，那是拿**极差**除以 σ 得到的。
正确的表述只有一句：**单次 σ 不能代表跨运行散布**。
不能据此宣称"σ 系统性低估 17 倍"——去掉离群点后只有 1.6×，属正常范围；
`7.5216` 是**一次显著离群运行**，应当先当异常个案查清，而不是当作 σ 口径的证据。

**定位到正确的优先级**：这属于 P1-19 那类"已知统计限制"，
按 §17 的排序依赖，P1-19 阻止的是**进生产**，**并没有**规定它阻止
B3/B4/B5 的方法开发。所以：
* **现在不处理**，不要让它把工程闭环拽住；
* 真正做**三重复 / benchmark / 生产验收**时再按 §13.4 的口径收口；
* 届时先查清那次离群（λ 路径不同？起点坐标不同？前后半程漂移超容差？）。

**不要**用"多跑几次取平均"当作 σ 口径的修复 —— 那是两件不同的事。

---

### 0.5.13 第一次端到端跑通，暴露溶剂腿两处缺陷（2026-08-04）

`memtest/output_membrane_100ns` 08-04 08:28 跑完了**整条主链**：
预平衡 100 ns → Stage 0 attachment → Stage 1 去电荷 → Stage 2 去 VDW → 溶剂腿 → 汇总。
§17.0 第 ① 步（工程 smoke test）的目标达到：主链 / 缓存 / REMD / λ 预优化 /
溶剂腿构建 / 结果汇总全部走通，没有中途挂掉。耗时与占用见 §15（已据此关闭）。

```
复合物腿 ΔG_cplx  = 175.57 ± 1.50 kJ/mol     ← 用户确认没问题
溶剂腿   ΔG_solv  = 272.93 ± 1.46 kJ/mol     ← 有缺陷
ΔG_bind          = +97.36 ± 2.09 kJ/mol = +23.27 kcal/mol   ← **不可用**
```

⚠️ **这个 ΔG_bind 不得引用。** 配体是**中性 Atenolol**，与可溶生产体系是同一个分子
（`memtest/topol.top` 的 moleculetype `Atenolol-rank11`，41 原子，Σq = 0.000000），
所以两次运行的**溶剂腿必须可比** —— 但它们不可比：

| 溶剂腿分项（同一配体、同为纯水盒） | 可溶生产 `output_lrc_fix` | 膜运行 | 差 |
| --- | --- | --- | --- |
| decharging | 62.80 | **191.05** | **+128.25** |
| vanishing | 96.96 | 83.83 | −13.14 |
| 合计 | 157.84 | 272.93 | +115.09 |

#### 缺陷一：B6 把纯水腿**合法**的 bulk-water LRC 一起关掉了（已修）

判据当时是一个没有环境维度的全局布尔：

```python
apply_lrc = (dispersion_protocol == "legacy_uniform_density_lrc")
```

于是"膜口袋里局域密度不均匀"这个**对复合物腿正确**的理由，被原样套到了同一次运行的
**纯水溶剂腿**上 —— 那条腿的 `final_results.json` 里逐字写着
`disabled_by_membrane_forcefield_protocol: …配体所在口袋的局域密度既不是水也不是体相脂质`，
而那里配体周围恰恰**就是**均匀体相水。文档自己是对的（§1.3 只说膜口袋不成立），
是实现把两件事合并成了一个布尔。

**修法（2026-08-04，B6-FIX）：把"目标"与"每条腿的实现"拆开。**

```text
dispersion_protocol  = 目标：所选力场原始参数化时的色散条件
   ↓  每条腿在自己的环境里怎么达成
实现 = f(目标, 该腿配体所处环境)
```

| 目标 | 该腿环境 | 炼金 ligand–env LRC | target_met |
| --- | --- | --- | --- |
| `legacy_uniform_density_lrc` | soluble | 开 | 是（改动前唯一行为，逐位不变）|
| `ff_native_isotropic_lrc` | soluble（含膜运行的溶剂腿）| **开** | 是 |
| `ff_native_isotropic_lrc` | membrane | 关 | **否**（需路线 C，未实现）|
| `ff_native_force_switch_no_lrc` | 任意 | 关 | 是（力场本身不加 LRC）|

唯一实现 `abfe_core.resolve_leg_dispersion_implementation()`；
`ibs_lj_tail_lrc_is_applicable(potential_type, dispersion_protocol, environment_type)`
只是它的布尔投影，生产者与报告者仍共用同一真相。
环境维度取 `system_type` —— 那是**用户在输入文件里显式声明**的值，
B1 的规矩是「不许按残基名猜 system_type」，不是「不许用声明出来的值分派」，
所以按它自动切换实现是合法且可审计的。溶剂腿天然是 soluble：`runabfe` 构造溶剂腿
pipeline 时刻意不传 `environment_type`（B1 的接线契约测试钉着这一点）。

⚠️ 三点要说清楚，免得期待错方向：
1. 这条缺陷解释的是 **vanishing 的 −13.1**（符号也对：关掉吸引尾项 ⟹ 拔出配体更便宜），
   **不解释 decharging 的 +128**（LRC 根本不作用于去电荷腿：
   `needs_traditional_lrc = not is_pme_coulomb_leg`）；
2. 修回去之后溶剂腿会**变大**到 ≈286 kJ/mol，ΔG_bind 更正。修它是因为它本身错，
   **不是因为它能救那个数**；
3. 膜复合物腿的行为不变（仍然不加），但现在如实记为 `target_met=false` ——
   力场是开着各向同性色散修正拟合的，而这条腿的炼金 ligand–env 项是**截断**的，
   真正达成目标要 §1.3 路线 C（未实现）。把"关掉了"记成"处理好了"是不诚实的。

缓存影响：`_stage_protocol_key` 只在决定为 True 时写
`alchemical_uniform_density_lrc`，所以**行为真的变了的那一类**（膜运行的溶剂腿）
旧缓存被正确拒绝，膜复合物腿与可溶生产基线的缓存都不受影响（§7.7）。

#### 缺陷二：溶剂腿配体**丢了全部 71 个键角项**（P0-13，2026-08-04 已修）

decharging 62.80 → 191.05 只是最后一环。逐项排除（估计量、PME 自能/α、配体电荷、
L-L 冻结、盒组成、几何、u_kn 结构全部否掉，见 `docs/TODO.md` P0-12 的排除表）之后，
根因在 `runabfe.generate_ligand_xml_from_top`：

```python
angle_force = next((f for f in extracted_system.getForces()
                    if isinstance(f, openmm.HarmonicAngleForce)), None)   # 取"第一个"
```

而膜体系的 System 里有**两个** `HarmonicAngleForce`：

```
force[2]  31401 个角，配体 0 个    ← next() 抓到的是这个
force[4]     71 个角，配体 71 个   ← 配体的角全在这里
可溶体系只有 1 个角力（配体那 71 个混在里面）⟹ 侥幸一直没踩到
```

于是 `ligand_only.xml` 的 `<HarmonicAngleForce>` 段是**空的**，溶剂腿配体**没有任何
键角项** —— 分子是软的。因果链每一环都有实测：

```
无键角 ⟹ 预平衡里 0.996 → 0.660 nm 塌缩，12 个 replica 再没恢复（σ=0.005 nm）
      ⟹ 极性基团聚拢，配体–水静电耦合强 3 倍（⟨U⟩ −569 ± 90 vs −190 ± 34 kJ/mol）
      ⟹ 去电荷 62.80 → 191.05 kJ/mol
      ⟹ ΔG_bind = +23.27 kcal/mol
```

⚠️ **所以这不是"采样不足"也不是"两相构象偏好不同"这类物理问题**，是哈密顿量本身错了。
先前一度往"溶剂腿构象采样没收敛、需要双起点验证"的方向想，**那个方向被否掉了** ——
参数修对之后没有理由再塌缩（P0-12c 因此撤销，不做）。

修法两层：
* **聚合而不是取第一个**：bond/angle/torsion 三类都遍历**所有**同类型力
  （NonbondedForce 多于一个时直接报错让人收口拓扑，不猜）；
* **写完就对账**：写出的成键项数必须与源体系里配体的项数逐项相等，且"多原子配体
  0 个键角"直接 fail closed。这一步就是那个"7 小时之后才发现"的事故本该在 0.1 秒内
  被拦住的地方。

实测修复后同一份 memtest 拓扑：`<Angle>` **0 → 71**，对账 bond=41 angle=71 torsion=104 通过。
证据：`tests/test_ligand_xml_extraction.py`（5 条，含真体系回归 + "多于一个角力"这个
前提本身也被钉住 + 合成 floppy 拓扑触发 fail-closed）。

#### 顺带（P0-12a/b）：这次事故的**检测层**也补上了

即使根因修了，这类"某条腿的配体构象跑到另一族去"的失效模式也必须能当场被发现：
* 两条腿都逐 λ 记录 Rg / 重原子最大内距 / 内部极性接触（§3.0 末条要求的诊断，
  此前从未实现），另记逐 replica 均值 —— "12 个 replica 挤在同一个窄 basin"
  只有逐 replica 才看得见；
* **跨腿构象一致性门**接在 `combine_binding_free_energy`（循环闭合唯一实现）：
  两腿 [p5, p95] 不相交即不许汇总 ΔG_bind。实测膜运行 overlap = −0.631 nm（拦下）、
  可溶基线 +0.053 nm（放行），同一条判据分开，无可调旋钮；
* 溶剂腿缓存身份加入配体**起始构象**指纹（内部距离矩阵，刚体运动不敏感），
  `SOLVENT_CACHE_PROTOCOL_VERSION` 4 → 5 —— 此前两次运行拿到完全不同的构象却都判
  "缓存有效"。

落盘字段（重跑后要看的就是这几个，`final_results.json`）：
`ligand_conformer_diagnostics.observables.{max_internal_heavy_distance_nm,
radius_of_gyration_nm, internal_polar_contact_count}`、
`ligand_conformer_diagnostics.per_replica_mean_max_internal_heavy_distance_nm`、
`ligand_conformer_diagnostics.per_replica_spread_nm`、
`ligand_conformer_cross_leg.{evaluated,passed,overlap_nm,reason}`。

⚠️ **两件看着像 bug 但不是的**（查过，别再查）：
1. 那轮日志里的 `[膜质量门 · advisory]` —— provenance 记
   `config.membrane_quality_gate = advisory` 且 `config file = None`，说明那次是
   **命令行**跑的 advisory，不是配置文件被忽略；`memtest/abfe_config.json` 一直是
   `enforce`。重跑时别再加那个 flag。
2. `ligand_only.xml` 不在 `resolve_ligand_ffxml` 的候选名单里 ⟹ 每次都用当前生成器
   重写，**不存在复用旧 angle-less XML 的陷阱**。

重跑命令、会作废什么（预平衡 5 h 保留 / 之后 2.1 h 重做）、跑完看哪三个数：
见 `docs/handoffs/MEMBRANE_SOLVENT_LEG_P013_HANDOFF.md` §6。

---

### 0.5.14 RESUME-FP-01：三处 resume 协议指纹误把"坐标数组"当强判据（2026-08-05 已修）

在中性 Atenolol 那轮跑的真实日志里当场抓到：resume 之后
`Boresch attachment stage0 缓存...不匹配，重新运行` +
`已有 final_results.json 协议指纹不匹配`，即便 Boresch 平衡值自己的 σ 偏差核对
已经通过。根因与 MEM-00c 的坐标漂移是**同一个模式，换了个消费者**：

`abfe_pipeline.py` 的 `stage0_protocol_key` / `_stage_protocol_key` /
`_preopt_protocol_key` / `_build_top_level_protocol_key` 都把
`coordinates_nm_sha256`（`_positions_hash(self.positions)`）编进了协议指纹，而
`self.positions` 在两条路径下天然不同：

* 本次调用刚完成预平衡：`run_full_pipeline` 第 1 节额外叠了一次 2000 步最小化
  （"消除加载坐标的残余应力"）；
* resume 时直接读 `pre_equilibration.dcd` 末帧：不叠这次最小化。

同一段已完成、内容完全相同的物理轨迹，两条路径算出的坐标数组不同、哈希不同——
**跟协议/System/Boresch 参数是否真的变了完全无关**。`_preopt_protocol_key` 尤其
冤枉：它的 docstring 明确说这份指纹是特意收窄过的，为的就是不让"跟预优化无关的
改动"连带让这份要跑好几个小时的缓存失效，但收窄时漏看了这一项。

修法：这四处删掉 `coordinates_nm_sha256`。"坐标/构型是否还对得上"已经由
`boresch_equilibrium_committed.json` + `_assert_committed_boresch_still_matches_pose()`
（σ 偏差核对）与预平衡自己的 `pre_equilibration_fingerprint.json` +
`status=completed` 门分别守住，不需要再叠一份对实现细节（有没有跑那额外 2000 步）
比坐标数组的强判据。

⚠️ **每 λ 窗口级的 GPU 采样 checkpoint 不受影响**：`ibs_engine._build_main_window_checkpoint_manifest`
本来就不含这个坐标哈希（键在 `win_sys_xml`/`lambdas_coul`/`lambdas_vdw`/协议版本/
平台），已实测确认——这次修复只影响"这条腿/这个 stage 是否已整体完成、可以直接
跳过重算"这几处**粗粒度快捷路径**，不会让任何已完成的 λ 窗口采样重烧 GPU。
真实证据：那次 resume 里 `stage0_attachment.json`/`preopt_dual_decharging.json`
被重新写过，但 `stage1_decharging.json`/`production_window/` 时间戳完全没动。

完整记录见 `docs/TODO.md` 的 RESUME-FP-01 条目。回归：
`tests/test_dispersion_protocol_is_honored.py`（15 passed）+ 全套 offline 回归。

⚠️ **同一模式还存在于 `TraditionalABFEPipeline.run_leg()` 的 `initial_positions_sha256`
（`abfe_pipeline.py:7965`）**，但那是完全不同的"传统 REMD"路径，本轮跑的是 IBS
双 λ 管线，没有证据这条也受影响，**没有动它**——留给真正用到 traditional 模式
resume 时再验证，不要凭"看起来像"就顺手改。

---

### 0.5.16 窗口预热的排序 bug：EM 在 Boresch 限制力生效之前（2026-08-05 已修）

RESUME-FP-01 修完、`stage1_decharging.json` 缓存也正确复用后，跑到窗口 5（vdw/
vanishing，态 [19:23]）时在 Boresch 安全爬坡的 `sim.minimizeEnergy()` 内部直接抛
`openmm.OpenMMException: Particle coordinate is NaN`——异常发生在 C++ 层内部
迭代中途，`ibs_engine.py:9574`（当时行号）那段爬坡自己的能量/力检查全部是
minimize **返回之后**才做，根本挡不住这一种失败模式，进程直接崩溃，resume 后
从头再来，同一个窗口反复重演。

**根因**：`run_all_windows()` 里 `lambda_boresch_scale` 这个 System 级全局参数
默认值是 `0.0`，而"阶段1 能量最小化"（`sim.minimizeEnergy(maxIterations=20000)`）
在**没有任何地方显式把它设成非零值**的情况下就跑了——也就是说全新窗口的最小化
是在 Boresch 限制力完全关闭的状态下做的。这在深度解耦（vdW/softcore 几乎关闭）
的窗口尤其致命：配体没有真实的 vdW/Coulomb 力把它固定在结合位点附近，自由最小化
可以让它漂出委托好的 Boresch 平衡几何——实测 committed `r0=0.448nm` 漂到最小化后
测得 `0.650nm`，约 0.2nm 的缺口。随后"阶段3"那段 16 级自定义阶梯（0.01→1.0）
存在的**唯一理由**就是把这 0.2nm 的缺口，靠一根逐步加强、最终 kr=2000 kJ/mol/nm²
的弹簧，一点点拽回来——窗口 5 就是在拽的某个中间强度（scale=0.05）时，
`minimizeEnergy()` 内部一步踩过了头，直接把某个原子送进另一个原子的斥芯。

这是**排序 bug**，不是"需要更精细的爬坡/更多诊断"：限制力应该在最小化**之前**
就已经在生产强度，让最小化本身收敛到一个天然满足限制力平衡的构型，而不是先自由
最小化、再事后用一根硬弹簧把结果拽回限制力允许的范围。任何正常的 Boresch-restrained
FEP/REMD 协议都不会需要这种规模的分级爬坡——爬坡的复杂度本身就是这个排序 bug
存在的证据，不是需要被加固的东西。

**修法**（`ibs_engine.py::run_all_windows`）：

- `sim.context.setPositions(positions)` 之后、阶段1 最小化之前，新增
  `if _has_valid_boresch_restraint(self.boresch): sim.context.setParameter("lambda_boresch_scale", 1.0)`。
- 阶段1 的 20000 步最小化现在就在生产哈密顿量（Boresch 已在 scale=1.0）下进行。
- 阶段2 里那行把 scale 临时压到 0.01 的 `setParameter` 连同它的 print **注释掉**
  （不是删除）——不再需要，全程保持 1.0。
- 原"阶段3 Boresch 安全爬坡"整段（16 级自定义阶梯 + 能量/力检查 + 失败处理）
  **整段注释掉**（不是删除，留作历史参考）——它存在的前提（EM 后需要拽回限制力）
  已经不成立。
- 阶段2 的 dt 测试步进、约束死锁检测**原样保留**，只是现在全程在 Boresch 全强度
  下运行，不再需要单独处理"限制力还没爬到位"这个中间态。
- checkpoint-restore 路径本来就在 restore 后显式把 scale 重设为 1.0
  （防御性重设，未依赖 loadCheckpoint 是否保存了 global parameter），未改动——
  现在两条路径（fresh window 与 checkpoint-restore）殊途同归，都在 scale=1.0
  下进入采样。

**不需要的东西（用户明确排除）**：不加自适应爬坡、不加回滚/重试状态机、不新增
checkpoint 协议版本。已完成的生产窗口（`production_window/`）物理上全部是在
scale=1.0 下采样完成的，这个修复只改变"全新窗口如何进入采样"，不影响任何已
完成窗口的有效性，也不需要为此升级 `MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION`。

回归：`ibs_engine.py`/`abfe_pipeline.py` `py_compile` 通过，全套 offline 回归
（无新增失败，见 `docs/TODO.md` 的这条记录）。⚠️ **未在真实 GPU 上验证窗口 5
现在能否正常收敛**——这是代码层面的排序修复，物理效果需要用户在自己的计算节点上
实测确认。

---

### 0.5.2 已存在的、进膜之前必须先收口的非键协议不一致

- [ ] **MEM-00h：ligand–environment 与 environment–environment 的 LJ cutoff 目前就不一致。**
  实测常数：
  - 复合物腿基础 `NonbondedForce`：PME，cutoff **1.0 nm**（`runabfe.py:1143-1148`）；
  - 溶剂腿：`SOLVENT_NONBONDED_CUTOFF_NM = 1.0`（`abfe_core.py:3249`）；
  - 炼金 softcore CV：`SOFTCORE_CUTOFF_NM = 1.2`（`ibs_engine.py:59`）；
  - LJ 尾修积分下限：`LJ_TAIL_LRC_R_CUTOFF_NM = 1.2`（`ibs_engine.py:2178`）；
  - DEXP：`DEXP_VDW_CUTOFF_NM = 0.70`、`GAUSS_COUL_CUTOFF_NM = 0.70`（`abfe_core.py:53,56`）。

  §1.3 路线 A 要求"普通 NonbondedForce 与炼金 softcore 力使用同一非键协议"——
  **在当前可溶体系里这一条就已经不满足**。膜体系里 APL/膜厚对脂质 vdW cutoff 直接敏感，
  这个不一致会被放大。**这是可溶体系的存量问题，必须在膜工作之前单独立项收口，
  不要塞进膜 PR 里一起改。**

  **2026-08-05 复核（行号已随代码增长更新，值未变）**：
  - 复合物腿基础 `NonbondedForce`：`runabfe.py:1410`，硬编码 `1.0 nm`；
  - 溶剂腿：`SOLVENT_NONBONDED_CUTOFF_NM = 1.0`（`abfe_core.py:8868`）；
  - 炼金 softcore CV：`SOFTCORE_CUTOFF_NM = 1.2`，`SOFTCORE_SWITCH_NM = 1.0`
    （`ibs_engine.py:83-84`）；
  - LJ 尾修积分：`LJ_TAIL_LRC_R_SWITCH_NM = 1.0`、`LJ_TAIL_LRC_R_CUTOFF_NM = 1.2`
    （`ibs_engine.py:3158-3159`），与上面的 softcore CV 精确对应（有交叉测试钉住）；
  - `COION_LIGAND_MIN_IMAGE_RUNTIME_NM = 1.2`（`abfe_core.py:1672`）是**派生量**，
    跟随 `SOFTCORE_CUTOFF_NM`，不是第三个独立决策。

  实际是两组自洽的值在打架，不是"三处各写各的"：{基础力 1.0} vs
  {softcore CV / LJ 尾修下限 / co-ion 安全边距 1.2}，且 `SOFTCORE_SWITCH_NM`
  已经和基础力的 1.0 对齐，只有 cutoff 尾巴多出的 0.2 nm switching 区间对不上。

  **收口方向建议（尚未实施，只是分析结论，立项时才真正改代码）**：收敛到 **1.0 nm**，
  不是 1.2 nm。理由：1.0 nm 是与 Amber 系力场原始参数化匹配的截断（§1.3 里 Amber
  Lipid21/17 就是按 1.0 nm cutoff + LRC 拟合的），把基础力拉长到 1.2 会偏离已验证
  的力场截断距离；而 1.2 nm 这组值目前找不到独立的物理依据。当前的不一致会导致
  λ=1（完全耦合）端点与"关掉炼金描述、直接用普通力场"的真实系统不严格等价——1.0–1.2 nm
  这个壳层里炼金力仍有非零 switched 相互作用，基础力却已经硬截断为零。
  收口 PR 至少要覆盖：`SOFTCORE_CUTOFF_NM`→1.0、`LJ_TAIL_LRC_R_CUTOFF_NM`→1.0、
  派生的 `COION_LIGAND_MIN_IMAGE_RUNTIME_NM`→1.0（连带更新交叉检查测试的断言值）、
  以及 §1.3 路线 A 最后一条要求的 GROMACS↔OpenMM 单点能量对照。
  ⚠️ 这条只要改了会影响哈密顿量/坐标输入的常量，就会作废已完成的预平衡/生产窗口——
  立项执行前先盘一下哪些缓存结果会被作废。
- [x] **MEM-00i：没有任何 `MonteCarloMembraneBarostat`。** ✅ 已实现（2026-07-30，B1）
  `abfe_pipeline.py:1378-1382` 原来无条件加 `MonteCarloBarostat(pressure, temperature, 25)`，
  只检测同类型是否已存在。
  **实施补记**：原判据 `isinstance(f, openmm.MonteCarloBarostat)` 还漏检——实测
  OpenMM 里 `MonteCarloBarostat` / `MonteCarloAnisotropicBarostat` /
  `MonteCarloMembraneBarostat` 三者**都直接继承 `Force`，互不为子类**
  （`openmm.py:16777/17012/17406`），所以输入 System 已带膜 barostat 时旧逻辑
  检测不到，会再叠一个各向同性的，两个同时做体积移动且不报错。
  现改为按类名检测三种（`abfe_core.detect_barostats`）+
  `ensure_barostat_for_protocol` 三分支（复用 / 添加 / fail closed）。
  证据：`tests/test_membrane_barostat_protocol.py`。
- [ ] **MEM-00m：`pre_equilibrate` 的 integrator 步长与 DCD 保存间隔是
  §9 时间轴的唯一来源，已收敛为共用常量（2026-08-02，MEM-08）。**
  `PRE_EQUILIBRATION_TIMESTEP_PS = 0.002` 与
  `PRE_EQUILIBRATION_TRAJ_INTERVAL_STEPS = 10000` 落在 `abfe_core`，
  写轨迹的 reporter/integrator 与判门的一侧引用同一份（值不变，逐位兼容）。
  ⚠️ 若 MEM-00j 决定给膜体系上 HMR + 4 fs，**必须同时**确认这两个常量与
  §9 时间轴的关系（帧间隔会从 20 ps 变成 40 ps），否则所有 per-ns 判据会静默错 2 倍。
- [ ] **MEM-00j：没有 HMR，生产 dt = 2 fs + `constraints=HBonds`**
  （`ibs_engine.py:8388`，`abfe_core.py:3564` 的 `timestep_ps` 默认 0.002）。
  膜体系是否用 HMR 上到 4 fs 是 Phase A 的显式决策，不是默认继承。

### 0.5.3 力场族兼容性（已由现有输入锁死一半）

- [ ] **MEM-00k：现有 `topol.top:21` 用的是 `amber14sb_OL15_fs1.ff` + TIP3P
  （`topol.top:43275`），配体是 RESP/GAFF。**
  因此脂质力场应当选 **Amber 系（Lipid21 / Lipid17 / Slipids）**；选 CHARMM36 脂质
  就是跨力场族混用，必须显式论证并单独验证，不能默许。
- [ ] **MEM-00l：这条决策会反过来决定 §1.3 的 LJ 路线**（见 §1.3 开头新增的修正说明）。

---

## 1. 先冻结科学协议

### 1.1 目标体系

**✅ 已定（2026-07-29）：首个体系 = SERT（血清素转运体），配体默认净电荷 +1，
结合位点要求"非深埋"（水可及）。备选：GPCR。**

因此本轮走 `charge_treatment = co_alchemical_charge_transfer`，
co-ion 为 `0 → +1` 的阳离子型（§2.2）。

SERT 特有的坑，**必须在 Phase A 就处理，否则会静默选错 co-ion**：

- [ ] **SERT 的 S1 位点附近有结构性结合的 Na⁺ 和 Cl⁻。**
  现有候选筛选 `_select_bulk_water_counterion`（`ibs_engine.py:766`）的
  `ion_names = {"CL","CLA","NA","SOD","K","POT","MG","CA"}` 只按残基名和电荷筛，
  **完全可能把结合位点里的结构 Na⁺ 选成 co-ion**——这正是 §2.3
  "不与关键离子位点或通道电场耦合" 禁止的情形。
  - [ ] 建立**结构性离子白名单/黑名单**：所有结构结合的 Na⁺/Cl⁻ 按 index 显式排除出
    co-ion 候选集，并在 manifest 里单独登记为 `structural_ions`。
  - [ ] 这些结构离子在预平衡与全部 λ 中必须保持占位，不得漂走；
    加入 §9 膜质量门的监测项（占位率、与配位残基的距离）。
- [ ] **SERT 是转运体，有底物通透通路**：co-ion 的 bulk-water 判据必须排除
  胞外前庭（vestibule）与通道腔，不能只用"离配体最远"（§3.4 已禁止运行时重选）。
- [ ] 记录所用构象态（outward-open / occluded / inward-open）与来源结构 ID；
  不同构象态的 S1 可及性和水化程度不同，不可混用。
- [ ] 配体 +1 的质子化态必须与 pH 和实验测定条件一致，并记录来源。
- [ ] 确认 S1 位点**无脂质暴露面**（这是选它而非 GPCR 变构位点的理由），
  以此把 §3.0 的空腔填充迟滞风险降到最低；若实测发现有脂质暴露面，
  按 §14 的 R1 走。

- [ ] 记录受体、构象状态、突变、辅因子、结构水和缺失残基处理。
- [ ] 记录配体残基名、质子化态、互变异构体、形式电荷和参数来源。
- [ ] 记录膜组成、上下叶是否对称、胆固醇比例、盐浓度和目标温度。
- [ ] 记录蛋白、脂质、配体、水、离子的完整力场组合及版本。
- [ ] 明确实验比较的是：
  - [ ] 水相 1 M 标准态结合；
  - [ ] 表观结合（可能混入膜分配）；
  - [ ] 膜相二维/面密度标准态结合。
- [ ] 第一套实现优先选 `q = 0` 或 `|q| = 1` 的配体；不要用多价配体作为首个验证体系。
- [x] **✅ 已定（2026-07-29）；实现完成 2026-07-30：脂质力场按输入自动识别力场族**（`forcefield_family`）。
  `abfe_core.detect_forcefield_family_from_top()` / `resolve_forcefield_family()`。
  **实测补记**：本仓库 `topol.top` **没有 `[ defaults ]` 段**（它在
  `amber14sb_OL15_fs1.ff/forcefield.itp` 里），所以主判据只能是 `#include` 路径，
  `[ defaults ]` 若存在则一并记录作交叉检查、不作唯一判据。
  实测该文件判定为 amber（依据 3 条 `amber14sb_OL15_fs1.ff/*` include，
  `./Atenolol-rank1.itp` / `posre.itp` 等本地 include 正确忽略）。
  混合 include、自定义 ff 目录、opls/gromos 全部 fail closed。
  从 `.top` 的 `#include` 与 `[ defaults ]` 判定：
  - 识别到 amber 系（如现有 `amber14sb_OL15_fs1.ff`，`topol.top:21`）→ 走 Amber 脂质
    （Lipid21 / Lipid17）+ `dispersion_protocol = ff_native_isotropic_lrc`；
  - 识别到 charmm 系 → 走 CHARMM36 脂质 + `ff_native_force_switch_no_lrc`。
  - [ ] 识别结果必须写进 provenance 与所有缓存指纹，且**允许显式覆盖**
    （`--forcefield-family`），但覆盖必须留记录，不能静默。
  - [ ] 识别不出来（混合 include、自定义 ff 目录）时 **fail closed**，
    不许猜、不许默认回落到 amber。
  - [ ] ⚠️ **charmm 分支必须先卡住**：OpenMM 的 `NonbondedForce` 只有
    potential-switch，没有 force-switch，无法复现 CHARMM36 脂质的原始 Hamiltonian
    （详见 §1.3）。识别到 charmm 时默认 fail closed，只有在给出定量偏差论证
    （APL / 膜厚 / 单点能对照）后才允许放行。amber 分支无此问题，作为首选路径。
- [ ] 记录所选脂质力场**原始参数化时使用的非键设置**（cutoff、switch 类型、是否开
  色散长程修正、水模型、恒压方式），§1.3 只能在这个约束下选路线。
- [ ] **盒型必须是长方体（rectangular），膜法向对齐 z。**
  不得用截角八面体/十二面体。当前溶剂腿盒子逻辑（`runabfe.py:855-882`）产的是立方盒，
  膜复合物腿的 XY/Z 各向异性盒必须单独走一遍 minimum-image 与 PBC 修复验证。
- [ ] 记录结合位点的**溶剂暴露程度**：口袋是水相可及、位于水–脂界面、还是完全埋在
  脂双层疏水核内。这一条决定 §3.0 的风险等级，必须在选体系时就写死，不能事后补。

#### 1.1.1 明确非目标（本轮不做，写下来防止范围蔓延）

- [ ] 不做膜相二维/面密度标准态；本轮只交付水相 1 M 标准态。
- [ ] 不做膜分配腿（water → membrane transfer leg）。
- [ ] 不做多价配体（`|q| ≥ 2`）的生产资格，只保留接口不报错。
- [ ] 不做脂质力场自身的参数拟合或验证，只做"按原始参数化条件正确使用"。
- [ ] 不改 IBS 估计量层（TMBAR / 相邻 BAR 口径），膜工作不得顺手动 §P1-21/22/23 的结论。
- [ ] 不把膜功能设为默认开关；`system_type` 默认仍是 `soluble`。

### 1.2 电荷路线

新增显式配置，禁止根据“有没有 APBS 数值”猜协议：

```json
{
  "charge_treatment": "neutral",
  "co_alchemical_ion": null,
  "apbs_correction_kJ_mol": 0.0
}
```

允许值：

- [x] `neutral` ✅ 2026-07-30（B2）
  - 配体净电荷为 0；
  - 不创建 co-ion；
  - APBS/Rocklin 净电荷修正必须为 0。
- [x] `co_alchemical_charge_transfer` ✅ 复合物腿 2026-08-04（B3）；✅ 溶剂腿 2026-08-05（B4）
  - 配体净电荷不为 0；
  - 两腿都必须存在、选择并约束 co-ion —— ✅ 复合物腿已实现并冻结身份；
    ✅ **溶剂腿 builder（B4）已落地**：`runabfe._insert_reserved_coalchemical_ion_dummies`
    在 `charge_treatment=co_alchemical_charge_transfer` 且配体带净电荷时，摘掉离配体
    质心 minimum-image 最远的 `|q_L|` 个水分子，换成同号（Na⁺/Cl⁻ 形状、电荷强制
    清零）的 ion-shaped dummy，`build_and_cache_solvent_leg` 不再 fail closed；
    身份选择/restraint/charging 电荷映射复用既有的 leg-agnostic 实现
    （`ibs_engine.select_co_alchemical_ion_once` /
    `abfe_core.build_co_alchemical_ion_identity`），没有另造第二套判据。
    `abfe_core.CHARGE_TRANSFER_SOLVENT_LEG_IMPLEMENTED = True`，
    `SOLVENT_CACHE_PROTOCOL_VERSION` 5→6（manifest 加 `charge_treatment` /
    `reserved_coion_*` 字段，旧缓存 fail closed 重建）。
    单元测试：`tests/test_solvent_leg_coion_builder.py`（新增，插入逻辑本身）+
    `tests/test_charge_transfer_hamiltonian.py` / `test_charge_treatment_protocol.py`
    的相关断言已同步翻转。
    ⚠️ **只测过合成 topology，没有在真实带电配体体系上机验证过**——本仓库的
    生产体系 Atenolol 净电荷为 0，这条路径在这里测不出来；§4.2 的盒子尺寸敏感性、
    §4.4 的预平衡稳定性仍待真正带电配体上机验证，不算已收口。
  - 所有 charging λ 上总电荷必须恒定 ✅ 由 `Σscale = 0` 代数保证 + 读回真实 Force 核对；
  - APBS/Rocklin 必须为 0 ✅（B2 起就在拦双计数）。
- [ ] `rocklin_apbs_neutralizing_plasma`
  - 允许总电荷随 λ 改变；
  - 禁止创建 co-alchemical ion；
  - 最终结果必须提供真实 APBS manifest/result、膜介电图和脂质电荷图；
  - 在真实膜蛋白体系完成验证前不作为默认生产路线。
- [ ] `co_annihilation_experimental`（**实验对照专用，非生产**，见 MEM-00a-2）
  - 配体与异号反离子同步去电荷；
  - `system_type = membrane` 时一律 fail closed；
  - 输出必须带 `experimental_not_for_production: true`；
  - 只用于水盒 / lipid slab 的方法对照，其数值不得进入任何 ΔG_bind 汇总。

**生产默认值：`co_alchemical_charge_transfer`（带电配体）/ `neutral`（中性配体）。**

以下组合必须 fail closed：

- [ ] `co_alchemical_charge_transfer` 且 `apbs_correction_kJ_mol != 0`。
- [ ] `neutral` 但检测到配体净电荷不为 0。
- [ ] `co_alchemical_charge_transfer` 但缺 co-ion 身份、参数或 restraint。
- [ ] `rocklin_apbs_neutralizing_plasma` 但缺 APBS 来源说明/结果文件。
- [x] 配体电荷变化与 co-ion 电荷变化之和不为 0。✅ 2026-07-30（B2）

以上 5 条 fail-closed 组合与 4 个允许值全部落在
`abfe_core.resolve_charge_treatment()`，测试见 `tests/test_charge_treatment_protocol.py`。
**检查顺序有意义且有测试钉住**：APBS 双计数在"缺 co-ion"之前拦，"缺 co-ion"在
"B3 未实现"之前拦——协议错误不该被"反正还没实现"掩盖。

### 1.3 LJ/色散路线

> **⚠️ 修正（2026-07-29）：原文"膜体系禁止使用均匀密度 LRC"这条一刀切是错的，会
> 直接排除掉唯一与现有 Amber 力场同族的脂质力场。**
>
> 判据应该是**跟随所选脂质力场的原始参数化条件**，不是"膜体系一律禁止"：
>
> - **Amber Lipid21 / Lipid17 系**：参数化时就是在开着各向同性 vdW 长程修正
>   （GROMACS `DispCorr = EnerPres`）、cutoff 1.0 nm、无表面张力 NPT 下拟合的。
>   对这类力场**关掉 LRC 才是错的**，会系统性改变 APL。
> - **CHARMM36 脂质**：参数化时是 force-switch 1.0–1.2 nm、**不**加 LRC。
>   对它开 LRC 才是错的。
>
> 因此真正要禁止的是下面两件具体的事，而不是"LRC"这个名字：
>
> - [ ] **禁止把环境–环境的 LRC 口径直接套到炼金 ligand–environment 项上。**
>   现有 `lj_tail_lrc_coeff[k]/V(t)`（`ibs_engine.py:1743-1745`、`2261`）假设配体
>   周围是**均匀体相密度**。配体埋在脂双层口袋里时这个假设直接不成立——局域密度
>   既不是水也不是体相脂质。这才是膜体系下的真实缺陷。
> - [ ] **禁止 cutoff 口径不自洽**：LRC 积分下限 1.2 nm（`LJ_TAIL_LRC_R_CUTOFF_NM`）
>   与基础 `NonbondedForce` 的 1.0 nm 不一致（MEM-00h），这个在可溶体系里就已经错了。
>
> `dispersion_protocol` 的枚举值因此至少要包含 `ff_native_isotropic_lrc`（Amber 脂质）
> 与 `ff_native_force_switch_no_lrc`（CHARMM36 脂质），而不是只有"关掉"一个选项。

膜体系禁止继续默认使用当前均匀密度 `lrc_coeff[k] / V(t)`：

- [x] 新增 `dispersion_protocol` 显式配置。✅ 2026-07-30（B6）
- [x] `system_type = membrane` 且未选择已验证的 `dispersion_protocol` 时 fail closed。✅
- [x] 禁止把 APBS 当成 LJ 修正。✅ 解析结果带 `apbs_is_orthogonal_to_dispersion: true`
  并落 provenance。

候选路线只能选择一条：

#### 路线 A：复现目标膜力场的 cutoff/switch，不加均匀密度 LRC

- [ ] 从原始 GROMACS `.mdp/.top` 锁定 cutoff、switch 类型和距离。
- [ ] 普通 `NonbondedForce` 与炼金 softcore 力使用同一非键协议。
- [ ] 明确区分 energy-switch 与 force-switch；不能因为距离同为 1.0–1.2 nm
  就认为 Hamiltonian 相同。
- [x] 复合物腿和溶剂腿使用同一套 ligand–environment 非键定义。✅ 2026-07-30
  溶剂腿 pipeline 接同一个已解析的 `dispersion_protocol`（但**不**接膜恒压协议）。
  有 AST 契约测试钉住全部 6 个 `ABFEPipeline` 构造点。
- [x] 关闭当前 `lrc_coeff/V`，metadata 写
  `disabled_by_membrane_forcefield_protocol`，不能写成遗漏。✅ 2026-07-30（B6 接线）
  实现口径：被关掉的**只有炼金 ligand–environment 那一项**（它假设配体周围是均匀
  体相密度，埋在脂双层口袋里直接不成立）；环境–环境色散仍按所选力场的原始参数化
  条件由基础 `NonbondedForce` 处理——所以这不是"膜体系一律禁用长程色散修正"。
  理由字符串会进 `final_results.json`，措辞不要改。
- [ ] **OpenMM 的 `NonbondedForce` 没有 force-switch，只有 potential-switch
  （`setUseSwitchingFunction`）。**
  若脂质力场要求 force-switch（CHARMM36 系），OpenMM 侧**无法复现同一个 Hamiltonian**，
  用 potential-switch 顶替会移动 APL 与膜厚。这条要么直接排除该力场族（与 MEM-00k
  的结论一致），要么必须给出定量证据说明偏差可接受。不许默默用 potential-switch 冒充。
- [ ] **GROMACS ↔ OpenMM 单点能量逐项对照，作为非键协议复现的硬证据。**
  用同一膜体系的同一帧，比对 bonded / LJ / Coulomb(real+recip) 分项与总能，
  容差进 §13。仓库已有 `gromacs_reference/` 目录，沿用同一套对照流程，不要新造一套。

#### 路线 B：LJ-PME

- [ ] 普通环境–环境相互作用使用 LJ-PME。
- [ ] 炼金 ligand–environment 的 reciprocal-space LJ 也必须进入 λ Hamiltonian。
- [ ] 证明 λ=1 与完整 LJ-PME 参考 System 的能量和力一致。
- [ ] 证明 λ=0 的 ligand–environment LJ 严格归零。
- [ ] 在上述四项完成前，不能只把基础 `NonbondedForce` 切成 `LJPME` 就宣称支持。

#### 路线 C：膜非均匀色散修正

- [ ] 只有在给出推导、实现和已知答案验证后才允许启用。
- [ ] 修正必须依赖膜的空间密度/组分，而不是仅依赖总体积。
- [ ] 必须是逐 λ、与实际 softcore/switching 表达式一致的修正。

### 1.4 标准态和 Boresch

- [ ] 保持水相 1 M 标准态定义：

  ```text
  ΔG_bind = ΔG_solvent - ΔG_complex + ΔG_APBS
  ```

- [ ] 使用 charge-transfer 路线时 `ΔG_APBS = 0`。
- [ ] Boresch 只施加在复合物腿，不施加在溶剂腿。
- [ ] co-ion restraint 与 Boresch restraint 是两类独立限制，不能混用解析修正。

---

## 2. Dummy-charge / co-alchemical charge-transfer ion 的具体定义

### 2.1 λ 定义

设配体原始总电荷为 `q_L`，charging 坐标 `lambda_q` 满足：

```text
lambda_q = 1: ligand fully charged
lambda_q = 0: ligand electrostatics decoupled
```

逐原子配体电荷：

```text
q_lig_i(lambda_q) = lambda_q * q_lig_i(full)
```

同号 co-ion 从中性变为带电：

```text
q_coion(lambda_q) = (1 - lambda_q) * q_L
```

因此：

```text
sum(q_lig(lambda_q)) + q_coion(lambda_q) = q_L
```

总体系电荷在每个 λ 必须与 λ=1 完全相同。

### 2.2 co-ion 粒子模型

✅ 全部实现 2026-08-04（B3）。⚠️ **它必须是建系时额外预留的粒子**：λ=1 端的总电荷
必须等于物理体系的总电荷，而普通离子已按 §4.3 把配体的形式电荷配平掉了，
所以拿一个**已经带电**的物理盐离子当 co-ion 会让 λ=1 端总电荷少一个单位电荷。
代码在两处 fail closed（`co_alchemical_charge_offset_plan` 与
`verify_co_alchemical_ion_identity`），报错里直接写明"建系时额外加 |q_L| 个电荷为 0 的
ion-shaped 粒子"。

- [x] co-ion 是 System 中真实存在的粒子，必须进入 PME `NonbondedForce`。
- [x] 第一版只改变 charge；mass、sigma、epsilon 在所有 λ 保持不变。
- [x] λ=1 时它是“中性但保留 LJ 的 ion-shaped dummy”。
- [x] λ=0 时它成为与 `q_L` 同号的物理单价离子。
- [x] `q_L = +1`：使用 `0 -> +1` 的阳离子型 co-ion。
- [x] `q_L = -1`：使用 `0 -> -1` 的阴离子型 co-ion（share 取 `sign(q_L)`，同一段代码）。
- [x] `|q_L| > 1`：使用多个单价 co-ion，每个最多转移一个单位电荷。
- [x] 非整数净电荷必须先作为输入错误调查；不要静默把它塞给一个分数价 co-ion。

注意：当前 `abfe_core.py::GhostIonHandler` 不是这里需要的方法。它使用固定空间点和
自定义短程形式，不是一个进入 PME 粒子电荷表的 co-alchemical ion，不能用来证明
每个 λ 的周期体系净电荷恒定。

### 2.3 co-ion 的位置和约束

- [ ] co-ion 必须位于体相水区，不在膜头基层、膜疏水核、蛋白孔道或结合口袋。
- [ ] 复合物腿至少满足：
  - [ ] 与配体 minimum-image 距离大于预设阈值；
  - [ ] 与蛋白重原子距离大于预设阈值；
  - [ ] 与膜磷酸/头基和膜中心面的距离落在明确的 bulk-water 区间；
  - [ ] 不与关键离子位点或通道电场耦合。
- [ ] 溶剂腿满足：
  - [ ] 与配体距离足够远；
  - [ ] 与盒边的 minimum-image 距离安全；
  - [ ] 与复合物腿 co-ion 尽量使用相同水模型、离子类型和局域溶剂条件。
- [x] 使用可审计的 restraint；优先选择平坦区足够大的 flat-bottom restraint，
  避免把 co-ion 锁死在一个异常水构象。✅ 2026-08-04（MEM-00d）：r₀ = 0.5 nm、
  k = 100 kJ/mol/nm²，全部参数进身份指纹。
- [x] restraint 参考位置使用盒分数坐标或可随盒缩放的定义，不能在膜 NPT 中固定一个
  会漂入膜内的绝对笛卡尔坐标。✅ 走**锚点相对**：井心 = 锚点原子当前位置 + 冻结位移，
  所以它跟着 barostat 缩放；实现口径与被否掉的两个方案见 MEM-00d 条目。
- [x] restraint 势在所有 λ 完全相同。✅ force group 6，表达式里没有 λ，有测试逐 λ 比能量。
- [ ] restraint 的自由能是否在两腿抵消必须写进热力学循环说明；若几何/可用体积不同，
  需要显式修正或用数值对照证明差异可忽略。
  🔗 **MEM-00e 仍未完成**，但论证材料已经有了：两条腿用**同一条锚点规则**
  （配体重原子中离质心最近者）、同一个 k 与 r₀ ⟹ 可用体积在两腿相同，
  restraint 的自由能因此对消。这段论证要落到 `THERMODYNAMIC_CYCLE_DOC` 里
  （与 B4 一起做，那时溶剂腿的 co-ion 才真正存在，可同时给数值对照）。

### 2.4 charging Hamiltonian

- [x] 修改现有 charging builder，使 ligand 和 co-ion 的电荷由同一个 `lambda_q`
  控制，变化方向相反。✅ 2026-08-04：**新增** `configure_charge_transfer_decharging`，
  没有改 co-annihilation 那个（它按 MEM-00a-3 保留作负对照）；两者共用 restraint
  注入与电荷账目核对，分派由冻结 spec 的 `charge_treatment` 决定、只有一个分派点。
- [x] co-ion 电荷必须通过 PME `NonbondedForce`/其合法 λ 参数化实现。✅ particle
  parameter offset（`base + λ·scale`），与配体走同一条机制。
- [x] 禁止用 cutoff `CustomNonbondedForce` 模拟 co-ion 的长程静电。✅ 有测试断言
  co-ion 不出现在任何 `CustomNonbondedForce` 的相互作用组里。
- [x] 每个 charging state 的能量矩阵同时包含 ligand 与 co-ion 的变化。✅ u_kn 重算与
  动力学走同一个 prepared system 构造函数、消费同一份冻结 spec。
- [ ] TI/BAR/MBAR 的 `u_kn`、checkpoint、manifest 和协议指纹都必须包含 co-ion 身份和参数。
  ⚠️ **这条是 B5**，未完成：`configure_pme_ligand_charge_offsets` 现在会把
  `co_alchemical_ion_fingerprint` 放进返回的 info 里，但还没有写进窗口 manifest 与
  `_stage_protocol_key`。
- [x] stage 2 vanishing/vdW 阶段固定：✅ 机械保证，不需要额外记状态 ——
  co-ion 电荷是 `(1−λ_coul)·q_L`，而 vanishing 阶段 λ_coul 恒为 0。
  - ligand 电荷为 0；
  - co-ion 电荷为 `q_L`；
  - co-ion 不参与 ligand 的 vdW decoupling（它就是普通环境粒子，不进炼金 vdW 集合）。
- [x] 不能让 co-ion 在 stage 1 后被错误恢复成中性。✅ 同上：λ_coul=0 ⟹ 满电，
  有测试 `test_stage2_holds_the_ligand_at_zero_and_the_coion_fully_charged`。

---

## 3. 复合物腿改造

### 3.0 膜内结合位点的物理风险（原稿完全没有，但很可能是最大的单点风险）

配体消失后留下的空腔，在可溶体系里由水在几十 ps 内填满；**在脂双层内部由脂质尾链
填充，弛豫时间尺度是 10–100 ns 量级**，远长于单个 λ 窗口（当前 250k 步 = 500 ps，
`abfe_config.json`）。这会直接产生 vanishing 腿的迟滞与窗口间不可逆性。

- [ ] 明确记录结合位点类型：水相可及口袋 / 界面口袋 / 脂质暴露口袋 / 完全埋入疏水核。
- [ ] 若口袋有脂质暴露面，必须实测 stage 2 的迟滞：正反向扫 λ，或对同一 λ 用两组
  独立起始构型（一组来自 λ 大端、一组来自 λ 小端）比较 ΔF。
- [ ] 诊断量必须逐窗口保存：口袋内水分子数、口袋内脂质重原子数、口袋体积。
  这三条随 λ 的曲线若出现台阶或双峰，就是填充未弛豫，不是统计噪声。
- [ ] 迟滞超阈值时的正解是**延长每窗口采样 / 加窗口 / 上 REMD 交换**，
  **不是**调 IBS bias 或换估计量。
- [ ] 明确写清 Boresch restraint 在此处的作用：它固定的是配体位姿，
  **不会**加速空腔的脂质填充，不能拿它当迟滞的解释或借口。
- [ ] 若配体本身亲脂，溶剂腿（纯水）里可能出现构象塌缩或自聚集；
  记录溶剂腿配体回转半径与内部氢键随 λ 的变化。

### 3.1 膜体系识别

新增配置建议：

```json
{
  "system_type": "membrane",
  "membrane": {
    "normal_axis": "z",
    "surface_tension_bar_nm": 0.0,
    "xy_mode": "isotropic",
    "z_mode": "free",
    "barostat_frequency": 25
  }
}
```

- [x] 不根据残基名自动猜 `system_type`；用户必须显式声明。✅ 2026-07-30（B1）
  未声明即 soluble；拼错的值报错，不静默回落。
- [x] 可使用脂质残基检测做交叉检查，但不能用作唯一判据。✅
  **实施补记**：脂质名集合按误判后果拆成两套——
  宽集合（含 Amber Lipid21 模块化短名 `PC`/`PE`/`OL`/`ST`…）用于
  "membrane 但无脂质 → fail closed"这个方向，误认无害；
  窄集合（只含 `POPC`/`DOPC`/`CHL1` 等无歧义全名）用于
  "soluble 但有大量脂质 → 拦下来"这个方向，因为误判会挡住合法的可溶体系运行。
  实测生产体系 `solv_ions.gro` 的残基名（SOL / 20 种氨基酸 / ASH / CL / NA / MOL）
  与两套集合都无交集 → 对现有可溶路径零影响。
- [x] `system_type=membrane` 而拓扑中没有任何声明的脂质残基时 fail closed。✅
- [x] `system_type=soluble` 却检测到大量脂质时警告并要求确认。✅
  阈值 `SOLUBLE_LIPID_RESIDUE_WARN_THRESHOLD = 8`（窄集合计数），
  需显式 `--confirm-soluble-with-lipids` 放行并留记录。

### 3.2 膜恒压器

- [x] 预平衡使用 `MonteCarloMembraneBarostat`：✅ 2026-07-30（B1）
  - [x] XY 等比例缩放（默认 `xy_mode=isotropic`）；
  - [x] Z 独立变化（默认 `z_mode=free`）；
  - [x] 默认表面张力 0；
  - [x] 压力、温度和频率进入 provenance（`run_provenance.json` 的 `barostat_protocol`）。
  - ⚠️ **`normal_axis` 只能是 z**：OpenMM 的膜 barostat 把膜法向硬编码在 z
    （只区分 XY 平面与 Z 轴，无换轴选项），传 x/y 一律 fail closed。
    建系时必须把膜法向对齐 z。
- [x] 溶剂腿继续使用普通 `MonteCarloBarostat`。✅ 溶剂腿 pipeline 刻意不接
  `environment_type`/`membrane`，并有接线契约测试钉住。
- [x] 检测任意已有 barostat，禁止重复添加。✅ 按类名检测三种（见 MEM-00i 实施补记）。
- [x] 若输入 System 已有不兼容 barostat，fail closed，而不是再叠加一个。✅
- [x] barostat 类型、压力、表面张力、XY/Z 模式、频率进入预平衡 fingerprint。✅
- [x] 改变上述任一参数必须使旧预平衡 checkpoint 失效。✅
  实现口径：`barostat_fingerprint_payload()` 对 **legacy soluble 协议**
  （`MonteCarloBarostat` + 频率 25，即改动前唯一存在的协议）返回 `None`，
  调用方据此**一个键都不往 payload 里加**，于是不声明 `system_type` 的运行其指纹
  与本次改动前**逐字节相同**，已有生产预平衡 checkpoint 不失效（§7.7）；
  只要环境类型变 membrane 或频率偏离 25，指纹立刻改变。
- [ ] 生产阶段若采用 NVT，必须记录使用的是哪一帧/哪组固定盒矢量。

补充（原稿缺）：

- [ ] **显式决策：炼金生产阶段用 NPT 还是 NVT。**
  - 若 NPT：盒矢量逐帧变化，`compute_u_kn` 重算时必须逐帧使用**该帧自己的**盒矢量
    （现有 LRC 已按 `V(t)` 走，`ibs_engine.py:1743-1745`，但 PME reciprocal 重算的
    盒矢量路径要单独确认）；半各向异性缩放下 XY/Z 独立变化，minimum-image 判据
    不能再用"立方盒边长/2"。
  - 若 NVT：必须说明用哪一帧的盒矢量、为什么该帧代表平衡态、以及膜张力是否因此漂移。
- [ ] **时间步与约束是显式决策，不是继承**（见 MEM-00j）：
  记录 dt、`constraints`、是否 HMR、是否 `rigidWater`；改动任一项都必须让预平衡
  checkpoint 失效。
- [ ] **位置限制释放阶梯**：膜预平衡需要分级释放蛋白/脂质的位置限制
  （仓库已有 `posre.itp` / `posre_ligand.itp`）。记录每一级的力常数、时长和释放顺序；
  不允许"一步全放"。
- [ ] **恒压器与 IBS/REMD 的相互作用**：MC barostat 的体积移动会打断邻居表复用，
  并与 REMD 交换、bias 更新的节奏耦合。记录 barostat frequency 与
  `steps_per_update`（当前 500）、交换间隔的相对关系，确认没有共振或系统性偏置。

### 3.3 膜输入与拓扑

✅ 校验层完成 2026-07-30：`abfe_core.validate_membrane_input()` /
`assign_lipid_leaflets()`，测试 `tests/test_membrane_input_and_quality_gate.py`。

- [x] 输入必须是已经完成膜构建和主要平衡的 protein–lipid–ligand–water–ion 体系。
  实现为 `MEMBRANE_INPUT_REQUIRED_PROVENANCE_FIELDS` 全部必填，缺一项 fail closed。
- [x] 不依赖当前通用 10 ns 预平衡去完成脂质重排或蛋白插膜。
  由 §9 的"预平衡时长 ≥ 一个脂质横向弛豫时间"判据强制（弛豫尺度 30 ns 时 10 ns 必不过）。
- [x] `.gro`、`.top`、全部 `.itp`、位置限制文件和力场 include 一起归档。
  **复用**既有 `runabfe._gromacs_dependency_hashes()`——它已递归哈希整棵
  `#include` 树（含 `posre.itp` / 力场 include），未另造第二套。
- [x] 记录输入 SHA256、构建工具、构建参数和最终平衡作业。
- [x] 核对：
  - [x] 坐标/拓扑原子数；
  - [x] 上下叶脂质数 —— **实测而非假设对半分**：按头基参考原子相对膜中面分叶，
    报出每叶计数、组成与不平衡度；声明值与实测不符即报错；
    识别不到头基参考原子的脂质残基单独报出，不静默丢弃。
  - [x] 水和离子数（实测并与声明交叉核对）；
  - [ ] 蛋白跨膜方向（倾角进了 §9 质量门；建系期的插膜方向核对仍待做）；
  - [ ] 配体 pose（RMSD 进了 §9 质量门；建系期 pose 核对仍待做）；
  - [ ] 辅因子、结构水、二硫键和质子化态（仍待做，需要真实膜输入才好定判据）；
  - [x] 周期盒无膜间异常接触（`membrane_periodic_image_contacts` 必须为 0）。
  - [x] 追加：**盒型必须是长方体**，三斜/截角盒直接拒绝（§1.1）。

### 3.4 co-ion 选择

- [ ] 从 bulk-water 候选区选择一个粒子作为 co-ion，或在建系时显式加入。
- [ ] 不允许运行时根据“离配体最远”每次重新选择；选中后身份写入 manifest。
- [ ] 保存：
  - atom index；
  - residue index/name；
  - 元素和离子类型；
  - λ=1/λ=0 电荷；
  - sigma/epsilon/mass；
  - restraint 参数和参考坐标；
  - 初始 minimum-image 距离诊断。
- [ ] resume 时逐项核对，任何身份漂移都拒绝旧缓存。

### 3.5 Boresch

- [ ] 受体锚点必须来自蛋白，不得选到脂质、离子或 co-ion。
- [ ] 配体锚点使用 GROMACS 真实键拓扑，不使用 0.22 nm 几何近接作为生产依据。
- [ ] 完成 `update_boresch_from_last_frame` 的二面角偏差门。
- [ ] 检查锚点 minimum-image 几何和膜预平衡后的稳定性。
- [ ] attachment 腿、解析标准态修正和主 decoupling 使用同一套六原子顺序与符号。
- [ ] 保留 harmonicity、force clipping 和 pose drift 诊断。

---

## 4. 溶剂腿重新设计

这是代码结构上最大的改造部分，但不是唯一改造。

### 4.1 新的职责边界

溶剂腿 builder 必须一次性产出：

```text
ligand + water + ordinary salt/counterions + reserved co-alchemical ion
```

并返回：

```text
System
Topology
positions
box_vectors
ligand_indices
coion_indices
ordinary_ion_indices
manifest
```

- [ ] 禁止后续阶段仅凭残基名重新猜 co-ion。
- [ ] co-ion 与普通盐离子必须有不同、稳定的身份记录。
- [ ] co-ion 可以沿用物理离子的 LJ/mass，但 λ=1 charge 被显式设为 0。

### 4.2 盒子

- [ ] 保留当前“配体 extent + 两侧 padding”的显式盒长逻辑。
- [ ] 对 charged/co-ion 路线增加最小盒长要求：
  - [ ] 配体与 co-ion 的 minimum-image 安全距离；
  - [ ] cutoff/minimum-image 条件；
  - [ ] 足够的 bulk-water 壳层。
- [ ] 至少做两个溶剂盒尺寸的敏感性验证。
- [ ] 盒子大小不能只由“能放下配体”决定。

### 4.3 离子计数和离子强度

- [ ] 区分三类电荷来源：
  - 配体形式电荷；
  - 用于整体中和/目标盐浓度的普通离子；
  - reserved co-alchemical ion。
- [ ] λ=1 时：
  - ligand 带 `q_L`；
  - co-ion 为 0；
  - 普通离子使整个盒子达到目标净电荷和盐浓度定义。
- [ ] λ=0 时：
  - ligand 为 0；
  - co-ion 带 `q_L`；
  - 普通离子不变；
  - 全盒总电荷与 λ=1 相同。
- [ ] manifest 同时记录“物理盐对数”和“alchemical co-ion 数”，不能把 dummy
  错算成普通盐浓度。
- [ ] 复合物腿和溶剂腿使用相同 nominal ionic strength、离子模型和水模型。

### 4.4 co-ion 放置与 restraint

- [ ] 初始位置远离配体且不与周期镜像过近。
- [ ] restraint 定义与复合物腿使用同一函数形式和力常数。
- [ ] 对不同盒形，使用一致的可用 restraint 体积或计算差异。
- [ ] 预平衡中确认 neutral dummy 不会贴到配体疏水表面。
- [ ] charging 过程中确认带电端点保持正常水合。

### 4.5 缓存

提升 `SOLVENT_CACHE_PROTOCOL_VERSION`，fingerprint 至少加入：

- [ ] `charge_treatment`；
- [ ] ligand 净电荷和逐原子电荷哈希；
- [ ] co-ion 类型、数量、索引、端点电荷和 LJ 参数；
- [ ] co-ion restraint；
- [ ] 普通离子数与目标离子强度；
- [ ] 水模型；
- [ ] box/padding；
- [ ] dispersion protocol；
- [ ] charging protocol version。

旧的不含 co-ion 身份的溶剂缓存必须 fail closed，不能迁移猜测。

---

## 5. APBS/Rocklin 路线的保留边界

- [ ] 保留 `apbs_correction.py`，但标注为
  `neutralizing-plasma electrostatic correction`，不是膜 ABFE 默认路线。
- [ ] `prepare` 必须同时提供：
  - [ ] `diel-map-x/y/z`；
  - [ ] `lipid-charge-map`；
  - [ ] 代表性复合物快照 ensemble；
  - [ ] 正确的 ligand/receptor/complex PQR；
  - [ ] 实际净电荷。
- [ ] `run` 使用真实 APBS binary。
- [ ] `collect` 输出 NET+USV、RIP、EMP、DSC 分项和构象不确定度。
- [ ] 用已知答案案例核对系数与单位。
- [ ] 若最终选 co-alchemical charge-transfer：
  - [ ] APBS 结果不加进 ΔG；
  - [ ] 可作为诊断对照，但必须标记 `diagnostic_only`。
- [ ] APBS 不得改变 LJ `dispersion_protocol` 的判断。

---

## 6. 代码实施清单

### 6.1 `runabfe.py`

- [ ] CLI/config 新增并验证：
  - `system_type`；
  - `charge_treatment`；
  - `dispersion_protocol`；
  - `co_alchemical_ion.*`；
  - `membrane.*`。
- [ ] 自动计算 ligand 净电荷并与配置交叉核对。
- [ ] 在创建任何 Context 前完成协议组合的 fail-closed 检查。
- [ ] 复合物腿和溶剂腿都传递同一 charge-transfer 协议。
- [ ] 最终汇总禁止 co-ion + APBS 双计数。
- [ ] provenance 写清楚实际采用的电荷和 LJ 路线。

### 6.2 `abfe_pipeline.py`

- [ ] `pre_equilibrate()` 按 `system_type` 选择 barostat。
- [ ] barostat 检测覆盖普通、膜和各向异性类型。
- [ ] 预平衡 fingerprint 加入 membrane/co-ion/dispersion 身份。
- [ ] PBC repair 在真实膜体系上验证：
  - 脂质分子完整；
  - 膜不被错误重排；
  - 蛋白–配体保持同一周期镜像。
- [ ] pipeline state 和 checkpoint 写入 co-ion identity。
- [x] 复合物腿结果写膜质量诊断摘要。✅ 2026-07-30
  `ABFEPipeline._evaluate_membrane_quality_gate_after_equilibration()` 在**预平衡
  轨迹刚落盘时**就跑提取器 + 判定层，写 `membrane_quality_gate.json`，
  门未过直接 raise（§9 末句要求阻断，不是 warning）。位置选在这里是因为 §9 的要求
  是"**进入 ABFE 前**至少保存并审查"——门没过不该继续烧 λ 窗口。
  可溶体系不进这个分支。
  口袋原子 / co-ion 索引 / 文献 APL / 声明预平衡时长由 `--membrane-input-declaration`
  显式给出，不做运行时推断（同一体系两次跑必须用同一口袋定义，MEM-00c 的教训）。

### 6.3 `abfe_core.py`

- [ ] 新建真正的 `CoAlchemicalIonSpec`/等价数据结构。
- [ ] 不复用当前 `GhostIonHandler` 作为 PME charge-transfer 实现。
- [ ] 提供：
  - co-ion 参数验证；
  - λ 电荷映射；
  - 总电荷守恒检查；
  - restraint 构建；
  - manifest 序列化。
- [ ] SolventLegRunner 返回 co-ion/普通离子分离后的身份。
- [ ] Boresch 配体侧读取真实 GROMACS bond topology。

### 6.4 `ibs_engine.py`

- [ ] charging System 构建同时参数化 ligand 和 co-ion 电荷。
- [ ] 所有 charging λ 的实际粒子电荷和必须恒定。
- [ ] energy collection、fixed-H、REMD/bridge、BAR/MBAR 重算都看到同一 co-ion Hamiltonian。
- [ ] stage 2 固定 co-ion 在 fully charged 端点。
- [ ] 充电协议版本升级，旧 charging checkpoint 全部失效。
- [x] membrane + legacy uniform-density LRC 必须 fail closed。✅ 2026-07-30
  `resolve_dispersion_protocol()` 里判死；`ABFEPipeline` 走同一个校验实现，
  所以 pipeline 层自动继承这条，不需要第二套判据。
- [x] 新 LJ 协议进入窗口 manifest、energy cache 和 resume gate。✅ 2026-07-30
  非 legacy 时写进 `_stage_protocol_key` 协议指纹；legacy 时刻意不写以保住
  已有生产 stage 缓存（§7.7）。

与 IBS 机制本身的耦合（原稿缺，这部分不做会静默出错）：

- [ ] **co-ion 必须进入 alchemical/perturbed 集合的每一处定义**，不只是 charge offset。
  `ibs_engine.py:528-529` 的注释已经写明"co-alchemical counterion cycle 必须同时包含
  配体原子和所选反离子"——要逐处核对这个约定在新实现里仍然成立。
- [ ] **IBS bias 的 CV / f_k 学习必须看到 co-ion 的能量贡献**；
  若 co-ion 的静电走 `NonbondedForce` particle offset 而 CV 只覆盖 softcore 力，
  bias 学到的就是残缺 Hamiltonian。明确写出 co-ion 贡献落在哪个 force group / CV。
- [ ] **λ 阶梯与窗口数要重估**：charge-transfer 在两个位点同时改电荷，
  相邻窗口的 ΔF 与当前中性配体（stage1 12 态）不可比。
  用 pilot 重新定 stage1 窗口数，不要直接沿用 12。
- [ ] **ESS / overlap / 收敛门的阈值不得为了让膜体系变绿而放松。**
  尤其不得把已退役为 diagnostics-only 的 `min_occupancy_normalized` 重新塞回
  `converged`（见 `docs/TODO.md` TEST-GATE-01）。
- [ ] **co-ion restraint 力（force group 6）不得混进任何 λ 相关的能量分解或 u_kn 差值**；
  它逐 λ 相同，但要在代码里显式断言，而不是靠"应该会抵消"。

### 6.5 `apbs_correction.py`

- [ ] manifest/result 写入与 charge treatment 的兼容声明。
- [ ] collect 输出增加：
  - `applicable_only_to_neutralizing_plasma: true`；
  - `must_not_combine_with_co_alchemical_ion: true`。
- [ ] 最终 CLI 提示中明确 APBS 与 LJ LRC 正交。

---

## 7. 自动化测试

### 7.1 配置与 fail-closed

- [x] 中性配体 + `neutral`：通过。✅ 2026-07-30（B2）
- [x] 带电配体 + `neutral`：失败。✅
- [x] 带电配体 + co-ion：**规格校验通过，但 B3 未落地故 fail closed**（`NotImplementedError`）。✅
  测试分两步断言，把"规格合法"与"哈密顿量还没有"分开，B3 落地后能看出当初是哪一环在挡。
- [x] co-ion + 非零 APBS：失败。✅
- [ ] membrane + 未指定 dispersion protocol：失败。
- [ ] membrane + 普通各向同性 barostat：失败或明确替换，不能静默继续。

### 7.2 电荷守恒

对每个 charging λ（✅ 2026-08-04，`tests/test_charge_transfer_hamiltonian.py`）：

- [x] `sum(all NonbondedForce particle charges)` 恒定到严格数值容差（1e-6 e）。
  判据是代数的：`Σq(λ) = Σq_base + λ·Σq_scale`，`Σscale = 0` 覆盖所有 λ。
- [x] ligand charge 等于 `lambda_q * q_L`（容差 `LIGAND_CHARGE_LAMBDA_TOLERANCE_E`）。
- [x] co-ion charge 等于 `(1-lambda_q) * q_L`。
- [x] λ=1 和 λ=0 均满足预期，且与**独立手写参照体系**的能量（rel ≤ 1e-5）与
  逐原子力（max|Δ| ≤ 1e-3 kJ/mol/nm）逐项对照通过（§13.2）。λ=1 的参照就是物理体系。
- [x] λ 中间态也满足（λ = 0.37 同样做了参照对照；电荷账目取 11 个 λ 点）。
- [x] energy matrix 重算使用的 charge 与动力学 System 完全一致 —— 两者都走
  `_prepare_pme_coulomb_leg_system` / `_prepare_pme_mixed_alchemical_system`，
  消费同一份冻结 spec，分派点只有一个（MEM-00c + 本轮的源码契约测试）。

### 7.3 co-ion 物理测试

- [x] co-ion mass/LJ 在 λ 间不变（✅ 2026-08-04：sigma/epsilon offset 必须为 0）。
- [x] co-ion charge 通过 PME，不通过 cutoff ghost force（✅ 断言粒子在 PME 的
  particle-parameter-offset 里、且不在任何 `CustomNonbondedForce` 相互作用组里；
  `GhostIonHandler` 已从 `ibs_engine` 完全消失）。
- [x] restraint 在三斜/各向异性盒中 minimum-image 正确（✅ 6.0×6.0×12.0 nm 盒，
  含"离子被回卷到盒另一头"的构造）。
- [x] NPT 盒变化后 co-ion 不漂入膜（✅ z 方向坐标+盒同乘 1.05 后 flat-bottom 能量仍为 0；
  同一构造下退役的绝对参考点形式会产生 >1 kJ/mol 的伪能量与一个指向膜的力）。
- [x] 多 co-ion 分摊电荷时，总变化正确、每粒子不超过一个单位电荷（✅ q_L=+2 用 2 个
  单价 dummy；只预留 1 个即 fail closed）。
- [ ] ⚠️ 仍待真机：以上全部是 CPU 合成体系测试。带电体系的第一次真机验证是 §17.0 的 C1。

### 7.4 barostat

✅ 全部落地（2026-07-30，B1）：`tests/test_membrane_barostat_protocol.py`。

- [x] membrane complex 得到 `MonteCarloMembraneBarostat`。
- [x] solvent leg 得到 `MonteCarloBarostat`。
- [x] 已有 barostat 不会被重复添加。
- [x] 改表面张力/模式会让 checkpoint fingerprint 失效。
- [x] XY、Z 模式和膜法向写入 provenance。
- [x] 追加：不声明 `system_type` 时 `_pre_equilibration_fingerprint` 与改动前
  逐位相同（§7.7 的直接验证，不只验 payload）。
- [x] 追加：把腿身份 `complex`/`solvent` 传进膜协议解析会报错（命名撞车防护）。
- [x] 追加：输入已带膜 barostat 时不会被各向同性的"影子覆盖"（旧 isinstance 漏检点）。

### 7.5 LJ/end-state

- [ ] λ=1 的 ligand–environment 能量/力匹配冻结参考 Hamiltonian。
- [ ] λ=0 的 ligand–environment 静电与 LJ 都为零。
- [ ] NBFIX/pair-specific exceptions 若存在，不能被 Lorentz–Berthelot 静默覆盖。
- [ ] 复合物腿和溶剂腿的 ligand 非键协议一致。
- [ ] membrane 模式下没有 `lrc_coeff/V` 混入 u_kn，除非所选路线经过明确验证。

### 7.6 缓存/resume

- [ ] co-ion index、charge、LJ、restraint 任一变化均拒绝旧缓存。
- [ ] charge treatment 或 dispersion protocol 变化均拒绝旧缓存。
- [ ] complex/solvent 不能串用 co-ion manifest。
- [ ] resume 后 co-ion 仍是同一粒子、同一 restraint、同一 λ 方向。

### 7.7 原有体系回归

- [ ] 现有可溶性中性 Atenolol 路线数值不变。**回归基线钉死为**（`docs/TODO.md`
  2026-07-29 符号修复后全量）：

  ```text
  复合物腿 ΔG_cplx = 181.00 ± 1.76 kJ/mol
  溶剂腿   ΔG_solv = 157.84 ± 1.79 kJ/mol
  ΔG_bind = −5.535906 kcal/mol（落盘值，逐位比对）
  ```

  任何膜相关改动导致这三个数变化，都必须先解释清楚再合入，不许"看起来差不多"。
- [ ] `system_type=soluble` 不会意外加入膜 barostat/co-ion。
- [ ] **膜功能默认关闭**：不传 `system_type` 时行为与改动前逐位一致。
- [ ] 完整离线测试全绿（`tests/run_offline_tests.sh`），且新增测试全部可在
  **无 GPU** 环境跑（膜相关的纯构建/校验测试不许依赖 CUDA）。

---

## 8. 动态验证

### 8.1 最小解析/小盒验证

- [ ] ligand `+1 -> 0`、co-ion `0 -> +1` 的小水盒。
- [ ] ligand `-1 -> 0`、co-ion `0 -> -1` 的小水盒。
- [ ] 所有 λ 的总电荷严格不变。
- [ ] charging ΔG 对盒长的依赖在统计误差内。
- [ ] 对照 neutralizing-plasma + Rocklin 路线，只用于理解差异，不叠加结果。

### 8.2 膜 slab 验证

- [ ] 无蛋白的 lipid–water slab 中做 charge-transfer 小例。
- [ ] co-ion 位于 bulk water，并受 restraint 控制。
- [ ] 至少两种盒高/水层厚度。
- [ ] 至少两个 co-ion 初始位置。
- [ ] ΔG 对盒大小和位置的敏感性在预设容差内。
- [ ] co-annihilation 只作负对照，不作为默认生产方法。

### 8.3 膜蛋白 smoke test

- [ ] complex 与 solvent charging 各跑最小窗口集。
- [ ] 无 NaN、PME 异常、co-ion restraint runaway。
- [ ] 检查逐 λ：
  - 总电荷；
  - ligand/co-ion charge；
  - co-ion–ligand 距离；
  - co-ion–膜距离；
  - co-ion 水合；
  - potential energy 分解。
- [ ] stage 2 开始时 co-ion 保持 fully charged。
- [ ] Boresch 几何和 pose 无异常。

### 8.4 完整 ABFE

- [ ] 先完成当前 P1-19/P1-22/P1-23 的不确定度协议。
- [ ] 固定同一膜体系做至少 3 个独立种子。
- [ ] 报告腿内统计误差和跨重复标准差。
- [ ] 复合物腿检查：
  - overlap/ESS；
  - Boresch attachment/harmonicity；
  - ligand pose；
  - co-ion 位置；
  - 膜质量指标。
- [ ] 溶剂腿检查：
  - overlap/ESS；
  - box-size sensitivity；
  - co-ion restraint；
  - ligand/co-ion minimum-image separation。
- [ ] 用一个公开膜受体–配体 benchmark 或可追溯实验值做端到端验收。

---

## 9. 膜质量门

通用 10 ns 不是膜平衡充分性的证明。进入 ABFE 前至少保存并审查：

- [ ] 面积每脂（APL）及其时间序列；
- [ ] 双层膜厚度；
- [ ] 脂质尾链序参量或等价结构指标；
- [ ] 水/脂质沿膜法向的密度分布；
- [ ] 上下叶脂质数和组成；
- [ ] 蛋白跨膜倾角、骨架 RMSD；
- [ ] 口袋 RMSD、配体 RMSD/关键相互作用；
- [ ] 通道/口袋异常进水；
- [ ] co-ion 的 z 分布、与蛋白/配体/膜距离；
- [ ] 盒矢量、体积、XY 面积和 Z 厚度；
- [ ] 是否存在膜与周期镜像的异常接触。

补充（原稿只列了量，没列"看多久、怎么判"）：

✅ 判定层完成 2026-07-30：`abfe_core.evaluate_membrane_quality_gate()` /
`linear_drift_per_ns()`，测试 `tests/test_membrane_input_and_quality_gate.py`（30 条）。
阈值全部取自 §13.3 的命名常量，判定层不自带魔数。

- [x] 每个量都必须给出**时间序列**和**末段窗口的漂移斜率**，不能只报平均值。
  传标量会被拒（"不接受只报平均值"）；`REQUIRED_MEMBRANE_QUALITY_OBSERVABLES` /
  `REQUIRED_MEMBRANE_QUALITY_DIAGNOSTICS` 缺任一项 fail closed。
  co-ion 相关量只在共炼金路线下额外要求。
- [x] 判据统一为"末段 ≥ 20 ns 内线性漂移小于阈值"，阈值见 §13。
  序列跨度覆盖不了末段窗口时**报错而不是拿短轨迹凑**；
  实际使用的窗口写进报告，"缩窗口换绿灯"在 provenance 里藏不住。
  ⚠️ co-ion 几何几项判的是**整条序列最小值**（§13.1 写的是"全程 ≥"），
  不是末段也不是均值——只看均值会漏掉"末段掉进膜里"（有专门测试构造这一情形）。
- [x] 记录脂质横向弛豫的估计时间尺度，用它论证预平衡时长够。
  实现为硬判据：`equilibration_length_ns ≥ 弛豫尺度 ×
  MEMBRANE_EQUILIBRATION_MIN_RELAXATION_MULTIPLE`（=1.0，本实现补的，§13 未给此倍数）。
  弛豫 30 ns 时通用 10 ns 预平衡必然不过。
- [x] 报告里带 `remediation`：质量门失败时回到膜体系平衡，
  **不允许靠增加 ABFE 窗口、放宽阈值或缩短末段窗口掩盖**（§9 末句，有测试钉住）。
- [x] **APL 与文献值的对照口径已落地（MEM-03，2026-08-02）**：含蛋白膜必须比
  **蛋白横截面校正后**的 APL（新观测量 `apl_protein_corrected_nm2`），
  raw APL 把蛋白占掉的横向面积也摊给了脂质（实测 0.807 vs POPC 文献 0.645）。
  校正用**最近参考原子归属**（Voronoi 式），**没有探针半径** ——
  外扩法实测沿蛋白周长多算约 1.7 nm²、把校正后 APL 压到 0.564（低 12.6%），
  那样的门会因方法偏差而不过。栅格边长是唯一方法参数，2× 粗栅格复算随报告落盘。
  漂移判据仍判 raw APL（那测的是盒面积有没有平衡）。
  `MEMBRANE_QUALITY_GATE_PROTOCOL_VERSION` 1 → 2。
- [ ] 上下叶脂质数如何确定必须有依据（按每叶面积匹配，不是随手对半分），
  并记录膜是否出现整体起伏（undulation）或残余张力。
- [ ] 记录 co-ion 的 z 分布直方图，而不只是瞬时距离；它必须整段留在体相水层。

质量门失败时回到膜体系平衡，不允许靠增加 ABFE 窗口掩盖。

---

## 10. 结果与 provenance

最终结果必须明确记录：

```text
system_type
thermodynamic_reference
charge_treatment
ligand_net_charge
coion_identity
coion_charge_endpoints
coion_restraint
box_charge_at_every_lambda
apbs_applicable/applied
dispersion_protocol
barostat_protocol
membrane_composition
forcefield_versions
water_and_ion_models
independent_repeat_id
```

- [ ] 若 co-ion 已启用，最终说明必须写“总电荷在全部 λ 恒定，未应用 Rocklin/APBS”。
- [ ] 若 APBS 已启用，最终说明必须写“使用 neutralizing plasma；未使用 co-ion”。
- [ ] LJ 路线单独报告，不能写进 APBS note。
- [ ] 输出中保留 complex/solvent 两腿各自的 co-ion 和 box 诊断。

---

## 11. 推荐实施顺序

### Phase A：协议和输入

- [x] A1. 确定首个膜体系和配体电荷。
- [x] A2. 确定水相 1 M 标准态。
- [x] A3. 锁定力场与 LJ 协议。
- [x] A4. 选择 `neutral` 或 `co_alchemical_charge_transfer`。
- [x] 中性 Atenolol 膜体系 complex/solvent 双腿端到端运行完成。
  ✅ 2026-08-05：Resume、Boresch attachment、双 λ、溶剂腿及最终汇总均完成；
  `ΔG_bind = -26.49 ± 1.77 kJ/mol（-6.33 ± 0.42 kcal/mol）`。
  结果：`memtest/output_membrane_5ns/final_binding_results.json`。
  本项只证明中性工程主链跑通，不作为带电 co-ion 的 C4 验收。

  > **本次运行不能关闭以下验收项**：C1–C5（均要求真实带电配体）；§7.6 的
  > co-ion cache/resume 物理验收；§8 的带电小水盒与 lipid slab；C4 的带电膜体系
  > complex/solvent 双腿 smoke test；膜生产就绪/膜质量门（本次轨迹约 4.98 ns，
  > 不足要求的 20 ns 末段窗口）；以及 §7.7 的可溶性 Atenolol 数值逐位不变回归
  > （本次是膜体系，不是该回归基线）。
- [ ] A5. 准备并验证已平衡膜输入。

### Phase B：最小代码闭环

- [x] B1. 加 `system_type=membrane` 和膜 barostat。✅ 2026-07-30
  - `abfe_core.py`：`resolve_environment_type` / `resolve_membrane_protocol` /
    `detect_barostats` / `ensure_barostat_for_protocol` /
    `barostat_fingerprint_payload`，`MEMBRANE_BAROSTAT_PROTOCOL_VERSION = 1`。
  - `abfe_pipeline.py`：`ABFEPipeline(environment_type=..., membrane=...)`，
    `pre_equilibrate()` 按协议选 barostat，`_pre_equilibration_fingerprint`
    新增 `barostat_protocol` 参数。
  - `runabfe.py`：`--system-type` / `--membrane-*` / `--confirm-soluble-with-lipids`
    + 配置键 + 在建任何 Context 前解析（§6.1）+ provenance 落 `barostat_protocol`。
  - ⚠️ **命名撞车已处理**：`system_type` 在本仓库已被 `run_full_pipeline` 用作
    **腿身份**（complex/solvent，20+ 处）。环境类型（soluble/membrane）在代码里
    一律叫 `environment_type`，只在配置键/provenance 里叫 `system_type`；
    把腿身份传进膜协议解析会报错。
  - 证据：`tests/test_membrane_barostat_protocol.py`。
- [x] B2. 加 charge-treatment 配置与双计数 fail-closed。✅ 2026-07-30
  - `abfe_core.resolve_charge_treatment()` 是唯一校验实现，四值枚举 +
    §1.2 全部 5 条 fail-closed + MEM-00a-2 的膜禁用 + `_validate_co_alchemical_ion_spec()`
    （§3.4 字段齐全、每粒子 ≤1 单位电荷、同号而非异号、总电荷配平）。
  - `CHARGE_TRANSFER_PROTOCOL_VERSION = 1`（MEM-00a-1，独立版本号，不复用其它协议版本）。
  - `runabfe.py`：`--charge-treatment` / `--co-alchemical-ion` / `--apbs-evidence`
    + 配置键 + 在建任何 Context 前校验（§6.1）+ **自动算配体净电荷并交叉核对**
    （复用 `ibs_engine._compute_ligand_net_charge`，不另造判据）+ provenance 落
    `charge_protocol` / `charge_treatment` / `ligand_net_charge` / `coion_identity` /
    `apbs_applicable` / `apbs_applied`。
  - 🚧 `CHARGE_TRANSFER_HAMILTONIAN_IMPLEMENTED = False`：B3 未落地前，
    即使给出格式完全合法的 co-ion 规格也 fail closed（`NotImplementedError`），
    避免产出"声明 charge-transfer、实际跑 co-annihilation"的结果。
  - ⚠️ **带电配体行为已变更**：改动前会静默走
    `configure_coalchemical_neutral_decharging`（co-annihilation）跑完；
    现在必须显式声明 `co_annihilation_experimental`。这是 §1.2 与 MEM-00a-2/00a-4
    要求的门。中性配体（当前生产体系）路径不变，基线不受影响。
  - 证据：`tests/test_charge_treatment_protocol.py`。
- [x] B3. 实现 PME co-alchemical ion charging Hamiltonian。✅ 2026-08-04（含 MEM-00d）
  `ibs_engine.configure_charge_transfer_decharging`（ligand base 0/scale q_i，
  co-ion base share/scale −share ⟹ 逐 λ `Σscale = 0`）、
  `abfe_core.co_alchemical_charge_offset_plan`（λ 电荷映射唯一实现，纯数学）、
  `ibs_engine.charging_charge_conservation_report`（读回真实 Force 核对，§7.2）、
  `_create_co_alchemical_ion_restraint`（MEM-00d flat-bottom 锚点相对）、
  `_identify_reserved_neutral_co_ions`（charge-transfer 的身份来源，**与坐标无关**）。
  分派由冻结 spec 的 `charge_treatment` 决定，只有一个分派点。
  证据：`tests/test_charge_transfer_hamiltonian.py`。详见 `docs/TODO.md` 的 B3 条目。
- [x] B4. 重写溶剂腿 builder，插入 reserved co-ion dummy。✅ 2026-08-05
  `runabfe._insert_reserved_coalchemical_ion_dummies` + `build_and_cache_solvent_leg`
  接线，`abfe_core.CHARGE_TRANSFER_SOLVENT_LEG_IMPLEMENTED = True`。
  身份选择/restraint/charging 复用 B3 已有的 leg-agnostic 实现，没有重造一遍。
  证据：`tests/test_solvent_leg_coion_builder.py`（新）。
  ⚠️ 只在合成 topology 上单元测过，未在真实带电配体体系上机验证——详见 memtodolist
  §1.2 `co_alchemical_charge_transfer` 条目下的完整说明。
- [ ] B5. complex/solvent cache、resume 与 provenance 接入 co-ion 身份。
  **代码实现完成，定向测试 181 passed；待 `./tests/run_offline_tests.sh -q` 全套 0 failed
  后再勾选。**

  > **本条代码已经实现，但尚未满足最终离线测试证据门，因此暂不勾选。** 后面的
  > C1–C5 仍是未执行的交接指示，均保持未完成。Boresch `np.int64` JSON 序列化
  > 修复单独记录，不计入 B5。不要在本条里顺手改 charging Hamiltonian、离子选择算法
  > 或 restraint 物理形式，那些分别属于 B3/B4/C5。

  **B5 已实现的范围**：B4 的 `solvent_cache_manifest.json` 仍负责证明 builder 创建过
  中性 dummy；每条腿自己的 `checkpoints/coalchemical_ion_spec.json` 冻结身份现已通过
  同一 runtime fingerprint 接入 `_stage_protocol_key()`、`_preopt_protocol_key()`、PME
  `u_kn` metadata、窗口 resume 门和双腿总 provenance。身份变化会拒绝旧缓存；本条目前
  只剩 `./tests/run_offline_tests.sh -q` 全套 0 failed 的最终证据门。

  **先定两层身份，不要混成一个哈希：**

  1. `reserved_coion_builder_identity`：描述建系产物中“预留了哪些 dummy”。只含可由
     base `System + Topology` 稳定重算的字段：`charge_treatment`、配体净电荷、dummy 数量、
     atom/residue index、residue name、element、λ=1 电荷、sigma、epsilon、mass，以及
     restraint **协议版本/形式/默认 k/r0/force group**。它进入
     `system_cache_manifest.json` 和 `solvent_cache_manifest.json`，不得包含坐标。
  2. `co_alchemical_ion_runtime_identity`：直接引用
     `coalchemical_ion_spec.json` 的 `protocol_version` + `fingerprint`，并附腿身份
     `complex|solvent` 与 spec 相对路径。该 fingerprint 已包含 atom identity、端点电荷、
     LJ/mass、锚点、冻结位移及完整 restraint；它进入 stage/preopt/u_kn/resume/provenance。
     **不得另写第二套 fingerprint 算法**，唯一算法仍是
     `abfe_core.co_alchemical_ion_identity_fingerprint()`。

  `selection_provenance`、选择时距离/水配位数、绝对坐标和当前帧距离只作诊断，禁止进入
  上述任一 resume 强身份。否则首跑与跨进程 resume 的微小坐标差会再次制造
  RESUME-FP-01 一类假失效。

  **代码修改顺序：**

  1. `abfe_core.py`
     - 新增一个纯函数（建议名
       `co_alchemical_ion_cache_identity_payload(spec, *, system, topology, leg,
       spec_relative_path)`），先调用
       `verify_co_alchemical_ion_identity()`，再返回 JSON-safe 的最小 payload；中性路线
       返回 `None`，带电 charge-transfer 缺 spec 必须 raise，禁止返回空字典放行。
     - payload 至少有：`schema_version`、`leg`、`charge_treatment`、
       `identity_protocol_version`、`fingerprint`、`ligand_net_charge_e`、
       `lambda_direction`、`ion_atom_indices`、`spec_relative_path`。
     - builder identity 另用一个纯函数从 base System/Topology 重算；不要从 manifest
       自己抄回自己。两腿必须调用同一函数，不能在 `runabfe.py` 各写一份字段拼装。
  2. `runabfe.py`
     - `MAIN_SYSTEM_CACHE_PROTOCOL_VERSION` 与 `SOLVENT_CACHE_PROTOCOL_VERSION` 各升一版；
       旧 manifest 缺 builder identity 时，charge-transfer 路线 fail closed 并重建；
       `neutral` 路线保持原行为，不因空 co-ion 字段无谓失效。
     - `build_and_cache_solvent_leg()` 写入完整 builder identity；现有 `na_count/cl_count`
       仍是拓扑计数，另加 `ordinary_na_count/ordinary_cl_count`，计算时减掉 reserved dummy，
       防止 provenance 把 dummy 算进 0.15 M 物理盐。
     - complex `ABFEPipeline` 构造并调用 `resolve_co_alchemical_ion_spec()` 后，先原子化写入
       complex 身份；solvent pipeline 稍后构造时再原子化补入 solvent 身份。最终汇总前
       强制要求两腿字段都存在。无需为了 provenance 改成“两条 pipeline 同时常驻内存”。
       新增顶层结构应为

       ```json
       {
         "co_alchemical_ions": {
           "complex": {"fingerprint": "...", "spec_relative_path": "checkpoints/coalchemical_ion_spec.json"},
           "solvent": {"fingerprint": "...", "spec_relative_path": "solvent_leg/checkpoints/coalchemical_ion_spec.json"}
         }
       }
       ```

       实际对象还要包含上一步列出的全部最小字段。不要继续只写当前含义含混的单数
       `coion_identity`；为兼容旧读取器可暂时保留它，但必须标成 deprecated，且不得拿它
       做 resume 判定。
     - provenance 的二次写入必须用临时文件 + `os.replace`；禁止进程中断时留下半个 JSON。
       `final_results.json` 与 `final_binding_results.json` 也保存双腿 fingerprint，便于结果
       脱离 output 目录后仍可审计。
  3. `abfe_pipeline.py`
     - 在 `run_full_pipeline()` 生成 `_stage1_protocol_key` / `_stage2_protocol_key` /
       `_stage1_preopt_key` / `_stage2_preopt_key` **之前**只解析一次本腿 spec，并缓存最小
       identity payload；后面所有 key 只消费该 payload，不再次选择离子。
     - `_stage_protocol_key()`：decharging 与 vanishing 都加入 runtime identity。虽然
       stage 2 的 λ_coul 固定为 0，它仍依赖“哪个 co-ion 保持 fully charged”，不能只给
       stage 1 加。
     - `_preopt_protocol_key()`：同样加入 runtime identity；不要依赖完整
       `code_sha256` 间接失效，因为该函数刻意使用窄指纹。
     - `_pme_u_kn_meta_payload()`、`_is_pme_u_kn_cache_compatible()`、
       `_write_pme_u_kn_meta()` 增加同一个 `coion_identity` 参数。metadata 缺字段或
       fingerprint 不同必须返回 incompatible，随后重算 `u_kn`；不得只靠
       `system_xml_sha256` 暗示身份一致。
     - 顶层 `_build_top_level_protocol_key()` 也必须含本腿 runtime identity，否则
       `final_results.json` 的 early-return 可能比 stage gate 更早命中并绕开新检查。
  4. `ibs_engine.py`
     - 窗口级 `main_window` / `production_window` / fixed-H probe manifest 已包含 System XML
       哈希，物理上能随 restraint/offset 变化失效；为可审计性，再显式写入同一个
       runtime fingerprint。把它作为参数沿调用链传入，禁止在窗口内部读文件或重选离子。
     - `IBS_BIAS_PROTOCOL_VERSION` 不因“只加 manifest 字段”而升版；只提升对应 manifest
       schema 版本。若实际更改了 bias、采样或 Hamiltonian，才另行升 IBS 版本。
  5. `abfe_pipeline.py::resolve_co_alchemical_ion_spec()`
     - 已有 spec：继续“读盘 → 重算 fingerprint → 对当前 System/Topology 只读核对”；
       缺失/损坏/路线不符时 fail closed，不准自动挑一个替代品。
     - 首跑：每条腿各选一次并立即原子化落盘；复合物与溶剂腿允许 atom index 不同，
       但 `charge_treatment`、`ligand_net_charge_e`、`lambda_direction`、离子模型和 restraint
       协议必须一致。双腿 fingerprint **预期不同**，禁止要求两个 SHA 相等。

  **必须新增的测试**（建议集中到 `tests/test_coion_cache_resume_provenance.py`）：

  - charge-transfer 下分别改 atom index、λ=0/1 电荷、sigma、epsilon、mass、anchor index、
    frozen displacement、k、r0、force group、charge treatment、ligand net charge、lambda
    direction；每次只改一项，断言对应 stage/preopt/u_kn/checkpoint 缓存被拒绝。
  - spec 文件缺失、JSON 截断、自身 fingerprint 不符、complex spec 放到 solvent 腿，全部
    fail closed；错误消息必须包含腿身份和 spec 路径。
  - 原样 resume 两次：fingerprint 不变、atom index 不变、不得再次调用 selector、不得因
    `selection_provenance` 或当前坐标不同而失效。
  - neutral 路线：无 spec、无 co-ion payload，现有 Atenolol 缓存/结果语义不变。
  - provenance 同时出现 `complex` 与 `solvent`，且各自 fingerprint 与磁盘 spec 重算结果
    一致；普通盐计数不包含 reserved dummy。
  - AST/源码契约测试：禁止生产代码绕过唯一 helper 自行拼 co-ion fingerprint；禁止新的
    selector 调用点出现在 `resolve_co_alchemical_ion_spec()` 之外。

  **执行与验收命令：**

  ```bash
  cd /home/ruigengji/ABFE_IBS/Atenolol-rank11
  source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh
  mamba activate openmm_dev

  python -m pytest tests/test_coalchemical_ion_identity.py \
    tests/test_charge_transfer_hamiltonian.py \
    tests/test_solvent_leg_coion_builder.py \
    tests/test_coion_cache_resume_provenance.py -v
  ./tests/run_offline_tests.sh -q
  ```

  完成证据：四个定向测试文件全绿、全套离线测试 0 failed；测试临时目录中的两份 spec、
  两份 stage protocol payload、两份 `u_kn.meta.json` 和合并后的 provenance 均被断言。
  **不要把当前测试总数写死**，只记录实际命令、日期、passed/skipped/failed。

**独立修复记录（不计入 B5）**：Boresch 参数 JSON 落盘已修复 `np.int64` 等 NumPy 标量
无法被标准 `json.dump()` 序列化的问题；落盘改用 `_json_safe()` + `_atomic_write_json()`，
并在 `UnitFormatter.format_boresch_json()` 中显式转换 anchor indices。该修复不改变 B5
状态或 B5 测试计数。
- [x] B6. membrane 模式禁用 legacy uniform-density LRC。✅ 2026-07-30（校验层 + **接线**）
  ⚠️ **首版只做了校验层，是个真缺陷**：`resolve_dispersion_protocol()` 会接受
  `ff_native_isotropic_lrc`，但当时**没有任何代码消费它**——`build_ibs_dual_system`
  照旧只按 `potential_type` 决定要不要算 `lj_tail_lrc_coeff_kj_mol`。一次膜运行会
  通过校验、写进 provenance、然后照旧把均匀体相密度 LRC 加到炼金 ligand–environment
  项上，完全静默。同日补齐接线：
  - `ibs_engine.ibs_lj_tail_lrc_is_applicable(potential_type, dispersion_protocol)`
    ——**扩展同一个谓词**而不是加第二道门（该谓词 docstring 本就要求生产者与报告者
    共用同一真相），非 legacy 路线一律关闭炼金均匀密度 LRC，理由字符串按 §1.3 用
    `disabled_by_membrane_forcefield_protocol`（"不能写成遗漏"）。
  - 链路：`ABFEPipeline(dispersion_protocol=…)` → `IBSWindowManagerDualLambda`
    → `build_ibs_dual_system`；报告侧 `compute_final_results` 读同一谓词并落盘
    `dispersion_protocol` / `environment_type`，避免只看到 `applied=False` 时误判成 DEXP。
  - §6.4 resume gate：非 legacy 时写进 `_stage_protocol_key` 的协议指纹
    （legacy 时刻意不写，保证已有生产 stage 缓存不失效，§7.7）。
  - ⚠️ **两腿规矩方向相反，别搞混**：恒压器是溶剂腿**不**接膜协议；
    色散路线是溶剂腿**必须**接、且与复合物腿相同（§1.3 路线 A / §7.5
    "复合物腿和溶剂腿使用同一套 ligand–environment 非键定义"）。否则
    ΔG_bind = ΔG_solv − ΔG_cplx 的差值里会混进一个纯协议差且不报错。
  - 证据：`tests/test_dispersion_protocol_is_honored.py`、
    `tests/test_membrane_barostat_protocol.py`（AST 契约覆盖全部 6 个构造点）。

  校验层部分：`abfe_core.resolve_dispersion_protocol()` 是唯一校验实现，5 值枚举。
  soluble 不声明 → `legacy_uniform_density_lrc`（行为逐位不变）；
  membrane 不声明 → fail closed；membrane + legacy → fail closed。
  路线 B（`lj_pme`）与路线 C（`membrane_inhomogeneous`）作为**已识别但未实现**收进
  枚举并抛 `NotImplementedError`，好过被当成拼错的未知值。
  charmm 的 `ff_native_force_switch_no_lrc` 默认 fail closed（OpenMM 无 force-switch），
  需 `--force-switch-deviation-evidence` 给出 APL/膜厚/单点能定量论证才放行；
  即便放行，membrane 已验证集合仍只含 amber 路线。
  另有力场族 × 色散路线的交叉核对（amber 配 force-switch、charmm 配 isotropic LRC 都报错）。
  证据：`tests/test_dispersion_and_forcefield_protocol.py`。

### Phase C：物理与数值验证

- [ ] C1. 带电配体小水盒 charge-transfer 解析/盒长测试。

  **✅ 实现路径已定并部分落地（2026-08-06）**：下面这条要求"可追溯小分子输入"
  改用单原子离子 Na⁺(+1)/Cl⁻(−1)/Ca²⁺(+2) 代替质子化 Atenolol/acetate——不需要
  Gaussian/Sobtop 重新 RESP 参数化，理由与完整设计见脚本头部 docstring。脚本
  `tools/validation/validate_charge_transfer_waterbox.py` 六个子命令：
  `build`/`static-check`（纯 CPU，已自测 Na/Cl/Ca 三个 small-box case 全过，
  过程中自查自修了两处离子计数 bug）、`dynamics`/`ukn`（需 GPU，未执行，命令见
  下文/`docs/TODO.md`）、`report`/`compare-box`。下面原稿写的"目的"与矩阵设计
  仍然成立，只是"小分子"换成了"单原子离子"，其余判据（电荷守恒 1e-6e、盒长
  敏感性 ≤2σ 且 ≤1kcal/mol 等）原样适用。

  **目的**：第一次用真实带电粒子和真实 PME Context 证明 `ligand q→0 / co-ion 0→q`
  能运行、总电荷逐 λ 不变，并量化盒长依赖。当前 Atenolol 的净电荷是 0，不能拿它给
  C1 打勾；必须准备至少一个 `q_L=+1` 和一个 `q_L=-1` 的可追溯小分子输入。

  **应写的验证代码**：新增 `tools/validation/validate_charge_transfer_waterbox.py`，只做
  validation harness，不复制生产 Hamiltonian。它必须调用
  `select_co_alchemical_ion_once()`、`configure_pme_ligand_charge_offsets()`、
  `charging_charge_conservation_report()` 和生产 `compute_u_kn` 路径。脚本参数至少包括
  `--system-xml`、`--topology`、`--positions`、`--ligand-indices`、`--charge-sign`、
  `--box-edge-nm`、`--n-steps-per-state`、`--seed`、`--output-dir`；输出 JSON 与 CSV，
  不只打印日志。

  **矩阵固定为 4 个主 case**：`q=+1/-1 × 小/大盒`。同一电荷的两盒除盒长和水数外，
  水模型、离子模型、cutoff、PME tolerance、温度、盐浓度、λ 表、步数、seed 全相同。
  建议先用 `L_small = max(ligand_extent + 2×1.5 nm, 3.2 nm)`，`L_large = L_small + 1.0 nm`；
  若 minimum-image 条件不满足则只增大，不得缩 cutoff。每盒普通离子按 0.15 M + 中和
  规则生成，reserved dummy 单独计数。

  **每个 case 的执行顺序**：

  1. 从独立 output 目录建盒；禁止复用本仓库 `output/` 或 `memtest/output_*`。
  2. 落盘并重读 spec，核对 dummy 在 λ=1 为 0 e、LJ/mass 与同号物理离子一致。
  3. 对生产 λ 表逐态创建 Context；记录全体系/ligand/co-ion 电荷、势能、最大力、
     ligand–co-ion minimum-image 距离、co-ion 水配位数和 restraint 能量。
  4. 短平衡后计算 charging ΔG；同一 case 至少 3 个 seed。先跑 1 seed 的 pilot，出现
     NaN、PME error、距离越门或水合异常就停止，不烧完其余 seed。
  5. 另跑 `co_annihilation_experimental` 与 neutralizing-plasma 仅作诊断对照；结果标
     `diagnostic_only`，严禁加 APBS/Rocklin 到 charge-transfer 数值上。

  **硬验收**：逐 λ 总电荷误差 ≤ `1e-6 e`；ligand/co-ion 电荷满足 §2.1；无非有限能量
  或力；全程距离 ≥ §13.1；两盒 charging ΔG 差同时满足 `|ΔΔG| ≤ 2σ_combined` 与
  `|ΔΔG| ≤ 1.0 kcal/mol`。任一电荷符号失败，C1 不通过。输出固定为
  `validation/c1_waterbox/<case>/report.json`、`timeseries.csv`、`u_kn.npz`、
  `coalchemical_ion_spec.json` 和 `summary.json`；summary 记录输入 SHA256、命令行、seed、
  软件版本和明确的 `passed`。

  **进度（2026-08-06，用户在真机跑完矩阵）**：`q=±1` 的 4 个硬性 case 全部通过——
  `compare_box_Na.json`/`compare_box_Cl.json` 的 `|ΔΔG|` 分别 0.137 / 0.263
  kcal/mol，远低于 2σ 与 1.0 kcal/mol 两道阈值；四个 case MBAR 均
  `converged: true`（`min_overlap` 0.07~0.10）。**尚未勾选**：目前只跑了 1 个
  seed，原稿这里写的是"至少 3 seed"；用户已确认暂不补，正式关闭前仍需补跑。

  额外测的 Ca²⁺（+2，不在硬性矩阵内）不通过：MBAR `converged: false`
  （`min_overlap` 0.009~0.013），charging ΔG 算出 ~158 kcal/mol。查出根因是
  `runabfe._insert_reserved_coalchemical_ion_dummies` 对 `count>1` 的多个 dummy
  没有约束彼此的最小距离——方盒里"离配体最远的 N 个点"天然会挤在同一个远角，
  实测两个 dummy 只相距 0.18~0.43 nm，λ→0 时同号带电近乎贴脸。这是本仓库第一次
  在真实体系上测 §2.2"多个单价 co-ion 分摊"，一测就测出来。**已修**：新增
  `abfe_core.COION_COION_MIN_IMAGE_INITIAL_NM`（进 provenance，
  `ACCEPTANCE_THRESHOLDS_VERSION` 1→2），选点算法改为贪心 farthest-first + 跳过
  与已选 dummy 太近的候选，找不满就 fail closed；
  `tests/test_solvent_leg_coion_builder.py` 加了两条回归（复现-修复对照 + 数量够
  但挤在一起时 fail closed）。

  **复核（2026-08-06 稍晚）：几何 bug 修好了，但那不是 Ca 不收敛的（主要）根因，
  之前的判断要修正。** 用户重跑后 `Ca_large` 两个 dummy 相互距离从 0.18 nm
  变成 1.99 nm（确认修复生效），但 charging ΔG 几乎没变（658→663 kJ/mol），
  MBAR 仍 `converged: false`（`min_overlap` 仍 ~0.01）。`Ca_small` 这次没被
  真正重建（`build_manifest.json` mtime 早于代码修复时间），所以
  `compare_box_Ca.json` 这次的 `passed: true` 是新旧代码各一半拼出来的，不能
  当作"修复后通过"的证据。

  把 `u_kn.npz` 逐 λ 相邻态的平均能量差摊开看（纯读盘算术）：Ca 每步
  （Δλ_coul=0.1）平均位移 ~27~117 kJ/mol，全程都大、平滑变化，不是某一步
  异常；Na/Cl 每步只有 ~7~35 kJ/mol，比值约 3~4 倍，量级上更像"配体电荷
  Z² 标度"（Z=2 是 Z=1 的 4 倍）而不是"两个 dummy 靠太近"。更可能的解释：
  这条路线把配体的 1 个二价电荷拆成 2 个独立放置的单价 dummy——"1 个集中的
  2 价点电荷"自能（∝Z²=4）与"2 个分开摆放的 1 价点电荷"自能之和
  （∝1+1=2，另加一个较小的正 mutual 项）本来就不相等，这个差值是**真实的、
  该在的**电荷合并/拆分自由能，只是量级太大，11 个均匀 λ 点给不出足够
  overlap——是这条方案对 `|q_L|≥2` 需要更密 λ 表才能收敛，不是几何 bug。

  **结论**：几何修复本身该留着（防止真正的贴脸静电灾难），但不能说"Ca 现在
  该收敛了"。Ca 依然是未收敛、不可报数的状态，是本方案对多价配体的已知统计
  限制，不是这轮能顺手解决的。Atenolol 用不到 `|q_L|≥2`，**不追加投入**；
  以后真需要多价配体时按"加密 λ 表"处理。不影响 C1 本身（Ca 非硬性要求），
  但记录在案，供以后任何 `|q_L|≥2` 配体复用这条路径时参考。

- [ ] C2. protein-free lipid slab 测试。

  **输入要求**：使用与目标膜复合物相同的脂质力场、水模型、离子模型、cutoff、PME 与
  dispersion protocol，构建无蛋白、无口袋的 lipid–water slab。至少两种水层厚度
  （保持 XY 面积和每叶脂质数相同，只改 Z/水数），每个体系预先平衡并保存来源与 SHA256。
  不要从含蛋白的 `memtest/` 轨迹删掉蛋白后直接冒充平衡 slab。

  **应写的验证代码**：新增 `tools/validation/validate_charge_transfer_lipid_slab.py`，复用
  生产 GROMACS loader、组成分类、膜法向/PBC、co-ion identity/restraint 和 charging
  builder。只允许脚本层新增 slab 专用观测量汇总；禁止复制另一套离子选择判据。

  **运行矩阵**：`2 个水层厚度 × 2 个合法 bulk-water 初始位置 × 3 seeds`；先完成每格
  1 seed pilot，再扩展。固定同一个 `q=+1` 或 `q=-1` probe ligand/charge group，位置始终
  在水相；若目标项目未来同时支持两种符号，再镜像补另一个符号。co-annihilation 只跑
  小预算负对照，不进入通过判据。

  **逐帧保存**：盒矢量、APL、P–P 膜厚、密度剖面、co-ion 相对中面 z、到最近磷原子/
  ligand 的 minimum-image 距离、水配位数、restraint 能量、总电荷和各 λ 势能。slab
  没有蛋白，不能硬塞进要求 protein RMSD 的完整 `evaluate_membrane_quality_gate()`；写
  slab 专用 gate，只复用其中有定义的 APL/膜厚/周期镜像/水侵入判据。

  **硬验收**：所有 λ 总电荷误差 ≤ `1e-6 e`；co-ion 全程在同一侧 bulk water 且满足
  §13.1；无跨周期跳入另一叶片、无 restraint runaway；两种水层厚度及两个初始位置的
  charging ΔG 差均 ≤ `max(2σ_combined, 1.0 kcal/mol)`；纯脂 slab 的末段 APL 与该力场
  文献值差 ≤ 3%，膜厚无系统漂移。输出到 `validation/c2_lipid_slab/<case>/`，总表
  `summary.csv/json` 必须能从一行追到输入、spec、seed 与轨迹。

- [ ] C3. λ=1/λ=0 endpoint 能量与力测试。

  **不要把 B3 的合成单元测试当 C3。** C3 要在 C1 的真实小水盒和 C2 的真实 slab 上，
  用生产建出的 System 与一个**独立直接改粒子参数**的 reference System 对照。

  **应写的代码**：新增 `tools/validation/compare_charge_transfer_endpoints.py`。reference
  builder 不调用 `co_alchemical_charge_offset_plan()`，否则 production/reference 会共享
  同一个错误。它从 λ=1 base System 深拷贝后直接设置端点粒子电荷；vdW reference 直接
  删除 ligand–environment LJ/静电耦合，但保留 ligand internal、Boresch、co-ion
  restraint 和环境–环境项。固定同一 positions/box、同一 platform precision，并在
  `getState(getEnergy=True, getForces=True, groups=...)` 下按 force group 和总量比较。

  **必须对照四个物理端点**：

  1. charging λ=1：ligand fully charged，co-ion neutral；应等于物理 λ=1 base System。
  2. charging λ=0：ligand electrostatics off，co-ion fully charged。
  3. vanishing λ=1：必须与 charging λ=0 同一个 Hamiltonian；这是两 stage 的接缝门。
  4. vanishing λ=0：ligand–environment electrostatics 与 LJ 严格为零；co-ion 仍 fully
     charged，ligand internal 与所有应保留 restraint 不变。

  每个端点至少检查平衡轨迹抽取的 10 帧，不得只挑一帧。能量相对差 ≤ `1e-5`，逐原子力
  `max|ΔF| ≤ 1e-3 kJ/mol/nm`，λ=0 ligand–environment 非键能绝对值 ≤
  `1e-6 kJ/mol`，charging λ=0 与 vanishing λ=1 的能量/力也必须满足同一容差。
  报告列出最大差所在 frame、atom、force group，禁止只给 pass/fail。测试入口建议为
  `tests/test_charge_transfer_real_endpoints.py`（读取小型冻结 fixture，不依赖 GPU）；完整
  轨迹报告放 `validation/c3_endpoints/report.json`。

- [ ] C4. 带电膜体系 complex/solvent 双腿 smoke test。

  **前置条件**：B5、C1、C2、C3 全通过；复合物输入已经在建系阶段额外含 `|q_L|` 个
  λ=1 电荷为 0 的同号 ion-shaped dummy。普通盐/中和离子不能拿来顶替。当前中性
  Atenolol `memtest` 只能证明工程主链，不能给本条打勾。

  **配置制作**：复制目标 charged membrane 的生产配置为
  `validation/c4_smoke/config.json`，只降低采样预算；必须保留
  `system_type=membrane`、真实 membrane declaration、
  `charge_treatment=co_alchemical_charge_transfer`、与生产一致的
  `dispersion_protocol`、PME/cutoff、水盐模型和 Boresch 定义。smoke 可将质量门设为
  `advisory`，但必须在 declaration 写清 `equilibration_shortfall_justification`；这只允许
  工程验证，产物必须带 `production_qualified=false`。建议 pilot 用 4 个 charging λ、
  4 个 vdW λ、每窗 5k–20k 步；不要用 smoke 数值报告正式 ΔG。

  **命令模板**（路径和配体名替换成真实输入；第一次用全新目录，第二次只测 resume）：

  ```bash
  cd /home/ruigengji/ABFE_IBS/Atenolol-rank11
  source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh
  mamba activate openmm_dev

  python runabfe.py --config validation/c4_smoke/config.json \
    --output validation/c4_smoke/output --reset

  python runabfe.py --config validation/c4_smoke/config.json \
    --output validation/c4_smoke/output --resume
  ```

  **检查顺序**：先 complex charging，确认无 NaN/PME/restraint runaway；再 stage 2，确认
  co-ion 在进入和离开 stage 2 时都保持 fully charged；最后 solvent leg。第二次 resume
  必须命中同一 co-ion index/fingerprint，并复用已完成窗口，不允许重新 selector 或重算
  `u_kn`。另复制整个 output 到临时目录，单独篡改一份 spec 的 fingerprint，确认 resume
  在建 Context 前 fail closed；绝不在唯一证据目录上做破坏性测试。

  **硬验收产物**：

  - `output/checkpoints/coalchemical_ion_spec.json`；
  - `output/solvent_leg/checkpoints/coalchemical_ion_spec.json`；
  - 两腿 `decharging/decharging_pme_u_kn.meta.json`；
  - 两腿 stage1/stage2 protocol key 与窗口 manifest；
  - `run_provenance.json` 中 `co_alchemical_ions.complex/solvent`；
  - 两腿 `final_results.json` 和总 `final_binding_results.json`，均标 smoke/not production；
  - `first_run.log`、`resume.log` 与机器/GPU/seed/耗时摘要。

  所有 λ 电荷正确、全程有限、complex 用膜 barostat、solvent 用各向同性 barostat、两腿
  dispersion protocol 相同、stage 2 co-ion 不恢复中性、resume 身份完全一致，才算通过。

- [ ] C5. co-ion 位置与 restraint 敏感性测试。

  **只在 C4 已跑通后做**，并冻结除“位置/restraint”外的一切输入。不得手改
  `coalchemical_ion_spec.json`：每个变体都从 base System 重新调用
  `build_co_alchemical_ion_identity()` 生成新 spec，使用全新 output 目录和 seed 表。

  **预注册矩阵**：

  - 位置：至少 3 个均满足 §13.1 的 bulk-water 位点——同侧近端、同侧远端、对侧水层；
    不把头基层/孔道/结合口袋位置混进“合法位置”组。另加 1 个故意违规位置，只用于
    证明 placement gate 会在采样前拒绝。
  - restraint：基线 `k=100, r0=0.5`；弱/宽 `k=50, r0=0.7`；强/窄
    `k=200, r0=0.3`（单位 kJ/mol/nm²、nm）。若要改这些点，必须在开跑前写入
    `validation/c5_sensitivity/design.json`，看过结果后禁止补点挑结论。
  - 每个合法组合 complex/solvent 两腿都跑，至少 3 seeds；pilot 可先用位置三点 + 基线
    restraint，确认没有明显失败后再扩矩阵。

  **应写的代码**：新增 `tools/validation/run_coion_sensitivity_matrix.py` 生成独立配置/spec/
  命令，不直接执行生产目录；新增 `tools/validation/analyze_coion_sensitivity.py` 汇总两腿
  ΔG、净 ΔG_bind、统计误差、位置时间序列、hydration、restraint 能量与触壁比例。
  分析必须读取 B5 provenance 中的 fingerprint 连接每个结果，禁止用目录名猜参数。

  **需要回答的三个问题**：

  1. 换合法 bulk-water 位置后，complex/solvent 各腿与净 ΔG 是否在预注册容差内？
  2. 改 k/r0 后两腿 restraint 自由能是否抵消？若不抵消，给出显式修正或判路线失败，
     不能用“逐 λ 相同所以一定抵消”代替数据。
  3. neutral dummy 是否在预平衡中吸附配体/膜，charged endpoint 是否维持正常水合？

  **硬验收**：所有合法 case 全程满足 §13.1、restraint 非有限值为 0、触壁帧比例与最大
  restraint 能量均落盘；位置或 restraint 变体相对基线的净 ΔG_bind 差必须同时满足
  `|ΔΔG_bind| ≤ 2σ_combined` 与 `≤ 1.0 kcal/mol`。任何系统性位置趋势、跨叶片跳跃、
  hydration 崩塌或两腿不抵消都判失败。结果固定写
  `validation/c5_sensitivity/design.json`、`cases/<case>/...`、`summary.csv`、
  `summary.json` 和 `plots/`；最终 summary 明确列出 excluded/failed case 及原因，不得只
  汇报最好的一组。

### Phase D：生产资格

- [ ] D1. 关闭 P1-19/P1-22/P1-23 不确定度问题。
- [ ] D2. 完成 Boresch 真实键和二面角更新门。
- [ ] D3. 至少 3 个独立重复。
- [ ] D4. 公开 benchmark。
- [ ] D5. 完整 provenance 和复现实验脚本。

---

## 12. Definition of Done

只有同时满足以下条件，才可声明“支持膜受体–配体 ABFE”：

- [ ] 膜复合物使用正确半各向异性/膜恒压协议完成平衡。
- [ ] 膜体系不使用未经验证的均匀密度 LJ LRC。
- [ ] λ=1/λ=0 Hamiltonian 端点通过能量和力验收。
- [ ] 带电配体在所有 λ 保持总电荷恒定，或使用经过验证且不重复计数的 APBS 路线。
- [ ] co-ion 在复合物腿和溶剂腿均显式存在、受控、进入 PME 和缓存指纹。
- [ ] 溶剂腿离子计数、盒子、co-ion 和 ordinary ions 身份可审计。
- [ ] Boresch 锚点、符号、attachment 与标准态修正闭环。
- [ ] 膜结构质量门通过。
- [ ] overlap/ESS 和修正后的不确定度门通过。
- [ ] 至少 3 个独立重复一致。
- [ ] 至少一个公开/可追溯膜受体 benchmark 通过。
- [ ] 最终结果能明确回答：电荷怎么处理、LJ 怎么处理、膜怎么控压、溶剂腿怎么构建、
  哪些修正被应用、哪些没有被应用。
- [ ] 存量问题 MEM-00h（ligand–env 与 env–env cutoff 不一致）已单独收口。
- [ ] §3.0 的空腔填充迟滞已实测并在容差内，或已明确记入已知局限。

---

## 13. 阈值默认值（✅ 2026-07-30 已全部落成常量并进 provenance）

没有数就没法写 fail-closed 检查，也没法判验收。以下是起始提案，可以改，
但**必须在 Phase A 结束前落成常量并进 provenance**，不许运行时凭感觉判。

**已完成**：全部数值落在 `abfe_core.py`（`ACCEPTANCE_THRESHOLDS_VERSION = 1`），
`acceptance_thresholds_payload()` 的快照无条件写入 `run_provenance.json`
的 `acceptance_thresholds`，所以每份结果都能回答"当时用的哪套阈值"。
测试：`tests/test_dispersion_and_forcefield_protocol.py`。
⚠️ `COION_LIGAND_MIN_IMAGE_RUNTIME_NM = 1.2` 与 `ibs_engine.SOFTCORE_CUTOFF_NM`
是两处各一份（`abfe_core` 在下层不能反向 import），已有交叉检查测试防止各改一半。

### 13.1 co-ion 几何

| 量 | 提案阈值 | 判定时机 |
| --- | --- | --- |
| co-ion ↔ 配体 minimum-image 距离 | 初始 ≥ 1.6 nm，全程 ≥ 1.2 nm（= softcore cutoff） | 选择时 + 每帧诊断 |
| co-ion ↔ 蛋白重原子最近距离 | ≥ 1.2 nm | 同上 |
| co-ion ↔ 膜中心平面 \|z\| | ≥ 3.0 nm | 同上 |
| co-ion ↔ 最近磷原子（向水侧） | ≥ 1.0 nm | 同上 |
| co-ion 首层水配位数（O within 0.32 nm） | Na⁺ ≥ 5，Cl⁻ ≥ 6 | 每帧诊断 |
| flat-bottom restraint | 平坦半径 r₀ = 0.5 nm，外壁 k = 100 kJ/mol/nm² | 构建时 |

（对比现状 MEM-00d：当前是 k = 25 的纯谐振子、无平坦区。）

### 13.2 数值自洽

- 总电荷守恒：`|Σq(λ) − Σq(λ=1)| ≤ 1e-6 e`，**逐 λ**（含中间态）检查。
- 配体电荷：`|Σq_lig(λ) − λ·q_L| ≤ 1e-6 e`。
- 端点 Hamiltonian 对照：能量相对差 ≤ 1e-5，逐原子力 `max|Δ| ≤ 1e-3 kJ/mol/nm`。
- λ=0 端 ligand–environment 静电与 LJ：`|E| ≤ 1e-6 kJ/mol`（严格零，不是"很小"）。
- GROMACS ↔ OpenMM 单点能：分项相对差 ≤ 1e-4，总能绝对差 ≤ 0.1 kJ/mol。

### 13.3 膜质量门（末段 ≥ 20 ns）

- APL 漂移 ≤ 0.2 %/ns（判 raw APL），且与该脂质力场文献值差 ≤ 3%
  （判**蛋白横截面校正后**的 `apl_protein_corrected_nm2`，见 §9 与 §0.5.9；
  含蛋白膜拿 raw APL 比纯脂文献值必然偏大）。
- 双层厚度（P–P）漂移 ≤ 0.05 nm / 20 ns。
- 蛋白骨架 RMSD ≤ 0.30 nm，跨膜倾角漂移 ≤ 5°。
- 口袋 RMSD ≤ 0.20 nm；配体重原子 RMSD ≤ 0.25 nm。
- 疏水核内异常水分子数：与不含配体的对照体系相比无系统性增加。

### 13.4 结果验收

- 3 个独立种子的 ΔG_bind 跨重复标准差 ≤ 1.0 kcal/mol，且与腿内统计误差同量级
  （若跨重复散布远大于报出 σ，先回 `docs/TODO.md` P1-19，不是继续加窗口）。
- 公开 benchmark：≥ 5 个配体，MAE ≤ 1.5 kcal/mol，无 \|误差\| > 3 kcal/mol 的离群。
- §3.0 迟滞：正反向 / 双起点 stage 2 的 ΔF 差 ≤ 2σ。

---

## 14. 风险登记与放弃判据（原稿缺；没有 kill criteria 的计划会无限期拖）

| ID | 风险 | 早期信号 | 缓解 | 放弃/改道判据 |
| --- | --- | --- | --- | --- |
| R1 | 埋入式口袋的空腔脂质填充迟滞（§3.0） | stage 2 双起点 ΔF 分叉、口袋脂质数出台阶 | 加窗口、延长采样、REMD 交换 | 迟滞 > 2 kcal/mol 且加 4× 采样无改善 → 换水相可及口袋的体系 |
| R2 | 已有 co-annihilation 与新 charge-transfer 并存导致双路径（MEM-00a/00b） | 带电配体跑通了但没人知道走的哪条 | Phase A 先裁决、旧路径 fail closed | 无法在一个 PR 内收口 → 先只做中性配体膜体系 |
| R3 | u_kn 与动力学选到不同 co-ion（MEM-00c） | 能量重算与在线 f_k 系统性偏移 | 身份写 manifest、重算时只读不选 | —（这是必须修的 bug，不接受绕过） |
| R4 | 非键协议无法在 OpenMM 复现（force-switch，§1.3） | 单点能对照分项差 > 1e-4 | 改用 Amber 系脂质力场 | 同族力场也对不上 → 停，先解决 MEM-00h |
| R5 | 膜平衡时间不够，质量门反复不过 | APL 持续漂移 | 延长预平衡到 100 ns+ | 200 ns 仍漂 → 回建系（组成/叶片数错） |
| R6 | 成本超预算（§15） | Phase A 实测 ns/day 后即可判 | 减窗口数不可取；减重复数或换小体系 | 单个 ΔG_bind > 预算上限 → 缩体系规模 |
| R7 | 膜工作顺手改动了可溶路径的数值 | §7.7 基线三个数变了 | 膜代码全部走 `system_type` 分支 | —（直接回滚） |

---

## 15. 成本与资源（✅ 2026-08-04 已按**实测**关闭，不再是估算）

数据来源：`memtest/output_membrane_100ns` 第一次端到端跑通的那一轮
（POPC 膜 + PROA + 中性 Atenolol），时间戳全部取自 `pipeline.log`。
⚠️ 这一轮的**数值结果**有已知缺陷（见 §0.5.13），但**耗时与占用**不受影响。

- [x] **Phase A 实测（不是拍脑袋）**：
  - 膜复合物 **45354 原子**，预平衡后盒 5.9417 × 5.9417 × 12.4657 nm；
    溶剂盒 **12796 原子**，盒边 4.052 nm（padding 1.5 nm）。
    对照：可溶体系 `solv_ions.gro` 约 5 万原子 —— 本膜体系并不比它大，
    原稿担心的"2–3×"没有出现（小膜片 90 脂）。
  - GPU：单卡 16 GB（日志 `total=16303 MiB`）。
  - **预平衡 476 ns/day**（5e6 步 / 30.2 min，NPT + `MonteCarloMembraneBarostat`）。
- [x] 逐阶段实测（每窗口 250k 步 = 0.5 ns，2 fs）：

  | 阶段 | wall-clock | 态数 | 总采样 | 聚合 ns/day |
  | --- | --- | --- | --- | --- |
  | 预平衡 100 ns（5e7 步） | **≈ 5.0 h** | 1 | 100 ns | 476 |
  | 复合物 Stage 0 attachment | 4.9 min | 4 | 2.4 ns | 708 |
  | λ 预优化（stage1+2） | 4.5 min | — | — | — |
  | 复合物 Stage 1 去电荷 | 19.2 min | 12 | 6.0 ns | 449 |
  | 复合物 Stage 2 去 VDW | 52.3 min | 23 | 11.5 ns | 316 |
  | 溶剂腿 预平衡 + 预优化 | 15.1 min | — | — | — |
  | 溶剂腿 Stage 1 | 8.5 min | 12 | 6.0 ns | 1018 |
  | 溶剂腿 Stage 2 | 19.6 min | 23 | 11.5 ns | 846 |
  | **合计（不含 100 ns 预平衡）** | **2.07 h** | | 37.4 ns | |

  注：REMD 是 12/23 个副本在**同一张卡上分时**，所以表里给的是**聚合**吞吐
  （总采样 ÷ wall-clock），不是单副本速度 —— 估成本要用聚合值。
- [x] **串行 wall-clock 关键路径**（比 GPU-hours 总量更能决定能不能做）：

  ```text
  单次 ΔG_bind ≈ 5.0 h（100 ns 预平衡） + 2.07 h（两条腿全部炼金） ≈ 7.1 h
  ```

  **瓶颈是预平衡，占 70%**，而且它是**一次性**的：同一体系做 3 重复时可以共用
  同一条预平衡 + rebalance，只在 stage 层换种子 ⟹
  3 重复 ≈ 5.0 + 3 × 2.07 = **11.2 h**（各自重新预平衡则是 21.3 h）。
- [x] **磁盘占用与保留策略**（实测 `du`）：

  | 产物 | 大小 | 保留 |
  | --- | --- | --- |
  | `pre_equilibration.dcd`（5001 帧 × 45354 原子） | **2.6 GB** | 过完 §9 质量门后抽稀（保留末段 20 ns 或 100 ps/帧）|
  | `decharging/`（u_kn / energies `.npy`） | 312 MB | **长期保留**——重算 ΔG 的唯一输入 |
  | `checkpoints/` | 49 MB | 保留到该 stage 关闭 |
  | `system_native.xml` | 22 MB | 长期（重建 Hamiltonian 用）|
  | 溶剂腿全部 | 92 MB | 同上 |
  | **单次运行合计** | **3.1 GB** | 抽稀预平衡轨迹后 ≈ 0.5 GB |

- [x] **是否超预算（§14 R6 的早期信号）**：单个 ΔG_bind 7.1 h 串行、
  3 重复 11.2 h —— 远低于任何合理上限，**R6 不触发**，不需要缩小体系。
  真要压缩，压的是预平衡（一次性且可共用），不是窗口数或采样时长
  （砍那两个会直接打回 R1/R5）。
- [ ] ⚠️ **这张表的有效期**：它绑定当前 λ 阶梯（stage1 12 态 / stage2 23 态、
  每窗口 0.5 ns）。§6.4 明确要求 charge-transfer 用 pilot **重估** stage1 窗口数
  （两个位点同时改电荷，相邻 ΔF 与中性配体不可比），窗口数一变成本要按上表
  的聚合 ns/day 重算 —— 重算方法已经有了，这一条留着提醒别沿用旧窗口数。

---

## 16. 交付物、可追溯性与登记（原稿缺）

- [ ] **给每条 checkbox 分配 ID**：本文件统一用 `MEM-xx` 前缀，
  与 `docs/TODO.md` 现有的 `P0-/P1-/P2-/BOR-/VAL-` 体系并列、不冲突。
  没有 ID 的条目无法在提交信息、测试名和 handoff 里被引用。
- [ ] **在 `docs/TODO.md` 登记一条主条目**指向本文件，
  否则这份计划在主 TODO 流程之外，会被后续会话整体漏掉。
  （本仓库规定每轮都要更新 `docs/TODO.md`。）
- [ ] 每个 Phase 结束写一份 handoff 到 `docs/handoffs/`，
  格式对齐现有 `BORESCH_DIHEDRAL_SIGN_HANDOFF.md`：做了什么、验证证据、
  被拒绝的方案及原因、下一位接手者的禁区。
- [ ] **所有需要跑的验证都以"可直接执行的命令清单"交付**（含 conda env、
  完整参数、预期产物路径、预期耗时），由用户在计算节点执行；
  文档里不许出现"跑一下看看"这种没有具体命令的步骤。
- [ ] 每一项声称"已验证"的条目，必须同时给出**证据位置**
  （测试名 / 日志路径 / JSON 产物），只打勾不算。
- [ ] 本文件与 `docs/TODO.md` 的关系写在开头：
  本文件是膜专项的**设计与验收清单**，日常待办仍以 `docs/TODO.md` 为准。

---

## 17. 与现有工作的排序依赖（原稿只在 Phase D 提了一句）

### 17.0 当前主线（2026-08-04 拍定，按此顺序推进）

**进度（2026-08-05）**：② B3、③ B4 已完成；④ B5 **代码实现完成、定向测试
181 passed，但仍未勾选**，等待 `./tests/run_offline_tests.sh -q` 全套 0 failed 的最终证据。
⑤–⑨ C1–C5 均保持未完成，尚未执行，必须按 §11 的证据门逐项关闭。Boresch
`np.int64` JSON 序列化修复是独立事项，不计入 B5。
① `memtest/output_membrane_5ns` 已于 2026-08-05 完成中性 Atenolol
complex/solvent 双腿端到端工程 smoke test（Resume、Boresch attachment、双 λ、
溶剂腿和最终汇总均完成）；`ΔG_bind = -26.49 ± 1.77 kJ/mol`。该运行只证明
中性工程主链可运行，不关闭带电 co-ion 验收、C1–C5、膜质量门或 §7.7 可溶性回归。
同日在真实 resume 日志里抓到并修了 **RESUME-FP-01**（见 §0.5.14）；该修复已完成，
与本次 5ns 运行的结果记录分开。

```
① **中性 Atenolol 已跑完整个 Stage 2 + 溶剂腿**（`output_membrane_5ns`）
       ↓  这是膜 ABFE 的**工程 smoke test**：完整主链 / 缓存 / REMD / LJ / 溶剂腿
       ↓  已跑通。**不把它当统计学最终结果**（配体净电荷 = 0，见 §0.5.1 MEM-00c）
② B3：真正的 PME charge-transfer 哈密顿量  ✅ **已完成 2026-08-04（含 MEM-00d）**
       ↓  身份前置条件已就位（MEM-00c 已完成"选一次 / 冻结 / 六个消费点只读核对"）
       ↓  MEM-00d 一并解决：restraint 换成 flat-bottom + **锚点相对**（随体系缩放）。
       ↓  身份指纹协议版本 1 → 2，旧 spec/缓存自动作废（刻意）。
③ B4：溶剂腿 builder（reserved co-ion dummy）  ✅ **已完成 2026-08-05**
       ↓  `_insert_reserved_coalchemical_ion_dummies`：摘最远的 |q_L| 个水，换成
       ↓  同号中性 ion-shaped dummy；身份识别/restraint/charging 复用 B3 的实现。
       ↓  两腿都能跑了，热力学循环闭得上——但**只在合成 topology 上单元测过**，
       ↓  没有真实带电配体上机验证（Atenolol 净电荷=0，这条路径本仓库测不出来）。
④ B5：cache / resume / provenance  **代码实现完成，定向测试 181 passed；待全套离线测试 0 failed**
       ↓  暂不勾选；Boresch `np.int64` JSON 修复单独记录，不计入 B5
⑤ C1：带电配体小水盒验证
       ↓
⑥ C2：protein-free lipid slab
       ↓
⑦ C3：真实体系端点能量与力
       ↓
⑧ C4：带电膜体系 complex/solvent 双腿 smoke test
       ↓
⑨ C5：co-ion 位置与 restraint 敏感性
```

**不属于当前主线、明确暂不投入**：新的 APL 指标、新的膜弛豫硬门、更多 Stage 0 λ、
重调 Boresch 力常数、更多膜专用诊断、直接做三重复、直接做 benchmark。

⚠️ **MEM-00h（cutoff 不一致）走旁线**：独立立项、独立 PR，与 B3 并行分析即可，
不许和 charge-transfer 混在一个改动里（否则哈密顿量同时变两处，误差无法二分定位）。
但在 Phase C endpoint 验收与生产运行前必须收口。

### 17.1 进生产前必须收口的（⚠️ 这些**只**阻止"进生产"，不阻止 B3/B4/B5 方法开发）

- [ ] `docs/TODO.md` P1-19 / P1-19b（σ 口径与跨运行散布）——
  已知统计限制。膜体系上的实测见 §0.5.12：五次 attachment 的样本标准差 0.716
  （≈7× 单次 σ），但**去掉那一次显著离群后只有 0.158（≈1.6×）**，属正常范围。
  所以它能说明"单次 σ 不代表跨运行散布"，**不能**说明"σ 系统性低估十几倍"。
  留到真正做三重复 / benchmark / 生产验收时按 §13.4 收口，
  **现在不处理，也不让它拽住工程闭环。**
- [x] `docs/TODO.md` P1-23（σ 采纳路径 fail-open，真 bug）—— **已于 2026-08-03 修**：
  采纳 σ 后重算端点 σ 门并重判 `converged`。该标志仍默认关闭，故落盘基线不变。
- [ ] P1-22（vdW/stage2 帧选择与 σ 口径）——至少要有结论，即使结论是"维持现状"。
- [ ] MEM-00h（cutoff 不一致）——存量问题，**必须单独立项、单独 PR**，
  不许和膜改造混在一起改，否则出问题无法二分定位。
- [x] MEM-00c（co-ion 身份可能在动力学与 u_kn 之间漂移）—— **已于 2026-08-04 修**：
  身份选一次并冻结落盘，dynamics / replicas / u_kn / resume 全部只读核对，
  无 spec 即 fail closed。B3 的前置条件已就位。详见 §0.5.1 MEM-00c 与 `docs/TODO.md`。

允许并行推进的：§1.1 协议冻结、§3.3 膜输入准备、§9 膜平衡（这些是 CPU/建系工作，
不依赖上面的估计量修复）。

---

## 18. 本次补充（2026-07-29）改了什么

为便于复核，本轮相对原稿的实质性改动：

1. 新增 §0.5：**用 `file:line` 核对了仓库现状**，发现生产路径上已有
   co-annihilation 实现（MEM-00a/00b），与第 0 节选定的 charge-transfer 冲突；
   并列出 restraint 形式、身份重选、cutoff 不一致等 7 条与计划矛盾的存量事实。
2. §1.3 加**修正框**：原稿"膜体系一律禁用 LRC"是错的，判据应跟随所选脂质力场的
   原始参数化条件；同时指出 OpenMM 无 force-switch 这一硬限制。
3. 新增 §1.1.1 非目标、§3.0 膜内空腔填充迟滞、§13 阈值默认值、§14 风险与放弃判据、
   §15 成本估算、§16 交付物与登记、§17 排序依赖。
4. §7.7 把可溶体系回归基线钉成具体数字（181.00 / 157.84 / −5.535906 kcal/mol）。
5. §3.2 补 NPT/NVT、时间步与 HMR、位置限制释放、barostat 与 IBS 节奏耦合；
   §6.4 补 co-ion 与 IBS bias / λ 阶梯 / 收敛门的耦合；§9 补"怎么判"而不只是"看什么"。

**三项待拍板事项已于 2026-07-29 全部裁决**：

- [x] **MEM-00a：改成 charge-transfer。** co-annihilation 降级为
  `co_annihilation_experimental`，只作水盒 / lipid slab 的方法负对照，
  膜生产 preset 一律 fail closed，输出带 `experimental_not_for_production`；
  旧代码暂不删除，等 charge-transfer 端到端验收后再定退役；
  旧 charging checkpoint / u_kn / IBS 状态 / 预优化缓存全部作废不迁移；
  `GhostIonHandler` 直接标记退役。详见 §0.5.1 MEM-00a-1 ~ MEM-00a-5。
- [x] **脂质力场：按输入自动识别力场族**（amber → Amber 脂质 + isotropic LRC；
  charmm → CHARMM36 + force-switch）。识别不出 fail closed；
  charmm 分支因 OpenMM 无 force-switch 默认卡住，需定量论证才放行。详见 §1.1。
- [x] **首个体系：SERT，配体默认 +1，结合位点非深埋**（备选 GPCR）。
  新增 SERT 特有约束：结构性 Na⁺/Cl⁻ 必须排除出 co-ion 候选并单独登记监测，
  co-ion 的 bulk-water 判据必须排除胞外前庭与通道腔。详见 §1.1。

**Phase B 现在可以开工。** 第一批动作按 §17 的排序：先补 MEM-00c 的身份断言测试
（与膜无关，随时可做），并行推进 §1.1 协议冻结与 §3.3 SERT 膜体系建系。




• 要分成两件事做：先生成可信的 ±1 输入，再写验证脚本读取这些输入。验证脚本不能负责“伪造电荷”。

  ## 一、准备两个真实带电配体

  建议：

  - +1：质子化 Atenolol。不要修改现有中性 .itp；从明确加上胺氢的 +1 结构重新参数化。
  - −1：选择一个结构明确的阴离子，例如 acetate 或 benzoate 的去质子化形式，重新参数化为 −1。

  关键要求：

  1. 分子结构本身必须表达正确质子化状态。
  2. 从 SDF/MOL2 开始重新计算 AM1-BCC/RESP 电荷。
  3. -nc +1/-1 只是告诉参数化程序正确的分子总电荷，这是合法的；不能参数化完成后手改 .itp 电荷。
  4. 保存参数化工具版本、命令、输入结构和 SHA256。

  建议目录：

  validation/inputs/c1/
  ├── plus1_atenolol/
  │   ├── ligand.sdf
  │   ├── ligand.mol2
  │   ├── ligand.frcmod
  │   ├── ligand.itp
  │   ├── ligand.xml
  │   └── input_manifest.json
  └── minus1_probe/
      ├── ligand.sdf
      ├── ligand.mol2
      ├── ligand.frcmod
      ├── ligand.itp
      ├── ligand.xml
      └── input_manifest.json

  input_manifest.json 至少记录：

  {
    "molecule_name": "...",
    "canonical_smiles": "...",
    "formal_charge": 1,
    "protonation_state": "...",
    "parameterization_method": "GAFF2 + AM1-BCC",
    "parameterization_command": "...",
    "software_versions": {},
    "input_sha256": {},
    "partial_charge_sum_e": 1.0
  }

  ## 二、建立四个水盒

  矩阵：

  plus1_small
  plus1_large
  minus1_small
  minus1_large

  推荐：

  L_small = max(配体最大尺寸 + 3.0 nm, 3.2 nm)
  L_large = L_small + 1.0 nm

  两种盒子只能改变盒长和水数。以下必须一致：

  - TIP3P/TP3 水模型
  - AMBER/GAFF 参数体系
  - cutoff 和 switching
  - PME tolerance
  - 温度
  - 0.15 M 盐浓度
  - λ 表
  - 步数
  - seed

  每个盒子的粒子组成要注意：

  ### +1 配体

  物理配体：                 +1
  普通 Cl− 中和离子：        -1
  reserved Na-shaped dummy：  0（λ=1）

  转换过程中：

  λ_coul=1：ligand +1，co-ion 0
  λ_coul=0：ligand  0，co-ion +1

  ### −1 配体

  物理配体：                 -1
  普通 Na+ 中和离子：        +1
  reserved Cl-shaped dummy：  0（λ=1）

  转换过程中：

  λ_coul=1：ligand -1，co-ion 0
  λ_coul=0：ligand  0，co-ion -1

  普通反离子与 reserved dummy 是两个不同粒子，不能混用。

  reserved dummy 应通过现有生产 builder 插入：

  runabfe.build_and_cache_solvent_leg(
      ...,
      charge_treatment="co_alchemical_charge_transfer",
  )

  不要在验证脚本里自己复制“删除最远水并插入 dummy”的逻辑。

  ## 三、脚本的职责

  新增：

  tools/validation/validate_charge_transfer_waterbox.py

  它应该分为以下步骤。

  ### 1. 读取并核验输入

  必需参数：

  --system-xml
  --topology
  --positions
  --ligand-indices
  --charge-sign
  --box-edge-nm
  --n-steps-per-state
  --seed
  --output-dir

  启动后立即检查：

  - 配体部分电荷之和接近声明的 +1 或 −1
  - System 中恰好有一个对应的 reserved neutral dummy
  - dummy 的初始电荷严格为 0
  - dummy 的 mass/LJ 与同号物理离子模板一致
  - 盒子最短边大于 2 × cutoff
  - 配体与 dummy 初始 minimum-image 距离至少 1.6 nm

  任一失败立即停止。

  ### 2. 使用生产接口冻结身份

  只调用一次：

  select_co_alchemical_ion_once(...)

  参数明确传：

  charge_treatment="co_alchemical_charge_transfer"

  随后：

  - 用 verify_co_alchemical_ion_identity() 重读核验
  - 将原始 spec 写到 coalchemical_ion_spec.json
  - 后续所有 Context 和 u_kn 都使用该 spec
  - 不允许再次选择 co-ion

  ### 3. 配置生产 Hamiltonian

  复制 System 后调用：

  configure_pme_ligand_charge_offsets(
      ...,
      lambda_name="lambda_coul",
      allow_charged_ligand=True,
      co_alchemical_ion_spec=spec,
  )

  再调用：

  charging_charge_conservation_report(...)

  必须使用生产 λ 表，例如：

  1.00, 0.90, 0.80, ..., 0.10, 0.00

  不要只检查两个端点。

  ### 4. 逐 λ 做静态检查

  每个 λ 设置：

  context.setParameter("lambda_coul", lam)

  记录：

  - 全体系电荷
  - ligand 电荷
  - co-ion 电荷
  - 总势能
  - 各 force group 能量
  - 最大力
  - ligand–co-ion minimum-image 距离
  - co-ion 水配位数
  - co-ion restraint 能量
  - 盒矢量

  至少断言：

  所有 λ 总电荷变化 ≤ 1e-6 e
  能量和力全部为有限数
  λ=1/0 电荷符合预期
  restraint force group = 6
  restraint 不含 lambda 参数

  ### 5. 短动力学

  先做单 seed pilot：

  最小化
  短 NVT/NPT 平衡
  每个 λ 短采样

  建议第一轮预算：

  最小化：直到收敛或 5000 iterations
  平衡：每态 20,000 步
  采样：每态 50,000 步
  保存间隔：500 步

  pilot 通过后再提高到清单指定预算。

  运行中每帧记录：

  step,time_ps,lambda_coul,total_charge_e,ligand_charge_e,coion_charge_e,
  potential_kJ_mol,max_force_kJ_mol_nm,ligand_coion_distance_nm,
  coion_water_coordination,restraint_energy_kJ_mol,box_volume_nm3

  ### 6. 通过生产重算路径生成 u_kn

  不能只用当前态能量拼矩阵。轨迹完成后调用生产的：

  TraditionalMBARAnalyzer.compute_u_kn(
      ...,
      co_alchemical_ion_spec=spec,
  )

  保存：

  u_kn
  n_k
  lambdas_coul
  lambdas_vdw
  temperature
  beta
  coion_fingerprint
  system_sha256

  到 u_kn.npz。

  ## 四、输出文件含义

  每个 case：

  validation/c1_waterbox/<case>/
  ├── report.json
  ├── timeseries.csv
  ├── u_kn.npz
  ├── coalchemical_ion_spec.json
  └── summary.json

  ### report.json

  保存完整证据：

  - 输入及哈希
  - System/Topology 身份
  - λ 表
  - 电荷守恒报告
  - 每态能量/力
  - 距离和水合统计
  - PME、cutoff、温度
  - spec fingerprint
  - 所有验收项及失败原因

  ### summary.json

  只保存机器可判定的最终结果：

  {
    "case": "plus1_small",
    "passed": true,
    "charge_conservation_passed": true,
    "finite_energy_force_passed": true,
    "coion_geometry_passed": true,
    "restraint_passed": true,
    "endpoint_charge_passed": true,
    "delta_G_charging_kJ_mol": 0.0,
    "uncertainty_kJ_mol": 0.0,
    "failure_reasons": []
  }

  passed 必须由各项结果计算，不能硬写。

  ## 五、实际执行顺序

  不要一次跑四组。按以下顺序：

  1. plus1_small，1 seed，CPU 静态检查
  2. plus1_small，1 seed，CUDA 短动力学
  3. minus1_small，1 seed
  4. plus1_large、minus1_large，各 1 seed
  5. 全部正常后扩展至每 case 3 seeds
  6. 比较同一电荷的小盒/大盒 ΔG

  最终 C1 通过条件：

  逐 λ 总电荷误差 ≤ 1e-6 e
  无 NaN/Inf
  最大力没有异常爆炸
  co-ion 全程满足距离和水合要求
  λ=0/1 电荷方向正确
  同一电荷的小/大盒 |ΔΔG| ≤ 2σ_combined
  同一电荷的小/大盒 |ΔΔG| ≤ 1.0 kcal/mol
  +1 和 -1 两条路线都通过

  最先要做的不是写采样循环，而是生成两套重新参数化、带来源清单的 +1/-1 ligand 输入。输入没有通过电荷和来源核验之前，验证脚本应拒绝启动。
