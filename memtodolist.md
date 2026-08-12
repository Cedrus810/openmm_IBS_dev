# 膜受体–配体 ABFE 当前行动清单

更新日期：2026-08-11（**C3 与 MEM-00h 已正式关闭（用户确认），进入 C4**。
C3-0~C3-4 全部跑过一轮；co-ion/ParameterOffset 归因诊断完成；C3 protocol v2
双层门重设计已实现；C2 的 C-seam switch 不一致已用"MEM-00h 双边归一化"
修复并在全部真实 GPU 数据上验证——A/B 100/100 + C/D 50/50，全部 150 帧一次
通过，C2 的 C-seam 力差回落到机器精度；`summary.json`/`mem00h_report.json`
两份 fail-closed 汇总产物已生成，均 `status=complete, passed=true`）  
状态：Phase B 工程实现基本完成；B5 已关闭。C1、C2、C3 已关闭；MEM-00h 已
关闭。当前进入 C4。

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

---

## 1. 当前做到哪里

已完成的工程能力不再逐项放在本清单中，完整证据见归档。当前状态摘要：

- B1：膜体系识别和 `MonteCarloMembraneBarostat` 已实现。
- B2：`charge_treatment` 配置和双计数 fail-closed 已实现。
- B3：PME co-alchemical charge-transfer Hamiltonian 已实现。
- B4：溶剂腿 reserved co-ion dummy builder 已实现。
- B5 已关闭（2026-08-09）：cache、resume、provenance 全套离线测试 0 failed，
  co-ion 隔离/缓存拒绝/resume 一致性逐项复核通过。
- 中性 Atenolol 膜体系 complex/solvent 双腿工程 smoke test 已跑通。
- C1 已关闭：Na/Cl 硬性验收通过；采用单 seed pilot，不补 seed；Ca 为已知统计限制且不阻塞。

当前主线：

```text
C2 无蛋白 lipid slab（进行中，差 v8 代表 case GPU pilot/质量门）
    ↓
C3 真实端点能量/力
    ↓
C4 带电膜双腿 smoke test
    ↓
C5 co-ion 位置/restraint 敏感性
    ↓
Phase D 生产资格
```

---

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

## 4. Phase C：物理与数值验证

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

### C4：带电膜 complex/solvent 双腿 smoke test

前置：B5、C1、C2、C3 全部通过。**2026-08-11：确认 C3 与 MEM-00h 正式关闭，
C4 已解锁。**

**C4 是接线 smoke test，不是生产自由能计算**——不追求收敛，不出最终
ΔG；全部产物必须标 `production_qualified=false`（第 6 步）。C2 的纯脂质
slab（无蛋白）不能代替这里的真实 receptor–ligand complex；C4 第一次真正
需要"膜 + 蛋白 + 带净电配体"这套完整组合。

用户指定的执行顺序（2026-08-11 登记，按顺序执行，不并行跳步）：

**受体/配体组合——阻塞第 1 步，待用户决定，本文档不擅自选择**（2026-08-11
现状普查，只读，未改任何文件）：

- **已有、可复用的**：`memtest/` 下有一个真实的 283 残基 GPCR 样受体
  （`Atenolol-rank1apo.pdb`/`Atenolol-rank1.pdb`，含 TM3 的 `DRY`、TM7 的
  `NPxxY` 保守基序，疑似热稳定化突变体，ICL3 可能被截短）已经嵌入真实
  POPC 膜（`memtest/step7_production.gro`：`PROA 1 / POPC 90 / Na+ 25 /
  Cl- 36 / TP3 9542 / Atenolol-rank11 1`，45354 原子），配上中性 Atenolol
  （`Atenolol-rank1.gjf` 的 QM 电荷计算用的是 `Charge=0`，即去质子化的
  仲胺；`memtest/Atenolol-rank11.itp` 41 个原子电荷加总 Σq≈0），
  `memtest/README_MEMTEST.md` 记录了这套中性体系已经跑通的完整
  complex/solvent 双腿工程 smoke test（膜恒压器、quality gate、诊断脚本
  全部现成）。`abfe_core.py`/`runabfe.py` 的 charge-transfer + 膜恒压器
  通用接线（`--only-complex-charging`、`--membrane-input-declaration`、
  co-ion dummy 插入）已经用这套中性体系验证过，从未在带电配体上跑过。
- **真正的冲突**：`memtodolist_archive.md`（2026-07-29）记录过一条决定——
  **"首个体系 = SERT（血清素转运体），配体默认净电荷 +1"**。但实际建出来
  并跑通的是上面这个 GPCR + 中性 Atenolol，跟当年那条决定不是同一个体系：
  SERT 从未真正建过膜体系（没有对应的 CHARMM-GUI 产物、没有嵌膜、没有跑过
  任何 smoke）。
- **配体电荷缺口，跟选哪个受体无关，两条路都要补**：仓库里没有任何带电
  （质子化、净 +1）的 Atenolol 参数——所有现成拓扑（根目录
  `Atenolol-rank1.itp`、`memtest/Atenolol-rank11.itp`）都是从
  `Charge=0` 的 QM 计算导出的中性形式。要走"配体带净电"这条路，不管配哪个
  受体，都需要重新做一次质子化仲胺的 QM 电荷推导（Gaussian）+ 重新生成
  GAFF 拓扑——不是挪文件就能解决的工作量。
- **受体身份记录缺口**：`memtest/membrane_input.json` 明确写着
  "未记录上游 PDB ID"、构象态"unspecified"——呼应 §A5"记录受体结构 ID、
  构象状态、突变、缺失残基和质子化态"这条从未打勾的要求；C4 定位是接线
  smoke（`production_qualified=false`），这个记录缺口是否必须先补齐、
  还是可以先如实标注"未知"往前走，也需要用户决定。

**用户 2026-08-11 明确表示：这个选择稍后告诉我，现在只要求把决策点和现状
写清楚——不要自己选受体/配体组合，也不要开始任何构建。**

1. **准备真实带电膜 complex，以及匹配的 solvent leg**
   - [ ] ligand 必须带净电荷（不是 C1/C2 用的中性探针或单原子简化）；
   - [ ] build 时显式插入 reserved neutral ion-shaped dummy；
   - [ ] 排除结构性离子、孔道离子、口袋/膜头基/疏水核中的候选（呼应
     §A5 已经列出但从未做过的排除清单）；
   - [ ] complex 与 solvent 两腿冻结**同一个** co-ion identity 和 restraint
     定义（不能两腿各自独立选一次）。
2. **零步静态预检**（不积分，只建 Context 查一次）
   - [ ] charging 全部 λ 态总电荷恒定；
   - [ ] `λ_coul=1`：ligand 满电、co-ion 中性；`λ_coul=0`：ligand 去电、
     co-ion fully charged；
   - [ ] Stage2 输入已经 baking 完成，System 里不存在活的 `lam_coul`
     GlobalParameter；
   - [ ] complex 用膜恒压器（`MonteCarloMembraneBarostat`），solvent 用
     各向同性恒压器；
   - [ ] handoff protocol/version 和 co-ion fingerprint 都已经进入
     cache identity。
3. **最短 GPU smoke**（不追求自由能收敛，只要能跑）
   - [ ] complex charging 能建 Context、积分、写 checkpoint；
   - [ ] complex Stage2 能接上 charging 端点（真正走一次 Stage2 handoff）；
   - [ ] solvent charging/Stage2 同样可运行；
   - [ ] 全程 energy/force finite；无 NaN、PME error、粒子逃逸或
     restraint runaway；
   - [ ] Stage2 全程 co-ion 保持 fully charged。
4. **相同命令立即 resume 第二次**
   - [ ] 命中相同 co-ion identity；
   - [ ] 已完成窗口被复用，不重跑；
   - [ ] 不重复插入 dummy/offset/restraint；
   - [ ] handoff/cache protocol 字段一致。
5. **复制一份 co-ion spec、故意篡改**（atom index / fingerprint /
   endpoint charge 任选一种）
   - [ ] 必须在建 Context **之前** fail closed；
   - [ ] 原始产物不能被这次篡改测试覆盖/污染。
6. **所有 C4 输出统一标注**
   ```json
   {"production_qualified": false}
   ```
   C4 只是接线 smoke，即使全部 PASS 也不能当生产结果用。

**当前最先要做的是第 1 步**：确定并预检真实带电膜 complex/solvent 输入。
§A5"目标膜输入"下的清单（受体结构 ID、构象状态、配体质子化态/形式电荷、
结构性离子排除等）到目前为止都还没做过，是这一步要补的作业，不是重复劳动。

### C5：co-ion 位置与 restraint 敏感性

前置：C4 通过。

- [ ] 至少 3 个合法 bulk-water 位置和 1 个故意违规位置。
- [ ] restraint 基线：k=100、r0=0.5。
- [ ] 弱/宽：k=50、r0=0.7。
- [ ] 强/窄：k=200、r0=0.3。
- [ ] 每个合法组合跑 complex/solvent 两腿和至少 3 seeds。
- [ ] 检查 dummy 吸附、charged endpoint 水合、触壁比例和 restraint 能量。
- [ ] 净 `ΔΔG_bind` 同时满足 2σ 和 1 kcal/mol 门。
- [ ] 若两腿 restraint 自由能不抵消，给出显式修正或判定路线失败。

---

## 5. 膜输入与科学协议仍缺

### A5：目标膜输入

- [ ] 准备并验证真正用于带电生产的已平衡膜输入。
- [ ] 记录受体结构 ID、构象状态、突变、缺失残基和质子化态。
- [ ] 记录配体质子化态、互变异构体、形式电荷和参数来源。
- [ ] 核对结构性 Na⁺/Cl⁻，从 co-ion 候选中显式排除。
- [ ] 排除蛋白孔道、结合口袋、膜头基层和疏水核中的 co-ion 候选。
- [ ] 核对蛋白插膜方向、配体 pose、结构水、辅因子和二硫键。
- [ ] 记录膜组成、上下叶组成、胆固醇比例、盐浓度和温度。

### 热力学循环和 restraint 账目

- [ ] 写清 co-ion restraint 在 complex/solvent 两腿是否抵消。
- [ ] 若可用体积不同，推导并实现显式修正。
- [ ] charge-transfer 路线最终报告必须明确 `APBS/Rocklin = 0`。
- [ ] co-annihilation 只允许实验对照，禁止进入膜生产 preset。
- [ ] `shadow_ibs` 对带电配体明确 fail closed，或完整实现同一 co-ion 路线。

### 膜生产协议

- [ ] 明确炼金生产阶段使用 NPT 还是 NVT。
- [ ] 若使用 NVT，记录固定盒矢量来自哪一帧。
- [ ] 明确时间步、约束和是否使用 HMR。
- [ ] 明确膜位置限制的分级释放方案。
- [ ] 记录结合位点是水相可及、界面、脂质暴露还是疏水深埋。
- [ ] 对脂质暴露/空腔填充做正反向或双初态迟滞验证。

---

## 6. 生产资格 Phase D

- [ ] D1：关闭 P1-19/P1-19b 的跨运行不确定度问题。
- [ ] D1：对 P1-22 的 Stage 2 帧选择和 σ 口径形成正式结论。
- [ ] D2：完成 Boresch 真实键拓扑和二面角更新门。
- [ ] D3：至少 3 个独立生产重复一致。
- [ ] D4：至少一个公开或可追溯膜受体 benchmark 通过。
- [ ] D5：完整 provenance、运行命令、环境、seed、输入 SHA256 和复现实验脚本。
- [ ] 膜质量门通过。
- [ ] overlap/ESS 和修正后的不确定度门通过。

---

## 7. Definition of Done

只有以下项目全部完成，才能声明支持生产级膜受体–配体 ABFE：

- [x] MEM-00h 端点能量/力验收通过（2026-08-11，随 C3 关闭）。
- [x] B5 cache/resume/provenance 正式关闭（2026-08-09）。
- [ ] C2–C5 全部通过（C1、C2、C3 已关闭并归档；C4 已解锁，C5 未开始）。
- [ ] co-ion 两腿显式存在、进入 PME、受控并进入全部缓存指纹。
- [ ] 全部 λ 总电荷恒定，且未重复应用 APBS/Rocklin。
- [ ] 膜恒压和平衡质量门通过。
- [ ] Boresch、co-ion restraint 和标准态修正闭环。
- [ ] 至少 3 个独立重复一致。
- [ ] 公开 benchmark 通过。
- [ ] 最终结果可审计、可恢复、可复现。
