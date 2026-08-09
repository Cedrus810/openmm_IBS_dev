# 膜受体–配体 ABFE 当前行动清单

更新日期：2026-08-09（B5 正式关闭；C2 thick-vs-thin 矛盾已查明并修复，见 §4 C2）  
状态：Phase B 工程实现基本完成；**B5 已关闭**（全套离线测试 1161 passed/
0 failed，`test_exp012_schema.py` 的 11 个既有失败已修好，缓存/resume/
co-ion 隔离全部复核通过）。C1 已关闭。C2 thin/thick base 均已通过；四格
`build`/`static-check` 已用 `PROTOCOL_VERSION = 7` 重新跑过并通过——v6 版本
选的候选点位有 bug（离膜中面距离算错，选到了实际偏近膜的点），第一批 4 个
GPU pilot 已经跑出来但因为这个 bug 全部作废，**wiring smoke 和 4 个完整
pilot 都要在 v7 重建的站点上重新跑**；**C2 尚未关闭**——是当前主线唯一在
推进的项。  
历史清单、已完成项目和详细论证：`memtodolist_archive.md`

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
C2 无蛋白 lipid slab（进行中，差第 5 步 GPU pilot）
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

状态：**代码改动已存在，等待测试验收后关闭。**

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
- [ ] 在真实端点测试 C3 中证明 λ=1 能量和力与基础力场一致——**仍未做，
  是 MEM-00h 唯一剩下的关闭条件**。

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

脚本：`tools/validation/validate_charge_transfer_lipid_slab.py`（`PROTOCOL_VERSION = 7`，
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

### C3：真实体系 λ=1/λ=0 端点能量和力

- [ ] 独立 reference builder，不复用生产 charge-offset planner。
- [ ] 在 C1 水盒和 C2 slab 上各抽至少 10 帧。
- [ ] charging λ=1 对基础物理体系。
- [ ] charging λ=0 对直接改粒子电荷的 reference。
- [ ] vanishing λ=1 与 charging λ=0 接缝一致。
- [ ] vanishing λ=0 ligand–environment 静电和 LJ 严格归零。
- [ ] 能量相对差 ≤ 1e-5。
- [ ] 最大逐原子力差 ≤ 1e-3 kJ/mol/nm。
- [ ] 本项同时作为 MEM-00h 的最终真实端点验收。

### C4：带电膜 complex/solvent 双腿 smoke test

前置：B5、C1、C2、C3 全部通过。

- [ ] 准备真实带电膜复合物，并显式包含 reserved neutral ion-shaped dummy。
- [ ] complex charging、Stage 2、solvent leg 全部有限且无 NaN/PME error。
- [ ] complex 使用膜恒压器，solvent 使用各向同性恒压器。
- [ ] Stage 2 中 co-ion 保持 fully charged。
- [ ] 第二次 resume 命中同一 co-ion identity，并复用已完成窗口。
- [ ] 篡改副本 spec 后能在建 Context 前 fail closed。
- [ ] 所有产物标记 `production_qualified=false`。

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

- [ ] MEM-00h 端点能量/力验收通过。
- [x] B5 cache/resume/provenance 正式关闭（2026-08-09）。
- [ ] C2–C5 全部通过（C1 已关闭并归档）。
- [ ] co-ion 两腿显式存在、进入 PME、受控并进入全部缓存指纹。
- [ ] 全部 λ 总电荷恒定，且未重复应用 APBS/Rocklin。
- [ ] 膜恒压和平衡质量门通过。
- [ ] Boresch、co-ion restraint 和标准态修正闭环。
- [ ] 至少 3 个独立重复一致。
- [ ] 公开 benchmark 通过。
- [ ] 最终结果可审计、可恢复、可复现。
