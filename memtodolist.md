# 膜受体–配体 ABFE 专项行动清单

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

  - [ ] **MEM-00a-1**：新增正式协议标识 `charge_treatment = co_alchemical_charge_transfer`，
    并引入独立的 `CHARGE_TRANSFER_PROTOCOL_VERSION = 1`（与
    `SOLVENT_CACHE_PROTOCOL_VERSION`、`IBS_BIAS_PROTOCOL_VERSION` 并列，不复用）。
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
- [ ] **MEM-00c：反离子身份是运行时重选的，违反 §3.4。**
  `_select_bulk_water_counterion`（`ibs_engine.py:766`）按"到最近溶质的
  minimum-image 距离 + 水配位数"当场排序挑离子。因为 MEM-00b 的三个调用点各调一次，
  **必须实测确认：动力学用的离子与 `compute_u_kn` 重算用的离子是否同一个粒子。**
  若三处传入的 `positions` 不同（很可能），选出的离子就可能不同 →
  u_kn 与动力学 Hamiltonian 不一致的静默错误。这条优先级最高，先写一个断言测试。
- [ ] **MEM-00d：反离子 restraint 形式不满足 §2.3。**
  `_create_bulk_water_ion_restraint`（`ibs_engine.py:459`）是
  `0.5*k*periodicdistance(...)^2`，k = 25 kJ/mol/nm²，参考点是**初始帧的绝对笛卡尔
  坐标**。§2.3 要求 flat-bottom + 随盒缩放的参考定义。在膜半各向异性 NPT 下，
  Z 方向盒长会变而参考点不动，离子可能被拖向膜。必须改。
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
- [ ] **✅ 已定（2026-07-29）：脂质力场按输入自动识别力场族**（`forcefield_family`）。
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

- [ ] `neutral`
  - 配体净电荷为 0；
  - 不创建 co-ion；
  - APBS/Rocklin 净电荷修正必须为 0。
- [ ] `co_alchemical_charge_transfer`
  - 配体净电荷不为 0；
  - 两腿都必须存在、选择并约束 co-ion；
  - 所有 charging λ 上总电荷必须恒定；
  - APBS/Rocklin 必须为 0。
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
- [ ] 配体电荷变化与 co-ion 电荷变化之和不为 0。

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

- [ ] 新增 `dispersion_protocol` 显式配置。
- [ ] `system_type = membrane` 且未选择已验证的 `dispersion_protocol` 时 fail closed。
- [ ] 禁止把 APBS 当成 LJ 修正。

候选路线只能选择一条：

#### 路线 A：复现目标膜力场的 cutoff/switch，不加均匀密度 LRC

- [ ] 从原始 GROMACS `.mdp/.top` 锁定 cutoff、switch 类型和距离。
- [ ] 普通 `NonbondedForce` 与炼金 softcore 力使用同一非键协议。
- [ ] 明确区分 energy-switch 与 force-switch；不能因为距离同为 1.0–1.2 nm
  就认为 Hamiltonian 相同。
- [ ] 复合物腿和溶剂腿使用同一套 ligand–environment 非键定义。
- [ ] 关闭当前 `lrc_coeff/V`，metadata 写
  `disabled_by_membrane_forcefield_protocol`，不能写成遗漏。
  （**仅当所选脂质力场本身不要求 LRC 时**；Amber 脂质见上方修正框。）
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

- [ ] co-ion 是 System 中真实存在的粒子，必须进入 PME `NonbondedForce`。
- [ ] 第一版只改变 charge；mass、sigma、epsilon 在所有 λ 保持不变。
- [ ] λ=1 时它是“中性但保留 LJ 的 ion-shaped dummy”。
- [ ] λ=0 时它成为与 `q_L` 同号的物理单价离子。
- [ ] `q_L = +1`：使用 `0 -> +1` 的阳离子型 co-ion。
- [ ] `q_L = -1`：使用 `0 -> -1` 的阴离子型 co-ion。
- [ ] `|q_L| > 1`：使用多个单价 co-ion，每个最多转移一个单位电荷。
- [ ] 非整数净电荷必须先作为输入错误调查；不要静默把它塞给一个分数价 co-ion。

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
- [ ] 使用可审计的 restraint；优先选择平坦区足够大的 flat-bottom restraint，
  避免把 co-ion 锁死在一个异常水构象。
- [ ] restraint 参考位置使用盒分数坐标或可随盒缩放的定义，不能在膜 NPT 中固定一个
  会漂入膜内的绝对笛卡尔坐标。
- [ ] restraint 势在所有 λ 完全相同。
- [ ] restraint 的自由能是否在两腿抵消必须写进热力学循环说明；若几何/可用体积不同，
  需要显式修正或用数值对照证明差异可忽略。

### 2.4 charging Hamiltonian

- [ ] 修改现有 charging builder，使 ligand 和 co-ion 的电荷由同一个 `lambda_q`
  控制，变化方向相反。
- [ ] co-ion 电荷必须通过 PME `NonbondedForce`/其合法 λ 参数化实现。
- [ ] 禁止用 cutoff `CustomNonbondedForce` 模拟 co-ion 的长程静电。
- [ ] 每个 charging state 的能量矩阵同时包含 ligand 与 co-ion 的变化。
- [ ] TI/BAR/MBAR 的 `u_kn`、checkpoint、manifest 和协议指纹都必须包含 co-ion 身份和参数。
- [ ] stage 2 vanishing/vdW 阶段固定：
  - ligand 电荷为 0；
  - co-ion 电荷为 `q_L`；
  - co-ion 不参与 ligand 的 vdW decoupling。
- [ ] 不能让 co-ion 在 stage 1 后被错误恢复成中性。

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

- [ ] 输入必须是已经完成膜构建和主要平衡的 protein–lipid–ligand–water–ion 体系。
- [ ] 不依赖当前通用 10 ns 预平衡去完成脂质重排或蛋白插膜。
- [ ] `.gro`、`.top`、全部 `.itp`、位置限制文件和力场 include 一起归档。
- [ ] 记录输入 SHA256、构建工具、构建参数和最终平衡作业。
- [ ] 核对：
  - [ ] 坐标/拓扑原子数；
  - [ ] 上下叶脂质数；
  - [ ] 水和离子数；
  - [ ] 蛋白跨膜方向；
  - [ ] 配体 pose；
  - [ ] 辅因子、结构水、二硫键和质子化态；
  - [ ] 周期盒无膜间异常接触。

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
- [ ] 复合物腿结果写膜质量诊断摘要。

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
- [ ] membrane + legacy uniform-density LRC 必须 fail closed。
- [ ] 新 LJ 协议进入窗口 manifest、energy cache 和 resume gate。

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

- [ ] 中性配体 + `neutral`：通过。
- [ ] 带电配体 + `neutral`：失败。
- [ ] 带电配体 + co-ion：通过。
- [ ] co-ion + 非零 APBS：失败。
- [ ] membrane + 未指定 dispersion protocol：失败。
- [ ] membrane + 普通各向同性 barostat：失败或明确替换，不能静默继续。

### 7.2 电荷守恒

对每个 charging λ：

- [ ] `sum(all NonbondedForce particle charges)` 恒定到严格数值容差。
- [ ] ligand charge 等于 `lambda_q * q_L`。
- [ ] co-ion charge 等于 `(1-lambda_q) * q_L`。
- [ ] λ=1 和 λ=0 均满足预期。
- [ ] λ 中间态也满足，而不是只检查端点。
- [ ] energy matrix 重算使用的 charge 与动力学 System 完全一致。

### 7.3 co-ion 物理测试

- [ ] co-ion mass/LJ 在 λ 间不变。
- [ ] co-ion charge 通过 PME，不通过 cutoff ghost force。
- [ ] restraint 在三斜/各向异性盒中 minimum-image 正确。
- [ ] NPT 盒变化后 co-ion 不漂入膜。
- [ ] 多 co-ion 分摊电荷时，总变化正确、每粒子不超过一个单位电荷。

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

- [ ] 每个量都必须给出**时间序列**和**末段窗口的漂移斜率**，不能只报平均值。
- [ ] 判据统一为"末段 ≥ 20 ns 内线性漂移小于阈值"，阈值见 §13。
- [ ] 记录脂质横向弛豫的估计时间尺度（脂质横向 MSD 或首层脂质交换时间），
  用它论证预平衡时长够——**通用 10 ns（当前 `n_equil_steps = 5e6`）几乎肯定不够**。
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

- [ ] A1. 确定首个膜体系和配体电荷。
- [ ] A2. 确定水相 1 M 标准态。
- [ ] A3. 锁定力场与 LJ 协议。
- [ ] A4. 选择 `neutral` 或 `co_alchemical_charge_transfer`。
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
- [ ] B2. 加 charge-treatment 配置与双计数 fail-closed。
- [ ] B3. 实现 PME co-alchemical ion charging Hamiltonian。
- [ ] B4. 重写溶剂腿 builder，显式返回 co-ion identity。
- [ ] B5. complex/solvent cache 加 co-ion 指纹。
- [ ] B6. membrane 模式禁用 legacy uniform-density LRC。

### Phase C：物理与数值验证

- [ ] C1. 水盒 charge-transfer 解析/盒长测试。
- [ ] C2. lipid slab 测试。
- [ ] C3. λ=1/λ=0 endpoint 能量与力测试。
- [ ] C4. membrane complex/solvent 双腿 smoke test。
- [ ] C5. co-ion 位置与 restraint 敏感性测试。

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

## 13. 阈值默认值（**提案，Phase A 定稿；原稿全篇写"预设阈值"但没给数**）

没有数就没法写 fail-closed 检查，也没法判验收。以下是起始提案，可以改，
但**必须在 Phase A 结束前落成常量并进 provenance**，不许运行时凭感觉判。

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

- APL 漂移 ≤ 0.2 %/ns，且与该脂质力场文献值差 ≤ 3%。
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

## 15. 成本与资源（原稿完全没提，但它决定 Phase C/D 可行不可行）

- [ ] **Phase A 必须先实测**，不许拍脑袋：
  - 目标膜体系原子数（当前可溶体系 `solv_ions.gro` 约 5 万原子；
    GPCR + 脂双层 + 水通常 10–15 万，即 2–3×）；
  - 在目标 GPU 上的 ns/day（NPT 生产设置，含 barostat 与 IBS bias 开销）。
- [ ] 用下式估算并填表，两条腿分别算：

  ```text
  GPU-hours ≈ (窗口数 × 每窗口 ns + 预平衡 ns + warmup ns) / (ns/day) × 24 × 重复数
  ```

  当前可溶体系参数供对照（`abfe_config.json`）：
  stage1 12 态、stage2 17 态、每窗口 250k 步（2 fs → 0.5 ns）、
  warmup 500k 步（1 ns）、预平衡 5e6 步（10 ns）。
  **膜体系这几个数都会变大**（预平衡 ≥ 100 ns、窗口数按 §6.4 重估）。
- [ ] 估算轨迹/能量缓存的磁盘占用，确认输出目录容量够；
  给出保留策略（哪些帧长期留、哪些跑完即删）。
- [ ] 给出**串行 wall-clock 关键路径**，不只是 GPU-hours 总量。
- [ ] 若估算结果超出可接受范围，在 Phase A 就缩小体系（更小膜片 / 更少重复），
  **不许**靠砍窗口数或砍采样时长来凑——那会直接打回 R1/R5。

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

膜工作**不能**在下面这些没收口之前进生产：

- [ ] `docs/TODO.md` P1-19（per-window σ 系统性低估 2–4 倍）——
  不修的话，膜体系的跨重复散布会被同一个问题污染，§13.4 的验收无法判定。
- [ ] `docs/TODO.md` P1-23（σ 采纳路径 fail-open，真 bug）。
- [ ] P1-22（vdW/stage2 帧选择与 σ 口径）——至少要有结论，即使结论是"维持现状"。
- [ ] MEM-00h（cutoff 不一致）——存量问题，**必须单独立项、单独 PR**，
  不许和膜改造混在一起改，否则出问题无法二分定位。
- [ ] MEM-00c（co-ion 身份可能在动力学与 u_kn 之间漂移）——
  这是唯一可以、也应该**现在就查**的一条，与膜无关，只需要一个断言测试。

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
