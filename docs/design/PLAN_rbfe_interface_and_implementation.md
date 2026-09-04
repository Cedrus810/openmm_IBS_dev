# RBFE 接口与代码实现计划

日期：2026-08-31（撰写）／**2026-09-03（状态复核）**  
状态：**R0、R1 完成；R2 完成一半（独立窗口 + 分析已通，REMD 未验证）；R3/R4 未开始。**

## 0. 实施状态（2026-09-03）

| 阶段 | 状态 | 说明 |
|---|---|---|
| **R0** | ✅ **完成** | `rbfe_core.py` + `runrbfe.py`（validate/combine/template） |
| **R1a** | ✅ **完成** | 原子映射：分子图 / 环分析 / 片段分解 / A→B 映射 / 映射验证 / `runrbfe.py map`，55 条测试 |
| **R1b** | ✅ **完成** | 受限 hybrid builder：`build_hybrid_system` + 端点等价 / 有限差分力 / 逐 λ 电荷三项验收，44 条测试 |
| **R2** | 🟡 **一半** | 独立窗口采样 + `compute_hybrid_u_kn` + `analyze_leg` → `LegResult` 已通，21 条测试；**REMD 那半未验证**（§8 原文：「独立窗口先跑通，**再验证 REMD**」） |
| R3 / R4 | ⬜ NOT_STARTED | |

### 模块落地情况（§4 的四个大模块）

| 模块 | 状态 |
|---|---|
| `rbfe_core.py` | 🟢 已建：R0 契约 + 验证 + ΔΔG 汇总、R1a 映射层、R1b 受限 hybrid builder、**R2 的 u_kn 与 `analyze_leg`**。仍未实现的只剩 `prepare_edge`（要可跑的配体 B）与 `build_hybrid_leg`（要 `PreparedEdge`） |
| `free_energy_engine.py` | 🟡 已建：后端选择器（P0，已接线）+ 共用采样契约（P2′，未接线）+ **独立窗口采样 `run_independent_windows`**（新增路径，ABFE 三个 REMDManager 调用点完全不经过它）；见 [REMD 后端计划](PLAN_openmm_8_6_remd_backend.md) §0 |
| `rbfe_pipeline.py` | ✅ 已建（2026-09-01）：身份指纹 / 目录契约 / 续跑校验 / 两腿汇总 / 独立重复 / 网络闭环 / ABFE 锚点。⚠ 本表在 2026-09-02 之前一直写着「不存在」，是**文档失同步**，不是模块缺失 |
| `runrbfe.py` | ✅ 已建：validate / combine / template / **map**（R1a） |

### R2 完成一半（2026-09-03）

§8 的 R2 验收是「**独立窗口先跑通，再验证 REMD**；真实 hybrid 能量和样本归属一致」。
前半句做完了，后半句没有——所以是一半，不是完成。

#### 分工（§6 那句话的两半都照做）

§6 禁止复用 `TraditionalMBARAnalyzer.compute_u_kn`（它按单配体去耦与 LRC 假设
**重建评估系统**：PME self correction、ligand_charge_square_sum、co-ion、Boresch，
这些假设对 hybrid Hamiltonian 全不成立）。同一句话的后半段说**独立的 MBAR 数值
求解部分**可以复用。于是：

| 层 | 来源 |
|---|---|
| u_kn | RBFE 自己算（`rbfe_core.compute_hybrid_u_kn`）：hybrid System 只有一个，换态就是 `setParameter`，**结构上不可能算错评估体系** |
| MBAR 求解 | 复用 `ibs_engine.TraditionalMBARAnalyzer.solve()`（去相关子采样、overlap 诊断、多套 solver protocol 兜底），**惰性 import，不改 ABFE 一行** |
| 独立窗口采样 | 新增 `free_energy_engine.run_independent_windows`，engine 只按状态表推进、不解释 λ 语义、**不生成 seed**（拿不到 `integrator_seeds` 就报错）、**不往调用方 System 里加 barostat** |

#### 方向链（计划 §3 的落点）

`solve()` 返回的 `delta_G` 是 `(f[-1] − f[0])·kT`。`HybridLambdaSchedule` 强制端点
0→1，λ=0 对应 A、λ=1 对应 B，所以 `delta_G == G(B) − G(A) == ΔG(A→B)`，
正是 `LegResult.delta_g` 要求的含义。这条链跨三个模块，靠记忆维持迟早出错，
有测试钉着。

#### 验收数字

**A→A 自边 ΔG = +2×10⁻⁸ ± 3.5×10⁻⁷ kJ/mol**（真 MD，5 窗口）——
§8 里 R3 那条「A→A 为零」在分析层的落点。该性质对**任意**样本成立
（自边所有 λ 态 Hamiltonian 逐位相同 ⟹ u_kn 每行相同 ⟹ MBAR 恒为 0），
所以单元测试用合成样本验到底，只留一条真 MD 的贯通用例。

#### R2 期间抓到的一个真 bug（**这条最值得记**）

**一个 interaction group 都没有的 `CustomNonbondedForce` 会计算全体粒子对**——
这是 OpenMM 的默认行为，不是"什么都不算"。原代码是 `if a_only: force_a.addInteractionGroup(...)`，
于是**没有 dummy 原子的边**（A→A 自边，以及任何「只改参数、不增删原子」的突变——
那是很常见的一类边）会多挂两个退化成"全体粒子对"的力，与 core 力和 native 力
**重复计数**。

它一度被正确答案掩盖：A→A 的 ΔG 照样算出 0，因为 λ 表对称、两个端点上 softcore
lift 都是 0，两个力恰好互为镜像而抵消——**端点对了，中间态全错**。
端点验收（R1b 的三项）结构上抓不到它；抓到它的是 R2 新写的
`test_self_edge_u_kn_rows_are_identical`，因为那条验的是**中间态**。

教训：端点等价是必要条件，不是充分条件。已修（无 group 的力不再加进 System），
并补了四条回归测试，包括一条「只改电荷不增删原子」的边走完整端点 + 有限差分验收。

#### R2 未做的

* **REMD 未验证**：`run_independent_windows` 之外还没有 hybrid 状态表的副本交换路径。
  `run_sampling` 那条契约转交的是调用方构造好的 sampler，而
  `ibs_engine.REMDManager.__init__` 的签名写死了 ABFE 去耦语义
  （`lambdas_coul / lambdas_vdw / ligand_indices / boresch_params / co_alchemical_ion_spec`），
  RBFE 用不了，需要另写。
* 采样产物目前**留在内存**（`InMemoryWindowSamples`），没有落盘/续跑路径——
  §7 的目录契约在 `rbfe_pipeline.RunLayout` 里已经有了，两者还没接上。

### R1a 已完成（2026-09-03）

路线由用户拍板为 [`PROPOSAL_rbfe_r1_fragment_mapping.md`](PROPOSAL_rbfe_r1_fragment_mapping.md) §3 的
**A+B 混合**：片段级匹配定骨架对应，片段内部原子级对齐；rdkit 只在「位置对应但
组成不同」的那一对片段内部出场，且建 RWMol 时全部按单键、比较时忽略键级，
**不引入化学感知、不重参数化**（计划 §5.1 红线）。

`rbfe_core.py` 新增（仍然不 import openmm、不建 System、不启动 GPU）：

* `MolecularGraph`：配体键图。三个入口——`from_atoms_and_bonds` /
  `from_openmm_topology`（惰性用 openmm，模块本身不 import）/ `from_gromacs_itp`
  （只读 `[atomtypes]`/`[atoms]`/`[bonds]`，元素取 at.num，**拿不到就报错不猜**）。
  构造即校验：重复索引、自环、跨界键、**不连通**（= 共价配体，§2 拒绝）
* 环分析：`ring_bonds()` / `ring_atoms()` / `smallest_ring_size()` / `ring_size_profile()`
* `decompose_into_fragments()`：在**非环可旋转键**处切分（判据与
  `outer_lambda_neural_basis.discover_ligand_rotatable_torsions` 语义一致，但是
  重写的——那份实现是函数内闭包，import 不出来，且按依赖方向本模块不得 import ABFE）
* `map_atoms()`：M3-M5。种子（两边签名都唯一的最大片段）→ 沿切键生长 →
  片段内同构对齐 → 差异片段 MCS → 组装冻结映射
* `AtomMapping`：分子内索引为唯一坐标系；`project()` 投影到每条腿的全局索引
  （两腿物理上不可能用不同映射）；`hybrid_indices()` 给出 core→A-only→B-only 的
  hybrid 编号；`fingerprint()` 进边身份，**故意不含原子名**（改名不该让已跑完的采样作废）
* `validate_mapping()`：一一对应、索引范围、**元素一致**、**核心连通性**、
  **环断裂/闭合** —— R0 挂在 `unchecked` 里的环与元素两类在这里被真正查掉
* `validate_edge(spec, mapping=, graph_a=, graph_b=)`：给了映射证据就真查，
  对应的 `unchecked` 条目随之消失；只给一部分证据直接报错，不悄悄退回"没查"

`runrbfe.py map --ligand-a … --ligand-b … --out atom_mapping.json`：R1 验收
第一条「**映射可审计**」的落点。产物含片段配对、公共核心、A-only/B-only、
对称等价解数量、歧义说明、指纹。

在仓库自带的**真实** Atenolol（41 原子、含苯环）上验证：A→A 恒等零 dummy；
H→CH₃ 变换公共核心 40、A-only 恰好那个氢、B-only 恰好新甲基的 4 个原子。

#### R1a 期间踩到并已修的两件事（别放宽对应测试）

1. **按签名生长会漏掉差异片段之后的整段。** 第一版沿片段图按签名生长，苯环换
   取代基后签名变了，生长在苯环处断掉，苯环**后面**那一整段（酰胺尾，A/B 完全
   相同）被整段判成 dummy——公共核心 32 而不是 40，而且**验证照样 PASS**。
   改成按**接点原子对应**生长才对（接点对应不因取代基变化而失效）。
2. **差异在分子中部时，保守路径（不用 MCS）给不出可用映射。** 整块 dummy 会把
   公共核心从中间切断，`validate_mapping` 的连通性判据正确拒绝。结论：
   **rdkit MCS 不是优化项**，只有差异位于末端片段时保守路径才可用。错误信息
   已经把这条写进去了。

### R1a 未做的（诚实清单）

手性反转、结合姿势歧义、互变异构——都需要**坐标**，本层只有键图，一律进
`ValidationReport.unchecked`。虚拟位点/约束/自定义 Force 的支持范围需要建系，
属于 R1b。

### R1b 已完成（2026-09-03）

用户决定：**直接 import `abfe_core` 复用，但不改 ABFE 一行代码。** 本轮严格照办——
`abfe_core.py / abfe_pipeline.py / ibs_engine.py / runabfe.py` 一行未动，import 全部
**惰性**（放在函数体内），因为 `abfe_core` 会拉进 openmm，放模块顶部会让 R0/R1a
那两层「不 import openmm、不启动 GPU」的性质连同它们 110 条测试一起失效。

`rbfe_core.py` 新增 `build_hybrid_system(system_a, ligand_indices_a, system_b,
ligand_indices_b, mapping, schedule) -> HybridSystemBundle`。

#### 首版受限 builder 的分类约定

| 力项涉及的原子 | 处理 | 力组 |
|---|---|---|
| 纯环境 | 从 A 原样搬 | 0 |
| 纯 core，A、B 参数相同 | 原样搬（native 力） | 0 |
| 纯 core，键/角参数不同 | `Custom*Force` 按 `lambda_rbfe_bonded` **插值参数** | 0 |
| 纯 core，二面角不同 | **双项缩放**：A 的项 ×(1−λ)、B 的项 ×λ | 0 |
| 涉及 A-only（dummy） | **全强度保留，永不缩放** | 1 |
| 涉及 B-only（dummy） | **全强度保留，永不缩放** | 2 |

dummy 成键项全强度保留是它能抵消的**前提**：dummy 的构型积分因此是一个与 λ 无关的
可分离因子，在两腿之间严格相消（§5.3 要求「明确可分离因子及其抵消条件」）。
把它们放进**独立力组**则让这件事可以直接读数——端点验收就是靠扣掉对侧 dummy 力组
做精确等式比对，不用估容差。

静电全部留在 native `NonbondedForce` 走参数 offset（保 PME，这是从 ABFE 抄的最关键
一条）。LJ 从 native 力摘掉，拆成三个 `CustomNonbondedForce`：core（参数插值、
不加 softcore）、A-only（前因子 1−λ、softcore lift λ）、B-only（前因子 λ、lift 1−λ）。
**A-only × B-only 不在任何 interaction group 里且显式加了排除**——两组 dummy 永远
不能相互看见，这是 hybrid 拓扑的硬约束，ABFE 里没有对应概念。

#### 三项验收（都在真实 OpenMM Context 上跑，不是纸面推导）

* `verify_hybrid_endpoints`：`E_hybrid(λ=0) − E[dummy_B 力组] == E_A`，λ=1 同理。
  实测 **ΔE = 0 / 2×10⁻¹³ kJ/mol，逐原子最大力偏差 1.6×10⁻¹² kJ/mol/nm**。
* `verify_hybrid_forces_finite_difference`：中心差分 −dU/dx vs 解析力，确定性抽样。
  5 个 λ 态全通过，抽样覆盖 dummy 原子（softcore 分母是手写的，不测等于没测）。
* `hybrid_charge_ledger`：**读建好的 System** 的粒子电荷与 offset 逐态求和，
  不是把 builder 的意图重算一遍。

#### R1b 期间踩到并已修的三件事（别放宽对应测试）

1. **`max(0.0, nan)` 在 Python 里返回 `0.0`**（因为 `nan > 0.0` 是 False）。端点比对
   一度把一个 NaN 力偏差静默当成"零偏差"通过了验收——fail-closed 的检查器自己漏 NaN，
   比没有检查器更危险。现在非有限值在 `_energy_and_forces` 里就抛错。
2. NaN 的来源是 **`CustomTorsionForce` 在退化几何（四原子共线）处的解析导数没有定义**，
   而 OpenMM 内建的 `PeriodicTorsionForce` 在那里是安全的。因此纯 core 且 A、B 逐位
   相同的二面角一律走 native 力，把 Custom 的暴露面压到"真正随 λ 变的那几个"。
3. **LJ 长程校正（LRC）缺口**：炼金原子的 epsilon 在 native 力上被清零，native 的
   色散校正不再包含配体那部分，hybrid 与纯 A 体系之间差一个常数（力恒为 0）。
   端点等式因此在关掉 LRC 的副本上做，缺口本身**量化成 `lj_lrc_gap_kJ_per_mol`
   报出来**，不留含糊 caveat。

#### 从 ABFE 抄 offset 时必须改掉的系数（点名记一笔）

`ibs_engine.py:3583` 那段 exception 电荷 offset 的判据是
`(p1 in alchemical) ^ (p2 in alchemical)`——**异或**，只处理「单端炼金」，
系数写的是 `base=0, scale=chargeProd`，即插到**零**（ABFE 的端点）。
RBFE 的端点是 B，所以：机制照用（它保 PME），**系数必须换成
`base=值ᴬ, scale=值ᴮ−值ᴬ`**；配套的 `_assert_frozen_ligand_ligand_exceptions`
不能照搬，因为「单端」前提在 RBFE 里不成立。测试
`test_charge_offset_interpolates_to_B_not_to_zero` 专门守这条。

#### R1b 的已知缺口（写进产物的 `provenance.known_gaps`，不藏在注释里）

* 炼金区的 LJ 长程校正未计入。ABFE 侧有现成的
  `ibs_engine._lj_softcore_tail_radial_integrals`（switching-aware + softcore-aware
  径向积分）可接，R1b v1 未接。
* 手性 / 结合姿势 / 互变异构仍未检查（需要坐标）。
* 受限范围：只支持 `HarmonicBond/HarmonicAngle/PeriodicTorsion/Nonbonded`
  （加 `CMMotionRemover`、`MonteCarloBarostat`），虚拟位点、多个 `NonbondedForce`、
  core–core 的成键/断键、core 质量不同（同位素/HMR）一律 fail closed。

### R1b 未做的（诚实清单）

`prepare_edge`（从 `EdgeSpec` 的输入路径建出两个端点体系）与
`build_hybrid_leg`（`PreparedEdge` 上的薄封装）仍是 `NotImplementedError`。
两者都卡在同一个前提上：一份**可跑的**配体 B。R2 的 `analyze_leg` 也未开始。

### R0 已完成（2026-08-31）

`rbfe_core.py`（新建，2026-08-31），不 import openmm、不建 System、不碰 GPU：

* **数据契约**：`LigandEndpoint` / `EnvironmentSpec` / `ProtocolSpec` / `EdgeSpec`
  （§5.1 的身份字段，含输入 sha256、质子化态、立体化学、部分电荷来源）；
  `EdgeSpec.manifest()` 产出 §7 的 `edge_manifest.json` 内容
* **验证**：`validate_edge()` 实现 §2 首版范围拒绝——净电荷变化、同电荷带电配体
  （两条独立判据）、膜体系、质子化态改变、A→A 自边、未声明的身份字段、非法协议。
  错误信息保留「这是本项目首版的范围限制，不是 RBFE 方法不支持」的区分
* **诚实的未检查清单**：环断裂/闭合、手性反转、共价配体、互变异构等需要原子映射
  才能判定，R0 查不了，一律进 `ValidationReport.unchecked`，**不让 PASS 被读成
  「全都查过了」**
* **ΔΔG 汇总**：`combine_rbfe()` 实现 §3 符号约定（`complex − solvent`，与 ABFE
  的 `solvent − complex` 相反）+ §7 误差传播（独立时方差相加；有协方差时
  `−2Cov`；传播后方差为负时**报错而不是 clamp 到 0**）+ 两腿身份校验
* R1/R2 的 `prepare_edge` / `build_hybrid_leg` / `analyze_leg` 一律
  `raise NotImplementedError`——§4.1 明令不提供「尚未实现但返回成功」的占位实现
* `tests/test_rbfe_core_r0.py`：46 条，按 §8 的 R0 验收标准组织

### R0 的 CLI（已完成）

`runrbfe.py`，不 import openmm、不启动 GPU：

* `validate --config edge.json [--json]`——加载 + schema 校验 + R0 验证。
  **未知字段一律拒绝**（`temperature_kelvim` 这种 typo 静默忽略会用默认温度跑完全程）。
  `--json` 时 stdout 只有 JSON，人类可读文本走 stderr，可直接 `| jq`。
* `combine --complex-json --solvent-json [--covariance]`——两腿 ΔΔG，
  验证符号与误差传播
* `template`——打印一份**本身就能通过验证**的边配置模板
* `prepare` / `run` / `analyze` 注册为子命令但**明确报错退出**（退出码 3），不假装成功
* 三种退出码可区分：0 通过 / 2 输入被拒绝 / 3 尚未实现
* `tests/test_runrbfe_cli.py`：19 条

### 前置依赖现状

R2 需要的共用采样契约（`SamplingRequest` / `SamplingArtifacts` / `run_sampling`）
**已经写好并测过**，见 REMD 计划 P2′。但它**还没接进生产**——ABFE 那三个
`REMDManager` 调用点仍走老路。R2 可以直接对着这层契约写，不必等 OpenMM 升级；
真正被 8.6 阻塞的只有官方适配器（P1）。

**§8 硬前提的现状（2026-09-03 更新）**：用户决定**不从外部取配体 B**，而是
**从现有拓扑派生**——改动配体残基的一个末端基团。R1a 的映射层已按这条路线在
真实 Atenolol 上验证（图层面的 H→CH₃）。但要注意区分两件事：

* **R1a 只需要键图**，图手术就够，已完成；
* **R1b/R3 需要一份真正可跑的 B**：改基团必然要重新给部分电荷与成键参数，
  这条参数化路线本身要单独验收（§5.1 不允许悄悄重参数化）。这仍然是 R3 的
  硬前提，没有因为 R1a 完成而消失。

## 1. 是否需要

如果项目目标扩展到一系列相近配体的亲和力比较，应准备 RBFE；如果只继续做单个配体的绝对结合自由能，RBFE 不是 ABFE 的必需依赖。两条方法保留独立入口，复用经过抽象和验证的底层组件。

本计划采用常规的“两条腿、A→B 配体变换”的 RBFE 路线。它需要配体映射和混合拓扑；官方 REMD 只提供多状态采样，不能从两个配体自动生成正确的变换 Hamiltonian。OpenFE 的相对自由能流程也将映射、体系构建和两条腿明确分开。[OpenFE 官方协议说明](https://docs.openfree.energy/en/stable/guide/protocols/relativehybridtopology.html)

前置兼容工作单独维护：

- OpenMM >=8.6.0 的官方 REMD 选择见 [REMD 后端计划](PLAN_openmm_8_6_remd_backend.md)。
- openmm-ml >=1.7 可退出已修复的 MACE 本地模型 device 补丁；经典力场 RBFE 不依赖 openmm-ml，不应为了运行 RBFE 强制加载 ML。
- 不把 IBS/ACE/EXP-030 直接移植为 RBFE 生产默认值；先建立可验证的传统基线。

## 2. 首版范围

优先完成一个明确的 A→B 边，再扩展为配体网络。

首批资格验证采用：同一受体、相近骨架、固定质子化／互变异构状态、相容结合姿势、两个中性配体、可溶性体系、经典力场，以及可核对的端点结构和参数。lambda 与温度／压力／约束协议显式配置，不沿用 ABFE 的隐藏默认值。

首版在生产 Context 创建前拒绝：

- 净电荷变化，以及尚未验证的同电荷带电配体路线；
- 环断裂／闭合、环尺寸变化、手性反转、映射元素改变、共价配体；
- 膜体系、质子化或互变异构状态改变、大幅骨架或结合模式改变；
- 元素、虚拟位点、约束、力场或自定义 Force 不在已验证 builder 支持范围内的输入。

这些是本项目首版的范围限制，不代表 RBFE 方法或其他软件普遍不支持这些变换。

## 3. 热力学与符号契约

统一定义 A→B、lambda=0 对应 A、lambda=1 对应 B：

```text
ΔG_complex(A→B) = G_complex(B) - G_complex(A)
ΔG_solvent(A→B) = G_solvent(B) - G_solvent(A)

ΔΔG_bind(B-A) = ΔG_complex(A→B) - ΔG_solvent(A→B)
```

因此负的 ΔΔG 表示 B 的结合自由能更低。输出必须同时记录 A、B、方向、单位和每条腿的值。

当前 ABFE 使用 coupled→decoupled 的去耦约定，最终采用 solvent−complex。**RBFE 不可直接调用该符号约定下的 ABFE 汇总函数，也不能只把名称改成 relative。**

RBFE 不机械添加 ABFE 的 Boresch 解析释放项。若采用限制配体构象／位置的 restraints，必须明确其对目标自由能的影响，并证明两腿抵消或计算所需修正。用于标定绝对值的 ABFE／实验锚点是网络分析层的可选输入。

## 4. 代码组织与接口草案

采用四个大模块，和现有 `abfe_core.py / abfe_pipeline.py / runabfe.py` 的分层方式衔接。**不拆成 `schema.py / mapping.py / topology.py / sampling.py / network.py` 等大量小文件**；这些是大模块内的职责区块。

| 大模块 | 内部职责 | 明确不负责 |
| --- | --- | --- |
| `rbfe_core.py` | 输入／结果类型、配体身份、映射、hybrid Hamiltonian、端点验证、RBFE 能量评估与 ΔΔG 汇总 | 不调度网络、不选择硬件、不处理 CLI |
| `rbfe_pipeline.py` | prepare/run/analyze 编排、两腿、独立重复、边集合／网络、运行状态和续跑校验 | 不自行构建相互作用公式、不重写交换算法 |
| `free_energy_engine.py` | ABFE/RBFE 共用采样契约、官方／legacy 后端选择、通用独立窗口与 REMD 推进、reporter、平台资源与运行诊断 | 不识别 ligand A/B、不生成 ABFE 去耦或 RBFE hybrid System、不计算两腿结合自由能 |
| `runrbfe.py` | 参数解析、配置加载、调用 pipeline、用户可见结果与退出码 | 不持有科学算法或第二份计算流程 |

`rbfe_core.py` 内可按“数据契约 → 映射 → hybrid builder → 验证 → 分析”分节；`rbfe_pipeline.py` 内按“单腿 → 双腿边 → 网络”分节。大模块化不意味着复制已有代码或允许循环依赖。

依赖方向固定为：

```text
runrbfe.py
    └── rbfe_pipeline.py
            ├── rbfe_core.py
            └── free_energy_engine.py

runabfe.py
    └── abfe_pipeline.py
            ├── abfe_core.py / ibs_engine.py 的现有化学与 IBS 逻辑
            └── free_energy_engine.py 〔未来逐步接入〕
```

- shared engine 不反向 import pipeline、`rbfe_core` 或 `ibs_engine`。
- 两条 pipeline 分别把化学层输出转成 shared engine 的通用采样输入，并把采样输出交回自己的分析层。
- 现有 ABFE 先保持原有入口；只有已通过回归的公共部分才迁入 shared engine，不以 RBFE 设计为由整批重构 ABFE。
- 先前 REMD 计划中的 `remd_backends.py` 调整为 `free_energy_engine.py` 内部的后端适配区块，避免平行建立两套版本判断与交换引擎。
- CLI/API 均调用同一 pipeline；OpenFE/Perses 等候选 builder 的适配放在 `rbfe_core.py` 内，不额外扩散入口。

拟议 Python 接口如下，仅定义职责，不是可直接运行的现有 API：

```python
def validate_edge(spec: EdgeSpec) -> ValidationReport: ...
def prepare_edge(spec: EdgeSpec) -> PreparedEdge: ...
def build_hybrid_leg(edge: PreparedEdge, phase: str) -> PreparedLeg: ...
def run_leg(leg: PreparedLeg, protocol: ProtocolSpec) -> SamplingArtifacts: ...
def analyze_leg(leg: PreparedLeg, artifacts: SamplingArtifacts) -> LegResult: ...
def combine_rbfe(complex_result: LegResult, solvent_result: LegResult) -> EdgeResult: ...
```

规划中的 CLI：

```text
python runrbfe.py validate --config rbfe_edge.json
python runrbfe.py prepare  --config rbfe_edge.json
python runrbfe.py run      --prepared prepared_edge.json --backend auto
python runrbfe.py analyze  --run-dir output_rbfe/A_to_B/repeat_01
```

### 4.1 共用引擎的边界契约

以下为字段设计，不在本次实现类：

| 契约 | 必需信息 |
| --- | --- |
| `SamplingRequest` | 已构建的 System/Topology、有序 state 参数表、各副本初始构型／速度／盒矢量、积分器与系综设置、采样／交换／保存步数、seed 计划、平台预算、输出目标、调用方协议指纹 |
| `BackendResolution` | requested/resolved backend、实际 OpenMM 版本、能力检查结果、交换算法、选择理由 |
| `SamplingArtifacts` | 按状态的样本位置／索引、MD 步数与保存时点、状态映射、诊断、checkpoint 类型、实际后端、指纹 |
| `LegResult` | phase、A→B 身份、ΔG、单位、误差方法、有效样本与质量门、所用物理修正、产物指纹 |
| `EdgeResult` | 经过身份校验的两腿结果、ΔΔG、传播后的不确定度、方向、qualification 与理由 |

引擎保证的是“按指定 Hamiltonian 和状态表采样”，化学正确性由 core 验证，完整流程身份由 pipeline 验证。未提供某项能力时必须显式报告不支持，不由引擎猜测物理默认值。

不提供“尚未实现但返回成功”的占位 sampler；接口准备完成与可运行科学计算是两个验收阶段。

## 5. 输入、映射与混合拓扑

### 5.1 输入身份

EdgeSpec 至少保存：edge_id、ligand_A/B、受体／环境身份、各端点输入路径及 hash、带显式氢的化学结构、形式电荷、质子化／立体化学身份、原子映射、温度与系综、力场／水／离子模型、部分电荷来源、采样配置、seed 和输出目录。

已参数化的 GROMACS／OpenMM 输入与自动参数化的 SDF 路线分开。**不能为了调用第三方 builder 悄悄把当前力场重参数化。** 首版锁定一条可完整验证的参数化路线，其他输入先拒绝，不自动猜测转换。

### 5.2 原子映射

- 分子内索引、complex 全局索引、solvent 全局索引和 hybrid 索引分别记录。
- 验证一对一、索引范围、化学一致性和映射核心的连通性，识别对称等价映射与姿势歧义。
- 两腿使用同一份冻结的分子级 A→B 映射，再各自投影为全局原子索引。
- 映射评分只用于候选排序，不能代替化学与几何验收；配体构象或输入顺序变化后不能误用旧索引。
- 现有教师／student 的原子映射不能仅因为名字相同就视作 alchemical atom mapper。

### 5.3 Hybrid builder

- 明确 common core、A-only 和 B-only 原子，以及各自的 charge、LJ、bonded、exception／1-4、约束和 dummy 处理。
- 路径至少描述 charge、sterics 和 bonded 项；当前单配体的 `lambda_coul/lambda_vdw` 去耦规则不自动等价于 A→B 变换。
- 核对每个 λ 的有效总电荷。即使 A、B 净电荷相同，也不能因不同电荷分段切换而忽略中间态的电荷变化。
- 验证 λ 端点的物理相互作用恢复；hybrid 端点含 dummy 自由度时，明确可分离因子／修正及其在热力学循环中的抵消条件，不能仅凭端点“看起来像 A/B”判定正确。
- 保持跨 λ 相同的粒子、质量和约束结构；无法证明约束／dummy 贡献正确的变换不放行。

候选工程路线为封装经验证的 hybrid builder，或实现本项目所需的受限 builder。先做小体系对照再决定生产提供者；OpenFE/Perses 可作为参考与候选适配对象，但不代表其默认力场、采样器和全部变换已获本项目认可。官方文档也提醒某些 dummy bonded 处理存在不抵消导致系统误差的风险，应将此项纳入选择标准。[OpenFE hybrid topology 限制](https://docs.openfree.energy/en/stable/guide/protocols/relativehybridtopology.html#the-hybrid-topology-approach)

## 6. 现有代码的复用边界

可评估复用平台选择、seed 派生、通用日志、轨迹写入、数据指纹及独立的 MBAR 数值求解部分；保留这些组件既有的回归测试。

不能直接复用：

- `REMDManager` 的 ABFE System 构建分支。需先抽象为“已构建的 System + states + 初始构型 → 采样产物”的接口。
- `TraditionalMBARAnalyzer.compute_u_kn` 中按单配体去耦和 LRC 假设重建评估系统的逻辑。RBFE 必须对实际 hybrid Hamiltonian 评估。
- `TraditionalABFEPipeline` 的阶段编排及 ABFE 专用 Boresch／APBS／co-ion 自动修正。
- ABFE 的结果字段、采样 fingerprint 及 λ 调度协议版本。

`backend=auto` 复用 OpenMM >=8.6.0 的版本／能力判断原则，但只有 RBFE hybrid states 和 reporter 已完成适配时才能启用。旧版也必须有经过验证的通用 fallback；不能直接回退到一个会重新构建 ABFE 去耦系统的旧类。

第三方完整 RBFE 流程有自己的采样后端，不能假设升级 OpenMM 后它会自动使用 8.6 官方 sampler。

## 7. 输出、分析与恢复

建议每条边、每次重复独立保存：

```text
output_rbfe/A_to_B/repeat_01/
  edge_manifest.json
  atom_mapping.json
  endpoint_validation.json
  complex/  # hybrid System、状态表、样本、交叉能量、检查点、诊断
  solvent/
  rbfe_result.json
```

- 元数据包含 A/B 化学及输入身份、映射和参数 hash、hybrid builder 版本、λ 路径、平台、实际后端、有效 seeds、方向、单位和物理修正。
- 能量矩阵记录维度、状态顺序、样本状态索引、是否约化和是否包含 NPT 所需项；不按文件名猜测。
- 汇总保留两腿 ΔG 与不确定度、ΔΔG、误差估计方法、重复间方差及 qualification 状态；独立两腿时方差相加，有相关性时显式计入协方差。
- 分析结合 overlap、有效样本量、时间稳定性、交换混合和结合姿势诊断；不能只看交换接受率。
- 动态 checkpoint 与已完成采样缓存复用分开；禁止跨 A/B 方向、映射、力场、后端或 λ 路径直接追加。
- 网络阶段检查连通性和闭合环；闭合不通过需定位问题，闭合通过也不是所有系统误差均消失的证明。
- ABFE 锚点仅在受体状态、化学身份、力场、温度和自由能定义一致时用于转换绝对值，并传播锚点误差。

映射、打分、网络规划和两腿 Transformation 分层可参考 OpenFE 的官方教程；实现时固定依赖版本及输入来源。[OpenFE RBFE 网络教程](https://docs.openfree.energy/en/stable/tutorials/rbfe_python_tutorial.html)

## 8. 实施顺序与验收

| 阶段 | 代码交付 | 验收标准 |
| --- | --- | --- |
| R0 | schema、验证接口、方向明确的结果汇总、CLI validate | 错误输入被拒绝；合成两腿数据的符号／单位／误差传播正确；不启动 GPU |
| R1 | 映射和受限 hybrid builder | 映射可审计；端点相互作用及 dummy 处理有验证；有限差分力与解析力一致 |
| R2 | 通用采样接口及 RBFE 单腿 | 独立窗口先跑通，再验证 REMD；真实 hybrid 能量和样本归属一致 |
| R3 | complex+solvent 完整单边 | A→A 为零、A→B 与 B→A 相容；独立重复及参考实现对照通过 |
| R4 | 网络、恢复、生产 qualification | 三角闭合、断点恢复不混样本、既有 ABFE 回归不变 |
| 后续 | 带电／膜／大变换／IBS增强 | 每条路线独立协议和科学验收，不因接口存在自动开放 |

首个实际体系需要提供第二个配体 B 的结构／参数及可核对的结合姿势，或者明确使用的公开基准配体对。不能根据 Atenolol 文件名猜测 B，也不能把接口测试当作真实体系验证。

本次设计完成定义：四个大模块的职责、单向依赖、共享边界、数据契约和验证路线清楚，可直接作为后续实现任务书；不交付运行代码。未来实现从 R0 开始，只有 R3/R4 证据齐全才声明“RBFE 可用于生产”。