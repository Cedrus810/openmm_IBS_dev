# PME decharging v4 → v5：逐行改动记录

> 2026-09-14。本文只记**改了什么**与**验到什么程度**，不论证对错。
> 结论性叙述在 [STATUS.md](STATUS.md)，时间线在 [CHANGELOG.md](CHANGELOG.md)。
>
> ⚠️ **真机零验证由我这边完成**：我没有启动过任何 GPU run。文末「未验证」一节是完整清单。

---

## 0. 一句话

去电荷腿里，配体**分子内**（≥1-5、无 exception）的库仑不再随 λ² 一起湮灭，
改为由一个 `(1-λ²)` 前缀的 `CustomBondForce` 补回全强度，逐 λ 恒定。

协议常量：

```
v4: pme_decharge_v4_ll_exception_frozen_normal_pairs_annihilated_seam_20260831
v5: pme_decharge_v5_ll_exception_frozen_intramolecular_coulomb_decoupled_20260914
```

**破缓存。** 该常量无条件进 `ABFEPipeline._stage_protocol_key`，stage1 与 stage2 都用它
⟹ 所有已有 decharging / vanishing stage 缓存判定失配、强制重采。
预平衡不受影响（`_pre_equilibration_fingerprint` 的 payload 里没有这个键）。

---

## 1. 逐文件逐符号

### 1.1 `abfe_core.py`

| 行 | 符号 | 性质 | 内容 |
|---|---|---|---|
| 12201 | `ONE_4PI_EPS0_KJ_NM_PER_MOL_E2 = 138.935456` | **新增常量** | 1/(4πε₀)。同文件 `create_ligand_internal_force` 原有的两处内联字面量改为引用它（表达式字符串逐字节不变，已验证）。OpenMM 的 `unit` 包不暴露 ε₀，无上游权威源。⚠️ 仓库别处仍有同值副本 `ibs_engine._SHADOW_ONE_4PI_EPS0` 与**精度不同**的 `apbs_correction.COULOMB_FACTOR_KJ_NM_PER_MOL_E2 = 138.93545585`，未动。 |
| 12204 | `collect_ligand_internal_exclusions(...)` | **新增（纯提取）** | 把 `create_ligand_internal_force` 里收集配体内部 1-2/1-3/1-4 排除对的那段（键 / 角 / 刚性约束 / NB exception / 参考表）原样抽出成函数。代码未改，只是让两个消费者共用一份。 |
| 12264 | `create_ligand_internal_force(...)` | **改动** | 排除表收集改为调用上面那个函数；两处 `138.935456` 改为引用常量。**行为不变。** |
| 12356 | `LIGAND_INTERNAL_COULOMB_FORCE_NAME = "LigandInternalCoulomb"` | **新增常量** | 力的名字，用于跨 stage 去重。 |
| 12359 | `create_ligand_internal_coulomb_force(...)` | **新增** | 见 §2。 |
| 11920 | `_bake_global_parameter_into_custom_bond_force(...)` | **新增** | 把 global parameter 的取值整词代入 `CustomBondForce` 的能量表达式并重建力（OpenMM 没有 `removeGlobalParameter`）。**先建后删**——`system.getForce(i)` 返回的是 System 持有的引用，`removeForce` 会析构它，先删后读会拿到垃圾内存。 |
| 11969 | `bake_global_parameter_into_fixed_nonbonded_force(...)` | **改动（放宽守卫）** | 原来：目标参数被**任何**非 NonbondedForce 引用 ⟹ fail closed。现在：`CustomBondForce` 交给上面那个函数烘焙，其余力类型**照旧 fail closed**。契约「参数从 System 上彻底消失」不变。 |

> **2026-09-15 补丁**：上面那条改动引入过一个索引失效 bug —— 先 `removeForce` 掉
> CustomBondForce、再用**改动前**存下的 `nb_index` 去删 NonbondedForce。CustomBondForce
> 排在 NB **前面**时最后那一刀会砍到别的力上（末尾的 `remaining` 自检 fail closed 拦住了，
> 不会静默产出错的 System，但白跑一趟）。生产路径里补偿力恒为 `addForce` 追加、索引
> 恒大于 NB，**从未触发**。已改成「只建不删、全部替换在末尾按索引降序一次做完」，
> 两种顺序都已上机验过。

### 1.2 `ibs_engine.py`

| 行 | 符号 | 性质 | 内容 |
|---|---|---|---|
| 52 | import | 新增 | 从 `abfe_core` 引入 `create_ligand_internal_coulomb_force`、`LIGAND_INTERNAL_COULOMB_FORCE_NAME` |
| 3847 | `_freeze_ligand_internal_coulomb(...)` | **新增** | 见 §3 |
| 3604 | 调用点 | 新增一行调用 | `configure_charge_transfer_decharging` 末尾 |
| 3823 | 调用点 | 新增一行调用 | `configure_coalchemical_neutral_decharging` 末尾 |
| 4098 | 调用点 | 新增一行调用 | `configure_pme_ligand_charge_offsets` 中性分支末尾（**当前唯一的生产路径**） |
| 5460–5486 | `build_ibs_dual_system` | **改动** | Group 2 旁新增同一个力、前缀取 `1`、force group 2；并按**名字**去重（charge-transfer 的烘焙交接会把已固化的同一个力带过 stage 边界）。`ll_f` / `ll_14_f` 原有行为不变。 |

调用汇流：五个消费方（REMD 动力学 23695、离线 u_kn 25520、mixed 路径 2908、
`_prepare_pme_coulomb_leg_system` 1862、`abfe_pipeline` 5936）全部经过
`configure_pme_ligand_charge_offsets`，所以是单点修。

### 1.3 `abfe_pipeline.py`

**只有两处，函数体一行未改：**

- `:205` 协议常量 v4 → v5
- `:12873` 附近 `_stage_protocol_key` 里的一段**注释**，把对 v4 口径的描述更新为 v5

### 1.4 `tools/validation/compare_charge_transfer_endpoints.py`

`reference_charging_endpoint_system(..., lam=0.0)` 末尾新增：按**物理电荷**把 ≥1-5
普通 L–L 库仑全额加回（独立 `CustomBondForce`，对表在本文件里独立枚举，
**不调用**被测 builder）。λ=1 时参照就是物理体系本身，不需要动。
仍然**不补 exception**，P0-01 的约束未触碰。

### 1.5 测试

| 文件 | 性质 |
|---|---|
| `tests/test_intramolecular_coulomb_is_lambda_independent.py` | **新增**，6 条 |
| `tests/test_pme_decharge_endpoint_equivalence.py` | 独立参考实现更新到 v5（新增 `_add_reference_intramolecular_coulomb`，对表按该 fixture 自己的 `sep >= 4` 约定枚举） |
| `tests/test_charge_transfer_hamiltonian.py` | 同上，参照在 `_reference_system_at_lambda` 里独立构造 |
| `tests/test_bake_global_parameter.py` | 原 `test_bake_fails_closed_if_parameter_used_by_another_force` 用的是 `CustomBondForce`（现已可烘焙），改名为 `..._by_an_unbakeable_force` 并换成 `CustomExternalForce`；新增 `test_bake_substitutes_the_value_into_a_custom_bond_force`（λ=0/1 两参数） |

---

## 2. `create_ligand_internal_coulomb_force` 的确切形态

```
CustomBondForce("({scale_expr})*138.935456*chargeProd/r")
  per-bond 参数: chargeProd = q_i * q_j   （**原始**电荷，不是被清零/缩放后的）
  global 参数:   由 global_parameters 入参注册（OpenMM 要求每个 Force 各自声明）
  name:          "LigandInternalCoulomb"
  PBC:           False
  对表:          配体全部两两组合 − collect_ligand_internal_exclusions(...)
  自检:          n_普通对 + n_排除对 != n_全部对 ⟹ RuntimeError
  qq == 0 的对不加 bond；一条 bond 都没有则返回 None
```

用法：

- 去电荷腿：`scale_expr = f"1 - {lambda_name}^2"` ⟹ 主 NB 给 λ²·U_intra，本力给
  (1−λ²)·U_intra，合计恒为 U_intra；**λ=1 时本力恒等于 0**，物理端点逐位等于原 System
- vanishing 腿：`scale_expr = "1"` ⟹ 配体已整体从主 NB 剥离，本力就是全部

### 为什么 PBC = False

与 `NonbondedForce` 的 exception **同口径**。OpenMM 的 exception 不做最小镜像，
它假定成键原子总在同一个周期像里 —— 这一条由
`tests/...::test_nonbonded_exceptions_ignore_minimum_image` 现场建体系现场算证明，
不抄任何实测数字。

这条来回改过一次：09-14 一度改成 `True`，随后撤回。撤回理由：配体内部这一段唯一
正确的前提是「坐标到达时分子是完整的」；前提成立时最小镜像与直接距离逐位相同，
加 PBC 不解决任何问题；前提被破坏时，加 PBC 只让分子内两半口径不一致
（本项「对」、1-2/1-3/1-4 仍然错）。另外最小镜像在配体跨度 > L/2 时会静默给错值。

### 为什么不是 CustomNonbondedForce

后者必须带 cutoff；Group 2 的 `ll_force` 用的是 1.0 nm。跨度超过 1 nm 的配体会被
截掉最长的内部对，而去电荷腿那侧的 PME 不截断 —— 两边口径就不一致。

---

## 3. `_freeze_ligand_internal_coulomb` 的前提校验

三个 decharging builder 共用。挂力之前**逐个配体原子**校验，任一条不满足即 fail closed：

1. base 电荷必须为 0（否则 q_i(λ)·q_j(λ) 不是 λ²·q_i·q_j，(1−λ²) 补偿就不精确）
2. 该原子在 `lambda_name` 下**有且只有一条** particle parameter offset
3. 该 offset 的 scale 必须等于记录的原始电荷

（`co_alchemical_charge_offset_plan` 给配体原子的 plan 恒为 `(0.0, q)`，
只有 co-ion 的 base 可能非 0，而 co-ion 不是配体原子，所以三条路线都满足前提。）

---

## 4. 已验证

| 项 | 方式 | 结果 |
|---|---|---|
| λ=1 端点不变 | `test_pme_decharge_endpoint_equivalence`（真实 PME、能量+逐原子力，容差 1e-6） | 过 |
| 分子内库仑 λ 无关 | 新测试，**自校准**（摘掉补偿项的同一体系做对照臂），做过还原变异 | 过 |
| 残差只剩周期自镜像项 | 盒 4→8 nm 残差衰减 | 过 |
| 两个 stage 共用同一份排除表 | 断言排除集 == NB 的 L–L exception 集 | 过 |
| PBC 口径与 exception 一致 | 现场建体系现场算 | 过 |
| 烘焙交接 | `test_bake_global_parameter`，数值判据 = 「烘焙后 == 显式设 λ 的原 System」 | 过 |
| 相关测试面 | 101 条 | 全绿 |
| 全套离线 | 2389 passed / 0 failed / 3 skipped（**并发编辑之前的那次**） | 见 §6 |

---

## 5. 生产侧观察到的数（**不是我跑的**）

2026-09-14 20:02 起，`cyclod_ligand2` 的 decharging 被以 v5 重跑（CUDA，远端节点）。
`decharging_sampling.meta.json` 的 `pme_decharge_model_version` 已是 v5。

| | v5 stage1 | ± | converged | \|BAR−TI\| | TI 门 | v4 stage1 |
|---|---|---|---|---|---|---|
| rep2 | 70.28 | 0.56 | True | 0.224 | 过 | 623.94 |
| rep3 | 76.17 | 0.60 | True | 0.235 | 过 | 626.36 |

⚠️ 这两行是从 `checkpoints/stage1_decharging.json` 读出来的既成产物，
**不是**我发起或监督的实验。ΔG_bind 尚无 v5 数值。

---

## 6. 未验证 / 未解决

1. **v5 的 ΔG_bind 一个数都没有。** 我没有跑过任何 GPU run。
2. **`cyclod_ligand1/rep1` 与 `rep2` 的 decharging 帧有约半数不满足 System 约束**
   （偏差到 1.8 nm，MIC 后仍不满足；违反原子恒在索引末尾 3%）。
   同一份 v5 代码下 cyclod_ligand2 三个 rep 的同一检查全部 0/400。
   **原因未查明。** 我不能把它归给 v5，也不能排除。
3. ~~**`tests/test_charge_transfer_hamiltonian.py::test_ligand_internal_energy_is_quadratic_in_lambda_without_pme_bookkeeping`
   已退化为空转**~~ **已修（2026-09-15）**：它钉的是 v4 契约，在 v5 下量到的
   `intra = 2.83e-6 kJ/mol`（27156 kJ/mol 总能上的浮点噪声），刚好越过它自己
   `> 1e-6` 的守卫，而且噪声本身也 ∝ λ²，两条断言双双通过 —— v4/v5 都绿。
   已重写为 `test_ligand_internal_coulomb_is_lambda_independent_under_charge_transfer`：
   两条臂（摘掉补偿力 = v4 口径给量级 239.45 kJ/mol；装上必须压到 1e-5 倍以下），
   容差由对照臂自校准、不写绝对魔法数，并把恒定性断在整段 λ 上而不只两个端点。
   突变验证过：把被测臂也摘掉补偿力，测试变红（239.447 > 容差 0.00239）。
4. **配体内部库仑仍有两份实现、两种截断口径**：本文这个（逐对、无截断）与
   `create_ligand_internal_force` 的 `ll_force`（CustomNonbondedForce、1.0 nm 截断，
   被 shadow-coul / bridge 两个 builder 用来承载分子内库仑）。
   shadow 路径本身没有湮灭缺陷，但它与 stage2 的接缝口径不同。未统一。
5. **`abfe_core` 里仍有 4 处 `138.935456` 字面量**（`_build_*` 系列的表达式），
   以及 `apbs_correction` 里精度不同的那一份。未动。
6. **全套离线测试的 2389/0 是并发编辑之前的那次。** 之后另有会话在改
   `abfe_pipeline.py` / `abfe_preoptimizer.py`，期间跑出过 23 条红，
   全部是断言源码文本的那类，与本改动无关。当前基线需要在树稳定后重跑。

---

# 附录 A：完整调查记录（从"能量算错"到 v5）

> 本节记录**怎么查到的**，包含我中途说错又更正的部分。
> 之前这些只存在于会话里，没有落盘 —— 那是我的疏漏，这里补齐。

## A.1 起点：症状

abfe-benchmark `cyclod_ligand2`（CypD，实验 **−4.04 kcal/mol**，
`experimental.csv`：Kd 1.1 mM、CypD K175I、DOI 10.1016/j.bmcl.2019.126717）：

| rep | complex | solvent | ΔG_bind |
|---|---|---|---|
| rep1 | 541.94 | 527.31 | −3.49 kcal/mol |
| rep2 | 632.30 | 527.35 | **−25.08 kcal/mol** |
| rep3 | 635.48 | 526.60 | **−26.02 kcal/mol** |

rep2/rep3 互相吻合、rep1 离群 90 kJ/mol。

## A.2 第一层：rep1 是部分和（与 v5 无关，已独立成立）

rep1 的 `stage_diagnostics.stage2`：

```
coverage_diagnostics.input_window_indices = [5]
covered_lambda_indices = [17..23]      （7 个，不是 21 个）
covariance_chain_segments = 1 段, ΔG = -49.04
```

rep2/rep3 是 `[0..20]`、6 段、+44.09 / +40.97。

所以 rep1 的 vanishing 是**只解了末窗一个窗口的部分和**，被当成完整 ΔG。
rep1 自己的全路径中间结果是 **+43.00**（与 rep2 的 +44.09 差 1.1）。

**连带**：`docs/STAGE2_CONTROLLER_DESIGN_2026-09-12.md` 的头条
ΔG_bind = −3.48 ± 0.47 kcal/mol（1.19σ）**就是 rep1**，已作废。
它"对上实验"是两个 ~90 kJ/mol 的错误反号抵消：部分和 +92 / 分子内湮灭 −88.7。
用 rep1 自己的全路径重算 → **−25.49 kcal/mol**，与 rep2/rep3 一致。

## A.3 第二层：两腿之差在 stage 1 与 stage 2 之间怎么分（**这是分解，不是归因**）

现场重算的 stage 分解（`decoupling_delta_G` − `Σ covariance_chain_segments`）：

| | complex | solvent | 差 |
|---|---|---|---|
| stage1（decharging） | 623.94 | 532.29 | **−91.65** |
| stage2（vanishing） | +44.09 | −4.94 | −49.03 |
| Boresch 修正 | −35.73 | 0 | +35.73 |
| | | | −104.95 = −25.08 kcal/mol |

**stage2 + Boresch 单独 = −13.30 kJ/mol = −3.18 kcal/mol**。

> ⚠️ **更正（2026-09-15）。本文初版在这里写「误差 100% 在 stage 1」「问题全在
> stage 1」，那是错的，已删。**
>
> 上表是一个**分解**：它说的是"两腿之差怎么分摊到三项"，这是算术恒等式。
> 它**不能**证明"错误落在 stage 1"：
>
> 1. **「stage2 + Boresch ≈ 实验值」不是证据，是巧合的分割方式。** 真值没有理由
>    恰好等于某两项之和 —— 把三项按别的方式分组同样能凑出别的数。
> 2. **本文 A.7 自己承认 v4 的循环在充分采样下是闭合的。** 既然如此，
>    "某一个 stage 承载了全部误差"这句话本身就需要独立测量来支持，
>    而不是从分解表里读出来。
> 3. 真正被这张表支持的结论只有一条：**两腿之差的绝对值主要落在 stage 1**
>    （−91.65 vs −49.03 / +35.73），而 stage 1 的绝对量级（+624 / +532）
>    本身不合理 —— 这才是下一节去查 stage 1 的理由。
>
> **误差的归属仍未独立测定。**

## A.4 第三层：stage 1 里夹着什么

stage1 的绝对值本身不合理：一个 35 原子、Σq²=3.21 e²、**净电荷 0** 的配体，
水里 decharging 给 +532 kJ/mol，量级差一个数量级。

排除项（都已现场核实）：

- 净电荷 = 0（两腿电荷表逐位相同，`max|Δq| = 0`）⟹ 不是 PME 净电荷伪影
- `pme_self_correction.applied = False`（诊断项，未施加）
- 两腿 `charge_square_sum_e2` 相同

剩下唯一能吃掉几百 kJ/mol 的是**配体分子内库仑**。
`decharging_sampling.meta.json` 的
`pme_decharge_model_version = pme_decharge_v4_ll_exception_frozen_normal_pairs_**annihilated**_seam_20260831`
明写了它被湮灭。

## A.5 量化（⚠️ 见 A.8：complex 侧已不可复现）

从 DCD + System XML 直算 ≥1-5 普通 L–L 对的库仑，400 帧/腿：

```
<U_intra> complex = -539.91 ± 1.32   Rg(全原子) 0.457 nm
<U_intra> solvent = -451.25 ± 1.04   Rg(全原子) 0.364 nm
               差 = -88.66 kJ/mol

ΔG_bind 对实验的误差 = -88.05 kJ/mol
```

拆开 stage1：ligand–environment 那一半两腿只差 **+2.99 kJ/mol**（正常），
配体内部那一半差 88.66（全部误差）。扣掉后 ΔG_bind ≈ −3.89 kcal/mol（一阶估计）。

**已确认无 PBC 撕裂**：两腿轨迹里配体最大内部距离 1.35 / 1.04 nm，无 >2 nm 的帧。

## A.6 为什么一直没暴露

**4W53/toluene —— 本仓库唯一验过实验值的体系 —— 对这个失效模式免疫**：

```
4W53:  Σq² = 0.4992 e²,  stage1 complex -24.19 / solvent -25.75  ⟹ 两腿差 +1.56 kJ/mol
cyclod_ligand2: Σq² = 3.2109 e², 刚性 vs 柔性多极性          ⟹ 两腿差 -88.66 kJ/mol
```

⚠️ **别再拿「4W53 对上实验」当作 stage 1 正确的证据。**

**所有质量门也看不见**：88 kJ/mol 是两条**独立腿之间**的差，没有任何一道门跨腿。
单腿内的 overlap / ESS / split-half / target_support 全绿，rep2 报的
`total_error = 1.83 kJ/mol`。

## A.7 定性：我中途说过头、后来更正的

初版我写成「v4 的推理是错的 ⟹ v4 算错了」。**更正如下**，以这一版为准：

- **v4 的热力学循环形式上是闭合的** —— 两腿终态都是「内部库仑被湮灭的配体」，
  同一个参考态。完美采样极限下 v4 与 v5 给同一个 ΔG_bind。**v4 不是公式错。**
- 假的是 v4 注释里「湮灭项在 ΔG_bind 里**严格相消**」这四个字：它不是恒等抵消。
  哈密顿量相同不等于自由能相同 —— 自由能是系综平均，而结合态（Boresch 约束在
  口袋里）与自由态（体相水）的配体构象系综不同。写成「严格相消」等于宣称它不必
  被采样，那正是 v4 被放行的根据。
- 真正的失效是**条件数**：把一个 ~90 kJ/mol 的物理量做成两个 ~500 kJ/mol 项之差。
- **v5 不删物理**（⚠️ 这条 2026-09-15 更正过，见下）：参考态从「内部静电被删掉的
  虚构分子」变回真实分子在真空中。

> ⚠️ **更正（2026-09-15）：本文初版在这里写「该项变 λ 无关 ⟹ 对两腿 ΔG 各贡献 0」，
> 那是错的，已删。**
>
> λ 无关只意味着它的**直接 λ 导数为零**，不意味着它对自由能没有贡献。
> 自由能是配分函数之比：
>
> ```
> ΔG(stage1) = −kT ln [ Z(λ=0) / Z(λ=1) ],   Z(λ) = ∫ exp(−β[U_intra + U_λ(λ)]) dx
> ```
>
> `U_intra` 同时出现在分子和分母里，但它**不会约掉** —— 它改变每个 λ 下的
> 玻尔兹曼权重、从而改变构象分布，而 `U_λ(0) ≠ U_λ(1)`，所以两个积分被重加权的
> 方式不同。**加上或去掉这一项会改变每条腿的 ΔG(stage1)。**
>
> 真正能说的只有一条、而且要说清条件：在 **stage 2 末端的完全解耦参考态**上，
> 配体与环境无任何耦合，它的内部自由能不依赖于所处环境，因此在
> ΔG_bind = ΔG_solvent − ΔG_complex 里两腿相消。
> **这只对最终的 ΔG_bind 成立，不对单个 stage 成立。**
>
> 同样的错误也出现在下一段「不放进去按构造就是 0」那句话里，一并作废。

**不依赖任何数字吻合的论据**（⚠️ 已按 2026-09-15 的更正改写，去掉"按构造就是 0"
这个错误说法）：把一个 ~500 kJ/mol 的项放进 λ 路径，意味着它在每条腿里都必须被
采样收敛、再靠两腿相减来消掉大部分；把它移出 λ 路径并不让它"贡献为零"（见上面的
更正），但确实让 stage 1 承载的量级从 ~600 降到 ~80，相减时要求的相对精度低一个
数量级。具体代价三条：
① 中间态采的是不存在的分子（λ_coul=0.5 时内部静电只剩 25%）；
② 把体系最慢的自由度（构象转变）塞进一条没有任何增强采样手段的腿，而且它在复合物
里被口袋挡住、在溶剂里没挡；③ 要 ΔG_bind 准到 1 kJ/mol，得让一个 500 的项在**每条
腿里**都收敛到 0.2% 且两腿偏差相关。

## A.8 事实核查结果（用户要求的全量复核）

**核实无误（现场复现过）**：实验值来源、三个 rep 的 stage 分解、rep1 的部分和、
4W53 的 +1.56 与 Σq²、solvent ⟨U_intra⟩ = −451.25（两次测量完全一致）、
内部 LJ 接缝恒等、版本号进指纹的路径、三个 builder 里 `addException` 为 0 次、
`_freeze_ligand_internal_pairs` 无任何调用、`apply_pme_self_correction` 仍恒为 False。

**已不可复现**：
> 三个 rep 的 **complex** decharging 帧全部在 2026-09-14 20:02 被 v5 重跑覆盖。
> 因此 **⟨U_intra⟩ complex = −539.91 与由它得出的 −88.66 无法再验证**。
> 那是 19:00 在原始帧上量的，当时有效，但证据已不存在。
> 用 rep3 做的独立检验也因此做不成。

**我说错的**：
1. 「v4 错了」—— 过头，见 A.7。
2. 拿 Δ⟨U⟩（平均势能差）与 ΔΔG（自由能差）的 1:1 吻合当证据 —— 两类量，
   这个吻合只在「零重组、纯线性响应」的假设下才该出现，**有循环论证成分**。
3. **叙述矛盾（未解决）**：我说口袋里的配体「更内部稳定」。但冻结在 final_results
   里的 `ligand_conformer_diagnostics`（未被覆盖，三个重复一致）显示：

   | | Rg(重原子) | max 内部距离 | 内部极性接触 |
   |---|---|---|---|
   | complex rep1/2/3 | 0.3973 / 0.3975 / 0.3966 | 1.178 / 1.179 / 1.181 | 0.000 / 0.005 / 0.005 |
   | solvent rep1/2/3 | 0.3236 / 0.3233 / 0.3223 | 0.868 / 0.868 / 0.859 | **0.108 / 0.098 / 0.095** |

   口袋里更**伸展**、几乎**没有**内部极性接触，内部库仑却更负 —— 与直觉相反。
   方向在两批独立帧上都成立（旧帧 −539.91、v5 新帧 −590.22，对溶剂 −451.25），
   **但我对「为什么」的结构解释是错的，真实原因未知。**
4. 量玩具 fixture 的残差时手算 Ewald 自能 C 去扣 —— **方法错**（数字碰巧对）。
   自能项与倒空间自相互作用精确相消，不是可单独加减的物理量；本仓库为此撤销过一次
   `+C·λ²`（见 `ibs_engine.TraditionalMBARAnalyzer.compute_u_kn` 注释，当年把
   decharging 腿拉到 −954.81）。已改成直接读 E(λ=0)−E(λ=1) 且必须带未补偿对照。

**「两腿构象系综显著不同」这个机制本身站得住**（三个重复一致，证据冻结未被覆盖）；
**具体那个 88.66 已经没有证据支撑了。**

## A.9 修完之后的残差（真实体系，带未补偿对照）

环境电荷清零后读 E(λ=0)−E(λ=1)（此时全部 λ 依赖都是配体分子内的）：

| | 未补偿（v4 口径） | 补偿后（v5） | 盒 |
|---|---|---|---|
| complex | +617.80 | **+0.0013** | 6.75 nm |
| solvent | +479.18 | **+0.0436** | 4.11 nm |
| 两腿之差 | | **−0.0423 kJ/mol** | |

即 0.01 kcal/mol。剩下的是配体与自己周期镜像的相互作用，PME 下不可消除
（GROMACS 的 `couple-intramol=no` 同样留着）。

⚠️ 最小 fixture 上这个残差是 0.22–5.22 kJ/mol，**比真实体系大 1–2 个量级**
（人为拉直的 8 原子链，偶极远大于真实配体 μ = 2.55 / 4.83 D、Rg 0.41–0.46 nm）。
**别拿 fixture 的残差量级推断生产体系。**

## A.10 顺带核实过、结论是"没问题"的两条

1. **内部 LJ 的 stage 接缝**：怀疑 stage1（主 NB）与 stage2（Group 2）截断口径不同。
   实测真实体系主 NB 是 **1.0 nm、无 switching**，与 `BEUTLER_SOFTCORE_CUTOFF_NM`
   一致（MEM-00h 当初统一过）⟹ **精确恒等，两腿差 0.0000**。证伪。
2. **shadow-coul / bridge 路径是否也有湮灭缺陷**：没有。配体电荷在主 NB 里被永久
   清零，分子内库仑放在 `ll_f` 里、λ 无关。

## A.11 实现过程中抓到的三个真 bug（都是测试第一次跑就炸出来的）

1. `CustomBondForce` 没在**自己**身上注册 global parameter（OpenMM 要求每个 Force
   各自声明）⟹ 建 Context 时 `Unknown variable: lambda_coul`。
   已收成 builder 的入参，不留给调用方 —— 「注册是 caller 的活」这个形状在本仓库
   出过静默算错的事故（EXP-025 G4 Layer-1）。
2. 烘焙器里**先 `removeForce` 再读 `force` 句柄** ⟹ SWIG 悬垂引用，
   能量表达式读出来是乱码字节。必须**先建后删**。
3. `configure_pme_ligand_charge_offsets` 的**中性分支**里 `original_charges` 未绑定
   （只在带电分支由子 builder 返回），得用 `ligand_params`。

## A.12 PBC：改了又撤（全过程）

- 初版：`CustomBondForce` 非周期，并写了注释论证「配体是单个连通分子，OpenMM 不会
  在动力学中把它拆到不同周期像里」。
- 中途改成 `True`，理由是坐标可能从已折叠的来源载入。
- **撤回**。撤回依据是一条现场可跑的判据（已固化为
  `test_nonbonded_exceptions_ignore_minimum_image`）：OpenMM 的 `NonbondedForce`
  exception **不做最小镜像**，它用直接距离。所以配体内部这一段唯一正确的前提是
  「坐标到达时分子是完整的」（靠平移补全，PBC-01）。前提成立时最小镜像与直接距离
  逐位相同，加 PBC 不解决任何问题；前提被破坏时，加 PBC 只让分子内两半口径不一致
  （本项「对」、1-2/1-3/1-4 仍然错）。另外最小镜像在配体跨度 > L/2 时会静默给错值。
- 现状：`False`，与 exception 同口径，注释与测试都写死「不要改成 True」。

## A.13 魔法数字清理（用户指出后）

- 我一度把自己现编的 2 粒子玩具体系的实测值（`173.669` / `9.140` / `0.25` / `3.8`）
  抄进 docstring —— 仓库里没有这个 fixture，谁也复现不了。**已改成现场建体系、现场
  算的测试**，判据是 `ke*qq/(L-d)` 与 `ke*qq/d`，改盒长或改电荷都不会失效。
- 对照臂的 `> 1.0` 下界改成由 fixture 自身内部库仑现算（`0.1 × |U_intra|`）。
- 残差数表从测试注释里删除，只在 STATUS.md 留一份。
- 我新增的 `LIGAND_INTERNAL_COULOMB_KE_KJ_NM_PER_E2` 是第 9 个 `138.935456` 副本，
  **已删**；合并成 `ONE_4PI_EPS0_KJ_NM_PER_MOL_E2` 一份，`create_ligand_internal_force`
  那两处内联改为引用（表达式字符串逐字节不变，已验证）。abfe_core 从 5 处降到
  4 处 + 1 个定义。

## A.14 `cyclod_ligand1` 的 32924 崩溃（**未解决**，与 v5 无关但未证明无关）

现场报错（`cyclod_ligand1/rep2`，在 v5 指纹失配之后的行号）：

```
[ERR] [TI 门] BAR 与重加权 FD-TI 分歧 32924.5581 > 容差 0.5000 kJ/mol ⟹ converged=False
RuntimeError: Stage 1 (decharging) 阶段 total_error=nan
```

查到的链条：

1. `decharging_pme_u_kn.npy` 里 **1696/3200 个元素 > 1e6，最大 6.14e20**
2. 轨迹里全体系最小原子间距到 **0.0077 nm**
3. 用 **System 自己的约束表**（29395 条，与 topology.cif 无关）逐帧核：
   帧 4/12/16/28/32/44/48 违反 **0** 条（偏差 1e-6）；
   帧 0/8/20/24/36/40 违反 **395–878** 条，最大偏差 5.66 nm
4. 违反的原子**恒在索引末尾 3%**（29706–30653 / 共 30710）
5. 全帧全副本：`cyclod_ligand1/rep2` 坏帧 **184/400**

已排除的解释（逐条测过）：
- **不是 PBC 折叠**：逐帧用该帧自己的盒做最小镜像，违反数**一条不减**，MIC 后偏差
  仍到 1.8 nm；盒是立方 6.8624 nm（System 与 DCD 一致），**不是三斜**
- **不是陈旧坐标**：坏帧尾部与任何其它帧都不相同
- **不是原子平移错位**：尾部平移 ±1..6 个原子，坏数仍是 272–288/288
- **不是 X/Y/Z 块混写**：各分量尾部都最接近它自己（rms=0）
- **不是我的补偿力**：对表正确（193 = 351 − 158），最短对 0.1955 nm、电荷 0.03，
  配体本身完好（最大内部 0.90 nm）；而且它只作用在配体—配体对上，推不动两个水分子
- **不是拓扑识别错**：用 System 约束表（不读 topology.cif）复核，结论相同；
  且干净帧证明原子映射本身是对的

**同一份 v5 代码下的对照**：`cyclod_ligand2` rep1/rep2/rep3 各 **0/400** 坏帧，
stage1 收敛（70.28 / 76.17，TI 门 |BAR−TI| = 0.224 / 0.235，均通过）。

**结论：原因未查明。** 我不能把它归给 v5，也不能排除。
背景（同僚会话提供，未由我核实）：该 run 12:05 在运行中被删过产物、17:57 被
ValueError 打死、18:20 被重新拉起、`path_current.json` 版本被重置 2→1。

**未做的动作**：读帧时的完整性检查（用 System 约束表 fail-closed）——
用户明确说**不加**。
