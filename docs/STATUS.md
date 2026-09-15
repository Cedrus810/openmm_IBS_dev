# 当前科学状态与结果登记

[项目入口](../README.md) · [文档导航](README.md)

> **本页整理截至 2026-09-14**（协议版本一栏 09-12 逐个对过源码常量，六个值与 09-09 相同；
> 09-13/14 的改动**没有动任何协议版本号**，也不破缓存）。
> 逐日变更看 [CHANGELOG.md](CHANGELOG.md)。
>
> 这是本仓库**唯一**声明"当前科学结论"的文档。三份 README 只对外讲用法，
> 不再复制这里的任何表格——2026-09-05 之前同样的三张表在 `README.md`、
> `README_cn.md`、`README_en.md`、`docs/README.md` 各存了一份，
> 热力学路径版本已经在其中三份里烂成了旧值。别再复制。

## 主线体系

**4W53（T4 lysozyme L99A + toluene）**，不再是 Atenolol。

符号约定：

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

## 结果登记

| 数值 | 体系 / 运行 | 状态 | 能否作为最终结论引用 |
|---|---|---|---|
| **`−21.36 ± 0.93 kJ/mol`**（`−5.11 ± 0.22 kcal/mol`） | 4W53，`output_v3_seed20260908`，2026-09-02 | **注册标签待维护者指定** | **否：单 seed（`20260908`），本仓库内无第二个独立重复** |
| `541.94 ± 1.30 kJ/mol`（复合物腿 total，`cyclod_ligand2/rep1`） | abfe-benchmark，2026-09-12 | **已作废（2026-09-14 判定）** | **否：vanishing 项是「只解了末窗一个窗口」的部分和**（`covered_lambda_indices=[17…24]`），被当成完整 ΔG 写进 stage 缓存并被 `final_results` 采信。同体系完整路径是 `+44.09`，该 run 自己的全路径中间结果 `+43.00` —— 差的 90 kJ/mol 全部来自这次冒充。两道新门（子集不得自称裁决 / 覆盖度硬门）已落地 |
| `−23.1622 ± 2.5139 kJ/mol`（`output_lrc_fix`） | Atenolol-rank11 | **已作废（2026-08-24 判定）** | 否 |
| `+40.8362 ± 1.3178 kJ/mol`（旧 `output`） | Atenolol-rank11 | `INVALIDATED` | 否：旧符号约定与当前相反，且有诊断问题 |
| `+16.00 ± 2.20 kJ/mol`（2026-07-27） | Atenolol-rank11 | `INVALIDATED` | 否：陈旧且错误的 Boresch 平衡几何 |

4W53 那一行的对照：实验值 **−23.10 kJ/mol**（`−5.52 ± 0.04 kcal/mol`），
差 **0.41 kcal/mol、1.83σ 内**；质量门同时转健康（溶剂腿 stage2 raw ESS
2.93 → 173.33，top1% 0.828 → 0.047）。逐项证据见
[archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md](archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md)。

文件名包含 `final` 不代表结果可以引用。Atenolol 那三行的原始 artifact 和机器可读
登记表（`RESULT_REGISTRY.csv`）在 `Atenolol-rank11` 工作区，不在本分支；
4W53 那一行的证据在本 `docs/` 里。

## ⚠️ decharging 湮灭配体分子内库仑（2026-09-14 定位并修复，**所有已有 ΔG 作废**）

> **改了哪些文件哪些符号，逐行记录在**
> [PME_DECHARGE_V5_CHANGE_RECORD_2026-09-14.md](PME_DECHARGE_V5_CHANGE_RECORD_2026-09-14.md)。
> 那份文档同时列出**未验证 / 未解决**的六条，包括一条已退化为空转的旧测试。

`PME_DECHARGE_MODEL_VERSION` v4 → **v5**（`..._intramolecular_coulomb_decoupled_20260914`）。
**破缓存**：所有 decharging 采样与 u_kn 必须重跑。

### 症状

abfe-benchmark `cyclod_ligand2`（CypD，实验 **−4.04 kcal/mol**）：

| rep | complex | solvent | ΔG_bind |
|---|---|---|---|
| rep2 | 632.30 | 527.35 | **−25.08 kcal/mol** |
| rep3 | 635.48 | (527.33) | −25.85 kcal/mol |

系统性过度结合 **21 kcal/mol**，且 rep2/rep3 互相吻合 —— 是可复现的偏差，不是噪声。

### 根因

误差 **100% 在 stage 1（decharging）**，不在 vanishing：

| | complex | solvent | 差 |
|---|---|---|---|
| stage1 | 623.94 | 532.29 | **−91.65** |
| stage2 | +44.09 | −4.94 | −49.03 |
| Boresch | −35.73 | 0 | +35.73 |

stage2 + Boresch 单独 = −13.30 kJ/mol = **−3.18 kcal/mol**，与实验相符。

v4 把配体内部的普通 ≥1-5 L–L 对连同配体–环境对一起按 λ² 缩放（annihilation），
于是 stage 1 里夹了一个 ~500 kJ/mol 的分子内库仑项。

**先把定性说准**（这一段 2026-09-14 晚更正过，原先写得会让人读成「v4 形式上就错了」）：

* v4 的热力学循环**形式上是闭合的** —— 两腿的终态都是「内部库仑被湮灭的配体」，
  是同一个参考态。完美采样极限下 v4 与 v5 给同一个 ΔG_bind。
* 但 v4 注释里那句「湮灭项在 ΔG_bind 里**严格相消**」仍然是**假的**：它不是恒等抵消。
  哈密顿量相同不等于自由能相同 —— 自由能是系综平均，而结合态（Boresch 约束在口袋里）
  与自由态（体相水）的配体构象系综不同，这一项的两腿之差是**真实的构象重组功**，
  必须靠采样得到。写成「严格相消」等于宣称它不需要被采样，这是 v4 放行的根据。
* 真正的失效是**条件数**：把一个 ~90 kJ/mol 的物理量做成两个 ~500 kJ/mol 项之差，
  而结合态配体在 500 ps × 8 个 λ 态里完不成构象弛豫 ⟹ 滞后，估计器报的是**未弛豫的
  平均能差**（下面 1:1 的吻合正是这个特征：零补偿、纯线性响应）。
* **v5 不是删掉物理**：该项变成 λ 无关 ⟹ 对两腿 ΔG 各贡献 0，重组功改走配体–环境
  那条条件良好的路。顺带参考态从「内部静电被删掉的虚构分子」变回**真实分子在真空中**。

实测（400 帧/腿，直接从 DCD + System XML 算）：

```
<U_intra> complex = -539.91 ± 1.32 kJ/mol   Rg 0.457 nm（伸展）
<U_intra> solvent = -451.25 ± 1.04 kJ/mol   Rg 0.364 nm（塌缩）
               差 = -88.66 kJ/mol
ΔG_bind 对实验的误差 = -88.05 kJ/mol      <- 1:1 吻合到 0.7%
```

拆开 stage 1：ligand–environment 那一半两腿只差 **+2.99 kJ/mol**（正常），
配体内部那一半差 88.66（全部误差）。扣掉后 ΔG_bind ≈ **−3.89 kcal/mol**（一阶估计）。

### 为什么一直没暴露

**4W53/toluene —— 本仓库唯一验过实验值的体系 —— 对这个失效模式免疫**：
Σq² = 0.499 e²、刚性，stage1 两腿只差 **+1.56 kJ/mol**。
cyclod_ligand2 是 Σq² = 3.21 e²、柔性多极性。
⚠️ **别再拿「4W53 对上实验」当作 stage 1 正确的证据。**

质量门也看不见：88 kJ/mol 是两条**独立腿之间**的差，没有任何一道门跨腿。
单腿内 overlap / ESS / split-half / target_support 全绿，rep2 报的 total_error 是 1.83 kJ/mol。

### 修法

`_freeze_ligand_internal_coulomb`（`ibs_engine.py`）给去电荷系统挂一个 (1-λ²) 前缀的
`CustomBondForce`（`abfe_core.create_ligand_internal_coulomb_force`），逐对展开、不带
cutoff：主 NB 力给出 λ²·U_intra，补偿项给出 (1-λ²)·U_intra，合计逐 λ 恒定。
λ=1 时补偿项恒为 0，物理端点仍逐位等于原 System（P0-01 不变量不受影响）。
vanishing 腿同一份对表、前缀取 1，Stage-1/Stage-2 接缝按构造恒等。

**残留（PME 下无法消除）**：配体与自己周期镜像的相互作用同样按 λ² 走，逐对补偿
只覆盖主镜像。GROMACS 的 `couple-intramol=no` 同样留着这一项。

真实体系实测（`cyclod_ligand2/rep2` 的 decharging 首帧，环境电荷清零后
E(λ=0)−E(λ=1)，**带未补偿对照**）：

| | 未补偿（v4） | 补偿后（v5） | 盒 |
|---|---|---|---|
| complex | +617.80 | **+0.0013** | 6.75 nm |
| solvent | +479.18 | **+0.0436** | 4.11 nm |
| 两腿之差 | | **−0.0423 kJ/mol** | |

即 0.01 kcal/mol，比它取代的 −88.66 小 2000 倍，远在误差棒之下。
⚠️ 最小 fixture 上这个残差是 0.2–5 kJ/mol，**比真实体系大 1–2 个量级** —— 那是人为
拉直的 8 原子链、偶极远大于真实配体（实测 μ = 2.55 / 4.83 D，Rg 0.41–0.46 nm）。
**别拿 fixture 的残差量级去推断生产体系。**

验证：`tests/test_intramolecular_coulomb_is_lambda_independent.py`（自校准对照，
已做还原变异验证）。

### 连带作废

- 4W53 那一行 `−21.36 ± 0.93 kJ/mol` 与实验 1.83σ 的吻合：stage1 影响只有 1.56 kJ/mol，
  **结论方向不变**，但数字需在 v5 下重跑后才能再引用。
- `docs/STAGE2_CONTROLLER_DESIGN_2026-09-12.md` 的头条 **−3.48 ± 0.47 kcal/mol（1.19σ）
  已作废**：它是 `cyclod_ligand2/rep1`，而该 run 的 vanishing 是只覆盖 λ 17→23 的部分和。
  它"对上实验"是**两个 ~90 kJ/mol 的错误反号抵消**（部分和 +92 / 分子内湮灭 −88.7）。
  用它自己的全路径重算 → **−25.49 kcal/mol**，与 rep2/rep3 一致。

## Stage-2 自治控制器：代码已收口，**真机零验证**（2026-09-14）

2026-09-14 一天之内，控制器经历了**两轮静态复核 + 五次真机日志驱动的修复**：

| 批次 | 内容 | 状态 |
|---|---|---|
| 真机 bug | 路由信号炸管线、拼接失败炸管线、分析用错 f_k 字段、四起 no-op 死循环 | 已修 |
| 复核第一轮 | 子窗可调度、生产预算独立成账、S2-A 逐段门、DECORR 同输入对照 | 已修 |
| 复核第二轮 | `CTL-01`~`CTL-10`（读旧结果 / 过期预算账 / 调度状态≠归因 / 身份隔离 / 预算准入 …） | 已修 |
| 六路并行全面审计 | **63 条**缺陷（三条贯穿性根因：路由在知道窗口缺什么之前就锁定 / 「未知」四处四套语义 / 子窗是二等公民） | **44 条已修，18 条仍 `OPEN`**，逐条状态列见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md) |
| 审计后新增 | **#64** win0 空转（补帧目标每涨一次就把 EM + dt 测试 + Boresch 爬坡 + 冻结 burn-in + 只读复验整条前置链白跑一遍再被 checkpoint 逐字覆盖）、**#65** manifest 改名而消费者没跟 ⟹ A/B 报告全空且不报错 | 均已修（#64 的 P3 判定不做）|

`./tests/run_offline_tests.sh` **2383 passed / 3 skipped / 0 failed**；每一条都做过
**还原变异验证**（把改动退回旧行为，确认测试真的变红）。

> ⚠️ **但这些全部是离线验证。** 控制器修好之后**一次完整的真机 run 都没跑过** ——
> 三个 benchmark run 的 stage-2 产物已于当天清空重跑，那次 resume 将是它的
> **第一次上机**。在拿到一次完整闭环之前：
>
> * 不得声称「Stage-2 自治闭环可用」；
> * 不得删除任何旧修复路径（`S2-E` 明确等这次闭环）；
> * 控制器产出的任何 ΔG 都不是可引用结果。

> 🔴 **2026-09-14 追加：上表「已修」是逐条回源码核实过的，但那份审计仍有 18 条 `OPEN`** ——
> 其中 `#11`（`HALT_FK_REFUTED` 不在 `TERMINAL_EXITS` 里 ⟹ 统计驳回以未捕获 traceback
> 结束整跑）、`#12`（`_legalize_tail_window` 用插点**之前**的 `view` 解 anchor ⟹ `ValueError`
> 炸穿）、`#13`（5a-2 的 `INSERT_LAMBDA` 完全不查可行性）三条**会炸整跑**，
> `#2`（rewindow 目录被当采样段合并 ⟹ 静默错 ΔG）、`#4`（λ 表 fail-open 回退未量化内存值）
> 两条**会静默产出错误结果**。**在这五条关掉之前，别把真机闭环的失败当成物理问题去查。**
> 逐条见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md)。

## 仍开放（不阻塞，但必须随数字一起说）

- ~~**溶剂腿 stage2** 与真值差 −4.318 kJ/mol（5.5σ）~~ **→ 2026-09-10 已关闭。**
  那 5.5σ **100% 是参照臂的盒错了**：参照一直跑在**建系盒** 43.950 nm³，
  而它比 1 bar 平衡密度大 **3.15%**。配对重跑（同 λ 表、同 seed，只换盒）
  −6.594 → **−11.490 ± 0.342**，残差变 **+0.592（0.72σ）**。
  ⚠️ 09-09 写的「密度 20% + 单混合重加权 80%」**两半都作废**——
  单混合效应同盒重测后为零。
  ⚠️ 比较前仍**必须先对齐 LRC**，否则得 −1.49（1.9σ）的假象。
  证据与复现见 [STAGE2_SOLVENT_LEG_ERROR_BUDGET.md](STAGE2_SOLVENT_LEG_ERROR_BUDGET.md)。
- 独立重复、随机种子账本、时间相关不确定度**仍未闭合**。

## 协议身份

直接读自源码常量。**这张表由 `tests/test_doc_staleness_contract.py` 对着源码钉住**，
改了常量而没改这里会红。

| 协议 | 常量 | 值 |
|---|---|---|
| IBS 偏置 | `ibs_engine.IBS_BIAS_PROTOCOL_VERSION` | 33 |
| 热力学路径 | `abfe_preoptimizer.THERMODYNAMIC_PATH_PROTOCOL_VERSION` | 22 |
| LJ 长程修正 | `ibs_engine.TRADITIONAL_LJ_LRC_PROTOCOL_VERSION` | 3 |
| WCA 记账 | `ibs_engine.WCA_ACCOUNTING_VERSION` | 3 |
| ESS 门 | `ibs_engine.ESS_GATE_PROTOCOL_VERSION` | 5 |
| 配体 COM 约束 | `ibs_engine.LIGAND_COM_RESTRAINT_PROTOCOL_VERSION` | 2 |

`ibs_engine.WCA_SHIELD_RETIRED = True`（λ-WCA 防护壳已退役——这就是上面 4W53
那个数字从 +12.75 变成 −21.36 的原因）。

> **2026-09-09：`IBS_BIAS_PROTOCOL_VERSION` 32 → 33**（EXP-031 路线 A+B 并入主线，
> 详见 [EXP-031_GPU_OPTIMIZATION_2026-09-09.md](EXP-031_GPU_OPTIMIZATION_2026-09-09.md)；
> 融合内核本轮不接）。兼容集合已收窄成 `frozenset((33,))`。
>
> ⚠️ `system_xml_sha256` 随之改变 ⟹ **既有 `dual_window_*` / `ibs_state_*` /
> `convergence.json` 全部失配，必须重跑，不要试图复用。**
> 这不作废上表里已经登记的数字——那些是已完成运行的记录；失配影响的是 resume/缓存复用。

## 状态词

- `IMPLEMENTED`：代码存在；
- `VALIDATED`：在明确输入、环境和验收门下通过；
- `CANDIDATE`：可继续验证但不可宣称最终；
- `FAILED` / `INVALIDATED`：保留证据但不得作为当前科学结论；
- `PLAN` / `DESIGN`：计划不是执行结果，也不是生产授权。

DEXP、MACE、ORB、outer-lambda 神经基势、膜体系、charge-transfer、RBFE 都有代码或
计划，但都不在生产主线上——按上面的状态词判断，别按"仓库里有"判断。

## 证据保全

以下目录不得原地改写、移动或删除（原文在 `Atenolol-rank11`）：

```text
output/  output_lrc_fix/  output_lrc_fixonly-complex-charging/
validation/  solvent_box_scan/
memtest/output_membrane_100ns/  memtest/output_membrane_5ns/
```

新算法/新协议写进新的输出目录。旧结果、失败路线和无效结论继续保留，用于复核协议
演化和 bug 根因。

## 新数字进这张表的条件

来源 artifact、单位、符号约定、协议身份、有效性、是否可引用——六项齐了才登记。
不通过改写旧记录来"修正历史"，用替代关系保留。
