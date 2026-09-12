# EXP-033 Local-Residual：把「换配体重训」从实验链变成流水线一步（2026-09-10）

**状态：P1 ✅ 已落地（2026-09-12），P2 ⬜ 未开工，P3 ❌ 已划掉。**
P3 用户 2026-09-12 拍板不做 —— 它的前提（逐配体重训很贵）被 P1 消掉了；
理由与技术自洽性见落地文档 §9。
P1 的实现、门的定法与未验证项见
[EXP-033_P1_LANDED_2026-09-12.md](EXP-033_P1_LANDED_2026-09-12.md)。
本文以下内容是**原始登记**（方案与已核实的证据），保留不改 —— 其中 §5 P1 item 1
写的帧源「那次 run 自己的每窗口产物」在落地时改成了 `pre_equilibration.dcd`，
理由（`B_φ` 在 outer、不进哈密顿量）记在落地文档 §2。
参与者：`abfe-ibs-4e`（离线核查与实测）、`abfe-ibs-66`（独立复核与口径纠正）。

一句话：`outer_lambda_local_residual_ibs` 的 R1 权重现在**绑配体**，换配体要走
`docs/RETRAIN_LOCAL_RESIDUAL.md` 那条四步链。EXP-033 的目标**不是**让权重通用，
而是把「重训」压成流水线里的一次闭式求解，让「换配体要重训」不再是个成本问题。

---

## 0. 结论词

| 问题 | 结论 |
|---|---|
| 运行时是否绑配体 | ❌ 不绑。参数全按元素类型索引，绑的是**闸门 + 容量常量** |
| R1 训练链是否需要 MACE | ❌ 不需要。出厂权重是 `direct_gap`，信号纯 MM |
| 「拟一份跨配体通用权重」是否可行 | 🔴 **不推荐**：参数在同一配体的三条独立轨迹上都还没辨识出来 |
| 「把重训做成闭式求解」是否可行 | 🟡 **待上机判决**：离线信号支持，但离线口径不是验收口径（见 §4） |
| 想要真正的通用性 | 只能换目标（teacher 局部残差），代价是把 per-system MACE 标注重新引回来 |

---

## 1. 到底是什么在绑配体

**不是架构。** R1 的可训练参数只有两组，全部按**元素类型**索引：
`pair_weight[T,T,P]`（`local_residual/softlift.py:337`）和每个类型一个 ρ-MLP
（`:338`）。没有任何按配体原子序号索引的参数。运行时那份序列化 Force 里
`ligandTopologyIds` 是数组、`numTypes` 是运行时数据
（`LocalManyBodyResidualForceProxy.cpp:107`），不是编译期常量。

绑配体的是这三样：

1. **身份闸门**：`local_residual/openmm_plugin.py:499` 用
   `ligand_chemical_identity()`（局部原子序数序列 + 内部键图的 canonical-JSON
   SHA-256）逐项比对 manifest 里声明的配体，不一致就 fail-closed。
   （2026-09-10 之前这里是硬写字符串 `"Atenolol"`，已由 `abfe-ibs-66` 换成指纹。）
2. **容量常量**：`softlift.py:519-521` 的 `max_environment_atoms=320` /
   `max_edges=2048` / `max_neighbors_per_ligand=80` 是按 41 原子配体定的。
   **大配体首先是「重新定容」问题，不是重训问题。**
3. **非外延的输出头**：`softlift.py:363` 是 `raw = per_ligand.sum(dim=-1)`，
   然后 `b_max_reduced * tanh(raw / b_max_reduced)`，`b_max_reduced=10.0`
   （`:518`）。全局 tanh 套在**对原子求和**之外 ⟹ B 不是外延量，同一份权重
   换配体尺寸就换工作点。

---

## 2. 已核实的事实

### 2.1 R1 训练链上没有 MACE（已写回手册）

`local_residual/loss.py:106` 的 `bidirectional_gap_variance_loss` 只吃
`adjacent_gap_reduced`，而它来自 MM ledger：
`exp012_xed/mm_ledger.py:118` 就是 `np.diff(target_u, axis=1)`，`target_u` =
β·(base + softcore + lrc)。`local_residual/softlift_dataset.py:467-471` 只读
ledger 的 `adjacent_gap_reduced` / `log_importance_unnormalized` + 对应轨迹；
`abfe_scripts/run_exp019_softlift_d0.py` 的参数里没有任何 teacher/model 入口。
`DiffLift.MD` §15 也明写 `direct_gap` 变体「完全不看 MACE teacher」。

⟹ 手册里「贵的不是训练，是教师标签」那句是错的，`docs/RETRAIN_LOCAL_RESIDUAL.md`
已加更正块并重排链条。EXP-010 那套 `exp010-*` 教师子命令是**另一条谱系**。

### 2.2 训练数据是生产的副产物

`ibs_engine.py:13747` 起，每个窗口都落盘：

| 文件 | 形状 | 内容 |
|---|---|---|
| `dual_window_{i}_{stage}_energies.npy` | states × frames | 逐 λ 态目标能量 |
| `dual_window_{i}_{stage}_sampling_states.npy` | frames × states | 采样态能量 |
| `dual_window_{i}_{stage}_residual_basis.npy` | frames | 该帧的 B |
| `dual_window_{i}_{stage}_bias.npy` / `_base.npy` | frames | IBS 偏置 / 基础能量 |

加上同窗口的 DCD，这就是 `softlift_dataset_v1.npz` 的全部内容来源。
**⚠️ P1 的唯一口径风险**：`energies.npy` 是否已含 LRC、单位是 kJ/mol 还是
reduced，**必须现场对账**，不能照抄 `mm_ledger.compose_mm_ledger_arrays` 的组装式
（同类不对齐已经在溶剂腿上骗过一次 5.5σ，见 `STAGE2_SOLVENT_LEG_ERROR_BUDGET.md`）。

### 2.3 离线实测（abfe-ibs-4e，2026-09-10）

数据：`Atenolol-rank11/output/outer_lambda_exp020_softlift/dataset/softlift_dataset_v1.npz`
（1500 帧 / 5 态 / 4 邻边 / 1 920 801 条边，均值 1281 边每帧）。方法：用 numpy
重写 R1 前向与 `bidirectional_gap_variance_loss`，LORO 分折（按 run），内层 CV 选 λ。

**A. 词表与覆盖**

* `pair_weight` 的 **Na(11) / S(16) / Cl(17) 三个配体类型行逐比特为零**——那三种
  元素在配体里不存在，永远拿不到梯度。环境侧 Na、Cl 列同样逐比特为零（1500 帧里
  离子一次都没进过 5 Å 壳），环境 S 只有 162/1 920 801 条边。
* 实际训过的只有 H/C/N/O × H/C/N/O。
* `atom_type_index_for_topology`（`openmm_plugin.py:268`）要求**整个拓扑每个原子**
  落在词表内 ⟹ 膜体系的 P、K⁺/Mg²⁺/Ca²⁺、含 F/Br 的配体在建系阶段就抛异常。
* 词表来源是 `softlift_dataset.py:558` 的 `unique(all_topology_atomic_numbers)`
  ——从体系组成推，所以混进了一个 MACE-OFF24 **根本表示不了**的 Na。实测
  `~/.cache/mace/MACE-OFF24_medium.model` 的 z-table = {1,6,7,8,9,15,16,17,35,53}
  （H C N O F P S Cl Br I，`r_max=6 Å`），反过来 MACE 覆盖的 F/P/Br/I 又不在词表里。
  **两个词表从来没对齐过。**

**B. 工作点（推翻了先在纸上估的「已经饱和」）**

真实 q ∈ [−3.65, +1.97]，`S = Σ b_i` ∈ [−13.66, +17.52]，`dB/dS` 中位数 **0.88**
（最小 0.11）⟹ 41 原子时**没有饱和**。但 |S| ≈ 0.43/原子，`b_max=10` 的全局 tanh
在 |S| ≳ 10 开始压平 ⟹ **同一份权重迁到约 80 原子的配体基本进死区**。

**C. 基函数分层（LORO held-out gap-variance 改善，⚠️ 这不是验收指标，见 §4）**

| 基函数 | 参数 | held-out（3 折） | 三折系数相关 |
|---|---|---|---|
| 出厂非线性 R1（SGD 500 epoch × 3 seed），α 逐折重拟 | 3031 | +49.7 / +27.2 / +62.9 | — |
| **闭式岭回归**，同一个 typed 径向基 | 256 | +45.6 / +58.9 / +48.4 | +0.65 |
| 闭式，只按配体原子类型 | 64 | +42.5 / +17.3 / +26.4 | +0.72 |
| 闭式，只按环境原子类型 | 64 | +4.7 / +16.3 / +19.7 | +0.93 |
| 闭式，untyped | 16 | +14.3 / +16.6 / +22.9 | +1.00 |
| 零训练解析特征 `Σ c₂(r)·r⁻⁶` + 一个标量 | 1 | +10.8 / +10.5 / +13.1 | — |
| 零训练平滑接触计数 + 一个标量 | 1 | ≈ 0 | — |

三条读法：

1. **信号需要完整 type-pair 分辨率。** untyped 只有 ~17%，单边分型不到 27%。
   ⟹「换个解析式（LJ/DEXP 形状）省掉模型」这条路实测走不通。
2. **闭式岭回归与出厂非线性在 n=3 上分不出高下**（逐折配对 1 胜 2 负，差值幅度
   与折间离散同量级）。**不能说成「非线性没赚到东西」**，只能说测不出差别。
   但它意味着：重训**可能**便宜到只要一次线性求解。
3. **参数不可辨识。** 平滑先验下三折径向函数相关只有 0.65；换成纯岭回归会掉到
   **−0.33**、系数幅度到 1e11 kT 相互抵消（预测仍然泛化）。同一配体、同一条 λ 梯、
   三条独立轨迹都还没把系数钉住 ⟹ **反对参数迁移，不反对逐体系重训。**

---

## 3. 已撤回的说法（别重新论证）

| 说法 | 谁说的 | 为什么错 |
|---|---|---|
| 「贵的不是训练，是 MACE 教师标签」 | 手册（09-10 早） | R1 链上没有 MACE，见 §2.1。已更正 |
| 「held-out gap-variance % 说明模型行不行」 | abfe-ibs-4e | 那不是验收指标，见 §4 |
| 「非线性 ρ+tanh 没赚到东西」 | abfe-ibs-4e | n=3 得不出，正确说法是测不出差别 |
| 「要通用性必须把目标换成 teacher 局部残差」 | abfe-ibs-4e | 与上一条自相矛盾：闭式解若成立，重训便宜到问题自解，不需要换目标；换目标反而把 per-system MACE 标注引回来 |
| 「`bMaxReduced` 是可以按配体自由抬的 payload 字段」 | abfe-ibs-4e | 运行时确实不校验，但它是冻结 config 的一部分（`primary_r1_config` 硬写 10.0、payload 记 `protocol_sha256`），改它 = 换一份 config + 重新拟合；且 `max_abs_basis = kT·b_max·1.2`（`openmm_plugin.py:553`）会跟着放大，安全界一起松。大配体的正解是逐原子有界 |
| 「B 饱和了」 | abfe-ibs-4e（纸上估） | 实测 `dB/dS` 中位 0.88，41 原子时不饱和，见 §2.3B |

---

## 4. 验收口径（这是本实验最容易做错的一步）

**离线 gap-variance % 只能当「值得上机的信号」，不是验收。** 已经踩过的坑：

* **EXP-027 U4 被封为 `INVALID_FOR_PROMOTION_BASELINE_FK_REUSED_FOR_CANDIDATE`**
  ——候选臂复用了 baseline 冻结的 f_k，导致高残差系数窗口混合差、方差大，
  看起来像模型不行，实际是标定缺失（不是系统性偏差，跨 repeat 符号不一致）。
* **真验收是 EXP-027 U3 那条**：window-0 utility + ΔG 一致性，且**两臂必须各自
  独立标定并冻结自己的 f_k**。状态在 `PLAN_EXP-027_online_utility.md` §16，
  后续设计是 EXP-029 全量 A/B。

⟹ EXP-033 的任何 A/B 都按 U3 口径跑，双臂各自标定 f_k，不复用。

---

## 5. 方案

### P1（离线，无 GPU，不动 kernel、不动冻结产物）

1. 写一步闭式拟合：吃那次 run 自己的 `energies.npy` / `sampling_states.npy` /
   `residual_basis.npy` + DCD → 直接出 payload，跳过 torch / 500 epoch / 3 seed。
   目标函数与现有 loss 同一个（对 B 二次 ⟹ 对线性参数二次），
   `θ* = −(M + λΩ)⁻¹ b`，正则用 **r 方向的平滑先验**而不是纯岭（§2.3C 第 3 条）。
   λ 用内层 CV 选，不许看测试折。
2. 口径对账：`energies.npy` 的 LRC 与单位（§2.2 的 ⚠️）。
3. 容量常量按配体尺寸推导，替掉 `softlift.py:519-521` 的硬写值。
4. 词表改成从 **teacher z-table ∩ 体系**推，替掉 `softlift_dataset.py:558` 的
   `unique(all_topology_atomic_numbers)`。加元素只是 payload 变更，kernel 不动。
5. 线性 B 无界：实测量程到 ±28~40 kT，超过 `max_abs_basis`（12 kT）。
   处置二选一——减掉常数（逐态常数被 f_k 吸收，对 ΔG 无影响），或者保留 tanh 当安全界。

### P2（上机，U3 口径）

ridge 臂 vs 出厂非线性臂，**两臂各自标定并冻结自己的 f_k**，指标 = window-0
utility + ΔG 一致性。赢了就把「重训」永久变成流水线里一次线性求解。

### P3（只有真要「一份权重跑一整个配体系列」时才做）

逐原子有界（替掉全局 tanh，恢复外延性）+ 多配体联合拟合 + 平滑先验。
这一条要改 CUDA kernel ⟹ 拖 `KNOWN_PLUGIN_SOURCE_SHA256` 两处同步 + G0–G4 重验
+ 成本门重跑，**必须跟下一次 kernel 重验一起走，不单独为它跑一遍**。

判决实验（能不能迁移，目前唯一能回答的实验）：两个配体，A 上拟、B 上测。
在 P1 落地之后，这个实验的成本 = 一次普通 run 的产物 + 一次线性求解。

---

## 6. 明确不做

* 不拿离线 gap-variance % 当验收（§4）。
* 不为了通用性把目标换成 teacher 局部残差（§3 第 4 行）——除非用户明确要一份
  权重服务整个系列，且接受 per-system MACE 标注。
* 不动 `b_max_reduced` 当旋钮（§3 最后一行）。
* 不指望解析特征替代模型（§2.3C 第 1 条，实测 10~13%）。
* 种类覆盖不在本实验范围：用户已决定后续升级 teacher 模型解决。**但注意**
  teacher 的固定原子选择（`atom_selection == "fixed_indices"` +
  `qualify_wp4_basis` 硬要求 `ions_excluded` / `exchange_waters_excluded`）
  是架构约束，换多强的模型都还在。

---

## 7. 复现须知

* §2.3 的探针是会话内临时脚本（scratchpad，不入库）；方法在 §2.3 与 §5 P1 里写全了，
  P1 落地后由那一步取代。
* **EXP-020 那份 canonical fixture 的轨迹已经被清**
  （`hard_window0_run1/scratch_sample/*.dcd`），`active_edges=1206`
  （`abfe_scripts/export_exp025_g1_reference_payload.py:81`）复现不了。换同拓扑轨迹并传
  `--expected-active-edges none`，导出的权重仍逐字节等于出厂那份 `c4492f9d…`
  （abfe-ibs-66 于 2026-09-10 实测）。

## 8. 交叉引用

* `docs/RETRAIN_LOCAL_RESIDUAL.md` —— 现行四步重训链（P1 会取代它的 ①②）
* `PLAN_EXP-027_online_utility.md` §16 —— 验收口径与 U3/U4 状态（在 rank11）
* `DiffLift.MD` §1 / §14 / §15 / §E.2 / §E.6 —— R1 的设计契约与能力边界（在 rank11）
* `docs/STAGE2_SOLVENT_LEG_ERROR_BUDGET.md` —— LRC 口径不对齐的前科
