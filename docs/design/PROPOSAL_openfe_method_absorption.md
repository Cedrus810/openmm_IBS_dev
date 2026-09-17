# 提案：从 OpenFE 吸收方法以降低 benchmark MAE

状态：`PROPOSAL` —— 未执行、未授权。计划不是执行结果。
日期：2026-09-17
证据来源：`/home/ruigengji/openfe`（读源码，非文档）+ `/home/ruigengji/abfe-benchmark/openmm_IBS/runs`（读已落盘产物）

> 本文只回答一个问题：**为了降低 MAE，该从 OpenFE 拿什么。**
> 不讨论 `docs/TODO.md` 里的 P1 阻塞（`BM-B` / `AUDIT-S2-03` / `COMBINE-01` / `REWIND-01` / `LR-06`）——
> 那些是"跑不跑得完"，本文是"跑完了准不准"。两条线互不替代。

---

## 1. 误差现状

`results.csv` 11 行 / 6 体系，ME −4.73、MAE **5.91**、RMSE 7.35 kcal/mol。
⚠️ 11/11 出自 2026-09-16T05:03 之前的代码。

| 体系 | dev = 我们 − 实验（kcal/mol） |
|---|---|
| **brd4_ligand1** rep1 | **−13.67** |
| **brd4_ligand2** rep1/2/3 | **−9.56 / −11.61 / −9.85** |
| cyclod_ligand1 rep1/2 | −5.44 / −3.44 |
| cyclod_ligand2 rep1/2/3 | +3.43 / +0.42 / +2.57 |
| jnk1_ligand1 rep1 | +0.05 |
| p38_ligand2 rep2 | −4.95 |

**去掉 brd4 那 4 行，MAE 5.91 → 2.90。** brd4 是深疏水口袋 + 保守结构水；
这批实验值的来源（Aldeghi 2016）在同一批配体上做到过 ~1 kcal/mol ⟹ **不是体系难，是我们的问题**。

## 2. 误差的形状：复合物腿 vanishing 没收敛

`final_results.json` → `stage_diagnostics.stage2.split_half_diagnostics`：

| run | drift (kJ/mol) | drift / 2σ |
|---|---|---|
| brd4_ligand1 rep2（复合物） | **−13.81** | **4.17** |
| brd4_ligand2 rep3（复合物） | **−12.53** | 3.33 |
| brd4_ligand2 rep1（复合物） | +8.01（95.57 → 103.59） | 2.30 |
| cyclod_ligand2 rep2（复合物） | +11.14 | 2.12 |
| 所有溶剂腿 | \|drift\| ≤ 5.0 | ≤ 2.1 |

* 复合物腿 vanishing 前后半漂 8–14 kJ/mol，**符号不定**（不是单调 burn-in）。
* σ 膨胀因子实测 2.435（MBAR 报 1.745 → split-half 撑到 4.249）。
* 跨 rep：brd4_ligand1 复合物腿 153.96 / 142.77 / 151.60，差 11 kJ；报的误差棒 0.5 kcal。

⟹ **这是采样不足的签名，不是哈密顿量错。**

### 2.1 方差 ≠ 偏差（2026-09-17 补，由 `abfe-ibs-08` 指出）

上面的 drift 是**离散度**。但要解释的是 **ME = −7.85、8/9 同号偏负** —— 离散度再大也不产生同号偏移。
⟹ 欠采样要能解释它，必须是**有方向的**欠采样。

**初值就是那个方向**：所有窗口从**干起点**起跑（`prepare_wet_cavity_seed` 2026-09-01
随干/湿诊断一起退役，生产路径不再调用）。干空腔只会**高估**拔出配体的代价 ⟹ 同号；
空腔越大要灌的水越多 ⟹ 随重原子增长；每个窗口都这样 ⟹ 沿 λ 均匀；
体相水本来就在 ⟹ 复合物腿远重于溶剂腿。
本仓已有一次同物理实测：**λ-WCA 防护壳（把水挡在空腔外）退役 = +45 kJ/mol**。

**⚠️ 一条要撤回的推理**：本文 ⑥ 原写「drift 符号不定 ⟹ 不是单调 burn-in ⟹ 丢前缀未必有用」。
**错。** 体系若根本没开始弛豫，前后两半是**同等偏的**，其差只是噪声、符号当然随机，而绝对值整体偏。
**drift 符号不定不能排除有方向的初值偏差，恰恰是「弛豫还没启动」的预期表现。**

⟹ 合并后的形状：「从有偏初值出发 + 弛豫时间不足」= **偏差**；
「从无偏初值出发 + 采样不足」= **方差**。两条路线是同一条。

📌 **附带发现（2026-09-17 更正，原猜测已作废）**：`brd4_ligand1` 溶剂腿两个 rep = **56.60 vs 76.27 kJ**，
差 19.7 kJ = 4.7 kcal。

> ~~形状像「部分和冒充完整 ΔG」（`covered_lambda_indices`）。~~
> **作废。** `abfe-benchmark-08` 查盘核实：两边 coverage 完整（21 / 23 态全覆盖、`dropped_window_indices=[]`）、
> 门全过、都是 `ANALYSIS_COMPLETE`。差额全在 vanishing（−4.973 vs +16.289），decharging 只差 1.595 kJ。

真实差异是 **λ 路径 + 采样**：

| | λ 版本链 | 态数 | 迭代 | 整腿 split-half drift |
|---|---|---|---|---|
| rep1 | `[(1,'init')]` | 21 | 4 | 4.993 kJ = **1.94×2σ** |
| rep2 | `[init, insert_lambda, insert_lambda, tail_repartition]` | 23 | 11 | 0.574 kJ = 0.18×2σ |

rep1 的 w3 统计低效率 **16.96**（两边十个窗最高）。

⟹ **同一个配体在体相水里，仅因 λ 路径与采样差异就差 21 kJ/mol。**
这是 §2 欠采样结论的**正面证据**，不是一个需要单独修的 bug：
溶剂腿都能差这么多，复合物腿差 8 kcal 不需要一个结构性 bug 来解释。

### 2.2 平衡前缀丢弃扫描（2026-09-17 实测，零 GPU）

**做法**：离线重放 `dual_window_*_vdw_{energies,bias,base}.npy`，唯一变量是每窗丢掉最前 f 比例的帧
（`f_k` 是窗口级冻结量不随帧数变；`production_segments` 按丢弃量重映射）。
脚本留在会话 scratchpad 的 `prefix_scan.py`，未进仓库。

**管线自验**：f=0.5 重算得 **103.587**，与生产自记的
`total_delta_G_second_half_kJ_mol = 103.587` 逐位相同（brd4_ligand2 rep1）⟹ 离线重放 = 生产。

complex 腿 stage2 ΔG（kJ/mol），只列 `ANALYSIS_COMPLETE`（σ ≈ 1.6–2.6）：

| run | f=0 | 0.2 | 0.3 | 0.4 | 0.5 | 方向 |
|---|---|---|---|---|---|---|
| brd4_ligand2 rep1 | 97.36 | 103.04 | 102.89 | 102.13 | 103.59 | +5.7 升 |
| brd4_ligand1 rep1 | 125.52 | 127.61 | 130.39 | 122.50 | — | 噪声 |
| brd4_ligand1 rep2 | 122.13 | 122.15 | 122.66 | 120.07 | 122.22 | ≈0 |
| brd4_ligand2 rep2 | 99.65 | 95.80 | 94.63 | 97.47 | 96.07 | −4 降 |
| brd4_ligand2 rep3 | 95.10 | 92.92 | 88.01 | 85.88 | 86.48 | −9 降 |

**结论一：零结果，不是证伪。** 方向 +5.7 / ≈0 / ≈0 / −4 / −9 不一致 ⟹ 没有可复现的同号前缀瞬态。

> ⚠️ **零结果的不对称性（登记时必须带着这句）**：OpenFE 的平衡前缀检测跑在 10 ns/replica 上，
> 我们是 500 ps/窗。若空腔灌水是 ns 尺度，则整段都落在弛豫之前，丢前缀时
> **留下的和丢掉的同等偏** ⟹ ΔG 不动、split-half 两半同等偏 ⟹ 差值符号随机。
> 也就是说零结果同时符合「没有初值偏差」和「整段都在弛豫之前」。
> **不得把零结果当作初值偏差被证伪。** 只有正结果（单调下降并趋稳）是强结论。

**结论二：单 run 的正结果同样不可信。** brd4_ligand2 rep1 单看**就是**单调上升后平台
（97.4 → ~103 在 f=0.2 趋稳），只跑它会误读成"坐实"。多跑四个才看出符号不一致。

**结论三（意外发现）：帧不可交换。** brd4_ligand2 三 rep 的跨 rep 离散度随丢弃比例**变大**：

```
f=0.0   97.36 / 99.65 / 95.10    SD = 2.28   spread =  4.56
f=0.3  102.89 / 94.63 / 88.01    SD = 7.46   spread = 14.88
```

SD 涨 **3.27×**，而只丢 30% 帧、纯 √N 效应只应涨 **1.195×** ⟹ 多出的 2.7 倍不是数据变少。
若前 30% 是共享 burn-in，丢掉它应让独立 rep **更一致**；它让它们更不一致。

⟹ **留哪一段比留多少帧影响大得多。** 与已知的 top1% 权重偏斜、g 值是同一件事的三个面。含义：
* 报告 σ 低估比 §4 说的 2–3× 更严重 —— 丢掉一半帧能让 ΔG 动 **5σ**（⑦ 更紧迫）；
* 加采样收益不会是 √N（有效独立样本远少于帧数），但**方向仍然对**（②）；
* 任何 wet/dry 类 A/B 必须**多 rep**，单对起点的 ΔF 差会被这个淹掉。

**操作提醒**：`f ≥ 0.7` 普遍触发「去相关子采样后有效帧 < 10」⟹ `ANALYSIS_INCOMPLETE`，
brd4_ligand1 rep1 在 f=0.8 返回哨兵 `ΔG=0.000, σ=999.9`。别把这些点画进曲线。

**另登记一条（未查）**：`jnk1_ligand1/rep1` 两条腿都载不进来 ——
`窗口 0 lambda 内容与当前路径不匹配` / `窗口 3 状态数与 window_ranges 不符`，
盘上 λ 表与 preopt 不一致。与本节无关，归 `abfe-ibs-08` / `abfe-benchmark-08`。

## 3. OpenFE 的真实默认值

`src/openfe/protocols/openmm_afe/equil_binding_afe_method.py:129-207`：

| 项 | complex | solvent |
|---|---|---|
| λ 态数 / replica | **30** | 14 |
| 炼金生产 | **10 ns × 30 = 300 ns** | 10 ns × 14 = 140 ns |
| 炼金平衡 | 1 ns × 30 | 1 ns × 14 |
| 非炼金预平衡 | NVT 0.25 + NPT 0.5 + **生产 5 ns** | 0.1 + 0.2 + 0.5 ns |
| `protocol_repeats` | **3** | 3 |
| `solvent_padding` | 1.0 nm | 默认 |

λ 表**三段串联、互不重叠**：

```
complex[0:6]   restraints 0→0.2→0.4→0.6→0.8→1.0   （elec=0, vdw=0，配体完全耦合）
complex[6:16]  elec 0.1→1.0 步长 0.1               （restraints=1, vdw=0）
complex[16:30] vdw  0.1…0.6 步长 0.1，然后 0.65/0.7/0.75/0.8/0.85/0.9/0.95/1.0
               ← 14 个 vdW 态里 8 个落在 λ>0.6
solvent[0:5]   elec 0/0.25/0.5/0.75/1.0
solvent[5:14]  vdw  0.12/0.24/0.36/0.48/0.6/0.7/0.77/0.85/1.0
```

采样器（`openmm_utils/omm_settings.py:578-658`）：
`sampler_method="repex"` + `replica_mixing_scheme="swap-all"`，
`time_per_iteration=2.5 ps` ⟹ 10 ns 内 **4000 次交换**；
`timestep=4.0 fs` + `hydrogen_mass=3 amu`（HMR）；
`n_restart_attempts=20`；`real_time_analysis_interval=250 ps` + `early_termination_target_error`。

炼金细节（`equil_afe_settings.py:44-86`）：
`annihilate_sterics=False`（解耦而非湮灭）、静电湮灭、
softcore `alpha=0.5, a=1, b=1, c=6`（Pham–Shirts）、
`disable_alchemical_dispersion_correction=False`。

Boresch（`restraint_utils/geometry/boresch/host.py`）：
局部 RMSF < 0.1 nm → 可选 DSSP 二级结构 → 按到 guest anchor 的质心距离排序 →
`EvaluateHostAtoms1/2` 在**预平衡生产轨迹**上逐帧评角度/共线性。
`K_r=4184 kJ/mol/nm²`、`K_θ=334.72 kJ/mol/rad²`。依据 Baumann 2023 / Wu 2025 / Alibay 2022。

**⚠️ OpenFE 也是干起点**（`base_afe_units.py:991-995`）：

```python
sampler_state = SamplerState(positions=positions)
sampler_states = [sampler_state for _ in cmp_states]
```

`positions` 来自**完全耦合**体系的非炼金预平衡生产 ⟹ 30 个 λ 态（含完全解耦端）
**共用同一份干空腔坐标**。OpenFE 解法里**没有 wet seed 这一项**，
它靠三件"时间"类手段耗掉初值偏差：10 ns/replica、4000 次 HREX 交换、
以及分析时**自动检测并丢弃平衡前缀**。第三条正是为有方向初值偏差设计的标准解法，
**我们一条都没有**。

分析（`openmm_utils/multistate_analysis.py`）：
平衡段自动检测（`n_equilibration_iterations`）+ 统计低效率抽稀 →
MBAR **1000 次 bootstrap**（`bootstrap_solver_protocol="robust"`，`uncertainty_method="bootstrap"`）
+ overlap matrix + replica transition matrix + forward/reverse 10 分位
（前/后向若"去相关样本比例 = 1.0"则整段丢弃）。

## 4. 逐项对照

| | OpenFE | 我们 |
|---|---|---|
| **vdW 段总 MD 时间（complex）** | **140 ns**（14 态各 10 ns 独立轨迹） | **~2.5 ns**（5 窗 × 500 ps，16 态共享重加权） |
| timestep | **4 fs + HMR** | **2 fs，全仓无 HMR**（`ibs_engine.py` 7 处硬编码 `0.002`） |
| 副本交换 | 4000 次 / replica | 无 |
| 报告 σ | 去相关后 bootstrap ×1000 | MBAR 渐近 σ + split-half 事后膨胀 |
| Boresch 候选 | RMSF + DSSP + 距离排序 + 轨迹评角 | `boresch_source="simple"` |
| repeats | 默认 3 | benchmark 设计已是 13×3 |

**vdW 段差 56×**（140 ns vs 2.5 ns）。这和 §2 的 drift 互相印证。

---

## 5. 吸收清单（按 MAE 收益 / 成本排）

### ① HMR + 4 fs —— 免费的 2×

* **做什么**：H 质量 3 amu、重原子扣减、`constraints=HBonds`，timestep 2 → 4 fs。
* **机制**：同样 GPU 小时，采样翻倍。不改物理。
* **合理性**：✅ 无争议，OpenFE 默认如此。
* **风险**：**破全部缓存**（`system_xml_sha256` 变）；我们的 System 来自 GROMACS `.top`，
  不走 `createSystem(hydrogenMass=)`，要手工重分配质量并逐条核对虚位点/约束。
* **验收**：同一 run 4 fs vs 2 fs，能量守恒 + stage1 ΔG 在误差棒内相等。

### ② ~~复合物腿 vanishing 加采样~~ —— ❌ **2026-09-17 撤回，仓库已真机否过两次**

> **别再提这条。** 纯加帧已经在真机上被否掉，不是未验证的候选：
>
> * `cmet_ligand2` 2026-09-16 真机 A/B：失败窗口 λ 0.2616→0.1953 的
>   `n_eff=[369,909,1181]` 但 **`g=82.3`**（τ ≈ 4.1 万步的构象慢模态），
>   **加帧 4 块共 1M 步 → `3.32→4.30→4.48→3.80`，判 `marginal_gain_stalled`**。
>   同批四个杠杆（加帧 / 缩窗 / 插 λ / f_k 重标定）实测**全否**。
> * `cyclod_ligand2` 2026-09-11：纯加帧 250k→1M（4×），最差 top1%
>   **`0.545 → 0.762`，越加越差**；`ESS_ratio` 0.037→0.0264 同向变差。
>   原话：「加帧治不了偏斜——往同一个偏斜分布里多塞帧。」
> * §2.2 的「帧不可交换」是同一件事的第三个面：**有效独立样本 ≪ 帧数**。
>
> ⟹ §4 那个 56× 的差距是**真的**，但它**不能用步数去补**。
> 分母是 g（τ），不是分子 n_eff。对症的是打断慢模态 —— **λ 方向副本交换**。
> 仓库现成的是被叫停的 REMD（`docs/design/PLAN_openmm_8_6_remd_backend.md`），
> cmet 那份记录判定「这是它第一个有据的适应症」。
> **解不解冻是用户的决定，本文不替他推。**

<details><summary>原条目（保留以备复核，勿执行）</summary>

* **做什么**：`n_steps_per_window` 250k → 1M。
* **机制**：drift 8–14 kJ 就是没收敛的定义；补 56× 缺口里的 4–8×。
* **原写的合理性**：✅ 但**先证伪**：`brd4_ligand2` 单 rep 跑 4×，看 `total_drift_kJ_mol`
  与 `total_delta_G_complex` 往哪走。一晚上一张卡。
  * drift 掉 + cx 上涨 → 是采样问题，全量重跑
  * drift 不掉 → 慢模态锁死，跳 ③
  * cx 不动 → 采样不是主因，回头查哈密顿量 / Boresch

</details>

### ③ HREX，只上在 vanishing 段

* **做什么**：λ 态间交换，`swap-all`，2.5 ps 一次。
* **机制**：口袋脱水/再水合是慢模态；加时间线性收益，交换是混合收益。brd4 正是这个。
* **⚠️ 待拍板**：IBS 里"副本"是**窗口**不是 λ 态，交换单元是什么？
  * (a) 窗口间交换构型 —— 改动小，但窗内仍靠重加权
  * (b) vanishing 段退回逐态独立轨迹 + HREX —— 等于在这一段放弃 IBS
* **倾向 (b)**：IBS 的效率优势在中段有效，在 λ→1 那几个态上恰好帮倒忙
  （一条轨迹里配体始终占着体积）。**中段 IBS + 端点段 HREX** 有物理依据，不是妥协。
* **前置**：`docs/design/PLAN_openmm_8_6_remd_backend.md` 已落地但被叫停，先确认该线现状。

### ④ λ 表在 vdW 端点加密 —— 零 GPU，当体检

* **做什么**：把我们 metric-integral 布出来的 vdW 表，对着 OpenFE「λ>0.6 占一半态」量一次。
* **机制**：我们自己的 pilot 记录过 ⟨dU/dλ⟩ 与 Fisher 度规**反相关**
  ⟹ 纯 √g 布点在自由能落差最大处少放点；densify 的 2 个点可能不够。
* **合理性**：✅ 离线重放已有 pilot 就能算。**不是照抄 30/14**，是拿它当基准线体检。

### ⑤ Boresch 候选生成 —— 有实测证据

* **做什么**：移植 `boresch/host.py` 的筛选链（RMSF<0.1nm → DSSP → 距离排序 → 轨迹上评角度/共线性）；
  OpenFE 为此专门跑 5 ns 非炼金生产。
* **机制**：实测 `cyclod_ligand2` 三 rep 的 release 修正 = −36.41 / −35.73 / **−31.85** kJ
  —— 同体系同配体，anchor 抖了 4.6 kJ。松 anchor 让受约束系综更难采，漏进 cx。
* **合理性**：✅ 候选源我们已有（fluctuation/ORB/MACE），缺的是**筛选判据 + 够长的预平衡轨迹**。
  保留我们自己的角度奇异性 / PBC / 解析释放审计。

### ⑥ 平衡段检测 + 去相关抽稀 —— 零 GPU，可在已有数据上试

* **做什么**：MBAR 之前先 `detect_equilibration` 丢 burn-in 前缀，再按统计低效率抽稀。
* **合理性**：✅ **2026-09-17 升级（原为"收益存疑"）。** 见 §2.1：
  drift 符号不定**不能**排除有方向的初值偏差，原来的否定推理是错的。
  这是 OpenFE 针对同一问题的**标准解法**，且**零 GPU**。
* **与 wet seed 的关系**：测的是同一件事的两面。丢掉前 X% 后 cx 若单调下降并趋稳
  ⟹ 初值偏差的直接证据，**不用跑任何新轨迹**；若丢到只剩 20% 数据仍纹丝不动
  ⟹ 初值偏差被证伪，回到纯时间尺度问题。

### ⑦ bootstrap σ + 跨 rep SD —— 不改 MAE，改"能不能看见 MAE"

* **做什么**：去相关样本上 1000 次 bootstrap；报告 σ 用**配对 ΔG_bind** 的跨 rep SD。
* **机制**：实测 rep 间 SD ≈ 1.1–1.5 kcal，我们报 0.54 ⟹ **低估 2–3×**。
  不先修这个，①②③ 做完也**分不清改善是不是噪声**。
* **⚠️ 与既定判定的关系**：仓库对块 bootstrap 有明确否决
  （`ibs_engine.py:274 / 22472 / 24077`，理由是嵌套样本不满足独立性）。
  OpenFE 是在**去相关后的独立样本**上做，前提不同 —— 可以重新论证，但必须写清区别，不许当成推翻。
* **📌 比 OpenFE 做得更对的一点**：OpenFE 自己是把 complex-SD 与 solvent-SD 平方和开根
  （`afe_protocol_results.py`），且把标准态修正误差固定为 0。**配对求 SD 才对**，
  且我们有 model discrepancy，不该置零。

### ⑧ 完整诊断落盘

* **合理性**：⚠️ 降级。transition matrix / round trips 在没有 replica 的 IBS 里**不存在**；
  overlap 我们已有；forward/reverse 是 split-half 的多点版。**做完 ③ 再补**，那时才有 replica 可看。

---

## 6. 明确不吸收

| 项 | 理由 |
|---|---|
| electrostatics 湮灭配体内部库仑 | v5 刚往反方向走，有 1:1 实测（−88.66 vs −88.05 kJ/mol，见 STATUS.md）。OpenFE 默认在柔性多极性配体上暴露在同一失效模式 |
| 固定 30/14 λ 表 | 我们按腿各自 pilot；只借它的**端点密度**当体检基准（④） |
| `disable_alchemical_dispersion_correction` | 我们的 LRC 是 pair-specific、与软核对应，比它细 |
| 全流程 HREX | Boresch attachment 段没有慢模态问题，上了只是烧钱 |
| 标准态修正误差置 0 | 我们有 model discrepancy，置零是退步 |

---

## 7. 执行顺序（有依赖）

```
⑦ 跨 rep SD + bootstrap        ← 先立判据，否则后面实验读不出信号
   │
   ├── ④ λ 端点密度体检   ┐
   ├── ⑥ 平衡前缀检测/丢弃 ├ 零 GPU；见 §2.1，已升级为一等候选
   └── ~~§2 brd4_ligand1 溶剂腿异常~~ ✅ 已查明：λ 路径+采样，非 bug ┘
   ↓
① HMR / 4 fs（免费 2×）        ← 破缓存，与 ② 合并成一次重跑
   ↓
~~② brd4_ligand2 跑 4×~~        ❌ 撤回：加帧已真机否过两次（见 ②）
   ↓
⑤ Boresch 筛选链（独立，可并行）
③ 端点段 HREX（最大改动，最后；交换单元待拍板）
```

**最省的一步**：④ ⑥ 今天就能在已有 `u_kn` / pilot 上离线做完，不占 GPU，
而且可能直接告诉你 ② 值不值得跑。

---

## 8. 本提案没有回答的

* ~~MAE 里有多少是**力场**~~ —— **2026-09-17 已排除。** `/home/ruigengji/abfe-benchmark/README.md`：
  这批数据就是 OpenFE/OpenFF 社区的 **"ABFE Burn-in Set"**，配体参数是 **OpenFF v2.0 "Sage"**、
  蛋白 AMBER14SB + TIP3P，只以 GROMACS `.gro`/`.top` 分发（生成脚本 `burn_in_parameter_generation.ipynb`）。
  ⟹ **我们和 OpenFE 用的是同一套配体参数，力场维度不成立，误差全是方法学。**
  ⚠️ 顺带一条不一致待查：README 称 "All ligands are neutral"，而我们 `systems.csv` 记 thrombin_ligand1/2 为 `+1.000 e`。
* `results.csv` 11 行全部出自修复前的代码 ⟹ **本提案的所有数字都要在新码重跑后复核一次**。
  §2 的 drift 结论对采样长度不敏感，预计站得住；§1 的 dev 数值会变。
