# Stage-2 离线取证：09-18 benchmark 24 run 的误差与误差棒

**日期**：2026-09-18
**数据源**：`/home/ruigengji/abfe-benchmark/openmm_IBS/runs`（11 臂 / 24 个进
`results.csv` 的 run / 全目录 339 个 stage-2 窗口）
**方法**：全部离线，零 GPU，只读 run 产物与本仓库代码。没有改动、没有重跑、
没有碰 production。
**每一条都标了复算路径**，不依赖任何会话记忆。

---

## 0. 结论一句话

误差棒报的是「给定这条 λ 路径、给定这份冻结 `f_k`，重加权有多噪」。
真实跨 rep 散布里**没有一项**在这个定义域内。而「有的错得巨大」有一个
可定位的机制：**某些窗口的某些 λ 态一帧样本都没有**，MBAR 照样外推出一个
ΔG 并配一个 ~1 kJ/mol 的 σ。

这是**方差机制，不是偏差机制** —— 塌陷 rep 的 `cx_s2` 有时更高
（cyclod_ligand3 72.45 vs 42.15）有时更低（cmet_ligand1 131.41 vs 161.81）。
所以它不解释系统性偏负，那是另一条线。

---

## 1. 账是干净的（L0）

```
ΔG_bind = (sv_s1 − cx_s1) + (sv_s2 − cx_s2) + (−boresch) + (−attachment)
```

对 **24/24 个 run 逐位闭合到 0.00 kJ/mol**。没有漏项、没有重复计数、
没有符号错。

| 项 | 范围 (kJ/mol) | 跨体系 SD |
|---|---|---|
| `Δs2`（stage2 vdW 解耦） | [−155, −24] | **37.4** |
| `Δs1`（decharging） | — | 8.06 |
| −boresch | [31.85, 40.41] | 2.15 |
| −attachment | [−6.12, −3.12] | 0.61 |

**全部信号和全部误差都在 `Δs2` 一项里。** stage1 跨 rep 复现到 0.3–6 kJ；
**所有**跨 rep 散布都在 stage2。

复算：读每个 run 的 `final_binding_results.json` +
`checkpoints/stage{1_decharging,2_vanishing}.json` 与
`solvent_leg/checkpoints/` 同名文件的 `total_delta_G`。

---

## 2. rep 之间不是「同协议不同种子」（L3）

48 个 (run, leg) 的 `lambda_path_fingerprint.sha256` **无一重复**，
`n_states` 21–24、窗口数 5–7、`path_version` 1–5。

但端点 **48/48 全体一致**（`lambdas_var` 从 1.0 单调降到 0.0）。

⟹ 所有 rep 算的是**同一个热力学量**，只是走了不同路径。路径无关性要求它们
一致，所以 20–30 kJ 的分歧只能是没收敛，不是定义差异。

---

## 3. 最干净的一例：每道门都过，差 24.7 kJ/mol

`cmet_ligand1` 溶剂腿，rep1 vs rep2（配体 63 原子泡在水里，无蛋白、无空腔）：

| | rep1 | rep2 | 门 |
|---|---|---|---|
| stage2 ΔG | **6.50 ± 1.80** | **31.21 ± 1.27** | — |
| 控制器退出 | ANALYSIS_COMPLETE | ANALYSIS_COMPLETE | — |
| `target_support_gate` | passed, rawESS 54.7 | passed, rawESS 38.3 | ≥20 |
| top1% 权重 | 0.168 | 0.304 | ≤0.35 |
| `min_overlap` | 0.068 | 0.152 | ≥0.05 |
| 端点不确定度 | 0.888 | 0.677 | ≤1.0 |
| `stage_quality_failures` | **0** | **0** | — |
| `results_untrusted` | **False** | **False** | — |

逐窗口（段和与报出 total 逐位相等，无丢项）：

| 窗口 | λ 区间 | rep1 | rep2 | 差 | 报出 ± |
|---|---|---|---|---|---|
| 0 | 1.00→0.77 | 72.03 | 62.88 | **−9.15** | 0.81 / 0.34 |
| 1 | 0.77→0.47 | 40.86 | 42.79 | +1.93 | 0.36 / 0.45 |
| 2 | 0.47→0.32 | −6.26 | −2.73 | +3.53 | 0.89 / 0.67 |
| 3 | 0.34→0.28 | −14.36 | −6.04 | +8.32 | 0.79 / 0.44 |
| 4 | 0.29→0.24 | −20.86 | −21.59 | −0.73 | 0.73 / 0.68 |
| 5 | 0.26→0.00 | −64.91 | −44.10 | **+20.81** | 0.70 / 0.44 |

**窗口 0 在几乎全耦合处就差 9.15 kJ，报的是 ±0.81/±0.34 —— 约 10σ。**
λ 跨度差 3% 最多解释 ~2 kJ。

---

## 4. w0 承载了整个答案，而它只有 4 个态

| run | w0 ΔG | 占 `cx_s2` | dG/dλ |
|---|---|---|---|
| cmet_ligand1/rep1 | 108.52 | **67%** | 575 |
| cmet_ligand1/rep2 | 113.14 | 86% | 491 |
| cmet_ligand2/rep1 | 96.24 | 67% | 433 |
| brd4_ligand1/rep1 | 86.97 | 72% | 391 |
| cyclod_ligand1/rep1 | 48.74 | 91% | 224 |

25/25 个 run 的 w0 都是 4 个态（`stage2_first_window_max_states = 4`），
**相邻态 `f_k` 间隔中位 12.4 kT、最大 19.6 kT**（cmet_ligand1/rep2:
48.8 / 40.1 / 31.5 kJ/mol）。

这在 IBS 里本身不是错——偏置存在的目的就是让这种间隔照样被访问。但它把
整个答案押在偏置对不对上，而偏置的冻结门是：

```python
# ibs_engine.py
IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL = 10.0   # 注释自称「≈ 4 kT」
# 「故意…不等 f_k 稳定」
# 「warmup 的 loose gate 只用 200 帧判、只要求 <10 kJ/mol，
#   留下的偏置可能让混合塌向少数态」
```

**容差是绝对值，被验的边长是它的 3–5 倍。** 4 kT 容差落在 15 kT 的边上。

---

## 5. 混合塌陷：这条解释「有的错得巨大」

**口径**（与 `_softmax_occupancy_per_state` / `mixture_occupancy_normalized`
同式）：`p_k ∝ exp(β(f_k − u_k))`，逐帧 softmax、对帧取均值、归一化后
×`n_states`（理想 = 1.0）。

**数据源**：`vanishing/dual_window_{w}_vdw_energies.npy`（`u_kj_raw`，
**单位是 kJ/mol**——当 reduced 读会把一切塌到单态，明显错）+
`checkpoints/ibs_state_vdw_window_{w}.json` 的 `f_k`。零 GPU 可复算。

### 5.1 全量分布（339 个窗口）

占据不均 = max/min：**中位 2.8×、75 分位 6.6×、最大 391159×**。

| 窗口位置 | n | 中位 | 最大 |
|---|---|---|---|
| w0（耦合端） | 58 | 4.2× | 255× |
| 中间 | 227 | 2.4× | 7980× |
| **末窗（解耦端）** | 54 | **5.0×** | **391159×** |

灾难全在 complex 腿末窗。溶剂腿基本在 1–10×。

### 5.2 八个全塌陷窗口

**≥99% 的帧压在同一个 λ 态**：

| run | 腿 | 窗口 | 态数 | 集中度 | 最低占据 | 进 results.csv |
|---|---|---|---|---|---|---|
| cyclod_ligand3/rep2 | cx | w5 | 7 | **1.000** | 1.7e-05 | ✓ |
| cmet_ligand1/rep2 | cx | w6 | 4 | **1.000** | 1.8e-03 | ✓ |
| jnk1_ligand2/rep3 | cx | w5 | 5 | 0.998 | 1.8e-04 | |
| p38_ligand2/rep2 | cx | w5 | 5 | 0.998 | 2.3e-03 | |
| brd4_ligand2/rep2 | cx | w5 | 4 | 0.996 | 3.5e-02 | ✓ |
| brd4_ligand2/rep3 | cx | w5 | 4 | 0.995 | 1.7e-02 | ✓ |
| brd4_ligand1/rep2 | cx | w4 | 5 | 0.994 | 1.3e-02 | ✓ |
| cmet_ligand1/rep2 | sv | w3 | 4 | 0.994 | 2.0e-02 | ✓ |

7/8 在 complex 腿。**8/8 都报 `bias_converged=True` +
`f_k_evidence_status='verified'`。**

极端例 `cyclod_ligand3/rep2` cx w5：7 个态 500 帧，**500 帧全部落在态 0**，
另外 6 个态取得最大权重的帧数是 0，末态占据 = 自己份额的 1/58000。
相邻 `Δf_k/kT` = −6.4 / −5.2 / −5.7 / −4.7 / −5.0 / −1.5，
`learning_updates = 22`。

跨 rep 散布最大的两个臂（cmet_ligand1 13.37、cyclod_ligand3 10.10 kcal）
**正是各有一个 rep 出现全塌陷窗口的那两个**。

### 5.3 量化关系

`f_k` 每条边错 Δ ⟹ 占据比倾斜 `e^(Δ/kT)`，且沿边**连乘**。

- 实测：w0 的 11.2× = ln 2.4 kT 摊在 3 条边 ≈ **0.8 kT/边**；
  w5 的 41.5× ≈ **1.2 kT/边**。
- 门允许 **4 kT/边**。4 态 3 条边 ⟹ 允许上限 `e¹² ≈ 1.6e5 倍`。

**已经肉眼可见塌陷的实测值，只用掉了门额度的万分之一。**

---

## 6. 塌陷拦不住：五个独立结构缺口同向失效

| # | 缺口 | 证据 |
|---|---|---|
| 1 | 占据显式不是门 | `min_occupancy_is_gate: False`、`min_occupancy_normalized_threshold: None` |
| 2 | 冻结门自指 | loose gate 比较 `\|Δf_{k,k+1} − ΔF^MBAR_{k,k+1}\|`，而 `ΔF^MBAR` 是**从同一份塌陷样本**估的 ⟹ 拿它测不到的量做验证 |
| 3 | 接缝无冗余 | 见 §7 |
| 4 | σ 放大关着 | 见 §8 |
| 5 | 漂移诊断自己关 | 见 §9 |

---

## 7. 接缝在自由能层面冗余度为零（构造性，非采样问题）

`solve_stage_integrated` 里有**两套估计器**同时算：

- **A 拼接曲线**：在共享 λ 求逆方差加权 `offset` 把各窗口 f 曲线粘起来，
  `offset_var` 逐点累计 → `offset_error_contribution`。
- **B 协方差链**（**报出的就是这个**）：`Σ_w [f_w(end) − f_w(join)]`，
  方差 = `Σ_w dDelta_f[join,end]²`。

函数自己断言相邻窗口必须共享**恰好 1 个** λ 态
（`>1 个=区间重叠会被重复积分`）。于是：

```
overlap_lams 只有 1 个元素
  ⟹ offset = f_glob(join) − f_loc(join)   精确
  ⟹ 位移后两边在该点逐位相同
  ⟹ 逆方差平均是 no-op
  ⟹ A ≡ B
```

实测：A 的端点差 − B 的 `total_delta_G`，**42/42 全为 0.00**。

**推论**

1. `offset_error_contribution` 是**幽灵方差** —— 给一个自由度已被消掉的
   参数记误差。它**不是** B 的漏项。
2. **接缝上永远不可能做自由能层面的一致性检验**，与采样好坏无关。
3. 所以共享态的**构象分布**交叉验证是唯一可能的接缝检验。
4. 想要纯能量的便宜版：让相邻窗共享**两个**态，第二个只当检验不进积分。
   现在那条「>1 个 = 重复积分」的规则把检验和重复积分一起扔了，
   而重复积分是可以单独避免的。

---

## 8. split-half σ 放大：算了、存了、0/42 生效

产物 `sigma_inflation_from_split_half` 给出逐窗口

```
sigma_eff = max(sigma_mbar, |split_half_drift| / 2)
理由：两个半程之差的 SD 是 2σ，故观测到 |漂移| 蕴含 σ ≳ |漂移|/2
```

但 `inflate_sigma_from_split_half: bool = False` 是默认值，
`sigma_inflation_applied` 在 **0/42** 上是 False。
`split_half_max_z_threshold` 也是 `None`（不是门）。

用上之后总误差：**中位 1.26×、最大 2.59×**。
例：cmet_ligand1 sv 1.80→3.12 和 1.27→2.07；cyclod_ligand1/rep3 cx
1.52→3.70（w0 ×3.8、w4 ×3.1）。

**方向对、免费，但不够** —— 相对实测 24.7 kJ 的分歧还差 8 倍。

---

## 9. split-half 本身在最需要它的 run 上自己关掉

- complex 腿 **12/20 缺失**；溶剂腿 **0/22 缺失**。
- 12/12 全走同一个出口：**「半程解出的窗口集合与全量不一致，无法逐窗比较」**
  （`split_half_drift_diagnostics`，`ibs_engine.py:23927`）。

**机制**：半程数据喂回同一个 solver，`solver_kwargs` 原样带着同一套阈值。
帧数砍半后边缘窗口过不了合格线被 `continue` 掉 ⟹ 窗口集合与全量不一致
⟹ **整段的漂移诊断判 unavailable**。

**实证**（预测→验证）：

| 组 | n | `min_decorrelated_samples` 中位 | 全量已低于门(20)的个数 |
|---|---|---|---|
| split-half 可用 | 8 | 26.5 | 1 |
| split-half 不可用 | 12 | **18.0** | **6**（11/10/15/14/16/10） |

⟹ **与「最需要漂移检验」反相关，而且是构造出来的。**

**对症的懒改法**：只比较两半与全量窗口集合的**交集**、并记录被丢的窗口，
而不是整段返回 unavailable。

---

## 10. 已撤回的推论（曾经写过，不成立）

留在这里防止重新论证。

| 曾经的说法 | 为什么不成立 |
|---|---|
| 「N 个 run 有窗口没过 ESS 门却标 trusted」交叉表 | 判据一侧是噪声：`docs/AUDIT_GATES_AND_CRITERIA_2026-09-17.md` 实测同配置同 seed 三次 w0 得 50.33/10.74/1.46（34×，门 10），全仓噪声底 ~30× |
| `Spearman(worst_ESS, cx_s2) = 0.048`、配对 7/10 | 同上，且 n=10~24 在 30× 噪声下不能做设计决策 |
| 「31/53 判 `REANCHOR_DUE` ⟹ f_k 漂了没人管」 | 读错了。那个 `criterion` 是 `derail_early_trigger_or_fixed_cadence_ceiling` = **节奏到点**，不是测出漂移。同产物 `displacement_is_not_the_trigger` 写明 0.5 阈值对 50~80 kJ 量程等于谁都超 |
| 「`offset_error_contribution` 是被丢掉的漏项」 | 它是估计器 A 的误差，而报出的是 B；A≡B。见 §7 |
| 「f_k 重标定是这批误差的解药」 | `cmet_ligand2` 端点窗口真机 A/B 已证伪（重标定 no-op，加帧/缩窗/插 λ 也全否） |

**唯一与 ESS 读数无关、仍然成立的一条**：3 个 run 的控制器以
`NO_FEASIBLE_ACTION` 认输退出，`results_untrusted` 仍是 `False`
（退出状态是记录事实，不是统计读数）。

§5 的塌陷证据**不受这条影响** —— 占据是从 `f_k` + 能量确定性算出来的。

---

## 11. 给 replay / 后续工作用的数据可用性清单

**逐窗口逐帧落盘的**（`vanishing/`，每窗 ~500 帧）：

| 文件 | 形状 | 内容 |
|---|---|---|
| `dual_window_{w}_vdw_energies.npy` | (n_states, n_frames) | `u_kj_raw`，kJ/mol |
| `dual_window_{w}_vdw_sampling_states.npy` | (n_frames, n_states) | 采样态口径 |
| `dual_window_{w}_vdw_bias.npy` | (n_frames,) | `bias_kj` |
| `dual_window_{w}_vdw_base.npy` | (n_frames,) | 基态势能 |
| `dual_window_{w}_vdw_residual_basis.npy` | (n_frames,) | residual 关时全 0 |
| `dual_window_{w}_vdw_convergence.json` | — | **`production_segments`：逐 block 的 start/end_frame / n_frames / session_id** |
| `dual_join_{u}_{d}_vdw_support.json` | — | 共享 λ 两侧 rawESS / g / top1% |

**block 结构**：`production_segments` 就是 block；额外 block 落在**兄弟目录**
`vanishing_2/`、`vanishing_3/`（只含拿到额外 block 的那个窗口）。
⟹ **实际预算分配史在盘上，可直接与反事实分配对比。**

**steps 换算**：`steps_per_update = 500`，
`n_frames × 500 = steps`（500 帧 × 500 = 250000 ✓ 对上 `n_steps_per_window`）。

**`bias_to_signal_ratio`**（`ibs_engine.py:20549`）：
`sd(bias_kj) / min_k sd(u_kj_raw[k])`，离线可算，`_is_gated = False`。
⚠️ 那条 `Spearman = +0.886`（vs ESS 的 −0.600）的出处是
**6 个窗口、单一体系**（4W53 甲苯溶剂腿），注释自己写着
「阈值无跨体系标定，刻意不设门」。现在有 **339 窗口 / 11 体系**，
**这个先验应当当场重新标定，不要直接采信 n=6 的系数。**

**没有的**：stage-2 生产**坐标一帧都没留**。全 run 只有
`pre_equilibration.dcd`、`rebalance_traj.dcd`、stage-1 的
`decharging_rep*.dcd`。`checkpoints/production_window/vdw` 58M 是
checkpoint 末态，不是帧集。
⟹ 任何需要构象描述子 `z(x)` 的检验（共享态构象一致性）
**不能回溯到这 24 个 run**，只能对新 run 生效。

---

## 12. 还没拆的层

- **L5 帧内**：`statistical_inefficiency_per_window`、逐态相关时间。
  500 帧全压在一个态时 `g` 意味着什么。
- 三个还没打开的产物键：`cumulative_fk_residual_production`、
  `window_overlap_diagnostics`、`max_common_mode_log_sigma_kT`。
- **剔掉 8 个塌陷窗口后，剩下的散布还有多大** —— 唯一能把「塌陷」
  从"一个真实缺陷"升级成"是/不是这批数字的主因"的计算。零 GPU。

---

## 13. fixed_budget_precision_replay 的结果（2026-09-18，零 GPU）

工具：`tools/audit/fixed_budget_precision_replay_2026-09-18.py`。
只读、不碰 production、调**生产同一个** `solve_stage_integrated`。
**准入自检通过**：全量重解 vs 产物 `covariance_chain_segments`，
偏差 ~1e-11 kJ/mol（机器精度）。

途中修掉三个提取错误（都会静默给错数，记下来防止重犯）：
1. 段目录有第三种命名 `vanishing_rewindow_<hash>`，只认 `_2`/`_3` 会把
   多段窗口误判成单段；
2. 把多块窗口的帧声明成"单段" ⟹ 生产侧对多段是**逐段各自去相关**
   （`_decorr_segments = w.get("production_segments")`），声明错会让全量重解
   差 1.4–2.9 kJ/mol。必须传真实块边界；
3. 自检粒度要按**窗口**判不是按腿判（同腿里有的窗口跨目录有的不跨）。

### 13.1 固定总预算下的重分配收益

49 条腿（总块数固定 = 实际值，每窗口至少 1 块）：

| 类别 | n | σ 降幅 |
|---|---|---|
| 自由度为零（每窗口只 1 块） | 17（15 sv / 2 cx） | 恒等 0%，非测量结果 |
| 有自由度、实际分配**已经等于**贪心最优 | 10（7 sv / 3 cx） | 0% |
| **重分配有收益** | **22（20 cx / 2 sv）** | **中位 9.9%，[0.8%, 38.3%]** |

⟹ 溶剂腿几乎没有头寸；头寸全在 complex 腿。

### 13.2 块间散布 σ 比报出的 MBAR σ 大约一倍

同一份帧、同一个估计器，`a_i` 改成取**块间样本方差**：

| | 块级 σ | 报出 total_error |
|---|---|---|
| cyclod_ligand3/rep2 cx | 7.48 | 2.93 |
| cmet_ligand1/rep2 cx | 5.33 | 2.94 |
| cmet_ligand1/rep1 cx | 4.34 | 2.10 |

这是第三条独立证据（另两条：split-half 放大中位 1.26×、跨 rep 实测 8–16×）。

### 13.3 块级 σ 在干净情形下是**对的**

对 34 个 rep 对，比 `|Δ(stage2 ΔG)|` 与 `sqrt(σ_a² + σ_b²)`：

| 条件 | n | 比值中位 | >2 的 |
|---|---|---|---|
| 全部 | 34 | 2.10 | 17/34 |
| 覆盖 <100%（有窗口 a_i 估不出） | 14 | 4.39 | 10/14 |
| **覆盖 100% 且无塌陷窗口** | **18** | **0.89** | 5/18 |

⟹ 误差棒不是原理上坏掉的。它是被少数病窗口拖坏的，而病窗口恰好就是
**连块级方差都估不出来**的那些。

### 13.4 分界量是占据不均，不是 ESS、不是 bias_to_signal_ratio

按每条腿的**最大占据不均**分档（n=34 rep 对）：

| 最大占据不均 | n | 比值中位 | >2 的 |
|---|---|---|---|
| < 5× | 3 | 0.42 | 0/3 |
| 5–20× | 12 | 0.89 | 2/12 |
| **> 20×** | **19** | **4.36** | **15/19** |

`Spearman(最大占据不均, 比值) = +0.507`，n=34。

占据不均是**确定性量**（给定 f_k 与帧就唯一），不是 30× 噪声底的单次采样读数
——这是它与 ESS 类读数的本质区别，也是为什么 §10 撤回 ESS 结论、但这条保留。

**先验标定（n=343 窗口 / 11 体系，目标量 = 块级 σ）**：

| 诊断量 | Spearman |
|---|---|
| `g`（统计低效） | **+0.549** |
| 占据不均 | +0.349 |
| `lambda_span` | −0.151 |
| `n_states` | −0.086 |
| **`bias_to_signal_ratio`** | **−0.074** |

⚠️ `ibs_engine.py:20520` 注释里的 `Spearman = +0.886` 出处是
**6 个窗口 / 单一体系**（注释自己写着阈值无跨体系标定）。跨 11 体系、
343 窗口重标定后掉到 **−0.074**。**不要拿它当 allocator 的先验。**
可用的是 `g`，其次占据不均。

### 13.5 ⚠️ P0 判决：目标函数在病窗口上与误差棒同样地瞎

对 22 条"重分配有收益"的腿，看**最病的窗口**（占据不均最大）会不会被
allocator 优先加块：

| 最病窗口的处置 | n |
|---|---|
| `a_i` 最大 ⟹ 会被优先加块 | 4 |
| **`a_i` 不是最大（排第 2–6）⟹ 不会优先** | **16** |
| **`a_i` 根本估不出来 ⟹ 直接被排除在优化之外** | **2** |

更糟的是最优解会**从最病的窗口抽走**块：

| run | 最病窗口占据不均 | 实际块 → 最优块 |
|---|---|---|
| brd4_ligand2/rep3 cx | 164.6× | 4 → **1** |
| cmet_ligand1/rep1 cx | 41.8× | 3 → **1** |
| cyclod_ligand2/rep3 cx | 50.4× | 3 → **1** |
| brd4_ligand1/rep1 cx | 73.7× | 3 → 2 |
| cyclod_ligand3/rep2 cx | 391158× | 1 → 1（不动） |

**机制**：塌陷窗口的帧全压在一个态上，于是**每一块都给出同一个（错的）值**
⟹ 块间散布很小 ⟹ `a_i` 很小 ⟹ 边际方差下降 allocator 判定
"这个窗口已经收敛了，别再花钱" ⟹ **抽走它的块**。

**塌陷让病窗口看起来最精确。** 这与误差棒的失明是同一个根因，只是高了一层。

⟹ **结论**：P0 的头寸是真的（复合物腿中位 ~10%、最高 38.3%），但**必须先有
结构前置门**，否则期望收益为负。原提案里那句「永远把下一块给 score 最大
**且已经通过 structural/support gate** 的窗口」不是可选项 —— 缺了它
18/22 条腿会误分配。

该门必须键在**占据不均**上（确定性、跨体系分界在 ~20×），
**不能**键在 ESS（30× 噪声底）或 `bias_to_signal_ratio`（−0.074）上。
门的语义应当是「占据不均超限的窗口不得被判定为已收敛/不得被抽走预算，
其 `a_i` 视为无界」，而不是拿它去反向否决已产出的 ΔG
（后者与 `min_occupancy_gate_retired_reason` 的退役理由冲突，前者不冲突）。
阈值要定在哪需要用户拍板 —— 本仓库有把只在某些假设下成立的度量提成硬门
反复咬人的历史。

---

## 14. 已落地的修复（2026-09-18）

全量回归 **2757 passed / 7 skipped / 6 xfailed / 0 failed**（3m22s，三处改动全部落地后）。
⚠️ 期间出现过一次 8 failed（`test_outer_lambda_refit_wiring.py`）——
**是并发编辑的瞬态**：另一个会话正在改 `runabfe.py`，单独重跑 8 passed。
多会话同时改本仓时，回归结果要复跑一次才作数。
6 个 xfail 是既有的 `combine_ibs_and_independent_endpoint` 缺 `analysis_status`
那条，与本次改动无关。

### 14.1 split-half 交集修复（`ibs_engine.split_half_drift_diagnostics`）

原来半程窗口集合与全量不一致就整段 `available=False`（§9）。改成取**交集**、
缺的窗口记进 `windows_missing_from_halves`，并新增
`n_windows_compared` / `coverage_complete` / `coverage_note`。

⚠️ 总量项在集合不全时置 **`None`，不是 0** —— 两半积的不是同一段 λ，
相减无物理意义；0 会被读成「实测总漂移为零」。消费侧的打印也改成
不可比分支（原来无条件 `:+.3f`，遇到 None 会让诊断把主求解带崩）。

**实测收益**（39 条自检通过的腿，重解 ΔG 与产物逐位相同）：

| | |
|---|---|
| 从暗变亮 | **4 条**：brd4_ligand2/rep1 cx、cmet_ligand2/rep1 cx、cmet_ligand2/rep3 cx、jnk1_ligand1/rep1 cx |
| 新拿到的 `max_window_z` | 中位 2.09，2/4 超过 2 |
| 仍 unavailable | 6 条，但理由变成**「没有任何共同窗口」** —— 半程数据下整条路径都解不出来，这不是代码 bug 是预算不够 |

单例：`cmet_ligand2/rep1` cx 立刻抓到 **w1 漂移 +2.380 kJ/mol = 2.39×2σ**，
该窗口修复前完全不可见。

测试：`tests/test_split_half_uses_the_intersection_2026_09_18.py`（5 条行为测试，
monkeypatch 半程求解，只验集合逻辑）。

### 14.2 σ 放大接出来（`abfe_pipeline.py`，默认 True）

`inflate_sigma_from_split_half` 原来**没有任何调用方传过** —— 开关存在、
内部接通、外面够不着。现在两处（`_final_gate_thresholds` 与
`solve_stage_integrated` 调用）都读 `kwargs`，默认值逐字一致。

它进的是 `final_gate_thresholds` ⟹ `_stage_window_sampling_identity` 会 pop 掉
⟹ **stage 结果重算、窗口轨迹复用、不重跑一步 MD**。

端到端（`cyclod_ligand1/rep3` cx）：

```
inflate=False  ΔG=56.457  total_error=1.522  max_endpoint_unc=0.769
inflate=True   ΔG=56.457  total_error=3.704  max_endpoint_unc=2.313
               mbar_only=1.522
逐段 σ  [0.266,0.769,0.583,0.740,0.667,0.570]
     → [1.015,2.313,0.583,1.381,2.037,0.970]
```

ΔG 一位未动；无漂移证据的窗口 σ 不动（w2 保持 0.583）；端点门跟着重算
（P1-23 路径）；原值保留成 `total_error_mbar_only_kJ_mol`。

默认 True 的理由：`σ ≥ |漂移|/2` 是**下界**不是猜测。吃它的
`max_endpoint_uncertainty_kJ_mol` 门是「仅诊断、不拒绝、不触发补帧」，
所以打开不会硬失败任何 run。已知过估方向（总量稳而逐窗大 = 归属重排）
在注释里保留。

### 14.2b runabfe 透传：我自己踩了同一个坑

`inflate_sigma_from_split_half` 第一版只改到 `abfe_pipeline`，但 `runabfe.py`
是**逐键显式枚举**往 `run_full_pipeline` 传参的 —— 没枚举的键永远进不了
`kwargs`。后果不是默认值失效（默认 True 照常生效），而是**关不掉**：
做 σ 口径 A/B 时没有对照臂。

这与本仓库已记录的 `stage2_autonomous_controller` 是同一个形状
（注释原话：「在 abfe_pipeline 里一直是 `kwargs.get(..., True)`，但**从来没有
人往下传** ⟹ 配置里写了也没用」）。已补上两条腿的透传，并加断言校验
`config → runabfe(2 腿) → run_full_pipeline kwargs → solve + 指纹` 整条链。

⚠️ 两条腿必须同口径：一腿放大一腿不放大，ΔG_bind 的误差合成就是混口径的。

### 14.3 逐窗口占据：**不需要新代码，数据本来就在**

`mixture_occupancy_normalized`（逐窗口、逐态、已归一化）已经落在

```
final_binding_results.json
  /diagnostics/{complex,solvent}_stage_diagnostics/stage2
    /window_overlap_diagnostics[i]/mixture_occupancy_normalized
```

**24/24 个 run 都有。** 同一条目里还带 `min_occupancy_normalized`、
`bias_to_signal_ratio`、`statistical_inefficiency_per_lambda`、
`top1pct_raw_weight`、`raw_ess_ratio_per_lambda` 等 —— 未来的有效性层
所需的量基本齐了，不用加落盘。

### 14.4 ⚠️ 两个占据口径不是同一个量（定阈值前必读）

| | 帧 | f_k |
|---|---|---|
| 产物 `mixture_occupancy_normalized` | **去相关后**（n_decorr 34–417） | 最终分析用的 |
| 本文档 §5 的离线值 | 原始全部（500–1500） | `ibs_state_vdw_window_*.json` 的冻结值 |

实测对照（不均 = max/min）：

| run / 窗口 | 产物 | 离线 |
|---|---|---|
| cmet_ligand1/rep1 cx w0 | 3.0 | 11.2 |
| cmet_ligand1/rep1 cx w5 | 6.0 | 41.8 |
| cmet_ligand1/rep1 cx w6 | 40.2 | 15.3 |
| cyclod_ligand3/rep2 cx w4 | 45.7 | 50.7 |
| **cyclod_ligand3/rep2 cx w5** | **219194** | **391158** |

⟹ **灾难窗口两个口径一致（同量级），中等窗口最多差 7 倍，个别反向。**
§5、§13.4 的所有数字（含「> 20×」那条分界）用的是**离线口径**。
做 ROC / 定阈值时必须选定一个口径并全程不换，两套数不得混用。
