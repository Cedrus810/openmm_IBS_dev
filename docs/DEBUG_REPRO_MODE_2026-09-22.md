# 复现（debug）路线与 openfe 借用 —— 2026-09-22

[文档导航](README.md) · [当前科学状态](STATUS.md) · [变更记录](CHANGELOG.md)

> **本文记的是"改了什么、为什么、怎么核"**，不是科学结论。
> 这一整套的目的只有一个：**让同一体系的不同 rep / 不同机器共用逐位相同的布局，
> 把 ΔG 的差分压成"只反映采样"**，好在上面抓热力学循环的 bug。
> ⚠️ 全部开关**默认关闭**，不传时生产路线逐字不变。

---

## 0. 一句话总览

| 分组 | 键 | 默认 | 作用 |
|---|---|---|---|
| 布局冻结 | `stage2_uniform_lambda_states` | 未设 | 等分 λ 表 + 整套强制项（见 §2） |
| 布局冻结 | `stage2_window_ranges` | 未设 | 显式分窗（复现模式下**必填**） |
| 布局冻结 | `stage2_repro_fk_cycles` | 3 | f_k 固定节奏的轮数（09-23 起；原为 4） |
| 布局冻结 | `stage2_repro_fk_tilt_signs` | `[1, 0, -1]`（轮数=3 时） | 逐轮倾斜 +1 左 / 0 中 / -1 右，项数 = 轮数；`[0,0,0]` = 关 |
| 判据 | `attachment_enforce_convergence_gates` | 复现模式下 false | attachment 两道收敛门是否阻断 |
| Boresch | `boresch_force_constant_mode` | `"openfe_fixed"`（09-24 起；原为 `"fitted"`） | `"openfe_fixed"` = 4184 / 334.72 |
| Boresch | `boresch_anchor_mode` | `"openfe"`（09-24 起；原为 `"orb"`） | `"openfe"` = 逐帧判据 + 涨落排序 + 芳香环池 |
| 哈密顿量 | `softcore_power_lj` | `[1, 1]`（09-24 起；原为 `[2, 2]`） | `[1, 1]` = openfe/openmmtools 口径 |

---

## 1. 为什么要这条路线（诊断链）

按发现顺序，每一条都有真机数据支撑。

### 1.1 f_k 是极限环，不是发散也不是"没动"

真机 `cyclod_ligand1/rep3` 窗口 3（λ_vdw 0.40→0.20，26 态等分）的 `f_history_kj_mol`
50 轮谱宽：

```
35 44 54 64 74 | 71 61 51 41 31 21 14 | 15 25 ... 85 | 81 71 ... 7 | 9 19 ... 89 | 86 ... 12 | 17 27 37
```

峰 74 → 85 → 89，谷 14 → 7 → 12，周期 ≈18 轮，**逐轮步长 8.83 ≈ pairwise 上限 9.98
（每轮都顶格）**。50 轮 = 2.7 个周期 ⟹ 跑多久都不会停。

用该窗口**真实的 `u_kn`**（tmbar_history 5 批 × 40 帧）离线驱动真实更新律：
无论增益降到 0.35、加震荡刹车、还是对 f 做一阶平滑，**都单调收敛**（谱宽 145–213，
末步长 0.05）。⟹ **震荡不在 f_k 的代数里**，在"帧会跟着 f_k 变"这一环：
f_k 需要走 145，限幅每轮只给 10，而占据约 9 轮就翻到另一端、那批新帧的 u 结构跟着变。
追不上 ⟹ 锁进极限环。

### 1.2 145 kJ/mol 本身就不对

同一个 run 的 pilot 自测（⟨dU/dλ⟩ 积分）给窗口 3 的相邻 ΔA：

```
0.40→0.36 +1.38 | 0.36→0.32 +2.79 | 0.32→0.28 +5.30 | 0.28→0.24 +7.71 | 0.24→0.20 +7.56
合计 ≈ 24.7 kJ/mol
```

f_k 谱宽**应该**是 ~25，离线收敛到的是 **145**（≈ `mean_softcore_u` 的谱宽 155）。
⟹ 占据反馈的不动点 `K·p_k = 1` **在有重叠时**给 ΔA，**在零重叠时**退化成"把 ⟨u_k⟩ 对齐"，
两者差 6 倍，而代码里没有任何东西拿 pilot 已经测出来的尺度去约束它。

### 1.3 pilot 的 dU/dλ 没算错，但重尾说明配体被撞

三个解出来的窗口 vs pilot TI，符号相反（积分方向）、幅度差 7–16%：

| | 实测 ΔA | pilot TI |
|---|---|---|
| 窗口 0 λ 1.0→0.8 | +40.23 | −43.28 |
| 窗口 1 λ 0.8→0.6 | +22.94 | −25.83 |
| 窗口 2 λ 0.6→0.4 | +6.25 | −7.43 |

但逐点 std 只在 **λ_vdw ∈ [0.24, 0.43]** 炸起来（mean +158 / std **84**，两头 std 只有 7–16），
峰值 **+222 kJ/mol @ λ=0.238** 正落在失败的窗口 3 里。重尾 = 少数帧的巨大排斥顶起均值
⟹ 鬼影配体被撞。

### 1.4 撞不是 Boresch 缺席，是约束太软 + λ 太疏

- 我们 stage 1/2 **全程带 Boresch 满强度**（`boresch_params` 逐处透传），与 openfe 的
  循环顺序一致 ⟹ 不是"没拴住"。
- 但拟合出来的角力常数被 clip 在 **[10, 1000]**，真机实测常压在下限
  （`kphiB=10`）⟹ σ_θ = √(kT/K) ≈ 0.5 rad ≈ **29°**，在 r0≈0.5 nm 上就是 ~0.25 nm 横向摆动。
  预平衡实测位姿漂移 **0.497–0.731 nm**（超 0.4 上限）与之同向。
- 而 dU/dλ 峰所在的那段，度规分窗给 Δλ≈0.022，**等分给 0.04** —— 这是复现协议
  已知且已被接受的代价，不是 bug。

---

## 2. 复现模式：`stage2_uniform_lambda_states=N`

打开后 `_freeze_stage2_path_for_repro()` 做四件事（`abfe_pipeline.py`）：

1. **λ 表 = 等分**：`λ_i = 1 − i/(N−1)`，端点精确 1.0/0.0，IEEE 双精度下逐位确定。
   pilot **照跑**（f_k 的 TI 热启动、metric_g 诊断还要它），但布点不由它决定。
   替换发生时日志会打一行"上面那张热力学长度 λ 表已被丢弃"——上游 `abfe_preoptimizer`
   刚打过一张会被丢弃的表，不说一声会被当成"等分没生效"（09-22 真机报过一次假警报）。
2. **fail-closed 三条**：必须给 `stage2_window_ranges`；`stage2_final_n_states` 必须等于 N；
   每窗态数必须落在 `[stage2_window_min_states, stage2_window_max_states]`
   （这条在**开跑头几秒**判，不等 pilot 烧完才炸在 stage 2 门口）。
3. **强制置位**：`sampling_repair_policy=non_mutating_v1`、`max_path_insertions=0`、
   `stage2_frozen_layout=true`、`stage2_autonomous_controller=false`、
   `stage2_enable_bridge_rescue=false`、`allow_untrusted_stage_results=true`。
   ⚠️ 前两条必须**一起**关：benchmark 的 config 全部预置 `path_evolution_v1`，
   而「关掉控制器」**不冻结布局**。bridge rescue 的开关是 `and not _autonomous_on`，
   关控制器反而会打开它，所以必须显式关。
4. **f_k 固定节奏**（见 §3）。

`stage2_frozen_layout` 传进 `Stage2RepairController`（万一有人把控制器打开时的兜底）：
`feasible()` 与 `rewindow_feasible()` 把 INSERT_LAMBDA / SPLIT_TAIL_WINDOW /
IMMUTABLE_REWINDOW 一律判不可行，`_frames_admission` 对同一份 f_k 不再批补帧块
（= 固定长度采样），而 `hard_cap_only` 的动作（换 f_k / 探针 / 临时生产）照常放行。

## 3. f_k 固定节奏（替代控制器的"什么时候换"）

```
跑满一段（n_steps_per_window） → 拿这段的生产帧重解**全部**窗口的 f_k → 再跑一段
```
共 `stage2_repro_fk_cycles`（默认 3，2026-09-23 起；原为 4）轮，位移阈值传 **0.0**（不按位移挑窗口），
源段是**上一段**而不是永远段 1，最后所有段交给 multi_segment 合并求解
（段间换过 f_k ⟹ 不是同一个采样分布，不能把帧摞起来）。

只在**复现模式且控制器关闭**时生效——控制器开着时 `decide()` 的 5a 分支做的是同一件事，
两套并存正是本仓最贵的那个形状。

## 4. 判据一律降 WARN（debug 路线）

| 判据 | 生产路线 | 复现模式 |
|---|---|---|
| attachment BAR/TI 一致性 + split-half | 阻断 | 只记录（`[WARN] [门已关闭]`） |
| 下游 `validate_boresch_attachment_result` 的 `converged=true` | 硬判 | 按 `convergence_gates_enforced` 放宽 |
| 预热在预算内未收敛（`f_not_converged`） | raise | 走临时生产（`debug_provisional_on_warmup_failure`） |
| bridge rescue 的 min_overlap | 阻断 | 整个关掉 |
| 阈值类 stage 判据 | 按 untrusted | `allow_untrusted_stage_results=true` |

⚠️ **两条故意没关**：ΔG/σ 非有限、路径不完整（某窗一帧可用的都没有）。
它们不是"门太严"，是**根本没有这个数**，放行等于编一个。第二条现在不炸穿管线——
控制器关着时走 `_collect_indeterminate` / `_finalize_with_indeterminate`，
run 跑完、盘上有数、结果里带 `indeterminate_windows`。

⚠️ **记账一个没少**：`converged` 如实记 False，新增 `convergence_gates_enforced` 说明
"这次是拦了还是只记录了"；临时生产的 `bias_status=provisional_production`、
`f_k_evidence_status=indeterminate`、`warmup_failure.json` 保留不删。

## 5. 从 openfe 借的三件

### 5.1 Boresch 固定力常数（`boresch_force_constant_mode="openfe_fixed"`）

`kr = 4184 kJ/mol/nm²`（10 kcal/mol/Å²）、五个角/二面角项 `334.72 kJ/mol/rad²`
（80 kcal/mol/rad²）。力常数是**设置项**，不从轨迹拟合：挑锚点去适配固定的 K，
而不是拟合 K 去适配挑中的锚点。

⚠️ 生产走的是 `source=simple` → **`GeometricRestraintEstimator`**，所以键接在**它**上面；
只加在 `OrbBoreschEstimator` 上等于 config 写了没人读（09-22 真机踩过一次）。
⚠️ 还有**第二道裁剪**在 `runabfe.resolve_boresch_restraint` 末尾（`kr` 上限 2000），
会把 4184 裁成 2000 —— 固定档整段跳过那次裁剪（裁一个设置项没有意义），
并落一条 `force_constant_post_clip_skipped`。
📌 顺带一条读数：**拟合档下 kr 每次都恰好是 2000**（三次历史日志都是），
说明拟合值一直顶着那个上限，那个上限在拟合档也值得重看——但那是另一件事。

### 5.2 锚点判据（`boresch_anchor_mode="openfe"`）

本质差别：我们原来在 **frame 0 一帧**上校验几何，openfe 要求判据**每一帧**都成立，
再按**涨落最小**排序。

| 判据 | 阈值 |
|---|---|
| 任一帧 \|φ\| > 150° | 拒（远离 ±π 缠绕边界） |
| 角的圆标准差 ≥ 100° | 拒 |
| 二面角的圆标准差 ≥ 300° | 拒 |
| θ 平到撞 [0, π] 边界（给定 K、T） | 拒（= 我们那道 sinθ 奇点门，前移到选锚点） |
| 排序量 | Σ(二面角圆标准差 + 角圆标准差) + 键长方差，越小越好 |

圆标准差用 `sqrt(-2 ln R)`，绕边界安全（一组贴着 ±π 的角，线性 std 算出 >1.0 rad，
圆标准差 <0.1）。

**配体池**：优先**芳香环上的重原子**，芳香性交给 **RDKit** 判
（拓扑连接性 + 平均构型 → `DetermineBondOrders` → `GetIsAromatic`），
整环 max RMSF 低于 cutoff 才要这个环；芳香重原子不足 3 个或 RDKit 判不动就退回全部重原子
（cyclod 这类小片段基本都会走退回分支，符合预期）。

> **09-24 更新：候选搜索也照 openfe 重写了**（`GeometricRestraintEstimator._openfe_select_combo`）。上面那些判据只是**淘汰**；09-22 版的候选仍来自老路（第 0 帧 ≤0.5 nm 接触对 + [45°,135°]），而 openfe 要求 H0–G0 **每帧 0.5–1.5 nm** —— 两者几乎不重叠，所以常被筛空（jnk1 rep3 崩溃）。现在的流程：
>
> 1. 配体池：芳香环（整环**局部** RMSF<0.1 nm）→ 不够则 RMSF<0.1 nm 的重原子 → 再不够全部重原子。09-22 版 RMSF 没对齐，配体整体晃动被算成涨落，上面那句「基本都会走退回分支」其实是这个 bug。
> 2. G0-G1-G2：池内成键、不共线，按离分子图中心由近到远排。
> 3. 受体池：backbone、局部 RMSF<0.1 nm、DSSP 稳定 H/E（壳内不足 20 个则不筛）；H0 每帧距 G0 0.5–1.5 nm，H0-G0-G1 不平、H0-G0-G1-G2 |φ|≤150°；H1/H2 经真实键连到 H0 且也在池内。
> 4. 六原子全判据逐帧过（θA/θB 不平、三个 |φ|≤150°、不全帧共线），取 Σcircvar+Var(r/Å) 最小；第一个配得上的配体三元组即停。
>
> 与 openfe 的差别（有意）：芳香档不要求三元组落在同一个环里；芳香池配不上时继续试刚性重原子（openfe 直接报错）。

### 5.3 软核指数（`softcore_power_lj=[1,1]`）

> **09-24 更新：生产默认已改成 `(1, 1)`**（`abfe_core.DEFAULT_SOFTCORE_POWER_LJ`），这个键现在只在想换回别的指数时才需要写。

原来我们是 `(a, b) = (2, 2)`（前因子 λ²、软核偏移 (1−λ)²），openfe/openmmtools 是 `(1, 1)`。
两端完全一样（λ=1 精确 LJ、λ=0 恒 0），**差别全在路径上**。

~~b=2 让软化二次地打开 ⟹ 热力学长度往中段堆 ⟹ §1.3 那个 +222 的尖峰。~~
**09-24 对照否掉**：benchmark 09-09 [2,2] 批（`logs_20260909/*/pipeline.log`）与 09-23 [1,1] 批
（`preopt_dual_vanishing.json` 的 `path_diagnostics`）同一 pilot 协议（每态 30000 步）：

| 复合物腿总热力学长度 | [2,2] | [1,1] |
|---|---|---|
| brd4 L1 | 22.05 / 20.08 | 20.98 / 21.66 / 20.62 |
| brd4 L2 | 16.66 / 16.85 | 17.50 / 17.96 / 16.28 |
| cmet L1 | 22.94 / 23.16 | 22.98 / 23.60 / 24.71 |
| cmet L2 | 21.63 / 18.37 | 19.23 / 19.84 / 19.75 |
| cyclod L1 | 16.50 | 15.47 / 15.91 / 14.62 |
| cyclod L2 | 17.87 / 18.03 / 18.12 | 19.14 / 18.28 / 18.19 |

[1,1] 下 dU/dλ 的 std 峰仍在 λ≈0.19–0.29（cyclod L1 复合物 67–79，[2,2] 那次 84）。
⟹ 尖峰不是软核指数造成的；选 [1,1] 是为了与 openfe/openmmtools（及 Pham & Shirts 2011 的 1-1-6）同口径。
局限：[2,2] 侧是 09-09 代码、pilot 17 点（[1,1] 22 点，粗格子只会把 TL 估低），每个 TL 是单次 pilot、无误差棒。

口径走**一个模块级开关** `abfe_core.SOFTCORE_POWER_LJ_OVERRIDE`：`optimize_alpha()` 有近十个
调用点（pipeline / ibs_engine / preoptimizer 各自都建同一份软核），逐处加形参一定漏，
漏掉的那一处就是同一个 run 里两套哈密顿量。它进 `aces_softcore_params` ⟹ 自动进 stage 指纹。

### 5.4 借了但**没**照搬的

- **REPEX**：openfe 用副本交换把慢模态换出去（"它不在单个窗口里解"）。我们没有，
  REMD P0 08-31 叫停等大更新。
- **attachment 腿改 REMD**：❌ 已试过并记在 `run_boresch_attachment_leg` 的 docstring：
  产出 38.6±110、零 round trip，根因是 `k(1−cosΔ)` 翻转取 2k（三个二面角全反转 477 kT），
  且解析释放项本身假定单一简谐盆地。换固定力常数后翻转惩罚 ∝ k，更不能走。
- **attachment λ 阶梯加密**（openfe 是 6 窗 0→0.2→…→1.0，我们是 4 态 [0, 0.1, 0.35, 1.0]）：
  **待定，用户 09-22 决定搁置**。换固定力常数后 attachment ΔG 从 ~4.7 抬到 ~24 kJ/mol、
  σ 0.09→0.32，BAR/TI 差 2.4–3.0（容差 1.0）三次里两次不过，方向指向阶梯太疏。

---

## 6. 怎么核"真生效了"

重起后看这几行，缺哪行就是哪条没吃到：

```
[复现模式] stage2_uniform_lambda_states=26：λ 表等分（Δλ=0.04）…
[复现模式] ⚠️ 上面那张热力学长度 λ 表**已被丢弃**，改用等分表…
[锚点/openfe] 配体池：…
[锚点/openfe] 额外禁选 X/Y 个候选（|φ|>150° 或圆标准差超限）
Boresch 力常数: kr=4184, kthetaA=334.72, …
Boresch 力常数走固定档（openfe_fixed），跳过最终裁剪…
[复现模式] f_k 固定节奏 k/3：…（默认 3 轮）
```

盘上的事实看这两处（**不要**看 `pipeline.log` 里 preopt 那张中间表）：
`checkpoints/preopt_dual_vanishing.json` 的 `lambdas_var`、
`checkpoints/ibs_state_vdw_window_*.json` 的 `lambdas_vdw`。

---

## 7. 破缓存范围

| 改动 | 作废什么 |
|---|---|
| `softcore_power_lj` | 换哈密顿量 ⟹ 全部窗口缓存 |
| `boresch_*`（两个） | 换**束缚态定义** ⟹ `boresch_simple.json` / `boresch_equilibrium_committed.json` / `stage0_attachment.json` / `attachment/` 及全部下游；ΔG 会变 |
| `stage2_uniform_lambda_states` | 进 `_PREOPT_DERIVED_PATH_KEYS_OPTIONAL`（**只在显式设了才进指纹**，不开不作废任何既有缓存）；归派生层 ⟹ 可从旧 pilot 离线重算，不必重烧 GPU |
| `stage2_window_max_states` 等既有派生键 | 一直如此：改它作废 preopt 与下游 |

⚠️ 这批开关跑出来的数**与之前那批不可比**，别混着看。
