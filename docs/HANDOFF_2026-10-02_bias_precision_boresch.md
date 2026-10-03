# 交接：偏负 / 精度 / Boresch / 口袋水（2026-09-27 → 10-02）

> 一次对话（09-27 ~ 10-02）的交接。本文只记**状态、数字、坑**；逐条代码改动见
> [CHANGELOG.md](CHANGELOG.md)，科学结论的正式登记处仍是 [STATUS.md](STATUS.md)（本文没有改它）。
> 记忆里对应的条目：偏负排查清单（`project_negative_bias_status_2026-09-24`）、
> HSP90 散布归因（`project_hsp90_l3_precision_attribution_2026-09-29`）、口袋水线（`project_pocket_water_reopened_2026-09-27`）。

---

## 0. 一句话现状

- **偏负（ΔG_bind 比实验负 4–6 kcal）是三家共有的**：OpenFE、本仓 IBS、FEP suite（GROMACS，**加上 LRC 后**）
  在同一套 abfe-benchmark 输入上结果一致（`abfe-benchmark/openmm_IBS/crosscheck_3methods_20260930.csv`）。
  FEP suite 表里的数**没有 LRC**，它看起来更近实验只是口径差（用户 10-02 确认），别再当线索。
- 已排除：口袋水、Boresch 约束强度、锚点选择、起始结构、stage 1 起跑瞬态。剩下的只能是力场 / 输入 / 实验参照这一层，
  不在工作流里。
- **精度**（rep 间散布）是本仓真正能改的东西；本轮改了 stage 1 平衡段和锚点规则，stage 2 的「随机落盆」还在。

> **最新（10-03）**：PWWP1 L9/L10 跑完——复合物 stage 2 散布只有 1–2 kJ（HSP90 是 11–13），散布最大的变成复合物 stage 1；新键 `stage1_n_steps_per_window` 落地，PWWP1 L9 加长去电荷的 3 rep 在跑。见 §6。
>
> **最新（10-02 晚）**：stage 2 接力已实现并在 HSP90_ligand-27 上真机试过 ⟹ **把复合物 stage 2 整体压低 ~26 kJ、离论文更远，已关**（§2b）。
> 整批推荐配置 = 标准 `_lcr` + `stage1_remd_equilibration_steps: 125000`，**不开接力**。所有未结问题汇总在 §4。
>
> **10-02 方向（用户拍板）**：不再追 abfe-benchmark（三家方法全偏负，比不出工作流好坏）；
> **重心转 fragment-opt-abfe-benchmark**——它有同拓扑的论文 GROMACS 逐 rep 参考（`free_energy_data/abfe.csv`），
> 可以逐体系比「我们 vs 论文」。下面 §1 的 abfe-benchmark 结论只作背景。

---

## 1. 排除了什么（证据 + 数据位置）

| 候选 | 结论 | 证据 |
|---|---|---|
| 口袋水（端点态） | 不是 brd4 偏负来源 | 自写搬水 MC vs 纯 MD，λ=1/λ=0 各 1 ns，口袋水数差 ≤1.4（MC 净插入≈0）；`abfe-benchmark/openmm_IBS/pocket_water_sampler/` |
| 口袋水（λ 0.4→0） | 不是载体 | w34 双起点 + 鬼影体积计数：brd4 L2 两起点一致；L1 的 25 kJ 起点依赖由蛋白侧链占腔携带；HSP90 L3 两起点差 0.76 kJ；`hysteresis_w34_water/`（两个 benchmark 树下都有） |
| 起始结构 / MC-MD 水平衡 | 无影响 | HSP90 L3：我们起点 3 rep 均值 −6.03，论文 MC/MD 后起点 3 rep −8.05（方向相反、在散布内）；`fragment-opt-abfe-benchmark/openmmIBS/runs_lcr{,_paperstart,_startcmp}/` |
| Boresch 约束强度 | 均值不变（噪声内） | brd4 L2 新锚点：kθ 334.72 → −11.90±0.99；kθ 41.84 → −13.26±1.53；旧锚点 −13.29±1.16；`abfe-benchmark/openmm_IBS/runs_lcr_v2anchor_*` |
| 解析 release 近似 | 可忽略 | 完整雅可比数值积分：硬约束差 0.017、软约束差 0.147 kJ/mol |
| 加硬角度约束能否修 stage 2 双盆 | 不能，且有副作用 | kθ=1000：w0–w2 照样双盆；一个 rep 预平衡漂离姿态后被钉死在非晶体姿态（作废） |

---

## 2. HSP90 ligand-3（fragment-opt）精度归因

数据：`fragment-opt-abfe-benchmark/openmmIBS/runs_lcr*/HSP90_ligand-3/`。论文 GROMACS 5 rep −5.21（SD≈0.55，每 λ 态 20 ns），实验 −4.23。

- 我们 rep 间 SD ≈2.2 kcal，论文 0.55。采样量差约 40×（我们 stage 1 每态 0.5 ns、stage 2 每段 0.5 ns）。
- **stage 1（复合物去电荷）**：REMD 原来只有 10 ps 轻预热就开生产；前 125 ps 均值偏高 ~5 kJ。按残基归因 =
  Lys58 / Asn51 / Asp54 / Asp93（拓扑号 LYS43/ASN36/ASP39/ASP78，+15 = HSP90 编号）在各 rep 冻在不同态，水部分抵消。
  加平衡段后 5 个 run 的 stage 1 SD 8.4 → 2.6 kJ。
- **stage 2（去 vdW）**：没有起跑瞬态；w0–w2 每段落在两个离散值之一（差 8–12 kJ），由配体–Asp93 侧链距离决定
  （|r|≈0.84–0.92）。**真正动的是盖子螺旋**：在受体锚点（ALA40/LEU41 骨架）坐标系里配体只动 0.2–0.5 Å，
  Asp93 侧链动 0.5–1.3 Å ⟹ 口袋在弯，配体被拴在盖子侧被带离 Asp93。Boresch 六坐标两盆几乎相同（差 ~0.6 kT），
  所以换约束强度管不到它。这是蛋白自己的慢自由度，目前**没有修**。
- 工具：`tools/diagnostics/decharging_group_attribution.py`、`tools/diagnostics/stage2_basin_features.py`。

## 2b. HSP90 ligand-27：接力 / stage 1 平衡段对照（10-02）

三组都是 3 rep 串行、同锚点（受体 2161/2163/2178，配体 3350/3349/3348）、同拓扑；kJ/mol，取自各 rep `checkpoints/stage{1_decharging,2_vanishing}.json` 的 `total_delta_G`。

| | 基线 `runs_lcr/`（09-30~10-01 代码） | 对照 `runs_lcr_s1eq/`（只开 stage 1 平衡段） | `runs_lcr_relay/`（平衡段 + 接力） |
|---|---|---|---|
| 复合物 stage 1 | 82.2，SD 2.3 | 83.9，SD 3.3 | 84.9，SD 0.7 |
| 复合物 stage 2 | 57.1，SD 13.3 | **61.0**，SD 12.6 | **31.0**，SD 7.1 |
| 溶剂 stage 1 | 63.2，SD 0.3 | 61.6，SD 2.0 | 61.3，SD 2.1 |
| 溶剂 stage 2 | −4.0，SD 0.4 | −4.9，SD 0.6 | −6.5，SD 0.9 |
| ΔG_bind（kcal） | −11.17，SD 3.82 | **−13.03**，SD 3.03 | −6.52，SD 1.69 |

论文 GROMACS 5 rep 均值 −12.06（SD≈1.3），实验 −8.63。config 在 `fragment-opt-abfe-benchmark/openmmIBS/configs/HSP90_ligand-27_lcr_{s1eq,relay}.json`。

- **接力 ⟹ 复合物 stage 2 −26 kJ（t≈3），是接力本身造成的**（对照组回到基线）。逐窗：w1 −8、w3 −5 为主；逐窗 SD 没降（w2/w3 反而 ~10 kJ），总量 SD 降一半可能是 n=3 运气。
  已排除实现层面的盒子错位：stage 2 是固定盒，各段末态盒子都是 8.149 nm，接力起点坐标/盒子一致；每段起点 log 和 ledger `start_from` 都对、无回退。
  **为什么会移 26 kJ 没查**——物理上连续采样应该更平衡，而论文每态连续 20 ns 却给 −12，和接力结果矛盾。可能方向：接力链在低耦合窗里蛋白/口袋松弛到论文没到的态；f_k 固定在 pilot（耦合态起点）后、接力段的系综偏离 f_k 对应的分布；或接力链里有别的问题。
- **stage 1 平衡段**：L3 上复合物 stage 1 SD 8.4→2.6，L27 上没复现（2.3→3.3，n=3）。均值不变、无害，推荐照开。
- **溶剂 stage 1 SD 0.3→2.0**：开平衡段的两组都这样，各有一个 rep 落在 ~59（基线三个都 ~63）。猜测基线卡在同一个配体构象、散布被低估（未查）；均值影响 ~0.4 kcal。
- 溶剂腿 stage 2 只跑 1 段，接力根本没触发，但接力组仍比对照低 1.6 kJ（t≈2.6）——没解释，可能是噪声。

---

## 3. 本轮代码改动（都在主线，状态）

| 改动 | 默认 | 验证 |
|---|---|---|
| openfe 锚点 θA/θB 硬窗口 [30°,150°]，单帧选锚点（复现模式）收紧到 [45°,135°]；平衡值均值不在 [30°,150°] 时 fail-closed（`abfe_core.anchor_theta_in_window`、`_openfe_select_combo`、`openfe_anchor_metrics`、`estimate_from_trajectory`） | **默认生效**（只影响新估算锚点） | 测试 `tests/test_boresch_anchor_theta_window_2026_09_30.py` 6 条；真机 brd4 L2 新锚点 θA0≈95°/θB0≈84°，6 个 run 正常（软约束不再崩） |
| `stage1_remd_equilibration_steps`：去电荷 REMD 生产前平衡段 | 不设=关 | 真机生效（log「[stage1 平衡段]」）；REMD 平衡段路径**无单元测试** |
| `boresch_fixed_k_angle_kj_per_mol_rad2`：固定档角/二面角 K 覆盖（10–1000） | 不设=关 | 真机生效；用户结论「不放大」，留作实验开关 |
| `boresch_pose_guard_drmsd_nm`：预平衡姿态守卫（推荐 0.14） | 不设=关 | 用真机轨迹测过拒/放 |
| `tools/pocket_water/`（搬水 MC 原型，离线） | 不接管线 | 3 条单测；纯水盒 P(N) 对 MD **未做** |
| `stage2_lcr_relay`：stage 2 同窗段间接力（`ibs_engine.run_all_windows` 的 `window_start_positions` / `save_window_final_state`，`abfe_pipeline._run_stage2_rounds` 里取上一段 `dual_window_{w}_vdw_final_state.npz`） | 不设=关；**别开**（§2b） | 4 条单测（`tests/test_stage2_lcr_relay_2026_10_02.py`）；真机 L27：均值移 −26 kJ |
| 诊断脚本 `tools/diagnostics/{pocket_water_s1_probe,pocket_void_probe,w34_water_tracking,decharging_group_attribution,stage2_basin_features}.py` | 离线 | — |

整批推荐：标准 `_lcr` + `stage1_remd_equilibration_steps=125000`，不开接力。

---

## 4. 未结问题（按优先级）

1. **整批 fragment-opt**：用户手里有整批命令（`run_<sys>_lcr.zsh`）。跑前把 `stage1_remd_equilibration_steps: 125000` 加进各 `configs/*_lcr.json`（runner 脚本把 config 路径写死，改 config 或重新生成都要先问用户）。**不开接力。**
   已有基线：`runs_lcr/` 下 HSP90 L3/L26/L27/L30；ΔG_bind rep SD 1.7–3.8 kcal，论文 ~1 kcal。
2. **stage 2 精度（最大头，未解决）**：复合物 stage 2 rep 间 SD 11–13 kJ，集中在 w0–w3（λ_vdw≥0.2）。L3 的机制是盖子螺旋相对 Asp93 双稳、每段 0.5 ns 内不换盆（§2）；L27 未做同样归因。
   试过且无效 / 有害：加硬 Boresch（不能修双盆、有副作用，用户结论「不放大」）、接力（均值移 −26 kJ）。
   没试：每段加长（例如 1–2 ns，只增加 `n_steps_per_window`）、HREX/窗口间交换、增加 rep 数。
3. **接力为什么移 26 kJ**（§2b）：如果要继续用接力，必须先搞清。最便宜的查法：对 L27 接力组逐段（`vanishing`、`vanishing_2`…）看 w1/w3 的 ΔF 是否随段号单调漂移（漂移 ⟹ 系统在持续松弛 / 偏离 f_k），以及 ligand–口袋距离、口袋水数随段号的变化（`tools/diagnostics/stage2_basin_features.py`）。
4. **溶剂 stage 1 散布变大**（0.3→2.0 kJ，开平衡段后）：看 ~59 那个 rep 的配体扭转/分子内氢键是否和其他 rep 不同。小问题（~0.4 kcal）。
5. **「随机落盆」自动诊断**（同窗各段 ΔF 成簇、段内不换簇 ⟹ 标记），在所有 fragment-opt 体系上扫，看多普遍。
6. **我们 vs 论文的系统差**：逐体系比（同拓扑、`free_energy_data/abfe.csv` 有逐 rep 和分项 `comp_vdw/comp_coul/lig_*`），先看差是否系统性、出在哪一段。注意论文 4 fs/HMR，我们 2 fs。
7. 偏负（abfe-benchmark）不再追（10-02 用户定）；口袋水 v2 S3 以后暂缓；STATUS.md 未登记本轮数字（由维护者定）。

---

## 5. 坑（别再踩）

- **复现模式锚点从输入结构单帧选**，单帧擦边过关的近共线锚点会一路漏进生产（brd4 L2 旧锚点：输入帧 145.7°/30.2°，MD 均值 157°/23°、θB 低到 2°）。第一批 `runs_lcr_newanchor*` 因此**作废**（实际还是旧锚点）。
- 顶层 `rep*.log` 有 stdout 缓冲，看着停了不代表停了；进度看各 rep 目录下的 `pipeline.log`（溶剂腿在 `solvent_leg/pipeline.log`）。
- 在同一终端会话里 Ctrl+C 会把 nohup 起的 run 一起 SIGTERM（09-29 kang1000 第一次就这么死的）。
- **给用户的运行命令**：用 nohup，**不要 tmux**；同体系多个 rep 默认**串行**（`nohup zsh -c "for r in 1 2 3; do ...; done" &`），用户 10-02 两次纠正过。
- runner 脚本 `run_<sys>_lcr.zsh` 的 config 路径写死；跑别的 config 要直接调 `runabfe.py --config ... --gro ... --top ... --ligand LIG --preset production --output ...`。
- 坐标旁车 `dual_window_*_coords.npz` **不含水**；`idx` 每 5 帧一个。
- 蛋白跨盒时整体 CA 对齐会造出 1+ nm 的假位移；按受体锚点局部坐标系或口袋 CA 对齐。
- 生产 Boresch（source=simple）的平衡值是预平衡**系综均值**，不是末帧。
- fragment-opt 拓扑是 HMR（`#define HEAVY_H`）；09-28 修过溶剂腿配体质量（另一会话）。
- benchmark 两棵树当实验数据只读；新跑的输出放新目录。

---

## 6. 10-03 增补（接手会话 10-02 晚 ~ 10-03）

**做了什么**
- 59 份 `fragment-opt-abfe-benchmark/openmmIBS/configs/*_lcr.json` 加了 `"stage1_remd_equilibration_steps": 125000` + 一行 `_comment_s1eq`（改前确认都没这个键、都没开接力）。
  ⚠️ 直接改的 config；用 `make_configs.py` 重新生成会丢这个键。`runs_lcr/` 已有目录不受影响（runner 遇到非空输出目录拒跑）。
- 跑了 PWWP1 ligand-9 / ligand-10 各 3 rep（两个配体都中性，按 `ligand.hmr.itp` 电荷求和核过；论文对它们也没加 Rocklin）：

  | | 我们 3 rep | 均值 ± SD | 论文 5 rep | 实验 | rep 间 SD（kJ）：复合物 s1 / s2，溶剂 s1 / s2 |
  |---|---|---|---|---|---|
  | PWWP1 L9 | −6.92 / −8.98 / −7.89 | −7.93 ± 1.03 | −6.80 ± 0.34 | −6.35 | **4.9** / 1.8，0.5 / 1.1 |
  | PWWP1 L10 | −5.60 / −7.79 / −7.23 | −6.87 ± 1.14 | −6.12 ± 0.49 | −6.47 | **3.4** / 1.0，0.5 / 1.6 |

  ⟹ 复合物 stage 2 的大散布**只在 HSP90 上**（更像盖子螺旋这类 HSP90 特有的慢自由度）；PWWP1 上的大头是复合物 stage 1（去电荷），形状同 HSP90 L3 加平衡段前。
  均值比论文负 0.75–1.1 kcal，n=3 下 t ≈ 1.2–1.8，还不能说是系统差。
- 去电荷现状（PWWP1 L9 rep1）：12 个 λ 态 REMD、每 500 步交换、2 fs；平衡 125000 步（250 ps，~6.5 min）+ 生产 250000 步（0.5 ns/态，~12 min）；交换接受率均值 ~0.68、最差一对（4↔5）0.62。论文每态 20 ns = 我们生产段的 40×。
- **新键 `stage1_n_steps_per_window`**（CHANGELOG 10-03）：去电荷生产段每态步数，不设 = 沿用 `n_steps_per_window`（后者同时是 stage 2 LCR 每段长度）；只进 decharging 指纹。测试 4 条。
- **在跑**：PWWP1 L9 `runs_lcr_s1long/`（config `configs/PWWP1_ligand-9_lcr_s1long.json`：平衡 250000 步 = 0.5 ns + 生产 1000000 步 = 2 ns/态，stage 2 不变、不开接力），3 rep 串行，每 rep 约 3.5 h。
  **判据**：复合物 stage 1 SD 能否明显低于 4.9 kJ（TODO `FRAG-S1LONG-01`）。

**同日另外两件（不是本会话做的）**
- abfe-benchmark 三方法主线存档 `abfe-benchmark/archive_20261003/`：7 体系，IBS lcr / OpenFE / FEP suite（不含 LR）MSE −5.60 / −5.57 / −4.30；jnk1 / p38 / thrombin 10-03 决定不再做。已登记进 STATUS「当前现状」。
- 核心计算准确性复查 [archive/CODE_REVIEW_2026-10-03_accuracy.md](archive/CODE_REVIEW_2026-10-03_accuracy.md) + 设计提案 [design/PROPOSAL_stage2_accuracy_2026-10-03.md](design/PROPOSAL_stage2_accuracy_2026-10-03.md)（未实施）；
  与 §4 直接相关的两条：**共同漏采的盆地能带着 `CONVERGED` 通过**（就是 §2 的 HSP90 双盆形状，TODO `COVERAGE-1003`）、**接力段仍按独立种子计数**（TODO `RELAY-01`）。另外 burn-in 实际跑在 1 fs（25000 步 = 25 ps，不是 50 ps），判读「burn-in 够不够」时要按这个算。

**§4 的变化**：第 1 条（整批）配置已就位、等 `FRAG-S1LONG-01` 结论决定 stage 1 时长；第 2 条已知 stage 2 大散布是 HSP90 特有；第 7 条「STATUS 未登记本轮数字」10-03 已登记（标为登记、不可引用）。

