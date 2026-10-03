# 核心代码全量静态复查（2026-09-29/30，Opus 重审）

> **一次性审核的原始记录，不是待办清单。** 起因：`CODE_REVIEW_2026-09-29.md` 由低端模型完成且大面积未读，
> 本轮用 Opus 5.5 对生产代码（除 `refactor/`、`attic/`）**逐行全文**重读，并对全部 S2 做了对抗式核实。
> 只读：没改生产代码、没跑 pytest、没碰 GPU。唯一跑过的是两个读 benchmark `.npy` 的纯 numpy 脚本和一个
> 调分窗纯函数的合成数据脚本。HEAD `e576a54` + 工作树未提交 diff。

## 0. 方法与覆盖

- 15 个审查代理分段全文读（每段 ≤ ~7k 行，越界读调用方核实参），共用规则：只报可触发的问题、带触发路径、
  数值协议先对照 `/home/ruigengji/openfe`、已知/已否决项不重报、先对 6 份既有审查/TODO 文档去重。
- 4 个核实代理对全部 13 条 S2（+ 2 条追加项）做**反方证伪**：可达性（尤其 LCR 开时）、生产 config 是否触发、是否已登记、后果是否夸大。
- **本文 §1 只列核实过的条目；§2 是审查代理报的 S3，未经对抗核实。**

| 分段 | 行范围 | 覆盖 |
|---|---|---|
| A1–A4 | `ibs_engine.py` 1–28056 | 全文 |
| B1–B3 | `abfe_pipeline.py` 1–20462 | 全文 |
| C1–C2 | `abfe_core.py` 1–13967 | 全文 |
| D1–D2 | `abfe_preoptimizer.py` 1–13794 | 全文 |
| E | `runabfe.py` 1–9263 + 工作树 diff | 全文 |
| F | `outer_lambda_neural_basis.py` | 全文 |
| G | `free_energy_engine` / `abfe_diagnostics` / `apbs_correction` / `rbfe_core` / `rbfe_pipeline` / `runrbfe` | 全文 |
| H | `stage2_rounds*` / `stage2_ab_report` / `multi_segment_analysis` / `step_guard` / `lambda_path_versions` / `tools/pocket_water/*` | 全文 |

**总评：没有 S1。** 默认路径与 LCR 主线上没有发现会静默算错 ΔG 的问题；热力学循环符号/单位、Boresch 解析式（与 OpenFE 逐项一致）、
`lambda_boresch_scale` 在全部生产 Context 上的设定、pocket_water 搬水 MC 的细致平衡都核过是对的。13 条原报 S2 核实后只剩 3 条 S2，其中 **1 条在 LCR 主线上**。

---

## 1. 核实过的条目

### 1.1 影响 LCR 主线

#### [S2] LCR-GAUGE：第 2 轮起补段的固定 f_k 中心用了物理口径（含逐态 LJ 尾项常数 c_k），不是采样口径 — CONFIRMED
- 位置：`stage2_rounds_evidence.py:51-58`（用 `energies.npy` 解合并 F_k）→ `stage2_rounds.py:319` `_next_center` 直接作下一轮固定 f_k。
- 事实：`energies.npy` = `sampling_states` + c_k，c_k 为逐态严格常数（cmet_ligand1/rep1 实测逐帧 std ≤ 2e-13），每窗首末差
  **1.30 kT（复合物腿）/ 1.19 kT（溶剂腿）**，随 λ 线性。账本 `merged_f_k_kJ` 与目标口径 F 差 ≤ 6e-11、与采样口径 F 差 ~3 kJ/mol ⟹ 施加的确是目标口径。
- 后果：每个补段等于带了 ~1.3 kT 线性倾斜（偏向 λ 小端），平衡占据末/首比多 3.3–3.7×。设计 `PROPOSAL_stage2_lcr_rounds_2026-09-24.md` §11（约 310 行）
  标定规则时用的是 `sampling_states`，实现与标定不一致；"第 2 轮起不倾斜"的前提不成立，可能多补一轮或到轮数上限仍 PROVISIONAL。
  信赖域投影同样把目标口径 f 与采样口径锚点相减，末态吃掉 ~43% 半径（cmet rep1 本次未触发投影，这半条潜伏）。
- **ΔG 不受影响**（MBAR 对各段 f 的常数平移不敏感，只影响补段帧分配）。
- 修法方向：目标用 `sampling_states` 另解一份中心，或逐态减 c_k。
- 与 A2 "collect_energies 与 `_state_expr` 逐项一致"不矛盾：那说的是 `sampling_states` 与偏置表达式一致。
- 证据脚本：本轮 scratchpad `v14_gauge_probe.py`（未入库）。

#### [S3] BURNIN-1FS：新窗口/新 LCR 段的爬坡、LEARN/VALIDATE、固定 f_k burn-in 实际跑在 1 fs — CONFIRMED
- 位置：`ibs_engine.py:16395`（`if not is_resumed_ibs:` 内设 1 fs）… 16961（固定 f_k burn-in）… `19006` 才恢复 `original_dt`。
- 后果：LCR 的 25000 步 = **25 ps**（设计 §244/§467 与代码注释写 50 ps），每块 5 ps 非 10 ps；`cmet_ligand1_lcr_burnin.json` 的 150000 步 = 150 ps。
  中断后 resume 的段走 2 fs ⟹ 同一 config 下 burn-in 物理时长不一致。无证据表明有意（1 fs 注释只针对爬坡；恢复点是已删"生产前卸压"时代遗留位置）。
- ΔG 不直接受影响，但**影响"burn-in 够不够"的判读**（HSP90 L3 归因、LCR burn-in 实验）。
- 顺带：`CODE_REVIEW_2026-09-26` QUALITY-03 的"5000 步 = 10 ps"、§6 的"25000 步 = 50 ps"对新窗口都应减半。

#### [S3] PME-RUNTIME-ID：PME u_kn 缓存 meta 的 `runtime`（python/内核/openmm/pymbar 版本串）参与兼容比较 — CONFIRMED
- 位置：`abfe_pipeline.py:1523-1528`（写）、`1608-1614`（比较只 pop `code_sha256`）。stage1 decharging 默认 pme 路径，与 LCR 开关无关。
- 实测 `runs_lcr` 各 meta 已有三种内核串（211.16.1 / 211.49.1 / 211.58.1）⟹ 换节点必失配。
- 后果轻：只在 stage1 结果缓存也未命中时走到；代价是 u_kn CPU 重算（400 帧×8 态 ~46 s），不重跑 REMD，ΔG 不变；日志误报"模型版本不兼容"。
- 又一处"非用户输入进身份"的实例（同文件 1603-1607 注释描述的正是当年摘掉 `code_sha256` 的同一形状，漏了隔壁字段）。

#### [S3] REFINE-EXTRA-LAYER：`stage2_refine_extra_points_per_segment` 被划进 preopt 派生层 — CONFIRMED
- 位置：`abfe_pipeline.py:3273`、`17770-17865`；`abfe_preoptimizer.py:12731`。resume 改此键 → 离线重算用旧 pilot、新值不生效、写回新指纹。
- 119 份生产 config 无一设此键；LCR 下只影响 TI 种子与信赖域半径所用的 pilot dU/dλ。

#### [S3] BORESCH-K-RESUME：`--resume` 命中 `boresch_{source}.json` 时新键 `boresch_fixed_k_angle_kj_per_mol_rad2`（工作树新增）不生效 — PLAUSIBLE
- 位置：`runabfe.py:4487-4514` 缓存命中先 return，覆盖逻辑在 `4697-4709`。K 进 stage/顶层指纹，下游无别的缓存层兜 ⟹ 会原样返回旧 ΔG；删键 resume 反向同理。非 openfe_fixed 档设此键被静默忽略。
- 降级理由：唯一真实用法（`kang1000`）是新目录不带 `--resume`；CHANGELOG:64（09-24）已把"resume 沿用落盘 Boresch 参数"登记为有意约定（针对 anchor_mode）。
- **使用约定**：换 K 实验必须新开输出目录。

### 1.2 只影响旧自治控制器 / 复现 / 实验路径（LCR 开时不可达）

| 编号 | 判定 | 严重度 | 摘要 |
|---|---|---|---|
| FINAL-ANALYZE-REWINDOW | CONFIRMED | **S2** | 自治控制器收尾全路径 ANALYZE（`abfe_pipeline.py:13629-13647`）与循环内 ANALYZE（13211-13263）只解基准段+数字段，不读 rewindow 台账；`_solve_with_rewindow_children` 只在两个 rewindow 执行器内部用；下游三处 stage2 重赋值在控制器开时全关 ⟹ 最终 `stage2_vanishing.json` 不含子系综，**rewindow 修复被丢、结果退回修复前**（父窗原数据合法，不是"算错"）；父窗原本被跳过时整条腿 ANALYSIS_INCOMPLETE。代码默认 config 可达；现有 23 份 benchmark config（复现/LCR）不可达。未登记。 |
| PROVISIONAL-WASH | CONFIRMED | **S2** | 临时生产窗口经补帧（`skip_frozen_revalidation`）或 resume 复验（17615 抢在 `provisional_authorized` 之前）被洗成 `converged`/`verified`，`warmup_failure.json` 被删，最终 `provisional_production_windows` 与 9224 的 WARN 消失；只剩 `stage2_autonomous_history.json`/日志有痕迹、无下游读。位置 `ibs_engine.py:16163/16796/17615/18597/18638/20078`。ΔG 数值不变，证据标签失真。LCR 固定 f_k 跳过验证循环，不受影响。未登记。 |
| RESCUE-PROTO-NONE | PLAUSIBLE | S3 | `abfe_pipeline.py:6929` 对 `"vanishing_rescue"` 取协议键得 None，`ibs_engine.py:7857` None 即放行；上层无清理/作废旧产物机制 ⟹ 同目录改哈密顿量键再 resume（前一跑为旧控制器）会把旧协议 rewindow 子窗/`vanishing_N` 段合进新 ΔG。合并加载不传 `expected_window_protocol` 那半已是 REVIEW-0926 #1，但范围应从 LCR 扩到所有走 `_load_ibs_window_outputs_merged` 的合并；None 根因未登记。 |
| RECAL-FK-GAUGE | CONFIRMED | S3 | `recalibrate_f_k_from_production`（`ibs_engine.py:22165`）同 LCR-GAUGE 形状：用 energies 口径解 f_k，相邻态差 0.647 kJ/mol 已超 0.5 采纳阈值 ⟹ 即使 f_k 已准也会被换成倾斜种子。`ibs_engine.py:9404-9413` 修过同一个坑。原报告引作旁证的 `heldout_candidate_verdict` 全仓零调用。口径问题，**不涉及符号**。 |
| INSERT-LAMBDA-ARGS | PLAUSIBLE | S3 | 自治 INSERT_LAMBDA（`abfe_pipeline.py:12964-12970`）漏传 `partition_criterion`/`pilot_metric_g`。实际默认 arclength（18134，`abfe_config.json` 不自动加载），state_count 本就按弧长；只有"非 LCR + 显式 metric_integral"会出差别，现无此 config。`and not _lcr_on`（18714）使 LCR 不进。 |
| SPLIT-TAIL-ASSERT | PLAUSIBLE | S3 | SPLIT_TAIL_WINDOW 缩跨度断言比尾段整体最大 ∫g（`abfe_preoptimizer.py:9929-9948`）。合成 g 上 965 次可行重分拒 380 次（复现）。但行内注释**明文选择**了"尾段整体"、否定了"比失败窗口自身"——是设计取舍，需维护者拍板，不是写错。 |
| OUTER-SCORE-ID | PLAUSIBLE | S3 | `sampling_score_sha256` 含绝对路径（model/atom_indices/manifest/plugin）与插件 `.so` sha256；EXP-033 自动重训产物内容 sha 也进 score。开关开时 score 变化确实经 7764→16058→`ibs_engine.py:10462` raise→`abfe_pipeline.py:298` HALT。8 份 `*_lcr*.json` 全关此开关，只有 `cyclod_ligand1_outer.json` 开。`TODO_closed.md:51` 的 `cyclod_ligand1_outer/rep1` 真崩过此异常，当年"逐 rep 重训"的解释与 rep1 独立目录对不上，日志已被 09-23 覆盖无法追溯。重训逐位可复现性未实测；重训代价 ~30 s。 |
| PLATFORM-ID | PLAUSIBLE（潜伏） | S3 | REMD 采样指纹含 `platform_name`、`max_resident_contexts`（`abfe_pipeline.py:2778-2783`），窗口采样身份含带设备号的 platform（`ibs_engine.py:730`）；`charging_max_resident_contexts` 还经 kwargs 进 stage 协议键与窗口身份未被剥。现所有 config `"CUDA"`、无 max_resident 键，`run.zsh` 用 `CUDA_VISIBLE_DEVICES` ⟹ 当前不触发；一旦改动，stage1 结果与全部 stage2 窗口一起失配。 |
| SHADOW-BRIDGE-BORESCH | CONFIRMED | S3 | Shadow-Bridge REMD 副本与评估 Context 从不设 `lambda_boresch_scale`（默认 0.0；`ibs_engine.py:26790-26808, 26878`）⟹ 复合物腿在 Boresch 关着时采样。仅实验性 `--decharge-method shadow_ibs`，无 config 使用。**其余所有生产 Context（stage2 窗口/LCR 段/探针/端点/u_kn/stage1 REMD/再平衡/预平衡）逐个核过，均正确。** |

---

## 2. 审查代理报的 S3（未经对抗核实）

**LCR 相关**
- LCR 某窗所有段 FAILED 后永不重新计划且 resume 粘住；溶剂腿第一轮每窗 1 段，一次瞬时 OpenMMException 即永久卡死 stage2（A10"不重试"前提是三起点都崩，对单段不成立）。(B3)
- LCR 第 1 段共用默认目录且强制 resume：同目录先跑过旧控制器留下 `failed`/`calibrated_validation_failed` 时，在固定 f_k 判定前抛 terminal（`ibs_engine.py:16075`、`abfe_pipeline.py:14844-14853`）。fail-loud，潜伏。(A3)
- `lambda_path_versions.resolve_path`/`init_version` 已有 v1 时无视新 λ 表/分窗，LCR 账本布局校验也比旧布局 ⟹ 同目录改 λ 配置静默沿用旧路径。(H)
- `pilot_trust_radius_kJ` 梯形误差传播漏算内部节点共用，σ 最多低估 √2（通常被 3 kT 下限盖住）。(H)
- resume 改配置后 LCR 账本 `thresholds` 仍抄上一轮。仅审计记录。(H)
- `stage2_ab_report.py` 不认 LCR 产物，且会往 run checkpoints 写 `controller_comparison_manifest.json`，与 docstring"只读"不符。(H)

**Stage 1 / 预平衡 / 再平衡**
- REMD 开跑前不作废旧 `{stage}_sampling.meta.json`：X→改 Y resume 中途崩→改回 X，Y 下采的帧可能被当 X 的样本（λ 表/温度不同时静默错算）。需特定操作序列。(B1)
- 预平衡 checkpoint 续跑剩 0 步（或续跑段 < 4 block）时不做 NPT 均值盒缩放，退回 09-23 修掉的末帧盒（`abfe_pipeline.py:5243/5270`）。(B1)
- 再平衡 `rebalance_state.json` 缺、`rebalance.chk` 在时默认受信，跳过指纹门与 09-23 start_state 门（5457/5540）。(B1)
- `_rebalance_with_boresch` 把 `CUDA:N` 原样传 `getPlatformByName` ⟹ 必静默回退 CPU（5519）。(B1)
- 磁盘预检在全缓存命中的 resume 上也按全量估算（6370）。(B1)
- `n_equil_steps` 调大后 resume，预平衡按上次目标判已完成，只打 WARN（`runabfe.py:1271-1337`）。(E)

**其他**
- 复现模式左/右倾斜在 resume 后可能丢失（`ibs_engine.py:19013`，`not is_resumed_ibs`）。置信度中低。(A3)
- 旧控制器"全窗满额"判定漏算缺窗，可能以 `HALT_FRAMES_ADMISSION_CAP` 终止（`abfe_preoptimizer.py:5834-5851`）。(D1)
- 离线重算默认窗宽 4/6 vs 权威分窗 4/8，config 不显式给时白烧一轮 pilot（生产给了 4/5，不触发）。(D1)
- 多段窗口受门的 raw_*/top1pct/bias_to_signal 用各段 bias 直接拼接而非 `bias_mix`（`ibs_engine.py:23548`）；只进报告与旧控制器归因。(A4)
- 局部 MBAR 异常被 `continue` 的窗口不进 `skipped_windows`（23473/23860），`path_is_complete=True` 自相矛盾；`analysis_status` 兜住。(A4)
- 自治循环改布局却返回 `result=None` 时，调用方给旧布局 ΔG 盖新布局的 n_states/λ 指纹/path_version（`abfe_pipeline.py:13667`→18795/19477）。(B2)
- `--analyze` 对 `sampling_repair_policy` 字面比较，拒掉 `path_evolution_v1` 窗口并崩（`ibs_engine.py:726/745`）。(A1)
- `--openmm-cache-only` 需重估 Boresch 时仍读 `config.top`/gmx include（`runabfe.py:4519-4522`），cache-only 专用分支不可达。(E)
- 溶剂腿自动拾取 output 下 `ligand.xml`/`<resname>.xml` 但不进溶剂缓存身份（`runabfe.py:1472-1488`）。潜伏。(E)
- 带电配体溶剂腿用原始 top 拓扑配派生体系（含 co-ion）坐标（`runabfe.py:2694/8234`）。潜伏。(E)
- `validate-config` 把 `n_states_per_stage` 列为生效键，运行期只认 CLI（`abfe_diagnostics._EXTRA_CONSUMED_KEYS`、`runabfe.py:3603`）。置信度高。(G)
- Rocklin/APBS 修正只算复合物腿，DSC 一项（~−69 kJ/mol/e）不抵消；膜线暂停，潜伏。(G)
- 带电路线电荷容差 1e-3 e（上游门）vs 1e-6 e（`co_alchemical_charge_offset_plan`，`abfe_core.py:1972-1980`），预平衡后才崩。诊断开关下才可达。(C1)
- 拓扑解析展开未激活 `#ifdef` 分支里的 `#include`（`abfe_core.py:2999-3017`）。(C1)
- shadow_ibs 配体内部库仑 1.0 nm 硬截断裸 1/r（`ibs_engine.py:6315/6451`）。实验路线。(C2)
- `softcore_power_lj` 的 (a,b) 文档与代码相反；默认 [1,1] 无影响。(C2)
- `refine-lambda-path` 子命令写出的 preopt 缺 `protocol_key`/`path_diagnostics`，下次 resume 必重跑 pilot，原 pilot 只剩 `.bak`（`abfe_preoptimizer.py:11580-11598`）。(D2)
- `compute_2d_metric_grid` 每行从近解耦跳回 λ_vdw=1 只给 500 步（`abfe_preoptimizer.py:13360-13371`），与 09-10 修掉的形状相同；仅 `--decoupling 2d_geodesic`。(D2)
- `NeuralPathSafety` 生产未接线（只进哈希）；`atom_indices_path` 格式与读取器不符；`IBSSamplerNeuralPathAdapter` 与现行 `IBSSampler` 不兼容；CLI `label-existing`/`label-trajectory` 依赖已不存在的模块永远 exit 2。(F)
- `_recover_stage_identity_from_convergence` λ 守卫只比 `lambdas_vdw`（`ibs_engine.py:7602-7611`），仅 09-14 前旧 state 迁移。(A2)

**死代码（零生产调用）**：`IBSSampler.update_weights` 及 `_solve_tmbar_and_recenter`/`IBS_TMBAR_TRUST_*`/`apply_learning_rate_penalty`（`eta_penalty` 恒 1.0 却被持久化校验）、`IBS_EARLY_PROBE_TRIGGER_ENABLED`；
`_check_equilibration_convergence`、`_quantize_positions_for_identity`、`_q8`；`TraditionalABFEPipeline.pre_equilibration_identity_fingerprint`（引用未设置的 `self.pressure`）；
`update_boresch_from_last_frame`、`run_full_abfe_loop`/`SolventLegRunner`、`_apply_already_good_repairs`、`_diagnose_and_repair_*`、`_is_overlap_failure`、`_remap_*`、`_merge_overlapping_*`、`_load_window_bias_warmup_status`；
`SurrogateSystemBuilder`、`GhostIonHandler`、`OrbMMHybridFactory`（orb 档按约定保留）、`auto_select_boresch_anchors_rmsf`、`ChunkedMBARAnalyzer`、`_select_env_indices_from_mdtraj_frame`；PME 自能修正分支（生产硬编码 False）与 `pme_offset_charge_square_sum`。

---

## 3. 对既有审查文档的异议

- **MODULE_AUDIT_2026-09-28 §2.2**："`_quantize_positions_for_identity`/`_q8` 口径一致"——两函数零调用方，不能作"已查干净"证据。
- **MODULE_AUDIT_2026-09-28 TG-1**：前提错。PME 自能修正生产硬编码 False（`ibs_engine.py:27215`）且已因物理错误撤销（`abfe_core.py:8480`），该支是死代码；`pme_offset_charge_square_sum` 零调用；`lambda_endpoint_diagnostics` 并非零引用。应删/标死代码，不是补测试。**TG-3** 称 `delta_u_distribution_diagnostics`、`synthetic_mbar_u_kn` 零引用，也不对。
- **PERF_REVIEW_2026-09-28 §5**：把协议键哈希整体列为干净，漏了 `_pme_u_kn_meta_payload` 的 `runtime`（§1.1 PME-RUNTIME-ID）。
- **CODE_REVIEW_2026-09-26 QUALITY-03 / §6**：步数→ps 换算按 2 fs，新窗口实际 1 fs（§1.1 BURNIN-1FS）。
- **CODE_REVIEW_2026-09-26 REVIEW-0926 #1**：范围写窄，应扩到所有合并加载（§1.2 RESCUE-PROTO-NONE）。
- **CODE_REVIEW_2026-09-29**：M2 与 09-26 #5（非有限 σ_MBAR 使补采失效，TODO.md:106）已被工作树 `stage2_rounds.py` 未提交 diff 解决（补段门改为 `σ_jk > σ_tol`）；工作树 `ibs_engine.py` diff 删除重复 `n_eff_frame_set` 键 = M1，修复正确；L7（seed_id 跨窗复用）无害；H2、M3 的降级/驳回复核同意；09-26 #4（finalize 按 `window_index` 建表）仍未修。
- **CACHE-01 结案**："include 树一次都不会被解引用"对 `--openmm-cache-only` 重估 Boresch 路径不成立（§2 E 条）。
- `TODO_closed.md:51`（cyclod outer rep1 score mismatch 的归因）存疑，见 §1.2 OUTER-SCORE-ID；证据已被覆盖，不主张推翻。

## 4. 核过确认干净的高危段（摘要）

- 热力学循环：`compute_final_results` attach + decharge + vdw + release 的符号与单位，bind = solvent − complex，Boresch 项不重复扣；kT = R·T；runabfe 五条汇总路径的"已含/未含"开关一致。
- Boresch：解析释放式、固定力常数、锚点阈值、角"不平"判据、PBC 设置与 OpenFE 逐项一致；`lambda_boresch_scale` 在全部生产 Context 正确（仅 Shadow-Bridge 例外）。
- IBS：softmax 与偏置力表达式同号，f_k 取 TMBAR 物理 F_k 同号，无额外负号；`collect_energies` LRC/residual offset 与 `_state_expr` 逐项一致；v32（tmbar 批存未偏移能量）在位；LJ 解析尾项与 ACE 表达式一致；IBSBiasForce LSE 稳定。
- 缓存：`_resume_cached_window_gate_status` 17 道门 fail-closed；预平衡身份只含配体索引/T/P/步数/barostat；`_system_xml_hash`/`_topology_hash` 等恒 None；`_NON_IDENTITY_KWARGS` 摘键正确；preopt/λ/rewindow/seed 身份均为语义量。
- config：合并顺序 预设 → config 文件 → 显式 CLI → 默认值正确；`_path_evolution_kwargs` 全部有消费点。
- pocket_water：stage-1 接受式（体积比、Z 比、g(N)、被搬分子能量项相消）、stage-2 延迟接受、批内扫描、Haar 均匀取向、Kabsch 跟随使 V_R/V_S 常数、单位与 PBC 均正确；只支持 3 点水/HOH，TIP4P 会直接崩（不静默）。
- `multi_segment_analysis`、`step_guard`、LCR 执行层其余部分干净。

## 5. 建议优先级（仅建议，登记与否由用户定）

1. **LCR-GAUGE**（LCR 主线，修法明确、局部）。
2. **BURNIN-1FS**（影响 burn-in 时长判读与 LCR 实验设计；至少修正文档/注释口径，或把固定 f_k burn-in 挪到恢复 2 fs 之后）。
3. LCR 单段 FAILED 永久卡死（§2 第一条，建议先核实）。
4. 旧控制器两条 S2（FINAL-ANALYZE-REWINDOW、PROVISIONAL-WASH）——只在重启旧控制器前需要。
5. 缓存身份三条（PME-RUNTIME-ID、PLATFORM-ID、OUTER-SCORE-ID）：同一形状，可一次性按"只有用户输入才配做身份"清理。
