# 计划：stage 2 控制器重写（第一轮左中右、之后缺啥补啥）—— 2026-09-24

[设计文档索引](README.md) · 设计全文 → [PROPOSAL_stage2_lcr_rounds_2026-09-24.md](PROPOSAL_stage2_lcr_rounds_2026-09-24.md)（下文 §x 均指它）· [TODO](../TODO.md)

## 0. 状态

🟡 **第一期进行中**：第 1–4 步代码已落地（09-25，**开关 `stage2_lcr_rounds` 默认关，老路径逐字不变**，等大规模合并再切默认）；第 5 步（离线集成）部分覆盖，第 6 步（真机）未开始。待拍板见 §5。

**怎么开**：config 里 `stage2_lcr_rounds: true`，并且必须同时是冻结布局（`stage2_uniform_lambda_states` + `stage2_window_ranges`，即复现模式），否则启动时报错。可选计划参数 `stage2_lcr_{tilt_kT, trust_floor_kT, trust_min_ess, sigma_tol_kJ, max_rounds, burn_in_steps, zero_window_end_shift_kT}`，都不进身份。

本文是**执行计划**：做什么、按什么顺序、每步怎样算做完。为什么这样设计、数据依据，都在设计文档里，这里不重复。

---

## 1. 目标

用一个新控制器替掉 stage 2 现有的两套控制逻辑，同时解决复现批暴露的三个问题：

| # | 问题 | 新设计怎么解决 |
|---|---|---|
| 1 | f_k 在难窗口被占据反馈追飞（谱宽达 pilot 的 2–20 倍） | LEARN 退出主线；中心 = pilot 锚点，补段中心钉在 pilot ± 信赖域里 |
| 2 | 单 run 报的 σ 偏小（MBAR σ 低估 ~3.6 倍） | σ 加入段间 jackknife（与 rep 间 SD 中位比 1.02） |
| 3 | 「三段一致」被当成收敛 | 各段不同起点；收敛标签要求 ≥2 个不同起点 + σ_jk ≤ σ_tol |
| 7 | 固定节奏循环与控制器 5a 分支重复 | 控制器只剩「补一段」一种动作，一个纯函数决定 |

## 2. 已定的决策（09-24）

| 项 | 定了什么 | 出处 |
|---|---|---|
| 第一轮 | 每窗左/中/右三段，中心 = pilot 锚点，倾斜 2 kT（IBS 里再大会塌成单态） | §3.1 |
| λ=0 窗口 | 复合物腿中心末态上抬 2 kT（**暂定**，第一批后复核）；溶剂腿不抬；不做信赖域投影 | §3.1.1、§10-6 |
| 溶剂腿 | 第一轮每窗**只跑中段**；σ 用段内切三块 jackknife；CONVERGED 只要 1 个种子段；块间 σ 超标或缺帧才补（09-25；w0–2、w4 三段纯属浪费，w3 的灌水双稳态切块也抓得到） | §3.1.2、§11.5 |
| λ=0 诊断开关 | `stage2_lcr_zero_window_decoupled_seed`：补段起点改用 λ_vdw=0 平衡构型。**临时诊断，查清后连同代码删除** | §3.1.1 |
| 补段规则 | 缺帧（某态 ESS < 20）或段间不一致（σ_jk > max(2σ_MBAR, 1.5 kJ)、单段依赖、top-1% > 0.35）⟹ 补一段：**不倾斜、中心 = 合并 MBAR 的 F_k、换新起点**（09-25：倾斜只在第一轮用；线性倾斜在第二轮会压塌另一端） | §3.3、§11.3 |
| 补段长度 | 与第一轮等长（jackknife 要求各段可互换） | §12.9-1 |
| 数据流 | 第 r+1 轮怎么补来自第 r 轮后的合并结果；第 r+1 轮后的分析用第 1..r+1 轮**全部**段（相加不取代） | §12.9 |
| 段组织 | 每轮一个段（第 1 轮 3 个，之后每轮 1 个，只含被点名的窗口） | §12.8-3 |
| 三段都低 | 原因分不清（f 偏太远 / 构型慢模态 / λ 间距）⟹ 与其它补段同动作（重定中心 + 新起点）；到上限标 `SAMPLING_LIMITED_CAUSE_UNRESOLVED`，记 pilot 最长边与帧上相邻重叠。**插 λ（model B）只作预案，暂不自动执行**，等真实样本再定触发条件 | §3.3、§12.10 A13 |
| 边界情况 | 全段崩 ⟹ 不重试、INDETERMINATE；MBAR 解不出 ⟹ 按错误 fail-closed；没 pilot ⟹ fail-closed、不冷启动；补段后变差 ⟹ 照规则、记 `worsened`、永不丢段；resume 已有证据就用 | §12.10 |
| 信赖域 | 仅当合并证据弱（最小 ESS < 50）时，补段中心投影回 pilot ± max(3 kT, 3σ_pilot) | §3.5 |
| σ | σ_window = max(σ_MBAR, σ_jk, split-half) | §3.6 |
| σ_tol | 1.5 kJ/mol（19:16 批标定，膝点 1.0–1.5） | §11.2 |
| 轮数 | 最多 3 轮（含第一轮），每窗每轮最多补 1 段；**暂定** | §10-3 |
| 起点 | **预平衡只算一次**（这是我们和 OpenFE「算三遍」的根本区别）；**所有段从预平衡末帧起跑、只换速度种子**（照 OpenFE 同一 repeat 内的做法） | §10-2、§12.9-2 |
| burn-in | 先试 50 ps，**按 10 ps 分 5 块**记漂移（09-26：原来块长 = 总长只有 1 块，cmet L1 rep1 真机上看不出漂移），第一次真机再定（OpenFE 每 λ 1 ns） | §12.9-7 |
| 主线 | 新控制器进主线，而且**就是**控制器 | §4、§10-4 |
| 代码放哪 | `stage2_rounds.py`（只读决策）+ pipeline 执行层（落盘）；`Stage2RepairController` 整体退役 | §12.8-1 |
| 退役方式 | **两期：先关、后删**（TODO §4「旧修复机制只许关、不许删，等新路线跑通几次再说」） | §12.7 |
| 控制器硬规矩 | 先落账再采样（计划冻结，resume 不重新计划）；每轮完整落账 | §12.1 |

同日已落地的前置改动（与本计划独立）：软核 LJ 指数默认 (1,1)；Boresch 默认 openfe 路线（固定力常数 + openfe 锚点搜索）。见 [CHANGELOG](../CHANGELOG.md) 09-24。

## 3. 分期与步骤

### 第一期：新控制器接管主线（旧路径只关、不删）

| 步 | 内容 | 做完的判据 | 状态 |
|---|---|---|---|
| 1 | **纯函数层** `stage2_rounds.py`：倾斜、末态上抬、信赖域半径与投影、jackknife、逐窗标签、`first_round_specs`、`plan_next_round` | 测试全过；模块不 import OpenMM、不读写盘；常量与 `ibs_engine` 对账 | ✅ 09-24；09-25 按新决定改完：第二轮起不倾斜、单段依赖改按留一 ESS 塌陷判、病因标签（一边高 / 三段都低）、`start_id`→`seed_id`（起点统一是预平衡末帧）；27 条测试 |
| 2 | **引擎入口**（`run_all_windows`）：`fixed_f_k_by_window`（跳过 LEARN/VALIDATE → 爬坡 → burn-in（按 50 ps 分块记漂移）→ 生产）、段号进速度种子流（起点统一用预平衡末帧，不需要起点帧提供器）；~~升 `IBS_BIAS_PROTOCOL_VERSION`~~ → 改为只在开关打开时往 vanishing 指纹加 `stage2_lcr_rounds`（老路径缓存不作废） | 新入口单测；不传新参数时现有行为逐字不变（现有测试全绿）；种子逐段不同 | ✅ 09-25：`bias_status="fixed_f_k"`、`f_k_evidence_status=indeterminate`；burn-in 块诊断写进 convergence 的 `bias_warmup.fixed_f_k`；固定窗口不再套复现倾斜 |
| 3 ✅ 09-25 | **轮次账本 + 逐窗证据**：`checkpoints/stage2_rounds_ledger.json`（逐轮 specs / 状态 / 证据 / 标签 / 合并结果）；每段 `segment_spec.json`；证据构造器（合并求解 + 留一段 jackknife + 锚点与半径 + 第一轮三段各自的 ESS 模式；**只记录**：pilot 窗内最长边、帧上相邻态重叠 Σmin(w_i,w_{i+1})） | 账本读写与版本校验测试；证据构造器在 19:16 批真实产物上复现 §11.2 的 σ_jk | 🟡 证据数值层 `stage2_rounds_evidence.py`（MBAR 复用 `multi_segment_analysis`，top-1% 复用 `ibs_engine` 唯一实现）+ 账本在 pipeline；合成数据测试通过；19:16 批真实产物上 σ_jk 与 §11.2 重放逐位吻合（cmet L1 rep1/rep2 溶剂 w3 6.85/3.29、cyclod L3 rep1/rep3 复合物 w3 5.63/0.84，重放 6.8/3.3/5.6/0.8） |
| 4 ✅ 09-25 | **执行循环**（pipeline）：按 §12.1 主循环；接 `_finalize_with_indeterminate`；输出按 §12.6；开关切走旧路径（固定节奏循环、自治控制器） | resume 不重新计划（中断在任意一轮都恢复到同一状态）；部分段合并正确；某窗崩溃不连累其它窗；全量测试绿 | ✅ 假引擎测试覆盖：先落账再采、按窗口顺序、第二轮不倾斜且每轮一个段、resume 不重采不重算、中途崩只补缺的、段失败标 FAILED、全段崩不重试、没 pilot 报错、预算停、f_k 与账本不符拒收、λ=0 上抬只在复合物腿、布局变了拒用账本、收尾 σ_window 与 indeterminate；**全量绿，除 `test_fresh_clone_imports`（两个新模块待 git 跟踪）** |
| 4b | ~~插 λ~~ —— **推迟**（09-25）：预案写在设计 §3.3，等真实样本 | — | ⏸ |
| 5 | **离线集成**：两盆玩具体系上验证「三段困同一盆」vs「三段各在一盆」时 σ_jk 与标签的差别；19:16 批上端到端回放决策 | 两种情形标签不同且符合预期；回放的第二轮计划与 §11 重放结论一致 | ⬜ |
| 6 | **真机**：cmet L1 × 3 rep（溶剂 w3 是最坏窗口），再扩到整批 | 跑完不崩；rep 间 SD / 报出 σ 中位 ≈ 1；按 §10-6 复核 λ=0 上抬、按 §11.2 复核 σ_tol | ⬜ |

### 第二期：删旧（新控制器真机跑通若干次之后）

| 步 | 内容 |
|---|---|
| 7 | 删：复现固定节奏循环；`_decide_once` 与 `Stage2RepairController`；A 类动作执行分支（RUN_PRODUCTION / PROVISIONAL_PRODUCTION / RECALIBRATE_FK / PROBE_REANCHOR_EPOCH / CONTINUE_WARMUP / RELEARN_FK_EPOCH）；三份「下一段编号」；钉旧行为的约 20 个测试文件（及 `refactor/replay/shadow*/tests/` 镜像） |
| 8 | 顺带做 `TEST-02`（测试压缩押后到重构）；`refactor/replay/**` 回放链随 `_decide_once` 一起处理 |
| 9 | B 类（INSERT_LAMBDA / SPLIT_TAIL_WINDOW / IMMUTABLE_REWINDOW / bridge rescue / rewindow 账本）另开 TODO 退役 |

## 4. 缓存与身份

- ~~第 2 步升 `IBS_BIAS_PROTOCOL_VERSION`~~ —— **改（09-25）**：新控制器是开关、默认关，升全局版本会连带作废老路径缓存。改为开关打开时往 **vanishing** 的 stage 指纹加 `stage2_lcr_rounds=true`（`_scope_stage2_lcr_switch`）：老路径指纹逐位不变，新旧两条路也不会互用缓存；stage 1 指纹不受影响。
- ~~新增配置键（`stage2_lcr_*`）全部进 stage 指纹~~ —— **更正（09-25，设计 §12.10-C）**：多段 MBAR 对任何 f_k 采的段都成立，段是否可用只取决于哈密顿量和布局 ⟹ `stage2_lcr_*` 全是计划参数，✅ **不进身份**，加进 `_NON_IDENTITY_KWARGS`；每段实际用的值记在 `segment_spec.json`。原写法正是 09-11「阈值键进指纹作废全部缓存」的形状。
- 账本和 `segment_spec.json` 记的是**执行历史**，不进身份（「只有用户输入才配做身份」）。

## 5. 待拍板

1. **预算裁剪优先级**（B6）：⏸ 等真机看预算实情再定；过渡行为 = 预算不够就本轮不开、停，记 `GLOBAL_BUDGET_EXHAUSTED`。
6. **split-half 进标签**（设计 §12.10 A14，09-26 真机暴露）：⬜ 已记，待实现。提议「单窗漂移 > 2×2σ 且总量漂移 > 2×2σ ⟹ 不许 CONVERGED、补一段」。
~~2. 第 2 轮起的段组织~~ —— ✅ 09-25 定：**每轮一个段**（设计 §12.8-3）。
~~3. 第二期触发~~ —— ✅ 09-25 定：新代码真机跑通就删。
~~4. 起点帧~~ —— ✅ 09-25 定：照 OpenFE，全部从预平衡末帧起跑、只换速度种子。
~~5. 补段后反而更差~~ —— ✅ 09-25 定（设计 §12.10 A12）：是被藏住的方差露出来，按现有规则继续；记 `worsened` + 归因（段内漂移 / 段内平稳）；**永远不因结果变差丢段**。

## 6. 风险

- **单 run 看不到 rep 级方差**：同一条预平衡里慢侧链态基本不变（cmet L1 数据），σ_jk 抓不到那部分；按「只算一次」的定位，由独立 rep 去量。
- **λ=0 窗口是构型覆盖问题**：上抬 2 kT 只把 f 放对，治不了帧里没有的构型（代码里已写明原因，`ibs_engine.py:~12872`）。
- **改动面大**：`_decide_once` 周边的工具（`stage2_ab_report.py`、控制器 CLI、`tools/audit/static_controller_*`）第一期不动但会过时。
- 所有阈值都只在 19:16 一批上标定过。

## 7. 不在本计划

- stage 1 的多链 σ（同一类问题，stage 2 落地后另开）。
- 增强采样（HREX / REST）：REMD 线 08-31 叫停。
- ~~改布局的动作~~ —— 09-25 起插 λ（model B）已纳入（第 4b 步）；`IMMUTABLE_REWINDOW` / bridge rescue 仍不在本计划。
