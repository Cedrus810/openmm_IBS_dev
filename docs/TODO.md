# 当前行动清单

[项目入口](../README.md) · [文档导航](README.md) · [当前科学状态](STATUS.md) · [变更记录](CHANGELOG.md)

> **本文是本仓库唯一的待办清单**（2026-09-12 整理）。一条待办只在这里出现一次；
> 专题文档负责讲"为什么"，本文只负责"还欠什么、欠在哪一行"。
>
> **项目处于开发最末期**，不是在筹备一次首发：核心流水线、Stage-2 自治控制器
> （2026-09-12 17:22 首次独立跑完整条）、闭式重训都已跑通。剩下的是收尾 ——
> 所以本文里没有"要建什么新能力"，只有"哪一处还没收干净"。
> [RELEASE_READINESS](RELEASE_READINESS_2026-08-31.md) 里的「预览版前」「首发支持范围」
> 是 08-31 的措辞，**按末期收尾读**，别当成一次尚未开始的发布筹备。
>
> **已关闭的条目不留在这里**，整段移进 `archive/`：
>
> | 归档 | 内容 |
> |---|---|
> | [archive/TODO_closed_2026-09-09.md](archive/TODO_closed_2026-09-09.md) | `MIGRATE-01` / `PBC-01` / `XFAIL-01` / `XFAIL-02` / `CACHE-01` / `CFG-01` 六条已关闭缺陷的完整记录 |
> | [archive/TODO_2026-08-06_unreconciled.md](archive/TODO_2026-08-06_unreconciled.md) | 2026-08-06 的主表（1350 行，`ATT-xx`/`MEM-xx`/`P0-9~13` 编号）。**那里面的 `- [ ]` 只表示"当时未完成"**，要人逐条对账才能重新变成待办 |
>
> ⚠️ 不要按编号跨文档机械对账：`P1-19` 在 08-29 那份交接里是"v4 charging 接缝内静电失配"（已修），
> 在 08-06 主表里是"per-window σ 系统性低估 2–4 倍"（未完成）。**同名不同义。**

## 优先级

| | 在推 | 内容 |
|---|---|---|
| **1** | ✅ 验收已达成；`decide()` 重构中 | [Stage-2 自治闭环](#1-stage-2-自治闭环) —— 5 条缺口；另有 8 个测试因重构预期性红着，**别修** |
| **2** | 排在后面 | [local-residual / EXP-033](#2-local-residual--exp-033) —— P2 是唯一还开着的，P1 真机没跑过 |
| **3** | 收尾 | [发布工程门](#3-发布工程门) —— 判据是 clone 下来 import 得动、跑得动 |
| **4** | 不修 | [已定位、判定不改](#4-已定位判定不改) —— `AUDIT-01`~`07`，理由别重新论证 |
| **5** | 暂停 | [膜受体–配体路线](#5-膜受体配体路线暂停) —— 停在 C4，当前主线是可溶体系 |
| **6** | 条件性阻塞 | [PHY-03](#6-phy-03带电路线的条件性阻塞) —— 带电配体路线 |

---

## 1. Stage-2 自治闭环

> ### ✅ 验收口径已首次达成：2026-09-12 17:22
>
> 自治控制器独立走完整条 Stage-2（六窗全部 `ANALYSIS_ELIGIBLE` → `DONE`），
> 体系 `cyclod_ligand2/rep1`（环糊精主客体，可溶），
> ΔG_bind = **−3.48 ± 0.47 kcal/mol** vs 实验 −4.04，差 **1.19σ**。
>
> **设计、实证与十条陷阱全部在
> [STAGE2_CONTROLLER_DESIGN_2026-09-12.md](STAGE2_CONTROLLER_DESIGN_2026-09-12.md)
> —— 接手先读那份，别重推。** 本节只登记它 §7 列的缺口。
>
> ⚠️ 一次跑通 ≠ 通用。这是**单体系单 run**，没有独立重复。

### 剩余缺口（源：控制器 design §7）

- [ ] **S2-A 累计 f_k 残差门对多段窗口结构性不适用。**
  多段窗口的帧采自两份不同偏置，`_load_ibs_window_outputs_merged` 因此显式
  `base.pop("f_k")`。⟹ 循环每用换 Epoch 修好一个窗口，那个窗口就**永久失去**
  这项证据（那一跑 win0–3 已全丢）。逐段残差不与合并后的 ΔF 直接望远镜相消，
  **需要单独定口径**。
- [ ] **S2-B 续验路径上验证要求随预算膨胀。**
  `ibs_engine.py:15575` 续验时 `validation_attempt_budget_steps = full_bias_step_budget`，
  于是 `minimum_complete_validation_frames = max(200, budget/stride)`：
  **给的预算越多、要求的帧数越高**，可达性判据的第 2 档因此失效。
  正解是把「完整性要求」与「去相关要求」解耦 —— 200 是统计目标，不该随预算浮动。
- [ ] **S2-C 边际增长判据需要至少两段历史**，只有一段的窗口用不上。
- [ ] **S2-D 代码仍散在三个文件** —— `abfe_preoptimizer`（`decide` / `read_aggregated`）、
  `abfe_pipeline`（执行器 + 若干 helper）、`ibs_engine`（两个纯函数）。
  设计要求是**包在一起**，尚未收拢。
  🚧 **2026-09-12 维护者正在做这一条**（下面那 8 个红的测试就是它造成的）。
> ### 🚧 `decide()` / join-λ 支撑判据**正在重构中** —— 下面这 8 个红的别去修
>
> 2026-09-12 18:20 实测 `./tests/run_offline_tests.sh` → **8 failed / 2206 passed / 3 skipped**：
>
> | 文件 | 个数 |
> |---|---|
> | `tests/test_stage2_repair_controller.py` | 6 |
> | `tests/test_join_lambda_two_sided_support.py` | 2 |
>
> 样例：`test_short_production_runs_production` 断 `RUN_PRODUCTION`、实际得 `ANALYZE`。
> **这是重构本身造成的**（维护者本人在改），断言会随新语义一起重写，
> **不要当成回归去修、也不要为了让它绿而改回旧语义。**
> 这轮重构就是 **`S2-D`**（把散在三个文件里的控制器收拢）。
> 落地后这段整段删掉，届时以 `./tests/run_offline_tests.sh` 全绿为准。

- [ ] **S2-E 旧修复机制是「关掉」不是「删掉」** —— path_evolution 修复分支 /
  production rescue / rescue 后重标定。**等自治这条再跑通几次再删**，别现在删。

### 不许回退的约定

- **一个 stage 只许有一个控制器。** 不是风格偏好 —— 两套机制各判各的会直接打架。
- **只有三个真终态**：`DONE` / `GLOBAL_BUDGET_EXHAUSTED` / `NO_FEASIBLE_ACTION`。
  `LOCAL_VALIDATION_CAP` / `INSUFFICIENT_DATA` / `CUMULATIVE_FK_MISALIGNMENT` /
  `SKIPPED_WINDOW` / `HALT_BUDGET` **全是路由信号**，不得炸出流水线。
  异常同理：只有 `IBSFrozenCalibrationValidationError`（f_k 被**统计驳回**）是有功效的
  否决，且它也只封存该候选、最多换一次 Epoch，仍不终止 Stage-2。
- **四个量不许混用**：`n_decorr` 是求解器**资格**不是验收量；`N_eff,k / g_k` 是
  **主验收量**（门 = 10）；`top1% weight` 是否决**警报**；occupancy / coverage ESS
  **只诊断 f_k 训练**，永不决定 ΔG 是否有效。
- **低支撑永远不是 `FAIL`**，是「尚不可测」（`INSUFFICIENT_DATA` / `HARD_INSUFFICIENT`），
  动作是加预算。`FAIL` 是支撑够了却违反统计门，动作是换 Epoch。
- **缺证据 ≠ 通过，但缺证据 ≠ 有问题** —— 它是 `UNKNOWN`，动作是去产出证据。
- **累计 f_k 残差只能用 `sampling_states`** —— `energies` 多一个逐 λ 态常数（LRC），
  其逐边差会伪造出「全负号单调」的形状（实测把某窗 span 抬高 2.2 倍）。
  `N_eff/g` 则对它**不变**，两者不变性不同，别"统一"。
- **不得为了得到期待的标签而换误差估计器**（estimator shopping）。新 Epoch 才是拿
  独立证据的地方。
- **`(K−1)×门槛` 不许写** —— 累计偏差门不随边数增长。
- **控制器只读**；执行由独立的 execute 层落盘。

### 设计依据（live，不是待办）

- [STAGE2_CONTROLLER_DESIGN_2026-09-12.md](STAGE2_CONTROLLER_DESIGN_2026-09-12.md)
  —— **权威**：设计 + 实证 + 十条真机咬过的陷阱。最贵的一条是
  **「同一个不变量的 N 份实现」**（λ 身份有四份）。
- [AUTONOMOUS_STAGE2_LOOP_SPEC_2026-09-11.md](AUTONOMOUS_STAGE2_LOOP_SPEC_2026-09-11.md)
  —— 老板定的核心设计指标（唯一验收指标）。
- [STAGE2_AUTONOMOUS_LOOP_STATUS_2026-09-11.md](STAGE2_AUTONOMOUS_LOOP_STATUS_2026-09-11.md)
  —— 逐个 bug 的修复流水账（历史，不是待办）。
- [PLAN_PATH_REPAIR_2026-09-11.md](PLAN_PATH_REPAIR_2026-09-11.md) —— 设计要求的来源，
  分支编号（5a/5b/6）出自这里；代码里 10 余处引它作设计依据。
- [STAGE2_THREE_AXES_COUPLING_2026-09-11.md](STAGE2_THREE_AXES_COUPLING_2026-09-11.md)
  —— 分窗口/分 λ/分采样量三轴当前怎么耦合的。
- [STAGE2_ROOT_CAUSE_2026-08-28.md](STAGE2_ROOT_CAUSE_2026-08-28.md) —— **所有收敛门
  对单轨迹重加权这个失效模式是瞎的**：实测门全绿时 ΔG 错 42 kJ/mol。别拿质量门当验收。

---

## 2. local-residual / EXP-033

背景：`B_φ` **不是配体的物理模型**，是提高 λ 态混合的**采样增强项**；按体系在线学的只有 `f_k`。
方案见 [EXP-033_LOCAL_RESIDUAL_REFIT_2026-09-10.md](EXP-033_LOCAL_RESIDUAL_REFIT_2026-09-10.md)，
P1 落地见 [EXP-033_P1_LANDED_2026-09-12.md](EXP-033_P1_LANDED_2026-09-12.md)。
**P3（跨配体通用权重）已于 2026-09-12 由用户拍板划掉，前提没了，别重新提。**

- [ ] **EXP-033-P1-GPU（先做这个）P1 闭式重训真机一次没跑过。**
  `closed_form_refit()` 里 probe 那段要建 OpenMM Context，只过了静态校验。
  最小验证：4W53 开 `--outer-lambda-local-residual-ibs`（配体指纹对不上冻结的 Atenolol
  ⟹ 会走重训），跑到预平衡结束看它出不出 manifest。
- [ ] **EXP-033-P2（EXP-033 唯一还开着的一条）** ridge 臂 vs 出厂非线性臂，**U3 口径上机**：
  window-0 utility + ΔG 一致性，且**两臂各自独立标定并冻结自己的 `f_k`**。
  U4 就是栽在候选臂复用 baseline 的 `f_k`，被封为 `INVALID_FOR_PROMOTION`。
  ⚠️ 做 A/B **不要**走 P1 的自动重训（两臂各自重训 ⟹ `sampling_score_sha256` 变成
  run-dependent，两臂不再共用同一把尺子）。要先离线冻一份两臂共用
  （`tools/retrain_local_residual_offline.py`）。
- [ ] **LR-01 `residual_sampling` 无条件进每个 stage 的指纹** ——
  `abfe_pipeline.py:11008`，在 `if stage_name == "vanishing"` 分支**之前**。
  ⟹ 打开开关会让预平衡 / attachment / decharging 的缓存**全部失配、整条链从头重算**，
  而那几段的哈密顿量根本没被残差碰过。紧邻的 MEM-00h 注释写的正是相反的做法。
  正解：挪进 vanishing 作用域（运行时判据是 `stage_name in {"vanishing","vanishing_rescue"}`，`:5489`）。
- [ ] **LR-02 `skip_unsupported_frames` 该撤或反转** —— 支撑域外的帧正是"模型没覆盖这个
  体系"的证据，跳过它等于把本该触发停止的信号变成拟合时看不见的样本。
- [ ] **LR-03 `sample-hard-window-scratch` 在主线里是死的** ——
  实现在发布清理时被移出的 `archive/` 里（`outer_lambda_neural_basis.py` 有 **13 处**
  `from archive import`，全是空壳）。而且那份 legacy 实现读 `manifest["lambda_shield"]`，
  WCA 壳退役后该字段是 `None` → `TypeError`。**要么补实现，要么把入口一起删掉。**
- [ ] **LR-04 没有 solvent-only 入口** —— `--only-complex-charging` /
  `--only-boresch-attachment` 与 residual 互斥。
- [ ] **LR-05 重训用的 λ 表是默认值** `linspace(1.0, 0.5, 8)`
  （可用 `outer_lambda_refit_lambda_max/_min/_n_states` 改）。因为重训发生在预优化算出
  真实 stage-2 λ 路径**之前**。按 `B_φ` 的定位这不影响对错；要用真实窗口 λ 表得把重训
  往后挪一个位置 —— 那要改的是**时序**，不是这个模块。

> 🛑 **别在 run 内为了 A/B 重训**，三条理由：违反预注册；`sampling_score_sha256` 变成
> run-dependent；**拿缺构型的帧拟合会把缺掉的态焊进模型**，然后 ESS / overlap /
> split-half / 三方一致全都会更绿——它们只问"这批样本内部自洽吗"。
> 完整的坑清单见 [archive/HANDOFF_LOCAL_RESIDUAL_2026-09-11.md](archive/HANDOFF_LOCAL_RESIDUAL_2026-09-11.md) §6。

---

## 3. 发布工程门

发布定位是 **clone-and-run**（不打包、不发科学结论）。判据只有一条：
`pytest tests/test_fresh_clone_imports.py` 绿。完整论证见
[RELEASE_READINESS_2026-08-31.md](RELEASE_READINESS_2026-08-31.md)。

> ✅ **`REL-01`（预编译 `.so` 随仓库分发）2026-09-12 已执行**，故不再列：
> `.gitignore` 加了三条例外放行 `build` 符号链接 + `build_exp026_a2/*.so`，
> 并单独排掉没有扩展名的 gtest 可执行文件。`git add -An plugins/` 应当**只有 4 条**。
> 文档（[GETTING_STARTED.md](GETTING_STARTED.md)《CUDA 插件》/
> [TROUBLESHOOTING.md](TROUBLESHOOTING.md)）已同步成"随仓库分发、按环境文件建环境不用编"。

- [ ] **REL-02 `abfe_core.py` 分片没审完** —— 第九轮审查里它是唯一没有分片正文的
  （5 条 P2 只有汇总行）。而《五个文件分别应补什么》恰恰把它的职责定为"集中最终结果资格
  与协议登记"。补审属于预览版前的工作。
- [ ] **REL-03 到目前为止全部是 CPU / 静态验证，零 GPU 复验。** 最需要真机的：
  - residual 臂混合覆盖度门换口径后，EXP-030 candidate 臂的门读数（`ess_gate_mixture_gauge` 应为 `sampling_states`）；
  - 三个 decharging builder 新增的 `frozen_ll_pairs` 断言（真实体系上触发 ⟹ P0-01 的前提本来就不成立，那是新发现不是回归）；
  - 两条腿同进程时的 `pipeline.log` 分离（已用最小复现验证，未在真实两腿运行上确认）。
- [ ] **REL-04 2026-09-09 全仓审计那 52 处改动无一上过 GPU。**
  最需要复验的三处：偏置爬坡补 1.0 档、preopt 探针的 force group 重划、加密点的采样语义变更。
  复跑命令见 [archive/AUDIT_2026-09-09_full_repo.md](archive/AUDIT_2026-09-09_full_repo.md) §4。

---

## 4. 已定位、判定不改

> 完整记录见 [`archive/AUDIT_2026-09-09_full_repo.md`](archive/AUDIT_2026-09-09_full_repo.md)。
> 那轮审计共 62 条候选，**52 条已修**（含 NaN 根因、stdout 消失、离线重算挂死、
> 一批宿主内存放大、preopt 缓存两层拆分）。下面 7 条是**有意不修**的，
> 登记在这里是因为它们确实"已定位、未修" —— 下一个人有权知道，
> 也免得被当成新发现重查一遍。**理由都别重新论证。**

- [ ] **AUDIT-01** `abfe_core.minimum_image_displacement_nm` —— 候选立方体随长宽比增长，
  `radius > 64` 的 guard 在尝试 `N × 2.1e6 × 3 × 8` 字节之后才触发。
  *不修*：真实触发需要极端长宽比 + 大批量输入；`docs/design` 的盒型识别提案覆盖这一片。
- [ ] **AUDIT-02** `abfe_core` 膜 leaflet 的 wrapped/unwrapped 混用
  （`_protein_leaflet_cross_sections_nm2`、`assign_lipid_leaflets`、
  `verify_membrane_normal_axis`）。
  *不修*：只影响膜路径，且 `membrane_observables_from_trajectory` 已按最大空隙弧修过一次；
  当前生产是可溶体系，留给膜线单独一轮。
- [ ] **AUDIT-03** `free_energy_engine.run_independent_windows` 保留全部帧
  （73k × 1000 帧 × 8 态 ≈ 14 GB）。
  *不修*：当前无生产调用者，是"接线即爆"而不是现在就爆。接线前必须先改。
- [ ] **AUDIT-04** `apbs_correction._read_dx_values` —— 257³ 网格约 1 GB 峰值
  （原文本 + 切片副本 + Python float list 三份）。
  *不修*：APBS 修正当前不在主线路径上。改法是 `np.frombuffer`，可降到 ~136 MB。
- [ ] **AUDIT-05** `abfe_core.OnlineConvergenceMonitor` 的 K==1 会抛、
  `n_k_array` 与 `u_kn` 列数不一致。
  *不修*：`abfe_core` 之外无调用者。
- [ ] **AUDIT-06** `TraditionalABFEPipeline.pre_equilibration_identity_fingerprint`
  引用 `self.pressure` / `self.barostat_protocol`，该类 `__init__` 从未赋值。
  *不修*：全仓无活的调用点（runabfe 里那两个 baseline 都是 `ABFEPipeline` 实例）。
- [ ] **AUDIT-07** `abfe_pipeline._rebalance_fingerprint` 没有 System 身份绑定
  ⟹ System 变了而 Boresch 锚点没变时，`rebalance.chk` 会被复用。
  *🛑 明确不修*：唯一修法是把自产产物的 sha256 放进缓存身份，而那是**用户否决过
  4 次**的做法（`code_sha256` 2026-08-24、`system_xml_hash`+`positions_sha256`
  2026-09-09、`preopt_cache_sha256` 2026-09-09）。这类 payload 本来就有显式协议
  版本号承担"算法变了"的信号。**不要再提这个方案。**

> ⚠️ 那 52 处**没有一处上过 GPU**。最需要真机复验的：偏置爬坡补 1.0 档、
> preopt 探针的 force group 重划、加密点的采样语义变更。
> 复跑命令见审计文档 §4。

---

## 5. 膜受体–配体路线（暂停）

> **2026-09-12 状态：整条线暂停**，停在 C4。当前主线是**可溶体系 4W53**，
> 膜线不阻塞任何在推的工作。本节内容的时间戳是 **2026-08-11**，之后没有人动过——
> 重启这条线之前先对着源码核一遍，别直接照着勾。
> 带电配体那一半另见本文 [PHY-03](#6-phy-03带电路线的条件性阻塞)。

> 2026-08-31 发布整理并入，原文件 `docs/status/memtodolist.md`（在 `Atenolol-rank11`，**不在本仓**）。
>
> ⚠️ 本仓库的 `docs/status/` 已于 2026-09-02 撤销（见 [README](README.md)《运行期发现往哪写》）。
> `memtodolist*.md` 在 `Atenolol-rank11`，**别在本仓里找**。


更新日期：2026-08-11（**C3 与 MEM-00h 已正式关闭（用户确认），进入 C4**。
C3-0~C3-4 全部跑过一轮；co-ion/ParameterOffset 归因诊断完成；C3 protocol v2
双层门重设计已实现；C2 的 C-seam switch 不一致已用"MEM-00h 双边归一化"
修复并在全部真实 GPU 数据上验证——A/B 100/100 + C/D 50/50，全部 150 帧一次
通过，C2 的 C-seam 力差回落到机器精度；`summary.json`/`mem00h_report.json`
两份 fail-closed 汇总产物已生成，均 `status=complete, passed=true`）  
状态：Phase B 工程实现基本完成；B5 已关闭。C1、C2、C3 已关闭；MEM-00h 已
关闭。当前进入 C4。

**已关闭事项的完整过程、失败证据和验收记录均已原文迁移到 `docs/status/memtodolist_archive.md`（在 `Atenolol-rank11`，**不在本仓**）。**


---

### 1. 当前做到哪里

已完成的工程能力不再逐项放在本清单中，完整证据见归档。当前状态摘要：

- B1：膜体系识别和 `MonteCarloMembraneBarostat` 已实现。
- B2：`charge_treatment` 配置和双计数 fail-closed 已实现。
- B3：PME co-alchemical charge-transfer Hamiltonian 已实现。
- B4：溶剂腿 reserved co-ion dummy builder 已实现。
- B5 已关闭（2026-08-09）：cache、resume、provenance 全套离线测试 0 failed，
  co-ion 隔离/缓存拒绝/resume 一致性逐项复核通过。
- 中性 Atenolol 膜体系 complex/solvent 双腿工程 smoke test 已跑通。
- C1 已关闭：Na/Cl 硬性验收通过；采用单 seed pilot，不补 seed；Ca 为已知统计限制且不阻塞。

当前主线：

```text
B5、C1、C2、C3、MEM-00h（已关闭）
    ↓
C4 带电膜双腿 smoke test（当前，尚未开始）
    ↓
C5 co-ion 位置/restraint 敏感性
    ↓
Phase D 生产资格
```

---

### 2. Phase C：当前验证

#### C4：带电膜 complex/solvent 双腿 smoke test

前置：B5、C1、C2、C3 全部通过。**2026-08-11：确认 C3 与 MEM-00h 正式关闭，
C4 已解锁。**

**C4 是接线 smoke test，不是生产自由能计算**——不追求收敛，不出最终
ΔG；全部产物必须标 `production_qualified=false`（第 6 步）。C2 的纯脂质
slab（无蛋白）不能代替这里的真实 receptor–ligand complex；C4 第一次真正
需要"膜 + 蛋白 + 带净电配体"这套完整组合。

用户指定的执行顺序（2026-08-11 登记，按顺序执行，不并行跳步）：

**受体/配体组合——阻塞第 1 步，待用户决定，本文档不擅自选择**（2026-08-11
现状普查，只读，未改任何文件）：

- **已有、可复用的**：`memtest/` 下有一个真实的 283 残基 GPCR 样受体
  （`Atenolol-rank1apo.pdb`/`Atenolol-rank1.pdb`，含 TM3 的 `DRY`、TM7 的
  `NPxxY` 保守基序，疑似热稳定化突变体，ICL3 可能被截短）已经嵌入真实
  POPC 膜（`memtest/step7_production.gro`：`PROA 1 / POPC 90 / Na+ 25 /
  Cl- 36 / TP3 9542 / Atenolol-rank11 1`，45354 原子），配上中性 Atenolol
  （`Atenolol-rank1.gjf` 的 QM 电荷计算用的是 `Charge=0`，即去质子化的
  仲胺；`memtest/Atenolol-rank11.itp` 41 个原子电荷加总 Σq≈0），
  `memtest/README_MEMTEST.md` 记录了这套中性体系已经跑通的完整
  complex/solvent 双腿工程 smoke test（膜恒压器、quality gate、诊断脚本
  全部现成）。`abfe_core.py`/`runabfe.py` 的 charge-transfer + 膜恒压器
  通用接线（`--only-complex-charging`、`--membrane-input-declaration`、
  co-ion dummy 插入）已经用这套中性体系验证过，从未在带电配体上跑过。
- **真正的冲突**：`docs/status/memtodolist_archive.md`（在 `Atenolol-rank11`，**不在本仓**）（2026-07-29）记录过一条决定——
  **"首个体系 = SERT（血清素转运体），配体默认净电荷 +1"**。但实际建出来
  并跑通的是上面这个 GPCR + 中性 Atenolol，跟当年那条决定不是同一个体系：
  SERT 从未真正建过膜体系（没有对应的 CHARMM-GUI 产物、没有嵌膜、没有跑过
  任何 smoke）。
- **配体电荷缺口，跟选哪个受体无关，两条路都要补**：仓库里没有任何带电
  （质子化、净 +1）的 Atenolol 参数——所有现成拓扑（根目录
  `Atenolol-rank1.itp`、`memtest/Atenolol-rank11.itp`）都是从
  `Charge=0` 的 QM 计算导出的中性形式。要走"配体带净电"这条路，不管配哪个
  受体，都需要重新做一次质子化仲胺的 QM 电荷推导（Gaussian）+ 重新生成
  GAFF 拓扑——不是挪文件就能解决的工作量。
- **受体身份记录缺口**：`memtest/membrane_input.json` 明确写着
  "未记录上游 PDB ID"、构象态"unspecified"——呼应 §A5"记录受体结构 ID、
  构象状态、突变、缺失残基和质子化态"这条从未打勾的要求；C4 定位是接线
  smoke（`production_qualified=false`），这个记录缺口是否必须先补齐、
  还是可以先如实标注"未知"往前走，也需要用户决定。

**用户 2026-08-11 明确表示：这个选择稍后告诉我，现在只要求把决策点和现状
写清楚——不要自己选受体/配体组合，也不要开始任何构建。**

1. **准备真实带电膜 complex，以及匹配的 solvent leg**
   - [ ] ligand 必须带净电荷（不是 C1/C2 用的中性探针或单原子简化）；
   - [ ] build 时显式插入 reserved neutral ion-shaped dummy；
   - [ ] 排除结构性离子、孔道离子、口袋/膜头基/疏水核中的候选（呼应
     §A5 已经列出但从未做过的排除清单）；
   - [ ] complex 与 solvent 两腿冻结**同一个** co-ion identity 和 restraint
     定义（不能两腿各自独立选一次）。
2. **零步静态预检**（不积分，只建 Context 查一次）
   - [ ] charging 全部 λ 态总电荷恒定；
   - [ ] `λ_coul=1`：ligand 满电、co-ion 中性；`λ_coul=0`：ligand 去电、
     co-ion fully charged；
   - [ ] Stage2 输入已经 baking 完成，System 里不存在活的 `lam_coul`
     GlobalParameter；
   - [ ] complex 用膜恒压器（`MonteCarloMembraneBarostat`），solvent 用
     各向同性恒压器；
   - [ ] handoff protocol/version 和 co-ion fingerprint 都已经进入
     cache identity。
3. **最短 GPU smoke**（不追求自由能收敛，只要能跑）
   - [ ] complex charging 能建 Context、积分、写 checkpoint；
   - [ ] complex Stage2 能接上 charging 端点（真正走一次 Stage2 handoff）；
   - [ ] solvent charging/Stage2 同样可运行；
   - [ ] 全程 energy/force finite；无 NaN、PME error、粒子逃逸或
     restraint runaway；
   - [ ] Stage2 全程 co-ion 保持 fully charged。
4. **相同命令立即 resume 第二次**
   - [ ] 命中相同 co-ion identity；
   - [ ] 已完成窗口被复用，不重跑；
   - [ ] 不重复插入 dummy/offset/restraint；
   - [ ] handoff/cache protocol 字段一致。
5. **复制一份 co-ion spec、故意篡改**（atom index / fingerprint /
   endpoint charge 任选一种）
   - [ ] 必须在建 Context **之前** fail closed；
   - [ ] 原始产物不能被这次篡改测试覆盖/污染。
6. **所有 C4 输出统一标注**
   ```json
   {"production_qualified": false}
   ```
   C4 只是接线 smoke，即使全部 PASS 也不能当生产结果用。

**当前最先要做的是第 1 步**：确定并预检真实带电膜 complex/solvent 输入。
§A5"目标膜输入"下的清单（受体结构 ID、构象状态、配体质子化态/形式电荷、
结构性离子排除等）到目前为止都还没做过，是这一步要补的作业，不是重复劳动。

#### C5：co-ion 位置与 restraint 敏感性

前置：C4 通过。

- [ ] 至少 3 个合法 bulk-water 位置和 1 个故意违规位置。
- [ ] restraint 基线：k=100、r0=0.5。
- [ ] 弱/宽：k=50、r0=0.7。
- [ ] 强/窄：k=200、r0=0.3。
- [ ] 每个合法组合跑 complex/solvent 两腿和至少 3 seeds。
- [ ] 检查 dummy 吸附、charged endpoint 水合、触壁比例和 restraint 能量。
- [ ] 净 `ΔΔG_bind` 同时满足 2σ 和 1 kcal/mol 门。
- [ ] 若两腿 restraint 自由能不抵消，给出显式修正或判定路线失败。

---

### 3. 膜输入与科学协议仍缺

#### A5：目标膜输入

- [ ] 准备并验证真正用于带电生产的已平衡膜输入。
- [ ] 记录受体结构 ID、构象状态、突变、缺失残基和质子化态。
- [ ] 记录配体质子化态、互变异构体、形式电荷和参数来源。
- [ ] 核对结构性 Na⁺/Cl⁻，从 co-ion 候选中显式排除。
- [ ] 排除蛋白孔道、结合口袋、膜头基层和疏水核中的 co-ion 候选。
- [ ] 核对蛋白插膜方向、配体 pose、结构水、辅因子和二硫键。
- [ ] 记录膜组成、上下叶组成、胆固醇比例、盐浓度和温度。

#### 热力学循环和 restraint 账目

- [ ] 写清 co-ion restraint 在 complex/solvent 两腿是否抵消。
- [ ] 若可用体积不同，推导并实现显式修正。
- [ ] charge-transfer 路线最终报告必须明确 `APBS/Rocklin = 0`。
- [ ] co-annihilation 只允许实验对照，禁止进入膜生产 preset。
- [ ] `shadow_ibs` 对带电配体明确 fail closed，或完整实现同一 co-ion 路线。

#### 膜生产协议

- [ ] 明确炼金生产阶段使用 NPT 还是 NVT。
- [ ] 若使用 NVT，记录固定盒矢量来自哪一帧。
- [ ] 明确时间步、约束和是否使用 HMR。
- [ ] 明确膜位置限制的分级释放方案。
- [ ] 记录结合位点是水相可及、界面、脂质暴露还是疏水深埋。
- [ ] 对脂质暴露/空腔填充做正反向或双初态迟滞验证。

---

### 4. 生产资格 Phase D

- [ ] D1：关闭 P1-19/P1-19b 的跨运行不确定度问题。
- [ ] D1：对 P1-22 的 Stage 2 帧选择和 σ 口径形成正式结论。
- [ ] D2：完成 Boresch 真实键拓扑和二面角更新门。
- [ ] D3：至少 3 个独立生产重复一致。
- [ ] D4：至少一个公开或可追溯膜受体 benchmark 通过。
- [ ] D5：完整 provenance、运行命令、环境、seed、输入 SHA256 和复现实验脚本。
- [ ] 膜质量门通过。
- [ ] overlap/ESS 和修正后的不确定度门通过。

---

### 5. Definition of Done

只有以下项目全部完成，才能声明支持生产级膜受体–配体 ABFE：

- [ ] C2–C5 全部通过（C1、C2、C3 已关闭并归档；C4 已解锁，C5 未开始）。
- [ ] co-ion 两腿显式存在、进入 PME、受控并进入全部缓存指纹。
- [ ] 全部 λ 总电荷恒定，且未重复应用 APBS/Rocklin。
- [ ] 膜恒压和平衡质量门通过。
- [ ] Boresch、co-ion restraint 和标准态修正闭环。
- [ ] 至少 3 个独立重复一致。
- [ ] 公开 benchmark 通过。
- [ ] 最终结果可审计、可恢复、可复现。

---

## 6. PHY-03（带电路线的条件性阻塞）

**P1，实验路线。**charge-transfer 的 tethered charge carrier 不能按当前论证严格跨腿抵消

- 位置：`abfe_core.py` 的 co-ion restraint 说明与表达式（约 1088–1117 行）；
  `ibs_engine.py::_create_co_alchemical_ion_restraint`（约 807–848 行）；
  `abfe_core.py::resolve_charge_treatment` 的 `closes_thermodynamic_cycle`（约 926–947 行）。
- 触发：带净电配体使用 `co_alchemical_charge_transfer`。
- 问题一（配分函数）：代码以“两腿同一锚点规则、同一 k/r0”推断 restraint 自由能严格
  抵消。实际受限 charge carrier 的配分函数包含
  `integral exp[-beta*(U_env(r)+U_rest(r-r_anchor))] dr`。complex 与纯水腿的
  `U_env`、排除体积、anchor 系综均不同；lambda=0 时 carrier 还带真实电荷并与环境
  相互作用，因此 restraint 与 carrier 溶剂化不能分离成一个两腿相同的常数。
- 问题二（barostat）：`dx0/dy0/dz0` 是冻结的笛卡尔 nm per-bond 参数。barostat
  缩放盒矢量和粒子坐标时，`d0` 不缩放；“井心随体系/盒一起缩放”的注释不成立，
  半各向异性/三斜 NPT 下尤其明显。
- 影响：decoupled complex/solvent 端点未必共享可严格消掉的 reservoir 状态，最终差值
  可能含 carrier 位置、盒大小、蛋白排除体积和 restraint 的非物理贡献。项目当前已经把
  charge-transfer 标为 `production_qualified=False`，这一边界必须保留；但同时写
  `closes_thermodynamic_cycle=True` 仍过度承诺。
- 要求：给出包含 carrier restraint/标准态/环境项的完整热力学循环推导；若不能证明解析
  抵消，就显式计算两腿 restraint/reservoir correction。参考位移需要采用真正随盒变化的
  分数坐标定义，或改成不依赖冻结笛卡尔井心且有解析标准态修正的相对约束。
- 验收：carrier 平移、anchor 选择、盒尺寸、各向异性缩放和 restraint 强度扫描后，修正后
  ΔG 在统计误差内不变；complex/solvent reservoir 端点有独立 free-energy closure test。
  C4/C5 未通过前不得把数值提升为生产结果。

---

## 往这里加条目的规则

1. **一条待办只在本文出现一次。** 专题文档讲"为什么"，本文讲"还欠什么、欠在哪一行"。
2. 关闭一条就**整段移进 `archive/`**，页首写清结论去了哪 —— 不要在本文留 `[x]` 尸体。
3. 新条目必须带**位置**（`文件:行号` 或函数名）和**判据**（怎样算做完）。
   写不出判据的不是待办，是想法，去 `design/`。
4. 标"不修"的要写明理由，并注明**理由别重新论证** —— 否则下一个人会花一天重查一遍。
5. 科学结论不进本文，进 [STATUS.md](STATUS.md)；协议/代码变更进 [CHANGELOG.md](CHANGELOG.md)。
