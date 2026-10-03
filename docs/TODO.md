# 待办清单

[项目入口](../README.md) · [文档导航](README.md) · [当前科学状态](STATUS.md) · [变更记录](CHANGELOG.md)

> **本文是本仓库唯一的待办文件。** 2026-09-24 起 P1/P2/P3 合并成这一份（剩下的条目不多了）；
> **重要等级跟着条目走**，写在编号后的方括号里，清点用 `grep -n '\[P[123]\]' docs/TODO.md`：
>
> | 等级 | 含义 | 在本文哪节 |
> |---|---|---|
> | `[P1]` | **挡住在推的工作**，不做就拿不到可引用结果 | §1 |
> | `[P2]` | 该做，但不挡路（「不挡路」不是「不重要」） | §2 |
> | `[P3]` | **现在明确不做**：判定不改 / 不是 bug / 暂停 / 押后到某个触发条件。想重新论证某条之前先查这里，**理由别重新论证** | §3 |
>
> 判定不改 / 不是 bug 的 P3 条目**故意不用 `- [ ]`**（永远不会被打勾，用 checkbox 会让待办数虚高）。
>
> - 已关闭条目的原文 + 关闭记录：[archive/TODO_closed.md](archive/TODO_closed.md)（按日期倒序）
> - 重放表、预算实测表、报错 run 归因：[archive/TODO_evidence_2026-09-16.md](archive/TODO_evidence_2026-09-16.md)
> - 2026-08-06 旧主表：[archive/TODO_2026-08-06_unreconciled.md](archive/TODO_2026-08-06_unreconciled.md)，里面的 `- [ ]` 只表示"当时未完成"。
>   ⚠️ 编号**同名不同义**（`P1-19` 在两份材料里是两件事），别跨文档机械对账。

> 📌 **「某条缺陷修没修」的权威是源码，不是这里的 `- [ ]`。** 09-16 对账时 16 条未打勾里 13 条早已修掉、
> 没人回来打勾 ⟹ 关一条的动作是「回源码看缺陷在不在」。对账只证明缺陷不在了，**不证明修得对**。

## 目录

| 节 | 条目 | 等级 |
|---|---|---|
| 1 | （空） | P1 |
| 2.2 静态审核遗留 | `REVIEW-0924`、`REVIEW-0926` | P2 |
| 2.3 待维护者拍板 | `IDENT-01` | P2 |
| 2.4 local-residual / EXP-033 | `LR-06`、`EXP-033-P1-GPU`、`EXP-033-P2`、`LR-02`、`LR-05` | P2 |
| 2.5 发布工程门 | `REL-03` | P2 |
| 2.6 性能审查 0928（已核） | `BORESCH-AUTO-01`、`PERF-0928` | P2 |
| 2.7 代码复查 0929（已核） | `CI-RED-01`、`REVIEW-0929`（剩 L8） | P2 |
| 2.8 代码复查 0930（Opus，已对抗核实） | `REVIEW-0930` | P2 |
| 2.9 准确性复查 1003（合成复现） | `ACCURACY-1003`、`COVERAGE-1003` | P2 |
| 2.10 fragment-opt 精度线 | `FRAG-S1LONG-01`、`FRAG-BATCH-01` | P2 |
| 3.1 判定不改 | `AUDIT-01`~`07` | P3 |
| 3.2 不是 bug | `BM-A` | P3 |
| 3.3 膜线（暂停） | `MEM-LINE`、`PHY-03` | P3 |
| 3.4 押后到方法定稿 | `REL-05` / `REL-06` / `REL-10` + 两条明确不做 | P3 |
| 3.5 押后到独立端点段重新打开 | `COMBINE-01` | P3 |
| 3.6 押后到重构 | `TEST-02` | P3 |
| 3.7 性能审查 0928：非默认路径 / 驳回 | `PERF-0928-COND` + 4 条驳回 | P3 |
| 3.8 代码复查 0929：降级 / 驳回 | `REVIEW-0929-COND` + 5 条驳回 | P3 |
| 3.9 代码复查 0930：旧控制器 / 实验路径 / 未核 | `REVIEW-0930-COND` | P3 |
| 3.10 模块审计 0928：测试缺口 | `TEST-GAP-0928` | P3 |
| 3.11 已关的实验开关 | `RELAY-01` | P3 |
| 4 | 不许回退的约定 | — |

---

## 1. P1 · 挡住在推的工作

**当前为空**（2026-09-23 清零）。最近一批的去向：`REPRO-01`/`REPRO-02` 关闭；控制器时代的 `AUDIT-S2-03`/`BM-B`
判为已被取代（主线改走[复现路线](DEBUG_REPRO_MODE_2026-09-22.md)、控制器关闭）；`BOX-01`/`LR-06` 降 P2（`BOX-01` 10-03 已关）；
`COMBINE-01` 降 P3。原文都在 [archive/TODO_closed.md](archive/TODO_closed.md)。
> 📌 10-03 注：上面「主线改走复现路线、控制器关闭」指的是**旧**自治控制器；09-29 维护者确认主线 stage 2 走 LCR 轮次控制器（`stage2_lcr_rounds`）。

---

## 2. P2 · 该做，不挡路

### 2.2 静态审核遗留（2026-09-24）

- [ ] **REVIEW-0924 [P2] 09-24 两份静态审核里还没对照代码核过的条目。**
  **位置**：[archive/CODE_REVIEW_2026-09-24.md](archive/CODE_REVIEW_2026-09-24.md)（凌晨份，10 条**全部**没核）、
  [archive/CODE_REVIEW_2026-09-24_pm.md](archive/CODE_REVIEW_2026-09-24_pm.md)（下午份，剩 #5 #8 #10）。
  行号是审核当时的，代码之后又改过，**先按函数名重新定位**。
  ⚠️ 都是**审核代理的说法**，量级是它估的。对照本机 OpenFE 源码（`/home/ruigengji/openfe`）核；
  不触发的潜伏项、orb 线（基本不用）、膜线（暂停）直接进 P3 或关掉，别当 bug 修。
  先核凌晨份的两条"静默出错数"：#1 stage0 attachment 缓存 resume 用的是缓存里记下的 gate 标志、不是本次运行的；
  #4 `--analyze-only` 读窗口缓存的路径没接盒体积门。
  **判据**：每条都有结论（已修 / 不触发 / 有意为之 / 转成独立条目并带 `文件:行号`），写回两份审核的「处理结果」，本条关闭。

- [ ] **REVIEW-0926 [P2] 09-26 静态审核（含 09-27 补充）还没对照代码核过。**
  **位置**：[archive/CODE_REVIEW_2026-09-26.md](archive/CODE_REVIEW_2026-09-26.md)（行号是 HEAD `e576a54` 的）。规矩同 `REVIEW-0924`：量级是代理估的，不触发的潜伏项直接进 P3 或关掉。
  **先核两条 P1**（09-28 性能审查说到 `e576a54` 仍未修，也没核过）：
  #1 LCR 账本没绑定当前阶段协议，旧段可能混入新协议；#2 已完成的 final/stage 缓存能绕过生产盒校验。
  **其余要核的**：#3 analyze-only 没接盒体积门（和 `REVIEW-0924` 凌晨份 #4 **是同一件事**，一起核）；#4 LCR 跳过边缘窗口后 jackknife σ 配错窗；
  #5 非有限 MBAR σ 让补采条件失效；#6 固定 f_k burn-in 向上取整、超过配置步数；性能 `PERF-01`～`05`（09-28 的 `PERF-0928` 不重复这几条，实测时一起量）。
  **不核**：RBFE 三条（A/B/C，RBFE 线叫停）；「结果质量与采样效率改进补充（09-27）」的 QUALITY-01～07 是改进建议、写不出判据，要做就去 `design/` 立方案。
  线索：该审核的「已核实不应继续报」一节说 Stage0 attachment 缓存门已修，正对应 `REVIEW-0924` 凌晨份 #1，可以顺手核掉。
  **判据**：每条有结论（已修 / 不触发 / 有意为之 / 转成独立条目并带 `文件:行号`），写回该审核文件的「处理结果」，本条关闭。

  📌 **10-03 进展**：#1、#2、#3、#4、#6 已被 10-03 准确性复查用合成产物复现、确认仍在 ⟹ **转进 §2.9 `ACCURACY-1003`**（A1 / A5 / A7 / A3），这里不再重复核；#5（非有限 σ_MBAR 让补采失效）09-30 复查判定已被 09-29 补段门改成 `σ_jk > σ_tol` 解决（待写回原文）。**本条只剩 PERF-01～05**。

### 2.3 待维护者拍板（代理不要自行改）

- [ ] **IDENT-01 [P2] `allow_untrusted_stage_results` 在顶层 `final_results` 指纹里，stage 指纹里已剔除。**
  **位置**：`abfe_pipeline.py::ABFEPipeline._stage_protocol_key`（顶层与 `kwargs` 两处都剔）vs
  `_build_top_level_protocol_key` 的 `config = dict(self._last_run_config)`（只 pop 了 `resume` / `run_equilibration`）。
  两边口径不一致是事实，**哪边对没定**：
  - **该剔**：它只决定质量门没过时中止还是标 `results_untrusted` 继续，不改任何被算出来的数；
  - **不该剔**：顶层指纹代表整个 run 的身份，而这个开关确实改变了 `final_results.json`（带不带标记）——
    正是 `LR-01` 留下的规矩「顶层 run 指纹与 stage 指纹口径不同，别跟着统一」。
  ⚠️ **代价是单向的**：摘掉它会改 payload ⟹ 现存 run 的 final-result 缓存全部失配。
  **判据**：判「该剔」⟹ `tests/test_untrusted_switch_is_not_identity.py::test_flag_is_scrubbed_from_stage_protocol_key_both_levels`
  扩到顶层，同一改动写清缓存作废；判「不该剔」⟹ 在 `_build_top_level_protocol_key` 就地注释「刻意的，别统一」，然后归档本条。

### 2.4 local-residual / EXP-033

> `B_φ` **不是配体的物理模型**，是提高 λ 态混合的采样增强项（不进哈密顿量 ⟹ 训练帧源不影响 ΔG 正确性）。
> 方案：[EXP-033_LOCAL_RESIDUAL_REFIT_2026-09-10.md](archive/EXP-033_LOCAL_RESIDUAL_REFIT_2026-09-10.md)；
> P1 落地：[EXP-033_P1_LANDED_2026-09-12.md](archive/EXP-033_P1_LANDED_2026-09-12.md)。
> ⚠️ `EXP-033-P1/P2/P3` 是 EXP-033 自己的阶段编号，不是优先级。**P3（跨配体通用权重）09-12 用户划掉，别重提。**
>
> 🛑 **别在 run 内为了 A/B 重训**：违反预注册；`sampling_score_sha256` 变成 run-dependent；
> 拿缺构型的帧拟合会把缺掉的态焊进模型，然后 ESS / overlap / split-half 全都更绿。
> 坑清单：[archive/HANDOFF_LOCAL_RESIDUAL_2026-09-11.md](archive/HANDOFF_LOCAL_RESIDUAL_2026-09-11.md) §6。

- [ ] **LR-06 [P2] 预热鬼影期 × 插件 0.1 Å 硬门 ⟹ outer 臂在窗口 0 必死 —— 症状已消除，剩第 5、7 步。**
  第 1–4、6 步（插件 4 处硬门改夹取、Reference 侧同步、重编、回归、同步 sha256、`cyclod_ligand1_outer` 真机复跑通过）
  的完整记录（根因、方案 A/B 选择、真机 A/B、`r_floor = 1.5 Å` 的依据、排除项）在
  [archive/TODO_closed.md](archive/TODO_closed.md)「2026-09-24（未关闭条目的已完成部分）」。
  - **第 5 步 删临时脚手架**（09-18 押后条件已满足）：`em_no_residual.install()` 的两个 `Context` 补丁。
    `patched_set_parameter` 的 9 处调用点全在 `ibs_engine.py`、同一形状 `context.setParameter(f"{self.prefix}_bias_scale", X)`
    ⟹ 一个 helper 收掉；`patched_get_state` **故意是全局的**（逐点传参必漏）⟹ 建议保留并在注释写清为什么。
    `em_no_residual.py:138`「还没经过大规模验证」那条注释已过期。
  - **第 7 步 换疏水口袋体系复跑**（jnk1 / p38 / brd4）：还没做。rep2/rep3 是同一个体系，只证明「补丁能跑完」。
  ⚠️ `runs/cyclod_ligand1_outer` 与 `runs/cyclod_ligand1` **不能做 A/B**：配体对不上冻结 manifest ⟹ 每个 rep 各自闭式重训 B_φ；
  真要 A/B 得按 [RETRAIN_LOCAL_RESIDUAL.md](RETRAIN_LOCAL_RESIDUAL.md) 手工冻一份权重给两臂共用。
  ⚠️ `r_floor` 是编译期常量、绑定 shipped R1；换重训模型会静默沿用 1.5（升级路径在 `r1_model_layout.h` 的注释里）。
  **判据**：第 5 步后 `tests/test_lr06_residual_distance_clamp_2026_09_16.py` 仍全过；第 7 步一个疏水口袋体系 outer 臂窗口 0 不再 fail-closed。

- [ ] **EXP-033-P1-GPU [P2] 真机首跑 09-17 已通过，但拟合出来是空模型。**
  `cyclod_ligand1_outer/rep1` 走了自动重训、`closed_form_refit()` 的 probe 真机跑通，但 B 离散度 **2.9e-06 kT** ⟹ 只证明「不炸」。
  **判据**：查清为什么拟合成空模型——帧源（`pre_equilibration.dcd`）构型跨度不够，还是闭式解在这个体系上退化。
- [ ] **EXP-033-P2 [P2]** ridge 臂 vs 出厂非线性臂，**U3 口径上机**：window-0 utility + ΔG 一致性，且**两臂各自独立标定并冻结自己的 `f_k`**
  （U4 就是栽在候选臂复用 baseline 的 `f_k`，被封 `INVALID_FOR_PROMOTION`）。
  ⚠️ **不要**走 P1 的自动重训，先用 `tools/retrain_local_residual_offline.py` 离线冻一份两臂共用。
- [ ] **LR-02 [P2] `skip_unsupported_frames` 该撤或反转**：支撑域外的帧正是"模型没覆盖这个体系"的证据，跳过它等于把停止信号藏起来。
- [ ] **LR-05 [P2] 重训用的 λ 表是默认值** `linspace(1.0, 0.5, 8)`（`outer_lambda_refit_lambda_max/_min/_n_states` 可改），
  因为重训发生在预优化算出真实 stage-2 λ 路径**之前**。不影响对错；要用真实 λ 表得改**时序**，不是改这个模块。

### 2.5 发布工程门

> 发布定位是 **clone-and-run**（不打包、不发科学结论），判据只有 `pytest tests/test_fresh_clone_imports.py` 绿。
> 论证见 [archive/RELEASE_READINESS_2026-08-31.md](archive/RELEASE_READINESS_2026-08-31.md)（08-31 措辞，按收尾清单读）。
> 押后的 `REL-05`/`REL-06`/`REL-10` 在 §3.4。

- [ ] **REL-03 [P2] ① ② 两条仍欠真机**（③ 09-16 已关闭）。39 rep 全量 benchmark 上过 GPU，但没覆盖这两条：
  - ① residual 臂混合覆盖度门换口径后，EXP-030 candidate 臂的门读数（`ess_gate_mixture_gauge` 应为 `sampling_states`）——benchmark 没跑 candidate 臂；
  - ② 三个 decharging builder 的 `frozen_ll_pairs` 断言——39 rep 里没有一条报过它（真实体系上触发 ⟹ P0-01 前提不成立，是新发现不是回归）。
  **判据**：各有一条真机日志读到对应读数。

### 2.6 性能静态审查（2026-09-28，已逐条对照源码核过）

> 来源：[archive/PERF_REVIEW_2026-09-28.md](archive/PERF_REVIEW_2026-09-28.md)（26 条，HEAD `e576a54`）。09-28 逐条核过：行号基本都对，但**半数以上把「在不在默认路径」判错了，或优先级定高了**；
> 以下只收核实后的版本（位置、机制按核实结果改写过，**不要回去照原文修**）。非默认路径的条目和驳回的条目在 §3.7。
> ⚠️ 所有耗时/内存量级都是审查代理**推算**的，没有一条实测过。

- [ ] **BORESCH-AUTO-01 [P2] 没写 `boresch_source` 的配置走 `auto`（Orb）链，在默认 `openfe_fixed` 下必然拒掉全部候选并抛错。**（原 A3，行为问题，不是性能问题）
  **位置**：`abfe_core.py::OrbBoreschEstimator._apply_hybrid_filter`（`openfe_fixed` 时恒返回 kr=`OPENFE_FIXED_KR_KJ_PER_MOL_NM2`=4184）
  vs `estimate_multiple_anchors_from_trajectory(max_kr=2000.0)` 的 `if ks["kr"] > max_kr: continue`（快速验证与 finalize 两处）；
  `runabfe.py::resolve_boresch_restraint` 的 `source == "auto"` 分支不传 `max_kr` ⟹ `validated`/`fallback_pool` 全空 ⟹ `RuntimeError("自动 Boresch 估算失败…")`，
  而且是付完整个枚举 + MACE 口袋力之后才抛。无测试覆盖。
  **为什么没炸过**：自带 `abfe_config.json` 写了 `"boresch_source": "simple"`（走 `GeometricRestraintEstimator`），不是审查说的「有缓存」。
  **为什么是 P2 不是 P3**：按 §4「`abfe_config.json` 只是样例、不是默认值」，**手写配置不写这个键 = 走 auto = 必炸**，是 clone-and-run 的坑；当前生产不受影响。
  orb 档本身「留着别投入」，修法只求不炸，不优化这条链。
  **判据**：`force_constant_mode == "openfe_fixed"` 时 kr 门跳过（力常数是设置项，不该拿拟合上限筛），或把未写 `boresch_source` 的默认改成 `simple`（二选一，需维护者拍板）；
  补一条单测：`openfe_fixed` + 默认 `max_kr` 下 `estimate_multiple_anchors_from_trajectory` 返回非空。

- [ ] **PERF-0928 [P2] 默认主线上核实成立的性能项：先实测占比，再决定改哪条。**
  **先实测**：`ibs_engine.py` 的 `loop_timing_s`（integration/guard/cv_probe/ledger_io）已有入口，但**不覆盖** Context 构建、warmup 和分析期，A1/A2/A4 要另外计时。
  | 原编号 | 位置 | 核实后的机制 |
  |---|---|---|
  | A1 | `abfe_pipeline.py::_run_stage2_autonomous` 的 ANALYZE 轮与非 DONE 退出收尾；`_solve_merged_segments_if_any` | 窗口输出无进程内缓存，全量重读重解；有合并段时先 `run_once` 再合并解，前一个结果被丢弃。**只在 ANALYZE 轮和非 DONE 退出时发生，不是每轮**。退出那次全路径解承担逐窗支撑/σ/overlap 落盘，重组时保留 |
  | A2 | `ibs_engine.py::build_ibs_dual_system` 里的 `_common_system_xml = _serialize_ibs_common_system(...)` | 每窗无条件 clone + 序列化 U_common，常驻到窗口结束；`run_all_windows` 内部只在 `legacy_repair` 下消费。**另一个消费方** `abfe_pipeline.py::_run_independent_endpoint_segment` 也会读，改惰性时要保留 |
  | A11 | `run_all_windows` 生产循环里每个 update 对 K 个 `_f_{k}` 做 `getParameter` 并与 `production_f_k_lock` 比对 | 没有节流，也不计入 loop_timing。只有这一处是防篡改轮询；审查列的另外三处是计算本身要读的值，不动 |
  | A12 | 生产循环每 100 updates：`_enqueue_window_snapshot` 写完 npy 后，`_window_data_metadata` 又 `np.load` + `_sha256_file` 读回来 | 刚写的文件被读了两遍。修法**不能**用 `_array_content_fingerprint`（它哈希 `tobytes()`，和文件 sha256 不相等，而且注明只做诊断）；要在内存里对 npy 字节直接算 sha256 |
  | A13 | `ibs_engine.py::_save_coords_sidecar` | 每次 flush 都对累计 list 做 `np.stack` 再全量覆盖写，累计 O(n²)；默认开（`STAGE2_COORDS_EVERY=5`）。单位换算已经是向量化的，可以忽略 |
  | A4 | `run_all_windows` 开头的 `win_sys_xml = XmlSerializer.serialize(win_sys)` | 生产 checkpoint manifest 复用这一份，**进生产的窗口省不掉**。能省的只有两处：进生产前就退出的窗口；以及算完 sha256 后释放字符串 |
  | A5 | `build_ibs_dual_system` 的 `cv_template` 与 K 态循环 | 每窗 K+1 次力构建，模板用完即弃；每态重建排除表 set 做比对。`value_in_unit` 是纯 Python，不是 SWIG。shadow-coul 那处不在默认路径 |
  **判据**：每条有实测占比和处理结论（改 / 占比太小不改），改的那几条前后同体系同协议对照 ΔG 与账本逐字节一致。
  ⚠️ 09-26 审查的 PERF-01～05 与本条是两批，登记在 §2.2 `REVIEW-0926`（未核）。

### 2.7 代码复查（2026-09-29，已逐条对照源码核过）

> 来源：[archive/CODE_REVIEW_2026-09-29.md](archive/CODE_REVIEW_2026-09-29.md)（H/M/L 共 14 条，基于未提交工作区）。09-29 逐条核过：
> 描述大多属实，但 **H1、H2 两条「高」都是默认路径碰不到的潜伏项**，M3 的量级是推算且偏大一个数量级；
> 以下只收核实后的版本，**不要回去照原文修**。降级 / 驳回的在 §3.8。原文 §3「确认干净」和 ruff 结论**没复核**。

- [ ] **CI-RED-01 [P2] CPU CI 连续红，最近一次（09-26 run `36276233306`）是 `2 failed, 2840 passed, 6 xfailed`。两条都不是生产 bug。**
  ① `tests/test_doc_staleness_contract.py::test_snapshot_docs_are_not_stale`：`docs/STATUS.md` 第 5 行声明日期仍是「截至 2026-09-18」，阈值 3 天。
  ② `tests/test_nvcc_discovery_and_compile_failure_2026_09_15.py::test_a_compile_error_raises_instead_of_silently_using_cpu`：
  CI 的 OpenMM 没有 CUDA platform ⟹ `abfe_pipeline.py` 里 `getPlatformByName(requested_base)` 先抛 ⟹ 被判为资源类故障 ⟹ 走 CPU 回退 ⟹ 回退路径的 `openmm.Context`
  正好被测试 monkeypatch 成抛 `_Boom` ⟹ 逃出 `pytest.raises(RuntimeError)`。修的是测试，照同文件 resource 测试的写法把 `getPlatformByName` 也 stub 掉。
  ⚠️ 复查原文说「近 8 次全红都是这两条」，**只核了最近一次**；更早的 run 在 STATUS 还没过期时就已经红了，原因多半不同。
  🟢 **09-29 本地已修**：① STATUS.md 声明日期改成 09-28（09-28 那次只加了「跨软件对照」一节，头注写明了其余各节仍截至 09-18）；
  ② 照同文件 resource 测试的写法 stub 掉 `getPlatformByName`。
  同时把 `archive/CODE_REVIEW_2026-09-26.md`（未入库）里 84 处 `file.py:行号` 形式的链接改成 `file.py#L行号`：不改的话它一入库，`test_docs_internal_links_resolve` 就会红。
  本地全量 `-m "not needs_gpu"`：`2854 passed, 8 skipped, 6 xfailed, 0 failed`。
  **判据**：下一次推送的 CPU CI 为 0 failed（还没推，**未确认**）。

- [ ] **REVIEW-0929 [P2] 默认路径上核实成立、还没修的一条**（M1、M2 09-29 已修，见 [archive/TODO_closed.md](archive/TODO_closed.md)）。
  | 原编号 | 位置 | 核实后的机制 | 判据 |
  |---|---|---|---|
  | L8 | `tools/pocket_water/` 与 `tests/test_pocket_water_relocation_2026_09_28.py` | 只有 2 条测试（U0 差分对 OpenMM、stage-1 比值对独立重算），stage-2 延迟接受、`run_sampler` 的 apo ghost-ligand 路径、kabsch 跟踪都没有测试。口袋水线 09-27 已重新打开，S2 采样器开工前补 | stage-2 至少一条小体系测试：`min(1, exp(-β[ΔU−ΔU0]))` 与独立重算一致 |

### 2.8 代码复查 0930（Opus，已对抗核实）

> 来源：[archive/CODE_REVIEW_2026-09-30.md](archive/CODE_REVIEW_2026-09-30.md)（Opus 逐行全文重读 + 对全部 S2 做反方证伪，HEAD `e576a54` + 当时工作树）。**无 S1。**
> LCR 主线上的 `LCR-GAUGE`、`BURNIN-1FS` 两条 10-03 准确性复查又复现了一遍，登记在 §2.9 `ACCURACY-1003`（A4 / A3），这里不重复。
> 只影响旧控制器 / 实验路径的条目和 §2 未核 S3 在 §3.9。行号是审查当时的，**先按函数名重新定位**。

- [ ] **REVIEW-0930 [P2] 默认 / LCR 路径上核实成立、还没修的小项。**
  | 原编号 | 位置 | 核实后的机制 | 判据 |
  |---|---|---|---|
  | PME-RUNTIME-ID [S3] | `abfe_pipeline.py::_pme_u_kn_meta_payload`（写）与其兼容比较（只 pop 了 `code_sha256`） | meta 的 `runtime`（python / 内核 / openmm / pymbar 版本串）参与兼容比较 ⟹ 换节点必失配（`runs_lcr` 已见三种内核串）。只在 stage 1 结果缓存也未命中时走到，代价是 u_kn CPU 重算（~46 s），ΔG 不变、日志误报「模型版本不兼容」。又一处「非用户输入进身份」 | `runtime` 移出兼容比较（留在 meta 里作诊断），补一条换内核串仍命中的单测 |
  | REFINE-EXTRA-LAYER [S3] | `stage2_refine_extra_points_per_segment` 在 `_PREOPT_DERIVED_PATH_KEYS`（`abfe_pipeline.py`）与 `abfe_preoptimizer.py` 对应处 | 被划进 preopt 派生层 ⟹ resume 改此键时离线重算用旧 pilot、新值不生效、却写回新指纹。119 份生产 config 无一设此键；LCR 下只影响 TI 种子与信赖域半径用的 pilot dU/dλ | 改键后 resume 新值生效或明确拒绝，二选一并补测试 |
  | BORESCH-K-RESUME [S3, PLAUSIBLE] | `runabfe.py::resolve_boresch_restraint`：`boresch_{source}.json` 缓存命中先 return，`boresch_fixed_k_angle_kj_per_mol_rad2` 覆盖在其后 | `--resume` 命中缓存时新设 / 删掉的 K 覆盖都不生效，会原样返回旧 ΔG；非 `openfe_fixed` 档设此键被静默忽略。**现行使用约定：换 K 实验必须新开输出目录**（唯一真实用法 `kang1000` 就是这么做的） | 缓存命中时对比本次 K 与落盘 K，不一致 ⟹ 报错；非 `openfe_fixed` 档设了 ⟹ 报错 |

### 2.9 准确性复查 1003（合成复现）

> 来源：[archive/CODE_REVIEW_2026-10-03_accuracy.md](archive/CODE_REVIEW_2026-10-03_accuracy.md)（7 条；复现脚本 `tools/audit/core_accuracy_2026_10_03.py`，只在临时目录造合成产物）。
> 与 09-24 / 09-26 / 09-30 几轮审查重叠的条目在这里**合并成一份**，原条目改指这里。复查本身没有发现默认路径上的符号 / 单位 / Boresch 公式 / MBAR 代数错误；
> 也明确说了「不能由此推出剩下只能是力场」。修复方向的整体设计在 [design/PROPOSAL_stage2_accuracy_2026-10-03.md](design/PROPOSAL_stage2_accuracy_2026-10-03.md) §2（**未实施，动手前维护者拍板**）。

- [ ] **ACCURACY-1003 [P2] LCR / stage 2 采样身份、时间与口径缺口（合成复现确认仍在）。**
  | 编号 | 位置 | 机制 | 同一件事的旧登记 | 判据 |
  |---|---|---|---|---|
  | A1 账本不绑协议 | `abfe_pipeline.py::_stage2_rounds_ledger_load`（只比版本与 λ / 分窗布局）、`_run_stage2_rounds`（`SAMPLED` / `FAILED` 即跳过、已有 `evidence` 不重解）、`_load_ibs_window_outputs_merged`（不传 `expected_window_protocol`） | 复用输出目录、λ 布局不变但温度 / Boresch / softcore / 接力等身份变了 ⟹ 旧段原样复用（合成复现：换 kT 后新增引擎调用 0） | `REVIEW-0926` #1；09-30 `RESCUE-PROTO-NONE`（范围扩到所有走合并加载的路径） | 账本绑定阶段采样协议 + 已提交生产盒；读 SAMPLED 段与合并加载都校验；身份变了拒绝混用、**不许重贴当前指纹「升级」旧段**；补换协议后 resume 的测试 |
  | A3 burn-in 实为 1 fs | `ibs_engine.py`：新窗口爬坡设 1 fs（`if not is_resumed_ibs:`）→ 固定 f_k burn-in 沿用 → 生产前才恢复 `original_dt`；分块 `max(1, ceil(total/block))` 每块跑满 | 默认 25000 步 = **25 ps**（设计 / 注释写 50 ps），5000 步一块 = 5 ps；resume 跳过爬坡的分支跑在 2 fs ⟹ 同 config 两种物理时长；7000 → 10000、0 → 5000 的取整 | 09-30 `BURNIN-1FS`；`REVIEW-0926` #6 | 步长契约写死并记录（requested / actual / dt）；新起与 resume 同一时长；最后一块按剩余步数、0 步跳过、负值报错 |
  | A4 补段中心口径 | `stage2_rounds_evidence.py::solve_window`（用含 LJ 尾项的目标能量解 `merged_f_k_kJ`）→ `stage2_rounds.py::_next_center` 直接当下一轮固定 f_k | 每窗首末尾项差 1.2–1.3 kT ⟹ 每个补段等于带 ~1.3 kT 线性倾斜，占据末 / 首比多 3.3–3.7×；信赖域投影同样口径不一致。**ΔG 不受影响**，影响补段帧分配 | 09-30 `LCR-GAUGE` [S2] | 证据层另解 `sampling_f_k`（从 `sampling_states`）给调度用，`merged_f_k_kJ` 保留原含义作报告 |
  | A5 生产盒校验缺口 | `runabfe.py` 离线分析回退构造的 `expected_window_protocol` 不带期望盒体积；`ibs_engine.py` 盒体积门 `current_box_volume_nm3 is None` 即放行；`abfe_pipeline.py` 完成结果早退绕过逐窗盒门 | 复用 / 重建同一输出目录且已提交生产盒与旧产物不同时，旧盒产物被当作当前盒结果 | `REVIEW-0926` #2 #3；`REVIEW-0924` 凌晨份 #4 | 期望盒从持久化的已提交状态取，聚合缓存与 raw-window fallback 两处都校验；拿不到 ⟹ 标未验证 |
  | A7 不完整路径 σ 配错窗 | `abfe_pipeline.py` finalize 按局部 `window_index` 建表、按真实窗口号取值（`ibs_engine.py` 协方差链同时存 `window_index` / `source_window_index`） | 前面窗口去相关后被跳过时 σ 错位、后面窗口漏 jackknife 增量。只在已是 `ANALYSIS_INCOMPLETE` 时触发 | `REVIEW-0926` #4 | 改按 `source_window_index`（或 `_segment_physical_window`），补一条跳窗的单测 |

- [ ] **COVERAGE-1003 [P2] 共同漏采的构象盆地可以带着 `CONVERGED` 标签通过（收敛证据的覆盖限制，不是 MBAR 错误）。**
  **位置**：`stage2_rounds_evidence.py` 的逐窗证据 + `stage2_rounds.py` 的标签规则；执行层把贡献过帧的段数当 `n_independent_seeds`。
  **机制**：标签只看 ESS、top-1%、留一段差异、种子数，看不到所有段共同没去过的构象区。精确双盆反例：三段全困在 A ⟹ `CONVERGED`、σ_jk=0、估计 0 vs 真值 −18.2 kJ（合成模型，**不是任何真实体系的误差估计**）。
  真机上对应的形状：HSP90 w0–w2 每段 0.5 ns 内不换盆（[STATUS.md](STATUS.md)「当前现状」）。
  ⚠️ **不许**靠调低 ESS 阈值、或从同一起点多跑短段来宣称解决。
  **判据**：维护者对 [design/PROPOSAL_stage2_accuracy_2026-10-03.md](design/PROPOSAL_stage2_accuracy_2026-10-03.md) §3–§4（连续链 + 构象覆盖证据 `sampling_assessment`）拍板；
  落地前，报告 / 文档里 `CONVERGED` 的解释限定为「现有观测数据通过所列判据」，构象覆盖记为未验证。

### 2.10 fragment-opt 精度线（10-02 起的工作重心）

> 背景与数字：[STATUS.md](STATUS.md)「当前现状」、[HANDOFF_2026-10-02_bias_precision_boresch.md](HANDOFF_2026-10-02_bias_precision_boresch.md) §4 / §6。
> 数据目录 `fragment-opt-abfe-benchmark/openmmIBS/`，**当实验数据只读**，新跑的输出放新目录。

- [ ] **FRAG-S1LONG-01 [P2] PWWP1 L9：加长去电荷能否压下复合物 stage 1 的 rep 间散布。**
  **现状**：`runs_lcr/` 下 PWWP1 L9 / L10（开了 125000 步平衡段）复合物 stage 1 SD 4.9 / 3.4 kJ，是两体系最大的分项（stage 2 只有 1–2 kJ）。
  10-03 起在跑 `runs_lcr_s1long/PWWP1_ligand-9/`（config `configs/PWWP1_ligand-9_lcr_s1long.json`：`stage1_remd_equilibration_steps: 250000` + `stage1_n_steps_per_window: 1000000` ⟹ 平衡 0.5 ns + 生产 2 ns/态，stage 2 不变，3 rep 串行，每 rep 约 3.5 h）。
  **判据**：3 rep 跑完，复合物 stage 1 SD 与 4.9 kJ 对比给结论（明显降 ⟹ 整批把 stage 1 加长；不降 ⟹ 按残基归因，`tools/diagnostics/decharging_group_attribution.py`）。

- [ ] **FRAG-BATCH-01 [P2] fragment-opt 整批：逐体系「我们 vs 论文」。**
  **现状**：10-02 已给 59 份 `configs/*_lcr.json` 加了 `stage1_remd_equilibration_steps: 125000`（另有 `_comment_s1eq` 一行；直接改的 config，用 `make_configs.py` 重新生成会丢这个键）。跑过的：HSP90 L3 / L26 / L27 / L30、PWWP1 L9 / L10。整批命令在维护者手里（`run_<sys>_lcr.zsh`，config 路径写死）。
  **不开接力**（`RELAY-01`）。`FRAG-S1LONG-01` 出结论后可能要再改 stage 1 时长。
  **判据**：各体系 3 rep 跑完，按论文 `free_energy_data/abfe.csv` 的分项（`comp_vdw` / `comp_coul` / `lig_*`）逐段对照，回答「差是否系统性、出在哪一段」；注意论文 4 fs + HMR、我们 2 fs。

---

## 3. P3 · 现在明确不做

### 3.1 已定位、判定不改（`AUDIT-01`~`07` [P3]）

> 出自 09-09 全仓审计（62 条候选，52 条已修），完整记录见 [archive/AUDIT_2026-09-09_full_repo.md](archive/AUDIT_2026-09-09_full_repo.md)。

- **AUDIT-01 [P3]** `abfe_core.minimum_image_displacement_nm`：候选立方体随长宽比增长，`radius > 64` 的 guard 在分配 `N × 2.1e6 × 3 × 8` 字节之后才触发。
  *不修*：要极端长宽比 + 大批量才触发；`design/` 的盒型识别提案覆盖这一片。
- **AUDIT-02 [P3]** 膜 leaflet 的 wrapped/unwrapped 混用（`_protein_leaflet_cross_sections_nm2`、`assign_lipid_leaflets`、`verify_membrane_normal_axis`）。
  *不修*：只影响膜路径，留给膜线单独一轮。
- **AUDIT-03 [P3]** `free_energy_engine.run_independent_windows` 保留全部帧（≈14 GB）。*不修*：无生产调用者，接线前必须先改。
- **AUDIT-04 [P3]** `apbs_correction._read_dx_values` 257³ 网格约 1 GB 峰值。*不修*：APBS 不在主线；改法是 `np.frombuffer`（~136 MB）。
- **AUDIT-05 [P3]** `abfe_core.OnlineConvergenceMonitor` K==1 会抛、`n_k_array` 与 `u_kn` 列数不一致。*不修*：`abfe_core` 之外无调用者。
- **AUDIT-06 [P3]** `TraditionalABFEPipeline.pre_equilibration_identity_fingerprint` 引用从未赋值的 `self.pressure` / `self.barostat_protocol`。
  *不修*：全仓无活的调用点。（09-24：`barostat_protocol` 那一半已随测试修复改成 `getattr` 兜底，不再抛；`self.pressure` 仍未赋值。）
- **AUDIT-07 [P3]** `abfe_pipeline._rebalance_fingerprint` 没有 System 身份绑定 ⟹ System 变了而锚点没变时 `rebalance.chk` 会被复用。
  *🛑 明确不修*：唯一修法是把自产产物的 sha256 放进缓存身份，**用户否决过 4 次**（08-24 `code_sha256`、09-09 `system_xml_hash`+`positions_sha256`、09-09 `preopt_cache_sha256`）。
  算法变了由显式协议版本号承担。**不要再提这个方案。**

### 3.2 benchmark 里已确认「不是 bug」的（`BM-A` [P3]）

- **BM-A [P3] 求解器缺窗 4 个是真采样不足**，门抓对了：`p38_ligand1/rep2` win5 去相关后 5 帧 g=100.1；`cmet_ligand2/rep1` 跳 [3,4]，win3 g=82.6；
  `p38_ligand2/rep3` 跳 [4] g=60.0；`cmet_ligand1/rep2` 跳 [0] g=75.4。要治就加采样或缩窗跨度，不是改门。
- `cmet_ligand1/rep3` attachment 前后半程漂移 −0.7592 > 容差 0.5373：真·系综未收敛。
- `cmet_ligand2/rep3`（09-09 旧 run）50/50 次权重更新才判不收敛，预算给足了。
- `cyclod_ligand2/rep1_evidence` / `_run2`（09-11 旧 run）：前者已被后者取代；后者当时是真 bug（预算耗尽被写成 f_k 错了），**现已修**
  （走 `IBSValidationBudgetIndeterminateError` + `halt_budget_no_steps_available`）。
- **`Particle coordinate is NaN` 11 次**：
  - **10 次是 09-09 单批事故**（14:28–16:49 启动的 brd4/cmet/cyclod 共 10 个 run，全死在偏置爬坡第一档 `scale=0.2`，无守卫）。
    守卫 `step_with_chunk_rollback` 09-12 落地（commit `1769e71`）⟹ **已修**，旧 traceback 只是留在追加式 `launch.log` 里。
    ⚠️ `ibs_engine.py` 注释里的 `🔑 [2026-09-09]` 是诊断日期不是落地日期。
  - **1 次是已知设计**（`cyclod_ligand2/rep3`，09-14）：偏置**学习**循环用的是 `guarded_step`，按设计只换错误信息、不改动力学。
    不是 `LR-06` 的鬼影期（学习循环跑在 `bias_scale=1.0`）。
  - 📌 **要给学习循环加回滚的前提**：回滚坐标必须连 `energy_buffer` / `energy_history` / `sampling_state_energy_history` 一起回滚，
    否则偏置学习读到与构型对不上的样本。**不许当成"顺手补个守卫"。**
    判据钉在 `tests/test_warmup_guard_asymmetry_2026_09_16.py`。

### 3.3 膜受体–配体路线 [P3]（整条线暂停）

- [ ] **MEM-LINE [P3] 膜线整条暂停**：09-12 起暂停，停在 C4（带电膜双腿接线 smoke test）。当前主线是可溶体系。
  暂停时的完整清单（C4/C5、A5 目标膜输入、restraint 账目、膜生产协议、Phase D、Definition of Done）原样冻结在
  [archive/MEMBRANE_LINE_PLAN_2026-08-11.md](archive/MEMBRANE_LINE_PLAN_2026-08-11.md)。
  **重启时**：先对源码逐框核一遍再搬回来定等级；C4 的受体/配体组合要重新选（旧候选集作废，**不许代理自选**）；
  膜体系生产盒目前 = NPT 末帧盒（没走均值缩放），要先处理。

- [ ] **PHY-03 [P3] charge-transfer 的 tethered charge carrier 不能按当前论证严格跨腿抵消**（带电路线的条件性阻塞）。
  **位置**：`abfe_core.py` 的 co-ion restraint 说明与表达式；`ibs_engine.py::_create_co_alchemical_ion_restraint`；
  `abfe_core.py::resolve_charge_treatment` 的 `closes_thermodynamic_cycle`。
  **触发**：带净电配体用 `co_alchemical_charge_transfer`。
  - **配分函数**：代码以「两腿同一锚点规则、同一 k/r0」推断 restraint 自由能抵消。但受限 carrier 的配分函数含
    `∫exp[-β(U_env(r)+U_rest(r−r_anchor))]dr`，两腿的 `U_env`、排除体积、anchor 系综都不同；λ=0 时 carrier 带真实电荷，
    restraint 与 carrier 溶剂化分不开。
  - **barostat**：`dx0/dy0/dz0` 是冻结的笛卡尔参数，盒缩放时不跟着缩 ⟹「井心随盒缩放」的注释不成立。
  - 已标 `production_qualified=False`（这条边界要保留），但同时写 `closes_thermodynamic_cycle=True` 过度承诺。
  **要求**：给出含 carrier restraint / 标准态 / 环境项的完整循环推导；证不出解析抵消就显式算两腿修正；参考位移改成随盒变化的分数坐标，
  或换成有解析标准态修正的相对约束。
  **判据**：carrier 平移、anchor 选择、盒尺寸、各向异性缩放、restraint 强度扫描后，修正后 ΔG 在统计误差内不变；
  两腿 reservoir 端点有独立 closure test。C4/C5 过之前不得把数值提升为生产结果。

### 3.4 发布工程门里押后的三条（原 `REL-05` / `REL-06` / `REL-10`）

> 触发条件是**同一个**：方法定稿。在那之前做都是白做（要再改一遍、图要过期）。触发后改成 `[P2]` 挪到 §2.5。

- [ ] **REL-05 [P3] 代码内的中文切成英文**（末期收尾项）。09-16 重测：已跟踪 `.py` 271 个 / 176,467 行；含中文 37,360 行（237 个文件）；
  **同行含 `raise`/`logger`/`print` 的 1,673 行（46 个文件）**，这是用户直接看得见、非切不可的部分，注释与 docstring 可以最后再动。
  ⚠️ 数字每次改动都在涨，引用前先重测。⚠️ 异常文本在大量测试里被 `match=` 断言，批量改会和在推的改动撞车。
- [ ] **REL-06 [P3] 主页挂一张流程图。** 三份 README 的 `![` 计数都是 0；`docs/current-pipeline.svg` 现成但会过期，方法定稿后重新出图再挂。
- [ ] **REL-10 [P3] 软件版本号与变更说明**（[RELEASE_READINESS:178](archive/RELEASE_READINESS_2026-08-31.md)）。LICENSE（MIT + NOTICE）已有，
  只欠版本号本身：冻一个版本、写清支持范围。

> 🛑 **两条明确不做**，理由别重新论证：
> - **`CITATION.cff` / 引用信息**：方法还没做完，现在写引用就是承诺一个还不存在的东西。
> - **英文文档追平中文**：中文是主文档，这是既定选择；`README.md`（英文）覆盖完整流程即可，不做逐字对等。

### 3.5 独立端点段重新打开时才会触发（2026-09-23 从 P1 降级）

- [ ] **COMBINE-01 [P3] `combine_ibs_and_independent_endpoint` 不产出顶层 `analysis_status` ⟹ `--analyze-only` 恢复 vanishing 腿必然抛错**
  （09-16 测试对账挖出，真机未复现）。
  **触发**：重新打开 `stage2_independent_endpoint`（默认关、复现路线不用；拼接只在 `runabfe._analyze_dual_leg_artifacts` 的 `if endpoint_enabled:` 下调用）。
  **因果链**：09-15 `converged`→`analysis_status` 改造**显式排除了**这个函数，它只把 `analysis_status` 埋进 `ibs_segment`、顶层没有；
  vdw 腿强制走拼接，拼完立刻 `_assert_stage_result_sane` ⟹ fail-closed。生产侧 `abfe_pipeline.py` 用拼接结果整个替换 `stage_result`，同一个洞。
  **为什么不是补一行 passthrough**：端点段自己的 `converged` 里混着阈值门（`min_overlap`），原样抄过去就是把退役的阈值塞回接受判据。
  ⚠️ **需要维护者拍板**端点段完整性用哪个量表达。
  **判据**：`tests/test_issues_67_27_75_83.py` 的 5 条 + `tests/test_independent_endpoint_sampling.py::test_combined_result_passes_the_pipeline_stage_gate`
  共 6 条 `strict-xfail` 变红（修好即自动变红），届时删掉标记。

### 3.6 押后到重构（2026-09-24）

- [ ] **TEST-02 [P3] 测试集从 2810 条压到 ~1300 条，生产代码行覆盖一行不掉。**
  **方案、判定规则、09-24 分类快照、重算命令**：[design/PLAN_test_suite_pruning_2026-09-24.md](design/PLAN_test_suite_pruning_2026-09-24.md)。
  量级：逐文件最小覆盖集 1235 条必留；503 条一行生产代码都不跑（多为读源码字符串断言）；1072 条覆盖行是同文件别的测试的子集（**候选**，行覆盖相同 ≠ 断言冗余，要逐条判）。
  同批已落地：提速 263 s → 40 s（xdist 默认并行等，见方案 §1），删 EXP-012/019/020 测试 11 个文件 / 70 条。
  **触发**：重构。先重构再压——分类是按当前覆盖行算的，重构后要**重算**，现在删并会返工。
  ⚠️ 动手前先把 `tests/` 的未提交改动提交或打快照：大量日期文件未入库，删了找不回（09-24 权限分类器就是因此拦下的）。
  **判据**：全量绿、xfail/skip 数不变；收集数 ≤ 1500；压后生产行覆盖数不低于压前（重构后的新基线）。

### 3.7 性能审查 0928 里不在默认路径 / 已驳回的条目（2026-09-28 核实）

> 来源同 §2.6。**默认路径上成立的**在 §2.6 `PERF-0928`；这里的都**不修**，除非对应的触发条件出现。理由别重新论证。

- [ ] **PERF-0928-COND [P3] 机制属实，但只在非默认分支上触发。触发时顺手改。**
  | 原编号 | 触发条件 | 核实要点 |
  |---|---|---|
  | A7 | stage1 未完成就 resume（REMD 中途崩，或缓存失配） | `_is_traj_valid` 逐记录读到 EOF，abfe_pipeline 侧没有记忆化；真正读两遍的是「指纹匹配但 DCD 坏了」。审查写反了一处：runabfe 里有记忆化的是**预平衡 DCD**（`_TRAJ_STRICT_VALIDITY_CACHE`），checkpoint 是故意不缓存的。审查建议改成只看头部 + 文件大小，这和 docstring 里 fail-closed 的设计冲突，**要维护者拍板** |
  | A9 | 控制器 RECALIBRATE_FK / PROBE_CANDIDATE_FK / PROBE_REANCHOR_EPOCH | 已按 `only_windows` 只解 decide() 点名的窗口，不是全窗；只是同输入重复解，没有记忆化 |
  | A10 | `boresch_source=auto`（Orb 链） | 逐候选拷贝全轨迹 float64、逐帧 Python 二面角、MACE 口袋力逐候选重算；`openfe_fixed` 下 q/Fq 算完不被读。依附 `BORESCH-AUTO-01`，orb 档不投入 |
  | A8 | `--mode traditional` | `run_traditional_mode` 复合物腿每次都 `build_system_from_gromacs`，没走 native cache |
  | A14 | `TraditionalMBARAnalyzer` + 非 PME 腿（有 LRC） | `_compute_u_kn_chunk` 里盒体积放在 k 循环内，K 倍冗余 |
  | B1 / C6 | `--outer-lambda-local-residual-ibs` | B1：残差值已经在首次 CV 数组里，却又调一次 `getCollectiveVariableValues`；C6：每窗重建一份逐字节相同的力 XML |
  | C4 / C5 | 同上开关，**且**冻结 manifest 不覆盖当前配体（闭式重训，不是已摘线的 autofit） | C4：岭网格对与 ridge 无关的二次型重复计算；C5：约 251 帧 × 9 态串行 PME 单点能 |
  | B2 | `stage2_independent_endpoint` + `--analyze` | 读侧对 bank 全量重算 sha256。只在 `--analyze` 时发生，不是每次 resume |
  | B5 | legacy anchor 模式 | Python 双层循环做 unwrap；openfe 档只有 1 列，影响可以忽略 |
  | C1 | 口袋水 S1 探针（离线工具） | 每个水、每帧都 `updateParametersInContext`；但同帧的试探能量评估有几万次，它未必是瓶颈，P1 没有依据。S2 采样器开工前实测再说 |
  | C2 / C3 | APBS 修正（只服务膜线，膜线暂停） | DX 网格逐 token 正则解析；各快照串行跑子进程 |
  **判据**：无单独判据，随各自工作线重新打开时一并处理。

**驳回**（理由别重新论证）：
- **A6**（warmup 续采每 250 步重写 ibs_state）：实际是攒满 40 帧（约 1 万步）才写一次，只在 resume 且 `calibrated_pending_validation` 时触发；历史长度有截断（`_TMBAR_SLIDING_WINDOW`、`max_bias_updates`），不是 O(累计)。
- **B3**（`--only-complex-charging` 二次 MBAR）：raw 全帧 MBAR + 相邻 BAR 是另一种估计量，是**有意的交叉验收门**，不是重复计算。
- **B4**（fixed-H 探针每边 fsync）：`_persist_fixed_h_probe_edge` / `_probe_vdw_window_fixed_overlap` 在活代码里**没有调用方**（测试甚至断言它不在），是死代码。
- **C7**（mace_graph 逐原子 `.item()`）：`build_mace_graph` 在仓库里没有调用方，是死代码；审查说的「每图重算 SHA」也不对（`verify_sources` 默认 False）。
- B4、C7 的死代码删除归 §3.6 重构一起做。

### 3.8 代码复查 0929 里降级 / 驳回的条目（2026-09-29 核实）

> 来源同 §2.7。这里的都**不修**，除非对应的触发条件出现。理由别重新论证。

- [ ] **REVIEW-0929-COND [P3] 机制属实，但默认路径碰不到或收益可忽略。触发时顺手改。**
  | 原编号 | 触发条件 | 核实要点 |
  |---|---|---|
  | H1 | 重新打开 `stage2_independent_endpoint` | **就是 §3.5 `COMBINE-01`**，不重复登记。复查原文说「`--analyze-only` 恢复 vanishing 腿今天是断的」、「发布前必修」，**不对**：开关 09-01 起默认关（`abfe_pipeline.py::_run_dual_lambda_stage` 里 `_endpoint_requested`），默认配置下不走拼接 |
  | H2 | 留一法求解子集时抛 `MultiSegmentInputError` | `stage2_rounds_evidence.py` 留一法循环的 `except (…, ValueError)` 确实会连 `MultiSegmentInputError` 一起吞掉 ⟹ 写 nan ⟹ `single_segment_dependent` ⟹ 补段。但复查写的「输入账本损坏被静默降级」**碰不到**：完整求解 `solve_window` 在前面、没有包 try，账本坏了在那里就直接炸；留一法的输入是合法输入的子集，形状、有限性检查不会单独失败。和设计 A11（「解不出来 ⟹ fail-closed」）口径不一致，改的话把 `MultiSegmentInputError` 从捕获列表里剔掉即可 |
  | M4 | GPU Context 构建失败、被判为资源类，并且 CPU Context 也失败 | `abfe_pipeline.py` CPU 回退路径的 `openmm.Context(...)` 不在 try 里，会裸抛、不带上下文。复查说它是 CI nvcc 测试失败的「共因」：实际修法在测试侧，见 `CI-RED-01` |
  | L6 | outer-λ 生产路径再出 NaN、需要诊断现场时 | `step_guard.py` 的 `guarded_minimize` / `guarded_set_parameter` / `guarded_deserialize` 等在生产代码里没有调用点；裸 `.step()`：`outer_lambda_neural_basis.py` 3 处、`free_energy_engine.py` 2 处 |
  | L1 | 壳体积 ≈ 0，或 bounding box 远大于壳 | `tools/pocket_water/relocation.py::uniform_in_shell` 的 while 没有迭代上限 |
  | L7 | 别处开始假设 `seed_id` 全局唯一 | `stage2_rounds.py::first_round_specs` 每窗从 0 计（0/1/2 跨窗复用）；逐窗证据层不受影响，写进 docstring 即可 |
  **判据**：无单独判据，随各自工作线重新打开时一并处理。

**驳回**（理由别重新论证）：
- **M3**（`RelocationMC` 的 V_R/V_S 用 MC 积分，误差进接受判据）：机制属实（常数比值偏了 ⟹ 进出方向的平稳分布整体平移 ≈ ε kT），但复查写的「1–2% 相对误差、0.02–0.04 kT」是推算且偏大：2M 样本、命中率约 0.3 时 ε ≈ 0.1%，约 1e-3 kT/水，可以忽略。
- **L2**（`np.log(rng.random())` 取到 0 ⟹ -inf ⟹ 无条件接受）：概率约 2⁻⁵³，复查自己也判不值得。
- **L4**（新预检遇到旧运行目录缺 `ligand_indices_solvent.json` 时裸抛 `FileNotFoundError`）：**碰不到**。1.5 步的溶剂腿缓存完整性检查已经把这个文件列为必需（`runabfe.py` 缓存路径表 + `missing` 列表），走到预检之前就会被拦下或重建。
- **L5**（`_check_boresch_geometry_safe` 没查二面角 φ 的奇异性）：φA、φB 的奇异点就是 θA、θB 共线，已经被 sin 下限覆盖；只有 φC 用到的配体内部角 b1-b2-b3 不在运行时查，而那个角是选锚点时定的。
- **L3**（`final_state.npz` 存 float32）：离线工具的续跑起点，精度够用，复查自己也判不值得。

### 3.9 代码复查 0930 里只影响旧控制器 / 实验路径的条目，以及未核的 S3（2026-10-03 登记）

> 来源同 §2.8。旧自治控制器已关、LCR 开时这些路径不可达；**重新打开对应路线时再处理**。理由别重新论证。

- [ ] **REVIEW-0930-COND [P3] 触发条件出现时顺手改。**
  | 原编号 | 判定 | 触发条件 | 摘要 |
  |---|---|---|---|
  | FINAL-ANALYZE-REWINDOW | CONFIRMED · S2 | 重新打开旧自治控制器 | 收尾全路径 ANALYZE 不读 rewindow 台账 ⟹ 最终 `stage2_vanishing.json` 不含子系综、rewindow 修复被丢；父窗原本被跳过时整条腿 `ANALYSIS_INCOMPLETE` |
  | PROVISIONAL-WASH | CONFIRMED · S2 | 同上 | 临时生产窗口经补帧或 resume 复验被洗成 `converged` / `verified`，`warmup_failure.json` 被删；ΔG 数值不变、证据标签失真。LCR 固定 f_k 不受影响 |
  | RESCUE-PROTO-NONE | PLAUSIBLE · S3 | 同目录改哈密顿量键再 resume 且前一跑是旧控制器 | `"vanishing_rescue"` 取协议键得 None ⟹ 盒 / 协议门放行。合并加载那半并进 §2.9 A1 |
  | RECAL-FK-GAUGE | CONFIRMED · S3 | 用到 `recalibrate_f_k_from_production` 的旧路径 | 与 `LCR-GAUGE` 同形状（energies 口径解 f_k）；口径问题，**不涉及符号** |
  | INSERT-LAMBDA-ARGS | PLAUSIBLE · S3 | 非 LCR + 显式 `metric_integral` | 自治 INSERT_LAMBDA 漏传 `partition_criterion` / `pilot_metric_g` |
  | SPLIT-TAIL-ASSERT | PLAUSIBLE · S3 | 旧控制器 SPLIT_TAIL_WINDOW | 缩跨度断言比尾段整体最大 ∫g，行内注释**明文选择**了这个口径 ⟹ 设计取舍，要维护者拍板，不是写错 |
  | OUTER-SCORE-ID | PLAUSIBLE · S3 | 开 outer-λ local residual | `sampling_score_sha256` 含绝对路径与插件 `.so` sha256 ⟹ 自动重训后 HALT |
  | PLATFORM-ID | PLAUSIBLE（潜伏） | 改 platform 名 / 设 `max_resident_contexts` | REMD 采样指纹与窗口身份含 platform（带设备号）与 `max_resident_contexts` ⟹ stage 1 与全部 stage 2 窗口一起失配 |
  | SHADOW-BRIDGE-BORESCH | CONFIRMED · S3 | `--decharge-method shadow_ibs` | Shadow-Bridge REMD 副本从不设 `lambda_boresch_scale` ⟹ 复合物腿在 Boresch 关着时采样。其余所有生产 Context 都核过是对的 |
  **未核的 S3**（原文 §2，约 35 条，含 LCR 段全 FAILED 后 resume 粘住、REMD 不作废旧 sampling meta、预平衡续跑剩 0 步退回末帧盒等）与死代码清单：**没对抗核实，不进待办**；做到相关代码时去原文查，按 §2.2 的规矩先核再修。

### 3.10 模块审计 0928 的测试缺口（2026-10-03 登记）

- [ ] **TEST-GAP-0928 [P3] [archive/MODULE_AUDIT_2026-09-28.md](archive/MODULE_AUDIT_2026-09-28.md) §3 的 TG-1～TG-8（顶层公共符号 → tests/ 零引用）。**
  ⚠️ 09-30 复查的异议先看：**TG-1 前提错**（PME 自能修正生产硬编码 False、已因物理错误撤销，该支是死代码 ⟹ 该删 / 标死代码，不是补测试）；
  TG-3 说 `delta_u_distribution_diagnostics`、`synthetic_mbar_u_kn` 零引用不对；§2.2「`_quantize_positions_for_identity` / `_q8` 口径一致」不能作已查干净的证据（零调用方）。
  **为什么 P3**：测试缺口不是缺陷；TG-2（冻结 f_k 防篡改守卫）、TG-4（修正项与周期统计）随对应代码再动时补。
  **判据**：无单独判据，随 §3.6 重构或对应模块下次改动时一并处理。

### 3.11 已关的实验开关

- **RELAY-01 [P3] `stage2_lcr_relay`（stage 2 同窗段间接力，10-02 落地）不要开。**
  真机 HSP90_ligand-27 × 3（对照组只开 stage 1 平衡段）：复合物 stage 2 61.0 → 31.0 kJ，ΔG_bind 离论文更远；**原因没查**（维护者 10-02 说不分析漂移）。
  另有统计问题：接力段仍按独立种子计数（`_run_stage2_rounds` 把段数传给 `n_independent_seeds`），段间 jackknife σ 偏乐观（10-03 准确性复查 #6）。
  连续链的正确语义（恢复完整 checkpoint、不重新热化）在 [design/PROPOSAL_stage2_accuracy_2026-10-03.md](design/PROPOSAL_stage2_accuracy_2026-10-03.md) §3.1，**不是这个开关**。
  **重新打开的前提**：先查清 26 kJ（HANDOFF_2026-10-02 §4.3 给了最便宜的查法），并把段计数改成按链。理由别重新论证。

---

## 4. 不许回退的约定

拿来**否决别人改动**的规矩。每条都是真机或对账咬过的。

**通用**
- **同一个不变量只能有一份实现。** 本仓最贵的复发模式（λ 身份曾有四份）。
- **算对之前，沉没成本一律不计。** 先问「这条路径算得对吗」，再问「重算要多少 GPU」。
- **`abfe_config.json` 只是一份样例**，不是仓库默认值。某次运行里某键是什么值，只看该 run 的 `run_provenance.json`。
- **进指纹**：一个开关有几条进指纹的路径就得收窄几条；**顶层 run 指纹与 stage 指纹口径不同**，顶层无条件进是对的，别跟着 stage 那侧"统一"。
- **不得为了得到期待的标签而换误差估计器**（estimator shopping）。要独立证据就换 Epoch / 重采。
- **四个量不许混用**：`n_decorr` 是求解器**资格**；`N_eff,k / g_k` 是**主验收量**（门 = 10）；`top1% weight` 是否决**警报**；occupancy / coverage ESS **只诊断 f_k 训练**，永不决定 ΔG 是否有效。
- **低支撑永远不是 `FAIL`**，是「尚不可测」，动作是加预算；`FAIL` 是支撑够了却违反统计门。**缺证据 ≠ 通过，也 ≠ 有问题**，是 `UNKNOWN`，动作是去产出证据。
- **累计 f_k 残差只能用 `sampling_states`**（`energies` 多一个逐 λ 态的 LRC 常数，会伪造「全负号单调」）；`N_eff/g` 对它不变，两者别"统一"。**`(K−1)×门槛` 不许写**，累计偏差门不随边数增长。
- **路由信号不得炸出流水线**：`LOCAL_VALIDATION_CAP` / `INSUFFICIENT_DATA` / `CUMULATIVE_FK_MISALIGNMENT` / `SKIPPED_WINDOW` / `HALT_BUDGET` 都是路由信号。真终态清单**只看** `abfe_preoptimizer.py` 里 `EXIT_SPECS` 的 `terminal=True`（`TERMINAL_EXITS` 由它派生；个数一直在变，09-16 是 9 个、09-24 是 11 个），本文不维护第二份。唯一有功效的否决是 `IBSFrozenCalibrationValidationError`，也只封存该候选。
- **测试选择口径用收集数对账**，不能 grep 文件。

**控制器相关**（控制器目前关闭；重新打开时全部适用）
- **一个 stage 只许有一个控制器**；控制器只读，执行由独立的 execute 层落盘。
- **旧修复机制只许关、不许删**（path_evolution 修复分支 / production rescue / rescue 后重标定），等自治路线再跑通几次再说。
- **局部动作只载它要动的窗口。** 插 λ 到下游重采完成之间，下游产物描述上一套布局，这是**合法中间态**。
- **停滞保护的降级必须走 `decide()` 的守卫**，不许直接改写 `act`。**可行性与可落性是两问**，插 λ 两侧都要判。
- **死动作要接在各自兜底之前**，不是在链尾加分支；**一个父窗只切一层**（不许放宽）。
- 收拢代码的判据是「写盘的留 pipeline」，不是「塞进一个文件」。

**设计依据**（live 参考，不是待办）：
[STAGE2_CONTROLLER_DESIGN_2026-09-12.md](archive/STAGE2_CONTROLLER_DESIGN_2026-09-12.md)（设计 + 十条真机陷阱，接手先读）·
[PLAN_PATH_REPAIR_2026-09-11.md](archive/PLAN_PATH_REPAIR_2026-09-11.md)（分支编号 5a/5b/6 的出处，代码 10 余处引它）·
[STAGE2_ROOT_CAUSE_2026-08-28.md](archive/STAGE2_ROOT_CAUSE_2026-08-28.md)（**所有收敛门对单轨迹重加权是瞎的**：门全绿时 ΔG 错 42 kJ/mol）。

---

## 往这里加条目的规则

1. 一条待办只出现一次。专题文档讲"为什么"，本文讲"还欠什么、欠在哪一行"。
2. 新条目必须带**等级**（编号后 `[P1]`/`[P2]`/`[P3]`，放进对应的 §1/§2/§3）、**位置**（`文件:行号` 或函数名）和**判据**（怎样算做完）。写不出判据的是想法，去 `design/`。
3. **改优先级 = 改方括号 + 把整段挪到对应节**，不许只改一处。
4. 关闭一条就**整段移进** [archive/TODO_closed.md](archive/TODO_closed.md)，写清结论去了哪，不在这里留 `[x]`。
5. 标"不修"的写明理由，并注明**理由别重新论证**。
6. 科学结论进 [STATUS.md](STATUS.md)；代码/协议变更进 [CHANGELOG.md](CHANGELOG.md)；实测表与重放证据进 `archive/`。
