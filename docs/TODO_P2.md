# 待办 · P2 —— 该做，但不挡路

[项目入口](../README.md) · [文档导航](README.md) · [当前科学状态](STATUS.md) · [变更记录](CHANGELOG.md) · [P1 在推的](TODO.md) · [P3 现在不做](TODO_P3.md)

> **已关闭的条目不留在本文**，整段移进 `archive/` —— 2026-09-17 关闭的 5 条
> （`TEST-01` / `LR-03` / `LR-04` / `REL-08` / `REL-09`）在
> [archive/TODO_closed_2026-09-17.md](archive/TODO_closed_2026-09-17.md)。

> 手上没有 P1 的时候才来看。条目的位置/判据格式与 `TODO.md` 相同。
> **P2 的判据是「不挡路」，不是「不重要」** —— `BUD-06` / `IDENT-01` 在这里，
> 是因为两条都自证了今天不阻塞任何在推的工作，不是因为可以不管。

---

## 1. 测试选择口径

> ✅ **本节唯一的条目 `TEST-01` 已于 2026-09-17 关闭**，整段进
> [archive/TODO_closed_2026-09-17.md](archive/TODO_closed_2026-09-17.md)。
> 留下的规矩：**判据用收集数对账，不能用 grep 文件** —— 同一个文件里可以既有带标记的测试、
> 又有漏网的（`test_exp019_softlift_loro.py` 就是这么发现的）。保留本节编号，后面几节不动。

---

## 2. Stage-2：需单独立项的重构

> 当天做了一次**逐出口的系统梳理**（45 个出口、12 个动作全扫）。
> 抓到三条，两条当天修完；下面是**剩下的**与**结论**。
>
> **梳理的方法本身值得留下**：按"今天已知的失效形状"逐个出口对账 ——
> ① 动作在构造上不可能成功；② 调度状态代替失败归因；③ 同一个量两份实现；
> ④ 未知当成零/通过；⑤ 部分和冒充完整；⑥ 没有停止条件；
> ⑦ 动作与出口自相矛盾。七条里每一条今天都真实发生过至少一次。

- [ ] **AUDIT-S2-02 [P2] 判断函数已经 1795 行 / 63 个 return —— 这是所有「漏改一处」的共同成因。**
  📌 **2026-09-16 实测更新：这个数在变大。** 原条目记的是「1319 行 / 45 个出口」，
  今天 `abfe_preoptimizer.py::Stage2RepairController._decide_once` 是
  **4329–6123 行 = 1795 行、63 个 `return`**（`decide()` 本身只有 23 行，
  是它外面那层「退役一个窗口再判一次」的壳，别拿它的行数当数）。
  ⟹ 09-14 到 09-16 的每一批修复都在往这个函数里加分支，**成因一次都没有被触碰过**。
  实测漏过的：`1c`/`1d` 的准入门（脚本中途抛错、**整份写入被中止**而悄悄丢掉，
  当时测试照样绿）、`5b` 的偏斜归因、`9b` 的子窗补帧、held-out `REJECT` 的末窗判据、
  一处 `DONE` 配 `NO_FEASIBLE_ACTION`。
  **每次复核都能再挖出一条，不是因为复核的人厉害，是因为一个 1319 行的函数没人能一次看全。**
  当天的应对是**把判据往 `plan()` 收**（预算门、补帧准入门都收进去了，各自带结构性测试
  钉住"只许挂一处"）—— 这挡得住"漏挂"，**挡不住"分支顺序错"**。
  真要治得把 `decide()` 拆成按证据类型分派的几段，**那是一次重构，需要单独立项**。

---

---

## 3. Stage-2：两条待维护者拍板（代理不要自行改）

> 两条都**不阻塞在推的工作**，但都要动口径，代理自行改会踩本仓最贵的复发模式
> 「同一个量两份实现」。

- [ ] **BUD-06 [P2] 生产预算的「已用」是 lifetime 累计 —— 这不是接线错误，是一个口径分歧。** **需要维护者拍板，代理不要自行改。**
  位置：`abfe_preoptimizer.py:6899` 附近（合并视图的 `production_budget.per_window_used_steps`）
  与 `:3523`（`_make_production_budget` 的 `stage_used_steps`）。
  聚合视图按**所有采样段**累加逐窗生产步数，插 λ / 拆窗之后**已经作废的旧段**照样计入。
  **2026-09-16 复核：代码是有意这么写的** —— 注释逐字写着「旧段的帧是真烧过的 GPU，
  漏掉它账就永远比实际宽」。所以这条不是 `BUD-01`~`05` 那种接错了，
  是**同一个词（「已用」）在两个用途上要的不是同一个量**：
  - **当成本账**（这台机器到底烧了多少 GPU）⟹ 必须算旧段，现在的写法是对的；
  - **当准入闸**（这个动作还付不付得起）⟹ 算旧段等于让先前几次插 λ 把预算全额扣光，
    cyclod_ligand2/rep1 一 resume 就拿 **4,000,000** 去比 cap、立刻判耗尽。
  ⚠️ **今天它是休眠的**：`stage2_production_budget_steps` 默认 `null`（上限未知 ⟹ 不拦，
  见已关闭的 `BUD-01`），所以**不阻塞任何在推的工作** —— 它只在有人第一次给 cap
  配一个真数字的那天引爆。**「配上 cap」和「cap 生效后行为正确」是两件事。**
  判据：拍板后写明选哪条 —— ① 两个量分别命名、分别落盘（`stage_used_steps` 保持 lifetime，
  另给准入闸一个「当前布局内已用」）；② 准入闸仍读跨段累计，但 cap 也按段数放大；
  ③ 接受现状，在 cap 旁边写明「cap 是 lifetime 口径」。
  ⚠️ 选 ① 就是又一次「同一个量两份实现」—— 那必须**同时**写清谁是权威，
  否则正是本文 §1 反复记的那个最贵的复发模式。

> 当天修完三处「配置在中间层丢失 / 停止决定在收尾层被绕过」（CLI 两条腿都没传生产预算；
> `read_aggregated()` 造子控制器时丢 `effective_config`，于是合并视图的 cap/块大小又退默认；
> 退出前的全路径 `ANALYZE` 会在缺产物时重新采样，绕过刚做出的停止决定）。
> 接预算时顺带加了 `abfe_pipeline._strip_non_identity_kwargs()` ——
> **两个预算键今天不在任何指纹里，剔除它们对既有缓存逐位 no-op**，这是它能无痛落地的前提。
> 下面这条不同：它要动一个**今天确实在指纹里**的键，所以没自行处理。

- [ ] **IDENT-01 [P2] `allow_untrusted_stage_results` 在顶层 `final_results` 指纹里，stage 指纹里已剔除。**
  位置：`abfe_pipeline.py::ABFEPipeline._stage_protocol_key`（已 `run_config.pop(...)`，
  顶层与 `kwargs` 两处都剔）vs `_build_top_level_protocol_key` 里的
  `config = dict(self._last_run_config)`（只 pop 了 `resume` / `run_equilibration`）。
  **两边口径不一致是事实，但哪边对没定。** 两个方向都有像样的理由，别只看一边就改：
  - **该剔**：与 stage 指纹同一条论证 —— 它只决定「质量门没过时是中止还是标记
    `results_untrusted` 继续」，不改 Hamiltonian、不改任何被算出来的数，
    同一份轨迹在开关两种取值下 ΔG 逐位相同。
  - **不该剔**：顶层指纹代表的是**整个 run 的身份**，而这个开关确实改变了
    `final_results.json` 这件产物（带不带 `results_untrusted` 标记）。
    这正是 `LR-01` 关闭时留下的规矩：**顶层 run 指纹与 stage 指纹口径不同，
    顶层无条件进是对的，不要跟着 stage 那侧"统一"**
    （见 [archive/TODO_closed_2026-09-13.md](archive/TODO_closed_2026-09-13.md)）。
  ⚠️ **代价是单向的**：这个键今天**在**顶层指纹里且恒有值（`bool`），
  摘掉它会改变 payload ⟹ **现存 run 的 final-result 缓存全部失配**。
  所以「顺手统一一下」不是零成本动作。
  ⚠️ **需要维护者拍板**，代理不要自行改。
  判据（若判「该剔」）：`tests/test_untrusted_switch_is_not_identity.py` 里的
  `test_flag_is_scrubbed_from_stage_protocol_key_both_levels` 扩到顶层指纹，
  且**同一个 PR 里写清现存 run 的 final-result 缓存会作废**。
  判据（若判「不该剔」）：在 `_build_top_level_protocol_key` 就地写一行注释说明
  「顶层无条件进是刻意的，别跟着 `_stage_protocol_key` 统一」，并在本条下归档 ——
  否则下一轮复核还会把它当成"漏网的一处"重新开一遍。

---

## 4. local-residual / EXP-033

> 背景与已关闭条目见 [TODO.md §2](TODO.md#2-local-residual--exp-033)。
> `LR-06` 是 **P1**，在那边，不在本文。
> ⚠️ **`EXP-033-P1` / `P2` / `P3` 是 EXP-033 自己的阶段编号，不是本文的优先级**；
> 优先级只看方括号里的 `[P2]`。

- [ ] **EXP-033-P1-GPU [P2] 真机首跑 2026-09-17 已通过，但拟合出来是空模型 ⟹ 条目不关，判据换了。**
  📌 **原文「真机一次没跑过」已不成立**：`cyclod_ligand1_outer/rep1` 那次复跑（见 `TODO.md` 的 `LR-06` 第 6 步）
  走了自动重训，`outer_lambda_refit/` 有产物，`closed_form_refit()` 里那段建 Context 的 probe 真机跑通。
  **但拟合结果是空模型**（B 离散度 **2.9e-06 kT**）⟹ 只证明「不炸」，证明不了有增益。
  ⟹ 剩下的判据不再是"出不出 manifest"，是**为什么拟合成了空模型**：是帧源（`pre_equilibration.dcd`）
  的构型跨度不够，还是闭式解在这个体系上退化。
  <details><summary>原始条目（存档）</summary>

  `closed_form_refit()` 里 probe 那段要建 OpenMM Context，只过了静态校验。
  最小验证：4W53 开 `--outer-lambda-local-residual-ibs`（配体指纹对不上冻结的 Atenolol
  ⟹ 会走重训），跑到预平衡结束看它出不出 manifest。
  </details>
- [ ] **EXP-033-P2 [P2]（EXP-033 唯一还开着的一条）** ridge 臂 vs 出厂非线性臂，**U3 口径上机**：
  window-0 utility + ΔG 一致性，且**两臂各自独立标定并冻结自己的 `f_k`**。
  U4 就是栽在候选臂复用 baseline 的 `f_k`，被封为 `INVALID_FOR_PROMOTION`。
  ⚠️ 做 A/B **不要**走 P1 的自动重训（两臂各自重训 ⟹ `sampling_score_sha256` 变成
  run-dependent，两臂不再共用同一把尺子）。要先离线冻一份两臂共用
  （`tools/retrain_local_residual_offline.py`）。

- [ ] **LR-02 [P2] `skip_unsupported_frames` 该撤或反转** —— 支撑域外的帧正是"模型没覆盖这个
  体系"的证据，跳过它等于把本该触发停止的信号变成拟合时看不见的样本。
- [ ] **LR-05 [P2] 重训用的 λ 表是默认值** `linspace(1.0, 0.5, 8)`
  （可用 `outer_lambda_refit_lambda_max/_min/_n_states` 改）。因为重训发生在预优化算出
  真实 stage-2 λ 路径**之前**。按 `B_φ` 的定位这不影响对错；要用真实窗口 λ 表得把重训
  往后挪一个位置 —— 那要改的是**时序**，不是这个模块。

> 🛑 **别在 run 内为了 A/B 重训**，三条理由：违反预注册；`sampling_score_sha256` 变成
> run-dependent；**拿缺构型的帧拟合会把缺掉的态焊进模型**，然后 ESS / overlap /
> split-half / 三方一致全都会更绿——它们只问"这批样本内部自洽吗"。
> 完整的坑清单见 [archive/HANDOFF_LOCAL_RESIDUAL_2026-09-11.md](archive/HANDOFF_LOCAL_RESIDUAL_2026-09-11.md) §6。

---

---

---

## 5. 发布工程门

发布定位是 **clone-and-run**（不打包、不发科学结论）。判据只有一条：
`pytest tests/test_fresh_clone_imports.py` 绿。完整论证见
[RELEASE_READINESS_2026-08-31.md](RELEASE_READINESS_2026-08-31.md)。

> ⚠️ `RELEASE_READINESS` 里的「预览版前」「首发支持范围」是 **08-31 的措辞**，
> **按开发末期的收尾清单读**，别当成一次尚未开始的发布筹备。

> ✅ **`REL-01`（预编译 `.so` 随仓库分发）2026-09-12 已执行**，故不再列：
> `.gitignore` 加了三条例外放行 `build` 符号链接 + `build_exp026_a2/*.so`，
> 并单独排掉没有扩展名的 gtest 可执行文件。`git add -An plugins/` 应当**只有 4 条**。
> 文档（[GETTING_STARTED.md](GETTING_STARTED.md)《CUDA 插件》/
> [TROUBLESHOOTING.md](TROUBLESHOOTING.md)）已同步成"随仓库分发、按环境文件建环境不用编"。

- [ ] **REL-07 [P2] 「本仓库目前没有可以作为最终结论引用的结果」这句话的口径。**
  字面没错（单 seed、无独立重复），但读起来像"什么都没验证过"——
  实际上 4W53 差 **1.83σ**、环糊精差 **1.19σ** 两个闭环都在。
  ⚠️ 这条碰**科学结论口径**，`docs/README.md` 维护规则第 1 条要求科学结论只写
  [STATUS.md](STATUS.md)、别在 README 复制。所以**改法必须是"一行指路"而不是"搬数字"**，
  且**要维护者本人拍板**，不得由代理自行改写。

- [ ] **REL-02 [P2] `abfe_core.py` 分片没审完** —— 第九轮审查里它是唯一没有分片正文的
  （5 条 P2 只有汇总行）。而《五个文件分别应补什么》恰恰把它的职责定为"集中最终结果资格
  与协议登记"。补审属于预览版前的工作。
- [ ] **REL-03 [P2] 三条子项里 ③ 已关闭，①② 仍欠真机。**
  📌 **2026-09-16 更正：本条原文「到目前为止全部是 CPU / 静态验证，零 GPU 复验」已不成立**
  —— 全量 benchmark 39 rep 真上过 GPU（见 `BM-B`）。但**上过 GPU ≠ 这三条被复验过**，
  逐条状态：
  - ① residual 臂混合覆盖度门换口径后，EXP-030 candidate 臂的门读数
    （`ess_gate_mixture_gauge` 应为 `sampling_states`）—— **仍欠**（benchmark 没跑 candidate 臂）；
  - ② 三个 decharging builder 新增的 `frozen_ll_pairs` 断言（真实体系上触发 ⟹ P0-01
    的前提本来就不成立，那是新发现不是回归）—— **仍欠**（39 rep 里没有一条报过它）；
  - ③ ~~两条腿同进程时的 `pipeline.log` 分离~~ —— **2026-09-16 已关闭**：
    `p38_ligand1/rep3` 真实两腿运行下两份日志独立存在
    （`pipeline.log` 134KB / `solvent_leg/pipeline.log` 51KB）；复合物腿里 10 处命中
    "溶剂腿" 全是**一条 WARN 文案自带的词**（`[独立端点段] 未启用 … 该路径在溶剂腿上被论证为…`），
    **不是串日志**。原条目要求的"在真实两腿运行上确认"至此满足。
- [ ] **REL-04 [P2] 2026-09-09 全仓审计那 52 处改动，没有一处被**单独复验**过。**
  📌 **2026-09-16 更正：原文「无一上过 GPU」已不成立** —— 那 52 处在主线里，
  39 rep 的 benchmark 全都执行过它们。**但「跑过」不等于「复验过」**：
  没有任何一次运行是针对这三处设计的对照，跑绿只说明没有硬崩。
  仍需定向复验的三处：偏置爬坡补 1.0 档、preopt 探针的 force group 重划、加密点的采样语义变更。
  复跑命令见 [archive/AUDIT_2026-09-09_full_repo.md](archive/AUDIT_2026-09-09_full_repo.md) §4。

> 下面三条 2026-09-13 从 [RELEASE_READINESS_2026-08-31.md](RELEASE_READINESS_2026-08-31.md)
> 并入。它们在那份表里是"仍欠"，但本文没有对应条目 —— 违反本文规则 1「一条待办只在本文出现
> 一次」的反面：**一条待办一次都没出现**。判据栏原文保留在 RELEASE_READINESS 的同名行。

