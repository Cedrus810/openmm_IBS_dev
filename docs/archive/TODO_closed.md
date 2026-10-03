# 已关闭条目存档

[文档导航](../README.md) · [待办](../TODO.md)

> **本文不是待办。** 所有已关闭条目的原文集中在这里，按日期倒序。
> 原先每关一批就新开一份 `TODO_closed_<日期>.md`（6 份 1656 行），2026-09-18 合并成本文。
>
> 📌 「某条缺陷修没修」的权威是**源码**，不是这里的 `- [x]`。


---

# 2026-10-03（`BOX-01`）

## **BOX-01 [P2] 用修后代码重跑 `cyclod_ligand1/rep2`，判定生产盒 bug 是不是 ΔG 系统性偏负的主因。**

  **缺陷（09-23 已修，CHANGELOG 同日条）**：resume 判「预平衡已完成」后坐标/盒退回输入值 ⟹ 复合物腿在输入盒上采样
  （输入盒普遍比 NPT 平衡体积大 4–5%）；溶剂腿取 NPT 末帧单次快照。
  **为什么是 P2**：新起/续跑的 run 自动用对的盒、旧缓存自动作废 ⟹ 不会再产生错数，只剩归因。
  **触发**：溶剂腿扭转修复（09-23）之后**下一批新代码 benchmark 出数时顺带判定**，不单独排作业（单独跑会把两个修复的效果混在一起）。
  ⚠️ 在那之前，盒子中招的旧结果（cmet 全部、cyclod_ligand1 rep2/rep3）**不得引用、不得拿来比**。

  **benchmark 取证**（convergence json `lj_tail_lrc_box_volume_nm3` vs `pre_equilibration.dcd` 后半段体积均值；
  ΔG 偏差 = 计算 − 实验，kcal/mol，09-23 复现批）：

  | run | 复合物盒 | 溶剂盒 | ΔG 偏差 |
  |---|---|---|---|
  | cyclod_ligand3/rep1 | +0.32% | −0.18% | +2.14 |
  | cyclod_ligand2/rep1 | −0.34% | −0.43% | −0.79 |
  | cyclod_ligand1/rep3 | **+4.93%** | −0.14% | −3.06 |
  | cyclod_ligand1/rep2 | **+4.89%** | +0.79% | −4.17 |
  | cmet_ligand1/rep1 | **+5.37%** | **+3.50%** | −11.23 |
  | cmet_ligand2/rep1 | **+5.42%** | **+3.08%** | −15.01 |
  | cmet_ligand1/rep2 | **+5.40%** | −0.35% | −24.95 |

  方向两处吻合：复合物盒正常的两个 cyclod 偏差小、偏大的全部显著偏负；cmet_ligand1 rep1/rep2 只差溶剂盒，rep1 溶剂盒也大 ⟹ 少负 13.7 kcal。
  物理方向：盒大 ⟹ vanishing ΔG 更正（4W53 实测 ∂ΔA/∂V = 4.07 kJ/mol/nm³，见 [STATUS.md](../STATUS.md)）。
  **不是证明**：n=7、与体系混杂；线性外推 15 nm³×4.07≈61 kJ 远超实测，已离开线性区。

  **判据**：重跑后 cyclod_ligand1 偏差若缩到 cyclod_ligand2/3 的量级（−0.8 ~ +2 kcal）⟹ 盒子是主因，接着重跑全部 cmet；
  缩得不多 ⟹ 盒子只是其中一项。预平衡可复用，其余会被新缓存门自动作废重跑，**不用手删产物**。
  ⚠️ 「偏负 = 采样故障，已定论」这条旧口径 09-23 作废（采样器已修、复现路线是固定采样）。

  📌 **10-03 注（只列事实，没判定）**：09-26 起的 lcr 批都是修后代码。cyclod_ligand1 lcr（n=3）偏差 −0.83，落进上面判据的区间；但同批 cyclod_ligand2 −4.63、cmet 两个 −11.7 ~ −11.9，且 OpenFE / FEP suite（加 LRC 后）同样偏负（[STATUS.md](../STATUS.md)「当前现状」）。10-02 起 abfe-benchmark 不再追 ⟹ 本条的触发条件（「下一批 benchmark 出数」）不会再自然出现。**关闭还是降 P3 由维护者定**。

  **关闭（10-03，维护者拍板）**：不再追。abfe-benchmark 这套测试集本身有问题——cmet 等体系所有方法都跑不出来（IBS / OpenFE / FEP suite 加 LRC 后三家同样偏负，cmet 两个配体 −8.5 ~ −11.9 kcal，见 [STATUS.md](../STATUS.md)「当前现状」），拿它判不了生产盒是不是主因；10-02 起工作重心已换到 fragment-opt-abfe-benchmark（有同拓扑的论文逐 rep 参考）。生产盒的修复本身（09-23）照常生效，不回退。

---

# 2026-09-29（`REVIEW-0929` 的 M1、M2）
> `REVIEW-0929` 本身没关，还欠 L8，条目在 [TODO.md](../TODO.md) §2.7。

## **REVIEW-0929 · M2** LCR 补段门与标签门不一致 ⟹ σ_jk ∈ (σ_tol, 2σ_MBAR] 的窗口标 PROVISIONAL 却永远不补
  **原文**：`stage2_rounds.py::classify_window` 用 `σ_jk > σ_tol` 判 PROVISIONAL，`plan_next_round` 用 `σ_jk > max(2σ_MBAR, σ_tol)` 决定补段；设计表 A1/A2/A5 没覆盖这个区间。来源 [CODE_REVIEW_2026-09-29.md](CODE_REVIEW_2026-09-29.md) §2.2。
  **关闭（09-29，维护者拍板：补段门跟标签门走）**：补段门改成 `σ_jk > σ_tol`；理由字段区分两种情况：超过 2σ_MBAR 记 `segments_disagree_sigma_jk_*`，否则记 `sigma_jk_above_tol_*`。设计表补了一行 A5′（[PROPOSAL_stage2_lcr_rounds_2026-09-24.md](../design/PROPOSAL_stage2_lcr_rounds_2026-09-24.md)），计划表同步。新增 `tests/test_stage2_rounds_2026_09_24.py::test_sigma_jk_above_tol_but_within_2_sigma_mbar_still_triggers`。`STAGE2_ROUNDS_PROTOCOL_VERSION` 没接进任何缓存指纹 ⟹ 不作废现有缓存；已跑完的 run 不受影响，只有之后的补段决策会变（这类窗口会多补一段，最多到 `MAX_ROUNDS=3`）。

## **REVIEW-0929 · M1** `window_self_support_check` 返回 dict 里 `"n_eff_frame_set"` 重复定义
  **原文**：`ibs_engine.py::window_self_support_check` 的返回 dict 字面量里 `"n_eff_frame_set"` 写了两次，后一个常量
  `"segments_kept_by_decorrelation"` 无条件覆盖前面的条件表达式 ⟹ 没有 `production_segments` 时帧集身份被错标（实际用的是 `all_frames`）。
  只进账本、不影响数值。来源 [CODE_REVIEW_2026-09-29.md](CODE_REVIEW_2026-09-29.md) §2.2。
  **关闭（09-29）**：删掉后一个键（连同它那行注释），条件表达式那个保留。
  `tests/test_join_lambda_two_sided_support.py` 两条测试补了断言：不分段时读到 `all_frames`，分段时读到 `segments_kept_by_decorrelation`。
  `ruff --select F601` 已无命中。

---

# 2026-09-24（未关闭条目的已完成部分：`LR-06` 第 1–4、6 步）
> **`LR-06` 本身没关**，还欠第 5、7 步，条目在 [TODO.md](../TODO.md) §2.4。这里是 09-24 压缩 TODO 时挪过来的已完成部分：方案选择、根因、真机 A/B、`r_floor` 依据、排除项。原文一字未改。

## **LR-06（已完成部分的完整记录）** 预热鬼影期 × 插件 0.1 Å 硬门 ⟹ outer 臂在窗口 0 必死。**
  **🟢 2026-09-16：方案 A 已落地并真机验证。2026-09-17：第 6 步真机通过。**
  **⏸️ 剩下的第 5、7 步 2026-09-17 由维护者判为「不着急，等控制器修完再动」** —— 第 7 步换体系复跑会把控制器当前那批未收口的改动一起卷进去，跑了也说明不了插件；第 5 步删脚手架要在有真机复跑兜底时才敢动。

  🔺 **2026-09-18 更新：第 5 步的押后条件（「有真机复跑兜底」）已经满足。**
  `cyclod_ligand1_outer` rep2/rep3 **带着那两个猴子补丁跑完全程**（详见第 6 步）。
  第 7 步（换体系）**仍然没做** —— rep2/rep3 是同一个体系，不是换体系。

  ### 已做完（代码 + 真机）

  | 步 | 状态 |
  |---|---|
  | 1. 4 处硬门改夹取 | ✅ `CudaLocalManyBodyResidualKernels.cpp` **能量侧 2 处夹取**（`rAngstrom = max(r, floor)`，这条边照样计入）+ **力侧 2 处跳过**（下限内 dE/dr ≡ 0，`invRM = 1/r` 因此根本不求值）。两半语义不同，不是同一句抄四遍 |
  | 1b. **Reference 侧同步** | ✅ **原条目漏了这一步**：`g1_math_core.h` 的 `enumerateEdges` 也有同一道门（抛 `MathError`），而 Reference 平台经 `ReferenceLocalManyBodyResidualKernels.h` 用它 ⟹ **只改 CUDA 就是「同一个不变量两份实现」**，G1/G2 钉的正是两者一致。已改成同一套语义并复用同一个常量（`#include "r1_model_layout.h"`，不许再抄一份数字） |
  | 2. 重编 | ✅ `g0_build.sh` 三个 `.so` 全通过 |
  | 3. 回归 | ✅ `exp028_run_regression_suite.sh` **ALL TESTS PASS**（5 个原生 harness）。⚠️ **G1/G2/G3 三个一致性 harness 没跑** —— 它们要 `yyjson`，本机 `$CONDA_PREFIX/include/yyjson.h` 缺失，套件按设计跳过。**改动最需要它们，这是本次验证的已知缺口**，下面用真机 A/B 代偿 |
  | 4. 同步两处 sha256 | ✅ `local_residual/openmm_plugin.py::KNOWN_PLUGIN_SOURCE_SHA256` + `resources/…/manifest.json` 的 `plugin.source_sha256` → `5c74365dffaa…` |
  | 5. 删临时脚手架 | 🔓 **2026-09-18 解除押后**（押后理由是「要有真机复跑兜底」，第 6 步已给出两个完整 run）。两个 `Context` 级补丁**仍在**，但代码里那条「这套东西还没经过大规模验证」（`em_no_residual.py:138`）的注释已经**过期**。⚠️ 两半代价不同，别当一件事做：`patched_set_parameter` 的 9 处调用点全在 `ibs_engine.py`、全是 `context.setParameter(f"{self.prefix}_bias_scale", X)` 同一形状（6786 / 15912 / 15934 / 15953 / 15968 / 16115 / 16152 / 16196 / 19897）⟹ 一个 helper 收掉，顺带消掉「同一不变量 9 份实现」的形状；`patched_get_state` **故意是全局的** —— 它拦的是鬼影期内进程里**任何**不带 `groups` 的 `getState`，不只是 `step_guard.py:104/186` 那两处，改成逐点显式传参就是补丁注释自己写的「逐个去包必然漏」⟹ **建议保留，并把「为什么不合并」写进注释** |
  | 6. `cyclod_ligand1_outer` 复跑 | ✅ **2026-09-17 真机通过**。同一份 `runs/cyclod_ligand1_outer/rep1/launch.log` 里两次启动直接对照（判版本用**启动横幅**不用 mtime，`BM-B` 取证①）：**09-16 02:46（夹取前）** 行 303–357 窗口 0 热化 `fail-closed … pair distance below 0.1 A`，步长降到 0.008 fs 仍炸 → `RuntimeError`；**09-17 01:22（夹取后）** 行 358–1251 `fail-closed` **0 次**，窗口 0–4 全部跑出能量，走到 `[自治 15/40]`，02:29 终止于**人为 SIGTERM**（`TerminationRequested`），不是插件。⟹ 「outer 臂在窗口 0 必死」这个失效模式已消除。<br><br>🔺 **2026-09-18 追加（同僚会话 `abfe-benchmark-cf` 取证）：不只是「不死了」，是跑完了。** `rep2`/`rep3` **全程带着第 5 步那两个猴子补丁**，跑出完整结果：<br>· `rep2` ΔG_bind **-6.28 ± 0.48** kcal/mol，`untrusted=False`（complex 71.20 / solvent 44.94 / Boresch -37.02 kJ/mol），控制器 19 轮，末轮 `NO_ACTION`<br>· `rep3` ΔG_bind **-2.00 ± 0.52** kcal/mol，`untrusted=False`（complex 56.94 / solvent 48.57 / Boresch -32.24 kJ/mol），控制器 14 轮，末轮 `NO_ACTION`<br>· 两 rep 的 complex/solvent **两条腿 `analysis_status` 均为 `ANALYSIS_COMPLETE`**，`precision_status` 均为 `UNMEASURED`，`outcome.exit` 均为 `ANALYSIS_COMPLETE_PRECISION_UNMEASURED`<br>· 版本（按启动横幅，`BM-B` 取证①）：**2026-09-17 14:43:12-04:00** / **15:54:47-04:00**，`abfe_ibs_commit=9a18ded`，RTX 5080 ⟹ 落在「09-17 那批控制器改动之后、09-18 的 `S2-N`/`S2-O` 之前」<br>· `rep1` 无结果，崩在 `ExistingEnsembleRequiresRescueAudit`（IBS state score identity mismatch）—— **这是闸在正常工作，别放宽**：逐 rep 闭式重训 ⟹ `sampling_score_sha256` 逐 rep 不同 ⟹ 拒绝复用别的 rep 的 ensemble；要跑得给它自己的干净 checkpoint 目录<br><br>⚠️⚠️ **`runs/cyclod_ligand1_outer` 与 `runs/cyclod_ligand1` 两臂不能做 A/B。** 两个 config 逐键只差两行（`output` 与 `outer_lambda_local_residual_ibs`），但冻结 manifest 只覆盖 Atenolol、本配体 27 原子对不上指纹 ⟹ 走 EXP-033 P1 的**自动闭式重训**，**三个 rep 各自拟合各自的 B_φ**。所以 outer 均值 -4.14 vs 基线均值 -5.36 这个差**不构成 outer 臂的效果证据**。真要 A/B 得按 `docs/RETRAIN_LOCAL_RESIDUAL.md` 手工冻结一份权重给两臂共用 |
  | 7. 换疏水口袋体系再跑 | ⏸️ **仍然押后，2026-09-18 复核后不变**。`runs/` 下仍然只有 `cyclod_ligand1_outer` 一个 outer 目录，jnk1/p38/brd4 一个都没跑过。第 6 步新增的 rep2/rep3 **是同一个体系**，补的是「补丁能跑完」，不是「换了体系也成立」⟹ 这一步的缺口原样还在 |

  ### 真机 A/B（RTX 2080 Ti，mixed precision，改动前后两份独立编译的 `.so`、两个独立进程）

  | 全局最小 lig–env 距离 | 新旧残差能量 |
  |---|---|
  | **≥ 1.5 Å** | **逐位相同**（`20.874434910728507` / `19.078136849613827` / `19.07729951664752`） |
  | **< 1.5 Å** | 变了（夹取生效） |
  | **0.05 / 0.093 Å** | 旧：`fail-closed`；新：有限值。**`0.093 Å` 正是真机崩点** |

  ⚠️ **判据是「全局最小配体↔环境距离」，不是「被移动那个原子到锚点的距离」** ——
  把一个水原子推到距配体原子 0 为 1.8 Å 处时，它到**另一个**配体原子只有 1.146 Å，
  照样触发夹取。第一版对照表就是这么看岔过一次，结论一度写成"上限之上也变了"。

  ### `r_floor = 1.5 Å` 的依据（不是 0.1）

  `0.1` 的来历只是防 `1/r` 发散。`1.5` 取**训练支撑域下界**：shipped R1 的训练帧实测
  最近配体↔环境距离 **1.517 Å**（水氢）/ **1.524 Å**（带 LJ），中位 1.8 Å。
  📌 **本次新增的实证**（原条目没有这一条，它决定了"夹在 0.1 行不行"）：
  **径向基在小 r 处并不衰减** —— 16 个中心均匀铺在 `[0, 5] Å`、宽 0.333，
  0.1 Å 处基函数值仍有 **0.96**；而落在支撑域下界以下那 5 个中心
  （0.333 / 0.667 / 1.0 / 1.333 Å）上的 pair weight 量级与训练充分的**完全一样**
  （`max|w|` 0.30–0.37 vs 0.32–0.39）。**完全活跃、却从未被数据约束。**
  ⟹ 夹在 0.1 Å 只能止崩，止不住"拿没训过的权重外推"。

  ⚠️ **`r_floor` 目前是编译期常量，绑定 shipped R1 模型。** 换一个支撑域不同的重训模型
  会**静默沿用 1.5**。升级路径写在 `r1_model_layout.h` 的 `ponytail:` 注释里：
  把 `r_floor_angstrom` 放进 payload config + manifest（离线训练器本来就测得到），
  再经 `buildAndLoadKernels()` 的 `defines` 发给运行时编译的内核（和 `NUM_RADIAL_BASIS` 同路）。

  判据：`tests/test_lr06_residual_distance_clamp_2026_09_16.py`（**5 passed**）——
  钉住 ①下限必须是支撑域而非 0.1、②四个站点一个都不许再抛 `MIN_DISTANCE`、
  ③能量侧夹取/力侧跳过各 2 处、④Reference 与 CUDA 同源同语义、⑤两处 sha256 已同步。
  错误码 `EXP025_DEVICE_ERROR_MIN_DISTANCE = 2` **保留**（`exp026_control_plane_layout.h`
  有 `static_assert` 钉 ABI），只改触发策略。

  ⚠️ `B_φ` 在 `r < r_floor` 处的定义变了 ⟹ `sampling_score_sha256` 变 ⟹ 与**已冻结**的
  旧分数不再可比。对 ΔG 正确性无影响（它是偏置，MBAR 对任意冻结偏置无偏），但做 A/B 时
  两臂必须都用新插件。

  <details><summary>原条目（根因分析与方案选择，保留备查）</summary>


  **症状**：`cyclod_ligand1_outer` 三个 rep 全部死在 stage-2 窗口 0：
  `LocalManyBodyResidualForce (CUDA) fail-closed in K1/K6a (computeQ) (code 2):
  pair distance below 0.1 Angstrom`。步长 2 fs→0.02 fs 无效。本节点复现 3/3。

  ### 根因

  `bias_scale` 乘在 **Group-1 整个表达式外面**（`ibs_engine.py` 的
  `f"{prefix}_bias_scale * ({_state_expr(0)} - kt*(...))"`），而 `_state_expr(0)` 里就是
  `cv_0_int + cv_0_rest` —— **配体↔环境的软核相互作用**。所以 `bias_scale = 0` 不是
  "关掉偏置"，是**把配体关成完全的鬼影**：水分子直接穿过配体，`r → 0` 是**必然**。
  三处会进入这个状态（后两处是有意的设计）：

  | 位置 | 何时 |
  |---|---|
  | `local_residual/em_no_residual.py` post-EM | **已修，见下面 ①** |
  | `ibs_engine.py` 时间步长爬坡 | 全程 `bias_scale=0`（注释：避免 dt 与未校准偏置两个不稳定源混在一起） |
  | `ibs_engine.py` 偏置爬坡 | 0 → 0.2 → … → 1.0 |

  而 `CustomCVForce` 会求值它的**每一个** CV（与系数是否为 0 无关），于是插件照样在
  鬼影几何上跑 K1/K6a，撞硬门。

  ### 为什么不能靠"别在鬼影期求值残差"来绕

  一天之内已经找到**三个**求值点，每堵一个就冒出下一个：

  1. 积分器（`setIntegrationForceGroups` 可排除）
  2. 不带 `groups` 的 `getState`（`step_guard.finite_state_check` 每 500 步一次）
  3. **`ibs_engine.py:16033 → 8924` 的 `getCollectiveVariableValues()`** —— 既不走积分
     力组、也不走 `getState`，直接求值全部 CV。**真机就死在这里。**

  鬼影期**照样在产生** <0.1 Å 的构型，排除力组只是让积分器不碰它，构型本身还在往
  重叠里漂。⟹ 逐个堵求值点是打地鼠；两次真机跑谁过谁不过是**随机**的（速度种子不同）。

  ### 已落地（独立成立，与下面的选择无关）

  - **① `local_residual/em_no_residual.py` post-EM 只清 `_s_residual`，不再连
    `_bias_scale` 一起清。** 原来两个一起清，而恢复要等到热化之后 ⟹ 那 10000 步热化
    跑的是鬼影配体。这与 `ibs_engine` 自己的注释直接冲突（它在 EM 前**只**关
    `s_residual`，明写「不像 bias_scale=0 那样连 baseline 也在正常使用的物理
    softcore-state 混合力一起关掉」）。副作用也修掉了：**A/B 两臂不再是一个热化真配体、
    一个热化鬼影**。实测：热化从「3/3 必崩在第 2500 步」变成**稳定通过**。

  ⚠️ **临时脚手架，A 落地后删掉**：`em_no_residual.install()` 里另外两个 `Context` 级
  补丁（`setParameter` 同步积分力组、`getState` 按积分力组取 State）。它们把失败点从
  热化推到了 dt 爬坡之后，但**不是修复**（见上面第 3 个求值点）。主线
  `ibs_engine.py` / `step_guard.py` 目前**零改动**。

  ### 选定修法：**A —— 插件侧夹取**（不是 B）

  把 4 处 `rAngstrom < EXP025_MIN_DISTANCE_ANGSTROM` 的「报错 + `continue`」改成**夹取**：
  能量用 `r_eff = max(r, r_floor)`（下限内是平台、C0 连续、**这条边照样计入**），
  力在下限内取 0（`invRM = 1/r` 那条发散项因此根本不需要求值）。
  位置：`plugins/LocalManyBodyResidual/platforms/cuda/src/CudaLocalManyBodyResidualKernels.cpp`
  的 304 / 447（K1 computeQ / force-scatter）与 766 / 832（K6a 同上）。
  ⚠️ 别改成单纯 `continue`：那会在下限处制造能量跳变、给出冲量。

  **为什么 A 而不是 B**（B = 把 `em_no_residual` 的孪生 System 从"只保护 EM"扩到整个预热期）：
  A **消除失效模式**，B 只是**安排它不被观察到** —— B 的不变量「残差力永远不被喂到鬼影
  几何上」靠控制流维持，而今天已经漏了三次，以后加一个探针就再漏一次。另外 A 不碰
  `run_all_windows` 那 700 行（Context 迁移要搬位置/速度/盒子/全局参数/sampler 的
  context 引用/checkpoint，漏一项是静默错），且对**所有**体系生效。

  **`r_floor` 取训练支撑域下界、跟着模型走**（写进 payload/manifest，重训时从训练帧实测），
  不沿用 0.1 Å —— 那个数的来历只是防 `1/r` 发散。本次训练帧（`pre_equilibration.dcd`
  200 帧，λ=1，**按每帧自己的盒子**算）配体↔环境最小距离：水氢 **1.517 Å** / 带 LJ
  **1.524 Å**，中位都在 1.8 Å；而崩点 0.093 Å **比支撑域近 16 倍**，那一段 typed MLP
  在响应区外、输出与梯度都不受训练约束。⚠️ 水**在**模型环境里（type vocabulary 含
  Z=1/Z=8），所以只能夹取、**不能把水氢剔出环境**。

  ### 前提已验证（2026-09-16，本机）

  | 检查 | 结果 |
  |---|---|
  | nvcc | `/opt/cuda/bin/nvcc` 12.9 ✓ |
  | 构建脚本 | `plugins/LocalManyBodyResidual/g0_build.sh`（手写 g++，非 CMake），只要 `CONDA_PREFIX` 指向 openmm_dev ✓ |
  | 上次构建 | `build_exp026_a2/` 2026-09-10 建于本机，`.o` 还在 ✓ |
  | **CUDA 内核是运行时编译** | 内核源码在 `kDeviceSource` / `kDeviceSourceCSR` 两个 `R"CUDA(...)"` 字符串里，由 OpenMM `createModule` 运行时编译 ⟹ **改内核只需重编宿主 .so，不走 nvcc 路径** ✓ |

  ### 执行顺序与代价

  1. 改 4 处 → 2. `g0_build.sh` 重编 → 3. G0–G3（`exp028_run_regression_suite.sh`）
  → 4. 同步 `local_residual/openmm_plugin.py` 的 `KNOWN_PLUGIN_SOURCE_SHA256`
  **和** `resources/outer_lambda_local_residual/manifest.json` 的 `plugin.source_sha256`
  （run 内重训产出的 manifest 会自动带新值）→ 5. 删掉上面那两个临时 `Context` 补丁
  → 6. `cyclod_ligand1_outer` 真机复跑 → 7. 换一个疏水口袋体系（jnk1 / p38 / brd4）再跑一遍。

  ⚠️ `B_φ` 在 `r < r_floor` 处的定义变了 ⟹ `sampling_score_sha256` 变 ⟹ 与**已冻结**的
  旧分数不再可比。对 ΔG 正确性无影响（它是偏置，MBAR 对任意冻结偏置无偏），但做 A/B 时
  两臂必须都用新插件。
  ⚠️ 错误码 `EXP025_DEVICE_ERROR_MIN_DISTANCE = 2` **保留**（`exp026_control_plane_layout.h`
  有 `static_assert` 钉 ABI），只改触发策略。

  ### 已排除，别重查

  - 不是体系爆炸 —— 失败时逐力组全正常（g0 max|F|=4818、g2=677 kJ/mol/nm），坐标全有限，
    配体内部最近对 0.97 Å（正常 C–H）；
  - **不是合并主线改坏的** —— 插件源码与 EXP-030 实跑那份（`Atenolol-rank11/plugins/…`，
    sha `10afff53…`）`diff` **只差 17 行 MIT 许可证头**；`em_no_residual.py` 两份也都清
    `_bias_scale`；
  - 不是残差力在推 —— 失败坐标上把那个水 H 从 0.12 Å 扫到 4 Å，残差力恒 ~1e-5 kJ/mol/nm
    （本次是闭式重训出的空模型，`relative_improvement=9.2e-07`）；
  - 不是"软核允许重叠所以必然触雷" —— `power_lj=[2,2]`、`alpha_lj=0.5`，窗口 0 最深
    λ_vdw=0.5933 时带 LJ 的对势垒 ~26 kT、λ=0.7054 时 ~118 kT，热涨落进不去。**能进去的
    只有鬼影期。**
  - `candidates=0` 是**死字段** —— `EXP026_STATUS_CANDIDATES` 只被 `exp026ResetSupportStatus`
    置 0、全仓没有任何地方写过。要么补写入，要么从错误信息里删掉。
  </details>


---

# 2026-09-24（代码已回答：膜生产协议里 NPT/NVT 两条）
> 原在 [TODO_P3.md](../TODO.md) §3「膜生产协议」。原文：

## [x] 明确炼金生产阶段使用 NPT 还是 NVT。
## [x] 若使用 NVT，记录固定盒矢量来自哪一帧。

  ### 关闭记录（2026-09-24）
  维护者拍板关闭。答案以源码为准：

  - **只有预平衡是 NPT**：`abfe_pipeline.pre_equilibrate` 在 System 拷贝上
    `ensure_barostat_for_protocol`（可溶 = `MonteCarloBarostat`，膜 = `MonteCarloMembraneBarostat`）。
  - **炼金生产全程 NVT**：再平衡、stage1/2 窗口、overlap 探针的 System 都不带 barostat
    （`ibs_engine.py` 窗口元数据 `"ensemble": "NVT"`）；co-ion 笛卡尔 tether 与传统 REMD
    Beutler 软核检测到 barostat / 体积波动直接 fail closed。
  - **固定盒来自哪里**：可溶体系 = NPT 后半段 block 均值体积，末帧盒各向同性缩放、分子质心
    等比（`_rescale_to_npt_mean_volume`，09-23），落盘 `pre_equil_final_state.npz`，
    resume 经 `adopt_pre_equilibrated_state` 读回；窗口缓存有盒体积门。
  - ⚠️ **膜体系不走均值缩放**：`_rescale_to_npt_mean_volume` 对 `ENVIRONMENT_TYPE_MEMBRANE`
    原样返回 ⟹ 膜体系生产盒 = NPT **末帧**盒（与 09-23 之前溶剂腿同形状：一次瞬时涨落被冻成密度）。
    膜线整条暂停，未改；重启膜线时要先处理（半各向同性盒不能直接套各向同性缩放）。

---

# 2026-09-23（已被取代：发布工程门里针对旧快照的两条）
> 不是修好了。两条要验/要补的对象都是**已被后续大版本改写的旧快照**。原文一字未改。

## **REL-07 [P2] 「本仓库目前没有可以作为最终结论引用的结果」这句话的口径。**
  字面没错（单 seed、无独立重复），但读起来像"什么都没验证过"——
  实际上 4W53 差 **1.83σ**、环糊精差 **1.19σ** 两个闭环都在。
  ⚠️ 这条碰**科学结论口径**，`docs/README.md` 维护规则第 1 条要求科学结论只写
  [STATUS.md](../STATUS.md)、别在 README 复制。所以**改法必须是"一行指路"而不是"搬数字"**，
  且**要维护者本人拍板**，不得由代理自行改写。

  ### REL-07 关闭记录（2026-09-23）
  维护者拍板：「有了啊，我们不是在跑吗」。`README_cn.md` / `README.md` 那句改成
  「已有真机结果（多体系 benchmark 正在出数），能否引用、按什么口径只以 STATUS.md 为准」——
  按条目要求只指路、不搬数字。

---

## **REL-02 [P2] `abfe_core.py` 分片没审完** —— 第九轮审查里它是唯一没有分片正文的
  （5 条 P2 只有汇总行）。而《五个文件分别应补什么》恰恰把它的职责定为"集中最终结果资格
  与协议登记"。补审属于预览版前的工作。

  ### REL-02 归档说明（2026-09-23）
  要补的是 08-31 第九轮审查里 `abfe_core.py` 分片的正文（`archive/RELEASE_READINESS_2026-08-31.md:315-318` 仍写「正文完全缺失」）。那份审查本身已归档，`abfe_core.py` 之后经多轮大版本改写 ⟹ 补一份旧快照的正文没有意义。要审就对**当前**代码另开。

---

## **REL-04 [P2] 2026-09-09 全仓审计那 52 处改动，没有一处被**单独复验**过。**
  📌 **2026-09-16 更正：原文「无一上过 GPU」已不成立** —— 那 52 处在主线里，
  39 rep 的 benchmark 全都执行过它们。**但「跑过」不等于「复验过」**：
  没有任何一次运行是针对这三处设计的对照，跑绿只说明没有硬崩。
  仍需定向复验的三处：偏置爬坡补 1.0 档、preopt 探针的 force group 重划、加密点的采样语义变更。
  复跑命令见 [archive/AUDIT_2026-09-09_full_repo.md](AUDIT_2026-09-09_full_repo.md) §4。

> 下面三条 2026-09-13 从 [RELEASE_READINESS_2026-08-31.md](RELEASE_READINESS_2026-08-31.md)
> 并入。它们在那份表里是"仍欠"，但本文没有对应条目 —— 违反本文规则 1「一条待办只在本文出现
> 一次」的反面：**一条待办一次都没出现**。判据栏原文保留在 RELEASE_READINESS 的同名行。

  ### REL-04 归档说明（2026-09-23）
  要复验的是 09-09 全仓审计那 52 处改动（偏置爬坡补 1.0 档、preopt 探针 force group 重划、加密点采样语义）。09-09 之后几轮大版本已把这些代码路径改过/关掉（复现路线关了路径演化与布局动作）⟹「单独复验 09-09 那份 diff」不再对应任何现存代码状态。要验就按当前代码的真机 run 验。

---

# 2026-09-23（已被取代：控制器时代的条目）
> **这 5 条不是「修好了」，是「这条线不再推了」**：09-22 起主线改走复现路线（等分 λ、钉死分窗、
> f_k 固定节奏、**自治控制器关闭**；见 [DEBUG_REPRO_MODE_2026-09-22.md](../DEBUG_REPRO_MODE_2026-09-22.md)），
> 之后又经过几轮大版本更新。下面这些条目的判据都建立在「生产路线由 `decide()` 自治控制器驱动」
> 或「09-17 那批 benchmark」之上，在当前架构下已无法、也无需按原判据验收。
>
> 09-23 只读核查时的现状（供万一重启这条线时参考，**不是**验收结论）：
> - `AUDIT-S2-03`：09-19 两个控制器 run（`cyclod_ligand1/rep3`、`cyclod_ligand3/rep3`）两腿都以
>   `NO_ACTION … ANALYSIS_COMPLETE_PRECISION_UNMEASURED` 收口，但 stage JSON 已被 09-22/23 复现批覆盖，判据 4–6 无从核对。
> - `BM-B`：09-17 那批 30 个未完成 run 已全部被后续重跑取代；遗留未归类的是 `thrombin_ligand1` 三个 rep 的启动期错误
>   （rep1/rep2 `GROMACS 与轨迹拓扑原子数/配体索引不匹配`，rep3 charge-transfer closure 拒跑）与
>   `cyclod_ligand1_outer/rep1` 的 `ExistingEnsembleRequiresRescueAudit`。
> - `S2-A`：`_read_stage_result()` 仍只读 ANALYZE 之后才写的文件（`abfe_preoptimizer.py:3811`），09-17 起所有 `逐块判据量` 全 `None`。
> - `AUDIT-S2-02`：`_decide_once` 现为 **2789 行 / 73 return**（09-16 记的是 1795/63）；`refactor/` 明确不碰它（「无门，不许做，另立项」）。
> - `BUD-06`：未拍板，默认上限 `null` 仍休眠；另发现合并视图 `abfe_preoptimizer.py:9040-9065` 自建预算 dict、绕开「唯一构造器」`_make_production_budget`。
>
> **重启控制器这条线时**：把对应条目原样搬回 TODO，先按上面现状重新确认判据是否还适用。

| 条目 | 原在 |
|---|---|
| `AUDIT-S2-03` | `TODO.md` |
| `BM-B` | `TODO.md` |
| `S2-A` | `TODO_P2.md` |
| `AUDIT-S2-02` | `TODO_P2.md` |
| `BUD-06` | `TODO_P2.md` |

---

## **AUDIT-S2-03 [P1] 控制器已有真机，但**旧验收判据（`DONE`）单跑结构性不可达，
  2026-09-18 已换判据**；新判据下的闭环仍未在修复后的代码上拿到。**
  📌 **2026-09-16 更正：本条原文「控制器真机零验证」已不成立。** 全量 benchmark
  （13 体系 ×3 = 39 rep）真上过 GPU，9 个跑出 `final_binding_results.json`。
  但**其中只有 2 个跑的是修复后的代码**（`p38_ligand1/rep2` 启动 09-16 06:29:45、
  `rep3` 07:34:59；判「跑哪版代码」的口径见 `BM-B` 取证①），其余 7 个都是旧码启动。
  **那 2 个的终态是 `ANALYSIS_COMPLETE_PRECISION_UNMEASURED` + `NO_FEASIBLE_ACTION`，
  不是 `DONE`。**（两者都在 `TERMINAL_EXITS` 里，是**合法终态、不是 bug** ——
  但设计 §7 要的验收是「六窗全部 `ANALYSIS_ELIGIBLE` → `DONE`」，那个还没在修复后的代码上出现过。）
  ⚠️ 2026-09-14 那约 **30 处**修复 + 09-16 的 `BM-01`~`03`/`05`，真机覆盖仍然只有上面这 2 个 run。

  📌📌 **2026-09-18 再次更正：旧判据（`decide()` 给 `DONE` 且 `execution_status=COMPLETE`）
  在单跑里结构性不可达，已作废。** 逐条核实（与 `abfe-benchmark-cf` 会话对过 36 份台账）：
  · `DONE` ⟸ `precision_status == MEETS_CROSS_REPEAT_TARGET`
    （`abfe_preoptimizer.py:6038` / `:7805`），而 `MIN_INDEPENDENT_REPEATS = 3`
    （`abfe_core.py:2672`）。
  · Stage-2 IBS **真正计算**该字段的三条生产路径全传空列表
    （`ibs_engine.py:23632` / `:23789` / `:24158` 的 `cross_repeat_precision([])`），
    且仓内**没有跨-run 聚合器**回写它 ⟹ 单跑恒 `UNMEASURED`。
    ⚠️ **「全仓只有三个产出点」的说法不准确，别照抄** —— `"precision_status"` 全仓
    有 22 处（`grep -n '"precision_status"' *.py`），但它们分三类，没有一类能产出
    `MEETS`：① **计算**的就上面三处，全传空列表；② **硬写 `UNMEASURED`** 四处
    （`abfe_pipeline.py:2537` / `:6969` / `:10680` / `:10755`，外加
    `ibs_engine.py:23788` / `:24157`）；③ 其余是**转抄/汇总**
    （`abfe_pipeline.py:6658` / `:7830` / `:7858` / `:14885`、
    `abfe_preoptimizer.py:3388` / `:4450`、`runabfe.py:7125` / `:7137` / `:8804`、
    `ibs_engine.py:14505` / `:23700`），只搬不算，且 `runabfe.py:7125` 的两腿合并
    对缺失值兜底成 `"UNMEASURED"`。⟹ 结论不变，而且比原说法更强。
  · `execution_status=COMPLETE` 不是第二道条件，只是 `action == "DONE"` 的派生值
    （`abfe_preoptimizer.py:5853`）。
  · `--allow-untrusted-stage-results` 救不了 —— 它换出口的那两行写在
    **precision 已达标的 then 分支里面**（`:6045` / `:7813`）。
  · 单跑的合法终态本来就存在：`action=NO_ACTION` + `exit=ANALYSIS_COMPLETE_PRECISION_UNMEASURED`
    （`:6047` / `:7816`，`EXIT_SPECS` 里 `terminal=True`，见 `:2782`）。
  · 管线侧 2026-09-15 已经换过判据：非 `DONE` 出口**不再** raise，只记日志
    （`abfe_pipeline.py:8599`）；真正拦的是分析不完整 / 窗口子集 / λ 覆盖不全 / 非有限结果。
  · 「能不能当已验收结果发布」另有专门闸 `publishable_as_accepted_result`
    （`abfe_pipeline.py:7870`），`MEETS` 的要求在**那里**，控制器层不必再要一遍。
  ⚠️ 真机 36 份台账：`NO_FEASIBLE_ACTION` 20 / `ANALYSIS_COMPLETE_PRECISION_UNMEASURED` 16 /
  **`DONE` 0**；`precision_status=UNMEASURED` 25/25；**没有任何 run 出不来**（全部有
  `outcome.exit`）。上层语义也是接对的：未验收走 WARNING + 零退出码 + 产物齐全。
  ⚠️ 即使凑够 3 个独立重复也大概率拿不到 `MEETS`：按臂实测跨 rep 散布是报出 σ 的
  **3.7× 中位**（10 个臂无一小于 1；三个 n=3 的臂 3.3/3.0/1.9×），而门槛是
  `CROSS_REPEAT_MAX_STDDEV_KCAL_PER_MOL = 1.0`（`abfe_core.py:2671`）。
  把判据改写成「拿到 `MEETS`」只是把一个不可能事件换成另一个。

  **新判据（2026-09-18）—— 任一体系在当前代码上跑出：**
  1. `analysis_status = ANALYSIS_COMPLETE`（两条腿都要）；
  2. `action = NO_ACTION`；
  3. `autonomous_outcome.exit = ANALYSIS_COMPLETE_PRECISION_UNMEASURED`；
  4. `missing_windows` / `skipped_windows` / `skipped_sampling_units` /
     `out_of_range_windows` **全空**；
  5. 不带 `stage_scope = window_subset_no_stage_verdict`，也不带
     `subset_partial_sum_not_delta_G = true`；
  6. `coverage_diagnostics` 覆盖**全部** λ 态且 `dropped_window_indices` 为空。

  `DONE` 需要 ≥3 个独立重复，属 **BM 跨-run 验收**，不属控制器单跑验收 ⟹ 归 `BM-B`。
  **在拿到一次满足上面六条的闭环之前**：不得声称「Stage-2 自治闭环可用」；
  不得删除任何旧修复路径（`S2-E`）；控制器产出的任何 ΔG 都不是可引用结果。

  ⚠️ 本条上面那句「设计 §7 要的验收是『六窗全部 `ANALYSIS_ELIGIBLE` → `DONE`』」
  指的是 2026-09-12 那次真机 —— 那是 **09-15 收紧 `DONE` 判据之前**的事，不能拿它
  当现在的验收口径。设计文档 `STAGE2_CONTROLLER_DESIGN_2026-09-12.md:237` 那句
  「`DONE` 在结构上不可达」说的是**另一件事**（中间结果不落盘的死锁，已修），别混。

---

## **BM-B [P1] 全量 benchmark（13 体系 ×3 = 39 rep）：30 个未完成 run 里，最后启动在修复之后的是 0 个。**
  **先取证，再重跑 —— 但有 6 个不能直接重跑，见下。**
  来源：`abfe-benchmark-31` 会话 2026-09-16 的两轮汇总 + 本会话复核。
  阶段完成度 equilibration 36/39、attachment 36/39、decharging 35/39、**vanishing 9/39**。

  ### ⚠️⚠️ 三条取证口径，都是这一轮**栽过之后**才定下来的

  **① 判「跑的是哪版代码」只能用「最后一次启动横幅」（`已合并配置文件`）的时刻，
  不许用 `launch.log` 的 mtime。** Python 在进程启动时加载源码，所以决定版本的是
  **启动时刻**；mtime 是最后一次写日志。两者能差几小时，实测：

  | rep | 最后启动 | mtime | 差 |
  |---|---|---|---|
  | `p38_ligand1/rep1` | 09-16 **04:09:10** | 09-16 06:29 | 2h20m |
  | `brd4_ligand2/rep1` | 09-15 **15:23:38** | 09-16 05:10 | **13h47m** |

  📌 **本条第一版就是这么写错的**：用 mtime 判成「1 个 run 跑过修复后的代码」，
  实际是 **0 个**。修正后结论**方向不变而且更干净** —— 不需要再拿 `BM-A` 第 4 条
  去解释那个"例外"，例外根本不存在。

  **② `launch.log` 跨重启追加，「最后一条异常」可能来自早就被修好的那次启动。**
  实测启动次数：`cyclod_ligand2/rep1` **25 次**、`brd4_ligand2/rep1` 9 次。
  要看**最后一次启动横幅之后**的异常。按这条重扫 30 个：24 个分类不变，**6 个作废**。

  **③ 判「这个 run 跑完没有」时，`final_binding_results.json` 与 `pipeline_state.json`
  对**被人工归档过**的 run **两个都会漏** —— 前者漏在文件被搬走，后者漏在 state 被回退。**
  产物搬到的是 `<体系>/_backup_<rep>_<stage>_<时间戳>/` 与 `_trash_*`（**与 rep 目录同级，
  不在 rep 目录里** —— `ls */rep*/…` 正是这么漏掉的）。
  全仓 `final_binding_results.json`：现役 rep 目录下 **9 份**，含归档共 **20 份**。

  ### 30 个未完成 run 的归属

  | 对方分类 | 归属 |
  |---|---|
  | [11] 窗口/路径布局不一致 | `BM-01`/`02`/`03`（已修）+ `DATA-02`（`cyclod_ligand3/rep1,rep3` 非法布局已落盘，**待拍板**，重跑也起不来） |
  | [7] `ANALYSIS_INCOMPLETE` | `BM-04`（`cmet_ligand1/rep1`、`p38_ligand2/rep1`，**已拍板：重建**）+ `DATA-03`（`cmet_ligand1/rep2`、`cmet_ligand2/rep1`、`p38_ligand2/rep3`，**已拍板：21/4 续跑**）+ `BM-A` 真采样不足 |
  | [5] `TypeError: int() … not 'NoneType'` | **🔴 整桶签名作废，见下面「6 个半状态」** |
  | [3] `cyclod_ligand1_outer/rep1-3` 窗口 0 非有限坐标 | **`LR-06`**，rep 号逐个吻合。**修法已定但没动手 ⟹ 现在重跑还会照样死** |
  | [3] `thrombin_ligand1/rep1-3` co-ion 数量契约 | **已修**：日志 **02:09** vs `runabfe.py` **03:29** ⟹ 日志早 80 分钟，**重跑即可**（本条唯一可以立刻重跑的一批） |
  | [1] `cmet_ligand1/rep3` attachment 漂移 −0.7592 | `BM-A`，**不是 bug**（判据③下末条异常不变） |

  ### 🔴 6 个 cyclod rep 处于「产物被搬走 + state 回退」的半状态 —— 别直接重跑

  `cyclod_ligand1/rep1,rep2,rep3` + `cyclod_ligand2/rep1,rep2,rep3`：
  **最后一次启动之后一条异常都没有**（实测 6/6 异常数 = 0）。
  它们被归进 `[5] TypeError` 那一桶的 6 条 Traceback **全来自更早的启动**。

  实测 `cyclod_ligand2/rep1`：最后一次启动（09-15 17:51:53）**跑完了**，
  18:35:20 打出 `ΔG_bind = −0.61 ± 0.55 kcal/mol` 并保存了 `final_binding_results.json`；
  但现役目录里**没有**那份文件，`pipeline_state.json` 只剩 3 个 stage
  （`equilibration` / `sampling_dual_attachment` / `sampling_dual_decharging`）——
  20:15 那次人工归档把产物搬进了 `cyclod_ligand2/_backup_rep1_vdw_20260915-2015/`
  （含 `final_binding_results.json`、`vanishing`、`vanishing_2/3`、
  `checkpoints_pipeline_state.json.bak`），现役 state 被回退。

  ⚠️ **所以「重跑即可」对这 6 个是错的** —— 本条第一版据那个作废签名写过
  「`BM-05`② 已修 ⟹ 重跑即可」，**撤回**。直接重跑很可能只是把 20:15 那次归档再做一遍。
  **判据：先查清这 6 个现役目录为什么处在半状态、那次归档想达成什么，再决定重跑还是恢复。**
  （`BM-05`② 本身**仍然是已修的** —— 撤回的是"这 6 个 run 死于该 bug"这个取证，不是那条修复。）

  ### ✅ 修复生效的正面证据（比"旧码跑的所以不算数"强）

  | rep | 最后启动 | 结果 |
  |---|---|---|
  | `p38_ligand1/rep2` | 09-16 **06:29:45** | 07:35 出结果 |
  | `p38_ligand1/rep3` | 09-16 **07:34:59** | 08:40 出结果 |
  | `p38_ligand1/rep1` | 09-16 04:09:10（**旧码**） | 卡住 |

  **同一个体系、同一批输入，新码两个跑完、旧码那个卡住。** 这是 `BM-01`~`03`/`05`
  目前唯一的真机正面证据（其余 7 个跑完的都是旧码启动：`brd4_ligand1/rep2`、
  `brd4_ligand2/rep1-3`、`cmet_ligand2/rep3`、`jnk1_ligand1/rep1`、`p38_ligand2/rep2`）。
  ⚠️ `brd4_ligand2/rep1-3` 的**出结果**时间（05:10/05:45/06:09）在修复之后，但**启动**在
  09-15 下午 —— 只看产物 mtime 会误判成新码。

  ### 本次新登记的两条（都不新开条目）

  1. **`输出目录已被另一次运行独占（目录锁）` 历史出现 4 次，全部出自 `cyclod_ligand1`** ——
     `REL-09` 第一次拿到真机证据，已记到那条下面（该条 09-17 关闭，整段进
     [archive/TODO_closed.md](TODO_closed.md)）。
  2. **`thrombin_ligand2` 连 run 目录都没有**（同为 +1 e 配体）。**输入是齐的**
     （`systems.csv`：`status=ok`、`q_lig_e=+1.000`、43764 原子）—— 就是**一次都没启动过**。

  ### 判据

  在**当前代码**上重跑，按上面①②③三条口径重新分类。**重跑之后仍然失败的才是新信息。**
  可以立刻重跑的只有 `thrombin_ligand1/rep1-3`；
  `LR-06` 那 3 个要等插件修完；`DATA-02`/`DATA-03` 那 5 个等拍板；
  6 个 cyclod 先取证。

  📌 科学结论（`results.csv` 11 行 / 6 体系，ME −4.73、MAE 5.91、RMSE 7.35 kcal/mol）
  **不进本文**（规则 5）⟹ 去 [STATUS.md](../STATUS.md)。登记时必须写明：
  **11 行的时间戳全部早于 2026-09-16T05:03，即 11/11 出自修复前的代码**（来源列：归档 8 / 现役 3）；
  且 `results.csv` 的 11 行与「9 个现役 `final_binding_results.json`」是**两个不同的集合**
  （前者含归档产物与 `evidence` 目录），别混着数。

---

## **S2-A [P2] solver 侧证据在整个采样阶段恒空 ⟹ 三条判据不带电（2026-09-17 真机 ×3）**

  stage 结果**只在 `ANALYZE` 执行时**落盘，而 `ANALYZE`（分支 8）排在补帧那一族之后，
  只要还有窗口缺帧就永远轮不到 ⟹ `_read_stage_result()` 整个采样阶段恒 `None`：

  | 判据 | 输入 | 实际 |
  |---|---|---|
  | 补帧边际增益刹车 | `solver_n_frames_decorrelated` | 全窗 `None` → `marginal_gain_stalled()` 过滤后空列表 → 恒 `NOT_ENOUGH_POINTS` → **永不触发** |
  | 累计 f_k 偏差链（5a-1/5a-2） | `cum_fk_verdict` | 全 `None` → **永不进入** |
  | 跳窗路由（1d / 6） | solver 侧 `skipped_windows` | 恒空 → **永不触发** |

  实测：`cyclod_ligand2/rep1` w5 连批 3 块、`cmet_ligand1/rep2` w0 连批 3 块、
  `cmet_ligand2` w4 批满 4 块 —— 判据量全程 `None`（日志里原样印着
  `逐块判据量=[('vanishing', None), ...]`）。w4 的 4 块是被**块数上限**停的，
  不是被"没增益"停的。

  ⚠️ **别再试图靠把 `ANALYZE` 提前来补**：2026-09-17 试过两次，
  条件宽 → 全量 25 红、收窄 → 11 红（多块盘面正是刹车/上限/退役那族的标准盘面，
  也是生产主要工作区）。已回退，原位留注释。正解是 `AUDIT-S2-02` 的重构。

  附带：诊断词 `NOT_ENOUGH_POINTS` 误导 —— 读起来像"才跑了 1-2 块"，
  实际是"这个判据一个输入都没有"，两者处置完全不同。
  **2026-09-17 已做的那一半（只消除静默，行为一字未改）**：
  `marginal_gain_stalled()` 现在把「一个输入都没有」报成 `NO_CRITERION_INPUT_AT_ALL`
  （附 `inoperative_reason`），与「点数不够」`NOT_ENOUGH_POINTS` 分开——两者处置相反：
  前者继续跑就会有数据，后者**再跑多少轮也不会有**。
  `plan()` 每轮恒常输出 `inoperative_criteria`，列出哪些窗口的哪道刹车本轮没带电。
  ⚠️ **没有换替代判据**：自检侧 `min_n_eff_over_g` 单次读数噪声 34×，
  拿它当**停止**判据会误杀，而误杀代价当天已实测（两次生产跑连崩）。
  剩下的根因仍归本条 / `AUDIT-S2-02`。
  完整记录 → [STAGE2_CONTROLLER_WAVE_2026-09-17.md](STAGE2_CONTROLLER_WAVE_2026-09-17.md) §3.A

> ### ✅ 已关闭的 4 条整段移进 [archive/TODO_closed.md](TODO_closed.md)（2026-09-18 搬）
> 
> | 条目 | 一句话 |
> |---|---|
> | `S2-B` | 缺窗覆盖被纯补帧挡住 |
> | `S2-E` | 插 λ 对末窗是空动作，且边际停滞分支不考虑拆窗 |
> | `S2-I` | 末窗被推到 `K == hi` 就卡死：既不拆、也再插不进 |
> | `S2-F` | 布局动作没有任何事后有效性检查 |

---

## **AUDIT-S2-02 [P2] 判断函数已经 1795 行 / 63 个 return —— 这是所有「漏改一处」的共同成因。**
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

## **BUD-06 [P2] 生产预算的「已用」是 lifetime 累计 —— 这不是接线错误，是一个口径分歧。** **需要维护者拍板，代理不要自行改。**
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

---

# 2026-09-23
> 2026-09-23 关闭 3 条，原文整段移入，**一字未改**；关闭记录附在每条后面。

| 条目 | 原在 |
|---|---|
| `REPRO-01` | `TODO.md` §3 |
| `REPRO-02` | `TODO.md` §3 |
| `BOX-02` | `TODO_P2.md` §4 |

---

## **REPRO-01 [P1] 复现路线的「预热不收敛也走临时生产」分支从来没生效过 ——
  两个开关只接了一个。**

  **症状**：2026-09-23 `cyclod_ligand2/rep1` 第 3 段窗口 3 预热未收敛，
  `raise IBSWarmupConvergenceError` 当场打死整条流水线。而复现路线的设计明写这种情况
  应当走临时生产（`DEBUG_REPRO_MODE_2026-09-22.md` §4 表格第 3 行）。

  **为什么会这样**：那条分支的守卫是
  `ibs_engine.py:18676` `if provisional_authorized and debug_provisional_on_warmup_failure:`，
  两个开关分别接：
  · `provisional_authorized` ← `provisional_production_windows`，复现模式下由
    `abfe_pipeline.py` 的 `_run_stage2_once` 默认授权**全部**窗口 ✅ 接对了；
  · `debug_provisional_on_warmup_failure` ← `abfe_pipeline.py:6722`
    `bool(kwargs.get("stage2_frozen_layout", False))`，读的是
    **`_run_dual_lambda_stage` 自己的 `kwargs`**。而 `_run_stage2_once` 向它传参是
    **逐个具名转发**，`stage2_frozen_layout` 不在转发清单里 ⟹ 该键永远取不到
    ⟹ 恒 `False` ⟹ **整条 `and` 恒假**。

  📌 **同一形状在同一段代码里已经犯过一次**：`stage2_independent_endpoint` 因为漏在
  同一张具名转发清单里，`run_full_pipeline` 根本打不开它（该处 2026-09-22 的注释还在，
  就在新加的这行上方 6 行）。**根因是"逐个具名转发"这个形式本身**：加开关的人改的是
  读侧默认值，没人记得回来加转发。⟹ 关闭本条不等于修完这一类，见判据第 3 条。

  **建议的修法（未落盘）**：把 `stage2_frozen_layout` 加进 `_run_stage2_once` 的
  具名转发清单 —— 紧挨着 `stage2_independent_endpoint=kwargs.get(...)`
  （`abfe_pipeline.py:17876`，那条 09-22 的注释就记着同一形状的上一次）。
  生产路线不置位这个键 ⟹ 行为逐字不变。
  📌 2026-09-23 实测过这一行确实能让分支活过来，但**当天 revert 了**，仓库里没有。

  **判据**：
  1. 复现模式下人为制造一次 `f_not_converged`，日志出现
     `[临时生产][debug 路线]` 而不是 `IBSWarmupConvergenceError`，且 run 跑完；
  2. 落盘记账一个不少：`bias_status=provisional_production`、
     `f_k_evidence_status=indeterminate`、`warmup_failure.json` 保留；
  3. **把这张具名转发清单跟 `_run_dual_lambda_stage` 的读侧 `kwargs.get(...)` 对一遍**，
     列出还有哪些键只有读侧没有转发侧（这次是第 2 个被抓到的，不会是最后一个）。

  ### REPRO-01 关闭记录（2026-09-23）
  - **源码**：`abfe_pipeline.py` `_run_stage2_once` 的具名转发清单已有
    `stage2_frozen_layout=kwargs.get("stage2_frozen_layout", False)`（早上 revert 后重新落盘）。
  - **判据 1–2 真机满足**：`cyclod_ligand2/rep2`（09-23 跑完，ΔG_bind −16.16 kJ/mol）的
    `launch.log` 里窗口 4 预热 `f_not_converged` 后打出 `[临时生产] 窗口 4 已被调用方显式授权…`
    而不是 `IBSWarmupConvergenceError`，run 跑到 `final_binding_results.json`；
    `checkpoints/ibs_state_vdw_window_4.json`：`bias_status=provisional_production`、
    `f_k_evidence_status=indeterminate`；`vanishing/dual_window_4_vdw_warmup_failure.json` 保留。
  - **判据 3（对转发清单）**：用 AST 把 `_run_dual_lambda_stage` 读的 `kwargs.get` 键与三个调用点的
    具名实参对了一遍。**抓到第 3 个同形状的洞**：`stage2_accept_recalibrated_f_k_without_gate`
    （runabfe 会把 config 放进 kwargs、读侧也读，唯独 `_run_stage2_once` 不转发 ⟹ 开关从未生效），
    已补转发（默认 False，未设置时行为逐字不变）。其余只读不转发的键（`early_stop_*` 8 个、
    `n_windows`）上游**没有任何地方**设置，恒取默认值，不是 bug。
  - **防复发**：`tests/test_pre_equil_final_state_box_2026_09_23.py::test_stage2_forwards_every_config_key_the_stage_reads`
    —— runabfe `out[...]` 会设、`_run_dual_lambda_stage` 会读的键，`_run_stage2_once` 必须转发。

---

## **REPRO-02 [P1] f_k 固定节奏不锚定就是一条自激回路 —— 每圈增益 2–4×。**

  **症状**：同一个 run，窗口 3（λ_vdw 0.40→0.20，6 态，global 15–20）逐段恶化，
  第 3 段预热烧光 525000 步、39/50 次权重更新仍未收敛：

  | | 段1 | 段2 | 段3 |
  |---|---|---|---|
  | local-MBAR loose gate 残差 kJ/mol（门 10.0） | 1.735 | 7.770 | **17.271 ❌** |
  | f_k 种子跨度 kJ/mol | 6.1（pilot） | 11.8 | **40.4** |
  | 生产帧里端点态 s15 的重加权 ESS | 24.38/500 | **1.98/500** | — |
  | 单帧最大权重 | 0.062 | **0.694** | — |
  | 窗口验收量 `min N_eff/g`（门 10） | 0.664 | 0.084 | — |
  | 本段可用预热预算（账本累计不清零） | 955k | 740k | 525k |

  **为什么会这样**（回路，每一环都有实测）：
  ```
  f_k 偏 → 占据塌向解耦端 → 端点态重加权 ESS 掉（500 帧 → 1.98）
        → 携权重的帧只剩个位数 → 去相关 stride 正好把它们抽掉
        → F_端点 被单边高估 → 新 f_k 跨度更大 → 下一段占据塌得更狠 ──┘
  ```
  关键是**估计量的形态**：重解量是 `−kT ln⟨e^{−Δu/kT}⟩`（`ibs_engine.py:21691`
  `recalibrate_f_k_from_production` → `:22314` `_solve_single_window_local_mbar`），
  **指数均值的 log**，而这个窗口 `mean(u_k − u_mix)` 在 s15 上是 **80.8 kJ/mol = 32 kT**
  ⟹ 这个"平均"实际是个 **max**：丢帧只能把 F 往上推，**误差单向且无界**。
  实测——同一批 500 帧，**只改抽样密度、别的什么都不动**：

  ```
  全 500 帧   f_k = [ 1.48, 1.41, 1.37,  1.09, -0.67,  -4.68]
  stride=5    f_k = [ 3.71, 1.37, 0.44,  0.20, -1.03,  -4.69]
  stride=12   f_k = [11.31, 4.55, 0.27, -2.41, -4.97,  -8.75]
  stride=23   f_k = [20.74, 8.38, 0.19, -5.26, -9.85, -14.20]   ← g=23.6，代码实际用的
  上机的真实种子   [24.79, 9.88,-0.40, -7.13,-11.57, -15.58]
  删掉权重最大的那一帧(idx 81)：[3.69, 1.76, 0.94, 0.41, -1.40, -5.40]
  ```
  `f_k[s15]` 从 1.5 涨到 20.7，**物理没变一个字**，纯粹是去相关把携权重的帧（79/81/86，
  合占 s15 权重的 89%）抽掉了。**端点态的自由能是由单独一帧定的。**

  📌 **为什么"多解几次取平均"是反向修正**（别再提这条）：`f_k` 是个 log，
  Jensen 恒有 `mean(ln x) ≤ ln(mean x)` ⟹ **对 f_k 取平均永远比合并帧给出更高的 F**。
  实测同一批 500 帧：切 5 块各自解再平均 → `f_k[s15]=12.74`；直接合并 → `1.48`，
  **差 8.6×**。要平均就平均**帧**，别平均 `f_k`（合并帧还必须按
  `recalibrate_f_k_from_production` docstring 末段补齐交叉能量，两段不是同一个采样分布）。
  📌 **而且合并帧也救不了已经塌掉的段**：段1 端点 ESS 24.38 + 段2 的 1.98 = 26.36，
  段 2 那 500 帧贡献 ≈ 0。问题不是"样本不够多"，是那些帧**不携带端点信息**。

  **回路里三处本该有阻尼，全是失效的**：
  1. 采纳判据 `abfe_pipeline.py:14638` `if shift >= float(min_adjacent_shift_kJ_mol)`
     —— **增益是正的**：解得越离谱位移越大，越必然被采纳。复现模式传阈值 0.0，
     等于全程直通。
  2. `ibs_engine.py:21811` `heldout_candidate_verdict()` —— 「候选必须在 held-out 连续
     时间块上改善最差态 `N_eff/g` 才真正重标定」，**正是为这个失效写的**，
     **全仓零调用**（`grep -rn heldout_candidate_verdict *.py` 只有定义一行）。
  3. `min_ess_ratio` —— `_solve_single_window_local_mbar` 算了、返回了、**没有任何
     调用方读它**。

  **建议的修法（未落盘）**（仅 debug 路线，生产路径不传锚点 ⟹ 逐字不变）：
  给 `_recalibrate_f_k_and_resample_segment` 加一个 `f_k_anchor_by_window` 形参，
  每轮重解只当**相对固定锚点的有界修正** `f = anchor + cap(f_solved − anchor)`。
  三个落点（行号是 revert 后的当前值，仓库里**没有**这些代码）：
  · 应用点 —— `recalibrate_f_k_from_production()` 返回之后、
    `abfe_pipeline.py:14583` `if shift >= float(min_adjacent_shift_kJ_mol)`
    那道采纳判据**之前**（改 `out["f_k"]` 而不是只改 `seeds`，否则 `f_k_after`
    落盘的跟真正用出去的不是同一份），且必须同步重算 `max_adjacent_shift_kJ_mol`；
  · 构造点 —— 复现节奏那个 `if` 里（`abfe_pipeline.py:18318`），逐窗口调
    `estimate_f_k_from_pilot_ti()` interp 到本窗口的 λ 上；
  · 传入点 —— 同一段里调 `_recalibrate_f_k_and_resample_segment` 的那处。
  锚点 = `estimate_f_k_from_pilot_ti()`，即热力学长度那条 `∫⟨∂U/∂λ⟩dλ`，
  **也就是段 1 用的那个 pilot 热启动种子**（已从
  `checkpoints/preopt_dual_vanishing.json` 的 `path_diagnostics.pilot_points`
  逐位重算核对过）。

  **为什么锚点治得了而重解治不了 —— 是估计量形态的区别，不是运气**：

  | | pilot-TI 锚点 | 生产帧重解 |
  |---|---|---|
  | 数学形态 | **有界观测量的均值**（梯形积分 `∫⟨∂U/∂λ⟩dλ`） | **指数均值的 log** |
  | 样本 | 4 个 pilot 点 × 600 个导数样本 | 端点态 ESS = 1.98 帧 |
  | 误差 | **±0.3 kJ/mol，双向** | **+19 kJ/mol，单向** |
  | 依赖上一段的输出吗 | **不依赖** ⟹ 反馈边直接断开 | 依赖 ⟹ 自激 |

  **cap 取 8 kT 的理由**（建议新增一个常数，例如
  `ibs_engine.py` 的 `IBS_REPRO_ANCHOR_EXCURSION_KT`，挨着
  `IBS_MAX_APPLIED_PAIRWISE_STEP_KT`（`:7974`）放；**仓库里现在没有这个常数**）：
  ⚠️ **cap 在指数里**，2 kT 只等于极端两态权重比改 `e²=7.4×`。两个实测把它夹住：
  · **下界 7.4 kT** —— 段 1 的在线学习器**从这个锚点出发**实际走了 18.4 kJ/mol 才到达
    健康的 f_k（那段生产端点 ESS 24.38/500）。锚点的**幅度**没错（5.6 kT vs 学到的
    2.5 kT），错的是**形状**：锚点非单调 `[2.1, 3.8, 4.2, 2.4, -2.7, -9.7]`、学出来是
    单调阶梯，改形状很花 pairwise 跨度。**cap < 7.4 kT 等于禁止重解走到已被证明健康的
    地方。**
  · **上界 13.0 kT** —— 段 2 那次灾难性重解相对锚点的偏移，必须被夹住。
  · 参考：段 2 占据失衡 1406× = 7.2 kT；8 kT ⟹ `e⁸≈2981×`，压得动还有余量。
  ⚠️ **不要复用 `IBS_MAX_APPLIED_PAIRWISE_STEP_KT=2`**（09-23 先错用了一次，理由别重新论证）：
  那是在线学习**每次更新**的步长上限（一段里加几十次），锚定夹的是**一次性总偏移**，
  两个量不同；而且 2 kT 落在上面那个 7.4 kT 下界以内，直接把健康修正也掐死。

  **判据**：
  1. 复现模式下同一体系跑满 `stage2_repro_fk_cycles` 轮，窗口 3 的 loose gate 残差
     **不再逐段放大**（1.735 → 7.770 → 17.271 这条曲线消失）；
  2. 每轮日志有一行说明本轮锚定结果（建议标 `[f_k 锚定][debug 路线]`），且
     `f_k_after` 落盘的就是夹取后真正用出去的那份（不是夹取前的）；
  3. 拿不到 pilot 锚点时打 WARN 并说明退回链式的后果，不静默降级；
  4. **生产路线（不传锚点）诊断逐位不变。**
  5. 测试要断言**流水线实际用的那个 cap**，不能只打桩调
     `IBSSampler._apply_pairwise_cap()` 并把常数当入参传进去 —— 09-23 那版测试就是
     这么写的：它验的是夹取函数本身，**谁把 cap 从 8 kT 改成 20 kT 都不会红**。
     （那份测试已随 revert 删除，这里只留教训。）

  ⚠️ **不要顺手把锚定也接到生产路线上**：生产路线的重标定由控制器 `decide()` 5a 决定，
  它有自己的节奏与判据；两套并存正是本仓最贵的那个形状（「同一不变量四份实现」）。
  要接得先把 `heldout_candidate_verdict()` 那条真正的验收接上（上面阻尼第 2 条），
  而不是再加一个夹子。

  ### REPRO-02 关闭记录（2026-09-23）
  **落地的不是上面写的「锚定」**，而是：逐轮左/中/右倾斜（`stage2_repro_fk_tilt_signs`，默认
  `[1, 0, -1]`）+ 重标定点估计用全帧（不再按去相关 stride 丢帧 —— 对准回路里「丢帧只能把 F 往上推」
  那一环）。测试 `tests/test_repro_fk_tilt_and_full_frame_2026_09_23.py`。
  - **判据 1 真机满足**（`cyclod_ligand2/rep2` 复合物腿 vanishing，同体系同窗口）：窗口 3 loose gate
    残差 **2.457 → 8.519 → 5.738** kJ/mol（修前 1.735 → 7.770 → **17.271 ❌**），三段全过门、
    不再逐段放大，run 跑完（ΔG_bind −16.16 ± 4.53 kJ/mol = −3.86 kcal，实验 −4.04）。
    每段日志都有 `[f_k 倾斜][debug 路线] 窗口 k：偏左/偏右 Δ=… 生产用 […]`。
  - 判据 2/3/5 是为锚定方案写的（锚定行、pilot 锚点缺失 WARN、cap 常数），对倾斜方案不适用；
    判据 4（生产路线逐位不变）：倾斜只在复现模式生效、且键不进 stage 指纹（同一测试文件）。
  - ⚠️ **遗留，不属于本条**：该 run 仍 `results_untrusted`，唯一原因是**窗口 3 末态支撑**：
    global state 20（λ_vdw=0.20）重加权 abs ESS **1.08**、top-1% 原始权重 **0.92**、
    stage 单参考 ESS 比 0.0055。自激回路断了，但这个窗口对自己末态的覆盖仍然差 ——
    更像是 Δλ=0.04 在软核区太粗（布局问题），不是 f_k 节奏问题。要追的话另开条目。

---

## **BOX-02 [P2] stage-2 主窗口 checkpoint 续验时恢复的是 checkpoint 自带的旧盒。**
  09-23 生产盒修复（`TODO.md` 的 `BOX-01`）给窗口**能量缓存**加了盒体积门，但
  `calibrated_pending_validation` / `frozen_validation_indeterminate` 状态下
  `_main_window_checkpoint_is_usable` 命中就直接恢复 Context（含盒），manifest 只绑
  `win_sys_xml`（System 默认盒不随生产盒变）⟹ 修复前写下的 checkpoint 仍会把旧盒带回来。
  修法：manifest 加当前 `resolved_box` 体积，与能量缓存门同一口径（rel 1e-9）。
  只影响「中断在续验中途」的旧窗口，所以不挡 `BOX-01`。

  ### BOX-02 关闭记录（2026-09-23）
  `ibs_engine._build_main_window_checkpoint_manifest` 加 `box_volume_nm3`（round 1e-6），调用点传
  `run_all_windows` 里算好的 `_current_box_volume_nm3`（与窗口能量缓存门同一个量）。旧 manifest
  无此键 ⟹ 不匹配 ⟹ 拒绝恢复。测试
  `tests/test_pre_equil_final_state_box_2026_09_23.py::test_main_window_checkpoint_manifest_binds_box_volume`。
  ⚠️ 真机未跑。

---

# 2026-09-18
> **本文不是待办。** 2026-09-18 当天关闭的 2 条整段从 `TODO.md` 移进来，**一字未改**。
>
> 📌 「某条缺陷修没修」的权威仍然是**源码**，不是这里的 `- [x]`。
>
> ⚠️ **这两条真机零复验**：修复落盘于 09-18 08:30–08:36，而当时在跑的
> `cyclod_ligand1_outer` rep2/rep3 是 09-17 起的（commit `9a18ded`）⟹ 吃不到。
> 复验要看 09-18 16:11 之后起的那批（`p38_ligand1` rep1/rep2/rep3）。

| 条目 | 原在 |
|---|---|
| `S2-N` | `TODO.md` §1 |
| `S2-O` | `TODO.md` §1 |

---

## **S2-N [P1] 块账**写侧记 4 个动作、读侧只拦 1 个** ⟹ 重标定/探针/临时生产扣配额却从不过闸（2026-09-18 真机）**


  分诊全文（含逐轮台账与证据）：
  [STAGE2_BENCHMARK_CRASH_TRIAGE_2026-09-18.md](STAGE2_BENCHMARK_CRASH_TRIAGE_2026-09-18.md) §2。

      abfe_preoptimizer.py:3526  _BLOCK_CHARGING_ACTIONS = (RUN_PRODUCTION,
                                     PROBE_REANCHOR_EPOCH, RECALIBRATE_FK,
                                     PROVISIONAL_PRODUCTION)   ← 这 4 个都扣账
      abfe_preoptimizer.py:5533  if action == "RUN_PRODUCTION":  ← 只有这 1 个过 _frames_admission

  真机 `cmet_ligand2/rep3` w4：2 块 `RUN_PRODUCTION` + 3 块 `RECALIBRATE_FK` = **5 块**（上限 4）。
  那三块重标定（轮 12/13/14）还各开一个新段，把累计 500k 步作废回 250k ——
  正是 `S2-H` 注释里记载的形状。`S2-H` 的次数闸在第 4 次才拦，**前三次按定义免费**，
  而三次已经够把块配额吃光：**次数闸与块账是两本互不知情的账。**

  ⚠️ 这条与 `S2-L` 是同一族（块账语义），但 `S2-L` 修的是「账本身漏记」，
  这条修的是「记了但没人拿它拦」。别合并。

  **已修（2026-09-18）**：`plan()` 的闸条件改成 `if action in self._BLOCK_CHARGING_ACTIONS:`
  （写侧与读侧合成同一张表）；`_frames_admission` 新增 `hard_cap_only`，
  非 `RUN_PRODUCTION` 的三个**只过第一道**（块数硬上限）。
  ⚠️ **三道闸不能整套套给它们**：第二道（边际增益刹车）与第三道（射程闸）问的都是
  「同分布加帧还有没有用」，而 `RECALIBRATE_FK` 恰恰是「加帧没用」时的**对症动作**——
  拿"加帧没用"去否决它方向完全反了。块数硬上限是**资源账**，与动作是什么无关。
  测试：`tests/test_block_gate_covers_all_charging_actions_2026_09_18.py`
  （含方向钉子 `test_a_stalled_frame_series_does_not_veto_recalibration`）。
  两条改动都做过还原变异验证；全量 2736 passed / 0 failed。
  ⚠️ **尚未上机复验** —— 本条的真机判据（`cmet_ligand2/rep3` 重放时 w4 第 5 块
  必须被改写成终态）要等下一批 benchmark。

---

## **S2-O [P1] 退出前的全路径 ANALYZE 在有 stale 证据时被整个跳过 ⟹ 整跑零产出 + 自锁（2026-09-18 真机）**


  分诊全文：[STAGE2_BENCHMARK_CRASH_TRIAGE_2026-09-18.md](STAGE2_BENCHMARK_CRASH_TRIAGE_2026-09-18.md) §3。

  位置 `abfe_pipeline.py:12597-12618`（判据是 `missing_windows` / `stale_layout_evidence`
  任一非空就跳）。真机四个 run 的 `outcome.final_analyze_skipped` 全部非空
  （`cmet_ligand2/rep1`、`cmet_ligand2/rep3`、`brd4_ligand1/rep1`、`cmet_ligand1/rep2`），
  随后 `_assert_stage_result_sane`（`abfe_pipeline.py:8407`）抛 RuntimeError、进程退出、
  **连诊断都没有**。

  **自锁**：跳过 ANALYZE ⟹ `solver_n_frames_decorrelated` 永远算不出来 ⟹ 台账里那串
  `[('vanishing', None), ...]` ⟹ `S2-A` 的边际增益刹车永远不带电 ⟹ 下次 `--resume`
  逐字复现同一条链。

  ⚠️ `12584` 那段跳过的理由（「跑它就是重新采样，绕过控制器刚做的停止决定」）**本身成立，
  别推翻它** —— 没算到的只是代价：不是"少一份诊断"，是整跑炸掉。

  ⚠️⚠️ **先前写在这里的两条"候选修法"和那条判据都是错的，已作废：**
  ① 收窄判据跑不通 —— `run_once` 对 stale 窗口**就是会重采**，那正是跳过它的理由；
  ② 「别让 `8407` 炸」**不许做** —— 路径确实不完整，`_assert_stage_result_sane`
  是硬不变量的 fail-closed（缺窗口的和不是 ΔG），放宽它等于让截断的 ΔG 流进结果。
  原判据「必须产出可读的 stage 结果而不是 RuntimeError 退出」**把正确行为当成了缺陷**。

  **已修（2026-09-18）**：跳过的决定保留（它是对的），改成**跳过时落一份零采样的诊断**
  —— `ctl.write_comparison_manifest()`（`comparison_manifest()` 自称「纯聚合，不新算任何
  东西」），逐窗验收量 / ΔF±σ / 块账 / 动作序列全在里面，路径记进
  `outcome.final_diagnostic_manifest`。文件名 `controller_comparison_manifest.json`
  **刻意不以 `stage2_` 开头**（审计 #63），不会被 `_read_stage_result()` 的兜底 glob 误吃。
  测试：`tests/test_block_gate_covers_all_charging_actions_2026_09_18.py::test_the_skipped_final_analyze_still_leaves_a_diagnostic`。

  **范围别记大了**：这条修的是「停下时零产出」。它**不解决** `S2-A`
  （`solver_n_frames_decorrelated` 只存在于 stage 求解结果里，而这条路径按定义跑不了求解），
  也**不阻止** `8407` 抛错。

  🔺 **[2026-09-18 批次跑完后升级：`S2-O` 优先于 `S2-N`]** `cyclod_ligand3/rep3` 证明
  控制器有**两种**把自己停死的方式 —— 块配额吃满（`S2-N`）、以及 `S2-H` 效果闸拒发
  （那次是**正确行为**：`RECALIBRATE_FK` 把 w1 从 1.255/1.575 推到 0.378，比值 0.27 < 0.9）。
  两种停法的止血口是同一个 `S2-O`：`final_analyze_skipped` 非空 ⟹ 零产出 ⟹ `8407` 抛错。
  ⟹ **无论控制器因为什么停下，都不该零产出**，这条先做。
  分诊 §5.2：[STAGE2_BENCHMARK_CRASH_TRIAGE_2026-09-18.md](STAGE2_BENCHMARK_CRASH_TRIAGE_2026-09-18.md)。


---

# 2026-09-17
> **本文不是待办。** 2026-09-17 当天关闭的 6 条条目整段从 `TODO.md` / `TODO_P2.md`
> 移进来，**一字未改**（每条原文里的日期、判据、实测数字都保留）。
> 按 [docs/README.md](../README.md) 的规矩：已关闭的条目不留在待办文件里。
>
> ⚠️ 与 [TODO_closed.md](TODO_closed.md) 那次不同 ——
> 那一份是**一次对账**（13 条缺陷早已不在代码里、只是没人回来打勾）；
> 这一份是**当天真做掉的**，每条都有同日的改动与验证。
>
> 📌 「某条缺陷修没修」的权威仍然是**源码**，不是这里的 `- [x]`。

| 条目 | 原在 | 一句话 |
|---|---|---|
| `REWIND-01` [P1] | `TODO.md` §1 | `IMMUTABLE_REWINDOW` 曾是 13 个声明动作里唯一一个 `decide()` 发不出的；接在四条死线各自的兜底**之前**，一个父窗只切一层 |
| `TEST-01` [P2] | `TODO_P2.md` §1 | 全量 198 个测试文件都进了 `cpu_only` / `needs_gpu` 之一；判据用**收集数对账**，不是 grep 文件 |
| `LR-03` [P2] | `TODO_P2.md` §4 | 整片 EXP-010/011 的 CLI 表面已删，移进 `archive/` |
| `LR-04` [P2] | `TODO_P2.md` §4 | 加了 `--only-ligand-in-water` 调试入口 |
| `REL-08` [P2] | `TODO_P2.md` §5 | 基础环境本来就存在，欠的只是口径 |
| `REL-09` [P2] | `TODO_P2.md` §5 | 三项里两项早已落地，第三项（输出目录独占锁）是**已做出的判定**，不是缺口 |

> ### 留下的规矩
>
> 1. **死动作要接在各自兜底之前，不是在链尾加一条分支** —— 否则又变成「按书写顺序决定语义」（`REWIND-01`）。
> 2. **一个父窗只切一层**：不设深度上限就是新的无限循环；这条限制**不许放宽** —— 触发它的判定本身噪声 34×，会误触发在没问题的窗口上。
> 3. **测试选择口径的判据是收集数，不是 grep 文件**：同一个文件里可以既有带标记的测试、又有漏网的（`TEST-01` 就是这么发现的）。
> 4. **「真机证据」也会指向反面**：`REL-09` 原文引的那 4 次目录锁报错，咬的是**续跑**、来自后来被删掉的那把锁 ⟹ 它支持「删得对」，不是「该补回来」。

---

## **REWIND-01 [P1] ✅ 2026-09-17 已修** —— `IMMUTABLE_REWINDOW` 曾是 13 个声明动作里唯一一个 `decide()` 发不出的 ⟹ 一类盘面本来有对症动作却直接停机（2026-09-16 AST 实测）

  **修法**（`abfe_preoptimizer.py`）：
  ① 子窗划分 `vanishing_rescue_ranges()` 从 `abfe_pipeline.ABFEPipeline._build_vanishing_rescue_ranges`
  搬到 **模块级**（控制器 import 不动 abfe_pipeline：循环依赖 + 它 import openmm），
  pipeline 那个静态方法转成一行转发；成本表里那份 `2 if (b-a) > 2 else 1` 的复算随之删掉
  ⟹ 切法全仓一份。
  ② 新增 `Stage2RepairController.rewindow_feasible()` / `rewindow_children()`：
  可行 = **切得出两个各 ≥2 态的重叠子窗**（K ≥ 3）**且** 该父窗不在 `parents_done`
  （**一个父窗只切一层** —— 不设深度上限就是新的无限循环）。
  ③ 发出点接在 **四条死线各自的兜底之前**（1e 预热预算耗尽 / 5a-2 held-out REJECT /
  边际增益停滞 / 5b 偏斜类自检），不是在链尾加第 33 条分支 —— 否则又是「按书写顺序决定语义」。
  不可行时原来的 `NO_FEASIBLE_ACTION` 照旧，只是理由里多带一句有界重窗为什么不可行。

  **正向耦合**：同日分窗目标改成 `state_count`（min-max K）之后 23 个蛋白 rep 全变成
  `[5,5,5,5,5]` ⟹ **每个窗口 K=5 ⟹ 对每一个窗口都可行**（老布局 K=4 的窗只能切 3+2）。
  已钉成 `test_bounded_rewindow_is_feasible_for_every_window_of_a_state_count_layout`。

  ⚠️ **别期待它改善 ΔG。** 死线开了 = 循环能继续走，不等于答案更准；而且触发它的判定
  本身噪声 34×（[AUDIT_GATES_AND_CRITERIA_2026-09-17.md](AUDIT_GATES_AND_CRITERIA_2026-09-17.md)）⟹ 它会被误触发在其实
  没问题的窗口上。**「一层」那条限制因此不许放宽。**

  **验证**：3 条 strict-xfail 标记已删（全绿）；`tools/audit/static_controller_2026-09-17.py`
  死动作 1 → 0；全量 `2653 passed / 0 failed`（此前 2649，+1 新测试 +3 转正）。

  <details><summary>原始条目（存档）</summary>


  **实测**：对 `Stage2RepairController._decide_once` 做 AST 扫描（`plan()` 的首参 +
  `act=` 赋值，含条件表达式两支），13 个 `ACTIONS` 里 12 个发得出，
  **`IMMUTABLE_REWINDOW` 一处发出点都没有**。

  **它不是"还没实现"—— 是缺唯一入口**，周边全是齐的：
  - 成本表：`abfe_preoptimizer.py` 约 **4632** 有 `elif action == "IMMUTABLE_REWINDOW"`
    的逐子窗首块计价（审计 #44，还特地修过"硬编码 2 个子窗"的错）；
  - 执行器：`abfe_pipeline.py` 约 **12162** 的 `elif act == "IMMUTABLE_REWINDOW"`
    → `_immutable_rewindow_step()`（约 12867）完整实现；
  - 子窗调度：`sampling_units` / `rw:<id>:<i>` / `stage2_rewindow_ledger.json` 全在，
    并有 12 条测试覆盖（`tests/test_rewindow_sampling_units.py` 等）。

  **后果**：λ 表顶到溢出槽上限（末窗 K = 2·max−1）且失败落在**中间窗**时，
  拆末窗不对症、插 λ 不可行 ⟹ 本该退到**有界重窗**，现在直接 `NO_FEASIBLE_ACTION`
  停机。这正是 `BM-04` / `DATA-02` 那类死局的形状。

  **为什么一直没人发现**：本仓有"死出口"守卫（`test_no_dead_exits_left_in_the_declaration`）
  但**没有对应的"死动作"守卫**。2026-09-16 已补上
  （`tests/test_stage2_controller_vocabulary.py::test_no_dead_actions_left_in_the_declaration`）。

  **判据**：把发出点接回 `_decide_once` 的可行性链尾部（拆窗与插 λ 都不可行时的兜底），
  下列 3 条 `strict-xfail` 变红即修好，届时删标记：
  `test_no_dead_actions_left_in_the_declaration`、
  `test_stage2_repair_controller.py::test_fixed_lambda_table_middle_window_falls_back_to_bounded_rewindow`、
  `test_audit_2026_09_14_controller_budget.py::test_an_unknown_production_remainder_does_not_reject_a_bounded_rewindow`。

  </details>

---

## **TEST-01 ✅ 2026-09-17 关闭 —— 全量 198 个测试文件都进了 `cpu_only` / `needs_gpu` 之一。**
  改法：22 个无标记文件加 `pytestmark = pytest.mark.cpu_only`（8 个顺带补 `import pytest`），
  外加 `test_exp019_softlift_loro.py` 里那个**漏网的单个测试** —— 该文件已有
  `@pytest.mark.needs_gpu` 挂在另一个测试上，所以按文件 grep 看是"有标记的"，
  实际第一个测试两档都不属于。📌 **这就是为什么判据要用收集数对账，不能用 grep 文件。**

  **判据实测（`CUDA_VISIBLE_DEVICES=""`）**：
  `-m "not cpu_only and not needs_gpu"` → **no tests collected (2666 deselected)**；
  `cpu_only` + `needs_gpu` 并集 = 全量 **2666**。
  22 个新标文件在**无 GPU 可见**下 **235 passed**（不是"本机跑过了"——卡被藏掉了）。

  ⚠️ 判成 `cpu_only` 的依据是它们全都只读源码 / AST / monkeypatch：
  `test_lr06_…` 读 `.cpp` 文本、`test_nvcc_discovery_…` 全程 monkeypatch 环境变量，
  两个名字带 CUDA 的都不建 Context。

---

## **LR-03 ✅ 2026-09-17 关闭 —— 整片 EXP-010/011 的 CLI 表面已删，移进 `archive/`。**
  📌 **原条目的前提要修正**：不是「实现在发布清理时被移出的 `archive/` 里」，而是
  **`archive/` 从来没进过本仓库** —— 目录不存在、`git log --all -- archive/` 为空。
  13 个转发壳是 2026-08-31 `169514e`（主线迁移）引入的，1718 行真实现留在旧工地
  `Atenolol-rank11/archive/outer_lambda_exp010_exp011_legacy.py`。
  ⟹ 这些子命令**从迁移那天起就是 `ImportError`，一次都没跑起来过**；
  「要么补实现」在本仓库等于把旧库搬回来重新纳管，而「legacy 读 `lambda_shield` → `TypeError`」
  那半条**永远够不着**（`ImportError` 先炸）。
  ⟹ 而且不止 `sample-hard-window-scratch` 一个入口：**9 个子命令**
  （`exp011-coverage` / `exp011-fit-pmf` / `exp011-umbrella-sample` / `exp011-reweight-umbrella` /
  `exp010-label` / `exp010-fit` / `exp010-prepare-selection` / `freeze-slow-variable` /
  `sample-hard-window-scratch`）全部撞同一堵墙。
  **已删**：9 个子命令的 argparse 定义与处理块、13 个壳、`__all__` 里 12 个名字、
  `tests/test_outer_lambda_cli.py` 的 `--help` 清单对应行。原文逐字存进
  `archive/outer_lambda_exp010_exp011_removed_2026-09-17.py`（**不被任何东西 import**）。
  ⚠️ `screen-slow-variables` / `compare-slow-variable-screens` / `wp0-select` **没删** —— 不碰这些壳。
  验证：`outer_lambda_neural_basis.py` 9472 → 8109 行；`--help` 27 → 18 个子命令；
  删前逐个数过 13 个壳的外部引用 = **0**。

---

## **LR-04 ✅ 2026-09-17 关闭 —— 加了 `--only-ligand-in-water` 调试入口。**
  📌 **名字是维护者定的**：它跑的是「配体在水中」那条腿，产出 **ΔG_solvent（去耦自由能）**，
  **不是结合能** —— 帮助文案与收尾日志都按这个口径写死，防止有人把这个数当 ΔG_bind 引用。
  定位是**调试**不是生产：跳过 Boresch 估算、带限制力 rebalance 和整条复合物腿，
  只跑配体在水中那条腿，**不产出 ΔG_bind**、不覆盖主 `final_results.json`。
  三道 fail-closed：① 与 `--only-complex-charging` / `--only-boresch-attachment` 互斥；
  ② 与 `--outer-lambda-local-residual-ibs` 互斥（残差运行时是在复合物腿那段构造、
  再按溶剂腿拓扑重建的，跳过复合物腿就拿不到它 —— 抛错，不静默降级成 baseline 溶剂腿）；
  ③ 溶剂腿跑完**直接 return**，不让它走到第 8 节。
  📌 **③ 是刻意的**：比"补个 `None` 往下传"安全 —— 后者每加一个下游分支，
  就多一次把半份结果写成完整结果的机会（本仓库 `validate_final_leg_result` 那批
  fail-closed 就是为这个加的）。
  验证：`tests/test_only_ligand_in_water_lr04.py` 5 passed（控制流契约：开关/默认值/两条互斥/
  复合物腿整段在 `else` 里/早退存在）+ `test_no_undefined_globals` + CLI `--help`。
  ⚠️ **真跑一条腿要 GPU，没跑过** —— 钉住的只是"不跑也能错"的那部分。

---

## **REL-08 ✅ 2026-09-17 关闭 —— 基础环境本来就存在，欠的只是口径。**
  📌 **不用写第三个 yml**：`environment-ci.yml`（43 行手写）就是 CPU 基础环境
  —— python + openmm + pymbar-core + numpy/scipy/mdtraj/rdkit/pdbfixer，
  **不含** CUDA 12.9 工具链 / torch / openmm-torch / MACE / JAX（那些全在
  `environment.yml` 那份 333 行 `conda env export` 全钉死的生产环境里）。
  从 333 行的 export dump 里手术式删 CUDA 行删不干净，所以改的是**文档定位**：
  [GETTING_STARTED.md](../GETTING_STARTED.md) 新增「两个环境文件，装哪个」对照表。
  ⚠️ **`openmm=8.5.2` 的钉死是判定，不是疏漏** —— 8.6 尚不成熟、有已知严重 bug，
  本项目**暂**不使用；**是"暂"不是"永不"**，8.6 的官方 REMD 正是本项目想要的，
  **未来版本可能支持**，缺的是 8.6 本身稳定下来。它与 `free_energy_engine.OFFICIAL_REMD_MIN_OPENMM_VERSION = "8.6.0"`
  **不矛盾**：后者是「官方 REMD 采样器要 8.6 才有」的资格判据，前者是「今天不装 8.6」。
  📌 **本条曾被我误判成 bug 并动手改成下限 `>=8.5.2`（已全部还原）** ——
  理由现在写在 `environment-ci.yml` 与 `GETTING_STARTED.md` 里，别再重开一遍。

---

## **REL-09 ✅ 2026-09-17 关闭 —— 三项里两项早已落地，第三项是**已做出的判定**，不是缺口。**
  逐项回源码核实（本条原文的三项「欠覆盖」全部不成立）：

  | 子项 | 实际 |
  |---|---|
  | SIGTERM/SIGINT | ✅ `abfe_pipeline.install_termination_handlers`（`guard_run_directory` 调用）。`TerminationRequested` 基类是 `BaseException`，正是为了不被全仓 ~220 处 `except Exception` 吃掉。**2026-09-17 真机触发过**：`cyclod_ligand1_outer/rep1` 02:29 的有序退出 |
  | 磁盘预检 | ✅ `abfe_pipeline.ensure_free_disk_for_stage`：stage 开跑前按 `estimate_stage_trajectory_bytes` 估产出，要求 `free ≥ 2×`，不够 **fail-closed 拒绝开跑**（ATT-23 / issue #142）。原文「只有 doctor 里有 disk_usage」是它落地**之前**的状态 |
  | 输出目录独占锁 | ⚖️ **已判定不要**。`guard_run_directory` 的 docstring 写明理由：进程被 SIGKILL 时锁文件残留，而 stale 判定在共享盘上判不出来（PID 跨节点无意义、hostname 不同一律当活着）⟹ **锁挡住的是 `--resume` 本身** |

  ⚠️ **本条原来引的那条「真机证据」是反的**：`输出目录已被另一次运行独占（目录锁）` 那 4 次全在
  `runs/cyclod_ligand1/rep1/launch.log` 的 **2026-09-10** 段，traceback 指向
  `runabfe.py:6267 guard_run_directory` → `abfe_pipeline.py:1855`，即**后来被删掉的那把锁**，
  而且它咬的就是**续跑**。⟹ 那 4 次支持的是「删得对」，不是「该补回来」。**别拿它重开这条。**

  📌 **`RunDirectoryLock` 已于同日删除**（维护者确认「没用」）：生产侧零调用点，
  类 + 5 条锁测试进 `archive/run_directory_lock_removed_2026-09-17.py`。
  同文件的 SIGTERM 与磁盘预检两节**保留**（那两项仍在生产里跑，是本条结案依据）。
  ⚠️ 归档头部记了一条**复核确认的机制缺陷**，将来若有人想把锁接回来必须先看：
  `timeout_s=0.0` ⟹ 父类 `__enter__` 的 `deadline` 就是当下 ⟹ 第一次 `FileExistsError`
  后必抛 `TimeoutError`，**`os.open` 永不重试**。于是 `_break_stale_lock_if_needed()`
  即便成功清掉残留锁，**这一跑照样死**，清理成果留给下一次启动。
  ⟹ 「不等待」（设计如此）与「不重试」（缺陷）被写成了同一件事。
  📌 父类 `_PipelineStateLock`（`timeout_s=10.0`）**不受影响**，`pipeline_state.json` 照常用。

---

---

# 追加：Stage-2 控制器 wave 的 10 条（2026-09-18 补档）

> ⚠️ **这 10 条 2026-09-17 就已关闭，但当天的打扫没收它们** —— wave 的**叙事记录**
> 走了 [STAGE2_CONTROLLER_WAVE_2026-09-17.md](STAGE2_CONTROLLER_WAVE_2026-09-17.md)
> 这条路归档了，而 `TODO.md` / `TODO_P2.md` 里的**条目本身**没人搬，
> 于是在待办文件里多压了一天。2026-09-18 补搬，**原文一字未改**。
>
> 📌 教训：一次关闭有**两个**归档动作（叙事文档 + 条目搬家），做了前者不等于做了后者。

| 条目 | 原在 |
|---|---|
| `S2-G` | `TODO.md` §1 |
| `S2-H` | `TODO.md` §1 |
| `S2-J` | `TODO.md` §1 |
| `S2-K` | `TODO.md` §1 |
| `S2-L` | `TODO.md` §1 |
| `S2-M` | `TODO.md` §1 |
| `S2-B` | `TODO_P2.md` §1 |
| `S2-E` | `TODO_P2.md` §1 |
| `S2-I` | `TODO_P2.md` §1 |
| `S2-F` | `TODO_P2.md` §1 |

---

## **S2-G [P1] 退役判定看不见「从没采过的窗口」⟹ 整跑崩溃（2026-09-17 真机 `cmet_ligand2`）**


  `Stage2RepairController._retirable_window()` 第一行就取错了源：

      recs = {int(w["window_idx"]): w for w in (view.get("windows") or [])}

  `view["windows"]` 只收**盘上有产物**的窗口；从没采过的窗口在
  `view["missing_windows"]`，**不在这张表里**。而"别处还有没有活干"就是遍历 `recs`
  ⟹ 对一个**整窗未采、块配额分文未动**的窗口**恒答否** ⟹ 不退役 ⟹ 交出终态。

  **真机崩溃链条**：w4 跨全部段已批 4 块（上限 4）→ `plan()` 补帧准入拒 →
  `NO_ACTION` + `NO_FEASIBLE_ACTION`（`halt_scope=TARGET_LOCAL`，可退役）→
  `_retirable_window` 找不到候选（w5 未采、不在 `recs`）→ 不退役 → 终态 →
  退出前 ANALYZE 也跳过（`missing_window_5`）→ `_assert_stage_result_sane` 抛
  `RuntimeError`，整跑失败。**w5 一步都没跑过。**

  分支 6b（缺窗 → 补采）在结构上就在下游，`plan()` 里的准入闸先一步把整轮变成终态，
  根本落不到它。

  ⚠️ **这条不是分支顺序问题** —— 只是一个集合取错了源，修它不碰 `_decide_once` 的
  任何分支、不改任何优先级。与 `AUDIT-S2-02` 那批（`S2-A/B/E/F`）性质不同，可单独做。

  判据：布局里有、产物里没有的窗口必须算作"别处还有的活"。
  完整记录 → [STAGE2_CONTROLLER_WAVE_2026-09-17.md](STAGE2_CONTROLLER_WAVE_2026-09-17.md) §3.G

---

## **S2-H [P1] 「总是改变盘面」的动作对停滞保护与 no-op 台账结构性免疫（2026-09-17 真机 `cmet_ligand1/rep1`）**


  停滞保护判的是「同一个动作 + **盘面未变**」（`seen[key] >= 3`）。而 `RECALIBRATE_FK`
  `probe_only=False`、**开新采样段** ⟹ `_disk_signature()` 里的 `segment` /
  `aggregated_segments` 必变 ⟹ `seen[key]` 每轮重置成 1 ⟹ **永远到不了 3**。
  `action_noop_fingerprint()` 含段维度，同理永不匹配。
  ⟹ **这一族动作对唯一的通用刹车免疫**，叠加 `S2-F`（无事后有效性检查）即可无限重标定。

  **真机（31/40 轮里 17 轮花在窗口 4）**：

      20 RECALIBRATE_FK  seg=vanishing    steps=750000  ratio=9.645  ← 离门 10 差 0.355
      21 RECALIBRATE_FK  seg=vanishing_4  steps=250000  ratio=4.182  ← 新段，750k 步作废
      22 RECALIBRATE_FK  seg=vanishing_5  steps=250000  ratio=0.837
      23 RECALIBRATE_FK  seg=vanishing_6  steps=250000  ratio=7.017

  它在第 20 轮再补一块几乎必过；三次重标定把它推到 0.837，全程无人察觉。
  盘上留下 7 个采样段。**定 P1 的理由**：17/31 轮花在一个窗口上，
  这样的烧法会把 40 轮预算耗尽 ⟹ `ITERATION_CAP_NOT_CONVERGED`，拿不到结果。

  ⚠️ **别用「把 `RECALIBRATE_FK` 加进 `_NOOP_GUARDED`」来补**：no-op 判据问的是
  "盘面变没变"，而它**真的变了**（新段是真实产物）。要判的是"变好还是变坏"——
  那是 `S2-F` 的判据。两条必须一起修：只修 F，控制器仍可每轮换新段绕过重复计数；
  只修 H，第 3 次会被拦，前两次仍白烧。

  完整记录 → [STAGE2_CONTROLLER_WAVE_2026-09-17.md](STAGE2_CONTROLLER_WAVE_2026-09-17.md) §3.H

---

## **S2-J [P1] model B 的收尾动作「末窗一分为二」在自治控制器下**整个不存在**（E 与 I 的共同根因）**


  原始设计写在 `abfe_preoptimizer.py:9556`（`feasible_repair_actions` 内）：
  「这一条问的是**溢出槽长到能一分为二了没有**，是 model B 插点的**收尾条件**；
  继续插 λ 让末窗长大即可到达；拆窗只是末窗溢出压不住时的收尾动作。」
  即：插 λ 把末窗养大，到 `K ≥ 2·lo−1`（lo=4 时 K=7）就**一分为二**。

  | | 状态 |
  |---|---|
  | 判据 `split_last_window_in_two` | ✅ 每轮都算，两个真机盘面都答**可拆** |
  | 执行器 `split_window_from_ibs_lse_failure` | ✅ 存在（`abfe_preoptimizer.py:1497`） |
  | **调用者** | ❌ 全仓只有 `abfe_pipeline.py:10933`，在 **path_evolution 的异常处理**里 |

  而自治控制器**显式关闭**那条路径（每份日志都印着「关闭 … path_evolution 的插 λ
  修复分支」），并有断言钉死（`abfe_pipeline.py:17169`）。
  ⟹ **判据活着、执行器活着、动作不存在。**

  控制器自己的 `SPLIT_TAIL_WINDOW` 是**另一个更重的动作**
  （`repartition_tail_from_anchor`：从 anchor 起重分**整个尾段**，要 tail anchor），
  不是「末窗一分为二」。审计 #24 把两个问题分开时判据留在了
  `split_last_window_in_two`，**但没给控制器补上对应的动作**。

  **⟹ `S2-E` 与 `S2-I` 是它的两个症状**：
  `末窗失败 → 插 λ(跨度不变) → 变宽 → 能量+f_k 全废 → 重采重学 → 再插 → 撞 hi 卡死`。
  中间本该有一步「K≥7 就一分为二」。

  ⚠️ 修它**不是**重新打开 `path_evolution`（那条被关是有理由的：同一个决定不许两套
  机制各判一次）。要做的是把「末窗一分为二」作为**控制器的一个动作**补进动作表 +
  执行器分发，判据直接用现成的 `split_last_window_in_two`。**这条一补，E 和 I 自动消解。**

  完整记录 → [STAGE2_CONTROLLER_WAVE_2026-09-17.md](STAGE2_CONTROLLER_WAVE_2026-09-17.md) §3.J

---

## **S2-K [P1] `ANALYZE` 在有过期证据的盘面上炸穿流水线 —— 判据侧知道、执行器侧不看（2026-09-17 真机）**


      [自治] 执行 ANALYZE 失败：ValueError('窗口 4 lambda 内容与当前路径不匹配')
      _solve_merged_segments_if_any → _load_ibs_window_outputs_merged
        → ibs_engine.load_ibs_window_outputs_from_dir:664 → raise ValueError → 炸穿主循环

  控制器**对补帧类动作有这道闸**（分支 1d-0：过期窗口只能重采；停滞降级另有
  `_stale_for_escalation`），理由原文：「拿已有帧重解的动作在构造上都会维度不符，
  真机实测是直接 `ValueError` 炸出流水线」。**但 `ANALYZE` 没有这道闸** ——
  而它恰恰是**跨全部段、拿已有帧重解**的那个动作。

  **判据侧其实完全知道**：`classify_layout_evidence()` 逐 (段, 窗口) 比 λ 身份，
  `read_aggregated()` 把过期的排除出 `merged` 并记进 `stale_layout_evidence`。
  而 `_solve_merged_segments_if_any` 是执行器侧，从盘上**无过滤**加载所有段。
  ⟹ **同一个不变量两份实现**，这次的形态是"判据侧过滤了、执行器侧没过滤"。

  判据：发 `ANALYZE` 之前要么确认没有过期证据、要么让合并求解**按同一份判据跳过**
  过期的 (段, 窗口)；不得让它抛到主循环。
  完整记录 → [STAGE2_CONTROLLER_WAVE_2026-09-17.md](STAGE2_CONTROLLER_WAVE_2026-09-17.md) §3.K

---

## **S2-L [P1] 补帧的块数硬上限被「换布局」重置 ⟹ 补帧实际没有有效上限（2026-09-17 真机）**


  `_production_blocks_total_by_window()`（**硬上限**那本账）和
  `_production_blocks_ledger()`（**边际增益**那本）都按 `path_version` 过滤
  ⟹ **插一次 λ / 拆一次窗，配额就清零、重新发满 4 块**。
  这跟 `BUD-03` 是**同一个 bug 换了一个维度**（那次是"换段退钱"），
  而本函数的定义白纸黑字写着「硬上限 = **资源账**」—— 烧掉的 GPU 不会因为布局变了就回来。

  真机实测（修复后才看得见真实消耗）：

      cmet_ligand1/rep1  w4：旧口径 2 块 → 真实 9 块（上限 4）
      cmet_ligand2/rep1  w5：旧口径 4 块 → 真实 5 块
      brd4_ligand1/rep2  w3：累计 1,250,000 步 = 5 块

  叠加 `S2-A`（边际增益刹车全程没有输入、恒不触发）⟹ 在布局反复演化的 run 上
  **补帧没有任何有效上限**。这正是跨 run 统计里最准那条判据的机制：
  「至少一个窗口累计 ≥500k 步」覆盖 **9/9** 失败（maxK≥6 只覆盖 7/9）。

  **已修**：硬上限那本**不按 `path_version` 过滤**；边际增益那本**仍然要过滤**
  （跨布局比 `min N_eff/g` 没有意义，窗口几何都变了）——两本账的过滤维度本来就不同，
  这正是 BUD-03 把它们拆成两个函数的理由。
  测试：`tests/test_structural_action_budget_2026_09_17.py::test_the_hard_cap_counts_blocks_across_layout_changes`
  ⚠️ 该测试的 fixture 必须用**单调递增**的 `production_steps`：块账按步数去重，
  每个布局都从 250k 重来会被折叠掉，测不到要测的东西。

---

## **S2-M [P1] 「布局过期 + 冻结验证批次打满」= 死锁 ⟹ 整跑终止（2026-09-17 真机）**


  分支 1d-0（过期 ⟹ 只能重采）排在 1e（进不进得去）**之前**，且无条件发
  `RUN_PRODUCTION`。于是：重采要**重新进入**窗口 → 预热门看到 15/15 批打满 →
  抛 `LOCAL_VALIDATION_CAP` 弹回，盘面未变 → no-op 还记不下（stale 窗口
  `production_steps` 读不到 ⟹ 指纹没有可比身份）→ 连发 3 次 → 停滞保护 →
  降级又被 `_stale_for_escalation` 挡住 → `NO_FEASIBLE_ACTION`，整跑终止。

  而对「验证批次打满」控制器**有**答案 —— 分支 3a 的 `PROVISIONAL_PRODUCTION`，
  只是排在 1d-0 后面够不着。它产出的是**新布局下**的帧，一并解决了证据过期；
  普通 `RUN_PRODUCTION` 则在预热门上被弹回、一帧都产不出来。

  **已修**：1d-0 分流 —— 过期**且**正卡在验证批次上限 ⟹ 发 `PROVISIONAL_PRODUCTION`
  （语义/额度/证据口径全照 3a，判据复用**同一份** `local_validation_cap_hits()`）。

---

## **S2-B [P2] 缺窗覆盖被**纯补帧**挡住（2026-09-17 真机 `cmet_ligand1/rep2`）**


  盘面：布局 5 窗、产物 3 个 ⟹ `missing=[3,4]` **从没采过**；
  w0 已吃 3/4 块仍在继续，w2 还在 `WARMUP_VALIDATE` 一起被挡。

  缺窗分支（6b）**故意**排在因果顺序之后，理由是「上游**重锚**会作废下游 warmup
  lineage」。但 w0 拿到的是 `RUN_PRODUCTION` —— 同一份冻结 f_k、接着原段，
  **不重锚、不改布局、作废不了任何下游**。用它挡住两个从没采过的窗口**没有物理依据**。

  判据：缺窗覆盖只该被**变异动作**（重锚 / 换 Epoch / 改布局）挡住，不该被纯补帧挡住。
  完整记录 → [STAGE2_CONTROLLER_WAVE_2026-09-17.md](STAGE2_CONTROLLER_WAVE_2026-09-17.md) §3.B

---

## **S2-E [P2] 插 λ 对**末窗**是空动作，且边际停滞分支不考虑拆窗（2026-09-17 真机 `cmet_ligand2/rep1`）**


  失败窗口 == 末窗时，model B 的「溢出落末窗」落回它**自己**。执行器自己打的：

      [自治] 插 λ：窗口 (16, 21) 跨度 0.2247 → 0.2247；末窗吸收溢出。
      [自治] 插 λ：窗口 (16, 22) 跨度 0.2247 → 0.2247；末窗吸收溢出。
      ranges: (16,21) → (16,22) → (16,23)      K: 5 → 6 → 7

  这条动作的全部理由是「瓶颈是**跨度** ⟹ 缩小跨度」。末窗上它做不到，
  而 `跨度 X → X` 这行**零处消费**。

  当时**拆窗是可行的**（`feasible: split_tail_window=None`、`anchor=0.22465573`），
  拆窗会真的缩跨度。但命中的是**边际增益停滞**那条分支，它的动作写死成
  `INSERT_LAMBDA`、**从头到尾不考虑 `SPLIT_TAIL_WINDOW`**，而它排在会比较两者的
  5b **之前**。代价：插点终身预算 3 次用掉 2 次、跨度两次都没变、每次附带全量重采。
  `no-op` 台账抓不到它（插 λ 推进 `path_version`，指纹随之改变）。

  ⚠️ **非末窗的插 λ 是有效的**（同一批实测 `0.4067→0.3810→0.3532`、`0.2670→0.2473`）。
  本条只对"失败窗口恰好是末窗"成立。
  完整记录 → [STAGE2_CONTROLLER_WAVE_2026-09-17.md](STAGE2_CONTROLLER_WAVE_2026-09-17.md) §3.E

---

## **S2-I [P2] 末窗被推到 `K == hi` 就卡死：既不拆、也再插不进（2026-09-17 真机，窗口 5 索引 [16:24] K=8）**


  拆窗触发条件是 `abfe_pipeline.py:11429` 的 `if _tk > int(max_states_per_window)`
  —— **只在 K 超过 `hi` 之后才拆**，也就是布局已经非法了才补救。
  K 恰好 == `hi`（默认 4/8 下即 K=8）时：不会拆（`8 > 8` False）；
  它是全路径**最宽**的窗口且是**末窗**（最难的那个）；
  插 λ 也同时关死（`feasible()` 判 `K_after = 9 > hi`，取不到 tail anchor 即不可行）。
  ⟹ **末窗停在最差形态上，两个缩跨度动作一个都用不了。**

  **每次把它推宽的代价**（日志原文）：

      [WARN] 窗口 5 缓存能量形状 (5, 500) 与期望 (8, N) 不符，将重新采样该窗口。
      [WARN] IBS 状态与当前窗口不兼容 (cache n_states=5, current=8)，忽略旧状态

  即：① 整窗能量缓存作废 → 全量重采；② **学好的 f_k 被丢弃** → 从头重学。
  而按 `S2-E`，失败窗口恰好是末窗时插 λ **根本不缩跨度**。整条链：
  `末窗失败 → 插 λ(跨度不变) → 变宽 → 能量+f_k 全废 → 重采+重学 → 还是失败 → 再插 → 撞 hi 卡死`。
  **每轮更贵、更宽、更难，而判据认为"我在缩跨度"。**

  ⚠️ **别把触发条件简单改成 `>=`**：`_legalize_tail_window` 的死区讨论
  （`K ∈ (hi, 2·lo−1)` 既当不了单窗又拆不开）说明区间关系是刻意设计的；
  K == hi 是**合法**单窗，问题不是"它非法"而是"它是最差形态且没有出路"。
  要回答的是：**末窗涨到什么程度该主动拆**（而不是等非法），
  以及 `S2-E` 的「失败窗口是末窗时插 λ 不该被选中」。**两条一起看。**

  完整记录 → [STAGE2_CONTROLLER_WAVE_2026-09-17.md](STAGE2_CONTROLLER_WAVE_2026-09-17.md) §3.I

---

## **S2-F [P2] 布局动作没有任何事后有效性检查（2026-09-17 真机 `cyclod_ligand1_outer/rep1`）**


  补帧那条路有刹车（`marginal_gain_stalled`）。插 λ / 拆窗这条路**一个都没有**：

  | | 跨度 | ratio |
  |---|---|---|
  | 插 λ 前 | 0.2670 | 2.32 |
  | 插 λ 后 | 0.2473（−7.4%） | **1.14** ← 掉了一半 |
  | 补一块后 | 0.2473 | 1.26 |
  | 下一步 | — | 再插一次 |

  另一窗口更早两次连插 `0.4067 → 0.3810 → 0.3532`，每次只缩 6~7%。
  用 6% 的跨度去救一个离门 8 倍的比值，算术上就不成立，**没有任何判据会这么说**。

  **两笔没人算的附带代价**：① 每次插 λ 按 `path_version` 过滤的
  `min_n_eff_over_g_history` **清零** ⟹ 补帧那道唯一活着的刹车跟着失明；
  ② 每次插 λ **作废下游窗口的证据**（该 run 的 w4 已是 `stale_layout_evidence_only`）。
  ⟹ 一次"缩跨度"的真实成本 = 目标窗口重采 + 下游重采 + 增益历史清零，收益 −7% 跨度。

  判据：布局动作要有与补帧同级的事后判定（上一次插完验收量是变好还是变坏），
  变坏就不许再插。
  完整记录 → [STAGE2_CONTROLLER_WAVE_2026-09-17.md](STAGE2_CONTROLLER_WAVE_2026-09-17.md) §3.F


---

# 2026-09-16
[回到当前清单](../TODO.md)

> **这份归档的由来与前几份不同。** 前几份是「修完一条、搬走一条」；这一份是
> **一次逐条回源码的对账**：`docs/TODO.md` 里挂着一批 `- [ ]`，它们描述的缺陷
> 在工作树里**已经修好了**，只是没人回来打勾。对账方法与证据逐条写在下面。
>
> **为什么这件事本身要记一笔**：本仓已经有两份文档
> （本文的来源 `TODO.md` §1、以及 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md)）
> 同时出现过「状态栏落后于源码」。后果不是漏修，是**重复排查** —— 下一个人看到
> 一屏 `- [ ]` 会以为控制器还整条没接上，于是把已经修过的地方再查一遍。
> 这和本仓最贵的那个复发模式（「同一个不变量的 N 份实现」）是同一个形状，
> 只是发生在文档上：**「某条缺陷修没修」这个事实也有两份实现（源码、状态栏），
> 而没人定过谁是权威。** 权威是源码。
>
> ⚠️ **本文只对账「代码里那处缺陷还在不在」，不对账「修得对不对」。**
> 下面每一条都仍然是**真机零验证**（`AUDIT-S2-03` 仍开着）。

## 对账方法

1. **`[审计 #N]` 标记**：`CONTROLLER_BUDGET_AUDIT_2026-09-14.md` 定的约定是
   「源码里有 `[审计 #N]` 标记且落在真实代码改动上 ⟹ 该条已修」。
   实测 `grep -ohE '审计 #[0-9]+' abfe_preoptimizer.py abfe_pipeline.py ibs_engine.py
   runabfe.py lambda_path_versions.py abfe_config.json | sort -u`：
   **#1–#36、#38–#64 全部在场**（缺 #37、#65）。
   · **#65** 文档自己已标 `FIXED`；
   · **#37** 无标记，但缺陷本身已不在 —— `abfe_preoptimizer.py:4635` 现在是
     `_left = _pb.get("stage_remaining_steps")` + `if _left is not None and int(_left) < _cost`，
     紧邻注释逐字写着「先前 `or 0` 把 None 当成 0 ⟹ 一律拦死，等于"未知 = 耗尽"」。
   ⟹ **审计 65 条全部落地**，那份文档的 `OPEN` 状态列已于同日整列纠正。
2. **`[CTL-NN]` / `[BUD-NN]` 标记**：同法。实测在场的有
   `BUD-03 BUD-05 BUD-07 CTL-01~09 CTL-11 CTL-12 CTL-13 CTL-14`。
3. **没有标记的，直接回源码看缺陷在不在**（`BUD-01` / `BUD-02` / `BUD-04` /
   `AUDIT-S2-01` / `DECORR-01`），证据写在各条下面。

---

## A. 逐条对账结论

| 条目 | 原判 | 2026-09-16 实测 | 证据 |
|---|---|---|---|
| `DECORR-01` | 两份实现、谁是权威没定 | **已裁决 + 已接线** | `abfe_preoptimizer.py:3901/3931/4824`、`ibs_engine.py:20920/21271/21340/21386/22658` |
| `AUDIT-S2-01` | `decide()` 从不发 `SPLIT_TAIL_WINDOW` | **前提证伪**：两处发得出 | `abfe_preoptimizer.py:5814`（`_feas_split`）、`5955`（`_can_split_skew`）；且 `TODO.md` 自己的 09-14 重放表里 `cyclod_ligand2/rep1` 拿到的就是 `SPLIT_TAIL_WINDOW[5]` |
| `BUD-01` | cap 键全仓只有读侧 | **已补齐**（config + preset + CLI），默认 `null` = 上限未知 | `abfe_config.json:46/47`、`runabfe.py:333/334/375/376/384` |
| `BUD-02` | `view["path_version"]` 不存在 | **已补**：`read()` 返回值里有这个键 | `abfe_preoptimizer.py:4018` `"path_version": (path or {}).get("version")` |
| `BUD-03` | 块账换段即清零 | **已修**（审计 #40/#41） | `abfe_preoptimizer.py:4021` 注释 `[BUD-03] 硬上限用的是跨段累计的那本` |
| `BUD-04` | 补帧两道刹车都不触发 | **已修**：块账拆成四个平键并真的有值 | `abfe_preoptimizer.py:4019–4024`（`by_window` / `by_unit` / `total_by_window` / `total_by_unit`） |
| `BUD-05` | 同一个「预热余量」三套 unknown 语义 | **已修**（含原先仍 `OPEN` 的 #37） | `abfe_preoptimizer.py:6869`（`None if ... is None else int(...)`）、`6879`（`all_windows_budget_exhausted` 要求账完整）、`4635`（#37） |
| `BUD-07` | 跨段预热预算继承可能只认基准段（未验证） | **已修**，不再是「未验证」 | `ibs_engine.py:8218/8242/8283` 三处 `[BUD-07]` 标记 |
| `CTL-11` | converged 了也判不出 `DONE` | **已修**：`DONE` 提到分支 0a | `abfe_preoptimizer.py:4718` `0a) [CTL-11] stage 级判据已经通过 ⟹ DONE，不许再修` |
| `CTL-12` | 12 个动作写台账、只有 3 个读 | **已修**：`plan()` 里统一挂一道 `_NOOP_GUARDED` | `abfe_preoptimizer.py:4502–4545`，覆盖 `RUN_PRODUCTION`/`INSERT_LAMBDA`/`SPLIT_TAIL_WINDOW`/`IMMUTABLE_REWINDOW`/`RELEARN_FK_EPOCH`/`PROBE_CANDIDATE_FK`/`PROVISIONAL_PRODUCTION`（`ANALYZE` 刻意不挂，理由写在原处） |
| `CTL-13` | 补帧准入只查 `windows[0]` | **已修** | `abfe_preoptimizer.py:4438` 注释 `先前是 _tgt_w = int(windows[0])` |
| `CTL-14` | `RELEARN_FK_EPOCH` 两处 `break` 不写 `outcome` | **已修** | `abfe_pipeline.py:11992`、`12027` 两处 `[CTL-14] 终止必须同时写 outcome` |
| `CTL-15`~`19` | —— | 原文已是 `[x]`，按规则 2 搬走 | —— |
| `BM-01`/`02`/`03`/`05` | —— | 原文已是 `[x]`（2026-09-16 当天修完），按规则 2 搬走 | —— |
| `DATA-01` | —— | 原文已是 `[x]`（维护者选清空重跑），按规则 2 搬走 | —— |

> ⚠️ **仍然开着、没有搬走的**（别以为这一轮把 §1 清空了）：
> `AUDIT-S2-02`（`_decide_once` 现在 **1795 行 / 63 个 return**，比原条目记的 1319/45 **更大**）、
> `AUDIT-S2-03`（真机零验证）、`BUD-06`（已改判为待拍板，理由见 `TODO.md`）、
> `IDENT-01` / `BM-04` / `DATA-02` / `DATA-03`（四条待拍板）、`S2-E`。

---

## B. 原文留档

以下是搬出来的原始条目全文，一字未改。**读它们是为了知道当时踩了什么坑，
不是为了重新排查 —— 上表已经逐条核过缺陷不在了。**


### DECORR-01（去相关帧数两份实现）

- [ ] **DECORR-01 「去相关帧数」有两份实现，口径差 2–8 倍。**
  逐窗自检（`window_self_support_check`）与全局求解器各算一次：`cyclod_ligand2/rep2`
  实测 win3 自检 **56** 帧、`sufficient=True`，求解器报 **9 / 7** 帧并把它**跳过**；
  win0 自检 21 帧 (g=24.5)、求解器报 9。两个数**门着不同的东西**（自检喂控制器的
  `self_sufficient`，求解器的决定跳不跳），于是控制器会认为一个被跳过的窗口"帧数
  够"。这是"同一个不变量的 N 份实现"的又一例。**谁是权威没定** —— 定之前别把任何
  一侧改成向另一侧看齐。

> `CTL-01` ~ `CTL-10`（2026-09-14 第二轮静态复核的**全部**未收口项）
> ⚠️ 同日第三轮的六路并行审计把这一片重新完整扫了一遍，见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md)（#20/#22/#23/#25/#27/#30 等）。
> **2026-09-14 当天十条全部关闭**，逐条改动与判据见
> [CHANGELOG](../CHANGELOG.md) 同日那条（很长，是本组的完整记录）。
>
> 留下四条规矩，**别重新论证**：
>
> 1. **「同一个量两份实现」是本项目最贵的复发模式。** 这一轮又栽了一次
>    （`CTL-10` 的 `warmup_steps_left`），前四例是去相关帧数、f_k 字段、
>    跳窗清单、生产计账。**加任何"从盘上读一个量"的代码之前，先问它有没有
>    第二个来源、哪个是权威。**
> 2. **调度状态不能代替失败归因**（`CTL-02`）。「这个单元还没做完」和
>    「它为什么不合格」是两件事，压进一个布尔就会让偏斜类失败被当成样本量
>    不足反复加帧。这是本组里**唯一的思路错误**，其余九条都是接线错误。
> 3. **未知不是零，两个方向都不是。** 上限未知 ≠ 上限为零（会虚报耗尽）；
>    消耗未知 ≠ 消耗为零（会虚报余量）。账不完整时余量和"耗尽"都必须是 unknown。
> 4. **贵动作要在采样之前登记意图**（`CTL-04③`）。先跑后记账 ⟹ 中途退出
>    留下已烧 GPU 却未登记的孤儿，下次会重建一遍。
>
> ⚠️ **十条全部只有离线验证**（每条都做过还原变异确认会红），**真机零验证** ——
> 下一次完整 run 就是它们的第一次上机。

### AUDIT-S2-01（`SPLIT_TAIL_WINDOW` 从不发）

- [ ] **AUDIT-S2-01 `SPLIT_TAIL_WINDOW` 声明了、执行器也认，但 `decide()` 从不发。**
  它在 `ACTIONS` 里、`_run_stage2_autonomous` 有对应分支，但现在所有发它的位置都被
  `_tgt_is_tail` / `_can_split_skew` 之类的条件收窄掉了。
  **可能是对的**（`CTL-07` 正是在修"中间窗失败却拆末窗"），**也可能收得太死** ——
  末窗真的溢出到可拆区间时还有没有路径走到拆窗？
  ⚠️ **需要维护者拍板**，不要自行放宽：放宽的方向正是 `CTL-07` 刚修掉的那个错。
  判据建议：构造一个"末窗 K 落在 `[2lo−1, 2hi−1]` 且末窗自己是最差窗口"的盘面，
  看 `decide()` 给不给 `SPLIT_TAIL_WINDOW`；给不出就是收过头了。

### BUD-01 ~ BUD-05、BUD-07（预算系统）

- [ ] **BUD-01 生产预算的 cap 从来不存在 ⟹ `plan()` 的预算闸全程短路。**
  `Stage2RepairController.__init__` 读 config 键 `stage2_production_budget_steps`，
  **这个键不在 `abfe_config.json`，也不在任何一个 run 的 `run_provenance.json` 里**。
  6/6 实测 `cap=None / cap_source=unknown / cap_known=False / stage_remaining_steps=None`。
  连带三条：① `plan()` 里那道统一生产预算闸写成 `... and _pb.get("cap_known")` ⟹ **永远不进**；
  ② `GLOBAL_BUDGET_EXHAUSTED` 是 `TERMINAL_EXITS` 三大真终态之一，而**唯一发出点**就是那道闸
  ⟹ **真机上永不可能发出**，设计 §2 说的三个真终态实际只有两个；
  ③ `IMMUTABLE_REWINDOW` 的预留门 `(not cap_known) or 剩余 >= 预留` ⟹ 恒 True，
  「留不出来就别开系综」从未生效。
  ⚠️ **别只补 config**：先看 `BUD-06`，cap 一旦生效第一次 resume 就会误判耗尽。
  ✅ **已被审计 #31 覆盖并修复**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md) **#31**）：`stage2_production_budget_steps` / `stage2_max_production_blocks_per_window` 已在 **config + preset + CLI 三处**补齐，默认值严格等于原硬编码兜底 ⟹ 行为逐位不变；前者默认 **`null` = 上限未知，不是 0**（控制器写死「未知时不拦」），正是为了不触发 `BUD-06` 那个误判。`max_path_insertions` **刻意只补 CLI**（它进 stage 协议指纹，给默认值会让所有既有 run 的 Stage 1/2 结果缓存全部失配重跑）。**`BUD-06` 本身未被审计覆盖，仍然开着。**

- [ ] **BUD-02 `view["path_version"]` 这个键不存在 —— 一处打穿四个机制。**
  路径版本只在 `view["path"]["version"]`（实测 cyclod_ligand2/rep1 = 4）；顶层**没有**
  `path_version`，而代码里到处 `view.get("path_version")`，**恒为 `None`**：
  · `_run_stage2_autonomous` 写 history 的 `"path_version"` —— 实测 6 个 run 每一轮都是 `null`；
  · `_production_blocks_ledger` 拿它过滤 `it["path_version"] != path_version` ⟹ 真实版本 ≠ `None`
    ⟹ **每条 iteration 都被跳过，补帧块账恒空**；
  · `action_noop_fingerprint(w, view.get("path_version"))`（控制器与执行器两侧都传它）⟹
    指纹缺路径版本这一维 ⟹ **插 λ / 拆窗换了布局之后，旧的 no-op 记录不失效**；
  · `decide()` 里 `relearn_epoch_used(ckpt, int(view.get("path_version") or 0), w)` 读键用 `0`，
    而执行器 `mark_relearn_epoch_consumed` 写的是 `lambda_path_versions.load_current()` 的真实
    版本号 ⟹ **读写不同键，「一个窗口只给一次 fresh Epoch」在控制器侧永远判成「还没用过」**。

- [ ] **BUD-03 块账的第二重失效：换段即清零。**
  即使 `BUD-02` 修好，`_production_blocks_ledger` 还要求
  `snapshot[i]["segment"] == 当前视图里该窗口的 segment`，而**换段正是循环自己的动作**。
  实测 cyclod_ligand2/rep1 win5：history snapshot 里是 `vanishing`，当前视图里是 `vanishing_7`
  ⟹ 全部 `continue`。**窗口每换一次段，它烧过的帧账清零、重新发 4 块。**
  ✅ **已被审计 #40/#41 覆盖并修复**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md) **#40** / **#41**）：#41 修的正是`same_segment_only` 下 `seg_of.get(i)` 对不在当前视图里的窗口返 `None`、与字符串恒不等 ⟹ 刹车静默失效；#40 修的是块账双向错（每轮给视图里**每个**窗口白记一行 / `PROBE_REANCHOR_EPOCH` 确实花一块却被过滤掉）。

- [ ] **BUD-04 `BUD-02`+`BUD-03` 的合计后果：补帧完全没有刹车。**
  `_frames_admission` 的两道判据 —— ① 每窗 `max_production_blocks_per_window`（缺省 4，
  config 里也没有 `stage2_max_production_blocks_per_window`）；② 「上一块必须有实质增益
  （求解器侧去相关帧数）」（需 `len(rows) >= 2`）—— **都读 `production_blocks_by_window`，
  而它 6/6 run 全是 `{}`** ⟹ **两道判据一次都没触发过**。直接对应单窗烧到 1,500,000 步。
  ⚠️ 「补帧没有停止条件」这条被记成已修，实际**生产侧从头到尾没有任何刹车**。
  📌 **口径更新（2026-09-14）**：块账已**不再是单层**，现在是四个平键 ——
  `production_blocks_by_window` / `production_blocks_by_unit` /
  `production_blocks_total_by_window` / `production_blocks_total_by_unit`
  （**同段账 vs 跨段累计** × **物理窗口 vs 采样单元**）。读它的地方要先确认自己要的是哪一格。
  ✅ **已被审计 #8/#30/#31/#40/#41 覆盖并修复**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md) **#8**「子窗两道刹车同时失效」、
  **#30**「块数满额是路由信号不是终态」、**#31** 硬上限进 config/CLI、**#40/#41** 块账双向错）。

- [ ] **BUD-05 同一个「预热余量」有三套 unknown 语义。**
  `warmup_steps_left` 读不到（`None`）时：`per_window_budget_remaining` /
  `all_windows_budget_exhausted` 用 `int(... or 0)` ⟹ **0 = 耗尽**；
  `_epoch_validation_unaffordable` ⟹ **不可行**（fail-closed）；
  `decide()` 分支 1e 的 `left is not None and int(left) <= 0` ⟹ **有钱**。
  实测 cyclod_ligand1/rep3 win4 的 ledger 确实读不到，在第一处被显示成 `0`。
  这是本文上面那条规矩「未知不是零，两个方向都不是」的再次违反。
  ✅ **已被审计根因 ② 覆盖并基本修复**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md) **#15/#36/#38** 已修；**#37 仍 `OPEN`** ——「cap 已知但用量未知」那一路还在 `or 0`）。

- [ ] **BUD-07（未验证，查到一半被叫停）跨段预热预算继承可能只认基准段。**
  `ibs_engine.inherit_warmup_ledger_across_segments` 靠 `base_checkpoint_dir_for()` 找
  「上一层」，而那个函数的锚点是 `path_current.json` —— **只有基准目录有它**。
  所以 `checkpoints/segment_3` 找回的是 `checkpoints/`，**不是 `segment_2`**：
  第 3 个 Epoch 继承基准段的消耗、丢掉段 2 烧掉的量。
  形状与已归档的「多 Epoch 链永远从基准段重学 f_k」一致（那条当时的结论是「第一次换
  Epoch 完全正确，连换两次才踩到」）。⚠️ **只读了函数、没有验证**，需要一个有
  `segment_3` 及以上的 run 对账。

### CTL-11 ~ CTL-19（控制器第三轮复核）

- [ ] **CTL-11 stage 级判据已经通过，`decide()` 仍然不判 `DONE`。**
  实测 cyclod_ligand2/rep2：stage `converged=True`、`missing_windows=[]`、`skipped_windows=[]`，
  当前代码给出 **`PROBE_REANCHOR_EPOCH[0]`**。
  根因：`DONE` 在分支 7，排在「最早未解决窗口」路由**之后**；而 `earliest` 来自逐窗**自检**
  `self_verdict`（该 run win0=4.37、win3=4.81，门 10）。两者是同一个量的两份实现 ——
  stage 级 `converged` 由 `solve_stage_integrated` 在**合并后的全部段**的帧上算（五条合取），
  是权威；自检 `min N_eff/g` 只看该窗口**一个段**的帧，对多段窗口系统性偏悲观。
  ⟹ 任一窗口自检 < 10 就永远轮不到分支 7，**一个已经跑完的 stage，每次 resume 都会
  重开一个 Epoch 烧 GPU**。同形状的第二处：`_evidence_status()` 里「任一窗口 self_verdict
  不合格 ⟹ INSUFFICIENT_DATA」排在 `stage_converged is True ⟹ CONVERGED` **之前**。
  ✅ **部分被审计覆盖**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md)）：**#27** 已修（`_immutable_rewindow_step` 绕过统一入口写盘、不盖 `path_version` ⟹ `stage_result_path_version_verified` 恒 False ⟹ 分支 0a 的 `DONE` **结构上不可达**，这是 `DONE` 判不出来的第二条封锁）；**#18(a) 仍 `OPEN`** —— `stale` 表永不清除 ⟹ `stale_layout_evidence` 恒非空 ⟹ `DONE` / `CONVERGED` 同样被封死。**三条封锁要一起解开，只解一条仍然判不出 `DONE`。**

- [ ] **CTL-12 no-op 台账写 12 个动作，只有 3 个动作会去读。**
  执行器 `_record_noop_action` 对**任何**动作都记账（通用盘面指纹比对），但 `decide()` 里
  `_is_noop()` 只在四个位置被调用、覆盖三个动作：`CONTINUE_WARMUP`、`RECALIBRATE_FK`（两处）、
  `PROBE_REANCHOR_EPOCH`。**`SPLIT_TAIL_WINDOW` / `INSERT_LAMBDA` / `RUN_PRODUCTION` /
  `IMMUTABLE_REWINDOW` / `ANALYZE` 全都不读。**
  后果（旧代码日志实测，形状在当前代码里未变）：cyclod_ligand1/rep2 与 cyclod_ligand2/rep1
  都死在同一条路上 —— `SPLIT_TAIL_WINDOW` 连发 4 次、执行器每次都打了
  「记为当前盘面上的 no-op」、控制器每次都看不见，最后由停滞保护给出 `NO_FEASIBLE_ACTION`。
  ✅ **同形状的子窗半边已被审计 #9 覆盖并修复**（见 [CONTROLLER_BUDGET_AUDIT_2026-09-14.md](CONTROLLER_BUDGET_AUDIT_2026-09-14.md) **#9**：执行器写 `f"{{action}}:unit:{{uid}}"` 而 `_is_noop` 只查 `f"{{action}}:{{idx}}"` ⟹ 全仓无人读）。**「12 个动作写、只有 3 个动作读」这一半未被审计覆盖，仍然开着。**

- [ ] **CTL-13 补帧准入只查 `windows[0]`。**
  `plan()` 里 `_tgt_w = int(windows[0])`，只拿这一个窗口去问 `_frames_admission`；
  而分支 9c 在「端点 σ 归因不到具体窗口」时发的是 `windows=sorted(全部窗口)`（注释明写是
  有意的）。⟹ win0 块数一满就把**整批窗口**的补帧一起毙掉；反过来 win0 有额度时，
  其余窗口即使已超额也照批。

- [ ] **CTL-14 `RELEARN_FK_EPOCH` 的两处 `break` 没写 `outcome`。**
  `_run_stage2_autonomous` 里该分支的两个提前 `break`（「替代候选已用过一次」「剩余 warmup
  预算 < 新 Epoch 所需」）只写了 `history[-1]["exit"]`，**没有调 `_finish()`** ⟹ `outcome`
  停在 `{"status": "RUNNING", "exit": null, "iterations_used": 0}`。实测 brd4_ligand2/rep1
  的 history 现在就是这个状态 —— 正是 2026-09-14 加 `outcome` 要解决的那个问题。

- [x] ~~**CTL-15 三个 run 的布局已进入无恢复状态。**~~
  🔴 **本条原始判断是错的，已作废（2026-09-14 同日更正）。**
  原文说「brd4_ligand2/rep1 末窗 K=10 > 可拆上限 `2*hi−1 = 9`」—— 那是**拿
  cyclod 的 `lo/hi=4/5` 去套 brd4**。brd4 与 cyclod_ligand1 的
  `stage2_window_max_states` 是 **8**，可拆区间是 `[7, 15]`，K=10/11/12 **全都在
  区间内、可以拆**。实测三条全部能合法化：

  ```
  brd4_ligand2/rep1    (16,26) → (16,21)+(20,26)
  cyclod_ligand1/rep2  (15,27) → (15,21)+(20,27)
  cyclod_ligand1/rep3  (16,27) → (16,22)+(21,27)
  ```

  **留下的规矩**：`lo/hi` 是**逐 run** 从 `run_provenance.json` 读的
  （`Stage2RepairController.__init__` 就是这么写的，理由也写在那里），
  跨 run 引用可拆区间前必须先看那个 run 自己的 `lo/hi`，别拿手边那个体系的数去套。

  插 λ 超预算这件事本身仍然为真（版本链上 5/6 次 vs 预算 3），成因是旧代码
  `_read_path` 读错 `kind` 键、已于 2026-09-14 修好；但它**没有**造成不可恢复的布局。

- [x] ~~**CTL-16 `first_untrusted_window()` 把「证据被布局变更作废」当成「还不知道」。**~~
  **2026-09-14 已修。** 判据只有 `v is not None and v != ANALYSIS_ELIGIBLE`：
  `v is not None` 是对的（没跑过自检 ≠ 有问题，设计 §4 三态），但把
  `stale_layout_evidence_only` 的窗口一起漏掉了 —— 而设计 §4 同一节明写那是
  **确定的未解决**，`_window_state()` 也正是这么判的。**同一个不变量两份实现。**
  真机 cyclod_ligand1/rep3：win0–3 全 ANALYSIS_ELIGIBLE、win4 证据被作废
  ⟹ 返回 None ⟹ 取不到 tail anchor ⟹ 末窗 K=11 > hi=8 时
  `_legalize_tail_window` 在**启动布局校验**处 fail-closed 抛错，
  **`decide()` 一次都轮不到、整个 run 起不来**。判据已与 `_window_state()` 对齐。

- [x] ~~**CTL-17 `0b`（身份不一致）物理位置在分支 5 之后。**~~
  **2026-09-14 已修。** 编号写 `0b`、意图是"最早"，位置却在
  `5) 生产帧没攒够` 后面。而 `IDENTITY_MISMATCH` 的窗口 `production_steps`
  通常正好没到目标，分支 5 又只打 `earliest`（正是这个窗口）⟹ **先发
  `RUN_PRODUCTION`、对着另一个系综的产物补帧**，`HALT_INVALID_INPUT` 永远走不到。
  已移到全部路由之前，判据一字未改。
  ⚠️ `_window_state()` 里的 `IDENTITY_MISMATCH → PROBLEM` 是**另一条**修复
  （它让该窗口成为 `earliest`），必要但不充分，两条互补、别当重复删掉。

- [x] ~~**CTL-18 `_no_gain` 分支不查可行性就发 `INSERT_LAMBDA`。**~~
  **2026-09-14 已修。** 全代码唯一一处发布局动作却不问
  `feas.get("insert_lambda") is None` 的分支；插点预算耗尽 / 末窗顶满时它发出的是
  **构造上不可能成功**的动作。归因（加帧已被证伪）是对的，所以不可行时只改动作、
  不改归因：如实 `NO_ACTION` + `NO_FEASIBLE_ACTION`。

- [x] ~~**CTL-19 9b 的 `SPLIT_TAIL_WINDOW` 传 `windows=[]`，三重失效。**~~
  **2026-09-14 已修。** ① 执行器 `_record_noop_action` 是 `for w in (windows or [])`
  ⟹ 空列表**一条账都不记**；② `plan()` 的 no-op 闸要求 windows 非空 ⟹ 拦不到；
  ③ 主循环停滞保护的降级条件是 `act != PROBE... and wins` ⟹ wins 空就跳过降级、
  直接 `NO_FEASIBLE_ACTION`。真机 cyclod_ligand2/rep1 与 cyclod_ligand1/rep2
  都是这么死的。已改为点名末窗号。

### BM-01 / BM-02 / BM-03 / BM-05（2026-09-16 benchmark 报错分类）

- [x] ~~**BM-01 局部动作的执行器去载全路径。**~~ **2026-09-16 已修。**
  位置：`abfe_pipeline.py:13360`（`_recalibrate_f_k_and_resample_segment`）。
  `only_windows` 的过滤原来在 loader **之后**的循环里，而 loader 对每个载入的窗口
  做 fail-closed 布局校验 ⟹ 一个只针对窗口 3 的动作崩在窗口 4 上：
  `动作=PROBE_CANDIDATE_FK 窗口=[3]` → `ValueError('窗口 4 状态数与 window_ranges 不符')`。
  真机 7 个 run 同签名（brd4_ligand1/rep1、jnk1_ligand1/rep2、jnk1_ligand2/rep1-3、
  cyclod_ligand3/rep2、p38_ligand1/rep3）。改成在**载入时**把 `only_windows` 之外的
  窗口一并排除；`records` 内容与语义逐字不变（重锚节奏逻辑读它）。
  判据：`test_layout_change_does_not_crash_the_loop_2026_09_16.py::test_recalibrate_only_loads_the_windows_it_acts_on`
  （`only_windows=[1]` ⟹ 排除集合 `{0,2,3}`）+ `…::test_no_only_windows_still_loads_the_whole_path`。

- [x] ~~**BM-02 停滞保护的降级绕过 `decide()` 的全部可行性守卫。**~~ **2026-09-16 已修。**
  位置：`abfe_pipeline.py:11465`（`_run_stage2_autonomous` 的停滞保护）。
  它直接改写 `act = "PROBE_REANCHOR_EPOCH"`，而 `decide()` 里的 1d-0（布局过期）、
  1e（预热进不去）、`TERMINAL` 分流全在上游，一条也没走。真机 jnk1_ligand2/rep1：
  `RUN_PRODUCTION[3]` 被 `LOCAL_VALIDATION_CAP` 连弹 3 次 ⟹ 降级成
  `PROBE_REANCHOR_EPOCH[3]` ⟹ win3 的帧正是插 λ 之后的过期布局 ⟹ 炸穿主循环。
  已加闸：目标窗口在过期集合里 ⟹ 不降级，如实 `NO_FEASIBLE_ACTION`，
  并记 `history[-1]["escalation_blocked_by"]`。
  ⚠️ 顺带把「产物是不是过期布局的」收成**一份实现**
  `Stage2RepairController.stale_layout_windows()`（`abfe_preoptimizer.py:6158`），
  `decide()` 的 1d-0 改成调它 —— 这个判据有两个消费者，而 `ibs_engine` loader 对
  同一件事是 fail-closed 抛 `ValueError`，两边一分岔就是崩溃。
  判据：`…::test_the_escalation_gate_actually_sits_before_the_downgrade`（AST 检查闸的
  位置在降级赋值之前，沿用本仓对这个大循环的既有测法）+ `…::test_stale_layout_windows_is_one_implementation`。

- [x] ~~**BM-03 插 λ 能造出拆不开的末窗，且非法布局在 fail-closed 之前已落盘。**~~
  **2026-09-16 已修（表层；死局本身见 `BM-04`）。**
  真机 `cyclod_ligand3/rep1` 的 `path_versions` 末窗 K：v1=6 → v2=7 → v3=8 → **v4=9**，
  而 `stage2_window_max_states=8`。`append_version` 在 anchor 检查**之前**
  （`abfe_pipeline.py` 插 λ 分支 vs `_legalize_tail_window`），所以 v4 那个非法布局
  **已经写进版本链** —— 后果不是"这一轮崩掉"，是 resume 读到它照样合法化不了，
  这一跑再也走不出来。
  两道都加了，缺一不可：
  · 可行性侧 `abfe_preoptimizer.py:4179`（`feasible()`）——
    插完末窗越过 `hi` 且取不到 tail anchor ⟹ `insert_lambda` 判不可行（回答"该不该发"）；
  · 执行器侧 `abfe_pipeline.py:11802` —— 可落性在 `append_version` **之前**判（回答"发了能不能落"）。
  ⚠️ 闸只在 `末窗K+n_insert > hi` **且** 无 anchor 时触发：cyclod_ligand3/rep1 的
  前两次插点（6→7、7→8）照样放行，挡住的正是把布局搞成非法的第三次。
  判据：`…::test_insert_is_infeasible_when_it_would_strand_the_tail` +
  `…::test_insert_stays_feasible_when_the_tail_has_room`（后者钉的是**不许误伤**）。
  ⚠️ 顺带记一条口径：末窗吸收溢出自己的 fail-closed 门槛是 `K > 2·hi−1`（=15），
  而合法化门槛是 `K > hi`（=8）—— **8..15 是一段"吸收得进、拆不出来"的死区**，
  `BM-03` 只是不让人再走进去，死区是 `BM-04` 的事。

- [x] ~~**BM-05 两处会把排查引向错误对象的记账 bug。**~~ **2026-09-16 已修。**
  ① `ibs_engine.py:22526`（`solve_stage_integrated`）：`w_idx` 是
  `enumerate(valid_windows)` 的**列表位置**，而 `valid_windows` 只含被载入的窗口。
  部分段分析（`[部分窗口段] 本段只采了窗口 [3]`）里它是 0，于是 12 条
  `窗口 {w_idx}` 的日志把物理 win3 报成"窗口 0" —— 真机 brd4_ligand1/rep1 就是这么
  把排查引到错误窗口上的（第一轮归因错 4 条里有 1 条源于此）。结构化字段一直是对的
  （都走 `source_window_idx`）。修法是**在源头改名**：列表位置叫 `_list_pos`，
  `w_idx` 从此就是物理窗口号 —— 一次改对 12 条消息，而不是逐条改 f-string。
  ② `abfe_preoptimizer.py:2706`（`Stage2RepairController.__init__`）：`_explicit` 早就
  滤了 `None`，`_disk`（`run_provenance.json` 的 config，是 argparse 完整命名空间落盘的）
  **没滤**。实测 13 个 run 的 provenance 每份都带 17~18 个 `None` 键。今天不炸只是因为
  那几个键恰好没走 `int()`；09-15 `cyclod_ligand1/rep1` 的
  `TypeError: int() argument … not 'NoneType'` 就是同一个形状。
  **「未知」只有一种表达：缺键。**
  判据：`…::test_null_in_run_provenance_is_treated_as_absent`。

> （`BM-A`「这些不是 bug，别再查」是 **live 清单**，不是已关闭条目 ——
> 它留在 [../TODO.md](../TODO.md) §1，没有搬到这里。）

### DATA-01（`cyclod_ligand2/rep3` win4 manifest 与 f_k 对不上）

- [x] ~~**DATA-01 `cyclod_ligand2/rep3` win4 的 production manifest 与 f_k 全都对不上。**~~
  **2026-09-14 已消解**：维护者把三个 run 的整个 stage-2 产物（`vanishing/` +
  全部段目录 + `ibs_state_*` + `production_window/`）清空重跑，`checkpoints/` 只留
  Stage 0/1。那份 stale manifest 随之不存在了。**判据本身没有变**：分析 loader
  仍然对「manifest 与两份 f_k 都对不上」fail-closed，下次再出现照样拦。
  下面保留原始诊断，供再次出现时对照 ——

- [ ] ~~**DATA-01（原始诊断，留档）**~~
  实测哈希：`manifest == live f_k` **False**、`manifest == production_entry_f_k` **False**
  —— 那份 production checkpoint 是在**第三个** f_k 下写的，比 state 文件还老。
  分析 loader 因此 fail-closed（`窗口 4 冻结 f_k 与 production manifest 不一致`）。
  **这不是 2026-09-14 改出来的**：该窗口 `live == entry`，走的是改动前同一条路。
  对照：rep2 win4 与 brd4 win4 都是 `manifest == entry True`，那两个已由当天的
  `_resolve_analysis_f_k` 修复。
  ~~**待定**：这个窗口是重采、还是把那份 stale checkpoint 作废，需要维护者拍板。~~
  → 维护者选了**清空重跑**。

---

## C. 同日第二批：4 条「与源码不符的陈述」+ 1 条子项关闭

对账做完之后又扫了一遍**陈述**（不只是 checkbox），抓到 4 条。
**它们不是待办，是 `TODO.md` 正文里写着的、与源码不符的话** —— 危害比挂着的 `- [ ]` 大，
因为没有 checkbox 提示读者去核。

| 原文 | 实测 | 处置 |
|---|---|---|
| 「不许回退的约定」：**只有三个真终态** `DONE`/`GLOBAL_BUDGET_EXHAUSTED`/`NO_FEASIBLE_ACTION` | `abfe_preoptimizer.py:2553 TERMINAL_EXITS` 实际 **9 个**（多出 `DONE_UNTRUSTED`、`HALT_INVALID_INPUT`、`HALT_EVIDENCE_CONTRADICTS_DONE`、`HALT_LAMBDA_BUDGET_INSUFFICIENT`、`HALT_FRAMES_ADMISSION_CAP`、`ANALYSIS_COMPLETE_PRECISION_UNMEASURED`） | 改成**指向 `TERMINAL_EXITS`**，不在文档里维护第二份清单 |
| `AUDIT-S2-03`：**控制器真机零验证** | 39 rep 真上过 GPU，9 个出结果 | 改成「**有真机，但没有一次以 `DONE` 收口**」+ 判据 |
| `REL-03`：**到目前为止全部是 CPU / 静态验证，零 GPU 复验** | 同上 | 改成逐子项标状态；③ 关闭 |
| `REL-04`：**52 处改动无一上过 GPU** | 它们在主线里，39 rep 全执行过 | 改成「**跑过，但没有一处被单独复验**」 |

> 🔑 **最危险的是第一条，理由值得单独记：** 它属于「不许回退的约定」——
> 那一节是拿来**否决别人改动**的。一条过期的约定不会像过期的 `- [ ]` 那样只是浪费时间，
> 它会**主动打回正确的改动**：照旧清单去审，6 个合法终态会被当成违规。
> ⟹ **约定类文字比待办类文字更需要对账**，且**能指向源码就别抄一份到文档里**。

### 已关闭的子项：`REL-03` ③ 两条腿同进程的 `pipeline.log` 分离

原条目要求「已用最小复现验证，**未在真实两腿运行上确认**」。
2026-09-16 在 benchmark 的真实两腿运行上确认：`p38_ligand1/rep3` 的
`pipeline.log`（134 KB）与 `solvent_leg/pipeline.log`（51 KB）独立存在；
复合物腿日志里 10 处命中「溶剂腿」全部是**一条 WARN 文案自带的词**
（`[独立端点段] 未启用 … 该路径在溶剂腿上被论证为…`），**不是串日志**。
⟹ ③ 关闭。**①②（residual 臂门读数、`frozen_ll_pairs` 断言）仍开着**，留在 `TODO.md`。

### 顺带修掉的两处排版

- `TODO.md` 归档表第 5 行掉了 `> ` 前缀 ⟹ 表被劈成两半（**旧 bug**，非本次引入）；
- 优先级表第 1/2 行 530 / 362 字符（本次对账时自己写长的）⟹ 压到 ~175。

---

# D. 维护者拍板：`BM-04` / `DATA-02` / `DATA-03` 三条同日关闭（2026-09-16）

> 这三条覆盖 benchmark 里 **7 个走不动的 run**。三条是同一件事的三面，所以一起定。

## 🔑 本次确立的决策原则（比三条结论本身更重要）

> **「算对之前，沉没成本一律不计。」** —— 维护者原话：
> 「再计算正确之前，所有沉没成本都是可以无视的」。
>
> 代理在本轮**两次**把已有缓存的价值算高了，两次都被纠正：
> 一次是建议「别为 2 个 run 作废 39 个 run 的窗口缓存」，
> 一次是把 `PRESET_CONFIGS` 改动的半径估成「39 个 run 从头跑」。
> 实际上：那批缓存里**只有 2 个 run 是修复后代码产出的**（`p38_ligand1/rep2,rep3`），
> 其余 7 个完成的 run 全是旧码、本来就不可引用；而 30 个未完成的 run 无论如何要重做。
> **一个还没被证明算得对的结果，它的缓存没有保护价值。**
> ⟹ 今后同类权衡**先问「这条路径算得对吗」，再问「重算要多少 GPU」**，顺序不能反。

## 拍板前先堵上的一个缺口（否则决定 1 会落空）

`stage2_first_window_max_states` **此前从未在任何一次真实运行里生效过**。四条路全断：

| 来源 | 有没有 |
|---|---|
| `PRESET_CONFIGS` 三个预设（`test`/`production`/`high_accuracy`，各只有 6 个键） | ❌ |
| 14/14 benchmark config（`abfe-benchmark/openmm_IBS/configs/*.json`） | ❌ |
| CLI 开关 | ❌ **不存在** |
| `abfe_config.json`（= 4） | ✅ 有，**但它不会被自动加载**（`--config` 的 argparse 默认是 `None`），且 `runabfe.py:3302` 的合并基底是 `PRESET_CONFIGS[preset]`、**不是**这份文件 |

缺席时 `first_cap is None` ⟹ 键不发出 ⟹ `abfe_preoptimizer` 默认 `first_window_max_states=None` ⟹ **无 cap**，win0 照样能拿 8 个态。

> 📌 **留下的规矩**：`abfe_config.json` **不是"仓库默认配置"，它只是一份样例**。
> 它的 `_comment` 写得像全局默认（「全局 `stage2_window_max_states` 保持 8 不动」），
> 极易被读成已经生效。**判一个键在某次运行里是什么值，只能看
> 该 run 的 `run_provenance.json`，不能看 `abfe_config.json`。**

**处置（已落盘，代码改动）**：`stage2_first_window_max_states: 4` 加进 `runabfe.py::PRESET_CONFIGS`
的**三个**预设。选预设而不是「只改这 2 份 benchmark config」，理由是后者只解开 2 个 run、
其余 12 个体系下次照样踩同一个死局。
**半径**（实测 `_split_preopt_protocol_key` 的两层拆分，`abfe_pipeline.py:3090/3155`）：
本键属**第 2 层**派生路径键 ⟹ 第 1 层（pilot 采样语义）不变 ⟹
**Stage 0/1 与 pilot/preopt 全部保留**，布局离线重算，**只有 Stage-2 窗口轨迹全部重采**。

---

## 决定 1 —— `BM-04`：**不开放 `anchor = window 0`**

**选项：三条出路里都不选，改为「用正确的首窗 cap 重建」。**
适用 `cmet_ligand1/rep1`、`p38_ligand2/rep1`：

1. 保留 pilot/preopt；
2. 给 Stage-2 建**新的** checkpoint/path 命名空间；
3. 按 `stage2_first_window_max_states=4` 重新生成路径并重跑 Stage-2；
4. **不**增加 `stage2_final_n_states`。

**理由（已逐条回源码核实）**：window 0 没有前置共享态，把它塞进 tail-repartition
**实际等于新增「整条路径重分」这个操作**，不是放宽一个判断 ——
`abfe_preoptimizer.py:6174 tail_repartition_anchor` 的 docstring 写明
「冻结的是 anchor **之前**的窗口」，而 win0 之前没有窗口可冻；后续 repartition 契约
也不支持这种语义。原条目出路 1（允许 anchor=win0）因此被否。

**首窗 cap=4 是正确的预防性修复，但它不会改写旧版本链** —— 所以必须重建，不能原地续跑。
`abfe_config.json` 里该键的 `_comment` 独立给出了更强的论证：
**「布局类修复动作一个都够不着 win0」**（`SPLIT_TAIL_WINDOW` 要求 `window_idx>0`；
`INSERT_LAMBDA` 会重排全局边界）⟹ **win0 的跨度只能在分窗这一刻决定，事后没有任何修复路径。**
实测 cap=4 得 `[4,6,4,4,4,5]`：maxK 从 8 降到 6，且那个 6 落在 win1（split-tail 够得着），
代价 +1 个系综。⚠️ 峰值 ∫g 在 cap=8/6/4 三档都是 134.6 —— **封 win0 在 ∫g 账本上一分钱不赚，
它买的是 K 不是 ∫g**，别拿峰值去证明这个开关有效。

## 决定 2 —— `DATA-02`：**选 ②，清理 Stage-2 后重建**

适用 `cyclod_ligand3/rep1`、`rep3`：**不回退 v3、不手工伪造 v5。**
旧的 `path_current.json` / `path_versions/` / 窗口产物**作为整体归档**，随后用全新的
Stage-2 命名空间重建。

**理由（已核实）**：`lambda_path_versions.py:184 _publish()` 注释逐字写着
**「只前进不后退」**，回退直接 `raise ValueError(f"拒绝把当前路径从 v{...} 倒退到 v{...}")`
⟹ 出路 ① 在代码层面就走不通。而「只删窗口文件、保留 v4 指针」同样不行：
`load_current` 读的是指针，启动布局校验仍会读到那个 K=9 的非法布局。
⚠️ 重建时**必须同样带上 cap=4** —— 这两个 run 也是 `first_untrusted_window=0` 的死局，
不带 cap 重建就是重演一遍。

## 决定 3 —— `DATA-03`：**`max_path_insertions` 3 → 4，`stage2_final_n_states` 保持 21，原地 resume**

适用 `cmet_ligand1/rep2`、`cmet_ligand2/rep1`、`p38_ligand2/rep3` ——
这三个是「**结构上仍可插点，只是预算耗尽**」，与上面四个的死局不是一回事。

**理由（已核实）**：`ibs_engine.py:503` 把 `max_path_insertions` 从
**窗口采样身份**里 `pop` 掉了，且紧邻注释写明 stage **结果**缓存不受影响
（仍要完整 protocol key + `lambda_path_fingerprint` 全路径门）
⟹ stage protocol key 会变、要重新分析，但 **λ 未变的已有窗口可以复用，不需要全部重采**。

**⚠️ 暂时不要改成 23**：它会重建初始 λ 网格，导致多数甚至全部 Stage-2 窗口失去复用条件。
**若第 4 次插点后仍报预算不足，就停在 4，不许继续加到 5** —— 那时才说明初始网格确实偏小，
应切换到 23 做一次干净重跑。

**📌 成本里补一项（原估算漏了）**：这三个 run 的 config 里 `max_path_insertions` 本来就是
显式的，所以改值只影响它们；但它进 stage protocol key ⟹ **Stage 1 结果要重新求解，
实测 ~28 分钟/指纹 × 3 个 run**（只重解、不重采样）。不改变结论，但要计进预算。

## 总体策略与 GPU 代价

**4 个旧/非法 Stage-2 干净重建（`cmet_ligand1/rep1`、`p38_ligand2/rep1`、
`cyclod_ligand3/rep1`、`rep3`），3 个预算不足 run 用 21/4 有界续跑。**

- 决定 1 + 2：4 个 run 重跑 Stage-2，**pilot/preopt 与 Stage 0/1 保留**；
- 决定 3：每个约新增 1 个受影响窗口的 production（个别 tail 情况 2 个）+ Stage 1 重解 ~28 min；
  明显低于改成 23 后整段重跑；
- 预设改动的连带：**全部现存 run 的 Stage-2 窗口缓存失配重采**（Stage 0/1 与 pilot 保留）。
  按上面那条决策原则，这个代价**已被接受**。

## 验收判据

- 决定 1/2：四个 run 的新命名空间里，`path_versions/v1.json` 的 `window_ranges` 首窗 `K ≤ 4`，
  且启动布局校验能过、`decide()` 拿得到可执行动作；
- 决定 3：三个 run 能发出第 4 次 `INSERT_LAMBDA` 并继续，不再以
  `HALT_LAMBDA_BUDGET_INSUFFICIENT` 退出；
- 三者共同：重跑后仍失败的才是新信息，按 `BM-B` 的三条取证口径重新分类。

---

## 原文留档（`BM-04`）

- [ ] **🟠 BM-04 卡住的窗口是 window 0 时，「缩跨度」这一族动作在构造上全部不可行。** **需要维护者拍板，代理不要自行改。**
  位置：`abfe_preoptimizer.py::Stage2RepairController.tail_repartition_anchor`
  —— `if idx is None or idx <= 0: return None`。
  链条：拆末窗要 tail anchor → anchor 取自 `first_untrusted_window` 的**首态**
  → window 0 不可信时 `idx == 0` ⟹ anchor 恒 None ⟹ **拆窗永远不可行**
  ⟹ 只剩插 λ ⟹ 插到末窗满（K=hi）就没动作了 ⟹ `NO_FEASIBLE_ACTION`。
  `idx <= 0` 本身不是笔误：docstring 写明 anchor 的语义是「冻结 anchor 之前的窗口」，
  而 window 0 之前没有窗口可冻结。**所以这是设计边界，不是接线错误。**
  真机命中：`cyclod_ligand3/rep1,rep3`（崩）+ `cmet_ligand1/rep1`、`jnk1_ligand1/rep3`、
  `p38_ligand2/rep1`（插 λ 预算先用完所以没崩，终态理由逐字是
  「拆窗：取不到 tail anchor（没有 window_idx > 0 的不可信窗口）」+「插 λ 预算已用尽」）。
  ⚠️ `BM-01`~`03` **不会**让这些 run 跑完 —— 它们只把"崩溃 / 留下非法布局"换成
  "如实停下并说清原因"。**别把 `BM-04` 当成已经解决了。**
  三条出路，代价差很远，**没有默认答案**：
  1. 允许 anchor = window 0 的首态，即**整条路径重分窗** —— 语义上是"全部作废重采"，
     GPU 代价最大；但 window 0 恰恰是最常卡住的窗口（它是耦合端）；
  2. 加大 `stage2_final_n_states` 给末窗留余量 —— 只把死区推后，不消除；
  3. 接受现状，window 0 卡住就人工介入 —— 现在至少会明确说"卡在 window 0 且拆不开"。
  判据：拍板后在本条下写明选了哪条 + 理由，并归档；若选 1，需要一个
  "window 0 不可信 + 末窗顶满"的盘面能走到实际重分窗且不作废已合格窗口的测试。

## 原文留档（`DATA-02` / `DATA-03`）

- [ ] **🟠 DATA-02 `cyclod_ligand3/rep1` 与 `rep3` 的非法布局已经落盘，这两个 run 起不来。** **需要维护者拍板。**
  实测（2026-09-16，当前代码只读）：两者 `checkpoints/path_versions/v4.json` 的
  `window_ranges` 是 `K=[8, 6, 4, 9]`，末窗 9 > `stage2_window_max_states=8`；
  `first_untrusted_window=0` ⟹ `tail_repartition_anchor=None`。
  于是主循环的**启动布局校验**（`abfe_pipeline.py:11306`）看到 `_tk=9 > hi` 就去调
  `_legalize_tail_window`，那里 anchor 取不到 ⟹ `RuntimeError` ⟹ **`decide()` 一次都轮不到**。
  （只读重放 `decide()` 会给 `RUN_PRODUCTION[0]`，那是**假象** —— `decide()` 不做这道校验。）
  ⚠️ `BM-03` 只阻止**再造出**这种布局，**不会**清理已经写进版本链的。这是数据状态问题，
  和 `DATA-01` 同一类，代码侧修不掉。
  两个选项：① 把版本链回退到 `v3`（末窗 K=8，合法）—— 但三次插点都打在 window 0，
  win0 的产物在每一版都是过期的，回退后仍要重采 win0 及下游；
  ② 清空这两个 run 的 stage-2 产物重跑（`DATA-01` 当时维护者选的就是这条）。
  判据：拍板后在本条写明选了哪条，并确认该 run 能走过启动布局校验进入 `decide()`。

- [ ] **🟠 DATA-03 三个 run 的 λ 总数不够，控制器自己判成「输入问题」。** **需要维护者拍板。**
  `cmet_ligand1/rep2`、`cmet_ligand2/rep1`、`p38_ligand2/rep3` 当前 `decide()` 给
  `NO_ACTION` + `NO_FEASIBLE_ACTION`，理由逐字是：
  「插 λ 的跨 resume 累计预算已用尽（`max_path_insertions=3`，版本链上已插 3 次）⟹
  继续插就是无限循环。**λ 总数不够是输入问题，应判 `HALT_LAMBDA_BUDGET_INSUFFICIENT`
  由人工改输入**」。这是**设计内的正确退出**，不是 bug —— 但它就停在这里，
  除非有人改输入。
  当前 benchmark config：`stage2_final_n_states=21`、`max_path_insertions=3`
  （`abfe-benchmark/openmm_IBS/configs/*.json`）。
  ⚠️ 改 config 会动 `stage_protocol_key` ⟹ **作废现有窗口缓存**
  （见 `incident_config_change_invalidated_all_window_caches`）。改之前先把 GPU 代价摆出来。
  判据：拍板后写明改成多少、以及这几个 run 是重跑还是放弃。


---

# 2026-09-13
> 从 [TODO.md](../TODO.md) §2 整段移出的**一条已关闭条目**，**不是待办**。
>
> | 条目 | 留下的规矩 |
> |---|---|
> | `LR-01` | 一个开关往指纹里进，**有几条路径就得收窄几条**。这次是两条（`run_config` 一条、显式 payload 一条），只收一条等于没收。以及：**顶层 run 指纹和 stage 指纹的收窄口径不同** —— 顶层 `final_results` 代表整个 run 的身份，残差开关无条件进是对的，不要跟着一起收窄 |

---

## [x] `LR-01`（已关闭 2026-09-11，2026-09-13 核实归档）`residual_sampling` 无条件进每个 stage 的指纹

**原文**（TODO §2）：

- [ ] **LR-01 `residual_sampling` 无条件进每个 stage 的指纹** ——
  `abfe_pipeline.py:11008`，在 `if stage_name == "vanishing"` 分支**之前**。
  ⟹ 打开开关会让预平衡 / attachment / decharging 的缓存**全部失配、整条链从头重算**，
  而那几段的哈密顿量根本没被残差碰过。紧邻的 MEM-00h 注释写的正是相反的做法。
  正解：挪进 vanishing 作用域（运行时判据是 `stage_name in {"vanishing","vanishing_rescue"}`，`:5489`）。

### 关闭验收（2026-09-13 只读核实）

代码里已按"正解"落地，且**两条路径都收窄了**：

| 位置 | 现状 |
|---|---|
| `abfe_pipeline.py:2752` | `RESIDUAL_SAMPLING_STAGES = frozenset({"vanishing", "vanishing_rescue"})` —— 与原文写的运行时判据逐字一致 |
| `abfe_pipeline.py:11844` | 路径一（`run_config`）：`if stage_name not in RESIDUAL_SAMPLING_STAGES: run_config.pop("residual_sampling", None)` |
| `abfe_pipeline.py:11985` | 路径二（显式 payload）：`if stage_name in RESIDUAL_SAMPLING_STAGES:` 才插 `payload["residual_sampling"]` |
| `abfe_pipeline.py:11836-11845` | 落地注释（2026-09-11）写明"它有**两条**进指纹的路径……两条都得收窄，否则打开开关会让 decharging 的 stage 缓存无谓失配、整段约 28 分钟白重跑"，并确认关着时两条路径都是 no-op ⟹ 既有指纹逐位不变 |

**不是漏网的那一处**：`abfe_pipeline.py:12933`（`_build_top_level_protocol_key`）仍然无条件插
`residual_sampling`。那是顶层 `final_results.json` 的 run 身份指纹，残差开关确实改变了这一整个
run 是什么，**无条件进是正确的**，不要照着 stage 指纹的样子去"统一"。

原文里的行号 `:11008` / `:5489` 是修复前的位置，修复后已位移，按上表的行号读。

---

## [x] `S2-F`（已关闭 2026-09-13）10 条旧断言停在旧语义

**背景**：`decide()` 重构 + §7.5 跨腿构象门裁决落地后，4 个测试文件共 10 条红着。
TODO 原先写的是「别去修，断言会随新语义一起重写」；重构落地后它变成一条真待办。

### 关闭验收

```
改前：10 failed,  76 passed   （4 个文件单跑）
改后：0 failed,   86 passed
全量：./tests/run_offline_tests.sh → 2223 passed, 3 skipped, 0 failed
```

### 留下的规矩：**先分清「断言旧了」和「fixture 旧了」**

10 条里**只有 3 条真的是断言旧了**，另外 7 条是**造数据的 fixture 旧了**。
区别决定改法，弄反了就会把测试做废：

| 类别 | 条数 | 现象 | 正确改法 |
|---|---:|---|---|
| fixture 缺自检产物 | 5 | 窗口全判 `UNKNOWN` ⟹ 「产出证据」分支（`ANALYZE`）在**所有**目标分支之前把请求吞掉 | `_mkrun` 补 `dual_window_*_self_support.json`；**且预热中的窗口不许补**（那份产物是生产跑完才写的，伪造会让 `self_verdict` 判据盖过 `phase`） |
| fixture 缺预算台账 | 1 | `warmup_steps_left is None` ⟹ `budget_unknown_fail_closed`，预算门排在所有动作选择之前 | 补 `bias_warmup.warmup_budget_ledger`。**fail-closed 本身是对的**（不知道付不付得起验证就别开新 Epoch），修 fixture 不是修门 |
| fixture 场景自相矛盾 | 1 | 「被踢出协方差链」的窗口却带着 `ANALYSIS_ELIGIBLE` 自检 | 让场景自洽：自检也判 `INSUFFICIENT_DATA`，并补上 stage 侧 `cumulative_fk_residual_production`（分析既然跑过，这份证据必然在） |
| 断言真的旧了 | 3 | 见下 | 改断言 |

⚠️ **反面教材**：如果照着实际输出把这 7 条的断言改成 `== "ANALYZE"`，它们会全部退化成
同一个「没有自检产物就 ANALYZE」的测试，缺窗 / 短生产 / DONE / skipped 四条分支的覆盖
直接归零。**那是反方向的「为了让它绿」。**

### 3 条真的旧了的断言

| 断言 | 旧 → 新 | 依据 |
|---|---|---|
| 预算耗尽窗口的出口 | `HALT_NO_ATTRIBUTION` → `HALT_BUDGET` | 归因现在是**成功的**：说得出「付不起新 Epoch 的最低验证额度」。预算可行性被提到选动作**之前**（实测 `run2/vanishing_2` 在 555k/555k 零余量时仍被判 `RECALIBRATE_FK`，开出来的 f_k 永远验不了） |
| 健康窗口 `frames_short_by` | `== 0` → `is None` | 它是**去相关帧数**的缺口，只在去相关那关真没过时才有意义。真机据旧语义印出过「还差 0 帧」这种自相矛盾的话 |
| 跨腿构象门 | `pytest.raises(ValueError)` → `gate == "WARN"` 且不阻断 | 设计文档 §7.5 裁决。⚠️ 改断言时**连带钉住三件事**，少一件这道门就退化成「警告了就等于没事」：`cross_leg_conformer_gate` 永远写、WARN 时完整 report 必须带出来、判据一字未改（`passed is False`）。`strict_cross_leg_conformer=True` 仍然硬抛 |

### 顺带更正的两处文档

- `tests/test_stage2_repair_controller.py` 模块 docstring 第 1 条原写「**缺窗口优先于一切**」——
  那正是被实测推翻的口径（缺窗分支排全局最高时，rep1 里目标被判成 win5，把支撑不足的
  win3 和卡在 local cap 的 win4 整个盖住）。已改写为「按因果顺序路由最早的未解决窗口，
  缺窗口的正确位置在因果顺序**之后**」+ 三态说明。
- 两条测试改名以反映它们现在钉的语义：
  `test_insufficient_data_without_budget_halts_on_attribution` →
  `test_no_budget_is_consumed_before_action_selection`；
  `test_combine_refuses_to_report_delta_g_bind_when_ensembles_disagree` →
  `test_combine_warns_but_does_not_block_when_ensembles_disagree`。

---

## [x] `S2-B`（已关闭 2026-09-13）续验路径上验证要求随预算膨胀

**原文**（TODO §1）：

- [ ] **S2-B 续验路径上验证要求随预算膨胀。**
  `ibs_engine.py:15575` 续验时 `validation_attempt_budget_steps = full_bias_step_budget`，
  于是 `minimum_complete_validation_frames = max(200, budget/stride)`：
  **给的预算越多、要求的帧数越高**，可达性判据的第 2 档因此失效。
  正解是把「完整性要求」与「去相关要求」解耦 —— 200 是统计目标，不该随预算浮动。

### 修法（按原定正解）

预算那一支从公式里去掉，只剩统计目标：

```python
minimum_complete_validation_frames = int(required_consecutive_bias_updates) * 20
```

### ⚠️ 但原条目的**后果描述是错的**，别再照抄

「可达性判据的第 2 档因此失效」**从没发生过**：

| 事实 | 证据 |
|---|---|
| 这个量在本仓**从来没有当过门** | 自第一个 commit（`169514e`，2026-08-31）起就只有「赋值 + 写进报告」两处，零条件判断 |
| 可达性预检的 T 读的是**另一个量** | `abfe_preoptimizer.py` 取 `validation_indeterminate.decorrelated_frames_required`（去相关下限 10），回退到 `IBS_LOCAL_MBAR_GATE_MIN_FRAMES`，**绝不回退到 200** —— 那里还留着一段注释专门警告别混 |
| 控制器侧的字段名已自带免责 | `validation_completeness_frames_REPORT_ONLY`（且全仓无人消费） |

**所以修它不改变任何判定。** 真实危害是另外两条，都成立：

1. **这个数会骗读它的人。** 可达性的 T 一度就被错取成它，gcrit 算小 20 倍，把只差
   26% 帧数的 win4 判成「差 7.5 倍、预算内不可达」（见
   [STAGE2_AUTONOMOUS_LOOP_STATUS_2026-09-11.md](STAGE2_AUTONOMOUS_LOOP_STATUS_2026-09-11.md) §9）。
   一个名叫「要求」的量实际等于「预算」，是这个误读的温床。
2. **随预算浮动的数，两次 run 的报告没法横向比。**

### 留下的规矩：**改之前先确认那个量到底有没有被消费**

`minimum_complete_validation_frames` 和 `truncated_validation_frames_ignored`
（后者恒为 0）都是 write-only。按「它是门」去推导后果，会把修复的理由和验收口径
全写歪 —— 本条目原文就是这么写出来的。**先 grep 消费点，再写后果。**

### 守卫

`tests/test_fk_relearn_and_reachability.py`：
- `test_completeness_requirement_is_decoupled_from_budget` —— 按**源码 AST** 断言
  该赋值右侧不得引用 `validation_attempt_budget_steps` / `full_bias_step_budget` 等
  任何预算量（这个量活在 `run_ibs_bias_warmup` 内部，真跑到它得起 OpenMM + 完整
  预热循环，所以按源码断言）。已用旧公式变异验证过会红。
- `test_reachability_T_is_the_decorrelated_floor_not_the_completeness_target` ——
  钉住两个量不许合并。

### 顺带记下、**没动**的一处

`truncated_validation_frames_ignored` 恒等于 0（只有 `= 0` 和写进报告两处），
却以「被忽略的截断帧数」的名义出现在报告里。要么它该被递增而逻辑丢了，要么
它该删。**不在 S2-B 范围内，未处理。**


---

# 2026-09-12
> 从 [TODO.md](../TODO.md) 整段移出的**两条已关闭条目**，**都不是待办**。
>
> 保留原文的理由是每条都留下一条仍然有效的规矩：
>
> | 条目 | 留下的规矩 |
> |---|---|
> | `BOR-01` | 同一个几何量，**"写进哈密顿量的那份"和"做校验的那份"必须共用一个实现** —— 校验那份自己解了缠、提交那份没解 ⟹ 分歧被校验函数自己掩盖掉，这种形状最难发现（跟 λ 身份那次"四份实现"是同一个病） |
> | `S2-D` | 收拢的判据不是"都塞进一个文件"，是**决策同源的进 `abfe_preoptimizer`、写盘的留 `abfe_pipeline`**；"执行器"的定义是**写盘**，由测试钉住、不靠文档约定 |

---

## [x] `BOR-01`（已关闭 2026-09-12）同一组六原子 Boresch 几何有两份实现，minimum-image 口径不一致

**原文**（TODO §7）：
  - `ibs_engine.py:22491` `_check_boresch_geometry_safe`：逐跳
    `_minimum_image_displacement_nm` 解缠后再算 r0/θ（`:22504`）。
  - `abfe_core.py:10743` `calc_boresch_from_last_frame`：裸
    `np.linalg.norm(a-b)`，**不接 box、不做 minimum image**（r0 在 `:10755`）。

  **写进限制势的是后者**（`new_eq`），做校验的是前者 ⟹ 锚点对一旦跨周期边界，
  提交的 r0 静默差一个盒矢量（~4–5 nm），而校验函数因为解了缠**看不出分歧**。

  四个调用点里只有 `runabfe.py:3798` 安全（前面刚跑过 `image_molecules_by_system`
  + `center_coordinates`）。`abfe_pipeline.py:7040` / `7191` 直接喂 `self.positions`
  —— 该对象按 `abfe_pipeline.py:2986` 的注释"会被 PBC 修复、居中、再平衡反复改写"，
  调用时刻没有成像保证；`ibs_engine.py:2388` 喂的是**运行中的 context state**，
  解耦配体在窗口里漂过边界正是最可能触发的场景。

  跟"同一不变量多份实现"是同一个形状（λ 身份那次是四份）。

  **改法**：给 `calc_boresch_from_last_frame` 接 box，走 `abfe_core.py:1524`
  的 `minimum_image_displacement_nm` —— **一处解缠**，不在 5 个调用点各加守卫。

  **判据**：
  1. 两个函数在同一份跨边界坐标上给出**逐位相同**的 r0/θA/θB；
  2. 新增回归：构造一份锚点跨边界的坐标，旧实现会差约一个盒矢量、新实现不差；
  3. 不跨边界的既有坐标上 r0/θ/φ **逐位不变**（否则会作废全部 Boresch 缓存）。

  来源：GitHub #40（R-04 的第三项）2026-09-12 的核查。

---


---

### BOR-01 关闭记录（2026-09-12）

**改法**（就是当初写的那条）：新增**唯一**解缠实现
`abfe_core.unwrap_boresch_anchors_nm(rec_coords, lig_coords, box_vectors)`，
解缠链与原校验实现一致（`H0→H1→H2`、`H0→L0→L1→L2`，逐跳）。

关键取舍：平移量取**整数格矢**
（`raw - ((raw-ref) - minimum_image(raw-ref))`），而不是 `ref + minimum_image(raw-ref)`。
不跨边界时格矢恰为 `0.0` ⟹ 返回坐标与输入**逐位相同**，既有 Boresch 平衡值
不会被一次浮点重排整体作废（判据 3）。

| 改动 | 位置 |
|---|---|
| 唯一解缠实现 + `box_vectors_to_nm_array` | `abfe_core.py`（`minimum_image_displacement_nm` 之后）|
| `calc_boresch_from_last_frame(..., box_vectors=None)` | `abfe_core.py` |
| `_check_boresch_geometry_safe` 改调同一份 | `ibs_engine.py` |
| `_box_vectors_to_nm_array` 变成 abfe_core 那份的别名 | `ibs_engine.py` |
| 5 个调用点全部传 box | `abfe_pipeline.py` ×3（`getattr` 兜 `__new__` stub）、`runabfe.py`、`ibs_engine.py` |

**判据逐条验收**（`tests/test_boresch_minimum_image_bor01.py`，5 passed）：

1. 两份实现在同一份跨边界坐标上给出一致的 r0/θA/θB —— `test_both_implementations_share_one_unwrap`
   （同时断言两边 `unwrap_boresch_anchors_nm` **是同一个对象**）；
2. 跨边界坐标上旧口径差约一个盒矢量（直接撞 `[3, 20] Å` 硬门）、解缠后不差 ——
   `test_unwrapped_geometry_matches_the_physical_conformer`；
3. 不跨边界的坐标上六个几何量**逐位不变** —— `test_contiguous_geometry_is_bitwise_unchanged`。

全套离线测试对比动手前**零新增失败**。


---

## [x] `S2-D`（已关闭 2026-09-12）Stage-2 控制器代码散在三个文件

**原文**（TODO §1《剩余缺口》，源：控制器 design §7 第 4 条）：

> **S2-D 代码仍散在三个文件** —— `abfe_preoptimizer`（`decide` / `read_aggregated`）、
> `abfe_pipeline`（执行器 + 若干 helper）、`ibs_engine`（两个纯函数）。
> 设计要求是**包在一起**，尚未收拢。

### S2-D 关闭记录

**结论不是"都塞进一个文件"**，是 **决策同源的进 `abfe_preoptimizer`、写盘的留
`abfe_pipeline`**。"执行器"的定义就是**写盘**，由 `test_controller_never_writes_anything`
钉住，不靠文档约定。

| 符号 | 位置 | 理由 |
|---|---|---|
| `relearn_epoch_required_steps` | → `abfe_preoptimizer` | 决策同源 |
| `segment_dirs_for_evidence` | → `abfe_preoptimizer` | 决策同源 |
| `lambdas_from_version_record` | → `abfe_preoptimizer` | 决策同源 |
| `existing_segment_names` | → `abfe_preoptimizer` | 决策同源 |
| `_legalize_tail_window` | 留 `abfe_pipeline` | 会 `append_version` / `record_tail_repartition_version` —— **写盘**。它用到的纯函数（插 λ / 尾段重分 / 版本记录）都在 preopt，但"合法化"这个**动作**是执行器的事 |
| `_solve_merged_segments_if_any` | 留 `abfe_pipeline` | 合并求解是**求解**不是决策；为满足归档规则给它加个回调参数，是为形式增加间接层 |
| `_latest_segment_dirs` | **已删除** | 不是"没人调所以清理"，是被 `segment_dirs_for_evidence` **取代**：真机实证「段号最大的段」可能根本没有目标窗口的帧（`vanishing_2` 是 win0-3 的部分段，拿它去重解 win4 的 f_k 既炸 loader 又逻辑不通），正确语义是「**有这个窗口数据的**最新段」。留着等于把一个已修的崩溃摆在下一个人手边 —— 原地留了墓碑注释 |
| `ibs_engine` 的三个纯函数/常量 | 留 `ibs_engine` | 本来就该在引擎侧 |
| `_run_stage2_autonomous` 主循环 | 留 `abfe_pipeline` | 它**是**执行器 |

**顺带修的真 bug**：`_relearn_epoch_required_steps` 是 `ABFEPipeline` 的类级
`@staticmethod`，却在 `_run_stage2_autonomous` 里被当**裸名字**调
（`HEAD~:abfe_pipeline.py:10841`）⟹ `RELEARN_FK_EPOCH` 分支一走到就 `NameError`。
搬成模块级后改成 `_pre.relearn_epoch_required_steps()`。
**那条分支此前从没真跑过。**

**归属由测试钉住**：`tests/test_stage2_controller_module_boundary.py`（6 条），
包括"那个 NameError 不许回来"。
逐符号现状同步在 [设计文档 §9.4](STAGE2_CONTROLLER_DESIGN_2026-09-12.md)；
§9.5 是"以后再搬的话"的注意事项。

⚠️ **一处走过弯路，记下来免得重来**：本轮曾把 `_legalize_tail_window` 一并搬进
preopt（理由是它调的三个纯函数都在那边），又曾把 `_latest_segment_dirs` 当"死代码"
直接删。前者是错的 —— **它写盘**；后者删对了但**理由错了**（不是"没人调"，
是"语义被证伪、留着是个陷阱"）。两处都已更正，设计文档 §9.4 与 preopt
迁入处的注释块同步改过。


---

# 2026-09-09
> **已归档（2026-09-12）。** 本文是从 [TODO.md](../TODO.md)《未关闭的代码缺陷》整段移出的
> **六条已关闭条目**的完整记录，一字未改。它们**不是待办**。
>
> 保留原文的理由是每条都留下了一条仍然有效的规矩：
>
> | 条目 | 留下的规矩 |
> |---|---|
> | `MIGRATE-01` | 沙箱并线**别整份拷贝文件，逐 hunk 分拣** |
> | `PBC-01` | 分子归组只信 System，载入时删 `topology.cif` 往返造出的假键 |
> | `XFAIL-01` / `XFAIL-02` | xfail 的 `reason` 写错会让人查错方向；摘标记前先确认它红在哪一档 |
> | `CACHE-01` | `--openmm-cache-only` 下不得调用 GROMACS 解析 |
> | `CFG-01` | `abfe_config.json` 的 `gmx_path` 是机器本地值，不是模板 |

---

#### [x] MIGRATE-01（已关闭 2026-09-09）EXP-031 接入主线时整份拷贝文件会抹掉 09-09 的 MEM-15 重构

> 迁移已完成（详见下方原记录）。**本条剩下的价值是一条规矩、不是任务**：
> 下次沙箱并线**别整份拷贝文件，逐 hunk 分拣**。


> **已完成（2026-09-09）**：路线 A 与路线 B 都已接入主线，融合内核（S3）本轮不接。
> 详见 `docs/archive/EXP-031_GPU_OPTIMIZATION_2026-09-09.md` 两节顶部的落地记录。
> `IBS_BIAS_PROTOCOL_VERSION` 已到 **33**，兼容集合收窄成 `frozenset((33,))`。
> 回归 `pytest tests -m cpu_only` → 1079 passed。
> ⚠️ `system_xml_sha256` 已变 ⟹ 既有 `dual_window_*` / `ibs_state_*` /
> `convergence.json` 全部失配，**必须重跑**，不要试图复用。
> 本条（MIGRATE-01）的价值转为**留给下一次沙箱并线**：别整份拷贝文件，
> 逐 hunk 分拣。

- **发现**：2026-09-09 逐文件对账沙箱 `ABFE_IBS_CUDA` 与主线时发现。
- **事实**：三个文件里**只有 `ibs_engine.py` 是沙箱领先**；`abfe_core.py` 与
  `abfe_pipeline.py` **主线在 2026-09-09 12:25 刚动过，比沙箱新**。

  | 文件 | 主线 mtime | 沙箱 mtime | 谁领先 |
  |---|---|---|---|
  | `ibs_engine.py` | 09-03 20:48 | 09-08 16:22 | 沙箱 |
  | `abfe_core.py` | **09-09 12:25** | 09-05 10:44 | **主线** |
  | `abfe_pipeline.py` | **09-09 12:25** | 09-04 15:17 | **主线** |
  | `runabfe.py` / `free_energy_engine.py` | — | — | 零差异 |

- **危险动作**：主线领先的内容是 MEM-15 分子归组重构 —— 新增
  `abfe_core.py:4645 system_molecule_grouping()` 与 `:4696 image_molecules_by_system()`
  （按 **System** 的键+约束求连通分子，不信 topology 的键），并把 `abfe_pipeline.py`
  里内联的「约束补成键」换成调这两个 helper。**这两个函数在沙箱里完全不存在**
  （全仓 grep 零命中）。⟹ 整份拷贝沙箱那两个文件会抹掉 135 + 56 行。
  而「把 EXP-031 合并到主线」最自然的执行方式恰好就是整份拷贝。
- **为什么是 P0**：这是**静默数据丢失**，不报错。抹掉的 MEM-15 修复防的是刚性水
  被逐原子回卷撕开 → 729 个 PME 排除对跨盒 → `Particle coordinate is NaN`（不到 1 ps）。
  而该损坏对既有诊断是隐形的：键能、最大键长、最小化后 max|F| 全部正常，
  只有查排除对距离才看得见。
- **处置**：**逐条重放，不是合并。** `abfe_pipeline.py` 里没有任何 EXP-031 内容
  （那条线从未改过它），它的 56 行差异 100% 属于主线，一行都不要往主线迁；
  反过来沙箱应去拉主线那份。
- **清单**：`ABFE_IBS_CUDA/experiments/EXP-031_ibs_bias_fusion/MIGRATION_CHECKLIST_2026-09-09.md`
  （逐条动作、行号、代价栏、不迁清单）；主线侧摘要见
  [EXP-031_GPU_OPTIMIZATION_2026-09-09.md](EXP-031_GPU_OPTIMIZATION_2026-09-09.md) §1。
- **连带**：`docs/EXP-031_GPU_OPTIMIZATION_2026-09-04.md` 的两条结论已推翻、头条
  `1.45×` 是水盒数（真体系 1.162×）。已在该文件顶部加取代提示，正文留档未改。

#### [x] PBC-01（已关闭 2026-09-09）`topology.cif` 往返会凭空造出假键

- **发现**：2026-09-09，brd4/ligand1 benchmark（`abfe-benchmark/openmm_IBS/runs/brd4_ligand1/rep1`）。
  attachment 腿起点体检报「2 个 nonbonded_exceptions 对跨了周期镜像（最远 7.356 nm）」，
  而输入 `.gro` 干净（六个 target 实测 0 个跨镜像水、最大分子内跨度 0.096–0.100 nm）。
- **根因**：`app.PDBxFile` 写入端链 id 按 `chr(ord('A') + chainIndex % 26)` **循环**
  （`pdbxfile.py:392/473`），读取端 `_struct_conn` 只按 `(seq_id, asym_id, atom_name)`
  解析（`pdbxfile.py:218`），**不含链序号**。链数一超过 26 就有歧义。
  brd4/ligand1 有 12549 条链（每个水一条），第一个残基 `ACE(A,1)` 的三条键被解析到
  某个同样落在 `(A,1)` 的水上：

  ```
  (0, 39543) ACE1.CH3 — HOH12582.H1
  (0, 39544) ACE1.CH3 — HOH12582.H2
  (1, 39542) ACE1.C   — HOH12582.O
  ```

  `.top` 重建 27175 键，mmCIF 往返 27178 键，多的正好这 3 条；真键一条不缺
  （读取端 `createStandardBonds()` 补齐了，所以 3 条是**净多出**）。
  溶剂腿只有 3 条链，不触发。
- **已修（2026-09-09，两步都做完了）**：

  1. `repair_pbc_molecule_integrity` 的分子归组改成只信 System
     （`abfe_core.image_molecules_by_system` / `system_molecule_grouping`），
     回卷后逐对复查、fail closed。零指纹变动。
  2. `_load_system_from_native_cache` 载入 mmCIF 拓扑后调
     `abfe_core.prune_topology_bonds_unsupported_by_system()`，把 System
     完全不认的键删掉（判据宽：`HarmonicBondForce` ∪ `CustomBondForce` ∪
     `constraints`；漏删无害、误删致命）。同时两处裸 `traj.image_molecules()`
     换成 `image_molecules_by_system()`。

  原来记在这里的两个残留消费者，第 2 步一并解决了：

  | 位置 | 影响 |
  |---|---|
  | `ibs_engine.py` `compute_u_kn` 里的 `traj.image_molecules(inplace=True)` | **载荷相关**：重算 u_kn 之前给轨迹回卷，用的是 topology 的键，而且**连约束都没补**（比修好前的生产路径还弱），外面还包着 `except → warning` 的 fail-open |
  | `runabfe.py` 末帧 Boresch 诊断处的 `traj.image_molecules(inplace=True)` | 诊断用，末帧不重锚，影响面小 |

  `runabfe.py` 构造 Boresch 锚点图时会把**非配体键原样拷贝**
  （`for a, b in pipeline.topology.bonds()`），假边原本因此进了受体侧的锚点搜索图；
  拓扑在载入时就删干净了，这里不用再改。
- **本条最初写的方案（从 `.top` 重建拓扑）没有采用，理由记在这里，别再退回去**：
  删假键比换拓扑源好三点 —— 不需要 `.top`（`--openmm-cache-only` 也能用）、
  不引入任何新边（换成 System 的全图会多出 12478 条水的 H–H 边，
  可能干扰按键距计数的逻辑，如 `LIGAND_INTERNAL_POLAR_MIN_BOND_SEPARATION`）、
  且删完的键集与 `.top` **逐条相同**（实测 27178 → 27175 == `.top` 的 27175）。
- **代价：比原估的小得多，且不需要协议版本号 bump。**
  `abfe_pipeline._topology_hash()`（`abfe_pipeline.py:599`）把 bond 列表 +
  chain.id + residue.index 一起哈希，出现在 6 处 `_protocol_fingerprint(...)`
  （`abfe_pipeline.py:5710, 9363, 9564, 10224, 10554, 12421`）。但删键是**条件触发**的：
  一条都不用删时原样返回**同一个 topology 对象** ⟹ mmCIF 干净的体系
  （链数 ≤ 26，例如所有溶剂腿）`topology_sha256` 逐位不变、resume 全保住；
  只有真的带假键的体系指纹才变，而它们的既有结果本来就不可信。
  **不要再叠一个全局协议号 bump** —— 那会把没受影响的体系一起作废，
  正是「同一件事两套机制」那个反复踩过的坑（见 `code_sha256` 那条教训）。
- **物理量影响：无。** topology 不进哈密顿量，能量/力逐比特不变；唯一真实的坐标变化
  就是把被撕开的分子拼回去（以及随之而来的 ~0.005 nm 整体质心平移）。
  例外是 `compute_u_kn`：撕开的分子原本会给出跨盒 PME 排除对的错能量，
  现在不会了 —— 只在原本就错的情形下变，且回卷失败从 fail-open 改成直接抛。
- **验证**：从 `.top` + `.gro` 原始输入端到端复现整条机制（不依赖任何 run 产物），
  往返后多出 3 条、删完与 `.top` 键集逐条相同、原子/残基/链/盒矢量全保持；
  干净拓扑返回同一对象且哈希不变。回归 `pytest tests -m cpu_only`：1071 passed。
  测试：`tests/test_membrane_barostat_protocol.py` 的
  `test_image_molecules_ignores_phantom_topology_bonds` /
  `test_prune_drops_only_bonds_the_system_does_not_back` /
  `test_pbc_repair_groups_molecules_by_system_not_topology`。


#### [x] XFAIL-01（已关闭 2026-09-09）P1-19 的 C_seam 已修好，xfail 标记已摘除

- 位置：`tests/test_charge_transfer_real_endpoints.py:453` 的
  `@pytest.mark.xfail(strict=False)`，挂在
  `test_vanishing_lambda_one_seam_matches_charging_lambda_zero` 上。
- reason 自述「实测 118.5 kJ/mol（中性 4 原子 fixture）…… **修复后此标记应转
  XPASS 并摘除**」。它**现在正是 XPASS**，但 `strict=False` ⟹ 套件不会提醒任何人摘。
- **实测（2026-09-02）**：`tools/diagnostics/probe_p119_charge_transfer_seam.py`

  ```
  abs_delta_e = 3.63206042e-04 kJ/mol      ← 不是 118.5，小 5.51 个数量级
  rel_delta_e = 1.807e-06                  (门 1e-05)
  max|ΔF|分量 = 2.526e-06 kJ/mol/nm        (门 1e-03)
  ```

- **剩下这 3.6e-4 不是 seam 残余**：同文件
  `test_bake_handoff_seam_matches_for_charged_ligand_with_realistic_geometry`
  上方的注释精确描述过它——紧凑几何（配体 4 原子挤在 <0.2 nm 内）自带一个
  「与几何基本无关的 ~0.0005 kJ/mol 绝对残差」，数值性的，不是 Hamiltonian
  构造错误。量级吻合。**没有这条排除性说明，下一个人会以为 3.6e-4 是 seam 残余。**
- **已排掉「fixture 绕过失效路径」**（这是「真修好」与「绕过去了」的唯一分界）：
  `LIGAND_CHARGES_NEUTRAL_E = (0.5, 0.3, -0.4, -0.4)` 逐原子非零，
  `LIGAND_ORDINARY_PAIRS = {(0, 3)}` 是真正的 ordinary L-L 对（未定义任何
  exception、走标准 combining rule，q_i·q_j = −0.2 e²）⟹ 内部库仑真实存在、
  **机制被触发**，但常数不见了。
- **谁修的、什么时候修的：未知。** 跨会话核对过时间线，只能**排除**：不是
  2026-09-02 那两个会话中的任何一个，也不是 λ-WCA 壳退役、也不是力组切分收敛
  （`IBS_E_BASE_FORCE_GROUPS`/`IBS_E_BIAS_FORCE_GROUPS`）的连带效果——那天第一次
  全套跑之前 seam 就已经 XPASS。⚠️ **"未知"就是未知**，不要把它写成「大概是某次
  改动的连带效果」——那种猜测会被后人当结论。
- **已处置（2026-09-09）**：标记已摘除，原 reason 里的实测数据与「3.6e-4 是紧凑几何数值底噪、不是 seam 残余」这条排除性说明改写成函数上方的注释保留。
  `pytest tests/test_charge_transfer_real_endpoints.py` → 39 passed。
  ⚠️ 仍待人工确认：P1-19 在 issue 追踪里的状态该不该一起关（本仓没有 issue 追踪器，需要在上游做）。

#### [x] XFAIL-02（已关闭 2026-09-09）两个 xfail 的 reason 是错的：它们红在 D，不在 C——根因已查明并修复

- 位置：`tests/test_charge_transfer_real_endpoints.py:633` 与 `:822`
  （`test_run_protocol_v2_matrix_cd_wiring_passes_on_charged_fixture`、
  `test_run_protocol_v2_matrix_cd_normalizes_c2_style_switch_before_c_seam`）。
- 两条的 reason 都写「同 …… 的 xfail 理由」，即都记在 P1-19 的 C_seam 失配上。
  **这是错的。**
- **实测（2026-09-02）**，复现方式
  `python -m pytest tests/test_charge_transfer_real_endpoints.py --runxfail -q`，
  两条的 `failed_frames` 完全一致：

  ```
  'failing': ['D:gate1_reference_identity,gate3_mixed_production_vs_reference']
  ```

  **前缀是 `D:`。整个输出里 `failing` 一次都没出现 `C:`** ⟹ C（seam）在这两条里
  也是通过的，红的是 **D 端点**（全解耦：λ_coul≡0 且 λ_vdw=0）。
- 讽刺的是 `:822` 那条测试名叫 `..._normalizes_c2_style_switch_before_c_seam`
  ——它本身是为 C seam 写的，却卡在 D。
- 为什么与 XFAIL-01 是**不同机制**：这两条的 fixture 是 `_case(1, n_dummies=1)`
  （净电荷 +1 + reserved dummy），走完整 `run_protocol_v2_matrix_cd`，即
  **co-alchemical charge-transfer** 路径（配体 +1 e → 0、co-ion 0 → +1 e、
  flat-bottom 位置限制 k=100 kJ/mol/nm²）。失败的是 co-ion 在 λ=0 时的
  reference identity 与 mixed(CPU)-vs-reference 一致性，跟「配体内部库仑常数」
  没有关系。
- ⚠️ **不是静默的生产 bug**：`charge_treatment=co_alchemical_charge_transfer`
  本就 `production_qualified=False`，PHY-03（P1，见本节下方）仍挂着。属于
  **已知未合格路径上的已知未合格行为**，只是被错标成了 P1-19。
  **别当 P0 处理。**
- **已处置（2026-09-09）：不是重写 reason，是直接修好了根因；两个标记已摘除。**

  上面那段「跟 co-ion 有关」的推断**是错的，别再沿用**。逐 gate 打开 report 后
  （`gate1`/`gate3` 都给出同一个数）：

  ```
  D  abs_delta_e = 22.057534 kJ/mol   rel = 5.108e-05 (门 1e-05)
     max|ΔF| = 1143.130726 kJ/mol/nm  全部落在原子 0 的 x 分量
     reference 侧该原子受力 = 恰好 0.0     production 侧 = -1143.13
     strict_zero_reference / strict_zero_mixed 都 PASS ⟹ 配体–环境项确实是零
     C 两个 gate 都 PASS，abs_delta_e = 5.2e-4（就是 XFAIL-01 那个紧凑几何底噪）
  ```

  手算 fixture 里唯一的 ordinary L–L 对 `(0,3)`（Lorentz-Berthelot，
  r=0.265149 nm、σ=0.34 nm、ε=0.36 kJ/mol）：

  ```
  U_LJ(0,3) = 22.057512 kJ/mol      |F| = 1143.129513 kJ/mol/nm      方向 +x
  ```

  六位有效数字吻合 ⟹ **差值 100% 就是普通 L–L 对的内部 LJ**。

- **根因**：`tools/validation/compare_charge_transfer_endpoints.py`
  的 `reference_vanishing_zero_system` 把配体粒子 epsilon 一律置零，
  连**普通（无 exception）L–L 对**的 LJ 一起杀掉了；生产侧 `U_common` 正确地
  逐 λ_vdw 保留它。是 XFAIL-01（库仑版）的 **LJ 版本**，但错在**参照构造侧，
  不在生产侧**。
- **为什么会出现**：v3 [P0-01] 停掉了 `reference_charging_endpoint_system` 的
  内部对补 exception——对**库仑**是正确的（普通 L–L 库仑必须随粒子电荷线性湮灭），
  但普通对的 **LJ** 从此失去了庇护。该函数的 docstring 当时还写着
  「配体内部 LJ 已经被冻结成显式 exception」，已经陈旧。
- **已修**：置零之前先把普通 L–L 对冻结成显式 exception，搬进 combining-rule 的
  σ/ε；chargeProd 取当前粒子电荷之积（λ=0 时为 0）⟹ 库仑逐比特不变，
  PME 的 exception 倒扣项同样正比于 q_i·q_j ⟹ 也是 0。
  只补普通对，raw 拓扑自带的 excluded/1-4 exception 一个不动 ⟹ 没有触碰 P0-01。
- **影响面**：该常数逐 λ_vdw 不变 ⟹ **ΔG_vdw / ΔG_bind 不受影响**；且只改验证工具，
  生产哈密顿量一行未动，协议版本号未动。
- **定级结论：既不属于 PHY-03，也不是生产 bug**，是验证参照的构造缺陷。
  PHY-03 仍然独立挂着，状态不变。
- **回归**：新增 `test_reference_vanishing_zero_keeps_ordinary_intra_ligand_lj`
  直接对着机制断言（普通对必须存在且 σ/ε 等于 combining rule，配体粒子 epsilon 仍须为 0）——
  原来的 D 门失败信息只说 `D:gate1/gate3`，看不出是 LJ，正是这次查了半天的原因。
- **为什么这条值得单列**：三条 xfail 共用一条错 reason，是**能自我掩盖的**——
  谁照那两条的 reason 去修 C seam，会去修一个已经修好的东西；而真问题
  （co-ion 在 D 端点）继续没人管；而且它不会在测试里报警，因为 XFAIL 也算"预期"。

#### [x] CACHE-01（P2，纯噪音）`--openmm-cache-only` 下无条件调用 `find_gmx_include_dir`

**已修（2026-09-02）。** `find_gmx_include_dir(config.gmx_path)` 挪进了非
cache-only 的 `else` 分支，cache-only 路径 `include_dir` 留 `None`
（`runabfe.py` 里搜 `[CACHE-01`）。cache-only 下不再打那条警告。

- 原症状：带 `--openmm-cache-only` 跑时照样打「找不到 GROMACS 力场 include 目录」
  警告，而这一路根本不需要 include 树。
- 修改前已查清：审计通过的缓存上 `include_dir` **一次都不会被解引用**（三条使用
  路径逐个核对过，见 [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) 同名小节）。
  ⟹ 只是噪音，不影响任何数值，挪动对 cache-only 路径行为中立。
- 挪动后复核过的那一条：`system_cache_exists(...)` 仍然在 `or` 的右侧，
  `openmm_cache_only=True` 时整个调用不执行（短路顺序未变）。
- 来源：2026-09-02 运行期记录（原文已归档到
  [archive/RUNTIME_ISSUES_2026-09-02.md](RUNTIME_ISSUES_2026-09-02.md) BUG-1）。

#### [x] CFG-01（P2，配置）`abfe_config.json` 的 `gmx_path` 指向不存在的路径

- **2026-09-07 已修**：发布清理时清空为 `""`（机器本地键，见
  `abfe_diagnostics._MACHINE_LOCAL_KEYS`），留空回退到 `GMXLIB`/`GMXDATA`/`PATH`
  自动探测。下面是当时的记录。
- 原值 `/home/ruigengji/gmx26.0C`，**本机不存在**。
- 2026-09-01 那次实跑的 provenance 记的是
  `/home/ruigengji/gmx26.3/share/gromacs/top`，与 config 里的值不同 ⟹ 配置里的
  值从来没被那次运行用上（那次带了 `--openmm-cache-only`）。
- 与 CACHE-01 **是两件事**：CACHE-01 是「不该问」，这条是「问了但答案是错的」。
  非 cache-only 路径会真的用到它。
- 修法：写 GROMACS 的**安装前缀**，不要写 `share/gromacs/top`
  （解析逻辑见 `runabfe.py:557` `find_gmx_include_dir`，两种写法都能吃，但前缀是
  2026-08-31 之后的约定）。**改配置会动 provenance，需用户确认取哪个版本的 GROMACS。**
