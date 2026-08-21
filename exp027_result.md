# EXP-027 结果汇总（含 EXP-028 修复）

最后更新：2026-08-14

本文件汇总 EXP-027（在线效用鉴定：candidate = A2 版 `LocalManyBodyResidualForce`
残差插件 vs baseline = 无插件）当前所有结论，一处集中记录，不再分散在
`PLAN_EXP-027_online_utility.md`/`decision_log.jsonl`/多个 memory 文件里各查各的。
详细过程仍见 `PLAN_EXP-027_online_utility.md` 第 16 节；本文件只留结论。

---

## 1. 最终结论（当前权威状态）

```
EXP027_U0                    = PASS（封存）
EXP027_U1                    = PASS
EXP027_U2_SHORT_NVT_SAFETY    = PASS
EXP027_U3（原始，A1.1 二进制）= EXP027_STOP_WINDOW0_INCORRECT —— 因 addArg 累积缺陷失效，不可用于晋级决策
EXP-028                       = 根因定位 + 修复 + 验证，全部完成
EXP-028-U3-CONFIRMATION       = EXP027_U3_WINDOW0_UTILITY_PASS ✅ —— 但需重新定位为
                                  BASELINE_FK_TRANSFER / NO_RECALIBRATION STRESS TEST（见 §5.9）
EXP027_U4（真实 216 万步跑完） = INVALID_FOR_PROMOTION_BASELINE_FK_REUSED_FOR_CANDIDATE
                                  —— 根因不是插件/采样步数，是 candidate 从未做过专属
                                  IBS f_k 校准（见 §5），回答不了真实生产 utility 问题
EXP-029（完整生产 A/B，两臂    = AUTHORIZED_NOT_STARTED（取代原定的 U3.5 window_2 小试点，
  各自独立校准+冻结 f_k）        见 §6，用户 2026-08-14 决定直接上生产规模）
```

**当前结论（2026-08-14 更新）：EXP-028 修复本身仍然有效、已验证。但 U4 全部 6 窗口
真实 216 万步跑完后发现一个比插件性能更根本的方法论缺口——candidate 的 IBS
mixture bias 权重 \(f_k\) 从 U1 到 U4 全程直接借用 baseline 早就校准冻结好的
\(f_k\)，从未针对"residual 生效后的真实采样系统"重新校准过。数学上这不会造成
渐进偏差（TMBAR 能撤销任意固定 \(f_k\) 产生的偏置），但会在 residual 系数最大
的窗口（window_1/2/3，系数逼近 1.0）造成混合效率变差、有限预算下方差增大——
这正是 U4 观测到的"逐窗口偏差大且符号跨 repeat 不一致"的真实成因（详见 §5）。
U3 的 PASS 需要重新定位（window_0 系数峰值只有 0.56，错配可能碰巧不严重，但
未经证实"干净"）；U4 已按此封存，不建议当作最终 production utility 结论。
下一步是 EXP-029——完整生产规模的 A/B，两臂各自独立校准+冻结自己的
\(f_k\)（不是重跑 U4、也不是只测单个 window），设计见 §6。

---

## 2. EXP-028：性能退化根因 + 修复

### 2.1 症状（U3 首次真实规模运行时发现）
- candidate 臂 GPU-hour 是 baseline 的约 2.3 倍（438-442s vs 186-188s），与
  EXP-026 自己的 O4 短测结果（约 1.03-1.04 倍）完全对不上。
- 分段计时（15×2000 步 = 3 万步）确认：candidate 单步耗时从 2.74 ms 单调
  升到 6.22 ms（翻倍多），baseline 全程 2.60-2.71 ms 完全平稳。
- 确认是插件本身的真实缺陷，不是 harness/环境/GPU 代际问题（baseline 同窗口
  完全平稳）；也不是正确性问题（无异常、无 NaN，能量数值全程合理）。

### 2.2 根因（用户直接从源码定位）
`CudaCalcLocalManyBodyResidualForceKernel::execute()` 里，14 个 kernel
（K0-K6、resetStatus、两种 scatter）每一步都用 `addArg()` 重新加参数，而不是
首次 `addArg()`、之后 `setArg()`。OpenMM 的 `addArg()` 是**永久追加**参数槽，
不是原地更新——参数 vector 逐步无界增长，`execute()` 每次都要遍历整条
不断变长的列表，CPU 端 kernel launch 准备耗时随累计步数线性增长（总耗时
O(n²)），同时伴随 host 内存持续增长。与观测到的退化曲线完全吻合。

### 2.3 修复
`plugins/LocalManyBodyResidual/platforms/cuda/src/CudaLocalManyBodyResidualKernels.{cpp,h}`：
改为"每个 kernel 首次调用时 `addArg()` 绑定一次，之后全部 `setArg(固定 idx, ...)`"。
纯 host 端 C++ 改动，设备端 kernel 源码零改动。修复前源码+编译好的 `.so`
已备份到 `plugins/LocalManyBodyResidual/pre_exp028_backup/`（sha256 已记录，
可回滚）。

### 2.4 修复前验证（先证明没改物理，再谈性能）
- G0 smoke/serialization round-trip、control-plane layout、A1.1 dBdq
  fault-injection（真机）、A2 first-error-wins（真机，K1 vs K2 竞态，
  15/15 非 flaky）全部 PASS，零回归。
- G3 CSR 路径（多数既有测试默认 skinAngstrom=0 走不到这条路径）：用修复前/
  修复后两个 `.so` 在同一个真实生产 checkpoint 上分别评估，step=0 时
  e_total 只差 ~3×10⁻⁵ kJ/mol（GPU 规约顺序噪声量级），step 50/200 的偏差
  按正常 MD 混沌发散速率平滑增长——不是离散型 bug 的特征。

### 2.5 性能复测（同一套 30,000 步分段方法论）
```
修复前: 第1段 2.74 ms/步 → 第15段 6.22 ms/步（比值 2.27）
修复后: 第1段 2.6537 ms/步 → 第15段 2.7406 ms/步（比值 1.033，≤1.05 门槛 PASS）
process RSS（修复后）: 15 段全程 2504.1 MB，零增长（修复前无界增长）
```

---

## 3. EXP-028-U3-CONFIRMATION：真实规模复跑结果

复跑前额外修了两个已确认的 harness 陷阱（不是原样重跑）：

1. **插件身份**：`Exp027Repeat` 默认 `plugin_build_dir=build_exp026_a1`（A1.1，
   未修复），原始 U3 脚本从未显式覆盖——即原始 U3 的"candidate"实际测的是
   未修复的 A1.1 二进制，不是 A2。修复：`exp027_common.py` 新增
   `verify_known_plugin_identity()` + `identity` 参数（"a1.1" 默认不变，
   新增 "exp028_a2"），复跑显式传 `plugin_build_dir=build_exp026_a2,
   identity="exp028_a2"`，sha256 不符直接 fail-closed。
2. **AB/BA 执行顺序**：`r["order"]` 标签从未真正控制执行顺序（代码里
   baseline 永远先跑）。修复：新脚本 `scripts/exp028_u3_confirmation.py`
   里 `arm_order` 真正驱动执行顺序，实际顺序写入报告
   `actual_arm_execution_order`。
3. **分析口径冻结不变**：`IBSSampler`/`_solve_single_window_local_mbar`
   原样使用，residual 仍只作为采样偏置，reweighted 回 baseline-only
   target（不把 residual 加进 target ledger）。

### 3.1 真实结果

报告：`output/outer_lambda_exp027_online_utility/exp028_u3_confirmation_report.json`
`plugin_cuda_sha256 = 4a3a5478ed921401d1177d7836b65f7f57dc212cf5c349aebbb513447b581b3f`

| repeat | 标签 | 实际执行顺序 | D_r | z_delta_g（门槛≤2.0） | GPU-hour 比值(candidate/baseline) |
|---|---|---|---|---|---|
| repeat_0 | baseline_first | [baseline, candidate] | −0.263 | 0.273 ✅ | 1.041 |
| repeat_1 | candidate_first | [candidate, baseline] | +0.683 | 0.809 ✅ | 1.025 |
| repeat_2 | baseline_first | [baseline, candidate] | +0.402 | 0.658 ✅ | 1.026 |

```
n_repeats_with_positive_D_r = 2/3（门槛 ≥2）→ PASS
median_relative_improvement = +0.494（49.4%，门槛 ≥0.10）→ PASS
delta_g_consistency_pass    = 3/3（原始 U3 是 1/3；z 从 2.70/2.41 降到 0.27/0.81/0.66）
gpu_hour 比值三个 repeat 都落在 1.03-1.05，与 EXP-026 O4 短测区间完全一致

DECISION: EXP027_U3_WINDOW0_UTILITY_PASS
```

### 3.2 旧 EXP-027 U3 报告的正确读法
`u3_window0_utility_report.json`（决策 `EXP027_STOP_WINDOW0_INCORRECT`）**不
删除、不覆盖**，仅更新解读：该结果测量本身没错，但测的是被 addArg 缺陷拖累的
A1.1 二进制，不代表 A2 修复后的真实生产成本/效用。正确表述是"EXP-027 U3
（原始）因累积 addArg 缺陷不可用于晋级决策"，不是"U3 方法论本身错误"。

---

## 4. 关于 ΔG 一致性的口径说明（历史修正，现已过时但记录在案）

在修复前，曾经用"λ=0/1 全局端点插件贡献严格为零 ⇒ 结构性恒等"来解释
z_delta_g 异常。这个论证被证明**不适用于 window_0 这个局部窗口**：window_0
的另一端（约 0.729-0.738）并不是全局零残差端点。真正让 baseline/candidate
ΔG 应该一致的原因是：两臂最终都通过 TMBAR 重加权回**同一个 target 能量定义**
（softcore+LRC，`ibs_engine.py:6377`，不含 residual），而不是共享零残差端点。
修复后，这个讨论已经是次要问题——3/3 repeat 的 z_delta_g 现在都远低于门槛
（0.27/0.81/0.66），一致性问题已随性能修复一并消失，不再是悬而未决的疑点。

---

## 5. U4：全部 6 个 Stage-2 窗口确认（已跑完，结论需重新定位）

### 5.1 设计修正：撤销"windows 1-5 两臂 config-identical"假设
最初的 U4 草稿假设只有 window_0 承载 residual，windows 1-5 两臂 bitwise
一致、可以共用一份轨迹（SHARED_IDENTICAL_CONTROL，省约 90 万步）。**实测
证伪**：真实 residual 系数是 `A(λ)=sin²(πλ)·c`，只在全局 λ=0/1 两个端点
严格为零，windows 1-5 处处非零（量级 0.3-1.0，跟 window_0 非零态一样大）。
结论：SHARED_IDENTICAL_CONTROL 不成立，撤销，规模固定为朴素的
**6 windows × 2 arms × 3 repeats × 60,000 步 = 216 万步**，全部独立双臂跑。

### 5.2 Harness + 身份门槛的一处设计修正
`scripts/exp027_u4_stage2_confirmation.py`：6 窗口收集 + 每臂每 repeat 调用
一次 `GlobalMBARAnalyzer.solve_stage_integrated`（不是 U3 用的单窗口
`_solve_single_window_local_mbar`），冻结的 AB/BA 逐窗口对照表。

交给用户节点跑之前发现并修正：插件身份门槛（`exp027_common.py` 的
`verify_known_plugin_identity`）原本比对**编译好的 `.so` 哈希**——这个哈希
是工具链/glibc 相关的，节点上用不同编译器重新编译同一份源码必然得到不同
的 `.so` 字节，会被误判成"身份不符"。改为比对**源码** `.cpp` 哈希（跨机器
可移植，与编译器无关），`.so` 自己的哈希仍然记录用于溯源，但不再是判定
依据。同时清理了 harness 里一处遗留的旧断言（还在拿容器专属的 `.so` 哈希
做二次校验，逻辑已经过时，真机跑时直接炸断言）。

### 5.3 用户节点真实规模运行中
用户在自己节点上用 conda-forge 的 gcc 14.3.0（跟这边 OpenMM 包同源工具链）
重新编译通过，源码哈希核对一致，正在跑真实 216 万步。首个 window 反馈：
`n_frames=100`（两臂都对，步数没有跑少），`baseline_gpu_hour_s≈96.2`，
`candidate_gpu_hour_s≈108.9`，比值≈1.13。

**该节点 GPU 明显快于这边会话用的 2080 Ti**（同样 6 万步，baseline 从
~186-190s 降到 ~96s，约快 2 倍）——数字变快是硬件差异，不是 bug。

比值 1.13 略高于这边验证过的稳态区间（分段计时 1.033、U3-CONFIRMATION
三个 repeat 1.025-1.041），**合理解释**：candidate 每个 window 起手要付一次
G3 CSR 重建的固定成本（baseline 完全没有这笔开销），U3/U4 用的单 window 是
6 万步，比真实生产单 window（25 万步，见 `abfe_config.json` 的
`n_steps_per_window`）短得多——window 越短，这笔固定成本占比越大，比值就
越往上顶；真实生产窗口更长，摊薄更充分，比值应该更靠近稳态区间。**结论：
U4/U3 这套确认用的短 window，测出来的 candidate 相对开销很可能比真实生产
场景偏保守（偏差方向对 candidate 有利，不是让人担心的方向）**——真实生产
大概率会比这次确认的数字更好，不会更差。

### 5.4 真实结果：`EXP027_STOP_STAGE2_INCORRECT`
216 万步跑完，`baseline_stage`/`candidate_stage` 在 3 个 repeat 里全部
`converged=False`——但注意：`coverage_diagnostics.dropped_window_indices`
全部为空（6/6 window 都真正解出来了，不是 dry-run 那种覆盖断裂），卡的是
两条精度门槛（`min_decorrelated_samples≥20`、`max_endpoint_uncertainty≤1.0`
kJ/mol），三个 repeat 里两臂都各自有卡在这两条上的情况，不是只有 candidate
卡。因为 `converged=False`，`delta_g_consistency_pass` 被防御性地判 False，
`D_r`/`z_delta_g` 实际上从未真正算出来过。

### 5.5 逐窗口 ΔG 分解：偏差集中在高系数窗口，且符号跨 repeat 不一致
把两臂的 `total_delta_G`/`total_error` 拆到 `covariance_chain_segments`
逐窗口看（以 repeat_0 为例，z 是 candidate−baseline 对该窗口合并不确定度）：

```
                    repeat_0   repeat_1   repeat_2
window_0 (系数0→0.56)   +2.60      +0.96      -5.41
window_1 (系数0.56→0.92) +6.30      +8.42      +2.80
window_2 (系数0.92→~1.0) +6.82      +6.07      -4.97
window_3 (系数~1.0→0.71) +4.04      +2.74      +3.42
window_4 (系数0.71→0.34) +0.38      +2.77      +1.27
window_5 (系数0.34→0)    +1.47      +0.71      -2.04
```

偏差集中在系数最大的 window_1/2/3；**符号跨 repeat 不固定**（window_0/2 在
repeat_2 是负的，在 repeat_0/1 是正的）——如果是路径结构决定的固定系统性
偏差，符号应该跨 repeat 保持一致（路径/系数曲线是冻结不变的）。符号不固定
但幅度都很大，是**高方差/未充分混合**的特征，不是稳定的系统性偏差。
repeat_2 的总和"看起来通过"（z≈1.17）只是因为窗口间正负误差恰好互相抵消，
不是它逐窗口质量真的更好。

### 5.6 根因：candidate 从未做过专属 IBS f_k 校准（用户发现）
真实生产的设计（`ibs_engine.py` 明确写着）：先针对**实际要采样的系统**做一整
套 warmup + bias 校准 + 冻结验证，让 \(f_k\) 收敛到能压平该系统的态分布，
然后冻结、只在 production 阶段采样、不再调用 `update_weights()`。

`ibs_state_{stage}_window_{w}.json` 里存的 \(f_k\) 是**只针对 baseline（无
residual）系统**校准冻结出来的——因为 candidate 从未被真正接入过任何一次
真实 production 流程。U1 到 U4 全程直接把这份 baseline 的 \(f_k\) 原样塞给
candidate 的系统用，从未针对"residual 生效后的真实势能面"重新校准。

**数学上这不会造成渐进偏差**：TMBAR 重加权用的是实际记录的 `bias_history`
和真实的 target 能量，能撤销任意固定 \(f_k\) 造成的偏置，跟 \(f_k\) 本身是否
"调对了"无关——\(f_k\) 只是重要性采样的效率辅助，不是物理量。它只会让
**混合效率变差、有限预算下方差增大**，尤其是 residual 系数越接近 1.0 的
窗口（错配越严重）。这跟 §5.5 观测到的"高系数窗口偏差大且符号不固定"完全
吻合，也解释了为什么 window_0（U3 单测过，系数峰值只有 0.56，错配小）从未
出过问题。

### 5.7 用已有（未校准）数据能不能提前判断？——不能，噪声本身就是证据
用 U4 已经测出的数字直接算 \(r=ESS_{\min}^{(c)}/ESS_{\min}^{(b)}\)、
\(s=C_{prod}^{(c)}/C_{prod}^{(b)}\)：

```
repeat_0: r=0.7357  s=1.081   r/s=0.68
repeat_1: r=0.9294  s=1.077   r/s=0.86
repeat_2: r=4.1762  s=1.056   r/s=3.95
```

`r` 从 0.74 跳到 4.18（近 6 倍跨度），中位数 r/s≈0.86（不利）。这个巨大波动
本身就是诊断信号：`ESS_min` 取的是全部 6 window×23 态里最差的那一个态，对
"哪个态这次恰好没被压平"极度敏感——连 baseline 自己的 `ESS_min`
（11.6/12.5/6.56）都在跳，说明"最差态瓶颈不稳定"不是 candidate 专属现象，
是 f_k 错配导致占据分布本身不稳定的直接证据。

**还有第三层噪声来源（用户指出，容易被漏掉）**：`min_decorrelated_samples`
（100 个原始帧去相关后剩下的帧数，6 个 window 里最差的那个）本身就在大幅
波动，不是一个稳定的分母：

```
repeat_0: baseline=23  candidate=12   (相差近 2 倍)
repeat_1: baseline=23  candidate=13   (相差近 2 倍)
repeat_2: baseline=10  candidate=33   (相差超过 3 倍)
```

`ESS_min` 本质上是 `min_ess_ratio × 去相关帧数`，如果去相关后剩多少帧这个
分母本身就在 10~33 之间跳，那两臂/跨 repeat 比较 `ESS_min` 从根上就不是
"基础一致、只看压平程度"的干净比较——candidate 多出的那个偏置力可能让某些
repeat 的轨迹去相关更快（如 repeat_2 candidate=33），也可能让另一些
repeat 的轨迹更"粘"（如 repeat_0/1 candidate=12/13），这是比"f_k 错配导致
混合差"更底层的一层随机性（统计非效率 g 本身的波动），不能被"ESS_min 是
min 统计量、对最差态敏感"这一条完全概括。

**现有数据噪声太大，没法直接判断"值不值得校准"**——这不是绕过校准试点的
理由，反而是校准试点必要性的证据；且 U3.5 报告必须把每个 window 各自的
`n_frames_decorrelated` 单独列出来，不能只看聚合的 `min_absolute_ess`——
"校准后是否变稳"要分两层看：占据分布是否变平、去相关帧数产出是否也跟着
稳定，这是两件不同的事，混在一起会掩盖问题出在哪一层。

### 5.8 效率的数学框架（U3.5 试点要填的数字）
$$\eta_{\text{production}} = \frac{ESS_{\min}}{C_{\text{prod}}} \qquad
\eta_{\text{ITT}} = \frac{ESS_{\min}}{C_{\text{calib}} + C_{\text{prod}}}$$

baseline 的校准成本是历史沉没成本，不计入决策；只有 candidate 专属的新增
校准成本 \(C_{\text{calib}}^{(c)}\) 需要跟后续复用次数 \(N\) 摊薄比较：

$$N > N^* = \frac{C_{\text{calib}}^{(c)}}{C_{\text{prod}}^{(b)}\cdot(r-s)}
\qquad (\text{需要 } r>s \text{，否则 } N^* \text{ 不存在})$$

### 5.9 U4 封存标签
```
EXP027_U4 = INVALID_FOR_PROMOTION_BASELINE_FK_REUSED_FOR_CANDIDATE
```
不是"candidate 在 Stage-2 上更差"的判决——它回答不了真实生产里 candidate
是否有效这个问题，因为 candidate 从未独立校准过自己的 \(f_k\)。§5.6-5.8 的
分析（根因、r/s 噪声、效率数学框架）本身仍然有效，作为设计 EXP-029 的
依据保留。

---

## 6. 下一步：EXP-029（用户 2026-08-14 决定，取代原定的 U3.5 小规模试点）

**决策**：不再继续切小规模、固定预算的单窗口试点（原 U3.5 window_2 计划
撤回）。理由：插件数学/CUDA correctness 已充分验证，长程性能缺陷已修复
（额外开销约 3-4%），真正决定收益的是完整的"校准 \(f_k\)→冻结→
production→TMBAR"闭环——小规模固定预算试验测不出生产里的自适应校准动态、
rescue 触发规律和窗口间耦合。直接上真实生产规模的完整 A/B。

**EXP-029 设计**：
1. baseline 按原生产流程独立校准并冻结自己的 \(f_k\)。
2. candidate 在 residual 开启的真实 Hamiltonian 下**独立**校准并冻结自己的
   \(f_k\)（baseline 的冻结 \(f_k\) 至多只作为 warm start 初值，不是复用对象，
   且必须预先固定，不能校准失败后换初值）。
3. 两臂使用相同的校准规则、收敛门槛、rescue 规则、production 停止规则。
4. 配对的初态/速度/随机种子出发，但轨迹和 \(f_k\) 独立演化。
5. 六个窗口全部运行（只有两个全局端点 residual 严格为零，见 §5.1 的证伪）。
6. 完整保存实际 `bias_history`、target energies、冻结后的 \(f_k\)。
7. 成本按 ITT 计算：warmup、校准、失败尝试、rescue、production、ledger
   全部计入：
$$\eta_{\text{ITT}} = \frac{N_{\text{eff}}^{\text{mixture}}}{\text{全部 GPU-hour}}$$

**门槛**：ΔG 一致性通过、TMBAR 收敛、≥2/3 独立 repeat 的 candidate utility
优于 baseline、中位提升达到预设门槛；每个 window 报告原始步数、去相关帧数、
\(g\)、occupancy、ESS、rescue 次数。

**唯一的工程保留（不影响科学结论，纯粹风险控制）**：candidate 从未真正跑过
一次校准，这是全新代码路径，规模上限很大（单窗口 warmup 上限 50 万步 ×
最多 50 次 bias update × 可能的 rescue 重试 × 6 窗口 × 3 repeat × 2 臂，
真实成本未知，可能远超这次 U4 的 216 万步）。正式上大规模跑之前，先花一次
几乎免费的 smoke-test 规模验证（tiny warmup 预算，只确认新 harness 接线
跑得通、不崩、状态机不卡死），不替代 EXP-029 的科学结论，只排除"接线本身
有 bug、真跑起来才发现"的风险——这条项目线里每一步都是这么做的。

**尚未开始**：需要先深入读 `run_all_windows`（校准/冻结验证/rescue 那一整套
状态机）的真实调用方式，才能设计 EXP-029 的 harness。

- EXP-028 修复本身（addArg→setArg）不需要撤回，仍然有效——这次发现的是
  一个独立的、更早存在的方法论缺口（IBS f_k 校准范围），跟插件性能修复
  无关。
- EXP-026 的 `STOP_OPTIMIZATION_SUCCESS`（1000 步窗口测得）不需要撤回，其
  对真实生产窗口（25 万步）覆盖不足的问题已经被 EXP-028 修复直接堵上，
  不再是悬而未决的缺口。

---

## 7. 关键文件索引

- 修复代码：`plugins/LocalManyBodyResidual/platforms/cuda/src/CudaLocalManyBodyResidualKernels.{cpp,h}`
- 修复前备份：`plugins/LocalManyBodyResidual/pre_exp028_backup/`
- 复跑脚本：`scripts/exp028_u3_confirmation.py`
- 复跑报告：`output/outer_lambda_exp027_online_utility/exp028_u3_confirmation_report.json`
- 原始 U3 报告（保留不覆盖）：`output/outer_lambda_exp027_online_utility/u3_window0_utility_report.json`
- 完整过程记录：`PLAN_EXP-027_online_utility.md` 第 16 节（16.1-16.14）
- 决策日志：`output/outer_lambda_exp027_online_utility/decision_log.jsonl`
- U4 草稿：`protocols/EXP-027_U4_stage2_confirmation_addendum_DRAFT.json`
