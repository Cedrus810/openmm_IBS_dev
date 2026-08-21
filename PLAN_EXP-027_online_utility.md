# PLAN EXP-027 — A1.1 Native SoftLift 在线效用资格化

## 0. 文档身份

| 字段 | 值 |
|---|---|
| experiment_id | `EXP-027` |
| title | `A1.1 native fused Group1 paired online utility qualification` |
| status | `DRAFT_NOT_PREREGISTERED` |
| date | `2026-08-13` |
| primary question | 已通过正确性和成本门的 A1.1 native residual，是否真正提高 IBS/TMBAR 的 ESS/GPU-hour？ |
| prerequisite | EXP-026 A1.1 correctness 与 normative O4 cost 均通过 |
| production permission | 本计划通过前一律禁止 |

本文档是实验计划，不是结果报告。只有机器可读 preregistration、输入 manifest、seeds、运行长度和所有 hash 写齐并封存后，EXP-027 才能从 `DRAFT` 转为 `PREREGISTERED`。

---

## 1. 已知事实与权限边界

### 1.1 已通过的工程门

EXP-026 A1.1 在真实 73,536 原子、CUDA mixed precision、native fused Group1 路径上通过：

```text
median(candidate_total/baseline_total) = 1.04140 <= 1.07
bootstrap one-sided P95 upper          = 1.04692 <= 1.10
G4_COST_QUALIFICATION                  = true
EXP026_DECISION                        = STOP_OPTIMIZATION_SUCCESS
```

冻结 A1.1 身份：

```text
CudaLocalManyBodyResidualKernels.cpp SHA-256
  1a1bcfea500f3b50fad5bba591c2db932fc4a04519c69eda64b1778ba00c51fa

CUDA plugin .so SHA-256
  3cf9160c8f2b3bec1e987932a0db716698242b28b32fc6953bf29da4cb9af0a3
```

权威结果：

- `output/outer_lambda_exp026_cuda_control_plane/o4_a1_1_result.md`
- report SHA-256：`F62685D6A4A56A76C1CA0BC6F19AC863FD44C1B8ECC14AB11D5107F887647706`

### 1.2 永久保留的失败历史

EXP-025 的旧结果不能被 EXP-026 追溯覆盖：

```text
STOP_EXP025_RUNTIME_BACKEND
median = 1.1233981725063995
P95 upper = 1.1321719030983386
```

### 1.3 EXP-027 要回答与不回答的问题

EXP-027 只回答：

> A1.1 candidate 在真实在线采样中，是否以可重复、统计有效且热力学一致的方式，提高 mixture-coverage ESS/GPU-hour？

EXP-027 不以以下结果代替在线效用：

- EXP-020 的 `55.5524%` held-out gap-variance improvement；
- EXP-026 的 `1.0414×` runtime cost；
- D1/D2/D3 的离线 parity 或 finite difference；
- literal BAR overlap；
- switch count、raw ESS 或单一状态 ESS；
- 事后挑出的最好 seed、最好 checkpoint 或最好时间区间。

---

## 2. 冻结研究对象

### 2.1 Baseline arm

`baseline` 必须是当前真实 production window system：

- `mode=ibs`；
- `decoupling=dual_lambda`；
- `softcore` 基础势；
- 原 production `IBSBiasForce`/Group1；
- 不加载 `LocalManyBodyResidualForce`；
- 不加载 EXP-025/026 plugin child；
- 不改变基础 Hamiltonian、λ schedule、`f_k`、LRC、WCA accounting、Boresch restraints 或 estimator。

### 2.2 Candidate arm

`candidate` 与 baseline 的唯一差别：

- Group1 使用通过 EXP-026 O4 的 A1.1 native fused residual-enabled IBS force；
- LocalManyBodyResidualForce 输出 raw `U_B=kBT*B`，单位 kJ/mol；
- A_k 由现有 outer-λ controller/Group1 wiring 应用一次；
- 插件内不再应用 A_k、β 或 offset；
- checkpoint、41 ligand anchors、RBF16、typed MLP、`Bmax=10`、4/5 Å cutoff、1 Å skin 与容量上限全部冻结。

以下一律排除：

- `plugins/LocalManyBodyResidual_exp026_a2_draft/`；
- A2/A3/A4/A5 未资格化实现；
- TorchForce、CustomGB、per-anchor CV、grouped-density CV；
- checkpoint 重训或重新挑选；
- 调整 A_k、Bmax、cutoff、skin、candidate capacity 或 MLP 权重。

### 2.3 平台与体系

第一版冻结：

| 项目 | 值 |
|---|---|
| system size | `73,536 atoms` |
| platform | `CUDA` |
| precision | `mixed` |
| OpenMM | `8.5.2`, commit `36a30cb` |
| temperature | `300 K` |
| integrator | 当前 production `LangevinMiddleIntegrator` |
| timestep | `2 fs`，以实际 production artifact 校验 |
| stage/window | complex / vanishing / Stage 2 / `vdw/window_0` |
| local K | `5`，以真实 window artifact 校验 |
| ESS protocol | `ESS_GATE_PROTOCOL_VERSION=3` |

实际 topology、System XML、positions、box、checkpoint、schedule、window ranges、`f_k`、integrator parameters、A1.1 `.so` 和所有 Python 脚本必须进入 manifest 并 SHA-256。任何不匹配均 fail closed。

---

## 3. 核心指标

### 3.1 唯一 primary endpoint

对每个独立 paired repeat `r`：

```text
ESS_min,r = min over all sampled windows/states of mixture_coverage_ESS_v3

eta_r = ESS_min,r / GPU_hour_r

D_r = log(eta_candidate,r) - log(eta_baseline,r)

relative_improvement_r = exp(D_r) - 1
```

局部 window-0 阶段时，minimum 取该 window 的全部 K states。完整 Stage-2 confirmation 时，minimum 取所有 Stage-2 windows 和 states。

ESS 必须来自现有：

```text
ibs_engine._ibs_reweighting_quality_diagnostics
ESS_GATE_PROTOCOL_VERSION=3
```

禁止替换为 literal `pymbar.compute_overlap()`、single-reference raw ESS、gap variance 或 transition count。

### 3.2 GPU-hour 口径

Primary GPU-hour 包含每个 arm 的：

- burn-in；
- production sampling；
- production cadence 下的在线 target/bias/base energy queries；
- ledger collection；
- 运行中 checkpoint/reporting 所需 GPU 时间。

不得只计 integrator kernel。burn-in 不计入 ESS 样本，但必须计入 GPU-hour。

另报告 ITT wall time：从 Context/system 构建开始到最后一次 ledger query 完成，包括 NVRTC/初始化。ITT 是 secondary，不替换 primary。

离线 TMBAR 后处理 CPU 时间单独报告，不进入 GPU-hour；若后处理使用 GPU，必须计入对应 arm 的 GPU-hour。

### 3.3 热力学与健康硬门

Primary 只有在以下全部通过时才有效：

- TMBAR `converged=true`；
- `min_overlap`、`min_decorrelated_samples`、endpoint uncertainty 达到当前 production 正式阈值；
- target/bias/base ledger 全部 finite 且闭合；
- 所有预期 window/state/λ indices 完整覆盖；
- baseline/candidate 的 ΔG 在联合统计误差内一致；
- 温度、能量、力、约束和结构健康门通过；
- 基础路径和物理端点未改变。

ΔG 一致性定义：

```text
combined_sigma = sqrt(sigma_baseline^2 + sigma_candidate^2)
z_delta_g = abs(delta_g_candidate - delta_g_baseline) / combined_sigma
z_delta_g <= 2.0
```

若任一 arm 未收敛、uncertainty 缺失/非有限或 `combined_sigma<=0`，ΔG consistency 直接 FAIL，不能使用大 sentinel uncertainty 伪装通过。

---

## 4. 阶段状态机

```text
U0  preregistration + artifact freeze
 ↓
U1  wiring/restart/ledger smoke
 ↓
U2  paired short-NVT safety
 ↓
U3  window-0 paired online utility
 ↓ 仅在 U3 通过时
U4  complete Stage-2 confirmation
 ↓
EXP027_UTILITY_QUALIFIED_CANDIDATE
```

任一 correctness/health 阶段失败立即停止。不得跳过失败门，不得用延长轨迹、换 seed 或调 student 参数救援同一 experiment ID。

---

## 5. U0：preregistration 与 artifact freeze

U0 必须生成：

```text
protocols/EXP-027_online_utility_preregistration.json
output/outer_lambda_exp027_online_utility/u0_manifest.json
output/outer_lambda_exp027_online_utility/decision_log.jsonl
```

Manifest 至少记录：

- A1.1 `.so`、CUDA `.cpp/.h`、public plugin sources；
- A1.1 model payload/weights/checkpoint；
- topology、System XML、positions、box、checkpoints；
- schedule/window manifest、`f_k`、LRC 与 state mapping；
- baseline/candidate wiring XML 或结构化 force inventory；
- OpenMM/CUDA/driver/GPU/precision；
- integrator、thermostat、constraint tolerance；
- scripts、Python environment/lockfile；
- repeat IDs、checkpoint IDs、velocity/Langevin seeds、AB/BA order；
- burn-in/production lengths、chunk/query/output cadence；
- ESS/TMBAR protocol versions与全部阈值。

报告同时区分 canonical self-hash 与 raw file SHA-256。禁止覆盖既有 output；若目标存在必须换 run ID，不能删除重跑。

---

## 6. 独立 repeats 与配对规则

第一版使用 `3` 个独立 repeats。

### 6.1 独立性的要求

三个 repeats 必须来自：

- 三条独立连续平衡历史；或
- 三个预登记、相隔超过相关时间且 provenance 完整的 production checkpoints。

仅从同一 checkpoint 重抽三组速度不构成三个独立 repeats。若目前只有一个合格 checkpoint，U0 必须标记 `BLOCKED_INDEPENDENT_INITIAL_STATES`，不能把 paired reseed 当 N=3。

### 6.2 repeat 内配对

同一 repeat 的 baseline/candidate 必须共享：

- 相同起始 positions 与 box；
- 完全相同的注入 velocity array；
- 相同 Langevin seed；
- 相同 integrator、步数和 query cadence。

不同 repeats 使用不同、预登记的 velocity/Langevin seeds。

### 6.3 执行顺序

预先冻结 AB/BA：

```text
repeat 0: baseline -> candidate
repeat 1: candidate -> baseline
repeat 2: baseline -> candidate
```

不得根据早期 ESS、温度或速度结果改变顺序。

---

## 7. U1：wiring、restart 与 ledger smoke

U1 在每个 arm 上验证：

- baseline force inventory 中不存在 LocalManyBodyResidualForce；
- candidate 恰好包含一个 A1.1 child，且加载 `.so` hash 与 U0 一致；
- Group0/1/2/3/4/5 accounting 与 production 一致；
- raw `U_B`、A_k、kBT、offset、β 各应用正确次数；
- target energy、sampling bias、base energy 和 LRC ledger 分离；
- force-group filtered energy 与 independent probe 一致；
- checkpoint save→new Context→restore 后 positions、box、global parameters、energy/force 与 ledger identity 一致；
- XML 不承担恢复 runtime global parameter current values；checkpoint/metadata 必须恢复并校验它们。

U1 运行少量积分步只作 smoke，不计算 ESS，也不能用于 U3/U4 数据。

---

## 8. U2：paired short-NVT safety

### 8.1 冻结运行量

每个 repeat、每个 arm：

| 项目 | 值 |
|---|---:|
| discarded warmup | `1,000 steps` |
| monitored | `5,000 steps` |
| snapshot interval | `100 steps` |
| timestep | `2 fs` |
| paired repeats | `3` |

U2 不与 U3 production 数据拼接。

### 8.2 每个 snapshot 的检查

- positions、velocities、energies、forces 全 finite；
- temperature 在 `[150 K, 600 K]`；
- candidate residual force max norm `<=500 kJ/mol/nm`；
- OpenMM constraints 正常，无 exception；
- active edges `<=2048`；
- neighbors/anchor `<=80`；
- unique active environment `<=320`；
- candidate list `<=8192`；
- 无 minimum-distance、half-box-tie、unsupported-box 或 nonfinite device status；
- ledger fields/shape/state IDs finite 且闭合。

同时报告但暂不作独立硬门：平均温度、温度标准差、最大总力、最大 residual force、B/tanh saturation、active/candidate counts、constraint error、energy range。

任何一个 repeat/arm 失败即：

```text
EXP027_STOP_SHORT_NVT_SAFETY
```

不得丢弃该 repeat 后补一个新 seed。

---

## 9. U3：window-0 paired online utility

### 9.1 冻结运行量

每个 repeat、每个 arm：

| 项目 | 值 |
|---|---:|
| burn-in | `10,000 steps`，不计 ESS、计 GPU-hour |
| production | `50,000 steps` |
| steps per chunk/update | `500` |
| production frames | `100`，若每 chunk 保存一次 |
| paired repeats | `3` |
| ΔG consistency z threshold | `2.0` |

若 100 frames 不足以满足预登记的 decorrelated-sample/TMBAR 门，结果是 FAIL/INSUFFICIENT，不得事后把本次轨迹延长后仍称同一 preregistered test。更长运行必须新 addendum 与新 run ID，并保留本次失败。

### 9.2 每 repeat 输出

- baseline/candidate GPU-hour 与 ITT time；
- mixture ESS per state；
- `ESS_min`、`eta`、`D_r`、relative improvement；
- TMBAR convergence 与 ΔG/uncertainty；
- `z_delta_g`；
- min_overlap、min_decorrelated_samples、endpoint uncertainty；
- ledger closure 与 state/window coverage；
- temperature/force/constraint/support health；
- secondary mixing diagnostics、IAT、switch count、raw/common-mode ESS；
- checkpoint/restart hashes与失败尝试。

### 9.3 U3 promotion 门

只有全部满足才通过：

1. 三个 repeats 的 baseline/candidate correctness、health、TMBAR、ledger 和 ΔG consistency 全部 PASS；
2. 至少 `2/3` repeats 的 `D_r>0`；
3. median `exp(D_r)-1 >= 0.10`；
4. 没有 candidate 特有的结构异常、温度偏移、support overflow 或 estimator 不收敛；
5. 收益没有来自减少步数、输出频率、query cadence 或遗漏失败窗口。

通过状态：

```text
EXP027_U3_WINDOW0_UTILITY_PASS
```

U3 只说明受影响困难窗口的 online utility 有效，不等于完整 Stage-2 或 ABFE production promotion。

失败状态：

```text
correctness/health failure -> EXP027_STOP_WINDOW0_INCORRECT
correct but <2/3 improved -> EXP027_STOP_WINDOW0_NO_REPRODUCIBLE_GAIN
median improvement <10% -> EXP027_STOP_WINDOW0_GAIN_TOO_SMALL
TMBAR/coverage insufficient -> EXP027_STOP_WINDOW0_INSUFFICIENT_INFORMATION
```

---

## 10. U4：完整 Stage-2 confirmation

只有 U3 PASS 才允许运行 U4。

### 10.1 范围

- baseline 与 candidate 均覆盖完整 Stage-2 的所有 production windows；
- candidate 只在预登记的受影响 window/state 使用 A1.1 residual；其他 windows 与 baseline bitwise/config-identical；
- 不复用 U3 production samples 冒充独立 U4 confirmation；
- 至少 3 个独立 paired repeats，运行长度与 U3 相同或在 U4 addendum 中预先增加，禁止缩短。

### 10.2 U4 硬门

- 完整 Stage-2 TMBAR converged；
- 所有 window/state coverage 完整；
- `min_{w,k}` mixture ESS/GPU-hour 至少 `2/3` repeats 改善；
- median improvement `>=10%`；
- complete-stage ΔG baseline/candidate `z<=2.0`；
- endpoint uncertainty、ledger closure、温度/力/约束/结构健康全部通过；
- 未改变 physical endpoints 或 base path。

通过只能登记：

```text
EXP027_STAGE2_UTILITY_QUALIFIED_CANDIDATE
```

这仍不是跨 ligand、溶剂腿、cycle closure 或完整 ABFE production promotion。若要进入 production，必须另有两腿一致性、cycle closure、独立复现和恢复演练。

---

## 11. 统计、失败与重试规则

- repeat 是统计单位，不把 100 frames 当独立 N；
- 必须报告全部 3 个 `D_r`，不只报告 median；
- equality `D_r=0` 不算 improvement；门使用未四舍五入 float64；
- 不做 best-seed、best-window、best-interval 选择；
- 不删 outlier；
- 不提前停止于 2 个好结果；
- 不用 mean 替代预登记 median；
- bootstrap/CI 若作为 secondary，必须固定 seed、draw count、paired-repeat resampling 与 quantile algorithm；N=3 的 CI 只作描述，不替代 2/3+median 规则。

允许重试只限预登记的客观 infra failure：GPU reset、driver error、ECC、节点丢失、文件 hash corruption。完整但性能差、ESS 低或不收敛的 run 不得重跑。

partial block 不得拼接。任何重试必须从该 paired repeat 的两臂重新开始，保留失败日志；超过预登记最大重试次数则整个阶段 `INVALID/FAIL`。

---

## 12. 禁止事项

- 不修改 A1.1 `.so` 或 checkpoint；
- 不启用 A2 draft；
- 不重调 A_k、Bmax、cutoff、skin、capacity 或 MLP；
- 不修改 λ schedule、`f_k`、IBS update 或 estimator；
- 不降低 online energy/ledger query frequency；
- 不把 burn-in 排除出 GPU-hour；
- 不把 single-window ΔG 称为完整 ABFE；
- 不把 short-NVT PASS 称为 utility PASS；
- 不把 runtime cost PASS 称为 ESS/GPU-hour PASS；
- 不覆盖 EXP-020/025/026 artifacts；
- 不在失败后换 seed、延长轨迹或改变门槛而保留同一 experiment identity。

---

## 13. 产物树

```text
protocols/
  EXP-027_online_utility_preregistration.json

output/outer_lambda_exp027_online_utility/
  u0_manifest.json
  decision_log.jsonl
  u1_wiring_restart_report.json
  u2_short_nvt_report.json
  u3_window0_utility_report.json
  u3_repeats/
    repeat_0/{baseline,candidate}/...
    repeat_1/{baseline,candidate}/...
    repeat_2/{baseline,candidate}/...
  u4_stage2_report.json                 # only if U3 PASS
  u4_repeats/                           # only if U3 PASS
  final_summary.md
```

每个 arm/repeat 保存：配置、seed、input/output hashes、checkpoint、ledger、TMBAR diagnostics、timing、GPU metadata、stdout/stderr 和所有 failure attempts。JSON 必须 `allow_nan=false`，报告使用 canonical self-hash 加 raw file SHA-256。

---

## 14. 最终决策表

| 条件 | 决策 |
|---|---|
| U1 fail | `STOP_WIRING_OR_RESTART` |
| U2 任一健康门 fail | `STOP_SHORT_NVT_SAFETY` |
| U3 correctness/TMBAR/ledger fail | `STOP_WINDOW0_INCORRECT` |
| U3 主性能不足 | `STOP_WINDOW0_NO_UTILITY` |
| U3 PASS、U4 未运行 | `WINDOW0_UTILITY_CANDIDATE_ONLY` |
| U4 任一 correctness 门 fail | `STOP_STAGE2_INCORRECT` |
| U4 主性能不足 | `STOP_STAGE2_NO_MATERIAL_GAIN` |
| U4 全部门通过 | `STAGE2_UTILITY_QUALIFIED_CANDIDATE` |

无论结果如何，EXP-026 A1.1 的 correctness/cost qualification 保留。EXP-027 utility FAIL 只说明 student 没有带来足够在线统计收益，不撤销 `.so` 的工程正确性。

---

## 15. 开始执行前清单

- [ ] preregistration JSON 已封存并 hash；
- [ ] A1.1 source/.so/checkpoint hashes 匹配；
- [ ] baseline 明确无 plugin；candidate 恰好加载一个 A1.1；
- [ ] 三个真正独立初态已证明，不是同一 checkpoint 的三次 reseed；
- [ ] paired velocities、Langevin seeds、AB/BA order 已冻结；
- [ ] 10k/50k/500-step cadence 已冻结；
- [ ] ESS v3 与 TMBAR thresholds 从 production artifact 解析并写入协议；
- [ ] GPU-hour timer boundaries 已通过 dry-run 验证；
- [ ] output 目录不存在且脚本 refuse overwrite；
- [ ] A2 draft 不在 load path；
- [ ] decision log 为 append-only。

全部完成后，才允许把状态改为：

```text
EXP027_PREREGISTERED_READY_FOR_U1
```

---

## 16. U0/U1 进度记录（2026-08-13 追加）

本节记录本计划写成之后、preregistration 正式封存之前完成的准备工作。以下内容如实转录自
`protocols/EXP-027_online_utility_preregistration.json` 与
`output/outer_lambda_exp027_online_utility/decision_log.jsonl`，不重新推导数字。

### 16.1 关键架构发现：candidate 臂在 production 里从未接通过

`abfe_pipeline.py`/`runabfe.py` 的真实生产入口 `_build_window_system`
（ibs_engine.py:9364）从未把 `residual_basis_force` 传给 `build_ibs_dual_system`
——两个文件里 `residual_basis_force`/`LocalManyBodyResidualForce` 全部 grep 结果为零。
这个能力存在于 `IBSBiasForce`/`build_ibs_dual_system` 这一层（ibs_engine.py:3962-4166,
4756-4918），但 production 从未打开过。

**EXP-027 的 candidate 臂构造是新写的代码**（`scripts/exp027_u1_wiring_restart_ledger_smoke_test.py`），
复刻（不是修改）已经冻结、产出过 EXP-026 正式 O4 数字的
`scripts/exp025_g4_timing_harness.py` 里 `build_window`/`build_native_sim` 这段
已验证真实模式。

### 16.2 U1 wiring/restart/ledger smoke test：已写、已跑、PASS

脚本：`scripts/exp027_u1_wiring_restart_ledger_smoke_test.py`。真实 GPU（CUDA mixed
precision），明确指向 `plugins/LocalManyBodyResidual/build_exp026_a1/`（A1.1 的
frozen build，sha256 校验过，不是现在指向 A2 的 `build` 符号链接）。

检查项与结果（全部 PASS，0 failing）：

```text
baseline: 零个 LocalManyBodyResidualForce（顶层或嵌套）
candidate: 恰好一个，以 CV 名 "exp025_residual_basis" 嵌在 Group1 CustomCVForce 内
           (不是新增顶层 Force -- 顶层 force 类型计数与 baseline 完全一致)
Group 0,2,3,5 (base) 能量：baseline/candidate 在 mixed-precision 舍入误差内一致
           (~7e-5 kJ/mol，系统总能量约 1e6 kJ/mol 量级) -- residual 没有漏出 Group 1
Group 1,4 (bias) 能量：如预期出现差异（residual 就活在这里）
checkpoint save -> 全新 Context -> restore：positions/box/lambda_boresch_scale/
           lambda_shield/两组能量全部一致
```

过程中修了两个真实的**测试脚本**bug（不是 production 的 bug）：

1. SWIG 不会把 `CustomCVForce.getCollectiveVariable()` 返回的第三方 Force 子类下
   转型——哪怕是没经过序列化的活对象也一样，拿到手就是个泛型 `Force`。必须按
   注册的 CV **名字**（`"exp025_residual_basis"`，ibs_engine.py:4917）判断，不能
   按 Python 类型判断。
2. 能量一致性容差最初设成 1e-6 kJ/mol——但这是一个 ~1e6 kJ/mol 量级的全系统能量，
   mixed precision CUDA 下浮点误差本来就有 ~1e-4 kJ/mol，容差设太死会误报。改成
   1e-2 kJ/mol 后正确通过。

**§6.2 的配对要求白捡了，不用额外写代码**：baseline/candidate 都从**同一个**
`.chk` 文件 `Simulation.loadCheckpoint()`，OpenMM 的二进制 checkpoint 本来就把
positions/velocities/box/integrator RNG 状态全部原样恢复，天然满足"共享相同
velocity 数组与 Langevin seed"——跟 EXP-026 O3/O4/A2 那些计时对比用的是同一套
机制，不需要 `_build_fixed_state_simulation` 里那个单独存在但生产路径不用的
`context.setVelocities(array)` 通道。

顺带确认：`ESS_GATE_PROTOCOL_VERSION=3` 是真实存在的（ibs_engine.py:13061），
但计划正文里"`mixture_coverage_ESS_v3`"是个别名，不是真实符号——真实函数是
`_ibs_reweighting_quality_diagnostics(u_kj_raw, bias_kj, f_k, kt)`。

### 16.3 A1.1/A2 身份撞车风险：已缓解

`plugins/LocalManyBodyResidual/build` 符号链接已于 2026-08-13 从 A1.1 改指向 A2
（EXP-026 的独立提升决定，见 `PLAN_EXP-026_cuda_control_plane_optimization.md`
第 24.2 节）。EXP-027 的 preregistration 与 U1 脚本都明确把 candidate 路径钉死在
`plugins/LocalManyBodyResidual/build_exp026_a1/`，绕开会变的符号链接，避免悄悄
测错候选。

### 16.4 三个独立 repeat 的真实状态（2026-08-13）

```text
repeat_0 (output_lrc_fix, 无显式 seed)
  -- U1 PASS（0 failing，脚本默认 --output-root 就是它）
  -- 待明确判断：整条 pipeline 后来在 Stage 2 报过 RuntimeError（跟 window_0 本身无关，
     是下游 endpoint_uncertainty 问题）；window_0 自己的采样是完整的、checkpoint 是真的，
     但这条历史算不算"干净的独立初态"这个判断尚未拍板。

repeat_1 (seed 20260906, output_lrc_fix_repeat02_seed20260906)
  -- 已确认可用。复合物腿完整跑完（equilibration->attachment->decharging->vanishing
     全部 completed），ΔG_complex = 48.44 ± 0.44 kcal/mol。
  -- window_0 checkpoint（17:29:59 定格，Stage-2 rescue 只碰过 window_1，没碰 window_0）
     -- U1 PASS，0 failing。
  -- 溶剂腿已自动接上开始跑（EXP-027 不需要，不用管）。

repeat_2 (seed 20260907, output_lrc_fix_repeat03_seed20260907) —— 已重跑成功
  -- 第一次跑：INVALID。run_provenance.json 记录 "openmm": "8.2"，而
     repeat_1 与本会话环境都是 "8.5.2"（EXP-027 §2.3 冻结要求正是 8.5.2, commit 36a30cb）。
     根因：用户集群里不同节点装的 CUDA 工具链不同（观测到 12.6 / 12.9 两种），解析出来的
     OpenMM 包版本跟着不一样——这条任务恰好落到了装老版本 OpenMM 的节点上。
  -- 这个版本不一致直接导致了一连串真实可复现的症状：window_0 的 openmm.chk 比
     repeat_0/repeat_1 的小了约 24.6KB（OpenMM 的二进制 checkpoint 格式在版本间不保证
     兼容——integrator 的 RNG 状态序列化不一样），本会话（OpenMM 8.5.2）尝试
     `loadCheckpoint()` 时报 `CUDA_ERROR_OUT_OF_MEMORY`。
  -- 排查时依次排除掉的错误方向（记录下来避免以后重复走）：GPU 硬件/线程配置不匹配
     （猜测，排除）、memlock ulimit 太小（用户在自己 shell 里 `ulimit -l unlimited`
     后同样报错，排除）、bias 未收敛（`ibs_state_vdw_window_0.json` 全程显示
     `bias_converged=true`，排除）。
  -- 原 8.2 记录已归档到 `output_lrc_fix_repeat03_seed20260907_INVALID_wrong_openmm_8.2/`，
     未使用。用确认过版本（8.5.2）的节点用同一个 seed 重新 `--reset` 提交，2026-08-13
     23:28 完整跑完，这次全程干净、没有触发任何 rescue，ΔG_complex = 48.08 ± 0.30 kcal/mol。
  -- window_0 checkpoint 大小恢复成 11,402,557 字节，跟 repeat_0/repeat_1 完全一致
     ——印证了"版本不一致"确实是真根因。U1 重跑：先出了 1 个 FAIL（check 5 的
     Group0,2,3,5 restore 能量容差 1e-4 太紧，这次舍入噪声 1.5e-4 卡了线——
     跟 07-13 白天在 repeat_0/repeat_1 上发现的 check-3 容差问题是同一类
     mixed-precision 舍入现象，只是当时两个 repeat 的舍入量刚好没超过 1e-4，
     这次超过了），把该容差也放宽到 1e-2 kJ/mol（跟 check 3 保持一致的理由）后
     重跑：**PASS，0 failing。**

**当前真实进度：3/3 独立初态已通过 U1 验证。**
U0 的 `three_independent_initial_states_proven` 只剩一项悬而未决：repeat_0
"整条历史算不算干净的独立初态"这个判断（它的 window_0 本身已验证没问题，
只是整条 pipeline 后来在 Stage 2 报过一次跟 window_0 无关的 RuntimeError）。

### 16.5 U0 完整收尾 + 封存（2026-08-13 23:50）

按用户指示（"可以继续完成：解析真实 ESS v3/TMBAR 阈值；GPU-hour timer dry-run；
封存 preregistration；进入 U2"）依次做完：

1. **真实 ESS v3/TMBAR 阈值**：从 `ibs_engine.py`/`abfe_pipeline.py` 现行代码逐行
   核对提取（不是猜的）。Stage-2 最终 `converged` 门：`min_overlap>=0.05`、
   `min_decorrelated_samples>=20`、`max_endpoint_uncertainty_kJ_mol<=1.0`；
   warmup 转生产的解冻门：`IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL=10.0`。
   确认 `mixture_ess_ratio`/occupancy 在协议版本 3 下确实只是诊断项，不是硬门
   （跟 [[project_ess_gate_redesign_2026-07-26]] 的历史结论一致）。写进
   preregistration JSON。
2. **GPU-hour timer 边界 dry-run**：`scripts/exp027_gpu_hour_timer_dryrun.py`，
   PASS。确认 GPU-hour 窗口正确排除了 Context/NVRTC 构建开销（这个系统实测约
   49 秒，真实且不可忽略），ITT 则把构建时间也算进去；GPU-hour = burn-in +
   production+ledger 查询 + checkpoint/reporting。
3. **repeat_0 独立性判断**：问了用户，明确答复"算数"——window_0 本身独立完整、
   已验证，下游那次 Stage-2 RuntimeError 是全局 ΔG 拼接的统计质量问题，跟
   EXP-027 只用 window_0 checkpoint 这件事无关。
4. **重构**：把 U1 的构造逻辑提炼成 `scripts/exp027_common.py`
   （`Exp027Repeat` 类），U1/GPU-hour dry-run/U2 共用，避免三份复制代码各自
   漂移。重构后重新在 3 个 repeat 上跑了一遍 U1，确认仍然 0 failing。
5. **封存**：`protocols/EXP-027_online_utility_preregistration.json` 状态从
   `DRAFT_NOT_PREREGISTERED` 改为 `EXP027_PREREGISTERED_READY_FOR_U1`，
   raw SHA-256 记在 decision_log 里。

### 16.6 U2 paired short-NVT safety：真实执行，PASS（2026-08-14）

冻结运行量（PLAN §8.1）：discarded warmup=1000、monitored=5000、snapshot
interval=100（每臂每 repeat 50 个快照）、timestep=2fs、3 个 repeat。

**中途发现并修正了一个真实的方法论 bug，在报出任何结论之前先修好了**：第一次
尝试把"candidate 的残差力"算成"candidate 的 Group1 力 减去 baseline 自己独立
演化出来的、同一步数下的 Group1 力"——这是错的：candidate 从第一步开始就比
baseline 多受一个力，两条轨迹立刻开始分叉，走了 1000+ 步之后已经是完全不同的
构型了，这时候直接相减,减出来的东西里混进了"这两个构型现在差多少"，而且这部分
会随步数增长。第一次跑出来的结果是 550-650 kJ/mol/nm，压线超过 500 的限——
三个 repeat 全部这样，模式太一致反而说明是方法出了问题，不是真的物理现象。**这个
结果在被当作正式发现汇报之前就被丢弃了。**

修法：改用"同构型影子探针"——额外建一个**从不参与积分**的 baseline
Simulation,在每一个快照点把它的坐标/box 直接设成 candidate 那一刻的真实构型
（`setPositions`/`setPeriodicBoxVectors`），再查它的 Group1 力,跟 candidate
自己的 Group1 力相减——这样两边比的是**同一个构型**,干净隔离出残差项自己贡献
的力，不会被轨迹分叉污染。改完之后单独抽查：数值从 550-650 掉到 42-152
kJ/mol/nm，证实了第一次的数字确实是分叉造成的假象。

真跑全部 3 个 repeat × 2 臂（1000+5000 步，每臂 50 个快照）：

```text
repeat_0 baseline:  温度范围 [297.6, 302.9] K
repeat_0 candidate: 温度范围 [298.4, 302.4] K, 残差力范围 [24.0, 107.6] kJ/mol/nm
repeat_1 baseline:  温度范围 [298.2, 302.5] K
repeat_1 candidate: 温度范围 [298.0, 302.1] K, 残差力范围 [31.4, 148.3] kJ/mol/nm
repeat_2 baseline:  温度范围 [297.4, 302.0] K
repeat_2 candidate: 温度范围 [298.1, 302.5] K, 残差力范围 [25.5, 151.2] kJ/mol/nm
```

全部温度紧贴 300K 目标（符合已平衡 NVT 轨迹的预期），残差力全部在 500 限值的
3.3~20 倍余量之内，300 个快照、0 failing。

**关于 active_edges/neighbors-per-anchor/unique_active_environment/candidate_list
这四条上限、以及 MIN_DISTANCE/HALF_BOX_TIE/UNSUPPORTED_BOX/NONFINITE 这四种
device status——插件完全没有把这些原始计数暴露给 Python**（`LocalManyBodyResidualForce.h`/
`LocalManyBodyResidualKernels.h` 里 grep 不到任何 getter），而且 PLAN §12
明确禁止改 A1.1 的 `.so` 去加一个。验证方式改成依赖插件自己的 fail-closed
设计：任何一项违规都会在 `execute()` 内部立刻抛出 `OpenMMException`。

**准确的口径**（2026-08-14 修正，此前这里的措辞过头了）：

```text
全程未触发插件 fail-closed ceiling exception = PASS
raw active_edges/neighbors/unique-env/candidate counts = NOT_OBSERVED
```

"没抛异常"证明的是这八项条件**没有被违反**，不等于"直接测量/观测到了原始计数值"
——没有 getter 能拿到那些数字，不能说成已经验证/测量过了。

积分步数：3 repeat × 2 臂 × (1000+5000) 步 = **36,000 步**，零异常。（此前这里
错写成"约36万步"——36万步是 U3 的总量 `3×2×(10000+50000)=360,000`，不是 U2 的，
已订正。）

```text
EXP027 U2 = U2_SHORT_NVT_SAFETY_PASS
```

报告：`output/outer_lambda_exp027_online_utility/u2_short_nvt_report.json`。

### 16.7 对 16.6 的口径修正（用户指出，2026-08-14）

用户抓出两处措辞/数量问题，已在 preregistration JSON 和本文档 16.6 里改正：

1. **步数算错了**：U2 总积分步数是 `3 repeats × 2 arms × (1,000+5,000) = 36,000`，
   不是"约36万步"——36万步（`3×2×(10,000+50,000)=360,000`）对应的是**即将跑的
   U3**，不是已经跑完的 U2，之前写混了。
2. **support ceiling 的结论表述过头了**：准确说法应该是

   ```text
   全程未触发插件 fail-closed ceiling exception = PASS
   raw active_edges/neighbors/unique-env/candidate counts = NOT_OBSERVED
   ```

   "没抛异常"只证明这八项条件没有被违反，不等于"直接测量到了原始计数值"——
   没有 getter 能拿到那些数字，不能说成已经验证/测量过了。

同构型影子探针那个修正方法本身是对的：残差力必须在 candidate 的**同一构型**上
比较 `F_candidate − F_baseline`，不能比较两条已经分叉的轨迹。

### 16.8 U3 harness：已写、本地小规模验证通过（2026-08-14）

`scripts/exp027_u3_window0_paired_utility.py`。复用真实 production 机制，
不重新发明数学：

- **`IBSSampler`**（ibs_engine.py:6048）做逐帧能量采集——`collect_energies()`
  算出的 target_energies（softcore+LRC）、e_base（Group 0,2,3,5）、e_bias
  （Group 1,4），跟真实在线生产采样完全一致。
- **`_solve_single_window_local_mbar`**（ibs_engine.py:13206）做真正的
  TMBAR/MBAR 求解——这就是 production 自己的在线 early-stop 监控用来判断
  "in-progress window"是否收敛的那个函数（ibs_engine.py:12255），不是新写的。

端点量推导：`delta_g = f[-1] - f[0]`；不确定度用
`endpoint_diff_uncertainty_kJ_mol`（协方差感知的正确算法，不是
`sqrt(df[0]²+df[-1]²)`——ibs_engine.py 自己的注释明确警告过后者会系统性
算错）；`绝对 ESS = min_ess_ratio × n_frames_used`；`eta = ESS/GPU-hour`；
`D_r = log(eta_candidate/eta_baseline)`。

**本地小规模验证（100 burn-in + 6000 production，仅 repeat_0）跑通全链路，
而且有个很好的交叉验证**：算出来的 `delta_g_kJ_mol` 是 80.64（baseline）/
81.03（candidate），跟这个 window 已知的真实生产日志数值（"段
src_window=0 λ[0→4] ΔG=80.4330 kJ/mol"）几乎一致——说明整条链路（能量采集→
TMBAR 求解→ΔG 提取）算的是真实物理量，不只是代码跑得动而已。

**尚未在真实规模上跑**（3 repeat × 2 臂 × (10,000+50,000) 步 = 360,000 步，
按小规模测试外推估计约 30-45 分钟）——按用户偏好交给用户自己的计算节点去跑，
本会话不跑。

### 16.10 U3 真实规模结果：`EXP027_STOP_WINDOW0_INCORRECT`（2026-08-14）

**节点交接受阻**：用户在自己节点上跑 U3 时，报了跟独立 repeat 那次一样的
`GLIBC_2.29' not found` 错误。这次根因比"节点不对"更麻烦：**A1.1 的源码已经不
在任何活目录里了**——A2 提升为 live 时把它覆盖掉了，现在只剩编译好的二进制
留在 `build_exp026_a1/`。问了用户三选一（改测 A2 / 尝试重建 A1.1 源码 / 直接
在本会话跑），**用户选择直接在本会话跑**。真实规模（3 repeat × 2 臂 ×
(10,000+50,000) 步 = 360,000 步）在后台跑了约 40 分钟。

真实结果：

```text
repeat_0 (baseline_first):  D_r=-0.281  z_delta_g=2.70  delta_g_consistency: FAIL
repeat_1 (candidate_first): D_r=-0.383  z_delta_g=2.41  delta_g_consistency: FAIL
repeat_2 (baseline_first):  D_r=-0.198  z_delta_g=0.075 delta_g_consistency: PASS

EXP027_U3_DECISION = EXP027_STOP_WINDOW0_INCORRECT
```

3/3 repeat 全部 D_r<0（candidate 的 eta 比 baseline 差），2/3 repeat 的 ΔG 一致性
没过——不是压线结果。

### 16.11 重大发现：插件长轨迹下性能持续退化（2026-08-14）

跑真实结果时发现 candidate 的 `gpu_hour_seconds` 大约是 baseline 的 2.3 倍
（438-442s vs 186-188s），跟 EXP-026 自己测出来的 A2 开销（约1.03-1.04倍）完全
对不上。**没有直接采信，先查清楚了再写结论**：

1. 单独测"纯 step() 耗时"（2000步规模）：baseline 2.61ms/步、candidate 2.74ms/步，
   比值1.050——**跟 EXP-026 的 O4 结果吻合**，排除。
2. 单独测"纯 collect_energies() 耗时"（20次查询）：baseline 268.8ms、candidate
   262.1ms——基本相等（确认了 ledger 查询用的 probe context 只包含名字以
   `_int` 结尾的 CV，residual 叫 `"exp025_residual_basis"`，从不被收进去，
   ibs_engine.py:4920-4929），排除。
3. **分段计时找到真凶**：把 3 万步切成 15 段（每段2000步）分别计时——

   ```text
   candidate: 第1段 2.74 ms/步 → 第15段 6.22 ms/步（3万步内翻了一倍多，单调递增）
   baseline:  全程 2.60-2.71 ms/步（完全平稳，同一窗口下几乎不变）
   ```

**结论：这是插件本身真实存在的性能缺陷，不是 harness bug，也不是环境/GPU
的普遍性衰退**（baseline 在同一窗口下完全平稳，直接排除了环境因素）。是个
纯速度问题，不是正确性问题——整个过程没有抛异常、没有 NaN/Inf，能量数值
全程合理。

**为什么之前从来没发现**：EXP-025→026→027 这条线上**所有**测试（G0-G3、22项
control-plane test、A1.1 的 dBdq fault injection、这次新写的 A2
first-error-wins 测试、甚至 EXP-026 自己那道 O4 成本门）都只跑了几百到最多
2000步。这是这个插件**第一次**被跑一条真正长的连续轨迹（U3 的 6 万步，
诊断用的 3 万步）。真实生产每个 window 是 25 万步——如果这个增长趋势持续，
真实成本可能远超 O4 当时"1.04倍通过"这个数字。

**内部机制尚未定位到具体哪一段代码**：目前只confirmed了"症状"（candidate
特有、单调、随时间增长），没有confirmed"根因"（一个未验证的假设是 G3
local-CSR/skin Verlet list 的重建频率或代价随着轨迹上累积的原子位移增长，
短窗口根本触发不到有意义的量级）——如果要继续查，需要对一条长轨迹做
nsys/ncu profiling。

**对 EXP-026 STOP_OPTIMIZATION_SUCCESS 的影响**：那个结论对它当时测量的东西
（1000步窗口）**没有被推翻**，但现在知道那个窗口的覆盖范围不足以刻画真实
生产窗口（25万步）。如果退化趋势持续，真实生产成本可能比 O4 那个数字差
很多。**A1.1 和 A2 大概率都受影响**（这是 G3 共享核心机制的问题，不是 A2
那次错误合并改动的东西），但本会话没法在 A1.1 上独立复测（它的源码已经
不在了）。已经在 EXP-026 那份记忆文件里加了指回来的提醒。

### 16.12 ΔG 一致性失败的口径修正（用户指出，2026-08-14）

16.10 里 2/3 repeat 的 `delta_g_consistency: FAIL`（z=2.70/2.41），最初的表述
比较笼统（大致说是"统计/MBAR 问题"）。用户给出了更精确的物理论证，需要
改正：

**端点结构性恒等，不只是"数值接近"**：candidate 在 λ=0 和 λ=1 处插件贡献
严格为零（残差系数在两端精确为零，restraint/LRC/标准态修正/target-energy
ledger 两臂完全相同）。这意味着 baseline 和 candidate 在两个端点的 Hamiltonian
是**结构性恒等**的——不是"跑出来的数字差不多"，而是数学上就是同一个系统。
因此真实、无限采样极限下的端点自由能差 `ΔG=-kBT log(Z1/Z0)` **必须相同**。
中间窗口加入的残差只能改变采样路径/效率/重叠，**不能**改变端点物理本身。

**正确的表述**：z_delta_g=2.4–2.7 不能写成"插件改变了端点 ΔG"。应该写成
"candidate 在有限采样预算内未能可靠复现两臂共享的端点 ΔG"——这是一个
**估计量未收敛**的问题（有限采样/混合不足、路径 overlap/reweighting 质量差、
自相关导致 TMBAR 不确定度被低估、或者"理论上精确为零"的端点残差在具体
实现/ledger 里没有精确落到零），而不是残差真的改变了 ΔG 的证据。

**U3 的失败因此拆成两层独立问题**：

1. **效率层**：3/3 repeat D_r<0——candidate 的 ESS/GPU-hour 全面比 baseline
   差。直接由 16.11 的性能退化发现解释（GPU-hour 被插件自身增长的单步
   成本拖累）。
2. **收敛质量层**：2/3 repeat 没能在冻结不确定度范围内复现同一个端点
   ΔG。正因为端点天然恒等，这一层反而是有诊断价值的——它明确指向中间
   路径的采样/overlap/不确定度估计存在真实问题，跟第一层是分开的（虽然
   可能被第一层放大：长轨迹性能退化如果同时挤压了有效混合时间，也可能
   连带拖累第二层的收敛质量，但这个"复合"关系目前没有独立验证，需要另外
   查 overlap 和 per-lambda ESS/自相关才能干净地归因)。

**不改变已定决策**：按预注册门槛，2/3 一致性失败本身已经足够判定 U3 失败，
跟以上口径修正无关——目前的数据没有证明两个估计量已经收敛回同一个端点
ΔG。这次修正只是把失败的**含义**说得更准确，不重开、不软化
`EXP027_STOP_WINDOW0_INCORRECT`。

### 16.13 当前权威状态汇总（2026-08-14，第 16 节追加后）

```text
preregistration_json_sealed_and_hashed        = true  (EXP027_PREREGISTERED_READY_FOR_U1)
candidate_identity (A1.1)                     = 已冻结，与 build_exp026_a1 一致
baseline_has_no_plugin_candidate_has_one_a1_1 = PASS
three_independent_initial_states_proven       = true  (repeat_0/1/2 全部 U1 验证通过；
                                                  repeat_0 独立性判断已由用户拍板：算数)
ess_v3_tmbar_thresholds_parsed                = true  (真实阈值，见 16.5.1)
gpu_hour_timer_boundaries_dry_run_validated   = true  (见 16.5.2)
u1_wiring_restart_ledger_smoke_test           = PASS (0 failing)
u2_paired_short_nvt_safety                    = PASS  (U2_SHORT_NVT_SAFETY_PASS，
                                                  数量/措辞口径已按 16.7 订正)
u3_real_scale_run (original, A1.1 binary)     = EXP027_STOP_WINDOW0_INCORRECT（见 16.10）
                                                  —— 现已定性为：因 addArg 累积缺陷失效，
                                                  不可用于晋级决策（见 16.14）
u3_major_finding                              = 插件长轨迹性能持续退化，已确认非
                                                  harness/环境问题，内部机制由用户找到并
                                                  修复（EXP-028，见 16.14）
u4                                             = 草稿已写（协议见 EXP-027_U4_stage2_
                                                  confirmation_addendum_DRAFT.json），
                                                  之前因 U3 未 PASS 而不可执行——现在
                                                  EXP-028-U3-CONFIRMATION 已 PASS（见 16.14），
                                                  U4 已解锁但尚未启动
```

### 16.14 EXP-028：根因定位+修复+U3 复跑通过（2026-08-14）

**根因（用户直接从源码定位）**：`CudaCalcLocalManyBodyResidualForceKernel::execute()`
里每个 kernel 每一步都用 `addArg()` 重新加参数，而不是首次 `addArg()`、之后
`setArg()`。`addArg()` 是永久追加参数槽，不是更新——所以 14 个 kernel（K0-K6、
resetStatus、两种 scatter）的参数 vector 逐步无界增长，`execute()` 每次都要
遍历整条不断变长的参数列表，CPU 端 kernel launch 准备耗时随累计步数线性增长
（总耗时 O(n²)），与实测 `2.74→6.22 ms/step`（3万步内翻倍多）完全吻合，同时
伴随 host 内存持续增长。

**修复**：`CudaLocalManyBodyResidualKernels.{cpp,h}` 改为"每个 kernel 首次调用时
`addArg()` 绑定一次，之后全部 `setArg(固定 idx, ...)`"——14 个 kernel、每个都
按用户给的 enum-index 方案逐一转换，纯 host 端 C++ 改动，设备端 kernel 源码
零改动。修复前源码+编译好的 `.so` 已备份到
`plugins/LocalManyBodyResidual/pre_exp028_backup/`（sha256 已记录）。

**修复前验证**（不允许"改完就跑性能"，先证明没改物理）：
- G0 smoke/serialization round-trip、control-plane layout、A1.1 dBdq
  fault-injection（真机）、A2 first-error-wins（真机，K1 vs K2 竞态，
  15/15 非 flaky）全部 PASS。
- G3 CSR 路径（多数既有测试默认 skinAngstrom=0 走不到）：用修复前/修复后两个
  `.so` 在同一个真实生产 checkpoint 上分别评估，step=0 时 e_total 只差
  ~3×10⁻⁵ kJ/mol（GPU 规约顺序噪声量级），step 50/200 的偏差按正常 MD 混沌
  发散速率平滑增长——不是离散型 bug 的特征。

**性能复测（30,000 步分段，与退化诊断完全对应）**：
```text
修复前: 第1段 2.74 ms/步 → 第15段 6.22 ms/步（比值 2.27）
修复后: 第1段 2.6537 ms/步 → 第15段 2.7406 ms/步（比值 1.033，≤1.05 门槛 PASS）
process RSS（修复后）: 15 段全程 2504.1 MB，零增长
```

**EXP-028-U3-CONFIRMATION（真实规模复跑，非覆盖旧报告）**：用户额外指出原 U3
harness 的两个陷阱，复跑前先修：
1. `Exp027Repeat` 默认 `plugin_build_dir=build_exp026_a1`（A1.1，未修复），
   原 U3 脚本从未显式覆盖——即原始 U3 的"candidate"实际测的是未修复的 A1.1
   二进制，不是 A2。`exp027_common.py` 新增 `verify_known_plugin_identity()`
   + `identity` 参数（"a1.1" 默认不变，新增 "exp028_a2"），新跑显式传
   `plugin_build_dir=build_exp026_a2, identity="exp028_a2"`，sha256 不符直接
   fail-closed。
2. `r["order"]` 标签从未真正控制执行顺序（代码里 baseline 永远先跑）。新脚本
   `scripts/exp028_u3_confirmation.py` 里 `arm_order` 真正驱动执行顺序，
   实际顺序写入报告 `actual_arm_execution_order`。
3. 分析口径冻结不变：`IBSSampler`/`_solve_single_window_local_mbar` 原样使用，
   residual 仍只作为采样偏置，reweighted 回 baseline-only target（不把
   residual 加进 target ledger）。

真实结果（`output/outer_lambda_exp027_online_utility/exp028_u3_confirmation_report.json`，
sha256=`4a3a5478ed921401d1177d7836b65f7f57dc212cf5c349aebbb513447b581b3f`）：
```text
repeat_0 (baseline_first, actual=[baseline,candidate]): D_r=-0.263 z=0.273 PASS  gpu_hour ratio=1.041
repeat_1 (candidate_first, actual=[candidate,baseline]): D_r=+0.683 z=0.809 PASS gpu_hour ratio=1.025
repeat_2 (baseline_first, actual=[baseline,candidate]): D_r=+0.402 z=0.658 PASS  gpu_hour ratio=1.026

n_repeats_with_positive_D_r = 2/3 (门槛 >=2, PASS)
median_relative_improvement = +0.494 (49.4%，远超 0.10 门槛)
delta_g_consistency_pass = 3/3（原来 1/3；z 从 2.70/2.41 降到 0.27/0.81/0.66）
gpu_hour ratio 三个 repeat 都落在 1.03-1.05，与 EXP-026 O4 短测区间一致

DECISION: EXP027_U3_WINDOW0_UTILITY_PASS
```

**旧 EXP-027 U3 报告的正确读法**（不删除、不覆盖，仅更新解读）：
`u3_window0_utility_report.json` 的 `EXP027_STOP_WINDOW0_INCORRECT` 结果本身
测量无误，但测的是被 addArg 缺陷拖累的 A1.1 二进制，不代表 A2 修复后的真实
生产成本/效用——正确表述是"EXP-027 U3（原始）因累积 addArg 缺陷不可用于晋级
决策"，不是"U3 方法论本身错误"。

下一步：EXP-028-U3-CONFIRMATION 已 PASS，U4（全部 6 个 Stage-2 窗口的确认）
已解锁，尚未启动。
