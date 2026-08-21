# PLAN EXP-026 — Local Many-Body CUDA 控制面优化

## 0. 文档身份

| 字段 | 值 |
|---|---|
| `document_id` | `PLAN_EXP-026_cuda_control_plane_optimization` |
| `experiment_id` | `EXP-026` |
| `status` | `DRAFT_NOT_PREREGISTERED` |
| `date` | `2026-08-13` |
| `parent_experiment` | `EXP-025` |
| `method_identity` | `EXP025_R1_MATH_UNCHANGED__CUDA_CONTROL_PLANE_ONLY` |
| `production_authorization` | `FORBIDDEN` |

本计划建立一个全新的实验身份。它不修改、不覆盖、不重新解释 EXP-025 的封存结论：

> `STOP_EXP025_RUNTIME_BACKEND`

EXP-025 证明了插件数学和 OpenMM 接线正确，但当前实现没有通过运行成本门。EXP-026 只回答一个问题：

> 在完全不改变 R1 数学、模型权重、真实体系、mixed precision、IBS 接线和查询频率的条件下，能否仅通过 CUDA 控制面优化，把 fused Group1 的真实运行成本压入既定门槛？

本文件是实施计划，不是 sealed protocol。执行 O4 正式计时前，必须生成并冻结：

`protocols/EXP-026_control_plane_preregistration.json`

---

## 1. EXP-025 冻结事实

### 1.1 已通过、不得撤销的资格

- G0：OpenMM plugin ABI、Reference/CUDA 加载、NVRTC、XML round-trip、环境 manifest 通过。
- G1：独立 C++ double oracle 与 OpenMM Reference energy/force/finite-difference/no-contact/XML 通过。
- G2：CUDA brute-force correctness 通过。
- G3：本地 linked-cell/CSR/Verlet correctness 通过。
- mixed precision：`posq + posqCorrection` 支持和 G2/G3/mixed nested-CV 回归通过。
- Layer-1 residual oracle、Layer-2 native fused Group1、三方等价检查通过。

这些结论仅说明实现正确，不说明成本合格。

### 1.2 已失败、不得重命名为通过的 G4

真实条件：

- 73,536 atoms；
- CUDA；
- production `mixed` precision；
- 真实 checkpoint、真实 topology/frame；
- native fused Group1；
- 5 个配对重复；
- 每臂 100 warmup + 1000 measured steps；
- EXP-025 normative harness 的 query 口径保持不变：1000-step pure integration，另加 `measured_steps//500` 次 generic energy `getState` 形成 matched-total surrogate。它不是完整 production ledger 的 K+4 group-scoped queries + K probe queries，计划不得把它改写成“完整 production 成本”。

冻结结果：

| 候选 | median ratio | bootstrap one-sided P95 upper | 结论 |
|---|---:|---:|---|
| `native_fused_group1_total` | `1.1233981725063995` | `1.1321719030983386` | FAIL |

硬门：

- median ratio `<= 1.07`；
- bootstrap one-sided P95 upper `<= 1.10`。

两项是 AND 关系，等号通过；任何一项失败即失败。

### 1.3 Postmortem 诊断（仅为工程先验，不是新资格）

| 量 | 观测/估算 |
|---|---:|
| baseline | 约 `2.59 ms/step` |
| 当前总 overhead | 约 `319.6 us/step` |
| profiler 中具名 plugin kernels | 约 `90.8 us/step` |
| 总 overhead 中未由具名 kernels 解释的部分 | 约 `228.8 us/step` |
| 达到 median 1.07 的 overhead 上限 | 约 `181.3 us/step` |
| 需要削减 | 约 `138.3 us/step` |
| plugin launches | 约 `11–14/step` |
| D2H copies | 约 `8/step` |
| H2D copies | 约 `3/step` |
| error/status host sync | 约 `4/step` |
| unique-env flag 往返 | `73,536 × 4 × 2 = 588,288 B/step` |

解释边界：

1. `90.8 us` 是 profiler 中具名 plugin GPU kernels 的总时间，不等于“不可消除的 MLP 算术下界”。其中仍包含控制、检查和候选表相关 kernel。
2. `228.8 us` 是跨运行、跨计时口径形成的工程估算，不能宣称全部已严格证明为 CPU 时间。
3. “削减 138.3 us 后 P95 约为 1.075”只能作为规划推断，不能代替重新测量。
4. 正式决定只使用 O4 配对 wall-clock 数据，不从 profiler 时间中减去任何成分。

---

## 2. 科学目标与工程目标

### 2.1 科学目标不变

EXP-020 held-out gap-variance 改善为 `55.55240525117041%`。理想化地把统计效率看作与 gap variance 成反比：

`1 / (1 - 0.5555240525117041) = 2.249840527...`

这只是统计效率 proxy，不是已验证的 ESS/GPU-hour，也不是 MD steps/s 加速。

### 2.2 EXP-026 工程目标

EXP-026 的直接目标只有一个：

> 让同一个 R1 residual 在同一个 production workload 中，以 median ratio <=1.07 且 P95 upper <=1.10 运行。

EXP-026 不进行 online ESS/GPU-hour 宣称。若 EXP-026 通过，下一实验才允许做 paired online utility qualification。

### 2.3 可证伪假设

`H_EXP026`：EXP-025 的主要超额成本来自可移除的每步 host/device 控制开销，而不是 R1 many-body 数学本身；在冻结数学和 workload 后，设备端计数、同步合并和 launch 路径优化足以同时通过 1.07/1.10 成本门。

否证条件：

- correctness/invariant 任一失败；或
- 最终 median ratio >1.07；或
- 最终 P95 upper >1.10；或
- 只有通过改变数学、查询频率、错误语义或 workload 才能达标。

---

## 3. 冻结项：EXP-026 绝对不能改什么

### 3.1 模型与数学

- EXP-020 冻结 checkpoint 及 SHA-256；
- 41 个 ligand anchor IDs 及顺序；
- 7 atom types、16 RBF、pair weights、7 个 `1→16→16→1` SiLU MLP；
- `q_i = sum_j c2(r_ij) g_type(r_ij)`；
- `h_i = rho_type(q_i) - rho_type(0)`；
- `S = sum_i h_i`，严禁改成 mean；
- `B = 10*tanh(S/10)`；
- `U_B = kBT*B`，kBT 在 plugin 内恰好一次；
- outer `A_k`、offset 在 fused Group1 恰好一次；
- 不重复 beta、A、kBT、tanh 或 offset。

### 3.2 几何与支持域

- OpenMM 输入 nm，模型内部 Å，只转换一次；
- inner/outer cutoff `4/5 Å`；
- strict active membership `r < 5 Å`；
- C2 envelope 不变；
- triclinic MIC 语义不变；
- `r < 0.01 nm` fail-closed；
- active unique environment `<=320`；
- active edges `<=2048`；
- active neighbors per ligand `<=80`；
- candidate-list capacity 与 active ceilings 仍是两个不同概念；
- 不允许 top-k、截断、冻结 796/任意环境 ID 或 silent fallback。

### 3.3 运行与统计 workload

- 73,536-atom 真实体系；
- 同 topology、frame manifest、checkpoint；
- CUDA device、driver、OpenMM build、plugin ABI、mixed precision；
- 同 integrator、constraints、PME、force groups；
- native fused Group1 接线；
- 同 EXP-025 matched-total surrogate：每 500 measured steps 一次额外 generic energy `getState`；
- 同一 baseline 定义；
- 100 warmup、1000 measured、5 paired repeats；
- 计时边界必须与 EXP-025 normative harness 完全相同；不得把 surrogate 偷换为完整 ledger，也不得反过来宣称它已经覆盖完整 production ledger。

### 3.4 禁止用来“优化”的手段

- 改模型宽度、权重、RBF、cutoff、skin 或阈值；
- 改 block/grid 造成数学/归约语义变化而不做完整 correctness；
- 降低 ledger/query 频率；
- 删除 error/overflow 检查；
- 把错误延迟到多个积分步以后；
- overflow 时返回零 residual 并继续；
- 只挑便宜 frame、最好 repeat 或最好顺序；
- 改成 single/double precision；
- 关闭真实 production force/query；
- 用 profiler kernel time 代替 wall-clock qualification；
- 修改 EXP-025 protocol、report 或 decision log。

---

## 4. 允许修改的范围

允许修改仅限 LocalManyBodyResidual CUDA 控制面：

- device counter/status 的存储与归约；
- device buffer reset 方法；
- H2D/D2H 的数量、大小和调度；
- error/status 检查的合并与 piggyback；
- kernel launch orchestration；
- 没有语义副作用的 kernel 合并；
- CUDA stream/event 编排；
- CUDA Graph capture/replay（只有低风险阶段不足时）；
- 与上述变更直接相关的测试、profiling 和 manifest。

首选修改文件范围：

- `plugins/LocalManyBodyResidual/platforms/cuda/src/CudaLocalManyBodyResidualKernels.cpp`
- `plugins/LocalManyBodyResidual/platforms/cuda/src/CudaLocalManyBodyResidualKernels.h`
- 必要的同插件 CUDA kernel source/header；
- 新增 `scripts/exp026_*`；
- 新增 `protocols/EXP-026_*`；
- 新增 `output/outer_lambda_exp026_cuda_control_plane/`。

默认禁止修改：

- `local_residual/softlift.py`；
- EXP-020 checkpoint/export payload；
- `ibs_engine.py` 的数学与生产查询逻辑；
- fused Group1 energy expression；
- EXP-025 timing harness 的计时定义。

如必须修改 production Python 接线或数学文件，EXP-026 立即停止，另开实验。

---

## 5. 目标实现：Control Plane V1

### 5.1 单一 device status block

把分散的 host-visible flags/counters 合并为固定布局的 device status：

```text
DeviceStatusV1
  error_code
  error_stage
  active_edge_count
  max_neighbor_count
  unique_environment_count
  candidate_count
  rebuild_required
  epoch
```

要求：

- 每个字段类型、字节序、reset 时机和 overflow 行为冻结；
- 第一错误获胜，不能被后续 kernel 覆盖；
- downstream kernel 看到 fatal status 后不得提交不完整 residual force；
- host 只读取一个固定小 status block；
- status 读取最多保留一次 consolidated host-visible completion check/force evaluation，用它替代当前多个 blocking downloads；不得等待 500-step matched-total query 才报告错误，也不得额外叠加 `cudaDeviceSynchronize()`；
- 错误必须在当前 production API 边界报告，最大检测延迟不得超过 1 integration step。

EXP-025 matched-total query cadence 是 500 steps，因此不能把它当作 per-step error boundary。若不能同时满足“每次 force evaluation 至多一次 consolidated check”和“fail-closed 检测延迟 <=1 step”，该设计失败，不能用延迟检查换性能。

### 5.2 unique environment：epoch tag + device atomic count

替代每步上传零数组、下载 73,536 flags 并在 host 求和：

```text
unique_env_epoch[N]
current_epoch
unique_env_count
```

活动边首次访问环境原子 `j` 时，用 atomic CAS/exchange 把 `unique_env_epoch[j]` 更新为当前 epoch；只有第一位成功线程递增 `unique_env_count`。

要求：

- 不再每步传输 294,144 B flag array；
- host 不再对该数组求和；
- epoch wrap 在初始化/显式 reset 时处理，不能静默复用；
- exact count 与 EXP-025 reference 对所有 G3 fixtures 完全一致；
- count >320 写 fatal status，绝不截断。

### 5.3 consolidated error harvesting

把约 4 个独立 `checkDeviceErrorFlag` 路径收敛到一个 status harvest：

- kernel 间依赖仍留在同一 CUDA stream；
- 不允许每个阶段做 host-blocking download；
- 不允许用全局 device synchronize 替代多个局部同步；
- 故障注入必须证明每种错误仍映射为稳定的 error code/stage；
- OpenMM exception 文本须包含 stage、code 和关键计数。

### 5.4 launch 路径

低风险优先级：

1. 删除纯 host reset/copy launch；
2. 合并无全局 barrier、无归约顺序变化的控制 kernel；
3. 将多个小参数更新打包；
4. 保留 K0 device displacement 判定；
5. 不允许 host 为了判断 rebuildFlag 而新增同步。

关键限制：K1–K5 有 clear→bin→count→prefix→fill 的全局依赖，不能仅为了少 launch 粗暴拼成普通单 kernel。若需要跨 grid barrier，必须采用被目标 GPU/OpenMM/CUDA 明确支持并单独验证的机制。

### 5.5 CUDA Graph：条件后备，不是默认解

只有 V1 完成后，开发诊断仍显示 launch overhead 足以阻断成本门，才允许评估 CUDA Graph。

Graph 版本必须证明：

- topology、device addresses、stream 和 grid 生命周期稳定；
- box/reorder/rebuild/update 路径可正确 replay 或显式 recapture；
- recapture 不发生在每步；
- force-group 长期跳过后重新启用不会使用陈旧 list；
- checkpoint/restart、Context reinitialize、atom reorder 正确；
- error/status 不被 graph 隐藏；
- graph-disabled fallback 与 V1 数值等价。

条件 graph node、cooperative kernel 或 persistent kernel 属高风险变体。它们不得在看到正式 O4 结果后临时加入；若 V1 已进入 O4 且失败，新增这些设计需要 EXP-027 或 protocol amendment 后全量重新开始。

---

## 6. 实验状态机

| 阶段 | 名称 | 主要输出 | 进入条件 | 通过条件 |
|---|---|---|---|---|
| O0 | Freeze/Preregister | protocol + manifests | 本计划批准 | hashes、候选、统计、顺序冻结 |
| O1 | Build Identity | binary/PTX/source/env manifest | O0 PASS | ABI/load/XML smoke；scope diff PASS |
| O2 | Correctness Regression | full G0–G3 mixed reports | O1 PASS | 所有数值/异常/重启 gate PASS |
| O3 | Attribution Preflight | matched nsys + counters | O2 PASS | 目标传输/同步/launch 变化与设计一致 |
| O4 | Normative Cost | 5 paired repeats raw data | O3 PASS；代码/data locked | 5 对全部完成，无 protocol drift |
| O5 | Decision/Archive | report + decision log | O4 complete | AND gates 决策，append-only 封存 |

任何以下情况进入 `QUARANTINED`，不得继续：

- hash、frame、checkpoint、环境或 workload 不一致；
- correctness 失败；
- NaN/Inf、overflow、timeout、OOM；
- 不完整 measured block；
- 未预注册代码/参数变化；
- 缺失 raw timing/trace/manifest；
- 为了结果好看而重跑、删点或换顺序。

---

## 7. O0 — 预注册与数据锁定

### 7.1 必须冻结的 identity

- EXP-025 source/report/protocol raw-file SHA-256；
- EXP-020 checkpoint、payload、ligand indices、dataset SHA-256；
- plugin source tree、binary、PTX/NVRTC source hash；
- OpenMM/CUDA/driver/compiler/glibc/conda manifest；
- GPU UUID、型号、clock/power/persistence 状态；
- topology、positions、box、checkpoint、frame manifest；
- baseline/candidate System XML structural hash；
- integrator、precision、force groups、query cadence；
- timing harness hash；
- bootstrap method、seed、draw count、quantile algorithm；
- AB/BA 配对顺序；
- 最大 infra retry 次数及合法原因。

环境不是 git repository 时，必须记录源文件逐个 SHA-256 和 canonical tree hash；不能伪造 commit。

### 7.2 单一 primary candidate

O4 只允许一个 primary candidate：

`EXP026_CONTROL_PLANE_V1`

V1 必须在 O0 中列出 compile flags、device status ABI、epoch 算法和 launch graph。O3 的 profiler 只用于验证归因，不能用真实 G4 结果挑多个候选中的最好者。

---

## 8. O1 — 构建、ABI 与 scope 证明

必须通过：

- clean plugin build；
- Reference/CUDA platforms load；
- CUDA mixed Context construct；
- XML schema round-trip；
- forceGroup/name/model payload/temperature/skin/capacity 保存；
- binary/source/PTX hashes 写入 manifest；
- allowlist diff 审查证明只改控制面；
- 禁止数学、权重布局、energy expression 或 IBS cadence 漂移。

如 XML schema 必须升级，需给前一 schema 的显式兼容/拒绝策略。不能静默按新字段默认值读取旧 artifact。

---

## 9. O2 — 正确性与 fail-closed 回归

### 9.1 全量继承 gate

- G0 ABI/CUDA smoke；
- G0 XML round-trip；
- G1 independent oracle；
- G1 OpenMM Reference parity；
- G2 brute-force CUDA vs Reference；
- G3 CSR vs G2/reference；
- mixed precision suite；
- Layer-1 residual oracle；
- Layer-2 fused Group1；
- A=B / C=D=E three-way equivalence。

沿用冻结容差：

- CUDA energy error `<=1e-4 kJ/mol`；
- CUDA force max error `<=1e-3 kJ/mol/nm`；
- Reference finite difference 使用既有更严格阈值；
- no-contact energy/force exact zero；
- 不要求 CUDA atomic reduction 跨调用 bitwise identical，但必须在容差内。

### 9.2 新增控制面专用测试

1. unique-env epoch count 与 host reference exact match；
2. 连续多步不 reset 污染测试；
3. epoch wrap/reinitialize 测试；
4. edges=2048、neighbors=80、unique-env=320 边界通过；
5. 三类 +1 overflow 分别 fail-closed；
6. candidate capacity overflow；
7. r<0.01 nm；
8. half-box active tie；
9. zero-contact/zero-candidate；
10. rebuild/no-rebuild cutoff crossing；
11. displacement >skin/2；
12. box change和不支持盒；
13. atom reorder；
14. force group 长期跳过后重新启用；
15. checkpoint/restart；
16. XML deserialize 到全新 mixed CUDA Context；
17. 每个 error stage 故障注入；
18. fatal status 下不得提交部分 residual force；
19. error 在既定 API boundary 报出且延迟 <=1 step；
20. 重复至少 10,000 steps 检查 counter 污染、NaN、内存增长。

任何 correctness 失败都不是“性能失败”，而是 `REJECT_IMPLEMENTATION`。

---

## 10. O3 — Attribution preflight

O3 使用 matched tracing；baseline 和 candidate 使用相同 tracing 设置。trace 数据不进入 primary cost gate。

必须报告：

- end-to-end ms/step；
- named GPU kernel us/step；
- kernel launches/step；
- H2D/D2H call count 与 bytes/step；
- blocking synchronization count/time；
- CUDA API launch gaps；
- stream idle；
- neighbor-list rebuild frequency 与 spike；
- active/candidate edge 分布；
- GPU memory peak；
- temperature/clock/power envelope。

预期结构变化：

- 588,288 B/step unique-env 往返消失；
- unique-env host sum 消失；
- 多个 error/status downloads 合并；
- 不新增全局同步；
- launch 数下降或 host launch gap 明显下降；
- correctness 与 query cadence 不变。

若 wall time 下降但预注册目标计数完全不变，归因为 `UNEXPLAINED`，不能直接进入 O4。若计数改善但 wall time不改善，假设可能被否证，但仍可在代码锁定前按预注册规则停止，不能后验换方法。

---

## 11. O4 — 正式成本资格

### 11.1 计时设计

- 5 个完整 paired repeat；
- 每个 arm 恰好 100 warmup；
- 每个 arm 恰好 1000 measured；
- pairing order 严格继承 EXP-025：repeat 0/2/4 baseline-first，repeat 1/3 candidate-first；
- baseline 与 candidate 使用同一设备、输入、checkpoint、进程策略，以及每 500 steps 一次额外 generic energy `getState` 的 matched-total surrogate；
- primary run 不开 nsys/ncu；
- 不删 outlier，不按 kernel time 修正 wall time；
- 不在看到结果后延长 warmup 或 measured steps；
- 不增加 per-step timer。保存每个 arm 的整段 `pure_seconds`、`total_seconds`、执行顺序和 ratio；`integrator.step(1000)` 仍是单次 timed call，前后用与 EXP-025 相同的 energy `getState` 排空 GPU。

### 11.2 统计定义

严格继承 EXP-025 G4 的 per-arm summary 与 ratio 定义。新 protocol 必须逐字明确，避免把 dimensionless ratio 误标为 ms/step。

若继承定义为：

```text
tB_i = baseline total elapsed seconds for the frozen 1000-step block
tC_i = candidate total elapsed seconds for the frozen 1000-step block
r_i  = tC_i / tB_i
R    = median(r_1 ... r_5)
```

则 bootstrap 必须以 5 个 paired repeat 为 cluster 单位重采样，不得把 5000 steps 当独立样本。必须冻结：

- bootstrap draws：严格继承 EXP-025 的 `20,000`；
- RNG seed；
- replacement 规则；
- median statistic；
- one-sided 95% empirical quantile 的具体插值方法。

所有 `r_i`、raw summaries 和 bootstrap quantile 都写入报告。

### 11.3 Primary gates

```text
G4_MEDIAN_PASS = (R <= 1.07)
G4_P95_PASS    = (U95 <= 1.10)
FULL_COST_PASS = G4_MEDIAN_PASS AND G4_P95_PASS
```

缺失、NaN、Inf、少于 5 对、hash mismatch 均按 FAIL，不按 UNKNOWN 放行。

### 11.4 Infra retry

只允许预注册的客观 infra 原因：GPU reset、driver failure、ECC、OOM、thermal/clock 超出冻结 envelope、文件/hash 损坏。

- 一个不完整 pair 必须整体重跑，不能拼接；
- baseline 或 candidate 任一 arm 失败，整对作废；
- 所有失败尝试 append-only 保存；
- 完整但性能差的 pair 绝不允许重跑；
- 不得保留最好 retry；
- 超过预注册最大 retry 次数则实验失败。

---

## 12. O5 — 决策矩阵

| correctness | invariants | median | P95 upper | 决定 |
|---|---|---|---|---|
| FAIL | 任意 | 任意 | 任意 | `REJECT_EXP026_IMPLEMENTATION` |
| PASS | FAIL | 任意 | 任意 | `QUARANTINE_PROTOCOL_DRIFT` |
| PASS | PASS | `<=1.07` | `<=1.10` | `EXP026_RUNTIME_COST_QUALIFIED` |
| PASS | PASS | `>1.07` | 任意 | `STOP_EXP026_RUNTIME_BACKEND` |
| PASS | PASS | 任意 | `>1.10` | `STOP_EXP026_RUNTIME_BACKEND` |

通过 EXP-026 也不等于 production promotion。它只允许进入下一阶段：paired short-NVT safety 与 online ESS/GPU-hour utility qualification。

---

## 13. 性能通过后的下一门（不属于 EXP-026）

后续实验至少需要：

- 3 个 paired independent repeats；
- arm 间相同初始 positions、velocity array、Langevin seed；repeat 间不同；
- burn-in 计入 GPU-hour、不计入 ESS；
- ESS protocol 明确继承 mixture-coverage ESS v3 或新 sealed version；
- 每 repeat `eta = min_{window,state} ESS_mixture / GPU-hour`；
- 至少 2/3 repeats candidate 优于 baseline；
- median material improvement 预注册；
- TMBAR convergence、overlap、ledger closure、state/window coverage、DeltaG consistency、温度/能量/力/constraint health 全过。

不得把 EXP-020 的 55.55% gap-variance proxy直接报告为 online ESS/GPU-hour。

---

## 14. 产物树

```text
PLAN_EXP-026_cuda_control_plane_optimization.md
protocols/
  EXP-026_control_plane_preregistration.json
output/outer_lambda_exp026_cuda_control_plane/
  o0_freeze/
    identity_manifest.json
    source_tree_manifest.json
    input_manifest.json
    protocol_sha256.txt
  o1_build/
    build_report.json
    plugin_binary_manifest.json
    serialization_roundtrip_report.json
  o2_correctness/
    g0_g3_regression_report.json
    mixed_precision_report.json
    device_status_fault_injection_report.json
    long_run_counter_integrity_report.json
  o3_attribution/
    attribution_report.json
    nsys_profiles/
    api_call_counts.csv
    kernel_summary.csv
    memcpy_summary.csv
  o4_cost/
    raw_timing.jsonl
    paired_repeat_report.json
    bootstrap_samples_or_digest.json
    cost_qualification_report.json
  exp26_result.md
  decision_log.jsonl
```

每个 report 同时记录：

- `raw_file_sha256`；
- 排除 self-hash 字段后的 `canonical_self_sha256`；
- protocol/source/binary/input parent hashes；
- status enum；
- attempt ID；
- environment/device identity。

---

## 15. 实施顺序

### Patch A — Device counters

- 加入 `DeviceStatusV1`；
- unique-env epoch tags；
- device atomic count；
- 删除 294 KB flag H2D/D2H；
- 保留现有 kernel math；
- 运行 O2 专用测试。

### Patch B — Error/status consolidation

- 合并 error flags；
- 把多个 blocking checks 合并为每次 force evaluation 至多一个小 status harvest；不得推迟到 500-step matched-total query；
- 删除多余 blocking downloads；
- 故障注入证明 fail-closed；
- 再跑完整 O2。

### Patch C — Launch/control cleanup

- 删除无语义副作用的 reset/copy launches；
- 打包小参数更新；
- 不读取 rebuildFlag 到 host；
- 不跨 barrier 粗暴融合 K1–K5；
- 再跑完整 O2。

### Patch D — Attribution lock

- matched nsys；
- 验证 transfer/sync/launch 的预期变化；
- 冻结 V1 source/binary；
- 生成 O3 report。

### Patch E — Normative cost

- 从全新 Context 开始；
- 运行预注册 AB/BA 五对；
- 数据锁定后一次性运行固定统计；
- append-only 写 decision。

CUDA Graph/高风险融合默认不在 V1。只有在 O0 前就决定纳入并冻结，或 V1 失败后另开 EXP-027，才可正式评估。

---

## 16. 最短成功路径与停止规则

最短成功路径：

```text
device-only counters
  -> one consolidated status harvest at existing sync boundary
  -> remove redundant transfers/syncs
  -> correctness full pass
  -> attribution matches hypothesis
  -> same G4 harness passes 1.07 AND 1.10
```

停止规则：

1. 若必须修改 R1 数学才能达标：STOP。
2. 若必须降低 ledger/query 频率才能达标：STOP。
3. 若错误只能延迟多个 integration steps 才能省同步：STOP。
4. 若 O2 correctness 失败：STOP，不计性能。
5. 若正式 O4 任一成本门失败：封存 `STOP_EXP026_RUNTIME_BACKEND`。
6. 不允许在同一 EXP-026 内看到失败后继续叠加未预注册优化直到过线。

---

## 17. 当前判断

EXP-026 有明确的工程价值，也有真实成功可能：当前需要削减约 138.3 us，而 postmortem 估算的未由具名 plugin kernels 解释的 overhead 约 228.8 us。所需削减约为后者的 60.5%。

但这不是已证明可回收的预算。唯一有效结论来自同一个冻结 workload 上重新完成五对 paired measurement。项目目标不是让 profiler 图更漂亮，而是在不牺牲数学、错误语义和 production workload 的条件下，同时通过：

```text
median ratio <= 1.07
bootstrap one-sided P95 upper <= 1.10
```



---

## 18. 局部多体势的候选池与 device-resident epoch 设计

### 18.1 四个规模必须分开

EXP-026 的物理支持域仍严格是 5 Å。73,536 原子只是 System 总拓扑和稳定 device atom-ID 空间，不是每步进入 R1 MLP 的原子数。

| 层级 | 典型规模 | 作用 | 进入 R1 数学 |
|---|---:|---|---|
| Full System | 73,536 atoms | OpenMM 全体系 | 否 |
| Verlet candidate pool | 约 2,575 directed pairs | 距离小于 r_list=6 Å 的候选 | 只做轻量距离过滤 |
| Active support | 约 1,200 directed pairs | 严格 r<5 Å 的真实边 | 是 |
| Unique active environment | 不超过 320 atoms | active edges 接触的环境集合 | 只用于支持域审计 |

目标数据流：

~~~text
73,536 atoms
  -> 偶尔重建 GPU linked-cell/CSR
约 2,575 candidates at r<6 Å
  -> 每步 live MIC + strict r<5 Å
约 1,200 active directed edges
  -> C2 + RBF16 + typed pair weight
q_i for 41 anchors
  -> typed rho(q)-rho(0), ligand SUM, tanh
one reduced scalar B and conservative forces
~~~

R1 没有理由把 5 Å 外的原子送入 MLP。

### 18.2 当前全长 flag 的真实含义

EXP-025 使用：

~~~cpp
ComputeArray uniqueEnvFlagDevice; // int[numParticles]
~~~

active edge 用当前 OpenMM device slot 作为 envId，因此可直接执行：

~~~cuda
uniqueEnvFlag[envId] = 1;
~~~

长度为 73,536 只是为了允许任意 solvent 原子动态进入 5 Å，并用 dense atom-ID 做 O(1) 去重。它不是全体系物理计算。

当前低效路径是每次 execute 都：

~~~text
host 构造 73,536 个零
 -> H2D 294,144 B
GPU 标记约几百个 atom-ID
 -> D2H 294,144 B
host 扫描并求 unique count
~~~

即 588,288 B/evaluation 的往返和 blocking synchronization。EXP-026 要消灭的是每步搬运整张记账表，而不是 dense device-ID 空间。

### 18.3 冻结选择：dense epoch tags

V1 采用：

~~~cpp
ComputeArray uniqueEnvEpochDevice; // int[numParticles], device-resident
int uniqueEnvironmentEpoch;        // host positive epoch
~~~

初始化只清零一次：

~~~cpp
uniqueEnvEpochDevice.initialize<int>(cu, numParticles, "lmbrUniqueEnvEpoch");
uniqueEnvEpochDevice.upload(vector<int>(numParticles, 0));
uniqueEnvironmentEpoch = 0;
~~~

每次 evaluation 只递增 host 小整数。signed int epoch 到 INT_MAX 时，前一 evaluation 已完成，才允许罕见地整表清零并从 1 重启；正常模拟没有逐步 memset/H2D。

每条 active edge：

~~~cuda
int previous = atomicExch(
    &uniqueEnvEpoch[envId],
    currentEpoch);

if (previous != currentEpoch) {
    int count = atomicAdd(
        &status[EXP026_STATUS_UNIQUE_ENVIRONMENTS], 1) + 1;

    if (count > MAX_ENVIRONMENT_ATOMS)
        exp026SetFirstError(
            status,
            EXP025_DEVICE_ERROR_UNIQUE_ENV_OVERFLOW,
            EXP026_STAGE_COMPUTE_Q);
}
~~~

性质：

- 同一环境原子被多个 anchor 接触时只计一次；
- 新 solvent atom 可随时进入/离开支持域；
- 不冻结历史环境 IDs；
- 不排序 edge；
- 不下载 active atom-ID list；
- 正常步骤没有 numParticles 规模的清零、H2D 或 D2H；
- 只对约 1,200 条 active edges 做一次 atomicExch。

### 18.4 为什么 V1 不用 512-slot hash

unique environment ceiling 是 320，理论上可使用 512-slot hash。但 V1 不选它：

- hash 需要 collision probing 和更多 CAS；
- 必须处理 CLAIMED/READY 竞争；
- 更容易产生重复计数、table-full 和死锁边界；
- dense epoch array 只占约 294 KB persistent GPU memory；
- envId 已是 0 到 numParticles-1 的 dense slot，可直接寻址；
- persistent 显存不是 PCIe 成本。

只有 profiler 证明 dense epoch 的随机访问成为显著瓶颈，才允许在独立实验评估 compact hash。

### 18.5 为什么重建邻居表仍会遍历全体系

为了确定谁进入 6 Å candidate pool，真正 rebuild 时仍需处理环境原子。现有 G3 保持：

~~~text
K0 displacement check
K1 clear cell heads
K2 bin environment atoms
K3 count candidates for 41 anchors
K4 prefix-sum CSR offsets
K5 fill candidate atom IDs
K6 evaluate live r<5 Å support
~~~

K2 在 rebuild 步遍历体系做 cell binning，这是邻居搜索，不是 MLP。正常 no-rebuild step 中 K1-K5 由 rebuildFlagDevice self-gate。

不能永久固定环境 atom IDs，因为 solvent、离子和侧链会跨越 5/6 Å 边界。候选 IDs 只是性能结构；K6 仍每步重算 live MIC 和 strict r<5 Å。

### 18.6 完全复用 EXP-025 接口

不增加新 Force、KernelImpl、SWIG、Python API 或 XML 字段。保持：

~~~cpp
LocalManyBodyResidualForce
LocalManyBodyResidualForceImpl
CalcLocalManyBodyResidualForceKernel
CudaCalcLocalManyBodyResidualForceKernel
~~~

保持 initialize、execute、onAtomsReordered 原签名。serialization 继续 schema v2。epoch、status、CSR 和 cell heads 均为 Context-local scratch，不进入 XML。

第一补丁只允许修改：

~~~text
plugins/LocalManyBodyResidual/
  exp026_control_plane_layout.h
  exp026_control_plane_layout_test.cpp
  platforms/cuda/src/CudaLocalManyBodyResidualKernels.h
  platforms/cuda/src/CudaLocalManyBodyResidualKernels.cpp
~~~

### 18.7 private buffers 的最小替换

EXP-025：

~~~cpp
ComputeArray neighborCountDevice;   // int[41]
ComputeArray uniqueEnvFlagDevice;   // int[73536]
ComputeArray errorFlagDevice;       // int[1]
~~~

EXP-026 V1：

~~~cpp
ComputeArray neighborCountDevice;       // int[41], q kernel 每个 anchor 覆盖
ComputeArray uniqueEnvEpochDevice;      // int[73536], persistent
ComputeArray deviceStatusDevice;        // int[8]
int uniqueEnvironmentEpoch = 0;
ComputeKernel resetStatusKernel;
~~~

neighborCount 暂时保留以最小化 K1/K6a 改动，但删除逐步 zero upload 和 host download。总 edge 与 max-neighbor 直接归约到 status。

status 固定布局：

~~~text
0 error_code
1 error_stage
2 active_edges
3 max_neighbors
4 unique_environments
5 candidates
6 epoch
7 reserved
~~~

错误码继续使用冻结的 EXP025_DEVICE_ERROR 0..8，不重新编号。

### 18.8 device helper

第一错误获胜：

~~~cuda
__device__ inline void exp026SetFirstError(
        int* status, int code, int stage) {
    int old = atomicCAS(
        &status[EXP026_STATUS_ERROR_CODE],
        EXP025_DEVICE_ERROR_OK,
        code);

    if (old == EXP025_DEVICE_ERROR_OK)
        status[EXP026_STATUS_ERROR_STAGE] = stage;
}
~~~

每个 anchor 提交 counts：

~~~cuda
int total = atomicAdd(
    &status[EXP026_STATUS_ACTIVE_EDGES],
    anchorNeighbors) + anchorNeighbors;

atomicMax(
    &status[EXP026_STATUS_MAX_NEIGHBORS],
    anchorNeighbors);
~~~

超过 maxNeighbors 或 maxEdges 时写 first error。

一个 1-thread reset kernel 替代三次 host upload：

~~~cuda
extern "C" __global__ void exp026ResetStatus(
        int* status, int epoch) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    status[0] = EXP025_DEVICE_ERROR_OK;
    status[1] = EXP026_STAGE_NONE;
    status[2] = 0;
    status[3] = 0;
    status[4] = 0;
    status[5] = 0;
    status[6] = epoch;
    status[7] = 0;
}
~~~

### 18.9 kernel 参数最小变化

G2 q 和 G3 computeQFromCSR 的尾部从：

~~~cuda
real* q,
int* neighborCount,
int* uniqueEnvFlag,
int* errorFlag
~~~

变为：

~~~cuda
real* q,
int* neighborCount,
int* uniqueEnvEpoch,
int currentEpoch,
int* deviceStatus
~~~

RBF、C2、typed MLP 和 q reduction arithmetic 一行不变。

prefixSumOffsets 在 rebuild 时额外写 candidate count。readout、fill、scatter 的旧 errorFlag 参数换成 deviceStatus，写错统一使用 first-error helper。

### 18.10 execute 目标流程

~~~cpp
cu.setAsCurrent();
ensureDeviceStateCurrent();

advanceUniqueEnvironmentEpoch();
launchResetStatus(uniqueEnvironmentEpoch);

readAndValidateBoxExactlyAsExp025();

if (!g3Enabled)
    launchComputeQBruteForce();
else {
    launchK0DisplacementOrForceRebuild();
    launchSelfGatedK1ToK5();
    launchComputeQFromCSR();
}

launchReadout();

if (includeForces)
    launchScatter();

checkDeviceStatusOnce("completed evaluation");

if (includeEnergy)
    return downloadEnergyExactlyAsExp025();

return 0.0;
~~~

删除：

~~~text
neighborCountDevice.upload(41 zeros)
uniqueEnvFlagDevice.upload(73536 zeros)
errorFlagDevice.upload(OK)
neighborCountDevice.download()
uniqueEnvFlagDevice.download()
K1/K2/K3/K6 中间多个 errorFlag downloads
~~~

新增/保留：

~~~text
one tiny reset-status launch
one int[8] status D2H/check per force evaluation
existing energy scalar download when includeEnergy=true
~~~

status 不能延迟到每 500 步 matched-total getState；错误必须在当前 force evaluation/积分步 fail-closed。

### 18.11 partial-force 安全

不能简单等 scatter 完成后再看 status。现有 scatter 可能在发现 nonfinite、min-distance 或 tie 的同一次执行中已向 OpenMM force buffer atomicAdd。

V1 必须证明 q/K6a 对 scatter 的所有 pair 做同样的 distance、tie、minimum-distance 和 derivative-finite preflight。scatter 启动前若 status fatal，立即 return。

如果存在只能在 scatter arithmetic 发现的错误，合法方案只有：

1. 把同样检查前移到 q/preflight；或
2. scatter 先写 resident scratch，最终 commit kernel 仅在 status OK 时写 OpenMM force buffer。

优先方案 1，因为改动和成本更小，但必须用 fault injection 与 finite difference 证明。禁止提交 partial force 后再抛异常。

### 18.12 Patch A 验收

必须证明：

~~~text
73536-int flag H2D/evaluation == 0
73536-int flag D2H/evaluation == 0
neighbor-count D2H/evaluation == 0
device status D2H/evaluation == 1
device status bytes/evaluation == 32
unique count exact
active edge count exact
max neighbor count exact
candidate count exact
~~~

正确性回归覆盖 G2/G3 single、mixed、double、no-contact、triclinic、rebuild/no-rebuild、cutoff crossing、reorder、force-group skip/re-enable、四类 ceiling 与 +1 overflow、first-error race、10,000-step epoch contamination、XML new Context 和 fused Group1 equivalence。

Patch A 后单独 profile。只有证据显示同步或 launch 仍为主瓶颈，才进入 Patch B/C；不得同时重写邻居表、R1 数学和控制面。


把下面整段追加到 `PLAN_EXP-026_cuda_control_plane_optimization.md` 末尾即可：

```markdown
## 19. 当前实施状态（2026-08-13）

EXP-025 的冻结结论保持不变：

```text
STOP_EXP025_RUNTIME_BACKEND
normative fused Group1 median ratio = 1.1233981725063995
bootstrap one-sided P95 upper = 1.1321719030983386
```

EXP-026 是独立的控制面优化实验，不能覆盖、修改或重新解释 EXP-025 的失败结论。

### 19.1 A1 已实现范围

当前 CUDA 后端已经完成以下有界改造：

- 使用常驻 GPU 的 `uniqueEnvEpochDevice[int(N)]` 和正整数 evaluation epoch，替代每次 evaluation 对 `uniqueEnvFlagDevice[int(73536)]` 的清零、H2D、D2H 和 host 求和。
- active-edge、max-neighbor、unique-environment 计数改为在 GPU 上写入私有 `deviceStatusDevice[int(8)]`。
- 删除每次 evaluation 对 `neighborCountDevice` 的清零和 D2H。
- epoch-tag 整数组只在初始化时清零；之后仅在极罕见的 signed-int epoch wrap 时重新清零。
- EXP-025 的 `errorFlagDevice[int(1)]` 和多个中间 fail-closed 同步暂时保留；同步合并属于后续 A2，不在 A1 中删除。
- R1 数学、CSR membership、cutoff、skin、mixed-position reconstruction、public Force/KernelImpl、XML schema v2 和 IBS wiring 均未改变。

这里需要区分四种规模：

| 层级 | 当前规模 | 用途 |
|---|---:|---|
| 全体系原子 | 73,536 | 仅用于 rebuild 时的精确分箱和常驻 epoch 索引空间 |
| Verlet candidate pool | 约 2,575 | `r < r_list=6 Å` 的候选边 |
| active edges | 约 1,206 | `r < r_cut=5 Å`，真正进入 R1 数学 |
| unique active environment atoms | ≤320 | support ceiling 与诊断 |

因此，R1 MLP 从来没有处理全部 73,536 个原子。A1 删除的是针对全体系 dense bookkeeping array 的每步 CPU↔GPU 往返，而不是修改 R1 的局部数学。

### 19.2 A1 已通过证据

```text
EXP026_A1_CONTROL_LAYOUT_CONTRACT = PASS
EXP026_A1_BUILD_AND_LINK = PASS
EXP026_A1_NVRTC_CUDA_SMOKE = PASS
EXP026_A1_G2_MIXED_REGRESSION = PASS (0 failing checks)
EXP026_A1_G3_MIXED_REGRESSION = PASS (0 failing checks)
EXP026_A1_CORRECTNESS_CHECKPOINT = PASS
```

关键数值：

| 指标 | 观测 | 冻结容差 | 结果 |
|---|---:|---:|---:|
| canonical energy absolute error | `7.60572e-05 kJ/mol` | `1e-4` | PASS |
| G2 canonical maximum force error | `5.38032e-04 kJ/mol/nm` | `1e-3` | PASS |
| G3 canonical maximum force error | `5.38986e-04 kJ/mol/nm` | `1e-3` | PASS |
| G3 versus G2 maximum force difference | `1.52588e-05 kJ/mol/nm` | `1e-3` | PASS |

回归覆盖：

- 真实 73,536 原子 canonical frame；
- energy-only、force-only 和 energy+force；
- force-group mask；
- no-contact exact zero；
- triclinic PBC；
- finite difference；
- atom reorder；
- XML round-trip 到全新 CUDA Context；
- minimum-distance 和 neighbor overflow；
- cutoff crossing without rebuild；
- displacement、box-change 和 reorder rebuild；
- force-group skip/re-enable；
- candidate overflow；
- active-support overflow。

### 19.3 当前边界（2026-08-13 追加更新：以下四项已被后续 O3/O4 实测解决，见第 23 节完整结果）

```text
EXP026_A1_TRANSFER_ACCOUNTING = PASS（O3 attribution，见 23.1：H2D 次数 -2,206/-1.996 每 evaluation，H2D 字节 -324.7 MB/-68.2%，D2H 次数 -2,210/-2.000 每 evaluation，D2H 字节 -325.2 MB/-90.9%）
EXP026_A1_SINGLE_DOUBLE_REGRESSION = NOT_RUN
EXP026_A1_LONG_EPOCH_STRESS = NOT_RUN
EXP026_PATCH_A_FULL_QUALIFICATION = QUALIFIED（Patch A1.1，见 23.3：O4 normative cost gate PASS，STOP_OPTIMIZATION_SUCCESS）
EXP026_RUNTIME_COST_QUALIFICATION = PASS（Patch A1.1，见 23.3：median=1.04140<=1.07，bootstrap P95 upper=1.04692<=1.10）
```

A1 correctness PASS 只证明语义没有回归——这一判断本身仍然成立。现在另外可以声称（细节见第 23 节）：

- H2D/D2H 已按预期结构性减少，并有真实 nsys 归因数据支持（不再只是推断）；
- 5/5 配对 wall-clock 重复一致地更快（A_recon vs B_a1，O3，median ratio = 0.9427）；
- 针对真实 no-plugin baseline 的 normative median/P95 成本门（Patch A1.1，O4）已经双双通过；
- EXP-026 runtime backend（A1.1 candidate）已经 qualified，决定为 `STOP_OPTIMIZATION_SUCCESS`。

仍然不能声称：

- single/double precision 回归或 10,000-step 长时 epoch stress 已经验证（这两项仍是 `NOT_RUN`）；
- A2/A3/A4/A5 已经完成或必须全部完成才算数（按用户既定规则，在 A1.1 通过门槛后被明确跳过，见 23.3；A2 后来又被单独重新拾起做正确性与成本归因，见 23.4，但 A2 自己的 normative O4 尚未运行）；
- online ESS/GPU-hour 已经改善，或 production promotion 已被允许（第 22 节的限制原样保留）。

---

## 20. 后续四项控制面优化计划

四项优化必须顺序执行、单独归因。

每项必须先通过正确性回归，再单独 profile，之后才能决定是否进入下一项。禁止把四项一次性叠加，只报告一个无法归因的总加速比。

## 20.1 A2：合并 CPU↔GPU 错误检查同步

### 20.1.1 当前问题

每次 force evaluation 仍保留：

- `errorFlagDevice[int(1)]` 的 reset/upload；
- rebuild 后的 `checkDeviceErrorFlag()`；
- q/K6a 后的 `checkDeviceErrorFlag()`；
- readout 后的 `checkDeviceErrorFlag()`；
- scatter 后的 `checkDeviceErrorFlag()`。

这些数据量很小，但每一次 D2H 检查都可能形成 host-blocking synchronization。根据 EXP-025 postmortem，它们是 CPU wait 的主要候选来源之一。

### 20.1.2 目标接口

统一使用私有状态：

```cpp
struct Exp026ControlPlaneStatus {
    int errorCode;
    int errorStage;
    int activeEdges;
    int maxNeighbors;
    int uniqueEnvironments;
    int candidates;
    int epoch;
    int reserved;
};
```

每个设备 kernel 只写这个状态。错误语义改为 first-error-wins：

```cuda
__device__ inline void exp026SetFirstError(
        int* status,
        int code,
        int stage) {
    if (atomicCAS(&status[ERROR_CODE], OK, code) == OK) {
        status[ERROR_STAGE] = stage;
        __threadfence();
    }
}
```

禁止继续用下面这种语义作为最终错误记录：

```cuda
atomicExch(errorFlag, code);
```

因为它允许后面的错误覆盖最先发生的错误。

Host 端目标接口：

```cpp
Exp026ControlPlaneStatus checkDeviceStatusOnce(
    const char* completedStage,
    int expectedEpoch);
```

每次 force evaluation 最多 harvest 一次 status。

### 20.1.3 partial-force 安全

不能简单把所有错误检查移动到 scatter 之后。

当前 scatter kernel 可能在发现 nonfinite、minimum-distance 或 half-box tie 的同一次执行中，已经向 OpenMM force buffer 写入部分 atomic force。

因此必须满足以下之一：

1. q/K6a 在 scatter 前对 scatter 将访问的所有 active pairs 完成同样的 index、distance、tie、minimum-distance 和 derivative-finite preflight；
2. 或者 scatter 先写入 resident scratch force buffer，最终 commit kernel 只有在 status OK 时才把结果写入 OpenMM force buffer。

优先采用方案 1，因为成本较低。但必须用 fault injection 证明 q/K6a 覆盖 scatter 的所有 fatal 条件。

禁止：

```text
先向 OpenMM force buffer 写 partial force
-> scatter 末尾发现错误
-> host 下载 status
-> 抛异常
```

错误必须在当前 force evaluation 返回前可见，不能延迟到每 500 步的 ledger 查询。

### 20.1.4 A2 验收

```text
legacy errorFlag H2D/evaluation == 0
legacy errorFlag D2H/evaluation == 0
device status harvest/evaluation <= 1
device status bytes/evaluation <= 32
cudaDeviceSynchronize/evaluation == 0
all injected faults reported within the same force evaluation
first error code/stage stable under races
no partial residual force on every injected fatal path
```

Profile 必须分别报告：

- synchronization count；
- synchronization wall time；
- D2H/H2D bytes；
- host idle/gap；
- total step time。

如果同步次数下降，但 wall time 没有统计可分辨的改善，则 A2 的性能归因失败，不能把其他阶段的收益记到 A2。

---

## 20.2 A3：接入 OpenMM CUDA energy buffer

### 20.2.1 当前问题

当前 `includeEnergy=true` 时：

1. CUDA readout 写 `energyScratchDevice[1]`；
2. host 调用 `downloadRealScalar()`；
3. host 乘 `kBT`；
4. `execute()` 返回 `U_B`。

这会引入插件私有的能量 D2H 和同步，并影响 CustomCVForce/IBS 的真实能量查询路径。

### 20.2.2 目标实现

CUDA kernel 直接写入 OpenMM 当前 Context 的 energy buffer：

```text
U_B = kBT * B
```

必须遵守 OpenMM 8.5.2 commit `36a30cb` 的：

- energy-buffer layout；
- autoclear 规则；
- per-thread/per-block partial-energy 写入规则；
- platform reduction 生命周期。

CUDA `execute()` 不再为能量调用插件私有的：

```cpp
downloadRealScalar(energyScratchDevice, ...)
```

`includeEnergy=false` 时不得写入或下载能量。

插件输出保持为：

```text
raw U_B in kJ/mol
```

插件内部不得加入：

- `A_k`；
- β；
- residual energy offset；
- state-specific coefficient。

必须继续保证：

```text
tanh：一次
kBT：一次
offset：外层一次
A_k：OuterLambdaController 一次
β：ledger-owned，不重复应用
```

### 20.2.3 必须先完成的 API probe

LocalManyBodyResidualForce 既可能 standalone，也可能作为 CustomCVForce 的 child 位于 inner Context。

因此必须分别证明以下三条链读取到同一个 `U_B`：

1. standalone LocalManyBodyResidualForce；
2. nested CustomCVForce collective variable；
3. native fused Group1 IBS wrapper。

不能用 standalone PASS 推断 nested CV 自动 PASS。

### 20.2.4 A3 验收

```text
standalone energy/force parity = PASS
nested CustomCVForce energy/force parity = PASS
fused Group1 three-way equivalence = PASS
energy-only / force-only / both semantics = unchanged
plugin-private energy D2H/evaluation == 0
no duplicate kBT or A_k scaling
XML/restart mixed CUDA = PASS
force-group accounting unchanged
```

如果 OpenMM inner Context 无法正确归约 child 写入的 energy buffer：

```text
A3 = STOPPED_UNSUPPORTED_BY_TARGET_OPENMM
```

此时保留旧的 scalar download。禁止通过减少 production energy-query frequency 来伪造性能改善。

---

## 20.3 A4：降低 K1–K5 空 kernel launch/control 开销

### 20.3.1 当前问题

G3 每步先运行 K0 displacement check。

即使不需要 rebuild，host 仍然发射 K1–K5：

```text
K1 clearCellHeads
K2 binEnvironmentAtoms
K3 countCandidates
K4 prefixSumOffsets
K5 fillCandidates
```

这些 kernel 在设备端读取：

```text
rebuildFlag == 0
```

然后迅速 return。

计算量接近零，但仍然产生：

- CPU kernel-launch overhead；
- driver submission；
- stream scheduling；
- launch latency；
- GPU front-end work。

### 20.3.2 禁止的错误修法

禁止每步下载 `rebuildFlag` 到 host，再决定是否发射 K1–K5。

原因是：

```text
一次 host-blocking D2H
```

可能比五个快速 self-gated kernel 更慢。

同样禁止：

- 固定每 N 步 rebuild，而忽略真实 displacement；
- 忽略 box change；
- 忽略 atom reorder；
- 忽略 force-group 长期跳过后的 candidate invalidation；
- 冻结 candidate atom IDs；
- 删除隐藏的 event、counter、status 或 ordering side effect；
- 为减少 launch 而改变 R1 数学或 candidate semantics。

### 20.3.3 候选路线

按风险从低到高执行。

#### 路线 1：合并小型控制 kernel

可尝试合并：

- reset status；
- clear small counters；
- finalize status；
- 无依赖冲突的 metadata update。

不得跨越 K1→K5 必须存在的全局依赖边界。

#### 路线 2：CUDA Graph replay

固定以下对象后 capture K0–K6：

- topology；
- device addresses；
- array capacities；
- grid dimensions；
- CUDA stream；
- precision mode。

设备侧 `rebuildFlag` 继续控制 K1–K5 是否实际工作，但 host 从多次 kernel submission 降为一次 graph launch。

生命周期要求：

```text
all persistent arrays allocated before capture
atom reorder -> invalidate graph and candidate list
Context reinitialize -> destroy and recapture
device-address change -> recapture
cell-grid shape change -> existing fixed-NVT fail-closed
force-group skipped -> graph must not assume candidate list remains fresh
no per-step graph recapture
```

#### 路线 3：Conditional Graph nodes

只有当前 CUDA/OpenMM/driver 明确支持，并且 prototype 证明以下语义正确时才允许：

- mixed precision；
- error propagation；
- restart；
- force-group skip；
- graph invalidation；
- dynamic rebuild condition。

#### 路线 4：Cooperative/fused rebuild kernel

只有 CUDA Graph 后仍不足，并且能够证明：

- cooperative launch 支持；
- occupancy 足够；
- global-barrier 正确；
- 不产生新的全局同步；
- 不破坏 fault handling；

才考虑把多个 rebuild 阶段进一步融合。

它不是首选路线。

### 20.3.4 A4 验收

```text
host graph/kernel submissions materially reduced
no host read of rebuildFlag
rebuild trajectory matches frozen G3 tolerance
no-rebuild trajectory matches frozen G3 tolerance
box/reorder/force-group skip tests PASS
candidate/active overflow still fail closed
graph-disabled fallback PASS
no per-step graph recapture
```

报告必须分别记录：

- host submission count；
- graph launch count；
- graph 内实际 kernel count；
- self-gated/no-op kernel count；
- CPU launch time；
- GPU idle gaps；
- total step time。

不能把 CUDA Graph 内部仍然存在的 no-op kernel 错报成“kernel 数为零”。

---

## 20.4 A5：保留精确 rebuild，降低摊销成本

### 20.4.1 不可删除的物理工作

受体与配体环境总体稳定，不等于具体 environment atom identity 永远不变。

溶剂、离子和水分子仍然会交换。精确 dynamic membership 要求在以下情况重建：

- 任一相关 atom displacement 超过 `skin/2`；
- periodic box 改变；
- OpenMM atom reorder；
- force group 长期跳过后重新启用；
- Context reinitialize；
- candidate list 被明确 invalidated。

因此 rebuild 时扫描 73,536 个原子并进行 GPU 分箱不能完全删除。

删除该工作等价于冻结 environment membership，会改变模型。

### 20.4.2 可优化范围

允许优化：

- GPU binning 的 memory access；
- cell-head clear；
- candidate count/fill；
- rebuild kernel launch；
- resident buffer reuse；
- rebuild frequency 的摊销；
- skin 的独立预注册比较。

必须保持：

```text
r_list = r_cut + skin
candidate set = exact r < r_list
active set = exact r < r_cut
rebuild trigger = displacement > skin/2
```

Skin 调整必须作为独立预注册变体，并重新验证：

- candidate capacity；
- active ceilings；
- rebuild frequency；
- single rebuild cost；
- amortized cost；
- energy/force parity；
- cutoff crossing；
- total runtime。

只有 profiler 证明 rebuild 已成为剩余主要瓶颈时，才评估复用 OpenMM neighbor infrastructure。

由于该接口是 OpenMM 内部、版本敏感的 API，必须 pin：

```text
OpenMM 8.5.2
commit 36a30cb
CUDA platform
mixed precision
```

并保留当前自建 CSR fallback。

固定 NVT 假设继续保持。NPT 或 cell-grid dimension 变化继续 fail closed，除非另开实验实现安全重分配。

### 20.4.3 A5 验收

```text
candidate set == brute force r<r_list
active set == Reference r<r_cut
zero missed candidates
zero duplicate candidates
rebuild triggers unchanged
rebuild frequency explicitly reported
single-rebuild P50/P95 explicitly reported
amortized rebuild cost explicitly reported
no frozen environment IDs
no silent capacity truncation
```

A5 的目标是降低 rebuild 的摊销成本，而不是宣称“不再扫描全体系”。

---

## 21. 冻结执行顺序

```text
A1 transfer accounting
  -> A2 status/sync consolidation
  -> A3 OpenMM energy-buffer integration
  -> A4 launch/graph control plane
  -> A5 rebuild amortization only if profiling proves necessary
  -> full G2/G3 mixed/nested/fused regression
  -> frozen EXP-026 paired cost harness
```

每一阶段必须：

1. 保留独立 feature switch；
2. 保留上一阶段二进制和 SHA-256；
3. 使用同一真实 73,536 原子 mixed-precision workload；
4. 完成 correctness before performance；
5. 单独记录 wall time 与资源指标；
6. 不根据中间性能结果改变下一阶段阈值。

任何以下情况均 fail closed：

```text
correctness failure
missing metric
NaN/nonfinite
timeout
OOM
hash mismatch
fixture mismatch
precision mismatch
checkpoint mismatch
incomplete paired repeat
```

技术失败不能作为普通性能信号继续叠加下一项。

---

## 22. 最终成本门与结论权限

最终 normative 成本门保持不变：

```text
median(candidate/baseline) <= 1.07
paired-repeat bootstrap one-sided P95 upper <= 1.10
```

冻结 workload：

```text
real 73,536-atom system
CUDA mixed precision
native fused Group1 wiring
same checkpoint
same frame/order
100 warmup steps per arm
1000 measured steps per arm
5 paired repeats
same matched query cadence
paired-repeat bootstrap unit
```

通过该门只能说明：

```text
EXP-026 runtime backend cost is acceptable
on the frozen hardware and workload
```

不能自动说明：

- online ESS/GPU-hour 已改善；
- D4 utility 已通过；
- production promotion 已允许；
- R1 改善等于实际 sampling acceleration。

EXP-025 的：

```text
STOP_EXP025_RUNTIME_BACKEND
```

永久保留。

EXP-026 必须生成：

- 新 preregistration；
- 新源码 hash；
- 新 binary hash；
- 新 correctness report；
- 新 transfer/sync/launch profile；
- 新 paired cost report；
- 新 append-only decision-log 记录。

---

## 23. O3/O4/A2 结果与当前权威状态（2026-08-13 追加）

本节记录本计划第 18/19 节写成之后实际执行的结果：O3 attribution preflight、Patch A1.1 的 O4 normative cost qualification，以及 A2 draft 的正确性验证与成本归因。以下内容如实转录自对应产物文件，不重新推导数字。来源：

- `output/outer_lambda_exp026_cuda_control_plane/o3_attribution/o3_result.md`
- `output/outer_lambda_exp026_cuda_control_plane/o4_preregistration_a1_1.md`
- `output/outer_lambda_exp026_cuda_control_plane/o4_a1_1_result.md`
- `output/outer_lambda_exp026_cuda_control_plane/a2_vs_a11_cost_attribution.md`
- `plugins/LocalManyBodyResidual_exp026_a2_draft/A2_DRAFT_STATUS.md`

### 23.1 O3 attribution：A_recon vs B_a1

比较对象：

- `A_recon`：手工重建的、逻辑等价于 Patch A1 之前控制面的插件，隔离目录 `plugins/LocalManyBodyResidual_exp025_reconstructed/`。身份声明为 `RECONSTRUCTED_LOGICALLY_EQUIVALENT_NOT_FROZEN`——原始 EXP-025 二进制/源码已确认不可恢复（无 git、无备份，仅存一份 G0 阶段桩代码的哈希，与成熟版本不是同一份东西）。
- `B_a1`：当前 live Patch A1 树 `plugins/LocalManyBodyResidual/`（未改动；build 目录已重命名为 `build_exp026_a1/`，`build` 保留兼容符号链接）。

Correctness（A_recon 重跑 B_a1 已通过的全部套件）：G0 smoke/序列化往返、G2 single/mixed、G3 single/mixed/double、`exp026_control_plane_correctness_test`（22 项）、EXP-025 G4 三方等价，全部 PASS，A_recon 与 B_a1 能量数值逐位一致。

结构指纹（真实 nsys 采集，A_recon 与 B_a1 用完全相同脚本/参数各跑一次，1105 次 force evaluation）：

| 指标 | A_recon | B_a1 | 差值 | 差值/evaluation |
|---|---:|---:|---:|---:|
| H2D 次数 | 3,584 | 1,378 | −2,206 | −1.996 |
| H2D 字节 | 475.8 MB | 151.1 MB | −324.7 MB | −68.2% |
| D2H 次数 | 21,020 | 18,810 | −2,210 | −2.000 |
| D2H 字节 | 357.7 MB | 32.5 MB | −325.2 MB | −90.9% |

每 evaluation 精确减少约 2.00 次 H2D 和约 2.00 次 D2H，与 Patch A1 删除的两个 legacy 全体系数组（`uniqueEnvFlagDevice[73536]` 全清零+下载、`neighborCountDevice[41]` 全清零+下载）逐一对应。单次 nsys 采集下 B_a1 具名 GPU kernel 总耗时反而略高于 A_recon（约 101.4 us/eval vs 约 85.9 us/eval）——这是单次、未重复的 profiler 观测，如实记录但不作为判定依据，判定看下面的配对 wall-clock。

配对 wall-clock 计时（5 组，AB/BA 交替，各 100 warmup + 1000 measured steps，无 nsys/ncu 插桩）：

| repeat | 顺序 | A_recon total (s) | B_a1 total (s) | ratio (B_a1/A_recon) |
|---:|---|---:|---:|---:|
| 0 | A_recon_first | 2.8287 | 2.6817 | 0.94803 |
| 1 | B_a1_first | 2.8712 | 2.6600 | 0.92644 |
| 2 | A_recon_first | 2.8498 | 2.6865 | 0.94270 |
| 3 | B_a1_first | 2.8429 | 2.6748 | 0.94085 |
| 4 | A_recon_first | 2.8351 | 2.6910 | 0.94916 |

median(ratio) = **0.94270**；bootstrap one-sided P95 upper（20,000 resamples，seed=20260813）= **0.94916**；5/5 配对方向一致，B_a1 都比 A_recon 快。

结论：`EXP026_A1_ATTRIBUTION = PASS`。

限定（如实转录，不得省略）：

1. 配对 baseline 是逻辑重建（A_recon），不是不可恢复的原始 EXP-025 二进制。
2. 这个 0.9427 是 A_recon vs B_a1（同一份 native fused Group1 candidate，只换插件构建）的比值，**不是** EXP-025 G4 定义的 candidate/baseline（有插件 vs 无插件）比值。不能直接拿这个数字去替换或推翻 EXP-025 冻结的 `median=1.1234, P95=1.1322, STOP_EXP025_RUNTIME_BACKEND`——那个结论原样保留。
3. Patch A1 单独看确实是真实、一致、可重复方向的加速（5/5 同向）。基于历史 postmortem 数字的粗略投影（`1.123398 × 0.94270 ≈ 1.0590`，`1.132172 × 0.94916 ≈ 1.0746`）显示两项都落在 1.07/1.10 门内，但这只是规划估计，不能代替 qualification。真正的成本门必须对 no-plugin baseline 重新运行；A2→A5 仅在该门未通过、或 profiling 证明确有必要时才顺序启用，不预设必须全部做完。

在 O3 之后、A1.1 的 O4 之前，曾有一版临时决定：A2 draft 暂停，理由是上述粗略投影已落在门内，在 no-plugin baseline 的真实 O4 判定结果出来之前，没有理由预设必须做完 A2→A5。这一决定后来被 23.3/23.4 的实际结果取代（O4 对 A1.1 直接通过；A2 随后被重新拾起做正确性验证与成本归因，但仍是可选的更快候选，不是通过门槛的必要条件）。

### 23.2 Patch A1.1：dBdq 有限性检查缺口

在 A2 draft 的机械转换工作中，于 K2 readout（`exp025Readout`）发现一个真实的覆盖缺口：该 kernel 此前只检查 `bReduced`/`sech2Shared` 的有限性，从未检查最终逐 anchor 的 `dBdq[anchor] = sech2Shared * rhoQGrad` 乘积——一个非有限的 `rhoQGrad` 此前可以在 K2 中不被发现地通过，只能在后续 `scatterForce[FromCSR]` 自己的 `coeff` 检查中才可能被捕获。

按最小改动原则，这个修复被独立拆分为 **Patch A1.1**：只加一行 `isfinite(dBdq[anchor])` 检查，复用已有的 `errorFlag` 机制，不合并 status block，不新增同步；未并入 A2 draft，而是直接进入 live 树 `plugins/LocalManyBodyResidual/`，因为改动足够小、可独立验证。

### 23.3 O4 normative cost qualification（A1.1，真实 no-plugin baseline）——EXP-026 的正式成本资格判定

Harness：`scripts/exp025_g4_timing_harness.py`，**未经修改**（与产生 EXP-025 冻结 `STOP_EXP025_RUNTIME_BACKEND` 结果的同一份脚本），只有 `--plugin-build-dir` 指向当前 A1.1 build。所有默认值（5 repeats、100 warmup、1000 measured steps、mixed-capable production window `vdw/window_0`、同一 checkpoint）都是脚本自身的硬性下限，未改动。

Candidate identity（A1.1，运行前冻结）：

- `CudaLocalManyBodyResidualKernels.cpp` sha256：`1a1bcfea500f3b50fad5bba591c2db932fc4a04519c69eda64b1778ba00c51fa`
- `build_exp026_a1/libOpenMMLocalManyBodyResidualCUDA.so` sha256：`3cf9160c8f2b3bec1e987932a0db716698242b28b32fc6953bf29da4cb9af0a3`
- 与 O3 测量的纯 A1 二进制（`14991f44...`）相比，只多了 K2 readout 的 `isfinite(dBdq[anchor])` 一行（Patch A1.1），没有改动其他任何一行。
- 本次 run 前重新验证：G0 smoke、G2 single+mixed、G3 single+mixed+double、22-item control-plane test、EXP-025 G4 三方等价、新增的 8-check dBdq NaN/Inf fault-injection test——全部 PASS，0 failing。

Gate（与 EXP-025 完全一致）：`baseline` = 真实 production window，完全不接插件（`build_baseline_sim()`）；`candidate`（normative）= `native_fused_group1_total` = baseline 的 Group1 换成用 A1.1 的 native residual-enabled IBSBiasForce（`build_native_sim()`）。5 对 paired repeats，AB/BA 顺序，100 warmup + 1000 measured steps/arm，mixed precision，matched-total surrogate（每 500 steps 一次 ledger `getState`），完全遵循冻结的 EXP-025 方法学，未改动。

**THE NORMATIVE GATE — `native_fused_group1_total`：**

```text
median(candidate_total/baseline_total) = 1.04140   (gate: <= 1.07)  -> PASS
bootstrap one-sided P95 upper          = 1.04692   (gate: <= 1.10)  -> PASS

G4_COST_QUALIFICATION = true
```

诊断性基准（不参与判定，仅为完整性报告）：

```text
plugin_standalone_incremental: median=1.05611  p95_upper=1.06547
exact_split_residual_total:    median=1.31229  p95_upper=1.31748
```

决定（按用户 2026-08-13 的规则）：

```text
median <= 1.07 AND P95 upper <= 1.10  ->  STOP_OPTIMIZATION_SUCCESS
```

A1.1 单独就通过了真实 no-plugin baseline 成本门。A2/A3/A4/A5 未启动——不是因为它们失败，而是门槛已经舒适地通过，此时进一步优化的风险被认为不值得。这与之前基于 O3 比值的粗略投影一致（投影 `1.123398 × 0.94270 ≈ 1.059`，`1.132172 × 0.94916 ≈ 1.075`；实测 `1.0414`/`1.0469` 比投影更好）。

明确不代表：

- 不追溯改变 EXP-025 自己的冻结结果——`STOP_EXP025_RUNTIME_BACKEND`（median 1.123398，P95 1.132172）针对的是现已不可恢复的原始 EXP-025 二进制，永久保留、不变；
- 不自动代表 online ESS/GPU-hour 已改善，或 production promotion 已被允许——按第 22 节，通过这道门只允许进入下一个尚未开始的独立实验：paired online utility qualification。

此时 A2 draft 的状态记录：mechanically converted（全部 6 个 kernel + host 端整合已完成），host-side C++ 可编译，但当时从未 run/test 过（无 NVRTC 编译检查，无 correctness suite）。通过这道门并不需要它，按原计划它保持 paused，除非未来出现回归或新需求。

### 23.4 A2 draft 后续：正确性验证 + 与 A1.1 的成本归因（同日晚些时候）

在 O4/A1.1 已经通过之后，用户仍要求把 A2 draft 补完并验证——严格限定在隔离目录 `plugins/LocalManyBodyResidual_exp026_a2_draft/`，不触碰 live、已 qualified 的 A1.1 树 `plugins/LocalManyBodyResidual/`。

A2 自身的正确性验证结果：

```text
A2_BUILD_NVRTC          = PASS
A2_CORRECTNESS          = PASS
A2_THREE_WAY_EQUIVALENCE = PASS
```

覆盖：G0 NVRTC build（真正编译设备端 kernel 源码，不只是 host-side C++ 编译）、G2/G3 full suite、22-item control-plane test、A=B/C=D=E 三方等价、新的 dBdq finiteness fault-injection test——全部 PASS。

A2 vs A1.1 配对成本归因（方法：`scripts/exp026_o3_single_arm_timing.py`，process-per-arm-per-repeat——两份插件构建注册同一个 OpenMM kernel 名字，不能共存于同一进程，与 O3 的 A_recon-vs-B_a1 比较用同一原因；5 对 AB/BA，100 warmup + 1000 measured steps/arm，mixed precision，同一真实 production window/checkpoint，无 nsys/ncu 插桩）：

- A1.1 build：`plugins/LocalManyBodyResidual/build_exp026_a1`（O4-qualified 的二进制）
- A2 build：`plugins/LocalManyBodyResidual_exp026_a2_draft/build`（正确性已验证，本次 run 之前成本未测过）

| repeat | order | A1.1 total (s) | A2 total (s) | ratio (A2/A1.1) |
|---:|---|---:|---:|---:|
| 0 | A1_1_first | 2.6890 | 2.6517 | 0.98614 |
| 1 | A2_first   | 2.6795 | 2.6475 | 0.98808 |
| 2 | A1_1_first | 2.6814 | 2.6545 | 0.98996 |
| 3 | A2_first   | 2.6975 | 2.6573 | 0.98510 |
| 4 | A1_1_first | 2.6938 | 2.6608 | 0.98778 |

```text
median(A2_total/A1_1_total)      = 0.98778   ->  A2 比 A1.1 快约 1.22%
bootstrap one-sided P95 upper    = 0.98996   ->  仍明显 < 1.0
5/5 重复方向一致（A2 每次都更快）
```

Freeze gate（用户 2026-08-13 指令）：

```text
A2 total wall-time improvement >= 0.5%   : 1.222% >= 0.5%  -> PASS
5/5 或 >=4/5 配对方向一致              : 5/5            -> PASS
correctness 完全保持                    : PASS（G0/G2/G3/22-item/dBdq/three-way 全套）
```

三项条件全部 PASS。结论：`A2_COST_ATTRIBUTION = PASS`——A2 有实质收益，可作为比 A1.1 更快的备选 artifact。

**尚未完成、明确记录：** A2 自身的 normative O4（no-plugin baseline vs A2 candidate，使用同一份未改动的 `scripts/exp025_g4_timing_harness.py`，与对 A1.1 做的那次完全一样）尚未运行。这是 A2 在能够替换 A1.1 成为 qualified/shipped 候选之前必须先完成的下一步逻辑步骤。此比较本身是诊断性 wall-clock（无 nsys），与"不从 traced runs 推导 normative ratio"的原则一致。

### 23.5 A3 状态

A3（把能量输出路由到 OpenMM 自己的 CUDA energyBuffer，取代 host-side scalar download，以消除一个同步点，详见第 20.2 节设计）：状态 `NOT_STARTED`，只是刚开始一项隔离调查。其启动条件是 profiling 证明在 A2 之后，能量标量下载仍是剩余同步成本的主导来源（按用户自己设定的条件）。

### 23.6 当前权威状态汇总（2026-08-13）

```text
A1 / A1.1                = QUALIFIED (live tree, O3 PASS, O4 PASS, STOP_OPTIMIZATION_SUCCESS)
A2_BUILD_NVRTC           = PASS
A2_CORRECTNESS           = PASS
A2_THREE_WAY_EQUIVALENCE = PASS
A2_COST_ATTRIBUTION      = PASS (faster than A1.1, own O4 not yet run)
A3                       = NOT_STARTED (isolated investigation just beginning)
```

### 23.7 隔离规则重申

A2 与 A3 的工作只在隔离目录中进行：`plugins/LocalManyBodyResidual_exp026_a2_draft/`，以及若/当 A3 启动时新建的 `plugins/LocalManyBodyResidual_exp026_a3_draft/`。当前 live 树 `plugins/LocalManyBodyResidual/`（目前是 A1.1，已 qualified、已作为通过门槛的候选）在没有明确的另外授权之前，不得被修改。
```

---

## 24. A2 自身 O4 通过、A2 提升为 live 候选、A3 探针结果（2026-08-13 追加）

本节记录第 23 节写成之后进一步完成的工作：A2 自己针对真实 no-plugin baseline 的 normative O4、A2 被正式提升为 shipped/live 候选（**替换 A1.1**）、A3 的 API 可行性探针，以及 A3 的 nosync 上限探针。以下内容如实转录自对应产物文件，不重新推导数字。来源：

- `output/outer_lambda_exp026_cuda_control_plane/o4_a2_result.md`
- `plugins/exp026_a3_energybuffer_probe/A3_PROBE_REPORT.md`
- `output/outer_lambda_exp026_cuda_control_plane/a3_nosync_probe_preregistration.md`
- `output/outer_lambda_exp026_cuda_control_plane/a3_nosync_probe_result.md`
- `output/outer_lambda_exp026_cuda_control_plane/a2_promotion_record.md`
- `output/outer_lambda_exp026_cuda_control_plane/a3_acceptance_criteria.md`

### 24.1 A2 自身的 normative O4（真实 no-plugin baseline）——task 25

Harness 与方法与 23.3 对 A1.1 所用的完全相同：`scripts/exp025_g4_timing_harness.py`，**未经修改**，只有 `--plugin-build-dir` 指向 A2 的 build。预注册见 `o4_preregistration_a2.md`，原始报告见 `o4_a2_timing_report.json`。

Candidate identity（A2）：

- `CudaLocalManyBodyResidualKernels.cpp` sha256：`2b05e73da9d3a412cd6fed4983e550011f479d6201272053e1702a5c11b106f5`
- CUDA `.so` sha256：`14c0f2d20da4cfcdc204d36d944438ca4a574db65e94d2753fed38efe27a822b`
- 正确性已在第 23 节（task 20）验证：G0 NVRTC build、G2/G3 full suite、22-item control-plane test、three-way equivalence、dBdq finiteness fault-injection test，全部 PASS，0 failing。

**THE NORMATIVE GATE — `native_fused_group1_total`：**

```text
median(candidate_total/baseline_total) = 1.03206   (gate: <= 1.07)  -> PASS
bootstrap one-sided P95 upper          = 1.03665   (gate: <= 1.10)  -> PASS

G4_COST_QUALIFICATION = true
A2_NORMATIVE_NO_PLUGIN_COST = PASS
```

`baseline` = 真实 production window（`vdw/window_0`，K=5，prefix=`abfe_dual`），完全不接插件。`candidate` = baseline 的 Group1 换成用 A2 的 native residual-enabled IBSBiasForce。5 对 paired repeats，AB/BA 顺序，100 warmup + 1000 measured steps/arm，mixed precision——与冻结方法学完全一致。

诊断性基准（不参与判定）：

```text
plugin_standalone_incremental: median=1.04224  p95_upper=1.04433
exact_split_residual_total:    median=1.30312  p95_upper=1.30904
```

与 A1.1 自己的 O4（`o4_a1_1_result.md`）对比：

| candidate | median | P95 upper | gate |
|---|---:|---:|---|
| A1.1 | 1.04140 | 1.04692 | PASS |
| A2   | 1.03206 | 1.03665 | PASS |

A2 以更大的余量通过同一道门，方向与 23.4 中已直接测得的 A2-vs-A1.1 诊断性优势（约 1.22%）一致（但数值不必相等：那次比较是两个候选之间的直接配对 wall-clock 比值，不是各自相对 no-plugin baseline 的比值，二者只需方向一致，实际也确实一致）。

状态更新：

```text
A1.1_NORMATIVE_COST        = PASS  (median 1.04140, P95 1.04692)
A2_NORMATIVE_NO_PLUGIN_COST = PASS  (median 1.03206, P95 1.03665, 本节)
```

A2 现在是第二个独立通过真实 no-plugin 成本门的候选，且余量略优于 A1.1。是否把 A2 提升为 shipped/live 候选（替换 A1.1）是一个独立决定——该决定已经做出，见 24.2。

### 24.2 A2 提升为 shipped/live 候选（2026-08-13）——标准状态变更

**这是本节最重要的既定事实：`plugins/LocalManyBodyResidual/`（live、shipped 树）已于 2026-08-13 从 A1.1 提升为 A2。此后任何读者都不应认为 live 树仍是 A1.1。**

触发条件：A2 的三项资格全部独立 PASS，且在每一个已测量的维度上都优于 A1.1：

```text
A2_CORRECTNESS               = PASS
A2_COST_ATTRIBUTION          = PASS  (median 0.98778 vs A1.1, 约快 1.22%)
A2_NORMATIVE_NO_PLUGIN_COST  = PASS  (median 1.03206, P95 1.03665 —— 余量优于 A1.1 的 1.04140/1.04692)
```

用户确认并授权本次提升（2026-08-13）："A2 是不是目前效果最好 可以换了"。

变更内容：在动手前先 diff 了 live 树与 A2 draft，确认唯一有差异的文件就是 A2 已知会改的那两个——其余（openmmapi、序列化、Reference 平台、kernel factory）在 A1.1 与 A2 之间逐字节相同：

```text
plugins/LocalManyBodyResidual/platforms/cuda/src/CudaLocalManyBodyResidualKernels.cpp
plugins/LocalManyBodyResidual/platforms/cuda/src/CudaLocalManyBodyResidualKernels.h
```

提升步骤：

1. 把这两个文件从 `plugins/LocalManyBodyResidual_exp026_a2_draft/` 复制进 live 树；复制后的 `.cpp` sha256（`2b05e73da9d3a412cd6fed4983e550011f479d6201272053e1702a5c11b106f5`）与 `o4_preregistration_a2.md`/`o4_a2_result.md` 中已冻结的候选身份一致。
2. 在新目录 `plugins/LocalManyBodyResidual/build_exp026_a2` 中重新构建（保留 `build_exp026_a1/` 原样不动，其中仍是 A1.1 经 O3/O4 实测过的 `.so`，用于回滚/溯源）。
3. 重新指向 `build` 符号链接：`build_exp026_a1` -> `build_exp026_a2`。
4. 新构建的 `.so` 与 A2 draft 自己的 `.so` 不逐字节相同（`14c0f2d2...` vs 新的 `0bf39964...`）——用 `readelf -d` 验证唯一差异是烘焙进二进制的 `RUNPATH`（各自正确指向自己的 build 目录）。源码 sha256 一致、文件大小一致（177488 字节）加上这个仅限 RUNPATH 的 `readelf` diff，共同确认这是无害差异，不是功能性差异。
5. 对刚提升的 live build 重跑了 G0 smoke test（原生 C++ harness，强制真实 NVRTC JIT 编译+执行）：**PASS**（Reference 与 CUDA 对无接触 fixture 均报 energy=0、force=0，符合预期）。未在这个具体二进制上重跑完整 G2/G3/three-way suite——因为它与已经完整验证过的 A2 draft 源码级逐字节相同，视为多余；G0 是对两个 build 目录间环境/NVRTC 差异最敏感的测试，且已干净通过。

回滚方法：`plugins/LocalManyBodyResidual/build_exp026_a1/` 仍保留 A1.1 的确切 qualified 二进制（sha256 `3cf9160c8f2b3bec1e987932a0db716698242b28b32fc6953bf29da4cb9af0a3`）。回滚步骤：在 `plugins/LocalManyBodyResidual/` 下执行 `rm build && ln -s build_exp026_a1 build`，并从 git 历史或 `build_exp026_a1` 的配对源码恢复那两个 kernel 文件（未被删除）。

提升后的目录状态：

```text
plugins/LocalManyBodyResidual/                          (live, shipped)   = A2  (此前是 A1.1)
plugins/LocalManyBodyResidual_exp026_a2_draft/                            = A2 draft（提升的来源，原样保留）
plugins/LocalManyBodyResidual_exp026_a2_nosync_probe/                     = TIMING_ONLY_INVALID_ENERGY 探针，未受影响
plugins/exp026_a3_energybuffer_probe/                                     = 独立 A3 API 探针，未受影响
```

### 24.3 A3 API 可行性探针：OpenMM CUDA energyBuffer 契约

独立探针，`plugins/exp026_a3_energybuffer_probe/`——与 live `plugins/LocalManyBodyResidual/`（A2）或 A2 draft **不共享任何代码**。最小 `Exp026A3ProbeForce`，三个开关：`returnValueEnergy`（正常 `execute()` 返回值通道）、`bufferEnergy` + `writeToBuffer`（通过一个极简 NVRTC 编译的探针 kernel 直接 `energyBuffer[0] += bufferEnergy`，限定单线程写入以避免任何 buffer-size/GLOBAL_ID 假设）。真实 GPU（RTX 2080 Ti）、真实 `Context::getState(energy)`，3 种配置 × 2 种拓扑 × 3 种 CUDA 精度模式 = 18 次真实测量，全部在设备上进行，没有 mock。

三种配置：(A) 仅 return-only：`returnValueEnergy=111.0`，`writeToBuffer=false`；(B) 仅 buffer-only：`execute()` 返回 `0.0`，`energyBuffer[0]+=222.0`；(C) 两者都有：`execute()` 返回 `77.0` 且 `energyBuffer[0]+=33.0`。两种拓扑：Layer 1 = force 直接加到外层 System；Layer 2 = 同一个 force 作为 collective variable 嵌套在 `CustomCVForce("cv")` 内（这要求补一个最小 serialization proxy，因为 `addCollectiveVariable()` 内部会通过 serialize/deserialize 复制 inner Force 来建立它自己的私有 per-CV Context——这本身就是一个真实发现，见下）。

结果（single/mixed/double precision——三者完全一致）：

| case | Layer 1（直接） | Layer 2（嵌套 CustomCVForce） |
|---|---:|---:|
| (A) return-only | 111.0（精确） | 111.0（精确） |
| (B) buffer-only | 222.0（精确） | 222.0（精确） |
| (C) both (77+33) | 110.0（精确） | 110.0（精确） |

真实且已用 GPU 验证的结论：

1. 直接写入 `energyBuffer[0] += value`（`execute()` 返回 `0.0`）**确实**被 `Context::getState(energy).getPotentialEnergy()` 正确读取（case B/B' 确认）。
2. `energyBuffer` 通道与 `execute()` 返回值通道是**相加**关系，不是互斥关系（case C/C' 精确确认：77+33=110，不是 77、不是 33、也不是 144）。**这是任何真实 A3 实现的关键事实**：如果 readout kernel 改为写入 `cu.getEnergyBuffer()`，`execute()` 必须返回 `0.0`（不能再返回算出来的能量），否则贡献会被重复计数。
3. 嵌套在 `CustomCVForce` collective variable 内（真实 production 实际使用的拓扑）行为完全相同——9 种（精度 × case）组合中 Layer 1 与 Layer 2 没有任何分歧。
4. single/mixed/double CUDA 精度下完全一致——`mixed` typedef 与 `useDoublePrecision()||useMixedPrecision()` 的 host 端参数宽度分发（沿用 A2 draft kernel 文件中已验证的 `addMixedArg`/`addRealArg` 模式）在三种模式下均正确。

次要发现（不是主问题，但是真实的）：`CustomCVForce::addCollectiveVariable()` 要求 inner Force 注册 `SerializationProxy`，即使调用方从未对它做 XML 序列化——因为它内部会通过 serialize/deserialize 复制 inner Force 来建立自己的私有 per-CV Context。探针最初在 Layer 2 失败，报错 `"There is no serialization proxy registered for type ...Exp026A3ProbeForce"`，加了一个最小 proxy 后解决。这与 A3 的真实实现相关：没有新风险，因为 `LocalManyBodyResidualForce` 已经有完整的 serialization proxy（schema_version 2）。

本探针**没有**确定的事项：`energyScratchDevice` 的 host download 在 A2 之后是否仍是主导同步成本（没有做 profiling）——按用户自己设定的门槛，这是启动 A3 真实实现（task 24）前必须先确认的前提，不由本探针建立；力累加路径（本探针只探测了能量，真实插件的 force-scatter 路径与此问题无关，未被触及）；任何成本数字（这只是一个正确性/API 契约探针，未测量任何耗时）。

状态：`A3_ENERGYBUFFER_PROBE = PASS`（task 23）。task 24（真实实现 + 三方验证）仍是 `NOT_STARTED`，其启动前提是有 profiling 证据证明 scalar download 在 A2 之后仍是剩余主导同步成本——按用户自己陈述的决策规则，本探针不建立这个前提。

### 24.4 A3 nosync 上限探针（task 26）

**这一探针测的是什么**：消除当前私有能量 D2H 同步"最多能收回多少 wall time"的一个**上限**——不是对 A3 实际收益的预测。按用户指示（2026-08-13），真实 A3（把能量通过 OpenMM 自己的 `energyBuffer` 写出）仍可能承担本探针未测量的成本：energy-buffer reduction 本身的成本（OpenMM 内部对 buffer 的累积/归约，本探针的桩代码完全绕过了它，因为它压根不写任何地方）；外层 `CustomCVForce`/Group1 融合拓扑自身的能量路径（本探针未触及，探针跑的是同一个 `native_fused_group1_total` workload，但探针本身从不接触 CV/Group1 层）；"完全跳过调用" 与 "路由到另一个 OpenMM 拥有的 buffer" 之间的驱动/流调度差异。所以这个数字是 A3 可能收益的**天花板**，不是下限。

候选：

- `A2` = `plugins/LocalManyBodyResidual_exp026_a2_draft/build`（真实、已验证正确性、已 O4-qualified 的 A2 候选）。
- `NOSYNC` = `plugins/LocalManyBodyResidual_exp026_a2_nosync_probe/build`——**`TIMING_ONLY_INVALID_ENERGY` / `CORRECTNESS_INTENTIONALLY_BROKEN`**——与 A2 完全相同，唯一区别是跳过了那一次阻塞的 `downloadRealScalar(energyScratchDevice, ...)` D2H 调用，`includeEnergy` 为真时 `energy` 硬编码为 `0.0`。**`FORCE/ENERGY RESULTS MUST_NOT_BE_USED`**——这个 build 除了这一次配对 wall-clock 计时之外，不得用于任何其他目的，绝不能被提升、发布或当作正确性候选。

方法：与 O3/A2-vs-A1.1 相同的方法学：`scripts/exp026_o3_single_arm_timing.py`，未修改，process-per-arm-per-repeat，5 对 paired repeats，AB/BA 顺序，100 warmup + 1000 measured steps/arm，mixed precision，同一真实 production window/checkpoint。

预注册的 A3 启动门槛（按用户指示，2026-08-13）：

```text
5/5 或 >=4/5 重复：NOSYNC 比 A2 快
median(NOSYNC_total / A2_total) <= 0.995   （即 >= 0.5% 的潜在 wall-time 收益）
```

原始配对结果：

| repeat | A2 total (s) | NOSYNC total (s) | ratio (NOSYNC/A2) | 更快 |
|---:|---:|---:|---:|---|
| 0 | 2.657174 | 2.657810 | 1.000239 | A2（噪声范围内的平局） |
| 1 | 2.667883 | 2.632541 | 0.986753 | NOSYNC |
| 2 | 2.673611 | 2.653197 | 0.992365 | NOSYNC |
| 3 | 2.667492 | 2.655626 | 0.995552 | NOSYNC |
| 4 | 2.656437 | 2.640149 | 0.993869 | NOSYNC |

```text
median(NOSYNC_total / A2_total) = 0.993869   ->  约 0.61% 潜在 wall-time 天花板
bootstrap one-sided P95 upper   = 1.000239   ->  正好在收支平衡点
4/5 重复 NOSYNC 更快（repeat 0 是噪声范围内的平局，NOSYNC 约慢 0.02%）
```

门槛判定：

```text
gate 1: >=4/5 重复 NOSYNC 更快              -> 4/5              -> PASS
gate 2: median(NOSYNC/A2) <= 0.995          -> 0.993869 <= 0.995 -> PASS

A3_NOSYNC_GATE = PASS（marginal，即勉强通过）
```

**两个预注册条件都通过**——按规则字面意思应该进入 A3 真实实现。但必须如实报告这里的细微之处，而不是只报 pass/fail：这是一次**勉强**的通过。潜在天花板约 0.61%，只是刚超过 0.5% 的门槛，而 bootstrap P95 upper（1.000239）几乎正好落在收支平衡点上——这次测量的不确定性区间在其尾部包含了"根本没有真实收益"的可能性。这只是 A3 可能收益的一个上限（见预注册文件中关于真实 A3 可能拿到更少收益的理由：energy-buffer reduction 成本、外层 CustomCVForce/Group1 路径、驱动/流调度差异，均未被本探针测量）。

决定（按用户既定规则）：

```text
潜在收益 >= 0.5%  ->  继续：实现真实 A3 energy-buffer 候选
                      -> 完整正确性 -> A3 vs A2 配对归因
                      -> A3 自己的 no-plugin normative O4
```

`A3_IMPLEMENTATION` 从 `NOT_STARTED` 变为 authorized-to-start（task 24）。鉴于上面这个勉强的余量，这应被当作"值得尝试"，而不是"保证能赢"——如果真实实现的 A3-vs-A2 配对归因在纳入真实的 energy-buffer reduction/CV-path 成本之后回到 flat 或负值，那是一个合法的 `A3_STOP_LOW_VALUE` 结果，不构成对本门槛的否定。

A2 提升为 shipped 候选（见 24.2）与这道门槛无关，可以独立决定——按用户自己的说法，A2 已经是一个完整、正确、通过成本门的实现，无论 A3 是否继续都不受影响。

### 24.5 A3 canonical 状态标签与验收标准（逐字转录）

以下逐字转录自 `output/outer_lambda_exp026_cuda_control_plane/a3_acceptance_criteria.md`（按用户指示，2026-08-13）。

已确认的 API 契约（来自 `plugins/exp026_a3_energybuffer_probe/A3_PROBE_REPORT.md`）：

```text
OpenMM total energy
= ForceImpl/Kernel execute() returned energy
+ CUDA energyBuffer reduced energy
```

因此任何真实 A3 实现都必须满足：

```text
CUDA readout:
    energyBuffer[GLOBAL_ID] += kBT * B

CUDA execute():
    return 0.0

-- never return U_B at the same time as writing energyBuffer, or the
   contribution is double-counted.
```

探针已验证：single/mixed/double precision；standalone Force；nested inside CustomCVForce；inner Force requires a registered SerializationProxy（CustomCVForce 内部会复制它）。

Canonical 状态标签：

```text
A1.1_NORMATIVE_COST        = PASS  (median 1.04140, P95 1.04692)
A2_CORRECTNESS              = PASS
A2_COST_ATTRIBUTION         = PASS
A2_NORMATIVE_NO_PLUGIN_COST = PASS  (median 1.03206, P95 1.03665 -- see o4_a2_result.md)

A3_API_FEASIBILITY = PASS
A3_NOSYNC_GATE      = PASS (marginal -- median 0.993869, P95 upper 1.000239, 4/5 direction; see a3_nosync_probe_result.md)
A3_IMPLEMENTATION   = AUTHORIZED_TO_START (task 24; not yet started)
A3_CORRECTNESS      = NOT_RUN
A3_COST             = NOT_RUN

SHIPPED CANDIDATE: plugins/LocalManyBodyResidual/ (live) = A2 as of 2026-08-13
  (promoted from A1.1; see a2_promotion_record.md. A1.1 preserved for
  rollback at plugins/LocalManyBodyResidual/build_exp026_a1/.)
```

优先顺序（按用户指示，2026-08-13）：

```text
1. A2 normative O4 vs no-plugin baseline (task 25 -- DONE, PASS, see o4_a2_result.md).
2. Targeted profile of A2 (task 26 -- DONE, marginal PASS, see
   a3_nosync_probe_result.md: ~0.61% potential ceiling, P95 upper right at
   breakeven).
3. Implement A3 for real -- AUTHORIZED (task 24, not yet started). Per the
   user's own framing, this is "worth attempting," not "guaranteed win" --
   a flat/negative A3-vs-A2 result at the real-implementation stage is a
   legitimate A3_STOP_LOW_VALUE outcome, not a contradiction of this gate.
4. EXP-027 (online utility) main line continues regardless -- not blocked by
   any of the above.
```

A3_CORRECTNESS 验收标准（一旦真实实现开始，A3 若要上线必须先满足，目前尚未开始，逐字转录）：

```text
- [ ] Three-way energy AND force equivalence: standalone / nested-in-CustomCVForce
      / fused-Group1-production topologies all agree.
- [ ] energy-only, force-only, and both-requested call semantics all correct.
- [ ] includeEnergy=false calls must NOT write to energyBuffer at all.
- [ ] energyBuffer autoclear correctness verified every evaluation (no stale
      accumulation carried across steps).
- [ ] execute() return value is unconditionally 0.0 whenever energyBuffer is
      written (never both channels nonzero simultaneously).
- [ ] XML serialization / restart round-trip unaffected.
- [ ] No double-counted kBT, no double-counted A_k/offset terms anywhere in
      the new path.
- [ ] GPU energyBuffer index/layout never goes out of bounds (GLOBAL_ID vs.
      buffer size, for whatever launch configuration the real kernel uses --
      the probe only ever wrote index 0, real implementation may need more).
- [ ] Reference (CPU) platform kernel is completely untouched by this CUDA-only
      change -- it keeps using its own return-value channel unconditionally.
```

未开始。记录于此以免在实现开始前遗失。

### 24.6 当前权威状态汇总（2026-08-13，第 24 节追加后）

```text
A1.1                        = QUALIFIED, 已被 A2 取代为 live 候选（见 24.2；O3/O4 数据原样保留于 build_exp026_a1/，供回滚/溯源）
A2_CORRECTNESS               = PASS
A2_COST_ATTRIBUTION          = PASS  (vs A1.1, median 0.98778, 约快 1.22%)
A2_NORMATIVE_NO_PLUGIN_COST  = PASS  (median 1.03206, P95 upper 1.03665; 余量优于 A1.1)

SHIPPED / LIVE CANDIDATE: plugins/LocalManyBodyResidual/ = A2（自 2026-08-13 起；此前是 A1.1）

A3_API_FEASIBILITY = PASS
A3_NOSYNC_GATE      = PASS (marginal -- median 0.993869, P95 upper 1.000239, 4/5 方向一致)
A3_IMPLEMENTATION   = AUTHORIZED_TO_START (task 24；尚未开始)
A3_CORRECTNESS      = NOT_RUN
A3_COST             = NOT_RUN

EXP-025 STOP_EXP025_RUNTIME_BACKEND：原样永久保留，未被本节任何结果撤销或重新解释。
```